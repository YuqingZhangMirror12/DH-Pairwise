"""Random-init Full: fixed E1 6k/12k/24k, common updates and VAL clock.

All formal arms use 144k pair exposures, micro4/effective16, 24k VAL anchors,
global-exposure LR, seed260911. A fixed six-dataset-epoch snapshot is also kept
for the shorter-training diagnostic; it does not add a selection opportunity.
Model selection maximizes precision at empirical clean-VAL95% recall, then AP.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from .train_realism_data_ablation import (
    make_training_dataset, save_json, check_fixed_architecture, evaluate_pair_validation)
from .train_joint_damage import build_random_model
from .train_edge_weathering import _cpu_model_state, capture_rng_state, restore_rng_state, _sha256
from .resampled_input_support import make_ablation_loader
from .recall_operating_points import fit_operating_points
from staging.pairwise_v0_2.models.rachel_model_factory import model_metadata
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import MaterializedWeatheredDataset
from staging.pairwise_v0_2.pairwise_data.rachel_staged_damage_dataset import clean_report
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training.rachel_weathering_training import make_weathering_loader, train_weathering_epoch

SCHEMA, SEED, TOTAL, INTERVAL = "rachel-recall-data-volume/1", 260911, 144000, 24000


class CleanDataset:
    def __init__(self, base):
        self.base, self.split, self.contour_cap = base, "train", 512
    def __len__(self):
        return len(self.base)
    def __getitem__(self, index):
        sample = self.base[index]
        return sample, clean_report(sample, epoch=1)


def volume_plan(size):
    if size not in (6000, 12000, 24000):
        raise ValueError("fixed comparison supports6k/12k/24k unique TRAIN pairs")
    # Every6k slice preserves one deterministic permutation per dataset epoch.
    return [dict(start=s, stop=s + 6000, epoch=s // size + 1,
                 offset=s % size, validate=(s + 6000) % INTERVAL == 0,
                 short_snapshot=s + 6000 == size * 6)
            for s in range(0, TOTAL, 6000)]


def learning_rate(exposure):
    return 1e-4 if exposure < TOTAL // 2 else 2e-5


def run(args):
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("training requires the remote CUDA server")
    if args.resume and args.smoke:
        raise ValueError("smoke and resume are exclusive")
    torch.set_num_threads(1)
    source_path = Path(args.checkpoint).resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    model, architecture, loss_config, initial_digest = build_random_model(source, seed=SEED)
    del source
    if args.train_materialized_manifest:
        training = MaterializedWeatheredDataset(args.train_materialized_manifest)
        manifest = training.manifest_path
        data_record = dict(kind="fixed_e1_materialized", unique_count=len(training), stats=training.stats,
                           protocol=training.protocol)
    else:
        base, data_record = make_training_dataset(args.dataset, args.train_manifest)
        training, manifest = CleanDataset(base), Path(args.train_manifest).resolve()
    size = len(training)
    plan = volume_plan(size)
    identity = dict(schema_version=SCHEMA, seed=SEED, train_manifest=str(manifest),
        train_manifest_sha256=_sha256(manifest), unique_count=size, total_exposures=TOTAL,
        metadata_source_checkpoint_sha256=_sha256(source_path), initial_weights_sha256=initial_digest,
        architecture=architecture, validation_root=str(Path(args.dataset).resolve()),
        selection="cleanVAL precision_at_recall95 then AP; no REAL/TEST fit")
    root = Path(args.output).resolve()
    if args.resume:
        last = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        if last["resume_identity"] != identity:
            raise ValueError("resume experiment identity differs")
    else:
        root.mkdir(parents=True, exist_ok=False)
        last = None
    args.output, args.batch_size, args.effective_batch_size = str(root), 4, 16
    args.status_path = str(root / "status.json")
    args.enable_seam_loss, args.seam_weight = False, 0.
    model = model.to(args.device).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    protocol = dict(**identity, status="running", initialization="random", source_weights_loaded=False,
        training_data=data_record, plan=plan, arguments=vars(args),
        fixed_input_resolution_and_architecture=True, full_network_trainable=True,
        optimizer="AdamW", microbatch=4, effective_batch=16, weight_decay=1e-4,
        learning_rate_schedule={"before_72000_exposures":1e-4, "from_72000_exposures":2e-5},
        validation_anchors=list(range(INTERVAL, TOTAL + 1, INTERVAL)),
        planned_updates=TOTAL // 16, short_training_snapshot_exposure=size * 6,
        short_snapshot_not_used_for_selection=True, test_or_real_used_for_fit=False,
        caveat="fewer unique pairs at equal144k exposures repeat more often; not necessarily less overfit")
    save_json(root / "protocol.json", protocol)
    best_key, best_f1_key, freeze, completed = None, None, None, 0
    if last:
        model.load_state_dict(last["model_state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer_state_dict"])
        best_key, best_f1_key = last["best_key"], last["best_f1_key"]
        best_key = tuple(best_key) if best_key else None
        best_f1_key = tuple(best_f1_key) if best_f1_key else None
        freeze, completed = last["winner_freeze"], last["completed_segments"]
        restore_rng_state(last["rng_state"])
    validation = RachelPairDataset(args.dataset, "val")
    if len(validation) != 3000:
        raise ValueError("requires unchanged cleanVAL3000")
    started, samples, updates = time.monotonic(), completed * 6000, completed * 375

    def payload(epoch, role):
        return dict(model_state_dict=_cpu_model_state(model), **model_metadata(model),
            loss_config=asdict(loss_config), epoch=epoch, seed=SEED, global_exposure=samples,
            optimizer_updates=updates, initialization="random", source_weights_loaded=False,
            training_data=data_record, checkpoint_role=role, initial_weights_sha256=initial_digest,
            seam_loss_enabled=False, resample_contour_cap=None)

    try:
        for number, segment in enumerate(plan, 1):
            if number <= completed:
                continue
            epoch = segment["epoch"]
            order = runner.epoch_indices(size, seed=SEED, epoch=epoch, limit=None)
            count = 128 if args.smoke else 6000
            order = order[segment["offset"]:segment["offset"] + count]
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(segment["start"])
            loader = make_weathering_loader(training, order, batch_size=4, num_workers=args.workers,
                seed=SEED + number, contour_cap=512)
            save_json(root / "status.json", dict(status="running", pid=os.getpid(), phase="train",
                epoch=epoch, global_exposure=samples, optimizer_updates=updates))
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            report = train_weathering_epoch(model, loader, optimizer, loss_config, torch.device(args.device),
                                           args, epoch, args.arm)
            if report["samples"] != count or report["optimizer_updates"] != count // 16:
                raise RuntimeError("training exposure/update count differs")
            samples += count
            updates += report["optimizer_updates"]
            save_json(root / ("segment_%02d.json" % number), dict(segment=segment, training=report))
            if args.smoke:
                result = dict(status="complete", smoke=True, formal_training_counted=False,
                              weights_discarded=True, samples=samples, updates=updates, training=report)
                save_json(root / "smoke.json", result)
                save_json(root / "status.json", result)
                return
            if segment["short_snapshot"]:
                runner._atomic_torch_save(root / "six_epoch_snapshot.pt", payload(epoch, "fixed_six_epoch_diagnostic"))
            if segment["validate"] or segment["short_snapshot"]:
                loader = make_ablation_loader(validation, list(range(len(validation))), batch_size=8,
                    num_workers=args.workers, seed=SEED, contour_cap=512)
                val_report, rows = evaluate_pair_validation(model, loader, torch.device(args.device))
                points = fit_operating_points([r["label"] for r in rows], [r["classification"]["fused"] for r in rows])
                save_json(root / ("validation_%06d.json" % samples), dict(global_exposure=samples,
                    validation=val_report, operating_points=points, selection_eligible=segment["validate"]))
                key, f1_key = tuple(points["selection_key"]), tuple(val_report["selection_key"])
                current = payload(epoch, "exposure_anchor")
                if segment["validate"]:
                    runner._atomic_torch_save(root / ("exposure_%06d.pt" % samples), current)
                if segment["short_snapshot"]:
                    save_json(root / "six_epoch_validation.json", rows)
                    save_json(root / "six_epoch_freeze.json", dict(status="complete", selected_epoch=epoch,
                        selected_global_exposure=samples, checkpoint_sha256=_sha256(root / "six_epoch_snapshot.pt"),
                        validation=val_report, operating_points=points,
                        classifier_thresholds=val_report["thresholds"], primary_pair_threshold=points["thresholds"]["recall_first"],
                        selection_rule="fixed six dataset epochs, not selected by any held-out result",
                        test_or_real_used_for_fit=False))
                if segment["validate"] and val_report["decision_coverage"] == 1. and (best_f1_key is None or f1_key > best_f1_key):
                    best_f1_key = f1_key
                    runner._atomic_torch_save(root / "winner_f1.pt", current)
                    save_json(root / "winner_f1_validation.json", rows)
                if segment["validate"] and val_report["decision_coverage"] == 1. and (best_key is None or key > best_key):
                    best_key = key
                    runner._atomic_torch_save(root / "winner.pt", current)
                    save_json(root / "winner_validation.json", rows)
                    freeze = dict(status="provisional", selected_epoch=epoch, selected_global_exposure=samples,
                        unique_count=size, checkpoint=str(root / "winner.pt"), validation=val_report,
                        operating_points=points, classifier_thresholds=val_report["thresholds"],
                        primary_pair_threshold=points["thresholds"]["recall_first"],
                        selection_rule=identity["selection"], test_or_real_used_for_fit=False)
                    save_json(root / "train_val_freeze.json", freeze)
                print(json.dumps(dict(event="validation_complete", global_exposure=samples, operating_points=points)), flush=True)
            recovery = payload(epoch, "recovery_only")
            recovery.update(optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state(),
                resume_identity=identity, completed_segments=number, best_key=best_key,
                best_f1_key=best_f1_key, winner_freeze=freeze)
            runner._atomic_torch_save(root / "last.pt", recovery)
            completed = number
        if samples != TOTAL or updates != TOTAL // 16 or freeze is None:
            raise RuntimeError("incomplete formal budget/selection")
        freeze.update(status="complete", checkpoint_sha256=_sha256(root / "winner.pt"))
        save_json(root / "train_val_freeze.json", freeze)
        protocol.update(status="complete", completed_exposures=samples, completed_updates=updates,
            selected_epoch=freeze["selected_epoch"], elapsed_s=time.monotonic() - started)
        save_json(root / "protocol.json", protocol)
        save_json(root / "status.json", dict(status="complete", pid=os.getpid(), phase="train_val_complete",
            global_exposure=samples, optimizer_updates=updates))
    except Exception as error:
        save_json(root / "status.json", dict(status="failed", pid=os.getpid(), error=repr(error),
            completed_segments=completed, resumable_exposure=completed * 6000))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--train-manifest")
    group.add_argument("--train-materialized-manifest")
    p.add_argument("--output", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
