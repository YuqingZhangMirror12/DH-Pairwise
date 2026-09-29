"""E1: matched24k + label-consistent edge weathering, clean VAL selection only.

Fixed original 350bf9 warm start, FP32, seed260909, micro4/effective16,
120k incremental pair exposures, 7500 updates, and five clean VAL3000 events.
winner.pt alone is selected by VAL F1/AP. last.pt is an optimizer/RNG recovery
snapshot after each VAL, never an additional model-selection candidate.
--resume restores the latest complete VAL boundary; an interrupted segment is
replayed and its incomplete work is not counted as completed budget.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from experiments.rachel_n512_formal_30k.train_realism_data_ablation import (
    TOTAL_EXPOSURES, VALIDATION_INTERVAL, MICROBATCH, EFFECTIVE_BATCH,
    LEARNING_RATE, SEED, check_fixed_architecture, emit, evaluate_pair_validation,
    exposure_plan, make_training_dataset, save_json,
)
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint, model_metadata
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig


SOURCE_SHA256 = "350bf95413f828698f3396294a317569b013891d0906e2d4212d046fba059233"
TRAIN_COUNT = 24_000
VAL_COUNT = 3_000
SCHEMA = "rachel-edge-weathering-training/1"


def _weathering_interfaces():
    # Keep --help and the small protocol tests independent of optional workers.
    from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import RachelWeatheredDataset
    from staging.pairwise_v0_2.training.rachel_weathering_training import (
        make_weathering_loader, train_weathering_epoch,
    )
    return RachelWeatheredDataset, make_weathering_loader, train_weathering_epoch


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG recovery requires CUDA")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def resume_identity(source_path, manifest_path, original_root, architecture):
    """Only the two TRAIN provenance files and fixed recipe; no dataset scan."""
    source_sha = _sha256(source_path)
    if source_sha != SOURCE_SHA256:
        raise ValueError("E1 requires the original frozen 350bf9 warm-start checkpoint")
    return dict(schema_version=SCHEMA, source_checkpoint=str(source_path),
                source_checkpoint_sha256=source_sha, train_manifest=str(manifest_path),
                train_manifest_sha256=_sha256(manifest_path), dataset_root=str(original_root),
                architecture=architecture, seed=SEED, precision="fp32", train_count=TRAIN_COUNT,
                total_exposures=TOTAL_EXPOSURES, validation_interval=VALIDATION_INTERVAL,
                microbatch=MICROBATCH, effective_batch=EFFECTIVE_BATCH,
                learning_rate=LEARNING_RATE, weight_decay=1e-4,
                requested_endpoint_tiers=dict(clean=.70, mild=.25, moderate=.05))


def validate_resume(last, identity, plan):
    if last.get("resume_identity") != identity:
        raise ValueError("resume source checkpoint, manifest, or fixed protocol differs")
    boundary = last.get("completed_segments")
    if not isinstance(boundary, int) or not 1 <= boundary <= len(plan):
        raise ValueError("resume requires a completed VAL segment boundary")
    segment = plan[boundary - 1]
    if (not segment["validate_after"] or last.get("global_exposure") != segment["global_stop"]
            or last.get("optimizer_updates") != segment["global_stop"] // EFFECTIVE_BATCH
            or last.get("validation_event") != segment["global_stop"] // VALIDATION_INTERVAL
            or last.get("checkpoint_role") != "last_recovery_only"):
        raise ValueError("resume checkpoint is not a complete matched-budget VAL boundary")
    for key in ("optimizer_state_dict", "rng_state", "best_key", "winner_checkpoint",
                "winner_validation_rows", "winner_freeze"):
        if key not in last:
            raise ValueError("resume checkpoint missing " + key)


def _cpu_model_state(model):
    # Independent winner copy: subsequent optimizer steps cannot mutate it.
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def checkpoint_payload(model, loss_config, source, source_path, data_record, protocol,
                       epoch, samples, updates, validation_events):
    return dict(model_state_dict=_cpu_model_state(model), **model_metadata(model),
                loss_config=asdict(loss_config), source_loss_config=source["loss_config"],
                epoch=epoch, global_exposure=samples, optimizer_updates=updates,
                validation_event=validation_events, variant=protocol["arm"], initialization="warm-start",
                initial_checkpoint=str(source_path), initial_epoch=source.get("epoch"),
                seed=SEED, precision="fp32", resample_contour_cap=None,
                seam_loss_enabled=False, seam_loss_weight=0.0, unique_count=data_record["unique_count"],
                training_data=data_record, selection_rule=protocol["selection_rule"],
                weathering_policy=protocol["weathering_policy"])


def run(args, *, recipe=None):
    """Run E1 unchanged, or an explicit data-only recipe in a new source snapshot.

    A recipe supplies build_dataset(base, seed, cache_dir), variant, and a
    JSON-serializable policy. Architecture, losses, budgets and clean VAL stay
    fixed. Its policy is bound into the resume identity before any updates.
    """
    if recipe is not None and (set(recipe) != {"variant", "policy", "build_dataset"}
                               or not callable(recipe["build_dataset"])):
        raise ValueError("data recipe requires variant, policy, and build_dataset")
    variant = "edge_weathering_e1" if recipe is None else recipe["variant"]
    if not isinstance(variant, str) or not variant:
        raise ValueError("data recipe variant must be nonempty")
    if args.workers < 0 or args.log_every <= 0:
        raise ValueError("workers must be nonnegative and log_every positive")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("E1 full-network training must run explicitly on the remote CUDA server")
    source_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.train_manifest).resolve()
    original_root = Path(args.dataset).resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    model = load_rachel_checkpoint(source)
    architecture = check_fixed_architecture(model, source)
    identity = resume_identity(source_path, manifest_path, original_root, architecture)
    if recipe is not None:
        identity = dict(identity, data_recipe_variant=variant,
                        data_recipe_policy=json.loads(json.dumps(recipe["policy"], allow_nan=False)))
    loss_config = RachelN512LossConfig(**source["loss_config"])
    base_training, data_record = make_training_dataset(original_root, manifest_path)
    validation = RachelPairDataset(original_root, "val")
    if len(base_training) != TRAIN_COUNT or data_record["unique_count"] != TRAIN_COUNT or len(validation) != VAL_COUNT:
        raise ValueError("E1 requires unique matched TRAIN24k and unchanged clean VAL3000")
    data_record = dict(data_record, target_policy="E1 source-arc inherited correspondence; changed positive translation loss excluded")
    plan = exposure_plan(TRAIN_COUNT)
    destination = Path(args.output).resolve()
    if args.resume:
        if not destination.is_dir():
            raise ValueError("--resume needs an existing output with last.pt")
        last = torch.load(destination / "last.pt", map_location="cpu", weights_only=False)
        validate_resume(last, identity, plan)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        last = None
    device = torch.device(args.device)
    torch.set_num_threads(1)
    runner._set_determinism(SEED)
    model = model.to(device).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    args.batch_size, args.effective_batch_size = MICROBATCH, EFFECTIVE_BATCH
    args.enable_seam_loss, args.seam_weight = False, 0.0
    args.status_path = str(destination / "status.json")
    args.output = str(destination)
    dataset_type, make_loader, train_epoch = _weathering_interfaces()
    training = (dataset_type(base_training, seed=SEED, epoch=0, cache_dir=args.cache_dir)
                if recipe is None else recipe["build_dataset"](
                    base_training, seed=SEED, cache_dir=args.cache_dir))
    policy = dict(requested_tier_grain="fragment endpoint exposure, not pair", requested_endpoint_tier_probabilities=dict(clean=.70, mild=.25, moderate=.05),
                  actual_tier_counts="separate observed requested/applied endpoint and pair counters in every segment training.weathering",
                  correspondence_policy="inherit original correspondence through source contour arc provenance; do not relabel by nearest geometric fit",
                  translation_loss_policy="retain original GT placement; mask raw-boundary translation auxiliary on changed positives because edge-point deltas include material-loss gap",
                  pair_label_policy="preserve source pair label under label-consistent guarded weathering",
                  validation_weathered=False, target_rules_changed=True,
                  original_gt_translation_preserved=True, zero_gap_auxiliary_on_changed_pairs=False)
    if recipe is not None:
        policy.update(recipe["policy"])
        data_record = dict(data_record, target_policy=policy["correspondence_policy"])
    protocol = dict(schema_version=SCHEMA, status="running", arm=variant, initialization="warm-start",
                    initial_checkpoint=str(source_path), initial_checkpoint_sha256=identity["source_checkpoint_sha256"],
                    initial_epoch=source.get("epoch"), initial_model_metadata=architecture,
                    source_loss_config=source["loss_config"], effective_loss_config=asdict(loss_config),
                    full_network_trainable=True, optimizer="AdamW", learning_rate=LEARNING_RATE,
                    weight_decay=1e-4, gradient_clip_norm=5.0, precision="fp32", seed=SEED,
                    microbatch_size=MICROBATCH, effective_batch_size=EFFECTIVE_BATCH,
                    training_data=data_record, unique_count=data_record["unique_count"], weathering_policy=policy,
                    target_rules_changed=True, planned_global_exposures=TOTAL_EXPOSURES,
                    planned_optimizer_updates=TOTAL_EXPOSURES // EFFECTIVE_BATCH,
                    planned_dataset_epochs=TOTAL_EXPOSURES // TRAIN_COUNT,
                    validation_root=str(original_root), validation_split="val", validation_sample_count=VAL_COUNT,
                    validation_exposure_anchors=list(range(VALIDATION_INTERVAL, TOTAL_EXPOSURES + 1, VALIDATION_INTERVAL)),
                    selection_rule="maximize clean VAL fused equal-row F1; AP breaks ties; earliest exposure breaks exact ties; require full decision coverage",
                    threshold_rule="equal-row clean validation F1; largest threshold among ties",
                    no_pose_composite_selection=True, test_used_for_selection=False, real_used_for_selection=False,
                    exposure_plan=plan, arguments=vars(args), resume_identity=identity,
                    recovery_rule="last.pt after VAL contains model/optimizer/RNG and selected-winner bundle; only complete VAL boundaries are resumable; last is not a selection candidate",
                    budget_caveat="120k incremental exposures exclude shared historical warm-start training and discarded interrupted-segment work",
                    accumulation_caveat="micro4/effective16; auxiliary target-normalized losses need not equal a single batch16 loss")
    save_json(destination / "protocol.json", protocol)
    save_json(destination / "architecture.json", architecture)
    total_samples = total_updates = validation_events = completed_segments = 0
    best_key = best_checkpoint = best_rows = freeze = None
    if last is not None:
        model.load_state_dict(last["model_state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer_state_dict"])
        total_samples, total_updates = last["global_exposure"], last["optimizer_updates"]
        validation_events, completed_segments = last["validation_event"], last["completed_segments"]
        best_key = tuple(last["best_key"]) if last["best_key"] is not None else None
        best_checkpoint, best_rows, freeze = last["winner_checkpoint"], last["winner_validation_rows"], last["winner_freeze"]
        # Restore the atomic boundary's selection bundle if a failed later VAL
        # had already updated winner.pt but had not committed last.pt.
        if best_checkpoint is not None:
            runner._atomic_torch_save(destination / "winner.pt", best_checkpoint)
            save_json(destination / "winner_validation.json", best_rows)
            save_json(destination / "train_val_freeze.json", freeze)
        restore_rng_state(last["rng_state"])
    current_epoch = None
    started = time.perf_counter()
    try:
        for segment_index, segment in enumerate(plan, 1):
            if segment_index <= completed_segments:
                continue
            epoch = current_epoch = segment["epoch"]
            training.set_epoch(epoch)
            epoch_order = runner.epoch_indices(len(training), seed=SEED, epoch=epoch, limit=None)
            indices = epoch_order[segment["epoch_start"]:segment["epoch_stop"]]
            loader = make_loader(training, indices, batch_size=MICROBATCH,
                                 num_workers=args.workers, seed=SEED + segment_index,
                                 contour_cap=model.config.contour_cap)
            save_json(destination / "status.json", dict(status="running", phase="train", epoch=epoch,
                                                       global_exposure=total_samples, optimizer_updates=total_updates,
                                                       segment=segment_index))
            torch.cuda.reset_peak_memory_stats(device)
            train_report = train_epoch(model, loader, optimizer, loss_config, device, args, epoch, variant)
            if train_report["samples"] != segment["sample_count"] or train_report["optimizer_updates"] != segment["optimizer_updates"]:
                raise RuntimeError("observed weathering segment exposures/updates differ from the fixed budget")
            if not isinstance(train_report.get("weathering"), dict):
                raise RuntimeError("weathering trainer must report observed endpoint/pair counters separately")
            total_samples += train_report["samples"]
            total_updates += train_report["optimizer_updates"]
            if total_samples != segment["global_stop"]:
                raise RuntimeError("weathering exposure sequence skipped or duplicated a segment")
            segment_report = dict(segment=segment_index, epoch=epoch, epoch_complete=segment["epoch_complete"],
                                  epoch_slice=[segment["epoch_start"], segment["epoch_stop"]],
                                  global_exposure=total_samples, optimizer_updates=total_updates, training=train_report)
            save_json(destination / ("segment_%02d.json" % segment_index), segment_report)
            emit(dict(event="training_segment_complete", **segment_report))
            if not segment["validate_after"]:
                continue
            save_json(destination / "status.json", dict(status="running", phase="validation", epoch=epoch,
                                                       global_exposure=total_samples, optimizer_updates=total_updates))
            val_loader = make_ablation_loader(validation, tuple(range(len(validation))), batch_size=MICROBATCH,
                                               num_workers=args.workers, seed=SEED, contour_cap=model.config.contour_cap)
            report, rows = evaluate_pair_validation(model, val_loader, device)
            if (report["sample_count"], report["positive_count"], report["negative_count"]) != (VAL_COUNT, VAL_COUNT // 2, VAL_COUNT // 2):
                raise RuntimeError("validation must cover the unchanged balanced clean VAL3000")
            validation_events += 1
            result = dict(validation_event=validation_events, epoch=epoch, global_exposure=total_samples,
                          optimizer_updates=total_updates, validation=report)
            save_json(destination / ("validation_%06d.json" % total_samples), result)
            emit(dict(event="validation_complete", **result))
            key = tuple(report["selection_key"])
            if report["decision_coverage"] == 1.0 and (best_key is None or key > best_key):
                best_key = key
                best_checkpoint = checkpoint_payload(model, loss_config, source, source_path, data_record, protocol,
                                                     epoch, total_samples, total_updates, validation_events)
                best_checkpoint["checkpoint_role"] = "val_selected_winner"
                best_rows = rows
                runner._atomic_torch_save(destination / "winner.pt", best_checkpoint)
                save_json(destination / "winner_validation.json", rows)
                freeze = dict(status="provisional", selected_epoch=epoch, selected_global_exposure=total_samples,
                              selected_optimizer_updates=total_updates, selected_validation_event=validation_events,
                              unique_count=data_record["unique_count"], checkpoint=str(destination / "winner.pt"),
                              validation=report, classifier_thresholds=report["thresholds"], selection_rule=protocol["selection_rule"],
                              initial_checkpoint=str(source_path), seed=SEED, precision="fp32",
                              test_or_real_used_for_fit=False, original_validation_unchanged=True,
                              weathering_policy=policy, last_checkpoint_is_selection_candidate=False)
                save_json(destination / "train_val_freeze.json", freeze)
            recovery = checkpoint_payload(model, loss_config, source, source_path, data_record, protocol,
                                          epoch, total_samples, total_updates, validation_events)
            recovery.update(checkpoint_role="last_recovery_only", completed_segments=segment_index,
                            optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state(),
                            resume_identity=identity, best_key=best_key, winner_checkpoint=best_checkpoint,
                            winner_validation_rows=best_rows, winner_freeze=freeze)
            runner._atomic_torch_save(destination / "last.pt", recovery)
            completed_segments = segment_index
        if (total_samples, total_updates, validation_events) != (TOTAL_EXPOSURES, TOTAL_EXPOSURES // EFFECTIVE_BATCH, TOTAL_EXPOSURES // VALIDATION_INTERVAL):
            raise RuntimeError("completed E1 run did not meet 120k exposures/7500 updates/5 clean VAL events")
        if best_key is None:
            raise RuntimeError("no full-decision-coverage VAL winner; held-out evaluation remains unavailable")
        freeze.update(status="complete", completed_global_exposures=total_samples,
                      completed_optimizer_updates=total_updates, completed_dataset_epochs=TOTAL_EXPOSURES // TRAIN_COUNT,
                      completed_validation_events=validation_events)
        save_json(destination / "train_val_freeze.json", freeze)
        protocol.update(status="complete", completed_global_exposures=total_samples,
                        completed_optimizer_updates=total_updates, completed_dataset_epochs=TOTAL_EXPOSURES // TRAIN_COUNT,
                        completed_validation_events=validation_events, selected_epoch=freeze["selected_epoch"],
                        selected_global_exposure=freeze["selected_global_exposure"], elapsed_seconds_this_process=time.perf_counter() - started)
        save_json(destination / "protocol.json", protocol)
        save_json(destination / "status.json", dict(status="complete", epoch=TOTAL_EXPOSURES // TRAIN_COUNT,
                                                   global_exposure=total_samples, optimizer_updates=total_updates,
                                                   selected_global_exposure=freeze["selected_global_exposure"]))
    except Exception as error:
        # Do not overwrite the exact, resumable last VAL boundary with a model
        # containing uncounted partial updates. Retain explicit failure state.
        failure = dict(status="failed", epoch=current_epoch, global_exposure=total_samples,
                       optimizer_updates=total_updates, completed_validation_events=validation_events,
                       resumable_completed_segments=completed_segments,
                       resumable_global_exposure=(plan[completed_segments - 1]["global_stop"] if completed_segments else 0),
                       partial_segment_budget_counted=False, recovery_checkpoint=str(destination / "last.pt") if completed_segments else None,
                       error=repr(error))
        save_json(destination / "failure_state.json", failure)
        save_json(destination / "status.json", failure)
        protocol.update(status="failed", failure=failure)
        save_json(destination / "protocol.json", protocol)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="original frozen 350bf9 source checkpoint")
    p.add_argument("--dataset", required=True, help="original release; clean VAL3000 only")
    p.add_argument("--train-manifest", required=True, help="matched24k composite TRAIN manifest")
    p.add_argument("--output", required=True, help="new E1 training directory (or existing with --resume)")
    p.add_argument("--cache-dir", help="optional weathered dataset cache")
    p.add_argument("--resume", action="store_true", help="restore only output/last.pt completed VAL boundary")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    return p


def main(argv=None):
    run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
