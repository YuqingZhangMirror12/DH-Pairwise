"""One bounded S8 depth2/cap2048 classifier capacity gate; all updates discarded.

Run only after S6 exits on the same explicitly isolated GPU. Reuse the existing
TRAIN-only extreme-shape selection, without scanning 24K headers again. These
are real TRAIN points padded to cap2048, not 2048 invented valid points. No
capacity search, matcher training, held-out reads, or checkpoint publication.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from experiments.rachel_n512_formal_30k import train_score_decoupled as trainer
from experiments.rachel_n512_formal_30k.benchmark_step_decoupled_batching import (
    SCHEMA as SELECTION_SCHEMA, point_header_counts, sha, save,
)
from experiments.rachel_n512_formal_30k.train_joint_damage import state_digest
from staging.pairwise_v0_2.models.rachel_decoupled_score import DecoupledScoreModel
from staging.pairwise_v0_2.pairwise_data.rachel_step_dataset import StepSourceDataset
from staging.pairwise_v0_2.training.rachel_weathering_training import make_weathering_loader

SCHEMA = "rachel-s8-attention-capacity-gate/1"
CAP, PHYSICAL, EFFECTIVE = 2048, 4, 16


def validate_plan(plan, dataset, manifest_sha):
    """Reuse geometry evidence only, not the old CNN model/capacity results."""
    if (plan.get("schema_version") != SELECTION_SCHEMA or plan.get("status") != "cpu_plan_complete"
            or plan.get("dataset_identity") != dataset.identity or plan.get("manifest_sha256") != manifest_sha
            or dataset.split != "train" or len(dataset) != 24000 or dataset.contour_cap != CAP):
        raise ValueError("requires matching completed TRAIN24000 cap2048 shape selection")
    selection = plan["selection"]
    rows = selection.get("pairs", [])
    if (selection.get("full_TRAIN_scanned_pair_count") != 24000 or len(rows) != 16
            or selection.get("unique_pair_count") != 16
            or len(set(selection.get("indices", []))) != 16
            or selection["indices"] != [row.get("source_index") for row in rows]
            or not any(row.get("label") is True for row in rows)
            or not any(row.get("label") is False for row in rows)
            or selection.get("formal_collation", {}).get("cap") != CAP):
        raise ValueError("incomplete or non-TRAIN-extreme selection")
    for row in rows:
        index = row["source_index"]
        if type(index) is not int or not 0 <= index < 24000:
            raise ValueError("invalid stress source index")
        entry = dataset.entries[index]
        if row["pair_id"] != entry["pair_id"] or row["label"] != bool(entry["label"]):
            raise ValueError("stress selection identity/label differs")
        counts = (row["true_Na"], row["true_Nb"])
        if any(type(n) is not int or not 4 <= n <= CAP for n in counts) or row["true_matrix_cells"] != counts[0] * counts[1]:
            raise ValueError("stress shape invalid")
    return selection


def require_isolated_idle_gpu():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("S8 gate requires one explicit CUDA_VISIBLE_DEVICES GPU")
    output = subprocess.run(["nvidia-smi", "-i", visible, "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    pids = [int(x.strip()) for x in output.stdout.splitlines() if x.strip().isdigit()]
    if any(pid != os.getpid() for pid in pids):
        raise RuntimeError("S8 target GPU has another active process: " + str(pids))
    return visible


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=SCHEMA, experiment_name="S8", status="running",
        formal_training_counted=False, weights_discarded=True, model_checkpoint_written=False,
        head_kind="cross_attention", cross_attention_depth=2, contour_cap=CAP,
        physical_microbatch=PHYSICAL, logical_microbatch=1, effective_batch=EFFECTIVE,
        precision="fp32", AMP=False, held_out_read=False, capacity_search=False)
    device = torch.device("cuda:0")
    try:
        result["physical_gpu_mapping"] = require_isolated_idle_gpu()
        torch.set_num_threads(1)
        trainer.runner._set_determinism(trainer.SEED)
        dataset = StepSourceDataset(args.train_manifest, sampling="step3")
        plan = json.loads(Path(args.selection_file).read_text())
        selection = validate_plan(plan, dataset, sha(args.train_manifest))
        cached = []
        for row in selection["pairs"]:
            entry = dataset.entries[row["source_index"]]
            if point_header_counts(dataset.root / entry["artifact_path"]) != [row["true_Na"], row["true_Nb"]]:
                raise ValueError("actual stress NPZ shape changed since the existing selection")
            cached.append(dataset.weathered(row["source_index"]))
        checkpoint = torch.load(args.matcher_checkpoint, map_location="cpu", weights_only=False)
        previous = trainer.load_decoupled_checkpoint(checkpoint)
        if (checkpoint["completed_segments"] != 48 or checkpoint["phase"] != "matcher"
                or previous.config.contour_cap != CAP
                or checkpoint["resume_identity"]["populations"]["train"]["manifest_sha256"] != sha(args.train_manifest)):
            raise ValueError("S8 gate requires the exact matching S5 step3 M12")
        torch.manual_seed(trainer.HEAD_SEED)
        model = DecoupledScoreModel(previous.base_model, "cross_attention",
                                    model_options={"cross_attention_depth": 2})
        identity = deepcopy(checkpoint["resume_identity"])
        identity.update(head_kind="cross_attention", matrix_head_revision=None, model_options={"cross_attention_depth": 2})
        receipt = trainer.import_matcher(model, checkpoint, identity)
        loss_config = trainer.RachelN512LossConfig(**checkpoint["loss_config"])
        del previous, checkpoint
        model = model.to(device)
        optimizer = trainer.create_optimizer(model)  # New head/base groups; no inherited optimizer tensors.
        trainer.configure_phase(model, 13, identity, receipt, entering_classifier=True)
        initial_base, initial_head = state_digest(model.base_model), state_digest(model.score_head)
        loop_args = SimpleNamespace(microbatch=1, physical_microbatch=PHYSICAL,
            effective_batch=EFFECTIVE, runtime_effective_batch=EFFECTIVE, log_every=1000000, output=str(root))
        result.update(source_matcher_checkpoint=str(args.matcher_checkpoint), matcher_base_sha256=initial_base,
            gpu_name=torch.cuda.get_device_name(device), total_gpu_bytes=torch.cuda.get_device_properties(device).total_memory,
            selection_file=str(args.selection_file), selection=selection,
            geometry_scope="actual full-TRAIN extrema; both sides allocated/padded to2048, without invented valid points",
            source_CNN_capacity_evidence_reused=False, source_optimizer_loaded=False)
        torch.cuda.reset_peak_memory_stats(device)
        reports = []
        for _ in range(2):  # One warmup and one measured effective16 group, all discarded.
            loader = make_weathering_loader(cached, list(range(16)), batch_size=PHYSICAL,
                num_workers=0, seed=trainer.SEED, contour_cap=CAP)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            report = trainer.train_segment(model, loader, optimizer, loss_config, device, loop_args, 13)
            torch.cuda.synchronize(device)
            report["synchronized_elapsed_s"] = time.perf_counter() - started
            reports.append(report)
        trainer.verify_receipt(model, receipt, identity)
        frozen = state_digest(model.base_model) == initial_base
        head_changed = state_digest(model.score_head) != initial_head
        peak_reserved = torch.cuda.max_memory_reserved(device)
        result.update(reports=reports, discard_exposures=32, discard_optimizer_updates=2,
            backward_completed=True, frozen_base_unchanged=frozen, head_updated=head_changed,
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_gpu_bytes=peak_reserved,
            reserved_fraction=peak_reserved / result["total_gpu_bytes"], guard_reserved_fraction=.80)
        if not frozen or not head_changed or result["reserved_fraction"] >= .80:
            raise RuntimeError("S8 frozen-base/head-update/80-percent-memory gate failed")
        result["status"] = "complete"
        save(root / "result.json", result)
        return result
    except BaseException as error:
        result.update(status="oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed", error=repr(error))
        save(root / "result.json", result)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matcher-checkpoint", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--selection-file", required=True)
    p.add_argument("--output", required=True)
    return p


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({k: result[k] for k in ("status", "discard_exposures", "reserved_fraction", "frozen_base_unchanged")}))
