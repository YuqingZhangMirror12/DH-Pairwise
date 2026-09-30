"""S5 TRAIN-only capacity benchmark; default CPU plan, explicit --run-cuda.

Unlike benchmark_decoupled_batching, this reads complete materialized step3
TRAIN targets and collates at cap2048. Current formal collation pads BOTH sides
to2048 for every sample. This is an allocation shape, NOT a claim that every
fragment has2048 real points. We do not trim padding, rescale, change labels,
enable AMP, change a formal batch, or write any model checkpoint.

Every (physical batch, phase) runs in a separate sequential process; all
weights/optimizer updates are discarded. The caller must arrange an idle GPU.
Physical2 is capacity evidence only; its formal registration is reported from
the imported trainer. No automatically selected or deployed production batch
exists. This independent benchmark defaults explicitly to per_pair_norm_v3.

Possible future work, NOT implemented here: crop invalid padding to each
batch's effective Na/Nb after proving the Sinkhorn, labels, dustbin, and cap
contracts remain equivalent. Benchmark-only trimming would misstate capacity.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from experiments.rachel_n512_formal_30k import train_score_decoupled as trainer
from staging.pairwise_v0_2.models.rachel_decoupled_score import build_decoupled_score_model, MATRIX_HEAD_REVISIONS
from staging.pairwise_v0_2.pairwise_data.rachel_step_dataset import StepSourceDataset
from staging.pairwise_v0_2.training.rachel_weathering_training import make_weathering_loader

SCHEMA = "step3-cap2048-discard-batching-benchmark/2"
MODULE = "experiments.rachel_n512_formal_30k.benchmark_step_decoupled_batching"
CAP, EFFECTIVE, PHYSICAL = 2048, 16, (1, 2, 4)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def point_header_counts(path):
    """Read only the two small .npy headers, never masks or source ancestry."""
    counts = []
    with zipfile.ZipFile(path) as archive:
        for side in "ab":
            with archive.open("points_rc_" + side + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported point-array header version")
            if len(shape) != 2 or shape[1] != 2 or not 4 <= shape[0] <= CAP or not np.issubdtype(dtype, np.floating):
                raise ValueError("step3 point-array shape/type differs from the materialized contract")
            counts.append(int(shape[0]))
    return counts


def stress_indices(counts, labels, count=16):
    """Global extrema plus each class's maximum Na*Nb, then largest areas."""
    if len(counts) < count or count < 5:
        raise ValueError("need at least16 genuine TRAIN pairs for the fixed stress pool")
    if any(len(x) != 2 or any(type(n) is not int or not 4 <= n <= CAP for n in x) for x in counts):
        raise ValueError("invalid materialized true point counts")
    if len(labels) != len(counts) or any(label not in (0, 1, False, True) for label in labels):
        raise ValueError("stress selection requires aligned binary TRAIN labels")
    if not any(labels) or all(labels):
        raise ValueError("stress selection requires both positive and negative TRAIN pairs")
    orders = [sorted(range(len(counts)), key=lambda i: (-counts[i][0], i)),
              sorted(range(len(counts)), key=lambda i: (-counts[i][1], i)),
              sorted(range(len(counts)), key=lambda i: (-counts[i][0] * counts[i][1], i))]
    result = []
    class_maxima = [next(i for i in orders[2] if bool(labels[i]) == label) for label in (True, False)]
    for index in [order[0] for order in orders] + class_maxima + orders[2]:
        if index not in result:
            result.append(index)
        if len(result) == count:
            return result


def describe_selection(dataset, counts):
    labels = [entry["label"] for entry in dataset.entries]
    indices = stress_indices(counts, labels)
    selected = [dict(source_index=i, pair_id=dataset.entries[i]["pair_id"],
        label=bool(dataset.entries[i]["label"]), true_Na=counts[i][0], true_Nb=counts[i][1],
        true_matrix_cells=counts[i][0] * counts[i][1]) for i in indices]
    return dict(indices=indices, unique_pair_count=16, pairs=selected,
        rule="global-max Na / global-max Nb / global-max Na*Nb, then max-Na*Nb positive and negative, then descending Na*Nb; ties by source index",
        positive_pair_count=sum(row["label"] for row in selected),
        negative_pair_count=sum(not row["label"] for row in selected),
        class_maximum_true_matrix_cells={name: max(counts[i][0] * counts[i][1] for i in range(len(counts))
            if bool(labels[i]) == label) for name, label in (("positive", True), ("negative", False))},
        full_TRAIN_scanned_pair_count=len(counts), global_max_Na=max(x[0] for x in counts),
        global_max_Nb=max(x[1] for x in counts), global_max_true_matrix_cells=max(x[0]*x[1] for x in counts),
        point_header_read_only=True, model_scores_or_layout_GT_used_for_selection=False,
        train_pair_labels_used_for_stratification=True,
        formal_collation=dict(mode="fixed-cap padding", cap=CAP,
            coordinates_shape_per_side="[physicalB,2048,2]", Sinkhorn_matrix_shape="[physicalB,2048,2048]",
            padding_is_not_real_points=True, benchmark_trimming=False))


def require_idle_gpu():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    pids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    if pids:
        raise RuntimeError("GPU has active compute processes; caller must arrange serial idle execution: " + str(pids))


def require_capacity_checkpoint(checkpoint):
    if checkpoint.get("phase") != "matcher" or checkpoint.get("epoch") != 12 or checkpoint.get("completed_segments") != 48:
        raise ValueError("capacity source must be an exact typed M12 checkpoint")
    optimizer = checkpoint.get("optimizer_state_dict", {})
    groups, states = optimizer.get("param_groups", []), optimizer.get("state", {})
    base_ids = {i for g in groups if g.get("phase_family") == "base" for i in g["params"]}
    if not states or not base_ids.intersection(states):
        raise ValueError("use original trained M12 with Adam state, not an empty-optimizer imported M12 anchor")
    if any(i not in base_ids for i in states):
        raise ValueError("M12 must not contain inherited classifier optimizer state")


def require_worker_plan(args, plan):
    """Reject stale/default-head plans before touching CUDA or model weights."""
    if (plan.get("schema_version") != SCHEMA or
            plan.get("matrix_head_revision") != args.matrix_head_revision or
            args.matrix_head_revision not in MATRIX_HEAD_REVISIONS or
            plan.get("effective_batch") != EFFECTIVE or
            args.worker_physical not in plan.get("physical_candidates", []) or
            args.worker_phase not in plan.get("phases", []) or
            plan.get("measured_groups") != args.measured_groups):
        raise ValueError("worker revision or bounded capacity plan differs")
    selection = plan.get("selection", {})
    rows = selection.get("pairs", [])
    if (selection.get("unique_pair_count") != 16 or len(rows) != 16 or
            selection.get("indices") != [row.get("source_index") for row in rows] or
            len(set(selection["indices"])) != 16 or
            not any(row.get("label") is True for row in rows) or
            not any(row.get("label") is False for row in rows)):
        raise ValueError("worker needs complete class-covered stress16 selection")


def worker_command(args, destination, selection_file, physical, phase):
    return [sys.executable, "-u", "-m", MODULE, "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--train-manifest", str(Path(args.train_manifest).resolve()), "--output", str(destination),
        "--selection-file", str(selection_file), "--worker-phase", phase,
        "--worker-physical", str(physical), "--measured-groups", str(args.measured_groups),
        "--matrix-head-revision", args.matrix_head_revision]


def reuse_cpu_plan(args, dataset):
    """Reuse one owned completed shape plan; workers still check actual16 NPZs."""
    path = Path(args.selection_file).resolve(strict=True)
    plan = json.loads(path.read_text())
    if (plan.get("schema_version") != SCHEMA or plan.get("status") != "cpu_plan_complete" or
            plan.get("dataset_identity") != dataset.identity or
            plan.get("manifest_sha256") != sha(args.train_manifest) or
            plan.get("checkpoint_sha256") != sha(args.checkpoint) or
            plan.get("matrix_head_revision") != args.matrix_head_revision or
            plan.get("physical_candidates") != list(PHYSICAL) or
            plan.get("phases") != ["matcher", "classifier"] or
            plan.get("guard_reserved_fraction") != .80):
        raise ValueError("saved CPU plan differs from the registered TRAIN source/revision/capacity contract")
    validation = SimpleNamespace(matrix_head_revision=args.matrix_head_revision, worker_physical=PHYSICAL[0],
        worker_phase="matcher", measured_groups=args.measured_groups)
    require_worker_plan(validation, plan)
    selection = plan["selection"]
    if (selection.get("full_TRAIN_scanned_pair_count") != len(dataset) or
            selection.get("formal_collation", {}).get("cap") != CAP or
            selection["formal_collation"].get("benchmark_trimming") is not False):
        raise ValueError("saved CPU plan is not a complete untrimmed TRAIN cap2048 selection")
    for row in selection["pairs"]:
        index = row["source_index"]
        if type(index) is not int or not 0 <= index < len(dataset):
            raise ValueError("saved CPU plan source index outside TRAIN")
        entry = dataset.entries[index]
        counts = [row["true_Na"], row["true_Nb"]]
        if (any(type(n) is not int or not 4 <= n <= CAP for n in counts) or
                row["true_matrix_cells"] != counts[0] * counts[1] or
                row["pair_id"] != entry["pair_id"] or row["label"] != bool(entry["label"])):
            raise ValueError("saved CPU plan IDs/labels/point-count bounds differ")
    if (any(type(selection.get(key)) is not int or not 4 <= selection[key] <= CAP
            for key in ("global_max_Na", "global_max_Nb")) or
            type(selection.get("global_max_true_matrix_cells")) is not int or
            not 16 <= selection["global_max_true_matrix_cells"] <= CAP * CAP or
            any(row["true_Na"] > selection["global_max_Na"] or row["true_Nb"] > selection["global_max_Nb"] or
                row["true_matrix_cells"] > selection["global_max_true_matrix_cells"] for row in selection["pairs"])):
        raise ValueError("saved CPU plan global point-count bounds differ")
    return dict(plan, source_CPU_plan=str(path), source_CPU_plan_sha256=sha(path),
        CPU_header_scan_performed=False, original_CPU_header_scan_elapsed_s=plan["CPU_header_scan_elapsed_s"],
        CPU_header_scan_elapsed_s=0., GPU_run_requested=bool(args.run_cuda))


def worker(args):
    """One isolated candidate; copies M12 state only for capacity realism."""
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(args.selection_file).read_text())
    require_worker_plan(args, plan)
    result = dict(schema_version=SCHEMA, physical_microbatch=args.worker_physical,
        matrix_head_revision=args.matrix_head_revision,
        phase=args.worker_phase, logical_microbatch=1, effective_batch=16,
        formal_training_counted=False, weights_discarded=True, precision="fp32", AMP=False,
        auto_deployment=False,
        physical2_formal_registered=2 in getattr(trainer, "FORMAL_PHYSICAL_MICROBATCHES", (1,4,8,16)),
        selection=plan["selection"],
        status="starting", same_padding_as_formal=True)
    device = torch.device("cuda:0")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.set_num_threads(1)
        trainer.runner._set_determinism(trainer.SEED)
        dataset = StepSourceDataset(args.train_manifest, sampling="step3")
        if dataset.split != "train" or len(dataset) != 24000 or dataset.contour_cap != CAP:
            raise ValueError("only complete step3 TRAIN24000 is allowed")
        if (dataset.identity != plan["dataset_identity"] or sha(args.checkpoint) != plan["checkpoint_sha256"] or
                sha(args.train_manifest) != plan["manifest_sha256"]):
            raise ValueError("benchmark source changed after CPU selection")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        require_capacity_checkpoint(checkpoint)
        previous = trainer.load_decoupled_checkpoint(checkpoint)
        config = replace(previous.config, contour_cap=CAP)
        model = build_decoupled_score_model(config, head_kind="matrix_cnn", matrix_head_revision=args.matrix_head_revision)
        if model.matrix_head_revision != plan["matrix_head_revision"]:
            raise ValueError("constructed matrix head revision differs from the benchmark plan")
        model.base_model.load_state_dict(previous.base_model.state_dict(), strict=True)
        source_revision = previous.matrix_head_revision
        del previous
        model = model.to(device)
        optimizer = trainer.create_optimizer(model)
        # Retain actual M Adam tensors to include their memory in both phases.
        # The new matrix head has no inherited C state or score-performance claim.
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss_config = trainer.RachelN512LossConfig(**checkpoint["loss_config"])
        result.update(source_M12_used_for_capacity_only=True, source_cap=checkpoint["model_config"]["contour_cap"],
            no_new_M12_training=True, source_optimizer_states_loaded_for_memory=True,
            source_matrix_head_revision=source_revision,
            new_classifier_random=True, GPU=torch.cuda.get_device_name(device),
            total_gpu_bytes=torch.cuda.get_device_properties(device).total_memory)
        del checkpoint
        model.set_phase(args.worker_phase).train()
        result.update(model_metadata=model.metadata(), benchmark_implementation_sha256=sha(__file__),
            trainer_implementation_sha256=sha(trainer.__file__))
        selected = plan["selection"]["indices"]
        # Load only the16 selected true TRAIN samples, once. This benchmark
        # measures compute/transfer, not repeated NPZ decompression throughput.
        cached = [dataset.weathered(i) for i in selected]
        supervision = []
        for (sample, _), row in zip(cached, plan["selection"]["pairs"]):
            if (sample.pair_id != row["pair_id"] or len(sample.points_rc_a) != row["true_Na"] or
                    len(sample.points_rc_b) != row["true_Nb"] or bool(sample.label) != row["label"]):
                raise ValueError("selected actual sample differs from point-header selection")
            counts = [int(np.count_nonzero((getattr(sample, "target_" + side) >= 0) &
                       getattr(sample, "contour_valid_" + side))) for side in "ab"]
            if sample.label and min(counts) <= 0:
                raise ValueError("positive stress sample lacks effective correspondence supervision")
            if not sample.label and any(counts):
                raise ValueError("negative stress sample has positive correspondence supervision")
            supervision.append(dict(pair_id=sample.pair_id, label=bool(sample.label),
                                    supervised_matches_a=counts[0], supervised_matches_b=counts[1]))
        result["actual_stress_supervision"] = supervision
        loop_args = SimpleNamespace(microbatch=1, effective_batch=16, physical_microbatch=args.worker_physical,
            runtime_effective_batch=16, log_every=1000000, output=str(root))
        epoch = 12 if args.worker_phase == "matcher" else 13
        for group in optimizer.param_groups:
            group["lr"] = trainer.learning_rate(epoch)
        def measure(groups):
            order = list(range(16)) * groups
            loader = make_weathering_loader(cached, order, batch_size=args.worker_physical,
                num_workers=0, seed=trainer.SEED, contour_cap=CAP)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            report = trainer.train_segment(model, loader, optimizer, loss_config, device, loop_args, epoch)
            torch.cuda.synchronize(device)
            report["synchronized_elapsed_s"] = time.perf_counter() - started
            report["samples_per_second"] = report["samples"] / report["synchronized_elapsed_s"]
            return report
        # Start tracking before warmup too: an allocation peak must not be
        # hidden just because a later measured loop reuses memory.
        torch.cuda.reset_peak_memory_stats(device)
        warmup = measure(1)
        warm_alloc = torch.cuda.max_memory_allocated(device)
        warm_reserve = torch.cuda.max_memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        measured = measure(args.measured_groups)
        result.update(status="complete", warmup=warmup, measured=measured,
            peak_allocated_gpu_bytes=max(warm_alloc, torch.cuda.max_memory_allocated(device)),
            peak_reserved_gpu_bytes=max(warm_reserve, torch.cuda.max_memory_reserved(device)),
            measured_groups=args.measured_groups, warmup_groups=1,
            total_discard_pair_exposures=16*(1+args.measured_groups),
            total_discard_optimizer_updates=1+args.measured_groups,
            throughput_scope="repeated same16 longest/stress TRAIN pairs, cached CPU samples; not population-average I/O speed",
            per_epoch_projection_not_reported=True)
        result["reserved_fraction"] = result["peak_reserved_gpu_bytes"] / result["total_gpu_bytes"]
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", error=str(error)[:800],
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(device))
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        save(root / "result.json", result)
        raise
    save(root / "result.json", result)
    # Process termination, not an in-process retry, releases every CUDA object.
    return result


def run(args):
    if args.measured_groups not in (1, 2, 3, 4):
        raise ValueError("bounded measured-groups must be1..4; default2")
    if args.matrix_head_revision not in MATRIX_HEAD_REVISIONS:
        raise ValueError("unregistered benchmark matrix head revision")
    if args.worker_phase:
        if args.worker_physical not in PHYSICAL or not args.selection_file:
            raise ValueError("invalid isolated worker configuration")
        return worker(args)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    dataset = StepSourceDataset(args.train_manifest, sampling="step3")
    if dataset.split != "train" or len(dataset) != 24000:
        raise ValueError("requires complete TRAIN24000; VAL/TEST/REAL/OOD are prohibited")
    if args.selection_file:
        plan = reuse_cpu_plan(args, dataset)
    else:
        started = time.perf_counter()
        counts = [point_header_counts(dataset.root / entry["artifact_path"]) for entry in dataset.entries]
        selection = describe_selection(dataset, counts)
        plan = dict(schema_version=SCHEMA, status="cpu_plan_complete", dataset_identity=dataset.identity,
            matrix_head_revision=args.matrix_head_revision,
            train_manifest=str(Path(args.train_manifest).resolve()), manifest_sha256=sha(args.train_manifest),
            checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha(args.checkpoint),
            selection=selection, CPU_header_scan_elapsed_s=time.perf_counter()-started, CPU_header_scan_performed=True,
            physical_candidates=list(PHYSICAL), effective_batch=16, phases=["matcher", "classifier"],
            measured_groups=args.measured_groups, guard_reserved_fraction=.80,
            weights_discarded=True, formal_training_counted=False, GPU_run_requested=bool(args.run_cuda),
            counts_source="NPZ points_rc_a/b.npy shape headers across complete TRAIN only",
            future_optimization_not_implemented="crop invalid padding only after Sinkhorn/labels/dustbin/cap numerical equivalence proof",
            no_formal_batch_selected_or_changed=True)
        save(root / "train_shape_index.json", dict(dataset_identity=dataset.identity, counts_Na_Nb=counts))
    save(root / "selection.json", plan)
    if not args.run_cuda:
        return plan
    records = []
    status = "complete"
    reason = None
    for physical in PHYSICAL:
        for phase in ("matcher", "classifier"):
            require_idle_gpu()
            destination = root / ("physical%d_%s" % (physical, phase))
            command = worker_command(args, destination, root / "selection.json", physical, phase)
            with (root / (destination.name + ".log")).open("x") as log:
                child = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            path = destination / "result.json"
            row = json.loads(path.read_text()) if path.exists() else dict(status="failed", returncode=child.returncode,
                phase=phase, physical_microbatch=physical, error="worker exited without result")
            records.append(row)
            if child.returncode or row["status"] != "complete":
                status, reason = "stopped", "isolated worker failure/OOM; no larger candidate or formal training started"
            elif row["reserved_fraction"] >= .80:
                status, reason = "stopped", "80 percent reserved-memory guard; no larger candidate attempted"
            save(root / "benchmark.json", {**plan, "status": status, "results": records, "stop_reason": reason,
                "capacity_evidence_only": True, "automatic_deployment": False})
            if reason:
                return dict(status=status, results=records, stop_reason=reason)
    return dict(status=status, results=records, stop_reason=reason)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="original trained typed M12 with Adam state, NOT an imported empty-optimizer anchor; capacity only")
    p.add_argument("--train-manifest", required=True, help="complete S5 step_data/train/manifest.json only")
    p.add_argument("--output", required=True, help="new isolated benchmark directory")
    p.add_argument("--run-cuda", action="store_true", help="explicit sequential discarded-weight GPU test; caller must ensure idle GPU")
    p.add_argument("--measured-groups", type=int, default=2, help="effective16 groups per measured trial; one additional warmup group")
    p.add_argument("--matrix-head-revision", choices=MATRIX_HEAD_REVISIONS, default="per_pair_norm_v3",
                   help="explicit head capacity contract; independent benchmark default is per_pair_norm_v3")
    p.add_argument("--selection-file", help="reuse an existing matching CPU selection.json without rescanning TRAIN headers")
    p.add_argument("--worker-phase", choices=("matcher", "classifier"), help=argparse.SUPPRESS)
    p.add_argument("--worker-physical", type=int, help=argparse.SUPPRESS)
    return p


if __name__ == "__main__":
    outcome = run(parser().parse_args())
    print(json.dumps({k:v for k,v in outcome.items() if k not in ("selection","results")}, ensure_ascii=False))
