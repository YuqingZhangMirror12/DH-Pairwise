"""CPU-only companion to the unchanged fixed matcher SIMVAL evaluator.

Default: read-only preflight for one M12 checkpoint and 24 balanced timing pairs.
--execute is required for any inference. No GPU lock, CUDA model, optimization,
REAL/OOD access, threshold fitting, or best-checkpoint selection is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence import evaluate_matcher_simval as original

SCHEMA = "fixed-matcher-simval-convergence-cpu/1"


def selected_indices(manifest_rows, count):
    original.validate_manifest(manifest_rows)
    if count not in (24, 32, 3000):
        raise ValueError("only 24/32 balanced timing pairs or full 3000 are permitted")
    if count == 3000:
        return tuple(range(3000))
    # Systematically span each label's full manifest order; never select by score.
    half = count // 2
    indices = []
    for label in (0, 1):
        members = [i for i, row in enumerate(manifest_rows) if row["label"] == label]
        indices.extend(members[j * (len(members) - 1) // (half - 1)] for j in range(half))
    return tuple(sorted(indices))


def preflight(args):
    if args.cpu_threads not in (1, 2, 4) or args.workers != 0:
        raise ValueError("CPU companion allows 1/2/4 threads and workers=0 only")
    plan = original.preflight(args)
    entries = [entry for entry in plan["checkpoint_entries"]
               if args.epoch == "all" or entry["epoch"] == int(args.epoch)]
    manifest_rows = [json.loads(line) for line in Path(plan["simval_manifest"]).read_text().splitlines() if line]
    indices = selected_indices(manifest_rows, args.pairs)
    ids = [manifest_rows[i]["pair_id"] for i in indices]
    plan.update(schema_version=SCHEMA, checkpoint_entries=entries, device="cpu",
        purpose="full_fixed_simval" if args.pairs == 3000 else "timing_subset_not_validation_curve",
        source_full_sample_count=3000, sample_count=len(ids), positive_count=len(ids) // 2,
        negative_count=len(ids) // 2, indices=list(indices), pair_ids=ids,
        pair_order_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        gpu_lock_used=False, cuda_visible_devices="", cpu_threads=args.cpu_threads,
        cpu_interop_threads=1, workers=0,
        device_comparability="same data/loss/decoder definitions; CPU FP32 is not bitwise CUDA-equivalent; do not interpret tiny cross-device differences as convergence")
    plan.pop("gpu_lock")
    return plan


def configure_cpu(threads):
    # Called BEFORE importing model/loader code. This process cannot see a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    import numpy as np
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("refuse CPU companion in a process with initialized CUDA")
    torch.set_num_threads(threads)
    # PyTorch permits setting interop threads only once, even to the same value.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    random.seed(260913)
    np.random.seed(260913)
    # Seed only CPU's default generator, avoiding CUDA seed callbacks.
    torch.random.default_generator.manual_seed(260913)
    torch.use_deterministic_algorithms(True)
    return torch


def timing_summary(samples):
    # No extra examples are inferred just for warmup. Exclude first two measured
    # pairs from the rough steady estimate but retain their metrics and timings.
    if len(samples) < 3:
        raise ValueError("at least three measured pairs required for timing")
    steady = [r["data_seconds"] + r["evaluate_seconds"] for r in samples[2:]]
    median = statistics.median(steady)
    return dict(measured_pairs=len(samples), warmup_pairs_excluded_from_estimate=2,
        loop_seconds=sum(r["data_seconds"] + r["evaluate_seconds"] for r in samples),
        mean_data_seconds=statistics.mean(r["data_seconds"] for r in samples),
        mean_evaluate_seconds=statistics.mean(r["evaluate_seconds"] for r in samples),
        steady_pair_seconds=dict(mean=statistics.mean(steady), median=median,
                                 min=min(steady), max=max(steady)),
        rough_3000_pair_loop_seconds=3000 * median,
        rough_six_checkpoint_loop_seconds=18000 * median,
        estimate_caveat="small systematic subset, not a runtime guarantee; excludes checkpoint/hash/setup cost; CPU/RAM/I/O contention with training can change throughput")


def execute(args, plan):
    torch = configure_cpu(args.cpu_threads)
    from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
    from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
    from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
    from experiments.rachel_n512_formal_30k.train_score_decoupled import load_decoupled_checkpoint, matcher_contract
    from experiments.rachel_n512_formal_30k.train_joint_damage import state_digest
    from experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint import TOP2_CONFIG
    from experiments.rachel_n512_formal_30k import decoupled_samplewise_loss
    device = torch.device("cpu")
    dataset = RachelPairDataset(args.dataset, "val")
    if len(dataset) != 3000 or dataset.split != "val":
        raise ValueError("reader must be fixed SIMVAL3000")
    destination = Path(plan["output"])
    destination.mkdir(parents=True, exist_ok=False)
    protocol = dict(plan, status="running", source_sha256=original.sha256(__file__),
        original_evaluator_sha256=original.sha256(original.__file__),
        loss_source_sha256=original.sha256(decoupled_samplewise_loss.__file__),
        decoder="full_top2_mode", decoder_config=asdict(TOP2_CONFIG),
        batch_size=1, precision="fp32", model_mode="eval+inference_mode; frozen base matcher only",
        training_performed=False, checkpoint_selection_performed=False, thresholds_fitted=False,
        real_or_ood_read=False, classifier_metrics_reported=False,
        loss_semantics="same registered logical-micro1 M loss and conditional denominators as original evaluator",
        runtime=dict(torch=str(torch.__version__), platform=platform.platform(), device="cpu",
                     cuda_initialized=torch.cuda.is_initialized(), threads=torch.get_num_threads(),
                     interop_threads=torch.get_num_interop_threads()),
        started_at_unix=time.time(), evaluations=[])
    original.save(destination / "protocol.json", protocol)
    contracts = {}
    try:
        for entry in plan["checkpoint_entries"]:
            checkpoint = Path(entry["checkpoint"])
            digest = original.sha256(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            identity = original.validate_payload_metadata(payload, entry["epoch"])
            contract = matcher_contract(identity)
            if entry["arm"] in contracts and contracts[entry["arm"]] != contract:
                raise ValueError("matcher contract changes within arm")
            contracts[entry["arm"]] = contract
            wrapper = load_decoupled_checkpoint(payload)
            config = replace(RachelN512LossConfig(**payload["loss_config"]), validate_runtime_targets=True)
            model = wrapper.base_model.eval().requires_grad_(False).to(device)
            if any(p.device.type != "cpu" for p in model.parameters()):
                raise RuntimeError("model escaped CPU")
            before = state_digest(model)
            del payload, wrapper
            output = destination / entry["arm"] / ("m%02d" % entry["epoch"])
            output.mkdir(parents=True, exist_ok=False)
            loader = make_ablation_loader(dataset, tuple(plan["indices"]), batch_size=1,
                num_workers=0, seed=260913, contour_cap=512)
            iterator = iter(loader)
            rows, samples, started = [], [], time.perf_counter()
            with (output / "pair_metrics.jsonl").open("x") as stream:
                for _ in plan["indices"]:
                    t0 = time.perf_counter()
                    batch = next(iterator)
                    t1 = time.perf_counter()
                    # Exact original CPU-compatible forward/loss/raw decoder path.
                    current = original.evaluate_batch(model, batch, config, device)
                    t2 = time.perf_counter()
                    if len(current) != 1:
                        raise RuntimeError("batch1 contract violated")
                    samples.append(dict(pair_id=current[0]["pair_id"], data_seconds=t1 - t0,
                                        evaluate_seconds=t2 - t1))
                    stream.write(json.dumps(current[0], allow_nan=False) + "\n")
                    rows.extend(current)
            if [r["pair_id"] for r in rows] != plan["pair_ids"]:
                raise ValueError("reader changed selected ordered population")
            if state_digest(model) != before or original.sha256(checkpoint) != digest:
                raise RuntimeError("model/source changed during inference")
            if torch.cuda.is_initialized():
                raise RuntimeError("unexpected CUDA initialization")
            summary = dict(schema_version=SCHEMA, status="complete", **entry,
                purpose=plan["purpose"], checkpoint_sha256=digest, matcher_state_sha256=before,
                frozen_matcher_unchanged=True, fixed_simval_sha256=original.VAL_HASH,
                sample_count=len(rows), source_base_loss_config=asdict(config),
                effective_matcher_objective_weights=dict(assignment_nll=.5,
                    translation_smooth_l1=.5, sinkhorn_residual=.05, all_pair_bce=0.),
                metrics=original.summarize(rows), timing=timing_summary(samples),
                elapsed_loop_seconds=time.perf_counter() - started,
                pair_metrics_sha256=original.sha256(output / "pair_metrics.jsonl"))
            original.save(output / "timing_pairs.json", samples)
            original.save(output / "summary.json", summary)
            protocol["evaluations"].append(dict(arm=entry["arm"], epoch=entry["epoch"],
                summary=str(output / "summary.json"), summary_sha256=original.sha256(output / "summary.json")))
            original.save(destination / "protocol.json", protocol)
            print(json.dumps(dict(arm=entry["arm"], epoch=entry["epoch"], pairs=len(rows),
                                  timing=summary["timing"])), flush=True)
            del model
        if original.sha256(plan["simval_manifest"]) != original.VAL_HASH:
            raise ValueError("SIMVAL manifest changed")
        protocol.update(status="complete", completed_evaluations=len(protocol["evaluations"]))
    except BaseException as error:
        protocol.update(status="failed", error=repr(error))
        raise
    finally:
        protocol.update(finished_at_unix=time.time(), elapsed_seconds=time.time() - protocol["started_at_unix"])
        original.save(destination / "protocol.json", protocol)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/dataset_rachel_pairwise_n512_v1"))
    p.add_argument("--arm", choices=("all",) + tuple(original.ARMS), default="shared_s3_s4_s6")
    p.add_argument("--epoch", choices=("all", "8", "10", "12"), default="12")
    p.add_argument("--pairs", type=int, choices=(24, 32, 3000), default=24)
    p.add_argument("--cpu-threads", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--batch-size", type=int, choices=(1,), default=1)
    p.add_argument("--workers", type=int, choices=(0,), default=0)
    p.add_argument("--output", type=Path, default=original.DIAGNOSIS / "matcher_convergence/cpu_timing24_v1")
    p.add_argument("--execute", action="store_true")
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    planned = preflight(arguments)
    if arguments.execute:
        execute(arguments, planned)
    else:
        print(json.dumps({k: v for k, v in planned.items() if k not in ("pair_ids", "indices")}, indent=2))
