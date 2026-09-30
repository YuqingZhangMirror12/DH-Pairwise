"""Bounded, discarded-weight TRAIN-only CUDA batching benchmark.

The caller pauses the formal trainer. This module never controls its process,
changes formal artifacts, selects a model, or reads held-out examples.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from experiments.rachel_n512_formal_30k import train_score_decoupled as trainer
from experiments.rachel_n512_formal_30k.decoupled_samplewise_loss import compute_samplewise_phase_loss
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import MaterializedWeatheredDataset
from staging.pairwise_v0_2.training.rachel_weathering_training import make_weathering_loader, WeatheringStatistics


def batches(dataset, indices, batch):
    return make_weathering_loader(dataset, indices, batch_size=batch, num_workers=4,
                                  seed=trainer.SEED, contour_cap=512)


def loop(model, optimizer, loss_config, dataset, indices, physical, effective, *, update):
    """Include input transfer and the existing per-microbatch training metrics."""
    device = torch.device("cuda:0")
    loader = batches(dataset, indices, physical)
    accumulation = effective // physical
    optimizer.zero_grad(set_to_none=True)
    statistics = WeatheringStatistics()
    count, updates, weighted_loss, valid = 0, 0, 0., 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step, wrapped in enumerate(loader):
        group_start = (step // accumulation) * accumulation * physical
        group_size = min(effective, len(indices) - group_start)
        inputs, targets = trainer.runner._full_batch(wrapped.batch, device)
        output = model(*inputs)
        pose = torch.as_tensor(wrapped.pose_supervision_enabled, dtype=torch.bool, device=device)
        loss_fn = trainer.phase_loss if physical == 1 else compute_samplewise_phase_loss
        total, components = loss_fn(output, targets, pose, loss_config, "matcher")
        n = len(wrapped.batch.pair_ids)
        (total * (n / group_size)).backward()
        valid += int(output.training_valid.sum().item())
        values = torch.stack([total.detach()] + [components[k].detach() for k in trainer.LOSS_NAMES]).cpu().tolist()
        weighted_loss += values[0] * n
        statistics.add(wrapped)
        count += n
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            if update:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            updates += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return dict(samples=count, optimizer_updates=updates if update else 0,
                elapsed_s=elapsed, samples_per_s=count / elapsed,
                mean_loss=weighted_loss / count, valid_pairs=valid)


def gradient_vector(model):
    return torch.cat([p.grad.detach().flatten().cpu() for p in model.parameters() if p.grad is not None])


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(1)
    trainer.runner._set_determinism(trainer.SEED)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("phase") != "matcher":
        raise ValueError("this benchmark is only for the current matcher phase")
    identity = checkpoint["resume_identity"]
    if identity["microbatch"] != 1 or identity["effective_batch"] != 16:
        raise ValueError("expected the original logical micro1/effective16 trajectory")
    dataset = MaterializedWeatheredDataset(args.train_manifest)
    if dataset.split != "train" or len(dataset) != 24000:
        raise ValueError("only the registered full TRAIN is allowed")
    model = trainer.load_decoupled_checkpoint(checkpoint).to("cuda:0")
    optimizer = trainer.create_optimizer(model)
    loss_config = trainer.RachelN512LossConfig(**checkpoint["loss_config"])
    order = list(trainer.runner.epoch_indices(len(dataset), seed=trainer.SEED, epoch=max(1, checkpoint["epoch"]), limit=1024))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    result = dict(schema="decoupled-batching-benchmark/1", formal_training_counted=False,
                  weights_discarded=True, population="fixed TRAIN only", source_checkpoint=str(args.checkpoint),
                  source_completed_segments=checkpoint["completed_segments"], source_exposure=checkpoint["global_exposure"],
                  gpu=torch.cuda.get_device_name(0), total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                  precision="fp32", gradient_checks=[], throughput=[])

    def save():
        (root / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    def reset():
        trainer.restore_training_state(model, optimizer, checkpoint, identity)
        model.set_phase("matcher")
        model.train()
        optimizer.zero_grad(set_to_none=True)

    reference = None
    reference_loss = None
    for physical in (1, 4, 16):
        reset()
        report = loop(model, optimizer, loss_config, dataset, order[:16], physical, 16, update=False)
        vector = gradient_vector(model)
        if reference is None:
            reference, reference_loss = vector, report["mean_loss"]
        relative_l2 = float(torch.linalg.vector_norm(vector - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-12))
        relative_loss = abs(report["mean_loss"] - reference_loss) / max(abs(reference_loss), 1e-12)
        ok = bool(torch.isfinite(vector).all()) and relative_l2 < 1e-3 and relative_loss < 1e-4
        item = dict(physical_microbatch=physical, effective_batch=16, gradient_relative_l2=relative_l2,
                    loss_relative_difference=relative_loss, passed=ok)
        result["gradient_checks"].append(item)
        print(json.dumps(dict(event="gradient_check", **item)), flush=True)
        save()
        if not ok:
            raise RuntimeError("physical batching changed gradients beyond registered FP32 tolerance")
        del vector
    del reference

    for physical in args.batches:
        effective = max(16, physical)
        if effective % physical:
            raise ValueError("physical must divide effective batch")
        reset()
        gc.collect()
        torch.cuda.empty_cache()
        try:
            loop(model, optimizer, loss_config, dataset, order[:effective], physical, effective, update=True)
            torch.cuda.reset_peak_memory_stats()
            measured, offset = [], effective
            while sum(x["elapsed_s"] for x in measured) < args.min_seconds:
                n = math.ceil(64 / effective) * effective
                if offset + n > len(order):
                    offset = effective
                measured.append(loop(model, optimizer, loss_config, dataset, order[offset:offset + n], physical, effective, update=True))
                offset += n
                if sum(x["samples"] for x in measured) >= 512:
                    break
            seconds = sum(x["elapsed_s"] for x in measured)
            samples = sum(x["samples"] for x in measured)
            item = dict(physical_microbatch=physical, effective_batch=effective, status="complete",
                        samples=samples, elapsed_s=seconds, samples_per_s=samples / seconds,
                        optimizer_updates=sum(x["optimizer_updates"] for x in measured),
                        projected_24k_loop_minutes=24000 * seconds / samples / 60,
                        peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved())
        except torch.cuda.OutOfMemoryError as error:
            item = dict(physical_microbatch=physical, effective_batch=effective, status="oom", error=str(error)[:400])
        result["throughput"].append(item)
        print(json.dumps(dict(event="throughput", **item)), flush=True)
        save()
        if item["status"] == "oom":
            break
        if item["peak_reserved_gpu_bytes"] >= result["total_gpu_bytes"] * .80:
            result["stopped_before_larger_batch"] = "80 percent reserved-memory guard"
            break
    result["status"] = "complete"
    save()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16, 20, 36, 54])
    parser.add_argument("--min-seconds", type=float, default=8.)
    run(parser.parse_args())
