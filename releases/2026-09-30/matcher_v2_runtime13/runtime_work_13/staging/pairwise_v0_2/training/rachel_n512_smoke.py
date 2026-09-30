"""One-step CUDA smoke on the formal Rachel N=512 training release."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, Optional, Sequence

# Deterministic CUDA GEMM must be configured before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from staging.pairwise_v0_2.models.rachel_n512 import (
    RachelN512Config,
    RachelN512Pairwise,
)
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import (
    RachelPairDataset,
    collate_rachel_pairs,
)
from staging.pairwise_v0_2.training.rachel_n512_loss import (
    RachelN512LossConfig,
    compute_rachel_n512_loss,
)


SCHEMA_VERSION = "rachel-n512-cuda-smoke/1.0"


class RachelN512SmokeError(RuntimeError):
    """The real-release CUDA integration smoke failed closed."""


def _set_determinism(seed: int) -> None:
    if type(seed) is not int or seed < 0:  # noqa: E721
        raise ValueError("seed must be a non-negative integer")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _balanced_samples(dataset: RachelPairDataset, batch_size: int):
    if type(batch_size) is not int or batch_size < 2 or batch_size % 2:  # noqa: E721
        raise ValueError("batch_size must be a positive even integer of at least two")
    per_class = batch_size // 2
    positive = []
    negative = []
    for index in range(len(dataset)):
        sample = dataset[index]
        target = positive if float(sample.label) == 1.0 else negative
        if len(target) < per_class:
            target.append(sample)
        if len(positive) == per_class and len(negative) == per_class:
            break
    if len(positive) != per_class or len(negative) != per_class:
        raise RachelN512SmokeError("train split did not provide a balanced smoke batch")
    # Interleave labels so no implementation can assume class-contiguous rows.
    return tuple(item for pair in zip(positive, negative) for item in pair)


def _tensor(value: np.ndarray, device: torch.device, dtype=None) -> torch.Tensor:
    result = torch.from_numpy(np.ascontiguousarray(value))
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result.to(device=device, non_blocking=False)


def run_smoke(
    *,
    dataset_root: Path,
    device: str = "cuda:0",
    batch_size: int = 2,
    seed: int = 260831,
    precision: str = "fp32",
) -> Dict[str, object]:
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    parsed_device = torch.device(device)
    if parsed_device.type != "cuda" or not torch.cuda.is_available():
        raise RachelN512SmokeError("the Rachel smoke requires an available CUDA device")
    _set_determinism(seed)
    started = time.perf_counter()
    dataset = RachelPairDataset(dataset_root, "train")
    manifest_seconds = time.perf_counter() - started
    load_started = time.perf_counter()
    batch = collate_rachel_pairs(_balanced_samples(dataset, batch_size))
    load_seconds = time.perf_counter() - load_started

    mask_a = _tensor(batch.mask_a, parsed_device, torch.float32)
    mask_b = _tensor(batch.mask_b, parsed_device, torch.float32)
    points_a = _tensor(batch.points_rc_a, parsed_device, torch.float32)
    points_b = _tensor(batch.points_rc_b, parsed_device, torch.float32)
    valid_a = _tensor(batch.contour_valid_a, parsed_device, torch.bool)
    valid_b = _tensor(batch.contour_valid_b, parsed_device, torch.bool)
    labels = _tensor(batch.labels, parsed_device, torch.float32)
    target_a = _tensor(batch.target_a, parsed_device, torch.long)
    target_b = _tensor(batch.target_b, parsed_device, torch.long)
    translation = _tensor(batch.translation_a_to_b_rc, parsed_device, torch.float32)
    translation_valid = _tensor(batch.translation_valid, parsed_device, torch.bool)

    model = RachelN512Pairwise(RachelN512Config()).to(parsed_device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(parsed_device)
    torch.cuda.synchronize(parsed_device)
    step_started = time.perf_counter()
    autocast_enabled = precision == "bf16"
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
    ):
        # Targets are deliberately absent from the model call.
        output = model(mask_a, mask_b, points_a, points_b, valid_a, valid_b)
        loss_config = RachelN512LossConfig()
        loss = compute_rachel_n512_loss(
            output,
            labels,
            target_a,
            target_b,
            translation,
            translation_valid,
            loss_config,
        )
    assignment_affinity_gradient = torch.autograd.grad(
        loss_config.assignment_weight * loss.assignment_nll,
        output.affinity,
        retain_graph=True,
    )[0]
    translation_affinity_gradient = torch.autograd.grad(
        loss_config.translation_weight * loss.translation_smooth_l1,
        output.affinity,
        retain_graph=True,
    )[0]
    assignment_affinity_gradient_norm = float(
        assignment_affinity_gradient.norm().detach().cpu().item()
    )
    translation_affinity_gradient_norm = float(
        translation_affinity_gradient.norm().detach().cpu().item()
    )
    translation_to_assignment_gradient_ratio = translation_affinity_gradient_norm / max(
        assignment_affinity_gradient_norm, 1e-30
    )
    loss.total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=5.0, error_if_nonfinite=True
    )
    optimizer.step()
    torch.cuda.synchronize(parsed_device)
    step_seconds = time.perf_counter() - step_started
    gradient_value = float(gradient_norm.detach().cpu().item())
    if not math.isfinite(gradient_value) or gradient_value <= 0.0:
        raise RachelN512SmokeError("the model did not produce finite nonzero gradients")
    diagnostics = output.transport.diagnostics
    if not output.training_valid.all().item():
        raise RachelN512SmokeError(
            "a smoke sample did not produce finite train evidence"
        )
    if not output.decision_valid.all().item():
        raise RachelN512SmokeError(
            "a smoke transport did not meet decision convergence"
        )
    if not torch.isfinite(output.translation_hat_rc).all().item():
        raise RachelN512SmokeError("translation prediction is non-finite")
    allocated = int(torch.cuda.max_memory_allocated(parsed_device))
    reserved = int(torch.cuda.max_memory_reserved(parsed_device))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "scope": {
            "dataset": "rachel_pairwise_n512_v1_train_only",
            "batch_size": batch_size,
            "positive_count": int((labels == 1.0).sum().item()),
            "negative_count": int((labels == 0.0).sum().item()),
            "contour_cap": int(points_a.shape[1]),
            "cardinal_gate_used": False,
            "rotation_used": False,
            "rgb_used": False,
            "targets_entered_model_forward": False,
        },
        "model": {
            "parameter_count": int(parameter_count),
            "precision": precision,
            "window_sizes_px": [32, 64],
            "patch_size": 16,
            "sinkhorn_iterations": 100,
            "assignment_shape": list(output.assignment.shape),
        },
        "execution": {
            "manifest_init_seconds": manifest_seconds,
            "balanced_batch_load_seconds": load_seconds,
            "forward_backward_step_seconds": step_seconds,
            "loss_total": float(loss.total.detach().cpu().item()),
            "loss_fused_pair": float(loss.fused_pair_bce.detach().cpu().item()),
            "loss_assignment": float(loss.assignment_nll.detach().cpu().item()),
            "loss_translation": float(loss.translation_smooth_l1.detach().cpu().item()),
            "gradient_norm_before_clip": gradient_value,
            "assignment_affinity_gradient_norm": assignment_affinity_gradient_norm,
            "translation_affinity_gradient_norm": translation_affinity_gradient_norm,
            "translation_to_assignment_affinity_gradient_ratio": (
                translation_to_assignment_gradient_ratio
            ),
            "peak_memory_allocated_bytes": allocated,
            "peak_memory_reserved_bytes": reserved,
            "row_residual_max": float(
                diagnostics.row_residual_max.max().detach().cpu().item()
            ),
            "col_residual_max": float(
                diagnostics.col_residual_max.max().detach().cpu().item()
            ),
            "all_sinkhorn_converged": bool(diagnostics.converged.all().item()),
            "supervised_match_count": loss.supervised_match_count,
            "supervised_dustbin_a_count": loss.supervised_dustbin_a_count,
            "supervised_dustbin_b_count": loss.supervised_dustbin_b_count,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260831)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise RachelN512SmokeError("refusing to overwrite an existing smoke receipt")
    result = run_smoke(
        dataset_root=arguments.dataset_root,
        device=arguments.device,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        precision=arguments.precision,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RachelN512SmokeError", "SCHEMA_VERSION", "main", "run_smoke"]
