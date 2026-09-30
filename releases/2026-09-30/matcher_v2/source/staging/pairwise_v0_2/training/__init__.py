"""Training contracts and reproducibility helpers for Pairwise v0.2.

The package initializer is deliberately model-free.  Preflight and cache
qualification import pure ``training.*`` support modules; eagerly importing
the engine/loss stack here would also import torch and the pairwise model before
a no-model receipt could truthfully be emitted.  Historical public attributes
remain available through PEP 562 lazy resolution.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple


_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "CheckpointReceipt": (".checkpoint", "CheckpointReceipt"),
    "canonical_config_hash": (".checkpoint", "canonical_config_hash"),
    "load_trusted_checkpoint": (".checkpoint", "load_trusted_checkpoint"),
    "save_checkpoint": (".checkpoint", "save_checkpoint"),
    "KeypointTensorBatch": (".keypoint_batch", "KeypointTensorBatch"),
    "build_keypoint_tensor_batch": (
        ".keypoint_batch",
        "build_keypoint_tensor_batch",
    ),
    "PairwiseBatch": (".contracts", "PairwiseBatch"),
    "DirectionalStepResult": (".engine", "DirectionalStepResult"),
    "EvalStepResult": (".engine", "EvalStepResult"),
    "TrainStepResult": (".engine", "TrainStepResult"),
    "TransportStepDiagnostics": (".engine", "TransportStepDiagnostics"),
    "eval_directional_step": (".engine", "eval_directional_step"),
    "eval_step": (".engine", "eval_step"),
    "swap_directional_model_inputs": (".engine", "swap_directional_model_inputs"),
    "train_directional_step": (".engine", "train_directional_step"),
    "train_step": (".engine", "train_step"),
    "LossBreakdown": (".losses", "LossBreakdown"),
    "PairwiseLossConfig": (".losses", "PairwiseLossConfig"),
    "compute_pairwise_loss": (".losses", "compute_pairwise_loss"),
    "directional_candidate_loss": (".losses", "directional_candidate_loss"),
    "monotonic_score_loss": (".losses", "monotonic_score_loss"),
    "ragged_directional_swap_consistency_loss": (
        ".losses",
        "ragged_directional_swap_consistency_loss",
    ),
    "sinkhorn_residual_penalty": (".losses", "sinkhorn_residual_penalty"),
    "swap_consistency_loss": (".losses", "swap_consistency_loss"),
    "binary_metrics": (".metrics", "binary_metrics"),
    "CoarseGateArtifact": (".thresholds", "CoarseGateArtifact"),
    "CoarseGateDecision": (".thresholds", "CoarseGateDecision"),
    "apply_high_recall_gate": (".thresholds", "apply_high_recall_gate"),
    "fit_high_recall_gate": (".thresholds", "fit_high_recall_gate"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(__name__, name)
        ) from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "CheckpointReceipt",
    "CoarseGateArtifact",
    "CoarseGateDecision",
    "DirectionalStepResult",
    "EvalStepResult",
    "LossBreakdown",
    "KeypointTensorBatch",
    "PairwiseBatch",
    "PairwiseLossConfig",
    "TrainStepResult",
    "TransportStepDiagnostics",
    "apply_high_recall_gate",
    "binary_metrics",
    "build_keypoint_tensor_batch",
    "canonical_config_hash",
    "compute_pairwise_loss",
    "directional_candidate_loss",
    "eval_directional_step",
    "eval_step",
    "fit_high_recall_gate",
    "load_trusted_checkpoint",
    "monotonic_score_loss",
    "ragged_directional_swap_consistency_loss",
    "save_checkpoint",
    "sinkhorn_residual_penalty",
    "swap_directional_model_inputs",
    "swap_consistency_loss",
    "train_step",
    "train_directional_step",
]
