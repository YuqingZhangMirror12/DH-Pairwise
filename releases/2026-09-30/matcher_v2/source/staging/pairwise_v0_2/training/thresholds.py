"""Validation-only high-recall rejection policy for the coarse gate."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CoarseGateArtifact:
    threshold: float
    target_recall: float
    achieved_recall: float
    negative_rejection_rate: float
    validation_sample_count: int
    validation_positive_count: int
    validation_negative_count: int
    source_split: str
    checkpoint_id: str
    config_hash: str
    frozen: bool = True
    schema_version: str = "pairwise-coarse-high-recall-gate/0.2"

    def __post_init__(self) -> None:
        for name in (
            "threshold",
            "target_recall",
            "achieved_recall",
            "negative_rejection_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("{} must be finite and in [0, 1]".format(name))
        if self.validation_positive_count <= 0 or self.validation_negative_count <= 0:
            raise ValueError("gate validation requires both classes")
        if self.validation_sample_count != (
            self.validation_positive_count + self.validation_negative_count
        ):
            raise ValueError("gate validation counts are inconsistent")
        if not self.frozen:
            raise ValueError("gate artifact must be frozen")
        if not self.checkpoint_id.strip() or not self.config_hash.strip():
            raise ValueError("checkpoint_id and config_hash must be non-empty")


@dataclass(frozen=True)
class CoarseGateDecision:
    """Only ``reject`` may skip local matching; invalid scores fail open."""

    reject: Tensor
    pass_to_local: Tensor
    requires_review: Tensor
    score_valid: Tensor


def _validation_split(source_split: str) -> str:
    normalized = source_split.strip().casefold()
    if not normalized:
        raise ValueError("source_split must be non-empty")
    allowed = normalized in {"val", "validation", "dev"} or (
        (
            "validation" in normalized
            or normalized.startswith("val_")
            or normalized.endswith("_val")
        )
        and "test" not in normalized
        and "train" not in normalized
    )
    if not allowed:
        raise ValueError("coarse gate threshold must be selected on validation only")
    return source_split


def fit_high_recall_gate(
    probability: Tensor,
    label: Tensor,
    valid: Tensor,
    *,
    checkpoint_id: str,
    config_hash: str,
    source_split: str = "validation",
    target_recall: float = 0.995,
) -> CoarseGateArtifact:
    """Choose the most aggressive observed threshold meeting target recall."""

    _validation_split(source_split)
    if probability.ndim != 1 or not probability.is_floating_point():
        raise TypeError("probability must be a floating-point [N] tensor")
    if label.dtype != torch.bool or tuple(label.shape) != tuple(probability.shape):
        raise TypeError("label must be bool and match probability")
    if valid.dtype != torch.bool or tuple(valid.shape) != tuple(probability.shape):
        raise TypeError("valid must be bool and match probability")
    if not math.isfinite(target_recall) or not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be finite and in (0, 1]")
    usable = valid & torch.isfinite(probability)
    if ((probability[usable] < 0.0) | (probability[usable] > 1.0)).any().item():
        raise ValueError("valid probabilities must remain in [0, 1]")
    labels = label[usable]
    scores = probability[usable]
    positive_scores = scores[labels]
    negative_scores = scores[~labels]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        raise ValueError("gate validation requires valid samples from both classes")
    # At most floor((1-target)*P) positives may be rejected.  Selecting the
    # corresponding observed positive score is conservative under >= ties.
    allowed_misses = int(
        math.floor((1.0 - target_recall) * positive_scores.numel() + 1e-12)
    )
    ordered = torch.sort(positive_scores).values
    threshold = float(ordered[allowed_misses].item())
    retained = positive_scores >= threshold
    rejected_negatives = negative_scores < threshold
    achieved = float(retained.to(torch.float64).mean().item())
    if achieved + 1e-12 < target_recall:
        raise RuntimeError("selected coarse threshold missed its recall target")
    return CoarseGateArtifact(
        threshold=threshold,
        target_recall=float(target_recall),
        achieved_recall=achieved,
        negative_rejection_rate=float(
            rejected_negatives.to(torch.float64).mean().item()
        ),
        validation_sample_count=int(scores.numel()),
        validation_positive_count=int(positive_scores.numel()),
        validation_negative_count=int(negative_scores.numel()),
        source_split=source_split,
        checkpoint_id=checkpoint_id,
        config_hash=config_hash,
    )


def apply_high_recall_gate(
    probability: Tensor,
    valid: Tensor,
    artifact: CoarseGateArtifact,
    *,
    checkpoint_id: str,
    config_hash: str,
) -> CoarseGateDecision:
    """Apply a matching frozen artifact; invalid values pass onward/review."""

    if not isinstance(artifact, CoarseGateArtifact) or not artifact.frozen:
        raise TypeError("artifact must be a frozen CoarseGateArtifact")
    if artifact.checkpoint_id != checkpoint_id or artifact.config_hash != config_hash:
        raise ValueError("coarse gate artifact does not match model checkpoint/config")
    if probability.ndim != 1 or not probability.is_floating_point():
        raise TypeError("probability must be a floating-point [N] tensor")
    if valid.dtype != torch.bool or tuple(valid.shape) != tuple(probability.shape):
        raise TypeError("valid must be bool and match probability")
    score_valid = (
        valid
        & torch.isfinite(probability)
        & (probability >= 0.0)
        & (probability <= 1.0)
    )
    reject = score_valid & (probability < artifact.threshold)
    # Fail open for matching recall, but mark review so bad inputs cannot be
    # silently emitted as ordinary predictions.
    return CoarseGateDecision(
        reject=reject,
        pass_to_local=~reject,
        requires_review=~score_valid,
        score_valid=score_valid,
    )


__all__ = [
    "CoarseGateArtifact",
    "CoarseGateDecision",
    "apply_high_recall_gate",
    "fit_high_recall_gate",
]
