"""Losses for pair labels, local assignment and robustness constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.pairwise import (
    DirectionalPairwiseOutput,
    HierarchicalDirectionalOutput,
    PairwiseOutput,
)


@dataclass(frozen=True)
class PairwiseLossConfig:
    pair_bce_weight: float = 1.0
    coarse_bce_weight: float = 0.25
    local_bce_weight: float = 0.5
    correspondence_weight: float = 0.0
    swap_weight: float = 0.1
    swap_transport_weight: float = 0.1
    monotonic_weight: float = 0.0
    monotonic_margin: float = 0.0
    sinkhorn_residual_weight: float = 0.05
    sinkhorn_residual_target: float = 1e-3

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError("{} must be non-negative".format(name))


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    pair_bce: Tensor
    coarse_bce: Tensor
    local_bce: Tensor
    correspondence: Tensor
    swap: Tensor
    monotonic: Tensor
    sinkhorn_residual: Tensor
    valid_pair_count: int
    valid_local_count: int


def _masked_bce(logits: Tensor, label: Tensor, valid: Tensor) -> Tensor:
    target = label.to(dtype=logits.dtype)
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if valid.any().item():
        return losses[valid].mean()
    return logits.sum() * 0.0


def monotonic_score_loss(
    clean_probability: Tensor,
    degraded_probability: Tensor,
    positive_mask: Tensor,
    margin: float = 0.0,
) -> Tensor:
    """Penalize degradation increasing a positive pair's confidence."""

    if tuple(clean_probability.shape) != tuple(degraded_probability.shape):
        raise ValueError("clean/degraded probability shapes differ")
    if positive_mask.dtype != torch.bool or tuple(positive_mask.shape) != tuple(
        clean_probability.shape
    ):
        raise TypeError("positive_mask must be bool and match probability shape")
    values = torch.relu(degraded_probability - clean_probability + float(margin))
    if positive_mask.any().item():
        return values[positive_mask].mean()
    return clean_probability.sum() * 0.0


def swap_consistency_loss(
    forward: PairwiseOutput,
    swapped: PairwiseOutput,
    *,
    transport_weight: float = 0.1,
    use_training_validity: bool = False,
) -> Tensor:
    """A/B score consistency plus optional transport-transpose consistency."""

    if tuple(forward.fused_probability.shape) != tuple(swapped.fused_probability.shape):
        raise ValueError("forward/swapped batch shapes differ")
    valid = (
        forward.training_valid & swapped.training_valid
        if use_training_validity
        else forward.decision_valid & swapped.decision_valid
    )
    score = (forward.fused_probability - swapped.fused_probability).square()
    result = score[valid].mean() if valid.any().item() else score.sum() * 0.0
    if transport_weight > 0.0:
        left = forward.local.assignment
        right = swapped.local.assignment.transpose(1, 2)
        if tuple(left.shape) != tuple(right.shape):
            raise ValueError("swapped assignment is not the transpose shape")
        transport = (left - right).square().flatten(1).mean(dim=1)
        if valid.any().item():
            result = result + float(transport_weight) * transport[valid].mean()
    return result


def ragged_directional_swap_consistency_loss(
    forward: HierarchicalDirectionalOutput,
    swapped: HierarchicalDirectionalOutput,
    *,
    transport_weight: float = 0.1,
) -> Tensor:
    """Runnable A/B-swap ablation for flattened four-direction candidates.

    Candidate order and sample ownership must be preserved by the input swap;
    only the direction slots change under the frozen ``left<->right`` and
    ``above<->below`` permutation.  Training-valid scores are compared, while
    production decision validity remains independently fail-closed.
    """

    if transport_weight < 0.0 or not math.isfinite(float(transport_weight)):
        raise ValueError("transport_weight must be finite and non-negative")
    if not torch.equal(forward.sample_index, swapped.sample_index):
        raise ValueError("swapped candidates changed sample ownership or order")
    inverse = torch.tensor(
        [1, 0, 3, 2], dtype=torch.long, device=forward.direction_index.device
    )
    expected_direction = inverse.index_select(0, forward.direction_index)
    if not torch.equal(expected_direction, swapped.direction_index):
        raise ValueError("swapped candidate directions do not use the inverse slots")

    left_direction = forward.training_direction_output
    right_direction = swapped.training_direction_output
    pair_valid = left_direction.pair_valid & right_direction.pair_valid
    pair_error = (
        left_direction.pair_probability - right_direction.pair_probability
    ).square()
    result = (
        pair_error[pair_valid].mean()
        if pair_valid.any().item()
        else pair_error.sum() * 0.0
    )

    aligned_right_probability = right_direction.candidate_probabilities.index_select(
        1, inverse
    )
    aligned_right_valid = right_direction.candidate_valid.index_select(1, inverse)
    direction_valid = left_direction.candidate_valid & aligned_right_valid
    direction_error = (
        left_direction.candidate_probabilities - aligned_right_probability
    ).square()
    if direction_valid.any().item():
        result = result + direction_error[direction_valid].mean()

    left_arc = forward.arc_pairwise_output
    right_arc = swapped.arc_pairwise_output
    if (left_arc is None) != (right_arc is None):
        raise ValueError("forward/swapped local candidate evidence differs")
    if left_arc is not None and right_arc is not None:
        arc_valid = left_arc.training_valid & right_arc.training_valid
        arc_error = (
            left_arc.fused_probability - right_arc.fused_probability
        ).square()
        if arc_valid.any().item():
            result = result + arc_error[arc_valid].mean()
        if transport_weight > 0.0:
            left_transport = left_arc.local.assignment
            right_transport = right_arc.local.assignment.transpose(1, 2)
            if tuple(left_transport.shape) != tuple(right_transport.shape):
                raise ValueError("swapped ragged transport is not the transpose shape")
            transport_error = (left_transport - right_transport).square().flatten(1)
            transport_error = transport_error.mean(dim=1)
            if arc_valid.any().item():
                result = result + float(transport_weight) * transport_error[
                    arc_valid
                ].mean()
    return result


def _correspondence_loss(
    output: PairwiseOutput,
    target: Optional[Tensor],
    mask: Optional[Tensor],
    valid_local: Tensor,
) -> Tensor:
    if target is None or mask is None:
        return output.local.assignment.sum() * 0.0
    if tuple(target.shape) != tuple(output.local.assignment.shape):
        raise ValueError("correspondence target shape differs from assignment")
    if ((target[mask] < 0.0) | (target[mask] > 1.0)).any().item():
        raise ValueError("correspondence target must remain in [0, 1]")
    candidate = output.local.assignment.clamp(1e-6, 1.0 - 1e-6)
    loss = F.binary_cross_entropy(
        candidate, target.to(candidate.dtype), reduction="none"
    )
    valid_mask = mask & valid_local[:, None, None]
    return loss[valid_mask].mean() if valid_mask.any().item() else candidate.sum() * 0.0


def sinkhorn_residual_penalty(
    output: PairwiseOutput,
    *,
    target: float = 1e-3,
) -> Tensor:
    """Penalize finite Sinkhorn marginal error without relaxing decisions.

    A non-converged but finite plan remains differentiable under the default
    training policy.  This term gives that plan an explicit route toward the
    frozen production tolerance instead of silently removing all local-model
    gradients.  Dual-softmax ablations return an exact graph-connected zero.
    """

    if isinstance(target, bool) or not math.isfinite(float(target)) or target < 0.0:
        raise ValueError("Sinkhorn residual target must be finite and non-negative")
    transport = output.local.transport
    if transport is None:
        return output.local_logit.sum() * 0.0
    diagnostics = transport.diagnostics
    residual = torch.maximum(
        diagnostics.row_residual_max, diagnostics.col_residual_max
    )
    finite = output.local.finite_problem & torch.isfinite(residual)
    excess = torch.relu(residual - float(target))
    return excess[finite].mean() if finite.any().item() else residual.sum() * 0.0


def compute_pairwise_loss(
    output: PairwiseOutput,
    label: Tensor,
    config: Optional[PairwiseLossConfig] = None,
    *,
    swapped_output: Optional[PairwiseOutput] = None,
    degraded_output: Optional[PairwiseOutput] = None,
    correspondence: Optional[Tensor] = None,
    correspondence_mask: Optional[Tensor] = None,
    use_training_validity: bool = False,
) -> LossBreakdown:
    config = config or PairwiseLossConfig()
    if label.dtype != torch.bool or tuple(label.shape) != tuple(
        output.fused_logit.shape
    ):
        raise TypeError("label must be bool with one value per sample")
    valid_pair = output.training_valid if use_training_validity else output.decision_valid
    valid_local = (
        output.local_training_valid if use_training_validity else output.local_valid
    )
    pair_bce = _masked_bce(output.fused_logit, label, valid_pair)
    coarse_bce = _masked_bce(output.coarse_logit, label, output.coarse_valid)
    local_bce = _masked_bce(output.local_logit, label, valid_local)
    correspondence_loss = _correspondence_loss(
        output, correspondence, correspondence_mask, valid_local
    )
    swap = output.fused_logit.sum() * 0.0
    if swapped_output is not None:
        swap = swap_consistency_loss(
            output,
            swapped_output,
            transport_weight=config.swap_transport_weight,
            use_training_validity=use_training_validity,
        )
    monotonic = output.fused_logit.sum() * 0.0
    if degraded_output is not None:
        output_valid = (
            output.training_valid if use_training_validity else output.decision_valid
        )
        degraded_valid = (
            degraded_output.training_valid
            if use_training_validity
            else degraded_output.decision_valid
        )
        valid_positive = label & output_valid & degraded_valid
        monotonic = monotonic_score_loss(
            output.fused_probability,
            degraded_output.fused_probability,
            valid_positive,
            margin=config.monotonic_margin,
        )
    residual = sinkhorn_residual_penalty(
        output, target=config.sinkhorn_residual_target
    )
    total = (
        config.pair_bce_weight * pair_bce
        + config.coarse_bce_weight * coarse_bce
        + config.local_bce_weight * local_bce
        + config.correspondence_weight * correspondence_loss
        + config.swap_weight * swap
        + config.monotonic_weight * monotonic
        + config.sinkhorn_residual_weight * residual
    )
    return LossBreakdown(
        total=total,
        pair_bce=pair_bce,
        coarse_bce=coarse_bce,
        local_bce=local_bce,
        correspondence=correspondence_loss,
        swap=swap,
        monotonic=monotonic,
        sinkhorn_residual=residual,
        valid_pair_count=int(valid_pair.sum().item()),
        valid_local_count=int(valid_local.sum().item()),
    )


def directional_candidate_loss(
    output: DirectionalPairwiseOutput,
    pair_label: Tensor,
    direction_target: Tensor,
    *,
    direction_target_valid: Optional[Tensor] = None,
    pair_weight: float = 1.0,
    direction_weight: float = 0.25,
) -> Tensor:
    """Pair MIL loss plus positive-direction/negative-all-sides supervision.

    ``direction_target`` is 0..K-1 only when an upstream direction label is
    known.  New 5k synthetic positives legitimately use -1 with
    ``direction_target_valid=False`` and receive pair-MIL supervision only.
    Negatives always use -1 and supervise every valid direction as negative.
    Direction supervision affects outputs only; it is never model input.
    """

    if pair_label.dtype != torch.bool or tuple(pair_label.shape) != tuple(
        output.pair_logit.shape
    ):
        raise TypeError("pair_label must be bool [B]")
    if direction_target.dtype != torch.long or tuple(direction_target.shape) != tuple(
        pair_label.shape
    ):
        raise TypeError("direction_target must be int64 [B]")
    if direction_target_valid is None:
        direction_target_valid = pair_label & (direction_target >= 0)
    if direction_target_valid.dtype != torch.bool or tuple(
        direction_target_valid.shape
    ) != tuple(pair_label.shape):
        raise TypeError("direction_target_valid must be bool [B]")
    if (direction_target_valid & ~pair_label).any().item():
        raise ValueError("negative pairs cannot have valid direction targets")
    if ((~direction_target_valid) & (direction_target != -1)).any().item():
        raise ValueError("unknown directions require direction_target=-1")
    positive = pair_label & output.pair_valid
    negative = ~pair_label & output.pair_valid
    supervised_positive = positive & direction_target_valid
    if supervised_positive.any().item():
        targets = direction_target[supervised_positive]
        if ((targets < 0) | (targets >= output.candidate_logits.shape[1])).any().item():
            raise ValueError("positive direction target is out of range")
    if (direction_target[~pair_label] != -1).any().item():
        raise ValueError("negative pairs require direction_target=-1")
    pair_loss = _masked_bce(output.pair_logit, pair_label, output.pair_valid)
    direction_loss = output.candidate_logits.sum() * 0.0
    parts = []
    if supervised_positive.any().item():
        masked_logits = torch.where(
            output.candidate_valid[supervised_positive],
            output.candidate_logits[supervised_positive],
            torch.full_like(output.candidate_logits[supervised_positive], -1e4),
        )
        target_candidate_valid = output.candidate_valid[supervised_positive].gather(
            1, direction_target[supervised_positive].unsqueeze(1)
        )
        if not target_candidate_valid.all().item():
            raise ValueError("supervised positive direction has no valid candidate")
        parts.append(
            F.cross_entropy(masked_logits, direction_target[supervised_positive])
        )
    if negative.any().item():
        logits = output.candidate_logits[negative]
        valid = output.candidate_valid[negative]
        losses = F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction="none"
        )
        if valid.any().item():
            parts.append(losses[valid].mean())
    if parts:
        direction_loss = torch.stack(parts).mean()
    return float(pair_weight) * pair_loss + float(direction_weight) * direction_loss


__all__ = [
    "LossBreakdown",
    "PairwiseLossConfig",
    "compute_pairwise_loss",
    "directional_candidate_loss",
    "monotonic_score_loss",
    "ragged_directional_swap_consistency_loss",
    "sinkhorn_residual_penalty",
    "swap_consistency_loss",
]
