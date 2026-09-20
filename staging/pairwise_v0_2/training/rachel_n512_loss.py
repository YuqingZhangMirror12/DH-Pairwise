"""Target-side losses for the Rachel full-contour N=512 matcher."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Output


@dataclass(frozen=True)
class RachelN512LossConfig:
    fused_pair_weight: float = 1.0
    coarse_pair_weight: float = 0.25
    local_pair_weight: float = 0.5
    assignment_weight: float = 0.5
    # Calibrated on a real Rachel train batch so the weighted translation
    # gradient is a small-but-material auxiliary (~3.7% of assignment at
    # initialization), while exact correspondence remains the primary signal.
    translation_weight: float = 0.5
    sinkhorn_residual_weight: float = 0.05
    sinkhorn_residual_target: float = 1e-3
    # Translation errors are expressed in physical pixels.  Dividing by the
    # full 800-pixel canvas made this gradient effectively vanish beside the
    # partial-assignment objective.  A 32-pixel scale keeps the loss
    # dimensionless while preserving useful geometric gradients.
    translation_scale_px: float = 32.0
    epsilon: float = 1e-8
    validate_runtime_targets: bool = True
    collect_cpu_diagnostics: bool = True

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name in {"validate_runtime_targets", "collect_cpu_diagnostics"}:
                if type(value) is not bool:  # noqa: E721
                    raise TypeError(name + " must be bool")
                continue
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(name + " must be finite and non-negative")
        if self.translation_scale_px <= 0.0:
            raise ValueError("translation_scale_px must be positive")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must be in (0,1)")


@dataclass(frozen=True)
class RachelN512Loss:
    total: Tensor
    fused_pair_bce: Tensor
    coarse_pair_bce: Tensor
    local_pair_bce: Tensor
    assignment_nll: Tensor
    match_nll: Tensor
    dustbin_a_nll: Tensor
    dustbin_b_nll: Tensor
    translation_smooth_l1: Tensor
    sinkhorn_residual: Tensor
    valid_pair_count: int
    supervised_match_count: int
    supervised_dustbin_a_count: int
    supervised_dustbin_b_count: int
    translation_count: int


def _validate_targets(
    output: RachelN512Output,
    labels: Tensor,
    target_a: Tensor,
    target_b: Tensor,
    translation_target_rc: Tensor,
    translation_valid: Tensor,
    *,
    validate_values: bool,
) -> Tuple[Tensor, Tensor]:
    batch, count_a, count_b = output.assignment.shape
    if labels.shape != (batch,) or not labels.is_floating_point():
        raise TypeError("labels must be floating-point [B]")
    if target_a.dtype != torch.long or target_a.shape != (batch, count_a):
        raise TypeError("target_a must be int64 [B,Na]")
    if target_b.dtype != torch.long or target_b.shape != (batch, count_b):
        raise TypeError("target_b must be int64 [B,Nb]")
    if (
        translation_target_rc.shape != (batch, 2)
        or not translation_target_rc.is_floating_point()
    ):
        raise TypeError("translation_target_rc must be floating-point [B,2]")
    if translation_valid.dtype != torch.bool or translation_valid.shape != (batch,):
        raise TypeError("translation_valid must be bool [B]")
    positive = labels == 1.0
    if validate_values:
        if (
            not torch.isfinite(labels).all().item()
            or ((labels != 0.0) & (labels != 1.0)).any().item()
        ):
            raise ValueError("labels must contain only 0 or 1")
        if ((target_a < -2) | (target_a >= count_b)).any().item():
            raise ValueError("target_a contains an invalid assignment code")
        if ((target_b < -2) | (target_b >= count_a)).any().item():
            raise ValueError("target_b contains an invalid assignment code")
        if not torch.isfinite(translation_target_rc[translation_valid]).all().item():
            raise ValueError("valid translation targets must be finite")
        if not torch.equal(translation_valid, positive):
            raise ValueError("translation supervision must exist for positives only")
        # Keep validation vectorized: a Python loop over CUDA targets caused
        # one synchronization per correspondence in every training step.
        matched_a_mask = target_a >= 0
        positive_has_match = matched_a_mask.any(dim=1)
        if (positive & ~positive_has_match).any().item():
            raise ValueError("positive sample has no compact correspondence target")
        if (
            ((~positive) & (matched_a_mask.any(dim=1) | (target_b >= 0).any(dim=1)))
            .any()
            .item()
        ):
            raise ValueError("negative sample cannot carry a real correspondence")
        safe_b = target_a.clamp(min=0)
        reciprocal_a = target_b.gather(1, safe_b)
        expected_a = torch.arange(count_a, device=target_a.device)[None, :]
        if (matched_a_mask & (reciprocal_a != expected_a)).any().item():
            raise ValueError("compact assignment targets are not reciprocal")
    return positive, output.training_valid


def _per_sample_masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    if value.shape != mask.shape or value.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("masked token values must have matching [B,N] shapes")
    counts = mask.sum(dim=1)
    means = (value * mask.to(value.dtype)).sum(dim=1) / counts.clamp_min(1)
    selected = counts > 0
    return (means * selected.to(means.dtype)).sum() / selected.sum().clamp_min(1)


def compute_rachel_n512_loss(
    output: RachelN512Output,
    labels: Tensor,
    target_a: Tensor,
    target_b: Tensor,
    translation_target_rc: Tensor,
    translation_valid: Tensor,
    config: RachelN512LossConfig = RachelN512LossConfig(),
) -> RachelN512Loss:
    """Pair, partial-assignment, and pure-translation supervision."""

    positive, valid_pair = _validate_targets(
        output,
        labels,
        target_a,
        target_b,
        translation_target_rc,
        translation_valid,
        validate_values=config.validate_runtime_targets,
    )
    pair_target = labels.to(dtype=output.fused_logit.dtype)

    def pair_bce(logit: Tensor, valid: Tensor) -> Tensor:
        values = F.binary_cross_entropy_with_logits(
            logit, pair_target, reduction="none"
        )
        return (values * valid.to(values.dtype)).sum() / valid.sum().clamp_min(1)

    fused_bce = pair_bce(output.fused_logit, valid_pair)
    coarse_bce = pair_bce(output.coarse_logit, output.coarse.valid_problem)
    local_bce = pair_bce(output.local_logit, valid_pair)

    match_mask = (target_a >= 0) & valid_pair[:, None]
    safe_match = target_a.clamp(min=0)
    match_probability = output.assignment.gather(2, safe_match[:, :, None]).squeeze(2)
    match_values = -torch.log(match_probability.clamp_min(config.epsilon))
    match_nll = _per_sample_masked_mean(match_values, match_mask)

    dustbin_a_mask = (target_a == -1) & valid_pair[:, None]
    dustbin_b_mask = (target_b == -1) & valid_pair[:, None]
    dustbin_a_values = -torch.log(output.unmatched_a.clamp_min(config.epsilon))
    dustbin_b_values = -torch.log(output.unmatched_b.clamp_min(config.epsilon))
    dustbin_a_nll = _per_sample_masked_mean(dustbin_a_values, dustbin_a_mask)
    dustbin_b_nll = _per_sample_masked_mean(dustbin_b_values, dustbin_b_mask)
    dustbin_presence = torch.stack(
        (
            dustbin_a_mask.any().to(output.assignment.dtype),
            dustbin_b_mask.any().to(output.assignment.dtype),
        )
    )
    dustbin_nll = (
        torch.stack((dustbin_a_nll, dustbin_b_nll)) * dustbin_presence
    ).sum() / dustbin_presence.sum().clamp_min(1.0)
    assignment_nll = match_nll + dustbin_nll

    translation_mask = translation_valid & valid_pair
    translation_values = F.smooth_l1_loss(
        output.translation_hat_rc / config.translation_scale_px,
        translation_target_rc.to(output.translation_hat_rc.dtype)
        / config.translation_scale_px,
        reduction="none",
    ).mean(dim=1)
    translation_loss = (
        translation_values * translation_mask.to(translation_values.dtype)
    ).sum() / translation_mask.sum().clamp_min(1)

    diagnostics = output.transport.diagnostics
    residual = torch.maximum(diagnostics.row_residual_max, diagnostics.col_residual_max)
    residual_mask = output.training_valid & torch.isfinite(residual)
    residual_values = torch.relu(residual - config.sinkhorn_residual_target)
    residual_loss = (
        residual_values * residual_mask.to(residual_values.dtype)
    ).sum() / residual_mask.sum().clamp_min(1)

    total = (
        config.fused_pair_weight * fused_bce
        + config.coarse_pair_weight * coarse_bce
        + config.local_pair_weight * local_bce
        + config.assignment_weight * assignment_nll
        + config.translation_weight * translation_loss
        + config.sinkhorn_residual_weight * residual_loss
    )
    if config.validate_runtime_targets and not torch.isfinite(total).item():
        raise FloatingPointError("Rachel N=512 loss is non-finite")
    if config.collect_cpu_diagnostics:
        valid_pair_count = int(valid_pair.sum().item())
        supervised_match_count = int(match_mask.sum().item())
        supervised_dustbin_a_count = int(dustbin_a_mask.sum().item())
        supervised_dustbin_b_count = int(dustbin_b_mask.sum().item())
        translation_count = int(translation_mask.sum().item())
    else:
        valid_pair_count = -1
        supervised_match_count = -1
        supervised_dustbin_a_count = -1
        supervised_dustbin_b_count = -1
        translation_count = -1
    return RachelN512Loss(
        total=total,
        fused_pair_bce=fused_bce,
        coarse_pair_bce=coarse_bce,
        local_pair_bce=local_bce,
        assignment_nll=assignment_nll,
        match_nll=match_nll,
        dustbin_a_nll=dustbin_a_nll,
        dustbin_b_nll=dustbin_b_nll,
        translation_smooth_l1=translation_loss,
        sinkhorn_residual=residual_loss,
        valid_pair_count=valid_pair_count,
        supervised_match_count=supervised_match_count,
        supervised_dustbin_a_count=supervised_dustbin_a_count,
        supervised_dustbin_b_count=supervised_dustbin_b_count,
        translation_count=translation_count,
    )


__all__ = [
    "RachelN512Loss",
    "RachelN512LossConfig",
    "compute_rachel_n512_loss",
]
