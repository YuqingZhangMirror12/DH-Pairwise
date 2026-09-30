"""Vectorized physical batches with the decoupled logical-microbatch1 loss.

The old global loss intentionally normalizes many terms over only samples
having that supervision. Applying it directly to a physical batch would change
the microbatch1 experiment. Here EVERY term remains [B] until the final mean:
an unsupervised sample contributes zero, not removal from the batch denominator.

This is loss equivalence for the same per-sample outputs. It does not promise
bitwise equality of different-batch model kernels, stochastic layers or batch
normalization. No model forward is called here and no old default is replaced.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from staging.pairwise_v0_2.training.rachel_n512_loss import (
    RachelN512LossConfig, _validate_targets)

SCHEMA = "rachel-decoupled-samplewise-loss/1"
LOSS_NAMES = ("fused_pair_bce", "coarse_pair_bce", "local_pair_bce",
              "assignment_nll", "translation_smooth_l1", "sinkhorn_residual")


def _token_mean(values: Tensor, mask: Tensor) -> Tensor:
    """[B,N] -> [B], averaging tokens within each sample, zero if none."""
    if values.ndim != 2 or values.shape != mask.shape or mask.dtype != torch.bool:
        raise TypeError("token values/mask must have matching [B,N] shapes")
    return (values * mask.to(values.dtype)).sum(1) / mask.sum(1).clamp_min(1)


def _pair_values(logit: Tensor, labels: Tensor, valid: Tensor) -> Tensor:
    if logit.ndim != 1 or labels.shape != logit.shape or valid.shape != logit.shape or valid.dtype != torch.bool:
        raise TypeError("pair logit/label/valid must be matching [B] vectors")
    return F.binary_cross_entropy_with_logits(logit, labels.to(logit.dtype), reduction="none") * valid.to(logit.dtype)


def samplewise_phase_terms(output, targets, pose_supervision_enabled, base_config: RachelN512LossConfig, phase: str):
    """Return un-reduced [B] terms with exactly one logical sample per entry.

    In M all pair BCE terms are diagnostic only; their coefficient is zero.
    In C only fused PairBCE is computed; correspondence/pose targets are unread.
    Runtime target checks remain the original checks for M. Masked terms with
    no effective supervision are zero, as in the valid finite B=1 legacy path.
    """
    labels = targets[0]
    if labels.ndim != 1 or not len(labels):
        raise ValueError("samplewise loss requires a nonempty physical batch")
    if phase == "classifier":
        fused = _pair_values(output.fused_logit, labels, output.training_valid)
        terms = {key: torch.zeros_like(fused) for key in LOSS_NAMES}
        terms["fused_pair_bce"] = fused
        terms["total"] = fused
        return terms
    if phase != "matcher":
        raise ValueError("phase must be matcher or classifier")
    labels, target_a, target_b, translation_target_rc, translation_valid = targets
    _, valid = _validate_targets(output, *targets, validate_values=base_config.validate_runtime_targets)
    pose = pose_supervision_enabled
    if pose.dtype != torch.bool or pose.shape != translation_valid.shape:
        raise TypeError("pose_supervision_enabled must be bool [B]")
    if (pose & ~translation_valid).any().item():
        raise ValueError("weathering cannot enable pose supervision for a negative sample")

    match_mask = (target_a >= 0) & valid[:, None]
    probabilities = output.assignment.gather(2, target_a.clamp(min=0)[:, :, None]).squeeze(2)
    match_nll = _token_mean(-torch.log(probabilities.clamp_min(base_config.epsilon)), match_mask)

    dustbin_a_mask = (target_a == -1) & valid[:, None]
    dustbin_b_mask = (target_b == -1) & valid[:, None]
    dustbin_a_nll = _token_mean(-torch.log(output.unmatched_a.clamp_min(base_config.epsilon)), dustbin_a_mask)
    dustbin_b_nll = _token_mean(-torch.log(output.unmatched_b.clamp_min(base_config.epsilon)), dustbin_b_mask)
    # Presence is per sample, not .any() over the entire physical batch.
    present_a = dustbin_a_mask.any(1).to(output.assignment.dtype)
    present_b = dustbin_b_mask.any(1).to(output.assignment.dtype)
    dustbin_nll = (dustbin_a_nll * present_a + dustbin_b_nll * present_b) / (present_a + present_b).clamp_min(1.)
    assignment_nll = match_nll + dustbin_nll

    translation_mask = pose & translation_valid & valid
    translation = F.smooth_l1_loss(output.translation_hat_rc / base_config.translation_scale_px,
        translation_target_rc.to(output.translation_hat_rc.dtype) / base_config.translation_scale_px,
        reduction="none").mean(1) * translation_mask.to(output.translation_hat_rc.dtype)

    diagnostics = output.transport.diagnostics
    residual = torch.maximum(diagnostics.row_residual_max, diagnostics.col_residual_max)
    residual_mask = valid & torch.isfinite(residual)
    residual_loss = torch.relu(residual - base_config.sinkhorn_residual_target) * residual_mask.to(residual.dtype)
    terms = dict(fused_pair_bce=_pair_values(output.fused_logit, labels, valid),
        coarse_pair_bce=_pair_values(output.coarse_logit, labels, output.coarse.valid_problem),
        local_pair_bce=_pair_values(output.local_logit, labels, valid),
        assignment_nll=assignment_nll, translation_smooth_l1=translation, sinkhorn_residual=residual_loss,
        match_nll=match_nll, dustbin_a_nll=dustbin_a_nll, dustbin_b_nll=dustbin_b_nll, dustbin_nll=dustbin_nll)
    # Registered M12 coefficients; no coarse/local/fused/R classification term.
    terms["total"] = .5 * assignment_nll + .5 * translation + .05 * residual_loss
    return terms


def compute_samplewise_phase_loss(output, targets, pose_supervision_enabled, base_config, phase):
    """Trainer API: (scalar mean over ALL B samples, six scalar components).

    Multiply the returned mean once by physicalB/actual_effective_group_size.
    This preserves logical per-sample weighting for any positive batch sizes;
    changing the effective batch changes optimizer cadence/trajectory and is
    NOT claimed equivalent to the historical effective16 optimization path.
    """
    terms = samplewise_phase_terms(output, targets, pose_supervision_enabled, base_config, phase)
    total = terms["total"].mean()
    if not torch.isfinite(total).item():
        raise FloatingPointError("samplewise decoupled loss is nonfinite")
    return total, {key: terms[key].mean() for key in LOSS_NAMES}


__all__ = ["SCHEMA", "LOSS_NAMES", "samplewise_phase_terms", "compute_samplewise_phase_loss"]
