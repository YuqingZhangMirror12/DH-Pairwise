"""One-step train/eval primitives; orchestration remains dataset-agnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.pairwise import (
    DunhuangPairwiseV02,
    HierarchicalDirectionalOutput,
    PairwiseOutput,
    PairwiseScoreSource,
)
from ..pairwise_data.exact_seam_supervision import (
    ExactPartialAssignmentLoss,
    ExactSeamTensorBatch,
    exact_partial_assignment_nll,
)
from .contracts import PairwiseBatch
from .losses import (
    LossBreakdown,
    PairwiseLossConfig,
    compute_pairwise_loss,
    directional_candidate_loss,
    ragged_directional_swap_consistency_loss,
    sinkhorn_residual_penalty,
)


@dataclass(frozen=True)
class TransportStepDiagnostics:
    """Portable aggregate of assignment health for one model invocation."""

    matcher_mode: str
    candidate_count: int
    finite_problem_count: int
    converged_count: int
    decision_valid_count: int
    nonconverged_finite_count: int
    row_residual_max: Optional[float]
    col_residual_max: Optional[float]
    iteration_min: Optional[int]
    iteration_max: Optional[int]


@dataclass(frozen=True)
class TrainStepResult:
    output: PairwiseOutput
    loss: LossBreakdown
    gradient_norm: float
    transport_diagnostics: TransportStepDiagnostics


@dataclass(frozen=True)
class EvalStepResult:
    output: PairwiseOutput
    loss: LossBreakdown
    transport_diagnostics: TransportStepDiagnostics


@dataclass(frozen=True)
class DirectionalStepResult:
    """Result for the geometry bridge's flattened ragged candidate batch."""

    output: HierarchicalDirectionalOutput
    total_loss: Tensor
    directional_loss: Tensor
    coarse_loss: Tensor
    sinkhorn_residual_loss: Tensor
    swap_loss: Tensor
    exact_assignment_loss: Tensor
    exact_assignment_breakdown: Optional[ExactPartialAssignmentLoss]
    gradient_norm: Optional[float]
    transport_diagnostics: Optional[TransportStepDiagnostics]
    swapped_output: Optional[HierarchicalDirectionalOutput]


def _transport_step_diagnostics(output: PairwiseOutput) -> TransportStepDiagnostics:
    local = output.local
    finite = local.finite_problem
    converged = local.transport_converged & finite
    row_max: Optional[float] = None
    col_max: Optional[float] = None
    iteration_min: Optional[int] = None
    iteration_max: Optional[int] = None
    if local.transport is not None and finite.any().item():
        diagnostics = local.transport.diagnostics
        row_max = float(diagnostics.row_residual_max[finite].max().detach().cpu())
        col_max = float(diagnostics.col_residual_max[finite].max().detach().cpu())
        iterations = diagnostics.iteration_count[finite]
        iteration_min = int(iterations.min().detach().cpu())
        iteration_max = int(iterations.max().detach().cpu())
    return TransportStepDiagnostics(
        matcher_mode=local.matcher_mode,
        candidate_count=int(local.logit.numel()),
        finite_problem_count=int(finite.sum().item()),
        converged_count=int(converged.sum().item()),
        decision_valid_count=int(local.decision_valid.sum().item()),
        nonconverged_finite_count=int((finite & ~converged).sum().item()),
        row_residual_max=row_max,
        col_residual_max=col_max,
        iteration_min=iteration_min,
        iteration_max=iteration_max,
    )


def _forward(model: DunhuangPairwiseV02, batch: PairwiseBatch) -> PairwiseOutput:
    return model(
        batch.coarse_a,
        batch.coarse_b,
        batch.local_a,
        batch.local_b,
        batch.token_mask_a,
        batch.token_mask_b,
    )


def train_step(
    model: DunhuangPairwiseV02,
    batch: PairwiseBatch,
    optimizer: torch.optim.Optimizer,
    *,
    loss_config: Optional[PairwiseLossConfig] = None,
    max_gradient_norm: float = 5.0,
    compute_swapped: bool = True,
    degraded_batch: Optional[PairwiseBatch] = None,
) -> TrainStepResult:
    """Run one differentiable step; invalid local samples retain coarse loss."""

    if max_gradient_norm <= 0.0:
        raise ValueError("max_gradient_norm must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = _forward(model, batch)
    swapped = _forward(model, batch.swapped()) if compute_swapped else None
    degraded = _forward(model, degraded_batch) if degraded_batch is not None else None
    loss = compute_pairwise_loss(
        output,
        batch.label,
        loss_config,
        swapped_output=swapped,
        degraded_output=degraded,
        correspondence=batch.correspondence,
        correspondence_mask=batch.correspondence_mask,
        use_training_validity=True,
    )
    if not torch.isfinite(loss.total).item():
        raise FloatingPointError("training loss is non-finite")
    loss.total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_gradient_norm, error_if_nonfinite=True
    )
    optimizer.step()
    return TrainStepResult(
        output=output,
        loss=loss,
        gradient_norm=float(gradient_norm.detach().cpu().item()),
        transport_diagnostics=_transport_step_diagnostics(output),
    )


def eval_step(
    model: DunhuangPairwiseV02,
    batch: PairwiseBatch,
    *,
    loss_config: Optional[PairwiseLossConfig] = None,
) -> EvalStepResult:
    model.eval()
    with torch.inference_mode():
        output = _forward(model, batch)
        loss = compute_pairwise_loss(output, batch.label, loss_config)
    return EvalStepResult(
        output=output,
        loss=loss,
        transport_diagnostics=_transport_step_diagnostics(output),
    )


def _directional_forward(
    model: DunhuangPairwiseV02,
    model_inputs: Mapping[str, Tensor],
    *,
    score_source: PairwiseScoreSource = PairwiseScoreSource.FUSED,
) -> HierarchicalDirectionalOutput:
    required = {
        "coarse_a",
        "coarse_b",
        "local_a",
        "local_b",
        "token_mask_a",
        "token_mask_b",
        "sample_index",
        "direction_index",
    }
    missing = sorted(required - set(model_inputs))
    if missing:
        raise ValueError(
            "directional model inputs are missing {}".format(", ".join(missing))
        )
    allowed = required | {"candidate_valid", "correspondence_mask"}
    unexpected = sorted(set(model_inputs) - allowed)
    if unexpected:
        raise ValueError(
            "directional model inputs contain unexpected {}".format(
                ", ".join(unexpected)
            )
        )
    return model.forward_flat_candidates(
        **dict(model_inputs), score_source=score_source
    )


def swap_directional_model_inputs(
    model_inputs: Mapping[str, Tensor],
) -> Mapping[str, Tensor]:
    """Swap A/B in a ragged batch and invert its four direction slots.

    Candidate order and sample ownership are preserved, so the resulting
    forward pass can be compared arc-for-arc.  The geometry tests establish
    that candidate sequences require no hidden reversal beyond A/B exchange.
    """

    required = {
        "coarse_a",
        "coarse_b",
        "local_a",
        "local_b",
        "token_mask_a",
        "token_mask_b",
        "sample_index",
        "direction_index",
    }
    missing = sorted(required - set(model_inputs))
    if missing:
        raise ValueError(
            "directional model inputs are missing {}".format(", ".join(missing))
        )
    direction = model_inputs["direction_index"]
    if direction.dtype != torch.long or direction.ndim != 1:
        raise TypeError("direction_index must be int64 [N]")
    if ((direction < 0) | (direction >= 4)).any().item():
        raise ValueError("direction_index is out of range")
    inverse = torch.tensor([1, 0, 3, 2], dtype=torch.long, device=direction.device)
    swapped = {
        "coarse_a": model_inputs["coarse_b"],
        "coarse_b": model_inputs["coarse_a"],
        "local_a": model_inputs["local_b"],
        "local_b": model_inputs["local_a"],
        "token_mask_a": model_inputs["token_mask_b"],
        "token_mask_b": model_inputs["token_mask_a"],
        "sample_index": model_inputs["sample_index"],
        "direction_index": inverse.index_select(0, direction),
    }
    if "candidate_valid" in model_inputs:
        swapped["candidate_valid"] = model_inputs["candidate_valid"]
    if "correspondence_mask" in model_inputs:
        correspondence = model_inputs["correspondence_mask"]
        if correspondence.dtype != torch.bool or correspondence.ndim != 3:
            raise TypeError("correspondence_mask must be bool [N, L_a, L_b]")
        swapped["correspondence_mask"] = correspondence.transpose(1, 2)
    return swapped


def _coarse_auxiliary_loss(
    output: HierarchicalDirectionalOutput, label: Tensor
) -> Tensor:
    coarse = output.coarse_output
    if output.score_source == PairwiseScoreSource.LOCAL.value:
        if coarse is not None:
            raise RuntimeError(
                "local-only output unexpectedly contains coarse evidence"
            )
        # A constant zero keeps the auxiliary out of the local graph as well as
        # the coarse graph.  Direction supervision remains attached solely to
        # the selected local arc logits.
        return output.direction_output.pair_logit.new_zeros(())
    if coarse is None:
        raise RuntimeError("directional model output is missing coarse evidence")
    if label.dtype != torch.bool or tuple(label.shape) != tuple(coarse.logit.shape):
        raise TypeError("directional label must be bool [B]")
    losses = F.binary_cross_entropy_with_logits(
        coarse.logit, label.to(coarse.logit.dtype), reduction="none"
    )
    if coarse.valid_problem.any().item():
        return losses[coarse.valid_problem].mean()
    return coarse.logit.sum() * 0.0


def _directional_sinkhorn_residual(
    output: HierarchicalDirectionalOutput, target: float
) -> Tensor:
    candidate = output.arc_pairwise_output
    if candidate is None:
        return output.direction_output.pair_logit.sum() * 0.0
    return sinkhorn_residual_penalty(candidate, target=target)


def train_directional_step(
    model: DunhuangPairwiseV02,
    model_inputs: Mapping[str, Tensor],
    label: Tensor,
    direction_target: Tensor,
    direction_target_valid: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    score_source: PairwiseScoreSource = PairwiseScoreSource.FUSED,
    coarse_loss_weight: float = 0.25,
    direction_loss_weight: float = 0.25,
    sinkhorn_residual_weight: float = 0.05,
    sinkhorn_residual_target: float = 1e-3,
    compute_swapped: bool = False,
    swap_loss_weight: float = 0.1,
    swap_transport_weight: float = 0.1,
    exact_assignment_targets: Optional[ExactSeamTensorBatch] = None,
    exact_assignment_loss_weight: float = 0.0,
    exact_match_weight: float = 1.0,
    exact_dustbin_weight: float = 1.0,
    max_gradient_norm: float = 5.0,
) -> DirectionalStepResult:
    """Train on ragged arc candidates from ``RaggedGeometryBatch`` tensors.

    ``score_source='local'`` is a strict local-only arm: coarse/fusion modules
    are not executed, and ``coarse_loss`` is an exact constant zero even when
    the backwards-compatible ``coarse_loss_weight`` default is non-zero.
    """

    for name, value in (
        ("coarse_loss_weight", coarse_loss_weight),
        ("direction_loss_weight", direction_loss_weight),
        ("sinkhorn_residual_weight", sinkhorn_residual_weight),
        ("sinkhorn_residual_target", sinkhorn_residual_target),
        ("swap_loss_weight", swap_loss_weight),
        ("swap_transport_weight", swap_transport_weight),
        ("exact_assignment_loss_weight", exact_assignment_loss_weight),
        ("exact_match_weight", exact_match_weight),
        ("exact_dustbin_weight", exact_dustbin_weight),
    ):
        if not math.isfinite(float(value)) or value < 0.0:
            raise ValueError("{} must be finite and non-negative".format(name))
    if type(compute_swapped) is not bool:
        raise TypeError("compute_swapped must be bool")
    if exact_assignment_loss_weight > 0.0 and exact_assignment_targets is None:
        raise ValueError("positive exact assignment weight requires exact targets")
    if exact_assignment_targets is not None and not isinstance(
        exact_assignment_targets, ExactSeamTensorBatch
    ):
        raise TypeError("exact_assignment_targets must be ExactSeamTensorBatch")
    if any(str(name).startswith("exact_") for name in model_inputs):
        raise ValueError("exact supervision tensors cannot enter model_inputs")
    if max_gradient_norm <= 0.0:
        raise ValueError("max_gradient_norm must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = _directional_forward(model, model_inputs, score_source=score_source)
    directional = directional_candidate_loss(
        output.training_direction_output,
        label,
        direction_target,
        direction_target_valid=direction_target_valid,
        direction_weight=direction_loss_weight,
    )
    coarse = _coarse_auxiliary_loss(output, label)
    residual = _directional_sinkhorn_residual(output, sinkhorn_residual_target)
    exact_breakdown = None
    exact_assignment = output.direction_output.pair_logit.sum() * 0.0
    if exact_assignment_targets is not None:
        if not torch.equal(
            exact_assignment_targets.sample_index.to(
                device=model_inputs["sample_index"].device
            ),
            model_inputs["sample_index"],
        ) or not torch.equal(
            exact_assignment_targets.direction_index.to(
                device=model_inputs["direction_index"].device
            ),
            model_inputs["direction_index"],
        ):
            raise ValueError("exact assignment targets changed candidate ownership")
        if output.arc_pairwise_output is None:
            raise ValueError("exact assignment targets require local arc output")
        exact_breakdown = exact_partial_assignment_nll(
            output.arc_pairwise_output.local,
            exact_assignment_targets,
            match_weight=exact_match_weight,
            dustbin_weight=exact_dustbin_weight,
        )
        exact_assignment = exact_breakdown.total
    swapped_output = None
    swap = output.direction_output.pair_logit.sum() * 0.0
    if compute_swapped:
        swapped_output = _directional_forward(
            model,
            swap_directional_model_inputs(model_inputs),
            score_source=score_source,
        )
        swap = ragged_directional_swap_consistency_loss(
            output,
            swapped_output,
            transport_weight=swap_transport_weight,
        )
    total = (
        directional
        + float(coarse_loss_weight) * coarse
        + float(sinkhorn_residual_weight) * residual
        + float(swap_loss_weight) * swap
        + float(exact_assignment_loss_weight) * exact_assignment
    )
    if not torch.isfinite(total).item():
        raise FloatingPointError("directional training loss is non-finite")
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_gradient_norm, error_if_nonfinite=True
    )
    optimizer.step()
    return DirectionalStepResult(
        output=output,
        total_loss=total,
        directional_loss=directional,
        coarse_loss=coarse,
        sinkhorn_residual_loss=residual,
        swap_loss=swap,
        exact_assignment_loss=exact_assignment,
        exact_assignment_breakdown=exact_breakdown,
        gradient_norm=float(gradient_norm.detach().cpu().item()),
        transport_diagnostics=(
            _transport_step_diagnostics(output.arc_pairwise_output)
            if output.arc_pairwise_output is not None
            else None
        ),
        swapped_output=swapped_output,
    )


def eval_directional_step(
    model: DunhuangPairwiseV02,
    model_inputs: Mapping[str, Tensor],
    label: Tensor,
    direction_target: Tensor,
    direction_target_valid: Tensor,
    *,
    score_source: PairwiseScoreSource = PairwiseScoreSource.FUSED,
    coarse_loss_weight: float = 0.25,
    direction_loss_weight: float = 0.25,
    sinkhorn_residual_weight: float = 0.05,
    sinkhorn_residual_target: float = 1e-3,
) -> DirectionalStepResult:
    model.eval()
    with torch.inference_mode():
        output = _directional_forward(model, model_inputs, score_source=score_source)
        directional = directional_candidate_loss(
            output.direction_output,
            label,
            direction_target,
            direction_target_valid=direction_target_valid,
            direction_weight=direction_loss_weight,
        )
        coarse = _coarse_auxiliary_loss(output, label)
        residual = _directional_sinkhorn_residual(output, sinkhorn_residual_target)
        total = (
            directional
            + float(coarse_loss_weight) * coarse
            + float(sinkhorn_residual_weight) * residual
        )
    return DirectionalStepResult(
        output=output,
        total_loss=total,
        directional_loss=directional,
        coarse_loss=coarse,
        sinkhorn_residual_loss=residual,
        swap_loss=output.direction_output.pair_logit.sum() * 0.0,
        exact_assignment_loss=output.direction_output.pair_logit.sum() * 0.0,
        exact_assignment_breakdown=None,
        gradient_norm=None,
        transport_diagnostics=(
            _transport_step_diagnostics(output.arc_pairwise_output)
            if output.arc_pairwise_output is not None
            else None
        ),
        swapped_output=None,
    )


__all__ = [
    "DirectionalStepResult",
    "EvalStepResult",
    "TrainStepResult",
    "TransportStepDiagnostics",
    "eval_directional_step",
    "eval_step",
    "swap_directional_model_inputs",
    "train_directional_step",
    "train_step",
]
