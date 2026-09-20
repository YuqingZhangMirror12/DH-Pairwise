"""Minimal fitting adapter for the frozen structured transport readout.

This module intentionally contains no checkpoint loader, cache, CLI, or real
data runner.  A future 6,964-pair pilot can detach one Exact-winner forward,
pass the frozen matrices to :func:`frozen_transport_matrix_fit_step`, and
optimize only ``FrozenTransportMatrixReadout.parameters()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from staging.pairwise_v0_2.models.frozen_transport_matrix_readout import (
    FrozenTransportMatrixReadout,
    StructuredTransportReadoutOutput,
)
from staging.pairwise_v0_2.models.pairwise import (
    ArcPoolingConfig,
    HierarchicalDirectionalOutput,
    aggregate_flat_arc_candidates,
)
from staging.pairwise_v0_2.training.losses import directional_candidate_loss


@dataclass(frozen=True)
class FrozenTransportMatrixFitOutput:
    """One differentiable head-only fit step; the caller owns the optimizer."""

    loss: Tensor
    readout: StructuredTransportReadoutOutput
    hierarchy: HierarchicalDirectionalOutput


def frozen_transport_matrix_fit_step(
    readout: FrozenTransportMatrixReadout,
    *,
    base_arc_logit: Tensor,
    assignment: Tensor,
    affinity: Tensor,
    unmatched_a: Tensor,
    unmatched_b: Tensor,
    token_mask_a: Tensor,
    token_mask_b: Tensor,
    correspondence_mask: Tensor,
    arc_decision_valid: Tensor,
    arc_training_valid: Tensor,
    sample_index: Tensor,
    direction_index: Tensor,
    geometry_valid: Tensor,
    pair_label: Tensor,
    direction_target: Tensor,
    direction_target_valid: Tensor,
    arc_pooling: Optional[ArcPoolingConfig] = None,
    direction_temperature: float = 0.25,
    pair_weight: float = 1.0,
    direction_weight: float = 0.25,
) -> FrozenTransportMatrixFitOutput:
    """Build pair/four-direction supervision around target-free model inputs.

    Backbone tensors should be detached by the caller.  Crucially,
    ``direction_target`` is passed only to ``directional_candidate_loss`` after
    the readout forward has completed; it is not an input to the matrix CNN.
    """

    if not isinstance(readout, FrozenTransportMatrixReadout):
        raise TypeError("readout must be FrozenTransportMatrixReadout")
    if pair_label.ndim != 1 or pair_label.dtype != torch.bool:
        raise TypeError("pair_label must be bool [B]")
    batch_size = int(pair_label.numel())
    if batch_size < 1:
        raise ValueError("pair_label cannot be empty")
    if geometry_valid.dtype != torch.bool or tuple(geometry_valid.shape) != (
        batch_size,
    ):
        raise TypeError("geometry_valid and pair_label must both be bool [B]")
    for name, value in (
        ("arc_decision_valid", arc_decision_valid),
        ("arc_training_valid", arc_training_valid),
    ):
        if value.dtype != torch.bool or tuple(value.shape) != tuple(
            base_arc_logit.shape
        ):
            raise TypeError(name + " must be bool [N]")
    for name, value in (
        ("sample_index", sample_index),
        ("direction_index", direction_index),
    ):
        if value.dtype != torch.long or tuple(value.shape) != tuple(
            base_arc_logit.shape
        ):
            raise TypeError(name + " must be int64 [N]")

    structured = readout(
        assignment,
        affinity,
        unmatched_a,
        unmatched_b,
        token_mask_a,
        token_mask_b,
        correspondence_mask,
        base_arc_logit=base_arc_logit,
    )
    geometry_for_arc = geometry_valid.index_select(0, sample_index)
    decision_valid = arc_decision_valid & structured.transport_valid & geometry_for_arc
    training_valid = arc_training_valid & structured.transport_valid & geometry_for_arc
    hierarchy = aggregate_flat_arc_candidates(
        structured.arc_logit,
        decision_valid,
        sample_index,
        direction_index,
        batch_size=batch_size,
        arc_pooling=arc_pooling,
        direction_temperature=direction_temperature,
        training_arc_valid=training_valid,
    )
    loss = directional_candidate_loss(
        hierarchy.training_direction_output,
        pair_label,
        direction_target,
        direction_target_valid=direction_target_valid,
        pair_weight=pair_weight,
        direction_weight=direction_weight,
    )
    return FrozenTransportMatrixFitOutput(
        loss=loss,
        readout=structured,
        hierarchy=hierarchy,
    )


__all__ = [
    "FrozenTransportMatrixFitOutput",
    "frozen_transport_matrix_fit_step",
]
