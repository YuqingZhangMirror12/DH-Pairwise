"""Frozen pairwise anti-order concordance for Exact-Sinkhorn transport.

Production keypoints expose real contour arc fractions.  They are circularly
unwrapped per facing side and scale using a geometry-only largest-gap cut.
Opposing image-clockwise contours traverse a true seam in opposite order.

Absolute endpoint alignment is deliberately not assumed: a seam may cover a
subsequence or cross several cardinal runs.  Instead, the readout measures
transported *pairs* of edges.  Two real assignments are anti-concordant when
their A order is increasing and their B order is decreasing.  The statistic
is the anti-concordant pair mass divided by the available within-scale pair
capacity.  It is differentiable in the Sinkhorn plan, loses mass quadratically
when tokens route to dustbin, and is invariant to swapping A/B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from staging.pairwise_v0_2.geometry.keypoint_candidates import (
    KeypointPairCandidate,
    KeypointScaleToken,
)


READOUT_ORDER_COHERENT = "frozen_exact_plus_pairwise_anti_order_mass"


@dataclass(frozen=True)
class OrderCoordinateSidecar:
    """Readout-only coordinates aligned with one ragged candidate batch."""

    coordinate_a: Tensor
    coordinate_b: Tensor
    scale_index_a: Tensor
    scale_index_b: Tensor
    valid_a: Tensor
    valid_b: Tensor

    def __post_init__(self) -> None:
        if self.coordinate_a.ndim != 2 or not self.coordinate_a.is_floating_point():
            raise TypeError("coordinate_a must be floating-point [N,La]")
        if self.coordinate_b.ndim != 2 or not self.coordinate_b.is_floating_point():
            raise TypeError("coordinate_b must be floating-point [N,Lb]")
        for suffix in ("a", "b"):
            coordinate = getattr(self, "coordinate_" + suffix)
            scale = getattr(self, "scale_index_" + suffix)
            valid = getattr(self, "valid_" + suffix)
            if scale.dtype != torch.long or tuple(scale.shape) != tuple(
                coordinate.shape
            ):
                raise TypeError("scale_index_{} must be int64".format(suffix))
            if valid.dtype != torch.bool or tuple(valid.shape) != tuple(
                coordinate.shape
            ):
                raise TypeError("valid_{} must be bool".format(suffix))
            if (scale[valid] < 0).any().item() or (scale[~valid] != -1).any().item():
                raise ValueError("validity and scale-index sentinels disagree")
        if self.coordinate_a.shape[0] != self.coordinate_b.shape[0]:
            raise ValueError("sidecar candidate counts differ")
        if not torch.isfinite(self.coordinate_a).all().item() or not torch.isfinite(
            self.coordinate_b
        ).all().item():
            raise ValueError("order coordinates must be finite")
        if (
            ((self.coordinate_a < 0.0) | (self.coordinate_a > 1.0)).any().item()
            or ((self.coordinate_b < 0.0) | (self.coordinate_b > 1.0)).any().item()
        ):
            raise ValueError("order coordinates must lie in [0,1]")

    def to(self, device: torch.device, *, dtype: torch.dtype) -> "OrderCoordinateSidecar":
        return OrderCoordinateSidecar(
            coordinate_a=self.coordinate_a.to(device=device, dtype=dtype),
            coordinate_b=self.coordinate_b.to(device=device, dtype=dtype),
            scale_index_a=self.scale_index_a.to(device=device),
            scale_index_b=self.scale_index_b.to(device=device),
            valid_a=self.valid_a.to(device=device),
            valid_b=self.valid_b.to(device=device),
        )


@dataclass(frozen=True)
class ExactTargetOrderConcordance:
    """Per-candidate/scale pair-order counts from synthetic exact targets."""

    candidate_index: Tensor
    scale_index: Tensor
    matched_edge_count: Tensor
    anti_pair_count: Tensor
    monotone_pair_count: Tensor
    tied_pair_count: Tensor

    def __post_init__(self) -> None:
        shape = tuple(self.candidate_index.shape)
        for name in (
            "candidate_index",
            "scale_index",
            "matched_edge_count",
            "anti_pair_count",
            "monotone_pair_count",
            "tied_pair_count",
        ):
            value = getattr(self, name)
            if value.dtype != torch.long or tuple(value.shape) != shape:
                raise TypeError(name + " must be int64 [G]")
        if self.candidate_index.ndim != 1:
            raise TypeError("exact concordance rows must be one-dimensional")


def _unwrap_scale_tokens(
    tokens: Sequence[KeypointScaleToken],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circularly unwrap each scale using only emitted patch arc fractions."""

    count = len(tokens)
    coordinate = np.zeros(count, dtype=np.float64)
    valid = np.zeros(count, dtype=np.bool_)
    scale_index = np.full(count, -1, dtype=np.int64)
    by_scale: dict[int, list[int]] = {}
    for index, token in enumerate(tokens):
        by_scale.setdefault(int(token.scale_index), []).append(index)
    for scale, indices in sorted(by_scale.items()):
        fractions = np.asarray(
            [float(tokens[index].source_arc_fraction) % 1.0 for index in indices],
            dtype=np.float64,
        )
        unique = np.unique(fractions)
        if len(unique) < 2:
            continue
        circular_gaps = (np.roll(unique, -1) - unique) % 1.0
        cut_after = int(np.argmax(circular_gaps))
        origin = float(unique[(cut_after + 1) % len(unique)])
        unwrapped = (fractions - origin) % 1.0
        span = float(np.max(unwrapped) - np.min(unwrapped))
        if not np.isfinite(span) or span <= 0.0:
            continue
        array_indices = np.asarray(indices, dtype=np.int64)
        coordinate[array_indices] = (unwrapped - np.min(unwrapped)) / span
        valid[array_indices] = True
        scale_index[array_indices] = scale
    return coordinate, valid, scale_index


def build_order_coordinate_sidecar(
    candidates: Sequence[KeypointPairCandidate],
    *,
    padded_length_a: int,
    padded_length_b: int,
) -> OrderCoordinateSidecar:
    """Build a padded sidecar in the observer's exact candidate order."""

    values = tuple(candidates)
    if any(not isinstance(value, KeypointPairCandidate) for value in values):
        raise TypeError("every sidecar candidate must be KeypointPairCandidate")
    if padded_length_a < 1 or padded_length_b < 1:
        raise ValueError("padded lengths must be positive")
    count = len(values)
    coordinate_a = torch.zeros((count, padded_length_a), dtype=torch.float64)
    coordinate_b = torch.zeros((count, padded_length_b), dtype=torch.float64)
    scale_a = torch.full((count, padded_length_a), -1, dtype=torch.long)
    scale_b = torch.full((count, padded_length_b), -1, dtype=torch.long)
    valid_a = torch.zeros((count, padded_length_a), dtype=torch.bool)
    valid_b = torch.zeros((count, padded_length_b), dtype=torch.bool)
    for row, candidate in enumerate(values):
        if (
            len(candidate.tokens_a) > padded_length_a
            or len(candidate.tokens_b) > padded_length_b
        ):
            raise ValueError("candidate exceeds requested sidecar padding")
        first_coordinate, first_valid, first_scale = _unwrap_scale_tokens(
            candidate.tokens_a
        )
        second_coordinate, second_valid, second_scale = _unwrap_scale_tokens(
            candidate.tokens_b
        )
        coordinate_a[row, : len(first_coordinate)] = torch.from_numpy(first_coordinate)
        coordinate_b[row, : len(second_coordinate)] = torch.from_numpy(second_coordinate)
        scale_a[row, : len(first_scale)] = torch.from_numpy(first_scale)
        scale_b[row, : len(second_scale)] = torch.from_numpy(second_scale)
        valid_a[row, : len(first_valid)] = torch.from_numpy(first_valid.copy())
        valid_b[row, : len(second_valid)] = torch.from_numpy(second_valid.copy())
    return OrderCoordinateSidecar(
        coordinate_a=coordinate_a,
        coordinate_b=coordinate_b,
        scale_index_a=scale_a,
        scale_index_b=scale_b,
        valid_a=valid_a,
        valid_b=valid_b,
    )


def _validate_inputs(
    token_mask_a: Tensor,
    token_mask_b: Tensor,
    correspondence_mask: Tensor,
    sidecar: OrderCoordinateSidecar,
) -> tuple[int, int, int]:
    if token_mask_a.ndim != 2 or token_mask_a.dtype != torch.bool:
        raise TypeError("token_mask_a must be bool [N,La]")
    if token_mask_b.ndim != 2 or token_mask_b.dtype != torch.bool:
        raise TypeError("token_mask_b must be bool [N,Lb]")
    count, length_a = token_mask_a.shape
    if token_mask_b.shape[0] != count:
        raise ValueError("token-mask candidate counts differ")
    length_b = int(token_mask_b.shape[1])
    if (
        correspondence_mask.dtype != torch.bool
        or tuple(correspondence_mask.shape) != (count, length_a, length_b)
    ):
        raise TypeError("correspondence_mask must be bool [N,La,Lb]")
    if tuple(sidecar.coordinate_a.shape) != (count, length_a) or tuple(
        sidecar.coordinate_b.shape
    ) != (count, length_b):
        raise ValueError("order sidecar does not align with candidate tensors")
    if (
        correspondence_mask
        & ~(token_mask_a[:, :, None] & token_mask_b[:, None, :])
    ).any().item():
        raise ValueError("correspondence mask enables a padded token")
    return int(count), int(length_a), int(length_b)


def transport_pairwise_anti_order_mass(
    assignment: Tensor,
    token_mask_a: Tensor,
    token_mask_b: Tensor,
    correspondence_mask: Tensor,
    sidecar: OrderCoordinateSidecar,
) -> Tensor:
    """Return anti-concordant transported-pair mass per candidate arc."""

    count, length_a, length_b = _validate_inputs(
        token_mask_a, token_mask_b, correspondence_mask, sidecar
    )
    if (
        not assignment.is_floating_point()
        or tuple(assignment.shape) != (count, length_a, length_b)
    ):
        raise TypeError("assignment must be floating-point [N,La,Lb]")
    active_a = token_mask_a & sidecar.valid_a & correspondence_mask.any(dim=2)
    active_b = token_mask_b & sidecar.valid_b & correspondence_mask.any(dim=1)
    edge = (
        correspondence_mask
        & active_a[:, :, None]
        & active_b[:, None, :]
        & (
            sidecar.scale_index_a[:, :, None]
            == sidecar.scale_index_b[:, None, :]
        )
    )
    plan = torch.where(edge, assignment, torch.zeros_like(assignment))
    same_a = (
        sidecar.scale_index_a[:, :, None]
        == sidecar.scale_index_a[:, None, :]
    ) & active_a[:, :, None] & active_a[:, None, :]
    same_b = (
        sidecar.scale_index_b[:, :, None]
        == sidecar.scale_index_b[:, None, :]
    ) & active_b[:, :, None] & active_b[:, None, :]
    later_a = same_a & (
        sidecar.coordinate_a[:, None, :] > sidecar.coordinate_a[:, :, None]
    )
    earlier_b = same_b & (
        sidecar.coordinate_b[:, None, :] < sidecar.coordinate_b[:, :, None]
    )
    later_mass = torch.bmm(later_a.to(assignment.dtype), plan)
    anti_completion = torch.bmm(
        later_mass, earlier_b.to(assignment.dtype).transpose(1, 2)
    )
    anti_mass = (plan * anti_completion).sum(dim=(1, 2))

    upper_a = torch.triu(
        torch.ones((length_a, length_a), dtype=torch.bool, device=assignment.device),
        diagonal=1,
    )
    upper_b = torch.triu(
        torch.ones((length_b, length_b), dtype=torch.bool, device=assignment.device),
        diagonal=1,
    )
    pair_capacity_a = (same_a & upper_a[None]).sum(dim=(1, 2))
    pair_capacity_b = (same_b & upper_b[None]).sum(dim=(1, 2))
    capacity = torch.minimum(pair_capacity_a, pair_capacity_b)
    quality = anti_mass / capacity.clamp_min(1).to(assignment.dtype)
    quality = torch.where(capacity > 0, quality, torch.zeros_like(quality))
    finite = torch.isfinite(assignment).flatten(1).all(dim=1)
    return torch.where(finite, quality, torch.zeros_like(quality)).clamp(0.0, 1.0)


def exact_target_order_concordance(
    assignment_target_a: Tensor,
    token_mask_a: Tensor,
    token_mask_b: Tensor,
    correspondence_mask: Tensor,
    sidecar: OrderCoordinateSidecar,
) -> ExactTargetOrderConcordance:
    """Count anti/monotone exact-target edge pairs per candidate and scale."""

    count, length_a, length_b = _validate_inputs(
        token_mask_a, token_mask_b, correspondence_mask, sidecar
    )
    if assignment_target_a.dtype != torch.long or tuple(
        assignment_target_a.shape
    ) != (count, length_a):
        raise TypeError("assignment_target_a must be int64 [N,La]")
    rows = []
    for candidate in range(count):
        matched_a = torch.nonzero(
            assignment_target_a[candidate] >= 0, as_tuple=False
        ).flatten()
        if not matched_a.numel():
            continue
        matched_b = assignment_target_a[candidate].index_select(0, matched_a)
        if ((matched_b < 0) | (matched_b >= length_b)).any().item():
            raise ValueError("matched target index is out of range")
        if not correspondence_mask[candidate, matched_a, matched_b].all().item():
            raise ValueError("matched target uses a disabled correspondence edge")
        scales = sidecar.scale_index_a[candidate].index_select(0, matched_a)
        opposite_scales = sidecar.scale_index_b[candidate].index_select(0, matched_b)
        if not torch.equal(scales, opposite_scales):
            raise ValueError("matched target crosses order-coordinate scale blocks")
        for scale in torch.unique(scales[scales >= 0], sorted=True):
            selected = scales == scale
            index_a = matched_a[selected]
            index_b = matched_b[selected]
            edge_count = int(index_a.numel())
            first = sidecar.coordinate_a[candidate].index_select(0, index_a)
            second = sidecar.coordinate_b[candidate].index_select(0, index_b)
            upper = torch.triu(
                torch.ones(
                    (edge_count, edge_count),
                    dtype=torch.bool,
                    device=assignment_target_a.device,
                ),
                diagonal=1,
            )
            product = (first[:, None] - first[None, :]) * (
                second[:, None] - second[None, :]
            )
            anti = int(((product < 0.0) & upper).sum().item())
            mono = int(((product > 0.0) & upper).sum().item())
            tied = int(((product == 0.0) & upper).sum().item())
            rows.append((candidate, int(scale), edge_count, anti, mono, tied))
    if rows:
        values = torch.tensor(rows, dtype=torch.long, device=assignment_target_a.device)
    else:
        values = torch.empty((0, 6), dtype=torch.long, device=assignment_target_a.device)
    return ExactTargetOrderConcordance(
        candidate_index=values[:, 0],
        scale_index=values[:, 1],
        matched_edge_count=values[:, 2],
        anti_pair_count=values[:, 3],
        monotone_pair_count=values[:, 4],
        tied_pair_count=values[:, 5],
    )


__all__ = [
    "ExactTargetOrderConcordance",
    "OrderCoordinateSidecar",
    "READOUT_ORDER_COHERENT",
    "build_order_coordinate_sidecar",
    "exact_target_order_concordance",
    "transport_pairwise_anti_order_mass",
]
