"""Swap-invariant structured readout for a frozen local transport plan.

The current local matcher reduces an ordered ``La x Lb`` assignment to five
scalars.  This additive research head keeps the Exact-Sinkhorn backbone frozen
and instead treats its assignment matrix as an image, following the fine-stage
intuition in ShreddingNet.  Facing contour sides traverse a true seam in
opposite order, so the B axis is reversed before a three-layer CNN looks for a
continuous high-confidence band.

The head consumes only frozen model evidence and geometry masks.  Pair labels
and four-way direction targets live in the external fit adapter and can never
become input features here.  A/B swap invariance is architectural: the same
CNN encodes the A-by-B view and its B-by-A transform, and their embeddings are
averaged before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


TRANSPORT_MATRIX_CHANNELS = (
    "assignment",
    "affinity",
    "row_unmatched",
    "column_unmatched",
    "correspondence_allowed",
)


@dataclass(frozen=True)
class StructuredTransportReadoutOutput:
    """Per-arc frozen-baseline score plus a learned structural correction."""

    arc_logit: Tensor
    correction: Tensor
    embedding: Tensor
    continuity: Tensor
    transport_valid: Tensor

    def __post_init__(self) -> None:
        if self.arc_logit.ndim != 1 or not self.arc_logit.is_floating_point():
            raise TypeError("arc_logit must be floating-point [N]")
        shape = tuple(self.arc_logit.shape)
        for name in ("correction", "continuity"):
            value = getattr(self, name)
            if tuple(value.shape) != shape or not value.is_floating_point():
                raise TypeError(name + " must be floating-point [N]")
        if (
            self.embedding.ndim != 2
            or self.embedding.shape[0] != self.arc_logit.shape[0]
        ):
            raise ValueError("embedding must have shape [N,D]")
        if not self.embedding.is_floating_point():
            raise TypeError("embedding must be floating point")
        if (
            self.transport_valid.dtype != torch.bool
            or tuple(self.transport_valid.shape) != shape
        ):
            raise TypeError("transport_valid must be bool [N]")


def _positive_integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return value


def _validate_transport_inputs(
    assignment: Tensor,
    affinity: Tensor,
    unmatched_a: Tensor,
    unmatched_b: Tensor,
    token_mask_a: Tensor,
    token_mask_b: Tensor,
    correspondence_mask: Tensor,
    base_arc_logit: Optional[Tensor],
) -> tuple[int, int, int, Tensor, Tensor]:
    if not isinstance(assignment, Tensor) or assignment.ndim != 3:
        raise ValueError("assignment must have shape [N,La,Lb]")
    if not assignment.is_floating_point():
        raise TypeError("assignment must be floating point")
    count, length_a, length_b = assignment.shape
    if count < 1 or length_a < 1 or length_b < 1:
        raise ValueError("transport dimensions must be positive")
    for name, value, shape in (
        ("affinity", affinity, (count, length_a, length_b)),
        ("unmatched_a", unmatched_a, (count, length_a)),
        ("unmatched_b", unmatched_b, (count, length_b)),
    ):
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError(name + " must be floating point")
        if tuple(value.shape) != shape:
            raise ValueError("{} must have shape {}".format(name, shape))
        if value.device != assignment.device or value.dtype != assignment.dtype:
            raise ValueError(name + " must share assignment device and dtype")
    for name, value, shape in (
        ("token_mask_a", token_mask_a, (count, length_a)),
        ("token_mask_b", token_mask_b, (count, length_b)),
        ("correspondence_mask", correspondence_mask, (count, length_a, length_b)),
    ):
        if not isinstance(value, Tensor) or value.dtype != torch.bool:
            raise TypeError(name + " must be bool")
        if tuple(value.shape) != shape:
            raise ValueError("{} must have shape {}".format(name, shape))
        if value.device != assignment.device:
            raise ValueError(name + " must share assignment device")
    if base_arc_logit is not None:
        if (
            not isinstance(base_arc_logit, Tensor)
            or not base_arc_logit.is_floating_point()
            or tuple(base_arc_logit.shape) != (count,)
        ):
            raise TypeError("base_arc_logit must be floating-point [N]")
        if (
            base_arc_logit.device != assignment.device
            or base_arc_logit.dtype != assignment.dtype
        ):
            raise ValueError("base_arc_logit must share assignment device and dtype")

    allowed = correspondence_mask & token_mask_a[:, :, None] & token_mask_b[:, None, :]
    finite = torch.where(allowed, torch.isfinite(assignment), True).flatten(1).all(1)
    finite = finite & torch.where(allowed, torch.isfinite(affinity), True).flatten(
        1
    ).all(1)
    finite = finite & torch.where(token_mask_a, torch.isfinite(unmatched_a), True).all(
        1
    )
    finite = finite & torch.where(token_mask_b, torch.isfinite(unmatched_b), True).all(
        1
    )
    if base_arc_logit is not None:
        finite = finite & torch.isfinite(base_arc_logit)
    transport_valid = (
        finite & token_mask_a.any(1) & token_mask_b.any(1) & allowed.flatten(1).any(1)
    )
    return count, length_a, length_b, allowed, transport_valid


def _swap_raw_matrix(raw: Tensor) -> Tensor:
    """Return the raw B-by-A view while preserving channel semantics."""

    # Row/column dustbin channels exchange roles under transpose.  The other
    # channels transpose directly.  Keeping this as a deterministic tensor
    # transform makes the subsequent orbit average exactly symmetric.
    order = torch.tensor((0, 1, 3, 2, 4), dtype=torch.long, device=raw.device)
    return raw.index_select(1, order).transpose(2, 3)


def _diagonal_continuity(view: Tensor) -> Tensor:
    """Differentiable coherent-band prior, normalized for equal plan mass.

    Both diagonal orientations are retained.  The deterministic B-axis
    reversal maps the expected anti-order Dunhuang seam to the main diagonal,
    while the second orientation prevents the prior from becoming brittle to
    an upstream contour traversal convention.
    """

    assignment = view[:, 0].clamp_min(0.0)
    allowed = view[:, 4].clamp(0.0, 1.0)
    assignment = assignment * allowed
    denominator = assignment.square().sum(dim=(1, 2)).clamp_min(1e-8)
    main = (
        assignment[:, :-1, :-1]
        * assignment[:, 1:, 1:]
        * allowed[:, :-1, :-1]
        * allowed[:, 1:, 1:]
    ).sum(dim=(1, 2))
    anti = (
        assignment[:, :-1, 1:]
        * assignment[:, 1:, :-1]
        * allowed[:, :-1, 1:]
        * allowed[:, 1:, :-1]
    ).sum(dim=(1, 2))
    return torch.maximum(main, anti) / denominator


class FrozenTransportMatrixReadout(nn.Module):
    """Three-layer matrix CNN over detached Exact-Sinkhorn evidence.

    Variable-length arc matrices are first stripped of padded rows/columns and
    resized to a fixed square.  The correspondence channel preserves disjoint
    scale blocks, so multi-scale candidate structure remains observable.  The
    output is a residual correction to the frozen arc logit.
    """

    def __init__(
        self,
        *,
        grid_size: int = 32,
        widths: Tuple[int, int, int] = (12, 20, 28),
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.grid_size = _positive_integer(grid_size, "grid_size", minimum=4)
        if len(widths) != 3 or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in widths
        ):
            raise ValueError("widths must contain exactly three positive integers")
        self.embedding_dim = _positive_integer(embedding_dim, "embedding_dim")
        layers = []
        previous = len(TRANSPORT_MATRIX_CHANNELS)
        for width in widths:
            layers.extend(
                (
                    nn.Conv2d(previous, width, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(1, width),
                    nn.SiLU(inplace=True),
                )
            )
            previous = width
        self.matrix_cnn = nn.Sequential(*layers)
        self.embedding_head = nn.Sequential(
            nn.Linear(2 * previous + 1, self.embedding_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(self.embedding_dim),
        )
        self.correction_head = nn.Linear(self.embedding_dim, 1)
        # Keep the source-motivated continuity prior dominant at initialization
        # while retaining non-zero gradients through every learned CNN channel.
        nn.init.normal_(self.correction_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.correction_head.bias)
        self.continuity_log_weight = nn.Parameter(
            torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float32)
        )

    def _raw_matrix(
        self,
        assignment: Tensor,
        affinity: Tensor,
        unmatched_a: Tensor,
        unmatched_b: Tensor,
        token_mask_a: Tensor,
        token_mask_b: Tensor,
        allowed: Tensor,
    ) -> Tensor:
        rows = []
        for index in range(assignment.shape[0]):
            index_a = torch.nonzero(token_mask_a[index], as_tuple=False).flatten()
            index_b = torch.nonzero(token_mask_b[index], as_tuple=False).flatten()
            if index_a.numel() == 0 or index_b.numel() == 0:
                rows.append(
                    assignment.new_zeros(
                        (
                            1,
                            len(TRANSPORT_MATRIX_CHANNELS),
                            self.grid_size,
                            self.grid_size,
                        )
                    )
                )
                continue
            plan = assignment[index].index_select(0, index_a).index_select(1, index_b)
            score = affinity[index].index_select(0, index_a).index_select(1, index_b)
            edge = allowed[index].index_select(0, index_a).index_select(1, index_b)
            row_dustbin = unmatched_a[index].index_select(0, index_a)
            col_dustbin = unmatched_b[index].index_select(0, index_b)
            safe_plan = torch.where(
                edge, torch.nan_to_num(plan), torch.zeros_like(plan)
            )
            safe_score = torch.where(
                edge, torch.nan_to_num(score), torch.zeros_like(score)
            )
            row_plane = torch.nan_to_num(row_dustbin)[:, None].expand_as(safe_plan)
            col_plane = torch.nan_to_num(col_dustbin)[None, :].expand_as(safe_plan)
            continuous = torch.stack(
                (safe_plan, safe_score, row_plane, col_plane), dim=0
            )[None]
            resized = F.interpolate(
                continuous,
                size=(self.grid_size, self.grid_size),
                mode="bilinear",
                align_corners=False,
            )
            resized_edge = F.interpolate(
                edge.to(assignment.dtype)[None, None],
                size=(self.grid_size, self.grid_size),
                mode="nearest",
            )
            # Interpolation cannot leak plan/affinity evidence across separate
            # scale blocks because those planes are gated again after resize.
            resized = torch.cat((resized[:, :2] * resized_edge, resized[:, 2:]), dim=1)
            rows.append(torch.cat((resized, resized_edge), dim=1))
        return torch.cat(rows, dim=0)

    def _encode_view(self, view: Tensor) -> Tensor:
        feature_map = self.matrix_cnn(view)
        mean = feature_map.mean(dim=(2, 3))
        maximum = feature_map.amax(dim=(2, 3))
        return torch.cat((mean, maximum), dim=1)

    def forward(
        self,
        assignment: Tensor,
        affinity: Tensor,
        unmatched_a: Tensor,
        unmatched_b: Tensor,
        token_mask_a: Tensor,
        token_mask_b: Tensor,
        correspondence_mask: Tensor,
        *,
        base_arc_logit: Optional[Tensor] = None,
    ) -> StructuredTransportReadoutOutput:
        """Score frozen arc evidence without consuming pair/direction targets."""

        count, _, _, allowed, transport_valid = _validate_transport_inputs(
            assignment,
            affinity,
            unmatched_a,
            unmatched_b,
            token_mask_a,
            token_mask_b,
            correspondence_mask,
            base_arc_logit,
        )
        raw = self._raw_matrix(
            assignment,
            affinity,
            unmatched_a,
            unmatched_b,
            token_mask_a,
            token_mask_b,
            allowed,
        )
        # Reverse the facing-side traversal so an expected anti-order seam is
        # presented to the CNN as a continuous main-diagonal band.
        view_ab = torch.flip(raw, dims=(3,))
        view_ba = torch.flip(_swap_raw_matrix(raw), dims=(3,))
        pooled = 0.5 * (self._encode_view(view_ab) + self._encode_view(view_ba))
        continuity = 0.5 * (
            _diagonal_continuity(view_ab) + _diagonal_continuity(view_ba)
        )
        embedding = self.embedding_head(torch.cat((pooled, continuity[:, None]), dim=1))
        learned = self.correction_head(embedding).squeeze(1)
        continuity_weight = F.softplus(
            self.continuity_log_weight.to(dtype=assignment.dtype)
        )
        correction = learned + continuity_weight * continuity
        correction = torch.where(
            transport_valid, correction, torch.zeros_like(correction)
        )
        baseline = (
            assignment.new_zeros((count,)) if base_arc_logit is None else base_arc_logit
        )
        arc_logit = baseline + correction
        embedding = torch.where(
            transport_valid[:, None], embedding, torch.zeros_like(embedding)
        )
        continuity = torch.where(
            transport_valid, continuity, torch.zeros_like(continuity)
        )
        return StructuredTransportReadoutOutput(
            arc_logit=arc_logit,
            correction=correction,
            embedding=embedding,
            continuity=continuity,
            transport_valid=transport_valid,
        )


__all__ = [
    "FrozenTransportMatrixReadout",
    "StructuredTransportReadoutOutput",
    "TRANSPORT_MATRIX_CHANNELS",
]
