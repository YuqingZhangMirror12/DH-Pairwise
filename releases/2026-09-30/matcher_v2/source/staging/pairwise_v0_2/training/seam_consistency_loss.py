"""Training-only, local joint-transition supervision for partial contour matches.

The original exact-cell/dustbin NLL must remain in the training objective. This
auxiliary is *not* ``-log(P[i,j] * P[k,l])`` (which would just repeat that NLL):
it sums a non-separable reverse-path kernel over small GT-anchored arc windows.
No displacement average, predicted argmax, or whole-contour alignment is used.

For each adjacent GT pair (i,j),(k,l), A advances and B retreats in canonical
CCW order. For candidate u near j and v near l, let d be the GT reverse arc
step, and d_uv the candidate reverse step. Then

    C_uv = exp(-0.5 * ((d_uv-d)/sigma)**2) * 1[0 < d_uv <= perimeter/2]
    T_e  = sum_uv C_uv P[i,u] P[k,v]
    L_e  = -0.5 log((T_e + epsilon)/(1 + epsilon)).

The mirror transition uses P[u,j] P[v,l] and forward A steps. Average both
directions and all edges *within each sample*, then average eligible samples.
Distances and windows are in pixels, not token indices, so the same settings
apply at 512/1024 or unequal contour sizes. Default external multiplier: 0.1.
Exact-cell NLL anchors the precise GT mode; this loss alone deliberately allows
a small common shift inside the GT corridor. It penalizes dustbin mass and
incoherent multimodal products without soft-mean cancellation.

GT belongs only in ``build_seam_consistency_targets`` on the training target
side. Its result must never be passed to the model encoder or test decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


DEFAULT_SEAM_LOSS_WEIGHT = 0.1


@dataclass(frozen=True)
class SeamConsistencyConfig:
    max_gap_px: float = 24.0
    arc_mismatch_absolute_px: float = 4.0
    arc_mismatch_relative: float = 0.5
    candidate_radius_px: float = 6.0
    transition_sigma_px: float = 2.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(name + " must be finite")
            if float(value) < 0.0:
                raise ValueError(name + " must be non-negative")
        if min(self.max_gap_px, self.transition_sigma_px, self.epsilon) <= 0.0:
            raise ValueError("max_gap_px, transition_sigma_px and epsilon must be positive")
        if self.epsilon >= 1.0:
            raise ValueError("epsilon must be less than one")


@dataclass(frozen=True)
class SeamConsistencyTargets:
    """Sparse GT-only transition plan; indices address flattened [B,Na,Nb]."""

    assignment_shape: Tuple[int, int, int]
    first_index: Tensor
    second_index: Tensor
    kernel: Tensor
    transition_edge: Tensor
    edge_sample: Tensor
    edge_direction: Tensor  # 0: row distributions / reverse B; 1: columns / forward A
    epsilon: float

    @property
    def edge_count(self) -> int:
        """Directed edges: each accepted physical adjacency contributes two."""

        return self.edge_sample.numel()

    @property
    def adjacency_count(self) -> int:
        return self.edge_count // 2

    def to(self, device: torch.device) -> "SeamConsistencyTargets":
        return replace(
            self,
            **{
                name: getattr(self, name).to(device=device)
                for name in (
                    "first_index", "second_index", "kernel", "transition_edge",
                    "edge_sample", "edge_direction",
                )
            },
        )


@dataclass(frozen=True)
class SeamConsistencyLoss:
    total: Tensor
    per_sample: Tensor
    active_sample: Tensor
    edge_count_per_sample: Tensor
    transition_mass: Tensor


def _numpy(value: object) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class _Arc:
    indices: np.ndarray
    coordinate: np.ndarray
    perimeter: float


def _closed_arc(points: np.ndarray, valid: np.ndarray, orientation: str) -> _Arc:
    indices = np.flatnonzero(valid)
    coordinate = np.full(len(points), np.nan, dtype=np.float64)
    if len(indices) < 3:
        return _Arc(indices, coordinate, 0.0)
    selected = np.asarray(points[indices], dtype=np.float64)
    if not np.isfinite(selected).all():
        raise ValueError("valid contour points must be finite")
    if orientation == "auto":
        x, y = selected[:, 1], -selected[:, 0]  # RC -> Cartesian
        area = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
        if abs(float(area)) <= 1e-9:
            raise ValueError("degenerate contour winding: specify orientation explicitly")
        clockwise = area < 0.0
    else:
        clockwise = orientation == "cw"
    if clockwise:
        indices = indices[::-1].copy()
        selected = selected[::-1]
    lengths = np.linalg.norm(np.roll(selected, -1, axis=0) - selected, axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 0.0:
        raise ValueError("contour must have positive perimeter")
    coordinate[indices] = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    return _Arc(indices, coordinate, perimeter)


def _window(arc: _Arc, anchor: int, radius: float) -> np.ndarray:
    offset = (
        (arc.coordinate[arc.indices] - arc.coordinate[anchor] + arc.perimeter / 2.0)
        % arc.perimeter
    ) - arc.perimeter / 2.0
    return arc.indices[np.abs(offset) <= radius + 1e-7]


def build_seam_consistency_targets(
    points_rc_a: object,
    points_rc_b: object,
    contour_valid_a: object,
    contour_valid_b: object,
    target_a: object,
    target_b: object,
    config: SeamConsistencyConfig = SeamConsistencyConfig(),
    *,
    orientation_a: str = "auto",
    orientation_b: str = "auto",
) -> SeamConsistencyTargets:
    """Build a CPU target plan from NumPy arrays or detached tensors.

    Shapes are [B,Na,2], [B,Nb,2], [B,Na/Nb], with reciprocal int targets:
    >=0 opposite token, -1 unmatched, -2 ignored. No label/translation is needed.
    Contours must be ordered, closed, simple outer contours. ``auto`` detects
    winding by signed area; explicit ``ccw``/``cw`` describe the INPUT order.
    Reversing either input and remapping GT indices is therefore supported.

    Only consecutive GT correspondences with short, approximately equal
    physical forward-A / reverse-B arcs become edges. Small GT missing spans
    are allowed; large gaps or same-direction GT pairs break the seam. Invalid
    tokens are excluded; arcs over missing coordinates use the polygon chord.
    This is intended for the loader's complete contour plus padding, not for
    reconstructing unknown long arcs. Plans can be cached in a bounded CPU LRU;
    avoid retaining expanded plans for every training pair unnecessarily.
    """

    if orientation_a not in {"auto", "ccw", "cw"} or orientation_b not in {"auto", "ccw", "cw"}:
        raise ValueError("orientation must be auto, ccw or cw")
    pa, pb, va, vb, ta, tb = map(
        _numpy,
        (points_rc_a, points_rc_b, contour_valid_a, contour_valid_b, target_a, target_b),
    )
    if pa.ndim != 3 or pb.ndim != 3 or pa.shape[2:] != (2,) or pb.shape[2:] != (2,):
        raise ValueError("points must have shapes [B,Na,2] and [B,Nb,2]")
    batch, na, _ = pa.shape
    if pb.shape[0] != batch:
        raise ValueError("contour batch sizes disagree")
    nb = pb.shape[1]
    if va.shape != (batch, na) or vb.shape != (batch, nb) or va.dtype != np.bool_ or vb.dtype != np.bool_:
        raise ValueError("valid masks must be bool [B,N]")
    if ta.shape != va.shape or tb.shape != vb.shape or ta.dtype.kind not in "iu" or tb.dtype.kind not in "iu":
        raise ValueError("targets must be integer [B,N]")
    if np.any((ta < -2) | (ta >= nb)) or np.any((tb < -2) | (tb >= na)):
        raise ValueError("target indices out of range")
    first, second, kernels, transition_edges = [], [], [], []
    edge_samples, edge_directions = [], []

    def append_transition(b: int, direction: int, first_indices: np.ndarray,
                          second_indices: np.ndarray, weights: np.ndarray) -> None:
        keep = weights > 0.0
        if not np.any(keep):
            return
        edge = len(edge_samples)
        first.append(np.broadcast_to(first_indices, weights.shape)[keep])
        second.append(np.broadcast_to(second_indices, weights.shape)[keep])
        kernels.append(weights[keep])
        transition_edges.append(np.full(int(keep.sum()), edge, dtype=np.int64))
        edge_samples.append(b)
        edge_directions.append(direction)

    for b in range(batch):
        matched = np.flatnonzero(ta[b] >= 0)
        opposite = ta[b, matched].astype(np.int64)
        if np.any(~va[b, matched]) or np.any(~vb[b, opposite]) or np.any(tb[b, opposite] != matched):
            raise ValueError("GT matches must be valid and reciprocal")
        matched_b = np.flatnonzero(tb[b] >= 0)
        if np.any(~vb[b, matched_b]) or np.any(ta[b, tb[b, matched_b]] != matched_b):
            raise ValueError("GT matches must be reciprocal in both directions")
        if len(matched) < 2:
            continue
        aa = _closed_arc(pa[b], va[b], orientation_a)
        ab = _closed_arc(pb[b], vb[b], orientation_b)
        if min(aa.perimeter, ab.perimeter) <= 0.0:
            continue
        matched = matched[np.argsort(aa.coordinate[matched], kind="stable")]
        offset = b * na * nb
        for i_value, k_value in zip(matched, np.roll(matched, -1)):
            i, k = int(i_value), int(k_value)
            j, l = int(ta[b, i]), int(ta[b, k])
            da = float((aa.coordinate[k] - aa.coordinate[i]) % aa.perimeter)
            db = float((ab.coordinate[j] - ab.coordinate[l]) % ab.perimeter)
            if not (0.0 < da <= min(config.max_gap_px, aa.perimeter / 2.0)
                    and 0.0 < db <= min(config.max_gap_px, ab.perimeter / 2.0)):
                continue
            if abs(da - db) > config.arc_mismatch_absolute_px + config.arc_mismatch_relative * max(da, db):
                continue
            for direction, arc, anchor1, anchor2, gt_step in (
                (0, ab, j, l, db), (1, aa, i, k, da),
            ):
                u = _window(arc, anchor1, config.candidate_radius_px)
                v = _window(arc, anchor2, config.candidate_radius_px)
                difference = arc.coordinate[u, None] - arc.coordinate[v][None, :]
                step = (difference if direction == 0 else -difference) % arc.perimeter
                weights = np.exp(-0.5 * ((step - gt_step) / config.transition_sigma_px) ** 2)
                weights *= (step > 1e-8) & (step <= arc.perimeter / 2.0)
                if direction == 0:
                    first_indices = offset + i * nb + u[:, None]
                    second_indices = offset + k * nb + v[None, :]
                else:
                    first_indices = offset + u[:, None] * nb + j
                    second_indices = offset + v[None, :] * nb + l
                append_transition(b, direction, first_indices, second_indices, weights)

    def packed(parts: list, dtype: torch.dtype) -> Tensor:
        values = np.concatenate(parts) if parts else np.empty(0)
        return torch.as_tensor(values, dtype=dtype)

    return SeamConsistencyTargets(
        assignment_shape=(batch, na, nb),
        first_index=packed(first, torch.long),
        second_index=packed(second, torch.long),
        kernel=packed(kernels, torch.float32),
        transition_edge=packed(transition_edges, torch.long),
        edge_sample=torch.tensor(edge_samples, dtype=torch.long),
        edge_direction=torch.tensor(edge_directions, dtype=torch.long),
        epsilon=config.epsilon,
    )


def compute_seam_consistency_loss(
    assignment: Tensor,
    targets: SeamConsistencyTargets,
    training_valid: Tensor = None,
) -> SeamConsistencyLoss:
    """Differentiable auxiliary for post-Sinkhorn sub-stochastic real matches.

    ``loss = original_loss + DEFAULT_SEAM_LOSS_WEIGHT * result.total``.
    Move a CPU plan with ``targets.to(assignment.device)`` before this call.
    Target construction is detached, but both endpoint probabilities receive
    gradients. FP16/BF16 products/logs accumulate in FP32. Empty/no-positive
    batches return a graph-connected zero. Counts stay on device.
    """

    if not assignment.is_floating_point() or tuple(assignment.shape) != targets.assignment_shape:
        raise ValueError("assignment must be floating-point with the target plan's [B,Na,Nb] shape")
    if targets.first_index.device != assignment.device:
        raise ValueError("move targets to the assignment device before computing loss")
    batch = assignment.shape[0]
    if training_valid is None:
        training_valid = torch.ones(batch, dtype=torch.bool, device=assignment.device)
    elif training_valid.shape != (batch,) or training_valid.dtype != torch.bool or training_valid.device != assignment.device:
        raise ValueError("training_valid must be bool [B] on the assignment device")
    dtype = torch.float64 if assignment.dtype == torch.float64 else torch.float32
    flat = assignment.to(dtype=dtype).reshape(-1)
    values = flat[targets.first_index] * flat[targets.second_index] * targets.kernel.to(dtype=dtype)
    mass = torch.zeros(targets.edge_count, dtype=dtype, device=assignment.device).index_add(
        0, targets.transition_edge, values,
    )
    edge_loss = -0.5 * torch.log((mass.clamp(max=1.0) + targets.epsilon) / (1.0 + targets.epsilon))
    counts = torch.zeros(batch, dtype=dtype, device=assignment.device).index_add(
        0, targets.edge_sample, torch.ones_like(edge_loss),
    )
    sums = torch.zeros(batch, dtype=dtype, device=assignment.device).index_add(
        0, targets.edge_sample, edge_loss,
    )
    per_sample = sums / counts.clamp_min(1.0)
    active = (counts > 0.0) & training_valid
    total = (per_sample * active.to(dtype)).sum() / active.sum().clamp_min(1)
    # The connected zero also makes the all-empty case safe for backward().
    total = total + flat.sum() * 0.0
    return SeamConsistencyLoss(total, per_sample, active, counts, mass)


__all__ = [
    "DEFAULT_SEAM_LOSS_WEIGHT", "SeamConsistencyConfig", "SeamConsistencyTargets",
    "SeamConsistencyLoss", "build_seam_consistency_targets", "compute_seam_consistency_loss",
]
