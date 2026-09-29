"""Valid-only contour geometry. Storage order is never canonicalized here.

Arc cells are quadrature weights, NOT evidence that an interval is a seam.
The consensus evidence module must separately determine observed support.
Coordinates and translations are row/column: p_b ~= p_a + t_a_to_b.
"""
from dataclasses import dataclass

import torch
from torch import Tensor


def deterministic_prefix_sum(values: Tensor) -> Tensor:
    """Fixed-tree inclusive sum on the last axis, including strict CUDA mode.

    The deployed PyTorch rejects floating CUDA cumsum under deterministic
    algorithms. Do not turn that safeguard off for contour bookkeeping.
    """
    result = values.clone()
    offset = 1
    while offset < values.shape[-1]:
        result = torch.cat((result[..., :offset],
                            result[..., offset:] + result[..., :-offset]), -1)
        offset *= 2
    return result


@dataclass(frozen=True)
class ContourGeometry:
    points: Tensor
    valid: Tensor
    compact_to_original: Tensor
    original_to_compact: Tensor
    arc_px: Tensor
    next_step_px: Tensor
    cell_px: Tensor
    perimeter_px: Tensor
    tangent_rc: Tensor
    outward_normal_rc: Tensor
    normal_reliability: Tensor
    counts: Tensor

    def gather(self, values: Tensor) -> Tensor:
        """Gather an original-storage tensor; invalid contents may be NaN."""
        if values.shape[:2] != self.original_to_compact.shape:
            raise ValueError('values must use original contour storage')
        index = self.compact_to_original.clamp_min(0)
        tail = values.shape[2:]
        for _ in tail:
            index = index.unsqueeze(-1)
        index = index.expand(*self.valid.shape, *tail)
        value = values.gather(1, index)
        mask = self.valid.reshape(*self.valid.shape, *([1] * len(tail)))
        return torch.where(mask, value, torch.zeros_like(value))

    def scatter(self, values: Tensor) -> Tensor:
        """Scatter back without allowing padded indices to overwrite index0."""
        if values.shape[:2] != self.valid.shape:
            raise ValueError('values must use compact contour storage')
        result = values.new_zeros(*self.original_to_compact.shape, *values.shape[2:])
        for b in range(len(result)):
            n = int(self.counts[b])
            ids = self.compact_to_original[b, :n]
            result[b].index_copy_(0, ids, values[b, :n])
        return result


@torch.no_grad()
def compact_contour(points: Tensor, valid: Tensor) -> ContourGeometry:
    """Remove padding and close each valid cycle, retaining caller order.

    Consecutive duplicate locations are legal diagnostic inputs: zero-length
    steps carry no arc length. They are not extra independent observations.
    Degenerate contours have unreliable normals, not invented directions.
    """
    if points.ndim != 3 or points.shape[-1] != 2 or not points.is_floating_point():
        raise ValueError('points must be floating [B,N,2]')
    if valid.dtype != torch.bool or valid.shape != points.shape[:2]:
        raise ValueError('valid must be bool [B,N]')
    if valid.device != points.device:
        raise ValueError('points and valid devices differ')
    if not torch.isfinite(points[valid]).all():
        raise ValueError('VALID points must be finite')
    counts = valid.sum(1)
    width = max(1, int(counts.max())) if len(counts) else 1
    batch = len(points)
    index = torch.full((batch, width), -1, dtype=torch.long, device=points.device)
    inverse = torch.full(valid.shape, -1, dtype=torch.long, device=points.device)
    mask = torch.arange(width, device=points.device)[None] < counts[:, None]
    p = points.new_zeros(batch, width, 2)
    arc = points.new_zeros(batch, width)
    step = torch.zeros_like(arc)
    cell = torch.zeros_like(arc)
    perimeter = points.new_zeros(batch)
    tangent = torch.zeros_like(p)
    normal = torch.zeros_like(p)
    reliability = torch.zeros_like(arc)
    for b, n in enumerate(counts.tolist()):
        if not n:
            continue
        ids = valid[b].nonzero(as_tuple=False).flatten()
        q = points[b, ids]
        index[b, :n] = ids
        inverse[b, ids] = torch.arange(n, device=points.device)
        p[b, :n] = q
        edges = q.roll(-1, 0) - q
        lengths = edges.norm(dim=-1)
        step[b, :n] = lengths
        cell[b, :n] = .5 * (lengths + lengths.roll(1))
        perimeter[b] = lengths.sum()
        chord = q.roll(-1, 0) - q.roll(1, 0)
        chord_length = chord.norm(dim=-1)
        tangent[b, :n] = chord / chord_length.clamp_min(1e-12)[:, None]
        # For rc rather than xy coordinates, use the signed rc polygon area.
        area_rc = (q[:, 0] * q.roll(-1, 0)[:, 1] - q[:, 1] * q.roll(-1, 0)[:, 0]).sum()
        turn = torch.stack((tangent[b, :n, 1], -tangent[b, :n, 0]), -1)
        normal[b, :n] = turn * area_rc.sign()
        reliability[b, :n] = (chord_length / (lengths + lengths.roll(1)).clamp_min(1e-12)).clamp(0, 1)
        reliability[b, :n] *= (area_rc.abs() > 1e-8).to(points.dtype)
    inclusive = deterministic_prefix_sum(step)
    arc = torch.where(mask, torch.cat((step.new_zeros(batch, 1), inclusive[:, :-1]), 1), torch.zeros_like(step))
    return ContourGeometry(p, mask, index, inverse, arc, step, cell, perimeter,
                           tangent, normal, reliability, counts)


def compact_matrix(values: Tensor, first: ContourGeometry, second: ContourGeometry) -> Tensor:
    if values.shape != (len(first.valid), first.original_to_compact.shape[1],
                        second.original_to_compact.shape[1]):
        raise ValueError('matrix must use original A/B storage')
    rows = first.compact_to_original.clamp_min(0)
    cols = second.compact_to_original.clamp_min(0)
    result = values.gather(1, rows[:, :, None].expand(-1, -1, values.shape[2]))
    result = result.gather(2, cols[:, None].expand(-1, rows.shape[1], -1))
    keep = first.valid[:, :, None] & second.valid[:, None, :]
    return torch.where(keep, result, torch.zeros_like(result))


def cyclic_delta_px(first: Tensor, second: Tensor, perimeter: Tensor) -> Tensor:
    return torch.remainder(first - second + perimeter / 2, perimeter.clamp_min(1e-12)) - perimeter / 2


def pair_frame(normal_a: Tensor, normal_b: Tensor, reliability_a: Tensor,
               reliability_b: Tensor):
    """A-outward frame: positive residual normal means a material gap.

    A/B swap negates both the residual and frame, preserving signed components.
    Unreliable directions are flagged rather than assigned an arbitrary sign.
    Inputs may broadcast over all Q edges; no GT enters this function.
    """
    delta = normal_a - normal_b
    norm = delta.norm(dim=-1)
    normal = delta / norm.clamp_min(1e-12)[..., None]
    agreement = (.5 * (1 - (normal_a * normal_b).sum(-1))).clamp(0, 1)
    reliability = torch.minimum(reliability_a, reliability_b) * agreement
    reliability = reliability * (norm > 1e-8).to(reliability.dtype)
    tangent = torch.stack((-normal[..., 1], normal[..., 0]), -1)
    return normal, tangent, reliability
