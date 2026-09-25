"""Canonical valid closed contours, explicit original/compact maps and arc units."""
from types import SimpleNamespace
import torch


def clean(x, valid):
    return torch.where(valid[..., None], x, torch.zeros_like(x))


@torch.no_grad()
def compact(points, valid):
    counts = valid.sum(1)
    width = max(1, int(counts.max()))
    batch, original_width = valid.shape
    index = torch.zeros(batch, width, dtype=torch.long, device=points.device)
    inverse = torch.full(valid.shape, -1, dtype=torch.long, device=points.device)
    mask = torch.arange(width, device=points.device)[None] < counts[:, None]
    for b in range(batch):
        ids = torch.nonzero(valid[b], as_tuple=False).flatten()
        if not len(ids):
            continue
        p = points[b, ids]
        if not torch.isfinite(p).all():
            raise ValueError('nonfinite VALID contour coordinate')
        if len(ids) < 3:
            raise ValueError('a nonempty closed contour needs at least3 distinct points')
        area = (p[:, 0]*p.roll(-1, 0)[:, 1]-p[:, 1]*p.roll(-1, 0)[:, 0]).sum()
        if area.abs() < 1e-7:
            raise ValueError('degenerate contour')
        if area < 0:
            ids = ids.flip(0)
            p = p.flip(0)
        # Geometric origin, not caller storage origin. Column breaks row ties.
        candidates = torch.nonzero(p[:, 0] == p[:, 0].min()).flatten()
        start = candidates[p[candidates, 1].argmin()]
        ids = ids.roll(-int(start))
        p = points[b, ids]
        if ((p-p.roll(-1, 0)).square().sum(1) < 1e-10).any():
            raise ValueError('duplicate adjacent contour samples; fix source, do not duplicate tokens')
        index[b, :len(ids)] = ids
        inverse[b, ids] = torch.arange(len(ids), device=points.device)
    p = clean(points.gather(1, index[..., None].expand(-1, -1, 2)), mask)
    arclength = points.new_zeros(batch, width)
    cell = points.new_zeros(batch, width)
    perimeter = points.new_ones(batch)
    normals = points.new_zeros(batch, width, 2)
    for b, n0 in enumerate(counts.tolist()):
        if not n0:
            continue
        q = p[b, :n0]
        steps = (q.roll(-1, 0)-q).norm(dim=1)
        arclength[b, :n0] = torch.cat((steps.new_zeros(1), steps.cumsum(0)[:-1]))
        perimeter[b] = steps.sum()
        cell[b, :n0] = .5*(steps+steps.roll(1))
        tangent = q.roll(-1, 0)-q.roll(1, 0)
        normal = torch.stack((tangent[:, 1], -tangent[:, 0]), -1)
        normals[b, :n0] = normal/normal.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return SimpleNamespace(points=p, valid=mask, index=index, inverse=inverse,
                           arc=arclength, cell=cell, perimeter=perimeter,
                           normals=normals, counts=counts, original_width=original_width)


def remap_target(target, own, other):
    source = target.gather(1, own.index)
    mapped = other.inverse.gather(1, source.clamp_min(0))
    if ((source >= 0) & (mapped < 0) & own.valid).any():
        raise ValueError('GT correspondence points into padding')
    return torch.where(own.valid, torch.where(source >= 0, mapped, source), -2)


def cyclic_delta(a, b, perimeter):
    return torch.remainder(a-b+perimeter/2, perimeter)-perimeter/2


def masked_mean(x, valid, dim=1):
    return clean(x, valid).sum(dim)/valid.sum(dim).clamp_min(1)[..., None]
