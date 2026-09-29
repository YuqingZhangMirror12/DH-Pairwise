"""Exact union membership with absolute online Q; no hidden geometric veto."""
from dataclasses import dataclass
import torch

from .evidence import RecalledEvidence, _side


@dataclass(frozen=True)
class UnionEvidence(RecalledEvidence):
    directional_kernels: torch.Tensor
    union_edge_ids: torch.Tensor


def recall_union(pair, pose, config, edge_ids, edge_mask=None, block_rows=64, observation_radius_px=3.5):
    """Only the deduplicated cluster participates; Q/dustbin remain absolute.

    Other Q mass is still available through other_mass, not misrepresented as
    unmatched. Every member retains Q_ij for classification and cross-attn.
    Directional geometry still controls localization precision and residual
    features, but is not multiplied into membership/attention a second time.
    """
    if not pair.numeric_valid or pair.q.shape != (len(pair.local_a), len(pair.local_b)):
        raise ValueError('invalid compact pair')
    ids = edge_ids.to(device=pair.q.device, dtype=torch.long)
    if ids.ndim != 2 or ids.shape[1] != 2 or not len(ids):
        raise ValueError('nonempty pair-of-indices union required')
    if bool((ids < 0).any()) or bool((ids[:, 0] >= pair.q.shape[0]).any()) or bool((ids[:, 1] >= pair.q.shape[1]).any()):
        raise ValueError('union edge outside the compact full Q')
    ids = torch.unique(ids, dim=0)
    mask = torch.zeros_like(pair.q, dtype=torch.bool)
    mask[ids[:, 0], ids[:, 1]] = True
    if edge_mask is not None:
        if edge_mask.shape != mask.shape or edge_mask.dtype != torch.bool:
            raise ValueError('extension mask must name compact full-Q edges')
        mask = mask & edge_mask
    rows = [pair.compatibility(pose, config, start, min(start + block_rows, len(pair.local_a)))
            for start in range(0, len(pair.local_a), block_rows)]
    residual = torch.cat([x[0] for x in rows])
    directional = torch.cat([x[1].kernel for x in rows])
    location = torch.cat([x[1].localization_kernel for x in rows])
    normal = torch.cat([x[1].normal_px for x in rows])
    tangent = torch.cat([x[1].tangent_px for x in rows])
    reliability = torch.cat([x[1].normal_reliability for x in rows])
    membership = mask.to(pair.q.dtype)
    weights = pair.q * membership
    a = _side(pair, weights, location, residual, normal, tangent, reliability, 'a', config, observation_radius_px)
    b = _side(pair, weights.T, location.T, -residual.permute(1, 0, 2), normal.T, tangent.T,
              reliability.T, 'b', config, observation_radius_px)
    return UnionEvidence(pose, weights, membership, location, a, b, pair, directional, ids)
