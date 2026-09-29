"""Independent candidate heads; new weights must be trained, not transplanted.

Patch sum control: log(1 + sum_e Q_e * arc_e * sigmoid(phi(edge_e))).
No channelwise maximum, no division by cluster mass, no explicit conflict or
overlap penalty. Removing overlap from inputs does NOT disable the builder's
physical-interpenetration constraint. Not a ResNet or an Attention model.
"""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class EvidenceReadout:
    logit: torch.Tensor
    score: torch.Tensor
    inputs: object
    pooled_features: object
    contributions: object


def sum_evidence(mass_weights, gates):
    """Nonnegative bounded per-edge contribution; no softmax normalization."""
    if gates.ndim != 2 or len(gates) != len(mass_weights):
        raise ValueError('one feature-gate vector per unique correspondence required')
    contributions = mass_weights[:, None] * gates
    return torch.log1p(contributions.sum(0)), contributions


class EvidenceClusterHead(nn.Module):
    def __init__(self, variant='patch_sum', feature_dim=96, remove_overlap=True,
                 input_builder=None):
        super().__init__()
        if variant not in ('patch_sum', 'patch_mean', 'patch_meanmax', 'stats'):
            raise ValueError('unknown controlled architecture')
        # Lazy import resolves against the explicitly bound training snapshot.
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.binary_scorer_v1.head import inputs_for_cluster, SCALAR_NAMES
        self.input_builder = input_builder or inputs_for_cluster
        self.variant = variant
        self.feature_dim = feature_dim
        self.remove_overlap = remove_overlap
        self.scalar_names = tuple(n for n in SCALAR_NAMES
                                  if not (remove_overlap and n == 'overlap_min_area'))
        self.scalar_indices = tuple(SCALAR_NAMES.index(n) for n in self.scalar_names)
        self.edge_mlp = (None if variant == 'stats' else nn.Sequential(
            nn.Linear(4 * feature_dim + 8, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU()))
        pooled_width = dict(stats=0, patch_sum=32, patch_mean=32, patch_meanmax=64)[variant]
        self.cluster_mlp = nn.Sequential(nn.Linear(pooled_width + len(self.scalar_names), 64),
            nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    @property
    def bias(self):
        return self.cluster_mlp[-1].bias[0]

    def forward(self, pair, proposal):
        x = self.input_builder(pair, proposal, self.variant != 'stats')
        pooled = contributions = None
        if self.edge_mlp is not None:
            h = self.edge_mlp(torch.cat((x.patch_context, x.edge_geometry), -1))
            if self.variant == 'patch_sum':
                pooled, contributions = sum_evidence(x.mass_weights, h.sigmoid())
            elif self.variant == 'patch_mean':
                contributions = h * x.normalized_weights[:, None]
                pooled = contributions.sum(0)
            else:
                pooled = torch.cat(((h * x.normalized_weights[:, None]).sum(0), h.max(0).values))
        statistics = x.statistics[list(self.scalar_indices)]
        value = statistics if pooled is None else torch.cat((pooled, statistics))
        logit = self.cluster_mlp(value).squeeze(-1)
        return EvidenceReadout(logit, logit.sigmoid(), x, pooled, contributions)


def raw_q_baselines(pair, proposal):
    """Uncalibrated ranking measures, not probabilities or a trained Scorer."""
    ids = torch.unique(proposal.edge_ids.to(device=pair.q.device, dtype=torch.long), dim=0)
    if not len(ids):
        raise ValueError('no-candidate is not a fabricated score')
    q = pair.q[ids[:, 0], ids[:, 1]]
    return dict(unique_count=len(ids), sum_q=q.sum(), max_q=q.max())


def ablate_frozen_binary_overlap(head, cluster_input, scalar_names):
    """Inference-only diagnostic. Zeroing an old feature is not retraining."""
    x = cluster_input.clone()
    index = len(x) - len(scalar_names) + scalar_names.index('overlap_min_area')
    x[index] = 0.
    return head.cluster_mlp(x).squeeze(-1)
