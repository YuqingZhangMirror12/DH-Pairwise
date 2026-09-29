"""Two small whole-cluster logits, without local support/conflict classifiers.

The exact (i,j) union is deduplicated before reading online Q. Conditional
pooling does not discard absolute mass: count, sum(Q) and sum(Q*arc) are separate
inputs. This module has no GT, second Sinkhorn, Attention, or fixed penalties.
"""
from dataclasses import dataclass
import torch
from torch import nn
from ..s7_consensus_v1.evidence import observed_arc_cells, material_overlap


SCALAR_NAMES = ('log_count', 'log_sum_q', 'log_mass_px', 'mean_q', 'max_q',
    'effective_count_fraction', 'coverage_mean', 'coverage_min',
    'residual_mean_over20', 'residual_rms_over20', 'residual_max_over20',
    'unmatched_mean', 'outside_mass_mean', 'overlap_min_area',
    'original_pose_diameter_over16', 'member_count_log')


@dataclass(frozen=True)
class ClusterInputs:
    edge_ids: torch.Tensor
    q: torch.Tensor
    arc_px: torch.Tensor
    mass_weights: torch.Tensor
    normalized_weights: torch.Tensor
    edge_geometry: torch.Tensor
    statistics: torch.Tensor
    patch_context: object
    pose: torch.Tensor


@dataclass(frozen=True)
class BinaryReadout:
    logit: torch.Tensor
    score: torch.Tensor
    inputs: ClusterInputs
    pooled_features: object


def normalized(weights):
    total = weights.double().sum()
    if not bool(torch.isfinite(total)) or not bool(total > 0):
        raise ValueError('candidate must have finite positive absolute Q mass')
    return (weights.double()/total).to(weights.dtype)


def inputs_for_cluster(pair, proposal, include_features):
    ids = torch.unique(proposal.edge_ids.to(device=pair.q.device, dtype=torch.long), dim=0)
    if ids.ndim != 2 or ids.shape[1] != 2 or not len(ids):
        raise ValueError('nonempty exact (i,j) union required')
    if bool((ids < 0).any()) or bool((ids[:,0] >= len(pair.q)).any()) or bool((ids[:,1] >= pair.q.shape[1]).any()):
        raise ValueError('candidate index outside Q')
    i,j = ids.unbind(1)
    q = pair.q[i,j]
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError('invalid Q')
    pose = proposal.translation.to(pair.q)
    aa,_ = observed_arc_cells(pair.ga);ab,_ = observed_arc_cells(pair.gb)
    arc = (aa[i]+ab[j])/2
    w = q*arc;cw = normalized(w)
    residual = pair.gb.points[0,j]-pair.ga.points[0,i]-pose
    distance = residual.norm(dim=-1)
    row = torch.zeros_like(pair.unmatched_a).index_add(0,i,q)
    col = torch.zeros_like(pair.unmatched_b).index_add(0,j,q)
    outside_a = (pair.q.sum(1)-row).clamp_min(0)
    outside_b = (pair.q.sum(0)-col).clamp_min(0)
    unmatched = (pair.unmatched_a[i]+pair.unmatched_b[j])/2
    outside = (outside_a[i]+outside_b[j])/2
    # Invariant to swapping fragments. No latent Feature/Context access in
    # the stats-only arm, even indirectly through another learned module.
    edge = torch.stack((torch.log1p(q*100), q, distance/20,
        (distance/20).square(), unmatched,
        (pair.unmatched_a[i]-pair.unmatched_b[j]).abs(), outside,
        torch.log1p(arc)/4),-1)
    coverage = torch.stack((aa[i.unique()].sum()/pair.ga.perimeter_px[0].clamp_min(1),
                           ab[j.unique()].sum()/pair.gb.perimeter_px[0].clamp_min(1)))
    overlap = material_overlap(pair,pose)
    overlap_value = overlap['fraction_min_area'] if overlap['available'] else 0.
    stats = torch.stack((q.new_tensor(float(len(ids))).log1p(),q.sum().log1p(),w.sum().log1p(),
        q.mean(),q.max(),1/(cw.square().sum()*len(cw)),coverage.mean(),coverage.min(),
        (cw*distance).sum()/20,torch.sqrt((cw*distance.square()).sum().clamp_min(0))/20,
        distance.max()/20,(cw*unmatched).sum(),(cw*outside).sum(),q.new_tensor(overlap_value),
        q.new_tensor(proposal.actual_diameter_px/16),q.new_tensor(float(len(proposal.merged_hypothesis_ids))).log1p()))
    features = None
    if include_features:
        features = torch.cat(((pair.local_a[i]+pair.local_b[j])/2,
            (pair.local_a[i]-pair.local_b[j]).abs(),
            (pair.context_a[i]+pair.context_b[j])/2,
            (pair.context_a[i]-pair.context_b[j]).abs()),-1)
    if not bool(torch.isfinite(stats).all()) or not bool(torch.isfinite(edge).all()):
        raise ValueError('nonfinite cluster features')
    return ClusterInputs(ids,q,arc,w,cw,edge,stats,features,pose)


class BinaryClusterHead(nn.Module):
    def __init__(self, variant='patch', feature_dim=96):
        super().__init__()
        if variant not in ('patch','stats'):raise ValueError('explicit patch/stats arm required')
        self.variant=variant;self.feature_dim=feature_dim
        self.edge_mlp=(nn.Sequential(nn.Linear(4*feature_dim+8,64),nn.GELU(),
            nn.Linear(64,32),nn.GELU()) if variant=='patch' else None)
        self.cluster_mlp=nn.Sequential(nn.Linear(len(SCALAR_NAMES)+(64 if variant=='patch' else 0),64),
            nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))

    @property
    def bias(self):return self.cluster_mlp[-1].bias[0]

    def forward(self,pair,proposal):
        x=inputs_for_cluster(pair,proposal,self.variant=='patch')
        pooled=None
        if self.edge_mlp is not None:
            h=self.edge_mlp(torch.cat((x.patch_context,x.edge_geometry),-1))
            pooled=torch.cat(((h*x.normalized_weights[:,None]).sum(0),h.max(0).values))
        value=x.statistics if pooled is None else torch.cat((pooled,x.statistics))
        logit=self.cluster_mlp(value).squeeze(-1)
        return BinaryReadout(logit,logit.sigmoid(),x,pooled)
