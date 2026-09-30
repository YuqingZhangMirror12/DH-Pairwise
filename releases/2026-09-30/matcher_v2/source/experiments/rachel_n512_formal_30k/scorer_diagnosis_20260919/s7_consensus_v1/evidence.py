"""Full-Q candidate evidence. No GT, selected-chain cache or second Sinkhorn."""
from dataclasses import dataclass

import torch
from torch import Tensor

from .compatibility import CompatibilityConfig, damage_compatibility
from .geometry import ContourGeometry, compact_contour


@dataclass(frozen=True)
class PairEvidence:
    local_a: Tensor
    local_b: Tensor
    context_a: Tensor
    context_b: Tensor
    q: Tensor
    unmatched_a: Tensor
    unmatched_b: Tensor
    ga: ContourGeometry
    gb: ContourGeometry
    original_a: Tensor
    original_b: Tensor
    mask_a: Tensor = None
    mask_b: Tensor = None
    numeric_valid: bool = True

    @classmethod
    def from_matcher(cls, output, batch_index, mask_a=None, mask_b=None):
        b = batch_index
        ia = output.valid_a[b].nonzero(as_tuple=False).flatten()
        ib = output.valid_b[b].nonzero(as_tuple=False).flatten()
        pa, pb = output.points_rc_a[b,ia], output.points_rc_b[b,ib]
        ga = compact_contour(pa[None], torch.ones_like(pa[None,:,0],dtype=torch.bool))
        gb = compact_contour(pb[None], torch.ones_like(pb[None,:,0],dtype=torch.bool))
        return cls(output.local_a[b,ia], output.local_b[b,ib],
            output.context_a[b,ia], output.context_b[b,ib],
            output.assignment[b][ia][:,ib].float(), output.unmatched_a[b,ia].float(),
            output.unmatched_b[b,ib].float(), ga, gb, ia, ib,
            None if mask_a is None else mask_a[b].squeeze(0),
            None if mask_b is None else mask_b[b].squeeze(0), bool(output.numeric_valid[b]))

    @property
    def points_a(self):
        return self.ga.points[0,:len(self.local_a)]

    @property
    def points_b(self):
        return self.gb.points[0,:len(self.local_b)]

    def compatibility(self, pose, config, start=0, end=None):
        end = len(self.local_a) if end is None else end
        residual = self.points_b[None]-self.points_a[start:end,None]-pose
        result = damage_compatibility(residual,
            self.ga.outward_normal_rc[0,start:end,None], self.gb.outward_normal_rc[0,None,:len(self.local_b)],
            self.ga.normal_reliability[0,start:end,None], self.gb.normal_reliability[0,None,:len(self.local_b)],
            self.ga.cell_px[0,start:end,None], self.gb.cell_px[0,None,:len(self.local_b)],config)
        return residual, result


def observed_arc_cells(geometry, observation_radius_px=3.5):
    """Nonoverlapping observed Voronoi intervals, capped by the7px patch radius.

    Never extend across a missing stretch to a remote selected endpoint. This
    is visible sampling measure, not a label that the entire cell is true seam.
    With denser samples, cells partition the same visible contour measure.
    """
    n = int(geometry.counts[0])
    step = geometry.next_step_px[0,:n]
    left = (.5*step.roll(1)).clamp_max(observation_radius_px)
    right = (.5*step).clamp_max(observation_radius_px)
    arc = geometry.arc_px[0,:n]
    return left+right, torch.stack((arc-left,arc+right),-1)


def relative_arc_features(geometry, mass):
    """Candidate-relative circular moments; invariant to changing storage start.

    An ambiguous circular centroid produces near-zero relative coordinates,
    with its concentration explicitly carried, instead of an arbitrary origin.
    """
    n = len(mass)
    angle = 2*torch.pi*geometry.arc_px[0,:n]/geometry.perimeter_px[0].clamp_min(1.)
    unit = torch.stack((angle.cos(),angle.sin()),-1)
    center = (unit*mass[:,None]).sum(0)/mass.sum().clamp_min(1e-12)
    cosrel = unit @ center
    sinrel = unit[:,1]*center[0]-unit[:,0]*center[1]
    return torch.stack((cosrel,sinrel,center.norm().expand(n)),-1)


@dataclass(frozen=True)
class SideEvidence:
    local: Tensor
    context: Tensor
    opposite_local: Tensor
    opposite_context: Tensor
    mass: Tensor
    unmatched: Tensor
    other_mass: Tensor
    valid: Tensor
    scalar_features: Tensor
    residual_mean: Tensor
    residual_variance: Tensor
    observed_arc_px: Tensor
    arc_intervals: Tensor


@dataclass(frozen=True)
class RecalledEvidence:
    pose: Tensor
    weights: Tensor
    kernels: Tensor
    localization_kernels: Tensor
    a: SideEvidence
    b: SideEvidence
    pair: PairEvidence

    def correspondence_ids(self, numerical_floor=0.):
        """Diagnostic sparse export only; never limits the scoring input."""
        return (self.weights > numerical_floor).nonzero(as_tuple=False)


def _side(pair, w, kernel_location, residual, normal, tangent, reliability,
          side, config, radius):
    is_a = side == 'a'
    q = pair.q if is_a else pair.q.T
    g = pair.ga if is_a else pair.gb
    opposite_g = pair.gb if is_a else pair.ga
    local = pair.local_a if is_a else pair.local_b
    context = pair.context_a if is_a else pair.context_b
    other_local = pair.local_b if is_a else pair.local_a
    other_context = pair.context_b if is_a else pair.context_a
    unmatched = pair.unmatched_a if is_a else pair.unmatched_b
    mass = w.sum(-1)
    conditional = w/mass[:,None].clamp_min(1e-20)
    # A conditional mean is a statistic, NOT a synthetic correspondence point.
    mean = (conditional[...,None]*residual).sum(1)
    variance = (conditional[...,None]*(residual-mean[:,None]).square()).sum(1)
    arcs, intervals = observed_arc_cells(g,radius)
    opposite_arcs, _ = observed_arc_cells(opposite_g,radius)
    entropy = -(conditional*torch.log((conditional/opposite_arcs[None].clamp_min(1e-6)).clamp_min(1e-20))).sum(1)
    other_mass = (q.sum(1)-mass).clamp_min(0)
    valid = mass > 0  # Only numerical zeros absent; no globalTop512/Q percentile cut.
    delta_scale = max(config.damage_normal_upper_px, config.sigma_floor_px)
    statistics = [mass, unmatched, other_mass,
        (conditional*normal).sum(1)/delta_scale,
        (conditional*tangent).sum(1)/delta_scale,
        (conditional*reliability).sum(1),
        (conditional*kernel_location).sum(1),
        entropy/torch.log1p(opposite_g.perimeter_px[0]).clamp_min(1),
        torch.log1p(g.cell_px[0,:len(mass)])/4.,
        torch.log1p(arcs)/4.]
    scalar = torch.cat((torch.stack(statistics,-1),mean/delta_scale,
        torch.sqrt(variance.clamp_min(0)+1e-12)/delta_scale,
        relative_arc_features(g,mass*arcs)), -1)
    return SideEvidence(local,context,conditional@other_local,conditional@other_context,
        mass,unmatched,other_mass,valid,scalar,mean,variance,arcs,intervals)


def recall_full_q(pair: PairEvidence, pose: Tensor, config: CompatibilityConfig,
                  block_rows=64, observation_radius_px=3.5, edge_mask=None) -> RecalledEvidence:
    """Revisit EVERY Q entry in row blocks; retain all alternate partners.

    edge_mask is reserved for explicitly labelled TRAIN extension/duplication
    tests. Ordinary inference must leave it None. It never changes Q/dustbin.
    """
    if pair.q.shape != (len(pair.local_a),len(pair.local_b)) or block_rows <= 0:
        raise ValueError('inconsistent compact full Q')
    if not pair.numeric_valid or not len(pair.local_a) or not len(pair.local_b):
        raise ValueError('cannot recall an empty/numerically invalid pair')
    rows = []
    for start in range(0,len(pair.local_a),block_rows):
        end = min(start+block_rows,len(pair.local_a))
        residual, c = pair.compatibility(pose,config,start,end)
        rows.append((residual,c))
    residual = torch.cat([x[0] for x in rows])
    kernel = torch.cat([x[1].kernel for x in rows])
    location = torch.cat([x[1].localization_kernel for x in rows])
    normal = torch.cat([x[1].normal_px for x in rows])
    tangent = torch.cat([x[1].tangent_px for x in rows])
    reliability = torch.cat([x[1].normal_reliability for x in rows])
    w = pair.q*kernel
    if edge_mask is not None:
        if edge_mask.shape != w.shape or edge_mask.dtype != torch.bool:
            raise ValueError('extension mask must name full-Q edges')
        w = w*edge_mask
    a = _side(pair,w,location,residual,normal,tangent,reliability,'a',config,observation_radius_px)
    # B frame negates BOTH normal direction and residual, preserving signed
    # normal/tangent components, while the vector residual changes sign.
    b = _side(pair,w.T,location.T,-residual.permute(1,0,2),normal.T,tangent.T,
        reliability.T,'b',config,observation_radius_px)
    return RecalledEvidence(pose,w,kernel,location,a,b,pair)


@torch.no_grad()
def material_overlap(pair, pose):
    """Exact integer-raster overlap at rounded pose, not bounding-box overlap.

    B is placed at -t. Discreteness is explicit; this is not a claimed smooth
    refinement gradient. Both common denominators are returned.
    """
    if pair.mask_a is None or pair.mask_b is None:
        return {'available':False,'intersection_px':None,'fraction_min_area':None,'fraction_sum_area':None}
    a,b = pair.mask_a,pair.mask_b
    dr,dc = [int(x) for x in pose.detach().round().cpu().tolist()]
    ar0,ac0 = max(0,-dr),max(0,-dc)
    ar1,ac1 = min(a.shape[0],b.shape[0]-dr),min(a.shape[1],b.shape[1]-dc)
    intersection = 0.
    if ar1>ar0 and ac1>ac0:
        intersection = float((a[ar0:ar1,ac0:ac1]*b[ar0+dr:ar1+dr,ac0+dc:ac1+dc]).sum())
    aa,ab = float(a.sum()),float(b.sum())
    return dict(available=True,intersection_px=intersection,
                fraction_min_area=intersection/max(1.,min(aa,ab)),
                fraction_sum_area=intersection/max(1.,aa+ab))
