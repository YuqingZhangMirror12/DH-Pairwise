"""Raw displacement modes -> radial support -> fitted/contact pose -> dedup.

Only three GROUPING knobs: pose radius, budget, physical overlap limit.
Legacy directional geometry is used in positioning, NEVER in membership,
mode density, dedup, or supported-arc ordering.
"""
from dataclasses import dataclass, fields, replace
import math
import numpy as np
import torch

from modes import weighted_modes
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.legacy_pose_consensus import (
    PoseConsensusBuilder as LegacyBuilder, ProposalConfig, PoseCluster, ConsensusProposals, _cloud)
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.geometry import pair_frame
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import observed_arc_cells, material_overlap


@dataclass(frozen=True)
class SimplePolicy:
    pose_radius_px: float = 10.
    candidate_budget: int = 8
    maximum_interpenetration_sum: float = .10

    def __post_init__(self):
        if not math.isfinite(self.pose_radius_px) or self.pose_radius_px<=0:
            raise ValueError('positive finite pixel radius required')
        if self.candidate_budget<=0 or not 0<=self.maximum_interpenetration_sum<=1:
            raise ValueError('invalid budget/interpenetration')


@dataclass(frozen=True)
class SimpleCluster(PoseCluster):
    mode_centers_rc: torch.Tensor
    independent_arc_px: float
    raw_absolute_mass_px: float
    contact_shift_px: float
    original_union_edge_ids: torch.Tensor


def subset(cloud, index):
    return replace(cloud, **{f.name:getattr(cloud,f.name)[index] for f in fields(cloud)
        if isinstance(getattr(cloud,f.name),torch.Tensor)})


def canonical(cloud):
    """Identity dedup only; no Q/distance/reliability filtering."""
    if not len(cloud.ids): return cloud
    _, first, inverse=np.unique(cloud.ids.numpy(),axis=0,return_index=True,return_inverse=True)
    for f in fields(cloud):
        v=getattr(cloud,f.name)
        if isinstance(v,torch.Tensor) and not torch.equal(v, v[first][inverse]):
            raise ValueError('conflicting copies of a single edge identity')
    return subset(cloud,torch.as_tensor(first))


class SimplePoseBuilder(LegacyBuilder):
    def __init__(self,geometry,proposal=None,policy=None):
        super().__init__(geometry,proposal or ProposalConfig())
        self.policy=policy or SimplePolicy()
        self.all_clusters=()
        self.audit={}

    def _seeds(self,cloud):
        raise RuntimeError('heuristic seeds are forbidden in this builder')

    def _contact(self,cloud,pose,information):
        """Nearest retained reliable edge contacts along weak normal axis.

        This convention is NOT a recovered pre-erosion GT pose. No mass-loss
        guard, percentile, or new damage allowance. Preserve the existing
        finite damage interval; final material overlap is the sole veto.
        """
        val,vec=torch.linalg.eigh(information)
        if float(val[0])>=.05: return pose
        normal,_,rel=pair_frame(cloud.normal_a,cloud.normal_b,cloud.reliability_a,cloud.reliability_b)
        reliable=rel>=self.geometry.normal_reliability_min
        if not reliable.any(): return pose
        axis=vec[:,0]
        weights=cloud.q*cloud.arc_weight
        if float(axis@(normal*weights[:,None]).sum(0))<0: axis=-axis
        projection=normal@axis
        valid=reliable & (projection>1e-8)
        if not valid.any(): return pose
        residual=(cloud.displacement-pose)*normal
        distance=residual.sum(1)[valid]/projection[valid]
        step=distance.min().clamp(0.,self.geometry.damage_normal_upper_px)
        return pose+step*axis

    def _make(self,cloud,indices,initial,mode_ids,mode_centers,cells,overlap_fn):
        selected=subset(cloud,torch.as_tensor(sorted(indices),dtype=torch.long))
        # Exactly one robust fit call; its membership return is IGNORED.
        pose,_,mass,info=self._fit(selected,initial)
        before=pose
        pose=self._contact(selected,pose,info)
        overlap={} if overlap_fn is None else overlap_fn(pose)
        ids=selected.ids
        if cells is None:
            # Only used by synthetic controls. Real evaluations supply the
            # exact original observed Voronoi cells from both full contours.
            lengths=[]
            for side in (0,1):
                _,first=np.unique(ids[:,side].numpy(),return_index=True)
                spacing=selected.spacing_a if side==0 else selected.spacing_b
                lengths.append(float(spacing[first].clamp_max(2*self.config.observation_radius_px).sum()))
        else:
            lengths=[float(cells[s][ids[:,s].unique()].sum()) for s in (0,1)]
        return SimpleCluster(pose,ids,tuple(mode_ids),tuple(mode_ids),mode_centers,
            float(mass.sum()),info,bool(torch.linalg.eigvalsh(info).min()<.05),overlap,
            mode_centers,min(lengths),float((selected.q*selected.arc_weight).sum()),
            float((pose-before).norm()),ids)

    @torch.no_grad()
    def build_from_cloud(self,cloud,seeds=None,overlap_fn=None,cells=None):
        if seeds is not None:
            raise ValueError('raw mode version does not accept heuristic seeds')
        cloud=canonical(cloud)
        radius=self.policy.pose_radius_px
        modes=weighted_modes(cloud.displacement.double().numpy(),(cloud.q.double()*cloud.arc_weight.double()).numpy(),radius)
        centers=torch.tensor(np.asarray([m['center'] for m in modes]).reshape(-1,2),dtype=cloud.displacement.dtype)
        fitted=[self._make(cloud,m['members'],centers[i],(i,),centers[i:i+1],cells,overlap_fn)
            for i,m in enumerate(modes)]
        order=sorted(range(len(fitted)),key=lambda i:(-fitted[i].independent_arc_px,
            -fitted[i].raw_absolute_mass_px,*fitted[i].translation.tolist(),i))
        edge_sets=[set(map(tuple,c.edge_ids.tolist())) for c in fitted]
        groups=[]
        for i in order:
            # A single fixed-representative assignment pass: no nearest-
            # neighbor connected components, transitive chain, or fit/retry.
            found=None
            for g in groups:
                j=g[0]
                union=edge_sets[i]|edge_sets[j]
                jac=len(edge_sets[i]&edge_sets[j])/len(union) if union else 0.
                if float((fitted[i].translation-fitted[j].translation).norm())<radius or jac>=.9:
                    found=g;break
            if found is None: groups.append([i])
            else: found.append(i)
        final=[];vetoes=0
        for group in groups:
            if len(group)==1:
                c=fitted[group[0]]
            else:
                member=np.unique(np.concatenate([modes[i]['members'] for i in group]))
                # A merged candidate is the true union, refitted ONCE jointly.
                # Member admission is never recomputed from directional fit.
                c=self._make(cloud,member,fitted[group[0]].translation,tuple(group),centers[group],cells,overlap_fn)
            if c.overlap.get('available') and c.overlap['fraction_sum_area']>=self.policy.maximum_interpenetration_sum:
                vetoes+=1;continue
            final.append(c)
        final.sort(key=lambda c:(-c.independent_arc_px,-c.raw_absolute_mass_px,*c.translation.tolist()))
        self.all_clusters=tuple(final)
        self.audit=dict(radius_px=radius,raw_modes=len(modes),dedup_groups=len(groups),
            interpenetration_vetoes=vetoes,prebudget=len(final),candidate_budget=self.policy.candidate_budget,
            no_directional_membership=True,no_extra_cloud_filter=True)
        return ConsensusProposals(cloud,centers,tuple(fitted),tuple(final[:self.policy.candidate_budget]),
            tuple(dict(mode_ids=g) for g in groups))

    @torch.no_grad()
    def __call__(self,pair):
        cloud=_cloud(pair,self.config)
        cells=tuple(observed_arc_cells(g,self.config.observation_radius_px)[0].detach().cpu() for g in (pair.ga,pair.gb))
        return self.build_from_cloud(cloud,overlap_fn=lambda t:material_overlap(pair,t),cells=cells)
