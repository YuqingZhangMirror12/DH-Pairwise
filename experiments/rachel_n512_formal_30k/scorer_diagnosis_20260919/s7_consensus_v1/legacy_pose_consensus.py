"""Deterministic translation consensus; no learned clustering or path scores.

Proposals are sparse for cost control. Their edge union is a diagnostic seed
record, NEVER the final evidence boundary: scoring recalls the full Q again.
"""
from dataclasses import dataclass, fields, replace
import math

import numpy as np
import torch
from torch import Tensor

from .compatibility import CompatibilityConfig, damage_compatibility
from .evidence import PairEvidence, observed_arc_cells, material_overlap
from .geometry import pair_frame


@dataclass(frozen=True)
class ProposalConfig:
    row_column_topk: int = 2
    initial_seeds: int = 16
    max_clusters: int = 8
    minimum_absolute_q: float = 1e-4
    iterations: int = 4
    membership_sigma: float = 3.
    merge_sigma: float = 3.
    normal_vote_fractions: tuple = (0., .25, .5, .75, 1.)
    observation_radius_px: float = 3.5
    maximum_merge_overlap_sum: float = .10


@dataclass(frozen=True)
class EdgeCloud:
    ids: Tensor
    displacement: Tensor
    q: Tensor
    arc_weight: Tensor
    normal_a: Tensor
    normal_b: Tensor
    reliability_a: Tensor
    reliability_b: Tensor
    spacing_a: Tensor
    spacing_b: Tensor
    arc_a: Tensor
    arc_b: Tensor
    perimeter_a: float
    perimeter_b: float

    def compatibility(self, pose, config):
        return damage_compatibility(self.displacement-pose,self.normal_a,self.normal_b,
            self.reliability_a,self.reliability_b,self.spacing_a,self.spacing_b,config)


@dataclass(frozen=True)
class PoseCluster:
    translation: Tensor
    edge_ids: Tensor
    initial_seed_ids: tuple
    merged_hypothesis_ids: tuple
    seed_translations: Tensor
    absolute_support_mass_px: float
    localization_information: Tensor
    underconstrained: bool
    overlap: dict


@dataclass(frozen=True)
class ConsensusProposals:
    cloud: EdgeCloud
    seeds: Tensor
    hypotheses: tuple
    clusters: tuple
    merge_trace: tuple


def _cloud(pair, cfg):
    q = pair.q.detach().float().cpu()
    na,nb = q.shape
    keep = torch.zeros_like(q,dtype=torch.bool)
    if na and nb:
        # sorted=True and coordinate order are tie-stabilized downstream; the
        # full-Q recall does not inherit any proposal TopK truncation.
        k = min(cfg.row_column_topk,nb)
        rows = q.argsort(dim=1,descending=True,stable=True)[:,:k]
        keep.scatter_(1,rows,True)
        k = min(cfg.row_column_topk,na)
        cols = q.argsort(dim=0,descending=True,stable=True)[:k]
        keep.scatter_(0,cols,True)
    ids = (keep & (q >= cfg.minimum_absolute_q)).nonzero(as_tuple=False)
    ia,ib = ids.unbind(-1)
    ga,gb = pair.ga,pair.gb
    def take(g,name,i):
        return getattr(g,name)[0].detach().cpu()[i]
    aa,_ = observed_arc_cells(ga,cfg.observation_radius_px)
    ab,_ = observed_arc_cells(gb,cfg.observation_radius_px)
    return EdgeCloud(ids,pair.points_b.detach().cpu()[ib]-pair.points_a.detach().cpu()[ia],q[ia,ib],
        .5*(aa.detach().cpu()[ia]+ab.detach().cpu()[ib]),
        take(ga,'outward_normal_rc',ia),take(gb,'outward_normal_rc',ib),
        take(ga,'normal_reliability',ia),take(gb,'normal_reliability',ib),
        take(ga,'cell_px',ia),take(gb,'cell_px',ib),
        take(ga,'arc_px',ia),take(gb,'arc_px',ib),float(ga.perimeter_px[0]),float(gb.perimeter_px[0]))


def _pose_key(t):
    r,c = map(float,t)
    # Ordering ties consistently under t -> -t (nonambiguous modes); signed
    # lexicographic values are used only as the final exact-symmetry tiebreak.
    sign = 1 if r>0 or (r==0 and c>=0) else -1
    return (r*r+c*c, sign*r, sign*c,r,c)


class PoseConsensusBuilder:
    def __init__(self, geometry: CompatibilityConfig, proposal=None):
        self.geometry = geometry
        self.config = proposal or ProposalConfig()
        if self.config.initial_seeds < self.config.max_clusters:
            raise ValueError('merge before restricting distinct-cluster budget')

    def _local_scale(self, cloud):
        if not len(cloud.ids):
            return self.geometry.sigma_floor_px
        spacing = .5*(cloud.spacing_a+cloud.spacing_b)
        return max(self.geometry.sigma_floor_px,
                   float(spacing.median())*max(self.geometry.normal_sigma_per_spacing,
                                               self.geometry.tangent_sigma_per_spacing))

    def _seeds(self, cloud):
        if not len(cloud.ids):
            return cloud.displacement.new_zeros((0,2))
        n,_,rel = pair_frame(cloud.normal_a,cloud.normal_b,cloud.reliability_a,cloud.reliability_b)
        fractions = torch.tensor(self.config.normal_vote_fractions,dtype=n.dtype)
        allowance = self.geometry.damage_normal_upper_px*(rel>=self.geometry.normal_reliability_min)
        votes = cloud.displacement[:,None]-n[:,None]*allowance[:,None,None]*fractions[None,:,None]
        weights = (cloud.q*cloud.arc_weight)[:,None].expand(-1,len(fractions))/len(fractions)
        # Only bin proposals, not evidence. Deterministic NumPy accumulation
        # avoids CUDA scatter-add nondeterminism and preserves absolute mass.
        bin_width = 2*self._local_scale(cloud)
        xy,w = votes.reshape(-1,2).numpy(),weights.reshape(-1).numpy()
        bins = np.floor(xy/bin_width+.5).astype(np.int64)
        unique,inverse = np.unique(bins,axis=0,return_inverse=True)
        mass = np.bincount(inverse,weights=w,minlength=len(unique))
        sums = np.stack([np.bincount(inverse,weights=w*xy[:,d],minlength=len(unique)) for d in range(2)],-1)
        centers = sums/np.maximum(mass[:,None],1e-20)
        # Physical-arc diversity without a storage-index origin: after choosing
        # a mode, its contributor arc positions reduce priority of same-region
        # modes. Different regions and displacements remain eligible.
        representative = []
        for k in range(len(unique)):
            members = np.flatnonzero(inverse==k)
            representative.append(int(members[np.argmax(w[members])])//len(fractions))
        order = sorted(range(len(unique)),key=lambda k:(-mass[k],_pose_key(centers[k])))
        candidates = order[:max(128,self.config.initial_seeds*8)]
        chosen=[]
        while candidates and len(chosen)<self.config.initial_seeds:
            def priority(k):
                if not chosen:
                    return float(mass[k])
                i = representative[k]
                diversity=[]
                for h in chosen:
                    j=representative[h]
                    da=abs(float(cloud.arc_a[i]-cloud.arc_a[j])); da=min(da,cloud.perimeter_a-da)
                    db=abs(float(cloud.arc_b[i]-cloud.arc_b[j])); db=min(db,cloud.perimeter_b-db)
                    arc=min(1.,.5*(da+db)/max(1.,8*self.config.observation_radius_px))
                    pose=min(1.,float(np.linalg.norm(centers[k]-centers[h]))/max(1.,4*bin_width))
                    diversity.append(max(arc,pose))
                return float(mass[k])*(.1+.9*min(diversity))
            k=min(candidates,key=lambda x:(-priority(x),_pose_key(centers[x])))
            chosen.append(k); candidates.remove(k)
        return torch.as_tensor(centers[chosen],dtype=cloud.displacement.dtype)

    def _fit(self, cloud, initial, allowed=None):
        t=initial.clone()
        use=torch.ones(len(cloud.ids),dtype=torch.bool) if allowed is None else allowed
        base=cloud.q*cloud.arc_weight*use
        for _ in range(self.config.iterations):
            compat=cloud.compatibility(t,self.geometry)
            error=compat.unexplained_residual_rc
            scale=max(self._local_scale(cloud),1.)
            robust=1/torch.sqrt(1+(error/scale).square().sum(-1))
            weights=base*compat.kernel*robust
            if not bool(weights.sum()>1e-12):
                break
            step=(weights[:,None]*error).sum(0)/weights.sum()
            t=t+step
        c=cloud.compatibility(t,self.geometry)
        membership=(c.kernel>=math.exp(-.5*self.config.membership_sigma**2)) & use
        mass=base*c.kernel
        # Tangential observations constrain translation without pretending that
        # every erosion-offset normal residual is a precise anchor.
        n,tangent,reliability=pair_frame(cloud.normal_a,cloud.normal_b,cloud.reliability_a,cloud.reliability_b)
        reliable=reliability>=self.geometry.normal_reliability_min
        eye=torch.eye(2,dtype=t.dtype).expand(len(mass),-1,-1)
        information=torch.where(reliable[:,None,None],tangent[:,:,None]*tangent[:,None,:],eye)
        information=(mass[:,None,None]*information).sum(0)/mass.sum().clamp_min(1e-12)
        return t,membership,mass,information

    def _hypothesis(self, cloud, initial, seed_ids, seed_positions, hypothesis_ids,
                    overlap_fn=None, allowed=None):
        t,member,mass,info=self._fit(cloud,initial,allowed)
        overlap={} if overlap_fn is None else overlap_fn(t)
        return PoseCluster(t,cloud.ids[member],tuple(sorted(set(seed_ids))),
            tuple(sorted(set(hypothesis_ids))),seed_positions,
            float(mass.sum()),info,bool(torch.linalg.eigvalsh(info).min()<.05),overlap)

    @torch.no_grad()
    def build_from_cloud(self, cloud, seeds=None, overlap_fn=None):
        # Canonical edge identity, not repetition count. Duplicate edge entries
        # cannot contribute twice; conflicting copies are invalid input.
        if len(cloud.ids):
            _,first,inverse=np.unique(cloud.ids.numpy(),axis=0,return_index=True,return_inverse=True)
            if len(first)!=len(cloud.ids):
                updates={}
                for field in fields(cloud):
                    value=getattr(cloud,field.name)
                    if isinstance(value,Tensor):
                        selected=value[torch.as_tensor(first)]
                        if not torch.equal(selected[torch.as_tensor(inverse)],value):
                            raise ValueError('duplicate correspondence records disagree: '+field.name)
                        updates[field.name]=selected
                cloud=replace(cloud,**updates)
        keep=cloud.q>=self.config.minimum_absolute_q
        cloud=replace(cloud,**{f.name:getattr(cloud,f.name)[keep] for f in fields(cloud)
                              if isinstance(getattr(cloud,f.name),Tensor)})
        seeds=self._seeds(cloud) if seeds is None else seeds.detach().cpu()
        hypotheses=tuple(self._hypothesis(cloud,t,(i,),t[None],(i,),overlap_fn)
                         for i,t in enumerate(seeds))
        clusters=[h for h in hypotheses if len(h.edge_ids)]
        trace=[]
        radius=math.sqrt(2)*self.config.merge_sigma*self._local_scale(cloud)
        # Agglomeration always refits ALL union evidence and checks ALL original
        # centers. A succession of neighboring links cannot expand its radius.
        while True:
            options=[]
            for a in range(len(clusters)):
                for b in range(a+1,len(clusters)):
                    ca,cb=clusters[a],clusters[b]
                    if float((ca.translation-cb.translation).norm())<=2*radius:
                        options.append((float((ca.translation-cb.translation).norm()),a,b))
            merged=False
            for _,a,b in sorted(options):
                ca,cb=clusters[a],clusters[b]
                union=torch.unique(torch.cat((ca.edge_ids,cb.edge_ids)),dim=0)
                # Set membership uses exact(i,j), never a span between endpoints.
                stride=int(cloud.ids[:,1].max())+1
                allowed=torch.isin(cloud.ids[:,0]*stride+cloud.ids[:,1],union[:,0]*stride+union[:,1])
                initial=.5*(ca.translation+cb.translation)
                candidate=self._hypothesis(cloud,initial,ca.initial_seed_ids+cb.initial_seed_ids,
                    torch.cat((ca.seed_translations,cb.seed_translations)),
                    ca.merged_hypothesis_ids+cb.merged_hypothesis_ids,overlap_fn,allowed)
                if float((candidate.seed_translations-candidate.translation).norm(dim=-1).max())>radius:
                    continue
                c=cloud.compatibility(candidate.translation,self.geometry)
                if not bool((c.kernel[allowed]>=math.exp(-.5*self.config.membership_sigma**2)).all()):
                    continue
                # Do not average two mutually exclusive, separated partners of
                # the same endpoint into an invented correspondence.
                ids=cloud.ids[allowed]; d=cloud.displacement[allowed]
                alternate_scale=max(self._local_scale(cloud),
                    float((.5*(cloud.spacing_a+cloud.spacing_b)).median())*
                    self.geometry.evidence_tangent_sigma_per_spacing)
                exclusive=(ids[:,None]==ids[None]).any(-1) & ((d[:,None]-d[None]).norm(dim=-1)>2*alternate_scale)
                if bool(exclusive.any()):
                    continue
                if candidate.overlap.get('available') and candidate.overlap['fraction_sum_area']>=self.config.maximum_merge_overlap_sum:
                    continue
                # Recollect against the whole sparse proposal cloud after union.
                candidate=self._hypothesis(cloud,candidate.translation,candidate.initial_seed_ids,
                    candidate.seed_translations,candidate.merged_hypothesis_ids,overlap_fn)
                final_c=cloud.compatibility(candidate.translation,self.geometry)
                if (float((candidate.seed_translations-candidate.translation).norm(dim=-1).max())>radius
                        or not bool((final_c.kernel[allowed]>=math.exp(-.5*self.config.membership_sigma**2)).all())):
                    continue
                trace.append(dict(merged_hypothesis_ids=candidate.merged_hypothesis_ids,
                    union_edge_count=len(union),recollected_edge_count=len(candidate.edge_ids),
                    translation=candidate.translation.tolist()))
                clusters=[c for i,c in enumerate(clusters) if i not in (a,b)]+[candidate]
                merged=True; break
            if not merged:
                break
        clusters.sort(key=lambda c:(-c.absolute_support_mass_px,_pose_key(c.translation)))
        return ConsensusProposals(cloud,seeds,hypotheses,tuple(clusters[:self.config.max_clusters]),tuple(trace))

    @torch.no_grad()
    def __call__(self, pair):
        cloud=_cloud(pair,self.config)
        if not pair.numeric_valid:
            return ConsensusProposals(cloud,torch.zeros((0,2)),(),(),())
        return self.build_from_cloud(cloud,overlap_fn=lambda t:material_overlap(pair,t))
