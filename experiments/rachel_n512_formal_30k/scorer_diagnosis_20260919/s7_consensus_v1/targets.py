"""Candidate and local labels. No source metadata enters inference features."""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PairLabels:
    label: bool
    translation_known: bool
    translation_a_to_b_rc: Tensor
    target_a: Tensor  # original/storage IDs; -2unknown, -1known unmatched
    target_b: Tensor
    precise_anchor_a: Tensor
    precise_anchor_b: Tensor

    @classmethod
    def from_sample(cls,sample,corrosion_recipe,device='cpu'):
        import numpy as np
        def t(x,dtype):
            return torch.as_tensor(np.array(x,copy=True),dtype=dtype,device=device)
        a,b=t(sample.target_a,torch.long),t(sample.target_b,torch.long)
        return cls(bool(sample.label),bool(sample.translation_valid),
            t(sample.translation_a_to_b_rc,torch.float32),a,b,
            (a>=0)&(corrosion_recipe=='clean'),(b>=0)&(corrosion_recipe=='clean'))


@dataclass(frozen=True)
class LocalTargets:
    source_support_known: Tensor
    edge_target_known: Tensor
    precise_anchor_known: Tensor


def local_targets(pair,labels):
    ia,ib=pair.original_a,pair.original_b
    ta,tb=labels.target_a[ia],labels.target_b[ib]
    if not labels.label:
        return LocalTargets(torch.zeros_like(pair.q,dtype=torch.bool),
            torch.ones_like(pair.q,dtype=torch.bool),torch.zeros_like(pair.q,dtype=torch.bool))
    support=(ta[:,None]==ib[None])&(tb[None]==ia[:,None])
    known=(ta[:,None]!=-2)&(tb[None]!=-2)
    precise=support&labels.precise_anchor_a[ia,None]&labels.precise_anchor_b[None,ib]
    return LocalTargets(support,known,precise)


def candidate_quality(translations,labels,success_tolerance_px=20.):
    """20px belongs ONLY to evaluation/quality labels, not the geometry kernel."""
    if not len(translations):
        return labels.translation_a_to_b_rc.new_empty(0),torch.empty(0,dtype=torch.bool,device=labels.target_a.device)
    poses=torch.stack(translations)
    if not labels.label:
        return poses.new_zeros(len(poses)),torch.ones(len(poses),dtype=torch.bool,device=poses.device)
    if not labels.translation_known:
        return poses.new_zeros(len(poses)),torch.zeros(len(poses),dtype=torch.bool,device=poses.device)
    distance=(poses.detach()-labels.translation_a_to_b_rc).norm(dim=-1)
    return (distance<=success_tolerance_px).to(poses.dtype),torch.ones_like(distance,dtype=torch.bool)


def local_distributions(recalled,targets,side):
    w=recalled.weights.detach()
    support=targets.source_support_known
    known=targets.edge_target_known
    precise=targets.precise_anchor_known
    if side=='b':
        w,support,known,precise=w.T,support.T,known.T,precise.T
    known_mass=(w*known).sum(1)
    true_mass=(w*support).sum(1)
    negative_mass=(w*(known&~support)).sum(1)
    # Missing source truth is masked, never forced into the "unknown" class.
    class_target=torch.stack((true_mass,torch.zeros_like(true_mass),negative_mass),-1)/known_mass[:,None].clamp_min(1e-20)
    anchor_mass=(w*precise).sum(1)
    location_known=anchor_mass+negative_mass
    location_target=anchor_mass/location_known.clamp_min(1e-20)
    return dict(class_target=class_target,class_weight=known_mass,
                local_target_valid=known_mass>0,
                location_target=location_target,location_weight=location_known,
                precise_anchor_known=anchor_mass>0)
