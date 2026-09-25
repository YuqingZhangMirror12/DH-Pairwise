"""Pair-normalized candidate supervision plus reliable evidence extension.

No positive PairBCE on max(logit). A positive pair with no correct proposal is
recorded as a coverage miss; its wrong candidates retain negative quality GT.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .evidence import recall_full_q,material_overlap
from .targets import candidate_quality,local_targets,local_distributions


@dataclass(frozen=True)
class LossConfig:
    candidate_weight: float = 1.
    ranking_weight: float = .2
    local_weight: float = .1
    localization_weight: float = .1
    extension_weight: float = .1
    ranking_margin: float = 1.
    extension_epsilon: float = .02
    quality_tolerance_px: float = 20.
    extension_min_edges: int = 4
    extension_min_kernel: float = .5


@dataclass(frozen=True)
class PairLoss:
    total: torch.Tensor
    components: dict
    counts: dict


def supervised_mass_mean(values, weight):
    """Same weighted mean, without a reciprocal-subnormal gradient in FP32.

    These are detached supervision weights, not inference Q or attention mass.
    Normalize them before multiplying differentiable losses. With the previous
    `(loss * weight).sum() / weight.sum()`, a finite tiny denominator produced
    an infinite intermediate backward gradient, even when the final analytic
    derivative should be bounded. Float64 here is only for the detached sum.
    No nonzero evidence is dropped or boosted in the inference readout.
    """
    if weight.requires_grad:
        raise ValueError('supervision mass must be detached')
    measure = weight.double()
    total = measure.sum()
    if not bool(total > 0):
        return values.sum() * 0
    normalized = (measure / total).to(values.dtype)
    return (values * normalized).sum()


def extension_loss(model,cluster,labels,cfg):
    """Hold pose fixed; remove and restore only known compatible source edges.

    Unknown/conflicting observations are kept identical in both versions. The
    inequality is not a high-confidence target for the remaining partial seam.
    """
    pair=cluster.encoded.evidence.pair
    targets=local_targets(pair,labels)
    pose=cluster.translation.detach()
    full=recall_full_q(pair,pose,model.geometry)
    usable=targets.source_support_known&(full.kernels.detach()>=cfg.extension_min_kernel)&(full.weights.detach()>0)
    ids=usable.nonzero(as_tuple=False)
    if len(ids)<cfg.extension_min_edges:
        return model.head.bias*0,dict(extension_pairs=0,extension_added_edges=0)
    # Existing reciprocal source edges have unique endpoints. Different ordered
    # halves are disjoint observed pieces; intervening unknown pixels stay unknown.
    chosen=ids[len(ids)//2:]
    partial_mask=torch.ones_like(usable)
    partial_mask[chosen[:,0],chosen[:,1]]=False
    partial=recall_full_q(pair,pose,model.geometry,edge_mask=partial_mask)
    overlap=material_overlap(pair,pose)
    penalty=0. if not overlap['available'] else overlap['fraction_min_area']
    partial_logit=model.head.readout(model.head(partial),penalty).logit
    complete_logit=model.head.readout(model.head(full),penalty).logit
    value=F.relu(partial_logit-complete_logit-cfg.extension_epsilon)
    return value,dict(extension_pairs=1,extension_added_edges=len(chosen),
                     extension_fixed_pose=True)


def pair_loss(model,prediction,labels,config=None,include_extension=False):
    cfg=config or LossConfig()
    zero=model.head.bias*0
    ys,known=candidate_quality([c.translation for c in prediction.clusters],labels,cfg.quality_tolerance_px)
    counts=dict(pairs=1,positive_pairs=int(labels.label),candidate_count=len(ys),
        correct_clusters=int(((ys==1)&known).sum()),wrong_clusters=int(((ys==0)&known).sum()),
        unknown_clusters=int((~known).sum()),quality_labeled_pairs=int(bool(known.any())),
        positive_coverage_misses=int(labels.label and labels.translation_known and not bool((ys[known]>0).any())),
        local_known_mass=0.,local_supervised_tokens=0,localization_known_mass=0.,
        ranking_comparisons=0,extension_pairs=0,extension_added_edges=0)
    components=dict(candidate=zero,ranking=zero,local=zero,localization=zero,extension=zero)
    if len(ys) and bool(known.any()):
        logits=torch.stack([c.readout.logit for c in prediction.clusters])
        components['candidate']=F.binary_cross_entropy_with_logits(logits[known],ys[known])
        positive=logits[known&(ys==1)];negative=logits[known&(ys==0)]
        if len(positive) and len(negative):
            components['ranking']=F.softplus(negative[None]-positive[:,None]+cfg.ranking_margin).mean()
            counts['ranking_comparisons']=len(positive)*len(negative)
    locals_,locations_=[] , []
    for c in prediction.clusters:
        targets=local_targets(c.encoded.evidence.pair,labels)
        losses=[];loclosses=[]
        for side in 'ab':
            truth=local_distributions(c.encoded.evidence,targets,side)
            out=getattr(c.encoded,side)
            weight=truth['class_weight'];total=weight.sum()
            counts['local_known_mass']+=float(total)
            counts['local_supervised_tokens']+=int(truth['local_target_valid'].sum())
            if bool(total>0):
                ce=-(truth['class_target']*out.local_logits.log_softmax(-1)).sum(-1)
                losses.append(supervised_mass_mean(ce,weight))
            weight=truth['location_weight'];total=weight.sum()
            counts['localization_known_mass']+=float(total)
            if bool(total>0):
                loss=F.binary_cross_entropy(out.localization_reliability.clamp(1e-7,1-1e-7),
                    truth['location_target'],reduction='none')
                loclosses.append(supervised_mass_mean(loss,weight))
        if losses:
            locals_.append(torch.stack(losses).mean())
        if loclosses:
            locations_.append(torch.stack(loclosses).mean())
    if locals_:
        components['local']=torch.stack(locals_).mean()
    if locations_:
        components['localization']=torch.stack(locations_).mean()
    if include_extension and labels.label and labels.translation_known and bool((ys[known]>0).any()):
        # One deterministic correct native cluster per chosen training pair.
        # Does not inject a GT pose or create a new candidate.
        index=int((known&(ys==1)).nonzero(as_tuple=False)[0,0])
        components['extension'],extra=extension_loss(model,prediction.clusters[index],labels,cfg)
        counts.update(extra)
    total=sum(components[name]*getattr(cfg,name+'_weight') for name in components)
    return PairLoss(total,components,counts)


def batch_loss(model,predictions,labels,config=None,extension_flags=None):
    if len(predictions)!=len(labels) or not predictions:
        raise ValueError('one label record required per prediction')
    flags=extension_flags or [False]*len(predictions)
    if len(flags)!=len(predictions):
        raise ValueError('extension schedule length differs')
    items=[pair_loss(model,p,l,config,bool(flag)) for p,l,flag in zip(predictions,labels,flags)]
    # Always divide by the actual batch pairs, not varying valid candidates or
    # positives in a microbatch; accumulation normalization is explicitly stable.
    return torch.stack([x.total for x in items]).mean(),items
