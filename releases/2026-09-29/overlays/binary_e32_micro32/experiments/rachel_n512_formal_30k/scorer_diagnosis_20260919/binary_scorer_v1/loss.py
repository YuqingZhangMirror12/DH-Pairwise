"""Whole-cluster binary quality + within-pair ranking, no local class targets."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from ..s7_consensus_v1.targets import candidate_quality


@dataclass(frozen=True)
class BinaryLossConfig:
    candidate_weight:float=1.
    ranking_weight:float=.2
    ranking_margin:float=1.
    quality_tolerance_px:float=20.


@dataclass(frozen=True)
class BinaryPairLoss:
    total:object
    components:dict
    counts:dict


def pair_loss(model,prediction,labels,config=None):
    cfg=config or BinaryLossConfig();zero=model.head.bias*0
    y,known=candidate_quality([c.translation for c in prediction.clusters],labels,cfg.quality_tolerance_px)
    candidate=ranking=zero;comparisons=0
    if bool(known.any()):
        logits=torch.stack([c.readout.logit for c in prediction.clusters])
        candidate=F.binary_cross_entropy_with_logits(logits[known],y[known])
        good=logits[known&(y==1)];bad=logits[known&(y==0)]
        if len(good) and len(bad):
            ranking=F.softplus(bad[None]-good[:,None]+cfg.ranking_margin).mean()
            comparisons=len(good)*len(bad)
    return BinaryPairLoss(candidate*cfg.candidate_weight+ranking*cfg.ranking_weight,
        dict(candidate=candidate,ranking=ranking),dict(pairs=1,positive_pairs=int(labels.label),
            candidate_count=len(y),correct_clusters=int(((y==1)&known).sum()),
            wrong_clusters=int(((y==0)&known).sum()),unknown_clusters=int((~known).sum()),
            ranking_comparisons=comparisons,
            positive_coverage_misses=int(labels.label and labels.translation_known and not bool((y[known]>0).any()))))


def batch_loss(model,predictions,labels,config=None,extension_flags=None):
    if len(predictions)!=len(labels) or not predictions:raise ValueError('one label record per pair')
    # Old engine supplies flags; this registered head has NO extension/local/
    # localization supervision. No unused complex head is trained for them.
    items=[pair_loss(model,p,l,config) for p,l in zip(predictions,labels)]
    return torch.stack([x.total for x in items]).mean(),items
