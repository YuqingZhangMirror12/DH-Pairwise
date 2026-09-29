"""Fixed threshold builder pose + direct binary cluster score.

No inherited neural localizer/three-class head is hidden in these experiments.
The common-pose geometric fit is the existing threshold builder's union fit.
"""
from dataclasses import dataclass
import torch
from torch import nn
from ..s7_consensus_v1.evidence import PairEvidence
from ..s7_consensus_v1.pose_consensus import PoseConsensusBuilder
from ..s7_consensus_v1.model import PairPrediction
from .head import BinaryClusterHead


@dataclass(frozen=True)
class BinaryScoredCluster:
    proposal: object
    readout: object

    @property
    def translation(self):return self.readout.inputs.pose


class BinaryConsensus(nn.Module):
    def __init__(self,matcher,geometry,proposal=None,head=None):
        super().__init__();self.matcher=matcher;self.geometry=geometry
        self.builder=PoseConsensusBuilder(geometry,proposal)
        self.head=head if head is not None else BinaryClusterHead()

    def score_pair(self,pair,threshold=.5,proposals=None,capture_diagnostics=False):
        proposals=self.builder(pair) if proposals is None else proposals
        clusters=tuple(BinaryScoredCluster(p,self.head(pair,p)) for p in proposals.clusters)
        if not clusters:
            return PairPrediction(False,pair.numeric_valid,-1,None,pair.q.new_zeros(()),False,
                dict(underconstrained=True,reason='no_candidate'),(),proposals)
        winner=int(torch.stack([c.readout.logit for c in clusters]).detach().argmax())
        c=clusters[winner]
        valid=pair.numeric_valid and bool(torch.isfinite(c.translation).all() & torch.isfinite(c.readout.score))
        return PairPrediction(True,valid,winner,c.translation,c.readout.score,
            bool(valid and c.readout.score.detach()>=threshold),
            dict(underconstrained=bool(c.proposal.underconstrained),
                 note='threshold-builder geometric observability; no learned pose refinement'),clusters,proposals)

    def forward(self,mask_a,mask_b,points_rc_a,points_rc_b,contour_valid_a,contour_valid_b,threshold=.5,
                capture_diagnostics=False):
        output=self.matcher(mask_a,mask_b,points_rc_a,points_rc_b,contour_valid_a,contour_valid_b)
        return [self.score_pair(PairEvidence.from_matcher(output,i,mask_a,mask_b),threshold)
                for i in range(len(mask_a))]
