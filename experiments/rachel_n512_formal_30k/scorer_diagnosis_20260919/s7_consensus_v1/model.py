"""New common-pose model; oneMatcher/oneSinkhorn, shared two-pass evidence head."""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .consensus_head import ConsensusEvidenceHead, EncodedEvidence, Readout
from .evidence import PairEvidence, recall_full_q, material_overlap
from .pose_consensus import PoseConsensusBuilder, PoseCluster, ConsensusProposals
from .pose_refinement import refine_pose, Refinement


@dataclass(frozen=True)
class ScoredCluster:
    proposal: PoseCluster
    refinement: Refinement
    encoded: EncodedEvidence
    readout: Readout
    overlap: dict
    # Opt-in diagnostic retention only; no additional network evaluation.
    initial_encoded: Optional[EncodedEvidence] = None

    @property
    def translation(self):
        return self.refinement.translation


@dataclass(frozen=True)
class PairPrediction:
    has_candidate: bool
    numeric_valid: bool
    selected_cluster_id: int
    translation_a_to_b_rc: Tensor
    score: Tensor
    accepted: bool
    pose_uncertainty: dict
    clusters: tuple
    proposals: ConsensusProposals


class S7Consensus(nn.Module):
    def __init__(self,matcher,geometry,proposal=None,head=None):
        super().__init__()
        self.matcher=matcher
        self.geometry=geometry
        self.builder=PoseConsensusBuilder(geometry,proposal)
        self.head=head or ConsensusEvidenceHead()

    def score_pair(self,pair:PairEvidence,threshold=.5,proposals=None,capture_diagnostics=False):
        proposals=self.builder(pair) if proposals is None else proposals
        scored=[]
        initial=[recall_full_q(pair,p.translation.to(device=pair.q.device,dtype=pair.q.dtype),self.geometry)
                 for p in proposals.clusters]
        encoded=self.head.forward_many(initial)
        refinements=[refine_pose(e,self.geometry) for e in encoded]
        finals=[recall_full_q(pair,r.translation,self.geometry) for r in refinements]
        final_encoded=self.head.forward_many(finals)
        for proposal,refined,before,final in zip(proposals.clusters,refinements,encoded,final_encoded):
            # Rebuild EVERYTHING depending on pose. No stale latent re-use and
            # no secondMatcher/secondSinkhorn call in this scoring pass.
            overlap=material_overlap(pair,refined.translation)
            penalty=0. if not overlap['available'] else overlap['fraction_min_area']
            readout=self.head.readout(final,penalty)
            scored.append(ScoredCluster(proposal,refined,final,readout,overlap,
                before if capture_diagnostics else None))
        if not scored:
            return PairPrediction(False,pair.numeric_valid,-1,None,pair.q.new_zeros(()),False,
                {'underconstrained':True,'reason':'no_candidate'},(),proposals)
        index=int(torch.stack([r.readout.logit for r in scored]).detach().argmax())
        winner=scored[index]
        numeric=pair.numeric_valid and bool(torch.isfinite(winner.translation).all() & torch.isfinite(winner.readout.score))
        return PairPrediction(True,numeric,index,winner.translation,winner.readout.score,
            bool(numeric and winner.readout.score.detach()>=threshold),
            dict(underconstrained=winner.refinement.underconstrained,
                 curvature_inverse_eigenvalues=winner.refinement.uncertainty_eigenvalues.detach().tolist(),
                 note='geometric observability proxy, not calibrated error probability'),tuple(scored),proposals)

    def forward(self,mask_a,mask_b,points_rc_a,points_rc_b,contour_valid_a,contour_valid_b,threshold=.5,
                capture_diagnostics=False):
        evidence=self.matcher(mask_a,mask_b,points_rc_a,points_rc_b,contour_valid_a,contour_valid_b)
        pairs=[PairEvidence.from_matcher(evidence,i,mask_a,mask_b) for i in range(len(mask_a))]
        return [self.score_pair(p,threshold,capture_diagnostics=capture_diagnostics) for p in pairs]
