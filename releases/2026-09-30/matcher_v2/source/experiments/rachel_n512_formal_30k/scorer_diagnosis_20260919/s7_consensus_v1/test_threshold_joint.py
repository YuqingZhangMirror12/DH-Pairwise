"""Actual Attention/refinement/loss path, not a union-metadata-only test."""
from dataclasses import replace
import unittest
import torch

from .compatibility import CompatibilityConfig
from .consensus_head import ConsensusEvidenceHead
from .evidence import PairEvidence, recall_full_q
from .geometry import compact_contour
from .losses import pair_loss
from .model import S7Consensus
from .targets import PairLabels
from .test_consensus import make_cloud
from .test_threshold import native_groups
from .threshold_builder import ThresholdPoseBuilder
from .threshold_evidence import recall_union


def setup_pair():
    torch.manual_seed(71)
    geometry=CompatibilityConfig(.5,.5,.5,.5,1.,9.)
    cloud=make_cloud([0,4,10,15,80]);n=len(cloud.q)
    pa=torch.stack((cloud.arc_a,torch.zeros(n)),1);pb=pa+cloud.displacement
    valid=torch.ones(1,n,dtype=torch.bool)
    ga=replace(compact_contour(pa[None],valid),cell_px=cloud.spacing_a[None],
        outward_normal_rc=cloud.normal_a[None],normal_reliability=cloud.reliability_a[None],
        arc_px=cloud.arc_a[None],perimeter_px=torch.tensor([cloud.perimeter_a]))
    gb=replace(compact_contour(pb[None],valid),cell_px=cloud.spacing_b[None],
        outward_normal_rc=cloud.normal_b[None],normal_reliability=cloud.reliability_b[None],
        arc_px=cloud.arc_b[None],perimeter_px=torch.tensor([cloud.perimeter_b]))
    features=[torch.randn(n,4,requires_grad=True) for _ in range(4)]
    q=torch.diag(cloud.q)
    pair=PairEvidence(*features,q,1-q.sum(1),1-q.sum(0),ga,gb,torch.arange(n),torch.arange(n))
    hyps=native_groups(cloud,[[x,0] for x in [0,4,10,15,80]])
    proposals=ThresholdPoseBuilder(geometry).build_from_hypotheses(cloud,hyps)
    model=S7Consensus(None,geometry,head=ConsensusEvidenceHead(feature_dim=4))
    return model,pair,proposals,features


class ThresholdJointTests(unittest.TestCase):
    def test_union_all_groups_reach_initial_final_attention_and_refinement(self):
        model,pair,proposals,features=setup_pair()
        prediction=model.score_pair(pair,proposals=proposals,capture_diagnostics=True)
        near=min(prediction.clusters,key=lambda c:float(c.translation.norm()))
        self.assertEqual(len(near.proposal.edge_ids),16)
        for start in (0,4,8,12):
            ids=torch.arange(start,start+4)
            for evidence in [near.initial_encoded.evidence,near.encoded.evidence]:
                torch.testing.assert_close(evidence.weights[ids,ids],pair.q[ids,ids])
                self.assertTrue((evidence.kernels[ids,ids]==1).all())
                self.assertTrue(evidence.a.valid[ids].all())
            self.assertTrue((near.refinement.localization_weights[ids,ids]+near.refinement.compatibility_weights[ids,ids]>0).all())
        self.assertEqual(float(near.encoded.evidence.weights[16:,16:].sum()),0.)
        self.assertIs(near.translation,near.encoded.evidence.pose)
        gradients=torch.autograd.grad(near.readout.logit,features,retain_graph=True)
        for start in (0,4,8,12):
            self.assertGreater(sum(float(g[start:start+4].abs().sum()) for g in gradients),1e-8)

    def test_members_cannot_be_suppressed_by_old_directional_admission(self):
        model,pair,proposals,_=setup_pair();c=proposals.clusters[0]
        old=recall_full_q(pair,c.translation,model.geometry)
        new=model.recall_cluster(pair,c,c.translation)
        ids=c.edge_ids
        self.assertTrue((new.weights[ids[:,0],ids[:,1]]==pair.q[ids[:,0],ids[:,1]]).all())
        self.assertLess(float(old.weights.sum()),float(new.weights.sum()))
        torch.testing.assert_close(new.a.unmatched,pair.unmatched_a)
        torch.testing.assert_close(new.a.other_mass,(pair.q.sum(1)-new.weights.sum(1)).clamp_min(0))

    def test_duplicate_union_does_not_change_features_mass_or_score(self):
        model,pair,proposals,_=setup_pair();c=proposals.clusters[0]
        a=model.recall_cluster(pair,c,c.translation)
        b=recall_union(pair,c.translation,model.geometry,c.edge_ids.repeat(4,1))
        torch.testing.assert_close(a.weights,b.weights)
        torch.testing.assert_close(a.a.scalar_features,b.a.scalar_features)
        torch.testing.assert_close(model.head.readout(model.head(a),0).score,model.head.readout(model.head(b),0).score)

    def test_finite_backward_candidate_local_extension_losses(self):
        model,pair,proposals,features=setup_pair()
        prediction=model.score_pair(pair,proposals=proposals,capture_diagnostics=True)
        n=len(pair.q)
        labels=PairLabels(True,True,torch.tensor([7.,0.]),torch.arange(n),torch.arange(n),
                         torch.ones(n,dtype=torch.bool),torch.ones(n,dtype=torch.bool))
        loss=pair_loss(model,prediction,labels,include_extension=True)
        self.assertEqual(loss.counts['correct_clusters'],1)
        self.assertEqual(loss.counts['wrong_clusters'],1)
        self.assertEqual(loss.counts['extension_pairs'],1)
        loss.total.backward()
        for p in list(model.head.parameters())+features:
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())

    def test_final_score_recomputed_at_same_final_pose_and_same_union(self):
        model,pair,proposals,_=setup_pair()
        prediction=model.score_pair(pair,proposals=proposals,capture_diagnostics=True)
        for c in prediction.clusters:
            repeated=model.recall_cluster(pair,c.proposal,c.translation)
            torch.testing.assert_close(c.readout.logit,model.head.readout(model.head(repeated),0.).logit)
            torch.testing.assert_close(c.initial_encoded.evidence.weights,c.encoded.evidence.weights)
            self.assertTrue(((c.translation-c.proposal.member_translations_rc).norm(dim=1)<=16.0001).all())

    def test_no_candidate_and_negative_labels_remain_negative(self):
        model,pair,proposals,_=setup_pair()
        prediction=model.score_pair(pair,proposals=replace(proposals,clusters=()))
        self.assertFalse(prediction.has_candidate)
        n=len(pair.q)
        labels=PairLabels(False,False,torch.zeros(2),torch.full((n,),-1),torch.full((n,),-1),
                         torch.zeros(n,dtype=torch.bool),torch.zeros(n,dtype=torch.bool))
        prediction=model.score_pair(pair,proposals=proposals)
        loss=pair_loss(model,prediction,labels)
        self.assertEqual(loss.counts['correct_clusters'],0)
        self.assertEqual(loss.counts['wrong_clusters'],2)


if __name__=='__main__':unittest.main()
