"""Behavioral test: a real union, not merely duplicate-candidate suppression."""
from dataclasses import replace
import unittest
import torch

from .compatibility import CompatibilityConfig
from .consensus_head import ConsensusEvidenceHead
from .evidence import PairEvidence,recall_full_q
from .geometry import compact_contour
from .model import S7Consensus
from .pose_refinement import refine_pose
from .pose_consensus_repair import RepairedPoseConsensusBuilder
from .test_consensus import make_cloud


class JointMergedEvidenceTests(unittest.TestCase):
    def test_each_disjoint_group_reaches_joint_attention_and_pose_fit(self):
        torch.manual_seed(71)
        geometry=CompatibilityConfig(.5,.5,.5,.5,1.,9.)
        cloud=make_cloud([1,3,5,10,210]);n=len(cloud.q)
        pa=torch.stack((cloud.arc_a,torch.zeros(n)),1)
        pb=pa+cloud.displacement
        valid=torch.ones(1,n,dtype=torch.bool)
        # Known synthetic local frames: four disjoint tangential evidence
        # groups and a fifth 200px-away competing pose. No GT enters the model.
        ga=replace(compact_contour(pa[None],valid),cell_px=cloud.spacing_a[None],
            outward_normal_rc=cloud.normal_a[None],normal_reliability=cloud.reliability_a[None],
            arc_px=cloud.arc_a[None],perimeter_px=torch.tensor([cloud.perimeter_a]))
        gb=replace(compact_contour(pb[None],valid),cell_px=cloud.spacing_b[None],
            outward_normal_rc=cloud.normal_b[None],normal_reliability=cloud.reliability_b[None],
            arc_px=cloud.arc_b[None],perimeter_px=torch.tensor([cloud.perimeter_b]))
        features=[torch.randn(n,4,requires_grad=True) for _ in range(4)]
        q=torch.diag(cloud.q)
        pair=PairEvidence(*features,q,1-q.sum(1),1-q.sum(0),ga,gb,torch.arange(n),torch.arange(n))
        builder=RepairedPoseConsensusBuilder(geometry)
        seeds=torch.tensor([[1.,0.],[3.,0.],[5.,0.],[10.,0.],[210.,0.]])
        proposals=builder.build_from_cloud(cloud,seeds)
        model=S7Consensus(None,geometry,head=ConsensusEvidenceHead(feature_dim=4))
        out=model.score_pair(pair,proposals=proposals,capture_diagnostics=True)
        self.assertEqual(len(out.clusters),2)
        near=min(out.clusters,key=lambda c:float(c.translation.norm()))
        self.assertEqual(set(near.proposal.merged_hypothesis_ids),{0,1,2,3})
        self.assertEqual(set(near.proposal.original_union_edge_ids[:,0].tolist()),set(range(16)))
        self.assertIs(near.translation,near.encoded.evidence.pose)
        for start in (0,4,8,12):
            ids=torch.arange(start,start+4)
            # Every group participates in initial attention, joint refinement,
            # and the final-pose evidence, not just metadata on the candidate.
            self.assertTrue((near.initial_encoded.evidence.weights[ids,ids]>0).all())
            self.assertTrue((near.refinement.localization_weights[ids,ids]+
                             near.refinement.compatibility_weights[ids,ids]>0).all())
            self.assertTrue((near.encoded.evidence.weights[ids,ids]>0).all())
        self.assertEqual(float(near.encoded.evidence.weights[16:,16:].sum()),0.)
        grads=torch.autograd.grad(near.readout.logit,features,retain_graph=True)
        for start in (0,4,8,12):
            self.assertGreater(sum(float(g[start:start+4].abs().sum()) for g in grads),1e-8)
        # Removing any complete supporting group at the SAME initial pose must
        # change the joint fit. It is not "fit best subcandidate, pool scores".
        for start in (0,4,8,12):
            changed=q.clone();changed[start:start+4,:]=0
            reduced=replace(pair,q=changed,unmatched_a=1-changed.sum(1),unmatched_b=1-changed.sum(0))
            encoded=model.head(recall_full_q(reduced,near.proposal.translation,geometry))
            translated=refine_pose(encoded,geometry).translation
            self.assertGreater(float((translated-near.translation).norm()),1e-5)
        torch.testing.assert_close(near.readout.logit,model.head.readout(
            model.head(recall_full_q(pair,near.translation,geometry)),0.).logit)


if __name__=='__main__':unittest.main()
