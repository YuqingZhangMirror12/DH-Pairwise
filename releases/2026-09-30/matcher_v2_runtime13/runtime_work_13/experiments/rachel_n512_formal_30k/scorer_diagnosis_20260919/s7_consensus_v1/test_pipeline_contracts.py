from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from .compatibility import CompatibilityConfig
from .consensus_head import ConsensusEvidenceHead
from .evidence import PairEvidence,recall_full_q
from .model import S7Consensus
from .pose_refinement import refine_pose
from .test_evidence import fixture


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6)
        self.cfg=CompatibilityConfig(.5,.5,.5,.5,1.,30.)
        self.head=ConsensusEvidenceHead(feature_dim=4)

    def test_fixed_matcher_evidence_padding_and_batch_composition(self):
        # The checkpoint's old Context is NOT changed here. This gate concerns
        # the new downstream modules at identical physical Matcher outputs.
        pair=fixture()
        def packed(width,batch,index):
            out={}
            for name in ('local_a','local_b','context_a','context_b'):
                x=torch.full((batch,width,4),float('nan'))
                x[index,:5]=getattr(pair,name);out[name]=x
            for side in 'ab':
                x=torch.full((batch,width,2),float('nan'))
                x[index,:5]=getattr(pair,'points_'+side)
                out['points_rc_'+side]=x
                valid=torch.zeros(batch,width,dtype=torch.bool);valid[index,:5]=True
                out['valid_'+side]=valid
                u=torch.full((batch,width),float('nan'))
                u[index,:5]=getattr(pair,'unmatched_'+side);out['unmatched_'+side]=u
            q=torch.full((batch,width,width),float('nan'));q[index,:5,:5]=pair.q
            out['assignment']=q;out['numeric_valid']=torch.ones(batch,dtype=torch.bool)
            return PairEvidence.from_matcher(SimpleNamespace(**out),index)
        model=S7Consensus(None,self.cfg,head=self.head)
        a=model.score_pair(packed(5,1,0));b=model.score_pair(packed(16,3,2))
        torch.testing.assert_close(a.score,b.score)
        torch.testing.assert_close(a.translation_a_to_b_rc,b.translation_a_to_b_rc)
        self.assertEqual(len(a.clusters),len(b.clusters))

    def test_all_eroded_parallel_edges_do_not_claim_precise_normal_location(self):
        pair=fixture(torch.eye(5)*.4)
        n=torch.tensor([0.,1.]).repeat(1,5,1)
        t=torch.tensor([1.,0.]).repeat(1,5,1)
        ga=replace(pair.ga,outward_normal_rc=n,tangent_rc=t,normal_reliability=torch.ones(1,5),
            cell_px=torch.full((1,5),2.))
        gb=replace(ga,points=ga.points+torch.tensor([0.,15.]),outward_normal_rc=-n,tangent_rc=-t)
        pair=replace(pair,ga=ga,gb=gb)
        encoded=self.head(recall_full_q(pair,torch.zeros(2),self.cfg))
        refined=refine_pose(encoded,self.cfg)
        self.assertTrue(refined.underconstrained)
        self.assertGreater(float(refined.compatibility_weights.sum()),.1)
        self.assertLess(float(refined.localization_weights.sum()),1e-20)
        self.assertLess(abs(float(refined.translation[1])),1e-5)


if __name__=='__main__':
    unittest.main()
