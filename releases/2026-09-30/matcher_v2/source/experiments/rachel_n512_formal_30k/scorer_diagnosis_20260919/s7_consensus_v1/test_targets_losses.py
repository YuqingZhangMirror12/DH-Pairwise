import unittest
from dataclasses import replace
from types import SimpleNamespace
import torch

from .compatibility import CompatibilityConfig
from .consensus_head import ConsensusEvidenceHead
from .evidence import recall_full_q
from .model import S7Consensus,PairPrediction
from .targets import PairLabels,candidate_quality,local_targets,local_distributions
from .losses import pair_loss,batch_loss
from .test_evidence import fixture


def labels(positive=True,known=True):
    a=torch.tensor([-1,-2,4,-2,-1]);b=torch.tensor([-2,-1,-1,-2,2])
    return PairLabels(positive,known,torch.zeros(2),a,b,a>=0,b>=0)


class TargetsLossTests(unittest.TestCase):
    def setUp(self):
        self.cfg=CompatibilityConfig(.5,.5,.5,.5,1.,30.)
        self.model=S7Consensus(None,self.cfg,head=ConsensusEvidenceHead(feature_dim=4))

    def test_wrong_pose_on_positive_pair_is_negative(self):
        y,k=candidate_quality([torch.tensor([2.,0.]),torch.tensor([200.,0.])],labels())
        torch.testing.assert_close(y,torch.tensor([1.,0.]))
        self.assertTrue(k.all())

    def test_unknown_pose_not_fabricated_and_negatives_valid(self):
        y,k=candidate_quality([torch.zeros(2)],labels(known=False))
        self.assertFalse(k.any())
        y,k=candidate_quality([torch.zeros(2)],labels(positive=False,known=False))
        self.assertTrue(k.all());self.assertEqual(float(y[0]),0.)

    def test_artificial_cut_unknown_never_becomes_source_support(self):
        p=fixture();truth=local_targets(p,labels())
        self.assertEqual(int(truth.source_support_known.sum()),1)
        self.assertFalse(truth.edge_target_known[1].any())
        self.assertFalse(truth.edge_target_known[:,3].any())
        self.assertTrue(truth.precise_anchor_known[2,4])

    def test_corroded_support_not_automatically_precise_anchor(self):
        p=fixture();l=labels()
        l=replace(l,precise_anchor_a=torch.zeros(5,dtype=torch.bool),precise_anchor_b=torch.zeros(5,dtype=torch.bool))
        t=local_targets(p,l)
        self.assertEqual(int(t.source_support_known.sum()),1)
        self.assertEqual(int(t.precise_anchor_known.sum()),0)
        view=recall_full_q(p,torch.zeros(2),self.cfg)
        local=local_distributions(view,t,'a')
        self.assertEqual(float(local['location_weight'][2]),0.)

    def test_no_candidate_does_not_change_pair_to_negative(self):
        p=fixture(torch.zeros(5,5));prediction=self.model.score_pair(p)
        l=labels();loss=pair_loss(self.model,prediction,l)
        self.assertEqual(float(loss.total),0.)
        self.assertEqual(loss.counts['positive_coverage_misses'],1)
        self.assertEqual(loss.counts['wrong_clusters'],0)
        self.assertTrue(l.label)

    def test_pair_normalization_is_independent_of_microbatch_partition(self):
        p=fixture();prediction=self.model.score_pair(p)
        la,lb=labels(),labels(positive=False)
        combined,items=batch_loss(self.model,[prediction,prediction],[la,lb])
        one=pair_loss(self.model,prediction,la).total
        two=pair_loss(self.model,prediction,lb).total
        torch.testing.assert_close(combined,(one+two)/2)
        combined.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))


if __name__=='__main__':
    unittest.main()
