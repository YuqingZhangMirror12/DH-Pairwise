from dataclasses import replace
import unittest
import torch

from .compatibility import CompatibilityConfig
from .consensus_head import ConsensusEvidenceHead, MeasureAttention
from .evidence import PairEvidence, recall_full_q
from .geometry import compact_contour
from .model import S7Consensus
from .pose_consensus import PoseConsensusBuilder
from .test_evidence import fixture


class HeadModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.cfg=CompatibilityConfig(.5,.5,.5,.5,1.,30.)
        self.head=ConsensusEvidenceHead(feature_dim=4)

    def test_layer_independence_and_side_parameter_sharing(self):
        self.assertEqual(len(self.head.layers),2)
        self.assertEqual(self.head.layers[0].cross_attention.heads,4)
        self.assertIsNot(self.head.layers[0].cross_attention.q.weight,
                         self.head.layers[1].cross_attention.q.weight)
        evidence=recall_full_q(fixture(),torch.zeros(2),self.cfg)
        out=self.head(evidence)
        self.assertEqual(out.a.local_probabilities.shape,(5,3))
        self.assertTrue((out.a.localization_reliability>=0).all())
        self.assertTrue((out.a.localization_reliability<=1).all())

    def test_attention_duplicate_measure_not_duplicate_votes(self):
        layer=MeasureAttention(8,2).eval()
        q=torch.randn(3,8);k=torch.randn(4,8)
        validq=torch.ones(3,dtype=torch.bool);validk=torch.ones(4,dtype=torch.bool)
        measure=torch.tensor([1.,2.,3.,4.])
        a=layer(q,k,validq,validk,measure)
        b=layer(q,k.repeat_interleave(3,dim=0),validq,validk.repeat_interleave(3),measure.repeat_interleave(3)/3)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)

    def test_no_keys_outputs_finite_zeros(self):
        layer=MeasureAttention(8,2)
        q=torch.zeros(3,8);k=torch.zeros(4,8)
        out=layer(q,k,torch.ones(3,dtype=torch.bool),torch.zeros(4,dtype=torch.bool),torch.zeros(4))
        self.assertEqual(float(out.abs().sum()),0.)

    def test_monotone_readout_at_fixed_local_evidence(self):
        evidence=recall_full_q(fixture(),torch.zeros(2),self.cfg)
        out=self.head(evidence)
        one=torch.zeros_like(out.a.local_probabilities);one[:,0]=1.
        out=replace(out,a=replace(out.a,local_probabilities=one),b=replace(out.b,local_probabilities=one))
        initial=self.head.readout(out,0.)
        expanded=replace(out,evidence=replace(evidence,
            a=replace(evidence.a,observed_arc_px=evidence.a.observed_arc_px*2),
            b=replace(evidence.b,observed_arc_px=evidence.b.observed_arc_px*2)))
        self.assertGreater(float(self.head.readout(expanded,0.).logit),float(initial.logit))
        self.assertLess(float(self.head.readout(out,.3).logit),float(initial.logit))

    def test_cyclic_origin_equivariance_fixed_features_and_q(self):
        p=fixture();ia=torch.arange(5).roll(2);ib=torch.arange(5).roll(3)
        ga=compact_contour(p.points_a[ia][None],torch.ones(1,5,dtype=torch.bool))
        gb=compact_contour(p.points_b[ib][None],torch.ones(1,5,dtype=torch.bool))
        other=PairEvidence(p.local_a[ia],p.local_b[ib],p.context_a[ia],p.context_b[ib],p.q[ia][:,ib],
            p.unmatched_a[ia],p.unmatched_b[ib],ga,gb,p.original_a[ia],p.original_b[ib])
        a=self.head.readout(self.head(recall_full_q(p,torch.zeros(2),self.cfg)),0.)
        b=self.head.readout(self.head(recall_full_q(other,torch.zeros(2),self.cfg)),0.)
        torch.testing.assert_close(a.logit,b.logit,atol=3e-6,rtol=3e-6)

    def test_exchange_invariance_and_vector_translation_flip(self):
        p=fixture()
        other=PairEvidence(p.local_b,p.local_a,p.context_b,p.context_a,p.q.T,
            p.unmatched_b,p.unmatched_a,p.gb,p.ga,p.original_b,p.original_a)
        model=S7Consensus(None,self.cfg,head=self.head)
        a=model.score_pair(p)
        b=model.score_pair(other)
        self.assertTrue(a.has_candidate and b.has_candidate)
        torch.testing.assert_close(a.score,b.score,atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(a.translation_a_to_b_rc,-b.translation_a_to_b_rc,atol=3e-5,rtol=3e-5)

    def test_final_position_is_exact_score_input_and_gradients_finite(self):
        p=fixture()
        model=S7Consensus(None,self.cfg,head=self.head)
        prediction=model.score_pair(p)
        self.assertTrue(prediction.has_candidate)
        for c in prediction.clusters:
            self.assertIs(c.translation,c.encoded.evidence.pose)
            direct=self.head.readout(self.head(recall_full_q(p,c.translation,self.cfg)),0.)
            torch.testing.assert_close(c.readout.logit,direct.logit)
        loss=sum(c.readout.logit for c in prediction.clusters)
        loss.backward()
        self.assertTrue(self.head.input[0].weight.grad.abs().sum()>0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.head.parameters()))

    def test_empty_q_returns_no_fake_layout(self):
        p=fixture(torch.zeros(5,5))
        prediction=S7Consensus(None,self.cfg,head=self.head).score_pair(p)
        self.assertFalse(prediction.has_candidate)
        self.assertFalse(prediction.accepted)
        self.assertIsNone(prediction.translation_a_to_b_rc)

    def test_representational_duplication_full_head_fixed_observations(self):
        evidence=recall_full_q(fixture(),torch.zeros(2),self.cfg)
        def duplicated(side):
            from dataclasses import fields
            values={f.name:getattr(side,f.name).repeat_interleave(2,0) for f in fields(side)}
            # A duplicate record is the SAME physical patch, not a new denser
            # physical sample. Partition only its integration measure.
            values['observed_arc_px']=values['observed_arc_px']/2
            return replace(side,**values)
        extra=replace(evidence,a=duplicated(evidence.a),b=duplicated(evidence.b),
            weights=evidence.weights.repeat_interleave(2,0).repeat_interleave(2,1)/2,
            kernels=evidence.kernels.repeat_interleave(2,0).repeat_interleave(2,1),
            localization_kernels=evidence.localization_kernels.repeat_interleave(2,0).repeat_interleave(2,1))
        a=self.head.readout(self.head(evidence),0.)
        b=self.head.readout(self.head(extra),0.)
        torch.testing.assert_close(a.logit,b.logit,atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(a.positive_evidence_px,b.positive_evidence_px,atol=3e-6,rtol=3e-6)

    def test_packed_candidate_batch_matches_dense_outputs_and_gradients(self):
        import copy
        other=copy.deepcopy(self.head)
        p=fixture()
        evidences=[recall_full_q(p,torch.tensor(t),self.cfg) for t in ([0.,0.],[2.,3.],[300.,300.])]
        dense=[self.head.dense_reference(e) for e in evidences]
        packed=other.forward_many(evidences)
        for a,b in zip(dense,packed):
            for side in 'ab':
                valid=getattr(a.evidence,side).valid
                for name in ('state','local_probabilities','localization_reliability'):
                    torch.testing.assert_close(getattr(getattr(a,side),name)[valid],
                        getattr(getattr(b,side),name)[valid],atol=3e-6,rtol=3e-6)
            torch.testing.assert_close(self.head.readout(a,0.).logit,other.readout(b,0.).logit,atol=3e-6,rtol=3e-6)
        sum(self.head.readout(a,0.).logit for a in dense).backward()
        sum(other.readout(b,0.).logit for b in packed).backward()
        for a,b in zip(self.head.parameters(),other.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,atol=5e-6,rtol=5e-5)


if __name__=='__main__':
    unittest.main()
