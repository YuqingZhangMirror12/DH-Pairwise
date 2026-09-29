from dataclasses import replace
import unittest
import torch
from .head import BinaryClusterHead, inputs_for_cluster
from .model import BinaryConsensus
from .loss import pair_loss,batch_loss
from ..s7_consensus_v1.test_threshold_joint import setup_pair
from ..s7_consensus_v1.targets import PairLabels


def fixture(variant='patch'):
    old,pair,proposals,features=setup_pair()
    model=BinaryConsensus(None,old.geometry,head=BinaryClusterHead(variant,feature_dim=4))
    return model,pair,proposals,features


class BinaryTests(unittest.TestCase):
    def test_deduplicated_union_preserves_each_q_and_absolute_mass(self):
        model,pair,proposals,_=fixture();p=proposals.clusters[0]
        a=model.head(pair,p);b=model.head(pair,replace(p,edge_ids=p.edge_ids.repeat(3,1)))
        torch.testing.assert_close(a.logit,b.logit)
        torch.testing.assert_close(a.inputs.q,pair.q[p.edge_ids[:,0],p.edge_ids[:,1]])
        self.assertEqual(len(a.inputs.q),len(p.edge_ids))
        self.assertAlmostEqual(float(a.inputs.normalized_weights.sum()),1.,6)
        self.assertNotAlmostEqual(float(a.inputs.mass_weights.sum()),1.,3)

    def test_geometry_arm_never_reads_patch_or_context(self):
        model,pair,proposals,_=fixture('stats');p=proposals.clusters[0]
        a=model.head(pair,p)
        broken=replace(pair,local_a=None,local_b=None,context_a=None,context_b=None)
        b=model.head(broken,p)
        torch.testing.assert_close(a.logit,b.logit)
        self.assertIsNone(a.inputs.patch_context);self.assertIsNone(model.head.edge_mlp)

    def test_features_reach_patch_arm_all_member_fragments(self):
        model,pair,proposals,features=fixture();c=model.head(pair,proposals.clusters[0])
        grad=torch.autograd.grad(c.logit,features)
        for start in (0,4,8,12):
            self.assertGreater(sum(float(g[start:start+4].abs().sum()) for g in grad),0)

    def test_both_arms_share_raw_union_and_builder_pose(self):
        model,pair,proposals,_=fixture();other=BinaryConsensus(None,model.geometry,head=BinaryClusterHead('stats',4))
        a=model.score_pair(pair,proposals=proposals);b=other.score_pair(pair,proposals=proposals)
        for x,y,p in zip(a.clusters,b.clusters,proposals.clusters):
            torch.testing.assert_close(x.translation,p.translation)
            torch.testing.assert_close(x.translation,y.translation)
            torch.testing.assert_close(x.readout.inputs.q,y.readout.inputs.q)
            torch.testing.assert_close(x.readout.inputs.statistics,y.readout.inputs.statistics)

    def test_no_attention_local_classes_or_learned_refinement(self):
        model,_,_,_=fixture()
        self.assertFalse(any(isinstance(x,torch.nn.MultiheadAttention) for x in model.modules()))
        self.assertFalse(any('conflict' in k or 'localizer' in k for k in model.state_dict()))
        self.assertEqual(model.head.cluster_mlp[-1].out_features,1)

    def test_loss_wrong_pose_not_inherit_positive_pair(self):
        model,pair,proposals,_=fixture();n=len(pair.q)
        labels=PairLabels(True,True,torch.tensor([7.,0.]),torch.arange(n),torch.arange(n),
            torch.ones(n,dtype=torch.bool),torch.ones(n,dtype=torch.bool))
        pred=model.score_pair(pair,proposals=proposals);loss=pair_loss(model,pred,labels)
        self.assertEqual(loss.counts['correct_clusters'],1);self.assertEqual(loss.counts['wrong_clusters'],1)
        self.assertEqual(set(loss.components),{'candidate','ranking'})
        loss.total.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters()))

    def test_no_candidate_counts_coverage_miss_no_fabricated_pose(self):
        model,pair,proposals,_=fixture();n=len(pair.q)
        labels=PairLabels(True,True,torch.zeros(2),torch.arange(n),torch.arange(n),
            torch.ones(n,dtype=torch.bool),torch.ones(n,dtype=torch.bool))
        pred=model.score_pair(pair,proposals=replace(proposals,clusters=()))
        loss=pair_loss(model,pred,labels);self.assertEqual(loss.counts['positive_coverage_misses'],1)
        self.assertIsNone(pred.translation_a_to_b_rc);self.assertEqual(float(loss.total),0.)

    def test_unlabelled_layout_is_not_negative_cluster_label(self):
        model,pair,proposals,_=fixture();n=len(pair.q)
        labels=PairLabels(True,False,torch.zeros(2),torch.arange(n),torch.arange(n),
            torch.zeros(n,dtype=torch.bool),torch.zeros(n,dtype=torch.bool))
        loss=pair_loss(model,model.score_pair(pair,proposals=proposals),labels)
        self.assertEqual(loss.counts['unknown_clusters'],2);self.assertEqual(float(loss.total),0.)

    def test_fragment_exchange_invariant(self):
        for variant in ('patch','stats'):
            model,pair,proposals,_=fixture(variant);p=proposals.clusters[0]
            swapped=replace(pair,local_a=pair.local_b,local_b=pair.local_a,context_a=pair.context_b,
                context_b=pair.context_a,q=pair.q.T,unmatched_a=pair.unmatched_b,unmatched_b=pair.unmatched_a,
                ga=pair.gb,gb=pair.ga,original_a=pair.original_b,original_b=pair.original_a)
            sp=replace(p,translation=-p.translation,edge_ids=p.edge_ids.flip(1))
            torch.testing.assert_close(model.head(pair,p).logit,model.head(swapped,sp).logit)

    def test_absolute_q_scaling_not_lost_by_conditional_pooling(self):
        _,pair,proposals,_=fixture();p=proposals.clusters[0]
        a=inputs_for_cluster(pair,p,False);b=inputs_for_cluster(replace(pair,q=pair.q*.1),p,False)
        torch.testing.assert_close(a.normalized_weights,b.normalized_weights)
        self.assertGreater(float(a.statistics[1]),float(b.statistics[1]))
        self.assertGreater(float(a.statistics[2]),float(b.statistics[2]))

    def test_subnormal_q_finite_head_backward(self):
        model,pair,proposals,_=fixture('stats');pair=replace(pair,q=pair.q*1e-35)
        logit=model.head(pair,proposals.clusters[0]).logit;logit.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.head.parameters()))

    def test_bad_q_rejected(self):
        model,pair,proposals,_=fixture();q=pair.q.clone();q[0,0]=float('nan')
        with self.assertRaises(ValueError):model.head(replace(pair,q=q),proposals.clusters[0])

if __name__=='__main__':unittest.main()
