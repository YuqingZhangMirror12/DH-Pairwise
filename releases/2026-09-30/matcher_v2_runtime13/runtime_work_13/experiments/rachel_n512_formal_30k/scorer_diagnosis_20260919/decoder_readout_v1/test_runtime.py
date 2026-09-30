"""Synthetic CPU validation of new heads, loss and update1 recovery."""
from copy import deepcopy
from dataclasses import replace
import io
import unittest
import torch
from .head import EvidenceClusterHead
from ..binary_scorer_v1.model import BinaryConsensus
from ..binary_scorer_v1.loss import pair_loss
from ..s7_consensus_v1.test_threshold_joint import setup_pair
from ..s7_consensus_v1.targets import PairLabels
from ..s7_consensus_v1 import train


def fixture(variant='patch_mean'):
    old,pair,proposals,features=setup_pair()
    model=BinaryConsensus(None,old.geometry,head=EvidenceClusterHead(variant,feature_dim=4,remove_overlap=False))
    return model,pair,proposals,features


class RuntimeTests(unittest.TestCase):
    def test_training_factory_and_exact_config(self):
        for v in ('patch_mean','patch_sum'):
            h=train.fresh_head(26092406,v);cfg=train.TrainingConfig(scorer_variant=v)
            self.assertEqual(sum(p.numel() for p in h.parameters()),32481)
            self.assertFalse(h.remove_overlap)
            self.assertEqual((cfg.microbatch,cfg.world_size,cfg.accumulate,cfg.effective_batch),(32,1,1,32))
            self.assertEqual(cfg.record()['head']['cluster_mlp'],[48,64,32,1])
            self.assertEqual(cfg.record()['threshold_policy']['pose_diameter_px'],16.)
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(h.parameters(),train.fresh_head(26092406,v).parameters())))

    def test_nonzero_loss_and_finite_gradients(self):
        for v in ('patch_mean','patch_sum'):
            model,pair,proposals,_=fixture(v);n=len(pair.q)
            labels=PairLabels(True,True,torch.tensor([7.,0.]),torch.arange(n),torch.arange(n),
                              torch.ones(n,dtype=torch.bool),torch.ones(n,dtype=torch.bool))
            loss=pair_loss(model,model.score_pair(pair,proposals=proposals),labels)
            self.assertEqual(loss.counts['correct_clusters'],1);self.assertEqual(loss.counts['wrong_clusters'],1)
            loss.total.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters()))

    def test_duplicate_union_and_order_and_same_search(self):
        for v in ('patch_mean','patch_sum'):
            model,pair,proposals,_=fixture(v);p=proposals.clusters[0]
            a=model.head(pair,p);b=model.head(pair,replace(p,edge_ids=p.edge_ids.flip(0).repeat(3,1)))
            torch.testing.assert_close(a.logit,b.logit,rtol=0,atol=0)
            torch.testing.assert_close(a.inputs.pose,p.translation)
            torch.testing.assert_close(a.inputs.q,pair.q[a.inputs.edge_ids[:,0],a.inputs.edge_ids[:,1]])
            self.assertEqual(model.builder.config.row_column_topk,2)

    def test_serialized_adamw_recovery_exact(self):
        for v in ('patch_mean','patch_sum'):
            model,pair,proposals,_=fixture(v)
            pair=replace(pair,q=pair.q.detach(),local_a=pair.local_a.detach(),local_b=pair.local_b.detach(),
                         context_a=pair.context_a.detach(),context_b=pair.context_b.detach())
            opt=torch.optim.AdamW(model.head.parameters(),lr=1e-4)
            def step(m,o):
                o.zero_grad();loss=sum(c.readout.logit.square() for c in m.score_pair(pair,proposals=proposals).clusters)
                loss.backward();o.step()
            step(model,opt);buf=io.BytesIO();torch.save((model.state_dict(),opt.state_dict()),buf);buf.seek(0)
            for _ in range(11):step(model,opt)
            expected=deepcopy(model.state_dict());state,optimizer=torch.load(buf,weights_only=False)
            model.load_state_dict(state);opt.load_state_dict(optimizer)
            for _ in range(11):step(model,opt)
            self.assertTrue(all(torch.equal(expected[k],x) for k,x in model.state_dict().items()))


if __name__=='__main__':unittest.main()
