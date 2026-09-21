"""Focused synthetic evidence tests: no dataset, trained weights or network."""
import unittest
import importlib.util

import numpy as np
import torch

from ..matched_only import model as old
from ..matched_only.test_model import fixture
from .model import Config, LocalEvidenceScorer, batched_readout


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def net(self, **kwargs):
        torch.manual_seed(47)
        return LocalEvidenceScorer(Config(dim=8, heads=4, **kwargs)).eval()

    def test_reference_same_logits_and_gradients_as_old(self):
        a,b,va,vb,s,meta,_ = fixture()
        new = self.net()
        ref = old.FreshMatchedScorer("matched_edges", 8, 4).eval()
        ref.load_state_dict(new.state_dict(), strict=True)
        x,y = new(a,b,va,vb,s,**meta).logit, ref(a,b,va,vb,s,**meta).logit
        torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        x.sum().backward(); y.sum().backward()
        for (_,p),(_,q) in zip(new.named_parameters(),ref.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4)

    def test_padded_batch_matches_individual(self):
        net = self.net()
        g = torch.Generator().manual_seed(2)
        a,b = torch.randn(3,9,8,generator=g),torch.randn(3,9,8,generator=g)
        v = torch.arange(9)[None] < torch.tensor([[3],[7],[0]])
        x,_ = batched_readout(net.head,a,b,v)
        y = net.head(a,b,v,v)
        torch.testing.assert_close(x,y,atol=3e-6,rtol=2e-5)

    def test_stable_finite_and_candidate_aux(self):
        a,b,va,vb,s,meta,_ = fixture()
        for kwargs in (dict(stable=True),dict(joint_d=True),dict(cap=128),dict(cap=256)):
            net = self.net(**kwargs)
            result=net(a,b,va,vb,s,**meta)
            self.assertTrue(torch.isfinite(result.logit).all())
            self.assertLessEqual(int(result.used_edge_count.max()), net.config_v2.cap)
            if kwargs.get("joint_d"):
                self.assertTrue(torch.isfinite(result.candidate_logit).all())
            result.logit.sum().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_raw_no_gt_model_api(self):
        a,b,va,vb,s,meta,_ = fixture()
        net = self.net()
        self.assertEqual(net.config_v2.depth,2)
        before=net(a,b,va,vb,s,**meta).logit
        outside = a.clone(); outside[~s.mask_a] = 10000
        torch.testing.assert_close(before,net(outside,b,va,vb,s,**meta).logit)

    def test_no_evidence_stays_finite(self):
        a,b,va,vb,s,meta,q = fixture()
        s=old.select_predicted_inliers(torch.zeros_like(q),meta['points_a_rc'],meta['points_b_rc'],va,vb)
        meta['candidate_weights']=torch.zeros_like(meta['candidate_weights'])
        for kwargs in ({},dict(stable=True),dict(joint_d=True)):
            net=self.net(**kwargs); result=net(a,b,va,vb,s,**meta)
            self.assertTrue(result.used_fallback.all())
            self.assertTrue(torch.isfinite(result.logit).all())

    def check_graph(self, graph):
        a,b,va,vb,s,meta,_ = fixture()
        net = self.net(graph=graph)
        result = net(a,b,va,vb,s,**meta)
        self.assertTrue(torch.isfinite(result.logit).all())
        result.logit.sum().backward()
        gradients = [p.grad for p in net.graph.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_pairing_graph_forward_and_gradient(self):
        self.check_graph("pairing")

    @unittest.skipUnless(importlib.util.find_spec("torch_geometric"), "optional torch-geometric is not installed")
    def test_shredding_graph_forward_and_gradient(self):
        self.check_graph("shredding")


if __name__ == '__main__':
    unittest.main()
