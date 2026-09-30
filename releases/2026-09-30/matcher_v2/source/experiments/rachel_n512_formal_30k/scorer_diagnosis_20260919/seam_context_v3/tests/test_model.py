import unittest
import numpy as np
import torch
from dataclasses import replace
from ..config import Config
from ..valid_contour import compact
from ..primal_dual import PrimalDual
from ..model import SeamContextModel
from ..seam_proposals import open_arc_indices
from ..targets import visible_components
from ..beam_kernel import beam_paths


def fixture(n=12, storage=32, nan=False, shift=0., batch=1):
    theta = torch.arange(n)*2*torch.pi/n
    p = torch.stack((30+(.9+theta.sin()*.15)*15*theta.sin(),
                     30+(.9+theta.cos()*.1)*15*theta.cos()), -1)
    point = torch.full((batch, storage, 2), float('nan') if nan else 0.)
    valid = torch.zeros(batch, storage, dtype=torch.bool)
    point[:, :n] = p+shift; valid[:, :n] = True
    mask = torch.zeros(batch, 1, 64, 64); mask[:, :, 13:48, 13:48] = 1
    return mask, point, valid


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def model(self):
        torch.manual_seed(9)
        cfg = Config(canvas_size=64, contour_cap=512, arc_layers=1, correspondence_layers=1,
                     verifier_layers=1, neighbors=16, arc_bins=4, sinkhorn_iterations=100)
        return SeamContextModel(cfg)

    def test_primal_dual(self):
        pd = PrimalDual(12)
        self.assertNotEqual(pd.primal.weight.data_ptr(), pd.dual.weight.data_ptr())
        a, b = torch.randn(2, 11, 12), torch.randn(2, 7, 12)
        s, (pa, da, pb, db) = pd(a, b)
        torch.testing.assert_close(s, .5*(pa@db.transpose(1, 2)+da@pb.transpose(1, 2)))
        torch.testing.assert_close(pd(b, a)[0], s.transpose(1, 2))
        self.assertLessEqual(s.abs().max(), 1.)

    def test_padding_holes_roll_and_batch(self):
        m = self.model().eval()
        a, p, v = fixture()
        b, q, w = fixture(n=15, shift=2.)
        with torch.no_grad():
            o = m(a, b, p, q, v, w)
            _, pp, vv = fixture(storage=512, nan=True)
            _, qq, ww = fixture(n=15, storage=128, nan=True, shift=2.)
            # Move valid points into interior holes and cyclically roll input.
            pp = pp.roll(47, 1); vv = vv.roll(47, 1)
            qq = qq.roll(11, 1); ww = ww.roll(11, 1)
            other = m(a, b, pp, qq, vv, ww)
            torch.testing.assert_close(o.s1, other.s1, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(o.ot1.real_transport, other.ot1.real_transport, atol=1e-5, rtol=1e-4)
            batched = m(a.repeat(2,1,1,1), b.repeat(2,1,1,1), p.repeat(2,1,1), q.repeat(2,1,1),v.repeat(2,1),w.repeat(2,1))
            torch.testing.assert_close(o.s1[0], batched.s1[0], atol=1e-5, rtol=1e-4)

    def test_swap_and_nonzero_structural_delta(self):
        m = self.model().eval()
        torch.nn.init.normal_(m.correspondence_context.delta.weight, std=.02)
        a, p, v = fixture(n=24)
        b, q, w = fixture(n=19, shift=2.)
        with torch.no_grad():
            o = m(a,b,p,q,v,w,decode=True,verify=True)
            r = m(b,a,q,p,w,v,decode=True,verify=True)
        torch.testing.assert_close(o.s1, r.s1.transpose(1,2),atol=1e-5,rtol=1e-4)
        torch.testing.assert_close(o.verified[0].score,r.verified[0].score,atol=2e-4,rtol=1e-4)
        ta=o.verified[0].translations[o.verified[0].winner]
        tb=r.verified[0].translations[r.verified[0].winner]
        torch.testing.assert_close(ta,-tb,atol=1e-3,rtol=1e-4)

    def test_zero_delta_and_gradients(self):
        m=self.model(); a,p,v=fixture(); b,q,w=fixture(n=15,shift=2.)
        o=m(a,b,p,q,v,w,decode=True,verify=True)
        torch.testing.assert_close(o.s0,o.s1,atol=0,rtol=0)
        torch.testing.assert_close(o.ot0.real_transport,o.ot1.real_transport,atol=0,rtol=0)
        loss=o.s1.square().mean()+o.records[0].support.square().mean()+o.records[0].links.square().mean()+o.verified[0].logits.square().mean()
        loss.backward()
        for name in ('primal_dual.primal.weight','primal_dual.dual.weight','patch_encoder.projection.2.weight','arc_context.position.weight'):
            grad=dict(m.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all(),name)
            self.assertGreater(float(grad.norm()),0.,name)

    def test_empty_and_cross_origin_arc(self):
        m=self.model().eval(); a,p,v=fixture();v.zero_(); p.fill_(float('nan'))
        with torch.no_grad():
            o=m(a,a,p,p,v,v,decode=True,verify=True)
        self.assertFalse(o.verified[0].has_candidate)
        self.assertTrue(torch.isfinite(o.s1).all())
        indices,u=open_arc_indices(np.arange(100.),100.,[97,98,0,1],extension=2.)
        self.assertEqual(len(indices),9)
        self.assertEqual(len(set(indices)),len(indices))
        self.assertNotIn(50,indices)

    def test_no_invented_gap_labels(self):
        target=np.array([-1,2,3,-2,-2,6,7,-1]);valid=np.ones(8,bool)
        component,stop=visible_components(target,valid)
        self.assertNotEqual(component[2],component[5])
        self.assertEqual(component[3],-1)

    def test_beam_order_and_one_loop_limit(self):
        n=4;neighbors=np.tile(np.arange(n),(n,1));order=np.arange(n)
        da=np.arange(n)[:,None]-neighbors;allowed=da>0
        probability=np.tile(np.array([.8,.1,.1]),(n,n,1))
        node=np.ones(n)*2;stop=np.ones(n)*.5
        paths=beam_paths(order,neighbors,allowed,probability,da,da,node,stop,100.,4)
        self.assertEqual(paths[0][1],(0,1,2,3))
        self.assertAlmostEqual(paths[0][0],8+3*np.log(.8+1e-8)+np.log(.5+1e-8))
        short=beam_paths(order,neighbors,allowed,probability,da,da,node,stop,2.5,4)
        self.assertLessEqual(max(len(chain) for _,chain in short),3)


if __name__=='__main__': unittest.main()
