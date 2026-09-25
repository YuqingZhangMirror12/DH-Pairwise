from dataclasses import replace
import unittest
import torch

from .compatibility import CompatibilityConfig
from .evidence import PairEvidence, recall_full_q, observed_arc_cells, material_overlap
from .geometry import compact_contour


def fixture(q=None):
    a=torch.tensor([[0.,0.],[0.,8.],[5.,8.],[10.,8.],[10.,0.]])
    b=torch.tensor([[0.,13.],[0.,20.],[10.,20.],[10.,13.],[5.,13.]])
    if q is None:
        q=torch.zeros(5,5);q[2,4]=.12;q[2,3]=.4
    ga=compact_contour(a[None],torch.ones(1,5,dtype=torch.bool))
    gb=compact_contour(b[None],torch.ones(1,5,dtype=torch.bool))
    f=torch.arange(20).float().reshape(5,4)/20
    return PairEvidence(f,f+1,f+2,f+3,q,1-q.sum(1),1-q.sum(0),ga,gb,
                        torch.arange(5),torch.arange(5))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg=CompatibilityConfig(.5,.5,.5,.5,1.,30.)

    def test_secondary_correspondence_recalled_with_absolute_mass(self):
        pair=fixture()
        out=recall_full_q(pair,torch.zeros(2),self.cfg,block_rows=2)
        self.assertAlmostEqual(float(out.weights[2,4]),.12,places=6)
        self.assertLess(float(out.a.mass[2]),.52)
        torch.testing.assert_close(out.a.mass+out.a.other_mass+out.a.unmatched,torch.ones(5))
        torch.testing.assert_close(out.b.mass+out.b.other_mass+out.b.unmatched,torch.ones(5))
        self.assertEqual(out.a.scalar_features.shape,(5,17))

    def test_more_than512_edges_not_truncated(self):
        n=40
        angle=torch.arange(n)*2*torch.pi/n
        points=torch.stack((angle.sin(),angle.cos()),1)
        g=compact_contour(points[None],torch.ones(1,n,dtype=torch.bool))
        q=torch.ones(n,n)*.01;f=torch.ones(n,4)
        p=PairEvidence(f,f,f,f,q,1-q.sum(1),1-q.sum(0),g,g,torch.arange(n),torch.arange(n))
        out=recall_full_q(p,torch.zeros(2),self.cfg,block_rows=7)
        self.assertEqual(len(out.correspondence_ids()),1600)

    def test_multimodal_partner_distribution_retained(self):
        q=torch.zeros(5,5);q[2,0]=.2;q[2,3]=.2
        out=recall_full_q(fixture(q),torch.zeros(2),self.cfg)
        self.assertEqual(int((out.weights[2]>0).sum()),2)
        self.assertGreater(float(out.a.residual_variance[2,0]),0.)
        # The two original positions remain available; no mean point replaces Q.
        torch.testing.assert_close(out.pair.q,q)

    def test_row_blocking_has_no_numerical_or_gradient_effect(self):
        p=fixture();t=torch.tensor([.3,.5],requires_grad=True)
        a=recall_full_q(p,t,self.cfg,block_rows=1)
        b=recall_full_q(p,t,self.cfg,block_rows=5)
        torch.testing.assert_close(a.weights,b.weights)
        torch.testing.assert_close(a.a.scalar_features,b.a.scalar_features)
        a.weights.sum().backward()
        self.assertTrue(torch.isfinite(t.grad).all())

    def test_arc_measure_does_not_fill_unseen_stretch(self):
        p=fixture()
        lengths,intervals=observed_arc_cells(p.ga)
        self.assertTrue((lengths<=7).all())
        self.assertLess(float(lengths.sum()),float(p.ga.perimeter_px[0]))
        self.assertLessEqual(float(lengths[2]),7.)
        self.assertEqual(intervals.shape,(5,2))

    def test_material_overlap_uses_negative_layout_shift(self):
        p=fixture();a=torch.zeros(10,12);b=torch.zeros_like(a)
        a[:,:4]=1;b[:,6:10]=1
        p=replace(p,mask_a=a,mask_b=b)
        self.assertEqual(material_overlap(p,torch.tensor([0.,0.]))['intersection_px'],0.)
        self.assertEqual(material_overlap(p,torch.tensor([0.,6.]))['fraction_min_area'],1.)

    def test_exchange_preserves_masses_and_conditional_features(self):
        p=fixture()
        other=PairEvidence(p.local_b,p.local_a,p.context_b,p.context_a,p.q.T,
            p.unmatched_b,p.unmatched_a,p.gb,p.ga,p.original_b,p.original_a)
        t=torch.tensor([1.,2.])
        a=recall_full_q(p,t,self.cfg)
        b=recall_full_q(other,-t,self.cfg)
        torch.testing.assert_close(a.weights,b.weights.T)
        torch.testing.assert_close(a.a.opposite_local,b.b.opposite_local)
        torch.testing.assert_close(a.a.scalar_features[:,:10],b.b.scalar_features[:,:10])


if __name__=='__main__':
    unittest.main()
