from dataclasses import fields, replace
import unittest
import torch
from .simple_builder import SimplePoseBuilder,SimplePolicy
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_consensus import make_cloud


class SimpleTests(unittest.TestCase):
    def make(self,radius=10,budget=8):
        return SimplePoseBuilder(CompatibilityConfig(.2308525608,.2842015547,.5878077664,.5,1.,9.),
            policy=SimplePolicy(radius,budget))

    def test_60px_not_merged(self):
        b=self.make(16);p=b.build_from_cloud(make_cloud([0,60]))
        self.assertEqual(len(p.clusters),2)
        self.assertTrue(all(len(c.edge_ids)==4 for c in p.clusters))

    def test_drift_chain_not_joined(self):
        for r in [8,10,12,16]:
            p=self.make(r).build_from_cloud(make_cloud([0,20,40,60,80]))
            self.assertTrue(all(len(c.edge_ids)<20 for c in p.clusters))

    def test_complementary_four_and_distractor(self):
        p=self.make(16).build_from_cloud(make_cloud([0,4,10,15,200]))
        self.assertEqual(len(p.clusters),2)
        self.assertEqual(sorted(len(c.edge_ids) for c in p.clusters),[4,16])

    def test_directional_kernel_cannot_remove_member(self):
        b=self.make(16);c=make_cloud([0,4,10,15])
        c=replace(c,spacing_a=torch.ones(16)*3,spacing_b=torch.ones(16)*3)
        p=b.build_from_cloud(c)
        self.assertEqual(len(p.clusters[0].edge_ids),16)
        kernel=c.compatibility(p.clusters[0].translation,b.geometry).kernel
        self.assertLess(int((kernel>=.0111).sum()),16)

    def test_long_lowq_ranks_over_short_highq(self):
        c=make_cloud([0,0,60])
        c=replace(c,q=torch.tensor([.02]*8+[.8]*4))
        p=self.make().build_from_cloud(c)
        self.assertEqual(len(p.clusters[0].edge_ids),8)
        self.assertLess(p.clusters[0].raw_absolute_mass_px,p.clusters[1].raw_absolute_mass_px)

    def test_arc_distance_not_a_gate(self):
        c=make_cloud([0,4]);c=replace(c,arc_a=c.arc_a*100,arc_b=c.arc_b*100)
        p=self.make().build_from_cloud(c)
        self.assertEqual(len(p.clusters),1)

    def test_no_heuristic_seed_input(self):
        with self.assertRaises(ValueError): self.make().build_from_cloud(make_cloud([0]),torch.zeros(1,2))

    def test_no_extra_q_filter(self):
        c=make_cloud([0]);c=replace(c,q=c.q*1e-9)
        self.assertEqual(len(self.make().build_from_cloud(c).clusters),1)

    def test_interpenetration_is_only_veto(self):
        p=self.make().build_from_cloud(make_cloud([0]),overlap_fn=lambda t:dict(available=True,fraction_sum_area=.2))
        self.assertEqual(len(p.clusters),0)

    def test_budget(self):
        b=self.make(budget=2);p=b.build_from_cloud(make_cloud([0,40,80]))
        self.assertEqual(len(p.clusters),2);self.assertEqual(len(b.all_clusters),3)

    def test_duplicate_identity_invariance(self):
        c=make_cloud([0,4])
        double=replace(c,**{f.name:getattr(c,f.name).repeat((2,)+(1,)*(getattr(c,f.name).ndim-1))
            for f in fields(c) if isinstance(getattr(c,f.name),torch.Tensor)})
        a=self.make().build_from_cloud(c).clusters[0]; b=self.make().build_from_cloud(double).clusters[0]
        self.assertEqual(len(a.edge_ids),len(b.edge_ids))
        self.assertAlmostEqual(a.independent_arc_px,b.independent_arc_px)

if __name__=='__main__':unittest.main()
