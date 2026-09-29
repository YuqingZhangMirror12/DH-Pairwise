from dataclasses import fields, replace
import unittest
import torch

from .compatibility import CompatibilityConfig
from .pose_consensus import EdgeCloud, PoseConsensusBuilder, ProposalConfig


def make_cloud(centers, per_group=4):
    n=len(centers)*per_group
    ids=torch.arange(n)
    displacement=torch.tensor([[float(c),0.] for c in centers for _ in range(per_group)])
    normals=torch.tensor([[0.,1.]]).repeat(n,1)
    one=torch.ones(n)
    arc=torch.tensor([float(200*g+8*k) for g in range(len(centers)) for k in range(per_group)])
    return EdgeCloud(torch.stack((ids,ids),1),displacement,one*.3,one*7,normals,-normals,
                     one,one,one*8,one*8,arc,arc,200.*len(centers),200.*len(centers))


class ConsensusTests(unittest.TestCase):
    def setUp(self):
        self.builder=PoseConsensusBuilder(CompatibilityConfig(.5,.5,.5,.5,1.,30.))

    def test_four_complementary_groups_merge_far_fifth_does_not(self):
        cloud=make_cloud([1,3,5,10,210])
        seeds=torch.tensor([[1.,0.],[3.,0.],[5.,0.],[10.,0.],[210.,0.]])
        out=self.builder.build_from_cloud(cloud,seeds)
        self.assertEqual(len(out.clusters),2)
        near=min(out.clusters,key=lambda c:float(c.translation.norm()))
        self.assertEqual(set(near.edge_ids[:,0].tolist()),set(range(16)))
        self.assertEqual(set(near.merged_hypothesis_ids),{0,1,2,3})
        self.assertTrue(out.merge_trace)

    def test_disconnected_arcs_need_no_bridge(self):
        cloud=make_cloud([5,5])
        cloud=replace(cloud,arc_a=cloud.arc_a*100,arc_b=cloud.arc_b*100,
                      perimeter_a=40000.,perimeter_b=40000.)
        out=self.builder.build_from_cloud(cloud,torch.tensor([[5.,0.],[5.,0.]]))
        self.assertEqual(len(out.clusters),1)
        self.assertEqual(len(out.clusters[0].edge_ids),8)

    def test_drift_chain_cannot_join_unbounded_span(self):
        centers=[0,20,40,60,80]
        out=self.builder.build_from_cloud(make_cloud(centers),torch.tensor([[float(x),0.] for x in centers]))
        self.assertGreater(len(out.clusters),1)
        for cluster in out.clusters:
            self.assertLessEqual(float((cluster.seed_translations-cluster.translation).norm(dim=1).max()),17.)

    def test_duplicate_edges_and_seeds_do_not_multiply_evidence(self):
        cloud=make_cloud([5,5])
        duplicate=replace(cloud,**{f.name:getattr(cloud,f.name).repeat((3,)+(1,)*(getattr(cloud,f.name).ndim-1))
            for f in fields(cloud) if isinstance(getattr(cloud,f.name),torch.Tensor)})
        a=self.builder.build_from_cloud(cloud,torch.tensor([[5.,0.]]))
        b=self.builder.build_from_cloud(duplicate,torch.tensor([[5.,0.]]).repeat(4,1))
        self.assertEqual(len(b.clusters),1)
        self.assertEqual(len(a.clusters[0].edge_ids),len(b.clusters[0].edge_ids))
        self.assertAlmostEqual(a.clusters[0].absolute_support_mass_px,b.clusters[0].absolute_support_mass_px)

    def test_conflicting_duplicate_rejected(self):
        cloud=make_cloud([5])
        duplicate=replace(cloud,**{f.name:torch.cat((getattr(cloud,f.name),getattr(cloud,f.name)[:1]))
            for f in fields(cloud) if isinstance(getattr(cloud,f.name),torch.Tensor)})
        duplicate.q[-1]=.9
        with self.assertRaises(ValueError):
            self.builder.build_from_cloud(duplicate)

    def test_low_absolute_mass_not_rescaled_to_valid_candidate(self):
        cloud=make_cloud([5])
        cloud=replace(cloud,q=cloud.q*1e-9)
        out=self.builder.build_from_cloud(cloud)
        self.assertEqual(len(out.clusters),0)

    def test_all_parallel_support_reports_underconstrained(self):
        out=self.builder.build_from_cloud(make_cloud([5]),torch.tensor([[5.,-15.]]))
        self.assertTrue(out.clusters[0].underconstrained)

    def test_overlap_prevents_union(self):
        out=self.builder.build_from_cloud(make_cloud([1,3]),torch.tensor([[1.,0.],[3.,0.]]),
            overlap_fn=lambda t:dict(available=True,fraction_sum_area=.3))
        self.assertEqual(len(out.clusters),2)

    def test_exclusive_partner_modes_not_averaged(self):
        cloud=make_cloud([-10,10])
        ids=cloud.ids.clone();ids[4:,0]-=4
        cloud=replace(cloud,ids=ids)
        out=self.builder.build_from_cloud(cloud,torch.tensor([[-10.,0.],[10.,0.]]))
        self.assertEqual(len(out.clusters),2)
        self.assertTrue(all(abs(float(c.translation[0]))>9. for c in out.clusters))
        self.assertFalse(out.merge_trace)


if __name__=='__main__':
    unittest.main()
