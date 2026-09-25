from dataclasses import fields, replace
import unittest
import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_consensus import make_cloud
from .pose_consensus_repair import MergeRepairPolicy, RepairedPoseConsensusBuilder


class RepairedConsensusTests(unittest.TestCase):
    def setUp(self):
        self.builder = RepairedPoseConsensusBuilder(CompatibilityConfig(.5, .5, .5, .5, 1., 30.))

    def test_four_complementary_regions_and_separate_far_mode(self):
        cloud = make_cloud([1, 3, 5, 10, 210])
        out = self.builder.build_from_cloud(cloud, torch.tensor([[1.,0.],[3.,0.],[5.,0.],[10.,0.],[210.,0.]]))
        self.assertEqual(len(out.clusters), 2)
        near = min(out.clusters, key=lambda c:float(c.translation.norm()))
        self.assertEqual(set(near.original_union_edge_ids[:,0].tolist()), set(range(16)))
        self.assertEqual(set(near.merged_hypothesis_ids), {0, 1, 2, 3})
        self.assertTrue(out.merge_trace)

    def test_identical_ambiguous_correspondence_distribution_is_one_mode(self):
        b = RepairedPoseConsensusBuilder(CompatibilityConfig(.5,1.,1.,1.,1.,9.,1.))
        cloud = make_cloud([0], per_group=4)
        cloud = replace(cloud, ids=torch.tensor([[0,0],[0,1],[1,2],[2,3]]),
            displacement=torch.tensor([[-2.,0.],[2.,0.],[0.,0.],[0.,0.]]),
            spacing_a=torch.ones(4), spacing_b=torch.ones(4))
        one = b.build_from_cloud(cloud, torch.zeros((1,2)))
        repeated = b.build_from_cloud(cloud, torch.zeros((4,2)))
        self.assertEqual(len(repeated.clusters), 1)
        self.assertEqual(len(repeated.clusters[0].original_union_edge_ids), 4)
        self.assertEqual(one.clusters[0].absolute_support_mass_px, repeated.clusters[0].absolute_support_mass_px)
        torch.testing.assert_close(one.clusters[0].translation, repeated.clusters[0].translation, rtol=0, atol=0)

    def test_disconnected_arcs_need_no_bridge(self):
        cloud = make_cloud([5,5])
        cloud = replace(cloud, arc_a=cloud.arc_a*100, arc_b=cloud.arc_b*100,
                        perimeter_a=40000., perimeter_b=40000.)
        out = self.builder.build_from_cloud(cloud, torch.tensor([[5.,0.],[5.,0.]]))
        self.assertEqual(len(out.clusters), 1)
        self.assertEqual(len(out.clusters[0].original_union_edge_ids), 8)

    def test_drift_chain_bounded_by_all_original_fitted_centers(self):
        cloud = make_cloud([0,20,40,60,80])
        out = self.builder.build_from_cloud(cloud, torch.tensor([[float(x),0.] for x in (0,20,40,60,80)]))
        self.assertGreater(len(out.clusters), 1)
        for c in out.clusters:
            centers = torch.stack([out.hypotheses[i].translation for i in c.merged_hypothesis_ids])
            self.assertLessEqual(float((centers-c.translation).norm(dim=1).max()), 17.)

    def test_duplicate_edge_records_do_not_multiply_mass(self):
        cloud = make_cloud([5,5])
        doubled = replace(cloud, **{f.name:getattr(cloud,f.name).repeat((2,)+(1,)*(getattr(cloud,f.name).ndim-1))
            for f in fields(cloud) if isinstance(getattr(cloud,f.name),torch.Tensor)})
        a = self.builder.build_from_cloud(cloud, torch.tensor([[5.,0.]]))
        b = self.builder.build_from_cloud(doubled, torch.tensor([[5.,0.],[5.,0.]]))
        self.assertEqual(len(b.clusters), 1)
        self.assertEqual(a.clusters[0].absolute_support_mass_px, b.clusters[0].absolute_support_mass_px)

    def test_conflicting_copies_still_rejected(self):
        cloud = make_cloud([5])
        doubled = replace(cloud, **{f.name:torch.cat((getattr(cloud,f.name),getattr(cloud,f.name)[:1]))
            for f in fields(cloud) if isinstance(getattr(cloud,f.name),torch.Tensor)})
        doubled.q[-1] = .9
        with self.assertRaises(ValueError):
            self.builder.build_from_cloud(doubled)

    def test_mutually_exclusive_partner_modes_remain_separate(self):
        cloud = make_cloud([-10,10]); ids = cloud.ids.clone(); ids[4:,0] -= 4
        cloud = replace(cloud, ids=ids)
        out = self.builder.build_from_cloud(cloud, torch.tensor([[-10.,0.],[10.,0.]]))
        self.assertEqual(len(out.clusters), 2)
        self.assertFalse(out.merge_trace)
        self.assertTrue(all(abs(float(c.translation[0])) > 9 for c in out.clusters))

    def test_material_overlap_blocks_new_union(self):
        out = self.builder.build_from_cloud(make_cloud([1,3]), torch.tensor([[1.,0.],[3.,0.]]),
            overlap_fn=lambda t:dict(available=True,fraction_sum_area=.3))
        self.assertEqual(len(out.clusters), 2)

    def test_duplicate_bad_layout_is_deduplicated_not_made_valid(self):
        out = self.builder.build_from_cloud(make_cloud([5]), torch.tensor([[5.,0.],[5.,0.]]),
            overlap_fn=lambda t:dict(available=True,fraction_sum_area=.3))
        self.assertEqual(len(out.clusters), 1)
        self.assertEqual(out.clusters[0].overlap['fraction_sum_area'], .3)

    def test_tiny_absolute_q_still_has_no_proposals(self):
        cloud = make_cloud([5]); cloud = replace(cloud,q=cloud.q*1e-9)
        out = self.builder.build_from_cloud(cloud)
        self.assertEqual(len(out.clusters), 0)

    def test_parallel_evidence_remains_underconstrained(self):
        out = self.builder.build_from_cloud(make_cloud([5]), torch.tensor([[5.,-15.]]))
        self.assertTrue(out.clusters[0].underconstrained)

    def test_policy_cannot_discard_majority_of_evidence(self):
        with self.assertRaises(ValueError):
            MergeRepairPolicy(maximum_lost_explained_mass_fraction=.9)

    def test_exchanging_fragments_negates_same_common_poses(self):
        cloud = make_cloud([1,3,5,10,210])
        seeds = torch.tensor([[1.,0.],[3.,0.],[5.,0.],[10.,0.],[210.,0.]])
        swapped = replace(cloud, ids=cloud.ids.flip(1), displacement=-cloud.displacement,
            normal_a=cloud.normal_b, normal_b=cloud.normal_a,
            reliability_a=cloud.reliability_b, reliability_b=cloud.reliability_a,
            spacing_a=cloud.spacing_b, spacing_b=cloud.spacing_a,
            arc_a=cloud.arc_b, arc_b=cloud.arc_a,
            perimeter_a=cloud.perimeter_b, perimeter_b=cloud.perimeter_a)
        a = self.builder.build_from_cloud(cloud,seeds)
        b = self.builder.build_from_cloud(swapped,-seeds)
        self.assertEqual(len(a.clusters),len(b.clusters))
        for x,y in zip(a.clusters,b.clusters):
            torch.testing.assert_close(x.translation,-y.translation,atol=1e-5,rtol=1e-5)
            self.assertAlmostEqual(x.absolute_support_mass_px,y.absolute_support_mass_px,places=5)

    def test_full_q_head_refinement_export_and_gradients_still_agree(self):
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.consensus_head import ConsensusEvidenceHead
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.model import S7Consensus
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_evidence import fixture
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.diagnostics import snapshot_prediction
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import recall_full_q
        torch.manual_seed(51)
        model=S7Consensus(None,self.builder.geometry,head=ConsensusEvidenceHead(feature_dim=4))
        model.builder=self.builder
        pair=fixture()
        out=model.score_pair(pair,capture_diagnostics=True)
        self.assertTrue(out.has_candidate)
        self.assertTrue(out.proposals.merge_trace)
        for c in out.clusters:
            self.assertIs(c.translation,c.encoded.evidence.pose)
            expected=model.head.readout(model.head(recall_full_q(pair,c.translation,self.builder.geometry)),0.)
            torch.testing.assert_close(c.readout.logit,expected.logit)
        meta,arrays=snapshot_prediction('structural-test-only',pair,out,threshold=.5,
            provenance={'purpose':'repaired merger CPU integration, not formal trained model'})
        self.assertEqual(len(meta['clusters']),len(out.clusters))
        for c in meta['clusters']:
            self.assertIn('original_union_edge_ids',c['proposal'])
            self.assertIn(c['proposal']['original_union_edge_ids'],arrays)
        sum(c.readout.logit for c in out.clusters).backward()
        self.assertTrue(model.head.input[0].weight.grad.abs().sum()>0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.head.parameters()))


if __name__ == '__main__':
    unittest.main()
