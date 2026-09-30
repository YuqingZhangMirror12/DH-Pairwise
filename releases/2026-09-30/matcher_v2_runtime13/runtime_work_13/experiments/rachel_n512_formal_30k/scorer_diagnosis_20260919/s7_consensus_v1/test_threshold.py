from dataclasses import fields, replace
import inspect
import unittest
import torch

from .compatibility import CompatibilityConfig
from .legacy_pose_consensus import PoseCluster, PoseConsensusBuilder as LegacyBuilder
from .test_consensus import make_cloud
from .threshold_builder import ThresholdPolicy, ThresholdPoseBuilder, diameter, bound_common_pose


def native_groups(cloud, poses):
    """Synthetic Matcher proposals, no GT-based selection or construction."""
    poses = torch.as_tensor(poses, dtype=torch.float32).reshape(-1, 2)
    size = len(cloud.ids) // len(poses)
    return tuple(PoseCluster(p, cloud.ids[i*size:(i+1)*size], (i,), (i,), p[None],
        float((cloud.q[i*size:(i+1)*size]*cloud.arc_weight[i*size:(i+1)*size]).sum()),
        torch.eye(2), False, {}) for i, p in enumerate(poses))


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.geometry = CompatibilityConfig(.5, .5, .5, .5, 1., 9.)
        self.builder = ThresholdPoseBuilder(self.geometry)

    def build(self, centers, **kwargs):
        cloud = make_cloud(centers)
        hyps = native_groups(cloud, [[c, 0.] for c in centers])
        return self.builder.build_from_hypotheses(cloud, hyps, **kwargs)

    def test_four_local_groups_form_one_union_not_best_group(self):
        out = self.build([0, 4, 10, 15, 80])
        near = min(out.clusters, key=lambda c: float(c.translation.norm()))
        self.assertEqual(set(near.merged_hypothesis_ids), {0, 1, 2, 3})
        self.assertEqual(set(map(tuple, near.edge_ids.tolist())), {(i, i) for i in range(16)})
        self.assertEqual(near.actual_diameter_px, 15.)
        self.assertEqual(len(out.clusters), 2)

    def test_abc_neighbor_chain_cannot_become_two_thresholds(self):
        out = self.build([0, 12, 24])
        self.assertEqual(len(out.clusters), 2)
        self.assertTrue(all(len(c.merged_hypothesis_ids) < 3 for c in out.clusters))
        self.assertTrue(all(c.actual_diameter_px <= 16 for c in out.clusters))

    def test_long_chain_checks_original_poses_not_moving_centers(self):
        out = self.build([0, 14, 28, 42, 56, 70, 84])
        for c in self.builder.all_clusters:
            self.assertLessEqual(diameter(c.member_translations_rc), 16.)
            self.assertLessEqual(float((c.member_translations_rc-c.translation).norm(dim=1).max()), 16.0001)
        self.assertGreaterEqual(len(out.clusters), 4)

    def test_threshold_is_full_2d_vector_not_magnitude(self):
        cloud = make_cloud([20, -20])
        out = self.builder.build_from_hypotheses(cloud, native_groups(cloud, [[20, 0], [-20, 0]]))
        self.assertEqual(len(out.clusters), 2)

    def test_diagonal_distance_uses_euclidean_norm(self):
        cloud = make_cloud([0, 0])
        d = cloud.displacement.clone(); d[4:] = torch.tensor([12., 12.])
        cloud = replace(cloud, displacement=d)
        out = self.builder.build_from_hypotheses(cloud, native_groups(cloud, [[0, 0], [12, 12]]))
        self.assertEqual(len(out.clusters), 2)

    def test_exact_threshold_and_just_outside(self):
        self.assertEqual(len(self.build([0, 16]).clusters), 1)
        self.assertEqual(len(self.build([0, 16.01]).clusters), 2)

    def test_contour_distance_cannot_block_same_pose(self):
        cloud = make_cloud([1, 3])
        cloud = replace(cloud, arc_a=cloud.arc_a*1000, arc_b=cloud.arc_b*1000,
                        perimeter_a=400000., perimeter_b=400000.)
        out = self.builder.build_from_hypotheses(cloud, native_groups(cloud, [[1, 0], [3, 0]]))
        self.assertEqual(len(out.clusters), 1)
        self.assertEqual(len(out.clusters[0].edge_ids), 8)

    def test_duplicate_hypotheses_and_edges_do_not_boost_mass(self):
        cloud = make_cloud([0, 4])
        hyps = native_groups(cloud, [[0, 0], [4, 0]])
        a = self.builder.build_from_hypotheses(cloud, hyps).clusters[0]
        dup = replace(cloud, **{f.name:getattr(cloud,f.name).repeat((3,)+(1,)*(getattr(cloud,f.name).ndim-1))
            for f in fields(cloud) if isinstance(getattr(cloud,f.name),torch.Tensor)})
        b = self.builder.build_from_hypotheses(dup, hyps + hyps + hyps).clusters[0]
        self.assertTrue(torch.equal(a.edge_ids, b.edge_ids))
        self.assertAlmostEqual(a.absolute_support_mass_px, b.absolute_support_mass_px, places=6)
        torch.testing.assert_close(a.translation, b.translation)

    def test_conflicting_duplicate_edge_is_not_silently_averaged(self):
        cloud = make_cloud([0])
        dup = replace(cloud, **{f.name:torch.cat([getattr(cloud,f.name),getattr(cloud,f.name)[:1]])
            for f in fields(cloud) if isinstance(getattr(cloud,f.name),torch.Tensor)})
        dup.q[-1] = .9
        with self.assertRaises(ValueError): self.builder.build_from_cloud(dup)

    def test_physical_conflict_blocks_union(self):
        out = self.build([0, 10], overlap_fn=lambda p: dict(available=True,
            fraction_sum_area=.3 if 3 < float(p[0]) < 7 else 0.))
        self.assertEqual(len(out.clusters), 2)
        self.assertFalse(out.merge_trace)

    def test_60px_competitor_is_not_absorbed(self):
        out = self.build([0, 4, 10, 15, 60])
        self.assertEqual(len(out.clusters), 2)
        self.assertTrue(all(not ({0, 4} <= set(c.merged_hypothesis_ids)) for c in out.clusters))

    def test_budget_applies_after_merge(self):
        b = ThresholdPoseBuilder(self.geometry, policy=ThresholdPolicy(candidate_budget=1))
        cloud = make_cloud([0, 4, 10, 15, 80])
        out = b.build_from_hypotheses(cloud, native_groups(cloud, [[x, 0] for x in [0,4,10,15,80]]))
        self.assertEqual(len(out.clusters), 1)
        self.assertEqual(len(b.all_clusters), 2)
        self.assertEqual(len(out.clusters[0].merged_hypothesis_ids), 4)

    def test_native_hypothesis_generation_is_unchanged(self):
        cloud = make_cloud([0, 4, 10, 15, 80])
        old = LegacyBuilder(self.geometry)
        seeds = old._seeds(cloud)
        expected = [old._hypothesis(cloud, t, (i,), t[None], (i,)) for i,t in enumerate(seeds)]
        actual = self.builder.build_from_cloud(cloud).hypotheses
        self.assertEqual(len(expected), len(actual))
        for a,b in zip(expected,actual):
            torch.testing.assert_close(a.translation,b.translation)
            self.assertTrue(torch.equal(a.edge_ids,b.edge_ids))

    def test_gt_is_not_an_inference_argument(self):
        for method in [self.builder.__call__, self.builder.build_from_cloud, self.builder.build_from_hypotheses]:
            self.assertFalse({'labels','gt','ground_truth'} & set(inspect.signature(method).parameters))

    def test_low_mass_is_not_normalized_into_a_confident_candidate(self):
        cloud = make_cloud([0])
        out = self.builder.build_from_cloud(replace(cloud,q=cloud.q*1e-9))
        self.assertEqual(len(out.clusters),0)

    def test_empty_cloud_is_valid_no_candidate(self):
        cloud = make_cloud([0])
        cloud = replace(cloud, **{f.name:getattr(cloud,f.name)[:0] for f in fields(cloud)
            if isinstance(getattr(cloud,f.name),torch.Tensor)})
        self.assertEqual(len(self.builder.build_from_cloud(cloud).clusters),0)

    def test_each_group_changes_joint_layout(self):
        cloud = make_cloud([0,4,10,15])
        hyps = native_groups(cloud,[[x,0] for x in [0,4,10,15]])
        full = self.builder.build_from_hypotheses(cloud,hyps).clusters[0].translation
        for i in range(4):
            reduced = self.builder.build_from_hypotheses(cloud,tuple(h for j,h in enumerate(hyps) if i!=j))
            self.assertGreater(float((reduced.clusters[0].translation-full).norm()),1e-4)

    def test_pose_constraint_has_finite_boundary_gradient(self):
        start=torch.tensor([16.,0.]);proposed=torch.tensor([16.,5.],requires_grad=True)
        pose=bound_common_pose(start,proposed,torch.zeros(1,2),16.)
        self.assertLessEqual(float(pose.norm()),16.0001)
        self.assertTrue(torch.isfinite(torch.autograd.grad(pose.sum(),proposed)[0]).all())

    def test_pose_refinement_cannot_escape_any_original_member(self):
        centers=torch.tensor([[0.,0.],[16.,0.]])
        pose=bound_common_pose(torch.tensor([8.,0.]),torch.tensor([100.,100.]),centers,16.)
        self.assertTrue(((pose-centers).norm(dim=1)<=16.0001).all())

    def test_invalid_policy_rejected(self):
        for x in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):ThresholdPolicy(pose_diameter_px=x)


if __name__=='__main__':unittest.main()
