from dataclasses import replace, fields
import unittest

import torch

from measure import CompatibilityConfig, PoseConsensusBuilder, make_cloud
from pose_scale import IsotropicPoseBuilder, PoseScalePolicy


CFG = CompatibilityConfig(.5, .23085256082452021, .2842015546641426,
    .27137619747785585, 1., 9., .5878077664079844)


def cloud_at(centers, spacing=4.):
    c = make_cloud(centers)
    return replace(c, spacing_a=torch.full_like(c.spacing_a, spacing),
        spacing_b=torch.full_like(c.spacing_b, spacing), arc_weight=torch.full_like(c.arc_weight, spacing))


class PoseScaleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.builder = IsotropicPoseBuilder(CFG, scale_policy=PoseScalePolicy(16.))

    def test_far_modes_do_not_merge(self):
        c = cloud_at([0, 60])
        p = self.builder.build_from_cloud(c, torch.tensor([[0., 0.], [60., 0.]]))
        self.assertEqual(len(p.clusters), 2)

    def test_drift_checks_every_original_center(self):
        c = cloud_at([0, 10, 20, 30, 40, 50, 60, 70, 80])
        p = self.builder.build_from_cloud(c, torch.tensor([[float(x), 0.] for x in range(0, 81, 10)]))
        self.assertGreater(len(p.clusters), 1)
        for cluster in self.builder.all_clusters:
            self.assertLessEqual(float((cluster.original_fitted_centers_rc-cluster.translation).norm(dim=1).max()), 16.00001)

    def test_four_groups_true_union_and_far_distractor(self):
        c = cloud_at([0, 4, 10, 15, 200])
        p = self.builder.build_from_cloud(c, torch.tensor([[float(x), 0.] for x in [0, 4, 10, 15, 200]]))
        self.assertEqual(len(p.clusters), 2)
        near = min(p.clusters, key=lambda x: float(x.translation.norm()))
        self.assertEqual(set(near.merged_hypothesis_ids), {0, 1, 2, 3})
        self.assertEqual(len(near.original_union_edge_ids), 16)

    def test_arc_separation_is_not_a_gate(self):
        c = cloud_at([0, 4, 10, 15, 200])
        seeds = torch.tensor([[float(x), 0.] for x in [0, 4, 10, 15, 200]])
        a = self.builder.build_from_cloud(c, seeds)
        d = replace(c, arc_a=c.arc_a*100, arc_b=c.arc_b*100, perimeter_a=c.perimeter_a*100, perimeter_b=c.perimeter_b*100)
        b = self.builder.build_from_cloud(d, seeds)
        self.assertEqual(len(a.clusters), len(b.clusters))
        for x, y in zip(a.clusters, b.clusters):
            torch.testing.assert_close(x.translation, y.translation, rtol=0, atol=0)
            self.assertTrue(torch.equal(x.original_union_edge_ids, y.original_union_edge_ids))

    def test_overlap_blocks_absorption_and_dedup(self):
        c = cloud_at([0, 0])
        p = self.builder.build_from_cloud(c, torch.zeros((2, 2)),
            overlap_fn=lambda t: dict(available=True, fraction_sum_area=.3))
        self.assertEqual(len(p.clusters), 2)

    def test_duplicate_evidence_not_multiple_votes(self):
        c = cloud_at([0, 0])
        repeated = replace(c, **{f.name: torch.cat([getattr(c, f.name)]*3) for f in fields(c)
            if isinstance(getattr(c, f.name), torch.Tensor)})
        a = self.builder.build_from_cloud(c, torch.zeros((1, 2)))
        b = self.builder.build_from_cloud(repeated, torch.zeros((4, 2)))
        self.assertEqual(len(b.clusters), 1)
        self.assertAlmostEqual(a.clusters[0].absolute_support_mass_px, b.clusters[0].absolute_support_mass_px)

    def test_penetration_not_relaxed_to_pose_radius(self):
        c = cloud_at([0])
        self.assertTrue(self.builder.pose_member(c, torch.tensor([0., -8.]), 16.).all())
        self.assertFalse(self.builder.pose_member(c, torch.tensor([0., 8.]), 16.).any())

    def test_directional_kernel_and_fit_unchanged(self):
        c = cloud_at([0, 4, 10])
        baseline = PoseConsensusBuilder(CFG)
        pose = torch.tensor([3., -5.])
        a = baseline._fit(c, pose)
        b = self.builder._fit(c, pose)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        torch.testing.assert_close(c.compatibility(pose, baseline.geometry).localization_kernel,
            c.compatibility(pose, self.builder.geometry).localization_kernel, rtol=0, atol=0)

    def test_contact_convention_collapses_normal_seed_ambiguity(self):
        c = cloud_at([0])
        c = replace(c, displacement=c.displacement+torch.tensor([0., 8.]))
        p = self.builder.build_from_cloud(c, torch.tensor([[0., 0.], [0., 4.], [0., 8.]]))
        self.assertEqual(len(p.clusters), 1)
        torch.testing.assert_close(p.clusters[0].translation, torch.tensor([0., 8.]), rtol=0, atol=1e-5)

    def test_95_percent_mass_check_is_explicit(self):
        c = cloud_at([0, 4, 10, 15])
        p = self.builder.build_from_cloud(c, torch.tensor([[float(x), 0.] for x in [0, 4, 10, 15]]))
        self.assertTrue(p.merge_trace)
        self.assertTrue(all(t['maximum_lost_explained_mass_fraction'] <= .0500001 for t in p.merge_trace))

    def test_frozen_hypotheses_preserved(self):
        c = cloud_at([0, 4, 10])
        original = PoseConsensusBuilder(CFG).build_from_cloud(c)
        new = self.builder.build_from_cloud(original.cloud, original.seeds, frozen_hypotheses=original.hypotheses)
        self.assertEqual(len(original.hypotheses), len(new.hypotheses))
        self.assertTrue(all(x is y for x, y in zip(original.hypotheses, new.hypotheses)))

    def test_adaptive_radius_uses_mean_edge_spacing_with_floor(self):
        p = PoseScalePolicy(adaptive=True)
        self.assertEqual(p.radius(cloud_at([0], 3.)), 10.)
        self.assertEqual(p.radius(cloud_at([0], 4.)), 12.)
        self.assertEqual(p.radius(cloud_at([0], 5.)), 15.)


if __name__ == '__main__':
    unittest.main()
