"""CPU tests of derived candidate stages; no training, cache or remote writes."""
import inspect
import unittest

import numpy as np

from staging.pairwise_v0_2.models.translation_layout import (
    TranslationLayoutConfig, estimate_translation_layout,
)
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_modes_v1.analyze import propose_modes
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.candidate_groups import build_candidate_groups


def production(a, b, q, min_inliers=3):
    return estimate_translation_layout(a, b, q, config=TranslationLayoutConfig(
        correspondence_mode="topk_union", top_k=2, max_candidates=512,
        inlier_radius_px=10., min_inliers=min_inliers))


def cache_inputs(a, b, q, result, padded=False):
    n = result.candidate_count
    size = n + 3 if padded else n
    slots = np.arange(n) + (1 if padded else 0)
    edges = np.full((size, 2), -1, np.int64)
    weights = np.full(size, np.nan)
    valid, final = np.zeros(size, bool), np.zeros(size, bool)
    edges[slots] = result.candidate_indices
    weights[slots] = q[result.candidate_indices[:, 0], result.candidate_indices[:, 1]]
    valid[slots], final[slots] = True, result.inlier_mask
    return dict(points_a_rc=np.asarray(a, np.float32), points_b_rc=np.asarray(b, np.float32),
                candidate_indices=edges, candidate_valid=valid, candidate_weights=weights,
                final_inliers=final, final_translation_rc=result.t_a_to_b_rc.astype(np.float32),
                layout_valid=result.valid)


def offset_fixture(offsets, weights=None):
    offsets = np.asarray(offsets, np.float32)
    if offsets.ndim == 1:
        offsets = np.column_stack([offsets, np.zeros(len(offsets), np.float32)])
    a = np.column_stack([np.arange(len(offsets)) * 50, np.arange(len(offsets)) * 70]).astype(np.float32)
    b = a + offsets
    q = np.diag(np.ones(len(a), np.float32) if weights is None else np.asarray(weights, np.float32))
    return a, b, q


class CandidateGroupsTest(unittest.TestCase):
    def test_identical_seed_and_final_is_reported(self):
        a, b, q = offset_fixture([30, 30, 30, 30])
        ref = production(a, b, q)
        result = build_candidate_groups(**cache_inputs(a, b, q, ref))
        self.assertEqual(result.production_status, "ok")
        self.assertTrue(result.diagnostics["seed_final_identical"])
        self.assertEqual(result.diagnostics["seed_final_edge_jaccard"], 1.)
        self.assertEqual(len(result.multi_modes), 1)
        np.testing.assert_array_equal(result.final_decoded.indices, ref.candidate_indices[ref.inlier_mask])

    def test_seed_is_before_refinement_and_edges_can_change(self):
        a, b, q = offset_fixture([-9, 0, 9, -11], [2, 1, 1, .2])
        ref = production(a, b, q)
        result = build_candidate_groups(**cache_inputs(a, b, q, ref))
        self.assertTrue(ref.valid)
        np.testing.assert_array_equal(result.single_seed.translation_rc, [0., 0.])
        self.assertFalse(result.diagnostics["seed_final_edges_equal"])
        self.assertFalse(result.diagnostics["seed_final_identical"])
        self.assertAlmostEqual(result.diagnostics["seed_final_edge_jaccard"], .5)
        self.assertGreater(result.diagnostics["seed_final_translation_l2_px"], 6.)
        self.assertTrue(result.diagnostics["first_mode_replay_checked"])

    def test_separated_modes_equal_original_propose_modes(self):
        a, b, q = offset_fixture([0, 1, -1, 60, 61, 59, 120, 119, 121],
                                 [1, .9, .8, .7, .6, .5, .4, .3, .2])
        ref = production(a, b, q)
        inputs = cache_inputs(a, b, q, ref)
        result = build_candidate_groups(**inputs)
        edges = ref.candidate_indices
        expected = propose_modes(b[edges[:, 1]].astype(np.float64) - a[edges[:, 0]],
                                 q[edges[:, 0], edges[:, 1]])
        self.assertEqual(len(result.multi_modes), 3)
        self.assertEqual(len(expected), 3)
        for actual, old in zip(result.multi_modes, expected):
            self.assertEqual(actual.rank, old["rank"])
            self.assertEqual(actual.seed_candidate_id, old["seed_index"])
            np.testing.assert_array_equal(actual.candidate_ids, old["candidate_inlier_ids"])
            np.testing.assert_allclose(actual.translation_rc, old["translation_rc"], rtol=0, atol=1e-12)
        self.assertEqual(len({m.group_id for m in result.multi_modes}), 3)
        self.assertFalse(result.diagnostics["mode_groups_merged"])

    def test_padding_retains_original_slot_ids_and_raw_weights(self):
        a, b, q = offset_fixture([2, 3, 4, 70, 71, 72], [.8, .7, .6, .5, .4, .3])
        ref = production(a, b, q)
        inputs = cache_inputs(a, b, q, ref, padded=True)
        result = build_candidate_groups(**inputs)
        for group in (result.single_seed, result.final_decoded) + result.multi_modes:
            self.assertTrue((group.candidate_ids >= 1).all())
            np.testing.assert_array_equal(group.indices, inputs["candidate_indices"][group.candidate_ids])
            np.testing.assert_array_equal(group.weights, inputs["candidate_weights"][group.candidate_ids])
        inputs["candidate_weights"][:] = 999
        self.assertLess(result.final_decoded.weights.max(), 1.)

    def test_random_production_replay(self):
        rng = np.random.RandomState(8)
        for _ in range(20):
            a = rng.uniform(-100, 100, (12, 2)).astype(np.float32)
            b = a + np.array([30, -45], np.float32) + rng.normal(0, 2, (12, 2)).astype(np.float32)
            q = rng.uniform(.0001, .01, (12, 12)).astype(np.float32) + np.eye(12, dtype=np.float32)
            ref = production(a, b, q)
            self.assertTrue(ref.valid)
            result = build_candidate_groups(**cache_inputs(a, b, q, ref))
            np.testing.assert_array_equal(result.final_decoded.candidate_ids, np.flatnonzero(ref.inlier_mask))
            np.testing.assert_array_equal(result.final_decoded.translation_rc, ref.t_a_to_b_rc.astype(np.float32))
            self.assertLess(result.diagnostics["first_mode_replay_l2_px"], 1e-4)

    def test_no_candidates_is_not_an_invented_layout(self):
        a = np.zeros((3, 2), np.float32)
        q = np.zeros((3, 3), np.float32)
        ref = production(a, a, q)
        result = build_candidate_groups(**cache_inputs(a, a, q, ref, padded=True))
        self.assertEqual(result.production_status, "no_candidates")
        self.assertIsNone(result.single_seed)
        self.assertIsNone(result.final_decoded)
        self.assertEqual(result.multi_modes, ())

    def test_insufficient_is_explicit(self):
        a, b, q = offset_fixture([5, 6])
        ref = production(a, b, q)
        result = build_candidate_groups(**cache_inputs(a, b, q, ref))
        self.assertEqual(result.production_status, "insufficient_inliers")
        self.assertEqual(result.single_seed.status, "insufficient_inliers")
        self.assertIsNone(result.final_decoded)
        self.assertEqual(result.multi_modes, ())

    def test_invalid_best_can_still_have_secondary_proposals(self):
        a, b, q = offset_fixture([0, 1, 80, 81, 82], [1, 1, .1, .1, .1])
        ref = production(a, b, q)
        self.assertFalse(ref.valid)
        result = build_candidate_groups(**cache_inputs(a, b, q, ref))
        self.assertEqual(result.production_status, "insufficient_inliers")
        self.assertIsNone(result.final_decoded)
        self.assertEqual(len(result.multi_modes), 1)
        self.assertEqual(result.multi_modes[0].inlier_count, 3)
        self.assertGreater(result.multi_modes[0].translation_rc[0], 75.)
        self.assertEqual(result.diagnostics["mode_insufficient_attempts"], 1)

    def test_max_five_separate_groups_not_merged(self):
        offsets = [60 * mode + d for mode in range(7) for d in (-1, 0, 1)]
        a, b, q = offset_fixture(offsets)
        result = build_candidate_groups(**cache_inputs(a, b, q, production(a, b, q)))
        self.assertEqual(len(result.multi_modes), 5)
        self.assertTrue(all(m.inlier_count == 3 for m in result.multi_modes))
        for left, right in zip(result.multi_modes, result.multi_modes[1:]):
            self.assertGreater(np.linalg.norm(left.translation_rc - right.translation_rc), 20.)

    def test_ab_exchange_preserves_geometric_groups(self):
        a, b, q = offset_fixture([2, 3, 4, 70, 71, 72], [.8, .7, .6, .5, .4, .3])
        ab = build_candidate_groups(**cache_inputs(a, b, q, production(a, b, q)))
        ba = build_candidate_groups(**cache_inputs(b, a, q.T, production(b, a, q.T)))
        for forward, reverse in zip(ab.multi_modes, ba.multi_modes):
            np.testing.assert_allclose(forward.translation_rc, -reverse.translation_rc, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(forward.indices, reverse.indices[:, ::-1])
            np.testing.assert_array_equal(forward.weights, reverse.weights)

    def test_ambiguous_equal_modes_not_silently_valid(self):
        a = np.array([[0., 0.], [100., 0.]], np.float32)
        q = np.array([[0., 1.], [1., 0.]], np.float32)
        ref = production(a, a, q, min_inliers=1)
        self.assertEqual(ref.reason, "ambiguous_equal_modes")
        result = build_candidate_groups(**cache_inputs(a, a, q, ref), min_inliers=1)
        self.assertEqual(result.production_status, "ambiguous_equal_modes")
        self.assertEqual(result.single_seed.status, "ambiguous_equal_modes")
        self.assertIsNone(result.final_decoded)
        self.assertEqual(len(result.multi_modes), 2)
        self.assertEqual(result.multi_modes[0].status, "ambiguous_equal_modes")

    def test_cache_mismatch_raises(self):
        a, b, q = offset_fixture([5, 6, 7, 8])
        ref = production(a, b, q)
        inputs = cache_inputs(a, b, q, ref)
        inputs["final_translation_rc"] += 1
        with self.assertRaisesRegex(ValueError, "translation fails"):
            build_candidate_groups(**inputs)
        inputs = cache_inputs(a, b, q, ref)
        inputs["final_inliers"][0] = False
        with self.assertRaisesRegex(ValueError, "membership fails"):
            build_candidate_groups(**inputs)

    def test_invalid_active_weight_and_endpoint_raise(self):
        a, b, q = offset_fixture([5, 6, 7])
        inputs = cache_inputs(a, b, q, production(a, b, q))
        inputs["candidate_weights"][0] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            build_candidate_groups(**inputs)
        inputs = cache_inputs(a, b, q, production(a, b, q))
        inputs["candidate_indices"][0, 0] = 100
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            build_candidate_groups(**inputs)

    def test_no_gt_label_or_matcher_inputs(self):
        fields = set(inspect.signature(build_candidate_groups).parameters)
        self.assertFalse(fields & {"gt", "label", "target", "assignment", "features_a", "features_b"})
        a, b, q = offset_fixture([5, 6, 7])
        with self.assertRaises(TypeError):
            build_candidate_groups(**cache_inputs(a, b, q, production(a, b, q)), label=1)


if __name__ == "__main__":
    unittest.main()
