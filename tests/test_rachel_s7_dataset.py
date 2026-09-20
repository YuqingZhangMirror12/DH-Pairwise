"""Bounded S7 adapter tests; no dataset materialization, files, or GPU."""
from collections import Counter
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from staging.pairwise_v0_2.pairwise_data import rachel_s7_dataset as s7
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairSample
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import _identity_view, inherit_pair_targets
from staging.pairwise_v0_2.pairwise_data.rachel_strong_weathering import strong_weather_fragment, StrongWeatheringConfig


def sample(label=True):
    mask = np.zeros((800, 800), bool); mask[180:620, 200:600] = True
    points, valid = extract_ordered_outer_contour(mask, cap=512, smoothing_sigma=3.)
    targets = np.arange(len(points), dtype=np.int64) if label else np.full(len(points), -1, np.int64)
    # This independent source-token fixture intentionally carries an arbitrary
    # pose canary; augmentation inheritance must not use it to re-match points.
    return RachelPairSample(
        pair_id="positive" if label else "negative", fragment_a_token="a", fragment_b_token="b",
        mask_a=mask[None].astype(np.float32), mask_b=mask[None].astype(np.float32),
        coarse_mask_a=np.ones((1, 128, 128), np.float32), coarse_mask_b=np.ones((1, 128, 128), np.float32),
        points_rc_a=points, points_rc_b=points, contour_valid_a=valid, contour_valid_b=valid,
        target_a=targets, target_b=targets.copy(), label=np.float32(label),
        translation_a_to_b_rc=np.array([123., -47.], np.float32),
        translation_a_to_b_xy_cartesian=np.array([-47., -123.], np.float32), translation_valid=np.bool_(label))


class S7DatasetTests(unittest.TestCase):
    def test_geometry_ignore_regions_disable_new_notch_positive_ancestry(self):
        original = sample()
        points = np.array([[100.,100.], [200.,100.], [300.,100.], [400.,100.], [500.,100.], [600.,100.]], np.float32)
        valid = np.ones(6, bool); targets = np.arange(6, dtype=np.int64)
        original = replace(original, points_rc_a=points, points_rc_b=points,
            contour_valid_a=valid, contour_valid_b=valid, target_a=targets, target_b=targets.copy())
        changed = original.mask_a[0].astype(bool).copy(); changed[180,200] = False
        geometry = dict(applied=True, ignore_source_regions=[dict(source_points_rc=[[100.,100.]])])
        projection = lambda *args, **kwargs: (np.arange(6, dtype=np.int64), np.arange(6, dtype=np.int64), {})
        with patch.object(s7, "extract_ordered_outer_contour", return_value=(points, valid)), patch.object(s7, "source_arc_ancestry", side_effect=projection):
            view = s7._changed_view(original, "a", changed, geometry, "local")
            self.assertEqual(view.ancestor[0], -1)
            self.assertEqual(view.representatives[0], -1)
            other = _identity_view(original.mask_b[0].astype(bool), points, valid, "identity", {})
            a, b = inherit_pair_targets(original, view, other)
            self.assertEqual((a[0], b[0]), (-2, -2))
            np.testing.assert_array_equal(a[1:], np.arange(1, 6))
            np.testing.assert_array_equal(b[1:], np.arange(1, 6))
            # No silent positive-label inheritance if notch provenance is absent.
            with self.assertRaises(ValueError):
                s7._changed_view(original, "a", changed, {}, "seam_gaps")
            wave = s7._changed_view(original, "a", changed, {}, "wave")
            np.testing.assert_array_equal(wave.ancestor, np.arange(6))

    def test_negative_gap_uses_own_contour_and_all_dustbin_targets(self):
        original = sample(False); seen = []
        def geometry(mask, rng, **kwargs):
            self.assertEqual(kwargs["mode"], "seam_gaps")
            contour = kwargs["seam_points_rc"]
            self.assertIsNotNone(contour)
            self.assertGreater(len(contour), 1000)
            np.testing.assert_array_equal(contour, extract_ordered_outer_contour(mask, cap=mask.size, smoothing_sigma=0.)[0])
            seen.append(True)
            return strong_weather_fragment(mask, rng, **kwargs, config=StrongWeatheringConfig(gap_count_range=(1,1)))
        with patch.object(s7, "_seam_points", side_effect=AssertionError("negative must not request pair GT seam")):
            result, report = s7.strong_pair(original, "seam_gaps", np.random.default_rng(8), endpoints="a", geometry_function=geometry)
        self.assertEqual(len(seen), 1)
        self.assertTrue(report["changed_pair"], report)
        self.assertTrue(np.all(result.target_a[result.contour_valid_a] == -1))
        self.assertTrue(np.all(result.target_b[result.contour_valid_b] == -1))
        self.assertFalse(report["pose_supervision_enabled"])
        np.testing.assert_array_equal(result.translation_a_to_b_rc, original.translation_a_to_b_rc)
        np.testing.assert_array_equal(result.translation_a_to_b_xy_cartesian, original.translation_a_to_b_xy_cartesian)
        json.dumps(report, allow_nan=False)

    def test_wave_preserves_gt_and_reciprocal_source_edges(self):
        original = sample(True)
        old_mask = original.mask_a.copy()
        result, report = s7.strong_pair(original, "wave", np.random.default_rng(5), endpoints="a")
        self.assertTrue(report["changed_pair"], report)
        self.assertFalse(report["pose_supervision_enabled"])
        self.assertGreaterEqual(int(np.sum(result.target_a >= 0)), 4)
        np.testing.assert_array_equal(result.translation_a_to_b_rc, original.translation_a_to_b_rc)
        np.testing.assert_array_equal(result.translation_a_to_b_xy_cartesian, original.translation_a_to_b_xy_cartesian)
        self.assertEqual(result.translation_valid, original.translation_valid)
        np.testing.assert_array_equal(original.mask_a, old_mask)
        np.testing.assert_array_equal(result.mask_b, original.mask_b)
        for i in np.flatnonzero(result.target_a >= 0):
            self.assertEqual(result.target_b[result.target_a[i]], i)
        json.dumps(report, allow_nan=False)

    def test_coupled_rejection_returns_both_original_members(self):
        positive, negative = sample(True), sample(False)
        fake_positive = replace(positive, mask_a=np.zeros_like(positive.mask_a))
        good_report = s7.changed_report(positive, fake_positive, "wave", detail={"attempt":"positive"})
        skip_report = s7.changed_report(negative, negative, "wave", fallback_reason="thin")
        with patch.object(s7, "strong_pair", side_effect=[(fake_positive, good_report), (negative, skip_report)]):
            rows = s7.strong_group(positive, negative, "wave")
        for source, (result, report) in zip((positive, negative), rows):
            self.assertIs(result, source)
            self.assertFalse(report["changed_pair"])
            self.assertEqual(report["fallback_reason"], "coupled_geometry_or_supervision_rejection")
            np.testing.assert_array_equal(result.target_a, source.target_a)

    def test_24k_schedule_has_exact_balanced_recipe_counts(self):
        protected = tuple(range(1500))
        schedule = s7.recipe_schedule(12000, protected_groups=protected)
        self.assertEqual(schedule, s7.recipe_schedule(12000, protected_groups=protected))
        counts = Counter(schedule)
        self.assertEqual(counts, {key: 12000 * pct // 100 for key, pct in s7.RECIPE_PERCENT.items()})
        self.assertTrue(all(schedule[i] != "gen5_partition" for i in protected))
        # Each group contributes one positive and one negative in the same recipe.
        expanded = Counter((recipe, label) for recipe in schedule for label in (True, False))
        self.assertEqual(sum(expanded.values()), 24000)
        for recipe, count in counts.items():
            self.assertEqual((expanded[recipe, True], expanded[recipe, False]), (count, count))


if __name__ == "__main__":
    unittest.main()
