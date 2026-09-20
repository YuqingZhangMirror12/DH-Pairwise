"""Small CPU geometry tests, no datasets, files, model weights or GPU."""
import json
import unittest

import numpy as np
from scipy import ndimage

from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour
from staging.pairwise_v0_2.pairwise_data.rachel_strong_weathering import StrongWeatheringConfig, strong_weather_fragment, _seam_eligible


class StrongWeatheringTests(unittest.TestCase):
    @staticmethod
    def paper():
        mask = np.zeros((800, 800), bool)
        mask[160:640, 180:620] = True
        return mask

    def assert_valid(self, original, changed, report):
        self.assertTrue(report["applied"], report)
        self.assertFalse(np.any(changed & ~original))
        self.assertEqual(ndimage.label(changed)[1], 1)
        self.assertEqual(report["removed_area_px"], int(original.sum() - changed.sum()))
        self.assertGreater(report["removed_area_px"], 0)
        self.assertEqual(report["pixels_added"], 0)
        self.assertFalse(report["strength_reduced_on_retry"])
        json.dumps(report, allow_nan=False)

    def test_wave_is_continuous_nonuniform_strong_and_deterministic(self):
        mask = self.paper(); before = mask.copy()
        changed, report = strong_weather_fragment(mask, np.random.default_rng(19), mode="wave", return_depth=True)
        self.assert_valid(mask, changed, report)
        np.testing.assert_array_equal(mask, before)
        depth = np.array(report["depth_diagnostics"]["depth_px"])
        self.assertAlmostEqual(float(depth.min()), 10.)
        self.assertAlmostEqual(float(depth.max()), 30.)
        self.assertGreater(float(depth.std()), 2.)
        self.assertLess(float(np.abs(np.roll(depth, -1) - depth).max()), 2.)
        self.assertGreaterEqual(report["applied_max_depth_px"], 28.5)
        self.assertEqual(report["ignore_source_regions"], [])
        again, again_report = strong_weather_fragment(mask, np.random.default_rng(19), mode="wave", return_depth=True)
        np.testing.assert_array_equal(changed, again)
        self.assertEqual(report, again_report)

    def test_local_notches_have_real_depth_and_ignore_source_geometry(self):
        mask = self.paper()
        changed, report = strong_weather_fragment(mask, np.random.default_rng(23), mode="local")
        self.assert_valid(mask, changed, report)
        self.assertEqual(report["local_notch_count_applied"], report["local_notch_count_requested"])
        for region in report["ignore_source_regions"]:
            self.assertGreaterEqual(region["requested_peak_depth_px"], 10.)
            self.assertLessEqual(region["requested_peak_depth_px"], 30.)
            self.assertGreaterEqual(region["applied_max_depth_px"], region["requested_peak_depth_px"] - 1.5)
            self.assertGreater(len(region["source_points_rc"]), 0)
        self.assertEqual(report["gap_k_applied"], 0)

    def test_five_seam_gaps_are_on_requested_arc_and_not_shortened(self):
        mask = self.paper()
        contour, _ = extract_ordered_outer_contour(mask, cap=mask.size, smoothing_sigma=0.)
        eligible = (contour[:, 0] == 160) & (contour[:, 1] > 195) & (contour[:, 1] < 605)
        config = StrongWeatheringConfig(gap_count_range=(5, 5))
        changed, report = strong_weather_fragment(mask, np.random.default_rng(4), mode="seam_gaps", seam_arc_indices=np.flatnonzero(eligible), config=config)
        self.assert_valid(mask, changed, report)
        self.assertEqual((report["gap_k_requested"], report["gap_k_applied"]), (5, 5))
        removed_rc = np.argwhere(mask & ~changed)
        self.assertTrue(np.all(removed_rc[:, 0] < 191))
        self.assertTrue(np.all(removed_rc[:, 1] > 192))
        for region in report["applied_regions"]:
            self.assertTrue(15 <= region["support_length_px"] <= 50)
            self.assertTrue(np.all(np.asarray(region["source_points_rc"])[:, 0] == 160))
        # Dense source points are an alternate representation, not another mask's NN labels.
        other, r2 = strong_weather_fragment(mask, np.random.default_rng(4), mode="seam_gaps", seam_points_rc=contour[eligible], config=config)
        self.assert_valid(mask, other, r2)
        self.assertEqual(r2["gap_k_applied"], 5)

    def test_short_seam_and_thin_fragment_skip_without_lowering_strength(self):
        thin = np.zeros((120, 120), bool); thin[20:25, 10:100] = True
        changed, report = strong_weather_fragment(thin, np.random.default_rng(2), mode="wave")
        np.testing.assert_array_equal(changed, thin)
        self.assertTrue(report["skipped"])
        self.assertEqual(report["applied_max_depth_px"], 0.)
        self.assertEqual(report["requested_peak_depths_px"], [30.])
        mask = self.paper()
        contour, _ = extract_ordered_outer_contour(mask, cap=mask.size, smoothing_sigma=0.)
        changed, report = strong_weather_fragment(mask, np.random.default_rng(3), mode="seam_gaps", seam_points_rc=contour[:6], config=StrongWeatheringConfig(max_attempts=2))
        np.testing.assert_array_equal(changed, mask)
        self.assertTrue(report["skipped"])
        self.assertEqual(report["gap_k_applied"], 0)
        self.assertTrue(all(15 <= w <= 50 for w in report["requested_support_lengths_px"]))

    def test_topology_rejection_small_contour_and_global_rng_unchanged(self):
        tiny = np.zeros((10, 10), bool); tiny[3, 3] = True
        changed, report = strong_weather_fragment(tiny, np.random.default_rng(9))
        np.testing.assert_array_equal(changed, tiny)
        self.assertTrue(report["skipped"])
        islands = self.paper(); islands[2:5, 2:5] = True
        changed, report = strong_weather_fragment(islands, np.random.default_rng(9), mode="local")
        np.testing.assert_array_equal(changed, islands)
        self.assertEqual(report["component_count"], 2)
        state = np.random.get_state()
        strong_weather_fragment(self.paper(), np.random.default_rng(9), mode="mixed")
        after = np.random.get_state()
        self.assertEqual(state[0], after[0]); np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        with self.assertRaises(ValueError):
            StrongWeatheringConfig(depth_range_px=(2., 4.))
        with self.assertRaises(ValueError):
            strong_weather_fragment(self.paper(), np.random.default_rng(0), mode="seam_gaps", seam_arc_indices=[-1])

    def test_thin_neck_disconnect_proposals_are_not_cropped_to_largest_piece(self):
        mask = np.zeros((800, 800), bool)
        mask[200:500, 100:300] = True
        mask[200:500, 500:700] = True
        mask[347:353, 300:500] = True
        self.assertEqual(ndimage.label(mask)[1], 1)
        changed, report = strong_weather_fragment(mask, np.random.default_rng(41), mode="wave",
                                                  config=StrongWeatheringConfig(max_attempts=2))
        np.testing.assert_array_equal(changed, mask)
        self.assertTrue(report["skipped"])
        self.assertEqual(report["component_count"], 1)
        self.assertEqual(report["last_proposed_component_count"], 2)
        self.assertEqual(report["rejection_counts"]["proposal_empty_or_disconnected"], 2)
        self.assertEqual(report["removed_area_px"], 0)

    def test_only_short_same_source_arc_gaps_bridge_placement(self):
        mask = self.paper()
        contour, _ = extract_ordered_outer_contour(mask, cap=mask.size, smoothing_sigma=0.)
        seam = np.array([[160., col] for col in list(range(220,281,5)) + list(range(330,381,5))])
        strict = _seam_eligible(contour, seam, None, 1.5)
        bridged = _seam_eligible(contour, seam, None, 1.5, 8.)
        segment = (contour[:,0] == 160) & (contour[:,1] >= 220) & (contour[:,1] <= 280)
        self.assertFalse(strict[segment].all())
        self.assertTrue(bridged[segment].all())
        midpoint = (contour[:,0] == 160) & (contour[:,1] == 300)
        self.assertFalse(bridged[midpoint].any())  # Never bridge the 50px unsupported gap.
        self.assertTrue(np.all(~strict | bridged))

    def test_eight_connected_input_retained_without_largest_component_crop(self):
        mask = self.paper(); mask[159,179] = True  # One original corner-touch pixel.
        self.assertEqual(ndimage.label(mask)[1], 2)
        self.assertEqual(ndimage.label(mask, structure=np.ones((3,3)))[1], 1)
        unchanged, strict = strong_weather_fragment(mask, np.random.default_rng(3), mode="wave")
        np.testing.assert_array_equal(unchanged, mask)
        self.assertTrue(strict["skipped"])
        changed, report = strong_weather_fragment(mask, np.random.default_rng(3), mode="wave",
            config=StrongWeatheringConfig(topology_connectivity=8))
        self.assertTrue(report["applied"], report)
        self.assertFalse(np.any(changed & ~mask))
        self.assertEqual(report["original_component_count_4"], 2)
        self.assertEqual(report["original_component_count_8"], 1)
        self.assertEqual(report["component_count_8"], 1)
        self.assertEqual(report["topology_connectivity"], 8)


if __name__ == "__main__":
    unittest.main()
