import copy
import unittest

from .analyze import displacement, measure


def fixture():
    return {"label": True, "layout_gt_available": True, "target_translation_rc": [3., 4.],
            "points_rc_a": [[5., 9.]],
            "points_rc_b": [[8., 13.], [14., 21.], [20., 29.], [8., 33.0000001]],
            "valid_a": [True], "valid_b": [True] * 4,
            "layout": {"candidate_indices": [[0, j] for j in range(4)],
                       "candidate_sinkhorn_mass": [.4, .3, .2, .1],
                       "inlier_mask": [True, True, False, False], "candidate_count": 4,
                       "inlier_count": 2, "valid": True, "t_a_to_b_rc": [3., 4.],
                       "translation_l2_px": 0.}}


class CandidateGTTests(unittest.TestCase):
    def test_translation_sign_and_boundary(self):
        self.assertEqual(displacement([5, 9], [8, 13]), (3, 4))
        m = measure(fixture())
        self.assertEqual(m["gt_neighborhoods"]["10"]["candidate_count"], 2)
        self.assertEqual(m["gt_neighborhoods"]["20"]["candidate_count"], 3)
        self.assertAlmostEqual(m["gt_neighborhoods"]["10"]["candidate_mass"], .7)
        self.assertEqual(m["gt_neighborhoods"]["10"]["predicted_inlier_gt_consistent_count_fraction"], 1.)
        self.assertAlmostEqual(m["gt_neighborhoods"]["20"]["mass_relative_to_predicted_inliers"], .9/.7)
        self.assertEqual(m["gt_neighborhoods"]["20"]["gt_consistent_edges_with_both_endpoints_in_predicted_token_sets"], 2)
        self.assertAlmostEqual(m["gt_neighborhoods"]["20"]["fraction_gt_consistent_mass_with_both_endpoints_retained"], .7/.9)

    def test_swap_invariance(self):
        row = fixture(); swapped = copy.deepcopy(row)
        swapped["points_rc_a"], swapped["points_rc_b"] = row["points_rc_b"], row["points_rc_a"]
        swapped["valid_a"], swapped["valid_b"] = row["valid_b"], row["valid_a"]
        swapped["target_translation_rc"] = [-3., -4.]
        swapped["layout"]["t_a_to_b_rc"] = [-3., -4.]
        swapped["layout"]["candidate_indices"] = [[j, i] for i, j in row["layout"]["candidate_indices"]]
        a, b = measure(row), measure(swapped)
        for radius in ("10", "20"):
            for name in ("candidate_count", "candidate_mass", "predicted_inlier_gt_consistent_count_fraction"):
                self.assertEqual(a["gt_neighborhoods"][radius][name], b["gt_neighborhoods"][radius][name])

    def test_exclude_nonpositive_or_missing_gt_and_zero_support(self):
        for k in ("label", "layout_gt_available"):
            row = fixture(); row[k] = False
            with self.assertRaises(ValueError): measure(row)
        row = fixture()
        row["layout"].update(inlier_mask=[False]*4, inlier_count=0, valid=False, translation_l2_px=None)
        self.assertIsNone(measure(row)["gt_neighborhoods"]["10"]["mass_relative_to_predicted_inliers"])


if __name__ == "__main__":
    unittest.main()
