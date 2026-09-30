import unittest

import numpy as np

from .seam_supervision import correspondence_targets, projected_interval_damage


class SeamSupervisionTests(unittest.TestCase):
    def arrays(self, distance=1.):
        a = np.column_stack([np.arange(0, 100, 4), np.zeros(25)])
        b = a + [0., distance]
        return a, b, np.ones(25, bool), np.zeros(25, bool)

    def test_damage_intervals_project_to_both_sides_include_shoulders(self):
        damage = projected_interval_damage(np.arange(20), [(5, 8), (12, 15)], margin_px=1)
        expected = ((np.arange(20) >= 4) & (np.arange(20) <= 9)) | ((np.arange(20) >= 11) & (np.arange(20) <= 16))
        np.testing.assert_array_equal(damage, expected)
        with self.assertRaises(ValueError):
            projected_interval_damage(np.arange(20), [(8, 5)])

    def test_nearby_mismatch_is_never_targeted_despite_distance(self):
        a, b, eligible, _ = self.arrays()
        damage = projected_interval_damage(a[:, 0], [(32, 48)])
        ta, tb, proof = correspondence_targets(a, b, eligible, eligible, damage, damage, label=True)
        self.assertGreater(proof['rejected_mismatch_matches'], 0)
        np.testing.assert_array_equal(ta[damage], -2)
        np.testing.assert_array_equal(tb[damage], -2)
        np.testing.assert_array_equal(ta[~damage], np.flatnonzero(~damage))
        np.testing.assert_array_equal(tb[ta[ta >= 0]], np.flatnonzero(ta >= 0))

    def test_only_one_damaged_partner_does_not_make_other_false_dustbin(self):
        a, b, eligible, none = self.arrays()
        damage = none.copy(); damage[5] = True
        ta, tb, _ = correspondence_targets(a, b, eligible, eligible, damage, none, label=True)
        self.assertEqual(ta[5], -2)
        self.assertEqual(tb[5], -2)

    def test_nonseam_close_outer_edges_excluded(self):
        a, b, eligible, none = self.arrays()
        eligible[:10] = False
        ta, tb, proof = correspondence_targets(a, b, eligible, eligible, none, none, label=True)
        self.assertEqual(proof['correspondence_count'], 15)
        np.testing.assert_array_equal(ta[:10], -1)
        np.testing.assert_array_equal(tb[:10], -1)

    def test_uniform_wear_does_not_reroll_harder_examples(self):
        a, b, eligible, none = self.arrays(distance=4.5)
        _, _, narrow = correspondence_targets(a, b, eligible, eligible, none, none, label=True)
        _, _, worn = correspondence_targets(a, b, eligible, eligible, none, none, label=True, uniform_wear=(1.2, 1.2))
        self.assertEqual(narrow['correspondence_count'], 0)
        self.assertEqual(worn['correspondence_count'], 25)
        self.assertAlmostEqual(worn['distance_tolerance_px'], 5.4)

    def test_negatives_never_inherit_accidental_geometric_matches(self):
        a, b, eligible, none = self.arrays()
        ta, tb, proof = correspondence_targets(a, b, eligible, eligible, eligible, eligible, label=False)
        np.testing.assert_array_equal(ta, -1)
        np.testing.assert_array_equal(tb, -1)
        self.assertEqual(proof['correspondence_count'], 0)

    def test_invalid_provenance_and_depth_fail_closed(self):
        a, b, eligible, none = self.arrays()
        with self.assertRaises(ValueError):
            correspondence_targets(a, b, eligible.astype(int), eligible, none, none, label=True)
        with self.assertRaises(ValueError):
            correspondence_targets(a, b, eligible, eligible, none, none, label=True, uniform_wear=(2., 0.))


if __name__ == '__main__':
    unittest.main()
