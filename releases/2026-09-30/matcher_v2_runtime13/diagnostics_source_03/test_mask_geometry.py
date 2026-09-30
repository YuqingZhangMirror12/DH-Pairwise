import unittest
import numpy as np

from mask_geometry import rectangularity, seam_profile


def halves():
    a, b = np.zeros((800, 800), bool), np.zeros((800, 800), bool)
    a[100:600, 100:350] = True
    b[100:600, 350:600] = True
    return a, b


class MaskGeometry(unittest.TestCase):
    def test_straight_masks_have_zero_bend(self):
        a, b = halves()
        row = seam_profile(a, b, np.zeros(2))
        self.assertGreater(row['extent_px'], 490)
        self.assertLess(row['bend_range'], 6)
        self.assertGreaterEqual(rectangularity(a), .9)

    def test_no_close_mask_seam(self):
        a, b = halves()
        self.assertIsNone(seam_profile(a, b, np.array([2000., 0.])))

    def test_independent_canvas_translation(self):
        a, b = halves()
        shifted_b = np.roll(b, 30, axis=1)
        self.assertEqual(seam_profile(a, b, np.zeros(2)), seam_profile(a, shifted_b, np.array([0., 30.])))

    def test_empty_mask_is_not_classified_rectangular(self):
        with self.assertRaises(ValueError):
            rectangularity(np.zeros((800, 800), bool))


if __name__ == '__main__':
    unittest.main()
