import unittest
import numpy as np
from .probe import anchor_indices
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour


class SamplingTests(unittest.TestCase):
    def test_nested_uniform_anchors_and_mask_preserved(self):
        mask = np.zeros((800, 800), bool)
        mask[40:760, 100:700] = True
        original = mask.copy()
        a, _ = extract_ordered_outer_contour(mask, cap=512, smoothing_sigma=3.)
        for cap in (1024, 2048):
            b, _ = extract_ordered_outer_contour(mask, cap=cap, smoothing_sigma=3.)
            ids, detail = anchor_indices(a, b)
            self.assertEqual(detail['unique_current_anchors'], 512)
            self.assertEqual(detail['exact_fraction'], 1.)
            np.testing.assert_array_equal(b[ids], a)
        np.testing.assert_array_equal(mask, original)

    def test_duplicate_nearest_anchors_reported(self):
        ids, detail = anchor_indices(np.array([[0., 0.], [0., .1]]), np.array([[0., 0.], [0., 4.]]))
        self.assertEqual(detail['unique_current_anchors'], 1)
        self.assertEqual(ids.tolist(), [0, 0])


if __name__ == '__main__':
    unittest.main()
