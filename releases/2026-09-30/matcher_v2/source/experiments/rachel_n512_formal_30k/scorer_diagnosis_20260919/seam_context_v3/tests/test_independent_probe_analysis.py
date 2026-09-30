import unittest
import numpy as np
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.seam_context_v3.analyze_independent_probes import feature_difference, pool_summary


class ProbeAnalysisTests(unittest.TestCase):
    def test_equal_and_rescaled_features(self):
        x = np.array([[1., 0.], [0., 2.]])
        same = feature_difference(x, x)
        self.assertEqual(same['relative_l2'], 0.)
        larger = feature_difference(x, x*2)
        self.assertAlmostEqual(larger['relative_l2'], 1.)
        self.assertAlmostEqual(larger['cosine']['q10_q25_median_q75_q90'][2], 1.)
        self.assertAlmostEqual(larger['norm_ratio']['q10_q25_median_q75_q90'][2], 2.)

    def test_zeros_and_bad_alignment(self):
        x = np.zeros((2, 3))
        self.assertEqual(feature_difference(x, x)['nonzero_tokens'], 0)
        self.assertIsNone(feature_difference(x, x)['cosine']['q10_q25_median_q75_q90'])
        with self.assertRaises(ValueError):
            feature_difference(x, np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            feature_difference(x, x+np.nan)

    def test_empty_pool_is_not_zero_evidence(self):
        self.assertEqual(pool_summary([])['support_mass']['n'], 0)
        self.assertIsNone(pool_summary([])['support_mass']['q10_q25_median_q75_q90'])


if __name__ == '__main__':
    unittest.main()
