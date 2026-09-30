import unittest

import numpy as np

from .analyze import propose_modes


class ModeTests(unittest.TestCase):
    def test_separated_modes_and_scale(self):
        delta = np.array([[0., 0.], [0., 1.], [1., 0.], [100., 0.], [100., 1.], [101., 0.]])
        mass = np.array([2., 2., 2., 1., 1., 1.])
        result = propose_modes(delta, mass)
        self.assertEqual([m['inlier_count'] for m in result], [3, 3])
        np.testing.assert_allclose(result[0]['translation_rc'], [1/3, 1/3])
        np.testing.assert_allclose(result[1]['translation_rc'], [100+1/3, 1/3])
        scaled = propose_modes(delta, mass * .001)
        for a, b in zip(result, scaled):
            np.testing.assert_allclose(a['translation_rc'], b['translation_rc'])

    def test_sign_reversal(self):
        delta = np.array([[0., 0.], [0., 1.], [1., 0.], [100., 0.], [100., 1.], [101., 0.]])
        mass = np.array([2., 2., 2., 1., 1., 1.])
        for a, b in zip(propose_modes(delta, mass), propose_modes(-delta, mass)):
            np.testing.assert_allclose(a['translation_rc'], -np.array(b['translation_rc']))

    def test_too_few_inliers(self):
        self.assertEqual(propose_modes(np.array([[0., 0.], [100., 0.]]), np.ones(2)), [])


if __name__ == '__main__':
    unittest.main()
