import unittest
from types import SimpleNamespace

import numpy as np
import torch

from .calibrate_geometry import inherited_edges, quantile, derive_parameters
from .geometry import pair_frame


def sample():
    # A square to the left of B, both in one coordinate frame. A right edge
    # retreats left by 2 and B left edge retreats right by3: positive gap5.
    return SimpleNamespace(label=1, translation_valid=True,
        points_rc_a=np.array([[0.,0.],[0.,8.],[5.,8.],[10.,8.],[10.,0.]]),
        points_rc_b=np.array([[0.,13.],[0.,20.],[10.,20.],[10.,13.],[5.,13.]]),
        contour_valid_a=np.ones(5,bool), contour_valid_b=np.ones(5,bool),
        target_a=np.array([-1,-2,4,-2,-1]), target_b=np.array([-2,-1,-1,-2,2]),
        translation_a_to_b_rc=np.zeros(2))


class CalibrationTests(unittest.TestCase):
    def test_erosion_is_positive_normal_not_tangential_slip(self):
        edges = inherited_edges(sample(), 'wave')
        self.assertEqual(edges.shape, (1,11))
        np.testing.assert_allclose(edges[0,4:7], [5,0,5])
        self.assertEqual(edges[0,10], 0)

    def test_precise_flag_not_inferred_from_residual(self):
        s = sample()
        self.assertEqual(inherited_edges(s,'clean')[0,10], 1)
        s.points_rc_b[:,1] -= 5
        edges = inherited_edges(s, 'wave')
        self.assertEqual(edges[0,6], 0)
        self.assertEqual(edges[0,10], 0)

    def test_unknowns_never_filled_and_reciprocity_required(self):
        s = sample()
        self.assertEqual(len(inherited_edges(s,'gaps')), 1)
        s.target_b[4] = -2
        with self.assertRaises(ValueError):
            inherited_edges(s,'gaps')

    def test_mirror_keeps_gap_sign(self):
        s = sample()
        base = inherited_edges(s, 'wave')
        s.points_rc_a[:,1] = 40-s.points_rc_a[:,1]
        s.points_rc_b[:,1] = 40-s.points_rc_b[:,1]
        reflected = inherited_edges(s, 'wave')
        np.testing.assert_allclose(base[:,4:7], reflected[:,4:7])

    def test_frame_swap_and_unreliable_normals(self):
        a = torch.tensor([[0.,1.],[1.,0.]])
        b = torch.tensor([[0.,-1.],[1.,0.]])
        r = torch.ones(2)
        n,t,v = pair_frame(a,b,r,r)
        ns,ts,vs = pair_frame(b,a,r,r)
        torch.testing.assert_close(n,-ns)
        torch.testing.assert_close(t,-ts)
        torch.testing.assert_close(v,vs)
        self.assertEqual(v[1],0)
        self.assertTrue(torch.isfinite(n).all())

    def test_weighted_statistics_reject_invalid_values(self):
        q = quantile([0,10], [1,3])
        self.assertAlmostEqual(q['mean'],7.5)
        self.assertEqual(quantile([])['n'],0)
        with self.assertRaises(ValueError):
            quantile([1,np.nan])
        with self.assertRaises(ValueError):
            quantile([1,2], [0,0])

    def test_tolerances_only_from_reliable_inherited_edges(self):
        clean = {'reliable_normal_edges':1000, 'pair_weighted':{
            'abs_normal_per_spacing':{'p95':.4}, 'abs_tangent_per_spacing':{'p95':.6},
            'norm_per_spacing':{'p95':.7}}}
        damaged = {'reliable_normal_edges':1000, 'pair_weighted':{
            'normal_px':{'p99':36.5}, 'abs_tangent_per_spacing':{'p95':1.5}}}
        p = derive_parameters({'recipe:clean':clean, 'corroded':damaged,
                               'projected_pair_p99_distribution':{'p99':9999.}})
        self.assertEqual(p['damage_normal_upper_px'],37.)
        self.assertGreater(p['evidence_tangent_sigma_per_spacing'],p['tangent_sigma_per_spacing'])
        damaged['reliable_normal_edges'] = 3
        with self.assertRaises(ValueError):
            derive_parameters({'recipe:clean':clean,'corroded':damaged})


if __name__ == '__main__':
    unittest.main()
