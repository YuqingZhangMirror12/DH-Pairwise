"""Three focused numerical tests of unchanged zero-gap production decoding."""
import unittest

import numpy as np

from . import oracle_geometry as o


class OracleGeometryTests(unittest.TestCase):
    def inputs(self,count=5,missing_band=0.):
        pa=np.array([[10.,2.],[10.,12.],[10.,22.],[10.,32.],[10.,42.]])[:count]
        gt=np.array([7.,-3.])
        pb=pa+gt+np.array([missing_band,0.])
        valid=np.ones(count,bool)
        target=np.arange(count,dtype=np.int64)
        return pa,pb,valid,valid.copy(),target,target.copy(),gt

    def test_clean_translation_recovers_with_original_noncontiguous_indices(self):
        args=list(self.inputs(count=3))
        for side in (0,1):
            points=np.zeros((8,2));points[[0,3,7]]=args[side]
            args[side]=points
            valid=np.zeros(8,bool);valid[[0,3,7]]=True
            args[2+side]=valid
            target=np.full(8,-2,np.int64);target[[0,3,7]]=[0,3,7]
            args[4+side]=target
        row=o.oracle_evidence(*args)
        self.assertTrue(row['oracle_layout_valid'])
        self.assertTrue(row['oracle_layout20_success'])
        self.assertEqual(row['oracle_error_px'],0.)
        self.assertEqual(row['oracle_translation_a_to_b_rc'],[7.,-3.])
        self.assertEqual(row['gt_edge_count'],3)
        self.assertEqual(row['oracle_candidate_count'],3)
        self.assertEqual(row['gt_edge_error_median_px'],0.)

    def test_missing_material_band_biases_perfect_label_correspondences(self):
        for band in (20.,25.,30.):
            with self.subTest(missing_band=band):
                row=o.oracle_evidence(*self.inputs(missing_band=band))
                self.assertTrue(row['oracle_layout_valid'])
                self.assertEqual(row['gt_edge_count'],5)
                self.assertEqual(row['oracle_candidate_count'],5)
                self.assertEqual(row['oracle_inlier_count'],5)
                self.assertEqual(row['oracle_residual_px'],0.)
                self.assertEqual(row['oracle_error_px'],band)
                self.assertEqual(row['gt_edge_error_median_px'],band)
                self.assertEqual(row['gt_edge_error_p90_px'],band)
                self.assertEqual(row['oracle_layout20_success'],band<=20.)

    def test_fewer_than_three_targets_invalid_but_retained_in_positive_denominator(self):
        good=o.oracle_evidence(*self.inputs())
        sparse=o.oracle_evidence(*self.inputs(count=2))
        self.assertFalse(sparse['oracle_layout_valid'])
        self.assertFalse(sparse['oracle_layout20_success'])
        self.assertEqual(sparse['oracle_reason'],'insufficient_inliers')
        self.assertIsNone(sparse['oracle_error_px'])
        rows=[dict(good,model_layout20_success=False,model_raw_error_px=30.),
              dict(sparse,model_layout20_success=False,model_raw_error_px=None)]
        summary=o.summarize(rows)
        self.assertEqual(summary['positive_denominator'],2)
        self.assertEqual(summary['oracle_layout20_rate'],.5)
        self.assertEqual(summary['fewer_than_3_target_edges_count'],1)
        self.assertEqual(summary['joint']['model_fail_oracle_success'],1)
        self.assertEqual(summary['joint']['both_fail'],1)


if __name__=='__main__':
    unittest.main()
