"""Lossless raster-field receipts; samples and tolerances must not change."""
import io
import unittest
from unittest.mock import patch

import numpy as np

from .audit_layered import archived_depth,reconstruct
from .background_recession import fragment
from .conservative_weather import contour,weather_fragment


class RasterPrecisionTests(unittest.TestCase):
    def test_float32_rounding_can_change_a_boundary_pixel(self):
        mask=np.ones((7,7),bool);points=contour(mask)[0]
        exact=np.full(len(points),.5-1e-9,dtype=np.float64)
        data={'a_total':exact.astype(np.float32),'a_total_exact':exact}
        expected=reconstruct(mask,points,exact,3.)
        self.assertTrue(np.array_equal(expected,mask))
        self.assertFalse(np.array_equal(reconstruct(mask,points,data['a_total'],3.),mask))
        self.assertTrue(np.array_equal(reconstruct(mask,points,archived_depth(data,'a'),3.),mask))

    def test_npz_preserves_original_threshold_exactly(self):
        exact=np.array([.5-1e-9,1.5+1e-9,np.sqrt(5.)-.5-1e-9],dtype=np.float64)
        f=io.BytesIO();np.savez_compressed(f,a_total=exact.astype(np.float32),a_total_exact=exact)
        f.seek(0)
        with np.load(f,allow_pickle=False) as data:
            self.assertTrue(np.array_equal(archived_depth(data,'a'),exact))

    def test_inconsistent_or_lossy_exact_field_rejected(self):
        for exact in (np.array([.7]),np.array([.5],dtype=np.float32),np.array([np.nan])):
            with self.assertRaises(ValueError):
                archived_depth({'a_total':np.array([.5],dtype=np.float32),'a_total_exact':exact},'a')

    def test_old_archives_are_not_silently_rewritten(self):
        field=np.array([.5],dtype=np.float32)
        self.assertIs(archived_depth({'a_total':field},'a'),field)

    def test_background_stores_the_field_used_for_rasterization(self):
        mask=np.zeros((96,96),bool);mask[10:86,10:86]=True
        result,info,arrays=fragment(mask,mask,None,np.random.default_rng(391))
        self.assertIsNotNone(result)
        self.assertEqual(arrays['total_exact'].dtype,np.dtype('float64'))
        self.assertTrue(np.array_equal(arrays['total_exact'].astype(np.float32),arrays['total']))
        self.assertTrue(np.array_equal(reconstruct(mask,arrays['points'],arrays['total_exact'],3.),result))

    def test_primary_stores_the_field_used_for_rasterization(self):
        mask=np.zeros((96,96),bool);mask[10:86,10:86]=True
        points=contour(mask)[0];eligible=np.ones(len(points),bool)
        result,info,arrays=weather_fragment(mask,np.random.default_rng(512),eligible,'wave',False,1)
        self.assertIsNotNone(result)
        self.assertEqual(arrays['total_exact'].dtype,np.dtype('float64'))
        self.assertTrue(np.array_equal(arrays['total_exact'].astype(np.float32),arrays['total']))
        self.assertTrue(np.array_equal(reconstruct(mask,arrays['points'],arrays['total_exact'],9.),result))


if __name__=='__main__':unittest.main()
