import unittest
import numpy as np
from .geometry import topology,reconstruct,weighted_quantile
from .fixed_gap import source_pairs


class GeometryTests(unittest.TestCase):
    def test_cumulative_removal_not_only_stagewise(self):
        original=np.zeros((100,100),bool);original[10:90,10:90]=True
        stage=original.copy();stage[45:]=False
        final=stage.copy();final[:,45:]=False
        topology(original,stage);topology(stage,final)
        with self.assertRaisesRegex(ValueError,'25%'):topology(original,final)

    def test_connected_but_added_material_is_rejected(self):
        original=np.zeros((50,50),bool);original[10:40,10:40]=True
        new=original.copy();new[9,10:40]=True
        with self.assertRaisesRegex(ValueError,'added'):topology(original,new)

    def test_disconnected_is_rejected(self):
        original=np.ones((50,50),bool);new=original.copy();new[:,24:26]=False
        with self.assertRaisesRegex(ValueError,'disconnected'):topology(original,new)

    def test_new_hole_is_rejected(self):
        original=np.ones((50,50),bool);new=original.copy();new[20:25,20:25]=False
        with self.assertRaisesRegex(ValueError,'hole'):topology(original,new)

    def test_zero_depth_does_not_remove_pixels(self):
        original=np.zeros((50,50),bool);original[10:40,10:40]=True
        points=np.argwhere(original & ~np.pad(np.ones((28,28),bool),11))
        np.testing.assert_array_equal(reconstruct(original,points,np.zeros(len(points))),original)

    def test_weighted_quantile_orders_curve(self):
        q=weighted_quantile(np.array([3,1,2]),np.ones(3),[0,1])
        np.testing.assert_array_equal(q,[1,3])

    def test_original_partner_is_not_rematched_after_cut(self):
        proof=dict(a_physically_retained=np.array([True,False]),a_source_points=np.array([[0.,0.],[10.,0.]]),
            a_partner_points=np.array([[0.,1.],[10.,1.]]),a_source_weights=np.ones(2))
        p,q,w,ok=source_pairs(proof,'a',np.zeros(2))
        np.testing.assert_array_equal(q,[[0.,1.]])
        self.assertTrue(ok[0]);self.assertEqual(w.sum(),1.)

    def test_uncertain_source_partner_is_not_zero_filled(self):
        proof=dict(a_physically_retained=np.array([True]),a_source_points=np.array([[0.,0.]]),
            a_partner_points=np.array([[20.,1.]]),a_source_weights=np.ones(1))
        p,q,w,ok=source_pairs(proof,'a',np.zeros(2))
        self.assertFalse(ok[0]);self.assertEqual(w.sum(),1.)
        self.assertGreater(np.linalg.norm(p-q),20.)

if __name__=='__main__':unittest.main()
