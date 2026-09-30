import unittest
from types import SimpleNamespace
import numpy as np
from .geometry import area_check,selected_side,curve_field,mask_from_bounds,replay
from .endpoints import source_order,ordered_end_check

class CurvedTrimTests(unittest.TestCase):
    def test_both_size_classes_have_area_cap(self):
        for shape in ((20,20),(40,40)):
            old=np.ones(shape,bool);new=old.copy();new[:shape[0]//4]=False
            with self.assertRaisesRegex(ValueError,'20%'):area_check(old,new)
    def test_exact20percent_area_allowed(self):
        old=np.ones((20,20),bool);new=old.copy();new[:4]=False
        self.assertAlmostEqual(area_check(old,new),.2)
    def test_side_by_pretrim_area_not_letter(self):
        s=SimpleNamespace(mask_a=np.ones((1,20,20)),mask_b=np.ones((1,10,10)))
        self.assertEqual(selected_side(s,'smaller'),'b');self.assertEqual(selected_side(s,'larger'),'a')
    def test_straight_donor_rejected(self):
        for p in (np.zeros(129),np.linspace(0,1,129)):
            with self.assertRaisesRegex(ValueError,'straight'):curve_field(np.ones((20,20),bool),p,30,False,False)
    def test_curve_replay_pixel_exact(self):
        old=np.ones((80,80),bool);p=.035*np.sin(np.linspace(0,4*np.pi,129))
        pixels,scores,frame=curve_field(old,p,43,True,False)
        v=scores(pixels);upper=float(np.quantile(v,.8))
        new=mask_from_bounds(old,pixels,v,None,upper)
        np.testing.assert_array_equal(new,replay(old,dict(frame,profile=p.tolist(),lower=None,upper=upper)))
    def test_no_material_addition(self):
        old=np.zeros((80,80),bool);old[5:70,10:60]=True
        p=.02*np.sin(np.linspace(0,6*np.pi,129));pixels,scores,_=curve_field(old,p,0,False,False)
        new=mask_from_bounds(old,pixels,scores(pixels),None,45)
        self.assertFalse(np.any(new&~old))
    def test_single_end_only(self):
        for keep in ([0,0,1,1,1],[1,1,1,0,0]):
            self.assertEqual(ordered_end_check(keep,np.ones(5),'one')['interior_removed_px'],0)
    def test_both_ends(self):
        out=ordered_end_check([0,1,1,1,0],np.ones(5),'both')
        self.assertEqual(out['removed_start_px'],1);self.assertEqual(out['removed_end_px'],1)
    def test_interior_cut_rejected(self):
        for keep in ([1,1,0,1,1],[0,1,0,1,0]):
            with self.assertRaisesRegex(ValueError,'interior'):ordered_end_check(keep,np.ones(5),'both')
    def test_one_cannot_mean_two_and_vice_versa(self):
        with self.assertRaisesRegex(ValueError,'requested'):ordered_end_check([0,1,0],np.ones(3),'one')
        with self.assertRaisesRegex(ValueError,'requested'):ordered_end_check([0,1,1],np.ones(3),'both')
    def test_unwrap_original_not_proposed_cut(self):
        valid=np.array([1,1,0,0,0,0,1,1,1,0],bool)
        np.testing.assert_array_equal(source_order(np.ones(10),valid),[6,7,8,0,1])
    def test_closed_seam_rejected(self):
        with self.assertRaisesRegex(ValueError,'closed'):source_order(np.ones(10),np.ones(10,bool))
    def test_fractional_edges_do_not_create_false_wrap_gaps(self):
        weights=np.full(20,np.sqrt(2.));valid=np.zeros(20,bool);valid[2:10]=True
        np.testing.assert_array_equal(source_order(weights,valid),np.arange(2,10))

if __name__=='__main__':unittest.main()
