"""CPU-only range, continuous shape, K4 pixels and source-isolation checks."""
import unittest
import numpy as np
from .weather import depth_field,smooth_profile,contour,weather_fragment
from ..s7_balanced_v2.conservative_weather import smooth_profile as original_shape

class DepthTests(unittest.TestCase):
    def setUp(self):
        self.arc=np.arange(1200,dtype=float);self.edge=np.ones(1200);self.eligible=np.zeros(1200,bool);self.eligible[100:1100]=True
    def field(self,major,weak,k=1,seed=7):
        f=depth_field(self.arc,self.edge,self.eligible,np.random.default_rng(seed),major,weak,k)
        self.assertIsNotNone(f);return f
    def test_weak_new_range(self):
        values=[self.field(None,True,seed=i)['weak_peak'] for i in range(60)]
        self.assertGreaterEqual(min(values),3);self.assertLess(max(values),8);self.assertGreater(max(values),7)
    def test_every_major_peak_range(self):
        for major in ('wave','local_abrupt','local_gradual','gaps'):
            values=[r['requested_peak_depth_px'] for i in range(30) for r in self.field(major,False,4 if major=='gaps' else 1,i)['regions']]
            self.assertGreaterEqual(min(values),5);self.assertLess(max(values),15);self.assertGreater(max(values),13)
    def test_combined_cap_and_no_major_combination(self):
        for major in ('wave','local_abrupt','local_gradual','gaps'):
            f=self.field(major,True,4 if major=='gaps' else 1)
            np.testing.assert_array_equal(f['total'],np.minimum(15,f['base']+f['weak']))
            self.assertLessEqual(f['affected_fraction'],.5)
    def test_shapes_unchanged(self):
        u=np.linspace(-1.5,1.5,1001)
        for mode in ('wave','abrupt','gradual','gap','weak_gradual','weak_inset'):
            np.testing.assert_array_equal(smooth_profile(u,7,mode,.31),original_shape(u,7,mode,.31))
    def test_gradual_starts_at_one(self):
        x=smooth_profile(np.array([-.999999]),12,'gradual',0.)[0]
        self.assertAlmostEqual(x,1,places=8)
    def test_background_original_module_unchanged(self):
        from ..s7_balanced_v2.background_recession import make_field
        f,info=make_field(self.arc,self.edge,self.eligible,np.ones(1200)*.5,np.random.default_rng(5))
        self.assertLessEqual(f.max(),3);self.assertTrue(all(1<=x<3 for x in info['requested_peak_depths_px']))
    def test_four_effective_notches_on_real_pixels(self):
        mask=np.zeros((400,800),bool);mask[40:360,40:760]=True
        p,edge,arc=contour(mask);eligible=(p[:,0]==40)&(p[:,1]>90)&(p[:,1]<710)
        new,info,arrays=weather_fragment(mask,np.random.default_rng(19),eligible,'gaps',False,4)
        self.assertIsNotNone(new);self.assertEqual(info['notch_count'],4)
        self.assertTrue(all(v>=8 for v in info['notch_independently_removed_pixels']))
        self.assertGreater(np.count_nonzero(mask&~new),0)
    def test_k_bounds(self):
        for k in (0,5):
            with self.assertRaises(ValueError):self.field('gaps',False,k)
    def test_no_depth_outside_eligible(self):
        for major in (None,'wave','local_abrupt','local_gradual','gaps'):
            f=self.field(major,True,4 if major=='gaps' else 1)
            self.assertTrue(np.all(f['total'][~self.eligible]==0))

if __name__=='__main__':unittest.main()
