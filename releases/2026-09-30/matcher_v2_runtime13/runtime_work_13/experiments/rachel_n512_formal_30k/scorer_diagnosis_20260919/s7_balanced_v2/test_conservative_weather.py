import unittest
import numpy as np
from scipy import ndimage

from .conservative_weather import smooth_profile,depth_field,contour,weather_fragment,parts
from .geometry import schedules
from pathlib import Path
import json


class ConservativeTest(unittest.TestCase):
    def test_exact_quotas(self):
        profile=json.loads(Path(__file__).with_name('distribution_profile_v12.json').read_text())
        from collections import Counter
        for n in (400,12000):
            recipes,partial,bins,mirror=schedules(n,26092412,profile)
            self.assertEqual(Counter(recipes),{k:n*v//100 for k,v in profile['corrosion_percent'].items()})
            self.assertEqual(sum(parts(r)[1] for r in recipes),n//2)
            self.assertEqual(sum(partial),n*7//10)
            self.assertEqual(sum(v>0 for v in mirror),n*15//100)
            self.assertEqual(sum(r.startswith('local_abrupt') for r in recipes),n//10)
            self.assertEqual(sum(r.startswith('local_gradual') for r in recipes),n//10)

    def test_weak_smooth_connected_support(self):
        u=np.linspace(-1.05,1.05,4201)
        for mode in ('weak_gradual','weak_inset'):
            for peak in (1.,2.5,4.):
                p=smooth_profile(u,peak,mode,1.3)
                self.assertLessEqual(p.max(),peak)
                self.assertEqual(ndimage.label(p>0)[1],1)
                self.assertLess(np.abs(np.diff(p)).max(),.025)
                self.assertEqual(p[0]+p[-1],0.)

    def test_gradual_vs_abrupt(self):
        u=np.array([-.999,0.,.999])
        gradual=smooth_profile(u,9.,'gradual')
        abrupt=smooth_profile(u,9.,'abrupt')
        self.assertLess(gradual[0],1.001)
        self.assertEqual(gradual[1],9.)
        self.assertGreater(abrupt[0],8.)

    def test_masks_depth_coverage_and_connectivity(self):
        mask=np.zeros((220,220),bool);mask[20:200,20:200]=True
        p,e,a=contour(mask);eligible=p[:,1]>198
        for recipe in ('mild','wave','wave_weak','local_abrupt','local_gradual_weak','gaps','gaps_weak'):
            major,weak=parts(recipe)
            for seed in range(4):
                out,d,arrays=weather_fragment(mask,np.random.default_rng(seed),eligible,major,weak,2 if major=='gaps' else 1)
                self.assertIsNotNone(out,(recipe,d))
                self.assertLessEqual(d['applied_max_depth_px'],9.)
                self.assertLessEqual(d['affected_fraction'],.5)
                self.assertEqual(ndimage.label(out,np.ones((3,3)))[1],1)
                self.assertFalse(np.any(out & ~mask))
                self.assertFalse(np.any(arrays['total'][~eligible]))
                if weak:self.assertGreaterEqual(d['weak_independently_removed_pixels'],4)

    def test_short_invalid_scope_is_rejected(self):
        edge=np.ones(100);arc=np.arange(100);eligible=arc<8
        self.assertIsNone(depth_field(arc,edge,eligible,np.random.default_rng(1),'wave',True))

    def test_weak_overlay_is_one_continuous_arc_even_with_three_gaps(self):
        edge=np.ones(1000);arc=np.arange(1000);eligible=(arc>=100)&(arc<=650)
        for seed in range(20):
            f=depth_field(arc,edge,eligible,np.random.default_rng(seed),'gaps',True,3)
            self.assertIsNotNone(f)
            self.assertEqual(ndimage.label(f['weak']>0)[1],1)


if __name__=='__main__':unittest.main()
