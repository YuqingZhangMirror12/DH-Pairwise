import unittest
import numpy as np
from geometry_diagnostics import measure, runs


class GeometryTests(unittest.TestCase):
    def pair(self, gap):
        a = np.zeros((130, 130), np.uint8); a[10:110, 10:50] = 1
        b = np.zeros_like(a); b[10:110, 10:50] = 1
        pa = np.array([[10,10],[10,49],[109,49],[109,10]],float)
        # Moving A right by -(40+gap) puts B's left edge next to A's right edge in A frame.
        return measure(a,b,[0,-(40+gap)],pa,pa)
    def test_gap_increase(self):
        near, far = self.pair(2), self.pair(18)
        self.assertGreater(near['length_10_px'], 80)
        self.assertEqual(far['length_10_px'], 0)
        self.assertGreater(far['length_20_px'], 80)
        self.assertGreater(far['gap40']['p50'], near['gap40']['p50'])
    def test_bands_nested(self):
        d = self.pair(7)
        a = [d['length_%d_px'%v] for v in (4,10,20,40,64)]
        self.assertEqual(a,sorted(a))
    def test_cyclic_runs(self):
        self.assertEqual(sorted(map(len,runs([1,1,0,0,1]))),[3])
        self.assertEqual(runs([0,0]),[])
        self.assertEqual(len(runs([1,1])[0]),2)
    def test_swap(self):
        a=np.zeros((130,130),np.uint8);a[10:110,10:50]=1
        b=a.copy();p=np.array([[10,10],[10,49],[109,49],[109,10]],float)
        x=measure(a,b,[0,-47],p,p); y=measure(b,a,[0,47],p,p)
        self.assertAlmostEqual(x['length_20_px'],y['length_20_px'])


if __name__=='__main__':unittest.main()
