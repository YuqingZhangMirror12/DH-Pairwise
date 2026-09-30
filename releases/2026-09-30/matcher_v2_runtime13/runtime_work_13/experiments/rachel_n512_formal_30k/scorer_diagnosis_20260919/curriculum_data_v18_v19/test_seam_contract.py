import unittest
from dataclasses import replace
from scipy import ndimage
from scipy.spatial import cKDTree
import numpy as np
from .seam_contract import decide, evaluate_masks, require, require_original
from .test_crop_floor import fixture as old_fixture
from ..s7_balanced_v2.latent_seam import contour


def fixture(short=False):
    a=np.zeros((220,240),bool);a[20:201,10:100]=True
    b=np.zeros_like(a);b[20:201,100:210]=True
    masks=dict(a=a,b=b);reference={};points={};bits={};weights={}
    for s,col in [('a',99),('b',100)]:
        p,w=contour(masks[s]);points[s]=p;weights[s]=w
        bits[s]=(p[:,1]==col)&(p[:,0]>=25)&(p[:,0]<=(75 if short else 195))
    for s,t in [('a','b'),('b','a')]:
        q=points[t][bits[t]];p=points[s]
        partner=q[cKDTree(q).query(p)[1]]
        reference.update({s+'_points':p,s+'_partner_points':partner,s+'_edge':weights[s],
                          s+'_source_edges':bits[s]&np.roll(bits[s],-1)})
    return masks,reference


def measure(base,primary,final,reference):
    return evaluate_masks(base,primary,final,reference,np.zeros(2))[0]


class ContractTests(unittest.TestCase):
    def test_boundaries_inclusive(self):
        self.assertTrue(decide(.20,dict(a=.30,b=.30),.30)['eligible'])
        self.assertFalse(decide(.199999,dict(a=1,b=1),1)['eligible'])
        self.assertFalse(decide(.3,dict(a=.299999,b=.299999),.299999)['eligible'])

    def test_does_not_add_an_extra_per_side_percentage_gate(self):
        self.assertTrue(decide(.25,dict(a=.29,b=.33),.31)['eligible'])

    def test_nonfinite_is_not_pass(self):
        with self.assertRaises(ValueError):decide(float('nan'),dict(a=1,b=1),1)
        with self.assertRaises(ValueError):decide(1,dict(a=1,b=float('inf')),1)

    def test_clean_qualifies(self):
        base,ref=fixture();r=measure(base,base,base,ref)
        self.assertTrue(r['eligible']);self.assertAlmostEqual(r['final_over_original'],1)

    def test_originally_short_excluded_even_without_crop(self):
        base,ref=fixture(True);r=measure(base,base,base,ref)
        self.assertFalse(r['original20_pass']);self.assertTrue(r['final30_pass'])
        with self.assertRaisesRegex(ValueError,'original_seam'):require(r)

    def test_early_original_gate_cannot_keep_short_exception(self):
        base,bands=old_fixture(True)
        with self.assertRaisesRegex(ValueError,'original_seam'):require_original(base,bands)

    def test_final_light_four_pixels_on_both_sides_allowed(self):
        base,ref=fixture()
        final={s:ndimage.binary_erosion(m,iterations=4) for s,m in base.items()}
        r=measure(base,base,final,ref)
        self.assertTrue(r['eligible']);self.assertGreater(r['conservative_fraction'],.95)

    def test_same_small_depth_as_primary_does_not_count(self):
        base,ref=fixture();primary={s:ndimage.binary_erosion(m,iterations=1) for s,m in base.items()}
        r=measure(base,primary,primary,ref)
        self.assertEqual(r['final_over_original'],0);self.assertFalse(r['eligible'])

    def test_light_above_four_not_allowed(self):
        base,ref=fixture();final={s:ndimage.binary_erosion(m,iterations=5) for s,m in base.items()}
        with self.assertRaisesRegex(ValueError,'exceeds4'):measure(base,base,final,ref)

    def test_primary_loss_is_not_added_back_by_final_rays(self):
        base,ref=fixture();primary={s:m.copy() for s,m in base.items()}
        primary['a'][25:165,98:100]=False
        r=measure(base,primary,primary,ref)
        self.assertLess(r['conservative_fraction'],.30);self.assertFalse(r['final30_pass'])

    def test_fixed_original_denominator_not_shortened(self):
        base,ref=fixture();primary={s:m.copy() for s,m in base.items()}
        primary['a'][:75]=False
        r=measure(base,primary,primary,ref);clean=measure(base,base,base,ref)
        self.assertEqual(r['original_common_length_px'],clean['original_common_length_px'])
        self.assertLess(r['final_over_original'],.75)
        self.assertTrue(r['eligible'])

    def test_disconnected_segments_do_not_get_bridge_length(self):
        base,ref=fixture();primary={s:m.copy() for s,m in base.items()}
        primary['a'][70:150,98:100]=False
        r=measure(base,primary,primary,ref)
        self.assertLess(r['conservative_fraction'],.55)
        self.assertGreater(r['conservative_fraction'],.4)

    def test_reference_curve_cannot_be_changed(self):
        base,ref=fixture();ref=dict(ref);ref['a_points']=ref['a_points']+1
        with self.assertRaisesRegex(ValueError,'original full'):measure(base,base,base,ref)

    def test_no_added_material(self):
        base,ref=fixture();final={s:m.copy() for s,m in base.items()};final['a'][5,5]=True
        with self.assertRaisesRegex(ValueError,'added material'):measure(base,base,final,ref)


if __name__=='__main__':unittest.main()
