import unittest
from types import SimpleNamespace
from dataclasses import dataclass, replace
from unittest.mock import patch
import numpy as np
from .crop_floor import measure, evaluate, check
from .trim import positive_trim, negative_trim,proposal_mode
from .light import fragment, contour
from .spec import SPECS


@dataclass
class Pair:
    mask_a: np.ndarray
    mask_b: np.ndarray
    translation_a_to_b_rc: np.ndarray
    label: bool = True


def fixture(short=False):
    a=np.zeros((240,240),bool);a[20:201,10:100]=True
    b=np.zeros_like(a);b[20:201,100:210]=True
    p=np.c_[np.arange(20,201),np.full(181,99.)];q=p+np.array([0.,1.])
    valid=np.ones(181,bool);valid[-1]=False
    if short:valid[70:]=False
    return Pair(a[None],b[None],np.zeros(2)),dict(a=(p,np.ones(181),valid),b=(q,np.ones(181),valid))


class CropOnlyTests(unittest.TestCase):
    def test_both_approved_end_modes_tried_before_skip(self):
        for preferred in ['one','both']:
            self.assertEqual(proposal_mode(preferred,0),preferred)
            self.assertEqual({proposal_mode(preferred,i) for i in range(96)},{'one','both'})
    def test_smaller_is_pixel_area_full_postcut_perimeter(self):
        base,bands=fixture();info,_=measure(base,base,bands)
        self.assertEqual(info['smaller_fragment'],'a')
        self.assertAlmostEqual(info['full_perimeter_px']['a'],contour(base.mask_a[0])[1].sum())
        self.assertAlmostEqual(info['common_over_smaller_perimeter'],179/info['full_perimeter_px']['a'])

    def test_actual_crop_that_crosses20_is_rejected(self):
        base,bands=fixture();m=base.mask_a.copy();m[:,:145]=False
        cropped=replace(base,mask_a=m)
        with self.assertRaisesRegex(ValueError,'crop-only'):evaluate(base,cropped,bands)

    def test_originally_short_must_keep_crop_pixels(self):
        base,bands=fixture(short=True)
        self.assertTrue(evaluate(base,base,bands)['originally_short'])
        m=base.mask_a.copy();m[:,20,10]=False
        with self.assertRaisesRegex(ValueError,'forbidden'):evaluate(base,replace(base,mask_a=m),bands)

    def test_exact20_not_originally_short(self):
        x={'common_over_smaller_perimeter':.20}
        self.assertFalse(check(x,x,False)['originally_short'])
        with self.assertRaises(ValueError):check(x,{'common_over_smaller_perimeter':.19999},True)

    def test_short_sample_skips_without_using_donor_or_rng(self):
        base,bands=fixture(short=True)
        with patch('experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.curriculum_data_v18_v19.trim.source_band',return_value=bands):
            out,plan,_=positive_trim(base,base,None,'one',None,'smaller',.3)
        self.assertIs(out,base);self.assertFalse(plan['applied']);self.assertEqual(plan['mode'],'skipped')
        neg=replace(base,label=False);out,control=negative_trim(neg,plan,None)
        self.assertIs(out,neg);self.assertIsNone(control['crop_floor']);self.assertFalse(control['gt_seam_used'])

    def test_later_erosion_may_cross20_without_changing_crop_verdict(self):
        base,bands=fixture();receipt=evaluate(base,base,bands)
        damaged=base.mask_a.copy();damaged[:,:,98:100]=False
        final=replace(base,mask_a=damaged)
        self.assertLess(measure(base,final,bands)[0]['common_over_smaller_perimeter'],.20)
        self.assertTrue(receipt['passed']);self.assertFalse(receipt['erosion_and_light_subject_to_this_gate'])

    def test_negative_never_gets_fake_common_seam(self):
        base,bands=fixture()
        with self.assertRaises(ValueError):evaluate(replace(base,label=False),base,bands)

    def test_whole_light_does_not_exclude_existing_cut_or_corrosion(self):
        rr,cc=np.indices((240,240));old=(rr-120)**2+(cc-120)**2<90**2
        current=old&(rr>55);p,_,_=contour(old)
        primary={'points':p,'total':np.ones(len(p))*4}
        new,info,field=fragment(old,current,primary,np.random.default_rng(11),peaks=(1,4),whole_contour=True)
        self.assertIsNotNone(new);self.assertTrue(field['eligible'].all())
        self.assertTrue(info['whole_postprimary_contour']);self.assertTrue(.68<=info['actual_affected_fraction']<=.72)
        self.assertLessEqual(info['applied_max_depth_px'],4)

    def test_specs_separate_crop_and_light(self):
        for s in SPECS.values():
            self.assertEqual(s['crop_min_smaller_perimeter_fraction'],.20)
            self.assertEqual(s['light_scope'],'whole_postprimary_contour')
            self.assertEqual(s['light'],[1,4]);self.assertFalse(s['pristine_protection_enabled'])

if __name__=='__main__':unittest.main()
