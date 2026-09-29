import unittest
import numpy as np
from types import SimpleNamespace
from .pristine import measure,require,audit_exact,unchanged_vertices,negative_guards
from .spec import SPECS
from .weather import depth_field,smooth_profile,contour,weather_fragment
from .light import fragment
from .plan import bucket,group_names
from .evidence import components
from .trim import TARGET_RANGE,AREA_CAP,fraction_accepted

class ReviewTests(unittest.TestCase):
 def test_reference_and_caps(self):
  self.assertEqual(TARGET_RANGE,(.25,.40));self.assertEqual(AREA_CAP,.20)
  for s in SPECS.values():
   self.assertTrue(s['full_generation_authorized']);self.assertEqual(s['review_examples_per_type'],10)
 def test_depth_contracts_and_regions(self):
  arc=np.arange(1000.);edge=np.ones(1000);eligible=(arc>150)&(arc<850)
  for s in SPECS.values():
   for major,weak in [(None,True),('wave',False),('wave',True),('local_abrupt',False),('local_gradual',True),('gaps',False),('gaps',True)]:
    for k in s['notch_counts'] if major=='gaps' else [1]:
     found=[]
     for seed in range(12):
      f=depth_field(arc,edge,eligible,np.random.default_rng(seed),major,weak,k,s)
      if f is None:continue
      found.append(f)
      self.assertTrue(s['main_coverage'][0]<=f['affected_fraction']<=s['main_coverage'][1])
      self.assertLessEqual(f['total'].max(),s['primary_cap'])
      self.assertFalse(np.any((f['total']>0)&~eligible))
      if major=='gaps':self.assertEqual(len(f['regions']),k)
      if major:
       low,high=s['notch'] if major=='gaps' else s['major']
       self.assertTrue(all(low<=r['requested_peak_depth_px']<high for r in f['regions']))
      if weak:self.assertTrue(s['weak'][0]<=f['weak_peak']<s['weak'][1])
     self.assertGreater(len(found),0)
 def test_protected_light(self):
  rr,cc=np.indices((240,240));mask=(rr-120)**2+(cc-120)**2<90**2
  p,_,_=contour(mask);protect=p[:60]
  new,info,field=fragment(mask,mask,None,np.random.default_rng(11),peaks=(1,3),protected=protect)
  self.assertIsNotNone(new);self.assertTrue(.68<=info['actual_affected_fraction']<=.72)
  self.assertTrue(all(1<=p<3 for p in info['requested_peak_depths_px']))
  self.assertTrue(unchanged_vertices(mask,new,protect).all())
  self.assertLess(info['fraction_of_all_undamaged_contour'],.70)
 def test_clean_control_separate(self):
  for v in SPECS:self.assertIn('clean',group_names(v))
 def test_islands_do_not_bridge_gaps(self):
  p=np.c_[np.arange(12),np.zeros(12)];keep=np.ones(12,bool);keep[4:6]=False
  parts=components(p,np.ones(12),keep);self.assertEqual(len(parts),2)
  self.assertEqual(sum(len(p['indices']) for p in parts),10)
 def test_assignment_never_depends_on_augmentation_seed(self):
  self.assertEqual(bucket('parent::original'),bucket('parent::original'))
  self.assertEqual({bucket(str(i)) for i in range(100)},{'v17.5','v18'})
 def test_v19_retired_and_notches_reduced(self):
  self.assertEqual(set(SPECS),{'v17.5','v18'})
  for s in SPECS.values():
   self.assertEqual(s['notch_counts'],[1,2,3,4]);self.assertEqual(s['notch'],[5.,15.])
   self.assertEqual(s['pristine_min_fraction'],0.)
   self.assertFalse(s['pristine_protection_enabled']);self.assertEqual(s['light'],[1.,4.])
   self.assertEqual(s['light_protection_guard_px'],0.)
 def test_unprotected_light_has_no_hidden_preserved_arc(self):
  rr,cc=np.indices((240,240));mask=(rr-120)**2+(cc-120)**2<90**2
  new,info,field=fragment(mask,mask,None,np.random.default_rng(11),peaks=(1,4))
  self.assertIsNotNone(new);self.assertTrue(.68<=info['actual_affected_fraction']<=.72)
  self.assertTrue(all(1<=p<4 for p in info['requested_peak_depths_px']))
  self.assertFalse(info['contact_protection_enabled']);self.assertEqual(info['protection_guard_px'],0.)
  self.assertEqual(info['excluded_protection_length_px'],0.)
  self.assertLessEqual(info['applied_max_depth_px'],4.)
 def test_descriptive_pristine_audit_accepts_below25(self):
  base,bands=self.fixture();new=SimpleNamespace(**vars(base));new.mask_b=base.mask_b.copy();new.mask_b[:,50:130,100]=False
  summary,arrays,_=measure(base,new,bands)
  self.assertLess(summary['conservative_fraction'],.25)
  self.assertEqual(audit_exact(base,new,bands,summary,arrays,minimum=0.)['status'],'passed')
 def fixture(self):
  a=np.zeros((220,220),bool);a[50:151,10:100]=True
  b=np.zeros_like(a);b[50:151,100:190]=True
  base=SimpleNamespace(label=True,mask_a=a[None],mask_b=b[None],translation_a_to_b_rc=np.zeros(2))
  p=np.c_[np.arange(50,151),np.full(101,99.)];q=p+np.array([0.,1.]);v=np.ones(101,bool);v[-1]=False
  bands={'a':(p,np.ones(101),v),'b':(q,np.ones(101),v)}
  return base,bands
 def test_original_denominator_not_surviving_fraction(self):
  base,bands=self.fixture();new=SimpleNamespace(**vars(base));new.mask_a=base.mask_a.copy();new.mask_a[:,:80]=False
  summary,arrays,_=measure(base,new,bands);require(summary)
  self.assertEqual(summary['original_common_length_px'],99.)
  self.assertLess(summary['conservative_fraction'],.71)
  self.assertGreater(summary['conservative_fraction'],.65)
  self.assertEqual(audit_exact(base,new,bands,summary,arrays)['status'],'passed')
 def test_one_side_altered_invalidates_bilateral_arc(self):
  base,bands=self.fixture();new=SimpleNamespace(**vars(base));new.mask_b=base.mask_b.copy();new.mask_b[:,50:130,100]=False
  summary,arrays,_=measure(base,new,bands)
  with self.assertRaises(ValueError):require(summary)
  with self.assertRaises(AssertionError):audit_exact(base,new,bands,summary,arrays)
 def test_neighbour_loss_does_not_count_as_perfect_boundary(self):
  base,bands=self.fixture();new=SimpleNamespace(**vars(base));new.mask_a=base.mask_a.copy();new.mask_a[:,90,98]=False
  summary,arrays,_=measure(base,new,bands)
  self.assertEqual(summary['original_common_length_px'],99.)
  self.assertLess(summary['pristine_common_length_px'],98.)
  self.assertEqual(audit_exact(base,new,bands,summary,arrays)['status'],'passed')
 def test_negative_has_no_fake_common_arc(self):
  base,bands=self.fixture();base.label=False
  with self.assertRaises(ValueError):measure(base,base,bands)
  guard,info=negative_guards(base,base,{'a':.15,'b':.15},np.random.default_rng(5))
  self.assertTrue(all(len(x)>0 for x in guard.values()))
  self.assertTrue(all(not x['gt_seam_used'] for x in info.values()))
 def test_smooth_shoulder_not_minimum_gap(self):
  for mode in ('gradual','wave','weak_gradual','weak_inset','gap'):
   f=smooth_profile(np.array([0.,.999,1.]),12,mode)
   self.assertEqual(f[-1],0);self.assertLess(f[1],12)
 def test_raster_tolerance_never_expands_user_range(self):
  self.assertFalse(fraction_accepted(.245,.25));self.assertFalse(fraction_accepted(.405,.40))
  self.assertTrue(fraction_accepted(.251,.25));self.assertTrue(fraction_accepted(.399,.40))
 def test_actual_target_consistency(self):
  self.assertTrue(fraction_accepted(.333,.33));self.assertFalse(fraction_accepted(.39,.33))
if __name__=='__main__':unittest.main()
