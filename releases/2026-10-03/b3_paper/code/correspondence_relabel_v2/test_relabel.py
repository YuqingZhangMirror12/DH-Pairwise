"""New label-policy tests, deliberately no training or neural forwards."""
import unittest
import numpy as np
from dataclasses import asdict
from scipy.spatial import cKDTree
from relabel import run, EvidenceError, POLICY, arc_expand, runs, parent_to_canvas, added_overlap_on_tokens
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour

def fixture(erosion=0,recipe='wave',negative=False):
    shape=(220,220); orig={s:np.zeros(shape,bool) for s in 'ab'}
    orig['a'][20:200,20:101]=True;orig['b'][20:200,101:202]=True
    final={s:m.copy() for s,m in orig.items()}
    if erosion:final['a'][:,101-erosion:]=False
    z={}; proof={}; pre={'translation_a_to_b_rc':np.zeros(2)};latent={}
    for s in 'ab':
        p,v=extract_ordered_outer_contour(final[s],cap=128,smoothing_sigma=3.)
        z['points_rc_'+s]=p;z['contour_valid_'+s]=v
        z['target_'+s]=np.full(len(p),-1,np.int64)
        z['mask_'+s+'_packed']=np.packbits(final[s][None],axis=-1)
        z['mask_'+s+'_shape']=np.array((1,*shape))
        for st,m in [('fragment',orig[s]),('trim',orig[s]),('primary',final[s]),('final',final[s])]:
            # Production uses800 (divisible by8), so tests pad to a divisible canvas.
            proof['packed_'+st+'_'+s]=np.packbits(m,axis=-1)
        pre['packed_preweather_'+s]=np.packbits(orig[s],axis=-1)
    z['label']=np.array(0 if negative else 1);z['translation_a_to_b_rc']=np.zeros(2)
    if not negative:
        pa,pb=z['points_rc_a'],z['points_rc_b']; d,j=cKDTree(pb).query(pa);back=cKDTree(pa).query(pb)[1]
        for i,jj in enumerate(j):
            if back[jj]==i and d[i]<=40:z['target_a'][i]=jj;z['target_b'][jj]=i
    # Use 224-wide masks throughout to avoid implicit packed padding.
    for s in 'ab':
        for st in ('fragment','trim','primary','final'):
            proof['packed_'+st+'_'+s]=np.packbits(np.pad(orig[s] if st in ('fragment','trim') else final[s],((0,0),(0,4))),axis=-1)
        pre['packed_preweather_'+s]=proof['packed_fragment_'+s].copy()
        m=np.pad(final[s],((0,0),(0,4)));z['mask_'+s+'_packed']=np.packbits(m[None],axis=-1);z['mask_'+s+'_shape']=np.array((1,220,224))
        src=np.c_[np.arange(25,195),np.full(170,100 if s=='a' else 101)].astype(float)
        q=src.copy();q[:,1]=101 if s=='a' else 100
        p=src.copy();pp=q.copy()
        if s=='a':p[:,1]-=erosion
        else:pp[:,1]-=erosion
        values=dict(source_points=src,partner_source_points=q,projected_points=p,partner_projected_points=pp,
                    valid=np.ones(170,bool),gap=np.linalg.norm(p-pp,axis=1),source_gap=np.ones(170),
                    recession=np.linalg.norm(p-src,axis=1),partner_recession=np.linalg.norm(pp-q,axis=1))
        latent.update({s+'_'+k:v for k,v in values.items()})
    return z,proof,dict(recipe=recipe,v14_fallback=False),pre,latent

class PolicyTests(unittest.TestCase):
    def test_unchanged_subpixel_gt_contact_is_not_augmented_overlap(self):
        z,p,r,w,l=fixture(recipe='clean');t=np.array([0.,2.2])
        z['translation_a_to_b_rc']=t;w['translation_a_to_b_rc']=t
        for s in 'ab':
            shift=t if s=='a' else -t
            l[s+'_gap']=np.linalg.norm(l[s+'_projected_points']+shift-l[s+'_partner_projected_points'],axis=1)
            l[s+'_source_gap']=np.linalg.norm(l[s+'_source_points']+shift-l[s+'_partner_source_points'],axis=1)
        arrays,info=run(z,p,r,w,l)
        self.assertGreater((arrays['full_a']>=0).sum(),20)
        self.assertFalse(arrays['destructive_a'].any())
        self.assertFalse(arrays['destructive_b'].any())
    def test_unrecorded_added_raster_overlap_protected(self):
        a=np.zeros((30,30),bool);b=a.copy();a[3:27,3:15]=1;b[3:27,15:27]=1
        original={'a':a,'b':b};aa=a.copy();aa[3:27,15:19]=1;final={'a':aa,'b':b}
        points=np.array([[12.,18.],[12.,3.]])
        result=added_overlap_on_tokens(original,final,points,np.zeros(2),'a','b')
        np.testing.assert_array_equal(result,[True,False])
    def test_clean_near_points_not_dustbin(self):
        z,p,r,w,l=fixture(recipe='clean');z['target_a'][:]=-1;z['target_b'][:]=-1
        a,info=run(z,p,r,w,l)
        self.assertGreater((a['full_a']>=0).sum(),20)
        self.assertFalse(((a['full_a']==-1)&(a['near_distance_a']<=3)).any())
        self.assertFalse((a['tight_a']>=0).any())
    def test_deep_recession_not_matched(self):
        z,p,r,w,l=fixture(erosion=16);a,info=run(z,p,r,w,l)
        self.assertGreater((z['target_a']>=0).sum(),0)
        self.assertFalse((a['full_a']>=0).any());self.assertFalse((a['tight_a']>=0).any())
    def test_smooth_5_to_8_source_anchored(self):
        z,p,r,w,l=fixture(erosion=5);a,info=run(z,p,r,w,l)
        wide=[x for x in info['pairs'] if x['distance_px']>5]
        self.assertGreater(len(wide),4)
        self.assertTrue(all(x['pairing']=='pre_damage_source_mnn' and x['damage']=='smooth_recession' for x in wide))
    def test_local_region_excluded_with_shoulder(self):
        z,p,r,w,l=fixture(erosion=2,recipe='local_abrupt')
        for s in 'ab':
            pts=l[s+'_source_points'];p['primary_'+s+'_points']=pts
            p['primary_'+s+'_major']=((pts[:,0]>70)&(pts[:,0]<150)).astype(float)
        a,info=run(z,p,r,w,l)
        self.assertFalse(np.any((a['full_a']>=0)&a['destructive_a']))
        self.assertFalse(np.any((a['tight_a']>=0)&a['destructive_a']))
        self.assertTrue(a['destructive_a'].any())
    def test_negative_arrays_unchanged(self):
        z,p,r,w,l=fixture(negative=True);a,info=run(z,None,r)
        for s in 'ab':
            for mode in ('full','tight'):np.testing.assert_array_equal(a[mode+'_'+s],z['target_'+s])
    def test_unchanged_side_missing_primary_field_not_destroyed(self):
        z,p,r,w,l=fixture(erosion=2,recipe='local_abrupt')
        pts=l['a_source_points'];p['primary_a_points']=pts
        p['primary_a_major']=((pts[:,0]>90)&(pts[:,0]<130)).astype(float)
        a,info=run(z,p,r,w,l)
        self.assertGreater((a['full_a']>=0).sum(),5)
        self.assertFalse(a['destructive_b'].all())
    def test_missing_ancestry_is_error_not_rematch(self):
        z,p,r,w,l=fixture()
        with self.assertRaises(EvidenceError):run(z,p,r,None,l)
    def test_bad_pose_is_error(self):
        z,p,r,w,l=fixture();z['translation_a_to_b_rc']=np.ones(2)*10
        with self.assertRaises(EvidenceError):run(z,p,r,w,l)
    def test_bad_latent_gap_is_error(self):
        z,p,r,w,l=fixture();l['a_gap']+=20
        with self.assertRaises(EvidenceError):run(z,p,r,w,l)
    def test_pixel_mismatch_is_error(self):
        z,p,r,w,l=fixture();p['packed_final_a'][100,10]^=1
        with self.assertRaises(EvidenceError):run(z,p,r,w,l)
    def test_arc_guard_wrap(self):
        b=np.zeros(100,bool);b[0]=True
        x=arc_expand(b,np.arange(100),100,8)
        self.assertTrue(x[92:].all());self.assertTrue(x[:9].all());self.assertFalse(x[91])
    def test_circular_runs(self):
        rr=runs(np.array([1,1,0,0,1],bool))
        np.testing.assert_array_equal(rr[0],[4,0,1])
    def test_tight_no_new_matches(self):
        z,p,r,w,l=fixture(erosion=3);a,info=run(z,p,r,w,l)
        i=np.flatnonzero(a['tight_a']>=0)
        np.testing.assert_array_equal(a['tight_a'][i],z['target_a'][i])
    def test_reciprocal_and_bound(self):
        z,p,r,w,l=fixture(erosion=3);a,info=run(z,p,r,w,l)
        for mode in ('full','tight'):
            i=np.flatnonzero(a[mode+'_a']>=0);j=a[mode+'_a'][i]
            np.testing.assert_array_equal(a[mode+'_b'][j],i)
            self.assertEqual(len(j),len(np.unique(j)))
    def test_strict_integer_translation(self):
        m=np.zeros((10,10),bool);m[2,3]=True
        o=parent_to_canvas(m,np.array([2,-1]),(10,10));self.assertTrue(o[4,2]);self.assertEqual(o.sum(),1)

if __name__=='__main__': unittest.main()
