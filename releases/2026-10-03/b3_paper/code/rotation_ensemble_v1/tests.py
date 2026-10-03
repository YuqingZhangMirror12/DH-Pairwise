import json
import unittest
import numpy as np
from .core import INPUTS,rotate_inputs,rotate_vectors,combine,annotate,deduplicate
from .metrics import auc,fpr_threshold,summarize,fit_threshold


def candidate(index=0,angle=0,pose=(0.,0.),q=1.,score=.7):
    return dict(index=index,angle=angle,translation=list(pose),q_sum=q,
                heads={'stats_sim':dict(score=score,logit=float(np.log(score/(1-score))))})


def views(*cs):
    result={}
    for c in cs:result.setdefault(c['angle'],dict(numeric_valid=True,candidates=[]))['candidates'].append(c)
    return result


class Correctness(unittest.TestCase):
    def test_exact_pixels_and_order(self):
        b={}
        for side in 'ab':
            m=np.zeros((1,1,800,800),np.float32);m[0,0,100,203]=1
            p=np.zeros((1,512,2),np.float32);p[0,0]=[100,203];p[0,1]=[500,600]
            v=np.zeros((1,512),bool);v[0,:2]=True
            b['mask_'+side]=m;b['points_rc_'+side]=p;b['contour_valid_'+side]=v
        for a,expected in [(0,[100,203]),(90,[596,100]),(180,[699,596]),(270,[203,699])]:
            r=rotate_inputs(b,a);self.assertEqual(r['points_rc_a'][0,0].tolist(),expected)
            self.assertEqual(r['mask_a'][0,0,expected[0],expected[1]],1)
            self.assertTrue(np.array_equal(r['contour_valid_a'],b['contour_valid_a']))
            self.assertFalse(r['points_rc_a'][0,2:].any())
        r=b
        for _ in range(4):r=rotate_inputs(r,90)
        for k in INPUTS:self.assertTrue(np.array_equal(r[k],b[k]))

    def test_targets_forbidden(self):
        with self.assertRaises(ValueError):rotate_inputs({'label':True},0)
        v=views(candidate());v[0]['label']=True
        with self.assertRaises(ValueError):combine(v)

    def test_gt_frame_distance(self):
        rng=np.random.default_rng(19);pose=rng.normal(size=(80,2))*200;gt=rng.normal(size=(80,2))*200
        for a in (0,90,180,270):
            r=rotate_vectors(pose,a);self.assertTrue(np.array_equal(rotate_vectors(r,a,True),pose))
            np.testing.assert_array_equal(np.linalg.norm(r-rotate_vectors(gt,a),axis=1)<=20,np.linalg.norm(pose-gt,axis=1)<=20)

    def test_identity_all_methods(self):
        v=views(candidate(0,q=8,score=.3),candidate(1,pose=(100,40),q=3,score=.9))
        for h,idx in [('q',0),('stats_sim',1)]:
            for m in ('A','B','C'):
                r=combine(v,method=m,head=h);self.assertEqual(r['winner']['index'],idx);json.dumps(r)

    def test_no_pose_averaging(self):
        v=views(candidate(0,0,(0,0),10,.6),candidate(0,90,(200,400),5,.8))
        a=combine(v,(0,90),'A','stats_sim');b=combine(v,(0,90),'B','stats_sim');c=combine(v,(0,90),'C','stats_sim')
        self.assertEqual(a['translation'],[0,0]);self.assertEqual(b['translation'],[200,400]);self.assertEqual(c['translation'],b['translation']);self.assertAlmostEqual(c['score'],.7)
        json.dumps([a,b,c])

    def test_dedup_not_single_link(self):
        cs=[candidate(i,0,(15*i,0),score=.9-i*.1) for i in range(3)]
        g=deduplicate(cs,'stats_sim');self.assertEqual(list(map(len,g)),[2,1]);self.assertIs(g[0][0],cs[0])

    def test_no_candidate_and_unknown(self):
        p=combine({0:dict(numeric_valid=True,candidates=[])})
        a=annotate(p,True,None);self.assertIsNone(a['layout20']);self.assertFalse(p['has_candidate'])
        s=summarize([a,annotate(p,False)],0);self.assertIsNone(s['joint_f1']);self.assertEqual(s['pair_tp'],0)

    def test_score_finite(self):
        c=candidate(q=float('nan'))
        with self.assertRaises(ValueError):combine(views(c),head='stats_sim')

    def test_fast_report_matches_wrapper(self):
        from .report import settings_rows
        rng=np.random.default_rng(99);cs=[]
        for a in (0,90,180,270):
            for i in range(4):cs.append(candidate(i,a,rng.normal(size=2)*40,float(rng.random()*6),float(rng.uniform(.1,.9))))
        v=views(*cs);t=dict(label=True,target=[0,0],fold=2,seam_type='J')
        raw=[dict(pair_id='fixture',views=v)];fast=settings_rows((raw,{'fixture':t}),(0,90,180,270),'stats_sim')
        for m in ('A','B','C'):
            expected=annotate(combine(v,(0,90,180,270),m,'stats_sim'),True,[0,0])
            for k in ('translation','score','winner','layout20','candidate_coverage','agreement_views'):
                self.assertEqual(fast[m][0][k],expected[k])

    def test_auc_ties_and_fpr_boundary(self):
        rows=[]
        for i in range(100):
            p=combine(views(candidate(score=.9 if i<3 else .2)),head='stats_sim')
            rows.append(annotate(p,False))
        t=fpr_threshold(rows,.02);self.assertGreater(t,.9);self.assertEqual(summarize(rows,t)['pair_fp'],0)
        pos=annotate(combine(views(candidate(score=.9)),head='stats_sim'),True,[0,0])
        self.assertAlmostEqual(auc(rows+[pos]),.985)
        self.assertTrue(.2<=fit_threshold(rows+[pos],'stats_sim')<=.8)


if __name__=='__main__':unittest.main()
