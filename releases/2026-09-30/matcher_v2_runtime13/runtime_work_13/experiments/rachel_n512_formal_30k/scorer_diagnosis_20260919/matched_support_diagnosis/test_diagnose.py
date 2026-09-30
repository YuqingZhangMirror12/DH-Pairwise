"""Small numerical support/target tests: no network, dataset, or remote access."""
from copy import deepcopy
import math
import unittest

import numpy as np

from . import diagnose as d


class EvidenceTests(unittest.TestCase):
    def test_variable_contours_align_standard_padding_without_counting_ignored_padding(self):
        cached,_,_,gt,gt_valid,report=self.fixture()
        archive={}
        for side,n in (('a',483),('b',237)):
            points=(np.arange(n*2,dtype=np.float32).reshape(n,2)%37)
            target=np.full(n,-1,np.int64);target[0]=0;target[1]=-2
            archive.update({'points_rc_'+side:points,'contour_valid_'+side:np.ones(n,bool),
                            'target_'+side:target})
            cached['points_'+side]=np.zeros((512,2),np.float32)
            cached['points_'+side][:n]=points
            cached['valid_'+side]=np.arange(512)<n
        ta,tb=d.align_targets(archive,cached)
        np.testing.assert_array_equal(ta[:483],archive['target_a'])
        np.testing.assert_array_equal(tb[:237],archive['target_b'])
        self.assertTrue((ta[483:]==-2).all())
        self.assertTrue((tb[237:]==-2).all())
        cached['candidate_valid'][:]=False;cached['candidate_valid'][0]=True
        cached['candidate_inliers'][:]=False;cached['candidate_inliers'][0]=True
        cached['candidate_indices'][0]=[0,0]
        metrics=d.row_evidence(cached,ta,tb,gt,gt_valid,report)
        self.assertEqual(metrics['final_ignored_tokens'],2)  # Only valid damaged points, not collate padding.
        archive['points_rc_a'][2,0]+=.25
        with self.assertRaisesRegex(ValueError,'coordinates do not match'):
            d.align_targets(archive,cached)

    def fixture(self):
        points=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.],[4.,0.],[999.,999.]])
        valid=np.array([1,1,1,1,1,0],bool)
        row=dict(candidate_indices=np.array([[0,1],[1,0],[2,2],[3,3],[4,1],[-1,-1]]),
            candidate_valid=np.array([1,1,1,1,1,0],bool),
            candidate_inliers=np.array([1,1,1,1,0,0],bool),
            candidate_weights=np.array([1.,3.,2.,4.,5.,0.]),
            points_a=points,points_b=points+np.array([10.,20.]),valid_a=valid,valid_b=valid.copy(),
            layout_valid=True,translation_a_to_b_rc=np.array([10.,20.]),
            label=1,training_valid=True,decision_valid=True)
        # Includes a true correspondence TO index0, a dustbin(-1), and ignored(-2).
        targets=np.array([1,0,-1,-2,-1,-2])
        report=dict(pose_supervision_enabled=False,changed_pair=True,fallback_reason=None)
        return row,targets,targets.copy(),np.array([10.,20.]),True,report

    def test_reciprocal_zero_index_dustbin_and_ignored_precision(self):
        args=self.fixture();r=d.row_evidence(*args)
        self.assertEqual(r['candidate_count'],5)
        self.assertEqual(r['inlier_count'],4)
        self.assertEqual(r['final_target_matches'],2)
        self.assertEqual(r['correct_candidate_edges'],2)
        self.assertEqual(r['correct_inlier_edges'],2)
        self.assertEqual(r['supervised_inlier_edges'],3)
        self.assertAlmostEqual(r['exact_inlier_precision'],2/3)
        self.assertEqual(r['exact_target_recall_candidates'],1.)
        self.assertEqual(r['exact_target_recall_inliers'],1.)
        self.assertEqual(r['inlier_q_mass'],10.)
        self.assertEqual(r['inlier_q_mean'],2.5)
        self.assertAlmostEqual(r['inlier_weighted_rms_px'],math.sqrt(.4))
        self.assertEqual(r['unique_endpoints_min'],4)
        self.assertEqual(r['endpoint_coverage_min'],.8)
        self.assertEqual(r['supported_chord_length_min_px'],3.)
        self.assertEqual(r['longest_run_min_tokens'],4)
        self.assertTrue(r['raw_layout20_success'])

    def test_candidate_recall_and_final_recall_use_distinct_predicted_sets(self):
        args=list(self.fixture())
        args[0]['candidate_inliers']=np.array([1,0,1,1,0,0],bool)
        r=d.row_evidence(*args)
        self.assertEqual(r['correct_candidate_edges'],2)
        self.assertEqual(r['correct_inlier_edges'],1)
        self.assertEqual(r['exact_target_recall_candidates'],1.)
        self.assertEqual(r['exact_target_recall_inliers'],.5)
        self.assertEqual(r['exact_inlier_precision'],.5)

    def test_all_ignored_final_edges_have_no_precision_denominator(self):
        args=list(self.fixture())
        args[0]['candidate_inliers']=np.array([0,0,0,1,0,0],bool)
        r=d.row_evidence(*args)
        self.assertEqual(r['supervised_inlier_edges'],0)
        self.assertIsNone(r['exact_inlier_precision'])
        self.assertEqual(r['exact_target_recall_inliers'],0.)

    def test_invalid_layout_retained_as_failure_not_nonfinite_error(self):
        args=list(self.fixture())
        args[0]['layout_valid']=False
        args[0]['translation_a_to_b_rc']=np.array([np.nan,np.nan])
        r=d.row_evidence(*args)
        self.assertIsNone(r['raw_layout_error_px'])
        self.assertIsNone(r['inlier_weighted_rms_px'])
        self.assertFalse(r['raw_layout20_success'])
        args[0]['layout_valid']=True
        with self.assertRaisesRegex(ValueError,'nonfinite translation'):
            d.row_evidence(*args)

    def test_target_reciprocity_and_padding_guards(self):
        args=list(self.fixture());args[2][0]=2
        with self.assertRaisesRegex(ValueError,'not reciprocal'):
            d.row_evidence(*args)
        args=list(self.fixture());args[0]['candidate_inliers'][-1]=True
        with self.assertRaisesRegex(ValueError,'padding'):
            d.row_evidence(*args)
        args=list(self.fixture());args[0]['candidate_indices'][0]=[5,0]
        with self.assertRaisesRegex(ValueError,'invalid contour'):
            d.row_evidence(*args)

    def test_negative_targets_and_empty_inlier_support(self):
        args=list(self.fixture());args[0]['label']=0
        with self.assertRaisesRegex(ValueError,'negative pair'):
            d.row_evidence(*args)
        args[1][:]=-1;args[2][:]=-1;args[4]=False
        args[0]['candidate_inliers'][:]=False
        r=d.row_evidence(*args)
        self.assertEqual(r['final_target_matches'],0)
        self.assertEqual(r['inlier_count'],0)
        self.assertEqual(r['inlier_q_mass'],0.)
        self.assertEqual(r['endpoint_coverage_min'],0.)
        self.assertIsNone(r['exact_target_recall_inliers'])
        self.assertIsNone(r['exact_inlier_precision'])
        self.assertFalse(r['raw_layout20_success'])

    def test_layout20_all_positive_denominator_includes_invalid_not_classifier_gated(self):
        base=self.fixture()
        good=d.row_evidence(*base)
        at_boundary=list(deepcopy(base))
        at_boundary[0]['translation_a_to_b_rc']=np.array([30.,20.])
        at_boundary[0]['training_valid']=False
        at_boundary[0]['decision_valid']=False
        boundary=d.row_evidence(*at_boundary)
        self.assertTrue(boundary['raw_layout20_success'])  # No classifier/transport gate.
        bad=list(deepcopy(base));bad[0]['translation_a_to_b_rc']=np.array([31.,20.])
        invalid=list(deepcopy(base));invalid[0]['layout_valid']=False
        missing=list(deepcopy(base));missing[4]=False
        with self.assertRaisesRegex(ValueError,'eligibility contract'):
            d.row_evidence(*missing)
        negative=list(deepcopy(base));negative[0]['label']=0
        negative[1][:]=-1;negative[2][:]=-1;negative[4]=False
        result=d.summarize([good,boundary,d.row_evidence(*bad),d.row_evidence(*invalid),
                            d.row_evidence(*negative)])
        self.assertEqual(result['count'],5)
        self.assertEqual(result['raw_layout20'],dict(success=2,denominator=4,missing_gt=0,rate=.5))


class CyclicSupportTests(unittest.TestCase):
    def test_wraparound_run_and_chord_support(self):
        p=np.array([[0.,0.],[0.,1.],[1.,1.],[1.,0.]])
        valid=np.ones(4,bool)
        self.assertEqual(d.cyclic_support(p,valid,[1,0,1,1]),
                         dict(count=3,coverage=.75,chord_length_px=2.,longest_run_tokens=3))
        self.assertEqual(d.cyclic_support(p,valid,[1,1,1,1]),
                         dict(count=4,coverage=1.,chord_length_px=4.,longest_run_tokens=4))
        self.assertEqual(d.cyclic_support(p,valid,[0,0,0,0]),
                         dict(count=0,coverage=0.,chord_length_px=0.,longest_run_tokens=0))

    def test_padding_not_counted_as_support_or_break_in_valid_contour(self):
        p=np.array([[0.,0.],[999.,999.],[0.,1.],[1.,1.],[1.,0.],[999.,999.]])
        valid=np.array([1,0,1,1,1,0],bool)
        self.assertEqual(d.cyclic_support(p,valid,[1,1,0,1,1,1]),
                         dict(count=3,coverage=.75,chord_length_px=2.,longest_run_tokens=3))
        self.assertEqual(d.cyclic_support(p,np.zeros(6,bool),np.ones(6,bool)),
                         dict(count=0,coverage=0.,chord_length_px=0.,longest_run_tokens=0))


class AucTests(unittest.TestCase):
    def test_perfect_reverse_and_ties(self):
        labels=[0,1,0,1]
        self.assertEqual(d.auc(labels,[0.,1.,0.,1.]),1.)
        self.assertEqual(d.auc(labels,[1.,0.,1.,0.]),0.)
        self.assertEqual(d.auc(labels,[.5,.5,.5,.5]),.5)
        self.assertEqual(d.auc(labels,[0.,1.,1.,2.]),.875)
        self.assertIsNone(d.auc([1,1],[0.,1.]))
        self.assertIsNone(d.auc([0,0],[0.,1.]))

    def test_nonfinite_or_nonbinary_values_rejected(self):
        for labels,scores in (([],[]),([0,1],[0.,np.nan]),([0,2],[0.,1.])):
            with self.subTest(labels=labels,scores=scores),self.assertRaises(ValueError):
                d.auc(labels,scores)


if __name__=='__main__':
    unittest.main()
