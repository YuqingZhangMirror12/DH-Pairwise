import math
import unittest
from .full_review import candidate_scores, sigmoid, tally


class FullReviewTests(unittest.TestCase):
    def test_light_no_conflict_is_exact_noop(self):
        scores, error = candidate_scores({'local_classification_present':False,'score':.234},None)
        self.assertEqual(scores,dict(actual=.234,no_conflict=.234));self.assertEqual(error,0)
        self.assertNotIn('all_support',scores)

    def test_no_conflict_does_not_transfer_mass_to_support(self):
        w=dict(bias=-2,positive=2,conflict=1,overlap=4,length_scale_px=32)
        logit=-2+2*math.log1p(10/32)-math.log1p(20/32)-4*.1
        c=dict(positive_evidence_px=10,conflict_evidence_px=20,observed_mass_length_px=50,
               overlap={'fraction_min_area':.1},logit=logit,score=sigmoid(logit))
        scores,_=candidate_scores(c,w)
        self.assertLess(scores['actual'],scores['no_conflict'])
        self.assertLess(scores['no_conflict'],scores['all_support'])

    def test_readout_corruption_is_rejected(self):
        c=dict(positive_evidence_px=0,conflict_evidence_px=0,observed_mass_length_px=0,
               overlap={'fraction_min_area':0},logit=3,score=.1)
        with self.assertRaises(AssertionError):candidate_scores(c,dict(bias=-2,positive=2,conflict=1,overlap=4,length_scale_px=32))

    def test_reranking_and_fp_cost_are_both_counted(self):
        c=lambda i,a,n,err:dict(cluster_id=i,selected=i==0,scores=dict(actual=a,no_conflict=n),gt_error_px=err)
        rows=[dict(label=True,numeric_valid=True,candidates=[c(0,.4,.45,40),c(1,.2,.8,5)]),
              dict(label=False,numeric_valid=True,candidates=[c(0,.1,.7,None)])]
        actual=tally(rows,'actual',.3,True);cf=tally(rows,'no_conflict',.3,True)
        self.assertEqual((actual['tp'],actual['fp'],actual['layout20_count']),(1,0,0))
        self.assertEqual((cf['tp'],cf['fp'],cf['layout20_count'],cf['changed_winners']),(1,1,1,1))
        self.assertAlmostEqual(cf['joint_f1'],2/3)

    def test_turufan_has_no_layout_claim(self):
        r=dict(label=True,numeric_valid=True,candidates=[dict(cluster_id=0,selected=True,scores={'actual':.1},gt_error_px=None)])
        s=tally([r],'actual',.2,False)
        self.assertIsNone(s['layout20_count']);self.assertIsNone(s['joint_f1']);self.assertEqual(s['fn'],1)


if __name__=='__main__':unittest.main()
