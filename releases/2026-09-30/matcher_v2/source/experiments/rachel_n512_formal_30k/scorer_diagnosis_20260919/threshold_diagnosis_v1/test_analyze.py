import unittest
from .analyze import decompose, count_decisions, sigmoid


class ReadoutTests(unittest.TestCase):
    def test_ceiling_and_interventions(self):
        w=dict(bias=-1.4,positive=2.5,conflict=1.5,overlap=4.5,length_scale_px=32.)
        c=dict(positive_evidence_px=1.,conflict_evidence_px=.5,observed_mass_length_px=2.,
               overlap=dict(fraction_min_area=.02),logit=0.)
        d=decompose(c,w)
        self.assertGreaterEqual(d['perfect_local_score'],d['reconstructed_score'])
        self.assertGreaterEqual(d['no_overlap_score'],d['reconstructed_score'])
        self.assertGreaterEqual(d['no_conflict_score'],d['reconstructed_score'])
        self.assertAlmostEqual(sigmoid(d['bias']+d['positive_logit']-d['conflict_penalty']-d['overlap_penalty']),d['reconstructed_score'])

    def test_unknown_gt_is_null_and_missing_candidate_negative(self):
        rows=[dict(candidates=[],label=True,numeric_valid=True),
              dict(candidates=[],label=False,numeric_valid=True)]
        c=count_decisions(rows,'score',.21,False)
        self.assertEqual((c['tp'],c['fp'],c['fn'],c['tn']),(0,0,1,1))
        self.assertIsNone(c['layout_correct'])

    def test_reranking_is_not_fixed_winner(self):
        rows=[dict(label=True,numeric_valid=True,candidates=[dict(score=.2,alternative=.3,gt_error_px=30),
              dict(score=.19,alternative=.4,gt_error_px=1)])]
        self.assertEqual(count_decisions(rows,'score',.21,True)['layout_correct'],0)
        self.assertEqual(count_decisions(rows,'alternative',.21,True)['layout_correct'],1)

    def test_overlap_missing_rejected(self):
        with self.assertRaises(AssertionError):
            decompose(dict(positive_evidence_px=1,conflict_evidence_px=0,observed_mass_length_px=1,
                           overlap=dict(fraction_min_area=None)),{})


if __name__=='__main__':unittest.main()
