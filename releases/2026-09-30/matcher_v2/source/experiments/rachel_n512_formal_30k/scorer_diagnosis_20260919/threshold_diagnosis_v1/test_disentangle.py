import unittest

from .disentangle import decisions, metrics, matched_threshold, crossing
from .distributions import stats


def pair(name, positive, score, error=None, valid=True):
    return dict(pair_id=name, label=positive, numeric_valid=valid,
                candidates=[dict(cluster_id=0, score=score, gt_error_px=error)])


class DiagnosticTests(unittest.TestCase):
    def test_empty_distribution_is_missing_not_zero(self):
        s=stats([])
        self.assertEqual(s['n'],0)
        self.assertIsNone(s['median'])

    def test_quantiles_keep_all_tails(self):
        s=stats([1,2,3,4])
        self.assertEqual((s['minimum'],s['p25'],s['median'],s['p75'],s['maximum']),(1.,1.75,2.5,3.25,4.))

    def test_cut_uses_negative_scores_only(self):
        rows=[pair('n1',False,.4),pair('n2',False,.3),pair('p',True,.2,0)]
        cut=matched_threshold(rows,'score',1)
        rows[-1]['candidates'][0]['score']=.99
        self.assertEqual(cut,matched_threshold(rows,'score',1))
        self.assertGreater(cut,.3)
        self.assertEqual(metrics(decisions(rows,'score',cut,True),cut,True)['fp'],1)

    def test_boundary_ties_not_broken_by_id(self):
        rows=[pair('n1',False,.4),pair('n2',False,.4)]
        t=matched_threshold(rows,'score',1)
        self.assertEqual(metrics(decisions(rows,'score',t,False),t,False)['fp'],0)

    def test_invalid_sample_cannot_be_accepted(self):
        rows=[pair('n',False,.99,valid=False),pair('p',True,.8,None)]
        t=matched_threshold(rows,'score',0)
        m=metrics(decisions(rows,'score',t,False),t,False)
        self.assertEqual(m['fp'],0)
        self.assertIsNone(m['layout_correct'])
        self.assertIsNone(m['layout_correct_accepted'])

    def test_threshold_does_not_change_winner(self):
        rows=[pair('p',True,.2,2)]
        a=metrics(decisions(rows,'score',.21,True),.21,True)
        b=metrics(decisions(rows,'score',.19,True),.19,True)
        self.assertEqual(a['layout_correct'],b['layout_correct'])
        self.assertEqual((a['layout_correct_accepted'],b['layout_correct_accepted']),(0,1))

    def test_reranking_and_joint_gains_differ_from_pair_gains(self):
        rows=[pair('p',True,.3,40)]
        rows[0]['candidates'].append(dict(cluster_id=1,score=.2,gt_error_px=2))
        before=decisions(rows,'score',.1,True)
        rows[0]['candidates'][1]['score']=.4
        after=decisions(rows,'score',.1,True)
        self.assertEqual(crossing(before,after,joint=False)['net'],0)
        self.assertEqual(crossing(before,after,joint=True)['net'],1)


if __name__=='__main__':unittest.main()
