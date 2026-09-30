import copy
import unittest

from score_native import native_development_rows


def fixture():
    candidate=dict(index=0,q_sum=.25,q_arc_mass_px=3.)
    row=dict(pair_id='positive',schema='curriculum-matcher-pair/1',scorer_used=False,
        q_modified=False,gt_used_in_proposal=False,label=True,numeric_valid=True,gt_known=True,
        retained_correct_coverage=True,retained_count=1,candidates=[candidate],q_sum_winner=0,
        q_arc_winner=0,q_sum_winner_layout20=True,q_arc_winner_layout20=True)
    strata=[dict(pair_id='positive',label=True,role='real_select',fold=2)]
    return [row],strata


class FrozenNativeRowsTests(unittest.TestCase):
    def test_both_original_winners_and_mass_scales_preserved(self):
        rows,strata=fixture(); before=copy.deepcopy(rows)
        self.assertEqual(native_development_rows(rows,strata,'q_sum')[0]['score'],.25)
        self.assertEqual(native_development_rows(rows,strata,'q_arc')[0]['score'],3.)
        self.assertEqual(rows,before)

    def test_unselected_test_row_never_enters_new_metrics(self):
        rows,strata=fixture()
        rows.append(dict(pair_id='heldout_test',unused='not an analysis row'))
        self.assertEqual(len(native_development_rows(rows,strata,'q_sum')),1)
        strata[0]['fold']=0
        with self.assertRaisesRegex(ValueError,'TEST'):
            native_development_rows(rows,strata,'q_sum')

    def test_missing_candidates_count_as_incorrect_not_missing_ground_truth(self):
        rows,strata=fixture(); rows[0].update(q_sum_winner=None,q_sum_winner_layout20=None,
                                            retained_correct_coverage=False,retained_count=0,candidates=[])
        actual=native_development_rows(rows,strata,'q_sum')[0]
        self.assertFalse(actual['has_candidate']); self.assertFalse(actual['layout20'])
        self.assertEqual(actual['score'],0.)

    def test_prebudget_candidate_cannot_win(self):
        rows,strata=fixture(); rows[0]['retained_count']=0
        with self.assertRaisesRegex(ValueError,'outside retained'):
            native_development_rows(rows,strata,'q_sum')

    def test_missing_identity_or_ground_truth_rejected(self):
        rows,strata=fixture()
        with self.assertRaisesRegex(ValueError,'membership'):
            native_development_rows([],strata,'q_sum')
        rows[0]['gt_known']=False
        with self.assertRaisesRegex(ValueError,'known positive'):
            native_development_rows(rows,strata,'q_sum')


if __name__=='__main__':unittest.main()
