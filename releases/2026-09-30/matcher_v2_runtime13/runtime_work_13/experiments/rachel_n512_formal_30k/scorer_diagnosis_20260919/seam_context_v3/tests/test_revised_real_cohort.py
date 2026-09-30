import copy
import unittest
from ..revised_real_cohort import revise


def fixture():
    model = dict(score=.6, decision_valid=True, threshold_sim=.7, threshold_cv=.3,
                 accepted_cv=True, layout20=True)
    cases = [dict(dataset=ds, pair_id=str(i), label=i != 2,
                  case_name='main/Ground Truth Simple/' + str(i) + ' · 1 + 2',
                  fold=i, models={'v3_B22': copy.deepcopy(model)})
             for ds in ('敦煌', 'Turufan') for i in range(3)]
    for c in cases:
        if c['dataset'] == 'Turufan':
            c['models']['v3_B22']['layout20'] = None
    return ({'queries': {'cases': {'rows': cases}}},
            {'schema': 'user-confirmed-gt-exclusions-v1', 'records': [dict(
                dataset='敦煌', pair_id='0', case_folder='main/Ground Truth Simple/0')]})


class RevisedCohortTests(unittest.TestCase):
    def test_exclusion_changes_denominator_not_decisions_or_source(self):
        snapshot, exclusions = fixture()
        before = copy.deepcopy(snapshot)
        result = revise(snapshot, exclusions)
        self.assertEqual(snapshot, before)
        real_sim, real_cv, ood_sim, ood_cv = result['rows']
        self.assertEqual((real_cv['n'], real_cv['positive'], real_cv['tp'], real_cv['fp']), (2, 1, 1, 1))
        self.assertEqual(real_sim['tp'], 0)
        self.assertEqual(ood_cv['n'], 3)
        self.assertNotIn('layout_correct_total', ood_cv)
        self.assertFalse(result['protocol']['thresholds_refit'])

    def test_wrong_identity_or_negative_exclusion_rejected(self):
        snapshot, exclusions = fixture()
        exclusions['records'][0]['case_folder'] = 'main/Ground Truth Simple/01'
        with self.assertRaisesRegex(ValueError, 'exact positive'):
            revise(snapshot, exclusions)
        exclusions['records'][0].update(pair_id='2', case_folder='main/Ground Truth Simple/2')
        with self.assertRaisesRegex(ValueError, 'exact positive'):
            revise(snapshot, exclusions)

    def test_stale_decision_rejected(self):
        snapshot, exclusions = fixture()
        snapshot['queries']['cases']['rows'][1]['models']['v3_B22']['accepted_cv'] = False
        with self.assertRaisesRegex(ValueError, 'decision mismatch'):
            revise(snapshot, exclusions)

    def test_missing_models_not_silently_denominator_dropped(self):
        snapshot, exclusions = fixture()
        snapshot['queries']['cases']['rows'][2]['models'] = {}
        with self.assertRaisesRegex(ValueError, 'partial model'):
            revise(snapshot, exclusions)


if __name__ == '__main__':
    unittest.main()
