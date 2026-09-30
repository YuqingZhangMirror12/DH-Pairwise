"""Synthetic structural fixtures only; never used as reported model results."""
import copy
import unittest
from ..extend_independent_report import ARMS, SPLITS, APP_ID, extend


def fixture():
    cases = []
    for split, dataset in SPLITS.items():
        gt = [1., 2.] if split == 'dunhuang_cv' else None
        cases.append(dict(pair_id=split, dataset=dataset, label=True, fragment_a='a', fragment_b='b',
            case_name='synthetic unit fixture', fold=0, gt=gt,
            models={k: dict(translation=[1., 2.], score=.7) for k in ('v3_B22', 'S7_M12_matched_C16', 'S7_H')}))
    snapshot = dict(id=APP_ID, title='Preserved title', presentation={'hiddenBlocks': ['test']},
        queries=dict(cases=dict(rows=cases), metrics=dict(rows=[dict(model='old', value='unchanged')]),
            fragments=dict(rows=[dict(fragment_id='a', png='fixture')]), heatmaps=dict(rows=['original'])))
    comparison = dict(independent_feature_extension=dict(same_source_folds=True,
        checkpoint_selection='simulation CAL/SELECT only', score_rescaling=False), splits={})
    oof, bundles = {}, {arm: {} for arm in ARMS}
    for split, dataset in SPLITS.items():
        comparison['splits'][split] = dict(models={}, independent_feature_proposals={'max_delta': 0},
            independent_feature_paired={})
        oof[split] = {}
        for arm in ARMS:
            metric = dict(accuracy=1., precision=1., recall=1., f1=1., auroc=None,
                tp=1, fp=0, fn=0, tn=0, n=1, positive=1, negative=0, threshold=.5,
                layout_correct_total=1 if split=='dunhuang_cv' else None,
                layout_correct_accepted=1 if split=='dunhuang_cv' else None)
            comparison['splits'][split]['models'][arm] = dict(checkpoint_sha256=arm, selected_epoch=2,
                sim_frozen=metric, cv=dict(bounded_max_f1=dict(pooled_out_of_fold=metric,
                    folds=[dict(threshold=.5)])))
            oof[split][arm] = dict(bounded_max_f1=[dict(pair_id=split, score=.7, threshold=.5, accepted=True)])
            bundles[arm][split] = dict(protocol=dict(status='complete', arm=arm, split=split,
                frozen_matcher_verified=True, real_used_to_select_checkpoint=False,
                threshold_fitting=False, gt_used_to_generate_candidates=False,
                checkpoint_sha256=arm, selected_epoch=2, threshold=.5, sample_count=1),
                status=dict(status='complete', count=1), rows=[dict(pair_id=split, label=True,
                    score=.7, translation=[1., 2.], numeric_valid=True, has_candidate=True,
                    target_translation_rc=[1., 2.] if split=='dunhuang_cv' else None,
                    error_px=0. if split=='dunhuang_cv' else None, layout20=True,
                    gt_known=split=='dunhuang_cv', winner_index=0,
                    candidates=[dict(edge_count=12, residual_median_px=.1)])])
    return snapshot, comparison, oof, bundles


class IndependentReportTests(unittest.TestCase):
    def test_preserves_manual_review_cases_and_assets_exactly(self):
        args = fixture(); before = copy.deepcopy(args)
        result = extend(*args)
        self.assertEqual(args, before)
        for key in ('cases', 'fragments', 'heatmaps'):
            self.assertEqual(result['queries'][key], args[0]['queries'][key])
        self.assertEqual(result['presentation'], args[0]['presentation'])
        self.assertEqual(result['title'], args[0]['title'])
        self.assertEqual(result['queries']['metrics']['rows'][0], args[0]['queries']['metrics']['rows'][0])
        self.assertEqual(len(result['queries']['metrics']['rows']), 9)
        self.assertEqual(len(result['queries']['independent_cases']['rows']), 2)
        self.assertEqual(set(result['queries']['independent_cases']['rows'][0]['models']),
            {'v3_B22', 'S7_M12_matched_C16', 'S7_H', *ARMS})

    def test_no_layout_ground_truth_is_not_converted_to_failure_or_success(self):
        result = extend(*fixture())
        tur = next(r for r in result['queries']['independent_cases']['rows'] if r['dataset']=='Turufan')
        self.assertIsNone(tur['models']['independent_features']['layout20'])
        self.assertIsNone(tur['models']['independent_features']['layout_error_px'])

    def test_incomplete_or_wrong_model_is_rejected(self):
        args = fixture(); p = args[3]['frozen_features']['dunhuang_cv']['protocol']
        p['status'] = 'running'
        with self.assertRaisesRegex(ValueError, 'incomplete'): extend(*args)
        p['status'] = 'complete'; p['checkpoint_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'binding'): extend(*args)

    def test_missing_and_duplicate_cases_are_rejected(self):
        args = fixture(); rows = args[3]['frozen_features']['dunhuang_cv']['rows']
        row = rows.pop()
        with self.assertRaisesRegex(ValueError, 'population'): extend(*args)
        rows.extend([row, row])
        with self.assertRaisesRegex(ValueError, 'duplicate'): extend(*args)

    def test_changed_score_or_invalid_acceptance_is_rejected(self):
        args = fixture(); f = args[2]['dunhuang_cv']['frozen_features']['bounded_max_f1'][0]
        f['score'] = .8
        with self.assertRaisesRegex(ValueError, 'changed during join'): extend(*args)
        f['score'] = .7; args[3]['frozen_features']['dunhuang_cv']['rows'][0]['numeric_valid'] = False
        with self.assertRaisesRegex(ValueError, 'changed during join'): extend(*args)

    def test_repeated_extension_does_not_duplicate_new_metrics(self):
        args = fixture(); once = extend(*args)
        twice = extend(once, *args[1:])
        self.assertEqual(once, twice)


if __name__ == '__main__':
    unittest.main()
