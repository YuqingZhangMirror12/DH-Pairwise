"""Synthetic arithmetic/contracts tests; no models or performance estimates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import baselines as b
from .compare import combined_evidence


def pair(name, label, score=.8, good=True, valid=True, fold=0):
    return dict(pair_id=name, label=label, score=score, layout_good_20=good if label else None,
                decision_valid=valid, fold=fold)


def roles_fixture():
    rows = [pair(str(i), i % 2 == 0, fold=(1 if i < 2 else 2 if i < 6 else 0)) for i in range(9)]
    roles = {}
    for name, indices, folds in (('real_cal', range(2), [1]), ('real_select', range(2, 6), [2, 3, 4]),
                                 ('real_test', range(6, 8), [0])):
        roles[name] = dict(pair_ids=[str(i) for i in indices], folds=folds,
                           positive=sum(rows[i]['label'] for i in indices))
    plan = dict(datasets=dict(dunhuang_cv=dict(roles=roles, excluded_gt_pair_ids=['8'])))
    counts = dict(all_development_context=9, gt_corrected_800_development_context=8,
                  real_cal=2, real_select=4, real_test=2)
    return rows, plan, counts


def reference_fixture():
    return dict(schema='binary-report-complex-baselines/1', status='baseline_evidence_ready',
                real_plan_sha256=b.REAL_PLAN_SHA, threshold_fitted=False,
                five_fold_metrics_substituted=False, caveats=[], rows=[dict(
                    model='threshold_m12', selection_kind='sim', split='dunhuang_cv', population='real_test',
                    policy='primary', metrics=dict(pairs=161, positives=59, negatives=102))])


class BaselineTests(unittest.TestCase):
    def test_joint_wrong_pose_counts_as_fp_and_fn(self):
        rows = [pair('a', True), pair('b', True, good=False), pair('c', False), pair('d', False, .1)]
        r = b.summarize(rows, .5, True)
        self.assertEqual((r['tp'], r['fp'], r['fn']), (2, 1, 0))
        self.assertEqual((r['joint_tp'], r['joint_fp'], r['joint_fn']), (1, 2, 1))
        self.assertEqual(r['joint_f1'], .4)

    def test_invalid_high_score_not_accepted(self):
        r = b.summarize([pair('a', True, .99, valid=False), pair('b', False, .1)], .5, True)
        self.assertEqual(r['tp'], 0)
        self.assertEqual(r['fn'], 1)

    def test_exact_threshold_is_accepted(self):
        r = b.summarize([pair('a', True, .3), pair('b', False, .1)], .3, True)
        self.assertEqual(r['tp'], 1)

    def test_turufan_layout_remains_null(self):
        r = b.summarize([pair('a', True), pair('b', False, .1)], .5, False)
        for key in ('layout20', 'layout20_count', 'joint_tp', 'joint_f1', 'wrong_pose_accepted'):
            self.assertIsNone(r[key])

    def test_missing_positive_gt_not_zero_filled(self):
        with self.assertRaisesRegex(ValueError, 'missing positive GT'):
            b.summarize([pair('a', True, good=None), pair('b', False)], .5, True)

    def test_registered_roles_not_original_five_fold_calibration(self):
        rows, plan, counts = roles_fixture()
        with patch.dict(b.REAL_GROUP_COUNTS, {'dunhuang_cv': counts}):
            result = b.groups_for(rows, plan, 'dunhuang_cv')
        self.assertEqual([r['pair_id'] for r in result['real_test']], ['6', '7'])
        self.assertEqual([r['pair_id'] for r in result['real_cal']], ['0', '1'])
        self.assertEqual(len(result['all_development_context']), 9)

    def test_role_wrong_fold_fails(self):
        rows, plan, counts = roles_fixture(); rows[6]['fold'] = 4
        with patch.dict(b.REAL_GROUP_COUNTS, {'dunhuang_cv': counts}), self.assertRaisesRegex(ValueError, 'fold'):
            b.groups_for(rows, plan, 'dunhuang_cv')

    def test_role_label_count_fails(self):
        rows, plan, counts = roles_fixture(); rows[6]['label'] = False
        with patch.dict(b.REAL_GROUP_COUNTS, {'dunhuang_cv': counts}), self.assertRaisesRegex(ValueError, 'label'):
            b.groups_for(rows, plan, 'dunhuang_cv')

    def test_duplicate_pair_cannot_supply_missing_role(self):
        rows, plan, counts = roles_fixture(); rows[6] = copy.deepcopy(rows[7])
        with patch.dict(b.REAL_GROUP_COUNTS, {'dunhuang_cv': counts}), self.assertRaisesRegex(ValueError, 'population'):
            b.groups_for(rows, plan, 'dunhuang_cv')

    def test_unregistered_plan_rejected_before_data_import(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'plan.json'; p.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'plan changed'):
                b.build_baselines({'threshold_m12': temp}, p, p)

    def test_prior_binding_conflict_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'analysis.json'
            path.write_text(json.dumps(dict(summary=dict(schema='frozen-real-comparison/1',
                status='complete', all_source_hashes_unchanged=True, inputs={'same': 'a'},
                extensions=[dict(input_sha256={'same': 'b'})]))))
            with self.assertRaisesRegex(ValueError, 'conflict'):
                b.prior_bindings(path)

    def test_tampered_previously_audited_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = (Path(temp) / 'data.json').resolve(); p.write_text('{}')
            binding = {str(p): b.sha(p)}; used = {}
            b.bound_file(p, binding, used)
            p.write_text('{"changed":1}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                b.bound_file(p, binding, used)

    def test_unbound_file_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not bound'):
            b.bound_file('not_registered.json', {}, {})


class ComparisonTests(unittest.TestCase):
    def test_missing_experiments_are_not_zero_results(self):
        r = combined_evidence(reference_fixture())
        self.assertEqual(len(r['rows']), 1)
        self.assertEqual(len(r['unavailable_light_experiments']), 3)
        self.assertFalse(r['automatic_ranking'])

    def test_does_not_mutate_source_or_mix_selection_policies(self):
        source = reference_fixture(); before = copy.deepcopy(source)
        source['rows'].append(dict(copy.deepcopy(source['rows'][0]), selection_kind='real'))
        before = copy.deepcopy(source)
        out = combined_evidence(source)
        self.assertNotEqual(out['rows'][0]['comparison_group'], out['rows'][1]['comparison_group'])
        self.assertEqual(source, before)

    def test_five_fold_policy_is_not_matched_test(self):
        source = reference_fixture(); source['rows'][0]['policy'] = 'bounded_max_f1'
        with self.assertRaisesRegex(ValueError, 'five-fold'):
            combined_evidence(source)

    def test_duplicate_result_grain_fails(self):
        source = reference_fixture(); source['rows'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate result'):
            combined_evidence(source)

    def test_mismatched_class_counts_fail(self):
        source = reference_fixture(); second = copy.deepcopy(source['rows'][0])
        second['model'] = 'mergefix_m12'; second['metrics']['positives'] = 60
        source['rows'].append(second)
        with self.assertRaisesRegex(ValueError, 'class counts'):
            combined_evidence(source)

    def test_turufan_zero_gt_rejected(self):
        source = reference_fixture(); source['rows'][0].update(split='turufan',
            metrics=dict(pairs=122, positives=61, negatives=61, layout20_count=0, joint_f1=0))
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            combined_evidence(source)

    def test_new_sim_population_not_v14(self):
        source = reference_fixture(); source['rows'][0].update(model='aggressive_binary_patch', split='sim_test_v14')
        with self.assertRaisesRegex(ValueError, 'distinct'):
            combined_evidence(source)


if __name__ == '__main__':
    unittest.main()
