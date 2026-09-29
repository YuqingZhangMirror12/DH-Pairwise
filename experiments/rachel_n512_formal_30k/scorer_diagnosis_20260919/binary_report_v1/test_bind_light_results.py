"""Synthetic report-interface tests; no model inference or remote queries."""
import copy
import json
import unittest

from .assemble import EXPERIMENTS, REAL_PLAN_SHA, tasks
from .bind_light_results import CASES_QUERY, METRICS_QUERY, REPORT_ID, bind_snapshot, query_rows


def fixture(experiment='binary_patch'):
    rows, groups = [], []
    for choice, split in sorted(tasks(experiment)):
        checkpoint = ('1' if choice == 'sim' else '2') * 64
        epoch = 8 if choice == 'sim' else 10
        n, pos, neg = ((3000, 1500, 1500) if split.startswith('sim_test_') else
                      (161, 59, 102) if split == 'dunhuang_cv' else (122, 61, 61))
        for policy in ('primary', 'fixed03'):
            rows.append(dict(model=experiment, selection_kind=choice, split=split,
                population='all' if split.startswith('sim_test_') else 'real_test', policy=policy,
                real_plan_sha256=REAL_PLAN_SHA, checkpoint_sha256=checkpoint, selected_epoch=epoch,
                source={'synthetic_test': True}, metrics=dict(pairs=n, positives=pos, negatives=neg,
                    threshold=.21 if policy == 'primary' else .3, tp=1, fp=0, f1=2/(pos+1),
                    joint_f1=None if split == 'turufan' else 0.,
                    layout20_count=None if split == 'turufan' else 1)))
        cases = []
        for i in range(10 if split == 'dunhuang_cv' else 1 if split == 'turufan' else 0):
            cases.append(dict(schema='binary-report-case/1', pair_id=f'synthetic-{split}-{i}',
                variant=EXPERIMENTS[experiment]['variant'], threshold=.21, score=.4, has_candidate=True,
                provenance=dict(selection_kind=choice, split=split, selected_epoch=epoch,
                                checkpoint_sha256=checkpoint),
                verification=dict(imported_numeric_audit_status='passed', this_export_repeated_model_inference=False),
                semantics={k: False for k in ('attention_present', 'local_conflict_classifier',
                    'learned_refinement', 'pooling_weights_are_attention', 'layer_values_are_causal_importance')}))
        groups.append(dict(experiment=experiment, selection_kind=choice, split=split, cases=cases))
    return dict(schema='binary-report-all-models/1', status='evidence_ready_not_final_conclusions',
        neural_inference_repeated=False, thresholds_refitted=False, rows=rows, binary_case_groups=groups,
        available_light_experiments=[experiment], caveats=['Synthetic test, not performance.'])


def continuation_fixture(experiment='binary_patch'):
    value = fixture(experiment)
    execution = dict(schema='binary-microbatch-continuation/1', updates=200, exposures=6400,
                     original_checkpoint_sha256='a' * 64, resume_origin_sha256='b' * 64)
    for row in value['rows']:
        row['execution_continuation'] = copy.deepcopy(execution)
    for group in value['binary_case_groups']:
        for case in group['cases']:
            case['provenance']['execution_continuation'] = copy.deepcopy(execution)
    return value, execution


class LightBindingTests(unittest.TestCase):
    def test_two_policies_and_all_twenty_two_cases_preserved(self):
        value = fixture(); before = copy.deepcopy(value)
        rows, cases = query_rows(value)
        self.assertEqual((len(rows), len(cases)), (12, 22))
        self.assertEqual(value, before)
        self.assertEqual(len({c['case_key'] for c in cases}), 22)
        self.assertTrue(all(json.loads(c['record_json'])['threshold'] == .21 for c in cases))

    def test_missing_result_not_a_zero_score(self):
        value = fixture(); value['available_light_experiments'] = []
        with self.assertRaisesRegex(ValueError, 'no completed'): query_rows(value)

    def test_incomplete_and_duplicate_metric_rows_fail(self):
        for kind in ('missing', 'duplicate'):
            value = fixture()
            if kind == 'missing': value['rows'].pop()
            else: value['rows'].append(copy.deepcopy(value['rows'][0]))
            with self.assertRaises(ValueError): query_rows(value)

    def test_no_pooling_of_overlapping_real_populations(self):
        value = fixture(); extra = copy.deepcopy(value['rows'][0]); extra['population'] = 'real_select'
        value['rows'].append(extra)
        self.assertEqual(len(query_rows(value)[0]), 12)

    def test_roles_and_class_counts_verified(self):
        for kind in ('roles', 'counts'):
            value = fixture(); row = next(r for r in value['rows'] if r['split'] == 'dunhuang_cv')
            if kind == 'roles': row['real_plan_sha256'] = 'different'
            else: row['metrics'].update(positives=58, negatives=103)
            with self.assertRaises(ValueError): query_rows(value)

    def test_new_simulation_test_is_not_v14(self):
        value = fixture('aggressive_binary_patch')
        self.assertTrue(any(r['split'] == 'sim_test_aggressive' for r in query_rows(value)[0]))
        next(r for r in value['rows'] if r['split'] == 'sim_test_aggressive')['split'] = 'sim_test_v14'
        with self.assertRaisesRegex(ValueError, 'dataset scope'): query_rows(value)

    def test_turufan_unavailable_not_zero(self):
        value = fixture(); next(r for r in value['rows'] if r['split'] == 'turufan')['metrics']['joint_f1'] = 0.
        with self.assertRaisesRegex(ValueError, 'no Layout GT'): query_rows(value)

    def test_six_case_inventories_including_sim_required(self):
        value = fixture(); value['binary_case_groups'] = [g for g in value['binary_case_groups'] if g['cases']]
        with self.assertRaisesRegex(ValueError, 'all six'): query_rows(value)

    def test_case_checkpoint_threshold_audit_and_semantics(self):
        for kind in ('checkpoint', 'threshold', 'audit', 'attention'):
            value = fixture(); case = next(g for g in value['binary_case_groups'] if g['cases'])['cases'][0]
            if kind == 'checkpoint': case['provenance']['checkpoint_sha256'] = 'different'
            elif kind == 'threshold': case['threshold'] = .3
            elif kind == 'audit': case['verification']['imported_numeric_audit_status'] = 'failed'
            else: case['semantics']['pooling_weights_are_attention'] = True
            with self.assertRaises(ValueError): query_rows(value)

    def test_fixed_cases_must_not_be_rechosen_after_results(self):
        value = fixture(); group = next(g for g in value['binary_case_groups']
                                      if g['selection_kind'] == 'sim' and g['split'] == 'dunhuang_cv')
        group['cases'][0]['pair_id'] = 'different-case'
        with self.assertRaisesRegex(ValueError, 'fixed cases changed'): query_rows(value)

    def test_existing_report_identity_evidence_and_asof_preserved(self):
        original = dict(id=REPORT_ID, queries={'existing': {'rows': [1], 'source': {'hash': 'existing'}}},
                        report={'asOf': '2026-09-27'}, title='Original title', buildStatus='complete')
        before = copy.deepcopy(original)
        result = bind_snapshot(original, fixture(), '/synthetic/evidence.json', 'synthetic')
        self.assertEqual(original, before)
        for key in ('id', 'report', 'title'): self.assertEqual(result[key], original[key])
        self.assertEqual(result['queries']['existing'], original['queries']['existing'])
        self.assertEqual(result['queries'][CASES_QUERY]['payloadColumns'], ['record_json'])
        self.assertEqual(result['buildStatus'], 'updating')
        unchanged = bind_snapshot(result, fixture(), '/new/synthetic.json', 'new')
        self.assertEqual(unchanged['queries'][METRICS_QUERY]['rows'], result['queries'][METRICS_QUERY]['rows'])

    def test_refresh_cannot_silently_change_imported_metrics(self):
        result = bind_snapshot(dict(id=REPORT_ID, queries={}), fixture(), '/synthetic', 's')
        value = fixture(); value['rows'][0]['metrics']['f1'] = .8
        with self.assertRaisesRegex(ValueError, 'preserve already imported'):
            bind_snapshot(result, value, '/synthetic', 's')

    def test_continuation_survives_both_head_host_bindings(self):
        for experiment in ('binary_patch', 'binary_stats'):
            with self.subTest(experiment=experiment):
                value, execution = continuation_fixture(experiment)
                before = copy.deepcopy(value)
                result = bind_snapshot(dict(id=REPORT_ID, queries={}), value, '/synthetic', 's')
                rows = result['queries'][METRICS_QUERY]['rows']
                cases = result['queries'][CASES_QUERY]['rows']
                self.assertEqual((len(rows), len(cases)), (12, 22))
                self.assertTrue(all(r['execution_continuation'] == execution for r in rows))
                self.assertTrue(all(json.loads(c['record_json'])['provenance']['execution_continuation']
                                    == execution for c in cases))
                self.assertEqual(value, before)

    def test_missing_case_continuation_cannot_look_like_unmigrated_results(self):
        value, _ = continuation_fixture()
        case = next(g for g in value['binary_case_groups'] if g['cases'])['cases'][0]
        del case['provenance']['execution_continuation']
        with self.assertRaisesRegex(ValueError, 'case execution continuation differs'):
            query_rows(value)

    def test_different_continuation_cannot_reuse_same_checkpoint_identity(self):
        value, _ = continuation_fixture()
        case = next(g for g in value['binary_case_groups'] if g['cases'])['cases'][0]
        case['provenance']['execution_continuation']['resume_origin_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'case execution continuation differs'):
            query_rows(value)

    def test_case_continuation_cannot_be_silently_dropped_from_metric_row(self):
        value, _ = continuation_fixture()
        row = next(r for r in value['rows'] if r['split'] == 'dunhuang_cv' and r['policy'] == 'primary')
        del row['execution_continuation']
        with self.assertRaisesRegex(ValueError, 'case execution continuation differs'):
            query_rows(value)

    def test_compare_preserves_continuation_before_host_binding(self):
        from .compare import combined_evidence
        for experiment in ('binary_patch', 'binary_stats'):
            with self.subTest(experiment=experiment):
                fixture_bundle, execution = continuation_fixture(experiment)
                rows = copy.deepcopy(fixture_bundle['rows'])
                for row in rows:
                    row['experiment'] = row.pop('model')
                imported = dict(schema='binary-report-comparison/1', status='frozen_evidence_assembly_only',
                    available_experiments=[experiment], rows=rows,
                    experiments=[dict(experiment=experiment, status='six_frozen_jobs_imported',
                        selected={k: dict(real_plan_sha256=REAL_PLAN_SHA) for k in ('sim', 'real')},
                        jobs=[dict(selection_kind=g['selection_kind'], split=g['split'], cases=g['cases'])
                              for g in fixture_bundle['binary_case_groups']])])
                baseline = dict(schema='binary-report-complex-baselines/1', status='baseline_evidence_ready',
                    real_plan_sha256=REAL_PLAN_SHA, threshold_fitted=False, five_fold_metrics_substituted=False,
                    rows=[], caveats=['Synthetic interface fixture, no model results.'])
                combined = combined_evidence(baseline, imported)
                result = bind_snapshot(dict(id=REPORT_ID, queries={}), combined, '/synthetic', 's')
                self.assertTrue(all(r['execution_continuation'] == execution
                                    for r in result['queries'][METRICS_QUERY]['rows']))
                self.assertEqual(len(result['queries'][CASES_QUERY]['rows']), 22)


if __name__ == '__main__':
    unittest.main()
