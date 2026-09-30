"""Synthetic schema/arithmetic fixtures only; no experimental performance claims."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report_projection as report


EVENTS = ('classification_gained', 'classification_lost', 'positive_recovered', 'positive_lost',
          'false_positive_added', 'false_positive_removed', 'layout_gained', 'layout_lost',
          'coverage_gained', 'coverage_lost', 'correct_and_accepted_gained',
          'correct_and_accepted_lost', 'winner_pose_changed')


def fixture_metric(split, n):
    # Synthetic counts chosen to exercise both false negatives and false positives.
    p = n // 2; neg = n - p; tp = p // 2; fp = 1; good = p - 1
    m = dict(pairs=n, positives=p, negatives=neg, threshold=.3, tp=tp, fp=fp,
             fn=p-tp, tn=neg-fp, accuracy=(tp+neg-fp)/n, f1=2*tp/(tp+fp+p),
             precision=tp/(tp+fp), recall=tp/p, ap=.9, auroc=.9,
             false_positive_rate=fp/neg, no_candidate_or_invalid=0,
             layout20_count=good, candidate_coverage_count=p, layout20=good/p,
             candidate_coverage=1., winner_correct_but_rejected=good-tp,
             covered_but_winner_wrong=1, positive_no_correct_candidate=0,
             wrong_pose_accepted=0, joint_tp=tp, joint_fp=fp, joint_fn=p-tp,
             joint_f1=2*tp/(tp+fp+p), joint_precision=tp/(tp+fp), joint_recall=tp/p,
             known_positive_layouts=p)
    if split == 'turufan': m.update({k: None for k in report.LAYOUT_FIELDS})
    return m


def fixture_analysis():
    result = dict(schema='decoder-controls-independent-analysis/1', status='complete',
                  test_used=False, training_performed=False, model_modified=False,
                  historical_real_exposure=True, analysis_source_sha256='a'*64,
                  controller_complete_sha256='b'*64, real_role_plan_sha256='c'*64, models={})
    for model in report.MODELS:
        rows = []
        for split, role, search, readout, mode in sorted(report._expected(model)):
            n = report.ROLE_SIZE[(split, role)]; metric = fixture_metric(split, n)
            paired = dict(counts={k: 0 for k in EVENTS}, pair_ids={k: [] for k in EVENTS})
            if split == 'turufan':
                for key in EVENTS:
                    if key.startswith(('layout_', 'coverage_', 'correct_and_accepted_')):
                        paired['counts'][key] = paired['pair_ids'][key] = None
            distributions = {}
            for name, count in (('all_positive_winners', metric['positives']),
                                ('all_negative_winners', metric['negatives']),
                                ('correct_positive_winners', metric['layout20_count'] or 0)):
                distributions[name] = {field: dict(count=count, mean=1. if count else None,
                    **{'p'+str(q): 1. if count else None for q in (10,25,50,75,90)})
                    for field in ('pairs', 'sum_q', 'max_q', 'candidate_count', 'score')}
            rows.append(dict(split=split, role=role, search=search, readout=readout, threshold_mode=mode,
                metrics=metric, baseline_threshold=.3,
                delta={k: 0 if metric[k] is not None else None for k in report.DELTA_FIELDS},
                paired=paired, distributions=distributions,
                mechanism_only=readout.startswith('zero_'),
                not_a_calibrated_q_classifier=readout.startswith('raw_'),
                runtime_seconds=dict(count=n, mean=.1, p90=.2),
                strict_cpu_gpu_reproduction_subset=None if split == 'sim_select' else dict(
                    pairs=n, excluded_pairs=0, threshold_refitted_on_subset=False,
                    metrics=deepcopy(metric), paired=deepcopy(paired))))
        result['models'][model] = dict(model=model, rows=rows, development_ranking=[],
            protocol=dict(model=model, threshold=.3, checkpoint_sha256='d'*64),
            sim_select_manifest_sha256='e'*64, new_pooling_heads_included=False,
            automatic_production_change=False)
    return result


def find(analysis, *, model='binary_patch', split='dunhuang_cv', role='real_select',
         search='top3_only', readout='baseline', mode='original_frozen_cal'):
    return next(r for r in analysis['models'][model]['rows']
                if report._key(r) == (split, role, search, readout, mode))


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.fixture = fixture_analysis()

    def setUp(self): self.data = deepcopy(self.fixture)

    def test_complete_projection_keeps_all_comparisons_and_deduplicates_distributions(self):
        p = report.project(self.data, source_name='independent_analysis.json', source_sha256='f'*64)
        q = p['queries']; metrics = q['decoder_factor_metrics']['rows']
        self.assertEqual(len(metrics), 1080)
        self.assertEqual(len(q['decoder_factor_paired']['rows']), 1080)
        self.assertEqual(len(q['decoder_factor_distributions']['rows']), 9000)
        self.assertEqual(len(q['decoder_factor_strict']['rows']), 960)
        self.assertEqual(len(q['decoder_factor_models']['rows']), 4)
        self.assertFalse(p['new_pooling_heads_included']); self.assertFalse(p['live_report_modified'])
        self.assertEqual(q['decoder_factor_paired']['payloadColumns'], ['pair_ids_json'])

    def test_incomplete_and_missing_model_rejected(self):
        for change in ('status', 'model_complete', 'missing_model', 'extra_model'):
            bad = deepcopy(self.data)
            if change == 'status': bad['status'] = 'running'
            elif change == 'model_complete': bad['status'] = 'model_complete'
            elif change == 'missing_model': del bad['models']['binary_stats']
            else: bad['models']['mean_only'] = deepcopy(bad['models']['binary_patch'])
            with self.assertRaises(ValueError): report.validate(bad)

    def test_no_test_training_or_automatic_promotion(self):
        for key in ('test_used', 'training_performed', 'model_modified'):
            bad = deepcopy(self.data); bad[key] = True
            with self.assertRaises(ValueError): report.validate(bad)
        self.data['models']['binary_patch']['automatic_production_change'] = True
        with self.assertRaises(ValueError): report.validate(self.data)

    def test_role_population_and_duplicate_checks(self):
        for change in ('role', 'population', 'duplicate', 'missing'):
            bad = deepcopy(self.data); r = find(bad)
            if change == 'role': r['role'] = 'real_test'
            elif change == 'population': r['metrics']['pairs'] -= 1
            elif change == 'duplicate': bad['models']['binary_patch']['rows'].append(deepcopy(r))
            else: bad['models']['binary_patch']['rows'].remove(r)
            with self.assertRaises(ValueError): report.validate(bad)

    def test_confusion_and_f1_recompute(self):
        for key in ('tp', 'accuracy', 'f1'):
            bad = deepcopy(self.data); find(bad)['metrics'][key] += .1
            with self.assertRaises(ValueError): report.validate(bad)

    def test_joint_and_failure_category_recompute(self):
        for key in ('joint_f1', 'wrong_pose_accepted', 'winner_correct_but_rejected'):
            bad = deepcopy(self.data); find(bad)['metrics'][key] += 1
            with self.assertRaises(ValueError): report.validate(bad)

    def test_turufan_null_not_zero(self):
        r = find(self.data, split='turufan'); r['metrics']['layout20_count'] = 0
        with self.assertRaisesRegex(ValueError, 'Turufan'): report.validate(self.data)

    def test_frozen_threshold_preserved(self):
        find(self.data)['metrics']['threshold'] = .31
        with self.assertRaisesRegex(ValueError, 'frozen threshold'): report.validate(self.data)

    def test_select_threshold_inherited_only_from_cal(self):
        find(self.data, mode='separate_real_cal')['metrics']['threshold'] = .31
        with self.assertRaisesRegex(ValueError, 'SELECT differs'): report.validate(self.data)

    def test_matching_baseline_not_other_population(self):
        find(self.data)['delta']['f1'] = .1
        with self.assertRaisesRegex(ValueError, 'same-population'): report.validate(self.data)

    def test_paired_counts_and_ids_reconcile(self):
        for change in ('count', 'duplicate', 'arithmetic'):
            bad = deepcopy(self.data); r = find(bad)['paired']
            if change == 'count': r['counts']['classification_gained'] = 1
            elif change == 'duplicate':
                r['counts']['classification_gained'] = 2; r['pair_ids']['classification_gained'] = ['x','x']
            else:
                r['counts']['false_positive_added'] = 1; r['pair_ids']['false_positive_added'] = ['x']
            with self.assertRaises(ValueError): report.validate(bad)

    def test_paired_gain_and_loss_can_coexist_but_not_for_the_same_id(self):
        r = find(self.data); m = r['metrics']; before = deepcopy(m)
        m['tp'] += 1; m['fn'] -= 1; m['joint_tp'] += 1; m['joint_fn'] -= 1
        m['winner_correct_but_rejected'] -= 1
        m['accuracy'] = (m['tp']+m['tn'])/m['pairs']
        m['f1'] = m['joint_f1'] = 2*m['tp']/(2*m['tp']+m['fp']+m['fn'])
        r['delta'] = {k: m[k]-before[k] for k in report.DELTA_FIELDS}
        for gain, loss in (('classification_gained', 'classification_lost'),
                           ('positive_recovered', 'positive_lost'),
                           ('correct_and_accepted_gained', 'correct_and_accepted_lost')):
            r['paired']['counts'].update({gain: 2, loss: 1})
            r['paired']['pair_ids'].update({gain: ['gain1', 'gain2'], loss: ['loss1']})
        r['strict_cpu_gpu_reproduction_subset']['metrics'] = deepcopy(m)
        r['strict_cpu_gpu_reproduction_subset']['paired'] = deepcopy(r['paired'])
        report.validate(self.data)
        r['paired']['pair_ids']['classification_lost'] = ['gain1']
        with self.assertRaisesRegex(ValueError, 'both gained and lost'): report.validate(self.data)

    def test_distributions_keep_full_population_not_only_survivors(self):
        r = find(self.data)
        r['distributions']['all_positive_winners']['sum_q']['count'] -= 1
        with self.assertRaisesRegex(ValueError, 'survivor'): report.validate(self.data)

    def test_empty_distribution_is_null_not_zero(self):
        r = find(self.data, split='turufan')
        r['distributions']['correct_positive_winners']['score']['mean'] = 0.
        with self.assertRaisesRegex(ValueError, 'distribution'): report.validate(self.data)

    def test_distribution_cannot_change_with_only_the_acceptance_threshold(self):
        r = find(self.data, mode='separate_real_cal')
        r['distributions']['all_negative_winners']['sum_q']['mean'] = .5
        with self.assertRaisesRegex(ValueError, 'threshold-independent'): report.validate(self.data)

    def test_raw_q_and_pooling_diagnostic_flags_required(self):
        find(self.data, readout='raw_sum_q_rank')['not_a_calibrated_q_classifier'] = False
        with self.assertRaisesRegex(ValueError, 'semantic flag'): report.validate(self.data)

    def test_strict_subset_does_not_refit_or_disappear(self):
        for change in ('threshold', 'refitted', 'missing'):
            bad = deepcopy(self.data); r = find(bad)['strict_cpu_gpu_reproduction_subset']
            if change == 'threshold': r['metrics']['threshold'] = .31
            elif change == 'refitted': r['threshold_refitted_on_subset'] = True
            else: r['metrics'] = None
            with self.assertRaises(ValueError): report.validate(bad)

    def test_nonfinite_is_not_silently_zero_filled(self):
        find(self.data)['metrics']['f1'] = math.nan
        with self.assertRaises(ValueError): report.validate(self.data)

    def test_view_preserves_distinct_sim_datasets_and_control_semantics(self):
        p = report.project(self.data, source_name='analysis.json', source_sha256='f'*64)
        rows = p['queries']['decoder_factor_metrics']['rows']
        for r in rows:
            self.assertEqual(r['sim_data'], 'v17' if r['model'] == 'aggressive_binary_patch' else 'v14')
            if r['readout'] in ('zero_mean_max', 'no_conflict_or_overlap') or r['search'] == 'combined':
                self.assertEqual(r['control_class'], '联合干预')
            if r['split'] == 'turufan': self.assertIsNone(r['joint_f1'])
            if r['raw_q_ranking_only']: self.assertIn('保留其Scorer分数', r['readout_label'])

    def test_local_paths_not_in_source_identity(self):
        with self.assertRaisesRegex(ValueError, 'safe source identity'):
            report.project(self.data, source_name='/private/analysis.json', source_sha256='f'*64)

    def test_cli_never_overwrites_an_existing_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); src = folder/'analysis.json'; dst = folder/'queries.json'
            src.write_text(json.dumps(self.data)); dst.write_text('user-owned')
            with patch('sys.argv', ['report_projection.py', '--analysis', str(src), '--out', str(dst)]):
                with self.assertRaises(FileExistsError): report.main()
            self.assertEqual(dst.read_text(), 'user-owned')


if __name__ == '__main__': unittest.main()
