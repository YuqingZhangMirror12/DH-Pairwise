import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import score_scorer as adapter
import dispatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'diagnostics_source_03'))
import diagnostic_metrics as metrics


def pair(pid, score=.7, label=True, fold=2, correct=True, present=True, valid=True):
    pose = [2., 5.] if present else None
    candidates = [dict(cluster_id=0, selected=True, score=score, logit=score,
                       refined_translation=pose)] if present else []
    return dict(pair_id=pid, label=label, fold=fold, gt_known=label, score=score if present else 0.,
        numeric_valid=valid, has_candidate=present, accepted=bool(present and valid and score >= .5),
        selected_cluster_id=0 if present else -1, candidates=candidates, candidate_count=len(candidates),
        translation=pose, target_translation_rc=[1., 4.] if label else None,
        layout20=bool(label and correct and present and valid), candidate_coverage=bool(label and correct and present))


def strata(rows):
    return [dict(pair_id=r['pair_id'], label=r['label'], fold=r['fold'],
        role='real_cal' if r['fold'] == 1 else 'real_select', seam_group='J' if r['label'] else None) for r in rows]


def fixtures():
    rows = [pair('c'+str(i), score=(i+1)/110, label=False, fold=1, correct=False) for i in range(102)]
    rows += [pair('cp', .99, fold=1), pair('j', .96), pair('r', .94), pair('curve', .92),
             pair('wrong', .97, correct=False), pair('missing', present=False), pair('invalid', valid=False),
             pair('negative', .98, label=False, correct=False), pair('true_negative', .01, label=False, correct=False)]
    normalized = adapter.development_rows(rows, strata(rows), .5)
    for r in normalized:
        if r['pair_id'] == 'r':r['seam_group'] = 'R'
        if r['pair_id'] == 'curve':r['seam_group'] = 'curved'
    return rows, normalized


def origin(arm='B0'):
    return dict(schema='curriculum-scorer-evaluation-origin/1' if arm == 'B0' else 'matcher-v2-terminal-origin/1',
        arm=arm, module='scorer_patch', selection_kind='sim_best', selection_on_real=False, selection_on_test=False,
        threshold_refitted=False, gt_used_for_prediction=False, matcher_updated_during_training=False,
        old_head_imported=False, local_conflict_head_present=False, real_plan_sha256='roles', split='dunhuang_cv',
        total_completed_updates=24000 if arm in ('B0', 'B2') else 31667, selected_updates=6000,
        threshold=.5, thresholds=dict(sim_test=.5, dunhuang_cv=.5))


class Extraction(unittest.TestCase):
    def test_identity_order_winner_and_original_immutable(self):
        rows = [pair('b'), pair('a')]; before = copy.deepcopy(rows)
        out = adapter.development_rows(rows, strata(list(reversed(rows))), .5)
        self.assertEqual([r['pair_id'] for r in out], ['a', 'b'])
        self.assertEqual(out[0]['selected_cluster_id'], 0); self.assertEqual(rows, before)

    def test_test_prediction_values_never_used(self):
        rows = [pair('a')]; out = adapter.development_rows(rows, strata(rows), .5)
        rows.append(dict(pair_id='heldout', score=float('nan'), label='deliberately unreadable'))
        self.assertEqual(adapter.development_rows(rows, strata(rows[:1]), .5), out)

    def test_test_membership_rejected(self):
        rows = [pair('a', fold=0)]
        with self.assertRaisesRegex(ValueError, 'TEST'):adapter.development_rows(rows, strata(rows), .5)

    def test_wrong_cal_fold_rejected(self):
        rows = [pair('a')]; members = strata(rows); members[0]['role'] = 'real_cal'
        with self.assertRaisesRegex(ValueError, 'role'):adapter.development_rows(rows, members, .5)

    def test_duplicate_prediction_rejected(self):
        rows = [pair('a')]
        with self.assertRaisesRegex(ValueError, 'duplicate'):adapter.development_rows(rows*2, strata(rows), .5)

    def test_duplicate_membership_rejected(self):
        rows = [pair('a')]
        with self.assertRaisesRegex(ValueError, 'duplicate'):adapter.development_rows(rows, strata(rows)*2, .5)

    def test_missing_member_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing'):adapter.development_rows([], strata([pair('a')]), .5)

    def test_wrong_label_and_fold_rejected(self):
        for change in (dict(label=False), dict(fold=3)):
            rows = [pair('a')]; members = strata(rows); rows[0].update(change)
            with self.assertRaisesRegex(ValueError, 'labels/folds'):adapter.development_rows(rows, members, .5)

    def test_wrong_winner_or_score_rejected(self):
        for change in (dict(selected_cluster_id=8), dict(score=.8), dict(translation=[1., 1.])):
            rows = [pair('a')]; rows[0].update(change)
            with self.assertRaises(ValueError):adapter.development_rows(rows, strata(rows), .5)

    def test_nonmax_logit_cannot_be_selected(self):
        rows = [pair('a')]; rows[0]['candidates'].append(dict(cluster_id=1, selected=False, logit=2., score=.9))
        rows[0]['candidate_count'] = 2
        with self.assertRaisesRegex(ValueError, 'reranked'):adapter.development_rows(rows, strata(rows), .5)

    def test_nonfinite_probability_rejected(self):
        for score in (math.nan, math.inf, -1., 1.01):
            rows = [pair('a', score)]
            with self.assertRaisesRegex(ValueError, 'probability'):adapter.development_rows(rows, strata(rows), .5)

    def test_original_acceptance_checked(self):
        rows = [pair('a')]; rows[0]['accepted'] = False
        with self.assertRaisesRegex(ValueError, 'acceptance'):adapter.development_rows(rows, strata(rows), .5)

    def test_false_negative_layout_normalized_null(self):
        rows = [pair('n', label=False)]; out = adapter.development_rows(rows, strata(rows), .5)
        self.assertIsNone(out[0]['layout20']); self.assertIsNone(out[0]['candidate_coverage'])

    def test_negative_with_gt_rejected(self):
        rows = [pair('n', label=False)]; rows[0]['gt_known'] = True
        with self.assertRaisesRegex(ValueError, 'no Layout GT'):adapter.development_rows(rows, strata(rows), .5)

    def test_no_candidate_and_invalid_retained(self):
        rows = [pair('absent', present=False), pair('invalid', valid=False)]
        out = adapter.development_rows(rows, strata(rows), .5)
        self.assertEqual(len(out), 2); self.assertFalse(out[0]['has_candidate']); self.assertFalse(out[1]['numeric_valid'])

    def test_missing_positive_gt_rejected(self):
        rows = [pair('p')]; rows[0]['gt_known'] = False
        with self.assertRaisesRegex(ValueError, 'known Dunhuang'):adapter.development_rows(rows, strata(rows), .5)


class BudgetMetrics(unittest.TestCase):
    def test_independent_counts_and_original_threshold_separate(self):
        _, rows = fixtures(); result = adapter.compute(rows, .5, metrics)
        original = result['original_sim_threshold']; budget = result['cal_negative_budgets']['0.02']
        self.assertEqual(original['schema'], 'select-at-frozen-sim-threshold/1')
        self.assertEqual((original['tp'], original['fp'], original['tn'], original['fn']), (4, 1, 1, 2))
        self.assertEqual((original['layout_correct'], original['layout_correct_and_accepted']), (3, 3))
        self.assertEqual(original['wrong_layout_accepted'], 1)
        self.assertEqual(original['positive_count'], 6)
        self.assertEqual(original['no_candidate'], 1); self.assertEqual(original['numeric_invalid'], 1)
        self.assertGreater(budget['threshold'], .9)
        self.assertEqual(budget['observed_select_fpr'], .5)
        self.assertNotEqual(budget['observed_select_fpr'], budget['calibration']['actual_fraction'])

    def test_cal_positives_cannot_change_threshold(self):
        _, rows = fixtures(); a = adapter.compute(rows, .5, metrics)
        for r in rows:
            if r['role'] == 'real_cal' and r['label']:r['score'] = 0.
        b = adapter.compute(rows, .5, metrics)
        self.assertEqual(a, b)

    def test_select_scores_cannot_change_threshold(self):
        _, rows = fixtures(); a = adapter.compute(rows, .5, metrics)
        for r in rows:
            if r['role'] == 'real_select':r['score'] = 0.
        b = adapter.compute(rows, .5, metrics)
        for key in a['cal_negative_budgets']:
            self.assertEqual(a['cal_negative_budgets'][key]['calibration'], b['cal_negative_budgets'][key]['calibration'])

    def test_frozen_acceptance_not_reused(self):
        _, rows = fixtures(); a = adapter.compute(rows, .5, metrics)
        for r in rows:r['frozen_accepted'] = not r['frozen_accepted']
        self.assertEqual(a, adapter.compute(rows, .5, metrics))

    def test_tied_cal_negatives_not_split(self):
        _, rows = fixtures()
        for r in rows:
            if r['role'] == 'real_cal' and not r['label']:r['score'] = .5
        result = adapter.compute(rows, .5, metrics)['cal_negative_budgets']['0.02']
        self.assertEqual(result['calibration']['actual_false_positives'], 0)
        self.assertGreater(result['threshold'], .5)

    def test_cal_failure_rows_stay_in_denominator(self):
        _, rows = fixtures()
        for r in rows:
            if r['role'] == 'real_cal' and not r['label']:r['has_candidate'] = False
        result = adapter.compute(rows, .5, metrics)['cal_negative_budgets']['0.02']
        self.assertEqual(result['calibration']['negative_count'], 102)
        self.assertEqual(result['calibration']['allowed_false_positives'], 2)
        self.assertEqual(result['threshold'], 0.)
        self.assertEqual(result['independent_set_recount']['no_candidate'], 1)

    def test_group_totals_reconcile(self):
        _, rows = fixtures(); result = adapter.compute(rows, .5, metrics)
        for branch in result['cal_negative_budgets'].values():
            groups = branch['positive_groups']
            self.assertEqual(sum(g['positive_count'] for g in groups.values()), 6)
            self.assertEqual(sum(g['layout_correct_and_accepted'] for g in groups.values()), branch['layout_correct_and_accepted'])
            self.assertEqual(groups['unmeasurable']['positive_count'], 0)

    def test_missing_group_rejected(self):
        _, rows = fixtures(); rows = [r for r in rows if r['role'] == 'real_select']
        with self.assertRaisesRegex(ValueError, 'complete positive'):
            adapter.fixed_threshold_readout(rows, .5, {})


class Origins(unittest.TestCase):
    def test_all_arms_and_modules(self):
        for arm in ('B0', 'B1', 'B2', 'B3'):
            for module in adapter.MODULES:
                value = origin(arm); value['module'] = module
                adapter.verify_origin(value, arm, module, 'roles')

    def test_real_test_refitted_or_gt_prediction_rejected(self):
        for key in ('selection_on_real', 'selection_on_test', 'threshold_refitted', 'gt_used_for_prediction'):
            value = origin(); value[key] = True
            with self.assertRaisesRegex(ValueError, 'SIM-selected'):adapter.verify_origin(value, 'B0', 'scorer_patch', 'roles')

    def test_wrong_budget_arm_and_threshold_rejected(self):
        for key, value in (('total_completed_updates', 24000), ('arm', 'B2'), ('threshold', .3)):
            obj = origin('B3'); obj[key] = value
            with self.assertRaises(ValueError):adapter.verify_origin(obj, 'B3', 'scorer_patch', 'roles')


class QueueContract(unittest.TestCase):
    def test_isolated_b0_binds_original_baseline_before_eval_aliases(self):
        source = Path('/fixture/source'); package = source/adapter.PACKAGE.replace('.', '/')
        entry = SimpleNamespace(__file__=str(package/'curriculum_scorer_eval_v1/entry.py'),
            bind_evaluation=mock.Mock())
        controller = SimpleNamespace(__file__=str(package/'curriculum_scorer_eval_v1/controller.py'))
        auditor = SimpleNamespace(__file__=str(package/'curriculum_scorer_eval_v1/audit.py'))
        calls = []
        entry.bind_evaluation.side_effect = lambda *_:calls.append('aliases')
        with mock.patch.object(adapter.importlib, 'import_module', side_effect=[entry, controller, auditor]), \
             mock.patch.object(adapter, 'baseline_for_b0', side_effect=lambda *_:calls.append('baseline')):
            self.assertIs(adapter.load_verifier(source, Path('/evaluation'), 'B0'), controller)
        self.assertEqual(calls, ['baseline', 'aliases'])

    def test_registry_only_eight_approved_head_readouts(self):
        jobs = dispatch.registered_jobs()
        self.assertEqual(len(jobs), 8); self.assertEqual(len({j['name'] for j in jobs}), 8)
        self.assertEqual([j['arm'] for j in jobs], ['B0', 'B0', 'B3', 'B3', 'B1', 'B1', 'B2', 'B2'])
        self.assertEqual(set(j['module'] for j in jobs), set(adapter.MODULES))

    def test_command_cpu_posthoc_only(self):
        for job in dispatch.registered_jobs():
            values = dispatch.command(job, Path('/new'), Path('/python'))
            self.assertIn('--evaluation-root', values)
            self.assertEqual(values[values.index('--out')+1], '/new')
            self.assertNotIn('--gpus', values); self.assertNotIn('--device', values)
            self.assertNotIn('--threshold', values)

    def test_waiting_until_full_pipeline_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); pipeline = root/'pipeline'; pipeline.mkdir()
            job = dict(arm='B0', module='scorer_patch', pipeline=str(pipeline))
            adapter.save(pipeline/'status.json', dict(status='complete'))
            self.assertEqual(dispatch.upstream(job, root)['state'], 'waiting')
            adapter.save(pipeline/'pipeline_complete.json', dict(status='training_and_required_evaluation_complete', module='scorer_patch'))
            self.assertEqual(dispatch.upstream(job, root)['state'], 'ready')

    def test_b0_does_not_use_v2_terminal_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); adapter.save(root/'complete.json', dict(status='complete', module='scorer_patch'))
            job = dict(arm='B0', module='scorer_patch', pipeline=str(root))
            self.assertEqual(dispatch.upstream(job, root)['state'], 'waiting')

    def test_v2_terminal_filename_and_wrong_module(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); pipeline = root/'b3'; pipeline.mkdir()
            job = dict(arm='B3', module='scorer_patch', pipeline=str(pipeline))
            adapter.save(pipeline/'complete.json', dict(status='complete', module='scorer_stats'))
            with self.assertRaisesRegex(ValueError, 'wrong-module'):dispatch.upstream(job, root)

    def test_failure_precedes_stale_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); adapter.save(root/'complete.json', dict(status='complete'))
            adapter.save(root/'failure.json', dict(status='failed'))
            job = dict(arm='B0', module='scorer_patch', pipeline=str(root))
            self.assertEqual(dispatch.upstream(job, root)['state'], 'failed')

    def test_head_queue_failure_detected_before_output_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'b3_head_queue_01').mkdir()
            adapter.save(root/'b3_head_queue_01/failure.json', dict(status='failed'))
            job = dict(arm='B3', module='scorer_patch', pipeline=str(root/'never_started'))
            self.assertEqual(dispatch.upstream(job, root)['state'], 'failed')

    def test_real_cpu_child_success_then_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = Path(folder); job = dict(name='fixture', arm='B0', module='scorer_patch')
            verifier = mock.Mock(return_value=dict(status='passed'))
            def cmd(*_):
                return [sys.executable, '-c', 'import os; assert os.environ["CUDA_VISIBLE_DEVICES"] == ""; print("fixture only")']
            result = dispatch.execute_job(job, queue, Path(sys.executable), {}, build=cmd, verify=verifier)
            self.assertEqual(adapter.read(queue/'fixture_return.json')['returncode'], 0)
            self.assertEqual(result['audit']['status'], 'passed'); verifier.assert_called_once()

    def test_real_cpu_child_failure_never_audited_or_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = Path(folder); job = dict(name='fixture', arm='B0', module='scorer_patch')
            verifier = mock.Mock()
            def cmd(*_):return [sys.executable, '-c', 'raise SystemExit(7)']
            with self.assertRaisesRegex(ValueError, 'child failed'):
                dispatch.execute_job(job, queue, Path(sys.executable), {}, build=cmd, verify=verifier)
            self.assertEqual(adapter.read(queue/'fixture_return.json')['returncode'], 7)
            verifier.assert_not_called()

    def test_output_hash_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            adapter.save(root/'complete.json', dict(schema='scorer-cal-budgets-complete/1', status='complete',
                arm='B0', module='scorer_patch', development_pairs=639, negative_cal=102, negative_select=304,
                gpu_used=False, model_inference=False, source_sha256={}, files={}))
            with self.assertRaisesRegex(ValueError, 'output changed'):
                adapter.verify_output(root, 'B0', 'scorer_patch', {})


if __name__ == '__main__':unittest.main()
