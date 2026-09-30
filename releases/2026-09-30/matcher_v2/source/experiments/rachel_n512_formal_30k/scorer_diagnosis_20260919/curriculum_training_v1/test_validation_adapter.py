import copy
import importlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from .checkpoint_io import tree_sha
from .exposure import digest
from .runtime_plan import lock_record
from .test_runtime_plan import fixture
from .training_core import Topology, run_updates
from .validation_adapter import (BASE, RULES, ValidationAdapter, check_report,
                                 from_bound_baseline, select_history)


def setup_plan(module='matcher'):
    ledger, record = fixture(); record['module'] = module
    record['selection_rule']['id'] = RULES[module]
    return ledger, lock_record(record, ledger)


def report(module='matcher', score=.6, elapsed=1.):
    sim = dict(stage='matcher' if module == 'matcher' else 'scorer', real_used=False,
        key=[score, .5, -.2 if module == 'matcher' else .7], selection_value=score,
        elapsed_seconds=elapsed, threshold=.3)
    real = None if module == 'matcher' else dict(status='development_selection',
        test_used=False, gradients_used=False, real_used=True, development_evaluation=True,
        thresholds={'dunhuang_cv': .24, 'turufan': .3}, key=[score, .5, .7],
        selection_value=score, elapsed_seconds=elapsed)
    return sim, real


def writer(update, record, rows):
    return dict(path='fixture/validation_%d.json' % update, sha256=digest([record, rows]))


def observations(plan, values=None):
    result = []
    values = values or [.99, .7, .6, .5]
    for update, score in zip(plan.validation_updates, values):
        sim, real = report(plan.record['module'], score)
        adapter = ValidationAdapter(plan, Topology(0, 1, 4, 1), lambda: (sim, ['sim']),
            None if real is None else lambda: (real, ['real']), writer)
        result.append(dict(update=update, report=adapter(update)))
    return result


class ValidationAdapterTests(unittest.TestCase):
    def test_matcher_uses_native_sim_without_a_scorer_call(self):
        _, plan = setup_plan(); sim, _ = report(); calls = []
        result = ValidationAdapter(plan, Topology(0, 1, 4, 1), lambda: (sim, []), None,
            lambda *args: calls.append(args) or writer(*args))(7)
        self.assertEqual(result['simulation'], sim)
        self.assertIsNone(result['real_development']); self.assertEqual(len(calls), 1)

    def test_only_registered_observation_updates_are_allowed(self):
        _, plan = setup_plan(); sim, _ = report()
        adapter = ValidationAdapter(plan, Topology(0, 1, 4, 1), lambda: (sim, []), None, writer)
        for update in (1, 3, 15, 17):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'schedule'):
                adapter(update)

    def test_unknown_selection_rule_does_not_start_evaluation(self):
        ledger, record = fixture(); plan = lock_record(record, ledger)
        with self.assertRaisesRegex(ValueError, 'not implemented'):
            ValidationAdapter(plan, Topology(0, 1, 4, 1), lambda: None, None, writer)

    def test_matcher_forbids_real_scorer_and_heads_require_it(self):
        for module, real in [('matcher', lambda: None), ('scorer_patch', None), ('scorer_stats', None)]:
            _, plan = setup_plan(module)
            with self.subTest(module=module), self.assertRaises(ValueError):
                ValidationAdapter(plan, Topology(0, 1, 4, 1), lambda: None, real, writer)

    def test_real_test_or_gradients_cannot_enter_selection(self):
        for field in ('test_used', 'gradients_used'):
            sim, real = report('scorer_patch'); real[field] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'non-backprop'):
                check_report('scorer_patch', sim, real)

    def test_classifier_threshold_policy_is_preserved(self):
        for threshold in (.19, .81, float('nan'), True):
            sim, real = report('scorer_stats'); real['thresholds']['turufan'] = threshold
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, 'threshold'):
                check_report('scorer_stats', sim, real)

    def test_selection_metrics_must_be_native_finite_and_consistent(self):
        for key in ([], [float('nan'), .3, .4], [.2, False, .4], [.2, .3, .4]):
            sim, _ = report(); sim['key'] = key
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_report('matcher', sim, None)

    def test_scorer_report_not_relabelled_as_matcher_report(self):
        sim, _ = report('scorer_patch')
        with self.assertRaisesRegex(ValueError, 'stage'):
            check_report('matcher', sim, None)

    def test_validation_rows_require_an_artifact_hash(self):
        _, plan = setup_plan(); sim, _ = report()
        for receipt in ({}, {'path': 'file', 'sha256': 'bad'}, None):
            with self.subTest(receipt=receipt), self.assertRaisesRegex(ValueError, 'artifact'):
                ValidationAdapter(plan, Topology(0, 1, 4, 1), lambda: (sim, []), None,
                    lambda *args: receipt)(0)

    def test_rank_zero_canonical_wall_time_shared_without_changing_metrics(self):
        _, plan = setup_plan(); packet = []; written = []
        sim0, _ = report(elapsed=1.); sim1, _ = report(elapsed=9.)
        a = ValidationAdapter(plan, Topology(0, 2, 1, 2), lambda: (sim0, []), None,
            lambda *args: written.append(args) or writer(*args),
            broadcast=lambda value: packet.append(copy.deepcopy(value)) or value)
        b = ValidationAdapter(plan, Topology(1, 2, 1, 2), lambda: (sim1, []), None,
            lambda *args: self.fail('rank1 wrote'), broadcast=lambda value: packet[-1])
        x = a(7); y = b(7)
        self.assertEqual(x, y); self.assertEqual(x['simulation']['elapsed_seconds'], 1.)
        self.assertEqual(len(written), 1); self.assertEqual(sim1['elapsed_seconds'], 9.)

    def test_different_rank_metrics_are_not_hidden_by_broadcast(self):
        _, plan = setup_plan(); packet = []
        sim, _ = report()
        a = ValidationAdapter(plan, Topology(0, 2, 1, 2), lambda: (sim, []), None, writer,
            broadcast=lambda value: packet.append(value) or value)
        a(7); different, _ = report(score=.5)
        b = ValidationAdapter(plan, Topology(1, 2, 1, 2), lambda: (different, []), None, writer,
            broadcast=lambda value: packet[-1])
        with self.assertRaisesRegex(ValueError, 'ranks disagree'):
            b(7)

    def test_writer_failure_is_broadcast_to_nonzero_rank(self):
        _, plan = setup_plan(); packet = []; sim, _ = report()
        def fail(*args):
            raise OSError('fixture write failure')
        a = ValidationAdapter(plan, Topology(0, 2, 1, 2), lambda: (sim, []), None, fail,
            broadcast=lambda value: packet.append(value) or value)
        b = ValidationAdapter(plan, Topology(1, 2, 1, 2), lambda: (sim, []), None, writer,
            broadcast=lambda value: packet[-1])
        for callback in (a, b):
            with self.assertRaisesRegex(ValueError, 'fixture write failure'):
                callback(7)

    def test_update_zero_is_recorded_but_never_selected(self):
        _, plan = setup_plan(); rows = observations(plan)
        result = select_history(plan, rows[:1], 0)
        self.assertIsNone(result['best_sim']); self.assertFalse(result['fixed_budget_reached'])
        result = select_history(plan, rows, 16)
        self.assertEqual(result['best_sim']['update'], 7)

    def test_equal_budget_endpoint_kept_even_when_best_is_earlier(self):
        _, plan = setup_plan(); result = select_history(plan, observations(plan), 16)
        self.assertEqual(result['best_sim']['update'], 7)
        self.assertEqual(result['equal_budget_endpoint']['update'], 16)
        self.assertEqual(result['completed_exposures'], 64)
        self.assertEqual(result['stop_reason'], 'fixed_shared_update_budget')
        self.assertFalse(result['affects_training_schedule'])

    def test_real_and_sim_checkpoint_choices_remain_separate(self):
        _, plan = setup_plan('scorer_patch'); rows = observations(plan)
        rows[2]['report']['real_development'].update(key=[.9, .6, .7], selection_value=.9)
        result = select_history(plan, rows, 16)
        self.assertEqual(result['best_sim']['update'], 7)
        self.assertEqual(result['best_real']['update'], 12)

    def test_lexicographic_ties_keep_earliest_trained_observation(self):
        _, plan = setup_plan(); rows = observations(plan, [.99, .7, .7, .7])
        self.assertEqual(select_history(plan, rows, 16)['best_sim']['update'], 7)
        rows[2]['report']['simulation']['key'][1] = .6
        self.assertEqual(select_history(plan, rows, 16)['best_sim']['update'], 12)

    def test_missing_duplicate_or_future_history_rejected(self):
        _, plan = setup_plan(); rows = observations(plan)
        for changed in (rows[1:], rows + [rows[-1]], rows[:2] + rows[3:], rows[::-1]):
            with self.subTest(rows=changed), self.assertRaisesRegex(ValueError, 'observations'):
                select_history(plan, changed, 16)
        with self.assertRaisesRegex(ValueError, 'budget'):
            select_history(plan, rows, 17)

    def test_plan_or_update_binding_cannot_be_relabelled(self):
        _, plan = setup_plan()
        for field, value in [('common_plan_sha256', digest('other')), ('update', 1),
                             ('module', 'scorer_patch'), ('selection_eligible', False), ('test_used', True)]:
            rows = observations(plan); rows[1]['report'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'binding'):
                select_history(plan, rows, 16)

    def test_missing_artifact_hash_rejected_on_resume_selection(self):
        _, plan = setup_plan(); rows = observations(plan)
        rows[1]['report']['artifact']['sha256'] = ''
        with self.assertRaisesRegex(ValueError, 'artifact'):
            select_history(plan, rows, 16)

    def test_selection_does_not_mutate_saved_observations(self):
        _, plan = setup_plan(); rows = observations(plan); before = copy.deepcopy(rows)
        result = select_history(plan, rows, 16)
        result['best_sim']['report']['key'][0] = 0.
        self.assertEqual(rows, before)

    def test_update_loop_resume_preserves_observations_and_training_rng(self):
        ledger, plan = setup_plan()
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([.2]))
            def forward(self, x):
                loss = (self.weight * x - .5).square().mean() + torch.rand(()) * self.weight.sum()
                return loss, {'fixture': loss.detach()}, {'pairs': len(x)}
        def execute(resume=None, end=None):
            torch.manual_seed(91); model = Toy(); optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
            def simulation():
                value = float(model.weight.detach()[0])
                torch.rand(3)  # Validation must not advance training RNG.
                return report(score=value)[0], ['synthetic-only']
            adapter = ValidationAdapter(plan, Topology(0, 1, 2, 2), simulation, None, writer)
            result = run_updates(model, optimizer, [torch.tensor(float(r.label)) for r in ledger.catalog],
                ledger, 'curriculum', Topology(0, 1, 2, 2), 'cpu', plan.learning_rate_knots,
                plan.validation_updates, resume=resume, binding={'plan': plan.sha256},
                stop_after=end, evaluate=adapter)
            return copy.deepcopy(result)
        full = execute(); partial = execute(end=9); resumed = execute(partial)
        self.assertEqual(tree_sha(full), tree_sha(resumed))
        self.assertEqual(select_history(plan, full['observations'], 16),
                         select_history(plan, resumed['observations'], 16))


class BoundValidationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['CURRICULUM_BASELINE_SOURCE']).resolve()
        cls.evaluation = importlib.import_module(BASE + 's7_consensus_v1.evaluation')
        cls.real_module = importlib.import_module(BASE + 's7_consensus_v1.real_development')
        cls.config = importlib.import_module(BASE + 's7_consensus_v1.config').TrainingConfig()
        if cls.root not in Path(cls.evaluation.__file__).resolve().parents:
            raise ValueError('tests must use the immutable prepared baseline')

    def fixture(self):
        contract = dict(validation_design={'kind': 'single_mixed'}, validation={
            'cal_mixed': {'pair_count': 4}, 'select_mixed': {'pair_count': 4}})
        def rows(prefix, gt=True):
            return [dict(pair_id=prefix + str(i), label=i < 2, gt_known=gt and i < 2,
                score=[.9, .8, .4, .1][i], numeric_valid=True, has_candidate=True,
                layout20=gt and i < 2, candidate_coverage=gt and i < 2,
                recipe='synthetic', matcher_batch_loss=.5) for i in range(4)]
        simulation = {split: {'mixed': rows(split)} for split in ('cal', 'select')}
        real = {ds: {role: rows(ds + role, ds == 'dunhuang_cv')
                for role in ('real_cal', 'real_select')} for ds in ('dunhuang_cv', 'turufan')}
        return contract, simulation, real

    def test_existing_native_matcher_metric_uses_coverage_not_untrained_classification(self):
        contract, rows, _ = self.fixture()
        metric = self.evaluation.summarize_validation(rows, contract, 'matcher', self.config)
        check_report('matcher', metric, None)
        self.assertEqual(metric['key'], [1., 1., -.5])
        _, plan = setup_plan()
        with patch.object(self.evaluation, 'validate', return_value=(metric, rows)) as validate:
            adapter = from_bound_baseline(plan, Topology(0, 1, 4, 1), object(), contract, 'cpu',
                self.config, self.root, writer)
            self.assertEqual(adapter(7)['simulation']['key'], metric['key'])
            self.assertEqual(validate.call_args.args[2], 'matcher')

    def test_existing_real_calibration_and_null_turufan_layout_preserved(self):
        contract, rows, real_rows = self.fixture()
        sim = self.evaluation.summarize_validation(rows, contract, 'scorer', self.config)
        real = self.real_module.select_from_development(real_rows, self.config)
        check_report('scorer_patch', sim, real)
        self.assertIsNone(real['domains']['turufan']['select']['layout20'])
        self.assertIsNone(real['domains']['turufan']['select']['joint_f1'])
        _, plan = setup_plan('scorer_patch')
        real_object = object.__new__(self.real_module.RealDevelopment)
        with patch.object(self.evaluation, 'validate', return_value=(sim, rows)), \
                patch.object(real_object, 'evaluate', return_value=(real, real_rows)) as evaluate:
            adapter = from_bound_baseline(plan, Topology(0, 1, 4, 1), object(), contract, 'cpu',
                self.config, self.root, writer, real_development=real_object)
            self.assertEqual(adapter(7)['real_development']['thresholds'], real['thresholds'])
            self.assertEqual(evaluate.call_count, 1)

    def test_bound_factory_rejects_wrong_source_or_mutable_matcher_cache(self):
        _, plan = setup_plan(); contract, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, 'different baseline'):
            from_bound_baseline(plan, Topology(0, 1, 4, 1), None, contract, 'cpu',
                self.config, self.root / 'wrong-package', writer)
        with self.assertRaisesRegex(ValueError, 'frozen-weight'):
            from_bound_baseline(plan, Topology(0, 1, 4, 1), None, contract, 'cpu',
                self.config, self.root, writer, caches={})


if __name__ == '__main__':
    unittest.main()
