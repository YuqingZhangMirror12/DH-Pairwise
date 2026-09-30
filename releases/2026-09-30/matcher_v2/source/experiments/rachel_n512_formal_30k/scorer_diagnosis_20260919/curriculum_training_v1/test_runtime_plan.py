import copy
import unittest

from .exposure import STAGES, digest
from .runtime_plan import check_light_heads, check_matched_matchers, experiment_binding, lock_record
from .test_exposure import ledger


def fixture():
    value = ledger()
    return value, dict(schema='curriculum-runtime-plan/1', protocol_locked=True, module='matcher',
        ledger_sha256=value.sha256, total_updates=value.total_updates,
        stage_updates=dict(zip(STAGES, value.stage_updates)), effective_batch=value.effective_batch,
        seed=value.seed, model_seed=26092407, head_seed=26092406,
        learning_rate_knots=[[0, .0001], [12, .00005]], validation_updates=[0, 7, 12, 16],
        weight_decay=.0001, gradient_clip_norm=5., checkpoint_every_updates=4, precision='float32',
        termination='fixed_shared_update_budget', optimizer_reset_at_stage=False,
        selection_rule={'id':'synthetic-fixture-only', 'test_used':False, 'affects_training_schedule':False,
                        'include_equal_budget_endpoint':True, 'trained_observations_only':True},
        data_admission_sha256=digest('synthetic-data'), geometry_sha256=digest('synthetic-geometry'),
        baseline_sources_sha256=digest('synthetic-source'))


class RuntimePlanTests(unittest.TestCase):
    def test_same_common_plan_in_both_orders(self):
        value, record = fixture(); plan = lock_record(record, value)
        result = check_matched_matchers(experiment_binding(plan, 'curriculum'), experiment_binding(plan, 'mixed'))
        self.assertEqual(result['status'], 'matched')
        self.assertFalse(result['training_started'])

    def test_unlocked_or_incomplete_plan_not_accepted(self):
        for field in ('protocol_locked', 'stage_updates', 'geometry_sha256'):
            value, record = fixture()
            if field == 'protocol_locked':
                record[field] = False
            else:
                record.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                lock_record(record, value)

    def test_stage_or_total_budget_cannot_disagree_with_ledger(self):
        for field in ('total_updates', 'stage_updates', 'ledger_sha256', 'seed'):
            value, record = fixture()
            if field == 'stage_updates':
                record[field]['v18'] += 1
            elif field == 'ledger_sha256':
                record[field] = digest('other-ledger')
            else:
                record[field] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'exposure ledger'):
                lock_record(record, value)

    def test_lr_schedule_is_explicit_common_and_in_budget(self):
        for knots in ([], [[1, .0001]], [[0, .0001], [16, .00005]], [[0, 0.]]):
            value, record = fixture(); record['learning_rate_knots'] = knots
            with self.subTest(knots=knots), self.assertRaises(ValueError):
                lock_record(record, value)

    def test_boundaries_and_endpoint_observed_in_both_orders(self):
        for marks in ([0, 7, 16], [0, 7, 12], [0, 12, 7, 16], [7, 12, 16], [0, 7, 12, 16, 17]):
            value, record = fixture(); record['validation_updates'] = marks
            with self.subTest(marks=marks), self.assertRaises(ValueError):
                lock_record(record, value)

    def test_independent_early_stop_or_optimizer_reset_rejected(self):
        for field, changed in [('termination', 'independent_plateau'), ('optimizer_reset_at_stage', True),
                               ('precision', 'float16')]:
            value, record = fixture(); record[field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                lock_record(record, value)

    def test_selection_cannot_use_test_or_change_budget(self):
        for field, changed in [('test_used', True), ('affects_training_schedule', True),
                               ('include_equal_budget_endpoint', False), ('trained_observations_only', False)]:
            value, record = fixture(); record['selection_rule'][field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                lock_record(record, value)

    def test_each_light_head_can_have_curriculum_but_no_mixed_addition(self):
        for name in ('scorer_patch', 'scorer_stats'):
            value, record = fixture(); record['module'] = name; plan = lock_record(record, value)
            self.assertEqual(experiment_binding(plan, 'curriculum', digest('same-matcher'))['module'], name)
            with self.assertRaises(ValueError):
                experiment_binding(plan, 'mixed', digest('same-matcher'))

    def test_two_matchers_cannot_use_different_common_learning_rates(self):
        value, record = fixture(); plan = lock_record(record, value)
        a = experiment_binding(plan, 'curriculum'); b = copy.deepcopy(experiment_binding(plan, 'mixed'))
        b['common_plan']['learning_rate_knots'][0][1] *= .5
        with self.assertRaisesRegex(ValueError, 'more than'):
            check_matched_matchers(a, b)

    def test_lock_copies_record_instead_of_keeping_mutable_caller_reference(self):
        value, record = fixture(); plan = lock_record(record, value)
        before = plan.sha256; record['learning_rate_knots'][0][1] = 1.
        self.assertEqual(digest(plan.record), before)

    def test_head_binding_must_name_a_frozen_matcher_and_random_stage_cannot_import(self):
        value, record = fixture(); plan = lock_record(record, value)
        with self.assertRaises(ValueError):
            experiment_binding(plan, 'curriculum', digest('old-E32'))
        record['module'] = 'scorer_patch'; plan = lock_record(record, value)
        with self.assertRaises(ValueError):
            experiment_binding(plan, 'curriculum')

    def test_two_light_heads_freeze_same_matcher_and_share_full_curriculum(self):
        value, record = fixture(); rows = []
        for name in ('scorer_patch', 'scorer_stats'):
            record['module'] = name
            rows.append(experiment_binding(lock_record(record, value), 'curriculum', digest('same-selected')))
        self.assertEqual(check_light_heads(*rows)['status'], 'matched')
        rows[1]['fixed_matcher_sha256'] = digest('different')
        with self.assertRaisesRegex(ValueError, 'same selected'):
            check_light_heads(*rows)


if __name__ == '__main__':
    unittest.main()
