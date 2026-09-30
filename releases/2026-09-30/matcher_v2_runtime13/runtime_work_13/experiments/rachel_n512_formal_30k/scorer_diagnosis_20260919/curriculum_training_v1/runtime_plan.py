"""Validate an explicitly locked schedule without choosing budget or starting GPU.

The data-admission/selection/export/launcher adapters are separate. In particular,
the current unlocked experiment_plan.json is deliberately NOT accepted here.
"""
from dataclasses import dataclass

from .exposure import STAGES, digest, integer, is_sha, learning_rate_at


@dataclass(frozen=True)
class RuntimePlan:
    record: dict
    sha256: str

    @property
    def validation_updates(self):
        return tuple(self.record['validation_updates'])

    @property
    def learning_rate_knots(self):
        return tuple(tuple(row) for row in self.record['learning_rate_knots'])


def lock_record(record, ledger):
    """Require all choices to be supplied; do not infer formal defaults."""
    fields = {'schema', 'protocol_locked', 'module', 'ledger_sha256', 'total_updates',
              'stage_updates', 'effective_batch', 'seed', 'model_seed', 'head_seed',
              'learning_rate_knots', 'validation_updates', 'weight_decay',
              'gradient_clip_norm', 'checkpoint_every_updates', 'precision',
              'termination', 'optimizer_reset_at_stage', 'selection_rule',
              'data_admission_sha256', 'geometry_sha256', 'baseline_sources_sha256'}
    if set(record) != fields or record['schema'] != 'curriculum-runtime-plan/1' or record['protocol_locked'] is not True:
        raise ValueError('complete explicitly locked runtime plan required')
    if record['module'] not in ('matcher', 'scorer_patch', 'scorer_stats'):
        raise ValueError('unregistered curriculum module')
    if (record['ledger_sha256'] != ledger.sha256
            or record['total_updates'] != ledger.total_updates
            or record['stage_updates'] != dict(zip(STAGES, ledger.stage_updates))
            or record['effective_batch'] != ledger.effective_batch or record['seed'] != ledger.seed):
        raise ValueError('runtime schedule does not match the audited exposure ledger')
    for name in ('model_seed', 'head_seed', 'checkpoint_every_updates'):
        integer(record[name], name, 1)
    for name in ('data_admission_sha256', 'geometry_sha256', 'baseline_sources_sha256'):
        if not is_sha(record[name]):
            raise ValueError('missing bound artifact: ' + name)
    if (record['precision'] != 'float32' or record['termination'] != 'fixed_shared_update_budget'
            or record['optimizer_reset_at_stage'] is not False):
        raise ValueError('no independent early stopping, precision change, or stage optimizer reset')
    import math
    for name, allow_zero in (('weight_decay', True), ('gradient_clip_norm', False)):
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError('finite numeric optimizer setting required')
        if value < 0 or (not allow_zero and value == 0):
            raise ValueError('invalid optimizer setting: ' + name)
    learning_rate_at(0, record['learning_rate_knots'])
    if any(update >= ledger.total_updates for update, _ in record['learning_rate_knots']):
        raise ValueError('LR knot outside actual shared budget')
    marks = record['validation_updates']
    for at in marks:
        integer(at, 'validation update')
    ends = []; end = 0
    for count in ledger.stage_updates:
        end += count; ends.append(end)
    if (marks != sorted(set(marks)) or not marks or marks[-1] != ledger.total_updates
            or not {0, *ends}.issubset(marks)):
        raise ValueError('common observations must include baseline, every stage boundary and final budget')
    # Selection is a declared choice, not an excuse to alter completed updates.
    rule = record['selection_rule']
    if (not isinstance(rule, dict) or not rule.get('id') or rule.get('test_used') is not False
            or rule.get('affects_training_schedule') is not False
            or rule.get('include_equal_budget_endpoint') is not True
            or rule.get('trained_observations_only') is not True):
        raise ValueError('explicit non-TEST selection and equal-budget endpoint required')
    import json
    canonical = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
    return RuntimePlan(canonical, digest(canonical))


def experiment_binding(plan, order, fixed_matcher_sha256=None):
    if order not in ('curriculum', 'mixed') or (order == 'mixed' and plan.record['module'] != 'matcher'):
        raise ValueError('mixed extra Scorer training is not registered')
    matcher_training = plan.record['module'] == 'matcher'
    if ((matcher_training and fixed_matcher_sha256 is not None)
            or (not matcher_training and not is_sha(fixed_matcher_sha256))):
        raise ValueError('new random Matcher or explicitly bound frozen curriculum Matcher required')
    if digest(plan.record) != plan.sha256:
        raise ValueError('locked runtime plan was mutated')
    return dict(schema='curriculum-runtime-binding/1', module=plan.record['module'],
                order=order, common_plan_sha256=plan.sha256, common_plan=plan.record,
                fixed_matcher_sha256=fixed_matcher_sha256, matcher_training=matcher_training,
                head_training=not matcher_training)


def check_matched_matchers(curriculum, mixed):
    if (curriculum.get('module') != 'matcher' or mixed.get('module') != 'matcher'
            or curriculum.get('order') != 'curriculum' or mixed.get('order') != 'mixed'):
        raise ValueError('one curriculum Matcher and one mixed Matcher required')
    without_order = lambda row: {k: v for k, v in row.items() if k != 'order'}
    if without_order(curriculum) != without_order(mixed):
        raise ValueError('paired Matcher settings differ in more than presentation order')
    if digest(curriculum['common_plan']) != curriculum['common_plan_sha256']:
        raise ValueError('common plan hash changed')
    return dict(status='matched', only_designed_variable='global_batch_presentation_order',
                common_plan_sha256=curriculum['common_plan_sha256'],
                total_updates=curriculum['common_plan']['total_updates'], training_started=False)


def check_light_heads(patch, stats):
    if (patch.get('module') != 'scorer_patch' or stats.get('module') != 'scorer_stats'
            or patch.get('order') != 'curriculum' or stats.get('order') != 'curriculum'
            or patch.get('matcher_training') is not False or stats.get('matcher_training') is not False
            or patch.get('head_training') is not True or stats.get('head_training') is not True
            or not is_sha(patch.get('fixed_matcher_sha256'))
            or patch.get('fixed_matcher_sha256') != stats.get('fixed_matcher_sha256')):
        raise ValueError('two curriculum heads must freeze the same selected Matcher')
    for row in (patch, stats):
        if digest(row['common_plan']) != row['common_plan_sha256']:
            raise ValueError('light head plan hash changed')
    clean = lambda row: {k: v for k, v in row['common_plan'].items() if k != 'module'}
    if clean(patch) != clean(stats):
        raise ValueError('light head exposure, schedule or other common settings differ')
    return dict(status='matched', only_designed_variable='lightweight_scorer_architecture',
                fixed_matcher_sha256=patch['fixed_matcher_sha256'], training_started=False)
