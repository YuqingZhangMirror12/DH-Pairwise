"""Existing native SIM/REAL validation, indexed by shared optimizer updates.

Does not choose a training budget, change LR, use TEST, or launch a process.
Rows are handed to the rank-zero artifact writer; compact canonical reports go
into the committed training state. No rank-local wall time enters that state.
"""
import importlib
import json
import math
from pathlib import Path

from .exposure import digest, integer, is_sha

BASE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'
RULES = {
    'matcher': 'native_coverage_layout_negative_matcher_loss/1',
    'scorer_patch': 'joint_f1_layout_ap+real_domain_joint_pair/1',
    'scorer_stats': 'joint_f1_layout_ap+real_domain_joint_pair/1',
}


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def validate_rule(plan):
    record = plan.record
    if digest(record) != plan.sha256 or record['selection_rule']['id'] != RULES.get(record['module']):
        raise ValueError('declared curriculum selection rule is not implemented or changed')
    expected = {'test_used': False, 'affects_training_schedule': False,
                'include_equal_budget_endpoint': True, 'trained_observations_only': True}
    if any(record['selection_rule'].get(key) is not value for key, value in expected.items()):
        raise ValueError('selection cannot change training or use TEST/untrained checkpoints')


def finite_key(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError('three finite lexicographic validation metrics required')
    return list(value)


def check_report(module, simulation, real):
    stage = 'matcher' if module == 'matcher' else 'scorer'
    if simulation.get('stage') != stage or simulation.get('real_used') is not False:
        raise ValueError('native simulation stage/selection boundary differs')
    finite_key(simulation.get('key'))
    if simulation.get('selection_value') != simulation['key'][0]:
        raise ValueError('simulation selection value differs from its native key')
    if stage == 'matcher':
        if real is not None:
            raise ValueError('Matcher selection uses native SIM coverage, not an untrained Scorer')
    else:
        if (not isinstance(real, dict) or real.get('status') != 'development_selection'
                or real.get('test_used') is not False or real.get('gradients_used') is not False
                or real.get('real_used') is not True or real.get('development_evaluation') is not True):
            raise ValueError('source-isolated, non-backprop REAL development report required')
        finite_key(real.get('key'))
        if real.get('selection_value') != real['key'][0]:
            raise ValueError('real selection value differs from its declared key')
        if set(real.get('thresholds', {})) != {'dunhuang_cv', 'turufan'}:
            raise ValueError('both separately calibrated real domains required')
        for threshold in [simulation.get('threshold'), *real['thresholds'].values()]:
            if type(threshold) not in (int, float) or not .2 <= threshold <= .8:
                raise ValueError('classification threshold outside the registered .20-.80 range')
    canonical({'simulation': simulation, 'real_development': real})


class ValidationAdapter:
    """Evaluation callback for training_core.run_updates.

    simulation() and real() reuse the existing full-view evaluators. They run
    on every rank and return globally gathered rows. write_artifact(update,
    record, rows) runs only on rank0 and must return a bound file receipt.
    Broadcast is injectable for CPU checks; real multi-rank runs use the
    already-initialized process group. This callback does not initialize it.
    """
    def __init__(self, plan, topology, simulation, real, write_artifact,
                 broadcast=None):
        validate_rule(plan)
        if not callable(simulation) or not callable(write_artifact):
            raise ValueError('explicit evaluator and artifact writer required')
        if (plan.record['module'] == 'matcher') != (real is None):
            raise ValueError('only trained Scorer stages perform real score selection')
        if real is not None and not callable(real):
            raise ValueError('real evaluator must be callable')
        if topology.rank < 0 or topology.rank >= topology.world_size:
            raise ValueError('rank outside topology')
        self.plan = plan; self.topology = topology
        self.simulation = simulation; self.real = real
        self.write_artifact = write_artifact; self.broadcast = broadcast

    def _broadcast(self, packet):
        if self.topology.world_size == 1:
            return packet
        if self.broadcast is not None:
            return self.broadcast(packet)
        import torch.distributed as dist
        if (not dist.is_initialized() or dist.get_world_size() != self.topology.world_size
                or dist.get_rank() != self.topology.rank):
            raise ValueError('validation process group differs from registered topology')
        values = [packet]; dist.broadcast_object_list(values, src=0)
        return values[0]

    def __call__(self, completed):
        integer(completed, 'completed update')
        validate_rule(self.plan)
        if completed not in self.plan.validation_updates:
            raise ValueError('evaluation outside the locked common observation schedule')
        sim, sim_rows = self.simulation()
        real, real_rows = (None, None) if self.real is None else self.real()
        packet = None
        if self.topology.rank == 0:
            try:
                check_report(self.plan.record['module'], sim, real)
                record = canonical(dict(schema='curriculum-observation/1', update=completed,
                    module=self.plan.record['module'], common_plan_sha256=self.plan.sha256,
                    simulation=sim, real_development=real,
                    selection_eligible=completed > 0, test_used=False))
                receipt = self.write_artifact(completed, record,
                    {'simulation': sim_rows, 'real_development': real_rows})
                if (not isinstance(receipt, dict) or not receipt.get('path')
                        or not is_sha(receipt.get('sha256'))):
                    raise ValueError('validation rows must have a bound artifact receipt')
                record['artifact'] = canonical(receipt)
                packet = dict(ok=True, observation=record)
            except Exception as error:
                # Communicate a writer/check failure before ranks try to commit
                # a different shared state or wait at the next collective.
                packet = dict(ok=False, error_type=type(error).__name__, message=str(error))
        packet = self._broadcast(packet)
        if not packet['ok']:
            raise ValueError('rank-zero validation failed: ' + packet['error_type'] + ': ' + packet['message'])
        without_elapsed = lambda report: (None if report is None else canonical(
            {key: value for key, value in report.items() if key != 'elapsed_seconds'}))
        if (without_elapsed(sim) != without_elapsed(packet['observation']['simulation'])
                or without_elapsed(real) != without_elapsed(packet['observation']['real_development'])):
            raise ValueError('ranks disagree on validation metrics, not just wall time')
        return packet['observation']


def from_bound_baseline(plan, topology, model, contract, device, config,
                        source_root, write_artifact, real_development=None, caches=None):
    """Reuse the immutable full SIM and source-isolated REAL implementations."""
    root = Path(source_root).resolve()
    evaluation = importlib.import_module(BASE + 's7_consensus_v1.evaluation')
    if root not in Path(evaluation.__file__).resolve().parents:
        raise ValueError('validation imported a different baseline source')
    stage = 'matcher' if plan.record['module'] == 'matcher' else 'scorer'
    if stage == 'matcher' and real_development is not None:
        raise ValueError('Matcher selection cannot call an untrained real Scorer')
    if stage == 'matcher' and caches is not None:
        raise ValueError('updating Matcher cannot reuse a frozen-weight candidate cache')
    if stage == 'scorer':
        real_module = importlib.import_module(BASE + 's7_consensus_v1.real_development')
        if (root not in Path(real_module.__file__).resolve().parents
                or not isinstance(real_development, real_module.RealDevelopment)):
            raise ValueError('Scorer needs the bound source-isolated real evaluator')
    return ValidationAdapter(plan, topology,
        lambda: evaluation.validate(model, contract, stage, device, config, caches=caches),
        None if stage == 'matcher' else lambda: real_development.evaluate(model, device, config),
        write_artifact)


def select_history(plan, observations, completed):
    """Replay selection; never change budget/LR and always retain the endpoint."""
    validate_rule(plan); integer(completed, 'completed update')
    if completed > plan.record['total_updates']:
        raise ValueError('observation history exceeds the fixed budget')
    expected = [update for update in plan.validation_updates if update <= completed]
    if [row['update'] for row in observations] != expected:
        raise ValueError('missing, duplicated or out-of-order validation observations')
    for row in observations:
        report = row['report']
        if (report.get('schema') != 'curriculum-observation/1'
                or report.get('update') != row['update']
                or report.get('common_plan_sha256') != plan.sha256
                or report.get('module') != plan.record['module']
                or report.get('selection_eligible') is not (row['update'] > 0)
                or report.get('test_used') is not False
                or not is_sha(report.get('artifact', {}).get('sha256'))):
            raise ValueError('observation binding, eligibility or artifact changed')
        check_report(plan.record['module'], report['simulation'], report['real_development'])
    trained = [row for row in observations if row['update'] > 0]

    def best(key):
        if not trained:
            return None
        # Python max preserves the earliest observation for exact key ties,
        # matching the existing strictly-greater checkpoint replacement rule.
        row = max(trained, key=lambda value: tuple(value['report'][key]['key']))
        return dict(update=row['update'], report=canonical(row['report'][key]),
                    observation_artifact=canonical(row['report']['artifact']))

    complete = completed == plan.record['total_updates']
    return dict(schema='curriculum-selection-progress/1', module=plan.record['module'],
        common_plan_sha256=plan.sha256, completed_updates=completed,
        completed_exposures=completed * plan.record['effective_batch'],
        fixed_budget_reached=complete, best_sim=best('simulation'),
        best_real=None if plan.record['module'] == 'matcher' else best('real_development'),
        equal_budget_endpoint=canonical(observations[-1]) if complete else None,
        test_used=False, affects_training_schedule=False,
        stop_reason='fixed_shared_update_budget' if complete else None)
