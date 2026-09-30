"""New-arm selection policy: native SIM plus Dunhuang development only.

Turufan is not opened, evaluated or used for threshold/epoch selection here.
The prior two-domain policy is deliberately left unchanged in its old package.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from ..curriculum_training_v1.checkpoint_io import file_sha
from ..curriculum_training_v1.exposure import digest, integer, is_sha
from ..curriculum_training_v1.model_adapter import BASE, bound_module, require
from ..curriculum_training_v1.validation_adapter import canonical, finite_key

RULES = {'matcher': 'native_coverage_layout_negative_matcher_loss/1',
         'scorer_patch': 'joint_f1_layout_ap+dunhuang_joint_only/1',
         'scorer_stats': 'joint_f1_layout_ap+dunhuang_joint_only/1'}
ROLES = ('real_cal', 'real_select')
FOLDS = {'real_cal': [1], 'real_select': [2, 3, 4], 'real_test': [0]}


def validate_rule(plan):
    require(digest(plan.record) == plan.sha256
            and plan.record['selection_rule'].get('id') == RULES.get(plan.record['module']),
            'new-arm selection must use native SIM / Dunhuang-only REAL')
    expected = {'test_used': False, 'affects_training_schedule': False,
                'include_equal_budget_endpoint': True, 'trained_observations_only': True}
    require(all(plan.record['selection_rule'].get(k) is v for k, v in expected.items()),
            'selection cannot change training or use TEST/untrained checkpoints')


def select_dunhuang(rows, config, metric):
    require(set(rows) == {'dunhuang_cv'} and set(rows['dunhuang_cv']) == set(ROLES),
            'only Dunhuang CAL/SELECT is permitted; Turufan and TEST are excluded')
    cal, select = (rows['dunhuang_cv'][r] for r in ROLES)
    ids = [{r['pair_id'] for r in part} for part in (cal, select)]
    require(len(ids[0]) == len(cal) and len(ids[1]) == len(select) and not ids[0] & ids[1],
            'duplicated/overlapping real development pairs')
    for part in (cal, select):
        require(any(r['label'] for r in part) and not all(r['label'] for r in part),
                'CAL and SELECT each need both classes')
        require(all(not r['label'] or r['gt_known'] for r in part), 'Dunhuang positive layout GT required')
    choices = []
    for at in range(20, 81):
        threshold = at / 100; result = metric(cal, threshold)
        choices.append(((result['joint_f1'], -abs(threshold-config.threshold_tie_preference),
                         result['joint_precision'], threshold), threshold, result))
    _, threshold, calibration = max(choices, key=lambda item: item[0])
    selected = metric(select, threshold)
    key = [selected['joint_f1'], selected['layout20'], selected['f1']]
    return canonical(dict(status='development_selection', thresholds={'dunhuang_cv': threshold},
        key=key, selection_value=key[0], domains={'dunhuang_cv': dict(threshold=threshold,
            calibration=calibration, select=selected)}, test_used=False, turufan_used=False,
        real_used=True, development_evaluation=True, gradients_used=False))


def bind_dunhuang(path):
    """Bind only Dunhuang files; a Turufan path may be absent/inaccessible."""
    path = Path(path); plan = json.loads(path.read_text())
    require(plan.get('schema') == 'threshold-joint-real-split/1' and plan.get('source_disjoint') is True
            and plan['role_folds'] == FOLDS, 'registered source-disjoint real split required')
    spec = plan['datasets']['dunhuang_cv']; manifest = Path(spec['remote_manifest'])
    require(file_sha(manifest) == spec['manifest_sha256'], 'Dunhuang source manifest changed')
    meta = json.loads(manifest.read_text()); excluded = set(spec['excluded_gt_pair_ids'])
    seen = set(); fragments = {}; groups = {}
    for role, folds in FOLDS.items():
        expected = [r for r in meta['pairs'] if r['fold'] in folds and r['pair_id'] not in excluded]
        ids = [r['pair_id'] for r in expected]
        require(ids == spec['roles'][role]['pair_ids'] and len(ids) == len(set(ids))
                and not seen & set(ids), 'Dunhuang role identities overlap or changed')
        seen.update(ids)
        fragments[role] = {r[k] for r in expected for k in ('fragment_a_id', 'fragment_b_id')}
        groups[role] = {meta['fragment_source_group'][f] for f in fragments[role]}
        for previous in fragments:
            require(previous == role or not (fragments[role] & fragments[previous] or groups[role] & groups[previous]),
                    'Dunhuang source/fragment leakage')
    require(seen == {r['pair_id'] for r in meta['pairs'] if r['pair_id'] not in excluded}, 'missing Dunhuang role rows')
    return dict(schema='matcher-v2-dunhuang-development-binding/1', plan_sha256=file_sha(path),
        gt_sha256=file_sha(plan['gt_path']), manifest_sha256=file_sha(manifest),
        inputs_sha256=file_sha(Path(spec['prepared']) / 'inputs.npz'),
        inference_roles=list(ROLES), turufan_opened=False, test_inferred=False)


class DunhuangDevelopment:
    def __init__(self, path, binding, source_root):
        require(bind_dunhuang(path) == binding, 'Dunhuang development binding changed')
        self.plan = json.loads(Path(path).read_text()); self.binding = canonical(binding)
        self.spec = self.plan['datasets']['dunhuang_cv']
        self.meta = json.loads(Path(self.spec['remote_manifest']).read_text())
        by_id = {r['pair_id']: r for r in self.meta['pairs']}
        self.pairs = {role: [by_id[i] for i in self.spec['roles'][role]['pair_ids']] for role in ROLES}
        fragment_ids = sorted({r[k] for values in self.pairs.values() for r in values
                               for k in ('fragment_a_id', 'fragment_b_id')})
        lookup = {f: i for i, f in enumerate(self.meta['fragment_ids'])}
        indices = [lookup[f] for f in fragment_ids]
        with np.load(Path(self.spec['prepared']) / 'inputs.npz', allow_pickle=False) as z:
            n = len(self.meta['fragment_ids'])
            require(z['packed_masks'].shape == (n, 800, 100) and z['points'].shape == (n, 512, 2)
                    and z['valid'].shape == (n, 512), 'real preprocessing shapes changed')
            self.arrays = {k: z[k][indices] for k in ('packed_masks', 'points', 'valid')}
        self.lookup = {f: i for i, f in enumerate(fragment_ids)}
        positives = {r['pair_id'] for values in self.pairs.values() for r in values if r['label']}
        self.gt = {r['pair_id']: r for r in json.loads(Path(self.plan['gt_path']).read_text())['positive_pairs']
                   if r['pair_id'] in positives}
        require(set(self.gt) == positives, 'Dunhuang development positive GT missing')
        self.evidence = bound_module(BASE + 's7_consensus_v1.evidence', source_root)
        self.matcher_api = bound_module(BASE + 's7_consensus_v1.matcher', source_root)
        self.metric = bound_module(BASE + 's7_consensus_v1.metrics', source_root).summarize

    @torch.no_grad()
    def evaluate(self, model, device, config):
        started = time.monotonic(); model.eval(); data = {'dunhuang_cv': {}}
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        require(1 <= config.microbatch <= 8, 'registered validation microbatch is at most eight')
        for role in ROLES:
            pairs = self.pairs[role]; local = pairs[rank::world]; rows = []
            for offset in range(0, len(local), config.microbatch):
                items = local[offset:offset+config.microbatch]; batch = {}
                for side in 'ab':
                    ix = [self.lookup[r['fragment_'+side+'_id']] for r in items]
                    batch['mask_'+side] = torch.from_numpy(np.unpackbits(self.arrays['packed_masks'][ix], axis=-1)
                                                         .astype(np.float32)[:, None]).to(device)
                    batch['points_rc_'+side] = torch.from_numpy(self.arrays['points'][ix].astype(np.float32)).to(device)
                    batch['contour_valid_'+side] = torch.from_numpy(self.arrays['valid'][ix].astype(bool)).to(device)
                output = model.matcher(*(batch[k] for k in self.matcher_api.INPUTS))
                for i, item in enumerate(items):
                    pair = self.evidence.PairEvidence.from_matcher(output, i, batch['mask_a'], batch['mask_b'])
                    pred = model.score_pair(pair)  # GT is joined only after scoring.
                    target = None
                    if item['label']:
                        gt = self.gt[item['pair_id']]
                        require((gt['fragment_a_token'], gt['fragment_b_token']) ==
                                (item['fragment_a_id'], item['fragment_b_id']), 'GT endpoint order differs')
                        target = pair.q.new_tensor(gt['translation_gt_a_to_b_rc'])
                    error = float((pred.translation_a_to_b_rc-target).norm()) if target is not None and pred.has_candidate else None
                    rows.append(dict(pair_id=item['pair_id'], label=bool(item['label']), gt_known=target is not None,
                        score=float(pred.score), has_candidate=pred.has_candidate, numeric_valid=pred.numeric_valid,
                        translation=None if not pred.has_candidate else pred.translation_a_to_b_rc.cpu().tolist(),
                        layout20=bool(error is not None and error <= 20), error_px=error,
                        candidate_coverage=bool(target is not None and any(float((c.translation-target).norm()) <= 20
                                                                            for c in pred.clusters))))
            if world > 1:
                gathered = [None] * world; dist.all_gather_object(gathered, rows)
                rows = [r for part in gathered for r in part]
            require(len(rows) == len(pairs) and {r['pair_id'] for r in rows} == {r['pair_id'] for r in pairs},
                    'missing/duplicated Dunhuang predictions')
            data['dunhuang_cv'][role] = sorted(rows, key=lambda r: r['pair_id'])
        result = select_dunhuang(data, config, self.metric)
        result['elapsed_seconds'] = time.monotonic()-started
        return result, data


def check_report(module, simulation, real):
    matcher = module == 'matcher'
    require(simulation.get('stage') == ('matcher' if matcher else 'scorer')
            and simulation.get('real_used') is False, 'native simulation selection boundary differs')
    finite_key(simulation.get('key'))
    require(simulation.get('selection_value') == simulation['key'][0], 'simulation native key differs')
    if matcher:
        require(real is None, 'Matcher selection cannot use real data or an untrained Scorer')
        return
    require(isinstance(real, dict) and real.get('status') == 'development_selection'
            and real.get('test_used') is False and real.get('turufan_used') is False
            and real.get('gradients_used') is False and real.get('real_used') is True
            and real.get('development_evaluation') is True, 'Dunhuang-only non-backprop development report required')
    finite_key(real.get('key'))
    require(real.get('selection_value') == real['key'][0]
            and set(real.get('thresholds', {})) == {'dunhuang_cv'}
            and set(real.get('domains', {})) == {'dunhuang_cv'}, 'Turufan/other domain entered development selection')
    for threshold in (simulation.get('threshold'), real['thresholds']['dunhuang_cv']):
        require(type(threshold) in (int, float) and .2 <= threshold <= .8, 'threshold outside .20-.80')
    canonical({'simulation': simulation, 'real_development': real})


class ValidationAdapter:
    def __init__(self, plan, topology, simulation, real, write_artifact, broadcast=None):
        validate_rule(plan)
        require(callable(simulation) and callable(write_artifact), 'explicit evaluator and writer required')
        require((real is None) == (plan.record['module'] == 'matcher') and (real is None or callable(real)),
                'only trained Scorer stages select on Dunhuang')
        self.plan, self.topology = plan, topology
        self.simulation, self.real, self.writer, self.broadcast = simulation, real, write_artifact, broadcast

    def __call__(self, completed):
        validate_rule(self.plan); integer(completed, 'completed update')
        require(completed in self.plan.validation_updates, 'evaluation outside registered schedule')
        sim, sim_rows = self.simulation()
        real, real_rows = (None, None) if self.real is None else self.real()
        packet = None
        if self.topology.rank == 0:
            try:
                check_report(self.plan.record['module'], sim, real)
                record = canonical(dict(schema='curriculum-observation/1', update=completed,
                    module=self.plan.record['module'], common_plan_sha256=self.plan.sha256,
                    simulation=sim, real_development=real, selection_eligible=completed > 0, test_used=False))
                receipt = self.writer(completed, record, {'simulation': sim_rows, 'real_development': real_rows})
                require(receipt.get('path') and is_sha(receipt.get('sha256')), 'bound observation artifact required')
                record['artifact'] = canonical(receipt); packet = dict(ok=True, observation=record)
            except Exception as error:
                packet = dict(ok=False, error_type=type(error).__name__, message=str(error))
        if self.topology.world_size > 1:
            if self.broadcast:
                packet = self.broadcast(packet)
            else:
                require(dist.is_initialized() and dist.get_world_size() == self.topology.world_size
                        and dist.get_rank() == self.topology.rank, 'validation topology differs')
                values = [packet]; dist.broadcast_object_list(values, src=0); packet = values[0]
        require(packet['ok'], 'rank-zero validation failed: '+str(packet.get('message')))
        clean = lambda row: None if row is None else canonical({k: v for k, v in row.items() if k != 'elapsed_seconds'})
        require(clean(sim) == clean(packet['observation']['simulation'])
                and clean(real) == clean(packet['observation']['real_development']), 'rank validation metrics disagree')
        return packet['observation']


def from_bound_baseline(plan, topology, model, contract, device, config,
                        source_root, write_artifact, real_development=None, caches=None):
    """Keep the bound native SIM evaluator; substitute only the new REAL policy."""
    validate_rule(plan)
    evaluation = bound_module(BASE + 's7_consensus_v1.evaluation', source_root)
    stage = 'matcher' if plan.record['module'] == 'matcher' else 'scorer'
    require(1 <= config.microbatch <= 8, 'validation microbatch must stay at most eight')
    if stage == 'matcher':
        require(real_development is None and caches is None,
                'Matcher SIM selection cannot use real Scorer or frozen candidate cache')
    else:
        require(isinstance(real_development, DunhuangDevelopment), 'bound Dunhuang-only evaluator required')
    return ValidationAdapter(plan, topology,
        lambda: evaluation.validate(model, contract, stage, device, config, caches=caches),
        None if stage == 'matcher' else lambda: real_development.evaluate(model, device, config),
        write_artifact)


def select_history(plan, observations, completed):
    validate_rule(plan); integer(completed, 'completed update')
    require(completed <= plan.record['total_updates'], 'history exceeds the locked budget')
    require([r['update'] for r in observations] == [i for i in plan.validation_updates if i <= completed],
            'missing/duplicated/out-of-order observations')
    for row in observations:
        report = row['report']
        require(report.get('schema') == 'curriculum-observation/1' and report.get('update') == row['update']
                and report.get('module') == plan.record['module'] and report.get('common_plan_sha256') == plan.sha256
                and report.get('selection_eligible') is (row['update'] > 0) and report.get('test_used') is False
                and is_sha(report.get('artifact', {}).get('sha256')), 'observation identity differs')
        check_report(plan.record['module'], report['simulation'], report['real_development'])
    trained = [row for row in observations if row['update'] > 0]
    def best(key):
        if not trained:return None
        row = max(trained, key=lambda value: tuple(value['report'][key]['key']))
        return dict(update=row['update'], report=canonical(row['report'][key]),
                    observation_artifact=canonical(row['report']['artifact']))
    complete = completed == plan.record['total_updates']
    return dict(schema='curriculum-selection-progress/1', module=plan.record['module'],
        common_plan_sha256=plan.sha256, completed_updates=completed,
        completed_exposures=completed*plan.record['effective_batch'], fixed_budget_reached=complete,
        best_sim=best('simulation'), best_real=None if plan.record['module'] == 'matcher' else best('real_development'),
        equal_budget_endpoint=canonical(observations[-1]) if complete else None,
        test_used=False, turufan_used=False, affects_training_schedule=False,
        stop_reason='fixed_shared_update_budget' if complete else None)
