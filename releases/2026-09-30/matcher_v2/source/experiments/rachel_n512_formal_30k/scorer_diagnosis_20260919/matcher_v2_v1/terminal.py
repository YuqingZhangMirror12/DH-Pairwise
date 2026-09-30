"""Strict completed-run loading for native Matcher and lightweight heads."""
from pathlib import Path

import torch

from ..curriculum_training_v1.checkpoint_io import file_sha, tree_sha
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.model_adapter import BASE, bound_module, require
from ..curriculum_training_v1.runtime_io import read, component_state, check_observations
from .launcher import verify_formal
from .model_runtime import check_export, load_export_matcher, load_export_scorer
from .validation import check_report


def thresholds_for_export(saved):
    require(saved['module'] in ('scorer_patch', 'scorer_stats'), 'a trained Scorer is required')
    observation = saved['observation']; choice = saved['selection_kind']
    require(choice in ('sim_best', 'real_best', 'equal_budget_endpoint')
            and observation['update'] == saved['updates'] > 0 and observation['selection_eligible'] is True
            and observation['test_used'] is False, 'trained selected observation required')
    sim = observation['simulation']; real = observation['real_development']
    check_report(saved['module'], sim, real)
    thresholds = dict(sim_test=sim['threshold'], dunhuang_cv=sim['threshold'], turufan=sim['threshold'])
    origins = dict.fromkeys(thresholds, 'SIM-CAL at the frozen selected update; no target-domain fitting')
    if choice == 'real_best':
        thresholds['dunhuang_cv'] = real['thresholds']['dunhuang_cv']
        origins['dunhuang_cv'] = 'Dunhuang CAL at the Dunhuang SELECT-selected update'
    return thresholds, origins


def verified_export(controller_root, spec_path, plan, selection_kind):
    root = Path(controller_root).resolve(); spec_path = Path(spec_path).resolve()
    matcher = plan.record['module'] == 'matcher'
    require(selection_kind in (('sim_best', 'equal_budget_endpoint') if matcher else
                              ('sim_best', 'real_best', 'equal_budget_endpoint')), 'unregistered selected export')
    for name in ('controller_failure.json', 'failure.json', 'formal/failure.json', 'formal/exports/failure.json'):
        require(not (root/name).exists(), 'failure precedes stale completion')
    result = verify_formal(root/'formal', spec_path, plan)
    returned = read(root/'formal_return.json'); launch = read(root/'formal_launch.json')
    require(returned['phase'] == launch['phase'] == 'formal' and returned['returncode'] == 0
            and returned['launch_sha256'] == file_sha(root/'formal_launch.json'), 'successful actual formal return required')
    command = launch['command']
    require(__package__+'.execution' in command and '--order' not in command, 'not the dedicated v2 execution command')
    for flag, expected in (('--spec', str(spec_path)), ('--mode', 'formal'), ('--out', str(root/'formal'))):
        require(command.count(flag) == 1 and command[command.index(flag)+1] == expected, 'formal command identity differs')
    receipt = read(root/'export_process_return.json'); controller = read(root/'controller_complete.json')
    require(receipt == result and controller == dict(result,
        successful_return_sha256=file_sha(root/'export_process_return.json'),
        formal_return_sha256=file_sha(root/'formal_return.json'), gpu_gate_sha256=file_sha(root/'gpu_gate.json')),
        'controller/export/return identity differs')
    exports = Path(result['export_root']); complete = read(exports/'training_complete.json')
    selection = read(exports/'selection.json'); binding = complete['binding']
    gate = read(root/'gpu_gate.json')
    require(gate['status'] == 'passed' and gate['formal_binding_sha256'] == digest(
        {k: v for k, v in binding.items() if k != 'run_mode'}), 'gate/formal binding differs')
    chosen = complete['exports'][selection_kind]
    saved = torch.load(chosen['path'], map_location='cpu', weights_only=False); check_export(saved)
    expected = plan.record['total_updates'] if selection_kind == 'equal_budget_endpoint' else selection[
        'best_real' if selection_kind == 'real_best' else 'best_sim']['update']
    require(saved['binding'] == binding and saved['selection_kind'] == selection_kind and saved['updates'] == expected
            and saved['total_completed_updates'] == plan.record['total_updates']
            and saved['exposures'] == expected*plan.record['effective_batch']
            and saved['optimizer_imported'] is False and saved['training_rng_included'] is False,
            'selected update/budget/architecture differs')
    check_observations([dict(update=saved['updates'], report=saved['observation'])], binding)
    frozen = 'head.' if matcher else 'matcher.'
    require(tree_sha(component_state(saved['model'], frozen)) == binding['model_spec'][
        'initial_head_state_sha256' if matcher else 'initial_matcher_state_sha256'], 'inactive module changed')
    common = dict(schema='matcher-v2-terminal-origin/1', arm=binding['matcher_v2_experiment']['arm'],
        module=saved['module'], order='curriculum', selection_kind=selection_kind,
        updates=expected, selected_updates=expected, selected_epoch=None,
        total_completed_updates=plan.record['total_updates'], common_plan_sha256=plan.sha256,
        checkpoint=str(Path(chosen['path']).resolve()), checkpoint_sha256=chosen['sha256'], selected_file_sha256=chosen['sha256'],
        model_state_sha256=chosen['model_state_sha256'], matcher_state_sha256=tree_sha(component_state(saved['model'], 'matcher.')),
        controller_complete_sha256=file_sha(root/'controller_complete.json'), selection_sha256=file_sha(exports/'selection.json'),
        stop_reason=complete['stop_reason'], claimed_converged=False, selection_on_test=False,
        turufan_used_for_selection=False, threshold_refitted=False, scorer_used=not matcher,
        real_used_for_selection=not matcher and selection_kind == 'real_best',
        development_evaluation=True, historical_real_development_exposure=True,
        matcher_implementation=binding['model_spec']['matcher_implementation'])
    if not matcher:
        thresholds, origins = thresholds_for_export(saved)
        common.update(variant='binary_'+saved['module'][len('scorer_'):], thresholds=thresholds,
            threshold_origins=origins, selection_on_real=selection_kind == 'real_best',
            matcher_updated_during_training=False, scorer_attention_present=False,
            local_conflict_head_present=False, learned_refinement=False)
    return saved, common


def load_model(saved, origin, source_root):
    binding, spec = check_export(saved)
    require(origin['schema'] == 'matcher-v2-terminal-origin/1' and origin['module'] == saved['module']
            and origin['arm'] == binding['matcher_v2_experiment']['arm']
            and origin['common_plan_sha256'] == digest(binding['common_plan'])
            and origin['model_state_sha256'] == tree_sha(saved['model']), 'verified export identity changed')
    matcher = saved['module'] == 'matcher'
    if not matcher:
        choice = spec['selected_matcher']; require(file_sha(choice['path']) == choice['sha256'], 'selected Matcher source changed')
        selected = torch.load(choice['path'], map_location='cpu', weights_only=False); check_export(selected)
        require(selected['module'] == 'matcher' and selected['selection_kind'] == 'sim_best'
                and selected['binding'] == choice['source_binding']
                and selected['binding']['matcher_v2_experiment']['arm'] == origin['arm']
                and tree_sha(component_state(selected['model'], 'matcher.')) == origin['matcher_state_sha256']
                == spec['initial_matcher_state_sha256'], 'head no longer contains its own frozen SIM-selected Matcher')
    model = (load_export_matcher if matcher else load_export_scorer)(saved, source_root)
    geometry = bound_module(BASE+'s7_consensus_v1.compatibility', source_root).CompatibilityConfig(**spec['geometry'])
    return model, geometry, dict(origin, architecture=spec['architecture'], geometry=spec['geometry'],
        proposal_revision=spec['proposal_revision'], optimizer_imported=False, old_head_imported=False,
        evidence_mode='exact_union_q_then_q_arc_pool', baseline_sources_sha256=binding['common_plan']['baseline_sources_sha256'])
