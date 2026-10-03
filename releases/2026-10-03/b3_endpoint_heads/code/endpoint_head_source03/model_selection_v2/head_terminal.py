"""New matched-head terminal admission, without legacy Matcher relabelling.

Loading is not evaluation completion. This module performs no inference and
does not choose a TEST threshold. Fresh head exports retain native head keys,
but their Matcher origin remains explicitly mixed_sim_v2_best or the separately
user-designated endpoint; neither is disguised as the old SIM winner.
"""
from pathlib import Path

from . import head_bridge as bridge, head_execution as execution, posthoc_export as ex
from .checkpoint_scan import module
from .protocol import digest, require
from .released_protocol import checked

ORIGIN_SCHEMA = 'mixed-select-matched-head-terminal/1'


def returned_phase(root, phase, expected_module, action, flags, world=1):
    root = Path(root)
    returned = ex.read(root/(phase+'_return.json')); launch = ex.read(root/(phase+'_launch.json'))
    require(returned.get('phase') == launch.get('phase') == phase
            and type(returned.get('returncode')) is int and returned['returncode'] == 0
            and returned.get('launch_sha256') == ex.file_sha(root/(phase+'_launch.json')),
            'actual successful bound process return required')
    command = launch['command']
    prefix = (['-m', expected_module, action] if world == 1 else ['-m', 'torch.distributed.run',
        '--standalone', '--nnodes=1', '--nproc_per_node=2', '--module', expected_module, action])
    require(type(world) is int and world in (1, 2)
            and command[1:len(prefix)+1] == prefix,
            'wrong independent execution command')
    for key, value in flags.items():
        require(command.count(key) == 1 and command[command.index(key)+1] == value,
                'actual command identity differs: '+key)
    require('--resume' not in command and '--order' not in command, 'unregistered terminal training invocation')
    return ex.receipt(root/(phase+'_return.json'))


def verify_head_binding(saved, adoption, contract, runtime):
    """Check the head's frozen Matcher against the actual new typed export."""
    import torch
    binding, spec = module('matcher_v2_v1.model_runtime', runtime).check_export(saved)
    bridge.check_terminal_adoption(adoption)
    require(saved['module'] in execution.HEADS and binding.get('run_mode') == 'formal'
            and spec.get('selected_matcher') == adoption
            and spec.get('initialization') == bridge.initialization_kind(adoption)
            and binding['fixed_matcher_sha256'] == adoption['model']['sha256'],
            'not a fresh head paired with the selected new Matcher')
    expected = dict(schema=bridge.HEAD_SCHEMA, selected_matcher_adoption_sha256=adoption['sha256'],
        validation_contract=contract, validation_contract_sha256=digest(contract),
        original_train_and_labels_preserved=True, task3_overlay_applied=False,
        rotation_ensemble_in_training=False)
    require(binding.get('mixed_sim_head_experiment') == expected,
            'new SELECT/CAL or unchanged-label binding differs')
    require(spec.get('optimizer_imported') is False and spec.get('old_head_imported') is False,
            'old optimizer/head import is not admitted')
    model_ref = adoption['model']
    require(ex.file_sha(model_ref['path']) == model_ref['sha256'], 'selected Matcher source changed')
    selected = torch.load(model_ref['path'], map_location='cpu', weights_only=False)
    hashing = module('curriculum_training_v1.checkpoint_io', runtime)
    bridge.verify_model_record(selected, adoption, hashing.tree_sha)
    component = lambda state: {k[8:]: v for k, v in state.items() if k.startswith('matcher.')}
    require(hashing.tree_sha(component(saved['model'])) == hashing.tree_sha(component(selected['model']))
            == spec['initial_matcher_state_sha256'], 'head training changed its frozen selected Matcher')
    return binding, spec


def verified_export(driver_root, spec_path, selection_kind):
    import torch
    driver_root = Path(driver_root).resolve(strict=True); root = driver_root/'training'
    spec_path = Path(spec_path).resolve(strict=True)
    require(selection_kind in ('sim_best', 'real_best', 'equal_budget_endpoint'), 'registered head export required')
    failures = [p for folder in (driver_root, root, root/'formal')
                for p in folder.glob('*failure*.json')]
    require(not failures, 'failure precedes stale head completion')
    spec, values = execution.load_inputs(spec_path); runtime = spec['runtime']
    driver = ex.read(driver_root/'driver_complete.json')
    require(driver.get('status') == 'training_evaluation_pending'
            and driver.get('execution') == ex.receipt(spec_path)
            and driver.get('controller_complete') == ex.receipt(root/'controller_complete.json'),
            'completed training driver required, not gate-only or an unfinished child')
    actual_controller = returned_phase(driver_root, 'controller', 'model_selection_v2.head_launcher',
        'controller', {'--spec': str(spec_path), '--out': str(root)})
    require(driver['actual_controller_return'] == actual_controller
            and driver['controller_launch'] == ex.receipt(driver_root/'controller_launch.json'),
            'driver actual return ancestry differs')
    actual_formal = returned_phase(root, 'formal', 'model_selection_v2.head_execution', 'train',
        {'--spec': str(spec_path), '--out': str(root/'formal'), '--mode': 'formal',
         '--gate-receipt': str(root/'gpu_gate.json')}, world=spec['topology']['world_size'])
    complete = ex.read(root/'controller_complete.json')
    require(complete.get('schema') == 'mixed-select-head-training-complete/1'
            and complete.get('formal_return') == actual_formal
            and complete.get('selected_matcher_adoption') == spec['selected_matcher_adoption']
            and complete.get('validation_contract') == spec['validation_contract']
            and complete.get('gpu_gate') == ex.receipt(root/'gpu_gate.json')
            and complete.get('task3_overlay_applied') is False, 'head controller terminal ancestry differs')
    native = module('matcher_v2_v1.launcher', runtime)
    result = native.verify_formal(root/'formal', spec_path, values['plan'])
    require(all(complete.get(k) == v for k, v in result.items()), 'formal verifier/controller result differs')
    formal = ex.read(root/'formal/training_complete.json'); export = formal['exports'][selection_kind]
    saved = torch.load(export['path'], map_location='cpu', weights_only=False)
    adoption = values['selected_matcher_adoption']; contract = values['contract']
    binding, model_spec = verify_head_binding(saved, adoption, contract, runtime)
    require(saved['binding'] == formal['binding'] and saved['selection_kind'] == selection_kind
            and saved['updates'] == export['update']
            and saved.get('total_completed_updates') == values['plan'].record['total_updates']
            and saved.get('exposures') == saved['updates']*values['plan'].record['effective_batch']
            and saved.get('optimizer_imported') is False and saved.get('training_rng_included') is False,
            'trained head export identity/budget differs')
    gate = ex.read(root/'gpu_gate.json')
    controller_launch = ex.read(root/'controller_launch.json')
    require(gate['status'] == 'passed' and gate['formal_binding_sha256']
            == digest({k: v for k, v in binding.items() if k != 'run_mode'}), 'GPU/formal bindings differ')
    if controller_launch['preparation'] is not None:
        cpu = checked(controller_launch['preparation'])
        require(gate['formal_binding_sha256'] == cpu['formal_binding_sha256'], 'CPU/GPU bindings differ')
    else:
        from . import head_endpoint
        require(controller_launch.get('admission_mode') == 'direct_native_gpu_gate'
                and spec['topology'] == adoption.get('head_topology') == head_endpoint.DUAL_TOPOLOGY,
                'direct GPU admission must retain the explicit dual-head request')
        names = checked(gate['gradient_uninterrupted'])['parameter_names']
        for key in ('gradient_uninterrupted', 'gradient_resumed_from_update1'):
            module('matcher_v2_v1.gradient_gate', runtime).check_gradient_receipt(
                gate[key], {k: v for k, v in binding.items() if k != 'run_mode'}, names, 2)
    thresholds, origins = module('matcher_v2_v1.terminal', runtime).thresholds_for_export(saved)
    common = dict(schema=ORIGIN_SCHEMA, arm='B3', module=saved['module'], selection_kind=selection_kind,
        selected_updates=saved['updates'], total_completed_updates=values['plan'].record['total_updates'],
        checkpoint=export['path'], checkpoint_sha256=export['sha256'],
        model_state_sha256=export['model_state_sha256'],
        matcher_state_sha256=model_spec['initial_matcher_state_sha256'],
        matcher_selection_kind=adoption['selection_kind'], matcher_update=adoption['update'],
        matcher_choice_basis=adoption.get('matcher_choice_basis', 'predeclared_new_SIM_SELECT_rule'),
        development_informed_matcher_choice=adoption.get('development_informed_choice', False),
        selected_matcher_adoption_sha256=adoption['sha256'], validation_contract_sha256=digest(contract),
        new_sim_cal=contract['validation']['cal_mixed'], new_sim_select=contract['validation']['select_mixed'],
        thresholds=thresholds, threshold_origins=origins, selection_on_real=selection_kind == 'real_best',
        selection_on_test=False, turufan_used_for_selection=False, threshold_refitted=False,
        task3_overlay_applied=False, matcher_retrained=False, matcher_updated_during_head_training=False,
        controller_complete=ex.receipt(root/'controller_complete.json'), actual_controller_return=actual_controller,
        actual_formal_return=actual_formal, terminal_evaluation_complete=False)
    return saved, common


def load_model(saved, origin, runtime):
    hashing = module('curriculum_training_v1.checkpoint_io', runtime)
    require(origin.get('schema') == ORIGIN_SCHEMA and origin.get('module') == saved['module']
            and origin.get('model_state_sha256') == hashing.tree_sha(saved['model']),
            'new matched-head terminal identity required')
    binding = saved['binding']; spec = binding['model_spec']; adoption = spec['selected_matcher']
    contract = binding['mixed_sim_head_experiment']['validation_contract']
    verify_head_binding(saved, adoption, contract, runtime)
    require(origin.get('selected_matcher_adoption_sha256') == adoption['sha256']
            and origin.get('validation_contract_sha256') == digest(contract)
            and origin.get('matcher_state_sha256') == spec['initial_matcher_state_sha256'],
            'origin lost its Matcher/validation binding')
    # Do not invoke the old terminal.load_model: it requires historical sim_best.
    model = module('matcher_v2_v1.model_runtime', runtime).load_export_scorer(saved, runtime)
    require(not any(p.requires_grad for p in model.parameters()) and not any(m.training for m in model.modules())
            and hashing.tree_sha(model.state_dict()) == origin['model_state_sha256'],
            'terminal import changed state or is not entirely frozen/eval')
    geometry = module('s7_consensus_v1.compatibility', runtime).CompatibilityConfig(**spec['geometry'])
    return model, geometry, dict(origin, architecture=spec['architecture'], geometry=spec['geometry'],
        proposal_revision=spec['proposal_revision'], optimizer_imported=False)
