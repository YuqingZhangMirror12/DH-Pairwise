"""Load exact update-selected exports, never reinterpret them as old epochs.

This independent adapter does not change the already-bound training package.
SIM-best and REAL-development-best are evaluated separately. The retained
endpoint can be evaluated explicitly; it is not another model-selection rule.
"""
from pathlib import Path

import torch

from ..curriculum_training_v1.checkpoint_io import file_sha, tree_sha
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.launcher import verify_formal
from ..curriculum_training_v1.model_adapter import BASE, bound_module, require
from ..curriculum_training_v1.runtime_io import read, component_state, check_observations
from ..curriculum_training_v1.validation_adapter import check_report

CHOICES = ('sim_best', 'real_best', 'equal_budget_endpoint')
MODULES = ('scorer_patch', 'scorer_stats')


def thresholds_for_export(saved):
    """Use the selected update's CAL values; no fitting against final TEST."""
    choice = saved['selection_kind']; observation = saved['observation']
    require(choice in CHOICES and saved['module'] in MODULES, 'registered Scorer export required')
    require(observation['update'] == saved['updates'] > 0
            and observation['selection_eligible'] is True and observation['test_used'] is False,
            'only trained, non-TEST observations may supply thresholds')
    sim = observation['simulation']; real = observation['real_development']
    check_report(saved['module'], sim, real)
    result = dict(sim_test=sim['threshold'], dunhuang_cv=sim['threshold'], turufan=sim['threshold'])
    if choice == 'real_best':
        result.update(real['thresholds'])
    # The endpoint is a fixed-budget SIM-CAL reference, not a newly optimized
    # blend of the best SIM, best REAL, or heldout-domain thresholds.
    return result


def verified_export(controller_root, spec_path, plan, selection_kind):
    root = Path(controller_root).resolve(); spec_path = Path(spec_path).resolve()
    require(plan.record['module'] in MODULES and selection_kind in CHOICES,
            'registered curriculum lightweight Scorer evaluation required')
    order = 'curriculum'
    for path in (root/'controller_failure.json', root/'failure.json', root/'formal/failure.json',
                 root/'formal/exports/failure.json'):
        require(not path.exists(), 'failure precedes stale completion: ' + str(path))
    result = verify_formal(root/'formal', spec_path, plan, order)
    returned = read(root/'formal_return.json'); launch = read(root/'formal_launch.json')
    require(returned['phase'] == 'formal' and returned['returncode'] == 0
            and returned['launch_sha256'] == file_sha(root/'formal_launch.json')
            and launch['phase'] == 'formal', 'successful formal return/launch required')
    values = launch['command']
    for flag, expected in (('--spec', str(spec_path)), ('--order', order), ('--mode', 'formal'),
                           ('--out', str(root/'formal'))):
        require(values.count(flag) == 1 and values.index(flag)+1 < len(values)
                and values[values.index(flag)+1] == expected, 'formal command differs: ' + flag)
    receipt = read(root/'export_process_return.json'); controller = read(root/'controller_complete.json')
    require(receipt == result and controller == dict(result,
        successful_return_sha256=file_sha(root/'export_process_return.json'),
        formal_return_sha256=file_sha(root/'formal_return.json'), gpu_gate_sha256=file_sha(root/'gpu_gate.json')),
        'controller/export/return identity differs')
    exports = Path(result['export_root']); complete = read(exports/'training_complete.json')
    selection = read(exports/'selection.json'); binding = complete['binding']
    require(binding['module'] == plan.record['module'] and binding['order'] == order
            and digest(binding['common_plan']) == plan.sha256
            and selection['best_real'] is not None, 'Scorer SIM/REAL selection binding differs')
    gate = read(root/'gpu_gate.json')
    require(gate['status'] == 'passed'
            and gate['formal_binding_sha256'] == digest({k:v for k,v in binding.items() if k != 'run_mode'}),
            'formal binding differs from passed GPU gate')
    chosen = complete['exports'][selection_kind]
    saved = torch.load(chosen['path'], map_location='cpu', weights_only=False)
    expected = (plan.record['total_updates'] if selection_kind == 'equal_budget_endpoint'
                else selection['best_sim' if selection_kind == 'sim_best' else 'best_real']['update'])
    require(saved['schema'] == 'curriculum-model-export/1' and saved['module'] == plan.record['module']
            and saved['stage'] == 'scorer' and saved['order'] == order
            and saved['selection_kind'] == selection_kind and saved['updates'] == expected
            and saved['total_completed_updates'] == plan.record['total_updates']
            and saved['exposures'] == expected * plan.record['effective_batch']
            and saved['optimizer_imported'] is False and saved['training_rng_included'] is False,
            'selected export is not this trained Scorer/budget')
    check_observations([dict(update=saved['updates'], report=saved['observation'])], binding)
    if selection_kind != 'equal_budget_endpoint':
        best = selection['best_sim' if selection_kind == 'sim_best' else 'best_real']
        field = 'simulation' if selection_kind == 'sim_best' else 'real_development'
        require(best['report'] == saved['observation'][field]
                and best['observation_artifact'] == saved['observation']['artifact'], 'selected observation differs')
    matcher_sha = tree_sha(component_state(saved['model'], 'matcher.'))
    require(matcher_sha == binding['model_spec']['initial_matcher_state_sha256'],
            'frozen Matcher changed during Scorer training')
    origin = dict(schema='curriculum-scorer-evaluation-origin/1', module=saved['module'], order=order,
        variant='binary_' + saved['module'][len('scorer_'):], selection_kind=selection_kind,
        selected_updates=saved['updates'], total_completed_updates=saved['total_completed_updates'],
        selected_epoch=None, epoch_note='update-budget experiment; no fabricated epoch number',
        common_plan_sha256=plan.sha256, checkpoint=str(Path(chosen['path']).resolve()),
        checkpoint_sha256=chosen['sha256'], model_state_sha256=chosen['model_state_sha256'],
        matcher_state_sha256=matcher_sha, controller_complete_sha256=file_sha(root/'controller_complete.json'),
        selection_sha256=file_sha(exports/'selection.json'), stop_reason=complete['stop_reason'],
        claimed_converged=False, selection_on_real=selection_kind == 'real_best', selection_on_test=False,
        thresholds=thresholds_for_export(saved), matcher_updated_during_training=False,
        attention_present=False, local_conflict_head_present=False, learned_refinement=False,
        development_evaluation=True, historical_real_development_exposure=True, threshold_refitted=False)
    return saved, origin


def load_model(saved, origin, source_root):
    """Reconstruct the actual trained binary model, checking every tensor."""
    binding = saved['binding']; spec = binding['model_spec']; common = binding['common_plan']
    require(binding['module'] in MODULES and binding['order'] == 'curriculum'
            and origin['schema'] == 'curriculum-scorer-evaluation-origin/1'
            and origin['module'] == binding['module']
            and origin['model_state_sha256'] == tree_sha(saved['model'])
            and origin['common_plan_sha256'] == digest(common), 'verified Scorer export identity differs')
    variant = binding['module'][len('scorer_'):]
    require(spec['scorer_variant'] == variant
            and spec['proposal_revision'] == 'native-hypothesis-complete-link-union/1-diameter16'
            and spec['initialization'] == 'selected_curriculum_matcher_new_head'
            and spec['old_head_imported'] is False and spec['optimizer_imported'] is False,
            'registered T16 curriculum Matcher/new lightweight head required')
    selected = spec['selected_matcher']; path = Path(selected['path'])
    require(file_sha(path) == selected['sha256'], 'selected curriculum Matcher file changed')
    source = torch.load(path, map_location='cpu', weights_only=False)
    require(source['schema'] == 'curriculum-model-export/1' and source['binding'] == selected['source_binding']
            and source['module'] == 'matcher' and source['order'] == 'curriculum'
            and source['selection_kind'] == 'sim_best' and source['updates'] == selected['updates'] > 0
            and selected['old_head_imported'] is False and selected['optimizer_imported'] is False,
            'not the selected curriculum Matcher origin')
    for key in ('data_admission_sha256', 'geometry_sha256', 'baseline_sources_sha256', 'model_seed'):
        require(source['binding']['common_plan'][key] == common[key], 'Matcher/head source differs: ' + key)
    require(source['binding']['model_spec']['architecture'] == spec['architecture']
            and source['binding']['model_spec']['geometry'] == spec['geometry'], 'Matcher/head geometry or architecture differs')
    matcher_sha = tree_sha(component_state(saved['model'], 'matcher.'))
    require(matcher_sha == origin['matcher_state_sha256'] == spec['initial_matcher_state_sha256']
            == tree_sha(component_state(source['model'], 'matcher.')), 'actual frozen Matcher tensors changed')
    source_root = Path(source_root).resolve()
    inventory = {str(p.relative_to(source_root)):file_sha(p) for p in source_root.rglob('*.py')}
    require(inventory and digest(inventory) == common['baseline_sources_sha256'], 'bound baseline source changed')
    architecture_api = bound_module('staging.pairwise_v0_2.models.rachel_n512', source_root)
    scratch_api = bound_module(BASE+'s7_consensus_v1.scratch_matcher', source_root)
    geometry_api = bound_module(BASE+'s7_consensus_v1.compatibility', source_root)
    policy = bound_module(BASE+'s7_consensus_v1.pose_consensus', source_root)
    head_api = bound_module(BASE+'binary_scorer_v1.head', source_root)
    model_api = bound_module(BASE+'binary_scorer_v1.model', source_root)
    require(policy.REVISION == spec['proposal_revision'], 'builder revision changed')
    architecture = architecture_api.RachelN512Config(**spec['architecture'])
    require(architecture.feature_dim == 96, 'registered feature width96 required')
    geometry = geometry_api.CompatibilityConfig(**spec['geometry'])
    with torch.random.fork_rng(devices=[]):
        matcher = scratch_api.fresh_matcher(architecture, common['model_seed'])
        torch.manual_seed(common['head_seed'])
        model = model_api.BinaryConsensus(matcher, geometry, head=head_api.BinaryClusterHead(variant))
    require(all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in saved['model'].values()),
            'nonfinite selected model')
    model.load_state_dict(saved['model'], strict=True)
    model.matcher.set_frozen(True); model.requires_grad_(False); model.eval()
    require(tree_sha(model.state_dict()) == origin['model_state_sha256'], 'actual Scorer loading changed tensors')
    return model, dict(origin, architecture=spec['architecture'], geometry=spec['geometry'],
        proposal_revision=spec['proposal_revision'], baseline_sources_sha256=common['baseline_sources_sha256'],
        matcher_origin=dict(checkpoint=str(path), sha256=selected['sha256'], selected_updates=selected['updates'],
            matcher_retrained_for_this_head=False, initialization='SIM-selected curriculum random-start Matcher'),
        evidence_mode='exact_union_q_then_q_arc_pool', old_head_imported=False, optimizer_imported=False)
