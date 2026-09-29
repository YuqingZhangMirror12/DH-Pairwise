"""Read a successfully completed C/M export; no training or GPU allocation.

Both SIM-best and equal-budget endpoint are legal terminal evaluations.
Mixed-order exports must never be substituted for the curriculum Matcher that
initializes the two Scorer experiments; that separate admission rule remains
in runtime_io.selected_curriculum_matcher.
"""
from pathlib import Path

import torch

from .checkpoint_io import file_sha, tree_sha
from .exposure import digest
from .launcher import verify_formal
from .model_adapter import BASE, bound_module, require
from .runtime_io import read, component_state


def verified_export(controller_root, spec_path, plan, order, selection_kind):
    root = Path(controller_root).resolve(); spec_path = Path(spec_path).resolve()
    require(plan.record['module'] == 'matcher' and order in ('curriculum', 'mixed')
            and selection_kind in ('sim_best', 'equal_budget_endpoint'), 'registered native Matcher evaluation required')
    for path in (root / 'controller_failure.json', root / 'failure.json', root / 'formal/failure.json',
                 root / 'formal/exports/failure.json'):
        require(not path.exists(), 'failure precedes stale completion: ' + str(path))
    result = verify_formal(root / 'formal', spec_path, plan, order)
    returned = read(root / 'formal_return.json'); launch = read(root / 'formal_launch.json')
    require(returned['phase'] == 'formal' and returned['returncode'] == 0
            and returned['launch_sha256'] == file_sha(root / 'formal_launch.json')
            and launch['phase'] == 'formal', 'successful formal return/launch required')
    values = launch['command']
    for flag, expected in (('--spec', str(spec_path)), ('--order', order), ('--mode', 'formal'),
                           ('--out', str(root / 'formal'))):
        require(values.count(flag) == 1 and values.index(flag) + 1 < len(values)
                and values[values.index(flag) + 1] == expected, 'formal command differs: ' + flag)
    receipt = read(root / 'export_process_return.json'); controller = read(root / 'controller_complete.json')
    require(receipt == result and controller == dict(result,
        successful_return_sha256=file_sha(root / 'export_process_return.json'),
        formal_return_sha256=file_sha(root / 'formal_return.json'), gpu_gate_sha256=file_sha(root / 'gpu_gate.json')),
        'controller/export/return identity differs')
    exports = Path(result['export_root']); complete = read(exports / 'training_complete.json')
    selection = read(exports / 'selection.json'); binding = complete['binding']
    require(binding['module'] == 'matcher' and binding['order'] == order
            and digest(binding['common_plan']) == plan.sha256
            and selection['best_real'] is None, 'Matcher must be selected by simulation only')
    gate = read(root / 'gpu_gate.json')
    require(gate['status'] == 'passed'
            and gate['formal_binding_sha256'] == digest({k:v for k,v in binding.items() if k != 'run_mode'}),
            'formal binding differs from passed GPU gate')
    chosen = complete['exports'][selection_kind]
    saved = torch.load(chosen['path'], map_location='cpu', weights_only=False)
    expected_update = selection['best_sim']['update'] if selection_kind == 'sim_best' else plan.record['total_updates']
    require(saved['schema'] == 'curriculum-model-export/1' and saved['module'] == 'matcher'
            and saved['stage'] == 'matcher' and saved['order'] == order
            and saved['selection_kind'] == selection_kind and saved['updates'] == expected_update
            and saved['total_completed_updates'] == plan.record['total_updates']
            and saved['exposures'] == expected_update * plan.record['effective_batch']
            and saved['optimizer_imported'] is False and saved['training_rng_included'] is False,
            'selected export is not this trained Matcher/budget')
    require(tree_sha(component_state(saved['model'], 'head.')) == binding['model_spec']['initial_head_state_sha256'],
            'inactive Scorer changed during Matcher training')
    origin = dict(schema='curriculum-matcher-evaluation-origin/1', module='matcher', order=order,
        selection_kind=selection_kind, updates=saved['updates'], total_completed_updates=saved['total_completed_updates'],
        common_plan_sha256=plan.sha256, selected_file_sha256=chosen['sha256'],
        model_state_sha256=chosen['model_state_sha256'], matcher_state_sha256=tree_sha(component_state(saved['model'], 'matcher.')),
        controller_complete_sha256=file_sha(root / 'controller_complete.json'),
        selection_sha256=file_sha(exports / 'selection.json'), stop_reason=complete['stop_reason'],
        claimed_converged=False, scorer_used=False, real_used_for_selection=False)
    return saved, origin


def load_matcher(saved, origin, source_root):
    """Reconstruct only the actual Matcher. No new/old Scorer is instantiated."""
    spec = saved['binding']['model_spec']; common = saved['binding']['common_plan']
    require(saved['binding']['module'] == 'matcher' and origin['schema'] == 'curriculum-matcher-evaluation-origin/1'
            and origin['model_state_sha256'] == tree_sha(saved['model'])
            and origin['common_plan_sha256'] == digest(common), 'verified export identity differs')
    require(spec['proposal_revision'] == 'native-hypothesis-complete-link-union/1-diameter16'
            and spec['initialization'] == 'shared_random_seed' and spec['selected_matcher'] is None,
            'registered random-start T16 Matcher required')
    source_root = Path(source_root).resolve()
    inventory = {str(p.relative_to(source_root)):file_sha(p) for p in source_root.rglob('*.py')}
    require(inventory and digest(inventory) == common['baseline_sources_sha256'], 'bound baseline source changed')
    architecture_api = bound_module('staging.pairwise_v0_2.models.rachel_n512', source_root)
    scratch_api = bound_module(BASE + 's7_consensus_v1.scratch_matcher', source_root)
    geometry_api = bound_module(BASE + 's7_consensus_v1.compatibility', source_root)
    architecture = architecture_api.RachelN512Config(**spec['architecture'])
    require(architecture.feature_dim == 96, 'registered feature width96 required')
    geometry = geometry_api.CompatibilityConfig(**spec['geometry'])
    matcher = scratch_api.fresh_matcher(architecture, common['model_seed'])
    state = component_state(saved['model'], 'matcher.')
    require(tree_sha(state) == origin['matcher_state_sha256'], 'exported Matcher state changed')
    require(all(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()) for tensor in state.values()),
            'nonfinite Matcher export')
    matcher.load_state_dict(state, strict=True); matcher.set_frozen(True); matcher.eval()
    require(tree_sha(matcher.state_dict()) == origin['matcher_state_sha256'], 'actual Matcher loading changed tensor values')
    provenance = dict(origin, initial_matcher_state_sha256=spec['initial_matcher_state_sha256'],
        architecture=spec['architecture'], geometry=spec['geometry'], proposal_revision=spec['proposal_revision'],
        baseline_sources_sha256=common['baseline_sources_sha256'], old_head_imported=False, optimizer_imported=False)
    return matcher, geometry, provenance
