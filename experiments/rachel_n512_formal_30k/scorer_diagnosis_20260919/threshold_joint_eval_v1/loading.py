"""Explicitly different loaders: frozen E32 control versus updated joint Matcher."""
from pathlib import Path

import torch

from consensus_joint_eval_common import frozen as base
from .contracts import (E32_SHA, PLAN_SHA, DOMAINS, read, sha, inventory,
                        validate_archive, validate_joint_terminal, matcher_change,
                        choose_real_best, validate_budget, metric_identity)


def source_artifacts(root, reference, binding):
    base.verify_code_binding(binding)
    root, reference = Path(root), Path(reference)
    files = ((root/'data_contract.json', 'data_contract_sha256'),
             (root/'geometry_calibration_v2/geometry_calibration.json', 'geometry_calibration_sha256'),
             (reference, 'reference_checkpoint_sha256'))
    for path, key in files:
        if sha(path) != binding[key]:
            raise ValueError('bound source/data/reference changed: ' + key)
    contract, calibration = read(files[0][0]), read(files[1][0])
    if (contract.get('status') != 'passed' or contract.get('source_disjoint') is not True
            or contract.get('schema') != 's7-consensus-data-contract/3'
            or calibration.get('contract_sha256') != binding['data_contract_sha256']):
        raise ValueError('completed v14 data/calibration contract required')
    return contract, calibration


def freeze_model(model, state):
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('nonfinite selected model tensor')
    model.eval().requires_grad_(False)
    model.matcher.set_frozen(True)  # Evaluation only; not a claim about training.
    return model


def make_model(reference, calibration, state):
    adapter = base.S7MatcherAdapter.from_s7_m12(reference)
    model = base.S7Consensus(adapter, base.CompatibilityConfig.from_calibration(calibration),
                            head=base.training.fresh_head(base.TrainingConfig().head_seed))
    return freeze_model(model, state)


def verify_origin(root, binding):
    spec = binding.get('origin_matcher', {})
    if (spec.get('sha256') != E32_SHA or spec.get('head_imported') is not False
            or spec.get('matcher_training') is not True):
        raise ValueError('joint origin must be approved E32 Matcher only')
    path = Path(spec['path']).resolve()
    if (path.name != 'best_joint.pt' or path.parent.name != 'matcher'
            or path.parent.parent.name != 'formal_scratch' or sha(path) != E32_SHA):
        raise ValueError('E32 origin content/path differs')
    cp = torch.load(path, map_location='cpu', weights_only=False)
    selection, complete = read(path.parent/'selection.json'), read(path.parent/'complete.json')
    origin = cp.get('binding', {})
    if (cp.get('stage') != 'matcher' or cp.get('epoch') != 32
            or origin.get('arm') != 'scratch' or origin.get('formal_training') is not True
            or origin.get('preflight_steps') != 0
            or selection.get('status') != 'selected' or complete.get('status') != 'stage_complete'
            or selection.get('selection_on_real') is not False
            or selection.get('best', {}).get('epoch') != 32
            or cp.get('metrics', {}).get('key') != selection['best']['key']
            or complete.get('best') != selection['best']
            or any(r.get('binding') != origin or r.get('best_joint_sha256') != E32_SHA
                   for r in (selection, complete))):
        raise ValueError('origin is not completed SIM-selected E32')
    validate_budget(selection)
    for key in ('actual_epochs', 'updates', 'exposures', 'stop_reason'):
        if complete.get(key) != selection.get(key):
            raise ValueError('origin terminal metadata differs')
    for key in ('data_contract_sha256', 'geometry_calibration_sha256', 'reference_checkpoint_sha256'):
        if not origin.get(key) or origin[key] != binding.get(key):
            raise ValueError('joint origin data/reference differs')
    package = Path(*base.training.__package__.split('.'))
    if inventory(path.parents[2]/'source'/package) != origin['implementation_sha256']:
        raise ValueError('origin source changed')
    imported = read(Path(root)/'formal_scratch_joint/origin_matcher_import.json')
    if (Path(imported.get('path', '')).resolve() != path or imported.get('sha256') != E32_SHA
            or imported.get('epoch') != 32 or imported.get('old_head_imported') is not False
            or imported.get('optimizer_imported') is not False
            or imported.get('matcher_frozen_for_import_only') is not True
            or imported.get('joint_training_will_unfreeze') is not True):
        raise ValueError('joint origin import receipt differs')
    return cp, dict(checkpoint=str(path), sha256=E32_SHA, selected_epoch=32,
                    historical_matcher_epochs=selection['actual_epochs'])


def load_joint(root, reference, choice, real_plan, helper):
    root = Path(root).resolve()
    arm = root/'formal_scratch_joint'
    if any((arm/p).exists() for p in ('failure.json', 'scorer/failure.json')):
        raise ValueError('joint failure requires explicit recovery review')
    stage = arm/'scorer'
    terminal = read(arm/'training_complete.json')
    selection, complete = read(stage/'selection.json'), read(stage/'complete.json')
    filename = {'sim': 'best_joint.pt', 'real': 'best_real.pt'}[choice]
    path = stage/filename
    selected_sha = sha(path)
    if selected_sha != selection[filename[:-3]+'_sha256']:
        raise ValueError('selected joint checkpoint hash changed')
    cp = torch.load(path, map_location='cpu', weights_only=False)
    binding = validate_joint_terminal(cp, selection, complete, terminal,
        base.training.canonical_record(base.TrainingConfig().record()), choice)
    contract, calibration = source_artifacts(root, reference, binding)
    if sha(real_plan) != PLAN_SHA or helper.bind_plan(real_plan) != binding['real_development']:
        raise ValueError('joint real-development binding changed')
    origin_cp, origin = verify_origin(root, binding)
    # Epoch0 can win the old SIM rule. It has no joint updates and is labelled
    # explicitly; never reinterpret it as a trained joint improvement.
    delta = matcher_change(origin_cp['model'], cp['model'], expect_updated=cp['epoch'] > 0)
    validation = read(stage/f"epoch_{cp['epoch']:03d}_validation.json")
    if choice == 'sim':
        if validation['key'] != selection['best']['key'] or validation['threshold'] != cp['threshold']:
            raise ValueError('joint SIM validation differs')
        thresholds = {name: cp['threshold'] for name in ('sim_test_v14',) + DOMAINS}
    else:
        if metric_identity(validation.get('real_development', {})) != metric_identity(cp['metrics']):
            raise ValueError('joint REAL validation differs')
        thresholds = dict(cp['thresholds'], sim_test_v14=validation['threshold'])
    model = make_model(reference, calibration, cp['model'])
    provenance = dict(arm='scratch_joint', variant='threshold_joint', evidence_mode='exact_union_q',
        checkpoint=str(path), checkpoint_sha256=selected_sha, selected_epoch=cp['epoch'],
        last_epoch=selection['actual_epochs'], selection_kind=choice, thresholds=thresholds,
        selection_on_real=choice == 'real', selection_on_test=False,
        real_metrics_recorded_during_training=True, real_used_for_stopping=False,
        matcher_origin=origin, matcher_updated_during_training=True,
        selected_matcher_change_from_e32=delta, selected_epoch_zero=cp['epoch'] == 0,
        stop_reason=selection['stop_reason'], real_plan_sha256=PLAN_SHA,
        real_development_binding=binding['real_development'],
        data_contract_sha256=binding['data_contract_sha256'],
        geometry_calibration_sha256=binding['geometry_calibration_sha256'],
        training_implementation_sha256=binding['implementation_sha256'],
        selection_sha256=sha(stage/'selection.json'), terminal_receipt_sha256=sha(arm/'training_complete.json'),
        model_selection_on_test_or_real=choice == 'real', threshold_refitted=False,
        development_evaluation=True, historical_real_development_exposure=True)
    return model, contract, provenance


def load_frozen_control(root, reference):
    # Keep every original fail-closed check, including full E32 tensor equality.
    model, contract, provenance = base.load_selected(root, 'scratch_fixed', reference,
                                                    completed_arm_only=True)
    if provenance['variant'] != 'threshold' or provenance['matcher_origin']['sha256'] != E32_SHA:
        raise ValueError('control must be original threshold frozen E32')
    stage = Path(root)/'formal_scratch_fixed/scorer'
    selection = read(stage/'selection.json')
    return model, contract, provenance, selection


def load_frozen_real(root, reference, selection_path, real_plan, helper):
    model, contract, provenance, original = load_frozen_control(root, reference)
    record = read(selection_path)
    binding = helper.bind_plan(real_plan)
    if (sha(real_plan) != PLAN_SHA or record.get('schema') != 'frozen-e32-real-reselection/1'
            or record.get('status') != 'complete' or record.get('test_used') is not False
            or record.get('original_selection_sha256') != provenance['selection_sha256']
            or record.get('terminal_receipt_sha256') != provenance['terminal_receipt_sha256']
            or record.get('development_binding') != binding):
        raise ValueError('frozen REAL reselection identity differs')
    curve_path = Path(selection_path).parent/'real_curve.json'
    if sha(curve_path) != record['curve_sha256']:
        raise ValueError('REAL reselection curve changed')
    curve = read(curve_path)
    if [r['epoch'] for r in curve] != list(range(0, original['actual_epochs'] + 1, 2)):
        raise ValueError('REAL reselection omitted archived epochs')
    chosen = choose_real_best(curve)
    if record['best'] != chosen:
        raise ValueError('not the registered development winner')
    stage = Path(root)/'formal_scratch_fixed/scorer'
    path = stage/f"epoch_{chosen['epoch']:03d}_weights.pt"
    if Path(chosen['checkpoint']).resolve() != path.resolve() or sha(path) != chosen['checkpoint_sha256']:
        raise ValueError('chosen archived control checkpoint changed')
    cp = torch.load(path, map_location='cpu', weights_only=False)
    validate_archive(cp, read(stage/f"epoch_{chosen['epoch']:03d}_validation.json"),
                     original['binding'], chosen['epoch'])
    # model currently contains the validated original SIM winner's frozen E32.
    matcher_change(model.state_dict(), cp['model'], expect_updated=False)
    freeze_model(model, cp['model'])
    thresholds = dict(chosen['real_report']['thresholds'], sim_test_v14=cp['threshold'])
    provenance.update(checkpoint=str(path), checkpoint_sha256=chosen['checkpoint_sha256'],
        selected_epoch=chosen['epoch'], selection_kind='real', thresholds=thresholds,
        selection_on_real=True, selection_on_test=False, model_selection_on_test_or_real=True,
        development_evaluation=True, real_used_for_stopping=False, real_plan_sha256=PLAN_SHA,
        original_sim_selection_preserved=True, matcher_updated_during_training=False,
        real_development_binding=binding,
        real_selection_sha256=sha(selection_path), threshold_refitted=False)
    return model, contract, provenance
