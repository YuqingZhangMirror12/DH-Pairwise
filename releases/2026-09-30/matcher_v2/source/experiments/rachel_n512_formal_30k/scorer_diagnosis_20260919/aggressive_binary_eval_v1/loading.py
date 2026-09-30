"""Freeze the newly trained Matcher + Patch head; never import E32 by alias."""
from pathlib import Path
import torch
from consensus_binary_eval_common import frozen as common
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.aggressive_binary_v1 import runtime
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.aggressive_binary_v1.admission import validate_data
from .contracts import (read, sha, inventory, PLAN_SHA, DOMAINS, SIM_SPLIT, validate_terminal,
                        validate_budget, verify_selection_curve, metric_identity)


def need(value, message):
    if not value:
        raise ValueError(message)


def verify_origin(formal, binding, scorer_state, config, initial_matcher_sha):
    stage = formal / 'matcher'
    spec = runtime.selected_matcher(stage, binding, config)
    need(spec == binding.get('fixed_matcher'), 'selected new Matcher binding changed')
    cp = torch.load(spec['path'], map_location='cpu', weights_only=False)
    selection = read(stage / 'selection.json')
    validate_budget(selection)
    history = read(stage / 'learning_curve.json')
    chosen = next(r for r in history if r['epoch'] == cp['epoch'])
    recorded = {k:v for k,v in chosen.items() if k not in ('epoch','updates','exposures')}
    need(metric_identity(cp.get('metrics', {})) == metric_identity(recorded)
         and cp['metrics'].get('real_used') is False, 'Matcher did not use SIM native-coverage selection')
    for row in history:
        need(row.get('updates') == row['epoch'] * 750 and row.get('exposures') == row['epoch'] * 24000,
             'Matcher validation update identity differs')
    initial = read(formal / 'matcher_initialization.json')
    need(initial.get('initialization') == 'random_seed_no_weight_import'
         and initial.get('seed') == config.matcher_seed
         and initial.get('initial_state_sha256') == initial_matcher_sha
         and initial.get('reference_weights_imported') is False
         and initial.get('optimizer_imported') is False
         and initial.get('old_head_imported') is False, 'random Matcher initialization receipt differs')
    imported = read(formal / 'scorer_initialization.json')
    need(imported.get('initialization') == 'new_aggressive_selected_matcher'
         and Path(imported.get('path', '')).resolve() == Path(spec['path']).resolve()
         and imported.get('sha256') == spec['sha256'] and imported.get('epoch') == spec['epoch']
         and imported.get('old_head_imported') is False and imported.get('optimizer_imported') is False
         and imported.get('matcher_frozen') is True, 'new Matcher import into fresh Scorer differs')
    common.equal_matcher(cp['model'], scorer_state)
    return dict(checkpoint=spec['path'], sha256=spec['sha256'], selected_epoch=spec['epoch'],
        actual_matcher_epochs=selection['actual_epochs'], updates=selection['updates'], exposures=selection['exposures'],
        matcher_trained_from_random_in_this_experiment=True, epoch_selected_on_real=False,
        selection_sha256=spec['selection_sha256'], terminal_sha256=spec['terminal_sha256'])


def load_selected(root, reference, choice, real_plan, helper):
    root = Path(root).resolve(); formal = root / 'formal_scratch_aggressive'; stage = formal / 'scorer'
    need(not any((formal / name).exists() for name in
        ('failure.json', 'failure_matcher.json', 'failure_scorer.json', 'matcher/failure.json', 'scorer/failure.json')),
        'experiment3 failure requires explicit recovery review')
    terminal = read(formal/'training_complete.json'); selection = read(stage/'selection.json')
    complete = read(stage/'complete.json')
    filename = {'sim':'best_joint.pt','real':'best_real.pt'}[choice]; path = stage/filename
    need(sha(path) == selection[filename[:-3]+'_sha256'], 'selected aggressive Scorer file changed')
    cp = torch.load(path, map_location='cpu', weights_only=False)
    training = common.training; config = training.TrainingConfig(scorer_variant='patch')
    binding = validate_terminal(cp,selection,complete,terminal,training.canonical_record(config.record()),choice)
    common.verify_code_binding(binding)
    package = Path(training.__file__).parent.parent
    for directory, key in (('binary_scorer_v1','binary_scorer_sha256'),
                           ('aggressive_binary_v1','aggressive_implementation_sha256')):
        need(inventory(package/directory) == binding[key], 'model/stage implementation changed: '+directory)
    for file,key in ((root/'data_contract.json','data_contract_sha256'),
        (root/'geometry_calibration_v2/geometry_calibration.json','geometry_calibration_sha256'),
        (Path(reference),'reference_checkpoint_sha256')):
        need(sha(file) == binding[key], 'source artifact differs: '+key)
    admission = validate_data(root/'data_contract.json',root/'geometry_calibration_v2/geometry_calibration.json',
                              root/'review_approval.json')
    need(admission == binding['admission'], 'new data/full-review approval changed')
    need(sha(real_plan) == PLAN_SHA and helper.bind_plan(real_plan) == binding['real_development'],
         'REAL development source binding differs')
    contract = read(root/'data_contract.json'); calibration = read(root/'geometry_calibration_v2/geometry_calibration.json')
    reference_model = common.S7MatcherAdapter.from_s7_m12(reference)
    geometry = common.CompatibilityConfig.from_calibration(calibration)
    model, initial = runtime.make_model(reference_model.base.config,geometry,config,'matcher')
    del reference_model
    origin = verify_origin(formal,binding,cp['model'],config,initial['initial_state_sha256'])
    rows = [read(stage/f'epoch_{e:03d}_validation.json') for e in range(0,selection['actual_epochs']+1,2)]
    verify_selection_curve(rows,selection)
    validation = rows[cp['epoch']//2]
    recorded = ({k:v for k,v in validation.items() if k not in ('epoch','updates','exposures')}
                if choice == 'sim' else validation['real_development'])
    need(metric_identity(recorded) == metric_identity(cp['metrics']), 'Scorer metrics differ from recorded validation')
    thresholds = ({s:cp['threshold'] for s in (SIM_SPLIT,)+DOMAINS} if choice == 'sim'
                  else dict(cp['thresholds'], **{SIM_SPLIT:validation['threshold']}))
    model.load_state_dict(cp['model'], strict=True)
    need(all(torch.isfinite(v).all() for v in model.state_dict().values()), 'nonfinite selected model')
    model.eval().requires_grad_(False); model.matcher.set_frozen(True)
    provenance = dict(arm='scratch_aggressive',variant='binary_patch',experiment_variant='aggressive_binary_patch',
        evidence_mode='exact_union_q_then_q_arc_pool',checkpoint=str(path),checkpoint_sha256=sha(path),
        selected_epoch=cp['epoch'],last_epoch=selection['actual_epochs'],selection_kind=choice,
        thresholds=thresholds,selection_on_real=choice=='real',selection_on_test=False,
        selected_epoch_zero=cp['epoch']==0,matcher_origin=origin,
        matcher_updated_during_scorer_training=False,matcher_trained_from_random_in_this_experiment=True,
        stop_reason=selection['stop_reason'],real_plan_sha256=PLAN_SHA,real_development_binding=binding['real_development'],
        data_contract_sha256=binding['data_contract_sha256'],geometry_calibration_sha256=binding['geometry_calibration_sha256'],
        review_approval_sha256=binding['admission']['approval']['sha256'],
        training_implementation_sha256=binding['implementation_sha256'],
        binary_implementation_sha256=binding['binary_scorer_sha256'],
        aggressive_implementation_sha256=binding['aggressive_implementation_sha256'],
        selection_sha256=sha(stage/'selection.json'),terminal_receipt_sha256=sha(formal/'training_complete.json'),
        real_used_for_stopping=False,threshold_refitted=False,development_evaluation=True,
        historical_real_development_exposure=True,attention_present=False,local_conflict_head_present=False,
        learned_refinement=False)
    return model,contract,provenance
