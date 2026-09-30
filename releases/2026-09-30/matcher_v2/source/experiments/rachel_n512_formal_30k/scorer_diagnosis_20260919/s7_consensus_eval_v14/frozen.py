"""Load only the predeclared, terminal, simulation-selected Consensus winner.

No model ranking, threshold fitting, data generation or optimizer belongs here.
The historical default waits for both arms. The explicitly parallel threshold
and simple queues may evaluate one terminal arm, with the same full selection,
budget, source, data and Matcher checks. A running arm is never evaluated.
"""
import hashlib
import json
from pathlib import Path

import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1 import train as training
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.config import TrainingConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.matcher import S7MatcherAdapter
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.model import S7Consensus


def registered_protocol(config):
    """Arm names come from the bound implementation, never a bypass CLI flag."""
    revision = config.get('proposal_revision')
    variants = {
        None: ('legacy', ('m12', 'scratch'), 'directional_full_q'),
        'common-pose-merge-repair/1': ('mergefix', ('m12', 'scratch'), 'directional_full_q'),
        'native-hypothesis-complete-link-union/1-diameter16':
            ('threshold', ('m12', 'scratch_fixed'), 'exact_union_q'),
        'raw-displacement-modes-simple/1-sim16':
            ('simple', ('m12', 'scratch_fixed'), 'directional_full_q'),
    }
    if revision not in variants:
        raise ValueError('unregistered proposal revision')
    variant, arms, evidence_mode = variants[revision]
    if variant == 'threshold' and config.get('threshold_policy') != dict(
            pose_diameter_px=16., candidate_budget=8, maximum_interpenetration_sum=.10):
        raise ValueError('registered fixed16 diameter policy changed')
    return dict(variant=variant, arms=arms, evidence_mode=evidence_mode)


ARMS = registered_protocol(TrainingConfig().record())['arms']
STOP_REASONS = ('simulation_plateau_after_lr_reductions',
                'budget_limit_not_claimed_converged')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def terminal_training(root, *, completed_arm=None):
    """Check terminal receipts before any heldout input; scope is explicit."""
    root = Path(root)
    if completed_arm is not None:
        protocol = registered_protocol(TrainingConfig().record())
        if (protocol['variant'] not in ('threshold', 'simple')
                or completed_arm not in ARMS):
            raise ValueError('selected-arm completion scope is not registered')
    arms = ARMS if completed_arm is None else (completed_arm,)
    result = {}
    for arm in arms:
        directory = root / ('formal_' + arm)
        if any((directory / rel).exists() for rel in ('failure.json', 'scorer/failure.json')):
            raise ValueError('preserved training failure requires explicit recovery review: ' + arm)
        path = directory / 'training_complete.json'
        if not path.exists():
            prefix = ('both arms must finish' if completed_arm is None
                      else 'selected arm must finish')
            raise ValueError(prefix + ' before frozen evaluation: ' + arm)
        value = read(path)
        stages = ['matcher', 'scorer'] if arm == 'scratch' else ['scorer']
        if (value.get('status') != 'training_complete' or value.get('arm') != arm
                or value.get('stages') != stages
                or value.get('last_stage', {}).get('status') != 'stage_complete'):
            raise ValueError('invalid terminal training receipt: ' + arm)
        result[arm] = value
    return result


def validate_selection(checkpoint, selection, complete, terminal, arm):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    binding = checkpoint.get('binding', {})
    if (checkpoint.get('stage') != 'scorer' or binding.get('arm') != arm
            or binding.get('formal_training') is not True
            or binding.get('preflight_steps') != 0
            or binding.get('config') != training.canonical_record(TrainingConfig().record())):
        raise ValueError('not the registered formal Scorer checkpoint')
    if (selection.get('status') != 'selected' or complete.get('status') != 'stage_complete'
            or terminal.get('status') != 'training_complete'
            or any(x.get('binding') != binding for x in (selection, complete, terminal))
            or terminal.get('last_stage') != complete):
        raise ValueError('checkpoint, selection and terminal identity differ')
    chosen = selection['best']
    if (complete.get('best') != chosen or checkpoint.get('epoch') != chosen['epoch']
            or checkpoint.get('threshold') != chosen['threshold']
            or checkpoint.get('metrics', {}).get('key') != chosen['key']
            or checkpoint['metrics'].get('threshold') != chosen['threshold']):
        raise ValueError('not the selected joint checkpoint and CAL threshold')
    if not .2 <= chosen['threshold'] <= .8 or selection.get('selection_on_real') is not False:
        raise ValueError('unregistered threshold or real-selected checkpoint')
    for key in ('actual_epochs', 'updates', 'exposures', 'stop_reason',
                'best_joint_sha256', 'matcher_unchanged'):
        if selection.get(key) != complete.get(key):
            raise ValueError('terminal selection metadata differs: ' + key)
    epochs = selection['actual_epochs']
    if (type(epochs) is not int or not 16 <= epochs <= 48
            or selection['updates'] != epochs * 750
            or selection['exposures'] != epochs * 24000
            or selection['stop_reason'] not in STOP_REASONS
            or selection.get('matcher_unchanged') is not True
            or not 0 <= chosen['epoch'] <= epochs):
        raise ValueError('completed training budget or frozen Matcher receipt differs')
    return binding


def verify_code_binding(binding):
    directory = Path(training.__file__).resolve().parent
    actual = {p.name: sha(p) for p in directory.glob('*.py')}
    if actual != binding['implementation_sha256']:
        raise ValueError('evaluation imported different training implementation')


def equal_matcher(selected, scorer):
    expected = {k: v for k, v in selected.items() if k.startswith('matcher.')}
    actual = {k: v for k, v in scorer.items() if k.startswith('matcher.')}
    if not expected or set(expected) != set(actual):
        raise ValueError('selected Matcher tensor membership differs')
    for key, tensor in expected.items():
        if not torch.equal(tensor, actual[key]):
            raise ValueError('selected scratch Matcher changed in Scorer: ' + key)


def verify_scratch_matcher(root, binding, scorer_checkpoint):
    """The Scorer must retain the exact independently selected scratch Matcher."""
    directory = root/'formal_scratch'/'matcher'
    selection, complete = read(directory/'selection.json'), read(directory/'complete.json')
    path = directory/'best_joint.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (selection.get('status') != 'selected' or complete.get('status') != 'stage_complete'
            or any(x.get('binding') != binding for x in (selection, complete, checkpoint))
            or checkpoint.get('stage') != 'matcher'
            or selection.get('selection_on_real') is not False
            or complete.get('best') != selection.get('best')
            or checkpoint.get('epoch') != selection['best']['epoch']
            or checkpoint.get('metrics', {}).get('key') != selection['best']['key']
            or sha(path) != selection.get('best_joint_sha256')
            or selection.get('best_joint_sha256') != complete.get('best_joint_sha256')):
        raise ValueError('scratch Matcher is not the simulation-selected checkpoint')
    equal_matcher(checkpoint['model'], scorer_checkpoint['model'])
    return dict(checkpoint=str(path), sha256=sha(path), selected_epoch=selection['best']['epoch'])


def verify_fixed_matcher(root, binding, scorer_checkpoint):
    """Later heads use the prior SIM winner, not a new/real-selected Matcher."""
    spec = binding.get('fixed_matcher', {})
    if spec.get('head_imported') is not False or spec.get('matcher_training') is not False:
        raise ValueError('explicit frozen Matcher-only import required')
    path = Path(spec['path']).resolve()
    if (path.name != 'best_joint.pt' or path.parent.name != 'matcher'
            or path.parent.parent.name != 'formal_scratch' or sha(path) != spec.get('sha256')):
        raise ValueError('external selected Matcher path/hash differs')
    selection, complete = read(path.parent/'selection.json'), read(path.parent/'complete.json')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    origin = checkpoint.get('binding', {})
    chosen = selection.get('best', {})
    epochs = selection.get('actual_epochs')
    if (selection.get('status') != 'selected' or complete.get('status') != 'stage_complete'
            or any(x.get('binding') != origin for x in (selection, complete))
            or origin.get('arm') != 'scratch' or origin.get('formal_training') is not True
            or origin.get('preflight_steps') != 0 or checkpoint.get('stage') != 'matcher'
            or selection.get('selection_on_real') is not False
            or complete.get('best') != chosen or checkpoint.get('epoch') != chosen.get('epoch')
            or checkpoint.get('metrics', {}).get('key') != chosen.get('key')
            or selection.get('best_joint_sha256') != spec['sha256']
            or complete.get('best_joint_sha256') != spec['sha256']
            or type(epochs) is not int or not 16 <= epochs <= 48
            or not 0 < chosen.get('epoch', 0) <= epochs
            or selection.get('updates') != epochs*750
            or selection.get('exposures') != epochs*24000
            or selection.get('stop_reason') not in STOP_REASONS
            or any(selection.get(k) != complete.get(k) for k in
                   ('actual_epochs', 'updates', 'exposures', 'stop_reason'))):
        raise ValueError('external Matcher is not a completed simulation selection')
    for key in ('data_contract_sha256', 'geometry_calibration_sha256', 'reference_checkpoint_sha256'):
        if not origin.get(key) or origin[key] != binding.get(key):
            raise ValueError('external Matcher data/geometry/reference binding differs: ' + key)
    # Source remains a separate immutable experiment; never import it over the
    # current variant merely to inspect the origin checkpoint.
    package = Path(*training.__package__.split('.'))
    directory = path.parents[2]/'source'/package
    if {p.name: sha(p) for p in directory.glob('*.py')} != origin.get('implementation_sha256'):
        raise ValueError('external Matcher source binding differs')
    imported = read(root/'formal_scratch_fixed'/'fixed_matcher_import.json')
    if (Path(imported.get('path', '')).resolve() != path
            or imported.get('sha256') != spec['sha256'] or imported.get('epoch') != chosen['epoch']
            or imported.get('old_head_imported') is not False
            or imported.get('optimizer_imported') is not False
            or imported.get('matcher_frozen') is not True):
        raise ValueError('external Matcher import receipt differs')
    equal_matcher(checkpoint['model'], scorer_checkpoint['model'])
    return dict(checkpoint=str(path), sha256=spec['sha256'], selected_epoch=chosen['epoch'],
                selection_sha256=sha(path.parent/'selection.json'),
                complete_sha256=sha(path.parent/'complete.json'),
                historical_matcher_epochs=epochs, matcher_retrained_for_this_head=False)


def load_selected(root, arm, reference, *, completed_arm_only=False):
    root, reference = Path(root).resolve(), Path(reference).resolve()
    if arm not in ARMS:
        raise ValueError('arm is not registered for the imported training implementation')
    terminal = terminal_training(root, completed_arm=arm if completed_arm_only else None)[arm]
    directory = root / ('formal_' + arm) / 'scorer'
    checkpoint_path = directory / 'best_joint.pt'
    selection, complete = read(directory / 'selection.json'), read(directory / 'complete.json')
    if sha(checkpoint_path) != selection['best_joint_sha256']:
        raise ValueError('selected checkpoint content changed')
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    binding = validate_selection(cp, selection, complete, terminal, arm)
    verify_code_binding(binding)
    contract_path = root / 'data_contract.json'
    calibration_path = root / 'geometry_calibration_v2' / 'geometry_calibration.json'
    for path, key in ((contract_path, 'data_contract_sha256'),
                      (calibration_path, 'geometry_calibration_sha256'),
                      (reference, 'reference_checkpoint_sha256')):
        if sha(path) != binding[key]:
            raise ValueError('frozen artifact binding differs: ' + key)
    contract, calibration = read(contract_path), read(calibration_path)
    if (contract.get('status') != 'passed' or contract.get('schema') != 's7-consensus-data-contract/3'
            or contract.get('source_disjoint') is not True
            or calibration.get('contract_sha256') != sha(contract_path)):
        raise ValueError('completed source-isolated v14 contract required')
    adapter = S7MatcherAdapter.from_s7_m12(reference)
    model = S7Consensus(adapter, CompatibilityConfig.from_calibration(calibration),
                        head=training.fresh_head(TrainingConfig().head_seed))
    if arm == 'm12':
        for key, tensor in adapter.state_dict().items():
            if not torch.equal(tensor, cp['model']['matcher.' + key]):
                raise ValueError('M12 Matcher changed: ' + key)
        matcher_origin = dict(checkpoint=str(reference), sha256=sha(reference), selected_epoch=12)
    elif arm == 'scratch':
        matcher_origin = verify_scratch_matcher(root, binding, cp)
    else:
        matcher_origin = verify_fixed_matcher(root, binding, cp)
    # Scratch reference supplies architecture only: every model tensor is
    # overwritten strictly by the selected scratch checkpoint, no missing keys.
    model.load_state_dict(cp['model'], strict=True)
    model.eval().requires_grad_(False)
    model.matcher.set_frozen(True)
    protocol = registered_protocol(binding['config'])
    provenance = dict(arm=arm, variant=protocol['variant'], evidence_mode=protocol['evidence_mode'],
        checkpoint=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
        selected_epoch=selection['best']['epoch'], last_epoch=selection['actual_epochs'],
        threshold=selection['best']['threshold'], threshold_origin='synthetic CAL at preselected best joint epoch',
        stop_reason=selection['stop_reason'], data_contract_sha256=sha(contract_path),
        geometry_calibration_sha256=sha(calibration_path), selection_sha256=sha(directory / 'selection.json'),
        terminal_receipt_sha256=sha(root / ('formal_' + arm) / 'training_complete.json'),
        matcher_origin=matcher_origin,
        completion_scope='selected_arm' if completed_arm_only else 'all_arms',
        other_arm_metrics_used=False,
        training_implementation_sha256=binding['implementation_sha256'],
        model_selection_on_test_or_real=False, threshold_refitted=False,
        model_initialization_note=('historical M12 frozen' if arm == 'm12'
            else 'selected scratch Matcher and Scorer; reference used only to instantiate shapes'))
    return model, contract, provenance
