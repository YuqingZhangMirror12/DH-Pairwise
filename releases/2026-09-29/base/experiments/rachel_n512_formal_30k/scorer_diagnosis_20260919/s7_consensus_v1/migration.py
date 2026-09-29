"""Explicit C04 migration: retain scratch optimization, replace candidate selection.

No checkpoint is silently rebound. The original run stays immutable. A manifest
binds the imported optimizer/RNG state and every available weight reevaluation.
The new head always starts fresh. TEST and real data are never read here.
"""
import hashlib
import json
from pathlib import Path

import torch

from .config import Plateau
from .preflight_matcher import digest


SCHEMA = 's7-consensus-c04-authorized-migration/1'
ADDED_CONFIG_KEYS = {'proposal_revision', 'merge_repair_policy', 'validation_checkpoint_archive'}
CHANGED_ORIGINAL_FILES = {'pose_consensus.py', 'config.py', 'train.py', 'launch.py',
                          'preflight_consensus.py'}  # adds a zero-update diagnostic option only


def record_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('schema') != SCHEMA or plan.get('authorization') != 'repair_now_and_retrain':
        raise ValueError('explicit authorized C04 migration manifest required')
    if plan.get('head_policy') != 'fresh_same_original_seed' or plan.get('maximum_epochs') != 48:
        raise ValueError('migration cannot extend budget or continue the obsolete head')
    if plan.get('selection_policy') != 'reevaluate_available_weights_CAL_SELECT_only':
        raise ValueError('new decoder requires explicit simulation reselection')
    return plan


def validate_bindings(origin, target):
    """Allow only the disclosed merger/trainer/archive change, not a new loss."""
    for key in ('schema', 'reference_checkpoint_sha256', 'data_contract_sha256',
                'geometry_calibration_sha256', 'validation_contract', 'validation_design'):
        if origin[key] != target[key]:
            raise ValueError('migration changed protected binding: ' + key)
    if origin.get('arm') != 'scratch' or origin.get('formal_training') is not True:
        raise ValueError('only the original formal scratch Matcher may be imported')
    new_config = {k:v for k,v in target['config'].items() if k not in ADDED_CONFIG_KEYS}
    if new_config != origin['config']:
        raise ValueError('migration changed loss/network/data/batch/optimizer/budget configuration')
    old_files, new_files = origin['implementation_sha256'], target['implementation_sha256']
    for name, sha in old_files.items():
        if name not in CHANGED_ORIGINAL_FILES and new_files.get(name) != sha:
            raise ValueError('migration changed protected original source: ' + name)
    if new_files.get('legacy_pose_consensus.py') != old_files['pose_consensus.py']:
        raise ValueError('original candidate implementation must be retained byte-for-byte')


def load_bound(path_record, origin_binding):
    path = Path(path_record['path'])
    if digest(path) != path_record['sha256']:
        raise ValueError('preserved checkpoint bytes changed: ' + str(path))
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('binding') != origin_binding or saved.get('stage') != 'matcher':
        raise ValueError('preserved checkpoint belongs to a different experiment/stage')
    if saved['epoch'] != path_record['epoch']:
        raise ValueError('preserved epoch metadata differs')
    return saved


def validate_resume_state(saved, config):
    for key in ('optimizer', 'rng', 'plateau', 'epoch', 'offset', 'updates', 'exposures'):
        if key not in saved:
            raise ValueError('not a full optimizer/RNG checkpoint: ' + key)
    if saved['world_size'] != config.world_size or len(saved['rng']) != config.world_size:
        raise ValueError('migration must retain the two-rank topology')
    if saved['offset'] != 24000:
        raise ValueError('this reviewed migration requires an epoch-end training checkpoint')
    if saved['updates'] != saved['epoch'] * 24000 // config.effective_batch:
        raise ValueError('Matcher exposure/update accounting differs')
    if saved['exposures'] != saved['updates'] * config.effective_batch:
        raise ValueError('Matcher exposure accounting differs')
    if not 0 < saved['epoch'] < config.maximum_epochs:
        raise ValueError('no training budget remains for the repaired-decoder observation window')
    reductions = saved['plateau']['reductions']
    expected_lr = config.learning_rate * config.learning_rate_factor ** reductions
    if not 0 <= reductions <= config.maximum_lr_reductions:
        raise ValueError('invalid consumed LR reductions')
    if any(abs(group['lr'] - expected_lr) > 1e-14 for group in saved['optimizer']['param_groups']):
        raise ValueError('imported optimizer LR does not match the consumed schedule')


def load_migration(path, target_binding, config):
    plan = read_plan(path)
    origin_path = Path(plan['origin_config']['path'])
    if digest(origin_path) != plan['origin_config']['sha256']:
        raise ValueError('original experiment config changed')
    origin = json.loads(origin_path.read_text())
    if record_digest(origin) != plan['origin_binding_digest']:
        raise ValueError('original experiment binding differs')
    validate_bindings(origin, target_binding)
    saved = load_bound(plan['resume'], origin)
    validate_resume_state(saved, config)
    epochs = [r['epoch'] for r in plan['reevaluate']]
    if len(set(epochs)) != len(epochs) or epochs != sorted(epochs):
        raise ValueError('available evaluation epochs must be unique and ordered')
    if epochs[-1] != saved['epoch'] or plan['reevaluate'][-1] != plan['resume']:
        raise ValueError('the current imported Matcher must be reevaluated last')
    for row in plan['reevaluate']:
        if digest(row['path']) != row['sha256']:
            raise ValueError('available evaluation snapshot changed')
    return plan, origin, saved


def rebase_plateau(saved_plateau, new_baseline):
    # Old metric-specific patience cannot be mixed into the new decoder's
    # history. Consumed reductions and optimizer LR are NOT reset. Historical
    # re-evaluations select weights but do not pretend to replay past scheduling.
    return Plateau(best=float(new_baseline), bad=0,
                   reductions=int(saved_plateau['reductions']), since_reduction=0)
