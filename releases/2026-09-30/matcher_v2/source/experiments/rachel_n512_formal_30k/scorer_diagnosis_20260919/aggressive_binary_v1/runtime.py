"""Explicit matcher/scorer stages; no E32 or trained Scorer import."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import traceback
import time

import numpy as np
import torch
import torch.distributed as dist

from .admission import read, sha, need, validate_data
from ..binary_scorer_v1.head import BinaryClusterHead
from ..binary_scorer_v1.model import BinaryConsensus
from ..s7_consensus_v1.config import TrainingConfig
from ..s7_consensus_v1.compatibility import CompatibilityConfig
from ..s7_consensus_v1.frozen_start import load_selected_matcher
from ..s7_consensus_v1.matcher import S7MatcherAdapter
from ..s7_consensus_v1.preflight_matcher import state_digest
from ..s7_consensus_v1.real_development import bind_plan
from ..s7_consensus_v1.scratch_matcher import fresh_matcher


def fresh_patch_head(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return BinaryClusterHead('patch')


def make_model(reference_config, geometry, config, stage, selected=None):
    need(stage in ('matcher', 'scorer'), 'explicit stage required')
    matcher = fresh_matcher(reference_config, config.matcher_seed)
    receipt = dict(initialization='random_seed_no_weight_import', seed=config.matcher_seed,
                   initial_state_sha256=state_digest(matcher), reference_weights_imported=False,
                   optimizer_imported=False, old_head_imported=False)
    if stage == 'scorer':
        need(selected is not None, 'scorer requires its NEW selected Matcher')
        receipt = dict(initialization='new_aggressive_selected_matcher',
                       **load_selected_matcher(matcher, selected['path'], selected['sha256']))
    else:
        need(selected is None, 'Matcher stage cannot import any selected checkpoint')
    head = fresh_patch_head(config.head_seed)
    model = BinaryConsensus(matcher, geometry, head=head)
    matcher.set_frozen(stage == 'scorer')
    head.requires_grad_(stage == 'scorer')
    return model, receipt


def selected_matcher(stage_dir, binding, config):
    root = Path(stage_dir)
    need(not (root / 'failure.json').exists() and not (root.parent / 'failure_matcher.json').exists(),
         'Matcher stage failure is not a completion')
    complete = read(root / 'complete.json')
    selection = read(root / 'selection.json')
    need(complete.get('status') == 'stage_complete' and selection.get('status') == 'selected',
         'the NEW Matcher must finish its registered training budget first')
    for key in ('data_contract_sha256', 'geometry_calibration_sha256', 'admission',
                'implementation_sha256', 'aggressive_implementation_sha256', 'binary_scorer_sha256', 'config'):
        need(selection['binding'][key] == binding[key], 'Matcher stage binding differs: ' + key)
    need(selection['binding'].get('stage') == 'matcher'
         and selection['binding'].get('matcher_initialization') == 'random_seed_no_weight_import'
         and selection.get('selection_on_real') is False
         and selection.get('migration_origin') is None, 'not a new simulation-selected Matcher')
    need(config.minimum_epochs <= selection['actual_epochs'] <= config.maximum_epochs
         and selection['updates'] == selection['actual_epochs'] * 24000 // config.effective_batch
         and selection['exposures'] == selection['actual_epochs'] * 24000,
         'Matcher epoch/update/exposure completion is inconsistent')
    for key in ('best', 'best_joint_sha256', 'actual_epochs', 'updates', 'exposures', 'binding', 'stop_reason'):
        need(complete[key] == selection[key], 'Matcher terminal receipts differ: ' + key)
    curve = read(root / 'learning_curve.json')
    need([r['epoch'] for r in curve] == list(range(0, selection['actual_epochs'] + 1, 2)),
         'Matcher validation history incomplete')
    winner = max((r for r in curve if r['epoch'] > 0), key=lambda r: tuple(r['key']))
    need(winner['epoch'] == selection['best']['epoch']
         and winner['key'] == selection['best']['key'], 'Matcher selection differs from native coverage curve')
    checkpoint = root / 'best_joint.pt'
    need(sha(checkpoint) == selection['best_joint_sha256'], 'selected Matcher file changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    need(saved['binding'] == selection['binding'] and saved['stage'] == 'matcher'
         and saved['epoch'] == selection['best']['epoch'], 'selected Matcher state identity differs')
    return dict(path=str(checkpoint.resolve()), sha256=sha(checkpoint), epoch=saved['epoch'],
                selection_sha256=sha(root / 'selection.json'),
                terminal_sha256=sha(root / 'complete.json'), old_head_imported=False,
                optimizer_imported=False, matcher_training=False)


def make_binding(args, config, contract):
    admission = validate_data(args.contract, args.calibration, args.review_approval)
    package = Path(__file__).parent
    consensus = package.parent / 's7_consensus_v1'
    binding = dict(schema='aggressive-binary-experiment/1', arm='scratch_aggressive', stage=args.stage,
        reference_checkpoint_sha256=sha(args.checkpoint), reference_checkpoint_use='architecture only',
        data_contract_sha256=sha(args.contract), geometry_calibration_sha256=sha(args.calibration),
        admission=admission, config=config.record(),
        implementation_sha256={p.name: sha(p) for p in consensus.glob('*.py')},
        aggressive_implementation_sha256={p.name: sha(p) for p in package.glob('*.py')},
        binary_scorer_sha256={p.name: sha(p) for p in (package.parent / 'binary_scorer_v1').glob('*.py')},
        matcher_initialization='random_seed_no_weight_import',
        validation_contract=contract['validation'], validation_design=contract['validation_design'],
        real_development=bind_plan(args.real_split),
        real_evaluation_stage='scorer only; Matcher remains SIM native-coverage selected',
        formal_training=not bool(args.preflight_steps), preflight_steps=args.preflight_steps)
    binding = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
    if args.stage == 'scorer':
        need(args.selected_matcher_stage, 'explicit completed NEW Matcher stage required')
        binding['fixed_matcher'] = selected_matcher(args.selected_matcher_stage, binding, config)
    else:
        need(not args.selected_matcher_stage, 'random Matcher stage forbids imported weights')
    return binding


def run(args):
    from ..s7_consensus_v1.train import run_stage, save_json, barrier
    need(args.stage in ('matcher', 'scorer') and args.preflight_steps in (0, 12),
         'only registered formal or 12-update disposable stages are supported')
    config = TrainingConfig(scorer_variant='patch')
    need(config.schema == 'aggressive-binary-training/1', 'use the prepared experiment3 source only')
    args.arm = 'scratch_aggressive'
    contract = read(args.contract)
    # Data and human-approval gates precede any CUDA initialization.
    binding = make_binding(args, config, contract)
    root = Path(args.out)
    if args.stage == 'scorer' and not args.preflight_steps:
        need(Path(args.selected_matcher_stage).resolve() == (root / 'matcher').resolve(),
             'formal Scorer must use this experiment\'s own completed Matcher stage')
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    need(world == config.world_size, 'registered topology is two GPUs, effective batch32')
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        random.seed(config.data_seed + rank); np.random.seed(config.data_seed + rank)
        torch.manual_seed(config.data_seed + rank)
        if rank == 0:
            root.mkdir(parents=True, exist_ok=True)
            manifest = root / ('CONFIG_' + args.stage + '.json')
            need(not manifest.exists() or read(manifest) == binding, 'experiment stage identity changed')
            save_json(manifest, binding)
        barrier()
        geometry = CompatibilityConfig.from_calibration(read(args.calibration))
        reference = S7MatcherAdapter.from_s7_m12(args.checkpoint)
        reference_config = reference.base.config
        del reference  # No trained state tensor is passed to fresh_matcher.
        model, imported = make_model(reference_config, geometry, config, args.stage, binding.get('fixed_matcher'))
        if rank == 0:
            save_json(root / (args.stage + '_initialization.json'), imported)
        model = model.to(device)
        outcome = run_stage(model, args.stage, args, config, contract, binding, device, rank, world)
        if rank == 0 and not args.preflight_steps:
            if args.stage == 'matcher':
                save_json(root / 'matcher_training_complete.json', dict(status='matcher_complete_scorer_pending',
                    stage=outcome, binding=binding, whole_experiment_complete=False))
            else:
                save_json(root / 'training_complete.json', dict(status='training_complete', arm=args.arm,
                    stages=['matcher', 'scorer'], selected_matcher=binding['fixed_matcher'],
                    last_stage=outcome, binding=binding, real_evaluation_pending=True))
    except Exception as error:
        if rank == 0 and root.is_dir():
            save_json(root / ('failure_' + args.stage + '.json'), dict(status='failed', time_unix=time.time(),
                type=type(error).__name__, message=str(error), traceback=traceback.format_exc(), binding=binding))
        raise
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('matcher', 'scorer'), required=True)
    for name in ('out', 'checkpoint', 'contract', 'calibration', 'review-approval', 'real-split'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--selected-matcher-stage')
    parser.add_argument('--preflight-steps', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
