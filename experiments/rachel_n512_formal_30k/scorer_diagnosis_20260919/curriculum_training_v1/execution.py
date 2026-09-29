"""Bound update-based training entry, deliberately unavailable to unlocked plans.

No scheduler, free-GPU search, automatic retry, or budget inference lives here.
Run through a dedicated verified launcher only after data/schedule admission.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import re
import time
import traceback
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .checkpoint_io import (commit, file_sha, load_committed, save_rank, state_identity, tree_sha,
                            write_json)
from .exposure import STAGES, SampleRef, build_ledger, digest
from .model_adapter import AdmittedDataset, bound_module, make_components
from .runtime_io import ObservationWriter, checkpoint_at, export_completed, selected_curriculum_matcher
from .runtime_plan import lock_record
from .training_core import Topology, run_updates
from .validation_adapter import from_bound_baseline, validate_rule

BASE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def read_bound(spec):
    path = Path(spec['path'])
    require(path.is_absolute() and file_sha(path) == spec['sha256'], 'execution input file binding differs')
    return read(path)


def check_sources(spec, own=False):
    root = Path(spec['root']).resolve()
    if own:
        require(root == Path(__file__).resolve().parent, 'executing a different curriculum source')
    actual = {str(p.relative_to(root)): file_sha(p) for p in root.rglob('*.py')}
    require(actual and actual == spec['python_sha256'], 'bound implementation source changed')
    return root, digest(actual)


def load_inputs(spec):
    required = {'schema', 'locked', 'runtime_plan', 'admission', 'geometry', 'simulation_contract',
                'real_split', 'reference_checkpoint', 'implementation', 'baseline', 'topology',
                'selected_matcher'}
    require(set(spec) == required and spec['schema'] == 'curriculum-execution/1'
            and spec['locked'] is True, 'complete locked execution manifest required')
    implementation, _ = check_sources(spec['implementation'], own=True)
    baseline, baseline_sha = check_sources(spec['baseline'])
    admission = read_bound(spec['admission']); record = read_bound(spec['runtime_plan'])
    require(admission['status'] == 'passed' and admission['schema'] == 'curriculum-data-admission/1',
            'completed full data admission required')
    ledger = build_ledger([SampleRef(**r) for r in admission['catalog']], record['stage_updates'],
                          record['seed'], effective_batch=record['effective_batch'])
    require(record['data_admission_sha256'] == spec['admission']['sha256']
            and record['geometry_sha256'] == spec['geometry']['sha256']
            and record['baseline_sources_sha256'] == baseline_sha, 'plan data/geometry/source binding differs')
    plan = lock_record(record, ledger); validate_rule(plan)
    geometry = read_bound(spec['geometry'])
    require(geometry['status'] == 'complete' and geometry['schema'] == 's7-consensus-train-geometry/2'
            and geometry['curriculum_training_catalog_sha256'] == admission['catalog_sha256']
            and geometry['curriculum_data_admission_sha256'] == spec['admission']['sha256']
            and geometry['real_used'] is False and geometry['test_used'] is False,
            'shared all-TRAIN curriculum geometry calibration required')
    contract = read_bound(spec['simulation_contract'])
    require(contract['status'] == 'passed' and contract['source_disjoint'] is True
            and set(contract['validation']) == {'cal_mixed', 'select_mixed'},
            'fixed source-isolated simulation CAL/SELECT contract required')
    for name, expected_split in [('cal_mixed', 'cal'), ('select_mixed', 'select')]:
        view = read_bound(contract['validation'][name])
        require(view['split'] == expected_split and len(view['entries']) == contract['validation'][name]['pair_count'],
                'simulation validation population changed')
    reference = spec['reference_checkpoint']
    require(file_sha(reference['path']) == reference['sha256'], 'reference architecture checkpoint changed')
    expected_topology = (dict(world_size=2, microbatch=8, accumulate=2, workers=4)
        if record['module'] == 'matcher' else dict(world_size=1, microbatch=32, accumulate=1, workers=4))
    require(spec['topology'] == expected_topology,
            'unregistered Matcher/head training topology')
    require(ledger.effective_batch == 32, 'registered effective batch32 required')
    if record['module'] == 'matcher':
        require(spec['selected_matcher'] is None, 'new Matcher cannot import weights')
    else:
        require(isinstance(spec['selected_matcher'], dict), 'head requires completed curriculum Matcher')
    read_bound(spec['real_split'])
    return dict(baseline=baseline, implementation=implementation, admission=admission,
                plan=plan, ledger=ledger, geometry=geometry, contract=contract)


class CoordinatedPause(Exception):
    """An explicit external pause request was observed after a committed update."""


def rank0_call(fn, rank, world, label):
    """A rank0-only file operation must fail on every rank, not strand a barrier."""
    packet = None
    if rank == 0:
        try:
            packet = dict(ok=True, value=fn())
        except Exception as error:
            packet = dict(ok=False, error=type(error).__name__ + ': ' + str(error))
    if world > 1:
        require(dist.is_initialized() and dist.get_world_size() == world
                and dist.get_rank() == rank, 'rank0 operation process group differs')
        packets = [packet]; dist.broadcast_object_list(packets, src=0); packet = packets[0]
    if not packet['ok']:
        raise RuntimeError(label + ': ' + packet['error'])
    return packet['value']


def archive_resume_leftovers(root, state):
    """Explicit resume only: preserve unacknowledged files before replaying work.

    The caller has already verified the committed pointer on every rank. Newer
    shards, including a commit whose pointer update was interrupted, are NOT
    adopted. They are moved intact to a separate recovery-attempt directory.
    """
    root = Path(root); checkpoints = root / 'checkpoints'; binding = state['binding']
    pointer = read(checkpoints / 'last_committed.json')
    update = state['sampling']['completed_updates']
    require(pointer['completed_updates'] == update and pointer['relative_path'] == 'update_%06d/committed.json' % update
            and file_sha(checkpoints / pointer['relative_path']) == pointer['file_sha256'],
            'resume pointer changed before recovery')
    candidates = []
    for directory in sorted(checkpoints.glob('update_*')):
        require(re.fullmatch(r'update_[0-9]{6}', directory.name) is not None
                and directory.is_dir() and not directory.is_symlink(), 'unexpected checkpoint directory')
        cursor = int(directory.name[len('update_'):])
        if cursor <= update:
            continue
        require(cursor <= binding['common_plan']['total_updates'], 'uncommitted cursor exceeds registered budget')
        for shard in directory.glob('rank_*.pt'):
            require(not shard.is_symlink(), 'checkpoint shard symlink cannot be recovered')
            saved = torch.load(shard, map_location='cpu', weights_only=False)
            identity = state_identity(saved)
            require(saved['binding'] == binding and identity['completed_updates'] == cursor
                    and identity['world_size'] == state['sampling']['world_size']
                    and shard.name == 'rank_%02d.pt' % identity['rank'],
                    'uncommitted shard belongs to another experiment')
        if (directory / 'committed.json').exists():
            checkpoint_at(checkpoints, cursor, binding)
        candidates.append(directory)
    request = root / 'pause_request.json'; paused = root / 'pause_complete.json'
    if request.exists():
        value = read(request)
        require(value.get('binding_sha256') == digest(binding)
                and value.get('action') == 'pause_after_committed_update', 'unbound pause request on resume')
        candidates.append(request)
    if paused.exists():
        value = read(paused)
        require(value.get('binding_sha256') == digest(binding) and value.get('last_committed') == pointer,
                'pause receipt differs from the committed resume point')
        candidates.append(paused)
    if not candidates:
        return None
    inventory = []
    for source in candidates:
        paths = list(source.rglob('*')) if source.is_dir() else [source]
        require(not any(p.is_symlink() for p in paths), 'recovery must not follow symlinks')
        inventory.append(dict(original=str(source.relative_to(root)),
            files={str(p.relative_to(root)): file_sha(p) for p in paths if p.is_file()}))
    target = root / ('resume_preserved_' + uuid.uuid4().hex); target.mkdir()
    receipt = dict(status='planned', committed_pointer=pointer, binding_sha256=digest(binding),
        explicitly_requested_resume=True, adopted_newer_uncommitted_state=False, preserved=inventory)
    write_json(target / 'planned.json', receipt)
    for source in candidates:
        destination = target / source.relative_to(root); destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, destination)
    receipt['status'] = 'preserved'; write_json(target / 'complete.json', receipt)
    return dict(path=str(target.resolve()), sha256=file_sha(target / 'complete.json'))


class CompleteCheckpoint:
    """Propagate local save/commit failures before any rank waits at a barrier."""
    def __init__(self, root, rank, world, pause_path=None, binding_sha256=None):
        self.root = Path(root); self.rank = rank; self.world = world
        self.pause_path = None if pause_path is None else Path(pause_path)
        self.binding_sha256 = binding_sha256

    def gather(self, packet):
        if self.world == 1:
            return [packet]
        require(dist.is_initialized() and dist.get_world_size() == self.world
                and dist.get_rank() == self.rank, 'checkpoint process group differs')
        packets = [None] * self.world; dist.all_gather_object(packets, packet)
        return packets

    def broadcast(self, packet):
        if self.world == 1:
            return packet
        values = [packet]; dist.broadcast_object_list(values, src=0)
        return values[0]

    def __call__(self, state):
        require(state['rank'] == self.rank and state['sampling']['world_size'] == self.world,
                'checkpoint rank/topology differs')
        try:
            packet = dict(ok=True, receipt=save_rank(self.root, state))
        except Exception as error:
            packet = dict(ok=False, error=type(error).__name__ + ': ' + str(error), rank=self.rank)
        packets = self.gather(packet)
        if any(not p['ok'] for p in packets):
            raise RuntimeError('distributed checkpoint save failed: ' + repr([p for p in packets if not p['ok']]))
        result = None
        if self.rank == 0:
            try:
                pointer = commit(self.root, [p['receipt'] for p in packets])
                pause = False
                if self.pause_path is not None and self.pause_path.exists():
                    request = read(self.pause_path)
                    require(request.get('binding_sha256') == self.binding_sha256
                            and request.get('action') == 'pause_after_committed_update',
                            'unbound external pause request')
                    pause = True
                result = dict(ok=True, pointer=pointer, pause=pause)
            except Exception as error:
                result = dict(ok=False, error=type(error).__name__ + ': ' + str(error))
        result = self.broadcast(result)
        if not result['ok']:
            raise RuntimeError('distributed checkpoint commit failed: ' + result['error'])
        if result['pause']:
            raise CoordinatedPause('explicit pause after committed update%d' % state['sampling']['completed_updates'])
        return result['pointer']


def candidate_caches(root, binding, contract, rank, world, source):
    if binding['module'] == 'matcher':
        return None, None
    api = bound_module(BASE + 's7_consensus_v1.proposal_cache', source)
    views = dict(train=dict(sha256=binding['common_plan']['data_admission_sha256']), **contract['validation'])
    caches = {}
    for name, view in views.items():
        cache_binding = dict(experiment=binding, manifest_sha256=view['sha256'],
            matcher_state_sha256=binding['model_spec']['initial_matcher_state_sha256'],
            cached_content='deterministic geometry proposals only; fullQ/F/H/unmatched online')
        path = Path(root) / name
        rank0_call(lambda: bool(api.ProposalCache(path, cache_binding)), rank, world, 'initialize candidate cache')
        caches[name] = api.ProposalCache(path, cache_binding)
    return caches['train'], {k: v for k, v in caches.items() if k != 'train'}


def check_gate(path, formal_binding):
    """Both 12-update runs must match, including optimizer and per-rank RNG."""
    proof = read(path)
    require(proof['status'] == 'passed' and proof['formal_binding_sha256'] == digest(formal_binding),
            'matching verified GPU gate required before formal training')
    records = []
    for name in ('uninterrupted', 'resumed_from_update1'):
        item = proof[name]; record = read_bound(item)
        require(record['status'] == 'gpu_gate_complete' and record['updates'] == 12
                and record['formal_binding_sha256'] == digest(formal_binding)
                and record['active_module_changed'] is True and record['inactive_module_unchanged'] is True,
                'real update/freeze gate did not pass')
        pointer = read_bound(record['committed_pointer'])
        require(pointer['schema'] == 'curriculum-checkpoint-pointer/1'
                and pointer['completed_updates'] == 12
                and pointer['relative_path'] == 'update_000012/committed.json', 'gate cursor is not12')
        checkpoints = Path(record['committed_pointer']['path']).parent
        require(file_sha(checkpoints / pointer['relative_path']) == pointer['file_sha256'],
                'gate committed manifest changed')
        state, _ = checkpoint_at(checkpoints, 12, dict(formal_binding, run_mode='gate'))
        require(gate_record(checkpoints.parent, state, formal_binding, record['resumed_from_update']) == record,
                'gate receipt is not supported by its actual checkpoint state')
        records.append(record)
    require(records[0]['shared_state_sha256'] == records[1]['shared_state_sha256']
            and records[0]['rank_full_state_sha256'] == records[1]['rank_full_state_sha256']
            and records[1]['resumed_from_update'] == 1
            and records[0]['resumed_from_update'] is None, 'update1→12 replay differs')
    return dict(path=str(Path(path).resolve()), sha256=file_sha(path), status='passed')


def gate_record(root, state, formal_binding, resumed_from):
    require(state['rng']['cuda'] is not None, 'CPU state is not a genuine CUDA gate')
    require(state['binding'] == dict(formal_binding, run_mode='gate') and not state['observations']
            and state['sampling']['completed_updates'] in (1, 12), 'invalid gate state or validation history')
    pointer_path = Path(root) / 'checkpoints' / 'last_committed.json'
    pointer = read(pointer_path); manifest = read(Path(root) / 'checkpoints' / pointer['relative_path'])
    spec = formal_binding['model_spec']; trained_matcher = formal_binding['module'] == 'matcher'
    matcher = {k[len('model.matcher.'):]: v for k, v in state['model'].items() if k.startswith('model.matcher.')}
    head = {k[len('model.head.'):]: v for k, v in state['model'].items() if k.startswith('model.head.')}
    matcher_changed = tree_sha(matcher) != spec['initial_matcher_state_sha256']
    head_changed = tree_sha(head) != spec['initial_head_state_sha256']
    active = matcher_changed if trained_matcher else head_changed
    inactive = not head_changed if trained_matcher else not matcher_changed
    require(active and inactive, 'actual gate did not update only its intended module')
    return dict(status='gpu_gate_complete', formal_binding_sha256=digest(formal_binding),
        updates=state['sampling']['completed_updates'], exposures=state['sampling']['completed_exposures'],
        resumed_from_update=resumed_from, active_module_changed=active, inactive_module_unchanged=inactive,
        shared_state_sha256=manifest['ranks'][0]['shared_state_sha256'],
        rank_full_state_sha256=[r['full_state_sha256'] for r in manifest['ranks']],
        committed_pointer=dict(path=str(pointer_path.resolve()), sha256=file_sha(pointer_path)),
        formal_training_started=False, short_test_weights_not_formal=True)


def run(args):
    spec = read(args.spec); inputs = load_inputs(spec); plan = inputs['plan']; ledger = inputs['ledger']
    require(args.order in ('curriculum', 'mixed') and not (args.order == 'mixed' and plan.record['module'] != 'matcher'),
            'unregistered presentation order')
    require(args.mode in ('gate', 'formal') and (args.gate_stop in (1, 12) if args.mode == 'gate' else args.gate_stop is None),
            'gate requires1 or12 real updates; formal has no artificial stop')
    require(plan.record['total_updates'] >= 12, 'formal registered schedule too short for the required gate')
    from .verify_validation_preparation import bind_baseline, reference_architecture
    bind_baseline(inputs['baseline'])
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1)); local = int(os.environ.get('LOCAL_RANK', 0))
    topology = Topology(rank=rank, **spec['topology'])
    require(world == topology.world_size and 0 <= rank < world and 0 <= local < world,
            'launcher distributed topology differs')
    root = Path(args.out).resolve()
    require(not (root / 'training_complete.json').exists(),
            'completed training must not be rerun')
    require(args.resume or not root.exists(), 'existing training output requires explicit resume')
    if args.resume:
        require(root.is_dir(), 'no training output to resume')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    random.seed(plan.record['seed'] + rank); np.random.seed(plan.record['seed'] + rank)
    torch.manual_seed(plan.record['seed'] + rank)
    prefix = BASE + 's7_consensus_v1.'
    architecture = reference_architecture(spec['reference_checkpoint']['path'])
    geometry_api = bound_module(prefix + 'compatibility', inputs['baseline'])
    geometry = geometry_api.CompatibilityConfig.from_calibration(inputs['geometry'])
    selected = None
    if spec['selected_matcher'] is not None:
        s = spec['selected_matcher']
        selected = selected_curriculum_matcher(s['export_root'], s['process_return'], s['common_plan_sha256'])
    parts = make_components(plan, args.order, topology, inputs['baseline'], architecture, geometry, selected)
    formal_binding = dict(parts['binding'], execution_manifest_sha256=file_sha(args.spec))
    gate = None
    if args.mode == 'formal':
        require(args.gate_receipt is not None, 'formal training requires its verified GPU gate')
        gate = check_gate(args.gate_receipt, formal_binding)
    binding = dict(formal_binding, run_mode=args.mode)
    dataset = AdmittedDataset(spec['admission']['path'], spec['admission']['sha256'], ledger, inputs['baseline'])
    # All files, explicit topology and formal gate are checked before CUDA.
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    try:
        def initialize():
            root.mkdir(parents=True, exist_ok=args.resume)
            if args.resume:
                require(read(root / 'binding.json') == binding, 'resume implementation/data/plan binding differs')
            else:
                write_json(root / 'binding.json', binding)
                write_json(root / 'initialization.json', dict(model_spec=binding['model_spec'], gate=gate,
                    actual_updates=0, gate_weights_imported=False, time_unix=time.time()))
        rank0_call(initialize, rank, world, 'initialize bound training output')
        parts['module'].to(device)
        train_cache, eval_caches = candidate_caches(root / 'proposal_cache', binding, inputs['contract'], rank, world, inputs['baseline'])
        parts['module'].cache = train_cache
        raw = parts['module']
        wrapped = DistributedDataParallel(raw, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False) if world > 1 else raw
        optimizer = torch.optim.AdamW([p for p in raw.parameters() if p.requires_grad],
            lr=plan.learning_rate_knots[0][1], weight_decay=plan.record['weight_decay'])
        resume = load_committed(root / 'checkpoints', binding, ledger, args.order, topology) if args.resume else None
        if resume is not None:
            rank0_call(lambda: archive_resume_leftovers(root, resume), rank, world, 'preserve interrupted resume files')
        resumed_from = None if resume is None else resume['sampling']['completed_updates']
        evaluate = None
        if args.mode == 'formal':
            development = None
            if plan.record['module'] != 'matcher':
                real_api = bound_module(prefix + 'real_development', inputs['baseline'])
                development = real_api.RealDevelopment(spec['real_split']['path'], real_api.bind_plan(spec['real_split']['path']))
            evaluate = from_bound_baseline(plan, topology, parts['model'], inputs['contract'], device,
                parts['validation_config'], inputs['baseline'], ObservationWriter(root / 'validation', plan, binding),
                real_development=development, caches=eval_caches)
        def status(row):
            if rank == 0 and (row['completed_updates'] % 25 == 0 or row['completed_updates'] == 1):
                write_json(root / 'status.json', dict(status='running_' + args.mode, module=plan.record['module'],
                    order=args.order, **row, loss_scope='rank0 mean over this update, not global validation',
                    total_updates=plan.record['total_updates'], time_unix=time.time()), replace=True)
        checkpoint = CompleteCheckpoint(root / 'checkpoints', rank, world,
            pause_path=root / 'pause_request.json', binding_sha256=digest(binding))
        state = run_updates(wrapped, optimizer, dataset, ledger, args.order, topology, device,
            plan.learning_rate_knots, () if args.mode == 'gate' else plan.validation_updates,
            collate_fn=parts['collate'], resume=resume, binding=binding,
            checkpoint_every=1 if args.mode == 'gate' else plan.record['checkpoint_every_updates'],
            gradient_clip_norm=plan.record['gradient_clip_norm'], evaluate=evaluate, on_update=status,
            on_checkpoint=checkpoint, stop_after=args.gate_stop)
        def finalize():
            if args.mode == 'gate':
                outcome = gate_record(root, state, formal_binding, resumed_from)
                write_json(root / ('gate_update%d.json' % args.gate_stop), outcome)
            else:
                export_root = root / 'exports'
                if export_root.exists():
                    # Preserve an interrupted export attempt; the committed
                    # training state does not need to be trained again.
                    export_root = root / ('exports_attempt_' + uuid.uuid4().hex)
                outcome = export_completed(export_root, root / 'checkpoints', plan, binding)
                outcome['export_root'] = str(export_root.resolve())
                write_json(root / 'training_complete.json', outcome)
            write_json(root / 'status.json', dict(outcome, time_unix=time.time()), replace=True)
        rank0_call(finalize, rank, world, 'finalize committed training output')
        return 0
    except CoordinatedPause as error:
        if rank == 0:
            write_json(root / 'pause_complete.json', dict(status='paused_at_committed_update', message=str(error),
                binding_sha256=digest(binding), last_committed=read(root / 'checkpoints' / 'last_committed.json'),
                automatic_resume=False, time_unix=time.time()))
        return 75
    except Exception as error:
        if rank == 0 and root.is_dir():
            write_json(root / ('failure_attempt_' + str(time.time_ns()) + '.json'), dict(status='failed',
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                automatic_retry=False, time_unix=time.time()))
        raise
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--order', choices=('curriculum', 'mixed'), required=True)
    parser.add_argument('--mode', choices=('gate', 'formal'), required=True)
    parser.add_argument('--gate-stop', type=int, choices=(1, 12))
    parser.add_argument('--gate-receipt', type=Path)
    parser.add_argument('--resume', action='store_true')
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
