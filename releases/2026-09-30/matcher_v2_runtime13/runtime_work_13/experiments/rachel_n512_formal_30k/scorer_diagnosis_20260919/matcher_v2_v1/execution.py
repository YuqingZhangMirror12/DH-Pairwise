"""Isolated B1--B3 CUDA entry: admission -> discarded gate -> fresh formal run.

No scheduling, retry, preemption or free-GPU discovery is performed. This entry
does not use the historical trainer's two-domain selection or legacy-only
Matcher export constructor. Its checkpoint/update machinery is reused intact.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time
import traceback
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..curriculum_training_v1.checkpoint_io import file_sha, load_committed, write_json
from ..curriculum_training_v1.execution import (CompleteCheckpoint, CoordinatedPause, archive_resume_leftovers,
    candidate_caches, check_gate, gate_record, rank0_call)
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.model_adapter import BASE, AdmittedDataset, bound_module, require
from ..curriculum_training_v1.runtime_io import ObservationWriter
from ..curriculum_training_v1.training_core import Topology, run_updates
from ..curriculum_training_v1.verify_validation_preparation import reference_architecture
from .data_runtime import CombinedDataset
from .gradient_gate import GradientProbe, check_gradient_receipt, gradient_receipt
from .model_runtime import make_components
from .runtime_inputs import load_inputs
from .runtime_io import export_completed, selected_matcher
from .validation import DunhuangDevelopment, bind_dunhuang, from_bound_baseline


def read(path):return json.loads(Path(path).read_text())


def prepare(args):
    """Fail closed on source/data/budget/architecture before touching CUDA."""
    spec = read(args.spec); values = load_inputs(spec); plan = values['plan']
    require(args.mode in ('gate', 'formal') and (args.gate_stop in (1, 12) if args.mode == 'gate' else args.gate_stop is None),
            'gate stop must be1 or12; formal has no artificial stop')
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1)); local = int(os.environ.get('LOCAL_RANK', 0))
    topology = Topology(rank=rank, **spec['topology'])
    require(world == topology.world_size and 0 <= rank < world and 0 <= local < world, 'launcher topology differs')
    base = values['base_execution']; source = values['source']
    os.environ['CURRICULUM_BASELINE_SOURCE'] = str(source)
    architecture = reference_architecture(base['reference_checkpoint']['path'])
    require(architecture.canvas_size == 800 and architecture.contour_cap == 512 and architecture.feature_dim == 96,
            'registered full800/N512 architecture required')
    api = bound_module(BASE+'s7_consensus_v1.compatibility', source)
    geometry = api.CompatibilityConfig.from_calibration(values['geometry'])
    selected = None
    if spec['selected_matcher'] is not None:
        choice = spec['selected_matcher']
        require(set(choice) == {'export_root', 'process_return', 'common_plan_sha256'}, 'explicit completed Matcher binding required')
        selected = selected_matcher(choice['export_root'], choice['process_return'], choice['common_plan_sha256'], spec['arm'])
    parts = make_components(plan, values['schedule'], topology, source, architecture, geometry, selected)
    formal = dict(parts['binding'], execution_manifest_sha256=file_sha(args.spec))
    names = sorted(name for name, p in parts['module'].named_parameters() if p.requires_grad)
    gate = None
    if args.mode == 'formal':
        require(args.gate_receipt is not None, 'formal training requires the matching discarded GPU gate')
        gate = check_gate(args.gate_receipt, formal)
        proof = read(args.gate_receipt)
        for name in ('gradient_uninterrupted', 'gradient_resumed_from_update1'):
            check_gradient_receipt(proof[name], formal, names, world)
    if spec['arm'] == 'B2':
        dataset = AdmittedDataset(base['admission']['path'], base['admission']['sha256'], values['base_ledger'], source)
    else:
        dataset = CombinedDataset(spec['combined_admission'], base['admission'], values['base_ledger'], values['ledger'], source)
    return spec, values, topology, local, parts, formal, gate, dataset


def run(args):
    spec, values, topology, local, parts, formal, gate, dataset = prepare(args)
    plan = values['plan']; ledger = values['ledger']; rank = topology.rank; world = topology.world_size
    root = Path(args.out).resolve(); binding = dict(formal, run_mode=args.mode)
    require(not (root/'training_complete.json').exists(), 'completed training must not be rerun')
    require(root.is_dir() if args.resume else not root.exists(), 'resume/new-output contract differs')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    random.seed(plan.record['seed']+rank); np.random.seed(plan.record['seed']+rank); torch.manual_seed(plan.record['seed']+rank)
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    if world > 1:dist.init_process_group('nccl', timeout=timedelta(hours=2))
    probe = None
    try:
        def initialize():
            root.mkdir(parents=True, exist_ok=args.resume)
            if args.resume:
                require(read(root/'binding.json') == binding, 'resume source/data/plan differs')
            else:
                write_json(root/'binding.json', binding)
                write_json(root/'initialization.json', dict(model_spec=binding['model_spec'], gate=gate,
                    actual_updates=0, gate_weights_imported=False, time_unix=time.time()))
        rank0_call(initialize, rank, world, 'initialize v2 output')
        raw = parts['module'].to(device)
        train_cache, eval_caches = candidate_caches(root/'proposal_cache', binding, values['contract'], rank, world, values['source'])
        raw.cache = train_cache
        wrapped = DistributedDataParallel(raw, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False) if world > 1 else raw
        optimizer = torch.optim.AdamW([p for p in raw.parameters() if p.requires_grad],
            lr=plan.learning_rate_knots[0][1], weight_decay=plan.record['weight_decay'])
        resume = load_committed(root/'checkpoints', binding, ledger, 'curriculum', topology) if args.resume else None
        if resume is not None:
            rank0_call(lambda: archive_resume_leftovers(root, resume), rank, world, 'preserve interrupted state')
        resumed_from = None if resume is None else resume['sampling']['completed_updates']
        if args.mode == 'gate':
            require(resumed_from is None or resumed_from == 1, 'gate may only replay the explicit update1 checkpoint')
            probe = GradientProbe(raw, root/'gradient_steps', formal, rank, world, device)
        evaluate = None
        if args.mode == 'formal':
            development = None
            if spec['module'] != 'matcher':
                path = values['base_execution']['real_split']['path']
                development = DunhuangDevelopment(path, bind_dunhuang(path), values['source'])
            evaluate = from_bound_baseline(plan, topology, parts['model'], values['contract'], device,
                parts['validation_config'], values['source'], ObservationWriter(root/'validation', plan, binding),
                real_development=development, caches=eval_caches)
        def status(row):
            if probe is not None:probe.update(row)
            if rank == 0 and (row['completed_updates'] % 25 == 0 or row['completed_updates'] == 1):
                counts = ledger.counts(row['completed_updates'])
                write_json(root/'status.json', dict(status='running_'+args.mode, arm=spec['arm'], module=spec['module'],
                    **row, original_update_clock=counts, total_updates=ledger.total_updates,
                    loss_scope='rank0 update mean, not validation', time_unix=time.time()), replace=True)
        checkpoint = CompleteCheckpoint(root/'checkpoints', rank, world,
            pause_path=root/'pause_request.json', binding_sha256=digest(binding))
        state = run_updates(wrapped, optimizer, dataset, ledger, 'curriculum', topology, device,
            plan.learning_rate_knots, () if args.mode == 'gate' else plan.validation_updates,
            collate_fn=parts['collate'], resume=resume, binding=binding,
            checkpoint_every=1 if args.mode == 'gate' else plan.record['checkpoint_every_updates'],
            gradient_clip_norm=plan.record['gradient_clip_norm'], evaluate=evaluate, on_update=status,
            on_checkpoint=checkpoint, stop_after=args.gate_stop)
        def finalize():
            if args.mode == 'gate':
                outcome = gate_record(root, state, formal, resumed_from)
                write_json(root/f'gate_update{args.gate_stop}.json', outcome)
                receipt = gradient_receipt(root/'gradient_steps', formal, probe.names, world, args.gate_stop)
                write_json(root/f'gradients_update{args.gate_stop}.json', receipt)
            else:
                out = root/'exports'
                if out.exists():out = root/('exports_attempt_'+uuid.uuid4().hex)
                outcome = export_completed(out, root/'checkpoints', plan, binding)
                outcome['export_root'] = str(out)
                write_json(root/'training_complete.json', outcome)
            write_json(root/'status.json', dict(outcome, time_unix=time.time()), replace=True)
        rank0_call(finalize, rank, world, 'finalize bound v2 run')
        return 0
    except CoordinatedPause as exc:
        if rank == 0:
            write_json(root/'pause_complete.json', dict(status='paused_at_committed_update', message=str(exc),
                binding_sha256=digest(binding), last_committed=read(root/'checkpoints/last_committed.json'),
                automatic_resume=False, time_unix=time.time()))
        return 75
    except Exception as exc:
        if rank == 0 and root.is_dir():
            write_json(root/f'failure_attempt_{time.time_ns()}.json', dict(status='failed', error=repr(exc),
                traceback=traceback.format_exc(), automatic_retry=False, time_unix=time.time()))
        raise
    finally:
        if probe is not None:probe.close()
        if world > 1 and dist.is_initialized():dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=('gate', 'formal'), required=True)
    p.add_argument('--gate-stop', type=int, choices=(1, 12)); p.add_argument('--gate-receipt', type=Path)
    p.add_argument('--resume', action='store_true')
    return run(p.parse_args())


if __name__ == '__main__':raise SystemExit(main())
