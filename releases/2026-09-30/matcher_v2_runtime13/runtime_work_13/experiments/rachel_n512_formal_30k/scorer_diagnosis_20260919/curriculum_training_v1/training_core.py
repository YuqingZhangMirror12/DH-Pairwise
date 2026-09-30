"""Update-based loop for an already-bound Matcher or binary-head TrainModule.

No model construction, data admission, selection rule, launch, or retry lives
here. Callers must supply the locked ledger/LR/validation plan. Full formal
admission and the dedicated controller remain separate preparation work.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .exposure import RankMicrobatches, integer, learning_rate_at


@dataclass(frozen=True)
class Topology:
    rank: int
    world_size: int
    microbatch: int
    accumulate: int
    workers: int = 0


def capture_rng(device):
    device = torch.device(device)
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None)


def restore_rng(value, device):
    device = torch.device(device)
    if (value['cuda'] is None) != (device.type != 'cuda'):
        raise ValueError('RNG record/device mismatch')
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state(value['cuda'].cpu(), device)


def move_batch(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_batch(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_batch(x, device) for x in value)
    if isinstance(value, list):
        return [move_batch(x, device) for x in value]
    return value


def evaluate_without_changing_training_rng(callback, completed, device):
    before = capture_rng(device)
    try:
        return callback(completed)
    finally:
        restore_rng(before, device)


def run_updates(module, optimizer, dataset, ledger, order, topology, device,
                learning_rate_knots, validation_updates, collate_fn=None,
                resume=None, binding=None, checkpoint_every=100, gradient_clip_norm=5.,
                evaluate=None, on_update=None, on_checkpoint=None, stop_after=None):
    """FP32 full optimizer steps, with a cursor that excludes DataLoader prefetch.

    module returns (scalar_loss, component_dict, count_dict), matching the
    existing training wrapper. DDP must already be initialized/wrapped outside.
    Callbacks execute on every rank; the caller gathers/writes only as appropriate.
    They must not change parameters, optimizer state or the locked schedule.
    stop_after is a bounded preparation-test hook, not a formal early-stop rule.
    """
    device = torch.device(device)
    if len(dataset) != len(ledger.catalog):
        raise ValueError('dataset/catalog index mapping length mismatch')
    if binding is None:
        raise ValueError('explicit implementation/data/protocol binding required')
    integer(topology.workers, 'workers'); integer(checkpoint_every, 'checkpoint interval', 1)
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
        raise ValueError('positive finite gradient clip required')
    marks = tuple(validation_updates)
    for update in marks:
        integer(update, 'validation update')
    if tuple(sorted(set(marks))) != marks or (marks and marks[-1] > ledger.total_updates):
        raise ValueError('validation schedule must be explicit, sorted, unique and in budget')
    if bool(marks) != (evaluate is not None):
        raise ValueError('validation callback and locked observations must agree')
    # Validate every LR knot before any optimizer update.
    learning_rate_at(0, learning_rate_knots)
    if any(at >= ledger.total_updates for at, _ in learning_rate_knots):
        raise ValueError('LR knot outside actual update range')
    if stop_after is not None:
        integer(stop_after, 'bounded stop', 1)
    end = ledger.total_updates if stop_after is None else min(stop_after, ledger.total_updates)
    raw = module.module if hasattr(module, 'module') else module
    parameters = [p for p in raw.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 or p.device != device for p in parameters):
        raise ValueError('trainable parameters must be FP32 on the registered device')
    if topology.world_size > 1 and not hasattr(module, 'no_sync'):
        raise ValueError('multi-rank execution requires an already-wrapped DDP module')
    groups = [p for group in optimizer.param_groups for p in group['params']]
    if len(groups) != len(parameters) or {id(p) for p in groups} != {id(p) for p in parameters}:
        raise ValueError('optimizer must contain exactly the intended trainable parameters')
    if resume is None:
        completed = 0; observations = []
        sampler = RankMicrobatches(ledger, order, 0, topology.rank, topology.world_size,
                                  topology.microbatch, topology.accumulate)
    else:
        if resume.get('binding') != binding or resume.get('rank') != topology.rank:
            raise ValueError('checkpoint implementation/data/protocol/rank binding changed')
        sampler = RankMicrobatches.from_cursor(ledger, resume['sampling'], topology.rank, order,
                                              topology.world_size, topology.microbatch, topology.accumulate)
        completed = sampler.completed; observations = list(resume['observations'])
        expected = [x for x in marks if x <= completed]
        if [x['update'] for x in observations] != expected:
            raise ValueError('checkpoint has missing/extra validation observations')
        raw.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer']); restore_rng(resume['rng'], device)
    if end < completed:
        raise ValueError('cannot run backward from a completed checkpoint')

    def state():
        return dict(binding=binding, rank=topology.rank, model=raw.state_dict(),
                    optimizer=optimizer.state_dict(), sampling=sampler.cursor(completed),
                    rng=capture_rng(device), observations=list(observations))

    def validate_now():
        if completed in marks:
            value = evaluate_without_changing_training_rng(evaluate, completed, device)
            observations.append(dict(update=completed, report=value))
            module.train()

    module.train()
    if not resume:
        validate_now()
        if on_checkpoint is not None:
            on_checkpoint(state())
    if completed == end:
        return state()
    generator = torch.Generator().manual_seed(ledger.seed + topology.rank + completed)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn,
                        num_workers=topology.workers, pin_memory=device.type == 'cuda', generator=generator)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.; component_sums = {}; count_sums = {}
    for local_micro, batch in enumerate(loader):
        lr = learning_rate_at(completed, learning_rate_knots)
        for group in optimizer.param_groups:
            group['lr'] = lr
        sync = (local_micro + 1) % topology.accumulate == 0
        context = module.no_sync() if topology.world_size > 1 and not sync else nullcontext()
        with context:
            loss, components, counts = module(move_batch(batch, device))
            if loss.ndim or not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite or nonscalar training loss')
            (loss / topology.accumulate).backward()
        loss_sum += float(loss.detach()) / topology.accumulate
        for key, value in components.items():
            component_sums[key] = component_sums.get(key, 0.) + float(value) / topology.accumulate
        for key, value in counts.items():
            count_sums[key] = count_sums.get(key, 0) + int(value)
        if not sync:
            continue
        norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('nonfinite gradient norm')
        optimizer.step(); optimizer.zero_grad(set_to_none=True); completed += 1
        if on_update is not None:
            on_update(dict(completed_updates=completed, exposures=completed * ledger.effective_batch,
                           learning_rate=lr, rank=topology.rank, loss=loss_sum,
                           components=component_sums, counts=count_sums))
        loss_sum = 0.; component_sums = {}; count_sums = {}
        validate_now()
        if on_checkpoint is not None and (completed % checkpoint_every == 0 or completed in marks or completed == end):
            on_checkpoint(state())
        if completed == end:
            break
    if completed != end:
        raise ValueError('loader ended before the registered complete-update boundary')
    return state()
