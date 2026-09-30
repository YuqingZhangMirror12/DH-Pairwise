"""Commit only complete, mutually consistent multi-rank update checkpoints.

Files from an interrupted save are retained but never become the resume pointer.
This is checkpoint infrastructure, not a launcher or permission to resume jobs.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct

import numpy as np
import torch

from .exposure import RankMicrobatches, digest, integer


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tree_sha(value):
    """Deterministic value/dtype/shape hash, independent of torch.save metadata."""
    h = hashlib.sha256()
    def block(tag, raw):
        h.update(tag); h.update(len(raw).to_bytes(8, 'little')); h.update(raw)
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            block(b'T', json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
            block(b'V', tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError('object arrays cannot be checkpointed')
            block(b'N', json.dumps([item.dtype.str, list(item.shape)]).encode())
            block(b'V', np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            keys = sorted(item, key=lambda k: (type(k).__name__, repr(k)))
            block(b'D', str(len(keys)).encode())
            for key in keys:
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            block(b'U' if isinstance(item, tuple) else b'L', str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None:
            block(b'0', b'')
        elif type(item) is bool:
            block(b'B', str(item).encode())
        elif type(item) is int:
            block(b'I', str(item).encode())
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError('nonfinite checkpoint scalar')
            block(b'F', struct.pack('!d', item))
        elif isinstance(item, str):
            block(b'S', item.encode())
        elif isinstance(item, bytes):
            block(b'Y', item)
        else:
            raise TypeError('unsupported checkpoint value: ' + type(item).__name__)
    visit(value)
    return h.hexdigest()


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def write_json(path, value, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def state_identity(state):
    required = {'binding', 'rank', 'model', 'optimizer', 'sampling', 'rng', 'observations'}
    if set(state) != required or not state['binding']:
        raise ValueError('explicit complete update state required')
    sampling = state['sampling']
    rank = integer(state['rank'], 'rank')
    world = integer(sampling['world_size'], 'world size', 1)
    update = integer(sampling['completed_updates'], 'completed updates')
    if rank >= world:
        raise ValueError('checkpoint rank outside registered topology')
    batch = world * integer(sampling['microbatch'], 'microbatch', 1) * integer(sampling['accumulate'], 'accumulate', 1)
    if sampling['completed_exposures'] != update * batch:
        raise ValueError('checkpoint cursor and exposures differ')
    observations = [row['update'] for row in state['observations']]
    if observations != sorted(set(observations)) or any(x > update for x in observations):
        raise ValueError('checkpoint observations exceed completed updates')
    shared = {k: state[k] for k in ('binding', 'model', 'optimizer', 'sampling', 'observations')}
    return dict(rank=rank, world_size=world, completed_updates=update,
                binding_sha256=digest(state['binding']), shared_state_sha256=tree_sha(shared),
                full_state_sha256=tree_sha(state), sampling=sampling)


def save_rank(root, state):
    """Each rank writes once; no committed pointer changes here."""
    state = cpu_copy(state); identity = state_identity(state)
    directory = Path(root) / ('update_%06d' % identity['completed_updates'])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('rank_%02d.pt' % identity['rank'])
    if path.exists():
        raise FileExistsError('retaining an earlier checkpoint shard: ' + str(path))
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('xb') as stream:
        torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    return dict(**identity, relative_path=str(path.relative_to(root)), file_sha256=file_sha(path))


def verified_state(root, receipt):
    update = integer(receipt['completed_updates'], 'completed updates')
    rank = integer(receipt['rank'], 'rank')
    relative = 'update_%06d/rank_%02d.pt' % (update, rank)
    if receipt['relative_path'] != relative:
        raise ValueError('checkpoint shard path escaped its registered update/rank')
    path = Path(root) / relative
    if file_sha(path) != receipt['file_sha256']:
        raise ValueError('checkpoint shard hash changed')
    state = torch.load(path, map_location='cpu', weights_only=False)
    expected = dict(**state_identity(state), relative_path=relative, file_sha256=receipt['file_sha256'])
    if expected != receipt:
        raise ValueError('checkpoint shard does not match its receipt')
    return state


def commit(root, receipts):
    """Rank0 commits only after every rank successfully saved the same update."""
    root = Path(root); receipts = sorted(receipts, key=lambda x: x['rank'])
    if not receipts:
        raise ValueError('cannot commit zero checkpoint ranks')
    first = receipts[0]; world = first['world_size']; update = first['completed_updates']
    if [x['rank'] for x in receipts] != list(range(world)):
        raise ValueError('missing or duplicated checkpoint ranks')
    common = ('world_size', 'completed_updates', 'binding_sha256', 'shared_state_sha256', 'sampling')
    for item in receipts:
        if any(item[key] != first[key] for key in common):
            raise ValueError('ranks disagree on model/optimizer/observations or sampling')
        verified_state(root, item)
    pointer = root / 'last_committed.json'
    if pointer.exists():
        previous = json.loads(pointer.read_text())
        if previous['completed_updates'] >= update:
            raise ValueError('checkpoint pointer cannot move backward or overwrite an update')
        expected = 'update_%06d/committed.json' % integer(previous['completed_updates'], 'previous update')
        if (previous.get('schema') != 'curriculum-checkpoint-pointer/1'
                or previous['relative_path'] != expected or file_sha(root / expected) != previous['file_sha256']):
            raise ValueError('previous committed pointer changed')
        previous_manifest = json.loads((root / expected).read_text())
        sampler_keys = ('schema', 'ledger_sha256', 'order', 'world_size', 'microbatch', 'accumulate')
        if (previous_manifest['binding_sha256'] != first['binding_sha256']
                or any(previous_manifest['sampling'][k] != first['sampling'][k] for k in sampler_keys)):
            raise ValueError('checkpoint output root belongs to a different experiment')
    manifest = dict(schema='curriculum-checkpoint-commit/1', completed_updates=update,
                    world_size=world, binding_sha256=first['binding_sha256'],
                    sampling=first['sampling'], ranks=receipts)
    path = root / ('update_%06d' % update) / 'committed.json'
    write_json(path, manifest)
    value = dict(schema='curriculum-checkpoint-pointer/1', completed_updates=update,
                 relative_path=str(path.relative_to(root)), file_sha256=file_sha(path))
    write_json(pointer, value, replace=True)
    return value


def load_committed(root, binding, ledger, order, topology):
    """Explicit resume uses only the committed pointer, not the newest shard."""
    root = Path(root); pointer = json.loads((root / 'last_committed.json').read_text())
    update = integer(pointer['completed_updates'], 'completed updates')
    relative = 'update_%06d/committed.json' % update
    if pointer.get('schema') != 'curriculum-checkpoint-pointer/1' or pointer['relative_path'] != relative:
        raise ValueError('invalid committed checkpoint pointer')
    path = root / relative
    if file_sha(path) != pointer['file_sha256']:
        raise ValueError('committed checkpoint manifest hash changed')
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'curriculum-checkpoint-commit/1'
            or manifest['completed_updates'] != update
            or manifest['binding_sha256'] != digest(binding)
            or manifest['world_size'] != topology.world_size):
        raise ValueError('committed checkpoint binding/topology changed')
    receipts = manifest['ranks']
    if [x['rank'] for x in receipts] != list(range(topology.world_size)):
        raise ValueError('committed checkpoint is missing a rank')
    # All files must still exist and match before accepting any one rank.
    for item in receipts:
        expected = 'update_%06d/rank_%02d.pt' % (update, item['rank'])
        if (item['relative_path'] != expected or item['sampling'] != manifest['sampling']
                or item['binding_sha256'] != manifest['binding_sha256']
                or item['world_size'] != topology.world_size
                or item['completed_updates'] != update
                or item['shared_state_sha256'] != receipts[0]['shared_state_sha256']
                or file_sha(root / expected) != item['file_sha256']):
            raise ValueError('committed checkpoint rank set is incomplete or changed')
    state = verified_state(root, receipts[topology.rank])
    RankMicrobatches.from_cursor(ledger, state['sampling'], topology.rank, order,
                                topology.world_size, topology.microbatch, topology.accumulate)
    if state['binding'] != binding:
        raise ValueError('checkpoint experiment binding differs')
    return state


class DistributedCheckpoint:
    """Callback for run_updates; distributed process group is owned by the caller."""
    def __init__(self, root, rank, world_size):
        self.root = Path(root); self.rank = rank; self.world = world_size

    def __call__(self, state):
        if state['rank'] != self.rank or state['sampling']['world_size'] != self.world:
            raise ValueError('callback rank/topology differs')
        if self.world > 1:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() != self.rank or dist.get_world_size() != self.world:
                raise ValueError('checkpoint process group differs')
        receipt = save_rank(self.root, state)
        if self.world == 1:
            return commit(self.root, [receipt])
        receipts = [None] * self.world
        dist.all_gather_object(receipts, receipt)
        value = commit(self.root, receipts) if self.rank == 0 else None
        dist.barrier()
        return value
