"""Per-parameter/per-rank gate evidence in addition to exact checkpoint replay.

This never changes a gradient. It is enabled only for the discarded 12-update
gate, not the formal training loop. Zero-initialized residual internals may
have zero gradients on update1; by update2 each enabled branch must participate.
"""
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.model_adapter import require
from .data_runtime import read_bound


def branch(name):
    if '.upgrades.' not in name:return 'legacy_matcher' if '.matcher.' in name else 'scorer'
    part = name.split('.upgrades.', 1)[1]
    if part.startswith('self_blocks.'):return 'self_attention'
    if part.startswith('cross_blocks.'):return 'cross_attention'
    if part.startswith(('scale_primal.', 'scale_dual.')) or part == 'scale_weight':return 'late_scale_logits'
    if part.startswith('concat.'):return 'concat_residual'
    if part == 'log_sharpness':return 'sharpness'
    return part.split('.')[0]


def check_gradients(values, names, update):
    require(set(values) == set(names), 'unused or unexpected parameter gradients')
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values.values()),
            'nonfinite/invalid parameter gradient')
    totals = {}
    for name, value in values.items():totals[branch(name)] = totals.get(branch(name), 0.) + value
    if update >= 2:
        require(all(v > 0 for v in totals.values()), 'enabled branch still has zero gradients after initialization')
    require(any(v > 0 for v in totals.values()), 'all gradients zero')
    return totals


class GradientProbe:
    def __init__(self, module, root, binding, rank, world, device):
        self.root = Path(root); self.binding = binding; self.rank = rank; self.world = world; self.device = device
        self.names = sorted(n for n, p in module.named_parameters() if p.requires_grad)
        self.values = {}; self.hooks = []
        for name, param in module.named_parameters():
            if param.requires_grad:
                def hook(grad, name=name):
                    value = float(grad.detach().abs().sum())
                    self.values[name] = self.values.get(name, 0.) + value
                self.hooks.append(param.register_hook(hook))
        self.shape_calls = 0
        def inputs(_module, args):
            batch = args[0]
            require(batch['mask_a'].shape[1:] == (1, 800, 800)
                    and batch['mask_b'].shape[1:] == (1, 800, 800)
                    and batch['points_rc_a'].shape[1:] == (512, 2)
                    and batch['points_rc_b'].shape[1:] == (512, 2), 'gate did not use full800/N512 input')
            self.shape_calls += 1
        self.hooks.append(module.register_forward_pre_hook(inputs))
        torch.cuda.reset_peak_memory_stats(device)

    def update(self, row):
        update = row['completed_updates']; error = None; totals = None
        try:
            totals = check_gradients(self.values, self.names, update)
            require(self.shape_calls > 0, 'gate had no verified input forwards')
            total = torch.cuda.get_device_properties(self.device).total_memory
            reserved = torch.cuda.max_memory_reserved(self.device)
            require(reserved <= .9*total, 'less than10 percent GPU memory headroom')
            # Both ranks may arrive first. Keep filesystem errors inside the
            # collective error path, and never assume the launcher made this
            # per-run receipt directory (including a resumed run).
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root/f'update_{update:06d}_rank_{self.rank:02d}.json'
            write_json(path, dict(schema='matcher-v2-gradient-step/1', rank=self.rank, world=self.world,
                update=update, binding_sha256=digest(self.binding), parameter_l1=self.values,
                branch_l1=totals, full_size_forwards=self.shape_calls,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                cuda_peak_reserved_bytes=reserved, cuda_total_memory_bytes=total))
        except Exception as exc:error = type(exc).__name__+': '+str(exc)
        errors = [error]
        if self.world > 1:
            errors = [None]*self.world; dist.all_gather_object(errors, error)
        self.values = {}; self.shape_calls = 0
        require(not any(errors), 'gradient gate failed on rank: '+repr(errors))

    def close(self):
        for hook in self.hooks:hook.remove()


def gradient_receipt(root, formal_binding, names, world, updates):
    root = Path(root); files = []
    for update in range(1, updates+1):
        for rank in range(world):
            path = root/f'update_{update:06d}_rank_{rank:02d}.json'; row = json.loads(path.read_text())
            require(row['schema'] == 'matcher-v2-gradient-step/1' and row['rank'] == rank and row['world'] == world
                    and row['update'] == update and row['binding_sha256'] == digest(formal_binding)
                    and row['full_size_forwards'] > 0, 'gradient evidence belongs to another run/shape')
            require(check_gradients(row['parameter_l1'], names, update) == row['branch_l1'], 'branch summary differs')
            require(0 < row['cuda_peak_allocated_bytes'] <= row['cuda_peak_reserved_bytes'] <= .9*row['cuda_total_memory_bytes'],
                    'invalid CUDA memory evidence')
            files.append(dict(path=str(path.resolve()), sha256=file_sha(path)))
    return dict(schema='matcher-v2-gradient-gate/1', status='passed', updates=updates,
        world_size=world, formal_binding_sha256=digest(formal_binding), parameter_names=list(names),
        shape=dict(canvas=800, points=512), files=files, gpu=True, weights_discarded_before_formal=True)


def check_gradient_receipt(spec, formal_binding, names, world):
    row = read_bound(spec)
    require(row['status'] == 'passed' and row['updates'] == 12 and row['world_size'] == world,
            'complete12-update gradient gate required')
    require(row == gradient_receipt(Path(row['files'][0]['path']).parent, formal_binding, names, world, 12),
            'gradient receipt not supported by actual complete rank/update files')
    return row
