"""Synthetic receipt validation only; these tests are NOT CUDA evidence."""
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.exposure import digest
from .gradient_gate import GradientProbe, branch, check_gradients, gradient_receipt, check_gradient_receipt


class GradientGateTests(unittest.TestCase):
    def test_probe_creates_missing_receipt_directory_with_real_disk_write(self):
        class Fixture(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, batch):
                return self.weight * batch['points_rc_a'].sum()

        with tempfile.TemporaryDirectory() as folder, \
                patch('torch.cuda.reset_peak_memory_stats'), \
                patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=4)), \
                patch('torch.cuda.max_memory_reserved', return_value=3), \
                patch('torch.cuda.max_memory_allocated', return_value=2):
            root = Path(folder)/'new_run'/'gradient_steps'
            binding = dict(explicit_synthetic_CPU_fixture=True)
            module = Fixture(); probe = GradientProbe(module, root, binding, 0, 1, 'cpu')
            batch = {name: torch.zeros((1, 1, 800, 800)) for name in ('mask_a', 'mask_b')}
            batch.update({name: torch.ones((1, 512, 2)) for name in ('points_rc_a', 'points_rc_b')})
            self.assertFalse(root.exists())
            try:
                for update in (1, 2):
                    module.zero_grad(); module(batch).backward()
                    probe.update(dict(completed_updates=update))
                proof = gradient_receipt(root, binding, ['weight'], 1, 2)
                self.assertEqual(len(proof['files']), 2)
                self.assertFalse(list(root.glob('*.tmp.*')))
            finally:
                probe.close()

    def test_all_names_required_even_initial_zero_parameters(self):
        names = ['model.matcher.base.weight', 'model.matcher.upgrades.self_blocks.0.q.weight']
        self.assertEqual(check_gradients(dict(zip(names, [1., 0.])), names, 1)['self_attention'], 0.)
        with self.assertRaisesRegex(ValueError, 'zero gradients'):check_gradients(dict(zip(names, [1., 0.])), names, 2)
        with self.assertRaisesRegex(ValueError, 'unused'):check_gradients({names[0]: 1.}, names, 1)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):check_gradients(dict(zip(names, [1., math.nan])), names, 1)

    def test_immutable_all_rank_all_update_receipts_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); binding = dict(explicit_synthetic_CPU_fixture=True)
            names = ['model.matcher.upgrades.self_blocks.0.q.weight']
            for update in range(1, 13):
                for rank in range(2):
                    write_json(root/f'update_{update:06d}_rank_{rank:02d}.json', dict(schema='matcher-v2-gradient-step/1',
                        rank=rank, world=2, update=update, binding_sha256=digest(binding), parameter_l1={names[0]: 1.},
                        branch_l1={'self_attention': 1.}, full_size_forwards=2,
                        cuda_peak_allocated_bytes=2, cuda_peak_reserved_bytes=3, cuda_total_memory_bytes=4))
            receipt = gradient_receipt(root, binding, names, 2, 12)
            path = root/'receipt.json'; write_json(path, receipt)
            spec = dict(path=str(path), sha256=file_sha(path))
            self.assertEqual(check_gradient_receipt(spec, binding, names, 2), receipt)
            (root/'update_000012_rank_01.json').unlink()
            with self.assertRaises(FileNotFoundError):check_gradient_receipt(spec, binding, names, 2)

    def test_scorer_and_new_branch_groups_distinct(self):
        self.assertEqual(branch('model.head.cluster.0.weight'), 'scorer')
        self.assertEqual(branch('model.matcher.upgrades.log_sharpness'), 'sharpness')
        self.assertNotEqual(branch('model.matcher.upgrades.scale_primal.0.weight'), branch('model.matcher.upgrades.cross_blocks.0.q.weight'))


if __name__ == '__main__':unittest.main()
