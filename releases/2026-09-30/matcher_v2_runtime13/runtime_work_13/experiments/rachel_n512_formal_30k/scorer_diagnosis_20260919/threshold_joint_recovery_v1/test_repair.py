"""Exercise the actual stage initialization and CPU backward, not only imports."""
import ast
from contextlib import ExitStack
import importlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from recovery_contract import REL, expected_repair, verify_source_delta, validate_previous_failure


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        import types
        if 'experiments' in sys.modules:
            raise AssertionError('repair test needs a fresh bound-source process')
        namespace = types.ModuleType('experiments')
        namespace.__path__ = [str(Path(os.environ['JOINT_REPAIR_SOURCE']) / 'experiments')]
        sys.modules['experiments'] = namespace

    def source(self):
        return Path(os.environ['JOINT_REPAIR_SOURCE'])

    def test_scope_no_longer_shadows_import(self):
        import sys
        sys.path.insert(0, str(self.source()))
        train = importlib.import_module('.'.join(REL.parts) + '.train')
        self.assertEqual(Path(train.__file__).resolve(), (self.source() / REL / 'train.py').resolve())
        self.assertNotIn('snapshot', train.run_stage.__code__.co_varnames)
        self.assertNotIn('snapshot', train.run_stage.__code__.co_cellvars)

    def test_source_delta_is_only_four_local_renames(self):
        result = verify_source_delta(self.source().with_name('training_source_02'), self.source())
        self.assertEqual(result['changed'], [str(REL / 'train.py')])

    def test_unrelated_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder) / 'old', Path(folder) / 'new'
            for root in (old, new):
                (root / REL).mkdir(parents=True)
            text = (self.source().with_name('training_source_02') / REL / 'train.py').read_text()
            (old / REL / 'train.py').write_text(text)
            (new / REL / 'train.py').write_text(expected_repair(text) + '\n# extra change\n')
            with self.assertRaisesRegex(ValueError, 'reviewed four'):
                verify_source_delta(old, new)

    def test_prior_work_is_never_overwritten(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); gate = root / 'preflight/ddp_scratch_joint_01'
            gate.mkdir(parents=True)
            (root/'failure_scratch_joint_controller.json').write_text(json.dumps(
                dict(status='failed', current_job=dict(phase='gate'))))
            (root/'scratch_joint_gate_exit.json').write_text('{"returncode":1}')
            (gate/'failure.json').write_text(json.dumps(dict(type='UnboundLocalError', message='snapshot')))
            self.assertEqual(len(validate_previous_failure(root)), 3)
            (root/'formal_scratch_joint').mkdir()
            with self.assertRaisesRegex(ValueError, 'existing formal'):
                validate_previous_failure(root)

    def exercise_stage(self, original=False):
        import sys
        sys.path.insert(0, str(self.source()))
        import torch
        from torch import nn
        train = importlib.import_module('.'.join(REL.parts) + '.train')
        toy = importlib.import_module('.'.join(REL.parts) + '.test_joint_policy')
        function = train.run_stage
        if original:
            # Compile the old actual function with identical globals; reproduce
            # the same UnboundLocalError even though the migration branch is off.
            text = (self.source().with_name('training_source_02') / REL / 'train.py').read_text()
            node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'run_stage')
            namespace = dict(train.__dict__)
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<original-run-stage>', 'exec'), namespace)
            function = namespace['run_stage']
        model = nn.Module(); model.matcher = toy.ToyMatcher(); model.head = nn.Linear(2, 1)

        class MiniStep(nn.Module):
            def __init__(self, model, *args, **kwargs):
                super().__init__(); self.model = model
            def forward(self, batch):
                loss = self.model.head(self.model.matcher(torch.ones(16, 2))).square().mean()
                return loss, {}, {}

        config = SimpleNamespace(learning_rate=1e-4, matcher_learning_rate=1e-6, weight_decay=1e-4,
            effective_batch=32, microbatch=16, accumulate=2, workers_per_rank=0,
            data_seed=17, maximum_epochs=1, gradient_clip_norm=1.)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            args = SimpleNamespace(out=folder, arm='scratch_joint', preflight_steps=1, resume=False)
            mocks = dict(Dataset=lambda *a: range(24000), TrainModule=MiniStep,
                DataLoader=lambda *a, **k: [dict(labels=[1]*16)]*2,
                to_device=lambda x, d: x, rng_state=lambda: dict(cpu_fixture=True))
            # Patching the original function's globals reproduces the same path.
            if original:
                function.__globals__.update(mocks)
            else:
                for key, value in mocks.items(): stack.enter_context(patch.object(train, key, value))
            stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
            return function(model, 'scorer', args, config,
                dict(train=dict(path='fixture', sha256='fixture')), {}, torch.device('cpu'), 0, 1)

    def test_original_fails_before_first_update(self):
        with self.assertRaisesRegex(UnboundLocalError, 'snapshot'):
            self.exercise_stage(original=True)

    def test_actual_fixed_run_stage_cpu_update_reaches_both_modules(self):
        result = self.exercise_stage()
        self.assertEqual(result['updates'], 1)
        self.assertEqual(result['status'], 'passed')
        self.assertGreater(result['matcher_change']['changed_tensors'], 0)
        self.assertGreater(result['head_change']['changed_tensors'], 0)
        self.assertEqual(result['matcher_change']['unused_changed'], [])
        self.assertTrue(all(v > 0 for v in result['gradient_l2_max'].values()))


if __name__ == '__main__':
    unittest.main()
