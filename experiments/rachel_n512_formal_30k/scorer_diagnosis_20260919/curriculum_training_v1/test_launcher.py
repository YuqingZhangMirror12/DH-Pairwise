"""Controller protocol fixtures only; no CUDA or actual job launch."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import launcher as api
from .checkpoint_io import file_sha, write_json
from .runtime_io import export_completed
from .runtime_plan import experiment_binding
from . import test_runtime_io as fixture


class LauncherProtocolTests(unittest.TestCase):
    def test_explicit_idle_devices_not_utilization_heuristic(self):
        result = api.assigned_free_devices([4, 7], 2, '4, GPU-four\n7, GPU-seven\n', 'GPU-other, 800\n')
        self.assertEqual(result, [dict(index=4, uuid='GPU-four'), dict(index=7, uuid='GPU-seven')])

    def test_compute_process_prevents_preemption(self):
        with self.assertRaisesRegex(ValueError, 'still have compute processes'):
            api.assigned_free_devices([4], 1, '4, GPU-four\n', 'GPU-four, 12\n')

    def test_duplicate_missing_wrong_count_or_bool_assignment_rejected(self):
        for values, world in [([4, 4], 2), ([3], 1), ([4, 7], 1), ([True], 1)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                api.assigned_free_devices(values, world, '4, GPU-four\n7, GPU-seven\n', '')

    def test_head_is_single_process_and_formal_starts_fresh(self):
        value = api.command('/python', '/spec', '/out', 'curriculum', 1, 'formal', gate='/proof')
        self.assertNotIn('torch.distributed.run', value); self.assertNotIn('--resume', value)
        self.assertNotIn('--gate-stop', value); self.assertIn('--gate-receipt', value)

    def test_matcher_is_two_process_with_exact_gate_stop(self):
        value = api.command('/python', '/spec', '/out', 'mixed', 2, 'gate', stop=12, resume=True)
        self.assertIn('--nproc_per_node=2', value); self.assertIn('--standalone', value)
        self.assertEqual(value[value.index('--gate-stop') + 1], '12'); self.assertIn('--resume', value)

    def test_invalid_mode_or_gate_cannot_build_command(self):
        for mode, stop, gate in [('formal', None, None), ('formal', 12, '/proof'), ('gate', 2, None), ('gate', 12, '/proof')]:
            with self.subTest(mode=mode, stop=stop), self.assertRaises(ValueError):
                api.command('/python', '/spec', '/out', 'curriculum', 1, mode, stop=stop, gate=gate)

    def test_child_nonzero_return_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); values = [sys.executable, '-c', 'raise SystemExit(7)']
            environment = dict(api.os.environ, CUDA_VISIBLE_DEVICES='')
            with patch.object(api, 'identity', return_value=dict(pid=999, synthetic=True)), self.assertRaises(api.ChildFailure):
                api.execute(root, 'synthetic_cpu_failure', values, environment, root)
            returned = json.loads((root / 'synthetic_cpu_failure_return.json').read_text())
            self.assertEqual(returned['returncode'], 7); self.assertFalse(returned['automatic_retry'])
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                api.execute(root, 'synthetic_cpu_failure', values, environment, root)

    def test_child_zero_is_only_process_return_not_training_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(api, 'identity', return_value=dict(pid=999, synthetic=True)):
                api.execute(root, 'synthetic_cpu_success', [sys.executable, '-c', 'print("CPU fixture")'],
                    dict(api.os.environ, CUDA_VISIBLE_DEVICES=''), root)
            self.assertEqual(json.loads((root / 'synthetic_cpu_success_return.json').read_text())['returncode'], 0)
            self.assertFalse((root / 'controller_complete.json').exists())

    def test_gate_sequence_has_two_fresh_runs_and_one_update1_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); calls = []; binding = dict(run_mode='gate', synthetic_only=True)
            def child(phase, command):
                calls.append((phase, command)); out = Path(command[command.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)
                if not (out / 'binding.json').exists(): write_json(out / 'binding.json', binding)
                if phase != 'gate_update1': write_json(out / 'gate_update12.json', dict(synthetic_parser_only=True))
            with patch.object(api, 'check_gate', return_value=dict(status='synthetic_test_only')) as check:
                proof = api.gate_sequence(root, '/fixture_spec', 'curriculum', 2, '/python', child)
            self.assertEqual([p for p, _ in calls], ['gate_full12', 'gate_update1', 'gate_resume12'])
            self.assertNotIn('--resume', calls[0][1]); self.assertNotIn('--resume', calls[1][1]); self.assertIn('--resume', calls[2][1])
            self.assertNotEqual(calls[0][1][calls[0][1].index('--out')+1], calls[1][1][calls[1][1].index('--out')+1])
            self.assertTrue(proof.exists()); self.assertEqual(check.call_count, 1)

    def test_gate_failure_does_not_make_formal_gpu_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def child(phase, command):
                out = Path(command[command.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)
                if not (out / 'binding.json').exists(): write_json(out / 'binding.json', dict(run_mode='gate'))
                if phase != 'gate_update1': write_json(out / 'gate_update12.json', {})
            with patch.object(api, 'check_gate', side_effect=ValueError('synthetic mismatch')), self.assertRaises(ValueError):
                api.gate_sequence(root, '/fixture', 'curriculum', 1, '/python', child)
            self.assertTrue((root / 'gate_candidate.json').exists()); self.assertFalse((root / 'gpu_gate.json').exists())


class LauncherCompletionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.spec = self.root / 'synthetic_spec.json'
        write_json(self.spec, dict(synthetic_fixture=True)); self.out = self.root / 'formal'
        def binding(*args):
            return dict(experiment_binding(*args), run_mode='formal', execution_manifest_sha256=file_sha(self.spec))
        with patch.object(fixture, 'experiment_binding', side_effect=binding):
            _, self.plan, self.binding, _ = fixture.train_fixture(self.out)
        result = export_completed(self.out / 'exports', self.out / 'checkpoints', self.plan, self.binding)
        result['export_root'] = str(self.out / 'exports'); write_json(self.out / 'training_complete.json', result)

    def verify(self):
        return api.verify_formal(self.out, self.spec, self.plan, 'curriculum')

    def test_complete_verified_but_frozen_evaluation_still_pending(self):
        value = self.verify()
        self.assertEqual(value['status'], 'training_complete_evaluation_pending')
        self.assertFalse(value['gpu_releasable_by_this_receipt']); self.assertFalse(value['frozen_evaluation_complete'])
        self.assertEqual(value['training_complete_sha256'], file_sha(self.out / 'exports/training_complete.json'))

    def test_changed_actual_model_file_rejected(self):
        (self.out / 'exports/sim_best.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'actual selected model changed'): self.verify()

    def test_changed_selection_rejected(self):
        (self.out / 'exports/selection.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'selection changed'): self.verify()

    def test_stale_complete_with_wrong_spec_rejected(self):
        write_json(self.spec, dict(changed_fixture=True), replace=True)
        with self.assertRaisesRegex(ValueError, 'completion/plan identity'): self.verify()

    def test_success_cannot_hide_failure(self):
        write_json(self.out / 'failure.json', dict(error='synthetic'))
        with self.assertRaisesRegex(ValueError, 'completion/plan identity'): self.verify()


if __name__ == '__main__':
    unittest.main()
