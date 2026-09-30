from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.exposure import digest
from . import launcher


class LauncherTests(unittest.TestCase):
    def test_explicit_new_entry_no_legacy_order_and_topologies(self):
        for world in (1, 2):
            cmd = launcher.command('/python', '/spec', '/out', world, 'gate', stop=12)
            self.assertIn(launcher.__package__+'.execution', cmd)
            self.assertNotIn('--order', cmd)
            self.assertEqual('--nproc_per_node=2' in cmd, world == 2)
            self.assertIn('--resume', launcher.command('/p', '/s', '/o', world, 'gate', stop=12, resume=True))
            self.assertIn('--gate-receipt', launcher.command('/p', '/s', '/o', world, 'formal', gate='/proof'))
        for world, mode, kw in [(3, 'gate', dict(stop=12)), (2, 'gate', dict(stop=2)),
                                (1, 'formal', {}), (2, 'gate', dict(stop=12, gate='/bad'))]:
            with self.assertRaises(ValueError):launcher.command('/p', '/s', '/o', world, mode, **kw)

    def test_gate_requires_both_gradient_proofs_and_model_rng_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); calls = []; binding = dict(run_mode='gate', explicit_CPU_fixture=True)
            def child(name, cmd):
                calls.append(name); folder = Path(cmd[cmd.index('--out')+1]); folder.mkdir(parents=True, exist_ok=True)
                if not (folder/'binding.json').exists():write_json(folder/'binding.json', binding)
                if name != 'gate_update1':
                    write_json(folder/'gate_update12.json', dict(explicit_CPU_fixture=True))
                    write_json(folder/'gradients_update12.json', dict(parameter_names=['model.matcher.fake']))
            with patch.object(launcher, 'check_gradient_receipt') as gradient, patch.object(launcher, 'check_gate') as replay:
                proof = launcher.gate_sequence(root, '/spec', 2, '/python', child)
                self.assertTrue(proof.exists()); self.assertEqual(gradient.call_count, 2); replay.assert_called_once()
                self.assertEqual(calls, ['gate_full12', 'gate_update1', 'gate_resume12'])
                self.assertEqual(replay.call_args[0][1], {'explicit_CPU_fixture':True})
            # These mocks only verify controller ordering; no passed CUDA
            # preparation is manufactured or retained outside the temp fixture.

    def test_gate_failure_does_not_start_any_later_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            def child(name, cmd):calls.append(name); raise RuntimeError('fixture failure')
            with self.assertRaises(RuntimeError):launcher.gate_sequence(Path(temp), '/spec', 2, '/python', child)
            self.assertEqual(calls, ['gate_full12'])
            self.assertFalse((Path(temp)/'gpu_gate.json').exists())

    def test_gate_only_never_starts_formal_or_claims_training_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source'; source.mkdir()
            for name in ('source_binding.json', 'baseline_composition.json'):
                write_json(source/name, {'fixture': True})
            spec = root/'spec.json'; write_json(spec, dict(topology=dict(world_size=2), arm='B2', module='matcher'))
            preparation = root/'prep.json'
            write_json(preparation, dict(schema='matcher-v2-runtime-cpu-preparation/1', status='passed',
                binding_sha256=file_sha(source/'source_binding.json'),
                baseline_composition_sha256=file_sha(source/'baseline_composition.json'),
                errors=0, failures=0, skipped=0, cuda_initialized=False))
            interpreter = root/'python'; interpreter.touch(); out = root/'out'
            args = SimpleNamespace(spec=spec, preparation=preparation, out=out, python=interpreter,
                                   gpus=[3, 4], gate_only=True)
            def fake_gate(*_):
                path = out/'gpu_gate.json'; write_json(path, dict(explicit_CPU_fixture=True)); return path
            with patch.object(launcher, 'load_inputs', return_value=dict(plan=None, source=source)), \
                 patch.object(launcher, 'check_free', return_value=[]), \
                 patch.object(launcher, 'identity', return_value={}), \
                 patch.object(launcher, 'gate_sequence', side_effect=fake_gate), \
                 patch.object(launcher, 'execute') as execute, patch.object(launcher, 'verify_formal') as formal:
                self.assertEqual(launcher.run(args), 0)
                execute.assert_not_called(); formal.assert_not_called()
            self.assertFalse((out/'controller_complete.json').exists())
            self.assertFalse((out/'formal').exists())
            result = launcher.read(out/'gate_only_complete.json')
            self.assertFalse(result['formal_started']); self.assertEqual(result['actual_formal_updates'], 0)


if __name__ == '__main__':unittest.main()
