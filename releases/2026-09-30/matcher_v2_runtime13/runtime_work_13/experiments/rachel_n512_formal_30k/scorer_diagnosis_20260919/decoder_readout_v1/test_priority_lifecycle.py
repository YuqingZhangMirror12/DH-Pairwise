"""Test priority dependencies without processes, GPUs or remote writes."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import launch_priority as launcher


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PriorityLifecycle(unittest.TestCase):
    def group_fixture(self, root, codes=(0, 0), missing=None):
        events = []
        children = []
        for index, (variant, gpu) in enumerate(launcher.GPUS.items()):
            if variant != missing:
                dump(root / (variant + '_complete.json'), {'status': 'training_and_evaluation_complete'})
            child = Mock(pid=100 + index)
            def wait(index=index):
                events.append(('wait', index)); return codes[index]
            child.wait.side_effect = wait; children.append(child)
        return events, children

    def run_group(self, root, children, events, resume_error=None):
        def resumed(pause):
            events.append(('resume', pause))
            if resume_error:
                raise RuntimeError(resume_error)
        stack = ExitStack()
        stack.enter_context(patch.object(launcher, 'ROOT', root))
        stack.enter_context(patch.object(launcher, 'prepare_formal', return_value={'updates': 1900}))
        stack.enter_context(patch.object(launcher, 'free', return_value='synthetic-device'))
        stack.enter_context(patch.object(launcher, 'identity', side_effect=lambda pid: dict(pid=pid, starttime=1, cmdline='fake')))
        stack.enter_context(patch.object(launcher.time, 'sleep'))
        stack.enter_context(patch.object(launcher.subprocess, 'Popen', side_effect=children))
        resumed_mock = stack.enter_context(patch.object(launcher, 'resume_joint', side_effect=resumed))
        return stack, resumed_mock

    def test_resume_only_after_both_lane_returns_and_full_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); events, children = self.group_fixture(root)
            stack, resumed = self.run_group(root, children, events)
            with stack: launcher.group()
            self.assertEqual(events, [('wait', 0), ('wait', 1), ('resume', {'updates': 1900})])
            resumed.assert_called_once()
            self.assertEqual(launcher.read(root / 'priority_queue_complete.json')['status'], 'complete')
            self.assertFalse((root / 'queue_failure.json').exists())

    def test_one_lane_failure_preserves_reservation_and_never_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); events, children = self.group_fixture(root, codes=(1, 0))
            stack, resumed = self.run_group(root, children, events)
            with stack, self.assertRaisesRegex(ValueError, 'lane failed'): launcher.group()
            resumed.assert_not_called()
            self.assertEqual(events, [('wait', 0), ('wait', 1)])
            self.assertFalse((root / 'pooling_complete.json').exists())
            self.assertFalse(launcher.read(root / 'queue_failure.json')['automatic_retry'])

    def test_zero_return_without_full_lane_receipt_is_not_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); events, children = self.group_fixture(root, missing='patch_sum')
            stack, resumed = self.run_group(root, children, events)
            with stack, self.assertRaises(FileNotFoundError): launcher.group()
            resumed.assert_not_called()
            self.assertFalse((root / 'priority_queue_complete.json').exists())

    def test_resume_failure_does_not_claim_joint_or_priority_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); events, children = self.group_fixture(root)
            stack, resumed = self.run_group(root, children, events, resume_error='simulated resume failure')
            with stack, self.assertRaisesRegex(RuntimeError, 'resume failure'): launcher.group()
            resumed.assert_called_once()
            self.assertTrue((root / 'pooling_complete.json').exists())
            self.assertFalse((root / 'priority_queue_complete.json').exists())

    def test_resume_busy_gpu_does_not_launch_or_create_resume_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(launcher, 'JOINT', root), \
                 patch.object(launcher, 'free', side_effect=ValueError('GPU occupied')), \
                 patch.object(launcher.subprocess, 'Popen') as spawned:
                with self.assertRaisesRegex(ValueError, 'occupied'): launcher.resume_joint({})
                spawned.assert_not_called()
                self.assertFalse((root / 'pooling_priority_resume_20260929').exists())

    def test_joint_resumes_exact_original_command_without_gate_then_nine_evaluations(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / 'source'; source.mkdir()
            formal = root / 'formal_scratch_joint'; stage = formal / 'scorer'; stage.mkdir(parents=True)
            checkpoint = stage / 'last.pt'; checkpoint.write_bytes(b'synthetic full checkpoint')
            for name in ('best_joint', 'best_real'): (stage / (name + '.pt')).write_bytes(name.encode())
            binding = {'same': 'original training binding'}
            selection = dict(status='selected', binding=binding, matcher_unchanged=False,
                             **{name + '_sha256': launcher.sha(stage / (name + '.pt'))
                                for name in ('best_joint', 'best_real')})
            dump(stage / 'selection.json', selection)
            dump(formal / 'training_complete.json', dict(status='training_complete'))
            original = ['python', '-m', 'torch.distributed.run', '--nproc_per_node=2',
                        '-m', launcher.PACKAGE + '.train', '--out', str(formal)]
            pause = dict(checkpoint=str(checkpoint), checkpoint_sha256=launcher.sha(checkpoint),
                         binding=binding, source_sha256={}, training_command=original)
            events = []
            child = Mock(pid=500)
            child.wait.side_effect = lambda: events.append('training_returned') or 0
            def evaluate_all(common, gpus, out):
                self.assertEqual(events, ['training_returned'])
                self.assertEqual(gpus, '3,4')
                events.append('required_evaluations')
                dump(out / 'evaluation_complete.json', dict(status='complete', final_evaluations=9))
                return dict(status='complete', final_evaluations=9)
            queue = SimpleNamespace(QUEUE=root / 'old_queue', load=Mock(return_value=object()),
                                    evaluate_all=Mock(side_effect=evaluate_all))
            with patch.object(launcher, 'JOINT', root), patch.object(launcher, 'free'), \
                 patch.object(torch, 'load', return_value=dict(binding=binding, updates=1900)), \
                 patch.object(launcher, 'identity', return_value=dict(pid=500, starttime=5, cmdline='fake')), \
                 patch.object(launcher.time, 'sleep'), \
                 patch.object(launcher.subprocess, 'Popen', return_value=child) as spawned, \
                 patch.object(launcher.importlib, 'import_module', return_value=queue):
                launcher.resume_joint(pause)
            self.assertEqual(spawned.call_count, 1)
            self.assertEqual(spawned.call_args.args[0], original + ['--resume'])
            self.assertNotIn('--preflight-steps', spawned.call_args.args[0])
            self.assertEqual(spawned.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '3,4')
            self.assertEqual(events, ['training_returned', 'required_evaluations'])
            resumed = launcher.read(root / 'recovery_complete.json')
            self.assertTrue(resumed['authorized_pause_resume'])
            self.assertEqual(resumed['evaluation_terminal_sha256'],
                             launcher.sha(root / 'postprocess_joint_01/evaluation_complete.json'))


if __name__ == '__main__':
    unittest.main()
