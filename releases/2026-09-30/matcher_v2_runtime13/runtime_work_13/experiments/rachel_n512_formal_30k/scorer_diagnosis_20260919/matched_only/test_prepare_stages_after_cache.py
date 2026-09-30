"""Dependency state checks, no remote process or real subprocess operations."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import prepare_stages_after_cache as s


class DependencyTests(unittest.TestCase):
    def receipt(self, status='running'):
        return dict(pid=100, gpu_jobs_started=False, status=status,
            completed_splits=['train', 'val'] if status == 'complete' else [])

    def test_wait_even_if_commit_precedes_exit(self):
        for status in ('running', 'complete'):
            self.assertEqual(s.dependency_state(self.receipt(status), dict(state='S', startticks=123), 100, 123), 'waiting')
        self.assertEqual(s.dependency_state(self.receipt('complete'), None, 100, 123), 'ready')
        self.assertEqual(s.dependency_state(self.receipt('complete'), dict(state='Z', startticks=123), 100, 123), 'ready')

    def test_failure_missing_pid_reuse_are_not_restarts(self):
        for receipt, proc in ((self.receipt('failed'), dict(state='S', startticks=123)),
                (self.receipt(), None), (self.receipt('complete'), dict(state='S', startticks=456)),
                (dict(self.receipt('complete'), completed_splits=['train']), None)):
            with self.assertRaises(RuntimeError):
                s.dependency_state(receipt, proc, 100, 123)

    def test_finite_cpu_dispatch_and_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / 'base'
            base.mkdir()
            s.save(base / 'status.json', self.receipt('complete'))
            output = root / 'output'
            args = SimpleNamespace(output=str(output), base_root=str(base), pid=100, startticks=123, wait_hours=1.)
            calls = []
            class Child:
                pid = 999
                def __init__(self, command, **kwargs):
                    calls.append(command)
                    self.split = command[command.index('--split')+1]
                    self.out = Path(command[command.index('--output')+1])
                    self.out.mkdir()
                def wait(self):
                    s.save(self.out / 'protocol.json', dict(status='complete', formal_training_eligible=True,
                        completed_pairs={'train':24000,'val':3000}[self.split], gt_used=False, labels_used=False))
                    return 0
            with patch.dict(s.os.environ, CUDA_VISIBLE_DEVICES=''), patch.object(s, 'process_identity', return_value=None), \
                    patch.object(s.subprocess, 'Popen', Child), patch.object(s.time, 'sleep', side_effect=AssertionError('no sleep')):
                result = s.execute(args)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['completed_splits'], ['train', 'val'])
            self.assertEqual(len(calls), 2)
            self.assertFalse(result['gpu_jobs_started'])
            with self.assertRaisesRegex(ValueError, 'already started'):
                s.execute(args)


if __name__ == '__main__':
    unittest.main()
