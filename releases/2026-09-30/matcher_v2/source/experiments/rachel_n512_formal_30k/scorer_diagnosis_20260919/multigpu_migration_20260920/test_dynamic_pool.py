import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import dynamic_pool as pool


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'pool'
        self.plan = dict(output_root=str(self.base/'original'), device_wrapper='/fixed/device_runtime.py', lanes=[])
        self.plan_path = self.base/'plan.json'
        self.plan_path.write_text(json.dumps(self.plan))

    def stage(self, name, prerequisites=None):
        return dict(name=name, gpu=True, command=['/fixed/python', '-m', 'fixed.train', '--epochs', '16'],
            cwd=str(self.base), env={'PYTHONPATH': '/fixed/source'},
            completion=str(self.base/(name+'.json')), completion_expect={'status': 'complete'},
            original_runtime_receipt=str(self.base/'original'/('runtime_'+name+'.json')),
            prerequisites=prerequisites or [])

    def records(self, packages):
        pool.registry.register_experiments(self.root, packages)
        return pool.registry.read_registry(self.root)

    def test_missing_second_proof_waits_and_failed_producer_refuses(self):
        a, b = str(self.base/'a.json'), str(self.base/'b.json')
        package = dict(name='p', origin_lane='lane1', stages=[self.stage('train', [
            {'path': a, 'expect': {'status': 'complete'}},
            {'path': b, 'expect': {'status': 'complete'}, 'producer_lane': 'lane6'}])])
        records = self.records([package])
        Path(a).write_text('{"status":"complete"}')
        self.assertFalse(pool.prerequisites_ready(package, records, plan=self.plan))
        producer = Path(self.plan['output_root'])/'lane6'/'status.json'
        producer.parent.mkdir(parents=True)
        producer.write_text('{"status":"failed"}')
        with self.assertRaisesRegex(RuntimeError, 'producer lane failed'):
            pool.prerequisites_ready(package, records, plan=self.plan)
        producer.unlink()
        source = self.stage('source')
        source['completion'] = b
        records['experiments']['source'] = dict(status='failed', stages=[source])
        with self.assertRaisesRegex(RuntimeError, 'transferred prerequisite failed'):
            pool.prerequisites_ready(package, records, plan=self.plan)

    def test_selection_priority_internal_proofs_and_only_idle_completed_lanes(self):
        packages = []
        for name, priority in [('late', 20), ('early', 10)]:
            train = self.stage(name+'_train')
            evaluate = self.stage(name+'_eval', [{'path': train['completion'], 'expect': train['completion_expect']}])
            packages.append(dict(name=name, priority=priority, origin_lane='lane1', stages=[train, evaluate]))
        records = self.records(packages)
        values = {}
        for i in range(4):
            name, gpu = 'lane'+str(i), 'GPU-'+str(i)
            self.plan['lanes'].append(dict(name=name, gpu_uuid=gpu, stages=[{}]))
            values[str(Path(self.plan['output_root'])/name/'status.json')] = dict(
                status='complete' if i != 1 else 'running', active_pid=None,
                completed_stages=1, stages=[dict(status='complete', returncode=0)])
        def read(path):
            if path not in values:
                raise FileNotFoundError(path)
            return values[path]
        result = pool.choose_assignments(self.plan, records, read=read, idle=lambda uuid, _: uuid != 'GPU-3')
        self.assertEqual(result, [('early', 'GPU-0'), ('late', 'GPU-2')])
        records['experiments']['early'].update(status='running', assigned_gpu='GPU-0')
        self.assertEqual(pool.choose_assignments(self.plan, records, read=read,
            idle=lambda uuid, _: uuid != 'GPU-3'), [('late', 'GPU-2')])

    def run_worker(self, *, code=0, valid=True):
        stage = self.stage('train')
        stage['additional_completions'] = [{'path': str(self.base/'extra.json'), 'expect': {'frozen': True}}]
        self.records([dict(name='p', origin_lane='lane1', stages=[stage])])
        with pool.registry.locked_registry(self.root) as records:
            records['experiments']['p'].update(status='running', assigned_gpu='GPU-2',
                worker_pid=os.getpid(), worker_startticks=17)
        calls = []
        class Child:
            pid = 123456
            def wait(inner):
                Path(stage['completion']).write_text(json.dumps({'status': 'complete' if valid else 'running'}))
                Path(stage['additional_completions'][0]['path']).write_text('{"frozen":true}')
                return code
        class Runtime:
            @staticmethod
            def inspect(pid):
                return dict(pid=pid, state='R', startticks=17)
            @staticmethod
            def popen(command, **kwargs):
                calls.append((command, kwargs))
                return Child()
        with patch.object(pool, 'gpu_idle', side_effect=AssertionError('worker must not acquire parent lease')):
            if code or not valid:
                with self.assertRaises((RuntimeError, ValueError)):
                    pool.worker(self.plan_path, self.root, 'p', 'GPU-2', runtime=Runtime())
            else:
                pool.worker(self.plan_path, self.root, 'p', 'GPU-2', runtime=Runtime())
        return stage, calls, pool.registry.read_registry(self.root)

    def test_worker_preserves_leaf_and_completes_only_with_proofs(self):
        stage, calls, records = self.run_worker()
        self.assertEqual(len(calls), 1)
        command, options = calls[0]
        self.assertEqual(command[0:2], ['/fixed/python', '/fixed/device_runtime.py'])
        self.assertEqual(command[-5:], ['--module', 'fixed.train', '--', '--epochs', '16'])
        self.assertNotEqual(command[command.index('--receipt')+1], stage['original_runtime_receipt'])
        self.assertEqual(options['cwd'], stage['cwd'])
        self.assertEqual(options['env']['PYTHONPATH'], '/fixed/source')
        self.assertEqual(options['env']['RACHEL_TRANSFER_WORKER'], 'p')
        self.assertEqual(options['env']['CUDA_VISIBLE_DEVICES'], 'GPU-2')
        self.assertEqual(records['experiments']['p']['status'], 'complete')
        self.assertEqual(records['stages'][pool.receipt_key(stage)]['returncode'], 0)
        self.assertEqual(records['claims'], {})

    def test_exit_zero_without_completion_is_failed_not_retried(self):
        stage, calls, records = self.run_worker(valid=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(records['experiments']['p']['status'], 'failed')
        self.assertEqual(records['stages'][pool.receipt_key(stage)]['status'], 'failed')
        self.assertEqual(records['stages'][pool.receipt_key(stage)]['returncode'], 0)

    def test_nonzero_and_pid_identity(self):
        stage, calls, records = self.run_worker(code=7)
        self.assertEqual(len(calls), 1)
        self.assertEqual(records['stages'][pool.receipt_key(stage)]['returncode'], 7)
        self.assertEqual(records['experiments']['p']['status'], 'failed')
        self.assertFalse(pool.require_alive(1, 20, lambda _: None))
        self.assertFalse(pool.require_alive(1, 20, lambda _: {'startticks': 20, 'state': 'Z'}))
        with self.assertRaisesRegex(RuntimeError, 'PID reused'):
            pool.require_alive(1, 20, lambda _: {'startticks': 21, 'state': 'R'})


if __name__ == '__main__':
    unittest.main()
