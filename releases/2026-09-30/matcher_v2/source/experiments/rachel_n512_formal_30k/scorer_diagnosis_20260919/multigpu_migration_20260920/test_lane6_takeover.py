import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from . import lane6_takeover as takeover


class TakeoverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'lane6'
        self.root.mkdir()
        self.checkpoint = self.base/'epoch_020.pt'
        self.checkpoint.touch()
        def stage(name, gpu):
            output = self.base/'outputs'/name
            return dict(name=name, gpu=gpu, cwd=str(self.base), output=str(output),
                env={'PYTHONPATH': '/sealed/source'}, prerequisites=[],
                command=['/fixed/python', '-m', 'fixed.module', '--checkpoint', str(self.checkpoint)],
                completion=str(output/'status.json'), completion_expect={'status': 'complete'})
        self.training = stage('S7_M13_M20_train', True)
        self.active = stage('M16_all_tokens_C16_train', True)
        self.cpu = [stage(name, False) for name in takeover.CPU_NAMES]
        self.after = stage('M20_all_tokens_C16_test', True)
        self.selected = dict(name='lane6', gpu_uuid='GPU-6',
            stages=[self.training, self.active]+self.cpu+[self.after])
        self.plan = dict(schema='seven-gpu-migration/1', output_root=str(self.base),
            device_wrapper='/external/device_runtime.py', lanes=[dict(name='lane'+str(i),
            gpu_uuid='GPU-'+str(i), stages=[stage('unused'+str(i), True)]) for i in range(6)]+[self.selected])
        self.plan_path = self.base/'plan.json'
        self.plan_path.write_text(json.dumps(self.plan))
        for s in (self.training, self.active):
            Path(s['completion']).parent.mkdir(parents=True)
            Path(s['completion']).write_text('{"status":"complete"}')
        self.wrapper = self.root/('runtime_'+self.active['name']+'.json')
        self.wrapper.write_text(json.dumps(dict(schema=takeover.WRAPPER_SCHEMA, pid=20,
            status='complete', module=self.active['command'][2], arguments=self.active['command'][3:],
            gpu={'uuid':'GPU-6'})))
        self.state = dict(lane='lane6', gpu_uuid='GPU-6', status='running', pid=10,
            completed_stages=1, active_name=self.active['name'], active_pid=20, stages=[
            dict(name=self.training['name'], status='complete', returncode=0),
            dict(name=self.active['name'], status='running', pid=20, cwd=self.active['cwd'],
                 command=takeover.lane.wrapped_command(self.active,self.selected,self.plan))])
        (self.root/'status.json').write_text(json.dumps(self.state))
        self.calls = []
        owner = self
        class Runtime:
            @staticmethod
            def inspect(pid):
                if pid == 10:
                    return None
                return dict(pid=pid, startticks=200 if pid == 20 else 300, state='Z' if pid == 20 else 'R')
            @staticmethod
            def sleep(seconds):
                raise AssertionError('fixture jobs complete immediately')
            @staticmethod
            def popen(command, **kwargs):
                owner.calls.append((command, kwargs))
                original = next(s for s in owner.cpu+[owner.after] if
                    command == s['command'] or command == takeover.lane.wrapped_command(s, owner.selected, owner.plan))
                class Child:
                    pid = 100+len(owner.calls)
                    def poll(inner):
                        path=Path(original['completion'])
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text('{"status":"complete"}')
                        return 0
                    def wait(inner):
                        return inner.poll()
                return Child()
        self.runtime = Runtime()
        # Distinct preserved CLI per stage, as in the real plan.
        for s in self.cpu+[self.after]:
            s['command'].extend(['--output',s['output']])
        self.plan_path.write_text(json.dumps(self.plan))

    def test_adopt_finished_child_and_cpu_exactly_once(self):
        result = takeover.execute(self.plan_path,20,200,runtime=self.runtime)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['completed_stages'],6)
        self.assertEqual(len(result['stages']),6)
        self.assertEqual(len(self.calls),4)
        self.assertTrue(result['stages'][1]['adopted'])
        self.assertTrue(all(s['prefetched'] for s in result['stages'][2:5]))
        for (command, options), stage in zip(self.calls[:3],self.cpu):
            self.assertEqual(command,stage['command'])
            self.assertEqual(options['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(options['env']['PYTHONPATH'],'/sealed/source')
            self.assertEqual(options['cwd'],stage['cwd'])
        self.assertEqual(self.calls[3][0],takeover.lane.wrapped_command(self.after,self.selected,self.plan))
        self.assertEqual(json.loads((self.root/'status.pre_takeover.json').read_text()),self.state)
        for s in self.cpu:
            record=json.loads((self.root/'takeover'/(s['name']+'.json')).read_text())
            self.assertEqual((record['status'],record['returncode']),('complete',0))

    def test_missing_or_failed_adopted_proof_refuses(self):
        for status in ('running','failed'):
            value=json.loads(self.wrapper.read_text());value['status']=status
            self.wrapper.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                takeover.wait_adopted(self.plan,self.selected,self.active,20,200,Mock(),self.runtime)
        self.wrapper.unlink()
        with self.assertRaises(FileNotFoundError):
            takeover.wait_adopted(self.plan,self.selected,self.active,20,200,Mock(),self.runtime)
        self.assertEqual(self.calls,[])

    def test_pid_reuse_rejected_and_zombie_uses_complete_wrapper(self):
        with self.assertRaisesRegex(RuntimeError,'PID reused'):
            takeover.alive_exact(20,200,lambda pid:dict(pid=pid,startticks=201,state='R'))
        self.assertFalse(takeover.alive_exact(20,200,self.runtime.inspect))
        takeover.wait_adopted(self.plan,self.selected,self.active,20,200,Mock(),self.runtime)
        value=json.loads(self.wrapper.read_text());value['pid']=21
        self.wrapper.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            takeover.wait_adopted(self.plan,self.selected,self.active,20,200,Mock(),self.runtime)

    def test_preflight_has_no_writes_and_existing_cpu_output_rejected(self):
        before=set(self.base.rglob('*'))
        takeover.preflight(self.plan,self.state,20,200,runtime=self.runtime,require_old_terminal=True)
        self.assertEqual(before,set(self.base.rglob('*')))
        Path(self.cpu[0]['output']).mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            takeover.execute(self.plan_path,20,200,runtime=self.runtime)
        self.assertEqual(self.calls,[])
        self.assertFalse((self.root/'status.pre_takeover.json').exists())
        self.assertFalse((self.root/'takeover').exists())

    def test_cpu_exit_zero_missing_proof_is_failed(self):
        output=self.base/'cpu_receipts';output.mkdir()
        child=Mock(pid=123);child.poll.return_value=0
        runtime=Mock();runtime.inspect.return_value={'startticks':100};runtime.popen.return_value=child
        prep=takeover.CPUPreparation([self.cpu[0]],output,self.selected,runtime)
        prep.start();prep.poll()
        self.assertEqual(runtime.popen.call_count,1)
        self.assertEqual(prep.jobs[self.cpu[0]['name']][2]['status'],'failed')
        with self.assertRaises(RuntimeError):
            prep.finish()
        self.assertEqual(runtime.popen.call_count,1)

    def test_cpu_failure_does_not_interrupt_adopted_gpu(self):
        runtime=Mock()
        runtime.inspect.side_effect=[dict(pid=20,startticks=200,state='R'),
                                    dict(pid=20,startticks=200,state='Z')]
        prep=Mock(errors=['CPU preparation failed'])
        takeover.wait_adopted(self.plan,self.selected,self.active,20,200,prep,runtime)
        prep.poll.assert_called_once()
        runtime.sleep.assert_called_once_with(5)
        runtime.popen.assert_not_called()


if __name__=='__main__':
    unittest.main()
