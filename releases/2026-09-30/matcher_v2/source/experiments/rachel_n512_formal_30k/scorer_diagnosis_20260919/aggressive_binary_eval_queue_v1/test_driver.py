from contextlib import ExitStack
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import launch_evaluation as launch
from queue_contracts import read,save,hashes

VARIANT=os.environ.get('BINARY_VERIFY_VARIANT','patch')

class DriverTests(unittest.TestCase):
    def run_queue(self,root,failure=False):
        out=root/'evaluation';out.mkdir();(out/'logs').mkdir()
        inventory={0:{'uuid':'GPU-0','name':'synthetic'},1:{'uuid':'GPU-1','name':'synthetic'}}
        bindings={'fixture':'same'};selected={'source_bindings':bindings}
        save(out/'selected_models.json',selected)
        save(out/'plan.json',dict(queue_python_sha256=hashes(Path(launch.__file__).parent),
                                gpu_inventory={str(i):v for i,v in inventory.items()}))
        args=SimpleNamespace(root=str(root),prepared=str(root/'prepared'),out=str(out),variant=VARIANT,gpus='0,1')
        started=[];state={'live':0,'peak':0};workers=[]
        class Process:
            def __init__(self,command,**kwargs):
                self.index=len(started);self.pid=100+self.index;self.finished=False
                started.append(dict(command=command,env=kwargs['env']));state['live']+=1
                state['peak']=max(state['peak'],state['live'])
            def poll(self):
                if not self.finished:state['live']-=1;self.finished=True
                return 1 if failure and self.index==0 else 0
        def worker(_args,operation,*unused,**kwargs):
            workers.append(operation)
            if operation=='trace':
                return dict(schema='binary-trace-device-gate/1',status='passed',variant=VARIANT,device='cuda:0',
                    parameters_unchanged=True,rng_unchanged=True,capture_bitwise_equal=True,numeric_replay_passed=True,
                    real_inference_performed=False,source_bindings=bindings,head_parameters=34529 if VARIANT=='patch' else 3201,
                    optimizer_updates=0,trained_checkpoint_opened=False)
            return dict(status='passed')
        with ExitStack() as stack:
            stack.enter_context(patch.object(launch,'verify_preparation',return_value={}))
            stack.enter_context(patch.object(launch,'validate_selected_gate',return_value=selected))
            stack.enter_context(patch.object(launch,'gpu_inventory',return_value=inventory))
            free=stack.enter_context(patch.object(launch,'assert_free'))
            stack.enter_context(patch.object(launch,'execute_worker',side_effect=worker))
            stack.enter_context(patch.object(launch,'identity',side_effect=lambda pid:{'pid':pid,'starttime':1,'cmdline':'mock'}))
            stack.enter_context(patch.object(launch.subprocess,'Popen',side_effect=Process))
            stack.enter_context(patch.object(launch.time,'sleep'))
            if failure:
                with self.assertRaisesRegex(RuntimeError,'evaluation incomplete'):launch.driver(args)
            else:launch.driver(args)
            self.assertGreater(free.call_count,0)
        self.assertEqual(state['live'],0);self.assertLessEqual(state['peak'],2)
        self.assertEqual(workers[0],'trace')
        self.assertTrue(all(x['env']['CUDA_VISIBLE_DEVICES'] in ('0','1') for x in started))
        return out,started,workers

    def test_six_jobs_finish_once_with_at_most_two_live(self):
        with tempfile.TemporaryDirectory() as t:
            out,started,workers=self.run_queue(Path(t))
            self.assertEqual(len(started),6);self.assertEqual(workers.count('verify-job'),6)
            complete=read(out/'evaluation_complete.json');self.assertEqual(complete['status'],'complete')
            self.assertEqual(len(complete['jobs']),6);self.assertEqual(complete['fixed_case_evaluations'],22)
            self.assertTrue(all(j['returncode']==0 and j['status']=='complete' for j in complete['jobs']))
            self.assertFalse(complete['training_modified']);self.assertEqual(complete['automatic_retries'],0)

    def test_one_failure_drains_sibling_and_never_starts_more(self):
        with tempfile.TemporaryDirectory() as t:
            out,started,workers=self.run_queue(Path(t),failure=True)
            self.assertEqual(len(started),2);self.assertEqual(workers.count('verify-job'),1)
            self.assertTrue((out/'failure.json').exists());self.assertFalse((out/'evaluation_complete.json').exists())
            self.assertEqual(len(read(out/'failure.json')['jobs']),2)

if __name__=='__main__':unittest.main()

