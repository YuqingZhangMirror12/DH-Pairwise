import json
from pathlib import Path
import tempfile
from unittest import TestCase,main
from unittest.mock import patch,Mock
import ready

class ReadyTests(TestCase):
    def binding(self,root):
        ready.save(root/'config.json',{'config':'fixture'});ready.save(root/'pipeline_launch.json',{'pid':1})
        return dict(launch_sha256=ready.sha(root/'pipeline_launch.json'),config_sha256=ready.sha(root/'config.json'),controller={'pid':1})
    def test_only_live_bound_process_counts_as_wait(self):
        with tempfile.TemporaryDirectory() as t,patch.object(ready,'DATA',Path(t)),patch.object(ready,'live',return_value=True):
            self.assertEqual(ready.state(self.binding(Path(t))),'waiting_for_live_data_pipeline')
    def test_status_file_alone_cannot_replace_actual_process(self):
        with tempfile.TemporaryDirectory() as t,patch.object(ready,'DATA',Path(t)),patch.object(ready,'live',return_value=False):
            root=Path(t);b=self.binding(root);ready.save(root/'pipeline_status.json',{'status':'running'})
            with self.assertRaisesRegex(RuntimeError,'exited'):ready.state(b)
    def test_failure_precedes_terminal(self):
        with tempfile.TemporaryDirectory() as t,patch.object(ready,'DATA',Path(t)):
            root=Path(t);b=self.binding(root)
            for name in ('pipeline_failure.json','pipeline_complete.json'):ready.save(root/name,{})
            with self.assertRaisesRegex(RuntimeError,'failed'):ready.state(b)
    def test_terminal_can_survive_producer_exit(self):
        with tempfile.TemporaryDirectory() as t,patch.object(ready,'DATA',Path(t)),patch.object(ready,'live',return_value=False):
            root=Path(t);b=self.binding(root);ready.save(root/'pipeline_complete.json',{})
            self.assertEqual(ready.state(b),'data_terminal')
    def test_changed_launch_or_config_never_attaches_new_process(self):
        for name in ('config.json','pipeline_launch.json'):
            with tempfile.TemporaryDirectory() as t,patch.object(ready,'DATA',Path(t)):
                root=Path(t);b=self.binding(root);ready.save(root/name,{'different':'task'})
                with self.assertRaises(ValueError):ready.state(b)
    def test_pid_reuse_and_zombie_are_not_live(self):
        expected=dict(pid=123,starttime=456,cmdline='bound data pipeline')
        for update in ({'starttime':457},{'cmdline':'other'},{'state':'Z'}):
            actual=dict(expected,state='S');actual.update(update)
            with patch.object(ready,'identity',return_value=actual):self.assertFalse(ready.live(expected))
        with patch.object(ready,'identity',return_value=dict(expected,state='S')):self.assertTrue(ready.live(expected))
    def test_existing_valid_admission_not_recreated(self):
        with tempfile.TemporaryDirectory() as t,patch.object(ready,'ADMISSION',Path(t)/'admission.json'),patch.object(ready,'verify_admission',return_value={}) as check,patch.object(ready.subprocess,'Popen') as start:
            ready.ADMISSION.write_text('{}');r=ready.admit();self.assertFalse(r['new_registration']);check.assert_called_once();start.assert_not_called()
    def test_registrar_failure_is_not_retried_or_treated_as_training(self):
        with tempfile.TemporaryDirectory() as t,patch.object(ready,'ROOT',Path(t)),patch.object(ready,'ADMISSION',Path(t)/'absent.json'),patch.object(ready,'identity',return_value={'pid':1}),patch.object(ready.subprocess,'Popen') as start,patch.object(ready,'verify_admission') as verify:
            start.return_value=Mock(pid=1);start.return_value.wait.return_value=1
            with self.assertRaisesRegex(RuntimeError,'refused'):ready.admit()
            self.assertEqual(start.call_count,1);verify.assert_not_called()
            cmd=start.call_args[0][0];self.assertEqual(cmd,[ready.PYTHON,str(ready.REGISTER)])
            self.assertEqual(start.call_args[1]['env']['CUDA_VISIBLE_DEVICES'],'')
    def test_successful_registrar_must_yield_verified_admission(self):
        with tempfile.TemporaryDirectory() as t,patch.object(ready,'ROOT',Path(t)),patch.object(ready,'ADMISSION',Path(t)/'absent.json'),patch.object(ready,'identity',return_value={'pid':1}),patch.object(ready.subprocess,'Popen') as start,patch.object(ready,'verify_admission') as verify:
            start.return_value=Mock(pid=1);start.return_value.wait.return_value=0
            r=ready.admit();self.assertTrue(r['new_registration']);verify.assert_called_once()
    def test_environment_never_exposes_gpu(self):
        with patch.dict(ready.os.environ,CUDA_VISIBLE_DEVICES='0,1'):
            env=ready.environment();self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(env['OMP_NUM_THREADS'],'1')

    def test_existing_admission_must_bind_terminal_calibration_and_exact_queue_command(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data=root/'data';queue=root/'queue';register=root/'register.py';admission=root/'admission.json'
            files={}
            for name in ('pipeline_complete.json','data_contract.json','human_approval.json','geometry_calibration_v2/geometry_calibration.json'):
                p=data/name;ready.save(p,{'fixture':name});files[str(p)]=ready.sha(p)
            record=dict(schema='verified-future-gpu-task/1',status='ready',task='aggressive_scratch',
                work=str(queue/'work/aggressive_scratch'),gpu_jobs_started=False,data_admission={'passed':True},files_sha256=files,
                command=[ready.PYTHON,str(register.with_name('run_queued.py')),'--gpus','{gpus}',
                    '--release','{release}','--work',str(queue/'work/aggressive_scratch')])
            with patch.object(ready,'ADMISSION',admission),patch.object(ready,'DATA',data),patch.object(ready,'QUEUE',queue),patch.object(ready,'REGISTER',register):
                ready.save(admission,record);self.assertEqual(ready.verify_admission()['status'],'ready')
                for field,value in [('task','joint_e32'),('gpu_jobs_started',True),('files_sha256',{}),('command',['python','train.py'])]:
                    bad=dict(record);bad[field]=value;ready.save(admission,bad)
                    with self.assertRaises(ValueError):ready.verify_admission()
                ready.save(admission,record);ready.save(data/'data_contract.json',{'changed':True})
                with self.assertRaisesRegex(ValueError,'changed'):ready.verify_admission()

if __name__=='__main__':main()
