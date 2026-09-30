"""CPU-only process/receipt tests; no real signals, CUDA or remote access."""
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from . import dp_package_handoff as h


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.output = self.base/'migration'
        self.root = self.output/'dynamic_pool'
        self.training = self.base/'training'
        self.training.mkdir()
        (self.training/'last.pt').write_bytes(b'atomic committed checkpoint')
        self.source = self.base/'scorer_dp_runtime.py'
        self.source.write_text('# sealed fixture\n')
        self.env = {'PYTHONPATH': str(self.base), 'HOME': '/root'}
        module = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.train'
        self.stages = []
        for i, name in enumerate(['edge_multi_C16_train']+['edge_multi_'+x for x in (
                'C16_test','C16_real','C16_ood','C8_test','C8_real','C8_ood')]):
            stage = dict(name=name,gpu=True,command=['/fixed/python','-m',module,
                '--arm','edge_multi','--output',str(self.training)] if i==0 else
                ['/fixed/python','-m','fixed.evaluate','--split',name],
                cwd=str(self.base),env=deepcopy(self.env),completion=str(self.base/(name+'.json')),
                completion_expect=dict(status='complete',completed_segments=64) if i==0 else dict(status='complete'),
                original_runtime_receipt=str(self.output/'lane1'/('runtime_'+name+'.json')))
            self.stages.append(stage)
        self.m20 = dict(name='M20_all_tokens_C16_train',gpu=True,
            command=['/fixed/python','-m','fixed.m20'],cwd=str(self.base),env=self.env,
            completion=str(self.base/'m20.json'),completion_expect=dict(status='complete'),
            original_runtime_receipt=str(self.output/'lane6'/'runtime_M20_all_tokens_C16_train.json'))
        self.plan = dict(output_root=str(self.output),device_wrapper=str(self.base/'device_runtime.py'),
            inventory=[dict(index=i,uuid='GPU-'+str(i)) for i in range(7)],lanes=[])
        for i in range(7):
            stages = self.stages if i==1 else [self.m20] if i==6 else []
            self.plan['lanes'].append(dict(name='lane'+str(i),gpu_uuid='GPU-'+str(i),stages=stages))
            directory=self.output/('lane'+str(i)); directory.mkdir(parents=True)
            h.lane.save(directory/'status.json',dict(status='running' if stages else 'complete',
                completed_stages=0,stages=[],active_pid=None))
        self.plan_path=self.base/'plan.json'; h.lane.save(self.plan_path,self.plan)
        self.original_receipt=self.root/'experiments'/'edge_multi'/'runtime_edge_multi_C16_train.json'
        self.original_receipt.parent.mkdir(parents=True)
        h.lane.save(self.original_receipt,dict(status='running',pid=101))
        old_command=['/fixed/python','/fixed/device_runtime.py','--receipt',str(self.original_receipt)]
        with h.registry.locked_registry(self.root) as records:
            records['experiments']['edge_multi']=dict(name='edge_multi',status='running',stages=self.stages,
                assigned_gpu='GPU-2',worker_pid=100,worker_startticks=10,active_pid=101,
                active_startticks=11,active_stage=self.stages[0]['name'],completed_stages=0)
            records['experiments']['M20_all_tokens']=dict(name='M20_all_tokens',status='queued',
                stages=[self.m20],assigned_gpu=None)
            for stage in self.stages+[self.m20]:
                key=h.pool.receipt_key(stage)
                records['stages'][key]=dict(status='queued',stage_name=stage['name'],
                    experiment='edge_multi' if stage in self.stages else 'M20_all_tokens')
            records['stages'][h.pool.receipt_key(self.stages[0])].update(status='running',
                worker_pid=100,worker_startticks=10,pid=101,startticks=11,command=old_command)
        self.runtime_receipt=self.base/'new_training_runtime.json'
        self.spec=dict(schema=h.SCHEMA,gpu_indices=[2,0,3,6],cwd=str(self.base),env=self.env,
            source_sha256={str(self.source):h.digest(self.source)},command=['/fixed/python',str(self.source),
                '--gpus','GPU-2,GPU-0,GPU-3,GPU-6','--lock-root',str(self.output/'gpu_locks'),
                '--receipt',str(self.runtime_receipt),'--mode','resume','--module','matched_only.train',
                '--']+self.stages[0]['command'][3:]+['--resume'])

    def runtime(self, *, child_code=0, valid=True, stuck_leaf=False):
        test=self
        class Fake:
            def __init__(self):
                self.processes={p:dict(pid=p,startticks=t,state='S') for p,t in [(100,10),(101,11),(900,90)]}
                self.signals=[];self.calls=[];self.t=0.
            def pid(self):return 900
            def inspect(self,pid):return self.processes.get(pid)
            def monotonic(self):return self.t
            def sleep(self,seconds):self.t+=seconds
            def foreign(self,uuid):
                if uuid!='GPU-2':
                    with (test.root/'dispatch.lock').open('a+') as handle:
                        with test.assertRaises(BlockingIOError):
                            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                return []
            def signal(self,pid,sig):
                self.signals.append((pid,sig))
                if sig==signal.SIGSTOP:
                    test.assertEqual(pid,100)
                    with (test.root/'dispatch.lock').open('a+') as handle:
                        with test.assertRaises(BlockingIOError):
                            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    self.processes[pid]['state']='T'
                elif sig==signal.SIGINT:
                    test.assertEqual(pid,101)
                    if not stuck_leaf:
                        self.processes[pid]['state']='Z'
                        h.lane.save(test.original_receipt,dict(status='failed',error='KeyboardInterrupt'))
                elif sig==signal.SIGKILL:
                    test.assertEqual(pid,100)
                    exp=h.registry.read_registry(test.root)['experiments']['edge_multi']
                    test.assertEqual((exp['worker_pid'],exp['worker_startticks']),(900,90))
                    self.processes[pid]['state']='Z'
                elif sig==signal.SIGCONT:self.processes[pid]['state']='S'
                else:raise AssertionError('unexpected signal')
            def popen(self,command,**options):
                index=len(self.calls);self.calls.append((command,options))
                test.assertTrue(options['start_new_session'])
                if index==0:
                    desc=json.loads(options['env']['RACHEL_DP_LEASE_FDS_JSON'])
                    test.assertEqual([x['uuid'] for x in desc],['GPU-'+str(i) for i in test.spec['gpu_indices']])
                    test.assertEqual(options['pass_fds'],tuple(x['fd'] for x in desc))
                    for row in desc:
                        test.assertEqual(os.fstat(row['fd']).st_ino,Path(row['path']).stat().st_ino)
                        with Path(row['path']).open('a+') as handle:
                            with test.assertRaises(BlockingIOError):fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    test.assertEqual(options['env']['RACHEL_DP_RESUME_CHECKPOINT_SHA256'],h.digest(test.training/'last.pt'))
                else:
                    test.assertEqual(options['env']['CUDA_VISIBLE_DEVICES'],'GPU-2')
                    test.assertNotIn('pass_fds',options)
                pid=1000+index;self.processes[pid]=dict(pid=pid,state='R',startticks=100+index)
                runtime=self
                class Child:
                    def __init__(self):self.pid=pid
                    def wait(self):
                        stage=test.stages[index]
                        h.lane.save(stage['completion'],stage['completion_expect'] if valid else dict(status='running'))
                        if index==0:h.lane.save(test.runtime_receipt,dict(status='complete' if valid else 'failed'))
                        runtime.processes[pid]['state']='Z'
                        return child_code
                return Child()
        return Fake()

    def test_prepare_is_read_only_and_accepts_alternative_gpu1(self):
        original=(self.root/'registry.json').read_bytes()
        value=h.prepare(self.plan_path,self.root,self.spec)
        self.assertEqual(value['uuids'],['GPU-2','GPU-0','GPU-3','GPU-6'])
        self.spec['gpu_indices']=[2,1,3,6]
        self.spec['command'][self.spec['command'].index('--gpus')+1]='GPU-2,GPU-1,GPU-3,GPU-6'
        self.assertEqual(h.prepare(self.plan_path,self.root,self.spec)['uuids'][1],'GPU-1')
        self.assertEqual(original,(self.root/'registry.json').read_bytes())
        self.assertFalse((self.original_receipt.parent/'dp_handoff_v1').exists())

    def test_owned_m20_card_and_untransferred_gpu6_fail_before_signals(self):
        with h.registry.locked_registry(self.root) as records:
            records['experiments']['M20_all_tokens'].update(status='running',assigned_gpu='GPU-0')
        with self.assertRaisesRegex(ValueError,'another package'):
            h.prepare(self.plan_path,self.root,self.spec)
        with h.registry.locked_registry(self.root) as records:
            records['experiments']['M20_all_tokens']['assigned_gpu']='GPU-5'
            del records['stages'][h.pool.receipt_key(self.m20)]
        with self.assertRaisesRegex(ValueError,'not delegated'):
            h.prepare(self.plan_path,self.root,self.spec)

    def test_unchanged_args_and_source_hash_required(self):
        self.spec['command'].append('--different-budget')
        with self.assertRaisesRegex(ValueError,'unchanged resume CLI'):
            h.prepare(self.plan_path,self.root,self.spec)
        self.spec['command'].pop();self.source.write_text('# mutated\n')
        with self.assertRaisesRegex(ValueError,'source changed'):
            h.prepare(self.plan_path,self.root,self.spec)

    def test_short_original_module_alias(self):
        with h.registry.locked_registry(self.root) as records:
            records['experiments']['edge_multi']['stages'][0]['command'][2]='matched_only.train'
        h.prepare(self.plan_path,self.root,self.spec)

    def test_success_owner_commit_signal_order_fd_order_and_original_endpoints(self):
        rt=self.runtime()
        result=h.execute(self.plan_path,self.root,self.spec,runtime=rt)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(rt.signals,[(100,signal.SIGSTOP),(101,signal.SIGINT),(100,signal.SIGKILL)])
        self.assertEqual(len(rt.calls),7)
        records=h.registry.read_registry(self.root);exp=records['experiments']['edge_multi']
        self.assertEqual((exp['status'],exp['completed_stages']),('complete',7))
        self.assertEqual(records['experiments']['M20_all_tokens']['status'],'queued')
        row=records['stages'][h.pool.receipt_key(self.stages[0])]
        self.assertEqual(row['attempt_history'][0]['status'],'interrupted_for_dp_handoff')
        self.assertNotEqual(row['attempt_history'][0].get('returncode'),0)
        self.assertEqual(h.lane.read(self.original_receipt),dict(status='failed',error='KeyboardInterrupt'))
        for index,(command,_) in enumerate(rt.calls[1:],1):
            self.assertEqual(command[command.index('--')+1:],self.stages[index]['command'][3:])
            self.assertEqual(h.option(command,'--module'),self.stages[index]['command'][2])
        self.assertEqual(records['claims'],{})

    def test_racing_m20_reservation_refuses_before_stop(self):
        rt=self.runtime();prepare=h.prepare
        def preflight_then_m20(*args):
            context=prepare(*args)
            with h.registry.locked_registry(self.root) as records:
                records['experiments']['M20_all_tokens'].update(status='running',assigned_gpu='GPU-0')
            return context
        with patch.object(h,'prepare',side_effect=preflight_then_m20):
            with self.assertRaisesRegex(ValueError,'another package'):
                h.execute(self.plan_path,self.root,self.spec,runtime=rt)
        self.assertEqual(rt.signals,[]);self.assertEqual(rt.calls,[])

    def test_pid_reuse_refuses_without_stop(self):
        rt=self.runtime();rt.processes[101]['startticks']=12
        with self.assertRaisesRegex(RuntimeError,'PID identity reused'):
            h.execute(self.plan_path,self.root,self.spec,runtime=rt)
        self.assertEqual(rt.signals,[])

    def test_busy_extra_gpu_refuses_without_stop(self):
        lock=self.output/'gpu_locks'/'GPU-0.lock';lock.parent.mkdir()
        with lock.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            rt=self.runtime()
            with self.assertRaises(BlockingIOError):h.execute(self.plan_path,self.root,self.spec,runtime=rt)
        self.assertEqual(rt.signals,[])

    def test_interrupted_leaf_timeout_never_kills_or_resumes_any_worker(self):
        rt=self.runtime(stuck_leaf=True)
        with self.assertRaises(TimeoutError):
            h.execute(self.plan_path,self.root,self.spec,runtime=rt,interrupt_timeout=.2)
        self.assertEqual(rt.signals,[(100,signal.SIGSTOP),(101,signal.SIGINT)])
        self.assertEqual(rt.calls,[])
        receipt=h.lane.read(self.original_receipt.parent/'dp_handoff_v1'/'handoff.json')
        self.assertTrue(receipt['manual_recovery_required']);self.assertFalse(receipt['adopted'])
        self.assertEqual(h.registry.read_registry(self.root)['experiments']['edge_multi']['worker_pid'],100)

    def test_zero_exit_without_authoritative_complete_does_not_dispatch_endpoints(self):
        rt=self.runtime(valid=False)
        with self.assertRaises(ValueError):h.execute(self.plan_path,self.root,self.spec,runtime=rt)
        self.assertEqual(len(rt.calls),1)
        exp=h.registry.read_registry(self.root)['experiments']['edge_multi']
        self.assertEqual(exp['status'],'failed')
        self.assertEqual(h.lane.read(self.original_receipt)['status'],'failed')

    def test_nonzero_child_no_retry(self):
        rt=self.runtime(child_code=7)
        with self.assertRaisesRegex(RuntimeError,'resume attempt failed'):
            h.execute(self.plan_path,self.root,self.spec,runtime=rt)
        self.assertEqual(len(rt.calls),1)
        self.assertEqual(h.registry.read_registry(self.root)['experiments']['edge_multi']['status'],'failed')

    def test_close_preserves_inherited_open_description_lock(self):
        rt=self.runtime();rt.foreign=lambda uuid:[]
        leases=h.Leases(self.output/'gpu_locks',rt);leases.acquire('GPU-0')
        record=leases.descriptor_records(['GPU-0'])[0];inherited=os.dup(record['fd'])
        try:
            leases.close()
            with Path(record['path']).open('a+') as stream:
                with self.assertRaises(BlockingIOError):fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            os.close(inherited);inherited=None
            with Path(record['path']).open('a+') as stream:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:
            if inherited is not None:os.close(inherited)
            leases.close()


if __name__=='__main__':unittest.main()
