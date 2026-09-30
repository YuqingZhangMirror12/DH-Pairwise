import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

import contracts as c
import dispatch as d
import jobs as j


class RegistryTests(unittest.TestCase):
    def test_exact15_jobs_only_completed_b1_b2_b3(self):
        rows=j.registry()
        self.assertEqual(len(rows),15);self.assertEqual(len({r['id'] for r in rows}),15)
        self.assertEqual({r['arm'] for r in rows},{'B1','B2','B3'})
        self.assertEqual([r['arm'] for r in rows[:5]],['B3']*5)

    def test_no_conditional_ablations_or_b0_repeats(self):
        for arm in ('B0','B4','B5','B6'):
            with self.assertRaises(ValueError):c.paths(arm,'matcher')
        with self.assertRaises(ValueError):j.check_job(dict(id='extra',arm='B3',module='matcher',kind='test'))

    def test_b3_priority_and_bounded_capacity(self):
        pending=j.registry();ready={(a,m):{} for a in c.ARMS for m in c.MODULES}
        self.assertEqual([r['arm'] for r in d.choose_ready(pending,ready,2)],['B3','B3'])
        self.assertEqual(d.choose_ready(pending,ready,0),[])
        with self.assertRaises(ValueError):d.choose_ready(pending,ready,3)

    def test_slow_b3_does_not_block_ready_other_arm(self):
        chosen=d.choose_ready(j.registry(),{('B1','matcher'):{}},2)
        self.assertEqual([x['arm'] for x in chosen],['B1','B1'])

    def test_no_unready_jobs_are_selected(self):self.assertEqual(d.choose_ready(j.registry(),{},2),[])

    def test_cpu_environment_overrides_parent_cuda_threads(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='64'):
            env=c.cpu_environment()
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
        self.assertTrue(all(env[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')))

    def test_commands_never_train_or_select_or_use_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=root/'spec.json';c.save(spec,{})
            ready=dict(execution=c.bound(spec))
            with patch.object(c,'paths',return_value=(spec,root)):
                for job in j.registry():
                    values=j.command(job,root/'out',ready)
                    self.assertEqual(values[0],str(c.PYTHON))
                    self.assertNotIn('--gpus',values);self.assertNotIn('--device',values)
                    self.assertNotIn('--threshold',values);self.assertNotIn('--checkpoint',values)
                    self.assertFalse(any('launcher.py' in value or '.launcher' in value for value in values))
                    self.assertNotIn('turufan',values);self.assertNotIn('sim_test',values)
                    if job['kind']!='native_budgets':self.assertEqual(values[values.index('--selection')+1],'sim_best')

    def test_changed_execution_rejected_before_building_command(self):
        with patch.object(c,'bound',return_value={'path':'changed'}):
            with self.assertRaisesRegex(ValueError,'execution changed'):
                j.command(j.registry()[0],Path('/out'),dict(execution={}))


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.override=patch.object(c,'ROOT',self.root);self.override.start()
        self.addCleanup(self.override.stop);self.addCleanup(self.tmp.cleanup)

    def make(self,path,value):path.parent.mkdir(parents=True,exist_ok=True);c.save(path,value)

    def test_launch_gate12_or_progress_cannot_make_terminal_ready(self):
        _,root=c.paths('B3','matcher')
        self.make(root/'training/formal/status.json',{'completed_updates':31667})
        self.make(root/'training/gpu_gate.json',{'updates':12,'passed':True})
        self.assertIsNone(c.terminal_ready('B3','matcher',{}))

    def test_actual_failure_precedes_stale_complete(self):
        _,root=c.paths('B3','matcher');self.make(root/'complete.json',{'status':'complete'})
        self.make(root/'training/formal/failure_attempt_7.json',{'error':'actual'})
        with self.assertRaisesRegex(ValueError,'upstream failure'):c.terminal_ready('B3','matcher',{})

    def test_head_queue_per_lane_failure_is_visible(self):
        self.make(self.root/'b3_head_queue_01/scorer_patch/failure.json',{'error':'actual'})
        self.assertIsNotNone(c.upstream_failure('B3','scorer_patch'))
        self.assertIsNone(c.upstream_failure('B3','scorer_stats'))

    def test_control_failure_is_not_endless_wait(self):
        self.make(self.root/'b12_control_queue_01/B2/failure.json',{'error':'actual'})
        with self.assertRaisesRegex(ValueError,'upstream failure'):c.terminal_ready('B2','matcher',{})
        self.assertIsNone(c.upstream_failure('B1','matcher'))

    def fixture(self,arm='B3',module='matcher',budget=31667):
        spec,root=c.paths(arm,module);self.make(spec,dict(arm=arm,module=module))
        actual=dict(status='complete',arm=arm,module=module)
        self.make(root/'complete.json',actual)
        origin=dict(arm=arm,module=module,selection_kind='sim_best',total_completed_updates=budget,real_used_for_selection=False)
        runtime=dict(runtime_inputs=SimpleNamespace(load_inputs=Mock(return_value=dict(plan=SimpleNamespace(record={'total_updates':budget})))),
            pipeline=SimpleNamespace(commands=Mock(return_value=('train','evaluate')),verify_all=Mock(return_value=dict(actual,completed_unix=1))),
            terminal=SimpleNamespace(verified_export=Mock(return_value=({},origin))))
        return root,runtime

    def test_complete_receipt_still_invokes_full_verifier_and_terminal_export(self):
        root,runtime=self.fixture();result=c.terminal_ready('B3','matcher',runtime)
        self.assertTrue(result['ordinary_terminal_evaluations_verified'])
        runtime['pipeline'].verify_all.assert_called_once();runtime['terminal'].verified_export.assert_called_once()
        self.assertEqual(result['pipeline_complete'],c.bound(root/'complete.json'))

    def test_incorrect_locked_budget_rejected(self):
        _,runtime=self.fixture(budget=24000)
        with self.assertRaisesRegex(ValueError,'budget'):c.terminal_ready('B3','matcher',runtime)

    def test_incomplete_mandatory_evaluation_never_becomes_ready(self):
        _,runtime=self.fixture();runtime['pipeline'].verify_all.side_effect=ValueError('missing12job')
        with self.assertRaisesRegex(ValueError,'missing12job'):c.terminal_ready('B3','matcher',runtime)
        runtime['terminal'].verified_export.assert_not_called()

    def test_different_terminal_arm_or_real_choice_rejected(self):
        _,runtime=self.fixture();runtime['terminal'].verified_export.return_value[1]['arm']='B1'
        with self.assertRaisesRegex(ValueError,'own completed'):c.terminal_ready('B3','matcher',runtime)

    def test_heads_use_registered_single_gpu_identity_only(self):
        _,runtime=self.fixture(module='scorer_stats')
        c.terminal_ready('B3','scorer_stats',runtime)
        args=runtime['pipeline'].commands.call_args.args[0]
        self.assertEqual(args.gpus,[5])


class ExecutionTests(unittest.TestCase):
    def run_child(self,code,verify=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=root/'spec.json';end=root/'prior.json'
            c.save(spec,{});c.save(end,{})
            ready=dict(execution=c.bound(spec),pipeline_complete=c.bound(end))
            job=j.registry()[0];callback=verify or Mock(return_value={'audited':True})
            values=[sys.executable,'-c','import sys; sys.exit('+str(code)+')']
            with patch.object(c,'ROOT',root),patch.object(d,'unchanged'),patch.object(c,'upstream_failure',return_value=None),\
                 patch.object(j,'command',return_value=values),patch.object(d,'identity',side_effect=lambda pid:dict(pid=pid)):
                if code or isinstance(callback.side_effect,Exception):
                    with self.assertRaises(Exception):d.execute(job,root,ready,{},verify=callback)
                else:d.execute(job,root,ready,{},verify=callback)
            work=root/job['id']
            return c.read(work/'return.json'),(work/'complete.json').exists(),(work/'failure.json').exists(),callback

    def test_real_child_exit_zero_plus_audit_required(self):
        returned,complete,failed,callback=self.run_child(0)
        self.assertEqual(returned['returncode'],0);self.assertTrue(complete);self.assertFalse(failed)
        callback.assert_called_once()

    def test_real_child_exit_nonzero_preserved_no_retry_or_audit(self):
        returned,complete,failed,callback=self.run_child(3)
        self.assertEqual(returned['returncode'],3);self.assertFalse(complete);self.assertTrue(failed)
        callback.assert_not_called()

    def test_zero_exit_and_failed_artifact_audit_not_complete(self):
        returned,complete,failed,_=self.run_child(0,Mock(side_effect=ValueError('bad artifact')))
        self.assertEqual(returned['returncode'],0);self.assertFalse(complete);self.assertTrue(failed)

    def test_source_change_stops_before_process_start(self):
        popen=Mock();job=j.registry()[0]
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'unchanged',side_effect=ValueError('changed')):
            with self.assertRaisesRegex(ValueError,'changed'):d.execute(job,Path(tmp),{}, {},popen=popen)
        popen.assert_not_called()

    def test_existing_output_not_overwritten_or_relaunched(self):
        job=j.registry()[0];popen=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/job['id']).mkdir()
            with self.assertRaises(FileExistsError):d.execute(job,root,{}, {},popen=popen)
        popen.assert_not_called()


class ArtifactTests(unittest.TestCase):
    def test_empty_file_manifest_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'membership'):j.file_set(Path(tmp),dict(files={}),('summary.json',))

    def test_artifact_hash_mutation_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c.save(root/'summary.json',{})
            with self.assertRaisesRegex(ValueError,'hash'):j.file_set(root,dict(files={'summary.json':'bad'}),('summary.json',))

    def test_child_failure_precedes_valid_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c.save(root/'failure.json',{})
            with self.assertRaisesRegex(ValueError,'failure'):j.file_set(root,dict(files={}),())

    def test_origin_cannot_cross_arm_or_weights(self):
        fields=('arm','module','selection_kind','checkpoint_sha256','model_state_sha256',
                'matcher_state_sha256','selected_updates','total_completed_updates','common_plan_sha256')
        origin={k:str(i) for i,k in enumerate(fields)};origin['checkpoint']='model.pt'
        ready=dict(origin=copy.deepcopy(origin));origin['matcher_state_sha256']='other'
        with self.assertRaisesRegex(ValueError,'differs'):c.check_origin(origin,ready)

    def test_preparation_requires_actual_runtime_preflight(self):
        proof=dict(status='passed',tests=28,errors=0,failures=0,skipped=0,source_unchanged=True,
            source_sha256={},dependencies_sha256={},cuda_initialized=False,runtime_preflight={'status':'missing','gpu_probed':False})
        with patch.object(c,'read',return_value=proof),patch.object(c,'own_code',return_value={}),\
             patch.object(c,'dependency_code',return_value={}):
            with self.assertRaisesRegex(ValueError,'CPU preparation'):d.verify_preparation(Path('/proof'))


if __name__=='__main__':unittest.main()
