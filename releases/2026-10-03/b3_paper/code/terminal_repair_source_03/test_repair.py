import argparse
import ast
import copy
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import common as c
import controller as ec
import pipeline as p
import continue_b3 as follow
import strict_population_bridge as bridge


class CommonTests(unittest.TestCase):
    def test_bridge_is_identical_to_already_audited_adapter(self):
        self.assertEqual(c.sha(c.HERE/'strict_population_bridge.py'),c.BRIDGE_SHA)
    def test_return_missing_is_not_a_live_handle(self):
        self.assertFalse(c.same_process(None,lambda _:None))
    def test_live_identity_required_not_only_pid(self):
        row=dict(pid=10,starttime=123,cmdline='run')
        self.assertTrue(c.same_process(row,lambda _:dict(row,state='S')))
        self.assertFalse(c.same_process(row,lambda _:dict(row,starttime=124)))
        self.assertFalse(c.same_process(row,lambda _:dict(row,cmdline='other')))
    def test_missing_handle_is_not_restarted(self):
        def missing(_):raise FileNotFoundError()
        self.assertFalse(c.same_process(dict(pid=10),missing))
    def test_inaccessible_handle_does_not_count_as_dead(self):
        def denied(_):raise PermissionError()
        with self.assertRaises(PermissionError):c.same_process(dict(pid=10),denied)
    def test_environment_keeps_full_fp32_contract(self):
        env=c.environment('/source',[0,5]);self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,5')
        self.assertEqual(env['OMP_NUM_THREADS'],'2');self.assertEqual(env['PYTHONPATH'],'/source')
    def test_bound_file_modified_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'a';c.save(f,{});binding=c.bound(f);c.check_binding(binding)
            binding['sha256']='0'*64
            with self.assertRaises(ValueError):c.check_binding(binding)
    def test_outputs_are_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'a';c.save(f,{})
            with self.assertRaises(FileExistsError):c.save(f,{})
    def test_preparation_requires_full_current_sources(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'prep';c.save(f,dict(schema='matcher-v2-terminal-repair-cpu/1',status='passed',tests=30,
                failures=0,errors=0,skipped=0,cuda_initialized=False,source_files={}))
            with self.assertRaises(ValueError):c.check_preparation(f)
    def test_existing_correct_alias_can_be_verified_twice(self):
        import sys
        root=Path('/frozen');package=root/c.PACKAGE.replace('.','/')
        values={'consensus_binary_eval_common':SimpleNamespace(__path__=[str(package/'s7_consensus_eval_v14')]),
                'consensus_binary_eval_adapter':SimpleNamespace(__path__=[str(package/'binary_eval_v1')])}
        with patch.dict(sys.modules,values),patch.object(c,'api',side_effect=AssertionError('must not rebind')):
            c.ensure_evaluation_binding(root);c.ensure_evaluation_binding(root)
    def test_alias_to_another_runtime_rejected(self):
        import sys
        values={'consensus_binary_eval_common':SimpleNamespace(__path__=['/wrong']),
                'consensus_binary_eval_adapter':SimpleNamespace(__path__=['/wrong'])}
        with patch.dict(sys.modules,values),self.assertRaisesRegex(ValueError,'different runtime'):
            c.ensure_evaluation_binding('/frozen')


class FailedJobTests(unittest.TestCase):
    job=dict(name='sim_best_sim_straight_select',selection='sim_best',split='sim_straight_select')
    def fixture(self,root,code=1,job=None):
        job=job or self.job;name=job['name']
        c.save(root/(name+'_launch.json'),dict(job=job,process=dict(pid=12,starttime=4,cmdline='eval')))
        c.save(root/(name+'_return.json'),dict(returncode=code,launch_sha256=c.sha(root/(name+'_launch.json'))))
        # Test fixture only, never production output editing.
        (root/(name+'.log')).write_text('Traceback (most recent call last):\n'+ec.SEED_ERROR+'\n')
        return job
    def classify(self,root,job=None,verify=None,identity=None):
        return ec.classify_old_job(root,job or self.job,verify or (lambda *_:dict(passed=True)),
            identity or (lambda _:dict(pid=12,starttime=5,cmdline='other')))
    def test_known_seed_error_accepted_with_actual_return(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);got=self.classify(root)
            self.assertEqual(got['state'],'known_seed_failure');self.assertEqual(got['returned'],c.bound(root/(self.job['name']+'_return.json')))
    def test_live_child_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            with self.assertRaisesRegex(ValueError,'still active'):
                self.classify(root,identity=lambda _:dict(pid=12,starttime=4,cmdline='eval'))
    def test_missing_return_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c.save(root/(self.job['name']+'_launch.json'),dict(process=None))
            with self.assertRaisesRegex(ValueError,'return missing'):self.classify(root)
    def test_signal_exit_is_not_retryable(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,-15)
            with self.assertRaisesRegex(ValueError,'signal-related'):self.classify(root)
    def test_nonstraight_failure_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);job=dict(name='sim_best_sim_test',selection='sim_best',split='sim_test');self.fixture(root,job=job)
            with self.assertRaisesRegex(ValueError,'known canonical'):self.classify(root,job)
    def test_exact_native_mode_failure_is_admitted(self):
        for split in ('sim_select', 'sim_test'):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as d:
                root=Path(d);job=dict(name='sim_best_'+split,selection='sim_best',split=split)
                self.fixture(root,job=job)
                (root/(job['name']+'.log')).write_text(
                    'File "runtime/curriculum_training_v1/matcher_evaluation.py", line 46, in predict_batch\n'
                    +ec.MODE_ERROR+'\n')
                self.assertEqual(self.classify(root,job)['state'],'known_eval_mode_failure')
    def test_mode_error_without_native_stack_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);job=dict(name='sim_best_sim_test',selection='sim_best',split='sim_test')
            self.fixture(root,job=job);(root/(job['name']+'.log')).write_text(ec.MODE_ERROR+'\n')
            with self.assertRaisesRegex(ValueError,'known canonical'):self.classify(root,job)
    def test_mode_empty_stream_requires_exact_zero_prediction_receipt(self):
        for content,processed,valid in (('',0,True),('',1,False),('{}\n',0,False)):
            with self.subTest(content=content,processed=processed),tempfile.TemporaryDirectory() as d:
                root=Path(d);job=dict(name='sim_best_sim_select',selection='sim_best',split='sim_select')
                self.fixture(root,job=job)
                (root/(job['name']+'.log')).write_text(
                    'File "runtime/curriculum_training_v1/matcher_evaluation.py", line 46, in predict_batch\n'+ec.MODE_ERROR+'\n')
                out=root/job['name'];out.mkdir();(out/'pair_predictions.jsonl').write_text(content)
                c.save(out/'failure.json',dict(processed=processed,error="ValueError('frozen eval Matcher required')"))
                if valid:self.assertEqual(self.classify(root,job)['state'],'known_eval_mode_failure')
                else:
                    with self.assertRaises(ValueError):self.classify(root,job)
    def test_oom_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);(root/(self.job['name']+'.log')).write_text('RuntimeError: CUDA out of memory\n')
            with self.assertRaisesRegex(ValueError,'known canonical'):self.classify(root)
    def test_error_substring_is_not_enough(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);(root/(self.job['name']+'.log')).write_text(ec.SEED_ERROR+'\nNew unrelated failure')
            with self.assertRaisesRegex(ValueError,'known canonical'):self.classify(root)
    def test_any_partial_predictions_rejected(self):
        for name in ('pair_predictions.jsonl','prediction_complete.json','summary.json','evaluation_complete.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                root=Path(d);self.fixture(root);out=root/self.job['name'];out.mkdir();(out/name).touch()
                with self.assertRaisesRegex(ValueError,'already produced predictions'):self.classify(root)
    def test_unattempted_job_can_be_scheduled_once(self):
        with tempfile.TemporaryDirectory() as d:self.assertEqual(self.classify(Path(d))['state'],'unattempted')
    def test_orphaned_outputs_are_not_unattempted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/self.job['name']).mkdir()
            with self.assertRaisesRegex(ValueError,'orphaned'):self.classify(root)
    def test_old_success_requires_independent_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,0);calls=[]
            got=self.classify(root,verify=lambda r,j:calls.append(j) or {'real_artifact_audit':True})
            self.assertEqual(got['state'],'reused_success');self.assertEqual(calls,[self.job])
    def test_bad_old_success_audit_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,0)
            def bad(*_):raise ValueError('artifact changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):self.classify(root,verify=bad)
    def test_mismatched_return_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            returned=root/(self.job['name']+'_return.json');row=c.read(returned);row['launch_sha256']='0'*64
            returned.write_text(__import__('json').dumps(row))
            with self.assertRaisesRegex(ValueError,'identity differs'):self.classify(root)


class RunEquivalenceTests(unittest.TestCase):
    def runtime(self):
        return c.HERE.parent/'runtime_work_13'/c.PACKAGE.replace('.','/')/'matcher_v2_v1'
    def run_ast(self,path):
        return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    def test_final_evaluator_diff_is_only_population_dispatch_and_package_location(self):
        old=self.run_ast(self.runtime()/'evaluate.py');new=self.run_ast(c.HERE/'entry.py')
        class Normalize(ast.NodeTransformer):
            def visit_Attribute(self,node):
                node=self.generic_visit(node)
                if isinstance(node.value,ast.Name) and node.value.id=='population' and node.attr=='load_population':
                    return ast.Name(id='load_population',ctx=node.ctx)
                if isinstance(node.value,ast.Name) and node.value.id=='original' and node.attr=='__file__':
                    return ast.Name(id='__file__',ctx=node.ctx)
                return node
            def visit_BinOp(self,node):
                if isinstance(node.left,ast.Subscript) and ast.unparse(node.left)=="__package__.rsplit('.', 1)[0]":
                    node.left=ast.Name(id='PACKAGE',ctx=ast.Load())
                return self.generic_visit(node)
        self.assertEqual(ast.dump(Normalize().visit(old),include_attributes=False),
                         ast.dump(Normalize().visit(new),include_attributes=False))
    def test_bridge_keeps_original_getitem_and_targets(self):
        source=inspect.getsource(bridge.load_population)
        self.assertNotIn('def __getitem__',source);self.assertNotIn('targets_after_prediction',source)
    def test_negative_seed_fallback_matches_preparation(self):
        seen=[]
        api=SimpleNamespace(canonical_entry=lambda e,a,r,s,l:seen.append(s) or {'id':e['pair_id']})
        bridge.canonical_rows(api,[{'pair_id':'a','generation_seed':71},{'pair_id':'b'}],{'a':{},'b':{}},'test',42,None)
        self.assertEqual(seen,[71,42])
    def test_canonical_pixel_target_mismatch_is_still_fatal(self):
        api=SimpleNamespace(STRAIGHT=('sim_straight_test',),require=c.require,StraightPopulation=type('Base',(),{}),
            digest=lambda rows:'ids' if rows==['p'] else 'wrong',bound_module=lambda *_:SimpleNamespace(load_sample=None),
            canonical_entry=lambda *_:{'input':'changed'})
        api.straight_manifest=lambda *_:({'populations':{'test':{'canonical_row_identity_sha256':'expected'}}},
            {'entries':[{'pair_id':'p','generation_seed':12}],'seed':1},{'records':[{'pair_id':'p'}]}, {})
        plan={'canonical_straight':{},'straight':{'sim_straight_test':{'pair_ids_sha256':'ids'}}}
        with self.assertRaisesRegex(ValueError,'canonical inputs/targets differ'):bridge.load_population(api,'sim_straight_test',plan,None)
    def test_bridge_never_accepts_real_population(self):
        with self.assertRaises(ValueError):bridge.load_population(SimpleNamespace(STRAIGHT=(),require=c.require),'dunhuang_cv',{},None)


class ExportModeRegressionTests(unittest.TestCase):
    def setUp(self):
        import torch
        from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_v2_v1.adapter import fresh_matcher_v2
        import entry
        self.entry=entry
        torch.set_num_threads(1)
        self.model=fresh_matcher_v2(RachelN512Config(canvas_size=32,coarse_size=32,contour_cap=16,
            landmark_count=2,context_layers=2,activation_checkpointing=False)).set_frozen(True)
    def repaired(self):
        with patch.object(self.entry.original,'load_model',return_value=(self.model,None,{'same':'origin'})):
            return self.entry.load_model({'module':'matcher'}, {}, None)[0]
    def test_reproduces_root_flag_and_repairs_actual_native_inference(self):
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.curriculum_training_v1.checkpoint_io import tree_sha
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.curriculum_training_v1.matcher_evaluation import predict_batch
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.matcher import INPUTS
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_matcher import inputs
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
        source=Path(c.api('matcher_v2_v1.adapter').__file__).resolve().parents[4]
        values=dict(zip(INPUTS,inputs()));ids=['p'+str(i) for i in range(len(values['mask_a']))]
        geometry=CompatibilityConfig(.5,.5,.5,.5,1.,15.)
        self.assertTrue(self.model.training);before=tree_sha(self.model.state_dict())
        with self.assertRaisesRegex(ValueError,'frozen eval'):predict_batch(self.model,geometry,source,values,ids)
        model=self.repaired();rows=predict_batch(model,geometry,source,values,ids)
        self.assertEqual(len(rows),len(ids));self.assertEqual(tree_sha(model.state_dict()),before)
        self.assertFalse(any(m.training for m in model.modules()))
    def test_already_eval_is_idempotent(self):
        self.model.eval();self.assertIs(self.repaired(),self.model)
    def test_trainable_parameter_is_not_hidden(self):
        next(self.model.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(ValueError,'fully frozen'):self.repaired()
    def test_training_child_is_not_hidden(self):
        self.model.base.train()
        with self.assertRaisesRegex(ValueError,'fully frozen'):self.repaired()
    def test_unfrozen_export_is_not_hidden(self):
        self.model.set_frozen(False)
        with self.assertRaisesRegex(ValueError,'fully frozen'):self.repaired()
    def test_wrong_adapter_is_not_hidden(self):
        import torch
        self.model=torch.nn.Linear(2,2)
        with self.assertRaisesRegex(ValueError,'fully frozen'):self.repaired()


class PipelineTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(python=Path('/python'),spec=Path('/spec'),preparation=Path('/prep'),
            out=Path('/out'),gpus=[0,5],canonical_straight=Path('/canonical'),case_plan=Path('/cases'),
            repair_preparation=Path('/repair'),adopt_pipeline=None)
    def test_training_is_original_launcher_with_original_arguments(self):
        a=self.args();values=p.train_command(a)
        self.assertEqual(values[:3],['/python','-m',c.PACKAGE+'.matcher_v2_v1.launcher'])
        self.assertEqual(values[-3:],['--gpus','0','5'])
        for forbidden in ('--resume','--lr','--budget','--batch','--order'):self.assertNotIn(forbidden,values)
    def test_evaluation_uses_truthful_new_entry_not_spoofed_module(self):
        a=self.args();a.controller_root=Path('/training');j=FailedJobTests.job
        values=ec.command(a,Path('/population'),Path('/out/job'),j)
        self.assertEqual(values[1],str(c.HERE/'entry.py'));self.assertNotIn('-m',values)
        self.assertIn('--repair-preparation',values)
    def test_adopt_is_explicit_and_training_root_is_original(self):
        a=self.args();a.adopt_pipeline=Path('/old')
        values=p.eval_command(a,Path('/old/training'))
        ec.exact_flag(values,'--controller-root','/old/training')
        ec.exact_flag(values,'--reuse-evaluation','/old/evaluation')
    def test_duplicate_flag_rejected(self):
        with self.assertRaises(ValueError):ec.exact_flag(['--spec','a','--spec','a'],'--spec','a')
    def test_failure_precedes_new_pipeline_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c.save(root/'failure.json',{})
            with self.assertRaisesRegex(ValueError,'failure precedes'):p.verify_complete(root)
    def test_failure_precedes_new_evaluation_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c.save(root/'controller_failure.json',{})
            with self.assertRaisesRegex(ValueError,'failure precedes'):ec.verify_complete(root)
    def test_b3_head_same_selected_matcher_and_original_topology(self):
        text=inspect.getsource(follow.run_head)
        self.assertIn("compile_execution(old.BASE,'B3',module",text)
        self.assertIn('combined_admission=old.COMBINED,selected=selected',text)
        self.assertIn("spec['topology']['microbatch']==32",text)


class OldPipelineIntegrationTests(unittest.TestCase):
    """Synthetic actual launch/return files exercise the complete recovery gate."""
    def make(self,root):
        task=root/'old';old=task/'evaluation';old.mkdir(parents=True);(task/'training').mkdir()
        for name in ('spec','prep','python','canonical','cases'):c.save(root/name,{})
        args=argparse.Namespace(reuse_evaluation=old,controller_root=task/'training',spec=root/'spec',
            preparation=root/'prep',python=root/'python',canonical_straight=root/'canonical',case_plan=root/'cases',gpus=[0,5])
        jobs=[dict(name='sim_best_sim_select',selection='sim_best',split='sim_select'),FailedJobTests.job]
        inputs={'plan':SimpleNamespace(record={'module':'matcher'})};selected={'sim_best':{'checkpoint_sha256':'model'}};plan={'same':'population'}
        c.save(old/'population_plan.json',plan)
        values=['eval-controller','--out',str(old)];train=['original-train','--out',str(task/'training')]
        c.save(task/'pipeline_launch.json',dict(controller=None,commands=[train,values],execution_sha256=c.sha(args.spec)))
        c.save(task/'training_launch.json',dict(phase='training',command=train,process=None))
        c.save(task/'training_return.json',dict(phase='training',returncode=0,launch_sha256=c.sha(task/'training_launch.json')))
        c.save(task/'evaluation_launch.json',dict(phase='evaluation',command=values,process=None))
        c.save(task/'evaluation_return.json',dict(phase='evaluation',returncode=1,launch_sha256=c.sha(task/'evaluation_launch.json')))
        c.save(task/'failure.json',dict(error="ChildFailure('evaluation exited1; preserve output and inspect before any explicit recovery')"))
        c.save(old/'controller_launch.json',dict(controller=None,execution=c.bound(args.spec),selected_models=selected,
            training_controller_root=str(args.controller_root),preparation_sha256=c.sha(args.preparation),
            population_plan_sha256=c.sha(old/'population_plan.json'),jobs=jobs))
        failures=[dict(job=jobs[1]['name'],returncode=1)]
        c.save(old/'controller_failure.json',dict(error=repr(ValueError('Scorer evaluation queue incomplete: '+repr(failures)))))
        def command(python,a,pop,out,j,m):return ['old-evaluator','--split',j['split'],'--out',str(out)]
        for i,j in enumerate(jobs):
            name=j['name'];c.save(old/(name+'_launch.json'),dict(job=j,process=None,command=command(None,None,None,old/name,j,None)))
            c.save(old/(name+'_return.json'),dict(returncode=0 if i==0 else 1,launch_sha256=c.sha(old/(name+'_launch.json'))))
            (old/(name+'.log')).write_text('done' if i==0 else ec.SEED_ERROR+'\n')
        def verify_child(r,phase,expected):
            launch=c.read(r/(phase+'_launch.json'));ret=c.read(r/(phase+'_return.json'))
            c.require(launch['command']==expected and ret['returncode']==0 and ret['launch_sha256']==c.sha(r/(phase+'_launch.json')),'bad train return')
        original=SimpleNamespace(jobs=lambda _:jobs,command=command,verify_job=lambda r,j,m:{'job':j,'audit':'verified'})
        pipe=SimpleNamespace(commands=lambda *_:(train,values),verify_child=verify_child)
        launcher=SimpleNamespace(identity=lambda _:None)
        mapping={'matcher_v2_v1.evaluation_controller':original,'matcher_v2_v1.pipeline':pipe,'curriculum_training_v1.launcher':launcher}
        return args,inputs,selected,plan,mapping,task
    def test_old_success_is_reused_known_failure_is_planned(self):
        with tempfile.TemporaryDirectory() as d:
            a,i,s,p0,modules,task=self.make(Path(d))
            with patch.object(ec,'api',side_effect=lambda n:modules[n]):got=ec.inspect_old(a,i,s,p0)
            self.assertEqual([x['state'] for x in got['ledger']],['reused_success','known_seed_failure'])
            self.assertTrue(got['successful_jobs_not_repeated'])
    def test_old_training_failure_never_adopted(self):
        with tempfile.TemporaryDirectory() as d:
            a,i,s,p0,modules,task=self.make(Path(d));path=task/'training_return.json';row=c.read(path);row['returncode']=1
            path.write_text(__import__('json').dumps(row))
            with patch.object(ec,'api',side_effect=lambda n:modules[n]),self.assertRaisesRegex(ValueError,'bad train return'):
                ec.inspect_old(a,i,s,p0)
    def test_extra_controller_failure_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,i,s,p0,modules,task=self.make(Path(d));path=task/'evaluation/controller_failure.json'
            row={'error':repr(ValueError('Scorer evaluation queue incomplete: '+repr([{'job':'other','error':'audit changed'}])))}
            path.write_text(__import__('json').dumps(row))
            with patch.object(ec,'api',side_effect=lambda n:modules[n]),self.assertRaisesRegex(ValueError,'additional failure'):
                ec.inspect_old(a,i,s,p0)
    def test_model_or_population_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,i,s,p0,modules,task=self.make(Path(d))
            with patch.object(ec,'api',side_effect=lambda n:modules[n]),self.assertRaisesRegex(ValueError,'models/population changed'):
                ec.inspect_old(a,i,{},p0)
    def test_undrained_children_block_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            a,i,s,p0,modules,task=self.make(Path(d));c.save(task/'evaluation/live_children_after_controller_error.json',{})
            with patch.object(ec,'api',side_effect=lambda n:modules[n]),self.assertRaisesRegex(ValueError,'undrained'):
                ec.inspect_old(a,i,s,p0)
    def test_false_old_success_receipt_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            a,i,s,p0,modules,task=self.make(Path(d));c.save(task/'complete.json',{})
            with patch.object(ec,'api',side_effect=lambda n:modules[n]),self.assertRaisesRegex(ValueError,'terminal failed old pipeline'):
                ec.inspect_old(a,i,s,p0)


class WaiterTests(unittest.TestCase):
    def test_old_script_binds_actual_running_waiter_source(self):
        self.assertEqual(follow.OLD_SCRIPT,follow.ROOT/'head_queue_source_01/continue_b3_heads.py')
        self.assertEqual(follow.SCRIPT_SHA,'bc21621f2d5833cee4f7d40a81b7ce5c3d14753e46755f406039a08225b22a2f')
    def dirs(self,tmp):
        root=Path(tmp);old=root/'matcher';old.mkdir();queue=root/'waiter';queue.mkdir()
        return root,old,queue
    def test_waits_without_training_failure_or_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root,old,q=self.dirs(d)
            with patch.multiple(follow,ROOT=root,OLD=old,OLD_QUEUE=q):
                self.assertEqual(follow.pending_state(lambda _:None),'waiting_original_matcher')
    def test_training_failure_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            root,old,q=self.dirs(d);(old/'training').mkdir();c.save(old/'training/controller_failure.json',{})
            with patch.multiple(follow,ROOT=root,OLD=old,OLD_QUEUE=q),self.assertRaisesRegex(ValueError,'training failure'):
                follow.pending_state(lambda _:None)
    def test_healthy_old_completion_never_takes_over_heads(self):
        with tempfile.TemporaryDirectory() as d:
            root,old,q=self.dirs(d);c.save(old/'complete.json',{})
            with patch.multiple(follow,ROOT=root,OLD=old,OLD_QUEUE=q):
                self.assertEqual(follow.pending_state(lambda _:None),'original_completed_no_takeover')
    def test_old_live_owner_must_exit_first(self):
        with tempfile.TemporaryDirectory() as d:
            root,old,q=self.dirs(d);c.save(old/'failure.json',{});process=dict(pid=7,starttime=6,cmdline='owner')
            c.save(q/'launch.json',dict(controller=process))
            with patch.multiple(follow,ROOT=root,OLD=old,OLD_QUEUE=q):
                self.assertEqual(follow.pending_state(lambda _:process),'waiting_old_owners_to_exit')
    def test_vanished_owner_without_return_is_not_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root,old,q=self.dirs(d);c.save(old/'failure.json',{})
            with patch.multiple(follow,ROOT=root,OLD=old,OLD_QUEUE=q),self.assertRaisesRegex(ValueError,'terminal failure receipt'):
                follow.pending_state(lambda _:None)
    def test_wrong_waiter_failure_not_handled(self):
        with tempfile.TemporaryDirectory() as d:
            root,old,q=self.dirs(d);c.save(old/'failure.json',{});c.save(q/'failure.json',dict(error='OOM'))
            with patch.multiple(follow,ROOT=root,OLD=old,OLD_QUEUE=q),self.assertRaisesRegex(ValueError,'another reason'):
                follow.pending_state(lambda _:None)
    def test_started_old_heads_block_takeover(self):
        with tempfile.TemporaryDirectory() as d:
            root,old,q=self.dirs(d);(q/'scorer_patch').mkdir()
            with patch.multiple(follow,ROOT=root,OLD=old,OLD_QUEUE=q),self.assertRaisesRegex(ValueError,'already started'):
                follow.require_heads_unstarted()
    def test_armed_waiter_only_hands_off_to_full_verifier(self):
        with tempfile.TemporaryDirectory() as d:
            root,old,q=self.dirs(d);c.save(old/'failure.json',{})
            c.save(q/'failure.json',dict(error="ValueError('Matcher failure overrides completion; no automatic retry')"))
            with patch.multiple(follow,ROOT=root,OLD=old,OLD_QUEUE=q):
                self.assertEqual(follow.pending_state(lambda _:None),'ready_for_exact_failure_verification')


if __name__=='__main__':unittest.main()
