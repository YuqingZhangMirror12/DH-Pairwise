from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import contracts as c
import run_queued as run
import worker

def model_proof(choice='sim'):
    return dict(model_state_sha256='full-tensor-digest',training_implementation_sha256={'model.py':'a'},
        data_contract_sha256='data',geometry_calibration_sha256='geometry',checkpoint_sha256=choice,
        selected_epoch=8,selection_kind=choice,variant='threshold_joint',thresholds={s:.3 for s in c.SPLITS})

class QueueTests(unittest.TestCase):
    def fixture(self,root,kind,choice,split):
        n=c.SPLITS[split];ids=['p'+str(i) for i in range(n)];proof=model_proof(choice)
        variant='threshold_joint' if kind=='joint' else 'threshold';proof['variant']=variant
        identity=dict(variant=variant,selection_kind=choice,split=split,total_pairs=n,
            checkpoint_sha256=choice,selected_epoch=8,threshold=.3)
        (root/'pair_predictions.jsonl').write_text(''.join(json.dumps(dict(pair_id=i,score=.5,
            has_candidate=True,numeric_valid=True,accepted=True))+'\n' for i in ids))
        excluded=ids[-3:] if split=='dunhuang_cv' else []
        roles={k:{'pair_ids':ids[a:b]} for k,a,b in [('real_cal',0,100),('real_select',100,400),('real_test',400,n-len(excluded))]}
        plan={'datasets':{split:dict(excluded_gt_pair_ids=excluded,roles=roles)}}
        groups={'all':{'primary':{'pairs':n}}} if split=='sim_test_v14' else {
            k:{'primary':dict(pairs=len(v['pair_ids']),layout20=None,joint_f1=None,joint_fp=None,candidate_coverage=None)} for k,v in roles.items()}
        if split=='dunhuang_cv':groups['gt_corrected_800_development_context']={'primary':{'pairs':800}}
        count={'sim_test_v14':0,'dunhuang_cv':10,'turufan':1}[split]
        cases=[];caseplan={'cases':[dict(pair_id=i,split=split) for i in ids[:count]]}
        for i in range(count):
            d=root/('case'+str(i));d.mkdir();(d/'a.npz').write_bytes(b'synthetic fixture, numeric audit mocked')
            digest=c.sha(d/'a.npz')
            c.save(d/'evidence.json',dict(pair_id=ids[i],provenance={'checkpoint_sha256':choice},
                semantics=dict(variant='threshold',evidence_mode='exact_union_q'),sidecar=dict(path='a.npz',sha256=digest)))
            c.save(d/'audit.json',dict(status='passed'))
            cases.append(dict(pair_id=ids[i],evidence=d.name+'/evidence.json',sidecar_sha256=digest,
                numerical_audit=d.name+'/audit.json',numerical_audit_status='passed'))
        c.save(root/'status.json',dict(status='complete',pairs=n))
        c.save(root/'protocol.json',dict(status='complete',**identity))
        c.save(root/'prediction_complete.json',dict(status='all_predictions_frozen',pairs=n,
            model_state_unchanged=True,sha256=c.sha(root/'pair_predictions.jsonl'),**identity))
        c.save(root/'diagnostic_index.json',dict(cases=cases,selected_by_new_results=False))
        c.save(root/'summary.json',dict(status='complete',groups=groups,diagnostic_cases=cases,
            main_group='all' if split=='sim_test_v14' else 'real_test',real_test_is_historically_unseen=False,
            layout_gt_available=split!='turufan',**identity))
        return proof,plan,caseplan

    def runner(self,root,kind='joint',choice='real',split='turufan',audit=None):
        proof,plan,cases=self.fixture(root,kind,choice,split)
        return lambda:c.verify_result(root,kind,choice,split,proof,plan,cases,audit or (lambda p:{'status':'passed'}))

    def test_nine_finals_have_no_frozen_sim_repeat(self):
        self.assertEqual(len(c.TASKS),9);self.assertEqual(len(set(c.TASKS)),9)
        for k,ch,s in c.TASKS:
            cmd=c.evaluator_command('python',k,ch,s,'out')
            self.assertIn('evaluate',cmd);self.assertNotIn('--resume',cmd)
            self.assertEqual('--real-selection' in cmd,k=='frozen')
        with self.assertRaises(ValueError):c.evaluator_command('p','frozen','sim','turufan','out')

    def test_all_nine_output_contracts(self):
        for kind,choice,split in c.TASKS:
            with tempfile.TemporaryDirectory() as t:
                proof=self.runner(Path(t),kind,choice,split)()
                self.assertEqual(proof['status'],'passed');self.assertEqual(proof['pairs'],c.SPLITS[split])

    def test_wrong_weights_epoch_threshold_rejected(self):
        for key,value in [('selected_epoch',12),('threshold',.7),('checkpoint_sha256','other'),('variant','simple')]:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);check=self.runner(root);s=c.read(root/'summary.json');s[key]=value;c.save(root/'summary.json',s)
                with self.assertRaises(ValueError):check()

    def test_failures_duplicates_short_rows_or_targets_rejected(self):
        for case in ('failure','duplicate','short','target','acceptance'):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);check=self.runner(root)
                if case=='failure':c.save(root/'failure.json',{'status':'failed'})
                else:
                    p=root/'pair_predictions.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
                    if case=='duplicate':rows[-1]=rows[0]
                    if case=='short':rows.pop()
                    if case=='target':rows[0]['label']=True
                    if case=='acceptance':rows[0]['accepted']=False
                    p.write_text(''.join(json.dumps(r)+'\n' for r in rows));r=c.read(root/'prediction_complete.json');r['sha256']=c.sha(p);c.save(root/'prediction_complete.json',r)
                with self.assertRaises(ValueError):check()

    def test_real_roles_and_turufan_nulls(self):
        for case in ('role','blind','layout'):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);check=self.runner(root);s=c.read(root/'summary.json')
                if case=='role':s['groups']['real_test']['primary']['pairs']=0
                if case=='blind':s['real_test_is_historically_unseen']=True
                if case=='layout':s['groups']['real_select']['primary']['joint_f1']=.9
                c.save(root/'summary.json',s)
                with self.assertRaises(ValueError):check()

    def test_fixed_case_actual_hash_and_numeric_audit(self):
        for case in ('identity','archive','numeric','missing'):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);check=self.runner(root,audit=(lambda p:{'status':'failed'}) if case=='numeric' else None)
                if case=='identity':
                    p=root/'case0/evidence.json';r=c.read(p);r['provenance']['checkpoint_sha256']='other';c.save(p,r)
                if case=='archive':(root/'case0/a.npz').write_bytes(b'changed')
                if case=='missing':c.save(root/'diagnostic_index.json',dict(cases=[],selected_by_new_results=False))
                with self.assertRaises(ValueError):check()

    def test_reuse_requires_full_model_source_and_geometry_not_only_epoch(self):
        a=model_proof();b=model_proof('real');self.assertTrue(c.same_model(a,b))
        for key in ('model_state_sha256','training_implementation_sha256','data_contract_sha256','geometry_calibration_sha256'):
            bad=deepcopy(b);bad[key]='changed';self.assertFalse(c.same_model(a,bad))
            bad=deepcopy(b);bad.pop(key);self.assertFalse(c.same_model(a,bad))
        self.assertFalse(c.same_model({'selected_epoch':8},{'selected_epoch':8}))

    def test_real_reuse_waits_for_sim_completion_and_control_for_reselection(self):
        pending=[('frozen','real','turufan'),('joint','real','turufan'),('joint','sim','turufan')]
        self.assertEqual(run.ready_task(pending,set(),False,True),pending[2])
        self.assertEqual(run.ready_task(pending,{pending[2]},False,True),pending[1])
        self.assertEqual(run.ready_task(pending,set(),True,True),pending[0])
        self.assertEqual(run.ready_task(pending,set(),False,False),pending[1])

    def test_trace_requires_cuda_and_unchanged_actual_attention(self):
        g=dict(status='passed',kind='joint',device='cuda:0',parameters_unchanged=True,rng_unchanged=True,
            trained_checkpoint_or_dataset_opened=False,training_started=False,
            protocol=dict(variant='threshold',evidence_mode='exact_union_q'),
            synthetic_model_cases=[dict(bitwise_outputs_equal=True,audit='passed')])
        c.validate_trace(g,'joint')
        for key,val in [('kind','frozen'),('device','cpu'),('rng_unchanged',False),('training_started',True),('synthetic_model_cases',[])]:
            bad=deepcopy(g);bad[key]=val
            with self.assertRaises(ValueError):c.validate_trace(bad,'joint')

    def priority_fixture(self,root):
        for i,name in enumerate(('binary_patch','binary_stats','aggressive_scratch')):
            work=root/'work'/name;work.mkdir(parents=True)
            c.save(root/'launches'/((str(i)+'_' if name.startswith('binary_') else '')+name+'.json'),
                dict(operation=name,work=str(work),identity={'pid':i}))
    def test_priority_uses_actual_launches_not_stale_aggregate(self):
        with tempfile.TemporaryDirectory() as t,patch.object(run,'QUEUE',Path(t)):
            self.priority_fixture(Path(t));run.check_priority(SimpleNamespace(live=lambda _:True))
    def test_priority_missing_failed_or_dead_blocks_joint(self):
        for case in ('missing','failed','dead'):
            with tempfile.TemporaryDirectory() as t,patch.object(run,'QUEUE',Path(t)):
                root=Path(t);self.priority_fixture(root)
                if case=='missing':(root/'launches/aggressive_scratch.json').unlink()
                if case=='failed':c.save(root/'work/aggressive_scratch/failure.json',{'status':'failed'})
                with self.assertRaises(ValueError):run.check_priority(SimpleNamespace(live=lambda _:case!='dead'))
    def test_priority_completed_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as t,patch.object(run,'QUEUE',Path(t)):
            root=Path(t);self.priority_fixture(root)
            for task in ('binary_patch','binary_stats','aggressive_scratch'):c.save(root/'work'/task/'complete.json',dict(status='complete'))
            run.check_priority(SimpleNamespace(live=lambda _:False))

    def test_reuse_changes_only_threshold_metadata_preserves_scores_and_arrays(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'source';source.mkdir();out=root/'reused';row=dict(pair_id='a',score=.4,has_candidate=True,numeric_valid=True,accepted=True)
            for name in ('pair_predictions.jsonl','case_diagnostics.jsonl'):(source/name).write_text(json.dumps(row)+'\n')
            a=model_proof();b=model_proof('real');b['thresholds']['turufan']=.5
            c.save(source/'status.json',dict(status='complete'))
            c.save(source/'prediction_complete.json',dict(status='all_predictions_frozen',model_state_unchanged=True,
                checkpoint_sha256='sim',split='turufan',sha256=c.sha(source/'pair_predictions.jsonl')))
            c.save(source/'protocol.json',dict(status='complete',split='turufan',total_pairs=1,threshold=.3))
            d=source/'case';d.mkdir();(d/'arrays.npz').write_bytes(b'unchanged exact network arrays')
            c.save(d/'evidence.json',dict(row,threshold=.3,provenance={},sidecar={'path':'arrays.npz','sha256':c.sha(d/'arrays.npz')}))
            c.save(d/'audit.json',dict(status='passed'))
            c.save(source/'diagnostic_index.json',dict(cases=[dict(evidence='case/evidence.json',numerical_audit='case/audit.json')]))
            c.save(root/'real_split.json',{})
            ev=SimpleNamespace(snapshot_provenance=lambda p:p,make_summary=lambda rows,split,roles,p:dict(status='complete',**p))
            before={str(p):c.sha(p) for p in source.rglob('*') if p.is_file()}
            with patch.object(worker,'PREPARED',root):worker.reuse(source,out,a,b,'turufan',ev,lambda p:{'status':'passed'})
            reused=json.loads((out/'pair_predictions.jsonl').read_text())
            self.assertFalse(reused['accepted']);self.assertEqual(reused['score'],.4)
            self.assertEqual(c.sha(out/'case/arrays.npz'),c.sha(source/'case/arrays.npz'))
            self.assertEqual(c.read(out/'prediction_reuse.json')['network_forward_calls'],0)
            self.assertEqual(before,{str(p):c.sha(p) for p in source.rglob('*') if p.is_file()})
            with self.assertRaises(ValueError):worker.reuse(source,out,a,b,'turufan',ev,lambda p:{'status':'passed'})

    def test_reuse_refuses_different_weights_before_writing(self):
        a=model_proof();b=model_proof('real');b['model_state_sha256']='other'
        with self.assertRaisesRegex(ValueError,'different model'):worker.reuse('absent','absent',a,b,'turufan',None,None)

    def simulate_driver(self,root,fail=False,reuse=True):
        launches=[];joint={'status':'passed','selected':{'sim':model_proof(),'real':model_proof('real')}}
        if not reuse:joint['selected']['real']['model_state_sha256']='different'
        def cpu(common,op,kind,out,log,extra=()):
            result=joint if op=='selected' and kind=='joint' else dict(status='passed',selected={'real':model_proof('real')})
            c.save(out,result);return result
        def trace(cmd,log,env):
            kind=cmd[cmd.index('--kind')+1];out=cmd[cmd.index('--out')+1]
            c.save(out,dict(status='passed',kind=kind,device='cuda:0',parameters_unchanged=True,rng_unchanged=True,
                trained_checkpoint_or_dataset_opened=False,training_started=False,
                protocol=dict(variant='threshold',evidence_mode='exact_union_q'),synthetic_model_cases=[dict(bitwise_outputs_equal=True,audit='passed')]))
        def popen(cmd,**kwargs):
            launches.append(cmd);code=1 if fail and 'evaluate' in cmd else 0
            return SimpleNamespace(pid=len(launches),poll=lambda:code)
        common=SimpleNamespace(PYTHON='python',env=lambda gpu='':{},free=lambda _:True,identity=lambda pid:{'pid':pid},launch_wait=trace)
        with patch.object(run,'cpu_worker',side_effect=cpu),patch.object(run,'verify_frozen_sim'),patch.object(run.subprocess,'Popen',side_effect=popen),patch.object(run.time,'sleep'),patch.object(run,'sha',return_value='fixture'):
            if fail:
                with self.assertRaises(RuntimeError):run.evaluate_all(common,'4,5',root/'out')
                self.assertFalse((root/'out/evaluation_complete.json').exists())
            else:
                result=run.evaluate_all(common,'4,5',root/'out')
                self.assertEqual(result['final_evaluations'],9);self.assertEqual(len(result['jobs']),10)
                self.assertTrue(all(r['status']=='complete' and r['returncode']==0 for r in result['jobs']))
        return launches
    def test_two_slot_lifecycle_completes_reselection_plus_nine(self):
        with tempfile.TemporaryDirectory() as t:
            cmds=self.simulate_driver(Path(t));self.assertEqual(sum('reuse' in x for x in cmds),3)
            self.assertEqual(sum('reselect' in x for x in cmds),1)
    def test_different_joint_choices_run_two_frozen_models(self):
        with tempfile.TemporaryDirectory() as t:
            cmds=self.simulate_driver(Path(t),reuse=False);self.assertEqual(sum('reuse' in x for x in cmds),0)
            self.assertEqual(sum('evaluate' in x for x in cmds),9)
    def test_failed_worker_does_not_retry_or_publish_complete(self):
        with tempfile.TemporaryDirectory() as t:
            cmds=self.simulate_driver(Path(t),fail=True);self.assertLess(len(cmds),10)
    def test_diagnostic_paths_stay_inside_job(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):c.safe_child(Path(t),'../other')

if __name__=='__main__':unittest.main()
