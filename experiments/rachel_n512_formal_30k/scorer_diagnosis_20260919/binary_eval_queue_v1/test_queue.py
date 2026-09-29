from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import queue_contracts as contracts
import launch_evaluation as launch

VARIANT=os.environ.get('BINARY_VERIFY_VARIANT','patch')

class QueueTests(unittest.TestCase):
    def fixture(self,root,split,choice='real'):
        n=contracts.EXPECTED[split];ids=['p'+str(i) for i in range(n)]
        selected=dict(checkpoint_sha256='synthetic',selected_epoch=4,thresholds={s:.3 for s in contracts.EXPECTED})
        identity=dict(variant='binary_'+VARIANT,selection_kind=choice,split=split,total_pairs=n,
            checkpoint_sha256='synthetic',selected_epoch=4,threshold=.3)
        predictions=root/'pair_predictions.jsonl'
        predictions.write_text(''.join(json.dumps(dict(pair_id=i,score=.5))+'\n' for i in ids))
        roles={name:{'pair_ids':ids[start:stop]} for name,start,stop in
               [('real_cal',0,100),('real_select',100,400),('real_test',400,n-(3 if split=='dunhuang_cv' else 0))]}
        spec=dict(excluded_gt_pair_ids=ids[-3:] if split=='dunhuang_cv' else [],roles=roles)
        plan={'datasets':{split:spec}}
        groups={'all':{'primary':{'pairs':n}}} if split=='sim_test_v14' else {
            name:{'primary':dict(pairs=len(role['pair_ids']),layout20=None,joint_f1=None,joint_fp=None,candidate_coverage=None)}
            for name,role in roles.items()}
        if split=='dunhuang_cv':groups['gt_corrected_800_development_context']={'primary':{'pairs':800}}
        count={'sim_test_v14':0,'dunhuang_cv':10,'turufan':1}[split]
        cases=[];caseplan={'cases':[dict(pair_id=i,split=split) for i in ids[:count]]}
        for i in range(count):
            d=root/('case'+str(i));d.mkdir();(d/'arrays.npz').write_bytes(b'synthetic archive; numeric callback is mocked')
            sh=contracts.sha(d/'arrays.npz')
            contracts.save(d/'evidence.json',dict(pair_id=ids[i],variant=VARIANT,provenance={'checkpoint_sha256':'synthetic'},
                sidecar={'path':'arrays.npz','sha256':sh}))
            contracts.save(d/'audit.json',dict(status='passed'))
            cases.append(dict(pair_id=ids[i],evidence=d.name+'/evidence.json',sidecar_sha256=sh,
                numerical_audit=d.name+'/audit.json',numerical_audit_status='passed'))
        contracts.save(root/'diagnostic_index.json',dict(cases=cases,selected_by_new_results=False))
        contracts.save(root/'status.json',dict(status='complete',pairs=n))
        contracts.save(root/'protocol.json',dict(status='complete',**identity))
        contracts.save(root/'prediction_complete.json',dict(status='all_predictions_frozen',pairs=n,model_state_unchanged=True,
            sha256=contracts.sha(predictions),**identity))
        summary=dict(status='complete',groups=groups,diagnostic_cases=cases,main_group='all' if split=='sim_test_v14' else 'real_test',
            real_test_is_historically_unseen=False,layout_gt_available=split!='turufan',**identity)
        contracts.save(root/'summary.json',summary)
        return selected,plan,caseplan

    def verify(self,root,split,choice='real',audit=None):
        selected,roles,cases=self.fixture(root,split,choice)
        return lambda:contracts.verify_result(root,VARIANT,choice,split,selected,roles,cases,
            audit or (lambda _:dict(status='passed',errors=[])))

    def test_exact_six_distinct_tasks_and_binary_command(self):
        self.assertEqual(len(contracts.TASKS),6);self.assertEqual(len(set(contracts.TASKS)),6)
        for choice,split in contracts.TASKS:
            cmd=contracts.evaluator_command('python','root','prepared','output',VARIANT,choice,split)
            self.assertEqual(cmd[cmd.index('--variant')+1],VARIANT);self.assertEqual(cmd[cmd.index('--selection')+1],choice)
            self.assertNotIn('--resume',cmd);self.assertTrue(cmd[1].endswith('binary_eval_v1/entry.py'))
        for v,c,s in [('m12','real','turufan'),(VARIANT,'best','turufan'),(VARIANT,'real','train')]:
            with self.assertRaises(ValueError):contracts.evaluator_command('p','r','p','o',v,c,s)

    def test_all_six_frozen_population_contracts(self):
        for choice,split in contracts.TASKS:
            with tempfile.TemporaryDirectory() as t:
                result=self.verify(Path(t),split,choice)()
                self.assertEqual(result['pairs'],contracts.EXPECTED[split]);self.assertEqual(result['status'],'passed')

    def test_wrong_model_epoch_threshold_or_split_rejected(self):
        for field,value in [('checkpoint_sha256','other'),('selected_epoch',6),('threshold',.4),('split','dunhuang_cv')]:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);run=self.verify(root,'turufan');p=root/'summary.json';r=contracts.read(p);r[field]=value;contracts.save(p,r)
                with self.assertRaises(ValueError):run()

    def test_failure_missing_rows_duplicate_or_target_leak_rejected(self):
        for case in ('failure','short','duplicate','target'):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);run=self.verify(root,'sim_test_v14');p=root/'pair_predictions.jsonl'
                if case=='failure':contracts.save(root/'failure.json',{'error':'synthetic'})
                else:
                    rows=p.read_text().splitlines()
                    if case=='short':rows.pop()
                    elif case=='duplicate':rows[-1]=rows[0]
                    else:r=json.loads(rows[0]);r['label']=True;rows[0]=json.dumps(r)
                    p.write_text('\n'.join(rows)+'\n');f=contracts.read(root/'prediction_complete.json');f['sha256']=contracts.sha(p);contracts.save(root/'prediction_complete.json',f)
                with self.assertRaises(ValueError):run()

    def test_turufan_never_invents_layout_metrics(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run=self.verify(root,'turufan');s=contracts.read(root/'summary.json')
            s['groups']['real_test']['primary']['joint_f1']=.7;contracts.save(root/'summary.json',s)
            with self.assertRaisesRegex(ValueError,'invented'):run()

    def test_fixed_cases_need_original_ids_archives_and_numeric_replay(self):
        for case in ('id','hash','numeric','missing'):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);run=self.verify(root,'dunhuang_cv',audit=(lambda _:dict(status='failed',errors=['actual mismatch'])) if case=='numeric' else None)
                if case=='id':
                    p=root/'case0/evidence.json';r=contracts.read(p);r['pair_id']='other';contracts.save(p,r)
                elif case=='hash':(root/'case0/arrays.npz').write_bytes(b'changed')
                elif case=='missing':contracts.save(root/'diagnostic_index.json',dict(cases=[],selected_by_new_results=False))
                with self.assertRaises(ValueError):run()

    def test_device_gate_cannot_be_satisfied_by_cpu_or_other_variant(self):
        g=dict(schema='binary-trace-device-gate/1',status='passed',variant=VARIANT,device='cuda:0',
            parameters_unchanged=True,rng_unchanged=True,capture_bitwise_equal=True,numeric_replay_passed=True,
            head_parameters=34529 if VARIANT=='patch' else 3201,optimizer_updates=0,trained_checkpoint_opened=False,
            real_inference_performed=False,source_bindings={'source':'same'})
        contracts.validate_gpu_gate(g,VARIANT,g['source_bindings'])
        for key,value in [('device','cpu'),('parameters_unchanged',False),('capture_bitwise_equal',False),
                          ('head_parameters',0),('optimizer_updates',1),('variant','wrong')]:
            bad=deepcopy(g);bad[key]=value
            with self.assertRaises(ValueError):contracts.validate_gpu_gate(bad,VARIANT,g['source_bindings'])

    def test_gpu_checks_are_read_only_and_refuse_occupied_card(self):
        for s in ('0','0,0','0,1,2','-1,0'):
            with self.assertRaises(ValueError):launch.gpu_indices(s)
        self.assertEqual(launch.gpu_indices('4,5'),(4,5))
        with patch.object(launch.subprocess,'check_output',return_value='123, GPU-5\n'):
            with self.assertRaises(ValueError):launch.assert_free((4,5),{4:{'uuid':'GPU-4'},5:{'uuid':'GPU-5'}})
        self.assertEqual(launch.environment()['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(launch.environment(5)['CUDA_VISIBLE_DEVICES'],'5')

    def test_archive_path_cannot_escape_job(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):contracts.safe_child(Path(t),'../other')

if __name__=='__main__':unittest.main()
