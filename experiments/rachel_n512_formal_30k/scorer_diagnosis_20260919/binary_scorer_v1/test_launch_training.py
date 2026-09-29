"""External controller tests; never launches a process, GPU job, or training."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('binary_external_controller',Path(__file__).with_name('launch_training.py'))
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)

class LauncherTests(unittest.TestCase):
    def test_new_order_releases_one_verified_lane_not_all_old_experiments(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p,r,expected=self.queue(root)
            auth=root/'authorization.json'
            controller.save(auth,dict(status='user_authorized',order=['binary_patch','binary_stats','aggressive_scratch','joint_e32'],keep_existing_training_unchanged=True))
            r.update(schema='binary-released-lane/2',authorization_sha256=controller.digest(auth));controller.save(p,r)
            with patch.object(controller,'SCHEDULE_AUTHORIZATION',auth),patch.object(controller,'PARALLEL_PREDECESSORS',expected):
                self.assertEqual(controller.validate_queue_release(p),r)
                controller.save(auth,dict(status='user_authorized',order=['joint_e32']))
                with self.assertRaises(ValueError):controller.validate_queue_release(p)
    def receipt(self):
        return dict(schema='binary-evaluation-preparation/1',status='cpu_preparation_passed',errors=0,failures=0,skipped=0,tests=2,
            verified_variants=['patch','stats'],source_files_unchanged=True,real_inference_performed=False,
            real_checkpoints_opened=False,gpu_preflight=False,formal_training_started=False,
            adapter_python_sha256={'entry.py':'a'},common_python_sha256={'frozen.py':'b'},training_source_sha256={'train.py':'c'},
            results=[dict(variant=v,status='passed',returncode=0,tests=1,errors=0,failures=0,skipped=0) for v in ('patch','stats')])

    def queue(self,root,count=3):
        formal=root/'formal';stage=formal/'scorer';stage.mkdir(parents=True)
        (stage/'best_joint.pt').write_bytes(b'SIM checkpoint fixture')
        (stage/'best_real.pt').write_bytes(b'REAL checkpoint fixture')
        selected=dict(status='selected',binding={'source':'fixture'},best={'epoch':2},best_real={'epoch':4},
            best_joint_sha256=controller.digest(stage/'best_joint.pt'),best_real_sha256=controller.digest(stage/'best_real.pt'))
        complete=dict(selected,status='stage_complete')
        controller.save(stage/'selection.json',selected);controller.save(stage/'complete.json',complete)
        controller.save(formal/'training_complete.json',dict(status='training_complete',binding=selected['binding'],last_stage=complete))
        jobs=[]
        for choice in ('sim','real') if count==6 else ('sim',):
            for split in ('sim_test_v14','dunhuang_cv','turufan'):
                directory=root/(choice+'_'+split);directory.mkdir()
                predictions=directory/'pair_predictions.jsonl';predictions.write_text('{"pair_id":"synthetic"}\n')
                cp=stage/('best_joint.pt' if choice=='sim' else 'best_real.pt');epoch=2 if choice=='sim' else 4
                identity=dict(checkpoint=str(cp),checkpoint_sha256=controller.digest(cp),selected_epoch=epoch,split=split,selection_kind=choice)
                controller.save(directory/'protocol.json',dict(status='complete',**identity))
                controller.save(directory/'summary.json',dict(status='complete',**identity))
                controller.save(directory/'status.json',{'status':'complete'})
                controller.save(directory/'prediction_complete.json',dict(status='all_predictions_frozen',model_state_unchanged=True,
                    sha256=controller.digest(predictions),**identity))
                jobs.append(dict(root=str(directory),returncode=0))
        release=dict(schema='binary-prior-queue-release/1',status='complete',branches=[dict(formal_root=str(formal),evaluations=jobs)])
        path=root/'release.json';controller.save(path,release)
        return path,release,{str(formal):count}

    def test_cpu_preparation_requires_both_actual_runs(self):
        r=self.receipt();controller.validate_evaluation_preparation(r)
        for k,v in [('tests',0),('skipped',1),('source_files_unchanged',False),('verified_variants',['patch']),
                    ('real_checkpoints_opened',True),('gpu_preflight',True),('formal_training_started',True),('results',[])]:
            bad=deepcopy(r);bad[k]=v
            with self.assertRaises(ValueError):controller.validate_evaluation_preparation(bad)

    def test_complete_three_and_six_job_queue(self):
        for n in (3,6):
            with tempfile.TemporaryDirectory() as t:
                p,r,expected=self.queue(Path(t),n);self.assertEqual(controller.validate_queue_release(p,expected),r)
                # A synthetic single branch can never release the actual four-branch queue.
                with self.assertRaises(ValueError):controller.validate_queue_release(p)

    def test_other_model_or_epoch_cannot_release_queue(self):
        for target in ('protocol','summary','prediction_complete'):
            for field,value in [('checkpoint_sha256','other'),('selected_epoch',14),('split','wrong')]:
                with tempfile.TemporaryDirectory() as t:
                    p,r,expected=self.queue(Path(t));d=Path(r['branches'][0]['evaluations'][0]['root'])
                    f=d/(target+'.json');record=json.loads(f.read_text());record[field]=value;controller.save(f,record)
                    with self.assertRaises(ValueError):controller.validate_queue_release(p,expected)

    def test_failure_or_modified_predictions_block_release(self):
        for case in ('failure','predictions','nonzero','duplicate','stale_complete'):
            with tempfile.TemporaryDirectory() as t:
                p,r,expected=self.queue(Path(t));row=r['branches'][0];d=Path(row['evaluations'][0]['root'])
                if case=='failure':controller.save(Path(row['formal_root'])/'failure.json',{'error':'fixture'})
                elif case=='predictions':(d/'pair_predictions.jsonl').write_text('changed')
                elif case=='nonzero':row['evaluations'][0]['returncode']=1;controller.save(p,r)
                elif case=='duplicate':row['evaluations'][1]=row['evaluations'][0];controller.save(p,r)
                else:controller.save(Path(row['formal_root'])/'scorer/complete.json',{'status':'stage_complete'})
                with self.assertRaises(ValueError):controller.validate_queue_release(p,expected)

    def test_gpu_inventory_never_interrupts_occupied_cards(self):
        def output(command,**kwargs):
            return '0, GPU-0, synthetic\n1, GPU-1, synthetic\n' if '--query-gpu=index,uuid,name' in command else 'GPU-1, 123\n'
        with patch.object(controller.subprocess,'check_output',side_effect=output):
            for devices in ('0,1','0,0','0','0,6'):
                with self.assertRaises(ValueError):controller.gpu_check(devices)
        with patch.object(controller.subprocess,'check_output',side_effect=['0, GPU-0, synthetic\n1, GPU-1, synthetic\n','']):
            self.assertEqual(set(controller.gpu_check('0,1')),{0,1})

    def test_gate_binds_variant_and_full_matcher_freezing(self):
        binding={'config':{'scorer_variant':'patch'}}
        r=dict(status='passed',formal_training=False,updated_weights_discarded=True,arm='scratch_fixed',stage='scorer',
            updates=12,exposures=384,world_size=2,microbatch=8,accumulate=2,effective_batch=32,
            matcher_unchanged=True,model_state_hashes=['same','same'],binding=binding,resume_matches_uninterrupted=True)
        controller.validate_gate(r,binding,'scratch_fixed',replay=True)
        for k,v in [('updates',0),('matcher_unchanged',False),('model_state_hashes',['a','b']),
                    ('binding',{'config':{'scorer_variant':'stats'}}),('resume_matches_uninterrupted',False)]:
            bad=deepcopy(r);bad[k]=v
            with self.assertRaises(ValueError):controller.validate_gate(bad,binding,'scratch_fixed',replay=True)

    def test_frozen_real_reselection_is_separate_and_required(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);_,release,_=self.queue(root)
            formal=Path(release['branches'][0]['formal_root']);stage=formal/'scorer'
            selection=json.loads((stage/'selection.json').read_text());selection['actual_epochs']=16
            controller.save(stage/'selection.json',selection)
            curve=[]
            for epoch in range(0,18,2):
                checkpoint=stage/f'epoch_{epoch:03d}_weights.pt';checkpoint.write_bytes(('epoch'+str(epoch)).encode())
                curve.append(dict(epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=controller.digest(checkpoint),
                    synthetic_cal_threshold=.3,model_state_unchanged=True,
                    real_report={'key':[.99 if epoch==0 else .8,.9,.95],'thresholds':{'dunhuang_cv':.4,'turufan':.5}}))
            reselect=root/'reselection';reselect.mkdir();controller.save(reselect/'real_curve.json',curve)
            record=dict(schema='frozen-e32-real-reselection/1',status='complete',test_used=False,real_used=True,
                gradients_used=False,optimizer_updates=0,original_training_outputs_unchanged=True,
                original_selection_sha256=controller.digest(stage/'selection.json'),
                terminal_receipt_sha256=controller.digest(formal/'training_complete.json'),
                curve_sha256=controller.digest(reselect/'real_curve.json'),best=curve[1])
            path=reselect/'real_selection.json';controller.save(path,record)
            spec=dict(real_selection=str(path),returncode=0,evaluations=release['branches'][0]['evaluations'])
            for job in spec['evaluations']:
                d=Path(job['root']);split=json.loads((d/'protocol.json').read_text())['split']
                threshold=.3 if split=='sim_test_v14' else record['best']['real_report']['thresholds'][split]
                for name in ('protocol','summary','prediction_complete'):
                    p=d/(name+'.json');r=json.loads(p.read_text());r.update(checkpoint=curve[1]['checkpoint'],
                        checkpoint_sha256=curve[1]['checkpoint_sha256'],selected_epoch=2,selection_kind='real',threshold=threshold,
                        real_selection_sha256=controller.digest(path));controller.save(p,r)
            before=controller.digest(stage/'selection.json')
            controller.validate_frozen_real_release(spec,formal)
            self.assertEqual(controller.digest(stage/'selection.json'),before)
            for key,value in [('optimizer_updates',1),('test_used',True),('best',curve[0])]:
                bad=deepcopy(record);bad[key]=value;controller.save(path,bad)
                with self.assertRaises(ValueError):controller.validate_frozen_real_release(spec,formal)
            controller.save(path,record);bad=deepcopy(spec);bad['evaluations']=bad['evaluations'][:2]
            with self.assertRaises(ValueError):controller.validate_frozen_real_release(bad,formal)

if __name__=='__main__':unittest.main()
