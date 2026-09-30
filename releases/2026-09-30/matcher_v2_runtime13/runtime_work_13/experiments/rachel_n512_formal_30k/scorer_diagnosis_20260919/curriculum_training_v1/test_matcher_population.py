"""Synthetic population/IO/controller tests, plus actual small-canvas inference."""
import copy
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from time import sleep as actual_sleep
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .checkpoint_io import file_sha,tree_sha,write_json
from .exposure import digest
from .matcher_diagnostics import inspect_pair
from .matcher_evaluation import prediction_sha
from . import matcher_population as pop
from . import matcher_run as run
from . import matcher_eval_controller as controller
from .matcher_entry import check_preparation
from . import test_matcher_evaluation as native_fixture

BASE='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'


class PopulationContractTests(unittest.TestCase):
    def roles(self):
        return dict(datasets={ds:dict(excluded_gt_pair_ids=['bad'] if ds=='dunhuang_cv' else [],
            roles={r:dict(pair_ids=[r]) for r in ('real_cal','real_select','real_test')})
            for ds in ('dunhuang_cv','turufan')})

    def test_real_source_roles_and_original_vs_corrected_populations(self):
        rows=[dict(pair_id=i) for i in ('bad','real_cal','real_select','real_test')]
        groups=pop.population_groups(rows,'dunhuang_cv',self.roles())
        self.assertEqual(len(groups['all_development_context']),4)
        self.assertEqual(len(groups['gt_corrected_800_development_context']),3)
        self.assertEqual(groups['real_test'],[rows[-1]])

    def test_overlapping_real_roles_or_missing_predictions_rejected(self):
        rows=[dict(pair_id=i) for i in ('bad','real_cal','real_select','real_test')];roles=self.roles()
        with self.assertRaises(ValueError):pop.population_groups(rows[:-1],'dunhuang_cv',roles)
        roles['datasets']['dunhuang_cv']['roles']['real_test']['pair_ids']=['real_cal']
        with self.assertRaisesRegex(ValueError,'overlapping'):pop.population_groups(rows,'dunhuang_cv',roles)

    def test_sim_select_and_test_are_distinct_registered_roles(self):
        self.assertEqual(pop.COUNTS['sim_select'],1500);self.assertEqual(pop.COUNTS['sim_test'],3000)
        rows=[dict(pair_id='fixture')]
        self.assertEqual(pop.population_groups(rows,'sim_test',{}),{'all':rows})
        with self.assertRaises(ValueError):pop.population_groups(rows,'sim_test_v14',{})

    def test_tensor_allowlist_discards_loader_targets_without_using_them(self):
        batch={k:np.ones((1,1),dtype=bool if k.startswith('contour_valid') else np.float32) for k in pop.INPUTS}
        batch.update(labels='must not be converted',translation_a_to_b_rc='must not be used')
        result=pop.tensor_inputs(batch,'cpu')
        self.assertEqual(set(result),set(pop.INPUTS));self.assertEqual(result['mask_a'].dtype,torch.float32)
        self.assertEqual(result['contour_valid_a'].dtype,torch.bool)

    def test_turufan_positive_does_not_receive_layout_gt(self):
        rows=pop.targets_after_prediction(dict(pairs=[dict(pair_id='t',label=True)]),'turufan',{})
        self.assertEqual(rows,[dict(pair_id='t',label=True,gt_pose=None)])

    def test_sim_target_identity_and_positive_gt_required(self):
        sample=SimpleNamespace(pair_id='p',label=1.,translation_valid=True,translation_a_to_b_rc=np.array([3.,4.]))
        dataset=[(sample,{}, {})];meta=dict(pairs=[dict(pair_id='p')])
        self.assertEqual(pop.targets_after_prediction(meta,'sim_test',{},dataset)[0]['gt_pose'],[3.,4.])
        sample.translation_valid=False
        with self.assertRaisesRegex(ValueError,'missing'):pop.targets_after_prediction(meta,'sim_test',{},dataset)

    def test_dunhuang_gt_binding_and_endpoint_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);gt=root/'gt.json';roles=root/'roles.json'
            write_json(gt,dict(positive_pairs=[dict(pair_id='p',fragment_a_token='A',fragment_b_token='B',translation_gt_a_to_b_rc=[1.,2.])]))
            write_json(roles,dict(gt_path=str(gt)))
            plan=dict(real_split=dict(path=str(roles),sha256=file_sha(roles)),real_binding=dict(gt_sha256=file_sha(gt)))
            meta=dict(pairs=[dict(pair_id='p',label=True,fragment_a_id='A',fragment_b_id='B')])
            self.assertEqual(pop.targets_after_prediction(meta,'dunhuang_cv',plan)[0]['gt_pose'],[1.,2.])
            meta['pairs'][0]['fragment_a_id']='B'
            with self.assertRaisesRegex(ValueError,'order'):pop.targets_after_prediction(meta,'dunhuang_cv',plan)

    def test_plan_uses_contract_version_not_a_hardcoded_v14_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=Path(os.environ['CURRICULUM_CASE_PLAN']).resolve()
            real=Path(os.environ['CURRICULUM_REAL_PLAN']).resolve()
            views={}
            for role,n in [('select',1500),('test',3000)]:
                path=root/(role+'.json');write_json(path,dict(split=role,entries=[dict(pair_id=role+str(i)) for i in range(n)]))
                views[role]=dict(path=str(path),sha256=file_sha(path),pair_count=n)
            contract=root/'contract.json';write_json(contract,dict(status='passed',source_disjoint=True,
                augmentation_revision='v17-synthetic-contract-fixture',validation=dict(select_mixed=views['select']),test=dict(mixed=views['test'])))
            spec=root/'spec.json';write_json(spec,dict(schema='curriculum-execution/1',locked=True,
                real_split=dict(path=str(real),sha256=file_sha(real)),simulation_contract=dict(path=str(contract),sha256=file_sha(contract))))
            fake=SimpleNamespace(bind_plan=lambda path:dict(synthetic_binding=True))
            with patch.object(pop,'bound_module',return_value=fake):
                plan=pop.freeze_plan(spec,case,root);pop.validate_plan(plan,spec,root)
            self.assertEqual(plan['simulation']['sim_test']['path'],views['test']['path'])
            self.assertEqual(plan['simulation_revision'],'v17-synthetic-contract-fixture')


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=importlib.import_module(BASE+'s7_consensus_v1.test_threshold_joint')

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        _,pair,proposals,_=self.fixture.setup_pair();self.target=proposals.clusters[0].translation.tolist()
        row=inspect_pair('p',pair,proposals,label=None,all_clusters=proposals.clusters,capture_edges=True)
        row['model_inputs_sha256']='c'*64;row['contour_points']=dict(a=pair.points_a.tolist(),b=pair.points_b.tolist(),
            original_a=pair.original_a.tolist(),original_b=pair.original_b.tolist(),coordinates='synthetic actual union fixture')
        self.row=row;self.matcher=torch.nn.Linear(1,1);self.meta=dict(pairs=[dict(pair_id='p')])
        self.batch={k:torch.zeros((1,1),dtype=torch.bool if k.startswith('contour_valid') else torch.float32) for k in pop.INPUTS}
        self.origin=dict(scorer_used=False,real_used_for_selection=False,split='sim_test',synthetic_cpu_fixture=True)

    def invoke(self,*,target=None,predictor=None,meta=None,batches=None,wanted=None):
        out=self.root/'out'
        def targets():
            self.assertTrue((out/'prediction_complete.json').exists())
            raw=run.read_jsonl(out/'pair_predictions.jsonl');self.assertIsNone(raw[0]['label'])
            return [dict(pair_id='p',label=True,gt_pose=self.target)] if target is None else target()
        with patch.object(run,'predict_batch',side_effect=predictor or (lambda *args:[copy.deepcopy(self.row)])):
            return run.run_population(self.matcher,None,self.root,self.meta if meta is None else meta,
                [(self.meta['pairs'],self.batch)] if batches is None else batches,dict(synthetic=True),
                out=out,provenance=self.origin,wanted_ids={'p'} if wanted is None else wanted,
                groups={('all' if self.origin['split'].startswith('sim_') else 'real_test'):['p']},
                targets_callback=targets,device='cpu')

    def test_full_file_completion_then_target_join_and_recount(self):
        self.invoke();audit=run.verify_population(self.root/'out')
        self.assertEqual(audit['pairs'],1);self.assertEqual(audit['diagnostic_cases'],1)
        self.assertTrue(audit['process_return_must_be_checked_separately'])
        summary=run.read(self.root/'out/summary.json')
        self.assertIsNone(summary['classification_accuracy']);self.assertEqual(summary['groups']['all']['q_sum_layout20'],1.)

    def test_corrupt_file_cannot_be_reported_complete(self):
        self.invoke();(self.root/'out/summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact changed'):run.verify_population(self.root/'out')

    def test_summary_recomputed_even_if_file_hashes_were_updated(self):
        self.invoke();out=self.root/'out';summary=run.read(out/'summary.json')
        summary['groups']['all']['pairs']=999;write_json(out/'summary.json',summary,replace=True)
        complete=run.read(out/'evaluation_complete.json');complete['files']['summary.json']=file_sha(out/'summary.json')
        write_json(out/'evaluation_complete.json',complete,replace=True)
        with self.assertRaisesRegex(ValueError,'summary differs'):run.verify_population(out)

    def test_gt_error_preserves_frozen_predictions_but_not_completion(self):
        with self.assertRaisesRegex(ValueError,'negative'):
            self.invoke(target=lambda:[dict(pair_id='p',label=False,gt_pose=self.target)])
        out=self.root/'out';self.assertTrue((out/'prediction_complete.json').exists())
        self.assertTrue((out/'failure.json').exists());self.assertFalse((out/'evaluation_complete.json').exists())

    def test_missing_or_reordered_batches_not_complete(self):
        with self.assertRaisesRegex(ValueError,'missing'):self.invoke(batches=[])
        self.assertFalse((self.root/'out/prediction_complete.json').exists())

    def test_reordered_batch_rejected_before_network(self):
        predictor=lambda *args:(_ for _ in ()).throw(AssertionError('network should not run'))
        with self.assertRaisesRegex(ValueError,'batch order'):
            self.invoke(batches=[([dict(pair_id='other')],self.batch)],predictor=predictor)

    def test_model_state_change_prevents_prediction_completion(self):
        def predictor(*args):
            with torch.no_grad():self.matcher.weight.add_(1.)
            return [copy.deepcopy(self.row)]
        with self.assertRaisesRegex(ValueError,'model state'):self.invoke(predictor=predictor)
        self.assertFalse((self.root/'out/prediction_complete.json').exists())

    def test_existing_output_is_preserved_not_replayed(self):
        self.invoke();before=file_sha(self.root/'out/evaluation_complete.json')
        with self.assertRaises(FileExistsError):self.invoke()
        self.assertEqual(before,file_sha(self.root/'out/evaluation_complete.json'))

    def test_failure_takes_precedence_over_complete(self):
        self.invoke();write_json(self.root/'out/failure.json',dict(status='failed'))
        with self.assertRaisesRegex(ValueError,'failure'):run.verify_population(self.root/'out')

    def test_edge_numeric_audit_recomputes_q_and_geometry(self):
        audit=run.audit_edges(self.row);self.assertEqual(audit['status'],'passed');self.assertGreater(audit['edges_across_candidates'],0)
        for field in ('raw_q','residual_px','q_arc'):
            row=copy.deepcopy(self.row);row['candidates'][0]['edges'][field][0]+=10.
            with self.subTest(field=field),self.assertRaises(ValueError):run.audit_edges(row)

    def test_edge_duplicates_or_original_mapping_corruption_refused(self):
        row=copy.deepcopy(self.row);row['candidates'][0]['edges']['compact_indices'][1]=row['candidates'][0]['edges']['compact_indices'][0]
        with self.assertRaises(ValueError):run.audit_edges(row)
        row=copy.deepcopy(self.row);row['candidates'][0]['edges']['original_indices'][0][0]+=1
        with self.assertRaisesRegex(ValueError,'mapping'):run.audit_edges(row)

    def test_turufan_result_does_not_invent_layout(self):
        self.origin['split']='turufan'
        # Production supplies source-role groups; this core fixture uses one
        # group to test unknown-GT handling, not a claimed122-pair evaluation.
        self.invoke(target=lambda:[dict(pair_id='p',label=True,gt_pose=None)])
        summary=run.read(self.root/'out/summary.json')['groups']['real_test']
        self.assertIsNone(summary['q_sum_layout20']);self.assertEqual(summary['positive_layout_gt_count'],0)

    def completed_job(self):
        job=controller.jobs()[0];self.origin.update(selection_kind=job['selection'],split=job['split'])
        self.invoke();out=self.root/job['name'];(self.root/'out').rename(out)
        write_json(out/'independent_artifact_audit.json',run.verify_population(out))
        launch=self.root/(job['name']+'_launch.json');write_json(launch,dict(job=job,synthetic=True))
        write_json(self.root/(job['name']+'_return.json'),dict(returncode=0,launch_sha256=file_sha(launch)))
        return job

    def test_controller_reopens_real_runner_artifacts_after_return(self):
        job=self.completed_job();result=controller.verify_job(self.root,job)
        self.assertEqual(result['audit']['pairs'],1);self.assertEqual(result['audit']['diagnostic_cases'],1)

    def test_controller_return_failure_cannot_hide_complete_files(self):
        job=self.completed_job();path=self.root/(job['name']+'_return.json');returned=run.read(path)
        returned['returncode']=9;write_json(path,returned,replace=True)
        with self.assertRaisesRegex(ValueError,'successfully return'):controller.verify_job(self.root,job)


class ActualNetworkRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):native_fixture.ActualNativeMatcherTests.setUpClass.__func__(cls)

    def setUp(self):native_fixture.ActualNativeMatcherTests.setUp(self)

    def test_actual_small96_matcher_through_files_and_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'actual_cpu_fixture';meta=dict(pairs=[dict(pair_id='actual')])
            before=tree_sha(self.matcher.state_dict())
            run.run_population(self.matcher,self.geometry,self.root,meta,[(meta['pairs'],self.inputs)],
                dict(synthetic_small_canvas=True),out=out,
                provenance=dict(split='sim_test',scorer_used=False,real_used_for_selection=False,synthetic=True),
                wanted_ids={'actual'},groups={'all':['actual']},targets_callback=lambda:[dict(pair_id='actual',label=False,gt_pose=None)],device='cpu')
            self.assertEqual(run.verify_population(out)['pairs'],1)
            self.assertEqual(before,tree_sha(self.matcher.state_dict()))


class ControllerTests(unittest.TestCase):
    def test_exact_eight_jobs_no_scorer_or_extra_selection(self):
        jobs=controller.jobs();self.assertEqual(len(jobs),8)
        self.assertEqual({r['selection'] for r in jobs},{'sim_best','equal_budget_endpoint'})
        self.assertEqual({r['split'] for r in jobs},set(pop.SPLITS))
        self.assertEqual(len({r['name'] for r in jobs}),8)

    def test_explicit_command_uses_one_gpu_and_no_training_arguments(self):
        args=SimpleNamespace(spec='/spec',preparation='/prep',controller_root='/trained',order='mixed')
        command=controller.command('/python',args,'/population','/out',controller.jobs()[0])
        self.assertIn('--device',command);self.assertNotIn('torch.distributed.run',command)
        self.assertNotIn('--resume',command);self.assertNotIn('--learning-rate',command)

    def test_successful_cpu_children_are_each_verified_before_releasing_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seen=[]
            def verify(path,job):seen.append(job['name']);return dict(synthetic=True,job=job)
            with patch.object(controller,'identity',return_value=dict(pid=1,synthetic=True)),patch.object(controller.time,'sleep',side_effect=lambda _:actual_sleep(.001)):
                result=controller.execute_queue(root,[7],dict(os.environ,PYTHONPATH=tmp),
                    lambda job,out:[sys.executable,'-c','pass'],free_check=lambda *a:None,verify=verify)
            self.assertEqual(len(result),8);self.assertEqual(seen,[j['name'] for j in controller.jobs()])

    def test_nonzero_cpu_child_not_retried_and_remaining_jobs_not_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(controller,'identity',return_value=dict(pid=1,synthetic=True)),patch.object(controller.time,'sleep',side_effect=lambda _:actual_sleep(.001)),self.assertRaisesRegex(ValueError,'incomplete'):
                controller.execute_queue(root,[7],dict(os.environ,PYTHONPATH=tmp),
                    lambda job,out:[sys.executable,'-c','raise SystemExit(9)'],free_check=lambda *a:None,
                    verify=lambda *a:(_ for _ in ()).throw(AssertionError('not a successful child')))
            self.assertEqual(len(list(root.glob('*_launch.json'))),1)
            self.assertEqual(len(list(root.glob('*_return.json'))),1)

    def test_previous_cpu_preparation_cannot_authorize_new_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'previous.json';write_json(path,dict(status='passed',gpu_used=False,failures=0,errors=0,skipped=0))
            with self.assertRaisesRegex(ValueError,'exact native'):check_preparation(path)


if __name__=='__main__':unittest.main()
