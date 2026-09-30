from contextlib import ExitStack
import copy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import evaluate_reference_scorer as adapter


class GuardTests(unittest.TestCase):
    def origin(self, **change):
        return dict(selection_kind='sim_best', selection_on_test=False, threshold_refitted=False,
            thresholds=dict(sim_test=.37, dunhuang_cv=.25), **change)

    def test_reference_threshold_is_frozen_sim_not_real(self):
        origin = self.origin(); before = copy.deepcopy(origin)
        result = adapter.frozen_reference_threshold(origin)
        self.assertEqual(result['thresholds'][adapter.SPLIT], .37)
        self.assertEqual(origin, before)
        self.assertIn('not fitted', result['threshold_origins'][adapter.SPLIT])

    def test_endpoint_keeps_its_own_sim_threshold(self):
        origin = self.origin(); origin['selection_kind'] = 'equal_budget_endpoint'
        self.assertEqual(adapter.frozen_reference_threshold(origin)['thresholds'][adapter.SPLIT], .37)

    def test_real_test_or_refitted_selection_rejected(self):
        for key, value in (('selection_kind', 'real_best'), ('selection_on_test', True), ('threshold_refitted', True)):
            origin = self.origin(); origin[key] = value
            with self.assertRaises(ValueError):
                adapter.frozen_reference_threshold(origin)

    def test_invalid_cal_threshold_cannot_be_passed_through(self):
        for value in (True, .1, .9, float('nan')):
            origin = self.origin(); origin['thresholds']['sim_test'] = value
            with self.assertRaisesRegex(ValueError, 'SIM-CAL'):
                adapter.frozen_reference_threshold(origin)

    def test_cuda_request_rejected_before_model_or_data(self):
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES':'0'}), \
             patch.object(adapter, 'terminal_model', side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                adapter.run(SimpleNamespace())

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES':''}), \
             patch.object(adapter, 'terminal_model', side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError, 'exclusive'):
                adapter.run(SimpleNamespace(out=Path(tmp)))

    def test_incomplete_model_cannot_open_population(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES':''}), \
             patch.object(adapter, 'terminal_model', side_effect=ValueError('training incomplete')), \
             patch.object(adapter, 'reference_api', side_effect=AssertionError('must not open SELECT')):
            with self.assertRaisesRegex(ValueError, 'training incomplete'):
                adapter.run(SimpleNamespace(out=Path(tmp)/'new'))

    def test_dependency_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); adapter.save(root/'file.py', {'fixture':True})
            adapter.verify_code(root, {'file.py':adapter.sha(root/'file.py')})
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                adapter.verify_code(root, {'file.py':'0'*64})

    def test_only_lightweight_modules_and_registered_arms(self):
        for arm, module in (('B4','scorer_patch'), ('B0','matcher'), ('B3','old_complex')):
            with self.assertRaisesRegex(ValueError, 'registered'):
                adapter.terminal_model(SimpleNamespace(arm=arm, module=module))


class PopulationTests(unittest.TestCase):
    def test_groups_keep_original_order_no_result_filter(self):
        entries = [dict(pair_id=str(i), recipe='straight_'+kind, label=i%2 == 0)
                   for i, kind in enumerate(('J','M','R','J','M','R'))]
        groups = adapter.group_rows(entries, adapter.SPLIT, {'ignored':True})
        self.assertIs(groups['all'], entries)
        self.assertEqual([row['pair_id'] for row in groups['straight_J']], ['0','3'])
        self.assertEqual(set(groups), {'all','straight_M','straight_J','straight_R'})

    def test_other_population_cannot_enter_reference_adapter(self):
        with self.assertRaises(ValueError):
            adapter.group_rows([], 'turufan', {})

    def test_population_plan_binds_metadata_and_complete_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); adapter.save(root/'roles.json', {}); adapter.save(root/'case.json', {})
            roles = adapter.binding(root/'roles.json')
            result = adapter.population_plan({'rows':[{}]*900}, {'real_split':roles}, root/'case.json')
            self.assertEqual(result['pair_counts'], {adapter.SPLIT:900})
            self.assertEqual(result['case_plan'], adapter.binding(root/'case.json'))
            roles['sha256'] = '0'*64
            with self.assertRaises(ValueError):
                adapter.population_plan({'rows':[{}]*900}, {'real_split':roles}, root/'case.json')

    def test_join_requires_complete_actual_prediction_hash(self):
        class Dataset:
            def __len__(self):return 2
            def __getitem__(self, i):raise AssertionError('GT must not be loaded')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):adapter.target_join(root, Dataset())
            adapter.save(root/'pair_predictions.jsonl', {'pair_id':'p'})
            adapter.save(root/'prediction_complete.json', dict(status='all_predictions_frozen', pairs=2,
                model_state_unchanged=True, gt_used_for_prediction=False, sha256='0'*64))
            with self.assertRaisesRegex(ValueError, 'durable'):
                adapter.target_join(root, Dataset())

    def test_join_only_attaches_positive_gt_and_keeps_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); adapter.save(root/'pair_predictions.jsonl', {'pair_id':'p'})
            adapter.save(root/'prediction_complete.json', dict(status='all_predictions_frozen', pairs=2,
                model_state_unchanged=True, gt_used_for_prediction=False, sha256=adapter.sha(root/'pair_predictions.jsonl')))
            dataset = [(SimpleNamespace(label=True, translation_valid=True, translation_a_to_b_rc=np.array([2.,3.])),
                        None, dict(pair_id='p')),
                       (SimpleNamespace(label=False, translation_valid=False), None, dict(pair_id='n'))]
            self.assertEqual(adapter.target_join(root, dataset), [dict(pair_id='p',label=True,gt_pose=[2.,3.]),
                dict(pair_id='n',label=False,gt_pose=None)])


class DummyMatcher(torch.nn.Module):
    def forward(self, *inputs):
        if len(inputs) != 6:raise AssertionError('Only six model inputs allowed')
        return None


class ActualScorerReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        entry = importlib.import_module(adapter.PACKAGE+'.curriculum_scorer_eval_v1.entry')
        package = Path(entry.__file__).resolve().parent.parent
        if 'consensus_binary_eval_common' not in __import__('sys').modules:
            entry.bind_evaluation(package/'s7_consensus_eval_v14', package/'binary_eval_v1')
        source = package.parents[2]
        cls.evaluator, cls.auditor = adapter.readout_api(source, entry)
        cls.fixture = staticmethod(importlib.import_module(adapter.PACKAGE+'.binary_scorer_v1.test_binary').fixture)
        cls.checkpoint = importlib.import_module(adapter.PACKAGE+'.curriculum_training_v1.checkpoint_io')

    def fixture_data(self, root, variant):
        model, pair, proposals, _ = self.fixture(variant)
        model.matcher = DummyMatcher(); model.eval().requires_grad_(False)
        entries = [dict(pair_id=f'fixture/{kind}/{label}', recipe='straight_'+kind, label=label)
                   for kind in ('M','J','R') for label in (True,False)]
        cases = dict(schema='s7-consensus-fixed-diagnostics/1', selected_by_new_results=False,
            guide_sha256='0'*64, synthetic_test_fixture=True,
            cases=[dict(alias=f'fixture_dun_{i}',split='dunhuang_cv',pair_id=f'fixture/dun/{i}') for i in range(10)]
                + [dict(alias='fixture_turu',split='turufan',pair_id='fixture/turu')],
            user_confirmed_gt_exclusions=[f'fixture/excluded/{i}' for i in range(3)])
        adapter.save(root/'cases.json', cases); adapter.save(root/'roles.json', {})
        plan = adapter.population_plan({'rows':entries}, {'real_split':adapter.binding(root/'roles.json')}, root/'cases.json')
        origin = adapter.frozen_reference_threshold(dict(model_state_sha256=self.checkpoint.tree_sha(model.state_dict()),
            variant='binary_'+variant, thresholds=dict(sim_test=.37), selection_kind='sim_best', selected_updates=12,
            checkpoint_sha256='synthetic unit fixture', selection_on_test=False, threshold_refitted=False,
            historical_real_development_exposure=True))
        targets = [dict(pair_id=row['pair_id'],label=row['label'],gt_pose=[0.,0.] if row['label'] else None) for row in entries]
        batch = {k:np.zeros((6,5) if k.startswith('contour_valid') else (6,5,2)) for k in self.evaluator.INPUTS}
        batch.update(label=np.ones(6), translation_a_to_b_rc=np.ones((6,2))*12345)
        return model,pair,proposals,entries,plan,origin,targets,batch

    def run_fixture(self, variant, *, interrupt=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model,pair,proposals,entries,plan,origin,targets,batch = self.fixture_data(root,variant)
            out = root/'out'; actual = model.score_pair; before = self.checkpoint.tree_sha(model.state_dict())
            def join(meta, split, _plan, dataset):
                frozen = adapter.read(out/'prediction_complete.json')
                self.assertEqual(frozen['sha256'], adapter.sha(out/'pair_predictions.jsonl'))
                raw = [json.loads(line) for line in (out/'pair_predictions.jsonl').read_text().splitlines()]
                self.assertTrue(all('label' not in row and 'target_translation_rc' not in row for row in raw))
                return targets
            with ExitStack() as stack:
                stack.enter_context(patch.object(self.evaluator.common.PairEvidence,'from_matcher',return_value=pair))
                scoring = stack.enter_context(patch.object(model,'score_pair',side_effect=(ValueError('interrupted')
                    if interrupt else lambda p,**kw:actual(p,proposals=proposals,**kw))))
                from unittest.mock import Mock
                joined = Mock(side_effect=join)
                kwargs = dict(registered_splits=(adapter.SPLIT,), population_loader=lambda *a:(dict(pairs=entries),
                    iter([(entries,batch)]),{'synthetic_fixture':True},None), group_builder=adapter.group_rows,
                    target_loader=joined, extra_diagnostic_ids=[row['pair_id'] for row in entries if row['label']])
                if interrupt:
                    with self.assertRaisesRegex(ValueError,'interrupted'):
                        self.evaluator.run_population(model,None,plan,adapter.SPLIT,origin,out,'cpu',**kwargs)
                    joined.assert_not_called(); self.assertTrue((out/'failure.json').exists())
                    self.assertFalse((out/'prediction_complete.json').exists()); return
                summary = self.evaluator.run_population(model,None,plan,adapter.SPLIT,origin,out,'cpu',**kwargs)
                self.assertEqual(scoring.call_count,6); joined.assert_called_once()
            audit = self.auditor.verify_population(out)
            self.assertEqual(audit['pairs'],6); self.assertEqual(audit['diagnostic_cases'],3)
            self.assertTrue(audit['actual_rows_and_numeric_evidence_recomputed'])
            self.assertEqual(before,self.checkpoint.tree_sha(model.state_dict()))
            self.assertEqual(summary['threshold'],.37)
            for group in ('straight_M','straight_J','straight_R'):
                self.assertEqual(summary['groups'][group]['primary']['pairs'],2)
            self.assertFalse(summary['threshold_refitting'])
            self.assertTrue(summary['layout_gt_available'])
            return out, summary

    def test_patch_readout_actual_mlp_q_trace_and_postfreeze_metrics(self):
        self.run_fixture('patch')

    def test_stats_readout_actual_mlp_q_trace_and_postfreeze_metrics(self):
        self.run_fixture('stats')

    def test_interruption_no_target_join_or_false_completion(self):
        self.run_fixture('patch',interrupt=True)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = importlib.import_module('run_scorer_reference')

    def config(self):
        jobs=[]
        for module in adapter.MODULES:
            jobs.append(dict(arm='B0', module=module, source_root='/source10', engine_root='/source13',
                spec=dict(path='/spec/'+module,sha256='s'*64), preparation=dict(path='/cpu-proof',sha256='p'*64),
                training_root='/training/'+module, reference_source='/reference-code',
                admission=dict(root='/admission',complete_sha256='a'*64), selection='sim_best'))
        return dict(schema='reference-scorer-cpu-pipeline/1',cpu_workers=2,jobs=jobs)

    def test_exact_two_heads_same_arm(self):
        config=self.config()
        self.assertEqual(self.controller.validate_config(config),config['jobs'])
        for mutation in ('duplicate','cross_arm','new_field','real_selection','different_selection'):
            invalid=copy.deepcopy(config)
            if mutation=='duplicate':invalid['jobs'][1]=invalid['jobs'][0]
            elif mutation=='cross_arm':invalid['jobs'][1]['arm']='B3'
            elif mutation=='new_field':invalid['jobs'][0]['threshold']=.01
            elif mutation=='real_selection':invalid['jobs'][0]['selection']='real_best'
            else:invalid['jobs'][0]['selection']='equal_budget_endpoint'
            with self.assertRaises(ValueError):self.controller.validate_config(invalid)

    def test_no_extra_worker_or_unbound_spec(self):
        for change in ('workers','relative_spec'):
            config=self.config()
            if change=='workers':config['cpu_workers']=8
            else:config['jobs'][0]['spec']['path']='relative'
            with self.assertRaises(ValueError):self.controller.validate_config(config)

    def test_command_uses_terminal_and_no_threshold_or_gpu_argument(self):
        command=self.controller.command(self.config()['jobs'][0],Path('/out'))
        self.assertIn('--training-root',command);self.assertIn('--spec-sha',command)
        self.assertIn('--admission-sha',command);self.assertNotIn('--checkpoint',command)
        self.assertNotIn('--threshold',command);self.assertNotIn('--gpus',command)
        self.assertNotIn('--device',command)

    def test_parent_gpu_environment_never_leaks_into_cpu_child(self):
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0,5','OMP_NUM_THREADS':'32'}):
            env=self.controller.cpu_environment()
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            self.assertEqual(env[name],'1')

    def test_failed_child_never_certifies_output_or_retries(self):
        from unittest.mock import Mock
        controller=self.controller
        with tempfile.TemporaryDirectory() as tmp, patch.object(controller,'verify_inputs'), \
             patch.object(controller,'identity',return_value={'pid':123}), \
             patch.object(controller.subprocess,'Popen',return_value=Mock(pid=123,wait=lambda:1)) as spawn, \
             patch.object(controller,'verify_completed',side_effect=AssertionError('not called')):
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'child failed'):
                controller.run_job(self.config()['jobs'][0],root)
            spawn.assert_called_once()
            work=root/'scorer_patch'
            self.assertTrue((work/'failure.json').exists());self.assertFalse((work/'complete.json').exists())
            self.assertEqual(adapter.read(work/'return.json')['returncode'],1)

    def test_success_requires_actual_return_plus_verified_artifacts(self):
        from unittest.mock import Mock
        controller=self.controller
        with tempfile.TemporaryDirectory() as tmp, patch.object(controller,'verify_inputs'), \
             patch.object(controller,'identity',return_value={'pid':123}), \
             patch.object(controller.subprocess,'Popen',return_value=Mock(pid=123,wait=lambda:0)), \
             patch.object(controller,'verify_completed',return_value={'status':'complete'}) as verify:
            root=Path(tmp);job=self.config()['jobs'][0]
            result=controller.run_job(job,root)
            verify.assert_called_once_with(job,root/'scorer_patch/evaluation')
            returned=adapter.read(root/'scorer_patch/return.json')
            self.assertEqual(returned['launch_sha256'],adapter.sha(root/'scorer_patch/launch.json'))
            self.assertEqual(result['actual_child_return'],adapter.binding(root/'scorer_patch/return.json'))

    def test_failure_artifact_takes_priority_over_old_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);adapter.save(root/'failure.json',{'error':'synthetic'})
            adapter.save(root/'evaluation_complete.json',{'status':'evaluation_complete'})
            with self.assertRaisesRegex(ValueError,'failure precedes'):
                self.controller.verify_completed(self.config()['jobs'][0],root)

    def test_empty_artifact_manifest_cannot_pass_by_vacuous_hash_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            adapter.save(root/'evaluation_complete.json',dict(schema='curriculum-scorer-evaluation-complete/1',
                status='evaluation_complete',files={}))
            adapter.save(root/'independent_artifact_audit.json',{'status':'passed'})
            with self.assertRaisesRegex(ValueError,'artifact membership'):
                self.controller.verify_completed(self.config()['jobs'][0],root)

    def test_preparation_must_bind_all_actual_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'prep'
            adapter.save(path,dict(status='passed',tests=30,errors=0,failures=0,skipped=0,cuda_initialized=False,
                source_unchanged=True,source_sha256={},readout_code_sha256=adapter.READOUT_CODE,
                reference_code_sha256=adapter.REFERENCE_CODE))
            with self.assertRaisesRegex(ValueError,'source-bound'):
                self.controller.verify_preparation(path)


if __name__ == '__main__':
    unittest.main()
