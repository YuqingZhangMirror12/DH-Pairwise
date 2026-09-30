"""Synthetic completed-file comparisons; no claimed real population results."""
import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import matcher_comparison as comp
from . import matcher_run as runner
from . import matcher_eval_controller as controller
from .checkpoint_io import file_sha, tree_sha, write_json
from .matcher_diagnostics import inspect_pair
from .runtime_io import read


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api = importlib.import_module('experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_threshold_joint')
        _, pair, proposal, _ = api.setup_pair()
        cls.target = proposal.clusters[0].translation.tolist()
        cls.row = inspect_pair('fixture', pair, proposal, label=None, all_clusters=proposal.clusters)
        cls.row['model_inputs_sha256'] = 'a'*64

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.plan = dict(synthetic_cpu_fixture=True)
        self.prep = self.root/'preparation.json'; write_json(self.prep, dict(source_sha256={'fixture.py': 'b'*64}))
        self.matcher = torch.nn.Linear(1, 1)
        self.batch = {k: torch.zeros((1,1), dtype=torch.bool if k.startswith('contour_valid') else torch.float32)
                      for k in ('mask_a','mask_b','points_rc_a','points_rc_b','contour_valid_a','contour_valid_b')}

    def origin(self, order, choice):
        return dict(schema='curriculum-matcher-evaluation-origin/1', order=order, module='matcher',
            stop_reason='fixed_shared_update_budget', scorer_used=False, real_used_for_selection=False,
            selection_kind=choice, total_completed_updates=10, updates=10 if choice=='equal_budget_endpoint' else 5,
            common_plan_sha256='c'*64, initial_matcher_state_sha256='d'*64,
            architecture={'fixture_width': 96}, geometry={'fixture': True}, proposal_revision='diameter16-fixture',
            matcher_state_sha256=tree_sha(self.matcher.state_dict()), selected_file_sha256='e'*64)

    def build_controller(self, order):
        root=self.root/order; root.mkdir(); plan=root/'population_plan.json'; write_json(plan,self.plan)
        launch=dict(order=order, jobs=controller.jobs(), population_plan_sha256=file_sha(plan),
                    preparation_sha256=file_sha(self.prep), automatic_retry=False, synthetic_cpu_fixture=True)
        write_json(root/'controller_launch.json',launch); records=[]
        for job in controller.jobs():
            origin=self.origin(order,job['selection'])
            origin.update(split=job['split'],population_plan_sha256=file_sha(plan),
                preparation_sha256=file_sha(self.prep), evaluation_source_sha256=read(self.prep)['source_sha256'])
            groups={'all':['fixture']} if job['split'].startswith('sim_') else {'real_test':['fixture']}
            target=None if job['split']=='turufan' else self.target
            with patch.object(runner,'predict_batch',side_effect=lambda *args:[copy.deepcopy(self.row)]):
                runner.run_population(self.matcher,None,root,dict(pairs=[dict(pair_id='fixture')]),
                    [([dict(pair_id='fixture')],self.batch)],dict(synthetic=True),out=root/job['name'],
                    provenance=origin,wanted_ids=set(),groups=groups,device='cpu',
                    targets_callback=lambda:[dict(pair_id='fixture',label=True,gt_pose=target)])
            write_json(root/job['name']/'independent_artifact_audit.json',runner.verify_population(root/job['name']))
            job_launch=root/(job['name']+'_launch.json'); write_json(job_launch,dict(job=job,synthetic_cpu_fixture=True))
            write_json(root/(job['name']+'_return.json'),dict(returncode=0,launch_sha256=file_sha(job_launch)))
            records.append(controller.verify_job(root,job))
        write_json(root/'evaluation_complete.json',dict(schema='curriculum-native-controller-complete/1',
            status='complete',order=order,job_count=8,jobs=records,frozen_model_evaluations=2,
            population_plan_sha256=file_sha(plan),scorer_used=False,automatic_retry=False))
        write_json(root/'driver_status.json',dict(status='complete',completed=8,active={},pending=[]))
        returned=root/'actual_fixture_process_return.json';write_json(returned,comp.return_receipt(root,order,0))
        return root,returned

    def verify(self,order):
        root=self.root/order;returned=root/'actual_fixture_process_return.json'
        expected={choice:self.origin(order,choice) for choice in ('sim_best','equal_budget_endpoint')}
        # One pair per role is strictly a synthetic file fixture, never the
        # production admission rule (1500/3000/803/602).
        with patch.object(comp,'COUNTS',{k:1 for k in comp.SPLITS}):
            return comp.verify_controller(root,returned,order,expected,self.plan,self.prep)

    def pair(self):
        self.build_controller('curriculum');self.build_controller('mixed')
        return self.verify('curriculum'),self.verify('mixed')

    def rewrite_controller(self,order,mutate):
        root=self.root/order;path=root/'evaluation_complete.json';value=read(path);mutate(value)
        write_json(path,value,replace=True)
        write_json(root/'actual_fixture_process_return.json',comp.return_receipt(root,order,0),replace=True)

    def test_completed_actual_files_reopened_for_both_eight_job_controllers(self):
        result=comp.assemble(*self.pair())
        self.assertEqual(len(result['comparisons']),8);self.assertIsNone(result['classification_accuracy'])
        self.assertFalse(result['scorer_used']);self.assertIsNone(result['joint_f1'])

    def test_turufan_never_gets_a_layout_accuracy_from_native_comparison(self):
        result=comp.assemble(*self.pair())['comparisons']['sim_best_turufan']['groups']['real_test']
        self.assertIsNone(result['curriculum']['q_arc_layout20'])
        self.assertEqual(result['paired_layout_transitions']['q_sum_winner_layout20']['positive_layout_gt_count'],0)

    def test_missing_controller_return_is_not_success(self):
        root,returned=self.build_controller('curriculum');returned.unlink()
        with self.assertRaises(FileNotFoundError):self.verify('curriculum')

    def test_nonzero_return_cannot_hide_complete_files(self):
        root,returned=self.build_controller('curriculum')
        write_json(returned,comp.return_receipt(root,'curriculum',9),replace=True)
        with self.assertRaisesRegex(ValueError,'return binding'):self.verify('curriculum')

    def test_stale_controller_failure_wins_over_complete(self):
        root,_=self.build_controller('curriculum');write_json(root/'controller_failure.json',dict(status='failed'))
        with self.assertRaisesRegex(ValueError,'failure'):self.verify('curriculum')

    def test_controller_pending_work_prevents_completion(self):
        root,_=self.build_controller('curriculum')
        write_json(root/'driver_status.json',dict(status='running',completed=8,active={'0':123},pending=[]),replace=True)
        with self.assertRaisesRegex(ValueError,'live/pending'):self.verify('curriculum')

    def test_duplicate_job_cannot_replace_missing_population(self):
        self.build_controller('curriculum')
        self.rewrite_controller('curriculum',lambda x:x['jobs'].__setitem__(-1,x['jobs'][0]))
        with self.assertRaisesRegex(ValueError,'missing/duplicate'):self.verify('curriculum')

    def test_modified_complete_file_is_not_a_successful_return(self):
        root,_=self.build_controller('curriculum');value=read(root/'evaluation_complete.json');value['fixture_extra']=True
        write_json(root/'evaluation_complete.json',value,replace=True)
        with self.assertRaisesRegex(ValueError,'return binding'):self.verify('curriculum')

    def test_job_prediction_file_corruption_is_reopened_not_trusted(self):
        root,_=self.build_controller('curriculum')
        (root/controller.jobs()[0]['name']/'pair_predictions.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.verify('curriculum')

    def test_different_verified_matcher_export_is_rejected(self):
        self.build_controller('curriculum');original=self.origin
        def changed(order,choice):return dict(original(order,choice),selected_file_sha256='f'*64)
        with patch.object(self,'origin',side_effect=changed),self.assertRaisesRegex(ValueError,'verified selected'):
            self.verify('curriculum')

    def test_full_population_counts_cannot_use_the_tiny_fixture(self):
        root,returned=self.build_controller('curriculum')
        expected={choice:self.origin('curriculum',choice) for choice in ('sim_best','equal_budget_endpoint')}
        with self.assertRaisesRegex(ValueError,'entire registered'):
            comp.verify_controller(root,returned,'curriculum',expected,self.plan,self.prep)

    def test_mismatched_groups_or_inputs_cannot_be_filtered_away(self):
        a,b=self.pair();name='sim_best_sim_test'
        b['jobs'][name]['rows'][0]['model_inputs_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'input identity'):comp.assemble(a,b)

    def test_different_population_plan_is_rejected(self):
        a,b=self.pair();b['population_plan_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'paired C/M'):comp.assemble(a,b)

    def test_different_total_update_budget_is_not_a_matched_comparison(self):
        a,b=self.pair();b['jobs']['sim_best_sim_test']['provenance']['total_completed_updates']=11
        with self.assertRaisesRegex(ValueError,'total_completed_updates'):comp.assemble(a,b)

    def test_different_sim_selected_updates_are_reported_not_hidden(self):
        a,b=self.pair();b['jobs']['sim_best_sim_test']['provenance']['updates']=4
        value=comp.assemble(a,b)['comparisons']['sim_best_sim_test']['groups']['all']
        self.assertFalse(value['same_selected_update']);self.assertEqual(value['selected_updates'],dict(curriculum=5,mixed=4))

    def test_endpoint_cannot_be_a_shorter_selected_model(self):
        a,b=self.pair();b['jobs']['equal_budget_endpoint_sim_test']['provenance']['updates']=4
        with self.assertRaisesRegex(ValueError,'endpoint'):comp.assemble(a,b)

    def test_changed_cases_are_paired_not_unmatched_aggregate_differences(self):
        a,b=self.pair();b['jobs']['sim_best_sim_test']['rows'][0]['q_arc_winner_layout20']=False
        value=comp.assemble(a,b)['comparisons']['sim_best_sim_test']['groups']['all']
        self.assertEqual(value['paired_changed_case_ids']['q_arc_winner_layout20'],
                         dict(curriculum_only=['fixture'],mixed_only=[]))

    def test_controller_completion_order_need_not_equal_dispatch_order(self):
        self.build_controller('curriculum');self.rewrite_controller('curriculum',lambda x:x['jobs'].reverse())
        self.assertEqual(len(self.verify('curriculum')['jobs']),8)


if __name__=='__main__':unittest.main()
