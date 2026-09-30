"""CPU synthetic unions/actual small-canvas Matcher; no heldout inference."""
import copy
from dataclasses import asdict, replace
import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .checkpoint_io import file_sha, tree_sha, write_json
from .exposure import digest
from .matcher_diagnostics import inspect_pair, summarize
from .matcher_evaluation import (annotate_population, annotate_prediction, compare_orders,
                                 predict_batch, prediction_sha)
from .matcher_terminal import load_matcher, verified_export
from .runtime_io import export_completed
from .runtime_plan import experiment_binding
from . import launcher, test_runtime_io as io_fixture

BASE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'


class PosthocTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = importlib.import_module(BASE + 's7_consensus_v1.test_threshold_joint')

    def setUp(self):
        _, self.pair, self.proposals, _ = self.fixture.setup_pair()
        self.target = self.proposals.clusters[0].translation.tolist()
        self.raw = inspect_pair('pair', self.pair, self.proposals, label=None,
                                all_clusters=self.proposals.clusters, capture_edges=True)
        self.raw['model_inputs_sha256'] = 'e' * 64

    def origin(self, order, choice='sim_best', update=8):
        return dict(schema='curriculum-matcher-evaluation-origin/1', module='matcher', order=order,
            selection_kind=choice, updates=update, total_completed_updates=16,
            common_plan_sha256='a'*64, stop_reason='fixed_shared_update_budget', scorer_used=False,
            real_used_for_selection=False, initial_matcher_state_sha256='b'*64, architecture={'width':96},
            geometry={'synthetic':True}, proposal_revision='native-hypothesis-complete-link-union/1-diameter16',
            evaluation_population_sha256='c'*64)

    def annotated(self, label=True, target=None):
        return annotate_prediction(self.raw, label=label, gt_pose=self.target if label and target is None else target)

    def test_unlabelled_rows_cannot_be_summarized_as_negatives(self):
        self.assertIsNone(self.raw['negative_max_q_sum'])
        with self.assertRaisesRegex(ValueError, 'native'):
            summarize([self.raw])

    def test_gt_not_accepted_during_unlabelled_prediction(self):
        with self.assertRaisesRegex(ValueError, 'unlabelled'):
            inspect_pair('pair', self.pair, self.proposals, label=None, gt_pose=self.target)

    def test_posthoc_matches_existing_labelled_diagnostic(self):
        expected = inspect_pair('pair', self.pair, self.proposals, label=True, gt_pose=self.target,
                                all_clusters=self.proposals.clusters, capture_edges=True)
        actual = self.annotated()
        # The saved-pose target join uses FP64, while the older online helper
        # uses the pair's FP32 dtype. Evidence and discrete decisions stay
        # exact here; only the target-derived distances admit roundoff.
        for a,b in zip(actual['candidates'],expected['candidates']):
            self.assertAlmostEqual(a['pose_error_px'],b['pose_error_px'],delta=1e-5)
            a['pose_error_px']=b['pose_error_px']
        for name in ('q_sum_winner_error_px','q_arc_winner_error_px'):
            self.assertAlmostEqual(actual[name],expected[name],delta=1e-5)
            actual[name]=expected[name]
        for name in expected:
            self.assertEqual(actual[name], expected[name], name)

    def test_positive_negative_and_turufan_keep_identical_evidence_and_ranking(self):
        before = copy.deepcopy(self.raw)
        for label, target in ((True,self.target), (True,[1000.,0.]), (False,None), (True,None)):
            row = annotate_prediction(self.raw, label=label, gt_pose=target)
            self.assertEqual(prediction_sha(row), prediction_sha(self.raw))
        self.assertEqual(self.raw, before)

    def test_turufan_does_not_get_fake_layout_or_classifier_metric(self):
        row = annotate_prediction(self.raw, label=True)
        summary = summarize([row])
        for name in ('q_sum_layout20', 'q_arc_layout20', 'classification_accuracy', 'joint_f1'):
            self.assertIsNone(summary[name])
        self.assertIsNone(row['budget_lost_correct'])

    def test_negative_gets_real_mass_but_not_common_pose(self):
        row = self.annotated(False)
        self.assertGreater(row['negative_max_q_sum'], 0.)
        self.assertIsNone(row['q_sum_winner_layout20'])
        with self.assertRaisesRegex(ValueError, 'negative'):
            self.annotated(False, self.target)

    def test_binary_labels_and_finite_gt_enforced(self):
        for label, target in ((1,None), ('positive',None), (True,[1.]), (True,[0.,float('nan')])):
            with self.subTest(label=label, target=target), self.assertRaises(ValueError):
                annotate_prediction(self.raw, label=label, gt_pose=target)

    def test_gt20_is_inclusive_and_not_changed_to_t16(self):
        raw = copy.deepcopy(self.raw)
        for c in raw['candidates']: c['translation_rc'] = [20., 0.]
        self.assertTrue(annotate_prediction(raw, label=True, gt_pose=[0.,0.])['q_sum_winner_layout20'])
        for c in raw['candidates']: c['translation_rc'] = [20.000001, 0.]
        self.assertFalse(annotate_prediction(raw, label=True, gt_pose=[0.,0.])['q_sum_winner_layout20'])

    def test_no_candidate_positive_kept_in_denominator(self):
        raw = inspect_pair('empty', self.pair, replace(self.proposals, clusters=()), label=None, all_clusters=())
        row = annotate_prediction(raw, label=True, gt_pose=self.target)
        self.assertFalse(row['retained_correct_coverage'])
        self.assertEqual(summarize([row])['q_sum_layout20'], 0.)

    def test_target_join_not_repeated_or_prepopulated(self):
        for raw in (self.annotated(), dict(self.raw, negative_max_q_sum=0.)):
            with self.assertRaises(ValueError): annotate_prediction(raw, label=True, gt_pose=self.target)

    def test_target_order_and_population_must_match(self):
        target = dict(pair_id='pair', label=True, gt_pose=self.target)
        self.assertEqual(len(annotate_population([self.raw], [target])), 1)
        for rows, targets in (([self.raw], []), ([self.raw,self.raw], [target,target]),
                              ([self.raw], [dict(target,pair_id='other')])):
            with self.assertRaises(ValueError): annotate_population(rows, targets)

    def test_candidate_retained_flags_and_prebudget_count_checked(self):
        for raw in (dict(self.raw, retained_count=0), dict(self.raw, prebudget_count=100)):
            with self.assertRaises(ValueError): annotate_prediction(raw, label=True, gt_pose=self.target)

    def test_paired_comparison_reports_model_gains_and_losses(self):
        correct = self.annotated(); wrong = copy.deepcopy(self.raw)
        for c in wrong['candidates']: c['translation_rc'] = [1000.,1000.]
        wrong = annotate_prediction(wrong, label=True, gt_pose=self.target)
        result = compare_orders([correct], [wrong], self.origin('curriculum'), self.origin('mixed',update=10))
        t = result['paired_layout_transitions']['retained_correct_coverage']
        self.assertEqual(t['curriculum_only'], 1); self.assertEqual(t['mixed_only'], 0)
        self.assertFalse(result['same_selected_update']); self.assertIsNone(result['classification_accuracy'])

    def test_same_endpoint_requires_actual_budget_update(self):
        row = self.annotated()
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            compare_orders([row],[row],self.origin('curriculum','equal_budget_endpoint',8),
                           self.origin('mixed','equal_budget_endpoint',16))
        value = compare_orders([row],[row],self.origin('curriculum','equal_budget_endpoint',16),
                               self.origin('mixed','equal_budget_endpoint',16))
        self.assertTrue(value['same_selected_update'])

    def test_sim_best_cannot_be_compared_as_other_endpoint(self):
        row = self.annotated()
        with self.assertRaisesRegex(ValueError, 'selection_kind'):
            compare_orders([row],[row],self.origin('curriculum'), self.origin('mixed','equal_budget_endpoint',16))

    def test_comparison_refuses_other_plan_seed_geometry_or_population(self):
        row = self.annotated()
        for field in ('common_plan_sha256','initial_matcher_state_sha256','geometry','evaluation_population_sha256'):
            other = dict(self.origin('mixed'), **{field:'changed'})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                compare_orders([row],[row],self.origin('curriculum'),other)

    def test_actual_model_inputs_and_gt_cannot_differ_between_orders(self):
        row = self.annotated()
        for field, value in (('pair_id','other'),('model_inputs_sha256','f'*64),('target_translation_rc',[999.,0.])):
            other = dict(row, **{field:value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                compare_orders([row],[other],self.origin('curriculum'),self.origin('mixed'))

    def test_no_real_selected_matcher_comparison(self):
        row = self.annotated()
        with self.assertRaises(ValueError):
            compare_orders([row],[row],dict(self.origin('curriculum'),real_used_for_selection=True), self.origin('mixed'))


class ActualNativeMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['CURRICULUM_BASELINE_SOURCE']).resolve()
        cls.arch = importlib.import_module('staging.pairwise_v0_2.models.rachel_n512')
        cls.scratch = importlib.import_module(BASE+'s7_consensus_v1.scratch_matcher')
        cls.matcher_api = importlib.import_module(BASE+'s7_consensus_v1.matcher')
        cls.geom_api = importlib.import_module(BASE+'s7_consensus_v1.compatibility')
        cls.fixture = importlib.import_module(BASE+'s7_consensus_v1.test_matcher')
        cls.head = importlib.import_module(BASE+'binary_scorer_v1.head')
        cls.source_sha = digest({str(p.relative_to(cls.root)):file_sha(p) for p in cls.root.rglob('*.py')})

    def setUp(self):
        torch.set_num_threads(1)
        self.architecture = self.arch.RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=16,
            landmark_count=2, context_layers=2, activation_checkpointing=False)
        self.matcher = self.scratch.fresh_matcher(self.architecture,26092407).set_frozen(True).eval()
        self.geometry = self.geom_api.CompatibilityConfig(.5,.5,.5,.5,1.,15.)
        self.inputs = dict(zip(self.matcher_api.INPUTS, self.fixture.inputs()))

    def predict(self, inputs=None):
        values = self.inputs if inputs is None else inputs
        return predict_batch(self.matcher,self.geometry,self.root,values,
                             ['pair'+str(i) for i in range(len(values['mask_a']))],capture_ids=('pair0',))

    def test_actual_network_no_scorer_no_old_head_and_one_sinkhorn(self):
        before = tree_sha(self.matcher.state_dict()); rng = torch.get_rng_state().clone()
        with patch.object(self.head.BinaryClusterHead,'forward',side_effect=AssertionError('Scorer')), \
                patch.object(self.matcher.base.coarse,'forward',side_effect=AssertionError('coarse')), \
                patch.object(self.matcher_api,'dustbin_sinkhorn',wraps=self.matcher_api.dustbin_sinkhorn) as sinkhorn:
            rows = self.predict()
        self.assertEqual(sinkhorn.call_count,1); self.assertEqual(before,tree_sha(self.matcher.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state())); self.assertIsNone(rows[0]['label'])
        self.assertEqual(len(rows[0]['model_inputs_sha256']),64); self.assertIn('contour_points',rows[0])

    def test_targets_rejected_in_predict_inputs(self):
        with self.assertRaisesRegex(ValueError, 'six'):
            self.predict(dict(self.inputs,labels=torch.ones(1)))

    def test_training_mode_or_trainable_matcher_refused(self):
        self.matcher.train()
        with self.assertRaisesRegex(ValueError, 'frozen'): self.predict()
        self.matcher.eval().set_frozen(False)
        with self.assertRaisesRegex(ValueError, 'frozen'): self.predict()

    def test_non_fp32_inputs_refused(self):
        with self.assertRaisesRegex(ValueError,'FP32'):
            self.predict(dict(self.inputs,mask_a=self.inputs['mask_a'].double()))

    def test_actual_input_change_updates_hash(self):
        old = self.predict()[0]
        new = self.predict(dict(self.inputs,mask_b=self.inputs['mask_b'].roll(1,-1)))[0]
        self.assertNotEqual(old['model_inputs_sha256'],new['model_inputs_sha256'])

    def test_each_pair_gets_fresh_builder_and_invalid_pair_has_no_stale_budget_list(self):
        policy=importlib.import_module(BASE+'s7_consensus_v1.pose_consensus')
        batch={k:v.repeat(2,*([1]*(v.ndim-1))) for k,v in self.inputs.items()}
        evidence=self.matcher(**batch)
        evidence=replace(evidence,numeric_valid=torch.tensor([True,False]))
        with patch.object(self.matcher,'forward',return_value=evidence), \
                patch.object(policy,'PoseConsensusBuilder',wraps=policy.PoseConsensusBuilder) as builder:
            rows=predict_batch(self.matcher,self.geometry,self.root,batch,['valid','invalid'])
        self.assertEqual(builder.call_count,2);self.assertFalse(rows[1]['numeric_valid'])
        self.assertEqual(rows[1]['retained_count'],0);self.assertEqual(rows[1]['prebudget_count'],0)

    def test_diagnostic_capture_does_not_change_predictions(self):
        a=self.predict()[0]
        b=predict_batch(self.matcher,self.geometry,self.root,self.inputs,['pair0'])[0]
        a.pop('contour_points')
        for candidate in a['candidates']:candidate.pop('edges',None)
        self.assertEqual(a,b)

    def export_fixture(self):
        head = self.head.BinaryClusterHead('patch')
        model = {'matcher.'+k:v.clone() for k,v in self.matcher.state_dict().items()}
        model.update({'head.'+k:v.clone() for k,v in head.state_dict().items()})
        common = dict(model_seed=26092407,baseline_sources_sha256=self.source_sha,synthetic_cpu_fixture=True)
        spec = dict(architecture=asdict(self.architecture),geometry=asdict(self.geometry),
            proposal_revision='native-hypothesis-complete-link-union/1-diameter16',initialization='shared_random_seed',
            selected_matcher=None,initial_matcher_state_sha256=tree_sha(self.matcher.state_dict()))
        saved = dict(model=model,binding=dict(module='matcher',common_plan=common,model_spec=spec))
        origin = dict(schema='curriculum-matcher-evaluation-origin/1',model_state_sha256=tree_sha(model),
            common_plan_sha256=digest(common),matcher_state_sha256=tree_sha(self.matcher.state_dict()))
        return saved, origin

    def test_exact_matcher_load_never_instantiates_scorer(self):
        saved, origin = self.export_fixture(); rng = torch.get_rng_state().clone()
        with patch.object(self.head.BinaryClusterHead,'__init__',side_effect=AssertionError('new Scorer')):
            matcher, geometry, provenance = load_matcher(saved,origin,self.root)
        self.assertEqual(tree_sha(matcher.state_dict()),tree_sha(self.matcher.state_dict()))
        self.assertTrue(matcher.frozen); self.assertFalse(matcher.training)
        self.assertFalse(provenance['old_head_imported']); self.assertEqual(geometry,self.geometry)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_load_refuses_wrong_state_hash_or_baseline_source(self):
        saved, origin = self.export_fixture()
        with self.assertRaisesRegex(ValueError, 'identity'):
            load_matcher(saved,dict(origin,model_state_sha256='x'*64),self.root)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError,'baseline'):
            load_matcher(saved,origin,Path(tmp))


class TerminalMatcherTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.spec = self.root/'synthetic_spec.json'
        write_json(self.spec,{'synthetic_cpu_fixture':True}); self.out = self.root/'formal'

    def make(self, order='curriculum'):
        def binding(*args):
            return dict(experiment_binding(*args),run_mode='formal',execution_manifest_sha256=file_sha(self.spec))
        with patch.object(io_fixture,'experiment_binding',side_effect=binding):
            _,self.plan,self.binding,_ = io_fixture.train_fixture(self.out,order=order)
        complete=export_completed(self.out/'exports',self.out/'checkpoints',self.plan,self.binding)
        complete['export_root']=str(self.out/'exports'); write_json(self.out/'training_complete.json',complete)
        launch=dict(phase='formal',command=['/python','--spec',str(self.spec),'--order',order,'--mode','formal','--out',str(self.out)])
        write_json(self.root/'formal_launch.json',launch)
        write_json(self.root/'formal_return.json',dict(phase='formal',returncode=0,launch_sha256=file_sha(self.root/'formal_launch.json')))
        write_json(self.root/'gpu_gate.json',dict(status='passed',formal_binding_sha256=digest({k:v for k,v in self.binding.items() if k!='run_mode'}),synthetic_parser_fixture_only=True))
        result=launcher.verify_formal(self.out,self.spec,self.plan,order)
        write_json(self.root/'export_process_return.json',result)
        write_json(self.root/'controller_complete.json',dict(result,successful_return_sha256=file_sha(self.root/'export_process_return.json'),
            formal_return_sha256=file_sha(self.root/'formal_return.json'),gpu_gate_sha256=file_sha(self.root/'gpu_gate.json')))

    def verify(self, order='curriculum', choice='sim_best'):
        return verified_export(self.root,self.spec,self.plan,order,choice)

    def test_both_registered_choices_with_real_export_hashes(self):
        self.make()
        for choice,update in [('sim_best',7),('equal_budget_endpoint',16)]:
            saved, origin=self.verify(choice=choice)
            self.assertEqual(origin['updates'],update);self.assertFalse(origin['scorer_used'])
            self.assertFalse(origin['claimed_converged']);self.assertEqual(origin['model_state_sha256'],tree_sha(saved['model']))

    def test_mixed_matcher_is_legal_for_native_evaluation(self):
        self.make('mixed'); _,origin=self.verify('mixed');self.assertEqual(origin['order'],'mixed')

    def test_nonzero_return_overrides_training_complete(self):
        self.make();path=self.root/'formal_return.json';row=launcher.read(path);row['returncode']=9
        write_json(path,row,replace=True)
        with self.assertRaisesRegex(ValueError,'return'):self.verify()

    def test_failure_overrides_old_success(self):
        self.make();write_json(self.root/'controller_failure.json',{'error':'synthetic'})
        with self.assertRaisesRegex(ValueError,'failure'):self.verify()

    def test_modified_model_file_is_not_a_valid_terminal(self):
        self.make();(self.out/'exports/sim_best.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'model changed'):self.verify()

    def test_changed_launch_or_controller_chain_refused(self):
        self.make();path=self.root/'formal_launch.json';value=launcher.read(path);value['command'][-1]='other'
        write_json(path,value,replace=True)
        with self.assertRaisesRegex(ValueError,'return/launch'):self.verify()

    def test_no_real_selected_or_unknown_order_evaluation(self):
        self.make()
        for order,choice in [('curriculum','real_best'),('unknown','sim_best')]:
            with self.assertRaisesRegex(ValueError,'registered'):self.verify(order,choice)


if __name__=='__main__':
    unittest.main()
