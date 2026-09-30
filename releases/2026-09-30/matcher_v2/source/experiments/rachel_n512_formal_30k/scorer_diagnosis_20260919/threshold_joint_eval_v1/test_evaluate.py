from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from consensus_joint_eval_common import evaluate as common
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.consensus_head import ConsensusEvidenceHead
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.model import S7Consensus
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_evidence import fixture
from . import evaluate, loading
from .contracts import read, save, sha


class DummyMatcher(torch.nn.Module):
    def forward(self,*args):
        if len(args)!=6:raise AssertionError('only image/contour input allowed')
        return None


class EvaluationTests(unittest.TestCase):
    def test_actual_joint_attention_snapshot_keeps_experiment_and_protocol_distinct(self):
        model=S7Consensus(None,CompatibilityConfig(.5,.5,.5,.5,1.,30.),
                          head=ConsensusEvidenceHead(feature_dim=4)).eval().requires_grad_(False)
        pair=fixture()
        with torch.no_grad(), common.AttentionTrace(model.head) as trace:
            prediction=model.score_pair(pair,threshold=.3,capture_diagnostics=True)
        provenance=evaluate.snapshot_provenance(dict(variant='threshold_joint',
            evidence_mode='exact_union_q',checkpoint_sha256='synthetic-no-checkpoint'))
        self.assertEqual(provenance['experiment_variant'],'threshold_joint')
        self.assertEqual(provenance['variant'],'threshold')
        metadata,arrays=common.snapshot_prediction('synthetic-joint-case',pair,prediction,
            threshold=.3,provenance=provenance,attention_trace=trace)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'case'
            common.write_snapshot(out,metadata,arrays)
            audit=common.audit_snapshot(out/'evidence.json')
        self.assertEqual(audit['status'],'passed')
        self.assertEqual(audit.get('errors',[]),[])

    def test_no_inference_before_terminal_loader(self):
        args=SimpleNamespace(model_kind='joint',root='r',reference='c',selection='real',real_plan='p')
        with patch.object(evaluate,'load_joint',side_effect=ValueError('not terminal')),patch.object(common,'load_population') as pop:
            with self.assertRaisesRegex(ValueError,'terminal'):evaluate.run(args,None)
            pop.assert_not_called()

    def test_frozen_control_loader_preserves_strict_existing_guard(self):
        with patch.object(loading.base,'load_selected',side_effect=ValueError('Matcher changed')) as strict:
            with self.assertRaisesRegex(ValueError,'Matcher changed'):loading.load_frozen_control('r','c')
            strict.assert_called_once_with('r','scratch_fixed','c',completed_arm_only=True)

    def test_turufan_cannot_get_layout_or_joint_numbers(self):
        rows=[dict(pair_id=str(i),label=i==0,gt_known=False,layout20=False,error_px=None,
                   candidate_coverage=False,score=.8 if i==0 else .1,has_candidate=True,numeric_valid=True)
              for i in range(2)]
        m=evaluate.population_summary(rows,.3,'turufan')
        self.assertEqual(m['f1'],1.)
        for key in ('layout20','candidate_coverage','joint_f1','joint_fp','wrong_pose_accepted'):
            self.assertIsNone(m[key])

    def test_real_summary_main_population_is_holdout_not_all(self):
        rows=[dict(pair_id=i,label=j%2==0,gt_known=False,layout20=False,error_px=None,
            candidate_coverage=False,score=.8,has_candidate=True,numeric_valid=True) for j,i in enumerate('abcdef')]
        plan={'datasets':{'turufan':{'excluded_gt_pair_ids':[],'roles':{
            'real_cal':{'pair_ids':['a','b']},'real_select':{'pair_ids':['c','d']},'real_test':{'pair_ids':['e','f']}}}}}
        provenance=dict(thresholds={'turufan':.3})
        result=evaluate.make_summary(rows,'turufan',plan,provenance)
        self.assertEqual(result['main_group'],'real_test')
        self.assertFalse(result['real_test_is_historically_unseen'])
        self.assertFalse(result['threshold_refitting'])
        self.assertEqual(result['groups']['real_test']['primary']['pairs'],2)

    def test_actual_cpu_scorer_runner_freezes_predictions_before_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);real_plan=root/'roles.json';prep=root/'preparation.json'
            fixed=Path(common.__file__).with_name('case_plan.json')
            save(real_plan,{'datasets':{'dunhuang_cv':{'excluded_gt_pair_ids':read(fixed)['user_confirmed_gt_exclusions']}}})
            save(prep,{'fixture':True})
            args=SimpleNamespace(model_kind='joint',root=root,reference='not_read',selection='real',
                real_plan=real_plan,case_plan=fixed,preparation=prep,out=root/'output',device='cpu',split='sim_test_v14')
            model=S7Consensus(DummyMatcher(),CompatibilityConfig(.5,.5,.5,.5,1.,30.),
                              head=ConsensusEvidenceHead(feature_dim=4)).eval().requires_grad_(False)
            before=common.state_digest(model)
            meta=dict(pairs=[dict(pair_id='positive'),dict(pair_id='negative')])
            batch={k:np.zeros((2,5) if k.startswith('contour_valid') else (2,5,2)) for k in common.INPUTS}
            data=[(SimpleNamespace(label=True,translation_valid=True,translation_a_to_b_rc=np.array([3.,4.])),{},{}),
                  (SimpleNamespace(label=False,translation_valid=True,translation_a_to_b_rc=np.array([0.,0.])),{},{})]
            provenance=dict(thresholds={'sim_test_v14':.3},checkpoint_sha256='CPU synthetic fixture')
            actual_attach=common.attach_targets
            def checked_attach(*a,**kw):
                self.assertTrue((args.out/'prediction_complete.json').exists())
                return actual_attach(*a,**kw)
            with ExitStack() as stack:
                stack.enter_context(patch.object(evaluate,'load_joint',return_value=(model,{},provenance)))
                stack.enter_context(patch.object(evaluate,'PLAN_SHA',sha(real_plan)))
                stack.enter_context(patch.object(common,'load_population',return_value=(meta,iter([(meta['pairs'],batch)]),{},data)))
                stack.enter_context(patch.object(common.PairEvidence,'from_matcher',side_effect=lambda *a:fixture()))
                stack.enter_context(patch.object(common,'attach_targets',side_effect=checked_attach))
                evaluate.run(args,None)
            self.assertEqual(common.state_digest(model),before)
            self.assertEqual(read(args.out/'status.json')['status'],'complete')
            receipt=read(args.out/'prediction_complete.json')
            self.assertTrue(receipt['model_state_unchanged'])
            self.assertEqual(receipt['sha256'],sha(args.out/'pair_predictions.jsonl'))
            raw=[json.loads(line) for line in (args.out/'pair_predictions.jsonl').read_text().splitlines()]
            self.assertTrue(all('label' not in row and 'target_translation_rc' not in row for row in raw))
            self.assertEqual(read(args.out/'summary.json')['groups']['all']['primary']['pairs'],2)


if __name__=='__main__':unittest.main()
