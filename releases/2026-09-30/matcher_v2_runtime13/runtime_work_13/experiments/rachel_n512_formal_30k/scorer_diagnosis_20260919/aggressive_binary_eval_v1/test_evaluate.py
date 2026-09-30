from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from consensus_binary_eval_common import evaluate as common
from consensus_binary_eval_adapter.test_evaluate import DummyMatcher
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.binary_scorer_v1.test_binary import fixture
from . import evaluate
from .contracts import read,save,sha,SIM_SPLIT


class EvaluationTests(unittest.TestCase):
    def test_nonterminal_is_rejected_before_any_population(self):
        a=SimpleNamespace(root='unused',reference='unused',selection='real',real_plan='unused')
        with patch.object(evaluate,'load_selected',side_effect=ValueError('not terminal')),patch.object(evaluate,'load_population') as opened:
            with self.assertRaisesRegex(ValueError,'terminal'):evaluate.run(a,None)
            opened.assert_not_called()

    def test_turufan_does_not_invent_layout(self):
        rows=[dict(pair_id=str(i),label=i==0,gt_known=False,layout20=False,error_px=None,candidate_coverage=False,
                   score=.8 if i==0 else .1,has_candidate=True,numeric_valid=True) for i in range(2)]
        result=evaluate.population_summary(rows,.3,'turufan');self.assertEqual(result['f1'],1.)
        for k in ('layout20','joint_f1','joint_fp','candidate_coverage'):self.assertIsNone(result[k])

    def test_actual_patch_inference_logs_same_forward_layers_without_labels(self):
        model,pair,proposals,_=fixture('patch');model.matcher=DummyMatcher();model.eval().requires_grad_(False)
        before=common.state_digest(model);score=model.score_pair
        cases=Path(common.__file__).with_name('case_plan.json');plan=read(cases)
        pid=next(c['pair_id'] for c in plan['cases'] if c['split']=='turufan')
        ids=[pid,'negative1','positive2','negative2','positive3','negative3']
        meta=dict(pairs=[dict(pair_id=p,label=i%2==0) for i,p in enumerate(ids)])
        batch={k:np.zeros((6,5) if k.startswith('contour_valid') else (6,5,2)) for k in common.INPUTS}
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);roles=root/'roles.json';prep=root/'prep.json'
            rs={role:{'pair_ids':ids[2*i:2*i+2]} for i,role in enumerate(('real_cal','real_select','real_test'))}
            save(roles,dict(datasets=dict(dunhuang_cv={'excluded_gt_pair_ids':plan['user_confirmed_gt_exclusions']},
                turufan=dict(excluded_gt_pair_ids=[],roles=rs,manifest_sha256='manifest'))));save(prep,{'synthetic':True})
            provenance=dict(arm='scratch_aggressive',variant='binary_patch',experiment_variant='aggressive_binary_patch',
                thresholds={'turufan':.3},checkpoint_sha256='CPU synthetic fixture',
                real_development_binding={'sources':{'turufan':{'inputs_sha256':'inputs'}}})
            a=SimpleNamespace(root=root,reference='unused',selection='real',real_plan=roles,case_plan=cases,
                preparation=prep,out=root/'output',device='cpu',split='turufan')
            actual=evaluate.attach_targets
            def attach(*args,**kwargs):
                self.assertTrue((a.out/'prediction_complete.json').exists())
                self.assertEqual(read(a.out/'prediction_complete.json')['sha256'],sha(a.out/'pair_predictions.jsonl'))
                return actual(*args,**kwargs)
            with ExitStack() as stack:
                stack.enter_context(patch.object(evaluate,'load_selected',return_value=(model,{},provenance)))
                stack.enter_context(patch.object(evaluate,'PLAN_SHA',sha(roles)))
                stack.enter_context(patch.object(evaluate,'load_population',return_value=(meta,iter([(meta['pairs'],batch)]),
                    dict(manifest_sha256='manifest',inputs_sha256='inputs'),None)))
                stack.enter_context(patch.object(common.PairEvidence,'from_matcher',return_value=pair))
                forward=stack.enter_context(patch.object(model,'score_pair',side_effect=lambda p,**kw:score(p,proposals=proposals,**kw)))
                stack.enter_context(patch.object(evaluate,'attach_targets',side_effect=attach))
                evaluate.run(a,None);self.assertEqual(forward.call_count,6)
            self.assertEqual(common.state_digest(model),before)
            self.assertEqual(read(a.out/'status.json')['status'],'complete')
            raw=[json.loads(s) for s in (a.out/'pair_predictions.jsonl').read_text().splitlines()]
            self.assertTrue(all('label' not in r and 'target_translation_rc' not in r for r in raw))
            self.assertTrue(all(c['local_classification_present'] is False and c['conflict_evidence_px'] is None for r in raw for c in r['candidates']))
            result=read(a.out/'summary.json');self.assertEqual(result['main_group'],'real_test')
            self.assertEqual(result['groups']['real_test']['primary']['pairs'],2)
            self.assertFalse(result['real_test_is_historically_unseen']);self.assertIsNone(result['groups']['real_test']['primary']['joint_f1'])
            self.assertEqual(len(result['diagnostic_cases']),1)
            self.assertEqual(result['diagnostic_cases'][0]['numerical_audit_status'],'passed')

    def test_new_sim_summary_uses_new_split_not_legacy_test(self):
        rows=[dict(pair_id='synthetic',label=False,gt_known=False,score=.1,has_candidate=True,numeric_valid=True,
            layout20=False,error_px=None,candidate_coverage=False)]
        p=dict(thresholds={SIM_SPLIT:.3})
        result=evaluate.make_summary(rows,SIM_SPLIT,{},p)
        self.assertEqual(result['main_group'],'all');self.assertEqual(result['groups']['all']['primary']['pairs'],1)


if __name__=='__main__':unittest.main()
