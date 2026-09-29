from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import evaluate
from .evaluate import tensor_inputs, prediction_record, attach_targets, population_summary
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.consensus_head import ConsensusEvidenceHead
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.model import S7Consensus
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_evidence import fixture


def batch():
    return {k: np.zeros((2,5) if k.startswith('contour_valid') else (2,5,2))
            for k in evaluate.INPUTS}


def raw(pair_id='p', score=.7, translation=(3.,4.), candidates=None):
    return dict(pair_id=pair_id, score=score, has_candidate=translation is not None,
                numeric_valid=True, translation=translation, accepted=score>=.5 and translation is not None,
                candidates=([dict(proposal_translation=(3.,5.), refined_translation=(3.,4.))]
                            if candidates is None else candidates))


class InferenceTests(unittest.TestCase):
    def test_matcher_receives_only_six_inputs(self):
        data = batch()
        data.update(labels=np.ones(2), target_a=np.arange(5), recipes=['GT'],
                    translation_a_to_b_rc=np.ones((2,2)), source_row='private target')
        result = tensor_inputs(data, 'cpu')
        self.assertEqual(tuple(result), evaluate.INPUTS)
        self.assertTrue(all(v.dtype == (torch.bool if k.startswith('contour_valid') else torch.float32)
                            for k,v in result.items()))
        data['mask_a'][:] = 1
        self.assertEqual(float(result['mask_a'].sum()), 0.)

    def test_dunhuang_gt_order_and_posthoc_error(self):
        meta = dict(pairs=[dict(pair_id='p',label=True,fragment_a_id='a',fragment_b_id='b',fold=3)])
        gt = {'p':dict(fragment_a_token='a',fragment_b_token='b',translation_gt_a_to_b_rc=[3.,4.])}
        prediction = raw()
        before = json.dumps(prediction)
        row = attach_targets([prediction],meta,'dunhuang_cv',ground_truth=gt)[0]
        self.assertEqual(json.dumps(prediction),before)
        self.assertTrue(row['gt_known'] and row['layout20'] and row['candidate_coverage'])
        self.assertEqual(row['error_px'],0.)
        self.assertEqual(row['proposal_errors_px'],[1.])
        self.assertEqual(row['fold'],3)
        gt['p']['fragment_b_token']='a'
        with self.assertRaisesRegex(ValueError,'endpoint order'):
            attach_targets([prediction],meta,'dunhuang_cv',ground_truth=gt)

    def test_turufan_unknown_layout_is_never_zero_accuracy(self):
        meta=dict(pairs=[dict(pair_id='p',label=True),dict(pair_id='n',label=False)])
        rows=attach_targets([raw(),raw('n',.2)],meta,'turufan')
        summary=population_summary(rows,.5)
        self.assertEqual(summary['accuracy'],1.)
        self.assertEqual(summary['auroc'],1.)
        for key in ('layout20','layout20_count','joint_f1','wrong_pose_accepted',
                    'candidate_coverage_count','positive_no_correct_candidate'):
            self.assertIsNone(summary[key])

    def test_synthetic_gt_only_positive_and_present(self):
        data=[(SimpleNamespace(label=True,translation_valid=True,
                    translation_a_to_b_rc=np.array([3.,4.])),{},{}),
              (SimpleNamespace(label=False,translation_valid=True,
                    translation_a_to_b_rc=np.array([0.,0.])),{},{})]
        meta=dict(pairs=[dict(pair_id='p'),dict(pair_id='n',negative_kind='same_parent_nonadjacent')])
        rows=attach_targets([raw(),raw('n')],meta,'sim_test_v14',data)
        self.assertTrue(rows[0]['gt_known'])
        self.assertFalse(rows[1]['gt_known'])
        self.assertEqual(rows[1]['negative_kind'],'same_parent_nonadjacent')

    def test_missing_reordered_predictions_rejected(self):
        meta=dict(pairs=[dict(pair_id='p',label=True)])
        with self.assertRaisesRegex(ValueError,'missing'):
            attach_targets([],meta,'turufan')
        with self.assertRaisesRegex(ValueError,'reordered'):
            attach_targets([raw('wrong')],meta,'turufan')

    def test_no_candidate_cannot_pass_by_score(self):
        rows=attach_targets([raw(score=.9,translation=None,candidates=[])],
                            dict(pairs=[dict(pair_id='p',label=True)]),'turufan')
        self.assertEqual(population_summary(rows,.3)['tp'],0)

    def test_actual_cpu_network_record_and_empty_q(self):
        model=S7Consensus(None,CompatibilityConfig(.5,.5,.5,.5,1.,30.),
                          head=ConsensusEvidenceHead(feature_dim=4))
        for pair in (fixture(),fixture(torch.zeros(5,5))):
            with torch.no_grad():
                pred=model.score_pair(pair,capture_diagnostics=True)
                record=prediction_record('test',pair,pred)
            json.dumps(record,allow_nan=False)
            self.assertEqual(record['has_candidate'],pred.has_candidate)
            self.assertEqual(record['candidate_count'],len(pred.clusters))
            self.assertNotIn('label',record)
            if pred.has_candidate:
                self.assertEqual(record['translation'],record['candidates'][pred.selected_cluster_id]['refined_translation'])

    def test_nonfinite_matcher_rejection_is_inspectable(self):
        q=torch.zeros(5,5);q[0,0]=float('nan')
        pair=replace(fixture(q),numeric_valid=False)
        model=S7Consensus(None,CompatibilityConfig(.5,.5,.5,.5,1.,30.),
                          head=ConsensusEvidenceHead(feature_dim=4))
        with torch.no_grad():
            record=prediction_record('invalid',pair,model.score_pair(pair))
        self.assertFalse(record['numeric_valid'])
        self.assertFalse(record['accepted'])
        self.assertIsNone(record['absolute_q_mass'])
        json.dumps(record,allow_nan=False)

    def test_run_refuses_ongoing_before_loading_population(self):
        with tempfile.TemporaryDirectory() as root:
            args=SimpleNamespace(root=root,arm='m12',reference='not-opened')
            with patch.object(evaluate,'load_population') as read_data:
                with self.assertRaisesRegex(ValueError,'both arms must finish'):
                    evaluate.run(args)
                read_data.assert_not_called()

    def test_parallel_run_also_gates_before_opening_population(self):
        args=SimpleNamespace(root='/unopened',arm='m12',reference='/unopened/reference',
                             completed_arm_only=True)
        with patch.object(evaluate, 'load_selected', side_effect=ValueError('selected arm must finish')) as gate, \
             patch.object(evaluate, 'load_population') as read_data:
            with self.assertRaisesRegex(ValueError, 'selected arm must finish'):
                evaluate.run(args)
            gate.assert_called_once_with(args.root, 'm12', args.reference, completed_arm_only=True)
            read_data.assert_not_called()

    def test_fixed_case_plan_keeps_both_main252_pairs(self):
        plan=evaluate.read(Path(evaluate.__file__).with_name('case_plan.json'))
        self.assertEqual(len(plan['cases']),11)
        self.assertEqual(len({c['pair_id'] for c in plan['cases']}),11)
        self.assertEqual(sum(c['alias'].startswith('main252') for c in plan['cases']),2)
        self.assertFalse(plan['selected_by_new_results'])
        self.assertEqual(len(plan['user_confirmed_gt_exclusions']),3)

    def test_cpu_orchestration_exports_same_pass_then_attaches_labels(self):
        class DummyMatcher(torch.nn.Module):
            def forward(self,*tensors):
                if len(tensors)!=6:
                    raise AssertionError('non-input data reached Matcher')
                return None
        model=S7Consensus(DummyMatcher(),CompatibilityConfig(.5,.5,.5,.5,1.,30.),
                          head=ConsensusEvidenceHead(feature_dim=4)).eval()
        pair=fixture()
        case_plan=Path(evaluate.__file__).with_name('case_plan.json')
        pid=next(c['pair_id'] for c in evaluate.read(case_plan)['cases'] if c['split']=='turufan')
        meta=dict(pairs=[dict(pair_id=pid,label=True),dict(pair_id='negative',label=False)])
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(root=temp,reference='unused',arm='m12',split='turufan',
                                 device='cpu',case_plan=str(case_plan),out=str(Path(temp)/'new'))
            with patch.object(evaluate,'load_selected',return_value=(model,{},dict(threshold=.3))), \
                 patch.object(evaluate,'load_population',return_value=(meta,[(meta['pairs'],batch())],{},None)), \
                 patch.object(evaluate.PairEvidence,'from_matcher',return_value=pair), \
                 patch.object(model,'score_pair',wraps=model.score_pair) as forward:
                evaluate.run(args)
                self.assertEqual(forward.call_count,2)
            out=Path(args.out)
            predictions=[json.loads(r) for r in (out/'pair_predictions.jsonl').read_text().splitlines()]
            self.assertTrue(all('label' not in r and 'target_translation_rc' not in r for r in predictions))
            labeled=[json.loads(r) for r in (out/'case_diagnostics.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['target_translation_rc'] is None for r in labeled))
            summary=evaluate.read(out/'summary.json')
            self.assertEqual(summary['status'],'complete')
            self.assertEqual(len(summary['diagnostic_cases']),1)
            detail=evaluate.read(out/summary['diagnostic_cases'][0]['evidence'])
            self.assertTrue(detail['semantics']['attention_weights_exported'])
            self.assertEqual(detail['attention_capture']['attention_calls'],16)
            self.assertIsNone(summary['groups']['all']['primary']['joint_f1'])
            self.assertTrue(evaluate.read(out/'prediction_complete.json')['model_state_unchanged'])
            self.assertFalse((out/'failure.json').exists())


if __name__=='__main__':
    unittest.main()
