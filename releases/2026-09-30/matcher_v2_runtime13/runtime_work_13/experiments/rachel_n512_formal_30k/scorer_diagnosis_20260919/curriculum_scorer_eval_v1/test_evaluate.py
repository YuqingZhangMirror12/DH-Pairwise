"""Real lightweight forward/snapshots on synthetic pairs; data loader mocked."""
from contextlib import ExitStack
import copy
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from consensus_binary_eval_common import evaluate as common
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.binary_scorer_v1.test_binary import fixture
from ..curriculum_training_v1.checkpoint_io import file_sha, tree_sha, write_json
from ..curriculum_training_v1.runtime_io import read
from . import evaluate, population


class DummyMatcher(torch.nn.Module):
    def forward(self,*inputs):
        if len(inputs)!=6: raise AssertionError('only six mask/contour inputs may enter Matcher')
        return None


class PopulationTests(unittest.TestCase):
    def test_posthoc_target_join_preserves_prediction_values(self):
        raw = dict(pair_id='p',translation=[0.,0.],numeric_valid=True,score=.4,
            candidates=[dict(proposal_translation=[20.,0.],refined_translation=[20.,0.])])
        targets = [dict(pair_id='p',label=True,gt_pose=[0.,0.])]; before=copy.deepcopy(raw)
        result=population.attach_targets([raw],targets,dict(pairs=[dict(pair_id='p')]))[0]
        self.assertTrue(result['layout20']);self.assertTrue(result['candidate_coverage'])
        self.assertEqual(raw,before)
        target=[dict(pair_id='p',label=True,gt_pose=None)]
        self.assertFalse(population.attach_targets([raw],target,dict(pairs=[dict(pair_id='p')]))[0]['gt_known'])

    def test_target_reorder_duplicates_and_prejoined_labels_refused(self):
        for raw, targets in [([dict(pair_id='p')],[dict(pair_id='other')]),
                            ([dict(pair_id='p'),dict(pair_id='p')],[]),
                            ([dict(pair_id='p',label=True)],[dict(pair_id='p',label=True,gt_pose=None)])]:
            with self.assertRaises(ValueError): population.attach_targets(raw,targets,dict(pairs=[dict(pair_id='p')]))

    def test_turufan_no_fake_layout_accuracy_or_joint_f1(self):
        rows=[dict(pair_id=str(i),label=i==0,gt_known=False,layout20=False,error_px=None,
                   candidate_coverage=False,score=.8 if i==0 else .1,has_candidate=True,numeric_valid=True)
              for i in range(2)]
        result=evaluate.population_summary(rows,.3,'turufan')
        self.assertEqual(result['f1'],1.)
        for name in ('layout20','joint_f1','joint_fp','candidate_coverage'):self.assertIsNone(result[name])


class InferenceTests(unittest.TestCase):
    def setup_fixture(self, root, variant):
        model,pair,proposals,_=fixture(variant);model.matcher=DummyMatcher();model.eval().requires_grad_(False)
        # Pure synthetic unit fixture: no private real-case IDs/GT exclusions
        # are needed to exercise post-prediction joins and snapshot auditing.
        case_path=root/'synthetic_case_plan.json'; pid='synthetic/turufan/positive1'
        cases=dict(schema='s7-consensus-fixed-diagnostics/1', selected_by_new_results=False,
            guide_sha256='0'*64, synthetic_test_fixture=True,
            cases=[dict(alias=f'synthetic_dun_{i}', split='dunhuang_cv', pair_id=f'synthetic/dun/{i}')
                   for i in range(10)]+[dict(alias='synthetic_turu', split='turufan', pair_id=pid)],
            user_confirmed_gt_exclusions=[f'synthetic/excluded/{i}' for i in range(3)])
        write_json(case_path,cases)
        ids=[pid,'negative1','positive2','negative2','positive3','negative3']
        roles={role:{'pair_ids':ids[2*i:2*i+2]} for i,role in enumerate(('real_cal','real_select','real_test'))}
        role_path=root/'roles.json'
        write_json(role_path,dict(datasets=dict(turufan=dict(excluded_gt_pair_ids=[],roles=roles))))
        plan=dict(case_plan=dict(path=str(case_path),sha256=file_sha(case_path)),
            real_split=dict(path=str(role_path),sha256=file_sha(role_path)),pair_counts=dict(turufan=6),
            simulation_revision='explicit CPU synthetic fixture, not real inference')
        origin=dict(model_state_sha256=tree_sha(model.state_dict()),variant='binary_'+variant,
            thresholds=dict(turufan=.3),selection_kind='real_best',selected_updates=12,
            checkpoint_sha256='synthetic fixture',historical_real_development_exposure=True)
        meta=dict(pairs=[dict(pair_id=p,label=i%2==0) for i,p in enumerate(ids)])
        targets=[dict(pair_id=p,label=i%2==0,gt_pose=None) for i,p in enumerate(ids)]
        batch={k:np.zeros((6,5) if k.startswith('contour_valid') else (6,5,2)) for k in common.INPUTS}
        return model,pair,proposals,plan,origin,meta,targets,batch

    def run_fixture(self, root, variant, interrupt=False):
        model,pair,proposals,plan,origin,meta,targets,batch=self.setup_fixture(root,variant)
        before=tree_sha(model.state_dict()); out=root/'out'; actual=model.score_pair
        def join(*args):
            self.assertTrue((out/'prediction_complete.json').exists())
            self.assertEqual(read(out/'prediction_complete.json')['sha256'],file_sha(out/'pair_predictions.jsonl'))
            return targets
        with ExitStack() as stack:
            stack.enter_context(patch.object(evaluate,'load_population',return_value=(meta,
                iter([(meta['pairs'],batch)]),{'synthetic':True},None)))
            stack.enter_context(patch.object(common.PairEvidence,'from_matcher',return_value=pair))
            forward=stack.enter_context(patch.object(model,'score_pair',side_effect=(ValueError('synthetic interruption')
                if interrupt else lambda p,**kw:actual(p,proposals=proposals,**kw))))
            joining=stack.enter_context(patch.object(evaluate,'targets_after_prediction',side_effect=join))
            if interrupt:
                with self.assertRaisesRegex(ValueError,'interruption'):
                    evaluate.run_population(model,None,plan,'turufan',origin,out,'cpu')
                joining.assert_not_called();self.assertTrue((out/'failure.json').is_file())
                self.assertFalse((out/'prediction_complete.json').exists());return
            result=evaluate.run_population(model,None,plan,'turufan',origin,out,'cpu')
            self.assertEqual(forward.call_count,6);joining.assert_called_once()
        self.assertEqual(before,tree_sha(model.state_dict()))
        raw=[json.loads(line) for line in (out/'pair_predictions.jsonl').read_text().splitlines()]
        self.assertEqual(len(raw),6); self.assertTrue(all('label' not in r and 'target_translation_rc' not in r for r in raw))
        self.assertEqual(len(result['diagnostic_cases']),1)
        self.assertEqual(result['diagnostic_cases'][0]['numerical_audit_status'],'passed')
        self.assertEqual(result['groups']['real_test']['primary']['pairs'],2)
        self.assertIsNone(result['groups']['real_test']['primary']['joint_f1'])
        receipt=read(out/'evaluation_complete.json'); self.assertTrue(receipt['process_success_not_yet_certified'])
        self.assertEqual(receipt['summary_sha256'],file_sha(out/'summary.json'))
        self.assertEqual(receipt['actual_updates'],12)

    def test_patch_full_prediction_and_actual_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:self.run_fixture(Path(tmp),'patch')

    def test_stats_full_prediction_and_actual_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:self.run_fixture(Path(tmp),'stats')

    def test_interruption_never_joins_targets_or_claims_complete(self):
        with tempfile.TemporaryDirectory() as tmp:self.run_fixture(Path(tmp),'patch',interrupt=True)

    def test_frozen_guard_precedes_population_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,_,_,plan,origin,_,_,_=self.setup_fixture(Path(tmp),'patch');model.train()
            with patch.object(evaluate,'load_population') as loader,self.assertRaisesRegex(ValueError,'frozen'):
                evaluate.run_population(model,None,plan,'turufan',origin,Path(tmp)/'out','cpu')
            loader.assert_not_called()

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,_,_,plan,origin,_,_,_=self.setup_fixture(root,'patch');(root/'out').mkdir()
            with self.assertRaises(FileExistsError):evaluate.run_population(model,None,plan,'turufan',origin,root/'out','cpu')


if __name__=='__main__':unittest.main()
