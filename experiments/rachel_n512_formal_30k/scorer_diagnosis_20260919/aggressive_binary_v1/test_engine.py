"""Stage control with real CPU optimizer steps, mocked data/validation/CUDA.

This is NOT a DDP or real-data training gate. Actual network gradients have
separate tests; the production two-GPU gates remain mandatory.
"""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from ..s7_consensus_v1 import train
from ..s7_consensus_v1.config import TrainingConfig
from ..s7_consensus_v1.preflight_matcher import state_digest


class ToyMatcher(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.ones(1))

    def set_frozen(self, flag):
        self.requires_grad_(not flag)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__(); self.matcher = ToyMatcher(); self.head = nn.Linear(1,1)
        with torch.no_grad(): self.head.weight.fill_(.5); self.head.bias.zero_()


class ToyTrainModule(nn.Module):
    def __init__(self, model, stage, config, cache):
        super().__init__(); self.model = model; self.stage = stage

    def forward(self, batch):
        value = self.model.matcher.weight if self.stage == 'matcher' else self.model.head(self.model.matcher.weight.detach())
        loss = (value - 3.).square().mean()
        return loss, {'toy':loss.detach()}, {'pairs':1}


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = replace(TrainingConfig(), world_size=1, microbatch=24000, accumulate=1,
                              minimum_epochs=2, maximum_epochs=2, workers_per_rank=0)
        self.args = SimpleNamespace(out=str(self.root), arm='scratch_aggressive', resume=False,
                                    preflight_steps=0, real_split='synthetic-not-read')
        self.contract = {'train':{'path':'synthetic-not-read','sha256':'fixture'}}

    def invoke(self, stage, model):
        dataset = Mock(); dataset.__len__ = Mock(return_value=24000)
        real = Mock(); real.evaluate.return_value = (
            dict(key=[.7,.8,.9],thresholds={'dunhuang_cv':.3,'turufan':.4}), {})
        report = dict(key=[.6,.8,.7], threshold=.3, selection_value=.6)
        with ExitStack() as stack:
            mocks = {}
            replacements = dict(Dataset=Mock(return_value=dataset),
                DataLoader=Mock(return_value=[{'labels':torch.zeros(1)}]), TrainModule=ToyTrainModule,
                RealDevelopment=Mock(return_value=real), make_caches=Mock(return_value={'train':None}),
                validate=Mock(return_value=(report,{})), learning_curve_csv=Mock(return_value='fixture\n'),
                rng_state=Mock(return_value={'synthetic':True}), restore_rng=Mock())
            for name,value in replacements.items():
                mocks[name] = stack.enter_context(patch.object(train,name,value))
            stack.enter_context(patch.object(torch.cuda,'max_memory_allocated',return_value=0))
            result = train.run_stage(model,stage,self.args,self.config,self.contract,
                                     {'fixture':True,'real_development':{}},torch.device('cpu'),0,1)
        return result,mocks

    def test_matcher_uses_no_real_evaluation_and_no_head_updates(self):
        model=ToyModel(); h=state_digest(model.head); m=state_digest(model.matcher)
        result,mocks=self.invoke('matcher',model)
        mocks['RealDevelopment'].assert_not_called()
        self.assertFalse(result['selection_on_real']); self.assertIsNone(result['best_real'])
        self.assertEqual(result['best']['epoch'],2)  # E0 is measured, never a trained Matcher winner.
        self.assertEqual(h,state_digest(model.head)); self.assertNotEqual(m,state_digest(model.matcher))
        self.assertFalse((self.root/'matcher'/'best_real.pt').exists())

    def test_scorer_records_sim_and_real_and_keeps_matcher_frozen(self):
        model=ToyModel(); m=state_digest(model.matcher); h=state_digest(model.head)
        result,mocks=self.invoke('scorer',model)
        mocks['RealDevelopment'].assert_called_once()
        self.assertTrue(result['selection_on_real']); self.assertEqual(result['best_real']['epoch'],2)
        self.assertTrue((self.root/'scorer'/'best_real.pt').is_file())
        self.assertEqual(m,state_digest(model.matcher)); self.assertNotEqual(h,state_digest(model.head))

    def test_disposable_cpu_matcher_resume_matches_uninterrupted(self):
        self.config=replace(self.config, maximum_epochs=48)
        self.args.preflight_steps=12
        first,_=self.invoke('matcher',ToyModel())
        self.assertFalse(first['matcher_unchanged']); self.assertTrue(first['head_unchanged'])
        self.assertFalse(first['matcher_frozen_expected'])
        self.args.resume=True
        second,_=self.invoke('matcher',ToyModel())
        self.assertTrue(second['resume_matches_uninterrupted'])
        self.assertEqual(first['model_state_hashes'],second['model_state_hashes'])

    def test_disposable_cpu_scorer_resume_matches_uninterrupted(self):
        self.config=replace(self.config,maximum_epochs=48)
        self.args.preflight_steps=12
        first,_=self.invoke('scorer',ToyModel())
        self.assertTrue(first['matcher_unchanged']); self.assertFalse(first['head_unchanged'])
        self.assertTrue(first['matcher_frozen_expected'])
        self.args.resume=True
        second,_=self.invoke('scorer',ToyModel())
        self.assertTrue(second['resume_matches_uninterrupted'])
        self.assertEqual(first['model_state_hashes'],second['model_state_hashes'])


if __name__ == '__main__': unittest.main()
