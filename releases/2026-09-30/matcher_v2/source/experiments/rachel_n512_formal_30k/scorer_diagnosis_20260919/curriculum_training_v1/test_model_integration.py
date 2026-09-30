"""CPU integration with bound S7/threshold/light modules and synthetic inputs.

Matcher feature width96 is real; fixture canvas/contour are deliberately small.
Head tests use constructed union evidence; none of these are full GPU gates.
"""
from dataclasses import replace
import copy
import importlib
import os
from pathlib import Path
import unittest

import torch
from torch import nn

from .exposure import STAGES, build_ledger
from .test_exposure import samples
from .training_core import Topology, run_updates

BASE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'


class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ['CURRICULUM_BASELINE_SOURCE']).resolve()
        cls.config = importlib.import_module(BASE + 's7_consensus_v1.config')
        cls.config_path = Path(cls.config.__file__).resolve()
        if root not in cls.config_path.parents or cls.config.TrainingConfig().schema != 'aggressive-binary-training/1':
            raise ValueError('tests must use the immutable prepared threshold/light baseline')
        cls.matcher_module = importlib.import_module(BASE + 's7_consensus_v1.scratch_matcher')
        cls.model_module = importlib.import_module(BASE + 'binary_scorer_v1.model')
        cls.head_module = importlib.import_module(BASE + 'binary_scorer_v1.head')
        cls.loss_module = importlib.import_module(BASE + 'binary_scorer_v1.loss')
        cls.train_module = importlib.import_module(BASE + 's7_consensus_v1.train')
        cls.geometry_module = importlib.import_module(BASE + 's7_consensus_v1.compatibility')
        cls.fixture_module = importlib.import_module(BASE + 's7_consensus_v1.test_matcher')
        cls.union_fixture = importlib.import_module(BASE + 's7_consensus_v1.test_threshold_joint')
        cls.labels_module = importlib.import_module(BASE + 's7_consensus_v1.targets')
        cls.hash_module = importlib.import_module(BASE + 's7_consensus_v1.preflight_matcher')
        cls.architecture = importlib.import_module('staging.pairwise_v0_2.models.rachel_n512')
        policy = importlib.import_module(BASE + 's7_consensus_v1.pose_consensus')
        if policy.REVISION != 'native-hypothesis-complete-link-union/1-diameter16':
            raise ValueError('not the fixed16 threshold builder')

    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(812)
        config = self.architecture.RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=16,
            landmark_count=2, context_layers=2, activation_checkpointing=False)
        self.matcher = self.matcher_module.fresh_matcher(config, 26092407)
        self.geometry = self.geometry_module.CompatibilityConfig(.5, .5, .5, .5, 1., 15.)
        self.ledger = build_ledger(samples((2, 2, 2)), dict(zip(STAGES, (2, 2, 2))), 33, effective_batch=2)

    def execute(self, module, data, collate, order='curriculum', resume=None, stop=None):
        optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=.0001)
        return copy.deepcopy(run_updates(module, optimizer, data, self.ledger, order,
            Topology(0, 1, 1, 2), 'cpu', [(0, .0001), (4, .00005)], (), collate_fn=collate,
            resume=resume, binding={'source':'bound96-fixture', 'synthetic':True}, stop_after=stop))

    def matcher_wrapper(self):
        head = self.head_module.BinaryClusterHead('patch'); head.requires_grad_(False)
        model = self.model_module.BinaryConsensus(self.matcher, self.geometry, head=head)
        return self.train_module.TrainModule(model, 'matcher', self.config.TrainingConfig())

    def matcher_data(self):
        args = self.fixture_module.inputs()
        keys = ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b', 'contour_valid_a', 'contour_valid_b')
        result = []
        for row in self.ledger.catalog:
            values = {key:value[0].detach().clone() for key, value in zip(keys, args)}
            positive = row.label
            target = torch.tensor([0, 1, 2, 3, -2, -2]) if positive else torch.tensor([-1, -1, -1, -1, -2, -2])
            if not positive:
                values['mask_b'] = values['mask_b'].roll(1, -1)
            values.update(labels=torch.tensor(float(positive)), target_a=target, target_b=target.clone(),
                translation_a_to_b_rc=torch.tensor([3.,14.]) if positive else torch.zeros(2),
                translation_valid=torch.tensor(positive), pose_enabled=torch.tensor(False))
            result.append(values)
        return result

    def test_matcher_actual_update_and_cross_stage_resume(self):
        module = self.matcher_wrapper(); data = self.matcher_data()
        before_matcher = self.hash_module.state_digest(module.model.matcher)
        before_head = self.hash_module.state_digest(module.model.head)
        full = self.execute(module, data, None)
        self.assertNotEqual(before_matcher, self.hash_module.state_digest(module.model.matcher))
        self.assertEqual(before_head, self.hash_module.state_digest(module.model.head))
        self.setUp(); partial = self.execute(self.matcher_wrapper(), self.matcher_data(), None, stop=3)
        self.setUp(); resumed = self.execute(self.matcher_wrapper(), self.matcher_data(), None, resume=partial)
        for key in full['model']:
            self.assertTrue(torch.equal(full['model'][key], resumed['model'][key]), key)

    def head_check(self, variant):
        self.matcher.set_frozen(True)
        model = self.model_module.BinaryConsensus(self.matcher, self.geometry,
                                                 head=self.head_module.BinaryClusterHead(variant))
        _, pair, proposals, _ = self.union_fixture.setup_pair()
        pair = replace(pair, **{name:getattr(pair,name).detach().repeat(1,24)
                                for name in ('local_a','local_b','context_a','context_b')})
        n = len(pair.q); PairLabels = self.labels_module.PairLabels
        data = [PairLabels(r.label, r.label, torch.tensor([7.,0.]), torch.arange(n), torch.arange(n),
                           torch.ones(n,dtype=torch.bool), torch.ones(n,dtype=torch.bool))
                for r in self.ledger.catalog]
        loss_module = self.loss_module
        class ScorerWrapper(nn.Module):
            def __init__(self):
                super().__init__(); self.model = model
            def forward(self, items):
                predictions = [self.model.score_pair(pair, proposals=proposals) for _ in items]
                loss, records = loss_module.batch_loss(self.model, predictions, items)
                parts = {k:torch.stack([x.components[k].detach() for x in records]).mean() for k in records[0].components}
                counts = {k:sum(x.counts[k] for x in records) for k in records[0].counts}
                return loss, parts, counts
        wrapper = ScorerWrapper()
        frozen = self.hash_module.state_digest(model.matcher); old_head = self.hash_module.state_digest(model.head)
        result = self.execute(wrapper, data, lambda items:items)
        self.assertEqual(result['sampling']['completed_updates'], 6)
        self.assertEqual(frozen, self.hash_module.state_digest(model.matcher))
        self.assertNotEqual(old_head, self.hash_module.state_digest(model.head))
        self.assertEqual(sum(p.numel() for p in model.head.parameters()), 34529 if variant == 'patch' else 3201)

    def test_patch_head_actual_update_full_matcher_frozen(self):
        self.head_check('patch')

    def test_stats_head_actual_update_full_matcher_frozen(self):
        self.head_check('stats')


if __name__ == '__main__':
    unittest.main()
