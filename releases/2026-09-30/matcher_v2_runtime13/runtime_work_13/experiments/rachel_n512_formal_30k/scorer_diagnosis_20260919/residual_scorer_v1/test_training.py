"""CPU wiring/resume test, not a GPU gate or a real-data experiment.

Runs the actual 96-wide Matcher on a small synthetic 32px canvas. Candidate
unions are supplied synthetically so a random untrained Matcher cannot make
the head test vacuously pass with zero proposals. Builder tests are separate.
"""
from copy import deepcopy
from dataclasses import replace
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from ..curriculum_training_v1.exposure import STAGES, build_ledger
from ..curriculum_training_v1.test_exposure import samples
from ..curriculum_training_v1.training_core import Topology, run_updates
from ..binary_scorer_v1.loss import batch_loss
from ..binary_scorer_v1.model import BinaryConsensus
from ..s7_consensus_v1 import matcher as matcher_module
from ..s7_consensus_v1.scratch_matcher import fresh_matcher
from ..s7_consensus_v1.preflight_matcher import state_digest
from ..s7_consensus_v1.test_matcher import inputs
from ..s7_consensus_v1.test_threshold_joint import setup_pair
from ..s7_consensus_v1.targets import PairLabels
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from .head import ResidualClusterHead, ARCHITECTURE


class ResidualTrainingTests(unittest.TestCase):
    def make(self):
        root = Path(os.environ['CURRICULUM_BASELINE_SOURCE']).resolve()
        self.assertIn(root, Path(matcher_module.__file__).resolve().parents)
        old, _, proposals, _ = setup_pair()
        config = RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=16,
            landmark_count=2, context_layers=2, activation_checkpointing=False)
        matcher = fresh_matcher(config, 26092407).set_frozen(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(26092406)
            head = ResidualClusterHead()
        model = BinaryConsensus(matcher, old.geometry, head=head)
        ids = torch.stack((torch.arange(4), torch.arange(4)), 1)
        pose = torch.tensor([3., 14.])
        proposal = replace(proposals.clusters[0], edge_ids=ids, original_union_edge_ids=ids.clone(),
            translation=pose, member_translations_rc=pose[None], actual_diameter_px=0., merged_hypothesis_ids=(0,))
        supplied = replace(proposals, clusters=(proposal,))

        class SyntheticUnion:
            def __call__(self, pair):
                if len(pair.q) != 4 or pair.local_a.shape[1] != 96:
                    raise ValueError('actual compact Matcher evidence is required')
                return supplied

        model.builder = SyntheticUnion()

        class TrainingModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self.forward_count = 0

            def forward(self, labels):
                self.forward_count += 1
                predictions = [self.model(*inputs())[0] for _ in labels]
                loss, records = batch_loss(self.model, predictions, labels)
                parts = {key: torch.stack([x.components[key].detach() for x in records]).mean()
                         for key in records[0].components}
                counts = {key: sum(x.counts[key] for x in records) for key in records[0].counts}
                return loss, parts, counts

        ledger = build_ledger(samples((2, 2, 2)), dict(zip(STAGES, (2, 2, 2))), 33, effective_batch=2)
        data = [PairLabels(row.label, row.label, pose.clone(), torch.arange(4), torch.arange(4),
                           torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool))
                for row in ledger.catalog]
        return TrainingModule(), ledger, data

    def execute(self, wrapper, ledger, data, resume=None, stop=None):
        parameters = [p for p in wrapper.parameters() if p.requires_grad]
        self.assertEqual(sum(p.numel() for p in parameters), 36673)
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        result = run_updates(wrapper, optimizer, data, ledger, 'curriculum', Topology(0, 1, 1, 2),
            'cpu', [(0, 1e-4), (4, 5e-5)], (), collate_fn=lambda items: items,
            resume=resume, binding=dict(architecture=ARCHITECTURE, synthetic_fixture=True), stop_after=stop)
        return deepcopy(result)

    def test_actual_matcher_single_sinkhorn_and_head_update_with_frozen_matcher(self):
        wrapper, ledger, data = self.make()
        matcher_before = state_digest(wrapper.model.matcher)
        head_before = state_digest(wrapper.model.head)
        with patch.object(matcher_module, 'dustbin_sinkhorn', wraps=matcher_module.dustbin_sinkhorn) as sink:
            result = self.execute(wrapper, ledger, data)
        self.assertEqual(result['sampling']['completed_updates'], 6)
        self.assertEqual(sink.call_count, 12)
        self.assertEqual(wrapper.forward_count, 12)
        self.assertEqual(matcher_before, state_digest(wrapper.model.matcher))
        self.assertNotEqual(head_before, state_digest(wrapper.model.head))
        self.assertFalse(wrapper.model.matcher.base.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in wrapper.model.matcher.parameters()))
        self.assertTrue(wrapper.model.head.training)

    def test_update1_serialized_resume_matches_full_model_optimizer_rng_and_cursor(self):
        full = self.execute(*self.make())
        first = self.execute(*self.make(), stop=1)
        buffer = io.BytesIO()
        torch.save(first, buffer); buffer.seek(0)
        loaded = torch.load(buffer, map_location='cpu', weights_only=False)
        resumed = self.execute(*self.make(), resume=loaded)

        def equal(a, b, name='root'):
            self.assertEqual(type(a), type(b), name)
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b), name)
            elif isinstance(a, np.ndarray):
                self.assertTrue(np.array_equal(a, b), name)
            elif isinstance(a, dict):
                self.assertEqual(set(a), set(b), name)
                for key in a:
                    equal(a[key], b[key], name + '/' + str(key))
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b), name)
                for index, (x, y) in enumerate(zip(a, b)):
                    equal(x, y, name + '/' + str(index))
            else:
                self.assertEqual(a, b, name)

        for key in ('model', 'optimizer', 'sampling', 'observations', 'binding', 'rng'):
            equal(full[key], resumed[key], key)


if __name__ == '__main__':
    unittest.main()
