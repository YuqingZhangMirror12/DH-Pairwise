"""Actual CPU gradients with a small fixture model; no production/GPU claims."""
import copy
import random
import unittest

import numpy as np
import torch
from torch import nn

from .exposure import digest
from .test_exposure import ledger
from . import test_optimizer_cursor
from .training_core import Topology, run_updates


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Dropout(.2), nn.Linear(4, 1))
        self.frozen = nn.Parameter(torch.tensor([42.]), requires_grad=False)

    def forward(self, batch):
        x, target = batch
        value = self.network(x).flatten()
        loss = nn.functional.binary_cross_entropy_with_logits(value, target)
        return loss, dict(bce=loss.detach()), dict(pairs=len(x))


def prepare():
    torch.set_num_threads(1); torch.manual_seed(731); random.seed(811); np.random.seed(231)
    model = Wrapper(); optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    value = ledger()
    dataset = [(torch.tensor([i / 30., (i % 5) / 5., 1.]), torch.tensor(float(row.label)))
               for i, row in enumerate(value.catalog)]
    return model, optimizer, dataset, value


def experiment(order='curriculum', resume=None, stop=None, observations=None):
    model, optimizer, dataset, value = prepare()
    def evaluate(update):
        model.eval()
        random.random(); np.random.random(); torch.rand(9)
        with torch.no_grad():
            return dict(sum=float(model.network(torch.ones(1, 3)).sum()), update=update)
    def checkpoint(state):
        if observations is not None:
            observations.append(copy.deepcopy(state))
    out = run_updates(model, optimizer, dataset, value, order, Topology(0, 1, 2, 2), 'cpu',
        [(0, .001), (8, .0005), (13, .00025)], (0, 7, 12, 16),
        resume=resume, binding=dict(source='synthetic', data_sha256=digest('fixture')),
        evaluate=evaluate, on_checkpoint=checkpoint, checkpoint_every=4, stop_after=stop)
    return copy.deepcopy(out)


class CoreTests(unittest.TestCase):
    assert_nested_equal = test_optimizer_cursor.OptimizerResumeTests.assert_nested_equal

    def compare(self, a, b):
        for key in ('model', 'optimizer', 'sampling', 'observations'):
            self.assert_nested_equal(a[key], b[key])
        self.assertTrue(torch.equal(a['rng']['torch'], b['rng']['torch']))
        self.assertEqual(a['rng']['python'], b['rng']['python'])
        self.assertTrue(np.array_equal(a['rng']['numpy'][1], b['rng']['numpy'][1]))

    def test_resume_actual_core_across_each_stage_boundary(self):
        for order in ('curriculum', 'mixed'):
            full = experiment(order)
            for boundary in (1, 6, 7, 8, 12, 13, 15):
                with self.subTest(order=order, boundary=boundary):
                    self.compare(full, experiment(order, resume=experiment(order, stop=boundary)))

    def test_callbacks_only_at_complete_updates_and_keep_frozen_parameter(self):
        saved = []; out = experiment(observations=saved)
        self.assertEqual([s['sampling']['completed_updates'] for s in saved], [0, 4, 7, 8, 12, 16])
        self.assertEqual(float(out['model']['frozen']), 42.)
        self.assertEqual(out['sampling']['completed_exposures'], 64)

    def test_returning_at_completed_endpoint_does_not_re_evaluate(self):
        full = experiment(); self.compare(full, experiment(resume=full))

    def test_resume_requires_complete_validation_history(self):
        saved = experiment(stop=7); saved['observations'] = []
        with self.assertRaisesRegex(ValueError, 'validation observations'):
            experiment(resume=saved)

    def test_resume_binding_cannot_change(self):
        saved = experiment(stop=7); saved['binding']['source'] = 'different'
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            experiment(resume=saved)

    def test_cannot_train_frozen_extra_parameter(self):
        model, _, data, value = prepare()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        with self.assertRaisesRegex(ValueError, 'intended trainable'):
            run_updates(model, optimizer, data, value, 'curriculum', Topology(0, 1, 4, 1), 'cpu',
                        [(0, .001)], (), binding={'synthetic':True})

    def test_no_implicit_lr_or_validation_schedule(self):
        for lr, marks in [([], ()), ([(0, .001)], (7, 0)), ([(0, .001), (16, .0001)], ())]:
            model, optimizer, data, value = prepare()
            with self.subTest(lr=lr, marks=marks), self.assertRaises(ValueError):
                run_updates(model, optimizer, data, value, 'curriculum', Topology(0, 1, 4, 1), 'cpu',
                            lr, marks, binding={'synthetic':True})


if __name__ == '__main__':
    unittest.main()
