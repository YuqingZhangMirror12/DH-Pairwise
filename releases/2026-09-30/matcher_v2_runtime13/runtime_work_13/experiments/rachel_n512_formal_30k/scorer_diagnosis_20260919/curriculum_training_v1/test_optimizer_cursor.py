"""Synthetic CPU DataLoader/AdamW/RNG resume; NOT real Matcher or CUDA proof."""
import copy
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from .exposure import RankMicrobatches, learning_rate_at
from .test_exposure import ledger


def run(order, end=16, resume=None):
    torch.set_num_threads(1); torch.manual_seed(421)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Dropout(.2), nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    value = ledger()
    dataset = [(torch.tensor([i / 30., (i % 5) / 5., 1.]), torch.tensor(float(row.label)))
               for i, row in enumerate(value.catalog)]
    if resume is None:
        sampler = RankMicrobatches(value, order, 0, 0, 1, 2, 2)
        completed = 0
    else:
        sampler = RankMicrobatches.from_cursor(value, resume['sampling'], 0, order, 1, 2, 2)
        model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['rng']); completed = resume['sampling']['completed_updates']
    # DataLoader iterator seed consumption must not perturb model dropout RNG.
    generator = torch.Generator().manual_seed(17 + completed)
    loader = DataLoader(dataset, batch_sampler=sampler, generator=generator, num_workers=0)
    optimizer.zero_grad(set_to_none=True)
    for micro_step, (x, target) in enumerate(loader):
        for group in optimizer.param_groups:
            group['lr'] = learning_rate_at(completed, [(0, .001), (8, .0005), (13, .00025)])
        prediction = model(x).flatten()
        loss = nn.functional.binary_cross_entropy_with_logits(prediction, target)
        (loss / 2).backward()
        if micro_step % 2 == 1:
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); completed += 1
            if completed == end:
                break
    return copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        rng=torch.get_rng_state(), sampling=sampler.cursor(completed)))


class OptimizerResumeTests(unittest.TestCase):
    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_resume_before_at_and_after_stage_boundary(self):
        for order in ('curriculum', 'mixed'):
            complete = run(order)
            for boundary in (1, 6, 7, 8, 12, 13, 15):
                with self.subTest(order=order, boundary=boundary):
                    resumed = run(order, resume=run(order, end=boundary))
                    self.assert_nested_equal(complete, resumed)

    def test_two_orders_are_not_accidentally_same_training(self):
        curriculum = run('curriculum'); mixed = run('mixed')
        self.assertTrue(any(not torch.equal(curriculum['model'][k], mixed['model'][k])
                            for k in curriculum['model']))
        self.assertEqual(curriculum['sampling']['completed_exposures'], mixed['sampling']['completed_exposures'])


if __name__ == '__main__':
    unittest.main()
