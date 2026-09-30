"""Small CPU tests of post-selection interventions; no data/model imports."""
from copy import deepcopy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from .probe_edge_metadata import VARIANTS, intervention


class TinyEdgeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_projection = nn.Linear(3, 3, bias=False, dtype=torch.float64)
        self.mate_projection = nn.Linear(3, 3, bias=False, dtype=torch.float64)
        self.edge_projection = nn.Linear(2, 3, bias=True, dtype=torch.float64)
        with torch.no_grad():
            self.self_projection.weight.copy_(torch.eye(3, dtype=torch.float64))
            self.mate_projection.weight.copy_(torch.tensor(
                [[2., 0., -1.], [1., 3., 0.], [0., -2., 4.]], dtype=torch.float64))
            self.edge_projection.weight.copy_(torch.tensor(
                [[2., -3.], [5., 7.], [-11., 13.]], dtype=torch.float64))
            self.edge_projection.bias.copy_(torch.tensor([17., -19., 23.], dtype=torch.float64))

    def forward(self, a, b, meta):
        common = self.edge_projection(meta)
        return (self.self_projection(a) + self.mate_projection(b) + common,
                self.self_projection(b) + self.mate_projection(a) + common)


def fixture():
    a = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]], dtype=torch.float64)
    b = torch.tensor([[[-2., 3., 1.], [5., -1., 2.]]], dtype=torch.float64)
    meta = torch.tensor([[[.125, .75], [.25, .5]]], dtype=torch.float64)
    return TinyEdgeHead().eval(), (a, b, meta)


class InterventionTests(unittest.TestCase):
    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)

    def assert_clean(self, model):
        self.assertFalse(model.edge_projection._forward_pre_hooks)
        self.assertFalse(model.mate_projection._forward_hooks)

    def test_noop_is_exact_and_handles_are_removed(self):
        model, args = fixture()
        baseline = model(*args)
        with intervention(model, "noop"):
            actual = model(*args)
        for x, y in zip(actual, baseline):
            self.assert_exact(x, y)
        self.assert_clean(model)

    def test_q_and_residual_scaling_are_separate_and_preserve_bias(self):
        model, (_, _, meta) = fixture()
        original = meta.clone()
        bias = model.edge_projection.bias.detach().clone()
        for variant in ("q_zero", "q_half", "q_double", "residual_zero",
                        "residual_half", "residual_double", "both_zero"):
            with self.subTest(variant=variant):
                qs, rs, _ = VARIANTS[variant]
                expected_meta = meta * torch.tensor([qs, rs], dtype=meta.dtype)
                expected = F.linear(expected_meta, model.edge_projection.weight, bias)
                with intervention(model, variant):
                    actual = model.edge_projection(meta)
                self.assert_exact(actual, expected)
                self.assert_exact(meta, original)
                self.assert_exact(model.edge_projection.bias, bias)
                if variant == "both_zero":
                    self.assert_exact(actual, bias.expand_as(actual))
                self.assert_clean(model)

    def test_mate_zero_affects_both_directions_and_not_common_branch(self):
        model, (a, b, meta) = fixture()
        common = model.edge_projection(meta)
        expected = (model.self_projection(a) + common,
                    model.self_projection(b) + common)
        baseline = model(a, b, meta)
        with intervention(model, "mate_zero"):
            actual = model(a, b, meta)
            self.assert_exact(model.edge_projection(meta), common)
        for x, y, original in zip(actual, expected, baseline):
            self.assert_exact(x, y)
            self.assertFalse(torch.equal(x, original))
        self.assert_clean(model)

    def test_exception_removes_both_hooks_and_restores_baseline(self):
        model, args = fixture()
        baseline = model(*args)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with intervention(model, "mate_zero"):
                model(*args)
                raise RuntimeError("intentional")
        self.assert_clean(model)
        for actual, expected in zip(model(*args), baseline):
            self.assert_exact(actual, expected)

    def test_every_variant_leaves_original_inputs_and_state_unchanged(self):
        model, args = fixture()
        saved_state = deepcopy(model.state_dict())
        originals = tuple(t.clone() for t in args)
        baseline = model(*args)
        for name in VARIANTS:
            with self.subTest(variant=name), torch.inference_mode():
                with intervention(model, name):
                    model(*args)
                self.assert_clean(model)
                for tensor, original in zip(args, originals):
                    self.assert_exact(tensor, original)
                for key, tensor in model.state_dict().items():
                    self.assert_exact(tensor, saved_state[key])
                for actual, expected in zip(model(*args), baseline):
                    self.assert_exact(actual, expected)


if __name__ == "__main__":
    unittest.main()
