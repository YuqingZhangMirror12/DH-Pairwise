"""Small CPU fixtures only; real-data execution is separate."""
from types import SimpleNamespace
import unittest

import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.pixel_max_branch_probe import MaxBranches, check_close
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.test_pixel_attribution import PixelTests
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919 import pixel_attribution as pixel


class BranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tiny_forward_gradient_fd_and_restore(self):
        model, tensors = PixelTests().fixture(depth=2)
        masks = [x.clone().requires_grad_(True) for x in tensors[:2]]
        fixed = tensors[2:]
        raw = pixel.raw_mask_logit(model, *masks, *fixed)[0]
        gradients = torch.autograd.grad(raw, masks)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        with MaxBranches(model) as branch:
            captured, trace = branch.run("capture", masks, fixed)
            check_close(captured.detach(), raw.detach(), "capture")
            frozen, _ = branch.run("frozen", masks, fixed)
            frozen_g = torch.autograd.grad(frozen, masks)
            check_close(frozen.detach(), raw.detach(), "frozen")
            for actual, expected in zip(frozen_g, gradients):
                check_close(actual, expected, "gradient")
            self.assertEqual(len(trace["max_operations"]), 8)
            self.assertGreater(sum(x["tied_outputs"] for x in trace["max_operations"][:6]), 0)
            directions = []
            for gradient in gradients:
                direction = torch.zeros_like(gradient).flatten()
                ids = gradient.abs().flatten().argsort(descending=True)[:16]
                direction[ids] = gradient.flatten()[ids].sign()
                directions.append(direction.reshape_as(gradient))
            analytic = float(sum((g * d).sum() for g, d in zip(frozen_g, directions)))
            with torch.no_grad():
                hi, _ = branch.run("frozen", [m + .001 * d for m, d in zip(masks, directions)], fixed)
                lo, _ = branch.run("frozen", [m - .001 * d for m, d in zip(masks, directions)], fixed)
            numeric = float((hi - lo) / .002)
            self.assertLess(abs(numeric - analytic), .001 + .1 * abs(analytic))
        self.assertNotIn("_pool", model.score_head.__dict__)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))

    def test_amax_ties_keep_equal_gradient_weights(self):
        model, _ = PixelTests().fixture(depth=2)
        value = torch.ones(3, 16, requires_grad=True)
        value = value * torch.arange(1., 17.)[None]
        expected = model.score_head._pool(value)
        expected_g = torch.autograd.grad(expected.sum(), value)[0]
        with MaxBranches(model) as branch:
            branch.mode, branch.cursor, branch.records, branch.pooled = "capture", 0, [], []
            observed = model.score_head._pool(value)
            self.assertTrue(torch.equal(observed, expected))
            self.assertTrue(torch.equal(branch.baseline[0]["weights"], torch.ones(3, 16) / 3))
            branch.mode, branch.cursor, branch.records, branch.pooled = "frozen", 0, [], []
            frozen = model.score_head._pool(value)
            actual_g = torch.autograd.grad(frozen.sum(), value)[0]
            check_close(actual_g, expected_g, "tied amax gradient")
            self.assertTrue(torch.equal(frozen, expected))

    def test_exception_restores_instance_methods(self):
        model, _ = PixelTests().fixture()
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with MaxBranches(model):
                raise RuntimeError("fixture")
        self.assertNotIn("_pool", model.score_head.__dict__)
        for module in model.base_model.patch_encoder.modules():
            if isinstance(module, torch.nn.MaxPool2d):
                self.assertNotIn("forward", module.__dict__)

    def test_unknown_pool_rejected(self):
        model, _ = PixelTests().fixture()
        model.base_model.patch_encoder.cnn[3] = torch.nn.MaxPool2d(3)
        with self.assertRaisesRegex(ValueError, "unregistered pool"):
            MaxBranches(model)


if __name__ == "__main__":
    unittest.main()
