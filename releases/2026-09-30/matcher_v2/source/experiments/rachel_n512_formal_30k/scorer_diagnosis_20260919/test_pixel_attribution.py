"""Tiny CPU derivative checks; no real-data result or GPU cost is inferred."""
from copy import deepcopy
import json
import unittest

import numpy as np
import torch

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.models.rachel_decoupled_score import build_decoupled_score_model
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.pixel_attribution import (
    DEFAULT_STRATA, differentiable_patches, finite_difference_check, gradient_determinism,
    probe_pixels, raw_mask_logit, select_pixel_cases,
)


class PixelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, depth=1):
        torch.manual_seed(271)
        config = RachelN512Config(canvas_size=80, coarse_size=32, contour_cap=16, patch_size=8,
            feature_dim=16, num_heads=4, landmark_count=4, context_layers=1, evidence_dim=8,
            sinkhorn_iterations=40, activation_checkpointing=False)
        model = build_decoupled_score_model(config, "cross_attention", phase="classifier",
            model_options={"cross_attention_depth": depth}).eval().requires_grad_(False)
        a, b = torch.zeros(1, 1, 80, 80), torch.zeros(1, 1, 80, 80)
        a[:, :, 15:57, 12:43] = 1
        b[:, :, 10:61, 28:63] = 1
        pa = torch.tensor([[[15., 12.], [15., 27.], [15., 42.], [35., 42.],
                            [56., 42.], [56., 27.], [56., 12.], [35., 12.]]])
        pb = torch.tensor([[[10., 28.], [10., 45.], [10., 62.], [35., 62.],
                            [60., 62.], [60., 45.], [60., 28.], [35., 28.]]])
        return model, [a, b, pa, pb, torch.ones(1, 8, dtype=torch.bool), torch.ones(1, 8, dtype=torch.bool)]

    def test_exact_full_forward_and_mask_ancestry_for_all_depths(self):
        for depth in (1, 2, 4):
            model, tensors = self.fixture(depth)
            before = deepcopy(model.state_dict())
            points_before = [value.clone() for value in tensors[2:]]
            with torch.no_grad():
                result, arrays = probe_pixels(model, tensors)
            self.assertLess(result["original_raw_logit_error"], 2e-6)
            self.assertLess(result["wrapper_logit_error"], 2e-6)
            self.assertEqual(result["original_patch_max_abs_error"], dict(a=0., b=0.))
            # Repeated binary pixels can produce exact MaxPool ties: an
            # autograd subgradient need not equal a central directional slope.
            # The diagnostic must expose, not hide, the numerical discrepancy.
            numerical = result["finite_difference"]
            self.assertIn(numerical["status"], ("passed", "needs_review", "low_signal"))
            error = min(check["absolute_error"] for check in numerical["checks"])
            limit = numerical["absolute_tolerance"] + numerical["relative_tolerance"] * abs(numerical["analytic_directional_derivative"])
            if numerical["status"] != "low_signal":
                self.assertEqual(numerical["status"], "passed" if error <= limit else "needs_review")
            for side in "ab":
                gradient = arrays["signed_gradient_" + side]
                self.assertEqual(gradient.shape, (80, 80))
                self.assertGreater(float(np.abs(gradient).sum()), 0)
                self.assertGreater(result["pixel_sensitivity"][side]["background_absolute_sum"], 0)
                np.testing.assert_allclose(arrays["absolute_gradient_" + side], abs(gradient))
                np.testing.assert_allclose(arrays["input_times_gradient_" + side], arrays["mask_" + side] * gradient)
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(tensors[2:], points_before)))
            json.dumps(result, allow_nan=False)

    def test_sampler_replay_exact_nearest_fractional_coordinates_and_padding(self):
        model, tensors = self.fixture()
        sampler = model.base_model.patch_sampler
        mask = torch.rand_like(tensors[0]).requires_grad_(True)
        points = torch.tensor([[[0., 0.], [79., 79.], [0.1, 79.], [40.4, 22.6],
                                [32.5, 62.5], [79., 0.2], [35., 12.], [79., 79.]]])
        valid = torch.tensor([[True, True, True, True, True, True, True, False]])
        expected = sampler(mask, points, valid)
        actual = differentiable_patches(sampler, mask, points, valid)
        self.assertTrue(torch.equal(expected, actual))
        self.assertFalse(expected.requires_grad)
        self.assertTrue(actual.requires_grad)
        self.assertEqual(float(actual[:, -1].abs().sum()), 0.)
        gradient = torch.autograd.grad(actual.sum(), mask)[0]
        # Nearest sampling's mask derivative is an integer sample count, not
        # bilinear fractional interpolation weights. Padding contributes zero.
        self.assertTrue(torch.equal(gradient, gradient.round()))
        self.assertGreater(int((gradient > 0).sum()), 0)
        self.assertTrue(torch.equal(points, points.detach()))

    def test_probe_preserves_valid_padding_and_overrides_outer_inference_mode(self):
        model, tensors = self.fixture()
        tensors[4][0, 3] = False
        tensors[5][0, 6] = False
        with torch.inference_mode():
            result, arrays = probe_pixels(model, tensors)
        self.assertEqual(result["original_patch_max_abs_error"], dict(a=0., b=0.))
        np.testing.assert_equal(arrays["valid_a"], tensors[4][0].numpy())
        np.testing.assert_equal(arrays["valid_b"], tensors[5][0].numpy())
        self.assertIn(result["finite_difference"]["status"], ("passed", "needs_review", "low_signal"))

    def test_continuous_input_derivative_matches_numerical_direction(self):
        # This verifies the replay's differentiation without asserting that a
        # continuous mask is a valid physical fragment or real-world example.
        model, tensors = self.fixture()
        masks = [torch.rand_like(value).requires_grad_(True) for value in tensors[:2]]
        raw = raw_mask_logit(model, *masks, *tensors[2:])[0]
        gradients = torch.autograd.grad(raw, masks)
        result = finite_difference_check(model, masks, tensors[2:], gradients,
            reference_logit=raw.detach())
        self.assertEqual(result["status"], "passed")
        self.assertLess(min(check["relative_error"] for check in result["checks"]), .02)

    def test_only_original_full_forward_runs_the_sinkhorn_ancestry(self):
        model, tensors = self.fixture()
        primal_calls = []
        hook = model.base_model.primal.register_forward_hook(lambda *args: primal_calls.append(1))
        try:
            probe_pixels(model, tensors)
        finally:
            hook.remove()
        # Two primal projections in the single original A/B full forward;
        # differentiable score replay and numerical checks do not invoke them.
        self.assertEqual(len(primal_calls), 2)

    def test_default_selection_is_fixed_before_gradients_and_order_stable(self):
        rows = []
        for i, (dataset, stratum) in enumerate(DEFAULT_STRATA):
            rows.extend([dict(dataset=dataset, stratum=stratum, pair_id="first-%d" % i),
                         dict(dataset=dataset, stratum=stratum, pair_id="second-%d" % i)])
        chosen = select_pixel_cases(rows)
        self.assertEqual([row["pair_id"] for row in chosen], ["first-%d" % i for i in range(8)])
        self.assertEqual(sum(row["dataset"] == "real" for row in chosen), 4)
        self.assertEqual(sum(row["dataset"] == "ood" for row in chosen), 4)
        with self.assertRaisesRegex(ValueError, "missing"):
            select_pixel_cases(rows[2:])

    def test_requires_frozen_parameters_and_registered_head(self):
        model, tensors = self.fixture()
        model.score_head.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "freeze every"):
            probe_pixels(model, tensors)
        model.requires_grad_(False).set_phase("matcher")
        with self.assertRaisesRegex(ValueError, "classifier-phase"):
            probe_pixels(model, tensors)

    def test_rejects_nonbinary_observed_input(self):
        model, tensors = self.fixture()
        tensors[0][0, 0, 10, 10] = .5
        with self.assertRaisesRegex(ValueError, "finite binary"):
            probe_pixels(model, tensors)

    def test_cpu_determinism_is_preserved_and_cuda_requires_opt_in(self):
        original = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with gradient_determinism(torch.device("cpu"), False) as receipt:
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(receipt["relaxed_for_backward"])
            with self.assertRaisesRegex(ValueError, "explicit"):
                with gradient_determinism(torch.device("cuda:0"), False):
                    pass
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            # The context manager itself uses no CUDA device operations, so its
            # restoration can be tested on CPU without claiming GPU validation.
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with gradient_determinism(torch.device("cuda:0"), True):
                    self.assertFalse(torch.are_deterministic_algorithms_enabled())
                    raise RuntimeError("fixture")
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(original, warn_only=warn)


if __name__ == "__main__":
    unittest.main()
