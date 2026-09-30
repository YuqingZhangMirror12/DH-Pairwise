"""Pure CPU model tests, not training-data coverage or performance evidence."""
import copy
import unittest

import torch

from experiments.rachel_n512_formal_30k.score_design_density_model import (
    ARCHITECTURES, build_score_density_model, restore_score_density_model)
from experiments.rachel_n512_formal_30k.score_design_variant_model import build_score_input_model


class DensityModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_states_equal(self, a, b):
        self.assertEqual(set(a), set(b))
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]), name)

    def test_caps_share_every_initial_tensor_and_original_baseline(self):
        for architecture in ARCHITECTURES:
            small = build_score_density_model(512, architecture)
            dense = build_score_density_model(1024, architecture)
            baseline = build_score_input_model(architecture=architecture)
            self.assert_states_equal(small.model.state_dict(), dense.model.state_dict())
            self.assert_states_equal(small.model.state_dict(), baseline.model.state_dict())
            self.assertEqual(small.metadata["initial_weights_sha256"], dense.metadata["initial_weights_sha256"])
            self.assertEqual(dense.model.config.contour_cap, 1024)

    def test_typed_trained_state_roundtrip_and_sampling_grid(self):
        for architecture in ARCHITECTURES:
            built = build_score_density_model(1024, architecture)
            with torch.no_grad():
                next(built.model.parameters()).add_(.01)
            restored = restore_score_density_model(built.metadata, built.model.state_dict())
            self.assert_states_equal(built.model.state_dict(), restored.model.state_dict())
            prefix = "" if architecture == "original" else "base_model."
            broken = dict(built.model.state_dict())
            broken[prefix + "patch_sampler.offsets_rc"] = broken[prefix + "patch_sampler.offsets_rc"] * 2
            with self.assertRaisesRegex(ValueError, "physical patch"):
                restore_score_density_model(built.metadata, broken)

    def test_both_caps_forward_identical_for_same_token_inputs(self):
        # Same token inputs isolate network configuration from density effects.
        a = torch.zeros(1, 1, 800, 800)
        a[:, :, 250:451, 200:401] = 1
        b = torch.roll(a, 160, dims=3)
        p = torch.tensor([[[250., 200.], [250., 300.], [250., 400.], [350., 400.],
                           [450., 400.], [450., 300.], [450., 200.], [350., 200.]]])
        valid = torch.ones(1, 8, dtype=torch.bool)
        for architecture in ARCHITECTURES:
            small = build_score_density_model(512, architecture).model.eval()
            dense = build_score_density_model(1024, architecture).model.eval()
            with torch.no_grad():
                x = small(a, b, p, p + torch.tensor([0., 160.]), valid, valid)
                y = dense(a, b, p, p + torch.tensor([0., 160.]), valid, valid)
            for field in ("assignment", "affinity", "translation_hat_rc", "fused_logit", "local_logit", "coarse_logit"):
                self.assertTrue(torch.equal(getattr(x, field), getattr(y, field)), field)

    def test_factory_preserves_caller_rng(self):
        torch.manual_seed(93)
        before = torch.get_rng_state().clone()
        build_score_density_model(1024, "candidate_dual")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_density_cannot_silently_change_other_axes(self):
        built = build_score_density_model(1024)
        for field, value in (("coarse_size", 512), ("window_sizes_px", [7.])):
            broken = copy.deepcopy(built.metadata)
            broken["base_model_metadata"]["model_config"][field] = value
            with self.assertRaisesRegex(ValueError, "single-cap"):
                restore_score_density_model(broken, built.model.state_dict())
        with self.assertRaisesRegex(ValueError, "independent"):
            restore_score_density_model(dict(built.metadata, schema_version="rachel-score-input-model/1"), built.model.state_dict())

    def test_only_registered_caps_architectures_and_seeds(self):
        for cap in (True, 0, 513, 2048):
            with self.assertRaises(ValueError):
                build_score_density_model(cap)
        with self.assertRaises(ValueError):
            build_score_density_model(512, "staged")
        with self.assertRaises(ValueError):
            build_score_density_model(512, seed=True)


if __name__ == "__main__":
    unittest.main()
