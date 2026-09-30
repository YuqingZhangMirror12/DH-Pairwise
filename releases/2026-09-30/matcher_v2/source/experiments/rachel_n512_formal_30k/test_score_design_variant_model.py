"""CPU model-kind/initialization tests; no dataset, GPU or training writes."""
import copy
import unittest

import torch

from experiments.rachel_n512_formal_30k.score_design_input_variants import InputVariantSpec, full24_reference_config
from experiments.rachel_n512_formal_30k.score_design_variant_model import build_score_input_model, restore_score_input_model
from staging.pairwise_v0_2.models.rachel_candidate_score import RachelCandidateScore
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Pairwise
from staging.pairwise_v0_2.models.rachel_multiscale_transport import RachelMultiscaleTransportPairwise


class ScoreInputModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_state_equal(self, a, b):
        self.assertEqual(set(a), set(b))
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]), name)

    def test_default_wrapper_is_identical_to_live_random_candidate_initialization(self):
        torch.manual_seed(260913)
        live = RachelCandidateScore(RachelN512Pairwise(full24_reference_config()), architecture="candidate_dual")
        current = build_score_input_model(architecture="candidate_dual")
        self.assert_state_equal(live.state_dict(), current.model.state_dict())

    def test_all_variants_share_the_identical_p_and_r_head(self):
        reference = build_score_input_model()
        for spec in (InputVariantSpec(coarse_size=256), InputVariantSpec(window_sizes_px=(7.,)),
                     InputVariantSpec(transport_fusion="post")):
            variant = build_score_input_model(spec)
            self.assert_state_equal(reference.model.score_head.state_dict(), variant.model.score_head.state_dict())
            self.assert_state_equal(reference.model.base_model.fusion.state_dict(), variant.model.base_model.fusion.state_dict())

    def test_post_wrapper_trained_weights_restore_as_post_not_full(self):
        original = build_score_input_model(InputVariantSpec(transport_fusion="post"), architecture="candidate_dual")
        with torch.no_grad():
            original.model.base_model.scale_logits.add_(torch.arange(4) / 10.)
            original.model.score_head.correctness_head.bias.add_(.25)
        restored = restore_score_input_model(original.metadata, original.model.state_dict())
        self.assertIsInstance(restored.model.base_model, RachelMultiscaleTransportPairwise)
        self.assert_state_equal(original.model.state_dict(), restored.model.state_dict())

    def test_post_wrapper_forward_and_restore_preserve_assignment_and_layout(self):
        built = build_score_input_model(InputVariantSpec(transport_fusion="post"))
        restored = restore_score_input_model(built.metadata, built.model.state_dict())
        built.model.eval(); restored.model.eval()
        ma, mb = torch.zeros(1, 1, 800, 800), torch.zeros(1, 1, 800, 800)
        ma[:, :, 210:491, 200:441] = 1; mb[:, :, 240:521, 400:641] = 1
        pa = torch.tensor([[[210., 200.], [210., 320.], [210., 440.], [350., 440.],
                            [490., 440.], [490., 320.], [490., 200.], [350., 200.]]])
        valid = torch.ones(1, 8, dtype=torch.bool)
        inputs = ma, mb, pa, pa + torch.tensor([30., 200.]), valid, valid
        with torch.no_grad():
            base = built.model.base_model(*inputs)
            wrapped = built.model(*inputs)
            reloaded = restored.model(*inputs)
        for name in ("assignment", "affinity", "translation_hat_rc"):
            self.assertTrue(torch.equal(getattr(base, name), getattr(wrapped, name)), name)
            self.assertTrue(torch.equal(getattr(wrapped, name), getattr(reloaded, name)), name)
        self.assertTrue(torch.equal(wrapped.fused_logit, reloaded.fused_logit))
        self.assertTrue(torch.isfinite(wrapped.fused_logit).all())

    def test_wrong_base_kind_legacy_schema_and_grid_are_rejected(self):
        built = build_score_input_model(InputVariantSpec(transport_fusion="post"))
        broken = copy.deepcopy(built.metadata)
        broken["base_model_metadata"]["model_kind"] = "full"
        with self.assertRaisesRegex(ValueError, "kind/config/options"):
            restore_score_input_model(broken, built.model.state_dict())
        broken = dict(built.metadata, schema_version="rachel-score-design-training/1")
        with self.assertRaisesRegex(ValueError, "independent"):
            restore_score_input_model(broken, built.model.state_dict())
        state = dict(built.model.state_dict())
        state["base_model.patch_sampler.offsets_rc"] = state["base_model.patch_sampler.offsets_rc"] * 2
        with self.assertRaisesRegex(ValueError, "sampling offsets"):
            restore_score_input_model(built.metadata, state)

    def test_original_and_single_window_restore_without_wrapper(self):
        built = build_score_input_model(InputVariantSpec(window_sizes_px=(32.,)), architecture="original")
        restored = restore_score_input_model(built.metadata, built.model.state_dict())
        self.assertIs(type(restored.model), RachelN512Pairwise)
        self.assert_state_equal(built.model.state_dict(), restored.model.state_dict())


if __name__ == "__main__":
    unittest.main()
