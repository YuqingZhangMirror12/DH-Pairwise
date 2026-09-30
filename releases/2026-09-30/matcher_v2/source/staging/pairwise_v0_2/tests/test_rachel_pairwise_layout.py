"""Known correspondences test layout mathematics and the classifier boundary."""

from types import SimpleNamespace
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from staging.pairwise_v0_2.models.rachel_pairwise_layout import (
    RachelPairwiseLayout,
    decode_pair_layouts,
    load_frozen_pairwise_layout,
)
from staging.pairwise_v0_2.models.translation_layout import TranslationLayoutConfig


class AnalyticPairEvidence(nn.Module):
    """Small fixed evidence fixture; geometric decoding is the real decoder."""

    def __init__(self):
        super().__init__()
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.evidence = SimpleNamespace(
            assignment=torch.eye(4)[None].repeat(2, 1, 1),
            affinity=10.0 * torch.eye(4)[None].repeat(2, 1, 1),
            coarse_logit=torch.tensor([-8.0, 1.0]),
            coarse_probability=torch.sigmoid(torch.tensor([-8.0, 1.0])),
            local_logit=torch.tensor([-6.0, 2.0]),
            local_probability=torch.sigmoid(torch.tensor([-6.0, 2.0])),
            fused_logit=torch.tensor([-7.0, 3.0]),
            fused_probability=torch.sigmoid(torch.tensor([-7.0, 3.0])),
            translation_hat_rc=torch.tensor([[40.0, -60.0], [30.0, -50.0]]),
            translation_dispersion_px=torch.tensor([20.0, 25.0]),
        )

    def forward(self, *inputs):
        return self.evidence


class PrecisionPairEvidence(AnalyticPairEvidence):
    def forward(self, *inputs):
        output = SimpleNamespace(**vars(self.evidence))
        # Matrix multiplication, unlike a precomputed score tensor, exercises
        # actual CPU autocast and lets the factory's precision be verified.
        output.coarse_logit = (torch.ones((2, 2)) @ torch.ones((2, 2)))[:, 0]
        return output


def inputs():
    points = torch.tensor([[0.0, 0.0], [20.0, 0.0], [40.0, 7.0], [61.0, 18.0]])
    a = points[None].repeat(2, 1, 1)
    translation = torch.tensor([[3.0, -5.0], [-4.0, 8.0]])
    b = a + translation[:, None]
    valid = torch.ones((2, 4), dtype=torch.bool)
    masks = torch.zeros((2, 1, 2, 2))
    return (masks, masks, a, b, valid, valid), translation


class RachelPairwiseLayoutTest(unittest.TestCase):
    def test_layout_keeps_original_scores_and_dispersion_and_decodes_low_score(self):
        pair_model = AnalyticPairEvidence()
        model = RachelPairwiseLayout(pair_model)
        values, expected = inputs()
        output = model(*values)
        self.assertIs(output.pair_output, pair_model.evidence)
        for name in ("coarse_logit", "coarse_probability", "local_logit",
                     "local_probability", "fused_logit", "fused_probability"):
            self.assertIs(getattr(output, name), getattr(pair_model.evidence, name))
        self.assertLess(float(output.fused_probability[0]), 0.001)
        layout = output.layouts["translation_consensus"]
        self.assertTrue(layout.valid.all())
        self.assertTrue(layout.computed.all())
        torch.testing.assert_close(layout.t_a_to_b_rc, expected)
        torch.testing.assert_close(layout.offset_b_in_a_rc, -expected)
        torch.testing.assert_close(
            values[3] + layout.offset_b_in_a_rc[:, None], values[2]
        )
        torch.testing.assert_close(
            output.pair_output.translation_dispersion_px, torch.tensor([20.0, 25.0])
        )
        torch.testing.assert_close(
            output.pair_output.translation_hat_rc,
            torch.tensor([[40.0, -60.0], [30.0, -50.0]]),
        )
        self.assertEqual(layout.diagnostics[0]["inlier_count"], 4)

    def test_multiple_decoders_share_pair_output(self):
        pair_model = AnalyticPairEvidence()
        values, expected = inputs()
        output = decode_pair_layouts(
            pair_model.evidence, values[2], values[3], values[4], values[5],
            layout_configs={
                "consensus": TranslationLayoutConfig(),
                "median": TranslationLayoutConfig(decoder="median_cauchy"),
                "affinity": TranslationLayoutConfig(score_mode="dual_softmax"),
            },
        )
        self.assertEqual(set(output.layouts), {"consensus", "median", "affinity"})
        for layout in output.layouts.values():
            torch.testing.assert_close(layout.t_a_to_b_rc, expected)
        self.assertIs(output.pair_output, pair_model.evidence)

    def test_requested_mask_skips_only_explicit_samples(self):
        model = RachelPairwiseLayout(AnalyticPairEvidence())
        values, expected = inputs()
        output = model(*values, layout_mask=torch.tensor([True, False]))
        layout = output.layouts["translation_consensus"]
        self.assertEqual(layout.computed.tolist(), [True, False])
        self.assertEqual(layout.valid.tolist(), [True, False])
        torch.testing.assert_close(layout.t_a_to_b_rc[0], expected[0])
        self.assertTrue(torch.isnan(layout.t_a_to_b_rc[1]).all())
        self.assertIsNone(layout.results[1])
        self.assertEqual(layout.diagnostics[1]["reason"], "not_requested")

    def test_affinity_ablation_uses_affinity_instead_of_zero_transport(self):
        pair_model = AnalyticPairEvidence()
        pair_model.evidence.assignment.zero_()
        values, expected = inputs()
        output = decode_pair_layouts(
            pair_model.evidence, values[2], values[3], values[4], values[5],
            layout_configs={
                "transport": TranslationLayoutConfig(),
                "affinity": TranslationLayoutConfig(score_mode="dual_softmax"),
            },
        )
        self.assertFalse(output.layouts["transport"].valid.any())
        self.assertTrue(output.layouts["affinity"].valid.all())
        torch.testing.assert_close(output.layouts["affinity"].t_a_to_b_rc, expected)

    def test_disabled_layout_and_invalid_layout_preserve_pair_scores(self):
        model = RachelPairwiseLayout(AnalyticPairEvidence())
        values, _ = inputs()
        disabled = model(*values, compute_layout=False)
        self.assertEqual(disabled.layouts, {})
        invalid_points = list(values)
        invalid_points[4] = torch.zeros_like(values[4])
        invalid = model(*invalid_points)
        layout = invalid.layouts["translation_consensus"]
        self.assertFalse(layout.valid.any())
        self.assertTrue(layout.computed.all())
        self.assertTrue(torch.isnan(layout.t_a_to_b_rc).all())
        self.assertIs(disabled.fused_probability, invalid.fused_probability)

    def test_classifier_stays_frozen_if_wrapper_enters_training_mode(self):
        pair_model = AnalyticPairEvidence()
        model = RachelPairwiseLayout(pair_model).train()
        self.assertFalse(pair_model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        values, _ = inputs()
        output = model(*values)
        self.assertFalse(output.layouts["translation_consensus"].t_a_to_b_rc.requires_grad)

    def test_explicit_layout_mask_cannot_broadcast_across_pairs(self):
        model = RachelPairwiseLayout(AnalyticPairEvidence())
        values, _ = inputs()
        with self.assertRaisesRegex(ValueError, "layout_mask"):
            model(*values, layout_mask=torch.tensor([True]))


class FrozenPairwiseLayoutFactoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.freeze = Path(self.temporary.name) / "validation_freeze.json"
        self.authority = {
            "source_split": "validation", "sample_count": 3000,
            "probe_only": False, "test_or_real_used_for_fit": False,
            "selected_full_decoder": "selected", "checkpoint_sha256": "a" * 64,
            "original_fused_threshold": 0.75,
            "decoders": {
                "selected": asdict(TranslationLayoutConfig(inlier_radius_px=5.0)),
                "unselected": asdict(TranslationLayoutConfig()),
            },
        }

    def write_freeze(self):
        self.freeze.write_text(json.dumps(self.authority), encoding="utf-8")

    def loader_result(self):
        full = SimpleNamespace(
            arm="full_n512", checkpoint_sha256="a" * 64, epoch=47,
            threshold=SimpleNamespace(threshold=0.75), model=AnalyticPairEvidence(),
        )
        coarse = SimpleNamespace(arm="coarse_only", model=nn.Linear(1, 1))
        return ({"config": {"precision": "bf16"}}, "b" * 64, (coarse, full))

    def test_factory_retains_only_validation_selected_decoder_and_frozen_full(self):
        from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed

        self.write_freeze()
        restored = self.loader_result()
        with patch.object(sealed, "_freeze_completed_winners", return_value=restored) as loader:
            model = load_frozen_pairwise_layout(self.temporary.name, self.freeze)
        loader.assert_called_once_with(Path(self.temporary.name))
        self.assertIs(model.pair_model, restored[2][1].model)
        self.assertEqual(list(model.layout_configs), ["selected"])
        self.assertEqual(model.layout_configs["selected"].inlier_radius_px, 5.0)
        self.assertFalse(model.training)
        self.assertFalse(model.pair_model.training)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertEqual(model.deployment_metadata["checkpoint_sha256"], "a" * 64)
        self.assertEqual(model.inference_precision, "bf16")

    def test_factory_applies_receipt_precision_and_explicit_override(self):
        from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed

        self.write_freeze()
        restored = self.loader_result()
        restored[2][1].model = PrecisionPairEvidence()
        values, _ = inputs()
        with patch.object(sealed, "_freeze_completed_winners", return_value=restored):
            default = load_frozen_pairwise_layout(self.temporary.name, self.freeze)
            self.assertEqual(default(*values, compute_layout=False).coarse_logit.dtype,
                             torch.bfloat16)
            override = load_frozen_pairwise_layout(
                self.temporary.name, self.freeze, precision="fp32"
            )
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                self.assertEqual(override(*values, compute_layout=False).coarse_logit.dtype,
                                 torch.float32)
            self.assertEqual(override.deployment_metadata["checkpoint_precision"], "bf16")
            self.assertEqual(override.deployment_metadata["inference_precision"], "fp32")

    def test_factory_rejects_probe_or_nonvalidation_selection(self):
        changes = (
            ("probe_only", True), ("source_split", "test"),
            ("test_or_real_used_for_fit", True),
            ("selected_full_decoder", "missing"), ("sample_count", 0),
        )
        for key, value in changes:
            with self.subTest(key=key):
                previous = self.authority[key]
                self.authority[key] = value
                self.write_freeze()
                with self.assertRaises(ValueError):
                    load_frozen_pairwise_layout(self.temporary.name, self.freeze)
                self.authority[key] = previous

    def test_factory_rejects_changed_checkpoint_or_classification_threshold(self):
        from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed

        for key, value in (("checkpoint_sha256", "c" * 64),
                           ("original_fused_threshold", 0.5)):
            with self.subTest(key=key):
                previous = self.authority[key]
                self.authority[key] = value
                self.write_freeze()
                with patch.object(sealed, "_freeze_completed_winners", return_value=self.loader_result()):
                    with self.assertRaises(ValueError):
                        load_frozen_pairwise_layout(self.temporary.name, self.freeze)
                self.authority[key] = previous


if __name__ == "__main__":
    unittest.main()
