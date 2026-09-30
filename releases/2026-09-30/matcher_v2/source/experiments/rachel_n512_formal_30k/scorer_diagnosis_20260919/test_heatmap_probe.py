"""CPU tests only; these fixtures do not establish trained-model findings."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from staging.pairwise_v0_2.models.rachel_decoupled_score import (
    CrossAttentionPairHead, ThresholdedMatrixCNN, build_decoupled_score_model,
)
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.models.translation_layout import TranslationLayoutConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.heatmap_probe import (
    trace_head, probe_pair, perturb_tokens, _alignment, attach_targets, selected_batches,
)


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, depth):
        torch.manual_seed(32)
        head = CrossAttentionPairHead(16, 4, depth=depth).eval().requires_grad_(False)
        a, b = torch.randn(7, 16), torch.randn(5, 16)
        va, vb = torch.tensor([True, False, True, True, False, True, True]), torch.tensor([False, True, True, True, True])
        return head, a, b, va, vb

    def test_all_depths_exact_gradient_and_pool_contract(self):
        for depth in (1, 2, 4):
            head, a, b, va, vb = self.fixture(depth)
            before = deepcopy(head.state_dict())
            result, arrays = trace_head(head, a, b, va, vb)
            self.assertEqual(len(result["layers"]), depth + 1)
            self.assertLess(result["original_head_max_abs_error"], 2e-6)
            self.assertEqual(result["layers"][0]["a"]["token_indices"], [0, 2, 3, 5, 6])
            self.assertGreater(sum(result["layers"][0]["a"]["absolute"]), 0)
            for side in ("a", "b"):
                self.assertAlmostEqual(sum(result["pool"][side]["weights"]), 1, places=5)
                self.assertAlmostEqual(sum(result["pool"][side]["max_pool_channel_share"]), 1, places=5)
            self.assertEqual(arrays["decoder_1_a_to_b_attention_mean"].shape, (5, 4))
            np.testing.assert_allclose(arrays["decoder_1_a_to_b_attention_mean"].sum(-1), 1, atol=1e-6)
            for name, value in head.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))
            self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))
            # Head-input derivative in exact valid-index coordinates agrees
            # with central finite differences of the original unmodified head.
            eps, index, channel = 1e-3, 2, 7
            high, low = a.clone(), a.clone()
            high[index, channel] += eps
            low[index, channel] -= eps
            with torch.no_grad():
                derivative = (head(high[None], b[None], va[None], vb[None]) -
                              head(low[None], b[None], va[None], vb[None])).item() / (2 * eps)
            self.assertAlmostEqual(float(arrays["context_input_a_gradients"][1, channel]), derivative, delta=2e-4)

    def test_padding_does_not_change_logit_or_attribution(self):
        head, a, b, va, vb = self.fixture(2)
        first, _ = trace_head(head, a, b, va, vb)
        a[~va] = 1e8
        b[~vb] = -1e8
        second, _ = trace_head(head, a, b, va, vb)
        self.assertEqual(first["raw_head_logit"], second["raw_head_logit"])
        self.assertEqual(first["layers"], second["layers"])

    def test_trace_overrides_outer_no_grad_and_inference_contexts(self):
        head, a, b, va, vb = self.fixture(4)
        expected, _ = trace_head(head, a, b, va, vb)
        with torch.no_grad():
            no_grad_result, _ = trace_head(head, a, b, va, vb)
        with torch.inference_mode():
            # This matches the captured context tensors from the official
            # evaluator: inference tensors cannot themselves require gradients.
            inference_a, inference_b = a.clone(), b.clone()
            inference_result, arrays = trace_head(head, inference_a, inference_b, va, vb)
        self.assertEqual(expected["layers"], no_grad_result["layers"])
        self.assertEqual(expected["layers"], inference_result["layers"])
        self.assertGreater(float(np.abs(arrays["context_input_a_gradients"]).sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))

    def test_signed_and_absolute_summaries_retain_channel_cancellation(self):
        head, a, b, va, vb = self.fixture(2)
        result, arrays = trace_head(head, a, b, va, vb)
        cancellations = []
        for layer in result["layers"]:
            for side in ("a", "b"):
                prefix = layer["name"] + "_" + side
                product = arrays[prefix + "_features"] * arrays[prefix + "_gradients"]
                np.testing.assert_allclose(layer[side]["signed"], product.sum(-1), atol=1e-7)
                np.testing.assert_allclose(layer[side]["absolute"], np.abs(product).sum(-1), atol=1e-7)
                cancellations.extend(np.abs(product).sum(-1) - np.abs(product.sum(-1)))
        self.assertGreater(max(cancellations), 1e-5)

    def test_zero_and_mask_are_explicit_distinct_interventions(self):
        head, a, b, va, vb = self.fixture(1)
        before = a.clone()
        baseline, _ = trace_head(head, a, b, va, vb)
        empty = perturb_tokens(head, a, b, va, vb, [], [], "zero")
        self.assertAlmostEqual(empty["raw_head_logit"], baseline["raw_head_logit"])
        zero = perturb_tokens(head, a, b, va, vb, [0, 2], [1], "zero")
        mask = perturb_tokens(head, a, b, va, vb, [0, 2], [1], "mask")
        self.assertNotEqual(zero["raw_head_logit"], mask["raw_head_logit"])
        self.assertTrue(torch.equal(a, before))
        with self.assertRaisesRegex(ValueError, "indices"):
            perturb_tokens(head, a, b, va, vb, [1], [], "mask")

    def test_rejects_unknown_head_training_mode_and_empty_evidence(self):
        head, a, b, va, vb = self.fixture(1)
        with self.assertRaisesRegex(ValueError, "registered"):
            trace_head(ThresholdedMatrixCNN(), a, b, va, vb)
        with self.assertRaisesRegex(ValueError, "eval"):
            trace_head(head.train(), a, b, va, vb)
        with self.assertRaisesRegex(ValueError, "empty contour"):
            trace_head(head.eval(), a, b, va * False, vb)

    def test_full_model_uses_original_base_once_and_remains_frozen(self):
        torch.manual_seed(27)
        config = RachelN512Config(canvas_size=80, coarse_size=32, contour_cap=16, patch_size=8,
            feature_dim=16, num_heads=4, landmark_count=4, context_layers=1, evidence_dim=8,
            sinkhorn_iterations=40, activation_checkpointing=False)
        model = build_decoupled_score_model(config, "cross_attention", phase="classifier",
            model_options={"cross_attention_depth": 2}).eval().requires_grad_(False)
        a, b = torch.zeros(1, 1, 80, 80), torch.zeros(1, 1, 80, 80)
        a[:, :, 15:57, 12:43] = 1
        b[:, :, 10:61, 28:63] = 1
        pa = torch.tensor([[[15., 12.], [15., 27.], [15., 42.], [35., 42.],
                            [56., 42.], [56., 27.], [56., 12.], [35., 12.]]])
        pb = torch.tensor([[[10., 28.], [10., 45.], [10., 62.], [35., 62.],
                            [60., 62.], [60., 45.], [60., 28.], [35., 28.]]])
        # Non-contiguous invalid padding checks original-index layout mapping.
        # It is deliberately not equivalent to dropping every token after Nvalid.
        va, vb = torch.ones(1, 8, dtype=torch.bool), torch.ones(1, 8, dtype=torch.bool)
        va[0, 1] = False
        vb[0, 5] = False
        tensors = [a, b, pa, pb, va, vb]
        count = []
        hook = model.base_model.register_forward_hook(lambda *args: count.append(1))
        try:
            result, arrays = probe_pair(model, tensors,
                decoder_config=TranslationLayoutConfig(correspondence_mode="topk_union"))
        finally:
            hook.remove()
        self.assertEqual(len(count), 1)
        self.assertEqual(len(result["perturbations"]), 10)
        self.assertLess(result["wrapper_max_abs_error"], 2e-6)
        self.assertEqual(arrays["sinkhorn_assignment"].shape, (8, 8))
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(result["layout"]["minimum_arc_length_gate_present"])
        # Compare to the actual public evaluator, not another rewritten decoder.
        from experiments.rachel_n512_formal_30k import evaluate_score_design as official
        batch = SimpleNamespace(pair_ids=("fixture",), fragment_a_tokens=("a",), fragment_b_tokens=("b",),
            **{name: value.numpy() for name, value in zip(official.FIELDS, tensors)})
        reference = official.predict_batch(model, batch, torch.device("cpu"))[0]
        self.assertAlmostEqual(result["score"], reference["classification"]["fused"], places=7)
        self.assertEqual(result["decision_valid"], reference["decision_valid"])
        self.assertEqual(result["attribution_is_valid_decision"], reference["decision_valid"])
        np.testing.assert_allclose(result["layout"]["t_a_to_b_rc"],
            reference["layouts"][official.fixed.DECODER_NAME]["translation_rc"], atol=1e-7)
        np.testing.assert_allclose(result["layout"]["offset_b_in_a_rc"],
            -np.asarray(result["layout"]["t_a_to_b_rc"]), atol=0)
        candidates = np.asarray(result["layout"]["candidate_indices"])
        inliers = candidates[np.asarray(result["layout"]["inlier_mask"], dtype=bool)]
        self.assertEqual(result["layout"]["inlier_token_indices_a"], np.unique(inliers[:, 0]).tolist())
        self.assertEqual(result["layout"]["inlier_token_indices_b"], np.unique(inliers[:, 1]).tolist())
        self.assertNotIn(1, candidates[:, 0])
        self.assertNotIn(5, candidates[:, 1])
        # All JSON diagnostics and compressed arrays round-trip without NaN JSON.
        reloaded = json.loads(json.dumps(result, allow_nan=False))
        self.assertEqual(reloaded["layers"], result["layers"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "probe.npz"
            np.savez_compressed(path, **arrays)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["points_rc_a"], pa[0].numpy())
                np.testing.assert_array_equal(archive["mask_a"], a[0, 0].numpy())

    def test_alignment_uses_original_indices_not_compact_positions(self):
        values = dict(token_indices=[1, 4, 8], absolute=[1., 2., 7.], signed=[-1., 2., 7.])
        result = _alignment([dict(name="input", a=values, b=values)], [8], [1])[0]
        self.assertAlmostEqual(result["a"]["absolute_attribution_share_on_inliers"], .7)
        self.assertAlmostEqual(result["a"]["enrichment_over_uniform"], 2.1)
        self.assertAlmostEqual(result["b"]["signed_sum_on_inliers"], -1.)

    def test_attach_test_and_ood_targets_after_prediction(self):
        rows = [dict(dataset="test", pair_id="test-positive", _test_label=True, _test_target=[3., 4.],
                     layout=dict(valid=True, t_a_to_b_rc=[6., 8.])),
                dict(dataset="test", pair_id="test-negative", _test_label=False, _test_target=None,
                     layout=dict(valid=False, t_a_to_b_rc=[None, None])),
                dict(dataset="ood", pair_id="ood-positive", layout=dict(valid=True, t_a_to_b_rc=[20., 1.]))]
        attach_targets(rows, SimpleNamespace())
        self.assertEqual(rows[0]["layout"]["translation_l2_px"], 5.)
        self.assertTrue(rows[0]["layout_gt_available"])
        self.assertFalse(rows[1]["layout_gt_available"])
        self.assertIsNone(rows[2]["layout"]["translation_l2_px"])
        self.assertIsNone(rows[2]["target_translation_rc"])
        self.assertTrue(rows[2]["label"])
        self.assertTrue(all(not any(key.startswith("_test_") for key in row) for row in rows))
        json.dumps(rows, allow_nan=False)

    def test_attach_real_gt_checks_fragment_order_and_layout_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"pairs": [dict(pair_id="p", label=True, case_cluster="case"),
                                   dict(pair_id="n", label=False, case_cluster="negative")]}
            gt = {"positive_pairs": [dict(pair_id="p", fragment_a_token="a", fragment_b_token="b",
                                           translation_gt_a_to_b_rc=[2., 3.])]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "gt.json").write_text(json.dumps(gt))
            args = SimpleNamespace(prepared_cache=directory, translation_gt_json=str(root / "gt.json"))
            rows = [dict(dataset="real", pair_id="p", fragment_a="a", fragment_b="b",
                         layout=dict(valid=True, t_a_to_b_rc=[2., 3.])),
                    dict(dataset="real", pair_id="n", fragment_a="a", fragment_b="c",
                         layout=dict(valid=True, t_a_to_b_rc=[0., 0.]))]
            attach_targets(rows, args)
            self.assertEqual(rows[0]["layout"]["translation_l2_px"], 0.)
            self.assertIsNone(rows[1]["target_translation_rc"])
            rows[0]["fragment_a"] = "wrong"
            with self.assertRaisesRegex(ValueError, "endpoints"):
                attach_targets(rows, args)

    def test_selected_test_loader_reads_only_requested_indices(self):
        from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluation
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pairs").mkdir()
            (root / "pairs/test.jsonl").write_text('\n'.join(json.dumps({"pair_id": "p%d" % i}) for i in range(5)))
            args = SimpleNamespace(dataset=directory)
            selection = [dict(dataset="test", pair_id="p3"), dict(dataset="test", pair_id="p1")]
            identity = dict(sampling="original512", seed=12)
            batches = [SimpleNamespace(pair_ids=("p3",)), SimpleNamespace(pair_ids=("p1",))]
            with patch.object(evaluation.core, "RachelPairDataset", return_value="test-fixture"), \
                    patch.object(evaluation.core, "make_ablation_loader", return_value=batches) as loader:
                actual = list(selected_batches(args, selection, identity))
            self.assertEqual(loader.call_args.args[1], [3, 1])
            self.assertEqual([row[1].pair_ids[0] for row in actual], ["p3", "p1"])
            with self.assertRaisesRegex(ValueError, "absent"):
                list(selected_batches(args, [dict(dataset="test", pair_id="missing")], identity))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable; this does not claim GPU numerical validation")
    def test_cuda_fp32_attribution_matches_original_head(self):
        for depth in (1, 2, 4):
            head, a, b, va, vb = self.fixture(depth)
            head = head.cuda()
            result, _ = trace_head(head, a.cuda(), b.cuda(), va.cuda(), vb.cuda())
            self.assertLess(result["original_head_max_abs_error"], 2e-5)


if __name__ == "__main__":
    unittest.main()
