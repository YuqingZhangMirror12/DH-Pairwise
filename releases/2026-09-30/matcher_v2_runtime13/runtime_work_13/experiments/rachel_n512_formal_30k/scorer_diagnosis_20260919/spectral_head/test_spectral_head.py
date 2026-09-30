"""CPU synthetic tests only; no real-data fitting, model training job or network."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from staging.pairwise_v0_2.models.rachel_decoupled_score import CrossAttentionPairHead
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_diagnostics import spectral_metrics
from .features import (FEATURE_NAMES, SOURCE_HASH_FIELDS, compute_summary, fit_train_statistics,
                       load_cache, make_record, model_input_sha256, write_cache)
from .model import CASpectralResidualScorer, FrozenTrainStandardizer, tensor_summary_batch


def identity(split="train"):
    return dict(split=split, fixed_physical_inputs=True, **{key: "1" * 64 for key in SOURCE_HASH_FIELDS})


def record(pair_id="p0", multiplier=1.):
    return make_record(pair_id, "2" * 64, multiplier * np.arange(1, 13).reshape(3, 4) / 25,
                       np.ones(3, bool), np.ones(4, bool))


def cached(split="train"):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "summary.json"
        digest = write_cache(path, identity(split), [record(), record("p1", 2.)])
        return load_cache(path, identity(split), digest)


def statistics():
    return fit_train_statistics(cached())


def make_head(variant="mass_spectral", train_ca=True):
    torch.manual_seed(411)
    source = CrossAttentionPairHead(16, 4, depth=2).eval().requires_grad_(False)
    head = CASpectralResidualScorer(source, statistics(), variant=variant, residual_seed=91,
        source_checkpoint_sha256="3" * 64, train_ca=train_ca)
    return source, head


def inputs():
    torch.manual_seed(73)
    a, b = torch.randn(2, 5, 16), torch.randn(2, 6, 16)
    va = torch.tensor([[True, False, True, True, False], [True, True, True, False, False]])
    vb = torch.tensor([[True, True, False, True, False, True], [True, True, True, True, False, False]])
    batch = cached().batch(["p0", "p1"], ["2" * 64, "2" * 64])
    return a, b, va, vb, tensor_summary_batch(batch, "cpu")


class SpectralHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_feature_definitions_match_completed_diagnostics(self):
        matrix = np.array([[.2, 0, .1, .3], [0, .4, .1, 0], [.3, .1, 0, 0]])
        actual = compute_summary(matrix, np.ones(3, bool), np.ones(4, bool), matrix_kind="real_transport")
        ref = spectral_metrics(matrix)
        per_axis = np.log1p([matrix.sum() / 3, matrix.sum() / 4])
        shape = ref["spectral_shape"]
        expected = [np.log1p(matrix.sum()), per_axis.mean(), abs(per_axis[0] - per_axis[1]),
            shape["sigma1_over_frobenius"], *[shape["topk_energy_fraction"][str(k)] for k in (2, 4, 8, 16)],
            shape["effective_rank_entropy_singular"] / 3, shape["participation_ratio_energy"] / 3]
        np.testing.assert_allclose(actual.values, expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(len(FEATURE_NAMES), 10)

    def test_rectangular_padding_permutation_and_exchange_invariance(self):
        rng = np.random.default_rng(8)
        matrix = rng.random((7, 9))
        va = np.array([1, 0, 1, 1, 0, 1, 1], bool)
        vb = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1], bool)
        get = lambda p, a, b: np.asarray(compute_summary(p, a, b, matrix_kind="real_transport").values)
        expected = get(matrix, va, vb)
        ia, ib = rng.permutation(7), rng.permutation(9)
        np.testing.assert_allclose(get(matrix[np.ix_(ia, ib)], va[ia], vb[ib]), expected, atol=1e-12)
        np.testing.assert_allclose(get(matrix.T, vb, va), expected, atol=1e-12)
        altered = matrix.copy()
        altered[~(va[:, None] & vb)] = 100000.
        np.testing.assert_allclose(get(altered, va, vb), expected, atol=1e-12)
        scaled = get(17. * matrix, va, vb)
        np.testing.assert_allclose(scaled[3:], expected[3:], atol=1e-12)
        self.assertNotAlmostEqual(scaled[0], expected[0])

    def test_spectrum_is_not_ordered_seam_evidence(self):
        ordered = np.eye(8)
        scrambled = ordered[:, [0, 4, 1, 5, 2, 6, 3, 7]]
        get = lambda p: compute_summary(p, np.ones(8, bool), np.ones(8, bool), matrix_kind="real_transport").values
        np.testing.assert_allclose(get(ordered), get(scrambled), atol=1e-12)
        self.assertFalse(np.array_equal(ordered.argmax(1), scrambled.argmax(1)))

    def test_zero_empty_rectangular_and_dustbin_contract(self):
        get = lambda p, a, b: compute_summary(p, np.asarray(a), np.asarray(b), matrix_kind="real_transport")
        zero = get(np.zeros((3, 4)), np.ones(3, bool), np.ones(4, bool))
        self.assertTrue(zero.valid)
        self.assertEqual(zero.values, (0.,) * 10)
        empty = get(np.ones((3, 4)), np.zeros(3, bool), np.ones(4, bool))
        self.assertFalse(empty.valid)
        self.assertEqual(empty.values, (0.,) * 10)
        rank_one = get(np.ones((3, 5)), np.ones(3, bool), np.ones(5, bool))
        np.testing.assert_allclose(rank_one.values[3:8], 1., atol=1e-12)
        np.testing.assert_allclose(rank_one.values[8:], 1. / 3, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "dustbin"):
            compute_summary(np.eye(3), np.ones(3, bool), np.ones(3, bool), matrix_kind="full_with_dustbin")
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            get(np.ones((4, 5)), np.ones(3, bool), np.ones(4, bool))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            get(-np.ones((3, 4)), np.ones(3, bool), np.ones(4, bool))

    def test_cache_hash_input_source_and_order_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            digest = write_cache(path, identity(), [record(), record("p1", 2)])
            with self.assertRaises(FileExistsError):
                write_cache(path, identity(), [record()])
            cache = load_cache(path, identity(), digest)
            batch = cache.batch(["p1", "p0"], ["2" * 64] * 2)
            self.assertGreater(batch["values"][0, 0], batch["values"][1, 0])
            with self.assertRaisesRegex(ValueError, "physical input changed"):
                cache.batch(["p0"], ["4" * 64])
            changed_identity = {**identity(), "matcher_state_sha256": "4" * 64}
            with self.assertRaisesRegex(ValueError, "identity"):
                load_cache(path, changed_identity, digest)
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "file hash"):
                load_cache(path, identity(), digest)

    def test_training_statistics_exclude_all_held_out_fitting(self):
        stats = statistics()
        self.assertEqual(stats["fit_split"], "train")
        self.assertEqual(stats["fitted_count"], 2)
        self.assertEqual(stats["scale"][3], 1.)  # scalar-scale invariant shape feature
        for split in ("val", "test", "real", "ood"):
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                fit_train_statistics(cached(split))
        val_cache = cached("val")
        val_cache.identity["split"] = "train"
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            fit_train_statistics(val_cache)
        tampered = {**stats, "fit_split": "real"}
        with self.assertRaisesRegex(ValueError, "TRAIN-only"):
            FrozenTrainStandardizer(tampered)

    def test_controls_share_capacity_initialization_and_exact_ca_output(self):
        tensor_inputs = inputs()
        outputs, residual_states = [], []
        for variant in ("zero", "mass", "mass_spectral"):
            source, head = make_head(variant)
            before = deepcopy(source.state_dict())
            out = head(*tensor_inputs)
            with torch.no_grad():
                expected = source(*tensor_inputs[:4])
            self.assertTrue(torch.equal(out.logit, expected))
            self.assertTrue(torch.equal(out.residual_logit, torch.zeros_like(expected)))
            self.assertEqual(sum(p.numel() for p in head.residual.parameters()), 193)
            self.assertTrue(all(torch.equal(value, before[key]) for key, value in source.state_dict().items()))
            self.assertTrue(all(not p.requires_grad for p in source.parameters()))
            if variant == "zero":
                self.assertEqual(int(torch.count_nonzero(out.branch_input)), 0)
            if variant == "mass":
                self.assertEqual(int(torch.count_nonzero(out.branch_input[:, 3:])), 0)
            outputs.append(out)
            residual_states.append(deepcopy(head.residual.state_dict()))
        for state in residual_states[1:]:
            self.assertTrue(all(torch.equal(value, residual_states[0][key]) for key, value in state.items()))

    def test_nonzero_residual_exchange_symmetry_and_empty_gating(self):
        _, head = make_head()
        a, b, va, vb, summary = inputs()
        with torch.no_grad():
            head.residual[-1].weight.fill_(.05)
            head.residual[-1].bias.fill_(.3)
        left = head(a, b, va, vb, summary)
        right = head(b, a, vb, va, {**summary, "n_a": summary["n_b"], "n_b": summary["n_a"]})
        torch.testing.assert_close(left.logit, right.logit, atol=1e-6, rtol=1e-6)
        va = va.clone()
        va[0] = False
        summary = {key: value.clone() for key, value in summary.items()}
        summary["values"][0] = 0
        summary["valid"][0] = False
        summary["n_a"][0] = 0
        empty = head(a, b, va, vb, summary)
        self.assertEqual(float(empty.residual_logit[0]), 0.)
        self.assertEqual(float(empty.logit[0]), float(head.ca.no_evidence_logit))

    def test_pair_bce_learns_residual_without_svd_or_matcher_gradients(self):
        source, head = make_head()
        tensor_inputs = inputs()
        source_before = deepcopy(source.state_dict())
        optimizer = torch.optim.SGD(head.parameters(), lr=.05)
        targets = torch.ones(2)
        with patch("numpy.linalg.svd", side_effect=AssertionError("SVD must not run in a training step")):
            cache = cached_without_svd_fixture()
            cache.batch(["p0"], ["2" * 64])
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(head(*tensor_inputs).logit, targets)
                loss.backward()
                if step == 0:
                    self.assertGreater(float(head.residual[-1].weight.grad.abs().sum()), 0.)
                    self.assertEqual(float(head.residual[0].weight.grad.abs().sum()), 0.)
                else:
                    self.assertGreater(float(head.residual[0].weight.grad.abs().sum()), 0.)
                optimizer.step()
        self.assertTrue(all(torch.equal(value, source_before[key]) for key, value in source.state_dict().items()))
        self.assertTrue(all(value.grad is None for value in tensor_inputs[:2]))

    def test_source_depth_frozen_context_dimension_and_checkpoint_guards(self):
        source, head = make_head()
        a, b, va, vb, summary = inputs()
        with self.assertRaisesRegex(ValueError, "frozen"):
            head(a.requires_grad_(True), b, va, vb, summary)
        a = a.detach()
        with self.assertRaisesRegex(ValueError, "effective dimensions"):
            head(a, b, va, vb, {**summary, "n_a": summary["n_a"] + 1})
        with self.assertRaisesRegex(ValueError, "depth-two"):
            CASpectralResidualScorer(CrossAttentionPairHead(16, 4, 1), statistics(), variant="mass",
                residual_seed=91, source_checkpoint_sha256="3" * 64)
        _, same = make_head()
        same.load_state_dict(head.state_dict(), strict=True)
        _, wrong_arm = make_head("mass")
        with self.assertRaisesRegex(ValueError, "configuration"):
            wrong_arm.load_state_dict(head.state_dict(), strict=True)

    def test_residual_initialization_preserves_rng_and_optional_frozen_ca(self):
        source = CrossAttentionPairHead(16, 4, 2).eval()
        stats = statistics()
        torch.manual_seed(132)
        before = torch.get_rng_state().clone()
        head = CASpectralResidualScorer(source, stats, variant="mass", residual_seed=77,
            source_checkpoint_sha256="3" * 64, train_ca=False)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        head.train()
        self.assertFalse(head.ca.training)
        self.assertTrue(all(not p.requires_grad for p in head.ca.parameters()))
        self.assertTrue(all(p.requires_grad for p in head.residual.parameters()))
        self.assertFalse(any(p.requires_grad for p in head.standardizer.parameters()))

    def test_input_digest_binds_masks_points_validity_and_dtypes(self):
        arrays = dict(mask_a=np.zeros((1, 8, 8), np.float32), mask_b=np.ones((1, 8, 8), np.float32),
            points_rc_a=np.zeros((4, 2), np.float32), points_rc_b=np.ones((4, 2), np.float32),
            contour_valid_a=np.ones(4, bool), contour_valid_b=np.ones(4, bool))
        original = model_input_sha256(arrays)
        for key in arrays:
            changed = {k: v.copy() for k, v in arrays.items()}
            changed[key].flat[0] = not changed[key].flat[0] if changed[key].dtype == bool else 7
            self.assertNotEqual(model_input_sha256(changed), original)
        self.assertNotEqual(model_input_sha256({**arrays, "mask_a": arrays["mask_a"].astype(np.float64)}), original)

    def test_wrapper_preserves_layout_assignment_and_original_validity_gate(self):
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.test_pixel_attribution import PixelTests
        model, tensor_inputs = PixelTests().fixture(depth=2)
        with torch.no_grad():
            original = model.base_model(*tensor_inputs)
        va, vb = tensor_inputs[4:]
        summary = compute_summary(original.assignment[0].numpy(), va[0].numpy(), vb[0].numpy(), matrix_kind="real_transport")
        batch = tensor_summary_batch(dict(values=[summary.values], valid=[summary.valid],
            n_a=[summary.n_a], n_b=[summary.n_b]), "cpu")
        head = CASpectralResidualScorer(model.score_head, statistics(), variant="mass_spectral",
            residual_seed=91, source_checkpoint_sha256="3" * 64)
        output, detail = head.apply_to_frozen_output(original, va, vb, batch)
        self.assertIs(output.assignment, original.assignment)
        self.assertIs(output.translation_hat_rc, original.translation_hat_rc)
        self.assertIs(output.translation_hat_xy_cartesian, original.translation_hat_xy_cartesian)
        self.assertIs(output.transport, original.transport)
        invalid = replace(original, training_valid=torch.zeros_like(original.training_valid))
        invalid_output, _ = head.apply_to_frozen_output(invalid, va, vb, batch)
        self.assertEqual(float(invalid_output.fused_logit[0]), 0.)
        self.assertFalse(bool(invalid_output.decision_valid[0]))


def cached_without_svd_fixture():
    # A previously persisted record need not recompute a decomposition at load.
    from .features import SummaryRecord
    row = SummaryRecord("p0", "2" * 64, "5" * 64, (1., .3, .1, .8, 1., 1., 1., 1., .5, .4), 3, 4, True)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "frozen.json"
        digest = write_cache(path, identity(), [row])
        return load_cache(path, identity(), digest)


if __name__ == "__main__":
    unittest.main()
