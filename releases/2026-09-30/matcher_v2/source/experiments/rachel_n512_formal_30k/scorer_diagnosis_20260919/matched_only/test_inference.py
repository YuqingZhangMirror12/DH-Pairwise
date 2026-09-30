"""Original base -> cached head and online adapter parity on CPU tiny fixtures."""
from copy import deepcopy
from dataclasses import fields, replace
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config, RachelN512Output, RachelN512Pairwise
from experiments.rachel_n512_formal_30k.test_score_design_stages import inputs
from . import model
from .inference import (
    FrozenMatchedInference, MatchedInferenceOutput, REPLACED_FIELDS, online_stage_groups,
)
from . import stage_cache
from .train import CandidateStageScorer


def base():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        return RachelN512Pairwise(RachelN512Config(canvas_size=80, coarse_size=32,
            contour_cap=16, patch_size=8, feature_dim=16, num_heads=4,
            landmark_count=4, context_layers=1, evidence_dim=8,
            sinkhorn_iterations=40, activation_checkpointing=False)).eval()


def head(arm):
    return model.make_fresh_scorer(arm, seed=42, feature_dim=16, num_heads=4).eval()


def cache_style(output, args, scorer):
    with torch.inference_mode():
        selection = model.select_predicted_inliers(output.assignment, args[2], args[3], args[4], args[5])
        weights = torch.zeros_like(selection.candidate_valid, dtype=output.assignment.dtype)
        for row in range(len(args[0])):
            present = selection.candidate_valid[row]
            edges = selection.candidate_indices[row, present]
            weights[row, present] = output.assignment[row, edges[:, 0], edges[:, 1]]
        result = scorer(output.token_features_a.clone(), output.token_features_b.clone(),
            args[4], args[5], selection, candidate_weights=weights,
            points_a_rc=args[2], points_b_rc=args[3])
        deployed = torch.where(output.training_valid, result.logit, torch.zeros_like(result.logit))
    return result, deployed, selection, weights


class GeometryCacheFixture:
    """Only geometry fields; passing labels/GT is deliberately impossible here."""
    def __init__(self, root, args, selection, weights):
        self.root, self.split = Path(root), "unit_fixture"
        self.binding = {"synthetic_fixture_only": True}
        values = dict(points_a=args[2], points_b=args[3], valid_a=args[4], valid_b=args[5],
            candidate_indices=selection.candidate_indices, candidate_valid=selection.candidate_valid,
            candidate_weights=weights, candidate_inliers=selection.candidate_inliers,
            translation_a_to_b_rc=selection.translation_a_to_b_rc, layout_valid=selection.layout_valid)
        self.arrays = {k: v.detach().cpu().numpy() for k, v in values.items()}

    def __len__(self):
        return len(self.arrays["points_a"])


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cached_head_online_adapter_parity_all_three_arms(self):
        args = inputs()
        for arm in model.ARMS:
            teacher, scorer = base(), head(arm)
            with torch.inference_mode(): original = teacher(*args)
            expected, deployed, selection, weights = cache_style(original, args, scorer)
            adapter = FrozenMatchedInference(teacher, scorer)
            got = adapter(*args)
            self.assertIsInstance(got, RachelN512Output)
            self.assertIsInstance(got, MatchedInferenceOutput)
            torch.testing.assert_close(got.fused_logit, deployed, rtol=0, atol=0)
            torch.testing.assert_close(got.local_logit, deployed, rtol=0, atol=0)
            torch.testing.assert_close(got.fused_probability, deployed.sigmoid(), rtol=0, atol=0)
            torch.testing.assert_close(got.score_details["raw_head_logit"], expected.logit, rtol=0, atol=0)
            torch.testing.assert_close(got.assignment, original.assignment, rtol=0, atol=0)
            torch.testing.assert_close(got.token_features_a, original.token_features_a, rtol=0, atol=0)
            torch.testing.assert_close(got.token_features_b, original.token_features_b, rtol=0, atol=0)
            torch.testing.assert_close(got.score_details["candidate_weights"], weights, rtol=0, atol=0)
            self.assertTrue(torch.equal(got.score_details["selection"].candidate_inliers, selection.candidate_inliers))

    def test_every_nonclassifier_field_is_original_object(self):
        teacher, args = base(), inputs()
        with torch.inference_mode(): original = teacher(*args)
        adapter = FrozenMatchedInference(teacher, head("matched_edges"))
        with patch.object(teacher, "forward", return_value=original):
            got = adapter(*args)
        for field in fields(RachelN512Output):
            if field.name not in REPLACED_FIELDS:
                self.assertIs(getattr(got, field.name), getattr(original, field.name), field.name)
        self.assertIs(got.transport, original.transport)
        self.assertIs(got.assignment, original.assignment)
        self.assertIs(got.training_valid, original.training_valid)
        self.assertIs(got.decision_valid, original.decision_valid)

    def test_old_coarse_and_classification_scores_do_not_enter_head(self):
        args, teacher = inputs(), base()
        with torch.inference_mode(): original = teacher(*args)
        altered = replace(original, coarse_logit=original.coarse_logit+1000,
            coarse_probability=torch.zeros_like(original.coarse_probability),
            fused_logit=original.fused_logit-2000, local_logit=original.local_logit+3000)
        for arm in model.ARMS:
            adapter = FrozenMatchedInference(teacher, head(arm))
            with patch.object(teacher, "forward", return_value=original): first = adapter(*args)
            with patch.object(teacher, "forward", return_value=altered): second = adapter(*args)
            torch.testing.assert_close(first.fused_logit, second.fused_logit, rtol=0, atol=0)
            self.assertIs(second.coarse_logit, altered.coarse_logit)

    def test_invalid_training_mask_matches_cache_policy_not_a_new_decision_rule(self):
        args, teacher = inputs(), base()
        with torch.inference_mode(): original = teacher(*args)
        invalid = replace(original, training_valid=torch.zeros_like(original.training_valid),
                          decision_valid=torch.zeros_like(original.decision_valid))
        adapter = FrozenMatchedInference(teacher, head("all_tokens"))
        with patch.object(teacher, "forward", return_value=invalid): got = adapter(*args)
        self.assertEqual(float(got.fused_logit), 0.)
        self.assertEqual(float(got.fused_probability), .5)
        self.assertIs(got.training_valid, invalid.training_valid)
        self.assertIs(got.decision_valid, invalid.decision_valid)

    def test_no_candidate_keeps_trained_fallback_not_forced_positive(self):
        args, teacher = inputs(), base()
        with torch.inference_mode(): original = teacher(*args)
        empty = replace(original, assignment=torch.zeros_like(original.assignment),
                        training_valid=torch.ones_like(original.training_valid))
        for arm in ("matched_tokens", "matched_edges"):
            scorer = head(arm)
            with torch.no_grad(): scorer.head.no_evidence_logit.fill_(-2.)
            adapter = FrozenMatchedInference(teacher, scorer)
            with patch.object(teacher, "forward", return_value=empty): got = adapter(*args)
            self.assertTrue(bool(got.score_details["used_fallback"]))
            self.assertFalse(bool(got.score_details["has_decoded_candidate"]))
            self.assertEqual(float(got.fused_logit), -2.)
            self.assertLess(float(got.fused_probability), .2)
            self.assertIs(got.assignment, empty.assignment)

    def test_frozen_inference_changes_no_weights_and_creates_no_grad(self):
        teacher, scorer = base(), head("matched_edges")
        old_base, old_head = deepcopy(teacher.state_dict()), deepcopy(scorer.state_dict())
        adapter = FrozenMatchedInference(teacher, scorer)
        result = adapter(*inputs())
        self.assertFalse(adapter.training)
        self.assertFalse(teacher.training)
        self.assertFalse(scorer.training)
        self.assertFalse(any(p.requires_grad for p in adapter.parameters()))
        self.assertFalse(result.fused_logit.requires_grad)
        for k, v in old_base.items(): self.assertTrue(torch.equal(v, teacher.state_dict()[k]), k)
        for k, v in old_head.items(): self.assertTrue(torch.equal(v, scorer.state_dict()[k]), k)
        with self.assertRaisesRegex(ValueError, "inference-only"): adapter.train()

    def test_exact_six_inputs_no_gt_label_or_threshold(self):
        names = set(inspect.signature(FrozenMatchedInference.forward).parameters)
        self.assertEqual(names, {"self", "mask_a", "mask_b", "points_rc_a", "points_rc_b",
                                "contour_valid_a", "contour_valid_b"})
        adapter = FrozenMatchedInference(base(), head("matched_tokens"))
        for key in ("label", "target", "gt_translation", "threshold", "candidate_override"):
            with self.assertRaises(TypeError): adapter(*inputs(), **{key: 1})

    def test_configuration_mismatch_or_wrong_wrapper_rejected(self):
        with self.assertRaises(TypeError): FrozenMatchedInference(torch.nn.Identity(), head("all_tokens"))
        wrong = model.make_fresh_scorer("all_tokens", feature_dim=8, num_heads=2)
        with self.assertRaisesRegex(ValueError, "dimension"): FrozenMatchedInference(base(), wrong)
        with self.assertRaises(TypeError): FrozenMatchedInference(base(), torch.nn.Identity())

    def test_nonfinite_head_is_not_silently_masked(self):
        args, teacher, scorer = inputs(), base(), head("all_tokens")
        with torch.inference_mode(): original = teacher(*args)
        result, _, _, _ = cache_style(original, args, scorer)
        adapter = FrozenMatchedInference(teacher, scorer)
        with patch.object(scorer, "forward", return_value=replace(result, logit=torch.full((1,), float("nan")))):
            with self.assertRaisesRegex(ValueError, "nonfinite"): adapter(*args)

    def test_seed_multi_online_equals_actual_sidecache_and_original_layout_preserved(self):
        args, teacher = inputs(), base()
        with torch.inference_mode(): original = teacher(*args)
        _, _, selection, weights = cache_style(original, args, head("all_tokens"))
        with TemporaryDirectory() as temporary:
            source = GeometryCacheFixture(Path(temporary)/"source", args, selection, weights)
            directory = Path(temporary)/"stages"
            stage_cache.prepare(source, directory)
            side_cache = stage_cache.StageCache(directory, source)
            for arm, slots in (("edge_seed", 1), ("edge_multi", 5)):
                scorer = CandidateStageScorer(arm, seed=42, feature_dim=16, num_heads=4).eval()
                cached_groups = side_cache.batch([0], arm)
                derived, diagnostics = online_stage_groups(arm, selection, weights, *args[2:])
                self.assertEqual(derived.candidate_inliers.shape, (1, slots, 512))
                for field in fields(stage_cache.StageGroups):
                    if field.name == "stage": continue
                    torch.testing.assert_close(getattr(derived, field.name), getattr(cached_groups, field.name),
                                               rtol=0, atol=0, equal_nan=True)
                with torch.inference_mode():
                    expected = scorer(original.token_features_a, original.token_features_b,
                        args[4], args[5], selection, groups=cached_groups,
                        candidate_weights=weights, points_a_rc=args[2], points_b_rc=args[3])
                adapter = FrozenMatchedInference(teacher, scorer)
                deployed = torch.where(original.training_valid, expected.logit, torch.zeros_like(expected.logit))
                # Exercise genuine six-input base inference, not only an output
                # fixture, before separately checking exact object preservation.
                online = adapter(*args)
                torch.testing.assert_close(online.fused_logit, deployed, rtol=0, atol=0)
                torch.testing.assert_close(online.assignment, original.assignment, rtol=0, atol=0)
                with patch.object(teacher, "forward", return_value=original): got = adapter(*args)
                torch.testing.assert_close(got.fused_logit, deployed, rtol=0, atol=0)
                torch.testing.assert_close(got.score_details["group_logits"], expected.group_logits,
                                           rtol=0, atol=0, equal_nan=True)
                self.assertTrue(torch.equal(got.score_details["selected_group_rank"], expected.selected_group_rank))
                self.assertEqual(got.score_details["group_diagnostics"], diagnostics)
                for field in fields(RachelN512Output):
                    if field.name not in REPLACED_FIELDS:
                        self.assertIs(getattr(got, field.name), getattr(original, field.name), field.name)

    def test_selected_alternative_pose_is_diagnostic_never_layout_replacement(self):
        args, teacher = list(inputs()), base()
        with torch.inference_mode(): original = teacher(*args)
        # Two genuine geometric modes in a controlled PREDICTED Q (no GT).
        args[2] = torch.tensor([[[0., 5.*i] for i in range(8)]])
        args[3] = args[2] + torch.tensor([[[d, 0.] for d in (3,3,3,33,33,33,60,60)]])
        q = torch.diag(torch.tensor([1.,1.,1.,.5,.5,.5,.01,.01]))[None]
        original = replace(original, assignment=q, training_valid=torch.ones_like(original.training_valid))
        _, _, selection, weights = cache_style(original, args, head("all_tokens"))
        groups, _ = online_stage_groups("edge_multi", selection, weights, *args[2:])
        self.assertEqual(int(groups.eligible.sum()), 2)
        scorer = CandidateStageScorer("edge_multi", seed=42, feature_dim=16, num_heads=4).eval()
        with torch.inference_mode():
            values = vars(scorer(original.token_features_a, original.token_features_b, args[4], args[5],
                selection, groups=groups, candidate_weights=weights,
                points_a_rc=args[2], points_b_rc=args[3])).copy()
        # Force a consistent head outcome selecting the NONproduction second mode.
        logits = torch.tensor([[-2., 2., float("nan"), float("nan"), float("nan")]])
        values.update(logit=torch.tensor([2.]), group_logits=logits,
                      selected_group_rank=torch.tensor([2], dtype=groups.ranks.dtype))
        adapter = FrozenMatchedInference(teacher, scorer)
        with patch.object(teacher, "forward", return_value=original), patch.object(scorer, "forward", return_value=SimpleNamespace(**values)):
            got = adapter(*args)
        torch.testing.assert_close(got.score_details["selected_group_translation_rc"], torch.tensor([[33., 0.]]))
        torch.testing.assert_close(got.score_details["selection"].translation_a_to_b_rc, torch.tensor([[3., 0.]]))
        self.assertIs(got.translation_hat_rc, original.translation_hat_rc)
        self.assertIs(got.assignment, original.assignment)
        self.assertIn("NOT replaced", got.score_details["selected_pose_role"])

    def test_stage_no_group_fallback_retains_no_pose(self):
        args, teacher = inputs(), base()
        with torch.inference_mode(): original = teacher(*args)
        empty = replace(original, assignment=torch.zeros_like(original.assignment),
                        training_valid=torch.ones_like(original.training_valid))
        for arm in ("edge_seed", "edge_multi"):
            scorer = CandidateStageScorer(arm, seed=42, feature_dim=16, num_heads=4).eval()
            with torch.no_grad(): scorer.edge_head.head.no_evidence_logit.fill_(-2.)
            adapter = FrozenMatchedInference(teacher, scorer)
            with patch.object(teacher, "forward", return_value=empty): got = adapter(*args)
            self.assertEqual(float(got.fused_logit), -2.)
            self.assertFalse(bool(got.score_details["has_selected_candidate_group"]))
            self.assertEqual(int(got.score_details["selected_group_rank"]), 0)
            self.assertTrue(torch.isnan(got.score_details["selected_group_translation_rc"]).all())
            self.assertIs(got.translation_hat_rc, empty.translation_hat_rc)

    def test_original_endpoint_serializer_retains_edge_identity_and_group_membership(self):
        from experiments.rachel_n512_formal_30k import evaluate_score_design as core
        args, teacher = inputs(), base()
        with torch.inference_mode(): original = teacher(*args)
        for arm in (*model.ARMS, "edge_seed", "edge_multi"):
            scorer = (head(arm) if arm in model.ARMS else
                CandidateStageScorer(arm, seed=42, feature_dim=16, num_heads=4).eval())
            adapter = FrozenMatchedInference(teacher, scorer)
            with patch.object(teacher, "forward", return_value=original): output = adapter(*args)
            row = core._candidate_details(output, 0)
            selected = output.score_details["selection"]
            self.assertEqual(row["candidate_indices"], selected.candidate_indices[0].tolist())
            self.assertEqual(row["candidate_valid"], selected.candidate_valid[0].tolist())
            self.assertEqual(row["candidate_inliers"], selected.candidate_inliers[0].tolist())
            self.assertEqual(row["candidate_weights"], output.score_details["candidate_weights"][0].tolist())
            self.assertEqual(len(row["candidate_indices"]), 512)
            self.assertIs(output.decision_valid, original.decision_valid)
            self.assertIs(output.training_valid, original.training_valid)
            if arm in ("edge_seed", "edge_multi"):
                groups = output.score_details["groups"]
                self.assertEqual(row["group_candidate_inliers"], groups.candidate_inliers[0].tolist())
                self.assertEqual(row["group_eligible"], groups.eligible[0].tolist())
                for membership in row["group_candidate_inliers"]:
                    recovered_edges = [edge for edge, keep in zip(row["candidate_indices"], membership) if keep]
                    self.assertTrue(all(min(edge) >= 0 for edge in recovered_edges))
            else:
                self.assertNotIn("group_candidate_inliers", row)
            self.assertFalse(set(row) & {"gt", "label", "target", "assignment", "full_q"})


if __name__ == "__main__":
    unittest.main()
