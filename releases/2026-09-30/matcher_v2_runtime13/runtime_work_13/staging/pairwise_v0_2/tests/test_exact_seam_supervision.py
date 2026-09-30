from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from staging.pairwise_v0_1.baselines.b1_contours import CardinalSide
from staging.pairwise_v0_2.geometry.keypoint_candidates import (
    KeypointComplexity,
    KeypointPairCandidate,
    KeypointPairResult,
    KeypointScaleToken,
)
from staging.pairwise_v0_2.geometry.schema import PairDirection
from staging.pairwise_v0_2.models.local_matcher import MatcherMode, OrderedLocalMatcher
from staging.pairwise_v0_2.pairwise_data.exact_seam_supervision import (
    DUSTBIN_ASSIGNMENT,
    IGNORE_ASSIGNMENT,
    ExactSeamSupervisionError,
    ExactSeamTargetConfig,
    build_exact_seam_tensor_batch,
    build_pair_exact_seam_targets,
    exact_partial_assignment_nll,
    export_group_exact_seams,
    extract_aligned_mask_seam,
)
from staging.pairwise_v0_2.training.keypoint_batch import build_keypoint_tensor_batch


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _token(
    token_id: str,
    side: CardinalSide,
    center: tuple[float, float],
    *,
    scale_index: int = 0,
) -> KeypointScaleToken:
    channels = np.zeros((3, 8, 8), dtype=np.float32)
    channels[0, 2:6, 2:6] = 1.0
    return KeypointScaleToken(
        token_id=token_id,
        anchor_id=token_id,
        side=side,
        run_index=0,
        scale_index=scale_index,
        scale_fraction=0.08,
        source_patch_index=0,
        source_arc_fraction=0.0,
        source_path_fraction=0.0,
        center_row_col=center,
        channels=_readonly(channels, np.float32),
    )


def _candidate() -> KeypointPairCandidate:
    tokens_a = (
        _token("a0", CardinalSide.RIGHT, (2.5, 8.0)),
        _token("a1", CardinalSide.RIGHT, (11.5, 8.0)),
        _token("a_outer", CardinalSide.RIGHT, (8.0, 0.0)),
    )
    tokens_b = (
        _token("b0", CardinalSide.LEFT, (3.0, 8.0)),
        _token("b1", CardinalSide.LEFT, (11.0, 8.0)),
        _token("b_outer", CardinalSide.LEFT, (8.0, 15.0)),
    )
    patches_a = _readonly(np.stack([value.channels for value in tokens_a]), np.float32)
    patches_b = _readonly(np.stack([value.channels for value in tokens_b]), np.float32)
    allowed = _readonly(np.ones((3, 3), dtype=np.bool_), np.bool_)
    return KeypointPairCandidate(
        candidate_id="b_right_of_a:fixture",
        direction=PairDirection.B_RIGHT_OF_A,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        patches_a=patches_a,
        patches_b=patches_b,
        correspondence_mask=allowed,
    )


def _result(candidate: KeypointPairCandidate) -> KeypointPairResult:
    return KeypointPairResult(
        candidates=(candidate,),
        unavailable_directions=(),
        complexity=KeypointComplexity(
            candidate_count=1,
            max_tokens_a=len(candidate.tokens_a),
            max_tokens_b=len(candidate.tokens_b),
            affinity_elements=len(candidate.tokens_a) * len(candidate.tokens_b),
            sinkhorn_elements=(len(candidate.tokens_a) + 1)
            * (len(candidate.tokens_b) + 1),
            configured_max_tokens_per_fragment_direction=3,
            configured_max_affinity_elements_per_direction=9,
            configured_max_sinkhorn_elements_per_pair=16,
        ),
    )


def _complementary_halves() -> tuple[np.ndarray, np.ndarray]:
    first = np.zeros((16, 16), dtype=np.uint8)
    second = np.zeros_like(first)
    first[:, :8] = 255
    second[:, 8:] = 255
    return first, second


def test_exact_seam_is_parent_canvas_grid_edge_mapping() -> None:
    first, second = _complementary_halves()
    seam = extract_aligned_mask_seam(first, second)

    assert seam is not None
    assert seam.correspondence_count == 16
    assert seam.path_count == 1
    assert seam.has_branch_vertex is False
    np.testing.assert_array_equal(seam.a_pixels_rc[:, 1], np.full(16, 7))
    np.testing.assert_array_equal(seam.b_pixels_rc[:, 1], np.full(16, 8))
    np.testing.assert_allclose(seam.edge_midpoints_rc[:, 1], 8.0)
    np.testing.assert_allclose(seam.path_arclength_px, np.arange(16) + 0.5)
    np.testing.assert_allclose(seam.path_lengths_px, [16.0])


def test_keypoint_target_has_reciprocal_matches_dustbins_and_no_teacher_forcing() -> None:
    first, second = _complementary_halves()
    candidate = _candidate()
    keypoints = _result(candidate)
    targets = build_pair_exact_seam_targets(
        first,
        second,
        keypoints,
        pair_key="fixture/0--1",
        pair_is_adjacent=True,
        config=ExactSeamTargetConfig(
            max_token_to_seam_distance_px=0.6,
            max_match_arclength_gap_px=2.0,
        ),
    )
    target = targets.candidates[0]

    assert target.matched_token_pair_count == 2
    assert target.assignment_target_a[2] == DUSTBIN_ASSIGNMENT
    assert target.assignment_target_b[2] == DUSTBIN_ASSIGNMENT
    for index in (0, 1):
        opposite = int(target.assignment_target_a[index])
        assert opposite >= 0
        assert int(target.assignment_target_b[opposite]) == index
    # The model still receives the full label-blind graph, not the two GT edges.
    assert candidate.correspondence_mask.all()
    assert int(candidate.correspondence_mask.sum()) == 9


def test_positive_gap_is_rejected_instead_of_inventing_exact_correspondence() -> None:
    first = np.zeros((16, 16), dtype=np.uint8)
    second = np.zeros_like(first)
    first[:, :6] = 255
    second[:, 10:] = 255
    assert extract_aligned_mask_seam(first, second) is None

    with pytest.raises(ExactSeamSupervisionError, match="no exact"):
        build_pair_exact_seam_targets(
            first,
            second,
            _result(_candidate()),
            pair_key="legacy-gap/0--1",
            pair_is_adjacent=True,
        )


def test_negative_pair_supervises_every_real_token_to_dustbin() -> None:
    first = np.zeros((16, 16), dtype=np.uint8)
    second = np.zeros_like(first)
    first[1:5, 1:5] = 255
    second[11:15, 11:15] = 255
    targets = build_pair_exact_seam_targets(
        first,
        second,
        _result(_candidate()),
        pair_key="negative/0--1",
        pair_is_adjacent=False,
    )
    target = targets.candidates[0]

    assert np.all(target.assignment_target_a == DUSTBIN_ASSIGNMENT)
    assert np.all(target.assignment_target_b == DUSTBIN_ASSIGNMENT)
    assert target.matched_token_pair_count == 0


def test_wrong_positive_direction_can_remain_ignored() -> None:
    first, second = _complementary_halves()
    tokens_a = (_token("a-left", CardinalSide.LEFT, (8.0, 0.0)),)
    tokens_b = (_token("b-right", CardinalSide.RIGHT, (8.0, 16.0)),)
    wrong = KeypointPairCandidate(
        candidate_id="b_left_of_a:wrong",
        direction=PairDirection.B_LEFT_OF_A,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        patches_a=_readonly(np.stack([tokens_a[0].channels]), np.float32),
        patches_b=_readonly(np.stack([tokens_b[0].channels]), np.float32),
        correspondence_mask=_readonly(np.ones((1, 1), dtype=np.bool_), np.bool_),
    )
    targets = build_pair_exact_seam_targets(
        first,
        second,
        _result(wrong),
        pair_key="fixture/0--1",
        pair_is_adjacent=True,
    )

    assert targets.candidates[0].assignment_target_a.tolist() == [IGNORE_ASSIGNMENT]
    assert targets.candidates[0].assignment_target_b.tolist() == [IGNORE_ASSIGNMENT]


def test_target_batch_aligns_exactly_with_model_input_batch() -> None:
    first, second = _complementary_halves()
    result = _result(_candidate())
    pair_targets = build_pair_exact_seam_targets(
        first,
        second,
        result,
        pair_key="fixture/0--1",
        pair_is_adjacent=True,
        config=ExactSeamTargetConfig(
            max_token_to_seam_distance_px=0.6,
            max_match_arclength_gap_px=2.0,
        ),
    )
    model_batch = build_keypoint_tensor_batch((result,))
    target_batch = build_exact_seam_tensor_batch((pair_targets,))

    target_batch.assert_aligned_keypoint_batch(model_batch)
    assert target_batch.candidate_ids == model_batch.candidate_ids
    assert target_batch.assignment_target_a.shape == model_batch.token_mask_a.shape
    assert target_batch.assignment_target_b.shape == model_batch.token_mask_b.shape


def test_exact_partial_assignment_nll_backpropagates_through_sinkhorn() -> None:
    torch.manual_seed(7)
    first, second = _complementary_halves()
    result = _result(_candidate())
    pair_targets = build_pair_exact_seam_targets(
        first,
        second,
        result,
        pair_key="fixture/0--1",
        pair_is_adjacent=True,
        config=ExactSeamTargetConfig(
            max_token_to_seam_distance_px=0.6,
            max_match_arclength_gap_px=2.0,
        ),
    )
    model_batch = build_keypoint_tensor_batch((result,))
    target_batch = build_exact_seam_tensor_batch((pair_targets,))
    matcher = OrderedLocalMatcher(
        input_channels=3,
        feature_dim=8,
        num_heads=2,
        ff_dim=16,
        matcher_mode=MatcherMode.DUSTBIN_SINKHORN,
        sinkhorn_iterations=40,
        sinkhorn_tolerance=1.0,
        dropout=0.0,
    )
    output = matcher(**model_batch.model_inputs())
    loss = exact_partial_assignment_nll(output, target_batch)

    assert loss.supervised_match_count == 2
    assert loss.supervised_dustbin_a_count == 1
    assert loss.supervised_dustbin_b_count == 1
    assert torch.isfinite(loss.total).item()
    loss.total.backward()
    assert matcher.dustbin_score.grad is not None
    assert torch.isfinite(matcher.dustbin_score.grad).item()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in matcher.parameters()
    )


def test_generator_sidecar_export_is_compact_and_mask_only(tmp_path: Path) -> None:
    first, second = _complementary_halves()
    output = export_group_exact_seams(tmp_path, {"0": first, "1": second})

    with np.load(str(output), allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["metadata_json_utf8"]).decode("utf-8"))
        assert metadata["seam_definition"] == "all_4_neighbor_cross_child_grid_edges"
        assert metadata["pairs"][0]["edge_count"] == 16
        np.testing.assert_array_equal(
            archive["pair_0_1_a_pixels_rc"][:, 1], np.full(16, 7)
        )
        assert not any("rgb" in key.lower() or "text" in key.lower() for key in archive)
