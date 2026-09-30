from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from staging.pairwise_v0_2.geometry import (
    CandidateBuilderConfig,
    ContourKeypointConfig,
    DEFAULT_DIRECTION_ORDER,
    build_fragment_geometry,
    build_fragment_keypoints,
    build_keypoint_pair_candidates,
)
from staging.pairwise_v0_2.models.local_matcher import (
    MatcherMode,
    OrderedLocalMatcher,
)
from staging.pairwise_v0_2.training.keypoint_batch import (
    build_keypoint_tensor_batch,
)


def _rectangle(top: int, left: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[top : top + height, left : left + width] = 255
    return mask


def _notched(top: int, left: int, height: int, width: int) -> np.ndarray:
    mask = _rectangle(top, left, height, width)
    notch_height = max(5, height // 3)
    notch_width = max(5, width // 3)
    mask[
        top + height // 3 : top + height // 3 + notch_height,
        left + width - notch_width : left + width,
    ] = 0
    return mask


def _geometry_config() -> CandidateBuilderConfig:
    return CandidateBuilderConfig(
        min_run_length_fraction=0.0,
        min_run_length_px=2.0,
        window_scale_fractions=(0.08, 0.16),
        window_min_px=2.0,
        window_max_px=128.0,
        output_size=(16, 16),
        side_resample_count=48,
    )


def _selector_config() -> ContourKeypointConfig:
    return ContourKeypointConfig(
        max_keypoints_per_side=8,
        uniform_keypoints_per_side=3,
        curvature_keypoints_per_side=3,
        curvature_radius_fractions=(0.02, 0.04, 0.08),
        curvature_nms_arc_fraction=0.01,
    )


def _fragment(mask: np.ndarray):
    result = build_fragment_geometry(mask, _geometry_config())
    assert result.ok and result.artifact is not None
    return result.artifact


def test_four_is_direction_count_not_contour_segment_count() -> None:
    """The keypoint arm emits four hypotheses but many anchors per side."""

    artifact_a = _fragment(_notched(12, 14, 92, 74))
    artifact_b = _fragment(_notched(17, 33, 82, 65))
    settings = _selector_config()
    keypoints_a = build_fragment_keypoints(artifact_a, settings)
    keypoints_b = build_fragment_keypoints(artifact_b, settings)
    result = build_keypoint_pair_candidates(keypoints_a, keypoints_b, settings)

    assert tuple(candidate.direction for candidate in result.candidates) == (
        DEFAULT_DIRECTION_ORDER
    )
    assert len(result.candidates) == 4
    assert all(count > 1 for _, count in keypoints_a.keypoint_counts_by_side)
    assert all(
        count <= settings.max_keypoints_per_side
        for _, count in (keypoints_a.keypoint_counts_by_side)
    )
    reasons = {
        reason
        for keypoint in keypoints_a.keypoints
        for reason in keypoint.selection_reasons
    }
    assert "uniform_arc_length" in reasons
    assert "retained_run_endpoint_nearest_window" in reasons
    assert "curvature_multiradius" in reasons


def test_fragment_selection_has_no_pair_label_or_target_direction_api() -> None:
    parameters = set(inspect.signature(build_fragment_keypoints).parameters)
    pair_parameters = set(inspect.signature(build_keypoint_pair_candidates).parameters)
    forbidden = {"label", "pair_label", "direction", "direction_b_wrt_a"}

    assert parameters.isdisjoint(forbidden)
    assert pair_parameters.isdisjoint(forbidden)


def test_multiscale_tokens_reuse_exact_cached_patch_tensors() -> None:
    artifact = _fragment(_notched(12, 14, 92, 74))
    selected = build_fragment_keypoints(artifact, _selector_config())
    source = artifact.sequence_map

    for keypoint in selected.keypoints:
        assert tuple(token.scale_index for token in keypoint.tokens) == (0, 1)
        for token in keypoint.tokens:
            sequence = source[(token.side, token.run_index, token.scale_index)]
            np.testing.assert_array_equal(
                token.channels,
                sequence.channels[token.source_patch_index],
            )
            assert np.shares_memory(
                token.channels, sequence.channels[token.source_patch_index]
            )
            assert token.channels.flags.writeable is False


def test_pair_candidates_use_facing_sides_and_same_scale_sparse_mask() -> None:
    settings = _selector_config()
    keypoints_a = build_fragment_keypoints(
        _fragment(_notched(12, 14, 92, 74)), settings
    )
    keypoints_b = build_fragment_keypoints(
        _fragment(_notched(17, 33, 82, 65)), settings
    )
    result = build_keypoint_pair_candidates(keypoints_a, keypoints_b, settings)

    for candidate in result.candidates:
        side_a, side_b = candidate.direction.facing_sides
        assert {token.side for token in candidate.tokens_a} == {side_a}
        assert {token.side for token in candidate.tokens_b} == {side_b}
        expected = np.asarray(
            [
                [
                    first.scale_index == second.scale_index
                    for second in candidate.tokens_b
                ]
                for first in candidate.tokens_a
            ],
            dtype=np.bool_,
        )
        np.testing.assert_array_equal(candidate.correspondence_mask, expected)
        assert candidate.allowed_pair_count < expected.size

    maximum = settings.max_keypoints_per_side * keypoints_a.scale_count
    assert result.complexity.max_tokens_a <= maximum
    assert result.complexity.max_tokens_b <= maximum
    assert result.complexity.affinity_elements <= 4 * maximum * maximum
    assert result.complexity.sinkhorn_elements <= (
        result.complexity.configured_max_sinkhorn_elements_per_pair
    )


def test_keypoint_candidates_are_exact_under_a_b_swap_and_direction_inversion() -> None:
    settings = _selector_config()
    keypoints_a = build_fragment_keypoints(
        _fragment(_notched(12, 14, 92, 74)), settings
    )
    keypoints_b = build_fragment_keypoints(
        _fragment(_notched(17, 33, 82, 65)), settings
    )
    forward = build_keypoint_pair_candidates(keypoints_a, keypoints_b, settings)
    reverse = build_keypoint_pair_candidates(keypoints_b, keypoints_a, settings)
    reverse_by_direction = {
        candidate.direction: candidate for candidate in reverse.candidates
    }

    for candidate in forward.candidates:
        swapped = reverse_by_direction[candidate.direction.inverse]
        np.testing.assert_array_equal(candidate.patches_a, swapped.patches_b)
        np.testing.assert_array_equal(candidate.patches_b, swapped.patches_a)
        np.testing.assert_array_equal(
            candidate.correspondence_mask,
            swapped.correspondence_mask.T,
        )


@pytest.mark.parametrize(
    "mode", (MatcherMode.DUAL_SOFTMAX, MatcherMode.DUSTBIN_SINKHORN)
)
def test_sparse_keypoint_matcher_is_trainable_and_forbidden_edges_stay_zero(
    mode: MatcherMode,
) -> None:
    torch.manual_seed(37)
    settings = _selector_config()
    keypoints_a = build_fragment_keypoints(
        _fragment(_notched(12, 14, 92, 74)), settings
    )
    keypoints_b = build_fragment_keypoints(
        _fragment(_notched(17, 33, 82, 65)), settings
    )
    geometry = build_keypoint_pair_candidates(keypoints_a, keypoints_b, settings)
    batch = build_keypoint_tensor_batch((geometry,))
    # Two candidates are enough to cover a non-square sparse plan while
    # keeping the CPU fixture quick.
    patches_a = batch.local_a[:2].clone().requires_grad_(True)
    patches_b = batch.local_b[:2].clone().requires_grad_(True)
    mask_a = batch.token_mask_a[:2]
    mask_b = batch.token_mask_b[:2]
    correspondence = batch.correspondence_mask[:2]
    matcher = OrderedLocalMatcher(
        input_channels=3,
        feature_dim=8,
        num_heads=2,
        ff_dim=16,
        matcher_mode=mode,
        matcher_temperature=0.25,
        sinkhorn_iterations=60,
        sinkhorn_tolerance=1.0,
        dropout=0.0,
    )
    output = matcher(
        patches_a,
        patches_b,
        mask_a,
        mask_b,
        correspondence_mask=correspondence,
    )

    assert output.training_valid.all().item()
    assert torch.isfinite(output.logit).all().item()
    assert torch.equal(
        output.assignment[~correspondence],
        torch.zeros_like(output.assignment[~correspondence]),
    )
    loss = output.logit.sum() + 0.01 * output.assignment.square().sum()
    loss.backward()
    assert patches_a.grad is not None and torch.isfinite(patches_a.grad).all().item()
    assert patches_b.grad is not None and torch.isfinite(patches_b.grad).all().item()
    trainable_gradients = [
        parameter.grad
        for parameter in matcher.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_gradients
    assert all(torch.isfinite(value).all().item() for value in trainable_gradients)
    assert any(value.abs().sum().item() > 0.0 for value in trainable_gradients)


def test_empty_correspondence_graph_fails_closed() -> None:
    matcher = OrderedLocalMatcher(
        input_channels=3,
        feature_dim=8,
        num_heads=2,
        ff_dim=16,
        matcher_mode=MatcherMode.DUSTBIN_SINKHORN,
        sinkhorn_iterations=5,
        sinkhorn_tolerance=1.0,
    )
    patches_a = torch.ones((1, 2, 3, 16, 16), dtype=torch.float32)
    patches_b = torch.ones((1, 3, 3, 16, 16), dtype=torch.float32)
    mask_a = torch.ones((1, 2), dtype=torch.bool)
    mask_b = torch.ones((1, 3), dtype=torch.bool)
    correspondence = torch.zeros((1, 2, 3), dtype=torch.bool)

    output = matcher(
        patches_a,
        patches_b,
        mask_a,
        mask_b,
        correspondence_mask=correspondence,
    )

    assert not output.training_valid.item()
    assert not output.decision_valid.item()
    assert output.assignment.count_nonzero().item() == 0
    assert output.logit.item() == 0.0


def test_sparse_sinkhorn_preserves_a_b_swap_symmetry() -> None:
    torch.manual_seed(91)
    matcher = OrderedLocalMatcher(
        input_channels=3,
        feature_dim=8,
        num_heads=2,
        ff_dim=16,
        matcher_mode=MatcherMode.DUSTBIN_SINKHORN,
        sinkhorn_iterations=80,
        sinkhorn_tolerance=1.0,
        dropout=0.0,
    ).eval()
    patches_a = torch.rand((1, 4, 3, 16, 16))
    patches_b = torch.rand((1, 5, 3, 16, 16))
    mask_a = torch.ones((1, 4), dtype=torch.bool)
    mask_b = torch.ones((1, 5), dtype=torch.bool)
    correspondence = torch.tensor(
        [
            [
                [True, False, True, False, True],
                [False, True, False, True, False],
                [True, False, True, False, True],
                [False, True, False, True, False],
            ]
        ],
        dtype=torch.bool,
    )

    forward = matcher(
        patches_a,
        patches_b,
        mask_a,
        mask_b,
        correspondence_mask=correspondence,
    )
    swapped = matcher(
        patches_b,
        patches_a,
        mask_b,
        mask_a,
        correspondence_mask=correspondence.transpose(1, 2),
    )

    torch.testing.assert_close(forward.affinity, swapped.affinity.transpose(1, 2))
    torch.testing.assert_close(
        forward.assignment,
        swapped.assignment.transpose(1, 2),
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(forward.logit, swapped.logit, atol=2e-6, rtol=2e-6)


def test_sparse_mask_validation_is_strict() -> None:
    matcher = OrderedLocalMatcher(
        input_channels=3,
        feature_dim=8,
        num_heads=2,
        ff_dim=16,
        matcher_mode=MatcherMode.DUAL_SOFTMAX,
    )
    patches = torch.ones((1, 2, 3, 16, 16), dtype=torch.float32)
    tokens = torch.ones((1, 2), dtype=torch.bool)

    with pytest.raises(TypeError, match="correspondence_mask"):
        matcher(patches, patches, tokens, tokens, torch.ones((1, 2, 2)))
    with pytest.raises(ValueError, match="correspondence_mask"):
        matcher(
            patches,
            patches,
            tokens,
            tokens,
            torch.ones((1, 2, 3), dtype=torch.bool),
        )
