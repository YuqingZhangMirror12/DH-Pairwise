import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from staging.pairwise_v0_1.baselines.b1_contours import CardinalSide
from staging.pairwise_v0_2.geometry import (
    CandidateBuilderConfig,
    CorrosionConfig,
    GeometryStatus,
    PairDirection,
    build_fragment_geometry,
    build_pair_candidates,
    geometry_config_fingerprint,
)


def _rectangle(
    shape=(256, 256),
    top=36,
    left=48,
    height=160,
    width=96,
):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[top : top + height, left : left + width] = 255
    return mask


def _config(**kwargs):
    values = {
        "window_scale_fractions": (0.1, 0.2),
        "window_min_px": 2.0,
        "window_max_px": 256.0,
        "output_size": (16, 18),
        "min_run_length_fraction": 0.0,
        "min_run_length_px": 2.0,
    }
    values.update(kwargs)
    return CandidateBuilderConfig(**values)


def _u_shape():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[6:54, 6:18] = 255
    mask[42:54, 6:54] = 255
    mask[6:54, 42:54] = 255
    return mask


def _saddle_shape():
    mask = np.asarray(
        (
            (1, 1, 1, 0),
            (1, 1, 0, 0),
            (1, 0, 1, 0),
            (1, 1, 1, 0),
        ),
        dtype=np.uint8,
    )
    return np.pad(mask, 3) * 255


def test_default_candidate_config_fingerprints_opt_in_saddle_semantics() -> None:
    config = _config(
        min_component_pixels=1,
        min_contour_points=4,
        min_run_length_px=1.0,
        window_scale_fractions=(0.5,),
        window_min_px=1.0,
        output_size=(8, 8),
    )
    legacy = replace(config, saddle_policy="reject")

    assert asdict(config)["saddle_policy"] == "foreground_4_background_8"
    assert geometry_config_fingerprint(config) != geometry_config_fingerprint(legacy)
    repaired = build_fragment_geometry(_saddle_shape(), config)
    rejected = build_fragment_geometry(_saddle_shape(), legacy)
    assert repaired.ok
    assert rejected.status is GeometryStatus.INVALID_CONTOUR
    assert rejected.failure_reason == "non_manifold_boundary"


def test_unknown_relative_direction_emits_four_complementary_facing_families() -> None:
    mask_a = _rectangle(top=30, left=28, height=170, width=92)
    mask_b = _rectangle(top=42, left=132, height=150, width=78)
    result = build_pair_candidates(mask_a, mask_b, config=_config())

    assert result.status is GeometryStatus.OK
    assert result.failure_reason is None
    assert len(result.candidates) == 8
    assert result.quality.requested_directions == tuple(
        direction.value for direction in PairDirection
    )
    assert result.quality.emitted_directions == result.quality.requested_directions
    assert tuple(group.short_name for group in result.direction_groups) == (
        "left",
        "right",
        "above",
        "below",
    )
    assert tuple(group.slot_index for group in result.direction_groups) == (0, 1, 2, 3)
    assert result.quality.direction_candidate_counts == (
        ("left", 2),
        ("right", 2),
        ("above", 2),
        ("below", 2),
    )
    assert tuple(
        index for group in result.direction_groups for index in group.candidate_indices
    ) == tuple(range(len(result.candidates)))
    assert result.quality.rotation_search_performed is False
    assert result.quality.upright_orientation_assumed is True
    assert result.quality.channel_order == (
        "mask",
        "signed_distance",
        "boundary_gradient",
    )
    assert result.quality.window_normalization == "per_fragment_bbox_min_dimension"
    json.dumps(result.quality.to_dict(), allow_nan=False)

    for candidate in result.candidates:
        side_a, side_b = candidate.direction.facing_sides
        assert candidate.sequence_a.side is side_a
        assert candidate.sequence_b.side is side_b
        assert side_a is not side_b
        assert candidate.patches_a.ndim == 4
        assert candidate.patches_a.shape[1:] == (3, 16, 18)
        assert candidate.patches_b.shape[1:] == (3, 16, 18)
        assert candidate.valid_a.shape == (candidate.patches_a.shape[0],)
        assert candidate.valid_b.shape == (candidate.patches_b.shape[0],)
        assert np.all(candidate.valid_a) and np.all(candidate.valid_b)


def test_supervised_direction_restricts_family_but_never_rotates_masks() -> None:
    result = build_pair_candidates(
        _rectangle(),
        _rectangle(top=44, left=132, height=140, width=70),
        PairDirection.B_RIGHT_OF_A,
        _config(),
    )

    assert result.ok
    assert len(result.candidates) == 2
    assert {candidate.direction for candidate in result.candidates} == {
        PairDirection.B_RIGHT_OF_A
    }
    assert {candidate.sequence_a.side for candidate in result.candidates} == {
        CardinalSide.RIGHT
    }
    assert {candidate.sequence_b.side for candidate in result.candidates} == {
        CardinalSide.LEFT
    }
    assert result.quality.requested_directions == ("b_right_of_a",)
    assert result.quality.rotation_search_performed is False


def test_multirun_cross_product_is_retained_without_best_arc_selection() -> None:
    result = build_pair_candidates(
        _u_shape(),
        _u_shape(),
        PairDirection.B_LEFT_OF_A,
        _config(window_scale_fractions=(0.1,), output_size=(12, 12)),
    )

    assert result.ok
    assert result.quality.fragment_a is not None
    assert result.quality.fragment_b is not None
    assert dict(result.quality.fragment_a.retained_runs_by_side) == {
        "left": 2,
        "right": 2,
        "top": 2,
        "bottom": 2,
    }
    # Every A-left x B-right run pairing survives; no score or label selects a
    # single longest/best run in this geometry stage.
    assert len(result.candidates) == 4
    assert len(result.direction_groups) == 1
    assert result.direction_groups[0].candidate_indices == (0, 1, 2, 3)
    assert result.direction_groups[0].short_name == "left"
    assert result.candidates_for_direction(PairDirection.B_LEFT_OF_A) == (
        result.candidates
    )
    assert {
        (candidate.sequence_a.run_index, candidate.sequence_b.run_index)
        for candidate in result.candidates
    } == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_multiscale_windows_are_normalized_overlap_and_remain_variable_length() -> None:
    mask_a = _rectangle(height=180, width=80)
    mask_b = _rectangle(top=70, left=140, height=90, width=80)
    config = _config(window_scale_fractions=(0.1, 0.25), overlap_fraction=0.5)
    result = build_pair_candidates(
        mask_a,
        mask_b,
        PairDirection.B_RIGHT_OF_A,
        config,
    )

    assert result.ok
    assert len(result.candidates) == 2
    fine, coarse = result.candidates
    for candidate in result.candidates:
        for sequence in (candidate.sequence_a, candidate.sequence_b):
            assert sequence.requested_window_px == pytest.approx(
                sequence.bbox_reference_px * sequence.scale_fraction
            )
            assert sequence.resolved_window_px == pytest.approx(
                sequence.requested_window_px
            )
            assert sequence.stride_px == pytest.approx(
                sequence.resolved_window_px * 0.5
            )
            distances = np.asarray(
                [patch.path_distance_px for patch in sequence.patches]
            )
            if len(distances) > 1:
                assert np.all(np.diff(distances) <= sequence.stride_px + 1e-8)
                assert np.all(np.diff(distances) > 0.0)
            assert sequence.provenance_dict()["window_normalization"] == (
                "per_fragment_bbox_min_dimension"
            )

    assert fine.sequence_a.length > fine.sequence_b.length
    assert fine.sequence_a.length > coarse.sequence_a.length
    assert fine.sequence_b.length > coarse.sequence_b.length


def test_normalized_scale_tracks_fragment_resolution_not_absolute_pixels() -> None:
    small_a = _rectangle(shape=(256, 256), top=40, left=50, height=150, width=100)
    small_b = _rectangle(shape=(256, 256), top=50, left=155, height=140, width=80)
    large_a = _rectangle(shape=(800, 800), top=120, left=150, height=450, width=300)
    large_b = _rectangle(shape=(800, 800), top=150, left=470, height=420, width=240)
    config = _config(window_scale_fractions=(0.1,))

    small = build_pair_candidates(small_a, small_b, PairDirection.B_RIGHT_OF_A, config)
    large = build_pair_candidates(large_a, large_b, PairDirection.B_RIGHT_OF_A, config)

    assert small.ok and large.ok
    small_candidate = small.candidates[0]
    large_candidate = large.candidates[0]
    assert small_candidate.sequence_a.resolved_window_px == pytest.approx(10.0)
    assert large_candidate.sequence_a.resolved_window_px == pytest.approx(30.0)
    assert small_candidate.sequence_b.resolved_window_px == pytest.approx(8.0)
    assert large_candidate.sequence_b.resolved_window_px == pytest.approx(24.0)
    assert small_candidate.patches_a.shape[1:] == large_candidate.patches_a.shape[1:]
    assert small_candidate.patches_b.shape[1:] == large_candidate.patches_b.shape[1:]


def test_local_patch_features_do_not_encode_full_canvas_translation() -> None:
    """The local branch must not inherit the synthetic placement shortcut."""

    config = _config(window_scale_fractions=(0.1, 0.2))
    original = build_pair_candidates(
        _rectangle(top=30, left=25, height=150, width=90),
        _rectangle(top=40, left=140, height=140, width=75),
        PairDirection.B_RIGHT_OF_A,
        config,
    )
    independently_translated = build_pair_candidates(
        _rectangle(top=60, left=70, height=150, width=90),
        _rectangle(top=80, left=110, height=140, width=75),
        PairDirection.B_RIGHT_OF_A,
        config,
    )

    assert original.ok and independently_translated.ok
    assert len(original.candidates) == len(independently_translated.candidates)
    for before, after in zip(
        original.candidates,
        independently_translated.candidates,
    ):
        assert before.candidate_id == after.candidate_id
        np.testing.assert_allclose(before.patches_a, after.patches_a, atol=1e-7)
        np.testing.assert_allclose(before.patches_b, after.patches_b, atol=1e-7)
        assert before.sequence_a.length == after.sequence_a.length
        assert before.sequence_b.length == after.sequence_b.length


def test_three_mask_geometry_channels_are_finite_canonical_and_read_only() -> None:
    result = build_pair_candidates(
        _rectangle(),
        _rectangle(top=38, left=145, height=160, width=80),
        PairDirection.B_RIGHT_OF_A,
        _config(window_scale_fractions=(0.15,)),
    )
    assert result.ok

    for sequence in (
        result.candidates[0].sequence_a,
        result.candidates[0].sequence_b,
    ):
        assert sequence.channels.dtype == np.float32
        assert sequence.channels.flags.writeable is False
        assert sequence.valid.flags.writeable is False
        assert np.all(np.isfinite(sequence.channels))
        mask_channel = sequence.channels[:, 0]
        distance_channel = sequence.channels[:, 1]
        gradient_channel = sequence.channels[:, 2]
        assert np.min(mask_channel) >= 0.0 and np.max(mask_channel) <= 1.0
        assert np.min(distance_channel) >= -1.0 and np.max(distance_channel) <= 1.0
        assert np.min(gradient_channel) >= 0.0 and np.max(gradient_channel) <= 1.0
        assert np.max(gradient_channel) > 0.0
        # Canonical patch rows increase toward fragment interior for both a
        # right-facing and b left-facing contour despite opposite image normals.
        split = mask_channel.shape[1] // 2
        assert float(np.mean(mask_channel[:, split:, :])) > float(
            np.mean(mask_channel[:, :split, :])
        )
        assert float(np.mean(distance_channel[:, split:, :])) > float(
            np.mean(distance_channel[:, :split, :])
        )
        for patch in sequence.patches:
            assert patch.valid_fraction >= 0.0
            assert patch.padding_fraction == pytest.approx(1.0 - patch.valid_fraction)


def test_explicit_dark_foreground_polarity_matches_bright_mask_geometry() -> None:
    bright_a = _rectangle(top=30, left=25, height=170, width=90)
    bright_b = _rectangle(top=45, left=140, height=145, width=75)
    bright = build_pair_candidates(
        bright_a,
        bright_b,
        PairDirection.B_RIGHT_OF_A,
        _config(window_scale_fractions=(0.1,)),
    )
    dark = build_pair_candidates(
        255 - bright_a,
        255 - bright_b,
        PairDirection.B_RIGHT_OF_A,
        _config(
            window_scale_fractions=(0.1,),
            foreground_polarity="dark",
        ),
    )

    assert bright.ok and dark.ok
    np.testing.assert_array_equal(
        bright.candidates[0].patches_a,
        dark.candidates[0].patches_a,
    )
    np.testing.assert_array_equal(
        bright.candidates[0].patches_b,
        dark.candidates[0].patches_b,
    )


def test_a_b_swap_is_exact_after_direction_inversion() -> None:
    mask_a = _rectangle(top=30, left=25, height=170, width=90)
    mask_b = _rectangle(top=45, left=140, height=145, width=75)
    config = _config()
    original = build_pair_candidates(
        mask_a,
        mask_b,
        PairDirection.B_RIGHT_OF_A,
        config,
    )
    swapped = build_pair_candidates(
        mask_b,
        mask_a,
        PairDirection.B_LEFT_OF_A,
        config,
    )

    assert original.ok and swapped.ok
    assert len(original.candidates) == len(swapped.candidates)
    for first, second in zip(original.candidates, swapped.candidates):
        assert second.direction is first.direction.inverse
        np.testing.assert_array_equal(first.patches_a, second.patches_b)
        np.testing.assert_array_equal(first.patches_b, second.patches_a)
        np.testing.assert_array_equal(first.valid_a, second.valid_b)
        np.testing.assert_array_equal(first.valid_b, second.valid_a)
        assert (
            first.sequence_a.resolved_window_px == second.sequence_b.resolved_window_px
        )
        assert (
            first.sequence_b.resolved_window_px == second.sequence_a.resolved_window_px
        )


@pytest.mark.parametrize(
    ("mask_a", "mask_b", "expected_status", "reason_part"),
    [
        (
            np.zeros((32, 32), dtype=np.uint8),
            _rectangle(),
            GeometryStatus.INVALID_CONTOUR,
            "constant_mask",
        ),
        (
            np.zeros((32, 32, 3), dtype=np.uint8),
            _rectangle(),
            GeometryStatus.INVALID_INPUT,
            "mask_must_be_nonempty_2d",
        ),
        (
            np.full((32, 32), np.nan, dtype=np.float32),
            _rectangle(),
            GeometryStatus.INVALID_INPUT,
            "nonfinite",
        ),
    ],
)
def test_invalid_masks_fail_closed_without_partial_candidates(
    mask_a, mask_b, expected_status, reason_part
) -> None:
    result = build_pair_candidates(mask_a, mask_b, config=_config())

    assert result.status is expected_status
    assert reason_part in result.failure_reason
    assert result.candidates == ()
    assert result.direction_groups == ()
    assert result.quality.candidate_count == 0


def test_invalid_direction_and_reserved_corrosion_fail_closed() -> None:
    invalid_direction = build_pair_candidates(
        _rectangle(), _rectangle(), "diagonal", _config()
    )
    corrosion = build_pair_candidates(
        _rectangle(),
        _rectangle(),
        config=replace(_config(), corrosion=CorrosionConfig(enabled=True)),
    )
    silent_corrosion = build_pair_candidates(
        _rectangle(),
        _rectangle(),
        config=replace(
            _config(),
            corrosion=CorrosionConfig(erosion_radius_fraction=0.01),
        ),
    )

    assert invalid_direction.status is GeometryStatus.INVALID_INPUT
    assert invalid_direction.candidates == ()
    assert corrosion.status is GeometryStatus.UNSUPPORTED_CONFIGURATION
    assert corrosion.failure_reason == (
        "controlled_corrosion_is_reserved_but_disabled_in_v0_2_clean"
    )
    assert corrosion.candidates == ()
    assert silent_corrosion.status is GeometryStatus.UNSUPPORTED_CONFIGURATION
    assert silent_corrosion.candidates == ()


def test_config_rejects_non_normalized_or_unsafe_window_settings() -> None:
    with pytest.raises(ValueError, match="window_scale_fractions"):
        CandidateBuilderConfig(window_scale_fractions=())
    with pytest.raises(ValueError, match="unique and increasing"):
        CandidateBuilderConfig(window_scale_fractions=(0.2, 0.1))
    with pytest.raises(ValueError, match="overlap_fraction"):
        CandidateBuilderConfig(overlap_fraction=1.0)
    with pytest.raises(ValueError, match="output_size"):
        CandidateBuilderConfig(output_size=(1, 16))
    with pytest.raises(ValueError, match="foreground_polarity"):
        CandidateBuilderConfig(foreground_polarity="auto")
