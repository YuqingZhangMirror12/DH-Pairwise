import json

import numpy as np
import pytest

from staging.pairwise_v0_1.baselines.b1_contours import (
    ContourPreprocessConfig,
    ForegroundPolarity,
    preprocess_mask_contour_only,
)
from staging.pairwise_v0_2.geometry import (
    CandidateBuilderConfig,
    PairDirection,
    build_pair_candidates,
)
from staging.pairwise_v0_2.preflight.audit_candidate_arc_recall_v0_2 import (
    _aggregate,
    derive_weak_contact_proxy,
    diagnose_pixel_cell_topology,
    evaluate_candidate_output,
    overlap_slice,
    patch_sequence_contact_coverage,
)


def _rectangle(shape, top, left, height, width):
    mask = np.zeros(shape, dtype=bool)
    mask[top : top + height, left : left + width] = True
    return mask


def test_overlap_slices_are_exhaustive_only_through_frozen_maximum():
    assert [overlap_slice(value) for value in (0, 1, 2, 4, 5, 8, 9, 16, 17, 32)] == [
        "0",
        "1",
        "2-4",
        "2-4",
        "5-8",
        "5-8",
        "9-16",
        "9-16",
        "17-32",
        "17-32",
    ]
    with pytest.raises(ValueError, match="through 32"):
        overlap_slice(33)


def test_contact_proxy_reproduces_dilation_and_direction_without_model():
    left = _rectangle((80, 100), 15, 10, 50, 30)
    right = _rectangle((80, 100), 15, 45, 50, 30)
    proxy = derive_weak_contact_proxy(left, right)

    assert proxy.raw_overlap_pixels == 0
    assert proxy.legacy_dilated_overlap_pixels == 250
    assert proxy.contact_mode == "disjoint_dilation_proxy"
    assert proxy.rough_direction is PairDirection.B_RIGHT_OF_A
    assert proxy.rough_direction_status == "unambiguous"
    # The symmetric weak proxy also retains the four top/bottom boundary
    # pixels on either side that fall inside the Chebyshev dilation radius.
    assert len(proxy.support_a_rc) == 58
    assert len(proxy.support_b_rc) == 58
    assert not proxy.support_a_rc.flags.writeable
    assert not proxy.support_b_rc.flags.writeable


def test_label_blind_four_direction_output_covers_simple_contact():
    left = _rectangle((100, 120), 20, 15, 60, 35)
    right = _rectangle((100, 120), 20, 55, 60, 35)
    config = CandidateBuilderConfig(
        min_run_length_fraction=0.0,
        min_run_length_px=2.0,
        window_scale_fractions=(0.15, 0.30),
        window_min_px=2.0,
        output_size=(10, 10),
    )
    proxy = derive_weak_contact_proxy(left, right)
    result = build_pair_candidates(left, right, direction_b_wrt_a=None, config=config)
    evaluation = evaluate_candidate_output(result, proxy, config)

    assert result.ok
    assert result.quality.requested_directions == tuple(
        direction.value for direction in PairDirection
    )
    assert evaluation["all_four_directions_emitted"]
    assert evaluation["rough_primary_direction_emitted"]
    assert evaluation["rough_primary_facing_side_run_pair_hit"]
    assert evaluation["rough_primary_candidate_patch_pair_hit"]
    assert evaluation["any_direction_candidate_patch_pair_hit"]
    assert np.all(evaluation["covered_all_a"])
    assert np.all(evaluation["covered_all_b"])

    candidate = result.candidates_for_direction(PairDirection.B_RIGHT_OF_A)[0]
    covered = patch_sequence_contact_coverage(
        candidate.sequence_a, proxy.support_a_rc
    )
    assert covered.dtype == np.bool_
    assert not covered.flags.writeable
    assert np.any(covered)


def test_topology_diagnostic_identifies_checkerboard_saddle_without_ids():
    # The center diagonal cells remain in one 4-connected component through
    # the left/bottom path, but their pixel-cell edges branch at one vertex.
    mask = np.asarray(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 0],
        ],
        dtype=bool,
    )
    diagnostic = diagnose_pixel_cell_topology(mask)
    assert diagnostic["checkerboard_saddle_vertex_count"] == 1
    assert sum(diagnostic["checkerboard_saddle_context_counts"].values()) == 1
    contour = preprocess_mask_contour_only(
        mask.astype(np.uint8) * 255,
        ContourPreprocessConfig(
            polarity=ForegroundPolarity.BRIGHT,
            min_component_pixels=1,
            min_contour_points=4,
        ),
    )
    assert not contour.ok
    assert contour.failure_reason == "non_manifold_boundary"
    encoded = json.dumps(diagnostic, allow_nan=False)
    assert "fragment_id" not in encoded
    assert "group_id" not in encoded


def test_aggregate_keeps_failed_pairs_in_candidate_recall_denominators():
    base = {
        "generator_family": "source",
        "fragment_count": 2,
        "overlap_slice": "0",
        "support_pixels_a": 10,
        "support_pixels_b": 10,
        "rough_direction_eligible": True,
        "rough_direction_unambiguous": True,
        "rough_primary_direction_emitted": True,
        "rough_primary_facing_side_run_pair_hit": True,
        "rough_primary_candidate_patch_pair_hit": True,
        "any_direction_facing_side_run_pair_hit": True,
        "any_direction_candidate_patch_pair_hit": True,
        "all_four_directions_emitted": True,
        "contact_spans_multiple_hit_runs": False,
        "all_direction_contact_recall_combined": 1.0,
        "all_direction_contact_recall_min_fragment": 1.0,
        "rough_primary_contact_recall_combined": 1.0,
        "best_single_run_contact_recall_combined": 1.0,
        "multi_run_union_gain_combined": 0.0,
        "covered_all_pixels_a": 10,
        "covered_all_pixels_b": 10,
        "scale_0_candidate_patch_pair_hit": True,
        "scale_0_contact_recall_combined": 1.0,
    }
    passed = dict(
        base,
        _group_id="g1",
        geometry_success=True,
        geometry_status="ok",
        geometry_failure_reason=None,
        _non_manifold_topology_diagnostic=None,
    )
    failed = dict(
        base,
        _group_id="g2",
        geometry_success=False,
        geometry_status="invalid_contour",
        geometry_failure_reason="a:non_manifold_boundary",
        _non_manifold_topology_diagnostic={
            "checkerboard_saddle_vertex_count": 2,
            "checkerboard_saddle_context_counts": {
                "two_exterior_background_channels": 2
            },
            "interior_hole_component_count_4": 0,
            "interior_hole_component_count_8": 0,
        },
    )
    for key in (
        "rough_primary_direction_emitted",
        "rough_primary_facing_side_run_pair_hit",
        "rough_primary_candidate_patch_pair_hit",
        "any_direction_facing_side_run_pair_hit",
        "any_direction_candidate_patch_pair_hit",
        "all_four_directions_emitted",
        "scale_0_candidate_patch_pair_hit",
    ):
        failed[key] = False
    for key in (
        "all_direction_contact_recall_combined",
        "all_direction_contact_recall_min_fragment",
        "rough_primary_contact_recall_combined",
        "best_single_run_contact_recall_combined",
        "scale_0_contact_recall_combined",
    ):
        failed[key] = 0.0
    failed["covered_all_pixels_a"] = 0
    failed["covered_all_pixels_b"] = 0

    summary = _aggregate([passed, failed], scale_count=1)
    assert summary["rates"]["geometry_success"]["row_rate"] == pytest.approx(0.5)
    assert summary["rates"]["any_direction_candidate_patch_pair_hit"][
        "row_rate"
    ] == pytest.approx(0.5)
    assert summary["contact_arc_recall"][
        "all_direction_contact_recall_combined"
    ]["row_mean"] == pytest.approx(0.5)
    assert summary["conditional_on_geometry_success"][
        "any_direction_candidate_patch_pair_hit"
    ]["row_rate"] == pytest.approx(1.0)
    assert summary["conditional_on_geometry_success"][
        "all_direction_contact_recall_combined"
    ]["row_mean"] == pytest.approx(1.0)
    assert summary["non_manifold_boundary_topology_diagnostic"] == {
        "denominator_failed_fragments": 1,
        "failed_fragments_with_checkerboard_saddle": 1,
        "checkerboard_saddle_vertices_total": 2,
        "checkerboard_saddle_vertices_minimum": 2,
        "checkerboard_saddle_vertices_median": 2.0,
        "checkerboard_saddle_vertices_maximum": 2,
        "checkerboard_saddle_context_counts": {
            "two_exterior_background_channels": 2
        },
        "failed_fragments_with_4_connected_interior_hole": 0,
        "failed_fragments_with_8_connected_interior_hole": 0,
    }
    json.dumps(summary, allow_nan=False)
