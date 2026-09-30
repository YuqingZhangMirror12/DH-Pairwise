"""Fixed GT seam gap post-analysis; synthetic masks, CPU, no predictions."""
from dataclasses import replace
from itertools import cycle
import json
from unittest.mock import patch

import numpy as np
import pytest

from staging.pairwise_v0_2.pairwise_data.rachel_seam_gap_diagnostics import seam_gap_diagnostics
from staging.pairwise_v0_2.pairwise_data import rachel_seam_gap_diagnostics as diagnostics
from staging.pairwise_v0_2.tests.test_rachel_gap_stress import adjacent_pair


def _retreat(clean, *, a=0, b=0):
    ma, mb = clean.mask_a.copy(), clean.mask_b.copy()
    if a:
        ma[0, 200:600, 400-a:400] = 0
    if b:
        mb[0, 150:550, 300:300+b] = 0
    # Deliberately keep old token metadata: this diagnostic must inspect only
    # stress masks, not depend on stress contour extraction or assignments.
    return replace(clean, mask_a=ma, mask_b=mb)


def test_zero_identity_is_exact_added_zero_and_inputs_unchanged():
    clean = adjacent_pair()
    snapshots = {key: value.copy() for key, value in vars(clean).items() if isinstance(value, np.ndarray)}
    result = seam_gap_diagnostics(clean, clean, include_samples=True)
    assert result["valid"] and result["zero_identity"]
    assert result["actual_seam_damaged"] is False
    assert result["added_gap"] == {"weighted_mean_px": 0., "weighted_p90_px": 0.}
    assert result["damage_added_ge_1px_arc_fraction"] == 0
    assert result["negative_added_gap_count"] == 0
    assert len(result["samples"]) <= 512
    assert all(row["added_gap_px"] == 0. for row in result["samples"] if row["valid"])
    assert result["support"]["coverage_fraction"] == pytest.approx(1.)
    for key, value in snapshots.items():
        np.testing.assert_array_equal(getattr(clean, key), value)
    json.dumps(result, allow_nan=False)


def test_nonseam_erosion_does_not_define_damaged_subset():
    clean = adjacent_pair()
    ma = clean.mask_a.copy()
    ma[0, 200:600, 200:202] = 0  # A's LEFT outer edge, not its RIGHT seam.
    student = replace(clean, mask_a=ma)
    result = seam_gap_diagnostics(clean, student)
    assert result["valid"] and not result["zero_identity"]
    assert result["actual_seam_damaged"] is False
    assert result["added_gap"]["weighted_mean_px"] == 0
    assert result["damage_added_ge_1px_arc_fraction"] == 0
    assert result["removed_pixels_touched_by_supported_rays_a"] == 0
    assert "samples" not in result


def test_two_pixel_retreat_each_side_gives_four_pixel_added_separation_at_gt():
    clean = adjacent_pair()
    student = _retreat(clean, a=2, b=2)
    result = seam_gap_diagnostics(clean, student, include_samples=True)
    assert result["valid"] and result["actual_seam_damaged"] is True
    assert result["clean_gap"]["weighted_mean_px"] == pytest.approx(0.)
    assert result["stress_gap"]["weighted_mean_px"] == pytest.approx(4.)
    assert result["added_gap"] == {"weighted_mean_px": 4., "weighted_p90_px": 4.}
    assert result["damage_added_ge_1px_arc_fraction"] == pytest.approx(1.)
    assert result["removed_pixels_touched_by_supported_rays_a"] > 0
    assert result["removed_pixels_touched_by_supported_rays_b"] > 0
    assert all(row["added_gap_px"] == 4. for row in result["samples"] if row["valid"])
    assert not result["prediction_used"] and not result["model_input"]
    assert not result["physical_gap_width_ground_truth"]
    np.testing.assert_array_equal(clean.translation_a_to_b_rc, student.translation_a_to_b_rc)
    assert clean.label == student.label == 1


def _swap(sample):
    return replace(sample, fragment_a_token=sample.fragment_b_token, fragment_b_token=sample.fragment_a_token,
        mask_a=sample.mask_b, mask_b=sample.mask_a, points_rc_a=sample.points_rc_b, points_rc_b=sample.points_rc_a,
        contour_valid_a=sample.contour_valid_b, contour_valid_b=sample.contour_valid_a,
        target_a=sample.target_b, target_b=sample.target_a,
        translation_a_to_b_rc=-sample.translation_a_to_b_rc,
        translation_a_to_b_xy_cartesian=-sample.translation_a_to_b_xy_cartesian)


def test_b_placement_minus_t_and_swapped_endpoint_sign_are_consistent():
    clean = adjacent_pair()
    # B's local coordinates differ by [-50,-100]; omitting or reversing +t_GT
    # for B ray probes cannot find the correct seam in this fixture.
    student = _retreat(clean, a=2, b=2)
    original = seam_gap_diagnostics(clean, student)
    swapped = seam_gap_diagnostics(_swap(clean), _swap(student))
    assert original["valid"] and swapped["valid"]
    assert swapped["added_gap"]["weighted_mean_px"] == pytest.approx(4.)
    assert swapped["actual_seam_damaged"] is True
    wrong_gt = replace(clean, translation_a_to_b_rc=-clean.translation_a_to_b_rc,
        translation_a_to_b_xy_cartesian=-clean.translation_a_to_b_xy_cartesian)
    assert not seam_gap_diagnostics(wrong_gt, wrong_gt)["valid"]


def test_multiple_line_crossings_and_unbracketed_endpoints_are_explicitly_unmeasured():
    clean = adjacent_pair()
    ma = clean.mask_a.copy()
    ma[0, 200:600, 395:396] = 0  # Paper then hole then paper then outside.
    result = seam_gap_diagnostics(clean, replace(clean, mask_a=ma), include_samples=True)
    assert not result["valid"] and result["actual_seam_damaged"] is None
    assert result["invalid_reason"] == "no_unambiguous_line_measurements"
    assert any("multiple_line_crossings" in key for key in result["support"]["invalid_token_reasons"])
    assert result["support"]["measured_arc_px"] == 0
    assert result["support"]["coverage_fraction"] == 0
    beyond_range = seam_gap_diagnostics(clean, _retreat(clean, a=9))
    assert not beyond_range["valid"] and beyond_range["actual_seam_damaged"] is None
    assert any("within_range" in key or "endpoints" in key
               for key in beyond_range["support"]["invalid_token_reasons"])


def test_subset_denominator_is_clean_supported_arc_not_surviving_measurements():
    clean = adjacent_pair()
    ma = clean.mask_a.copy()
    # Make almost all original supported seam rays unmeasured, while a short
    # remaining 20px arc has a 2px retreat. It must not become 100% seam damage.
    ma[0, 200:600, 395:396] = 0
    ma[0, 300:320, 395:396] = clean.mask_a[0, 300:320, 395:396]
    ma[0, 300:320, 398:400] = 0
    result = seam_gap_diagnostics(clean, replace(clean, mask_a=ma))
    assert result["valid"] and result["support"]["measured_arc_px"] >= 8
    assert result["damage_fraction_of_measured_arc"] == pytest.approx(1.)
    assert result["damage_added_ge_1px_arc_fraction"] < .10
    assert result["actual_seam_damaged"] is False
    assert result["support"]["coverage_fraction"] < .10


def test_signed_digital_overlap_is_retained_not_clipped():
    clean = adjacent_pair()
    # Move B one pixel left at GT; a small overlap is deliberately preserved as
    # signed clean separation rather than being mislabeled a zero-width gap.
    gt = clean.translation_a_to_b_rc + np.asarray([0., 1.], np.float32)
    overlapped = replace(clean, translation_a_to_b_rc=gt,
        translation_a_to_b_xy_cartesian=np.asarray([gt[1], -gt[0]], np.float32))
    result = seam_gap_diagnostics(overlapped, _retreat(overlapped, a=2, b=2))
    assert result["valid"]
    assert result["clean_gap"]["weighted_mean_px"] == pytest.approx(-1.)
    assert result["stress_gap"]["weighted_mean_px"] == pytest.approx(3.)
    assert result["added_gap"]["weighted_mean_px"] == pytest.approx(4.)
    assert result["clean_negative_overlap_arc_fraction"] == pytest.approx(1.)


def test_no_gt_seam_is_unknown_and_identity_or_subset_contract_changes_rejected():
    clean = adjacent_pair()
    negative = replace(clean, label=np.float32(0), translation_valid=np.bool_(False))
    result = seam_gap_diagnostics(negative, negative)
    assert result["actual_seam_damaged"] is None and result["invalid_reason"] == "negative_pair_has_no_gt_seam"
    no_seam = replace(clean, target_a=np.full_like(clean.target_a, -1), target_b=np.full_like(clean.target_b, -1))
    assert seam_gap_diagnostics(no_seam, no_seam)["actual_seam_damaged"] is None
    with pytest.raises(ValueError, match="identity"):
        seam_gap_diagnostics(clean, replace(clean, pair_id="other"))
    with pytest.raises(ValueError, match="GT placement"):
        seam_gap_diagnostics(clean, replace(clean, translation_a_to_b_rc=clean.translation_a_to_b_rc + 1))
    added = clean.mask_a.copy()
    added[0, 200, 500] = 1
    with pytest.raises(ValueError, match="inward-only"):
        seam_gap_diagnostics(clean, replace(clean, mask_a=added))


def test_repeated_calls_are_deterministic_and_do_not_accept_prediction_arguments():
    clean = adjacent_pair()
    student = _retreat(clean, a=2)
    assert seam_gap_diagnostics(clean, student, include_samples=True) == seam_gap_diagnostics(clean, student, include_samples=True)
    with pytest.raises(TypeError):
        seam_gap_diagnostics(clean, student, prediction=[1, 2])


def test_endpoint_instability_and_negative_added_measurement_are_not_silently_repaired():
    profile = np.zeros(len(diagnostics.PROBE_OFFSETS_PX), bool)
    profile[0] = True  # A crossing too near the negative endpoint to be stable.
    position, reason = diagnostics._single_boundary(profile, exiting=True)
    assert position is None and reason == "unstable_or_wrong_direction_endpoints"
    clean = adjacent_pair()
    # An intentionally inconsistent numerical boundary fixture exercises the
    # fail-closed guard; real inward-subset single-crossing profiles are monotone.
    with patch.object(diagnostics, "_single_boundary", side_effect=cycle(
            [(0., None), (0., None), (1., None), (0., None)])):
        result = seam_gap_diagnostics(clean, clean, include_samples=True)
    assert result["negative_added_gap_count"] > 0
    assert result["actual_seam_damaged"] is None
    assert result["added_gap"]["weighted_mean_px"] is None
    assert all(row["added_gap_px"] == -1. and not row["valid"] for row in result["samples"])
