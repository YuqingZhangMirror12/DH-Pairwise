"""Single-mask inward weathering geometry contracts; no pair/model/target IO."""
import json
from unittest.mock import patch

import numpy as np
import pytest
from scipy import ndimage

from staging.pairwise_v0_2.pairwise_data import rachel_edge_weathering as weather


def rectangle():
    mask = np.zeros((800, 800), bool)
    mask[180:620, 180:620] = True
    return mask


@pytest.mark.parametrize("depth", [2., 4.])
def test_inward_subset_area_topology_and_json_report(depth):
    original = rectangle()
    changed, report = weather.weather_fragment_edges(original, seed=260909, fragment_id="fragment-A",
        config=weather.EdgeWeatheringConfig(max_depth_px=depth), return_depth=True)
    assert changed.dtype == np.bool_ and changed.shape == (800, 800)
    assert not np.any(changed & ~original)
    assert report["applied"] and report["contour_changed"]
    assert report["pixels_added"] == 0 and report["frame_unchanged"]
    assert report["removed_area_px"] == int(original.sum() - changed.sum())
    assert 0 < report["removed_fraction"] <= .03
    assert 0 <= report["max_sampled_depth_px"] <= depth
    assert .2 <= report["requested_coverage_fraction"] <= .5
    assert 16 <= report["correlation_support_px"] <= 64
    assert ndimage.label(changed)[1] == 1
    assert np.array_equal(ndimage.binary_fill_holes(changed), changed)
    profile = report["depth_diagnostics"]
    assert len(profile["arclength_px"]) == len(profile["depth_px"]) == len(profile["contour_points_rc"])
    assert np.all(np.diff(profile["arclength_px"]) > 0)
    assert not report["correspondence_targets_generated"]
    json.dumps(report, allow_nan=False)


def test_deterministic_original_immutable_and_fragment_local_seed():
    original = rectangle()
    snapshot = original.copy()
    original.setflags(write=False)
    np.random.seed(171)
    global_before = np.random.get_state()
    first, report1 = weather.weather_fragment_edges(original, seed=17, fragment_id="a")
    second, report2 = weather.weather_fragment_edges(original, seed=17, fragment_id="a")
    other, _ = weather.weather_fragment_edges(original, seed=17, fragment_id="b")
    assert np.array_equal(first, second) and report1 == report2
    assert not np.array_equal(first, other)
    assert np.array_equal(original, snapshot)
    assert not np.shares_memory(first, original)
    assert np.array_equal(np.random.get_state()[1], global_before[1])


def test_zero_strength_is_identity_without_contour_work():
    original = rectangle()
    with patch.object(weather, "extract_ordered_outer_contour", side_effect=AssertionError("unnecessary contour work")):
        result, report = weather.weather_fragment_edges(original, seed=1, fragment_id="a",
            config=weather.EdgeWeatheringConfig(max_depth_px=0))
    assert np.array_equal(original, result)
    assert report["status"] == "identity" and report["identity_reason"] == "zero_strength"
    assert report["removed_area_px"] == 0 and not report["contour_changed"]


def test_disconnected_proposal_returns_original_without_largest_component_repair():
    original = np.zeros((800, 800), bool)
    original[100:300, 100:300] = True
    original[100:300, 310:510] = True
    original[199:200, 300:310] = True  # A thin connected neck.

    def all_edge_depth(points, rng, config):
        return np.arange(len(points), dtype=float), np.full(len(points), 2.), {}

    with patch.object(weather, "_arc_depth", side_effect=all_edge_depth):
        result, report = weather.weather_fragment_edges(original, seed=1, fragment_id="thin-neck")
    assert np.array_equal(result, original)
    assert report["skipped"] and report["skip_reason"] == "proposal_disconnected_mask"
    assert report["proposed_removed_area_px"] > 0
    assert report["removed_area_px"] == 0 and not report["contour_changed"]


def test_area_guard_skips_instead_of_rescaling_depth_or_repairing():
    original = rectangle()
    result, report = weather.weather_fragment_edges(original, seed=1, fragment_id="a",
        config=weather.EdgeWeatheringConfig(max_depth_px=4, max_area_loss_fraction=0))
    assert np.array_equal(result, original)
    assert report["skip_reason"] == "area_loss_exceeds_limit"
    assert report["proposed_removed_area_px"] > 0 and report["removed_area_px"] == 0


def test_tiny_or_invalid_input_returns_explicit_skip_and_never_empty_proposal():
    tiny = np.zeros((800, 800), bool)
    tiny[400, 400] = True
    result, report = weather.weather_fragment_edges(tiny, seed=2, fragment_id="tiny")
    assert np.array_equal(result, tiny)
    assert report["skipped"] and report["skip_reason"] == "input_contour_too_small_or_invalid"
    holed = rectangle()
    holed[390:410, 390:410] = False
    result, report = weather.weather_fragment_edges(holed, seed=2, fragment_id="hole")
    assert np.array_equal(result, holed) and report["skip_reason"] == "input_mask_has_holes"
    with pytest.raises(ValueError, match="bool"):
        weather.weather_fragment_edges(tiny.astype(np.uint8), seed=1, fragment_id="a")
