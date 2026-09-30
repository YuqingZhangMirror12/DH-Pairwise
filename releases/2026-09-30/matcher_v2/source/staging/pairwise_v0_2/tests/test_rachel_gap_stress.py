"""TEST gap-stress geometry and metric convention fixtures, entirely CPU/local."""
from dataclasses import replace
import json
from unittest.mock import patch

import numpy as np
import pytest

from staging.pairwise_v0_2.pairwise_data import rachel_gap_stress as stress
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import (
    extract_ordered_outer_contour, recover_mutual_contour_correspondences,
)
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairSample


class TestFixtureDataset:
    __test__ = False
    split = "test"

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def adjacent_pair():
    a, b = np.zeros((800, 800), bool), np.zeros((800, 800), bool)
    a[200:600, 200:400] = True
    b[150:550, 300:500] = True
    pa, va = extract_ordered_outer_contour(a)
    pb, vb = extract_ordered_outer_contour(b)
    gt = np.asarray([-50., -100.], np.float32)
    pairs = recover_mutual_contour_correspondences(pa, pb - gt, max_distance_px=3.)
    # Clear straight portion avoids ambiguous corner normals in this sign fixture.
    pairs = pairs[(pa[pairs[:, 0], 0] > 220) & (pa[pairs[:, 0], 0] < 580)]
    ta, tb = np.full(len(pa), -1, np.int64), np.full(len(pb), -1, np.int64)
    ta[pairs[:, 0]], tb[pairs[:, 1]] = pairs[:, 1], pairs[:, 0]
    assert len(pairs) > 100
    return RachelPairSample("fixture-positive", "a", "b", a[None].astype(np.float32),
        b[None].astype(np.float32), np.zeros((1, 128, 128), np.float32), np.zeros((1, 128, 128), np.float32),
        pa, pb, va, vb, ta, tb, np.float32(1), gt,
        np.asarray([gt[1], -gt[0]], np.float32), np.bool_(True))


def test_depth_zero_is_exact_original_object_and_six_inputs_only():
    clean = adjacent_pair()
    original = {key: value.copy() for key, value in vars(clean).items() if isinstance(value, np.ndarray)}
    with patch.object(stress, "extract_ordered_outer_contour", side_effect=AssertionError("zero contour rebuild")):
        item = stress.RachelGapStressDataset(TestFixtureDataset([clean]), 0)[0]
    assert item.student is clean and not item.report["changed_pair"]
    assert item.student.target_a is clean.target_a  # Zero's old assignments untouched.
    inputs = stress.gap_stress_model_inputs(item.student)
    assert len(inputs) == 6
    assert all(left is right for left, right in zip(inputs, (clean.mask_a, clean.mask_b,
        clean.points_rc_a, clean.points_rc_b, clean.contour_valid_a, clean.contour_valid_b)))
    for key, value in original.items():
        np.testing.assert_array_equal(getattr(clean, key), value)
    json.dumps(item.report, allow_nan=False)


@pytest.mark.parametrize("depth", [2, 4])
def test_weathering_inward_frame_gt_and_deterministic_model_shared_geometry(depth, tmp_path):
    clean = adjacent_pair()
    dataset = stress.RachelGapStressDataset(TestFixtureDataset([clean]), depth, cache_dir=tmp_path)
    first = dataset[0]
    second = stress.RachelGapStressDataset(TestFixtureDataset([clean]), depth, cache_dir=tmp_path)[0]
    assert first.report == second.report and first.report["changed_pair"]
    for side in "ab":
        mask, old = getattr(first.student, "mask_" + side), getattr(clean, "mask_" + side)
        assert mask.shape == (1, 800, 800) and not np.any((mask > 0) & (old == 0))
        np.testing.assert_array_equal(mask, getattr(second.student, "mask_" + side))
        assert np.all(getattr(first.student, "target_" + side) == -2)
        assert len(getattr(first.student, "points_rc_" + side)) <= 512
        assert first.report["side_" + side]["actual_applied"]
        assert first.report["side_" + side]["removed_area_px"] == int(old.sum() - mask.sum())
    assert first.student.label == clean.label and first.student.translation_valid == clean.translation_valid
    assert first.student.translation_a_to_b_rc is clean.translation_a_to_b_rc
    assert first.student.translation_a_to_b_xy_cartesian is clean.translation_a_to_b_xy_cartesian
    clean_metrics = stress.clean_gt_gap_metrics(clean)
    np.testing.assert_array_equal(first.metrics.normals_rc, clean_metrics.normals_rc)
    np.testing.assert_array_equal(first.metrics.arc_weights_px, clean_metrics.arc_weights_px)
    assert first.metrics.support == clean_metrics.support
    assert not first.report["pair_fallback"] and not first.report["metric_sidecar_model_input"]


def test_geometry_independent_of_label_or_partner_and_cache_has_no_pair_supervision(tmp_path):
    clean = adjacent_pair()
    negative = replace(clean, pair_id="negative", label=np.float32(0), translation_valid=np.bool_(False),
        target_a=np.full_like(clean.target_a, -1), target_b=np.full_like(clean.target_b, -1))
    dataset = stress.RachelGapStressDataset(TestFixtureDataset([clean, negative]), 2, cache_dir=tmp_path)
    positive, other = dataset[0], dataset[1]
    for side in "ab":
        np.testing.assert_array_equal(getattr(positive.student, "mask_" + side), getattr(other.student, "mask_" + side))
        assert positive.report["side_" + side] == other.report["side_" + side]
    assert not other.metrics.support["valid"] and other.metrics.translation_gt_rc is None
    fresh = stress.RachelGapStressDataset(TestFixtureDataset([clean]), 2, cache_dir=tmp_path)
    with patch.object(stress, "weather_fragment_edges", side_effect=AssertionError("cached geometry regenerated")):
        fresh[0]
    assert fresh.cache_info()["disk_hits"] == 2
    for path in tmp_path.rglob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            assert set(data.files) == {"packed_mask", "points", "valid"}
    for path in tmp_path.rglob("*.json"):
        content = json.loads(path.read_text())
        assert set(content) == {"schema_version", "cache_key", "report"}
        assert not any(key in content["report"] for key in ("label", "target_a", "translation_gt_rc", "normals_rc"))


def test_topology_skip_is_reported_keeps_original_side_and_does_not_drop_pair():
    clean = adjacent_pair()
    holed = clean.mask_a.copy()
    holed[0, 300:310, 250:260] = 0
    clean = replace(clean, mask_a=holed)
    item = stress.RachelGapStressDataset(TestFixtureDataset([clean]), 2)[0]
    assert item.report["side_a"]["skipped"]
    assert item.report["side_a"]["skip_reason"] == "input_mask_has_holes"
    assert item.report["side_a"]["requested_attempt"]
    assert not item.report["changed_a"] and not item.report["side_a"]["actual_applied"]
    assert item.report["side_a"]["removed_area_px"] == 0
    assert item.student.mask_a is clean.mask_a and item.student.points_rc_a is clean.points_rc_a
    assert np.all(item.student.target_a == -2)
    assert item.student.label == 1 and not item.report["pair_fallback"]


def test_closing_sign_and_projection_components_with_unchanged_support():
    clean = adjacent_pair()
    metric = stress.clean_gt_gap_metrics(clean)
    assert metric.support["valid"] and metric.support["coverage_fraction"] > .99
    assert metric.support["supported_arc_px"] <= metric.support["seam_arc_px"] <= metric.support["perimeter_px"]
    np.testing.assert_allclose(metric.normals_rc, np.tile([0., 1.], (len(metric.normals_rc), 1)), atol=1e-7)
    equal = stress.closing_direction_bias(clean.translation_a_to_b_rc, metric)
    closing = stress.closing_direction_bias(clean.translation_a_to_b_rc + [0, 5], metric)
    opening = stress.closing_direction_bias(clean.translation_a_to_b_rc - [0, 5], metric)
    tangent = stress.closing_direction_bias(clean.translation_a_to_b_rc + [5, 0], metric)
    assert equal["signed_closing_bias_px"] == 0
    assert closing["signed_closing_bias_px"] == pytest.approx(5)
    assert opening["signed_closing_bias_px"] == pytest.approx(-5)
    assert tangent["signed_closing_bias_px"] == pytest.approx(0)
    assert closing["positive_closing_component_px"] == pytest.approx(5)
    assert opening["negative_opening_component_px"] == pytest.approx(-5)
    assert closing["closing_arc_fraction"] == pytest.approx(1)
    assert opening["opening_arc_fraction"] == pytest.approx(1)
    assert closing["support"] == opening["support"] == equal["support"]
    assert not closing["gap_width_ground_truth"]
    json.dumps(closing, allow_nan=False)


def test_normal_sign_survives_reversed_contour_order_and_endpoint_swap():
    clean = adjacent_pair()
    order = np.arange(len(clean.points_rc_a))[::-1]
    inverse = np.argsort(order)
    tb = clean.target_b.copy()
    tb[tb >= 0] = inverse[tb[tb >= 0]]
    reversed_a = replace(clean, points_rc_a=clean.points_rc_a[order], contour_valid_a=clean.contour_valid_a[order],
                         target_a=clean.target_a[order], target_b=tb)
    metric = stress.clean_gt_gap_metrics(reversed_a)
    value = stress.closing_direction_bias(clean.translation_a_to_b_rc + [0, 5], metric)
    assert value["signed_closing_bias_px"] == pytest.approx(5)
    swapped = replace(clean, fragment_a_token=clean.fragment_b_token, fragment_b_token=clean.fragment_a_token,
        mask_a=clean.mask_b, mask_b=clean.mask_a, points_rc_a=clean.points_rc_b, points_rc_b=clean.points_rc_a,
        contour_valid_a=clean.contour_valid_b, contour_valid_b=clean.contour_valid_a,
        target_a=clean.target_b, target_b=clean.target_a,
        translation_a_to_b_rc=-clean.translation_a_to_b_rc,
        translation_a_to_b_xy_cartesian=-clean.translation_a_to_b_xy_cartesian)
    value = stress.closing_direction_bias(-clean.translation_a_to_b_rc - [0, 5], stress.clean_gt_gap_metrics(swapped))
    assert value["signed_closing_bias_px"] == pytest.approx(5)


def test_unsupported_gt_and_missing_predictions_explicitly_invalid():
    clean = adjacent_pair()
    wrong_gt = replace(clean, translation_a_to_b_rc=clean.translation_a_to_b_rc + [0, 100])
    metric = stress.clean_gt_gap_metrics(wrong_gt)
    assert not metric.support["valid"]
    value = stress.closing_direction_bias(wrong_gt.translation_a_to_b_rc, metric)
    assert value["signed_closing_bias_px"] is None
    metric = stress.clean_gt_gap_metrics(clean)
    assert stress.closing_direction_bias(None, metric)["invalid_reason"] == "missing_predicted_translation"
    assert not stress.closing_direction_bias([np.nan, 1], metric)["valid"]


def test_test_only_gate_and_frozen_conditions():
    for source in ([], type("TrainFixture", (), {"split": "train"})(), type("ValFixture", (), {"split": "val"})()):
        with pytest.raises(ValueError, match="TEST-only"):
            stress.RachelGapStressDataset(source)
    for depth in (1, 3, 8, True):
        with pytest.raises(ValueError, match="0/2/4"):
            stress.RachelGapStressDataset(TestFixtureDataset([]), depth)
