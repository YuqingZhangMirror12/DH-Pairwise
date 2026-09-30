"""Focused E1 wrapper fixtures; no source files, model, GPU or pair-GT discovery."""
from dataclasses import replace
import json
from unittest.mock import patch

import numpy as np
import pytest

from staging.pairwise_v0_2.pairwise_data import rachel_weathered_dataset as weather
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairSample


def sample(label=True):
    mask = np.zeros((800, 800), bool)
    mask[180:620, 180:620] = True
    points, valid = extract_ordered_outer_contour(mask, cap=512, smoothing_sigma=3.)
    targets = np.full(len(points), -1, np.int64)
    if label:
        targets[:128] = np.arange(128)
    return RachelPairSample(pair_id="p" if label else "n", fragment_a_token="fragment-A",
        fragment_b_token="fragment-B", mask_a=mask[None].astype(np.float32), mask_b=mask[None].astype(np.float32),
        coarse_mask_a=np.zeros((1, 128, 128), np.float32), coarse_mask_b=np.zeros((1, 128, 128), np.float32),
        points_rc_a=points, points_rc_b=points, contour_valid_a=valid, contour_valid_b=valid,
        target_a=targets.copy(), target_b=targets.copy(), label=np.float32(label),
        translation_a_to_b_rc=np.asarray([9., -7.], np.float32),
        translation_a_to_b_xy_cartesian=np.asarray([-7., -9.], np.float32), translation_valid=np.bool_(label))


def test_clean_exact_identity_and_original_unmodified():
    clean = sample()
    snapshots = {name: value.copy() for name, value in vars(clean).items() if isinstance(value, np.ndarray)}
    dataset = weather.RachelWeatheredDataset([clean], clean_probability=1., mild_probability=0.)
    with patch.object(weather, "source_arc_ancestry", side_effect=AssertionError("clean ancestry rebuilt")):
        returned, report = dataset[0]
    assert returned is clean
    assert report["pose_supervision_enabled"] and not report["changed_pair"]
    assert report["tier"] == {"a": "clean", "b": "clean"}
    assert report["inherited_match_count"] == 128
    assert all(np.array_equal(getattr(clean, key), before) for key, before in snapshots.items())


def test_weathered_positive_inherits_reciprocal_targets_without_reclosing_gt_gap():
    clean = sample()
    before = clean.mask_a.copy()
    dataset = weather.RachelWeatheredDataset([clean], clean_probability=0., mild_probability=1.)
    # The deliberately unrelated translation in this unit fixture must NOT be
    # used for corrupted A-B nearest-neighbour target generation/residual gating.
    result, report = dataset[0]
    assert report["changed_pair"] and report["inherited_match_count"] > 0
    assert report["fallback_reason"] is None and not report["pose_supervision_enabled"]
    assert result.label == clean.label and result.translation_valid == clean.translation_valid
    np.testing.assert_array_equal(result.translation_a_to_b_rc, clean.translation_a_to_b_rc)
    np.testing.assert_array_equal(result.translation_a_to_b_xy_cartesian, clean.translation_a_to_b_xy_cartesian)
    for i in np.flatnonzero(result.target_a >= 0):
        assert result.target_b[result.target_a[i]] == i
    assert result.target_a.dtype == np.int64 and result.target_b.dtype == np.int64
    assert not np.any(result.mask_a.astype(bool) & ~clean.mask_a.astype(bool))
    np.testing.assert_array_equal(clean.mask_a, before)
    assert result.points_rc_a.shape[0] <= 512 and result.coarse_mask_a.shape == (1, 128, 128)
    assert not report["cross_fragment_geometry_used_for_targets"]
    json.dumps(report, allow_nan=False)


def test_fragment_choice_and_geometry_independent_of_pair_label_and_epoch_deterministic():
    positive, negative = sample(True), sample(False)
    dataset = weather.RachelWeatheredDataset([positive, negative], clean_probability=0., mild_probability=1.)
    p, pr = dataset[0]
    n, nr = dataset[1]
    assert pr["tier"] == nr["tier"] and pr["side_a"] == nr["side_a"]
    np.testing.assert_array_equal(p.mask_a, n.mask_a)
    assert np.all(n.target_a[n.contour_valid_a] == -1)
    assert np.all(n.target_b[n.contour_valid_b] == -1)
    assert not nr["pose_supervision_enabled"] and nr["inherited_match_count"] == 0
    assert dataset.cache_info()["hits"] >= 2
    dataset.set_epoch(2)
    changed, report2 = dataset[0]
    again = weather.RachelWeatheredDataset([positive], epoch=2, clean_probability=0., mild_probability=1.)
    repeated, repeated_report = again[0]
    np.testing.assert_array_equal(changed.mask_a, repeated.mask_a)
    assert report2 == repeated_report
    assert not np.array_equal(changed.mask_a, p.mask_a)


def test_source_arc_mapping_handles_closed_origin_shift_and_extra_descendant_ignore():
    clean = sample()
    old = clean.points_rc_a
    shift = 31
    new = np.roll(old, shift, axis=0).copy()
    ancestor, representatives, report = weather.source_arc_ancestry(clean.mask_a[0].astype(bool), old,
        clean.contour_valid_a, new, max_depth_px=4.)
    expected = np.roll(np.arange(len(old)), shift)
    assert np.count_nonzero(ancestor >= 0) > 490
    np.testing.assert_array_equal(ancestor[ancestor >= 0], expected[ancestor >= 0])
    assert report["source_projection_radius_px"] == 10.
    # Duplicate token gives an exactly tied representative: ignore both children,
    # never mark an extra putative seam descendant as dustbin.
    duplicate = np.insert(old, 20, old[20], axis=0)
    ancestor, representatives, _ = weather.source_arc_ancestry(clean.mask_a[0].astype(bool), old,
        clean.contour_valid_a, duplicate, max_depth_px=2.)
    assert ancestor[20] == ancestor[21] == -1
    assert representatives[20] == -1


def test_zero_inherited_positive_explicit_full_clean_fallback():
    clean = sample()
    dataset = weather.RachelWeatheredDataset([clean], clean_probability=0., mild_probability=1.)

    def empty_targets(sample, a, b):
        return np.full(len(a.points), -2, np.int64), np.full(len(b.points), -2, np.int64)

    with patch.object(weather, "inherit_pair_targets", side_effect=empty_targets):
        result, report = dataset[0]
    assert result is clean and report["fallback_reason"] == "no_reliable_inherited_positive_correspondence"
    assert report["inherited_match_count"] == 0 and report["effective_supervised_match_count"] == 128
    assert not report["changed_pair"] and not report["changed_a"] and not report["changed_b"]
    assert report["pose_supervision_enabled"]
    assert report["side_a"]["attempted_applied"] and not report["side_a"]["effective_applied"]
    assert report["side_a"]["effective_removed_area_px"] == 0


def test_disk_cache_reuses_geometry_and_does_not_store_targets_or_labels(tmp_path):
    clean = sample()
    kwargs = dict(cache_dir=tmp_path, clean_probability=0., mild_probability=1.)
    first, report = weather.RachelWeatheredDataset([clean], **kwargs)[0]
    dataset = weather.RachelWeatheredDataset([clean], **kwargs)
    with patch.object(weather, "weather_fragment_edges", side_effect=AssertionError("cached geometry recomputed")):
        second, second_report = dataset[0]
    np.testing.assert_array_equal(first.mask_a, second.mask_a)
    np.testing.assert_array_equal(first.target_a, second.target_a)
    assert report == second_report and dataset.cache_info()["disk_hits"] == 2
    for path in tmp_path.rglob("*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == {"packed_mask", "points", "valid", "ancestor", "representatives"}
    # Changed clean source under the same token must never reuse stale geometry.
    changed = replace(clean, points_rc_a=clean.points_rc_a + np.asarray([.02, 0.], np.float32))
    other = weather.RachelWeatheredDataset([changed], **kwargs)
    other[0]
    assert other.cache_info()["misses"] == 1 and other.cache_info()["disk_hits"] == 1


def test_epoch_probabilities_and_train_only_gate():
    class ValDataset:
        split = "val"

    with pytest.raises(ValueError, match="TRAIN-only"):
        weather.RachelWeatheredDataset(ValDataset())
    with pytest.raises(ValueError, match="probabilities"):
        weather.RachelWeatheredDataset([], clean_probability=.8, mild_probability=.4)
    dataset = weather.RachelWeatheredDataset([])
    with pytest.raises(ValueError, match="epoch"):
        dataset.set_epoch(-1)
    # Tiny deterministic ID-only draw sanity, not a corpus distribution audit.
    counts = {tier: sum(dataset.tier_for_fragment("id-" + str(i)) == tier for i in range(1000))
              for tier in ("clean", "mild", "moderate")}
    assert 640 <= counts["clean"] <= 760 and 20 <= counts["moderate"] <= 90
