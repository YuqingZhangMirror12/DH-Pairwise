import json
from dataclasses import replace

import numpy as np
import pytest

from staging.pairwise_v0_2.geometry import (
    CandidateBuilderConfig,
    PairDirection,
    build_fragment_geometry,
    build_pair_candidates,
    combine_fragment_results,
)
from staging.pairwise_v0_2.training.fragment_geometry_cache import (
    FragmentGeometryCacheError,
    fragment_result_from_cache_artifact,
    fragment_result_to_cache_payload,
    load_or_build_fragment_geometry,
)
from staging.pairwise_v0_2.training.geometry_cache import (
    GeometryArtifactCache,
    fragment_cache_identity,
)


def _mask(*, top=18, left=22, height=106, width=64, notch=False):
    value = np.zeros((152, 176), dtype=bool)
    value[top : top + height, left : left + width] = True
    if notch:
        value[top + 28 : top + 62, left + width - 18 : left + width] = False
    return value


def _config(**kwargs):
    values = {
        "window_scale_fractions": (0.1, 0.18),
        "window_min_px": 2.0,
        "window_max_px": 128.0,
        "output_size": (12, 14),
        "min_run_length_fraction": 0.0,
        "min_run_length_px": 2.0,
        "side_resample_count": 24,
    }
    values.update(kwargs)
    return CandidateBuilderConfig(**values)


def _assert_pair_results_equal(first, second):
    assert first.status == second.status
    assert first.failure_reason == second.failure_reason
    assert first.quality == second.quality
    assert first.direction_groups == second.direction_groups
    assert tuple(item.candidate_id for item in first.candidates) == tuple(
        item.candidate_id for item in second.candidates
    )
    for expected, observed in zip(first.candidates, second.candidates):
        assert expected.direction == observed.direction
        assert expected.sequence_a.provenance_dict() == (
            observed.sequence_a.provenance_dict()
        )
        assert expected.sequence_b.provenance_dict() == (
            observed.sequence_b.provenance_dict()
        )
        assert np.array_equal(expected.patches_a, observed.patches_a)
        assert np.array_equal(expected.patches_b, observed.patches_b)
        assert np.array_equal(expected.valid_a, observed.valid_a)
        assert np.array_equal(expected.valid_b, observed.valid_b)


def test_role_neutral_build_and_combine_preserve_legacy_pair_behavior():
    mask_a = _mask(notch=True)
    mask_b = _mask(top=29, left=91, height=92, width=57)
    config = _config()
    fragment_a = build_fragment_geometry(mask_a, config)
    fragment_b = build_fragment_geometry(mask_b, config)
    assert fragment_a.ok and fragment_b.ok
    assert fragment_a.artifact is not None and fragment_b.artifact is not None
    assert all(
        sequence.fragment_role == "fragment"
        for sequence in fragment_a.artifact.sequences + fragment_b.artifact.sequences
    )

    for direction in (None, *tuple(PairDirection)):
        legacy = build_pair_candidates(mask_a, mask_b, direction, config)
        combined = combine_fragment_results(
            fragment_a,
            fragment_b,
            direction_b_wrt_a=direction,
            config=config,
        )
        _assert_pair_results_equal(legacy, combined)

        legacy_swapped = build_pair_candidates(mask_b, mask_a, direction, config)
        combined_swapped = combine_fragment_results(
            fragment_b,
            fragment_a,
            direction_b_wrt_a=direction,
            config=config,
        )
        _assert_pair_results_equal(legacy_swapped, combined_swapped)


def test_fragment_cache_roundtrip_is_role_neutral_and_portable(tmp_path):
    mask = _mask(notch=True)
    config = _config()
    cache = GeometryArtifactCache(tmp_path)
    first = load_or_build_fragment_geometry(
        mask, "grayscale_uint8_gt_127", config, cache
    )
    second = load_or_build_fragment_geometry(
        mask, "grayscale_uint8_gt_127", config, cache
    )
    assert not first.cache_hit and second.cache_hit
    assert first.identity == second.identity
    assert first.result.artifact is not None
    assert second.result.artifact is not None
    assert all(
        sequence.fragment_role == "fragment"
        for sequence in second.result.artifact.sequences
    )

    encoded = json.dumps(
        cache.get(first.identity).metadata_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).casefold()
    for forbidden in (
        "pair_id",
        "dataset_id",
        "canonical_group_id",
        "component_id",
        '"label"',
        '"split"',
        '"path"',
        "archive_member",
        "fragment_a",
        "fragment_b",
    ):
        assert forbidden not in encoded
    assert '"role":"fragment"' in encoded

    changed = replace(config, foreground_polarity="dark")
    assert (
        first.identity.key
        != fragment_cache_identity(mask, "grayscale_uint8_gt_127", changed).key
    )


def test_semantically_invalid_hash_valid_payload_fails_closed(tmp_path):
    mask = _mask()
    config = _config()
    result = build_fragment_geometry(mask, config)
    arrays, metadata = fragment_result_to_cache_payload(result, config)
    invalid_metadata = dict(metadata)
    invalid_artifact = dict(invalid_metadata["artifact"])
    invalid_quality = dict(invalid_artifact["quality"])
    invalid_quality["role"] = "a"
    invalid_artifact["quality"] = invalid_quality
    invalid_metadata["artifact"] = invalid_artifact

    identity = fragment_cache_identity(mask, "grayscale_uint8_gt_127", config)
    cache = GeometryArtifactCache(tmp_path)
    cached = cache.put(identity, arrays, invalid_metadata)
    with pytest.raises(FragmentGeometryCacheError, match="role"):
        fragment_result_from_cache_artifact(cached, config)
    with pytest.raises(FragmentGeometryCacheError, match="role"):
        load_or_build_fragment_geometry(mask, "grayscale_uint8_gt_127", config, cache)


@pytest.mark.parametrize(
    "channel_index, invalid_value",
    [(0, 42.0), (1, 99.0), (2, -7.0)],
)
def test_hash_valid_out_of_range_channels_fail_typed_decode(
    tmp_path, channel_index, invalid_value
):
    mask = _mask(notch=True)
    config = _config()
    result = build_fragment_geometry(mask, config)
    arrays, metadata = fragment_result_to_cache_payload(result, config)
    corrupted = dict(arrays)
    channels = np.asarray(corrupted["channels"]).copy()
    assert channels.shape[0] > 0
    channels[:, channel_index] = invalid_value
    corrupted["channels"] = channels
    identity = fragment_cache_identity(mask, "grayscale_uint8_gt_127", config)
    cache = GeometryArtifactCache(tmp_path)
    artifact = cache.put(identity, corrupted, metadata)
    with pytest.raises(FragmentGeometryCacheError, match="semantic range"):
        fragment_result_from_cache_artifact(artifact, config)


@pytest.mark.parametrize(
    "mutation, error",
    [
        ("negative_bbox", "integer >= 0"),
        ("foreground_exceeds_area", "image area"),
        ("contour_perimeter", "point/perimeter"),
        ("requested_window", "window provenance"),
        ("negative_stride", "must be positive"),
        ("direction_rule", "frozen side convention"),
        ("patch_distance", "sliding-window contract"),
    ],
)
def test_hash_valid_typed_provenance_tampering_fails_closed(tmp_path, mutation, error):
    mask = _mask(notch=True)
    config = _config()
    result = build_fragment_geometry(mask, config)
    source_arrays, source_metadata = fragment_result_to_cache_payload(result, config)
    arrays = {name: np.asarray(value).copy() for name, value in source_arrays.items()}
    metadata = json.loads(json.dumps(source_metadata))
    quality = metadata["artifact"]["quality"]
    sequence = metadata["artifact"]["sequences"][0]
    if mutation == "negative_bbox":
        quality["bbox_rc_exclusive"][0] = -1
    elif mutation == "foreground_exceeds_area":
        image_area = int(np.prod(quality["input_shape"]))
        quality["foreground_pixels"] = image_area + 1
        quality["discarded_foreground_pixels"] = (
            quality["foreground_pixels"] - quality["largest_component_pixels"]
        )
    elif mutation == "contour_perimeter":
        quality["contour_perimeter_px"] += 5.0
    elif mutation == "requested_window":
        sequence["requested_window_px"] *= 2.0
    elif mutation == "negative_stride":
        sequence["stride_px"] = -1.0
    elif mutation == "direction_rule":
        sequence["direction_rule"] = "bottom_to_top"
    elif mutation == "patch_distance":
        offset = sequence["offset"]
        arrays["patch_float"][offset, 7] += 0.25
    else:  # pragma: no cover - parametrization is closed
        raise AssertionError("unknown mutation")

    identity = fragment_cache_identity(mask, "grayscale_uint8_gt_127", config)
    artifact = GeometryArtifactCache(tmp_path / mutation).put(
        identity, arrays, metadata
    )
    with pytest.raises(FragmentGeometryCacheError, match=error):
        fragment_result_from_cache_artifact(artifact, config)


def test_failed_fragment_is_cached_without_partial_geometry(tmp_path, monkeypatch):
    from staging.pairwise_v0_2.training import fragment_geometry_cache as bridge

    mask = np.zeros((64, 72), dtype=bool)
    config = _config()
    cache = GeometryArtifactCache(tmp_path)
    original = bridge.build_fragment_geometry
    calls = []

    def counted(value, settings):
        calls.append(True)
        return original(value, settings)

    monkeypatch.setattr(bridge, "build_fragment_geometry", counted)
    first = load_or_build_fragment_geometry(
        mask, "grayscale_uint8_gt_127", config, cache
    )
    second = load_or_build_fragment_geometry(
        mask, "grayscale_uint8_gt_127", config, cache
    )
    assert not first.result.ok and not second.result.ok
    assert first.result.artifact is None and second.result.artifact is None
    assert not first.cache_hit and second.cache_hit
    assert len(calls) == 1
    cached = cache.get(first.identity)
    assert all(value.size == 0 for value in cached.arrays.values())
