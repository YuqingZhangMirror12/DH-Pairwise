import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from staging.pairwise_v0_2.geometry import CandidateBuilderConfig
from staging.pairwise_v0_2.models.frozen_transport_order_readout import (
    build_order_coordinate_sidecar,
    exact_target_order_concordance,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.geometry_batch import (
    COARSE_NUMERIC_CONTRACT,
    DATA_DIRECTION_TO_INDEX,
    DIRECTION_NAMES,
    GEOMETRY_BATCH_VERSION,
    KEYPOINT_REPRESENTATION,
    GeometryBatchConfig,
    GeometryBatchError,
    build_geometry_batch,
    data_direction_to_pair_direction,
    data_direction_to_target_index,
    geometry_cache_key,
    inverse_data_direction,
    inverse_direction_index,
    _enforce_candidate_quadratic_bounds,
    preprocess_coarse_mask,
)
from staging.pairwise_v0_2.training.geometry_cache import GeometryArtifactCache
from staging.pairwise_v0_2.training.local_q1_provider import (
    local_q1_prepared_digests,
)


def _rectangle(
    shape=(192, 224),
    top=22,
    left=28,
    height=132,
    width=78,
):
    mask = np.zeros(shape, dtype=bool)
    mask[top : top + height, left : left + width] = True
    return mask


def _u_shape(shape=(192, 224), top=18, left=25, height=145, width=150):
    mask = np.zeros(shape, dtype=bool)
    thickness = 28
    mask[top : top + height, left : left + thickness] = True
    mask[top : top + height, left + width - thickness : left + width] = True
    mask[top + height - thickness : top + height, left : left + width] = True
    return mask


def _geometry_config(**kwargs):
    values = {
        "window_scale_fractions": (0.12,),
        "window_min_px": 2.0,
        "window_max_px": 256.0,
        "output_size": (12, 14),
        "min_run_length_fraction": 0.0,
        "min_run_length_px": 2.0,
        "side_resample_count": 24,
    }
    values.update(kwargs)
    return CandidateBuilderConfig(**values)


def _batch_config(**kwargs):
    values = {
        "geometry": _geometry_config(),
        "coarse_output_size": (48, 56),
        "max_candidates_per_sample": 256,
        "max_candidates_per_batch": 512,
        "max_sequence_length": 1024,
        "max_local_tensor_elements": 20_000_000,
    }
    values.update(kwargs)
    return GeometryBatchConfig(**values)


def _record(
    *,
    label=True,
    direction="right",
    first="fragment/0",
    second="fragment/1",
):
    binding = ArchiveBinding(
        logical_id="fixture://geometry-batch/masks",
        archive_format="zip",
        sha256="a" * 64,
    )
    common = {
        "binding": binding,
        "dataset_id": "fixture_geometry_batch",
        "canonical_group_id": "fixture/group/1",
        "component_id": "fixture/component/1",
        "split": "train",
        "threshold_rule": "grayscale_uint8_gt_127",
    }
    fragment_a = MaskMemberRef(
        archive_member="masks/{}.png".format(first.rsplit("/", 1)[-1]),
        fragment_id=first,
        content_sha256="b" * 64,
        **common,
    )
    fragment_b = MaskMemberRef(
        archive_member="masks/{}.png".format(second.rsplit("/", 1)[-1]),
        fragment_id=second,
        content_sha256="c" * 64,
        **common,
    )
    return TrainingPairRecord(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        label=label,
        direction_b_wrt_a=direction,
        dataset_id="fixture_geometry_batch",
        canonical_group_id="fixture/group/1",
        component_id="fixture/component/1",
        split="train",
        canonical_pair_key=tuple(sorted((first, second))),
        label_origin="fixture",
        provenance={"real_dunhuang_sealed_test": False},
    )


class _MaskLoader:
    def __init__(self, masks):
        self.masks = masks
        self.requests = []

    def __call__(self, reference):
        self.requests.append(reference.fragment_id)
        return self.masks[reference.fragment_id]


def test_default_coarse_preprocess_removes_common_canvas_position_and_size():
    first = np.zeros((96, 128), dtype=bool)
    first[8:48, 11:35] = True
    second = np.zeros((260, 310), dtype=bool)
    second[177:217, 241:265] = True
    config = _batch_config(coarse_output_size=(64, 64))

    first_output = preprocess_coarse_mask(first, config)
    second_output = preprocess_coarse_mask(second, config)

    assert config.coarse_preprocess_mode == "tight_crop_letterbox"
    assert torch.equal(first_output, second_output)
    assert config.provenance_dict()["coarse_spatial_contract"].endswith(
        "no_common_canvas_position"
    )


def test_ragged_keypoint_batch_carries_exact_targets_outside_model_inputs(
    tmp_path,
):
    first = np.zeros((192, 224), dtype=np.bool_)
    second = np.zeros_like(first)
    first[:, :112] = True
    second[:, 112:] = True
    batch = build_geometry_batch(
        [_record(label=True, direction="right")],
        _MaskLoader({"fragment/0": first, "fragment/1": second}),
        _batch_config(),
        geometry_artifact_cache=GeometryArtifactCache(tmp_path / "cache"),
        candidate_representation=KEYPOINT_REPRESENTATION,
        exact_seam_supervision=True,
    )

    targets = batch.exact_loss_targets()
    assert targets is not None
    assert (targets.assignment_target_a >= 0).any().item()
    # The exact seam is vertical and B is right of A.  Other cardinal
    # hypotheses remain ignored rather than being trained as false matches.
    correct = batch.direction_index == DATA_DIRECTION_TO_INDEX["right"]
    assert (targets.assignment_target_a[correct] >= 0).any().item()
    assert (targets.assignment_target_a[~correct] == -2).all().item()
    assert not any(name.startswith("exact_") for name in batch.model_inputs())


def test_keypoint_candidate_observer_is_readout_only_and_byte_parity_preserving(
    tmp_path,
):
    first = np.zeros((192, 224), dtype=np.bool_)
    second = np.zeros_like(first)
    first[:, :112] = True
    second[:, 112:] = True
    records = [_record(label=True, direction="right")]
    masks = {"fragment/0": first, "fragment/1": second}
    cache = GeometryArtifactCache(tmp_path / "cache")
    baseline = build_geometry_batch(
        records,
        _MaskLoader(masks),
        _batch_config(),
        geometry_artifact_cache=cache,
        candidate_representation=KEYPOINT_REPRESENTATION,
        exact_seam_supervision=True,
    )
    observed = []
    with_observer = build_geometry_batch(
        records,
        _MaskLoader(masks),
        _batch_config(),
        geometry_artifact_cache=cache,
        candidate_representation=KEYPOINT_REPRESENTATION,
        exact_seam_supervision=True,
        keypoint_candidate_observer=lambda candidates: observed.append(candidates),
    )

    assert len(observed) == 1
    assert len(observed[0]) == with_observer.candidate_count
    assert local_q1_prepared_digests(baseline) == local_q1_prepared_digests(
        with_observer
    )
    assert baseline.model_inputs().keys() == with_observer.model_inputs().keys()
    for name, value in baseline.model_inputs().items():
        assert torch.equal(value, with_observer.model_inputs()[name])
    sidecar = build_order_coordinate_sidecar(
        observed[0],
        padded_length_a=int(with_observer.local_a.shape[1]),
        padded_length_b=int(with_observer.local_b.shape[1]),
    )
    target_values = exact_target_order_concordance(
        with_observer.exact_assignment_target_a,
        with_observer.token_mask_a,
        with_observer.token_mask_b,
        with_observer.correspondence_mask,
        sidecar,
    )
    assert target_values.candidate_index.numel() > 0
    candidate_direction = with_observer.direction_index.index_select(
        0, target_values.candidate_index
    )
    candidate_sample = with_observer.sample_index.index_select(
        0, target_values.candidate_index
    )
    correct = candidate_direction == with_observer.direction_target.index_select(
        0, candidate_sample
    )
    assert target_values.anti_pair_count[correct].sum() > 0
    assert (
        target_values.anti_pair_count[correct].sum()
        > target_values.monotone_pair_count[correct].sum()
    )


def test_coarse_bilinear_endpoint_overshoot_is_clamped_to_exact_unit_interval():
    # A one-pixel crop expanded to the frozen 112-pixel content extent
    # reproduces PyTorch's float32 endpoint overshoot (1.000000119 locally).
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    config = _batch_config(coarse_output_size=(128, 128))

    output = preprocess_coarse_mask(mask, config)

    assert output.shape == (1, 128, 128)
    assert output.dtype == torch.float32
    assert output.is_contiguous()
    assert output.min().item() == 0.0
    assert output.max().item() == 1.0
    assert ((output >= 0.0) & (output <= 1.0)).all().item()
    provenance = config.provenance_dict()
    assert GEOMETRY_BATCH_VERSION.endswith("/0.4")
    assert provenance["coarse_numeric_contract"] == COARSE_NUMERIC_CONTRACT


def test_legacy_full_canvas_stretch_is_explicit_and_not_translation_invariant():
    first = np.zeros((96, 128), dtype=bool)
    first[8:48, 11:35] = True
    second = np.zeros((96, 128), dtype=bool)
    second[44:84, 87:111] = True
    safe = _batch_config(coarse_output_size=(64, 64))
    legacy = replace(safe, coarse_preprocess_mode="full_canvas_stretch_legacy")

    assert torch.equal(
        preprocess_coarse_mask(first, safe), preprocess_coarse_mask(second, safe)
    )
    assert not torch.equal(
        preprocess_coarse_mask(first, legacy),
        preprocess_coarse_mask(second, legacy),
    )
    assert legacy.fingerprint != safe.fingerprint
    assert (
        "spatial_shortcut_risk" in legacy.provenance_dict()["coarse_spatial_contract"]
    )


def test_coarse_letterbox_preserves_foreground_aspect_ratio():
    mask = np.zeros((100, 140), dtype=bool)
    mask[31:51, 20:100] = True
    config = _batch_config(
        coarse_output_size=(64, 64),
        coarse_content_fraction=0.75,
        coarse_resize_mode="nearest",
    )
    output = preprocess_coarse_mask(mask, config)[0].numpy() > 0.5
    rows, columns = np.nonzero(output)
    height = int(rows.max() - rows.min() + 1)
    width = int(columns.max() - columns.min() + 1)

    assert (height, width) == (12, 48)
    assert width / height == pytest.approx(4.0)


def test_coarse_preprocess_uses_only_deterministic_largest_component():
    clean = np.zeros((100, 140), dtype=bool)
    clean[31:51, 20:100] = True
    contaminated = clean.copy()
    contaminated[2:4, 131:133] = True
    config = _batch_config(
        coarse_output_size=(64, 64),
        coarse_content_fraction=0.75,
        coarse_resize_mode="nearest",
    )

    assert torch.equal(
        preprocess_coarse_mask(clean, config),
        preprocess_coarse_mask(contaminated, config),
    )
    provenance = config.provenance_dict()
    assert provenance["coarse_component_connectivity"] == 4
    assert "largest_connected_component" in provenance["coarse_spatial_contract"]


def test_coarse_preprocess_config_fails_closed():
    with pytest.raises(ValueError, match="coarse_preprocess_mode"):
        _batch_config(coarse_preprocess_mode="full_canvas_auto")
    with pytest.raises(ValueError, match="coarse_content_fraction"):
        _batch_config(coarse_content_fraction=0.49)
    with pytest.raises(ValueError, match="coarse_component_connectivity"):
        _batch_config(coarse_component_connectivity=6)
    empty = preprocess_coarse_mask(
        np.zeros((32, 32), dtype=bool),
        _batch_config(),
    )
    assert not empty.any().item()


def test_direction_mapping_is_explicit_stable_and_inverts_after_swap():
    assert DIRECTION_NAMES == (
        "b_left_of_a",
        "b_right_of_a",
        "b_above_a",
        "b_below_a",
    )
    assert dict(DATA_DIRECTION_TO_INDEX) == {
        "left": 0,
        "right": 1,
        "above": 2,
        "below": 3,
    }
    for name, index in DATA_DIRECTION_TO_INDEX.items():
        parsed = data_direction_to_pair_direction(name)
        assert parsed.value == DIRECTION_NAMES[index]
        inverted_name = inverse_data_direction(name)
        inverted_index = inverse_direction_index(index)
        assert data_direction_to_target_index(inverted_name) == inverted_index
        assert inverse_direction_index(inverted_index) == index
    assert data_direction_to_target_index(None) == -1
    with pytest.raises(GeometryBatchError, match="unsupported"):
        data_direction_to_target_index("diagonal")
    with pytest.raises(GeometryBatchError, match="out of range"):
        inverse_direction_index(-1)


def test_known_positive_direction_is_output_target_only_and_all_four_are_built():
    record = _record(label=True, direction="right")
    masks = {
        "fragment/0": _rectangle(height=140, width=74),
        "fragment/1": _rectangle(top=35, left=130, height=118, width=63),
    }
    batch = build_geometry_batch([record], _MaskLoader(masks), _batch_config())

    assert batch.labels.tolist() == [True]
    assert batch.direction_target.tolist() == [1]
    assert batch.direction_target_valid.tolist() == [True]
    assert batch.direction_slot_valid.tolist() == [[True, True, True, True]]
    assert set(batch.direction_index.tolist()) == {0, 1, 2, 3}
    assert batch.sample_receipts[0].emitted_directions == DIRECTION_NAMES
    assert batch.config_provenance["candidate_generation_direction_argument"] is None
    assert batch.config_provenance["direction_label_usage"] == (
        "output_target_only_never_geometry_input"
    )
    with pytest.raises(TypeError):
        batch.config_provenance["geometry"]["corrosion"]["enabled"] = True
    with pytest.raises(TypeError):
        batch.sample_receipts[0].geometry_quality["fragment_a"]["bbox_rc_exclusive"][
            0
        ] = 999
    with pytest.raises(TypeError):
        batch.complexity_receipt.sequence_bucket_counts["0-16"] = 999
    detached = batch.sample_receipts[0].to_dict()
    detached["geometry_quality"]["fragment_a"]["bbox_rc_exclusive"][0] = 999
    assert (
        batch.sample_receipts[0].geometry_quality["fragment_a"]["bbox_rc_exclusive"][0]
        != 999
    )


def test_negative_and_positive_without_direction_both_keep_four_inputs():
    negative = _record(label=False, direction=None)
    unknown_positive = _record(label=True, direction=None)
    masks = {
        "fragment/0": _rectangle(),
        "fragment/1": _rectangle(top=30, left=126, height=126, width=70),
    }
    config = _batch_config()
    first = build_geometry_batch([negative], _MaskLoader(masks), config)
    second = build_geometry_batch([unknown_positive], _MaskLoader(masks), config)

    assert first.labels.tolist() == [False]
    assert first.direction_target.tolist() == [-1]
    assert first.direction_target_valid.tolist() == [False]
    assert second.labels.tolist() == [True]
    assert second.direction_target.tolist() == [-1]
    assert second.direction_target_valid.tolist() == [False]
    assert first.direction_slot_valid.tolist() == [[True, True, True, True]]
    assert torch.equal(first.local_a, second.local_a)
    assert torch.equal(first.local_b, second.local_b)
    assert torch.equal(first.direction_index, second.direction_index)


def test_label_and_target_do_not_affect_cache_key_or_geometry_inputs():
    positive = _record(label=True, direction="below")
    negative = _record(label=False, direction=None)
    masks = {
        "fragment/0": _u_shape(),
        "fragment/1": _rectangle(top=40, left=125, height=110, width=66),
    }
    config = _batch_config()
    positive_batch = build_geometry_batch([positive], _MaskLoader(masks), config)
    negative_batch = build_geometry_batch([negative], _MaskLoader(masks), config)

    assert geometry_cache_key(positive, config) == geometry_cache_key(negative, config)
    assert positive_batch.geometry_cache_keys == negative_batch.geometry_cache_keys
    for name in positive_batch.model_inputs():
        assert torch.equal(
            positive_batch.model_inputs()[name], negative_batch.model_inputs()[name]
        )
    assert positive_batch.direction_target.tolist() == [3]
    assert negative_batch.direction_target.tolist() == [-1]


def test_a_b_swap_exchanges_tensors_and_inverts_candidate_direction_slots():
    forward_record = _record(label=True, direction="right")
    reverse_record = _record(
        label=True,
        direction="left",
        first="fragment/1",
        second="fragment/0",
    )
    masks = {
        "fragment/0": _rectangle(top=20, left=25, height=142, width=74),
        "fragment/1": _rectangle(top=38, left=130, height=112, width=62),
    }
    config = _batch_config()
    forward = build_geometry_batch([forward_record], _MaskLoader(masks), config)
    reverse = build_geometry_batch([reverse_record], _MaskLoader(masks), config)

    assert torch.equal(forward.coarse_a, reverse.coarse_b)
    assert torch.equal(forward.coarse_b, reverse.coarse_a)
    assert inverse_direction_index(forward.direction_target.item()) == (
        reverse.direction_target.item()
    )
    for index in range(forward.candidate_count):
        direction = int(forward.direction_index[index].item())
        inverse = inverse_direction_index(direction)
        reverse_indices = torch.nonzero(
            reverse.direction_index == inverse, as_tuple=False
        ).flatten()
        assert len(reverse_indices) == 1
        other = int(reverse_indices.item())
        length_a = int(forward.token_mask_a[index].sum().item())
        length_b = int(forward.token_mask_b[index].sum().item())
        reverse_length_a = int(reverse.token_mask_a[other].sum().item())
        reverse_length_b = int(reverse.token_mask_b[other].sum().item())
        assert length_a == reverse_length_b
        assert length_b == reverse_length_a
        assert torch.equal(
            forward.local_a[index, :length_a], reverse.local_b[other, :length_a]
        )
        assert torch.equal(
            forward.local_b[index, :length_b], reverse.local_a[other, :length_b]
        )


def test_ragged_candidates_are_flattened_padded_and_model_agnostic():
    first = _record(first="fragment/0", second="fragment/1")
    second = _record(first="fragment/2", second="fragment/3", direction="above")
    masks = {
        "fragment/0": _rectangle(height=146, width=60),
        "fragment/1": _rectangle(top=55, left=125, height=90, width=82),
        "fragment/2": _u_shape(height=150, width=164),
        "fragment/3": _rectangle(top=30, left=132, height=128, width=55),
    }
    batch = build_geometry_batch([first, second], _MaskLoader(masks), _batch_config())

    assert batch.coarse_a.shape == (2, 1, 48, 56)
    assert batch.local_a.ndim == 5 and batch.local_b.ndim == 5
    assert batch.local_a.shape[0] == batch.candidate_count
    assert batch.local_a.shape[2:] == (3, 12, 14)
    assert batch.local_b.shape[2:] == (3, 12, 14)
    assert batch.sample_index.dtype == torch.long
    assert batch.direction_index.dtype == torch.long
    assert batch.token_mask_a.dtype == torch.bool
    assert batch.token_mask_b.dtype == torch.bool
    assert set(batch.sample_index.tolist()) == {0, 1}
    assert (~batch.token_mask_a).any().item() or (~batch.token_mask_b).any().item()
    assert torch.count_nonzero(batch.local_a[~batch.token_mask_a]) == 0
    assert torch.count_nonzero(batch.local_b[~batch.token_mask_b]) == 0
    assert len(batch.candidate_ids) == len(set(batch.candidate_ids))
    assert set(batch.model_inputs()) == {
        "coarse_a",
        "coarse_b",
        "local_a",
        "local_b",
        "token_mask_a",
        "token_mask_b",
        "sample_index",
        "direction_index",
        "candidate_valid",
    }
    moved = batch.to(torch.device("cpu"))
    assert moved.candidate_ids == batch.candidate_ids
    assert moved.config_fingerprint == batch.config_fingerprint
    receipt = batch.complexity_receipt
    assert receipt.candidate_count == batch.candidate_count
    assert len(batch.sequence_length_buckets) == batch.candidate_count
    assert sum(receipt.sequence_bucket_counts.values()) == batch.candidate_count
    assert receipt.affinity_elements == (
        batch.candidate_count * batch.local_a.shape[1] * batch.local_b.shape[1]
    )
    assert receipt.sinkhorn_elements == (
        batch.candidate_count
        * (batch.local_a.shape[1] + 1)
        * (batch.local_b.shape[1] + 1)
    )


def test_missing_direction_slots_and_fully_invalid_geometry_are_explicit():
    record = _record()
    wide = _rectangle(shape=(220, 220), top=75, left=25, height=50, width=160)
    partial_config = _batch_config(geometry=_geometry_config(min_run_length_px=90.0))
    partial = build_geometry_batch(
        [record],
        _MaskLoader({"fragment/0": wide, "fragment/1": wide.copy()}),
        partial_config,
    )

    assert partial.direction_slot_valid.tolist() == [[False, False, True, True]]
    assert partial.geometry_valid.tolist() == [True]
    assert set(partial.direction_index.tolist()) == {2, 3}

    invalid = build_geometry_batch(
        [record],
        _MaskLoader(
            {
                "fragment/0": np.zeros((64, 64), dtype=bool),
                "fragment/1": _rectangle(
                    shape=(64, 64), top=8, left=10, height=45, width=30
                ),
            }
        ),
        _batch_config(),
    )
    assert invalid.candidate_count == 0
    assert invalid.local_a.shape == (0, 1, 3, 12, 14)
    assert invalid.local_b.shape == (0, 1, 3, 12, 14)
    assert invalid.direction_slot_valid.tolist() == [[False, False, False, False]]
    assert invalid.geometry_valid.tolist() == [False]
    assert invalid.sample_receipts[0].geometry_status != "ok"
    assert invalid.sample_receipts[0].geometry_failure_reason


def test_config_cache_receipt_is_deterministic_json_safe_and_cache_aware():
    record = _record()
    config = _batch_config()
    assert config.fingerprint == config.fingerprint
    assert geometry_cache_key(record, config) == geometry_cache_key(record, config)
    changed = replace(
        config,
        geometry=replace(config.geometry, foreground_polarity="dark"),
    )
    assert changed.fingerprint != config.fingerprint
    assert geometry_cache_key(record, changed) != geometry_cache_key(record, config)
    coarse_only = replace(config, coarse_output_size=(56, 56))
    assert geometry_cache_key(record, coarse_only) == geometry_cache_key(record, config)
    path_and_pair_changed = _record(first="different/path/a", second="different/path/b")
    assert geometry_cache_key(path_and_pair_changed, config) == geometry_cache_key(
        record, config
    )
    payload = config.provenance_dict()
    json.dumps(payload, sort_keys=True, allow_nan=False)
    assert payload["cache_policy"] == (
        "optional_content_addressed_role_neutral_fragment_npz_same_tensor_contract"
    )


def test_repeated_sample_has_shared_cache_key_but_unique_flat_candidate_ids():
    record = _record()
    masks = {
        "fragment/0": _rectangle(),
        "fragment/1": _rectangle(top=30, left=125, height=124, width=70),
    }
    batch = build_geometry_batch([record, record], _MaskLoader(masks), _batch_config())
    assert batch.geometry_cache_keys[0] == batch.geometry_cache_keys[1]
    assert len(batch.candidate_ids) == len(set(batch.candidate_ids))
    assert set(batch.sample_index.tolist()) == {0, 1}


def test_fragment_cache_shared_across_pairs_computes_once_and_matches_uncached(
    tmp_path, monkeypatch
):
    from staging.pairwise_v0_2.training import fragment_geometry_cache as bridge

    first = _record(first="fragment/0", second="fragment/shared")
    second = _record(first="fragment/shared", second="fragment/2", direction="above")
    masks = {
        "fragment/0": _rectangle(height=146, width=60),
        "fragment/shared": _u_shape(height=150, width=164),
        "fragment/2": _rectangle(top=30, left=132, height=128, width=55),
    }
    records = [first, second]
    config = _batch_config()
    uncached = build_geometry_batch(records, _MaskLoader(masks), config)

    original = bridge.build_fragment_geometry
    calls = []

    def counted(mask, settings):
        calls.append(np.packbits(mask.reshape(-1)).tobytes())
        return original(mask, settings)

    monkeypatch.setattr(bridge, "build_fragment_geometry", counted)
    cache = GeometryArtifactCache(tmp_path)
    cached = build_geometry_batch(
        records,
        _MaskLoader(masks),
        config,
        geometry_artifact_cache=cache,
    )
    assert len(calls) == 3
    assert len(list(tmp_path.rglob("*.npz"))) == 3

    for name in uncached.model_inputs():
        assert torch.equal(uncached.model_inputs()[name], cached.model_inputs()[name])
    assert uncached.labels.tolist() == cached.labels.tolist()
    assert torch.equal(uncached.direction_target, cached.direction_target)
    assert torch.equal(uncached.direction_target_valid, cached.direction_target_valid)
    assert torch.equal(uncached.direction_slot_valid, cached.direction_slot_valid)
    assert torch.equal(uncached.geometry_valid, cached.geometry_valid)
    assert uncached.candidate_ids == cached.candidate_ids
    assert uncached.sample_ids == cached.sample_ids
    assert uncached.geometry_cache_keys == cached.geometry_cache_keys
    assert uncached.sample_receipts == cached.sample_receipts
    assert uncached.config_fingerprint == cached.config_fingerprint
    assert uncached.config_provenance == cached.config_provenance
    assert uncached.complexity_receipt == cached.complexity_receipt
    assert uncached.sequence_length_buckets == cached.sequence_length_buckets

    repeated = build_geometry_batch(
        records,
        _MaskLoader(masks),
        config,
        geometry_artifact_cache=cache,
    )
    assert len(calls) == 3
    for name in cached.model_inputs():
        assert torch.equal(cached.model_inputs()[name], repeated.model_inputs()[name])


def test_fragment_cache_and_mask_loader_are_memoized_per_unique_batch_fragment(
    tmp_path, monkeypatch
):
    from staging.pairwise_v0_2.training import geometry_batch as bridge

    pair = _record()
    swapped = replace(
        pair,
        fragment_a=pair.fragment_b,
        fragment_b=pair.fragment_a,
        direction_b_wrt_a="left",
    )
    records = [pair, pair, swapped]
    masks = {
        "fragment/0": _rectangle(),
        "fragment/1": _rectangle(top=30, left=125, height=124, width=70),
    }
    loader = _MaskLoader(masks)
    original = bridge.load_or_build_fragment_geometry
    rehydrate_calls = []

    def counted(*args, **kwargs):
        rehydrate_calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(bridge, "load_or_build_fragment_geometry", counted)
    batch = build_geometry_batch(
        records,
        loader,
        _batch_config(),
        geometry_artifact_cache=GeometryArtifactCache(tmp_path),
    )
    assert len(batch.sample_ids) == 3
    assert len(loader.requests) == 2
    assert sorted(loader.requests) == ["fragment/0", "fragment/1"]
    assert len(rehydrate_calls) == 2


def test_fail_closed_bounds_and_loader_bool_contract():
    record = _record()
    masks = {
        "fragment/0": _rectangle(),
        "fragment/1": _rectangle(top=30, left=125, height=124, width=70),
    }
    with pytest.raises(GeometryBatchError, match="max_candidates_per_sample"):
        build_geometry_batch(
            [record],
            _MaskLoader(masks),
            _batch_config(max_candidates_per_sample=1),
        )
    with pytest.raises(GeometryBatchError, match="canonical bool"):
        build_geometry_batch(
            [record],
            _MaskLoader({key: value.astype(np.uint8) for key, value in masks.items()}),
            _batch_config(),
        )
    with pytest.raises(GeometryBatchError, match="input pixel bound"):
        build_geometry_batch(
            [record],
            _MaskLoader(masks),
            _batch_config(max_input_pixels_per_mask=100),
        )


@pytest.mark.parametrize(
    "field, expected",
    [
        (
            "max_attention_score_elements_per_candidate",
            "attention score elements",
        ),
        ("max_affinity_elements_per_candidate", "affinity elements"),
        ("max_sinkhorn_elements_per_candidate", "Sinkhorn elements"),
    ],
)
def test_per_candidate_quadratic_guards_fail_before_model_call(field, expected):
    config = _batch_config(**{field: 1})
    with pytest.raises(GeometryBatchError, match=expected):
        _enforce_candidate_quadratic_bounds(10, 12, config)


def test_padded_batch_quadratic_guard_fails_before_tensor_allocation():
    record = _record()
    masks = {
        "fragment/0": _rectangle(),
        "fragment/1": _rectangle(top=30, left=125, height=124, width=70),
    }
    with pytest.raises(GeometryBatchError, match="padded attention score elements"):
        build_geometry_batch(
            [record],
            _MaskLoader(masks),
            _batch_config(max_attention_score_elements_per_batch=1),
        )
