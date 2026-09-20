"""Typed serialization bridge for role-neutral fragment geometry artifacts.

The generic :mod:`geometry_cache` container owns content addressing, atomic
commits, allocation limits, and cryptographic integrity.  This module owns the
stricter semantic schema needed to serialize and reconstruct one fragment's
contour-window features without admitting pair metadata or supervision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from staging.pairwise_v0_1.baselines.b1_contours import CardinalSide
from staging.pairwise_v0_2.geometry import (
    CHANNEL_ORDER,
    FRAGMENT_GEOMETRY_ARTIFACT_VERSION,
    GEOMETRY_VERSION,
    CandidateBuilderConfig,
    FragmentGeometryArtifact,
    FragmentGeometryQuality,
    FragmentGeometryResult,
    FragmentSequenceFailure,
    GeometryStatus,
    PatchProvenance,
    PatchSequence,
    build_fragment_geometry,
    geometry_config_fingerprint,
)
from staging.pairwise_v0_2.training.geometry_cache import (
    CACHE_TYPED_PAYLOAD_VERSION,
    FragmentCacheIdentity,
    GeometryArtifactCache,
    GeometryCacheArtifact,
    fragment_cache_identity,
)


FRAGMENT_CACHE_PAYLOAD_SCHEMA_VERSION = CACHE_TYPED_PAYLOAD_VERSION
_ARRAY_NAMES = frozenset({"channels", "valid", "patch_int", "patch_float"})
_PATCH_FLOAT_FIELDS = 10
_PATCH_INT_FIELDS = 3


class FragmentGeometryCacheError(RuntimeError):
    """A hash-valid cache payload violates the typed fragment schema."""


@dataclass(frozen=True)
class FragmentGeometryCacheLookup:
    identity: FragmentCacheIdentity
    result: FragmentGeometryResult
    cache_hit: bool
    logical_payload_sha256: str


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], name: str) -> None:
    if set(value) != set(expected):
        raise FragmentGeometryCacheError("{} fields do not match schema".format(name))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FragmentGeometryCacheError("{} must be an object".format(name))
    return value


def _list(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise FragmentGeometryCacheError("{} must be a list".format(name))
    return value


def _string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise FragmentGeometryCacheError("{} must be a string".format(name))
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FragmentGeometryCacheError(
            "{} must be an integer >= {}".format(name, minimum)
        )
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FragmentGeometryCacheError("{} must be numeric".format(name))
    result = float(value)
    if not np.isfinite(result):
        raise FragmentGeometryCacheError("{} must be finite".format(name))
    return result


def _quality_from_dict(
    value: Any, geometry_config: CandidateBuilderConfig
) -> FragmentGeometryQuality:
    item = _mapping(value, "quality")
    expected = (
        "role",
        "input_shape",
        "input_dtype",
        "foreground_pixels",
        "component_count",
        "largest_component_pixels",
        "discarded_foreground_pixels",
        "bbox_rc_exclusive",
        "bbox_reference_px",
        "contour_points",
        "contour_perimeter_px",
        "retained_runs_by_side",
        "dropped_short_runs_by_side",
        "warnings",
    )
    _exact_keys(item, expected, "quality")
    if item["role"] != "fragment":
        raise FragmentGeometryCacheError("quality role is not neutral")
    shape = tuple(
        _integer(part, "quality.input_shape")
        for part in _list(item["input_shape"], "quality.input_shape")
    )
    if len(shape) != 2 or min(shape) < 1:
        raise FragmentGeometryCacheError("quality input shape is invalid")
    bbox = tuple(
        _integer(part, "quality.bbox_rc_exclusive")
        for part in _list(item["bbox_rc_exclusive"], "quality.bbox_rc_exclusive")
    )
    if len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise FragmentGeometryCacheError("quality bbox is invalid")
    retained = _mapping(item["retained_runs_by_side"], "retained runs")
    dropped = _mapping(item["dropped_short_runs_by_side"], "dropped runs")
    side_names = tuple(side.value for side in CardinalSide)
    _exact_keys(retained, side_names, "retained runs")
    _exact_keys(dropped, side_names, "dropped runs")
    warnings = tuple(
        _string(part, "quality warning")
        for part in _list(item["warnings"], "quality warnings")
    )
    foreground_pixels = _integer(
        item["foreground_pixels"], "quality.foreground_pixels", minimum=1
    )
    largest_component_pixels = _integer(
        item["largest_component_pixels"],
        "quality.largest_component_pixels",
        minimum=1,
    )
    discarded_foreground_pixels = _integer(
        item["discarded_foreground_pixels"],
        "quality.discarded_foreground_pixels",
    )
    if foreground_pixels != (largest_component_pixels + discarded_foreground_pixels):
        raise FragmentGeometryCacheError(
            "quality foreground accounting is inconsistent"
        )
    image_area = math.prod(shape)
    if foreground_pixels > image_area:
        raise FragmentGeometryCacheError("quality foreground count exceeds image area")
    component_count = _integer(
        item["component_count"], "quality.component_count", minimum=1
    )
    if component_count > foreground_pixels:
        raise FragmentGeometryCacheError(
            "quality component count exceeds foreground count"
        )
    if not (0 <= bbox[0] < bbox[2] <= shape[0] and 0 <= bbox[1] < bbox[3] <= shape[1]):
        raise FragmentGeometryCacheError("quality bbox exceeds input shape")
    bbox_reference_px = _finite_float(
        item["bbox_reference_px"], "quality.bbox_reference_px"
    )
    expected_reference = float(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    if abs(bbox_reference_px - expected_reference) > 1e-6:
        raise FragmentGeometryCacheError("quality bbox reference is inconsistent")
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if largest_component_pixels > bbox_area:
        raise FragmentGeometryCacheError(
            "quality largest component exceeds its bounding box"
        )
    if largest_component_pixels < geometry_config.min_component_pixels:
        raise FragmentGeometryCacheError(
            "quality largest component violates geometry config"
        )
    contour_points = _integer(
        item["contour_points"],
        "quality.contour_points",
        minimum=geometry_config.min_contour_points,
    )
    contour_perimeter_px = _finite_float(
        item["contour_perimeter_px"], "quality.contour_perimeter_px"
    )
    if contour_perimeter_px <= 0.0:
        raise FragmentGeometryCacheError("quality contour perimeter must be positive")
    # The frozen contour recipes trace unit pixel-cell edges.  Consequently
    # every canonical loop point owns exactly one unit outgoing edge.
    if abs(contour_perimeter_px - float(contour_points)) > 1e-6:
        raise FragmentGeometryCacheError(
            "quality contour point/perimeter accounting is inconsistent"
        )
    if contour_points > 4 * largest_component_pixels:
        raise FragmentGeometryCacheError(
            "quality contour exceeds the exposed-edge bound"
        )
    retained_counts = tuple(
        (name, _integer(retained[name], "retained run count")) for name in side_names
    )
    if not any(count for _, count in retained_counts):
        raise FragmentGeometryCacheError(
            "successful quality requires a retained contour run"
        )
    dropped_counts = tuple(
        (name, _integer(dropped[name], "dropped run count")) for name in side_names
    )
    if sum(count for _, count in retained_counts + dropped_counts) > contour_points:
        raise FragmentGeometryCacheError(
            "quality contour-run counts exceed contour size"
        )
    input_dtype = _string(item["input_dtype"], "quality.input_dtype")
    if input_dtype != "bool":
        raise FragmentGeometryCacheError(
            "quality input dtype is inconsistent with canonical cache masks"
        )
    return FragmentGeometryQuality(
        role="fragment",
        input_shape=shape,
        input_dtype=input_dtype,
        foreground_pixels=foreground_pixels,
        component_count=component_count,
        largest_component_pixels=largest_component_pixels,
        discarded_foreground_pixels=discarded_foreground_pixels,
        bbox_rc_exclusive=bbox,  # type: ignore[arg-type]
        bbox_reference_px=bbox_reference_px,
        contour_points=contour_points,
        contour_perimeter_px=contour_perimeter_px,
        retained_runs_by_side=retained_counts,
        dropped_short_runs_by_side=dropped_counts,
        warnings=warnings,
    )


def _patch_arrays(
    sequences: Sequence[PatchSequence],
    output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = sum(sequence.length for sequence in sequences)
    if sequences:
        channels = np.concatenate([sequence.channels for sequence in sequences], axis=0)
        valid = np.concatenate([sequence.valid for sequence in sequences], axis=0)
        patch_height, patch_width = sequences[0].channels.shape[-2:]
    else:
        patch_height, patch_width = output_size
        channels = np.empty(
            (0, len(CHANNEL_ORDER), patch_height, patch_width), dtype=np.float32
        )
        valid = np.empty((0,), dtype=np.bool_)
    patch_int = np.empty((total, _PATCH_INT_FIELDS), dtype=np.int64)
    patch_float = np.empty((total, _PATCH_FLOAT_FIELDS), dtype=np.float64)
    offset = 0
    for sequence in sequences:
        if tuple(sequence.channels.shape[-2:]) != (patch_height, patch_width):
            raise FragmentGeometryCacheError("sequence patch sizes are inconsistent")
        for patch in sequence.patches:
            patch_int[offset] = (
                patch.sequence_index,
                patch.contour_segment_indices[0],
                patch.contour_segment_indices[1],
            )
            patch_float[offset] = (
                patch.center_row_col[0],
                patch.center_row_col[1],
                patch.tangent_row_col[0],
                patch.tangent_row_col[1],
                patch.inward_normal_row_col[0],
                patch.inward_normal_row_col[1],
                patch.contour_arc_fraction,
                patch.path_distance_px,
                patch.valid_fraction,
                patch.padding_fraction,
            )
            offset += 1
    return (
        np.ascontiguousarray(channels, dtype=np.float32),
        np.ascontiguousarray(valid, dtype=np.bool_),
        patch_int,
        patch_float,
    )


def fragment_result_to_cache_payload(
    result: FragmentGeometryResult,
    geometry_config: CandidateBuilderConfig,
) -> Tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
    """Encode a typed result into bounded numeric arrays and portable metadata."""

    if not isinstance(result, FragmentGeometryResult):
        raise TypeError("result must be FragmentGeometryResult")
    if not isinstance(geometry_config, CandidateBuilderConfig):
        raise TypeError("geometry_config must be CandidateBuilderConfig")
    empty_channels = np.empty(
        (0, len(CHANNEL_ORDER), *geometry_config.output_size), dtype=np.float32
    )
    arrays: Dict[str, np.ndarray] = {
        "channels": empty_channels,
        "valid": np.empty((0,), dtype=np.bool_),
        "patch_int": np.empty((0, _PATCH_INT_FIELDS), dtype=np.int64),
        "patch_float": np.empty((0, _PATCH_FLOAT_FIELDS), dtype=np.float64),
    }
    metadata: Dict[str, Any] = {
        "payload_schema_version": FRAGMENT_CACHE_PAYLOAD_SCHEMA_VERSION,
        "artifact_version": FRAGMENT_GEOMETRY_ARTIFACT_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "role_policy": "single_fragment_role_neutral",
        "config_fingerprint": geometry_config_fingerprint(geometry_config),
        "status": result.status.value,
        "failure_reason": result.failure_reason,
        "artifact": None,
    }
    if not result.ok:
        return arrays, metadata
    artifact = result.artifact
    if artifact is None:  # pragma: no cover - dataclass invariant
        raise RuntimeError("successful result lost its artifact")
    if artifact.config_fingerprint != metadata["config_fingerprint"]:
        raise FragmentGeometryCacheError("artifact config fingerprint mismatch")
    channels, valid, patch_int, patch_float = _patch_arrays(
        artifact.sequences, geometry_config.output_size
    )
    if channels.shape[-2:] != geometry_config.output_size:
        raise FragmentGeometryCacheError("artifact patch size disagrees with config")
    arrays = {
        "channels": channels,
        "valid": valid,
        "patch_int": patch_int,
        "patch_float": patch_float,
    }
    offset = 0
    sequence_metadata = []
    for sequence in artifact.sequences:
        sequence_metadata.append(
            {
                "side": sequence.side.value,
                "run_index": sequence.run_index,
                "scale_index": sequence.scale_index,
                "scale_fraction": sequence.scale_fraction,
                "bbox_reference_px": sequence.bbox_reference_px,
                "requested_window_px": sequence.requested_window_px,
                "resolved_window_px": sequence.resolved_window_px,
                "stride_px": sequence.stride_px,
                "run_length_px": sequence.run_length_px,
                "direction_rule": sequence.direction_rule,
                "offset": offset,
                "length": sequence.length,
            }
        )
        offset += sequence.length
    metadata["artifact"] = {
        "window_scale_fractions": list(artifact.window_scale_fractions),
        "run_indices_by_side": {
            name: list(indices) for name, indices in artifact.run_indices_by_side
        },
        "sequences": sequence_metadata,
        "sequence_failures": [
            failure.to_dict() for failure in artifact.sequence_failures
        ],
        "quality": artifact.quality.to_dict(),
    }
    return arrays, metadata


def _checked_arrays(
    artifact: GeometryCacheArtifact, geometry_config: CandidateBuilderConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if set(artifact.arrays) != _ARRAY_NAMES:
        raise FragmentGeometryCacheError("fragment cache arrays do not match schema")
    channels = artifact.arrays["channels"]
    valid = artifact.arrays["valid"]
    patch_int = artifact.arrays["patch_int"]
    patch_float = artifact.arrays["patch_float"]
    expected_tail = (len(CHANNEL_ORDER), *geometry_config.output_size)
    if channels.dtype != np.float32 or channels.ndim != 4:
        raise FragmentGeometryCacheError("cached channels type or rank is invalid")
    if tuple(channels.shape[1:]) != expected_tail:
        raise FragmentGeometryCacheError("cached channels shape is invalid")
    total = int(channels.shape[0])
    if valid.dtype != np.bool_ or tuple(valid.shape) != (total,):
        raise FragmentGeometryCacheError("cached validity shape is invalid")
    if patch_int.dtype != np.int64 or tuple(patch_int.shape) != (
        total,
        _PATCH_INT_FIELDS,
    ):
        raise FragmentGeometryCacheError("cached patch integer shape is invalid")
    if patch_float.dtype != np.float64 or tuple(patch_float.shape) != (
        total,
        _PATCH_FLOAT_FIELDS,
    ):
        raise FragmentGeometryCacheError("cached patch float shape is invalid")
    if not np.all(np.isfinite(channels)) or not np.all(np.isfinite(patch_float)):
        raise FragmentGeometryCacheError("cached geometry contains non-finite values")
    tolerance = 1e-6
    channel_bounds = ((0.0, 1.0), (-1.0, 1.0), (0.0, 1.0))
    for channel_index, (lower, upper) in enumerate(channel_bounds):
        values = channels[:, channel_index]
        if values.size and (
            float(np.min(values)) < lower - tolerance
            or float(np.max(values)) > upper + tolerance
        ):
            raise FragmentGeometryCacheError(
                "cached channel values violate semantic range"
            )
    if not np.all(valid):
        raise FragmentGeometryCacheError("cached emitted patches must all be valid")
    if np.any(patch_int < 0):
        raise FragmentGeometryCacheError("cached patch indices must be non-negative")
    return channels, valid, patch_int, patch_float


def _patch_distance_contract(
    run_length_px: float,
    resolved_window_px: float,
    stride_px: float,
) -> Tuple[int, bool, float, float]:
    """Return regular count, append-stop flag, regular start, and stop."""

    if run_length_px <= resolved_window_px:
        return 1, False, run_length_px / 2.0, run_length_px / 2.0
    start = resolved_window_px / 2.0
    stop = run_length_px - resolved_window_px / 2.0
    regular_count = int(math.floor((stop + 1e-9 - start) / stride_px)) + 1
    if regular_count < 1:
        raise FragmentGeometryCacheError("sequence patch-distance contract is empty")
    last = start + (regular_count - 1) * stride_px
    append_stop = stop - last > max(1e-6, stride_px * 0.1)
    return regular_count, append_stop, start, stop


def fragment_result_from_cache_artifact(
    cache_artifact: GeometryCacheArtifact,
    geometry_config: CandidateBuilderConfig,
) -> FragmentGeometryResult:
    """Strictly decode a cryptographically verified generic cache artifact."""

    if not isinstance(cache_artifact, GeometryCacheArtifact):
        raise TypeError("cache_artifact must be GeometryCacheArtifact")
    if not isinstance(geometry_config, CandidateBuilderConfig):
        raise TypeError("geometry_config must be CandidateBuilderConfig")
    # Defense in depth for artifacts retained in memory: a caller cannot
    # mutate a nested array/identity and continue using the stale digest.
    cache_artifact.verify_logical_payload()
    metadata = _mapping(cache_artifact.metadata, "fragment metadata")
    expected_top = (
        "payload_schema_version",
        "artifact_version",
        "geometry_version",
        "role_policy",
        "config_fingerprint",
        "status",
        "failure_reason",
        "artifact",
    )
    _exact_keys(metadata, expected_top, "fragment metadata")
    if metadata["payload_schema_version"] != FRAGMENT_CACHE_PAYLOAD_SCHEMA_VERSION:
        raise FragmentGeometryCacheError("fragment payload schema mismatch")
    if metadata["artifact_version"] != FRAGMENT_GEOMETRY_ARTIFACT_VERSION:
        raise FragmentGeometryCacheError("fragment artifact version mismatch")
    if metadata["geometry_version"] != GEOMETRY_VERSION:
        raise FragmentGeometryCacheError("fragment geometry version mismatch")
    if metadata["role_policy"] != "single_fragment_role_neutral":
        raise FragmentGeometryCacheError("fragment role policy mismatch")
    expected_fingerprint = geometry_config_fingerprint(geometry_config)
    if metadata["config_fingerprint"] != expected_fingerprint:
        raise FragmentGeometryCacheError("fragment config fingerprint mismatch")
    channels, valid, patch_int, patch_float = _checked_arrays(
        cache_artifact, geometry_config
    )
    try:
        status = GeometryStatus(metadata["status"])
    except (TypeError, ValueError) as exc:
        raise FragmentGeometryCacheError("fragment status is invalid") from exc

    if status is not GeometryStatus.OK:
        if metadata["artifact"] is not None or channels.shape[0] != 0:
            raise FragmentGeometryCacheError("failed fragment exposes partial data")
        reason = _string(metadata["failure_reason"], "failure_reason")
        return FragmentGeometryResult(
            status=status, failure_reason=reason, artifact=None
        )
    if metadata["failure_reason"] is not None:
        raise FragmentGeometryCacheError("successful fragment has failure reason")

    value = _mapping(metadata["artifact"], "fragment artifact")
    expected_artifact = (
        "window_scale_fractions",
        "run_indices_by_side",
        "sequences",
        "sequence_failures",
        "quality",
    )
    _exact_keys(value, expected_artifact, "fragment artifact")
    quality = _quality_from_dict(value["quality"], geometry_config)
    scale_fractions = tuple(
        _finite_float(part, "window scale fraction")
        for part in _list(value["window_scale_fractions"], "window_scale_fractions")
    )
    if len(scale_fractions) != len(geometry_config.window_scale_fractions) or any(
        abs(observed - expected) > 1e-12
        for observed, expected in zip(
            scale_fractions, geometry_config.window_scale_fractions
        )
    ):
        raise FragmentGeometryCacheError(
            "artifact window scales disagree with geometry config"
        )
    run_map_value = _mapping(value["run_indices_by_side"], "run indices")
    side_names = tuple(side.value for side in CardinalSide)
    _exact_keys(run_map_value, side_names, "run indices")
    run_indices_by_side = tuple(
        (
            name,
            tuple(
                _integer(part, "run index")
                for part in _list(run_map_value[name], "run indices for " + name)
            ),
        )
        for name in side_names
    )

    sequences = []
    expected_offset = 0
    sequence_fields = (
        "side",
        "run_index",
        "scale_index",
        "scale_fraction",
        "bbox_reference_px",
        "requested_window_px",
        "resolved_window_px",
        "stride_px",
        "run_length_px",
        "direction_rule",
        "offset",
        "length",
    )
    for raw_sequence in _list(value["sequences"], "sequences"):
        item = _mapping(raw_sequence, "sequence")
        _exact_keys(item, sequence_fields, "sequence")
        offset = _integer(item["offset"], "sequence.offset")
        length = _integer(item["length"], "sequence.length", minimum=1)
        if offset != expected_offset or offset + length > channels.shape[0]:
            raise FragmentGeometryCacheError("sequence offsets are not contiguous")
        try:
            side = CardinalSide(item["side"])
        except (TypeError, ValueError) as exc:
            raise FragmentGeometryCacheError("sequence side is invalid") from exc
        run_index = _integer(item["run_index"], "sequence.run_index")
        scale_index = _integer(item["scale_index"], "sequence.scale_index")
        if run_index not in dict(run_indices_by_side)[side.value]:
            raise FragmentGeometryCacheError(
                "sequence run index is absent from side provenance"
            )
        if scale_index >= len(scale_fractions):
            raise FragmentGeometryCacheError("sequence scale index is invalid")
        scale_fraction = _finite_float(
            item["scale_fraction"], "sequence.scale_fraction"
        )
        if abs(scale_fraction - scale_fractions[scale_index]) > 1e-12:
            raise FragmentGeometryCacheError(
                "sequence scale fraction disagrees with scale index"
            )
        run_length_px = _finite_float(item["run_length_px"], "sequence.run_length_px")
        if run_length_px <= 0.0:
            raise FragmentGeometryCacheError("sequence run length must be positive")
        if run_length_px > quality.contour_perimeter_px + 1e-6:
            raise FragmentGeometryCacheError(
                "sequence run length exceeds contour perimeter"
            )
        minimum_run_length = max(
            geometry_config.min_run_length_px,
            geometry_config.min_run_length_fraction * quality.bbox_reference_px,
        )
        if run_length_px + 1e-9 < minimum_run_length:
            raise FragmentGeometryCacheError(
                "sequence run length violates geometry config"
            )
        bbox_reference_px = _finite_float(
            item["bbox_reference_px"], "sequence.bbox_reference_px"
        )
        requested_window_px = _finite_float(
            item["requested_window_px"], "sequence.requested_window_px"
        )
        resolved_window_px = _finite_float(
            item["resolved_window_px"], "sequence.resolved_window_px"
        )
        stride_px = _finite_float(item["stride_px"], "sequence.stride_px")
        if (
            min(
                bbox_reference_px,
                requested_window_px,
                resolved_window_px,
                stride_px,
            )
            <= 0.0
        ):
            raise FragmentGeometryCacheError(
                "sequence bbox/window/stride values must be positive"
            )
        expected_requested = quality.bbox_reference_px * scale_fraction
        expected_resolved = float(
            np.clip(
                expected_requested,
                geometry_config.window_min_px,
                geometry_config.window_max_px,
            )
        )
        expected_stride = expected_resolved * (1.0 - geometry_config.overlap_fraction)
        if (
            abs(bbox_reference_px - quality.bbox_reference_px) > 1e-9
            or abs(requested_window_px - expected_requested) > 1e-9
            or abs(resolved_window_px - expected_resolved) > 1e-9
            or abs(stride_px - expected_stride) > 1e-9
        ):
            raise FragmentGeometryCacheError(
                "sequence window provenance disagrees with geometry config"
            )
        expected_direction_rule = (
            "top_to_bottom"
            if side in {CardinalSide.LEFT, CardinalSide.RIGHT}
            else "left_to_right"
        )
        direction_rule = _string(item["direction_rule"], "sequence.direction_rule")
        if direction_rule != expected_direction_rule:
            raise FragmentGeometryCacheError(
                "sequence direction rule disagrees with frozen side convention"
            )
        regular_count, append_stop, distance_start, distance_stop = (
            _patch_distance_contract(run_length_px, resolved_window_px, stride_px)
        )
        expected_length = regular_count + int(append_stop)
        if length != expected_length:
            raise FragmentGeometryCacheError(
                "sequence length disagrees with sliding-window contract"
            )
        sequence_patches = []
        for local_index in range(length):
            index = offset + local_index
            ints = patch_int[index]
            floats = patch_float[index]
            if int(ints[0]) != local_index:
                raise FragmentGeometryCacheError(
                    "patch sequence indices are not contiguous"
                )
            valid_fraction = float(floats[8])
            padding_fraction = float(floats[9])
            if not (0.0 <= valid_fraction <= 1.0) or not (
                0.0 <= padding_fraction <= 1.0
            ):
                raise FragmentGeometryCacheError("cached patch fractions are invalid")
            if abs(valid_fraction + padding_fraction - 1.0) > 1e-6:
                raise FragmentGeometryCacheError(
                    "cached valid and padding fractions disagree"
                )
            if (
                int(ints[1]) >= quality.contour_points
                or int(ints[2]) >= quality.contour_points
            ):
                raise FragmentGeometryCacheError(
                    "cached contour segment index is out of range"
                )
            contour_step = (int(ints[2]) - int(ints[1])) % quality.contour_points
            if contour_step not in {1, quality.contour_points - 1}:
                raise FragmentGeometryCacheError(
                    "cached contour segment indices are not adjacent"
                )
            row, column = float(floats[0]), float(floats[1])
            if not (
                0.0 <= row <= quality.input_shape[0]
                and 0.0 <= column <= quality.input_shape[1]
            ):
                raise FragmentGeometryCacheError(
                    "cached patch center exceeds input frame"
                )
            tangent = np.asarray(floats[2:4], dtype=np.float64)
            normal = np.asarray(floats[4:6], dtype=np.float64)
            if (
                abs(float(np.linalg.norm(tangent)) - 1.0) > 1e-5
                or abs(float(np.linalg.norm(normal)) - 1.0) > 1e-5
                or abs(float(np.dot(tangent, normal))) > 1e-5
            ):
                raise FragmentGeometryCacheError(
                    "cached patch frame is not orthonormal"
                )
            if not 0.0 <= float(floats[6]) <= 1.0:
                raise FragmentGeometryCacheError(
                    "cached contour arc fraction is invalid"
                )
            expected_distance = (
                distance_stop
                if append_stop and local_index == length - 1
                else distance_start + local_index * stride_px
            )
            if (
                not 0.0 <= float(floats[7]) <= run_length_px + 1e-6
                or abs(float(floats[7]) - expected_distance) > 1e-6
            ):
                raise FragmentGeometryCacheError(
                    "cached patch path distance violates sliding-window contract"
                )
            sequence_patches.append(
                PatchProvenance(
                    sequence_index=local_index,
                    center_row_col=(float(floats[0]), float(floats[1])),
                    tangent_row_col=(float(floats[2]), float(floats[3])),
                    inward_normal_row_col=(float(floats[4]), float(floats[5])),
                    contour_segment_indices=(int(ints[1]), int(ints[2])),
                    contour_arc_fraction=float(floats[6]),
                    path_distance_px=float(floats[7]),
                    valid_fraction=valid_fraction,
                    padding_fraction=padding_fraction,
                )
            )
        sequence_channels = np.ascontiguousarray(
            channels[offset : offset + length], dtype=np.float32
        )
        sequence_valid = np.ascontiguousarray(
            valid[offset : offset + length], dtype=np.bool_
        )
        sequence_channels.setflags(write=False)
        sequence_valid.setflags(write=False)
        sequences.append(
            PatchSequence(
                fragment_role="fragment",
                side=side,
                run_index=run_index,
                scale_index=scale_index,
                scale_fraction=scale_fraction,
                bbox_reference_px=bbox_reference_px,
                requested_window_px=requested_window_px,
                resolved_window_px=resolved_window_px,
                stride_px=stride_px,
                run_length_px=run_length_px,
                direction_rule=direction_rule,
                channels=sequence_channels,
                valid=sequence_valid,
                patches=tuple(sequence_patches),
            )
        )
        expected_offset += length
    if expected_offset != channels.shape[0]:
        raise FragmentGeometryCacheError("unclaimed patch rows remain in cache")

    failures = []
    failure_fields = ("side", "run_index", "scale_index", "reason")
    for raw_failure in _list(value["sequence_failures"], "sequence_failures"):
        item = _mapping(raw_failure, "sequence failure")
        _exact_keys(item, failure_fields, "sequence failure")
        try:
            side = CardinalSide(item["side"])
        except (TypeError, ValueError) as exc:
            raise FragmentGeometryCacheError("failure side is invalid") from exc
        failure_run_index = _integer(item["run_index"], "failure.run_index")
        failure_scale_index = _integer(item["scale_index"], "failure.scale_index")
        if failure_run_index not in dict(run_indices_by_side)[side.value]:
            raise FragmentGeometryCacheError(
                "failure run index is absent from side provenance"
            )
        if failure_scale_index >= len(scale_fractions):
            raise FragmentGeometryCacheError("failure scale index is invalid")
        failures.append(
            FragmentSequenceFailure(
                side=side,
                run_index=failure_run_index,
                scale_index=failure_scale_index,
                reason=_string(item["reason"], "failure.reason"),
            )
        )
    try:
        fragment = FragmentGeometryArtifact(
            config_fingerprint=expected_fingerprint,
            window_scale_fractions=scale_fractions,
            run_indices_by_side=run_indices_by_side,
            sequences=tuple(sequences),
            sequence_failures=tuple(failures),
            quality=quality,
        )
    except (TypeError, ValueError) as exc:
        raise FragmentGeometryCacheError("fragment artifact invariant failed") from exc
    return FragmentGeometryResult(
        status=GeometryStatus.OK,
        failure_reason=None,
        artifact=fragment,
    )


def load_or_build_fragment_geometry(
    mask: np.ndarray,
    threshold_rule: str,
    geometry_config: CandidateBuilderConfig,
    cache: GeometryArtifactCache,
) -> FragmentGeometryCacheLookup:
    """Read or compute one fragment using only content/config cache identity."""

    if not isinstance(cache, GeometryArtifactCache):
        raise TypeError("cache must be GeometryArtifactCache")
    identity = fragment_cache_identity(mask, threshold_rule, geometry_config)

    def producer() -> Tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
        result = build_fragment_geometry(mask, geometry_config)
        return fragment_result_to_cache_payload(result, geometry_config)

    lookup = cache.get_or_compute(identity, producer)
    result = fragment_result_from_cache_artifact(lookup.artifact, geometry_config)
    return FragmentGeometryCacheLookup(
        identity=identity,
        result=result,
        cache_hit=lookup.cache_hit,
        logical_payload_sha256=lookup.artifact.logical_payload_sha256,
    )


__all__ = [
    "FRAGMENT_CACHE_PAYLOAD_SCHEMA_VERSION",
    "FragmentGeometryCacheError",
    "FragmentGeometryCacheLookup",
    "fragment_result_from_cache_artifact",
    "fragment_result_to_cache_payload",
    "load_or_build_fragment_geometry",
]
