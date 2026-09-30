"""Known-upright multi-run contour and sliding-window patch preprocessing.

This module reuses the audited v0.1 external-contour and cardinal multi-run
extractors, then adds only the v0.2 contract needed by the local matcher:

* no rotation search;
* four complementary facing-edge families when relative direction is unknown;
* deterministic top-to-bottom / left-to-right run ordering;
* overlapping, bbox-normalized multi-scale windows;
* mask, signed-distance, and boundary-gradient channels; and
* variable-length, immutable sequences with full provenance.

It contains no learned model and never uses pair labels to select an arc.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage

from staging.pairwise_v0_1.baselines.b1_contours import (
    BoundarySaddlePolicy,
    CardinalSide,
    CanonicalContour,
    ContourPreprocessConfig,
    ContourSide,
    ForegroundPolarity,
    extract_cardinal_side_runs,
    preprocess_mask_contour_only,
)
from staging.pairwise_v0_2.geometry.schema import (
    ArcPairCandidate,
    CandidateBuilderConfig,
    DEFAULT_DIRECTION_ORDER,
    DirectionCandidateGroup,
    FRAGMENT_GEOMETRY_ARTIFACT_VERSION,
    FragmentGeometryArtifact,
    FragmentGeometryQuality,
    FragmentGeometryResult,
    FragmentSequenceFailure,
    GEOMETRY_VERSION,
    GeometryStatus,
    PairCandidateResult,
    PairDirection,
    PairGeometryQuality,
    PatchProvenance,
    PatchSequence,
)


MaskInput = Union[np.ndarray, Image.Image]


@dataclass(frozen=True)
class _OrderedRun:
    side: CardinalSide
    run_index: int
    points: np.ndarray
    contour_indices: np.ndarray
    contour_arc_fractions: np.ndarray
    contour_point_count: int
    cumulative_lengths: np.ndarray
    length_px: float
    direction_rule: str


@dataclass(frozen=True)
class _FragmentGeometry:
    role: str
    contour: CanonicalContour
    runs: Mapping[CardinalSide, Tuple[_OrderedRun, ...]]
    component_mask: np.ndarray
    signed_distance_px: np.ndarray
    boundary_gradient: np.ndarray
    quality: FragmentGeometryQuality


def _readonly(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    output = np.asarray(array, dtype=dtype).copy()
    output.setflags(write=False)
    return output


def _base_quality(
    requested: Sequence[PairDirection],
    fragment_a: Optional[FragmentGeometryQuality] = None,
    fragment_b: Optional[FragmentGeometryQuality] = None,
) -> PairGeometryQuality:
    return PairGeometryQuality(
        requested_directions=tuple(direction.value for direction in requested),
        emitted_directions=(),
        candidate_count=0,
        sequence_count_a=0,
        sequence_count_b=0,
        fragment_a=fragment_a,
        fragment_b=fragment_b,
    )


def _failure(
    status: GeometryStatus,
    reason: str,
    quality: PairGeometryQuality,
) -> PairCandidateResult:
    return PairCandidateResult(
        status=status,
        failure_reason=reason,
        candidates=(),
        direction_groups=(),
        quality=quality,
    )


def _mask_array(mask: MaskInput) -> np.ndarray:
    if isinstance(mask, Image.Image):
        return np.asarray(mask)
    return np.asarray(mask)


def _largest_component_from_receipt(
    raw: np.ndarray,
    threshold: float,
    polarity: str,
    connectivity: int,
    expected_pixels: int,
    expected_bbox: Tuple[int, int, int, int],
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    numeric = raw.astype(np.float64, copy=False)
    bright = numeric > threshold
    foreground = bright if polarity == "bright" else ~bright
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, component_count = ndimage.label(foreground, structure=structure)
    if component_count < 1:
        return None, "component_reconstruction_found_no_foreground"

    counts = np.bincount(labels.reshape(-1), minlength=component_count + 1)
    objects = ndimage.find_objects(labels, max_label=component_count)
    matches: List[Tuple[Tuple[int, int, int, int], int]] = []
    for label in np.flatnonzero(counts[1:] == expected_pixels) + 1:
        component_slice = objects[int(label) - 1]
        if component_slice is None:
            continue
        bbox = (
            int(component_slice[0].start),
            int(component_slice[1].start),
            int(component_slice[0].stop),
            int(component_slice[1].stop),
        )
        matches.append((bbox, int(label)))
    matches.sort()
    selected = [label for bbox, label in matches if bbox == expected_bbox]
    if len(selected) != 1:
        return None, "largest_component_does_not_match_contour_receipt"
    return labels == selected[0], None


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _ordered_run(
    side: ContourSide,
    run_index: int,
    contour_point_count: int,
) -> _OrderedRun:
    points = np.asarray(side.points, dtype=np.float64).copy()
    indices = np.asarray(side.contour_indices, dtype=np.int64).copy()
    fractions = np.asarray(side.contour_arc_fractions, dtype=np.float64).copy()
    if side.side in {CardinalSide.LEFT, CardinalSide.RIGHT}:
        direction_rule = "top_to_bottom"
        forward = (points[-1, 0], points[-1, 1]) >= (points[0, 0], points[0, 1])
    else:
        direction_rule = "left_to_right"
        forward = (points[-1, 1], points[-1, 0]) >= (points[0, 1], points[0, 0])
    if not forward:
        points = points[::-1].copy()
        indices = indices[::-1].copy()
        fractions = fractions[::-1].copy()
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if len(points) < 2 or np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
        raise ValueError("side_run_contains_duplicate_or_invalid_points")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return _OrderedRun(
        side=side.side,
        run_index=run_index,
        points=_readonly(points, np.float64),
        contour_indices=_readonly(indices, np.int64),
        contour_arc_fractions=_readonly(fractions, np.float64),
        contour_point_count=contour_point_count,
        cumulative_lengths=_readonly(cumulative, np.float64),
        length_px=float(cumulative[-1]),
        direction_rule=direction_rule,
    )


def _fragment_geometry(
    mask: MaskInput,
    role: str,
    config: CandidateBuilderConfig,
) -> Tuple[Optional[_FragmentGeometry], GeometryStatus, Optional[str]]:
    try:
        raw = _mask_array(mask)
    except (TypeError, ValueError):
        return None, GeometryStatus.INVALID_INPUT, "mask_conversion_failed"

    contour_config = ContourPreprocessConfig(
        polarity=ForegroundPolarity(config.foreground_polarity),
        connectivity=config.connectivity,
        saddle_policy=BoundarySaddlePolicy(config.saddle_policy),
        min_component_pixels=config.min_component_pixels,
        min_contour_points=config.min_contour_points,
        contour_resample_count=max(64, config.side_resample_count * 2),
        side_resample_count=config.side_resample_count,
    )
    contour_result = preprocess_mask_contour_only(raw, contour_config)
    if not contour_result.ok or contour_result.contour is None:
        reason = contour_result.failure_reason or "contour_preprocessing_failed"
        status = (
            GeometryStatus.INVALID_INPUT
            if reason.startswith("mask_")
            else GeometryStatus.INVALID_CONTOUR
        )
        return None, status, "%s:%s" % (role, reason)

    receipt = contour_result.quality
    if (
        receipt.threshold is None
        or receipt.resolved_polarity is None
        or receipt.largest_component_bbox_rc_exclusive is None
    ):
        return (
            None,
            GeometryStatus.INVALID_CONTOUR,
            "%s:incomplete_contour_receipt" % role,
        )
    bbox = receipt.largest_component_bbox_rc_exclusive
    component, component_reason = _largest_component_from_receipt(
        raw,
        receipt.threshold,
        receipt.resolved_polarity,
        config.connectivity,
        receipt.largest_component_pixels,
        bbox,
    )
    if component is None:
        return (
            None,
            GeometryStatus.INVALID_CONTOUR,
            "%s:%s"
            % (
                role,
                component_reason or "component_reconstruction_failed",
            ),
        )

    runs_result = extract_cardinal_side_runs(
        contour_result.contour,
        min_side_points=2,
        side_resample_count=config.side_resample_count,
    )
    if not runs_result.ok:
        return (
            None,
            GeometryStatus.INVALID_CONTOUR,
            "%s:%s"
            % (
                role,
                runs_result.failure_reason or "side_run_extraction_failed",
            ),
        )

    bbox_height = bbox[2] - bbox[0]
    bbox_width = bbox[3] - bbox[1]
    bbox_reference = float(min(bbox_height, bbox_width))
    minimum_run_length = max(
        config.min_run_length_px,
        config.min_run_length_fraction * bbox_reference,
    )
    retained: Dict[CardinalSide, Tuple[_OrderedRun, ...]] = {}
    dropped_by_side: Dict[CardinalSide, int] = {}
    dropped_below_two = dict(runs_result.quality.dropped_short_run_counts)
    for side in CardinalSide:
        side_runs: List[_OrderedRun] = []
        dropped = int(dropped_below_two.get(side.value, 0))
        for original_index, raw_run in enumerate(runs_result.runs[side]):
            if _path_length(raw_run.points) + 1e-9 < minimum_run_length:
                dropped += 1
                continue
            try:
                side_runs.append(
                    _ordered_run(
                        raw_run,
                        original_index,
                        contour_result.contour.point_count,
                    )
                )
            except ValueError:
                return (
                    None,
                    GeometryStatus.INVALID_CONTOUR,
                    ("%s:side_run_contains_duplicate_or_invalid_points" % role),
                )
        retained[side] = tuple(side_runs)
        dropped_by_side[side] = dropped

    if not any(retained.values()):
        return (
            None,
            GeometryStatus.NO_USABLE_CANDIDATES,
            "%s:all_side_runs_too_short" % role,
        )

    component_float = component.astype(np.float32)
    inside_distance = ndimage.distance_transform_edt(component)
    outside_distance = ndimage.distance_transform_edt(~component)
    signed_distance = (inside_distance - outside_distance).astype(np.float32)
    gradient_row = ndimage.sobel(component_float, axis=0, mode="constant")
    gradient_column = ndimage.sobel(component_float, axis=1, mode="constant")
    gradient = np.hypot(gradient_row, gradient_column).astype(np.float32)
    gradient_max = float(np.max(gradient))
    if not np.isfinite(gradient_max) or gradient_max <= 0.0:
        return (
            None,
            GeometryStatus.INVALID_CONTOUR,
            "%s:boundary_gradient_is_degenerate" % role,
        )
    gradient /= gradient_max

    retained_counts = tuple((side.value, len(retained[side])) for side in CardinalSide)
    dropped_counts = tuple((side.value, dropped_by_side[side]) for side in CardinalSide)
    warnings = list(receipt.warnings)
    if any(dropped_by_side.values()):
        warnings.append("short_cardinal_runs_dropped_by_normalized_length")
    quality = FragmentGeometryQuality(
        role=role,
        input_shape=tuple(int(value) for value in raw.shape),
        input_dtype=str(raw.dtype),
        foreground_pixels=receipt.foreground_pixel_count,
        component_count=receipt.component_count,
        largest_component_pixels=receipt.largest_component_pixels,
        discarded_foreground_pixels=receipt.discarded_foreground_pixels,
        bbox_rc_exclusive=bbox,
        bbox_reference_px=bbox_reference,
        contour_points=contour_result.contour.point_count,
        contour_perimeter_px=contour_result.contour.perimeter_px,
        retained_runs_by_side=retained_counts,
        dropped_short_runs_by_side=dropped_counts,
        warnings=tuple(warnings),
    )
    return (
        _FragmentGeometry(
            role=role,
            contour=contour_result.contour,
            runs=retained,
            component_mask=_readonly(component_float, np.float32),
            signed_distance_px=_readonly(signed_distance, np.float32),
            boundary_gradient=_readonly(gradient, np.float32),
            quality=quality,
        ),
        GeometryStatus.OK,
        None,
    )


def _interpolate_run(
    run: _OrderedRun,
    distance: float,
) -> Tuple[np.ndarray, int, float, float]:
    bounded = float(np.clip(distance, 0.0, run.length_px))
    segment = int(np.searchsorted(run.cumulative_lengths, bounded, side="right") - 1)
    segment = min(max(segment, 0), len(run.points) - 2)
    segment_length = (
        run.cumulative_lengths[segment + 1] - run.cumulative_lengths[segment]
    )
    alpha = float((bounded - run.cumulative_lengths[segment]) / segment_length)
    point = run.points[segment] + alpha * (
        run.points[segment + 1] - run.points[segment]
    )

    start_fraction = float(run.contour_arc_fractions[segment])
    end_fraction = float(run.contour_arc_fractions[segment + 1])
    start_index = int(run.contour_indices[segment])
    end_index = int(run.contour_indices[segment + 1])
    forward_delta = (end_index - start_index) % run.contour_point_count
    backward_delta = (start_index - end_index) % run.contour_point_count
    if forward_delta <= backward_delta:
        while end_fraction < start_fraction:
            end_fraction += 1.0
    else:
        while end_fraction > start_fraction:
            end_fraction -= 1.0
    arc_fraction = float(
        (start_fraction + alpha * (end_fraction - start_fraction)) % 1.0
    )
    return point, segment, alpha, arc_fraction


def _scalar_bilinear(
    image: np.ndarray, row: float, column: float, padding: float
) -> float:
    rows, columns = image.shape
    row_floor = int(np.floor(row))
    column_floor = int(np.floor(column))
    row_fraction = row - row_floor
    column_fraction = column - column_floor
    value = 0.0
    for delta_row, row_weight in ((0, 1.0 - row_fraction), (1, row_fraction)):
        for delta_column, column_weight in (
            (0, 1.0 - column_fraction),
            (1, column_fraction),
        ):
            sample_row = row_floor + delta_row
            sample_column = column_floor + delta_column
            sample = (
                float(image[sample_row, sample_column])
                if 0 <= sample_row < rows and 0 <= sample_column < columns
                else padding
            )
            value += row_weight * column_weight * sample
    return float(value)


def _expected_inward(side: CardinalSide) -> np.ndarray:
    if side is CardinalSide.LEFT:
        return np.asarray((0.0, 1.0), dtype=np.float64)
    if side is CardinalSide.RIGHT:
        return np.asarray((0.0, -1.0), dtype=np.float64)
    if side is CardinalSide.TOP:
        return np.asarray((1.0, 0.0), dtype=np.float64)
    return np.asarray((-1.0, 0.0), dtype=np.float64)


def _choose_inward_normal(
    signed_distance: np.ndarray,
    center_cell: np.ndarray,
    tangent: np.ndarray,
    side: CardinalSide,
    probe_distance: float,
) -> np.ndarray:
    option = np.asarray((tangent[1], -tangent[0]), dtype=np.float64)
    image_center = center_cell - 0.5
    score_positive = _scalar_bilinear(
        signed_distance,
        float(image_center[0] + probe_distance * option[0]),
        float(image_center[1] + probe_distance * option[1]),
        -probe_distance,
    )
    score_negative = _scalar_bilinear(
        signed_distance,
        float(image_center[0] - probe_distance * option[0]),
        float(image_center[1] - probe_distance * option[1]),
        -probe_distance,
    )
    if abs(score_positive - score_negative) > 1e-6:
        return option if score_positive > score_negative else -option
    return option if float(np.dot(option, _expected_inward(side))) >= 0.0 else -option


def _sample_channels(
    image: np.ndarray,
    grid_rows: np.ndarray,
    grid_columns: np.ndarray,
    padding: np.ndarray,
) -> np.ndarray:
    rows, columns, channels = image.shape
    row_floor = np.floor(grid_rows).astype(np.int64)
    column_floor = np.floor(grid_columns).astype(np.int64)
    row_fraction = grid_rows - row_floor
    column_fraction = grid_columns - column_floor
    output = (
        np.broadcast_to(padding, grid_rows.shape + (channels,))
        .astype(np.float64)
        .copy()
    )
    neighbours = (
        (row_floor, column_floor, (1.0 - row_fraction) * (1.0 - column_fraction)),
        (row_floor, column_floor + 1, (1.0 - row_fraction) * column_fraction),
        (row_floor + 1, column_floor, row_fraction * (1.0 - column_fraction)),
        (row_floor + 1, column_floor + 1, row_fraction * column_fraction),
    )
    for neighbour_rows, neighbour_columns, weights in neighbours:
        valid = (
            (neighbour_rows >= 0)
            & (neighbour_rows < rows)
            & (neighbour_columns >= 0)
            & (neighbour_columns < columns)
        )
        if not np.any(valid):
            continue
        values = image[neighbour_rows[valid], neighbour_columns[valid]]
        output[valid] += weights[valid, None] * (values - padding)
    return output.astype(np.float32)


def _patch_distances(
    run_length: float, window_px: float, stride_px: float
) -> np.ndarray:
    if run_length <= window_px:
        return np.asarray((run_length / 2.0,), dtype=np.float64)
    start = window_px / 2.0
    stop = run_length - window_px / 2.0
    targets = list(np.arange(start, stop + 1e-9, stride_px, dtype=np.float64))
    if not targets or stop - targets[-1] > max(1e-6, stride_px * 0.1):
        targets.append(stop)
    return np.asarray(targets, dtype=np.float64)


def _build_sequence(
    fragment: _FragmentGeometry,
    run: _OrderedRun,
    scale_index: int,
    scale_fraction: float,
    config: CandidateBuilderConfig,
) -> Tuple[Optional[PatchSequence], Optional[str]]:
    requested_window = fragment.quality.bbox_reference_px * scale_fraction
    window_px = float(
        np.clip(requested_window, config.window_min_px, config.window_max_px)
    )
    stride_px = window_px * (1.0 - config.overlap_fraction)
    targets = _patch_distances(run.length_px, window_px, stride_px)
    output_rows, output_columns = config.output_size
    local_rows = (
        np.arange(output_rows, dtype=np.float64) - (output_rows - 1) / 2.0
    ) * (window_px / output_rows)
    local_columns = (
        np.arange(output_columns, dtype=np.float64) - (output_columns - 1) / 2.0
    ) * (window_px / output_columns)
    distance_clip = max(1e-6, window_px * config.signed_distance_clip_fraction)
    normalized_sdf = np.clip(fragment.signed_distance_px / distance_clip, -1.0, 1.0)
    feature_image = np.stack(
        (fragment.component_mask, normalized_sdf, fragment.boundary_gradient),
        axis=2,
    ).astype(np.float32)
    tangent_span = max(1.0, window_px * config.tangent_span_fraction)
    probe_distance = max(1.0, window_px * config.inward_probe_fraction)

    patch_arrays: List[np.ndarray] = []
    provenance: List[PatchProvenance] = []
    image_rows, image_columns = fragment.component_mask.shape
    for sequence_index, target in enumerate(targets):
        center, segment, alpha, arc_fraction = _interpolate_run(run, float(target))
        before, _, _, _ = _interpolate_run(
            run, max(0.0, float(target) - tangent_span / 2.0)
        )
        after, _, _, _ = _interpolate_run(
            run,
            min(run.length_px, float(target) + tangent_span / 2.0),
        )
        tangent = after - before
        tangent_norm = float(np.linalg.norm(tangent))
        if not np.isfinite(tangent_norm) or tangent_norm <= 1e-8:
            tangent = run.points[segment + 1] - run.points[segment]
            tangent_norm = float(np.linalg.norm(tangent))
        if not np.isfinite(tangent_norm) or tangent_norm <= 1e-8:
            return None, "degenerate_local_tangent"
        tangent /= tangent_norm
        inward = _choose_inward_normal(
            fragment.signed_distance_px,
            center,
            tangent,
            run.side,
            probe_distance,
        )
        image_center = center - 0.5
        grid_rows = (
            image_center[0]
            + local_rows[:, None] * inward[0]
            + local_columns[None, :] * tangent[0]
        )
        grid_columns = (
            image_center[1]
            + local_rows[:, None] * inward[1]
            + local_columns[None, :] * tangent[1]
        )
        valid_grid = (
            (grid_rows >= 0.0)
            & (grid_rows <= image_rows - 1)
            & (grid_columns >= 0.0)
            & (grid_columns <= image_columns - 1)
        )
        valid_fraction = float(np.mean(valid_grid))
        if valid_fraction < config.min_image_valid_fraction:
            return None, "patch_image_valid_fraction_below_minimum"
        sampled = _sample_channels(
            feature_image,
            grid_rows,
            grid_columns,
            np.asarray((0.0, -1.0, 0.0), dtype=np.float64),
        )
        channel_first = np.transpose(sampled, (2, 0, 1))
        if not np.all(np.isfinite(channel_first)):
            return None, "sampled_patch_contains_nonfinite_values"
        patch_arrays.append(channel_first)
        contour_pair = (
            int(run.contour_indices[segment]),
            int(run.contour_indices[segment + 1]),
        )
        provenance.append(
            PatchProvenance(
                sequence_index=sequence_index,
                center_row_col=(float(center[0]), float(center[1])),
                tangent_row_col=(float(tangent[0]), float(tangent[1])),
                inward_normal_row_col=(float(inward[0]), float(inward[1])),
                contour_segment_indices=contour_pair,
                contour_arc_fraction=arc_fraction,
                path_distance_px=float(target),
                valid_fraction=valid_fraction,
                padding_fraction=1.0 - valid_fraction,
            )
        )

    channels = _readonly(np.stack(patch_arrays, axis=0), np.float32)
    valid = _readonly(np.ones(len(patch_arrays), dtype=np.bool_), np.bool_)
    return (
        PatchSequence(
            fragment_role=fragment.role,
            side=run.side,
            run_index=run.run_index,
            scale_index=scale_index,
            scale_fraction=scale_fraction,
            bbox_reference_px=fragment.quality.bbox_reference_px,
            requested_window_px=float(requested_window),
            resolved_window_px=window_px,
            stride_px=stride_px,
            run_length_px=run.length_px,
            direction_rule=run.direction_rule,
            channels=channels,
            valid=valid,
            patches=tuple(provenance),
        ),
        None,
    )


def _parse_directions(
    direction_b_wrt_a: Optional[Union[PairDirection, str]],
) -> Tuple[Optional[Tuple[PairDirection, ...]], Optional[str]]:
    if direction_b_wrt_a is None:
        return DEFAULT_DIRECTION_ORDER, None
    try:
        return (PairDirection(direction_b_wrt_a),), None
    except (TypeError, ValueError):
        return None, "direction_b_wrt_a_is_invalid"


def geometry_config_fingerprint(config: CandidateBuilderConfig) -> str:
    """Hash every fragment-artifact-affecting geometry setting."""

    if not isinstance(config, CandidateBuilderConfig):
        raise TypeError("config must be CandidateBuilderConfig")
    payload = {
        "artifact_version": FRAGMENT_GEOMETRY_ARTIFACT_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "geometry_config": asdict(config),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _role_neutral_reason(reason: Optional[str]) -> str:
    value = reason or "fragment_preprocessing_failed"
    prefix = "fragment:"
    return value[len(prefix) :] if value.startswith(prefix) else value


def build_fragment_geometry(
    mask: MaskInput,
    config: Optional[CandidateBuilderConfig] = None,
) -> FragmentGeometryResult:
    """Precompute every role-neutral side/run/scale feature for one mask.

    This is the cacheable half of pair candidate construction.  It receives
    neither pair metadata nor supervision and always covers all four sides.
    Per-sequence failures are retained so later pair combination can reproduce
    the original fail-closed behavior without recomputing the mask geometry.
    """

    settings = config or CandidateBuilderConfig()
    if not isinstance(settings, CandidateBuilderConfig):
        raise TypeError("config must be CandidateBuilderConfig")
    if (
        settings.corrosion.enabled
        or settings.corrosion.erosion_radius_fraction > 0.0
        or settings.corrosion.missing_arc_fraction > 0.0
    ):
        return FragmentGeometryResult(
            status=GeometryStatus.UNSUPPORTED_CONFIGURATION,
            failure_reason=(
                "controlled_corrosion_is_reserved_but_disabled_in_v0_2_clean"
            ),
            artifact=None,
        )

    fragment, status, reason = _fragment_geometry(mask, "fragment", settings)
    if fragment is None:
        return FragmentGeometryResult(
            status=status,
            failure_reason=_role_neutral_reason(reason),
            artifact=None,
        )

    sequences: List[PatchSequence] = []
    failures: List[FragmentSequenceFailure] = []
    run_indices_by_side = tuple(
        (
            side.value,
            tuple(sorted(run.run_index for run in fragment.runs[side])),
        )
        for side in CardinalSide
    )
    for side in CardinalSide:
        for scale_index, scale_fraction in enumerate(settings.window_scale_fractions):
            for run in fragment.runs[side]:
                sequence, sequence_reason = _build_sequence(
                    fragment,
                    run,
                    scale_index,
                    scale_fraction,
                    settings,
                )
                if sequence is None:
                    failures.append(
                        FragmentSequenceFailure(
                            side=side,
                            run_index=run.run_index,
                            scale_index=scale_index,
                            reason=sequence_reason or "sequence_failed",
                        )
                    )
                else:
                    sequences.append(sequence)

    artifact = FragmentGeometryArtifact(
        config_fingerprint=geometry_config_fingerprint(settings),
        window_scale_fractions=settings.window_scale_fractions,
        run_indices_by_side=run_indices_by_side,
        sequences=tuple(sequences),
        sequence_failures=tuple(failures),
        quality=fragment.quality,
    )
    return FragmentGeometryResult(
        status=GeometryStatus.OK,
        failure_reason=None,
        artifact=artifact,
    )


def _fragment_quality_for_role(
    artifact: FragmentGeometryArtifact, role: str
) -> FragmentGeometryQuality:
    if role not in {"a", "b"}:
        raise ValueError("pair fragment role must be a or b")
    return replace(artifact.quality, role=role)


def combine_fragment_artifacts(
    fragment_a: FragmentGeometryArtifact,
    fragment_b: FragmentGeometryArtifact,
    direction_b_wrt_a: Optional[Union[PairDirection, str]] = None,
    config: Optional[CandidateBuilderConfig] = None,
) -> PairCandidateResult:
    """Combine two cached role-neutral artifacts into Pairwise candidates."""

    settings = config or CandidateBuilderConfig()
    if not isinstance(fragment_a, FragmentGeometryArtifact) or not isinstance(
        fragment_b, FragmentGeometryArtifact
    ):
        raise TypeError("fragment_a and fragment_b must be fragment artifacts")
    if not isinstance(settings, CandidateBuilderConfig):
        raise TypeError("config must be CandidateBuilderConfig")
    requested, direction_reason = _parse_directions(direction_b_wrt_a)
    if requested is None:
        return _failure(
            GeometryStatus.INVALID_INPUT,
            direction_reason or "invalid_direction",
            _base_quality(()),
        )

    quality_a = _fragment_quality_for_role(fragment_a, "a")
    quality_b = _fragment_quality_for_role(fragment_b, "b")
    quality = _base_quality(requested, quality_a, quality_b)
    expected_fingerprint = geometry_config_fingerprint(settings)
    if (
        fragment_a.config_fingerprint != expected_fingerprint
        or fragment_b.config_fingerprint != expected_fingerprint
        or fragment_a.window_scale_fractions != settings.window_scale_fractions
        or fragment_b.window_scale_fractions != settings.window_scale_fractions
    ):
        return _failure(
            GeometryStatus.INVALID_INPUT,
            "fragment_artifact_config_mismatch",
            quality,
        )

    source_a = fragment_a.sequence_map
    source_b = fragment_b.sequence_map
    failures_a = fragment_a.failure_map
    failures_b = fragment_b.failure_map
    runs_a_by_side = fragment_a.run_map
    runs_b_by_side = fragment_b.run_map
    role_cache_a: Dict[Tuple[CardinalSide, int, int], PatchSequence] = {}
    role_cache_b: Dict[Tuple[CardinalSide, int, int], PatchSequence] = {}
    candidates: List[ArcPairCandidate] = []
    direction_groups: List[DirectionCandidateGroup] = []
    emitted_directions: List[PairDirection] = []
    unavailable_directions: List[str] = []

    for direction in requested:
        direction_start = len(candidates)
        side_a, side_b = direction.facing_sides
        run_indices_a = runs_a_by_side[side_a]
        run_indices_b = runs_b_by_side[side_b]
        if not run_indices_a or not run_indices_b:
            unavailable_directions.append(direction.value)
            continue
        direction_candidate_count = 0
        for scale_index, _ in enumerate(settings.window_scale_fractions):
            for run_index_a in run_indices_a:
                key_a = (side_a, run_index_a, scale_index)
                failure_a = failures_a.get(key_a)
                if failure_a is not None:
                    return _failure(
                        GeometryStatus.NO_USABLE_CANDIDATES,
                        "a:%s:%s" % (direction.value, failure_a.reason),
                        quality,
                    )
                if key_a not in role_cache_a:
                    role_cache_a[key_a] = replace(
                        source_a[key_a], fragment_role="a"
                    )
                for run_index_b in run_indices_b:
                    key_b = (side_b, run_index_b, scale_index)
                    failure_b = failures_b.get(key_b)
                    if failure_b is not None:
                        return _failure(
                            GeometryStatus.NO_USABLE_CANDIDATES,
                            "b:%s:%s" % (direction.value, failure_b.reason),
                            quality,
                        )
                    if key_b not in role_cache_b:
                        role_cache_b[key_b] = replace(
                            source_b[key_b], fragment_role="b"
                        )
                    candidate_id = "%s:a-%s-r%03d:b-%s-r%03d:s%02d" % (
                        direction.value,
                        side_a.value,
                        run_index_a,
                        side_b.value,
                        run_index_b,
                        scale_index,
                    )
                    candidates.append(
                        ArcPairCandidate(
                            candidate_id=candidate_id,
                            direction=direction,
                            sequence_a=role_cache_a[key_a],
                            sequence_b=role_cache_b[key_b],
                        )
                    )
                    direction_candidate_count += 1
        if direction_candidate_count:
            emitted_directions.append(direction)
            indices = tuple(range(direction_start, len(candidates)))
            direction_groups.append(
                DirectionCandidateGroup(
                    direction=direction,
                    candidate_indices=indices,
                    candidate_ids=tuple(
                        candidates[index].candidate_id for index in indices
                    ),
                )
            )

    if not candidates:
        warnings = (
            ("unavailable_facing_directions:" + ",".join(unavailable_directions),)
            if unavailable_directions
            else ()
        )
        return _failure(
            GeometryStatus.NO_USABLE_CANDIDATES,
            "no_complementary_facing_side_candidates",
            replace(quality, warnings=warnings),
        )
    warnings = ()
    if unavailable_directions:
        warnings = (
            "unavailable_facing_directions:" + ",".join(unavailable_directions),
        )
    quality = replace(
        quality,
        emitted_directions=tuple(direction.value for direction in emitted_directions),
        candidate_count=len(candidates),
        sequence_count_a=len(role_cache_a),
        sequence_count_b=len(role_cache_b),
        direction_candidate_counts=tuple(
            (group.short_name, group.count) for group in direction_groups
        ),
        warnings=warnings,
    )
    return PairCandidateResult(
        status=GeometryStatus.OK,
        failure_reason=None,
        candidates=tuple(candidates),
        direction_groups=tuple(direction_groups),
        quality=quality,
    )


def combine_fragment_results(
    fragment_a: FragmentGeometryResult,
    fragment_b: FragmentGeometryResult,
    direction_b_wrt_a: Optional[Union[PairDirection, str]] = None,
    config: Optional[CandidateBuilderConfig] = None,
) -> PairCandidateResult:
    """Fail closed before pair combination if either fragment is invalid."""

    settings = config or CandidateBuilderConfig()
    requested, direction_reason = _parse_directions(direction_b_wrt_a)
    if requested is None:
        return _failure(
            GeometryStatus.INVALID_INPUT,
            direction_reason or "invalid_direction",
            _base_quality(()),
        )
    if not isinstance(fragment_a, FragmentGeometryResult) or not isinstance(
        fragment_b, FragmentGeometryResult
    ):
        raise TypeError("fragment_a and fragment_b must be fragment results")
    if not fragment_a.ok:
        return _failure(
            fragment_a.status,
            "a:%s" % fragment_a.failure_reason,
            _base_quality(requested),
        )
    if fragment_a.artifact is None:  # pragma: no cover - dataclass invariant
        raise RuntimeError("successful fragment A result lost its artifact")
    quality_a = _fragment_quality_for_role(fragment_a.artifact, "a")
    if not fragment_b.ok:
        return _failure(
            fragment_b.status,
            "b:%s" % fragment_b.failure_reason,
            _base_quality(requested, quality_a),
        )
    if fragment_b.artifact is None:  # pragma: no cover - dataclass invariant
        raise RuntimeError("successful fragment B result lost its artifact")
    return combine_fragment_artifacts(
        fragment_a.artifact,
        fragment_b.artifact,
        direction_b_wrt_a=direction_b_wrt_a,
        config=settings,
    )


def build_pair_candidates(
    mask_a: MaskInput,
    mask_b: MaskInput,
    direction_b_wrt_a: Optional[Union[PairDirection, str]] = None,
    config: Optional[CandidateBuilderConfig] = None,
) -> PairCandidateResult:
    """Build complementary contour-window candidates for two upright masks.

    Args:
        mask_a: Two-dimensional foreground mask for fragment A.
        mask_b: Two-dimensional foreground mask for fragment B.
        direction_b_wrt_a: Optional supervised relative direction.  ``None``
            is the inference default and emits all four facing-edge families.
        config: Frozen geometry configuration.

    Returns:
        A fail-closed result.  Failures contain no partial candidates.  On
        success, every candidate exposes variable-length ``patches_a`` and
        ``patches_b`` arrays in ``[L, 3, H, W]`` order plus provenance.
    """

    settings = config or CandidateBuilderConfig()
    requested, direction_reason = _parse_directions(direction_b_wrt_a)
    if requested is None:
        return _failure(
            GeometryStatus.INVALID_INPUT,
            direction_reason or "invalid_direction",
            _base_quality(()),
        )
    quality = _base_quality(requested)
    if (
        settings.corrosion.enabled
        or settings.corrosion.erosion_radius_fraction > 0.0
        or settings.corrosion.missing_arc_fraction > 0.0
    ):
        return _failure(
            GeometryStatus.UNSUPPORTED_CONFIGURATION,
            "controlled_corrosion_is_reserved_but_disabled_in_v0_2_clean",
            quality,
        )

    fragment_a, status_a, reason_a = _fragment_geometry(mask_a, "a", settings)
    if fragment_a is None:
        return _failure(
            status_a, reason_a or "fragment_a_preprocessing_failed", quality
        )
    quality = _base_quality(requested, fragment_a.quality)
    fragment_b, status_b, reason_b = _fragment_geometry(mask_b, "b", settings)
    if fragment_b is None:
        return _failure(
            status_b, reason_b or "fragment_b_preprocessing_failed", quality
        )
    quality = _base_quality(requested, fragment_a.quality, fragment_b.quality)

    cache_a: Dict[Tuple[CardinalSide, int, int], PatchSequence] = {}
    cache_b: Dict[Tuple[CardinalSide, int, int], PatchSequence] = {}
    candidates: List[ArcPairCandidate] = []
    direction_groups: List[DirectionCandidateGroup] = []
    emitted_directions: List[PairDirection] = []
    unavailable_directions: List[str] = []
    for direction in requested:
        direction_start = len(candidates)
        side_a, side_b = direction.facing_sides
        runs_a = fragment_a.runs[side_a]
        runs_b = fragment_b.runs[side_b]
        if not runs_a or not runs_b:
            unavailable_directions.append(direction.value)
            continue
        direction_candidate_count = 0
        for scale_index, scale_fraction in enumerate(settings.window_scale_fractions):
            for run_a in runs_a:
                key_a = (side_a, run_a.run_index, scale_index)
                if key_a not in cache_a:
                    sequence_a, reason = _build_sequence(
                        fragment_a,
                        run_a,
                        scale_index,
                        scale_fraction,
                        settings,
                    )
                    if sequence_a is None:
                        return _failure(
                            GeometryStatus.NO_USABLE_CANDIDATES,
                            "a:%s:%s" % (direction.value, reason or "sequence_failed"),
                            quality,
                        )
                    cache_a[key_a] = sequence_a
                for run_b in runs_b:
                    key_b = (side_b, run_b.run_index, scale_index)
                    if key_b not in cache_b:
                        sequence_b, reason = _build_sequence(
                            fragment_b,
                            run_b,
                            scale_index,
                            scale_fraction,
                            settings,
                        )
                        if sequence_b is None:
                            return _failure(
                                GeometryStatus.NO_USABLE_CANDIDATES,
                                "b:%s:%s"
                                % (direction.value, reason or "sequence_failed"),
                                quality,
                            )
                        cache_b[key_b] = sequence_b
                    candidate_id = "%s:a-%s-r%03d:b-%s-r%03d:s%02d" % (
                        direction.value,
                        side_a.value,
                        run_a.run_index,
                        side_b.value,
                        run_b.run_index,
                        scale_index,
                    )
                    candidates.append(
                        ArcPairCandidate(
                            candidate_id=candidate_id,
                            direction=direction,
                            sequence_a=cache_a[key_a],
                            sequence_b=cache_b[key_b],
                        )
                    )
                    direction_candidate_count += 1
        if direction_candidate_count:
            emitted_directions.append(direction)
            direction_indices = tuple(range(direction_start, len(candidates)))
            direction_groups.append(
                DirectionCandidateGroup(
                    direction=direction,
                    candidate_indices=direction_indices,
                    candidate_ids=tuple(
                        candidates[index].candidate_id for index in direction_indices
                    ),
                )
            )

    if not candidates:
        warnings = (
            ("unavailable_facing_directions:" + ",".join(unavailable_directions),)
            if unavailable_directions
            else ()
        )
        return _failure(
            GeometryStatus.NO_USABLE_CANDIDATES,
            "no_complementary_facing_side_candidates",
            replace(quality, warnings=warnings),
        )
    warnings = ()
    if unavailable_directions:
        warnings = (
            "unavailable_facing_directions:" + ",".join(unavailable_directions),
        )
    quality = replace(
        quality,
        emitted_directions=tuple(direction.value for direction in emitted_directions),
        candidate_count=len(candidates),
        sequence_count_a=len(cache_a),
        sequence_count_b=len(cache_b),
        direction_candidate_counts=tuple(
            (group.short_name, group.count) for group in direction_groups
        ),
        warnings=warnings,
    )
    return PairCandidateResult(
        status=GeometryStatus.OK,
        failure_reason=None,
        candidates=tuple(candidates),
        direction_groups=tuple(direction_groups),
        quality=quality,
    )


__all__ = [
    "build_fragment_geometry",
    "build_pair_candidates",
    "combine_fragment_artifacts",
    "combine_fragment_results",
    "geometry_config_fingerprint",
]
