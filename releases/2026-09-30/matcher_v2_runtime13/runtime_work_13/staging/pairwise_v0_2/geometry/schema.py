"""Typed, fail-closed contracts for Pairwise v0.2 geometry preprocessing.

The geometry stage is deliberately model-free.  It turns two upright binary
masks into one or more complementary facing-edge candidate pairs.  A caller
may provide the *relative* direction for supervised synthetic data.  When it
does not, all four relative directions are emitted deterministically; known
text orientation never implies that the left/right/above/below relation is
known at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from staging.pairwise_v0_1.baselines.b1_contours import (
    BoundarySaddlePolicy,
    CardinalSide,
)


GEOMETRY_VERSION = "upright-facing-multirun-patches/v0.3"
FRAGMENT_GEOMETRY_ARTIFACT_VERSION = "upright-role-neutral-fragment/v0.3"
CHANNEL_ORDER = ("mask", "signed_distance", "boundary_gradient")
WINDOW_NORMALIZATION = "per_fragment_bbox_min_dimension"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PairDirection(str, Enum):
    """Position of fragment B with respect to fragment A.

    These values constrain *facing edges* only.  They never rotate either
    input mask.
    """

    B_LEFT_OF_A = "b_left_of_a"
    B_RIGHT_OF_A = "b_right_of_a"
    B_ABOVE_A = "b_above_a"
    B_BELOW_A = "b_below_a"

    @property
    def facing_sides(self) -> Tuple[CardinalSide, CardinalSide]:
        if self is PairDirection.B_LEFT_OF_A:
            return CardinalSide.LEFT, CardinalSide.RIGHT
        if self is PairDirection.B_RIGHT_OF_A:
            return CardinalSide.RIGHT, CardinalSide.LEFT
        if self is PairDirection.B_ABOVE_A:
            return CardinalSide.TOP, CardinalSide.BOTTOM
        return CardinalSide.BOTTOM, CardinalSide.TOP

    @property
    def inverse(self) -> "PairDirection":
        if self is PairDirection.B_LEFT_OF_A:
            return PairDirection.B_RIGHT_OF_A
        if self is PairDirection.B_RIGHT_OF_A:
            return PairDirection.B_LEFT_OF_A
        if self is PairDirection.B_ABOVE_A:
            return PairDirection.B_BELOW_A
        return PairDirection.B_ABOVE_A

    @property
    def short_name(self) -> str:
        """Historical/data-model label with the same B-w.r.t.-A semantics."""

        return {
            PairDirection.B_LEFT_OF_A: "left",
            PairDirection.B_RIGHT_OF_A: "right",
            PairDirection.B_ABOVE_A: "above",
            PairDirection.B_BELOW_A: "below",
        }[self]


DEFAULT_DIRECTION_ORDER = (
    PairDirection.B_LEFT_OF_A,
    PairDirection.B_RIGHT_OF_A,
    PairDirection.B_ABOVE_A,
    PairDirection.B_BELOW_A,
)


class GeometryStatus(str, Enum):
    OK = "ok"
    INVALID_INPUT = "invalid_input"
    INVALID_CONTOUR = "invalid_contour"
    NO_USABLE_CANDIDATES = "no_usable_candidates"
    UNSUPPORTED_CONFIGURATION = "unsupported_configuration"


@dataclass(frozen=True)
class CorrosionConfig:
    """Reserved controlled-corrosion interface; disabled in clean v0.2.

    The fields are part of the frozen preprocessing receipt so a later
    corrosion ablation cannot silently change the clean baseline.  Setting
    ``enabled=True`` causes preprocessing to fail closed in this version.
    """

    enabled: bool = False
    erosion_radius_fraction: float = 0.0
    missing_arc_fraction: float = 0.0

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.erosion_radius_fraction)
            or self.erosion_radius_fraction < 0
        ):
            raise ValueError("erosion_radius_fraction must be finite and non-negative")
        if not np.isfinite(self.missing_arc_fraction) or not (
            0.0 <= self.missing_arc_fraction < 1.0
        ):
            raise ValueError("missing_arc_fraction must be finite and in [0, 1)")


@dataclass(frozen=True)
class CandidateBuilderConfig:
    """Validation-freezable geometry parameters.

    ``window_scale_fractions`` are resolved separately for each fragment using
    its largest-component bounding-box short side.  This makes the same config
    meaningful for 256, 600, 800, and larger masks.  Windows are resampled to
    the fixed ``output_size`` only after their image-space size is recorded.
    """

    foreground_polarity: str = "bright"
    connectivity: int = 4
    saddle_policy: str = BoundarySaddlePolicy.FOREGROUND_4_BACKGROUND_8.value
    min_component_pixels: int = 16
    min_contour_points: int = 12
    min_run_length_fraction: float = 0.02
    min_run_length_px: float = 4.0
    window_scale_fractions: Tuple[float, ...] = (0.08, 0.16)
    window_min_px: float = 4.0
    window_max_px: float = 192.0
    overlap_fraction: float = 0.5
    output_size: Tuple[int, int] = (32, 32)
    tangent_span_fraction: float = 0.5
    signed_distance_clip_fraction: float = 0.5
    inward_probe_fraction: float = 0.125
    min_image_valid_fraction: float = 0.2
    side_resample_count: int = 32
    corrosion: CorrosionConfig = field(default_factory=CorrosionConfig)

    def __post_init__(self) -> None:
        if self.foreground_polarity not in {"bright", "dark"}:
            raise ValueError("foreground_polarity must be 'bright' or 'dark'")
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        try:
            saddle_policy = BoundarySaddlePolicy(self.saddle_policy)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported boundary saddle policy") from exc
        if (
            saddle_policy is BoundarySaddlePolicy.FOREGROUND_4_BACKGROUND_8
            and self.connectivity != 4
        ):
            raise ValueError(
                "foreground_4_background_8 saddle policy requires connectivity=4"
            )
        if isinstance(self.min_component_pixels, bool) or not isinstance(
            self.min_component_pixels, int
        ):
            raise TypeError("min_component_pixels must be an integer")
        if self.min_component_pixels < 1:
            raise ValueError("min_component_pixels must be positive")
        if isinstance(self.min_contour_points, bool) or not isinstance(
            self.min_contour_points, int
        ):
            raise TypeError("min_contour_points must be an integer")
        if self.min_contour_points < 4:
            raise ValueError("min_contour_points must be at least four")
        if (
            not np.isfinite(self.min_run_length_fraction)
            or self.min_run_length_fraction < 0
        ):
            raise ValueError("min_run_length_fraction must be finite and non-negative")
        if not np.isfinite(self.min_run_length_px) or self.min_run_length_px <= 0:
            raise ValueError("min_run_length_px must be finite and positive")
        if not self.window_scale_fractions:
            raise ValueError("window_scale_fractions cannot be empty")
        if any(
            not np.isfinite(value) or value <= 0
            for value in self.window_scale_fractions
        ):
            raise ValueError("window_scale_fractions must be finite and positive")
        if (
            tuple(sorted(set(self.window_scale_fractions)))
            != self.window_scale_fractions
        ):
            raise ValueError("window_scale_fractions must be unique and increasing")
        if not np.isfinite(self.window_min_px) or self.window_min_px <= 0:
            raise ValueError("window_min_px must be finite and positive")
        if (
            not np.isfinite(self.window_max_px)
            or self.window_max_px < self.window_min_px
        ):
            raise ValueError("window_max_px must be finite and >= window_min_px")
        if not np.isfinite(self.overlap_fraction) or not (
            0.0 <= self.overlap_fraction < 1.0
        ):
            raise ValueError("overlap_fraction must be finite and in [0, 1)")
        if len(self.output_size) != 2 or min(self.output_size) < 2:
            raise ValueError("output_size must contain two integers >= 2")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.output_size
        ):
            raise TypeError("output_size values must be integers")
        if (
            not np.isfinite(self.tangent_span_fraction)
            or self.tangent_span_fraction <= 0
        ):
            raise ValueError("tangent_span_fraction must be finite and positive")
        if (
            not np.isfinite(self.signed_distance_clip_fraction)
            or self.signed_distance_clip_fraction <= 0
        ):
            raise ValueError(
                "signed_distance_clip_fraction must be finite and positive"
            )
        if (
            not np.isfinite(self.inward_probe_fraction)
            or self.inward_probe_fraction <= 0
        ):
            raise ValueError("inward_probe_fraction must be finite and positive")
        if not np.isfinite(self.min_image_valid_fraction) or not (
            0.0 <= self.min_image_valid_fraction <= 1.0
        ):
            raise ValueError("min_image_valid_fraction must be in [0, 1]")
        if isinstance(self.side_resample_count, bool) or not isinstance(
            self.side_resample_count, int
        ):
            raise TypeError("side_resample_count must be an integer")
        if self.side_resample_count < 2:
            raise ValueError("side_resample_count must be an integer >= 2")


@dataclass(frozen=True)
class PatchProvenance:
    sequence_index: int
    center_row_col: Tuple[float, float]
    tangent_row_col: Tuple[float, float]
    inward_normal_row_col: Tuple[float, float]
    contour_segment_indices: Tuple[int, int]
    contour_arc_fraction: float
    path_distance_px: float
    valid_fraction: float
    padding_fraction: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "center_row_col": list(self.center_row_col),
            "tangent_row_col": list(self.tangent_row_col),
            "inward_normal_row_col": list(self.inward_normal_row_col),
            "contour_segment_indices": list(self.contour_segment_indices),
            "contour_arc_fraction": self.contour_arc_fraction,
            "path_distance_px": self.path_distance_px,
            "valid_fraction": self.valid_fraction,
            "padding_fraction": self.padding_fraction,
        }


@dataclass(frozen=True)
class PatchSequence:
    """One variable-length, fixed-channel sequence for one contour run/scale."""

    fragment_role: str
    side: CardinalSide
    run_index: int
    scale_index: int
    scale_fraction: float
    bbox_reference_px: float
    requested_window_px: float
    resolved_window_px: float
    stride_px: float
    run_length_px: float
    direction_rule: str
    channels: np.ndarray
    valid: np.ndarray
    patches: Tuple[PatchProvenance, ...]
    window_normalization: str = WINDOW_NORMALIZATION
    channel_order: Tuple[str, ...] = CHANNEL_ORDER

    def __post_init__(self) -> None:
        if self.fragment_role not in {"fragment", "a", "b"}:
            raise ValueError("fragment_role must be 'fragment', 'a', or 'b'")
        if self.channels.ndim != 4 or self.channels.shape[1] != len(CHANNEL_ORDER):
            raise ValueError("channels must have shape [L, 3, H, W]")
        if self.valid.shape != (self.channels.shape[0],):
            raise ValueError("valid must have shape [L]")
        if len(self.patches) != self.channels.shape[0] or not self.patches:
            raise ValueError("patch provenance must match a non-empty sequence")
        if self.channels.dtype != np.float32 or self.valid.dtype != np.bool_:
            raise TypeError("channels must be float32 and valid must be bool")
        if self.channels.flags.writeable or self.valid.flags.writeable:
            raise ValueError("sequence arrays must be read-only")
        if not np.all(np.isfinite(self.channels)) or not np.all(self.valid):
            raise ValueError("emitted sequence must contain only finite valid patches")

    @property
    def length(self) -> int:
        return int(self.channels.shape[0])

    def provenance_dict(self) -> Dict[str, Any]:
        return {
            "fragment_role": self.fragment_role,
            "side": self.side.value,
            "run_index": self.run_index,
            "scale_index": self.scale_index,
            "scale_fraction": self.scale_fraction,
            "bbox_reference_px": self.bbox_reference_px,
            "requested_window_px": self.requested_window_px,
            "resolved_window_px": self.resolved_window_px,
            "stride_px": self.stride_px,
            "run_length_px": self.run_length_px,
            "direction_rule": self.direction_rule,
            "window_normalization": self.window_normalization,
            "channel_order": list(self.channel_order),
            "sequence_length": self.length,
            "patches": [patch.to_dict() for patch in self.patches],
        }


@dataclass(frozen=True)
class FragmentSequenceFailure:
    """A deterministic per-run/scale failure in a role-neutral artifact."""

    side: CardinalSide
    run_index: int
    scale_index: int
    reason: str

    def __post_init__(self) -> None:
        if isinstance(self.run_index, bool) or not isinstance(self.run_index, int):
            raise TypeError("run_index must be an integer")
        if self.run_index < 0:
            raise ValueError("run_index must be non-negative")
        if isinstance(self.scale_index, bool) or not isinstance(
            self.scale_index, int
        ):
            raise TypeError("scale_index must be an integer")
        if self.scale_index < 0:
            raise ValueError("scale_index must be non-negative")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("sequence failure reason is required")

    @property
    def key(self) -> Tuple[CardinalSide, int, int]:
        return self.side, self.run_index, self.scale_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side.value,
            "run_index": self.run_index,
            "scale_index": self.scale_index,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FragmentGeometryArtifact:
    """Reusable geometry for one fragment, independent of pair role.

    Every retained run at every configured scale is represented by exactly
    one successful sequence or one explicit failure.  No pair ID, label,
    split, filesystem path, or A/B role is permitted in this object.
    """

    config_fingerprint: str
    window_scale_fractions: Tuple[float, ...]
    run_indices_by_side: Tuple[Tuple[str, Tuple[int, ...]], ...]
    sequences: Tuple[PatchSequence, ...]
    sequence_failures: Tuple[FragmentSequenceFailure, ...]
    quality: "FragmentGeometryQuality"
    artifact_version: str = FRAGMENT_GEOMETRY_ARTIFACT_VERSION
    geometry_version: str = GEOMETRY_VERSION

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(self.config_fingerprint):
            raise ValueError("config_fingerprint must be lowercase SHA-256")
        if self.artifact_version != FRAGMENT_GEOMETRY_ARTIFACT_VERSION:
            raise ValueError("fragment artifact version mismatch")
        if self.geometry_version != GEOMETRY_VERSION:
            raise ValueError("fragment geometry version mismatch")
        if not self.window_scale_fractions or any(
            not np.isfinite(value) or value <= 0.0
            for value in self.window_scale_fractions
        ):
            raise ValueError("window_scale_fractions must be finite and positive")
        if self.quality.role != "fragment":
            raise ValueError("fragment artifact quality must be role-neutral")

        expected_side_names = tuple(side.value for side in CardinalSide)
        observed_side_names = tuple(name for name, _ in self.run_indices_by_side)
        if observed_side_names != expected_side_names:
            raise ValueError("run_indices_by_side must use canonical side order")
        run_map = {}
        for name, indices in self.run_indices_by_side:
            if any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            ):
                raise ValueError("run indices must be non-negative integers")
            if tuple(sorted(set(indices))) != indices:
                raise ValueError("run indices must be unique and increasing")
            run_map[CardinalSide(name)] = indices

        retained_counts = dict(self.quality.retained_runs_by_side)
        if set(retained_counts) != set(expected_side_names):
            raise ValueError("quality retained-run sides are incomplete")
        if any(
            retained_counts[side.value] != len(run_map[side])
            for side in CardinalSide
        ):
            raise ValueError("quality retained-run counts disagree with artifact")

        sequence_keys = []
        canonical_sequence_order = []
        side_order = {side: index for index, side in enumerate(CardinalSide)}
        for sequence in self.sequences:
            if sequence.fragment_role != "fragment":
                raise ValueError("cached fragment sequences must be role-neutral")
            key = (sequence.side, sequence.run_index, sequence.scale_index)
            sequence_keys.append(key)
            canonical_sequence_order.append(
                (side_order[sequence.side], sequence.scale_index, sequence.run_index)
            )
        if len(set(sequence_keys)) != len(sequence_keys):
            raise ValueError("fragment sequence keys must be unique")
        if canonical_sequence_order != sorted(canonical_sequence_order):
            raise ValueError("fragment sequences are not in canonical order")

        failure_keys = [failure.key for failure in self.sequence_failures]
        canonical_failure_order = [
            (side_order[side], scale, run) for side, run, scale in failure_keys
        ]
        if len(set(failure_keys)) != len(failure_keys):
            raise ValueError("fragment sequence failure keys must be unique")
        if canonical_failure_order != sorted(canonical_failure_order):
            raise ValueError("fragment sequence failures are not in canonical order")
        if set(sequence_keys) & set(failure_keys):
            raise ValueError("a fragment sequence cannot both succeed and fail")

        observed_keys = set(sequence_keys) | set(failure_keys)
        expected_keys = {
            (side, run_index, scale_index)
            for side in CardinalSide
            for scale_index in range(len(self.window_scale_fractions))
            for run_index in run_map[side]
        }
        if observed_keys != expected_keys:
            raise ValueError("fragment artifact does not cover every run and scale")
        for sequence in self.sequences:
            if sequence.run_index not in run_map[sequence.side]:
                raise ValueError("fragment sequence refers to an unknown run")
            if sequence.scale_index >= len(self.window_scale_fractions):
                raise ValueError("fragment sequence scale index is out of range")
            if sequence.scale_fraction != self.window_scale_fractions[
                sequence.scale_index
            ]:
                raise ValueError("fragment sequence scale fraction mismatch")

    @property
    def sequence_map(self) -> Dict[Tuple[CardinalSide, int, int], PatchSequence]:
        return {
            (sequence.side, sequence.run_index, sequence.scale_index): sequence
            for sequence in self.sequences
        }

    @property
    def failure_map(
        self,
    ) -> Dict[Tuple[CardinalSide, int, int], FragmentSequenceFailure]:
        return {failure.key: failure for failure in self.sequence_failures}

    @property
    def run_map(self) -> Dict[CardinalSide, Tuple[int, ...]]:
        return {
            CardinalSide(name): indices for name, indices in self.run_indices_by_side
        }


@dataclass(frozen=True)
class FragmentGeometryResult:
    """Fail-closed result for single-fragment geometry preprocessing."""

    status: GeometryStatus
    failure_reason: Optional[str]
    artifact: Optional[FragmentGeometryArtifact]

    def __post_init__(self) -> None:
        if self.status is GeometryStatus.OK:
            if self.failure_reason is not None or self.artifact is None:
                raise ValueError("successful fragment result requires an artifact")
        elif self.artifact is not None:
            raise ValueError("failed fragment result cannot expose an artifact")
        elif not isinstance(self.failure_reason, str) or not self.failure_reason:
            raise ValueError("failed fragment result requires a reason")

    @property
    def ok(self) -> bool:
        return self.status is GeometryStatus.OK


@dataclass(frozen=True)
class ArcPairCandidate:
    candidate_id: str
    direction: PairDirection
    sequence_a: PatchSequence
    sequence_b: PatchSequence

    def __post_init__(self) -> None:
        side_a, side_b = self.direction.facing_sides
        if self.sequence_a.side is not side_a or self.sequence_b.side is not side_b:
            raise ValueError("candidate sequences violate complementary facing sides")
        if self.sequence_a.scale_index != self.sequence_b.scale_index:
            raise ValueError(
                "candidate sequences must use the same normalized scale index"
            )

    @property
    def patches_a(self) -> np.ndarray:
        return self.sequence_a.channels

    @property
    def patches_b(self) -> np.ndarray:
        return self.sequence_b.channels

    @property
    def valid_a(self) -> np.ndarray:
        return self.sequence_a.valid

    @property
    def valid_b(self) -> np.ndarray:
        return self.sequence_b.valid


@dataclass(frozen=True)
class DirectionCandidateGroup:
    """Deterministic flat-index group for hierarchical MIL aggregation.

    Multiple run pairs and scales may exist for one relative direction.  The
    local matcher should first aggregate the indices in each group, then
    aggregate the resulting direction scores across the four groups.
    """

    direction: PairDirection
    candidate_indices: Tuple[int, ...]
    candidate_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_indices:
            raise ValueError("direction candidate group cannot be empty")
        if len(self.candidate_indices) != len(self.candidate_ids):
            raise ValueError("candidate indices and ids must have equal length")
        if any(index < 0 for index in self.candidate_indices):
            raise ValueError("candidate indices must be non-negative")
        if tuple(sorted(self.candidate_indices)) != self.candidate_indices:
            raise ValueError("candidate indices must be increasing")
        if len(set(self.candidate_indices)) != len(self.candidate_indices):
            raise ValueError("candidate indices must be unique")

    @property
    def short_name(self) -> str:
        return self.direction.short_name

    @property
    def slot_index(self) -> int:
        """Stable index in the four-direction inference order."""

        return DEFAULT_DIRECTION_ORDER.index(self.direction)

    @property
    def count(self) -> int:
        return len(self.candidate_indices)

    @property
    def start(self) -> int:
        return self.candidate_indices[0]

    @property
    def stop(self) -> int:
        return self.candidate_indices[-1] + 1


@dataclass(frozen=True)
class FragmentGeometryQuality:
    role: str
    input_shape: Tuple[int, ...]
    input_dtype: str
    foreground_pixels: int
    component_count: int
    largest_component_pixels: int
    discarded_foreground_pixels: int
    bbox_rc_exclusive: Tuple[int, int, int, int]
    bbox_reference_px: float
    contour_points: int
    contour_perimeter_px: float
    retained_runs_by_side: Tuple[Tuple[str, int], ...]
    dropped_short_runs_by_side: Tuple[Tuple[str, int], ...]
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "foreground_pixels": self.foreground_pixels,
            "component_count": self.component_count,
            "largest_component_pixels": self.largest_component_pixels,
            "discarded_foreground_pixels": self.discarded_foreground_pixels,
            "bbox_rc_exclusive": list(self.bbox_rc_exclusive),
            "bbox_reference_px": self.bbox_reference_px,
            "contour_points": self.contour_points,
            "contour_perimeter_px": self.contour_perimeter_px,
            "retained_runs_by_side": dict(self.retained_runs_by_side),
            "dropped_short_runs_by_side": dict(self.dropped_short_runs_by_side),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PairGeometryQuality:
    requested_directions: Tuple[str, ...]
    emitted_directions: Tuple[str, ...]
    candidate_count: int
    sequence_count_a: int
    sequence_count_b: int
    fragment_a: Optional[FragmentGeometryQuality]
    fragment_b: Optional[FragmentGeometryQuality]
    direction_candidate_counts: Tuple[Tuple[str, int], ...] = ()
    rotation_search_performed: bool = False
    upright_orientation_assumed: bool = True
    window_normalization: str = WINDOW_NORMALIZATION
    channel_order: Tuple[str, ...] = CHANNEL_ORDER
    geometry_version: str = GEOMETRY_VERSION
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_directions": list(self.requested_directions),
            "emitted_directions": list(self.emitted_directions),
            "candidate_count": self.candidate_count,
            "sequence_count_a": self.sequence_count_a,
            "sequence_count_b": self.sequence_count_b,
            "fragment_a": self.fragment_a.to_dict() if self.fragment_a else None,
            "fragment_b": self.fragment_b.to_dict() if self.fragment_b else None,
            "direction_candidate_counts": dict(self.direction_candidate_counts),
            "rotation_search_performed": self.rotation_search_performed,
            "upright_orientation_assumed": self.upright_orientation_assumed,
            "window_normalization": self.window_normalization,
            "channel_order": list(self.channel_order),
            "geometry_version": self.geometry_version,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PairCandidateResult:
    status: GeometryStatus
    failure_reason: Optional[str]
    candidates: Tuple[ArcPairCandidate, ...]
    direction_groups: Tuple[DirectionCandidateGroup, ...]
    quality: PairGeometryQuality

    def __post_init__(self) -> None:
        if self.status is GeometryStatus.OK:
            if self.failure_reason is not None or not self.candidates:
                raise ValueError("successful result requires candidates and no failure")
            if self.quality.candidate_count != len(self.candidates):
                raise ValueError("quality candidate_count does not match candidates")
            covered = tuple(
                index
                for group in self.direction_groups
                for index in group.candidate_indices
            )
            if covered != tuple(range(len(self.candidates))):
                raise ValueError(
                    "direction groups must cover candidates exactly in order"
                )
            for group in self.direction_groups:
                ids = tuple(
                    self.candidates[index].candidate_id
                    for index in group.candidate_indices
                )
                if ids != group.candidate_ids:
                    raise ValueError(
                        "direction group candidate ids do not match candidates"
                    )
                if any(
                    self.candidates[index].direction is not group.direction
                    for index in group.candidate_indices
                ):
                    raise ValueError("direction group contains a mismatched candidate")
        elif self.candidates:
            raise ValueError("failed result cannot expose partial candidates")
        elif self.direction_groups:
            raise ValueError("failed result cannot expose partial direction groups")
        elif self.failure_reason is None:
            raise ValueError("failed result requires a reason")

    @property
    def ok(self) -> bool:
        return self.status is GeometryStatus.OK

    def candidates_for_direction(
        self,
        direction: Union[PairDirection, str],
    ) -> Tuple[ArcPairCandidate, ...]:
        """Return one direction's candidates without changing flat order."""

        parsed = PairDirection(direction)
        for group in self.direction_groups:
            if group.direction is parsed:
                return tuple(
                    self.candidates[index] for index in group.candidate_indices
                )
        return ()


__all__ = [
    "ArcPairCandidate",
    "CandidateBuilderConfig",
    "CHANNEL_ORDER",
    "CorrosionConfig",
    "DEFAULT_DIRECTION_ORDER",
    "DirectionCandidateGroup",
    "FRAGMENT_GEOMETRY_ARTIFACT_VERSION",
    "FragmentGeometryArtifact",
    "FragmentGeometryQuality",
    "FragmentGeometryResult",
    "FragmentSequenceFailure",
    "GEOMETRY_VERSION",
    "GeometryStatus",
    "PairCandidateResult",
    "PairDirection",
    "PairGeometryQuality",
    "PatchProvenance",
    "PatchSequence",
    "WINDOW_NORMALIZATION",
]
