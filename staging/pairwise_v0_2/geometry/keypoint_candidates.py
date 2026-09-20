"""Label-blind contour-keypoint candidates for an experimental local arm.

The production v0.2 geometry represents a direction by a cross product of
cardinal contour runs.  This module is an additive research alternative: it
selects a bounded set of anchors on each fragment *before the fragments are
paired*, reuses the already-cached multi-scale patch tensors nearest to those
anchors, and emits all four upright facing-side hypotheses.

Selection combines three complementary signals:

* approximately uniform circular arc-length coverage;
* the window nearest each retained run endpoint; and
* curvature measured at several arc-length radii and aggregated by a median.

The multi-radius median is deliberately insensitive to tangent sign and less
sensitive to one-pixel boundary chatter than a single discrete second
difference.  Pair labels and the ground-truth relative direction are not
accepted by any public function in this module.  The only direction use is to
enumerate all four known-upright facing-side hypotheses after both fragment
keypoint sets have been frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from staging.pairwise_v0_1.baselines.b1_contours import CardinalSide
from staging.pairwise_v0_2.geometry.schema import (
    CHANNEL_ORDER,
    DEFAULT_DIRECTION_ORDER,
    FragmentGeometryArtifact,
    PairDirection,
    PatchSequence,
)


KEYPOINT_GEOMETRY_VERSION = "upright-contour-keypoint-patches/v0.1-prototype"


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    source = np.asarray(value, dtype=dtype)
    output = source if not source.flags.writeable else source.copy()
    output.setflags(write=False)
    return output


def _circular_distance(first: float, second: float) -> float:
    delta = abs(float(first) - float(second)) % 1.0
    return min(delta, 1.0 - delta)


@dataclass(frozen=True)
class ContourKeypointConfig:
    """Bounded, validation-freezable keypoint-selection settings.

    ``max_keypoints_per_side`` is an explicit compute cap.  Each selected
    anchor contributes one token per cached patch scale, so a direction has at
    most ``max_keypoints_per_side * scale_count`` tokens per fragment.
    """

    max_keypoints_per_side: int = 24
    uniform_keypoints_per_side: int = 8
    curvature_keypoints_per_side: int = 8
    curvature_radius_fractions: Tuple[float, ...] = (0.015, 0.03, 0.06)
    curvature_nms_arc_fraction: float = 0.015
    include_run_endpoint_windows: bool = True
    require_all_scales: bool = True
    same_scale_correspondence_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_keypoints_per_side",
            "uniform_keypoints_per_side",
            "curvature_keypoints_per_side",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("{} must be an integer".format(name))
        if self.max_keypoints_per_side < 1:
            raise ValueError("max_keypoints_per_side must be positive")
        if not 1 <= self.uniform_keypoints_per_side <= self.max_keypoints_per_side:
            raise ValueError(
                "uniform_keypoints_per_side must be in [1, max_keypoints_per_side]"
            )
        if not 1 <= self.curvature_keypoints_per_side <= self.max_keypoints_per_side:
            raise ValueError(
                "curvature_keypoints_per_side must be in [1, max_keypoints_per_side]"
            )
        if (
            self.uniform_keypoints_per_side + self.curvature_keypoints_per_side
            > self.max_keypoints_per_side
        ):
            raise ValueError(
                "uniform and curvature keypoint budgets exceed the side cap"
            )
        if not self.curvature_radius_fractions:
            raise ValueError("curvature_radius_fractions cannot be empty")
        if any(
            not math.isfinite(value) or not 0.0 < value < 0.5
            for value in self.curvature_radius_fractions
        ):
            raise ValueError(
                "curvature_radius_fractions must be finite values in (0, 0.5)"
            )
        if (
            tuple(sorted(set(self.curvature_radius_fractions)))
            != self.curvature_radius_fractions
        ):
            raise ValueError("curvature_radius_fractions must be unique and increasing")
        if (
            not math.isfinite(self.curvature_nms_arc_fraction)
            or not 0.0 <= self.curvature_nms_arc_fraction < 0.5
        ):
            raise ValueError("curvature_nms_arc_fraction must be in [0, 0.5)")
        for name in (
            "include_run_endpoint_windows",
            "require_all_scales",
            "same_scale_correspondence_only",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("{} must be bool".format(name))


@dataclass(frozen=True)
class KeypointScaleToken:
    """One existing cached patch attached to a selected contour anchor."""

    token_id: str
    anchor_id: str
    side: CardinalSide
    run_index: int
    scale_index: int
    scale_fraction: float
    source_patch_index: int
    source_arc_fraction: float
    source_path_fraction: float
    center_row_col: Tuple[float, float]
    channels: np.ndarray

    def __post_init__(self) -> None:
        if self.channels.ndim != 3 or self.channels.shape[0] != len(CHANNEL_ORDER):
            raise ValueError("keypoint token channels must have shape [3, H, W]")
        if self.channels.dtype != np.float32 or self.channels.flags.writeable:
            raise ValueError("keypoint token channels must be read-only float32")
        if not np.isfinite(self.channels).all():
            raise ValueError("keypoint token channels must be finite")


@dataclass(frozen=True)
class ContourKeypoint:
    """A fragment-only anchor with one token at every retained scale."""

    anchor_id: str
    side: CardinalSide
    run_index: int
    anchor_arc_fraction: float
    anchor_path_fraction: float
    center_row_col: Tuple[float, float]
    curvature_score_radians: float
    selection_reasons: Tuple[str, ...]
    tokens: Tuple[KeypointScaleToken, ...]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("a contour keypoint must expose at least one scale")
        if any(token.anchor_id != self.anchor_id for token in self.tokens):
            raise ValueError("keypoint tokens refer to another anchor")
        if any(
            token.side is not self.side or token.run_index != self.run_index
            for token in self.tokens
        ):
            raise ValueError("keypoint token side/run differs from its anchor")
        scales = tuple(token.scale_index for token in self.tokens)
        if scales != tuple(sorted(set(scales))):
            raise ValueError("keypoint token scales must be unique and increasing")
        if not self.selection_reasons:
            raise ValueError("keypoint selection provenance cannot be empty")


@dataclass(frozen=True)
class FragmentKeypointSet:
    """Bounded keypoints selected without any pair-level information."""

    fragment_config_fingerprint: str
    scale_count: int
    keypoints: Tuple[ContourKeypoint, ...]
    keypoint_counts_by_side: Tuple[Tuple[str, int], ...]
    selector_version: str = KEYPOINT_GEOMETRY_VERSION

    def __post_init__(self) -> None:
        expected = tuple(side.value for side in CardinalSide)
        if tuple(name for name, _ in self.keypoint_counts_by_side) != expected:
            raise ValueError("keypoint counts must use canonical side order")
        observed = {side.value: 0 for side in CardinalSide}
        for keypoint in self.keypoints:
            observed[keypoint.side.value] += 1
        if tuple((name, observed[name]) for name in expected) != (
            self.keypoint_counts_by_side
        ):
            raise ValueError("keypoint side counts disagree with keypoints")
        identifiers = tuple(keypoint.anchor_id for keypoint in self.keypoints)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("keypoint anchor ids must be unique")

    @property
    def by_side(self) -> Mapping[CardinalSide, Tuple[ContourKeypoint, ...]]:
        return {
            side: tuple(
                keypoint for keypoint in self.keypoints if keypoint.side is side
            )
            for side in CardinalSide
        }


@dataclass(frozen=True)
class KeypointPairCandidate:
    """One facing-side hypothesis with a sparse keypoint correspondence graph."""

    candidate_id: str
    direction: PairDirection
    tokens_a: Tuple[KeypointScaleToken, ...]
    tokens_b: Tuple[KeypointScaleToken, ...]
    patches_a: np.ndarray
    patches_b: np.ndarray
    correspondence_mask: np.ndarray

    def __post_init__(self) -> None:
        if not self.tokens_a or not self.tokens_b:
            raise ValueError("keypoint candidate requires tokens on both fragments")
        if self.patches_a.ndim != 4 or self.patches_b.ndim != 4:
            raise ValueError("keypoint patches must have shape [L, C, H, W]")
        if self.patches_a.shape[0] != len(self.tokens_a) or self.patches_b.shape[
            0
        ] != len(self.tokens_b):
            raise ValueError("keypoint patch count differs from token provenance")
        if self.patches_a.shape[1:] != self.patches_b.shape[1:]:
            raise ValueError("A/B keypoint patch shapes differ")
        if self.correspondence_mask.shape != (
            len(self.tokens_a),
            len(self.tokens_b),
        ):
            raise ValueError("correspondence mask shape differs from token counts")
        if self.correspondence_mask.dtype != np.bool_:
            raise TypeError("correspondence mask must be bool")
        if (
            self.patches_a.flags.writeable
            or self.patches_b.flags.writeable
            or self.correspondence_mask.flags.writeable
        ):
            raise ValueError("keypoint candidate arrays must be read-only")
        if not self.correspondence_mask.any():
            raise ValueError("keypoint candidate needs at least one allowed pair")
        side_a, side_b = self.direction.facing_sides
        if any(token.side is not side_a for token in self.tokens_a) or any(
            token.side is not side_b for token in self.tokens_b
        ):
            raise ValueError("keypoint tokens violate the facing-side hypothesis")

    @property
    def allowed_pair_count(self) -> int:
        return int(self.correspondence_mask.sum())


@dataclass(frozen=True)
class KeypointComplexity:
    candidate_count: int
    max_tokens_a: int
    max_tokens_b: int
    affinity_elements: int
    sinkhorn_elements: int
    configured_max_tokens_per_fragment_direction: int
    configured_max_affinity_elements_per_direction: int
    configured_max_sinkhorn_elements_per_pair: int


@dataclass(frozen=True)
class KeypointPairResult:
    """All available upright direction hypotheses; no target direction input."""

    candidates: Tuple[KeypointPairCandidate, ...]
    unavailable_directions: Tuple[str, ...]
    complexity: KeypointComplexity
    selector_version: str = KEYPOINT_GEOMETRY_VERSION


@dataclass(frozen=True)
class _Observation:
    side: CardinalSide
    run_index: int
    patch_index: int
    arc_fraction: float
    path_fraction: float
    center: Tuple[float, float]
    sequence: PatchSequence = field(compare=False)

    @property
    def key(self) -> Tuple[int, int, int]:
        return (list(CardinalSide).index(self.side), self.run_index, self.patch_index)


def _ring_points(
    base_observations: Sequence[_Observation],
) -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicate arc positions for circular interpolation of contour shape."""

    grouped: Dict[float, List[Tuple[float, float]]] = {}
    for observation in base_observations:
        key = round(observation.arc_fraction % 1.0, 12)
        grouped.setdefault(key, []).append(observation.center)
    fractions = np.asarray(sorted(grouped), dtype=np.float64)
    points = np.asarray(
        [
            np.mean(np.asarray(grouped[value], dtype=np.float64), axis=0)
            for value in fractions
        ],
        dtype=np.float64,
    )
    if len(fractions) < 3:
        raise ValueError("at least three distinct contour samples are required")
    return fractions, points


def _circular_point(
    fractions: np.ndarray, points: np.ndarray, target: float
) -> np.ndarray:
    extended_fractions = np.concatenate((fractions - 1.0, fractions, fractions + 1.0))
    extended_points = np.concatenate((points, points, points), axis=0)
    bounded = float(target % 1.0)
    row = np.interp(bounded, extended_fractions, extended_points[:, 0])
    column = np.interp(bounded, extended_fractions, extended_points[:, 1])
    return np.asarray((row, column), dtype=np.float64)


def _curvature_score(
    observation: _Observation,
    fractions: np.ndarray,
    points: np.ndarray,
    radius_fractions: Sequence[float],
) -> float:
    """Median turn angle over several circular arc radii."""

    center = _circular_point(fractions, points, observation.arc_fraction)
    values = []
    for radius in radius_fractions:
        before = _circular_point(fractions, points, observation.arc_fraction - radius)
        after = _circular_point(fractions, points, observation.arc_fraction + radius)
        incoming = center - before
        outgoing = after - center
        norm = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        if not math.isfinite(norm) or norm <= 1e-10:
            values.append(0.0)
            continue
        cosine = float(np.clip(np.dot(incoming, outgoing) / norm, -1.0, 1.0))
        values.append(float(math.acos(cosine)))
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _farthest_arc_selection(
    observations: Sequence[_Observation], count: int
) -> Tuple[_Observation, ...]:
    """Deterministic circular farthest-point sampling for arc-length coverage."""

    if not observations or count <= 0:
        return ()
    ordered = sorted(observations, key=lambda value: (value.arc_fraction, value.key))
    selected = [ordered[0]]
    remaining = list(ordered[1:])
    while remaining and len(selected) < count:
        scored = [
            (
                min(
                    _circular_distance(candidate.arc_fraction, chosen.arc_fraction)
                    for chosen in selected
                ),
                -candidate.arc_fraction,
                tuple(-value for value in candidate.key),
                candidate,
            )
            for candidate in remaining
        ]
        chosen = max(scored, key=lambda item: item[:3])[-1]
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


def _nearest_scale_token(
    anchor: _Observation,
    sequence: PatchSequence,
    anchor_id: str,
) -> KeypointScaleToken:
    distances = [
        abs(
            (patch.path_distance_px / max(sequence.run_length_px, 1e-12))
            - anchor.path_fraction
        )
        for patch in sequence.patches
    ]
    patch_index = min(
        range(len(distances)), key=lambda index: (distances[index], index)
    )
    patch = sequence.patches[patch_index]
    source_path_fraction = patch.path_distance_px / max(sequence.run_length_px, 1e-12)
    token_id = "{}:s{:02d}".format(anchor_id, sequence.scale_index)
    return KeypointScaleToken(
        token_id=token_id,
        anchor_id=anchor_id,
        side=anchor.side,
        run_index=anchor.run_index,
        scale_index=sequence.scale_index,
        scale_fraction=sequence.scale_fraction,
        source_patch_index=patch_index,
        source_arc_fraction=patch.contour_arc_fraction,
        source_path_fraction=float(source_path_fraction),
        center_row_col=patch.center_row_col,
        channels=_readonly(sequence.channels[patch_index], np.float32),
    )


def build_fragment_keypoints(
    artifact: FragmentGeometryArtifact,
    config: Optional[ContourKeypointConfig] = None,
) -> FragmentKeypointSet:
    """Select bounded keypoints from one role-neutral cached artifact.

    This signature intentionally has no pair, label, or direction parameter.
    """

    if not isinstance(artifact, FragmentGeometryArtifact):
        raise TypeError("artifact must be FragmentGeometryArtifact")
    settings = config or ContourKeypointConfig()
    if not isinstance(settings, ContourKeypointConfig):
        raise TypeError("config must be ContourKeypointConfig")
    sequence_map = artifact.sequence_map
    scale_count = len(artifact.window_scale_fractions)
    base_observations: List[_Observation] = []
    for side in CardinalSide:
        for run_index in artifact.run_map[side]:
            required = [
                sequence_map.get((side, run_index, scale_index))
                for scale_index in range(scale_count)
            ]
            if settings.require_all_scales and any(value is None for value in required):
                continue
            base = required[0]
            if base is None:
                continue
            for patch_index, patch in enumerate(base.patches):
                path_fraction = patch.path_distance_px / max(base.run_length_px, 1e-12)
                base_observations.append(
                    _Observation(
                        side=side,
                        run_index=run_index,
                        patch_index=patch_index,
                        arc_fraction=float(patch.contour_arc_fraction % 1.0),
                        path_fraction=float(path_fraction),
                        center=patch.center_row_col,
                        sequence=base,
                    )
                )
    if len(base_observations) < 3:
        raise ValueError("fragment has too few complete keypoint observations")

    ring_fractions, ring_points = _ring_points(base_observations)
    curvature = {
        observation.key: _curvature_score(
            observation,
            ring_fractions,
            ring_points,
            settings.curvature_radius_fractions,
        )
        for observation in base_observations
    }
    selected_by_side: Dict[CardinalSide, Tuple[ContourKeypoint, ...]] = {}
    for side in CardinalSide:
        side_values = [value for value in base_observations if value.side is side]
        reasons: Dict[Tuple[int, int, int], Set[str]] = {}
        selected: Dict[Tuple[int, int, int], _Observation] = {}

        uniform = _farthest_arc_selection(
            side_values,
            min(settings.uniform_keypoints_per_side, len(side_values)),
        )
        for observation in uniform:
            selected[observation.key] = observation
            reasons.setdefault(observation.key, set()).add("uniform_arc_length")

        ranked_curvature = sorted(
            side_values,
            key=lambda value: (-curvature[value.key], value.arc_fraction, value.key),
        )
        curvature_selected: List[_Observation] = []
        for observation in ranked_curvature:
            if len(curvature_selected) >= settings.curvature_keypoints_per_side:
                break
            if any(
                _circular_distance(observation.arc_fraction, existing.arc_fraction)
                < settings.curvature_nms_arc_fraction
                for existing in curvature_selected
            ):
                continue
            curvature_selected.append(observation)
            if len(selected) < settings.max_keypoints_per_side:
                selected[observation.key] = observation
                reasons.setdefault(observation.key, set()).add("curvature_multiradius")

        if settings.include_run_endpoint_windows:
            by_run: Dict[int, List[_Observation]] = {}
            for observation in side_values:
                by_run.setdefault(observation.run_index, []).append(observation)
            endpoint_values = []
            for run_index in sorted(by_run):
                ordered = sorted(
                    by_run[run_index],
                    key=lambda value: (value.path_fraction, value.key),
                )
                endpoint_values.extend((ordered[0], ordered[-1]))
            # Extremely fragmented contours remain bounded.  Circular farthest
            # sampling retains endpoint coverage without a longest-run oracle.
            room = settings.max_keypoints_per_side - len(selected)
            endpoint_pool = [
                value for value in endpoint_values if value.key not in selected
            ]
            for observation in _farthest_arc_selection(endpoint_pool, max(0, room)):
                selected[observation.key] = observation
                reasons.setdefault(observation.key, set()).add(
                    "retained_run_endpoint_nearest_window"
                )
            for observation in endpoint_values:
                if observation.key in selected:
                    reasons.setdefault(observation.key, set()).add(
                        "retained_run_endpoint_nearest_window"
                    )

        # If curvature and uniform/endpoint anchors overlapped, fill any spare
        # capacity with the next non-max-suppressed curvature extrema.
        for observation in ranked_curvature:
            if len(selected) >= settings.max_keypoints_per_side:
                break
            if observation.key in selected:
                continue
            if any(
                _circular_distance(observation.arc_fraction, existing.arc_fraction)
                < settings.curvature_nms_arc_fraction
                for existing in curvature_selected
            ):
                continue
            selected[observation.key] = observation
            curvature_selected.append(observation)
            reasons.setdefault(observation.key, set()).add("curvature_multiradius")

        anchors: List[ContourKeypoint] = []
        for ordinal, observation in enumerate(
            sorted(selected.values(), key=lambda value: (value.arc_fraction, value.key))
        ):
            anchor_id = "kp-{}-r{:03d}-{:03d}".format(
                side.value, observation.run_index, ordinal
            )
            tokens = []
            for scale_index in range(scale_count):
                sequence = sequence_map.get((side, observation.run_index, scale_index))
                if sequence is None:
                    if settings.require_all_scales:
                        raise RuntimeError("complete-scale selection invariant failed")
                    continue
                tokens.append(_nearest_scale_token(observation, sequence, anchor_id))
            if not tokens:
                continue
            anchors.append(
                ContourKeypoint(
                    anchor_id=anchor_id,
                    side=side,
                    run_index=observation.run_index,
                    anchor_arc_fraction=observation.arc_fraction,
                    anchor_path_fraction=observation.path_fraction,
                    center_row_col=observation.center,
                    curvature_score_radians=curvature[observation.key],
                    selection_reasons=tuple(sorted(reasons[observation.key])),
                    tokens=tuple(tokens),
                )
            )
        selected_by_side[side] = tuple(anchors)

    keypoints = tuple(
        keypoint for side in CardinalSide for keypoint in selected_by_side[side]
    )
    counts = tuple((side.value, len(selected_by_side[side])) for side in CardinalSide)
    return FragmentKeypointSet(
        fragment_config_fingerprint=artifact.config_fingerprint,
        scale_count=scale_count,
        keypoints=keypoints,
        keypoint_counts_by_side=counts,
    )


def _flatten_tokens(
    keypoints: Sequence[ContourKeypoint],
) -> Tuple[KeypointScaleToken, ...]:
    # Scale-major order preserves a complete ordered contour sequence within
    # each scale.  It also makes the default same-scale correspondence graph
    # block diagonal instead of interleaving scales between adjacent anchors.
    ordered = sorted(
        keypoints,
        key=lambda value: (
            value.anchor_arc_fraction,
            value.run_index,
            value.anchor_id,
        ),
    )
    scales = sorted(
        {token.scale_index for keypoint in ordered for token in keypoint.tokens}
    )
    return tuple(
        token
        for scale_index in scales
        for keypoint in ordered
        for token in keypoint.tokens
        if token.scale_index == scale_index
    )


def build_keypoint_pair_candidates(
    fragment_a: FragmentKeypointSet,
    fragment_b: FragmentKeypointSet,
    config: Optional[ContourKeypointConfig] = None,
) -> KeypointPairResult:
    """Enumerate all four upright keypoint-matching direction hypotheses.

    There is deliberately no target-direction argument.  Facing sides are a
    geometric sparsity prior, not supervision: every available direction is
    emitted in the frozen :data:`DEFAULT_DIRECTION_ORDER`.
    """

    if not isinstance(fragment_a, FragmentKeypointSet) or not isinstance(
        fragment_b, FragmentKeypointSet
    ):
        raise TypeError("fragment_a and fragment_b must be FragmentKeypointSet")
    settings = config or ContourKeypointConfig()
    if not isinstance(settings, ContourKeypointConfig):
        raise TypeError("config must be ContourKeypointConfig")
    if fragment_a.scale_count != fragment_b.scale_count:
        raise ValueError("fragment keypoint scale counts differ")

    by_side_a = fragment_a.by_side
    by_side_b = fragment_b.by_side
    candidates = []
    unavailable = []
    affinity_elements = 0
    sinkhorn_elements = 0
    max_a = 0
    max_b = 0
    for direction in DEFAULT_DIRECTION_ORDER:
        side_a, side_b = direction.facing_sides
        tokens_a = _flatten_tokens(by_side_a[side_a])
        tokens_b = _flatten_tokens(by_side_b[side_b])
        if not tokens_a or not tokens_b:
            unavailable.append(direction.value)
            continue
        patches_a = _readonly(
            np.stack([token.channels for token in tokens_a], axis=0), np.float32
        )
        patches_b = _readonly(
            np.stack([token.channels for token in tokens_b], axis=0), np.float32
        )
        if settings.same_scale_correspondence_only:
            allowed = np.asarray(
                [
                    [first.scale_index == second.scale_index for second in tokens_b]
                    for first in tokens_a
                ],
                dtype=np.bool_,
            )
        else:
            allowed = np.ones((len(tokens_a), len(tokens_b)), dtype=np.bool_)
        allowed = _readonly(allowed, np.bool_)
        candidates.append(
            KeypointPairCandidate(
                candidate_id="{}:contour-keypoints".format(direction.value),
                direction=direction,
                tokens_a=tokens_a,
                tokens_b=tokens_b,
                patches_a=patches_a,
                patches_b=patches_b,
                correspondence_mask=allowed,
            )
        )
        max_a = max(max_a, len(tokens_a))
        max_b = max(max_b, len(tokens_b))
        affinity_elements += len(tokens_a) * len(tokens_b)
        sinkhorn_elements += (len(tokens_a) + 1) * (len(tokens_b) + 1)

    scale_count = fragment_a.scale_count
    max_tokens = settings.max_keypoints_per_side * scale_count
    complexity = KeypointComplexity(
        candidate_count=len(candidates),
        max_tokens_a=max_a,
        max_tokens_b=max_b,
        affinity_elements=affinity_elements,
        sinkhorn_elements=sinkhorn_elements,
        configured_max_tokens_per_fragment_direction=max_tokens,
        configured_max_affinity_elements_per_direction=max_tokens * max_tokens,
        configured_max_sinkhorn_elements_per_pair=(
            len(DEFAULT_DIRECTION_ORDER) * (max_tokens + 1) * (max_tokens + 1)
        ),
    )
    return KeypointPairResult(
        candidates=tuple(candidates),
        unavailable_directions=tuple(unavailable),
        complexity=complexity,
    )


__all__ = [
    "ContourKeypoint",
    "ContourKeypointConfig",
    "FragmentKeypointSet",
    "KEYPOINT_GEOMETRY_VERSION",
    "KeypointComplexity",
    "KeypointPairCandidate",
    "KeypointPairResult",
    "KeypointScaleToken",
    "build_fragment_keypoints",
    "build_keypoint_pair_candidates",
]
