#!/usr/bin/env python3
"""Deterministic balanced-distractor evaluation for strict real Dunhuang pairs.

The original 547-pair strict evaluation remains a separate, unchanged result.
This optional evaluation keeps those records in their original order and appends
469 *constructed* cross-case distractors.  The strict population contains 938
fragment occurrences, so every occurrence can be used exactly once.  Together
with the 39 manifest-labelled strict negatives this yields 508 positives and
508 negatives.

Constructed distractors are not ground-truth negative joins.  They are frozen
before model inference from alpha-mask dimensions and foreground area only.
They must never be used for training, threshold selection, or hyperparameter
tuning, and their error rates are reported separately from the 39 strict
manifest negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from staging.pairwise_v0_2.baselines.historical_mm_siamese import (
    HISTORICAL_MM_BASELINE_ID,
    load_historical_mm_checkpoint,
)
from staging.pairwise_v0_2.baselines.matched_route_a_siamese import (
    MATCHED_ROUTE_A_SIAMESE_ID,
)
from staging.pairwise_v0_2.baselines.real_dunhuang_evaluation import (
    DEFAULT_TARGET_LONG_SIDE,
    EXPECTED_STRICT_NEGATIVE_COUNT,
    EXPECTED_STRICT_POSITIVE_COUNT,
    REAL_DUNHUANG_DATASET_ID,
    RealArmScorer,
    RealDunhuangEvaluationError,
    RealPairDataset,
    _winner_loader_from_run,
    evaluate_real_dunhuang_four_arms,
    load_strict_real_pair_dataset,
    score_loaded_winner_direct,
    write_real_dunhuang_evaluation,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.geometry_cache import (
    GeometryArtifactCache,
    GeometryCacheLimits,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArmName,
    record_sequence_fingerprint,
)


BALANCED_DISTRACTOR_SCHEMA_VERSION = (
    "dunhuang-real-balanced-constructed-distractor-evaluation/0.1"
)
BALANCED_DISTRACTOR_DATASET_ID = (
    "real_dunhuang_strict_plus_constructed_balanced_alpha_v0_1"
)
BALANCED_DISTRACTOR_LABEL_ORIGIN = (
    "constructed_cross_case_alpha_scale_matched_distractor_not_gt_v0_1"
)
DEFAULT_BALANCED_DISTRACTOR_SEED = (
    "real-dunhuang-balanced-distractors-v1-fixed-20260830"
)
EXPECTED_REAL_FRAGMENT_OCCURRENCE_COUNT = 938
EXPECTED_CONSTRUCTED_DISTRACTOR_COUNT = 469
EXPECTED_BALANCED_PAIR_COUNT = 1016
EXPECTED_BALANCED_POSITIVE_COUNT = 508
EXPECTED_BALANCED_NEGATIVE_COUNT = 508
_SCALE_FEATURE_NAMES = (
    "log_alpha_height_over_target",
    "log_alpha_width_over_target",
    "log_alpha_foreground_area_over_target_squared",
)
_LOCAL_METHODS = (
    AblationArmName.LOCAL_DUAL_SOFTMAX.value,
    AblationArmName.LOCAL_DUSTBIN_SINKHORN.value,
    AblationArmName.KEYPOINT_DUAL_SOFTMAX.value,
    AblationArmName.KEYPOINT_DUSTBIN_SINKHORN.value,
)
_SIX_METHODS = frozenset(
    _LOCAL_METHODS + (HISTORICAL_MM_BASELINE_ID, MATCHED_ROUTE_A_SIAMESE_ID)
)


def _canonical_json(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(child) for child in item]
        return item

    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seeded_integer(seed: str, *parts: str) -> int:
    payload = _canonical_json([seed, *parts])
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _quantiles(values: Sequence[float]) -> Mapping[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise RealDunhuangEvaluationError("scale-match summary is invalid")
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.9)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


@dataclass(frozen=True)
class BalancedDistractorConfig:
    """Frozen, label-blind construction contract."""

    seed: str = DEFAULT_BALANCED_DISTRACTOR_SEED
    strict_positive_count: int = EXPECTED_STRICT_POSITIVE_COUNT
    strict_negative_count: int = EXPECTED_STRICT_NEGATIVE_COUNT
    unique_fragment_occurrence_count: int = EXPECTED_REAL_FRAGMENT_OCCURRENCE_COUNT
    constructed_distractor_count: int = EXPECTED_CONSTRUCTED_DISTRACTOR_COUNT
    max_fragments_per_case: int = 4
    forbid_same_case: bool = True
    forbid_same_alpha_sha256: bool = True
    max_constructed_uses_per_fragment_occurrence: int = 1
    max_constructed_pairs_per_unordered_case_pair: int = 1
    max_alpha_dimension_ratio: float = 2.0
    max_alpha_foreground_area_ratio: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.seed, str) or not self.seed:
            raise ValueError("balanced distractor seed must be a nonempty string")
        for name in (
            "strict_positive_count",
            "strict_negative_count",
            "unique_fragment_occurrence_count",
            "constructed_distractor_count",
            "max_fragments_per_case",
            "max_constructed_uses_per_fragment_occurrence",
            "max_constructed_pairs_per_unordered_case_pair",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(name + " must be a non-negative integer")
        if self.strict_positive_count < 1 or self.constructed_distractor_count < 1:
            raise ValueError("balanced construction counts must be positive")
        if self.max_fragments_per_case < 1:
            raise ValueError("max_fragments_per_case must be positive")
        if self.unique_fragment_occurrence_count != (
            2 * self.constructed_distractor_count
        ):
            raise ValueError(
                "every fragment occurrence must be used exactly once by construction"
            )
        if self.max_constructed_uses_per_fragment_occurrence != 1:
            raise ValueError("formal balanced construction requires one use per fragment")
        if self.max_constructed_pairs_per_unordered_case_pair != 1:
            raise ValueError("formal balanced construction requires unique case pairs")
        if not self.forbid_same_case or not self.forbid_same_alpha_sha256:
            raise ValueError("same-case and same-alpha distractors must be forbidden")
        for name in (
            "max_alpha_dimension_ratio",
            "max_alpha_foreground_area_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(name + " must be finite and at least one")
        if self.balanced_positive_count != self.balanced_negative_count:
            raise ValueError("constructed population must be exactly class balanced")

    @property
    def balanced_positive_count(self) -> int:
        return self.strict_positive_count

    @property
    def balanced_negative_count(self) -> int:
        return self.strict_negative_count + self.constructed_distractor_count


@dataclass(frozen=True)
class _AlphaScaleDescriptor:
    reference: MaskMemberRef
    source_case_uid: str
    height: int
    width: int
    foreground_area: int
    feature: Tuple[float, float, float]

    @property
    def fragment_id(self) -> str:
        return self.reference.fragment_id

    @property
    def alpha_sha256(self) -> str:
        value = self.reference.content_sha256
        if value is None:  # pragma: no cover - strict alpha records always bind it
            raise RealDunhuangEvaluationError("real fragment lacks alpha SHA-256")
        return value


@dataclass(frozen=True)
class BalancedRealPairBuild:
    dataset: RealPairDataset
    receipt: Mapping[str, Any]
    constructed_records: Tuple[TrainingPairRecord, ...]

    def __post_init__(self) -> None:
        if not self.constructed_records:
            raise ValueError("balanced build lacks constructed records")
        if tuple(self.dataset.records[-len(self.constructed_records) :]) != (
            self.constructed_records
        ):
            raise ValueError("constructed records must be the balanced dataset suffix")


def _strict_fragment_occurrences(
    dataset: RealPairDataset,
    config: BalancedDistractorConfig,
) -> Tuple[Tuple[MaskMemberRef, ...], Mapping[str, Tuple[MaskMemberRef, ...]]]:
    records = tuple(dataset.records)
    observed_positive = sum(record.label for record in records)
    observed_negative = len(records) - observed_positive
    if (
        dataset.positive_count != config.strict_positive_count
        or dataset.negative_count != config.strict_negative_count
        or observed_positive != config.strict_positive_count
        or observed_negative != config.strict_negative_count
    ):
        raise RealDunhuangEvaluationError(
            "balanced construction requires the exact strict label population"
        )
    if any(
        record.dataset_id != REAL_DUNHUANG_DATASET_ID
        or record.split != "val"
        or record.label_origin != "real_test_v0_1_strict_manifest"
        for record in records
    ):
        raise RealDunhuangEvaluationError(
            "balanced construction accepts strict manifest records only"
        )

    by_fragment: Dict[str, MaskMemberRef] = {}
    by_case: Dict[str, Dict[str, MaskMemberRef]] = defaultdict(dict)
    for record in records:
        if (
            record.fragment_a.component_id != record.fragment_b.component_id
            or record.fragment_a.component_id != record.component_id
        ):
            raise RealDunhuangEvaluationError("strict pair unexpectedly crosses cases")
        for reference in (record.fragment_a, record.fragment_b):
            previous = by_fragment.setdefault(reference.fragment_id, reference)
            if previous != reference:
                raise RealDunhuangEvaluationError(
                    "strict fragment occurrence has inconsistent references"
                )
            by_case[reference.component_id][reference.fragment_id] = reference
    if len(by_fragment) != config.unique_fragment_occurrence_count:
        raise RealDunhuangEvaluationError(
            "strict fragment occurrence count changed: {} != {}".format(
                len(by_fragment), config.unique_fragment_occurrence_count
            )
        )
    if len(by_case) != dataset.case_count:
        raise RealDunhuangEvaluationError("strict case coverage changed")
    if any(
        not values or len(values) > config.max_fragments_per_case
        for values in by_case.values()
    ):
        raise RealDunhuangEvaluationError("strict per-case fragment bound changed")
    return (
        tuple(by_fragment[key] for key in sorted(by_fragment)),
        {
            case_uid: tuple(values[key] for key in sorted(values))
            for case_uid, values in sorted(by_case.items())
        },
    )


def _scale_descriptors(
    dataset: RealPairDataset,
    references: Sequence[MaskMemberRef],
) -> Tuple[_AlphaScaleDescriptor, ...]:
    target = float(dataset.target_long_side)
    output = []
    for reference in references:
        mask = np.asarray(dataset.mask_loader(reference))
        if (
            mask.ndim != 2
            or mask.dtype != np.bool_
            or mask.size == 0
            or not mask.any()
        ):
            raise RealDunhuangEvaluationError(
                "balanced scale matching requires a nonempty bool alpha mask"
            )
        height, width = (int(value) for value in mask.shape)
        foreground_area = int(np.count_nonzero(mask))
        feature = (
            math.log(height / target),
            math.log(width / target),
            math.log(foreground_area / (target * target)),
        )
        if not all(math.isfinite(value) for value in feature):
            raise RealDunhuangEvaluationError("alpha scale feature is non-finite")
        output.append(
            _AlphaScaleDescriptor(
                reference=reference,
                source_case_uid=reference.component_id,
                height=height,
                width=width,
                foreground_area=foreground_area,
                feature=feature,
            )
        )
    return tuple(output)


def _balanced_case_partition(
    descriptors: Sequence[_AlphaScaleDescriptor], seed: str
) -> Tuple[Tuple[_AlphaScaleDescriptor, ...], Tuple[_AlphaScaleDescriptor, ...]]:
    by_case: Dict[str, list[_AlphaScaleDescriptor]] = defaultdict(list)
    for value in descriptors:
        by_case[value.source_case_uid].append(value)
    target_left = len(descriptors) // 2
    base_left = sum(len(values) // 2 for values in by_case.values())
    odd_cases = sorted(
        (case_uid for case_uid, values in by_case.items() if len(values) % 2),
        key=lambda case_uid: (_seeded_integer(seed, "odd-case-side", case_uid), case_uid),
    )
    extra_left = target_left - base_left
    if extra_left < 0 or extra_left > len(odd_cases):
        raise RealDunhuangEvaluationError("case-balanced bipartition is infeasible")
    left_heavy = set(odd_cases[:extra_left])
    left_quota = {
        case_uid: len(values) // 2 + int(case_uid in left_heavy)
        for case_uid, values in by_case.items()
    }
    right_quota = {
        case_uid: len(values) - left_quota[case_uid]
        for case_uid, values in by_case.items()
    }
    left = []
    right = []
    used_left = Counter()
    used_right = Counter()
    # Traverse alpha scale from small to large and distribute nearby fragments
    # across opposite sides whenever the per-case quotas allow it.  This avoids
    # putting all rare small or thin fragments on one side of the bipartite
    # problem while retaining a fixed, label-blind partition.
    ordered = sorted(
        descriptors,
        key=lambda value: (
            value.feature[2],
            value.feature[0],
            value.feature[1],
            _seeded_integer(seed, "fragment-side-tie", value.fragment_id),
            value.fragment_id,
        ),
    )
    for value in ordered:
        case_uid = value.source_case_uid
        left_available = used_left[case_uid] < left_quota[case_uid]
        right_available = used_right[case_uid] < right_quota[case_uid]
        if not left_available and not right_available:  # pragma: no cover
            raise RealDunhuangEvaluationError("case partition quota was exceeded")
        if left_available and right_available:
            if len(left) != len(right):
                choose_left = len(left) < len(right)
            else:
                choose_left = (
                    _seeded_integer(seed, "fragment-side", value.fragment_id) % 2 == 0
                )
        else:
            choose_left = left_available
        if choose_left:
            left.append(value)
            used_left[case_uid] += 1
        else:
            right.append(value)
            used_right[case_uid] += 1
    left.sort(key=lambda value: value.fragment_id)
    right.sort(key=lambda value: value.fragment_id)
    if len(left) != len(right) or len(left) + len(right) != len(descriptors):
        raise RealDunhuangEvaluationError("balanced fragment partition changed size")
    return tuple(left), tuple(right)


def _scale_distance(
    left: _AlphaScaleDescriptor, right: _AlphaScaleDescriptor
) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(left.feature, dtype=np.float64)
                - np.asarray(right.feature, dtype=np.float64)
            )
        )
    )


def _scale_optimization_cost(
    left: _AlphaScaleDescriptor, right: _AlphaScaleDescriptor
) -> float:
    difference = np.abs(
        np.asarray(left.feature, dtype=np.float64)
        - np.asarray(right.feature, dtype=np.float64)
    )
    # The mean term gives smooth nearest-scale matching.  Squaring the worst
    # log-ratio prevents the minimum-sum assignment from sacrificing one very
    # mismatched fragment to obtain many tiny improvements elsewhere.
    return float(np.mean(difference) + np.max(difference) ** 2)


def _allowed_cross_case_pair(
    left: _AlphaScaleDescriptor, right: _AlphaScaleDescriptor
) -> bool:
    return (
        left.source_case_uid != right.source_case_uid
        and left.alpha_sha256 != right.alpha_sha256
    )


def _case_pair_key(
    left: _AlphaScaleDescriptor, right: _AlphaScaleDescriptor
) -> Tuple[str, str]:
    return tuple(sorted((left.source_case_uid, right.source_case_uid)))


def _duplicate_excess(
    left: Sequence[_AlphaScaleDescriptor],
    right: Sequence[_AlphaScaleDescriptor],
    assignment: Sequence[int],
) -> Tuple[int, Counter[Tuple[str, str]]]:
    counts = Counter(
        _case_pair_key(left[index], right[right_index])
        for index, right_index in enumerate(assignment)
    )
    return sum(max(0, count - 1) for count in counts.values()), counts


def _repair_repeated_case_pairs(
    *,
    left: Sequence[_AlphaScaleDescriptor],
    right: Sequence[_AlphaScaleDescriptor],
    assignment: Sequence[int],
    optimization_cost: np.ndarray,
    seed: str,
) -> Tuple[Tuple[int, ...], int]:
    """Use deterministic 2-opt swaps until unordered case pairs are unique."""

    selected = list(int(value) for value in assignment)
    repair_count = 0
    while True:
        current_excess, counts = _duplicate_excess(left, right, selected)
        if current_excess == 0:
            return tuple(selected), repair_count
        duplicate_indices = [
            index
            for index, right_index in enumerate(selected)
            if counts[_case_pair_key(left[index], right[right_index])] > 1
        ]
        best = None
        for first in duplicate_indices:
            first_right = selected[first]
            for second in range(len(selected)):
                if first == second:
                    continue
                second_right = selected[second]
                new_first = right[second_right]
                new_second = right[first_right]
                if not _allowed_cross_case_pair(left[first], new_first):
                    continue
                if not _allowed_cross_case_pair(left[second], new_second):
                    continue
                candidate = list(selected)
                candidate[first], candidate[second] = second_right, first_right
                new_excess, _new_counts = _duplicate_excess(left, right, candidate)
                if new_excess >= current_excess:
                    continue
                cost_delta = float(
                    optimization_cost[first, second_right]
                    + optimization_cost[second, first_right]
                    - optimization_cost[first, first_right]
                    - optimization_cost[second, second_right]
                )
                tie = _seeded_integer(
                    seed,
                    "case-pair-repair",
                    left[first].fragment_id,
                    right[first_right].fragment_id,
                    left[second].fragment_id,
                    right[second_right].fragment_id,
                )
                key = (new_excess, cost_delta, tie, first, second)
                if best is None or key < best[0]:
                    best = (key, candidate)
        if best is None:
            raise RealDunhuangEvaluationError(
                "cannot remove repeated constructed case pairs by deterministic 2-opt"
            )
        selected = best[1]
        repair_count += 1
        if repair_count > len(selected):
            raise RealDunhuangEvaluationError("case-pair repair did not converge")


def _minimum_scale_matching(
    descriptors: Sequence[_AlphaScaleDescriptor], seed: str
) -> Tuple[
    Tuple[Tuple[_AlphaScaleDescriptor, _AlphaScaleDescriptor, float], ...], int
]:
    left, right = _balanced_case_partition(descriptors, seed)
    count = len(left)
    distance = np.empty((count, count), dtype=np.float64)
    optimization_cost = np.empty((count, count), dtype=np.float64)
    allowed = np.empty((count, count), dtype=np.bool_)
    selection_cost = np.empty((count, count), dtype=np.float64)
    finite_costs = []
    for row, first in enumerate(left):
        for column, second in enumerate(right):
            value = _scale_distance(first, second)
            distance[row, column] = value
            robust_value = _scale_optimization_cost(first, second)
            optimization_cost[row, column] = robust_value
            valid = _allowed_cross_case_pair(first, second)
            allowed[row, column] = valid
            if valid:
                finite_costs.append(robust_value)
                jitter = (
                    _seeded_integer(
                        seed,
                        "hungarian-tie",
                        first.fragment_id,
                        second.fragment_id,
                    )
                    / float(2**256)
                ) * 1e-10
                selection_cost[row, column] = robust_value + jitter
            else:
                selection_cost[row, column] = np.nan
    if not finite_costs:
        raise RealDunhuangEvaluationError("no cross-case scale matches are available")
    forbidden_cost = max(finite_costs) + 1_000_000.0
    selection_cost[~allowed] = forbidden_cost
    row_index, column_index = linear_sum_assignment(selection_cost)
    if not np.array_equal(row_index, np.arange(count, dtype=row_index.dtype)):
        raise RealDunhuangEvaluationError("Hungarian matching did not cover left side")
    assignment = tuple(int(value) for value in column_index)
    if any(not allowed[index, value] for index, value in enumerate(assignment)):
        raise RealDunhuangEvaluationError(
            "same-case or same-alpha edge entered constructed matching"
        )
    repaired, repair_count = _repair_repeated_case_pairs(
        left=left,
        right=right,
        assignment=assignment,
        optimization_cost=optimization_cost,
        seed=seed,
    )
    output = tuple(
        (left[index], right[right_index], float(distance[index, right_index]))
        for index, right_index in enumerate(repaired)
    )
    return output, repair_count


def _constructed_cluster_id(case_a: str, case_b: str) -> str:
    pair = sorted((case_a, case_b))
    return "constructed-casepair/sha256/" + _sha256(_canonical_json(pair))


def _descriptor_receipt(value: _AlphaScaleDescriptor) -> Mapping[str, Any]:
    return {
        "fragment_id": value.fragment_id,
        "source_case_uid": value.source_case_uid,
        "alpha_sha256": value.alpha_sha256,
        "alpha_height": value.height,
        "alpha_width": value.width,
        "alpha_foreground_area": value.foreground_area,
    }


def _constructed_record(
    first: _AlphaScaleDescriptor,
    second: _AlphaScaleDescriptor,
    *,
    scale_distance: float,
    seed: str,
) -> Tuple[TrainingPairRecord, Mapping[str, Any]]:
    if not _allowed_cross_case_pair(first, second):
        raise RealDunhuangEvaluationError("invalid constructed distractor edge")
    orientation = _seeded_integer(
        seed,
        "pair-orientation",
        *sorted((first.fragment_id, second.fragment_id)),
    )
    if orientation % 2:
        first, second = second, first
    cluster_id = _constructed_cluster_id(
        first.source_case_uid, second.source_case_uid
    )
    fragment_a = replace(
        first.reference,
        canonical_group_id=cluster_id,
        component_id=cluster_id,
    )
    fragment_b = replace(
        second.reference,
        canonical_group_id=cluster_id,
        component_id=cluster_id,
    )
    height_ratio = max(first.height, second.height) / min(first.height, second.height)
    width_ratio = max(first.width, second.width) / min(first.width, second.width)
    area_ratio = max(first.foreground_area, second.foreground_area) / min(
        first.foreground_area, second.foreground_area
    )
    provenance = {
        "real_dunhuang_external_test": True,
        "alpha_mask_only": True,
        "bbox_origin_exposed_to_model": False,
        "gt_composite_exposed_to_model": False,
        "constructed_distractor": True,
        "constructed_is_ground_truth_negative": False,
        "constructed_semantics": (
            "cross_case_scale_matched_distractor_not_a_ground_truth_negative"
        ),
        "construction_seed": seed,
        "source_case_uid_a": first.source_case_uid,
        "source_case_uid_b": second.source_case_uid,
        "alpha_scale_distance": float(scale_distance),
        "alpha_height_ratio": float(height_ratio),
        "alpha_width_ratio": float(width_ratio),
        "alpha_foreground_area_ratio": float(area_ratio),
    }
    record = TrainingPairRecord(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        label=False,
        direction_b_wrt_a=None,
        dataset_id=REAL_DUNHUANG_DATASET_ID,
        canonical_group_id=cluster_id,
        component_id=cluster_id,
        split="val",
        canonical_pair_key=tuple(
            sorted((fragment_a.fragment_id, fragment_b.fragment_id))
        ),
        label_origin=BALANCED_DISTRACTOR_LABEL_ORIGIN,
        provenance=provenance,
    )
    row = {
        "pair_id": record.pair_id,
        "canonical_pair_key": list(record.canonical_pair_key),
        "constructed_is_ground_truth_negative": False,
        "case_pair_cluster_id": cluster_id,
        "fragment_a": _descriptor_receipt(first),
        "fragment_b": _descriptor_receipt(second),
        "alpha_scale_distance": float(scale_distance),
        "alpha_height_ratio": float(height_ratio),
        "alpha_width_ratio": float(width_ratio),
        "alpha_foreground_area_ratio": float(area_ratio),
    }
    return record, row


def build_balanced_real_pair_dataset(
    strict: RealPairDataset,
    *,
    config: BalancedDistractorConfig = BalancedDistractorConfig(),
) -> BalancedRealPairBuild:
    """Keep strict records untouched and append a frozen constructed plan."""

    if not isinstance(strict, RealPairDataset):
        raise TypeError("strict must be a RealPairDataset")
    if not isinstance(config, BalancedDistractorConfig):
        raise TypeError("config must be BalancedDistractorConfig")
    references, by_case = _strict_fragment_occurrences(strict, config)
    descriptors = _scale_descriptors(strict, references)
    selected, repair_count = _minimum_scale_matching(descriptors, config.seed)
    if len(selected) != config.constructed_distractor_count:
        raise RealDunhuangEvaluationError("constructed distractor count changed")

    records_and_rows = [
        _constructed_record(
            first,
            second,
            scale_distance=distance,
            seed=config.seed,
        )
        for first, second, distance in selected
    ]
    records_and_rows.sort(key=lambda value: value[0].canonical_pair_key)
    constructed_records = tuple(value[0] for value in records_and_rows)
    constructed_rows = tuple(value[1] for value in records_and_rows)

    fragment_use = Counter(
        fragment_id
        for record in constructed_records
        for fragment_id in record.canonical_pair_key
    )
    if set(fragment_use) != {reference.fragment_id for reference in references} or any(
        value != 1 for value in fragment_use.values()
    ):
        raise RealDunhuangEvaluationError(
            "constructed plan does not use every fragment occurrence exactly once"
        )
    source_case_incidence = Counter()
    case_pair_counts = Counter()
    for row in constructed_rows:
        case_a = row["fragment_a"]["source_case_uid"]
        case_b = row["fragment_b"]["source_case_uid"]
        source_case_incidence.update((case_a, case_b))
        case_pair_counts[tuple(sorted((case_a, case_b)))] += 1
    expected_case_incidence = {
        case_uid: len(values) for case_uid, values in by_case.items()
    }
    if dict(source_case_incidence) != expected_case_incidence:
        raise RealDunhuangEvaluationError("constructed case incidence is unbalanced")
    if max(case_pair_counts.values(), default=0) != 1:
        raise RealDunhuangEvaluationError("constructed unordered case pair was repeated")

    combined_records = tuple(strict.records) + constructed_records
    if len({record.pair_id for record in combined_records}) != len(combined_records):
        raise RealDunhuangEvaluationError("balanced pair IDs are not unique")
    positive_count = sum(record.label for record in combined_records)
    negative_count = len(combined_records) - positive_count
    if (
        positive_count != config.balanced_positive_count
        or negative_count != config.balanced_negative_count
    ):
        raise RealDunhuangEvaluationError("balanced label counts changed")

    distances = [float(row["alpha_scale_distance"]) for row in constructed_rows]
    height_ratios = [float(row["alpha_height_ratio"]) for row in constructed_rows]
    width_ratios = [float(row["alpha_width_ratio"]) for row in constructed_rows]
    area_ratios = [
        float(row["alpha_foreground_area_ratio"]) for row in constructed_rows
    ]
    if (
        max(height_ratios) > config.max_alpha_dimension_ratio
        or max(width_ratios) > config.max_alpha_dimension_ratio
        or max(area_ratios) > config.max_alpha_foreground_area_ratio
    ):
        raise RealDunhuangEvaluationError(
            "constructed alpha-scale match exceeds the frozen ratio bound"
        )
    alpha_counts = Counter(value.alpha_sha256 for value in descriptors)
    strict_sequence_sha256 = record_sequence_fingerprint(strict.records)
    constructed_sequence_sha256 = record_sequence_fingerprint(constructed_records)
    combined_sequence_sha256 = record_sequence_fingerprint(combined_records)
    selection_sha256 = _sha256(_canonical_json(constructed_rows))
    receipt = {
        "schema_version": BALANCED_DISTRACTOR_SCHEMA_VERSION,
        "status": "complete_label_blind_constructed_distractor_plan",
        "seed": config.seed,
        "seed_sha256": _sha256(config.seed.encode("utf-8")),
        "strict_population": {
            "pair_count": len(strict.records),
            "positive_count": strict.positive_count,
            "manifest_negative_count": strict.negative_count,
            "pair_sequence_sha256": strict_sequence_sha256,
            "records_preserved_as_exact_prefix": True,
        },
        "balanced_population": {
            "pair_count": len(combined_records),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "manifest_negative_count": strict.negative_count,
            "constructed_distractor_count": len(constructed_records),
            "pair_sequence_sha256": combined_sequence_sha256,
        },
        "construction": {
            "selection_uses_model_scores": False,
            "selection_uses_pair_labels": False,
            "selection_uses_gt_bbox_or_direction": False,
            "selection_features": list(_SCALE_FEATURE_NAMES),
            "selection_algorithm": (
                "scale_stratified_case_balanced_seeded_bipartition_then_"
                "hungarian_robust_alpha_scale_cost_then_deterministic_"
                "casepair_2opt"
            ),
            "same_case_forbidden": True,
            "same_alpha_sha256_forbidden": True,
            "fragment_occurrence_count": len(descriptors),
            "unique_alpha_sha256_count": len(alpha_counts),
            "duplicate_alpha_occurrence_count": len(descriptors) - len(alpha_counts),
            "max_alpha_sha256_multiplicity": max(alpha_counts.values()),
            "uses_per_fragment_occurrence": 1,
            "max_constructed_pairs_incident_per_case": max(
                source_case_incidence.values()
            ),
            "max_constructed_pairs_per_unordered_case_pair": max(
                case_pair_counts.values()
            ),
            "frozen_scale_ratio_bounds": {
                "max_alpha_height_or_width_ratio": (
                    config.max_alpha_dimension_ratio
                ),
                "max_alpha_foreground_area_ratio": (
                    config.max_alpha_foreground_area_ratio
                ),
                "all_constructed_pairs_within_bounds": True,
            },
            "case_pair_2opt_repair_count": repair_count,
            "constructed_pair_sequence_sha256": constructed_sequence_sha256,
            "constructed_selection_sha256": selection_sha256,
            "scale_match": {
                "distance_mean_absolute_log_ratio": _quantiles(distances),
                "alpha_height_ratio": _quantiles(height_ratios),
                "alpha_width_ratio": _quantiles(width_ratios),
                "alpha_foreground_area_ratio": _quantiles(area_ratios),
            },
        },
        "negative_semantics": {
            "manifest_negative_count": strict.negative_count,
            "manifest_negatives_are_ground_truth_labelled": True,
            "constructed_distractor_count": len(constructed_records),
            "constructed_distractors_are_ground_truth_negatives": False,
            "required_label": (
                "constructed cross-case scale-matched distractor; not GT-negative"
            ),
            "scientific_use": (
                "external evaluation only; never training, threshold fitting, or tuning"
            ),
        },
        "constructed_pairs": list(constructed_rows),
    }
    receipt["content_sha256"] = _sha256(_canonical_json(receipt))
    dataset = RealPairDataset(
        records=combined_records,
        mask_loader=strict.mask_loader,
        manifest_sha256=strict.manifest_sha256,
        case_count=strict.case_count,
        positive_count=positive_count,
        negative_count=negative_count,
        target_long_side=strict.target_long_side,
    )
    return BalancedRealPairBuild(
        dataset=dataset,
        receipt=receipt,
        constructed_records=constructed_records,
    )


def _score_stratum(
    *,
    probability: Sequence[Optional[float]],
    valid: Sequence[bool],
    selected: Sequence[bool],
    positive: bool,
) -> Mapping[str, Any]:
    if not (len(probability) == len(valid) == len(selected)):
        raise RealDunhuangEvaluationError("origin-stratum vectors differ in length")
    indices = [index for index, keep in enumerate(selected) if keep]
    usable = [
        index
        for index in indices
        if valid[index]
        and probability[index] is not None
        and math.isfinite(float(probability[index]))
    ]
    predicted_positive = sum(float(probability[index]) >= 0.5 for index in usable)
    row: Dict[str, Any] = {
        "count": len(indices),
        "valid_count": len(usable),
        "coverage": len(usable) / len(indices) if indices else None,
        "mean_probability": (
            float(np.mean([float(probability[index]) for index in usable]))
            if usable
            else None
        ),
    }
    if positive:
        row["true_positive_rate_at_0_5"] = (
            predicted_positive / len(usable) if usable else None
        )
    else:
        row["false_positive_rate_at_0_5"] = (
            predicted_positive / len(usable) if usable else None
        )
    return row


def _attach_origin_strata(
    result: Dict[str, Any], strict_count: int
) -> None:
    labels = tuple(bool(value) for value in result["labels"])
    total = len(labels)
    strict_positive = tuple(index < strict_count and labels[index] for index in range(total))
    strict_negative = tuple(
        index < strict_count and not labels[index] for index in range(total)
    )
    constructed = tuple(index >= strict_count for index in range(total))
    common_valid = tuple(bool(value) for value in result["common_valid"])
    for row in result["methods"].values():
        probability = tuple(row["probability"])
        native_valid = tuple(bool(value) for value in row["valid"])
        row["origin_strata_native"] = {
            "strict_manifest_positive": _score_stratum(
                probability=probability,
                valid=native_valid,
                selected=strict_positive,
                positive=True,
            ),
            "strict_manifest_negative": _score_stratum(
                probability=probability,
                valid=native_valid,
                selected=strict_negative,
                positive=False,
            ),
            "constructed_cross_case_distractor_not_gt_negative": _score_stratum(
                probability=probability,
                valid=native_valid,
                selected=constructed,
                positive=False,
            ),
        }
        row["origin_strata_common_six_method"] = {
            "strict_manifest_positive": _score_stratum(
                probability=probability,
                valid=common_valid,
                selected=strict_positive,
                positive=True,
            ),
            "strict_manifest_negative": _score_stratum(
                probability=probability,
                valid=common_valid,
                selected=strict_negative,
                positive=False,
            ),
            "constructed_cross_case_distractor_not_gt_negative": _score_stratum(
                probability=probability,
                valid=common_valid,
                selected=constructed,
                positive=False,
            ),
        }


def evaluate_balanced_distractor_six_methods(
    *,
    strict_dataset: RealPairDataset,
    geometry_config: GeometryBatchConfig,
    geometry_cache: GeometryArtifactCache,
    arm_winner_loader: Callable[[AblationArmName], Any],
    historical_model: nn.Module,
    matched_model: nn.Module,
    arm_scorer: RealArmScorer = score_loaded_winner_direct,
    batch_size: int = 1,
    historical_device: Union[str, torch.device] = "cpu",
    historical_batch_size: int = 256,
    matched_device: Union[str, torch.device] = "cpu",
    matched_batch_size: int = 256,
    construction_config: BalancedDistractorConfig = BalancedDistractorConfig(),
) -> Mapping[str, Any]:
    """Evaluate four local winners and both Siamese controls on one frozen set."""

    if not isinstance(historical_model, nn.Module) or not isinstance(
        matched_model, nn.Module
    ):
        raise TypeError("balanced evaluation requires both Siamese controls")
    built = build_balanced_real_pair_dataset(
        strict_dataset,
        config=construction_config,
    )
    core = evaluate_real_dunhuang_four_arms(
        dataset=built.dataset,
        geometry_config=geometry_config,
        geometry_cache=geometry_cache,
        arm_winner_loader=arm_winner_loader,
        arm_scorer=arm_scorer,
        batch_size=batch_size,
        historical_model=historical_model,
        historical_device=historical_device,
        historical_batch_size=historical_batch_size,
        matched_model=matched_model,
        matched_device=matched_device,
        matched_batch_size=matched_batch_size,
    )
    result = dict(core)
    if frozenset(result.get("methods", {})) != _SIX_METHODS:
        raise RealDunhuangEvaluationError(
            "balanced evaluation requires exactly four local and two Siamese methods"
        )
    if result.get("pair_ids") != [record.pair_id for record in built.dataset.records]:
        raise RealDunhuangEvaluationError("six-method balanced pair order changed")
    _attach_origin_strata(result, len(strict_dataset.records))
    result.update(
        {
            "schema_version": BALANCED_DISTRACTOR_SCHEMA_VERSION,
            "status": (
                "complete_six_methods_balanced_with_constructed_"
                "cross_case_distractors_not_gt_negative"
            ),
            "dataset_id": BALANCED_DISTRACTOR_DATASET_ID,
            "strict_reference_dataset_id": REAL_DUNHUANG_DATASET_ID,
            "strict_547_result_replaced_or_modified": False,
            "strict_result_policy": (
                "the original strict-only 547-pair result remains primary and separate"
            ),
            "construction_receipt": built.receipt,
            "fairness": {
                "same_ordered_pair_ids_all_six_methods": True,
                "same_labels_all_six_methods": True,
                "same_source_alpha_masks_all_six_methods": True,
                "same_prepared_tensors_dual_vs_sinkhorn_within_representation": True,
                "architecture_specific_tensorization": {
                    "local_methods": (
                        "same cached contour-patch tensor within each representation"
                    ),
                    "siamese_controls": (
                        "same bool alpha masks and same deterministic 64x64 transform"
                    ),
                },
            },
            "negative_semantics": built.receipt["negative_semantics"],
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-path-receipt", type=Path, required=True)
    parser.add_argument("--main-root", type=Path)
    parser.add_argument("--supp-root", type=Path)
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--geometry-cache-dir", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--matched-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--historical-batch-size", type=int, default=256)
    parser.add_argument("--matched-batch-size", type=int, default=256)
    parser.add_argument(
        "--target-long-side", type=int, default=DEFAULT_TARGET_LONG_SIDE
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from staging.pairwise_v0_2.training.local_q1_route_a_research import (
        route_a_planning_config,
    )

    args = _parser().parse_args(argv)
    strict = load_strict_real_pair_dataset(
        args.manifest,
        args.local_path_receipt,
        main_root=args.main_root,
        supp_root=args.supp_root,
        target_long_side=args.target_long_side,
    )
    try:
        config = route_a_planning_config(args.route_config)
        cache = GeometryArtifactCache(
            args.geometry_cache_dir,
            limits=GeometryCacheLimits(),
        )
        winner_loader = _winner_loader_from_run(
            args.run_directory,
            device=args.device,
        )
        historical_model = load_historical_mm_checkpoint(
            args.historical_checkpoint,
            device=args.device,
        )
        matched_model = load_historical_mm_checkpoint(
            args.matched_checkpoint,
            device=args.device,
        )
        result = evaluate_balanced_distractor_six_methods(
            strict_dataset=strict,
            geometry_config=config.geometry_batch_config,
            geometry_cache=cache,
            arm_winner_loader=winner_loader,
            historical_model=historical_model,
            matched_model=matched_model,
            batch_size=args.batch_size,
            historical_device=args.device,
            historical_batch_size=args.historical_batch_size,
            matched_device=args.device,
            matched_batch_size=args.matched_batch_size,
        )
    finally:
        strict.mask_loader.close()
    write_real_dunhuang_evaluation(args.output, result)
    compact = {
        "status": result["status"],
        "pair_count": result["pair_count"],
        "positive_count": result["positive_count"],
        "negative_count": result["negative_count"],
        "common_valid_count": result["common_valid_count"],
        "constructed_selection_sha256": result["construction_receipt"][
            "construction"
        ]["constructed_selection_sha256"],
        "output": str(args.output),
        "metrics": {
            method: {
                "auroc": row["common_valid_metrics"]["row"]["auroc"],
                "auprc": row["common_valid_metrics"]["row"]["auprc"],
                "constructed_distractor_fpr": row[
                    "origin_strata_common_six_method"
                ]["constructed_cross_case_distractor_not_gt_negative"][
                    "false_positive_rate_at_0_5"
                ],
            }
            for method, row in result["methods"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "BALANCED_DISTRACTOR_DATASET_ID",
    "BALANCED_DISTRACTOR_LABEL_ORIGIN",
    "BALANCED_DISTRACTOR_SCHEMA_VERSION",
    "DEFAULT_BALANCED_DISTRACTOR_SEED",
    "EXPECTED_BALANCED_NEGATIVE_COUNT",
    "EXPECTED_BALANCED_PAIR_COUNT",
    "EXPECTED_BALANCED_POSITIVE_COUNT",
    "EXPECTED_CONSTRUCTED_DISTRACTOR_COUNT",
    "EXPECTED_REAL_FRAGMENT_OCCURRENCE_COUNT",
    "BalancedDistractorConfig",
    "BalancedRealPairBuild",
    "build_balanced_real_pair_dataset",
    "evaluate_balanced_distractor_six_methods",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
