#!/usr/bin/env python3
"""Read-only spatial-shortcut audit for Pairwise v0.2 mask archives.

This utility intentionally cannot open the sealed real-world test set.  It
accepts only the three synthetic-training archive families registered by the
v0.2 data stream, verifies their archive SHA-256 values through the canonical
lazy loader, and measures how well canvas coordinates alone recover pair
labels.  It never trains a model and never serializes runtime filesystem paths.

The frozen synthetic-subset protocol is:

* retained groups ranked by ``sha256(canonical_group_id)``;
* first 600 groups by default;
* every explicit unordered pair in those groups;
* row and equal-group-weight AUROC for three fixed coordinate-only scores.

Historical MM/ECCV are diagnostic supplements.  They use the same group rank,
then retain at most 32 pair rows per selected group by ``sha256(pair_id)``.
The cap is label-blind and keeps the read-only audit tractable.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from staging.pairwise_v0_2.pairwise_data.lazy_mask_loader import (
    ArchiveSourceSpec,
    LazyMaskArchiveLoader,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
    iter_historical_pair_records,
    iter_synthetic_pair_records,
)


SCHEMA_VERSION = "dunhuang-pairwise-spatial-shortcut-audit/0.2"
AUDIT_VERSION = "spatial-shortcut-audit-v0.2.0"
SYNTHETIC_DEFAULT_GROUPS = 600
HISTORICAL_DEFAULT_GROUPS = 128
HISTORICAL_DEFAULT_PAIRS_PER_GROUP = 32


@dataclass(frozen=True)
class MaskSpatialFeature:
    shape: Tuple[int, int]
    foreground_pixels: int
    centroid_row_col: Tuple[float, float]
    bbox_rc_exclusive: Tuple[int, int, int, int]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _rank(identifier: str) -> int:
    return int.from_bytes(hashlib.sha256(identifier.encode("utf-8")).digest(), "big")


def _selection_fingerprint(identifiers: Iterable[str]) -> str:
    digest = hashlib.sha256()
    count = 0
    for identifier in sorted(identifiers):
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\n")
        count += 1
    digest.update(("count={}".format(count)).encode("ascii"))
    return digest.hexdigest()


def mask_spatial_feature(mask: np.ndarray) -> MaskSpatialFeature:
    value = np.asarray(mask)
    if value.ndim != 2 or value.dtype != np.bool_ or value.size == 0:
        raise ValueError("spatial audit requires a non-empty 2D bool mask")
    rows, columns = np.nonzero(value)
    if rows.size == 0:
        raise ValueError("spatial audit cannot score an empty mask")
    return MaskSpatialFeature(
        shape=(int(value.shape[0]), int(value.shape[1])),
        foreground_pixels=int(rows.size),
        centroid_row_col=(float(rows.mean()), float(columns.mean())),
        bbox_rc_exclusive=(
            int(rows.min()),
            int(columns.min()),
            int(rows.max()) + 1,
            int(columns.max()) + 1,
        ),
    )


def spatial_scores(
    left: MaskSpatialFeature, right: MaskSpatialFeature
) -> Mapping[str, float]:
    """Return three frozen canvas-only scores; larger means more adjacent."""

    if left.shape != right.shape:
        raise ValueError("a spatial-shortcut pair must share one canvas shape")
    height, width = left.shape
    diagonal = math.hypot(float(height), float(width))
    center_distance = math.hypot(
        left.centroid_row_col[0] - right.centroid_row_col[0],
        left.centroid_row_col[1] - right.centroid_row_col[1],
    )
    ay0, ax0, ay1, ax1 = left.bbox_rc_exclusive
    by0, bx0, by1, bx1 = right.bbox_rc_exclusive
    intersection_height = max(0, min(ay1, by1) - max(ay0, by0))
    intersection_width = max(0, min(ax1, bx1) - max(ax0, bx0))
    intersection = float(intersection_height * intersection_width)
    area_a = float((ay1 - ay0) * (ax1 - ax0))
    area_b = float((by1 - by0) * (bx1 - bx0))
    union = area_a + area_b - intersection
    gap_x = float(max(0, ax0 - bx1, bx0 - ax1))
    gap_y = float(max(0, ay0 - by1, by0 - ay1))
    return {
        "negative_foreground_centroid_distance_over_canvas_diagonal": (
            -center_distance / diagonal
        ),
        "bbox_intersection_over_union": intersection / union if union > 0.0 else 0.0,
        "negative_axis_aligned_bbox_gap_over_canvas_diagonal": (
            -math.hypot(gap_y, gap_x) / diagonal
        ),
    }


def weighted_auroc(
    score: Sequence[float], label: Sequence[bool], weight: Sequence[float]
) -> float:
    values = np.asarray(score, dtype=np.float64)
    targets = np.asarray(label)
    weights = np.asarray(weight, dtype=np.float64)
    if (
        values.ndim != 1
        or targets.shape != values.shape
        or weights.shape != values.shape
        or targets.dtype != np.bool_
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("invalid weighted AUROC vectors")
    positive_total = float(weights[targets].sum())
    negative_total = float(weights[~targets].sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise ValueError("weighted AUROC requires both classes")
    order = np.argsort(-values, kind="mergesort")
    values = values[order]
    targets = targets[order]
    weights = weights[order]
    true_positive = false_positive = 0.0
    previous_tpr = previous_fpr = 0.0
    area = 0.0
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[stop] == values[start]:
            stop += 1
        group_target = targets[start:stop]
        group_weight = weights[start:stop]
        true_positive += float(group_weight[group_target].sum())
        false_positive += float(group_weight[~group_target].sum())
        tpr = true_positive / positive_total
        fpr = false_positive / negative_total
        area += (fpr - previous_fpr) * (tpr + previous_tpr) * 0.5
        previous_tpr = tpr
        previous_fpr = fpr
        start = stop
    return float(area)


def _equal_cluster_weights(cluster_ids: Sequence[str]) -> np.ndarray:
    values = np.asarray(cluster_ids, dtype=object)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return 1.0 / (len(unique) * counts[inverse].astype(np.float64))


def _top_group_ids(group_ids: Iterable[str], limit: int) -> Tuple[str, ...]:
    unique = set(group_ids)
    return tuple(sorted(unique, key=lambda value: (_rank(value), value))[:limit])


def _synthetic_group_ids(manifest_path: Path) -> Tuple[str, ...]:
    output = []
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid synthetic JSONL line {}".format(line_number)
                ) from exc
            quarantine = row.get("quarantine")
            if not isinstance(quarantine, Mapping):
                raise ValueError("synthetic row lacks quarantine state")
            if quarantine.get("status") == "retained":
                output.append(str(row.get("group_id", "")))
            elif quarantine.get("status") != "quarantined":
                raise ValueError("unsupported synthetic quarantine state")
    if not output or any(not value for value in output) or len(output) != len(set(output)):
        raise ValueError("synthetic retained group identities are invalid")
    return tuple(output)


def _historical_group_ids(
    *,
    split_manifest: Path,
    mm_archive: Optional[Path] = None,
    eccv_archive: Optional[Path] = None,
) -> Tuple[str, ...]:
    return tuple(
        sorted(
            {
                record.canonical_group_id
                for record in iter_historical_pair_records(
                    split_manifest=split_manifest,
                    split="train",
                    mm_archive=mm_archive,
                    eccv_archive=eccv_archive,
                )
            }
        )
    )


def _historical_records(
    *,
    split_manifest: Path,
    selected_groups: Sequence[str],
    pairs_per_group: int,
    mm_archive: Optional[Path] = None,
    eccv_archive: Optional[Path] = None,
) -> Tuple[TrainingPairRecord, ...]:
    selected = set(selected_groups)
    heaps: Dict[str, List[Tuple[int, str, TrainingPairRecord]]] = {
        group: [] for group in selected_groups
    }
    for record in iter_historical_pair_records(
        split_manifest=split_manifest,
        split="train",
        mm_archive=mm_archive,
        eccv_archive=eccv_archive,
    ):
        if record.canonical_group_id not in selected:
            continue
        rank = _rank(record.pair_id)
        # A min-heap over negative ranks retains the smallest positive ranks.
        item = (-rank, record.pair_id, record)
        heap = heaps[record.canonical_group_id]
        if len(heap) < pairs_per_group:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    output = []
    for group in sorted(selected_groups):
        output.extend(item[2] for item in heaps[group])
    return tuple(sorted(output, key=lambda record: record.pair_id))


def _score_records(
    records: Sequence[TrainingPairRecord], loader: LazyMaskArchiveLoader
) -> Mapping[str, Any]:
    if not records:
        raise ValueError("spatial audit selection is empty")
    features: Dict[Tuple[str, str, str], MaskSpatialFeature] = {}
    labels: List[bool] = []
    clusters: List[str] = []
    scores: Dict[str, List[float]] = {}
    shapes: Dict[str, int] = {}
    non_tight = 0

    def feature(reference: MaskMemberRef) -> MaskSpatialFeature:
        key = (
            reference.binding.logical_id,
            reference.archive_member,
            reference.threshold_rule,
        )
        if key not in features:
            features[key] = mask_spatial_feature(loader(reference))
        return features[key]

    for record in records:
        left = feature(record.fragment_a)
        right = feature(record.fragment_b)
        for item in (left, right):
            shapes["{}x{}".format(*item.shape)] = shapes.get(
                "{}x{}".format(*item.shape), 0
            ) + 1
            y0, x0, y1, x1 = item.bbox_rc_exclusive
            non_tight += int((y0, x0, y1, x1) != (0, 0, item.shape[0], item.shape[1]))
        pair_scores = spatial_scores(left, right)
        for name, value in pair_scores.items():
            scores.setdefault(name, []).append(float(value))
        labels.append(record.label)
        clusters.append(record.component_id)

    row_weight = np.full(len(records), 1.0 / len(records), dtype=np.float64)
    cluster_weight = _equal_cluster_weights(clusters)
    metrics = {}
    for name in sorted(scores):
        metrics[name] = {
            "row_auroc": weighted_auroc(scores[name], labels, row_weight),
            "equal_cluster_weight_auroc": weighted_auroc(
                scores[name], labels, cluster_weight
            ),
            "score_semantics": "larger_means_more_likely_adjacent",
        }
    return {
        "pair_count": len(records),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "cluster_count": len(set(clusters)),
        "unique_mask_count": len(features),
        "pair_selection_sha256": _selection_fingerprint(
            record.pair_id for record in records
        ),
        "canvas_shape_occurrences": dict(sorted(shapes.items())),
        "non_tight_mask_occurrence_fraction": non_tight / float(2 * len(records)),
        "coordinate_only_metrics": metrics,
    }


def _synthetic_binding(manifest_path: Path) -> ArchiveBinding:
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        first = json.loads(next(stream))
    archive = first["archive"]
    return ArchiveBinding(
        str(archive["logical_id"]), str(archive["format"]), str(archive["sha256"])
    )


def build_audit(args: argparse.Namespace) -> Mapping[str, Any]:
    sources: Dict[str, ArchiveSourceSpec] = {}
    selections: List[Tuple[str, Tuple[TrainingPairRecord, ...], Mapping[str, Any]]] = []
    input_artifacts: Dict[str, Any] = {}

    if args.synthetic_manifest is not None or args.synthetic_archive is not None:
        if args.synthetic_manifest is None or args.synthetic_archive is None:
            raise ValueError("synthetic manifest and archive must be supplied together")
        binding = _synthetic_binding(args.synthetic_manifest)
        sources[binding.logical_id] = ArchiveSourceSpec(binding, args.synthetic_archive)
        retained = _synthetic_group_ids(args.synthetic_manifest)
        selected_groups = _top_group_ids(retained, args.synthetic_groups)
        selected_set = set(selected_groups)
        records = tuple(
            record
            for record in iter_synthetic_pair_records(args.synthetic_manifest)
            if record.canonical_group_id in selected_set
        )
        selections.append(
            (
                "dunhuang_voronoi_masks_no_erode_v0_2",
                records,
                {
                    "group_policy": "first_N_by_sha256_canonical_group_id",
                    "group_limit": args.synthetic_groups,
                    "selected_group_count": len(selected_groups),
                    "selected_group_set_sha256": _selection_fingerprint(selected_groups),
                    "within_group_pair_policy": "all_explicit_unordered_pairs",
                },
            )
        )
        input_artifacts["synthetic_manifest"] = {
            "sha256": _sha256_path(args.synthetic_manifest)
        }

    historical_requested = args.mm_archive is not None or args.eccv_archive is not None
    if historical_requested and args.split_manifest is None:
        raise ValueError("historical archives require the frozen split manifest")
    if args.mm_archive is not None:
        sources[MM_CANONICAL_BINDING.logical_id] = ArchiveSourceSpec(
            MM_CANONICAL_BINDING, args.mm_archive
        )
        groups = _historical_group_ids(
            split_manifest=args.split_manifest, mm_archive=args.mm_archive
        )
        selected_groups = _top_group_ids(groups, args.historical_groups)
        records = _historical_records(
            split_manifest=args.split_manifest,
            selected_groups=selected_groups,
            pairs_per_group=args.historical_pairs_per_group,
            mm_archive=args.mm_archive,
        )
        selections.append(
            (
                "mm_augmented",
                records,
                {
                    "source_split": "train",
                    "group_policy": "first_N_by_sha256_canonical_group_id",
                    "group_limit": args.historical_groups,
                    "selected_group_count": len(selected_groups),
                    "selected_group_set_sha256": _selection_fingerprint(selected_groups),
                    "within_group_pair_policy": (
                        "first_N_by_sha256_pair_id_label_blind"
                    ),
                    "pairs_per_group_limit": args.historical_pairs_per_group,
                },
            )
        )
    if args.eccv_archive is not None:
        sources[ECCV_CANONICAL_BINDING.logical_id] = ArchiveSourceSpec(
            ECCV_CANONICAL_BINDING, args.eccv_archive
        )
        groups = _historical_group_ids(
            split_manifest=args.split_manifest, eccv_archive=args.eccv_archive
        )
        selected_groups = _top_group_ids(groups, args.historical_groups)
        records = _historical_records(
            split_manifest=args.split_manifest,
            selected_groups=selected_groups,
            pairs_per_group=args.historical_pairs_per_group,
            eccv_archive=args.eccv_archive,
        )
        selections.append(
            (
                "eccv_1113data",
                records,
                {
                    "source_split": "train",
                    "group_policy": "first_N_by_sha256_canonical_group_id",
                    "group_limit": args.historical_groups,
                    "selected_group_count": len(selected_groups),
                    "selected_group_set_sha256": _selection_fingerprint(selected_groups),
                    "within_group_pair_policy": (
                        "first_N_by_sha256_pair_id_label_blind"
                    ),
                    "pairs_per_group_limit": args.historical_pairs_per_group,
                },
            )
        )
    if historical_requested:
        input_artifacts["historical_split_manifest"] = {
            "sha256": _sha256_path(args.split_manifest)
        }
    if not sources:
        raise ValueError("at least one synthetic-training archive is required")

    datasets = {}
    with LazyMaskArchiveLoader(sources) as loader:
        for dataset_id, records, policy in selections:
            scored = dict(_score_records(records, loader))
            scored["selection_policy"] = dict(policy)
            datasets[dataset_id] = scored
        loader_receipt = loader.provenance()
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_version": AUDIT_VERSION,
        "status": "completed_coordinate_only_diagnostic",
        "scope": {
            "allowed_datasets": sorted(datasets),
            "historical_split": "train_only",
            "real_dunhuang_images_accessed": False,
            "historical_test_accessed": False,
            "models_trained_or_invoked": False,
            "runtime_paths_serialized": False,
        },
        "score_definitions": {
            "center": "negative Euclidean foreground-centroid distance / canvas diagonal",
            "bbox_overlap": "axis-aligned bounding-box intersection / union",
            "bbox_gap": "negative Euclidean axis-aligned bbox separation / canvas diagonal",
        },
        "input_artifacts": input_artifacts,
        "archive_verification": loader_receipt,
        "datasets": datasets,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", type=Path)
    parser.add_argument("--synthetic-archive", type=Path)
    parser.add_argument("--mm-archive", type=Path)
    parser.add_argument("--eccv-archive", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--synthetic-groups", type=int, default=SYNTHETIC_DEFAULT_GROUPS)
    parser.add_argument("--historical-groups", type=int, default=HISTORICAL_DEFAULT_GROUPS)
    parser.add_argument(
        "--historical-pairs-per-group",
        type=int,
        default=HISTORICAL_DEFAULT_PAIRS_PER_GROUP,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    for name in (
        "synthetic_groups",
        "historical_groups",
        "historical_pairs_per_group",
    ):
        if getattr(args, name) <= 0:
            raise ValueError("{} must be positive".format(name))
    result = build_audit(args)
    payload = json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
