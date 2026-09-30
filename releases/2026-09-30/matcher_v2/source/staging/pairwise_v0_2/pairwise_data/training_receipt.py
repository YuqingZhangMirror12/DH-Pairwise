"""Build a portable Pairwise v0.2 train/validation split and data receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    SYNTHETIC_MANIFEST_SCHEMA_VERSION,
)


RECEIPT_SCHEMA_VERSION = "dunhuang-pairwise-training-split-receipt/0.2"
SUMMARY_SCHEMA_VERSION = "dunhuang-pairwise-training-data-summary/0.2"
UPSTREAM_SPLIT_SCHEMA_VERSION = "pairwise-real-data-gate/0.1"
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class TrainingReceiptError(ValueError):
    """Raised when upstream artifacts cannot prove a safe training split."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> Dict[str, Any]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise TrainingReceiptError("{} root must be an object".format(path.name))
    return value


def _nonportable_pointers(value: Any, pointer: str = "") -> List[str]:
    failures: List[str] = []
    if isinstance(value, str):
        if (
            value.startswith("/")
            or value.startswith("file://")
            or WINDOWS_ABSOLUTE_RE.match(value)
        ):
            failures.append(pointer or "/")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            failures.extend(_nonportable_pointers(item, pointer + "/" + str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            failures.extend(_nonportable_pointers(item, pointer + "/" + str(index)))
    return failures


def _historical_split_audit(split_payload: Mapping[str, Any]) -> Dict[str, Any]:
    if split_payload.get("schema_version") != UPSTREAM_SPLIT_SCHEMA_VERSION:
        raise TrainingReceiptError("unsupported historical split schema")
    assignments = split_payload.get("assignments")
    if not isinstance(assignments, Mapping):
        raise TrainingReceiptError("historical split requires assignments")
    groups = assignments.get("groups")
    components = assignments.get("components")
    group_components = assignments.get("group_components")
    if not all(
        isinstance(value, Mapping) for value in (groups, components, group_components)
    ):
        raise TrainingReceiptError("historical split assignment maps are missing")
    if set(groups) != set(group_components):
        raise TrainingReceiptError("historical group/component keys differ")

    component_sets: Dict[str, set] = {name: set() for name in ("train", "val", "test")}
    group_sets: Dict[str, set] = {name: set() for name in ("train", "val", "test")}
    prefix_by_split: Dict[str, Counter] = {
        name: Counter() for name in ("train", "val", "test")
    }
    for group_id, split_value in groups.items():
        split = str(split_value)
        if split not in group_sets:
            raise TrainingReceiptError("invalid historical split name")
        component_id = str(group_components[group_id])
        if components.get(component_id) != split:
            raise TrainingReceiptError("group/component split mismatch")
        if group_id.startswith("mm/base/"):
            expected_prefix = "mm/source/"
            prefix = "mm_source"
        elif group_id.startswith(("eccv/2x2/", "eccv/2x2-negative/")):
            expected_prefix = "eccv/signature/"
            prefix = "eccv_signature"
        else:
            raise TrainingReceiptError("unexpected historical group id")
        if not component_id.startswith(expected_prefix):
            raise TrainingReceiptError("historical component namespace mismatch")
        component_sets[split].add(component_id)
        group_sets[split].add(str(group_id))
        prefix_by_split[split][prefix] += 1

    overlaps = {}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps["{}_{}_component_overlap".format(first, second)] = len(
            component_sets[first].intersection(component_sets[second])
        )
        overlaps["{}_{}_group_overlap".format(first, second)] = len(
            group_sets[first].intersection(group_sets[second])
        )
    if any(overlaps.values()):
        raise TrainingReceiptError("historical split leakage overlap is nonzero")
    return {
        "candidate_id": split_payload.get("candidate_id"),
        "seed": split_payload.get("seed"),
        "ratios": split_payload.get("ratios"),
        "upstream_status": split_payload.get("status"),
        "upstream_authorization": split_payload.get("authorization"),
        "assignment_counts": {
            split: {
                "group_count": len(group_sets[split]),
                "component_count": len(component_sets[split]),
                "mm_group_count": prefix_by_split[split]["mm_source"],
                "eccv_group_count": prefix_by_split[split]["eccv_signature"],
            }
            for split in ("train", "val", "test")
        },
        "overlap_counts": overlaps,
    }


def _synthetic_audit(manifest_path: Path) -> Dict[str, Any]:
    counts: Counter = Counter()
    fragment_groups: Counter = Counter()
    archive_binding: Optional[Tuple[str, str, str]] = None
    group_ids = set()
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                group = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingReceiptError(
                    "invalid synthetic JSONL line {}".format(line_number)
                ) from exc
            if not isinstance(group, Mapping):
                raise TrainingReceiptError("synthetic rows must be objects")
            if group.get("schema_version") != SYNTHETIC_MANIFEST_SCHEMA_VERSION:
                raise TrainingReceiptError("unsupported synthetic group schema")
            group_id = str(group.get("group_id", ""))
            if not group_id or group_id in group_ids:
                raise TrainingReceiptError("missing or duplicate synthetic group")
            group_ids.add(group_id)
            archive = group.get("archive")
            if not isinstance(archive, Mapping):
                raise TrainingReceiptError("synthetic group lacks archive binding")
            observed_binding = (
                str(archive.get("logical_id", "")),
                str(archive.get("format", "")),
                str(archive.get("sha256", "")),
            )
            if archive_binding is None:
                archive_binding = observed_binding
            elif archive_binding != observed_binding:
                raise TrainingReceiptError("synthetic archive binding changed")
            counts["group_count"] += 1
            quarantine = group.get("quarantine", {})
            status = (
                quarantine.get("status") if isinstance(quarantine, Mapping) else None
            )
            if status == "quarantined":
                counts["quarantine_group_count"] += 1
                continue
            if status != "retained":
                raise TrainingReceiptError("invalid synthetic quarantine status")
            counts["retained_group_count"] += 1
            fragment_count = group.get("fragment_count")
            if type(fragment_count) is not int or not 2 <= fragment_count <= 5:  # noqa: E721
                raise TrainingReceiptError("invalid synthetic fragment count")
            fragment_groups[int(fragment_count)] += 1
            measurements = group.get("pair_measurements")
            if not isinstance(measurements, list):
                raise TrainingReceiptError("synthetic pair measurements missing")
            expected_pair_count = fragment_count * (fragment_count - 1) // 2
            if len(measurements) != expected_pair_count:
                raise TrainingReceiptError("synthetic pair measurement count mismatch")
            counts["pair_count"] += len(measurements)
            for measurement in measurements:
                if not isinstance(measurement, Mapping):
                    raise TrainingReceiptError("measurement must be an object")
                label = measurement.get("is_neighbor")
                if type(label) is not bool:  # noqa: E721
                    raise TrainingReceiptError("synthetic labels must be strict bool")
                counts["positive_count" if label else "negative_count"] += 1
                overlap = measurement.get("legacy_dilated_overlap_pixels")
                if type(overlap) is not int or overlap < 0:  # noqa: E721
                    raise TrainingReceiptError(
                        "synthetic overlap must be a non-negative int"
                    )
                if not label and overlap > 0:
                    counts["static_hard_negative_candidate_count"] += 1
    if archive_binding is None:
        raise TrainingReceiptError("synthetic manifest is empty")
    return {
        "archive": {
            "logical_id": archive_binding[0],
            "format": archive_binding[1],
            "sha256": archive_binding[2],
        },
        "counts": dict(counts),
        "retained_group_count_by_fragment_count": {
            str(key): fragment_groups[key] for key in sorted(fragment_groups)
        },
        "source_lineage_status": "unavailable_training_only",
    }


def _dataset_statistics(split_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    statistics = split_payload.get("statistics")
    if not isinstance(statistics, Mapping):
        raise TrainingReceiptError("historical split statistics are missing")
    value = statistics.get("by_dataset_and_split")
    if not isinstance(value, Mapping):
        raise TrainingReceiptError("historical dataset statistics are missing")
    return value


def _add_counts(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    frozen_rows = tuple(rows)
    fields = (
        "group_count",
        "component_count",
        "usable_pair_count",
        "positive_count",
        "negative_count",
    )
    return {
        field: sum(int(row.get(field, 0)) for row in frozen_rows) for field in fields
    }


def build_training_receipt(
    *,
    historical_split_path: Path,
    synthetic_manifest_path: Path,
    synthetic_summary_path: Path,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Create deterministic portable receipt and summary JSON artifacts."""

    historical_split_path = historical_split_path.resolve(strict=True)
    synthetic_manifest_path = synthetic_manifest_path.resolve(strict=True)
    synthetic_summary_path = synthetic_summary_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    split_payload = _load_json(historical_split_path)
    synthetic_summary = _load_json(synthetic_summary_path)
    historical_audit = _historical_split_audit(split_payload)
    synthetic_audit = _synthetic_audit(synthetic_manifest_path)
    summary_counts = synthetic_summary.get("counts")
    if not isinstance(summary_counts, Mapping):
        raise TrainingReceiptError("synthetic summary counts are missing")
    for name in ("group_count", "retained_group_count", "quarantine_group_count"):
        if int(summary_counts.get(name, -1)) != int(
            synthetic_audit["counts"].get(name, -2)
        ):
            raise TrainingReceiptError(
                "synthetic summary/manifest {} mismatch".format(name)
            )
    if int(synthetic_audit["counts"].get("quarantine_group_count", -1)) != 8:
        raise TrainingReceiptError(
            "expected exactly eight quarantined synthetic groups"
        )

    historical_stats = _dataset_statistics(split_payload)
    datasets: Dict[str, Dict[str, Any]] = {}
    for dataset_id in ("mm_augmented", "eccv_1113data"):
        raw = historical_stats.get(dataset_id)
        if not isinstance(raw, Mapping):
            raise TrainingReceiptError(
                "missing historical statistics for {}".format(dataset_id)
            )
        datasets[dataset_id] = {
            split: dict(raw[split]) for split in ("train", "val", "test")
        }
    synthetic_counts = synthetic_audit["counts"]
    datasets["dunhuang_voronoi_masks_no_erode_v0_2"] = {
        "train": {
            "group_count": int(synthetic_counts["retained_group_count"]),
            "component_count": int(synthetic_counts["retained_group_count"]),
            "usable_pair_count": int(synthetic_counts["pair_count"]),
            "positive_count": int(synthetic_counts["positive_count"]),
            "negative_count": int(synthetic_counts["negative_count"]),
            "static_hard_negative_candidate_count": int(
                synthetic_counts["static_hard_negative_candidate_count"]
            ),
        },
        "val": {
            "group_count": 0,
            "component_count": 0,
            "usable_pair_count": 0,
            "positive_count": 0,
            "negative_count": 0,
        },
        "test": {
            "group_count": 0,
            "component_count": 0,
            "usable_pair_count": 0,
            "positive_count": 0,
            "negative_count": 0,
        },
    }
    by_split = {
        split: _add_counts(dataset[split] for dataset in datasets.values())
        for split in ("train", "val", "test")
    }
    data_summary: Dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "pass",
        "task_scope": "mask_only_binary_pairwise_known_orientation_no_rotation_search",
        "pair_rows_materialized": False,
        "datasets": datasets,
        "by_split": by_split,
        "count_semantics": {
            "group_count": "upstream_assigned_groups_including_zero_usable_pair_groups",
            "component_count": "upstream_assigned_components_including_zero_usable_pair_components",
            "usable_pair_count": "records_expected_from_the_pair_stream",
            "positive_count": "strict_bool_true_records_expected_from_the_pair_stream",
            "negative_count": "strict_bool_false_records_expected_from_the_pair_stream",
            "actual_emitted_group_and_component_counts": "see_stream_audit_json",
        },
        "synthetic": synthetic_audit,
        "held_out": {
            "historical_v0_1_test": "withheld_from_v0_2_train_and_validation_streams",
            "real_dunhuang_external_test": "sealed_no_training_no_threshold_tuning_no_model_selection",
        },
    }
    checks = [
        {
            "check_id": "historical_train_val_component_overlap",
            "observed": historical_audit["overlap_counts"][
                "train_val_component_overlap"
            ],
            "expected": 0,
            "status": "pass",
        },
        {
            "check_id": "historical_train_val_group_overlap",
            "observed": historical_audit["overlap_counts"]["train_val_group_overlap"],
            "expected": 0,
            "status": "pass",
        },
        {
            "check_id": "synthetic_quarantine_excluded",
            "observed": synthetic_counts["quarantine_group_count"],
            "expected": 8,
            "status": "pass",
        },
        {
            "check_id": "synthetic_validation_exposure",
            "observed": 0,
            "expected": 0,
            "status": "pass",
        },
        {
            "check_id": "real_dunhuang_sealed_test_references",
            "observed": 0,
            "expected": 0,
            "status": "pass",
        },
    ]
    receipt: Dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "authorized_ready_for_model_training",
        "authorization": {
            "status": "authorized_for_model_training_by_current_user_instruction",
            "scope": "MM_ECCV_frozen_train_split_plus_canonical_5k_training_only_masks",
            "allowed": [
                "model_training_on_train",
                "model_selection_and_threshold_tuning_on_frozen_validation",
            ],
            "prohibited": [
                "training_on_historical_v0_1_test",
                "training_on_real_dunhuang_sealed_test",
                "threshold_tuning_on_real_dunhuang_sealed_test",
                "model_selection_on_real_dunhuang_sealed_test",
            ],
        },
        "task_contract": {
            "input_modality": "binary_fragment_masks",
            "target": "strict_bool_can_join",
            "orientation": "known_no_rotation_search",
            "archive_access": "lazy_read_only_members_no_extraction",
            "local_patch_generation": "outside_data_layer_geometry_adapter",
        },
        "split_policy": {
            "historical": "reuse_v0_1_candidate_assignments_unchanged",
            "mm_unit": "source_document_component",
            "eccv_unit": "whole_group_signature_component",
            "synthetic": "training_only_due_missing_source_lineage",
            "validation": "frozen_source_order_no_balance_no_hard_negative_mining",
            "train_sampler": "configurable_bounded_balanced_reservoir_with_hard_negatives",
        },
        "upstream_artifacts": {
            "historical_split": _artifact(historical_split_path),
            "synthetic_manifest": _artifact(synthetic_manifest_path),
            "synthetic_summary": _artifact(synthetic_summary_path),
            "archives": [
                {
                    "logical_id": MM_CANONICAL_BINDING.logical_id,
                    "format": MM_CANONICAL_BINDING.archive_format,
                    "sha256": MM_CANONICAL_BINDING.sha256,
                },
                {
                    "logical_id": ECCV_CANONICAL_BINDING.logical_id,
                    "format": ECCV_CANONICAL_BINDING.archive_format,
                    "sha256": ECCV_CANONICAL_BINDING.sha256,
                },
                synthetic_audit["archive"],
            ],
        },
        "historical_split_audit": historical_audit,
        "synthetic_split_audit": synthetic_audit,
        "checks": checks,
        "counts": by_split,
        "count_semantics": data_summary["count_semantics"],
        "real_dunhuang_external_test": {
            "referenced_by_training_receipt": False,
            "allowed_use": "one_time_external_evaluation_after_model_and_threshold_freeze",
        },
        "portable_outputs_contain_absolute_local_paths": False,
    }
    for value, name in ((receipt, "receipt"), (data_summary, "summary")):
        failures = _nonportable_pointers(value)
        if failures:
            raise TrainingReceiptError(
                "non-portable strings in {}: {}".format(name, failures[:5])
            )
    receipt_path = output_dir / "split_receipt.json"
    summary_path = output_dir / "data_summary.json"
    _atomic_write(receipt_path, _json_bytes(receipt))
    _atomic_write(summary_path, _json_bytes(data_summary))
    return receipt_path, summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-split", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    receipt, summary = build_training_receipt(
        historical_split_path=args.historical_split,
        synthetic_manifest_path=args.synthetic_manifest,
        synthetic_summary_path=args.synthetic_summary,
        output_dir=args.output_dir,
    )
    print(
        json.dumps({"receipt": receipt.name, "summary": summary.name}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "TrainingReceiptError",
    "build_training_receipt",
]
