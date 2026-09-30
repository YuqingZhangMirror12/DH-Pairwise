"""One-shot audit of actual Pairwise v0.2 train/validation record emission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .training_receipt import _nonportable_pointers
from .training_stream import TrainingPairRecord, iter_training_pair_records


STREAM_AUDIT_SCHEMA_VERSION = "dunhuang-pairwise-training-stream-audit/0.2"


class TrainingStreamAuditError(ValueError):
    """Raised when emitted records contradict the frozen data summary."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
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


def _frozen_validation_update(digest: Any, record: TrainingPairRecord) -> None:
    value = {
        "pair_id": record.pair_id,
        "label": record.label,
        "dataset_id": record.dataset_id,
        "group_id": record.canonical_group_id,
        "component_id": record.component_id,
        "fragment_a": record.fragment_a.archive_member,
        "fragment_b": record.fragment_b.archive_member,
    }
    digest.update(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )


def audit_training_stream(
    *,
    historical_split_path: Path,
    mm_archive: Any,
    eccv_archive: Any,
    synthetic_manifest_path: Path,
    data_summary_path: Path,
    output_path: Path,
) -> Path:
    """Traverse each legal split once and freeze actual emission counts."""

    with data_summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, Mapping) or summary.get("status") != "pass":
        raise TrainingStreamAuditError("data summary is missing or not passing")
    expected_by_split = summary.get("by_split")
    if not isinstance(expected_by_split, Mapping):
        raise TrainingStreamAuditError("data summary lacks by_split")

    split_results: Dict[str, Any] = {}
    split_components: Dict[str, set] = {}
    checks = []
    for split in ("train", "val"):
        counts: Counter = Counter()
        groups = set()
        components = set()
        by_dataset: Dict[str, Counter] = defaultdict(Counter)
        dataset_groups: Dict[str, set] = defaultdict(set)
        dataset_components: Dict[str, set] = defaultdict(set)
        directions: Counter = Counter()
        validation_digest = hashlib.sha256()
        for record in iter_training_pair_records(
            split=split,
            split_manifest=historical_split_path,
            mm_archive=mm_archive,
            eccv_archive=eccv_archive,
            synthetic_manifest=synthetic_manifest_path,
        ):
            if not isinstance(record, TrainingPairRecord):
                raise TrainingStreamAuditError("stream emitted an invalid record type")
            if type(record.label) is not bool:  # noqa: E721
                raise TrainingStreamAuditError("stream emitted a non-bool label")
            if record.split != split:
                raise TrainingStreamAuditError("stream emitted a cross-split record")
            if record.provenance.get("real_dunhuang_sealed_test") is not False:
                raise TrainingStreamAuditError("sealed-test provenance is not false")
            counts["pair_count"] += 1
            counts["positive_count"] += int(record.label)
            counts["negative_count"] += int(not record.label)
            groups.add(record.canonical_group_id)
            components.add(record.component_id)
            dataset = by_dataset[record.dataset_id]
            dataset["pair_count"] += 1
            dataset["positive_count"] += int(record.label)
            dataset["negative_count"] += int(not record.label)
            dataset_groups[record.dataset_id].add(record.canonical_group_id)
            dataset_components[record.dataset_id].add(record.component_id)
            directions[str(record.direction_b_wrt_a)] += 1
            if split == "val":
                _frozen_validation_update(validation_digest, record)
        counts["emitted_group_count"] = len(groups)
        counts["emitted_component_count"] = len(components)
        split_components[split] = components
        expected = expected_by_split.get(split)
        if not isinstance(expected, Mapping):
            raise TrainingStreamAuditError("missing expected split counts")
        for field in ("usable_pair_count", "positive_count", "negative_count"):
            observed_field = "pair_count" if field == "usable_pair_count" else field
            observed = int(counts[observed_field])
            target = int(expected.get(field, -1))
            status = "pass" if observed == target else "fail"
            checks.append(
                {
                    "check_id": "{}_{}".format(split, field),
                    "observed": observed,
                    "expected": target,
                    "status": status,
                }
            )
            if status != "pass":
                raise TrainingStreamAuditError(
                    "{} {} count mismatch".format(split, field)
                )
        split_results[split] = {
            "counts": dict(counts),
            "assigned_but_zero_emission_group_count": int(expected["group_count"])
            - len(groups),
            "assigned_but_zero_emission_component_count": int(
                expected["component_count"]
            )
            - len(components),
            "by_dataset": {
                dataset_id: {
                    **dict(dataset_counts),
                    "emitted_group_count": len(dataset_groups[dataset_id]),
                    "emitted_component_count": len(dataset_components[dataset_id]),
                }
                for dataset_id, dataset_counts in sorted(by_dataset.items())
            },
            "direction_b_wrt_a_counts": dict(sorted(directions.items())),
            "frozen_order_sha256": (
                validation_digest.hexdigest() if split == "val" else None
            ),
        }

    component_overlap = len(
        split_components["train"].intersection(split_components["val"])
    )
    checks.append(
        {
            "check_id": "actual_train_val_component_overlap",
            "observed": component_overlap,
            "expected": 0,
            "status": "pass" if component_overlap == 0 else "fail",
        }
    )
    if component_overlap:
        raise TrainingStreamAuditError("actual train/val components overlap")
    result: Dict[str, Any] = {
        "schema_version": STREAM_AUDIT_SCHEMA_VERSION,
        "status": "pass",
        "scope": "actual_metadata_emission_no_mask_pixels_loaded",
        "inputs": {
            "historical_split": {
                "filename": historical_split_path.name,
                "sha256": _sha256_path(historical_split_path),
            },
            "synthetic_manifest": {
                "filename": synthetic_manifest_path.name,
                "sha256": _sha256_path(synthetic_manifest_path),
            },
            "data_summary": {
                "filename": data_summary_path.name,
                "sha256": _sha256_path(data_summary_path),
            },
        },
        "splits": split_results,
        "checks": checks,
        "real_dunhuang_sealed_test_record_count": 0,
        "mask_pixels_loaded": False,
        "portable_outputs_contain_absolute_local_paths": False,
    }
    failures = _nonportable_pointers(result)
    if failures:
        raise TrainingStreamAuditError(
            "stream audit contains non-portable strings: {}".format(failures[:5])
        )
    _atomic_json(output_path, result)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-split", type=Path, required=True)
    parser.add_argument("--mm-archive", type=Path, required=True)
    parser.add_argument("--eccv-archive", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--data-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_training_stream(
        historical_split_path=args.historical_split.resolve(strict=True),
        mm_archive=args.mm_archive.resolve(strict=True),
        eccv_archive=args.eccv_archive.resolve(strict=True),
        synthetic_manifest_path=args.synthetic_manifest.resolve(strict=True),
        data_summary_path=args.data_summary.resolve(strict=True),
        output_path=args.output.resolve(),
    )
    print(json.dumps({"stream_audit": result.name}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "STREAM_AUDIT_SCHEMA_VERSION",
    "TrainingStreamAuditError",
    "audit_training_stream",
]
