"""Read-only, streaming manifest builder for Pairwise v0.2 Voronoi masks.

The builder opens the source ZIP in read-only mode, never extracts members,
and keeps decoded pixels for at most one 2--5 fragment group in memory.  ZIP
central-directory metadata and compact aggregate counters are retained so that
the resulting JSONL can be deterministic and independently audited.

The legacy adjacency rule is reproduced from ``src/final_dataset_generator.py``
inside the archive: for every stable ``i < j`` pair, dilate fragment ``i`` with
a 3x3 all-ones kernel for 10 iterations, intersect with fragment ``j``, and
declare an edge when at least 30 pixels overlap.  Raw (undilated) overlap is
recorded separately and never changes the adjacency label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


SCHEMA_VERSION = "dunhuang-pairwise-synthetic-group-manifest/0.2"
BUILDER_VERSION = "synthetic-manifest-builder/0.2"
LEGACY_NEIGHBOR_RULE_ID = "legacy_neighbor_rule_v1"
MASK_PREFIX = "output/voronoi_masks/"
LABEL_PREFIX = "output/datasets/"
MASK_MEMBER_RE = re.compile(
    r"^output/voronoi_masks/([^/]+)/([^/]+)/([^/]+)/([^/]+)\.png$",
    re.IGNORECASE,
)
LABEL_MEMBER_RE = re.compile(
    r"^output/datasets/([^/]+)/([^/]+)/([^/]+)/label\.csv$",
    re.IGNORECASE,
)
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SyntheticManifestError(RuntimeError):
    """Raised for malformed archives or unsafe output contracts."""


@dataclass(frozen=True)
class ArchiveExpectations:
    """Optional count gates used to freeze a known archive revision."""

    group_count: Optional[int] = None
    mask_member_count: Optional[int] = None
    no_erode_group_count: Optional[int] = None
    disconnected_group_count: Optional[int] = None
    quarantine_group_count: Optional[int] = None
    raw_overlap_group_count: Optional[int] = None
    legacy_label_group_count: Optional[int] = None
    legacy_label_mismatch_count: Optional[int] = 0

    @classmethod
    def dunhuang_datas_v0_2(cls) -> "ArchiveExpectations":
        return cls(
            group_count=5_000,
            mask_member_count=15_000,
            no_erode_group_count=5_000,
            disconnected_group_count=8,
            quarantine_group_count=8,
            raw_overlap_group_count=646,
            legacy_label_group_count=2_000,
            legacy_label_mismatch_count=0,
        )


@dataclass(frozen=True)
class BuildResult:
    manifest_path: Path
    summary_path: Path
    local_receipt_path: Path
    summary: Mapping[str, Any]


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_stream(stream: BinaryIO, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _artifact(path: Path) -> Dict[str, Any]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _portable_string(value: str) -> bool:
    return not (
        value.startswith("/")
        or value.startswith("file://")
        or WINDOWS_ABSOLUTE_RE.match(value)
    )


def _find_nonportable_strings(value: Any, pointer: str = "") -> List[str]:
    failures: List[str] = []
    if isinstance(value, str):
        if not _portable_string(value):
            failures.append(pointer or "/")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            failures.extend(_find_nonportable_strings(item, pointer + "/" + str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for index, item in enumerate(value):
            failures.extend(_find_nonportable_strings(item, pointer + "/" + str(index)))
    return failures


def _safe_zip_member(name: str) -> bool:
    member = PurePosixPath(name)
    return (
        not member.is_absolute()
        and ".." not in member.parts
        and "\\" not in name
        and not WINDOWS_ABSOLUTE_RE.match(name)
    )


def _group_sort_key(key: Tuple[str, str, str]) -> Tuple[Any, ...]:
    family, erosion, group_id = key
    if group_id.isdigit():
        return family, erosion, 0, int(group_id)
    return family, erosion, 1, group_id


def _fragment_sort_key(info: zipfile.ZipInfo) -> Tuple[Any, ...]:
    match = MASK_MEMBER_RE.fullmatch(info.filename)
    assert match is not None
    fragment_id = match.group(4)
    if fragment_id.isdigit():
        return 0, int(fragment_id)
    return 1, fragment_id


def _member_sha256(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    with archive.open(info, "r") as stream:
        return _sha256_stream(stream)


def _decode_mask(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo
) -> Tuple[np.ndarray, Dict[str, Any]]:
    with archive.open(info, "r") as stream:
        with Image.open(stream) as image:
            image.load()
            source_mode = image.mode
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.ndim != 2 or gray.size == 0:
        raise SyntheticManifestError("invalid mask image: {}".format(info.filename))
    mask = np.ascontiguousarray(gray > 127)
    metadata = {
        "archive_member": info.filename,
        "content_sha256": _member_sha256(archive, info),
        "member_bytes": info.file_size,
        "zip_crc32": "{:08x}".format(info.CRC),
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "source_mode": source_mode,
        "threshold_rule": "grayscale_uint8_gt_127",
        "foreground_pixels": int(np.count_nonzero(mask)),
    }
    return mask, metadata


def _parse_label_edges(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo
) -> Tuple[List[List[int]], str]:
    digest = _member_sha256(archive, info)
    edges = set()
    with archive.open(info, "r") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            reader = csv.DictReader(text)
            required = {"id", "neighbors"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise SyntheticManifestError(
                    "label.csv missing id/neighbors: {}".format(info.filename)
                )
            for row in reader:
                fragment_id = int(row["id"])
                neighbors = (row.get("neighbors") or "").strip()
                if not neighbors:
                    continue
                for item in neighbors.split(";"):
                    neighbor_id = int(item)
                    if fragment_id == neighbor_id:
                        raise SyntheticManifestError(
                            "self neighbor in {}".format(info.filename)
                        )
                    edges.add(tuple(sorted((fragment_id, neighbor_id))))
    return [list(edge) for edge in sorted(edges)], digest


def _connected_components(
    fragment_ids: Sequence[int], edges: Sequence[Sequence[int]]
) -> List[List[int]]:
    adjacency = {fragment_id: set() for fragment_id in fragment_ids}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    components: List[List[int]] = []
    unseen = set(fragment_ids)
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component: List[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda item: (item[0], len(item)))


def _pair_measurements(
    fragment_ids: Sequence[int], masks: Sequence[np.ndarray]
) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
    kernel = np.ones((3, 3), dtype=np.uint8)
    measurements: List[Dict[str, Any]] = []
    edges: List[List[int]] = []
    for left_index in range(len(masks)):
        dilated = cv2.dilate(
            masks[left_index].astype(np.uint8),
            kernel,
            iterations=10,
        )
        for right_index in range(left_index + 1, len(masks)):
            raw_overlap = int(np.count_nonzero(masks[left_index] & masks[right_index]))
            legacy_overlap = int(np.count_nonzero(dilated & masks[right_index]))
            is_neighbor = legacy_overlap >= 30
            first = int(fragment_ids[left_index])
            second = int(fragment_ids[right_index])
            measurements.append(
                {
                    "fragment_a": first,
                    "fragment_b": second,
                    "raw_overlap_pixels": raw_overlap,
                    "legacy_dilated_overlap_pixels": legacy_overlap,
                    "is_neighbor": is_neighbor,
                }
            )
            if is_neighbor:
                edges.append([first, second])
    return measurements, edges


def _expectation_checks(
    actual: Mapping[str, int], expectations: ArchiveExpectations
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for field_name in expectations.__dataclass_fields__:
        expected = getattr(expectations, field_name)
        if expected is None:
            continue
        observed = int(actual[field_name])
        checks.append(
            {
                "check_id": field_name,
                "expected": int(expected),
                "observed": observed,
                "status": "pass" if observed == expected else "fail",
            }
        )
    return checks


def _write_manifest_line(stream: BinaryIO, record: Mapping[str, Any]) -> None:
    failures = _find_nonportable_strings(record)
    if failures:
        raise SyntheticManifestError(
            "non-portable path leaked into manifest: {}".format(failures[:5])
        )
    stream.write(_json_bytes(record))


def build_synthetic_manifest(
    *,
    archive_path: Path,
    output_dir: Path,
    dataset_id: str = "dunhuang_voronoi_masks_no_erode_v0_2",
    logical_archive_id: str = "local_asset://dunhuang_datas_zip",
    expectations: Optional[ArchiveExpectations] = None,
) -> BuildResult:
    """Build a portable JSONL manifest plus summary and local path receipt."""

    archive_path = Path(archive_path).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    if not zipfile.is_zipfile(archive_path):
        raise SyntheticManifestError("not a valid ZIP archive: {}".format(archive_path))
    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be empty")
    if not _portable_string(logical_archive_id) or "://" not in logical_archive_id:
        raise ValueError("logical_archive_id must be a portable logical locator")
    expectations = expectations or ArchiveExpectations.dunhuang_datas_v0_2()

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "synthetic_groups.jsonl"
    summary_path = output_dir / "synthetic_summary.json"
    local_receipt_path = output_dir / "synthetic_local_path_receipt.json"

    archive_hash_observations = [
        _sha256_path(archive_path),
        _sha256_path(archive_path),
    ]
    if len(set(archive_hash_observations)) != 1:
        raise SyntheticManifestError(
            "archive SHA-256 is unstable across two complete reads: {}".format(
                archive_hash_observations
            )
        )
    archive_sha256 = archive_hash_observations[0]
    archive_descriptor = {
        "logical_id": logical_archive_id,
        "filename": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": archive_sha256,
        "format": "zip",
        "member_prefix": MASK_PREFIX,
    }

    groups: Dict[Tuple[str, str, str], List[zipfile.ZipInfo]] = defaultdict(list)
    labels: Dict[Tuple[str, str, str], zipfile.ZipInfo] = {}
    unsafe_members: List[str] = []
    duplicate_label_keys: List[Tuple[str, str, str]] = []
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        for info in archive.infolist():
            if not _safe_zip_member(info.filename):
                unsafe_members.append(info.filename)
                continue
            mask_match = MASK_MEMBER_RE.fullmatch(info.filename)
            if mask_match is not None and not info.is_dir():
                key = tuple(mask_match.group(index) for index in (1, 2, 3))
                groups[key].append(info)
                continue
            label_match = LABEL_MEMBER_RE.fullmatch(info.filename)
            if label_match is not None and not info.is_dir():
                key = tuple(label_match.group(index) for index in (1, 2, 3))
                if key in labels:
                    duplicate_label_keys.append(key)
                labels[key] = info

        if unsafe_members:
            raise SyntheticManifestError(
                "unsafe ZIP members detected: {}".format(unsafe_members[:5])
            )
        if duplicate_label_keys:
            raise SyntheticManifestError(
                "duplicate label members detected: {}".format(duplicate_label_keys[:5])
            )
        if not groups:
            raise SyntheticManifestError(
                "no masks found beneath {}".format(MASK_PREFIX)
            )

        counters: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        fragment_counts: Counter[int] = Counter()
        quarantine_reasons: Counter[str] = Counter()
        overlap_pixel_histogram: Counter[int] = Counter()
        disconnected_group_ids: List[str] = []
        label_mismatch_examples: List[Dict[str, Any]] = []

        fd, tmp_manifest_name = tempfile.mkstemp(
            prefix="." + manifest_path.name + ".", dir=str(output_dir)
        )
        try:
            with os.fdopen(fd, "wb") as manifest_stream:
                for key in sorted(groups, key=_group_sort_key):
                    family, erosion_profile, source_group_id = key
                    infos = sorted(groups[key], key=_fragment_sort_key)
                    parsed_fragment_ids: List[int] = []
                    masks: List[np.ndarray] = []
                    members: List[Dict[str, Any]] = []
                    reasons: List[str] = []

                    for info in infos:
                        match = MASK_MEMBER_RE.fullmatch(info.filename)
                        assert match is not None
                        fragment_surface = match.group(4)
                        if not fragment_surface.isdigit():
                            reasons.append("non_numeric_fragment_id")
                            fragment_id = len(parsed_fragment_ids)
                        else:
                            fragment_id = int(fragment_surface)
                        mask, metadata = _decode_mask(archive, info)
                        metadata["fragment_id"] = fragment_id
                        parsed_fragment_ids.append(fragment_id)
                        masks.append(mask)
                        members.append(metadata)

                    counters["group_count"] += 1
                    counters["mask_member_count"] += len(masks)
                    family_counts[family] += 1
                    fragment_counts[len(masks)] += 1
                    if erosion_profile == "no_erode":
                        counters["no_erode_group_count"] += 1
                    else:
                        reasons.append("not_no_erode")
                    if not 2 <= len(masks) <= 5:
                        reasons.append("fragment_count_outside_2_to_5")
                    if len(set(parsed_fragment_ids)) != len(parsed_fragment_ids):
                        reasons.append("duplicate_fragment_id")
                    if sorted(parsed_fragment_ids) != list(range(len(masks))):
                        reasons.append("fragment_ids_not_contiguous_zero_based")
                    dimensions = {(mask.shape[1], mask.shape[0]) for mask in masks}
                    if len(dimensions) != 1:
                        reasons.append("inconsistent_group_dimensions")

                    measurements, adjacency_edges = _pair_measurements(
                        parsed_fragment_ids, masks
                    )
                    raw_overlaps = [item["raw_overlap_pixels"] for item in measurements]
                    nonzero_raw = [value for value in raw_overlaps if value > 0]
                    if nonzero_raw:
                        counters["raw_overlap_group_count"] += 1
                        counters["raw_overlap_pair_count"] += len(nonzero_raw)
                        counters["raw_overlap_pixels_total"] += sum(nonzero_raw)
                        counters["raw_overlap_pixels_max"] = max(
                            counters["raw_overlap_pixels_max"], max(nonzero_raw)
                        )
                        for value in nonzero_raw:
                            overlap_pixel_histogram[value] += 1

                    components = _connected_components(
                        parsed_fragment_ids, adjacency_edges
                    )
                    connected = len(components) == 1
                    group_id = "{}/{}/{}/{}".format(
                        dataset_id, family, erosion_profile, source_group_id
                    )
                    if not connected:
                        counters["disconnected_group_count"] += 1
                        reasons.append("disconnected_legacy_adjacency_graph")
                        disconnected_group_ids.append(group_id)

                    label_info = labels.get(key)
                    label_regression: Dict[str, Any]
                    if label_info is None:
                        label_regression = {
                            "available": False,
                            "label_member": None,
                            "label_content_sha256": None,
                            "declared_adjacency_edges": None,
                            "matches_legacy_rule": None,
                        }
                    else:
                        declared_edges, label_sha256 = _parse_label_edges(
                            archive, label_info
                        )
                        matches = declared_edges == adjacency_edges
                        counters["legacy_label_group_count"] += 1
                        if matches:
                            counters["legacy_label_match_count"] += 1
                        else:
                            counters["legacy_label_mismatch_count"] += 1
                            reasons.append("legacy_label_regression_mismatch")
                            if len(label_mismatch_examples) < 20:
                                label_mismatch_examples.append(
                                    {
                                        "group_id": group_id,
                                        "computed": adjacency_edges,
                                        "declared": declared_edges,
                                    }
                                )
                        label_regression = {
                            "available": True,
                            "label_member": label_info.filename,
                            "label_content_sha256": label_sha256,
                            "declared_adjacency_edges": declared_edges,
                            "matches_legacy_rule": matches,
                        }

                    reasons = sorted(set(reasons))
                    if reasons:
                        counters["quarantine_group_count"] += 1
                        for reason in reasons:
                            quarantine_reasons[reason] += 1
                    else:
                        counters["retained_group_count"] += 1

                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "builder_version": BUILDER_VERSION,
                        "dataset_id": dataset_id,
                        "archive": archive_descriptor,
                        "group_id": group_id,
                        "archive_group_member_prefix": "{}/{}/{}/{}/".format(
                            MASK_PREFIX.rstrip("/"),
                            family,
                            erosion_profile,
                            source_group_id,
                        ),
                        "source_group_id": source_group_id,
                        "generator_family": family,
                        "erosion_profile": erosion_profile,
                        "no_erode": erosion_profile == "no_erode",
                        "fragment_count": len(masks),
                        "members": members,
                        "legacy_neighbor_rule": {
                            "rule_id": LEGACY_NEIGHBOR_RULE_ID,
                            "stable_pair_order": "numeric_fragment_id_i_lt_j",
                            "dilate_fragment": "i",
                            "kernel_shape": [3, 3],
                            "kernel_values": "all_ones",
                            "iterations": 10,
                            "minimum_dilated_overlap_pixels_inclusive": 30,
                        },
                        "pair_measurements": measurements,
                        "adjacency_edges": adjacency_edges,
                        "raw_overlap": {
                            "nonzero_pair_count": len(nonzero_raw),
                            "total_pairwise_pixels": sum(nonzero_raw),
                            "maximum_pairwise_pixels": max(nonzero_raw, default=0),
                        },
                        "connectivity": {
                            "rule_id": LEGACY_NEIGHBOR_RULE_ID,
                            "connected": connected,
                            "component_count": len(components),
                            "components": components,
                        },
                        "legacy_label_regression": label_regression,
                        "quarantine": {
                            "status": "quarantined" if reasons else "retained",
                            "reasons": reasons,
                        },
                    }
                    _write_manifest_line(manifest_stream, record)
                manifest_stream.flush()
                os.fsync(manifest_stream.fileno())
            os.replace(tmp_manifest_name, manifest_path)
        except BaseException:
            try:
                os.unlink(tmp_manifest_name)
            except FileNotFoundError:
                pass
            raise

    actual = {
        "group_count": int(counters["group_count"]),
        "mask_member_count": int(counters["mask_member_count"]),
        "no_erode_group_count": int(counters["no_erode_group_count"]),
        "disconnected_group_count": int(counters["disconnected_group_count"]),
        "quarantine_group_count": int(counters["quarantine_group_count"]),
        "raw_overlap_group_count": int(counters["raw_overlap_group_count"]),
        "legacy_label_group_count": int(counters["legacy_label_group_count"]),
        "legacy_label_mismatch_count": int(counters["legacy_label_mismatch_count"]),
    }
    checks = _expectation_checks(actual, expectations)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "status": "pass"
        if all(check["status"] == "pass" for check in checks)
        else "fail",
        "dataset_id": dataset_id,
        "archive": archive_descriptor,
        "manifest_artifact": _artifact(manifest_path),
        "processing_contract": {
            "source_zip_open_mode": "read_only",
            "archive_sha256_complete_read_passes": 2,
            "archive_sha256_stable": True,
            "members_extracted": False,
            "decoded_pixel_memory_bound": "one_group_maximum_5_masks",
            "central_directory_metadata_retained": True,
            "record_output": "streaming_jsonl",
            "portable_outputs_contain_absolute_local_paths": False,
        },
        "counts": {
            **actual,
            "retained_group_count": int(counters["retained_group_count"]),
            "legacy_label_match_count": int(counters["legacy_label_match_count"]),
            "raw_overlap_pair_count": int(counters["raw_overlap_pair_count"]),
            "raw_overlap_pixels_total": int(counters["raw_overlap_pixels_total"]),
            "raw_overlap_pixels_max": int(counters["raw_overlap_pixels_max"]),
        },
        "group_count_by_generator_family": dict(sorted(family_counts.items())),
        "group_count_by_fragment_count": {
            str(key): value for key, value in sorted(fragment_counts.items())
        },
        "raw_overlap_pair_count_by_pixel_count": {
            str(key): value for key, value in sorted(overlap_pixel_histogram.items())
        },
        "quarantine_reason_counts": dict(sorted(quarantine_reasons.items())),
        "disconnected_group_ids": disconnected_group_ids,
        "legacy_label_mismatch_examples": label_mismatch_examples,
        "expectation_checks": checks,
    }
    portable_failures = _find_nonportable_strings(summary)
    if portable_failures:
        raise SyntheticManifestError(
            "non-portable path leaked into summary: {}".format(portable_failures[:5])
        )
    _atomic_write_bytes(summary_path, _json_bytes(summary))

    receipt = {
        "schema_version": "dunhuang-pairwise-local-path-receipt/0.2",
        "portable": False,
        "source_archive_absolute_path": str(archive_path),
        "source_archive_sha256": archive_sha256,
        "output_directory_absolute_path": str(output_dir),
        "portable_manifest": _artifact(manifest_path),
        "portable_summary": _artifact(summary_path),
        "source_archive_modified": False,
    }
    _atomic_write_bytes(local_receipt_path, _json_bytes(receipt))
    return BuildResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        local_receipt_path=local_receipt_path,
        summary=summary,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="dunhuang_voronoi_masks_no_erode_v0_2")
    parser.add_argument(
        "--logical-archive-id", default="local_asset://dunhuang_datas_zip"
    )
    parser.add_argument(
        "--expectations",
        choices=("dunhuang_datas_v0_2", "none"),
        default="dunhuang_datas_v0_2",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    expectations = (
        ArchiveExpectations.dunhuang_datas_v0_2()
        if args.expectations == "dunhuang_datas_v0_2"
        else ArchiveExpectations(legacy_label_mismatch_count=0)
    )
    result = build_synthetic_manifest(
        archive_path=args.archive,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        logical_archive_id=args.logical_archive_id,
        expectations=expectations,
    )
    print(
        json.dumps(
            {
                "status": result.summary["status"],
                "manifest_path": str(result.manifest_path),
                "summary_path": str(result.summary_path),
                "local_receipt_path": str(result.local_receipt_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
