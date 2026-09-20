"""Leakage-safe Pairwise v0.2 train/validation metadata streams.

This module never decodes an image.  It adapts the frozen MM/ECCV v0.1
group split and the canonical no-erosion Voronoi manifest into a single strict
binary-pair contract.  Archive pixels remain lazy ``MaskMemberRef`` objects.

The real Dunhuang external-test manifest is deliberately not accepted by any
public function in this module.  Historical ``test`` assignments are also
withheld: the only legal experiment splits here are ``train`` and ``val``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, Union

from staging.pairwise_v0_1.baselines.real_pair_stream import (
    ECCVSplitExposureAudit,
    RealPairRecord,
    SplitManifestView,
    iter_eccv_pair_records,
    iter_mm_pair_records,
    load_split_manifest,
)


STREAM_SCHEMA_VERSION = "dunhuang-pairwise-training-stream/0.2"
SYNTHETIC_MANIFEST_SCHEMA_VERSION = "dunhuang-pairwise-synthetic-group-manifest/0.2"
VALID_EXPERIMENT_SPLITS = frozenset(("train", "val"))
VALID_DIRECTIONS = frozenset(("left", "right", "above", "below"))
GEOMETRY_DIRECTION_MAP = MappingProxyType(
    {
        "left": "b_left_of_a",
        "right": "b_right_of_a",
        "above": "b_above_a",
        "below": "b_below_a",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TrainingDataError(ValueError):
    """Raised when training metadata violates a fail-closed contract."""


def direction_to_geometry_relation(direction_b_wrt_a: Optional[str]) -> Optional[str]:
    """Map B-with-respect-to-A labels to the geometry adapter vocabulary.

    ``None`` is retained for negatives and synthetic masks without a directed
    label.  Unknown strings fail closed instead of silently changing semantics.
    """

    if direction_b_wrt_a is None:
        return None
    try:
        return GEOMETRY_DIRECTION_MAP[direction_b_wrt_a]
    except KeyError as exc:
        raise TrainingDataError("unsupported B-with-respect-to-A direction") from exc


@dataclass(frozen=True)
class ArchiveBinding:
    """Portable identity of one canonical archive; never a local path."""

    logical_id: str
    archive_format: str
    sha256: str

    def __post_init__(self) -> None:
        if (
            not self.logical_id
            or "://" not in self.logical_id
            or self.logical_id.startswith("file://")
            or self.logical_id.startswith("/")
        ):
            raise ValueError("logical_id must be a portable non-file locator")
        if self.archive_format not in {"zip", "tar"}:
            raise ValueError("archive_format must be zip or tar")
        if not SHA256_RE.fullmatch(self.sha256):
            raise ValueError("archive sha256 must be 64 lowercase hex characters")


MM_CANONICAL_BINDING = ArchiveBinding(
    logical_id="canonical://mm_augmented/dunhuang_augmented_data",
    archive_format="zip",
    sha256="b66dd129e7c7877a2a595c5a6ea748e81dfa5f6f67d4f7f154a6b2c90869e8d2",
)
ECCV_CANONICAL_BINDING = ArchiveBinding(
    logical_id="canonical://eccv_1113data/1113data",
    archive_format="tar",
    sha256="98042e1b2068500f803be817a17e09470aefa34fb432047bebc50bebc06d9e72",
)


def _safe_member_path(value: str) -> bool:
    if not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and str(path) == value
        and path.suffix.casefold() == ".png"
    )


@dataclass(frozen=True)
class MaskMemberRef:
    """Lazy, portable reference to a scalar PNG mask inside an archive."""

    binding: ArchiveBinding
    archive_member: str
    fragment_id: str
    dataset_id: str
    canonical_group_id: str
    component_id: str
    split: str
    threshold_rule: str
    content_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if not _safe_member_path(self.archive_member):
            raise ValueError("archive_member must be a safe relative PNG path")
        for name, value in (
            ("fragment_id", self.fragment_id),
            ("dataset_id", self.dataset_id),
            ("canonical_group_id", self.canonical_group_id),
            ("component_id", self.component_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("{} is required".format(name))
        if self.split not in VALID_EXPERIMENT_SPLITS:
            raise ValueError("mask reference split must be train or val")
        if self.threshold_rule not in {
            "binary_brighter_value",
            "grayscale_uint8_gt_127",
        }:
            raise ValueError("unsupported threshold_rule")
        if self.content_sha256 is not None and not SHA256_RE.fullmatch(
            self.content_sha256
        ):
            raise ValueError("content_sha256 must be lowercase SHA-256")


@dataclass(frozen=True)
class TrainingPairRecord:
    """One binary Pairwise example with complete split/archive provenance."""

    fragment_a: MaskMemberRef
    fragment_b: MaskMemberRef
    label: bool
    direction_b_wrt_a: Optional[str]
    dataset_id: str
    canonical_group_id: str
    component_id: str
    split: str
    canonical_pair_key: Tuple[str, str]
    label_origin: str
    static_hard_negative_score: Optional[float] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = STREAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.label) is not bool:  # noqa: E721 - exact bool is intentional
            raise TypeError("label must be an explicit built-in bool")
        if self.split not in VALID_EXPERIMENT_SPLITS:
            raise ValueError("training records may only use train or val")
        if self.direction_b_wrt_a is not None:
            if self.direction_b_wrt_a not in VALID_DIRECTIONS:
                raise ValueError("unsupported direction_b_wrt_a")
            if not self.label:
                raise ValueError("negative records cannot carry a direction")
        if self.fragment_a.fragment_id == self.fragment_b.fragment_id:
            raise ValueError("self-pairs are not valid")
        expected_pair = tuple(
            sorted((self.fragment_a.fragment_id, self.fragment_b.fragment_id))
        )
        if self.canonical_pair_key != expected_pair:
            raise ValueError("canonical_pair_key disagrees with endpoints")
        if not self.label_origin:
            raise ValueError("label_origin is required")
        for fragment in (self.fragment_a, self.fragment_b):
            if (
                fragment.dataset_id != self.dataset_id
                or fragment.canonical_group_id != self.canonical_group_id
                or fragment.component_id != self.component_id
                or fragment.split != self.split
            ):
                raise ValueError("fragment provenance disagrees with pair provenance")
        score = self.static_hard_negative_score
        if score is not None:
            if self.label:
                raise ValueError("positive records cannot carry hard-negative scores")
            if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
                raise ValueError("hard-negative score must be finite in [0, 1]")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def pair_id(self) -> str:
        payload = json.dumps(
            [self.dataset_id, list(self.canonical_pair_key)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return "pair/sha256/" + hashlib.sha256(payload).hexdigest()

    @property
    def is_adjacent(self) -> bool:
        """Compatibility alias while preserving the strict bool contract."""

        return self.label


PathLike = Union[str, Path]


def _direction_value(record: RealPairRecord) -> Optional[str]:
    direction = record.direction_b_wrt_a
    return direction.value if direction is not None else None


def _adapt_historical_record(
    record: RealPairRecord,
    binding: ArchiveBinding,
) -> TrainingPairRecord:
    if record.split not in VALID_EXPERIMENT_SPLITS:
        raise TrainingDataError("historical test records are withheld")
    if binding.archive_format != record.fragment_a.archive_format:
        raise TrainingDataError("archive binding format disagrees with parser record")

    def adapt_fragment(fragment: Any) -> MaskMemberRef:
        return MaskMemberRef(
            binding=binding,
            archive_member=fragment.archive_member,
            fragment_id=fragment.fragment_id,
            dataset_id=record.dataset_id,
            canonical_group_id=record.canonical_group_id,
            component_id=record.cluster_id,
            split=record.split,
            threshold_rule="binary_brighter_value",
        )

    fragment_a = adapt_fragment(record.fragment_a)
    fragment_b = adapt_fragment(record.fragment_b)
    return TrainingPairRecord(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        label=record.is_adjacent,
        direction_b_wrt_a=_direction_value(record),
        dataset_id=record.dataset_id,
        canonical_group_id=record.canonical_group_id,
        component_id=record.cluster_id,
        split=record.split,
        canonical_pair_key=tuple(
            sorted((fragment_a.fragment_id, fragment_b.fragment_id))
        ),
        label_origin="historical_explicit_csv",
        static_hard_negative_score=None,
        provenance={
            "source_member": record.source_member,
            "source_row_number": record.source_row_number,
            "condition_raw": record.condition_raw,
            "upstream_manifest_candidate_id": record.manifest_candidate_id,
            "upstream_manifest_status": record.manifest_status,
            "upstream_manifest_authorization": record.manifest_authorization,
            "source_id": record.source_id,
            "variant_id": record.variant_id,
            "profile": record.profile,
            "derived_reverse": record.derived_reverse,
            "real_dunhuang_sealed_test": False,
        },
    )


def iter_historical_pair_records(
    *,
    split_manifest: Union[PathLike, Mapping[str, Any], SplitManifestView],
    split: str,
    mm_archive: Optional[Any] = None,
    eccv_archive: Optional[Any] = None,
    mm_binding: ArchiveBinding = MM_CANONICAL_BINDING,
    eccv_binding: ArchiveBinding = ECCV_CANONICAL_BINDING,
    eccv_exposure_audit: Optional[ECCVSplitExposureAudit] = None,
) -> Iterator[TrainingPairRecord]:
    """Stream frozen MM/ECCV records without admitting either held-out test.

    At least one archive is required.  Pair labels and explicit group/component
    split inheritance are delegated to the audited v0.1 parser/stream.
    """

    if split not in VALID_EXPERIMENT_SPLITS:
        raise TrainingDataError(
            "split must be train or val; historical test is withheld"
        )
    if mm_archive is None and eccv_archive is None:
        raise TrainingDataError("at least one historical archive is required")
    if eccv_exposure_audit is not None and eccv_archive is None:
        raise TrainingDataError("eccv_exposure_audit requires an ECCV archive source")
    manifest = (
        split_manifest
        if isinstance(split_manifest, SplitManifestView)
        else load_split_manifest(split_manifest)
    )
    if mm_archive is not None:
        for record in iter_mm_pair_records(
            mm_archive,
            manifest,
            splits=split,
            include_reverse=False,
        ):
            yield _adapt_historical_record(record, mm_binding)
    if eccv_archive is not None:
        for record in iter_eccv_pair_records(
            eccv_archive,
            manifest,
            splits=split,
            include_reverse=False,
            exposure_audit=eccv_exposure_audit,
        ):
            yield _adapt_historical_record(record, eccv_binding)


def _synthetic_binding(archive: Mapping[str, Any]) -> ArchiveBinding:
    try:
        return ArchiveBinding(
            logical_id=str(archive["logical_id"]),
            archive_format=str(archive["format"]),
            sha256=str(archive["sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingDataError("invalid synthetic archive binding") from exc


def _synthetic_component_id(group_id: str) -> str:
    digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    return "synthetic/group-sha256/" + digest


def _validate_synthetic_group(
    record: Mapping[str, Any], line_number: int
) -> Tuple[ArchiveBinding, Dict[int, Mapping[str, Any]], Tuple[Mapping[str, Any], ...]]:
    if record.get("schema_version") != SYNTHETIC_MANIFEST_SCHEMA_VERSION:
        raise TrainingDataError(
            "synthetic line {} has an unsupported schema".format(line_number)
        )
    if record.get("no_erode") is not True:
        raise TrainingDataError(
            "synthetic line {} is not the frozen no-erode profile".format(line_number)
        )
    binding = _synthetic_binding(record.get("archive", {}))
    if binding.archive_format != "zip":
        raise TrainingDataError("synthetic masks must use a ZIP archive")
    members = record.get("members")
    measurements = record.get("pair_measurements")
    if not isinstance(members, list) or not isinstance(measurements, list):
        raise TrainingDataError(
            "synthetic group requires members and pair_measurements"
        )
    by_id: Dict[int, Mapping[str, Any]] = {}
    for member in members:
        if not isinstance(member, Mapping):
            raise TrainingDataError("synthetic member must be an object")
        fragment_id = member.get("fragment_id")
        if type(fragment_id) is not int:  # noqa: E721
            raise TrainingDataError("synthetic fragment_id must be an int")
        if fragment_id in by_id:
            raise TrainingDataError("duplicate synthetic fragment_id")
        by_id[fragment_id] = member
    if int(record.get("fragment_count", -1)) != len(by_id) or not 2 <= len(by_id) <= 5:
        raise TrainingDataError("synthetic groups must contain exactly 2--5 members")

    expected_pairs = {
        (left, right)
        for index, left in enumerate(sorted(by_id))
        for right in sorted(by_id)[index + 1 :]
    }
    observed_pairs = set()
    for measurement in measurements:
        if not isinstance(measurement, Mapping):
            raise TrainingDataError("pair measurement must be an object")
        first = measurement.get("fragment_a")
        second = measurement.get("fragment_b")
        if type(first) is not int or type(second) is not int:  # noqa: E721
            raise TrainingDataError("measurement fragment ids must be ints")
        pair = tuple(sorted((first, second)))
        if first == second or pair in observed_pairs:
            raise TrainingDataError("invalid or duplicate pair measurement")
        if pair not in expected_pairs:
            raise TrainingDataError("measurement references an unknown fragment")
        if type(measurement.get("is_neighbor")) is not bool:  # noqa: E721
            raise TrainingDataError("is_neighbor must be an explicit bool")
        observed_pairs.add(pair)
    if observed_pairs != expected_pairs:
        raise TrainingDataError("pair measurements must cover every unordered pair")
    return binding, by_id, tuple(measurements)


def _open_synthetic_manifest_source(
    source: Union[PathLike, bytes],
):
    if type(source) is bytes:  # noqa: E721
        try:
            return io.StringIO(source.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise TrainingDataError("synthetic manifest is not UTF-8") from exc
    return Path(source).open("r", encoding="utf-8")


def _iter_synthetic_pair_records_source(
    manifest_source: Union[PathLike, bytes],
) -> Iterator[TrainingPairRecord]:
    seen_groups = set()
    canonical_binding: Optional[ArchiveBinding] = None
    with _open_synthetic_manifest_source(manifest_source) as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                group = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataError(
                    "invalid JSON on synthetic line {}".format(line_number)
                ) from exc
            if not isinstance(group, Mapping):
                raise TrainingDataError("synthetic JSONL rows must be objects")
            group_id = str(group.get("group_id", "")).strip()
            if not group_id or group_id in seen_groups:
                raise TrainingDataError("missing or duplicate synthetic group_id")
            seen_groups.add(group_id)
            quarantine = group.get("quarantine")
            if not isinstance(quarantine, Mapping):
                raise TrainingDataError("synthetic group requires quarantine status")
            quarantine_status = quarantine.get("status")
            if quarantine_status == "quarantined":
                continue
            if quarantine_status != "retained":
                raise TrainingDataError("unsupported quarantine status")

            binding, members, measurements = _validate_synthetic_group(
                group, line_number
            )
            if canonical_binding is None:
                canonical_binding = binding
            elif binding != canonical_binding:
                raise TrainingDataError("synthetic manifest changes archive binding")
            component_id = _synthetic_component_id(group_id)

            def fragment_ref(fragment_id: int) -> MaskMemberRef:
                member = members[fragment_id]
                return MaskMemberRef(
                    binding=binding,
                    archive_member=str(member.get("archive_member", "")),
                    fragment_id="{}/fragment/{}".format(group_id, fragment_id),
                    dataset_id=str(group.get("dataset_id", "")),
                    canonical_group_id=group_id,
                    component_id=component_id,
                    split="train",
                    threshold_rule=str(member.get("threshold_rule", "")),
                    content_sha256=str(member.get("content_sha256", "")),
                )

            for measurement in measurements:
                first = int(measurement["fragment_a"])
                second = int(measurement["fragment_b"])
                label = measurement["is_neighbor"]
                fragment_a = fragment_ref(first)
                fragment_b = fragment_ref(second)
                legacy_overlap = measurement.get("legacy_dilated_overlap_pixels")
                if type(legacy_overlap) is not int or legacy_overlap < 0:  # noqa: E721
                    raise TrainingDataError(
                        "legacy_dilated_overlap_pixels must be a non-negative int"
                    )
                minimum = 30
                hard_score = None
                if not label:
                    hard_score = min(float(legacy_overlap) / float(minimum), 1.0)
                yield TrainingPairRecord(
                    fragment_a=fragment_a,
                    fragment_b=fragment_b,
                    label=label,
                    direction_b_wrt_a=None,
                    dataset_id=str(group.get("dataset_id", "")),
                    canonical_group_id=group_id,
                    component_id=component_id,
                    split="train",
                    canonical_pair_key=tuple(
                        sorted((fragment_a.fragment_id, fragment_b.fragment_id))
                    ),
                    label_origin="legacy_neighbor_rule_v1",
                    static_hard_negative_score=hard_score,
                    provenance={
                        "generator_family": group.get("generator_family"),
                        "erosion_profile": group.get("erosion_profile"),
                        "source_group_id": group.get("source_group_id"),
                        "legacy_dilated_overlap_pixels": legacy_overlap,
                        "raw_overlap_pixels": measurement.get("raw_overlap_pixels"),
                        "legacy_minimum_overlap_pixels_inclusive": minimum,
                        "source_lineage_status": "unavailable_training_only",
                        "component_semantics": "group_isolation_only",
                        "real_dunhuang_sealed_test": False,
                    },
                )


def iter_synthetic_pair_records(
    manifest_path: PathLike,
) -> Iterator[TrainingPairRecord]:
    """Stream retained synthetic pairs from a filesystem manifest."""

    yield from _iter_synthetic_pair_records_source(manifest_path)


def iter_synthetic_pair_records_payload(
    payload: bytes,
) -> Iterator[TrainingPairRecord]:
    """Stream retained synthetic pairs from already captured immutable bytes."""

    if type(payload) is not bytes:  # noqa: E721
        raise TrainingDataError("synthetic manifest payload must be immutable bytes")
    yield from _iter_synthetic_pair_records_source(payload)


def iter_training_pair_records(
    *,
    split: str,
    split_manifest: Union[PathLike, Mapping[str, Any], SplitManifestView],
    mm_archive: Optional[Any] = None,
    eccv_archive: Optional[Any] = None,
    synthetic_manifest: Optional[PathLike] = None,
) -> Iterator[TrainingPairRecord]:
    """Compose the deterministic historical and synthetic train/val streams."""

    if split not in VALID_EXPERIMENT_SPLITS:
        raise TrainingDataError("split must be train or val")
    if mm_archive is not None or eccv_archive is not None:
        yield from iter_historical_pair_records(
            split_manifest=split_manifest,
            split=split,
            mm_archive=mm_archive,
            eccv_archive=eccv_archive,
        )
    if split == "train" and synthetic_manifest is not None:
        yield from iter_synthetic_pair_records(synthetic_manifest)


__all__ = [
    "ArchiveBinding",
    "ECCVSplitExposureAudit",
    "ECCV_CANONICAL_BINDING",
    "MM_CANONICAL_BINDING",
    "MaskMemberRef",
    "GEOMETRY_DIRECTION_MAP",
    "STREAM_SCHEMA_VERSION",
    "TrainingDataError",
    "TrainingPairRecord",
    "direction_to_geometry_relation",
    "iter_historical_pair_records",
    "iter_synthetic_pair_records",
    "iter_synthetic_pair_records_payload",
    "iter_training_pair_records",
]
