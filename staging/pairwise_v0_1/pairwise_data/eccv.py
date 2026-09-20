"""Read-only parser for the canonical ECCV ``1113data.tar.gz`` archive.

The historical archive mixes three kinds of supervision:

* per-group 2x2 ``pair.csv`` / ``pair_false.csv`` files;
* per-group 3x3 ``pair.csv`` / ``pair_false.csv`` files; and
* aggregate Siamese manifests such as ``pair_120000_1_1.csv`` together
  with ``updated_`` copies whose only material change is an absolute path.

This module never extracts archive members to disk.  It canonicalises pair
identity as an *undirected* key before accepting a negative.  Consequently a
3x3 ``pair_false.csv`` row that is merely the reverse of a positive adjacency
is retained as an audit conflict and is not exposed as a usable negative.
"""

from __future__ import annotations

import csv
import io
import os
import posixpath
import re
import tarfile
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)


ArchiveInput = Union[str, os.PathLike, bytes, bytearray, BinaryIO]
PairKey = Tuple[str, str]
ECCV_READ_SCOPE_SCHEMA_VERSION = "pairwise-eccv-read-scope/0.1"


class ECCVParseError(ValueError):
    """Raised when a canonical ECCV CSV cannot be interpreted safely."""


@dataclass(frozen=True)
class _ExtractedTarMember:
    """The tiny ``TarInfo`` surface consumed by :class:`ECCVArchive`."""

    name: str
    path: Path

    def isfile(self) -> bool:
        return True


class _ExtractedTarView:
    """Expose an extracted archive tree through the parser's tar read surface."""

    def __init__(self, root: Path):
        self._root = Path(root)
        if not self._root.is_dir():
            raise FileNotFoundError(self._root)
        members = []
        for directory, child_directories, filenames in os.walk(self._root):
            child_directories.sort()
            for filename in sorted(filenames):
                path = Path(directory) / filename
                members.append(
                    _ExtractedTarMember(
                        name=path.relative_to(self._root).as_posix(), path=path
                    )
                )
        self._members = tuple(members)
        self._closed = False

    def __iter__(self) -> Iterator[_ExtractedTarMember]:
        if self._closed:
            raise ValueError("extracted tar view is closed")
        return iter(self._members)

    def extractfile(self, member: _ExtractedTarMember) -> BinaryIO:
        if self._closed:
            raise ValueError("extracted tar view is closed")
        relative = PurePosixPath(member.name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe extracted tar member")
        expected = self._root.joinpath(*relative.parts)
        if expected != member.path or not expected.is_file():
            raise KeyError(member.name)
        return expected.open("rb")

    def close(self) -> None:
        self._closed = True


_DIRECTION_SWAP = {
    "left-right": "right-left",
    "right-left": "left-right",
    "up-down": "down-up",
    "down-up": "up-down",
}

_TRUE_VALUES = {"true", "1", "yes", "y", "positive", "adjacent"}
_FALSE_VALUES = {"false", "0", "no", "n", "negative", "non-adjacent", "nonadjacent"}

_GROUP_PATTERNS = (
    (
        "2x2-negative",
        re.compile(
            r"^(?P<root>.*?/)?siamese/2x2/negative/(?P<group>[^/]+)/(?P<file>[^/]+)$",
            re.IGNORECASE,
        ),
    ),
    (
        "2x2",
        re.compile(
            r"^(?P<root>.*?/)?siamese/2x2/(?P<group>[^/]+)/(?P<file>[^/]+)$",
            re.IGNORECASE,
        ),
    ),
    (
        "3x3",
        re.compile(
            r"^(?P<root>.*?/)?3x3/(?P<group>[^/]+)/(?P<file>[^/]+)$",
            re.IGNORECASE,
        ),
    ),
)

_AGGREGATE_BASENAME = re.compile(r"^(?:updated_)?pair_.+\.csv$", re.IGNORECASE)
_CANONICAL_PAIR_GROUP = re.compile(
    r"^eccv/(?P<profile>2x2|2x2-negative)/(?P<group>[^/]+)$"
)


def _normalise_pair_group_scope(
    values: Iterable[str],
) -> frozenset:
    if isinstance(values, (str, bytes, bytearray)):
        raise ECCVParseError(
            "canonical ECCV pair-group scope must be an iterable of ids"
        )
    scope = frozenset(str(value).strip() for value in values)
    invalid = sorted(
        value
        for value in scope
        if not value or _CANONICAL_PAIR_GROUP.fullmatch(value) is None
    )
    if invalid:
        raise ECCVParseError(
            "invalid canonical ECCV pair-group scope: {!r}".format(invalid[0])
        )
    return scope


def swap_direction(direction: Optional[str]) -> Optional[str]:
    """Return the direction after swapping the ordered pair endpoints."""

    if direction is None:
        return None
    try:
        return _DIRECTION_SWAP[direction]
    except KeyError as error:
        raise ECCVParseError("unsupported direction: {!r}".format(direction)) from error


def _normalise_direction(value: str) -> Optional[str]:
    text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    text = re.sub(r"-+", "-", text)
    aliases = {
        "leftright": "left-right",
        "rightleft": "right-left",
        "updown": "up-down",
        "downup": "down-up",
        "top-bottom": "up-down",
        "bottom-top": "down-up",
    }
    text = aliases.get(text, text)
    return text if text in _DIRECTION_SWAP else None


def _condition(value: Any, source: str, row_number: int) -> Tuple[bool, Optional[str]]:
    raw = str(value or "").strip()
    direction = _normalise_direction(raw)
    if direction is not None:
        return True, direction
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True, None
    if lowered in _FALSE_VALUES:
        return False, None
    raise ECCVParseError(
        "unknown condition {!r} in {} row {}".format(raw, source, row_number)
    )


def _normalise_member_name(value: str) -> str:
    text = str(value).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return posixpath.normpath(text)


def normalise_fragment_path(value: str) -> str:
    """Normalise legacy relative/absolute fragment paths to archive identities."""

    text = _normalise_member_name(str(value).strip().strip("\"'"))
    lowered = text.lower()
    markers = (
        ("/siamese/2x2/", "2x2/"),
        ("/3x3/", "3x3/"),
    )
    for marker, replacement in markers:
        index = lowered.rfind(marker)
        if index >= 0:
            return replacement + text[index + len(marker) :]
    if lowered.startswith("siamese/2x2/"):
        return "2x2/" + text[len("siamese/2x2/") :]
    if lowered.startswith("2x2/") or lowered.startswith("3x3/"):
        return text
    return text


def _fragment_sort_key(fragment_id: str) -> Tuple[Any, ...]:
    parts = normalise_fragment_path(fragment_id).split("/")
    output: List[Any] = []
    for part in parts:
        stem = part[:-4] if part.lower().endswith(".png") else part
        if stem.isdigit():
            output.append((0, int(stem)))
        else:
            output.append((1, part))
    return tuple(output)


def canonical_pair_key(fragment_a: str, fragment_b: str) -> PairKey:
    """Return a stable undirected pair key after legacy-path normalisation."""

    left = normalise_fragment_path(fragment_a)
    right = normalise_fragment_path(fragment_b)
    if _fragment_sort_key(left) <= _fragment_sort_key(right):
        return left, right
    return right, left


def _piece_id(value: str) -> str:
    name = posixpath.basename(str(value).strip())
    return name[:-4] if name.lower().endswith(".png") else name


def _group_fragment_id(profile: str, group_id: str, value: str) -> str:
    piece = _piece_id(value)
    if profile == "2x2-negative":
        return "2x2/negative/{}/{}.png".format(group_id, piece)
    return "{}/{}/{}.png".format(profile, group_id, piece)


def _normalise_profile(value: str) -> str:
    text = str(value).strip().lower().replace("×", "x").replace("_", "-")
    aliases = {
        "2x2negative": "2x2-negative",
        "2x2-negative": "2x2-negative",
        "negative-2x2": "2x2-negative",
        "2x2": "2x2",
        "3x3": "3x3",
    }
    if text not in aliases:
        raise KeyError("unknown ECCV group profile: {!r}".format(value))
    return aliases[text]


def _known_grid_relation(fragment_a: str, fragment_b: str) -> Optional[Tuple[str, bool]]:
    """Infer a historical row-major adjacency and whether it is forward."""

    parsed_a = _parse_fragment_identity(fragment_a)
    parsed_b = _parse_fragment_identity(fragment_b)
    if parsed_a is None or parsed_b is None:
        return None
    profile_a, group_a, piece_a = parsed_a
    profile_b, group_b, piece_b = parsed_b
    if profile_a != profile_b or group_a != group_b:
        return None
    size = 2 if profile_a in {"2x2", "2x2-negative"} else 3
    try:
        first = int(piece_a)
        second = int(piece_b)
    except ValueError:
        return None
    maximum = size * size
    if not (1 <= first <= maximum and 1 <= second <= maximum):
        return None
    row_a, col_a = divmod(first - 1, size)
    row_b, col_b = divmod(second - 1, size)
    if row_a == row_b and col_b == col_a + 1:
        return "left-right", True
    if row_a == row_b and col_a == col_b + 1:
        return "right-left", False
    if col_a == col_b and row_b == row_a + 1:
        return "up-down", True
    if col_a == col_b and row_a == row_b + 1:
        return "down-up", False
    return None


def _parse_fragment_identity(fragment_id: str) -> Optional[Tuple[str, str, str]]:
    text = normalise_fragment_path(fragment_id)
    negative = re.match(r"^2x2/negative/([^/]+)/([^/]+?)(?:\.png)?$", text, re.IGNORECASE)
    if negative:
        return "2x2-negative", negative.group(1), negative.group(2)
    match = re.match(r"^(2x2|3x3)/([^/]+)/([^/]+?)(?:\.png)?$", text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower(), match.group(2), match.group(3)


@dataclass(frozen=True)
class PairRecord:
    """One ordered view of a canonical pair relationship."""

    fragment_a_id: str
    fragment_b_id: str
    is_adjacent: bool
    direction: Optional[str]
    group_id: Optional[str]
    profile: Optional[str]
    source_member: str
    row_number: int
    raw_condition: str
    phase: str = "group"
    excluded: bool = False
    exclusion_reason: Optional[str] = None
    conflict_kind: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def pair_id(self) -> str:
        key = self.canonical_key
        return "eccv:{}::{}".format(key[0], key[1])

    @property
    def canonical_key(self) -> PairKey:
        return canonical_pair_key(self.fragment_a_id, self.fragment_b_id)

    @property
    def label(self) -> bool:
        return self.is_adjacent

    @property
    def usable_as_negative(self) -> bool:
        return (not self.is_adjacent) and not self.excluded

    @property
    def source_document(self) -> None:
        """Physical source is unknown; never union the whole archive by dataset id."""

        return None

    @property
    def dataset_id(self) -> str:
        return "eccv_1113data"

    @property
    def canonical_group_id(self) -> Optional[str]:
        if self.profile is None or self.group_id is None:
            return None
        return "eccv/{}/{}".format(self.profile, self.group_id)

    @property
    def template_id(self) -> None:
        return None

    @property
    def content_hash(self) -> None:
        return None

    @property
    def hash_group(self) -> None:
        return None

    @property
    def geometry(self) -> None:
        return None

    @property
    def split(self) -> None:
        return None

    def swapped(self) -> "PairRecord":
        return PairRecord(
            fragment_a_id=self.fragment_b_id,
            fragment_b_id=self.fragment_a_id,
            is_adjacent=self.is_adjacent,
            direction=swap_direction(self.direction),
            group_id=self.group_id,
            profile=self.profile,
            source_member=self.source_member,
            row_number=self.row_number,
            raw_condition=self.raw_condition,
            phase=self.phase,
            excluded=self.excluded,
            exclusion_reason=self.exclusion_reason,
            conflict_kind=self.conflict_kind,
            metadata=dict(self.metadata),
        )

    def canonicalised(self) -> "PairRecord":
        key = self.canonical_key
        if (self.fragment_a_id, self.fragment_b_id) == key:
            return self
        return self.swapped()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "fragment_a_id": self.fragment_a_id,
            "fragment_b_id": self.fragment_b_id,
            "canonical_pair": list(self.canonical_key),
            "group_id": self.group_id,
            "canonical_group_id": self.canonical_group_id,
            "profile": self.profile,
            "source_document": self.source_document,
            "dataset_id": self.dataset_id,
            "is_adjacent": self.is_adjacent,
            "label": self.label,
            "direction": self.direction,
            "phase": self.phase,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
            "conflict_kind": self.conflict_kind,
            "source_member": self.source_member,
            "row_number": self.row_number,
            "raw_condition": self.raw_condition,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PairConflict:
    canonical_pair: PairKey
    reason: str
    positive_rows: Tuple[PairRecord, ...]
    negative_rows: Tuple[PairRecord, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_pair": list(self.canonical_pair),
            "reason": self.reason,
            "positive_rows": [row.to_dict() for row in self.positive_rows],
            "negative_rows": [row.to_dict() for row in self.negative_rows],
        }


@dataclass(frozen=True)
class GroupRecord:
    profile: str
    group_id: str
    member_prefix: str
    fragment_ids: Tuple[str, ...]
    pairs: Tuple[PairRecord, ...]
    conflicts: Tuple[PairConflict, ...]
    raw_positive_rows: int
    raw_false_rows: int
    excluded: bool = False
    exclusion_reason: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def canonical_group_id(self) -> str:
        return "eccv/{}/{}".format(self.profile, self.group_id)

    @property
    def source_document(self) -> None:
        """No manuscript/source anchor survives in the canonical ECCV archive."""

        return None

    @property
    def dataset_id(self) -> str:
        return "eccv_1113data"

    @property
    def benchmark_eligible(self) -> bool:
        return not self.excluded

    @property
    def usable_negative_count(self) -> int:
        return sum(1 for pair in self.pairs if pair.usable_as_negative)

    @property
    def positive_count(self) -> int:
        return sum(1 for pair in self.pairs if pair.is_adjacent)

    @property
    def suppressed_negative_count(self) -> int:
        return sum(len(conflict.negative_rows) for conflict in self.conflicts)

    def _normalise_lookup_fragment(self, value: str) -> str:
        text = str(value)
        if "/" not in text:
            return _group_fragment_id(self.profile, self.group_id, text)
        return normalise_fragment_path(text)

    def relationship(self, fragment_a: str, fragment_b: str) -> Optional[PairRecord]:
        """Return the relationship oriented from ``fragment_a`` to ``fragment_b``."""

        first = self._normalise_lookup_fragment(fragment_a)
        second = self._normalise_lookup_fragment(fragment_b)
        key = canonical_pair_key(first, second)
        for pair in self.pairs:
            if pair.canonical_key != key:
                continue
            if pair.fragment_a_id == first and pair.fragment_b_id == second:
                return pair
            return pair.swapped()
        return None

    pair = relationship

    def directed_pairs(
        self, include_negatives: bool = False, include_reverse: bool = True
    ) -> Iterator[PairRecord]:
        for pair in self.pairs:
            if not pair.is_adjacent and not include_negatives:
                continue
            yield pair
            if include_reverse:
                yield pair.swapped()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "canonical_group_id": self.canonical_group_id,
            "profile": self.profile,
            "fragment_ids": list(self.fragment_ids),
            "source_document": self.source_document,
            "dataset_id": self.dataset_id,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "raw_positive_rows": self.raw_positive_rows,
            "raw_false_rows": self.raw_false_rows,
            "positive_count": self.positive_count,
            "usable_negative_count": self.usable_negative_count,
            "suppressed_negative_count": self.suppressed_negative_count,
            "benchmark_eligible": self.benchmark_eligible,
            "generalization_eligible": self.benchmark_eligible,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
            "allowed_use": self.metadata.get("allowed_use", "benchmark_candidate"),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AggregateManifestProfile:
    manifest_id: str
    member_path: str
    raw_rows: int
    positive_rows: int
    false_rows: int
    usable_negative_rows: int
    undirected_conflict_rows: int
    direct_false_pollution_rows: int
    reverse_positive_false_rows: int
    updated_fallback: bool

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AggregateManifest:
    manifest_id: str
    member_path: str
    records: Tuple[PairRecord, ...]
    updated_fallback: bool = False

    @property
    def usable_records(self) -> Tuple[PairRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.is_adjacent or record.usable_as_negative
        )

    @property
    def conflicts(self) -> Tuple[PairRecord, ...]:
        return tuple(record for record in self.records if record.conflict_kind is not None)

    def profile(self) -> AggregateManifestProfile:
        false_rows = [record for record in self.records if not record.is_adjacent]
        return AggregateManifestProfile(
            manifest_id=self.manifest_id,
            member_path=self.member_path,
            raw_rows=len(self.records),
            positive_rows=sum(record.is_adjacent for record in self.records),
            false_rows=len(false_rows),
            usable_negative_rows=sum(record.usable_as_negative for record in false_rows),
            undirected_conflict_rows=sum(record.conflict_kind is not None for record in false_rows),
            direct_false_pollution_rows=sum(
                record.conflict_kind == "positive_edge_labeled_false" for record in false_rows
            ),
            reverse_positive_false_rows=sum(
                record.conflict_kind == "reverse_positive_edge_labeled_false"
                for record in false_rows
            ),
            updated_fallback=self.updated_fallback,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "member_path": self.member_path,
            "updated_fallback": self.updated_fallback,
            "profile": self.profile().to_dict(),
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class ECCVArchiveProfile:
    group_counts: Mapping[str, int]
    group_positive_pairs: Mapping[str, int]
    group_usable_negative_pairs: Mapping[str, int]
    group_conflict_rows: Mapping[str, int]
    aggregate_manifest_count: int
    aggregate_raw_rows: int
    aggregate_positive_rows: int
    aggregate_false_rows: int
    aggregate_usable_negative_rows: int
    aggregate_undirected_conflict_rows: int
    aggregate_direct_false_pollution_rows: int
    aggregate_reverse_positive_false_rows: int
    skipped_updated_manifests: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_counts": dict(self.group_counts),
            "group_positive_pairs": dict(self.group_positive_pairs),
            "group_usable_negative_pairs": dict(self.group_usable_negative_pairs),
            "group_conflict_rows": dict(self.group_conflict_rows),
            "aggregate_manifest_count": self.aggregate_manifest_count,
            "aggregate_raw_rows": self.aggregate_raw_rows,
            "aggregate_positive_rows": self.aggregate_positive_rows,
            "aggregate_false_rows": self.aggregate_false_rows,
            "aggregate_usable_negative_rows": self.aggregate_usable_negative_rows,
            "aggregate_undirected_conflict_rows": self.aggregate_undirected_conflict_rows,
            "aggregate_direct_false_pollution_rows": self.aggregate_direct_false_pollution_rows,
            "aggregate_reverse_positive_false_rows": self.aggregate_reverse_positive_false_rows,
            "skipped_updated_manifests": list(self.skipped_updated_manifests),
        }


@dataclass(frozen=True)
class ECCVReadScopeEvidence:
    """Immutable application-level evidence for one tar metadata/payload scan.

    A streaming tar implementation may drain compressed container bytes while
    advancing to the next header.  The zero-read claims here deliberately
    cover member payload handles and application parsing only: ``extractfile``,
    payload caching, decode, and CSV row parsing.  They do not claim that the
    underlying compressed transport performed zero reads.
    """

    mode: str
    allowed_pair_group_ids: Optional[Tuple[str, ...]]
    aggregate_payloads_enabled: bool
    container_header_scan_completed: bool
    tar_member_headers_scanned: int
    discovered_pair_group_count: int
    selected_pair_group_count_present: int
    selected_group_csv_headers_seen: int
    selected_group_csv_extractfile_calls: int
    selected_group_csv_payload_read_calls: int
    selected_group_csv_payloads_cached: int
    selected_group_csv_decode_calls: int
    selected_group_csv_parse_calls: int
    nonselected_group_csv_headers_seen: int
    nonselected_group_csv_extractfile_calls: int
    nonselected_group_csv_payload_read_calls: int
    nonselected_group_csv_payloads_cached: int
    nonselected_group_csv_decode_calls: int
    nonselected_group_csv_parse_calls: int
    selected_fragment_headers_seen: int
    selected_fragment_extractfile_calls: int
    nonselected_fragment_headers_seen: int
    nonselected_fragment_extractfile_calls: int
    aggregate_csv_headers_seen: int
    aggregate_csv_extractfile_calls: int
    aggregate_csv_payload_read_calls: int
    aggregate_csv_payloads_cached: int
    aggregate_csv_decode_calls: int
    aggregate_csv_parse_calls: int
    schema_version: str = ECCV_READ_SCOPE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.__dict__)
        if self.allowed_pair_group_ids is not None:
            payload["allowed_pair_group_ids"] = list(
                self.allowed_pair_group_ids
            )
        payload["guarantee_scope"] = {
            "tar_container_transport_bytes_may_be_drained": True,
            "tar_member_headers_are_scanned_in_archive_order": True,
            "zero_payload_read_means": [
                "no_tarfile_extractfile_call",
                "no_extracted_member_payload_read_call",
                "no_application_payload_cache",
                "no_application_decode",
                "no_csv_row_parse",
            ],
        }
        return payload


class ECCVArchive:
    """Indexed, read-only view over a canonical ECCV tar archive.

    Construction performs exactly one forward tar scan.  Only relevant CSV
    payloads are retained in memory; image members are represented by name.
    This matters for ``.tar.gz`` inputs because reopening the archive and
    extracting groups in numeric order otherwise repeatedly seeks through the
    compressed stream.
    """

    def __init__(
        self,
        source: ArchiveInput,
        *,
        allowed_pair_group_ids: Optional[Iterable[str]] = None,
    ):
        """Scan ``source`` once, optionally under an exact pair-group scope.

        ``None`` preserves the legacy parser: every group CSV and aggregate
        CSV payload is cached.  A non-``None`` scope is held-out-safe: only
        per-group CSV payloads whose canonical group id is in the scope may be
        extracted, and aggregate CSV payloads are disabled because they mix
        groups and cannot be classified before payload access.
        """

        self._pair_group_payload_scope = (
            None
            if allowed_pair_group_ids is None
            else _normalise_pair_group_scope(allowed_pair_group_ids)
        )
        self._aggregate_payloads_enabled = self._pair_group_payload_scope is None
        self._container_header_scan_completed = False
        self._tar_member_headers_scanned = 0
        self._discovered_pair_group_ids: Set[str] = set()
        self._selected_group_csv_headers_seen = 0
        self._selected_group_csv_extractfile_calls = 0
        self._selected_group_csv_payload_read_calls = 0
        self._selected_group_csv_payloads_cached = 0
        self._selected_group_csv_decode_calls = 0
        self._selected_group_csv_parse_calls = 0
        self._nonselected_group_csv_headers_seen = 0
        self._nonselected_group_csv_extractfile_calls = 0
        self._nonselected_group_csv_payload_read_calls = 0
        self._nonselected_group_csv_payloads_cached = 0
        self._nonselected_group_csv_decode_calls = 0
        self._nonselected_group_csv_parse_calls = 0
        self._selected_fragment_headers_seen = 0
        self._selected_fragment_extractfile_calls = 0
        self._nonselected_fragment_headers_seen = 0
        self._nonselected_fragment_extractfile_calls = 0
        self._aggregate_csv_headers_seen = 0
        self._aggregate_csv_extractfile_calls = 0
        self._aggregate_csv_payload_read_calls = 0
        self._aggregate_csv_payloads_cached = 0
        self._aggregate_csv_decode_calls = 0
        self._aggregate_csv_parse_calls = 0
        self._group_csv_members: Set[str] = set()
        self._aggregate_csv_members: Set[str] = set()

        archive: Any
        source_stream: Optional[BinaryIO] = None
        original_position: Optional[int] = None
        if isinstance(source, (str, os.PathLike)):
            source_path = Path(source)
            if source_path.is_dir():
                archive = _ExtractedTarView(source_path)
            else:
                archive = tarfile.open(os.fspath(source), mode="r|*")
        elif isinstance(source, (bytes, bytearray)):
            archive = tarfile.open(fileobj=io.BytesIO(bytes(source)), mode="r|*")
        elif hasattr(source, "read"):
            source_stream = source  # type: ignore[assignment]
            if hasattr(source_stream, "tell"):
                try:
                    original_position = source_stream.tell()
                except (OSError, io.UnsupportedOperation):
                    original_position = None
            if hasattr(source_stream, "seek"):
                try:
                    source_stream.seek(0)
                except (OSError, io.UnsupportedOperation):
                    pass
            archive = tarfile.open(fileobj=source_stream, mode="r|*")
        else:
            raise TypeError("source must be a tar path, bytes, or a binary file object")

        self._groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._csv_payloads: Dict[str, bytes] = {}
        aggregate_candidates: Dict[str, List[Tuple[bool, str]]] = {}
        try:
            for member in archive:
                self._tar_member_headers_scanned += 1
                if not member.isfile():
                    continue
                name = _normalise_member_name(member.name)
                matched_group = False
                for profile, pattern in _GROUP_PATTERNS:
                    match = pattern.match(name)
                    if not match:
                        continue
                    matched_group = True
                    group_id = match.group("group")
                    filename = match.group("file")
                    lowered = filename.lower()
                    canonical_group_id = "eccv/{}/{}".format(
                        profile, group_id
                    )
                    if profile in {"2x2", "2x2-negative"}:
                        self._discovered_pair_group_ids.add(canonical_group_id)
                    in_scope = (
                        self._pair_group_payload_scope is None
                        or canonical_group_id in self._pair_group_payload_scope
                    )
                    is_group_csv = lowered in {"pair.csv", "pair_false.csv"}
                    is_fragment = lowered.endswith(".png")
                    if not in_scope:
                        if is_group_csv:
                            self._nonselected_group_csv_headers_seen += 1
                        elif is_fragment:
                            self._nonselected_fragment_headers_seen += 1
                        # The scope decision is based only on the tar header
                        # name.  No bucket, payload handle, cache, decode, or
                        # CSV parser is created for an out-of-scope group.
                        break
                    bucket = self._groups.setdefault(
                        (profile, group_id),
                        {
                            "prefix": posixpath.dirname(name),
                            "pair": None,
                            "pair_false": None,
                            "fragments": [],
                        },
                    )
                    if lowered == "pair.csv":
                        self._selected_group_csv_headers_seen += 1
                        bucket["pair"] = name
                        self._cache_csv_payload(
                            archive, member, name, payload_kind="group"
                        )
                    elif lowered == "pair_false.csv":
                        self._selected_group_csv_headers_seen += 1
                        bucket["pair_false"] = name
                        self._cache_csv_payload(
                            archive, member, name, payload_kind="group"
                        )
                    elif lowered.endswith(".png"):
                        self._selected_fragment_headers_seen += 1
                        bucket["fragments"].append(name)
                    break
                if matched_group:
                    continue
                basename = posixpath.basename(name)
                if not _AGGREGATE_BASENAME.match(basename):
                    continue
                self._aggregate_csv_headers_seen += 1
                if not self._aggregate_payloads_enabled:
                    # Aggregate manifests mix groups.  Their split cannot be
                    # established from the member name, so scoped mode must
                    # not open or cache them at all.
                    continue
                updated = basename.lower().startswith("updated_")
                canonical_name = basename[len("updated_") :] if updated else basename
                aggregate_candidates.setdefault(canonical_name, []).append((updated, name))
                self._cache_csv_payload(
                    archive, member, name, payload_kind="aggregate"
                )
            self._container_header_scan_completed = True
        finally:
            archive.close()
            if (
                source_stream is not None
                and original_position is not None
                and hasattr(source_stream, "seek")
            ):
                try:
                    source_stream.seek(original_position)
                except (OSError, io.UnsupportedOperation):
                    pass

        self._aggregate_members: Dict[str, Tuple[str, bool]] = {}
        skipped: List[str] = []
        for manifest_id, candidates in aggregate_candidates.items():
            candidates.sort(key=lambda item: (item[0], len(item[1]), item[1]))
            selected_updated, selected_name = candidates[0]
            self._aggregate_members[manifest_id] = (selected_name, selected_updated)
            skipped.extend(name for _updated, name in candidates[1:])
        self._skipped_updated_manifests = tuple(sorted(skipped))
        for member_name in self._skipped_updated_manifests:
            self._csv_payloads.pop(member_name, None)

    def _cache_csv_payload(
        self,
        archive: tarfile.TarFile,
        member: tarfile.TarInfo,
        member_name: str,
        *,
        payload_kind: str,
    ) -> None:
        if payload_kind == "group":
            self._selected_group_csv_extractfile_calls += 1
        elif payload_kind == "aggregate":
            self._aggregate_csv_extractfile_calls += 1
        else:
            raise AssertionError("unknown ECCV CSV payload kind")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ECCVParseError("cannot read archive member {}".format(member_name))
        try:
            if payload_kind == "group":
                self._selected_group_csv_payload_read_calls += 1
            else:
                self._aggregate_csv_payload_read_calls += 1
            self._csv_payloads[member_name] = extracted.read()
        finally:
            extracted.close()
        if payload_kind == "group":
            self._selected_group_csv_payloads_cached += 1
            self._group_csv_members.add(member_name)
        else:
            self._aggregate_csv_payloads_cached += 1
            self._aggregate_csv_members.add(member_name)

    @property
    def pair_group_payload_scope(self) -> Optional[frozenset]:
        """Exact canonical pair-group allowlist, or ``None`` for legacy mode."""

        return self._pair_group_payload_scope

    @property
    def aggregate_payloads_enabled(self) -> bool:
        """Whether cross-group aggregate CSV payload access was permitted."""

        return self._aggregate_payloads_enabled

    @property
    def discovered_pair_group_ids(self) -> Tuple[str, ...]:
        """Pair-group identities observed from tar member names only."""

        return tuple(sorted(self._discovered_pair_group_ids))

    def read_scope_evidence(self) -> ECCVReadScopeEvidence:
        """Return an immutable snapshot of application payload exposure."""

        if not self._container_header_scan_completed:
            raise ECCVParseError("ECCV tar header scan did not complete")
        if self._pair_group_payload_scope is None:
            mode = "legacy_unscoped"
            selected_present = len(self._discovered_pair_group_ids)
            allowed = None
        else:
            mode = "split_scoped"
            selected_present = len(
                self._discovered_pair_group_ids.intersection(
                    self._pair_group_payload_scope
                )
            )
            allowed = tuple(sorted(self._pair_group_payload_scope))
            forbidden_counts = (
                self._nonselected_group_csv_extractfile_calls,
                self._nonselected_group_csv_payload_read_calls,
                self._nonselected_group_csv_payloads_cached,
                self._nonselected_group_csv_decode_calls,
                self._nonselected_group_csv_parse_calls,
                self._nonselected_fragment_extractfile_calls,
                self._aggregate_csv_extractfile_calls,
                self._aggregate_csv_payload_read_calls,
                self._aggregate_csv_payloads_cached,
                self._aggregate_csv_decode_calls,
                self._aggregate_csv_parse_calls,
            )
            if any(forbidden_counts) or self._aggregate_payloads_enabled:
                raise ECCVParseError(
                    "split-scoped ECCV archive violated its payload boundary"
                )
        return ECCVReadScopeEvidence(
            mode=mode,
            allowed_pair_group_ids=allowed,
            aggregate_payloads_enabled=self._aggregate_payloads_enabled,
            container_header_scan_completed=True,
            tar_member_headers_scanned=self._tar_member_headers_scanned,
            discovered_pair_group_count=len(self._discovered_pair_group_ids),
            selected_pair_group_count_present=selected_present,
            selected_group_csv_headers_seen=(
                self._selected_group_csv_headers_seen
            ),
            selected_group_csv_extractfile_calls=(
                self._selected_group_csv_extractfile_calls
            ),
            selected_group_csv_payload_read_calls=(
                self._selected_group_csv_payload_read_calls
            ),
            selected_group_csv_payloads_cached=(
                self._selected_group_csv_payloads_cached
            ),
            selected_group_csv_decode_calls=(
                self._selected_group_csv_decode_calls
            ),
            selected_group_csv_parse_calls=(
                self._selected_group_csv_parse_calls
            ),
            nonselected_group_csv_headers_seen=(
                self._nonselected_group_csv_headers_seen
            ),
            nonselected_group_csv_extractfile_calls=(
                self._nonselected_group_csv_extractfile_calls
            ),
            nonselected_group_csv_payload_read_calls=(
                self._nonselected_group_csv_payload_read_calls
            ),
            nonselected_group_csv_payloads_cached=(
                self._nonselected_group_csv_payloads_cached
            ),
            nonselected_group_csv_decode_calls=(
                self._nonselected_group_csv_decode_calls
            ),
            nonselected_group_csv_parse_calls=(
                self._nonselected_group_csv_parse_calls
            ),
            selected_fragment_headers_seen=self._selected_fragment_headers_seen,
            selected_fragment_extractfile_calls=(
                self._selected_fragment_extractfile_calls
            ),
            nonselected_fragment_headers_seen=(
                self._nonselected_fragment_headers_seen
            ),
            nonselected_fragment_extractfile_calls=(
                self._nonselected_fragment_extractfile_calls
            ),
            aggregate_csv_headers_seen=self._aggregate_csv_headers_seen,
            aggregate_csv_extractfile_calls=(
                self._aggregate_csv_extractfile_calls
            ),
            aggregate_csv_payload_read_calls=(
                self._aggregate_csv_payload_read_calls
            ),
            aggregate_csv_payloads_cached=self._aggregate_csv_payloads_cached,
            aggregate_csv_decode_calls=self._aggregate_csv_decode_calls,
            aggregate_csv_parse_calls=self._aggregate_csv_parse_calls,
        )

    @property
    def group_profiles(self) -> Tuple[str, ...]:
        present = {profile for profile, _group_id in self._groups}
        return tuple(profile for profile in ("2x2", "2x2-negative", "3x3") if profile in present)

    @property
    def skipped_updated_manifests(self) -> Tuple[str, ...]:
        return self._skipped_updated_manifests

    def group_ids(self, profile: str) -> Tuple[str, ...]:
        canonical = _normalise_profile(profile)

        def key(value: str) -> Tuple[int, Any]:
            return (0, int(value)) if value.isdigit() else (1, value)

        return tuple(sorted((group_id for item_profile, group_id in self._groups if item_profile == canonical), key=key))

    def _csv_rows(
        self, member_name: Optional[str]
    ) -> Iterator[Tuple[str, str, str, int]]:
        if member_name is None:
            return
        try:
            payload = self._csv_payloads[member_name]
        except KeyError as error:
            raise ECCVParseError(
                "CSV payload was not cached during archive scan: {}".format(member_name)
            ) from error
        if member_name in self._group_csv_members:
            payload_kind = "group"
            self._selected_group_csv_decode_calls += 1
        elif member_name in self._aggregate_csv_members:
            payload_kind = "aggregate"
            self._aggregate_csv_decode_calls += 1
        else:
            raise ECCVParseError(
                "CSV member is outside the registered payload scope: {}".format(
                    member_name
                )
            )
        text: Optional[str] = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                text = payload.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise ECCVParseError("cannot decode archive member {}".format(member_name))
        if payload_kind == "group":
            self._selected_group_csv_parse_calls += 1
        else:
            self._aggregate_csv_parse_calls += 1
        reader = csv.reader(io.StringIO(text))
        try:
            first_row = next(reader)
        except StopIteration:
            return
        header = [cell.strip().lower() for cell in first_row]
        has_header = len(header) >= 3 and header[0] in {"pair_1", "pair1", "fragment_a", "image1"}
        if has_header:
            data_rows: Iterator[List[str]] = reader
            starting_row = 2
        else:
            data_rows = chain((first_row,), reader)
            starting_row = 1
        for index, row in enumerate(data_rows, start=starting_row):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < 3:
                raise ECCVParseError("{} row {} has fewer than three columns".format(member_name, index))
            yield row[0].strip(), row[1].strip(), row[2].strip(), index

    def _group_from_cache(self, profile: str, group_id: str) -> GroupRecord:
        bucket = self._groups[(profile, group_id)]
        positive_by_key: Dict[PairKey, List[PairRecord]] = {}
        negative_by_key: Dict[PairKey, List[PairRecord]] = {}
        raw_positive_rows = 0
        raw_false_rows = 0

        def consume(member_name: Optional[str], expected_false: bool) -> None:
            nonlocal raw_positive_rows, raw_false_rows
            for left, right, raw_condition, row_number in self._csv_rows(member_name):
                is_adjacent, direction = _condition(raw_condition, member_name or "", row_number)
                if expected_false and is_adjacent:
                    raise ECCVParseError(
                        "{} row {} is positive inside pair_false.csv".format(member_name, row_number)
                    )
                record = PairRecord(
                    fragment_a_id=_group_fragment_id(profile, group_id, left),
                    fragment_b_id=_group_fragment_id(profile, group_id, right),
                    is_adjacent=is_adjacent,
                    direction=direction,
                    group_id=group_id,
                    profile=profile,
                    source_member=member_name or "",
                    row_number=row_number,
                    raw_condition=raw_condition,
                ).canonicalised()
                if record.is_adjacent:
                    raw_positive_rows += 1
                    positive_by_key.setdefault(record.canonical_key, []).append(record)
                else:
                    raw_false_rows += 1
                    negative_by_key.setdefault(record.canonical_key, []).append(record)

        consume(bucket.get("pair"), expected_false=False)
        consume(bucket.get("pair_false"), expected_false=True)

        pairs: List[PairRecord] = []
        conflicts: List[PairConflict] = []
        all_keys = sorted(set(positive_by_key) | set(negative_by_key), key=lambda item: (_fragment_sort_key(item[0]), _fragment_sort_key(item[1])))
        for key in all_keys:
            positives = positive_by_key.get(key, [])
            negatives = negative_by_key.get(key, [])
            if positives:
                pairs.append(positives[0])
                if negatives:
                    conflicts.append(
                        PairConflict(
                            canonical_pair=key,
                            reason="positive_reverse_present_in_false_labels",
                            positive_rows=tuple(positives),
                            negative_rows=tuple(negatives),
                        )
                    )
                continue
            if negatives:
                pairs.append(negatives[0])

        fragments = {
            normalise_fragment_path(member_name)
            for member_name in bucket.get("fragments", [])
        }
        for pair in pairs:
            fragments.add(pair.fragment_a_id)
            fragments.add(pair.fragment_b_id)

        excluded = profile == "3x3"
        exclusion_reason = (
            "eccv_3x3_template_leakage_solver_smoke_only" if excluded else None
        )
        metadata = {
            "generalization_eligible": not excluded,
            "allowed_use": "solver_smoke_only" if excluded else "benchmark_candidate",
            "native_boundary_ground_truth": False,
            "native_relative_pose_ground_truth": False,
        }
        return GroupRecord(
            profile=profile,
            group_id=group_id,
            member_prefix=bucket["prefix"],
            fragment_ids=tuple(sorted(fragments, key=_fragment_sort_key)),
            pairs=tuple(pairs),
            conflicts=tuple(conflicts),
            raw_positive_rows=raw_positive_rows,
            raw_false_rows=raw_false_rows,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
            metadata=metadata,
        )

    def group(self, profile: str, group_id: Union[str, int]) -> GroupRecord:
        canonical = _normalise_profile(profile)
        key = (canonical, str(group_id))
        if key not in self._groups:
            raise KeyError("ECCV group not found: {}/{}".format(canonical, group_id))
        return self._group_from_cache(canonical, str(group_id))

    get_group = group

    def groups(self, profile: Optional[str] = None) -> Iterator[GroupRecord]:
        profiles = self.group_profiles if profile is None else (_normalise_profile(profile),)
        for item_profile in profiles:
            for group_id in self.group_ids(item_profile):
                yield self._group_from_cache(item_profile, group_id)

    iter_groups = groups

    def aggregate_manifest_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._aggregate_members))

    def _resolve_manifest_id(self, manifest_id: str) -> str:
        basename = posixpath.basename(str(manifest_id))
        if basename.lower().startswith("updated_"):
            basename = basename[len("updated_") :]
        if not basename.lower().endswith(".csv"):
            basename += ".csv"
        if basename not in self._aggregate_members:
            raise KeyError("ECCV aggregate manifest not found: {}".format(manifest_id))
        return basename

    def _aggregate_from_cache(self, manifest_id: str) -> AggregateManifest:
        member_path, updated_fallback = self._aggregate_members[manifest_id]
        parsed: List[PairRecord] = []
        positive_orders: Dict[PairKey, set] = {}
        raw_rows = self._csv_rows(member_path)
        for left, right, raw_condition, row_number in raw_rows:
            fragment_a = normalise_fragment_path(left)
            fragment_b = normalise_fragment_path(right)
            is_adjacent, direction = _condition(raw_condition, member_path, row_number)
            relation = _known_grid_relation(fragment_a, fragment_b)
            if is_adjacent and direction is None and relation is not None:
                direction = relation[0]
            identity_a = _parse_fragment_identity(fragment_a)
            identity_b = _parse_fragment_identity(fragment_b)
            group_id = None
            profile = None
            if identity_a is not None and identity_b is not None and identity_a[:2] == identity_b[:2]:
                profile, group_id = identity_a[:2]
            record = PairRecord(
                fragment_a_id=fragment_a,
                fragment_b_id=fragment_b,
                is_adjacent=is_adjacent,
                direction=direction,
                group_id=group_id,
                profile=profile,
                source_member=member_path,
                row_number=row_number,
                raw_condition=raw_condition,
                phase="aggregate",
            )
            parsed.append(record)
            if is_adjacent:
                positive_orders.setdefault(record.canonical_key, set()).add(
                    (record.fragment_a_id, record.fragment_b_id)
                )

        records: List[PairRecord] = []
        for record in parsed:
            if record.is_adjacent:
                records.append(record)
                continue
            relation = _known_grid_relation(record.fragment_a_id, record.fragment_b_id)
            observed_positive = record.canonical_key in positive_orders
            conflict_kind: Optional[str] = None
            if relation is not None or observed_positive:
                direct = False
                if relation is not None:
                    direct = relation[1]
                elif (
                    record.fragment_a_id,
                    record.fragment_b_id,
                ) in positive_orders.get(record.canonical_key, set()):
                    direct = True
                conflict_kind = (
                    "positive_edge_labeled_false"
                    if direct
                    else "reverse_positive_edge_labeled_false"
                )
            if conflict_kind is None:
                records.append(record)
                continue
            records.append(
                PairRecord(
                    fragment_a_id=record.fragment_a_id,
                    fragment_b_id=record.fragment_b_id,
                    is_adjacent=False,
                    direction=None,
                    group_id=record.group_id,
                    profile=record.profile,
                    source_member=record.source_member,
                    row_number=record.row_number,
                    raw_condition=record.raw_condition,
                    phase=record.phase,
                    excluded=True,
                    exclusion_reason="canonical_undirected_pair_is_positive",
                    conflict_kind=conflict_kind,
                    metadata={"usable_as_negative": False},
                )
            )

        return AggregateManifest(
            manifest_id=manifest_id,
            member_path=member_path,
            records=tuple(records),
            updated_fallback=updated_fallback,
        )

    def aggregate_manifest(self, manifest_id: str) -> AggregateManifest:
        canonical = self._resolve_manifest_id(manifest_id)
        return self._aggregate_from_cache(canonical)

    get_aggregate_manifest = aggregate_manifest

    def aggregate_manifests(self) -> Iterator[AggregateManifest]:
        for manifest_id in self.aggregate_manifest_ids():
            yield self._aggregate_from_cache(manifest_id)

    iter_aggregate_manifests = aggregate_manifests

    def _aggregate_profile_from_cache(
        self, manifest_id: str
    ) -> AggregateManifestProfile:
        """Profile one manifest without materialising hundreds of thousands of records."""

        member_path, updated_fallback = self._aggregate_members[manifest_id]
        positive_orders: Dict[PairKey, set] = {}
        false_rows: List[Tuple[str, str, Optional[Tuple[str, bool]], PairKey]] = []
        raw_count = 0
        positive_count = 0
        for left, right, raw_condition, row_number in self._csv_rows(member_path):
            raw_count += 1
            fragment_a = normalise_fragment_path(left)
            fragment_b = normalise_fragment_path(right)
            is_adjacent, _direction = _condition(
                raw_condition, member_path, row_number
            )
            key = canonical_pair_key(fragment_a, fragment_b)
            if is_adjacent:
                positive_count += 1
                positive_orders.setdefault(key, set()).add(
                    (fragment_a, fragment_b)
                )
                continue
            false_rows.append(
                (
                    fragment_a,
                    fragment_b,
                    _known_grid_relation(fragment_a, fragment_b),
                    key,
                )
            )

        direct_conflicts = 0
        reverse_conflicts = 0
        for fragment_a, fragment_b, relation, key in false_rows:
            observed_orders = positive_orders.get(key)
            if relation is None and observed_orders is None:
                continue
            if relation is not None:
                direct = relation[1]
            else:
                direct = (fragment_a, fragment_b) in observed_orders
            if direct:
                direct_conflicts += 1
            else:
                reverse_conflicts += 1

        conflict_count = direct_conflicts + reverse_conflicts
        return AggregateManifestProfile(
            manifest_id=manifest_id,
            member_path=member_path,
            raw_rows=raw_count,
            positive_rows=positive_count,
            false_rows=len(false_rows),
            usable_negative_rows=len(false_rows) - conflict_count,
            undirected_conflict_rows=conflict_count,
            direct_false_pollution_rows=direct_conflicts,
            reverse_positive_false_rows=reverse_conflicts,
            updated_fallback=updated_fallback,
        )

    def profile(self) -> ECCVArchiveProfile:
        group_counts: Dict[str, int] = {}
        group_positives: Dict[str, int] = {}
        group_negatives: Dict[str, int] = {}
        group_conflicts: Dict[str, int] = {}
        for group in self.groups():
            profile = group.profile
            group_counts[profile] = group_counts.get(profile, 0) + 1
            group_positives[profile] = group_positives.get(profile, 0) + group.positive_count
            group_negatives[profile] = group_negatives.get(profile, 0) + group.usable_negative_count
            group_conflicts[profile] = group_conflicts.get(profile, 0) + group.suppressed_negative_count

        aggregate_profiles = [
            self._aggregate_profile_from_cache(manifest_id)
            for manifest_id in self.aggregate_manifest_ids()
        ]
        return ECCVArchiveProfile(
            group_counts=group_counts,
            group_positive_pairs=group_positives,
            group_usable_negative_pairs=group_negatives,
            group_conflict_rows=group_conflicts,
            aggregate_manifest_count=len(aggregate_profiles),
            aggregate_raw_rows=sum(item.raw_rows for item in aggregate_profiles),
            aggregate_positive_rows=sum(item.positive_rows for item in aggregate_profiles),
            aggregate_false_rows=sum(item.false_rows for item in aggregate_profiles),
            aggregate_usable_negative_rows=sum(item.usable_negative_rows for item in aggregate_profiles),
            aggregate_undirected_conflict_rows=sum(item.undirected_conflict_rows for item in aggregate_profiles),
            aggregate_direct_false_pollution_rows=sum(item.direct_false_pollution_rows for item in aggregate_profiles),
            aggregate_reverse_positive_false_rows=sum(item.reverse_positive_false_rows for item in aggregate_profiles),
            skipped_updated_manifests=self.skipped_updated_manifests,
        )


def parse_eccv_archive(source: ArchiveInput) -> ECCVArchive:
    """Create an indexed read-only archive view."""

    return ECCVArchive(source)


def profile_eccv_archive(source: ArchiveInput) -> ECCVArchiveProfile:
    """Profile group and aggregate labels without extracting the archive."""

    return ECCVArchive(source).profile()


__all__ = [
    "AggregateManifest",
    "AggregateManifestProfile",
    "ECCVArchive",
    "ECCVArchiveProfile",
    "ECCVParseError",
    "ECCVReadScopeEvidence",
    "ECCV_READ_SCOPE_SCHEMA_VERSION",
    "GroupRecord",
    "PairConflict",
    "PairRecord",
    "canonical_pair_key",
    "normalise_fragment_path",
    "parse_eccv_archive",
    "profile_eccv_archive",
    "swap_direction",
]
