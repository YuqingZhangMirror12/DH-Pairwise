"""Manifest-bound real-data streams for the B0 Siamese evaluation harness.

The adapter is intentionally read-only and group-streaming.  It consumes the
canonical MM/ECCV parser APIs and a provisional Pairwise v0.1 split manifest,
then yields one :class:`RealPairRecord` at a time.  It never expands or writes
the complete pair population and has no checkpoint/model-loading capability.

Every emitted record inherits its split from an explicit *group* assignment
and its leakage cluster from an explicit group-to-component assignment.  There
is no fallback to a component id, hash, or random split.  Missing or invalid
assignments therefore fail closed.  ECCV 3x3 is excluded from the pairwise
stream even if a malformed manifest attempts to assign it.
"""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from staging.pairwise_v0_1.pairwise_data.eccv import (
    ECCVArchive,
    ECCVReadScopeEvidence,
    GroupRecord as ECCVGroupRecord,
    PairRecord as ECCVPairRecord,
)
from staging.pairwise_v0_1.pairwise_data.mm import MMCanonicalZip, ZipSource
from staging.pairwise_v0_1.pairwise_data.schema import (
    DirectedRelation,
    PairRecord as MMPairRecord,
)


STREAM_SCHEMA_VERSION = "pairwise-b0-real-stream/0.1"
ECCV_STREAM_EXPOSURE_SCHEMA_VERSION = "pairwise-eccv-stream-exposure/0.1"
MANIFEST_SCHEMA_VERSION = "pairwise-real-data-gate/0.1"
VALID_SPLITS = frozenset(("train", "val", "test"))
VALID_DATASETS = frozenset(("mm_augmented", "eccv_1113data"))
VALID_PROFILES = frozenset(("mm-augmented", "2x2", "2x2-negative"))

ManifestSource = Union[str, Path, Mapping[str, Any]]
ECCVSource = Union[str, Path, bytes, bytearray, BinaryIO, ECCVArchive]


class PairStreamError(ValueError):
    """Base exception for fail-closed stream construction or iteration."""


class SplitManifestError(PairStreamError):
    """Raised when a group cannot safely inherit an explicit split."""


class ExcludedProfileError(PairStreamError):
    """Raised when ECCV 3x3 is explicitly requested or assigned."""


@dataclass(frozen=True)
class ECCVSplitExposureEvidence:
    """Immutable split-stream evidence backed by an archive scope snapshot."""

    selected_splits: Tuple[str, ...]
    selected_profiles: Tuple[str, ...]
    expected_pair_group_scope: Tuple[str, ...]
    manifest_test_pair_group_count: int
    records_emitted: int
    nonselected_split_records_emitted: int
    test_records_emitted: int
    archive_read_scope: ECCVReadScopeEvidence
    schema_version: str = ECCV_STREAM_EXPOSURE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        test_is_held_out = "test" not in self.selected_splits
        archive = self.archive_read_scope
        return {
            "schema_version": self.schema_version,
            "selected_splits": list(self.selected_splits),
            "selected_profiles": list(self.selected_profiles),
            "expected_pair_group_scope": list(
                self.expected_pair_group_scope
            ),
            "manifest_test_pair_group_count": (
                self.manifest_test_pair_group_count
            ),
            "records_emitted": self.records_emitted,
            "nonselected_split_records_emitted": (
                self.nonselected_split_records_emitted
            ),
            "test_records_emitted": self.test_records_emitted,
            "archive_read_scope": archive.to_dict(),
            "held_out_test_guarantee": {
                "applicable": test_is_held_out,
                "test_group_csv_extractfile_calls": (
                    archive.nonselected_group_csv_extractfile_calls
                    if test_is_held_out
                    else None
                ),
                "test_group_csv_payloads_cached": (
                    archive.nonselected_group_csv_payloads_cached
                    if test_is_held_out
                    else None
                ),
                "test_group_csv_payload_read_calls": (
                    archive.nonselected_group_csv_payload_read_calls
                    if test_is_held_out
                    else None
                ),
                "test_group_csv_decode_calls": (
                    archive.nonselected_group_csv_decode_calls
                    if test_is_held_out
                    else None
                ),
                "test_group_csv_parse_calls": (
                    archive.nonselected_group_csv_parse_calls
                    if test_is_held_out
                    else None
                ),
                "test_fragment_extractfile_calls": (
                    archive.nonselected_fragment_extractfile_calls
                    if test_is_held_out
                    else None
                ),
                "test_records_emitted": (
                    self.test_records_emitted if test_is_held_out else None
                ),
            },
        }


class ECCVSplitExposureAudit:
    """One-use audit handle for a split-scoped ECCV pair iterator."""

    def __init__(self) -> None:
        self._archive: Optional[ECCVArchive] = None
        self._selected_splits: Tuple[str, ...] = ()
        self._selected_profiles: Tuple[str, ...] = ()
        self._expected_scope: Tuple[str, ...] = ()
        self._manifest_test_pair_group_count = 0
        self._records_emitted = 0
        self._nonselected_split_records_emitted = 0
        self._test_records_emitted = 0

    def _bind(
        self,
        archive: ECCVArchive,
        *,
        selected_splits: frozenset,
        selected_profiles: frozenset,
        expected_scope: frozenset,
        manifest_test_pair_group_count: int,
    ) -> None:
        if self._archive is not None:
            raise PairStreamError("ECCV exposure audit cannot be reused")
        self._archive = archive
        self._selected_splits = tuple(sorted(selected_splits))
        self._selected_profiles = tuple(sorted(selected_profiles))
        self._expected_scope = tuple(sorted(expected_scope))
        self._manifest_test_pair_group_count = manifest_test_pair_group_count

    def _record(self, record: "RealPairRecord") -> None:
        self._records_emitted += 1
        if record.split not in self._selected_splits:
            self._nonselected_split_records_emitted += 1
        if record.split == "test":
            self._test_records_emitted += 1

    def evidence(self) -> ECCVSplitExposureEvidence:
        if self._archive is None:
            raise PairStreamError(
                "ECCV exposure audit is not bound to an archive scan"
            )
        archive_evidence = self._archive.read_scope_evidence()
        if archive_evidence.mode != "split_scoped":
            raise PairStreamError("ECCV exposure audit requires split-scoped mode")
        if tuple(archive_evidence.allowed_pair_group_ids or ()) != (
            self._expected_scope
        ):
            raise PairStreamError("ECCV exposure scope changed after binding")
        if self._nonselected_split_records_emitted:
            raise PairStreamError(
                "ECCV split-scoped iterator emitted a nonselected split"
            )
        if "test" not in self._selected_splits and self._test_records_emitted:
            raise PairStreamError(
                "ECCV held-out test record entered a non-test iterator"
            )
        return ECCVSplitExposureEvidence(
            selected_splits=self._selected_splits,
            selected_profiles=self._selected_profiles,
            expected_pair_group_scope=self._expected_scope,
            manifest_test_pair_group_count=(
                self._manifest_test_pair_group_count
            ),
            records_emitted=self._records_emitted,
            nonselected_split_records_emitted=(
                self._nonselected_split_records_emitted
            ),
            test_records_emitted=self._test_records_emitted,
            archive_read_scope=archive_evidence,
        )


@dataclass(frozen=True)
class SplitManifestView:
    """Validated, immutable inheritance view over a provisional manifest."""

    candidate_id: str
    status: str
    authorization: str
    group_assignments: Mapping[str, str]
    group_components: Mapping[str, str]
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # ``frozen=True`` only protects attributes, not a caller-owned dict.
        # Defensive copies make both public inheritance maps deeply immutable.
        object.__setattr__(
            self,
            "group_assignments",
            MappingProxyType(dict(self.group_assignments)),
        )
        object.__setattr__(
            self,
            "group_components",
            MappingProxyType(dict(self.group_components)),
        )

    def split_for_group(self, canonical_group_id: str) -> str:
        try:
            return self.group_assignments[canonical_group_id]
        except KeyError as exc:
            raise SplitManifestError(
                "group {!r} has no explicit split assignment".format(
                    canonical_group_id
                )
            ) from exc

    def component_for_group(self, canonical_group_id: str) -> str:
        """Return the explicit leakage component for ``canonical_group_id``."""

        try:
            return self.group_components[canonical_group_id]
        except KeyError as exc:
            raise SplitManifestError(
                "group {!r} has no explicit component assignment".format(
                    canonical_group_id
                )
            ) from exc


@dataclass(frozen=True)
class ArchiveImageRef:
    """A lazy image identity for an archive-aware ``image_loader`` callback."""

    dataset_id: str
    archive_format: str
    archive_member: str
    fragment_id: str
    canonical_group_id: str
    cluster_id: str
    split: str
    profile: str
    source_id: Optional[str] = None
    variant_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.dataset_id not in VALID_DATASETS:
            raise ValueError("unsupported image dataset_id")
        if self.archive_format not in {"zip", "tar"}:
            raise ValueError("archive_format must be zip or tar")
        if not self.archive_member or not self.fragment_id or not self.cluster_id:
            raise ValueError(
                "image references require member, fragment, and cluster ids"
            )
        if self.split not in VALID_SPLITS:
            raise ValueError("image reference has an invalid split")
        if self.profile not in VALID_PROFILES:
            raise ValueError("image reference has an invalid profile")


@dataclass(frozen=True)
class RealPairRecord:
    """B0-compatible ordered pair plus split and archive provenance."""

    fragment_a: ArchiveImageRef
    fragment_b: ArchiveImageRef
    is_adjacent: bool
    direction_b_wrt_a: Optional[DirectedRelation]
    dataset_id: str
    canonical_group_id: str
    cluster_id: str
    split: str
    profile: str
    source_id: Optional[str]
    variant_id: Optional[str]
    canonical_pair_key: Tuple[str, str]
    source_member: str
    source_row_number: int
    condition_raw: str
    manifest_candidate_id: str
    manifest_status: str
    manifest_authorization: str
    derived_reverse: bool = False
    direction_origin: str = "historical_label"
    schema_version: str = STREAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.is_adjacent) is not bool:  # noqa: E721 - exact bool required
            raise TypeError("is_adjacent must be an explicit bool")
        if self.is_adjacent != (self.direction_b_wrt_a is not None):
            raise ValueError(
                "positive records require a direction; negatives require null"
            )
        if self.dataset_id not in VALID_DATASETS:
            raise ValueError("unsupported record dataset_id")
        if self.split not in VALID_SPLITS:
            raise ValueError("record split is invalid")
        if self.profile not in VALID_PROFILES:
            raise ValueError("record profile is invalid")
        if not self.cluster_id:
            raise ValueError("record cluster_id is required")
        if self.source_row_number <= 0:
            raise ValueError("source_row_number must be positive")
        if self.fragment_a.fragment_id == self.fragment_b.fragment_id:
            raise ValueError("self-pairs are not valid")
        expected_key = tuple(
            sorted((self.fragment_a.fragment_id, self.fragment_b.fragment_id))
        )
        if self.canonical_pair_key != expected_key:
            raise ValueError("canonical_pair_key does not match image references")
        for fragment in (self.fragment_a, self.fragment_b):
            if (
                fragment.dataset_id != self.dataset_id
                or fragment.canonical_group_id != self.canonical_group_id
                or fragment.cluster_id != self.cluster_id
                or fragment.split != self.split
                or fragment.profile != self.profile
            ):
                raise ValueError("fragment provenance differs from pair provenance")
        if not self.manifest_candidate_id or not self.manifest_status:
            raise ValueError("manifest provenance is required")
        if self.direction_origin not in {"historical_label", "derived_inverse"}:
            raise ValueError("unsupported direction_origin")

    @property
    def pair_id(self) -> str:
        return "{}:{}::{}".format(
            self.dataset_id,
            self.canonical_pair_key[0],
            self.canonical_pair_key[1],
        )

    @property
    def provenance(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_id": self.source_id,
            "canonical_group_id": self.canonical_group_id,
            "cluster_id": self.cluster_id,
            "split": self.split,
            "profile": self.profile,
            "variant_id": self.variant_id,
            "source_member": self.source_member,
            "source_row_number": self.source_row_number,
            "condition_raw": self.condition_raw,
            "manifest_candidate_id": self.manifest_candidate_id,
            "manifest_status": self.manifest_status,
            "manifest_authorization": self.manifest_authorization,
            "derived_reverse": self.derived_reverse,
            "direction_origin": self.direction_origin,
        }

    def reversed(self) -> "RealPairRecord":
        """Return the explicit inverse ordered view of this physical pair."""

        derived_reverse = not self.derived_reverse
        return RealPairRecord(
            fragment_a=self.fragment_b,
            fragment_b=self.fragment_a,
            is_adjacent=self.is_adjacent,
            direction_b_wrt_a=(
                self.direction_b_wrt_a.inverse()
                if self.direction_b_wrt_a is not None
                else None
            ),
            dataset_id=self.dataset_id,
            canonical_group_id=self.canonical_group_id,
            cluster_id=self.cluster_id,
            split=self.split,
            profile=self.profile,
            source_id=self.source_id,
            variant_id=self.variant_id,
            canonical_pair_key=self.canonical_pair_key,
            source_member=self.source_member,
            source_row_number=self.source_row_number,
            condition_raw=self.condition_raw,
            manifest_candidate_id=self.manifest_candidate_id,
            manifest_status=self.manifest_status,
            manifest_authorization=self.manifest_authorization,
            derived_reverse=derived_reverse,
            direction_origin=(
                "derived_inverse" if derived_reverse else "historical_label"
            ),
            schema_version=self.schema_version,
        )


def load_split_manifest(source: ManifestSource) -> SplitManifestView:
    """Load compact group/split/component inheritance; never load pair rows."""

    if isinstance(source, Mapping):
        payload: Mapping[str, Any] = source
    else:
        with Path(source).open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, Mapping):
            raise SplitManifestError("split manifest root must be a JSON object")
        payload = loaded

    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SplitManifestError("unsupported split manifest schema_version")
    status = str(payload.get("status", "")).strip()
    if status not in {"provisional", "complete"}:
        raise SplitManifestError("manifest status must be provisional or complete")
    candidate_id = str(payload.get("candidate_id", "")).strip()
    authorization = str(payload.get("authorization", "")).strip()
    if not candidate_id or not authorization:
        raise SplitManifestError("candidate_id and authorization are required")
    if payload.get("pair_rows_materialized") is not False:
        raise SplitManifestError(
            "real stream requires compact group inheritance, not pair rows"
        )
    assignments = payload.get("assignments")
    if not isinstance(assignments, Mapping):
        raise SplitManifestError("manifest requires assignments")
    groups = assignments.get("groups")
    if not isinstance(groups, Mapping) or not groups:
        raise SplitManifestError("manifest requires non-empty assignments.groups")
    components = assignments.get("components")
    if not isinstance(components, Mapping) or not components:
        raise SplitManifestError(
            "manifest requires non-empty assignments.components"
        )
    raw_group_components = assignments.get("group_components")
    if not isinstance(raw_group_components, Mapping):
        raise SplitManifestError("manifest requires assignments.group_components")

    copied: Dict[str, str] = {}
    for raw_group_id, raw_split in groups.items():
        group_id = str(raw_group_id).strip()
        split = str(raw_split).strip().casefold()
        if not group_id or split not in VALID_SPLITS:
            raise SplitManifestError("invalid group assignment")
        if group_id in copied:
            raise SplitManifestError("duplicate normalized group assignment")
        if group_id.startswith("eccv/3x3/"):
            raise ExcludedProfileError(
                "ECCV 3x3 cannot receive a Pairwise B0 split assignment"
            )
        copied[group_id] = split

    component_assignments: Dict[str, str] = {}
    for raw_component_id, raw_split in components.items():
        component_id = str(raw_component_id).strip()
        split = str(raw_split).strip().casefold()
        if not component_id or split not in VALID_SPLITS:
            raise SplitManifestError("invalid component assignment")
        if component_id in component_assignments:
            raise SplitManifestError("duplicate normalized component assignment")
        component_assignments[component_id] = split

    copied_group_components: Dict[str, str] = {}
    for raw_group_id, raw_component_id in raw_group_components.items():
        group_id = str(raw_group_id).strip()
        component_id = str(raw_component_id).strip()
        if not group_id or not component_id:
            raise SplitManifestError("invalid group component assignment")
        if group_id in copied_group_components:
            raise SplitManifestError(
                "duplicate normalized group component assignment"
            )
        copied_group_components[group_id] = component_id

    group_keys = set(copied)
    component_group_keys = set(copied_group_components)
    if group_keys != component_group_keys:
        missing = sorted(group_keys - component_group_keys)
        extra = sorted(component_group_keys - group_keys)
        detail = "missing={!r}, extra={!r}".format(
            missing[0] if missing else None,
            extra[0] if extra else None,
        )
        raise SplitManifestError(
            "assignments.group_components keys must exactly match groups ({})".format(
                detail
            )
        )

    observed_component_splits: Dict[str, str] = {}
    for group_id, component_id in copied_group_components.items():
        group_split = copied[group_id]
        prior_split = observed_component_splits.setdefault(
            component_id, group_split
        )
        if prior_split != group_split:
            raise SplitManifestError(
                "component {!r} spans multiple splits".format(component_id)
            )
        try:
            component_split = component_assignments[component_id]
        except KeyError as exc:
            raise SplitManifestError(
                "group {!r} references unknown component {!r}".format(
                    group_id, component_id
                )
            ) from exc
        if component_split != group_split:
            raise SplitManifestError(
                "group {!r} and component {!r} have different splits".format(
                    group_id, component_id
                )
            )
        if group_id.startswith("mm/base/"):
            required_prefix = "mm/source/"
        elif group_id.startswith(("eccv/2x2/", "eccv/2x2-negative/")):
            required_prefix = "eccv/signature/"
        else:
            raise SplitManifestError(
                "unsupported canonical group id: {!r}".format(group_id)
            )
        if not component_id.startswith(required_prefix):
            raise SplitManifestError(
                "group {!r} has incompatible component {!r}".format(
                    group_id, component_id
                )
            )

    excluded = payload.get("excluded_groups", [])
    if not isinstance(excluded, list):
        raise SplitManifestError("excluded_groups must be a list")
    excluded_ids = {
        str(item.get("canonical_group_id"))
        for item in excluded
        if isinstance(item, Mapping) and item.get("canonical_group_id") is not None
    }
    overlap = excluded_ids.intersection(copied)
    if overlap:
        raise SplitManifestError(
            "groups cannot be both assigned and excluded: {}".format(
                sorted(overlap)[0]
            )
        )
    return SplitManifestView(
        candidate_id=candidate_id,
        status=status,
        authorization=authorization,
        group_assignments=MappingProxyType(copied),
        group_components=MappingProxyType(copied_group_components),
    )


def _normalise_filter(
    value: Optional[Union[str, Iterable[str]]],
    aliases: Mapping[str, str],
    field_name: str,
) -> Optional[frozenset]:
    if value is None:
        return None
    values: Iterable[str] = (value,) if isinstance(value, str) else value
    normalised = set()
    for item in values:
        key = str(item).strip().casefold().replace("×", "x")
        try:
            normalised.add(aliases[key])
        except KeyError as exc:
            raise PairStreamError(
                "unsupported {} value: {!r}".format(field_name, item)
            ) from exc
    if not normalised:
        raise PairStreamError("{} filter cannot be empty".format(field_name))
    return frozenset(normalised)


def _split_filter(
    splits: Optional[Union[str, Iterable[str]]]
) -> Optional[frozenset]:
    aliases = {name: name for name in VALID_SPLITS}
    aliases.update({"validation": "val", "valid": "val"})
    return _normalise_filter(splits, aliases, "split")


def _dataset_filter(
    datasets: Optional[Union[str, Iterable[str]]]
) -> Optional[frozenset]:
    return _normalise_filter(
        datasets,
        {
            "mm": "mm_augmented",
            "mm_augmented": "mm_augmented",
            "eccv": "eccv_1113data",
            "eccv_1113data": "eccv_1113data",
        },
        "dataset",
    )


def _profile_filter(
    profiles: Optional[Union[str, Iterable[str]]]
) -> Optional[frozenset]:
    normalised = _normalise_filter(
        profiles,
        {
            "mm": "mm-augmented",
            "mm-augmented": "mm-augmented",
            "augmented": "mm-augmented",
            "2x2": "2x2",
            "2x2-negative": "2x2-negative",
            "2x2negative": "2x2-negative",
            "3x3": "3x3",
        },
        "profile",
    )
    if normalised is not None and "3x3" in normalised:
        raise ExcludedProfileError(
            "ECCV 3x3 is solver-smoke-only and cannot enter a B0 pair stream"
        )
    return normalised


_ECCV_PAIR_PROFILES = frozenset(("2x2", "2x2-negative"))


def _eccv_pair_group_profile(canonical_group_id: str) -> Optional[str]:
    if canonical_group_id.startswith("eccv/2x2-negative/"):
        profile = "2x2-negative"
        suffix = canonical_group_id[len("eccv/2x2-negative/") :]
    elif canonical_group_id.startswith("eccv/2x2/"):
        profile = "2x2"
        suffix = canonical_group_id[len("eccv/2x2/") :]
    elif canonical_group_id.startswith("eccv/3x3/"):
        raise ExcludedProfileError(
            "ECCV 3x3 cannot enter a Pairwise pair payload scope"
        )
    elif canonical_group_id.startswith("eccv/"):
        raise SplitManifestError(
            "unsupported canonical ECCV group id: {!r}".format(
                canonical_group_id
            )
        )
    else:
        return None
    if not suffix or "/" in suffix:
        raise SplitManifestError(
            "invalid canonical ECCV pair group id: {!r}".format(
                canonical_group_id
            )
        )
    return profile


def _eccv_manifest_pair_groups(
    manifest: SplitManifestView,
    selected_profiles: frozenset,
    selected_splits: Optional[frozenset] = None,
) -> frozenset:
    selected = set()
    for raw_group_id, assigned_split in manifest.group_assignments.items():
        group_id = str(raw_group_id)
        profile = _eccv_pair_group_profile(group_id)
        if profile is None or profile not in selected_profiles:
            continue
        if selected_splits is None or assigned_split in selected_splits:
            selected.add(group_id)
    return frozenset(selected)


def _validate_discovered_eccv_groups(
    archive: ECCVArchive,
    manifest: SplitManifestView,
    selected_profiles: frozenset,
) -> None:
    """Validate every relevant header identity before any CSV is parsed."""

    for group_id in archive.discovered_pair_group_ids:
        profile = _eccv_pair_group_profile(group_id)
        if profile not in selected_profiles:
            continue
        manifest.split_for_group(group_id)
        manifest.component_for_group(group_id)


def _manifest_view(source: Union[ManifestSource, SplitManifestView]) -> SplitManifestView:
    return source if isinstance(source, SplitManifestView) else load_split_manifest(source)


def _manifest_fields(manifest: SplitManifestView) -> Dict[str, str]:
    return {
        "manifest_candidate_id": manifest.candidate_id,
        "manifest_status": manifest.status,
        "manifest_authorization": manifest.authorization,
    }


def _mm_record(
    pair: MMPairRecord,
    split: str,
    cluster_id: str,
    manifest: SplitManifestView,
) -> RealPairRecord:
    group_key = pair.variant.group
    canonical_group_id = "mm/base/{}".format(group_key.id)

    def image_ref(fragment: Any) -> ArchiveImageRef:
        return ArchiveImageRef(
            dataset_id="mm_augmented",
            archive_format="zip",
            archive_member=fragment.member_path,
            fragment_id=fragment.fragment_id,
            canonical_group_id=canonical_group_id,
            cluster_id=cluster_id,
            split=split,
            profile="mm-augmented",
            source_id=group_key.source.source_id,
            variant_id=pair.variant.variant_id,
        )

    fragment_a = image_ref(pair.fragment_a)
    fragment_b = image_ref(pair.fragment_b)
    return RealPairRecord(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        is_adjacent=pair.is_adjacent,
        direction_b_wrt_a=pair.direction_b_wrt_a,
        dataset_id="mm_augmented",
        canonical_group_id=canonical_group_id,
        cluster_id=cluster_id,
        split=split,
        profile="mm-augmented",
        source_id=group_key.source.source_id,
        variant_id=pair.variant.variant_id,
        canonical_pair_key=tuple(
            sorted((fragment_a.fragment_id, fragment_b.fragment_id))
        ),
        source_member=pair.csv_member_path,
        source_row_number=pair.csv_row_number,
        condition_raw=pair.condition_raw,
        **_manifest_fields(manifest),
    )


_ECCV_DIRECTIONS = {
    "left-right": DirectedRelation.RIGHT,
    "right-left": DirectedRelation.LEFT,
    "up-down": DirectedRelation.BELOW,
    "down-up": DirectedRelation.ABOVE,
}


def _eccv_direction(pair: ECCVPairRecord) -> Optional[DirectedRelation]:
    if not pair.is_adjacent:
        if pair.direction is not None:
            raise PairStreamError("ECCV negative unexpectedly carries a direction")
        return None
    try:
        return _ECCV_DIRECTIONS[str(pair.direction)]
    except KeyError as exc:
        raise PairStreamError(
            "ECCV positive lacks a supported explicit direction: {!r}".format(
                pair.direction
            )
        ) from exc


def _eccv_record(
    group: ECCVGroupRecord,
    pair: ECCVPairRecord,
    split: str,
    cluster_id: str,
    manifest: SplitManifestView,
) -> RealPairRecord:
    canonical_group_id = group.canonical_group_id

    def image_ref(fragment_id: str) -> ArchiveImageRef:
        return ArchiveImageRef(
            dataset_id="eccv_1113data",
            archive_format="tar",
            archive_member=posixpath.join(
                group.member_prefix, posixpath.basename(fragment_id)
            ),
            fragment_id=fragment_id,
            canonical_group_id=canonical_group_id,
            cluster_id=cluster_id,
            split=split,
            profile=group.profile,
        )

    fragment_a = image_ref(pair.fragment_a_id)
    fragment_b = image_ref(pair.fragment_b_id)
    return RealPairRecord(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        is_adjacent=pair.is_adjacent,
        direction_b_wrt_a=_eccv_direction(pair),
        dataset_id="eccv_1113data",
        canonical_group_id=canonical_group_id,
        cluster_id=cluster_id,
        split=split,
        profile=group.profile,
        source_id=None,
        variant_id=None,
        canonical_pair_key=tuple(
            sorted((fragment_a.fragment_id, fragment_b.fragment_id))
        ),
        source_member=pair.source_member,
        source_row_number=pair.row_number,
        condition_raw=pair.raw_condition,
        **_manifest_fields(manifest),
    )


def _with_reverse(
    records: Iterable[RealPairRecord], include_reverse: bool
) -> Iterator[RealPairRecord]:
    for record in records:
        yield record
        if include_reverse:
            yield record.reversed()


def _bounded(
    records: Iterable[RealPairRecord], max_records: Optional[int]
) -> Iterator[RealPairRecord]:
    if max_records is not None and max_records < 0:
        raise PairStreamError("max_records cannot be negative")
    if max_records == 0:
        return
    emitted = 0
    for record in records:
        yield record
        emitted += 1
        if max_records is not None and emitted >= max_records:
            return


def _iter_mm_core(
    archive: MMCanonicalZip,
    manifest: SplitManifestView,
    selected_splits: Optional[frozenset],
) -> Iterator[RealPairRecord]:
    for group in archive.iter_groups():
        canonical_group_id = "mm/base/{}".format(group.key.id)
        split = manifest.split_for_group(canonical_group_id)
        cluster_id = manifest.component_for_group(canonical_group_id)
        expected_cluster_id = "mm/source/{}".format(group.key.source.source_id)
        if cluster_id != expected_cluster_id:
            raise SplitManifestError(
                "MM group {!r} must inherit source cluster {!r}, not {!r}".format(
                    canonical_group_id, expected_cluster_id, cluster_id
                )
            )
        if selected_splits is not None and split not in selected_splits:
            continue
        for pair in archive.iter_pairs(group=group.key, include_reverse=False):
            yield _mm_record(pair, split, cluster_id, manifest)


def iter_mm_pair_records(
    archive_source: Union[ZipSource, MMCanonicalZip],
    split_manifest: Union[ManifestSource, SplitManifestView],
    *,
    splits: Optional[Union[str, Iterable[str]]] = None,
    include_reverse: bool = False,
    max_records: Optional[int] = None,
) -> Iterator[RealPairRecord]:
    """Stream MM pairs; every augmentation variant inherits its base-group split."""

    if max_records is not None and max_records < 0:
        raise PairStreamError("max_records cannot be negative")
    manifest = _manifest_view(split_manifest)
    selected_splits = _split_filter(splits)

    def records() -> Iterator[RealPairRecord]:
        if isinstance(archive_source, MMCanonicalZip):
            yield from _iter_mm_core(archive_source, manifest, selected_splits)
            return
        with MMCanonicalZip(archive_source) as archive:
            yield from _iter_mm_core(archive, manifest, selected_splits)

    yield from _bounded(_with_reverse(records(), include_reverse), max_records)


def _iter_eccv_core(
    archive: ECCVArchive,
    manifest: SplitManifestView,
    selected_splits: Optional[frozenset],
    selected_profiles: Optional[frozenset],
) -> Iterator[RealPairRecord]:
    for group in archive.groups():
        if group.profile == "3x3":
            # Always excluded.  Explicit requests were rejected before archive
            # iteration, and no manifest lookup is attempted for smoke-only data.
            continue
        if selected_profiles is not None and group.profile not in selected_profiles:
            continue
        if group.excluded or not group.benchmark_eligible:
            continue
        split = manifest.split_for_group(group.canonical_group_id)
        cluster_id = manifest.component_for_group(group.canonical_group_id)
        if selected_splits is not None and split not in selected_splits:
            continue
        for pair in group.pairs:
            if pair.excluded:
                continue
            if not pair.is_adjacent and not pair.usable_as_negative:
                continue
            yield _eccv_record(group, pair, split, cluster_id, manifest)


def iter_eccv_pair_records(
    archive_source: ECCVSource,
    split_manifest: Union[ManifestSource, SplitManifestView],
    *,
    splits: Optional[Union[str, Iterable[str]]] = None,
    profiles: Optional[Union[str, Iterable[str]]] = None,
    include_reverse: bool = False,
    max_records: Optional[int] = None,
    exposure_audit: Optional[ECCVSplitExposureAudit] = None,
) -> Iterator[RealPairRecord]:
    """Stream eligible ECCV pairs with pre-payload split filtering.

    When ``splits`` is supplied, raw sources are opened under an exact
    manifest-derived group allowlist.  Prebuilt archives are accepted only if
    they already prove that exact scope and disabled aggregate payload access.
    """

    if max_records is not None and max_records < 0:
        raise PairStreamError("max_records cannot be negative")
    if exposure_audit is not None and not isinstance(
        exposure_audit, ECCVSplitExposureAudit
    ):
        raise PairStreamError("exposure_audit must be ECCVSplitExposureAudit")
    manifest = _manifest_view(split_manifest)
    selected_splits = _split_filter(splits)
    selected_profiles = _profile_filter(profiles)
    if selected_profiles is not None and "mm-augmented" in selected_profiles:
        raise PairStreamError("MM profile cannot be requested from an ECCV stream")
    effective_profiles = (
        _ECCV_PAIR_PROFILES
        if selected_profiles is None
        else frozenset(selected_profiles)
    )
    # All manifest/profile validation above is read-free.  A zero record bound
    # must not construct, seek, or scan an archive.
    if max_records == 0:
        return

    expected_scope: Optional[frozenset]
    if selected_splits is None:
        expected_scope = None
        if exposure_audit is not None:
            raise PairStreamError(
                "ECCV exposure audit requires an explicit split filter"
            )
    else:
        expected_scope = _eccv_manifest_pair_groups(
            manifest, effective_profiles, selected_splits
        )

    if isinstance(archive_source, ECCVArchive):
        archive = archive_source
        if expected_scope is None:
            if (
                archive.pair_group_payload_scope is not None
                or not archive.aggregate_payloads_enabled
            ):
                raise PairStreamError(
                    "unfiltered ECCV iteration requires a legacy unscoped archive"
                )
        elif (
            archive.pair_group_payload_scope != expected_scope
            or archive.aggregate_payloads_enabled
        ):
            raise PairStreamError(
                "prebuilt ECCVArchive does not prove the exact requested "
                "split payload scope"
            )
    else:
        archive = ECCVArchive(
            archive_source,
            allowed_pair_group_ids=expected_scope,
        )

    if expected_scope is not None:
        scope_evidence = archive.read_scope_evidence()
        if (
            scope_evidence.mode != "split_scoped"
            or frozenset(scope_evidence.allowed_pair_group_ids or ())
            != expected_scope
            or scope_evidence.aggregate_payloads_enabled
        ):
            raise PairStreamError(
                "ECCV archive exposure evidence does not match requested scope"
            )
    _validate_discovered_eccv_groups(
        archive, manifest, effective_profiles
    )
    if expected_scope is not None:
        missing_scope = expected_scope.difference(
            archive.discovered_pair_group_ids
        )
        if missing_scope:
            raise SplitManifestError(
                "requested ECCV split group is absent from archive headers: "
                "{!r}".format(sorted(missing_scope)[0])
            )
    if exposure_audit is not None:
        assert selected_splits is not None
        assert expected_scope is not None
        test_groups = _eccv_manifest_pair_groups(
            manifest, effective_profiles, frozenset(("test",))
        )
        exposure_audit._bind(
            archive,
            selected_splits=selected_splits,
            selected_profiles=effective_profiles,
            expected_scope=expected_scope,
            manifest_test_pair_group_count=len(test_groups),
        )

    records: Iterable[RealPairRecord] = _iter_eccv_core(
        archive, manifest, selected_splits, selected_profiles
    )
    records = _with_reverse(records, include_reverse)
    if exposure_audit is not None:
        upstream_records = records

        def audited_records() -> Iterator[RealPairRecord]:
            for record in upstream_records:
                exposure_audit._record(record)
                yield record

        records = audited_records()
    yield from _bounded(records, max_records)


def iter_real_pair_records(
    split_manifest: Union[ManifestSource, SplitManifestView],
    *,
    mm_archive: Optional[Union[ZipSource, MMCanonicalZip]] = None,
    eccv_archive: Optional[ECCVSource] = None,
    splits: Optional[Union[str, Iterable[str]]] = None,
    datasets: Optional[Union[str, Iterable[str]]] = None,
    profiles: Optional[Union[str, Iterable[str]]] = None,
    include_reverse: bool = False,
    max_records: Optional[int] = None,
) -> Iterator[RealPairRecord]:
    """Stream selected real B0 records in deterministic MM-then-ECCV order."""

    manifest = _manifest_view(split_manifest)
    selected_datasets = _dataset_filter(datasets)
    selected_profiles = _profile_filter(profiles)
    selected_splits = _split_filter(splits)
    if max_records is not None and max_records < 0:
        raise PairStreamError("max_records cannot be negative")
    if max_records == 0:
        return

    available = set()
    if mm_archive is not None:
        available.add("mm_augmented")
    if eccv_archive is not None:
        available.add("eccv_1113data")
    if not available:
        raise PairStreamError("at least one archive source is required")
    requested = available if selected_datasets is None else set(selected_datasets)
    missing = requested - available
    if missing:
        raise PairStreamError(
            "requested dataset has no archive source: {}".format(sorted(missing)[0])
        )

    def selected_records() -> Iterator[RealPairRecord]:
        if "mm_augmented" in requested and (
            selected_profiles is None or "mm-augmented" in selected_profiles
        ):
            assert mm_archive is not None
            yield from iter_mm_pair_records(
                mm_archive,
                manifest,
                splits=selected_splits,
                include_reverse=include_reverse,
            )
        if "eccv_1113data" in requested and (
            selected_profiles is None
            or selected_profiles.intersection({"2x2", "2x2-negative"})
        ):
            assert eccv_archive is not None
            eccv_profiles = (
                None
                if selected_profiles is None
                else selected_profiles.intersection({"2x2", "2x2-negative"})
            )
            yield from iter_eccv_pair_records(
                eccv_archive,
                manifest,
                splits=selected_splits,
                profiles=eccv_profiles,
                include_reverse=include_reverse,
            )

    yield from _bounded(selected_records(), max_records)


__all__ = [
    "ArchiveImageRef",
    "ECCVSplitExposureAudit",
    "ECCVSplitExposureEvidence",
    "ECCV_STREAM_EXPOSURE_SCHEMA_VERSION",
    "ExcludedProfileError",
    "PairStreamError",
    "RealPairRecord",
    "SplitManifestError",
    "SplitManifestView",
    "STREAM_SCHEMA_VERSION",
    "iter_eccv_pair_records",
    "iter_mm_pair_records",
    "iter_real_pair_records",
    "load_split_manifest",
]
