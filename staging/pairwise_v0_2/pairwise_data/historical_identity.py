"""Fail-closed physical-identity freeze for historical MM/ECCV pairs.

The v0.2 pair stream deliberately keeps image bytes lazy, so its historical
``MaskMemberRef`` values do not carry content hashes.  This module joins those
references to the already frozen v0.1 fingerprint caches and split manifest.
It never decodes a PNG and it rejects, rather than guesses, every disagreement.

Portable receipts contain aggregate counts and cryptographic commitments only.
Member, group, component, fragment and pair identifiers remain local runtime
state and must not be serialized into a portable receipt.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
)

from staging.pairwise_v0_1.baselines.real_pair_stream import load_split_manifest

from .sampling import validation_stream_fingerprint
from .training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    ArchiveBinding,
    TrainingPairRecord,
)


IDENTITY_SCHEMA_VERSION = "dunhuang-pairwise-historical-identity/0.2"
IDENTITY_INDEX_CONTENT_SCHEMA = (
    "dunhuang-pairwise-historical-identity-index-content/0.2"
)
FREEZE_SCHEMA_VERSION = "dunhuang-pairwise-c0-n-q1-freeze/0.3"
FINGERPRINT_CACHE_SCHEMA = "pairwise-group-fingerprint-cache/0.1"
FINGERPRINT_PAYLOAD_SCHEMA = "pairwise-group-fingerprint/0.1"
EXPECTED_SPLIT_CANDIDATE = "70_15_15__pairwise-v0.1-expanded-056"
EXPECTED_SPLIT_SEED = "pairwise-v0.1-expanded-056"
EXPECTED_VALIDATION_FINGERPRINT = (
    "23c7c7446adbddf205ff815240973d2d6e1e89c1c77f27f5a2330b11ccd51078"
)
EXPECTED_VALIDATION_COUNTS = {
    ("mm_augmented", False): 208_502,
    ("mm_augmented", True): 53_488,
    ("eccv_1113data", False): 1_288,
    ("eccv_1113data", True): 9_768,
}
EXPECTED_VALIDATION_TOTAL = 273_046
DEFAULT_TRAIN_TARGET = 4_096
DEFAULT_MAX_PER_COMPONENT_LABEL = 32
DEFAULT_SEED = "260828"
HISTORICAL_TEST_ACCESS_EVIDENCE: Mapping[str, bool] = MappingProxyType(
    {
        "assignment_split_and_cache_metadata_parsed_for_exclusion": True,
        "pair_stream_read": False,
        "archive_members_opened": False,
        "mask_pixels_decoded": False,
        "identity_entered_runtime_index": False,
    }
)


class HistoricalIdentityError(ValueError):
    """Raised when historical physical identity cannot be proven exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _commit_tokens(tokens: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for token in sorted(tokens):
        digest.update(str(token).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _ordered_commit(tokens: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(str(token).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class FragmentIdentity:
    dataset_id: str
    binding: ArchiveBinding
    archive_member: str
    fragment_id: str
    pair_group_id: str
    component_id: str
    split: str
    content_sha256: str

    @property
    def physical_member_key(self) -> str:
        return "{}\0{}".format(self.binding.sha256, self.archive_member)


@dataclass(frozen=True)
class VerifiedPair:
    record: TrainingPairRecord
    fragment_a: FragmentIdentity
    fragment_b: FragmentIdentity


def _binding_for_dataset(dataset_id: str) -> ArchiveBinding:
    if dataset_id == "mm_augmented":
        return MM_CANONICAL_BINDING
    if dataset_id == "eccv_1113data":
        return ECCV_CANONICAL_BINDING
    raise HistoricalIdentityError("historical identity rejects unknown dataset")


_IdentityKey = Tuple[str, str]
_IdentityRow = Tuple[str, str, str, str, str, str, str, str, str, str]


def _require_identity_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise HistoricalIdentityError("invalid identity {}".format(name))
    return value


def _require_identity_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HistoricalIdentityError("invalid identity {} SHA-256".format(name))
    return value


def _safe_identity_member(value: str) -> bool:
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


def _identity_to_row(
    key: Any,
    identity: Any,
) -> Tuple[_IdentityKey, _IdentityRow]:
    if (
        not isinstance(key, tuple)
        or len(key) != 2
        or not all(isinstance(item, str) and item for item in key)
    ):
        raise HistoricalIdentityError("identity mapping key is invalid")
    if not isinstance(identity, FragmentIdentity):
        raise HistoricalIdentityError("identity mapping value is invalid")
    dataset_id = _require_identity_text(identity.dataset_id, "dataset")
    if dataset_id not in {"mm_augmented", "eccv_1113data"}:
        raise HistoricalIdentityError("identity mapping dataset is not historical")
    archive_member = _require_identity_text(identity.archive_member, "archive member")
    if not _safe_identity_member(archive_member):
        raise HistoricalIdentityError("identity archive member is unsafe")
    normalized_key = (dataset_id, archive_member)
    if key != normalized_key:
        raise HistoricalIdentityError("identity mapping key disagrees with value")
    canonical_binding = _binding_for_dataset(dataset_id)
    if identity.binding != canonical_binding:
        raise HistoricalIdentityError("identity archive binding is not canonical")
    if identity.split not in {"train", "val"}:
        raise HistoricalIdentityError("identity mapping split is invalid")
    fragment_id = _require_identity_text(identity.fragment_id, "fragment")
    pair_group_id = _require_identity_text(identity.pair_group_id, "pair group")
    component_id = _require_identity_text(identity.component_id, "component")
    content_sha256 = _require_identity_sha256(identity.content_sha256, "content")
    row: _IdentityRow = (
        dataset_id,
        canonical_binding.logical_id,
        canonical_binding.archive_format,
        canonical_binding.sha256,
        archive_member,
        fragment_id,
        pair_group_id,
        component_id,
        identity.split,
        content_sha256,
    )
    return normalized_key, row


def _row_to_identity(row: _IdentityRow) -> FragmentIdentity:
    (
        dataset_id,
        logical_id,
        archive_format,
        archive_sha256,
        archive_member,
        fragment_id,
        pair_group_id,
        component_id,
        split,
        content_sha256,
    ) = row
    binding = _binding_for_dataset(dataset_id)
    if (
        binding.logical_id != logical_id
        or binding.archive_format != archive_format
        or binding.sha256 != archive_sha256
    ):
        raise HistoricalIdentityError("stored identity archive binding changed")
    return FragmentIdentity(
        dataset_id=dataset_id,
        binding=binding,
        archive_member=archive_member,
        fragment_id=fragment_id,
        pair_group_id=pair_group_id,
        component_id=component_id,
        split=split,
        content_sha256=content_sha256,
    )


class _ReadOnlyIdentityMapping(Mapping[_IdentityKey, FragmentIdentity]):
    """Deeply immutable primitive rows exposed as fresh identity values."""

    __slots__ = ("_rows",)

    def __init__(self, rows: Mapping[_IdentityKey, _IdentityRow]) -> None:
        object.__setattr__(self, "_rows", MappingProxyType(dict(rows)))

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("historical identity mapping is immutable")

    def __getitem__(self, key: _IdentityKey) -> FragmentIdentity:
        return _row_to_identity(self._rows[key])

    def __iter__(self) -> Iterator[_IdentityKey]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def _normalize_identity_mapping(
    by_member: Mapping[Tuple[str, str], FragmentIdentity],
) -> Dict[_IdentityKey, _IdentityRow]:
    if not isinstance(by_member, Mapping) or not by_member:
        raise HistoricalIdentityError("identity mapping must be a non-empty mapping")
    normalized: Dict[_IdentityKey, _IdentityRow] = {}
    for key, identity in by_member.items():
        normalized_key, row = _identity_to_row(key, identity)
        if normalized_key in normalized:
            raise HistoricalIdentityError("duplicate historical identity member")
        normalized[normalized_key] = row
    return normalized


def _identity_rows_content_sha256(
    rows: Mapping[_IdentityKey, _IdentityRow],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "schema_version": IDENTITY_INDEX_CONTENT_SCHEMA,
                "member_count": len(rows),
            }
        )
    )
    digest.update(b"\n")
    for key in sorted(rows):
        row = rows[key]
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(row, tuple)
            or len(row) != 10
            or key != (row[0], row[4])
        ):
            raise HistoricalIdentityError(
                "identity mapping key disagrees with stored primitive row"
            )
        digest.update(
            _canonical_json(
                {
                    "mapping_key": {
                        "dataset_id": key[0],
                        "archive_member": key[1],
                    },
                    "dataset_id": row[0],
                    "archive_binding": {
                        "logical_id": row[1],
                        "archive_format": row[2],
                        "sha256": row[3],
                    },
                    "archive_member": row[4],
                    "fragment_id": row[5],
                    "pair_group_id": row[6],
                    "component_id": row[7],
                    "split": row[8],
                    "content_sha256": row[9],
                }
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _eccv_pair_group(cache_group: Mapping[str, Any]) -> str:
    profile = str(cache_group.get("profile", ""))
    group_id = str(cache_group.get("group_id", ""))
    if profile == "siamese/2x2":
        return "eccv/2x2/{}".format(group_id)
    if profile == "siamese/2x2/negative":
        return "eccv/2x2-negative/{}".format(group_id)
    if profile == "3x3":
        return "eccv/3x3/{}".format(group_id)
    raise HistoricalIdentityError("unsupported ECCV fingerprint-cache profile")


class HistoricalIdentityIndex:
    """Immutable cache/split join used to verify every streamed endpoint."""

    __slots__ = (
        "_artifact_locks",
        "_by_member",
        "_content_sha256",
        "_sealed",
        "_split_candidate_id",
        "_split_seed",
    )

    def __init__(
        self,
        *,
        by_member: Mapping[Tuple[str, str], FragmentIdentity],
        split_candidate_id: str,
        split_seed: str,
        artifact_locks: Mapping[str, Mapping[str, Any]],
    ) -> None:
        rows = _normalize_identity_mapping(by_member)
        split_candidate_id = _require_identity_text(
            split_candidate_id, "split candidate"
        )
        split_seed = _require_identity_text(split_seed, "split seed")
        if not isinstance(artifact_locks, Mapping):
            raise HistoricalIdentityError("historical artifact locks must be a mapping")
        expected_roles = {
            "mm_fingerprint_cache",
            "eccv_fingerprint_cache",
            "historical_split",
        }
        if set(artifact_locks) != expected_roles:
            raise HistoricalIdentityError("historical artifact lock inventory changed")
        locked_artifacts = {}
        for role, lock in artifact_locks.items():
            if not isinstance(lock, Mapping) or set(lock) != {"bytes", "sha256"}:
                raise HistoricalIdentityError("historical artifact lock is invalid")
            byte_count = lock.get("bytes")
            if type(byte_count) is not int or byte_count <= 0:  # noqa: E721
                raise HistoricalIdentityError(
                    "historical artifact byte count is invalid"
                )
            locked_artifacts[role] = MappingProxyType(
                {
                    "bytes": byte_count,
                    "sha256": _require_identity_sha256(lock.get("sha256"), "artifact"),
                }
            )
        object.__setattr__(self, "_by_member", _ReadOnlyIdentityMapping(rows))
        object.__setattr__(self, "_split_candidate_id", split_candidate_id)
        object.__setattr__(self, "_split_seed", split_seed)
        object.__setattr__(self, "_artifact_locks", MappingProxyType(locked_artifacts))
        object.__setattr__(self, "_content_sha256", _identity_rows_content_sha256(rows))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("HistoricalIdentityIndex is immutable")
        object.__setattr__(self, _name, _value)

    @property
    def split_candidate_id(self) -> str:
        return self._split_candidate_id

    @property
    def split_seed(self) -> str:
        return self._split_seed

    @property
    def artifact_locks(self) -> Mapping[str, Mapping[str, Any]]:
        return self._artifact_locks

    @property
    def content_sha256(self) -> str:
        """Portable commitment to every canonical identity mapping field."""

        return self._content_sha256

    @property
    def identity_count(self) -> int:
        return len(self._by_member)

    @classmethod
    def from_files(
        cls,
        *,
        mm_cache_path: Path,
        eccv_cache_path: Path,
        split_path: Path,
    ) -> "HistoricalIdentityIndex":
        paths = {
            "mm_fingerprint_cache": Path(mm_cache_path),
            "eccv_fingerprint_cache": Path(eccv_cache_path),
            "historical_split": Path(split_path),
        }
        raw_payloads: Dict[str, bytes] = {}
        for name, path in paths.items():
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise HistoricalIdentityError("invalid {} JSON".format(name)) from exc
            raw_payloads[name] = raw
        return cls.from_payloads(
            mm_cache_payload=raw_payloads["mm_fingerprint_cache"],
            eccv_cache_payload=raw_payloads["eccv_fingerprint_cache"],
            split_payload=raw_payloads["historical_split"],
        )

    @classmethod
    def from_payloads(
        cls,
        *,
        mm_cache_payload: bytes,
        eccv_cache_payload: bytes,
        split_payload: bytes,
    ) -> "HistoricalIdentityIndex":
        """Build the exact identity join from already captured immutable bytes."""

        raw_payloads = {
            "mm_fingerprint_cache": mm_cache_payload,
            "eccv_fingerprint_cache": eccv_cache_payload,
            "historical_split": split_payload,
        }
        payloads: Dict[str, Mapping[str, Any]] = {}
        for name, raw in raw_payloads.items():
            if type(raw) is not bytes:  # noqa: E721
                raise HistoricalIdentityError(
                    "{} payload must be immutable bytes".format(name)
                )
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HistoricalIdentityError("invalid {} JSON".format(name)) from exc
            if not isinstance(value, Mapping):
                raise HistoricalIdentityError("{} root must be an object".format(name))
            payloads[name] = value

        split_document = payloads["historical_split"]
        split_view = load_split_manifest(split_document)
        if split_view.candidate_id != EXPECTED_SPLIT_CANDIDATE:
            raise HistoricalIdentityError("unexpected historical split candidate")
        split_seed = str(split_document.get("seed", ""))
        if split_seed != EXPECTED_SPLIT_SEED:
            raise HistoricalIdentityError("unexpected historical split seed")

        by_member: Dict[Tuple[str, str], FragmentIdentity] = {}
        cache_specs = (
            (
                "mm_augmented",
                MM_CANONICAL_BINDING,
                payloads["mm_fingerprint_cache"],
            ),
            (
                "eccv_1113data",
                ECCV_CANONICAL_BINDING,
                payloads["eccv_fingerprint_cache"],
            ),
        )
        for dataset_id, binding, cache in cache_specs:
            if cache.get("cache_schema_version") != FINGERPRINT_CACHE_SCHEMA:
                raise HistoricalIdentityError("unsupported fingerprint-cache schema")
            archive = cache.get("archive")
            fingerprints = cache.get("fingerprints")
            if not isinstance(archive, Mapping) or not isinstance(
                fingerprints, Mapping
            ):
                raise HistoricalIdentityError("fingerprint cache is incomplete")
            if archive.get("sha256") != binding.sha256:
                raise HistoricalIdentityError(
                    "cache archive SHA disagrees with binding"
                )
            if archive.get("expected_sha256") != binding.sha256:
                raise HistoricalIdentityError(
                    "cache expected archive SHA is not canonical"
                )
            if fingerprints.get("schema_version") != FINGERPRINT_PAYLOAD_SCHEMA:
                raise HistoricalIdentityError("unsupported fingerprint payload schema")
            groups = fingerprints.get("groups")
            if not isinstance(groups, list):
                raise HistoricalIdentityError("fingerprint groups must be a list")
            for group in groups:
                if not isinstance(group, Mapping):
                    raise HistoricalIdentityError("fingerprint group must be an object")
                if dataset_id == "mm_augmented":
                    base_group = str(group.get("base_group_id", ""))
                    source_id = str(group.get("source_id", ""))
                    if not base_group or not source_id:
                        raise HistoricalIdentityError(
                            "MM cache group lacks base/source"
                        )
                    pair_group = "mm/base/{}".format(base_group)
                    expected_component = "mm/source/{}".format(source_id)
                else:
                    pair_group = _eccv_pair_group(group)
                    signature = str(group.get("group_signature", ""))
                    expected_component = "eccv/signature/{}".format(signature)

                # 3x3 is cache-audited but deliberately absent from Pairwise.
                if pair_group.startswith("eccv/3x3/"):
                    continue
                try:
                    split = split_view.split_for_group(pair_group)
                    component = split_view.component_for_group(pair_group)
                except Exception as exc:
                    raise HistoricalIdentityError(
                        "cache group lacks exact split inheritance"
                    ) from exc
                if component != expected_component:
                    raise HistoricalIdentityError(
                        "split component disagrees with cache-derived component"
                    )
                # The C0 runtime is strictly train+validation.  Historical test
                # groups remain cache-audited by their source-file locks but are
                # deliberately absent from the in-memory identity universe, so
                # neither verification nor a provider can ever resolve them.
                if split == "test":
                    continue
                if split not in {"train", "val"}:
                    raise HistoricalIdentityError(
                        "cache group has an unsupported split assignment"
                    )
                fragments = group.get("fragments")
                if not isinstance(fragments, list) or not fragments:
                    raise HistoricalIdentityError("cache group has no fragments")
                for fragment in fragments:
                    if not isinstance(fragment, Mapping):
                        raise HistoricalIdentityError(
                            "cache fragment must be an object"
                        )
                    member = str(fragment.get("member_path", ""))
                    content_sha = str(fragment.get("content_sha256", ""))
                    canonical_fragment = str(fragment.get("canonical_fragment_id", ""))
                    if len(content_sha) != 64:
                        raise HistoricalIdentityError("invalid cached content SHA")
                    if dataset_id == "mm_augmented":
                        fragment_id = str(PurePosixPath(member).with_suffix(""))
                    else:
                        profile = (
                            "2x2/negative"
                            if pair_group.startswith("eccv/2x2-negative/")
                            else "2x2"
                        )
                        group_id = pair_group.rsplit("/", 1)[-1]
                        fragment_id = "{}/{}/{}.png".format(
                            profile, group_id, canonical_fragment
                        )
                    key = (dataset_id, member)
                    if key in by_member:
                        raise HistoricalIdentityError("duplicate cached archive member")
                    by_member[key] = FragmentIdentity(
                        dataset_id=dataset_id,
                        binding=binding,
                        archive_member=member,
                        fragment_id=fragment_id,
                        pair_group_id=pair_group,
                        component_id=component,
                        split=split,
                        content_sha256=content_sha,
                    )

        locks = {
            name: {
                "bytes": len(raw_payloads[name]),
                "sha256": hashlib.sha256(raw_payloads[name]).hexdigest(),
            }
            for name in raw_payloads
        }
        return cls(
            by_member=by_member,
            split_candidate_id=split_view.candidate_id,
            split_seed=split_seed,
            artifact_locks=locks,
        )

    def verify_record(self, record: TrainingPairRecord) -> VerifiedPair:
        if not isinstance(record, TrainingPairRecord):
            raise HistoricalIdentityError("expected TrainingPairRecord")
        if record.split not in {"train", "val"}:
            raise HistoricalIdentityError("test/real records are prohibited")
        expected_binding = _binding_for_dataset(record.dataset_id)

        identities = []
        enriched_refs = []
        for ref in (record.fragment_a, record.fragment_b):
            if ref.binding != expected_binding:
                raise HistoricalIdentityError("record archive binding mismatch")
            try:
                identity = self._by_member[(record.dataset_id, ref.archive_member)]
            except KeyError as exc:
                raise HistoricalIdentityError(
                    "record member absent from cache"
                ) from exc
            comparisons = (
                (ref.fragment_id, identity.fragment_id, "fragment"),
                (ref.canonical_group_id, identity.pair_group_id, "group"),
                (ref.component_id, identity.component_id, "component"),
                (ref.split, identity.split, "split"),
                (record.canonical_group_id, identity.pair_group_id, "pair group"),
                (record.component_id, identity.component_id, "pair component"),
                (record.split, identity.split, "pair split"),
            )
            for observed, expected, field in comparisons:
                if observed != expected:
                    raise HistoricalIdentityError(
                        "record {} disagrees with cache/split".format(field)
                    )
            if ref.content_sha256 not in {None, identity.content_sha256}:
                raise HistoricalIdentityError("record content SHA mismatch")
            identities.append(identity)
            enriched_refs.append(replace(ref, content_sha256=identity.content_sha256))
        enriched = replace(
            record, fragment_a=enriched_refs[0], fragment_b=enriched_refs[1]
        )
        return VerifiedPair(enriched, identities[0], identities[1])

    def identities_for_split(self, split: str) -> Tuple[FragmentIdentity, ...]:
        """Return the complete cache-derived fragment universe for a split.

        This includes fragments in assigned Pairwise groups even when no usable
        pair row happens to reference them.  Using this conservative universe
        prevents an unreferenced validation fragment from reappearing through a
        training row with byte-identical content.
        """

        if split not in {"train", "val"}:
            raise HistoricalIdentityError("identity split must be train or val")
        return tuple(
            identity
            for _key, identity in sorted(self._by_member.items())
            if identity.split == split
        )


def historical_identity_index_content_sha256(
    identity_index: HistoricalIdentityIndex,
) -> str:
    """Recompute the full mapping commitment without trusting a stored digest."""

    if type(identity_index) is not HistoricalIdentityIndex:  # noqa: E721
        raise HistoricalIdentityError("expected exact HistoricalIdentityIndex")
    mapping = identity_index._by_member
    if not isinstance(mapping, _ReadOnlyIdentityMapping):
        raise HistoricalIdentityError("historical identity mapping storage changed")
    return _identity_rows_content_sha256(mapping._rows)


@dataclass(frozen=True)
class C0FreezeResult:
    selected_train_records: Tuple[TrainingPairRecord, ...]
    receipt: Mapping[str, Any]


def _record_private_token(verified: VerifiedPair) -> str:
    record = verified.record
    return hashlib.sha256(
        _canonical_json(
            {
                "dataset": record.dataset_id,
                "label": record.label,
                "component": record.component_id,
                "pair": list(record.canonical_pair_key),
                "members": [
                    verified.fragment_a.physical_member_key,
                    verified.fragment_b.physical_member_key,
                ],
                "content": [
                    verified.fragment_a.content_sha256,
                    verified.fragment_b.content_sha256,
                ],
            }
        )
    ).hexdigest()


def _priority(seed: str, namespace: str, token: str) -> str:
    return hashlib.sha256(_canonical_json([seed, namespace, token])).hexdigest()


def _push_bounded(
    heap: list[Tuple[int, str, VerifiedPair]],
    item: VerifiedPair,
    *,
    priority: str,
    limit: int,
) -> None:
    # A min-heap over negative priorities retains the lexicographically lowest
    # ``limit`` hashes without depending on input order.
    numeric = int(priority, 16)
    entry = (-numeric, priority, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif numeric < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def _balanced_select(
    buckets: Mapping[str, Sequence[VerifiedPair]],
    *,
    dataset_id: str,
    label: bool,
    seed: str,
    target: int,
) -> Tuple[VerifiedPair, ...]:
    namespace = "{}|{}".format(dataset_id, int(label))
    ordered_components = sorted(
        buckets,
        key=lambda component: (
            _priority(seed, namespace + "|component", component),
            component,
        ),
    )
    ordered: Dict[str, Tuple[VerifiedPair, ...]] = {}
    for component in ordered_components:
        ordered[component] = tuple(
            sorted(
                buckets[component],
                key=lambda item: (
                    _priority(seed, namespace + "|record", _record_private_token(item)),
                    _record_private_token(item),
                ),
            )
        )
    selected = []
    depth = 0
    while len(selected) < target:
        added = 0
        for component in ordered_components:
            values = ordered[component]
            if depth < len(values):
                selected.append(values[depth])
                added += 1
                if len(selected) == target:
                    break
        if added == 0:
            break
        depth += 1
    if len(selected) != target:
        raise HistoricalIdentityError(
            "insufficient eligible records for {} label {}: {} < {}".format(
                dataset_id, int(label), len(selected), target
            )
        )
    return tuple(selected)


def freeze_c0_n_q1(
    *,
    identity_index: HistoricalIdentityIndex,
    validation_records: Iterable[TrainingPairRecord],
    training_records: Iterable[TrainingPairRecord],
    archive_paths: Mapping[str, Path],
    seed: str = DEFAULT_SEED,
    target_per_dataset_label: int = DEFAULT_TRAIN_TARGET,
    max_per_component_label: int = DEFAULT_MAX_PER_COMPONENT_LABEL,
    expected_validation_fingerprint: str = EXPECTED_VALIDATION_FINGERPRINT,
    expected_validation_counts: Mapping[
        Tuple[str, bool], int
    ] = EXPECTED_VALIDATION_COUNTS,
) -> C0FreezeResult:
    """Freeze C0-N-Q1 metadata and selection without loading mask pixels."""

    if not seed or target_per_dataset_label <= 0 or max_per_component_label <= 0:
        raise HistoricalIdentityError("invalid C0 selection configuration")
    expected_archive_keys = {"mm_augmented", "eccv_1113data"}
    if set(archive_paths) != expected_archive_keys:
        raise HistoricalIdentityError("exactly canonical MM/ECCV archives are required")
    archive_locks = {}
    for dataset_id, path_value in sorted(archive_paths.items()):
        path = Path(path_value)
        expected = _binding_for_dataset(dataset_id)
        observed = sha256_file(path)
        if observed != expected.sha256:
            raise HistoricalIdentityError("actual archive SHA is not canonical")
        archive_locks[dataset_id] = {
            "format": expected.archive_format,
            "bytes": path.stat().st_size,
            "sha256": observed,
        }

    val_records_for_fingerprint = []
    val_counts: Counter[Tuple[str, bool]] = Counter()
    val_universe = identity_index.identities_for_split("val")
    val_content = {identity.content_sha256 for identity in val_universe}
    val_members = {identity.physical_member_key for identity in val_universe}
    val_components = {identity.component_id for identity in val_universe}
    val_endpoint_content = set()
    val_endpoint_members = set()
    val_record_tokens = []
    for record in validation_records:
        verified = identity_index.verify_record(record)
        if verified.record.split != "val":
            raise HistoricalIdentityError("validation stream contains non-val record")
        val_records_for_fingerprint.append(verified.record)
        val_counts[(verified.record.dataset_id, verified.record.label)] += 1
        for identity in (verified.fragment_a, verified.fragment_b):
            val_endpoint_content.add(identity.content_sha256)
            val_endpoint_members.add(identity.physical_member_key)
        val_record_tokens.append(_record_private_token(verified))

    if dict(val_counts) != dict(expected_validation_counts):
        raise HistoricalIdentityError("full validation dataset/label counts changed")
    if len(val_records_for_fingerprint) != sum(expected_validation_counts.values()):
        raise HistoricalIdentityError("full validation total changed")
    validation_fingerprint = validation_stream_fingerprint(val_records_for_fingerprint)
    if validation_fingerprint["sha256"] != expected_validation_fingerprint:
        raise HistoricalIdentityError(
            "validation stream fingerprint changed; receipt was not generated"
        )

    heaps: MutableMapping[
        Tuple[str, bool, str], list[Tuple[int, str, VerifiedPair]]
    ] = defaultdict(list)
    train_counts: Counter[Tuple[str, bool]] = Counter()
    eligible_counts: Counter[Tuple[str, bool]] = Counter()
    quarantine_counts: Counter[Tuple[str, bool]] = Counter()
    quarantine_endpoint_hits = 0
    quarantined_physical_pairs = set()
    for record in training_records:
        verified = identity_index.verify_record(record)
        if verified.record.split != "train":
            raise HistoricalIdentityError("training stream contains non-train record")
        stratum = (verified.record.dataset_id, verified.record.label)
        train_counts[stratum] += 1
        hits = sum(
            identity.content_sha256 in val_content
            for identity in (verified.fragment_a, verified.fragment_b)
        )
        if hits:
            quarantine_counts[stratum] += 1
            quarantine_endpoint_hits += hits
            quarantined_physical_pairs.add(
                hashlib.sha256(
                    _canonical_json(
                        [
                            verified.record.dataset_id,
                            sorted(
                                (
                                    verified.fragment_a.physical_member_key,
                                    verified.fragment_b.physical_member_key,
                                )
                            ),
                        ]
                    )
                ).hexdigest()
            )
            continue
        eligible_counts[stratum] += 1
        token = _record_private_token(verified)
        priority = _priority(seed, "bucket-record", token)
        bucket_key = (
            verified.record.dataset_id,
            verified.record.label,
            verified.record.component_id,
        )
        _push_bounded(
            heaps[bucket_key],
            verified,
            priority=priority,
            limit=max_per_component_label,
        )

    selected_verified = []
    selected_counts: Counter[Tuple[str, bool]] = Counter()
    for dataset_id in ("mm_augmented", "eccv_1113data"):
        for label in (False, True):
            buckets = {
                component: tuple(entry[2] for entry in heap)
                for (dataset, bucket_label, component), heap in heaps.items()
                if dataset == dataset_id and bucket_label is label
            }
            values = _balanced_select(
                buckets,
                dataset_id=dataset_id,
                label=label,
                seed=seed,
                target=target_per_dataset_label,
            )
            selected_verified.extend(values)
            selected_counts[(dataset_id, label)] += len(values)

    selected_components = {item.record.component_id for item in selected_verified}
    selected_members = {
        identity.physical_member_key
        for item in selected_verified
        for identity in (item.fragment_a, item.fragment_b)
    }
    selected_content = {
        identity.content_sha256
        for item in selected_verified
        for identity in (item.fragment_a, item.fragment_b)
    }
    overlaps = {
        "component": len(selected_components & val_components),
        "member": len(selected_members & val_members),
        "content_sha256": len(selected_content & val_content),
    }
    if any(overlaps.values()):
        raise HistoricalIdentityError("selected train still overlaps full validation")

    component_label_counts = Counter(
        (item.record.component_id, item.record.label) for item in selected_verified
    )
    observed_max = max(component_label_counts.values(), default=0)
    if observed_max > max_per_component_label:
        raise HistoricalIdentityError("component cap violated")

    selected_tokens = [_record_private_token(item) for item in selected_verified]
    receipt: Dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "pass_metadata_only_no_model_execution",
        "scope": {
            "experiment": "C0-N-Q1",
            "datasets": ["mm_augmented", "eccv_1113data"],
            "pair_stream_splits_read": ["train", "val"],
            "historical_test_access": dict(HISTORICAL_TEST_ACCESS_EVIDENCE),
            "sealed_real_read": False,
            "mask_pixels_decoded": False,
            "model_executed": False,
        },
        "locks": {
            "archives": archive_locks,
            "metadata_artifacts": {
                role: dict(lock) for role, lock in identity_index.artifact_locks.items()
            },
            "split_candidate_sha256": hashlib.sha256(
                identity_index.split_candidate_id.encode("utf-8")
            ).hexdigest(),
            "split_seed_sha256": hashlib.sha256(
                identity_index.split_seed.encode("utf-8")
            ).hexdigest(),
        },
        "validation": {
            "count": len(val_records_for_fingerprint),
            "by_dataset_label": {
                "{}|{}".format(dataset, "positive" if label else "negative"): count
                for (dataset, label), count in sorted(val_counts.items())
            },
            "frozen_order_sha256": validation_fingerprint["sha256"],
            "record_sequence_commitment_sha256": _ordered_commit(val_record_tokens),
            "identity_universe_policy": (
                "all_cached_fragments_in_val_assigned_pairwise_groups_including_"
                "fragments_not_referenced_by_usable_pair_rows"
            ),
            "identity_universe_fragment_count": len(val_universe),
            "unique_content_sha256_count": len(val_content),
            "content_set_commitment_sha256": _commit_tokens(val_content),
            "unique_physical_member_count": len(val_members),
            "unique_component_count": len(val_components),
            "pair_stream_endpoint_unique_content_sha256_count": len(
                val_endpoint_content
            ),
            "pair_stream_endpoint_content_set_commitment_sha256": _commit_tokens(
                val_endpoint_content
            ),
            "pair_stream_endpoint_unique_physical_member_count": len(
                val_endpoint_members
            ),
        },
        "training": {
            "input_by_dataset_label": {
                "{}|{}".format(dataset, "positive" if label else "negative"): count
                for (dataset, label), count in sorted(train_counts.items())
            },
            "content_quarantine": {
                "policy": (
                    "raw_train_row_quarantined_when_either_endpoint_content_sha256_"
                    "occurs_in_complete_val_assigned_cache_fragment_universe"
                ),
                "prefilter_or_deduplication_before_quarantine": False,
                "raw_row_count": sum(quarantine_counts.values()),
                "unique_physical_pair_count": len(quarantined_physical_pairs),
                "endpoint_hit_count": quarantine_endpoint_hits,
                "by_dataset_label_raw_row_count": {
                    "{}|{}".format(dataset, "positive" if label else "negative"): count
                    for (dataset, label), count in sorted(quarantine_counts.items())
                },
            },
            "eligible_by_dataset_label": {
                "{}|{}".format(dataset, "positive" if label else "negative"): count
                for (dataset, label), count in sorted(eligible_counts.items())
            },
            "selection": {
                "seed_sha256": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
                "policy": "sha256_priority_component_round_robin",
                "target_per_dataset_label": target_per_dataset_label,
                "max_per_component_label": max_per_component_label,
                "observed_max_per_component_label": observed_max,
                "count": len(selected_verified),
                "by_dataset_label": {
                    "{}|{}".format(dataset, "positive" if label else "negative"): count
                    for (dataset, label), count in sorted(selected_counts.items())
                },
                "record_set_commitment_sha256": _commit_tokens(selected_tokens),
                "record_order_commitment_sha256": _ordered_commit(selected_tokens),
            },
        },
        "selected_train_vs_full_validation_overlap": overlaps,
        "portable_privacy": {
            "absolute_paths_present": False,
            "member_identifiers_present": False,
            "group_identifiers_present": False,
            "component_identifiers_present": False,
            "pair_identifiers_present": False,
            "content_hash_values_present": False,
            "aggregate_commitments_only": True,
        },
    }
    receipt["content_sha256"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    return C0FreezeResult(
        selected_train_records=tuple(item.record for item in selected_verified),
        receipt=receipt,
    )


def assert_portable_receipt(receipt: Mapping[str, Any]) -> None:
    """Reject identifiers or local paths accidentally added to a receipt."""

    text = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    forbidden = (
        "/Users/",
        "\\Users\\",
        "canonical_group_id",
        "component_id",
        "fragment_id",
        "pair_id",
        "member_path",
    )
    # Privacy assertions such as ``component_identifiers_present: false`` are
    # allowed.  Raw identity field names and known local-path roots are not.
    scrubbed = text.replace("component_identifiers_present", "")
    scrubbed = scrubbed.replace("member_identifiers_present", "")
    scrubbed = scrubbed.replace("group_identifiers_present", "")
    scrubbed = scrubbed.replace("fragment_identifiers_present", "")
    scrubbed = scrubbed.replace("pair_identifiers_present", "")
    if any(token in scrubbed for token in forbidden):
        raise HistoricalIdentityError("portable receipt exposes private identity")


__all__ = [
    "C0FreezeResult",
    "DEFAULT_MAX_PER_COMPONENT_LABEL",
    "DEFAULT_SEED",
    "DEFAULT_TRAIN_TARGET",
    "EXPECTED_VALIDATION_COUNTS",
    "EXPECTED_VALIDATION_FINGERPRINT",
    "EXPECTED_VALIDATION_TOTAL",
    "FREEZE_SCHEMA_VERSION",
    "HISTORICAL_TEST_ACCESS_EVIDENCE",
    "FragmentIdentity",
    "HistoricalIdentityError",
    "HistoricalIdentityIndex",
    "IDENTITY_INDEX_CONTENT_SCHEMA",
    "VerifiedPair",
    "assert_portable_receipt",
    "freeze_c0_n_q1",
    "historical_identity_index_content_sha256",
    "sha256_file",
]
