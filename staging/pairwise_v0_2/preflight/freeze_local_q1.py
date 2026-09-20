#!/usr/bin/env python3
"""Freeze the metadata-only population shared by local-matcher Q1 arms.

The freeze reads only the authorized MM/ECCV train/validation pair metadata
and the canonical-new training manifest.  It neither opens a mask member nor
imports a model.  Raw pair/member/component identifiers remain runtime-only;
the portable receipt contains counts and cryptographic commitments.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import hmac
import json
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    EXPECTED_VALIDATION_COUNTS,
    EXPECTED_VALIDATION_FINGERPRINT,
    HISTORICAL_TEST_ACCESS_EVIDENCE,
    HistoricalIdentityError,
    HistoricalIdentityIndex,
    VerifiedPair,
    historical_identity_index_content_sha256,
    sha256_file,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    ArchiveBinding,
    TrainingPairRecord,
    iter_historical_pair_records,
    iter_synthetic_pair_records,
)
from staging.pairwise_v0_2.preflight.local_q1_geometry_eligibility_authority import (
    LocalQ1GeometryEligibilityAuthority,
    LocalQ1GeometryQualificationError,
)
from staging.pairwise_v0_2.preflight.local_q1_selection_frontier import (
    LocalQ1FrontierCandidate,
    LocalQ1FrontierSelectionResult,
    LocalQ1SelectionFrontier,
    LocalQ1SelectionFrontierError,
)


SCHEMA_VERSION = "dunhuang-pairwise-local-q1-freeze/0.1"
ROUTE_A_SCHEMA_VERSION = "dunhuang-pairwise-local-q1-freeze/0.2"
SELECTION_SEED = "local-q1-260828"
TRAIN_TARGET_PER_DATASET_LABEL = 512
VALIDATION_CAPS = {"mm_augmented": 32, "eccv_1113data": 4}
EXPECTED_VALIDATION_COMPONENT_COUNTS = {"mm_augmented": 55, "eccv_1113data": 533}
SYNTHETIC_DATASET_ID = "dunhuang_voronoi_masks_no_erode_v0_2"
SYNTHETIC_RECEIPT_NAME = "canonical_new"
SYNTHETIC_MANIFEST_SHA256 = (
    "e9772ec8e074873e4343ca42906a4056ea382536d2cbdf6ec881c21891587326"
)
SYNTHETIC_ARCHIVE_BINDING = ArchiveBinding(
    logical_id="local_asset://pairwise_mask_subset_v0_2",
    archive_format="zip",
    sha256="e44e4c0e5825d8577861d79eaf4063888a349c6ba3e531be6e41f2b7dde505df",
)
EXPECTED_TRAIN_INPUT_COUNTS = {
    ("mm_augmented", False): 972_449,
    ("mm_augmented", True): 247_598,
    ("eccv_1113data", False): 6_002,
    ("eccv_1113data", True): 48_328,
    (SYNTHETIC_DATASET_ID, False): 3_262,
    (SYNTHETIC_DATASET_ID, True): 14_170,
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MM_CACHE = (
    REPOSITORY_ROOT / "staging/pairwise_v0_1/cache/mm_group_fingerprints.json"
)
DEFAULT_ECCV_CACHE = (
    REPOSITORY_ROOT / "staging/pairwise_v0_1/cache/eccv_group_fingerprints.json"
)
DEFAULT_SPLIT = (
    REPOSITORY_ROOT / "staging/pairwise_v0_1/data_gate/split_candidate_70_15_15.json"
)
DEFAULT_MM_ARCHIVE = Path(
    "data/external/dunhuang_augmented_data.zip"
)
DEFAULT_ECCV_ARCHIVE = Path("data/external/1113data.tar.gz")
DEFAULT_SYNTHETIC_MANIFEST = (
    REPOSITORY_ROOT / "staging/pairwise_v0_2/manifests/"
    "dunhuang_pairwise_mask_subset_v0_2/manifest/synthetic_groups.jsonl"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "local_q1_freeze.json"
DEFAULT_LOCAL_NOTES = Path(__file__).resolve().parent / "local_q1_freeze.local.md"


class LocalQ1FreezeError(ValueError):
    """Raised when the exact local-Q1 population cannot be proven."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha_object(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_json_object(payload: bytes, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, bytes):
        raise LocalQ1FreezeError("{} bytes are required".format(name))

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("non-finite JSON constant: " + token)
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise LocalQ1FreezeError("{} is not strict JSON".format(name)) from exc
    if not isinstance(value, Mapping):
        raise LocalQ1FreezeError("{} root must be an object".format(name))
    return value


def _load_locked_predecessor_freeze_receipt(
    payload: bytes,
    *,
    expected_bytes: int,
    expected_file_sha256: str,
    expected_content_sha256: str,
) -> Mapping[str, Any]:
    """Snapshot semantics for the independently byte/hash-locked v0.1 freeze."""

    if len(payload) != expected_bytes or not hmac.compare_digest(
        hashlib.sha256(payload).hexdigest(), expected_file_sha256
    ):
        raise LocalQ1FreezeError("predecessor freeze file byte lock changed")
    receipt = _strict_json_object(payload, "predecessor freeze receipt")
    expected_root = {
        "schema_version",
        "status",
        "scope",
        "locks",
        "selection_contract",
        "training",
        "validation",
        "selected_train_vs_selected_validation_overlap",
        "selected_train_vs_complete_validation_identity_universe_overlap",
        "portable_privacy",
        "content_sha256",
    }
    if (
        set(receipt) != expected_root
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status")
        != "pass_metadata_only_population_frozen_no_model_execution"
    ):
        raise LocalQ1FreezeError("predecessor freeze receipt schema/status changed")
    claimed = receipt.get("content_sha256")
    if (
        not isinstance(claimed, str)
        or not re.fullmatch(r"[0-9a-f]{64}", claimed)
        or not hmac.compare_digest(claimed, expected_content_sha256)
    ):
        raise LocalQ1FreezeError("predecessor freeze content lock changed")
    unsigned = dict(receipt)
    unsigned.pop("content_sha256")
    if not hmac.compare_digest(_sha_object(unsigned), claimed):
        raise LocalQ1FreezeError("predecessor freeze self-content hash changed")
    scope = receipt.get("scope")
    if not isinstance(scope, Mapping) or (
        scope.get("historical_test_access") != dict(HISTORICAL_TEST_ACCESS_EVIDENCE)
        or scope.get("synthetic_validation_read") is not False
        or scope.get("sealed_real_read") is not False
        or scope.get("archive_mask_members_opened") is not False
        or scope.get("mask_pixels_decoded") is not False
        or scope.get("model_imported") is not False
        or scope.get("model_executed") is not False
    ):
        raise LocalQ1FreezeError("predecessor freeze scope is not metadata-only")
    zero_overlap = {"component": 0, "physical_member": 0, "content_sha256": 0}
    if (
        receipt.get("selected_train_vs_selected_validation_overlap")
        != zero_overlap
        or receipt.get(
            "selected_train_vs_complete_validation_identity_universe_overlap"
        )
        != zero_overlap
    ):
        raise LocalQ1FreezeError("predecessor freeze overlap proof changed")
    return receipt


def _assert_predecessor_input_lock_replay(
    receipt: Mapping[str, Any],
    *,
    normalized_archive_locks: Mapping[str, Mapping[str, Any]],
    identity_index: HistoricalIdentityIndex,
    synthetic_manifest_lock: Mapping[str, Any],
) -> None:
    locks = receipt.get("locks")
    if not isinstance(locks, Mapping) or set(locks) != {
        "historical_archive_containers",
        "historical_metadata_artifacts",
        "historical_identity_index",
        "synthetic_manifest",
        "synthetic_archive_reference",
    }:
        raise LocalQ1FreezeError("predecessor freeze input-lock schema changed")
    expected_identity = {
        "member_count": identity_index.identity_count,
        "content_sha256": historical_identity_index_content_sha256(identity_index),
    }
    expected_synthetic_archive = {
        "format": SYNTHETIC_ARCHIVE_BINDING.archive_format,
        "logical_id": SYNTHETIC_ARCHIVE_BINDING.logical_id,
        "sha256": SYNTHETIC_ARCHIVE_BINDING.sha256,
        "opened_by_freeze": False,
    }
    if (
        locks.get("historical_archive_containers")
        != normalized_archive_locks
        or locks.get("historical_metadata_artifacts")
        != {
            role: dict(lock)
            for role, lock in identity_index.artifact_locks.items()
        }
        or locks.get("historical_identity_index") != expected_identity
        or locks.get("synthetic_manifest") != dict(synthetic_manifest_lock)
        or locks.get("synthetic_archive_reference")
        != expected_synthetic_archive
    ):
        raise LocalQ1FreezeError(
            "predecessor freeze inputs differ from Route-A refreeze inputs"
        )


def _assert_predecessor_population_replay(
    receipt: Mapping[str, Any], selection_proof: Mapping[str, Any]
) -> None:
    predecessor = selection_proof.get("predecessor_population")
    training = receipt.get("training")
    validation = receipt.get("validation")
    if (
        not isinstance(predecessor, Mapping)
        or not isinstance(training, Mapping)
        or not isinstance(validation, Mapping)
        or not isinstance(training.get("population"), Mapping)
        or not isinstance(validation.get("population"), Mapping)
    ):
        raise LocalQ1FreezeError("predecessor population proof is missing")
    train_population = training["population"]
    validation_population = validation["population"]
    expected = {
        "training_count": train_population.get("count"),
        "validation_count": validation_population.get("count"),
        "training_record_order_sha256": train_population.get(
            "record_order_commitment_sha256"
        ),
        "training_record_set_sha256": train_population.get(
            "record_set_commitment_sha256"
        ),
        "validation_record_order_sha256": validation_population.get(
            "record_order_commitment_sha256"
        ),
        "validation_record_set_sha256": validation_population.get(
            "record_set_commitment_sha256"
        ),
    }
    if predecessor != expected:
        raise LocalQ1FreezeError(
            "Route-A old population differs from locked predecessor freeze"
        )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    return value


def _commit_order(tokens: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(str(token).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _commit_set(tokens: Iterable[str]) -> str:
    return _commit_order(sorted(tokens))


def _priority(seed: str, namespace: str, token: str) -> str:
    return _sha_object([seed, namespace, token])


def _receipt_dataset(dataset_id: str) -> str:
    if dataset_id == SYNTHETIC_DATASET_ID:
        return SYNTHETIC_RECEIPT_NAME
    if dataset_id in {"mm_augmented", "eccv_1113data"}:
        return dataset_id
    raise LocalQ1FreezeError("unexpected dataset in local-Q1 population")


def _stratum_name(dataset_id: str, label: bool) -> str:
    return "{}|{}".format(
        _receipt_dataset(dataset_id), "positive" if label else "negative"
    )


@dataclass(frozen=True)
class _QualifiedPair:
    record: TrainingPairRecord
    dataset_name: str
    component_token: str
    member_tokens: Tuple[str, str]
    content_tokens: Tuple[str, str]
    pair_identity_token: str
    record_token: str


@dataclass(frozen=True)
class LocalQ1FreezeResult:
    training_records: Tuple[TrainingPairRecord, ...]
    validation_records: Tuple[TrainingPairRecord, ...]
    receipt: Mapping[str, Any]


def _qualify(
    record: TrainingPairRecord,
    *,
    member_tokens: Sequence[str],
    content_tokens: Sequence[str],
) -> _QualifiedPair:
    if len(member_tokens) != 2 or len(content_tokens) != 2:
        raise LocalQ1FreezeError("a qualified pair requires exactly two endpoints")
    if any(not value for value in member_tokens) or any(
        not value for value in content_tokens
    ):
        raise LocalQ1FreezeError("qualified endpoint identity is incomplete")
    dataset_name = _receipt_dataset(record.dataset_id)
    component_token = _sha_object(["component", dataset_name, record.component_id])
    endpoints = sorted(zip(member_tokens, content_tokens))
    pair_identity_token = _sha_object(
        {
            "dataset": dataset_name,
            "component": component_token,
            "endpoints": endpoints,
        }
    )
    record_token = _sha_object(
        {
            "pair_identity": pair_identity_token,
            "label": record.label,
        }
    )
    return _QualifiedPair(
        record=record,
        dataset_name=dataset_name,
        component_token=component_token,
        member_tokens=tuple(member_tokens),
        content_tokens=tuple(content_tokens),
        pair_identity_token=pair_identity_token,
        record_token=record_token,
    )


def _qualify_historical(verified: VerifiedPair) -> _QualifiedPair:
    return _qualify(
        verified.record,
        member_tokens=(
            verified.fragment_a.physical_member_key,
            verified.fragment_b.physical_member_key,
        ),
        content_tokens=(
            verified.fragment_a.content_sha256,
            verified.fragment_b.content_sha256,
        ),
    )


def _qualify_synthetic(record: TrainingPairRecord) -> _QualifiedPair:
    if record.dataset_id != SYNTHETIC_DATASET_ID:
        raise LocalQ1FreezeError("canonical-new stream changed dataset identity")
    if record.split != "train":
        raise LocalQ1FreezeError("canonical-new is training-only")
    for ref in (record.fragment_a, record.fragment_b):
        if ref.binding != SYNTHETIC_ARCHIVE_BINDING:
            raise LocalQ1FreezeError("canonical-new archive binding changed")
        if ref.content_sha256 is None:
            raise LocalQ1FreezeError("canonical-new endpoint lacks content SHA-256")
    return _qualify(
        record,
        member_tokens=(
            "{}\0{}".format(
                record.fragment_a.binding.sha256, record.fragment_a.archive_member
            ),
            "{}\0{}".format(
                record.fragment_b.binding.sha256, record.fragment_b.archive_member
            ),
        ),
        content_tokens=(
            str(record.fragment_a.content_sha256),
            str(record.fragment_b.content_sha256),
        ),
    )


def _update_validation_fingerprint(
    digest: "hashlib._Hash", record: TrainingPairRecord
) -> None:
    payload = {
        "pair_id": record.pair_id,
        "label": record.label,
        "dataset_id": record.dataset_id,
        "group_id": record.canonical_group_id,
        "component_id": record.component_id,
        "fragment_a": record.fragment_a.archive_member,
        "fragment_b": record.fragment_b.archive_member,
    }
    digest.update(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )


class LocalQ1RouteAFrontierSpool:
    """Public source-to-frontier ingress for live Route-A qualification.

    The spool performs the same identity verification, validation-universe
    overlap quarantine, and SHA ranking as the later refreeze.  The refreeze
    independently rebuilds and replays the result, so this pre-authority path
    cannot silently omit a higher-ranked record.
    """

    def __init__(
        self,
        *,
        database: Path,
        identity_index: HistoricalIdentityIndex,
        seed: str = SELECTION_SEED,
        target_per_dataset_label: int = TRAIN_TARGET_PER_DATASET_LABEL,
        validation_caps: Mapping[str, int] = VALIDATION_CAPS,
        expected_validation_counts: Mapping[
            Tuple[str, bool], int
        ] = EXPECTED_VALIDATION_COUNTS,
        expected_validation_fingerprint: str = EXPECTED_VALIDATION_FINGERPRINT,
        expected_validation_component_counts: Mapping[
            str, int
        ] = EXPECTED_VALIDATION_COMPONENT_COUNTS,
        expected_train_input_counts: Mapping[
            Tuple[str, bool], int
        ] = EXPECTED_TRAIN_INPUT_COUNTS,
        _fixture_scale_test_only: bool = False,
    ) -> None:
        if not isinstance(identity_index, HistoricalIdentityIndex):
            raise TypeError("identity_index must be HistoricalIdentityIndex")
        if type(_fixture_scale_test_only) is not bool:  # noqa: E721
            raise TypeError("_fixture_scale_test_only must be bool")
        if (
            not seed
            or isinstance(target_per_dataset_label, bool)
            or not isinstance(target_per_dataset_label, int)
            or target_per_dataset_label <= 0
            or set(validation_caps) != {"mm_augmented", "eccv_1113data"}
            or any(
                isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
                for cap in validation_caps.values()
            )
        ):
            raise LocalQ1FreezeError("invalid Route-A frontier configuration")
        if not _fixture_scale_test_only and (
            seed != SELECTION_SEED
            or target_per_dataset_label != TRAIN_TARGET_PER_DATASET_LABEL
            or dict(validation_caps) != VALIDATION_CAPS
            or dict(expected_validation_counts) != dict(EXPECTED_VALIDATION_COUNTS)
            or expected_validation_fingerprint != EXPECTED_VALIDATION_FINGERPRINT
            or dict(expected_validation_component_counts)
            != EXPECTED_VALIDATION_COMPONENT_COUNTS
            or dict(expected_train_input_counts) != EXPECTED_TRAIN_INPUT_COUNTS
        ):
            raise LocalQ1FreezeError(
                "production Route-A frontier differs from preregistered population"
            )
        self._frontier = LocalQ1SelectionFrontier(Path(database))
        self._identity_index = identity_index
        self._seed = seed
        self._target = target_per_dataset_label
        self._validation_caps = dict(validation_caps)
        self._expected_validation_counts = dict(expected_validation_counts)
        self._expected_validation_fingerprint = expected_validation_fingerprint
        self._expected_validation_component_counts = dict(
            expected_validation_component_counts
        )
        self._expected_train_input_counts = dict(expected_train_input_counts)
        self._fixture_scale_test_only = _fixture_scale_test_only
        self._validation_digest = hashlib.sha256()
        self._validation_input_counts = Counter()
        self._validation_component_counts = Counter()
        self._validation_component_rows = Counter()
        self._validation_count = 0
        self._training_input_counts = Counter()
        self._training_ordinal = 0
        self._validation_sealed = False
        self._selected = False
        self._closed = False
        val_universe = identity_index.identities_for_split("val")
        self._val_universe_components = {
            _sha_object(
                ["component", _receipt_dataset(item.dataset_id), item.component_id]
            )
            for item in val_universe
        }
        self._val_universe_members = {
            item.physical_member_key for item in val_universe
        }
        self._val_universe_content = {item.content_sha256 for item in val_universe}

    def add_validation(self, record: TrainingPairRecord) -> None:
        if self._closed or self._validation_sealed:
            raise LocalQ1FreezeError("validation frontier is no longer writable")
        try:
            verified = self._identity_index.verify_record(record)
        except HistoricalIdentityError as exc:
            raise LocalQ1FreezeError(
                "validation identity verification failed"
            ) from exc
        if verified.record.split != "val":
            raise LocalQ1FreezeError("validation stream contains a non-val record")
        item = _qualify_historical(verified)
        if item.dataset_name not in self._validation_caps:
            raise LocalQ1FreezeError("validation stream contains an unknown dataset")
        ordinal = self._validation_count
        self._validation_count += 1
        _update_validation_fingerprint(self._validation_digest, verified.record)
        self._validation_input_counts[
            (verified.record.dataset_id, verified.record.label)
        ] += 1
        component_key = (item.dataset_name, item.component_token)
        if self._validation_component_rows[component_key] == 0:
            self._validation_component_counts[item.dataset_name] += 1
        self._validation_component_rows[component_key] += 1
        try:
            self._frontier.add(
                LocalQ1FrontierCandidate(
                    record=item.record,
                    dataset_name=item.dataset_name,
                    component_token=item.component_token,
                    pair_identity_token=item.pair_identity_token,
                    record_token=item.record_token,
                    record_rank=_priority(
                        self._seed,
                        "validation-record-label-blind|{}".format(
                            item.dataset_name
                        ),
                        item.pair_identity_token,
                    ),
                    component_rank=_priority(
                        self._seed,
                        "validation-component|{}".format(item.dataset_name),
                        item.component_token,
                    ),
                    source_ordinal=ordinal,
                )
            )
        except (LocalQ1SelectionFrontierError, sqlite3.Error) as exc:
            raise LocalQ1FreezeError(
                "cannot persist complete validation selection frontier"
            ) from exc

    def seal_validation(self) -> None:
        if self._closed or self._validation_sealed:
            raise LocalQ1FreezeError("validation frontier sealing order changed")
        if dict(self._validation_input_counts) != self._expected_validation_counts:
            raise LocalQ1FreezeError("full validation dataset/class counts changed")
        if self._validation_digest.hexdigest() != (
            self._expected_validation_fingerprint
        ):
            raise LocalQ1FreezeError("full validation order fingerprint changed")
        if dict(self._validation_component_counts) != (
            self._expected_validation_component_counts
        ):
            raise LocalQ1FreezeError("full validation component counts changed")
        self._validation_sealed = True

    def _add_training(self, item: _QualifiedPair) -> None:
        if self._closed or not self._validation_sealed or self._selected:
            raise LocalQ1FreezeError("training frontier ingress order changed")
        record = item.record
        if record.split != "train":
            raise LocalQ1FreezeError("training stream contains a non-train record")
        stratum = (record.dataset_id, record.label)
        if stratum not in self._expected_train_input_counts:
            raise LocalQ1FreezeError("training stream contains an unknown stratum")
        self._training_input_counts[stratum] += 1
        source_ordinal = self._training_ordinal
        self._training_ordinal += 1
        if item.component_token in self._val_universe_components:
            raise LocalQ1FreezeError("train/validation component overlap detected")
        if set(item.member_tokens) & self._val_universe_members:
            raise LocalQ1FreezeError(
                "train/validation physical-member overlap detected"
            )
        if set(item.content_tokens) & self._val_universe_content:
            return
        label_value = int(record.label)
        try:
            self._frontier.add(
                LocalQ1FrontierCandidate(
                    record=record,
                    dataset_name=item.dataset_name,
                    component_token=item.component_token,
                    pair_identity_token=item.pair_identity_token,
                    record_token=item.record_token,
                    record_rank=_priority(
                        self._seed,
                        "train-record|{}|{}".format(
                            item.dataset_name, label_value
                        ),
                        item.pair_identity_token,
                    ),
                    component_rank=_priority(
                        self._seed,
                        "train-component|{}|{}".format(
                            item.dataset_name, label_value
                        ),
                        item.component_token,
                    ),
                    source_ordinal=source_ordinal,
                )
            )
        except (LocalQ1SelectionFrontierError, sqlite3.Error) as exc:
            raise LocalQ1FreezeError(
                "cannot persist complete training selection frontier"
            ) from exc

    def add_historical_training(self, record: TrainingPairRecord) -> None:
        try:
            verified = self._identity_index.verify_record(record)
        except HistoricalIdentityError as exc:
            raise LocalQ1FreezeError(
                "training identity verification failed"
            ) from exc
        self._add_training(_qualify_historical(verified))

    def add_synthetic_training(self, record: TrainingPairRecord) -> None:
        self._add_training(_qualify_synthetic(record))

    def select(
        self, *, decide: Any
    ) -> LocalQ1FrontierSelectionResult:
        if self._closed or not self._validation_sealed or self._selected:
            raise LocalQ1FreezeError("Route-A frontier selection order changed")
        if dict(self._training_input_counts) != self._expected_train_input_counts:
            raise LocalQ1FreezeError("full training dataset/class counts changed")
        try:
            result = self._frontier.select(
                decide=decide,
                train_datasets=(
                    "mm_augmented",
                    "eccv_1113data",
                    SYNTHETIC_RECEIPT_NAME,
                ),
                train_target_per_dataset_label=self._target,
                validation_caps=self._validation_caps,
            )
        except (LocalQ1SelectionFrontierError, sqlite3.Error) as exc:
            raise LocalQ1FreezeError(
                "Route-A complete lazy frontier failed closed"
            ) from exc
        self._selected = True
        return result

    def close(self) -> None:
        if not self._closed:
            self._frontier.close()
            self._closed = True

    def __enter__(self) -> "LocalQ1RouteAFrontierSpool":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


_HeapEntry = Tuple[int, str, int, _QualifiedPair]


def _push_lowest(
    heap: list[_HeapEntry],
    item: _QualifiedPair,
    *,
    priority: str,
    ordinal: int,
    limit: int,
) -> None:
    if limit <= 0:
        raise LocalQ1FreezeError("selection heap limit must be positive")
    numeric = int(priority, 16)
    entry = (-numeric, priority, ordinal, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif numeric < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def _trim_heap(heap: list[_HeapEntry], limit: int) -> None:
    if len(heap) <= limit:
        return
    retained = sorted(heap, key=lambda entry: (entry[1], entry[2]))[:limit]
    heap[:] = retained
    heapq.heapify(heap)


def _minimum_round_robin_depth(counts: Iterable[int], target: int) -> int:
    values = tuple(int(value) for value in counts if int(value) > 0)
    if sum(values) < target:
        return target
    low, high = 1, min(target, max(values))
    while low < high:
        middle = (low + high) // 2
        if sum(min(value, middle) for value in values) >= target:
            high = middle
        else:
            low = middle + 1
    return low


class _TrainBucket:
    """Bounded exact top-hash records for component round-robin selection."""

    def __init__(self, *, seed: str, dataset_name: str, label: bool, target: int):
        self.seed = seed
        self.dataset_name = dataset_name
        self.label = label
        self.target = target
        self.counts: Counter[str] = Counter()
        self.heaps: MutableMapping[str, list[_HeapEntry]] = defaultdict(list)
        self.depth_limit = target
        self.ordinal = 0

    def consider(self, item: _QualifiedPair) -> None:
        component = item.component_token
        new_component = component not in self.counts
        self.counts[component] += 1
        self.ordinal += 1
        if new_component:
            next_limit = _minimum_round_robin_depth(self.counts.values(), self.target)
            if next_limit < self.depth_limit:
                self.depth_limit = next_limit
                for heap in self.heaps.values():
                    _trim_heap(heap, self.depth_limit)
        rank = _priority(
            self.seed,
            "train-record|{}|{}".format(self.dataset_name, int(self.label)),
            item.pair_identity_token,
        )
        _push_lowest(
            self.heaps[component],
            item,
            priority=rank,
            ordinal=self.ordinal,
            limit=self.depth_limit,
        )

    def select(self) -> Tuple[_QualifiedPair, ...]:
        if sum(self.counts.values()) < self.target:
            raise LocalQ1FreezeError(
                "insufficient eligible {} label {} records: {} < {}".format(
                    self.dataset_name,
                    int(self.label),
                    sum(self.counts.values()),
                    self.target,
                )
            )
        exact_depth = _minimum_round_robin_depth(self.counts.values(), self.target)
        if exact_depth > self.depth_limit:
            raise LocalQ1FreezeError("internal round-robin depth underflow")
        for heap in self.heaps.values():
            _trim_heap(heap, exact_depth)
        components = sorted(
            self.heaps,
            key=lambda component: (
                _priority(
                    self.seed,
                    "train-component|{}|{}".format(self.dataset_name, int(self.label)),
                    component,
                ),
                component,
            ),
        )
        ordered = {
            component: tuple(
                entry[3]
                for entry in sorted(
                    self.heaps[component], key=lambda entry: (entry[1], entry[2])
                )
            )
            for component in components
        }
        selected = []
        depth = 0
        while len(selected) < self.target:
            added = 0
            for component in components:
                items = ordered[component]
                if depth < len(items):
                    selected.append(items[depth])
                    added += 1
                    if len(selected) == self.target:
                        break
            if not added:
                break
            depth += 1
        if len(selected) != self.target:
            raise LocalQ1FreezeError(
                "component round robin could not reach the frozen target"
            )
        if len({item.pair_identity_token for item in selected}) != len(selected):
            raise LocalQ1FreezeError("selected training population contains duplicates")
        return tuple(selected)


def _archive_lock(
    binding: ArchiveBinding, *, bytes_count: int, sha256: str
) -> Mapping[str, Any]:
    if type(bytes_count) is not int or bytes_count <= 0:  # noqa: E721
        raise LocalQ1FreezeError("archive byte count is invalid")
    if sha256 != binding.sha256:
        raise LocalQ1FreezeError("historical archive lock is not canonical")
    return {
        "format": binding.archive_format,
        "logical_id": binding.logical_id,
        "bytes": bytes_count,
        "sha256": sha256,
    }


def _validate_artifact_lock(lock: Mapping[str, Any], *, expected_sha: str) -> None:
    if set(lock) != {"bytes", "sha256"}:
        raise LocalQ1FreezeError("metadata artifact lock shape changed")
    if type(lock.get("bytes")) is not int or int(lock["bytes"]) <= 0:  # noqa: E721
        raise LocalQ1FreezeError("metadata artifact byte count is invalid")
    if lock.get("sha256") != expected_sha:
        raise LocalQ1FreezeError("metadata artifact SHA-256 changed")


def _population_commitments(items: Sequence[_QualifiedPair]) -> Mapping[str, Any]:
    count_payload = {
        "count": len(items),
        "by_dataset": dict(
            sorted(Counter(item.dataset_name for item in items).items())
        ),
        "by_dataset_label": dict(
            sorted(
                Counter(
                    "{}|{}".format(
                        item.dataset_name,
                        "positive" if item.record.label else "negative",
                    )
                    for item in items
                ).items()
            )
        ),
        "unique_component_count": len({item.component_token for item in items}),
    }
    return {
        **count_payload,
        "count_commitment_sha256": _sha_object(count_payload),
        "record_set_commitment_sha256": _commit_set(
            item.record_token for item in items
        ),
        "record_order_commitment_sha256": _commit_order(
            item.record_token for item in items
        ),
        "dataset_sequence_commitment_sha256": _commit_order(
            item.dataset_name for item in items
        ),
        "class_sequence_commitment_sha256": _commit_order(
            "positive" if item.record.label else "negative" for item in items
        ),
        "component_set_commitment_sha256": _commit_set(
            item.component_token for item in items
        ),
        "component_sequence_commitment_sha256": _commit_order(
            item.component_token for item in items
        ),
        "physical_member_set_commitment_sha256": _commit_set(
            member for item in items for member in item.member_tokens
        ),
        "endpoint_content_set_commitment_sha256": _commit_set(
            content for item in items for content in item.content_tokens
        ),
    }


def freeze_local_q1(
    *,
    identity_index: HistoricalIdentityIndex,
    validation_records: Iterable[TrainingPairRecord],
    historical_training_records: Iterable[TrainingPairRecord],
    synthetic_training_records: Iterable[TrainingPairRecord],
    historical_archive_locks: Mapping[str, Mapping[str, Any]],
    synthetic_manifest_lock: Mapping[str, Any],
    seed: str = SELECTION_SEED,
    target_per_dataset_label: int = TRAIN_TARGET_PER_DATASET_LABEL,
    validation_caps: Mapping[str, int] = VALIDATION_CAPS,
    expected_validation_counts: Mapping[
        Tuple[str, bool], int
    ] = EXPECTED_VALIDATION_COUNTS,
    expected_validation_fingerprint: str = EXPECTED_VALIDATION_FINGERPRINT,
    expected_validation_component_counts: Mapping[
        str, int
    ] = EXPECTED_VALIDATION_COMPONENT_COUNTS,
    expected_train_input_counts: Mapping[
        Tuple[str, bool], int
    ] = EXPECTED_TRAIN_INPUT_COUNTS,
    expected_synthetic_manifest_sha256: str = SYNTHETIC_MANIFEST_SHA256,
    geometry_eligibility: Optional[LocalQ1GeometryEligibilityAuthority] = None,
    frontier_database: Optional[Path] = None,
    expected_predecessor_freeze_file_sha256: Optional[str] = None,
    expected_predecessor_freeze_content_sha256: Optional[str] = None,
    expected_predecessor_freeze_bytes: Optional[int] = None,
    predecessor_freeze_file_bytes: Optional[bytes] = None,
    synthetic_archive_lock: Optional[Mapping[str, Any]] = None,
    expected_source_bundle_manifest_sha256: Optional[str] = None,
    expected_geometry_batch_config_sha256: Optional[str] = None,
    expected_planning_guard_config_sha256: Optional[str] = None,
    _fixture_scale_test_only: bool = False,
) -> LocalQ1FreezeResult:
    """Freeze exact local train/validation records without loading mask pixels.

    With ``geometry_eligibility`` absent this replays the predecessor metadata
    freeze byte-for-byte.  Route A requires an externally bound eligibility
    authority and uses a complete disk-backed metadata frontier; only the
    eligibility run, never this refreeze function, opens mask members.
    """

    if not seed or target_per_dataset_label <= 0:
        raise LocalQ1FreezeError("invalid local-Q1 training selection configuration")
    if type(_fixture_scale_test_only) is not bool:  # noqa: E721
        raise TypeError("_fixture_scale_test_only must be bool")
    if geometry_eligibility is not None and not isinstance(
        geometry_eligibility, LocalQ1GeometryEligibilityAuthority
    ):
        raise TypeError(
            "geometry_eligibility must be LocalQ1GeometryEligibilityAuthority"
        )
    if geometry_eligibility is None and frontier_database is not None:
        raise LocalQ1FreezeError(
            "frontier_database is meaningful only for Route-A eligibility"
        )
    route_a_expected_hashes = (
        expected_predecessor_freeze_file_sha256,
        expected_predecessor_freeze_content_sha256,
        expected_source_bundle_manifest_sha256,
        expected_geometry_batch_config_sha256,
        expected_planning_guard_config_sha256,
    )
    if geometry_eligibility is None and (
        any(value is not None for value in route_a_expected_hashes)
        or expected_predecessor_freeze_bytes is not None
        or predecessor_freeze_file_bytes is not None
        or synthetic_archive_lock is not None
        or _fixture_scale_test_only
    ):
        raise LocalQ1FreezeError(
            "Route-A external hashes cannot be supplied to legacy freeze"
        )
    if geometry_eligibility is not None and any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in route_a_expected_hashes
    ):
        raise LocalQ1FreezeError(
            "Route-A requires every predecessor/source/config external hash"
        )
    if geometry_eligibility is not None and (
        geometry_eligibility.production_authorized == _fixture_scale_test_only
    ):
        raise LocalQ1FreezeError(
            "fixture-scale and production Route-A authorities cannot be confused"
        )
    if geometry_eligibility is not None and not _fixture_scale_test_only and (
        seed != SELECTION_SEED
        or target_per_dataset_label != TRAIN_TARGET_PER_DATASET_LABEL
        or dict(validation_caps) != VALIDATION_CAPS
        or dict(expected_validation_counts) != dict(EXPECTED_VALIDATION_COUNTS)
        or expected_validation_fingerprint != EXPECTED_VALIDATION_FINGERPRINT
        or dict(expected_validation_component_counts)
        != EXPECTED_VALIDATION_COMPONENT_COUNTS
        or dict(expected_train_input_counts) != EXPECTED_TRAIN_INPUT_COUNTS
        or expected_synthetic_manifest_sha256 != SYNTHETIC_MANIFEST_SHA256
    ):
        raise LocalQ1FreezeError(
            "Route-A production population/config arguments differ from authority"
        )
    if geometry_eligibility is not None and (
        isinstance(expected_predecessor_freeze_bytes, bool)
        or not isinstance(expected_predecessor_freeze_bytes, int)
        or expected_predecessor_freeze_bytes <= 0
        or not isinstance(predecessor_freeze_file_bytes, bytes)
        or not isinstance(synthetic_archive_lock, Mapping)
        or set(synthetic_archive_lock) != {"bytes", "sha256"}
        or isinstance(synthetic_archive_lock.get("bytes"), bool)
        or not isinstance(synthetic_archive_lock.get("bytes"), int)
        or synthetic_archive_lock.get("bytes") <= 0
        or synthetic_archive_lock.get("sha256")
        != SYNTHETIC_ARCHIVE_BINDING.sha256
    ):
        raise LocalQ1FreezeError(
            "Route-A requires exact predecessor/synthetic archive byte locks"
        )
    predecessor_receipt = None
    if geometry_eligibility is not None:
        assert expected_predecessor_freeze_bytes is not None
        assert expected_predecessor_freeze_file_sha256 is not None
        assert expected_predecessor_freeze_content_sha256 is not None
        assert predecessor_freeze_file_bytes is not None
        predecessor_receipt = _load_locked_predecessor_freeze_receipt(
            predecessor_freeze_file_bytes,
            expected_bytes=expected_predecessor_freeze_bytes,
            expected_file_sha256=expected_predecessor_freeze_file_sha256,
            expected_content_sha256=expected_predecessor_freeze_content_sha256,
        )
    if set(validation_caps) != {"mm_augmented", "eccv_1113data"} or any(
        type(cap) is not int or cap <= 0
        for cap in validation_caps.values()  # noqa: E721
    ):
        raise LocalQ1FreezeError("validation requires exact positive per-domain caps")
    if set(historical_archive_locks) != {"mm_augmented", "eccv_1113data"}:
        raise LocalQ1FreezeError("exact MM/ECCV archive locks are required")
    canonical_bindings = {
        "mm_augmented": MM_CANONICAL_BINDING,
        "eccv_1113data": ECCV_CANONICAL_BINDING,
    }
    normalized_archive_locks = {}
    for dataset, binding in canonical_bindings.items():
        lock = historical_archive_locks[dataset]
        if set(lock) != {"format", "logical_id", "bytes", "sha256"}:
            raise LocalQ1FreezeError("historical archive lock shape changed")
        if (
            lock.get("format") != binding.archive_format
            or lock.get("logical_id") != binding.logical_id
        ):
            raise LocalQ1FreezeError("historical archive lock binding changed")
        normalized_archive_locks[dataset] = dict(
            _archive_lock(
                binding,
                bytes_count=lock.get("bytes"),
                sha256=lock.get("sha256"),
            )
        )
    _validate_artifact_lock(
        synthetic_manifest_lock, expected_sha=expected_synthetic_manifest_sha256
    )
    if geometry_eligibility is not None:
        trust = geometry_eligibility.trust
        expected_authority_values = (
            trust.predecessor_freeze_file_sha256,
            trust.predecessor_freeze_content_sha256,
            trust.source_bundle_manifest_sha256,
            trust.geometry_batch_config_sha256,
            trust.planning_guard_config_sha256,
        )
        if expected_authority_values != route_a_expected_hashes:
            raise LocalQ1FreezeError(
                "Route-A authority differs from predecessor/source/config locks"
            )
        actual_roles = {
            "mm_archive": {
                "bytes": normalized_archive_locks["mm_augmented"]["bytes"],
                "sha256": normalized_archive_locks["mm_augmented"]["sha256"],
            },
            "eccv_archive": {
                "bytes": normalized_archive_locks["eccv_1113data"]["bytes"],
                "sha256": normalized_archive_locks["eccv_1113data"]["sha256"],
            },
            "mm_fingerprint_cache": dict(
                identity_index.artifact_locks["mm_fingerprint_cache"]
            ),
            "eccv_fingerprint_cache": dict(
                identity_index.artifact_locks["eccv_fingerprint_cache"]
            ),
            "historical_split": dict(
                identity_index.artifact_locks["historical_split"]
            ),
            "synthetic_manifest": dict(synthetic_manifest_lock),
        }
        for role, observed in actual_roles.items():
            if trust.input_role_locks[role].portable_dict() != observed:
                raise LocalQ1FreezeError(
                    "Route-A authority input role differs from refreeze: " + role
                )
        if (
            trust.input_role_locks["freeze_receipt"].portable_dict()
            != {
                "bytes": expected_predecessor_freeze_bytes,
                "sha256": expected_predecessor_freeze_file_sha256,
            }
            or trust.input_role_locks["synthetic_archive"].portable_dict()
            != dict(synthetic_archive_lock)
        ):
            raise LocalQ1FreezeError(
                "Route-A predecessor/synthetic archive role lock changed"
            )
        assert predecessor_receipt is not None
        _assert_predecessor_input_lock_replay(
            predecessor_receipt,
            normalized_archive_locks=normalized_archive_locks,
            identity_index=identity_index,
            synthetic_manifest_lock=synthetic_manifest_lock,
        )

    val_universe = identity_index.identities_for_split("val")
    val_universe_components = {
        _sha_object(["component", _receipt_dataset(item.dataset_id), item.component_id])
        for item in val_universe
    }
    val_universe_members = {item.physical_member_key for item in val_universe}
    val_universe_content = {item.content_sha256 for item in val_universe}

    frontier_temporary = None
    frontier = None
    if geometry_eligibility is not None:
        if frontier_database is None:
            frontier_temporary = tempfile.TemporaryDirectory(
                prefix="local_q1_route_a_frontier_"
            )
            frontier_path = Path(frontier_temporary.name) / "frontier.sqlite3"
        else:
            frontier_path = Path(frontier_database)
        try:
            frontier = LocalQ1SelectionFrontier(frontier_path)
        except LocalQ1SelectionFrontierError as exc:
            raise LocalQ1FreezeError(
                "cannot create complete Route-A selection frontier"
            ) from exc

    validation_digest = hashlib.sha256()
    validation_input_counts: Counter[Tuple[str, bool]] = Counter()
    validation_component_counts: Counter[str] = Counter()
    validation_component_rows: Counter[Tuple[str, str]] = Counter()
    validation_heaps: MutableMapping[Tuple[str, str], list[_HeapEntry]] = defaultdict(
        list
    )
    validation_count = 0
    for validation_count, record in enumerate(validation_records, start=1):
        try:
            verified = identity_index.verify_record(record)
        except HistoricalIdentityError as exc:
            raise LocalQ1FreezeError("validation identity verification failed") from exc
        if verified.record.split != "val":
            raise LocalQ1FreezeError("validation stream contains a non-val record")
        qualified = _qualify_historical(verified)
        if qualified.dataset_name not in validation_caps:
            raise LocalQ1FreezeError("validation stream contains an unknown dataset")
        _update_validation_fingerprint(validation_digest, verified.record)
        validation_input_counts[
            (verified.record.dataset_id, verified.record.label)
        ] += 1
        component_key = (qualified.dataset_name, qualified.component_token)
        if validation_component_rows[component_key] == 0:
            validation_component_counts[qualified.dataset_name] += 1
        validation_component_rows[component_key] += 1
        # The ranking namespace and token deliberately contain neither label nor
        # direction.  Labels are inspected only after the capped rows are fixed.
        rank = _priority(
            seed,
            "validation-record-label-blind|{}".format(qualified.dataset_name),
            qualified.pair_identity_token,
        )
        if frontier is None:
            _push_lowest(
                validation_heaps[component_key],
                qualified,
                priority=rank,
                ordinal=validation_count - 1,
                limit=validation_caps[qualified.dataset_name],
            )
        else:
            try:
                frontier.add(
                    LocalQ1FrontierCandidate(
                        record=qualified.record,
                        dataset_name=qualified.dataset_name,
                        component_token=qualified.component_token,
                        pair_identity_token=qualified.pair_identity_token,
                        record_token=qualified.record_token,
                        record_rank=rank,
                        component_rank=_priority(
                            seed,
                            "validation-component|{}".format(
                                qualified.dataset_name
                            ),
                            qualified.component_token,
                        ),
                        source_ordinal=validation_count - 1,
                    )
                )
            except (LocalQ1SelectionFrontierError, sqlite3.Error) as exc:
                raise LocalQ1FreezeError(
                    "cannot persist complete validation selection frontier"
                ) from exc

    if dict(validation_input_counts) != dict(expected_validation_counts):
        raise LocalQ1FreezeError("full validation dataset/class counts changed")
    if validation_digest.hexdigest() != expected_validation_fingerprint:
        raise LocalQ1FreezeError("full validation order fingerprint changed")
    if dict(validation_component_counts) != dict(expected_validation_component_counts):
        raise LocalQ1FreezeError("full validation component counts changed")

    if frontier is None:
        retained_validation = sorted(
            (entry[2], entry[3])
            for heap in validation_heaps.values()
            for entry in heap
        )
        retained_indices = [index for index, _item in retained_validation]
        if retained_indices != sorted(retained_indices) or len(retained_indices) != len(
            set(retained_indices)
        ):
            raise LocalQ1FreezeError("validation source-order subsequence is invalid")
        selected_validation = tuple(item for _index, item in retained_validation)
    else:
        selected_validation = ()

    train_buckets = {
        (dataset, label): _TrainBucket(
            seed=seed,
            dataset_name=_receipt_dataset(dataset),
            label=label,
            target=target_per_dataset_label,
        )
        for dataset in (
            "mm_augmented",
            "eccv_1113data",
            SYNTHETIC_DATASET_ID,
        )
        for label in (False, True)
    }
    training_input_counts: Counter[Tuple[str, bool]] = Counter()
    quarantine_counts: Counter[Tuple[str, bool]] = Counter()
    quarantine_endpoint_hits = 0
    training_ordinal = 0

    def admit_train(item: _QualifiedPair) -> None:
        nonlocal quarantine_endpoint_hits, training_ordinal
        record = item.record
        if record.split != "train":
            raise LocalQ1FreezeError("training stream contains a non-train record")
        stratum = (record.dataset_id, record.label)
        if stratum not in train_buckets:
            raise LocalQ1FreezeError("training stream contains an unknown stratum")
        training_input_counts[stratum] += 1
        source_ordinal = training_ordinal
        training_ordinal += 1
        if item.component_token in val_universe_components:
            raise LocalQ1FreezeError("train/validation component overlap detected")
        if set(item.member_tokens) & val_universe_members:
            raise LocalQ1FreezeError(
                "train/validation physical-member overlap detected"
            )
        hits = sum(content in val_universe_content for content in item.content_tokens)
        if hits:
            quarantine_counts[stratum] += 1
            quarantine_endpoint_hits += hits
            return
        if frontier is None:
            train_buckets[stratum].consider(item)
            return
        dataset_name = item.dataset_name
        label_value = int(record.label)
        try:
            frontier.add(
                LocalQ1FrontierCandidate(
                    record=record,
                    dataset_name=dataset_name,
                    component_token=item.component_token,
                    pair_identity_token=item.pair_identity_token,
                    record_token=item.record_token,
                    record_rank=_priority(
                        seed,
                        "train-record|{}|{}".format(dataset_name, label_value),
                        item.pair_identity_token,
                    ),
                    component_rank=_priority(
                        seed,
                        "train-component|{}|{}".format(dataset_name, label_value),
                        item.component_token,
                    ),
                    source_ordinal=source_ordinal,
                )
            )
        except (LocalQ1SelectionFrontierError, sqlite3.Error) as exc:
            raise LocalQ1FreezeError(
                "cannot persist complete training selection frontier"
            ) from exc

    for record in historical_training_records:
        try:
            verified = identity_index.verify_record(record)
        except HistoricalIdentityError as exc:
            raise LocalQ1FreezeError("training identity verification failed") from exc
        admit_train(_qualify_historical(verified))
    for record in synthetic_training_records:
        admit_train(_qualify_synthetic(record))

    if dict(training_input_counts) != dict(expected_train_input_counts):
        raise LocalQ1FreezeError("full training dataset/class counts changed")

    route_a_selection_proof = None
    if frontier is None:
        selected_train = []
        for dataset in (
            "mm_augmented",
            "eccv_1113data",
            SYNTHETIC_DATASET_ID,
        ):
            for label in (False, True):
                selected_train.extend(train_buckets[(dataset, label)].select())
        selected_train_tuple = tuple(selected_train)
    else:
        assert geometry_eligibility is not None
        try:
            route_a_result = frontier.select(
                decide=geometry_eligibility,
                train_datasets=(
                    "mm_augmented",
                    "eccv_1113data",
                    SYNTHETIC_RECEIPT_NAME,
                ),
                train_target_per_dataset_label=target_per_dataset_label,
                validation_caps=validation_caps,
            )
            observed_decision_order = tuple(
                decision.pair_identity_sha256
                for decision in route_a_result.decisions
            )
            if observed_decision_order != geometry_eligibility.decision_order:
                raise LocalQ1FreezeError(
                    "eligibility index contains missing, extra, or reordered decisions"
                )
            geometry_eligibility.assert_selection_proof(
                route_a_result.selection_proof
            )

            def requalify(record: TrainingPairRecord) -> _QualifiedPair:
                if record.dataset_id == SYNTHETIC_DATASET_ID:
                    return _qualify_synthetic(record)
                try:
                    return _qualify_historical(identity_index.verify_record(record))
                except HistoricalIdentityError as exc:
                    raise LocalQ1FreezeError(
                        "Route-A selected record identity verification failed"
                    ) from exc

            selected_train_tuple = tuple(
                requalify(record) for record in route_a_result.training_records
            )
            selected_validation = tuple(
                requalify(record) for record in route_a_result.validation_records
            )
            route_a_selection_proof = _thaw_json(
                route_a_result.selection_proof
            )
            assert predecessor_receipt is not None
            _assert_predecessor_population_replay(
                predecessor_receipt, route_a_selection_proof
            )
        except (
            LocalQ1SelectionFrontierError,
            LocalQ1GeometryQualificationError,
            sqlite3.Error,
        ) as exc:
            raise LocalQ1FreezeError(
                "Route-A complete lazy frontier failed closed"
            ) from exc
        finally:
            frontier.close()
            if frontier_temporary is not None:
                frontier_temporary.cleanup()

    selected_train_components = {item.component_token for item in selected_train_tuple}
    selected_train_members = {
        token for item in selected_train_tuple for token in item.member_tokens
    }
    selected_train_content = {
        token for item in selected_train_tuple for token in item.content_tokens
    }
    selected_val_components = {item.component_token for item in selected_validation}
    selected_val_members = {
        token for item in selected_validation for token in item.member_tokens
    }
    selected_val_content = {
        token for item in selected_validation for token in item.content_tokens
    }
    selected_overlaps = {
        "component": len(selected_train_components & selected_val_components),
        "physical_member": len(selected_train_members & selected_val_members),
        "content_sha256": len(selected_train_content & selected_val_content),
    }
    full_validation_overlaps = {
        "component": len(selected_train_components & val_universe_components),
        "physical_member": len(selected_train_members & val_universe_members),
        "content_sha256": len(selected_train_content & val_universe_content),
    }
    if any(selected_overlaps.values()) or any(full_validation_overlaps.values()):
        raise LocalQ1FreezeError("frozen train/validation identity overlap is nonzero")

    train_commitments = _population_commitments(selected_train_tuple)
    validation_commitments = _population_commitments(selected_validation)
    expected_train_count = 3 * 2 * target_per_dataset_label
    if train_commitments["count"] != expected_train_count:
        raise LocalQ1FreezeError("frozen training population count changed")
    expected_validation_count = sum(
        int(expected_validation_component_counts[dataset])
        * int(validation_caps[dataset])
        for dataset in validation_caps
    )
    if validation_commitments["count"] != expected_validation_count:
        raise LocalQ1FreezeError("frozen validation population count changed")
    if route_a_selection_proof is not None:
        expected_train_strata = {
            "{}|{}".format(dataset, label): target_per_dataset_label
            for dataset in (
                "mm_augmented",
                "eccv_1113data",
                SYNTHETIC_RECEIPT_NAME,
            )
            for label in ("negative", "positive")
        }
        if train_commitments["by_dataset_label"] != expected_train_strata:
            raise LocalQ1FreezeError("Route-A training quotas changed")
        if route_a_selection_proof.get("selected_failure_counts") != {
            "fragment": 0,
            "pair": 0,
            "direction_coverage": 0,
            "resource": 0,
        }:
            raise LocalQ1FreezeError(
                "Route-A selected geometry/resource failure count is nonzero"
            )

    validation_by_dataset = {}
    for dataset in ("mm_augmented", "eccv_1113data"):
        selected_domain = [
            item for item in selected_validation if item.dataset_name == dataset
        ]
        if route_a_selection_proof is None:
            component_sizes = [
                len(heap)
                for (heap_dataset, _component), heap in validation_heaps.items()
                if heap_dataset == dataset
            ]
        else:
            component_sizes = list(
                Counter(item.component_token for item in selected_domain).values()
            )
        validation_by_dataset[dataset] = {
            "cap_per_component": validation_caps[dataset],
            "input_component_count": validation_component_counts[dataset],
            "selected_component_count": len(component_sizes),
            "selected_count": len(selected_domain),
            "selected_positive_count": sum(
                item.record.label for item in selected_domain
            ),
            "selected_negative_count": sum(
                not item.record.label for item in selected_domain
            ),
            "selected_min_per_component": min(component_sizes, default=0),
            "selected_max_per_component": max(component_sizes, default=0),
            "component_cap_violations": sum(
                count > validation_caps[dataset] for count in component_sizes
            ),
        }
        if validation_by_dataset[dataset]["component_cap_violations"]:
            raise LocalQ1FreezeError("validation component cap was violated")
        if route_a_selection_proof is not None and (
            validation_by_dataset[dataset]["selected_component_count"]
            != expected_validation_component_counts[dataset]
            or validation_by_dataset[dataset]["selected_min_per_component"]
            != validation_caps[dataset]
            or validation_by_dataset[dataset]["selected_max_per_component"]
            != validation_caps[dataset]
        ):
            raise LocalQ1FreezeError(
                "Route-A validation component coverage/cap changed"
            )

    legacy_selection_contract = {
        "seed_sha256": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        "train_policy": "sha256_record_rank_then_sha256_component_round_robin",
        "train_target_per_dataset_label": target_per_dataset_label,
        "train_dataset_order": [
            "mm_augmented",
            "eccv_1113data",
            SYNTHETIC_RECEIPT_NAME,
        ],
        "train_label_order": ["negative", "positive"],
        "validation_policy": (
            "per_component_label_blind_sha256_rank_then_restore_source_order"
        ),
        "validation_rank_fields": [
            "dataset",
            "component_commitment",
            "unordered_physical_pair_commitment",
            "endpoint_content_commitment",
        ],
        "validation_rank_excluded_fields": ["label", "direction"],
        "validation_caps": dict(sorted(validation_caps.items())),
    }
    legacy_selection_contract["selection_commitment_sha256"] = _sha_object(
        legacy_selection_contract
    )
    if route_a_selection_proof is not None:
        assert predecessor_receipt is not None
        if predecessor_receipt.get("selection_contract") != (
            legacy_selection_contract
        ):
            raise LocalQ1FreezeError(
                "Route-A selection configuration differs from locked predecessor"
            )
    selection_contract = dict(legacy_selection_contract)
    if route_a_selection_proof is not None:
        selection_contract.pop("selection_commitment_sha256")
        selection_contract.update(
            {
                "geometry_qualification_policy": (
                    "externally_locked_route_a_v0_2_fragment_pair_four_direction_"
                    "and_individual_resource_gates"
                ),
                "selection_frontier": (
                    "complete_metadata_external_sort_consumed_lazily_without_"
                    "arbitrary_reserve_cutoff"
                ),
                "manual_exclusions_permitted": False,
                "unexplored_higher_priority_record_count": 0,
                "qualification_supervision_fields_read": [],
            }
        )
        selection_contract["selection_commitment_sha256"] = _sha_object(
            selection_contract
        )

    input_train_counts_portable = {
        _stratum_name(dataset, label): count
        for (dataset, label), count in sorted(training_input_counts.items())
    }
    quarantine_counts_portable = {
        _stratum_name(dataset, label): count
        for (dataset, label), count in sorted(quarantine_counts.items())
    }
    receipt: Dict[str, Any] = {
        "schema_version": (
            ROUTE_A_SCHEMA_VERSION
            if route_a_selection_proof is not None
            else SCHEMA_VERSION
        ),
        "status": (
            (
                "fixture_scale_test_only_not_authorized_for_real_scan"
                if _fixture_scale_test_only
                else "pass_route_a_geometry_qualified_population_refrozen_no_model_no_test"
            )
            if route_a_selection_proof is not None
            else "pass_metadata_only_population_frozen_no_model_execution"
        ),
        "scope": {
            "experiment": "LOCAL-Q1",
            "task": "mask_only_known_orientation_pairwise_local_matcher_qualification",
            "datasets": [
                "mm_augmented",
                "eccv_1113data",
                SYNTHETIC_RECEIPT_NAME,
            ],
            "pair_stream_splits_read": ["train", "val"],
            "historical_test_access": dict(HISTORICAL_TEST_ACCESS_EVIDENCE),
            "synthetic_validation_read": False,
            "sealed_real_read": False,
            "archive_mask_members_opened": False,
            "mask_pixels_decoded": False,
            "model_imported": False,
            "model_executed": False,
        },
        "locks": {
            "historical_archive_containers": normalized_archive_locks,
            "historical_metadata_artifacts": {
                role: dict(lock) for role, lock in identity_index.artifact_locks.items()
            },
            "historical_identity_index": {
                "member_count": identity_index.identity_count,
                "content_sha256": historical_identity_index_content_sha256(
                    identity_index
                ),
            },
            "synthetic_manifest": dict(synthetic_manifest_lock),
            "synthetic_archive_reference": {
                "format": SYNTHETIC_ARCHIVE_BINDING.archive_format,
                "logical_id": SYNTHETIC_ARCHIVE_BINDING.logical_id,
                "sha256": SYNTHETIC_ARCHIVE_BINDING.sha256,
                "opened_by_freeze": False,
            },
        },
        "selection_contract": selection_contract,
        "training": {
            "input_count": sum(training_input_counts.values()),
            "input_by_dataset_label": input_train_counts_portable,
            "input_count_commitment_sha256": _sha_object(input_train_counts_portable),
            "content_quarantine": {
                "policy": (
                    "raw_train_row_excluded_if_either_endpoint_content_occurs_in_"
                    "complete_validation_identity_universe"
                ),
                "raw_row_count": sum(quarantine_counts.values()),
                "endpoint_hit_count": quarantine_endpoint_hits,
                "by_dataset_label_raw_row_count": quarantine_counts_portable,
            },
            "population": train_commitments,
        },
        "validation": {
            "input_count": validation_count,
            "input_by_dataset_label": {
                _stratum_name(dataset, label): count
                for (dataset, label), count in sorted(validation_input_counts.items())
            },
            "input_component_count": sum(validation_component_counts.values()),
            "full_source_order_sha256": validation_digest.hexdigest(),
            "selection_label_blind": True,
            "source_order_restored_after_selection": True,
            "by_dataset": validation_by_dataset,
            "population": validation_commitments,
            "complete_identity_universe": {
                "fragment_count": len(val_universe),
                "unique_component_count": len(val_universe_components),
                "unique_physical_member_count": len(val_universe_members),
                "unique_content_sha256_count": len(val_universe_content),
                "component_set_commitment_sha256": _commit_set(val_universe_components),
                "physical_member_set_commitment_sha256": _commit_set(
                    val_universe_members
                ),
                "content_set_commitment_sha256": _commit_set(val_universe_content),
            },
        },
        "selected_train_vs_selected_validation_overlap": selected_overlaps,
        "selected_train_vs_complete_validation_identity_universe_overlap": (
            full_validation_overlaps
        ),
        "portable_privacy": {
            "absolute_paths_present": False,
            "member_identifiers_present": False,
            "group_identifiers_present": False,
            "component_identifiers_present": False,
            "pair_identifiers_present": False,
            "fragment_content_hash_values_present": False,
            "aggregate_commitments_only": True,
        },
    }
    if route_a_selection_proof is not None:
        assert geometry_eligibility is not None
        receipt["scope"].update(
            {
                "geometry_eligibility_authority_consumed": True,
                "geometry_qualification_executed_by_refreeze": False,
                "qualification_mask_pixels_decoded_by_refreeze": False,
                "partial_cache_tree_accessed_by_refreeze": False,
                "fixture_scale_test_only": _fixture_scale_test_only,
            }
        )
        receipt["geometry_eligibility_authority"] = {
            "authority_version": (
                "dunhuang-local-q1-route-a-external-dual-hash-authority/0.1"
            ),
            "production_authorized": geometry_eligibility.production_authorized,
            "external_locks": geometry_eligibility.trust.portable_dict(),
            "eligibility_index": {
                "decision_count": len(geometry_eligibility.decision_order),
                "decision_order_sha256": geometry_eligibility.index[
                    "decision_order_sha256"
                ],
                "decision_set_sha256": geometry_eligibility.index[
                    "decision_set_sha256"
                ],
            },
            "qualification_receipt_content_sha256": (
                geometry_eligibility.trust.eligibility_receipt_content_sha256
            ),
            "selection_replayed_exactly": True,
            "partial_cache_as_membership_input_permitted": False,
            "partial_cache_role_count": 0,
            "partial_cache_tree_accessed": False,
        }
        receipt["route_a_frontier_selection_proof"] = route_a_selection_proof
        receipt["old_to_new_population_difference"] = route_a_selection_proof[
            "old_to_new_population_difference"
        ]
    receipt["content_sha256"] = _sha_object(receipt)
    assert_portable_local_q1_receipt(receipt)
    return LocalQ1FreezeResult(
        training_records=tuple(item.record for item in selected_train_tuple),
        validation_records=tuple(item.record for item in selected_validation),
        receipt=receipt,
    )


def assert_portable_local_q1_receipt(receipt: Mapping[str, Any]) -> None:
    """Fail if the aggregate receipt exposes a local path or raw identity."""

    text = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    forbidden = (
        "/Users/",
        "\\Users\\",
        "archive_member",
        "canonical_group_id",
        "component_id",
        "fragment_id",
        "pair_id",
    )
    scrubbed = text
    for allowed_assertion in (
        "archive_members_opened",
        "archive_mask_members_opened",
        "member_identifiers_present",
        "group_identifiers_present",
        "component_identifiers_present",
        "pair_identifiers_present",
    ):
        scrubbed = scrubbed.replace(allowed_assertion, "")
    if any(token in scrubbed for token in forbidden):
        raise LocalQ1FreezeError("portable local-Q1 receipt exposes private identity")


def _attest_archive(path: Path, binding: ArchiveBinding) -> Mapping[str, Any]:
    observed = sha256_file(path)
    return _archive_lock(binding, bytes_count=path.stat().st_size, sha256=observed)


def _artifact_lock(path: Path) -> Mapping[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mm-cache", type=Path, default=DEFAULT_MM_CACHE)
    parser.add_argument("--eccv-cache", type=Path, default=DEFAULT_ECCV_CACHE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--mm-archive", type=Path, default=DEFAULT_MM_ARCHIVE)
    parser.add_argument("--eccv-archive", type=Path, default=DEFAULT_ECCV_ARCHIVE)
    parser.add_argument(
        "--synthetic-manifest", type=Path, default=DEFAULT_SYNTHETIC_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--local-notes", type=Path, default=DEFAULT_LOCAL_NOTES)
    return parser


def main() -> int:
    args = _parser().parse_args()
    index = HistoricalIdentityIndex.from_files(
        mm_cache_path=args.mm_cache,
        eccv_cache_path=args.eccv_cache,
        split_path=args.split,
    )

    def historical(split: str):
        return iter_historical_pair_records(
            split_manifest=args.split,
            split=split,
            mm_archive=args.mm_archive,
            eccv_archive=args.eccv_archive,
        )

    result = freeze_local_q1(
        identity_index=index,
        validation_records=historical("val"),
        historical_training_records=historical("train"),
        synthetic_training_records=iter_synthetic_pair_records(args.synthetic_manifest),
        historical_archive_locks={
            "mm_augmented": _attest_archive(args.mm_archive, MM_CANONICAL_BINDING),
            "eccv_1113data": _attest_archive(args.eccv_archive, ECCV_CANONICAL_BINDING),
        },
        synthetic_manifest_lock=_artifact_lock(args.synthetic_manifest),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    notes = [
        "# LOCAL-Q1 local path notes",
        "",
        "This file is local-only and is not part of the portable receipt.",
        "The freeze parsed only authorized train/validation pair metadata and ",
        "the canonical-new JSONL manifest. Historical archive containers were ",
        "byte-hashed and their CSV metadata was parsed; no mask archive member ",
        "was opened or decoded. No model module or sealed-real artifact was read.",
        "",
        "- MM archive: `{}`".format(args.mm_archive.resolve()),
        "- ECCV archive: `{}`".format(args.eccv_archive.resolve()),
        "- MM fingerprint cache: `{}`".format(args.mm_cache.resolve()),
        "- ECCV fingerprint cache: `{}`".format(args.eccv_cache.resolve()),
        "- Historical split: `{}`".format(args.split.resolve()),
        "- Canonical-new manifest: `{}`".format(args.synthetic_manifest.resolve()),
        "- Portable receipt: `{}`".format(args.output.resolve()),
        "",
    ]
    args.local_notes.write_text("\n".join(notes), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_TRAIN_INPUT_COUNTS",
    "EXPECTED_VALIDATION_COMPONENT_COUNTS",
    "LocalQ1FreezeError",
    "LocalQ1FreezeResult",
    "LocalQ1RouteAFrontierSpool",
    "ROUTE_A_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SELECTION_SEED",
    "SYNTHETIC_ARCHIVE_BINDING",
    "SYNTHETIC_DATASET_ID",
    "SYNTHETIC_MANIFEST_SHA256",
    "TRAIN_TARGET_PER_DATASET_LABEL",
    "VALIDATION_CAPS",
    "assert_portable_local_q1_receipt",
    "freeze_local_q1",
]
