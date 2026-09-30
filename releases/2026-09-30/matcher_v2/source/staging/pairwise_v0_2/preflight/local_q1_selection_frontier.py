"""Disk-backed complete deterministic frontier for LOCAL-Q1 Route A.

Only lightweight metadata is externally sorted in SQLite.  Geometry is built
on demand while traversing the exact deterministic prefix required to reach
the frozen train quotas and per-component validation caps.  There is no fixed
reserve or arbitrary prefix cutoff: a component cursor continues until an
eligible record is found or its complete metadata frontier is exhausted.

The SQLite file is local-only and can contain raw record identities.  Nothing
from it is emitted directly.  The portable selection proof contains aggregate
counts plus cryptographic commitments, and the refreeze must replay it against
an externally locked eligibility authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.preflight.local_q1_geometry_eligibility_authority import (
    LocalQ1EligibilityDecision,
    LocalQ1PairGeometryInput,
)


LOCAL_Q1_SELECTION_FRONTIER_VERSION = (
    "dunhuang-local-q1-complete-lazy-selection-frontier/0.1"
)


class LocalQ1SelectionFrontierError(RuntimeError):
    """The full frontier cannot prove exact deterministic refreeze selection."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _commit_order(tokens: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(str(token).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _commit_set(tokens: Iterable[str]) -> str:
    return _commit_order(sorted(tokens))


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LocalQ1SelectionFrontierError(
                "record provenance contains NaN/Inf"
            )
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise LocalQ1SelectionFrontierError(
        "record provenance is not canonical JSON-compatible"
    )


def _reference_payload(reference: MaskMemberRef) -> Mapping[str, Any]:
    return {
        "binding": {
            "logical_id": reference.binding.logical_id,
            "archive_format": reference.binding.archive_format,
            "sha256": reference.binding.sha256,
        },
        "archive_member": reference.archive_member,
        "fragment_id": reference.fragment_id,
        "dataset_id": reference.dataset_id,
        "canonical_group_id": reference.canonical_group_id,
        "component_id": reference.component_id,
        "split": reference.split,
        "threshold_rule": reference.threshold_rule,
        "content_sha256": reference.content_sha256,
    }


def _record_payload(record: TrainingPairRecord) -> Mapping[str, Any]:
    return {
        "fragment_a": _reference_payload(record.fragment_a),
        "fragment_b": _reference_payload(record.fragment_b),
        "label": record.label,
        "direction_b_wrt_a": record.direction_b_wrt_a,
        "dataset_id": record.dataset_id,
        "canonical_group_id": record.canonical_group_id,
        "component_id": record.component_id,
        "split": record.split,
        "canonical_pair_key": list(record.canonical_pair_key),
        "label_origin": record.label_origin,
        "static_hard_negative_score": record.static_hard_negative_score,
        "provenance": _json_value(record.provenance),
        "schema_version": record.schema_version,
    }


def _reference_from_payload(value: Any) -> MaskMemberRef:
    if not isinstance(value, Mapping):
        raise LocalQ1SelectionFrontierError("frontier reference payload is invalid")
    binding_value = value.get("binding")
    if not isinstance(binding_value, Mapping):
        raise LocalQ1SelectionFrontierError("frontier binding payload is invalid")
    binding = ArchiveBinding(
        logical_id=binding_value["logical_id"],
        archive_format=binding_value["archive_format"],
        sha256=binding_value["sha256"],
    )
    return MaskMemberRef(
        binding=binding,
        archive_member=value["archive_member"],
        fragment_id=value["fragment_id"],
        dataset_id=value["dataset_id"],
        canonical_group_id=value["canonical_group_id"],
        component_id=value["component_id"],
        split=value["split"],
        threshold_rule=value["threshold_rule"],
        content_sha256=value.get("content_sha256"),
    )


def _record_from_bytes(payload: bytes) -> TrainingPairRecord:
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
        raise LocalQ1SelectionFrontierError(
            "frontier record payload is invalid JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise LocalQ1SelectionFrontierError("frontier record payload is invalid")
    if payload != _canonical_json(value):
        raise LocalQ1SelectionFrontierError(
            "frontier record payload is not canonical JSON"
        )
    try:
        return TrainingPairRecord(
            fragment_a=_reference_from_payload(value["fragment_a"]),
            fragment_b=_reference_from_payload(value["fragment_b"]),
            label=value["label"],
            direction_b_wrt_a=value["direction_b_wrt_a"],
            dataset_id=value["dataset_id"],
            canonical_group_id=value["canonical_group_id"],
            component_id=value["component_id"],
            split=value["split"],
            canonical_pair_key=tuple(value["canonical_pair_key"]),
            label_origin=value["label_origin"],
            static_hard_negative_score=value["static_hard_negative_score"],
            provenance=value["provenance"],
            schema_version=value["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalQ1SelectionFrontierError(
            "frontier record reconstruction failed"
        ) from exc


@dataclass(frozen=True)
class LocalQ1FrontierCandidate:
    record: TrainingPairRecord
    dataset_name: str
    component_token: str
    pair_identity_token: str
    record_token: str
    record_rank: str
    component_rank: str
    source_ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.record, TrainingPairRecord):
            raise TypeError("frontier candidate requires TrainingPairRecord")
        for name in (
            "component_token",
            "pair_identity_token",
            "record_token",
            "record_rank",
            "component_rank",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("{} must be SHA-256 hex".format(name))
        if not isinstance(self.dataset_name, str) or not self.dataset_name:
            raise ValueError("dataset_name is required")
        if (
            isinstance(self.source_ordinal, bool)
            or not isinstance(self.source_ordinal, int)
            or self.source_ordinal < 0
        ):
            raise ValueError("source_ordinal must be a non-negative integer")

    def geometry_input(self) -> LocalQ1PairGeometryInput:
        return LocalQ1PairGeometryInput(
            fragment_a=self.record.fragment_a,
            fragment_b=self.record.fragment_b,
            pair_id=self.record.pair_id,
            pair_identity_sha256=self.pair_identity_token,
        )


@dataclass(frozen=True)
class LocalQ1FrontierSelectionResult:
    training_records: Tuple[TrainingPairRecord, ...]
    validation_records: Tuple[TrainingPairRecord, ...]
    training_geometry_inputs: Tuple[LocalQ1PairGeometryInput, ...]
    validation_geometry_inputs: Tuple[LocalQ1PairGeometryInput, ...]
    decisions: Tuple[LocalQ1EligibilityDecision, ...]
    selection_proof: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.training_records or not self.validation_records:
            raise LocalQ1SelectionFrontierError("frontier result cannot be empty")
        if (
            len(self.training_geometry_inputs) != len(self.training_records)
            or len(self.validation_geometry_inputs) != len(self.validation_records)
            or any(
                not isinstance(item, LocalQ1PairGeometryInput)
                for item in self.training_geometry_inputs
                + self.validation_geometry_inputs
            )
        ):
            raise LocalQ1SelectionFrontierError(
                "frontier selected geometry-input replay is incomplete"
            )
        object.__setattr__(self, "selection_proof", _deep_freeze(self.selection_proof))


class LocalQ1SelectionFrontier:
    """Complete SQLite metadata frontier with lazy, prefix-complete traversal."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists() or self.path.is_symlink():
            raise LocalQ1SelectionFrontierError(
                "frontier database output already exists"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA temp_store=FILE")
        self._connection.execute(
            """
            CREATE TABLE frontier (
                split TEXT NOT NULL CHECK(split IN ('train', 'val')),
                dataset_name TEXT NOT NULL,
                label INTEGER NOT NULL CHECK(label IN (0, 1)),
                component_token TEXT NOT NULL,
                pair_identity_token TEXT NOT NULL,
                record_token TEXT NOT NULL,
                record_rank TEXT NOT NULL,
                component_rank TEXT NOT NULL,
                source_ordinal INTEGER NOT NULL,
                record_payload BLOB NOT NULL,
                UNIQUE(split, dataset_name, label, component_token,
                       pair_identity_token, source_ordinal)
            )
            """
        )
        self._finalized = False
        self._closed = False
        self._count = 0

    def add(self, candidate: LocalQ1FrontierCandidate) -> None:
        if self._finalized or self._closed:
            raise LocalQ1SelectionFrontierError("frontier is not writable")
        if not isinstance(candidate, LocalQ1FrontierCandidate):
            raise TypeError("candidate must be LocalQ1FrontierCandidate")
        payload = _canonical_json(_record_payload(candidate.record))
        self._connection.execute(
            """
            INSERT INTO frontier (
                split, dataset_name, label, component_token,
                pair_identity_token, record_token, record_rank,
                component_rank, source_ordinal, record_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.record.split,
                candidate.dataset_name,
                int(candidate.record.label),
                candidate.component_token,
                candidate.pair_identity_token,
                candidate.record_token,
                candidate.record_rank,
                candidate.component_rank,
                candidate.source_ordinal,
                payload,
            ),
        )
        self._count += 1
        if self._count % 10_000 == 0:
            self._connection.commit()

    def finalize(self) -> None:
        if self._closed:
            raise LocalQ1SelectionFrontierError("frontier is closed")
        if self._finalized:
            return
        if self._count <= 0:
            raise LocalQ1SelectionFrontierError("frontier is empty")
        self._connection.commit()
        self._connection.execute(
            """
            CREATE INDEX frontier_train_order ON frontier (
                split, dataset_name, label, component_rank,
                component_token, record_rank, source_ordinal
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX frontier_validation_order ON frontier (
                split, dataset_name, component_rank,
                component_token, record_rank, source_ordinal
            )
            """
        )
        self._connection.commit()
        self._finalized = True

    @staticmethod
    def _row_candidate(row: Sequence[Any]) -> LocalQ1FrontierCandidate:
        (
            dataset_name,
            component_token,
            pair_token,
            record_token,
            record_rank,
            component_rank,
            source_ordinal,
            record_payload,
        ) = row
        return LocalQ1FrontierCandidate(
            record=_record_from_bytes(record_payload),
            dataset_name=dataset_name,
            component_token=component_token,
            pair_identity_token=pair_token,
            record_token=record_token,
            record_rank=record_rank,
            component_rank=component_rank,
            source_ordinal=source_ordinal,
        )

    def _components(self, *, split: str, dataset: str, label: Optional[bool]) -> Tuple[str, ...]:
        params = [split, dataset]
        where = "split = ? AND dataset_name = ?"
        if label is not None:
            where += " AND label = ?"
            params.append(int(label))
        rows = self._connection.execute(
            "SELECT component_token, MIN(component_rank) AS rank "
            "FROM frontier WHERE {} GROUP BY component_token "
            "ORDER BY rank, component_token".format(where),
            tuple(params),
        )
        return tuple(row[0] for row in rows)

    def _component_cursor(
        self,
        *,
        split: str,
        dataset: str,
        component: str,
        label: Optional[bool],
    ):
        params = [split, dataset, component]
        where = "split = ? AND dataset_name = ? AND component_token = ?"
        if label is not None:
            where += " AND label = ?"
            params.append(int(label))
        return self._connection.execute(
            "SELECT dataset_name, component_token, pair_identity_token, "
            "record_token, record_rank, component_rank, source_ordinal, "
            "record_payload FROM frontier WHERE {} "
            "ORDER BY record_rank, source_ordinal".format(where),
            tuple(params),
        )

    def _old_train_selection(
        self, *, dataset: str, label: bool, target: int
    ) -> Tuple[LocalQ1FrontierCandidate, ...]:
        components = self._components(split="train", dataset=dataset, label=label)
        cursors = {
            component: self._component_cursor(
                split="train", dataset=dataset, component=component, label=label
            )
            for component in components
        }
        selected = []
        while len(selected) < target:
            added = 0
            for component in components:
                row = cursors[component].fetchone()
                if row is not None:
                    selected.append(self._row_candidate(row))
                    added += 1
                    if len(selected) == target:
                        break
            if not added:
                break
        if len(selected) != target:
            raise LocalQ1SelectionFrontierError(
                "raw training frontier cannot reach frozen quota"
            )
        return tuple(selected)

    def _new_train_selection(
        self,
        *,
        dataset: str,
        label: bool,
        target: int,
        decide: Callable[[LocalQ1PairGeometryInput], LocalQ1EligibilityDecision],
        decision_order: list,
        decision_by_token: Dict[str, LocalQ1EligibilityDecision],
    ) -> Tuple[Tuple[LocalQ1FrontierCandidate, ...], Mapping[str, Any]]:
        components = self._components(split="train", dataset=dataset, label=label)
        if not components:
            raise LocalQ1SelectionFrontierError("training frontier stratum is empty")
        cursors = {
            component: self._component_cursor(
                split="train", dataset=dataset, component=component, label=label
            )
            for component in components
        }
        selected = []
        decided_rows = 0
        ineligible_rows = 0
        exhausted = set()
        round_index = 0
        final_cutoff = None
        trace_rows = []
        while len(selected) < target:
            added = 0
            for component_ordinal, component in enumerate(components):
                while True:
                    row = cursors[component].fetchone()
                    if row is None:
                        exhausted.add(component)
                        break
                    candidate = self._row_candidate(row)
                    decision = decision_by_token.get(candidate.pair_identity_token)
                    if decision is None:
                        decision = decide(candidate.geometry_input())
                        if not isinstance(decision, LocalQ1EligibilityDecision) or (
                            decision.pair_identity_sha256
                            != candidate.pair_identity_token
                        ):
                            raise LocalQ1SelectionFrontierError(
                                "geometry decision does not match frontier pair"
                            )
                        decision_by_token[candidate.pair_identity_token] = decision
                        decision_order.append(candidate.pair_identity_token)
                    decided_rows += 1
                    trace_rows.append(
                        [
                            candidate.pair_identity_token,
                            decision.assessment.eligible,
                            round_index,
                            component_ordinal,
                        ]
                    )
                    if decision.assessment.eligible:
                        selected.append(candidate)
                        added += 1
                        if len(selected) == target:
                            final_cutoff = component_ordinal
                        break
                    ineligible_rows += 1
                if len(selected) == target:
                    break
            if len(selected) == target:
                break
            if not added and len(exhausted) == len(components):
                raise LocalQ1SelectionFrontierError(
                    "geometry-qualified training quota is unreachable"
                )
            round_index += 1
        if final_cutoff is None:
            raise LocalQ1SelectionFrontierError("training frontier cutoff was not proven")
        proof = {
            "target": target,
            "selected_count": len(selected),
            "component_count": len(components),
            "rounds_started": round_index + 1,
            "final_round_component_cutoff_ordinal": final_cutoff,
            "decided_frontier_row_count": decided_rows,
            "ineligible_predecessor_count": ineligible_rows,
            "exhausted_component_count": len(exhausted),
            "unexplored_higher_priority_record_count": 0,
            "arbitrary_fixed_reserve_cutoff_used": False,
            "frontier_trace_sha256": _sha256(_canonical_json(trace_rows)),
            "selected_record_order_sha256": _commit_order(
                candidate.record_token for candidate in selected
            ),
            "selected_record_set_sha256": _commit_set(
                candidate.record_token for candidate in selected
            ),
        }
        return tuple(selected), proof

    def _old_validation_selection(
        self, *, dataset: str, cap: int
    ) -> Tuple[LocalQ1FrontierCandidate, ...]:
        selected = []
        for component in self._components(split="val", dataset=dataset, label=None):
            cursor = self._component_cursor(
                split="val", dataset=dataset, component=component, label=None
            )
            rows = [cursor.fetchone() for _ in range(cap)]
            if any(row is None for row in rows):
                raise LocalQ1SelectionFrontierError(
                    "raw validation component cannot reach frozen cap"
                )
            selected.extend(self._row_candidate(row) for row in rows if row is not None)
        return tuple(selected)

    def _new_validation_selection(
        self,
        *,
        dataset: str,
        cap: int,
        decide: Callable[[LocalQ1PairGeometryInput], LocalQ1EligibilityDecision],
        decision_order: list,
        decision_by_token: Dict[str, LocalQ1EligibilityDecision],
    ) -> Tuple[Tuple[LocalQ1FrontierCandidate, ...], Mapping[str, Any]]:
        components = self._components(split="val", dataset=dataset, label=None)
        if not components:
            raise LocalQ1SelectionFrontierError("validation domain has no components")
        selected = []
        decided_rows = 0
        ineligible_rows = 0
        max_prefix = 0
        min_prefix = None
        trace_digest = hashlib.sha256()
        for component_ordinal, component in enumerate(components):
            cursor = self._component_cursor(
                split="val", dataset=dataset, component=component, label=None
            )
            component_selected = 0
            component_decided = 0
            while component_selected < cap:
                row = cursor.fetchone()
                if row is None:
                    raise LocalQ1SelectionFrontierError(
                        "geometry-qualified validation component cap is unreachable"
                    )
                candidate = self._row_candidate(row)
                decision = decision_by_token.get(candidate.pair_identity_token)
                if decision is None:
                    decision = decide(candidate.geometry_input())
                    if not isinstance(decision, LocalQ1EligibilityDecision) or (
                        decision.pair_identity_sha256 != candidate.pair_identity_token
                    ):
                        raise LocalQ1SelectionFrontierError(
                            "geometry decision does not match validation frontier"
                        )
                    decision_by_token[candidate.pair_identity_token] = decision
                    decision_order.append(candidate.pair_identity_token)
                component_decided += 1
                decided_rows += 1
                trace_digest.update(
                    _canonical_json(
                        [
                            candidate.pair_identity_token,
                            decision.assessment.eligible,
                            component_ordinal,
                            component_decided - 1,
                        ]
                    )
                )
                trace_digest.update(b"\n")
                if decision.assessment.eligible:
                    selected.append(candidate)
                    component_selected += 1
                else:
                    ineligible_rows += 1
            max_prefix = max(max_prefix, component_decided)
            min_prefix = (
                component_decided
                if min_prefix is None
                else min(min_prefix, component_decided)
            )
        proof = {
            "cap_per_component": cap,
            "component_count": len(components),
            "selected_count": len(selected),
            "decided_frontier_row_count": decided_rows,
            "ineligible_predecessor_count": ineligible_rows,
            "minimum_decided_prefix_per_component": min_prefix,
            "maximum_decided_prefix_per_component": max_prefix,
            "component_cap_shortfall_count": 0,
            "unexplored_higher_priority_record_count": 0,
            "arbitrary_fixed_reserve_cutoff_used": False,
            "frontier_trace_sha256": trace_digest.hexdigest(),
            "selected_record_set_sha256": _commit_set(
                candidate.record_token for candidate in selected
            ),
        }
        return tuple(selected), proof

    @staticmethod
    def _diff(
        old: Sequence[LocalQ1FrontierCandidate],
        new: Sequence[LocalQ1FrontierCandidate],
    ) -> Mapping[str, int]:
        old_tokens = {candidate.record_token for candidate in old}
        new_tokens = {candidate.record_token for candidate in new}
        if len(old_tokens) != len(old) or len(new_tokens) != len(new):
            raise LocalQ1SelectionFrontierError(
                "selected population contains duplicate pair identities"
            )
        return {
            "old_count": len(old),
            "new_count": len(new),
            "unchanged_count": len(old_tokens & new_tokens),
            "removed_count": len(old_tokens - new_tokens),
            "replacement_count": len(new_tokens - old_tokens),
        }

    def select(
        self,
        *,
        decide: Callable[[LocalQ1PairGeometryInput], LocalQ1EligibilityDecision],
        train_datasets: Sequence[str],
        train_target_per_dataset_label: int,
        validation_caps: Mapping[str, int],
    ) -> LocalQ1FrontierSelectionResult:
        """Traverse complete frontiers lazily and emit an identity-free proof."""

        self.finalize()
        if not callable(decide):
            raise TypeError("decide must be callable")
        if (
            isinstance(train_target_per_dataset_label, bool)
            or not isinstance(train_target_per_dataset_label, int)
            or train_target_per_dataset_label <= 0
        ):
            raise ValueError("train target must be a positive integer")
        if not train_datasets or len(set(train_datasets)) != len(train_datasets):
            raise ValueError("train_datasets must be unique and non-empty")
        if not validation_caps or any(
            isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
            for cap in validation_caps.values()
        ):
            raise ValueError("validation caps must be positive integers")

        decision_order = []
        decision_by_token = {}  # type: Dict[str, LocalQ1EligibilityDecision]
        old_train = []
        new_train = []
        train_proof = {}
        train_diff = {}
        for dataset in train_datasets:
            for label in (False, True):
                stratum = "{}|{}".format(
                    dataset, "positive" if label else "negative"
                )
                old = self._old_train_selection(
                    dataset=dataset,
                    label=label,
                    target=train_target_per_dataset_label,
                )
                new, proof = self._new_train_selection(
                    dataset=dataset,
                    label=label,
                    target=train_target_per_dataset_label,
                    decide=decide,
                    decision_order=decision_order,
                    decision_by_token=decision_by_token,
                )
                old_train.extend(old)
                new_train.extend(new)
                train_proof[stratum] = proof
                train_diff[stratum] = self._diff(old, new)

        old_validation = []
        new_validation = []
        validation_proof = {}
        validation_diff = {}
        for dataset in sorted(validation_caps):
            old = self._old_validation_selection(
                dataset=dataset, cap=validation_caps[dataset]
            )
            new, proof = self._new_validation_selection(
                dataset=dataset,
                cap=validation_caps[dataset],
                decide=decide,
                decision_order=decision_order,
                decision_by_token=decision_by_token,
            )
            old_validation.extend(old)
            new_validation.extend(new)
            validation_proof[dataset] = proof
            validation_diff[dataset] = self._diff(old, new)

        # Validation membership is frozen per component but returned as a
        # subsequence of the canonical source stream.
        old_validation.sort(key=lambda candidate: candidate.source_ordinal)
        new_validation.sort(key=lambda candidate: candidate.source_ordinal)
        new_train_records = tuple(candidate.record for candidate in new_train)
        new_validation_records = tuple(candidate.record for candidate in new_validation)
        decision_trace = [
            [token, decision_by_token[token].assessment.eligible]
            for token in decision_order
        ]
        failure_counts = Counter(
            "{}|{}".format(
                decision.assessment.failure_stage or "eligible",
                decision.assessment.failure_reason or "eligible",
            )
            for decision in decision_by_token.values()
        )
        selected_tokens = [candidate.record_token for candidate in new_train] + [
            candidate.record_token for candidate in new_validation
        ]
        selected_pair_identity_tokens = [
            candidate.pair_identity_token for candidate in new_train
        ] + [candidate.pair_identity_token for candidate in new_validation]
        predecessor_train_tokens = [
            candidate.record_token for candidate in old_train
        ]
        predecessor_validation_tokens = [
            candidate.record_token for candidate in old_validation
        ]
        proof = {
            "frontier_version": LOCAL_Q1_SELECTION_FRONTIER_VERSION,
            "frontier_storage": (
                "complete_metadata_sqlite_external_sort_geometry_consumed_lazily"
            ),
            "arbitrary_fixed_reserve_cutoff_used": False,
            "manual_exclusions_used": False,
            "unexplored_higher_priority_record_count": 0,
            "training": train_proof,
            "validation": validation_proof,
            "old_to_new_population_difference": {
                "training_by_dataset_label": train_diff,
                "validation_by_dataset": validation_diff,
                "path_or_row_identifiers_present": False,
            },
            "predecessor_population": {
                "training_count": len(old_train),
                "validation_count": len(old_validation),
                "training_record_order_sha256": _commit_order(
                    predecessor_train_tokens
                ),
                "training_record_set_sha256": _commit_set(
                    predecessor_train_tokens
                ),
                "validation_record_order_sha256": _commit_order(
                    predecessor_validation_tokens
                ),
                "validation_record_set_sha256": _commit_set(
                    predecessor_validation_tokens
                ),
            },
            "decision_count": len(decision_order),
            "decision_order_sha256": _commit_order(decision_order),
            "decision_trace_sha256": _sha256(_canonical_json(decision_trace)),
            "decision_failure_stage_reason_counts": dict(sorted(failure_counts.items())),
            "selected_training_count": len(new_train_records),
            "selected_validation_count": len(new_validation_records),
            "selected_record_order_sha256": _commit_order(selected_tokens),
            "selected_record_set_sha256": _commit_set(selected_tokens),
            "selected_pair_commitment_order_sha256": _commit_order(
                selected_pair_identity_tokens
            ),
            "selected_pair_commitment_set_sha256": _commit_set(
                selected_pair_identity_tokens
            ),
            "selected_failure_counts": {
                "fragment": 0,
                "pair": 0,
                "direction_coverage": 0,
                "resource": 0,
            },
        }
        return LocalQ1FrontierSelectionResult(
            training_records=new_train_records,
            validation_records=new_validation_records,
            training_geometry_inputs=tuple(
                candidate.geometry_input() for candidate in new_train
            ),
            validation_geometry_inputs=tuple(
                candidate.geometry_input() for candidate in new_validation
            ),
            decisions=tuple(decision_by_token[token] for token in decision_order),
            selection_proof=proof,
        )

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "LocalQ1SelectionFrontier":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - last-resort exception cleanup
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "LOCAL_Q1_SELECTION_FRONTIER_VERSION",
    "LocalQ1FrontierCandidate",
    "LocalQ1FrontierSelectionResult",
    "LocalQ1SelectionFrontier",
    "LocalQ1SelectionFrontierError",
]
