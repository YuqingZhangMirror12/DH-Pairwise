"""Source-neutral, leakage-safe Pairwise 30k population protocol.

The module is deliberately metadata-only.  It assigns immutable lineage units
before it selects pairs, never duplicates a pair by reversing its endpoints,
and never borrows examples across splits.  Image representation (filled mask
or contour band) is intentionally absent so both representations can reuse the
same selected pair IDs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union


PROTOCOL_SCHEMA_VERSION = "dunhuang-pairwise-30k-protocol/0.1"
DEFAULT_SEED = "dunhuang-pairwise-30k-v1"
SPLITS = ("train", "val", "test")


class Pairwise30kProtocolError(ValueError):
    """The candidate pool violates the split or identity contract."""


class CandidateShortfallError(Pairwise30kProtocolError):
    """One or more explicitly requested source/split/label cells are short."""

    def __init__(self, shortfalls: Tuple[Mapping[str, object], ...]) -> None:
        self.shortfalls = shortfalls
        details = "; ".join(
            "source={source} split={split} label={label} requested={requested} "
            "available={available}".format(**item)
            for item in shortfalls
        )
        super().__init__("candidate shortfall: " + details)


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field_name + " must be a non-empty string")
    return value


@dataclass(frozen=True)
class PairCandidate:
    """One unordered classification candidate with endpoint lineage.

    Fragment tokens and split-unit IDs must be globally namespaced stable IDs.
    A/B order is preserved so the explicit ``direction_b_wrt_a`` field remains
    valid; unordered identity is nevertheless canonicalized for duplicate and
    conflict checks.
    """

    pair_id: str
    source_id: str
    fragment_a_parent_group_id: str
    fragment_b_parent_group_id: str
    fragment_a_split_unit_id: str
    fragment_b_split_unit_id: str
    fragment_a_token: str
    fragment_b_token: str
    label: bool
    label_origin: str
    direction_b_wrt_a: Optional[str] = None
    negative_origin: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    main_training_eligible: bool = True
    selection_exclusion_reason: Optional[str] = None

    def __post_init__(self) -> None:
        for name in (
            "pair_id",
            "source_id",
            "fragment_a_parent_group_id",
            "fragment_b_parent_group_id",
            "fragment_a_split_unit_id",
            "fragment_b_split_unit_id",
            "fragment_a_token",
            "fragment_b_token",
            "label_origin",
        ):
            _required_text(getattr(self, name), name)
        if type(self.label) is not bool:  # noqa: E721
            raise TypeError("label must be a built-in bool")
        if type(self.main_training_eligible) is not bool:  # noqa: E721
            raise TypeError("main_training_eligible must be a built-in bool")
        if self.main_training_eligible:
            if self.selection_exclusion_reason is not None:
                raise Pairwise30kProtocolError(
                    "eligible candidate cannot have selection_exclusion_reason"
                )
        else:
            _required_text(
                self.selection_exclusion_reason, "selection_exclusion_reason"
            )
        if self.direction_b_wrt_a is not None and self.direction_b_wrt_a not in {
            "left",
            "right",
            "above",
            "below",
        }:
            raise ValueError("direction_b_wrt_a must be left/right/above/below or None")
        if not self.label and self.direction_b_wrt_a is not None:
            raise Pairwise30kProtocolError(
                "negative candidate cannot have direction_b_wrt_a"
            )
        if self.label and self.negative_origin is not None:
            raise Pairwise30kProtocolError(
                "positive candidate cannot have negative_origin"
            )
        if self.negative_origin is not None:
            _required_text(self.negative_origin, "negative_origin")
        if self.fragment_a_token == self.fragment_b_token:
            raise Pairwise30kProtocolError("self pair is not allowed")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        copied_metadata = dict(self.metadata)
        try:
            json.dumps(copied_metadata, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be portable JSON") from error
        object.__setattr__(self, "metadata", MappingProxyType(copied_metadata))

    @property
    def canonical_pair_key(self) -> Tuple[str, str]:
        return tuple(sorted((self.fragment_a_token, self.fragment_b_token)))

    def to_dict(self) -> Dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "source_id": self.source_id,
            "fragment_a_parent_group_id": self.fragment_a_parent_group_id,
            "fragment_b_parent_group_id": self.fragment_b_parent_group_id,
            "fragment_a_split_unit_id": self.fragment_a_split_unit_id,
            "fragment_b_split_unit_id": self.fragment_b_split_unit_id,
            "fragment_a_token": self.fragment_a_token,
            "fragment_b_token": self.fragment_b_token,
            "label": self.label,
            "label_origin": self.label_origin,
            "direction_b_wrt_a": self.direction_b_wrt_a,
            "negative_origin": self.negative_origin,
            "metadata": dict(self.metadata),
            "main_training_eligible": self.main_training_eligible,
            "selection_exclusion_reason": self.selection_exclusion_reason,
        }


@dataclass(frozen=True)
class LabelQuota:
    positive: int
    negative: int

    def __post_init__(self) -> None:
        for name in ("positive", "negative"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:  # noqa: E721
                raise ValueError(name + " quota must be a non-negative int")

    def to_dict(self) -> Dict[str, int]:
        return {"positive": self.positive, "negative": self.negative}


DEFAULT_SPLIT_QUOTAS = MappingProxyType(
    {
        "train": LabelQuota(positive=12_000, negative=12_000),
        "val": LabelQuota(positive=1_500, negative=1_500),
        "test": LabelQuota(positive=1_500, negative=1_500),
    }
)


QuotaLike = Union[LabelQuota, Mapping[str, int]]


def _quota(value: QuotaLike, *, allow_zero_total: bool) -> LabelQuota:
    if isinstance(value, LabelQuota):
        quota = value
    elif isinstance(value, Mapping) and set(value) == {"positive", "negative"}:
        quota = LabelQuota(
            positive=value["positive"],
            negative=value["negative"],
        )
    else:
        raise ValueError("quota must define positive and negative exactly")
    if not allow_zero_total and quota.positive + quota.negative == 0:
        raise ValueError("split quota cannot be empty")
    return quota


def _split_quotas(
    value: Optional[Mapping[str, QuotaLike]],
) -> Mapping[str, LabelQuota]:
    if value is None:
        return DEFAULT_SPLIT_QUOTAS
    if not isinstance(value, Mapping) or set(value) != set(SPLITS):
        raise ValueError("split_quotas must define train, val, and test exactly")
    normalized = {
        split: _quota(value[split], allow_zero_total=False) for split in SPLITS
    }
    for split, quota in normalized.items():
        if quota.positive != quota.negative:
            raise ValueError(split + " quota must be 1:1 positive/negative")
    totals = [
        normalized[split].positive + normalized[split].negative for split in SPLITS
    ]
    if totals[0] != 8 * totals[1] or totals[1] != totals[2]:
        raise ValueError("split_quotas must have an exact 8:1:1 ratio")
    return MappingProxyType(normalized)


def stable_split_for_unit(
    split_unit_id: str,
    *,
    seed: str = DEFAULT_SEED,
) -> str:
    """Assign one globally namespaced lineage unit to an 80/10/10 split."""

    _required_text(split_unit_id, "split_unit_id")
    _required_text(seed, "seed")
    payload = (seed + "\0split-unit\0" + split_unit_id).encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    fraction = bucket / float(1 << 64)
    if fraction < 0.8:
        return "train"
    if fraction < 0.9:
        return "val"
    return "test"


def _rank(
    seed: str, split: str, source: str, label: bool, candidate: PairCandidate
) -> int:
    payload = "\0".join(
        (
            seed,
            "select",
            split,
            source,
            "positive" if label else "negative",
            candidate.pair_id,
            candidate.canonical_pair_key[0],
            candidate.canonical_pair_key[1],
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


@dataclass(frozen=True)
class SelectedPair:
    split: str
    candidate: PairCandidate

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError("selected split must be train, val, or test")

    def to_dict(self) -> Dict[str, object]:
        row = self.candidate.to_dict()
        row["split"] = self.split
        return row


@dataclass(frozen=True)
class ProtocolSelection:
    seed: str
    split_quotas: Mapping[str, LabelQuota]
    unit_assignments: Mapping[str, str]
    rows: Tuple[SelectedPair, ...]
    excluded_before_selection: Tuple[PairCandidate, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "seed": self.seed,
            "split_quotas": {
                split: self.split_quotas[split].to_dict() for split in SPLITS
            },
            "unit_assignments": dict(sorted(self.unit_assignments.items())),
            "rows": [row.to_dict() for row in self.rows],
        }

    def iter_jsonl_rows(self) -> Iterator[Dict[str, object]]:
        for row in self.rows:
            yield row.to_dict()

    def to_jsonl(self) -> str:
        return "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in self.iter_jsonl_rows()
        )


def _validate_pool(candidates: Tuple[PairCandidate, ...]) -> Tuple[str, ...]:
    if not candidates:
        raise Pairwise30kProtocolError("candidate pool is empty")
    pair_ids: Dict[str, PairCandidate] = {}
    unordered: Dict[Tuple[str, str], PairCandidate] = {}
    fragment_lineage: Dict[str, Tuple[str, str]] = {}
    parent_units: Dict[str, str] = {}
    sources = set()
    for candidate in candidates:
        if not isinstance(candidate, PairCandidate):
            raise TypeError("candidates must contain PairCandidate values")
        sources.add(candidate.source_id)
        if candidate.pair_id in pair_ids:
            raise Pairwise30kProtocolError("duplicate pair_id: " + candidate.pair_id)
        pair_ids[candidate.pair_id] = candidate
        previous = unordered.get(candidate.canonical_pair_key)
        if previous is not None:
            if previous.label is not candidate.label:
                raise Pairwise30kProtocolError(
                    "label conflict for unordered pair "
                    + repr(candidate.canonical_pair_key)
                )
            raise Pairwise30kProtocolError(
                "duplicate unordered pair (including A/B reversal): "
                + repr(candidate.canonical_pair_key)
            )
        unordered[candidate.canonical_pair_key] = candidate
        endpoints = (
            (
                candidate.fragment_a_token,
                candidate.fragment_a_parent_group_id,
                candidate.fragment_a_split_unit_id,
            ),
            (
                candidate.fragment_b_token,
                candidate.fragment_b_parent_group_id,
                candidate.fragment_b_split_unit_id,
            ),
        )
        for token, parent, unit in endpoints:
            lineage = (parent, unit)
            if token in fragment_lineage and fragment_lineage[token] != lineage:
                raise Pairwise30kProtocolError(
                    "fragment has inconsistent parent/split-unit provenance: " + token
                )
            fragment_lineage[token] = lineage
            if parent in parent_units and parent_units[parent] != unit:
                raise Pairwise30kProtocolError(
                    "parent group spans multiple split units: " + parent
                )
            parent_units[parent] = unit
    return tuple(sorted(sources))


def _source_quotas(
    sources: Tuple[str, ...],
    totals: Mapping[str, LabelQuota],
    value: Optional[Mapping[str, Mapping[str, QuotaLike]]],
) -> Mapping[str, Mapping[str, LabelQuota]]:
    if value is None:
        if len(sources) != 1:
            raise Pairwise30kProtocolError(
                "multiple sources require explicit per-source/per-split quotas"
            )
        return MappingProxyType({sources[0]: MappingProxyType(dict(totals))})
    if not isinstance(value, Mapping) or set(value) != set(sources):
        raise ValueError("source_quotas must define every candidate source exactly")
    normalized: Dict[str, Mapping[str, LabelQuota]] = {}
    for source in sources:
        source_value = value[source]
        if not isinstance(source_value, Mapping) or set(source_value) != set(SPLITS):
            raise ValueError(
                "source quota must define train, val, and test exactly: " + source
            )
        normalized[source] = MappingProxyType(
            {
                split: _quota(source_value[split], allow_zero_total=True)
                for split in SPLITS
            }
        )
    for split in SPLITS:
        for label_name in ("positive", "negative"):
            observed = sum(
                getattr(normalized[source][split], label_name) for source in sources
            )
            expected = getattr(totals[split], label_name)
            if observed != expected:
                raise ValueError(
                    "source quotas must sum to split quota: split={} label={} "
                    "expected={} observed={}".format(
                        split, label_name, expected, observed
                    )
                )
    return MappingProxyType(normalized)


def build_pairwise_30k_protocol(
    candidates: Iterable[PairCandidate],
    *,
    seed: str = DEFAULT_SEED,
    split_quotas: Optional[Mapping[str, QuotaLike]] = None,
    source_quotas: Optional[Mapping[str, Mapping[str, QuotaLike]]] = None,
    unit_assignments: Optional[Mapping[str, str]] = None,
) -> ProtocolSelection:
    """Split lineage units, then deterministically fill exact balanced quotas.

    Cross-parent negatives are supported when both endpoint split units resolve
    to the same split.  A cross-split candidate is rejected rather than copied,
    reassigned, or silently borrowed by either split.
    """

    _required_text(seed, "seed")
    pool = tuple(candidates)
    _validate_pool(pool)
    eligible_pool = tuple(
        candidate for candidate in pool if candidate.main_training_eligible
    )
    if not eligible_pool:
        raise Pairwise30kProtocolError("eligible candidate pool is empty")
    sources = tuple(sorted({candidate.source_id for candidate in eligible_pool}))
    quotas = _split_quotas(split_quotas)
    per_source = _source_quotas(sources, quotas, source_quotas)
    units = {candidate.fragment_a_split_unit_id for candidate in eligible_pool} | {
        candidate.fragment_b_split_unit_id for candidate in eligible_pool
    }
    if unit_assignments is None:
        assignments = {
            unit: stable_split_for_unit(unit, seed=seed) for unit in sorted(units)
        }
    else:
        if not isinstance(unit_assignments, Mapping):
            raise TypeError("unit_assignments must be a mapping")
        if set(unit_assignments) != units:
            missing = sorted(units - set(unit_assignments))
            extra = sorted(set(unit_assignments) - units)
            raise Pairwise30kProtocolError(
                "unit_assignments must cover referenced units exactly; "
                "missing={} extra={}".format(missing, extra)
            )
        assignments = dict(unit_assignments)
        invalid = {
            unit: split for unit, split in assignments.items() if split not in SPLITS
        }
        if invalid:
            raise Pairwise30kProtocolError(
                "invalid unit split assignments: " + repr(invalid)
            )

    cells: Dict[Tuple[str, str, bool], list] = {}
    for candidate in eligible_pool:
        split_a = assignments[candidate.fragment_a_split_unit_id]
        split_b = assignments[candidate.fragment_b_split_unit_id]
        if split_a != split_b:
            raise Pairwise30kProtocolError(
                "candidate endpoints resolve to different splits: pair_id={} "
                "a={} b={}".format(candidate.pair_id, split_a, split_b)
            )
        cells.setdefault((candidate.source_id, split_a, candidate.label), []).append(
            candidate
        )

    shortfalls = []
    for source in sources:
        for split in SPLITS:
            for label, label_name in ((True, "positive"), (False, "negative")):
                requested = getattr(per_source[source][split], label_name)
                available = len(cells.get((source, split, label), ()))
                if available < requested:
                    shortfalls.append(
                        {
                            "source": source,
                            "split": split,
                            "label": label_name,
                            "requested": requested,
                            "available": available,
                        }
                    )
    if shortfalls:
        raise CandidateShortfallError(tuple(shortfalls))

    selected = []
    for split in SPLITS:
        for source in sources:
            for label, label_name in ((True, "positive"), (False, "negative")):
                requested = getattr(per_source[source][split], label_name)
                ranked = sorted(
                    cells.get((source, split, label), ()),
                    key=lambda candidate: (
                        _rank(seed, split, source, label, candidate),
                        candidate.pair_id,
                    ),
                )
                selected.extend(
                    SelectedPair(split=split, candidate=candidate)
                    for candidate in ranked[:requested]
                )

    selected.sort(
        key=lambda row: (
            SPLITS.index(row.split),
            _rank(
                seed,
                row.split,
                row.candidate.source_id,
                row.candidate.label,
                row.candidate,
            ),
            row.candidate.pair_id,
        )
    )
    return ProtocolSelection(
        seed=seed,
        split_quotas=MappingProxyType(dict(quotas)),
        unit_assignments=MappingProxyType(dict(assignments)),
        rows=tuple(selected),
        excluded_before_selection=tuple(
            candidate for candidate in pool if not candidate.main_training_eligible
        ),
    )


__all__ = [
    "CandidateShortfallError",
    "DEFAULT_SEED",
    "DEFAULT_SPLIT_QUOTAS",
    "LabelQuota",
    "PROTOCOL_SCHEMA_VERSION",
    "PairCandidate",
    "Pairwise30kProtocolError",
    "ProtocolSelection",
    "SPLITS",
    "SelectedPair",
    "build_pairwise_30k_protocol",
    "stable_split_for_unit",
]
