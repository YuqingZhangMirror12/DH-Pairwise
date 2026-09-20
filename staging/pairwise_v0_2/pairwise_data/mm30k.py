"""Deterministic canonical-MM 30k/6k population for direct GPU research.

This module reads only the canonical MM train/validation metadata stream.  It
does not read the historical 280 MB aggregate CSV, ECCV, synthetic data, test,
or sealed manuscripts.  Variant ``0`` is the original-mask stratum; variants
``1`` through ``6`` are generated-mask strata, matching the historical MM
training recipe.
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord


MM30K_SELECTION_SCHEMA_VERSION = "dunhuang-pairwise-mm30k-population/0.1"
MM30K_DEFAULT_SEED = "mm30k-260829"
MM30K_STRATA = (
    "original_positive",
    "original_negative",
    "generated_positive",
    "generated_negative",
)
MM30K_TRAIN_QUOTAS = MappingProxyType(
    {
        "original_positive": 10_000,
        "original_negative": 10_000,
        "generated_positive": 5_000,
        "generated_negative": 5_000,
    }
)
MM30K_VALIDATION_QUOTAS = MappingProxyType(
    {
        "original_positive": 2_000,
        "original_negative": 2_000,
        "generated_positive": 1_000,
        "generated_negative": 1_000,
    }
)
_GENERATED_VARIANTS = frozenset(str(value) for value in range(1, 7))


class MM30KSelectionError(RuntimeError):
    """The canonical MM stream cannot satisfy the fixed population recipe."""


def _normalized_quotas(value: Mapping[str, int], name: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(MM30K_STRATA):
        raise ValueError(name + " must define every MM30K stratum exactly")
    normalized = {key: value[key] for key in MM30K_STRATA}
    if any(type(count) is not int or count <= 0 for count in normalized.values()):
        raise ValueError(name + " values must be positive integers")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class MM30KSelectionConfig:
    seed: str = MM30K_DEFAULT_SEED
    train_quotas: Mapping[str, int] = field(
        default_factory=lambda: MM30K_TRAIN_QUOTAS
    )
    validation_quotas: Mapping[str, int] = field(
        default_factory=lambda: MM30K_VALIDATION_QUOTAS
    )
    train_reserve_fraction: float = 0.20
    validation_reserve_fraction: float = 0.20
    minimum_reserve_per_stratum: int = 64
    fixture_scale_test_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.seed, str) or not self.seed.strip():
            raise ValueError("MM30K seed is required")
        object.__setattr__(
            self, "train_quotas", _normalized_quotas(self.train_quotas, "train_quotas")
        )
        object.__setattr__(
            self,
            "validation_quotas",
            _normalized_quotas(self.validation_quotas, "validation_quotas"),
        )
        for name in ("train_reserve_fraction", "validation_reserve_fraction"):
            value = getattr(self, name)
            if type(value) is not float or not 0.0 < value <= 1.0:
                raise ValueError(name + " must be a float in (0, 1]")
        if (
            type(self.minimum_reserve_per_stratum) is not int
            or self.minimum_reserve_per_stratum <= 0
        ):
            raise ValueError("minimum_reserve_per_stratum must be positive")
        if type(self.fixture_scale_test_only) is not bool:
            raise TypeError("fixture_scale_test_only must be bool")
        if not self.fixture_scale_test_only and (
            dict(self.train_quotas) != dict(MM30K_TRAIN_QUOTAS)
            or dict(self.validation_quotas) != dict(MM30K_VALIDATION_QUOTAS)
        ):
            raise ValueError("production MM30K quotas are fixed")

    def reserve_quotas(self, split: str) -> Mapping[str, int]:
        if split == "train":
            quotas = self.train_quotas
            fraction = self.train_reserve_fraction
        elif split == "val":
            quotas = self.validation_quotas
            fraction = self.validation_reserve_fraction
        else:
            raise ValueError("MM30K split must be train or val")
        return MappingProxyType(
            {
                stratum: quota
                + max(
                    self.minimum_reserve_per_stratum,
                    int(quota * fraction + 0.999999),
                )
                for stratum, quota in quotas.items()
            }
        )


def mm30k_stratum(record: TrainingPairRecord, *, expected_split: str) -> str:
    """Classify one canonical MM record into origin x label."""

    if not isinstance(record, TrainingPairRecord):
        raise TypeError("MM30K input must be TrainingPairRecord")
    if expected_split not in {"train", "val"} or record.split != expected_split:
        raise MM30KSelectionError("MM30K record split changed")
    if record.dataset_id != "mm_augmented":
        raise MM30KSelectionError("MM30K accepts canonical MM only")
    variant = record.provenance.get("variant_id")
    if variant == "0":
        origin = "original"
    elif variant in _GENERATED_VARIANTS:
        origin = "generated"
    else:
        raise MM30KSelectionError("MM variant must be 0 or one of 1..6")
    return origin + ("_positive" if record.label else "_negative")


def _rank(seed: str, purpose: str, stratum: str, pair_token: str) -> int:
    payload = "\0".join((seed, purpose, stratum, pair_token)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def select_mm30k_ranked_reserve(
    records: Iterable[TrainingPairRecord],
    *,
    split: str,
    config: MM30KSelectionConfig,
) -> Tuple[TrainingPairRecord, ...]:
    """One-pass, bounded-memory, input-order-independent stratified top-k."""

    if not isinstance(config, MM30KSelectionConfig):
        raise TypeError("config must be MM30KSelectionConfig")
    quotas = config.reserve_quotas(split)
    heaps: Dict[str, list] = {stratum: [] for stratum in MM30K_STRATA}
    seen = set()
    for record in records:
        stratum = mm30k_stratum(record, expected_split=split)
        token = record.pair_id
        if token in seen:
            raise MM30KSelectionError("canonical MM stream repeated a pair")
        seen.add(token)
        rank = _rank(config.seed, split + "-reserve", stratum, token)
        entry = (-rank, token, record)
        heap = heaps[stratum]
        if len(heap) < quotas[stratum]:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    shortages = {
        stratum: quotas[stratum] - len(heaps[stratum])
        for stratum in MM30K_STRATA
        if len(heaps[stratum]) < quotas[stratum]
    }
    if shortages:
        raise MM30KSelectionError("MM stream cannot fill reserve quotas: " + str(shortages))
    selected = [entry[2] for heap in heaps.values() for entry in heap]
    return tuple(
        sorted(
            selected,
            key=lambda record: (
                _rank(
                    config.seed,
                    split + "-reserve-order",
                    mm30k_stratum(record, expected_split=split),
                    record.pair_id,
                ),
                record.pair_id,
            ),
        )
    )


def select_mm30k_qualified(
    reserve: Sequence[TrainingPairRecord],
    *,
    split: str,
    quotas: Mapping[str, int],
    eligible_tokens: Mapping[str, bool],
    seed: str,
) -> Tuple[TrainingPairRecord, ...]:
    """Take exact quotas from a ranked reserve after label-blind qualification."""

    normalized = _normalized_quotas(quotas, "quotas")
    by_stratum: Dict[str, list] = {stratum: [] for stratum in MM30K_STRATA}
    for record in reserve:
        if eligible_tokens.get(record.pair_id) is True:
            by_stratum[mm30k_stratum(record, expected_split=split)].append(record)
    shortages = {
        stratum: normalized[stratum] - len(by_stratum[stratum])
        for stratum in MM30K_STRATA
        if len(by_stratum[stratum]) < normalized[stratum]
    }
    if shortages:
        raise MM30KSelectionError(
            "geometry-qualified MM reserve cannot fill quotas: " + str(shortages)
        )
    selected = []
    for stratum in MM30K_STRATA:
        ranked = sorted(
            by_stratum[stratum],
            key=lambda record: (
                _rank(seed, split + "-qualified", stratum, record.pair_id),
                record.pair_id,
            ),
        )
        selected.extend(ranked[: normalized[stratum]])
    return tuple(
        sorted(
            selected,
            key=lambda record: (
                _rank(seed, split + "-final-order", "all", record.pair_id),
                record.pair_id,
            ),
        )
    )


def mm30k_stratum_counts(
    records: Sequence[TrainingPairRecord], *, split: str
) -> Mapping[str, int]:
    counts = {stratum: 0 for stratum in MM30K_STRATA}
    for record in records:
        counts[mm30k_stratum(record, expected_split=split)] += 1
    return MappingProxyType(counts)


__all__ = [
    "MM30K_DEFAULT_SEED",
    "MM30K_SELECTION_SCHEMA_VERSION",
    "MM30K_STRATA",
    "MM30K_TRAIN_QUOTAS",
    "MM30K_VALIDATION_QUOTAS",
    "MM30KSelectionConfig",
    "MM30KSelectionError",
    "mm30k_stratum",
    "mm30k_stratum_counts",
    "select_mm30k_qualified",
    "select_mm30k_ranked_reserve",
]
