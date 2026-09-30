"""Deterministic bounded sampling for Pairwise v0.2 metadata streams.

Training uses an exact-ratio, bounded-memory reservoir.  It can mix static
geometry-hard negatives with model-mined scores supplied by ``pair_id``.
Validation has a separate API that preserves every record exactly once in
source order; it never balances, shuffles, oversamples, or mines negatives.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Union

from .training_stream import TrainingPairRecord


SAMPLER_VERSION = "dunhuang-pairwise-balanced-reservoir/0.2"


class SamplingError(ValueError):
    """Raised when an epoch cannot satisfy the requested sampling contract."""


@dataclass(frozen=True)
class BalancedSamplingConfig:
    """Exact epoch composition for train-only reservoir sampling."""

    epoch_size: int
    positive_fraction: float = 0.5
    hard_negative_fraction: float = 0.5
    hard_negative_threshold: float = 0.0
    seed: str = "pairwise-v0.2-balanced-train"
    replacement_on_shortfall: bool = True

    def __post_init__(self) -> None:
        if type(self.epoch_size) is not int or self.epoch_size <= 0:  # noqa: E721
            raise ValueError("epoch_size must be a positive int")
        if (
            not math.isfinite(self.positive_fraction)
            or not 0 < self.positive_fraction < 1
        ):
            raise ValueError("positive_fraction must be finite and strictly in (0, 1)")
        if (
            not math.isfinite(self.hard_negative_fraction)
            or not 0 <= self.hard_negative_fraction <= 1
        ):
            raise ValueError("hard_negative_fraction must be finite in [0, 1]")
        if (
            not math.isfinite(self.hard_negative_threshold)
            or not 0 <= self.hard_negative_threshold <= 1
        ):
            raise ValueError("hard_negative_threshold must be finite in [0, 1]")
        if not isinstance(self.seed, str) or not self.seed:
            raise ValueError("seed must be a non-empty string")

    @property
    def targets(self) -> Mapping[str, int]:
        positives = int(math.floor(self.epoch_size * self.positive_fraction + 0.5))
        negatives = self.epoch_size - positives
        hard = int(math.floor(negatives * self.hard_negative_fraction + 0.5))
        return {
            "positive": positives,
            "hard_negative": hard,
            "ordinary_negative": negatives - hard,
        }


@dataclass(frozen=True)
class SamplingAudit:
    sampler_version: str
    epoch: int
    input_count: int
    input_positive_count: int
    input_negative_count: int
    input_hard_negative_count: int
    output_count: int
    output_positive_count: int
    output_negative_count: int
    output_hard_negative_count: int
    unique_output_pair_count: int
    replacement_count: int
    config: Mapping[str, object]

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["config"] = dict(self.config)
        return result


class _Reservoir:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = capacity
        self.rng = rng
        self.observed = 0
        self.items: List[TrainingPairRecord] = []

    def consider(self, record: TrainingPairRecord) -> None:
        self.observed += 1
        if self.capacity == 0:
            return
        if len(self.items) < self.capacity:
            self.items.append(record)
            return
        selected = self.rng.randrange(self.observed)
        if selected < self.capacity:
            self.items[selected] = record


def _seed_int(seed: str, epoch: int, namespace: str) -> int:
    payload = json.dumps([seed, epoch, namespace], separators=(",", ":")).encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _fill_to_target(
    items: List[TrainingPairRecord],
    target: int,
    *,
    rng: random.Random,
    replacement: bool,
    category: str,
) -> List[TrainingPairRecord]:
    if target == 0:
        return []
    if not items:
        raise SamplingError("no {} records available".format(category))
    if len(items) >= target:
        return list(items[:target])
    if not replacement:
        raise SamplingError(
            "{} shortfall: requested {}, observed {}".format(
                category, target, len(items)
            )
        )
    output = list(items)
    while len(output) < target:
        output.append(items[rng.randrange(len(items))])
    return output


RecordFactory = Callable[[], Iterable[TrainingPairRecord]]


class BalancedPairSampler:
    """Select an exact-composition training epoch with bounded record memory."""

    def __init__(self, config: BalancedSamplingConfig) -> None:
        self.config = config
        self.last_audit: Optional[SamplingAudit] = None

    def sample_epoch(
        self,
        records: Union[Iterable[TrainingPairRecord], RecordFactory],
        *,
        epoch: int,
        mined_negative_scores: Optional[Mapping[str, float]] = None,
    ) -> Iterator[TrainingPairRecord]:
        if type(epoch) is not int or epoch < 0:  # noqa: E721
            raise ValueError("epoch must be a non-negative int")
        source = records() if callable(records) else records
        targets = self.config.targets
        reservoirs = {
            name: _Reservoir(
                target,
                random.Random(_seed_int(self.config.seed, epoch, name)),
            )
            for name, target in targets.items()
        }
        input_count = 0
        positive_count = 0
        negative_count = 0
        hard_count = 0
        score_map = mined_negative_scores or {}
        for record in source:
            if not isinstance(record, TrainingPairRecord):
                raise TypeError("sampler requires TrainingPairRecord values")
            if record.split != "train":
                raise SamplingError("balanced sampling is train-only")
            input_count += 1
            if record.label:
                positive_count += 1
                reservoirs["positive"].consider(record)
                continue
            negative_count += 1
            score = score_map.get(record.pair_id, record.static_hard_negative_score)
            is_hard = False
            if score is not None:
                if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
                    raise SamplingError("mined negative score must be finite in [0, 1]")
                is_hard = float(score) > self.config.hard_negative_threshold
            if is_hard:
                hard_count += 1
                reservoirs["hard_negative"].consider(record)
            else:
                reservoirs["ordinary_negative"].consider(record)

        selected: List[TrainingPairRecord] = []
        for name in ("positive", "hard_negative", "ordinary_negative"):
            rng = random.Random(_seed_int(self.config.seed, epoch, "fill/" + name))
            selected.extend(
                _fill_to_target(
                    reservoirs[name].items,
                    targets[name],
                    rng=rng,
                    replacement=self.config.replacement_on_shortfall,
                    category=name,
                )
            )
        shuffle_rng = random.Random(_seed_int(self.config.seed, epoch, "shuffle"))
        shuffle_rng.shuffle(selected)
        output_hard_count = sum(
            (not item.label)
            and (
                float(
                    score_map.get(item.pair_id, item.static_hard_negative_score) or 0.0
                )
                > self.config.hard_negative_threshold
            )
            for item in selected
        )
        unique_count = len({item.pair_id for item in selected})
        self.last_audit = SamplingAudit(
            sampler_version=SAMPLER_VERSION,
            epoch=epoch,
            input_count=input_count,
            input_positive_count=positive_count,
            input_negative_count=negative_count,
            input_hard_negative_count=hard_count,
            output_count=len(selected),
            output_positive_count=sum(item.label for item in selected),
            output_negative_count=sum(not item.label for item in selected),
            output_hard_negative_count=output_hard_count,
            unique_output_pair_count=unique_count,
            replacement_count=len(selected) - unique_count,
            config=asdict(self.config),
        )
        yield from selected


def iter_frozen_validation(
    records: Iterable[TrainingPairRecord],
) -> Iterator[TrainingPairRecord]:
    """Yield validation records once in source order with no sampling mutation."""

    for record in records:
        if not isinstance(record, TrainingPairRecord):
            raise TypeError("validation stream requires TrainingPairRecord values")
        if record.split != "val":
            raise SamplingError("frozen validation accepts only val records")
        yield record


def validation_stream_fingerprint(
    records: Iterable[TrainingPairRecord],
) -> Mapping[str, object]:
    """Hash the exact frozen validation order without loading mask pixels."""

    digest = hashlib.sha256()
    count = 0
    positives = 0
    for record in iter_frozen_validation(records):
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
        count += 1
        positives += int(record.label)
    return {
        "schema_version": "dunhuang-pairwise-validation-fingerprint/0.2",
        "count": count,
        "positive_count": positives,
        "negative_count": count - positives,
        "sha256": digest.hexdigest(),
        "order": "frozen_source_order",
        "mask_pixels_loaded": False,
    }


__all__ = [
    "BalancedPairSampler",
    "BalancedSamplingConfig",
    "SAMPLER_VERSION",
    "SamplingAudit",
    "SamplingError",
    "iter_frozen_validation",
    "validation_stream_fingerprint",
]
