"""Freeze and replay the deterministic exact-seam pilot population.

The expensive part of the exact-seam population builder is the geometry
eligibility predicate.  This module evaluates that predicate per parent group
in worker processes, replays the original quota logic in the original group
and pair order, and stores only the resulting ordered pair IDs.  Loading a
snapshot rescans masks from the first consumed groups to reconstruct records;
it never rebuilds geometry or calls the eligibility predicate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord
from staging.pairwise_v0_2.training import exact_seam_pilot as exact
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.geometry_cache import GeometryArtifactCache


SNAPSHOT_SCHEMA_VERSION = "dunhuang-exact-pilot-population-snapshot/0.1"


class ExactSeamPopulationSnapshotError(exact.ExactSeamPilotError):
    """A population snapshot is incomplete or incompatible with the pilot."""


def _require_exact_config(value: Any) -> None:
    """Accept the canonical config and its ``python -m`` ``__main__`` twin."""

    required = (
        "mask_root",
        "output_root",
        "cache_root",
        "max_pairs",
        "validation_fraction",
        "seed",
        "generator",
        "train_pair_count",
        "validation_pair_count",
    )
    if isinstance(value, exact.ExactSeamPilotConfig):
        return
    if type(value).__name__ != "ExactSeamPilotConfig" or any(
        not hasattr(value, name) for name in required
    ):
        raise TypeError("config must be ExactSeamPilotConfig")


def _require_exact_population(value: Any) -> None:
    """Accept the canonical population and its ``python -m`` class twin."""

    required = (
        "training_records",
        "validation_records",
        "discovered_group_count",
        "consumed_group_count",
    )
    if isinstance(value, exact.ExactSeamPilotPopulation):
        return
    if type(value).__name__ != "ExactSeamPilotPopulation" or any(
        not hasattr(value, name) for name in required
    ):
        raise TypeError("population must be ExactSeamPilotPopulation")


@dataclass(frozen=True)
class _GroupTask:
    index: int
    relative_group: str
    split: str


@dataclass(frozen=True)
class _GroupResult:
    index: int
    relative_group: str
    split: str
    pair_ids: Tuple[str, ...]
    labels: Tuple[bool, ...]
    eligible: Tuple[bool, ...]


_WORKER_MASK_ROOT: Optional[Path] = None
_WORKER_SEED: Optional[int] = None
_WORKER_LOADER: Optional[exact._DirectoryMaskLoader] = None
_WORKER_CACHE: Optional[GeometryArtifactCache] = None
_WORKER_GEOMETRY_CONFIG: Optional[GeometryBatchConfig] = None


def _initialize_worker(
    mask_root: str,
    cache_root: str,
    seed: int,
    geometry_config: GeometryBatchConfig,
) -> None:
    global _WORKER_MASK_ROOT
    global _WORKER_SEED
    global _WORKER_LOADER
    global _WORKER_CACHE
    global _WORKER_GEOMETRY_CONFIG

    root = Path(mask_root)
    _WORKER_MASK_ROOT = root
    _WORKER_SEED = seed
    _WORKER_LOADER = exact._DirectoryMaskLoader(root)
    # Keep this writable: a partially warmed cache may safely be completed by
    # concurrent workers using GeometryArtifactCache's atomic commits.
    _WORKER_CACHE = GeometryArtifactCache(Path(cache_root))
    _WORKER_GEOMETRY_CONFIG = geometry_config


def _evaluate_group(task: _GroupTask) -> _GroupResult:
    if (
        _WORKER_MASK_ROOT is None
        or _WORKER_SEED is None
        or _WORKER_LOADER is None
        or _WORKER_CACHE is None
        or _WORKER_GEOMETRY_CONFIG is None
    ):
        raise RuntimeError("exact-seam snapshot worker was not initialized")
    group = _WORKER_MASK_ROOT / task.relative_group
    records = exact._records_for_group(
        _WORKER_MASK_ROOT,
        group,
        split=task.split,
        seed=_WORKER_SEED,
    )
    decisions = exact.exact_seam_group_eligibility(
        records,
        loader=_WORKER_LOADER,
        cache=_WORKER_CACHE,
        geometry_config=_WORKER_GEOMETRY_CONFIG,
    )
    return _GroupResult(
        index=task.index,
        relative_group=task.relative_group,
        split=task.split,
        pair_ids=tuple(record.pair_id for record in records),
        labels=tuple(record.label for record in records),
        eligible=tuple(decisions),
    )


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactSeamPopulationSnapshotError("{} must be a JSON object".format(name))
    return value


def _load_json_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, (str, os.PathLike, Path)):
        path = Path(value)
        try:
            with path.open("r", encoding="utf-8") as stream:
                loaded = json.load(stream)
        except (OSError, ValueError) as exc:
            raise ExactSeamPopulationSnapshotError(
                "could not load {}: {}".format(name, path)
            ) from exc
        return _as_mapping(loaded, name)
    return _as_mapping(value, name)


def _summary_population(value: Any) -> Mapping[str, Any]:
    payload = _load_json_mapping(value, "summary_population")
    nested = payload.get("population")
    if nested is not None:
        return _as_mapping(nested, "summary_population.population")
    return payload


def _required_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExactSeamPopulationSnapshotError(
            "{} must be a positive integer".format(name)
        )
    return value


def _optional_count(payload: Mapping[str, Any], *names: str) -> Optional[int]:
    for name in names:
        if name in payload:
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExactSeamPopulationSnapshotError(
                    "{} must be a non-negative integer".format(name)
                )
            return value
    return None


def _consumed_group_count(summary_population: Mapping[str, Any]) -> int:
    value = _optional_count(
        summary_population,
        "consumed_group_count",
        "consumed_groups",
    )
    if value is None or value <= 0:
        raise ExactSeamPopulationSnapshotError(
            "summary population must provide consumed_group_count"
        )
    return value


def _collect_results(
    config: exact.ExactSeamPilotConfig,
    tasks: Sequence[_GroupTask],
    *,
    workers: int,
    geometry_config: GeometryBatchConfig,
    chunksize: int,
    progress_callback: Optional[Callable[[int, int], None]],
) -> Tuple[_GroupResult, ...]:
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(
            str(config.mask_root),
            str(config.cache_root),
            config.seed,
            geometry_config,
        ),
    ) as executor:
        iterator = executor.map(_evaluate_group, tasks, chunksize=chunksize)
        values: List[_GroupResult] = []
        for result in iterator:
            values.append(result)
            if progress_callback is not None:
                progress_callback(len(values), len(tasks))
    return tuple(values)


def _validate_ordered_group_result(
    result: _GroupResult,
    task: _GroupTask,
) -> None:
    if not isinstance(result, _GroupResult):
        raise ExactSeamPopulationSnapshotError(
            "parallel eligibility worker returned an invalid result"
        )
    if (
        result.index != task.index
        or result.relative_group != task.relative_group
        or result.split != task.split
    ):
        raise ExactSeamPopulationSnapshotError(
            "parallel group result does not match its canonical input position"
        )
    if not (len(result.pair_ids) == len(result.labels) == len(result.eligible)):
        raise ExactSeamPopulationSnapshotError(
            "group result must contain one eligibility decision per pair"
        )


def _parallel_population_results(
    config: exact.ExactSeamPilotConfig,
    tasks: Sequence[_GroupTask],
    *,
    workers: int,
    geometry_config: GeometryBatchConfig,
    progress_callback: Optional[Callable[[int, int], None]],
) -> Tuple[_GroupResult, ...]:
    """Evaluate a bounded worker window and consume it in canonical order."""

    targets = {
        ("train", True): config.train_pair_count // 2,
        ("train", False): config.train_pair_count // 2,
        ("val", True): config.validation_pair_count // 2,
        ("val", False): config.validation_pair_count // 2,
    }
    counts = {key: 0 for key in targets}
    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(
            str(config.mask_root),
            str(config.cache_root),
            config.seed,
            geometry_config,
        ),
    )
    futures: Dict[int, Any] = {}
    next_submit = 0
    values: List[_GroupResult] = []

    def submit_next() -> None:
        nonlocal next_submit
        if next_submit >= len(tasks):
            return
        task = tasks[next_submit]
        try:
            futures[task.index] = executor.submit(_evaluate_group, task)
        except Exception as exc:
            raise ExactSeamPopulationSnapshotError(
                "could not submit parallel eligibility group {}".format(
                    task.relative_group
                )
            ) from exc
        next_submit += 1

    try:
        for _ in range(min(workers, len(tasks))):
            submit_next()
        next_consume = 0
        while next_consume < len(tasks):
            task = tasks[next_consume]
            future = futures.pop(task.index)
            try:
                result = future.result()
            except Exception as exc:
                raise ExactSeamPopulationSnapshotError(
                    "parallel eligibility worker failed for group {}".format(
                        task.relative_group
                    )
                ) from exc
            _validate_ordered_group_result(result, task)
            values.append(result)
            for label, eligible in zip(result.labels, result.eligible):
                key = (result.split, bool(label))
                if bool(eligible) and counts[key] < targets[key]:
                    counts[key] += 1
            if progress_callback is not None:
                progress_callback(len(values), len(tasks))
            next_consume += 1
            if all(counts[key] >= target for key, target in targets.items()):
                break
            submit_next()
    finally:
        for future in futures.values():
            future.cancel()
        # At most ``workers - 1`` already-running groups remain here.  Waiting
        # for them avoids orphan workers while still preventing an all-groups
        # tail after the quota has been filled.
        executor.shutdown(wait=True)
    return tuple(values)


def build_exact_seam_pilot_population_parallel(
    config: exact.ExactSeamPilotConfig,
    *,
    workers: int,
    geometry_config: Optional[GeometryBatchConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> exact.ExactSeamPilotPopulation:
    """Build the exact population with deterministic parallel eligibility.

    Workers evaluate concrete ``exact_seam_group_eligibility`` calls.  The
    parent consumes results strictly in the original canonical group order and
    applies the historical per-split/per-label quotas there, so scheduling and
    worker count cannot affect selected pair IDs or their final order.
    """

    _require_exact_config(config)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    settings = geometry_config or GeometryBatchConfig()
    if not isinstance(settings, GeometryBatchConfig):
        raise TypeError("geometry_config must be GeometryBatchConfig")
    groups = exact.discover_exact_mask_groups(
        config.mask_root,
        generator=config.generator,
        seed=config.seed,
    )
    tasks = tuple(
        _GroupTask(
            index=index,
            relative_group=group.relative_to(config.mask_root).as_posix(),
            split=exact._split_for_group(
                group.relative_to(config.mask_root).as_posix(),
                config.seed,
                config.validation_fraction,
            ),
        )
        for index, group in enumerate(groups)
    )
    results = _parallel_population_results(
        config,
        tasks,
        workers=workers,
        geometry_config=settings,
        progress_callback=progress_callback,
    )
    training_ids, validation_ids, consumed = _replay_population_ids(config, results)
    selected_ids = set(training_ids) | set(validation_ids)
    records: Dict[str, TrainingPairRecord] = {}
    for result in results[:consumed]:
        group = config.mask_root / result.relative_group
        group_records = exact._records_for_group(
            config.mask_root,
            group,
            split=result.split,
            seed=config.seed,
        )
        if tuple(record.pair_id for record in group_records) != result.pair_ids:
            raise ExactSeamPopulationSnapshotError(
                "parallel pair IDs changed during ordered parent reconstruction"
            )
        for record in group_records:
            if record.pair_id not in selected_ids:
                continue
            if record.pair_id in records:
                raise ExactSeamPopulationSnapshotError(
                    "parallel population reconstructed a duplicate pair ID"
                )
            records[record.pair_id] = record
    if not selected_ids.issubset(records):
        raise ExactSeamPopulationSnapshotError(
            "parallel population omitted one or more selected records"
        )
    return exact.ExactSeamPilotPopulation(
        training_records=tuple(records[pair_id] for pair_id in training_ids),
        validation_records=tuple(records[pair_id] for pair_id in validation_ids),
        discovered_group_count=len(groups),
        consumed_group_count=consumed,
    )


def _replay_population_ids(
    config: exact.ExactSeamPilotConfig,
    results: Sequence[_GroupResult],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], int]:
    targets = {
        ("train", True): config.train_pair_count // 2,
        ("train", False): config.train_pair_count // 2,
        ("val", True): config.validation_pair_count // 2,
        ("val", False): config.validation_pair_count // 2,
    }
    buckets: Dict[Tuple[str, bool], List[str]] = {key: [] for key in targets}
    consumed = 0
    for expected_index, result in enumerate(results):
        if result.index != expected_index:
            raise ExactSeamPopulationSnapshotError(
                "parallel group results are not in deterministic input order"
            )
        if result.split not in {"train", "val"}:
            raise ExactSeamPopulationSnapshotError("group result split is invalid")
        if not (len(result.pair_ids) == len(result.labels) == len(result.eligible)):
            raise ExactSeamPopulationSnapshotError(
                "group result must contain one eligibility decision per pair"
            )
        consumed += 1
        for pair_id, label, eligible in zip(
            result.pair_ids, result.labels, result.eligible
        ):
            key = (result.split, bool(label))
            bucket = buckets[key]
            if len(bucket) < targets[key] and bool(eligible):
                bucket.append(pair_id)
        if all(len(buckets[key]) >= count for key, count in targets.items()):
            break

    shortages = {
        "{}:{}".format(split, "positive" if label else "negative"): (
            targets[(split, label)] - len(buckets[(split, label)])
        )
        for split, label in targets
        if len(buckets[(split, label)]) < targets[(split, label)]
    }
    if shortages:
        raise ExactSeamPopulationSnapshotError(
            "parallel eligibility did not fill the requested population: {}".format(
                shortages
            )
        )

    def combined(split: str) -> Tuple[str, ...]:
        values = buckets[(split, True)] + buckets[(split, False)]
        values.sort(
            key=lambda pair_id: exact._rank(config.seed, split + "-final", pair_id)
        )
        return tuple(values)

    return combined("train"), combined("val"), consumed


def _validate_summary_counts(
    summary: Mapping[str, Any],
    config: exact.ExactSeamPilotConfig,
) -> None:
    expected = {
        "train_pairs": config.train_pair_count,
        "validation_pairs": config.validation_pair_count,
        "train_positive_count": config.train_pair_count // 2,
        "train_negative_count": config.train_pair_count // 2,
        "validation_positive_count": config.validation_pair_count // 2,
        "validation_negative_count": config.validation_pair_count // 2,
    }
    aliases = {
        "train_pairs": ("train_pairs", "training_pair_count"),
        "validation_pairs": ("validation_pairs", "validation_pair_count"),
        "train_positive_count": ("train_positive_count",),
        "train_negative_count": ("train_negative_count",),
        "validation_positive_count": ("validation_positive_count",),
        "validation_negative_count": ("validation_negative_count",),
    }
    for canonical, names in aliases.items():
        observed = _optional_count(summary, *names)
        if observed is not None and observed != expected[canonical]:
            raise ExactSeamPopulationSnapshotError(
                "summary {} does not match config".format(canonical)
            )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(output))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def write_exact_seam_population_snapshot(
    config: exact.ExactSeamPilotConfig,
    population: exact.ExactSeamPilotPopulation,
    output: Path,
) -> Mapping[str, Any]:
    """Persist an already eligibility-filtered population without recomputing it."""

    _require_exact_config(config)
    _require_exact_population(population)
    training = population.training_records
    validation = population.validation_records
    expected_counts = {
        "train_pairs": config.train_pair_count,
        "validation_pairs": config.validation_pair_count,
        "train_positive_count": config.train_pair_count // 2,
        "train_negative_count": config.train_pair_count // 2,
        "validation_positive_count": config.validation_pair_count // 2,
        "validation_negative_count": config.validation_pair_count // 2,
    }
    observed_counts = {
        "train_pairs": len(training),
        "validation_pairs": len(validation),
        "train_positive_count": sum(record.label for record in training),
        "train_negative_count": sum(not record.label for record in training),
        "validation_positive_count": sum(record.label for record in validation),
        "validation_negative_count": sum(not record.label for record in validation),
    }
    if observed_counts != expected_counts:
        raise ExactSeamPopulationSnapshotError(
            "population counts do not match exact-seam config"
        )
    if any(record.split != "train" for record in training) or any(
        record.split != "val" for record in validation
    ):
        raise ExactSeamPopulationSnapshotError(
            "population records do not match their ordered split"
        )

    payload: Dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "pilot": {
            "seed": config.seed,
            "generator": config.generator,
            "max_pairs": config.max_pairs,
            "validation_fraction": config.validation_fraction,
            "consumed_group_count": population.consumed_group_count,
        },
        "population": {
            "training_pair_ids": [record.pair_id for record in training],
            "validation_pair_ids": [record.pair_id for record in validation],
            **observed_counts,
            "group_disjoint": True,
        },
    }
    _atomic_write_json(Path(output), payload)
    return payload


def freeze_exact_seam_population_snapshot(
    config: exact.ExactSeamPilotConfig,
    summary_population: Any,
    workers: int,
    output: Path,
    *,
    geometry_config: Optional[GeometryBatchConfig] = None,
    chunksize: int = 1,
    include_group_decisions: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Mapping[str, Any]:
    """Parallelize eligibility and persist the exact ordered population IDs.

    ``summary_population`` may be either the population object from a completed
    pilot summary, the full summary object, or a path to that JSON file.  Its
    consumed-group count bounds the parallel work and is checked against the
    replayed quota stopping point.
    """

    _require_exact_config(config)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    settings = geometry_config or GeometryBatchConfig()
    if not isinstance(settings, GeometryBatchConfig):
        raise TypeError("geometry_config must be GeometryBatchConfig")

    summary = _summary_population(summary_population)
    _validate_summary_counts(summary, config)
    expected_consumed = _consumed_group_count(summary)
    groups = exact.discover_exact_mask_groups(
        config.mask_root, generator=config.generator, seed=config.seed
    )
    if expected_consumed > len(groups):
        raise ExactSeamPopulationSnapshotError(
            "consumed_group_count exceeds discovered groups"
        )
    discovered = _optional_count(summary, "discovered_group_count")
    if discovered is not None and discovered != len(groups):
        raise ExactSeamPopulationSnapshotError(
            "summary discovered_group_count does not match mask_root"
        )

    tasks = tuple(
        _GroupTask(
            index=index,
            relative_group=group.relative_to(config.mask_root).as_posix(),
            split=exact._split_for_group(
                group.relative_to(config.mask_root).as_posix(),
                config.seed,
                config.validation_fraction,
            ),
        )
        for index, group in enumerate(groups[:expected_consumed])
    )
    results = _collect_results(
        config,
        tasks,
        workers=workers,
        geometry_config=settings,
        chunksize=chunksize,
        progress_callback=progress_callback,
    )
    training_ids, validation_ids, consumed = _replay_population_ids(config, results)
    if consumed != expected_consumed:
        raise ExactSeamPopulationSnapshotError(
            "replayed consumed_group_count {} != summary {}".format(
                consumed, expected_consumed
            )
        )

    pilot = {
        "seed": config.seed,
        "generator": config.generator,
        "max_pairs": config.max_pairs,
        "validation_fraction": config.validation_fraction,
        "consumed_group_count": consumed,
    }
    population: Dict[str, Any] = {
        "training_pair_ids": list(training_ids),
        "validation_pair_ids": list(validation_ids),
        "train_pairs": len(training_ids),
        "validation_pairs": len(validation_ids),
        "train_positive_count": config.train_pair_count // 2,
        "train_negative_count": config.train_pair_count // 2,
        "validation_positive_count": config.validation_pair_count // 2,
        "validation_negative_count": config.validation_pair_count // 2,
        "group_disjoint": True,
    }
    if include_group_decisions:
        population["group_decisions"] = [
            {
                "relative_group": result.relative_group,
                "split": result.split,
                "pair_ids": list(result.pair_ids),
                "labels": list(result.labels),
                "eligible": list(result.eligible),
            }
            for result in results
        ]
    payload: Dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "pilot": pilot,
        "population": population,
    }
    _atomic_write_json(Path(output), payload)
    return payload


def _pair_id_sequence(value: Any, name: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExactSeamPopulationSnapshotError("{} must be an array".format(name))
    values = tuple(value)
    if any(not isinstance(pair_id, str) or not pair_id for pair_id in values):
        raise ExactSeamPopulationSnapshotError(
            "{} must contain non-empty pair IDs".format(name)
        )
    if len(set(values)) != len(values):
        raise ExactSeamPopulationSnapshotError(
            "{} contains duplicate pair IDs".format(name)
        )
    return values


def _validate_pilot_identity(
    pilot: Mapping[str, Any], config: exact.ExactSeamPilotConfig
) -> int:
    expected = {
        "seed": config.seed,
        "generator": config.generator,
        "max_pairs": config.max_pairs,
    }
    for name, value in expected.items():
        if pilot.get(name) != value:
            raise ExactSeamPopulationSnapshotError(
                "snapshot pilot {} does not match config".format(name)
            )
    fraction = pilot.get("validation_fraction")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
        raise ExactSeamPopulationSnapshotError(
            "snapshot validation_fraction is invalid"
        )
    if abs(float(fraction) - config.validation_fraction) > 1e-12:
        raise ExactSeamPopulationSnapshotError(
            "snapshot pilot validation_fraction does not match config"
        )
    return _required_positive_int(
        pilot.get("consumed_group_count"), "pilot.consumed_group_count"
    )


def load_exact_seam_population_snapshot(
    path: Path,
    config: exact.ExactSeamPilotConfig,
) -> exact.ExactSeamPilotPopulation:
    """Reconstruct a snapshot population without geometry eligibility work."""

    _require_exact_config(config)
    payload = _load_json_mapping(path, "population snapshot")
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ExactSeamPopulationSnapshotError(
            "unsupported exact-seam population snapshot schema"
        )
    pilot = _as_mapping(payload.get("pilot"), "snapshot.pilot")
    population = _as_mapping(payload.get("population"), "snapshot.population")
    consumed = _validate_pilot_identity(pilot, config)
    population_consumed = _optional_count(population, "consumed_group_count")
    if population_consumed is not None and population_consumed != consumed:
        raise ExactSeamPopulationSnapshotError(
            "snapshot consumed_group_count fields disagree"
        )
    training_ids = _pair_id_sequence(
        population.get("training_pair_ids"), "population.training_pair_ids"
    )
    validation_ids = _pair_id_sequence(
        population.get("validation_pair_ids"), "population.validation_pair_ids"
    )
    if set(training_ids) & set(validation_ids):
        raise ExactSeamPopulationSnapshotError(
            "training and validation pair IDs overlap"
        )
    if len(training_ids) != config.train_pair_count:
        raise ExactSeamPopulationSnapshotError(
            "snapshot training pair count does not match config"
        )
    if len(validation_ids) != config.validation_pair_count:
        raise ExactSeamPopulationSnapshotError(
            "snapshot validation pair count does not match config"
        )
    _validate_summary_counts(population, config)
    if population.get("group_disjoint", True) is not True:
        raise ExactSeamPopulationSnapshotError("snapshot is not group-disjoint")

    groups = exact.discover_exact_mask_groups(
        config.mask_root, generator=config.generator, seed=config.seed
    )
    if consumed > len(groups):
        raise ExactSeamPopulationSnapshotError(
            "snapshot consumed_group_count exceeds discovered groups"
        )
    selected_ids = set(training_ids) | set(validation_ids)
    selected: Dict[str, TrainingPairRecord] = {}
    for group in groups[:consumed]:
        relative = group.relative_to(config.mask_root).as_posix()
        split = exact._split_for_group(
            relative, config.seed, config.validation_fraction
        )
        for record in exact._records_for_group(
            config.mask_root, group, split=split, seed=config.seed
        ):
            if record.pair_id in selected_ids:
                if record.pair_id in selected:
                    raise ExactSeamPopulationSnapshotError(
                        "a selected pair ID was reconstructed more than once"
                    )
                selected[record.pair_id] = record
    missing = selected_ids - set(selected)
    if missing:
        raise ExactSeamPopulationSnapshotError(
            "{} snapshot pair IDs were not found in the first consumed groups".format(
                len(missing)
            )
        )
    training = tuple(selected[pair_id] for pair_id in training_ids)
    validation = tuple(selected[pair_id] for pair_id in validation_ids)
    if any(record.split != "train" for record in training):
        raise ExactSeamPopulationSnapshotError(
            "a stored training pair reconstructs into validation"
        )
    if any(record.split != "val" for record in validation):
        raise ExactSeamPopulationSnapshotError(
            "a stored validation pair reconstructs into training"
        )
    expected_labels = {
        "train_positive_count": sum(record.label for record in training),
        "train_negative_count": sum(not record.label for record in training),
        "validation_positive_count": sum(record.label for record in validation),
        "validation_negative_count": sum(not record.label for record in validation),
    }
    for name, observed in expected_labels.items():
        if observed != population.get(name, observed):
            raise ExactSeamPopulationSnapshotError(
                "snapshot {} does not match reconstructed records".format(name)
            )
    if expected_labels != {
        "train_positive_count": config.train_pair_count // 2,
        "train_negative_count": config.train_pair_count // 2,
        "validation_positive_count": config.validation_pair_count // 2,
        "validation_negative_count": config.validation_pair_count // 2,
    }:
        raise ExactSeamPopulationSnapshotError(
            "snapshot population is not class-balanced"
        )

    return exact.ExactSeamPilotPopulation(
        training_records=training,
        validation_records=validation,
        discovered_group_count=len(groups),
        consumed_group_count=consumed,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or load an exact-seam population snapshot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--mask-root", type=Path, required=True)
        target.add_argument("--max-pairs", type=int, default=6964)
        target.add_argument("--validation-fraction", type=float, default=0.2)
        target.add_argument("--seed", type=int, default=260830)
        target.add_argument("--generator", default="gen4voronoi")

    build = subparsers.add_parser("build")
    add_config_arguments(build)
    build.add_argument("--cache-root", type=Path, required=True)
    build.add_argument("--pilot-summary", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    build.add_argument("--chunksize", type=int, default=1)
    build.add_argument("--include-group-decisions", action="store_true")

    validate = subparsers.add_parser("validate")
    add_config_arguments(validate)
    validate.add_argument("--snapshot", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    placeholder = args.output if args.command == "build" else args.snapshot.parent
    config = exact.ExactSeamPilotConfig(
        mask_root=args.mask_root,
        output_root=placeholder,
        cache_root=(args.cache_root if args.command == "build" else placeholder),
        max_pairs=args.max_pairs,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        generator=args.generator,
    )
    if args.command == "build":

        def progress(done: int, total: int) -> None:
            if done == total or done % 100 == 0:
                print("eligibility groups: {}/{}".format(done, total), flush=True)

        freeze_exact_seam_population_snapshot(
            config,
            args.pilot_summary,
            args.workers,
            args.output,
            chunksize=args.chunksize,
            include_group_decisions=args.include_group_decisions,
            progress_callback=progress,
        )
    else:
        loaded = load_exact_seam_population_snapshot(args.snapshot, config)
        print(
            "loaded train={} validation={} consumed_groups={}".format(
                len(loaded.training_records),
                len(loaded.validation_records),
                loaded.consumed_group_count,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
