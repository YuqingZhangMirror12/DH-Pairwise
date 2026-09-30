"""GPU pilot: weak keypoint Sinkhorn versus exact-seam supervision.

This runner is intentionally independent from the frozen four-arm LOCAL-Q1
runner.  It consumes only aligned scalar masks produced by the updated
``shredding_pipeline`` and compares two otherwise identical local models:

* ``keypoint_dustbin_sinkhorn`` (pair/direction weak supervision), and
* ``keypoint_dustbin_sinkhorn_exact_seam`` (the same losses plus a partial
  assignment NLL over exact parent-canvas seam correspondences and dustbins).

Both arms receive the same prepared batch, including the same target tensors;
the weak arm simply never passes those targets to its loss.  Targets are never
part of ``model_inputs``.  No RGB, text, OCR, rotation, or learned orientation
is used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

from staging.pairwise_v0_2.geometry import ContourKeypointConfig
from staging.pairwise_v0_2.models.local_matcher import MatcherMode
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.checkpoint import save_checkpoint
from staging.pairwise_v0_2.training.geometry_batch import (
    KEYPOINT_REPRESENTATION,
    GeometryBatchConfig,
    GeometryBatchError,
    build_geometry_batch,
)
from staging.pairwise_v0_2.training.geometry_cache import GeometryArtifactCache
from staging.pairwise_v0_2.training.local_q1_backend import (
    ExactSeamStepConfig,
    LocalQ1Backend,
    LocalQ1BackendMode,
)
from staging.pairwise_v0_2.training.local_q1_provider import (
    local_q1_prepared_digests,
)
from staging.pairwise_v0_2.training.metrics import binary_metrics
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    EvidenceMode,
    PreparedAblationBatch,
    record_sequence_fingerprint,
)


EXACT_SEAM_PILOT_VERSION = "dunhuang-exact-seam-synthetic-pilot/0.2"
_WINNER_SELECTION_POLICY = "synthetic_validation_auroc_then_auprc_then_earlier_epoch"
_DATASET_ID = "shredding_pipeline_exact_aligned_masks"
_BINDING = ArchiveBinding(
    logical_id="canonical://shredding_pipeline/exact_aligned_masks",
    archive_format="zip",
    # This pilot deliberately avoids an archive-integrity workflow.  The value
    # is a stable logical namespace token required by TrainingPairRecord, not a
    # claim about the bytes of a local directory.
    sha256=hashlib.sha256(
        b"shredding_pipeline/exact_aligned_masks/logical-namespace/v1"
    ).hexdigest(),
)
_SCALAR_MASK_MODES = frozenset({"1", "L", "I", "I;16"})
_MIN_PAIR_COUNT = 500
_MAX_PAIR_COUNT = 30000


class ExactSeamPilotError(RuntimeError):
    """The synthetic-only pilot cannot be constructed or executed."""


@dataclass(frozen=True)
class ExactSeamPilotConfig:
    mask_root: Path
    output_root: Path
    cache_root: Path
    max_pairs: int = 1000
    epochs: int = 3
    batch_size: int = 8
    validation_fraction: float = 0.2
    seed: int = 260830
    generator: str = "gen4voronoi"
    device: str = "cuda"
    exact_loss_weight: float = 0.25
    exact_loss_schedule: Optional[Tuple[float, ...]] = None
    initialization_seed: Optional[int] = None
    population_snapshot: Optional[Path] = None
    population_workers: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask_root", Path(self.mask_root))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "cache_root", Path(self.cache_root))
        if self.population_snapshot is not None:
            object.__setattr__(
                self, "population_snapshot", Path(self.population_snapshot)
            )
        if (
            isinstance(self.max_pairs, bool)
            or not isinstance(self.max_pairs, int)
            or not _MIN_PAIR_COUNT <= self.max_pairs <= _MAX_PAIR_COUNT
            or self.max_pairs % 2
        ):
            raise ValueError(
                "max_pairs must be an even integer in [{}, {}]".format(
                    _MIN_PAIR_COUNT, _MAX_PAIR_COUNT
                )
            )
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if (
            not math.isfinite(self.validation_fraction)
            or not 0.1 <= self.validation_fraction <= 0.4
        ):
            raise ValueError("validation_fraction must be finite in [0.1, 0.4]")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if self.initialization_seed is not None and (
            isinstance(self.initialization_seed, bool)
            or not isinstance(self.initialization_seed, int)
            or self.initialization_seed < 0
        ):
            raise ValueError("initialization_seed must be a non-negative integer")
        if (
            isinstance(self.population_workers, bool)
            or not isinstance(self.population_workers, int)
            or self.population_workers <= 0
        ):
            raise ValueError("population_workers must be a positive integer")
        if not self.generator:
            raise ValueError("generator is required; use 'all' to disable filtering")
        if torch.device(self.device).type != "cuda":
            raise ValueError("the exact-seam training pilot requires a CUDA device")
        if (
            isinstance(self.exact_loss_weight, bool)
            or not math.isfinite(float(self.exact_loss_weight))
            or self.exact_loss_weight <= 0.0
        ):
            raise ValueError("exact_loss_weight must be finite and positive")
        if self.exact_loss_schedule is not None:
            if isinstance(self.exact_loss_schedule, (str, bytes)):
                raise TypeError("exact_loss_schedule must be a numeric sequence")
            schedule = tuple(self.exact_loss_schedule)
            if len(schedule) != self.epochs:
                raise ValueError("exact_loss_schedule length must equal epochs")
            if any(
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in schedule
            ):
                raise ValueError(
                    "exact_loss_schedule values must be finite and non-negative"
                )
            if not any(float(value) > 0.0 for value in schedule):
                raise ValueError("exact_loss_schedule must contain a positive weight")
            object.__setattr__(
                self,
                "exact_loss_schedule",
                tuple(float(value) for value in schedule),
            )

    @property
    def train_pair_count(self) -> int:
        count = int(round(self.max_pairs * (1.0 - self.validation_fraction)))
        count -= count % 2
        return max(2, min(self.max_pairs - 2, count))

    @property
    def validation_pair_count(self) -> int:
        return self.max_pairs - self.train_pair_count

    @property
    def resolved_exact_loss_schedule(self) -> Tuple[float, ...]:
        """Return the explicit schedule or the legacy constant-weight schedule."""

        if self.exact_loss_schedule is not None:
            return self.exact_loss_schedule
        return tuple(float(self.exact_loss_weight) for _ in range(self.epochs))

    @property
    def resolved_initialization_seed(self) -> int:
        """Use the legacy shared seed unless an initialization seed is explicit."""

        if self.initialization_seed is not None:
            return self.initialization_seed
        return self.seed


@dataclass(frozen=True)
class ExactSeamPilotPopulation:
    training_records: Tuple[TrainingPairRecord, ...]
    validation_records: Tuple[TrainingPairRecord, ...]
    discovered_group_count: int
    consumed_group_count: int

    def __post_init__(self) -> None:
        if not self.training_records or not self.validation_records:
            raise ExactSeamPilotError(
                "both train and validation populations are required"
            )
        train_groups = {value.canonical_group_id for value in self.training_records}
        validation_groups = {
            value.canonical_group_id for value in self.validation_records
        }
        if train_groups & validation_groups:
            raise ExactSeamPilotError(
                "a synthetic parent group crossed train/validation"
            )
        for values in (self.training_records, self.validation_records):
            labels = {value.label for value in values}
            if labels != {False, True}:
                raise ExactSeamPilotError("each split must contain both pair classes")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(
        "{}\0{}\0{}".format(seed, namespace, value).encode("utf-8")
    ).hexdigest()


def _direct_pngs(directory: Path) -> Tuple[Path, ...]:
    return tuple(
        sorted(
            (
                value
                for value in directory.iterdir()
                if value.is_file() and value.suffix.casefold() == ".png"
            ),
            key=lambda value: value.name,
        )
    )


def discover_exact_mask_groups(
    mask_root: Path, *, generator: str, seed: int
) -> Tuple[Path, ...]:
    """Find parent groups under ``*/<generator>/no_erode/<group>/*.png``."""

    root = Path(mask_root)
    if not root.is_dir():
        raise ExactSeamPilotError("mask_root must be an unpacked directory")
    no_erode_roots: List[Path]
    if root.name == "no_erode":
        no_erode_roots = [root]
    else:
        no_erode_roots = [value for value in root.rglob("no_erode") if value.is_dir()]
    groups: List[Path] = []
    for no_erode in no_erode_roots:
        family = no_erode.parent.name
        if generator != "all" and family != generator:
            continue
        for candidate in no_erode.iterdir():
            if candidate.is_dir() and len(_direct_pngs(candidate)) >= 2:
                groups.append(candidate)
    if not groups:
        raise ExactSeamPilotError(
            "no scalar mask groups found; point --mask-root at unpacked "
            "shredding_pipeline/output/voronoi_masks"
        )
    groups.sort(
        key=lambda value: (
            _rank(seed, "group-order", value.relative_to(root).as_posix()),
            value.as_posix(),
        )
    )
    return tuple(groups)


def _load_scalar_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in _SCALAR_MASK_MODES:
            raise ExactSeamPilotError(
                "pilot accepts scalar masks only, not RGB/text imagery: {}".format(
                    path.name
                )
            )
        value = np.asarray(image)
    if value.ndim != 2 or value.size == 0:
        raise ExactSeamPilotError("mask must be a non-empty scalar image")
    if value.dtype == np.bool_:
        mask = value
    else:
        if not np.isfinite(value).all():
            raise ExactSeamPilotError("mask contains non-finite pixels")
        threshold = 0.0 if float(np.max(value)) <= 1.0 else 127.0
        mask = value > threshold
    if not mask.any():
        raise ExactSeamPilotError("mask contains no foreground")
    return np.ascontiguousarray(mask, dtype=np.bool_)


def _share_exact_grid_edge(first: np.ndarray, second: np.ndarray) -> bool:
    if first.shape != second.shape:
        raise ExactSeamPilotError("fragment masks in one parent group differ in shape")
    if np.logical_and(first, second).any():
        raise ExactSeamPilotError("aligned fragment masks overlap")
    return bool(
        (first[:, :-1] & second[:, 1:]).any()
        or (second[:, :-1] & first[:, 1:]).any()
        or (first[:-1, :] & second[1:, :]).any()
        or (second[:-1, :] & first[1:, :]).any()
    )


def _centroid_direction(first: np.ndarray, second: np.ndarray) -> str:
    center_a = np.argwhere(first).mean(axis=0)
    center_b = np.argwhere(second).mean(axis=0)
    row_delta, column_delta = center_b - center_a
    if abs(float(column_delta)) >= abs(float(row_delta)):
        return "right" if column_delta >= 0.0 else "left"
    return "below" if row_delta >= 0.0 else "above"


def _split_for_group(relative_group: str, seed: int, validation_fraction: float) -> str:
    value = int(_rank(seed, "group-split", relative_group)[:16], 16) / float(16**16)
    return "val" if value < validation_fraction else "train"


def _mask_ref(
    *, root: Path, path: Path, relative_group: str, split: str
) -> MaskMemberRef:
    member = path.relative_to(root).as_posix()
    fragment_id = "fragment/{}".format(member[:-4])
    group_id = "synthetic-parent/{}".format(relative_group)
    return MaskMemberRef(
        binding=_BINDING,
        archive_member=member,
        fragment_id=fragment_id,
        dataset_id=_DATASET_ID,
        canonical_group_id=group_id,
        component_id=group_id,
        split=split,
        threshold_rule="grayscale_uint8_gt_127",
    )


def _records_for_group(
    root: Path, group: Path, *, split: str, seed: int
) -> Tuple[TrainingPairRecord, ...]:
    paths = _direct_pngs(group)
    masks = {path: _load_scalar_mask(path) for path in paths}
    shapes = {value.shape for value in masks.values()}
    if len(shapes) != 1:
        raise ExactSeamPilotError("fragment masks in one parent group differ in shape")
    relative_group = group.relative_to(root).as_posix()
    rows: List[TrainingPairRecord] = []
    for first_index, first_path in enumerate(paths):
        for second_path in paths[first_index + 1 :]:
            first = masks[first_path]
            second = masks[second_path]
            label = _share_exact_grid_edge(first, second)
            first_ref = _mask_ref(
                root=root,
                path=first_path,
                relative_group=relative_group,
                split=split,
            )
            second_ref = _mask_ref(
                root=root,
                path=second_path,
                relative_group=relative_group,
                split=split,
            )
            rows.append(
                TrainingPairRecord(
                    fragment_a=first_ref,
                    fragment_b=second_ref,
                    label=label,
                    direction_b_wrt_a=(
                        _centroid_direction(first, second) if label else None
                    ),
                    dataset_id=_DATASET_ID,
                    canonical_group_id=first_ref.canonical_group_id,
                    component_id=first_ref.component_id,
                    split=split,
                    canonical_pair_key=tuple(
                        sorted((first_ref.fragment_id, second_ref.fragment_id))
                    ),
                    label_origin="exact_aligned_mask_4_neighbor_seam",
                    provenance={
                        "source": "shredding_pipeline/no_erode",
                        "mask_only": True,
                        "known_orientation": True,
                        "rotation_augmentation": False,
                        "real_dunhuang_sealed_test": False,
                    },
                )
            )
    rows.sort(key=lambda value: _rank(seed, "pair-order", value.pair_id))
    return tuple(rows)


def build_exact_seam_pilot_population(
    config: ExactSeamPilotConfig,
    *,
    record_filter: Optional[Callable[[TrainingPairRecord], bool]] = None,
    group_record_filter: Optional[
        Callable[[Sequence[TrainingPairRecord]], Sequence[bool]]
    ] = None,
) -> ExactSeamPilotPopulation:
    """Select a balanced, group-disjoint pair population without RGB/text."""

    if not isinstance(config, ExactSeamPilotConfig):
        raise TypeError("config must be ExactSeamPilotConfig")
    if record_filter is not None and group_record_filter is not None:
        raise ValueError("record_filter and group_record_filter are mutually exclusive")
    groups = discover_exact_mask_groups(
        config.mask_root, generator=config.generator, seed=config.seed
    )
    targets = {
        ("train", True): config.train_pair_count // 2,
        ("train", False): config.train_pair_count // 2,
        ("val", True): config.validation_pair_count // 2,
        ("val", False): config.validation_pair_count // 2,
    }
    buckets: Dict[Tuple[str, bool], List[TrainingPairRecord]] = {
        key: [] for key in targets
    }
    consumed = 0
    for group in groups:
        relative = group.relative_to(config.mask_root).as_posix()
        split = _split_for_group(relative, config.seed, config.validation_fraction)
        rows = _records_for_group(
            config.mask_root, group, split=split, seed=config.seed
        )
        consumed += 1
        if group_record_filter is None:
            eligibility = None
        else:
            eligibility = tuple(group_record_filter(rows))
            if len(eligibility) != len(rows):
                raise ExactSeamPilotError(
                    "group_record_filter must return one decision per record"
                )
        for row_index, row in enumerate(rows):
            bucket = buckets[(split, row.label)]
            if len(bucket) < targets[(split, row.label)] and (
                (
                    record_filter is None
                    and (eligibility is None or bool(eligibility[row_index]))
                )
                or (record_filter is not None and record_filter(row))
            ):
                bucket.append(row)
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
        raise ExactSeamPilotError(
            "not enough exact positive/negative pairs for the requested pilot: {}".format(
                shortages
            )
        )

    def combined(split: str) -> Tuple[TrainingPairRecord, ...]:
        values = buckets[(split, True)] + buckets[(split, False)]
        values.sort(
            key=lambda value: _rank(config.seed, split + "-final", value.pair_id)
        )
        return tuple(values)

    return ExactSeamPilotPopulation(
        training_records=combined("train"),
        validation_records=combined("val"),
        discovered_group_count=len(groups),
        consumed_group_count=consumed,
    )


def build_or_load_exact_seam_pilot_population(
    config: ExactSeamPilotConfig,
    *,
    group_record_filter: Callable[[Sequence[TrainingPairRecord]], Sequence[bool]],
) -> ExactSeamPilotPopulation:
    """Build once or replay an ordered population without geometry eligibility.

    The default path remains the historical in-memory builder.  When
    ``population_snapshot`` is supplied, the first caller performs the same
    eligibility-filtered build and atomically stores its ordered pair IDs;
    later Exact or matched-Siamese callers reconstruct those records without
    invoking the expensive eligibility predicate again.  More than one
    ``population_workers`` parallelizes only that first concrete exact-seam
    eligibility build.  It never changes the historical no-snapshot path and
    never creates a process pool for snapshot replay.
    """

    if not isinstance(config, ExactSeamPilotConfig):
        raise TypeError("config must be ExactSeamPilotConfig")
    if not callable(group_record_filter):
        raise TypeError("group_record_filter must be callable")
    snapshot_path = config.population_snapshot
    if snapshot_path is None:
        return build_exact_seam_pilot_population(
            config,
            group_record_filter=group_record_filter,
        )

    # Local import avoids a module cycle: the snapshot codec deliberately
    # reconstructs records through this module's deterministic mask adapter.
    from staging.pairwise_v0_2.training import exact_seam_population_snapshot

    if snapshot_path.is_file():
        return exact_seam_population_snapshot.load_exact_seam_population_snapshot(
            snapshot_path,
            config,
        )
    if snapshot_path.exists():
        raise ExactSeamPilotError("population_snapshot must be a file path")
    if config.population_workers == 1:
        population = build_exact_seam_pilot_population(
            config,
            group_record_filter=group_record_filter,
        )
    else:
        population = (
            exact_seam_population_snapshot.build_exact_seam_pilot_population_parallel(
                config,
                workers=config.population_workers,
            )
        )
    exact_seam_population_snapshot.write_exact_seam_population_snapshot(
        config,
        population,
        snapshot_path,
    )
    return population


class _DirectoryMaskLoader:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def __call__(self, reference: MaskMemberRef) -> np.ndarray:
        path = (self.root / reference.archive_member).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ExactSeamPilotError("mask reference escaped mask_root") from exc
        return _load_scalar_mask(path)


def exact_seam_group_eligibility(
    records: Sequence[TrainingPairRecord],
    *,
    loader: _DirectoryMaskLoader,
    cache: GeometryArtifactCache,
    geometry_config: GeometryBatchConfig,
) -> Tuple[bool, ...]:
    """Return the legacy single-record eligibility decision for every row.

    Healthy rows from one synthetic parent group are prepared in one geometry
    batch. A batch exception is bisected until the offending individual row
    is isolated, so one malformed pair cannot reject otherwise eligible rows.
    This eligibility-only path deliberately skips prepared-tensor digests.
    """

    values = tuple(records)
    if not values:
        return ()
    if any(not isinstance(value, TrainingPairRecord) for value in values):
        raise TypeError("every eligibility record must be TrainingPairRecord")
    decisions = [False] * len(values)

    def evaluate(indices: Tuple[int, ...]) -> None:
        selected = tuple(values[index] for index in indices)
        try:
            batch = build_geometry_batch(
                selected,
                loader,
                geometry_config,
                geometry_artifact_cache=cache,
                candidate_representation=KEYPOINT_REPRESENTATION,
                keypoint_config=ContourKeypointConfig(),
                exact_seam_supervision=True,
            )
            targets = batch.exact_loss_targets()
            if targets is None:
                raise ExactSeamPilotError(
                    "exact target construction returned no targets"
                )
            if any(name.startswith("exact_") for name in batch.model_inputs()):
                raise ExactSeamPilotError("exact targets escaped into model inputs")
        except (ExactSeamPilotError, GeometryBatchError):
            if len(indices) > 1:
                midpoint = len(indices) // 2
                evaluate(indices[:midpoint])
                evaluate(indices[midpoint:])
            return

        for local_index, source_index in enumerate(indices):
            eligible = bool(batch.geometry_valid[local_index].item())
            if eligible and bool(batch.labels[local_index].item()):
                candidate_rows = targets.sample_index == local_index
                eligible = bool(
                    (targets.assignment_target_a[candidate_rows] >= 0).any().item()
                )
            decisions[source_index] = eligible

    evaluate(tuple(range(len(values))))
    return tuple(decisions)


def _prepared_batch(
    records: Sequence[TrainingPairRecord],
    *,
    loader: _DirectoryMaskLoader,
    cache: GeometryArtifactCache,
    geometry_config: GeometryBatchConfig,
) -> PreparedAblationBatch:
    values = tuple(records)
    batch = build_geometry_batch(
        values,
        loader,
        geometry_config,
        geometry_artifact_cache=cache,
        candidate_representation=KEYPOINT_REPRESENTATION,
        keypoint_config=ContourKeypointConfig(),
        exact_seam_supervision=True,
    )
    if batch.exact_loss_targets() is None:
        raise ExactSeamPilotError("exact target construction returned no targets")
    if not batch.geometry_valid.all().item():
        raise ExactSeamPilotError(
            "a selected pilot pair did not emit all required local geometry"
        )
    targets = batch.exact_loss_targets()
    assert targets is not None
    for sample_index in torch.nonzero(batch.labels, as_tuple=False).flatten():
        candidate_rows = targets.sample_index == int(sample_index)
        if not (targets.assignment_target_a[candidate_rows] >= 0).any().item():
            raise ExactSeamPilotError(
                "a positive pilot pair produced no keypoint-to-seam match target"
            )
    if any(name.startswith("exact_") for name in batch.model_inputs()):
        raise ExactSeamPilotError("exact targets escaped into model inputs")
    prepared_sha, local_sha = local_q1_prepared_digests(batch)
    coarse_sha = hashlib.sha256(
        _canonical_bytes(
            {
                "mode": geometry_config.coarse_preprocess_mode,
                "output_size": geometry_config.coarse_output_size,
                "content_fraction": geometry_config.coarse_content_fraction,
            }
        )
    ).hexdigest()
    return PreparedAblationBatch(
        payload=batch,
        sample_count=len(values),
        record_sequence_sha256=record_sequence_fingerprint(values),
        prepared_input_sha256=prepared_sha,
        local_candidate_sha256=local_sha,
        coarse_preprocessing_sha256=coarse_sha,
        geometry_config_sha256=geometry_config.fingerprint,
        processing_counts={
            "mask_load_count": 2 * len(values),
            "coarse_preprocess_count": 2 * len(values),
            "geometry_build_count": 0,
            "geometry_cache_read_count": len(
                {
                    reference.fragment_id
                    for value in values
                    for reference in (value.fragment_a, value.fragment_b)
                }
            ),
            "geometry_cache_write_count": 0,
            "local_candidate_count": batch.candidate_count,
        },
        candidate_representation=KEYPOINT_REPRESENTATION,
    )


def _arm(backend: LocalQ1Backend, name: AblationArmName) -> AblationArm:
    if name not in {
        AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
        AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM,
    }:
        raise ValueError("exact-seam pilot accepts only its matched two arms")
    return AblationArm(
        name=name,
        evidence=EvidenceMode.LOCAL,
        matcher_mode=MatcherMode.DUSTBIN_SINKHORN.value,
        model_config=backend.model_config_for(name),
        optimizer_config=backend.optimizer_config,
        aggregation_config=backend.aggregation_config,
        arc_pooling=backend.model_template.arc_pooling,
    )


def _chunks(
    values: Sequence[TrainingPairRecord], size: int
) -> Iterable[Tuple[TrainingPairRecord, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def _epoch_order(
    records: Sequence[TrainingPairRecord], *, seed: int, epoch: int
) -> Tuple[TrainingPairRecord, ...]:
    return tuple(
        sorted(
            records,
            key=lambda value: _rank(seed + epoch, "train-epoch", value.pair_id),
        )
    )


def _select_arm_winner(
    epochs: Sequence[Mapping[str, object]], name: str
) -> Mapping[str, object]:
    """Select on synthetic validation only, with deterministic early tie-break."""

    if name not in {"weak", "exact"} or not epochs:
        raise ValueError("winner selection requires a known arm and epochs")

    def rank(row: Mapping[str, object]) -> Tuple[float, float, int]:
        validation = row["validation"]
        if not isinstance(validation, Mapping):
            raise TypeError("epoch validation must be a mapping")
        metrics = validation[name]
        if not isinstance(metrics, Mapping):
            raise TypeError("arm validation metrics must be a mapping")
        return (
            float(metrics["auroc"]),
            float(metrics["auprc"]),
            -int(row["epoch"]),
        )

    return max(epochs, key=rank)


def _validation_metrics(
    *,
    records: Sequence[TrainingPairRecord],
    sessions: Mapping[str, object],
    loader: _DirectoryMaskLoader,
    cache: GeometryArtifactCache,
    geometry_config: GeometryBatchConfig,
    batch_size: int,
) -> Mapping[str, Mapping[str, float]]:
    probabilities: Dict[str, List[torch.Tensor]] = {name: [] for name in sessions}
    validities: Dict[str, List[torch.Tensor]] = {name: [] for name in sessions}
    labels: List[torch.Tensor] = []
    for rows in _chunks(records, batch_size):
        prepared = _prepared_batch(
            rows,
            loader=loader,
            cache=cache,
            geometry_config=geometry_config,
        )
        labels.append(prepared.payload.labels.detach().cpu())
        for name, session in sessions.items():
            predicted = session.predict_batch(prepared, evidence=EvidenceMode.LOCAL)
            probabilities[name].append(predicted.probability.detach().cpu())
            validities[name].append(predicted.valid.detach().cpu())
    label = torch.cat(labels)
    return {
        name: binary_metrics(
            torch.cat(probabilities[name]),
            label,
            torch.cat(validities[name]),
        )
        for name in sessions
    }


def run_exact_seam_pilot(config: ExactSeamPilotConfig) -> Mapping[str, object]:
    """Train the matched weak/exact arms and write weights plus AUROC/AUPRC."""

    if not isinstance(config, ExactSeamPilotConfig):
        raise TypeError("config must be ExactSeamPilotConfig")
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ExactSeamPilotError("the training pilot must run on an available GPU")
    config.output_root.mkdir(parents=True, exist_ok=True)
    cache = GeometryArtifactCache(config.cache_root)
    loader = _DirectoryMaskLoader(config.mask_root)
    geometry_config = GeometryBatchConfig()

    # Irregular Voronoi fragments do not all expose usable tokens on every
    # upright side.  Filter before freezing the balanced population so both
    # arms receive exactly the same all-four-direction examples and positive
    # examples are guaranteed to carry at least one exact seam match.
    def group_records_are_eligible(
        records: Sequence[TrainingPairRecord],
    ) -> Tuple[bool, ...]:
        return exact_seam_group_eligibility(
            records,
            loader=loader,
            cache=cache,
            geometry_config=geometry_config,
        )

    snapshot_replay = bool(
        config.population_snapshot is not None and config.population_snapshot.is_file()
    )
    population = build_or_load_exact_seam_pilot_population(
        config,
        group_record_filter=group_records_are_eligible,
    )
    backend = LocalQ1Backend(
        device=device,
        mode=LocalQ1BackendMode.FORMAL,
        exact_seam_step_config=ExactSeamStepConfig(
            loss_weight=float(config.exact_loss_weight)
        ),
    )
    arm_names = {
        "weak": AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
        "exact": AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM,
    }
    arms = {name: _arm(backend, value) for name, value in arm_names.items()}
    initialization_seed = config.resolved_initialization_seed
    sessions = {
        name: backend.create_session(arm, seed=initialization_seed)
        for name, arm in arms.items()
    }
    initial_states = {
        session.initial_model_state_sha256 for session in sessions.values()
    }
    if len(initial_states) != 1:
        raise ExactSeamPilotError("weak/exact arms did not share initialization")

    exact_schedule = config.resolved_exact_loss_schedule
    optimizer_objects = {name: session.optimizer for name, session in sessions.items()}
    epochs: List[Mapping[str, object]] = []
    epoch_checkpoint_receipts: Dict[str, List[Mapping[str, object]]] = {
        name: [] for name in sessions
    }
    for epoch in range(1, config.epochs + 1):
        active_weights = {
            "weak": 0.0,
            "exact": exact_schedule[epoch - 1],
        }
        for name, session in sessions.items():
            session.set_exact_assignment_loss_weight(active_weights[name])
            if session.optimizer is not optimizer_objects[name]:
                raise ExactSeamPilotError(
                    "loss scheduling rebuilt an optimizer for " + name
                )
        ordered = _epoch_order(
            population.training_records, seed=initialization_seed, epoch=epoch
        )
        loss_sum = {name: 0.0 for name in sessions}
        exact_loss_sum = {name: 0.0 for name in sessions}
        exact_match_count = {name: 0 for name in sessions}
        exact_dustbin_a_count = {name: 0 for name in sessions}
        exact_dustbin_b_count = {name: 0 for name in sessions}
        sample_sum = {name: 0 for name in sessions}
        valid_sum = {name: 0 for name in sessions}
        for rows in _chunks(ordered, config.batch_size):
            prepared = _prepared_batch(
                rows,
                loader=loader,
                cache=cache,
                geometry_config=geometry_config,
            )
            # The same object is consumed by both arms.  Only the exact session
            # reads its supervision-only target fields.
            for name, session in sessions.items():
                trained = session.train_batch(prepared)
                loss_sum[name] += trained.loss * len(rows)
                exact_loss_sum[name] += float(
                    trained.diagnostics["exact_assignment_loss"]
                ) * len(rows)
                exact_match_count[name] += int(
                    trained.diagnostics["exact_supervised_match_count"]
                )
                exact_dustbin_a_count[name] += int(
                    trained.diagnostics["exact_supervised_dustbin_a_count"]
                )
                exact_dustbin_b_count[name] += int(
                    trained.diagnostics["exact_supervised_dustbin_b_count"]
                )
                sample_sum[name] += len(rows)
                valid_sum[name] += trained.valid_count
        metrics = _validation_metrics(
            records=population.validation_records,
            sessions=sessions,
            loader=loader,
            cache=cache,
            geometry_config=geometry_config,
            batch_size=config.batch_size,
        )
        epoch_row: Dict[str, object] = {
            "epoch": epoch,
            "train": {
                name: {
                    "mean_total_loss": loss_sum[name] / sample_sum[name],
                    "mean_exact_assignment_loss": (
                        exact_loss_sum[name] / sample_sum[name]
                    ),
                    "exact_assignment_loss_weight": active_weights[name],
                    "valid_count": valid_sum[name],
                    "sample_count": sample_sum[name],
                    "exact_supervised_match_count": exact_match_count[name],
                    "exact_supervised_dustbin_a_count": exact_dustbin_a_count[name],
                    "exact_supervised_dustbin_b_count": exact_dustbin_b_count[name],
                }
                for name in sessions
            },
            "validation": metrics,
            "exact_minus_weak": {
                "auroc": metrics["exact"]["auroc"] - metrics["weak"]["auroc"],
                "auprc": metrics["exact"]["auprc"] - metrics["weak"]["auprc"],
            },
        }
        row_checkpoints: Dict[str, Mapping[str, object]] = {}
        for name, session in sessions.items():
            receipt = save_checkpoint(
                config.output_root
                / (name + "_keypoint_sinkhorn_epoch_{:03d}.pt".format(epoch)),
                session.model,
                config={
                    "pilot_version": EXACT_SEAM_PILOT_VERSION,
                    "arm": dict(arms[name].model_config),
                },
                epoch=epoch,
                optimizer=session.optimizer,
                metrics=metrics[name],
                provenance={
                    "mask_only": True,
                    "known_orientation": True,
                    "rotation_augmentation": False,
                    "rgb_or_text_used": False,
                    "synthetic_validation_only": True,
                    "real_evaluation_accessed": False,
                    "continuous_optimizer_across_epochs": True,
                    "exact_assignment_loss_weight": active_weights[name],
                    "initial_model_state_sha256": (session.initial_model_state_sha256),
                },
            )
            portable_receipt = asdict(receipt)
            row_checkpoints[name] = portable_receipt
            epoch_checkpoint_receipts[name].append(portable_receipt)
        epoch_row["checkpoints"] = row_checkpoints
        epochs.append(epoch_row)

    winners: Dict[str, Mapping[str, object]] = {}
    checkpoint_receipts: Dict[str, Mapping[str, object]] = {}
    for name in sessions:
        winner_row = _select_arm_winner(epochs, name)
        row_checkpoints = winner_row["checkpoints"]
        validation = winner_row["validation"]
        if not isinstance(row_checkpoints, Mapping) or not isinstance(
            validation, Mapping
        ):
            raise ExactSeamPilotError("winner epoch is missing checkpoint metrics")
        receipt = row_checkpoints[name]
        metrics = validation[name]
        if not isinstance(receipt, Mapping) or not isinstance(metrics, Mapping):
            raise ExactSeamPilotError("winner arm receipt or metrics are malformed")
        checkpoint_receipts[name] = receipt
        winners[name] = {
            "epoch": int(winner_row["epoch"]),
            "validation": dict(metrics),
            "checkpoint": dict(receipt),
        }

    result: Dict[str, object] = {
        "pilot_version": EXACT_SEAM_PILOT_VERSION,
        "status": "complete",
        "comparison": (
            "matched_keypoint_dustbin_sinkhorn_weak_vs_exact_partial_assignment"
        ),
        "config": {
            "max_pairs": config.max_pairs,
            "train_pairs": config.train_pair_count,
            "validation_pairs": config.validation_pair_count,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "initialization_seed": initialization_seed,
            "initialization_seed_source": (
                "explicit"
                if config.initialization_seed is not None
                else "legacy_seed_fallback"
            ),
            "generator": config.generator,
            "device": str(device),
            "exact_loss_weight": config.exact_loss_weight,
            "exact_loss_schedule": list(exact_schedule),
            "weak_loss_schedule": [0.0] * config.epochs,
            "schedule_source": (
                "explicit"
                if config.exact_loss_schedule is not None
                else "legacy_constant_exact_loss_weight"
            ),
        },
        "population": {
            "discovered_group_count": population.discovered_group_count,
            "consumed_group_count": population.consumed_group_count,
            "group_disjoint": True,
            "train_positive_count": sum(
                value.label for value in population.training_records
            ),
            "train_negative_count": sum(
                not value.label for value in population.training_records
            ),
            "validation_positive_count": sum(
                value.label for value in population.validation_records
            ),
            "validation_negative_count": sum(
                not value.label for value in population.validation_records
            ),
            **(
                {
                    "resolution": (
                        "snapshot_replay" if snapshot_replay else "snapshot_first_build"
                    ),
                    "workers_requested": config.population_workers,
                    "workers_used": (
                        0 if snapshot_replay else config.population_workers
                    ),
                }
                if config.population_snapshot is not None
                else {}
            ),
        },
        "shared_initial_model_state_sha256": next(iter(initial_states)),
        "same_prepared_batch_per_arm": True,
        "continuous_session_and_optimizer_per_arm": True,
        "targets_in_model_inputs": False,
        "winner_selection": {
            "policy": _WINNER_SELECTION_POLICY,
            "population": "balanced_synthetic_validation_only",
            "real_evaluation_accessed": False,
            "winners": winners,
        },
        "epochs": epochs,
        "epoch_checkpoints": epoch_checkpoint_receipts,
        # ``checkpoints`` remains the compact consumer-facing mapping, but now
        # points to each arm's selected synthetic-validation winner.
        "checkpoints": checkpoint_receipts,
    }
    summary_path = config.output_root / "exact_seam_pilot_summary.json"
    summary_path.write_bytes(_canonical_bytes(result) + b"\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-root",
        type=Path,
        required=True,
        help="unpacked shredding_pipeline/output/voronoi_masks directory",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--population-snapshot",
        type=Path,
        help=(
            "optional ordered population JSON; build it on first use and "
            "reuse it to skip geometry eligibility in matched runs"
        ),
    )
    parser.add_argument(
        "--population-workers",
        type=int,
        default=1,
        help=(
            "deterministic exact-geometry worker count used only when an "
            "explicit --population-snapshot does not exist; replay never "
            "starts a worker pool"
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=1000,
        choices=range(_MIN_PAIR_COUNT, _MAX_PAIR_COUNT + 1, 2),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=260830)
    parser.add_argument(
        "--initialization-seed",
        type=int,
        help=(
            "model-initialization and epoch-shuffle seed; when omitted, "
            "--seed is used exactly as in legacy runs"
        ),
    )
    parser.add_argument("--generator", default="gen4voronoi")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exact-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--exact-loss-schedule",
        type=float,
        nargs="+",
        help=(
            "per-epoch exact weights, e.g. --epochs 5 "
            "--exact-loss-schedule 0.25 0.05 0 0 0; when omitted, "
            "--exact-loss-weight is repeated for every epoch"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    output_root = Path(arguments.output_root)
    config = ExactSeamPilotConfig(
        mask_root=arguments.mask_root,
        output_root=output_root,
        cache_root=arguments.cache_root or output_root / "geometry_cache",
        max_pairs=arguments.max_pairs,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
        initialization_seed=arguments.initialization_seed,
        generator=arguments.generator,
        device=arguments.device,
        exact_loss_weight=arguments.exact_loss_weight,
        exact_loss_schedule=(
            None
            if arguments.exact_loss_schedule is None
            else tuple(arguments.exact_loss_schedule)
        ),
        population_snapshot=arguments.population_snapshot,
        population_workers=arguments.population_workers,
    )
    result = run_exact_seam_pilot(config)
    print(json.dumps(result["epochs"][-1], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the remote CLI
    raise SystemExit(main())


__all__ = [
    "EXACT_SEAM_PILOT_VERSION",
    "ExactSeamPilotConfig",
    "ExactSeamPilotError",
    "ExactSeamPilotPopulation",
    "build_or_load_exact_seam_pilot_population",
    "build_exact_seam_pilot_population",
    "discover_exact_mask_groups",
    "exact_seam_group_eligibility",
    "main",
    "run_exact_seam_pilot",
]
