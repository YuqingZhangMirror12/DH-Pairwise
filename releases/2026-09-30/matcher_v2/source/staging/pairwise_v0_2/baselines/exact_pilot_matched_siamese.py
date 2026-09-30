#!/usr/bin/env python3
"""Whole-mask MobileNetV2 Siamese control for the exact-seam GPU pilot.

The control deliberately reuses the exact-seam pilot's deterministic gen4
population builder, geometry-eligibility filter, train/validation split, and
per-epoch pair order.  The only research-facing change is the model input and
architecture: each fragment is a complete 64 x 64 single-channel mask passed
through the historical ordered MobileNetV2 Siamese network.

No RGB, text/OCR, rotation search, direction input, or real Dunhuang sample is
available to this training runner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from staging.pairwise_v0_2.baselines import matched_route_a_siamese as siamese
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training import exact_seam_pilot as exact_pilot
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.geometry_cache import GeometryArtifactCache
from staging.pairwise_v0_2.training.metrics import binary_metrics
from staging.pairwise_v0_2.training.short_ablation import (
    record_sequence_fingerprint,
)


EXACT_PILOT_MATCHED_SIAMESE_VERSION = "dunhuang-exact-pilot-matched-mm-siamese/0.1"
_WINNER_POLICY = "synthetic_validation_auroc_then_auprc_then_earlier_epoch"


class ExactPilotMatchedSiameseError(RuntimeError):
    """The exact-population whole-mask control cannot be executed."""


@dataclass(frozen=True)
class ExactPilotMatchedSiameseConfig:
    """CLI configuration kept numerically aligned with ``ExactSeamPilotConfig``."""

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
    learning_rate: float = 1.0
    scheduler_gamma: float = 0.7
    preprocess_workers: int = 8
    population_snapshot: Optional[Path] = None
    initialization_seed: Optional[int] = None
    population_workers: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask_root", Path(self.mask_root))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "cache_root", Path(self.cache_root))
        if self.population_snapshot is not None:
            object.__setattr__(
                self, "population_snapshot", Path(self.population_snapshot)
            )
        if self.initialization_seed is not None and (
            isinstance(self.initialization_seed, bool)
            or not isinstance(self.initialization_seed, int)
            or self.initialization_seed < 0
        ):
            raise ValueError("initialization_seed must be a non-negative integer")
        # Constructing both shared contracts is the single source of validation
        # for population cardinalities and historical training hyperparameters.
        self.population_config
        self.training_contract

    @property
    def population_config(self) -> exact_pilot.ExactSeamPilotConfig:
        """Return the exact config consumed by the shared population builder."""

        return exact_pilot.ExactSeamPilotConfig(
            mask_root=self.mask_root,
            output_root=self.output_root,
            cache_root=self.cache_root,
            max_pairs=self.max_pairs,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_fraction=self.validation_fraction,
            seed=self.seed,
            generator=self.generator,
            device=self.device,
            population_snapshot=self.population_snapshot,
            population_workers=self.population_workers,
        )

    @property
    def training_contract(self) -> siamese.MatchedSiameseContract:
        """Return the historical recipe with exact-pilot exposure settings."""

        return siamese.MatchedSiameseContract(
            epochs=self.epochs,
            seed=self.resolved_initialization_seed,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            scheduler_gamma=self.scheduler_gamma,
            preprocess_workers=self.preprocess_workers,
        )

    @property
    def resolved_initialization_seed(self) -> int:
        """Use the historical shared seed unless explicitly separated."""

        if self.initialization_seed is not None:
            return self.initialization_seed
        return self.seed

    @property
    def initialization_seed_source(self) -> str:
        return (
            "explicit"
            if self.initialization_seed is not None
            else "legacy_seed_fallback"
        )

    @property
    def train_pair_count(self) -> int:
        return self.population_config.train_pair_count

    @property
    def validation_pair_count(self) -> int:
        return self.population_config.validation_pair_count


def build_exact_pilot_matched_siamese_population(
    config: ExactPilotMatchedSiameseConfig,
) -> exact_pilot.ExactSeamPilotPopulation:
    """Resolve exactly the population accepted by the keypoint pilot.

    Without a snapshot, the eligibility call is intentionally the same call
    used by ``run_exact_seam_pilot``.  With a shared snapshot, ordered records
    are reconstructed directly, preventing both repeated eligibility work and
    population drift between the local model and Siamese control.
    """

    if not isinstance(config, ExactPilotMatchedSiameseConfig):
        raise TypeError("config must be ExactPilotMatchedSiameseConfig")
    population_config = config.population_config
    loader = exact_pilot._DirectoryMaskLoader(config.mask_root)
    cache = GeometryArtifactCache(config.cache_root)
    geometry_config = GeometryBatchConfig()

    def group_records_are_eligible(
        records: Sequence[TrainingPairRecord],
    ) -> tuple[bool, ...]:
        return exact_pilot.exact_seam_group_eligibility(
            records,
            loader=loader,
            cache=cache,
            geometry_config=geometry_config,
        )

    return exact_pilot.build_or_load_exact_seam_pilot_population(
        population_config,
        group_record_filter=group_records_are_eligible,
    )


def _validation_metrics(
    model: nn.Module,
    records: Sequence[TrainingPairRecord],
    tensors: Mapping[tuple[str, ...], Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[Mapping[str, float], Tensor]:
    dataset = siamese.PreprocessedMaskPairDataset(records, tensors)
    probability = siamese._predict(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
    )
    label = torch.tensor([record.label for record in records], dtype=torch.bool)
    metrics = binary_metrics(
        probability,
        label,
        torch.ones_like(label, dtype=torch.bool),
    )
    return MappingProxyType(metrics), probability


def _ensure_fresh_outputs(output_root: Path) -> None:
    names = (
        "exact_pilot_matched_siamese_summary.json",
        "exact_pilot_matched_siamese_progress.json",
        "exact_pilot_matched_siamese_validation_scores.json",
        "exact_pilot_matched_siamese_winner.pt",
    )
    if any((output_root / name).exists() for name in names) or any(
        output_root.glob("exact_pilot_matched_siamese_epoch_*.pt")
    ):
        raise ExactPilotMatchedSiameseError(
            "matched-Siamese output artifacts already exist"
        )


def run_exact_population_matched_siamese_training(
    *,
    population: exact_pilot.ExactSeamPilotPopulation,
    loader_factory: Callable[[], Callable[[MaskMemberRef], np.ndarray]],
    output_root: Path,
    contract: siamese.MatchedSiameseContract,
    device: torch.device,
    model_factory: Optional[Callable[[], nn.Module]] = None,
    population_seed: Optional[int] = None,
    initialization_seed_source: str = "contract_seed",
    population_resolution: Optional[str] = None,
    population_workers_requested: Optional[int] = None,
    population_workers_used: Optional[int] = None,
) -> Mapping[str, Any]:
    """Train one historical whole-mask Siamese arm on an exact population."""

    if not isinstance(population, exact_pilot.ExactSeamPilotPopulation):
        raise TypeError("population must be ExactSeamPilotPopulation")
    if not isinstance(contract, siamese.MatchedSiameseContract):
        raise TypeError("contract must be MatchedSiameseContract")
    if population_seed is None:
        population_seed = contract.seed
    if (
        isinstance(population_seed, bool)
        or not isinstance(population_seed, int)
        or population_seed < 0
    ):
        raise ValueError("population_seed must be a non-negative integer")
    if (
        not isinstance(initialization_seed_source, str)
        or not initialization_seed_source
    ):
        raise ValueError("initialization_seed_source must be a non-empty string")
    if population_resolution is not None and (
        not isinstance(population_resolution, str) or not population_resolution
    ):
        raise ValueError("population_resolution must be a non-empty string")
    if population_resolution is None and (
        population_workers_requested is not None or population_workers_used is not None
    ):
        raise ValueError("population worker metadata requires a resolution")
    if population_resolution is not None and (
        population_workers_requested is None or population_workers_used is None
    ):
        raise ValueError("population resolution requires worker metadata")
    for name, value in (
        ("population_workers_requested", population_workers_requested),
        ("population_workers_used", population_workers_used),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError("{} must be a non-negative integer".format(name))
    output_directory = Path(output_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    _ensure_fresh_outputs(output_directory)

    siamese._seed_execution(contract.seed, device)
    if model_factory is None:
        model = siamese.make_random_historical_mm_model(contract.seed, device=device)
        model_semantics = "HistoricalMMSiamese_random_init_old_linear_xavier_bias_0.01"
    else:
        model = model_factory().to(device)
        model_semantics = "injected_test_model"
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return torch.nn.Module")

    tensors = siamese.preload_mask_tensors(
        population.training_records + population.validation_records,
        loader_factory,
        workers=contract.preprocess_workers,
    )
    criterion = siamese.HistoricalProbabilityFocalLoss(
        alpha=contract.focal_alpha,
        gamma=contract.focal_gamma,
    )
    optimizer = torch.optim.Adadelta(model.parameters(), lr=contract.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=contract.scheduler_gamma,
    )

    epochs: List[Mapping[str, Any]] = []
    best_key: Optional[tuple[float, float, int]] = None
    best_epoch: Optional[int] = None
    best_state: Optional[Dict[str, Tensor]] = None
    for epoch in range(1, contract.epochs + 1):
        # This is the exact pilot's deterministic pair order, not an independent
        # DataLoader shuffle.  Thus its batches contain the same pair IDs in the
        # same order and expose every training pair exactly once per epoch.
        ordered = exact_pilot._epoch_order(
            population.training_records,
            seed=contract.seed,
            epoch=epoch,
        )
        train_dataset = siamese.PreprocessedMaskPairDataset(ordered, tensors)
        train_loader = DataLoader(
            train_dataset,
            batch_size=contract.train_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        model.train()
        loss_sum = 0.0
        presented = 0
        optimizer_steps = 0
        for input_a, input_b, target in train_loader:
            input_a = input_a.to(device, non_blocking=device.type == "cuda")
            input_b = input_b.to(device, non_blocking=device.type == "cuda")
            target = target.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            probability = model(input_a, input_b)
            if tuple(probability.shape) != (len(target), 1):
                raise ExactPilotMatchedSiameseError(
                    "Siamese model output must be [B,1]"
                )
            loss = criterion(probability[:, 0], target)
            if not torch.isfinite(loss).item():
                raise ExactPilotMatchedSiameseError("training loss is non-finite")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(target)
            presented += len(target)
            optimizer_steps += 1
        if presented != len(population.training_records):
            raise ExactPilotMatchedSiameseError(
                "one epoch did not present every train pair exactly once"
            )

        validation, _ = _validation_metrics(
            model,
            population.validation_records,
            tensors,
            device=device,
            batch_size=contract.eval_batch_size,
        )
        state = siamese._cpu_state_dict(model)
        checkpoint_path = output_directory / (
            "exact_pilot_matched_siamese_epoch_{:03d}.pt".format(epoch)
        )
        siamese._save_raw_state_dict(checkpoint_path, state)
        key = (float(validation["auroc"]), float(validation["auprc"]), -epoch)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = state
        epoch_row = {
            "epoch": epoch,
            "train": {
                "mean_loss": loss_sum / presented,
                "sample_count": presented,
                "optimizer_steps": optimizer_steps,
                "learning_rate_used": float(scheduler.get_last_lr()[0]),
                "pair_sequence_sha256": record_sequence_fingerprint(ordered),
            },
            "validation": dict(validation),
            "checkpoint": {
                "path": checkpoint_path.name,
                "file_sha256": siamese._sha256(checkpoint_path.read_bytes()),
                "format": "tensor_only_raw_state_dict",
            },
        }
        epochs.append(epoch_row)
        siamese._write_json(
            output_directory / "exact_pilot_matched_siamese_progress.json",
            {
                "version": EXACT_PILOT_MATCHED_SIAMESE_VERSION,
                "status": "training",
                "completed_epochs": epoch,
                "total_epochs": contract.epochs,
                "current_winner_epoch": best_epoch,
                "epochs": epochs,
            },
        )
        print(siamese._canonical_json(epoch_row).decode("utf-8"), flush=True)
        scheduler.step()

    if best_state is None or best_epoch is None:
        raise ExactPilotMatchedSiameseError("winner selection failed")
    winner_path = output_directory / "exact_pilot_matched_siamese_winner.pt"
    siamese._save_raw_state_dict(winner_path, best_state)
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    winner_validation, winner_probability = _validation_metrics(
        model,
        population.validation_records,
        tensors,
        device=device,
        batch_size=contract.eval_batch_size,
    )
    siamese._write_json(
        output_directory / "exact_pilot_matched_siamese_validation_scores.json",
        {
            "records": [
                {
                    "pair_id": record.pair_id,
                    "component_id": record.component_id,
                    "label": record.label,
                    "probability": float(probability),
                }
                for record, probability in zip(
                    population.validation_records, winner_probability
                )
            ]
        },
    )

    summary: Dict[str, Any] = {
        "version": EXACT_PILOT_MATCHED_SIAMESE_VERSION,
        "status": "complete",
        "comparison": "exact_pilot_keypoint_models_vs_whole_mask_mm_siamese",
        "scope": {
            "input_modality": "binary_mask_only",
            "input_shape_per_fragment": [1, 64, 64],
            "rgb_used": False,
            "text_or_ocr_used": False,
            "rotation_search_used": False,
            "direction_supervision_used": False,
            "real_dunhuang_used": False,
        },
        "seeds": {
            "population_seed": population_seed,
            "initialization_seed": contract.seed,
            "initialization_seed_source": initialization_seed_source,
        },
        "fairness": {
            "same_population_builder_as_exact_seam_pilot": True,
            "same_geometry_eligibility_filter_as_exact_seam_pilot": True,
            "same_seed_and_group_split_as_exact_seam_pilot": True,
            "same_train_pair_order_as_exact_seam_pilot": True,
            "train_pair_presentations_per_epoch": 1,
            "total_presentations_per_train_pair": contract.epochs,
            "same_optimizer_as_keypoint_models": False,
            "architectural_difference": (
                "whole_64x64_masks_vs_contour_keypoint_patches_and_sinkhorn"
            ),
        },
        "population": {
            "train_pair_count": len(population.training_records),
            "validation_pair_count": len(population.validation_records),
            "train_positive_count": sum(
                record.label for record in population.training_records
            ),
            "train_negative_count": sum(
                not record.label for record in population.training_records
            ),
            "validation_positive_count": sum(
                record.label for record in population.validation_records
            ),
            "validation_negative_count": sum(
                not record.label for record in population.validation_records
            ),
            "train_pair_sequence_sha256": record_sequence_fingerprint(
                population.training_records
            ),
            "validation_pair_sequence_sha256": record_sequence_fingerprint(
                population.validation_records
            ),
            "discovered_group_count": population.discovered_group_count,
            "consumed_group_count": population.consumed_group_count,
            "group_disjoint": True,
            **(
                {
                    "resolution": population_resolution,
                    "workers_requested": population_workers_requested,
                    "workers_used": population_workers_used,
                }
                if population_resolution is not None
                else {}
            ),
        },
        "contract": contract.portable_dict(),
        "model": {
            "semantics": model_semantics,
            "architecture": ("ordered_shared_MobileNetV2_then_concat_256_1_sigmoid"),
            "pretrained_weights": False,
            "preprocessing": "bool_mask_PIL_bilinear_64x64_single_channel",
            "loss": "historical_probability_focal",
            "optimizer": "Adadelta",
        },
        "epochs": epochs,
        "winner": {
            "epoch": best_epoch,
            "selection_policy": _WINNER_POLICY,
            "validation": dict(winner_validation),
            "checkpoint": {
                "path": winner_path.name,
                "file_sha256": siamese._sha256(winner_path.read_bytes()),
                "format": "tensor_only_raw_state_dict",
            },
        },
        "score_artifact": "exact_pilot_matched_siamese_validation_scores.json",
    }
    siamese._write_json(
        output_directory / "exact_pilot_matched_siamese_summary.json", summary
    )
    return MappingProxyType(summary)


def run_exact_pilot_matched_siamese(
    config: ExactPilotMatchedSiameseConfig,
) -> Mapping[str, Any]:
    """Build the matched population and run the GPU whole-mask control."""

    if not isinstance(config, ExactPilotMatchedSiameseConfig):
        raise TypeError("config must be ExactPilotMatchedSiameseConfig")
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ExactPilotMatchedSiameseError(
            "the matched exact-pilot control must run on an available GPU"
        )
    snapshot_replay = bool(
        config.population_snapshot is not None and config.population_snapshot.is_file()
    )
    population = build_exact_pilot_matched_siamese_population(config)
    return run_exact_population_matched_siamese_training(
        population=population,
        loader_factory=lambda: exact_pilot._DirectoryMaskLoader(config.mask_root),
        output_root=config.output_root,
        contract=config.training_contract,
        device=device,
        population_seed=config.seed,
        initialization_seed_source=config.initialization_seed_source,
        population_resolution=(
            ("snapshot_replay" if snapshot_replay else "snapshot_first_build")
            if config.population_snapshot is not None
            else None
        ),
        population_workers_requested=(
            config.population_workers
            if config.population_snapshot is not None
            else None
        ),
        population_workers_used=(
            (0 if snapshot_replay else config.population_workers)
            if config.population_snapshot is not None
            else None
        ),
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--population-snapshot",
        type=Path,
        help=(
            "optional ordered population JSON shared with the Exact runner; "
            "an existing snapshot skips geometry eligibility"
        ),
    )
    parser.add_argument(
        "--population-workers",
        type=_positive_int,
        default=1,
        help=(
            "deterministic exact-geometry worker count used only to create "
            "a missing shared population snapshot"
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=1000,
        choices=range(exact_pilot._MIN_PAIR_COUNT, exact_pilot._MAX_PAIR_COUNT + 1, 2),
    )
    parser.add_argument("--epochs", type=_positive_int, default=3)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=260830)
    parser.add_argument(
        "--initialization-seed",
        type=int,
        help=(
            "model-initialization and epoch-order seed; when omitted, "
            "--seed is used exactly as in legacy runs"
        ),
    )
    parser.add_argument("--generator", default="gen4voronoi")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--scheduler-gamma", type=float, default=0.7)
    parser.add_argument("--preprocess-workers", type=_positive_int, default=8)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = run_exact_pilot_matched_siamese(
        ExactPilotMatchedSiameseConfig(
            mask_root=args.mask_root,
            output_root=args.output_root,
            cache_root=args.cache_root,
            max_pairs=args.max_pairs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            generator=args.generator,
            device=args.device,
            learning_rate=args.learning_rate,
            scheduler_gamma=args.scheduler_gamma,
            preprocess_workers=args.preprocess_workers,
            population_snapshot=args.population_snapshot,
            initialization_seed=args.initialization_seed,
            population_workers=args.population_workers,
        )
    )
    print(siamese._canonical_json(result).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover - remote CLI
    raise SystemExit(main())


__all__ = [
    "EXACT_PILOT_MATCHED_SIAMESE_VERSION",
    "ExactPilotMatchedSiameseConfig",
    "ExactPilotMatchedSiameseError",
    "build_exact_pilot_matched_siamese_population",
    "main",
    "run_exact_pilot_matched_siamese",
    "run_exact_population_matched_siamese_training",
]
