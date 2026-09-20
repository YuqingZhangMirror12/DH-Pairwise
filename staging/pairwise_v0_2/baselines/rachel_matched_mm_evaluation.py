"""Frozen evaluation adapter for the self-contained Rachel matched-MM control.

The functions that restore training artefacts intentionally accept no dataset
or real-image path.  Callers must freeze both the converged validation winner
and the separately thresholded epoch-5 same-exposure checkpoint before they
open a synthetic-test manifest or a real-Dunhuang population.

Inference consumes only binary model masks.  It uses the exact historical-MM
PIL bilinear 64x64 preprocessing implemented by
``rachel_matched_mm_siamese``; contour, correspondence, translation, RGB and
labels are never inputs to the model call.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import random
from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

# This must precede the first torch import in the process.  The training entry
# point has the same invariant and the combined evaluators verify the value.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import Tensor, nn

from staging.pairwise_v0_2.baselines.rachel_matched_mm_siamese import (
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_RECIPE,
    SCHEMA_VERSION as TRAIN_SCHEMA_VERSION,
    HistoricalMMSiamese,
    preprocess_historical_mm_mask,
)
from staging.pairwise_v0_2.training.evaluation import (
    PairwiseThresholdArtifact,
    evaluate_pairwise,
)


CONVERGED_METHOD = "matched_mm_converged"
SAME_EXPOSURE_METHOD = "matched_mm_same_exposure_epoch5"
MATCHED_METHODS = (CONVERGED_METHOD, SAME_EXPOSURE_METHOD)
MATCHED_SCORE = "historical_mm_probability"
PAIR_SCHEMA_VERSION = "rachel-n512-sealed-test-pair/1.0"
EXPECTED_MODEL_CONFIG = {
    "architecture": "HistoricalMMSiamese",
    "recipe": MODEL_RECIPE,
    "input": "bool_mask_PIL_bilinear_64x64_single_channel",
    "ordered_pair": True,
}


class RachelMatchedMMEvaluationError(RuntimeError):
    """A frozen matched-MM provenance or mask-only inference gate failed."""


@dataclass(frozen=True)
class FrozenMatchedMMWinner:
    """One checkpoint/threshold pair frozen before evaluation data are opened."""

    method: str
    epoch: int
    checkpoint_path: Path
    checkpoint_sha256: str
    model_config: Mapping[str, object]
    model_config_sha256: str
    threshold: PairwiseThresholdArtifact
    model: nn.Module


@dataclass(frozen=True)
class MatchedMMPrediction:
    pair_id: str
    probability: float
    valid: bool = True


@dataclass(frozen=True)
class SyntheticMaskPair:
    pair_id: str
    mask_a_path: Path
    mask_b_path: Path


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RachelMatchedMMEvaluationError(
            description + " is not readable JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise RachelMatchedMMEvaluationError(description + " must be a JSON object")
    return value


def _safe_run_member(root: Path, value: object, description: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RachelMatchedMMEvaluationError(description + " path is invalid")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise RachelMatchedMMEvaluationError(description + " path is unsafe")
    current = root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise RachelMatchedMMEvaluationError(description + " may not be a symlink")
    try:
        path = current.resolve(strict=True)
        path.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise RachelMatchedMMEvaluationError(
            description + " is missing or escapes the run"
        ) from error
    if not path.is_file():
        raise RachelMatchedMMEvaluationError(description + " must be a regular file")
    return path


def _safe_torch_load(path: Path) -> Mapping[str, object]:
    version = torch.__version__.split("+", 1)[0].split(".")
    try:
        major = int(version[0])
    except (IndexError, ValueError):
        major = 0
    try:
        # Production PyTorch uses the restricted weights-only unpickler.  The
        # local compatibility runtime is 1.13, where that keyword is absent;
        # bytes are SHA-bound to the finalized receipt before this call.
        value = torch.load(
            path,
            map_location="cpu",
            **({"weights_only": True} if major >= 2 else {}),
        )
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as error:
        raise RachelMatchedMMEvaluationError(
            "cannot safely load frozen matched-MM state"
        ) from error
    if not isinstance(value, Mapping):
        raise RachelMatchedMMEvaluationError("frozen matched-MM state is not a mapping")
    return value


def _require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RachelMatchedMMEvaluationError(description + " is not a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise RachelMatchedMMEvaluationError(description + " is not hexadecimal") from error
    return value.casefold()


def _require_no_evaluation_access(value: Mapping[str, object], description: str) -> None:
    if value.get("test_accessed") is not False:
        raise RachelMatchedMMEvaluationError(description + ".test_accessed must be false")
    if value.get("real_external_test_accessed") is not False:
        raise RachelMatchedMMEvaluationError(
            description + ".real_external_test_accessed must be false"
        )


def _validate_model_config(value: object) -> Tuple[Mapping[str, object], str]:
    if not isinstance(value, Mapping):
        raise RachelMatchedMMEvaluationError("matched-MM model_config is missing")
    config = dict(value)
    if any(config.get(key) != expected for key, expected in EXPECTED_MODEL_CONFIG.items()):
        raise RachelMatchedMMEvaluationError("matched-MM model recipe differs")
    if not isinstance(config.get("optimizer"), Mapping) or not isinstance(
        config.get("scheduler"), Mapping
    ):
        raise RachelMatchedMMEvaluationError("matched-MM training recipe is incomplete")
    return config, _canonical_sha256(config)


def _validate_threshold(
    value: object,
    *,
    checkpoint_sha256: str,
    model_config_sha256: str,
) -> PairwiseThresholdArtifact:
    if not isinstance(value, Mapping):
        raise RachelMatchedMMEvaluationError("matched-MM validation threshold is missing")
    try:
        threshold = PairwiseThresholdArtifact(**dict(value))
    except (TypeError, ValueError) as error:
        raise RachelMatchedMMEvaluationError(
            "matched-MM validation threshold is invalid"
        ) from error
    if (
        threshold.schema_version != "dunhuang-pairwise-threshold/0.2"
        or threshold.source_split.strip().casefold() != "val"
        or threshold.fit_method != "maximize_cluster_balanced_f1"
    ):
        raise RachelMatchedMMEvaluationError(
            "matched-MM threshold is not the declared validation cluster-F1 artifact"
        )
    if threshold.checkpoint_sha256 != checkpoint_sha256:
        raise RachelMatchedMMEvaluationError(
            "matched-MM threshold is bound to different checkpoint bytes"
        )
    if threshold.model_config_sha256 != model_config_sha256:
        raise RachelMatchedMMEvaluationError(
            "matched-MM threshold is bound to a different model config"
        )
    if threshold.aggregation_config_sha256 != _canonical_sha256(
        {"pair_score": MATCHED_SCORE}
    ):
        raise RachelMatchedMMEvaluationError(
            "matched-MM threshold is bound to a different score readout"
        )
    return threshold


def _validate_epoch_row(
    epochs: object,
    *,
    epoch: int,
    checkpoint_name: object,
    checkpoint_sha256: str,
) -> Mapping[str, object]:
    if not isinstance(epochs, list):
        raise RachelMatchedMMEvaluationError("matched-MM epoch history is missing")
    rows = [
        row
        for row in epochs
        if isinstance(row, Mapping) and row.get("epoch") == epoch
    ]
    if (
        len(rows) != 1
        or rows[0].get("checkpoint") != checkpoint_name
        or rows[0].get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise RachelMatchedMMEvaluationError(
            "matched-MM checkpoint differs from frozen epoch history"
        )
    return rows[0]


def _state_dicts_equal(
    first: Mapping[str, object], second: Mapping[str, object]
) -> bool:
    if tuple(first) != tuple(second):
        return False
    for key in first:
        left = first[key]
        right = second[key]
        if not isinstance(left, Tensor) or not isinstance(right, Tensor):
            return False
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(left.detach().cpu(), right.detach().cpu())
        ):
            return False
    return True


def _freeze_one(
    *,
    root: Path,
    receipt: Mapping[str, object],
    method: str,
    epoch: int,
    checkpoint_name: object,
    checkpoint_sha256: str,
    threshold_value: object,
    model_config: Mapping[str, object],
    model_config_sha256: str,
    target_device: torch.device,
) -> FrozenMatchedMMWinner:
    _validate_epoch_row(
        receipt.get("epochs"),
        epoch=epoch,
        checkpoint_name=checkpoint_name,
        checkpoint_sha256=checkpoint_sha256,
    )
    checkpoint_path = _safe_run_member(root, checkpoint_name, method + " checkpoint")
    if _sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RachelMatchedMMEvaluationError(method + " checkpoint SHA-256 differs")
    threshold = _validate_threshold(
        threshold_value,
        checkpoint_sha256=checkpoint_sha256,
        model_config_sha256=model_config_sha256,
    )
    checkpoint = _safe_torch_load(checkpoint_path)
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("epoch") != epoch
        or checkpoint.get("model_config") != model_config
    ):
        raise RachelMatchedMMEvaluationError(method + " checkpoint metadata differs")
    _require_no_evaluation_access(checkpoint, method + " checkpoint")
    if checkpoint.get("run_config") != receipt.get("config"):
        raise RachelMatchedMMEvaluationError(method + " run config differs")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RachelMatchedMMEvaluationError(method + " checkpoint lacks model state")
    model = HistoricalMMSiamese()
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RachelMatchedMMEvaluationError(
            method + " checkpoint cannot be restored strictly"
        ) from error
    model.to(target_device).eval()
    return FrozenMatchedMMWinner(
        method=method,
        epoch=epoch,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        model_config=model_config,
        model_config_sha256=model_config_sha256,
        threshold=threshold,
        model=model,
    )


def freeze_matched_mm_winners(
    run_directory: Path,
    *,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Mapping[str, object], str, Mapping[str, FrozenMatchedMMWinner]]:
    """Freeze converged and epoch-5 controls without accepting a dataset path."""

    try:
        root = Path(run_directory).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RachelMatchedMMEvaluationError("matched-MM run directory is missing") from error
    if not root.is_dir() or root.name.startswith(".partial-"):
        raise RachelMatchedMMEvaluationError("matched-MM run is not finalized")
    receipt_path = root / "run_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RachelMatchedMMEvaluationError(
            "matched-MM receipt must be a regular non-symlink file"
        )
    receipt = _read_json_object(receipt_path, "matched-MM training receipt")
    receipt_sha256 = _sha256_file(receipt_path)
    if (
        receipt.get("schema_version") != TRAIN_SCHEMA_VERSION
        or receipt.get("status") != "complete_train_validation_only"
    ):
        raise RachelMatchedMMEvaluationError("matched-MM run is not complete/frozen")
    _require_no_evaluation_access(receipt, "matched-MM receipt")
    run_config = receipt.get("config")
    if not isinstance(run_config, Mapping):
        raise RachelMatchedMMEvaluationError("matched-MM run config is missing")
    _require_no_evaluation_access(run_config, "matched-MM config")
    if run_config.get("source_splits_opened") != ["train", "val"]:
        raise RachelMatchedMMEvaluationError("matched-MM training split contract differs")
    population = receipt.get("population")
    if not isinstance(population, Mapping) or population != {
        "train_total": 24_000,
        "val_total": 3_000,
        "train_positive": 12_000,
        "val_positive": 1_500,
    }:
        raise RachelMatchedMMEvaluationError(
            "matched-MM population is not the frozen 24000/3000 exact-1:1 control"
        )
    if (
        receipt.get("stop_reason") != "validation_early_stop"
        or receipt.get("convergence_claim")
        != "validation_plateau_under_declared_rule"
    ):
        raise RachelMatchedMMEvaluationError(
            "matched_mm_converged requires a declared validation plateau"
        )
    model_config, model_config_sha256 = _validate_model_config(
        receipt.get("model_config")
    )

    winner_epoch = receipt.get("winner_epoch")
    winner_sha = _require_sha256(
        receipt.get("winner_checkpoint_sha256"), "matched-MM winner checkpoint SHA-256"
    )
    if type(winner_epoch) is not int or winner_epoch <= 0:  # noqa: E721
        raise RachelMatchedMMEvaluationError("matched-MM winner epoch is invalid")
    same_exposure = receipt.get("same_exposure_epoch5")
    if not isinstance(same_exposure, Mapping):
        raise RachelMatchedMMEvaluationError("matched-MM same-exposure epoch-5 is missing")
    _require_no_evaluation_access(same_exposure, "matched-MM same-exposure epoch-5")
    epoch5_sha = _require_sha256(
        same_exposure.get("checkpoint_sha256"), "matched-MM epoch-5 checkpoint SHA-256"
    )
    epoch5_row = _validate_epoch_row(
        receipt.get("epochs"),
        epoch=5,
        checkpoint_name=same_exposure.get("checkpoint"),
        checkpoint_sha256=epoch5_sha,
    )
    if (
        same_exposure.get("selection_metrics") != epoch5_row.get("selection_metrics")
        or same_exposure.get("validation_scores") != epoch5_row.get("validation_scores")
        or same_exposure.get("validation_scores_sha256")
        != epoch5_row.get("validation_scores_sha256")
        or same_exposure.get("selection_and_threshold_source") != "val_only"
    ):
        raise RachelMatchedMMEvaluationError(
            "matched-MM epoch-5 validation provenance differs"
        )
    validation_scores_path = _safe_run_member(
        root,
        same_exposure.get("validation_scores"),
        "matched-MM epoch-5 validation scores",
    )
    if _sha256_file(validation_scores_path) != _require_sha256(
        same_exposure.get("validation_scores_sha256"),
        "matched-MM epoch-5 validation score SHA-256",
    ):
        raise RachelMatchedMMEvaluationError(
            "matched-MM epoch-5 validation score SHA-256 differs"
        )

    target_device = torch.device(device)
    seed = run_config.get("seed")
    if type(seed) is not int or seed < 0:  # noqa: E721
        raise RachelMatchedMMEvaluationError("matched-MM seed is invalid")
    configure_matched_mm_determinism(seed)
    winners = {
        CONVERGED_METHOD: _freeze_one(
            root=root,
            receipt=receipt,
            method=CONVERGED_METHOD,
            epoch=winner_epoch,
            checkpoint_name=receipt.get("winner_checkpoint"),
            checkpoint_sha256=winner_sha,
            threshold_value=receipt.get("validation_threshold"),
            model_config=model_config,
            model_config_sha256=model_config_sha256,
            target_device=target_device,
        ),
        SAME_EXPOSURE_METHOD: _freeze_one(
            root=root,
            receipt=receipt,
            method=SAME_EXPOSURE_METHOD,
            epoch=5,
            checkpoint_name=same_exposure.get("checkpoint"),
            checkpoint_sha256=epoch5_sha,
            threshold_value=same_exposure.get("validation_threshold"),
            model_config=model_config,
            model_config_sha256=model_config_sha256,
            target_device=target_device,
        ),
    }

    raw_state_path = _safe_run_member(
        root, receipt.get("winner_state_dict"), "matched-MM raw winner state"
    )
    if _sha256_file(raw_state_path) != _require_sha256(
        receipt.get("winner_state_dict_sha256"), "matched-MM raw winner state SHA-256"
    ):
        raise RachelMatchedMMEvaluationError("matched-MM raw winner state SHA-256 differs")
    raw_state = _safe_torch_load(raw_state_path)
    converged_state = winners[CONVERGED_METHOD].model.state_dict()
    if not _state_dicts_equal(raw_state, converged_state):
        raise RachelMatchedMMEvaluationError(
            "matched-MM raw winner state differs from checkpoint state"
        )
    return receipt, receipt_sha256, winners


def configure_matched_mm_determinism(seed: int) -> None:
    """Apply the exact strict deterministic runtime used by matched training."""

    if type(seed) is not int or seed < 0:  # noqa: E721
        raise ValueError("seed must be a non-negative integer")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RachelMatchedMMEvaluationError(
            "CUBLAS deterministic workspace must be configured before torch import"
        )
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def safe_synthetic_model_mask_path(dataset_root: Path, value: object) -> Path:
    """Resolve only a ``model/masks_800/*.png`` member of a frozen release."""

    root = Path(dataset_root).resolve(strict=True)
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RachelMatchedMMEvaluationError("synthetic model_mask_path is invalid")
    logical = PurePosixPath(value)
    if (
        logical.is_absolute()
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.parts[:2] != ("model", "masks_800")
        or logical.suffix.lower() != ".png"
    ):
        raise RachelMatchedMMEvaluationError(
            "matched-MM synthetic input must be model/masks_800 PNG"
        )
    current = root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise RachelMatchedMMEvaluationError("symlinked synthetic model mask is forbidden")
    try:
        path = current.resolve(strict=True)
        path.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise RachelMatchedMMEvaluationError(
            "synthetic model mask is missing or escapes the release"
        ) from error
    if not path.is_file():
        raise RachelMatchedMMEvaluationError("synthetic model mask is not a file")
    return path


def load_historical_mask(path: Path) -> Tensor:
    """Decode an exact 800-square binary PNG and apply historical preprocessing."""

    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (800, 800):
                raise RachelMatchedMMEvaluationError(
                    "matched-MM synthetic mask must be an 800x800 PNG"
                )
            value = np.asarray(image)
    except RachelMatchedMMEvaluationError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise RachelMatchedMMEvaluationError("cannot decode matched-MM synthetic mask") from error
    if value.ndim != 2 or value.dtype != np.uint8:
        raise RachelMatchedMMEvaluationError(
            "matched-MM synthetic mask must be single-channel uint8"
        )
    unique = np.unique(value)
    if len(unique) < 2 or not set(int(item) for item in unique).issubset({0, 255}):
        raise RachelMatchedMMEvaluationError(
            "matched-MM synthetic mask must be non-constant binary 0/255"
        )
    return preprocess_historical_mm_mask(value == 255).contiguous()


def _predict_batches(
    winner: FrozenMatchedMMWinner,
    pair_ids: Sequence[str],
    first: Sequence[Tensor],
    second: Sequence[Tensor],
    *,
    batch_size: int,
) -> Tuple[MatchedMMPrediction, ...]:
    if type(batch_size) is not int or batch_size <= 0:  # noqa: E721
        raise ValueError("batch_size must be a positive integer")
    if (
        not pair_ids
        or len(pair_ids) != len(first)
        or len(first) != len(second)
        or any(not isinstance(value, str) or not value for value in pair_ids)
        or len(set(pair_ids)) != len(pair_ids)
    ):
        raise RachelMatchedMMEvaluationError("matched-MM score inputs are misaligned")
    device = next(winner.model.parameters()).device
    winner.model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(pair_ids), batch_size):
            stop = min(len(pair_ids), start + batch_size)
            input_a = torch.stack(tuple(first[start:stop])).to(
                device=device, dtype=torch.float32
            )
            input_b = torch.stack(tuple(second[start:stop])).to(
                device=device, dtype=torch.float32
            )
            probability = winner.model(input_a, input_b)[:, 0].detach().float().cpu()
            if (
                not torch.isfinite(probability).all()
                or ((probability < 0.0) | (probability > 1.0)).any()
            ):
                raise RachelMatchedMMEvaluationError(
                    "matched-MM model returned an invalid probability"
                )
            predictions.extend(
                MatchedMMPrediction(pair_id=pair_ids[index], probability=float(value))
                for index, value in zip(range(start, stop), probability)
            )
    return tuple(predictions)


def score_synthetic_mask_pairs(
    winner: FrozenMatchedMMWinner,
    pairs: Sequence[SyntheticMaskPair],
    *,
    batch_size: int = 128,
) -> Tuple[MatchedMMPrediction, ...]:
    """Score only pre-resolved synthetic model-mask PNGs."""

    rows = tuple(pairs)
    if any(
        not isinstance(row.pair_id, str)
        or not row.pair_id
        or not isinstance(row.mask_a_path, Path)
        or not isinstance(row.mask_b_path, Path)
        or row.mask_a_path == row.mask_b_path
        for row in rows
    ) or len({row.pair_id for row in rows}) != len(rows):
        raise RachelMatchedMMEvaluationError(
            "synthetic matched-MM pair identity/endpoints are malformed"
        )
    cache: Dict[Path, Tensor] = {}
    for path in sorted(
        {row.mask_a_path for row in rows} | {row.mask_b_path for row in rows}
    ):
        cache[path] = load_historical_mask(path)
    return _predict_batches(
        winner,
        [row.pair_id for row in rows],
        [cache[row.mask_a_path] for row in rows],
        [cache[row.mask_b_path] for row in rows],
        batch_size=batch_size,
    )


def score_alpha_derived_mask_pairs(
    winner: FrozenMatchedMMWinner,
    pair_inputs: Sequence[object],
    fragments: Mapping[str, object],
    *,
    batch_size: int = 128,
) -> Tuple[MatchedMMPrediction, ...]:
    """Score only prepared alpha-derived bool masks from a real population."""

    pair_ids = []
    first = []
    second = []
    cache: Dict[str, Tensor] = {}
    seen_pair_ids = set()
    for row in pair_inputs:
        try:
            pair_id = row.pair_id
            first_id = row.fragment_a_id
            second_id = row.fragment_b_id
            first_fragment = fragments[first_id]
            second_fragment = fragments[second_id]
        except (AttributeError, KeyError) as error:
            raise RachelMatchedMMEvaluationError(
                "real matched-MM pair endpoints are malformed"
            ) from error
        if not isinstance(pair_id, str) or not pair_id:
            raise RachelMatchedMMEvaluationError("real matched-MM pair_id is invalid")
        if pair_id in seen_pair_ids:
            raise RachelMatchedMMEvaluationError("real matched-MM pair_id is duplicated")
        seen_pair_ids.add(pair_id)
        if (
            not isinstance(first_id, str)
            or not first_id
            or not isinstance(second_id, str)
            or not second_id
            or first_id == second_id
        ):
            raise RachelMatchedMMEvaluationError(
                "real matched-MM pair endpoints are invalid"
            )
        for fragment_id, fragment in (
            (first_id, first_fragment),
            (second_id, second_fragment),
        ):
            if fragment_id not in cache:
                mask = np.asarray(getattr(fragment, "mask", None))
                if (
                    mask.shape != (800, 800)
                    or mask.dtype != np.bool_
                    or not bool(mask.any())
                    or bool(mask.all())
                ):
                    raise RachelMatchedMMEvaluationError(
                        "real matched-MM input must be a non-constant 800x800 bool mask"
                    )
                cache[fragment_id] = preprocess_historical_mm_mask(mask).contiguous()
        pair_ids.append(pair_id)
        first.append(cache[first_id])
        second.append(cache[second_id])
    return _predict_batches(
        winner, pair_ids, first, second, batch_size=batch_size
    )


def matched_metrics(
    predictions: Sequence[MatchedMMPrediction],
    labels: Sequence[bool],
    clusters: Sequence[str],
    *,
    threshold: float,
) -> Mapping[str, object]:
    """Compute frozen-threshold and threshold-free views for aligned pairs."""

    values = tuple(predictions)
    valid = tuple(row.valid for row in values)
    report = evaluate_pairwise(
        [row.probability for row in values],
        labels,
        valid,
        clusters,
        threshold=threshold,
    )
    report["coverage"] = {
        "record_count": len(values),
        "valid_count": sum(valid),
        "valid_fraction": float(sum(valid) / len(valid)),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
    }
    return report


def synthetic_pair_records(
    winner: FrozenMatchedMMWinner,
    predictions: Sequence[MatchedMMPrediction],
    manifest_rows: Sequence[object],
) -> Tuple[Mapping[str, object], ...]:
    """Emit the shared sealed-pair schema with honest historical score semantics."""

    values = tuple(predictions)
    rows = tuple(manifest_rows)
    if len(values) != len(rows):
        raise RachelMatchedMMEvaluationError("matched-MM prediction coverage differs")
    output = []
    for prediction, row in zip(values, rows):
        if prediction.pair_id != row.pair_id:
            raise RachelMatchedMMEvaluationError("matched-MM prediction order differs")
        output.append(
            {
                "schema_version": PAIR_SCHEMA_VERSION,
                "arm": winner.method,
                "pair_id": row.pair_id,
                "label": row.label,
                "cluster_id": row.cluster_id,
                "scores": {
                    MATCHED_SCORE: {
                        "probability": prediction.probability,
                        "valid": prediction.valid,
                    }
                },
                "main_score": MATCHED_SCORE,
                "decision": {
                    "validation_threshold": winner.threshold.threshold,
                    "valid": prediction.valid,
                    "predicted_label": (
                        prediction.probability >= winner.threshold.threshold
                        if prediction.valid
                        else None
                    ),
                },
            }
        )
    return tuple(output)


__all__ = (
    "CONVERGED_METHOD",
    "FrozenMatchedMMWinner",
    "MATCHED_METHODS",
    "MATCHED_SCORE",
    "MatchedMMPrediction",
    "RachelMatchedMMEvaluationError",
    "SAME_EXPOSURE_METHOD",
    "SyntheticMaskPair",
    "configure_matched_mm_determinism",
    "freeze_matched_mm_winners",
    "load_historical_mask",
    "matched_metrics",
    "safe_synthetic_model_mask_path",
    "score_alpha_derived_mask_pairs",
    "score_synthetic_mask_pairs",
    "synthetic_pair_records",
)
