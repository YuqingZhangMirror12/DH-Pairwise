"""Frozen same-data benchmark adapters for Rachel pairwise evaluation.

This module is the only bridge from the train/validation-only PairingNet and
ShreddingNet adaptations into later sealed-synthetic and real-Dunhuang
evaluators.  It deliberately contains no dataset path discovery: callers must
freeze both benchmark bundles first and may only then construct a test or real
population.

The adapters expose one common, label-free batch interface.  Ground-truth
labels, correspondences and translations are not accepted by that interface;
they remain evaluator-side data used only after a prediction has been formed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from staging.pairwise_v0_2.baselines import rachel_pairingnet_benchmark as pairing
from staging.pairwise_v0_2.baselines import rachel_shreddingnet_benchmark as shredding
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelBatch
from staging.pairwise_v0_2.training.evaluation import evaluate_pairwise


SCHEMA_VERSION = "rachel-same-data-benchmark-eval-adapter/1.0"
PAIRINGNET_METHOD_KEY = "pairingnet_adapted"
SHREDDINGNET_METHOD_KEY = "shreddingnet_adapted"
BENCHMARK_METHODS = (PAIRINGNET_METHOD_KEY, SHREDDINGNET_METHOD_KEY)
_SHA256_LENGTH = 64
DIRECT_TOLERANCES_PX = (2, 5, 8, 10)
PAIRINGNET_RR_THRESHOLD = 4.0


class RachelBenchmarkEvalAdapterError(RuntimeError):
    """A benchmark freeze, prediction, or common-interface gate failed."""


@dataclass(frozen=True)
class BenchmarkEvalManifestRow:
    """Supervision authority consumed only after one label-free forward."""

    pair_id: str
    label: bool
    cluster_id: str
    source_unit_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("benchmark manifest pair_id is missing")
        if type(self.label) is not bool:  # noqa: E721
            raise TypeError("benchmark manifest label must be bool")
        if not isinstance(self.cluster_id, str) or not self.cluster_id:
            raise ValueError("benchmark manifest cluster_id is missing")
        if (
            not 1 <= len(self.source_unit_ids) <= 2
            or self.source_unit_ids != tuple(sorted(set(self.source_unit_ids)))
            or any(not isinstance(value, str) or not value for value in self.source_unit_ids)
        ):
            raise ValueError("benchmark source units must be sorted and unique")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _lexical_absolute(path: Union[str, Path]) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    return Path(os.path.abspath(str(value)))


def _require_no_symlink_components(path: Path, description: str) -> Path:
    absolute = _lexical_absolute(path)
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise RachelBenchmarkEvalAdapterError(
                description + " is missing: " + str(current)
            ) from error
        if os.path.islink(current):
            raise RachelBenchmarkEvalAdapterError(
                description + " traverses a symlink: " + str(current)
            )
        if current != absolute and not current.is_dir():
            raise RachelBenchmarkEvalAdapterError(
                description + " has a non-directory parent: " + str(current)
            )
        if current == absolute and not (current.is_dir() or current.is_file()):
            raise RachelBenchmarkEvalAdapterError(
                description + " is not a regular file or directory"
            )
        del metadata
    return absolute


def _regular_file(root: Path, name: str, description: str) -> Path:
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        raise RachelBenchmarkEvalAdapterError(description + " name is unsafe")
    path = _require_no_symlink_components(root / name, description)
    if path.is_symlink() or not path.is_file():
        raise RachelBenchmarkEvalAdapterError(
            description + " must be a regular non-symlink file"
        )
    try:
        path.relative_to(root)
    except ValueError as error:  # pragma: no cover - lexical construction is direct
        raise RachelBenchmarkEvalAdapterError(description + " escapes bundle") from error
    return path


def _read_json(path: Path, description: str) -> Dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RachelBenchmarkEvalAdapterError(
            description + " is not readable strict JSON"
        ) from error
    if not isinstance(value, dict):
        raise RachelBenchmarkEvalAdapterError(description + " root is not an object")
    return value


def _require_hash(path: Path, expected: object, description: str) -> str:
    if not _is_sha256(expected):
        raise RachelBenchmarkEvalAdapterError(description + " SHA-256 is malformed")
    observed = _sha256_file(path)
    if observed != expected:
        raise RachelBenchmarkEvalAdapterError(description + " SHA-256 differs")
    return observed


@dataclass(frozen=True)
class CommonBatchPrediction:
    """Method-neutral label-free prediction for one exact Rachel batch."""

    schema_version: str
    method_key: str
    method_id: str
    pair_ids: Tuple[str, ...]
    pair_probability: np.ndarray
    decision_valid: np.ndarray
    translation_hat_rc: np.ndarray
    translation_valid: np.ndarray
    correspondence_indices: Optional[Tuple[np.ndarray, ...]]
    correspondence_scores: Optional[Tuple[np.ndarray, ...]]
    correspondence_semantics: Optional[str]
    auxiliary_scores: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        count = len(self.pair_ids)
        if self.schema_version != SCHEMA_VERSION or self.method_key not in BENCHMARK_METHODS:
            raise ValueError("common prediction identity differs")
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ValueError("common prediction method_id is missing")
        if len(set(self.pair_ids)) != count or any(
            not isinstance(value, str) or not value for value in self.pair_ids
        ):
            raise ValueError("common prediction pair IDs are empty or duplicated")
        probability = np.asarray(self.pair_probability)
        decision_valid = np.asarray(self.decision_valid)
        translation = np.asarray(self.translation_hat_rc)
        translation_valid = np.asarray(self.translation_valid)
        if (
            probability.shape != (count,)
            or decision_valid.shape != (count,)
            or translation.shape != (count, 2)
            or translation_valid.shape != (count,)
            or decision_valid.dtype != np.bool_
            or translation_valid.dtype != np.bool_
        ):
            raise ValueError("common prediction array shapes/dtypes differ")
        if not np.all(np.isfinite(probability)) or np.any(probability < 0.0) or np.any(
            probability > 1.0
        ):
            raise ValueError("common pair probabilities are outside [0,1]")
        if not bool(np.all(decision_valid)):
            raise ValueError("adapted benchmark pair heads must score every decoded pair")
        if np.any(~np.isfinite(translation[translation_valid])):
            raise ValueError("valid common translations must be finite")
        invalid = translation[~translation_valid]
        if invalid.size and not bool(np.all(np.isnan(invalid))):
            raise ValueError("invalid common translations must use NaN sentinels")
        if (self.correspondence_indices is None) != (
            self.correspondence_scores is None
        ):
            raise ValueError("correspondence indices/scores availability differs")
        if self.correspondence_indices is None:
            if self.correspondence_semantics is not None:
                raise ValueError("absent correspondence must have null semantics")
        else:
            if (
                len(self.correspondence_indices) != count
                or len(self.correspondence_scores) != count
                or not isinstance(self.correspondence_semantics, str)
                or not self.correspondence_semantics
            ):
                raise ValueError("common correspondence population differs")
            for indices, scores in zip(
                self.correspondence_indices, self.correspondence_scores
            ):
                index_value = np.asarray(indices)
                score_value = np.asarray(scores)
                if (
                    index_value.ndim != 2
                    or index_value.shape[1:] != (2,)
                    or not np.issubdtype(index_value.dtype, np.integer)
                    or score_value.shape != (len(index_value),)
                    or not np.all(np.isfinite(score_value))
                ):
                    raise ValueError("common sparse correspondence differs")
        for name, values in self.auxiliary_scores.items():
            if not isinstance(name, str) or not name:
                raise ValueError("auxiliary score name is invalid")
            array = np.asarray(values)
            if array.shape != (count,) or not np.all(np.isfinite(array)):
                raise ValueError("auxiliary score array differs")


@dataclass
class FrozenSameDataBenchmark:
    """Hash-bound train/val winner and validation threshold, ready for inference."""

    method_key: str
    method_id: str
    threshold: float
    threshold_artifact: Mapping[str, object]
    threshold_artifact_sha256: str
    freeze_authority_path: Path
    freeze_authority_sha256: str
    training_manifest_sha256: Mapping[str, str]
    checkpoint_sha256_by_stage: Mapping[str, str]
    adaptation_disclosure: Mapping[str, object]
    _predictor: object
    _device: torch.device

    def provenance(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "method_key": self.method_key,
            "method_id": self.method_id,
            "freeze_authority_path": str(self.freeze_authority_path),
            "freeze_authority_sha256": self.freeze_authority_sha256,
            "training_manifest_sha256": dict(self.training_manifest_sha256),
            "checkpoint_sha256_by_stage": dict(self.checkpoint_sha256_by_stage),
            "validation_threshold": dict(self.threshold_artifact),
            "validation_threshold_sha256": self.threshold_artifact_sha256,
            "validation_threshold_value": self.threshold,
            "all_winners_and_validation_threshold_frozen": True,
            "sealed_synthetic_accessed_during_freeze": False,
            "real_data_accessed_during_freeze": False,
            "adaptation": dict(self.adaptation_disclosure),
        }

    def release_to_cpu(self) -> None:
        """Release benchmark GPU state without mutating frozen provenance."""

        if self.method_key == PAIRINGNET_METHOD_KEY:
            self._predictor.to("cpu")
        elif self.method_key == SHREDDINGNET_METHOD_KEY:
            self._predictor.coarse.to("cpu")
            self._predictor.classify.to("cpu")
        else:  # pragma: no cover - constructor/freeze already gates this
            raise RachelBenchmarkEvalAdapterError("unknown frozen benchmark method")

    def predict_batch(
        self, batch: RachelBatch, *, return_correspondence: bool = False
    ) -> CommonBatchPrediction:
        """Run one label-free benchmark pass over every pair in ``batch``."""

        _validate_model_input_batch(batch)
        if self.method_key == PAIRINGNET_METHOD_KEY:
            return _predict_pairingnet(
                self, batch, return_correspondence=return_correspondence
            )
        if self.method_key == SHREDDINGNET_METHOD_KEY:
            return _predict_shreddingnet(
                self, batch, return_correspondence=return_correspondence
            )
        raise RachelBenchmarkEvalAdapterError("unknown frozen benchmark method")


def _validate_model_input_batch(batch: RachelBatch) -> None:
    count = len(batch.pair_ids)
    if count <= 0 or len(set(batch.pair_ids)) != count:
        raise RachelBenchmarkEvalAdapterError("batch pair IDs are empty or duplicated")
    expected_mask = (count, 1, 800, 800)
    expected_points = (count, 512, 2)
    expected_valid = (count, 512)
    if (
        np.asarray(batch.mask_a).shape != expected_mask
        or np.asarray(batch.mask_b).shape != expected_mask
        or np.asarray(batch.points_rc_a).shape != expected_points
        or np.asarray(batch.points_rc_b).shape != expected_points
        or np.asarray(batch.contour_valid_a).shape != expected_valid
        or np.asarray(batch.contour_valid_b).shape != expected_valid
    ):
        raise RachelBenchmarkEvalAdapterError("batch differs from Rachel 800/N512")
    for mask in (batch.mask_a, batch.mask_b):
        value = np.asarray(mask)
        if not np.all(np.isfinite(value)) or np.any(value < 0.0) or np.any(value > 1.0):
            raise RachelBenchmarkEvalAdapterError("model-facing masks are not binary")
        if not np.all((value == 0.0) | (value == 1.0)):
            raise RachelBenchmarkEvalAdapterError("model-facing masks are not 0/1")
    for points, valid in (
        (batch.points_rc_a, batch.contour_valid_a),
        (batch.points_rc_b, batch.contour_valid_b),
    ):
        point_value = np.asarray(points)
        valid_value = np.asarray(valid)
        if valid_value.dtype != np.bool_ or np.any(np.count_nonzero(valid_value, axis=1) < 4):
            raise RachelBenchmarkEvalAdapterError("ordered contour validity differs")
        lengths = np.count_nonzero(valid_value, axis=1)
        prefix = np.arange(512)[None, :] < lengths[:, None]
        if not np.array_equal(prefix, valid_value):
            raise RachelBenchmarkEvalAdapterError("ordered contour is not prefix-valid")
        if not np.all(np.isfinite(point_value[valid_value])):
            raise RachelBenchmarkEvalAdapterError("valid contour points are non-finite")


def build_target_blind_rachel_batch(
    *,
    pair_ids: Sequence[str],
    fragment_a_tokens: Sequence[str],
    fragment_b_tokens: Sequence[str],
    masks_a: Sequence[np.ndarray],
    masks_b: Sequence[np.ndarray],
    points_rc_a: Sequence[np.ndarray],
    points_rc_b: Sequence[np.ndarray],
    contour_valid_a: Sequence[np.ndarray],
    contour_valid_b: Sequence[np.ndarray],
) -> RachelBatch:
    """Build a real-inference batch from model inputs without GT or labels.

    ``RachelBatch`` is a shared transport dataclass that also has supervision
    fields.  This constructor fills those fields with fixed, non-authoritative
    sentinels.  Both benchmark predictors are separately implemented to read
    only masks, contours, validity and pair identity.
    """

    ids = tuple(pair_ids)
    tokens_a = tuple(fragment_a_tokens)
    tokens_b = tuple(fragment_b_tokens)
    count = len(ids)
    fields = (
        tokens_a,
        tokens_b,
        tuple(masks_a),
        tuple(masks_b),
        tuple(points_rc_a),
        tuple(points_rc_b),
        tuple(contour_valid_a),
        tuple(contour_valid_b),
    )
    if count <= 0 or any(len(value) != count for value in fields):
        raise ValueError("target-blind batch fields differ in length")
    if len(set(ids)) != count or any(not value for value in ids):
        raise ValueError("target-blind pair IDs are empty or duplicated")

    def masks(values: Sequence[np.ndarray]) -> np.ndarray:
        rows = []
        for raw in values:
            value = np.asarray(raw)
            if value.shape != (800, 800) or value.dtype != np.bool_:
                raise RachelBenchmarkEvalAdapterError(
                    "target-blind real mask must be bool [800,800]"
                )
            rows.append(value[None].astype(np.float32, copy=False))
        return np.stack(rows)

    mask_a = masks(masks_a)
    mask_b = masks(masks_b)
    # Nearest-neighbour indices match the model-facing binary semantics.  The
    # common benchmark predictors do not consume this compatibility field.
    nearest = np.floor(np.arange(128, dtype=np.float64) * 800.0 / 128.0).astype(
        np.int64
    )
    coarse_a = mask_a[:, :, nearest][:, :, :, nearest]
    coarse_b = mask_b[:, :, nearest][:, :, :, nearest]

    def contours(
        points_values: Sequence[np.ndarray], valid_values: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        point_rows = []
        valid_rows = []
        for raw_points, raw_valid in zip(points_values, valid_values):
            points = np.asarray(raw_points, dtype=np.float32)
            valid = np.asarray(raw_valid)
            if points.shape != (512, 2) or valid.shape != (512,) or valid.dtype != np.bool_:
                raise RachelBenchmarkEvalAdapterError(
                    "target-blind real contour must be padded N512"
                )
            length = int(np.count_nonzero(valid))
            if length < 4 or not np.array_equal(
                valid, np.arange(512) < length
            ):
                raise RachelBenchmarkEvalAdapterError(
                    "target-blind real contour must be prefix-valid"
                )
            point_rows.append(points)
            valid_rows.append(valid)
        return np.stack(point_rows), np.stack(valid_rows)

    point_a, valid_a = contours(points_rc_a, contour_valid_a)
    point_b, valid_b = contours(points_rc_b, contour_valid_b)
    target_a = np.full((count, 512), -2, dtype=np.int64)
    target_b = np.full((count, 512), -2, dtype=np.int64)
    target_a[valid_a] = -1
    target_b[valid_b] = -1
    return RachelBatch(
        pair_ids=ids,
        fragment_a_tokens=tokens_a,
        fragment_b_tokens=tokens_b,
        mask_a=mask_a,
        mask_b=mask_b,
        coarse_mask_a=coarse_a,
        coarse_mask_b=coarse_b,
        points_rc_a=point_a,
        points_rc_b=point_b,
        contour_valid_a=valid_a,
        contour_valid_b=valid_b,
        target_a=target_a,
        target_b=target_b,
        labels=np.zeros(count, dtype=np.float32),
        translation_a_to_b_rc=np.zeros((count, 2), dtype=np.float32),
        translation_a_to_b_xy_cartesian=np.zeros((count, 2), dtype=np.float32),
        translation_valid=np.zeros(count, dtype=np.bool_),
    )


def freeze_pairingnet_benchmark(
    run_directory: Union[str, Path], *, device: Union[str, torch.device]
) -> FrozenSameDataBenchmark:
    """Verify and restore one completed PairingNet train/val bundle."""

    root = _require_no_symlink_components(
        _lexical_absolute(run_directory), "PairingNet run directory"
    )
    if not root.is_dir() or root.name.startswith("."):
        raise RachelBenchmarkEvalAdapterError("PairingNet run is not finalized")
    receipt_path = _regular_file(root, "completion_receipt.json", "PairingNet receipt")
    receipt = _read_json(receipt_path, "PairingNet receipt")
    if (
        receipt.get("schema_version") != pairing.SCHEMA_VERSION
        or receipt.get("status") != "train_validation_complete"
        or receipt.get("method_id") != pairing.METHOD_ID
        or receipt.get("official_commit") != pairing.OFFICIAL_COMMIT
        or receipt.get("winner_checkpoint_kind") != pairing.WINNER_CHECKPOINT_KIND
        or receipt.get("adaptation_claim")
        != "same_data_method_adaptation_not_exact_reproduction"
        or receipt.get("selection_threshold_used") is not False
        or receipt.get("sealed_synthetic_accessed") is not False
        or receipt.get("real_data_accessed") is not False
        or receipt.get("convergence_demonstrated") is not True
        or receipt.get("stop_reason") != "validation_plateau"
    ):
        raise RachelBenchmarkEvalAdapterError(
            "PairingNet completion/convergence identity differs"
        )
    population = receipt.get("population_audit")
    if (
        not isinstance(population, Mapping)
        or population.get("formal_population_required") is not True
        or population.get("parent_lineage_disjoint") is not True
        or population.get("sealed_synthetic_accessed") is not False
        or population.get("real_data_accessed") is not False
        or not isinstance(population.get("population"), Mapping)
        or population["population"].get("train", {}).get("rows")
        != pairing.FORMAL_TRAIN_ROWS
        or population["population"].get("val", {}).get("rows")
        != pairing.FORMAL_VAL_ROWS
    ):
        raise RachelBenchmarkEvalAdapterError("PairingNet same-data population differs")
    manifest_table = population.get("manifests")
    if (
        not isinstance(manifest_table, Mapping)
        or not _is_sha256(manifest_table.get("train_sha256"))
        or not _is_sha256(manifest_table.get("val_sha256"))
    ):
        raise RachelBenchmarkEvalAdapterError(
            "PairingNet train/validation manifest binding differs"
        )

    files = {
        "winner": ("winner.pt", receipt.get("winner_checkpoint_sha256")),
        "last": ("last.pt", receipt.get("last_checkpoint_sha256")),
        "threshold": (
            "validation_threshold.json",
            receipt.get("validation_threshold_file_sha256"),
        ),
        "inference_contract": (
            "inference_contract.json",
            receipt.get("inference_contract_sha256"),
        ),
        "winner_validation_report": (
            "winner_validation_report.json",
            receipt.get("winner_validation_report_sha256"),
        ),
        "winner_validation_predictions": (
            "winner_validation_predictions.jsonl",
            receipt.get("winner_validation_predictions_sha256"),
        ),
    }
    resolved: Dict[str, Path] = {}
    for role, (name, digest) in files.items():
        path = _regular_file(root, name, "PairingNet " + role)
        _require_hash(path, digest, "PairingNet " + role)
        resolved[role] = path
    threshold_document = _read_json(resolved["threshold"], "PairingNet threshold")
    if receipt.get("validation_threshold") != threshold_document:
        raise RachelBenchmarkEvalAdapterError(
            "PairingNet receipt/threshold document differs"
        )
    threshold = pairing.load_frozen_validation_threshold(
        resolved["threshold"], resolved["winner"]
    )
    artifact = threshold_document.get("artifact")
    artifact_sha = threshold_document.get("artifact_content_sha256")
    if (
        not isinstance(artifact, Mapping)
        or not _is_sha256(artifact_sha)
        or _canonical_sha256(dict(artifact)) != artifact_sha
        or artifact.get("threshold") != threshold
    ):
        raise RachelBenchmarkEvalAdapterError(
            "PairingNet validation threshold artifact differs"
        )
    contract = _read_json(resolved["inference_contract"], "PairingNet inference contract")
    pair_decision = contract.get("pair_decision")
    if (
        contract.get("schema_version")
        != "rachel-pairingnet-frozen-inference-contract/1.0"
        or contract.get("method_id") != pairing.METHOD_ID
        or contract.get("orientation") != "upright_known_translation_only_primary"
        or contract.get("sealed_synthetic_accessed") is not False
        or contract.get("real_data_accessed") is not False
        or not isinstance(pair_decision, Mapping)
        or pair_decision.get("status") != "frozen"
        or pair_decision.get("threshold") != threshold
        or pair_decision.get("threshold_artifact_sha256") != artifact_sha
    ):
        raise RachelBenchmarkEvalAdapterError("PairingNet inference contract differs")
    try:
        model = pairing.load_frozen_pairingnet_checkpoint(
            resolved["winner"], torch.device(device)
        )
    except (OSError, RuntimeError, ValueError, pairing.PairingNetBenchmarkError) as error:
        raise RachelBenchmarkEvalAdapterError(
            "PairingNet winner restore failed"
        ) from error
    adaptation = receipt.get("adaptation")
    if (
        not isinstance(adaptation, Mapping)
        or adaptation.get("mask_only") is not True
        or adaptation.get("primary_pose") != "upright_translation_only_consensus"
        or adaptation.get("pair_head_is_official_component") is not False
    ):
        raise RachelBenchmarkEvalAdapterError("PairingNet adaptation disclosure differs")
    return FrozenSameDataBenchmark(
        method_key=PAIRINGNET_METHOD_KEY,
        method_id=pairing.METHOD_ID,
        threshold=threshold,
        threshold_artifact=dict(artifact),
        threshold_artifact_sha256=str(artifact_sha),
        freeze_authority_path=receipt_path,
        freeze_authority_sha256=_sha256_file(receipt_path),
        training_manifest_sha256={
            "train": str(manifest_table["train_sha256"]),
            "val": str(manifest_table["val_sha256"]),
        },
        checkpoint_sha256_by_stage={"winner": str(receipt["winner_checkpoint_sha256"])},
        adaptation_disclosure=dict(adaptation),
        _predictor=model,
        _device=torch.device(device),
    )


def freeze_shreddingnet_benchmark(
    freeze_path: Union[str, Path], *, device: Union[str, torch.device]
) -> FrozenSameDataBenchmark:
    """Verify and restore one complete three-stage ShreddingNet bundle."""

    path = _require_no_symlink_components(
        _lexical_absolute(freeze_path), "ShreddingNet train/val freeze"
    )
    if path.name != "train_val_freeze.json" or not path.is_file():
        raise RachelBenchmarkEvalAdapterError(
            "ShreddingNet authority must be train_val_freeze.json"
        )
    freeze = _read_json(path, "ShreddingNet train/val freeze")
    declared_content_sha = freeze.get("content_sha256")
    canonical = dict(freeze)
    canonical.pop("content_sha256", None)
    if not _is_sha256(declared_content_sha) or _canonical_sha256(canonical) != declared_content_sha:
        raise RachelBenchmarkEvalAdapterError("ShreddingNet freeze content SHA differs")
    scope = freeze.get("scope")
    threshold = freeze.get("threshold")
    checkpoints = freeze.get("checkpoints")
    if (
        freeze.get("schema_version") != shredding.FREEZE_SCHEMA_VERSION
        or freeze.get("checkpoint_kind") != shredding.FREEZE_CHECKPOINT_KIND
        or freeze.get("status") != "complete_train_val_frozen_no_test_or_real"
        or freeze.get("method_id") != shredding.METHOD_ID
        or freeze.get("official_commit") != shredding.OFFICIAL_COMMIT
        or not isinstance(scope, Mapping)
        or scope.get("mask_only") is not True
        or scope.get("rachel_n512") is not True
        or scope.get("upright_known_orientation") is not True
        or scope.get("train_val_only") is not True
        or scope.get("sealed_test_or_real_opened") is not False
        or scope.get("original_shreddingnet_reproduction") is not False
        or scope.get("global_assembly_performed") is not False
        or not isinstance(threshold, Mapping)
        or threshold.get("fit_method") != "maximize_cluster_balanced_f1"
        or threshold.get("source_split") not in {"val", "validation"}
        or threshold.get("threshold_used_for_stage_winner_selection") is not False
        or threshold.get("fit_after_all_three_winners_fixed") is not True
        or not isinstance(checkpoints, Mapping)
        or set(checkpoints) != {"coarse", "matching", "classify"}
    ):
        raise RachelBenchmarkEvalAdapterError("ShreddingNet freeze identity/scope differs")
    pair_threshold = threshold.get("threshold")
    threshold_sha = threshold.get("content_sha256")
    threshold_fields = {
        name: threshold.get(name)
        for name in shredding.PairwiseThresholdArtifact.__dataclass_fields__
    }
    try:
        threshold_artifact = shredding.PairwiseThresholdArtifact(**threshold_fields)
    except (TypeError, ValueError) as error:
        raise RachelBenchmarkEvalAdapterError(
            "ShreddingNet threshold artifact differs"
        ) from error
    if (
        isinstance(pair_threshold, bool)
        or not isinstance(pair_threshold, (int, float))
        or not math.isfinite(float(pair_threshold))
        or not 0.0 <= float(pair_threshold) <= 1.0
        or threshold_sha != threshold_artifact.content_sha256
        or threshold.get("checkpoint_sha256") != checkpoints["classify"].get("sha256")
    ):
        raise RachelBenchmarkEvalAdapterError(
            "ShreddingNet threshold/checkpoint binding differs"
        )
    checkpoint_hashes: Dict[str, str] = {}
    for stage in ("coarse", "matching", "classify"):
        row = checkpoints.get(stage)
        if not isinstance(row, Mapping) or not _is_sha256(row.get("sha256")):
            raise RachelBenchmarkEvalAdapterError(
                "ShreddingNet " + stage + " winner binding differs"
            )
        checkpoint_hashes[stage] = str(row["sha256"])
    dataset_binding = freeze.get("dataset_binding")
    manifest_table = (
        dataset_binding.get("manifest_content_sha256")
        if isinstance(dataset_binding, Mapping)
        else None
    )
    if (
        not isinstance(manifest_table, Mapping)
        or not _is_sha256(manifest_table.get("train"))
        or not _is_sha256(manifest_table.get("val"))
    ):
        raise RachelBenchmarkEvalAdapterError(
            "ShreddingNet train/validation manifest binding differs"
        )
    try:
        predictor = shredding.load_frozen_inference(path, device=device)
    except (OSError, RuntimeError, ValueError, shredding.BenchmarkContractError) as error:
        raise RachelBenchmarkEvalAdapterError(
            "ShreddingNet frozen inference restore failed"
        ) from error
    return FrozenSameDataBenchmark(
        method_key=SHREDDINGNET_METHOD_KEY,
        method_id=shredding.METHOD_ID,
        threshold=float(pair_threshold),
        threshold_artifact=threshold_artifact.to_dict(),
        threshold_artifact_sha256=str(threshold_sha),
        freeze_authority_path=path,
        freeze_authority_sha256=_sha256_file(path),
        training_manifest_sha256={
            "train": str(manifest_table["train"]),
            "val": str(manifest_table["val"]),
        },
        checkpoint_sha256_by_stage=checkpoint_hashes,
        adaptation_disclosure={
            "claim": "same_data_method_adaptation_not_exact_reproduction",
            "mask_only": True,
            "rachel_n512": True,
            "upright_known_translation_only_primary": True,
            "global_assembly_performed": False,
            "native_cm_fm_se_or_ga_claimed": False,
            "balanced_pair_list_diagnostics_are_native_cm_fm_se": False,
        },
        _predictor=predictor,
        _device=torch.device(device),
    )


def freeze_same_data_benchmarks(
    *,
    pairingnet_run_directory: Union[str, Path],
    shreddingnet_freeze_path: Union[str, Path],
    device: Union[str, torch.device],
) -> Mapping[str, FrozenSameDataBenchmark]:
    """Freeze both benchmark methods before a caller may open test or real data."""

    pairingnet = freeze_pairingnet_benchmark(
        pairingnet_run_directory, device=device
    )
    shreddingnet = freeze_shreddingnet_benchmark(
        shreddingnet_freeze_path, device=device
    )
    frozen = {
        PAIRINGNET_METHOD_KEY: pairingnet,
        SHREDDINGNET_METHOD_KEY: shreddingnet,
    }
    if tuple(frozen) != BENCHMARK_METHODS:
        raise RachelBenchmarkEvalAdapterError("benchmark freeze order changed")
    if (
        pairingnet.training_manifest_sha256
        != shreddingnet.training_manifest_sha256
    ):
        raise RachelBenchmarkEvalAdapterError(
            "PairingNet/ShreddingNet train-validation manifest bytes differ"
        )
    return frozen


def _predict_pairingnet(
    frozen: FrozenSameDataBenchmark,
    batch: RachelBatch,
    *,
    return_correspondence: bool,
) -> CommonBatchPrediction:
    model = frozen._predictor
    if not isinstance(model, pairing.PairingNetRachelAdapted):
        raise RachelBenchmarkEvalAdapterError("PairingNet predictor type differs")
    device = frozen._device
    inputs = (
        torch.as_tensor(batch.mask_a, dtype=torch.float32, device=device),
        torch.as_tensor(batch.mask_b, dtype=torch.float32, device=device),
        torch.as_tensor(batch.points_rc_a, dtype=torch.float32, device=device),
        torch.as_tensor(batch.points_rc_b, dtype=torch.float32, device=device),
        torch.as_tensor(batch.contour_valid_a, dtype=torch.bool, device=device),
        torch.as_tensor(batch.contour_valid_b, dtype=torch.bool, device=device),
    )
    with torch.inference_mode():
        output = model(*inputs)
    probability = output.pair_probability.detach().float().cpu().numpy()
    similarity = output.similarity.detach().float().cpu().numpy()
    count = len(batch.pair_ids)
    translation = np.full((count, 2), np.nan, dtype=np.float32)
    translation_valid = np.zeros(count, dtype=np.bool_)
    indices_out = []
    scores_out = []
    for index in range(count):
        estimate = pairing.translation_only_consensus(
            similarity[index],
            batch.points_rc_a[index],
            batch.points_rc_b[index],
            batch.contour_valid_a[index],
            batch.contour_valid_b[index],
            model.config,
        )
        if estimate.valid and estimate.translation_rc is not None:
            translation[index] = estimate.translation_rc
            translation_valid[index] = True
        if return_correspondence:
            indices, scores = pairing._thresholded_correspondences(
                similarity[index],
                batch.contour_valid_a[index],
                batch.contour_valid_b[index],
                model.config.inference_epsilon,
            )
            indices_out.append(indices)
            scores_out.append(scores)
    return CommonBatchPrediction(
        schema_version=SCHEMA_VERSION,
        method_key=frozen.method_key,
        method_id=frozen.method_id,
        pair_ids=tuple(batch.pair_ids),
        pair_probability=probability.astype(np.float32, copy=False),
        decision_valid=np.ones(count, dtype=np.bool_),
        translation_hat_rc=translation,
        translation_valid=translation_valid,
        correspondence_indices=tuple(indices_out) if return_correspondence else None,
        correspondence_scores=tuple(scores_out) if return_correspondence else None,
        correspondence_semantics=(
            "pairingnet_diagonal_morphology_probability_strictly_above_0.006"
            if return_correspondence
            else None
        ),
        auxiliary_scores={},
    )


def _predict_shreddingnet(
    frozen: FrozenSameDataBenchmark,
    batch: RachelBatch,
    *,
    return_correspondence: bool,
) -> CommonBatchPrediction:
    predictor = frozen._predictor
    if not isinstance(predictor, shredding.RachelShreddingNetInference):
        raise RachelBenchmarkEvalAdapterError("ShreddingNet predictor type differs")
    output = predictor.predict_batch(
        batch,
        return_correspondence=return_correspondence,
        compute_pose_for_rejected=True,
    )
    probability = np.asarray(output.pair_score, dtype=np.float32)
    translation = np.asarray(output.translation_hat_rc, dtype=np.float32).copy()
    translation_valid = np.asarray(output.translation_valid, dtype=np.bool_)
    translation[~translation_valid] = np.nan
    indices_out = []
    scores_out = []
    if return_correspondence:
        if output.correspondence_probability is None or output.correspondence_binary is None:
            raise RachelBenchmarkEvalAdapterError(
                "ShreddingNet correspondence output was requested but omitted"
            )
        for probability_matrix, binary_matrix in zip(
            output.correspondence_probability, output.correspondence_binary
        ):
            row, column = np.nonzero(binary_matrix)
            indices = np.column_stack((row, column)).astype(np.int64, copy=False)
            scores = np.asarray(
                probability_matrix[row, column], dtype=np.float32
            )
            indices_out.append(indices)
            scores_out.append(scores)
    return CommonBatchPrediction(
        schema_version=SCHEMA_VERSION,
        method_key=frozen.method_key,
        method_id=frozen.method_id,
        pair_ids=tuple(output.pair_ids),
        pair_probability=probability,
        decision_valid=np.ones(len(output.pair_ids), dtype=np.bool_),
        translation_hat_rc=translation,
        translation_valid=translation_valid,
        correspondence_indices=tuple(indices_out) if return_correspondence else None,
        correspondence_scores=tuple(scores_out) if return_correspondence else None,
        correspondence_semantics=(
            "shreddingnet_released_float_morphology_then_probability_above_0.006"
            if return_correspondence
            else None
        ),
        auxiliary_scores={
            "coarse_cosine_similarity": np.asarray(output.coarse_score, dtype=np.float32)
        },
    )


def _safe_ratio(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _prf(true_positive: int, predicted: int, target: int) -> Dict[str, object]:
    precision = _safe_ratio(true_positive, predicted)
    recall = _safe_ratio(true_positive, target)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive_count": int(true_positive),
        "predicted_count": int(predicted),
        "target_count": int(target),
        "false_positive_count": int(predicted - true_positive),
        "false_negative_count": int(target - true_positive),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _ordered_polygon_area_rc(points_rc: np.ndarray) -> float:
    points = np.asarray(points_rc, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (2,)
        or len(points) < 3
        or not np.all(np.isfinite(points))
    ):
        raise RachelBenchmarkEvalAdapterError(
            "PairingNet-compatible contour area input differs"
        )
    quantized = points.astype(np.int32).astype(np.float64)
    row = quantized[:, 0]
    column = quantized[:, 1]
    return float(
        0.5
        * abs(
            np.dot(column, np.roll(row, -1))
            - np.dot(row, np.roll(column, -1))
        )
    )


def _symmetric_hausdorff(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1:] != (2,)
        or right.shape[1:] != (2,)
        or not len(left)
        or not len(right)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise RachelBenchmarkEvalAdapterError(
            "PairingNet-compatible Hausdorff input differs"
        )

    def directed(source: np.ndarray, target: np.ndarray) -> float:
        nearest = np.full(len(source), np.inf, dtype=np.float64)
        for start in range(0, len(target), 128):
            block = target[start : start + 128]
            distance = np.linalg.norm(
                source[:, None, :] - block[None, :, :], axis=2
            )
            nearest = np.minimum(nearest, distance.min(axis=1))
        return float(nearest.max())

    return max(directed(left, right), directed(right, left))


def _reciprocal_top1(
    indices: np.ndarray, scores: np.ndarray
) -> Tuple[Tuple[int, int], ...]:
    pairs = np.asarray(indices, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if not len(pairs):
        return ()
    # Lexicographic tie breaks make the derived diagnostic deterministic.
    row_best: Dict[int, Tuple[float, int]] = {}
    column_best: Dict[int, Tuple[float, int]] = {}
    for (source, target), score in zip(pairs.tolist(), values.tolist()):
        previous_row = row_best.get(source)
        if previous_row is None or score > previous_row[0] or (
            score == previous_row[0] and target < previous_row[1]
        ):
            row_best[source] = (score, target)
        previous_column = column_best.get(target)
        if previous_column is None or score > previous_column[0] or (
            score == previous_column[0] and source < previous_column[1]
        ):
            column_best[target] = (score, source)
    return tuple(
        sorted(
            (source, target)
            for source, (_, target) in row_best.items()
            if column_best.get(target, (math.nan, -1))[1] == source
        )
    )


def _correspondence_record(
    prediction: CommonBatchPrediction,
    batch: RachelBatch,
    index: int,
) -> Dict[str, object]:
    if (
        prediction.correspondence_indices is None
        or prediction.correspondence_scores is None
        or prediction.correspondence_semantics is None
    ):
        return {
            "status": "not_applicable",
            "reason": "correspondence_output_not_requested_or_not_available",
            "metrics": None,
        }
    length_a = int(np.count_nonzero(batch.contour_valid_a[index]))
    length_b = int(np.count_nonzero(batch.contour_valid_b[index]))
    raw_indices = np.asarray(prediction.correspondence_indices[index], dtype=np.int64)
    raw_scores = np.asarray(prediction.correspondence_scores[index], dtype=np.float64)
    if raw_indices.shape != (len(raw_indices), 2) or raw_scores.shape != (
        len(raw_indices),
    ):
        raise RachelBenchmarkEvalAdapterError("sparse correspondence shape differs")
    if len(raw_indices) and (
        np.any(raw_indices < 0)
        or np.any(raw_indices[:, 0] >= length_a)
        or np.any(raw_indices[:, 1] >= length_b)
    ):
        raise RachelBenchmarkEvalAdapterError(
            "predicted correspondence references padding"
        )
    predicted = set((int(a), int(b)) for a, b in raw_indices.tolist())
    if len(predicted) != len(raw_indices):
        raise RachelBenchmarkEvalAdapterError(
            "predicted sparse correspondence contains duplicates"
        )
    target = set(
        (int(source), int(target_index))
        for source, target_index in enumerate(batch.target_a[index].tolist())
        if target_index >= 0
    )
    if any(source >= length_a or target_index >= length_b for source, target_index in target):
        raise RachelBenchmarkEvalAdapterError(
            "target correspondence references padding"
        )
    mutual = set(_reciprocal_top1(raw_indices, raw_scores))
    return {
        "status": "reported",
        "prediction_semantics": prediction.correspondence_semantics,
        "thresholded_exact": {
            "true_positive_count": len(predicted & target),
            "predicted_count": len(predicted),
            "target_count": len(target),
        },
        "reciprocal_top1_exact": {
            "true_positive_count": len(mutual & target),
            "predicted_count": len(mutual),
            "target_count": len(target),
        },
        "dustbin_aware": {
            "status": "not_applicable",
            "reason": "adapted_benchmark_has_no_explicit_Sinkhorn_dustbin_output",
        },
    }


def _registration_record(
    prediction: CommonBatchPrediction,
    batch: RachelBatch,
    index: int,
) -> Dict[str, object]:
    target_valid = bool(batch.translation_valid[index])
    pose_valid = bool(prediction.translation_valid[index])
    base: Dict[str, object] = {
        "target_valid": target_valid,
        "prediction_valid": pose_valid,
        "identity_fallback_used": (not pose_valid if target_valid else None),
        "e_rmse": None,
        "registration_recall_lt4_success": False if target_valid else None,
        "symmetric_hausdorff_px": None,
        "translation_l2_px": None,
        "normalized_translation_error": None,
        "source_contour_area_px2": None,
        "target_contour_area_px2": None,
        "rotation_error": {
            "status": "not_applicable_conditioned_upright_orientation",
            "estimated_or_supervised": False,
            "ground_truth_rotation_degrees": 0.0,
        },
    }
    if not target_valid:
        return base
    matched_source = np.flatnonzero(batch.target_a[index] >= 0)
    if not len(matched_source):
        raise RachelBenchmarkEvalAdapterError(
            "positive benchmark pair has no seam correspondences"
        )
    matched_target = batch.target_a[index, matched_source].astype(
        np.int64, copy=False
    )
    source = batch.points_rc_a[index, matched_source].astype(
        np.float64, copy=False
    )
    target = batch.points_rc_b[index, matched_target].astype(
        np.float64, copy=False
    )
    effective_translation = (
        prediction.translation_hat_rc[index].astype(np.float64, copy=False)
        if pose_valid
        else np.zeros(2, dtype=np.float64)
    )
    transformed = source + effective_translation[None, :]
    residual = np.linalg.norm(transformed - target, axis=1)
    e_rmse = float(np.sqrt(residual.mean()))
    translation_l2 = float(
        np.linalg.norm(
            effective_translation
            - batch.translation_a_to_b_rc[index].astype(np.float64, copy=False)
        )
    )
    source_area = _ordered_polygon_area_rc(
        batch.points_rc_a[index, batch.contour_valid_a[index]]
    )
    target_area = _ordered_polygon_area_rc(
        batch.points_rc_b[index, batch.contour_valid_b[index]]
    )
    area_sum = source_area + target_area
    if area_sum <= 0.0:
        raise RachelBenchmarkEvalAdapterError(
            "PairingNet-compatible contour area sum is non-positive"
        )
    base.update(
        {
            "e_rmse": e_rmse,
            "registration_recall_lt4_success": e_rmse < PAIRINGNET_RR_THRESHOLD,
            "symmetric_hausdorff_px": _symmetric_hausdorff(transformed, target),
            "translation_l2_px": translation_l2,
            "normalized_translation_error": translation_l2 / area_sum,
            "source_contour_area_px2": source_area,
            "target_contour_area_px2": target_area,
        }
    )
    return base


def _direct_geometry_record(
    frozen: FrozenSameDataBenchmark,
    prediction: CommonBatchPrediction,
    batch: RachelBatch,
    index: int,
) -> Dict[str, object]:
    target_valid = bool(batch.translation_valid[index])
    pose_valid = bool(prediction.translation_valid[index])
    probability = float(prediction.pair_probability[index])
    decision_valid = bool(prediction.decision_valid[index])
    predicted_edge = decision_valid and probability >= frozen.threshold
    translation = (
        [float(value) for value in prediction.translation_hat_rc[index]]
        if pose_valid
        else None
    )
    error: Optional[float] = None
    if target_valid and pose_valid:
        error = float(
            np.linalg.norm(
                prediction.translation_hat_rc[index].astype(np.float64, copy=False)
                - batch.translation_a_to_b_rc[index].astype(np.float64, copy=False)
            )
        )
        if not math.isfinite(error):
            raise RachelBenchmarkEvalAdapterError(
                "valid benchmark translation error is non-finite"
            )
    return {
        "schema_version": "rachel-common-direct-geometry/1.0",
        "orientation": {
            "condition": "known_upright_no_rotation_degree_of_freedom",
            "rotation_error": {
                "status": "not_applicable",
                "reason": "orientation_is_conditioned_and_not_estimated",
            },
        },
        "decision_valid": decision_valid,
        "translation_target_valid": target_valid,
        "translation_prediction_valid": pose_valid,
        "translation_hat_rc": translation,
        "translation_l2_px": error,
        "correspondence": _correspondence_record(prediction, batch, index),
        "pairingnet_style_registration": _registration_record(
            prediction, batch, index
        ),
        "assembly_edge": {
            "target_edge": target_valid,
            "predicted_edge": predicted_edge,
            "true_positive_by_tolerance": {
                "at_{}".format(tolerance): bool(
                    target_valid
                    and predicted_edge
                    and error is not None
                    and error <= tolerance
                )
                for tolerance in DIRECT_TOLERANCES_PX
            },
        },
    }


def summarize_common_direct_records(
    records: Sequence[Mapping[str, object]],
    *,
    method_key: str,
    method_id: str,
    threshold: float,
    threshold_artifact_sha256: str,
    elapsed_seconds: Optional[float] = None,
    forward_batch_count: Optional[int] = None,
) -> Dict[str, object]:
    """Aggregate common pairability, translation and assembly metrics."""

    rows = tuple(records)
    if not rows or method_key not in BENCHMARK_METHODS:
        raise ValueError("common direct records/method are invalid")
    labels = np.asarray([bool(row["label"]) for row in rows], dtype=np.bool_)
    probabilities = np.asarray(
        [float(row["scores"]["pair_probability"]["probability"]) for row in rows],
        dtype=np.float64,
    )
    valid = np.asarray(
        [bool(row["scores"]["pair_probability"]["valid"]) for row in rows],
        dtype=np.bool_,
    )
    clusters = tuple(str(row["cluster_id"]) for row in rows)
    geometries = tuple(row["geometry"] for row in rows)
    if not all(isinstance(value, Mapping) for value in geometries):
        raise RachelBenchmarkEvalAdapterError("common direct geometry is missing")
    pairwise = evaluate_pairwise(
        probabilities, labels, valid, clusters, threshold=float(threshold)
    )
    positive = labels
    pose_valid = np.asarray(
        [bool(value["translation_prediction_valid"]) for value in geometries],
        dtype=np.bool_,
    )
    errors = np.asarray(
        [
            np.nan
            if value["translation_l2_px"] is None
            else float(value["translation_l2_px"])
            for value in geometries
        ],
        dtype=np.float64,
    )
    valid_positive = positive & pose_valid & np.isfinite(errors)
    positive_count = int(positive.sum())
    conditioned = errors[valid_positive]
    translation: Dict[str, object] = {
        "scope": "all_positive_pairs_invalid_pose_counts_as_recall_failure",
        "eligible_positive_count": positive_count,
        "valid_prediction_count": int(valid_positive.sum()),
        "valid_prediction_fraction": _safe_ratio(
            int(valid_positive.sum()), positive_count
        ),
        "te_px_conditioned_on_valid_pose": {
            "count": int(len(conditioned)),
            "median": float(np.median(conditioned)) if len(conditioned) else None,
            "p90": float(np.quantile(conditioned, 0.9)) if len(conditioned) else None,
        },
        "unconditional_positive_recall": {
            "at_{}px".format(tolerance): _safe_ratio(
                int(np.count_nonzero(valid_positive & (errors <= tolerance))),
                positive_count,
            )
            for tolerance in DIRECT_TOLERANCES_PX
        },
    }
    predicted_edge = valid & (probabilities >= threshold)
    assembly = {}
    for tolerance in DIRECT_TOLERANCES_PX:
        correct = positive & predicted_edge & valid_positive & (errors <= tolerance)
        assembly["at_{}px".format(tolerance)] = _prf(
            int(correct.sum()), int(predicted_edge.sum()), positive_count
        )
    registration = [value["pairingnet_style_registration"] for value in geometries]
    eligible_registration = [
        value for value in registration if bool(value["target_valid"])
    ]
    if len(eligible_registration) != positive_count:
        raise RachelBenchmarkEvalAdapterError(
            "PairingNet-compatible registration population differs"
        )
    pairingnet_compatible = {
        "compatibility_source": (
            "PairingNet released matching_test.py upright translation-only specialization"
        ),
        "e_rmse_definition": (
            "sqrt(mean(per_correspondence_euclidean_distance)); not conventional RMSE"
        ),
        "registration_recall_definition": "e_rmse_strictly_less_than_4",
        "invalid_pose_fallback": "identity_translation_for_unconditional_aggregation",
        "eligible_positive_count": positive_count,
        "valid_pose_count": int(
            sum(bool(value["prediction_valid"]) for value in eligible_registration)
        ),
        "identity_fallback_count": int(
            sum(bool(value["identity_fallback_used"]) for value in eligible_registration)
        ),
        "rr_lt4": _safe_ratio(
            sum(
                bool(value["registration_recall_lt4_success"])
                for value in eligible_registration
            ),
            positive_count,
        ),
        "mean_e_rmse": float(
            np.mean([float(value["e_rmse"]) for value in eligible_registration])
        ),
        "mean_symmetric_hausdorff_px": float(
            np.mean(
                [
                    float(value["symmetric_hausdorff_px"])
                    for value in eligible_registration
                ]
            )
        ),
        "mean_normalized_translation_error": float(
            np.mean(
                [
                    float(value["normalized_translation_error"])
                    for value in eligible_registration
                ]
            )
        ),
        "rotation_error": {
            "status": "not_applicable",
            "reason": "upright_orientation_is_conditioned",
        },
    }
    correspondence_rows = [value["correspondence"] for value in geometries]
    reported = [value for value in correspondence_rows if value["status"] == "reported"]
    if reported and len(reported) != len(rows):
        raise RachelBenchmarkEvalAdapterError(
            "correspondence availability changed within one method"
        )
    if reported:
        correspondence = {
            "status": "reported",
            "scope": "method_sparse_correspondence_not_dustbin_comparable",
            "prediction_semantics": reported[0]["prediction_semantics"],
            "thresholded_exact": _prf(
                sum(value["thresholded_exact"]["true_positive_count"] for value in reported),
                sum(value["thresholded_exact"]["predicted_count"] for value in reported),
                sum(value["thresholded_exact"]["target_count"] for value in reported),
            ),
            "reciprocal_top1_exact": _prf(
                sum(value["reciprocal_top1_exact"]["true_positive_count"] for value in reported),
                sum(value["reciprocal_top1_exact"]["predicted_count"] for value in reported),
                sum(value["reciprocal_top1_exact"]["target_count"] for value in reported),
            ),
            "dustbin_aware": reported[0]["dustbin_aware"],
        }
    else:
        correspondence = {
            "status": "not_applicable",
            "reason": "correspondence_output_not_available",
            "metrics": None,
        }
    return {
        "schema_version": "rachel-common-direct-pairwise-report/1.0",
        "status": "complete_frozen_direct_pairwise_evaluation",
        "method_key": method_key,
        "method_id": method_id,
        "pair_count": len(rows),
        "positive_count": positive_count,
        "negative_count": len(rows) - positive_count,
        "pair_ids_sha256": _canonical_sha256([row["pair_id"] for row in rows]),
        "threshold": {
            "value": float(threshold),
            "source": "frozen_validation_only_artifact",
            "artifact_sha256": threshold_artifact_sha256,
            "fit_performed_here": False,
        },
        "pair_classification_diagnostic": pairwise,
        "translation": translation,
        "assembly_edge": {
            "definition": (
                "threshold-accepted edge is correct only for an adjacent pair with "
                "a valid translation within tolerance; wrong or invalid pose is FP+FN"
            ),
            "predicted_edge_count": int(predicted_edge.sum()),
            "target_edge_count": positive_count,
            "by_tolerance": assembly,
        },
        "pairingnet_style_registration": pairingnet_compatible,
        "correspondence": correspondence,
        "unavailable_or_not_applicable": {
            "rotation_error": {
                "status": "not_applicable",
                "reason": "known_upright_orientation_no_rotation_output",
            },
            "native_global_assembly_GA": {
                "status": "not_applicable",
                "reason": "pairwise_only_no_MST_CO_MCTS_or_global_placement",
            },
            "shreddingnet_native_CM_FM_SE": {
                "status": "not_reported",
                "reason": (
                    "Rachel population is a balanced selected 1-to-1 pair list, not "
                    "the exhaustive same-parent candidate graph required by native metrics"
                ),
            },
        },
        "scope": {
            "mask_only": True,
            "rgb_used": False,
            "rachel_n512": True,
            "upright_translation_only": True,
            "same_data_method_adaptation_not_exact_reproduction": True,
            "global_assembly_performed": False,
            "native_cm_fm_se_or_ga_claimed": False,
        },
        "single_forward_population_contract": {
            "forward_pair_count": len(rows),
            "unique_pair_count": len({row["pair_id"] for row in rows}),
            "each_pair_forwarded_exactly_once": True,
            "forward_batch_count": forward_batch_count,
            "elapsed_seconds": elapsed_seconds,
        },
    }


def evaluate_frozen_benchmark_synthetic(
    frozen: FrozenSameDataBenchmark,
    loader: Iterable[RachelBatch],
    manifest_rows: Sequence[BenchmarkEvalManifestRow],
) -> Tuple[Dict[str, object], Tuple[Dict[str, object], ...]]:
    """Forward one frozen method once over the exact synthetic population."""

    authority = tuple(manifest_rows)
    if not authority or len({row.pair_id for row in authority}) != len(authority):
        raise ValueError("benchmark synthetic authority is empty or duplicated")
    records = []
    cursor = 0
    batch_count = 0
    started = time.perf_counter()
    for batch in loader:
        prediction = frozen.predict_batch(batch, return_correspondence=True)
        batch_count += 1
        count = len(prediction.pair_ids)
        expected = authority[cursor : cursor + count]
        if tuple(row.pair_id for row in expected) != prediction.pair_ids:
            raise RachelBenchmarkEvalAdapterError(
                "benchmark prediction order differs from frozen manifest"
            )
        # Supervision is consulted only here, after the entire current batch's
        # label-free neural forward and pose computation have completed.
        expected_labels = np.asarray([row.label for row in expected], dtype=np.float32)
        if not np.array_equal(np.asarray(batch.labels), expected_labels):
            raise RachelBenchmarkEvalAdapterError(
                "benchmark loader labels differ from frozen manifest"
            )
        if not np.array_equal(
            np.asarray(batch.translation_valid), expected_labels.astype(np.bool_)
        ):
            raise RachelBenchmarkEvalAdapterError(
                "benchmark translation target eligibility differs"
            )
        for index, row in enumerate(expected):
            probability = float(prediction.pair_probability[index])
            valid = bool(prediction.decision_valid[index])
            records.append(
                {
                    "schema_version": "rachel-n512-sealed-test-pair/1.0",
                    "arm": frozen.method_key,
                    "method_id": frozen.method_id,
                    "pair_id": row.pair_id,
                    "label": row.label,
                    "cluster_id": row.cluster_id,
                    "source_unit_ids": list(row.source_unit_ids),
                    "scores": {
                        "pair_probability": {
                            "probability": probability,
                            "valid": valid,
                        }
                    },
                    "main_score": "pair_probability",
                    "decision": {
                        "validation_threshold": frozen.threshold,
                        "valid": valid,
                        "predicted_label": (
                            probability >= frozen.threshold if valid else None
                        ),
                    },
                    "geometry": _direct_geometry_record(
                        frozen, prediction, batch, index
                    ),
                }
            )
        cursor += count
    elapsed = time.perf_counter() - started
    if cursor != len(authority) or len(records) != len(authority):
        raise RachelBenchmarkEvalAdapterError(
            "benchmark single-forward population coverage is incomplete"
        )
    metrics = summarize_common_direct_records(
        records,
        method_key=frozen.method_key,
        method_id=frozen.method_id,
        threshold=frozen.threshold,
        threshold_artifact_sha256=frozen.threshold_artifact_sha256,
        elapsed_seconds=elapsed,
        forward_batch_count=batch_count,
    )
    return metrics, tuple(records)


__all__ = [
    "BENCHMARK_METHODS",
    "BenchmarkEvalManifestRow",
    "CommonBatchPrediction",
    "DIRECT_TOLERANCES_PX",
    "FrozenSameDataBenchmark",
    "PAIRINGNET_METHOD_KEY",
    "RachelBenchmarkEvalAdapterError",
    "SCHEMA_VERSION",
    "SHREDDINGNET_METHOD_KEY",
    "build_target_blind_rachel_batch",
    "freeze_pairingnet_benchmark",
    "freeze_same_data_benchmarks",
    "freeze_shreddingnet_benchmark",
    "evaluate_frozen_benchmark_synthetic",
    "summarize_common_direct_records",
]
