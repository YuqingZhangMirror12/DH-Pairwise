"""Deterministic train-only CUDA integration smoke for Pairwise v0.2.

The executable path is deliberately narrow: it accepts only the canonical new
synthetic no-erosion archive/manifest, selects one positive and one negative
training pair, builds all four upright relative-side hypotheses, and runs the
real dustbin-Sinkhorn ``DunhuangPairwiseV02`` model on CUDA.  It never opens a
historical test archive or the sealed real-Dunhuang test collection, and it
never serializes model weights, sample identifiers, archive members, or pixels.

The emitted JSON is a portable aggregate receipt.  Runtime filesystem paths
are command-line inputs only and are rejected if they leak into that receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Required by deterministic CUDA GEMM before the CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from staging.pairwise_v0_2.geometry import CandidateBuilderConfig
from staging.pairwise_v0_2.models import (
    CoarseModelConfig,
    DunhuangPairwiseV02,
    LocalModelConfig,
    PairwiseModelConfig,
)
from staging.pairwise_v0_2.pairwise_data.lazy_mask_loader import (
    ArchiveSourceSpec,
    LazyMaskArchiveLoader,
    LazyMaskLoaderConfig,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    TrainingPairRecord,
    iter_synthetic_pair_records,
)
from staging.pairwise_v0_2.training.checkpoint import canonical_config_hash
from staging.pairwise_v0_2.training.engine import (
    DirectionalStepResult,
    eval_directional_step,
    train_directional_step,
)
from staging.pairwise_v0_2.training.geometry_batch import (
    DIRECTION_NAMES,
    GeometryBatchConfig,
    RaggedGeometryBatch,
    build_geometry_batch,
)


SMOKE_SCHEMA_VERSION = "dunhuang-pairwise-cuda-integration-smoke/0.2"
EXPECTED_ARCHIVE_SHA256 = (
    "e44e4c0e5825d8577861d79eaf4063888a349c6ba3e531be6e41f2b7dde505df"
)
EXPECTED_MANIFEST_SHA256 = (
    "e9772ec8e074873e4343ca42906a4056ea382536d2cbdf6ec881c21891587326"
)
DEFAULT_SEED = 260828
DEFAULT_STEPS = 16
DEFAULT_LEARNING_RATE = 1e-3
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_BANNED_KEY_PARTS = (
    "archive_member",
    "candidate_id",
    "cluster_id",
    "component_id",
    "credential",
    "fragment_id",
    "geometry_cache_key",
    "pair_id",
    "password",
    "private_key",
    "remote_path",
    "secret",
    "source_path",
    "ssh_",
)
_BANNED_EXACT_KEYS = {
    "candidate_id",
    "candidate_ids",
    "fragment_id",
    "fragment_ids",
    "pair_id",
    "pair_ids",
    "sample_id",
    "sample_ids",
}
_BANNED_STRING_PARTS = (
    "/root/",
    "/users/",
    "file://",
    "ssh-rsa",
    "begin private key",
)
_CODE_UNITS = (
    "staging/pairwise_v0_2/geometry/candidate_builder.py",
    "staging/pairwise_v0_2/geometry/schema.py",
    "staging/pairwise_v0_2/models/coarse.py",
    "staging/pairwise_v0_2/models/local_matcher.py",
    "staging/pairwise_v0_2/models/optimal_transport.py",
    "staging/pairwise_v0_2/models/pairwise.py",
    "staging/pairwise_v0_2/pairwise_data/lazy_mask_loader.py",
    "staging/pairwise_v0_2/pairwise_data/training_stream.py",
    "staging/pairwise_v0_2/training/cuda_smoke.py",
    "staging/pairwise_v0_2/training/engine.py",
    "staging/pairwise_v0_2/training/geometry_batch.py",
    "staging/pairwise_v0_2/training/losses.py",
)


class CudaSmokeError(RuntimeError):
    """Raised when any integration invariant fails closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(source_root: Path) -> str:
    """Hash the exact logical code units without emitting their runtime paths."""

    digest = hashlib.sha256()
    for logical_name in _CODE_UNITS:
        path = source_root / logical_name
        if not path.is_file():
            raise CudaSmokeError("a required source unit is missing")
        payload = path.read_bytes()
        digest.update(logical_name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _set_determinism(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _model_config() -> PairwiseModelConfig:
    """Small but architecture-faithful configuration for the smoke only."""

    return PairwiseModelConfig(
        coarse=CoarseModelConfig(
            input_channels=1,
            widths=(8, 16, 24),
            embedding_dim=24,
            hidden_dim=24,
        ),
        local=LocalModelConfig(
            input_channels=3,
            feature_dim=24,
            num_heads=4,
            ff_dim=48,
            matcher_mode="dustbin_sinkhorn",
            # Explicit stability-smoke override.  This receipt must not be
            # read as evidence that the current 0.1/50 default is robust.
            matcher_temperature=0.25,
            # The first real CUDA attempt showed that 50 fixed iterations can
            # cease to meet 1e-3 residual tolerance after optimizer updates.
            # Keep the strict tolerance and double the deterministic budget.
            sinkhorn_iterations=100,
            sinkhorn_tolerance=1e-3,
            require_sinkhorn_convergence=True,
            dropout=0.0,
        ),
        fusion_hidden_dim=16,
        direction_aggregation_temperature=0.25,
    )


def _geometry_config() -> GeometryBatchConfig:
    """Bounded bbox-relative geometry while preserving all four directions."""

    geometry = CandidateBuilderConfig(
        min_component_pixels=16,
        min_contour_points=12,
        min_run_length_fraction=0.02,
        min_run_length_px=4.0,
        window_scale_fractions=(0.12,),
        window_min_px=4.0,
        window_max_px=192.0,
        overlap_fraction=0.5,
        output_size=(16, 16),
        side_resample_count=16,
    )
    return GeometryBatchConfig(
        geometry=geometry,
        coarse_output_size=(64, 64),
        max_batch_size=2,
        max_candidates_per_sample=128,
        max_candidates_per_batch=256,
        max_sequence_length=512,
        max_local_tensor_elements=40_000_000,
    )


def _record_is_train_only_synthetic(record: TrainingPairRecord) -> bool:
    return bool(
        record.split == "train"
        and record.dataset_id == "dunhuang_voronoi_masks_no_erode_v0_2"
        and record.fragment_a.binding.sha256 == EXPECTED_ARCHIVE_SHA256
        and record.fragment_b.binding.sha256 == EXPECTED_ARCHIVE_SHA256
        and record.provenance.get("source_lineage_status")
        == "unavailable_training_only"
        and record.provenance.get("real_dunhuang_sealed_test") is False
    )


def _collect_balanced_candidates(
    records: Iterable[TrainingPairRecord], per_class_limit: int = 32
) -> Tuple[Tuple[TrainingPairRecord, ...], Tuple[TrainingPairRecord, ...]]:
    """Collect a deterministic bounded prefix without exposing record identity."""

    positives: List[TrainingPairRecord] = []
    negatives: List[TrainingPairRecord] = []
    for record in records:
        if not _record_is_train_only_synthetic(record):
            raise CudaSmokeError("record escaped the canonical train-only boundary")
        target = positives if record.label else negatives
        if len(target) < per_class_limit:
            target.append(record)
        if len(positives) >= per_class_limit and len(negatives) >= per_class_limit:
            break
    if not positives or not negatives:
        raise CudaSmokeError("canonical stream did not provide both labels")
    return tuple(positives), tuple(negatives)


def _select_geometry_valid_balanced_batch(
    positives: Sequence[TrainingPairRecord],
    negatives: Sequence[TrainingPairRecord],
    loader: Callable[[Any], np.ndarray],
    config: GeometryBatchConfig,
) -> RaggedGeometryBatch:
    """Choose the first geometry-valid pair of each class deterministically."""

    selected: List[TrainingPairRecord] = []
    for candidates in (positives, negatives):
        chosen: Optional[TrainingPairRecord] = None
        for record in candidates:
            probe = build_geometry_batch([record], loader, config)
            if (
                probe.geometry_valid.all().item()
                and probe.direction_slot_valid.all().item()
                and set(probe.direction_index.tolist()) == {0, 1, 2, 3}
            ):
                chosen = record
                break
        if chosen is None:
            raise CudaSmokeError("no all-four-direction geometry-valid balanced sample")
        selected.append(chosen)
    batch = build_geometry_batch(selected, loader, config)
    if batch.labels.tolist() != [True, False]:
        raise CudaSmokeError("balanced batch label order changed")
    if not batch.geometry_valid.all().item() or not batch.direction_slot_valid.all().item():
        raise CudaSmokeError("balanced batch lost an all-four-direction candidate")
    if set(batch.direction_index.tolist()) != {0, 1, 2, 3}:
        raise CudaSmokeError("not all four direction indices were emitted")
    return batch


def _to_float(value: torch.Tensor) -> float:
    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise CudaSmokeError("a recorded scalar is non-finite")
    return result


def _transport_observation(result: DirectionalStepResult) -> Dict[str, Any]:
    arc = result.output.arc_pairwise_output
    if arc is None or arc.local.transport is None:
        raise CudaSmokeError("dustbin-Sinkhorn transport evidence is missing")
    diagnostics = arc.local.transport.diagnostics
    if not diagnostics.valid_problem.all().item():
        raise CudaSmokeError("Sinkhorn received an invalid transport problem")
    if not diagnostics.finite_output.all().item():
        raise CudaSmokeError("Sinkhorn emitted non-finite output")
    if not diagnostics.converged.all().item():
        raise CudaSmokeError("Sinkhorn did not converge for every arc candidate")
    if not arc.local.valid_problem.all().item() or not arc.decision_valid.all().item():
        raise CudaSmokeError("an arc candidate is not decision-valid")
    directional = result.output.direction_output
    if not directional.pair_valid.all().item():
        raise CudaSmokeError("a pair-level output is invalid")
    return {
        "transport_count": int(diagnostics.converged.numel()),
        "row_residual_max": _to_float(diagnostics.row_residual_max.max()),
        "col_residual_max": _to_float(diagnostics.col_residual_max.max()),
        "iteration_min": int(diagnostics.iteration_count.min().detach().cpu().item()),
        "iteration_max": int(diagnostics.iteration_count.max().detach().cpu().item()),
    }


def _summary(values: Sequence[float]) -> Dict[str, float]:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise CudaSmokeError("summary values must be non-empty and finite")
    numeric = [float(value) for value in values]
    return {
        "initial": numeric[0],
        "final": numeric[-1],
        "minimum": min(numeric),
        "maximum": max(numeric),
        "mean": sum(numeric) / len(numeric),
    }


def _aggregate_transport(observations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not observations:
        raise CudaSmokeError("transport observations are required")
    return {
        "all_converged": True,
        "all_finite": True,
        "observed_forward_passes": len(observations),
        "transport_count_per_forward_min": min(
            int(item["transport_count"]) for item in observations
        ),
        "transport_count_per_forward_max": max(
            int(item["transport_count"]) for item in observations
        ),
        "row_residual_max": max(
            float(item["row_residual_max"]) for item in observations
        ),
        "col_residual_max": max(
            float(item["col_residual_max"]) for item in observations
        ),
        "iteration_min": min(int(item["iteration_min"]) for item in observations),
        "iteration_max": max(int(item["iteration_max"]) for item in observations),
    }


def _assert_portable_aggregate(value: Any, location: str = "receipt") -> None:
    """Reject paths, credentials, record IDs, pixels, and non-JSON scalars."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).casefold()
            if key in _BANNED_EXACT_KEYS or any(
                part in key for part in _BANNED_KEY_PARTS
            ):
                raise CudaSmokeError("receipt contains a forbidden key")
            if key in {"image", "mask", "pixels", "tensor", "weights"}:
                raise CudaSmokeError("receipt contains non-aggregate data")
            _assert_portable_aggregate(item, location + "." + str(raw_key))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_portable_aggregate(item, "{}[{}]".format(location, index))
        return
    if isinstance(value, str):
        lowered = value.casefold()
        if value.startswith(("/", "~")) or _WINDOWS_ABSOLUTE.match(value):
            raise CudaSmokeError("receipt contains an absolute path")
        if any(part in lowered for part in _BANNED_STRING_PARTS):
            raise CudaSmokeError("receipt contains path or credential material")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise CudaSmokeError("receipt contains a non-portable scalar")


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the public receipt contract before serialization."""

    if receipt.get("schema_version") != SMOKE_SCHEMA_VERSION:
        raise CudaSmokeError("unexpected receipt schema")
    if receipt.get("status") != "passed":
        raise CudaSmokeError("only a passed smoke may be serialized")
    hashes = receipt.get("hashes")
    if not isinstance(hashes, Mapping) or not all(
        isinstance(value, str) and _HEX64.fullmatch(value)
        for value in hashes.values()
    ):
        raise CudaSmokeError("receipt hashes are incomplete or malformed")
    scope = receipt.get("scope")
    if not isinstance(scope, Mapping) or not (
        scope.get("new_synthetic_train_only") is True
        and scope.get("historical_data_used") is False
        and scope.get("historical_test_used") is False
        and scope.get("real_dunhuang_data_used") is False
        and scope.get("checkpoint_written") is False
        and scope.get("sample_identifiers_persisted") is False
        and scope.get("pixels_persisted") is False
    ):
        raise CudaSmokeError("receipt scope does not prove train-only isolation")
    results = receipt.get("results")
    if not isinstance(results, Mapping) or not (
        results.get("finite_loss") is True
        and results.get("finite_nonzero_gradients") is True
        and results.get("all_pair_outputs_valid") is True
        and results.get("final_loss_lower_than_initial") is True
    ):
        raise CudaSmokeError("receipt does not prove all smoke gates")
    _assert_portable_aggregate(receipt)
    json.dumps(
        receipt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    validate_receipt(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        receipt,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(str(temporary), str(path))


def run_cuda_smoke(
    *,
    manifest_path: Path,
    archive_path: Path,
    source_root: Path,
    seed: int = DEFAULT_SEED,
    steps: int = DEFAULT_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> Dict[str, Any]:
    """Execute the real train-only geometry/model/CUDA integration smoke."""

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise CudaSmokeError("CUDA is required for this integration smoke")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    _set_determinism(seed)

    hash_started = time.perf_counter()
    manifest_sha256 = _sha256_file(manifest_path)
    archive_sha256 = _sha256_file(archive_path)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise CudaSmokeError("manifest is not the canonical synthetic train manifest")
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise CudaSmokeError("archive is not the canonical synthetic train archive")
    source_tree_sha256 = _source_tree_sha256(source_root)
    hash_seconds = time.perf_counter() - hash_started

    selection_started = time.perf_counter()
    positives, negatives = _collect_balanced_candidates(
        iter_synthetic_pair_records(manifest_path)
    )
    binding = positives[0].fragment_a.binding
    if binding.sha256 != archive_sha256 or any(
        record.fragment_a.binding != binding or record.fragment_b.binding != binding
        for record in positives + negatives
    ):
        raise CudaSmokeError("manifest archive binding is inconsistent")
    selection_seconds = time.perf_counter() - selection_started

    geometry_config = _geometry_config()
    preprocess_started = time.perf_counter()
    source = ArchiveSourceSpec(binding, archive_path, sha256_verified=True)
    with LazyMaskArchiveLoader(
        {binding.logical_id: source},
        LazyMaskLoaderConfig(cache_size=16, max_cache_pixels=10_000_000),
    ) as loader:
        batch = _select_geometry_valid_balanced_batch(
            positives, negatives, loader, geometry_config
        )
        loader_stats = loader.stats.to_dict()
    preprocess_seconds = time.perf_counter() - preprocess_started

    if tuple(batch.direction_names) != tuple(DIRECTION_NAMES):
        raise CudaSmokeError("geometry/model direction order differs")
    direction_counts = {
        name: int((batch.direction_index == index).sum().item())
        for index, name in enumerate(DIRECTION_NAMES)
    }
    if min(direction_counts.values()) < 2:
        raise CudaSmokeError("each of four directions must cover both pairs")

    device = torch.device("cuda", 0)
    transfer_started = time.perf_counter()
    # Some PyTorch CUDA builds reject a ``torch.device`` object in the memory
    # statistics helpers even though tensor transfers accept it.  Freeze the
    # ordinal explicitly for a portable smoke across those builds.
    device_ordinal = 0
    torch.cuda.set_device(device_ordinal)
    torch.cuda.reset_peak_memory_stats(device_ordinal)
    device_batch = batch.to(device)
    torch.cuda.synchronize(device_ordinal)
    transfer_seconds = time.perf_counter() - transfer_started

    model_config = _model_config()
    model = DunhuangPairwiseV02(model_config).to(device)
    if model.config.local.matcher_mode != "dustbin_sinkhorn":
        raise CudaSmokeError("smoke must use dustbin-Sinkhorn")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=0.0
    )

    model_inputs = device_batch.model_inputs()
    observations: List[Mapping[str, Any]] = []
    initial_started = time.perf_counter()
    initial = eval_directional_step(
        model,
        model_inputs,
        device_batch.labels,
        device_batch.direction_target,
        device_batch.direction_target_valid,
    )
    torch.cuda.synchronize(device_ordinal)
    initial_seconds = time.perf_counter() - initial_started
    initial_loss = _to_float(initial.total_loss)
    observations.append(_transport_observation(initial))

    loss_values = [initial_loss]
    gradient_norms: List[float] = []
    training_started = time.perf_counter()
    for _ in range(steps):
        result = train_directional_step(
            model,
            model_inputs,
            device_batch.labels,
            device_batch.direction_target,
            device_batch.direction_target_valid,
            optimizer,
            max_gradient_norm=5.0,
        )
        gradient_norm = float(result.gradient_norm or 0.0)
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise CudaSmokeError("training gradient norm is non-finite or zero")
        gradient_norms.append(gradient_norm)
        loss_values.append(_to_float(result.total_loss))
        observations.append(_transport_observation(result))
    torch.cuda.synchronize(device_ordinal)
    training_seconds = time.perf_counter() - training_started

    final_started = time.perf_counter()
    final = eval_directional_step(
        model,
        model_inputs,
        device_batch.labels,
        device_batch.direction_target,
        device_batch.direction_target_valid,
    )
    torch.cuda.synchronize(device_ordinal)
    final_seconds = time.perf_counter() - final_started
    final_loss = _to_float(final.total_loss)
    loss_values.append(final_loss)
    observations.append(_transport_observation(final))
    if not final_loss < initial_loss:
        raise CudaSmokeError("short overfit did not lower final loss")

    candidate_count = batch.candidate_count
    sequence_a = batch.token_mask_a.sum(dim=1).to(torch.float64)
    sequence_b = batch.token_mask_b.sum(dim=1).to(torch.float64)
    peak_memory = int(torch.cuda.max_memory_allocated(device_ordinal))
    total_seconds = time.perf_counter() - started
    receipt: Dict[str, Any] = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "new_synthetic_train_only": True,
            "historical_data_used": False,
            "historical_test_used": False,
            "real_dunhuang_data_used": False,
            "checkpoint_written": False,
            "sample_identifiers_persisted": False,
            "pixels_persisted": False,
        },
        "hashes": {
            "archive_sha256": archive_sha256,
            "manifest_sha256": manifest_sha256,
            "source_tree_sha256": source_tree_sha256,
            "model_config_sha256": canonical_config_hash(model_config),
            "geometry_config_sha256": geometry_config.fingerprint,
        },
        "reproducibility": {
            "seed": seed,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "tf32_enabled": False,
            "source_unit_count": len(_CODE_UNITS),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": str(torch.version.cuda),
            "cudnn": int(torch.backends.cudnn.version() or 0),
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "pillow": _version("Pillow"),
            "device_type": "cuda",
            "device_name": torch.cuda.get_device_name(device_ordinal),
            "compute_capability": list(
                torch.cuda.get_device_capability(device_ordinal)
            ),
            "peak_allocated_bytes": peak_memory,
        },
        "configuration": {
            "model_family": "DunhuangPairwiseV02",
            "matcher_mode": "dustbin_sinkhorn",
            "matcher_temperature": model_config.local.matcher_temperature,
            "sinkhorn_iterations": model_config.local.sinkhorn_iterations,
            "sinkhorn_tolerance": model_config.local.sinkhorn_tolerance,
            "require_sinkhorn_convergence": True,
            "direction_hypotheses": list(DIRECTION_NAMES),
            "optimizer": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": 0.0,
            "overfit_steps": steps,
            "stability_smoke_override": True,
            "default_temperature_0_1_iterations_50_robustness_established": False,
            "override_reason": "strict_convergence_failed_during_initial_50_iteration_cuda_attempt",
        },
        "batch_aggregate": {
            "pair_count": batch.batch_size,
            "positive_count": int(batch.labels.sum().item()),
            "negative_count": int((~batch.labels).sum().item()),
            "all_four_directions_per_pair": bool(
                batch.direction_slot_valid.all().item()
            ),
            "arc_candidate_count": candidate_count,
            "direction_candidate_counts": direction_counts,
            "sequence_a_length": {
                "minimum": int(sequence_a.min().item()),
                "maximum": int(sequence_a.max().item()),
                "mean": float(sequence_a.mean().item()),
            },
            "sequence_b_length": {
                "minimum": int(sequence_b.min().item()),
                "maximum": int(sequence_b.max().item()),
                "mean": float(sequence_b.mean().item()),
            },
            "patch_channels": int(batch.local_a.shape[2]),
            "patch_height": int(batch.local_a.shape[3]),
            "patch_width": int(batch.local_a.shape[4]),
            "lazy_loader": {
                "requests": int(loader_stats["requests"]),
                "cache_hits": int(loader_stats["cache_hits"]),
                "cache_misses": int(loader_stats["cache_misses"]),
                "decoded_masks": int(loader_stats["decoded_masks"]),
                "archive_opens": int(loader_stats["archive_opens"]),
            },
        },
        "execution": {
            "timings_seconds": {
                "hash_verification": hash_seconds,
                "record_selection": selection_seconds,
                "geometry_preprocessing": preprocess_seconds,
                "device_transfer": transfer_seconds,
                "initial_evaluation": initial_seconds,
                "training": training_seconds,
                "final_evaluation": final_seconds,
                "total": total_seconds,
            },
            "loss": _summary(loss_values),
            "gradient_norm": _summary(gradient_norms),
            "sinkhorn": _aggregate_transport(observations),
        },
        "results": {
            "finite_loss": True,
            "finite_nonzero_gradients": True,
            "sinkhorn_all_converged": True,
            "all_pair_outputs_valid": True,
            "final_loss_lower_than_initial": True,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "absolute_loss_reduction": initial_loss - final_loss,
            "relative_loss_reduction": (initial_loss - final_loss)
            / max(abs(initial_loss), 1e-12),
        },
    }
    validate_receipt(receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = run_cuda_smoke(
        manifest_path=arguments.manifest,
        archive_path=arguments.archive,
        source_root=arguments.source_root,
        seed=arguments.seed,
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
    )
    _write_receipt(arguments.receipt, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "schema_version": receipt["schema_version"],
                "final_loss_lower_than_initial": receipt["results"][
                    "final_loss_lower_than_initial"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CudaSmokeError",
    "EXPECTED_ARCHIVE_SHA256",
    "EXPECTED_MANIFEST_SHA256",
    "SMOKE_SCHEMA_VERSION",
    "run_cuda_smoke",
    "validate_receipt",
]
