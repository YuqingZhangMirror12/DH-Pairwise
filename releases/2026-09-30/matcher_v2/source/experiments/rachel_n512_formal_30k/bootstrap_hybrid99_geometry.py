#!/usr/bin/env python3
"""Paired geometry bootstrap for the frozen Hybrid-99 experiment.

This is a CPU-only post-prediction analysis.  It reuses the exact6 population
validators and their established resampling units, but it does not modify or
invoke an evaluator, checkpoint, threshold fitter, or model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from experiments.rachel_n512_formal_30k import paired_cluster_bootstrap as exact6
except ModuleNotFoundError:  # direct ``python path/to/script.py`` invocation
    import paired_cluster_bootstrap as exact6


SCHEMA_VERSION = "rachel-hybrid99-paired-geometry-bootstrap/1.0"
HYBRID_METHOD = "hybrid99_full_classification_shreddingnet_pose"
FULL_METHOD = "full_n512"
SHREDDING_METHOD = "shreddingnet_adapted"
METHODS = (HYBRID_METHOD, FULL_METHOD, SHREDDING_METHOD)
ROUTE_THRESHOLD = 0.011353014037013054
EXPECTED_FULL_THRESHOLD = 0.9967334270477295
EXPECTED_SHREDDING_THRESHOLD = 0.666015625
EXPECTED_FULL_VALIDATION_SCORES_SHA256 = (
    "574f127762e59f81f6f53c400a398a04b1486fb9e94c20f51da4dea0d19176d9"
)
DEFAULT_REPLICATES = 20_000
REAL_BOOTSTRAP_SEED = "rachel-real-translation-case-bootstrap-v1-fixed-20260901"


class HybridBootstrapError(ValueError):
    """A frozen-input, alignment, or metric invariant failed."""


def _read_object(path: Path, description: str) -> Mapping[str, object]:
    try:
        value = exact6._loads(path.read_text(encoding="utf-8"), description)
    except OSError as error:
        raise HybridBootstrapError(description + " is unreadable") from error
    if not isinstance(value, Mapping):
        raise HybridBootstrapError(description + " root must be an object")
    return value


def _read_jsonl(path: Path, description: str) -> Tuple[Mapping[str, object], ...]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for ordinal, line in enumerate(stream):
                if not line.strip():
                    raise HybridBootstrapError(
                        "{} has a blank line at {}".format(description, ordinal + 1)
                    )
                value = exact6._loads(line, description + " row")
                if not isinstance(value, Mapping):
                    raise HybridBootstrapError(description + " row is not an object")
                rows.append(value)
    except OSError as error:
        raise HybridBootstrapError(description + " is unreadable") from error
    if not rows:
        raise HybridBootstrapError(description + " is empty")
    return tuple(rows)


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:  # noqa: E721
        raise HybridBootstrapError(name + " must be an explicit bool")
    return bool(value)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HybridBootstrapError(name + " must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HybridBootstrapError(name + " must be finite")
    return result


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _method_geometry(
    geometry: object, pair_id: str
) -> Tuple[bool, Optional[float]]:
    if not isinstance(geometry, Mapping):
        raise HybridBootstrapError(pair_id + " geometry is missing")
    registration = geometry.get("pairingnet_style_registration")
    if not isinstance(registration, Mapping):
        raise HybridBootstrapError(pair_id + " registration is missing")
    valid = _bool(registration.get("prediction_valid"), pair_id + ".pose_valid")
    error = registration.get("translation_l2_px")
    if error is None:
        return valid, None
    result = _number(error, pair_id + ".translation_l2_px")
    if result < 0.0:
        raise HybridBootstrapError(pair_id + " translation error is negative")
    return valid, result


def _synthetic_population(
    receipt_path: Path,
) -> Tuple[exact6.LoadedEvaluation, Mapping[str, np.ndarray], Mapping[str, object]]:
    loaded = exact6.load_sealed(receipt_path)
    if loaded.source_kind != "sealed_synthetic" or not loaded.formal_evaluation:
        raise HybridBootstrapError("synthetic input is not the formal sealed3000")
    if loaded.validation_thresholds[FULL_METHOD] != EXPECTED_FULL_THRESHOLD:
        raise HybridBootstrapError("Full frozen threshold differs")
    if loaded.validation_thresholds[SHREDDING_METHOD] != EXPECTED_SHREDDING_THRESHOLD:
        raise HybridBootstrapError("ShreddingNet frozen threshold differs")

    score_file = None
    score_sha = None
    for item in loaded.input_files:
        if item.get("role") == SHREDDING_METHOD + "_pair_scores":
            score_file = Path(str(item.get("path"))).resolve(strict=True)
            score_sha = str(item.get("sha256"))
            break
    if score_file is None or _file_sha256(score_file) != score_sha:
        raise HybridBootstrapError("validated ShreddingNet score file is unavailable")
    raw_rows = _read_jsonl(score_file, "synthetic ShreddingNet scores")
    raw_by_id: Dict[str, Mapping[str, object]] = {}
    for row in raw_rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in raw_by_id:
            raise HybridBootstrapError("synthetic ShreddingNet pair IDs differ")
        raw_by_id[pair_id] = row
    if set(raw_by_id) != {row.pair_id for row in loaded.rows}:
        raise HybridBootstrapError("synthetic ShreddingNet population differs")

    labels = np.asarray([row.label for row in loaded.rows], dtype=np.bool_)
    full_score = np.asarray(
        [row.probability[FULL_METHOD] for row in loaded.rows], dtype=np.float64
    )
    full_valid = np.asarray(
        [row.valid[FULL_METHOD] for row in loaded.rows], dtype=np.bool_
    )
    shredding_score = np.asarray(
        [row.probability[SHREDDING_METHOD] for row in loaded.rows], dtype=np.float64
    )
    shredding_valid = np.asarray(
        [row.valid[SHREDDING_METHOD] for row in loaded.rows], dtype=np.bool_
    )
    route = (~full_valid) | (full_score >= ROUTE_THRESHOLD)
    pose_success = {method: np.zeros(len(loaded.rows), dtype=np.bool_) for method in METHODS}
    for index, row in enumerate(loaded.rows):
        if row.geometry is None:
            raise HybridBootstrapError(row.pair_id + " lacks Full geometry")
        full_pose_valid, full_error = _method_geometry(row.geometry, row.pair_id)
        raw = raw_by_id[row.pair_id]
        if (
            raw.get("label") is not row.label
            or raw.get("cluster_id") != row.cluster
            or tuple(raw.get("source_unit_ids", ())) != row.dependency_units
        ):
            raise HybridBootstrapError(row.pair_id + " synthetic alignment differs")
        shredding_pose_valid, shredding_error = _method_geometry(
            raw.get("geometry"), row.pair_id
        )
        if row.label:
            pose_success[FULL_METHOD][index] = bool(
                full_pose_valid and full_error is not None and full_error <= 10.0
            )
            pose_success[SHREDDING_METHOD][index] = bool(
                shredding_pose_valid
                and shredding_error is not None
                and shredding_error <= 10.0
            )
    pose_success[HYBRID_METHOD] = route & pose_success[SHREDDING_METHOD]
    predicted = {
        FULL_METHOD: full_valid & (full_score >= EXPECTED_FULL_THRESHOLD),
        SHREDDING_METHOD: shredding_valid
        & (shredding_score >= EXPECTED_SHREDDING_THRESHOLD),
    }
    predicted[HYBRID_METHOD] = predicted[FULL_METHOD].copy()
    arrays = {
        "labels": labels,
        "route": route,
        **{"predicted_" + key: value for key, value in predicted.items()},
        **{"success_" + key: value for key, value in pose_success.items()},
    }
    source = {
        "receipt": str(receipt_path),
        "receipt_sha256": _file_sha256(receipt_path),
        "shreddingnet_pair_scores": str(score_file),
        "shreddingnet_pair_scores_sha256": score_sha,
    }
    return loaded, arrays, source


def _real_population(
    pair_path: Path, translation_path: Path
) -> Tuple[Tuple[exact6.PairRow, ...], Mapping[str, np.ndarray], Mapping[str, object]]:
    loaded = exact6.load_real(pair_path)
    if loaded.source_kind != "real_balanced1016" or not loaded.formal_evaluation:
        raise HybridBootstrapError("real pair input is not formal balanced1016")
    if loaded.validation_thresholds[FULL_METHOD] != EXPECTED_FULL_THRESHOLD:
        raise HybridBootstrapError("real Full frozen threshold differs")
    if loaded.validation_thresholds[SHREDDING_METHOD] != EXPECTED_SHREDDING_THRESHOLD:
        raise HybridBootstrapError("real ShreddingNet frozen threshold differs")
    if loaded.strict_prefix_count is None:
        raise HybridBootstrapError("real strict547 prefix is missing")
    rows = loaded.rows[: loaded.strict_prefix_count]

    document = _read_object(translation_path, "real translation evaluation")
    canonical = dict(document)
    declared_content_sha = canonical.pop("content_sha256", None)
    if declared_content_sha != hashlib.sha256(_canonical_bytes(canonical)).hexdigest():
        raise HybridBootstrapError("real translation content SHA differs")
    if (
        document.get("schema_version")
        != "rachel-n512-real-translation-gt-posteval/1.0"
        or document.get("status")
        != "complete_postprediction_real_positive_translation_gt"
        or document.get("formal_evaluation") is not True
    ):
        raise HybridBootstrapError("real translation result identity differs")
    source = document.get("source_pair_only_evaluation")
    if (
        not isinstance(source, Mapping)
        or source.get("file_sha256_frozen_before_gt_open") != loaded.source_sha256
        or source.get("full_n512_validation_threshold") != EXPECTED_FULL_THRESHOLD
    ):
        raise HybridBootstrapError("real translation source binding differs")
    positive_rows = document.get("positive_pairs")
    if not isinstance(positive_rows, list):
        raise HybridBootstrapError("real positive translation rows are missing")
    positive_by_id: Dict[str, Mapping[str, object]] = {}
    for raw in positive_rows:
        if not isinstance(raw, Mapping):
            raise HybridBootstrapError("real positive translation row is malformed")
        pair_id = raw.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in positive_by_id:
            raise HybridBootstrapError("real positive pair IDs differ")
        positive_by_id[pair_id] = raw
    expected_positive = {row.pair_id for row in rows if row.label}
    if set(positive_by_id) != expected_positive:
        raise HybridBootstrapError("real translation positives differ from strict547")

    labels = np.asarray([row.label for row in rows], dtype=np.bool_)
    full_score = np.asarray(
        [row.probability[FULL_METHOD] for row in rows], dtype=np.float64
    )
    full_valid = np.asarray([row.valid[FULL_METHOD] for row in rows], dtype=np.bool_)
    shredding_score = np.asarray(
        [row.probability[SHREDDING_METHOD] for row in rows], dtype=np.float64
    )
    shredding_valid = np.asarray(
        [row.valid[SHREDDING_METHOD] for row in rows], dtype=np.bool_
    )
    route = (~full_valid) | (full_score >= ROUTE_THRESHOLD)
    pose_success = {method: np.zeros(len(rows), dtype=np.bool_) for method in METHODS}
    for index, row in enumerate(rows):
        if not row.label:
            continue
        raw = positive_by_id[row.pair_id]
        if raw.get("case_uid") != row.dependency_units[0]:
            raise HybridBootstrapError(row.pair_id + " real case alignment differs")
        for method in (FULL_METHOD, SHREDDING_METHOD):
            value = raw.get(method)
            if not isinstance(value, Mapping):
                raise HybridBootstrapError(row.pair_id + " lacks " + method)
            valid = _bool(
                value.get("translation_prediction_valid"),
                row.pair_id + "." + method + ".pose_valid",
            )
            error = value.get("translation_l2_error_px")
            finite_error = None if error is None else _number(
                error, row.pair_id + "." + method + ".translation_l2_error_px"
            )
            pose_success[method][index] = bool(
                valid and finite_error is not None and finite_error <= 10.0
            )
    pose_success[HYBRID_METHOD] = route & pose_success[SHREDDING_METHOD]
    predicted = {
        FULL_METHOD: full_valid & (full_score >= EXPECTED_FULL_THRESHOLD),
        SHREDDING_METHOD: shredding_valid
        & (shredding_score >= EXPECTED_SHREDDING_THRESHOLD),
    }
    predicted[HYBRID_METHOD] = predicted[FULL_METHOD].copy()
    arrays = {
        "labels": labels,
        "route": route,
        **{"predicted_" + key: value for key, value in predicted.items()},
        **{"success_" + key: value for key, value in pose_success.items()},
    }
    sources = {
        "pair_only": str(pair_path),
        "pair_only_sha256": _file_sha256(pair_path),
        "translation_gt": str(translation_path),
        "translation_gt_sha256": _file_sha256(translation_path),
    }
    return tuple(rows), arrays, sources


def _point(
    weights: np.ndarray, arrays: Mapping[str, np.ndarray]
) -> Optional[Mapping[str, Mapping[str, float]]]:
    labels = arrays["labels"]
    target = float(weights[labels].sum())
    if target <= 0.0:
        return None
    output: Dict[str, Mapping[str, float]] = {}
    for method in METHODS:
        predicted = arrays["predicted_" + method]
        success = arrays["success_" + method]
        predicted_weight = float(weights[predicted].sum())
        pose_tp = float(weights[labels & success].sum())
        assembly_tp = float(weights[labels & predicted & success].sum())
        precision = assembly_tp / predicted_weight if predicted_weight else 0.0
        recall = assembly_tp / target
        assembly_f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        output[method] = {
            "r_at_10": pose_tp / target,
            "assembly_f1_at_10": assembly_f1,
        }
    return output


def _analyze_population(
    rows: Sequence[exact6.PairRow],
    arrays: Mapping[str, np.ndarray],
    *,
    replicates: int,
    rng: np.random.Generator,
    sampling_unit: str,
) -> Mapping[str, object]:
    if len(rows) != len(arrays["labels"]):
        raise HybridBootstrapError("row/array population differs")
    units, first, second = exact6._dependency_index(rows)
    point = _point(np.ones(len(rows), dtype=np.float64), arrays)
    if point is None:
        raise HybridBootstrapError("point population has no positive target")
    draws = {
        metric: {method: [] for method in METHODS}
        for metric in ("r_at_10", "assembly_f1_at_10")
    }
    deltas = {
        metric: {
            HYBRID_METHOD + "_minus_" + comparator: []
            for comparator in (FULL_METHOD, SHREDDING_METHOD)
        }
        for metric in draws
    }
    skipped = 0
    for _ in range(replicates):
        weights = exact6._pigeonhole_weights(len(units), first, second, rng)
        value = _point(weights, arrays)
        if value is None:
            skipped += 1
            continue
        for metric in draws:
            for method in METHODS:
                draws[metric][method].append(value[method][metric])
            for comparator in (FULL_METHOD, SHREDDING_METHOD):
                name = HYBRID_METHOD + "_minus_" + comparator
                deltas[metric][name].append(
                    value[HYBRID_METHOD][metric] - value[comparator][metric]
                )
    if replicates == skipped:
        raise HybridBootstrapError("all bootstrap draws lack positive targets")
    metrics = {}
    for metric in draws:
        metrics[metric] = {
            "methods": {
                method: exact6._interval(
                    draws[metric][method], point[method][metric]
                )
                for method in METHODS
            },
            "paired_deltas": {
                name: exact6._interval(
                    values,
                    point[HYBRID_METHOD][metric]
                    - point[name[len(HYBRID_METHOD + "_minus_") :]][metric],
                    delta=True,
                )
                for name, values in deltas[metric].items()
            },
        }
    labels = arrays["labels"]
    route = arrays["route"]
    return {
        "row_count": len(rows),
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
        "sampling_unit": sampling_unit,
        "sampling_dependency_unit_count": len(units),
        "sampling_pair_cluster_count_descriptive": len({row.cluster for row in rows}),
        "bootstrap": "endpoint-unit_pigeonhole_product_multiplicity",
        "replicates_requested": replicates,
        "valid_replicates": replicates - skipped,
        "skipped_no_positive_replicates": skipped,
        "route": {
            "count": int(route.sum()),
            "fraction": float(route.mean()),
            "positive_count": int((route & labels).sum()),
            "positive_recall": float((route & labels).sum() / labels.sum()),
        },
        "metrics": metrics,
    }


def analyze(
    synthetic_receipt: Path,
    real_pair_json: Path,
    real_translation_json: Path,
    *,
    replicates: int = DEFAULT_REPLICATES,
) -> Mapping[str, object]:
    if type(replicates) is not int or replicates <= 0:  # noqa: E721
        raise HybridBootstrapError("replicates must be a positive integer")
    if ROUTE_THRESHOLD >= EXPECTED_FULL_THRESHOLD:
        raise HybridBootstrapError("route threshold must cover every Full-positive pair")
    synthetic_loaded, synthetic_arrays, synthetic_sources = _synthetic_population(
        synthetic_receipt.resolve(strict=True)
    )
    real_rows, real_arrays, real_sources = _real_population(
        real_pair_json.resolve(strict=True), real_translation_json.resolve(strict=True)
    )

    seed_sequence = np.random.SeedSequence(exact6.DEFAULT_SEED)
    _, synthetic_geometry_seed = seed_sequence.spawn(2)
    real_seed_int = int.from_bytes(
        hashlib.sha256(REAL_BOOTSTRAP_SEED.encode("utf-8")).digest()[:8], "big"
    )
    result: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_frozen_hybrid99_paired_geometry_bootstrap",
        "protocol": {
            "cpu_only_post_prediction_analysis": True,
            "checkpoint_or_model_opened": False,
            "threshold_fit_performed": False,
            "test_or_real_parameter_tuning": False,
            "route_threshold": ROUTE_THRESHOLD,
            "route_rule": "Full invalid/nonfinite fail-open; otherwise fused_probability >= route_threshold",
            "full_classification_threshold": EXPECTED_FULL_THRESHOLD,
            "shreddingnet_classification_threshold": EXPECTED_SHREDDING_THRESHOLD,
            "hybrid_score_and_decision": "Full fused score and Full frozen threshold",
            "hybrid_pose": "ShreddingNet pose on routed pairs; non-routed pairs are failures",
            "r_at_10_denominator": "all authoritative positive pairs",
            "assembly_f1_at_10": (
                "threshold-accepted edge is correct only for a positive pair with "
                "valid routed pose and translation error <=10px"
            ),
            "replicates": replicates,
            "confidence_interval": "two-sided 95% percentile interval",
            "paired_delta_direction": "Hybrid-99 minus comparator",
            "synthetic_seed": exact6.DEFAULT_SEED,
            "synthetic_seed_stream": "SeedSequence(seed).spawn(2)[1], matching existing geometry stream",
            "real_seed": REAL_BOOTSTRAP_SEED,
            "full_validation_scores_sha256": EXPECTED_FULL_VALIDATION_SCORES_SHA256,
        },
        "methods": list(METHODS),
        "inputs": {
            "synthetic": synthetic_sources,
            "real": real_sources,
        },
        "synthetic": _analyze_population(
            synthetic_loaded.rows,
            synthetic_arrays,
            replicates=replicates,
            rng=np.random.default_rng(synthetic_geometry_seed),
            sampling_unit="source split-unit endpoint; two-unit rows use product multiplicity",
        ),
        "real": _analyze_population(
            real_rows,
            real_arrays,
            replicates=replicates,
            rng=np.random.default_rng(real_seed_int),
            sampling_unit="authoritative_real_case_uid; all same-case pairs move together",
        ),
    }
    result["content_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    target = path.expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise HybridBootstrapError("output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + target.name + ".tmp-", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    except FileExistsError as error:
        raise HybridBootstrapError("output already exists") from error
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-receipt", type=Path, required=True)
    parser.add_argument("--real-pair-json", type=Path, required=True)
    parser.add_argument("--real-translation-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    result = analyze(
        arguments.synthetic_receipt,
        arguments.real_pair_json,
        arguments.real_translation_json,
    )
    _write_new(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
