"""CPU-only six-way assignment ablation from a completed frozen matrix cache.

The classifier and its three original thresholds are never fitted here. Only
VAL may select a geometry decoder. TEST/REAL require that VAL receipt and the
same checkpoint, precision, thresholds and predeclared decoder configurations.
Predictions are closed/fsynced before even reading cached labels/targets; an
optional REAL translation file is also opened only after this prediction freeze.

Workers receive model outputs, points and input metadata, never supervision.
At most one cache chunk is queued at a time. No model, image or GPU is opened.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import time

# Each worker has its own small SciPy/NumPy decode. Avoid nested BLAS pools.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"
import numpy as np
from scipy.special import expit

from staging.pairwise_v0_2.models.assignment_layout_ablation import (
    assignment_ablation_configs, estimate_assignment_layout,
)

CACHE_SCHEMA = "rachel-matrix-pair-cache/v1"
FREEZE_SCHEMA = "rachel-cached-assignment-freeze/v1"
BRANCHES = ("coarse", "local", "fused")
INPUT_ARRAYS = ("real_transport", "points_a_rc", "points_b_rc", "valid_a", "valid_b",
                "unmatched_a", "unmatched_b", "pair_id", "coarse_logit", "local_logit", "fused_logit")
METADATA_ARRAYS = ("fragment_a", "fragment_b", "case_id", "strict_member")
PROBABILITY_ARRAYS = tuple(branch + "_probability" for branch in BRANCHES)


def _clean(value):
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    return value


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_line(stream, row):
    stream.write(json.dumps(_clean(row), ensure_ascii=False, allow_nan=False) + "\n")


def _thresholds(value):
    if not isinstance(value, dict) or set(value) != set(BRANCHES):
        raise ValueError("cache must provide original_thresholds for coarse/local/fused")
    result = {name: float(value[name]) for name in BRANCHES}
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in result.values()):
        raise ValueError("original probability thresholds must be finite in [0,1]")
    return result


def _load_cache(path, requested_split):
    path = Path(path)
    path = path / "manifest.json" if path.is_dir() else path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("a complete rachel-matrix-pair-cache/v1 manifest is required")
    split = manifest.get("split")
    if split not in ("val", "test", "real") or (requested_split and split != requested_split):
        raise ValueError("cache split must match the requested val/test/real split")
    if not isinstance(manifest.get("probe_only"), bool):
        raise ValueError("cache must explicitly declare probe_only")
    for key in ("matcher_checkpoint_id", "precision"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError("cache is missing " + key)
    chunks = manifest.get("chunks", [])
    if (not chunks or any(not isinstance(c, dict) or not isinstance(c.get("path"), str)
                          or not isinstance(c.get("sample_count"), int) or c["sample_count"] <= 0 for c in chunks)
            or len({c["path"] for c in chunks}) != len(chunks)
            or sum(c["sample_count"] for c in chunks) != manifest.get("sample_count")):
        raise ValueError("cache chunk counts/paths do not describe its complete population")
    return path, manifest, _thresholds(manifest["original_thresholds"])


def _load_freeze(path, manifest, thresholds):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("schema_version") != FREEZE_SCHEMA or value.get("source_split") != "validation"
            or value.get("probe_only") is not False or value.get("test_or_real_used_for_fit") is not False
            or value.get("classifier_threshold_fitted") is not False):
        raise ValueError("TEST/REAL require a non-probe validation-only assignment freeze")
    for key in ("matcher_checkpoint_id", "precision"):
        if value.get(key) != manifest[key]:
            raise ValueError("validation freeze and cache differ in " + key)
    if _thresholds(value.get("original_thresholds")) != thresholds:
        raise ValueError("validation freeze and cache have different original thresholds")
    expected = {name: asdict(config) for name, config in assignment_ablation_configs().items()}
    if value.get("decoders") != expected or value.get("selected_full_decoder") not in expected:
        raise ValueError("validation freeze does not contain the six predeclared decoders")
    return value


def _decode(task):
    """Only inference arrays and metadata enter this worker."""
    row = {"pair_id": str(task["pair_id"]), "classification": {}, "layouts": {}}
    for key in METADATA_ARRAYS:
        if key in task:
            row[key] = bool(task[key]) if key == "strict_member" else str(task[key])
    preserved = all(key in task for key in PROBABILITY_ARRAYS)
    row["classification_source"] = ("cached_model_probabilities" if preserved else
                                    "float32_sigmoid_reconstruction_not_bitwise_guaranteed")
    for branch in BRANCHES:
        logit = float(task[branch + "_logit"])
        if not np.isfinite(logit):
            raise ValueError("nonfinite cached classifier logit for " + row["pair_id"])
        probability = float(task[branch + "_probability"]) if preserved else float(expit(np.float32(logit)))
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("cached classifier probability must be finite in [0,1]")
        row["classification"][branch] = probability
    for name, config in assignment_ablation_configs().items():
        estimate = estimate_assignment_layout(
            task["points_a_rc"], task["points_b_rc"], task["real_transport"],
            task["valid_a"], task["valid_b"], unmatched_a=task["unmatched_a"],
            unmatched_b=task["unmatched_b"], config=config,
        )
        diagnostics = asdict(estimate)
        for key in ("candidate_indices", "inlier_mask", "t_a_to_b_rc", "valid"):
            diagnostics.pop(key)
        row["layouts"][name] = dict(
            translation_rc=estimate.t_a_to_b_rc, offset_b_in_a_rc=-estimate.t_a_to_b_rc,
            valid=bool(estimate.valid), diagnostics=diagnostics,
        )
    return _clean(row)


def _input_chunk(path, count):
    with np.load(path, allow_pickle=False) as archive:
        # NPZ arrays are lazy: never index label, target_translation_rc or target_a/b here.
        arrays = {key: archive[key] for key in INPUT_ARRAYS}
        arrays.update({key: archive[key] for key in METADATA_ARRAYS if key in archive.files})
        present = [key for key in PROBABILITY_ARRAYS if key in archive.files]
        if present and len(present) != len(PROBABILITY_ARRAYS):
            raise ValueError("cache must contain all three optional model probability arrays or none")
        arrays.update({key: archive[key] for key in present})
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("cached input array length differs from chunk manifest")
    return ({key: value[i] for key, value in arrays.items()} for i in range(count))


def _attach_targets(rows, manifest_path, manifest, real_gt):
    """Caller must have durably frozen ALL predictions before invoking this."""
    position = 0
    for chunk in manifest["chunks"]:
        with np.load(manifest_path.parent / chunk["path"], allow_pickle=False) as archive:
            ids, labels = archive["pair_id"], archive["label"]
            targets = None if real_gt else archive["target_translation_rc"]
        n = chunk["sample_count"]
        if (ids.shape != (n,) or labels.shape != (n,) or not np.isin(labels, (0, 1)).all()
                or (targets is not None and targets.shape != (n, 2))):
            raise ValueError("target array shape/labels differ from cache population")
        for i in range(n):
            row = rows[position + i]
            if row["pair_id"] != str(ids[i]):
                raise ValueError("target pair IDs are not aligned with predictions")
            row["label"] = bool(labels[i])
            row["target_translation_rc"] = None if targets is None or not row["label"] else targets[i]
        position += n
    if real_gt:
        targets = json.loads(Path(real_gt).read_text(encoding="utf-8"))["positive_pairs"]
        by_id = {row["pair_id"]: row for row in targets}
        positive_ids = {row["pair_id"] for row in rows if row["label"]}
        if len(targets) != 508 or len(by_id) != 508 or set(by_id) != positive_ids:
            raise ValueError("REAL GT must match exactly the cached population's 508 positive pair IDs")
        for row in rows:
            target = by_id.get(row["pair_id"])
            if target is None:
                continue
            for side in ("a", "b"):
                if row.get("fragment_" + side) != target["fragment_" + side + "_token"]:
                    raise ValueError("REAL GT ordered endpoints differ for " + row["pair_id"])
            row["target_translation_rc"] = target["translation_gt_a_to_b_rc"]
    for row in rows:
        gt = np.asarray(row["target_translation_rc"], dtype=np.float64) if row["label"] else None
        if row["label"] and (gt.shape != (2,) or not np.isfinite(gt).all()):
            raise ValueError("positive target translation is unavailable; REAL NaN caches require --real-gt")
        row["target_translation_rc"] = _clean(gt)
        for layout in row["layouts"].values():
            layout["translation_l2_px"] = (float(np.linalg.norm(np.asarray(layout["translation_rc"]) - gt))
                                           if gt is not None and layout["valid"] else None)


def _summarize(rows, thresholds):
    # Import the old Torch-dependent metrics helper only in the parent, after
    # the CPU pool has shut down. No model/device is constructed by this import.
    from experiments.rachel_n512_formal_30k import run_layout_decoder_experiment as common
    summary = common.summarize(rows, thresholds["fused"])
    labels = [row["label"] for row in rows]
    for branch in BRANCHES:
        summary["classification"][branch]["at_original_frozen_threshold"] = common.classification(
            labels, [row["classification"][branch] for row in rows], thresholds[branch])
    for name, metrics in summary["layout"].items():
        positive = [row["layouts"][name] for row in rows if row["label"]]
        metrics["positive_valid_pose_count"] = sum(layout["valid"] for layout in positive)
        metrics["recall_counts"] = {str(tolerance): sum(
            layout["valid"] and layout["translation_l2_px"] is not None
            and layout["translation_l2_px"] <= tolerance for layout in positive) for tolerance in (2, 5, 8, 10)}
        metrics["invalid_reason_counts"] = {}
        for row in rows:
            layout = row["layouts"][name]
            if not layout["valid"]:
                reason = layout["diagnostics"]["reason"]
                metrics["invalid_reason_counts"][reason] = metrics["invalid_reason_counts"].get(reason, 0) + 1
    return summary


def _choose_decoder(summary):
    # Preserve the existing Top2 control on an exact three-metric tie.
    names = ["top2_mode"] + [name for name in assignment_ablation_configs() if name != "top2_mode"]
    def key(name):
        metrics = summary["layout"][name]
        p90 = metrics["p90_px_conditional"]
        return (float(np.mean(list(metrics["recall"].values()))), metrics["assembly"]["10"]["f1"],
                -p90 if p90 is not None else -np.inf)
    return max(names, key=key)


def run(args):
    if args.workers < 1:
        raise ValueError("workers must be positive")
    path, manifest, thresholds = _load_cache(args.cache, args.split)
    split = manifest["split"]
    if split != "val" and not args.freeze:
        raise ValueError("TEST/REAL require --freeze from VAL")
    if args.real_gt and split != "real":
        raise ValueError("--real-gt is only applicable to REAL")
    authority = _load_freeze(args.freeze, manifest, thresholds) if args.freeze else None
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    started, rows, seen = time.perf_counter(), [], set()
    pool_context = (ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))
                    if args.workers > 1 else nullcontext(None))
    with pool_context as pool, (destination / "pair_predictions.jsonl").open("x", encoding="utf-8") as stream:
        for chunk in manifest["chunks"]:
            tasks = _input_chunk(path.parent / chunk["path"], chunk["sample_count"])
            results = pool.map(_decode, tasks, chunksize=1) if pool else map(_decode, tasks)
            for row in results:
                if not row["pair_id"] or row["pair_id"] in seen:
                    raise ValueError("cache contains an empty or duplicate pair ID")
                seen.add(row["pair_id"])
                rows.append(row)
                _write_line(stream, row)
            stream.flush()
            print(json.dumps(dict(processed=len(rows), total=manifest["sample_count"],
                                  elapsed_s=round(time.perf_counter() - started, 2))), flush=True)
        os.fsync(stream.fileno())
    if len(rows) != manifest["sample_count"]:
        raise ValueError("prediction count differs from complete cache")
    _write_json(destination / "prediction_complete.json", dict(
        status="complete", sample_count=len(rows), translation_gt_opened=False, cached_targets_opened=False,
        prediction_file="pair_predictions.jsonl", matcher_checkpoint_id=manifest["matcher_checkpoint_id"],
    ))
    _attach_targets(rows, path, manifest, args.real_gt)
    if not any(row["label"] for row in rows):
        raise ValueError("pose comparison/selection requires positive reference pairs")
    with (destination / "pair_results.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            _write_line(stream, row)
        stream.flush()
        os.fsync(stream.fileno())
    summary = _summarize(rows, thresholds)
    probability_sources = sorted({row["classification_source"] for row in rows})
    original_probabilities_preserved = probability_sources == ["cached_model_probabilities"]
    if authority is None:
        authority = dict(
            schema_version=FREEZE_SCHEMA, source_split="validation", sample_count=len(rows),
            source_cache=str(path.resolve()), matcher_checkpoint_id=manifest["matcher_checkpoint_id"],
            precision=manifest["precision"], original_thresholds=thresholds,
            original_fused_threshold=thresholds["fused"],
            decoders={name: asdict(config) for name, config in assignment_ablation_configs().items()},
            selected_full_decoder=_choose_decoder(summary), probe_only=manifest["probe_only"],
            selection_rule="maximize mean unconditional R@2/5/8/10, then joint F1@10, then minimize conditional P90; exact ties prefer Top2",
            classifier_threshold_fitted=False, test_or_real_used_for_fit=False,
            classification_probability_sources=probability_sources,
        )
        _write_json(destination / "validation_freeze.json", authority)
    summary.update(status="complete", split=split, selected_full_decoder=authority["selected_full_decoder"],
                   probe_only=manifest["probe_only"], original_thresholds=thresholds,
                   classification_shared_across_decoders=True, classifier_threshold_fitted=False,
                   original_probabilities_preserved=original_probabilities_preserved,
                   classification_probability_sources=probability_sources,
                   target_gt_evaluation_after_prediction_freeze=True,
                   real_seam_ground_truth_available=False if split == "real" else None)
    if split == "real" and all("strict_member" in row for row in rows):
        strict = [row for row in rows if row["strict_member"]]
        if len(strict) != 547 or sum(row["label"] for row in strict) != 508:
            raise ValueError("cached strict REAL membership is not the 547/508 population")
        summary["strict_summary"] = _summarize(strict, thresholds)
        summary["strict_summary"]["selected_full_decoder"] = authority["selected_full_decoder"]
        _write_json(destination / "strict547_summary.json", summary["strict_summary"])
    summary["elapsed_s"] = time.perf_counter() - started
    protocol = dict(
        status="complete", split=split, cache_manifest=str(path.resolve()), sample_count=len(rows),
        matcher_checkpoint_id=manifest["matcher_checkpoint_id"], precision=manifest["precision"],
        original_thresholds=thresholds, decoders=authority["decoders"],
        validation_freeze=str(Path(args.freeze).resolve()) if args.freeze else "validation_freeze.json",
        selected_full_decoder=authority["selected_full_decoder"], workers=args.workers,
        maximum_queued_cache_chunks=1, model_executed=False, gpu_used=False,
        classifier_modified=False, classifier_threshold_fitted=False, rotation_estimated=False,
        classification_shared_across_decoders=True,
        original_probabilities_preserved=original_probabilities_preserved,
        target_gt_evaluation_after_prediction_freeze=True,
        target_source=str(Path(args.real_gt).resolve()) if args.real_gt else "cached target_translation_rc",
        real_seam_ground_truth_available=False if split == "real" else None,
        classification_probability_sources=probability_sources,
        classification_probability_caveat="Use supplied model probabilities exactly when available; otherwise CPU float32 sigmoid reconstruction may differ from the original GPU result near a frozen threshold. Each pair uses identical scores for all six decoders.",
        metric_definition="existing common.summarize: invalid poses fail unconditional positive recall; joint FP includes every accepted incorrect pose/negative; median/P90 conditional on valid positive poses",
    )
    _write_json(destination / "protocol.json", protocol)
    _write_json(destination / "summary.json", summary)
    print(json.dumps(dict(status="complete", split=split, sample_count=len(rows),
                         selected_full_decoder=authority["selected_full_decoder"], elapsed_s=summary["elapsed_s"])), flush=True)
    return _clean(summary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="completed matrix cache directory or manifest.json")
    parser.add_argument("--output", required=True, help="new output directory; never overwrite prior results")
    parser.add_argument("--split", choices=("val", "test", "real"), help="optional check against cache split")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--freeze", help="own VAL validation_freeze.json; required for TEST/REAL")
    parser.add_argument("--real-gt", help="existing REAL positive_pairs translation JSON, opened after prediction freeze")
    return parser.parse_args(argv)


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
