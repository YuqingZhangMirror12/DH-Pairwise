"""Decode frozen four-model OOD layouts for visual review, without layout GT.

The existing OOD classification files are never modified. Every eligible pair
is decoded, including pairs rejected by either frozen classification threshold.
Historical E1 and Full24 retain their established full_top2_mode decoder;
PairingNet and ShreddingNet retain their own benchmark translation decoders.
This entry point does not train, select a model, refit a threshold, or rescue a
classification decision. Re-forward scores are explicitly non-canonical.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from experiments.rachel_n512_formal_30k.evaluate_turufan_ood import (
    METHODS, digest, freeze_specs, load_model, read_json, save_json,
)

SCHEMA = "turufan-ood-frozen-layout-review/1"
NUMERIC_FAILURES = (FloatingPointError, np.linalg.LinAlgError)


def json_safe(value):
    """Retain invalid diagnostics as null, not nonstandard JSON NaN values."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def layout_row(pair_id, decoder, translation, valid, *, reason, diagnostics=None,
               score=None, decision_valid=None):
    """Canonical coordinate contract: B's displayed offset is minus b-a."""
    valid = bool(valid)
    t = np.asarray(translation, dtype=float) if translation is not None else None
    if valid and (t is None or t.shape != (2,) or not np.isfinite(t).all()):
        raise ValueError("valid native translation must be finite [row,column]")
    if score is not None and (not np.isfinite(score) or not 0 <= score <= 1):
        raise ValueError("invalid re-forward probability")
    return json_safe(dict(pair_id=pair_id, decoder=decoder, valid=valid,
        t_a_to_b_rc=t if valid else None, offset_b_in_a_rc=-t if valid else None,
        reason=str(reason), diagnostics=diagnostics or {},
        reforward_score=None if score is None else float(score),
        reforward_decision_valid=None if decision_valid is None else bool(decision_valid),
        score_is_canonical=False, layout_decoded_independently_of_pair_probability=True))


def make_batch(pairs, indices, arrays):
    from staging.pairwise_v0_2.baselines.rachel_same_data_benchmark_eval_adapter import build_target_blind_rachel_batch
    a, b = indices[:, 0], indices[:, 1]
    return build_target_blind_rachel_batch(
        pair_ids=[p["pair_id"] for p in pairs],
        fragment_a_tokens=[p["fragment_a_id"] for p in pairs],
        fragment_b_tokens=[p["fragment_b_id"] for p in pairs],
        masks_a=np.unpackbits(arrays["packed_masks"][a], axis=2).astype(bool),
        masks_b=np.unpackbits(arrays["packed_masks"][b], axis=2).astype(bool),
        points_rc_a=arrays["points"][a], points_rc_b=arrays["points"][b],
        contour_valid_a=arrays["valid"][a], contour_valid_b=arrays["valid"][b])


def predict_full(model, batch, device):
    from experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint import DECODER_NAME, TOP2_CONFIG
    from staging.pairwise_v0_2.models.translation_layout import estimate_translation_layout
    names = ("mask_a", "mask_b", "points_rc_a", "points_rc_b", "contour_valid_a", "contour_valid_b")
    tensors = [torch.as_tensor(getattr(batch, name), device=device,
        dtype=torch.bool if name.startswith("contour_valid") else torch.float32) for name in names]
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = model(*tensors)
    assignment = output.assignment.detach().float().cpu().numpy()
    scores = output.fused_probability.detach().float().cpu().numpy()
    decision = output.decision_valid.detach().cpu().numpy()
    count = len(batch.pair_ids)
    if assignment.shape != (count, 512, 512) or scores.shape != (count,) or decision.shape != (count,):
        raise ValueError("Full output shape differs from the prepared input batch")
    rows = []
    for i, pair_id in enumerate(batch.pair_ids):
        evidence = dict(score=float(scores[i]), decision_valid=bool(decision[i]))
        va, vb = batch.contour_valid_a[i], batch.contour_valid_b[i]
        if not np.isfinite(assignment[i][np.ix_(va, vb)]).all():
            rows.append(layout_row(pair_id, DECODER_NAME, None, False,
                reason="nonfinite_predicted_assignment", **evidence))
            continue
        try:
            estimate = estimate_translation_layout(batch.points_rc_a[i], batch.points_rc_b[i],
                assignment[i], va, vb, config=TOP2_CONFIG)
        except NUMERIC_FAILURES as error:
            rows.append(layout_row(pair_id, DECODER_NAME, None, False,
                reason="decoder_numeric_failure", diagnostics=dict(error=repr(error)), **evidence))
            continue
        diagnostics = asdict(estimate)
        for key in ("t_a_to_b_rc", "candidate_indices", "inlier_mask", "valid", "reason"):
            diagnostics.pop(key, None)
        rows.append(layout_row(pair_id, DECODER_NAME, estimate.t_a_to_b_rc, estimate.valid,
            reason=estimate.reason, diagnostics=diagnostics, **evidence))
    return rows


def predict_benchmark(benchmark, batch):
    """Use the established target-blind adapter, including rejected pairs."""
    from experiments.rachel_n512_formal_30k.evaluate_recall_benchmarks import DECODERS
    output = benchmark.predict_batch(batch)
    if tuple(output.pair_ids) != tuple(batch.pair_ids):
        raise ValueError("benchmark prediction identity/order differs")
    rows = []
    for i, pair_id in enumerate(batch.pair_ids):
        valid = bool(output.translation_valid[i])
        rows.append(layout_row(pair_id, DECODERS[benchmark.method],
            output.translation_hat_rc[i], valid,
            reason="native_decoder_valid" if valid else "native_decoder_invalid",
            diagnostics=dict(auxiliary_scores={k: float(v[i]) for k, v in output.auxiliary_scores.items()},
                note="Original common adapter does not expose detailed native failure reason/counts"),
            score=float(output.pair_probability[i]), decision_valid=bool(output.decision_valid[i])))
    return rows


def predict_part(name, model, pairs, indices, arrays, device):
    batch = make_batch(pairs, indices, arrays)
    if name in ("historical_e1", "full_e1_24k"):
        return predict_full(model, batch, device)
    try:
        return predict_benchmark(model, batch)
    except NUMERIC_FAILURES as error:
        # Isolate an exceptional numerical pose failure to its pair. Structural
        # ValueError/TypeError/RuntimeError (including OOM) remain top-level bugs.
        if len(pairs) > 1:
            rows = []
            for i in range(len(pairs)):
                rows.extend(predict_part(name, model, pairs[i:i+1], indices[i:i+1], arrays, device))
            return rows
        return [layout_row(pairs[0]["pair_id"], name + "_native_translation_consensus",
            None, False, reason="decoder_numeric_failure", diagnostics=dict(error=repr(error)))]


def validate_completed(payload, name, expected_ids, binding):
    if payload.get("status") != "complete" or payload.get("method") != name or payload.get("binding") != binding:
        raise ValueError("existing completed output differs from this frozen run")
    rows = payload.get("predictions", [])
    if [r.get("pair_id") for r in rows] != expected_ids:
        raise ValueError("existing output does not retain the exact eligible pair order")
    for row in rows:
        checked = layout_row(row["pair_id"], row["decoder"], row["t_a_to_b_rc"], row["valid"], reason=row["reason"])
        if checked["offset_b_in_a_rc"] != row["offset_b_in_a_rc"]:
            raise ValueError("saved placement sign differs from translation convention")
    return rows


def run(args):
    from staging.pairwise_v0_2.training.rachel_n512_sealed_test import _set_determinism
    from experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint import TOP2_CONFIG
    from experiments.rachel_n512_formal_30k.evaluate_recall_benchmarks import load_experimental_benchmark
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch-size and optional limit must be positive")
    torch.set_num_threads(1)
    _set_determinism(260911)
    config = read_json(args.config)
    specs = freeze_specs(config)
    prepared = Path(args.prepared).resolve(strict=True)
    metadata = read_json(prepared / "manifest.json")
    ids, all_pairs = metadata["fragment_ids"], metadata["pairs"]
    if (len(all_pairs) != config["expected_positive_count"] or len(ids) != 2 * len(all_pairs)
            or len(set(ids)) != len(ids) or len({p["pair_id"] for p in all_pairs}) != len(all_pairs)):
        raise ValueError("prepared inputs differ from the fixed two-fragment inventory")
    pairs = all_pairs if args.limit is None else all_pairs[:args.limit]
    expected_ids = [p["pair_id"] for p in pairs]
    lookup = {fragment_id: i for i, fragment_id in enumerate(ids)}
    indices = np.asarray([[lookup[p["fragment_a_id"]], lookup[p["fragment_b_id"]]] for p in pairs])
    with np.load(prepared / "inputs.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("packed_masks", "points", "valid")}
    n = len(ids)
    if (arrays["packed_masks"].shape != (n, 800, 100) or arrays["points"].shape != (n, 512, 2)
            or arrays["valid"].shape != (n, 512) or not np.isfinite(arrays["points"]).all()):
        raise ValueError("prepared input shape/finite contract differs")
    binding = dict(config_sha256=digest(args.config), prepared_manifest_sha256=digest(prepared / "manifest.json"),
        prepared_inputs_sha256=digest(prepared / "inputs.npz"), script_sha256=digest(__file__),
        methods=list(METHODS), pair_ids=expected_ids, batch_size=args.batch_size,
        frozen_threshold_sha256={name: specs[name]["threshold_file_sha256"] for name in METHODS})
    root = Path(args.output).resolve()
    if root == prepared or root.name == "evaluation_v1":
        raise ValueError("layout outputs must be separate from inputs and original classification evaluation")
    root.mkdir(parents=True, exist_ok=True)
    frozen_path = root / "frozen_models.json"
    if frozen_path.exists():
        if read_json(frozen_path).get("binding") != binding:
            raise ValueError("resume requires the identical prepared cohort, sources and options")
    else:
        save_json(frozen_path, dict(binding=binding, methods=specs))
    protocol = dict(schema_version=SCHEMA, status="running", binding=binding,
        sample_count=len(pairs), full_eligible_count=len(all_pairs), smoke_test=args.limit is not None,
        no_layout_gt=True, no_layout_accuracy_reported=True, training_performed=False,
        classification_files_changed=False, classification_scores_canonical=False,
        classification_policy="Original evaluation_v1 scores/thresholds remain authoritative; new scores are diagnostics only",
        all_pairs_decoded=True, pair_score_gating=False, overlap_repair=False, rotation_estimated=False,
        pose_policy="Each model's existing decoder; no new decoder or fusion selection",
        full_decoder="full_top2_mode", full_decoder_config=asdict(TOP2_CONFIG),
        historical_decoder_source="run_edge_weathering_pipeline -> evaluate_realism_checkpoint",
        full24_decoder_source="evaluate_recall_data_volume -> evaluate_realism_checkpoint",
        benchmark_decoder_source="ExperimentalBenchmark.predict_batch (compute_pose_for_rejected=True)",
        translation_convention="t_a_to_b_rc=b-a; B offset in A canvas=-t_a_to_b_rc; prepared800 pixel units",
        secondary_pose_note="ShreddingNet's unchanged adapter internally computes an unused secondary SE2 result; only its translation-only result is exported")
    save_json(root / "protocol.json", protocol)
    device = torch.device(args.device)
    results = {}
    for name in METHODS:
        path = root / (name + "_layouts.json")
        if path.exists():
            saved = read_json(path)
            rows = validate_completed(saved, name, expected_ids, binding)
            results[name] = dict(count=len(rows), valid_count=sum(r["valid"] for r in rows), resumed=True,
                predictions_sha256=digest(path))
            continue
        save_json(root / "status.json", dict(status="running", active_method=name, completed=list(results)))
        started = time.monotonic()
        if name in ("historical_e1", "full_e1_24k"):
            model, identity = load_model(name, specs[name], device)
        else:
            model = load_experimental_benchmark(name, Path(specs[name]["training_run"]), device)
            identity = model.identity
            if identity["checkpoint_sha256_by_stage"] != specs[name]["threshold_source"]["model_identity"]["checkpoint_sha256_by_stage"]:
                raise ValueError("benchmark weights differ from original OOD frozen weights")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rows = []
        for start in range(0, len(pairs), args.batch_size):
            part = pairs[start:start + args.batch_size]
            output = predict_part(name, model, part, indices[start:start+len(part)], arrays, device)
            if [r["pair_id"] for r in output] != [p["pair_id"] for p in part]:
                raise ValueError("layout outputs dropped or reordered a pair")
            for row, pair in zip(output, part):
                row.update(prefix=pair.get("prefix", pair["pair_id"]), fragment_a_id=pair["fragment_a_id"],
                    fragment_b_id=pair["fragment_b_id"])
            rows.extend(output)
            if len(rows) % 32 == 0 or len(rows) == len(pairs):
                save_json(root / "status.json", dict(status="running", active_method=name,
                    processed=len(rows), count=len(pairs), completed=list(results)))
        if [r["pair_id"] for r in rows] != expected_ids:
            raise ValueError("method output does not cover every requested OOD pair")
        elapsed = time.monotonic() - started
        save_json(path, dict(schema_version=SCHEMA, status="complete", method=name, binding=binding,
            identity=identity, no_layout_gt=True, predictions=rows, elapsed_s=elapsed,
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None))
        results[name] = dict(count=len(rows), valid_count=sum(r["valid"] for r in rows),
            invalid_count=sum(not r["valid"] for r in rows), elapsed_s=elapsed, predictions_sha256=digest(path))
        print(json.dumps(dict(method=name, status="complete", **results[name])), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_json(root / "summary.json", dict(schema_version=SCHEMA, status="complete", methods=results,
        sample_count=len(pairs), full_eligible_count=len(all_pairs), no_layout_gt=True,
        no_layout_accuracy_reported=True, classification_files_changed=False))
    protocol["status"] = "complete"
    save_json(root / "protocol.json", protocol)
    save_json(root / "completion_receipt.json", dict(status="complete", methods=list(results),
        sample_count=len(pairs), summary_sha256=digest(root / "summary.json"),
        protocol_sha256=digest(root / "protocol.json")))
    save_json(root / "status.json", dict(status="complete", completed=list(results)))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, help="First N pairs only; use a separate smoke output directory")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if Path(args.output).is_dir():
            save_json(Path(args.output) / "failure.json", dict(status="failed", error=repr(error)))
        raise
