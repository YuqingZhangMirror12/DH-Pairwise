"""Frozen four-model, positive-only pairability evaluation; no layout GT.

Inputs contain masks/contours and named frag1->frag2 pairs only. No fitting,
Top-K, geometric pose decoding, negative synthesis or model selection occurs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

METHODS = ("historical_e1", "full_e1_24k", "pairingnet", "shreddingnet")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def positive_metrics(scores, thresholds, valid=None):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("requires one finite probability for every eligible positive pair")
    validity = np.ones(len(values), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if validity.shape != values.shape:
        raise ValueError("decision validity must retain every eligible pair")
    result = {}
    for policy, t in thresholds.items():
        if not np.isfinite(t) or not 0 <= t <= 1:
            raise ValueError("invalid frozen threshold")
        tp = int(np.count_nonzero(validity & (values >= float(t))))
        result[policy] = dict(threshold=float(t), positive_count=len(values), tp=tp,
                              fn=len(values)-tp, recall=tp/len(values), invalid_decisions=int((~validity).sum()))
    return result


def freeze_specs(config):
    if set(config["methods"]) != set(METHODS):
        raise ValueError("requires exactly the four user-requested models")
    frozen = {}
    for name in METHODS:
        spec = dict(config["methods"][name])
        p = Path(spec["threshold_file"])
        source = read_json(p)
        if name == "historical_e1":
            if source.get("threshold_fit_split") != "val" or source.get("real_or_test_opened_for_fit") is not False:
                raise ValueError("historical E1 thresholds are not the saved VAL freeze")
            thresholds = source["thresholds"]
        elif name == "full_e1_24k":
            if source.get("status") != "complete" or source.get("test_or_real_used_for_fit") is not False:
                raise ValueError("Full24k requires completed training freeze")
            thresholds = source["operating_points"]["thresholds"]
        else:
            if (source.get("status") != "complete_validation_frozen" or source.get("source_split") != "val"
                    or source.get("test_or_real_used_for_fit") is not False):
                raise ValueError("benchmark requires its existing VAL-only threshold freeze")
            thresholds = source["thresholds"]
        recall_first = (thresholds["recall_first"] if "recall_first" in thresholds
                        else min(thresholds["max_f1"], thresholds["recall_99"]))
        selected = dict(max_f1=float(thresholds["max_f1"]), recall_first=float(recall_first))
        positive_metrics([0.5], selected)
        frozen[name] = dict(**spec, thresholds=selected, threshold_file_sha256=digest(p),
                            threshold_source=source, score_direction="frag1 to frag2")
    return frozen


def load_model(name, spec, device):
    if name == "historical_e1":
        from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint
        from staging.pairwise_v0_2.training.rachel_n512_sealed_test import _torch_load_checkpoint
        path = Path(spec["training_run"]) / "winner.pt"
        freeze = read_json(path.parent / "train_val_freeze.json")
        sha = digest(path)
        if freeze.get("status") != "complete" or sha != spec["expected_checkpoint_sha256"]:
            raise ValueError("historical E1 winner identity differs")
        if freeze.get("checkpoint_sha256", sha) != sha:
            raise ValueError("historical E1 completion binding differs")
        model = load_rachel_checkpoint(_torch_load_checkpoint(path)).to(device).eval().requires_grad_(False)
        return model, dict(checkpoint_sha256=sha, precision="fp32", score_family="original fused probability")
    if name == "full_e1_24k":
        from experiments.rachel_n512_formal_30k.evaluate_recall_data_volume import load_winner
        model, identity, _ = load_winner(spec["training_run"])
        return model.to(device).eval().requires_grad_(False), identity
    from experiments.rachel_n512_formal_30k.evaluate_recall_benchmarks import load_experimental_benchmark
    benchmark = load_experimental_benchmark(name, Path(spec["training_run"]), device)
    expected = spec["threshold_source"]["model_identity"]["checkpoint_sha256_by_stage"]
    if benchmark.identity["checkpoint_sha256_by_stage"] != expected:
        raise ValueError("benchmark weights differ from frozen thresholds")
    return benchmark._predictor, benchmark.identity


def predict(name, model, identity, tensors, device):
    if name in ("historical_e1", "full_e1_24k"):
        with torch.autocast(device_type=device.type, enabled=False):
            output = model(*tensors)
        return ({k: getattr(output, k + "_probability").detach().float().cpu().numpy()
                for k in ("coarse", "local", "fused")}, output.decision_valid.detach().cpu().numpy().astype(bool))
    if name == "pairingnet":
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and identity["precision"] == "bf16"):
            output = model(*tensors)
        values = output.pair_probability.detach().float().cpu().numpy()
        return dict(pair=values), np.ones(len(values), dtype=bool)
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda" and model.runtime.amp):
        score, _, _ = model.classify(*tensors, model.recipe.correspondence_threshold)
    values = score.detach().float().cpu().numpy()
    return dict(pair=values), np.ones(len(values), dtype=bool)


def run(args):
    from experiments.rachel_n512_formal_30k.evaluate_native_retrieval_diagnostic import pair_tensors
    from staging.pairwise_v0_2.training.rachel_n512_sealed_test import _set_determinism
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    torch.set_num_threads(1)
    _set_determinism(260911)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = read_json(args.config)
    specs = freeze_specs(config)
    save_json(root / "frozen_models.json", dict(methods=specs, fitted_on_ood=False,
        created_before_prepared_inputs_opened=True, config_sha256=digest(args.config)))
    prepared = Path(args.prepared)
    meta = read_json(prepared / "manifest.json")
    ids, pairs = meta["fragment_ids"], meta["pairs"]
    if not pairs or len(set(ids)) != len(ids) or len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("invalid OOD input identity")
    if len(pairs) != config["expected_positive_count"] or len(ids) != 2 * len(pairs):
        raise ValueError("OOD cohort differs from the fixed two-fragment inventory")
    lookup = {x: i for i, x in enumerate(ids)}
    indices = np.asarray([[lookup[p["fragment_a_id"]], lookup[p["fragment_b_id"]]] for p in pairs])
    with np.load(prepared / "inputs.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in ("packed_masks", "points", "valid")}
    n = len(ids)
    if (arrays["packed_masks"].shape != (n,800,100) or arrays["points"].shape != (n,512,2)
            or arrays["valid"].shape != (n,512) or not arrays["valid"].any(1).all()
            or not np.isfinite(arrays["points"]).all()):
        raise ValueError("invalid prepared masks/contours")
    device = torch.device(args.device)
    protocol = dict(status="running", methods=list(METHODS), positive_count=len(pairs), negative_count=0,
        input_manifest_sha256=digest(prepared / "manifest.json"), prepared_inputs_sha256=digest(prepared / "inputs.npz"),
        ground_truth="user-provided pairability only; no layout ground truth", batch_size=args.batch_size,
        input_fields=["mask_a", "mask_b", "points_a", "points_b", "valid_a", "valid_b"],
        direction="frag1 to frag2", layout_decoded=False, topk_used=False, thresholds_fitted=False,
        training_performed=False, ood_used_for_model_selection=False, script_sha256=digest(__file__))
    save_json(root / "protocol.json", protocol)
    results = {}
    for name in METHODS:
        started = time.monotonic()
        save_json(root / "status.json", dict(status="running", active_method=name, completed=list(results)))
        model, identity = load_model(name, specs[name], device)
        microbatch = min(args.batch_size, int(model.runtime.classify_microbatch)) if name == "shreddingnet" else args.batch_size
        rows = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for start in range(0, len(pairs), microbatch):
                part = indices[start:start+microbatch]
                tensors = pair_tensors(arrays, part[:,0], part[:,1], device)
                values, valid = predict(name, model, identity, tensors, device)
                score_key = "fused" if "fused" in values else "pair"
                for offset, pair in enumerate(pairs[start:start+len(part)]):
                    score = float(values[score_key][offset])
                    if not np.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("invalid OOD probability")
                    rows.append(dict(pair_id=pair["pair_id"], prefix=pair.get("prefix",pair["pair_id"]),
                        score=score, decision_valid=bool(valid[offset]), branch_scores={k:float(v[offset]) for k,v in values.items()},
                        accepted={k:bool(valid[offset]) and score >= t for k,t in specs[name]["thresholds"].items()}))
                del values, tensors
        save_json(root / (name + "_predictions.json"), dict(status="complete", method=name, predictions=rows,
            identity=identity, thresholds=specs[name]["thresholds"], labels_used_by_forward=False))
        results[name] = dict(metrics=positive_metrics([r["score"] for r in rows], specs[name]["thresholds"], [r["decision_valid"] for r in rows]),
            effective_microbatch=microbatch,
            elapsed_s=time.monotonic()-started,
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else None,
            predictions_sha256=digest(root / (name + "_predictions.json")))
        print(json.dumps(dict(method=name,status="complete",positive_count=len(rows))), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_json(root / "summary.json", dict(status="complete", positive_count=len(pairs), negative_count=0,
        methods=results, scope="known positive two-fragment OOD pair recognition; Recall only",
        unavailable_metrics=["negative-class accuracy", "specificity", "meaningful Precision/F1/AUROC/AP", "layout accuracy"],
        no_threshold_fit=True, no_training=True, no_layout_ground_truth=True))
    protocol["status"] = "complete"
    save_json(root / "protocol.json", protocol)
    save_json(root / "completion_receipt.json", dict(status="complete", methods=list(results),
        positive_count=len(pairs), summary_sha256=digest(root / "summary.json"),
        frozen_models_sha256=digest(root / "frozen_models.json"), protocol_sha256=digest(root / "protocol.json")))
    save_json(root / "status.json", dict(status="complete", completed=list(results)))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()
    try:
        run(a)
    except Exception as e:
        if Path(a.output).is_dir():
            save_json(Path(a.output)/"failure.json",dict(status="failed",error=repr(e)))
        raise
