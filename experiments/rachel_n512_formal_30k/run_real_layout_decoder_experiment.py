"""Evaluate frozen Full pairability plus independent layout on real 1016/547.

All predictions are saved before opening real translation ground truth.  Decoder
and classifier thresholds come from a complete validation freeze.  An optional
packed-mask cache avoids repeating the several-minute real-data preparation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from experiments.rachel_n512_formal_30k import run_layout_decoder_experiment as common
from experiments.rachel_n512_formal_30k.run_full_shredding_real_selective_replay import (
    DEFAULT_REAL_AUTHORITY_ROOT, DEFAULT_REAL_CONTROL_ROOT,
)
from staging.pairwise_v0_2.baselines import rachel_n512_real_external as real_eval
from staging.pairwise_v0_2.baselines.rachel_same_data_benchmark_eval_adapter import build_target_blind_rachel_batch
from staging.pairwise_v0_2.models.translation_layout import TranslationLayoutConfig, estimate_translation_layout
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed

DEFAULT_TRANSLATION_GT = Path(
    "/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json"
)


def load_validation_freeze(path):
    authority = json.loads(Path(path).read_text(encoding="utf-8"))
    if (authority.get("source_split") != "validation" or authority.get("probe_only") is not False
            or authority.get("sample_count") != 3000 or authority.get("test_or_real_used_for_fit") is not False):
        raise ValueError("real evaluation requires a complete non-probe 3000-pair validation freeze")
    configs = {name: TranslationLayoutConfig(**value) for name, value in authority["decoders"].items()}
    if not configs or authority["selected_full_decoder"] not in configs:
        raise ValueError("validation freeze has no selected decoder")
    thresholds = authority["branch_validation_thresholds"]
    if set(thresholds) != {"coarse", "local", "fused"} or not all(np.isfinite(v) for v in thresholds.values()):
        raise ValueError("validation branch thresholds are missing or non-finite")
    return authority, configs


def prepare_inputs(args, cache):
    source = {name: str(Path(getattr(args, name)).resolve()) for name in
              ("real_manifest", "real_local_receipt", "real_main_root", "real_supp_root")}
    manifest_path, arrays_path = cache / "manifest.json", cache / "inputs.npz"
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata["source"] != source or metadata["schema"] != "real-layout-prepared-v1":
            raise ValueError("prepared cache refers to different real inputs")
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    else:
        if cache.exists() and any(cache.iterdir()):
            raise ValueError("prepared cache is incomplete; use a new cache directory")
        cache.mkdir(parents=True, exist_ok=True)
        strict = real_eval.prepare_strict_real_population(
            args.real_manifest, args.real_local_receipt,
            main_root=args.real_main_root, supp_root=args.real_supp_root,
            canvas_size=800, contour_cap=512,
        )
        population = real_eval.build_balanced_1016_population(strict)
        fragment_ids = sorted(population.fragments)
        fragments = [population.fragments[key] for key in fragment_ids]
        arrays = {
            "packed_masks": np.stack([np.packbits(f.mask, axis=1) for f in fragments]),
            "points": np.stack([f.points_rc for f in fragments]),
            "valid": np.stack([f.contour_valid for f in fragments]),
        }
        strict_ids = {p.pair_id for p in strict.pair_inputs}
        metadata = {"schema": "real-layout-prepared-v1", "source": source,
            "manifest_sha256": population.manifest_sha256, "fragment_ids": fragment_ids,
            "pairs": [dict(asdict(p), label=bool(label), case_cluster=cluster, strict=p.pair_id in strict_ids)
                      for p, label, cluster in zip(population.pair_inputs, population.labels, population.case_clusters)]}
        np.savez_compressed(arrays_path, **arrays)
        common.write_json(manifest_path, metadata)
    rows, n = metadata["pairs"], len(metadata["fragment_ids"])
    if (len(rows) != 1016 or sum(r["label"] for r in rows) != 508
            or sum(r["strict"] for r in rows) != 547 or len({r["pair_id"] for r in rows}) != 1016
            or arrays["packed_masks"].shape != (n, 800, 100)
            or arrays["points"].shape != (n, 512, 2) or arrays["valid"].shape != (n, 512)):
        raise ValueError("prepared real population or tensor shape differs")
    return metadata, arrays


def input_batches(metadata, arrays, batch_size):
    index = {key: i for i, key in enumerate(metadata["fragment_ids"])}
    pairs = metadata["pairs"]
    for start in range(0, len(pairs), batch_size):
        rows = pairs[start:start + batch_size]
        ia = [index[r["fragment_a_id"]] for r in rows]
        ib = [index[r["fragment_b_id"]] for r in rows]
        batch = build_target_blind_rachel_batch(
            pair_ids=[r["pair_id"] for r in rows],
            fragment_a_tokens=[r["fragment_a_id"] for r in rows],
            fragment_b_tokens=[r["fragment_b_id"] for r in rows],
            masks_a=np.unpackbits(arrays["packed_masks"][ia], axis=2).astype(bool),
            masks_b=np.unpackbits(arrays["packed_masks"][ib], axis=2).astype(bool),
            points_rc_a=arrays["points"][ia], points_rc_b=arrays["points"][ib],
            contour_valid_a=arrays["valid"][ia], contour_valid_b=arrays["valid"][ib],
        )
        yield batch


def attach_ground_truth(predictions, population_rows, path):
    """Called only after the complete prediction file has been closed/fsynced."""
    targets = json.loads(Path(path).read_text(encoding="utf-8"))["positive_pairs"]
    target_by_id = {row["pair_id"]: row for row in targets}
    positives = {row["pair_id"] for row in population_rows if row["label"]}
    if len(targets) != 508 or len(target_by_id) != 508 or set(target_by_id) != positives:
        raise ValueError("real translation GT must match exactly the population's 508 positive pair IDs")
    if [r["pair_id"] for r in predictions] != [r["pair_id"] for r in population_rows]:
        raise ValueError("prediction population order differs")
    for prediction, pair in zip(predictions, population_rows):
        prediction.update(label=bool(pair["label"]), case_cluster=pair["case_cluster"],
                          case_id=pair["case_cluster"], strict_member=pair["strict"])
        target = target_by_id.get(pair["pair_id"])
        gt = None
        if target is not None:
            if (target["fragment_a_token"] != pair["fragment_a_id"]
                    or target["fragment_b_token"] != pair["fragment_b_id"]):
                raise ValueError("GT ordered endpoints differ for " + pair["pair_id"])
            gt = np.asarray(target["translation_gt_a_to_b_rc"], dtype=float)
            if gt.shape != (2,) or not np.isfinite(gt).all():
                raise ValueError("positive translation GT must be finite [2]")
        prediction["target_translation_rc"] = common.clean(gt)
        for layout in prediction["layouts"].values():
            t = np.asarray(layout["translation_rc"], dtype=float)
            layout["translation_l2_px"] = float(np.linalg.norm(t - gt)) if gt is not None and layout["valid"] else None
    return predictions


def run(args):
    if args.batch_size <= 0 or args.shred_microbatch <= 0:
        raise ValueError("batch sizes must be positive")
    if not args.prepare_only and not args.freeze:
        raise ValueError("prediction requires --freeze from complete validation")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cache = Path(args.prepared_cache) if args.prepared_cache else destination / "prepared"
    metadata, arrays = prepare_inputs(args, cache)
    preparation_seconds = time.perf_counter() - started
    if args.prepare_only:
        common.write_json(destination / "summary.json", {
            "status": "prepared_only", "prepared_cache": str(cache.resolve()),
            "sample_count": 1016, "positive_count": 508, "preparation_seconds": preparation_seconds,
            "models_executed": False, "translation_gt_opened": False})
        print(json.dumps({"status": "prepared_only", "prepared_cache": str(cache.resolve())}), flush=True)
        return
    authority, configs = load_validation_freeze(args.freeze)
    torch.set_num_threads(1)
    receipt, _, winners = sealed._freeze_completed_winners(Path(args.run))
    full = next(w for w in winners if w.arm == "full_n512")
    threshold = float(full.threshold.threshold)
    if authority["checkpoint_sha256"] != full.checkpoint_sha256 or authority["original_fused_threshold"] != threshold:
        raise ValueError("validation freeze refers to another Full checkpoint or threshold")
    sealed._set_determinism(int(receipt["config"]["seed"]))
    device = torch.device(args.device)
    precision = receipt["config"]["precision"] if args.full_precision == "training" else args.full_precision
    amp_enabled = precision == "bf16"
    model = full.model.to(device).eval()
    head = None
    if args.shred_freeze:
        from staging.pairwise_v0_2.models.shredding_layout_head import load_shredding_layout_head
        head = load_shredding_layout_head(Path(args.shred_freeze), device=str(device),
            checkpoint_stage="classify", microbatch_size=args.shred_microbatch)
    common.write_json(destination / "protocol.json", {
        "status": "running", "population": "real_balanced1016_and_strict547", "sample_count": 1016,
        "validation_freeze": str(Path(args.freeze).resolve()), "checkpoint_sha256": full.checkpoint_sha256,
        "selected_full_decoder": authority["selected_full_decoder"], "decoders": {n: asdict(c) for n, c in configs.items()},
        "original_fused_threshold": threshold, "batch_size": args.batch_size, "prepared_cache": str(cache.resolve()),
        "full_forward_precision": "bf16_autocast" if amp_enabled else "fp32", "autocast_enabled": amp_enabled,
        "historical_real_full_precision": "fp32", "all_full_decoders_share_one_forward": True,
        "routing_used": False, "classifier_modified": False, "rotation_estimated": False,
        "shred_layout_head_used": bool(head), "shred_coarse_or_classifier_executed": False,
        "translation_gt_opened_after_all_predictions": True, "target_gt_evaluation_after_prediction_freeze": True,
        "test_or_real_used_for_fit": False})
    predictions = []
    prediction_started = time.perf_counter()
    with (destination / "pair_predictions.jsonl").open("x", encoding="utf-8") as stream, torch.inference_mode():
        for batch in input_batches(metadata, arrays, args.batch_size):
            tensors = [sealed._tensor(getattr(batch, n), device, t) for n, t in (
                ("mask_a", torch.float32), ("mask_b", torch.float32), ("points_rc_a", torch.float32),
                ("points_rc_b", torch.float32), ("contour_valid_a", torch.bool), ("contour_valid_b", torch.bool))]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                output = model(*tensors)
            values = {name: getattr(output, name).detach().float().cpu().numpy() for name in (
                "fused_probability", "coarse_probability", "local_probability", "assignment", "affinity", "translation_hat_rc")}
            shred_out = head.predict_batch(batch, return_correspondence=False) if head else None
            for i, pair_id in enumerate(batch.pair_ids):
                layouts = {"full_original": {"translation_rc": values["translation_hat_rc"][i],
                                            "valid": bool(output.decision_valid[i].item())}}
                for name, config in configs.items():
                    matrix = values["affinity" if config.score_mode == "dual_softmax" else "assignment"][i]
                    estimate = estimate_translation_layout(batch.points_rc_a[i], batch.points_rc_b[i], matrix,
                        batch.contour_valid_a[i], batch.contour_valid_b[i], config=config)
                    diagnostics = asdict(estimate)
                    for key in ("candidate_indices", "inlier_mask"):
                        diagnostics.pop(key)
                    layouts[name] = {"translation_rc": estimate.t_a_to_b_rc, "valid": estimate.valid, "diagnostics": diagnostics}
                if shred_out is not None:
                    layouts["full_with_shred_matching_layout"] = {"translation_rc": shred_out.translation_hat_rc[i],
                        "valid": bool(shred_out.translation_valid[i]), "diagnostics": {
                            "candidate_count": int(shred_out.correspondence_count[i]), "inlier_count": int(shred_out.inlier_count[i])}}
                for layout in layouts.values():
                    layout["offset_b_in_a_rc"] = -np.asarray(layout["translation_rc"])
                row = common.clean({"pair_id": pair_id, "fragment_a": batch.fragment_a_tokens[i],
                    "fragment_b": batch.fragment_b_tokens[i], "classification": {
                        b: float(values[b + "_probability"][i]) for b in ("coarse", "local", "fused")}, "layouts": layouts})
                predictions.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if len(predictions) % 25 < args.batch_size or len(predictions) == 1016:
                print(json.dumps({"processed": len(predictions), "total": 1016, "elapsed_s": round(time.perf_counter() - prediction_started, 2)}), flush=True)
        os.fsync(stream.fileno())
    common.write_json(destination / "prediction_complete.json", {
        "status": "all_predictions_frozen", "sample_count": len(predictions), "translation_gt_opened": False})
    rows = attach_ground_truth(predictions, metadata["pairs"], args.translation_gt_json)
    with (destination / "pair_results.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(common.clean(row), ensure_ascii=False, allow_nan=False) + "\n")
    summary = common.summarize(rows, threshold, authority["branch_validation_thresholds"])
    strict_summary = common.summarize([r for r in rows if r["strict_member"]], threshold, authority["branch_validation_thresholds"])
    summary.update({"status": "complete", "split": "real", "selected_full_decoder": authority["selected_full_decoder"],
        "population": "real_balanced1016", "strict_summary": strict_summary,
        "elapsed_s": time.perf_counter() - started, "preparation_seconds": preparation_seconds,
        "original_classification_preserved": True, "rotation_estimated": False, "routing_used": False,
        "full_forward_precision": "bf16_autocast" if amp_enabled else "fp32",
        "translation_gt_opened_after_all_predictions": True, "target_gt_evaluation_after_prediction_freeze": True,
        "translation_gt_json": str(args.translation_gt_json)})
    common.write_json(destination / "strict547_summary.json", strict_summary)
    common.write_json(destination / "summary.json", summary)
    print(json.dumps(common.clean(summary), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", help="required for prediction; unused by --prepare-only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", default=common.DEFAULT_RUN)
    parser.add_argument("--shred-freeze")
    parser.add_argument("--shred-microbatch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--full-precision", choices=("training", "fp32", "bf16"), default="training")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-cache", type=Path)
    parser.add_argument("--translation-gt-json", type=Path, default=DEFAULT_TRANSLATION_GT)
    parser.add_argument("--real-manifest", type=Path, default=DEFAULT_REAL_CONTROL_ROOT / "real_test_manifest.json")
    parser.add_argument("--real-local-receipt", type=Path, default=DEFAULT_REAL_CONTROL_ROOT / "local_path_receipt.json")
    parser.add_argument("--real-main-root", type=Path, default=DEFAULT_REAL_AUTHORITY_ROOT / "Dunhuang Dataset")
    parser.add_argument("--real-supp-root", type=Path, default=DEFAULT_REAL_AUTHORITY_ROOT / "Dunhuang Dataset Supp")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
