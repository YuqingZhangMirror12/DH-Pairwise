"""Frozen contiguous-seam ablation on cached real 1016/strict547 inputs.

Model loading and decoder definitions are shared with the validation runner.
Real data never selects parameters. All target-blind predictions are written
and fsynced before the existing 508-positive translation GT file is opened.
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

from experiments.rachel_n512_formal_30k import run_contiguous_seam_ablation as seam
from experiments.rachel_n512_formal_30k import run_layout_decoder_experiment as common
from experiments.rachel_n512_formal_30k import run_real_layout_decoder_experiment as real
from experiments.rachel_n512_formal_30k.resampled_input_support import InputContourResampler
from experiments.rachel_n512_formal_30k.fragment_size_strata import pair_size_metadata, summarize_size_strata
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed

DEFAULT_PREPARED_CACHE = Path(
    "/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared"
)


def load_prepared_cache(path):
    cache = Path(path)
    metadata = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "real-layout-prepared-v1":
        raise ValueError("expected the existing real-layout prepared cache")
    with np.load(cache / "inputs.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("packed_masks", "points", "valid")}
    rows, count = metadata["pairs"], len(metadata["fragment_ids"])
    if (len(rows) != 1016 or sum(row["label"] for row in rows) != 508
            or sum(row["strict"] for row in rows) != 547
            or len({row["pair_id"] for row in rows}) != 1016
            or arrays["packed_masks"].shape != (count, 800, 100)
            or arrays["points"].shape != (count, 512, 2)
            or arrays["valid"].shape != (count, 512)):
        raise ValueError("prepared cache population or model-input shapes differ")
    return metadata, arrays


def run(args):
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    torch.set_num_threads(1)
    model, identity = seam.load_model(args)
    authority, configs = seam.load_geometry_freeze(args.freeze, identity)
    branch_thresholds = authority["branch_validation_thresholds"]
    if set(branch_thresholds) != {"coarse", "local", "fused"} or not all(np.isfinite(v) for v in branch_thresholds.values()):
        raise ValueError("the validation freeze must supply finite branch thresholds")
    metadata, arrays = load_prepared_cache(args.prepared_cache)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    sealed._set_determinism(identity["seed"])
    device = torch.device(args.device)
    model = model.to(device).eval()
    threshold = identity["original_fused_threshold"]
    common.write_json(destination / "protocol.json", dict(identity,
        schema_version="real-contiguous-seam-ablation/1", population="real_balanced1016_and_strict547",
        population_size=len(metadata["pairs"]), prepared_cache=str(Path(args.prepared_cache).resolve()),
        prepared_manifest_sha256=metadata["manifest_sha256"], validation_freeze=str(Path(args.freeze).resolve()),
        selected_full_decoder=authority["selected_full_decoder"],
        decoders={name: {"kind": kind, "config": asdict(config)} for name, (kind, config) in configs.items()},
        all_decoders_share_one_forward=True, classifier_modified=False, rotation_estimated=False,
        routing_used=False, mask_normals_used=False,
        contour_normals_used=any(getattr(config, "use_normals", False) for _, config in configs.values()),
        fragment_overlap_used=any(getattr(config, "use_overlap", False) for _, config in configs.values()),
        seam_quality_evaluated=False,
        seam_quality_unavailable="no_dataset_correspondence_reference",
        test_or_real_used_for_fit=False, target_gt_evaluation_after_prediction_freeze=True,
        save_seam_membership=bool(args.save_seam_membership)))
    started, predictions = time.perf_counter(), []
    resampler = (InputContourResampler(identity["resample_contour_cap"])
                 if identity.get("resample_contour_cap") else None)
    prediction_path = destination / "pair_predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for batch in real.input_batches(metadata, arrays, args.batch_size):
            if resampler is not None:
                batch = resampler.resample_batch(batch)
            tensors = [sealed._tensor(getattr(batch, name), device, dtype) for name, dtype in (
                ("mask_a", torch.float32), ("mask_b", torch.float32), ("points_rc_a", torch.float32),
                ("points_rc_b", torch.float32), ("contour_valid_a", torch.bool), ("contour_valid_b", torch.bool))]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=identity["precision"] == "bf16"):
                output = model(*tensors)
            values = {name: getattr(output, name).detach().float().cpu().numpy() for name in (
                "assignment", "translation_hat_rc", "coarse_probability", "local_probability", "fused_probability")}
            for i, pair_id in enumerate(batch.pair_ids):
                layouts = seam.decode_pair(batch.points_rc_a[i], batch.points_rc_b[i], batch.contour_valid_a[i],
                    batch.contour_valid_b[i], values["assignment"][i], configs, args.save_seam_membership,
                    mask_a=batch.mask_a[i], mask_b=batch.mask_b[i])
                layouts["full_original"] = {"translation_rc": values["translation_hat_rc"][i],
                                            "valid": bool(output.decision_valid[i].item())}
                for layout in layouts.values():
                    layout["offset_b_in_a_rc"] = -np.asarray(layout["translation_rc"])
                row = common.clean({"pair_id": pair_id, "fragment_a": batch.fragment_a_tokens[i],
                    "fragment_b": batch.fragment_b_tokens[i], "classification": {
                        branch: float(values[branch + "_probability"][i]) for branch in ("coarse", "local", "fused")},
                    "layouts": layouts})
                row.update(pair_size_metadata(batch.mask_a[i], batch.mask_b[i]))
                predictions.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if len(predictions) % 25 < args.batch_size or len(predictions) == len(metadata["pairs"]):
                print(json.dumps({"processed": len(predictions), "total": len(metadata["pairs"]),
                    "elapsed_s": round(time.perf_counter() - started, 2)}), flush=True)
        os.fsync(stream.fileno())
    if [row["pair_id"] for row in predictions] != [row["pair_id"] for row in metadata["pairs"]]:
        raise ValueError("not every cached real pair was predicted in the frozen order")
    common.write_json(destination / "prediction_complete.json", {
        "status": "all_predictions_frozen", "sample_count": len(predictions), "translation_gt_opened": False})
    # This is the first access to the GT file. The shared helper enforces exact
    # 508-positive pair IDs and ordered endpoint alignment before scoring.
    rows = real.attach_ground_truth(predictions, metadata["pairs"], args.translation_gt_json)
    with (destination / "pair_results.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(common.clean(row), ensure_ascii=False, allow_nan=False) + "\n")
    summary = common.summarize(rows, threshold, branch_thresholds)
    strict = common.summarize([row for row in rows if row["strict_member"]], threshold, branch_thresholds)
    summary["size_strata"] = summarize_size_strata(rows, threshold, branch_thresholds)
    strict["size_strata"] = summarize_size_strata([row for row in rows if row["strict_member"]], threshold, branch_thresholds)
    summary.update(status="complete", split="real", population="real_balanced1016", strict_summary=strict,
        selected_full_decoder=authority["selected_full_decoder"], selection_source="frozen_validation_only",
        checkpoint_sha256=identity["checkpoint_sha256"], precision=identity["precision"],
        elapsed_s=time.perf_counter() - started, original_classification_preserved=True,
        all_decoders_share_one_forward=True, rotation_estimated=False, routing_used=False,
        test_or_real_used_for_fit=False, target_gt_evaluation_after_prediction_freeze=True,
        seam_quality_evaluated=False,
        seam_quality_unavailable="no_dataset_correspondence_reference",
        translation_gt_json=str(args.translation_gt_json))
    common.write_json(destination / "strict547_summary.json", strict)
    common.write_json(destination / "summary.json", summary)
    print(json.dumps(common.clean(summary), ensure_ascii=False), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--run", help="Original completed Full run")
    model.add_argument("--checkpoint", help="Architecture winner.pt with model_config and model_state_dict")
    parser.add_argument("--pair-threshold", type=float, help="Frozen fused threshold; required with --checkpoint")
    parser.add_argument("--freeze", required=True, help="Formal 3000-pair validation seam freeze")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared-cache", type=Path, default=DEFAULT_PREPARED_CACHE)
    parser.add_argument("--translation-gt-json", type=Path, default=real.DEFAULT_TRANSLATION_GT)
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-seam-membership", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
