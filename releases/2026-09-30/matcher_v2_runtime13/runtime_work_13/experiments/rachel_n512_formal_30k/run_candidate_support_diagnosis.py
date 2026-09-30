"""Replay validation-only wrong peaks to locate lost correspondence support.

Not a new decoder or accuracy benchmark. Select every positive >10px failure
from a completed frozen validation cache, run that same checkpoint, and compare
all positive transport mass with uncapped/capped Top2. GT is only used after
model inference and baseline decoding, for explicitly oracle diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from experiments.rachel_n512_formal_30k import run_layout_decoder_experiment as common
from experiments.rachel_n512_formal_30k.candidate_support_diagnostics import diagnose_candidate_support
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from experiments.rachel_n512_formal_30k.run_contiguous_seam_ablation import load_model, decoder_configs
from staging.pairwise_v0_2.models.translation_layout import estimate_translation_layout
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


def run(args):
    cache = Path(args.cache)
    summary = json.loads((cache / "summary.json").read_text())
    protocol = json.loads((cache / "protocol.json").read_text())
    if (summary.get("status") != "complete" or summary.get("split") != "val"
            or summary.get("sample_count") != 3000 or protocol.get("precision") != "fp32"):
        raise ValueError("Use a completed 3000-pair FP32 validation cache only")
    selected = {}
    with (cache / "pair_results.jsonl").open() as stream:
        for line in stream:
            raw = json.loads(line)
            layout = raw["layouts"]["full_top2_mode"]
            if raw["label"] and (not layout["valid"] or layout["translation_l2_px"] > 10):
                selected[raw["pair_id"]] = raw
    if len(selected) != args.expected_failures:
        raise ValueError("Unexpected validation failure population size")
    torch.set_num_threads(1)
    model, identity = load_model(SimpleNamespace(checkpoint=protocol["checkpoint_path"],
        pair_threshold=protocol["original_fused_threshold"], precision="fp32", seed=protocol["seed"]))
    if identity["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the completed failure-selection cache")
    if identity.get("resample_contour_cap"):
        raise ValueError("This diagnostic is scoped to the original frozen512 model")
    sealed._set_determinism(identity["seed"])
    dataset = RachelPairDataset(Path(args.dataset), "val")
    with (Path(args.dataset) / "pairs/val.jsonl").open() as stream:
        manifest = [json.loads(line) for line in stream if line.strip()]
    indices = [i for i, row in enumerate(manifest) if row["pair_id"] in selected]
    if len(indices) != len(selected):
        raise ValueError("Not all validation failure pair IDs occur in the dataset")
    loader = make_ablation_loader(dataset, indices, batch_size=4, num_workers=2,
                                  seed=identity["seed"], contour_cap=model.config.contour_cap)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "assignments").mkdir()
    common.write_json(destination / "protocol.json", dict(identity, split="val",
        diagnostic_only=True, diagnostic_uses_gt=True, models_trained=False,
        validation_selection="all positive Top2 invalid or error>10px in completed validation cache",
        failure_selection_cache=str(cache), population_size=len(indices),
        original_validation_population_size=3000, original_validation_positive_count=1500,
        no_test_or_real_read=True, no_new_decoder_selected=True,
        parent_source_snapshot=args.parent_source))
    device = torch.device(args.device)
    model = model.to(device).eval()
    config = decoder_configs()["full_top2_mode"][1]
    rows, started = [], time.perf_counter()
    with torch.inference_mode(), (destination / "pair_results.jsonl").open("x") as stream:
        for batch in loader:
            tensors = [sealed._tensor(getattr(batch, name), device, dtype) for name, dtype in (
                ("mask_a", torch.float32), ("mask_b", torch.float32),
                ("points_rc_a", torch.float32), ("points_rc_b", torch.float32),
                ("contour_valid_a", torch.bool), ("contour_valid_b", torch.bool))]
            output = model(*tensors)
            assignments = output.assignment.detach().float().cpu().numpy()
            for i, pair_id in enumerate(batch.pair_ids):
                a, b = batch.points_rc_a[i], batch.points_rc_b[i]
                va, vb = batch.contour_valid_a[i], batch.contour_valid_b[i]
                score = assignments[i]
                baseline = estimate_translation_layout(a, b, score, va, vb, config=config)
                previous = selected[pair_id]["layouts"]["full_top2_mode"]
                if baseline.valid != previous["valid"] or (baseline.valid and not np.allclose(
                        baseline.t_a_to_b_rc, previous["translation_rc"], rtol=0, atol=1e-4)):
                    raise ValueError("Replayed baseline differs from failure-selection cache: " + pair_id)
                # Only below this point are targets read, for oracle diagnostics.
                if not bool(batch.labels[i]) or not bool(batch.translation_valid[i]):
                    raise ValueError("The selected diagnostic population must have valid positive targets")
                gt = np.asarray(batch.translation_a_to_b_rc[i], dtype=float)
                diagnostic = diagnose_candidate_support(a, b, score, va, vb,
                    gt_translation_a_to_b_rc=gt,
                    baseline_translation_a_to_b_rc=baseline.t_a_to_b_rc if baseline.valid else None)
                artifact = "assignments/pair_%02d.npz" % len(rows)
                np.savez_compressed(destination / artifact, points_a=a, points_b=b, valid_a=va,
                    valid_b=vb, confidence=score, gt_translation=gt, baseline_translation=baseline.t_a_to_b_rc)
                row = common.clean(dict(pair_id=pair_id, original_error_px=previous["translation_l2_px"],
                    replayed_baseline_valid=baseline.valid,
                    replayed_baseline_error_px=float(np.linalg.norm(baseline.t_a_to_b_rc-gt)) if baseline.valid else None,
                    assignment_artifact=artifact, diagnostic=diagnostic))
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(dict(processed=len(rows), total=len(indices),
                                  elapsed_s=time.perf_counter()-started)), flush=True)
    result = dict(status="complete", sample_count=len(rows), elapsed_s=time.perf_counter()-started,
                  diagnostic_only=True, diagnostic_uses_gt=True, no_accuracy_claim=True,
                  no_new_decoder_selected=True, no_test_or_real_read=True,
                  interpretation_limit="GT-local support is an oracle observation, not proof a target-blind decoder can recover the correct mode")
    common.write_json(destination / "summary.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=common.DEFAULT_DATA)
    parser.add_argument("--expected-failures", type=int, default=11)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parent-source", required=True)
    run(parser.parse_args())
