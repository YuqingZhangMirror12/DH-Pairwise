"""One frozen S7 M12 forward, six fixed C8/C16 heads on materialized hard VAL.

Standalone diagnostic only: no training, selection, threshold fitting, new
preprocessing or deployment changes. The external launcher owns the GPU lease
before importing Torch. No heavyweight imports occur at module import time.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import time

SCHEMA = "hard-simval-six-frozen-scorers/1"
ARMS = ("all_tokens", "matched_tokens", "matched_edges")
SOURCE_SHA = "d8a93af1eb5f3b02baaf7d42b9d8675242a446a1b11e43cde0561ba89e670e07"
MANIFEST_SHA = "ad01c3011e0c9b6f6bed593edde14e8564818bc22bf1ee822b8ceaea91d79215"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_expected(root, manifest_sha, entries):
    root = Path(root).resolve(strict=True)
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    for record in (protocol, summary):
        if (record.get("schema") != "s7-hard-simval-matcher-evaluation/1"
                or record.get("status") != "complete"
                or record.get("manifest_sha256") != manifest_sha
                or record.get("model", {}).get("epoch") != 12
                or record.get("model", {}).get("checkpoint_sha256") != SOURCE_SHA):
            raise ValueError("expected layout must be completed same-manifest frozen M12")
    path = root / "pair_metrics.jsonl"
    if summary.get("pair_metrics_sha256") != sha(path):
        raise ValueError("expected M12 rows differ from completed summary")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    indexed = {row["pair_id"]: row for row in rows}
    if len(rows) != 6000 or len(indexed) != len(rows) or set(indexed) != {e["pair_id"] for e in entries}:
        raise ValueError("expected M12 IDs must equal all hard-VAL entries, not an intersection")
    for entry in entries:
        row = indexed[entry["pair_id"]]
        if any(row.get(k) != entry.get(k) for k in ("source_pair_id", "label", "recipe", "changed_pair")):
            raise ValueError("expected M12 row identity differs")
    return indexed, dict(directory=str(root), rows_sha256=sha(path),
                         role="CPU layout sanity only; no reused classification scores")


def shared_forward(base, heads, inputs, selector):
    """Exactly the existing adapter calculation, sharing base and decoder once.

    Deliberately accepts six image/point/valid tensors only, never labels/GT.
    All heads consume the same tensor objects and predicted selection.
    """
    import torch
    if len(inputs) != 6:
        raise ValueError("only original six model inputs are accepted")
    with torch.inference_mode():
        original = base(*inputs)
        selected = selector(original.assignment, inputs[2], inputs[3], inputs[4], inputs[5])
        safe = selected.candidate_indices.clamp_min(0)
        batch = torch.arange(len(inputs[0]), device=original.assignment.device)[:, None]
        weights = original.assignment[batch, safe[..., 0], safe[..., 1]]
        weights = torch.where(selected.candidate_valid, weights, torch.zeros_like(weights))
        scores = {}
        for key, head in heads.items():
            out = head(original.token_features_a, original.token_features_b,
                inputs[4], inputs[5], selected, candidate_weights=weights,
                points_a_rc=inputs[2], points_b_rc=inputs[3])
            if not torch.isfinite(out.logit).all():
                raise ValueError("nonfinite frozen head logit: " + key)
            deployed = torch.where(original.training_valid, out.logit, torch.zeros_like(out.logit))
            scores[key] = dict(raw_logit=out.logit.cpu().tolist(), logit=deployed.cpu().tolist(),
                probability=deployed.sigmoid().cpu().tolist(), used_fallback=out.used_fallback.cpu().tolist())
    return original, selected, scores


def rows_from_batch(entries, reports, batch, original, selected, scores, expected, start):
    import numpy as np
    arrays = {name: getattr(selected, name).detach().cpu().numpy() for name in (
        "layout_valid", "translation_a_to_b_rc", "candidate_valid", "candidate_inliers", "mask_a", "mask_b")}
    training_valid = original.training_valid.detach().cpu().tolist()
    decision_valid = original.decision_valid.detach().cpu().tolist()
    rows = []
    for i, entry in enumerate(entries):
        valid = bool(arrays["layout_valid"][i])
        translation = arrays["translation_a_to_b_rc"][i]
        error = float(np.linalg.norm(translation.astype(np.float64) - batch.translation_a_to_b_rc[i])) if valid and entry["label"] else None
        old = expected[entry["pair_id"]]
        old_error = old["raw_translation_l2_px"]
        row = dict(entry, ordinal=start+i, view="clean" if entry["recipe"] == "clean" else "damaged",
            pose_supervision_enabled=reports[i]["pose_supervision_enabled"],
            training_valid=bool(training_valid[i]), decision_valid=bool(decision_valid[i]),
            raw_layout_valid=valid, raw_layout_reason=selected.reasons[i],
            raw_translation_l2_px=error, raw_layout20_correct=error is not None and error <= 20,
            translation_a_to_b_rc=translation.tolist() if valid else None,
            candidate_count=int(arrays["candidate_valid"][i].sum()),
            inlier_edge_count=int((arrays["candidate_valid"][i] & arrays["candidate_inliers"][i]).sum()),
            unique_a=int(arrays["mask_a"][i].sum()), unique_b=int(arrays["mask_b"][i].sum()),
            scores={key: {field: values[i] for field, values in item.items()} for key, item in scores.items()},
            expected_cpu_layout=dict(raw_layout_valid=old["raw_layout_valid"],
                raw_translation_l2_px=old_error, raw_layout20_correct=old["raw_layout20_correct"],
                error_delta_px=error-old_error if error is not None and old_error is not None else None))
        rows.append(row)
    return rows


def layout_sanity(rows):
    deltas = [abs(r["expected_cpu_layout"]["error_delta_px"]) for r in rows
              if r["expected_cpu_layout"]["error_delta_px"] is not None]
    positives = [r for r in rows if r["label"]]
    return dict(count=len(rows), positive_count=len(positives),
        validity_agreement=sum(r["raw_layout_valid"] == r["expected_cpu_layout"]["raw_layout_valid"] for r in rows),
        positive_layout20_agreement=sum(r["raw_layout20_correct"] == r["expected_cpu_layout"]["raw_layout20_correct"] for r in positives),
        comparable_error_count=len(deltas), mean_absolute_error_delta_px=sum(deltas)/len(deltas) if deltas else None,
        maximum_absolute_error_delta_px=max(deltas) if deltas else None,
        mismatched_layout20_pair_ids=[r["pair_id"] for r in positives
            if r["raw_layout20_correct"] != r["expected_cpu_layout"]["raw_layout20_correct"]],
        interpretation="CPU batch1 versus GPU batched sanity; numerical differences are reported, not silently rejected or replaced")


def run(args):
    started = time.monotonic()
    if args.device != "cuda:0" or args.batch_size < 1 or (args.limit is not None and not 0 < args.limit < 6000):
        raise ValueError("guarded cuda:0, positive batch and explicit full6000 or pilot limit1..5999 required")
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence import evaluate_hard_validation as hard
    manifest = Path(args.manifest).resolve(strict=True)
    source = Path(args.source_val_manifest).resolve(strict=True)
    if sha(manifest) != MANIFEST_SHA or sha(source) != hard.original.VAL_HASH:
        raise ValueError("fixed hard6000/clean3000 manifest hashes differ")
    record = json.loads(manifest.read_text())
    source_rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    hard.original.validate_manifest(source_rows)
    counts = hard.validate_manifest(record, source_rows)
    all_entries = record["entries"]
    expected, expected_receipt = load_expected(args.expected_layout_dir, sha(manifest), all_entries)
    entries = all_entries if args.full else all_entries[:args.limit]
    data_root = Path(record["artifact_root"]).resolve(strict=True)
    training_root = Path(args.training_root).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("new diagnostic directory required; no overwrite/resume")
    for protected in (data_root, training_root, source.parent, Path(args.expected_layout_dir).resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("diagnostic output must be separate from source data/models/results")
    # The external launcher must acquire the existing per-UUID lease first.
    import torch
    from staging.pairwise_v0_2.training import rachel_n512_runner as runner
    from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import collate_rachel_pairs
    try:
        evaluation = importlib.import_module("matched_only.evaluate")
    except ModuleNotFoundError as error:
        if error.name != "matched_only":
            raise
        evaluation = importlib.import_module("experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.evaluate")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    heads, receipts, base = {}, {}, None
    for arm in ARMS:
        for budget in (8, 16):
            adapter, receipt = evaluation.load_frozen_model(training_root / arm, "fixed_epoch", budget=budget)
            if receipt["source_matcher_sha256"] != SOURCE_SHA or receipt["training_identity"]["arm"] != arm:
                raise ValueError("head arm or shared frozen Matcher identity differs")
            key = "%s_c%d" % (arm, budget)
            if base is None:
                base = adapter.base_model.eval().requires_grad_(False).to(device)
            heads[key] = adapter.score_head.eval().requires_grad_(False).to(device)
            receipts[key] = receipt
            del adapter
    thresholds = {key: receipt["operating_points"]["thresholds"] for key, receipt in receipts.items()}
    protocol = dict(schema=SCHEMA, status="running", pilot=not args.full, sample_count=len(entries),
        completed_count=0, manifest=str(manifest), manifest_sha256=sha(manifest), source_val_manifest_sha256=sha(source),
        source_matcher_sha256=SOURCE_SHA, source_protocol=record["protocol"], full_by_recipe=counts,
        pair_ids=[e["pair_id"] for e in entries], batch_size=args.batch_size, device=str(device),
        source_sha256=sha(__file__), heads=receipts, thresholds=thresholds, expected_layout=expected_receipt,
        decoder_config=asdict(evaluation.inference.DECODER_CONFIG),
        diagnostic_only=True, thresholds_fitted=False, training_performed=False, model_selection_performed=False,
        shared_matcher_forward_per_batch=1, model_inputs="unchanged six inputs only; labels/GT used solely after inference",
        geometry_policy="unchanged production decoder; all six heads share same tokens, raw Q and final selection",
        partial_seam_pairing="same source clean and requested variant; not independent populations",
        runtime=dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                     gpu_name=torch.cuda.get_device_name(device), deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", protocol)
    all_rows = []
    try:
        with (output / "cases.jsonl").open("x") as stream:
            for start in range(0, len(entries), args.batch_size):
                chunk = entries[start:start+args.batch_size]
                loaded = [hard.load_entry(data_root, entry) for entry in chunk]
                batch = collate_rachel_pairs([item[0] for item in loaded], contour_cap=512)
                inputs, _ = runner._full_batch(batch, device)
                original, selected, scores = shared_forward(base, heads, inputs, evaluation.inference.select_predicted_inliers)
                rows = rows_from_batch(chunk, [item[1] for item in loaded], batch, original, selected, scores, expected, start)
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                all_rows.extend(rows)
                protocol["completed_count"] = len(all_rows)
                if len(all_rows) % 256 == 0 or len(all_rows) == len(entries):
                    stream.flush()
                    save(output / "protocol.json", protocol)
        protocol.update(status="complete" if args.full else "pilot_complete",
            elapsed_seconds=time.monotonic()-started, cases_sha256=sha(output / "cases.jsonl"),
            layout_sanity=layout_sanity(all_rows))
        save(output / "protocol.json", protocol)
        return dict(status=protocol["status"], output=str(output), sample_count=len(all_rows),
                    elapsed_seconds=protocol["elapsed_seconds"], layout_sanity=protocol["layout_sanity"])
    except BaseException as error:
        protocol.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        save(output / "protocol.json", protocol)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source-val-manifest", "expected-layout-dir", "training-root", "output"):
        result.add_argument("--"+name, required=True, type=Path)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--batch-size", type=int, default=8)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--limit", type=int)
    return result


if __name__ == "__main__":
    raise SystemExit("Use the external assigned-GPU lease launcher, then call parser() and run(args).")
