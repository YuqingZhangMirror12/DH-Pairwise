"""Exploratory saved-score mean-probability control for three completed data arms.

freeze reads ONLY realism_training/<arm>/{train_val_freeze,winner_validation}.json.
It verifies the original VAL scores/thresholds and freezes one equal-row F1
threshold per arm for score = 0.5*p_coarse + 0.5*p_local. No weights are trained,
models selected, logits reconstructed, or checkpoints/tensors opened.

evaluate requires that all-arm freeze before opening the completed TEST/REAL
pair_results.jsonl, protocol.json and summary.json. The arm's original fixed
Top2 layout is reused unchanged. A wrongly placed accepted positive counts as
both a joint FP and FN; invalid poses remain in the positive denominator.

This supplement was proposed after observing the data-arm external results.
The simple policy was tested earlier in Stage1, but this is exploratory analysis,
NOT a preregistered new holdout or a REAL-selected model/threshold comparison.
All writes go into new exclusive directories; upstream artifacts are immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

# Registered trainer's exact metric and largest-threshold-among-F1-ties rule.
# This module imports Torch transitively, but no model, device or checkpoint API
# is called by this saved-JSON-only program.
from experiments.rachel_n512_formal_30k.run_layout_decoder_experiment import (
    classification, fit_threshold, summarize,
)


SCHEMA = "rachel-data-arm-mean-probability-control/v1"
ARMS = ("original24k", "matched24k", "realism60k")
BRANCHES = ("coarse", "local", "fused")
COUNTS = {"val": (3000, 1500), "test": (3000, 1500), "real": (1016, 508), "strict": (547, 508)}
DECODER = "full_top2_mode"
POLICY = "mean_probability"
DISCLOSURE = ("Exploratory supplement after observing the data-arm external results; "
              "the policy was tested previously in Stage1. REAL has been repeatedly "
              "observed, not a pristine holdout. No REAL threshold/model/weight fitting.")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _same_metrics(observed, expected, context):
    """Compare saved metrics recursively, including nulls and exact counts."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise ValueError(context + ": expected metric object")
        for key, value in expected.items():
            if key not in observed:
                raise ValueError(context + ": missing " + key)
            _same_metrics(observed[key], value, context + "/" + key)
    elif type(expected) in (float, int):
        if (type(observed) not in (float, int) or not math.isfinite(observed)
                or not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-10)):
            raise ValueError(context + ": saved metrics disagree")
    elif observed != expected:
        raise ValueError(context + ": saved metrics disagree")


def validate_rows(rows, split):
    """Validate the exact finite probability/identity population, without drops."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing or empty score rows")
    identities = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in identities:
            raise ValueError("pair IDs must be nonempty and unique")
        if type(row.get("label")) is not bool or type(row.get("decision_valid")) is not bool:
            raise ValueError("label and decision_valid must be explicit Boolean values")
        scores = row.get("classification", {})
        for branch in BRANCHES:
            value = scores.get(branch)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("missing/nonfinite/out-of-range saved probability")
        identity = [row["label"]]
        if split != "val":
            for side in ("a", "b"):
                value = row.get("fragment_" + side)
                if not isinstance(value, str) or not value:
                    raise ValueError("missing ordered fragment endpoint")
                identity.append(value)
            if row["fragment_a"] == row["fragment_b"]:
                raise ValueError("pair cannot have identical endpoints")
            if set(row.get("layouts", {})) != {DECODER}:
                raise ValueError("expected the original single fixed Top2 layout")
            layout = row["layouts"][DECODER]
            if type(layout.get("valid")) is not bool or "translation_l2_px" not in layout:
                raise ValueError("layout validity/error must be explicit")
            error = layout["translation_l2_px"]
            if row["label"] and layout["valid"]:
                if type(error) not in (float, int) or not math.isfinite(error) or error < 0:
                    raise ValueError("valid positive pose needs a finite nonnegative error")
            elif error is not None:
                raise ValueError("negative or invalid pose must have null target error")
            # GT is already attached in these saved rows; compare between arms.
            identity.append(row.get("target_translation_rc"))
        if split == "real":
            if type(row.get("strict_member")) is not bool or not isinstance(row.get("case_id"), str) or not row["case_id"]:
                raise ValueError("REAL rows require strict membership and case identity")
            identity.extend((row["strict_member"], row["case_id"]))
        identities[pair_id] = identity
    if (len(rows), sum(row["label"] for row in rows)) != COUNTS[split]:
        raise ValueError(split + " score population counts differ")
    if split == "val" and not all(row["decision_valid"] for row in rows):
        raise ValueError("selected VAL winner requires complete decision coverage")
    if split == "real":
        strict = [row for row in rows if row["strict_member"]]
        if (len(strict), sum(row["label"] for row in strict)) != COUNTS["strict"]:
            raise ValueError("REAL strict population counts differ")
    return identities


def mean_scores(rows):
    # Exact stored native probabilities, promoted only for arithmetic; NO sigmoid
    # reconstruction from logits and NO extra FP32 rounding of the mean.
    return np.asarray([0.5 * row["classification"]["coarse"]
                       + 0.5 * row["classification"]["local"] for row in rows], dtype=np.float64)


def _identity_sha(identities):
    return hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _training_complete(saved):
    if (saved.get("status") != "complete" or saved.get("precision") != "fp32"
            or saved.get("test_or_real_used_for_fit") is not False
            or saved.get("original_validation_unchanged") is not True
            or saved.get("completed_global_exposures") != 120000
            or saved.get("completed_optimizer_updates") != 7500
            or saved.get("completed_validation_events") != 5):
        raise ValueError("all data arms require the complete original training/VAL freeze")
    validation = saved["validation"]
    n, p = COUNTS["val"]
    if ((validation.get("sample_count"), validation.get("positive_count"), validation.get("negative_count")) != (n, p, n-p)
            or validation.get("pose_used_for_selection") is not False
            or validation.get("decision_coverage") != 1.0
            or saved.get("classifier_thresholds") != validation.get("thresholds")
            or set(saved["classifier_thresholds"]) != set(BRANCHES)):
        raise ValueError("training freeze differs from fixed VAL-only classification protocol")
    if not isinstance(saved.get("checkpoint"), str) or not saved["checkpoint"]:
        raise ValueError("training freeze lacks selected checkpoint identity")
    for key in ("selected_epoch", "selected_global_exposure", "selected_optimizer_updates", "selected_validation_event", "unique_count"):
        if type(saved.get(key)) is not int or saved[key] <= 0:
            raise ValueError("training freeze lacks positive selected identity field: " + key)


def freeze(root, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("use a new output directory")
    arms, reference = {}, None
    for arm in ARMS:
        directory = root / "realism_training" / arm
        train_path, rows_path = directory / "train_val_freeze.json", directory / "winner_validation.json"
        saved, rows = read_json(train_path), read_json(rows_path)
        _training_complete(saved)
        identities = validate_rows(rows, "val")
        if reference is not None and identities != reference:
            raise ValueError("VAL pair ID/label identities differ across arms")
        reference = identities
        labels = np.asarray([row["label"] for row in rows], bool)
        for branch in BRANCHES:
            scores = [row["classification"][branch] for row in rows]
            threshold = fit_threshold(labels, scores)
            if threshold != saved["classifier_thresholds"][branch]:
                raise ValueError("old VAL threshold does not reproduce: " + arm + "/" + branch)
            _same_metrics(classification(labels, scores, threshold), saved["validation"]["methods"][branch],
                          arm + "/old_VAL/" + branch)
        values = mean_scores(rows)
        threshold = fit_threshold(labels, values)
        arms[arm] = dict(threshold=threshold, validation=classification(labels, values, threshold),
                         original_validation=saved["validation"]["methods"],
                         source_training_freeze=str(train_path), source_training_freeze_sha256=sha256(train_path),
                         source_validation_rows=str(rows_path), source_validation_rows_sha256=sha256(rows_path),
                         checkpoint=saved["checkpoint"], checkpoint_sha256=saved.get("checkpoint_sha256"),
                         classifier_thresholds=saved["classifier_thresholds"],
                         selected_identity={key: saved[key] for key in ("selected_epoch", "selected_global_exposure",
                             "selected_optimizer_updates", "selected_validation_event", "unique_count")})
    payload = dict(schema_version=SCHEMA, status="complete", phase="freeze", root=str(root),
                   created_at=datetime.now(timezone.utc).isoformat(), policy=POLICY,
                   formula="0.5*p_coarse + 0.5*p_local", score_precision="float64 arithmetic on saved native probabilities",
                   threshold_fit="VAL equal-row F1; largest threshold among ties", selected_on="val",
                   arms=arms, val_identity_sha256=_identity_sha(reference),
                   model_selection_performed=False, weights_fit=False, model_executed=False,
                   test_or_real_opened_in_freeze=False, test_or_real_used_for_fit=False,
                   checkpoint_file_opened=False, exploration_disclosure=DISCLOSURE,
                   checkpoint_binding="training freeze path/hash and selection fields; evaluation protocols supply weight SHA")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "validation_freeze.json", payload)
    return payload


def _checked_freeze(root, path):
    saved = read_json(path)
    if (saved.get("schema_version") != SCHEMA or saved.get("status") != "complete"
            or saved.get("phase") != "freeze" or saved.get("policy") != POLICY
            or saved.get("selected_on") != "val" or saved.get("test_or_real_opened_in_freeze") is not False
            or saved.get("test_or_real_used_for_fit") is not False
            or saved.get("root") != str(root) or set(saved.get("arms", {})) != set(ARMS)):
        raise ValueError("evaluation requires a complete all-three-arm VAL probability-mean freeze")
    for arm in ARMS:
        record = saved["arms"][arm]
        if (sha256(root / "realism_training" / arm / "train_val_freeze.json") != record["source_training_freeze_sha256"]
                or sha256(root / "realism_training" / arm / "winner_validation.json") != record["source_validation_rows_sha256"]):
            raise ValueError("frozen training/VAL score source changed")
        threshold = record["threshold"]
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("invalid frozen threshold")
        if record["validation"]["threshold"] != threshold:
            raise ValueError("frozen mean threshold differs from VAL record")
    return saved


def _read_external(root, arm, split, frozen):
    directory = root / "realism_evaluation" / arm / split
    protocol, summary = read_json(directory / "protocol.json"), read_json(directory / "summary.json")
    if (protocol.get("schema_version") != "realism-checkpoint-evaluation/1"
            or any(item.get("status") != "complete" or item.get("split") != split
                   or item.get("precision") != "fp32" or item.get("test_or_real_used_for_fit") is not False
                   or item.get("selected_full_decoder") != DECODER for item in (protocol, summary))):
        raise ValueError("external score artifacts are incomplete or use another protocol")
    if (protocol.get("training_freeze_sha256") != frozen["source_training_freeze_sha256"]
            or protocol.get("checkpoint_path") != frozen["checkpoint"]):
        raise ValueError("evaluation checkpoint/training freeze binding differs")
    for observed, original in (("checkpoint_epoch", "selected_epoch"), ("global_exposure", "selected_global_exposure"),
                               ("optimizer_updates", "selected_optimizer_updates"), ("validation_event", "selected_validation_event"),
                               ("unique_count", "unique_count")):
        if protocol.get(observed) != frozen["selected_identity"][original]:
            raise ValueError("evaluation checkpoint selected identity differs: " + observed)
    checkpoint_id = protocol.get("checkpoint_sha256")
    if (not isinstance(checkpoint_id, str) or len(checkpoint_id) != 64
            or any(c not in "0123456789abcdef" for c in checkpoint_id)
            or summary.get("checkpoint_sha256") != checkpoint_id
            or (frozen.get("checkpoint_sha256") is not None and frozen["checkpoint_sha256"] != checkpoint_id)):
        raise ValueError("evaluation checkpoint SHA identities disagree")
    thresholds = frozen["classifier_thresholds"]
    for item in (protocol, summary):
        if item.get("branch_validation_thresholds") != thresholds or item.get("original_fused_threshold") != thresholds["fused"]:
            raise ValueError("original evaluation thresholds differ from own VAL freeze")
    required_decoder = dict(correspondence_mode="topk_union", top_k=2, max_candidates=512,
                            min_inliers=3, inlier_radius_px=10.0, decoder="mode_consensus")
    if any(protocol.get("decoder_config", {}).get(key) != value for key, value in required_decoder.items()):
        raise ValueError("original fixed Top2 decoder configuration differs")
    with (directory / "pair_results.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    identities = validate_rows(rows, split)
    if protocol.get("sample_count") != len(rows):
        raise ValueError("protocol pair count differs")
    original = summarize(rows, thresholds["fused"], thresholds)
    for key in ("sample_count", "positive_count", "classification", "layout"):
        _same_metrics(original[key], summary[key], arm + "/" + split + "/" + key)
    if split == "real":
        strict_original = summarize([row for row in rows if row["strict_member"]], thresholds["fused"], thresholds)
        for key in ("sample_count", "positive_count", "classification", "layout"):
            _same_metrics(strict_original[key], summary["strict_summary"][key], arm + "/strict/" + key)
    return rows, identities, checkpoint_id, dict(source_directory=str(directory),
        protocol_sha256=sha256(directory / "protocol.json"), summary_sha256=sha256(directory / "summary.json"),
        pair_results_sha256=sha256(directory / "pair_results.jsonl"))


def population_metrics(rows, threshold, original_threshold):
    """One score policy, unchanged geometry, all-positive joint denominators."""
    labels, values = np.asarray([row["label"] for row in rows], bool), mean_scores(rows)
    selected = values >= threshold
    valid = np.asarray([row["layouts"][DECODER]["valid"] for row in rows], bool)
    errors = np.asarray([row["layouts"][DECODER]["translation_l2_px"]
                         if row["layouts"][DECODER]["translation_l2_px"] is not None else np.inf for row in rows])
    positive_errors = errors[labels & valid & np.isfinite(errors)]
    layout = dict(positive_pose_coverage=float((labels & valid).sum() / max(1, labels.sum())),
                  median_px_conditional=float(np.median(positive_errors)) if len(positive_errors) else None,
                  p90_px_conditional=float(np.quantile(positive_errors, .9)) if len(positive_errors) else None,
                  recall={}, assembly={})
    for tolerance in (2, 5, 8, 10):
        correct = labels & valid & (errors <= tolerance)
        tp = int((selected & correct).sum())
        fp, fn = int(selected.sum()) - tp, int(labels.sum()) - tp
        layout["recall"][str(tolerance)] = float(correct.sum() / max(1, labels.sum()))
        layout["assembly"][str(tolerance)] = dict(tp=tp, fp=fp, fn=fn, precision=tp / max(1, tp + fp),
            recall=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn))
    return dict(sample_count=len(rows), positive_count=int(labels.sum()), threshold=threshold,
                decision_coverage=float(np.mean([row["decision_valid"] for row in rows])),
                classification={POLICY: classification(labels, values, threshold),
                    "original_fused": classification(labels, [row["classification"]["fused"] for row in rows], original_threshold)},
                layout={DECODER: layout}, original_fused_layout=summarize(rows, original_threshold)["layout"][DECODER])


def evaluate(root, freeze_path, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("use a new output directory")
    frozen = _checked_freeze(root, freeze_path)  # Before any TEST/REAL read.
    results, scored_rows, references = {}, {}, {}
    for arm in ARMS:
        results[arm], arm_checkpoint = {}, None
        for split in ("test", "real"):
            row_freeze = frozen["arms"][arm]
            rows, identities, checkpoint_id, source = _read_external(root, arm, split, row_freeze)
            if split in references and references[split] != identities:
                raise ValueError(split + " pair ID/label/ordered endpoint/strict identities differ across arms")
            references[split] = identities
            if arm_checkpoint is not None and arm_checkpoint != checkpoint_id:
                raise ValueError("TEST/REAL identify different arm checkpoints")
            arm_checkpoint = checkpoint_id
            metrics = population_metrics(rows, row_freeze["threshold"], row_freeze["classifier_thresholds"]["fused"])
            if split == "real":
                metrics["strict_summary"] = population_metrics([row for row in rows if row["strict_member"]],
                    row_freeze["threshold"], row_freeze["classifier_thresholds"]["fused"])
            metrics.update(status="complete", arm=arm, split=split, checkpoint_sha256=checkpoint_id,
                           policy=POLICY, selected_full_decoder=DECODER, source=source)
            results[arm][split] = metrics
            scored_rows[arm, split] = [dict(pair_id=row["pair_id"], label=row["label"],
                fragment_a=row["fragment_a"], fragment_b=row["fragment_b"],
                mean_probability=float(score), threshold=row_freeze["threshold"], accepted=bool(score >= row_freeze["threshold"]),
                original_fused_probability=row["classification"]["fused"], layout=row["layouts"][DECODER],
                **({"strict_member": row["strict_member"], "case_id": row["case_id"]} if split == "real" else {}))
                for row, score in zip(rows, mean_scores(rows))]
    payload = dict(schema_version=SCHEMA, status="complete", phase="evaluate", policy=POLICY,
                   validation_freeze=str(Path(freeze_path).resolve()), validation_freeze_sha256=sha256(freeze_path),
                   exploration_disclosure=DISCLOSURE, thresholds_fit="VAL only", weights_fit=False,
                   model_executed=False, logits_reconstructed=False, layout_recomputed=False,
                   original_files_modified=False, test_or_real_used_for_fit=False,
                   joint_definition="TP=accepted and true-positive and valid pose error<=tolerance; FP=accepted-TP; FN=all positives-TP",
                   arms=results)
    # No partial usable output if any of the six completed source populations fail.
    output.mkdir(parents=True, exist_ok=False)
    for (arm, split), rows in scored_rows.items():
        directory = output / arm / split
        directory.mkdir(parents=True)
        write_json(directory / "metrics.json", results[arm][split])
        with (directory / "pair_scores.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "summary.json", payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    for name in ("freeze", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--root", required=True)
        command.add_argument("--output", required=True, help="new exclusive directory")
        if name == "evaluate":
            command.add_argument("--freeze", required=True)
    args = parser.parse_args(argv)
    result = freeze(args.root, args.output) if args.phase == "freeze" else evaluate(args.root, args.freeze, args.output)
    print(json.dumps(dict(status=result["status"], phase=args.phase, output=args.output)), flush=True)


if __name__ == "__main__":
    main()
