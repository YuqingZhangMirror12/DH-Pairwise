"""Join completed hard-SIMVAL Matcher/GT-geometry diagnostics, full population only.

Read-only postprocessing of existing JSON evidence. No inference, fitting,
training or queue operations. GT-edge decoding is not a performance upper bound.
Clean/requested copies are paired by source, not independent observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "s7-hard-simval-model-oracle-join/1"
MODEL_SCHEMA = "s7-hard-simval-matcher-evaluation/1"
ORACLE_SCHEMA = "s7-hard-simval-gt-correspondence-geometry/1"
LOSS_NAMES = ("assignment_nll", "match_nll", "dustbin_nll")
CELLS = ("both_success", "model_failure_oracle_success", "model_success_oracle_failure", "both_failure")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return dict(denominator=len(values), mean=sum(values)/len(values) if values else None)


def valid_success(valid, error, stored):
    if error is not None and (not isinstance(error, (int, float)) or not math.isfinite(error) or error < 0):
        raise ValueError("nonfinite or negative layout error")
    success = bool(valid and error is not None and error <= 20.)
    if bool(stored) != success or bool(valid) != (error is not None):
        raise ValueError("positive layout validity/error/success disagree")
    return success


def join_rows(model_rows, oracle_rows):
    model = {r["pair_id"]: r for r in model_rows}
    oracle = {r["pair_id"]: r for r in oracle_rows}
    positive_ids = {r["pair_id"] for r in model_rows if r["label"]}
    if len(model) != len(model_rows) or len(oracle) != len(oracle_rows) or positive_ids != set(oracle):
        raise ValueError("unique IDs and exact positive membership are required")
    joined, sources = [], defaultdict(list)
    for row in model_rows:
        if row["label"] not in (0, 1):
            raise ValueError("binary model labels required")
        pose = bool(row["label"] and not row["changed_pair"])
        if bool(row["pose_supervision_enabled"]) != pose or bool(row["pose_supervised"]) != (pose and bool(row["training_valid"])):
            raise ValueError("pose eligibility differs from recorded damage/training validity")
        if any(not math.isfinite(row["losses"][key]) for key in (*LOSS_NAMES, "translation_smooth_l1")):
            raise ValueError("nonfinite recorded loss")
        evidence = oracle.get(row["pair_id"])
        if evidence is not None:
            for key in ("source_pair_id", "recipe", "changed_pair", "fallback_reason", "source_family_overlap"):
                if row.get(key) != evidence.get(key):
                    raise ValueError("model/oracle metadata differs: " + key)
            valid_success(row["raw_layout_valid"], row["raw_translation_l2_px"], row["raw_layout20_correct"])
            valid_success(evidence["oracle_layout_valid"], evidence["oracle_error_px"], evidence["oracle_layout20_success"])
        result = dict(row, oracle=evidence)
        joined.append(result)
        sources[row["source_pair_id"]].append(result)
    for source, group in sources.items():
        if (len(group) != 2 or sum(r["recipe"] == "clean" for r in group) != 1
                or len({bool(r["label"]) for r in group}) != 1
                or any(r["changed_pair"] for r in group if r["recipe"] == "clean")
                or len({r.get("source_family_overlap") for r in group}) != 1):
            raise ValueError("source must have matching clean/requested copies: " + source)
    return joined


def loss_summary(rows):
    valid = [r for r in rows if r["training_valid"]]
    matched = [r for r in valid if r["supervised_correspondence_count"] > 0]
    pose = [r for r in rows if r["pose_supervised"]]
    return dict(all_rows=len(rows), training_valid_rows=len(valid),
        all_row_means={k: mean([r["losses"][k] for r in rows]) for k in LOSS_NAMES},
        training_valid_means={k: mean([r["losses"][k] for r in valid]) for k in LOSS_NAMES},
        match_supervised_pair_mean=mean([r["losses"]["match_nll"] for r in matched]),
        supervised_correspondence_tokens=sum(r["supervised_correspondence_count"] for r in matched),
        pose_enabled_rows=sum(r["pose_supervision_enabled"] for r in rows),
        pose_effectively_supervised_rows=len(pose), pose_disabled_rows=sum(not r["pose_supervision_enabled"] for r in rows),
        pose_supervised_smooth_l1=mean([r["losses"]["translation_smooth_l1"] for r in pose]),
        dustbin_active_token_denominator=None,
        semantics="Pair-weighted recorded terms; invalid/unsupervised zeros remain in all-row means. Dustbin active token/side counts were not recorded. Pose is reported only on effectively supervised rows; total loss is intentionally not compared.")


def cohort(rows):
    positive = [r for r in rows if r["label"]]
    cells = Counter()
    for row in positive:
        m, o = row["raw_layout20_correct"], row["oracle"]["oracle_layout20_success"]
        cells["both_success" if m and o else "model_failure_oracle_success" if o
              else "model_success_oracle_failure" if m else "both_failure"] += 1
    return dict(count=len(rows), positive_layout_denominator=len(positive),
        four_cells={key: cells[key] for key in CELLS},
        model_layout20_success=sum(r["raw_layout20_correct"] for r in positive),
        oracle_layout20_success=sum(r["oracle"]["oracle_layout20_success"] for r in positive),
        model_invalid_positive_count=sum(not r["raw_layout_valid"] for r in positive),
        oracle_invalid_positive_count=sum(not r["oracle"]["oracle_layout_valid"] for r in positive),
        fallback_reasons=dict(Counter(r["fallback_reason"] for r in rows if r["fallback_reason"])),
        losses={name: loss_summary(group) for name, group in
            (("all", rows), ("positive", positive), ("negative", [r for r in rows if not r["label"]]))})


def paired(pairs):
    positives = [(a, b) for a, b in pairs if b["label"]]
    def transitions(oracle=False):
        values = Counter()
        for a, b in positives:
            x, y = ((r["oracle"]["oracle_layout20_success"] if oracle else r["raw_layout20_correct"]) for r in (a, b))
            values[("success" if x else "failure") + "_to_" + ("success" if y else "failure")] += 1
        return {key: values[key] for key in ("success_to_success", "success_to_failure", "failure_to_success", "failure_to_failure")}
    deltas = {}
    for key in (*LOSS_NAMES, "translation_smooth_l1"):
        if key == "translation_smooth_l1":
            eligible = [(a, b) for a, b in pairs if a["pose_supervised"] and b["pose_supervised"]]
        else:
            eligible = [(a, b) for a, b in pairs if a["training_valid"] and b["training_valid"]
                and (key != "match_nll" or min(a["supervised_correspondence_count"], b["supervised_correspondence_count"]) > 0)]
        deltas[key] = mean([b["losses"][key] - a["losses"][key] for a, b in eligible])
    return dict(source_count=len(pairs), positive_source_count=len(positives),
        clean=cohort([a for a, _ in pairs]), requested_variant=cohort([b for _, b in pairs]),
        model_clean_to_variant=transitions(), oracle_clean_to_variant=transitions(True),
        paired_loss_delta_variant_minus_clean= deltas,
        delta_denominators="Both training-valid for assignment/dustbin; both have matched supervision for match; both effectively pose-supervised for pose. No token-pooled interpretation.")


def summarize(rows, *, include_family=True):
    clean = {r["source_pair_id"]: r for r in rows if r["recipe"] == "clean"}
    variants = [r for r in rows if r["recipe"] != "clean"]
    pairs = [(clean[r["source_pair_id"]], r) for r in variants]
    recipes = sorted({r["recipe"] for r in variants})
    changed_keys = sorted({(r["recipe"], r["changed_pair"]) for r in rows})
    result = dict(paired_all=paired(pairs),
        by_requested_recipe={recipe: paired([(a, b) for a, b in pairs if b["recipe"] == recipe]) for recipe in recipes},
        by_recipe_and_actual_change={recipe+"|changed="+str(changed).lower(): cohort(
            [r for r in rows if r["recipe"] == recipe and r["changed_pair"] == changed]) for recipe, changed in changed_keys})
    if include_family:
        disjoint = [r for r in rows if r.get("source_family_overlap") is False]
        result["optional_source_family_disjoint"] = dict(row_count=len(disjoint),
            overlapping_rows=sum(r.get("source_family_overlap") is True for r in rows),
            unknown_rows=sum(r.get("source_family_overlap") is None for r in rows),
            metrics=summarize(disjoint, include_family=False))
    return result


def load_sources(model_dir, oracle_dir):
    model_dir, oracle_dir = Path(model_dir).resolve(strict=True), Path(oracle_dir).resolve(strict=True)
    ms, mp = read(model_dir/"summary.json"), read(model_dir/"protocol.json")
    os, op = read(oracle_dir/"summary.json"), read(oracle_dir/"protocol.json")
    if any(record.get("status") != "complete" for record in (ms, mp, os, op)):
        raise ValueError("full completed source summaries/protocols required; no pilot or partial join")
    if (any(r.get("schema") != MODEL_SCHEMA for r in (ms, mp))
            or any(r.get("schema") != ORACLE_SCHEMA for r in (os, op))
            or len({r.get("manifest_sha256") for r in (ms, mp, os, op)}) != 1
            or not isinstance(ms.get("manifest_sha256"), str) or len(ms["manifest_sha256"]) != 64
            or ms.get("model") != mp.get("model") or ms.get("model", {}).get("epoch") not in (12, 16, 20)):
        raise ValueError("schemas, same manifest or fixed Matcher identity differ")
    paths = dict(model_rows=model_dir/"pair_metrics.jsonl", model_summary=model_dir/"summary.json",
                 oracle_rows=oracle_dir/"rows.jsonl", oracle_summary=oracle_dir/"summary.json")
    hashes = {key: sha(path) for key, path in paths.items()}
    if (hashes["model_rows"] != ms.get("pair_metrics_sha256")
            or hashes["model_summary"] != mp.get("summary_sha256")
            or hashes["oracle_rows"] != os.get("rows_sha256")):
        raise ValueError("source result hashes differ")
    model_rows = [json.loads(s) for s in paths["model_rows"].read_text().splitlines() if s]
    oracle_rows = [json.loads(s) for s in paths["oracle_rows"].read_text().splitlines() if s]
    if (len(model_rows) != 6000 or len(oracle_rows) != 3000 or sum(r["label"] for r in model_rows) != 3000
            or ms.get("count") != 6000 or mp.get("completed_count") != 6000
            or op.get("completed_count") != 3000 or os.get("metrics", {}).get("positive_source_count") != 1500):
        raise ValueError("requires all6000 model rows and all3000 positive oracle rows")
    joined = join_rows(model_rows, oracle_rows)
    return joined, dict(model_dir=str(model_dir), oracle_dir=str(oracle_dir), hashes=hashes,
        model_protocol_sha256=sha(model_dir/"protocol.json"), oracle_protocol_sha256=sha(oracle_dir/"protocol.json"),
        manifest_sha256=ms["manifest_sha256"], model=ms["model"])


def run(args):
    rows, sources = load_sources(args.model_dir, args.oracle_dir)
    output = Path(args.output).resolve()
    for root in (Path(sources["model_dir"]), Path(sources["oracle_dir"])):
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("new join output must be separate from source result trees")
    output.mkdir(parents=True, exist_ok=False)
    with (output/"joined_rows.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    result = dict(schema=SCHEMA, status="complete", sources=sources, metrics=summarize(rows),
        joined_rows_sha256=sha(output/"joined_rows.jsonl"), model_inference=False, thresholds_fitted=False,
        training_performed=False, performance_upper_bound_claimed=False,
        interpretation="Four cells describe equal-source evidence, not an oracle upper bound or cause attribution. Invalid layouts remain failures. Clean/requested copies are paired, not iid; disabled pose and total loss are not quality improvements.")
    (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "oracle-dir", "output"):
        result.add_argument("--"+name, required=True)
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps(dict(status=result["status"], source=result["sources"]["model"])) )
