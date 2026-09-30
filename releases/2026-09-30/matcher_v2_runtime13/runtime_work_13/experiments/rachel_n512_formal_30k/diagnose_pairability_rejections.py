"""Bounded diagnostic of saved E0/E1 REAL rejections; no fitting or inference.

Each model retains its own frozen clean-VAL coarse/local/fused thresholds.
The coarse-OR-local rule is descriptive only, not a validated replacement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k import summarize_real_straight_strata as source

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("coarse_accept_local_accept", "coarse_accept_local_reject",
          "coarse_reject_local_accept", "coarse_reject_local_reject")


def decisions(row, thresholds):
    return {branch: row["classification"][branch] >= threshold for branch, threshold in thresholds.items()}


def four_groups(rows, thresholds):
    result = dict.fromkeys(GROUPS, 0)
    for row in rows:
        accepted = decisions(row, thresholds)
        key = "coarse_" + ("accept" if accepted["coarse"] else "reject")
        key += "_local_" + ("accept" if accepted["local"] else "reject")
        result[key] += 1
    assert sum(result.values()) == len(rows)
    return dict(total=len(rows), groups=result)


def metrics(rows, thresholds):
    rows = list(rows)
    counts = dict(sample_count=len(rows), positive_count=sum(row["label"] for row in rows),
                  negative_count=sum(not row["label"] for row in rows))
    fused = dict(tp=0, fp=0, fn=0, tn=0)
    rule = dict(tp=0, fp=0, fn=0, tn=0, extra_true_acceptances=0, extra_false_acceptances=0,
                lost_true_acceptances=0, lost_false_acceptances=0)
    rejected, rejected_correct = [], []
    for row in rows:
        accept = decisions(row, thresholds)
        label, either = row["label"], accept["coarse"] or accept["local"]
        fused[("t" if accept["fused"] == label else "f") + ("p" if accept["fused"] else "n")] += 1
        rule[("t" if either == label else "f") + ("p" if either else "n")] += 1
        kind = "true" if label else "false"
        if either and not accept["fused"]:
            rule["extra_" + kind + "_acceptances"] += 1
        if accept["fused"] and not either:
            rule["lost_" + kind + "_acceptances"] += 1
        if label and not accept["fused"]:
            rejected.append(row)
            if source.correct_layout(row):
                rejected_correct.append(row)
    for kind, cell in (("true", "tp"), ("false", "fp")):
        assert rule[cell] - fused[cell] == rule["extra_" + kind + "_acceptances"] - rule["lost_" + kind + "_acceptances"]
    assert len(rejected) == fused["fn"] and sum(fused.values()) == len(rows)
    return dict(counts, fused_confusion=fused, fused_false_negatives=four_groups(rejected, thresholds),
                fused_rejected_correct_layout10=four_groups(rejected_correct, thresholds),
                coarse_or_local_diagnostic=rule)


def transitions(ids, before, before_thresholds, after, after_thresholds):
    positive = dict(rescued_false_negative=0, new_false_negative=0, retained_true_positive=0, retained_false_negative=0)
    negative = dict(eliminated_false_positive=0, new_false_positive=0, retained_true_negative=0, retained_false_positive=0)
    for pair_id in ids:
        a, b = decisions(before[pair_id], before_thresholds)["fused"], decisions(after[pair_id], after_thresholds)["fused"]
        if before[pair_id]["label"]:
            key = {(False, True): "rescued_false_negative", (True, False): "new_false_negative",
                   (True, True): "retained_true_positive", (False, False): "retained_false_negative"}[(a, b)]
            positive[key] += 1
        else:
            key = {(True, False): "eliminated_false_positive", (False, True): "new_false_positive",
                   (False, False): "retained_true_negative", (True, True): "retained_false_positive"}[(a, b)]
            negative[key] += 1
    assert sum(positive.values()) + sum(negative.values()) == len(ids)
    return dict(positive=positive, negative=negative,
                net_tp_change=positive["rescued_false_negative"] - positive["new_false_negative"],
                net_fp_change=negative["new_false_positive"] - negative["eliminated_false_positive"])


def diagnose(snapshot, strata, output, evaluations=None):
    snapshot, output = Path(snapshot).resolve(strict=evaluations is None), Path(output).resolve()
    if (output / "summary.json").exists():
        raise FileExistsError("diagnostic summary already exists; use a new output directory")
    geometry = source.load_geometry(strata)
    membership = {"balanced1016": [row["pair_id"] for row in geometry["rows"]],
                  "strict547": [row["pair_id"] for row in geometry["rows"] if row["strict_member"]],
                  "constructed469": [row["pair_id"] for row in geometry["rows"] if not row["strict_member"]]}
    roots = dict(e0=snapshot / "e0/realism_evaluation/matched24k/real", e1=snapshot / "e1/evaluation/real")
    if evaluations is not None:
        roots = {}
        for name, path in evaluations:
            name = name.strip().lower()
            if not name or name in roots:
                raise ValueError("evaluation names must be unique and nonempty")
            roots[name] = Path(path).resolve(strict=True)
        if not roots:
            raise ValueError("at least one completed evaluation is required")
    loaded, models = {}, {}
    for name, root in roots.items():
        mapping, thresholds, provenance = source.load_evaluation(root, geometry)
        # In addition to the shared fused check, reproduce the two branch
        # confusion matrices at their exact saved VAL operating points.
        summary = source.read(root / "summary.json")
        recomputed = source.metrics(list(mapping.values()), thresholds)["branches"]
        for branch in source.BRANCHES:
            expected = summary["classification"][branch]["at_validation_row_f1_threshold"]
            if expected["threshold"] != thresholds[branch] or any(recomputed[branch][key] != expected[key] for key in ("tp", "fp", "fn", "tn")):
                raise ValueError("saved branch operating point does not reproduce: " + name + "/" + branch)
        loaded[name] = (mapping, thresholds)
        models[name] = dict(source=provenance, thresholds=thresholds,
            populations={population: metrics((mapping[pair_id] for pair_id in ids), thresholds)
                         for population, ids in membership.items()})
    changes = ({population: transitions(ids, *loaded["e0"], *loaded["e1"])
                for population, ids in membership.items()} if {"e0", "e1"} <= set(loaded) else {})
    for population, change in changes.items():
        a, b = (models[name]["populations"][population]["fused_confusion"] for name in ("e0", "e1"))
        assert change["net_tp_change"] == b["tp"] - a["tp"]
        assert change["net_fp_change"] == b["fp"] - a["fp"]
    result = dict(schema_version="rachel-pairability-rejection-diagnostic/1", status="complete", models=models,
        fused_e0_to_e1_transitions=changes, diagnostic_only=True, model_inference=False, training=False,
        thresholds_fitted=False, geometry_manifest=str(Path(strata).resolve()), geometry_manifest_sha256=source.digest(strata),
        geometry_used_for_membership_only=True, source_snapshot=str(snapshot) if evaluations is None else None,
        definitions=dict(acceptance="saved probability >= that model's own frozen clean-VAL branch threshold",
            layout10="true positive with valid saved full_top2_mode layout and translation_l2_px <= 10",
            four_groups="mutually exclusive coarse/local decisions; counts sum to the stated rejected-positive population",
            extra_acceptance="coarse OR local accepts while fused rejects", lost_acceptance="fused accepts while coarse OR local rejects"),
        caveats=["Saved outputs only; no new scores, training, inference, or threshold tuning.",
            "Four-way buckets describe branch decision disagreements, not causal explanations of fusion behavior.",
            "Each model uses its own frozen VAL thresholds, so E0/E1 transitions combine scoring and calibration changes.",
            "Coarse OR local is a post-hoc diagnostic, not a validated or selected new model or deployment rule.",
            "Layout correctness uses the already-attached saved GT error; no GT source file is opened here.",
            "All508 positives are strict members; strict547 has39 negatives and constructed469 adds only cross-case negatives.",
            "REAL has informed repeated research diagnostics; results are not a new blind confirmation or a causal attribution.",
            "The geometry manifest supplies aligned identities/membership only; no orientation threshold or geometry is changed."])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", default=str(ROOT / "reports/rachel_weathering_all_20260909_001/e0_e1_completed_snapshot"))
    parser.add_argument("--strata", default=str(ROOT / "reports/rachel_straight_strata_20260910_001/geometry/geometry_manifest.json"))
    parser.add_argument("--evaluation", action="append", help="NAME=completed REAL directory; repeat to replace defaults (names lowercased)")
    parser.add_argument("--output", default=str(ROOT / "reports/rachel_pairability_rejections_20260910_001"))
    args = parser.parse_args()
    evaluations = [value.split("=", 1) for value in args.evaluation] if args.evaluation else None
    if evaluations is not None and any(len(value) != 2 for value in evaluations):
        parser.error("evaluations must be NAME=path")
    result = diagnose(args.snapshot_root, args.strata, args.output, evaluations)
    print(json.dumps({name: model["populations"] for name, model in result["models"].items()}, sort_keys=True))
    print(json.dumps(result["fused_e0_to_e1_transitions"], sort_keys=True))
