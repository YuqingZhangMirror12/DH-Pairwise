"""Offline paired classification transitions at each model's frozen SIMVAL OPs.

No inference, threshold fitting, population intersection, or scorer-causality
claim. CLI: --reference DIR --new DIR --split test|real|ood --output NEW_DIR.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.endpoint_compare_v1 import compare as base

POSITIVE_TRANSITIONS = ("FN_to_TP", "TP_to_FN", "TP_to_TP", "FN_to_FN")
NEGATIVE_TRANSITIONS = ("FP_to_TN", "TN_to_FP", "TN_to_TN", "FP_to_FP")
LAYOUT_STATES = ("good", "bad", "unknown")


def layout_label(row, split):
    # Neither OOD nor negative pairs have a valid GT layout-success question.
    if split == "ood" or not row["label"]:
        return "unknown"
    return {True: "good", False: "bad", None: "unknown"}[base.layout_state(row)]


def same_final_layout(reference, new, split):
    """Exact valid decoded translation equality, not merely two Layout20 passes."""
    if split == "ood" or not reference["label"]:
        return False
    layouts = [r.get("layouts", {}).get("full_top2_mode", {}) for r in (reference, new)]
    translations = [r.get("translation_rc") for r in layouts]
    return (all(r.get("valid") is True for r in layouts)
            and all(isinstance(t, list) and len(t) == 2 and all(base.finite(v) for v in t)
                    for t in translations)
            and translations[0] == translations[1])


def classification_state(row, threshold):
    return ("TP" if row["label"] else "FP") if base.accepted(row, threshold) else (
        "FN" if row["label"] else "TN")


def bucket(ids):
    return dict(count=len(ids), pair_ids=sorted(ids))


def transition_summary(cases, op, names, with_layout):
    result = {}
    for name in names:
        selected = [r for r in cases if r["operating_points"][op]["transition"] == name]
        cell = bucket([r["pair_id"] for r in selected])
        if with_layout:
            cell["layout20_cross"] = {
                a+"_to_"+b: bucket([r["pair_id"] for r in selected
                                     if (r["reference_layout20"], r["new_layout20"]) == (a, b)])
                for a in LAYOUT_STATES for b in LAYOUT_STATES}
            cell["identical_valid_final_layout"] = bucket(
                [r["pair_id"] for r in selected if r["identical_valid_final_layout"]])
            if sum(x["count"] for x in cell["layout20_cross"].values()) != cell["count"]:
                raise AssertionError("layout transition counts do not conserve")
        result[name] = cell
    return result


def population_summary(cases, op, split):
    positive = [r for r in cases if r["label"]]
    negative = [r for r in cases if not r["label"]]
    pos = transition_summary(positive, op, POSITIVE_TRANSITIONS, True)
    neg = transition_summary(negative, op, NEGATIVE_TRANSITIONS, False)
    reference_tp = pos["TP_to_TP"]["count"] + pos["TP_to_FN"]["count"]
    new_tp = pos["TP_to_TP"]["count"] + pos["FN_to_TP"]["count"]
    conserved = (sum(r["count"] for r in pos.values()) == len(positive)
                 and sum(r["count"] for r in neg.values()) == len(negative)
                 and new_tp-reference_tp == pos["FN_to_TP"]["count"]-pos["TP_to_FN"]["count"])
    if not conserved:
        raise AssertionError("paired classification counts do not conserve")
    result = dict(sample_count=len(cases), positive_count=len(positive), negative_count=len(negative),
                  positive_transitions=pos, reference_true_positive=reference_tp,
                  new_true_positive=new_tp, net_positive_rescues=new_tp-reference_tp,
                  reference_recall=reference_tp/len(positive) if positive else None,
                  new_recall=new_tp/len(positive) if positive else None,
                  counts_conserved=conserved)
    if split == "ood":
        result.update(positive_only=True, layout_available=False,
                      unavailable="No negative classification metrics or GT layout for OOD")
    else:
        reference_fp = neg["FP_to_TN"]["count"] + neg["FP_to_FP"]["count"]
        new_fp = neg["TN_to_FP"]["count"] + neg["FP_to_FP"]["count"]
        result.update(negative_transitions=neg, reference_false_positive=reference_fp,
                      new_false_positive=new_fp, net_false_positive_reduction=reference_fp-new_fp)
    return result


def compare_endpoints(reference, new, split):
    if split not in base.EXPECTED:
        raise ValueError("unsupported split")
    endpoints = {name: base.load_endpoint(Path(path), split)
                 for name, path in (("reference", reference), ("new", new))}
    for name, endpoint in endpoints.items():
        if endpoint["status"] != "complete":
            raise ValueError(name + " endpoint unavailable: " + endpoint.get("reason", "unknown"))
    a, b = endpoints["reference"]["rows"], endpoints["new"]["rows"]
    identity = base.identity_difference(a, b, split)
    if not identity["equal"]:
        raise ValueError("paired identity mismatch; no intersection: " + json.dumps(identity))
    cases = []
    for pair_id in sorted(a):
        old, current = a[pair_id], b[pair_id]
        record = dict(pair_id=pair_id, label=bool(old["label"]),
                      reference_score=old["classification"]["fused"],
                      new_score=current["classification"]["fused"],
                      reference_decision_valid=old["decision_valid"],
                      new_decision_valid=current["decision_valid"],
                      reference_layout20=layout_label(old, split),
                      new_layout20=layout_label(current, split),
                      identical_valid_final_layout=same_final_layout(old, current, split),
                      operating_points={})
        for op in base.OPS:
            reference_state = classification_state(old, endpoints["reference"]["thresholds"][op])
            new_state = classification_state(current, endpoints["new"]["thresholds"][op])
            record["operating_points"][op] = dict(reference=reference_state, new=new_state,
                                                   transition=reference_state+"_to_"+new_state)
        cases.append(record)
    indexed = {r["pair_id"]: r for r in cases}
    groups = {}
    for group, selected in base.populations(a.values(), split).items():
        counts = (len(selected), sum(bool(r["label"]) for r in selected), sum(not r["label"] for r in selected))
        expected = base.REAL_COUNTS[group] if split == "real" else base.EXPECTED[split]
        if counts != expected:
            raise ValueError("cohort count mismatch " + group + ": " + str(counts))
        rows = [indexed[r["pair_id"]] for r in selected]
        groups[group] = {op: population_summary(rows, op, split) for op in base.OPS}
    return dict(schema="offline-paired-classification-changes/1", status="complete", split=split,
                created_at_utc=datetime.now(timezone.utc).isoformat(), identity_check=identity,
                threshold_mode="own_frozen_SIMVAL_each_model_no_refit",
                thresholds={name: e["thresholds"] for name, e in endpoints.items()},
                endpoints={name: {k: e[k] for k in ("path", "model", "sources")} for name, e in endpoints.items()},
                populations=groups, cases=cases, no_inference=True, no_threshold_fit=True,
                definitions=dict(layout20="positive GT, valid decoder and error<=20px; absent evidence stays unknown",
                                 identical_valid_final_layout="exact same valid translation_rc, not merely same good/bad state",
                                 invalid_decision="rejected regardless of finite score"),
                caveats=["Own thresholds may differ; these are operating-point decision changes, not pure raw-score changes.",
                         "Reference and new Layout20 states are reported separately; different Matchers can change layout.",
                         "Even identical decoded layouts do not establish unchanged Matcher features or causal scorer attribution.",
                         "Reviewed REAL subsets are post-review populations, not fresh untouched tests."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(base.EXPECTED), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_endpoints(args.reference, args.new, args.split)
    args.output.mkdir(parents=True, exist_ok=False)
    cases = result.pop("cases")
    (args.output/"cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in cases))
    (args.output/"summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
