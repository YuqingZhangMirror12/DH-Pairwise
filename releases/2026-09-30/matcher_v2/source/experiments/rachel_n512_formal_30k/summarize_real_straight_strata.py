"""Join fixed REAL straight-edge strata to completed native-model predictions.

Executable diagnostic companion, not a new model/threshold search. Presence of
opposing straight arcs is a geometry proxy, NOT a straight ground-truth seam.
Balanced1016 and original strict547 stay separate; orientation cuts overlap.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "rachel-real-straight-comparison/1"
DECODER = "full_top2_mode"
BRANCHES = ("coarse", "local", "fused")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def groups(row):
    result = ["all", "signature:" + row["orientation_signature"]]
    if not row["valid_geometry"]:
        return result + ["unavailable"]
    orientations = row["matching_orientations"]
    return result + (["any_straight"] + list(orientations) if orientations else ["no_straight"])


def populations(row):
    return ["balanced1016", "strict547" if row["strict_member"] else "constructed_cross469"]


def load_geometry(path):
    value = read(path)
    if value.get("schema_version") != "rachel-real-straight-strata/1" or value.get("status") != "complete":
        raise ValueError("requires completed frozen REAL geometry strata")
    rows = value["rows"]
    if (len(rows), sum(r["label"] for r in rows), sum(r["strict_member"] for r in rows)) != (1016,508,547):
        raise ValueError("geometry population differs from REAL1016/strict547")
    if len({r["pair_id"] for r in rows}) != 1016 or any(type(r["label"]) is not bool for r in rows):
        raise ValueError("invalid pair identities or labels")
    if any(r["label"] and not r["strict_member"] for r in rows):
        raise ValueError("all REAL positives must be strict members")
    for row in rows:
        orientations = row["matching_orientations"]
        if row["valid_geometry"]:
            if not isinstance(orientations,list) or orientations != sorted(set(orientations)) or set(orientations)-{"horizontal","vertical","oblique"}:
                raise ValueError("invalid known orientation membership")
            if row["orientation_signature"] != ("+".join(orientations) or "none"):
                raise ValueError("orientation signature disagrees")
        elif orientations is not None or row["orientation_signature"] != "unavailable":
            raise ValueError("unknown geometry cannot become no-straight")
    return value


def load_evaluation(root, geometry):
    root = Path(root)
    summary, receipt = read(root/"summary.json"), read(root/"receipt.json")
    if any(value.get("status") != "complete" or value.get("split") != "real" for value in (summary,receipt)):
        raise ValueError("requires completed REAL evaluation")
    if summary.get("test_or_real_used_for_fit") is not False or receipt.get("test_or_real_used_for_fit") is not False:
        raise ValueError("evaluation does not declare held-out-only use")
    thresholds = summary["branch_validation_thresholds"]
    if set(thresholds) != set(BRANCHES) or any(not isinstance(t,(int,float)) or not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds.values()):
        raise ValueError("invalid frozen branch thresholds")
    if (receipt.get("branch_validation_thresholds") != thresholds
            or any(value.get("selected_full_decoder") != DECODER for value in (summary,receipt))
            or summary.get("original_fused_threshold") != thresholds["fused"]):
        raise ValueError("threshold or decoder mismatch")
    if not summary.get("checkpoint_sha256") or summary["checkpoint_sha256"] != receipt.get("checkpoint_sha256"):
        raise ValueError("checkpoint identity mismatch")
    row_path = root/"pair_results.jsonl"
    if digest(row_path) != receipt.get("pair_results_sha256"):
        raise ValueError("saved pair results differ from completion receipt")
    rows = [json.loads(line) for line in row_path.read_text().splitlines() if line.strip()]
    mapping = {row["pair_id"]:row for row in rows}
    if len(rows) != 1016 or len(mapping) != 1016 or set(mapping) != {r["pair_id"] for r in geometry["rows"]}:
        raise ValueError("prediction population differs from fixed geometry")
    for item in geometry["rows"]:
        row = mapping[item["pair_id"]]
        if any(row[key] != item[key] for key in ("fragment_a","fragment_b","label","strict_member")):
            raise ValueError("ordered endpoint/label mismatch")
        for branch in BRANCHES:
            score = row["classification"][branch]
            if not isinstance(score,(int,float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("nonfinite/invalid probability")
        layout = row["layouts"][DECODER]
        error = layout.get("translation_l2_px")
        if row["label"] and layout["valid"] and (not isinstance(error,(int,float)) or not math.isfinite(error) or error < 0):
            raise ValueError("valid positive layout lacks finite error")
    counts = metrics(rows, thresholds)["branches"]["fused"]
    expected = summary["classification"]["fused"]["at_original_frozen_threshold"]
    if any(counts[key] != expected[key] for key in ("tp","fp","fn","tn")):
        raise ValueError("recomputed top-line confusion differs from source summary")
    return mapping, thresholds, dict(evaluation_root=str(root.resolve()),
        summary_sha256=digest(root/"summary.json"),receipt_sha256=digest(root/"receipt.json"),
        pair_results_sha256=digest(row_path),checkpoint_sha256=summary.get("checkpoint_sha256"))


def correct_layout(row):
    layout = row["layouts"][DECODER]
    error = layout.get("translation_l2_px")
    return bool(row["label"] and layout["valid"] and error is not None and error <= 10.)


def metrics(rows, thresholds):
    positive, negative = sum(r["label"] for r in rows), sum(not r["label"] for r in rows)
    output = dict(sample_count=len(rows),positive_count=positive,negative_count=negative,branches={})
    raw = sum(correct_layout(row) for row in rows)
    output.update(raw_layout10_count=raw,raw_layout10_recall=fraction(raw,positive))
    for branch,threshold in thresholds.items():
        accepted = [r for r in rows if r["classification"][branch] >= threshold]
        tp,fp = sum(r["label"] for r in accepted),sum(not r["label"] for r in accepted)
        joint = sum(correct_layout(r) for r in accepted)
        output["branches"][branch] = dict(threshold=threshold,tp=tp,fp=fp,fn=positive-tp,tn=negative-fp,
            recall=fraction(tp,positive),false_positive_rate=fraction(fp,negative),
            precision=fraction(tp,tp+fp),joint_layout10_count=joint,joint_layout10_recall=fraction(joint,positive),
            rejected_correct_layout10=raw-joint)
    return output


def summarize(strata, evaluations, output):
    geometry = load_geometry(strata)
    memberships = {}
    for row in geometry["rows"]:
        for population in populations(row):
            for group in groups(row):
                memberships.setdefault(population+"/"+group,[]).append(row["pair_id"])
    models = {}
    for name,path in evaluations:
        if not name or name in models:
            raise ValueError("model labels must be unique and nonempty")
        mapping,thresholds,source = load_evaluation(path,geometry)
        models[name] = dict(source=source,thresholds=thresholds,
            groups={group:metrics([mapping[i] for i in ids],thresholds) for group,ids in memberships.items()})
    result = dict(schema_version=SCHEMA,status="complete",geometry_manifest=str(Path(strata).resolve()),
        geometry_manifest_sha256=digest(strata),models=models,model_names=list(models),
        group_pair_ids=memberships,thresholds_fitted=False,model_inference=False,
        caveats=["Orientations overlap; only signature groups partition each population.",
            "Opposing straight-edge presence is a fixed geometry proxy, not an annotated straight seam or proof of the reason for a prediction.",
            "Strict547 contains508positive39negative; the additional469cross-case negatives are reported separately.",
            "Each model keeps its own frozen clean-VAL thresholds; differences combine score ranking and calibration effects.",
            "REAL informed earlier qualitative research design; this is repeated research evaluation, not a new blind test.",
            "Undefined rates are null; small strata do not establish statistically significant improvement."])
    destination = Path(output)
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strata",required=True)
    p.add_argument("--evaluation",action="append",required=True,help="NAME=completed REAL evaluation directory")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    values=[value.split("=",1) for value in args.evaluation]
    if any(len(v)!=2 for v in values):
        p.error("evaluations must be NAME=path")
    result=summarize(args.strata,values,args.output)
    print(json.dumps({name:model["groups"]["balanced1016/all"] for name,model in result["models"].items()}))
