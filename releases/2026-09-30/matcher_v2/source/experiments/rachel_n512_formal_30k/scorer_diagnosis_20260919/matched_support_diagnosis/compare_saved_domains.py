"""Describe archived S7 M12 Matcher support on TEST/REAL/OOD, without inference.

Read only stored candidate_count, inlier_count, weighted inlier RMS residual,
layout validity/error and labels. No Scorer score/threshold is used or fitted.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .diagnose import auc, sha

SCHEMA = "s7-m12-saved-domain-support/1"
CHECKPOINT_SHA = "7c1212e2d9d62c25954457add3f9319b03dfe8fbc42aa96116e40caae0dc2c37"
TRAIN_SHA = "79a9e959f32ef9899116e299425d6350b17f6a04bd5070b5aa9a730319447c36"
DECODER = "full_top2_mode"
EXPECTED = dict(test=(3000,1500),real=(1016,508),ood=(301,301))
DEFAULT_ROOT = Path(__file__).resolve().parents[1]/"continuation_s7_endpoint_v1/raw/s6_s7_20260915/priority_after_s5/s7_augmented_full24/evaluation/fixed_epoch"


def distribution(values):
    finite = [float(x) for x in values if x is not None and math.isfinite(x)]
    a = np.asarray(finite)
    return dict(available_count=len(finite),missing_count=len(values)-len(finite),
        mean=float(a.mean()) if len(a) else None,
        quantiles=dict(zip(("p10","p50","p90"),map(float,np.quantile(a,[.1,.5,.9])))) if len(a) else {})


def group(rows, *, layout_gt):
    positives = [r for r in rows if r["label"]]
    result = dict(count=len(rows),positive_count=len(positives),negative_count=len(rows)-len(positives),
        layout_valid_count=sum(r["layout_valid"] for r in rows),metrics={})
    for key in ("candidate_count","inlier_count","inlier_weighted_rms_px"):
        result["metrics"][key] = distribution([r[key] for r in rows])
    result["candidate_cap512_count"] = sum(r["candidate_count"]==512 for r in rows)
    y = [int(r["label"]) for r in rows]
    residual = [r for r in rows if r["inlier_weighted_rms_px"] is not None
                and math.isfinite(r["inlier_weighted_rms_px"])]
    result["diagnostic_auroc"] = dict(
        inlier_count=dict(value=auc(y,[r["inlier_count"] for r in rows]) if rows else None,
            available_count=len(rows),positive_count=len(positives),negative_count=len(rows)-len(positives),
            score_direction="higher inlier_count is more positive"),
        negative_inlier_weighted_rms_px=dict(
            value=auc([int(r["label"]) for r in residual],[-r["inlier_weighted_rms_px"] for r in residual]) if residual else None,
            available_count=len(residual),excluded_missing_count=len(rows)-len(residual),
            positive_count=sum(r["label"] for r in residual),negative_count=sum(not r["label"] for r in residual),
            score_direction="negative residual: smaller weighted inlier RMS is more positive",
            tie_policy="exact equal floating scores receive average ranks"))
    if layout_gt and positives:
        success = sum(r["layout_valid"] and r["layout_error_px"] is not None and r["layout_error_px"]<=20 for r in positives)
        result["raw_layout20"] = dict(success=success,denominator=len(positives),rate=success/len(positives),
            gt_available_count=sum(r["gt_available"] for r in positives),
            error_available_count=sum(r["layout_error_px"] is not None for r in positives),
            denominator_rule="all positive pairs; invalid or missing errors never count as success; <=20px")
    else:
        result["raw_layout20"] = None
    return result


def read_split(root, split):
    folder = root/split
    protocol = json.loads((folder/"protocol.json").read_text())
    model = protocol["model"]; identity = model["training_identity"]
    if (protocol.get("status")!="complete" or protocol.get("split")!=split
            or model.get("checkpoint_sha256")!=CHECKPOINT_SHA or model.get("epoch")!=20
            or model.get("selection")!="fixed_epoch" or protocol.get("resampling")!="original512"
            or protocol.get("decoder")!=DECODER or identity.get("schedule")!="M12_C8_pair_decoupled"
            or identity.get("matcher_epochs")!=12 or identity.get("classifier_epochs")!=8
            or identity.get("frozen_base_eval_in_classifier") is not True
            or identity["populations"]["train"].get("manifest_sha256")!=TRAIN_SHA):
        raise ValueError("not the specified frozen-S7M12 / C8 archived evaluation")
    source = [json.loads(line) for line in (folder/"pair_results.jsonl").read_text().splitlines() if line]
    ids = [r["pair_id"] for r in source]
    if len(set(ids))!=len(ids) or (len(source),sum(r["label"] for r in source))!=EXPECTED[split]:
        raise ValueError("wrong/duplicate archived population: "+split)
    if protocol["sample_count"]!=len(source):
        raise ValueError("protocol count differs")
    rows = []
    for r in source:
        layout = r["layouts"][DECODER]; d = layout["diagnostics"]
        if type(r["label"]) is not bool or type(layout["valid"]) is not bool:
            raise ValueError("label/layout validity must be boolean")
        if (type(d["candidate_count"]) is not int or type(d["inlier_count"]) is not int
                or not 0<=d["inlier_count"]<=d["candidate_count"]<=512):
            raise ValueError("invalid saved support counts")
        if d["residual_px"] is not None and (not math.isfinite(d["residual_px"]) or d["residual_px"]<0):
            raise ValueError("invalid saved residual")
        rows.append(dict(pair_id=r["pair_id"],label=r["label"],review_status=r.get("review_status"),
            candidate_count=d["candidate_count"],inlier_count=d["inlier_count"],
            inlier_weighted_rms_px=d["residual_px"],layout_valid=layout["valid"],
            layout_error_px=layout.get("translation_l2_px"),gt_available=r.get("target_translation_rc") is not None))
    provenance = dict(protocol=str(folder/"protocol.json"),protocol_sha256=sha(folder/"protocol.json"),
        pair_results=str(folder/"pair_results.jsonl"),pair_results_sha256=sha(folder/"pair_results.jsonl"),
        checkpoint_sha256=CHECKPOINT_SHA,schedule=identity["schedule"],matcher_epochs=12,classifier_epochs=8,
        frozen_base_eval_in_classifier=True,train_manifest_sha256=TRAIN_SHA,
        decoder=protocol["decoder"],decoder_config=protocol["decoder_config"],resampling="original512",
        inference_runtime={k:v for k,v in protocol["inference_runtime"].items() if k!="source_sha256"},
        binding_limit="archived protocol/checkpoint identity and frozen-phase recipe; no weights loaded or rehashed here")
    return rows,provenance


def run(root):
    root = Path(root).resolve(strict=True)
    raw, provenance = {},{}
    for split in EXPECTED:
        raw[split],provenance[split] = read_split(root,split)
    if any(provenance[s]["decoder_config"]!=provenance["test"]["decoder_config"] for s in EXPECTED):
        raise ValueError("decoder differs across populations")
    reviewed = [r for r in raw["real"] if not r["label"] or r["review_status"]=="keep"]
    if (len(reviewed),sum(r["label"] for r in reviewed))!=(803,295):
        raise ValueError("requires reviewed295 positives plus all508 negatives")
    domains = dict(TEST3000=raw["test"],REAL1016=raw["real"],REAL_reviewed803=reviewed,OOD301=raw["ood"])
    result = dict(schema=SCHEMA,status="complete",sources=provenance,cohorts={},
        metric_definition=dict(residual_px="production confidence-weighted inlier RMS distance; same formula as TRAIN inlier_weighted_rms_px",
            candidate_count="saved Top2 union capped512",inlier_count="saved final decoder inlier count",
            layout_valid="solver validity, NOT whether fragments are truly adjacent",auroc="univariate descriptive separation; no threshold fitted"),
        limitations=["TRAIN comparison uses in-sample S7 augmented TRAIN24K; TEST is clean simulation, REAL/OOD have different shape/scale populations.",
            "TRAIN caches were computed on CPU; these archives used GPU FP32. Differences are descriptive, not a controlled causal domain experiment.",
            "The stored S7 C8 checkpoint used M12 frozen Matcher; Scorer probabilities are not used here.",
            "REAL_reviewed803 was manually filtered; improved separation there cannot establish untouched real-world performance.",
            "OOD301 contains positives only and no layout GT: no AUROC, binary Accuracy/F1 or layout success claims.",
            "Missing residuals are excluded only from residual AUROC and their counts are reported; no invalid rows are dropped from inlier AUROC or raw-layout denominators.",
            "No inference, training, threshold fitting or re-selection performed."])
    for name,rows in domains.items():
        has_gt = name!="OOD301"
        result["cohorts"][name] = dict(overall=group(rows,layout_gt=has_gt),by_label={
            "positive":group([r for r in rows if r["label"]],layout_gt=has_gt),
            "negative":group([r for r in rows if not r["label"]],layout_gt=has_gt)})
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=run(args.root)
    with args.output.open("x") as output:
        json.dump(result,output,ensure_ascii=False,indent=2,allow_nan=False);output.write("\n")
    for name,value in result["cohorts"].items():
        overall=value["overall"]
        print(json.dumps(dict(cohort=name,count=overall["count"],auroc=overall["diagnostic_auroc"],
            raw_layout20=overall["raw_layout20"],by_label=value["by_label"]),ensure_ascii=False))
