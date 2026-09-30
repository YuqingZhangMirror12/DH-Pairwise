#!/usr/bin/env python3
"""Read existing fixed-epoch REAL records over SSH; never load models or data.

Only the three named pair_results/protocol files are read remotely. Local
outputs are projections and aggregations, not new predictions or fitted gates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[4]
FIELDS = ("candidate_count", "inlier_count", "inlier_fraction", "weighted_inlier_fraction",
          "residual_px", "runner_up_support_ratio", "support_weight", "runner_up_support_weight")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def describe(values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            **{name: quantile(values, q) for name, q in
               (("min", 0), ("p10", .1), ("p25", .25), ("median", .5),
                ("p75", .75), ("p90", .9), ("max", 1))}}


def summarize(rows, thresholds):
    result = {"n": len(rows), "layout_present": sum(r["layout_present"] for r in rows),
              "finite_translation": sum(r["finite_translation"] for r in rows),
              "layout_valid": sum(r["layout_valid"] for r in rows),
              "decision_valid": sum(r["decision_valid"] for r in rows),
              "reason_counts": dict(Counter(r["diagnostics"].get("reason", "MISSING") for r in rows)),
              "candidate_details_present": sum(r["candidate_details_present"] for r in rows),
              "diagnostics": {f: describe([r["diagnostics"].get(f) for r in rows]) for f in FIELDS},
              "score": describe([r["score"] for r in rows]), "descriptive_bins": {}}
    tests = {**{"inlier_count_le_%d" % n: lambda r, n=n: r["diagnostics"].get("inlier_count", math.inf) <= n
                for n in (3, 5, 10, 20)},
             "inlier_fraction_le_0.05": lambda r: r["diagnostics"].get("inlier_fraction", math.inf) <= .05,
             "inlier_fraction_le_0.10": lambda r: r["diagnostics"].get("inlier_fraction", math.inf) <= .1,
             "runner_up_ratio_ge_0.90": lambda r: r["diagnostics"].get("runner_up_support_ratio", -math.inf) >= .9,
             "runner_up_ratio_ge_1.00": lambda r: r["diagnostics"].get("runner_up_support_ratio", -math.inf) >= 1}
    for key, predicate in tests.items():
        selected = [r for r in rows if predicate(r)]
        result["descriptive_bins"][key] = {"count": len(selected),
            "valid_layout": sum(r["layout_valid"] for r in selected),
            "max_f1_accepted": sum(r["decision_valid"] and r["score"] >= thresholds["max_f1"] for r in selected)}
    result["operating_points"] = {}
    for op in ("max_f1", "recall_95"):
        accepted = [r for r in rows if r["decision_valid"] and r["score"] >= thresholds[op]]
        result["operating_points"][op] = {"threshold": thresholds[op], "accepted": len(accepted),
            "accepted_valid_layout": sum(r["layout_valid"] for r in accepted),
            "rejected_valid_layout": sum(r["layout_valid"] for r in rows) - sum(r["layout_valid"] for r in accepted),
            "accepted_diagnostics": {f: describe([r["diagnostics"].get(f) for r in accepted]) for f in FIELDS}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="root@connect.westb.seetacloud.com")
    parser.add_argument("--port", default="42993")
    parser.add_argument("--identity", default="/Users/yuqingzhang/.ssh/id_ed25519_rachel_benchmark_20260904")
    parser.add_argument("--remote-python", default="/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--reuse-extracted", action="store_true", help="Analyze saved projection without SSH")
    args = parser.parse_args()
    metrics_path = ROOT / "reports/rachel_all_models_review_20260914/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    keep_path = Path(metrics["keep_ids_source"]["file"])
    keep = set(json.loads(keep_path.read_text())["kept_positive_pair_ids"])
    selections = {m["id"]: next(s for s in m["selections"] if s["id"] == "fixed_epoch")
                  for m in metrics["models"] if m["id"] in ("s4", "s6", "s7")}
    sources = {}
    for key, selection in selections.items():
        source = selection["sources"]["real"]
        sources[key] = source.get("projection_of_remote_file") or str(
            Path(selection["checkpoint_path"]).parent.parent / "evaluation/fixed_epoch/real/pair_results.jsonl")
    args.output.mkdir(parents=True, exist_ok=True)
    extract_path = args.output / "extracted_rows.json"
    if args.reuse_extracted:
        extracted = json.loads(extract_path.read_text())
    else:
        remote = r'''
import hashlib,json,math,pathlib
sources = SOURCES
output = {}
for model,path in sources.items():
    raw=pathlib.Path(path).read_bytes()
    protocol_path=pathlib.Path(path).with_name('protocol.json')
    protocol_raw=protocol_path.read_bytes()
    protocol=json.loads(protocol_raw)
    rows=[]
    for line in raw.splitlines():
        if not line.strip(): continue
        r=json.loads(line); layout=r.get('layouts',{}).get('full_top2_mode',{})
        translation=layout.get('translation_rc')
        rows.append(dict(pair_id=r['pair_id'],case_id=r.get('case_id'),case_cluster=r.get('case_cluster'),
            label=bool(r['label']),review_status=r.get('review_status'),
            decision_valid=bool(r.get('decision_valid',False)),score=r['classification']['fused'],
            layout_present=bool(layout),layout_valid=bool(layout.get('valid',False)),
            finite_translation=isinstance(translation,list) and len(translation)==2 and
                all(isinstance(v,(int,float)) and math.isfinite(v) for v in translation),
            translation_rc=translation,diagnostics=layout.get('diagnostics',{}),
            translation_l2_px=layout.get('translation_l2_px'),
            overlap_small_fraction=layout.get('overlap_small_fraction'),
            candidate_details_present=r.get('candidate_details') is not None))
    output[model]=dict(source_path=path,source_sha256=hashlib.sha256(raw).hexdigest(),
        source_bytes=len(raw),protocol_path=str(protocol_path),
        protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(),
        protocol={k:protocol.get(k) for k in ('status','sample_count','decoder','decoder_config','model',
            'decoder_design_unchanged','thresholds_fitted','test_or_real_used_for_fit','script_sha256')},rows=rows)
print(json.dumps(output,allow_nan=False))
'''.replace("SOURCES", repr(sources))
        cmd = ["ssh", "-S", "none", "-i", args.identity, "-p", args.port, "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=15", args.host, shlex.quote(args.remote_python) + " -"]
        completed = subprocess.run(cmd, input=remote, text=True, capture_output=True, check=True, timeout=120)
        extracted = json.loads(completed.stdout)
        extract_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    output = {"schema": "negative-layout-diagnosis/1", "selection": "fixed_epoch20",
              "no_new_inference": True, "no_model_or_data_or_queue_mutation": True,
              "metrics_source": str(metrics_path), "metrics_sha256": sha(metrics_path),
              "keep_source": str(keep_path), "keep_sha256": sha(keep_path),
              "extract_sha256": sha(extract_path), "models": {}}
    ids = []
    for key, data in extracted.items():
        selection = selections[key]
        thresholds = selection["thresholds"]
        rows = data["rows"]
        assert len({r["pair_id"] for r in rows}) == len(rows), "duplicate pair IDs"
        assert data["protocol"]["status"] == "complete"
        assert len(rows) == data["protocol"]["sample_count"]
        assert data["protocol"]["model"]["checkpoint_sha256"] == selection["checkpoint_sha256"]
        assert data["protocol"]["model"]["epoch"] == selection["epoch"] == 20
        assert data["protocol"]["model"]["operating_points"]["thresholds"] == thresholds
        assert data["protocol"]["thresholds_fitted"] is False
        assert data["protocol"]["test_or_real_used_for_fit"] is False
        for r in rows:
            diag = r["diagnostics"]
            assert diag["valid"] == r["layout_valid"]
            assert math.isclose(diag["inlier_fraction"], diag["inlier_count"] / diag["candidate_count"])
            assert math.isclose(diag["runner_up_support_ratio"], diag["runner_up_support_weight"] / diag["support_weight"])
        ids.append({r["pair_id"] for r in rows})
        # Independently reconcile the remote records with the already-delivered
        # local UI projection. Array-heavy diagnostics are never taken from it.
        local = {r["pair_id"]: r for r in map(json.loads,
            Path(selection["sources"]["real"]["pair_scores_file"]).read_text().splitlines())}
        assert set(local) == ids[-1]
        for r in rows:
            old = local[r["pair_id"]]
            assert bool(old["label"]) == r["label"]
            assert old["classification"]["fused"] == r["score"]
            assert old["layouts"]["full_top2_mode"]["valid"] == r["layout_valid"]
            assert old["layouts"]["full_top2_mode"]["translation_rc"] == r["translation_rc"]
        positives = [r for r in rows if r["label"]]
        negatives = [r for r in rows if not r["label"]]
        kept = [r for r in positives if r["pair_id"] in keep]
        assert len(kept) == 295 and len(negatives) == 508 and len(positives) == 508
        groups = {"negative_all": negatives, "positive_all": positives, "positive_curated": kept}
        examples = {}
        for group_name, group in groups.items():
            accepted = [r for r in group if r["decision_valid"] and r["score"] >= thresholds["max_f1"]]
            by_short = lambda r: (r["diagnostics"].get("inlier_count", math.inf), -r["score"], r["pair_id"])
            examples[group_name] = {"lowest_inlier_count": sorted(group, key=by_short)[:8],
                "lowest_inlier_count_accepted_max_f1": sorted(accepted, key=by_short)[:8],
                "highest_score": sorted(group, key=lambda r: (-r["score"], r["pair_id"]))[:5]}
        output["models"][key] = {**{k: v for k, v in data.items() if k != "rows"},
            "checkpoint_sha256": selection["checkpoint_sha256"], "thresholds": thresholds,
            "local_projection_reconciled": True,
            "groups": {name: summarize(group, thresholds) for name, group in groups.items()}, "examples": examples}
    assert all(x == ids[0] for x in ids)
    a = {r["pair_id"]: r for r in extracted["s4"]["rows"]}
    b = {r["pair_id"]: r for r in extracted["s6"]["rows"]}
    output["s4_s6_geometry_comparison"] = {
        "same_pair_ids": True, "pair_count": len(a),
        "exact_translation_and_diagnostics_equal": sum(a[k]["translation_rc"] == b[k]["translation_rc"] and
            a[k]["diagnostics"] == b[k]["diagnostics"] for k in a)}
    output["limitations"] = [
        "inlier_fraction is inlier_count / candidate_count, not contour/arc-length coverage.",
        "Raw records omit candidate_indices/inlier_mask and point coordinates; spatial seam length and unique endpoint coverage cannot be reconstructed without additional saved evidence. No new inference performed.",
        "These are fixed epoch20 existing REAL predictions; descriptive post-hoc cuts are not proposed/fitted accept gates.",
        "Positive-curated295 is retrospective selection; raw positive508 is reported independently.",
        "A residual measures consistency within the selected mode, not correctness versus a negative-pair GT layout (none exists).",
        "S4/S6/S7 are not independent geometry observations when matchers are shared; no causal architecture claim."]
    source_lines = {
        "staging/pairwise_v0_2/models/translation_layout.py": [124, 137, 141, 151, 188, 211, 230, 243, 245],
        "experiments/rachel_n512_formal_30k/evaluate_score_design.py": [137, 150, 153, 160],
        "experiments/rachel_n512_formal_30k/evaluate_realism_checkpoint.py": [43],
        "reports/rachel_all_models_review_20260914/build_casebundle.py": [188, 191, 200]}
    output["local_source_evidence"] = {name: {"sha256": sha(ROOT / name), "line_anchors": lines}
        for name, lines in source_lines.items()}
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(destination), "models": {key: {g: {
        k: v for k, v in stats.items() if k in ("n", "layout_present", "layout_valid", "reason_counts")}
        for g, stats in value["groups"].items()} for key, value in output["models"].items()}}))


if __name__ == "__main__":
    main()
