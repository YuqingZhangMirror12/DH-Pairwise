"""Deterministic illustrative strata, not an unbiased performance estimate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS = ("s4", "s6", "s6_depth4", "s7")
SEED = "scorer-diagnosis-20260919"


def stable_key(row):
    return hashlib.sha256((SEED + row["pair_id"]).encode()).hexdigest()


def select(bundle, sim_rows):
    cases = bundle["cases"]
    result, groups, used = [], {}, set()

    def add(rows, group, n, dataset, score_sort=None):
        rows = [r for r in rows if (dataset, r["pair_id"]) not in used]
        rows = sorted(rows, key=score_sort or stable_key)
        chosen = rows[:n]
        if len(chosen) != n:
            raise ValueError(f"{group}: only {len(chosen)} eligible cases, expected {n}")
        groups[group] = {"eligible_count_before_selection": len(rows), "selected_count": n}
        for r in chosen:
            used.add((dataset, r["pair_id"]))
            entry = {"dataset": dataset, "pair_id": r["pair_id"], "stratum": group,
                     "label": bool(r["label"]), "name": r.get("name", r["pair_id"])}
            if "predictions" in r:
                entry["observed_scores"] = {m: r["predictions"][m]["score"] for m in MODELS}
                entry["negative_kind"] = r.get("negative_kind")
                entry["area_ratio"] = r.get("area_ratio")
                entry["layout_gt_available"] = bool(r.get("gt", {}).get("layout_gt_available"))
            result.append(entry)

    real = [r for r in cases if r["dataset"] == "dunhuang"]
    buckets = {k: [] for k in ("TP_good", "FN_good", "TP_bad", "FP", "TN")}
    for r in real:
        p = r["predictions"]["s6"]
        if not p["decision_valid"]:
            continue
        if r["label"]:
            err = p.get("layout_error_px")
            good = p["layout_valid"] and err is not None and err <= 20
            key = ("TP_good" if good else "TP_bad") if p["decision"] else ("FN_good" if good else None)
        else:
            key = "FP" if p["decision"] else "TN"
        if key:
            buckets[key].append(r)
    for key, rows in buckets.items():
        sort = (lambda r: (r.get("negative_kind") != "strict", stable_key(r))) if key in ("FP", "TN") else None
        add(rows, "real_" + key, 4, "real", sort)

    ood = [r for r in cases if r["dataset"] == "turufan"]
    def spread(r):
        values = [r["predictions"][m]["score"] for m in MODELS]
        return max(values) - min(values)
    add(ood, "ood_cross_model_disagreement", 4, "ood", lambda r: (-spread(r), stable_key(r)))
    remaining = sorted([r for r in ood if ("ood", r["pair_id"]) not in used],
                       key=lambda r: (r["predictions"]["s6"]["score"], stable_key(r)))
    for index in range(4):
        lo, hi = len(remaining) * index // 4, len(remaining) * (index + 1) // 4
        add(remaining[lo:hi], f"ood_s6_score_quartile_{index+1}", 2, "ood")

    sim = {"sim_positive": [], "sim_negative": []}
    for r in sim_rows:
        sim["sim_positive" if r["label"] else "sim_negative"].append(r)
    for group, rows in sim.items():
        add(rows, group, 4, "test")
    assert len(result) == 40 == len(used)
    return result, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "reports/rachel_all_models_review_20260914/cases.json")
    parser.add_argument("--sim-results", type=Path, default=ROOT / "reports/rachel_all_models_review_20260914/s5_s8_frozen_inputs/s6/test/pair_results.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.sim_results.read_text().splitlines() if line]
    selected, groups = select(json.loads(args.bundle.read_text()), rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        assert json.loads(args.output.read_text()) == selected, "refuse changing frozen selection"
    else:
        args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    receipt = {"status": "selection_frozen_before_attribution", "case_count": len(selected),
               "strata": groups, "seed": SEED,
               "dunhuang_strata_reference": "S6-depth2 epoch20, original SIMVAL max-F1 threshold, layout error<=20px",
               "negative_priority": "strict negatives before constructed, deterministic hash within type",
               "interpretation": "illustrative case-control sampling; not aggregate performance or causal proof",
               "no_gt_for_ood_layout": True,
               "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.bundle, args.sim_results)}}
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"count": len(selected), "strata": groups}, ensure_ascii=False))


if __name__ == "__main__":
    main()
