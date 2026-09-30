"""Fixed S7 matrix-cache dustbin diagnostic; not a trained Scorer experiment.

Four predeclared decoder variants on the existing selected40 heatmap cache.
No hyperparameter fit, new data selection, model update or pair-score update.
Label/GT fields are attached only after all target-blind predictions are made.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from staging.pairwise_v0_2.models.translation_layout import (
    TranslationLayoutConfig, estimate_translation_layout,
)

CONFIG = TranslationLayoutConfig(correspondence_mode="topk_union", top_k=2)
ARMS = ("baseline", "edge_beats_dustbin", "node_matchability_gt_half", "soft_matchability")


def predictions(z):
    q = np.asarray(z["sinkhorn_assignment"], dtype=np.float64)
    va, vb = z["valid_a"], z["valid_b"]
    q = np.where(va[:, None] & vb[None, :], q, 0.0)
    ua, ub = z["matcher_unmatched_a"].astype(np.float64), z["matcher_unmatched_b"].astype(np.float64)
    ra, rb = q.sum(1), q.sum(0)
    ma = np.divide(ra, ra + ua, out=np.zeros_like(ra), where=(ra + ua) > 0)
    mb = np.divide(rb, rb + ub, out=np.zeros_like(rb), where=(rb + ub) > 0)
    variants = {
        "baseline": q,
        "edge_beats_dustbin": q * ((q > ua[:, None]) & (q > ub[None, :])),
        "node_matchability_gt_half": q * ((ma[:, None] > .5) & (mb[None, :] > .5)),
        "soft_matchability": q * np.sqrt(ma[:, None] * mb[None, :]),
    }
    out = {}
    for arm, weights in variants.items():
        pose = estimate_translation_layout(z["points_rc_a"], z["points_rc_b"], weights,
                                           va, vb, config=CONFIG)
        out[arm] = dict(valid=bool(pose.valid), reason=pose.reason,
            translation_rc=pose.t_a_to_b_rc.tolist() if pose.valid else None,
            candidate_count=int(pose.candidate_count), inlier_count=int(pose.inlier_count))
    stats = dict(max_q=float(q.max()), unmatched_median=float(np.median(np.r_[ua[va], ub[vb]])),
        matchability_median=float(np.median(np.r_[ma[va], mb[vb]])),
        marginal_max_residual=float(max(abs(ra[va] + ua[va] - 1).max(), abs(rb[vb] + ub[vb] - 1).max())))
    return out, stats


def summarize(rows):
    groups = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        positives = [r for r in subset if r["label"] and r["layout_gt_available"]]
        negatives = [r for r in subset if not r["label"]]
        arms = {}
        for arm in ARMS:
            arms[arm] = dict(
                valid_layouts=sum(r["predictions"][arm]["valid"] for r in subset),
                positive_gt_count=len(positives),
                positive_layout20=sum(r["predictions"][arm]["correct20"] for r in positives),
                negative_count=len(negatives),
                negative_with_layout=sum(r["predictions"][arm]["valid"] for r in negatives),
                lost_baseline_correct20=sum(r["predictions"]["baseline"]["correct20"] and not r["predictions"][arm]["correct20"] for r in positives),
                rescued_baseline_wrong20=sum(not r["predictions"]["baseline"]["correct20"] and r["predictions"][arm]["correct20"] for r in positives),
            )
        groups[dataset] = dict(count=len(subset), arms=arms,
            stats_medians={key:float(np.median([r["stats"][key] for r in subset])) for key in subset[0]["stats"]})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    protocol = json.loads((args.cache / "protocol.json").read_text())
    assert protocol["status"] == "complete"
    cases = json.loads((args.cache / "cases.json").read_text())
    assert len(cases) == 40 and len({c["pair_id"] for c in cases}) == 40
    rows = []
    for case in cases:
        path = args.cache / case["arrays_path"]
        with np.load(path, allow_pickle=False) as arrays:
            pred, stats = predictions(arrays)
        baseline = pred["baseline"]
        old = case["layout"]
        assert baseline["valid"] == old["valid"]
        pose_delta = (float(np.linalg.norm(np.array(baseline["translation_rc"]) - old["t_a_to_b_rc"]))
                      if baseline["valid"] else 0.0)
        assert pose_delta < 1e-3, (case["pair_id"], pose_delta)
        # Attach labels/GT only after target-blind variants above are fixed.
        for value in pred.values():
            error = (float(np.linalg.norm(np.asarray(value["translation_rc"]) - case["target_translation_rc"]))
                     if value["valid"] and case["layout_gt_available"] and case["label"] else None)
            value.update(error_px=error, correct20=bool(error is not None and error <= 20))
        rows.append(dict(pair_id=case["pair_id"], dataset=case["dataset"], label=bool(case["label"]),
            layout_gt_available=bool(case["layout_gt_available"]), predictions=pred, stats=stats,
            baseline_reproduction_error_px=pose_delta))
    result = dict(schema="s7-selected40-dustbin-diagnostic/1", status="complete",
        limitations="Selected diagnostic cases, not population estimates. No Scorer retrained/recomputed; valid layout is NOT an adjacency prediction. OOD has no layout GT. No threshold tuning or variant selection.",
        model=protocol["model"], decoder=asdict(CONFIG),
        variants=dict(baseline="Original Q", edge_beats_dustbin="Qij>uAi AND Qij>uBj, otherwise zero",
            node_matchability_gt_half="mAi>0.5 AND mBj>0.5; m=sumQ/(sumQ+u)",
            soft_matchability="Qij*sqrt(mAi*mBj), no absolute rejection"),
        source_cache=str(args.cache), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sample_count=len(rows), groups=summarize(rows), rows=rows, elapsed_s=time.monotonic()-start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:result[k] for k in ("status", "sample_count", "groups", "elapsed_s")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
