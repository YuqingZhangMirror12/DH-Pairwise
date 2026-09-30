"""Read-only diagnosis of D's existing Pair and candidate-quality outputs.

No inference, training, threshold fitting, or source-file mutation. Run with the
experiment server's Python, optionally streamed over SSH stdin. JSON goes to
stdout so the caller can retain a local, private analytical record.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score


def read(p):
    return json.loads(Path(p).read_text())


def describe(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    if not len(a):
        return dict(n=0)
    return dict(n=len(a), mean=float(a.mean()),
                p10=float(np.quantile(a, .1)), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)))


def auc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None


def metric(rows, score, target, threshold, valid="decision_valid"):
    y = np.array([r[target] for r in rows], bool)
    s = np.array([r[score] for r in rows], float)
    a = np.array([r[valid] for r in rows], bool) & (s >= threshold)
    tp, fp = int((y & a).sum()), int((~y & a).sum())
    p, n = int(y.sum()), int((~y).sum())
    return dict(count=len(rows), positive=p, negative=n, threshold=threshold,
        tp=tp, fp=fp, fn=p-tp, tn=n-fp,
        accuracy=(tp+n-fp)/len(rows) if p and n else None,
        recall=tp/p if p else None, precision=(tp/(tp+fp) if tp+fp else 0.) if p and n else None,
        f1=(2*tp/(p+tp+fp) if p+tp+fp else 0.) if p and n else None, auroc=auc(y.tolist(), s),
        auprc=float(average_precision_score(y,s)) if p and n else None)


def compact(row):
    d = row["candidate_details"]
    l = row["layouts"]["full_top2_mode"]
    e = l["translation_l2_px"]
    indices, inliers = d["candidate_indices"], d["candidate_inliers"]
    weights = [w for w, active in zip(d["candidate_weights"], inliers) if active]
    return dict(pair_id=row["pair_id"], case_id=row.get("case_id"),
        fragment_a=row["fragment_a"], fragment_b=row["fragment_b"], label=bool(row["label"]),
        review_status=row.get("review_status"), decision_valid=row["decision_valid"],
        pair_score=row["classification"]["fused"], pair_logit=d["raw_head_logit"],
        quality_score=d["candidate_quality_probability"], quality_logit=d["candidate_quality_logit"],
        layout_valid=l["valid"], layout_error_px=e,
        quality_known=bool(l["valid"] and (not row["label"] or e is not None)),
        quality_target=bool(row["label"] and l["valid"] and e is not None and e <= 20),
        selected_a=d["selected_token_count_a"], selected_b=d["selected_token_count_b"],
        support_min=min(d["selected_token_count_a"], d["selected_token_count_b"]),
        inlier_edges=d["inlier_edge_count"], used_fallback=d["used_fallback"],
        inlier_q_mass=float(sum(weights)),
        residual_px=l["diagnostics"].get("residual_px"),
        runner_up_support_ratio=l["diagnostics"].get("runner_up_support_ratio"),
        overlap_small_fraction=l.get("overlap_small_fraction"), area_ratio=row.get("area_ratio"))


def group(rows, pair_threshold):
    return dict(count=len(rows),
        pair=describe([r["pair_score"] for r in rows]),
        quality=describe([r["quality_score"] for r in rows]),
        pair_accepted=sum(r["decision_valid"] and r["pair_score"] >= pair_threshold for r in rows),
        quality_ge_03=sum(r["layout_valid"] and r["quality_score"] >= .3 for r in rows),
        quality_ge_05=sum(r["layout_valid"] and r["quality_score"] >= .5 for r in rows),
        support_min=describe([r["support_min"] for r in rows]),
        residual=describe([r["residual_px"] for r in rows]),
        overlap=describe([r["overlap_small_fraction"] for r in rows]))


def decomposition(rows, accepted, quality_threshold):
    buckets = Counter()
    for r in rows:
        if not r["label"]:
            continue
        layout = "unknown_GT" if not r["quality_known"] else ("correct" if r["quality_target"] else "wrong")
        p = "pair_accept" if accepted(r) else "pair_reject"
        q = "quality_high" if r["layout_valid"] and r["quality_score"] >= quality_threshold else "quality_low"
        buckets[layout+"/"+p+"/"+q] += 1
    return dict(sorted(buckets.items()))


def split_analysis(rows, pair_threshold):
    positive = [r for r in rows if r["label"]]
    known = [r for r in rows if r["quality_known"]]
    positive_known = [r for r in known if r["label"]]
    good = [r for r in positive if r["quality_target"]]
    bad = [r for r in positive_known if not r["quality_target"]]
    negative = [r for r in rows if not r["label"]]
    out = dict(count=len(rows), positive=len(positive), negative=len(negative),
        missing_positive_layout_gt=sum(r["layout_error_px"] is None for r in positive),
        invalid_layout=sum(not r["layout_valid"] for r in rows),
        good_layout=len(good), wrong_known_positive_layout=len(bad),
        pair_metrics={str(t):metric(rows,"pair_score","label",t) for t in (pair_threshold,.3,.5)},
        quality_metrics={str(t):metric(known,"quality_score","quality_target",t,"layout_valid") for t in (.3,.5)} if known else {},
        quality_ranking_within_positive=dict(count=len(positive_known),correct=len(good),wrong=len(bad),
            quality_auc=auc([r["quality_target"] for r in positive_known],[r["quality_score"] for r in positive_known]),
            pair_score_auc_same_target=auc([r["quality_target"] for r in positive_known],[r["pair_score"] for r in positive_known])),
        score_correlation=dict(pearson_probability=float(np.corrcoef([r["pair_score"] for r in rows],[r["quality_score"] for r in rows])[0,1]),
            pearson_logit=float(np.corrcoef([r["pair_logit"] for r in rows],[r["quality_logit"] for r in rows])[0,1]),
            spearman=float(spearmanr([r["pair_score"] for r in rows],[r["quality_score"] for r in rows]).statistic)),
        groups=dict(correct_layout_positive=group(good,pair_threshold),wrong_layout_positive=group(bad,pair_threshold),
            negative=group(negative,pair_threshold)), decompositions={}, counterfactuals={})
    for t in (pair_threshold,.3,.5):
        accept=lambda r:r["decision_valid"] and r["pair_score"] >= t
        for q in (.3,.5):
            out["decompositions"][f"pair_{t}/quality_{q}"] = decomposition(rows,accept,q)
            new = [r for r in rows if not accept(r) and r["layout_valid"] and r["quality_score"] >= q]
            out["counterfactuals"][f"OR_pair_{t}/quality_{q}"]=dict(
                diagnostic_only=True, additional_good_layout=sum(r["quality_target"] for r in new),
                additional_wrong_positive=sum(r["label"] and r["quality_known"] and not r["quality_target"] for r in new),
                additional_unknown_positive=sum(r["label"] and not r["quality_known"] for r in new),
                additional_negative=sum(not r["label"] for r in new))
    if good:
        out["correct_layout_subgroups"] = {
            "low_support_le32":group([r for r in good if r["support_min"]<=32],pair_threshold),
            "support_gt32":group([r for r in good if r["support_min"]>32],pair_threshold),
            "both_reject_03":group([r for r in good if r["pair_score"]<.3 and r["quality_score"]<.3],pair_threshold),
        }
    out["support_stratified_positive_quality"]={}
    for lo,hi in ((0,16),(17,32),(33,64),(65,512)):
        subset=[r for r in positive_known if lo<=r["support_min"]<=hi]
        out["support_stratified_positive_quality"][f"{lo}_{hi}"]=dict(count=len(subset),
            correct=sum(r["quality_target"] for r in subset),
            quality_auc=auc([r["quality_target"] for r in subset],[r["quality_score"] for r in subset]),
            pair_auc_on_quality=auc([r["quality_target"] for r in subset],[r["pair_score"] for r in subset]))
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    args=parser.parse_args()
    root=Path(args.root)
    run=root/"local_evidence_v2_20260921"
    training=run/"training/joint_D_h4"
    freeze=read(training/"freeze.json")
    threshold=freeze["operating_points"]["thresholds"]["max_f1"]
    result=dict(schema="joint-D-dual-head-diagnosis/1",
        scope="fixed C16 saved predictions; no retraining or threshold optimization",
        candidate_threshold_note="0.3 and0.5 are diagnostic cuts, not calibrated quality thresholds",
        checkpoint_sha256=freeze["checkpoint_sha256"],pair_frozen_threshold=threshold,
        sources=[],splits={})
    selected={}
    for split in ("test","real","ood"):
        path=run/"evaluation/joint_D_h4"/split/"pair_results.jsonl"
        allrows=[compact(json.loads(line)) for line in path.open() if line.strip()]
        assert len({r["pair_id"] for r in allrows})==len(allrows)
        rows=[r for r in allrows if split!="real" or not r["label"] or r["review_status"]=="keep"]
        assert all(np.isfinite(r[k]) for r in rows for k in ("pair_score","quality_score"))
        selected[split]=rows
        result["sources"].append(dict(path=str(path),bytes=path.stat().st_size,original_count=len(allrows),selected_count=len(rows)))
        result["splits"][split]=split_analysis(rows,threshold)
    assert len(selected["real"])==803 and sum(r["label"] for r in selected["real"])==295
    assert len(selected["test"])==3000 and len(selected["ood"])==301
    # Recreate the exact executed TRAIN-only sampling order using the frozen
    # source file; no model import or GPU work is needed.
    source=run/"source/experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/local_evidence_v2/data.py"
    spec=importlib.util.spec_from_file_location("d_training_evidence",source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cache_root=Path(freeze["identity"]["train_cache"]["root"])
    class MetadataCache:
        records=read(cache_root/"pairs.json")
        def __len__(self):return len(self.records)
    evidence=module.TrainingEvidence(freeze["identity"]["data_intervention"]["path"],MetadataCache())
    sampling=[]
    for epoch in range(1,17):
        ids,ledger=evidence.order(epoch)
        assert ledger==read(training/("sampling_epoch_%03d.json"%epoch))
        differing=evidence.labels[ids] & ~evidence.correct[ids] & evidence.known[ids]
        sampling.append(dict(epoch=epoch,total=len(ids),positive=int(evidence.labels[ids].sum()),
            pair_positive_candidate_negative=int(differing.sum()),label_agreement=float(1-differing.mean())))
    result["training_supervision"]=dict(original_pairs=len(evidence.rows),
        original_positive=int(evidence.labels.sum()),original_good_candidates=int((evidence.labels&evidence.correct).sum()),
        original_positive_bad_candidate=int((evidence.labels&~evidence.correct&evidence.known).sum()),
        epochs=sampling,label_agreement=1-sum(r["pair_positive_candidate_negative"] for r in sampling)/384000)
    history=[]
    for epoch in range(1,17):
        segments=[read(training/("segment_%03d.json"%n)) for n in range((epoch-1)*4+1,epoch*4+1)]
        val=read(training/("validation_%03d.json"%epoch))
        history.append(dict(epoch=epoch,train_pair_bce=float(np.mean([s["pair_bce"] for s in segments])),
            train_candidate_bce=float(np.mean([s["candidate_bce"] for s in segments])),val_pair_bce=val["mean_loss"],
            val_pair_auc=val["metrics_at_0_5"]["auroc"]))
    result["training_history"]=history
    rows=selected["real"]
    selectors={
        "correct_layout_both_low_03":lambda r:r["quality_target"] and r["pair_score"]<.3 and r["quality_score"]<.3,
        "correct_layout_pair_low_quality_high":lambda r:r["quality_target"] and r["pair_score"]<.3 and r["quality_score"]>=.3,
        "wrong_layout_both_high_03":lambda r:r["label"] and r["quality_known"] and not r["quality_target"] and r["pair_score"]>=.3 and r["quality_score"]>=.3,
        "wrong_layout_pair_high_quality_low":lambda r:r["label"] and r["quality_known"] and not r["quality_target"] and r["pair_score"]>=.3 and r["quality_score"]<.3,
        "negative_pair_high_quality_low":lambda r:not r["label"] and r["pair_score"]>=.3 and r["quality_score"]<.3,
    }
    result["case_examples"]={k:sorted([r for r in rows if pick(r)],key=lambda r:(r["pair_score"],r["pair_id"]))[:5] for k,pick in selectors.items()}
    # Existing fold thresholds are applied as frozen diagnostics; never fit a
    # new quality threshold on these cases.
    cvroot=root/"bounded_real_calibration_v2_20260921"
    cal=read(cvroot/"results.json")
    result["existing_calibration_D"]=cal["results"]["real"]["models"]["joint_D_h4"]
    oof=read(cvroot/"oof/real/joint_D_h4/bounded_max_f1.json")
    lookup={r["pair_id"]:r for r in oof}
    positive=[r for r in rows if r["label"]]
    assert len(positive)==295 and all(r["pair_id"] in lookup for r in positive)
    assert all(abs(r["pair_score"]-lookup[r["pair_id"]]["score"])<1e-7 for r in positive)
    result["existing_oof_positive_decomposition"]={str(q):decomposition(positive,
        lambda r:lookup[r["pair_id"]]["accepted"],q) for q in (.3,.5)}
    result["oof_rejected_correct_layout"]=group([r for r in positive if r["quality_target"] and not lookup[r["pair_id"]]["accepted"]],threshold)
    result["oof_rejected_correct_low_support_count"]=sum(r["quality_target"] and not lookup[r["pair_id"]]["accepted"] and r["support_min"]<=32 for r in positive)
    result["positive_diagnostic_rows"]=[dict(r,oof_accepted=lookup[r["pair_id"]]["accepted"],
        oof_threshold=lookup[r["pair_id"]]["threshold"]) for r in positive]
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":
    main()
