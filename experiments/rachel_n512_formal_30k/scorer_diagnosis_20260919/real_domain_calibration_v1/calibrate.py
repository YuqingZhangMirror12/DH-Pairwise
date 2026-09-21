"""Cross-fit thresholds only. Report held-out predictions, never refit scores."""
import argparse
import json
import math
from pathlib import Path
import statistics

from .prepare import ARMS, K, read, save


def metrics(labels, accepted, scores=None, good=None):
    p=sum(labels);n=len(labels)-p
    tp=sum(y and a for y,a in zip(labels,accepted)); fp=sum(not y and a for y,a in zip(labels,accepted))
    result=dict(n=len(labels),positive=p,negative=n,tp=tp,fp=fp,fn=p-tp,tn=n-fp,
        accuracy=(tp+n-fp)/len(labels),precision=tp/(tp+fp) if tp+fp else 0.,
        recall=tp/p if p else None,f1=2*tp/(p+tp+fp) if p+tp+fp else 0.,
        false_positive_rate=fp/n if n else None)
    if good is not None:
        count=sum(y and bool(g) for y,g in zip(labels,good))
        kept=sum(y and bool(g) and a for y,g,a in zip(labels,good,accepted))
        result.update(layout_correct_total=count,layout_correct_accepted=kept,
            layout_correct_rejected=count-kept,accepted_positive_wrong_layout=tp-kept,
            end_to_end_positive_recall=kept/p if p else None)
    if scores is not None and p and n:
        order=sorted(zip(scores,labels))
        i=0;rank_sum=0.
        while i<len(order):
            j=i+1
            while j<len(order) and order[j][0]==order[i][0]:j+=1
            rank_sum+=sum(y for _,y in order[i:j])*(i+1+j)/2
            i=j
        result["auroc"]=(rank_sum-p*(p+1)/2)/(p*n)
    return result


def select_threshold(labels,scores,valid,policy):
    """Exact empirical operating point, computed from calibration rows only."""
    if not any(labels) or all(labels):
        raise ValueError("binary calibration requires both labels")
    thresholds=sorted({0.,1.,*(s for s,v in zip(scores,valid) if v)},reverse=True)
    best=None
    for threshold in thresholds:
        accepted=[v and s>=threshold for s,v in zip(scores,valid)]
        m=metrics(labels,accepted)
        if policy=="recall_95":
            if m["recall"]>=.95:
                return dict(threshold=threshold,calibration_metrics=m,target_met=True)
        elif policy=="max_f1":
            key=(m["f1"],m["precision"],threshold)
            if best is None or key>best[0]:best=(key,threshold,m)
        else:raise ValueError("unknown predeclared threshold policy")
    if policy=="max_f1":return dict(threshold=best[1],calibration_metrics=best[2])
    return dict(threshold=0.,calibration_metrics=metrics(labels,valid),target_met=False)


def evaluate(root):
    root=Path(root);freezes=read(root/"model_freezes.json")
    all_results={}
    for split in ("real","ood"):
        meta=read(root/split/"manifest.json");population=meta["pairs"]
        labels=[r["label"] for r in population]
        fold_index=[r["fold"] for r in population]
        by_model={}
        for arm in ARMS:
            pred=read(root/"predictions"/arm/(split+".json"))
            assert pred["status"]=="complete" and pred["checkpoint_sha256"]==freezes[arm]["checkpoint_sha256"]
            assert [r["pair_id"] for r in pred["rows"]]==[r["pair_id"] for r in population]
            scores=[r["score"] for r in pred["rows"]];valid=[r["decision_valid"] for r in pred["rows"]]
            good=[r["layout_good_20"] for r in pred["rows"]] if split=="real" else None
            baseline={}
            for name in ("max_f1","recall_99"):
                threshold=freezes[arm]["operating_points"]["thresholds"][name]
                baseline["simval_"+name]=dict(threshold=threshold,**metrics(labels,[v and s>=threshold for s,v in zip(scores,valid)],scores,good))
            policies={};oof={r["pair_id"]:dict(pair_id=r["pair_id"],fold=r["fold"],label=r["label"],score=scores[i],policies={}) for i,r in enumerate(population)}
            for policy in ("max_f1","recall_95"):
                accepted=[None]*len(population);folds=[]
                for k in range(K):
                    cal=[i for i,f in enumerate(fold_index) if f!=k];test=[i for i,f in enumerate(fold_index) if f==k]
                    # Source overlap is checked again independently of preparation.
                    def src(indices):return {meta["fragment_source_group"][population[i][s]] for i in indices for s in ("fragment_a_id","fragment_b_id")}
                    assert not src(cal)&src(test)
                    selection=select_threshold([labels[i] for i in cal],[scores[i] for i in cal],[valid[i] for i in cal],policy)
                    t=selection["threshold"]
                    for i in test:
                        assert accepted[i] is None
                        accepted[i]=bool(valid[i] and scores[i]>=t)
                        oof[population[i]["pair_id"]]["policies"][policy]=dict(threshold=t,accepted=accepted[i])
                    folds.append(dict(fold=k,calibration_count=len(cal),test_count=len(test),**selection,
                        held_out_metrics=metrics([labels[i] for i in test],[accepted[i] for i in test],
                            [scores[i] for i in test],[good[i] for i in test] if good is not None else None)))
                assert all(a is not None for a in accepted)
                thresholds=[f["threshold"] for f in folds]
                policies[policy]=dict(pooled_out_of_fold=metrics(labels,accepted,scores,good),folds=folds,
                    threshold_median=statistics.median(thresholds),threshold_min=min(thresholds),threshold_max=max(thresholds),
                    all_data_refit_for_future_only=select_threshold(labels,scores,valid,policy)["threshold"])
            by_model[arm]=dict(baseline_on_same_new_negative_population=baseline,cv=policies,
                model_weights_updated=False,checkpoint_sha256=pred["checkpoint_sha256"])
            save(root/"oof"/split/(arm+".json"),list(oof.values()))
            print(json.dumps(dict(split=split,arm=arm,sim_f1=baseline["simval_max_f1"]["f1"],
                cv_f1=policies["max_f1"]["pooled_out_of_fold"]["f1"],
                cv_recall=policies["max_f1"]["pooled_out_of_fold"]["recall"],
                median_threshold=policies["max_f1"]["threshold_median"],
                r95=policies["recall_95"]["pooled_out_of_fold"]["recall"])),flush=True)
        all_results[split]=dict(positive=sum(labels),negative=len(labels)-sum(labels),fold_checks=meta["fold_checks"],
            source_group_count=len(meta["source_folds"]),models=by_model)
    save(root/"results.json",dict(status="complete",threshold_cv_only=True,five_fold_once=True,
        old_real_benchmark_negative_pool_replaced=True,constructed_negative_assumption=True,
        prior_real_design_exposure=True,models_selected_on_test=False,
        method_reference="https://scikit-learn.org/stable/modules/classification_threshold.html",
        results=all_results))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True)
    a=p.parse_args();evaluate(a.root)
