"""Frozen hard-VAL score readout: 3000 paired sources, never6000 iid sources.

Reads saved predictions only. No fitting, checkpoint selection or inference.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ("all_tokens", "matched_tokens", "matched_edges")
HEADS = tuple(f"{arm}_c{epoch}" for arm in ARMS for epoch in (8,16))
OPS = ("max_f1", "recall_99")
RECIPES = ("wave", "local", "seam_gaps", "partial_curve")


def mean(values):
    return statistics.mean(values) if values else None


def bce(logit, label):
    return max(logit,0.)-float(label)*logit+math.log1p(math.exp(-abs(logit)))


def good(row):
    error = row["raw_translation_l2_px"]
    return bool(row["label"] and row["raw_layout_valid"] and error is not None and error <= 20.)


def accepted(row, head, threshold):
    return bool(row["decision_valid"] and row["scores"][head]["probability"] >= threshold)


def metric(rows, head, thresholds):
    """Stable PairBCE and original validity-aware operational metrics."""
    n = len(rows)
    positives = [r for r in rows if r["label"]]
    negatives = [r for r in rows if not r["label"]]
    logits = [r["scores"][head]["logit"] for r in rows]
    losses = [bce(value,r["label"]) for value,r in zip(logits,rows)]
    scores = [r["scores"][head]["probability"] if r["decision_valid"] else -1. for r in rows]
    auroc = ap = None
    if positives and negatives:
        from staging.pairwise_v0_2.training.metrics import _ranking_metrics
        auroc,ap = _ranking_metrics(scores,[bool(r["label"]) for r in rows])
    correct = [r for r in positives if good(r)]
    result = dict(n=n,source_count=len({r["source_pair_id"] for r in rows}),
        positives=len(positives),negatives=len(negatives),
        changed_count=sum(r["changed_pair"] for r in rows),
        source_family_overlap_count=sum(r["source_family_overlap"] for r in rows),
        training_valid_count=sum(r["training_valid"] for r in rows),
        decision_valid_count=sum(r["decision_valid"] for r in rows),
        mean_pair_bce=mean(losses),mean_training_pair_bce_all_rows=mean([v*int(r["training_valid"]) for v,r in zip(losses,rows)]),
        mean_logit=mean(logits),mean_positive_logit=mean([r["scores"][head]["logit"] for r in positives]),
        mean_negative_logit=mean([r["scores"][head]["logit"] for r in negatives]),
        auroc=auroc,average_precision=ap,raw_correct_layout20=len(correct),operating_points={})
    for op in OPS:
        threshold=thresholds[op]
        tp=sum(accepted(r,head,threshold) for r in positives)
        fp=sum(accepted(r,head,threshold) for r in negatives)
        fn,tn=len(positives)-tp,len(negatives)-fp
        accepted_good=sum(accepted(r,head,threshold) for r in correct)
        both=bool(positives and negatives)
        result["operating_points"][op]=dict(threshold=threshold,tp=tp,fp=fp,fn=fn,tn=tn,
            accuracy=(tp+tn)/n if both else None,
            recall=tp/len(positives) if positives else None,
            false_positive_rate=fp/len(negatives) if negatives else None,
            precision=tp/(tp+fp) if both and tp+fp else (0. if both else None),
            f1=2*tp/(2*tp+fp+fn) if both else None,
            correct_layout_accepted=accepted_good,correct_layout_rejected=len(correct)-accepted_good,
            accepted_positive_bad_layout=tp-accepted_good,
            end_to_end_positive_recall=accepted_good/len(positives) if positives else None)
    return result


def paired(left,right,left_head,right_head,left_thresholds,right_thresholds):
    """Position-aligned SAME-source rows; clean→variant or C8→C16."""
    if len(left)!=len(right) or any(a["source_pair_id"]!=b["source_pair_id"] or a["label"]!=b["label"] for a,b in zip(left,right)):
        raise ValueError("paired sources/labels differ")
    result=dict(n=len(left),source_count=len({r["source_pair_id"] for r in left}),
        mean_logit_delta=mean([b["scores"][right_head]["logit"]-a["scores"][left_head]["logit"] for a,b in zip(left,right)]),
        mean_positive_logit_delta=mean([b["scores"][right_head]["logit"]-a["scores"][left_head]["logit"] for a,b in zip(left,right) if a["label"]]),
        mean_negative_logit_delta=mean([b["scores"][right_head]["logit"]-a["scores"][left_head]["logit"] for a,b in zip(left,right) if not a["label"]]),
        mean_pair_bce_delta=mean([bce(b["scores"][right_head]["logit"],b["label"])-bce(a["scores"][left_head]["logit"],a["label"]) for a,b in zip(left,right)]),
        operating_points={})
    def transitions(values):
        counts=Counter((bool(a),bool(b)) for a,b in values)
        return dict(rejected_to_rejected=counts[False,False],rejected_to_accepted=counts[False,True],
                    accepted_to_rejected=counts[True,False],accepted_to_accepted=counts[True,True])
    result["positive_raw_layout20_transitions"]=transitions([(good(a),good(b)) for a,b in zip(left,right) if a["label"]])
    for op in OPS:
        states=[(a,b,accepted(a,left_head,left_thresholds[op]),accepted(b,right_head,right_thresholds[op])) for a,b in zip(left,right)]
        result["operating_points"][op]=dict(left_threshold=left_thresholds[op],right_threshold=right_thresholds[op],
            positive_acceptance=transitions([(x,y) for a,b,x,y in states if a["label"]]),
            negative_acceptance=transitions([(x,y) for a,b,x,y in states if not a["label"]]),
            positive_correct_layout_accepted=transitions([(x and good(a),y and good(b)) for a,b,x,y in states if a["label"]]))
    return result


def validate(rows,thresholds,expected_sources):
    if set(thresholds)!=set(HEADS) or any(not set(OPS)<=set(t) or any(not math.isfinite(v) or not 0<=v<=1 for v in t.values()) for t in thresholds.values()):
        raise ValueError("six fixed heads with original SIMVAL maxF1/R99 thresholds required")
    if len(rows)!=2*expected_sources or len({r["pair_id"] for r in rows})!=len(rows):
        raise ValueError("requires exactly two unique views per expected source")
    by_source={}
    for row in rows:
        if row["label"] not in (0,1) or row["recipe"] not in ("clean",*RECIPES):
            raise ValueError("invalid label or recipe")
        for key in ("changed_pair","source_family_overlap","training_valid","decision_valid","raw_layout_valid"):
            if type(row.get(key)) is not bool: raise ValueError("explicit bool required: "+key)
        if row["raw_translation_l2_px"] is not None and (not math.isfinite(row["raw_translation_l2_px"]) or row["raw_translation_l2_px"]<0):
            raise ValueError("invalid raw layout error")
        if row["label"] and row["raw_layout_valid"] and row["raw_translation_l2_px"] is None:
            raise ValueError("valid positive layout needs actual GT error")
        if "raw_layout20_correct" in row and row["raw_layout20_correct"]!=good(row):
            raise ValueError("stored layout20 flag disagrees with error/validity")
        if set(row["scores"])!=set(HEADS): raise ValueError("six complete scores required")
        for score in row["scores"].values():
            if not all(math.isfinite(score[k]) for k in ("logit","probability")) or not 0<=score["probability"]<=1:
                raise ValueError("finite score required")
            expected=math.exp(-max(-score["logit"],0.))/(1.+math.exp(-abs(score["logit"])))
            if abs(score["probability"]-expected)>1e-6: raise ValueError("logit/probability disagree")
        by_source.setdefault(row["source_pair_id"],[]).append(row)
    if len(by_source)!=expected_sources: raise ValueError("wrong unique source-pair count")
    if expected_sources==3000:
        counts=Counter((r["recipe"],bool(r["label"])) for r in rows)
        if any(counts["clean",label]!=1500 or any(counts[recipe,label]!=375 for recipe in RECIPES) for label in (False,True)):
            raise ValueError("full hardVAL must preserve clean1500+/1500- and375+/375- per requested recipe")
    clean,variants={},{}
    for source,views in by_source.items():
        if len(views)!=2 or sum(r["recipe"]=="clean" for r in views)!=1:
            raise ValueError("one clean and one requested variant required")
        a=next(r for r in views if r["recipe"]=="clean")
        b=next(r for r in views if r["recipe"]!="clean")
        if a["changed_pair"] or a["label"]!=b["label"] or a["source_family_overlap"]!=b["source_family_overlap"]:
            raise ValueError("source-paired metadata differs")
        if a.get("assigned_recipe",b["recipe"])!=b["recipe"] or b.get("assigned_recipe",b["recipe"])!=b["recipe"]:
            raise ValueError("clean assigned recipe differs from variant")
        clean[source],variants[source]=a,b
    return clean,variants


def summarize(rows,thresholds,*,expected_sources=3000):
    rows=list(rows)
    clean,variants=validate(rows,thresholds,expected_sources)
    source_ids=sorted(clean)
    clean_rows=[clean[i] for i in source_ids]; requested=[variants[i] for i in source_ids]
    changed=[r for r in requested if r["changed_pair"]]
    fallback=[r for r in requested if not r["changed_pair"]]
    result=dict(schema="frozen-hardval-six-scorer-summary/1",status="complete",view_count=len(rows),
        source_count=len(source_ids),not_independent_6000_sources=True,thresholds=thresholds,
        thresholds_fitted=False,checkpoint_selection_performed=False,heads={},c8_to_c16={},
        sources=[dict(source_pair_id=i,clean_pair_id=clean[i]["pair_id"],variant_pair_id=variants[i]["pair_id"],
            label=bool(clean[i]["label"]),requested_recipe=variants[i]["recipe"],
            changed_pair=variants[i]["changed_pair"],fallback_reason=variants[i].get("fallback_reason"),
            source_family_overlap=clean[i]["source_family_overlap"]) for i in source_ids])
    populations={"clean":clean_rows,"full_requested":requested,"actually_changed":changed,"unchanged_fallback":fallback}
    for head in HEADS:
        point=thresholds[head]
        item=dict(populations={name:metric(values,head,point) for name,values in populations.items()},by_recipe={},source_family={})
        for recipe in RECIPES:
            views=[r for r in requested if r["recipe"]==recipe]
            changed_views=[r for r in views if r["changed_pair"]]
            untouched=[r for r in views if not r["changed_pair"]]
            references=[clean[r["source_pair_id"]] for r in views]
            item["by_recipe"][recipe]=dict(clean=metric(references,head,point),full_requested=metric(views,head,point),
                actually_changed=metric(changed_views,head,point),unchanged_fallback=metric(untouched,head,point),
                fallback_reasons=dict(Counter(r.get("fallback_reason") or "unspecified" for r in untouched)),
                paired_full_requested=paired(references,views,head,head,point,point),
                paired_actually_changed=paired([clean[r["source_pair_id"]] for r in changed_views],changed_views,head,head,point,point))
        for overlap in (False,True):
            item["source_family"]["overlap" if overlap else "disjoint"]={name:metric([r for r in values if r["source_family_overlap"]==overlap],head,point) for name,values in populations.items()}
        result["heads"][head]=item
    for arm in ARMS:
        left,right=f"{arm}_c8",f"{arm}_c16"
        scopes=dict(populations)
        for recipe in RECIPES:
            scopes[recipe+"_requested"]=[r for r in requested if r["recipe"]==recipe]
            scopes[recipe+"_actually_changed"]=[r for r in changed if r["recipe"]==recipe]
        result["c8_to_c16"][arm]={name:paired(values,values,left,right,thresholds[left],thresholds[right]) for name,values in scopes.items()}
    result["definitions"]=dict(pair_bce="stable BCEWithLogits on saved deployed logits; mean_training_pair_bce_all_rows=mean(BCE*training_valid) over all rows",
        decision="decision_valid AND probability>=own original cleanSIMVAL frozen threshold; R99 is calibration target, not promised hardVAL recall",
        ranking="invalid decisions rank at -1; original grouped-tie AUROC/AP implementation; unavailable for one/zero label groups",
        layout20="positive and raw_layout_valid and actual decoded translation_l2<=20px, never soft-estimator displacement",
        pairing="one requested derivative and one clean reference from each same source; actually_changed excludes coupled fallback",
        limits="diagnostic derivative of previously used SIMVAL, not independent heldout sources; no REAL/OOD fitting/selection or new threshold calibration")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((args.root/"protocol.json").read_text())
    if protocol.get("status")!="complete": raise ValueError("complete saved inference required")
    rows=[json.loads(line) for line in (args.root/"cases.jsonl").read_text().splitlines()]
    result=summarize(rows,protocol["thresholds"])
    result["source"]=dict(root=str(args.root.resolve()),protocol_sha256=hashlib.sha256((args.root/"protocol.json").read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256((args.root/"cases.jsonl").read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream: json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(status="complete",view_count=result["view_count"],source_count=result["source_count"],output=str(args.output))))


if __name__=="__main__":
    main()
