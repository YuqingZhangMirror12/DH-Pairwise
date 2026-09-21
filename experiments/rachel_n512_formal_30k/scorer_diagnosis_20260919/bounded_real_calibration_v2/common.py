"""Frozen raw scores; source-disjoint CV with a prespecified threshold grid."""
import json
import hashlib
from pathlib import Path
import statistics

GRID = tuple(i / 100 for i in range(20, 81))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def metrics(labels, decisions, scores, good=None):
    n = len(labels); p = sum(labels); neg = n - p
    tp = sum(bool(y and a) for y, a in zip(labels, decisions))
    fp = sum(bool(not y and a) for y, a in zip(labels, decisions))
    out = dict(n=n, positive=p, negative=neg, tp=tp, fp=fp, fn=p-tp, tn=neg-fp,
        accuracy=(tp+neg-fp)/n, precision=tp/(tp+fp) if tp+fp else 0.,
        recall=tp/p if p else None, f1=2*tp/(p+tp+fp) if p+tp+fp else 0.,
        false_positive_rate=fp/neg if neg else None)
    if p and neg:
        order = sorted(zip(scores, labels)); i=0; rank_sum=0.
        while i<n:
            j=i+1
            while j<n and order[j][0] == order[i][0]: j+=1
            rank_sum += sum(y for _, y in order[i:j]) * (i+1+j)/2
            i=j
        out["auroc"]=(rank_sum-p*(p+1)/2)/(p*neg)
    if good is not None:
        count=sum(bool(y and g) for y,g in zip(labels,good))
        kept=sum(bool(y and g and a) for y,g,a in zip(labels,good,decisions))
        out.update(layout_correct_total=count, layout_correct_accepted=kept,
            layout_correct_rejected=count-kept, end_to_end_positive_recall=kept/p if p else None)
    return out


def select(labels, scores, valid, policy):
    if not any(labels) or all(labels):
        raise ValueError("both classes required for calibration")
    choices=[]
    for t in GRID:
        m=metrics(labels,[v and s>=t for s,v in zip(scores,valid)],scores)
        choices.append((t,m))
    if policy == "bounded_max_f1":
        # Fixed before evaluation. Equal F1 prefers proximity to the user's .3.
        t,m=max(choices,key=lambda tm:(tm[1]["f1"],-abs(tm[0]-.3),tm[1]["precision"],tm[0]))
        return dict(threshold=t,calibration_metrics=m)
    if policy == "bounded_recall95":
        eligible=[tm for tm in choices if tm[1]["recall"]>=.95]
        t,m=max(eligible,key=lambda tm:tm[0]) if eligible else choices[0]
        return dict(threshold=t,calibration_metrics=m,target_met=bool(eligible))
    raise ValueError(policy)


def crossfit(meta, predicted, policy):
    population=meta["pairs"]
    assert [r["pair_id"] for r in predicted] == [r["pair_id"] for r in population]
    labels=[r["label"] for r in population]; scores=[r["score"] for r in predicted]
    valid=[r["decision_valid"] for r in predicted]
    good=[r["layout_good_20"] for r in predicted] if meta["split"]=="real" else None
    decisions=[None]*len(population); result_rows=[]; folds=[]
    for k in range(5):
        cal=[i for i,r in enumerate(population) if r["fold"]!=k]
        test=[i for i,r in enumerate(population) if r["fold"]==k]
        def fragments(indices):
            return {population[i][s] for i in indices for s in ("fragment_a_id","fragment_b_id")}
        ca,te=fragments(cal),fragments(test)
        assert not ca&te
        assert not {meta["fragment_source_group"][s] for s in ca}&{meta["fragment_source_group"][s] for s in te}
        chosen=select([labels[i] for i in cal],[scores[i] for i in cal],[valid[i] for i in cal],policy)
        t=chosen["threshold"]
        assert .2<=t<=.8
        for i in test:
            assert decisions[i] is None
            decisions[i]=bool(valid[i] and scores[i]>=t)
            result_rows.append(dict(pair_id=population[i]["pair_id"],fold=k,label=labels[i],
                score=scores[i],threshold=t,accepted=decisions[i]))
        folds.append(dict(fold=k,calibration_count=len(cal),test_count=len(test),**chosen,
            held_out_metrics=metrics([labels[i] for i in test],[decisions[i] for i in test],
                [scores[i] for i in test],[good[i] for i in test] if good is not None else None)))
    assert all(x is not None for x in decisions)
    ts=[f["threshold"] for f in folds]
    return dict(pooled_out_of_fold=metrics(labels,decisions,scores,good),folds=folds,
        threshold_median=statistics.median(ts),threshold_min=min(ts),threshold_max=max(ts)),result_rows
