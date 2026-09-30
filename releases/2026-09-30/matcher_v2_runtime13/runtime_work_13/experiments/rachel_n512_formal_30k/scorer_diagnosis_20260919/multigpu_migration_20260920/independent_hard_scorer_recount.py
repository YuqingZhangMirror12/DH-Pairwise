"""Independent saved-row counts; no model, metric helper, fitting or selection.

--root accepts only the completed full6000 hard-scorer inference. The optional
--self-test uses tiny in-memory fixtures, never partial experimental results.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

ARMS = ("all_tokens", "matched_tokens", "matched_edges")
HEADS = tuple("%s_c%d" % (arm, epoch) for arm in ARMS for epoch in (8, 16))
OPS = ("max_f1", "recall_99")
RECIPES = ("wave", "local", "seam_gaps", "partial_curve")


def average(values):
    return math.fsum(values) / len(values) if values else None


def loss(logit, label):
    return max(logit, 0.) - label * logit + math.log1p(math.exp(-abs(logit)))


def layout_correct(row):
    error = row["raw_translation_l2_px"]
    return bool(row["label"] and row["raw_layout_valid"] and error is not None and error <= 20.)


def accepts(row, head, threshold):
    return bool(row["decision_valid"] and row["scores"][head]["probability"] >= threshold)


def counts(rows, head, threshold):
    positives = [row for row in rows if row["label"]]
    negatives = [row for row in rows if not row["label"]]
    tp = sum(accepts(row, head, threshold) for row in positives)
    fp = sum(accepts(row, head, threshold) for row in negatives)
    correct = [row for row in positives if layout_correct(row)]
    accepted_good = sum(accepts(row, head, threshold) for row in correct)
    return dict(threshold=threshold, n=len(rows), positives=len(positives), negatives=len(negatives),
        tp=tp, fp=fp, fn=len(positives)-tp, tn=len(negatives)-fp,
        raw_correct_layout20=len(correct), accepted_correct_layout20=accepted_good,
        rejected_correct_layout20=len(correct)-accepted_good)


def transitions(rows, left_head, right_head, left_threshold, right_threshold):
    def subset(values):
        states = defaultdict(list)
        for row in values:
            before = accepts(row, left_head, left_threshold)
            after = accepts(row, right_head, right_threshold)
            states[(before, after)].append(row["pair_id"])
        names = {(False, False): "both_rejected", (False, True): "gained_acceptance",
                 (True, False): "lost_acceptance", (True, True): "both_accepted"}
        return {name: dict(count=len(states[state]), pair_ids=sorted(states[state]))
                for state, name in names.items()}
    return dict(left_threshold=left_threshold, right_threshold=right_threshold,
        positives=subset([row for row in rows if row["label"]]),
        negatives=subset([row for row in rows if not row["label"]]),
        correct_layout_positives=subset([row for row in rows if layout_correct(row)]))


def source_views(rows, expected_sources):
    if len(rows) != 2*expected_sources or len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("exactly two unique views per source required")
    grouped = defaultdict(list)
    for row in rows:
        if row["label"] not in (0, 1) or row["recipe"] not in ("clean", *RECIPES):
            raise ValueError("invalid label/recipe")
        for key in ("changed_pair", "source_family_overlap", "training_valid", "decision_valid", "raw_layout_valid"):
            if type(row.get(key)) is not bool:
                raise ValueError("explicit boolean required: " + key)
        error = row["raw_translation_l2_px"]
        if error is not None and (not math.isfinite(error) or error < 0):
            raise ValueError("invalid decoded layout error")
        if row["label"] and row["raw_layout_valid"] and error is None:
            raise ValueError("valid positive layout lacks error")
        if row.get("raw_layout20_correct", layout_correct(row)) != layout_correct(row):
            raise ValueError("stored layout20 differs from decoded error")
        if set(row["scores"]) != set(HEADS):
            raise ValueError("six complete heads required")
        for score in row["scores"].values():
            logit, probability = score["logit"], score["probability"]
            if not math.isfinite(logit) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("nonfinite logit/probability")
            expected = math.exp(-max(-logit, 0.))/(1.+math.exp(-abs(logit)))
            if abs(expected-probability) > 1e-6:
                raise ValueError("saved logit/probability mismatch")
            if not row["training_valid"] and (logit != 0 or probability != .5):
                raise ValueError("deployed invalid-training masking differs")
        grouped[row["source_pair_id"]].append(row)
    if len(grouped) != expected_sources:
        raise ValueError("unique source count differs")
    clean, variants = [], []
    for source in sorted(grouped):
        pair = grouped[source]
        if len(pair) != 2 or sum(row["recipe"] == "clean" for row in pair) != 1:
            raise ValueError("requires one clean and one requested derivative")
        reference = next(row for row in pair if row["recipe"] == "clean")
        variant = next(row for row in pair if row["recipe"] != "clean")
        if (reference["changed_pair"] or reference["label"] != variant["label"]
                or reference["source_family_overlap"] != variant["source_family_overlap"]
                or reference["assigned_recipe"] != variant["recipe"]
                or variant["assigned_recipe"] != variant["recipe"]):
            raise ValueError("paired source metadata differs")
        clean.append(reference)
        variants.append(variant)
    if expected_sources == 3000:
        histogram = Counter((r["recipe"], bool(r["label"])) for r in rows)
        if any(histogram["clean", label] != 1500 or any(histogram[recipe, label] != 375 for recipe in RECIPES)
               for label in (False, True)):
            raise ValueError("unexpected source or requested-recipe label counts")
    return clean, variants


def recount(rows, thresholds, expected_sources=3000):
    clean, variants = source_views(rows, expected_sources)
    if set(thresholds) != set(HEADS) or any(not set(OPS) <= set(thresholds[h]) for h in HEADS):
        raise ValueError("original six-head maxF1/R99 thresholds required")
    if any(not math.isfinite(thresholds[h][op]) or not 0 <= thresholds[h][op] <= 1 for h in HEADS for op in OPS):
        raise ValueError("invalid frozen threshold")
    populations = dict(clean=clean, full_requested=variants,
        actually_changed=[r for r in variants if r["changed_pair"]],
        unchanged_fallback=[r for r in variants if not r["changed_pair"]])
    by_source = {r["source_pair_id"]: r for r in clean}
    for recipe in RECIPES:
        requested = [r for r in variants if r["recipe"] == recipe]
        populations[recipe+"/clean"] = [by_source[r["source_pair_id"]] for r in requested]
        populations[recipe+"/full_requested"] = requested
        populations[recipe+"/actually_changed"] = [r for r in requested if r["changed_pair"]]
        populations[recipe+"/unchanged_fallback"] = [r for r in requested if not r["changed_pair"]]
    result = dict(schema="independent-hard-scorer-recount/1", status="complete",
        source_count=expected_sources, view_count=len(rows), populations={}, thresholds=thresholds,
        thresholds_fitted=False, training_performed=False,
        source_pairs=[dict(source_pair_id=a["source_pair_id"], label=a["label"],
            clean_pair_id=a["pair_id"], requested_pair_id=b["pair_id"], recipe=b["recipe"],
            changed_pair=b["changed_pair"], fallback_reason=b.get("fallback_reason"),
            source_family_overlap=a["source_family_overlap"]) for a,b in zip(clean, variants)],
        interpretation="3000 paired source pairs, not 6000 independent sources. C16 at C8 threshold is diagnostic only, never a newly calibrated/deployed operating point.",
        bce_definition="stable BCE on deployed logit; training loss separately mean(BCE*training_valid) with all-row denominator",
        layout_definition="positive && production layout_valid && actual translation error<=20px")
    for name, selected in populations.items():
        item = dict(n=len(selected), source_count=len({r["source_pair_id"] for r in selected}),
            positives=sum(r["label"] for r in selected),
            source_family_overlap_count=sum(r["source_family_overlap"] for r in selected),
            changed_count=sum(r["changed_pair"] for r in selected),
            fallback_reasons=dict(Counter(r.get("fallback_reason") or "unspecified" for r in selected
                if r["recipe"] != "clean" and not r["changed_pair"])), heads={}, c8_to_c16={})
        for head in HEADS:
            losses = [loss(r["scores"][head]["logit"], r["label"]) for r in selected]
            item["heads"][head] = dict(mean_pair_bce=average(losses),
                mean_training_pair_bce_all_rows=average([value*int(r["training_valid"]) for value,r in zip(losses,selected)]),
                training_valid_count=sum(r["training_valid"] for r in selected),
                decision_valid_count=sum(r["decision_valid"] for r in selected),
                mean_logit=average([r["scores"][head]["logit"] for r in selected]),
                operating_points={op: counts(selected, head, thresholds[head][op]) for op in OPS})
        for arm in ARMS:
            left, right = arm+"_c8", arm+"_c16"
            changes = dict(mean_pair_bce_delta=average([loss(r["scores"][right]["logit"], r["label"])
                -loss(r["scores"][left]["logit"], r["label"]) for r in selected]), operating_points={})
            for op in OPS:
                old, new = thresholds[left][op], thresholds[right][op]
                changes["operating_points"][op] = dict(
                    own_thresholds=transitions(selected,left,right,old,new),
                    c16_at_fixed_c8_threshold=counts(selected,right,old),
                    score_change_at_fixed_c8_threshold=transitions(selected,left,right,old,old),
                    threshold_change_on_same_c16_scores=transitions(selected,right,right,old,new))
            item["c8_to_c16"][arm] = changes
        result["populations"][name] = item
    return result


def load_complete(root):
    root = Path(root)
    protocol_bytes = (root/"protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    if (protocol.get("schema") != "hard-simval-six-frozen-scorers/1"
            or protocol.get("status") != "complete" or protocol.get("pilot") is not False
            or protocol.get("sample_count") != 6000 or protocol.get("completed_count") != 6000
            or any(protocol.get(k) is not False for k in ("thresholds_fitted", "training_performed", "model_selection_performed"))):
        raise ValueError("only completed full6000 frozen inference may be recounted")
    thresholds = protocol["thresholds"]
    for arm in ARMS:
        for epoch in (8,16):
            key = "%s_c%d" % (arm,epoch)
            receipt = protocol["heads"][key]
            if (receipt["operating_points"]["thresholds"] != thresholds[key]
                    or receipt["head_budget"] != epoch or receipt["head_epoch"] != epoch
                    or receipt["selection"] != "fixed_epoch" or receipt["training_identity"]["arm"] != arm
                    or receipt["source_matcher_sha256"] != protocol["source_matcher_sha256"]):
                raise ValueError("threshold/checkpoint receipt differs")
    contents = (root/"cases.jsonl").read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != protocol["cases_sha256"]:
        raise ValueError("completed case receipt differs")
    rows = [json.loads(line) for line in contents.splitlines() if line.strip()]
    if [r["pair_id"] for r in rows] != protocol["pair_ids"]:
        raise ValueError("recorded case population differs")
    return rows, thresholds, dict(root=str(root.resolve()), cases_sha256=digest,
        protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest())


def self_test():
    rows = []
    thresholds = {h:dict(max_f1=.6 if h.endswith("c8") else .9, recall_99=.5) for h in HEADS}
    for label in (False,True):
        for view in ("clean","wave"):
            logits = {h:(1. if label else -1.)+(1. if h.endswith("c16") else 0.) for h in HEADS}
            rows.append(dict(pair_id=str(label)+view,source_pair_id=str(label),recipe=view,assigned_recipe="wave",
                label=label,changed_pair=view!="clean",source_family_overlap=False,
                training_valid=True,decision_valid=True,raw_layout_valid=True,
                raw_translation_l2_px=20. if label else None,
                scores={h:dict(logit=v,probability=1/(1+math.exp(-v))) for h,v in logits.items()}))
    result = recount(rows,thresholds,2)
    clean = result["populations"]["clean"]
    changes = clean["c8_to_c16"]["matched_edges"]["operating_points"]["max_f1"]
    assert clean["heads"]["matched_edges_c8"]["operating_points"]["max_f1"]["accepted_correct_layout20"] == 1
    assert changes["own_thresholds"]["correct_layout_positives"]["lost_acceptance"]["count"] == 1
    assert changes["c16_at_fixed_c8_threshold"]["accepted_correct_layout20"] == 1
    assert changes["threshold_change_on_same_c16_scores"]["positives"]["lost_acceptance"]["count"] == 1
    assert result["populations"]["local/actually_changed"]["heads"]["all_tokens_c8"]["mean_pair_bce"] is None
    assert result["source_count"] == 2 and result["view_count"] == 4
    assert math.isfinite(loss(10000,0)) and math.isfinite(loss(-10000,1))
    rows[0]["training_valid"] = rows[0]["decision_valid"] = False
    for score in rows[0]["scores"].values(): score.update(logit=0., probability=.5)
    masked = recount(rows,thresholds,2)["populations"]["clean"]["heads"]["all_tokens_c8"]
    assert abs(masked["mean_training_pair_bce_all_rows"]-loss(1.,1)/2) < 1e-12
    try:
        recount(rows[:-1],thresholds,2)
    except ValueError:
        pass
    else:
        raise AssertionError("partial rows accepted")
    print("self_test: counts, thresholds, paired sources, stable/masked BCE, empty and incomplete groups passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None or args.output is None:
        parser.error("--root and new --output are required")
    rows,thresholds,source = load_complete(args.root)
    result = recount(rows,thresholds)
    result["source"] = source
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status="complete",source_count=result["source_count"],view_count=result["view_count"],output=str(args.output))))


if __name__ == "__main__":
    main()
