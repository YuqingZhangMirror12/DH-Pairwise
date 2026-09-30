"""Independent saved-row C16 recount. No comparator/model imports or fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS = ("matched_tokens", "edge_seed", "matched_edges")
OPS = ("max_f1", "recall_99")
LAYOUT = "full_top2_mode"


def load(path):
    return json.loads(Path(path).read_text())


def accepted(row, threshold):
    return bool(row["decision_valid"] and row["classification"]["fused"] >= threshold)


def good(row):
    pose = row["layouts"][LAYOUT]
    return bool(row["label"] and pose["valid"] and pose["translation_l2_px"] is not None
                and pose["translation_l2_px"] <= 20.)


def auc(rows):
    # Direct positive-negative comparisons with exact half credit for ties.
    pos = [r["classification"]["fused"] if r["decision_valid"] else -1. for r in rows if r["label"]]
    neg = [r["classification"]["fused"] if r["decision_valid"] else -1. for r in rows if not r["label"]]
    assert pos and neg
    return sum(1. if p > n else .5 if p == n else 0. for p in pos for n in neg) / (len(pos) * len(neg))


def binary(rows, threshold):
    positive = [r for r in rows if r["label"]]
    negative = [r for r in rows if not r["label"]]
    tp = sum(accepted(r, threshold) for r in positive)
    fp = sum(accepted(r, threshold) for r in negative)
    fn, tn = len(positive)-tp, len(negative)-fp
    correct = [r for r in positive if good(r)]
    correct_accepted = sum(accepted(r, threshold) for r in correct)
    return dict(threshold=threshold, n=len(rows), positives=len(positive), negatives=len(negative),
        tp=tp, fp=fp, fn=fn, tn=tn, accuracy=(tp+tn)/len(rows), recall=tp/len(positive),
        precision=tp/(tp+fp) if tp+fp else 0., f1=2*tp/(2*tp+fp+fn),
        raw_layout_le20=len(correct), accepted_layout_le20=correct_accepted,
        correct_layout_rejected=len(correct)-correct_accepted,
        accepted_positive_bad_layout=tp-correct_accepted,
        strict_negative_count=sum(r["strict_member"] for r in negative),
        strict_fp=sum(accepted(r, threshold) and r["strict_member"] for r in negative),
        distractor_negative_count=sum(not r["strict_member"] for r in negative),
        distractor_fp=sum(accepted(r, threshold) and not r["strict_member"] for r in negative))


def read_endpoint(root, arm, split):
    path = root/arm/"evaluation"/"c16"/split
    protocol, summary = load(path/"protocol.json"), load(path/"summary.json")
    assert protocol["status"] == summary["status"] == "complete", path
    assert protocol["split"] == summary["split"] == split
    assert protocol["model"]["head_epoch"] == protocol["model"]["head_budget"] == 16
    assert protocol["model"]["training_identity"]["arm"] == arm
    assert protocol["thresholds_fitted"] is False
    assert protocol["test_or_real_used_for_fit"] is False and protocol["ood_used_for_fit"] is False
    rows = [json.loads(line) for line in (path/"pair_results.jsonl").read_text().splitlines()]
    assert len(rows) == protocol["sample_count"] == (1016 if split == "real" else 301)
    by_id = {r["pair_id"]: r for r in rows}
    assert len(by_id) == len(rows)
    assert all(math.isfinite(r["classification"]["fused"]) and 0 <= r["classification"]["fused"] <= 1 for r in rows)
    assert summary["model"]["checkpoint_sha256"] == protocol["model"]["checkpoint_sha256"]
    return protocol, summary, by_id


def run(root):
    values, inputs = {}, {}
    for arm in ARMS:
        values[arm], inputs[arm] = {}, {}
        for split in ("real", "ood"):
            protocol, summary, by_id = read_endpoint(root, arm, split)
            inputs[arm][split] = (protocol, by_id)
            rows = list(by_id.values())
            selected = [r for r in rows if not r["label"] or r["review_status"] == "keep"] if split == "real" else rows
            model = protocol["model"]
            values[arm][split] = dict(source=str((root/arm/"evaluation"/"c16"/split).resolve()),
                checkpoint_sha256=model["checkpoint_sha256"], source_matcher_sha256=model["source_matcher_sha256"],
                manifest_sha256=protocol["manifest_sha256"], keep_ids_sha256=protocol.get("keep_ids_sha256"),
                pair_ids=sorted(r["pair_id"] for r in selected), operating_points={})
            thresholds = model["operating_points"]["thresholds"]
            if split == "real":
                assert len(selected) == 803 and sum(r["label"] for r in selected) == 295
                assert sum(not r["label"] and r["strict_member"] for r in selected) == 39
                values[arm][split]["auroc"] = auc(selected)
                for op in OPS:
                    values[arm][split]["operating_points"][op] = binary(selected, thresholds[op])
                original = summary["groups"]["kept_plus_all_negative"]
                ours = values[arm][split]["operating_points"]["max_f1"]
                for key in ("tp", "fp", "fn", "tn"):
                    assert ours[key] == original["classification"]["fused"]["max_f1"][key]
                assert ours["accepted_layout_le20"] == original["layout"]["max_f1"]["20"]["accepted_correct"]
                assert abs(values[arm][split]["auroc"]-original["classification"]["fused"]["max_f1"]["auroc"]) < 1e-12
            else:
                assert all(r["label"] for r in rows)
                for op in OPS:
                    count = sum(accepted(r, thresholds[op]) for r in rows)
                    values[arm][split]["operating_points"][op] = dict(threshold=thresholds[op],
                        positive_count=301, accepted_positive_count=count, rejected_positive_count=301-count,
                        positive_acceptance_rate=count/301)
                assert values[arm][split]["operating_points"]["max_f1"]["accepted_positive_count"] == summary["groups"]["all"]["classification"]["fused"]["max_f1"]["accepted_positive_count"]
                values[arm][split]["unavailable"] = "positive-only: no full binary Accuracy/F1/AUROC; no layout ground truth"
    alignment, changes = {}, {}
    for split in ("real", "ood"):
        base_protocol, base_rows = inputs["matched_tokens"][split]
        for arm in ARMS[1:]:
            protocol, rows = inputs[arm][split]
            assert rows.keys() == base_rows.keys()
            for key in ("manifest_sha256", "keep_ids_sha256"):
                assert protocol.get(key) == base_protocol.get(key)
            assert protocol["model"]["source_matcher_sha256"] == base_protocol["model"]["source_matcher_sha256"]
            for pair_id, row in rows.items():
                prior = base_rows[pair_id]
                for key in ("fragment_a", "fragment_b", "label", "target_translation_rc", "strict_member", "review_status", "decision_valid", "layouts"):
                    assert row.get(key) == prior.get(key), (arm, split, pair_id, key)
        alignment[split] = dict(all_ids_labels_fragments_gt_review_decision_valid_and_final_layouts_identical=True,
                               source_matcher_and_manifest_identical=True, full_count=len(base_rows))
        changes[split] = {}
        _, final = inputs["matched_edges"][split]
        selected_ids = values["matched_edges"][split]["pair_ids"]
        for reference in ("matched_tokens", "edge_seed"):
            _, old = inputs[reference][split]
            changes[split][reference] = {}
            for op in OPS:
                threshold = values["matched_edges"][split]["operating_points"][op]["threshold"]
                old_threshold = values[reference][split]["operating_points"][op]["threshold"]
                groups = dict(positive_gained=[], positive_lost=[])
                if split == "real":
                    groups.update(negative_fp_gained=[], negative_fp_removed=[],
                                  correct_layout_accept_gained=[], correct_layout_accept_lost=[])
                for pair_id in selected_ids:
                    row = final[pair_id]
                    now, before = accepted(row, threshold), accepted(old[pair_id], old_threshold)
                    if now == before: continue
                    if row["label"]:
                        groups["positive_gained" if now else "positive_lost"].append(pair_id)
                        if split == "real" and good(row):
                            groups["correct_layout_accept_gained" if now else "correct_layout_accept_lost"].append(pair_id)
                    else:
                        groups["negative_fp_gained" if now else "negative_fp_removed"].append(pair_id)
                changes[split][reference][op] = {k:dict(count=len(v), pair_ids=v) for k,v in groups.items()}
    return dict(schema="independent-final-edges-c16-saved-row-recount/1", status="complete", results=values,
        alignment=alignment, paired_changes=changes,
        definitions=dict(decision="decision_valid AND fused score >= frozen SIMVAL threshold",
            layout="positive AND production final layout valid AND translation_l2_px <= 20",
            real_population="295 reviewed keep positives + all508 negatives; 39 strict and469 constructed distractors",
            auc="P(validity-adjusted positive score > negative score) + half ties; invalid scores rank -1",
            r99="threshold calibrated to 99% recall on SIMVAL; not a promised REAL/OOD recall",
            training_or_threshold_fit_performed=False,
            comparison_limit="matched_edges changes relation binding, Q/residual features and parameters versus matched_tokens; not one-factor count ablation"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent/"s7_direct")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("independent_recount.json"))
    args = parser.parse_args()
    result = run(args.root)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    for arm in ARMS:
        print(arm, json.dumps({split:{k:v for k,v in result["results"][arm][split].items()
              if k in ("auroc", "operating_points")} for split in ("real", "ood")}, ensure_ascii=False))
