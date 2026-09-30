"""Read-only numerical digest of seven arms' fixed C8/C16 held-out endpoints.

No model selection, inference, threshold fitting, input mutation, or HTML.
Sources are the evaluator's complete protocol/summary and unique pair rows.
The reviewed REAL cohort is295 positives PLUS all508 original negatives.
Positive-only OOD has recall/accepted/rejected counts, never binary or pose metrics.
"""
import argparse
import json
import math
from pathlib import Path
import sys

SCHEMA = "matched-priority-fixed-endpoint-summary/1"
ARMS = ("all_tokens", "matched_tokens", "edge_seed", "edge_multi", "matched_edges", "G0", "G1")
BUDGETS = (16, 8)
SPLITS = ("test", "real", "ood")
OPS = ("max_f1", "recall_95")
DECODER = "full_top2_mode"
COUNTS = {"test": (3000, 1500), "real": (1016, 508), "ood": (301, 301)}
COHORTS = {"test": "all", "real": "kept_plus_all_negative", "ood": "all"}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def close(actual, expected, name):
    if (type(actual) not in (int, float) or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10)):
        raise ValueError("summary differs from pair rows: " + name)


def probability(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid " + name)
    return float(value)


def population(rows, split):
    """Small ID/label/cohort checks, not a whole-checkpoint/file integrity audit."""
    ids = [r.get("pair_id") for r in rows]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicate pair IDs")
    if any(type(r.get("label")) is not bool or type(r.get("decision_valid")) is not bool for r in rows):
        raise ValueError("row label/decision validity must be bool")
    positives = [r for r in rows if r["label"]]
    negatives = [r for r in rows if not r["label"]]
    if (len(rows), len(positives)) != COUNTS[split]:
        raise ValueError("wrong complete population: " + split)
    kept, excluded = [], []
    if split == "real":
        kept = [r for r in positives if r.get("review_status") == "keep"]
        excluded = [r for r in positives if r.get("review_status") == "exclude"]
        if len(kept) != 295 or len(excluded) != 213:
            raise ValueError("REAL must retain reviewed295 positives and exclude213")
        selected = kept + negatives
    else:
        selected = rows
    if split == "ood" and any(r.get("layout_gt_available") is not False
            or r.get("target_translation_rc") is not None
            or r.get("layouts", {}).get(DECODER, {}).get("translation_l2_px") is not None for r in rows):
        raise ValueError("OOD must remain positive-only without layout GT")
    registry = dict(source_count=len(rows), source_positive_count=len(positives),
        source_negative_count=len(negatives), selected_count=len(selected),
        selected_positive_count=sum(r["label"] for r in selected),
        selected_negative_count=sum(not r["label"] for r in selected),
        pair_ids=sorted(ids), positive_pair_ids=sorted(r["pair_id"] for r in positives),
        negative_pair_ids=sorted(r["pair_id"] for r in negatives),
        selected_pair_ids=sorted(r["pair_id"] for r in selected),
        kept_positive_pair_ids=sorted(r["pair_id"] for r in kept),
        excluded_positive_pair_ids=sorted(r["pair_id"] for r in excluded),
        cohort=COHORTS[split])
    return selected, registry


def layout_counts(rows, accepted):
    positives = [r for r in rows if r["label"]]
    good, known = [], []
    for row in positives:
        layout = row["layouts"][DECODER]
        error = layout.get("translation_l2_px")
        if error is not None:
            if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
                raise ValueError("invalid layout error")
            known.append(row["pair_id"])
            if layout["valid"] and error <= 20:
                good.append(row["pair_id"])
    rejected = sorted(set(good) - accepted)
    n = len(positives)
    return dict(positive_count=n, tolerance_px=20, error_available_count=len(known),
        raw_correct=len(good), raw_recall=len(good)/n,
        accepted_correct=len(set(good) & accepted),
        classification_FN_but_layout_correct=len(rejected),
        accepted_positive_bad_layout=sum(r["pair_id"] in accepted and r["pair_id"] not in good for r in positives),
        end_to_end_positive_recall=len(set(good) & accepted)/n,
        accepted_negative_count=sum(r["pair_id"] in accepted for r in rows if not r["label"]),
        correct_pair_ids=sorted(good), classification_rejected_correct_pair_ids=rejected,
        denominator="all selected positive pairs; missing/invalid pose is not successful")


def metrics(rows, source, threshold, split):
    accepted = set()
    for row in rows:
        score = probability(row["classification"]["fused"], "fused probability")
        if row["decision_valid"] and score >= threshold:
            accepted.add(row["pair_id"])
    p = {r["pair_id"] for r in rows if r["label"]}
    n = {r["pair_id"] for r in rows if not r["label"]}
    tp, fn = len(p & accepted), len(p - accepted)
    close(source.get("threshold"), threshold, "threshold")
    result = dict(threshold=threshold, recall=tp/len(p), tp=tp, fn=fn,
        false_negative_pair_ids=sorted(p-accepted))
    if split == "ood":
        for key, value in dict(accepted_positive_count=tp, false_negative_count=fn, positive_recall=tp/len(p)).items():
            close(source.get(key), value, key)
        return result
    fp, tn = len(n & accepted), len(n-accepted)
    expected = dict(tp=tp, fn=fn, fp=fp, tn=tn, accuracy=(tp+tn)/len(rows),
        precision=tp/(tp+fp) if tp+fp else 0., recall=tp/len(p),
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    for key, value in expected.items():
        close(source.get(key), value, key)
    result.update(expected, auprc=probability(source.get("auprc"), "source AUPRC"),
        false_positive_pair_ids=sorted(n & accepted))
    return result


def endpoint(directory, arm, budget, split):
    directory = Path(directory)
    out = dict(arm=arm, head_budget=budget, split=split, status="pending", source=str(directory))
    paths = {name: directory/name for name in ("protocol.json", "summary.json", "pair_results.jsonl")}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return dict(out, pending_reason="missing endpoint files", missing=missing), None
    protocol, summary = read(paths["protocol.json"]), read(paths["summary.json"])
    if protocol.get("status") != "complete" or summary.get("status") != "complete":
        return dict(out, pending_reason="endpoint not complete", protocol_status=protocol.get("status"),
                    summary_status=summary.get("status")), None
    identity = summary["model"]
    if (summary.get("split") != split or protocol.get("split") != split
            or identity.get("selection") != "fixed_epoch" or identity.get("head_budget") != budget
            or identity.get("training_identity", {}).get("arm") != arm
            or summary.get("selection_on_this_population") is not False
            or summary.get("threshold_fitting_performed") is not False):
        raise ValueError("not the registered fixed-budget held-out endpoint: " + str(directory))
    rows = [json.loads(line) for line in paths["pair_results.jsonl"].read_text().splitlines() if line.strip()]
    selected, registry = population(rows, split)
    group = summary["groups"][COHORTS[split]]
    for key, value in dict(sample_count=len(selected), positive_count=registry["selected_positive_count"],
            negative_count=registry["selected_negative_count"], decision_valid_count=sum(r["decision_valid"] for r in selected)).items():
        close(group.get(key), value, key)
    close(protocol.get("sample_count"), len(rows), "protocol sample_count")
    thresholds = identity["operating_points"]["thresholds"]
    points = {}
    for op in OPS:
        threshold = probability(thresholds[op], op+" frozen threshold")
        points[op] = metrics(selected, group["classification"]["fused"][op], threshold, split)
        if split != "ood":
            accepted = {r["pair_id"] for r in selected if r["decision_valid"] and r["classification"]["fused"] >= threshold}
            layout = layout_counts(selected, accepted)
            official = group["layout"][op]["20"]
            for key in ("positive_count", "tolerance_px", "raw_correct", "raw_recall", "accepted_correct",
                    "classification_FN_but_layout_correct", "accepted_positive_bad_layout",
                    "end_to_end_positive_recall", "accepted_negative_count"):
                close(official.get(key), layout[key], "layout "+key)
            points[op]["layout_le20"] = layout
    if split != "ood":
        close(points["max_f1"]["auprc"], points["recall_95"]["auprc"], "same endpoint AUPRC")
    out.update(status="complete", cohort=COHORTS[split],
        sample_count=len(selected), positive_count=registry["selected_positive_count"],
        negative_count=registry["selected_negative_count"],
        decision_valid_count=group["decision_valid_count"], operating_points=points,
        checkpoint_sha256=identity.get("checkpoint_sha256"),
        selection="fixed_epoch", threshold_source="clean SIMVAL3000 only",
        source_summary=str(paths["summary.json"]), source_rows=str(paths["pair_results.jsonl"]))
    if split == "ood":
        out["unavailable"] = "Positive-only OOD301: Accuracy, Precision, F1, AUPRC, FP/TN and layout-GT metrics are unavailable."
    return out, registry


def collect(root):
    root = Path(root).resolve(strict=True)
    endpoints, populations = [], {}
    for arm in ARMS:
        folder = "adaptation_training" if arm in ("G0", "G1") else "training"
        for budget in BUDGETS:
            for split in SPLITS:
                row, registry = endpoint(root/folder/arm/"evaluation"/("c%d" % budget)/split, arm, budget, split)
                if registry is not None:
                    if split in populations and registry != populations[split]:
                        raise ValueError("pair IDs/labels/review cohort changed between endpoints: " + split)
                    populations[split] = registry
                endpoints.append(row)
    complete = sum(r["status"] == "complete" for r in endpoints)
    return dict(schema=SCHEMA, status="complete" if complete == 42 else "partial",
        expected_endpoints=42, completed_endpoints=complete, pending_endpoints=42-complete,
        root=str(root), checkpoint_selection_performed=False, threshold_fitting_performed=False,
        metrics_source="original evaluator summary; confusion/layout counts checked against unique pair rows",
        populations=populations, endpoints=endpoints,
        notes=["C16 primary and C8 retained are fixed budgets, not a held-out winner selection.",
            "REAL is the manually reviewed295 positives plus all508 original negatives (803 pairs).",
            "REAL review-based filtering is descriptive; it is not an independent untouched test population.",
            "Thresholds are fixed clean-SIMVAL max-F1 and R95; R95 need not deliver95% recall on held-out data.",
            "Invalid decisions reject operationally; layout success requires valid output and translation error <=20px.",
            "Missing endpoints remain pending, never zero-valued measurements."])


def markdown(report):
    lines = ["# Fixed-budget priority experiment summary", "",
        "Status: **%s** — %d/%d endpoints complete." % (report["status"], report["completed_endpoints"], report["expected_endpoints"]), "",
        "Thresholds come only from SIMVAL; no checkpoint selection or threshold fitting was performed here.", ""]
    for split, title in (("test", "SIMTEST:1500 positive +1500 negative"),
                         ("real", "Reviewed Dunhuang:295 positive +508 negative"),
                         ("ood", "Turufan OOD:301 positives only")):
        lines.extend(["## " + title, ""])
        if split == "ood":
            lines.extend(["No negative pairs or layout GT: Accuracy, F1, AUPRC and pose success are unavailable.", "",
                "| Arm | Budget | SIMVAL workpoint | Threshold | Recall | Accepted | Rejected |",
                "| --- | --- | --- | --- | --- | --- | --- |"])
        else:
            lines.extend(["| Arm | Budget | SIMVAL workpoint | Threshold | Accuracy | Recall | F1 | AUPRC | FP | FN | Layout≤20px | Correct layout rejected |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
        for row in report["endpoints"]:
            if row["split"] != split:
                continue
            if row["status"] != "complete":
                tail = ["pending"] + ["—"]*(3 if split == "ood" else 8)
                lines.append("| " + " | ".join([row["arm"], "C%d"%row["head_budget"], "pending"]+tail) + " |")
                continue
            for op in OPS:
                m = row["operating_points"][op]
                values = [row["arm"], "C%d"%row["head_budget"], op, "%.6g"%m["threshold"]]
                percent = lambda v: "%.2f%%"%(100*v)
                if split == "ood":
                    values += [percent(m["recall"]), str(m["tp"]), str(m["fn"])]
                else:
                    layout = m["layout_le20"]
                    values += [percent(m[k]) for k in ("accuracy", "recall", "f1", "auprc")]
                    values += [str(m["fp"]), str(m["fn"]), "%d/%d (%s)"%(layout["raw_correct"],
                        layout["positive_count"], percent(layout["raw_recall"])), str(layout["classification_FN_but_layout_correct"])]
                lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(["## Interpretation limits", ""] + ["- "+n for n in report["notes"]] + ["",
        "Complete population IDs, FP/FN IDs and correct-layout-but-rejected IDs are retained in summary.json.", ""])
    return "\n".join(lines)


def run(args):
    report = collect(args.root)
    output = Path(args.output).resolve()
    # New output only; never overwrite source endpoint folders or a prior report.
    if output == Path(args.root).resolve() or output in Path(args.root).resolve().parents:
        raise ValueError("summary output must not contain the source root")
    if any(output == Path(r["source"]) or output in Path(r["source"]).parents
            or Path(r["source"]) in output.parents for r in report["endpoints"]):
        raise ValueError("summary output must be separate from endpoint trees")
    output.mkdir(parents=True, exist_ok=False)
    save(output/"summary.json", report)
    (output/"summary.md").write_text(markdown(report))
    status = {key:report[key] for key in ("schema", "status", "expected_endpoints", "completed_endpoints", "pending_endpoints")}
    status.update(summary_json=str(output/"summary.json"), summary_markdown=str(output/"summary.md"))
    save(output/"status.json", status)
    if report["status"] != "complete" and not args.allow_partial:
        raise RuntimeError("incomplete fixed endpoints; partial report saved, use --allow-partial for an explicitly partial readout")
    return status


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-partial", action="store_true")
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))
