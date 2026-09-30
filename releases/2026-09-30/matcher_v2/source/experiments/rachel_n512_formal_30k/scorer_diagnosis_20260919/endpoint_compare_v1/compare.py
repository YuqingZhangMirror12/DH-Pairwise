"""Offline complete-endpoint comparison. Python stdlib; never infer, fit, or fetch.

Usage: python compare.py --endpoint C0:real=/path/to/real ... --output /new/dir
Each model should supply test/real/ood endpoints; omitted/incomplete inputs remain
explicitly unavailable. --reference-model C0 adds its frozen SIMVAL thresholds.
--demo-existing uses the already collected local S6/S7 C8/C16 endpoints.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPS = ("max_f1", "recall_95")
EXPECTED = {"test": (3000, 1500, 1500), "real": (1016, 508, 508), "ood": (301, 301, 0)}
REAL_COUNTS = {"all1016": (1016, 508, 508), "keep803": (803, 295, 508),
               "strict547": (547, 508, 39), "keep_strict334": (334, 295, 39)}
OFFICIAL = {"all1016": "all", "keep803": "kept_plus_all_negative",
            "keep_strict334": "kept_plus_strict_negative", "test3000": "all", "ood301": "all"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def source_read(path, sources, lines=False):
    data = path.read_bytes()
    sources[path.name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
    return [json.loads(x) for x in data.splitlines() if x.strip()] if lines else json.loads(data)


def load_endpoint(path, split):
    """Protocol readiness gates every read of summary/rows; no missing-as-zero."""
    result = {"path": str(path), "split": split, "status": "unavailable", "sources": {}}
    try:
        protocol = source_read(path / "protocol.json", result["sources"])
        if protocol.get("status") != "complete":
            result["reason"] = "protocol_not_complete: " + str(protocol.get("status"))
            return result
        summary = source_read(path / "summary.json", result["sources"])
        if summary.get("status") != "complete" or protocol["split"] != split or summary["split"] != split:
            raise ValueError("summary not complete or split mismatch")
        # These declarations are required, not inferred from the endpoint's name.
        for field in ("thresholds_fitted", "test_or_real_used_for_fit", "ood_used_for_fit"):
            if protocol.get(field) is not False:
                raise ValueError("missing/unsafe fit provenance: protocol." + field)
        for field in ("selection_on_this_population", "threshold_fitting_performed"):
            if summary.get(field) is not False:
                raise ValueError("missing/unsafe fit provenance: summary." + field)
        model = summary["model"]
        thresholds = model["operating_points"]["thresholds"]
        for op in OPS:
            if not finite(thresholds[op]) or not 0 <= thresholds[op] <= 1:
                raise ValueError("invalid SIMVAL threshold " + op)
            if thresholds[op] != protocol["model"]["operating_points"]["thresholds"][op]:
                raise ValueError("protocol/summary threshold mismatch")
        if model["checkpoint_sha256"] != protocol["model"]["checkpoint_sha256"]:
            raise ValueError("protocol/summary checkpoint mismatch")
        rows = source_read(path / "pair_results.jsonl", result["sources"], lines=True)
        indexed = {}
        for row in rows:
            for key in ("pair_id", "label", "fragment_a", "fragment_b",
                        "target_translation_rc", "decision_valid", "classification"):
                if key not in row:
                    raise ValueError("missing row field " + key + ": " + str(row.get("pair_id")))
            if split == "real" and any(k not in row for k in ("strict_member", "review_status")):
                raise ValueError("missing REAL strict_member/review_status: " + row["pair_id"])
            if row["label"] not in (True, False) or not isinstance(row["decision_valid"], bool):
                raise ValueError("invalid label or decision_valid")
            score = row["classification"]["fused"]
            if not finite(score) or not 0 <= score <= 1:
                raise ValueError("missing/nonfinite/nonprobability score: " + row["pair_id"])
            if row["pair_id"] in indexed:
                raise ValueError("duplicate pair_id: " + row["pair_id"])
            indexed[row["pair_id"]] = row
        counts = (len(rows), sum(bool(r["label"]) for r in rows), sum(not r["label"] for r in rows))
        result["observed_counts"] = counts
        if len(rows) != protocol["sample_count"] or counts != EXPECTED[split]:
            raise ValueError(f"population count mismatch: observed {counts}, expected {EXPECTED[split]}")
        result.update(status="complete", rows=indexed, summary=summary,
                      model={k: model.get(k) for k in ("checkpoint_path", "checkpoint_sha256", "epoch", "selection")},
                      thresholds={op: thresholds[op] for op in OPS})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def identity_difference(reference, current, split):
    a, b = set(reference), set(current)
    fields = ("label", "fragment_a", "fragment_b", "target_translation_rc")
    if split == "real":
        fields += ("strict_member", "review_status")
    mismatches = [{"pair_id": i, "fields": [k for k in fields if reference[i][k] != current[i][k]]}
                  for i in sorted(a & b) if any(reference[i][k] != current[i][k] for k in fields)]
    return {"equal": not (a ^ b) and not mismatches, "missing_pair_ids": sorted(a-b),
            "extra_pair_ids": sorted(b-a), "field_mismatches": mismatches,
            "order_equal": list(reference) == list(current)}


def populations(rows, split):
    rows = list(rows)
    if split != "real":
        return {"test3000" if split == "test" else "ood301": rows}
    return {"all1016": rows,
            "keep803": [r for r in rows if not r["label"] or r["review_status"] == "keep"],
            "strict547": [r for r in rows if r["label"] or r["strict_member"]],
            "keep_strict334": [r for r in rows if (r["label"] and r["review_status"] == "keep")
                                or (not r["label"] and r["strict_member"])]}


def ranking(rows):
    """Tie-grouped trapezoidal ROC and step Average Precision, not PR trapezoids."""
    groups = defaultdict(lambda: [0, 0])
    for r in rows:
        groups[r["classification"]["fused"] if r["decision_valid"] else -1][not r["label"]] += 1
    p = sum(r["label"] for r in rows); n = len(rows) - p
    if not p or not n:
        return {"auroc": None, "auprc": None}
    tp = fp = old_tpr = old_fpr = auc = ap = 0
    for score in sorted(groups, reverse=True):
        positive, negative = groups[score]; tp += positive; fp += negative
        tpr, fpr = tp/p, fp/n
        auc += (fpr-old_fpr)*(tpr+old_tpr)/2
        ap += (tpr-old_tpr)*tp/(tp+fp)
        old_tpr, old_fpr = tpr, fpr
    return {"auroc": auc, "auprc": ap}


def accepted(row, threshold):
    return row["decision_valid"] and row["classification"]["fused"] >= threshold


def layout_state(row):
    """None is unavailable GT/decoder evidence, False is an observed failed layout."""
    layout = row.get("layouts", {}).get("full_top2_mode")
    gt = row.get("target_translation_rc")
    if not isinstance(gt, list) or len(gt) != 2 or not all(finite(v) for v in gt):
        return None
    if not isinstance(layout, dict) or not isinstance(layout.get("valid"), bool):
        return None
    if not layout["valid"]:
        return False
    error = layout.get("translation_l2_px")
    return error <= 20 if finite(error) and error >= 0 else None


def metrics(rows, threshold, split):
    positive = sum(r["label"] for r in rows); negative = len(rows)-positive
    tp = sum(accepted(r, threshold) and r["label"] for r in rows)
    fp = sum(accepted(r, threshold) and not r["label"] for r in rows)
    out = {"threshold": threshold, "sample_count": len(rows), "positive_count": positive,
           "negative_count": negative, "decision_valid_count": sum(r["decision_valid"] for r in rows)}
    if split == "ood":
        out.update(accepted_positive_count=tp, false_negative_count=positive-tp,
                   positive_recall=tp/positive if positive else None,
                   binary_metrics_unavailable="positive-only OOD; no Accuracy/Precision/F1/AUROC/AP",
                   layout_unavailable="OOD has no GT layout")
        return out
    fn, tn = positive-tp, negative-fp
    out.update(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=(tp+tn)/len(rows) if rows else None,
               precision=tp/(tp+fp) if tp+fp else 0., recall=tp/positive if positive else None,
               f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., **ranking(rows))
    pos = [r for r in rows if r["label"]]
    missing = [r["pair_id"] for r in pos if layout_state(r) is None]
    good = [r for r in pos if layout_state(r) is True]
    out["layout20"] = {"status": "partial" if missing else "complete", "gt_positive_denominator": positive,
        "known_layout_positive_count": positive-len(missing), "missing_pair_ids": missing,
        "known_raw_correct": len(good), "known_accepted_correct": sum(accepted(r, threshold) for r in good),
        "known_rejected_correct": sum(not accepted(r, threshold) for r in good),
        "raw_correct": None if missing else len(good),
        "accepted_correct": None if missing else sum(accepted(r, threshold) for r in good),
        "classification_FN_but_layout_correct": None if missing else sum(not accepted(r, threshold) for r in good)}
    return out


def layout_changes(reference, current):
    ids = sorted(reference.keys() & current.keys())
    missing = [i for i in ids if not reference[i].get("layouts") or not current[i].get("layouts")]
    changed = [i for i in ids if i not in missing and reference[i]["layouts"] != current[i]["layouts"]]
    success_changed = [i for i in ids if reference[i]["label"] and layout_state(reference[i]) is not None
                       and layout_state(current[i]) is not None and layout_state(reference[i]) != layout_state(current[i])]
    return {"paired_count": len(ids), "full_layout_changed_count": len(changed),
            "full_layout_changed_pair_ids": changed, "layout20_success_changed_pair_ids": success_changed,
            "missing_layout_pair_ids": missing,
            "note": "Different Matchers may have different layouts; equality is not required. Full-object changes can include diagnostic-only changes."}


def compare(inputs, reference_model=None):
    names = sorted({name for name, split in inputs})
    if reference_model is not None and reference_model not in names:
        raise ValueError("reference model not among named inputs")
    endpoints, comparisons = {}, []
    for split in EXPECTED:
        loaded = {name: load_endpoint(inputs[(name, split)], split) if (name, split) in inputs
                  else {"status": "unavailable", "reason": "endpoint_not_supplied"} for name in names}
        baseline = reference_model if reference_model and loaded[reference_model]["status"] == "complete" else next(
            (name for name in names if loaded[name]["status"] == "complete"), None)
        for name, endpoint in loaded.items():
            endpoints[name+":"+split] = endpoint
            if endpoint["status"] != "complete":
                continue
            rows = endpoint["rows"]
            endpoint["paired_identity_reference"] = baseline
            endpoint["identity_check"] = identity_difference(loaded[baseline]["rows"], rows, split)
            if not endpoint["identity_check"]["equal"]:
                endpoint.update(status="identity_mismatch", reason="No intersection-only metrics are reported")
                continue
            endpoint["layout_changes_vs_reference"] = layout_changes(loaded[baseline]["rows"], rows)
            for group, selected in populations(rows.values(), split).items():
                counts = (len(selected), sum(r["label"] for r in selected), sum(not r["label"] for r in selected))
                expected = REAL_COUNTS[group] if split == "real" else EXPECTED[split]
                for op in OPS:
                    sources = [("own_SIMVAL", name)]
                    if reference_model and name != reference_model:
                        sources.append(("reference_SIMVAL", reference_model))
                    for mode, threshold_model in sources:
                        record = {"model": name, "split": split, "population": group, "operating_point": op,
                                  "threshold_mode": mode, "threshold_model": threshold_model}
                        if counts != expected:
                            record.update(status="unavailable", reason="cohort_count_mismatch", observed_counts=counts, expected_counts=expected)
                        elif loaded[threshold_model]["status"] != "complete":
                            record.update(status="unavailable", reason="reference endpoint/threshold unavailable")
                        else:
                            m = metrics(selected, loaded[threshold_model]["thresholds"][op], split)
                            record.update(status="complete", metrics=m)
                            if mode == "own_SIMVAL" and group in OFFICIAL:
                                official = endpoint["summary"]["groups"][OFFICIAL[group]]["classification"]["fused"][op]
                                diffs = {k: {"computed": m.get(k), "official": value} for k, value in official.items()
                                         if not (finite(m.get(k)) and math.isclose(m[k], value, rel_tol=1e-10, abs_tol=1e-10))}
                                record["source_summary_agreement"] = not diffs
                                if diffs:
                                    record.update(status="summary_mismatch", summary_differences=diffs)
                        comparisons.append(record)
    # Row-level data stay in the supplied originals, not duplicated in this receipt.
    for endpoint in endpoints.values():
        endpoint.pop("rows", None); endpoint.pop("summary", None)
    return {"schema": "offline-endpoint-compare/1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if all(e["status"] == "complete" for e in endpoints.values())
                  and all(r["status"] == "complete" for r in comparisons) else "needs_attention",
        "reference_model": reference_model, "endpoints": endpoints, "comparisons": comparisons,
        "definitions": {"all1016": "508 positive + 508 negative",
            "keep803": "295 kept positive + all 508 negative", "strict547": "all 508 positive + 39 strict negative",
            "keep_strict334": "295 kept positive + 39 strict negative", "accept": "decision_valid and fused score >= frozen threshold",
            "AP": "tie-grouped step Average Precision", "layout20": "valid decoded translation L2 <=20px, GT positives only",
            "invalid_decisions": "reject; rank score -1 below every valid probability"},
        "no_inference": True, "no_remote_access": True, "no_threshold_fit": True,
        "caveats": ["Reviewed REAL subsets are post-review populations, not a fresh untouched test set.",
                    "strict547/keep_strict334 contain only39 negatives; do not conflate their precision with all-negative cohorts.",
                    "Same numeric threshold across models is a diagnostic, not calibration or model selection."]}


def existing_inputs():
    root = HERE.parent
    sources = {
        "S6_C8": root/"continuation_endpoint_v1/raw/attention_depth_20260915/s4_cross_attention_depth2",
        "S6_C16": root/"continuation_endpoint_v1/raw/scorer_diagnosis_20260919/continuation_v1/s6_d2_c16",
        "S7_C8": root/"continuation_s7_endpoint_v1/raw/s6_s7_20260915/priority_after_s5/s7_augmented_full24",
        "S7_C16": root/"continuation_s7_endpoint_v1/raw/scorer_diagnosis_20260919/continuation_v1/s7_c16"}
    return {(name, split): path/"evaluation/fixed_epoch"/split for name, path in sources.items() for split in EXPECTED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", default=[], metavar="NAME:SPLIT=PATH")
    parser.add_argument("--reference-model")
    parser.add_argument("--demo-existing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = existing_inputs() if args.demo_existing else {}
    for value in args.endpoint:
        key, path = value.split("=", 1); name, split = key.rsplit(":", 1)
        if not name or split not in EXPECTED or (name, split) in inputs:
            parser.error("invalid/duplicate endpoint: " + key)
        inputs[(name, split)] = Path(path).expanduser().resolve()
    if not inputs:
        parser.error("supply --endpoint or --demo-existing")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.resolve() == path.resolve() or path.resolve() in args.output.resolve().parents for path in inputs.values()):
        parser.error("output must be separate from source endpoints")
    result = compare(inputs, args.reference_model)
    (args.output/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    lines = ["# 离线endpoint比较", "", "状态："+result["status"]+"。完整数值、缺项、来源和layout变化在results.json。", "",
             "| 模型 | keep803 Accuracy | Recall | F1 | TP/FP | 原始摆对/通过/漏判 |", "|---|---:|---:|---:|---|---|"]
    for r in result["comparisons"]:
        if r["status"] == "complete" and r["population"] == "keep803" and r["operating_point"] == "max_f1" and r["threshold_mode"] == "own_SIMVAL":
            m = r["metrics"]; l = m["layout20"]
            lines.append(f"| {r['model']} | {m['accuracy']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} | {m['tp']}/{m['fp']} | {l['raw_correct']}/{l['accepted_correct']}/{l['classification_FN_but_layout_correct']} |")
    lines += ["", "四个REAL分组不是互斥集合，不应相加。strict547=508正例+39严格负例；keep+strict334=295保留正例+39严格负例。OOD301只报告正例召回。", "",
              "未完成protocol先行拦截；缺字段/缺pair不补0、不取交集。不同Matcher允许layout不同，单列逐ID变化。阈值均直接读取SIMVAL冻结结果，不在REAL/OOD拟合。", ""]
    (args.output/"FINDINGS.md").write_text("\n".join(lines))
    print(json.dumps({"status": result["status"], "endpoints": len(result["endpoints"]), "comparisons": len(result["comparisons"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
