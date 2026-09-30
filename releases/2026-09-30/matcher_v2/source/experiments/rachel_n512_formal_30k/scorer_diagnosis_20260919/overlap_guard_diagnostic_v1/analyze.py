"""Saved-output-only test of whether low overlap certifies a reliable layout.

Diagnostic gate at the user's previously proposed10% level, NOT a deployed
classifier or a REAL-fitted threshold. A rejected placement does not imply a
negative pair; an alternative placement may be valid.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent
GATE = .10
RUNS = {
    "s6_C8": D / "continuation_endpoint_v1/raw/attention_depth_20260915/s4_cross_attention_depth2",
    "s6_C16": D / "continuation_endpoint_v1/raw/scorer_diagnosis_20260919/continuation_v1/s6_d2_c16",
    "s7_C8": D / "continuation_s7_endpoint_v1/raw/s6_s7_20260915/priority_after_s5/s7_augmented_full24",
    "s7_C16": D / "continuation_s7_endpoint_v1/raw/scorer_diagnosis_20260919/continuation_v1/s7_c16",
}
SOURCES = {}


def read(path, lines=False):
    SOURCES[str(path.relative_to(D))] = hashlib.sha256(path.read_bytes()).hexdigest()
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines() if line] if lines else json.loads(text)


def overlap_fractions(row):
    layout = row["layouts"]["full_top2_mode"]
    if not layout["valid"]:
        return None
    small = float(layout["overlap_small_fraction"])
    a, b = float(row["area_a_px"]), float(row["area_b_px"])
    assert row["size_metadata_valid"] and min(a, b) > 0
    assert 0 <= small <= 1
    intersection = small * min(a, b)
    # Saved overlap and area metadata both originate from the same input masks.
    assert abs(intersection - round(intersection)) < 1e-5
    return {"intersection_over_smaller": small,
            "intersection_over_union": intersection / (a + b - intersection)}


def subset_result(rows, threshold, denominator, has_layout_gt):
    counts = Counter()
    ids = {key: [] for key in ("accepted_negative_vetoed", "accepted_positive_vetoed",
                               "accepted_correct_layout_vetoed", "accepted_wrong_layout_vetoed")}
    for row in rows:
        label = bool(row["label"])
        accepted = bool(row["decision_valid"] and row["classification"]["fused"] >= threshold)
        layout = row["layouts"]["full_top2_mode"]
        fractions = overlap_fractions(row)
        veto = fractions is not None and fractions[denominator] >= GATE
        kind = "positive" if label else "negative"
        counts["total"] += 1
        counts[kind] += 1
        counts[kind + "_valid_layout"] += bool(layout["valid"])
        counts[kind + "_overlap_ge10"] += veto
        counts[kind + "_layout_valid_overlap_lt10"] += fractions is not None and not veto
        counts["accepted_" + kind] += accepted
        counts["accepted_" + kind + "_vetoed"] += accepted and veto
        if accepted and veto:
            ids["accepted_" + kind + "_vetoed"].append(row["pair_id"])
        if has_layout_gt and label:
            error = layout.get("translation_l2_px")
            correct = bool(layout["valid"] and error is not None and error <= 20)
            cls = "correct" if correct else "wrong"
            counts[cls + "_layout_positive"] += 1
            counts[cls + "_layout_overlap_ge10"] += veto
            counts["accepted_" + cls + "_layout"] += accepted
            counts["accepted_" + cls + "_layout_vetoed"] += accepted and veto
            if accepted and veto:
                ids["accepted_" + cls + "_layout_vetoed"].append(row["pair_id"])
    return dict(counts=dict(counts), affected_pair_ids=ids, threshold=threshold,
                overlap_denominator=denominator, overlap_veto_ge=GATE,
                layout_GT_available=has_layout_gt)


def main():
    results = {}
    for model, root in RUNS.items():
        result = {}
        for split in ("test", "real", "ood"):
            p = root / "evaluation/fixed_epoch" / split
            assert read(p / "protocol.json")["status"] == "complete"
            summary = read(p / "summary.json")
            rows = read(p / "pair_results.jsonl", lines=True)
            assert len(rows) == {"test": 3000, "real": 1016, "ood": 301}[split]
            threshold = summary["model"]["operating_points"]["thresholds"]["max_f1"]
            groups = {"all": rows}
            if split == "real":
                groups["kept_plus_all_negative"] = [r for r in rows if not r["label"] or r["review_status"] == "keep"]
                groups["kept_plus_strict_negative"] = [r for r in rows if (r["label"] and r["review_status"] == "keep") or (not r["label"] and r["strict_member"])]
                assert len(groups["kept_plus_all_negative"]) == 803
                assert len(groups["kept_plus_strict_negative"]) == 334
            result[split] = {group: {den: subset_result(rs, threshold, den, split != "ood")
                                      for den in ("intersection_over_smaller", "intersection_over_union")}
                             for group, rs in groups.items()}
        results[model] = result
    payload = dict(status="complete", executed_at=datetime.now(timezone.utc).isoformat(),
        new_inference=False, new_training=False, threshold_fitted=False,
        source_geometry="saved nearest-integer placement; full un-clipped mask-area denominator",
        proposed_constraint="10% fixed diagnostic, two explicitly distinct denominators",
        note="Veto is a placement-safety diagnostic, never evidence that the pair is truly negative. No reranking/repair was performed.",
        limitations=["Reviewed REAL is exploratory; do not deploy or select thresholds from these outputs.",
                     "Translation<=20px defines GT layout success; low overlap does not establish correct seam.",
                     "OOD contains only positive pairs and has no layout GT.",
                     "C8/C16 share each frozen Matcher; duplicate geometry is not independent evidence."],
        results=results, sources=SOURCES)
    with (HERE / "results.json").open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    for model, result in results.items():
        for den, group in result["real"]["kept_plus_all_negative"].items():
            print(model, den, group["counts"])


if __name__ == "__main__":
    main()
