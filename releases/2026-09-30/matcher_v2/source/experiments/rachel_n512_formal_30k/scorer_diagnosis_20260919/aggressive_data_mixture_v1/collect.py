"""Collect existing audited metadata; never import training or geometry code.

This does not re-audit pixels. It verifies the existing receipt/hash chain,
projects committed per-example metadata, and summarizes exactly those values.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


COUNTS = {"train": 24000, "cal": 1500, "select": 1500, "test": 3000}
GAP_BINS = [0., 2., 4., 8., 15., 20., 30., 40., 60., 100., None]
METRICS = (
    "additional_crop_fraction", "common_support_before_crop_px",
    "common_support_after_crop_px", "additional_crop_area_fraction",
    "inherited_correspondences", "primary_new_peak_px", "primary_old_peak_px",
    "primary_peak_delta_px", "resolved_point_gap_p10_px",
    "resolved_point_gap_p50_px", "resolved_point_gap_p90_px",
    "unresolved_arc_fraction", "resolved_arc_gap5to35_fraction",
    "resolved_arc_weighted_mean_gap_px",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path, expected=None):
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    require(expected is None or actual == expected, "source SHA differs: " + str(path))
    return json.loads(data), actual


def number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), name + " not numeric")
    value = float(value)
    require(math.isfinite(value), name + " not finite")
    return value


def optional(value, name):
    return None if value is None else number(value, name)


def distribution(values):
    """Equal-example linear quantiles; missing values never become zeros."""
    values = list(values)
    known = sorted(number(x, "distribution value") for x in values if x is not None)
    result = {"n": len(known), "missing": len(values) - len(known), "mean": None, "quantiles": None}
    if known:
        def q(p):
            k = (len(known) - 1) * p
            lo, hi = math.floor(k), math.ceil(k)
            return known[lo] + (known[hi] - known[lo]) * (k - lo)
        result.update(mean=math.fsum(known) / len(known),
                      quantiles={str(p): q(p) for p in (0., .1, .25, .5, .75, .9, 1.)})
    return result


def project(record, receipt, split, slot, group_sha, audit_sha):
    """One committed sample -> one row, preserving the positive-only GT scope."""
    require(record["id"] == receipt["id"] and receipt["status"] == "passed", "row audit identity")
    for key in ("sample_sha256", "proof_sha256"):
        require(record[key] == receipt[key], "row artifact binding: " + key)
    require(type(record["label"]) is bool and type(record["v14_fallback"]) is bool, "boolean labels")
    positive, fallback = record["label"], record["v14_fallback"]
    detail = record["detail"]
    row = dict(pair_id=record["pair_id"], source_pair_id=record["source_pair_id"],
               split=split, slot=slot, label=positive, recipe=record["recipe"],
               stratum=record["source_stratum"], v14_fallback=fallback,
               source_families=record["source_families"],
               planned_cut_side=record["planned_size_class"],
               applied_cut_side=None, endpoint_mode=None,
               group_sha256=group_sha, audit_sha256=audit_sha,
               sample_sha256=record["sample_sha256"],
               model_input_sha256=receipt["model_input_sha256"],
               requested_v17_notches=None if fallback else record["requested_gap_count"],
               gap_arc_length_counts=None, gap_bins=None)
    row.update({key: None for key in METRICS})
    row["inherited_correspondences"] = number(record["inherited_correspondences"], "inherited count")
    if fallback:
        require(receipt.get("baseline_numerical_identity") is True, "fallback identity not audited")
        require(detail["trim"] is None and detail["gap"] is None and receipt["paired_gap"] is None,
                "fallback claims new measurement")
        row["additional_crop_area_fraction"] = 0.
        if positive:
            row["additional_crop_fraction"] = 0.
        row["gap_missing_reason"] = "unchanged_v14_no_new_fixed_partner_measurement"
        return row
    trim = detail["trim"]
    row.update(applied_cut_side=trim["size_class"], endpoint_mode=trim.get("mode") if positive else None,
               additional_crop_area_fraction=number(trim["material_removed_fraction"], "area fraction"))
    if not positive:
        require(detail["gap"] is None and receipt["paired_gap"] is None, "negative has fabricated seam GT")
        row["gap_missing_reason"] = "negative_no_common_seam_gt"
        return row
    before = number(trim["common_length_before_px"], "before crop length")
    after = number(trim["common_length_after_px"], "after crop length")
    require(before > 0 and 0 <= after <= before, "invalid support lengths")
    require(math.isclose(after / before, trim["retained_fraction"], abs_tol=1e-6), "crop ratio mismatch")
    row.update(common_support_before_crop_px=before, common_support_after_crop_px=after,
               additional_crop_fraction=1. - after / before)
    paired = receipt["paired_gap"]
    gap = detail["gap"]
    require(isinstance(paired, dict) and isinstance(gap, dict), "v17 positive gap missing")
    row["primary_new_peak_px"] = optional(paired["new_primary_gap_peak_px"], "new peak")
    row["primary_old_peak_px"] = optional(paired["old_primary_gap_peak_px"], "old peak")
    new, old = row["primary_new_peak_px"], row["primary_old_peak_px"]
    if new is not None and old is not None:
        row["primary_peak_delta_px"] = new - old
    point_q = paired["resolved_arc_gap_p10_p50_p90"]
    require(len(point_q) == 3, "point quantile count")
    for p, v in zip((10, 50, 90), point_q):
        row[f"resolved_point_gap_p{p}_px"] = number(v, "point quantile")
    require(point_q == sorted(point_q), "unordered point quantiles")
    row["unresolved_arc_fraction"] = number(paired["unresolved_fraction"], "unresolved fraction")
    row["resolved_arc_gap5to35_fraction"] = number(paired["arc_fraction_gap5to35"], "gap band fraction")
    require(0 <= row["unresolved_arc_fraction"] <= 1 and
            0 <= row["resolved_arc_gap5to35_fraction"] <= 1, "invalid arc fraction")
    require(math.isclose(1 - row["unresolved_arc_fraction"], gap["ray_resolved_fraction"], abs_tol=1e-5),
            "stored/audited resolved fraction mismatch")
    require(gap["gap_bins"] == GAP_BINS, "incompatible gap histogram bins")
    counts = [number(x, "gap histogram weight") for x in gap["gap_arc_length_counts"]]
    require(len(counts) == len(GAP_BINS) - 1 and min(counts) >= 0 and sum(counts) > 0, "invalid histogram")
    row.update(gap_arc_length_counts=counts, gap_bins=GAP_BINS,
               resolved_arc_weighted_mean_gap_px=number(gap["gap_mean_px"], "weighted mean gap"),
               gap_missing_reason=None)
    return row


def summarize(rows):
    positives = [r for r in rows if r["label"]]
    cells = []
    for split in sorted({r["split"] for r in rows}):
        split_rows = [r for r in positives if r["split"] == split]
        for version in ("all", "v17", "retained_v14"):
            pop = [r for r in split_rows if version == "all" or r["v14_fallback"] == (version == "retained_v14")]
            for recipe in [None] + sorted({r["recipe"] for r in pop}):
                selected = [r for r in pop if recipe is None or r["recipe"] == recipe]
                measured = [r for r in selected if r["gap_arc_length_counts"] is not None]
                hist = [math.fsum(r["gap_arc_length_counts"][i] for r in measured) for i in range(len(GAP_BINS) - 1)]
                weight = math.fsum(hist)
                cells.append(dict(split=split, version=version, recipe=recipe, positive_pairs=len(selected),
                    metrics={key: distribution(r[key] for r in selected) for key in METRICS},
                    v17_gap_histogram=dict(measured_pairs=len(measured), missing_pairs=len(selected) - len(measured),
                        bins=GAP_BINS, bilateral_resolved_arc_weight_px=hist,
                        share=[v / weight for v in hist] if weight else None,
                        pooled_arc_weighted_mean_gap_px=(math.fsum(
                            r["resolved_arc_weighted_mean_gap_px"] * math.fsum(r["gap_arc_length_counts"])
                            for r in measured) / weight) if weight else None)))
    return dict(schema="aggressive-mixture-metadata/1", pairs=len(rows), positive_pairs=len(positives),
        actual_counts=dict(Counter(f'{r["split"]}/' + ('v14' if r['v14_fallback'] else 'v17') for r in rows)),
        cells=cells, methodology=dict(
            scope="Existing committed group metadata and SHA-bound pixel-audit receipts; no new pixel inference",
            unit="One augmented pair, not an independent manuscript; declared source families retained in rows",
            crop="Additional structural crop on original v14 reference support, not final L40 reduction",
            quantiles="Equal-positive-pair linear quantiles; point P10/P50/P90 are summaries within each pair, not pooled quantiles",
            histogram="Only v17 fixed-partner resolved rays; weights sum bilateral original contour arc, not unique physical seam length",
            paired_peak="New and old maxima at frozen partner locations may use different resolved subsets; delta is not a pointwise depth change",
            fallback="Unchanged v14 has zero additional crop but unknown new-protocol gap; never imputed as zero",
            negatives="No common seam GT; never in positive seam/gap summaries",
            limitations="No final L40 measurement, no common-protocol gap distribution for retained v14, no causal attribution of rejection"))


def collect(root, expected_contract, counts=None):
    root = Path(root).resolve()
    counts = COUNTS if counts is None else counts
    contract, contract_sha = read(root / "data_contract.json", expected_contract)
    complete, complete_sha = read(root / "pipeline_complete.json")
    require(complete["status"] == "complete" and complete["data_contract_sha256"] == contract_sha and
            complete["pairs"] == sum(counts.values()), "dataset incomplete")
    audit, audit_sha = read(root / "full_audit.json", contract["aggressive_full_audit"]["sha256"])
    require(audit["status"] == "passed" and audit["failures"] == 0 and
            audit["checked_pairs"] == sum(counts.values()), "full audit scope differs")
    rows, ids, evidence = [], set(), []
    for split, count in counts.items():
        pixel, pixel_sha = read(root / split / "full_pixel_audit.json", audit["split_audits"][split]["sha256"])
        require(pixel["status"] == "passed" and pixel["checked_pairs"] == count, "split audit incomplete")
        require(pixel["manifest_sha256"] == audit["manifest_sha256"][split], "split manifest receipt mismatch")
        require(sha(root / split / (split + ".json")) == pixel["manifest_sha256"], "manifest changed")
        receipts = pixel["group_receipts"]
        require(len(receipts) == count // 2, "group count differs")
        seen_slots = set()
        split_rows = []
        for item in receipts:
            name = Path(item["path"]).name
            slot = int(Path(name).stem)
            require(name == f"{slot:05d}.json" and slot not in seen_slots, "duplicate or invalid slot")
            seen_slots.add(slot)
            receipt, receipt_sha = read(root / split / "audits" / name, item["sha256"])
            require(receipt["status"] == "passed", "group audit failed")
            group, group_sha = read(root / split / "groups" / name, receipt["group_sha256"])
            require(group["status"] == "committed" and group["split"] == split and group["slot"] == slot,
                    "group identity differs")
            require(not group["positive_replaced"], "original pair replaced")
            records = group["records"]
            require(len(records) == len(receipt["rows"]) == 2 and
                    [r["label"] for r in records] == [True, False], "positive/negative group incomplete")
            require(all(r["v14_fallback"] == group["v14_fallback"] for r in records), "partial-group fallback")
            for record, checked in zip(records, receipt["rows"]):
                require(record["pair_id"] not in ids, "duplicate pair ID")
                ids.add(record["pair_id"])
                split_rows.append(project(record, checked, split, slot, group_sha, receipt_sha))
            evidence.append(dict(split=split, slot=slot, group_sha256=group_sha, audit_sha256=receipt_sha))
        require(seen_slots == set(range(count // 2)), "missing planned slots")
        summary = audit["summaries"][split]
        require(sum(r["v14_fallback"] for r in split_rows) == summary["v14_fallback_pairs"], "fallback count differs")
        require(Counter(r["recipe"] for r in split_rows if r["label"]) == summary["recipe_counts"], "recipe count differs")
        require(Counter(r["applied_cut_side"] for r in split_rows if r["label"] and not r["v14_fallback"])
                == summary["cut_side_counts"], "crop-side count differs")
        rows.extend(split_rows)
    summary = summarize(rows)
    summary.update(status="complete", collected_at=datetime.now(timezone.utc).isoformat(),
        dataset_root=str(root), data_contract_sha256=contract_sha, pipeline_complete_sha256=complete_sha,
        full_audit_sha256=audit_sha, source_receipts_sha256=hashlib.sha256(
            json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
        group_receipts=len(evidence), collector_sha256=sha(__file__),
        no_generation=True, no_training_or_inference=True, no_data_or_source_changes=True)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.dataset.resolve(), args.out.resolve()
    require(out != root and root not in out.parents, "output must be outside immutable dataset")
    require(not out.exists(), "output already exists; inspect it, do not repeat collection")
    if hasattr(os, "nice"):
        os.nice(15)
    rows, summary = collect(root, args.expected_contract_sha256)
    out.mkdir(parents=True)
    row_path = out / "rows.jsonl"
    with row_path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary["rows_sha256"] = sha(row_path)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (out / "collection_complete.json").write_text(json.dumps(dict(
        status="complete", rows=len(rows), rows_sha256=sha(row_path),
        summary_sha256=sha(out / "summary.json"), data_contract_sha256=summary["data_contract_sha256"]), indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "pairs", "positive_pairs", "group_receipts", "rows_sha256")}))


if __name__ == "__main__":
    main()
