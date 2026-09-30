"""Case-first numerical analysis of a completed frozen-scorer dilution probe.

No model imports, fitting, inference, plotting or image interpretation. The
experimental unit is a selected pair, not a random deletion or a model run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

SCHEMA = "frozen-scorer-token-dilution/1"
FRACTIONS = (0., .25, .5, .75, 1.)
# Numerical comparison tolerance only, NOT a significance/effect threshold.
EPS = 2e-5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, q):
    """Linear interpolation, equivalent to NumPy's default percentile."""
    values = sorted(float(x) for x in values)
    if not values:
        raise ValueError("empty distribution")
    point = (len(values) - 1) * q
    lo = int(math.floor(point))
    hi = int(math.ceil(point))
    return values[lo] + (point - lo) * (values[hi] - values[lo])


def distribution(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values),
                median=statistics.median(values), q25=quantile(values, .25),
                q75=quantile(values, .75), min=min(values), max=max(values))


def sigmoid(value):
    if value >= 0:
        return 1. / (1. + math.exp(-value))
    e = math.exp(value)
    return e / (1. + e)


def check_point(point, baseline):
    for key in ("logit", "probability"):
        if not math.isfinite(point[key]):
            raise ValueError("nonfinite " + key)
    if abs(sigmoid(point["logit"]) - point["probability"]) > 2e-7:
        raise ValueError("probability is not sigmoid(raw logit)")
    if "delta_logit" in point and abs(
            point["delta_logit"] - (point["logit"] - baseline["logit"])) > 1e-9:
        raise ValueError("delta_logit is not intervention minus baseline")
    if point["no_evidence"] != (point["count_a"] == 0 or point["count_b"] == 0):
        raise ValueError("inconsistent no_evidence")


def summarize_case(row, repeats):
    baseline = row["baseline"]
    check_point(baseline, baseline)
    dose = defaultdict(dict)
    random = {}
    removed = []
    na, nb = baseline["count_a"], baseline["count_b"]
    ia, ib = row["inlier_count_a"], row["inlier_count_b"]
    if not (0 <= ia <= na and 0 <= ib <= nb):
        raise ValueError("inlier counts outside valid token set")
    for point in row["interventions"]:
        check_point(point, baseline)
        kind, rep = point["kind"], point["repeat"]
        if kind == "retain_inliers_add_context":
            fraction = point["noninlier_fraction"]
            if fraction not in FRACTIONS or rep in dose[fraction]:
                raise ValueError("unknown or duplicate dose/repeat")
            dose[fraction][rep] = point
            expected = (ia + math.ceil(fraction * (na - ia)),
                        ib + math.ceil(fraction * (nb - ib)))
            if (point["count_a"], point["count_b"]) != expected:
                raise ValueError("dose count does not match retained non-inlier fraction")
            if (point["retained_inliers_a"], point["retained_inliers_b"]) != (ia, ib):
                raise ValueError("dose removed predicted inliers")
        elif kind in ("remove_inliers", "remove_random_same_count"):
            if (point["count_a"], point["count_b"]) != (na - ia, nb - ib):
                raise ValueError("deletion controls are not same-count on each side")
            if kind == "remove_inliers":
                removed.append(point)
                if point["retained_inliers_a"] or point["retained_inliers_b"]:
                    raise ValueError("remove_inliers retained inliers")
            else:
                if rep in random:
                    raise ValueError("duplicate random repeat")
                random[rep] = point
        else:
            raise ValueError("unknown intervention kind: " + kind)
    if len(removed) != 1 or set(random) != set(range(repeats)):
        raise ValueError("incomplete deletion interventions")
    for fraction in FRACTIONS:
        expected_reps = {0} if fraction in (0., 1.) else set(range(repeats))
        if set(dose[fraction]) != expected_reps:
            raise ValueError("incomplete dose interventions")
    if abs(dose[1.][0]["logit"] - baseline["logit"]) > EPS:
        raise ValueError("100% retained baseline mismatch")
    out = {key: row.get(key) for key in (
        "model", "dataset", "pair_id", "name", "stratum", "label", "layout_valid",
        "layout_error_px", "threshold", "saved_forward_error", "permutation_logit_error")}
    if (row["dataset"] == "ood" or not row["label"]) and row["layout_error_px"] is not None:
        raise ValueError("OOD/negative rows must not be labeled by GT layout error")
    local, removed = dose[0.][0], removed[0]
    out.update(baseline=baseline, inlier_count_a=ia, inlier_count_b=ib,
               inlier_endpoint_fraction_a=ia / na, inlier_endpoint_fraction_b=ib / nb,
               inlier_only=local,
               inlier_only_gain_logit=local["logit"] - baseline["logit"],
               inlier_only_gain_probability=local["probability"] - baseline["probability"],
               baseline_accepted=baseline["probability"] >= row["threshold"],
               inlier_only_accepted=not local["no_evidence"] and local["probability"] >= row["threshold"],
               remove_inliers=removed)
    random_rows = [random[rep] for rep in range(repeats)]
    random_delta = [point["delta_logit"] for point in random_rows]
    out["random_same_count_deletion"] = dict(
        delta_logit=distribution(random_delta),
        delta_probability=distribution(point["probability"] - baseline["probability"] for point in random_rows),
        retained_inliers_a=distribution(point["retained_inliers_a"] for point in random_rows),
        retained_inliers_b=distribution(point["retained_inliers_b"] for point in random_rows))
    out["remove_inlier_minus_mean_random_delta_logit"] = removed["delta_logit"] - statistics.mean(random_delta)
    out["remove_inlier_minus_median_random_delta_logit"] = removed["delta_logit"] - statistics.median(random_delta)
    out["remove_inlier_drop_larger_than_random_draws"] = sum(
        removed["delta_logit"] < value - EPS for value in random_delta)
    out["random_draw_count"] = repeats
    out["dose_curve"] = [dict(
        noninlier_fraction=fraction,
        count_a=points[0]["count_a"], count_b=points[0]["count_b"],
        logit=distribution(point["logit"] for point in points),
        probability=distribution(point["probability"] for point in points),
        delta_logit=distribution(point["delta_logit"] for point in points))
        for fraction in FRACTIONS for points in [list(dose[fraction].values())]]
    # Shared deterministic 0/100% endpoints complete each nested random path.
    paths = [[dose[f][0 if f in (0., 1.) else rep]["logit"] for f in FRACTIONS]
             for rep in range(repeats)]
    nonincreasing = lambda path: all(b <= a + EPS for a, b in zip(path, path[1:]))
    nondecreasing = lambda path: all(b >= a - EPS for a, b in zip(path, path[1:]))
    median_curve = [point["logit"]["median"] for point in out["dose_curve"]]
    out["median_curve_nonincreasing"] = nonincreasing(median_curve)
    out["median_curve_nondecreasing"] = nondecreasing(median_curve)
    out["nested_paths_nonincreasing"] = sum(map(nonincreasing, paths))
    out["nested_paths_nondecreasing"] = sum(map(nondecreasing, paths))
    return out


def aggregate_cases(rows):
    """Each row below contributes one value regardless of its 16 draws."""
    return dict(
        n_cases=len(rows),
        inlier_only_gain_logit=distribution(r["inlier_only_gain_logit"] for r in rows),
        inlier_only_gain_probability=distribution(r["inlier_only_gain_probability"] for r in rows),
        cases_inlier_only_raises_logit=sum(r["inlier_only_gain_logit"] > EPS for r in rows),
        cases_median_curve_nonincreasing=sum(r["median_curve_nonincreasing"] for r in rows),
        remove_inliers_delta_logit=distribution(r["remove_inliers"]["delta_logit"] for r in rows),
        # A distribution ACROSS case-specific random means, not pooled draws.
        case_mean_random_delta_logit=distribution(r["random_same_count_deletion"]["delta_logit"]["mean"] for r in rows),
        remove_minus_case_mean_random=distribution(r["remove_inlier_minus_mean_random_delta_logit"] for r in rows),
        cases_remove_inliers_lowers_logit=sum(r["remove_inliers"]["delta_logit"] < -EPS for r in rows),
        cases_inlier_drop_greater_than_all_random=sum(
            r["remove_inlier_drop_larger_than_random_draws"] == r["random_draw_count"] for r in rows),
        baseline_accept_count=sum(r["baseline_accepted"] for r in rows),
        inlier_only_accept_count=sum(r["inlier_only_accepted"] for r in rows),
        reject_to_accept_count=sum(not r["baseline_accepted"] and r["inlier_only_accepted"] for r in rows),
        accept_to_reject_count=sum(r["baseline_accepted"] and not r["inlier_only_accepted"] for r in rows),
        dose_median_logit_across_cases=[dict(noninlier_fraction=f,
            distribution=distribution(r["dose_curve"][j]["logit"]["median"] for r in rows))
            for j, f in enumerate(FRACTIONS)])


def analyze(results, protocol, selected):
    if protocol.get("schema_version") != SCHEMA or protocol.get("status") != "complete":
        raise ValueError("requires complete known probe schema")
    if protocol.get("parameters_fitted") or protocol.get("thresholds_fitted") or protocol.get("GT_used_for_token_selection"):
        raise ValueError("unexpected fitting/GT intervention selection")
    if tuple(protocol["fractions"]) != FRACTIONS:
        raise ValueError("unexpected fractions")
    selection = {(r["dataset"], r["pair_id"]): r for r in selected}
    if len(selection) != len(selected) or len(selection) != protocol["selected_count"]:
        raise ValueError("duplicate/incomplete fixed selection")
    expected = {(m, *key) for m in protocol["models"] for key in selection}
    actual = {(r["model"], r["dataset"], r["pair_id"]) for r in results}
    if actual != expected or len(actual) != len(results) or len(results) != protocol["completed_count"]:
        raise ValueError("duplicate or incomplete model/pair coverage")
    for row in results:
        reference = selection[(row["dataset"], row["pair_id"])]
        if row["stratum"] != reference["stratum"] or row["label"] != reference["label"]:
            raise ValueError("selection labels/strata mismatch")
    cases = [summarize_case(row, protocol["repeats"]) for row in results]
    groups = defaultdict(list)
    for case in cases:
        groups[(case["model"], case["dataset"], case["stratum"])].append(case)
    strata = [dict(model=model, dataset=dataset, stratum=stratum, **aggregate_cases(rows))
              for (model, dataset, stratum), rows in sorted(groups.items())]
    datasets = [dict(model=model, dataset=dataset,
                    **aggregate_cases([r for r in cases if r["model"] == model and r["dataset"] == dataset]))
                for model in protocol["models"] for dataset in sorted({r["dataset"] for r in selected})]
    return dict(schema_version="case-first-token-dilution-analysis/1",
        unit="selected pair; four checkpoints are paired repeated measurements of the same 40 cases",
        stratum_reference="REAL TP/FN/FP/TN and good/bad use S6-D2's original SIMVAL-maxF1 result, not the displayed model",
        numerical_comparison_tolerance_logit=EPS,
        uncertainty="IQR/ranges describe within-case random subset variation, not population confidence intervals; no significance tests",
        coverage=dict(model_case_measurements=len(cases), unique_selected_cases=len(selected),
            selected_dataset_counts=dict(Counter(r["dataset"] for r in selected)),
            max_saved_forward_error=max(r["saved_forward_error"] for r in results),
            max_permutation_logit_error=max(r["permutation_logit_error"] for r in results)),
        case_summaries=cases, stratum_summaries=strata, selected_dataset_summaries=datasets)


def case_markdown(report):
    lines = ["# Selected token-dilution cases", "",
        "Numerical diagnostic examples, not a population benchmark. Each intermediate dose is the median of 16 nested subset draws; endpoints are deterministic. Δz is intervention minus full-token logit. Random deletion is summarized within each case before comparison. A negative removal contrast means predicted-inlier removal lowers the score more than the mean equal-count random removal.", "",
        "REAL strata are fixed by S6-D2's original SIMVAL-maxF1 classification and GT layout. OOD and negatives have no valid GT-layout accuracy conclusion. All inputs are already globally contextualized tokens, not independently cropped physical fragments.", ""]
    for model in sorted({r["model"] for r in report["case_summaries"]}):
        lines += ["## " + model, "", "| Case / stratum | Full p | I-only p | I+25% / 50% / 75% p | Δz remove I | mean Δz random | contrast | larger drop than random | nonincreasing paths | GT error px |", "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        for r in report["case_summaries"]:
            if r["model"] != model:
                continue
            mids = " / ".join(f"{p['probability']['median']:.4f}" for p in r["dose_curve"][1:4])
            error = "—" if r["layout_error_px"] is None else f"{r['layout_error_px']:.2f}"
            lines.append(f"| {r['name']} / {r['stratum']} | {r['baseline']['probability']:.4f} | {r['inlier_only']['probability']:.4f} | {mids} | {r['remove_inliers']['delta_logit']:.3f} | {r['random_same_count_deletion']['delta_logit']['mean']:.3f} | {r['remove_inlier_minus_mean_random_delta_logit']:.3f} | {r['remove_inlier_drop_larger_than_random_draws']}/{r['random_draw_count']} | {r['nested_paths_nonincreasing']}/{r['random_draw_count']} | {error} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def run(args):
    root, selection_path, output = Path(args.input), Path(args.selection), Path(args.output)
    protocol = json.loads((root / "protocol.json").read_text())
    if sha(root / "results.json") != protocol["results_sha256"]:
        raise ValueError("results hash mismatch")
    if sha(selection_path) != protocol["selection_sha256"]:
        raise ValueError("fixed-selection hash mismatch")
    report = analyze(json.loads((root / "results.json").read_text()), protocol,
                     json.loads(selection_path.read_text()))
    report["provenance"] = dict(results_sha256=sha(root / "results.json"),
        protocol_sha256=sha(root / "protocol.json"), selection_sha256=sha(selection_path),
        analysis_script_sha256=sha(__file__), source_probe_sha256=protocol["source_sha256"],
        input_model_checkpoints=protocol["inputs"], results_path=str(root.resolve() / "results.json"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "CASE_SUMMARY.md").write_text(case_markdown(report))
    print(json.dumps(report["coverage"], ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--input", default=str(here / "token_dilution_v1"))
    parser.add_argument("--selection", default=str(here / "selected_cases.json"))
    parser.add_argument("--output", default=str(here / "token_dilution_v1" / "analysis"))
    run(parser.parse_args())
