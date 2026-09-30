"""Case-first projection of fixed-CA pooling contrasts; no model execution."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

try:
    from . import analyze_token_dilution as common
except ImportError:
    import analyze_token_dilution as common

VARIANTS = (
    "full_CA__local_attentive_global_max",
    "full_CA__global_attentive_local_max",
    "full_CA__local_both_pool",
    "local_CA__local_both_pool",
)


def project_case(row, repeats):
    if row.get("no_inlier_evidence") or row["attribution_is_additive"] or row["recomputes_matcher"]:
        raise ValueError("unexpected empty-inlier/additive/recomputed-Matcher case")
    values = defaultdict(dict)
    for point in row["interventions"]:
        population, variant = point["kind"].split("__", 1)
        key = population, variant
        if variant not in VARIANTS or point["repeat"] in values[key]:
            raise ValueError("unexpected variant or duplicate repeat")
        if not math.isfinite(point["logit"]) or abs(common.sigmoid(point["logit"]) - point["probability"]) > 2e-7:
            raise ValueError("invalid logit/probability")
        # Stored subtraction was performed in float32 on the source device.
        if abs(point["delta_logit"] - (point["logit"] - row["baseline"]["logit"])) > 2e-6:
            raise ValueError("wrong delta direction")
        if "count_a" in point and (point["count_a"], point["count_b"]) != (row["inlier_count_a"], row["inlier_count_b"]):
            raise ValueError("unmatched random/local token counts")
        values[key][point["repeat"]] = point
    expected = {(population, variant) for population in ("inliers", "random_same_count") for variant in VARIANTS}
    if set(values) != expected:
        raise ValueError("incomplete intervention population")
    for (population, variant), points in values.items():
        if set(points) != ({0} if population == "inliers" else set(range(repeats))):
            raise ValueError("incomplete repeats")
    out = {k: row[k] for k in ("model", "dataset", "pair_id", "name", "stratum", "label",
        "layout_error_px", "threshold", "baseline", "inlier_count_a", "inlier_count_b")}
    out["variants"] = {}
    for variant in VARIANTS:
        measured = values[("inliers", variant)][0]
        random = list(values[("random_same_count", variant)].values())
        dist = common.distribution(point["delta_logit"] for point in random)
        out["variants"][variant] = dict(inliers=measured, random_delta_logit=dist,
            inlier_minus_mean_random_delta_logit=measured["delta_logit"] - dist["mean"],
            accepted_at_original_threshold=measured["probability"] >= row["threshold"])
    zs = [values[("inliers", v)][0]["logit"] for v in VARIANTS]
    out["conditional_local_CA_minus_fixed_CA_local_pool_logit"] = zs[3] - zs[2]
    out["pool_factor_nonadditivity_logit"] = zs[2] - zs[0] - zs[1] + row["baseline"]["logit"]
    return out


def run(args):
    root = Path(args.input)
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["schema_version"] != "frozen-scorer-pooling-factor/1" or protocol["status"] != "complete":
        raise ValueError("incomplete/unknown pooling-factor schema")
    if common.sha(root / "results.json") != protocol["results_sha256"]:
        raise ValueError("results hash mismatch")
    selection_path = root.parent / "selected_cases.json"
    if common.sha(selection_path) != protocol["selection_sha256"]:
        raise ValueError("fixed selection hash mismatch")
    selection = json.loads(selection_path.read_text())
    raw = json.loads((root / "results.json").read_text())
    key = lambda row: (row["model"], row["dataset"], row["pair_id"])
    expected = {(m, s["dataset"], s["pair_id"]) for m in protocol["models"] for s in selection}
    if len(raw) != len(expected) or {key(r) for r in raw} != expected:
        raise ValueError("duplicate/incomplete model-case coverage")
    projected = [project_case(r, protocol["repeats"]) for r in raw]
    groups = defaultdict(list)
    for row in projected:
        groups[(row["model"], row["stratum"])].append(row)
    grouped = []
    for (model, stratum), rows in sorted(groups.items()):
        grouped.append(dict(model=model, stratum=stratum, n_cases=len(rows),
            variants={v: dict(
                case_delta_logit=common.distribution(r["variants"][v]["inliers"]["delta_logit"] for r in rows),
                case_contrast_to_random_mean=common.distribution(r["variants"][v]["inlier_minus_mean_random_delta_logit"] for r in rows),
                accepted_count=sum(r["variants"][v]["accepted_at_original_threshold"] for r in rows)) for v in VARIANTS},
            conditional_CA_change=common.distribution(r["conditional_local_CA_minus_fixed_CA_local_pool_logit"] for r in rows)))
    prior_path = root.parent / "token_dilution_v1" / "results.json"
    prior = {key(r): r for r in json.loads(prior_path.read_text())}
    cross_errors = []
    for row in projected:
        local = next(i for i in prior[key(row)]["interventions"] if i.get("noninlier_fraction") == 0.)
        cross_errors.append(abs(local["logit"] - row["variants"][VARIANTS[-1]]["inliers"]["logit"]))
    if max(cross_errors) > 2e-5:
        raise ValueError("local-CA contrast disagrees with previous dilution probe")
    result = dict(schema_version="case-first-pooling-factor-projection/1", case_summaries=projected,
        stratum_summaries=grouped, unique_cases=len(selection), model_case_count=len(raw),
        caveat="Conditional head interventions, not additive causal attribution; 16 draws summarized within each case. REAL strata reference original S6-D2. Fixed Matcher context contains global information. OOD/negatives have no GT-layout conclusion.",
        max_factor_replay_logit_error=max(r["factor_replay_logit_error"] for r in raw),
        max_prior_probe_local_CA_logit_error=max(cross_errors),
        provenance=dict(results_sha256=common.sha(root / "results.json"),
            protocol_sha256=common.sha(root / "protocol.json"), selection_sha256=common.sha(selection_path),
            prior_results_sha256=common.sha(prior_path), script_sha256=common.sha(__file__),
            shared_statistics_sha256=common.sha(common.__file__)))
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("unique_cases", "model_case_count", "max_factor_replay_logit_error", "max_prior_probe_local_CA_logit_error")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(Path(__file__).resolve().parent / "pooling_factor_v1"))
    run(p.parse_args())
