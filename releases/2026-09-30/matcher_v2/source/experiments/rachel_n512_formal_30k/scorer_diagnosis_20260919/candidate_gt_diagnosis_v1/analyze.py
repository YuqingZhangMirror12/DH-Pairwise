"""Offline diagnostic of saved Top2/capped layout candidates; no model runs.

GT is used ONLY to measure candidate agreement. No candidate is selected by GT,
no predicted layout/score is changed, and no new model performance is reported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


MODELS = ("s4", "s6", "s6_depth4", "s7")
RADII = (10.0, 20.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(a, b):
    return a / b if b else None


def displacement(a_rc, b_rc):
    # production translation_layout.py: delta = b[j] - a[i].
    # A->B point coordinates; placement of B in A uses the opposite sign.
    return (b_rc[0] - a_rc[0], b_rc[1] - a_rc[1])


def measure(row):
    if not row["label"] or not row.get("layout_gt_available"):
        raise ValueError("Only GT-available positive pairs are eligible")
    target = row["target_translation_rc"]
    if len(target) != 2 or not all(math.isfinite(x) for x in target):
        raise ValueError("finite A-to-B RC target required")
    layout = row["layout"]
    indices = layout["candidate_indices"]
    mass = layout["candidate_sinkhorn_mass"]
    pred = layout["inlier_mask"]
    if not len(indices) == len(mass) == len(pred) == layout["candidate_count"]:
        raise ValueError("Saved candidate arrays do not align")
    if sum(pred) != layout["inlier_count"]:
        raise ValueError("Saved inlier count does not align")
    if any(not math.isfinite(w) or w < 0 for w in mass):
        raise ValueError("Candidate mass must be finite/nonnegative")
    if len({tuple(ij) for ij in indices}) != len(indices):
        raise ValueError("Candidate endpoint pairs must be unique")
    dist = []
    for i, j in indices:
        if not row["valid_a"][i] or not row["valid_b"][j]:
            raise ValueError("Candidate includes padded token")
        delta = displacement(row["points_rc_a"][i], row["points_rc_b"][j])
        dist.append(math.dist(delta, target))
    total_mass = math.fsum(mass)
    pred_count = sum(pred)
    pred_mass = math.fsum(w for w, p in zip(mass, pred) if p)
    max_mass = max(mass, default=0.0)
    predicted_endpoint_a = {ij[0] for ij, p in zip(indices, pred) if p}
    predicted_endpoint_b = {ij[1] for ij, p in zip(indices, pred) if p}
    out = {
        "candidate_count": len(indices), "candidate_mass": total_mass,
        "predicted_inlier_count": pred_count, "predicted_inlier_mass": pred_mass,
        "predicted_inlier_mass_normalized_by_max_candidate": ratio(pred_mass, max_mass),
        "candidate_min_gt_distance_px": min(dist, default=None),
        "target_a_to_b_rc": target, "predicted_a_to_b_rc": layout["t_a_to_b_rc"],
        "layout_valid": layout["valid"], "saved_layout_error_px": layout["translation_l2_px"],
        "gt_neighborhoods": {},
    }
    if layout["valid"]:
        error = math.dist(layout["t_a_to_b_rc"], target)
        if not math.isclose(error, layout["translation_l2_px"], rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError("Saved layout GT error inconsistent with A-to-B sign")
        out["predicted_layout_within_20px"] = error <= 20.0
    else:
        out["predicted_layout_within_20px"] = False
    for radius in RADII:
        agree = [d <= radius for d in dist]
        count = sum(agree)
        weight = math.fsum(w for w, g in zip(mass, agree) if g)
        intersection_count = sum(g and p for g, p in zip(agree, pred))
        intersection_mass = math.fsum(w for w, g, p in zip(mass, agree, pred) if g and p)
        endpoint_retained = [ij[0] in predicted_endpoint_a and ij[1] in predicted_endpoint_b
                             for ij in indices]
        retained_count = sum(g and e for g, e in zip(agree, endpoint_retained))
        retained_mass = math.fsum(w for w, g, e in zip(mass, agree, endpoint_retained) if g and e)
        out["gt_neighborhoods"][str(int(radius))] = {
            "radius_px_inclusive": radius,
            "candidate_count": count, "candidate_mass": weight,
            "candidate_mass_normalized_by_max_candidate": ratio(weight, max_mass),
            "fraction_of_all_candidate_count": ratio(count, len(indices)),
            "fraction_of_all_candidate_mass": ratio(weight, total_mass),
            "count_relative_to_predicted_inliers": ratio(count, pred_count),
            "mass_relative_to_predicted_inliers": ratio(weight, pred_mass),
            "predicted_inlier_gt_consistent_count": intersection_count,
            "predicted_inlier_gt_consistent_mass": intersection_mass,
            "predicted_inlier_gt_consistent_count_fraction": ratio(intersection_count, pred_count),
            "predicted_inlier_gt_consistent_mass_fraction": ratio(intersection_mass, pred_mass),
            "distinct_gt_supported_endpoints_a": len({ij[0] for ij, g in zip(indices, agree) if g}),
            "distinct_gt_supported_endpoints_b": len({ij[1] for ij, g in zip(indices, agree) if g}),
            "gt_consistent_edges_with_both_endpoints_in_predicted_token_sets": retained_count,
            "gt_consistent_mass_with_both_endpoints_in_predicted_token_sets": retained_mass,
            "fraction_gt_consistent_mass_with_both_endpoints_retained": ratio(retained_mass, weight),
            "support_bucket": "absent" if count == 0 else "one_or_two" if count < 3 else "at_least_three",
        }
    return out


def run(root, output):
    root, output = Path(root), Path(output)
    selected_path = root / "selected_cases.json"
    selected = json.loads(selected_path.read_text())
    selection = {(r["dataset"], r["pair_id"]): r for r in selected}
    if len(selection) != len(selected) or len(selected) != 40:
        raise ValueError("Expected unchanged unique fixed 40-case selection")
    source_inputs, measured, excluded = [], [], []
    for model in MODELS:
        path = root / "heatmaps_v1" / model / "cases.json"
        cases = json.loads(path.read_text())
        keys = [(r["dataset"], r["pair_id"]) for r in cases]
        if len(keys) != 40 or set(keys) != set(selection):
            raise ValueError("Model must cover exact frozen 40 selected cases")
        source_inputs.append({"model": model, "path": str(path.resolve()), "sha256": sha(path)})
        for row in cases:
            ref = selection[(row["dataset"], row["pair_id"])]
            if row["label"] != ref["label"]:
                raise ValueError("Label differs from frozen selection")
            identity = {"model": model, "dataset": row["dataset"], "pair_id": row["pair_id"],
                        "name": ref["name"], "reference_s6_stratum": ref["stratum"]}
            if not row["label"] or not row.get("layout_gt_available"):
                excluded.append({**identity, "reason": "negative_pair" if not row["label"] else "no_layout_GT"})
                continue
            cfg = row["layout"]["decoder_config"]
            if not (cfg["correspondence_mode"] == "topk_union" and cfg["top_k"] == 2
                    and cfg["max_candidates"] == 512 and cfg["inlier_radius_px"] == 10):
                raise ValueError("Unexpected source candidate/decoder setting")
            measured.append({**identity, "source_score": row["score"], "metrics": measure(row)})
    groups = []
    for model in MODELS:
        for stratum in sorted({r["reference_s6_stratum"] for r in measured}):
            members = [r for r in measured if r["model"] == model and r["reference_s6_stratum"] == stratum]
            groups.append({"model": model, "reference_s6_stratum": stratum, "count": len(members),
                "saved_layout_within20_count": sum(r["metrics"]["predicted_layout_within_20px"] for r in members),
                "gt_support_buckets": {str(int(rad)): dict(Counter(r["metrics"]["gt_neighborhoods"][str(int(rad))]["support_bucket"] for r in members)) for rad in RADII}})
    decoder = root.parents[2] / "staging/pairwise_v0_2/models/translation_layout.py"
    protocol = {
        "schema_version": "saved-candidate-GT-diagnosis/1", "status": "complete",
        "inputs": source_inputs, "selection_path": str(selected_path.resolve()),
        "selection_sha256": sha(selected_path), "analysis_source_sha256": sha(__file__),
        "production_decoder_path": str(decoder.resolve()), "production_decoder_sha256": sha(decoder),
        "sign": "candidate displacement = point_b_rc - point_a_rc; target is A-to-B, not canvas placement",
        "distance": "Euclidean L2 in saved points_rc input canvas pixels; radius boundary included (<=)",
        "raw_mass": "sum saved Sinkhorn assignment values; decoder internally divides all candidate weights by max",
        "predicted_support": "saved final inlier_mask, radius10 around saved predicted estimate",
        "gt_support": "saved candidate displacements within fixed radius10/20 of supplied GT; not a re-decoded mode",
        "not_point_level_GT": "GT-consistent displacement is not a labeled true seam correspondence, especially with corrosion/gaps",
        "token_set_caveat": "selected endpoint sets can recombine edges; zero GT-consistent predicted inlier edges does not prove all GT-consistent endpoints absent",
        "three_candidates_is_not_proof_of_recoverable_mode": True,
        "saved_candidate_scope": "row/column Top2 union, positive mass, then global capped512; not full assignment",
        "stratum_reference": "original S6-D2 reference strata; not model-specific TP/FN/FP/TN classifications",
        "GT_used_for_production_prediction": False, "parameters_or_thresholds_fitted": False,
        "models_run": False, "new_performance_estimated": False,
        "coverage": {"fixed_unique_cases": len(selected), "model_case_measurements_available": 160,
                     "eligible_model_cases": len(measured), "excluded_model_cases": len(excluded),
                     "eligible_unique_pairs": len({(r['dataset'], r['pair_id']) for r in measured})},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps({"protocol": protocol, "cases": measured,
        "stratum_summaries": groups, "excluded": excluded}, ensure_ascii=False, indent=2) + "\n")
    (output / "CASE_METRICS.md").write_text(render(measured), encoding="utf-8")
    return protocol


def render(rows):
    lines = ["# Saved Top2 candidates vs GT (offline diagnostic)", "",
        "固定 40 对 ×4 模型；仅 16 对有 GT 的正例进入本表（敦煌12、SIM TEST4），共64条。",
        "分层名称固定参考原 S6-D2，不是其他模型自身的分类结果。GT只测量，不重选预测、不报告oracle性能。",
        "质量为原 Sinkhorn mass；G/P 为GT邻域质量 ÷ 原预测内点质量。20px邻域较10px更宽，不是公平的同半径模式竞争。",
        "预测内点GT%按对应条数；更多质量/独立端点指标见 results.json。", ""]
    for model in MODELS:
        lines += ["## " + model, "", "| Case / reference stratum | Error px | Pred count / mass | GT10 count / mass | GT20 count / mass | G10/P mass | G20/P mass | Pred inlier GT10% / GT20% |", "|---|---:|---|---|---|---:|---:|---|"]
        for row in (r for r in rows if r["model"] == model):
            m = row["metrics"]; a, b = (m["gt_neighborhoods"][x] for x in ("10", "20"))
            def f(v, fmt=".3f"):
                return "NA" if v is None else format(v, fmt)
            lines.append(f"| {row['name']} / {row['reference_s6_stratum']} | {f(m['saved_layout_error_px'], '.2f')} | {m['predicted_inlier_count']} / {f(m['predicted_inlier_mass'])} | {a['candidate_count']} / {f(a['candidate_mass'])} | {b['candidate_count']} / {f(b['candidate_mass'])} | {f(a['mass_relative_to_predicted_inliers'])} | {f(b['mass_relative_to_predicted_inliers'])} | {f(a['predicted_inlier_gt_consistent_count_fraction'], '.1%')} / {f(b['predicted_inlier_gt_consistent_count_fraction'], '.1%')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=here.parent)
    parser.add_argument("--output", type=Path, default=here)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output)["coverage"], indent=2))
