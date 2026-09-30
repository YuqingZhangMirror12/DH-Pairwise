"""Paired held-out hard-SIMVAL GT-correspondence geometry diagnostic.

Inherited mutual GT edges are supplied to the unchanged production decoder.
This is NOT a deployment result or a performance upper bound: material loss can
bias zero-gap translation despite correct inherited labels. Every positive clean
and requested-damage row remains, including unchanged fallbacks and invalid
decodes. Clean/damage copies are summarized as paired sources, not independent
observations. No model, Scorer, training, threshold fitting or queue operation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[5]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_support_diagnosis import oracle_geometry as geometry
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence import evaluate_hard_validation as hard
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence.hard_validation import materialize

SCHEMA = "s7-hard-simval-gt-correspondence-geometry/1"


def statistics(values):
    values = np.asarray([v for v in values if v is not None], float)
    return dict(available_count=len(values), mean=float(values.mean()) if len(values) else None,
        median=float(np.median(values)) if len(values) else None,
        p90=float(np.quantile(values, .9)) if len(values) else None)


def summarize_rows(rows):
    """One row per source within either the clean or requested cohort."""
    n = len(rows)
    success = sum(r["oracle_layout20_success"] for r in rows)
    return dict(positive_denominator=n, oracle_layout20_success_count=success,
        oracle_layout20_rate=success/n if n else None,
        oracle_invalid_count=sum(not r["oracle_layout_valid"] for r in rows),
        fewer_than_3_target_edges_count=sum(r["gt_edge_count"] < 3 for r in rows),
        fallback_reasons=dict(Counter(r["fallback_reason"] for r in rows if r["fallback_reason"])),
        metrics={key: statistics([r[key] for r in rows]) for key in
            ("gt_edge_count", "gt_edge_error_median_px", "gt_edge_error_p90_px", "oracle_error_px", "oracle_residual_px")})


def summarize_paired(pairs):
    transitions = Counter()
    differences = []
    for clean, damaged in pairs:
        transitions[("success" if clean["oracle_layout20_success"] else "failure") + "_to_" +
                    ("success" if damaged["oracle_layout20_success"] else "failure")] += 1
        if clean["oracle_layout_valid"] and damaged["oracle_layout_valid"]:
            differences.append(damaged["oracle_error_px"] - clean["oracle_error_px"])
    return dict(positive_source_count=len(pairs), clean=summarize_rows([p[0] for p in pairs]),
        requested_variant=summarize_rows([p[1] for p in pairs]),
        layout20_transitions={key: transitions[key] for key in
            ("success_to_success", "success_to_failure", "failure_to_success", "failure_to_failure")},
        error_delta_px_when_both_valid=statistics(differences),
        comparison="same source clean versus requested variant; unchanged fallbacks retained")


def summarize(rows):
    sources = defaultdict(list)
    for row in rows:
        sources[row["source_pair_id"]].append(row)
    pairs = []
    for source, group in sources.items():
        clean = [r for r in group if r["recipe"] == "clean"]
        variants = [r for r in group if r["recipe"] != "clean"]
        if len(clean) != 1 or len(variants) != 1:
            raise ValueError("one clean and one requested positive row required: " + source)
        pairs.append((clean[0], variants[0]))
    recipe_groups, changed_groups = defaultdict(list), defaultdict(list)
    for pair in pairs:
        variant = pair[1]
        recipe_groups[variant["recipe"]].append(pair)
        changed_groups[variant["recipe"] + "|actual_changed=" + str(variant["actual_changed_pair"]).lower()].append(pair)
    return dict(positive_source_count=len(pairs), evaluated_positive_rows=len(rows),
        paired_all=summarize_paired(pairs),
        by_requested_recipe={key: summarize_paired(group) for key, group in sorted(recipe_groups.items())},
        by_recipe_and_actual_change={key: summarize_paired(group) for key, group in sorted(changed_groups.items())},
        aggregation="source-paired descriptive comparisons; no pooled clean/damage independence or causal claim")


def preflight(args):
    manifest = Path(args.manifest).resolve(strict=True)
    record = json.loads(manifest.read_text())
    protocol = record.get("protocol", {})
    for flag in ("training_eligible", "threshold_fitting", "model_selection", "real_ood_used"):
        if protocol.get(flag) is not False:
            raise ValueError("hard-SIMVAL diagnostic must disable " + flag)
    sources = materialize.source_rows(Path(protocol["source_val_manifest"]).resolve(strict=True))
    hard.validate_manifest(record, sources, allow_pilot=args.allow_pilot)
    root = Path(record["artifact_root"]).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("new diagnostic output required")
    for protected in (root, manifest.parent, Path(protocol["source_val_manifest"]).resolve().parents[1]):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("output must be separate from materialized/source data")
    positives = [e for e in record["entries"] if e["label"]]
    for entry in positives:
        path = (root / entry["artifact_path"]).resolve(strict=True)
        if root not in path.parents or not path.is_file():
            raise ValueError("positive artifact must remain inside materialized root")
    return manifest, record, root, output, positives


def run(args):
    manifest, record, root, output, positives = preflight(args)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = dict(schema=SCHEMA, status="running", manifest=str(manifest),
        manifest_sha256=geometry.sha(manifest), source_status=record["status"],
        source_val_manifest_sha256=record["protocol"]["source_val_manifest_sha256"],
        positive_rows=len(positives), positive_sources=len(positives)//2, completed_count=0,
        decoder_config=asdict(geometry.CONFIG), decoder_source_sha256=geometry.sha(geometry.decoder.__file__),
        oracle_numerical_source_sha256=geometry.sha(geometry.__file__), implementation_sha256=geometry.sha(__file__),
        targets_used_to_construct_correspondence=True, model_forward=False, scorer_used=False,
        GPU_computation=False, training_performed=False, thresholds_fitted=False, queue_modified=False,
        performance_upper_bound_claimed=False, diagnostic_only=True, training_eligible=False,
        limitations=["GT correspondences are used; this is neither deployment nor learned-model performance.",
            "Correct source-arc correspondences can retain missing-material gaps and bias zero-gap translation.",
            "Equal GT weights and one displacement mode are not a performance upper bound.",
            "All requested variants and fallbacks remain; fewer than3 targets and invalid decodes are failures.",
            "Clean and damaged copies share sources and are compared pairwise, never treated as independent."])
    geometry.save(output/"protocol.json", protocol)
    rows, source_gt = [], {}
    try:
        with (output/"rows.jsonl").open("x") as stream:
            for entry in positives:
                sample, report, _ = hard.load_entry(root, entry)
                gt = np.asarray(sample.translation_a_to_b_rc)
                if not sample.translation_valid or not np.isfinite(gt).all():
                    raise ValueError("positive source requires finite preserved GT")
                source = entry["source_pair_id"]
                if source in source_gt and not np.array_equal(source_gt[source], gt):
                    raise ValueError("clean/requested original-frame GT differs")
                source_gt[source] = gt.copy()
                metrics = geometry.oracle_evidence(sample.points_rc_a, sample.points_rc_b,
                    sample.contour_valid_a, sample.contour_valid_b, sample.target_a, sample.target_b, gt)
                row = dict(pair_id=sample.pair_id, source_pair_id=source, recipe=entry["recipe"],
                    assigned_recipe=entry.get("assigned_recipe"), changed_pair=report["changed_pair"],
                    actual_changed_pair=report["changed_pair"],
                    fallback_reason=report.get("fallback_reason"), source_family_overlap=entry.get("source_family_overlap"),
                    gt_translation_a_to_b_rc=gt.tolist(), **metrics)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if len(rows) % 250 == 0:
                    protocol["completed_count"] = len(rows)
                    geometry.save(output/"protocol.json", protocol)
        if geometry.sha(manifest) != protocol["manifest_sha256"]:
            raise ValueError("materialized manifest changed during diagnosis")
        status = "pilot_complete" if record["status"] == "pilot_complete" else "complete"
        result = dict(schema=SCHEMA, status=status, metrics=summarize(rows),
            manifest_sha256=protocol["manifest_sha256"], rows_sha256=geometry.sha(output/"rows.jsonl"),
            diagnostic_only=True, training_eligible=False, performance_upper_bound_claimed=False,
            limitations=protocol["limitations"], elapsed_s=time.monotonic()-started)
        geometry.save(output/"summary.json", result)
        protocol.update(status=status, completed_count=len(rows), elapsed_s=time.monotonic()-started)
        return result
    except BaseException as error:
        protocol.update(status="failed", completed_count=len(rows), error=repr(error))
        raise
    finally:
        geometry.save(output/"protocol.json", protocol)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--allow-pilot", action="store_true")
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps(dict(status=result["status"], **result["metrics"]["paired_all"]["requested_variant"])))
