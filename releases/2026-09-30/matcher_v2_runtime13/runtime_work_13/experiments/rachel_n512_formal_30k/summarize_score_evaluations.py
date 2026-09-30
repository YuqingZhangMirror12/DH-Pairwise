"""Read completed frozen evaluations into JSON/Markdown; never choose a winner.

Examples (no torch, GPU, inference or training is needed):
  python -m experiments.rachel_n512_formal_30k.summarize_score_evaluations \
    --queue-config /path/to/queue/config.json --output reports/readout_001
  python -m experiments.rachel_n512_formal_30k.summarize_score_evaluations \
    --plan expected_evaluations.json --evaluation /path/to/one/evaluation \
    --output reports/readout_002

A plan is a JSON list (or {"evaluations": [...]}) of records containing
evaluation, budget, architecture, selection and split; optional schedule and
changed_axis distinguish joint/staged and input variants. Absent files are
explicitly pending and do not hide completed rows. Existing queues are read,
never executed or modified. All writes are confined to a new output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from experiments.rachel_n512_formal_30k.score_candidate_readout import candidate_diagnostics

SCHEMA = "rachel-score-evaluation-summary/1"
EVALUATORS = {
    "rachel-score-design-evaluation/1": "rachel-score-design-training/1",
    "rachel-score-input-evaluation/1": "rachel-score-input-training/1",
    "rachel-score-staged-evaluation/1": "rachel-score-design-s3-comparison/1",
    "rachel-score-density-evaluation/1": "rachel-score-density-training/1",
}
MODULES = ("evaluate_score_design", "evaluate_score_input_variant", "evaluate_score_staged", "evaluate_score_density")
BRANCHES = ("coarse", "local", "fused")
REAL_GROUPS = ("all", "kept_plus_all_negative", "kept_plus_strict_negative", "negative_strict",
               "negative_constructed", "excluded_positive", "kept_positive")
TEST_GROUPS = ("all", "positive", "negative")
GROUP_LABELS = {"all": "full", "kept_plus_all_negative": "keep + same all negatives",
    "kept_plus_strict_negative": "keep + strict negatives", "negative_strict": "strict negatives only",
    "negative_constructed": "constructed negatives only", "excluded_positive": "review-excluded positives only",
    "kept_positive": "review-kept positives only", "positive": "positives only", "negative": "negatives only"}


class Pending(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Evidence:
    """Hash each frozen artifact once per invocation, without unpickling it."""
    def __init__(self):
        self.hashes = {}

    def sha(self, path):
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.hashes:
            value = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(chunk)
            self.hashes[key] = value.hexdigest()
        return self.hashes[key]


def owned(root, value, name):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if path.resolve(strict=True) != root / name:
        raise ValueError("freeze is not bound to run-owned " + name)
    return path.resolve(strict=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_density_identity(protocol, freeze):
    """Check typed cap and common source identity without importing torch."""
    model, resume = protocol["model"], freeze["resume_identity"]
    metadata = freeze["score_density_metadata"]
    cap = model.get("contour_cap")
    if (type(cap) is not int or cap not in (512, 1024) or freeze.get("contour_cap") != cap
            or resume.get("contour_cap") != cap or model.get("resample_contour_cap") != cap
            or metadata.get("contour_cap") != cap or metadata.get("resample_contour_cap") != cap
            or metadata.get("schema_version") != "rachel-score-density-model/1"
            or model.get("score_density_metadata") != metadata or resume.get("score_density_metadata") != metadata
            or metadata.get("architecture") != model["architecture"] or resume.get("score_design") != model["architecture"]
            or metadata.get("seed") != model["seed"] or resume.get("seed") != model["seed"]
            or resume.get("schema_version") != "rachel-score-density-training/1"
            or metadata.get("training_mode") != "joint" or resume.get("training_mode") != "joint"
            or metadata.get("source_weights_loaded") is not False
            or model.get("original_prepared512_control") is not False or protocol.get("original_prepared512_control") is not False
            or freeze.get("resume_identity_sha256") != digest(resume)):
        raise ValueError("typed density cap/resampling/source identity differs from freeze")
    base = metadata["base_model_metadata"]
    if (base.get("model_kind") != "full" or base.get("model_options") != {}
            or base["model_config"].get("contour_cap") != cap or model.get("model_config") != base["model_config"]):
        raise ValueError("density base model/config differs from typed cap")
    contract = metadata["density_contract"]
    if (contract.get("contour_reextracted_at_both_caps") is not True
            or contract.get("dense_train_source_ancestry_required") is not True
            or contract.get("old_prepared512_is_not_the_paired_control") is not True
            or contract.get("physical_masks_and_coordinate_scale_changed") is not False
            or contract.get("other_input_axes_changed") is not False):
        raise ValueError("density model lost its single-axis/source-target contract")
    common = model["comparison_training_contract"]
    required = {"train_source_selection_sha256", "train_source_selection_file_sha256", "train_source_pipeline",
        "train_original_manifest_sha256", "validation_manifest_sha256", "validation_common_pipeline_sha256",
        "initial_weights_sha256", "loss_config"}
    if (not required <= set(common) or any(resume.get(key) != value for key, value in common.items())
            or model.get("comparison_training_contract_sha256") != digest(common)):
        raise ValueError("density common source/initialization/loss identity differs from freeze")
    val = resume["validation_density_protocol"]
    val_common = {key: value for key, value in val.items() if key not in ("contour_cap", "identity_sha256")}
    if (model.get("validation_density_protocol") != val or val.get("contour_cap") != cap
            or val.get("identity_sha256") != resume.get("validation_density_identity_sha256")
            or common["validation_common_pipeline_sha256"] != digest(val_common)
            or val.get("source_manifest_sha256") != common["validation_manifest_sha256"]
            or val.get("assignment_targets") != "fresh exact clean source-cell ancestry, not old512 token indices"):
        raise ValueError("density source-rebuilt VAL identity differs from freeze")


def validate_freeze(protocol, evidence):
    model = protocol["model"]
    root = Path(model["training_run"]).resolve(strict=True)
    freeze_path = Path(model["freeze_path"]).resolve(strict=True)
    if evidence.sha(freeze_path) != model["freeze_sha256"]:
        raise ValueError("freeze SHA256 differs from evaluation identity")
    freeze = read(freeze_path)
    budget, selection = model["budget"], model["selection"]
    staged = protocol["schema_version"] == "rachel-score-staged-evaluation/1"
    density = protocol["schema_version"] == "rachel-score-density-evaluation/1"
    expected_path = root / "s3_freezes/freeze.json" if staged else root / "budget_freezes" / ("%03d" % budget) / "freeze.json"
    if freeze_path != expected_path:
        raise ValueError("freeze path does not match evaluation budget/schema")
    if (freeze.get("schema_version") != EVALUATORS[protocol["schema_version"]]
            or freeze.get("selection_population") != ("source-rebuilt cleanVAL3000 only" if density else "cleanVAL3000 only")
            or freeze.get("held_out_used_for_fit") is not False
            or freeze.get("budget_epochs") != budget or freeze.get("budget_exposures") != budget * 24000):
        raise ValueError("not a matching SIM-VAL-only freeze")
    if staged:
        if (freeze.get("status") != "frozen_s3_epoch20" or budget != 20
                or freeze.get("eligible_epoch_range") != [13, 20]
                or freeze.get("live_s0_s2_winner_imported") is not False
                or freeze.get("schedule") != model.get("schedule")):
            raise ValueError("S3 fixed20/VAL13..20 identity differs")
        selected = freeze["selections"][selection]
        allowed = [20] if selection == "fixed_epoch" else range(13, 21)
    else:
        if (freeze.get("status") != "frozen_at_budget" or budget not in (5, 10, 20, 30, 50)
                or freeze.get("eligible_epoch_range") != [5, budget]):
            raise ValueError("incomplete or wrong budget freeze")
        selected = freeze["winners"][selection]
        allowed = range(5, budget + 1)
    if (selected.get("selection") != selection or model["epoch"] not in allowed
            or selected.get("selected_epoch") != model["epoch"]
            or selected.get("test_or_real_or_ood_used_for_fit") is not False
            or selected.get("selected_global_exposure") != model["epoch"] * 24000
            or selected != model["winner_record"]):
        raise ValueError("evaluation winner differs from frozen selection")
    for field in ("classifier_thresholds", "operating_points"):
        if selected[field] != model[field]:
            raise ValueError("evaluation " + field + " differs from freeze")
    checkpoint = owned(root, selected["checkpoint"], "epoch_%03d.pt" % model["epoch"])
    if (Path(model["checkpoint_path"]).resolve() != checkpoint
            or selected["checkpoint_sha256"] != model["checkpoint_sha256"]
            or evidence.sha(checkpoint) != model["checkpoint_sha256"]):
        raise ValueError("checkpoint SHA256 differs from frozen evaluation")
    # Old live freezes did not record a VAL-row digest. Do not pretend they did.
    val_bound = "validation_predictions_sha256" in selected
    if density and not val_bound:
        raise ValueError("density freeze requires source-rebuilt VAL row SHA256")
    if val_bound:
        val = owned(root, selected["validation_predictions"], "validation_%03d_rows.json" % model["epoch"])
        if evidence.sha(val) != selected["validation_predictions_sha256"]:
            raise ValueError("frozen VAL predictions changed")
    if protocol["schema_version"] == "rachel-score-input-evaluation/1":
        if (freeze["score_input_metadata"] != model["score_input_metadata"]
                or freeze["input_spec"] != model["input_spec"] or freeze["changed_axis"] != model["changed_axis"]):
            raise ValueError("typed input model identity differs from freeze")
    if density:
        validate_density_identity(protocol, freeze)
    return dict(freeze_path=str(freeze_path), freeze_sha256=model["freeze_sha256"],
        checkpoint_sha256=model["checkpoint_sha256"], validation_rows_sha_bound=val_bound)


def metric_number(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
        raise ValueError("metric must be finite numeric or explicitly unavailable")
    return value


def saved_classification_metrics(metrics, positives, negatives):
    """Derive rates/count aliases from the frozen confusion counts, never scores.

    Single-class outputs retain only their observed half of the matrix; absent
    classes are unknown, not fabricated TN/FP or TP/FN and not binary accuracy.
    """
    mixed = positives > 0 and negatives > 0
    if not mixed and {"accuracy", "auroc", "auprc", "precision", "f1"} & set(metrics):
        raise ValueError("single-class population contains unsupported binary metrics")
    result = dict(metrics)
    counts = {key: None for key in ("tp", "fp", "fn", "tn")}
    if mixed:
        counts.update({key: metrics[key] for key in counts})
    elif positives:
        counts.update(tp=metrics["accepted_positive_count"], fn=metrics["false_negative_count"])
    elif negatives:
        counts.update(fp=metrics["false_positive_count"], tn=metrics["true_negative_count"])
    for value in counts.values():
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("confusion counts must be nonnegative integers")
    if ((positives and counts["tp"] + counts["fn"] != positives)
            or (negatives and counts["fp"] + counts["tn"] != negatives)):
        raise ValueError("confusion counts differ from positive/negative population")
    derived = dict(positive_recall=counts["tp"] / positives if positives else None,
        false_positive_rate=counts["fp"] / negatives if negatives else None,
        accuracy=(counts["tp"] + counts["tn"]) / (positives + negatives) if mixed else None,
        accepted_positive_count=counts["tp"], false_negative_count=counts["fn"],
        false_positive_count=counts["fp"], true_negative_count=counts["tn"],
        accepted_count=sum(counts[key] or 0 for key in ("tp", "fp")),
        rejected_count=sum(counts[key] or 0 for key in ("fn", "tn")))
    for key, value in derived.items():
        if key in metrics and (value is None or not math.isclose(metric_number(metrics[key]), value)):
            raise ValueError("saved rate/count disagrees with frozen confusion matrix: " + key)
    result.update(derived, **counts)
    return result


def source_score_distribution(distribution, count):
    """Copy the evaluator's raw-score quantiles without refitting or regrouping."""
    levels, quantiles = distribution["quantile_levels"], distribution["quantiles"]
    if (not levels or len(levels) != len(quantiles)
            or any(metric_number(x) is None or not 0 <= x <= 1 for x in levels + quantiles)
            or levels != sorted(levels) or quantiles != sorted(quantiles)
            or metric_number(distribution["mean"]) is None or not 0 <= distribution["mean"] <= 1):
        raise ValueError("source score quantiles are malformed")
    for key in ("exact_zero_count", "exact_one_count"):
        if type(distribution[key]) is not int or not 0 <= distribution[key] <= count:
            raise ValueError("source score extreme count exceeds population")
    if distribution["exact_zero_count"] + distribution["exact_one_count"] > count:
        raise ValueError("source score zero/one counts overlap")
    return dict(quantile_levels=list(levels), quantiles=list(quantiles), mean=distribution["mean"],
        exact_zero_count=distribution["exact_zero_count"], exact_one_count=distribution["exact_one_count"])


def population_rows(rows, name):
    """Mirror existing evaluator populations; exclusions never become negatives."""
    if name == "all":
        return rows
    selected = []
    for row in rows:
        positive = bool(row["label"])
        keep = positive and row.get("review_status") == "keep"
        excluded = positive and row.get("review_status") == "exclude"
        strict = not positive and row.get("strict_member") is True
        take = {"positive": positive, "negative": not positive, "kept_positive": keep,
            "excluded_positive": excluded, "negative_strict": strict,
            "negative_constructed": not positive and not strict,
            "kept_plus_all_negative": keep or not positive,
            "kept_plus_strict_negative": keep or strict}[name]
        if take:
            selected.append(row)
    return selected


def flatten_group(group, name, common, model, paired):
    n, positives, negatives = (group[key] for key in ("sample_count", "positive_count", "negative_count"))
    if n != positives + negatives or min(n, positives, negatives) < 0:
        raise ValueError("population counts disagree")
    base = dict(common, population=name, population_label=GROUP_LABELS[name],
                sample_count=n, positive_count=positives, negative_count=negatives)
    classification_rows, layout_rows = [], []
    single_class = positives == 0 or negatives == 0
    scope = "mixed_positive_negative" if not single_class else "positive_only" if positives else "negative_only"
    for branch in BRANCHES:
        distribution = source_score_distribution(group["score_distributions"][branch], n)
        expected = {"max_f1": model["classifier_thresholds"][branch]}
        if branch == "fused":
            expected.update({op: model["operating_points"]["thresholds"][op] for op in ("max_f1", "recall_95")})
        for op, threshold in expected.items():
            metrics = saved_classification_metrics(group["classification"][branch][op], positives, negatives)
            if metrics["threshold"] != threshold or not 0 <= threshold <= 1:
                raise ValueError("summary metric threshold differs from frozen " + branch + "/" + op)
            extreme = group["extreme_score_cases"][branch]
            ids = extreme["positive_score_le_001_pair_ids"]
            if len(ids) != extreme["positive_score_le_001_count"] or len(set(ids)) != len(ids):
                raise ValueError("low-score positive IDs/count disagree")
            if len(ids) > positives:
                raise ValueError("low-score positive count exceeds population positives")
            high_ids = extreme["negative_score_ge_09_pair_ids"]
            if (len(high_ids) != extreme["negative_score_ge_09_count"] or len(set(high_ids)) != len(high_ids)
                    or len(high_ids) > negatives):
                raise ValueError("high-score negative IDs/count disagree with population")
            classification_rows.append(dict(base, branch=branch, operating_point=op, threshold=threshold,
                **{key: metric_number(metrics.get(key)) for key in ("auroc", "auprc", "accuracy", "f1", "precision", "recall",
                    "positive_recall", "false_positive_rate", "accepted_positive_count", "false_negative_count",
                    "false_positive_count", "true_negative_count", "accepted_count", "rejected_count", "tp", "fp", "fn", "tn")},
                low_score_positive_count=len(ids), low_score_positive_pair_ids=ids,
                high_score_negative_count=len(high_ids), high_score_negative_pair_ids=high_ids,
                score_distribution=distribution, score_distribution_scope=scope,
                binary_metrics_available=not single_class))
    if common["split"] == "ood":
        if negatives != 0 or "layout" in group:
            raise ValueError("OOD must be positive-only and have no layout metrics")
        return classification_rows, layout_rows
    for op in ("max_f1", "recall_95"):
        entry = dict(base, operating_point=op, threshold=model["operating_points"]["thresholds"][op])
        pair_group = ("kept_positive" if name.startswith("kept_") else
                      "all_positive" if name == "all" else None)
        for tolerance in (10, 20):
            layout = group["layout"][op][str(tolerance)]
            if layout["positive_count"] != positives or layout["tolerance_px"] != tolerance:
                raise ValueError("layout denominator/tolerance differs from population")
            if (not 0 <= layout["accepted_correct"] <= layout["raw_correct"] <= positives
                    or layout["accepted_correct"] + layout["classification_FN_but_layout_correct"] != layout["raw_correct"]
                    or not 0 <= layout["accepted_negative_count"] <= negatives):
                raise ValueError("layout counts are inconsistent")
            if positives and (not math.isclose(layout["raw_recall"], layout["raw_correct"] / positives)
                    or not math.isclose(layout["end_to_end_positive_recall"], layout["accepted_correct"] / positives)):
                raise ValueError("layout rates disagree with count/positive denominator")
            entry[str(tolerance)] = {key: metric_number(layout[key]) for key in (
                "raw_correct", "raw_recall", "accepted_correct", "end_to_end_positive_recall",
                "classification_FN_but_layout_correct", "accepted_positive_bad_layout", "accepted_negative_count")}
            # Negative-only groups do not inherit positive-cohort comparisons.
            entry[str(tolerance)]["paired_baseline"] = (paired.get(pair_group, {}).get(str(tolerance))
                                                       if positives and pair_group else None)
        layout_rows.append(entry)
    return classification_rows, layout_rows


def collect_entry(expected, evidence):
    root = Path(expected["evaluation"]).resolve()
    result = dict(expected, evaluation=str(root), status="pending", classification_rows=[], layout_rows=[])
    try:
        if not (root / "protocol.json").is_file():
            raise Pending("evaluation protocol not present")
        protocol = read(root / "protocol.json")
        if protocol.get("smoke") or protocol.get("formal_training_counted") is False:
            result.update(status="excluded_smoke", reason="smoke/discard-only result is not an experiment")
            return result
        if protocol.get("status") != "complete":
            result.update(status="failed" if protocol.get("status") == "failed" else "pending",
                          reason="evaluation status: " + str(protocol.get("status")))
            return result
        if protocol.get("schema_version") not in EVALUATORS:
            raise ValueError("unregistered evaluation schema")
        if any(not (root / file).is_file() for file in ("summary.json", "prediction_complete.json", "pair_results.jsonl")):
            raise Pending("completed protocol lacks one or more final result artifacts")
        summary, marker = read(root / "summary.json"), read(root / "prediction_complete.json")
        model = protocol["model"]
        if (summary.get("status") != "complete" or summary.get("model") != model
                or summary.get("split") != protocol["split"]
                or summary.get("selection_on_this_population") is not False
                or summary.get("threshold_fitting_performed") is not False
                or protocol.get("thresholds_fitted") is not False
                or protocol.get("test_or_real_used_for_fit") is not False
                or protocol.get("ood_used_for_fit") is not False):
            raise ValueError("summary identity is incomplete or used held-out fitting")
        if (marker.get("status") != "all_predictions_frozen" or marker.get("sample_count") != protocol["sample_count"]
                or marker.get("checkpoint_sha256") != model["checkpoint_sha256"]
                or marker.get("real_gt_opened") is not False or marker.get("review_labels_opened") is not False):
            raise ValueError("target-blind prediction-complete marker differs")
        common = dict(evaluation=str(root), budget=model["budget"], architecture=model["architecture"],
            selection=model["selection"], split=protocol["split"], selected_epoch=model["epoch"], seed=model["seed"],
            schedule=model.get("schedule", "joint"), changed_axis=model.get("changed_axis", "none"),
            input_spec=model.get("input_spec"), evaluation_schema=protocol["schema_version"])
        if protocol["schema_version"] == "rachel-score-density-evaluation/1":
            common.update(contour_cap=model["contour_cap"], density_label="new N%d" % model["contour_cap"],
                          changed_axis="contour_cap")
        for key in ("budget", "architecture", "selection", "split", "schedule", "changed_axis", "contour_cap"):
            if expected.get(key) is not None and expected[key] != common[key]:
                raise ValueError("expected evaluation differs in " + key)
        provenance = validate_freeze(protocol, evidence)
        seen_ids, rows = set(), []
        with (root / "pair_results.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    pair_id = row["pair_id"]
                    if pair_id in seen_ids:
                        raise ValueError("duplicate pair ID in final evaluation rows")
                    seen_ids.add(pair_id)
                    rows.append(row)
        if len(seen_ids) != protocol["sample_count"]:
            raise ValueError("final evaluation row count differs from prediction-complete marker")
        groups = summary["groups"]
        if groups["all"]["sample_count"] != protocol["sample_count"]:
            raise ValueError("summary population differs from complete prediction count")
        if common["split"] == "real":
            # The same old negatives must remain in the kept cohort: excluded
            # positives are not relabeled as negative or silently removed twice.
            if (groups["kept_plus_all_negative"]["negative_count"] != groups["all"]["negative_count"]
                    or groups["kept_plus_all_negative"]["positive_count"] != groups["kept_plus_strict_negative"]["positive_count"]
                    or groups["negative_strict"]["negative_count"] != groups["kept_plus_strict_negative"]["negative_count"]
                    or groups["negative_strict"]["positive_count"] != 0
                    or groups["negative_constructed"]["positive_count"] != 0
                    or groups["excluded_positive"]["negative_count"] != 0
                    or groups["negative_strict"]["negative_count"] + groups["negative_constructed"]["negative_count"]
                        != groups["all"]["negative_count"]
                    or groups["kept_plus_all_negative"]["positive_count"] + groups["excluded_positive"]["positive_count"]
                        != groups["all"]["positive_count"]
                    or groups["kept_positive"]["positive_count"] != groups["kept_plus_all_negative"]["positive_count"]
                    or groups["kept_positive"]["negative_count"] != 0):
                raise ValueError("REAL kept/negative cohort definitions differ")
        classes, layouts, candidates = [], [], {}
        names = REAL_GROUPS if common["split"] == "real" else TEST_GROUPS if common["split"] == "test" else ("all",)
        for name in names:
            selected_rows = population_rows(rows, name)
            if (len(selected_rows) != groups[name]["sample_count"]
                    or sum(bool(row["label"]) for row in selected_rows) != groups[name]["positive_count"]):
                raise ValueError("saved population counts differ from frozen rows: " + name)
            c, l = flatten_group(groups[name], name, common, model, summary.get("paired_baseline_layout", {}))
            classes.extend(c); layouts.extend(l)
            candidates[name] = candidate_diagnostics(selected_rows, split=common["split"], decoder=protocol["decoder"],
                thresholds={op: model["operating_points"]["thresholds"][op] for op in ("max_f1", "recall_95")})
            if (selected_rows and model["architecture"] in ("candidate_pair", "candidate_dual")
                    and candidates[name]["status"] == "unavailable"):
                raise ValueError("candidate model has no saved candidate details: " + name)
        provenance.update(protocol_sha256=evidence.sha(root / "protocol.json"),
                          summary_sha256=evidence.sha(root / "summary.json"),
                          pair_results_sha256=evidence.sha(root / "pair_results.jsonl"))
        result.update(common, status="complete", provenance=provenance,
            classification_rows=classes, layout_rows=layouts,
            candidate_diagnostics=candidates,
            paired_baseline_layout=summary.get("paired_baseline_layout"))
    except Pending as error:
        result.update(status="pending", reason=str(error))
    except (OSError, ValueError, TypeError, KeyError) as error:
        result.update(status="invalid", reason=str(error), classification_rows=[], layout_rows=[])
    return result


def aggregate(entries):
    evidence, results, seen = Evidence(), [], set()
    for entry in entries:
        key = str(Path(entry["evaluation"]).resolve())
        if key in seen:
            continue
        seen.add(key)
        results.append(collect_entry(entry, evidence))
    # Sort only by design identity, never by held-out performance.
    results.sort(key=lambda r: (r.get("budget") if isinstance(r.get("budget"), int) else 10 ** 9, r.get("architecture") or "",
        r.get("selection") or "", r.get("split") or "", r["evaluation"]))
    counts = dict(Counter(r["status"] for r in results))
    classification_rows = [x for r in results for x in r["classification_rows"]]
    # Distribution is a property of source scores, not a threshold workpoint;
    # retain it once per population/branch instead of duplicating fused OPs.
    distribution_keys = ("evaluation", "budget", "architecture", "selection", "split", "selected_epoch", "seed",
        "schedule", "changed_axis", "input_spec", "evaluation_schema", "contour_cap", "density_label",
        "population", "population_label", "sample_count", "positive_count", "negative_count", "branch",
        "score_distribution", "score_distribution_scope", "low_score_positive_count", "low_score_positive_pair_ids",
        "high_score_negative_count", "high_score_negative_pair_ids")
    distribution_rows = [{key: row[key] for key in distribution_keys if key in row}
        for row in classification_rows if row["operating_point"] == "max_f1"]
    return dict(schema_version=SCHEMA, generated_at=datetime.now(timezone.utc).isoformat(),
        status="complete" if results and all(r["status"] == "complete" for r in results) else "partial",
        counts=counts, evaluations=results, classification_rows=classification_rows,
        score_distribution_rows=distribution_rows,
        layout_rows=[x for r in results for x in r["layout_rows"]], held_out_model_selection_performed=False,
        threshold_fitting_performed=False, smoke_counted=False,
        caveats=["Rows describe frozen checkpoints; no held-out winner is selected.",
            "REAL full and reviewed-kept populations are different estimands; negatives remain explicitly identified.",
            "Source score quantiles describe their named population; mixed positive/negative quantiles are not positive-only scores.",
            "max_f1/recall95 name checkpoint selection; operating_point separately names its frozen decision threshold.",
            "OOD is positive-only: acceptance/positive recall only, no accuracy/P/F1/AUC or layout accuracy."])


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _rate(value):
    return "—" if value is None else "%.2f" % (100 * value)


def _score(value):
    return "—" if value is None else "%.6g" % value


def _label(row):
    axis, spec = row.get("changed_axis", "none"), row.get("input_spec") or {}
    suffix = ("/" + row["density_label"] if row.get("density_label") else
              "" if axis == "none" else "/%s=%s" % (axis, spec.get(axis, "?")))
    return "%s/%s%s/%s/%s" % (row.get("budget", "?"), row.get("architecture", "?"), suffix,
        row.get("schedule", "joint"), row.get("selection", "?"))


def markdown(report):
    lines = ["# Frozen score-design results", "", "Counts: " + ", ".join("%s=%d" % x for x in sorted(report["counts"].items())), "",
        "Selection = checkpoint selection on SIM VAL; OP = its frozen threshold. Rates are %. No held-out ranking or winner selection.", "",
        "## Classification", "", "| Budget/model/schedule/selection | Split · population | N (+/−) | Branch · OP | τ | AUROC / AP | Acc / F1 / P / R | TP / FP / FN / TN | Positive acceptance / negative FPR | Low-positive / high-negative |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["classification_rows"]:
        lines.append("| " + " | ".join(map(_cell, (_label(row), row["split"] + " · " + row["population_label"],
            "%d (%d/%d)" % (row["sample_count"], row["positive_count"], row["negative_count"]),
            row["branch"] + " · " + row["operating_point"], "%.6g" % row["threshold"],
            _rate(row["auroc"]) + " / " + _rate(row["auprc"]),
            " / ".join(_rate(row[k]) for k in ("accuracy", "f1", "precision", "recall")),
            " / ".join("—" if row[k] is None else str(row[k]) for k in ("tp", "fp", "fn", "tn")),
            _rate(row["positive_recall"]) + " / " + _rate(row["false_positive_rate"]),
            "%d / %d" % (row["low_score_positive_count"], row["high_score_negative_count"])))) + " |")
    lines += ["", "Low-positive: source score ≤ 0.01; high-negative: source score ≥ 0.9. Exact IDs are preserved in JSON. Single-class binary metrics and absent-class confusion counts are unavailable (—).", "",
        "## Source score distributions", "", "Raw source probabilities (0–1), independent of OP. Mixed positive/negative distributions must not be interpreted as positive-only scores; full quantile vectors are retained in JSON.", "",
        "| Budget/model/schedule/selection | Split · population | N (+/−) | Branch · scope | Mean | q10 / q50 / q90 | Exact 0 / 1 |",
        "|---|---|---:|---|---:|---:|---:|"]
    for row in report["score_distribution_rows"]:
        distribution = row["score_distribution"]
        quantiles = dict(zip(distribution["quantile_levels"], distribution["quantiles"]))
        values = " / ".join("—" if q not in quantiles else "%.6g" % quantiles[q] for q in (.1, .5, .9))
        lines.append("| " + " | ".join(map(_cell, (_label(row), row["split"] + " · " + row["population_label"],
            "%d (%d/%d)" % (row["sample_count"], row["positive_count"], row["negative_count"]),
            row["branch"] + " · " + row["score_distribution_scope"], "%.6g" % distribution["mean"], values,
            "%d / %d" % (distribution["exact_zero_count"], distribution["exact_one_count"])))) + " |")
    lines += ["",
        "## Layout on positive pairs", "", "| Budget/model/schedule/selection | Split · population | Fused OP | Raw @10 / @20 | End-to-end @10 / @20 | FN but good @10 / @20 | Gained/lost @10 ; @20 |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in report["layout_rows"]:
        if not row["positive_count"]:
            continue
        pairs = []
        for tol in ("10", "20"):
            paired = row[tol]["paired_baseline"]
            pairs.append("—" if paired is None else "%d/%d" % (paired["gained"], paired["lost"]))
        lines.append("| " + " | ".join(map(_cell, (_label(row), row["split"] + " · " + row["population_label"],
            row["operating_point"], " / ".join(_rate(row[t]["raw_recall"]) for t in ("10", "20")),
            " / ".join(_rate(row[t]["end_to_end_positive_recall"]) for t in ("10", "20")),
            " / ".join(str(row[t]["classification_FN_but_layout_correct"]) for t in ("10", "20")),
            " ; ".join(pairs)))) + " |")
    lines += ["", "OOD has no layout ground truth; decoded poses are not scored. Gained/lost is descriptive only; no fallback is applied.", "",
        "## Candidate P/R diagnostics", "", "Post-freeze diagnostics only: oracle correctness uses the fixed 20 px GT tolerance; it never changes pair decisions or reranks layout. R Brier is averaged equally over pairs with valid candidates. R ≥ 0.5 is a diagnostic flag, not proof of correct layout. Original models have no candidate head.", "",
        "| Budget/model/schedule/selection | Split · population | Status · OP | Positive GT N | Oracle / top-R recall | R Brier pair mean | FN but oracle | Negative high-R | OOD max-P / max-R mean |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for record in report["evaluations"]:
        if record["status"] != "complete":
            continue
        for name, diagnostic in record.get("candidate_diagnostics", {}).items():
            prefix = (_label(record), record["split"] + " · " + GROUP_LABELS[name])
            if diagnostic["status"] == "unavailable":
                lines.append("| " + " | ".join(map(_cell, prefix + ("unavailable · no candidate head",) + ("—",) * 6)) + " |")
                continue
            for op in ("max_f1", "recall_95"):
                pose = diagnostic["pose_diagnostics"]
                if pose is None:
                    positive = diagnostic["pair_score_distributions"]["positive"]
                    values = ("no pose GT · " + op, "—", "—", "—", "—", "—",
                        _score(positive["max_p"]["mean"]) + " / " + _score(positive["max_r"]["mean"]))
                else:
                    missed = pose["missed_evidence"][op]
                    values = ("available · " + op, pose["positive_with_gt_count"],
                        _rate(pose["oracle_candidate_recall"]) + " / " + _rate(pose["top_r_correct_positive_recall"]),
                        _score(pose["r_brier_pair_mean"]["mean"]),
                        len(missed["false_negative_but_oracle_correct_pair_ids"]),
                        len(missed["negative_high_r_pair_ids"]), "—")
                lines.append("| " + " | ".join(map(_cell, prefix + values)) + " |")
    lines += ["", "Candidate case IDs, full P/R quantiles and score-only flags are retained in JSON. OOD has no candidate correctness or oracle recall.", "",
              "## Pending / excluded", "", "| Expected evaluation | Status | Reason |", "|---|---|---|"]
    for record in report["evaluations"]:
        if record["status"] != "complete":
            lines.append("| %s | %s | %s |" % (_cell(record["evaluation"]), record["status"], _cell(record.get("reason", ""))))
    lines += ["", "## Sources", ""]
    lines += ["- [%s](%s/summary.json) · freeze `%s`" % (_label(r), r["evaluation"], r["provenance"]["freeze_sha256"])
              for r in report["evaluations"] if r["status"] == "complete"]
    return "\n".join(lines) + "\n"


def queue_entries(path):
    entries = []
    for stage in read(path).get("stages", []):
        command = stage.get("command", [])
        module = next((x.rsplit(".", 1)[-1] for x in command if isinstance(x, str)
                       and x.rsplit(".", 1)[-1] in MODULES), None)
        if module is None:
            continue
        def arg(name, default=None):
            return command[command.index(name) + 1] if name in command else default
        architecture = next((part for part in Path(arg("--training-run")).parts
                             if part in ("original", "candidate_pair", "candidate_dual")), None)
        entries.append(dict(evaluation=arg("--output"), budget=int(arg("--budget", 20)),
            architecture=architecture, selection=arg("--selection"), split=arg("--split")))
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", default=[])
    parser.add_argument("--queue-config", action="append", default=[])
    parser.add_argument("--plan", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    entries = []
    for path in args.plan:
        data = read(path)
        entries.extend(data if isinstance(data, list) else data["evaluations"])
    for path in args.queue_config:
        entries.extend(queue_entries(path))
    entries.extend(dict(evaluation=path) for path in args.evaluation)
    if not entries:
        parser.error("supply at least one evaluation, plan or evaluation queue config")
    report = aggregate(entries)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "RESULTS.md").write_text(markdown(report))
    print(json.dumps(dict(status=report["status"], counts=report["counts"], output=str(output.resolve()))))
    return report


if __name__ == "__main__":
    main()
