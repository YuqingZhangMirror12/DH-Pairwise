"""Local-only receipt collector for new S3/S4/S5, not historical staged S3.

Run with --root /local/copy/new_s345_20260914 --output /new/readout/directory.
Only JSON/JSONL are read; no torch, model inference, queue execution or metric
fitting. Remote paths in receipts remain provenance, never network requests.
Checkpoint hashes are copied from the evaluator's typed verification, not
recomputed by this lightweight collector. A copied local classifier freeze is
cross-checked when available; its absence is reported, not hidden.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from experiments.rachel_n512_formal_30k.summarize_score_evaluations import (
    Evidence, digest, metric_number, read, saved_classification_metrics,
    source_score_distribution,
)

SCHEMA = "rachel-score-decoupled-summary/1"
EVALUATION_SCHEMA = "rachel-score-decoupled-evaluation/1"
TRAINING_SCHEMA = "rachel-score-decoupled-training/1"
CONVERSION_SCHEMA = "rachel-decoupled-per-pair-norm-revalidation/1"
CONVERTED_ARM = "s3_matrix_per_pair_norm_v3"
ARMS = {
    "s3_matrix": ("matrix_cnn", "original512"),
    "s4_cross_attention": ("cross_attention", "original512"),
    "s5_control512": ("matrix_cnn", "paired512"),
    "s5_step3_cap2048": ("matrix_cnn", "step3"),
    CONVERTED_ARM: ("matrix_cnn", "original512"),
}
DEFAULT_ARMS = tuple(arm for arm in ARMS if arm != CONVERTED_ARM)
SELECTIONS = ("fixed_epoch", "max_f1", "recall95")
SPLITS = ("test", "real", "ood")
OPS = ("max_f1", "recall_95")
REAL_GROUPS = ("all", "kept_positive", "excluded_positive", "negative_all", "negative_strict",
    "negative_constructed", "kept_plus_all_negative", "kept_plus_strict_negative",
    "kept_plus_constructed_negative")
METRICS = ("auroc", "auprc", "accuracy", "f1", "precision", "recall", "positive_recall",
    "false_positive_rate", "accepted_positive_count", "false_negative_count", "false_positive_count",
    "true_negative_count", "accepted_count", "rejected_count", "tp", "fp", "fn", "tn")
LAYOUT_METRICS = ("raw_correct", "raw_recall", "accepted_correct", "end_to_end_positive_recall",
    "classification_FN_but_layout_correct", "accepted_positive_bad_layout", "accepted_negative_count")


def discover(root, arms=None):
    root = Path(root).resolve()
    if arms is None:
        arms = DEFAULT_ARMS + ((CONVERTED_ARM,) if (root / CONVERTED_ARM).is_dir() else ())
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("arms must be nonempty, unique registered arm names")
    return [dict(arm=arm, architecture=head, sampling=sampling, selection=selection, split=split,
        evaluation=str(root / arm / "evaluation" / selection / split),
        local_training_run=str(root / arm / "training"))
        for arm in arms for head, sampling in (ARMS[arm],) for selection in SELECTIONS for split in SPLITS]


def _mode(value):
    return value if isinstance(value, str) else value["mode"]


def _sha_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def training_budget(identity, epoch):
    """Account unique scheduled exposures, not measured wall time/retry cost."""
    result = dict(schedule=identity.get("schedule"), matcher_epochs=identity.get("matcher_epochs"),
        classifier_epochs=identity.get("classifier_epochs"), train_count=identity.get("train_count"),
        budget_epochs=identity.get("max_epochs"), selected_phase="classifier", selected_epoch=epoch,
        logical_budget_pair_exposures=None, selected_logical_pair_exposures=None,
        inherited_matcher_pair_exposures=None, arm_executed_budget_pair_exposures=None,
        arm_executed_selected_pair_exposures=None, matcher_checkpoint_sha256=identity.get("matcher_checkpoint_sha256"),
        accounting_basis="unavailable: incomplete training schedule metadata",
        measured_total_compute_or_wall_time=None)
    if not all(type(identity.get(k)) is int for k in ("train_count", "max_epochs", "matcher_epochs", "classifier_epochs")):
        return result
    n, budget, matcher, classifier = (identity[k] for k in ("train_count", "max_epochs", "matcher_epochs", "classifier_epochs"))
    if n <= 0 or (budget, matcher, classifier) != (20, 12, 8):
        raise ValueError("training metadata does not describe M12+C8 budget20")
    result.update(logical_budget_pair_exposures=n * budget, selected_logical_pair_exposures=n * epoch)
    if "matcher_checkpoint_sha256" in identity:
        reused = identity["matcher_checkpoint_sha256"] is not None
        if reused and not _sha_string(identity["matcher_checkpoint_sha256"]):
            raise ValueError("invalid reused matcher checkpoint digest")
        inherited = n * matcher if reused else 0
        result.update(inherited_matcher_pair_exposures=inherited,
            arm_executed_budget_pair_exposures=n * budget - inherited,
            arm_executed_selected_pair_exposures=n * epoch - inherited,
            accounting_basis="registered schedule and explicit matcher reuse identity; excludes smoke/retries")
    if identity.get("inference_only") is True:
        if (identity.get("matrix_head_revision") != "per_pair_norm_v3"
                or type(identity.get("new_training_pair_exposures")) is not int
                or identity["new_training_pair_exposures"] != 0
                or type(identity.get("new_optimizer_updates")) is not int
                or identity["new_optimizer_updates"] != 0):
            raise ValueError("inference-only conversion must have zero new optimization")
        # Preserve the source trajectory's costs, but never charge its C8 a
        # second time as optimization performed by the converted arm.
        source_cost = dict(result)
        result.update(source_training_budget=source_cost,
            execution_mode="inference_only_normalization_conversion",
            inherited_full_training_pair_exposures=n * budget,
            inherited_selected_training_pair_exposures=n * epoch,
            arm_executed_budget_pair_exposures=0, arm_executed_selected_pair_exposures=0,
            new_training_pair_exposures=0, new_optimizer_updates=0,
            accounting_basis="inherited source M12+C8 weights; zero new optimization; fresh SIM-VAL inference is not training")
    return result


def validate_conversion(expected, model):
    """Validate typed conversion provenance without reloading source weights."""
    identity = model["training_identity"]
    converted = expected["arm"] == CONVERTED_ARM
    if not converted:
        if (identity.get("inference_only") or "normalization_conversion" in identity
                or (expected["arm"] == "s3_matrix" and identity.get("matrix_head_revision") == "per_pair_norm_v3")):
            raise ValueError("converted S3 v3 must remain a separate registered arm")
        return
    conversion = identity.get("normalization_conversion", {})
    design = model.get("model_design")
    if (not isinstance(conversion, dict) or not isinstance(design, dict)
            or identity.get("matrix_head_revision") != "per_pair_norm_v3"
            or design.get("matrix_head_revision") != "per_pair_norm_v3"
            or identity.get("inference_only") is not True
            or any(type(identity.get(key)) is not int or identity[key] != count
                for key, count in (("train_count", 24000), ("max_epochs", 20), ("matcher_epochs", 12), ("classifier_epochs", 8)))
            or conversion.get("schema_version") != CONVERSION_SCHEMA
            or conversion.get("source_revision") != "bn_relu_pool_v2"
            or conversion.get("target_revision") != "per_pair_norm_v3"
            or conversion.get("inference_only") is not True
            or conversion.get("inherited_training_epochs") != [13, 20]
            or conversion.get("source_thresholds_or_winners_reused") is not False
            or conversion.get("fresh_selection_population") != "clean SIM VAL3000 only"
            or conversion.get("held_out_used_for_fit") is not False
            or conversion.get("GT_layout_used_for_selection") is not False
            or any(type(item.get(key)) is not int or item[key] != 0
                for item in (identity, conversion) for key in ("new_training_pair_exposures", "new_optimizer_updates"))):
        raise ValueError("S3 v3 lacks the registered inference-only zero-optimization conversion contract")
    source = conversion.get("source_training_run")
    checkpoints = conversion.get("source_checkpoints", {})
    if (not isinstance(source, str) or not source or not Path(source).is_absolute()
            or Path(source) == Path(model["training_run"])
            or not all(_sha_string(conversion.get(k)) for k in ("source_freeze_sha256", "source_resume_identity_sha256"))
            or not isinstance(checkpoints, dict)
            or set(checkpoints) != {str(epoch) for epoch in range(13, 21)}
            or any(not isinstance(checkpoints[str(epoch)], dict)
                or checkpoints[str(epoch)].get("path") != str(Path(source) / ("epoch_%03d.pt" % epoch))
                or not _sha_string(checkpoints[str(epoch)].get("sha256")) for epoch in range(13, 21))):
        raise ValueError("S3 v3 source checkpoint/freeze provenance is incomplete or not independently owned")


def validate_model(expected, model, evidence):
    epoch, identity, selected = model["epoch"], model["training_identity"], model["winner_record"]
    selection = expected["selection"]
    if (model["budget"] != 20 or model["selection"] != selection or type(epoch) is not int
            or epoch not in (range(20, 21) if selection == "fixed_epoch" else range(13, 21))
            or model["architecture"] != expected["architecture"] or _mode(model["sampling"]) != expected["sampling"]
            or identity.get("schema_version") != TRAINING_SCHEMA
            or identity.get("head_kind") != model["architecture"] or _mode(identity["sampling"]) != expected["sampling"]
            or identity.get("seed") != model["seed"]
            or identity.get("held_out_used_for_training_or_selection") is not False
            or model.get("classifier_only_pair_bce") is not True
            or model.get("coarse_is_untrained_diagnostic") is not True
            or model.get("local_and_fused_are_same_single_classifier") is not True
            or model.get("test_or_real_used_for_fit") is not False or model.get("ood_used_for_fit") is not False):
        raise ValueError("evaluation model differs from registered decoupled arm/selection")
    validate_conversion(expected, model)
    if (selected.get("selection") != selection or selected.get("selected_epoch") != epoch
            or selected.get("selected_global_exposure") != epoch * 24000
            or selected.get("test_or_real_or_ood_used_for_fit") is not False
            or selected.get("checkpoint") != model["checkpoint_path"]
            or selected.get("checkpoint_sha256") != model["checkpoint_sha256"]
            or selected.get("classifier_thresholds") != model["classifier_thresholds"]
            or selected.get("operating_points") != model["operating_points"]):
        raise ValueError("winner/threshold identity differs from evaluation")
    recorded_root = Path(model["training_run"])
    if (Path(model["checkpoint_path"]) != recorded_root / ("epoch_%03d.pt" % epoch)
            or Path(model["freeze_path"]) != recorded_root / "classifier_freezes/freeze.json"
            or not all(_sha_string(model[k]) for k in ("checkpoint_sha256", "freeze_sha256"))):
        raise ValueError("recorded checkpoint/freeze is not owned by its training run")
    primary_op = "recall_95" if selection == "recall95" else "max_f1"
    if selected.get("primary_pair_threshold") != model["operating_points"]["thresholds"][primary_op]:
        raise ValueError("primary decision threshold differs from checkpoint selection")
    if model["classifier_thresholds"]["fused"] != model["operating_points"]["thresholds"]["max_f1"]:
        raise ValueError("single classifier max-F1 threshold conflicts")
    local_freeze = Path(expected["local_training_run"]) / "classifier_freezes/freeze.json"
    provenance = dict(checkpoint_path=model["checkpoint_path"], checkpoint_sha256=model["checkpoint_sha256"],
        freeze_path=model["freeze_path"], freeze_sha256=model["freeze_sha256"],
        checkpoint_weights_reloaded_or_rehashed=False, local_freeze_status="not_copied",
        resume_identity_sha256=digest(identity))
    if local_freeze.exists():
        freeze = read(local_freeze)
        if (evidence.sha(local_freeze) != model["freeze_sha256"]
                or freeze.get("schema_version") != TRAINING_SCHEMA or freeze.get("status") != "complete"
                or freeze.get("eligible_epoch_range") != [13, 20]
                or freeze.get("selection_population") != "clean SIM VAL3000 only"
                or freeze.get("held_out_used_for_fit") is not False
                or freeze.get("resume_identity_sha256") != digest(identity)
                or digest(freeze.get("resume_identity")) != digest(identity)
                or freeze["selections"][selection] != selected):
            raise ValueError("copied local classifier freeze differs from evaluation receipt")
        provenance.update(local_freeze_status="verified", local_freeze_path=str(local_freeze))
    return provenance


def _population(rows, name):
    if name == "all":
        return rows
    def included(row):
        positive = bool(row["label"])
        kept = positive and row.get("review_status") == "keep"
        strict = not positive and row.get("strict_member") is True
        constructed = not positive and not strict
        return {"positive": positive, "negative": not positive, "kept_positive": kept,
            "excluded_positive": positive and row.get("review_status") == "exclude",
            "negative_all": not positive, "negative_strict": strict, "negative_constructed": constructed,
            "kept_plus_all_negative": kept or not positive,
            "kept_plus_strict_negative": kept or strict,
            "kept_plus_constructed_negative": kept or constructed}[name]
    return [row for row in rows if included(row)]


def flatten_group(group, name, rows, common, model):
    n = group["sample_count"]
    if type(n) is not int or n < 0 or n != len(rows):
        raise ValueError("summary cohort count differs from frozen pair rows")
    cohort = dict(common, population=name, sample_count=n,
        positive_count=group.get("positive_count"), negative_count=group.get("negative_count"))
    if not n:
        return cohort, [], []
    positives, negatives = group["positive_count"], group["negative_count"]
    if (positives != sum(bool(r["label"]) for r in rows) or positives + negatives != n
            or any(set(group[field]) != {"fused"} for field in ("classification", "score_distributions", "extreme_score_cases"))):
        raise ValueError("cohort labels or single-classifier branches differ")
    if common["split"] == "ood" and (negatives or "layout" in group
            or any(r.get("target_translation_rc") is not None or r.get("layout_gt_available") is not False for r in rows)):
        raise ValueError("OOD must remain positive-only without pose GT/layout metrics")
    distribution = source_score_distribution(group["score_distributions"]["fused"], n)
    classes, layouts = [], []
    for op in OPS:
        threshold = model["operating_points"]["thresholds"][op]
        metric_number(threshold)
        metrics = saved_classification_metrics(group["classification"]["fused"][op], positives, negatives)
        if metrics["threshold"] != threshold or not 0 <= threshold <= 1:
            raise ValueError("summary threshold differs from frozen operating point")
        classes.append(dict(cohort, branch="fused", operating_point=op, threshold=threshold,
            **{key: metric_number(metrics.get(key)) for key in METRICS},
            binary_metrics_available=bool(positives and negatives), score_distribution=distribution,
            extreme_score_cases=group["extreme_score_cases"]["fused"]))
        if common["split"] == "ood":
            continue
        for tolerance in (10, 20):
            layout = group["layout"][op][str(tolerance)]
            if (layout["positive_count"] != positives or layout["tolerance_px"] != tolerance
                    or not 0 <= layout["accepted_correct"] <= layout["raw_correct"] <= positives
                    or layout["accepted_correct"] + layout["classification_FN_but_layout_correct"] != layout["raw_correct"]):
                raise ValueError("raw/accepted/FN-but-good layout counts disagree")
            for key, count in (("raw_recall", "raw_correct"), ("end_to_end_positive_recall", "accepted_correct")):
                if (positives and not math.isclose(layout[key], layout[count] / positives)) or (not positives and layout[key] is not None):
                    raise ValueError("layout rate uses a different positive denominator")
            layouts.append(dict(cohort, operating_point=op, threshold=threshold, tolerance_px=tolerance,
                **{key: metric_number(layout[key]) for key in LAYOUT_METRICS},
                layout_semantics="raw canonical decoder vs same decoder accepted by frozen Pair threshold"))
    return cohort, classes, layouts


def collect_entry(expected, evidence):
    root = Path(expected["evaluation"])
    result = dict(expected, status="pending", cohorts=[], classification_rows=[], layout_rows=[])
    try:
        if not (root / "protocol.json").is_file():
            result["reason"] = "evaluation protocol not present"
            return result
        protocol = read(root / "protocol.json")
        if protocol.get("smoke") or protocol.get("formal_training_counted") is False:
            result.update(status="excluded_smoke", reason="discard-only result is not a formal evaluation")
            return result
        if protocol.get("schema_version") != EVALUATION_SCHEMA:
            raise ValueError("not the new decoupled evaluation schema (historical S3 is separate)")
        if protocol.get("status") != "complete":
            state = protocol.get("status")
            result.update(status="failed" if state in ("failed", "interrupted") else "pending",
                reason="evaluation status: " + str(state))
            return result
        required = ("summary.json", "prediction_complete.json", "pair_results.jsonl")
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            result.update(status="incomplete", reason="complete protocol missing: " + ", ".join(missing))
            return result
        summary, marker, model = read(root / "summary.json"), read(root / "prediction_complete.json"), protocol["model"]
        if (summary.get("status") != "complete" or summary.get("model") != model
                or summary.get("split") != expected["split"] or protocol.get("split") != expected["split"]
                or summary.get("selection_on_this_population") is not False
                or summary.get("threshold_fitting_performed") is not False
                or any(protocol.get(k) is not False for k in ("thresholds_fitted", "test_or_real_used_for_fit",
                    "ood_used_for_fit", "ground_truth_used_to_select_candidates"))
                or protocol.get("decoder_design_unchanged") is not True
                or protocol.get("gt_attached_after_complete_prediction_freeze") is not True):
            raise ValueError("summary/protocol identity or no-heldout-fitting contract differs")
        if (marker.get("status") != "all_predictions_frozen" or marker.get("sample_count") != protocol["sample_count"]
                or marker.get("checkpoint_sha256") != model["checkpoint_sha256"]
                or marker.get("real_gt_opened") is not False or marker.get("review_labels_opened") is not False):
            raise ValueError("prediction-complete receipt differs")
        provenance = validate_model(expected, model, evidence)
        rows = [json.loads(line) for line in (root / "pair_results.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != protocol["sample_count"] or len({r["pair_id"] for r in rows}) != len(rows):
            raise ValueError("final row count/unique IDs differ from complete receipt")
        if expected["split"] == "ood":
            if any(r["layouts"][protocol["decoder"]].get("translation_l2_px") is not None for r in rows):
                raise ValueError("OOD row contains a pose error without GT")
            paired = summary.get("paired_baseline_layout")
            if paired is not None and set(paired) - {"source", "selection_changed", "fallback_applied", "pose_comparison_unavailable"}:
                raise ValueError("OOD paired comparison contains unsupported layout metrics")
        names = REAL_GROUPS if expected["split"] == "real" else ("all", "positive", "negative") if expected["split"] == "test" else ("all",)
        groups = summary["groups"]
        if set(groups) != set(names):
            raise ValueError("summary omits/adds a cohort outside the registered split")
        common = {key: expected[key] for key in ("arm", "architecture", "sampling", "selection", "split", "evaluation")}
        common.update(budget=20, selected_epoch=model["epoch"], seed=model["seed"],
            checkpoint_path=model["checkpoint_path"], checkpoint_sha256=model["checkpoint_sha256"],
            schedule=model["training_identity"].get("schedule"))
        cohorts, classes, layouts = [], [], []
        for name in names:
            cohort, c, l = flatten_group(groups[name], name, _population(rows, name), common, model)
            cohorts.append(cohort); classes.extend(c); layouts.extend(l)
        if expected["split"] == "real" and (groups["kept_plus_all_negative"]["negative_count"] != groups["all"]["negative_count"]
                or groups["kept_positive"]["positive_count"] + groups["excluded_positive"]["positive_count"] != groups["all"]["positive_count"]):
            raise ValueError("reviewed cohort lost negatives or relabeled excluded positives")
        provenance.update({name.replace(".json", "").replace(".", "_") + "_sha256": evidence.sha(root / name)
            for name in ("protocol.json", "summary.json", "prediction_complete.json", "pair_results.jsonl")})
        result.update(common, status="complete", provenance=provenance, cohorts=cohorts,
            classification_rows=classes, layout_rows=layouts, winner_record=model["winner_record"],
            operating_points=model["operating_points"], classifier_thresholds=model["classifier_thresholds"],
            training_budget=training_budget(model["training_identity"], model["epoch"]),
            training_identity=model["training_identity"], model_design=model.get("model_design"),
            cohort_source={key: protocol[key] for key in ("sample_count", "manifest_sha256", "keep_ids_sha256") if key in protocol},
            decoder=protocol["decoder"], decoder_config=protocol["decoder_config"],
            paired_baseline_layout=summary.get("paired_baseline_layout"))
        if expected["arm"] == CONVERTED_ARM:
            result.update(inference_only_conversion=True, matrix_head_revision="per_pair_norm_v3",
                normalization_conversion=model["training_identity"]["normalization_conversion"],
                new_training_pair_exposures=0, new_optimizer_updates=0)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        result.update(status="invalid", reason=str(error), cohorts=[], classification_rows=[], layout_rows=[])
    return result


def aggregate(root, arms=None):
    evidence = Evidence()
    results = [collect_entry(expected, evidence) for expected in discover(root, arms)]
    return dict(schema_version=SCHEMA, generated_at=datetime.now(timezone.utc).isoformat(),
        status="complete" if all(r["status"] == "complete" for r in results) else "partial",
        counts=dict(Counter(r["status"] for r in results)), evaluations=results,
        classification_rows=[row for r in results for row in r["classification_rows"]],
        layout_rows=[row for r in results for row in r["layout_rows"]],
        threshold_fitting_performed=False, held_out_model_selection_performed=False, smoke_counted=False,
        caveats=["Selections are frozen SIM-VAL checkpoint selections; operating points are separate frozen thresholds.",
            "Missing/pending/incomplete/invalid records never contribute zero-valued metrics.",
            "Only the single new fused Pair classifier is reported; coarse is untrained diagnostic, local is duplicate.",
            "OOD is positive-only with no pose GT: no Accuracy/Precision/F1/AP/AUROC or layout metric.",
            "S4 reuses S3 M12: logical budget remains M12+C8; new-arm executed exposure excludes inherited M12.",
            "Optional S3 per_pair_norm_v3 is an explicit inference-only conversion of v2 M12+C8: zero new optimization, fresh SIM-VAL selection; v2 remains separate.",
            "Exposure costs follow explicit training identity, not measured wall time or retries.",
            "Checkpoint SHA values are evaluator-verified provenance, not a fresh local weight audit."])


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _rate(value):
    return "—" if value is None else "%.2f" % (value * 100)


def markdown(report):
    lines = ["# New S3/S4/S5 frozen results", "", "Status: " + report["status"] + "; " +
        ", ".join("%s=%d" % item for item in sorted(report["counts"].items())), "",
        "Rates are %. Selection names a SIM-VAL checkpoint; OP names its frozen threshold. No held-out winner is chosen.", "",
        "## Primary cohort classification", "", "| Arm · selection · epoch | Split · cohort | N (+/−) | OP · threshold | Acc / F1 / Precision | Positive recall | AUROC / AP |",
        "|---|---|---:|---|---:|---:|---:|"]
    for row in report["classification_rows"]:
        if row["population"] != ("kept_plus_all_negative" if row["split"] == "real" else "all"):
            continue
        lines.append("| " + " | ".join(map(_cell, ("%s · %s · %s" % (row["arm"], row["selection"], row["selected_epoch"]),
            row["split"] + " · " + row["population"], "%d (%d/%d)" % (row["sample_count"], row["positive_count"], row["negative_count"]),
            "%s · %.6g" % (row["operating_point"], row["threshold"]),
            " / ".join(_rate(row[k]) for k in ("accuracy", "f1", "precision")), _rate(row["positive_recall"]),
            _rate(row["auroc"]) + " / " + _rate(row["auprc"])))) + " |")
    lines += ["", "JSON retains all cohorts, both OPs, confusion counts, raw score distributions and exact frozen selection records. — means unavailable, not zero.", "",
        "## Canonical layout at 20 px", "", "| Arm · selection | Split · cohort · OP | Positive N | Raw good / recall | Pair-accepted good / end-to-end recall | FN but good |",
        "|---|---|---:|---:|---:|---:|"]
    for row in report["layout_rows"]:
        if row["tolerance_px"] != 20 or row["population"] != ("kept_plus_all_negative" if row["split"] == "real" else "all"):
            continue
        lines.append("| " + " | ".join(map(_cell, (row["arm"] + " · " + row["selection"],
            "%s · %s · %s" % (row["split"], row["population"], row["operating_point"]), row["positive_count"],
            "%s / %s" % (row["raw_correct"], _rate(row["raw_recall"])),
            "%s / %s" % (row["accepted_correct"], _rate(row["end_to_end_positive_recall"])),
            row["classification_FN_but_layout_correct"]))) + " |")
    lines += ["", "Raw layout bypasses the classifier only for diagnosis; FN-but-good is not an implemented rescue. JSON also retains the 10 px tolerance. OOD has no layout metrics.", "",
        "## Training exposure accounting", "", "| Arm | Schedule | Logical total | Inherited source training | New-arm executed total |", "|---|---|---:|---:|---:|"]
    seen = set()
    for record in report["evaluations"]:
        if record["status"] != "complete" or record["arm"] in seen:
            continue
        seen.add(record["arm"]); cost = record["training_budget"]
        values = [record["arm"]] + [cost.get(k) if cost.get(k) is not None else "—" for k in
            ("schedule", "logical_budget_pair_exposures",
             "inherited_full_training_pair_exposures" if record.get("inference_only_conversion") else "inherited_matcher_pair_exposures",
             "arm_executed_budget_pair_exposures")]
        lines.append("| " + " | ".join(map(_cell, values)) + " |")
    if any(r.get("inference_only_conversion") for r in report["evaluations"]):
        lines += ["", "S3 per_pair_norm_v3 is an inference-only normalization conversion, not new C8 training: the full source M12+C8 weights are inherited, with zero new training exposures and zero new optimizer updates. Its SIM-VAL checkpoint/threshold selection is recomputed; original v2 results remain separate."]
    lines += ["", "Unique scheduled pair exposures only; not measured GPU time and not smoke/retry cost. S4's inherited matcher cost remains part of its logical budget.", "",
        "## Pending / incomplete / invalid", "", "| Arm · selection · split | Status | Reason |", "|---|---|---|"]
    for record in report["evaluations"]:
        if record["status"] != "complete":
            lines.append("| %s | %s | %s |" % (_cell(" · ".join(record[k] for k in ("arm", "selection", "split"))),
                record["status"], _cell(record.get("reason", ""))))
    lines += ["", "## Sources", ""]
    for record in report["evaluations"]:
        if record["status"] == "complete":
            lines.append("- [%s · %s · %s](%s/summary.json); checkpoint `%s`, freeze copy: %s." % (
                record["arm"], record["selection"], record["split"], record["evaluation"],
                record["provenance"]["checkpoint_sha256"], record["provenance"]["local_freeze_status"]))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="local arm directory; four historical arms plus v3 when its directory exists")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), help="only these registered arms; absent requested arms remain pending")
    parser.add_argument("--output", required=True, help="new directory; existing outputs are never overwritten")
    args = parser.parse_args(argv)
    report = aggregate(args.root, args.arms)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "RESULTS.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], counts=report["counts"], output=str(output.resolve()))))
    return report


if __name__ == "__main__":
    main()
