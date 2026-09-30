"""Extract already-computed, validation-frozen metrics across listed arms.

Only pipeline_snapshot.json and each listed arm's pipeline_status.json,
val/validation_freeze.json, and val/test/real summary.json are read. No models,
per-pair JSONL, GT, or independent strict-summary file is opened. No metric,
threshold, or winner is recomputed. Output is a mechanical JSON aggregate, not
a claim that the current snapshot lists every planned experiment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path


SCHEMA_VERSION = "completed-ablation-arm-aggregate/1"
MAIN_CLASSIFICATION = "fused.at_original_frozen_threshold"
STAGES = ("val", "test", "real")
EXPECTED_COUNTS = {"val": 3000, "test": 3000, "real": 1016}


def _issue(kind, source, **details):
    return dict(type=kind, source=str(source), **details)


def _read_object(path):
    def reject_constant(value):
        raise ValueError("non-JSON numeric constant: " + value)
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream, parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value, []
    except FileNotFoundError:
        return None, [_issue("missing_input", path)]
    except (OSError, ValueError) as exc:
        return None, [_issue("invalid_input", path, message=str(exc))]


def _valid_threshold(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def _seam_quality(summary, decoder, parent=None):
    parent = parent or {}
    by_decoder = summary.get("seam_quality")
    metric = by_decoder.get(decoder) if isinstance(by_decoder, dict) else None
    unavailable = summary.get("seam_quality_unavailable", parent.get("seam_quality_unavailable"))
    reason = unavailable.get(decoder) if isinstance(unavailable, dict) else unavailable
    evaluated = summary.get("seam_quality_evaluated", parent.get("seam_quality_evaluated"))
    if metric is None and reason is None:
        reason = "selected_decoder_metrics_not_stored" if evaluated else "not_evaluated"
    return dict(status="available" if metric is not None else "unavailable",
                metrics=deepcopy(metric), reason=deepcopy(reason),
                evaluated=evaluated, source_unavailable=deepcopy(unavailable))


def _population(summary, *, split, population, path, pointer, decoder, threshold,
                stage_status, stage_issues, expected_count, parent=None, frozen_metadata=None):
    result = dict(status=stage_status, split=split, population=population,
                  sample_count=None, positive_count=None, selected_decoder=decoder,
                  source=dict(path=str(path), json_pointer=pointer),
                  classification_metric_key=MAIN_CLASSIFICATION,
                  classification=None, branch_classification=None, layout=None,
                  seam_quality=dict(status="unavailable", metrics=None, reason="summary_missing",
                                    evaluated=None, source_unavailable=None),
                  issues=deepcopy(stage_issues))
    if summary is None:
        if stage_status == "complete":
            result["status"] = "missing"
        return result
    if not isinstance(summary, dict):
        result["issues"].append(_issue("invalid_population_summary", path, json_pointer=pointer))
        result["status"] = "invalid"
        return result
    result["sample_count"] = summary.get("sample_count")
    result["positive_count"] = summary.get("positive_count")
    result["branch_classification"] = deepcopy(summary.get("classification"))
    result["seam_quality"] = _seam_quality(summary, decoder, parent)
    own_issues = []
    if summary.get("sample_count") != expected_count:
        own_issues.append(_issue("incomplete_population", path, json_pointer=pointer,
                                 expected=expected_count, observed=summary.get("sample_count")))
    positives = summary.get("positive_count")
    if type(positives) is not int or not 0 <= positives <= expected_count:
        own_issues.append(_issue("invalid_positive_count", path, json_pointer=pointer, observed=positives))
    declared = summary.get("selected_full_decoder")
    if declared is not None and declared != decoder:
        own_issues.append(_issue("decoder_conflict", path, json_pointer=pointer, expected=decoder, observed=declared))
    for key in ("checkpoint_sha256", "precision"):
        if key in summary and summary[key] != (frozen_metadata or {}).get(key):
            own_issues.append(_issue(key + "_conflict", path, json_pointer=pointer,
                                     expected=(frozen_metadata or {}).get(key), observed=summary[key]))
    branches = summary.get("classification")
    fused = branches.get("fused") if isinstance(branches, dict) else None
    classification = fused.get("at_original_frozen_threshold") if isinstance(fused, dict) else None
    if not isinstance(classification, dict):
        own_issues.append(_issue("missing_main_classification", path, json_pointer=pointer))
    elif not _valid_threshold(classification.get("threshold")) or classification["threshold"] != threshold:
        own_issues.append(_issue("threshold_conflict", path, json_pointer=pointer,
                                 expected=threshold, observed=classification.get("threshold")))
    layout = summary.get("layout", {}).get(decoder) if isinstance(summary.get("layout"), dict) else None
    if not isinstance(layout, dict):
        own_issues.append(_issue("missing_frozen_decoder_layout", path, json_pointer=pointer, selected_decoder=decoder))
    result["issues"].extend(own_issues)
    if any(issue["type"].endswith("_conflict") for issue in result["issues"]):
        result["status"] = "conflict"
    elif own_issues:
        result["status"] = "incomplete" if all(i["type"] == "incomplete_population" for i in own_issues) else "invalid"
    # Only publish main metrics from a complete, consistent population. Raw
    # branch_classification remains available as clearly status-tagged evidence.
    if result["status"] == "complete":
        result["classification"] = deepcopy(classification)
        result["layout"] = deepcopy(layout)
    return result


def _aggregate_arm(root, listed):
    key = listed.get("key")
    result = dict(key=key, arm=listed.get("arm"), training_root=listed.get("training_root"),
                  status="partial", snapshot_reported_status=listed.get("status"),
                  pipeline_reported_status=None, selected_decoder=None,
                  original_fused_threshold=None, checkpoint_epoch=None, checkpoint_sha256=None,
                  precision=None, checkpoint=listed.get("checkpoint"), sources={}, stages={}, populations={}, issues=[])
    if not isinstance(key, str) or not key or Path(key).name != key or key in (".", ".."):
        result["status"] = "needs_attention"
        result["issues"].append(_issue("invalid_arm_key", root / "pipeline_snapshot.json", observed=key))
        return result
    directory = root / key
    paths = dict(pipeline_status=directory / "pipeline_status.json",
                 validation_freeze=directory / "val" / "validation_freeze.json",
                 **{stage + "_summary": directory / stage / "summary.json" for stage in STAGES})
    result["sources"] = {name: str(path) for name, path in paths.items()}
    pipeline, pipeline_issues = _read_object(paths["pipeline_status"])
    freeze, freeze_issues = _read_object(paths["validation_freeze"])
    result["issues"].extend(pipeline_issues)
    if pipeline:
        result["pipeline_reported_status"] = pipeline.get("status")
        result["checkpoint"] = pipeline.get("checkpoint", result["checkpoint"])
    decoder, threshold = None, None
    if freeze is not None:
        decoder, threshold = freeze.get("selected_full_decoder"), freeze.get("original_fused_threshold")
        # The existing validation freeze intentionally has NO status field.
        if (freeze.get("source_split") != "validation" or freeze.get("probe_only") is not False
                or freeze.get("test_or_real_used_for_fit") is not False):
            freeze_issues.append(_issue("invalid_validation_freeze", paths["validation_freeze"]))
        if not isinstance(decoder, str) or not decoder or not _valid_threshold(threshold):
            freeze_issues.append(_issue("invalid_frozen_selection", paths["validation_freeze"]))
        if pipeline and pipeline.get("pair_threshold") != threshold:
            freeze_issues.append(_issue("threshold_conflict", paths["pipeline_status"],
                                        expected=threshold, observed=pipeline.get("pair_threshold")))
        result.update(selected_decoder=decoder, original_fused_threshold=threshold,
                      checkpoint_epoch=freeze.get("checkpoint_epoch"), checkpoint_sha256=freeze.get("checkpoint_sha256"),
                      precision=freeze.get("precision"))
    result["issues"].extend(freeze_issues)
    for stage in STAGES:
        path = paths[stage + "_summary"]
        summary, read_issues = _read_object(path)
        reported = (pipeline or {}).get("stages", {}).get(stage, {}).get("status")
        issues = deepcopy(read_issues + freeze_issues + pipeline_issues)
        if reported in ("failed", "needs_attention", "error"):
            status = "failed"
            issues.append(_issue("pipeline_stage_failed", paths["pipeline_status"], stage=stage, observed=reported))
        elif summary is None:
            status = "invalid" if any(i["type"] == "invalid_input" for i in read_issues) else "missing"
        elif reported != "complete":
            status = "incomplete"
            issues.append(_issue("pipeline_stage_not_complete", paths["pipeline_status"], stage=stage, observed=reported))
        elif (summary.get("status") != "complete" or summary.get("probe_only") is True or summary.get("split") != stage):
            status = "incomplete"
            issues.append(_issue("stage_summary_not_complete", path, observed=summary.get("status"), split=summary.get("split")))
        else:
            status = "complete"
        if freeze_issues or pipeline_issues:
            status = ("conflict" if any(i["type"].endswith("_conflict") for i in issues)
                      else "missing" if any(i["type"] == "missing_input" for i in issues) else "invalid")
        label = "real_balanced1016" if stage == "real" else "synthetic_" + stage
        population = _population(summary, split=stage, population=label, path=path, pointer="",
            decoder=decoder, threshold=threshold, stage_status=status, stage_issues=issues,
            expected_count=EXPECTED_COUNTS[stage], frozen_metadata=freeze)
        result["populations"]["real_balanced1016" if stage == "real" else stage] = population
        result["stages"][stage] = dict(status=population["status"], reported_status=reported, source=str(path))
        if stage == "real":
            strict = summary.get("strict_summary") if summary else None
            # The strict subset is stored within the same real evaluation;
            # it cannot escape a conflict or failure in the outer population.
            strict_issues = deepcopy(population["issues"])
            if strict is None:
                strict_issues.append(_issue("missing_strict_summary", path, json_pointer="/strict_summary"))
            strict_population = _population(strict, split="real", population="real_strict547", path=path,
                pointer="/strict_summary", decoder=decoder, threshold=threshold, stage_status=population["status"],
                stage_issues=strict_issues, expected_count=547, parent=summary, frozen_metadata=freeze)
            result["populations"]["real_strict547"] = strict_population
            if strict_population["status"] != "complete" and result["stages"][stage]["status"] == "complete":
                result["stages"][stage]["status"] = strict_population["status"]
    statuses = [p["status"] for p in result["populations"].values()]
    if (any(s in ("invalid", "failed", "conflict") for s in statuses)
            or result["pipeline_reported_status"] in ("needs_attention", "failed", "error")):
        result["status"] = "needs_attention"
    elif len(statuses) == 4 and all(s == "complete" for s in statuses) and result["pipeline_reported_status"] == "complete":
        result["status"] = "complete"
    return result


def summarize_completed_arms(evaluation_root):
    root = Path(evaluation_root).resolve()
    source = root / "pipeline_snapshot.json"
    snapshot, issues = _read_object(source)
    snapshot = snapshot or {}
    listed = snapshot.get("arms", [])
    if not isinstance(listed, list) or any(not isinstance(arm, dict) for arm in listed):
        issues.append(_issue("invalid_snapshot_arms", source))
        listed = []
    arms = [_aggregate_arm(root, arm) for arm in listed]
    keys = [arm["key"] for arm in arms]
    if len(set(str(k) for k in keys)) != len(keys):
        issues.append(_issue("duplicate_snapshot_arm_key", source))
    counts = dict(listed_arms=len(arms), complete=sum(a["status"] == "complete" for a in arms),
                  partial=sum(a["status"] == "partial" for a in arms),
                  needs_attention=sum(a["status"] == "needs_attention" for a in arms))
    status = "needs_attention" if issues or counts["needs_attention"] else "complete" if arms and counts["complete"] == len(arms) else "partial"
    return dict(schema_version=SCHEMA_VERSION, status=status, evaluation_root=str(root), source_snapshot=str(source),
                scope="Only arms listed in this pipeline snapshot; completion/counts do not establish coverage of all planned arms.",
                expected_total_arm_count=None, counts=counts,
                pipeline_observation={name: deepcopy(snapshot.get(name)) for name in (
                    "status", "updated_at", "follow", "live_training_pids", "pending_eligible",
                    "waiting_for_training", "discovery_needs_attention")},
                main_classification_metric_key=MAIN_CLASSIFICATION,
                decoder_selection_source="each_arm_val_validation_freeze_only", arms=arms, issues=issues)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    aggregate = summarize_completed_arms(args.evaluation_root)
    output = args.output_json.resolve()
    inputs = {Path(aggregate["source_snapshot"]).resolve()}
    inputs.update(Path(path).resolve() for arm in aggregate["arms"] for path in arm["sources"].values())
    if output in inputs:
        parser.error("--output-json must not replace an input artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(aggregate, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return aggregate


if __name__ == "__main__":
    main()
