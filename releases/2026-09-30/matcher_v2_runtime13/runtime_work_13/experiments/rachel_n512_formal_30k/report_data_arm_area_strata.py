"""CPU-only four-/three-bin area reports for completed Stage2 data arms.

Only saved TEST/REAL pair results, their own completed evaluation/training
receipts, and the compact 800-mask area sidecar are read. Scores and poses both
come from this arm's pair_results.jsonl, never from the old area source. No
model, raw mask, threshold fitting, decoder selection or GPU work is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.rachel_n512_formal_30k import report_frozen_area_strata as shared


SCHEMA = "rachel-data-arm-area-strata/v1"
ARMS = ("original24k", "matched24k", "realism60k")
BRANCHES = ("coarse", "local", "fused")
DECODER = "full_top2_mode"
COUNTS = {"test": (3000, 1500), "real": (1016, 508)}


def _probability(value):
    return type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1


def load_authority(root, arm, split):
    """Require this arm's finalized budget, VAL thresholds and fixed decoder."""
    if arm not in ARMS or split not in COUNTS:
        raise ValueError("only the three registered data arms and TEST/REAL are supported")
    source = root / "realism_evaluation" / arm / split
    training = root / "realism_training" / arm
    freeze_path = training / "train_val_freeze.json"
    freeze = shared.load_json(freeze_path)
    if (freeze.get("status") != "complete" or freeze.get("test_or_real_used_for_fit") is not False
            or freeze.get("original_validation_unchanged") is not True
            or freeze.get("completed_global_exposures") != 120000
            or freeze.get("completed_optimizer_updates") != 7500
            or freeze.get("completed_validation_events") != 5):
        raise ValueError("completed own-arm VAL-only 120k/7500/5-event training freeze required")
    summary = shared.load_json(source / "summary.json")
    protocol = shared.load_json(source / "protocol.json")
    for record in (summary, protocol):
        if (record.get("status") != "complete" or record.get("split") != split
                or record.get("precision") != "fp32" or record.get("test_or_real_used_for_fit") is not False
                or record.get("selected_full_decoder") != DECODER):
            raise ValueError("completed same-split FP32 data-arm evaluation with the fixed Top2 decoder required")
    checkpoint = summary.get("checkpoint_sha256")
    if not checkpoint or protocol.get("checkpoint_sha256") != checkpoint:
        raise ValueError("evaluation summary and protocol checkpoint identities disagree")
    if (Path(protocol.get("training_run", "")).resolve() != training.resolve()
            or Path(protocol.get("training_freeze", "")).resolve() != freeze_path.resolve()
            or protocol.get("training_freeze_sha256") != hashlib.sha256(freeze_path.read_bytes()).hexdigest()
            or protocol.get("checkpoint_epoch") != freeze.get("selected_epoch")):
        raise ValueError("evaluation provenance does not identify this arm's unchanged selected training freeze")
    config = protocol.get("decoder_config", {})
    if (config.get("correspondence_mode") != "topk_union" or config.get("top_k") != 2
            or config.get("max_candidates") != 512 or config.get("inlier_radius_px") != 10.0
            or config.get("min_inliers") != 3 or config.get("decoder") != "mode_consensus"):
        raise ValueError("data-arm decoder is not the registered common Top2 mode configuration")
    thresholds = freeze.get("classifier_thresholds", {})
    if (set(thresholds) != set(BRANCHES) or not all(_probability(value) for value in thresholds.values())
            or freeze.get("validation", {}).get("thresholds") != thresholds
            or any(record.get("branch_validation_thresholds") != thresholds for record in (summary, protocol))
            or any(record.get("original_fused_threshold") != thresholds["fused"] for record in (summary, protocol))):
        raise ValueError("three branch thresholds must equal this model's unchanged selected VAL thresholds")
    policies = {name: dict(branch=name, threshold=thresholds[name], gate_threshold=None) for name in BRANCHES}
    return source, freeze_path, summary, protocol, policies


def adapt_rows(predictions, areas, split):
    """Preserve prediction rows; require ordered endpoint identity for metadata."""
    scores, poses = {}, {}
    for pair_id, row in predictions.items():
        if type(row.get("label")) is not bool:
            raise ValueError("data-arm rows require explicit bool labels")
        for field in ("fragment_a", "fragment_b"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError("prediction ordered endpoints are required: " + field)
        if split == "real" and type(row.get("strict_member")) is not bool:
            raise ValueError("REAL rows require explicit bool strict membership")
        area = areas.get(pair_id)
        if area is not None:
            fields = ("label", "fragment_a", "fragment_b") + (("strict_member",) if split == "real" else ())
            if any(field not in area for field in fields):
                raise ValueError("area row lacks ordered pair identity fields for " + pair_id)
            shared.check_pair_metadata(row, area)
            # If the new evaluator also saved native areas, never override a
            # disagreement using metadata from an older input version.
            for field in ("area_a_px", "area_b_px"):
                if field in row and field in area and row[field] != area[field]:
                    raise ValueError("new prediction and area-sidecar native counts disagree: " + pair_id)
        values = row.get("classification", {})
        if any(not _probability(values.get(name)) for name in BRANCHES):
            raise ValueError("finite native coarse/local/fused probabilities required")
        score = {key: row[key] for key in ("pair_id", "label", "fragment_a", "fragment_b", "strict_member") if key in row}
        score["scores"] = {name: values[name] for name in BRANCHES}
        scores[pair_id] = score
        layout = row.get("layouts", {}).get(DECODER)
        if layout is not None and (not isinstance(layout, dict) or type(layout.get("valid")) is not bool):
            raise ValueError("saved fixed Top2 layout has a malformed validity field")
        poses[pair_id] = dict(score, layouts={} if layout is None else {DECODER: layout})
    # Missing area rows stay in the missing bin; missing poses count as failures.
    # Only these new prediction poses enter the shared metadata join.
    return shared.join_rows(scores, areas, poses)


def area_report(rows, policies):
    result = shared.summarize(rows, policies, layout_supplied=True)
    groups = [result["overall"]]
    groups += list(result["four_bin_diagnostic"]["groups"].values())
    groups += list(result["three_bin_compatible"]["groups"].values())
    for group in groups:
        if group["fixed_top2"] is not None:
            group["fixed_top2"]["decoder"] = DECODER
        if group["joint"] is not None:
            group["joint"] = {"fused": group["joint"]["fused"]}
    return result


def _check_population(rows, summary, expected):
    if (len(rows), sum(row["label"] for row in rows)) != expected:
        raise ValueError("saved data-arm prediction population is incomplete")
    if (summary.get("sample_count"), summary.get("positive_count")) != expected:
        raise ValueError("evaluation summary population differs from registered split")


def _check_overall_classification(derived, saved):
    for name in BRANCHES:
        actual = derived["overall"]["classification"]["policies"][name]
        expected = saved.get("classification", {}).get(name, {}).get("at_validation_row_f1_threshold", {})
        if any(actual[key] != expected.get(key) for key in ("threshold", "tp", "fp", "fn", "tn")):
            raise ValueError("saved pair scores do not reproduce own evaluation classification: " + name)


def run(args):
    root = Path(args.root).resolve()
    source, freeze_path, summary, protocol, policies = load_authority(root, args.arm, args.split)
    areas, area_provenance = shared.area_source(root / "area_metadata" / args.split)
    if area_provenance.get("status") != "complete" or area_provenance.get("scores_or_layouts_exported") is not False:
        raise ValueError("completed metadata-only native area sidecar required")
    expected_input = protocol.get("prepared_cache" if args.split == "real" else "dataset_root")
    if not expected_input or area_provenance.get("source_input_root") != expected_input:
        raise ValueError("area sidecar and data evaluation identify different model-mask input roots")
    source_manifest = area_provenance.get("source_prepared_manifest_sha256")
    if source_manifest is not None and source_manifest != protocol.get("prepared_manifest_sha256"):
        raise ValueError("area sidecar and data evaluation prepared-manifest identities differ")
    predictions = shared.indexed_rows(source / "pair_results.jsonl")
    rows = adapt_rows(predictions, areas, args.split)
    _check_population(rows, summary, COUNTS[args.split])
    groups = area_report(rows, policies)
    _check_overall_classification(groups, summary)
    result = dict(schema_version=SCHEMA, status="complete", arm=args.arm, split=args.split,
        checkpoint_sha256=summary["checkpoint_sha256"], precision="fp32", decoder=DECODER,
        decoder_policy="same fixed prior Top2 mode for every arm and pair, not selected on these data",
        source_scores_and_poses=str((source / "pair_results.jsonl").resolve()),
        source_summary=str((source / "summary.json").resolve()), source_protocol=str((source / "protocol.json").resolve()),
        frozen_policy_source=str(freeze_path.resolve()), frozen_policies=policies,
        threshold_regime="this data model's own selected VAL thresholds, unchanged across area bins",
        area_provenance=area_provenance, area_definition=shared.AREA_DEFINITION,
        area_ratio_definition="min(area_a_px,area_b_px)/max(area_a_px,area_b_px)",
        thresholds_refit=False, thresholds_modified=False, scores_modified=False, poses_modified=False,
        model_executed=False, old_area_source_scores_or_layouts_used=False, missing_rows_dropped=False,
        ordered_endpoint_agreement_required=True, unused_area_metadata_count=len(set(areas) - set(predictions)),
        joint_definition="TP=accepted fused positive with valid fixed Top2 error<=t; FP=all accepted-TP; FN=all positives-TP; missing poses fail",
        **groups)
    if args.split == "real":
        strict_rows = [row for row in rows if row["strict_member"]]
        strict_saved = summary.get("strict_summary", {})
        _check_population(strict_rows, strict_saved, (547, 508))
        strict = area_report(strict_rows, policies)
        _check_overall_classification(strict, strict_saved)
        result["strict547"] = strict
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shared.save_json(output / "summary.json", result)
    print(json.dumps(dict(status="complete", arm=args.arm, split=args.split, output=str(output.resolve()),
        sample_count=len(rows), missing_area_count=sum(row["area_ratio"] is None for row in rows))), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--split", required=True, choices=tuple(COUNTS))
    parser.add_argument("--output", required=True, help="new directory, never overwritten")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
