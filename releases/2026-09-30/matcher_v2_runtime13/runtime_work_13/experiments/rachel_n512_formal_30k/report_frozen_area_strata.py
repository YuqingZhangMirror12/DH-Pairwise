"""CPU-only area strata from saved pair scores and the SAME matcher's Top2.

No masks, NPZ arrays, model weights, or standalone GT files are opened. Areas
are existing count_nonzero values from 800x800 filled model-input masks, not
coarse resized masks, source-photo area, or physical manuscript area. Old area
rows contribute metadata only: their scores and layouts are never reused.

New diagnostic four-bin cutpoints .25/.5/.8 accompany the existing three-bin
.25/.5 report. Neither threshold fitting nor decoder selection occurs here.
Use --export-areas to make a small transferable metadata-only sidecar first.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rachel_n512_formal_30k.fragment_size_strata import (
    SIZE_RATIO_STRATA, _undefined_classification_metrics,
)
from experiments.rachel_n512_formal_30k.train_matrix_pair_head import population_metrics, policy_values


MISSING = "missing_or_invalid_area"
FOUR_BINS = (SIZE_RATIO_STRATA[0], SIZE_RATIO_STRATA[1], "0_5_to_lt_0_8", "ge_0_8")
TOLERANCES = (2, 5, 8, 10)
AREA_DEFINITION = "count_nonzero pixels of original 800x800 filled model-input masks; not coarse masks or physical area"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def indexed_rows(path, *, area_only=False):
    rows = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = row["pair_id"]
            if not isinstance(pair_id, str) or not pair_id or pair_id in rows:
                raise ValueError("empty or duplicate pair ID in " + str(path))
            if area_only:
                row = {key: row[key] for key in ("pair_id", "label", "fragment_a", "fragment_b",
                    "strict_member", "area_a_px", "area_b_px", "area_ratio", "size_metadata_valid") if key in row}
            rows[pair_id] = row
    return rows


def area_source(path, *, input_root=None, note=None):
    path = Path(path)
    path = path / "area_rows.jsonl" if path.is_dir() else path
    rows = indexed_rows(path, area_only=True)
    receipt_path = path.parent / "area_receipt.json"
    if receipt_path.exists():
        provenance = load_json(receipt_path)
        if provenance.get("model_mask_canvas") != [800, 800]:
            raise ValueError("area receipt must document native 800x800 model-mask counts")
    else:
        protocol_path = path.parent / "protocol.json"
        protocol = load_json(protocol_path) if protocol_path.exists() else {}
        provenance = dict(area_definition=AREA_DEFINITION, model_mask_canvas=[800, 800],
            area_source_rows=str(path.resolve()),
            area_source_protocol=str(protocol_path.resolve()) if protocol_path.exists() else None,
            source_input_root=input_root or protocol.get("prepared_cache"),
            source_prepared_manifest_sha256=protocol.get("prepared_manifest_sha256"),
            source_note=note,
            version_caveat="Source-path/command agreement is provenance evidence, not an export-time cache SHA lock.")
    return rows, provenance


def area_ratio(row):
    if row is None:
        return None, "missing_area_row"
    values = [row.get("area_" + side + "_px") for side in "ab"]
    if any(value is None for value in values):
        return None, "missing_area_fields"
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or value <= 0 or value > 800 * 800 or value != int(value) for value in values):
        return None, "invalid_or_zero_area"
    if row.get("size_metadata_valid") is False:
        return None, "area_metadata_marked_invalid"
    ratio = min(values) / max(values)
    if row.get("area_ratio") is not None and not np.isclose(float(row["area_ratio"]), ratio, rtol=0, atol=1e-12):
        raise ValueError("stored area ratio disagrees with native area counts for " + str(row.get("pair_id")))
    return ratio, "ok"


def check_pair_metadata(first, other):
    if other is None:
        return
    for key in ("label", "fragment_a", "fragment_b", "strict_member"):
        if key in first and key in other and first[key] != other[key]:
            raise ValueError("pair metadata conflict in " + key + " for " + first["pair_id"])


def join_rows(score_rows, areas, layout_rows=None):
    """Left-join onto predictions. Missing metadata/pose NEVER drops a pair."""
    joined = []
    for row in score_rows.values():
        area = areas.get(row["pair_id"])
        pose = (layout_rows or {}).get(row["pair_id"])
        check_pair_metadata(row, area)
        check_pair_metadata(row, pose)
        if area is not None and pose is not None:
            check_pair_metadata(area, pose)
        ratio, reason = area_ratio(area)
        layout = None
        if pose is not None:
            available = pose.get("layouts", {})
            names = [name for name in ("top2_mode", "full_top2_mode") if name in available]
            if len(names) > 1:
                raise ValueError("ambiguous duplicate Top2 layouts for " + row["pair_id"])
            layout = available[names[0]] if names else None
        error = layout.get("translation_l2_px") if layout else None
        valid = bool(layout and layout.get("valid"))
        finite_error = error is not None and np.isfinite(error) and error >= 0
        joined.append(dict(row, area_ratio=ratio, area_status=reason,
            top2_valid=valid, top2_error_px=float(error) if finite_error else None,
            top2_status=("missing_layout" if layout is None else
                         "invalid_layout" if not valid else
                         "missing_positive_pose_error" if row["label"] and not finite_error else "ok")))
    return joined


def normalized_policies(freeze):
    schema = freeze.get("schema_version")
    if schema == "rachel-matrix-pair-head/v1":
        return freeze["policies"]
    if schema == "rachel-cached-score-fusion/v1":
        return {name: dict(branch=name, threshold=policy["threshold"], gate_threshold=None)
                for name, policy in freeze["policies"].items()}
    raise ValueError("expected existing matrix-head or cached-score-fusion validation freeze")


def group_summary(rows, policies, *, layout_supplied):
    labels = np.asarray([row["label"] for row in rows], bool)
    positive, count = int(labels.sum()), len(rows)
    if not count:
        return dict(sample_count=0, positive_count=0, negative_count=0, classification=None,
                    fixed_top2=None, joint=None, area_status_counts={}, pose_status_counts={}, status="empty")
    predictions = dict(labels=labels, scores={key: np.asarray([row["scores"][key] for row in rows], float)
                       for key in rows[0]["scores"]})
    classification = population_metrics(predictions, policies)
    for metrics in classification["policies"].values():
        _undefined_classification_metrics(metrics, positive)
    result = dict(sample_count=count, positive_count=positive, negative_count=count - positive,
        classification=classification, area_status_counts=dict(Counter(row["area_status"] for row in rows)),
        pose_status_counts=dict(Counter(row["top2_status"] for row in rows)), status="computed",
        fixed_top2=None, joint=None)
    if not layout_supplied:
        result["joint_unavailable"] = "no same-checkpoint saved Top2 results supplied; no old-pose substitution"
        return result
    valid = np.asarray([row["top2_valid"] for row in rows], bool)
    error = np.asarray([row["top2_error_px"] if row["top2_error_px"] is not None else np.inf for row in rows])
    successes = {str(t): labels & valid & (error <= t) for t in TOLERANCES}
    result["fixed_top2"] = dict(positive_count=positive, positive_valid_count=int((labels & valid).sum()),
        positive_pose_coverage=float((labels & valid).sum() / positive) if positive else None,
        recall_counts={t: int(correct.sum()) for t, correct in successes.items()},
        recall={t: float(correct.sum() / positive) if positive else None for t, correct in successes.items()})
    joint = {}
    for name, policy in policies.items():
        if name not in classification["policies"]:
            continue
        accepted = policy_values(predictions, policy) >= policy["threshold"]
        joint[name] = {}
        for tolerance, correct in successes.items():
            tp = int((accepted & correct).sum())
            fp, fn = int(accepted.sum()) - tp, positive - tp
            joint[name][tolerance] = dict(tp=tp, fp=fp, fn=fn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / positive if positive else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
    result["joint"] = joint
    return result


def summarize(rows, policies, *, layout_supplied):
    output = dict(overall=group_summary(rows, policies, layout_supplied=layout_supplied))
    for name, bins, edges in (("four_bin_diagnostic", FOUR_BINS, (.25, .5, .8)),
                              ("three_bin_compatible", SIZE_RATIO_STRATA, (.25, .5))):
        groups = {key: [] for key in tuple(bins) + (MISSING,)}
        for row in rows:
            ratio = row["area_ratio"]
            key = MISSING if ratio is None else bins[int(np.searchsorted(edges, ratio, side="right"))]
            groups[key].append(row)
        output[name] = dict(fixed_bin_edges=list(edges), groups={key: group_summary(values, policies,
            layout_supplied=layout_supplied) for key, values in groups.items()})
    return output


def run(args):
    areas, provenance = area_source(args.area_rows, input_root=args.area_input_root, note=args.area_source_note)
    output = Path(args.output)
    if args.export_areas:
        output.mkdir(parents=True, exist_ok=False)
        with (output / "area_rows.jsonl").open("x", encoding="utf-8") as stream:
            for row in areas.values():
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        save_json(output / "area_receipt.json", dict(provenance, status="complete", sample_count=len(areas),
                  positive_count=sum(bool(row.get("label")) for row in areas.values()),
                  scores_or_layouts_exported=False))
        print(json.dumps(dict(status="complete", metadata_rows=len(areas), output=str(output))), flush=True)
        return
    if not args.source or not args.freeze:
        raise ValueError("reporting requires --source and its unchanged --freeze")
    source = Path(args.source)
    metrics, freeze = load_json(source / "metrics.json"), load_json(args.freeze)
    if metrics.get("status") != "complete" or freeze.get("status") != "complete":
        raise ValueError("completed score predictions and policy freeze required")
    if metrics.get("native_probability_reporting") is not True:
        raise ValueError("use the native-probability reporting artifact, not legacy FP64 sigmoid reconstruction")
    precision = freeze.get("matcher_precision", freeze.get("precision"))
    if precision != "fp32" or metrics.get("cache", {}).get("precision", "fp32") != precision:
        raise ValueError("this comparison requires the registered FP32 frozen matcher scores")
    split = metrics["split"]
    if split not in ("test", "real"):
        raise ValueError("this is a frozen TEST/REAL report, not model/threshold selection")
    identity = metrics.get("matcher_checkpoint_id", metrics.get("cache", {}).get("matcher_checkpoint_id"))
    if not identity or identity != freeze.get("matcher_checkpoint_id"):
        raise ValueError("scores and policy freeze refer to different matcher checkpoints")
    score_rows = indexed_rows(source / "pair_scores.jsonl")
    labels = [row["label"] for row in score_rows.values()]
    if (len(labels), sum(labels)) != ((3000, 1500) if split == "test" else (1016, 508)):
        raise ValueError("saved score population is not full TEST3000 or REAL1016")
    layout_rows, layout_provenance = None, None
    if args.layout_results:
        directory = Path(args.layout_results)
        layout_summary = load_json(directory / "summary.json")
        layout_protocol = load_json(directory / "protocol.json")
        layout_identity = layout_protocol.get("matcher_checkpoint_id", layout_protocol.get("checkpoint_sha256"))
        if (layout_summary.get("status") != "complete" or layout_summary.get("split") != split
                or layout_identity != identity or layout_protocol.get("precision") != precision):
            raise ValueError("joint metrics require completed SAME-checkpoint/split/precision Top2 predictions")
        layout_rows = indexed_rows(directory / "pair_results.jsonl")
        layout_provenance = dict(directory=str(directory.resolve()), matcher_checkpoint_id=layout_identity,
            decoder="fixed top2_mode/full_top2_mode; selected_full_decoder is ignored")
    policies = normalized_policies(freeze)
    joined = join_rows(score_rows, areas, layout_rows)
    result = dict(status="complete", split=split, matcher_checkpoint_id=identity, precision=precision,
        frozen_policy_source=str(Path(args.freeze).resolve()), frozen_policies=freeze["policies"],
        source_scores=str((source / "pair_scores.jsonl").resolve()), source_metrics=str((source / "metrics.json").resolve()),
        area_provenance=provenance, area_definition=AREA_DEFINITION, layout_provenance=layout_provenance,
        area_ratio_definition="min(area_a_px,area_b_px)/max(area_a_px,area_b_px)",
        thresholds_refit=False, thresholds_modified=False, scores_modified=False, model_executed=False,
        old_area_source_scores_or_layouts_used=False, missing_rows_dropped=False,
        unused_area_metadata_count=len(set(areas) - set(score_rows)),
        joint_definition="TP=accepted positive with valid fixed Top2 error<=t; FP=all accepted-TP; FN=all positives-TP; missing poses fail",
        **summarize(joined, policies, layout_supplied=layout_rows is not None))
    if split == "real":
        if any("strict_member" not in row for row in joined):
            raise ValueError("REAL requires explicit saved strict membership")
        strict = [row for row in joined if row["strict_member"]]
        if (len(strict), sum(row["label"] for row in strict)) != (547, 508):
            raise ValueError("strict REAL membership must remain 547 pairs/508 positives")
        result["strict547"] = summarize(strict, policies, layout_supplied=layout_rows is not None)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "summary.json", result)
    print(json.dumps(dict(status="complete", output=str(output), sample_count=len(joined),
                         missing_area_count=sum(row["area_ratio"] is None for row in joined))), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", help="completed native-probability score report directory")
    p.add_argument("--freeze", help="existing validation policy freeze; no fitting is performed")
    p.add_argument("--area-rows", required=True, help="saved area-bearing pair_results JSONL or compact area sidecar directory")
    p.add_argument("--layout-results", help="completed same-matcher assignment/data-arm result directory; only fixed Top2 is read")
    p.add_argument("--output", required=True, help="new directory, never overwritten")
    p.add_argument("--export-areas", action="store_true", help="only export small metadata sidecar; no scores or layouts")
    p.add_argument("--area-input-root", help="documented 800-mask input root; recorded, not scanned")
    p.add_argument("--area-source-note", help="documented input-version evidence/caveat, e.g. controller defaults unchanged")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
