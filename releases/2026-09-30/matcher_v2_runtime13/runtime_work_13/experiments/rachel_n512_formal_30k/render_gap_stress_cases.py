"""Static spatial mask cases from saved gap-stress predictions; no inference.

Visual contract: compare original-GT and saved-prediction placement after
material loss, not estimate success rates. Up to six cases are selected in
saved TEST row order without using predictions. Four equal-aspect panels use
paired whole-view bounds and paired, GT-defined 80px zoom bounds. White, blue A,
gold B, explicit A/B labels, grey missing material and dashed clean outlines.
No RGB, invented texture, rotation, recentering of failed predictions, or model
loading. --seam-diagnostics selects actual GT-seam damage; without it, selection
means any edge changed and explicitly does not imply damage at the true seam.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.pairwise_data.rachel_gap_stress import RachelGapStressDataset


SCHEMA = "rachel-gap-stress-cases/1"
DECODER, MAX_CASES, ZOOM_WIDTH_PX = "full_top2_mode", 6, 80.0
BLUE, GOLD, INK, GREY, REMOVED = "#2C6FA3", "#B58A2B", "#24303A", "#737B82", "#B9BEC2"


def _clean(value):
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    return value


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_evaluation(directory):
    root = Path(directory).resolve()
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or receipt.get("split") != "test":
        raise ValueError("rendering requires a completed TEST evaluation receipt (full or explicit probe)")
    if (receipt.get("selected_full_decoder") != DECODER or receipt.get("seed") != 260910
            or receipt.get("requested_max_inward_erosion_depth_px") not in (0, 2, 4)
            or receipt.get("test_or_real_used_for_fit") is not False):
        raise ValueError("receipt is not the frozen 0/2/4px gap-stress evaluation")
    with (root / "pair_results.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if (len(rows) != receipt.get("evaluated_pair_count")
            or len({r["pair_id"] for r in rows}) != len(rows)):
        raise ValueError("saved results do not cover the receipt's unique evaluated rows")
    if receipt.get("full_test") and (len(rows) != 3000 or sum(r["label"] for r in rows) != 1500):
        raise ValueError("full-test receipt does not contain the original balanced TEST3000")
    if not receipt.get("full_test") and receipt.get("probe_only") is not True:
        raise ValueError("a partial evaluation must be explicitly labelled as a probe")
    threshold = receipt["original_fused_threshold"]
    if not np.isfinite(threshold) or receipt["branch_validation_thresholds"]["fused"] != threshold:
        raise ValueError("saved native classifier threshold must be the original VAL-frozen threshold")
    return rows, receipt


def read_diagnostics(path, receipt, rows):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "rachel-seam-gap-manifest/1"
            or manifest.get("status") != "complete"
            or manifest.get("seed") != receipt["seed"]
            or manifest.get("max_depth_px") != receipt["requested_max_inward_erosion_depth_px"]
            or manifest.get("test_manifest_sha256") != receipt["test_manifest_sha256"]):
        raise ValueError("seam diagnostics do not match this TEST input recipe, depth and seed")
    records = manifest["rows"]
    by_id = {row["pair_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate seam diagnostic pair IDs")
    for row in rows:
        if row["pair_id"] not in by_id:
            raise ValueError("complete seam diagnostics do not cover every saved prediction row")
        source = by_id[row["pair_id"]]
        if any(source[key] != row[key] for key in ("fragment_a", "fragment_b", "label", "actual_changed")):
            raise ValueError("seam diagnostic endpoint, label or mask-change identity differs")
    return {pair_id: record["diagnostics"] for pair_id, record in by_id.items()}, manifest


def select_cases(rows, max_cases=6, diagnostics=None):
    if type(max_cases) is not int or not 1 <= max_cases <= MAX_CASES:
        raise ValueError("max_cases must be between 1 and 6")
    candidates, missing, unmeasured = [], 0, 0
    for row in rows:
        if not row["label"]:
            continue
        if diagnostics is None:
            eligible = row["actual_changed"] is True
        else:
            evidence = diagnostics.get(row["pair_id"])
            if evidence is None:
                missing += 1
                continue
            damaged = evidence.get("actual_seam_damaged")
            if damaged is None:
                unmeasured += 1
            eligible = damaged is True
        if eligible:
            candidates.append(row)
    selected = candidates[:max_cases]
    return selected, dict(rule="first positive rows in saved TEST order with " +
        ("actual GT-seam damage from metric-only diagnostics" if diagnostics is not None else
         "actual_changed mask report; any edge damage, not necessarily at the GT seam"),
        prediction_outcome_used=False, requested_case_count=max_cases, candidate_count=len(candidates),
        selected_case_count=len(selected), missing_positive_diagnostic_count=missing,
        unmeasured_positive_diagnostic_count=unmeasured,
        selected_pair_ids=[row["pair_id"] for row in selected])


def prediction_translation(row):
    layout = row["layouts"][DECODER]
    if not layout["valid"] or layout["translation_rc"] is None:
        return None
    value = np.asarray(layout["translation_rc"], dtype=float)
    return value if value.shape == (2,) and np.isfinite(value).all() else None


def _mask(value):
    mask = np.asarray(value).squeeze()
    if mask.ndim != 2 or not np.all((mask == 0) | (mask == 1)) or not mask.any():
        raise ValueError("figure inputs must be nonempty filled binary masks")
    return mask.astype(bool)


def mask_bounds(mask, offset_rc=(0., 0.)):
    rr, cc = np.nonzero(_mask(mask))
    dr, dc = np.asarray(offset_rc, float)
    return (float(cc.min() + dc - .5), float(cc.max() + dc + .5),
            float(rr.min() + dr - .5), float(rr.max() + dr + .5))


def shared_whole_bounds(clean_a, clean_b, gt_rc, predicted_rc):
    boxes = [mask_bounds(clean_a), mask_bounds(clean_b, -np.asarray(gt_rc, float))]
    if predicted_rc is not None:
        boxes.append(mask_bounds(clean_b, -np.asarray(predicted_rc, float)))
    x0, x1 = min(b[0] for b in boxes), max(b[1] for b in boxes)
    y0, y1 = min(b[2] for b in boxes), max(b[3] for b in boxes)
    pad = max(12., .045 * max(x1 - x0, y1 - y0))
    return (x0 - pad, x1 + pad), (y1 + pad, y0 - pad)


def choose_zoom_anchor(clean, diagnostic=None):
    """Only GT geometry/material-loss evidence sets the camera, never a pose."""
    if diagnostic is not None and diagnostic.get("samples"):
        eligible = [sample for sample in diagnostic["samples"]
            if sample["valid"] and sample["added_gap_px"] is not None
            and np.isfinite(sample["added_gap_px"]) and sample["added_gap_px"] >= 1.
            and np.asarray(sample["anchor_a_rc"]).shape == (2,)
            and np.isfinite(sample["anchor_a_rc"]).all()]
        if eligible:
            sample = max(eligible, key=lambda item: item["added_gap_px"])
            return np.asarray(sample["anchor_a_rc"], float), "GT-supported anchor with largest measured added gap >=1px; no prediction used"
    valid = np.asarray(clean.contour_valid_a, bool)
    seam = np.flatnonzero(valid & (np.asarray(clean.target_a) >= 0))
    if len(seam):
        return np.asarray(clean.points_rc_a[seam[len(seam) // 2]], float), "middle sampled clean GT-seam token; damage may be elsewhere"
    # Positive samples should have a GT seam, but an explicit fallback is safer
    # than silently aligning the zoom to a successful model prediction.
    points = np.asarray(clean.points_rc_a, float)[valid]
    return points[len(points) // 2], "middle clean A contour token; GT-seam samples unavailable"


def zoom_bounds(anchor_rc, width_px=ZOOM_WIDTH_PX):
    r, c = np.asarray(anchor_rc, float)
    half = width_px / 2
    return (c - half, c + half), (r + half, r - half)


def _image_mask(ax, mask, offset, color, alpha):
    rgba = np.empty(mask.shape + (4,), dtype=np.uint8)
    rgba[:] = np.round(np.asarray(to_rgba(color)) * 255).astype(np.uint8)
    rgba[:, :, 3] = mask.astype(np.uint8) * round(alpha * 255)
    r, c = offset
    h, w = mask.shape
    ax.imshow(rgba, origin="upper", interpolation="nearest",
        extent=(c - .5, c + w - .5, r + h - .5, r - .5), zorder=2)


def _outline(ax, mask, offset, color, style="solid", linewidth=.85):
    r, c = offset
    ax.contour(np.arange(-1, mask.shape[1] + 1) + c, np.arange(-1, mask.shape[0] + 1) + r,
        np.pad(mask.astype(float), 1), levels=[.5], colors=[color],
        linestyles=[style], linewidths=linewidth, zorder=4)


def _draw_fragment(ax, clean, damaged, offset, color, label, *, show_reference):
    if show_reference:
        _image_mask(ax, clean & ~damaged, offset, REMOVED, .9)
        _outline(ax, clean, offset, GREY, "dashed", .85)
    _image_mask(ax, damaged, offset, color, .63)
    _outline(ax, damaged, offset, color, linewidth=1.)
    rr, cc = np.nonzero(damaged)
    return (float(cc.mean() + offset[1]), float(rr.mean() + offset[0]), label)


def _inside(bounds, x, y):
    return bounds[0][0] <= x <= bounds[0][1] and bounds[1][1] <= y <= bounds[1][0]


def _intersects(bounds, box):
    return not (box[1] < bounds[0][0] or box[0] > bounds[0][1]
                or box[3] < bounds[1][1] or box[2] > bounds[1][0])


def render_case(clean, student, row, receipt, output_path, *, case_number=1, diagnostic=None):
    ca, cb, sa, sb = (_mask(getattr(sample, "mask_" + side))
                      for sample in (clean, student) for side in "ab")
    if np.any(sa & ~ca) or np.any(sb & ~cb):
        raise ValueError("stress rendering requires inward material loss, not added material")
    gt = np.asarray(row["target_translation_rc"], float)
    if gt.shape != (2,) or not np.isfinite(gt).all():
        raise ValueError("positive selected cases require original finite GT placement")
    pred = prediction_translation(row)
    whole = shared_whole_bounds(ca, cb, gt, pred)
    anchor, zoom_rule = choose_zoom_anchor(clean, diagnostic)
    zoom = zoom_bounds(anchor)
    score, threshold = float(row["classification"]["fused"]), float(receipt["original_fused_threshold"])
    accepted = score >= threshold
    delta = pred - gt if pred is not None else None
    error = float(np.linalg.norm(delta)) if delta is not None else None
    probe = not receipt["full_test"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GREY,
        "xtick.color": INK, "ytick.color": INK})
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    descriptions = (("Whole: clean at original GT", gt, False, whole),
                    ("GT placement: damaged seam zoom", gt, True, zoom),
                    ("Saved prediction: same GT zoom", pred, True, zoom),
                    ("Whole: saved damaged prediction", pred, True, whole))
    panel_records = []
    for index, (ax, (title, t, damaged, bounds)) in enumerate(zip(axes, descriptions)):
        first, second = (sa, sb) if damaged else (ca, cb)
        labels = [_draw_fragment(ax, ca, first, np.zeros(2), BLUE, "A", show_reference=damaged)]
        if t is not None:
            labels.append(_draw_fragment(ax, cb, second, -t, GOLD, "B", show_reference=damaged))
        else:
            ax.text(.5, .54, "INVALID POSE\nB prediction not drawn", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color=INK,
                bbox=dict(facecolor="white", edgecolor=GREY, alpha=.95))
        ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1]); ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("white")
        ax.tick_params(labelsize=8)
        ax.set_xlabel("column / x (model px)")
        ax.set_ylabel("row / y (model px)")
        ax.set_title(title, fontsize=10, pad=11)
        for x, y, label in labels:
            if _inside(bounds, x, y):
                ax.text(x, y, label, ha="center", va="center", fontsize=13, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.8), zorder=7)
        if index in (0, 3):
            ax.add_patch(Rectangle((zoom[0][0], zoom[1][1]), ZOOM_WIDTH_PX, ZOOM_WIDTH_PX,
                fill=False, edgecolor=INK, linestyle=":", linewidth=1.1, zorder=6))
        if index == 2 and pred is not None and not _intersects(zoom, mask_bounds(sb, -pred)):
            ax.text(.5, .04, "B is outside this fixed GT zoom", transform=ax.transAxes,
                ha="center", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=.95))
        panel_records.append(dict(title=title, x_limits=ax.get_xlim(), y_limits=ax.get_ylim(),
            a_offset_rc=[0., 0.], b_offset_rc=-t if t is not None else None,
            equal_aspect=True, b_prediction_drawn=t is not None))
    depth = receipt["requested_max_inward_erosion_depth_px"]
    main = "Case %02d | %s | max inward erosion %g px (not measured gap width)" % (case_number, receipt["model_label"].upper(), depth)
    scope = ("PROBE: %d / %d TEST pairs; illustrative only, not a benchmark" %
        (receipt["evaluated_pair_count"], receipt["total_test_pair_count"])) if probe else "Full TEST source; examples are not a success-rate estimate"
    fig.suptitle(main + "\n" + str(row["pair_id"]) + "\n" + scope, fontsize=11, y=.985)
    fig.legend(handles=[Patch(facecolor=BLUE, alpha=.63, label="A: fixed"),
        Patch(facecolor=GOLD, alpha=.63, label="B: offset = -t"),
        Line2D([], [], color=GREY, linestyle="--", label="Pre-erosion outline at panel placement"),
        Patch(facecolor=REMOVED, label="Removed material")], loc="upper center", bbox_to_anchor=(.5, .858),
        frameon=False, ncol=4, fontsize=8)
    pose_text = ("Pose INVALID (not zero displacement)" if error is None else
        "Pose L2 %.2f px | delta-t: dx(col) %+.2f, dy(row) %+.2f px" % (error, delta[1], delta[0]))
    selection_note = ("GT-seam damage selected; zoom chosen using GT damage only" if diagnostic is not None else
        "Any-edge damage selected; the true seam may be unchanged")
    footer = ("Native pair classifier %s | score %.6f vs frozen VAL threshold %.6f  ||  %s\n" %
        ("PASS" if accepted else "REJECT", score, threshold, pose_text))
    footer += selection_note + ".  Zoom = 80 x 80 model px; same coordinates, no recentering of failures.\n"
    footer += "GT is display/evaluation only. No rotation; original 800-pixel model frame. Grey is missing material, not predicted paper."
    fig.text(.5, .018, footer, ha="center", va="bottom", fontsize=8.5, linespacing=1.5)
    fig.subplots_adjust(left=.04, right=.99, bottom=.23, top=.73, wspace=.30)
    fig.savefig(output_path, dpi=170, facecolor="white")
    plt.close(fig)
    return _clean(dict(pair_id=row["pair_id"], image=str(Path(output_path).resolve()),
        whole_limits=whole, zoom_limits=zoom, zoom_anchor_rc=anchor, zoom_rule=zoom_rule,
        panels=panel_records, gt_translation_rc=gt, predicted_translation_rc=pred,
        predicted_b_offset_rc=-pred if pred is not None else None,
        native_score=score, frozen_val_threshold=threshold, pair_accepted=accepted,
        error_xy_px=[delta[1], delta[0]] if delta is not None else None, error_l2_px=error,
        probe_only=probe, scope_text=scope, selection_caveat=selection_note,
        head_scores=row.get("head_scores"), e3_thresholds=receipt.get("e3_thresholds")))


def run(args):
    rows, receipt = read_evaluation(args.evaluation)
    diagnostics, diag_manifest = (None, None)
    if args.seam_diagnostics:
        diagnostics, diag_manifest = read_diagnostics(args.seam_diagnostics, receipt, rows)
    selected, selection = select_cases(rows, args.max_cases, diagnostics)
    dataset_root = Path(args.dataset).resolve()
    manifest_path = dataset_root / "pairs" / "test.jsonl"
    if _sha256(manifest_path) != receipt["test_manifest_sha256"]:
        raise ValueError("rendering dataset is not the saved evaluation TEST manifest")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = [json.loads(line) for line in stream if line.strip()]
    by_id = {row["pair_id"]: (i, row) for i, row in enumerate(manifest)}
    for row in rows:
        if row["pair_id"] not in by_id:
            raise ValueError("saved prediction is outside the original TEST manifest")
    base = RachelPairDataset(dataset_root, "test")
    stress = RachelGapStressDataset(base, max_depth_px=receipt["requested_max_inward_erosion_depth_px"],
        seed=receipt["seed"], cache_dir=args.cache_dir)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for number, row in enumerate(selected, 1):
        index, _ = by_id[row["pair_id"]]
        clean, rebuilt = base[index], stress[index]
        if (row["fragment_a"] != clean.fragment_a_token or row["fragment_b"] != clean.fragment_b_token
                or not bool(clean.label) or not np.array_equal(clean.translation_a_to_b_rc, row["target_translation_rc"])):
            raise ValueError("selected saved pair differs from its original TEST identity/GT")
        for side in "ab":
            observed, saved = rebuilt.report["side_" + side], row["stress_report"]["side_" + side]
            for key in ("applied", "skipped", "removed_area_px", "config", "seed", "fragment_id"):
                if observed[key] != saved[key]:
                    raise ValueError("reconstructed selected stress recipe/report differs: " + key)
        diagnostic = None
        if diagnostics is not None:
            from staging.pairwise_v0_2.pairwise_data.rachel_seam_gap_diagnostics import seam_gap_diagnostics
            diagnostic = seam_gap_diagnostics(clean, rebuilt.student, include_samples=True)
            if diagnostic.get("actual_seam_damaged") is not True:
                raise ValueError("selected GT-seam damage is not reproduced under the saved input recipe")
        record = render_case(clean, rebuilt.student, row, receipt, output / ("case_%02d.png" % number),
            case_number=number, diagnostic=diagnostic)
        record.update(actual_changed=rebuilt.report["changed_pair"],
            actual_seam_damaged=diagnostic.get("actual_seam_damaged") if diagnostic else None,
            seam_diagnostics={key: value for key, value in diagnostic.items() if key != "samples"} if diagnostic else None,
            fragment_a=row["fragment_a"], fragment_b=row["fragment_b"])
        records.append(record)
    selection.update(schema_version=SCHEMA, status="complete", cases=records,
        evaluation=str(Path(args.evaluation).resolve()), receipt=receipt,
        dataset_root=str(dataset_root), cache_dir=str(Path(args.cache_dir).resolve()),
        seam_diagnostics=str(Path(args.seam_diagnostics).resolve()) if args.seam_diagnostics else None,
        diagnostics_full_test=diag_manifest.get("full_test") if diag_manifest else None,
        coordinate_convention="t=b-a; B placement=-t; no rotation; original model frame",
        source_masks="selected original TEST inputs plus deterministic same-recipe stress reconstruction; no model inference",
        visual_contract=dict(surface="static Matplotlib spatial mask panels", palette_roots=[BLUE, GOLD],
            neutral_reference=GREY, outcome_selection=False, zoom_width_model_px=ZOOM_WIDTH_PX,
            whole_views_share_bounds=True, zoom_views_share_gt_bounds=True))
    _write_json(output / "selection.json", selection)
    lines = ["# Gap-stress mask cases", "", selection["rule"] + ".",
        "", "%d eligible cases; showing %d (requested %d)." %
        (selection["candidate_count"], len(records), args.max_cases), "",
        "PROBE — illustrative only, not a full benchmark." if not receipt["full_test"] else
        "Examples from a full TEST evaluation; this gallery does not estimate success rates.", "",
        "Blue A is fixed; gold B is placed at −t. Grey/dashed marks are missing material/pre-erosion boundaries.",
        "The two whole views share bounds; the two 80px zooms share GT-derived bounds. Failed predictions are not recentered.",
        "GT is only a display/evaluation reference. Requested erosion depth is not the actual gap width.", ""]
    for number, record in enumerate(records, 1):
        lines += ["## Case %02d — %s" % (number, record["pair_id"]), "",
            "![Mask placement case](case_%02d.png)" % number, ""]
    if not records:
        lines += ["No cases met the pre-outcome selection rule; no substitutes were selected.", ""]
    with (output / "index.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps(dict(status="complete", output=str(output), selected=len(records),
                         eligible=selection["candidate_count"], full_test_source=receipt["full_test"])), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-cases", type=int, default=6)
    p.add_argument("--seam-diagnostics", type=Path)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
