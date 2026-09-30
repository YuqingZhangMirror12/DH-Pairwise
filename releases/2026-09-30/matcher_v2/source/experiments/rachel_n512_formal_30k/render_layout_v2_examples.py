"""Render fixed-order synthetic test examples of four translation layouts.

Only saved predictions and dataset masks/targets are read.  No checkpoint,
model, or GPU is used.  Example selection is independent of prediction quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


COLOR_A = "#1874B5"
COLOR_B = "#E68128"


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _selected_decoder(results_dir, explicit_freeze=None):
    sources = ([Path(explicit_freeze)] if explicit_freeze else []) + [
        results_dir / "validation_freeze.json", results_dir / "summary.json",
    ]
    for path in sources:
        if path.is_file():
            name = _read_json(path).get("selected_full_decoder")
            if isinstance(name, str) and name:
                return name, path
    raise ValueError("results need validation_freeze.json or summary.json with selected_full_decoder")


def _mask(root, fragment):
    path = root / fragment["model_mask_path"]
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"))
    mask = values > 0
    if mask.ndim != 2 or not mask.any():
        raise ValueError("model mask must be a non-empty 2-D mask: " + str(path))
    return mask


def _translation(value):
    if value is None:
        return None
    vector = np.asarray(value, dtype=float)
    return vector if vector.shape == (2,) and np.isfinite(vector).all() else None


def _bounds(mask, offset_rc):
    rows, columns = np.nonzero(mask)
    return np.asarray([
        columns.min() - 0.5 + offset_rc[1],
        columns.max() + 0.5 + offset_rc[1],
        rows.min() - 0.5 + offset_rc[0],
        rows.max() + 0.5 + offset_rc[0],
    ])


def _shared_limits(mask_a, mask_b, translations):
    bounds = [_bounds(mask_a, np.zeros(2))]
    bounds.extend(_bounds(mask_b, -translation) for translation in translations
                  if translation is not None)
    bounds = np.asarray(bounds)
    x0, x1 = bounds[:, 0].min(), bounds[:, 1].max()
    y0, y1 = bounds[:, 2].min(), bounds[:, 3].max()
    padding = max(8.0, 0.06 * max(x1 - x0, y1 - y0))
    return (x0 - padding, x1 + padding), (y1 + padding, y0 - padding)


def _outline(axis, mask, offset_rc, color):
    # Padding closes contours even when foreground reaches a canvas edge.
    values = np.pad(mask.astype(float), 1)
    columns = np.arange(-1, mask.shape[1] + 1, dtype=float) + offset_rc[1]
    rows = np.arange(-1, mask.shape[0] + 1, dtype=float) + offset_rc[0]
    axis.contour(columns, rows, values, levels=[0.5], colors=[color], linewidths=1.3)


def _short_pair_id(pair_id):
    return pair_id if len(pair_id) <= 23 else pair_id[:11] + "..." + pair_id[-9:]


def render(args):
    root = Path(args.dataset)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    selected, selected_source = _selected_decoder(results_dir, args.validation_freeze)
    rows_path = next((results_dir / name for name in ("pair_results.jsonl", "rows.jsonl")
                      if (results_dir / name).is_file()), None)
    if rows_path is None:
        raise ValueError("results directory needs pair_results.jsonl or rows.jsonl")
    result_rows = _read_jsonl(rows_path)
    by_id = {row["pair_id"]: row for row in result_rows}
    if len(by_id) != len(result_rows):
        raise ValueError("prediction rows contain duplicate pair IDs")
    manifest_path = root / "pairs" / "test.jsonl"
    manifest = _read_jsonl(manifest_path)
    positives = sorted((row for row in manifest if row["label"] is True),
                       key=lambda row: row["pair_id"])
    if len(positives) < args.count:
        raise ValueError("test manifest has fewer positive pairs than the requested count")
    chosen = positives[:args.count]
    missing = [row["pair_id"] for row in chosen if row["pair_id"] not in by_id]
    if missing:
        raise ValueError("fixed selected examples are missing predictions: " + ", ".join(missing))

    method_keys = ("ground_truth", "full_original", selected, "full_with_shred_matching_layout")
    method_labels = ("Ground truth", "Original Full", "New Full (selected)", "Shred matching layout")
    examples = []
    for manifest_row in chosen:
        pair_id = manifest_row["pair_id"]
        prediction = by_id[pair_id]
        with np.load(root / manifest_row["correspondence_path"], allow_pickle=False) as target:
            gt = _translation(target["translation_a_to_b_rc"])
        if gt is None:
            raise ValueError("positive example has no finite translation GT: " + pair_id)
        saved_gt = _translation(prediction.get("target_translation_rc"))
        if saved_gt is not None and not np.allclose(saved_gt, gt, rtol=0, atol=1e-4):
            raise ValueError("saved GT and dataset coordinates differ: " + pair_id)
        translations = [gt]
        diagnostics = [{"method": "ground_truth", "valid": True, "translation_l2_px": 0.0}]
        for method in method_keys[1:]:
            layout = prediction.get("layouts", {}).get(method)
            if layout is None:
                raise ValueError("selected pair lacks layout " + method + ": " + pair_id)
            translation = _translation(layout.get("translation_rc")) if layout.get("valid") is True else None
            translations.append(translation)
            diagnostics.append({
                "method": method,
                "valid": translation is not None,
                "translation_l2_px": float(np.linalg.norm(translation - gt)) if translation is not None else None,
            })
        mask_a = _mask(root, manifest_row["fragment_a"])
        mask_b = _mask(root, manifest_row["fragment_b"])
        limits = _shared_limits(mask_a, mask_b, translations)
        examples.append((pair_id, mask_a, mask_b, translations, diagnostics, limits))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(args.count, 4, figsize=(15.6, 3.7 * args.count),
                                squeeze=False, sharex="row", sharey="row")
    selection_rows = []
    for row_index, (pair_id, mask_a, mask_b, translations, diagnostics, limits) in enumerate(examples):
        selection_rows.append({
            "pair_id": pair_id,
            "layouts": diagnostics,
            "shared_x_limits": list(limits[0]),
            "shared_y_limits": list(limits[1]),
        })
        for column, (label, translation, diagnostic) in enumerate(zip(method_labels, translations, diagnostics)):
            axis = axes[row_index, column]
            _outline(axis, mask_a, np.zeros(2), COLOR_A)
            if translation is not None:
                _outline(axis, mask_b, -translation, COLOR_B)
                error = "error = {:.2f} px".format(diagnostic["translation_l2_px"])
            else:
                error = "FAILED"
                axis.text(0.5, 0.5, "FAILED\nNo valid B placement", transform=axis.transAxes,
                          ha="center", va="center", color="#A52727", fontsize=12,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
            axis.set_title("{} | {}\n{}".format(label, error, _short_pair_id(pair_id)), fontsize=9)
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.16, linewidth=0.5)
            axis.tick_params(labelsize=7)
            axis.set_xlabel("Column in A frame (px)", fontsize=8)
            if column == 0:
                axis.set_ylabel("Row in A frame (px)", fontsize=8)
    figure.suptitle("Pairwise translation layouts | rotation fixed to zero", fontsize=16, y=0.993)
    figure.text(0.5, 0.973,
                "First {} positive test pairs sorted by pair_id; selected decoder: {}".format(args.count, selected),
                ha="center", fontsize=10)
    figure.legend(handles=[Line2D([0], [0], color=COLOR_A, lw=2, label="Fragment A (fixed)"),
                           Line2D([0], [0], color=COLOR_B, lw=2, label="Fragment B (offset = -t A-to-B)")],
                  loc="lower center", bbox_to_anchor=(0.5, 0.009), ncol=2, frameon=False, fontsize=10)
    figure.text(0.5, 0.003,
                "Each row shares the same coordinate limits across all four panels. Errors are translation L2 distances to GT.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0.0, 0.034, 1.0, 0.958), h_pad=2.0, w_pad=1.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "layout_v2_examples.png"
    pdf_path = output_dir / "layout_v2_examples.pdf"
    figure.savefig(png_path, dpi=args.dpi, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)
    selection = {
        "selection_rule": "first N positive rows of the entire synthetic test manifest sorted lexicographically by pair_id",
        "selection_uses_prediction_success": False,
        "requested_count": args.count,
        "selected_pair_ids": [row["pair_id"] for row in chosen],
        "eligible_positive_pair_ids": [row["pair_id"] for row in positives],
        "selected_full_decoder": selected,
        "selected_decoder_source": str(selected_source),
        "test_manifest": str(manifest_path),
        "prediction_rows": str(rows_path),
        "rotation_estimated": False,
        "placement_convention": "A stays fixed; B offset in A frame is -t_a_to_b_rc",
        "coordinate_limits_shared_across_methods_per_pair": True,
        "examples": selection_rows,
        "png": str(png_path), "pdf": str(pdf_path),
    }
    with (output_dir / "selection.json").open("w", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"png": str(png_path), "pdf": str(pdf_path),
                      "selection": str(output_dir / "selection.json"),
                      "pair_ids": selection["selected_pair_ids"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Synthetic Rachel dataset root")
    parser.add_argument("--results-dir", required=True, help="Completed test predictions directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-freeze", help="Optional explicit validation_freeze.json")
    parser.add_argument("--count", type=int, choices=(4, 6), default=6)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    render(args)


if __name__ == "__main__":
    main()
