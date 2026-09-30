"""Compare four fixed-order real fragment layouts from saved predictions only.

Select the first four strict positive pair IDs lexicographically, independently
of prediction success.  The prepared real masks and completed prediction file
contain all required inputs. No model, checkpoint, or GPU is opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k.render_layout_v2_examples import (
    COLOR_A, COLOR_B, Line2D, _outline, _read_json, _read_jsonl,
    _selected_decoder, _shared_limits, _short_pair_id, _translation, np, plt,
)


def render(args):
    cache, results, destination = map(Path, (args.prepared_cache, args.results_dir, args.output_dir))
    metadata = _read_json(cache / "manifest.json")
    eligible = sorted((r for r in metadata["pairs"] if r["strict"] and r["label"]),
                      key=lambda r: r["pair_id"])
    chosen = eligible[:4]
    if len(chosen) != 4:
        raise ValueError("prepared real population has fewer than four strict positive pairs")
    rows_path = results / "pair_results.jsonl"
    rows = _read_jsonl(rows_path)
    predictions = {r["pair_id"]: r for r in rows}
    if len(predictions) != len(rows):
        raise ValueError("prediction pair IDs are duplicated")
    selected, selection_source = _selected_decoder(results, args.validation_freeze)
    methods = ("ground_truth", "full_original", selected, "full_with_shred_matching_layout")
    labels = ("Ground truth", "Original Full", "New Full (selected)", "Shred fine layout")
    indices = {key: i for i, key in enumerate(metadata["fragment_ids"])}
    examples = []
    with np.load(cache / "inputs.npz", allow_pickle=False) as archive:
        packed_masks = archive["packed_masks"]
        for pair in chosen:
            pair_id = pair["pair_id"]
            prediction = predictions[pair_id]
            if (prediction["fragment_a"] != pair["fragment_a_id"]
                    or prediction["fragment_b"] != pair["fragment_b_id"]):
                raise ValueError("prediction endpoint order differs for " + pair_id)
            gt = _translation(prediction.get("target_translation_rc"))
            if gt is None:
                raise ValueError("completed positive prediction lacks finite GT: " + pair_id)
            translations = [gt]
            diagnostics = [{"method": "ground_truth", "valid": True,
                            "translation_rc": gt.tolist(), "offset_b_in_a_rc": (-gt).tolist(),
                            "translation_l2_px": 0.0}]
            for method in methods[1:]:
                layout = prediction["layouts"][method]
                translation = _translation(layout["translation_rc"]) if layout["valid"] else None
                translations.append(translation)
                diagnostics.append({"method": method, "valid": translation is not None,
                    "translation_rc": translation.tolist() if translation is not None else None,
                    "offset_b_in_a_rc": (-translation).tolist() if translation is not None else None,
                    "translation_l2_px": float(np.linalg.norm(translation - gt)) if translation is not None else None})
            masks = [np.unpackbits(packed_masks[indices[pair[key]]], axis=1).astype(bool)
                     for key in ("fragment_a_id", "fragment_b_id")]
            limits = _shared_limits(*masks, translations)
            examples.append((pair, prediction, masks, translations, diagnostics, limits))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(4, 4, figsize=(15.6, 14.8), squeeze=False,
                                sharex="row", sharey="row")
    records = []
    for row_index, (pair, prediction, masks, translations, diagnostics, limits) in enumerate(examples):
        mask_a, mask_b = masks
        records.append({"pair_id": pair["pair_id"], "case_id": pair["case_cluster"],
            "fragment_a": pair["fragment_a_id"], "fragment_b": pair["fragment_b_id"],
            "classification": prediction["classification"], "layouts": diagnostics,
            "shared_x_limits": list(limits[0]), "shared_y_limits": list(limits[1])})
        for column, (label, translation, diagnostic) in enumerate(zip(labels, translations, diagnostics)):
            axis = axes[row_index, column]
            _outline(axis, mask_a, np.zeros(2), COLOR_A)
            if translation is not None:
                _outline(axis, mask_b, -translation, COLOR_B)
                error = "error = {:.2f} px".format(diagnostic["translation_l2_px"])
            else:
                error = "FAILED"
                axis.text(.5, .5, "FAILED\nNo valid B placement", transform=axis.transAxes,
                          ha="center", va="center", color="#A52727", fontsize=12,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
            axis.set_title("{} | {}\n{}".format(label, error, _short_pair_id(pair["pair_id"])), fontsize=9)
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=.16, linewidth=.5)
            axis.tick_params(labelsize=7)
            axis.set_xlabel("Column in A frame (px)", fontsize=8)
            if column == 0:
                axis.set_ylabel("Row in A frame (px)", fontsize=8)
    figure.suptitle("Real fragment translation layouts | rotation fixed to zero", fontsize=16, y=.993)
    figure.text(.5, .973, "First 4 strict positive pairs sorted by pair_id; selected decoder: " + selected,
                ha="center", fontsize=10)
    figure.legend(handles=[Line2D([0], [0], color=COLOR_A, lw=2, label="Fragment A (fixed)"),
                           Line2D([0], [0], color=COLOR_B, lw=2, label="Fragment B (offset = -t A-to-B)")],
                  loc="lower center", bbox_to_anchor=(.5, .009), ncol=2, frameon=False, fontsize=10)
    figure.text(.5, .003, "Each row shares one coordinate frame and scale. Errors are translation L2 distances to GT.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0., .034, 1., .958), h_pad=2., w_pad=1.)
    destination.mkdir(parents=True, exist_ok=True)
    png, pdf = destination / "real_layout_v2_examples.png", destination / "real_layout_v2_examples.pdf"
    figure.savefig(png, dpi=args.dpi, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    selection = {"selection_rule": "first 4 strict positive prepared-manifest rows sorted lexicographically by pair_id",
        "selection_uses_prediction_success": False, "selected_pair_ids": [r["pair_id"] for r in chosen],
        "eligible_positive_pair_ids": [r["pair_id"] for r in eligible], "selected_full_decoder": selected,
        "selected_decoder_source": str(selection_source), "prepared_cache": str(cache), "prediction_rows": str(rows_path),
        "rotation_estimated": False, "placement_convention": "A fixed; offset B into A = -t_a_to_b_rc",
        "coordinate_limits_shared_across_methods_per_pair": True, "examples": records,
        "png": str(png), "pdf": str(pdf)}
    with (destination / "real_selection.json").open("w", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"png": str(png), "pdf": str(pdf), "selection": str(destination / "real_selection.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-cache", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-freeze")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    render(args)


if __name__ == "__main__":
    main()
