"""Small TRAIN-only mask-weathering demonstration, NOT a trained-model result.

Choose four pairs in each label x tiny/non-tiny group of the frozen matched24k
TRAIN manifest. Apply the same label-blind single-fragment augmentation to both
endpoints. Preserve the 800-canvas frame and GT translation, re-extract contours,
and explicitly do not reuse old correspondence arrays as training targets.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from staging.pairwise_v0_2.pairwise_data.rachel_composite_training import CompositeRachelPairDataset
from staging.pairwise_v0_2.pairwise_data.rachel_edge_weathering import EdgeWeatheringConfig, weather_fragment_edges
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour


SEED = 260910
DEPTHS = (0.0, 2.0, 4.0)


def choose_indices(entries, per_group=4):
    groups = {}
    for index, item in enumerate(entries):
        row = item["row"]
        if row["split"] != "train":
            raise ValueError("prototype only accepts TRAIN entries")
        sizes = [row["fragment_" + side]["foreground_area"] for side in "ab"]
        key = (bool(row["label"]), min(sizes) / max(sizes) < .25)
        rank = hashlib.sha256((str(SEED) + row["pair_id"]).encode()).hexdigest()
        groups.setdefault(key, []).append((rank, index))
    if len(groups) != 4 or any(len(group) < per_group for group in groups.values()):
        raise ValueError("need four nonempty label x area-ratio strata")
    return [index for key in sorted(groups) for _, index in sorted(groups[key])[:per_group]]


def save_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def plot_examples(examples, output, filename="seam_weathering_examples.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(len(examples), 3, figsize=(13.5, 4.5 * len(examples)), squeeze=False)
    for row_index, (sample, variants, tiny) in enumerate(examples):
        target = np.asarray(sample.target_a)
        seam = sample.points_rc_a[(target >= 0) & sample.contour_valid_a]
        if not len(seam):
            raise ValueError("positive visualization requires original seam annotation")
        # Select a visible altered seam vicinity for illustration only. The
        # corruption function never receives this annotation or the pair label.
        translation = np.asarray(sample.translation_a_to_b_rc, float)
        from scipy.ndimage import distance_transform_edt
        distance_to_change = np.full(len(seam), np.inf)
        for side_index in (0, 1):
            removed = variants[0.0][side_index] & ~variants[4.0][side_index]
            if removed.any():
                side_seam = seam + (translation if side_index else 0)
                rrcc = np.clip(np.rint(side_seam).astype(int), 0, 799)
                distance = distance_transform_edt(~removed)
                distance_to_change = np.minimum(distance_to_change, distance[rrcc[:, 0], rrcc[:, 1]])
        center = seam[int(np.argmin(distance_to_change))] if np.isfinite(distance_to_change).any() else seam[len(seam)//2]
        for col_index, depth in enumerate(DEPTHS):
            ax = axes[row_index, col_index]
            for side_index, color in enumerate(("#4c91c5", "#dc914e")):
                mask = variants[depth][side_index]
                original = variants[0.0][side_index]
                shift = np.zeros(2) if side_index == 0 else -translation
                extent = (shift[1]-.5, shift[1]+799.5, shift[0]+799.5, shift[0]-.5)
                ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([color]),
                          vmin=0, vmax=1, origin="upper", extent=extent, interpolation="nearest")
                if depth:
                    ax.contour(original.astype(float), levels=[.5], colors="#555555", linewidths=.7,
                               linestyles="--", origin="upper", extent=extent)
            ax.set_xlim(center[1]-42, center[1]+42)
            ax.set_ylim(center[0]+42, center[0]-42)
            ax.set_aspect("equal")
            ax.set_title(("Clean" if depth == 0 else "Weathering depth <= %g px" % depth)
                         + ("\narea ratio < 1:4" if tiny else "\nordinary area ratio"))
            ax.set_xlabel("Original canvas x (px)")
            ax.set_ylabel("Original canvas y (px)")
    fig.suptitle("Synthetic TRAIN seam zoom: fixed GT placement, NOT model output\n"
                 "Blue / orange: remaining paper; dashed: original boundary; blank: material gap", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output / filename, dpi=130)
    plt.close(fig)


def render_saved(manifest, output):
    """Illustrate applied cases, including unilateral weathering; no regeneration.

    Original summary and first figure remain unchanged. Selection is explicit:
    among the sampled positives, prefer two changed endpoints, otherwise one,
    then the original sampled order, separately for tiny/ordinary size ratios.
    """
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    if summary.get("status") != "complete" or summary.get("split") != "train":
        raise ValueError("need the complete TRAIN prototype")
    if (output / "seam_weathering_examples_v2.png").exists():
        raise FileExistsError("do not overwrite the revised illustration")
    entries = json.loads(Path(manifest).read_text())["entries"]
    index_by_id = {row["row"]["pair_id"]: i for i, row in enumerate(entries)}
    dataset = CompositeRachelPairDataset(manifest)
    examples, selected = [], []
    for tiny in (False, True):
        candidates = [row for row in summary["records"] if row["label"] and row["max_depth_px"] == 4
                      and (row["area_ratio"] < .25) == tiny]
        row = max(candidates, key=lambda row: sum(d["applied"] for d in row["diagnostics"]))
        variants = {}
        for saved in summary["records"]:
            if saved["pair_id"] == row["pair_id"]:
                variants[saved["max_depth_px"]] = [np.asarray(Image.open(output / (d["artifact"] + ".png"))) > 0
                                                   for d in saved["diagnostics"]]
        examples.append((dataset[index_by_id[row["pair_id"]]], variants, tiny))
        selected.append(dict(pair_id=row["pair_id"], tiny=tiny,
            strong_endpoint_statuses=[d["status"] for d in row["diagnostics"]]))
    plot_examples(examples, output, "seam_weathering_examples_v2.png")
    save_json(output / "illustration_selection.json", dict(status="complete", selected=selected,
        rule="sampled TRAIN positives; prefer more applied endpoints, then existing order; no model score selection",
        plot="zoom near a changed part of an annotated original seam; GT placement, not predicted layout",
        original_summary_modified=False, augmentation_rerun=False))
    return {"status": "complete", "sample_count": len(examples)}


def run(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("use a new prototype output directory")
    payload = json.loads(manifest.read_text())
    if payload.get("split") != "train":
        raise ValueError("must use TRAIN")
    indices = choose_indices(payload["entries"])
    dataset = CompositeRachelPairDataset(manifest)
    output.mkdir(parents=True, exist_ok=False)
    records, cache, examples, visualized = [], {}, [], set()
    for index in indices:
        sample = dataset[index]
        row = payload["entries"][index]["row"]
        area = [float(row["fragment_" + side]["foreground_area"]) for side in "ab"]
        tiny = min(area)/max(area) < .25
        variants = {}
        for depth in DEPTHS:
            masks, diagnostics, contours = [], [], []
            for side in "ab":
                token = getattr(sample, "fragment_" + side + "_token")
                original = np.asarray(getattr(sample, "mask_" + side))[0].astype(bool)
                key = (token, depth)
                if key not in cache:
                    config = EdgeWeatheringConfig(max_depth_px=depth)
                    changed, report = weather_fragment_edges(original, seed=SEED, fragment_id=token, config=config)
                    assert changed.shape == original.shape and not np.any(changed & ~original)
                    points, valid = extract_ordered_outer_contour(changed, cap=512, smoothing_sigma=3)
                    artifact = hashlib.sha256(token.encode()).hexdigest()[:16] + "_d%g" % depth
                    Image.fromarray(changed.astype(np.uint8)*255).save(output / (artifact + ".png"))
                    np.savez_compressed(output / (artifact + ".npz"), points_rc=points, valid=valid)
                    cache[key] = (changed, dict(report, config=asdict(config), artifact=artifact), len(points))
                changed, report, count = cache[key]
                masks.append(changed)
                diagnostics.append(report)
                contours.append(count)
            variants[depth] = masks
            records.append(dict(pair_id=sample.pair_id, label=bool(sample.label), area_ratio=min(area)/max(area),
                max_depth_px=depth, diagnostics=diagnostics, contour_counts=contours,
                translation_rc=sample.translation_a_to_b_rc.tolist() if sample.translation_valid else None,
                original_pair_label_retained_for_prototype_only=True,
                training_correspondence_targets="NOT GENERATED; old token correspondences are invalid after resampling"))
        if sample.label and tiny not in visualized:
            examples.append((sample, variants, tiny))
            visualized.add(tiny)
        print(json.dumps(dict(processed_pairs=len(records)//3, total_pairs=len(indices))), flush=True)
    plot_examples(examples, output)
    summary = dict(status="complete", schema_version="rachel-edge-weathering-prototype/1", seed=SEED,
        source_manifest=str(manifest), split="train", sample_count=len(indices),
        positive_count=sum(bool(payload["entries"][i]["row"]["label"]) for i in indices),
        sampling="4 each: positive/negative x area ratio below/above .25; hash-selected, not model-selected",
        unique_fragment_condition_count=len(cache), max_depths_px=DEPTHS, records=records,
        model_executed=False, model_trained=False, test_or_real_opened=False, source_files_modified=False,
        mask_and_contour_regenerated=True, original_800_frame_preserved=True, translation_targets_recentered=False,
        recall_improvement_measured=False, training_ready=False,
        caveat="Geometry/augmentation prototype only. TRAIN examples are not generalization evaluation. "
               "Fresh lineage-aware partial-correspondence targets and gap-tolerant training remain future work.",
        figure="seam_weathering_examples.png")
    save_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--render-saved", action="store_true")
    args = parser.parse_args()
    result = render_saved(args.manifest, args.output) if args.render_saved else run(args.manifest, args.output)
    print(json.dumps({"status": result["status"], "sample_count": result["sample_count"]}), flush=True)
