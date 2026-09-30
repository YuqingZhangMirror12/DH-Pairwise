"""Render 20 successful and 20 failed real placements from frozen predictions.

RGB is used for display only. Its alpha is mapped into the exact prepared mask
frame and checked on the selected fragments; predictions are never modified.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = 300_000_000
BLUE, ORANGE = "#1874B5", "#E68128"


def read_json(path):
    return json.loads(Path(path).read_text())


def translation(row):
    value = row["layout"]["translation_rc"]
    if not row["layout"]["valid"] or value is None:
        return None
    vector = np.asarray(value, dtype=float)
    return vector if vector.shape == (2,) and np.isfinite(vector).all() else None


def error(row):
    vector = translation(row)
    return float(np.linalg.norm(vector - row["target_translation_rc"])) if vector is not None else float("inf")


def select_quantiles(rows, count, excluded_cases):
    """Evenly spaced error ranks, preferring distinct case IDs globally."""
    ordered = sorted(rows, key=lambda row: (error(row), row["pair_id"]))
    if len({row["case_cluster"] for row in ordered} - excluded_cases) < count:
        raise ValueError("Too few distinct cases for the requested gallery")
    chosen, used = [], set(excluded_cases)
    for target in np.linspace(0, len(ordered) - 1, count):
        indexes = sorted(range(len(ordered)), key=lambda i: (abs(i - target), i))
        index = next(i for i in indexes if ordered[i]["case_cluster"] not in used)
        chosen.append(ordered[index])
        used.add(ordered[index]["case_cluster"])
    return sorted(chosen, key=lambda row: (error(row), row["pair_id"]))


def model_rgb(path, canvas_wh, expected_mask):
    """Use the training/evaluation alpha resize, tight crop and center pad."""
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        scale = 800.0 / max(canvas_wh)
        shape = (max(1, round(rgba.width * scale)), max(1, round(rgba.height * scale)))
        raw_alpha = np.asarray(rgba.getchannel("A")) >= 128
        alpha = np.asarray(Image.fromarray(raw_alpha).resize(shape, Image.Resampling.NEAREST))
        # RGB is an illustration layer, not a model input.
        resized = np.asarray(rgba.resize(shape, Image.Resampling.BILINEAR)).copy()
    rr, cc = np.nonzero(alpha)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    alpha, resized = alpha[r0:r1, c0:c1], resized[r0:r1, c0:c1]
    h, w = alpha.shape
    r, c = (800 - h) // 2, (800 - w) // 2
    placed_mask = np.zeros((800, 800), dtype=bool)
    placed_mask[r:r+h, c:c+w] = alpha
    if not np.array_equal(placed_mask, expected_mask):
        raise ValueError("Display alpha differs from frozen model mask: " + str(path))
    image = np.zeros((800, 800, 4), dtype=np.uint8)
    resized[:, :, 3] = alpha.astype(np.uint8) * 255
    image[r:r+h, c:c+w] = resized
    return image


def bounds(mask, offset):
    rr, cc = np.nonzero(mask)
    return (cc.min() + offset[1], cc.max() + offset[1],
            rr.min() + offset[0], rr.max() + offset[0])


def shared_limits(masks, translations):
    boxes = [bounds(masks[0], np.zeros(2))]
    boxes += [bounds(masks[1], -t) for t in translations if t is not None]
    lo_x, hi_x = min(v[0] for v in boxes), max(v[1] for v in boxes)
    lo_y, hi_y = min(v[2] for v in boxes), max(v[3] for v in boxes)
    pad = max(12, .05 * max(hi_x - lo_x, hi_y - lo_y))
    return (lo_x - pad, hi_x + pad), (hi_y + pad, lo_y - pad)


def draw_fragment(ax, image, mask, offset, color):
    r, c = offset
    ax.imshow(image, origin="upper", interpolation="nearest",
              extent=(c - .5, c + 799.5, r + 799.5, r - .5))
    padded = np.pad(mask.astype(float), 1)
    ax.contour(np.arange(-1, 801) + c, np.arange(-1, 801) + r,
               padded, levels=[.5], colors=[color], linewidths=.8, zorder=5)


def make_sheet(paths, destination, heading):
    panels = [Image.open(path).convert("RGB") for path in paths]
    width = panels[0].width
    sheet = Image.new("RGB", (width, 100 + sum(panel.height for panel in panels)), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = matplotlib.font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, 32)
    small = ImageFont.truetype(font_path, 23)
    draw.text((28, 14), heading, fill="#172433", font=font)
    draw.text((28, 56), "GT | Baseline 32+64 | Candidate 7+16+32+64   -   blue A fixed / orange B shifted", fill="#415166", font=small)
    y = 100
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(destination)


def render(args):
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cases").mkdir(exist_ok=True)
    saved, prepared = read_json(args.predictions), read_json(Path(args.prepared_cache) / "manifest.json")
    source_manifest, receipt = read_json(args.source_manifest), read_json(args.local_receipt)
    source_cases = {case["case_uid"]: case for case in source_manifest["cases"]}
    indices = {key: i for i, key in enumerate(prepared["fragment_ids"])}
    with np.load(Path(args.prepared_cache) / "inputs.npz", allow_pickle=False) as archive:
        packed_masks = archive["packed_masks"]
    candidate = saved["candidate"]["rows"]
    baseline = {row["pair_id"]: row for row in saved["baseline"]["rows"]}
    if len(candidate) != 508 or len(baseline) != 508:
        raise ValueError("Expected the frozen 508 real positive pairs")
    for row in candidate:
        control = baseline[row["pair_id"]]
        for key in ("fragment_a", "fragment_b", "target_translation_rc", "case_cluster"):
            if row[key] != control[key]:
                raise ValueError("Saved prediction identity/GT mismatch: " + key)
        for observed in (row, control):
            stored = observed["layout"].get("translation_l2_px")
            if stored is not None and not np.isclose(error(observed), stored, atol=1e-6):
                raise ValueError("Saved error differs from recomputed error")
    success_pool = [row for row in candidate if error(row) <= args.threshold]
    failure_pool = [row for row in candidate if error(row) > args.threshold]
    success = select_quantiles(success_pool, 20, set())
    failure = select_quantiles(failure_pool, 20, {row["case_cluster"] for row in success})
    protocols = {key: saved[key]["protocol"] for key in ("baseline", "candidate")}
    records, images, sheets = [], {}, []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    for group, selected in (("success", success), ("failure", failure)):
        image_paths = []
        for number, row in enumerate(selected, 1):
            short = ("S" if group == "success" else "F") + str(number).zfill(2)
            control = baseline[row["pair_id"]]
            case = source_cases[row["case_cluster"]]
            accepted_occ = {occ["occurrence_uid"] for occ in case["occurrences"]
                            if occ["collection"] == case["canonical_collection"]
                            and occ["category"] == case["canonical_category"]}
            occurrence = sorted((occ for occ in receipt["cases"][case["case_uid"]]["occurrences"]
                                 if occ["occurrence_uid"] in accepted_occ), key=lambda occ: occ["occurrence_uid"])[0]
            paths = {Path(path).stem: Path(path) for path in occurrence["fragment_paths"]}
            masks, display_images, fragment_paths = [], [], []
            for field in ("fragment_a", "fragment_b"):
                fragment_id = row[field]
                mask = np.unpackbits(packed_masks[indices[fragment_id]], axis=1).astype(bool)
                path = paths[fragment_id.rsplit("/", 1)[1]]
                if fragment_id not in images:
                    images[fragment_id] = model_rgb(path, case["numeric_metadata"]["canvas_wh"], mask)
                masks.append(mask)
                display_images.append(images[fragment_id])
                fragment_paths.append(str(path))
            relative = Path(occurrence["group_directory"]).relative_to(receipt["dataset_roots"][occurrence["collection"]])
            readable = occurrence["collection"] + "/" + str(relative)
            transforms = [np.asarray(row["target_translation_rc"]), translation(control), translation(row)]
            limits = shared_limits(masks, transforms)
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
            record = {"id": short, "group": group, "pair_id": row["pair_id"],
                      "case_id": row["case_cluster"], "source_case": readable,
                      "fragment_paths": fragment_paths, "area_ratio": row["area_ratio"],
                      "ground_truth_translation_rc": row["target_translation_rc"],
                      "shared_x_limits": limits[0], "shared_y_limits": limits[1]}
            for ax, key, title, t, prediction in zip(axes,
                    ("gt", "baseline", "candidate"),
                    ("Ground truth", "Baseline 32+64", "Candidate 7+16+32+64"),
                    transforms, (None, control, row)):
                draw_fragment(ax, display_images[0], masks[0], np.zeros(2), BLUE)
                if t is not None:
                    draw_fragment(ax, display_images[1], masks[1], -t, ORANGE)
                else:
                    ax.text(.5, .5, "INVALID PLACEMENT", transform=ax.transAxes, ha="center", color="red")
                if prediction is None:
                    subtitle = "Reference; rotation fixed to zero"
                else:
                    score = prediction["classification"]["fused"]
                    accepted = score >= protocols[key]["original_fused_threshold"]
                    measured_error = error(prediction)
                    subtitle = "error {:.2f} px | Pair {} (score {:.3f})".format(measured_error, "PASS" if accepted else "REJECT", score)
                    record[key] = {"translation_rc": t.tolist() if t is not None else None,
                                   "error_px": measured_error if np.isfinite(measured_error) else None,
                                   "pair_score": score, "pair_accepted": accepted}
                ax.set_title(title + "\n" + subtitle, fontsize=10, pad=8)
                ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
                ax.set_aspect("equal"); ax.set_facecolor("#f5f6f8")
                ax.tick_params(labelsize=7)
                ax.set_xlabel("column (800-canvas px)", fontsize=8)
                if key == "gt":
                    ax.set_ylabel("row (800-canvas px)", fontsize=8)
            fig.suptitle("{}  |  {}  |  fragments {} + {}  |  area ratio {:.3f}".format(
                short, readable, Path(fragment_paths[0]).stem, Path(fragment_paths[1]).stem, row["area_ratio"]),
                fontsize=12, fontweight="bold", y=.985)
            fig.subplots_adjust(left=.04, right=.992, top=.80, bottom=.12, wspace=.19)
            output_path = destination / "cases" / (short + ".png")
            fig.savefig(output_path, dpi=args.dpi, facecolor="white")
            plt.close(fig)
            record["image"] = str(output_path.resolve())
            records.append(record); image_paths.append(output_path)
            print("rendered", short, readable, flush=True)
        for page in range(4):
            output_path = destination / "{}_{}-{}.png".format(group, page * 5 + 1, page * 5 + 5)
            make_sheet(image_paths[page*5:page*5+5], output_path,
                       "{} placements: {}-{} / 20 | candidate error {} 10 px".format(
                           group.upper(), page*5+1, page*5+5, "<=" if group == "success" else ">"))
            sheets.append(str(output_path.resolve()))
    summary = {
        "schema": "real-layout-40-case-gallery/1", "positive_population": len(candidate),
        "success_pool": len(success_pool), "failure_pool": len(failure_pool),
        "success_threshold_px": args.threshold,
        "selection": "20 evenly spaced candidate-error ranks per outcome; nearest available rank with a distinct case ID; success first; no repeated cases across the 40 examples",
        "conditioning": "Selected by geometric outcome only, irrespective of classifier acceptance. Not a success-rate estimate.",
        "pixel_frame": "800-pixel parent-canvas scale, tight crop and center pad; not original scan pixels",
        "coordinate_convention": "t_A_to_B = b - a; A fixed and B displayed at -t_A_to_B; rotation zero",
        "rgb_role": "Display only; never used for inference. Selected display alphas equal frozen prepared masks.",
        "sources": {key: saved[key]["source_predictions"] for key in ("baseline", "candidate")},
        "protocols": protocols, "sheets": sheets, "cases": records,
    }
    (destination / "selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 真实残卷摆放：20个成功＋20个失败", "",
             "每张图从左到右：标准摆放、32＋64基线、7＋16＋32＋64四尺度候选。两模型均为 seed 260909、epoch 1、Top2 位移解算。", "",
             "成功/失败按四尺度候选的平移 L2 误差≤10px / >10px分组；px为800画布坐标，不是原始扫描分辨率。Pair PASS/REJECT单独表示配对分类。", "",
             "从508个真实正对中，候选摆放成功204、失败304；这里按误差排名分散抽取各20个、覆盖40个不同案例，不能用本图的20:20比例估计准确率。", "",
             "RGB仅用于展示，预测只读取已有评估结果。蓝线A固定，橙线B按预测平移；同一案例三列坐标范围完全相同。", ""]
    for record in records:
        lines += ["## {} · {}".format(record["id"], record["source_case"]), "",
                  "![{}]({})".format(record["id"], record["image"]), ""]
    (destination / "GALLERY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"success_pool": len(success_pool), "failure_pool": len(failure_pool),
                      "rendered": len(records), "distinct_cases": len({r["case_id"] for r in records}),
                      "sheets": len(sheets)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prepared-cache", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--local-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=140)
    render(parser.parse_args())
