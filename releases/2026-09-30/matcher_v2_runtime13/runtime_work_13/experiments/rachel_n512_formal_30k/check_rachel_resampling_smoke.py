"""CPU-only 16-positive/16-negative train sample check for 512/1024 rebuilding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from staging.pairwise_v0_2.pairwise_data.rachel_resampled_dataset import RachelResampledDataset
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset


def compare_side(original, rebuilt, side):
    points, new_points = getattr(original, "points_rc_" + side), getattr(rebuilt, "points_rc_" + side)
    target, new_target = getattr(original, "target_" + side), getattr(rebuilt, "target_" + side)
    same_shape = points.shape == new_points.shape
    return {"release_valid_count": int(getattr(original, "contour_valid_" + side).sum()),
        "rebuilt_valid_count": int(getattr(rebuilt, "contour_valid_" + side).sum()),
        "points_shape_equal": same_shape, "points_exact_equal": bool(np.array_equal(points, new_points)),
        "points_max_abs_diff_px": float(np.max(np.abs(points.astype(float) - new_points))) if same_shape else None,
        "targets_exact_equal": bool(np.array_equal(target, new_target)),
        "target_index_difference_count": int(np.count_nonzero(target != new_target)) if target.shape == new_target.shape else None,
        "release_correspondence_count": int((target >= 0).sum()), "rebuilt_correspondence_count": int((new_target >= 0).sum())}


def rebuilt_description(sample):
    first = np.flatnonzero(sample.target_a >= 0)
    second = sample.target_a[first]
    reciprocal = bool(np.array_equal(sample.target_b[second], first))
    bounds = bool(np.all(second < len(sample.points_rc_b)) and np.all(sample.target_b < len(sample.points_rc_a)))
    residual = np.linalg.norm(sample.points_rc_a[first].astype(float) + sample.translation_a_to_b_rc
                              - sample.points_rc_b[second], axis=1)
    return {"valid_points_a": int(sample.contour_valid_a.sum()), "valid_points_b": int(sample.contour_valid_b.sum()),
        "unique_points_a": len(np.unique(sample.points_rc_a[sample.contour_valid_a], axis=0)),
        "unique_points_b": len(np.unique(sample.points_rc_b[sample.contour_valid_b], axis=0)),
        "correspondence_count": len(first), "reciprocal": reciprocal, "index_bounds_valid": bounds,
        "target_residual_median_px": float(np.median(residual)) if len(residual) else None,
        "target_residual_p95_px": float(np.quantile(residual, .95)) if len(residual) else None}


def run(args):
    started = time.perf_counter()
    dataset = RachelPairDataset(Path(args.dataset), "train")
    rng = np.random.default_rng(args.seed)
    positive = [i for i, row in enumerate(dataset._rows) if row.label]
    negative = [i for i, row in enumerate(dataset._rows) if not row.label]
    indices = np.r_[rng.choice(positive, 16, replace=False), rng.choice(negative, 16, replace=False)]
    rng.shuffle(indices)
    wrappers = {cap: RachelResampledDataset(dataset, contour_cap=cap, cache_size=128) for cap in (512, 1024)}
    rows = []
    for position, raw_index in enumerate(indices):
        index = int(raw_index)
        original = dataset[index]
        row = {"train_index": index, "pair_id": original.pair_id, "label": bool(original.label),
            "translation_a_to_b_rc": original.translation_a_to_b_rc.tolist(), "rebuilt": {}}
        for cap, wrapper in wrappers.items():
            before = time.perf_counter()
            try:
                rebuilt = wrapper[index]
                record = dict(rebuilt_description(rebuilt), status="ok", elapsed_s=time.perf_counter() - before)
                if cap == 512:
                    record["release_comparison"] = {side: compare_side(original, rebuilt, side) for side in ("a", "b")}
            except Exception as error:
                record = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                          "elapsed_s": time.perf_counter() - before}
            row["rebuilt"][str(cap)] = record
        rows.append(row)
        print(json.dumps({"checked": position + 1, "total": 32, "label": row["label"],
            "status512": row["rebuilt"]["512"]["status"], "status1024": row["rebuilt"]["1024"]["status"]}), flush=True)
    successful512 = [r for r in rows if r["rebuilt"]["512"]["status"] == "ok"]
    comparisons = [r["rebuilt"]["512"]["release_comparison"][side] for r in successful512 for side in ("a", "b")]
    expanded = [r for r in rows if all(r["rebuilt"][str(c)]["status"] == "ok" for c in (512, 1024))]
    summary = {"sample_count": len(rows), "positive_count": sum(r["label"] for r in rows),
        "negative_count": sum(not r["label"] for r in rows), "rebuild512": {
            "successful_pair_count": len(successful512), "compared_fragment_occurrences": len(comparisons),
            "points_exact_equal_occurrences": sum(c["points_exact_equal"] for c in comparisons),
            "targets_exact_equal_occurrences": sum(c["targets_exact_equal"] for c in comparisons),
            "maximum_point_abs_difference_px": max((c["points_max_abs_diff_px"] for c in comparisons if c["points_max_abs_diff_px"] is not None), default=None),
            "pairs_with_any_target_difference": sum(any(not c["targets_exact_equal"] for c in r["rebuilt"]["512"]["release_comparison"].values()) for r in successful512)},
        "density1024": {"compared_pairs": len(expanded), "fragment_occurrences_with_more_valid_points":
            sum(r["rebuilt"]["1024"]["valid_points_" + side] > r["rebuilt"]["512"]["valid_points_" + side] for r in expanded for side in ("a", "b")),
            "positive_correspondence_count512": sum(r["rebuilt"]["512"]["correspondence_count"] for r in expanded if r["label"]),
            "positive_correspondence_count1024": sum(r["rebuilt"]["1024"]["correspondence_count"] for r in expanded if r["label"])}}
    for cap in (512, 1024):
        records = [row["rebuilt"][str(cap)] for row in rows]
        summary[str(cap)] = {"failure_count": sum(record["status"] != "ok" for record in records),
            "positive_failure_count": sum(row["label"] and row["rebuilt"][str(cap)]["status"] != "ok" for row in rows),
            "total_rebuild_seconds_including_base_io": sum(record["elapsed_s"] for record in records),
            "median_rebuild_seconds_including_base_io": float(np.median([record["elapsed_s"] for record in records])),
            "cache": wrappers[cap].cache_info()}
    report = {"status": "complete_32_pair_cpu_smoke", "seed": args.seed,
        "selection": "train split; uniform 16 positives and 16 negatives without replacement; NumPy default_rng",
        "dataset": str(args.dataset), "gpu_or_model_used": False, "original_data_modified": False,
        "scope": "selected 32 pairs only; not a full-dataset equivalence claim", "summary": summary,
        "rows": rows, "elapsed_s": time.perf_counter() - started}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/root/autodl-tmp/dataset_rachel_pairwise_n512_v1")
    parser.add_argument("--seed", type=int, default=260907)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
