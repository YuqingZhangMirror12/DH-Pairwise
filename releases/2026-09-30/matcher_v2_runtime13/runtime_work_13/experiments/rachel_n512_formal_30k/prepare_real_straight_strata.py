"""Frozen REAL1016 input-only strata for plausible opposing straight edges.

This describes edge presence, not the true annotated seam or a causal failure
mechanism. No model, prediction, translation GT, TRAIN donor, or mining is used.
The previously fixed default geometry thresholds are not selected on REAL.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from staging.pairwise_v0_2.pairwise_data.rachel_straight_negatives import (
    StraightNegativeConfig, matching_straight_edges, straight_contour_arcs,
)

SCHEMA_VERSION = "rachel-real-straight-strata/1"
CONFIG = StraightNegativeConfig()


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _load(prepared):
    # Match load_prepared_cache's membership/shape contract without importing
    # its model/evaluator modules or materializing the unused packed masks.
    metadata = json.loads((prepared / "manifest.json").read_text())
    if metadata.get("schema") != "real-layout-prepared-v1":
        raise ValueError("expected the existing real-layout prepared cache")
    pairs, ids = metadata["pairs"], metadata["fragment_ids"]
    if (len(pairs) != 1016 or len({row["pair_id"] for row in pairs}) != 1016
            or any(type(row.get(key)) is not bool for row in pairs for key in ("label", "strict"))
            or sum(row["label"] for row in pairs) != 508
            or sum(row["strict"] and row["label"] for row in pairs) != 508
            or sum(row["strict"] and not row["label"] for row in pairs) != 39
            or len(set(ids)) != len(ids)):
        raise ValueError("prepared population must be frozen REAL1016/strict547 with 508 positives")
    known_ids = set(ids)
    if any(row["fragment_" + side + "_id"] not in known_ids for row in pairs for side in "ab"):
        raise ValueError("prepared pair endpoint is absent from fragment_ids")
    with np.load(prepared / "inputs.npz", allow_pickle=False) as archive:
        points, valid = archive["points"], archive["valid"]
    if points.shape != (len(ids), 512, 2) or valid.shape != (len(ids), 512) or valid.dtype != np.bool_:
        raise ValueError("prepared N512 contour shapes or valid-mask dtype differ")
    return metadata, points, valid


def _fragment_arcs(points, valid):
    # Prepared rows may be padded. Strip invalid tokens before applying the
    # unchanged miner predicate; invalid padding is not an observed contour gap.
    points = np.asarray(points[valid], dtype=float)
    if len(points) < 6:
        return None, "fewer_than_six_valid_points"
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 799):
        return None, "nonfinite_or_outside_model_canvas"
    r, c = points.T
    signed_area = .5 * np.sum(c * np.roll(r, -1) - np.roll(c, -1) * r)
    if abs(signed_area) < 1e-6:
        return None, "degenerate_contour_winding"
    return straight_contour_arcs(points, config=CONFIG), None


def build(prepared, output):
    prepared, output = Path(prepared).resolve(strict=True), Path(output).resolve()
    if output == prepared or prepared in output.parents:
        raise ValueError("write new strata outside the original prepared cache")
    destination = output / "geometry_manifest.json"
    if destination.exists():
        raise FileExistsError("geometry manifest already exists; refusing to overwrite")
    metadata, points, valid = _load(prepared)
    index = {token: i for i, token in enumerate(metadata["fragment_ids"])}
    cache, rows = {}, []
    preserved = ("pair_id", "fragment_a_id", "fragment_b_id", "label", "strict", "case_uid",
                 "case_cluster", "case_id", "source_case_uid", "source_case_uids")
    for source_index, source in enumerate(metadata["pairs"]):
        tokens = [source["fragment_" + side + "_id"] for side in "ab"]
        for token in tokens:
            if token not in cache:
                i = index[token]
                cache[token] = _fragment_arcs(points[i], valid[i])
        arcs, reasons = zip(*(cache[token] for token in tokens))
        available = all(reason is None for reason in reasons)
        orientations = sorted(matching_straight_edges(*arcs, config=CONFIG)) if available else None
        row = {key: source[key] for key in preserved if key in source}
        row.update(source_index=source_index, fragment_a=tokens[0], fragment_b=tokens[1],
                   strict_member=source["strict"], valid_geometry=available,
                   matching_orientations=orientations,
                   orientation_signature=("+".join(orientations) or "none") if available else "unavailable",
                   any_opposing_straight=bool(orientations) if available else None,
                   geometry_unavailable_reason={side: reason for side, reason in zip("ab", reasons) if reason} or None)
        rows.append(row)
    counts = dict(total=len(rows), positive=sum(row["label"] for row in rows), negative=sum(not row["label"] for row in rows),
        strict_total=sum(row["strict_member"] for row in rows), strict_positive=sum(row["strict_member"] and row["label"] for row in rows),
        strict_negative=sum(row["strict_member"] and not row["label"] for row in rows),
        valid_geometry=sum(row["valid_geometry"] for row in rows), unavailable_geometry=sum(not row["valid_geometry"] for row in rows),
        any_opposing_straight=sum(row["any_opposing_straight"] is True for row in rows),
        orientation_signatures=dict(sorted(Counter(row["orientation_signature"] for row in rows).items())),
        fragment_cache_entries=len(cache), unavailable_fragments=sum(reason is not None for _, reason in cache.values()))
    result = dict(schema_version=SCHEMA_VERSION, status="complete", rows=rows, counts=counts, config=asdict(CONFIG),
        provenance=dict(prepared_cache=str(prepared), prepared_manifest_sha256=_sha256(prepared / "manifest.json"),
            prepared_inputs_sha256=_sha256(prepared / "inputs.npz"), population_manifest_sha256=metadata.get("manifest_sha256"),
            input_array_fields_read=["points", "valid"], original_population_order_preserved=True,
            config_origin="existing StraightNegativeConfig defaults; no REAL-based threshold selection",
            geometry_functions="rachel_straight_negatives.straight_contour_arcs + matching_straight_edges",
            geometry_source="model-facing sampled N512 contours with invalid padding removed",
            model_or_prediction_used=False, translation_gt_opened=False, train_mining_or_donors_used=False,
            source=metadata.get("source")),
        interpretation="Presence of plausible opposing roughly straight contour arcs, not an actual straight GT seam; orientation is not evidence that vertical or horizontal edges cause model failure.")
    output.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-cache", required=True)
    parser.add_argument("--output", required=True, help="new output directory containing geometry_manifest.json")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.prepared_cache, args.output)["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
