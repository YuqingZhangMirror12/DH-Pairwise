"""Bounded TRAIN-only mining of CSV-labelled straight-edge distractors.

This writes an index overlay, never an altered composite or a new positive label.
Straight geometry selects *already negative* native pairs: it does not infer
nonadjacency from rectangles, masks, or model scores. Native positive labels and
every unordered pair in the base manifest are protected before geometry is read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np

from .rachel_composite_training import (
    AREA_EDGES, RATIO_BANDS, SCHEMA_VERSION as COMPOSITE_SCHEMA,
    _area_bin, _fragment, _ratio_band,
)
from .rachel_training_dataset import (
    RachelDatasetConfig, RachelDatasetError, _load_contour, _read_selected_pair,
    _safe_release_path,
)

SCHEMA_VERSION = "rachel-straight-negative-overlay/1"
CSV_NEGATIVE_ORIGIN = "rachel_csv_same_folder_nonneighbor_no_seam"
ORIENTATIONS = ("vertical", "horizontal", "oblique")


@dataclass(frozen=True)
class StraightNegativeConfig:
    """Resource ceilings fail closed or produce explicitly counted shortfalls."""
    minimum_arc_length_px: float = 40.0
    minimum_perimeter_fraction: float = .06
    maximum_deviation_px: float = 2.5
    minimum_chord_arc_ratio: float = .985
    axis_tolerance_degrees: float = 15.0
    parallel_tolerance_degrees: float = 15.0
    minimum_opposing_normal_cosine: float = .8
    minimum_arc_length_ratio: float = .5
    max_metadata_rows: int = 2_000_000
    max_candidate_pairs: int = 30_000
    max_contour_loads: int = 20_000
    max_positive_pairs: int = 12_000

    def __post_init__(self):
        for name in ("max_metadata_rows", "max_candidate_pairs", "max_contour_loads", "max_positive_pairs"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        for name in ("minimum_arc_length_px", "maximum_deviation_px"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("minimum_perimeter_fraction", "minimum_chord_arc_ratio", "minimum_opposing_normal_cosine", "minimum_arc_length_ratio"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(name + " must be in (0, 1]")
        for name in ("axis_tolerance_degrees", "parallel_tolerance_degrees"):
            if not 0 < getattr(self, name) < 45:
                raise ValueError(name + " must be between 0 and 45")


def _jsonl(path, limit):
    count = 0
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            count += 1
            if count > limit:
                raise ValueError("metadata row ceiling exceeded: " + str(path))
            yield json.loads(line)


def _rank(value, seed):
    return int.from_bytes(hashlib.sha256((str(seed) + "\0" + str(value)).encode()).digest(), "big")


def _pair_key(a, b):
    return tuple(sorted((a, b)))


def _row_key(row):
    return _pair_key(*(row["fragment_" + side]["fragment_token"] for side in "ab"))


def _joint_bin(row):
    return _ratio_band(row), _area_bin(row)


def _bin_name(key):
    return "%s|min_area_bin_%s" % key


def _apportion(weights, total):
    """Deterministic largest-remainder allocation; no ratio/area relaxation."""
    if not total:
        return {key: 0 for key in sorted(weights)}
    mass = sum(weights.values())
    if mass <= 0:
        raise ValueError("cannot allocate a positive total to empty weights")
    exact = {key: total * value / mass for key, value in weights.items()}
    result = {key: int(np.floor(value)) for key, value in exact.items()}
    for key in sorted(weights, key=lambda key: (-(exact[key] - result[key]), key))[:total - sum(result.values())]:
        result[key] += 1
    return result


def straight_contour_arcs(points_rc, valid=None, *, config=StraightNegativeConfig()):
    """Find long contiguous near-linear arcs, with winding-derived outward normals.

    Multi-scale windows include the contour wrap. Every point in an accepted
    window is checked against its endpoint chord, not merely its endpoints.
    Incomplete contours are excluded rather than guessing winding across gaps.
    """
    points = np.asarray(points_rc, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 6:
        return []
    valid = np.ones(len(points), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != (len(points),) or not np.all(np.isfinite(points[valid])):
        return []
    # A missing part of the boundary makes global winding unreliable.
    if not np.all(valid):
        return []
    rows, columns = points.T
    signed_area = .5 * np.sum(columns * np.roll(rows, -1) - np.roll(columns, -1) * rows)
    if abs(signed_area) < 1e-6:
        return []
    perimeter = float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())
    minimum_length = max(config.minimum_arc_length_px, config.minimum_perimeter_fraction * perimeter)
    n = len(points)
    window_sizes = sorted({max(6, min(n // 3, int(round(n * fraction)))) for fraction in (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 3)})
    found = []
    for size in window_sizes:
        if size < 6:
            continue
        indices = (np.arange(n)[:, None] + np.arange(size)[None, :]) % n
        windows = points[indices]
        delta = windows[:, -1] - windows[:, 0]
        lengths = np.linalg.norm(delta, axis=1)
        tangent = delta / np.maximum(lengths[:, None], 1e-12)
        offsets = windows - windows[:, :1]
        deviation = np.max(np.abs(offsets[:, :, 0] * tangent[:, None, 1] - offsets[:, :, 1] * tangent[:, None, 0]), axis=1)
        arc_lengths = np.linalg.norm(np.diff(windows, axis=1), axis=2).sum(axis=1)
        chord_ratio = lengths / np.maximum(arc_lengths, 1e-12)
        accepted = (lengths >= minimum_length) & (deviation <= config.maximum_deviation_px) & (chord_ratio >= config.minimum_chord_arc_ratio)
        for start in np.flatnonzero(accepted):
            dr, dc = tangent[start]
            tolerance = np.sin(np.deg2rad(config.axis_tolerance_degrees))
            orientation = "vertical" if abs(dc) <= tolerance else "horizontal" if abs(dr) <= tolerance else "oblique"
            normal = np.sign(signed_area) * np.array([-dc, dr])
            found.append(dict(start_token=int(start), token_count=int(size),
                endpoints_rc=windows[start, [0, -1]].tolist(), tangent_rc=tangent[start].tolist(),
                outward_normal_rc=normal.tolist(), orientation=orientation,
                chord_length_px=float(lengths[start]), arc_length_px=float(arc_lengths[start]),
                maximum_deviation_px=float(deviation[start]), chord_arc_ratio=float(chord_ratio[start]),
                perimeter_fraction=float(arc_lengths[start] / perimeter)))
    # Keep both outward-normal directions, and suppress overlapping windows.
    selected = []
    for arc in sorted(found, key=lambda arc: (-arc["chord_length_px"], arc["maximum_deviation_px"], arc["start_token"])):
        similar = [old for old in selected if old["orientation"] == arc["orientation"] and
                   np.dot(old["outward_normal_rc"], arc["outward_normal_rc"]) > .95]
        if len(similar) >= 2:
            continue
        tokens = {(arc["start_token"] + i) % n for i in range(arc["token_count"])}
        if any(len(tokens.intersection({(old["start_token"] + i) % n for i in range(old["token_count"])})) > .5 * min(arc["token_count"], old["token_count"]) for old in similar):
            continue
        selected.append(arc)
    return selected


def matching_straight_edges(arcs_a, arcs_b, *, config=StraightNegativeConfig()):
    """Return the strongest opposing-edge evidence per orientation, if present."""
    matches = {}
    for a in arcs_a:
        for b in arcs_b:
            if a["orientation"] != b["orientation"]:
                continue
            parallel = abs(float(np.dot(a["tangent_rc"], b["tangent_rc"])))
            opposition = -float(np.dot(a["outward_normal_rc"], b["outward_normal_rc"]))
            length_ratio = min(a["chord_length_px"], b["chord_length_px"]) / max(a["chord_length_px"], b["chord_length_px"])
            if parallel < np.cos(np.deg2rad(config.parallel_tolerance_degrees)) or opposition < config.minimum_opposing_normal_cosine or length_ratio < config.minimum_arc_length_ratio:
                continue
            score = min(a["chord_length_px"], b["chord_length_px"]) * opposition / (1 + a["maximum_deviation_px"] + b["maximum_deviation_px"])
            evidence = dict(orientation=a["orientation"], fragment_a_arc=a, fragment_b_arc=b,
                            outward_normal_opposition=opposition, absolute_tangent_cosine=parallel,
                            arc_length_ratio=length_ratio, geometry_score=score, model_hard=False)
            if score > matches.get(a["orientation"], {}).get("geometry_score", -1):
                matches[a["orientation"]] = evidence
    return matches


class _GeometryBudget:
    def __init__(self, config):
        self.config, self.cache, self.loads, self.errors = config, {}, 0, Counter()
        self.limit = config.max_contour_loads

    def arcs(self, root, fragment):
        key = (str(root), fragment["contour_path"])
        if key in self.cache:
            return self.cache[key]
        if self.loads >= self.limit:
            self.errors["contour_budget_exhausted"] += 1
            return None
        self.loads += 1
        try:
            path = _safe_release_path(Path(root), fragment["contour_path"], prefix=("model", "contours_n512"), suffix=".npz")
            points, valid = _load_contour(path, RachelDatasetConfig())
            result = straight_contour_arcs(points, valid, config=self.config)
        except (RachelDatasetError, OSError) as error:
            self.errors[type(error).__name__ + ": " + str(error)] += 1
            result = None
        self.cache[key] = result
        return result


def build_straight_negative_overlay(release_root, base_manifest, output_manifest, *,
                                    max_replacements=2400, seed=260910,
                                    config=StraightNegativeConfig()):
    """Build an ordinary-runtime-row overlay replacing <=20% of base negatives.

    All metadata is streamed with an explicit row ceiling. Candidate geometry
    uses a bounded deterministic hash sample, and a bounded contour cache. No
    held-out contours, masks, targets, RGB, or inference are read. Unavailable
    orientation/size cells stay unchanged and are reported as shortfalls.
    """
    if type(max_replacements) is not int or not 0 <= max_replacements <= 2400:
        raise ValueError("max_replacements must be an integer from 0 to 2400")
    root, manifest = Path(release_root).resolve(strict=True), Path(base_manifest).resolve(strict=True)
    destination = Path(output_manifest).resolve()
    if destination == manifest or destination == root or root in destination.parents:
        raise ValueError("overlay must be separate from the base manifest and original release")
    base_bytes = manifest.read_bytes()
    base = json.loads(base_bytes)
    if base.get("schema_version") != COMPOSITE_SCHEMA or base.get("split") != "train":
        raise ValueError("a TRAIN composite base manifest is required")
    if len(base["entries"]) > config.max_metadata_rows:
        raise ValueError("base manifest metadata row ceiling exceeded")
    splits = json.loads((root / "pairs/lineage_splits.json").read_text())
    base_keys, base_ids, positive_entries, slots = set(), set(), [], defaultdict(list)
    for index, entry in enumerate(base["entries"]):
        row = entry["row"]
        if row.get("split") != "train" or type(row.get("label")) is not bool:
            raise ValueError("base manifest must contain explicit TRAIN boolean labels")
        if any(splits.get(row["fragment_" + side]["split_unit_id"]) != "train" for side in "ab"):
            raise ValueError("base entry is outside frozen TRAIN lineages")
        key = _row_key(row)
        if key in base_keys or row["pair_id"] in base_ids or key[0] == key[1]:
            raise ValueError("duplicate unordered fragment pair, pair id, or self-pair in base")
        base_keys.add(key)
        base_ids.add(row["pair_id"])
        if row["label"]:
            positive_entries.append(entry)
        else:
            slots[_joint_bin(row)].append(index)
    negative_count = sum(map(len, slots.values()))
    requested = min(max_replacements, negative_count // 5)
    quotas = _apportion({key: len(value) for key, value in slots.items()}, requested)
    fragments = {}
    for row in _jsonl(root / "manifests/fragments.jsonl", config.max_metadata_rows):
        if splits.get(row["split_unit_id"]) == "train":
            fragment = _fragment(row)
            if fragment["foreground_area"] <= 0 or fragment["bbox_aspect_ratio"] <= 0:
                raise ValueError("TRAIN fragment has invalid size metadata")
            token = fragment["fragment_token"]
            if token in fragments:
                raise ValueError("duplicate TRAIN fragment token")
            fragments[token] = fragment
    rejected, native_positives, reservoir = Counter(), set(), []
    eligible = 0
    for line_index, candidate in enumerate(_jsonl(root / "manifests/within_candidates.jsonl", config.max_metadata_rows)):
        a, b = candidate["fragment_a_token"], candidate["fragment_b_token"]
        if a not in fragments or b not in fragments:
            rejected["outside_frozen_train"] += 1
            continue
        key = _pair_key(a, b)
        if candidate.get("label") is True:
            native_positives.add(key)
            continue
        if candidate.get("label") is not False or candidate.get("main_training_eligible") is not True:
            rejected["not_eligible_explicit_negative"] += 1
            continue
        if candidate.get("label_origin") != CSV_NEGATIVE_ORIGIN or candidate.get("negative_origin") != "same_folder_hard":
            rejected["not_native_csv_nonadjacency"] += 1
            continue
        if key in base_keys or candidate["pair_id"] in base_ids or a == b:
            rejected["already_in_base_or_self_pair"] += 1
            continue
        if any(candidate.get(field) is not None for field in ("correspondence_path", "translation_a_to_b_rc", "translation_a_to_b_xy_cartesian")):
            rejected["negative_carries_target_or_translation"] += 1
            continue
        row = dict(pair_id=candidate["pair_id"], split="train", label=False,
                   fragment_a=fragments[a], fragment_b=fragments[b], correspondence_path=None,
                   translation_a_to_b_rc=None, translation_a_to_b_xy_cartesian=None,
                   label_origin=candidate["label_origin"], negative_origin=candidate["negative_origin"])
        if not quotas.get(_joint_bin(row), 0):
            rejected["no_requested_size_bin"] += 1
            continue
        eligible += 1
        item = (-_rank((key, candidate["pair_id"]), seed), -line_index, row)
        if len(reservoir) < config.max_candidate_pairs:
            heapq.heappush(reservoir, item)
        elif item[:2] > reservoir[0][:2]:
            heapq.heapreplace(reservoir, item)
    geometry, pools, seen = _GeometryBudget(config), defaultdict(list), set()
    # Reserve some bounded reads for positive context even if mining exhausts
    # its geometry budget. Cached positive fragments do not need another read.
    positive_read_reserve = min(config.max_contour_loads // 5, 2 * min(len(positive_entries), config.max_positive_pairs))
    geometry.limit -= positive_read_reserve
    for _, _, row in sorted(reservoir, key=lambda item: (-item[0], -item[1])):
        key = _row_key(row)
        if key in native_positives:
            rejected["conflicts_with_native_positive"] += 1
            continue
        if key in seen:
            rejected["duplicate_native_unordered_pair"] += 1
            continue
        seen.add(key)
        try:
            _read_selected_pair(root, "train", row, 1)
        except RachelDatasetError:
            rejected["invalid_runtime_row"] += 1
            continue
        arcs = [geometry.arcs(root, row["fragment_" + side]) for side in "ab"]
        if any(value is None for value in arcs):
            rejected["unavailable_candidate_contour"] += 1
            continue
        matches = matching_straight_edges(*arcs, config=config)
        if not matches:
            rejected["no_opposing_long_straight_arcs"] += 1
            continue
        pools[_joint_bin(row)].append((row, matches))
    entries, cells, chosen_keys, chosen_ids = [], {}, set(), set()
    for key, quota in sorted(quotas.items()):
        # Oblique controls are a reserved 20%, not a fill for missing axis pairs.
        orientation_quotas = _apportion({"vertical": 2, "horizontal": 2, "oblique": 1}, quota)
        available = pools[key]
        remaining_slots = sorted(slots[key], key=lambda index: _rank(index, seed + 1))
        orientation_counts = Counter()
        selected_in_bin = 0
        # Assign scarce orientations first so flexible rectangle pairs remain usable.
        order = sorted(ORIENTATIONS, key=lambda orientation: (sum(orientation in matches for _, matches in available), orientation))
        for orientation in order:
            ranked = sorted(((row, matches[orientation]) for row, matches in available if orientation in matches),
                            key=lambda value: (-value[1]["geometry_score"], _rank(value[0]["pair_id"], seed)))
            for row, evidence in ranked:
                if orientation_counts[orientation] >= orientation_quotas[orientation]:
                    break
                pair_key = _row_key(row)
                if pair_key in chosen_keys or row["pair_id"] in chosen_ids:
                    continue
                source_index = remaining_slots[selected_in_bin]
                entries.append(dict(source_index=source_index, source_root=str(root), row=row,
                    geometry=dict(evidence, ratio_band=key[0], minimum_area_bin=key[1],
                                  native_label_origin=CSV_NEGATIVE_ORIGIN,
                                  unchanged_base_pair_id=base["entries"][source_index]["row"]["pair_id"])))
                chosen_keys.add(pair_key)
                chosen_ids.add(row["pair_id"])
                orientation_counts[orientation] += 1
                selected_in_bin += 1
        actual = sum(orientation_counts.values())
        cells[_bin_name(key)] = dict(requested=quota, selected=actual, shortfall=quota - actual,
            orientation_requested=orientation_quotas, orientation_selected={name: orientation_counts[name] for name in ORIENTATIONS})
    # Same geometry predicate on unchanged TRAIN positives: descriptive context,
    # never a relabel or a claim that a straight arc is the actual positive seam.
    geometry.limit = config.max_contour_loads
    context = Counter(total_positive_pairs=len(positive_entries), attempted_pairs=0, unavailable_pairs=0,
                      examined_pairs=0, at_least_one_straight_fragment_pairs=0,
                      both_straight_fragment_pairs=0, opposing_straight_positive_pairs=0)
    positive_orientations = Counter()
    for entry in sorted(positive_entries, key=lambda entry: _rank(entry["row"]["pair_id"], seed + 2))[:config.max_positive_pairs]:
        context["attempted_pairs"] += 1
        source = Path(entry["source_root"]).resolve(strict=True)
        arcs = [geometry.arcs(source, entry["row"]["fragment_" + side]) for side in "ab"]
        if any(value is None for value in arcs):
            context["unavailable_pairs"] += 1
            continue
        context["examined_pairs"] += 1
        context["at_least_one_straight_fragment_pairs"] += int(any(arcs))
        context["both_straight_fragment_pairs"] += int(all(arcs))
        matches = matching_straight_edges(*arcs, config=config)
        context["opposing_straight_positive_pairs"] += int(bool(matches))
        positive_orientations.update(matches.keys())
    context["unexamined_pairs"] = len(positive_entries) - context["examined_pairs"]
    payload = dict(schema_version=SCHEMA_VERSION, split="train", base_manifest=str(manifest),
        base_manifest_sha256=hashlib.sha256(base_bytes).hexdigest(), release_root=str(root), seed=seed,
        entries=sorted(entries, key=lambda entry: entry["source_index"]),
        counts=dict(base_total=len(base["entries"]), base_positive=len(positive_entries), base_negative=negative_count,
                    requested_replacements=requested, selected_replacements=len(entries), shortfall=requested - len(entries),
                    positive_rows_modified=0, native_eligible_size_matched=eligible,
                    candidate_reservoir_size=len(reservoir), candidate_reservoir_omitted=max(0, eligible - len(reservoir)),
                    contour_loads=geometry.loads, positive_contour_read_reserve=positive_read_reserve),
        size_bin_counts=cells, rejected_counts=dict(rejected), contour_errors=dict(geometry.errors),
        straight_positive_context=dict(context, orientation_counts=dict(positive_orientations),
            count_is_lower_bound=bool(context["unexamined_pairs"]),
            note="Unchanged base TRAIN positives under the same contour predicate; straight arcs need not be the annotated seam. Straightness is not a negative label."),
        mining_rule=dict(name="native_csv_nonadjacent_opposing_straight_contour_arcs", model_inference=False,
            label_origin=CSV_NEGATIVE_ORIGIN, frozen_train_lineages_only=True,
            max_replacements=2400, max_negative_fraction=.2, orientation_weights=dict(vertical=.4, horizontal=.4, oblique=.2),
            ratio_bands={name: list(band[:2]) for name, band in RATIO_BANDS.items()},
            absolute_min_area_bin_edges_px=list(AREA_EDGES), exact_joint_size_bins=True,
            no_bin_or_orientation_fallback=True, config=asdict(config),
            label_policy="Only original eligible CSV nonneighbor negatives; no synthetic rectangle labels or positive flips.",
            limitations="Geometry-hard, not model-hard. Native synthetic fragments approximate manuscript-edge distractors; real-manuscript generalization is unproven."))
    if manifest.read_bytes() != base_bytes:
        raise ValueError("base manifest changed during mining")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-replacements", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=260910)
    parser.add_argument("--max-candidate-pairs", type=int, default=30_000)
    parser.add_argument("--max-contour-loads", type=int, default=20_000)
    parser.add_argument("--max-positive-pairs", type=int, default=12_000)
    parser.add_argument("--max-metadata-rows", type=int, default=2_000_000)
    args = parser.parse_args(argv)
    config = StraightNegativeConfig(**{name: getattr(args, name) for name in
        ("max_candidate_pairs", "max_contour_loads", "max_positive_pairs", "max_metadata_rows")})
    result = build_straight_negative_overlay(args.release_root, args.base_manifest, args.output,
        max_replacements=args.max_replacements, seed=args.seed, config=config)
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
