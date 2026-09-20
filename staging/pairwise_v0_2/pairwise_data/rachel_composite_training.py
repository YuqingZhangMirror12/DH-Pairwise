"""New TRAIN-only manifests and runtime adapter; original release stays read-only.

The 60k and nested24k manifests share exact strata proportions. Cross negatives
are unique, different-frozen-TRAIN-lineage pairs mined using area/aspect metadata,
NOT model-hard negatives. No held-out mask, model score, or target is inspected.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .rachel_training_dataset import (RachelPairDataset, RachelDatasetConfig, RachelDatasetError,
    _read_selected_pair, _load_contour, _load_positive_target, _FragmentArrays)

SCHEMA_VERSION = "rachel-composite-training/1"
AREA_EDGES = (16384, 32768, 65536, 131072)
RATIO_BANDS = {"tiny": (0, .25, .18), "mid_ratio": (.25, .5, .375),
               "large_ratio": (.5, .8, .65), "scale_matched": (.8, 1.000001, .95)}


def _jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _rank(value, seed):
    return hashlib.sha256((str(seed) + str(value)).encode()).digest()


def _ratio(row):
    sizes = [float(row["fragment_" + side]["foreground_area"]) for side in "ab"]
    return min(sizes) / max(sizes)


def _area_bin(row):
    minimum = min(float(row["fragment_" + side]["foreground_area"]) for side in "ab")
    return int(np.searchsorted(AREA_EDGES, minimum, side="right"))


def _ratio_band(row):
    ratio = _ratio(row)
    return next(name for name, (_, high, _) in RATIO_BANDS.items() if ratio < high)


def _fragment(row):
    return {key: row[key] for key in ("fragment_token", "model_mask_path", "contour_path",
                                     "foreground_area", "bbox_aspect_ratio", "split_unit_id")}


def _entry(root, row, stratum):
    return dict(source_root=str(Path(root).resolve()), row=row, stratum=stratum)


class _RowsDataset(RachelPairDataset):
    def __init__(self, root, rows, config):
        self.root, self.split, self.config = Path(root).resolve(strict=True), "train", config
        self._rows = tuple(_read_selected_pair(self.root, "train", row, i + 1) for i, row in enumerate(rows))


class CompositeRachelPairDataset:
    """Return unchanged RachelPairSample items from a new mixed-source manifest."""
    def __init__(self, manifest, config=RachelDatasetConfig()):
        self.manifest, self.config = Path(manifest).resolve(strict=True), config
        data = json.loads(self.manifest.read_text())
        if data.get("schema_version") != SCHEMA_VERSION or data.get("split") != "train":
            raise ValueError("a TRAIN composite manifest is required")
        rows, indices, seen = defaultdict(list), [], set()
        for item in data["entries"]:
            row, root = item["row"], item["source_root"]
            if row.get("split") != "train" or row["pair_id"] in seen:
                raise ValueError("duplicate or held-out composite entry")
            seen.add(row["pair_id"])
            indices.append((root, len(rows[root])))
            rows[root].append(row)
        self._sources = {root: _RowsDataset(root, values, config) for root, values in rows.items()}
        self._indices = tuple(indices)
        self.stats = data["stats"]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        root, source_index = self._indices[index]
        return self._sources[root][source_index]


def _cross_negatives(fragments, *, count, tiny_count, seed, min_area_targets=None):
    """Mine size/aspect-near candidates, with a separate tiny/large stratum.

    Tiny negatives cannot have similar areas by definition: within that stratum
    the query target area ratio is .18 and aspect ratios must still be similar.
    Each unordered fragment pair occurs once; no fragment is used >8 times.
    """
    values = sorted(fragments, key=lambda row: _rank(row["fragment_token"], seed))
    feature = np.log([[x["foreground_area"], x["bbox_aspect_ratio"]] for x in values])
    tree, seen, usage = cKDTree(feature), set(), Counter()
    output = []
    requests = [("tiny", tiny_count, 1 / .18, None), ("scale_matched", count - tiny_count, 1.0, None)]
    if min_area_targets is not None:
        requests = [(name, number, 1 / RATIO_BANDS[name][2], area_bin)
                    for name in min_area_targets for area_bin, number in min_area_targets[name].items()]
        if sum(x[1] for x in requests) != count or sum(x[1] for x in requests if x[0] == "tiny") != tiny_count:
            raise ValueError("absolute area quotas disagree with cross counts")
    for name, requested, area_factor, area_bin in requests:
        chosen = 0
        for i, first in enumerate(values):
            if chosen == requested:
                break
            if usage[first["fragment_token"]] >= 8:
                continue
            if area_bin is not None and int(np.searchsorted(AREA_EDGES, first["foreground_area"], side="right")) != area_bin:
                continue
            query = feature[i].copy()
            query[0] += np.log(area_factor)
            _, neighbors = tree.query(query, k=min(96, len(values)))
            for j in np.atleast_1d(neighbors):
                second = values[int(j)]
                a, b = first["fragment_token"], second["fragment_token"]
                key = tuple(sorted((a, b)))
                if first["split_unit_id"] == second["split_unit_id"] or key in seen or usage[b] >= 8:
                    continue
                ratio = min(first["foreground_area"], second["foreground_area"]) / max(first["foreground_area"], second["foreground_area"])
                aspect_ratio = max(first["bbox_aspect_ratio"], second["bbox_aspect_ratio"]) / min(first["bbox_aspect_ratio"], second["bbox_aspect_ratio"])
                low, high, _ = RATIO_BANDS[name]
                if not low <= ratio < high or aspect_ratio > 1.25:
                    continue
                if area_bin is not None and int(np.searchsorted(AREA_EDGES,
                        min(first["foreground_area"], second["foreground_area"]), side="right")) != area_bin:
                    continue
                seen.add(key)
                usage[a] += 1
                usage[b] += 1
                output.append(dict(pair_id="rachel-geocross-" + hashlib.sha256(repr(key).encode()).hexdigest()[:24],
                    split="train", label=False, fragment_a=_fragment(first), fragment_b=_fragment(second),
                    correspondence_path=None, translation_a_to_b_rc=None, translation_a_to_b_xy_cartesian=None,
                    label_origin="different_frozen_TRAIN_image_name_nonmatch",
                    negative_origin="area_aspect_heuristic_cross_" + name,
                    metadata=dict(min_max_area_ratio=ratio, max_min_aspect_ratio=aspect_ratio,
                        mining="TRAIN_metadata_area_aspect_nearest_neighbors", model_hard=False,
                        absolute_min_area_fallback=bool(min_area_targets is not None and area_bin is None))))
                chosen += 1
                break
        if chosen != requested:
            if area_bin is not None:
                # Preserve the ratio-band count first; relax only absolute area.
                requests.append((name, requested - chosen, area_factor, None))
            else:
                raise ValueError("unique geometric cross negative shortfall: %s bin%s %d/%d" % (name, area_bin, chosen, requested))
    return output


def _statistics(entries):
    counts, sources, lineages, area_histograms, ratio_histograms = Counter(), Counter(), set(), defaultdict(Counter), defaultdict(Counter)
    for item in entries:
        row = item["row"]
        counts[item["stratum"]] += 1
        counts["positive" if row["label"] else "negative"] += 1
        counts[("positive" if row["label"] else "negative") + "_tiny"] += int(_ratio(row) < .25)
        area_histograms["positive" if row["label"] else "negative"][_area_bin(row)] += 1
        ratio_histograms["positive" if row["label"] else "negative"][_ratio_band(row)] += 1
        sources[item["source_root"]] += 1
        lineages.update(row["fragment_" + side]["split_unit_id"] for side in "ab")
    return dict(total=len(entries), counts=dict(counts), source_counts=dict(sources), unique_train_lineages=len(lineages),
                absolute_min_area_bin_edges_px=list(AREA_EDGES),
                absolute_min_area_histogram={name: dict(hist) for name, hist in area_histograms.items()},
                min_max_ratio_histogram={name: dict(hist) for name, hist in ratio_histograms.items()})


def _allocate_bins(weights, total):
    """Largest-remainder apportionment in units of5 for exact nested40%."""
    values = np.asarray([max(0, weights.get(i, 0)) for i in range(len(AREA_EDGES) + 1)], float)
    if total == 0:
        return {i: 0 for i in range(len(values))}
    if total % 5 or values.sum() == 0:
        raise ValueError("area allocation requires nonempty weights and count divisible by5")
    exact = values / values.sum() * (total // 5)
    assigned = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - assigned), kind="stable")[:total // 5 - assigned.sum()]:
        assigned[i] += 1
    return {i: int(value * 5) for i, value in enumerate(assigned)}


def _runtime_compatible_positives(root, rows, count):
    """Construct usable new positives using the existing target gate, no masks.

    Previously unselected native candidates still need model-facing target
    compatibility. Only selected TRAIN contours/targets are opened; neither raw
    images nor held-out data are read. Invalid residuals are excluded, not fixed.
    """
    config = RachelDatasetConfig()
    @lru_cache(maxsize=2048)
    def contour(relative):
        points, valid = _load_contour(root / relative, config)
        return _FragmentArrays(None, None, points, valid)
    chosen, excluded = [], []
    for row in rows:
        try:
            _load_positive_target(root / row["correspondence_path"], contour(row["fragment_a"]["contour_path"]),
                                  contour(row["fragment_b"]["contour_path"]), config)
        except RachelDatasetError as error:
            if "contour/translation residual gate failed" not in str(error):
                raise
            excluded.append(row["pair_id"])
            continue
        chosen.append(row)
        if len(chosen) == count:
            return chosen, excluded
    raise ValueError("insufficient runtime-compatible native positives")


def _write_manifest(path, entries, *, seed, note):
    values = sorted(entries, key=lambda x: _rank(x["row"]["pair_id"], seed))
    payload = dict(schema_version=SCHEMA_VERSION, split="train", seed=seed, entries=values,
                   stats=_statistics(values), note=note, independent_new_source_manuscripts=False)
    Path(path).write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return payload["stats"]


def nested_stratified_subset(entries, *, seed):
    strata = defaultdict(list)
    for item in entries:
        strata[item["stratum"]].append(item)
    matched = []
    for name, values in strata.items():
        if len(values) % 5:
            raise ValueError("stratum is not divisible by5: " + name)
        values.sort(key=lambda x: _rank(x["row"]["pair_id"], seed + 1))
        matched.extend(values[:len(values) * 2 // 5])
    return matched


def build_composites(release_root, output_root, union_roots=(), *, seed=260909):
    root, destination = Path(release_root).resolve(strict=True), Path(output_root).resolve()
    if destination == root or root in destination.parents:
        raise ValueError("new manifests must not be inside original release")
    splits = json.loads((root / "pairs/lineage_splits.json").read_text())
    fragments = {}
    for row in _jsonl(root / "manifests/fragments.jsonl"):
        if splits.get(row["split_unit_id"]) == "train":
            fragments[row["fragment_token"]] = _fragment(row)
    positives, hard = [], []
    for row in _jsonl(root / "manifests/within_candidates.jsonl"):
        a, b = row["fragment_a_token"], row["fragment_b_token"]
        if a not in fragments or b not in fragments or not row.get("main_training_eligible"):
            continue
        selected = dict(pair_id=row["pair_id"], split="train", label=row["label"],
            fragment_a=fragments[a], fragment_b=fragments[b], correspondence_path=row["correspondence_path"],
            label_origin=row["label_origin"], negative_origin=row["negative_origin"])
        (positives if row["label"] else hard).append(selected)
    available_native_hard = len(hard)
    # The four native tiny negatives are not essential to the6k native-hard
    # quota; put the entire4500 tiny-negative quota in explicitly matched cross.
    hard = [row for row in hard if _ratio(row) >= .25]
    positives.sort(key=lambda x: _rank(x["pair_id"], seed))
    hard.sort(key=lambda x: _rank(x["pair_id"], seed))
    union_tiny, union_regular, signatures = [], [], set()
    for union_root in union_roots:
        union_root = Path(union_root).resolve(strict=True)
        summary = json.loads((union_root / "summary.json").read_text())
        if summary.get("source_root") != str(root) or summary.get("split") != "train" or not summary.get("save_assets"):
            raise ValueError("union source must be complete materialized TRAIN from this release")
        audit = {row["pair_id"]: row for row in _jsonl(union_root / "candidates.jsonl")}
        for row in _jsonl(union_root / "pairs/train.jsonl"):
            provenance = audit[row["pair_id"]]
            if splits.get(provenance["lineage_id"]) != "train":
                raise ValueError("union source violates frozen TRAIN split")
            if provenance["geometry_signature"] in signatures:
                continue
            signatures.add(provenance["geometry_signature"])
            tiny = _ratio(row) < .25
            entry = _entry(union_root, row, "union_positive_" + ("tiny" if tiny else "regular"))
            (union_tiny if tiny else union_regular).append(entry)
    for values in (union_tiny, union_regular):
        values.sort(key=lambda x: _rank(x["row"]["pair_id"], seed))
    # Multiples of5 make the nested40% comparator exactly match every stratum.
    n_tiny, n_regular = min(4500, len(union_tiny) // 5 * 5), min(4500, len(union_regular) // 5 * 5)
    native_count = 30000 - n_tiny - n_regular
    if len(positives) < native_count or len(hard) < 6000:
        raise ValueError("insufficient distinct native TRAIN candidates")
    selected_native, native_residual_exclusions = _runtime_compatible_positives(root, positives, native_count)
    entries = union_tiny[:n_tiny] + union_regular[:n_regular]
    for row in selected_native:
        entries.append(_entry(root, row, "native_positive"))
    for row in hard[:6000]:
        entries.append(_entry(root, row, "native_hard_negative"))
    # Match the realized tiny-positive rate on the negative side to avoid a
    # label shortcut if union supply is short. Native tiny positives are counted.
    positive_tiny = sum(_ratio(x["row"]) < .25 for x in entries if x["row"]["label"])
    native_hard_tiny = sum(_ratio(row) < .25 for row in hard[:6000])
    cross_tiny = max(0, positive_tiny - native_hard_tiny)
    cross_tiny = cross_tiny // 5 * 5
    regular_names = [name for name in RATIO_BANDS if name != "tiny"]
    desired_ratios = Counter(_ratio_band(x["row"]) for x in entries if x["row"]["label"])
    hard_ratios = Counter(_ratio_band(row) for row in hard[:6000])
    allocated_regular = _allocate_bins({i: desired_ratios[name] - hard_ratios[name]
                                       for i, name in enumerate(regular_names)}, 24000 - cross_tiny)
    ratio_counts = {"tiny": cross_tiny, **{name: allocated_regular[i] for i, name in enumerate(regular_names)}}
    area_targets = {}
    for name, count in ratio_counts.items():
        positive_bins = Counter(_area_bin(x["row"]) for x in entries if x["row"]["label"] and _ratio_band(x["row"]) == name)
        hard_bins = Counter(_area_bin(row) for row in hard[:6000] if _ratio_band(row) == name)
        area_targets[name] = _allocate_bins({i: positive_bins[i] - hard_bins[i] for i in range(len(AREA_EDGES) + 1)}, count)
    cross = _cross_negatives(list(fragments.values()), count=24000, tiny_count=cross_tiny, seed=seed,
                             min_area_targets=area_targets)
    for row in cross:
        # Absolute-area fallback may break5-divisibility in individual bins;
        # ratio-band totals remain exact and form the nested comparator strata.
        entries.append(_entry(root, row, "cross_negative_" + _ratio_band(row)))
    if len(entries) != 60000 or len({x["row"]["pair_id"] for x in entries}) != 60000:
        raise ValueError("60k must contain exactly60000 distinct pair IDs")
    matched = nested_stratified_subset(entries, seed=seed)
    old = [_entry(root, row, "original_24k") for row in _jsonl(root / "pairs/train.jsonl")]
    for item in old:
        if any(splits.get(item["row"]["fragment_" + side]["split_unit_id"]) != "train" for side in "ab"):
            raise ValueError("old training manifest is not frozen TRAIN")
    destination.mkdir(parents=True, exist_ok=False)
    full_stats = _write_manifest(destination / "train_60k.json", entries, seed=seed, note="30kpositive+6knativehard+24kunique area/aspect heuristic cross; no GPU mining")
    matched_stats = _write_manifest(destination / "train_matched24k.json", matched, seed=seed, note="Nested40% of every60k stratum; same mix, fixed120k exposures comparison")
    old_stats = _write_manifest(destination / "train_original24k.json", old, seed=seed, note="Original24k distribution control; not the matched data-size comparator")
    summary = dict(schema_version=SCHEMA_VERSION, status="complete" if positive_tiny >= 4500 else "complete_with_tiny_positive_shortfall",
        seed=seed, source_root=str(root), available_native_positive=len(positives), available_native_hard=available_native_hard,
        usable_native_hard_non_tiny=len(hard), native_tiny_negative_quota_replaced_by_matched_cross=available_native_hard-len(hard),
        available_union_tiny=len(union_tiny), available_union_regular=len(union_regular),
        cross_absolute_min_area_targets=area_targets,
        cross_absolute_min_area_fallback_count=sum(row["metadata"]["absolute_min_area_fallback"] for row in cross),
        native_runtime_residual_exclusion_count=len(native_residual_exclusions),
        native_runtime_residual_exclusions=native_residual_exclusions,
        required_tiny_positive=4500, selected_tiny_positive=positive_tiny,
        train60k=full_stats, matched24k=matched_stats, original24k=old_stats,
        matching="exact strata counts; matched24k nested subset of60k",
        validation_and_test_unchanged=True, heldout_masks_read=False, model_hard_mining=False,
        exposure_protocol="matched24k*5epochs=60k*2epochs=120000 pair presentations",
        claim_limit="Derived pairs from frozen TRAIN380; not new independent source manuscripts")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--union-root", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=260909)
    args = parser.parse_args()
    print(json.dumps(build_composites(args.release_root, args.output_root, args.union_root, seed=args.seed)), flush=True)


if __name__ == "__main__":
    main()
