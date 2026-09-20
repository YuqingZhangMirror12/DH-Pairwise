"""TRAIN-only materialized-data hooks shared by the two benchmark adapters.

No augmentation or correspondence regeneration occurs here. Ordered surviving
contour points are compacted only because the official ring-graph ports require
prefix validity; inherited supervision and original translation are preserved.
"""
from dataclasses import replace
from contextlib import contextmanager
import json
import hashlib
from pathlib import Path

import numpy as np

from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import collate_rachel_pairs


def materialized_manifest_rows(manifest_path):
    """Return audited source rows in exactly the materialized TRAIN order."""
    path = Path(manifest_path).expanduser().resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import SCHEMA
    if (value.get("schema_version") != SCHEMA or value.get("split") != "train"
            or not isinstance(value.get("entries"), list)):
        raise ValueError("materialized override must be a TRAIN manifest with entries")
    rows, seen = [], set()
    for entry in value["entries"]:
        source = entry.get("source_row")
        if not isinstance(source, dict) or source.get("split") != "train":
            raise ValueError("materialized source row must remain TRAIN")
        if (not isinstance(entry.get("pair_id"), str) or not entry["pair_id"]
                or type(entry.get("label")) is not bool
                or entry["pair_id"] != source.get("pair_id")
                or entry["label"] is not source.get("label")):
            raise ValueError("materialized TRAIN identity/label differs from source")
        if entry["pair_id"] in seen:
            raise ValueError("duplicate materialized TRAIN pair_id")
        seen.add(entry["pair_id"])
        rows.append(source)
    if not rows or sum(row["label"] for row in rows) * 2 != len(rows):
        raise ValueError("materialized TRAIN must be nonempty and class balanced")
    return tuple(rows)


def training_hook_identity():
    """Bind the shared data/loss adapter and its serialized sample decoder."""
    source = Path(__file__).resolve()
    loader = source.parents[1] / "pairwise_data" / "rachel_materialized_dataset.py"
    recall = source.parents[3] / "experiments" / "rachel_n512_formal_30k" / "recall_operating_points.py"
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, loader, recall)}


def materialized_train_dataset(manifest_path):
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import MaterializedRachelDataset
    rows = materialized_manifest_rows(manifest_path)
    dataset = MaterializedRachelDataset(manifest_path)
    if dataset.split != "train" or len(dataset) != len(rows):
        raise ValueError("materialized TRAIN loader/manifest population differs")
    return dataset


def selected_manifest_path(root, split, train_materialized_manifest=None):
    if split not in {"train", "val"}:
        raise ValueError("benchmark manifest access is TRAIN/VAL only")
    return (Path(train_materialized_manifest) if split == "train" and train_materialized_manifest is not None
            else Path(root) / "pairs" / (split + ".jsonl"))


@contextmanager
def benchmark_manifest_lines(root, split, train_materialized_manifest=None):
    path = selected_manifest_path(root, split, train_materialized_manifest)
    if split == "train" and train_materialized_manifest is not None:
        yield (json.dumps(row) for row in materialized_manifest_rows(path))
    else:
        with path.open("r", encoding="utf-8") as stream:
            yield stream


def compact_benchmark_sample(sample):
    """Compact valid contour slots; remap, never reconstruct, original targets."""
    valid = [np.asarray(getattr(sample, "contour_valid_" + side), bool) for side in "ab"]
    if all(np.array_equal(v, np.arange(len(v)) < int(v.sum())) for v in valid):
        return sample
    indices = [np.flatnonzero(v) for v in valid]
    maps = []
    for v, chosen in zip(valid, indices):
        remap = np.full(len(v), -2, dtype=np.int64)
        remap[chosen] = np.arange(len(chosen))
        maps.append(remap)
    changes = {}
    for index, side in enumerate("ab"):
        chosen = indices[index]
        target = np.asarray(getattr(sample, "target_" + side), np.int64)[chosen].copy()
        matched = target >= 0
        if np.any(target[matched] >= len(maps[1 - index])):
            raise ValueError("materialized correspondence index exceeds opposite contour")
        target[matched] = maps[1 - index][target[matched]]
        changes["points_rc_" + side] = np.asarray(getattr(sample, "points_rc_" + side))[chosen].copy()
        changes["contour_valid_" + side] = np.ones(len(chosen), dtype=bool)
        changes["target_" + side] = target
    return replace(sample, **changes)


def collate_benchmark_pairs(samples, contour_cap=512):
    return collate_rachel_pairs([compact_benchmark_sample(sample) for sample in samples],
        contour_cap=contour_cap)


def supervised_contour_masks(batch):
    """Unknown inherited targets (-2) are not negative correspondence cells."""
    return tuple(np.asarray(getattr(batch, "contour_valid_" + side), bool)
        & (np.asarray(getattr(batch, "target_" + side)) != -2) for side in "ab")
