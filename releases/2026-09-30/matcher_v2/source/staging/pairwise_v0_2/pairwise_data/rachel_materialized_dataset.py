"""Shared fixed E1 TRAIN samples for architecture and data-volume comparisons.

Only this archive loader bypasses the clean-release residual validation: its
targets are already inherited by RachelWeatheredDataset, including ignored -2
tokens and nonzero boundary gaps. Model inputs never contain source metadata.
"""
from __future__ import annotations

from dataclasses import fields
import json
import os
from pathlib import Path

import numpy as np

from .rachel_training_dataset import RachelPairSample, _readonly

SCHEMA = "rachel-materialized-e1-train/1"
STRINGS = ("pair_id", "fragment_a_token", "fragment_b_token")
BOOLS = ("contour_valid_a", "contour_valid_b", "translation_valid")
INTS = ("target_a", "target_b")


def save_sample(path, sample, report):
    """Write exact model inputs/targets and a separate, non-model E1 sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {}
    for field in fields(RachelPairSample):
        name, value = field.name, getattr(sample, field.name)
        if name in ("mask_a", "mask_b"):
            values[name + "_packed"] = np.packbits(np.asarray(value, np.bool_), axis=-1)
            values[name + "_shape"] = np.asarray(value.shape, np.int64)
        else:
            values[name] = np.asarray(value)
    values["report_json"] = np.asarray(json.dumps(report, sort_keys=True, allow_nan=False))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def load_sample(path):
    with np.load(path, allow_pickle=False) as archive:
        values = {}
        for field in fields(RachelPairSample):
            name = field.name
            if name in ("mask_a", "mask_b"):
                shape = tuple(int(x) for x in archive[name + "_shape"])
                value = np.unpackbits(archive[name + "_packed"], axis=-1)[..., :shape[-1]]
                values[name] = _readonly(value.reshape(shape), np.float32)
            elif name in STRINGS:
                values[name] = str(archive[name].item())
            elif name == "label":
                values[name] = np.float32(archive[name].item())
            elif name == "translation_valid":
                values[name] = np.bool_(archive[name].item())
            else:
                dtype = np.bool_ if name in BOOLS else np.int64 if name in INTS else np.float32
                values[name] = _readonly(archive[name], dtype)
        report = json.loads(str(archive["report_json"].item()))
    sample = RachelPairSample(**values)
    if sample.label not in (0, 1) or bool(sample.translation_valid) != bool(sample.label):
        raise ValueError("materialized source pair/translation labels differ")
    if bool(report["pose_supervision_enabled"]) != (bool(sample.label) and not report["changed_pair"]):
        raise ValueError("materialized E1 pose sidecar differs")
    for side, other in (("a", "b"), ("b", "a")):
        target = getattr(sample, "target_" + side)
        valid = getattr(sample, "contour_valid_" + side)
        opposite = getattr(sample, "target_" + other)
        indices = np.flatnonzero(target >= 0)
        if len(target) != len(valid) or np.any(target < -2):
            raise ValueError("invalid inherited target shape/value")
        if len(indices) and (not sample.label or np.any(~valid[indices]) or np.any(target[indices] >= len(opposite))):
            raise ValueError("invalid inherited correspondence")
        if len(indices) and not np.array_equal(opposite[target[indices]], indices):
            raise ValueError("inherited correspondences are not reciprocal")
    return sample, report


class MaterializedRachelDataset:
    """RachelPairSample-compatible, TRAIN-only fixed augmentation table."""
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        record = json.loads(self.manifest_path.read_text())
        if record.get("schema_version") != SCHEMA or record.get("split") != "train":
            raise ValueError("requires a materialized TRAIN manifest")
        self.root = Path(record["artifact_root"]).resolve(strict=True)
        self.split, self.contour_cap = "train", 512
        self.entries = record["entries"]
        self.rows = [entry["source_row"] for entry in self.entries]
        self.stats = record.get("stats", {})
        self.protocol = record.get("protocol", {})
        ids = [entry["pair_id"] for entry in self.entries]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("empty or duplicate materialized TRAIN pair IDs")
        for entry in self.entries:
            if entry["source_row"].get("split") != "train":
                raise ValueError("held-out source in materialized TRAIN manifest")
            path = (self.root / entry["artifact_path"]).resolve()
            if self.root not in path.parents:
                raise ValueError("artifact outside materialized root")

    def __len__(self):
        return len(self.entries)

    def weathered(self, index):
        entry = self.entries[index]
        sample, report = load_sample(self.root / entry["artifact_path"])
        if sample.pair_id != entry["pair_id"] or bool(sample.label) != entry["label"]:
            raise ValueError("archive pair ID/label differs from manifest")
        return sample, report

    def __getitem__(self, index):
        return self.weathered(index)[0]

    def get_report(self, index):
        return self.weathered(index)[1]

    def set_epoch(self, epoch):
        """Intentional no-op: fixed physical samples across epochs/models."""


class MaterializedWeatheredDataset(MaterializedRachelDataset):
    def __getitem__(self, index):
        return self.weathered(index)
