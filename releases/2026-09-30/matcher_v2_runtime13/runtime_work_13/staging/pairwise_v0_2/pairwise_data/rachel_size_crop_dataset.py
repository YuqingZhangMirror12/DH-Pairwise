"""Unvalidated, train-only size-crop proposals with coupled label acceptance.

A deterministic positive/negative group shares the augmentation coin and each
attempt's target area ratio and rectangle aspect. BOTH original samples must
pass the SAME attempt; otherwise neither changes. This prevents an unequal
positive/negative *crop occurrence rate* caused by positive seam checks. It does
not prove that the augmented shape distributions have no label-dependent cues.

There is no new experiment registration, file access, or global preprocessing
change here. Labels/GT are used only for training crop protection. The returned
RachelPairSample has no extra model fields. Unchanged/rejected samples retain
their released targets and exact object identity.

Accepted samples keep pair/fragment tokens. DO NOT reuse token-only fragment
feature caches across these crops/epochs: train directly from the returned
masks, or use a cache keyed by actual input content and epoch. With persistent
DataLoader workers, set_epoch on the parent dataset does not update worker
copies; recreate workers or explicitly propagate the epoch before using them.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
import operator

import numpy as np

from .rachel_size_crop import (
    SizeCropConfig,
    _mask_2d,
    _original_gt_edges,
    crop_training_pair,
)
from .rachel_training_dataset import RachelPairSample

SCHEMA_VERSION = "rachel-training-coupled-size-crop-dataset/1"


@dataclass(frozen=True)
class SizeCropDatasetConfig:
    probability: float = 0.5
    ratio_min: float = 0.10
    ratio_max: float = 0.24
    max_attempts: int = 4
    aspect_min: float = 0.5
    aspect_max: float = 2.0
    cache_groups: int = 8
    crop_config: SizeCropConfig = field(default_factory=SizeCropConfig)

    def __post_init__(self):
        for name in ("probability", "ratio_min", "ratio_max", "aspect_min", "aspect_max"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(name + " must be a finite real number")
            object.__setattr__(self, name, float(value))
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must lie in [0,1]")
        if not 0 < self.ratio_min <= self.ratio_max < 1:
            raise ValueError("ratio bounds must satisfy 0 < ratio_min <= ratio_max < 1")
        if not 0 < self.aspect_min <= self.aspect_max:
            raise ValueError("aspect bounds must satisfy 0 < aspect_min <= aspect_max")
        for name in ("max_attempts", "cache_groups"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if not isinstance(self.crop_config, SizeCropConfig):
            raise TypeError("crop_config must be SizeCropConfig")


def _sha256(*parts):
    payload = json.dumps((SCHEMA_VERSION,) + parts, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _rectangle(center, height, aspect, shape):
    """Nested integer rectangles centred on a token, clipped to its canvas."""
    width = max(1, int(round(height / aspect)))
    r0 = int(math.floor(float(center[0]) - (height - 1) / 2.0))
    c0 = int(math.floor(float(center[1]) - (width - 1) / 2.0))
    return max(0, r0), min(shape[0], r0 + height), max(0, c0), min(shape[1], c0 + width)


def _rectangle_area(integral, bounds):
    r0, r1, c0, c1 = bounds
    return int(integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0])


def _area_target_bounds(integral, center, target_area, aspect):
    """Binary-search rectangle extent using exact original foreground counts.

    Discrete pixels may prevent equality with target_area. Return the closest
    visited crossing candidate; the caller independently enforces ratio bounds.
    """
    shape = (integral.shape[0] - 1, integral.shape[1] - 1)
    low = 1
    high = int(math.ceil(max(2 * shape[0] + 1, (2 * shape[1] + 1) * aspect)))
    best = None
    while low <= high:
        height = (low + high) // 2
        bounds = _rectangle(center, height, aspect, shape)
        area = _rectangle_area(integral, bounds)
        # Prefer undershooting on an equal-distance tie, then smaller extent.
        key = (abs(area - target_area), area > target_area, height)
        if best is None or key < best[0]:
            best = (key, bounds, area)
        if area < target_area:
            low = height + 1
        else:
            high = height - 1
    return best[1], best[2]


class RachelSizeCropDataset:
    """Lazy train-only wrapper; construction reads metadata, never samples.

    pair_metadata must describe base order exactly. Group membership is fixed
    by SHA256 sorting separately within each label, not Python hash or input
    access order. set_epoch changes the stable proposal streams and clears the
    bounded per-worker LRU. group_diagnostics(index) materializes the same group
    if needed and returns a detached, JSON-ready diagnostics dictionary.
    """

    def __init__(self, base, *, pair_metadata, seed, split="train",
                 config=SizeCropDatasetConfig()):
        if split != "train" or getattr(base, "split", None) != "train":
            raise ValueError("size crop augmentation is train-only; both split and base.split must be 'train'")
        if not isinstance(config, SizeCropDatasetConfig):
            raise TypeError("config must be SizeCropDatasetConfig")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        metadata = []
        ids = set()
        for item in pair_metadata:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("each pair_metadata entry must be (pair_id,label)")
            pair_id, label = item
            if not isinstance(pair_id, str) or not pair_id or pair_id in ids:
                raise ValueError("pair_metadata IDs must be nonempty unique strings")
            if not isinstance(label, (Real, np.bool_)) or not np.isfinite(label) or label not in (0, 1):
                raise ValueError("pair_metadata labels must be 0 or 1")
            ids.add(pair_id)
            metadata.append((pair_id, int(label)))
        if len(metadata) != len(base):
            raise ValueError("pair_metadata length must equal base length")
        positive = [i for i, (_, label) in enumerate(metadata) if label == 1]
        negative = [i for i, (_, label) in enumerate(metadata) if label == 0]
        if len(positive) != len(negative):
            raise ValueError("pair_metadata must contain balanced positive and negative counts")
        self.base = base
        self.dataset = base
        self.seed = int(seed)
        self.split = "train"
        self.root = getattr(base, "root", None)
        self.config = config
        self.pair_metadata = tuple(metadata)
        self.epoch = 0
        self._cache = OrderedDict()
        self._cache_hits = self._cache_misses = 0
        sort_key = lambda i: (_sha256(self.seed, "group-order", metadata[i][0]), metadata[i][0])
        positive.sort(key=sort_key)
        negative.sort(key=sort_key)
        self._groups = tuple(zip(positive, negative))
        self._index_to_group = [0] * len(metadata)
        for group_index, indices in enumerate(self._groups):
            for index in indices:
                self._index_to_group[index] = group_index

    def __len__(self):
        return len(self.pair_metadata)

    def set_epoch(self, epoch):
        if isinstance(epoch, (bool, np.bool_)) or not isinstance(epoch, (int, np.integer)) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = int(epoch)
        self._cache.clear()
        self._cache_hits = self._cache_misses = 0

    def cache_info(self):
        return {"size": len(self._cache), "capacity": self.config.cache_groups,
                "hits": self._cache_hits, "misses": self._cache_misses, "epoch": self.epoch}

    def _index(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("size crop dataset index out of range")
        return index

    def _rng(self, pair_ids, tag, *parts):
        seed_bytes = _sha256(self.seed, self.epoch, pair_ids, tag, *parts)
        return np.random.Generator(np.random.PCG64(int.from_bytes(seed_bytes, "big")))

    def _context(self, sample, pair_ids):
        ma, mb = _mask_2d(sample.mask_a), _mask_2d(sample.mask_b)
        area_a, area_b = int(ma.sum()), int(mb.sum())
        if not min(area_a, area_b):
            return None, "empty_original_fragment"
        if area_a == area_b:
            side = ("a", "b")[int(self._rng(pair_ids, "side", sample.pair_id).integers(2))]
        else:
            side = "a" if area_a < area_b else "b"
        mask, other_area = (ma, area_b) if side == "a" else (mb, area_a)
        points = np.asarray(getattr(sample, "points_rc_" + side), dtype=np.float64)
        valid = np.asarray(getattr(sample, "contour_valid_" + side), dtype=np.bool_)
        if points.ndim != 2 or points.shape[1:] != (2,) or valid.shape != (len(points),):
            return None, "invalid_original_contour"
        if bool(sample.label):
            try:
                edges = _original_gt_edges(sample)
            except ValueError:
                return None, "invalid_original_gt"
            tokens = np.unique(edges[:, 0 if side == "a" else 1])
        else:
            tokens = np.flatnonzero(valid)
        if len(tokens):
            anchors = points[tokens]
            keep = (np.isfinite(anchors).all(axis=1) & (anchors >= 0).all(axis=1)
                    & (anchors[:, 0] < mask.shape[0]) & (anchors[:, 1] < mask.shape[1]))
            tokens = tokens[keep]
        if not len(tokens):
            return None, "no_original_gt_anchors" if bool(sample.label) else "no_valid_contour_anchors"
        integral = np.pad(mask.astype(np.int64).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        return {"side": side, "integral": integral, "points": points, "tokens": tokens,
                "area": int(mask.sum()), "other_area": other_area}, "ready"

    def _attempt_member(self, sample, context, pair_ids, attempt, ratio, aspect):
        side = context["side"]
        info = {"side": side, "accepted": False, "target_ratio": ratio, "aspect_ratio": aspect}
        target_area = ratio * context["other_area"]
        if target_area >= context["area"]:
            info["reason"] = "target_area_does_not_delete_pixels"
            return None, info
        rng = self._rng(pair_ids, "anchor", attempt, sample.pair_id)
        tokens = context["tokens"]
        token = int(tokens[int(rng.integers(len(tokens)))])
        bounds, proposed_area = _area_target_bounds(context["integral"], context["points"][token], target_area, aspect)
        info.update(anchor_token_index=token, bounds_rc=list(bounds),
                    target_area_px=float(target_area), proposed_area_px=proposed_area)
        result = crop_training_pair(sample, side=side, bounds_rc=bounds, config=self.config.crop_config)
        info.update(reason=result.reason, geometry=deepcopy(result.diagnostics))
        if not result.accepted:
            return None, info
        # Independently enforce actual post-crop ratio, not just proposal area.
        aa = int(np.count_nonzero(result.sample.mask_a))
        ab = int(np.count_nonzero(result.sample.mask_b))
        actual_ratio = min(aa, ab) / max(aa, ab) if min(aa, ab) > 0 else None
        info["post_area_ratio"] = actual_ratio
        if actual_ratio is None or not self.config.ratio_min <= actual_ratio <= self.config.ratio_max:
            info["reason"] = "post_area_ratio_outside_configured_range"
            return None, info
        info["accepted"] = True
        return result.sample, info

    def _materialize(self, group_index):
        indices = self._groups[group_index]
        pair_ids = tuple(self.pair_metadata[index][0] for index in indices)
        originals = tuple(self.base[index] for index in indices)
        for index, sample in zip(indices, originals):
            if not isinstance(sample, RachelPairSample):
                raise TypeError("base must return RachelPairSample")
            if sample.pair_id != self.pair_metadata[index][0] or float(sample.label) != self.pair_metadata[index][1]:
                raise ValueError("base sample identity/label disagrees with pair_metadata")
        coin = float(self._rng(pair_ids, "coin").random())
        diagnostics = {"schema_version": SCHEMA_VERSION, "unvalidated_prototype": True,
            "group_pair_ids": list(pair_ids), "epoch": self.epoch, "seed": self.seed,
            "coin": coin, "probability": self.config.probability, "accepted": False,
            "accepted_attempt": None, "attempts": []}
        if coin >= self.config.probability:
            diagnostics["reason"] = "shared_coin_skipped"
            return originals, diagnostics
        contexts = tuple(self._context(sample, pair_ids) for sample in originals)
        diagnostics["context_reasons"] = [reason for _, reason in contexts]
        if any(context is None for context, _ in contexts):
            diagnostics["reason"] = "group_context_unavailable"
            return originals, diagnostics
        for attempt in range(1, self.config.max_attempts + 1):
            rng = self._rng(pair_ids, "shared-attempt", attempt)
            ratio = float(rng.uniform(self.config.ratio_min, self.config.ratio_max))
            aspect = float(rng.uniform(self.config.aspect_min, self.config.aspect_max))
            members = tuple(self._attempt_member(sample, context, pair_ids, attempt, ratio, aspect)
                            for sample, (context, _) in zip(originals, contexts))
            together = all(output is not None for output, _ in members)
            diagnostics["attempts"].append({"attempt": attempt, "target_ratio": ratio,
                "aspect_ratio": aspect, "positive": members[0][1], "negative": members[1][1],
                "coupled_accepted": together})
            if together:
                diagnostics.update(accepted=True, accepted_attempt=attempt, reason="coupled_accepted")
                return tuple(output for output, _ in members), diagnostics
        diagnostics["reason"] = "no_jointly_accepted_attempt"
        return originals, diagnostics

    def _get_group(self, group_index):
        key = (self.epoch, group_index)
        if key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._cache_misses += 1
        value = self._materialize(group_index)
        self._cache[key] = value
        if len(self._cache) > self.config.cache_groups:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index):
        index = self._index(index)
        group_index = self._index_to_group[index]
        outputs, _ = self._get_group(group_index)
        return outputs[0 if self._groups[group_index][0] == index else 1]

    def group_diagnostics(self, index):
        index = self._index(index)
        _, diagnostics = self._get_group(self._index_to_group[index])
        return deepcopy(diagnostics)


__all__ = ["RachelSizeCropDataset", "SizeCropDatasetConfig", "SCHEMA_VERSION"]
