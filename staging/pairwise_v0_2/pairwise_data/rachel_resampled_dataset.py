"""Re-extract real contour tokens for 512-control / denser-1024 experiments.

Input contours depend only on each fragment mask (Gaussian sigma=3), never its
label or translation. Only the supervision path uses the existing pair label
and A-to-B translation: dense mutual nearest neighbours within 3 px, the release
continuous-seam filter, then its error-sorted greedy one-to-one token mapping.

Rebuilt 512 targets are *not claimed* to equal the original release targets.
That requires a separate comparison: this wrapper works from released centred
masks and translations, while the release builder worked in parent coordinates.
Pair identities, order, labels, translations and split are not reselected.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .rachel_preprocess import (
    _accepted_continuous_seam,
    _dense_external_contour,
    extract_ordered_outer_contour,
    recover_mutual_contour_correspondences,
)
from .rachel_training_dataset import RachelPairSample, _readonly, collate_rachel_pairs

SCHEMA_VERSION = "rachel-mask-resampled-dense-seam-targets/1"
SMOOTHING_SIGMA = 3.0
DENSE_MATCH_DISTANCE_PX = 3.0


class RachelResamplingError(ValueError):
    """The released sample cannot support the fixed resampling protocol."""


@dataclass(frozen=True)
class _FragmentGeometry:
    points_rc: np.ndarray
    valid: np.ndarray
    dense_rc: np.ndarray


def _binary_mask(value):
    mask = np.asarray(value)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2 or not np.isfinite(mask).all() or not np.all((mask == 0) | (mask == 1)):
        raise RachelResamplingError("fragment mask must be binary [H,W] or [1,H,W]")
    if not mask.any():
        raise RachelResamplingError("fragment mask must contain foreground")
    return np.ascontiguousarray(mask, dtype=np.bool_)


def rebuild_correspondence_targets(points_a_rc, points_b_rc, dense_a_rc, dense_b_rc,
                                   *, label, translation_a_to_b_rc):
    """Return reciprocal token targets plus counts from the fixed GT path.

    Token-to-dense distances order proposals but do not reject them, matching
    the release's ``_token_correspondences`` implementation. The 3 px threshold
    applies to dense mutual matches before the continuous-seam filter.
    """
    a, b = np.asarray(points_a_rc, float), np.asarray(points_b_rc, float)
    da, db = np.asarray(dense_a_rc, float), np.asarray(dense_b_rc, float)
    for points in (a, b, da, db):
        if points.ndim != 2 or points.shape[1:] != (2,) or not len(points) or not np.isfinite(points).all():
            raise RachelResamplingError("token and dense contours must be finite nonempty [N,2]")
    target_a, target_b = np.full(len(a), -1, dtype=np.int64), np.full(len(b), -1, dtype=np.int64)
    diagnostics = {"schema_version": SCHEMA_VERSION, "dense_match_distance_px": DENSE_MATCH_DISTANCE_PX,
        "dense_raw_match_count": 0, "dense_accepted_match_count": 0, "token_match_count": 0,
        "original_release_target_equality_claimed": False}
    if not bool(label):
        return _readonly(target_a, np.int64), _readonly(target_b, np.int64), diagnostics
    translation = np.asarray(translation_a_to_b_rc, dtype=float)
    if translation.shape != (2,) or not np.isfinite(translation).all():
        raise RachelResamplingError("positive translation_a_to_b_rc must be finite [2]")
    aligned_a = da + translation
    raw = recover_mutual_contour_correspondences(aligned_a, db, max_distance_px=DENSE_MATCH_DISTANCE_PX)
    accepted = _accepted_continuous_seam(raw, aligned_a, db)
    diagnostics.update(dense_raw_match_count=len(raw), dense_accepted_match_count=len(accepted))
    if not len(accepted):
        raise RachelResamplingError("positive pair has no accepted dense seam after target rebuilding")
    # Translation cancels in each fragment's dense-to-token nearest-neighbour
    # query; use its own model frame, just as the release uses its parent frame.
    distance_a, token_a = cKDTree(a).query(da[accepted[:, 0]], k=1)
    distance_b, token_b = cKDTree(b).query(db[accepted[:, 1]], k=1)
    proposals = sorted((float(ea + eb), int(i), int(j)) for ea, eb, i, j in
                       zip(distance_a, distance_b, token_a, token_b))
    for _, i, j in proposals:
        if target_a[i] >= 0 or target_b[j] >= 0:
            continue
        target_a[i], target_b[j] = j, i
    diagnostics["token_match_count"] = int(np.count_nonzero(target_a >= 0))
    return _readonly(target_a, np.int64), _readonly(target_b, np.int64), diagnostics


class RachelResampledDataset:
    """Wrap selected Rachel samples without changing split or pair population.

    Example: ``RachelResampledDataset(base, contour_cap=1024)``. Contours remain
    variable length: a 700-point dense boundary produces 700 valid tokens at
    cap1024, not 1024 duplicated/padded points. Batch padding belongs to collate.
    Each worker owns a bounded LRU of mask-derived fragment geometry only.
    """

    def __init__(self, dataset, *, contour_cap=1024, cache_size=256):
        if type(contour_cap) is not int or contour_cap < 4:
            raise ValueError("contour_cap must be an integer of at least four")
        if type(cache_size) is not int or cache_size <= 0:
            raise ValueError("cache_size must be a positive integer")
        self.dataset = dataset
        self.contour_cap = contour_cap
        self.cache_size = cache_size
        self.root = getattr(dataset, "root", None)
        self.split = getattr(dataset, "split", None)
        self._cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def __len__(self):
        return len(self.dataset)

    def _geometry(self, token, raw_mask):
        mask = _binary_mask(raw_mask)
        fingerprint = hashlib.blake2b(mask.tobytes(), digest_size=16).digest()
        key = (str(token), mask.shape, fingerprint)
        if key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._cache_misses += 1
        points, valid = extract_ordered_outer_contour(mask, cap=self.contour_cap, smoothing_sigma=SMOOTHING_SIGMA)
        geometry = _FragmentGeometry(points, valid, _readonly(_dense_external_contour(mask), np.float32))
        self._cache[key] = geometry
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return geometry

    def cache_info(self):
        return {"size": len(self._cache), "capacity": self.cache_size,
                "hits": self._cache_hits, "misses": self._cache_misses}

    def __getitem__(self, index):
        sample = self.dataset[index]
        if not isinstance(sample, RachelPairSample):
            raise TypeError("base dataset must return RachelPairSample")
        # The model-facing contour path finishes before label/translation use.
        first = self._geometry(sample.fragment_a_token, sample.mask_a)
        second = self._geometry(sample.fragment_b_token, sample.mask_b)
        if float(sample.label) not in (0., 1.) or bool(sample.translation_valid) != bool(sample.label):
            raise RachelResamplingError("sample label/translation-valid contract differs")
        target_a, target_b, _ = rebuild_correspondence_targets(first.points_rc, second.points_rc,
            first.dense_rc, second.dense_rc, label=sample.label,
            translation_a_to_b_rc=sample.translation_a_to_b_rc)
        return replace(sample, points_rc_a=first.points_rc, points_rc_b=second.points_rc,
            contour_valid_a=first.valid, contour_valid_b=second.valid, target_a=target_a, target_b=target_b)


def collate_resampled_rachel_pairs(samples: Sequence[RachelPairSample], contour_cap=1024):
    """Use with ``functools.partial(..., contour_cap=N)`` in a DataLoader."""
    return collate_rachel_pairs(samples, contour_cap=contour_cap)


__all__ = ["RachelResampledDataset", "RachelResamplingError", "rebuild_correspondence_targets",
           "collate_resampled_rachel_pairs", "SCHEMA_VERSION"]
