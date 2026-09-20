"""Cap-aware loaders and mask-only inference contours for density ablations.

Training samples must already contain the desired resampled contours/targets.
Inference resampling uses masks and endpoint identities only, leaves masks and
other metadata untouched, and marks all correspondence targets as ignored.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from functools import partial
import hashlib

import numpy as np

from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour
from staging.pairwise_v0_2.pairwise_data.rachel_resampled_dataset import SMOOTHING_SIGMA, _binary_mask
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelBatch, collate_rachel_pairs
from staging.pairwise_v0_2.training import rachel_n512_runner as runner


def _validate_cap(contour_cap):
    if type(contour_cap) is not int or contour_cap < 4:
        raise ValueError("contour_cap must be an integer of at least four")


def make_ablation_loader(dataset, indices, *, batch_size, num_workers, seed, contour_cap):
    """Reuse the original order/worker seeding with an explicitly capped collate.

    This function does not resample a dataset implicitly. Use the resampled
    dataset wrapper first when testing denser model inputs and rebuilt targets.
    """
    _validate_cap(contour_cap)
    if hasattr(dataset, "contour_cap") and dataset.contour_cap != contour_cap:
        raise ValueError("dataset resampling cap differs from requested batch cap")
    loader = runner._loader(dataset, indices, batch_size=batch_size, num_workers=num_workers, seed=seed)
    # Workers begin on iteration, so this changes only this loader's collate
    # before any worker sees it; no shared runner global is temporarily patched.
    loader.collate_fn = partial(collate_rachel_pairs, contour_cap=contour_cap)
    return loader


class InputContourResampler:
    """Mask-only, bounded-LRU contour re-extraction for target-blind inference."""

    def __init__(self, contour_cap, cache_size=256):
        _validate_cap(contour_cap)
        if type(cache_size) is not int or cache_size <= 0:
            raise ValueError("cache_size must be a positive integer")
        self.contour_cap = contour_cap
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def _contour(self, token, raw_mask):
        mask = _binary_mask(raw_mask)
        fingerprint = hashlib.blake2b(mask.tobytes(), digest_size=16).digest()
        key = (str(token), mask.shape, fingerprint)
        if key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._cache_misses += 1
        contour = extract_ordered_outer_contour(mask, cap=self.contour_cap, smoothing_sigma=SMOOTHING_SIGMA)
        self._cache[key] = contour
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return contour

    def cache_info(self):
        return {"size": len(self._cache), "capacity": self.cache_size,
                "hits": self._cache_hits, "misses": self._cache_misses}

    def resample_batch(self, batch: RachelBatch) -> RachelBatch:
        """Replace only contours, validity and all-ignore correspondence targets.

        No labels, translation targets, old correspondences or old contour
        coordinates are inspected. Metadata is preserved by dataclass replace.
        This batch is for model inference, not supervised loss calculation.
        """
        if not isinstance(batch, RachelBatch):
            raise TypeError("resample_batch expects RachelBatch")
        count = len(batch.pair_ids)
        if not count or any(len(value) != count for value in (
                batch.mask_a, batch.mask_b, batch.fragment_a_tokens, batch.fragment_b_tokens)):
            raise ValueError("batch masks and endpoint identities must align")
        replacement = {}
        for side in ("a", "b"):
            points = np.zeros((count, self.contour_cap, 2), dtype=np.float32)
            valid = np.zeros((count, self.contour_cap), dtype=np.bool_)
            masks = getattr(batch, "mask_" + side)
            tokens = getattr(batch, "fragment_" + side + "_tokens")
            for index, (token, mask) in enumerate(zip(tokens, masks)):
                contour, contour_valid = self._contour(token, mask)
                length = len(contour)
                points[index, :length] = contour
                valid[index, :length] = contour_valid
            replacement["points_rc_" + side] = points
            replacement["contour_valid_" + side] = valid
            replacement["target_" + side] = np.full((count, self.contour_cap), -2, dtype=np.int64)
        return replace(batch, **replacement)


__all__ = ["make_ablation_loader", "InputContourResampler"]
