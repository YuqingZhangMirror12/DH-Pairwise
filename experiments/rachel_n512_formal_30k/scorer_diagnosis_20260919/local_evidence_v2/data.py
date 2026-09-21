"""TRAIN-only conditional resampling and candidate targets for joint D.

These are a new sampling distribution over the existing fixed S7 masks, NOT
newly synthesized images. No REAL/OOD labels/scores drive the sampler.
"""
import json
from pathlib import Path

import numpy as np


class TrainingEvidence:
    def __init__(self, path, cache):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
        lookup = {r["pair_id"]: r for r in rows}
        if len(rows) != len(cache) or len(lookup) != len(rows) or set(lookup) != {r["pair_id"] for r in cache.records}:
            raise ValueError("TRAIN diagnostics must match the fixed S7 cache exactly")
        self.rows = [lookup[r["pair_id"]] for r in cache.records]
        if any(int(a["label"]) != int(b["label"]) for a, b in zip(self.rows, cache.records)):
            raise ValueError("diagnostic/cache labels disagree")
        self.labels = np.array([r["label"] for r in self.rows], bool)
        self.correct = np.array([r["raw_layout20_success"] for r in self.rows], bool)
        self.known = np.array([bool(r["layout_valid"]) and (not r["label"] or bool(r["gt_translation_valid"])) for r in self.rows])
        n = np.array([r["unique_endpoints_min"] for r in self.rows])
        rms = np.array([r["inlier_weighted_rms_px"] or 0. for r in self.rows])
        mass = np.array([r["inlier_q_mass"] for r in self.rows])
        good = self.labels & self.correct & self.known
        positive_rms = np.quantile(rms[good], .75)
        negative_mass = np.quantile(mass[~self.labels], .75)
        self.pools = dict(
            low_support=np.flatnonzero(good & (n <= 32)),
            partial=np.flatnonzero(good & np.array([r["s7_recipe"] == "partial_curve" or
                r["endpoint_coverage_min"] <= .15 for r in self.rows])),
            high_residual=np.flatnonzero(good & (rms >= positive_rms)),
            hard_negative=np.flatnonzero(~self.labels & (mass >= negative_mass)),
        )
        self.summary = dict(path=str(path), count=len(rows), candidate_known=int(self.known.sum()),
            correct_candidates=int(self.correct.sum()), pool_counts={k: len(v) for k,v in self.pools.items()},
            high_residual_train_quantile_px=float(positive_rms), hard_negative_train_mass_quantile=float(negative_mass),
            candidate_label="negative pair=0; positive final predicted pose within20px of TRAIN GT=1, otherwise0",
            candidate_known_definition="numerical candidate and (negative pair or available TRAIN pose GT)",
            data_change="conditional resampling of fixed S7, not new masks or Matcher retraining",
            threshold_source="TRAIN evidence only; no REAL/OOD lookup")

    def order(self, epoch, seed=260913):
        rng = np.random.default_rng(seed + 1000003*epoch)
        counts = np.zeros(len(self.rows), np.int32)
        order, ledger = [], {}
        # Keep half the budget as an ordinary balanced sample; fill remaining
        # slots from trustworthy difficult positives and Q-strong negatives.
        for label in (True, False):
            ids = rng.choice(np.flatnonzero(self.labels == label), 6000, replace=False)
            order.extend(ids.tolist()); np.add.at(counts, ids, 1)
        def add(pool, requested, name):
            before = len(order)
            # Overall maximum four occurrences per epoch, including ordinary.
            available = np.repeat(pool, np.maximum(0, 4-counts[pool]))
            if len(available):
                chosen = rng.permutation(available)[:requested]
                order.extend(chosen.tolist()); np.add.at(counts, chosen, 1)
            ledger[name] = len(order)-before
        for name in ("low_support", "partial", "high_residual"):
            add(self.pools[name], 2000, name)
        positives_so_far = int(self.labels[order].sum())
        add(np.flatnonzero(self.labels & self.correct), 12000-positives_so_far, "positive_fill")
        if int(self.labels[order].sum()) != 12000:
            raise ValueError("not enough trustworthy positive occurrences within repeat cap")
        add(self.pools["hard_negative"], 6000, "hard_negative")
        if len(order) < 24000:
            add(np.flatnonzero(~self.labels), 24000-len(order), "negative_fill")
        if len(order) != 24000 or sum(self.labels[order]) != 12000 or counts.max() > 4:
            raise ValueError("joint D sampling budget changed")
        rng.shuffle(order)
        return np.asarray(order, np.int64), dict(epoch=epoch, counts=ledger,
            unique_pairs=int((counts>0).sum()), max_repeats=int(counts.max()),
            positive=12000, negative=12000, total=24000)
