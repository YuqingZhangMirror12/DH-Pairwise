"""Matched six-exposure CLEAN/E1 schedules; only temporal order differs.

Every pair receives three exact clean exposures and E1 weather epochs 1, 2, 3
once each. Joint order is a stable pair-ID hash permutation, never a label rule.
The E1 variants keep their own fixed weather epoch, independent of train epoch.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np

from .rachel_weathered_dataset import RachelWeatheredDataset

SCHEMA = "rachel-staged-damage-schedule/1"
SLOTS = (0, 0, 0, 1, 2, 3)


def exposure_schedule(pair_id, *, arm, seed=260910):
    if arm not in ("staged", "joint"):
        raise ValueError("schedule arm must be staged or joint")
    if type(seed) is not int or not isinstance(pair_id, str) or not pair_id:
        raise ValueError("a nonempty pair ID and integer seed are required")
    if arm == "staged":
        return SLOTS
    def key(slot):
        value = json.dumps([SCHEMA, seed, pair_id, slot], separators=(",", ":"))
        return hashlib.sha256(value.encode()).digest(), slot
    return tuple(SLOTS[i] for i in sorted(range(6), key=key))


def schedule_audit(pair_ids, *, arm, seed=260910):
    """Metadata-only count/hash audit; never reads masks or labels."""
    schedules = [(pair_id, exposure_schedule(pair_id, arm=arm, seed=seed)) for pair_id in pair_ids]
    counts = [Counter(schedule[e] for _, schedule in schedules) for e in range(6)]
    encode = lambda value: hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
    return dict(schema_version=SCHEMA, arm=arm, seed=seed, pair_count=len(schedules),
        pair_id_order_sha256=encode(list(pair_ids)), schedule_sha256=encode(schedules),
        per_epoch=[dict(epoch=e + 1, slots={str(slot): counts[e][slot] for slot in range(4)}) for e in range(6)],
        total_slots={str(slot): sum(c[slot] for c in counts) for slot in range(4)},
        slot_meanings={"0": "exact clean", "1": "E1 weather epoch 1", "2": "E1 weather epoch 2", "3": "E1 weather epoch 3"})


def clean_report(sample, *, epoch):
    """E1-collate-compatible report with zero attempted/applied augmentation."""
    matches = int(np.count_nonzero(sample.target_a >= 0))
    result = dict(schema_version=SCHEMA, pair_id=sample.pair_id, epoch=epoch,
        tier=dict(a="clean", b="clean"), changed_a=False, changed_b=False, changed_pair=False,
        pose_supervision_enabled=bool(sample.label), fallback_reason=None,
        inherited_match_count=matches, effective_supervised_match_count=matches,
        ignored_token_count=int(np.count_nonzero((sample.target_a == -2) & sample.contour_valid_a)
                              + np.count_nonzero((sample.target_b == -2) & sample.contour_valid_b)),
        inheritance_rule="unchanged original clean reciprocal targets",
        original_gt_translation_preserved=True, cross_fragment_geometry_used_for_targets=False)
    for side in "ab":
        area = int(getattr(sample, "mask_" + side).sum())
        result["side_" + side] = dict(tier="clean", applied=False, attempted_applied=False,
            effective_applied=False, skipped=False, skip_reason=None, original_area_px=area,
            retained_area_px=area, removed_area_px=0, effective_removed_area_px=0,
            removed_fraction=0., max_depth_px=0.)
    return result


class StagedDamageDataset:
    """Return (sample, report), with fixed three-variant E1 weathering objects."""
    def __init__(self, base, *, arm, seed=260910, epoch=1, cache_dir=None):
        # The composite loader has explicit validated TRAIN child sources.
        from .rachel_partial_seam_dataset import _train_base
        self.base = _train_base(base)
        exposure_schedule("protocol-check", arm=arm, seed=seed)
        self.arm, self.seed = arm, seed
        self.root, self.split, self.contour_cap = getattr(base, "root", None), "train", 512
        self.weather = {e: RachelWeatheredDataset(self.base, seed=seed, epoch=e, cache_dir=cache_dir)
                        for e in (1, 2, 3)}
        self.set_epoch(epoch)

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        if type(epoch) is not int or not 1 <= epoch <= 6:
            raise ValueError("training epoch must be in 1..6")
        self.epoch = epoch
        # Do NOT forward global epoch to E1: each variant's epoch is frozen.

    def __getitem__(self, index):
        original = self.base[index]
        schedule = exposure_schedule(original.pair_id, arm=self.arm, seed=self.seed)
        slot = schedule[self.epoch - 1]
        if slot:
            sample, report = self.weather[slot][index]
            report = dict(report)
        else:
            sample, report = original, clean_report(original, epoch=self.epoch)
        report["damage_schedule"] = dict(schema_version=SCHEMA, arm=self.arm, seed=self.seed,
            training_epoch=self.epoch, weather_epoch=slot if slot else None,
            clean_exposure=not bool(slot), six_slots=list(schedule), label_used_for_schedule=False)
        return sample, report


class GuidedStagedDamageDataset(StagedDamageDataset):
    """Only canonical weather slots receive paired TRAIN-profile truncations."""
    def __init__(self, base, *, arm, pair_metadata, bank, probability,
                 seed=260910, epoch=1, cache_dir=None):
        from .rachel_guided_partial_dataset import GuidedPartialDataset
        from .rachel_partial_seam_dataset import PartialSeamConfig
        if arm != "joint" or probability not in (.2, 1.):
            raise ValueError("guided controls require joint schedule and probability .2 or 1")
        super().__init__(base, arm=arm, seed=seed, epoch=epoch, cache_dir=cache_dir)
        self.partial = {e: GuidedPartialDataset(self.base, pair_metadata=pair_metadata,
            bank=bank, seed=seed, epoch=e,
            config=PartialSeamConfig(probability=probability, max_attempts=12))
            for e in (1, 2, 3)}
        self.weather = {e: RachelWeatheredDataset(self.partial[e], seed=seed, epoch=e,
            cache_dir=cache_dir) for e in (1, 2, 3)}

    def __getitem__(self, index):
        sample, report = super().__getitem__(index)
        slot = report["damage_schedule"]["weather_epoch"]
        if slot is not None:
            report["partial_seam"] = self.partial[slot].diagnostics(index)
        else:
            report["partial_seam"] = dict(applied=False, requested=False,
                reason="exact_clean_schedule_slot", label=int(sample.label))
        return sample, report


__all__ = ["StagedDamageDataset", "GuidedStagedDamageDataset", "exposure_schedule", "schedule_audit", "clean_report"]
