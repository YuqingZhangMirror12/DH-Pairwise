"""Oblique, real-TRAIN-outline partial source-seam augmentation."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict

import numpy as np

from .rachel_partial_seam_dataset import (PartialSeamDataset, PartialSeamConfig,
    source_seam_context, crop_training_pair, _ignore_ambiguous_nonmatches, _mask_2d)
from .rachel_preprocess import RachelPreprocessError
from .rachel_curve_cut import OutlineBank, curve_mask


class CurvedPartialDataset(PartialSeamDataset):
    def __init__(self, base, *, pair_metadata, bank, seed=260910, epoch=0,
                 config=PartialSeamConfig(max_attempts=12)):
        self.bank = bank if isinstance(bank, OutlineBank) else OutlineBank(bank)
        super().__init__(base, pair_metadata=pair_metadata, seed=seed, epoch=epoch, config=config)

    def proposal(self, ids, attempt):
        rng = self._rng(ids, "curve-attempt-v2", attempt)
        return dict(donor_index=int(rng.integers(len(self.bank.profiles))),
            angle_deg=float(rng.uniform(15.,75.) + 90*int(rng.integers(2))),
            fraction=float(rng.uniform(.15,.85)), keep_low=bool(rng.integers(2)),
            flip=bool(rng.integers(2)), reverse=bool(rng.integers(2)))

    def apply_proposal(self, sample, proposal):
        masks = [_mask_2d(getattr(sample,"mask_"+s)) for s in "ab"]
        side = "a" if masks[0].sum() <= masks[1].sum() else "b"
        mask = masks[0 if side == "a" else 1]
        params = {k:v for k,v in proposal.items() if k != "donor_index"}
        crop = curve_mask(mask, self.bank.profiles[proposal["donor_index"]], **params)
        result = crop_training_pair(sample, side=side, retained_mask=crop, config=self.config.crop_config)
        if result.accepted:
            from dataclasses import replace
            result = replace(result, sample=_ignore_ambiguous_nonmatches(sample, result.sample, side, None))
        return result, side, crop

    def _materialize(self, group_index):
        indices = self._groups[group_index]
        ids = tuple(self.pair_metadata[i][0] for i in indices)
        originals = tuple(self.base[i] for i in indices)
        if any(s.pair_id != self.pair_metadata[i][0] or int(s.label) != self.pair_metadata[i][1]
               for s,i in zip(originals,indices)):
            raise ValueError("reference TRAIN order/labels changed")
        requested = bool(self._rng(ids,"curve-coin-v2").random() < self.config.probability)
        group = dict(schema_version="rachel-curved-partial-source-seam/2", seed=self.seed, epoch=self.epoch,
            group_pair_ids=list(ids), requested=requested, applied=False, reason="shared_coin_skipped",
            attempts=0, attempt_reasons={}, members=[{},{}])
        if not requested:
            return originals, group
        positive = originals[0]
        masks = [_mask_2d(getattr(positive,"mask_"+s)) for s in "ab"]
        side = "a" if masks[0].sum() <= masks[1].sum() else "b"
        try:
            context = source_seam_context(positive, side, self.config.crop_config)
        except (ValueError,RachelPreprocessError):
            context = None
        if context is None:
            group["reason"] = "original_source_seam_unavailable"
            return originals,group
        reasons = Counter()
        for attempt in range(1,self.config.max_attempts+1):
            proposal = self.proposal(ids,attempt)
            group["attempts"] = attempt
            params = {k:v for k,v in proposal.items() if k != "donor_index"}
            crop = curve_mask(masks[0 if side == "a" else 1], self.bank.profiles[proposal["donor_index"]], **params)
            points = np.rint(context["points"]).astype(int)
            retention = float(context["weights"][crop[points[:,0],points[:,1]]].sum()/context["length"])
            if not self.config.retention_min <= retention <= self.config.retention_max:
                reasons["source_seam_not_partial"] += 1
                continue
            results = [self.apply_proposal(s,proposal) for s in originals]
            if not all(r[0].accepted for r in results):
                for label,(result,_,_) in zip(("positive","negative"),results):
                    if not result.accepted:
                        reasons[label+":"+result.reason] += 1
                continue
            donor = self.bank.metadata["arcs"][proposal["donor_index"]]
            reports = []
            for old,(result,endpoint,_) in zip(originals,results):
                areas = [int(getattr(old,"mask_"+s).sum()) for s in "ab"]
                reports.append(dict(side=endpoint, proposal=proposal, donor_lineage=donor["lineage"],
                    donor_family=donor["family"], donor_split="train", cut_kind="oblique_real_outline_profile",
                    original_area_ratio=min(areas)/max(areas), post_area_ratio=result.diagnostics["post_area_ratio"],
                    source_seam_retention=retention if old.label else None,
                    retained_supervised_seam_length_px=result.diagnostics["retained_contiguous_seam_length_px"] if old.label else None,
                    supervised_token_count=result.diagnostics["token_match_count"],
                    geometry=result.diagnostics))
            group.update(applied=True, reason="coupled_accepted", proposal=proposal,
                         members=reports, attempt_reasons=dict(reasons))
            return tuple(r[0].sample for r in results),group
        group.update(reason="no_jointly_accepted_attempt",attempt_reasons=dict(reasons))
        return originals,group
