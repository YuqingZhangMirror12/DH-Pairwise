"""Replacement pending data arms: E4v2 curve+E1 weather; E5v2 adds hard negatives.

The reference 24k geometry generator is shared before negative substitution.
All positive examples, including their deterministic curved crop and weather,
are identical between these arms. Hard-negative replacement changes only
negative slots, not class balance, exposure budget, architecture or thresholds.
"""
from __future__ import annotations
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k import train_edge_weathering as trainer
from experiments.rachel_n512_formal_30k.train_partial_seam import read_pair_metadata
from staging.pairwise_v0_2.pairwise_data.rachel_curved_partial_dataset import CurvedPartialDataset
from staging.pairwise_v0_2.pairwise_data.rachel_partial_seam_dataset import PartialSeamConfig

ARMS = {"e4v2": "curved_partial_weathering_e4v2", "e5v2": "curved_partial_weathering_straightneg_e5v2"}
CURVE_CONFIG = PartialSeamConfig(max_attempts=12)


class NegativeOverlayDataset:
    split = "train"
    def __init__(self, reference, overlay_path, *, expected_manifest, metadata):
        from staging.pairwise_v0_2.pairwise_data.rachel_composite_training import _RowsDataset
        from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelDatasetConfig
        self.reference, self.root = reference, getattr(reference,"root",None)
        content = json.loads(Path(overlay_path).read_text())
        if content.get("schema_version") != "rachel-straight-negative-overlay/1" or content.get("split") != "train":
            raise ValueError("only TRAIN negative overlays are accepted")
        if Path(content["base_manifest"]).resolve() != Path(expected_manifest).resolve():
            raise ValueError("negative overlay belongs to another reference manifest")
        self.mapping, grouped, self.entries = {}, {}, {}
        for item in content["entries"]:
            index, source, row = int(item["source_index"]), item["source_root"], item["row"]
            if not 0 <= index < len(reference) or metadata[index][1] or row["label"] or row["split"] != "train":
                raise ValueError("negative substitution cannot alter a positive or held-out row")
            if index in self.mapping:
                raise ValueError("duplicate replacement slot")
            rows = grouped.setdefault(source,[])
            self.mapping[index] = (source,len(rows))
            self.entries[index] = item
            rows.append(row)
        self.loaders = {s:_RowsDataset(s,rows,RachelDatasetConfig()) for s,rows in grouped.items()}
        if not self.mapping:
            raise ValueError("no actual hard-negative replacements were mined")

    def __len__(self):
        return len(self.reference)

    def set_epoch(self, epoch):
        self.reference.set_epoch(epoch)

    def materialize(self, index):
        original = self.reference[index]
        report = self.reference.diagnostics(index)
        report["reference_pair_id"] = report["pair_id"]
        if index not in self.mapping:
            report["hard_negative"] = dict(replaced=False)
            return original,report
        source,inner = self.mapping[index]
        negative = self.loaders[source][inner]
        if negative.label:
            raise ValueError("hard-negative loader returned a positive")
        detail = dict(replaced=True, reference_pair_id=original.pair_id,
            replacement_pair_id=negative.pair_id, curve_attempted=bool(report["applied"]),
            curve_applied=False, curve_fallback_reason=None)
        if report["applied"]:
            result,side,_ = self.reference.apply_proposal(negative,report["proposal"])
            detail["curve_applied"] = result.accepted
            if result.accepted:
                negative = result.sample
                report["side"] = side
                report["post_area_ratio"] = result.diagnostics["post_area_ratio"]
            else:
                detail["curve_fallback_reason"] = result.reason
                report.update(applied=False,reason="replacement_negative:"+result.reason)
        report.update(pair_id=negative.pair_id, hard_negative=detail)
        return negative,report

    def __getitem__(self,index):
        return self.materialize(index)[0]

    def diagnostics(self,index):
        return self.materialize(index)[1]


class CurvedTrainingDataset:
    split = "train"
    def __init__(self, base, *, seed, cache_dir, metadata, arm, outline_bank,
                 negative_overlay, train_manifest):
        from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import RachelWeatheredDataset
        self.root = getattr(base,"root",None)
        self.reference = CurvedPartialDataset(base,pair_metadata=metadata,bank=outline_bank,
            seed=260910,epoch=0,config=CURVE_CONFIG)
        self.geometry = self.reference if arm == "e4v2" else NegativeOverlayDataset(
            self.reference,negative_overlay,expected_manifest=train_manifest,metadata=metadata)
        self.weather = RachelWeatheredDataset(self.geometry,seed=seed,epoch=0,
            cache_dir=cache_dir,clean_probability=.70,mild_probability=.25)

    def __len__(self):
        return len(self.reference)

    def set_epoch(self,epoch):
        self.geometry.set_epoch(epoch)
        self.weather.set_epoch(epoch)

    def __getitem__(self,index):
        sample,report = self.weather[index]
        report = dict(report)
        detail = self.geometry.diagnostics(index)
        report["partial_seam"] = detail
        report["hard_negative"] = detail.get("hard_negative",dict(replaced=False))
        return sample,report


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_recipe(arm,train_manifest,outline_bank,negative_overlay):
    if arm not in ARMS:
        raise ValueError("unknown curved data arm")
    metadata = read_pair_metadata(train_manifest)
    policy = dict(curve_config=asdict(CURVE_CONFIG),curve_seed=260910,
        outline_bank=str(Path(outline_bank).resolve()),
        outline_profile_sha256=file_digest(Path(outline_bank)/"profiles.npz"),
        outline_provenance_sha256=file_digest(Path(outline_bank)/"bank.json"),
        negative_overlay=str(Path(negative_overlay).resolve()),
        negative_overlay_sha256=file_digest(negative_overlay),hard_negative_enabled=arm=="e5v2",
        correspondence_policy="original source-supported reciprocal seam only; oblique real-TRAIN-outline deletion, new cut ignored within8px; then E1 source-arc weathering",
        requested_endpoint_tier_probabilities=dict(clean=.70,mild=.25,moderate=.05),
        original_gt_translation_preserved=True, target_rules_changed=True,
        architecture_changed=False, positive_geometry_shared_between_arms=True,
        reference_epoch_curve_acceptance="coupled original positive and negative before overlay; replacement negative may independently fall back to uncut replacement, with actual counts recorded",
        negative_policy="only TRAIN native CSV nonadjacent hard-negative slots replaced; positive labels/inputs unchanged; no rectangle-implies-negative rule",
        comparison="E4v2 vs E1 adds real-outline partial data; E5v2 vs E4v2 adds negative overlay; replaces untrained old axis E4/E5, not the same recipes",
        real_usage="qualitative research design only; no REAL donor mask, negative mining, model fitting or threshold selection")
    return dict(variant=ARMS[arm],policy=policy,build_dataset=partial(CurvedTrainingDataset,
        metadata=metadata,arm=arm,outline_bank=outline_bank,negative_overlay=negative_overlay,
        train_manifest=train_manifest))


def parser():
    p=trainer.parser()
    p.add_argument("--arm",choices=sorted(ARMS),required=True)
    p.add_argument("--outline-bank",required=True)
    p.add_argument("--negative-overlay",required=True)
    return p


if __name__ == "__main__":
    args=parser().parse_args()
    trainer.run(args,recipe=build_recipe(args.arm,args.train_manifest,args.outline_bank,args.negative_overlay))
