"""CPU-only, fixed-group TRAIN coverage probe for low/high guided curves.

The high-probability geometry is materialized once. Low coverage is derived
from the exact same stable request coin (<0.2); it does not rerun proposal
search or replace native negatives. No model, checkpoint or GPU is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.rachel_n512_formal_30k.train_realism_data_ablation import make_training_dataset
from experiments.rachel_n512_formal_30k.train_partial_seam import read_pair_metadata
from staging.pairwise_v0_2.pairwise_data.rachel_guided_partial_dataset import GuidedPartialDataset
from staging.pairwise_v0_2.pairwise_data.rachel_partial_seam_dataset import PartialSeamConfig


def _identity(rows):
    return rows


class _GroupProbe:
    def __init__(self, dataset, group_indices):
        self.dataset, self.group_indices = dataset, group_indices

    def __len__(self):
        return len(self.group_indices)

    def __getitem__(self, index):
        group_index = self.group_indices[index]
        indices = self.dataset._groups[group_index]
        samples, group = self.dataset._get_group(group_index)
        members = []
        for source_index, sample in zip(indices, samples):
            report = self.dataset.diagnostics(source_index)
            report.update(source_index=source_index, group_index=group_index,
                output_area_a_px=int(sample.mask_a.sum()), output_area_b_px=int(sample.mask_b.sum()),
                output_token_matches=int((sample.target_a >= 0).sum()))
            members.append(report)
        return dict(group=group, members=members)


def _quantiles(values):
    if not values:
        return None
    return {str(q): float(np.quantile(values, q)) for q in (0., .1, .5, .9, 1.)}


def _save(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(args):
    if args.limit <= 0 or args.limit % 2:
        raise ValueError("limit must be a positive even number of paired rows")
    torch.set_num_threads(1)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    metadata = read_pair_metadata(args.train_manifest)
    base, provenance = make_training_dataset(args.dataset, args.train_manifest)
    if len(base) != 24000 or sum(label for _, label in metadata) != 12000:
        raise ValueError("requires the unchanged balanced matched24k TRAIN pool")
    config = PartialSeamConfig(probability=1., max_attempts=args.max_attempts)
    dataset = GuidedPartialDataset(base, pair_metadata=metadata, bank=args.outline_bank,
        seed=args.seed, epoch=args.epoch, config=config)
    group_count = min(args.limit // 2, len(dataset._groups))
    # Existing SHA256 group order is determined before proposals/outcomes.
    group_indices = tuple(range(group_count))
    loader = DataLoader(_GroupProbe(dataset, group_indices), batch_size=8,
        num_workers=args.workers, shuffle=False, collate_fn=_identity)
    counts = {arm: Counter() for arm in ("partial_low", "partial_high")}
    reasons, attempt_reasons = Counter(), Counter()
    retentions, seam_lengths, material_positive, material_negative, attempts = [], [], [], [], []
    requested_groups = applied_groups = low_requested_groups = low_applied_groups = 0
    started = time.perf_counter()
    manifest = root / ("guided_manifest_weather_epoch%d.jsonl" % args.epoch)
    with manifest.open("x", encoding="utf-8") as stream:
        for batch in loader:
            for item in batch:
                group, members = item["group"], item["members"]
                assert len(members) == 2 and members[0]["label"] and not members[1]["label"]
                assert members[0]["applied"] == members[1]["applied"] == group["applied"]
                requested_groups += int(group["requested"])
                applied_groups += int(group["applied"])
                low_requested = group["request_coin"] < .2
                low_applied = low_requested and group["applied"]
                low_requested_groups += int(low_requested)
                low_applied_groups += int(low_applied)
                reasons[group["reason"]] += 1
                attempt_reasons.update(group["attempt_reasons"])
                attempts.append(group["attempts"])
                if group["applied"]:
                    retentions.append(members[0]["source_seam_retention"])
                    seam_lengths.append(members[0]["retained_supervised_seam_length_px"])
                    material_positive.append(members[0]["material_retention"])
                    material_negative.append(members[1]["material_retention"])
                    assert .25 <= retentions[-1] <= .75 and seam_lengths[-1] >= 32.
                    assert members[0]["output_token_matches"] >= 4 and members[1]["output_token_matches"] == 0
                for report in members:
                    label = "positive" if report["label"] else "negative"
                    for arm, requested, applied in (("partial_high", group["requested"], group["applied"]),
                                                     ("partial_low", low_requested, low_applied)):
                        counts[arm]["rows"] += 1
                        counts[arm][label] += 1
                        counts[arm]["requested_" + label] += int(requested)
                        counts[arm]["applied_" + label] += int(applied)
                    report.update(low_requested=bool(low_requested), low_applied=bool(low_applied),
                        low_output_policy="same high arrays if low_applied, otherwise exact original sample")
                    stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
            print(json.dumps(dict(status="running", groups=counts["partial_high"]["rows"] // 2,
                counts={k: dict(v) for k, v in counts.items()}, seconds=time.perf_counter() - started)), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    for count in counts.values():
        assert count["positive"] == count["negative"] == group_count
        assert count["applied_positive"] == count["applied_negative"]
    rates = {arm: {label: counts[arm]["applied_" + label] / counts[arm][label]
                   for label in ("positive", "negative")} for arm in counts}
    identity = dict(dataset=str(Path(args.dataset).resolve()),
        train_manifest=str(Path(args.train_manifest).resolve()),
        outline_bank=str(Path(args.outline_bank).resolve()))
    for key, path in (("train_manifest_sha256", Path(args.train_manifest)),
                      ("outline_profiles_sha256", Path(args.outline_bank) / "profiles.npz"),
                      ("outline_metadata_sha256", Path(args.outline_bank) / "bank.json")):
        identity[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(schema_version="rachel-guided-partial-preparation/1", status="complete",
        cpu_geometry_only=True, epoch=args.epoch, canonical_weather_epoch=args.epoch, seed=args.seed,
        full_train=group_count == 12000, probe_only=group_count != 12000,
        source_count=len(base), selected_rows=group_count * 2, selected_groups=group_count,
        selection="first fixed SHA256 positive/native-negative groups; chosen before proposals/outcomes",
        counts={k: dict(v) for k, v in counts.items()}, actual_weather_stage_application_rates=rates,
        projected_six_slot_application_rates={arm: {k: .5 * v for k, v in rate.items()} for arm, rate in rates.items()},
        high_60_percent_target_met=all(v >= .6 for v in rates["partial_high"].values()),
        requested_groups=requested_groups, applied_groups=applied_groups,
        low_requested_groups=low_requested_groups, low_applied_groups=low_applied_groups,
        low_is_high_subset=True, low_accepted_arrays_identical_by_construction=True,
        low_derivation="high geometry searched once; same stable group request_coin <0.2 enables identical output; proposal RNG does not depend on probability",
        group_reasons=dict(reasons), attempt_reasons=dict(attempt_reasons),
        attempts_quantiles=_quantiles(attempts), physical_seam_retention_quantiles=_quantiles(retentions),
        retained_supervised_seam_length_px_quantiles=_quantiles(seam_lengths),
        material_retention_quantiles=dict(positive=_quantiles(material_positive), negative=_quantiles(material_negative)),
        config=asdict(config), input_identity=identity, provenance=provenance,
        guided_topology_connectivity=8,
        generation_manifest=manifest.name, hard_negative_overlay_used=False,
        original_natural_straight_positive_rows_preserved=True, original_files_modified=False,
        coverage_note="measured in this fixed CPU probe before later E1 weathering; whole-pool/other-epoch coverage is not assumed",
        seconds=time.perf_counter() - started)
    _save(root / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--outline-bank", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--epoch", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--seed", type=int, default=260910)
    p.add_argument("--max-attempts", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    return p


if __name__ == "__main__":
    prepare(parser().parse_args())
