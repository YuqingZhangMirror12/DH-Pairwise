"""Materialize a common E1 table and nested label/size/damage-stratified subsets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from .train_realism_data_ablation import make_training_dataset, save_json
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import RachelWeatheredDataset
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import SCHEMA, save_sample, load_sample


def rank(pair_id, seed):
    return hashlib.sha256((str(seed) + ":" + pair_id).encode()).digest()


def nested_subset(entries, count, seed=260911):
    """Hamilton quotas within each label; selecting 12k then6k ensures nesting."""
    if count <= 0 or count % 2 or count > len(entries):
        raise ValueError("subset must have an even positive size <= parent")
    selected = []
    for label in (False, True):
        by_stratum = defaultdict(list)
        group = [e for e in entries if e["label"] == label]
        if len(group) != len(entries) // 2:
            raise ValueError("parent must be label-balanced")
        for entry in group:
            key = (entry["source_stratum"], entry["area_ratio_band"], entry["changed_pair"])
            by_stratum[key].append(entry)
        budget = count // 2
        quota = {key: budget * len(values) // len(group) for key, values in by_stratum.items()}
        remainders = sorted(by_stratum, key=lambda key: (-(budget * len(by_stratum[key]) % len(group)), str(key)))
        for key in remainders[:budget - sum(quota.values())]:
            quota[key] += 1
        for key in sorted(by_stratum, key=str):
            selected.extend(sorted(by_stratum[key], key=lambda x: rank(x["pair_id"], seed))[:quota[key]])
    return sorted(selected, key=lambda x: rank(x["pair_id"], seed + 1))


class Materializer:
    def __init__(self, base, source_entries, root, seed, cache_dir):
        self.weather = RachelWeatheredDataset(base, seed=seed, epoch=1, cache_dir=cache_dir)
        self.entries, self.root = source_entries, Path(root)

    def __len__(self):
        return len(self.weather)

    def __getitem__(self, index):
        source = self.entries[index]
        pair_id = source["row"]["pair_id"]
        token = hashlib.sha256(pair_id.encode()).hexdigest()
        relative = "samples/" + token[:2] + "/" + token + ".npz"
        path = self.root / relative
        if path.exists():
            sample, report = load_sample(path)
        else:
            sample, report = self.weather[index]
            save_sample(path, sample, report)
        if sample.pair_id != pair_id:
            raise ValueError("source order differs from materialized record")
        row = source["row"]
        areas = [float(row["fragment_" + side]["foreground_area"]) for side in "ab"]
        ratio = min(areas) / max(areas)
        return dict(pair_id=pair_id, label=bool(sample.label), artifact_path=relative,
                    source_row=row, source_root=source["source_root"], source_stratum=source.get("stratum", "unknown"),
                    area_ratio_band="tiny_lt025" if ratio < .25 else "other",
                    changed_pair=bool(report["changed_pair"]),
                    surviving_correspondences=int((sample.target_a >= 0).sum()),
                    inherited_match_count=int(report.get("inherited_match_count", 0)))


def identity_collate(rows):
    return rows


def main(args):
    torch.set_num_threads(1)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.train_manifest).resolve()
    identity = dict(source_manifest=str(manifest), source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        weather_seed=args.seed, weather_epoch=1, requested_fragment_probabilities=dict(clean=.70, mild=.25, moderate=.05),
        mild_depth_px=2., moderate_depth_px=4., correspondence="E1 inherited source arc, ignored=-2; no geometric relabeling",
        partial_seam_added=False, fixed_physical_sample_across_models_and_epochs=True,
        real_or_test_used=False, subset_seed=260911)
    if (output / "protocol.json").exists():
        if json.loads((output / "protocol.json").read_text()) != identity:
            raise ValueError("materialization identity differs")
    else:
        save_json(output / "protocol.json", identity)
    base, _ = make_training_dataset(args.dataset, manifest)
    source = json.loads(manifest.read_text())["entries"]
    dataset = Materializer(base, source, output, args.seed, output / "weather_cache")
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=args.workers,
        shuffle=False, collate_fn=identity_collate, persistent_workers=False)
    entries, started = [], time.monotonic()
    for batch in loader:
        entries.extend(batch)
        record = dict(status="running", stage="materialize_e1", pid=os.getpid(), processed=len(entries),
                      total=len(dataset), elapsed_s=time.monotonic() - started)
        save_json(output / "status.json", record)
        if len(entries) % 512 == 0:
            print(json.dumps(record), flush=True)
    if len(entries) != 24000:
        raise ValueError("formal common table must contain24k")
    parent = entries
    manifests = {}
    for count in (24000, 12000, 6000):
        subset = parent if count == 24000 else nested_subset(parent, count)
        stats = dict(sample_count=len(subset), positive_count=sum(e["label"] for e in subset),
            changed_pair_count=sum(e["changed_pair"] for e in subset),
            positives_with_zero_surviving_correspondences=sum(e["label"] and not e["surviving_correspondences"] for e in subset),
            strata=dict(Counter(str((e["label"], e["source_stratum"], e["area_ratio_band"], e["changed_pair"])) for e in subset)))
        target = output / ("train_e1_%dk.json" % (count // 1000))
        save_json(target, dict(schema_version=SCHEMA, split="train", artifact_root=str(output),
                              entries=subset, stats=stats, protocol=identity))
        manifests[str(count)] = str(target)
        parent = subset
    save_json(output / "status.json", dict(status="complete", pid=os.getpid(), stage="common_training_ready",
        sample_count=len(entries), changed_pair_count=sum(e["changed_pair"] for e in entries),
        manifests=manifests, elapsed_s=time.monotonic() - started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=260909)
    parser.add_argument("--workers", type=int, default=2)
    main(parser.parse_args())
