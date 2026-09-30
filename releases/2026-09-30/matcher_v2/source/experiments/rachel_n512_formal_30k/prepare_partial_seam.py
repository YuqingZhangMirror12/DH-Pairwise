"""Materialize the first-epoch TRAIN generation manifest; optional GPU smoke.

All 24k rows retain source order/labels. Changed positive/negative proportions
are reported, not assumed from the 50% requested group coin. Sources are never
overwritten. Four first accepted positive pairs are saved for visual inspection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.rachel_n512_formal_30k.train_realism_data_ablation import (
    make_training_dataset, save_json, SEED, check_fixed_architecture)
from experiments.rachel_n512_formal_30k.train_partial_seam import (
    read_pair_metadata, build_recipe, PARTIAL_SEED)
from staging.pairwise_v0_2.pairwise_data.rachel_partial_seam_dataset import (
    PartialSeamDataset, PartialSeamConfig)


def identity_collate(rows):
    return rows


class DiagnosticRows:
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        i = self.indices[index]
        sample = self.dataset[i]
        report = self.dataset.diagnostics(i)
        report.update(source_index=i, output_area_a_px=int(sample.mask_a.sum()),
            output_area_b_px=int(sample.mask_b.sum()),
            output_token_matches=int((sample.target_a >= 0).sum()))
        return report


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in (.1, .5, .9)} if values else None


def prepare(args):
    torch.set_num_threads(1)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    base, provenance = make_training_dataset(args.dataset, args.train_manifest)
    metadata = read_pair_metadata(args.train_manifest)
    dataset = PartialSeamDataset(base, seed=PARTIAL_SEED, epoch=1, pair_metadata=metadata)
    if len(base) != 24000 or sum(label for _, label in metadata) != 12000:
        raise ValueError("requires the unchanged balanced matched24k TRAIN pool")
    limit = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
    loader = DataLoader(DiagnosticRows(dataset, tuple(range(limit))), batch_size=16,
        num_workers=args.workers, collate_fn=identity_collate, shuffle=False)
    counts, reasons, bins = Counter(), Counter(), Counter()
    lengths, ratios, selections = [], [], []
    started = time.perf_counter()
    with (root / "partial_manifest_epoch1.jsonl").open("x", encoding="utf-8") as stream:
        for reports in loader:
            for report in reports:
                label = "positive" if report["label"] else "negative"
                counts["rows"] += 1
                counts[label] += 1
                counts["requested_" + label] += int(report["requested"])
                counts["applied_" + label] += int(report["applied"])
                reasons[report["reason"]] += 1
                area_ratio = min(report["output_area_a_px"], report["output_area_b_px"]) / max(
                    report["output_area_a_px"], report["output_area_b_px"])
                bins[label + ("_area_lt025" if area_ratio < .25 else "_area_ge025")] += 1
                if report["applied"] and report["label"]:
                    lengths.append(report["source_seam_retention"])
                    ratios.append(area_ratio)
                    if len(selections) < 4:
                        selections.append(report["source_index"])
                stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
            if counts["rows"] % 256 == 0 or counts["rows"] == limit:
                status = dict(status="running", phase="prepare_train_partial", counts=dict(counts),
                              expected=limit, seconds=time.perf_counter() - started)
                save_json(root / "status.json", status)
                print(json.dumps(status), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    if args.limit is None and counts["applied_positive"] != counts["applied_negative"]:
        raise RuntimeError("full-manifest coupled application counts disagree")
    for number, index in enumerate(selections, 1):
        old, new = base[index], dataset[index]
        arrays = {"original_" + side: np.packbits(module_mask(getattr(old, "mask_" + side)), axis=1) for side in "ab"}
        arrays.update({"partial_" + side: np.packbits(module_mask(getattr(new, "mask_" + side)), axis=1) for side in "ab"})
        arrays.update(translation_rc=old.translation_a_to_b_rc,
                      points_a=new.points_rc_a, points_b=new.points_rc_b,
                      target_a=new.target_a, target_b=new.target_b)
        np.savez_compressed(root / ("example_%02d.npz" % number), **arrays)
    result = dict(schema_version="rachel-partial-train-preparation/1", status="complete",
        full_train=args.limit is None, probe_only=args.limit is not None,
        epoch=1, seed=PARTIAL_SEED, counts=dict(counts), reasons=dict(reasons), area_bins=dict(bins),
        accepted_positive_seam_retention_quantiles=quantiles(lengths),
        accepted_positive_area_ratio_quantiles=quantiles(ratios),
        config=asdict(PartialSeamConfig()), provenance=provenance,
        generation_manifest="partial_manifest_epoch1.jsonl", online_regenerated_each_epoch=True,
        example_indices=selections, example_selection="first four applied positives in source order; GT placements, no model prediction",
        original_files_modified=False, seconds=time.perf_counter() - started)
    save_json(root / "summary.json", result)
    save_json(root / "status.json", result)
    print(json.dumps(result), flush=True)
    return result


def module_mask(value):
    array = np.asarray(value, bool)
    return array[0] if array.ndim == 3 else array


def smoke(args):
    """256 total discarded optimizer exposures; no smoke weights are persisted."""
    from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint
    from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
    from staging.pairwise_v0_2.training.rachel_weathering_training import make_weathering_loader, train_weathering_epoch
    from staging.pairwise_v0_2.training import rachel_n512_runner as runner
    from experiments.rachel_n512_formal_30k.train_edge_weathering import _sha256, SOURCE_SHA256
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("smoke requires the remote CUDA server")
    if _sha256(args.checkpoint) != SOURCE_SHA256:
        raise ValueError("smoke source must be common 350bf9 checkpoint")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    base, _ = make_training_dataset(args.dataset, args.train_manifest)
    metadata = read_pair_metadata(args.train_manifest)
    # Same balanced bounded source subset for both arms, not selected by results.
    pos = [i for i, (_, label) in enumerate(metadata) if label][:64]
    neg = [i for i, (_, label) in enumerate(metadata) if not label][:64]
    indices = tuple(i for pair in zip(pos, neg) for i in pair)
    results = {}
    torch.set_num_threads(1)
    for arm in ("e4", "e5"):
        runner._set_determinism(SEED)
        source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = load_rachel_checkpoint(source)
        check_fixed_architecture(model, source)
        device = torch.device(args.device)
        model = model.to(device).train().requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
        recipe = build_recipe(arm, args.train_manifest)
        dataset = recipe["build_dataset"](base, seed=SEED, cache_dir=args.cache_dir)
        dataset.set_epoch(1)
        loader = make_weathering_loader(dataset, indices, batch_size=4,
            num_workers=args.workers, seed=SEED, contour_cap=512)
        from types import SimpleNamespace
        cfg = SimpleNamespace(batch_size=4, effective_batch_size=16, log_every=16, output=None)
        torch.cuda.reset_peak_memory_stats(device)
        report = train_weathering_epoch(model, loader, optimizer,
            RachelN512LossConfig(**source["loss_config"]), device, cfg, 1, recipe["variant"])
        if report["samples"] != 128 or report["optimizer_updates"] != 8 or not np.isfinite(report["mean_loss"]):
            raise RuntimeError("partial GPU smoke budget/nonfinite failure")
        if report["weathering"]["partial_seam"]["counts"].get("applied_pair_exposures", 0) == 0:
            raise RuntimeError("partial GPU smoke saw no actually truncated TRAIN pair")
        results[arm] = report
        del model, optimizer, source, loader
        torch.cuda.empty_cache()
    result = dict(status="complete", basic_contract_passed=True, weights_saved=False,
                  formal_weights_reused=False, results=results)
    save_json(root / "preflight.json", result)
    print(json.dumps(result), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "train-manifest", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.workers < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("invalid worker/limit count")
    smoke(args) if args.smoke else prepare(args)
