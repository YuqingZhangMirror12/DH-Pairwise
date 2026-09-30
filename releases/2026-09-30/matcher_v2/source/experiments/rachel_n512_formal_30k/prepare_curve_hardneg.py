"""Audit matched24k E4v2/E5v2 TRAIN geometry; optionally run 128-row GPU smokes.

CPU preparation shares one reference curve generator before negative overlay,
counts actual per-arm outcomes, and writes source-order epoch-1 manifests. A
--limit run is only a probe. GPU smoke compares every weathered positive input
and target across arms and discards all 256 total optimizer exposures/weights.
No validation/test/REAL examples are loaded or used for selection.
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

from experiments.rachel_n512_formal_30k.prepare_partial_seam import (
    identity_collate, module_mask, quantiles)
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import (
    make_training_dataset, save_json, SEED, check_fixed_architecture)
from experiments.rachel_n512_formal_30k.train_partial_seam import read_pair_metadata
from experiments.rachel_n512_formal_30k.train_curve_hardneg import (
    ARMS, CURVE_CONFIG, NegativeOverlayDataset, build_recipe, file_digest)
from staging.pairwise_v0_2.pairwise_data.rachel_curved_partial_dataset import CurvedPartialDataset

SCHEMA = "rachel-curve-hardneg-train-preparation/1"
SMOKE_SCHEMA = "rachel-curve-hardneg-gpu-preflight/1"
CURVE_SEED = 260910


def input_identity(args, *, checkpoint=False):
    result = {name: str(Path(getattr(args, name)).resolve()) for name in
              ("dataset", "train_manifest", "outline_bank", "negative_overlay")}
    result.update(train_manifest_sha256=file_digest(args.train_manifest),
        negative_overlay_sha256=file_digest(args.negative_overlay),
        outline_profiles_sha256=file_digest(Path(args.outline_bank) / "profiles.npz"),
        outline_metadata_sha256=file_digest(Path(args.outline_bank) / "bank.json"))
    if checkpoint:
        result.update(checkpoint=str(Path(args.checkpoint).resolve()),
                      checkpoint_sha256=file_digest(args.checkpoint))
    return result


def _validate_pool(base, metadata):
    labels = Counter(label for _, label in metadata)
    if len(base) != 24000 or len(metadata) != 24000 or labels != {0: 12000, 1: 12000}:
        raise ValueError("requires the unchanged balanced matched24k TRAIN pool")


def _assert_same_sample(first, second):
    """All model inputs, source identity and GT fields, not merely pair labels."""
    if type(first) is not type(second):
        raise AssertionError("positive sample types differ between arms")
    for name in first.__dataclass_fields__:
        left, right = getattr(first, name), getattr(second, name)
        if isinstance(left, np.ndarray):
            if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                raise AssertionError("positive arrays differ between arms: " + name)
        elif left != right:
            raise AssertionError("positive source/GT fields differ between arms: " + name)


def _sample_fingerprint(sample):
    digest = hashlib.sha256()
    for name in sample.__dataclass_fields__:
        value = getattr(sample, name)
        digest.update(name.encode() + b"\0")
        if isinstance(value, np.ndarray):
            digest.update(json.dumps([value.dtype.str, list(value.shape)]).encode() + b"\0")
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(json.dumps(value.item() if isinstance(value, np.generic) else value,
                                     allow_nan=False).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _output_report(sample, report):
    report = dict(report)
    area_a, area_b = int(sample.mask_a.sum()), int(sample.mask_b.sum())
    if not min(area_a, area_b):
        raise ValueError("empty output fragment")
    if bool(sample.label) != bool(report["label"]):
        raise ValueError("geometry changed the source label")
    if not sample.label and (sample.translation_valid or np.any(sample.target_a >= 0)
                             or np.any(sample.target_b >= 0)):
        raise ValueError("negative output acquired positive correspondence/pose GT")
    report.update(output_area_a_px=area_a, output_area_b_px=area_b,
        output_area_ratio=min(area_a, area_b) / max(area_a, area_b),
        output_token_matches=int((sample.target_a >= 0).sum()))
    return report


class CurveDiagnosticRows:
    """One overlay materialization; reference lookups reuse its current cache."""
    def __init__(self, overlay, indices):
        self.overlay, self.indices = overlay, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, offset):
        index = self.indices[offset]
        second, second_report = self.overlay.materialize(index)
        reference = self.overlay.reference
        first = reference[index]
        first_report = reference.diagnostics(index)
        if first.label:
            _assert_same_sample(first, second)
            if second_report["hard_negative"]["replaced"]:
                raise AssertionError("positive slot was replaced")
        return dict(schema_version="rachel-curve-hardneg-epoch-row/1", epoch=1,
            source_index=index, reference_pair_id=first.pair_id, label=bool(first.label),
            positive_arrays_identical=True if first.label else None,
            arms=dict(e4v2=_output_report(first, first_report),
                      e5v2=_output_report(second, second_report)))


class _Counts:
    def __init__(self):
        self.counts, self.reasons, self.area_bins, self.fallbacks = (Counter() for _ in range(4))
        self.retention = []
        self.ratios = {key: [] for key in ("positive", "negative", "applied_positive", "applied_negative")}

    def add(self, report):
        label = "positive" if report["label"] else "negative"
        self.counts.update(rows=1)
        self.counts[label] += 1
        self.counts["requested_" + label] += int(report["requested"])
        self.counts["applied_" + label] += int(report["applied"])
        self.reasons[report["reason"]] += 1
        ratio = report["output_area_ratio"]
        self.ratios[label].append(ratio)
        self.area_bins[label + ("_area_lt025" if ratio < .25 else "_area_ge025")] += 1
        if report["applied"]:
            self.ratios["applied_" + label].append(ratio)
            if report["label"]:
                retained = report["source_seam_retention"]
                if not .25 <= retained <= .75:
                    raise AssertionError("accepted source seam is not 25–75% partial")
                self.retention.append(retained)
        detail = report.get("hard_negative", {})
        if detail.get("replaced"):
            if label != "negative":
                raise AssertionError("overlay changed a positive slot")
            self.counts["replacements_negative"] += 1
            self.counts["replacement_curve_attempts"] += int(detail["curve_attempted"])
            self.counts["replacement_curve_applied"] += int(detail["curve_applied"])
            fallback = detail.get("curve_fallback_reason")
            self.counts["replacement_curve_fallbacks"] += int(bool(fallback))
            if fallback:
                self.fallbacks[fallback] += 1

    def report(self):
        counts = dict(self.counts)
        for key in ("rows", "positive", "negative", "requested_positive", "requested_negative",
                    "applied_positive", "applied_negative", "replacements_negative",
                    "replacement_curve_attempts", "replacement_curve_applied", "replacement_curve_fallbacks"):
            counts.setdefault(key, 0)
        return dict(counts=counts, reasons=dict(self.reasons), area_bins=dict(self.area_bins),
            replacement_curve_fallback_reasons=dict(self.fallbacks),
            accepted_positive_seam_retention_quantiles=quantiles(self.retention),
            area_ratio_quantiles={key: quantiles(values) for key, values in self.ratios.items()})


def _packed_pair(arrays, prefix, sample, *, targets=True):
    for side in "ab":
        mask = module_mask(getattr(sample, "mask_" + side))
        arrays[prefix + "_" + side] = np.packbits(mask, axis=1)
        arrays[prefix + "_shape_" + side] = np.asarray(mask.shape, np.int64)
        arrays[prefix + "_points_" + side] = getattr(sample, "points_rc_" + side)
        arrays[prefix + "_contour_valid_" + side] = getattr(sample, "contour_valid_" + side)
        if targets:
            arrays[prefix + "_target_" + side] = getattr(sample, "target_" + side)
    if targets:
        arrays[prefix + "_translation_rc"] = sample.translation_a_to_b_rc
        arrays[prefix + "_translation_xy_cartesian"] = sample.translation_a_to_b_xy_cartesian
        arrays[prefix + "_translation_valid"] = sample.translation_valid


def _save_examples(root, base, overlay, positives, negatives):
    for number, index in enumerate(positives, 1):
        old, curve = base[index], overlay.reference[index]
        report = overlay.reference.diagnostics(index)
        second, _ = overlay.materialize(index)
        _assert_same_sample(curve, second)
        arrays = {}
        _packed_pair(arrays, "original", old)
        _packed_pair(arrays, "curve", curve)
        np.savez_compressed(root / ("positive_%02d.npz" % number), **arrays)
        save_json(root / ("positive_%02d.json" % number), dict(source_index=index,
            shared_between_arms=True, placement="unchanged original GT; no prediction", report=report))
    for number, index in enumerate(negatives, 1):
        curve, report = overlay.materialize(index)
        source, inner = overlay.mapping[index]
        raw = overlay.loaders[source][inner]
        arrays = {}
        _packed_pair(arrays, "reference", overlay.reference[index], targets=False)
        _packed_pair(arrays, "replacement_original", raw, targets=False)
        _packed_pair(arrays, "replacement_output", curve, targets=False)
        # Negative source frames have no positive join transform; never draw one.
        np.savez_compressed(root / ("replacement_negative_%02d.npz" % number), **arrays)
        save_json(root / ("replacement_negative_%02d.json" % number), dict(source_index=index,
            label=False, join_gt_available=False, placement="separate native canvases; no invented GT",
            overlay_entry=overlay.entries[index], report=report))


def prepare(args):
    torch.set_num_threads(1)
    identity = input_identity(args)
    base, provenance = make_training_dataset(args.dataset, args.train_manifest)
    metadata = read_pair_metadata(args.train_manifest)
    _validate_pool(base, metadata)
    reference = CurvedPartialDataset(base, bank=args.outline_bank, pair_metadata=metadata,
        seed=CURVE_SEED, epoch=1, config=CURVE_CONFIG)
    overlay = NegativeOverlayDataset(reference, args.negative_overlay,
        expected_manifest=args.train_manifest, metadata=metadata)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    limit = min(len(reference), args.limit) if args.limit is not None else len(reference)
    full = limit == len(reference) and args.limit is None
    loader = DataLoader(CurveDiagnosticRows(overlay, tuple(range(limit))), batch_size=16,
        num_workers=args.workers, collate_fn=identity_collate, shuffle=False)
    statistics = {arm: _Counts() for arm in ARMS}
    positives, negatives = [], []
    checked = rows = 0
    started = time.perf_counter()
    manifest_name = "curve_hardneg_manifest_epoch1.jsonl"
    with (root / manifest_name).open("x", encoding="utf-8") as stream:
        for reports in loader:
            for record in reports:
                rows += 1
                for arm in ARMS:
                    statistics[arm].add(record["arms"][arm])
                checked += int(record["positive_arrays_identical"] is True)
                first, second = record["arms"]["e4v2"], record["arms"]["e5v2"]
                if first["label"] and first["applied"] and len(positives) < 4:
                    positives.append(record["source_index"])
                if second["hard_negative"]["replaced"] and len(negatives) < 4:
                    negatives.append(record["source_index"])
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if rows % 256 == 0 or rows == limit:
                status = dict(schema_version=SCHEMA, status="running", phase="prepare_train_curve_hardneg",
                    rows=rows, expected=limit, counts={arm: statistics[arm].report()["counts"] for arm in ARMS},
                    seconds=time.perf_counter() - started)
                save_json(root / "status.json", status)
                print(json.dumps(status, allow_nan=False), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    summaries = {arm: statistics[arm].report() for arm in ARMS}
    counts = {arm: summaries[arm]["counts"] for arm in ARMS}
    if rows != limit or checked != counts["e4v2"]["positive"]:
        raise RuntimeError("manifest/positive identity checks are incomplete")
    if full:
        for arm in ARMS:
            if (counts[arm]["rows"], counts[arm]["positive"], counts[arm]["negative"]) != (24000, 12000, 12000):
                raise RuntimeError("full manifest changed matched24k balance")
        if not 0 < counts["e4v2"]["applied_positive"] == counts["e4v2"]["applied_negative"]:
            raise RuntimeError("reference coupled curve application must be balanced and nonzero")
        if not 0 < counts["e5v2"]["replacements_negative"] <= 2400:
            raise RuntimeError("full overlay must replace 1–2400 negative slots")
        if counts["e5v2"]["replacement_curve_applied"] == 0:
            raise RuntimeError("full overlay contains no accepted curved replacement for branch smoke coverage")
        if counts["e4v2"]["applied_positive"] != counts["e5v2"]["applied_positive"]:
            raise RuntimeError("positive curve applications differ between arms")
    _save_examples(root, base, overlay, positives, negatives)
    result = dict(schema_version=SCHEMA, status="complete", full_train=full, probe_only=not full,
        epoch=1, curve_seed=CURVE_SEED, input_identity=identity, counts=counts, arms=summaries,
        positive_arrays_identical=True, positive_arrays_checked=checked,
        config=asdict(CURVE_CONFIG), provenance=provenance, generation_manifest=manifest_name,
        online_regenerated_each_epoch=True, cpu_geometry_only=True, weathering_check_requires_gpu_smoke=True,
        positive_example_indices=positives, replacement_negative_example_indices=negatives,
        example_selection="first four accepted positives and first four overlay negatives in source order",
        original_files_modified=False, seconds=time.perf_counter() - started)
    save_json(root / "summary.json", result)
    save_json(root / "status.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result


class _PositiveCheckedDataset:
    """Verify the exact positive arrays/GT actually consumed by smoke training."""
    def __init__(self, dataset, fingerprints):
        self.dataset, self.fingerprints = dataset, fingerprints

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, report = self.dataset[index]
        if sample.label and _sample_fingerprint(sample) != self.fingerprints.get(index):
            raise AssertionError("weathered positive input/GT differs from shared E4v2 reference")
        return sample, report


def _covered_replacement(args, mapping, identity):
    """Select data-augmentation branch coverage, never a model-result outcome."""
    summary_path = (Path(args.preparation_summary) if getattr(args, "preparation_summary", None)
                    else Path(args.output).parent / "preparation" / "summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("schema_version") != SCHEMA or summary.get("status") != "complete"
            or not summary.get("full_train") or summary.get("probe_only")
            or not summary.get("positive_arrays_identical")):
        raise ValueError("branch smoke requires a completed full24k CPU preparation receipt")
    expected = {key: value for key, value in identity.items() if key not in ("checkpoint", "checkpoint_sha256")}
    if summary.get("input_identity") != expected:
        raise ValueError("CPU preparation identity differs from smoke inputs")
    manifest_path = summary_path.parent / summary["generation_manifest"]
    with manifest_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            detail = record["arms"]["e5v2"].get("hard_negative", {})
            if detail.get("replaced") and detail.get("curve_applied"):
                index = int(record["source_index"])
                if index not in mapping or record["label"]:
                    raise ValueError("CPU curve-coverage row is not a mapped negative")
                return index, dict(summary=str(summary_path.resolve()),
                    summary_sha256=file_digest(summary_path), manifest=str(manifest_path.resolve()),
                    manifest_sha256=file_digest(manifest_path), source_index=index)
    raise ValueError("full CPU manifest has no actually curved replacement-negative branch")


def smoke(args):
    """64 positive + 64 negative exposures/arm, eight discarded updates each."""
    from types import SimpleNamespace
    from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint
    from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
    from staging.pairwise_v0_2.training.rachel_weathering_training import make_weathering_loader, train_weathering_epoch
    from staging.pairwise_v0_2.training import rachel_n512_runner as runner
    from experiments.rachel_n512_formal_30k.train_edge_weathering import _sha256, SOURCE_SHA256

    if not args.checkpoint:
        raise ValueError("--smoke requires --checkpoint")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("smoke requires an available CUDA device")
    if _sha256(args.checkpoint) != SOURCE_SHA256:
        raise ValueError("smoke source must be the common 350bf9 checkpoint")
    identity = input_identity(args, checkpoint=True)
    base, _ = make_training_dataset(args.dataset, args.train_manifest)
    metadata = read_pair_metadata(args.train_manifest)
    _validate_pool(base, metadata)
    torch.set_num_threads(1)
    recipes = {arm: build_recipe(arm, args.train_manifest, args.outline_bank, args.negative_overlay)
               for arm in ARMS}
    datasets = {arm: recipe["build_dataset"](base, seed=SEED, cache_dir=args.cache_dir)
                for arm, recipe in recipes.items()}
    for dataset in datasets.values():
        dataset.set_epoch(1)
    mapping = datasets["e5v2"].geometry.mapping
    positives = [i for i, (_, label) in enumerate(metadata) if label][:64]
    covered, coverage = _covered_replacement(args, mapping, identity)
    replacements = [covered] + [index for index in sorted(mapping) if index != covered][:31]
    other_negatives = [i for i, (_, label) in enumerate(metadata) if not label and i not in mapping][:32]
    if (len(positives), len(replacements), len(other_negatives)) != (64, 32, 32):
        raise ValueError("smoke needs 64 positives, 32 replacement and 32 other negative slots")
    negatives = replacements + other_negatives
    indices = tuple(i for pair in zip(positives, negatives) for i in pair)
    # Small digest sidecar, not 128 pairs of full-resolution arrays in RAM.
    fingerprints = {index: _sample_fingerprint(datasets["e4v2"][index][0]) for index in positives}
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    for arm in ARMS:
        runner._set_determinism(SEED)
        source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = load_rachel_checkpoint(source)
        check_fixed_architecture(model, source)
        device = torch.device(args.device)
        model = model.to(device).train().requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
        dataset = _PositiveCheckedDataset(datasets[arm], fingerprints)
        loader = make_weathering_loader(dataset, indices, batch_size=4,
            num_workers=args.workers, seed=SEED, contour_cap=512)
        cfg = SimpleNamespace(batch_size=4, effective_batch_size=16, log_every=16, output=None)
        torch.cuda.reset_peak_memory_stats(device)
        report = train_weathering_epoch(model, loader, optimizer,
            RachelN512LossConfig(**source["loss_config"]), device, cfg, 1, recipes[arm]["variant"])
        if report["samples"] != 128 or report["optimizer_updates"] != 8 or not np.isfinite(report["mean_loss"]):
            raise RuntimeError("curve/hard-negative GPU smoke budget or finite-loss check failed")
        weather = report["weathering"]
        if any(weather["by_label"][label]["pair_exposures"] != 64 for label in ("positive", "negative")):
            raise RuntimeError("GPU smoke changed its 64/64 class balance")
        if weather["by_label"]["positive"]["changed_pair_exposures"] == 0:
            raise RuntimeError("GPU smoke must exercise actual positive weathering")
        if weather["partial_seam"]["counts"].get("applied_pair_exposures", 0) == 0:
            raise RuntimeError("GPU smoke saw no actual curved crop")
        hard = weather.get("hard_negative", {})
        if arm == "e5v2" and (hard.get("replacement_exposures", 0) != 32
                               or hard.get("replacement_curve_applied", 0) == 0):
            raise RuntimeError("E5v2 smoke must exercise 32 replacements and an actual replacement crop")
        if arm == "e4v2" and hard.get("replacement_exposures", 0):
            raise RuntimeError("E4v2 smoke unexpectedly replaced negative slots")
        results[arm] = report
        del model, optimizer, source, loader
        torch.cuda.empty_cache()
    result = dict(schema_version=SMOKE_SCHEMA, status="complete", basic_contract_passed=True,
        input_identity=identity, epoch=1, seed=SEED, curve_seed=CURVE_SEED,
        positive_arrays_identical=True, positive_weathered_arrays_identical=True,
        positive_arrays_checked_per_arm=64, positive_weathered_fingerprints=fingerprints,
        selection_policy="first64positive; first actual curved overlay-negative from full CPU manifest then earliest other31 overlay negatives; first32 nonoverlay negatives. Selection covers an augmentation branch, never a model prediction or metric outcome.",
        replacement_curve_coverage=coverage,
        source_indices=list(indices), positive_indices=positives,
        replacement_negative_indices=replacements, other_negative_indices=other_negatives,
        weights_saved=False, formal_weights_reused=False, optimizer_exposures_total=256, results=results)
    save_json(root / "preflight.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "train-manifest", "outline-bank", "negative-overlay", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir")
    p.add_argument("--preparation-summary", help="smoke full24k receipt; default OUTPUT/../preparation/summary.json")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.workers < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("invalid worker/limit count")
    if args.smoke and args.limit is not None:
        raise ValueError("--limit is a CPU probe option; smoke always uses 128 rows per arm")
    return smoke(args) if args.smoke else prepare(args)


if __name__ == "__main__":
    main()
