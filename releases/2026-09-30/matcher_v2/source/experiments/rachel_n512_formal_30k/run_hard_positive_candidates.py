"""Frozen-E1 hard-positive P and independent candidate-supervised R experiment.

Only TRAIN receives curved partial seams/weathering and GT-generated R poses.
The original Top2 layout is preserved. VAL/TEST/REAL R queries use only that
predicted pose, and REAL GT is opened after all scores have been serialized.
Atomic per-batch caches make resumed extraction skip completed GPU work.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from . import run_damage_separate_heads as old
from . import run_seam_geometry_head as base
from . import hard_positive_head_fit as fit
from .train_partial_seam import read_pair_metadata
from staging.pairwise_v0_2.models.candidate_pose_evidence import (
    geometry_at_translations, generate_train_candidates,
)
from staging.pairwise_v0_2.pairwise_data.rachel_guided_partial_dataset import GuidedPartialDataset
from staging.pairwise_v0_2.pairwise_data.rachel_partial_seam_dataset import PartialSeamConfig
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import RachelWeatheredDataset
from staging.pairwise_v0_2.training.rachel_weathering_training import make_weathering_loader

SCHEMA = "rachel-hard-positive-candidates/1"
SEED, BATCH = 260911, 8
DECODER = base.DECODER


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(base.common.clean(value), ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(base.common.clean(row), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def status(args, stage, **extra):
    # Fit callbacks carry their own schema; the driver owns the status envelope.
    record = dict(extra)
    record.update(schema_version=SCHEMA, status="running", pid=os.getpid(), stage=stage)
    save(Path(args.output) / "status.json", record)
    base.emit(record)


def pair_seed(pair_id):
    return int.from_bytes(hashlib.sha256((str(SEED) + pair_id).encode()).digest()[:8], "little")


class HardInputDataset:
    def __init__(self, source, manifest, bank, cache):
        self.partial = GuidedPartialDataset(source, pair_metadata=read_pair_metadata(manifest),
            bank=bank, seed=260910, epoch=1,
            config=PartialSeamConfig(probability=1., max_attempts=12))
        self.weather = RachelWeatheredDataset(self.partial, seed=260909, epoch=1, cache_dir=cache)
        self.split, self.root, self.contour_cap = "train", getattr(source, "root", None), 512

    def __len__(self):
        return len(self.weather)

    def __getitem__(self, index):
        sample, report = self.weather[index]
        report = dict(report, partial_seam=self.partial.diagnostics(index))
        return sample, report


def reference_scores(rows, reference_heads, heads=None):
    if heads is not None:
        fit.score_heads(heads, rows)
    x = base.feature_matrix(rows)
    with torch.inference_mode():
        p = torch.sigmoid(reference_heads["pdamage"](x)).cpu().numpy()
        r = torch.sigmoid(reference_heads["rdamage"](x)).cpu().numpy()
    for row, pv, rv in zip(rows, p, r):
        row["old_pdamage_probability"] = float(pv)
        row.setdefault("head_scores", {}).update(old_pdamage=float(pv),
            original_fused=row["classification"]["fused"])
        row.setdefault("pose_reliability", {})["old_rdamage"] = dict(probability=float(rv))
    return rows


def predict_batch(model, batch, device, *, training_candidates=False):
    tensors = [base.sealed._tensor(getattr(batch, name), device, dtype) for name, dtype in (
        ("mask_a", torch.float32), ("mask_b", torch.float32), ("points_rc_a", torch.float32),
        ("points_rc_b", torch.float32), ("contour_valid_a", torch.bool), ("contour_valid_b", torch.bool))]
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = model(*tensors)
    probabilities = {n: getattr(output, n + "_probability").detach().float().cpu().numpy()
                     for n in ("coarse", "local", "fused")}
    logits = {n: getattr(output, n + "_logit").detach().float().cpu().numpy() for n in ("coarse", "local")}
    matrices = output.assignment.detach().float().cpu().numpy()
    rows, candidates = [], []
    for i, pair_id in enumerate(batch.pair_ids):
        inputs = (batch.points_rc_a[i], batch.points_rc_b[i], matrices[i],
            batch.contour_valid_a[i], batch.contour_valid_b[i], batch.mask_a[i], batch.mask_b[i])
        geometry, estimate = base.geometry_evidence(*inputs)
        score_features = [float(logits[n][i]) for n in ("coarse", "local")]
        proposed = []
        if training_candidates and bool(batch.labels[i]):
            if not bool(batch.translation_valid[i]):
                raise ValueError("TRAIN true pair lacks a valid unchanged translation GT")
            proposed = generate_train_candidates(batch.translation_a_to_b_rc[i],
                estimate.t_a_to_b_rc if estimate.valid else None, seed=pair_seed(pair_id),
                correct_count=1, near_wrong_count=1, far_wrong_count=1)
        positions = ([estimate.t_a_to_b_rc] if estimate.valid else []) + [p["translation_rc"] for p in proposed]
        evidence = geometry_at_translations(*inputs, positions) if positions else np.empty((0, 22), np.float32)
        native = evidence[0] if estimate.valid else np.zeros(22, np.float32)
        offset = int(estimate.valid)
        for proposal, features in zip(proposed, evidence[offset:]):
            candidates.append(dict(pair_id=pair_id, **proposal, features=score_features + features.tolist()))
        diagnostic = asdict(estimate)
        for key in ("t_a_to_b_rc", "candidate_indices", "inlier_mask"):
            diagnostic.pop(key, None)
        rows.append(base.common.clean(dict(pair_id=pair_id,
            fragment_a=batch.fragment_a_tokens[i], fragment_b=batch.fragment_b_tokens[i],
            decision_valid=bool(output.decision_valid[i].item()),
            classification={n: float(v[i]) for n, v in probabilities.items()},
            features=score_features + geometry.tolist(),
            reliability_features=score_features + native.tolist(),
            layouts={DECODER: dict(valid=bool(estimate.valid), translation_rc=estimate.t_a_to_b_rc,
                offset_b_in_a_rc=-estimate.t_a_to_b_rc, diagnostics=diagnostic)},
            **base.pair_size_metadata(batch.mask_a[i], batch.mask_b[i]))))
    return rows, candidates


def extract(args, split, model, identity, refs, *, dataset=None, indices=None, heads=None):
    root = Path(args.output) / "cache" / split
    root.mkdir(parents=True, exist_ok=True)
    complete = root / "complete.json"
    if complete.exists():
        record = read(complete)
        if record["matcher_checkpoint_id"] != identity["checkpoint_sha256"]:
            raise ValueError("cached matcher differs")
        return read_rows(root / "pair_results.jsonl"), read_rows(root / "candidates.jsonl")
    metadata, targets = None, {}
    training = split == "train_augmented"
    if split == "real":
        metadata, arrays = base.load_prepared_cache(args.prepared_cache)
        total = 1016
        batches = base.real.input_batches(metadata, arrays, BATCH)
    else:
        if dataset is None:
            dataset = base.RachelPairDataset(Path(args.dataset), split)
        indices = list(range(len(dataset))) if indices is None else list(indices)
        total = len(indices)
        loader = make_weathering_loader if training else base.make_ablation_loader
        batches = loader(dataset, indices, batch_size=BATCH, num_workers=args.workers,
            seed=SEED, contour_cap=512)
    (root / "chunks").mkdir(exist_ok=True)
    started, all_rows, all_candidates, partial_count, weather_count = time.monotonic(), [], [], 0, 0
    for chunk_index, item in enumerate(batches):
        batch = item.batch if training else item
        path = root / "chunks" / ("%06d.json" % chunk_index)
        if path.exists():
            chunk = read(path)
            if [r["pair_id"] for r in chunk["rows"]] != list(batch.pair_ids):
                raise ValueError("cached batch identities differ")
        else:
            rows, candidates = predict_batch(model, batch, torch.device(args.device), training_candidates=training)
            reference_scores(rows, refs, heads)
            target = {} if split == "real" else {pair_id: dict(label=bool(batch.labels[i]),
                source_unit_ids=[], translation_rc=base.common.clean(batch.translation_a_to_b_rc[i])
                    if bool(batch.translation_valid[i]) else None) for i, pair_id in enumerate(batch.pair_ids)}
            reports = item.reports if training else []
            chunk = dict(rows=rows, candidates=candidates, targets=target,
                partial_count=sum(bool(r.get("partial_seam", {}).get("applied")) for r in reports),
                weather_count=sum(bool(r["changed_pair"]) for r in reports))
            save(path, chunk)
        all_rows.extend(chunk["rows"])
        all_candidates.extend(chunk["candidates"])
        targets.update(chunk["targets"])
        partial_count += chunk["partial_count"]
        weather_count += chunk["weather_count"]
        if len(all_rows) % 64 == 0:
            status(args, "extract_" + split, processed=len(all_rows), total=total,
                elapsed_s=time.monotonic() - started)
    if len(all_rows) != total or len({r["pair_id"] for r in all_rows}) != total:
        raise ValueError("incomplete or duplicated extraction population")
    write_rows(root / "pair_predictions.jsonl", all_rows)
    save(root / "prediction_complete.json", dict(status="all_predictions_frozen", sample_count=total,
        real_translation_gt_opened=False, head_scores_frozen=heads is not None))
    rows = (base.real.attach_ground_truth(all_rows, metadata["pairs"], args.translation_gt_json)
        if split == "real" else base.fixed.attach_test_targets(all_rows, targets))
    write_rows(root / "pair_results.jsonl", rows)
    write_rows(root / "candidates.jsonl", all_candidates)
    record = dict(status="complete", split=split, sample_count=total,
        matcher_checkpoint_id=identity["checkpoint_sha256"], partial_changed_count=partial_count,
        weather_changed_count=weather_count, candidate_count=len(all_candidates),
        candidate_success=sum(r["label"] for r in all_candidates),
        candidate_failure=sum(not r["label"] for r in all_candidates),
        candidates_train_only=True, reliability_gates_pairability=False,
        target_gt_evaluation_after_prediction_freeze=split == "real")
    if training:
        positives = [r for r in rows if r["label"]]
        record["native_pose_failure_count"] = int((~old.pose_targets(positives)).sum())
        record["hard_positive_pool_count"] = sum(not bool(old.pose_targets([r])[0])
            or r["old_pdamage_probability"] < .2178770750761032 for r in positives)
    save(complete, record)
    return rows, all_candidates


def summarize(rows, freeze, old_freeze, *, with_groups=True):
    labels = np.asarray([r["label"] for r in rows], bool)
    success = old.pose_targets(rows)
    thresholds = {n: freeze["selected"][n]["threshold"] for n in ("pbase", "puniform", "phard")}
    thresholds.update(original_fused=old_freeze["original_fused_threshold"],
        old_pdamage=old_freeze["selected"]["pdamage"]["threshold"])
    result = dict(sample_count=len(rows), positive_count=int(labels.sum()),
        negative_count=int((~labels).sum()), methods={}, reliability={},
        raw_layout10_success_count=int(success.sum()), raw_layout_shared=True)
    for name, threshold in thresholds.items():
        score = np.asarray([r["head_scores"][name] for r in rows])
        accepted = score >= threshold
        count = int((accepted & success).sum())
        metric = base.common.classification(labels, score, threshold)
        if len(np.unique(labels)) != 2:
            metric.update(auroc=None, auprc=None)
        result["methods"][name] = dict(classification=metric,
            accepted_layout10_count=count, accepted_layout10_recall=count / max(1, int(labels.sum())),
            rejected_layout10_success_count=int((~accepted & success).sum()))
    for name in ("rnative", "rcandidate", "old_rdamage"):
        threshold = (old_freeze["selected"]["rdamage"]["threshold"] if name == "old_rdamage"
            else freeze["selected"][name]["threshold"])
        result["reliability"][name] = old.reliability_metrics(rows,
            [r["pose_reliability"][name]["probability"] for r in rows], threshold)
    if with_groups:
        result["size_strata"] = {key: summarize([r for r in rows if r["size_ratio_stratum"] == key],
            freeze, old_freeze, with_groups=False) for key in sorted({r["size_ratio_stratum"] for r in rows})}
        if rows and "strict_member" in rows[0]:
            result["strict547"] = summarize([r for r in rows if r["strict_member"]], freeze, old_freeze, with_groups=False)
    return result


def run(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        base.sealed._set_determinism(SEED)
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("run this experiment on the authorized remote GPU")
        signature = {k: str(v) for k, v in vars(args).items() if k != "resume"}
        protocol_path = root / "protocol.json"
        if protocol_path.exists() and read(protocol_path)["arguments"] != signature:
            raise ValueError("resume arguments differ")
        try:
            status(args, "load_frozen_matcher")
            model, identity, _ = old.load_e1_winner(args.checkpoint)
            model.to(torch.device(args.device)).eval().requires_grad_(False)
            baseline = Path(args.baseline_heads)
            previous_freeze = read(baseline / "validation_freeze.json")
            sources = dict(matcher_checkpoint_id=identity["checkpoint_sha256"],
                prior_heads_checkpoint_id=previous_freeze["head_checkpoint_sha256"],
                train_manifest_sha256=base.sealed._sha256_file(Path(args.train_manifest)))
            if sources["train_manifest_sha256"] != previous_freeze["input_identity"]["train_manifest_sha256"]:
                raise ValueError("new TRAIN differs from the original E1 head TRAIN")
            if protocol_path.exists() and read(protocol_path).get("source_identity") != sources:
                raise ValueError("resume source identities differ")
            refs, _ = old.load_frozen_heads(baseline, identity["checkpoint_sha256"], previous_freeze["input_identity"])
            save(protocol_path, dict(schema_version=SCHEMA, status="running", arguments=signature,
                source_identity=sources, matcher_identity=identity, seed=SEED, matcher_frozen=True, layout_modified=False,
                reliability_gates_pairability=False, test_or_real_used_for_fit=False,
                no_artificial_straight_overlay=True, r_gt_candidates_train_only=True,
                hard_positive_rule="TRAIN native pose failure >10px or frozen old-Pdamage rejection",
                partial_recipe="TRAIN curve profile; paired pos/neg acceptance; oblique; 25-75% seam retained"))
            clean = read_rows(baseline / "cache/train_clean/pair_results.jsonl")
            damage = read_rows(baseline / "cache/train_damage/pair_results.jsonl")
            if len(clean) != 24000 or len(damage) != 24000:
                raise ValueError("requires original aligned 24k head TRAIN caches")
            status(args, "prepare_train_inputs")
            source, _ = base.make_training_dataset(Path(args.dataset), Path(args.train_manifest))
            data = HardInputDataset(source, args.train_manifest, args.outline_bank, root / "weather_cache")
            indices = None
            if args.prepare_limit:
                if args.prepare_limit % 2 or not 0 < args.prepare_limit <= 24000:
                    raise ValueError("smoke limit must be positive, even and at most24000")
                indices = sorted([i for label in (False, True) for i in
                    [j for j, row in enumerate(clean) if bool(row["label"]) == label][:args.prepare_limit // 2]])
            augmented, candidates = extract(args, "train_augmented", model, identity, refs,
                dataset=data, indices=indices)
            del data, source
            if args.prepare_limit:
                record = read(root / "cache/train_augmented/complete.json")
                save(root / "smoke.json", dict(record, smoke_only=True, formal_training_counted=False))
                save(root / "status.json", dict(status="smoke_complete", stage=None))
                return
            val, _ = extract(args, "val", model, identity, refs)
            destination = root / "training"
            if (destination / "validation_freeze.json").exists():
                heads, freeze = fit.load_heads(destination)
            else:
                destination.mkdir(exist_ok=True)
                status(args, "fit_heads")
                heads, freeze = fit.fit_heads(clean, damage, augmented, candidates, val,
                    output=destination, identity=identity,
                    callback=lambda record: status(args, "fit_heads", **record))
                heads, freeze = fit.load_heads(destination)
            del clean, damage, augmented, candidates
            reference_scores(val, refs, heads)
            save(root / "val_summary.json", summarize(val, freeze, previous_freeze))
            for split in ("test", "real"):
                rows, _ = extract(args, split, model, identity, refs, heads=heads)
                result = summarize(rows, freeze, previous_freeze)
                result.update(schema_version=SCHEMA, status="complete", split=split,
                    matcher_checkpoint_id=identity["checkpoint_sha256"], layout_modified=False,
                    reliability_gates_pairability=False, test_or_real_used_for_fit=False,
                    target_gt_evaluation_after_prediction_freeze=split == "real")
                save(root / (split + "_summary.json"), result)
            protocol = read(protocol_path)
            protocol["status"] = "complete"
            save(protocol_path, protocol)
            save(root / "status.json", dict(status="complete", stage=None, completed_at=time.time()))
        except Exception as error:
            save(root / "status.json", dict(status="failed", pid=os.getpid(), error=repr(error)))
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "train-manifest", "outline-bank", "baseline-heads", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--prepared-cache", type=Path, default=base.DEFAULT_PREPARED_CACHE)
    p.add_argument("--translation-gt-json", type=Path, default=base.real.DEFAULT_TRANSLATION_GT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-limit", type=int)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
