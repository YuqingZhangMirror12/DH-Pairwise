"""E1-only training adapter; the inference model and original loss stay intact.

New boundary points are supervised by inherited source-arc assignments. Their
coordinate difference includes the missing material, so the old zero-gap pose
auxiliary is applied only to clean (including explicit fallback) positives.
Pair BCE, partial assignment, dustbins and Sinkhorn regularization are unchanged.
Neither augmentation metadata nor source coordinates enter the model forward.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import collate_rachel_pairs
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training.rachel_n512_loss import (
    RachelN512LossConfig, compute_rachel_n512_loss,
)


@dataclass(frozen=True)
class WeatheringBatch:
    batch: object
    reports: tuple
    pose_supervision_enabled: np.ndarray


def collate_weathered_pairs(samples, *, contour_cap=512):
    ordinary, reports = zip(*samples)
    batch = collate_rachel_pairs(ordinary, contour_cap=contour_cap)
    pose = np.asarray([r["pose_supervision_enabled"] for r in reports], dtype=np.bool_)
    changed = np.asarray([r["changed_pair"] for r in reports], dtype=np.bool_)
    if not np.array_equal(pose, (batch.labels == 1) & ~changed):
        raise ValueError("E1 pose sidecar must select exactly the unchanged positive samples")
    return WeatheringBatch(batch, tuple(reports), pose)


def make_weathering_loader(dataset, indices, *, batch_size, num_workers, seed, contour_cap):
    loader = runner._loader(dataset, indices, batch_size=batch_size,
                            num_workers=num_workers, seed=seed)
    loader.collate_fn = partial(collate_weathered_pairs, contour_cap=contour_cap)
    return loader


def compute_weathering_loss(output, labels, target_a, target_b,
                           translation_target_rc, translation_valid, *,
                           pose_supervision_enabled,
                           config=RachelN512LossConfig()):
    """Keep all old targets valid; remove zero-gap pose pressure on changed rows.

The retained pose loss is normalized over retained valid clean positives, as
the original loss was over its supervised positives. Clean-only batches take
the original implementation directly (including identical floating arithmetic).
"""
    if (pose_supervision_enabled.dtype != torch.bool or
            pose_supervision_enabled.shape != translation_valid.shape):
        raise TypeError("pose_supervision_enabled must be bool [B]")
    if (pose_supervision_enabled & ~translation_valid).any().item():
        raise ValueError("E1 cannot enable pose supervision for a negative sample")
    args = (output, labels, target_a, target_b, translation_target_rc, translation_valid)
    if torch.equal(pose_supervision_enabled, translation_valid):
        return compute_rachel_n512_loss(*args, config=config)
    base = compute_rachel_n512_loss(*args, config=replace(config, translation_weight=0.0))
    mask = pose_supervision_enabled & translation_valid & output.training_valid
    values = F.smooth_l1_loss(
        output.translation_hat_rc / config.translation_scale_px,
        translation_target_rc.to(output.translation_hat_rc.dtype) / config.translation_scale_px,
        reduction="none",
    ).mean(dim=1)
    pose = (values * mask.to(values.dtype)).sum() / mask.sum().clamp_min(1)
    total = base.total + config.translation_weight * pose
    if config.validate_runtime_targets and not torch.isfinite(total).item():
        raise FloatingPointError("E1 loss is nonfinite")
    return replace(base, total=total, translation_smooth_l1=pose,
                   translation_count=int(mask.sum().item()) if config.collect_cpu_diagnostics else -1)


class WeatheringStatistics:
    """Actual exposure counts, not an assumed 70/25/5 mixture after skips."""
    def __init__(self):
        self.counts = Counter()
        self.tiers = Counter()
        self.skip_reasons = Counter()
        self.fallback_reasons = Counter()
        self.by_label = {"positive": Counter(), "negative": Counter()}
        self.by_size = {"tiny_ratio_lt_025": Counter(), "other": Counter()}
        self.area_loss_sum = 0.0
        self.audit_examples = []
        self.partial = Counter()
        self.partial_by_label = {"positive": Counter(), "negative": Counter()}
        self.partial_reasons = Counter()
        self.hard_negative = Counter()

    def add(self, wrapped):
        batch = wrapped.batch
        for index, report in enumerate(wrapped.reports):
            if "hard_negative" in report:
                detail = report["hard_negative"]
                self.hard_negative.update(pair_exposures=1,
                    replacement_exposures=int(bool(detail.get("replaced"))),
                    replacement_curve_attempts=int(bool(detail.get("curve_attempted"))),
                    replacement_curve_applied=int(bool(detail.get("curve_applied"))))
            group = self.by_label["positive" if batch.labels[index] else "negative"]
            if "partial_seam" in report:
                detail = report["partial_seam"]
                values = dict(pair_exposures=1,
                              requested_pair_exposures=int(bool(detail.get("requested"))),
                              applied_pair_exposures=int(bool(detail.get("applied"))))
                self.partial.update(values)
                self.partial_by_label["positive" if batch.labels[index] else "negative"].update(values)
                reason = detail.get("fallback_reason") or detail.get("reason")
                if reason and not detail.get("applied"):
                    self.partial_reasons[str(reason)] += 1
            areas = [float(report.get("side_" + side, {}).get("original_area_px", 0)) for side in "ab"]
            if min(areas) <= 0:
                areas = [float(getattr(batch, "mask_" + side)[index].sum()) for side in "ab"]
            ratio = min(areas) / max(1.0, max(areas))
            size = self.by_size["tiny_ratio_lt_025" if ratio < .25 else "other"]
            values = dict(pair_exposures=1, changed_pair_exposures=int(report["changed_pair"]),
                          pose_supervised_pair_exposures=int(report["pose_supervision_enabled"]),
                          inherited_match_tokens=int(report.get("inherited_match_count", 0)),
                          effective_supervised_match_tokens=int(report.get("effective_supervised_match_count", 0)),
                          ignored_tokens=int(report.get("ignored_token_count", 0)),
                          fallback_pair_exposures=int(bool(report.get("fallback_reason"))))
            self.counts.update(values)
            group.update(values)
            size.update(values)
            if report.get("fallback_reason"):
                self.fallback_reasons[str(report["fallback_reason"])] += 1
            for side in "ab":
                detail = report["side_" + side]
                tier = str(detail["tier"])
                self.tiers[tier] += 1
                applied = bool(report["changed_" + side])
                self.counts["fragment_exposures"] += 1
                self.counts["changed_fragment_exposures"] += int(applied)
                self.counts["attempted_applied_fragment_exposures"] += int(detail.get("attempted_applied", detail.get("applied", False)))
                if detail.get("skipped"):
                    self.skip_reasons[str(detail.get("skip_reason"))] += 1
                if applied:
                    self.area_loss_sum += float(detail.get("removed_fraction", 0.0))
            if len(self.audit_examples) < 12:
                self.audit_examples.append(dict(pair_id=batch.pair_ids[index],
                    label=bool(batch.labels[index]), changed_pair=bool(report["changed_pair"]),
                    tiers=[report["side_" + side]["tier"] for side in "ab"],
                    inherited_match_count=int(report.get("inherited_match_count", 0)),
                    fallback_reason=report.get("fallback_reason")))

    def report(self):
        result = dict(counts=dict(self.counts), requested_fragment_tier_exposures=dict(self.tiers),
                    skipped_fragment_reasons=dict(self.skip_reasons),
                    fallback_pair_reasons=dict(self.fallback_reasons),
                    by_label={k: dict(v) for k, v in self.by_label.items()},
                    by_source_area_ratio={k: dict(v) for k, v in self.by_size.items()},
                    actual_changed_fragment_rate=self.counts["changed_fragment_exposures"] / max(1, self.counts["fragment_exposures"]),
                    actual_changed_pair_rate=self.counts["changed_pair_exposures"] / max(1, self.counts["pair_exposures"]),
                    mean_removed_fraction_among_applied=self.area_loss_sum / max(1, self.counts["changed_fragment_exposures"]),
                    audit_examples=self.audit_examples)
        if self.partial:
            result["partial_seam"] = dict(counts=dict(self.partial),
                by_label={k: dict(v) for k, v in self.partial_by_label.items()},
                fallback_reasons=dict(self.partial_reasons))
        if self.hard_negative:
            result["hard_negative"] = dict(self.hard_negative)
        return result


def train_weathering_epoch(model, loader, optimizer, loss_config, device, args, epoch, variant):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_samples, total_loss, updates = 0, 0.0, 0
    started = time.perf_counter()
    accumulation = args.effective_batch_size // args.batch_size
    if accumulation < 1 or args.effective_batch_size % args.batch_size:
        raise ValueError("effective batch must be an integer multiple of the microbatch")
    statistics = WeatheringStatistics()
    output_dir = Path(args.output) if getattr(args, "output", None) else None
    initial_status = {}
    if output_dir is not None and (output_dir / "status.json").exists():
        with (output_dir / "status.json").open() as stream:
            initial_status = json.load(stream)
    for step, wrapped in enumerate(loader):
        group_start = (step // accumulation) * accumulation
        group_samples = min(args.effective_batch_size, len(loader.dataset) - group_start * args.batch_size)
        inputs, targets = runner._full_batch(wrapped.batch, device)
        result = model(*inputs)
        pose = torch.as_tensor(wrapped.pose_supervision_enabled, dtype=torch.bool, device=device)
        loss = compute_weathering_loss(result, *targets, config=loss_config,
                                        pose_supervision_enabled=pose)
        count = len(wrapped.batch.pair_ids)
        (loss.total * (count / group_samples)).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        total_samples += count
        total_loss += float(loss.total.detach().cpu()) * count
        statistics.add(wrapped)
        if (step + 1) % args.log_every == 0:
            progress = dict(event="train_progress", variant=variant, epoch=epoch,
                            samples=total_samples, updates=updates, mean_loss=total_loss / total_samples,
                            seconds=time.perf_counter() - started,
                            weathering=statistics.report())
            print(json.dumps(progress, ensure_ascii=False, allow_nan=False), flush=True)
            if output_dir is not None:
                runner._atomic_json(output_dir / "status.json", dict(status="running", phase="train",
                    epoch=epoch, global_exposure=initial_status.get("global_exposure", 0) + total_samples,
                    optimizer_updates=initial_status.get("optimizer_updates", 0) + updates,
                    progress=progress))
    return dict(samples=total_samples, optimizer_updates=updates,
                mean_loss=total_loss / max(1, total_samples), mean_seam_loss=0.0,
                seconds=time.perf_counter() - started,
                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device),
                weathering=statistics.report())


__all__ = ["WeatheringBatch", "collate_weathered_pairs", "make_weathering_loader",
           "compute_weathering_loss", "WeatheringStatistics", "train_weathering_epoch"]
