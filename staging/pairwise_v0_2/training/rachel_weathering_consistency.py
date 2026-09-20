"""E2: E1-identical weathering plus clean-teacher and gap-tolerant transport loss.

Only TRAIN loss sidecars see the clean sample and source-arc ancestry. The
student forward, frozen 800px frame, original placement GT, and inference model
are unchanged. The online teacher is the same model in eval/no_grad mode (not
EMA, not an additional learned network). E2 is an experimental objective, not a
reproduction of a paper's weathering method.

The gap objective acts on assignment, NOT the raw boundary translation head:
delta_source = (new_b-new_a) - [(new_b-source_b)-(new_a-source_a)].
Thus missing material may separate the two boundaries at the correct original
placement. On inherited seam anchors, wrong candidate assignments still incur
a GT translation residual after this source correction. Source coordinates and
GT are used only during TRAIN; inference does not obtain a clean counterpart.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import collate_rachel_pairs
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import RachelWeatheredDataset
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
from staging.pairwise_v0_2.training.rachel_weathering_training import (
    WeatheringStatistics, collate_weathered_pairs, compute_weathering_loss,
)


SCHEMA = "rachel-weathering-source-consistency-e2/v1"


@dataclass(frozen=True)
class E2LossConfig:
    pair_consistency_weight: float = .1
    assignment_consistency_weight: float = .1
    gap_weight: float = .5
    gap_tolerance_px: float = 3.
    gap_scale_px: float = 32.
    epsilon: float = 1e-8

    def __post_init__(self):
        for key, value in vars(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(key + " must be finite and nonnegative")
        if self.gap_scale_px <= 0 or not 0 < self.epsilon < 1:
            raise ValueError("gap scale must be positive and epsilon in (0,1)")


class _CaptureBase:
    """Capture the E1 source read; avoid decoding the same sample twice."""
    def __init__(self, base):
        self.base = base
        self.root, self.split = getattr(base, "root", None), getattr(base, "split", None)
        self.last = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        self.last = self.base[index]
        return self.last


@dataclass(frozen=True)
class E2Sample:
    student: object
    report: dict
    clean: object
    ancestor_a: np.ndarray
    ancestor_b: np.ndarray


class E2Dataset(RachelWeatheredDataset):
    """Exactly E1's student/tier/seed/cache path, with loss-only source sidecars."""
    def __init__(self, base_dataset, seed=260909, epoch=0, cache_dir=None, **kwargs):
        super().__init__(_CaptureBase(base_dataset), seed=seed, epoch=epoch,
                         cache_dir=cache_dir, **kwargs)

    def __getitem__(self, index):
        student, report = super().__getitem__(index)
        clean = self.base_dataset.last
        ancestors = []
        for side in "ab":
            points = getattr(clean, "points_rc_" + side)
            valid = getattr(clean, "contour_valid_" + side)
            if report["changed_" + side]:
                view = self._fragment(getattr(clean, "fragment_" + side + "_token"),
                    getattr(clean, "mask_" + side), points, valid)
                if not np.array_equal(view.points, getattr(student, "points_rc_" + side)):
                    raise ValueError("E2 ancestry differs from the effective E1 student")
                ancestor = view.ancestor.copy()
            else:
                # Includes E1's explicit whole-pair clean fallback; never use an
                # attempted-but-rejected weathered mapping on a clean sample.
                ancestor = np.arange(len(points), dtype=np.int64)
                ancestor[~valid] = -1
            ancestors.append(ancestor)
        return E2Sample(student, report, clean, *ancestors)


@dataclass(frozen=True)
class E2Batch:
    batch: object
    reports: tuple
    pose_supervision_enabled: np.ndarray
    clean_batch: object
    ancestor_a: np.ndarray
    ancestor_b: np.ndarray


def collate_e2_pairs(samples, *, contour_cap=512):
    samples = tuple(samples)
    e1 = collate_weathered_pairs([(s.student, s.report) for s in samples], contour_cap=contour_cap)
    clean = collate_rachel_pairs([s.clean for s in samples], contour_cap=contour_cap)
    for name in ("pair_ids", "fragment_a_tokens", "fragment_b_tokens", "labels",
                 "translation_a_to_b_rc", "translation_valid"):
        if not np.array_equal(getattr(e1.batch, name), getattr(clean, name)):
            raise ValueError("E2 clean/student identity or original GT differs: " + name)
    maps = []
    for side in "ab":
        values = np.full((len(samples), contour_cap), -1, dtype=np.int64)
        for i, sample in enumerate(samples):
            ancestor = getattr(sample, "ancestor_" + side)
            new_valid = getattr(sample.student, "contour_valid_" + side)
            old_valid = getattr(sample.clean, "contour_valid_" + side)
            if ancestor.dtype != np.int64 or ancestor.shape != new_valid.shape:
                raise ValueError("E2 ancestry must be int64 matching student tokens")
            trusted = ancestor >= 0
            if (np.any(ancestor < -1) or np.any(ancestor[trusted] >= len(old_valid))
                    or np.any(trusted & ~new_valid)):
                raise ValueError("E2 ancestry references invalid token")
            if np.any(~old_valid[ancestor[trusted]]) or len(np.unique(ancestor[trusted])) != int(trusted.sum()):
                raise ValueError("E2 ancestry must be injective on valid clean tokens")
            values[i, :len(ancestor)] = ancestor
        maps.append(values)
    return E2Batch(e1.batch, e1.reports, e1.pose_supervision_enabled, clean, *maps)


def make_e2_loader(dataset, indices, *, batch_size, num_workers, seed, contour_cap):
    loader = runner._loader(dataset, indices, batch_size=batch_size,
                            num_workers=num_workers, seed=seed)
    loader.collate_fn = partial(collate_e2_pairs, contour_cap=contour_cap)
    return loader


def clean_teacher_forward(model, inputs):
    """Stop-gradient teacher, preserving even mixed per-submodule mode flags."""
    modes = tuple((module, module.training) for module in model.modules())
    try:
        model.eval()
        with torch.no_grad():
            output = model(*inputs)
            # Retain only loss evidence, not the teacher's token feature tensors.
            result = {name: getattr(output, name).detach() for name in
                ("assignment", "unmatched_a", "unmatched_b", "fused_logit", "coarse_logit",
                 "local_logit", "training_valid")}
            result["coarse"] = SimpleNamespace(valid_problem=output.coarse.valid_problem.detach())
            return SimpleNamespace(**result)
    finally:
        for module, mode in modes:
            module.training = mode


def _mean_or_zero(values, zero):
    return torch.stack(values).mean() if values else zero


def aligned_assignment_kl(student, teacher, ancestor_a, ancestor_b, *, epsilon=1e-8):
    """One pair: bidirectional categorical KL on injective source-aligned tokens.

    In each direction an explicit OTHER bucket retains all probability mass on
    unrepresented tokens plus the dustbin. No missing source token becomes a
    false real match or a new hard dustbin label. The teacher is always detached.
    """
    ids_a, ids_b = torch.where(ancestor_a >= 0)[0], torch.where(ancestor_b >= 0)[0]
    zero = student.assignment.sum() * 0.
    if not ids_a.numel() or not ids_b.numel():
        return zero, 0
    old_a, old_b = ancestor_a[ids_a], ancestor_b[ids_b]
    losses = []
    for rows, cols, old_rows, old_cols, matrix, teach, unmatched, teach_unmatched in (
        (ids_a, ids_b, old_a, old_b, student.assignment, teacher.assignment,
         student.unmatched_a, teacher.unmatched_a),
        (ids_b, ids_a, old_b, old_a, student.assignment.T, teacher.assignment.T,
         student.unmatched_b, teacher.unmatched_b),
    ):
        selected = matrix[rows][:, cols]
        selected_teacher = teach.detach()[old_rows][:, old_cols]
        other = (matrix[rows].sum(1) + unmatched[rows] - selected.sum(1)).clamp_min(0.)
        teacher_other = (teach.detach()[old_rows].sum(1) + teach_unmatched.detach()[old_rows]
                         - selected_teacher.sum(1)).clamp_min(0.)
        p = torch.cat((selected, other[:, None]), 1).clamp_min(epsilon)
        q = torch.cat((selected_teacher, teacher_other[:, None]), 1).clamp_min(epsilon)
        p, q = p / p.sum(1, keepdim=True), q / q.sum(1, keepdim=True)
        losses.append((q * (q.log() - p.log())).sum(1).mean().clamp_min(0.))
    return torch.stack(losses).mean(), int(ids_a.numel() + ids_b.numel())


def source_corrected_gap_loss(assignment, target_a, target_b, ancestor_a, ancestor_b,
                              student_a, student_b, clean_a, clean_b, translation_gt,
                              *, config=E2LossConfig()):
    """One changed positive: conditional transport cost on inherited seam anchors.

    All trusted candidate columns/rows compete, not just known correct edges.
    Conditioning on real-match mass avoids simply shrinking a row's match mass
    to reduce the loss (the unchanged inherited NLL also supervises that mass).
    Correct correspondence tolerates augmentation-induced gap exactly, plus a
    3px dead zone for pre-existing clean contour discretization. This is NOT a
    direct regression target for raw eroded boundary translation_hat_rc.
    """
    ia, ib = torch.where(ancestor_a >= 0)[0], torch.where(ancestor_b >= 0)[0]
    zero = assignment.sum() * 0.
    if not ia.numel() or not ib.numel():
        return zero, dict(anchor_count=0, raw_gap_sum_px=0., corrected_gap_sum_px=0., correspondence_count=0)
    source_a, source_b = clean_a[ancestor_a[ia]].detach(), clean_b[ancestor_b[ib]].detach()
    new_a, new_b = student_a[ia].detach(), student_b[ib].detach()
    raw_delta = new_b[None] - new_a[:, None]
    offset_delta = (new_b - source_b)[None] - (new_a - source_a)[:, None]
    residual = torch.linalg.vector_norm(raw_delta - offset_delta - translation_gt.detach(), dim=-1)
    excess = (residual - config.gap_tolerance_px).clamp_min(0.) / config.gap_scale_px
    cost = F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction="none")
    probability = assignment[ia][:, ib]
    row_anchor, col_anchor = target_a[ia] >= 0, target_b[ib] >= 0
    values = []
    for prob, costs, anchor in ((probability, cost, row_anchor), (probability.T, cost.T, col_anchor)):
        if anchor.any().item():
            expected = (prob * costs).sum(1) / prob.sum(1).clamp_min(config.epsilon)
            values.append(expected[anchor].mean())
    # Observed positive-edge diagnostics, separate from the candidate objective.
    rows = torch.where(target_a >= 0)[0]
    cols = target_a[rows]
    trusted = (ancestor_a[rows] >= 0) & (ancestor_b[cols] >= 0)
    rows, cols = rows[trusted], cols[trusted]
    raw = student_b[cols] - student_a[rows] - translation_gt
    corrected = clean_b[ancestor_b[cols]] - clean_a[ancestor_a[rows]] - translation_gt
    return _mean_or_zero(values, zero), dict(
        anchor_count=int(row_anchor.sum().item() + col_anchor.sum().item()),
        raw_gap_sum_px=float(torch.linalg.vector_norm(raw.detach(), dim=1).sum().cpu()),
        corrected_gap_sum_px=float(torch.linalg.vector_norm(corrected.detach(), dim=1).sum().cpu()),
        correspondence_count=int(rows.numel()))


@dataclass(frozen=True)
class E2Loss:
    total: torch.Tensor
    base: object
    pair_consistency: torch.Tensor
    assignment_consistency: torch.Tensor
    gap_transport: torch.Tensor
    diagnostics: dict


def compute_e2_loss(output, teacher, labels, target_a, target_b, translation_target_rc,
                    translation_valid, *, pose_supervision_enabled, changed_pair,
                    ancestor_a, ancestor_b, student_points_a, student_points_b,
                    clean_points_a, clean_points_b, config=RachelN512LossConfig(),
                    e2_config=E2LossConfig()):
    if changed_pair.dtype != torch.bool or changed_pair.shape != labels.shape:
        raise ValueError("changed_pair must be bool [B]")
    if not torch.equal(pose_supervision_enabled, (labels == 1) & ~changed_pair):
        raise ValueError("E2 must preserve E1 clean-only raw pose supervision")
    base = compute_weathering_loss(output, labels, target_a, target_b, translation_target_rc,
        translation_valid, pose_supervision_enabled=pose_supervision_enabled, config=config)
    zero = output.assignment.sum() * 0.
    pair_losses, assignment_losses, gap_losses = [], [], []
    diag = dict(changed_pair_count=int(changed_pair.sum().item()), teacher_pair_count=0,
        pair_consistency_pair_count=0, assignment_pair_count=0, aligned_token_count=0,
        gap_pair_count=0, gap_anchor_count=0, gap_correspondence_count=0,
        raw_gap_sum_px=0., corrected_gap_sum_px=0.)
    selected = torch.where(changed_pair & output.training_valid)[0].tolist()
    if selected and teacher is None:
        raise ValueError("changed valid pairs require an online clean teacher")
    for i in selected:
        if not teacher.training_valid[i].item():
            continue
        diag["teacher_pair_count"] += 1
        logits = ["fused_logit", "local_logit"]
        if output.coarse.valid_problem[i].item() and teacher.coarse.valid_problem[i].item():
            logits.append("coarse_logit")
        pair_losses.append(torch.stack([(torch.sigmoid(getattr(output, name)[i]) -
            torch.sigmoid(getattr(teacher, name)[i].detach())).square() for name in logits]).mean())
        diag["pair_consistency_pair_count"] += 1
        current = SimpleNamespace(**{key: getattr(output, key)[i] for key in
            ("assignment", "unmatched_a", "unmatched_b")})
        target = SimpleNamespace(**{key: getattr(teacher, key)[i].detach() for key in
            ("assignment", "unmatched_a", "unmatched_b")})
        kl, token_count = aligned_assignment_kl(current, target, ancestor_a[i], ancestor_b[i],
                                               epsilon=e2_config.epsilon)
        if token_count:
            assignment_losses.append(kl)
            diag["assignment_pair_count"] += 1
            diag["aligned_token_count"] += token_count
        if translation_valid[i].item():
            gap, detail = source_corrected_gap_loss(output.assignment[i], target_a[i], target_b[i],
                ancestor_a[i], ancestor_b[i], student_points_a[i], student_points_b[i],
                clean_points_a[i], clean_points_b[i], translation_target_rc[i], config=e2_config)
            if detail["anchor_count"]:
                gap_losses.append(gap)
                diag["gap_pair_count"] += 1
                diag["gap_anchor_count"] += detail["anchor_count"]
                diag["gap_correspondence_count"] += detail["correspondence_count"]
                diag["raw_gap_sum_px"] += detail["raw_gap_sum_px"]
                diag["corrected_gap_sum_px"] += detail["corrected_gap_sum_px"]
    pair = _mean_or_zero(pair_losses, zero)
    assignment = _mean_or_zero(assignment_losses, zero)
    gap = _mean_or_zero(gap_losses, zero)
    # Do not introduce even a rounding addition on clean-only batches.
    total = base.total if not selected else (base.total + e2_config.pair_consistency_weight * pair
        + e2_config.assignment_consistency_weight * assignment + e2_config.gap_weight * gap)
    if not torch.isfinite(total).item():
        raise FloatingPointError("E2 loss is nonfinite")
    return E2Loss(total, base, pair, assignment, gap, diag)


def train_e2_epoch(model, loader, optimizer, loss_config, device, args, epoch, variant):
    """E1-equivalent optimizer/exposure schedule with one clean no-grad teacher."""
    e2_config = getattr(args, "e2_loss_config", E2LossConfig())
    if isinstance(e2_config, dict):
        e2_config = E2LossConfig(**e2_config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_samples, total_loss, updates = 0, 0., 0
    started = time.perf_counter()
    accumulation = args.effective_batch_size // args.batch_size
    if accumulation < 1 or args.effective_batch_size % args.batch_size:
        raise ValueError("effective batch must be an integer multiple of the microbatch")
    statistics = WeatheringStatistics()
    diagnostic_totals, loss_totals = {}, dict(pair_consistency=0., assignment_consistency=0., gap_transport=0.)
    output_dir = Path(args.output) if getattr(args, "output", None) else None
    initial_status = {}
    if output_dir is not None and (output_dir / "status.json").exists():
        with (output_dir / "status.json").open() as stream:
            initial_status = json.load(stream)

    def extras():
        return dict(schema_version=SCHEMA, losses={key: value / max(1, total_samples)
            for key, value in loss_totals.items()}, counts=diagnostic_totals,
            mean_raw_inherited_gap_px=diagnostic_totals.get("raw_gap_sum_px", 0.) /
                max(1, diagnostic_totals.get("gap_correspondence_count", 0)),
            mean_source_corrected_inherited_gap_px=diagnostic_totals.get("corrected_gap_sum_px", 0.) /
                max(1, diagnostic_totals.get("gap_correspondence_count", 0)),
            teacher="online same-network eval/no_grad, no EMA", train_only_source_sidecar=True)

    for step, wrapped in enumerate(loader):
        group_start = (step // accumulation) * accumulation
        group_samples = min(args.effective_batch_size, len(loader.dataset) - group_start * args.batch_size)
        inputs, targets = runner._full_batch(wrapped.batch, device)
        changed = torch.as_tensor([r["changed_pair"] for r in wrapped.reports], dtype=torch.bool, device=device)
        teacher = None
        if changed.any().item():
            clean_inputs, _ = runner._full_batch(wrapped.clean_batch, device)
            teacher = clean_teacher_forward(model, clean_inputs)
            del clean_inputs
        output = model(*inputs)
        sidecar = {"ancestor_" + side: runner._tensor(getattr(wrapped, "ancestor_" + side), device, torch.long)
                   for side in "ab"}
        sidecar.update({"clean_points_" + side: runner._tensor(
            getattr(wrapped.clean_batch, "points_rc_" + side), device, torch.float32) for side in "ab"})
        loss = compute_e2_loss(output, teacher, *targets,
            pose_supervision_enabled=torch.as_tensor(wrapped.pose_supervision_enabled, dtype=torch.bool, device=device),
            changed_pair=changed, student_points_a=inputs[2], student_points_b=inputs[3],
            config=loss_config, e2_config=e2_config, **sidecar)
        count = len(wrapped.batch.pair_ids)
        (loss.total * (count / group_samples)).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        total_samples += count
        total_loss += float(loss.total.detach().cpu()) * count
        statistics.add(wrapped)
        for key in loss_totals:
            loss_totals[key] += float(getattr(loss, key).detach().cpu()) * count
        for key, value in loss.diagnostics.items():
            diagnostic_totals[key] = diagnostic_totals.get(key, 0) + value
        if (step + 1) % args.log_every == 0:
            progress = dict(event="train_progress", variant=variant, epoch=epoch,
                samples=total_samples, updates=updates, mean_loss=total_loss / total_samples,
                seconds=time.perf_counter() - started, weathering=statistics.report(), e2=extras())
            print(json.dumps(progress, ensure_ascii=False, allow_nan=False), flush=True)
            if output_dir is not None:
                runner._atomic_json(output_dir / "status.json", dict(status="running", phase="train",
                    epoch=epoch, global_exposure=initial_status.get("global_exposure", 0) + total_samples,
                    optimizer_updates=initial_status.get("optimizer_updates", 0) + updates, progress=progress))
    return dict(samples=total_samples, optimizer_updates=updates, mean_loss=total_loss / max(1, total_samples),
        mean_seam_loss=0., seconds=time.perf_counter() - started,
        peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else 0,
        weathering=statistics.report(), e2=extras())


__all__ = ["E2LossConfig", "E2Dataset", "E2Batch", "E2Sample", "collate_e2_pairs", "make_e2_loader",
           "clean_teacher_forward", "aligned_assignment_kl", "source_corrected_gap_loss", "E2Loss",
           "compute_e2_loss", "train_e2_epoch"]
