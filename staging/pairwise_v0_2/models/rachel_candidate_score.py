"""Experimental, target-blind candidate-conditioned pair classification.

This is a READOUT ablation: the underlying matcher, primal/dual projections,
Sinkhorn, translation outputs, coarse score, and four-input fusion formula are
unchanged. Canonical Top2/mode-consensus is used to *observe* candidate seams,
never to replace the downstream layout decoder. Proposal zero is its exact
canonical result. Other proposals reuse the existing separated-mode generator
with normal/overlap reranking disabled.

For each proposed translation, sparse Top2 edges are projected onto both
ordered contours. A shared circular sequence CNN sees candidate-consistent
confidence, affinity, residual, independent arc cells, correspondence order,
and gaps. Thus this is not another MLP on the former 8/24 global statistics.
The two side encodings are combined symmetrically. R predicts candidate-pose
correctness; P explicitly reads [encoding, R logit]. P is pooled to the local
pair score, without a hard R veto. candidate_pair and candidate_dual have the
same graph; only the extra R supervision differs. A positive pair with all
wrong proposals remains a positive pair in the unchanged pair BCE.

Discrete proposals/topology are detached. Gathered Q and affinity retain
gradients, allowing joint training. No target is accepted by model.forward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .physical_translation_layout import (
    PhysicalTranslationConfig, estimate_physical_translation_layout,
)
from .rachel_n512 import RachelN512Config, RachelN512Output, RachelN512Pairwise


@dataclass(frozen=True)
class CandidateScoreConfig:
    hidden_dim: int = 32
    max_additional_modes: int = 6
    # Registered canonical proposal constants, not layout changes.
    inlier_radius_px: float = 10.0
    mode_separation_px: float = 20.0
    max_correspondences: int = 512
    # This bounds an observed ordered link, not acceptance of a seam/pair.
    link_gap_scale_px: float = 64.0

    def __post_init__(self):
        for name in ("hidden_dim", "max_correspondences"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if type(self.max_additional_modes) is not int or self.max_additional_modes < 0:
            raise ValueError("max_additional_modes must be a nonnegative integer")
        for name in ("inlier_radius_px", "mode_separation_px", "link_gap_scale_px"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")


@dataclass(frozen=True)
class RachelCandidateOutput(RachelN512Output):
    score_details: Dict[str, Tensor]


def _arc_geometry(points: np.ndarray):
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    cells = .5 * (lengths + np.roll(lengths, 1))
    arc = np.r_[0., np.cumsum(lengths[:-1])]
    perimeter = max(float(lengths.sum()), 1e-6)
    area = np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                  - points[:, 1] * np.roll(points[:, 0], -1))
    return cells, arc, perimeter, (1 if area >= 0 else -1)


def _ordered_topology(source, target, edges, hard, edge_strength, gap_scale):
    """Independent vertex cells and gap-tolerant, winding-aware ordered links.

    Missing vertices do not break a link. It connects consecutive SUPPORTED
    vertices, reports their physical gap and ordering, and never labels a
    whole pair negative. Topology is discrete and intentionally target-blind.
    """
    cells, arc, perimeter, winding = _arc_geometry(source)
    _, target_arc, target_perimeter, target_winding = _arc_geometry(target)
    count = len(source)
    occupied = np.zeros(count, dtype=bool)
    best_target = np.zeros(count, dtype=np.int64)
    for i in np.unique(edges[hard, 0]):
        available = np.flatnonzero(hard & (edges[:, 0] == i))
        best = available[np.argmax(edge_strength[available])]
        occupied[i], best_target[i] = True, edges[best, 1]
    selected = np.flatnonzero(occupied)
    order, gap, stretch = (np.zeros(count, dtype=np.float32) for _ in range(3))
    if len(selected) > 1:
        nxt = np.roll(selected, -1)
        source_step = (arc[nxt] - arc[selected]) % perimeter
        signed_target_step = target_arc[best_target[nxt]] - target_arc[best_target[selected]]
        signed_target_step = ((signed_target_step + .5 * target_perimeter)
                              % target_perimeter) - .5 * target_perimeter
        expected_sign = -winding * target_winding
        target_step = np.abs(signed_target_step)
        link = (source_step <= gap_scale) & (target_step <= gap_scale)
        order[selected] = link * (np.sign(signed_target_step) == expected_sign)
        gap[selected] = np.minimum(source_step / gap_scale, 1.)
        stretch[selected] = link * np.exp(-np.abs(source_step - target_step) / gap_scale)
    # Nearby unsupported vertices retain physical distance-to-seam information.
    distance = np.ones(count, dtype=np.float32)
    if len(selected):
        delta = np.abs(arc[:, None] - arc[selected][None])
        distance = np.minimum(np.minimum(delta, perimeter - delta).min(1) / gap_scale, 1.)
    topology = np.stack((occupied, cells / perimeter, order, gap, stretch, distance), 1)
    return topology.astype(np.float32), cells, occupied


class OrderedCandidateSeamHead(nn.Module):
    """Shared two-sided contour-sequence encoder plus candidate P and R heads."""

    sequence_channels = 11

    def __init__(self, config: CandidateScoreConfig = CandidateScoreConfig()):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.sequence = nn.Sequential(
            nn.Conv1d(self.sequence_channels, h, 5, padding=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(h, h, 5, padding=4, dilation=2, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(h, h, 5, padding=8, dilation=4, padding_mode="circular"),
            nn.SiLU(),
        )
        self.candidate_encoder = nn.Sequential(nn.Linear(4 * h + 6, h), nn.SiLU())
        self.correctness_head = nn.Linear(h, 1)
        # R has an explicit differentiable path into pair classification.
        self.pair_head = nn.Sequential(nn.Linear(h + 1, h), nn.SiLU(), nn.Linear(h, 1))
        self.no_candidate_logit = nn.Parameter(torch.tensor(0.))

    def _side(self, source, target, edges, q, affinity, distance, hard, topology_strength):
        n = len(source)
        device, dtype = q.device, q.dtype
        edge_tensor = torch.as_tensor(edges[:, 0], device=device, dtype=torch.long)
        gate = torch.exp(-.5 * (distance / self.config.inlier_radius_px).square())
        weight = q * gate

        def scatter(value):
            return torch.zeros(n, device=device, dtype=dtype).index_add(0, edge_tensor, value)

        mass = scatter(weight)
        total = scatter(q).clamp_min(1e-8)
        weighted_affinity = scatter(weight * affinity) / mass.clamp_min(1e-8)
        residual = scatter(weight * (distance / self.config.inlier_radius_px).clamp_max(8.)) / mass.clamp_min(1e-8)
        topology, cells, occupied = _ordered_topology(
            source, target, edges, hard, topology_strength, self.config.link_gap_scale_px,
        )
        top = torch.as_tensor(topology, device=device, dtype=dtype)
        # Per-vertex evidence counts a repeated/sliding patch endpoint only once.
        features = torch.cat((torch.stack((mass / total, torch.log1p(mass * n),
                                          weighted_affinity, residual, mass), 1), top), 1)
        sequence = features.transpose(0, 1)[None]
        # Circular padding dilation=4 needs N>=8; repeat short contours without
        # changing their cyclic ordering, then pool only the original period.
        repeats = max(1, math.ceil(9 / n))
        encoded = self.sequence(sequence.repeat(1, 1, repeats))[0, :, :n]
        support_cell = torch.as_tensor(cells * occupied, device=device, dtype=dtype)
        mean = (encoded * support_cell[None]).sum(1) / support_cell.sum().clamp_min(1e-8)
        active = torch.as_tensor(occupied, device=device)
        maximum = encoded[:, active].amax(1) if occupied.any() else encoded.sum(1) * 0.
        coverage = float(np.sum(cells * occupied) / max(float(cells.sum()), 1e-8))
        return torch.cat((mean, maximum)), coverage, float(occupied.mean()), features

    def forward(self, assignment, affinity, points_a, points_b, valid_a, valid_b, mask_a, mask_b):
        """Return local logits and inspectable evidence, without target inputs."""
        cfg = self.config
        batch, k = len(assignment), 1 + cfg.max_additional_modes
        device, dtype = assignment.device, assignment.dtype
        cpu = [x.detach().cpu().numpy() for x in
               (assignment, points_a, points_b, valid_a, valid_b, mask_a, mask_b)]
        q_np, pa_np, pb_np, va_np, vb_np, ma_np, mb_np = cpu
        local, p_batch, r_batch, translations, validity, coverage_batch, order_batch = [], [], [], [], [], [], []
        proposal_config = PhysicalTranslationConfig(
            top_k=2, max_candidates=cfg.max_correspondences,
            inlier_radius_px=cfg.inlier_radius_px, min_inliers=3,
            refinement_iterations=3, max_additional_modes=cfg.max_additional_modes,
            mode_separation_px=cfg.mode_separation_px, use_normals=False, use_overlap=False,
        )
        for b in range(batch):
            proposed = estimate_physical_translation_layout(
                pa_np[b], pb_np[b], q_np[b], va_np[b], vb_np[b],
                mask_a=ma_np[b], mask_b=mb_np[b], config=proposal_config,
            )
            original_edges = proposed.candidate_indices
            ia, ib = np.flatnonzero(va_np[b]), np.flatnonzero(vb_np[b])
            a, bb = pa_np[b, ia], pb_np[b, ib]
            remap_a, remap_b = np.zeros(len(pa_np[b]), int), np.zeros(len(pb_np[b]), int)
            remap_a[ia], remap_b[ib] = np.arange(len(ia)), np.arange(len(ib))
            edges = np.column_stack((remap_a[original_edges[:, 0]], remap_b[original_edges[:, 1]]))
            idx = torch.as_tensor(original_edges, device=device, dtype=torch.long)
            q = assignment[b, idx[:, 0], idx[:, 1]]
            aff = affinity[b, idx[:, 0], idx[:, 1]]
            # Geometry/topology is fixed input, not a learned parameter. Reuse
            # the one CPU snapshot rather than synchronize GPU once per mode.
            delta_np = bb[edges[:, 1]] - a[edges[:, 0]]
            edge_q_np = q_np[b, original_edges[:, 0], original_edges[:, 1]]
            records, support = [], []
            t = torch.full((k, 2), float("nan"), device=device, dtype=dtype)
            ok = torch.zeros(k, dtype=torch.bool, device=device)
            for mode in proposed.mode_diagnostics:
                j = mode["mode_index"]
                t[j] = torch.as_tensor(mode["translation_rc"], device=device, dtype=dtype)
                ok[j] = mode["valid"]
                if not mode["valid"]:
                    records.append(None)
                    support.append(q.sum() * 0.)
                    continue
                distance_np = np.linalg.norm(delta_np - np.asarray(mode["translation_rc"]), axis=1)
                distance = torch.as_tensor(distance_np, device=device, dtype=dtype)
                hard = distance_np <= cfg.inlier_radius_px
                topology_strength = edge_q_np * np.exp(-.5 * (distance_np / cfg.inlier_radius_px) ** 2)
                side_a, arc_a, cover_a, f_a = self._side(a, bb, edges, q, aff, distance, hard, topology_strength)
                side_b, arc_b, cover_b, f_b = self._side(bb, a, edges[:, ::-1].copy(), q, aff, distance, hard, topology_strength)
                encoding = torch.cat((.5 * (side_a + side_b), torch.abs(side_a - side_b)))
                mass = (q * torch.as_tensor(hard, device=device, dtype=dtype)).sum()
                order_mean = .5 * (f_a[:, 7].mean() + f_b[:, 7].mean())
                records.append((encoding, arc_a, arc_b, cover_a, cover_b, order_mean))
                support.append(mass)
            p_values, r_values, covers, orders = [], [], [], []
            for j in range(k):
                if j >= len(records) or records[j] is None:
                    zero = q.sum() * 0. + self.no_candidate_logit * 0.
                    p_values.append(zero); r_values.append(zero)
                    covers.append(zero); orders.append(zero)
                    continue
                enc, aa, ab, ca, cb, order_mean = records[j]
                competitors = [value for z, value in enumerate(support) if z != j]
                runner = torch.stack(competitors).max() if competitors else support[j] * 0.
                summaries = torch.stack((support[j] / q.sum().clamp_min(1e-8),
                                         enc.new_tensor(min(ca, cb)), enc.new_tensor(max(ca, cb)),
                                         enc.new_tensor(min(aa, ab)), enc.new_tensor(max(aa, ab)),
                                         (runner / support[j].clamp_min(1e-8)).clamp_max(10.)))
                hidden = self.candidate_encoder(torch.cat((enc, summaries)))
                r = self.correctness_head(hidden).squeeze(-1)
                p = self.pair_head(torch.cat((hidden, r[None]))).squeeze(-1)
                p_values.append(p); r_values.append(r)
                covers.append(enc.new_tensor(min(aa, ab))); orders.append(order_mean)
            p, r = torch.stack(p_values), torch.stack(r_values)
            # No hard R cutoff and no count inflation from extra candidate modes.
            value = torch.logsumexp(p[ok], 0) - ok.sum().to(dtype).log() if ok.any() else self.no_candidate_logit
            local.append(value); p_batch.append(p); r_batch.append(r)
            translations.append(t); validity.append(ok)
            coverage_batch.append(torch.stack(covers)); order_batch.append(torch.stack(orders))
        return torch.stack(local), {
            "candidate_pair_logits": torch.stack(p_batch),
            "candidate_logits": torch.stack(r_batch),
            "translations_rc": torch.stack(translations),
            "valid": torch.stack(validity),
            "independent_arc_coverage": torch.stack(coverage_batch),
            "ordered_link_evidence": torch.stack(order_batch),
        }


class RachelCandidateScore(nn.Module):
    def __init__(self, base_model: RachelN512Pairwise,
                 candidate_config: CandidateScoreConfig = CandidateScoreConfig(),
                 architecture: str = "candidate_pair"):
        super().__init__()
        if architecture not in ("candidate_pair", "candidate_dual"):
            raise ValueError("unknown candidate architecture")
        self.base_model, self.candidate_config = base_model, candidate_config
        self.config, self.architecture = base_model.config, architecture
        self.score_head = OrderedCandidateSeamHead(candidate_config)
        self.training_mode = "joint"
        self.set_training_mode("joint")

    def set_training_mode(self, mode: str):
        """No hidden schedule: head_only needs an externally supplied matcher."""
        if mode not in ("joint", "head_only"):
            raise ValueError("training mode must be joint or head_only")
        self.training_mode = mode
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(mode == "joint")
        # The old readout is replaced, so it must never consume optimizer state.
        for parameter in self.base_model.local_head.parameters():
            parameter.requires_grad_(False)
        for parameter in self.base_model.fusion.parameters():
            parameter.requires_grad_(True)
        for parameter in self.score_head.parameters():
            parameter.requires_grad_(True)
        self.train(self.training)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.training_mode == "head_only":
            # Freezing includes BN buffers/dropout state, not just parameters.
            self.base_model.eval()
            self.base_model.fusion.train(mode)
        self.base_model.local_head.eval()
        return self

    def forward(self, mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b):
        original = self.base_model(mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b)
        local, details = self.score_head(original.assignment, original.affinity,
                                         points_rc_a, points_rc_b, contour_valid_a,
                                         contour_valid_b, mask_a, mask_b)
        coarse = original.coarse_logit
        fused = self.base_model.fusion(torch.stack((coarse, local, coarse * local,
                                                    torch.abs(coarse - local)), 1)).squeeze(1)
        finite = original.training_valid & torch.isfinite(local) & torch.isfinite(fused)
        local = torch.where(finite, local, torch.zeros_like(local))
        fused = torch.where(finite, fused, torch.zeros_like(fused))
        values = {field.name: getattr(original, field.name) for field in fields(RachelN512Output)}
        values.update(local_logit=local, local_probability=local.sigmoid(),
                      fused_logit=fused, fused_probability=fused.sigmoid(),
                      training_valid=finite, decision_valid=finite & original.transport.diagnostics.converged)
        return RachelCandidateOutput(**values, score_details=details)

    def metadata(self):
        return {"architecture": self.architecture, "model_config": asdict(self.config),
                "candidate_config": asdict(self.candidate_config), "training_mode": self.training_mode}


def build_score_model(base_config: Union[RachelN512Config, dict], architecture="original",
                      candidate_config: Optional[Union[CandidateScoreConfig, dict]] = None):
    if isinstance(base_config, dict):
        base_config = RachelN512Config(**base_config)
    if architecture not in ("original", "candidate_pair", "candidate_dual"):
        raise ValueError("unknown score architecture")
    base = RachelN512Pairwise(base_config)
    if architecture == "original":
        return base
    if isinstance(candidate_config, dict):
        candidate_config = CandidateScoreConfig(**candidate_config)
    return RachelCandidateScore(base, candidate_config or CandidateScoreConfig(), architecture)


def candidate_correctness_loss(output: RachelCandidateOutput, labels: Tensor,
                               translation_target_rc: Tensor, translation_valid: Tensor,
                               tolerance_px: float = 20.0):
    """R supervision only; call in dual arm in addition to unchanged old loss.

    GT is used HERE, after target-blind proposals, never to generate proposals.
    A negative pair has all R=0. A positive pair with a wrong candidate also
    has R=0 but retains its original positive pair BCE. Missing positive GT is
    ignored for R. ``translation_valid`` means available GT for R, NOT the
    legacy auxiliary shift-loss mask: weathered positives still have valid GT.
    Mean per pair prevents candidate-count weighting changes.
    """
    if not math.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("tolerance_px must be finite and positive")
    details = output.score_details
    logits, valid = details["candidate_logits"], details["valid"]
    if labels.shape != logits.shape[:1] or translation_target_rc.shape != (len(labels), 2):
        raise ValueError("candidate supervision shapes differ")
    positive = labels.to(torch.bool)
    gt_finite = torch.isfinite(translation_target_rc).all(1) & translation_valid
    supervise = valid & output.training_valid[:, None] & (~positive | gt_finite)[:, None]
    error = (details["translations_rc"].detach() - translation_target_rc[:, None]).norm(dim=2)
    target = positive[:, None] & gt_finite[:, None] & (error <= tolerance_px)
    bce = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none")
    counts = supervise.sum(1)
    per_pair = (bce * supervise).sum(1) / counts.clamp_min(1)
    loss = per_pair.sum() / (counts > 0).sum().clamp_min(1)
    return loss, {"candidate_target": target.detach(), "candidate_supervised": supervise.detach(),
                  "candidate_error_px": error.detach(), "candidate_pair_count": (counts > 0).sum().detach(),
                  "candidate_positive_count": (target & supervise).sum().detach()}


__all__ = ["CandidateScoreConfig", "RachelCandidateOutput", "RachelCandidateScore",
           "OrderedCandidateSeamHead", "build_score_model", "candidate_correctness_loss"]
