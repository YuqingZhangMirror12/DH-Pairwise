"""Small, target-blind E3 evidence head; never modifies the fixed Top2 pose.

Geometry features use only the predicted transport, ordered contours and filled
input masks. Independent arc coverage counts each contour vertex's arc cell at
most once; it is not coverage of an annotated true seam. Distances are in the
800-pixel model coordinate system, not physical scan units. The score-only
control has exactly the same MLP, initialization and input shape, but its
standardized geometry channels are zeroed. Neither head has a hard veto.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch
from torch import nn

from .translation_layout import TranslationLayoutConfig, estimate_translation_layout
from .physical_translation_layout import outward_contour_normals, translated_intersection_area


TOP2_CONFIG = TranslationLayoutConfig(correspondence_mode="topk_union", top_k=2,
    max_candidates=512, min_inliers=3, inlier_radius_px=10.0)
GEOMETRY_NAMES = (
    "layout_valid", "candidate_fraction", "inlier_fraction", "weighted_inlier_fraction",
    "log1p_support_weight", "runner_up_support_ratio", "residual_over_10px",
    "independent_support_fraction", "unique_arc_coverage_min", "unique_arc_coverage_max",
    "longest_supported_arc_min", "longest_supported_arc_max",
    "inlier_gap_le_3px", "inlier_gap_le_5px",
    "near_contour_3px_arc_min", "near_contour_3px_arc_max",
    "near_contour_5px_arc_min", "near_contour_5px_arc_max",
    "normal_available_fraction", "normal_complementarity",
    "filled_overlap_ratio", "deep_overlap_2px_ratio",
)
FEATURE_NAMES = ("coarse_logit", "local_logit") + GEOMETRY_NAMES


def arc_evidence(points, valid, supported_indices):
    """Independent Voronoi arc-cell coverage and longest cyclic supported run.

    No matching edge is counted twice. A run only connects adjacent sampled
    valid contour vertices; unsupported vertices break it, including across
    the contour's cyclic start. Returned fractions use the full perimeter.
    """
    indices = np.flatnonzero(valid)
    if len(indices) < 3:
        return 0.0, 0.0
    p = np.asarray(points, float)[indices]
    lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    cell = (lengths + np.roll(lengths, 1)) * .5
    perimeter = float(cell.sum())
    if not np.isfinite(perimeter) or perimeter <= 1e-10:
        return 0.0, 0.0
    selected = np.isin(indices, np.unique(supported_indices))
    coverage = float(cell[selected].sum() / perimeter)
    if selected.all():
        return coverage, 1.0
    # Start immediately after a known unsupported vertex, so cyclic runs are
    # counted once without joining across any unsupported gap.
    start = (int(np.flatnonzero(~selected)[0]) + 1) % len(indices)
    best = current = 0.0
    for offset in range(len(indices)):
        i = (start + offset) % len(indices)
        current = current + cell[i] if selected[i] else 0.0
        best = max(best, current)
    return coverage, float(best / perimeter)


def geometry_evidence(points_a, points_b, assignment, valid_a, valid_b, mask_a, mask_b,
                      *, estimate=None):
    """Return (feature vector, unchanged Top2 estimate), with no target inputs.

    Invalid geometry is represented by all-zero channels (including the
    explicit layout_valid flag), not an invented pose or classification veto.
    Optional estimate permits one decoder call per pair in a deployment flow.
    """
    a, b = np.asarray(points_a, float), np.asarray(points_b, float)
    va, vb = np.asarray(valid_a, bool), np.asarray(valid_b, bool)
    matrix = np.asarray(assignment, float)
    result = estimate if estimate is not None else estimate_translation_layout(
        a, b, matrix, va, vb, config=TOP2_CONFIG)
    features = np.zeros(len(GEOMETRY_NAMES), dtype=np.float32)
    if not result.valid:
        return features, result
    selected = result.candidate_indices[result.inlier_mask]
    if not len(selected):
        raise ValueError("valid Top2 must have correspondence inliers")
    weight = matrix[selected[:, 0], selected[:, 1]]
    support = float(weight.sum())
    if support <= 0 or not np.isfinite(support):
        raise ValueError("valid Top2 must have finite positive support")
    norm_weight = weight / support
    displacement = b[selected[:, 1]] - a[selected[:, 0]]
    gaps = np.linalg.norm(displacement - result.t_a_to_b_rc, axis=1)
    covered_a, longest_a = arc_evidence(a, va, selected[:, 0])
    covered_b, longest_b = arc_evidence(b, vb, selected[:, 1])
    wa, wb = np.zeros(len(a)), np.zeros(len(b))
    np.maximum.at(wa, selected[:, 0], weight)
    np.maximum.at(wb, selected[:, 1], weight)
    # The weaker independently supported endpoint, normalized by edge support.
    independent = min(float(wa.sum()), float(wb.sum())) / support
    ia, ib = np.flatnonzero(va), np.flatnonzero(vb)
    shifted_b = b[ib] - result.t_a_to_b_rc
    da = cKDTree(shifted_b).query(a[ia])[0]
    db = cKDTree(a[ia]).query(shifted_b)[0]
    near = []
    for radius in (3.0, 5.0):
        ca = arc_evidence(a, va, ia[da <= radius])[0]
        cb = arc_evidence(b, vb, ib[db <= radius])[0]
        near.extend((min(ca, cb), max(ca, cb)))
    na, good_a = outward_contour_normals(a, va, arc_half_length_px=8.0)
    nb, good_b = outward_contour_normals(b, vb, arc_half_length_px=8.0)
    good = good_a[selected[:, 0]] & good_b[selected[:, 1]]
    normal_available = float(norm_weight[good].sum())
    complementary = .5  # Neutral; the availability channel distinguishes it.
    if good.any():
        dot = np.sum(na[selected[good, 0]] * nb[selected[good, 1]], axis=1)
        complementary = float(np.sum(norm_weight[good] * (1 - np.clip(dot, -1, 1)) * .5)
                              / normal_available)
    ma, mb = np.asarray(mask_a).squeeze(), np.asarray(mask_b).squeeze()
    if ma.ndim != 2 or mb.ndim != 2 or not all(
            np.isfinite(m).all() and np.all((m == 0) | (m == 1)) for m in (ma, mb)):
        raise ValueError("geometry requires binary filled input masks")
    ma, mb = ma.astype(bool), mb.astype(bool)
    area = min(int(ma.sum()), int(mb.sum()))
    if area <= 0:
        raise ValueError("valid geometry requires nonempty masks")
    r, c = np.ogrid[-2:3, -2:3]
    disk = r * r + c * c <= 4
    ea = ndimage.binary_erosion(ma, structure=disk, border_value=0)
    eb = ndimage.binary_erosion(mb, structure=disk, border_value=0)
    overlap = translated_intersection_area(ma, mb, result.t_a_to_b_rc) / area
    deep = translated_intersection_area(ea, eb, result.t_a_to_b_rc) / area
    features[:] = (1.0, result.candidate_count / TOP2_CONFIG.max_candidates,
        result.inlier_fraction, result.weighted_inlier_fraction,
        np.log1p(result.support_weight), result.runner_up_support_ratio,
        float(result.residual_px) / 10.0, independent,
        min(covered_a, covered_b), max(covered_a, covered_b),
        min(longest_a, longest_b), max(longest_a, longest_b),
        float(norm_weight[gaps <= 3].sum()), float(norm_weight[gaps <= 5].sum()),
        *near, normal_available, complementary, overlap, deep)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite predicted geometry evidence")
    return features, result


@dataclass(frozen=True)
class SeamGeometryHeadConfig:
    hidden_dim: int = 32
    bottleneck_dim: int = 16
    use_geometry: bool = True


class SeamGeometryPairHead(nn.Module):
    """Two scores plus optional geometry; no transport/matcher gradients."""
    def __init__(self, config=SeamGeometryHeadConfig()):
        super().__init__()
        self.config = config
        if config.hidden_dim < 1 or config.bottleneck_dim < 1:
            raise ValueError("hidden dimensions must be positive")
        n = len(FEATURE_NAMES)
        self.register_buffer("feature_mean", torch.zeros(n))
        self.register_buffer("feature_scale", torch.ones(n))
        self.register_buffer("feature_active", torch.ones(n, dtype=torch.bool))
        self.net = nn.Sequential(nn.Linear(n, config.hidden_dim), nn.ReLU(),
            nn.Linear(config.hidden_dim, config.bottleneck_dim), nn.ReLU(),
            nn.Linear(config.bottleneck_dim, 1))

    def set_training_normalization(self, train_features):
        values = torch.as_tensor(train_features, dtype=torch.float32).detach()
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or not torch.isfinite(values).all():
            raise ValueError("TRAIN features must be finite [pairs,features]")
        self.feature_mean.copy_(values.mean(0).to(self.feature_mean))
        std = values.std(0, unbiased=False)
        self.feature_scale.copy_(std.clamp_min(1e-6).to(self.feature_scale))
        # A TRAIN-constant channel has no fitted effect. Do not let unseen
        # domain variation activate an arbitrary untrained first-layer weight.
        self.feature_active.copy_((std >= 1e-6).to(self.feature_active))

    def forward(self, features):
        x = features.detach().to(dtype=self.feature_mean.dtype)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError("head input must follow FEATURE_NAMES")
        x = (x - self.feature_mean) / self.feature_scale
        x = torch.where(self.feature_active, x, torch.zeros_like(x))
        if not self.config.use_geometry:
            x = torch.cat((x[:, :2], torch.zeros_like(x[:, 2:])), dim=1)
        return self.net(x).squeeze(1)

    def metadata(self):
        return dict(config=asdict(self.config), feature_names=list(FEATURE_NAMES),
            parameter_count=sum(p.numel() for p in self.parameters()),
            inactive_first_layer_geometry_weights=(0 if self.config.use_geometry else
                len(GEOMETRY_NAMES) * self.config.hidden_dim),
            normalization_fit_split="train", matcher_frozen=True, layout_modified=False,
            train_inactive_features=[name for name, active in zip(FEATURE_NAMES,
                self.feature_active.detach().cpu().tolist()) if not active],
            hard_gate=False)
