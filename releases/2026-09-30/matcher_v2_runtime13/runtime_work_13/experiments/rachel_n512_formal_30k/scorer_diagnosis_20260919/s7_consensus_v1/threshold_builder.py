"""Native local hypotheses -> bounded-diameter clusters -> exact edge unions.

No GT enters admission. T is a DIAMETER of original fitted hypothesis poses,
not a radius around a moving centroid and not a distance along the contour.
"""
from dataclasses import dataclass, fields, replace
import math
import numpy as np
import torch

from .legacy_pose_consensus import (
    PoseConsensusBuilder as LegacyBuilder, ProposalConfig, PoseCluster,
    ConsensusProposals, _cloud, _pose_key)
from .evidence import material_overlap, observed_arc_cells
from .geometry import pair_frame


@dataclass(frozen=True)
class ThresholdPolicy:
    pose_diameter_px: float = 16.
    candidate_budget: int = 8
    maximum_interpenetration_sum: float = .10

    def __post_init__(self):
        if not math.isfinite(self.pose_diameter_px) or self.pose_diameter_px <= 0:
            raise ValueError('positive finite pose DIAMETER required')
        if self.candidate_budget <= 0 or not 0 <= self.maximum_interpenetration_sum <= 1:
            raise ValueError('invalid candidate budget or physical overlap limit')


@dataclass(frozen=True)
class ThresholdCluster(PoseCluster):
    member_translations_rc: torch.Tensor
    original_union_edge_ids: torch.Tensor
    pose_diameter_px: float
    actual_diameter_px: float
    independent_arc_px: float
    raw_absolute_mass_px: float


def diameter(poses):
    return float(torch.cdist(poses.double(), poses.double()).max()) if len(poses) > 1 else 0.


def bound_common_pose(initial, proposed, centers, threshold):
    """Clip a step to the intersection of T-balls around ALL original poses.

    The segment starts from a feasible joint pose. This does not merge any
    new member, enlarge T, or form a nearest-neighbour chain. All arithmetic
    stays differentiable along the selected (piecewise) constraint.
    """
    centers = centers.to(initial)
    offset = initial[None] - centers
    if bool((offset.detach().norm(dim=1) > threshold + 1e-4).any()):
        raise ValueError('joint-pose constraint needs a feasible initial pose')
    step = proposed - initial
    a = step.square().sum()
    if float(a.detach()) <= 1e-20:
        return proposed
    b = 2 * (offset * step).sum(1)
    c = offset.square().sum(1) - threshold ** 2
    # Nonzero floor avoids sqrt'(0) at a tangent step from an active boundary.
    roots = (-b + (b.square() - 4 * a * c).clamp_min(1e-12).sqrt()) / (2 * a)
    fraction = roots.min().clamp(0., 1.)
    return initial + fraction * step


def canonical(cloud):
    if not len(cloud.ids):
        return cloud
    _, first, inverse = np.unique(cloud.ids.numpy(), axis=0, return_index=True, return_inverse=True)
    updates = {}
    for f in fields(cloud):
        value = getattr(cloud, f.name)
        if isinstance(value, torch.Tensor):
            selected = value[torch.as_tensor(first)]
            if not torch.equal(value, selected[torch.as_tensor(inverse)]):
                raise ValueError('conflicting duplicate edge: ' + f.name)
            updates[f.name] = selected
    return replace(cloud, **updates)


def subset(cloud, indices):
    return replace(cloud, **{f.name: getattr(cloud, f.name)[indices] for f in fields(cloud)
        if isinstance(getattr(cloud, f.name), torch.Tensor)})


class ThresholdPoseBuilder(LegacyBuilder):
    def __init__(self, geometry, proposal=None, policy=None):
        super().__init__(geometry, proposal or ProposalConfig())
        self.policy = policy or ThresholdPolicy()
        self.all_clusters = ()
        self.audit = {}

    def _joint_fit(self, cloud, initial, centers):
        """Fit the WHOLE union, not the best constituent hypothesis.

        Finite normal damage and directional residuals remain unchanged. Raw
        Q*arc is not multiplied by the old pose-admission kernel: a group's
        classification membership cannot disappear during this refit.
        """
        pose = initial
        base = cloud.q * cloud.arc_weight
        scale = max(self._local_scale(cloud), 1.)
        weights = base
        for _ in range(self.config.iterations):
            compatibility = cloud.compatibility(pose, self.geometry)
            error = compatibility.unexplained_residual_rc
            weights = base * torch.rsqrt(1 + (error / scale).square().sum(-1))
            if not bool(weights.sum() > 0):
                break
            step = (weights[:, None] * error).sum(0) / weights.sum()
            pose = bound_common_pose(initial, pose + step, centers, self.policy.pose_diameter_px)
        normal, tangent, reliability = pair_frame(cloud.normal_a, cloud.normal_b,
            cloud.reliability_a, cloud.reliability_b)
        reliable = reliability >= self.geometry.normal_reliability_min
        eye = torch.eye(2, dtype=pose.dtype).expand(len(base), -1, -1)
        matrices = torch.where(reliable[:, None, None], tangent[:, :, None] * tangent[:, None, :], eye)
        information = (weights[:, None, None] * matrices).sum(0) / weights.sum().clamp_min(1e-12)
        # Retain the existing contact convention only on an underconstrained
        # normal; it is not GT recovery and cannot bypass the common-pose bound.
        values, vectors = torch.linalg.eigh(information)
        if float(values[0]) < .05 and bool(reliable.any()):
            axis = vectors[:, 0]
            if float(axis @ (normal * base[:, None]).sum(0)) < 0:
                axis = -axis
            projection = normal @ axis
            valid = reliable & (projection > 1e-8)
            if bool(valid.any()):
                distance = ((cloud.displacement - pose) * normal).sum(1)[valid] / projection[valid]
                shift = distance.min().clamp(0., self.geometry.damage_normal_upper_px)
                pose = bound_common_pose(initial, pose + shift * axis, centers, self.policy.pose_diameter_px)
        return pose, information

    def _combine(self, cloud, hypotheses, members, cells, overlap_fn):
        centers = torch.stack([hypotheses[i].translation for i in members])
        if diameter(centers) > self.policy.pose_diameter_px + 1e-5:
            raise ValueError('attempted transitive/over-diameter merge')
        union = torch.unique(torch.cat([hypotheses[i].edge_ids for i in members]), dim=0)
        lookup = {tuple(x): i for i, x in enumerate(cloud.ids.tolist())}
        try:
            indices = torch.tensor([lookup[tuple(x)] for x in union.tolist()], dtype=torch.long)
        except KeyError as exc:
            raise ValueError('hypothesis names an absent source edge') from exc
        selected = subset(cloud, indices)
        # Duplicate hypotheses do not alter initialization or evidence mass.
        initial = torch.unique(centers, dim=0).mean(0)
        pose, information = self._joint_fit(selected, initial, centers)
        overlap = {} if overlap_fn is None else overlap_fn(pose)
        arc_lengths = []
        for side in (0, 1):
            if cells is not None:
                arc_lengths.append(float(cells[side][union[:, side].unique()].sum()))
            else:
                _, first = np.unique(selected.ids[:, side].numpy(), return_index=True)
                spacing = selected.spacing_a if side == 0 else selected.spacing_b
                arc_lengths.append(float(spacing[first].clamp_max(2 * self.config.observation_radius_px).sum()))
        mass = float((selected.q * selected.arc_weight).sum())
        seed_ids = tuple(sorted({j for i in members for j in hypotheses[i].initial_seed_ids}))
        return ThresholdCluster(pose, union, seed_ids, tuple(members), centers, mass, information,
            bool(torch.linalg.eigvalsh(information).min() < .05), overlap,
            centers, union, self.policy.pose_diameter_px, diameter(centers), min(arc_lengths), mass)

    @torch.no_grad()
    def build_from_hypotheses(self, cloud, hypotheses, seeds=None, overlap_fn=None, cells=None):
        """Also permits replay of frozen native proposals without Matcher work."""
        cloud = canonical(cloud)
        hypotheses = tuple(hypotheses)
        active = [i for i, h in enumerate(hypotheses) if len(h.edge_ids)
                  and bool(torch.isfinite(h.translation).all())]
        clusters = [self._combine(cloud, hypotheses, (i,), cells, overlap_fn) for i in active]
        trace = []
        while True:
            choices = []
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    members = tuple(sorted(clusters[a].merged_hypothesis_ids + clusters[b].merged_hypothesis_ids))
                    centers = torch.stack([hypotheses[i].translation for i in members])
                    span = diameter(centers)
                    if span <= self.policy.pose_diameter_px + 1e-5:
                        # Complete-link distance; never distance between two
                        # already-averaged centers or minimum neighbour links.
                        choices.append((span, _pose_key(centers.mean(0)), members, a, b))
            merged = False
            for span, _, members, a, b in sorted(choices):
                candidate = self._combine(cloud, hypotheses, members, cells, overlap_fn)
                if (candidate.overlap.get('available') and
                        candidate.overlap['fraction_sum_area'] >= self.policy.maximum_interpenetration_sum):
                    continue
                trace.append(dict(hypothesis_ids=list(members), diameter_px=span,
                    union_edge_count=len(candidate.edge_ids), translation_rc=candidate.translation.tolist()))
                clusters = [c for i, c in enumerate(clusters) if i not in (a, b)] + [candidate]
                merged = True
                break
            if not merged:
                break
        rejected = sum(bool(c.overlap.get('available') and
            c.overlap['fraction_sum_area'] >= self.policy.maximum_interpenetration_sum) for c in clusters)
        clusters = [c for c in clusters if not (c.overlap.get('available') and
            c.overlap['fraction_sum_area'] >= self.policy.maximum_interpenetration_sum)]
        clusters.sort(key=lambda c: (-c.raw_absolute_mass_px, -c.independent_arc_px, _pose_key(c.translation)))
        self.all_clusters = tuple(clusters)
        self.audit = dict(policy='complete-link-original-fitted-poses', threshold_is='diameter_not_radius',
            threshold_px=self.policy.pose_diameter_px, native_hypotheses=len(active),
            prebudget=len(clusters), physical_vetoes=rejected, retained=min(len(clusters), self.policy.candidate_budget),
            max_actual_diameter_px=max((c.actual_diameter_px for c in clusters), default=0.),
            gt_used=False, edge_union_not_reweighted_for_classification=True)
        seeds = torch.stack([h.translation for h in hypotheses]) if seeds is None and hypotheses else seeds
        if seeds is None:
            seeds = cloud.displacement.new_empty((0, 2))
        return ConsensusProposals(cloud, seeds, hypotheses,
            tuple(clusters[:self.policy.candidate_budget]), tuple(trace))

    @torch.no_grad()
    def build_from_cloud(self, cloud, seeds=None, overlap_fn=None, cells=None):
        cloud = canonical(cloud)
        cloud = subset(cloud, cloud.q >= self.config.minimum_absolute_q)
        seeds = self._seeds(cloud) if seeds is None else seeds.detach().cpu()
        hypotheses = tuple(self._hypothesis(cloud, t, (i,), t[None], (i,), overlap_fn)
                           for i, t in enumerate(seeds))
        return self.build_from_hypotheses(cloud, hypotheses, seeds, overlap_fn, cells)

    @torch.no_grad()
    def __call__(self, pair):
        cloud = _cloud(pair, self.config)
        if not pair.numeric_valid:
            return ConsensusProposals(cloud, cloud.displacement.new_empty((0, 2)), (), (), ())
        cells = tuple(observed_arc_cells(g, self.config.observation_radius_px)[0].detach().cpu()
                      for g in (pair.ga, pair.gb))
        return self.build_from_cloud(cloud, overlap_fn=lambda t: material_overlap(pair, t), cells=cells)
