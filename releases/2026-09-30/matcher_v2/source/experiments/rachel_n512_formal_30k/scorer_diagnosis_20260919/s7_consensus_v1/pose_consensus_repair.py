"""C04 repair integrated into an isolated source, not yet formally launched.

Same seeds, finite-damage kernel, absolute masses and candidate budget. The
repair concerns merge decisions only: fitted-center bounds, distributional
partner conflicts and a bounded soft-evidence tail instead of ANY-edge veto.
New 5% evidence-tail policies are explicit, not trained or picked on TEST.
"""
from dataclasses import dataclass, fields, replace
import math

import numpy as np
import torch

from .geometry import pair_frame
from .legacy_pose_consensus import (
    ConsensusProposals, PoseCluster, PoseConsensusBuilder, _pose_key)

def edge_membership(cloud_ids, edge_ids):
    stride = int(cloud_ids[:, 1].max()) + 1
    return torch.isin(cloud_ids[:, 0] * stride + cloud_ids[:, 1],
                      edge_ids[:, 0] * stride + edge_ids[:, 1])


@dataclass(frozen=True)
class MergeRepairPolicy:
    maximum_lost_explained_mass_fraction: float = .05
    maximum_conflicting_endpoint_mass_fraction: float = .05

    def __post_init__(self):
        if not (0 <= self.maximum_lost_explained_mass_fraction < .5 and
                0 <= self.maximum_conflicting_endpoint_mass_fraction < .5):
            raise ValueError('merge guardrails must retain a majority of each input evidence distribution')


@dataclass(frozen=True)
class MergedPoseCluster(PoseCluster):
    # Preserve the actual union, including any weak tail now marked incompatible.
    # edge_ids remains the current-pose compatible set, NOT the original union.
    original_union_edge_ids: torch.Tensor
    original_fitted_centers_rc: torch.Tensor
    maximum_lost_explained_mass_fraction: float


def _canonical_cloud(cloud, minimum_q):
    if len(cloud.ids):
        _, first, inverse = np.unique(cloud.ids.numpy(), axis=0, return_index=True, return_inverse=True)
        if len(first) != len(cloud.ids):
            updates = {}
            for f in fields(cloud):
                value = getattr(cloud, f.name)
                if isinstance(value, torch.Tensor):
                    selected = value[torch.as_tensor(first)]
                    if not torch.equal(selected[torch.as_tensor(inverse)], value):
                        raise ValueError('duplicate correspondence records disagree: ' + f.name)
                    updates[f.name] = selected
            cloud = replace(cloud, **updates)
    keep = cloud.q >= minimum_q
    return replace(cloud, **{f.name:getattr(cloud, f.name)[keep] for f in fields(cloud)
                            if isinstance(getattr(cloud, f.name), torch.Tensor)})


class RepairedPoseConsensusBuilder(PoseConsensusBuilder):
    def __init__(self, geometry, proposal=None, policy=None):
        super().__init__(geometry, proposal)
        self.policy = policy or MergeRepairPolicy()

    def _at_pose(self, cloud, pose, seed_ids, seed_positions, hypothesis_ids, overlap_fn):
        """Recollect without secretly applying another four localization steps."""
        comp = cloud.compatibility(pose, self.geometry)
        membership = comp.kernel >= math.exp(-.5 * self.config.membership_sigma ** 2)
        mass = cloud.q * cloud.arc_weight * comp.kernel
        _, tangent, reliability = pair_frame(cloud.normal_a, cloud.normal_b,
                                             cloud.reliability_a, cloud.reliability_b)
        reliable = reliability >= self.geometry.normal_reliability_min
        identity = torch.eye(2, dtype=pose.dtype).expand(len(mass), -1, -1)
        info = torch.where(reliable[:, None, None], tangent[:, :, None]*tangent[:, None, :], identity)
        info = (mass[:, None, None] * info).sum(0) / mass.sum().clamp_min(1e-12)
        overlap = {} if overlap_fn is None else overlap_fn(pose)
        return PoseCluster(pose, cloud.ids[membership], tuple(sorted(set(seed_ids))),
            tuple(sorted(set(hypothesis_ids))), seed_positions, float(mass.sum()), info,
            bool(torch.linalg.eigvalsh(info).min() < .05), overlap)

    def _endpoint_moments(self, cloud, hypothesis, side):
        """Conditional displacement distributions, not independent hard matches.

        Normalizing is only for mean/covariance. Absolute endpoint masses remain
        separate, are used in conflict guardrails, and never become confidence.
        """
        ids = cloud.ids[:, side]
        count = int(ids.max()) + 1
        member = edge_membership(cloud.ids, hypothesis.edge_ids)
        comp = cloud.compatibility(hypothesis.translation, self.geometry)
        w = cloud.q.double() * comp.kernel.double() * member
        x = cloud.displacement.double()
        mass = torch.zeros(count, dtype=torch.float64).index_add_(0, ids, w)
        first = torch.zeros((count, 2), dtype=torch.float64).index_add_(0, ids, w[:, None] * x)
        second = torch.zeros((count, 2, 2), dtype=torch.float64).index_add_(
            0, ids, w[:, None, None] * x[:, :, None] * x[:, None, :])
        mean = first / mass[:, None].clamp_min(1e-30)
        covariance = second / mass[:, None, None].clamp_min(1e-30) - mean[:, :, None]*mean[:, None, :]
        covariance = .5 * (covariance + covariance.transpose(-1, -2))
        # Sampling uncertainty: TRAIN calibrated, not a tolerance fitted on real cases.
        s = (.5*(cloud.spacing_a + cloud.spacing_b)).double()
        noise = (s * self.geometry.evidence_tangent_sigma_per_spacing).clamp_min(self.geometry.sigma_floor_px)
        variance = torch.zeros(count, dtype=torch.float64).index_add_(0, ids, w * noise.square())
        variance = variance / mass.clamp_min(1e-30)
        return mass, mean, covariance, variance

    def _conflict_fraction(self, moments_a, moments_b):
        totals, conflicts = [], []
        for (ma, xa, va, sa), (mb, xb, vb, sb) in zip(moments_a, moments_b):
            shared = (ma > 0) & (mb > 0)
            if not bool(shared.any()):
                totals.append(torch.minimum(ma.sum(), mb.sum()))
                conflicts.append(ma.new_zeros(())); continue
            delta = xa[shared] - xb[shared]
            covariance = va[shared] + vb[shared] + torch.eye(2, dtype=va.dtype)[None] * (sa[shared] + sb[shared])[:, None, None]
            # Eigh avoids cancellation-induced negative eigenvalues in moment sums.
            eigenvalues, vectors = torch.linalg.eigh(covariance)
            projected = torch.einsum('bij,bi->bj', vectors, delta)
            distance2 = (projected.square() / eigenvalues.clamp_min(self.geometry.sigma_floor_px**2)).sum(-1)
            contradictory = distance2 > self.config.merge_sigma**2
            conflicts.append(torch.minimum(ma[shared], mb[shared])[contradictory].sum())
            totals.append(torch.minimum(ma.sum(), mb.sum()))
        return float(torch.stack(conflicts).sum() / torch.stack(totals).sum().clamp_min(1e-30))

    @torch.no_grad()
    def build_from_cloud(self, cloud, seeds=None, overlap_fn=None):
        cloud = _canonical_cloud(cloud, self.config.minimum_absolute_q)
        seeds = self._seeds(cloud) if seeds is None else seeds.detach().cpu()
        hypotheses = tuple(self._hypothesis(cloud, t, (i,), t[None], (i,), overlap_fn)
                           for i, t in enumerate(seeds))
        clusters = [h for h in hypotheses if len(h.edge_ids)]
        if not clusters:
            return ConsensusProposals(cloud, seeds, hypotheses, (), ())
        radius = math.sqrt(2) * self.config.merge_sigma * self._local_scale(cloud)
        gate = math.exp(-.5 * self.config.membership_sigma**2)
        reference_mass = {}
        moments = {}
        conflicts = {}
        for i, h in enumerate(hypotheses):
            if not len(h.edge_ids):
                continue
            member = edge_membership(cloud.ids, h.edge_ids)
            reference_mass[i] = cloud.q * cloud.arc_weight * cloud.compatibility(h.translation, self.geometry).kernel * member
            moments[i] = tuple(self._endpoint_moments(cloud, h, side) for side in (0, 1))

        def conflict(a, b):
            key = tuple(sorted((a, b)))
            if key not in conflicts:
                conflicts[key] = self._conflict_fraction(moments[a], moments[b])
            return conflicts[key]

        def union_ids(c):
            return getattr(c, 'original_union_edge_ids', c.edge_ids)

        def check(pose, members):
            centers = torch.stack([hypotheses[i].translation for i in members])
            if float((centers - pose).norm(dim=-1).max()) > radius:
                return False, 1.
            compatible = cloud.compatibility(pose, self.geometry).kernel >= gate
            losses = [float(reference_mass[i][~compatible].sum() / reference_mass[i].sum().clamp_min(1e-30))
                      for i in members]
            maximum = max(losses, default=0.)
            return maximum <= self.policy.maximum_lost_explained_mass_fraction, maximum

        trace = []
        while True:
            options = []
            for a in range(len(clusters)):
                for b in range(a+1, len(clusters)):
                    distance = float((clusters[a].translation - clusters[b].translation).norm())
                    if distance <= 2*radius:
                        options.append((distance, a, b))
            merged = False
            for _, a, b in sorted(options):
                ca, cb = clusters[a], clusters[b]
                members = tuple(sorted(set(ca.merged_hypothesis_ids + cb.merged_hypothesis_ids)))
                if any(conflict(i, j) > self.policy.maximum_conflicting_endpoint_mass_fraction
                       for n, i in enumerate(members) for j in members[n+1:]):
                    continue
                union = torch.unique(torch.cat((union_ids(ca), union_ids(cb))), dim=0)
                allowed = edge_membership(cloud.ids, union)
                positions = torch.cat((ca.seed_translations, cb.seed_translations))
                seed_ids = ca.initial_seed_ids + cb.initial_seed_ids
                same = torch.equal(ca.translation, cb.translation) and torch.equal(ca.edge_ids, cb.edge_ids)
                if same:
                    pose = ca.translation.clone()  # Exact duplicate must not trigger more optimization.
                else:
                    pose, _, _, _ = self._fit(cloud, .5*(ca.translation+cb.translation), allowed)
                ok, maximum_lost = check(pose, members)
                if not ok:
                    continue
                candidate = self._at_pose(cloud, pose, seed_ids, positions, members, overlap_fn)
                # An exact duplicate cannot create new material overlap. Its
                # existing overlap remains attached and still penalizes scoring.
                if (not same and candidate.overlap.get('available') and
                        candidate.overlap['fraction_sum_area'] >= self.config.maximum_merge_overlap_sum):
                    continue
                candidate = MergedPoseCluster(**{f.name:getattr(candidate, f.name) for f in fields(PoseCluster)},
                    original_union_edge_ids=union,
                    original_fitted_centers_rc=torch.stack([hypotheses[i].translation for i in members]),
                    maximum_lost_explained_mass_fraction=maximum_lost)
                trace.append(dict(merged_hypothesis_ids=members, union_edge_count=len(union),
                    recollected_edge_count=len(candidate.edge_ids), translation=pose.tolist(),
                    exact_duplicate=same, maximum_lost_explained_mass_fraction=maximum_lost,
                    maximum_endpoint_conflict_fraction=max((conflict(i,j) for n,i in enumerate(members)
                                                           for j in members[n+1:]), default=0.)))
                clusters = [c for i,c in enumerate(clusters) if i not in (a,b)] + [candidate]
                merged = True
                break
            if not merged:
                break
        clusters.sort(key=lambda c:(-c.absolute_support_mass_px, _pose_key(c.translation)))
        return ConsensusProposals(cloud, seeds, hypotheses, tuple(clusters[:self.config.max_clusters]), tuple(trace))
