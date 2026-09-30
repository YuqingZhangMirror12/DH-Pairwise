"""Isolated, inference-only pose-scale experiment; never imported by training.

s_pose is a radius in prepared-image pixels, NOT a Gaussian sigma. Only the
candidate-to-current-pose screen, all-original-center check, and explained-
mass membership band use it. The production directional kernel/_fit/_at_pose
and the finite 0..9px damage set remain unchanged.
"""
from dataclasses import dataclass, fields
import math

import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.geometry import pair_frame
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.legacy_pose_consensus import ConsensusProposals, PoseCluster, _pose_key
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.pose_consensus_repair import (
    RepairedPoseConsensusBuilder, MergedPoseCluster, edge_membership, _canonical_cloud)


@dataclass(frozen=True)
class PoseScalePolicy:
    radius_px: float = 10.
    adaptive: bool = False
    contact: bool = True
    deduplicate: bool = True
    jaccard: float = .9

    def radius(self, cloud):
        if self.adaptive:
            spacing = float((.5 * (cloud.spacing_a + cloud.spacing_b)).median()) if len(cloud.ids) else 0.
            return max(10., 3. * spacing)
        if self.radius_px <= 0 or not math.isfinite(self.radius_px):
            raise ValueError('s_pose must be positive and finite')
        return self.radius_px


class IsotropicPoseBuilder(RepairedPoseConsensusBuilder):
    def __init__(self, geometry, proposal=None, scale_policy=None):
        super().__init__(geometry, proposal)
        self.scale_policy = scale_policy or PoseScalePolicy()
        self.all_clusters = ()
        self.audit = {}

    def pose_member(self, cloud, pose, radius):
        c = cloud.compatibility(pose, self.geometry)
        # Distance to the SAME finite damage set, not a larger normal erosion
        # allowance. A new isotropic radius must not excuse deeper penetration.
        spacing = .5 * (cloud.spacing_a + cloud.spacing_b)
        sn = (spacing * self.geometry.normal_sigma_per_spacing).clamp_min(self.geometry.sigma_floor_px)
        penetration = c.reliable_normal & (c.normal_px < -self.config.membership_sigma * sn)
        return (c.unexplained_residual_rc.norm(dim=-1) <= radius) & ~penetration

    def contact_pose(self, cloud, pose, allowed):
        """Resolve only the weakly identified normal degree of freedom.

        The weighted 5th-percentile contact is a deterministic convention, not
        GT recovery. It moves at most the existing finite damage width and must
        preserve >=95% directional absolute support mass. Fully identified
        poses and unreliable-normal evidence are left untouched.
        """
        c = cloud.compatibility(pose, self.geometry)
        mass = cloud.q * cloud.arc_weight * allowed * c.kernel
        normal, tangent, reliability = pair_frame(cloud.normal_a, cloud.normal_b,
            cloud.reliability_a, cloud.reliability_b)
        reliable = reliability >= self.geometry.normal_reliability_min
        total = mass.sum()
        if float(total) <= 1e-12 or float(mass[reliable].sum()) < .95 * float(total):
            return pose, False
        eye = torch.eye(2, dtype=pose.dtype).expand(len(mass), -1, -1)
        info = torch.where(reliable[:, None, None], tangent[:, :, None] * tangent[:, None, :], eye)
        info = (mass[:, None, None] * info).sum(0) / total
        values, vectors = torch.linalg.eigh(info)
        if float(values[0]) >= .05:
            return pose, False
        axis = vectors[:, 0]
        average_normal = (normal * mass[:, None]).sum(0)
        if float(axis @ average_normal) < 0:
            axis = -axis
        projection = normal @ axis
        eligible = reliable & (projection > .5) & (mass > 0)
        if float(mass[eligible].sum()) < .95 * float(total):
            return pose, False
        distances = c.normal_px[eligible] / projection[eligible]
        order = torch.argsort(distances, stable=True)
        weights = mass[eligible][order]
        k = int(torch.searchsorted(weights.cumsum(0), .05 * weights.sum()).clamp_max(len(order)-1))
        step = distances[order[k]].clamp(0., self.geometry.damage_normal_upper_px)
        candidate = pose + step * axis
        new = cloud.compatibility(candidate, self.geometry)
        new_mass = (cloud.q * cloud.arc_weight * allowed * new.kernel).sum()
        if float(new_mass) + 1e-10 < .95 * float(total):
            return pose, False
        return candidate, bool(float(step) > 1e-6)

    @torch.no_grad()
    def build_from_cloud(self, cloud, seeds=None, overlap_fn=None, frozen_hypotheses=None):
        cloud = _canonical_cloud(cloud, self.config.minimum_absolute_q)
        seeds = self._seeds(cloud) if seeds is None else seeds.detach().cpu()
        hypotheses = (tuple(frozen_hypotheses) if frozen_hypotheses is not None else
            tuple(self._hypothesis(cloud, t, (i,), t[None], (i,), overlap_fn) for i, t in enumerate(seeds)))
        radius = self.scale_policy.radius(cloud)
        active = [i for i, h in enumerate(hypotheses) if len(h.edge_ids)]
        self.all_clusters = ()
        self.audit = dict(radius_px=radius, attempted=0, absorbed=0, contact_moves=0,
            deduplicated=0, rejected_distance=0, rejected_center=0, rejected_mass=0,
            rejected_conflict=0, rejected_overlap=0)
        if not active:
            return ConsensusProposals(cloud, seeds, hypotheses, (), ())
        references = {}
        moments = {}
        conflicts = {}
        for i in active:
            h = hypotheses[i]
            member = edge_membership(cloud.ids, h.edge_ids)
            references[i] = cloud.q * cloud.arc_weight * cloud.compatibility(h.translation, self.geometry).kernel * member
            moments[i] = tuple(self._endpoint_moments(cloud, h, side) for side in (0, 1))

        def conflict(i, j):
            key = tuple(sorted((i, j)))
            if key not in conflicts:
                conflicts[key] = self._conflict_fraction(moments[i], moments[j])
            return conflicts[key]

        def union_ids(members):
            return torch.unique(torch.cat([hypotheses[i].edge_ids for i in members]), dim=0)

        def check(pose, members):
            centers = torch.stack([hypotheses[i].translation for i in members])
            if float((centers - pose).norm(dim=1).max()) > radius + 1e-6:
                return False, 1., 'center'
            compatible = self.pose_member(cloud, pose, radius)
            lost = max(float(references[i][~compatible].sum() / references[i].sum().clamp_min(1e-30)) for i in members)
            if lost > self.policy.maximum_lost_explained_mass_fraction + 1e-7:
                return False, lost, 'mass'
            return True, lost, None

        def cluster_at(pose, members, lost):
            c = self._at_pose(cloud, pose, members, seeds[list(members)], members, overlap_fn)
            return MergedPoseCluster(**{f.name: getattr(c, f.name) for f in fields(PoseCluster)},
                original_union_edge_ids=union_ids(members),
                original_fitted_centers_rc=torch.stack([hypotheses[i].translation for i in members]),
                maximum_lost_explained_mass_fraction=lost)

        def overlap_blocked(pose):
            o = {} if overlap_fn is None else overlap_fn(pose)
            return bool(o.get('available') and o['fraction_sum_area'] >= self.config.maximum_merge_overlap_sum)

        def canonical(pose, members):
            if not self.scale_policy.contact:
                return pose, False
            candidate, moved = self.contact_pose(cloud, pose, edge_membership(cloud.ids, union_ids(members)))
            if moved and check(candidate, members)[0] and not overlap_blocked(candidate):
                return candidate, True
            return pose, False

        trace = []
        def attempt(current, members_to_add, mode):
            self.audit['attempted'] += 1
            # Member-to-current joint pose, not nearest-neighbour chaining.
            if any(float((hypotheses[i].translation-current.translation).norm()) > radius + 1e-6
                   for i in members_to_add):
                self.audit['rejected_distance'] += 1
                return None
            members = tuple(sorted(set(current.merged_hypothesis_ids) | set(members_to_add)))
            maximum_conflict = max((conflict(i, j) for n, i in enumerate(members) for j in members[n+1:]), default=0.)
            if maximum_conflict > self.policy.maximum_conflicting_endpoint_mass_fraction:
                self.audit['rejected_conflict'] += 1
                return None
            union = union_ids(members)
            allowed = edge_membership(cloud.ids, union)
            # Initialize from each hypothesis' evidence ONCE, using per-edge
            # max Q mass to avoid duplicate seeds multiplying influence.
            unique_mass = torch.stack([references[i] for i in members]).amax(0)
            centers = torch.stack([hypotheses[i].translation for i in members])
            # Each original hypothesis is a candidate, not an extra Q vote.
            # Start at the current common pose and keep production _fit exact.
            pose, _, _, _ = self._fit(cloud, current.translation, allowed)
            pose, moved = canonical(pose, members)
            ok, lost, reason = check(pose, members)
            if not ok:
                self.audit['rejected_' + reason] += 1
                return None
            if overlap_blocked(pose):
                self.audit['rejected_overlap'] += 1
                return None
            candidate = cluster_at(pose, members, lost)
            self.audit['absorbed'] += 1
            self.audit['contact_moves'] += int(moved)
            trace.append(dict(mode=mode, merged_hypothesis_ids=members, translation=pose.tolist(),
                union_edge_count=len(union), recollected_edge_count=len(candidate.edge_ids),
                maximum_center_distance_px=float((centers-pose).norm(dim=1).max()),
                maximum_lost_explained_mass_fraction=lost, maximum_endpoint_conflict_fraction=maximum_conflict,
                contact_moved=moved, reference_unique_mass_px=float(unique_mass.sum())))
            return candidate

        remaining = set(active)
        clusters = []
        while remaining:
            anchor = min(remaining, key=lambda i: (-hypotheses[i].absolute_support_mass_px, _pose_key(hypotheses[i].translation), i))
            remaining.remove(anchor)
            pose, moved = canonical(hypotheses[anchor].translation, (anchor,))
            self.audit['contact_moves'] += int(moved)
            current = cluster_at(pose, (anchor,), check(pose, (anchor,))[1])
            while remaining:
                occupied = edge_membership(cloud.ids, current.original_union_edge_ids)
                # Prefer new independent support, then absolute support; never
                # sort by the nearest inter-candidate distance.
                order = sorted(remaining, key=lambda i: (-float(references[i][~occupied].sum()),
                    -hypotheses[i].absolute_support_mass_px, _pose_key(hypotheses[i].translation), i))
                accepted = False
                for i in order:
                    candidate = attempt(current, (i,), 'absorb_member')
                    if candidate is not None:
                        current = candidate
                        remaining.remove(i)
                        accepted = True
                        break
                if not accepted:
                    break
            clusters.append(current)

        # Duplicate evidence may survive contact normalization/order. Dedup is
        # another guarded UNION, not a highest-score discard or transitive link.
        if self.scale_policy.deduplicate:
            while True:
                clusters.sort(key=lambda c: (-c.absolute_support_mass_px, _pose_key(c.translation)))
                absorbed = False
                for a, ca in enumerate(clusters):
                    ea = {tuple(x) for x in ca.edge_ids.tolist()}
                    for b in range(a+1, len(clusters)):
                        cb = clusters[b]
                        eb = {tuple(x) for x in cb.edge_ids.tolist()}
                        if not ea or len(ea & eb) / len(ea | eb) < self.scale_policy.jaccard:
                            continue
                        candidate = attempt(ca, cb.merged_hypothesis_ids, 'jaccard_dedup')
                        if candidate is not None:
                            clusters = [c for i, c in enumerate(clusters) if i not in (a, b)] + [candidate]
                            self.audit['deduplicated'] += 1
                            absorbed = True
                            break
                    if absorbed:
                        break
                if not absorbed:
                    break
        clusters.sort(key=lambda c: (-c.absolute_support_mass_px, _pose_key(c.translation)))
        self.all_clusters = tuple(clusters)
        return ConsensusProposals(cloud, seeds, hypotheses, self.all_clusters[:self.config.max_clusters], tuple(trace))
