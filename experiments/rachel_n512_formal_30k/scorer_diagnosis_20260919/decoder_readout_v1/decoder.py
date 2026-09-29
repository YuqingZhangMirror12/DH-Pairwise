"""GT-free, explicit proposal budgets for an existing frozen threshold builder.

The original fitter, complete-link T16 merge, exact union, physical veto and
final eight-candidate budget are untouched. A copy is returned; no class or
running-source monkey patch is made. No new parameters or Matcher pass.
"""
from copy import copy
from dataclasses import dataclass, replace, asdict
from types import MethodType
import numpy as np
import torch


@dataclass(frozen=True)
class SearchPolicy:
    row_column_topk: int = 2
    mode_limit: object = 128  # None means all occupied voting bins, NOT all Q.
    initial_seeds: int = 16

    def __post_init__(self):
        if self.row_column_topk < 1 or self.initial_seeds < 8:
            raise ValueError('positive TopK and at least the existing eight-candidate budget')
        if self.mode_limit is not None and self.mode_limit < self.initial_seeds:
            raise ValueError('mode_limit must accommodate seeds; no implicit 8x coupling')


# Prespecified mechanisms, not a hyperparameter search selected on real labels.
CONTROLS = {
    'baseline': SearchPolicy(),
    'top3_only': SearchPolicy(3, 128, 16),
    'all_modes_only': SearchPolicy(2, None, 16),
    'seeds32_only': SearchPolicy(2, 128, 32),
    'combined': SearchPolicy(3, None, 32),
}


def select_seeds(builder, cloud):
    policy = builder.search_policy
    if not len(cloud.ids):
        builder.search_audit = dict(policy=asdict(policy), bins=0, eligible_modes=0,
                                    selected_modes=[], gt_used=False)
        return cloud.displacement.new_zeros((0, 2))
    frame, pose_key = builder._search_frame, builder._search_pose_key
    n, _, rel = frame(cloud.normal_a, cloud.normal_b,
                      cloud.reliability_a, cloud.reliability_b)
    fractions = torch.tensor(builder.config.normal_vote_fractions, dtype=n.dtype)
    allowance = builder.geometry.damage_normal_upper_px * (
        rel >= builder.geometry.normal_reliability_min)
    votes = cloud.displacement[:, None] - n[:, None] * allowance[:, None, None] * fractions[None, :, None]
    weights = (cloud.q * cloud.arc_weight)[:, None].expand(-1, len(fractions)) / len(fractions)
    bin_width = 2 * builder._local_scale(cloud)
    xy, w = votes.reshape(-1, 2).numpy(), weights.reshape(-1).numpy()
    bins = np.floor(xy / bin_width + .5).astype(np.int64)
    unique, inverse = np.unique(bins, axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=w, minlength=len(unique))
    sums = np.stack([np.bincount(inverse, weights=w * xy[:, d],
                                minlength=len(unique)) for d in range(2)], -1)
    centers = sums / np.maximum(mass[:, None], 1e-20)
    representative = []
    for k in range(len(unique)):
        members = np.flatnonzero(inverse == k)
        representative.append(int(members[np.argmax(w[members])]) // len(fractions))
    order = sorted(range(len(unique)), key=lambda k: (-mass[k], pose_key(centers[k])))
    candidates = order[:] if policy.mode_limit is None else order[:policy.mode_limit]
    eligible_count = len(candidates)
    chosen = []
    while candidates and len(chosen) < policy.initial_seeds:
        def priority(k):
            if not chosen:
                return float(mass[k])
            i = representative[k]
            diversity = []
            for h in chosen:
                j = representative[h]
                da = abs(float(cloud.arc_a[i] - cloud.arc_a[j])); da = min(da, cloud.perimeter_a - da)
                db = abs(float(cloud.arc_b[i] - cloud.arc_b[j])); db = min(db, cloud.perimeter_b - db)
                arc = min(1., .5 * (da + db) / max(1., 8 * builder.config.observation_radius_px))
                pose = min(1., float(np.linalg.norm(centers[k] - centers[h])) / max(1., 4 * bin_width))
                diversity.append(max(arc, pose))
            return float(mass[k]) * (.1 + .9 * min(diversity))
        # candidates retain descending raw-mass order. Diversity multiplier is
        # in[.1,1], so priority(k)<=mass(k). Once mass is STRICTLY below the
        # current best priority, no later mode can win. This exact bound makes
        # all occupied modes eligible without an arbitrary new cutoff.
        best_key = None
        k = None
        for candidate in candidates:
            if best_key is not None and float(mass[candidate]) < -best_key[0]:
                break
            key = (-priority(candidate), pose_key(centers[candidate]))
            if best_key is None or key < best_key:
                best_key, k = key, candidate
        chosen.append(k); candidates.remove(k)
    ranks = {k: rank + 1 for rank, k in enumerate(order)}
    builder.search_audit = dict(policy=asdict(policy), bins=len(unique),
        bin_width_px=bin_width, eligible_modes=eligible_count,
        selected_modes=[dict(mass_rank=ranks[k], vote_mass=float(mass[k]),
                             pose_rc=centers[k].tolist()) for k in chosen], gt_used=False)
    return torch.as_tensor(centers[chosen], dtype=cloud.displacement.dtype)


def copy_with_search(builder, policy):
    """Explicitly opt in on a new object; old builder/config/cache remain intact."""
    if not hasattr(builder, 'policy') or builder.policy.pose_diameter_px != 16.:
        raise ValueError('this control is for the bound T16 builder, not simple-radius or legacy merge')
    if builder.policy.candidate_budget != 8:
        raise ValueError('the final cluster budget must stay eight')
    globals_ = builder._seeds.__func__.__globals__
    result = copy(builder)
    result.config = replace(builder.config, row_column_topk=policy.row_column_topk,
                            initial_seeds=policy.initial_seeds)
    result.search_policy = policy
    result._search_frame = globals_['pair_frame']
    result._search_pose_key = globals_['_pose_key']
    result.search_audit = {}
    result._seeds = MethodType(select_seeds, result)
    return result
