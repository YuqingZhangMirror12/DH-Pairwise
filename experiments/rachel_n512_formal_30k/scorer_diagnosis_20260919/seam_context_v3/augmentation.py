"""TRAIN-only paired reflections; preserve point identity and correspondence."""
from dataclasses import replace
import numpy as np


def paired_mirror(sample, axis):
    """Reflect BOTH fragments in their respective pixel canvases.

    Point order/IDs, reciprocal targets and gap/component indices are unchanged.
    The model's existing compact() canonicalization handles the reversed winding.
    rc convention: p_B = p_A + t; after reflection t' = R*t + d_B-d_A.
    """
    if axis not in ('horizontal', 'vertical'):
        raise ValueError('axis must be horizontal or vertical')
    coord = 1 if axis == 'horizontal' else 0
    image_axis = -1 if coord == 1 else -2
    changes = {};offsets = []
    for side in 'ab':
        mask = getattr(sample, 'mask_'+side)
        extent = mask.shape[image_axis]-1
        offsets.append(extent)
        for prefix in ('mask_', 'coarse_mask_'):
            changes[prefix+side] = np.ascontiguousarray(np.flip(getattr(sample,prefix+side),image_axis))
        points = getattr(sample, 'points_rc_'+side).copy()
        valid = getattr(sample, 'contour_valid_'+side)
        points[valid,coord] = extent-points[valid,coord]
        changes['points_rc_'+side] = points
    t = sample.translation_a_to_b_rc.copy()
    if sample.translation_valid:
        t[coord] = -t[coord]+offsets[1]-offsets[0]
    changes['translation_a_to_b_rc'] = t
    changes['translation_a_to_b_xy_cartesian'] = np.array((t[1],-t[0]),dtype=t.dtype)
    return replace(sample, **changes)


def mirror_schedule(count, probability, seed, epoch):
    """Exactly half of the requested fraction per axis, reproducible on resume."""
    if not 0 <= probability <= 1:
        raise ValueError('mirror probability must be within [0,1]')
    codes = np.zeros(count,np.uint8)
    per_axis = int(count*probability/2)
    if per_axis:
        order = np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(count)
        codes[order[:per_axis]] = 1
        codes[order[per_axis:2*per_axis]] = 2
    return codes
