"""User-confirmed original20 / final30 curriculum gate, without new GT fitting.

The denominator is frozen before additional cutting or any corrosion. Primary
damage is not credited. Only the last light erosion may move surviving contact
vertices, by at most four pixels on each original inward normal. Count original
adjacent edges, never distances between disconnected surviving points.
"""
import numpy as np
from scipy import ndimage
from ..s7_balanced_v2.latent_seam import contour, ray_project

ORIGINAL_MINIMUM = .20
CONNECTABLE_MINIMUM = .30
LIGHT_LIMIT_PX = 4.
CONTRACT = 'curriculum-original20-final30-light4/2-common-arc'


def unchanged(original, current, points):
    """Independent nine-offset raster check, including both edge endpoints."""
    points = np.rint(points).astype(int)
    good = np.ones(len(points), bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            q = points + (dr, dc)
            inside = (q >= 0).all(1) & (q < np.array(original.shape)).all(1)
            p = q[inside]
            good[inside] &= original[tuple(p.T)] == current[tuple(p.T)]
    return good


def decide(original_ratio, fractions, common_fraction):
    if not np.isfinite(original_ratio) or not np.isfinite(common_fraction):
        raise ValueError('nonfinite seam measurement')
    if set(fractions) != {'a', 'b'} or not all(np.isfinite(x) for x in fractions.values()):
        raise ValueError('two finite side fractions required')
    original_pass = original_ratio >= ORIGINAL_MINIMUM
    # A and B may have slightly different raster arc lengths. The user's
    # threshold is FINAL COMMON / ORIGINAL COMMON, not an additional separate
    # per-side percentage gate. Keep per-side ratios as diagnostics only.
    final_pass = common_fraction >= CONNECTABLE_MINIMUM
    return dict(original20_pass=original_pass, final30_pass=final_pass,
                eligible=bool(original_pass and final_pass),
                exclusion_reasons=([] if original_pass else ['original_seam_below20_smaller_perimeter'])
                    + ([] if final_pass else ['final_light_only_seam_below30_original']))


def evaluate_masks(original, primary, final, reference, translation):
    """Reference contains hash-bound ORIGINAL full curves and frozen partners.

    `primary` is after structural cuts plus every main corrosion (including
    continuous weak corrosion), but before the last whole-contour light step.
    This function never edits masks or labels and never constructs new partners.
    """
    arrays = {}; originals = {}; prelight = {}; retained = {}; fractions = {}
    areas = {}; perimeters = {}; light_depths = {}
    for side in 'ab':
        for masks in (original, primary, final):
            if masks[side].ndim != 2 or masks[side].shape != original[side].shape:
                raise ValueError('stage shape mismatch')
        if np.any(primary[side] & ~original[side]) or np.any(final[side] & ~primary[side]):
            raise ValueError('added material in archived stages')
        areas[side] = int(original[side].sum())
        p, edge = contour(original[side])
        if not np.array_equal(p, reference[side+'_points']):
            raise ValueError('reference is not the original full outer contour')
        if not np.allclose(edge, reference[side+'_edge'], rtol=0, atol=1e-9):
            raise ValueError('original arclength weights changed')
        perimeters[side] = float(edge.sum())
        removed = primary[side] & ~final[side]
        if removed.any():
            filled = ndimage.binary_fill_holes(np.pad(primary[side], 1))
            depth = ndimage.distance_transform_edt(filled)[1:-1, 1:-1] - .5
            light_depths[side] = float(depth[removed].max())
        else:
            light_depths[side] = 0.
        if light_depths[side] > LIGHT_LIMIT_PX + 1e-6:
            raise ValueError('final light raster depth exceeds4px')
    for side, other, shift in [('a', 'b', translation), ('b', 'a', -translation)]:
        p = reference[side+'_points']; q = reference[side+'_partner_points']
        w = reference[side+'_edge']; source = reference[side+'_source_edges'].astype(bool)
        good = unchanged(original[side], primary[side], p)
        good &= unchanged(original[other], primary[other], q)
        good &= np.linalg.norm(p + shift - q, axis=1) <= 3. + 1e-6
        pre = source & good & np.roll(good, -1)
        _, depth_a, valid_a = ray_project(original[side], final[side], p)
        _, depth_b, valid_b = ray_project(original[other], final[other], q)
        light = valid_a & valid_b & (depth_a <= LIGHT_LIMIT_PX) & (depth_b <= LIGHT_LIMIT_PX)
        final_edges = pre & light & np.roll(light, -1)
        originals[side] = float(w[source].sum())
        if originals[side] <= 0:
            raise ValueError('missing positive original common seam')
        prelight[side] = float(w[pre].sum())
        retained[side] = float(w[final_edges].sum())
        fractions[side] = retained[side] / originals[side]
        arrays.update({side+'_original_edges': source, side+'_primary_free_edges': pre,
                       side+'_connectable_edges': final_edges})
    smaller = min('ab', key=lambda s: (areas[s], s))
    original_common = min(originals.values()); final_common = min(retained.values())
    ratio = original_common / perimeters[smaller]
    common_fraction = final_common / original_common
    return dict(contract=CONTRACT, original_common_length_px=original_common,
        original_length_by_side_px=originals, original_area_px=areas,
        original_full_perimeter_px=perimeters, original_smaller_fragment=smaller,
        original_over_smaller_perimeter=ratio, primary_free_length_by_side_px=prelight,
        connectable_length_by_side_px=retained, final_common_length_px=final_common,
        connectable_fraction_by_side=fractions, final_over_original=common_fraction,
        conservative_fraction=min(fractions.values()), light_max_raster_depth_px=light_depths,
        primary_damage_included=False, light_per_side_normal_limit_px=LIGHT_LIMIT_PX,
        denominator_reduced_after_cut_or_corrosion=False, **decide(ratio, fractions, common_fraction)), arrays


def require(summary):
    if not summary['eligible']:
        raise ValueError(';'.join(summary['exclusion_reasons']))


def require_original(base, bands=None):
    from .crop_floor import measure
    summary, _ = measure(base, base, bands)
    if summary['common_over_smaller_perimeter'] < ORIGINAL_MINIMUM:
        raise ValueError('original_seam_below20_smaller_perimeter')
    return summary


def evaluate(base, primary, final, bands=None):
    from .pristine import measure
    if not base.label:
        raise ValueError('negative pairs have no common-seam contract')
    _, reference, _ = measure(base, primary, bands)
    stage = lambda sample: {s: getattr(sample, 'mask_'+s)[0].astype(bool) for s in 'ab'}
    return evaluate_masks(stage(base), stage(primary), stage(final), reference,
                          np.asarray(base.translation_a_to_b_rc))
