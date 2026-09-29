"""Approved natural endpoint cuts and v17 erosion; final primary peak 5–35px."""
import numpy as np
from scipy.spatial import cKDTree
from ..aggressive_data_v16.geometry import positive_trim, negative_trim
from ..aggressive_data_v15.geometry import apply_partial, topology
from ..aggressive_data_v15.fixed_gap import measure_fixed
from ..aggressive_data_v17.geometry import weather_pair
from ..s7_balanced_v2.partial_v14 import retained_support
from ..s7_balanced_v2.background_recession import pair as light_pair
from ..s7_compound_v1.geometry import masks, area_ratio


def valid_primary_peak(peak):
    return np.isfinite(peak) and 5. <= peak <= 35.


def gap_check(base, final, bands, retained, fields, primary_present):
    _, proof = retained_support(base, retained, bands)
    summary, arrays = measure_fixed(base, final, proof)
    if summary is None:
        raise ValueError('no measurable source projection')
    damaged = []
    for side in 'ab':
        p = arrays[side+'_source_points']; active = np.zeros(len(p), bool)
        for source in 'ab':
            if source+'_points' not in fields:
                continue
            shift = (base.translation_a_to_b_rc if side == 'a' and source == 'b' else
                    -base.translation_a_to_b_rc if side == 'b' and source == 'a' else np.zeros(2))
            distance, index = cKDTree(fields[source+'_points']).query(p+shift)
            active |= (distance <= 3.5) & (fields[source+'_total_exact'][index] > 0)
        valid = active & arrays[side+'_valid']
        damaged.extend(arrays[side+'_gap'][valid]); arrays[side+'_primary_affected'] = active
    if primary_present:
        if not damaged or not valid_primary_peak(float(np.max(damaged))):
            raise ValueError('actual primary bilateral peak outside5–35px')
        summary.update(primary_gap_peak_px=float(np.max(damaged)),
            primary_gap_quantiles_px=np.quantile(damaged, [0,.1,.25,.5,.75,.9,1]).tolist())
    else:
        summary.update(primary_gap_peak_px=None, primary_gap_quantiles_px=None)
    return summary, arrays


def augment(pair, rng, mode, bank, size_class, k):
    recipe = pair[0]['entry']['corrosion_recipe']
    if recipe != pair[1]['entry']['corrosion_recipe']:
        raise ValueError('paired recipe mismatch')
    bases = [r['original'] for r in pair]
    pos, plan, bands = positive_trim(bases[0], pair[0]['old_primary'], rng, mode, bank, size_class)
    neg, neg_plan = negative_trim(bases[1], plan, rng)
    trimmed = (pos, neg); plans = (plan, neg_plan)
    current = []; primary = []; details = []; fields = []
    for index, row in enumerate(pair):
        base = bases[index]; now = trimmed[index]
        if recipe == 'partial':
            now = apply_partial(now, masks(row['old_primary'])); detail = {}; field = {}
            if index == 0:
                support, _ = retained_support(base, now, bands)
                if support['common_over_smaller_perimeter'] < .15:
                    raise ValueError('Partial15% floor')
        elif recipe == 'clean':
            detail = {}; field = {}
        else:
            now, detail, field = weather_pair(now, base, row['fields'], recipe, rng, k)
        primary.append(now)
        if recipe != 'clean':
            final, light, light_arrays = light_pair(base, now, field, rng, .70, .02)
            if final is None:
                raise ValueError('background:'+str(light['reason']))
        else:
            final = now; light = {}; light_arrays = {}
        if final.label and (final.target_a >= 0).sum() < 4:
            raise ValueError('fewer_than_four_inherited_targets')
        if area_ratio(final) < min(.125, area_ratio(base))*.95:
            raise ValueError('extreme_area_ratio')
        for side in 'ab':
            topology(getattr(base, 'mask_'+side)[0].astype(bool), getattr(final, 'mask_'+side)[0].astype(bool))
        current.append(final); fields.append(dict(primary=field, light=light_arrays))
        details.append(dict(primary_damage=detail, background=light, trim=plans[index],
            requested_gap_count=k if recipe.startswith('gaps') else 0,
            depth_revision='v17-3to8-5to15-k1to4-gap35'))
    gap, arrays = gap_check(bases[0], current[0], bands,
        primary[0] if recipe == 'partial' else trimmed[0], fields[0]['primary'], recipe not in ('clean', 'partial'))
    details[0]['gap'] = gap; details[1]['gap'] = None
    return tuple(current), details, fields, arrays, trimmed, tuple(primary)

