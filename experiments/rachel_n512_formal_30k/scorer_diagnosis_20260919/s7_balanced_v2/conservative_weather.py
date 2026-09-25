"""Conservative v12: one major damage type, optional smooth weak recession.

All depth fields are specified on an ordered original contour, not pixel noise.
Only one endpoint is weathered. Its total affected source arc (including weak
damage) is <= 50% of the pre-weather TRAIN-supported seam. Negatives use a
random own-contour corridor of the same relative length, without invented GT.
"""
from collections import Counter
from dataclasses import replace

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from ..s7_compound_v1.geometry import (
    EIGHT, area_ratio, eligible_seam, extract_ordered_outer_contour,
    _eligible_runs, _place_bumps, _signed_arc_distance, _changed_view)
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import _readonly
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import inherit_pair_targets


NAMES = {
    'partial': 'Partial Seam（仅加1–3px轻退化）',
    'clean': '无腐蚀', 'mild': '连续弱腐蚀 1–4px',
    'wave': '局部起伏退蚀 <10px', 'wave_weak': '起伏退蚀＋连续弱腐蚀',
    'local_abrupt': '局部腐蚀·突变型 5–9px',
    'local_abrupt_weak': '突变型局部腐蚀＋连续弱腐蚀',
    'local_gradual': '局部腐蚀·渐进型 5–9px',
    'local_gradual_weak': '渐进型局部腐蚀＋连续弱腐蚀',
    'gaps': '1–3处保守缺口', 'gaps_weak': '1–3处缺口＋连续弱腐蚀',
}


def parts(recipe):
    weak = recipe == 'mild' or recipe.endswith('_weak')
    major = recipe.removesuffix('_weak')
    if major in ('clean', 'mild', 'partial'):
        major = None
    return major, weak


def smooth_profile(u, peak, mode, phase=0.):
    """Continuous arc profiles; weak shoulders are C1, not pixelwise holes."""
    u = np.asarray(u, float)
    inside = np.abs(u) < 1.
    x = np.clip(1.-np.abs(u), 0., 1.)
    if mode == 'abrupt':
        result = np.full_like(x, peak)
    elif mode == 'gradual':
        # 1px at the edge, continuous undulating rise to the peak near centre.
        envelope = np.sin(np.pi*x/2)**2
        result = 1.+(peak-1.)*envelope*(.88+.12*np.cos(3*np.pi*u+phase))
    elif mode == 'wave':
        shoulder = np.sin(np.pi*np.minimum(x/.24, 1.)/2)**2
        result = (1.+(peak-1.)*(.65+.35*np.cos(2*np.pi*u+phase)))*shoulder
    elif mode == 'weak_gradual':
        result = peak*np.sin(np.pi*x/2)**2*(.9+.1*np.cos(2*np.pi*u+phase))
    elif mode == 'weak_inset':
        # Nearly uniform inset in the interior, smoothly joined at both ends.
        result = peak*np.sin(np.pi*np.minimum(x/.28, 1.)/2)**2
    elif mode == 'gap':
        result = peak*np.sin(np.pi*x/2)**2
    else:
        raise ValueError(mode)
    return np.where(inside, np.clip(result, 0., peak), 0.)


def contour(mask):
    dense, _ = extract_ordered_outer_contour(mask, cap=mask.size, smoothing_sigma=0.)
    dense = np.asarray(dense, float)
    edge = np.linalg.norm(np.roll(dense, -1, axis=0)-dense, axis=1)
    return dense, edge, np.r_[0., np.cumsum(edge[:-1])]


def depth_field(arc, edge, eligible, rng, major, weak, k=1):
    perimeter = float(edge.sum())
    eligible_length = float(edge[eligible].sum())
    runs = _eligible_runs(eligible, arc, edge)
    if not runs or eligible_length < 20.:
        return None
    # Reserve raster-cell margin below the user's hard 50% maximum.
    budget = .44*eligible_length
    count = k if major == 'gaps' else 1
    if major == 'gaps':
        upper = min(35., budget/count)
        if upper < 10.:
            return None
        widths = rng.uniform(min(15., upper), upper, count)
    else:
        upper = min(budget, max(r[1] for r in runs)-2.)
        if major and major.startswith('local'):
            upper = min(90., upper)
        if upper < 12.:
            return None
        widths = np.array([rng.uniform(max(12., upper*.7), upper)])
    centers = _place_bumps(widths, runs, perimeter, rng, 4.)
    if centers is None:
        return None
    phase = float(rng.uniform(-np.pi, np.pi))
    base = np.zeros(len(arc)); weak_field = base.copy(); regions = []
    weak_mode = str(rng.choice(['weak_gradual', 'weak_inset'])) if weak else None
    weak_peak = float(rng.uniform(1., 4.)) if weak else 0.
    weak_region = int(rng.integers(len(centers))) if weak else -1
    for ordinal, (center, width) in enumerate(zip(centers, widths)):
        u = _signed_arc_distance(arc, center, perimeter)/(width/2.)
        active = (np.abs(u) < 1.) & eligible
        if major:
            peak = float(rng.uniform(5., 9.))
            mode = {'wave':'wave', 'gaps':'gap', 'local_abrupt':'abrupt',
                    'local_gradual':'gradual'}[major]
            profile = smooth_profile(u, peak, mode, phase)*eligible
            base = np.maximum(base, profile)
        else:
            peak = 0.; profile = np.zeros(len(arc))
        if weak and ordinal == weak_region:
            # Same contiguous footprint: overlay never extends affected length.
            w = smooth_profile(u, weak_peak, weak_mode, phase)*eligible
            weak_field = np.maximum(weak_field, w)
        regions.append(dict(center_arc_px=float(center),support_length_px=float(width),
            requested_peak_depth_px=peak,indices=np.flatnonzero(active),profile=profile))
    total = np.minimum(9., base+weak_field)
    affected = (total > 0.)
    fraction = float(edge[affected].sum()/eligible_length)
    if fraction > .5+1e-9 or np.any(affected & ~eligible):
        return None
    return dict(base=base,weak=weak_field,total=total,regions=regions,
        eligible_length_px=eligible_length,affected_length_px=float(edge[affected].sum()),
        affected_fraction=fraction,weak_mode=weak_mode,weak_peak=weak_peak)


def weather_fragment(mask, rng, eligible, major, weak, k):
    mask = np.asarray(mask, bool)
    dense, edge, arc = contour(mask)
    if ndimage.label(mask, EIGHT)[1] != 1:
        return None, dict(reason='original_disconnected'), None
    filled = ndimage.binary_fill_holes(np.pad(mask, 1), structure=EIGHT)
    inward = ndimage.distance_transform_edt(filled)[1:-1,1:-1]-.5
    band = np.argwhere(mask & (inward <= 9.))
    nearest = cKDTree(dense).query(band)[1]
    depth = inward[tuple(band.T)]
    reasons = Counter()
    for attempt in range(16):
        field = depth_field(arc, edge, eligible, rng, major, weak, k)
        if field is None:
            reasons['placement_does_not_fit'] += 1; continue
        removed = depth <= field['total'][nearest]
        new = mask.copy(); new[tuple(band[removed].T)] = False
        if not removed.any():
            reasons['no_change'] += 1; continue
        if ndimage.label(new, EIGHT)[1] != 1:
            reasons['disconnected'] += 1; continue
        if new.sum() < max(64., .25*mask.sum()):
            reasons['retained_area'] += 1; continue
        if np.any(ndimage.binary_fill_holes(new, structure=EIGHT) & ~new & mask):
            reasons['new_enclosed_hole'] += 1; continue
        base_removed = depth <= field['base'][nearest]
        weak_extra = int(np.count_nonzero(removed & ~base_removed))
        if weak and weak_extra < 4:
            reasons['weak_not_effective'] += 1; continue
        regions = []
        failed = False
        for r in field['regions']:
            ii = np.isin(nearest, r['indices'])
            actual = float(depth[removed & ii].max(initial=0.))
            if major and (np.count_nonzero(base_removed & ii) < 8 or
                          actual < r['requested_peak_depth_px']-1.5):
                failed = True; break
            regions.append(dict(source_points_rc=dense[r['indices']].tolist(),
                center_arc_px=r['center_arc_px'],support_length_px=r['support_length_px'],
                requested_peak_depth_px=r['requested_peak_depth_px'],
                applied_max_depth_px=actual,
                independently_removed_pixels=int(np.count_nonzero(base_removed & ii))))
        if failed:
            reasons['major_not_effective'] += 1; continue
        local = bool(major and major.startswith('local'))
        notch_regions = regions if major == 'gaps' or local else []
        info = dict(applied=True,conservative=True,major_type=major,weak_applied=weak,
            weak_mode=field['weak_mode'],weak_peak_px=field['weak_peak'],
            weak_independently_removed_pixels=weak_extra,attempts=attempt+1,attempt_reasons=dict(reasons),
            wave=major=='wave',wave_range_px=[0.,9.] if major=='wave' else None,
            max_combined_depth_px=9.,applied_max_depth_px=float(depth[removed].max()),
            removed_area_px=int(removed.sum()),removed_fraction=float(removed.sum()/mask.sum()),
            component_count=1,notch_count=len(notch_regions),ignore_source_regions=notch_regions,
            notch_kinds=['gap' if major=='gaps' else 'local']*len(notch_regions),
            notch_independently_removed_pixels=[r['independently_removed_pixels'] for r in notch_regions],
            requested_peak_depths_px=[r['requested_peak_depth_px'] for r in regions] if major else [],
            requested_support_lengths_px=[r['support_length_px'] for r in regions],
            eligible_length_px=field['eligible_length_px'],
            affected_length_px=field['affected_length_px'],affected_fraction=field['affected_fraction'],
            no_major_corrosion_combination=True,weak_overlay_capped_to_total9=True,
            depth_coordinate='distance from original outer boundary; pixel EDT minus0.5',
            survival_island=None)
        arrays=dict(points=dense.astype(np.float32),arc=arc.astype(np.float32),
            edge=edge.astype(np.float32),eligible=eligible,major=field['base'].astype(np.float32),
            weak=field['weak'].astype(np.float32),total=field['total'].astype(np.float32),
            total_exact=field['total'].copy())
        return new, info, arrays
    return None, dict(reason='conservative_weather_rejected',attempt_reasons=dict(reasons)), None


def damaged_pair(sample, rng, plan, eligible_fraction=None):
    side=plan['side'];major,weak=parts(plan['recipe'])
    mask=np.asarray(getattr(sample,'mask_'+side)[0],bool)
    dense,edge,arc=contour(mask);perimeter=float(edge.sum())
    if sample.label:
        eligible=eligible_seam(sample,side,dense,True,
            cut_guard=plan['cut_guards'][side],cut_guard_radius=plan['guard_radius'],
            contact_tolerance=plan['contact_tolerance'],bridge_short_gaps_px=20.)
    else:
        # No clipping to a larger minimum corridor for negatives.
        center=float(rng.uniform(0.,perimeter))
        eligible=np.abs(_signed_arc_distance(arc,center,perimeter)) < perimeter*eligible_fraction/2.
    fraction=float(edge[eligible].sum()/perimeter)
    changed,info,arrays=weather_fragment(mask,rng,eligible,major,weak,plan['k'])
    if changed is None:
        return None,info,fraction,None
    views={};details={};updates={}
    for s in 'ab':
        geometry=info if s==side else dict(applied=False,reason='unselected_endpoint')
        m=changed if s==side else getattr(sample,'mask_'+s)[0]
        view=_changed_view(sample,s,m,geometry,'local' if geometry.get('ignore_source_regions') else 'wave')
        views[s]=view;details[s]=geometry
        coarse=np.asarray(Image.fromarray(np.uint8(view.mask)*255).resize((128,128),Image.Resampling.NEAREST))>0
        updates.update({'mask_'+s:_readonly(view.mask[None],np.float32),
            'coarse_mask_'+s:_readonly(coarse[None],np.float32),
            'points_rc_'+s:view.points,'contour_valid_'+s:view.valid})
    ta,tb=inherit_pair_targets(sample,views['a'],views['b'])
    if sample.label and (ta>=0).sum()<4:
        return None,dict(reason='fewer_than_four_inherited_correspondences'),fraction,None
    updates.update(target_a=_readonly(ta,np.int64),target_b=_readonly(tb,np.int64))
    out=replace(sample,**updates)
    if area_ratio(out)<min(.125,area_ratio(sample)):
        return None,dict(reason='weather_created_extreme_area_ratio'),fraction,None
    return out,details,fraction,{side+'_'+key:value for key,value in arrays.items()}


def weather_group(current,recipe,rng,profile,cut_guards,scale_info):
    rejected=Counter();last=None;scale=scale_info[0]['common_scale']
    for attempt in range(int(profile.get('damage_plan_attempts',32))):
        plan=dict(recipe=recipe,side=str(rng.choice(['a','b'])),
            k=int(rng.integers(1,4)) if parts(recipe)[0]=='gaps' else 1,
            cut_guards=cut_guards,guard_radius=max(6.,8.*scale),contact_tolerance=max(3.,3.*scale))
        pos,dp,fraction,pa=damaged_pair(current[0],rng,plan)
        if pos is None:
            last=dict(stage='positive_damage',detail=dp);rejected['positive:'+dp['reason']]+=1;continue
        neg,dn,_,na=damaged_pair(current[1],rng,plan,fraction)
        if neg is None:
            last=dict(stage='negative_damage',detail=dn);rejected['negative:'+dn['reason']]+=1;continue
        # Transient numerical evidence is removed before JSON serialization.
        dp['_arrays']=pa;dn['_arrays']=na
        return (pos,neg),[dp,dn],dict(attempts=attempt+1,rejected=dict(rejected),
            selected_endpoint=plan['side'],requested_gap_count=plan['k'] if 'gaps' in recipe else 0),None
    return None,None,dict(attempts=attempt+1,rejected=dict(rejected)),last
