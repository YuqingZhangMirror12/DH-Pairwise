"""Compound damage in the unchanged pixel frame, with inherited supervision.

Positive damage is located using original TRAIN seam annotations. Negative
damage uses random arcs of its own contour, never a fabricated correspondence.
New notch/cut descendants are ignored, not matched by nearest-neighbour geometry.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import json

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from staging.pairwise_v0_2.pairwise_data.rachel_guided_partial_dataset import (
    guided_curve_mask, material_matched_curve_mask)
from staging.pairwise_v0_2.pairwise_data.rachel_partial_seam_dataset import (
    PartialSeamConfig, crop_training_pair, source_seam_context,
    _ignore_ambiguous_nonmatches)
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour
from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import _changed_view, changed_report
from staging.pairwise_v0_2.pairwise_data.rachel_strong_weathering import (
    _seam_eligible, _eligible_runs, _place_bumps, _bump_profile,
    _signed_arc_distance)
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import _readonly
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import inherit_pair_targets

SCHEMA = 's7-compound-training/1'
EIGHT = np.ones((3, 3), bool)
RECIPE_PERCENT = {
    'clean_anchor': 10,
    'seam_local_deep': 10,
    'partial_curve': 20,
    'wave_gaps': 10,
    'partial_wave': 25,
    'partial_wave_gaps': 25,
}
# Joint, not independent quotas: crop-based recipes supply short contacts;
# a local notch must not be forced to remove most of an intact long seam.
RECIPE_LENGTH_PERCENT = {
    'clean_anchor': {'long': 10},
    'seam_local_deep': {'long': 10},
    'partial_curve': {'short': 10, 'medium': 10},
    'wave_gaps': {'medium': 5, 'long': 5},
    'partial_wave': {'short': 10, 'medium': 15},
    'partial_wave_gaps': {'short': 5, 'medium': 20},
}


def rng_for(seed, *keys):
    payload = json.dumps([SCHEMA, seed, keys], separators=(',', ':'), sort_keys=True)
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], 'big'))


def schedule(n, seed):
    if n <= 0 or n % 100:
        raise ValueError('balanced group count must be a positive multiple of100')
    recipes = [key for key, p in RECIPE_PERCENT.items() for _ in range(n*p//100)]
    return np.asarray(recipes, dtype=object)[rng_for(seed, 'schedule').permutation(n)].tolist()


def length_schedule(recipes, seed):
    n = len(recipes)
    result = [None]*n
    for recipe, quotas in RECIPE_LENGTH_PERCENT.items():
        slots = [i for i, r in enumerate(recipes) if r == recipe]
        bins = [b for b, percent in quotas.items() for _ in range(n*percent//100)]
        if len(slots) != len(bins):
            raise ValueError('joint recipe/length quota mismatch')
        bins = np.asarray(bins)[rng_for(seed, 'recipe-length', recipe).permutation(len(bins))]
        for slot, b in zip(slots, bins):
            result[slot] = str(b)
    return result


def masks(sample):
    return [np.asarray(getattr(sample, 'mask_'+s)[0], bool) for s in 'ab']


def area_ratio(sample):
    a, b = (int(m.sum()) for m in masks(sample))
    return min(a, b)/max(a, b)


def partial_group(positive, negative, bank, rng, max_attempts=24):
    """One oblique TRAIN-donor curve; coupled morphology and acceptance.

Often crop the larger endpoint to avoid turning every partial seam into a tiny
fragment. The selected area rank is shared across positive/negative members.
The foreground-retention and new-cut exclusion checks stay explicit.
"""
    config = PartialSeamConfig(retention_min=.25, retention_max=.65).crop_config
    pair = (positive, negative)
    reasons = Counter()
    choose_large = bool(rng.random() < .7)
    sides = []
    for sample in pair:
        a, b = (m.sum() for m in masks(sample))
        small = 'a' if a <= b else 'b'
        sides.append(('b' if small == 'a' else 'a') if choose_large else small)
    context = source_seam_context(positive, sides[0], config)
    if context is None:
        return None, dict(reason='no_source_seam', attempts=0)
    for attempt in range(max_attempts):
        donor_index = int(rng.integers(len(bank.profiles)))
        profile = bank.profiles[donor_index]
        params = dict(angle_deg=float(rng.uniform(15., 75.)+90*int(rng.integers(2))),
            keep_low=bool(rng.integers(2)), flip=bool(rng.integers(2)), reverse=bool(rng.integers(2)))
        target = float(rng.uniform(.25, .65))
        try:
            pm, pr = guided_curve_mask(getattr(positive, 'mask_'+sides[0])[0], profile,
                points=context['points'], weights=context['weights'], target_retention=target, **params)
            nm, nr = material_matched_curve_mask(getattr(negative, 'mask_'+sides[1])[0], profile,
                target_retention=pr['material_retention'], **params)
        except ValueError:
            reasons['unsolvable_curve'] += 1
            continue
        if not .25 <= pr['actual_retention'] <= .65:
            reasons['source_retention_outside_range'] += 1
            continue
        results = [crop_training_pair(s, side=side, retained_mask=m,
                    config=config, topology_connectivity=2)
                   for s, side, m in zip(pair, sides, (pm, nm))]
        if not all(r.accepted for r in results):
            for ordinal, r in enumerate(results):
                if not r.accepted:
                    reasons[('positive:' if ordinal == 0 else 'negative:')+r.reason] += 1
            continue
        outputs = tuple(_ignore_ambiguous_nonmatches(old, r.sample, side, None)
                        for old, r, side in zip(pair, results, sides))
        if any(area_ratio(new) < min(.125, area_ratio(old)) for old, new in zip(pair, outputs)):
            reasons['would_create_new_extreme_area_ratio'] += 1
            continue
        donor = bank.metadata['arcs'][donor_index]
        return outputs, dict(applied=True, attempts=attempt+1, sides=sides,
            choose_large=choose_large, source_seam_retention=pr['actual_retention'],
            source_seam_length_px=context['length'],
            material_retention=[pr['material_retention'], nr['material_retention']],
            proposal=params, donor_index=donor_index, donor_split=donor.get('split','train'),
            donor_lineage=donor['lineage'], donor_family=donor['family'],
            artificial_cut_exclusion_px=8., attempt_reasons=dict(reasons))
    return None, dict(applied=False, reason='partial_rejected', attempts=max_attempts,
                     attempt_reasons=dict(reasons))


def eligible_seam(sample, side, dense, dense_weather_scope=False, *,
                  cut_guard=None, cut_guard_radius=8., contact_tolerance=3.,bridge_short_gaps_px=0.):
    if dense_weather_scope:
        # Placement only: dense geometric contact corroborated by an existing
        # reciprocal TRAIN GT edge. Do not require raster indices to advance
        # exactly +1/-1, and never create new correspondence supervision here.
        from staging.pairwise_v0_2.pairwise_data.rachel_size_crop import _original_gt_edges,_gt_supported_dense
        edges=_original_gt_edges(sample)
        if not len(edges):return np.zeros(len(dense),bool)
        other='b' if side=='a' else 'a'
        q,_=extract_ordered_outer_contour(getattr(sample,'mask_'+other)[0],cap=sample.mask_a.size,smoothing_sigma=0.)
        q=np.asarray(q,float);p=np.asarray(dense,float)
        shift=sample.translation_a_to_b_rc if side=='a' else -sample.translation_a_to_b_rc
        distance,j=cKDTree(q).query(p+shift)
        def normals(x):
            smooth=ndimage.gaussian_filter1d(x,3,axis=0,mode='wrap')
            t=np.roll(smooth,-4,axis=0)-np.roll(smooth,4,axis=0)
            n=np.c_[-t[:,1],t[:,0]]
            return n/np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8)
        good=(distance<=contact_tolerance)&((normals(p)*normals(q)[j]).sum(1)<=-.25)
        ids=np.flatnonzero(good)
        pairs=np.c_[ids,j[ids]] if side=='a' else np.c_[j[ids],ids]
        pa=getattr(sample,'points_rc_'+side)[getattr(sample,'contour_valid_'+side)]
        spacing=np.linalg.norm(np.roll(pa,-1,axis=0)-pa,axis=1)
        radius=float(np.clip(3*np.median(spacing),10,24))
        supported=_gt_supported_dense(pairs,p if side=='a' else q,q if side=='a' else p,sample,edges,radius)
        bits=np.zeros(len(p),bool)
        if len(supported):bits[supported[:,0 if side=='a' else 1]]=True
        if bridge_short_gaps_px>0 and bits.any():
            # A sparse GT-anchor / raster-normal interruption is not a true
            # end of the old seam. Bridge only short arcs bounded by supported
            # positions and geometrically touching the other TRAIN fragment.
            # This is a weather-placement corridor, never new match labels.
            edge=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
            arc=np.r_[0.,np.cumsum(edge[:-1])];perimeter=edge.sum()
            for start,length,circular in _eligible_runs(~bits,arc,edge):
                if circular or length>bridge_short_gaps_px:continue
                inside=(arc-start)%perimeter<=length+1e-6
                if np.all(distance[inside]<=contact_tolerance+2.):bits[inside]=True
        if cut_guard is not None:
            # Target -2 also denotes a *real old seam* whose sparse token
            # correspondence is ambiguous after resampling. It is not proof
            # of an artificial cut. For placement, use the recorded material
            # deletion boundary; supervision still retains all -2 targets.
            cut_guard=np.asarray(cut_guard,float).reshape(-1,2)
            if len(cut_guard):bits &= cKDTree(cut_guard).query(p)[0]>cut_guard_radius
        else:
            # Historical recipe behavior is retained unless the caller has
            # explicit cut provenance in this coordinate frame.
            ignored=getattr(sample,'points_rc_'+side)[getattr(sample,'target_'+side)==-2]
            if len(ignored):bits &= cKDTree(ignored).query(p)[0]>6.
        return bits
    context = source_seam_context(sample, side, PartialSeamConfig().crop_config)
    if context is None:
        return np.zeros(len(dense), bool)
    return _seam_eligible(dense, context['points'], None, 1.5, 8.)


def arc_taper(bits, arc, edge, width=12.):
    """Cosine shoulders within supported source arcs, not outside the seam."""
    result = np.zeros(len(bits), float)
    perimeter = edge.sum()
    for start, length, circular in _eligible_runs(bits, arc, edge):
        u = (arc-start) % perimeter
        inside = u <= length+1e-6
        if circular:
            result[:] = 1.
        else:
            ramp = np.clip(np.minimum(u, length-u)/width, 0., 1.)
            result[inside] = np.maximum(result[inside], .5-.5*np.cos(np.pi*ramp[inside]))
    return result


def source_survival_window(dense, arc, perimeter, points, margin=4.):
    """Enclose existing TRAIN anchors on the original contour, not new labels.

    The short circular enclosure is invariant to contour starting index. The
    caller may reserve this near-contact island while eroding the rest deeply.
    """
    points=np.asarray(points,float).reshape(-1,2)
    if not len(points):raise ValueError('empty survival guide')
    values=np.sort(arc[np.unique(cKDTree(dense).query(points)[1])])
    gap=np.diff(np.r_[values,values[0]+perimeter])
    cut=int(np.argmax(gap));start=values[(cut+1)%len(values)]
    length=perimeter-float(gap[cut])
    return dict(center_arc_px=float((start+length/2)%perimeter),
                core_width_px=float(length+2*margin),anchor_count=len(points))


def weather_fragment(mask, rng, *, eligible, wave, notch_widths, notch_peaks,
                     wave_share=1., negative=False, max_attempts=12, wave_range=(3.,10.),
                     heterogeneous_strong=False, survival_points=None, survival_width=None):
    """One compound depth field: broad wavy gaps plus 10--30px notches.

Depths combine by max, not addition, so a compound example stays within30px.
The broad seam recession has a 3--10px envelope, not an unchanged 2px edge.
Its peak can be10px while deep local events reach10--30px independently.
"""
    mask = np.asarray(mask, bool)
    dense, _ = extract_ordered_outer_contour(mask, cap=mask.size, smoothing_sigma=0.)
    dense = np.asarray(dense, float)
    edge = np.linalg.norm(np.roll(dense, -1, axis=0)-dense, axis=1)
    perimeter = float(edge.sum()); arc = np.r_[0., np.cumsum(edge[:-1])]
    if eligible is None:
        eligible = np.ones(len(dense), bool)
    if eligible.shape != (len(dense),) or not eligible.any():
        return None, dict(reason='no_eligible_seam')
    if ndimage.label(mask, EIGHT)[1] != 1:
        return None, dict(reason='original_disconnected')
    # Holes are not extra erosion seeds. Filling is used ONLY for distance.
    filled = ndimage.binary_fill_holes(np.pad(mask, 1), structure=EIGHT)
    inward = ndimage.distance_transform_edt(filled)[1:-1, 1:-1]-.5
    band = np.argwhere(mask & (inward <= 30.))
    nearest = cKDTree(dense).query(band)[1]
    depth_inside = inward[band[:, 0], band[:, 1]]
    reasons = Counter()
    runs = _eligible_runs(np.ones(len(dense), bool) if negative else eligible, arc, edge)
    taper = arc_taper(eligible, arc, edge)
    for attempt in range(max_attempts):
        near_windows=[]
        survival=None;notch_runs=runs
        if survival_points is not None or survival_width is not None:
            if not (heterogeneous_strong and wave):raise ValueError('survival island requires strong wave')
            if survival_points is not None:
                survival=source_survival_window(dense,arc,perimeter,survival_points)
            else:
                # Negatives receive the same physical near-island width on a
                # random own-contour arc, never positive-pair coordinates.
                width=float(survival_width)
                placement=_place_bumps([width+16.],_eligible_runs(eligible,arc,edge),perimeter,rng,0.)
                if placement is None:
                    reasons['survival_island_does_not_fit']+=1;continue
                survival=dict(center_arc_px=float(placement[0]),core_width_px=width,anchor_count=0)
            distance=np.abs(_signed_arc_distance(arc,survival['center_arc_px'],perimeter))
            core=distance<=survival['core_width_px']/2
            if not core.any() or not np.all(eligible[core]):
                reasons['survival_island_outside_supported_arc']+=1;continue
            protected=distance<=survival['core_width_px']/2+8.
            notch_runs=_eligible_runs((np.ones(len(dense),bool) if negative else eligible)&~protected,arc,edge)
        if wave:
            count = max(8, int(np.ceil(perimeter)))
            noise = ndimage.gaussian_filter1d(rng.normal(size=count),
                sigma=float(rng.uniform(30., 80.)), mode='wrap')
            profile = np.interp(arc, np.linspace(0., perimeter, count+1), np.r_[noise, noise[0]])
            scaled = (profile-profile.min())/max(float(np.ptp(profile)), 1e-9)
            base = (wave_range[0]+(wave_range[1]-wave_range[0])*scaled)*taper*wave_share
            if heterogeneous_strong:
                # Normalize over the *eligible* old seam, not unrelated outer
                # edges. Most supported arcs recede 10--30px; a smooth valley
                # keeps a few-pixel portion on that same potentially joinable arc.
                active=np.flatnonzero(eligible)
                lo,hi=np.quantile(profile[active],[.03,.97])
                scaled=np.clip((profile-lo)/max(hi-lo,1e-9),0,1)
                base=(wave_range[0]+(wave_range[1]-wave_range[0])*scaled)*taper
                for start,length,circular in _eligible_runs(eligible,arc,edge):
                    if survival is not None:continue
                    if length<32.:continue
                    width=float(rng.uniform(.18,.28)*length)
                    center=(start+length*rng.uniform(.3,.7))%perimeter
                    distance=np.abs(_signed_arc_distance(arc,center,perimeter))
                    # Flat near-contact core with cosine shoulders; never a
                    # rectangular cut or a uniform whole-fragment shrink.
                    transition=np.clip((distance-width*.28)/(width*.22),0,1)
                    gate=.5-.5*np.cos(np.pi*transition)
                    floor=float(rng.uniform(1.,4.))
                    base=np.where(distance<width/2.,floor*taper+(base-floor*taper)*gate,base)
                    near_windows.append(dict(center_arc_px=float(center),width_px=width,depth_px=floor))
                if survival is not None:
                    distance=np.abs(_signed_arc_distance(arc,survival['center_arc_px'],perimeter))
                    transition=np.clip((distance-survival['core_width_px']/2)/8.,0.,1.)
                    gate=.5-.5*np.cos(np.pi*transition);floor=float(rng.uniform(1.,4.))
                    base=floor*taper+(base-floor*taper)*gate
                    near_windows.append(dict(center_arc_px=survival['center_arc_px'],
                        width_px=survival['core_width_px']+16.,depth_px=floor,
                        placement='inherited_train_anchor_island' if not negative else 'matched_width_random_island'))
        else:
            base = np.zeros(len(dense))
        centers = _place_bumps(notch_widths, notch_runs, perimeter, rng, 5.) if len(notch_widths) else []
        if centers is None:
            reasons['requested_notches_do_not_fit'] += 1
            continue
        profiles = [_bump_profile(arc, c, w, d, perimeter)
                    for c, w, d in zip(centers, notch_widths, notch_peaks)]
        if heterogeneous_strong and wave and profiles:
            # Notches must add real loss on top of the wavy recession. Merely
            # taking max could label an entirely hidden notch as "applied".
            combined=np.minimum(30.,base+np.maximum.reduce(profiles))
        else:
            combined = np.maximum.reduce([base]+profiles)
        selector = depth_inside <= combined[nearest]
        new = mask.copy(); new[tuple(band[selector].T)] = False
        if not selector.any():
            reasons['no_change'] += 1; continue
        if ndimage.label(new, EIGHT)[1] != 1:
            reasons['disconnected'] += 1; continue
        if new.sum() < max(64., .25*mask.sum()):
            reasons['retained_area'] += 1; continue
        if np.any(ndimage.binary_fill_holes(new, structure=EIGHT) & ~new & mask):
            reasons['new_enclosed_hole'] += 1; continue
        actual = [float(depth_inside[selector & (p[nearest] > 0)].max(initial=0)) for p in profiles]
        extra_counts=[int(np.count_nonzero(selector & (p[nearest]>0) &
                      (depth_inside>base[nearest]))) for p in profiles]
        if heterogeneous_strong and any(v<8 for v in extra_counts):
            reasons['notch_not_independently_effective']+=1;continue
        if any(a < p-1.5 for a, p in zip(actual, notch_peaks)):
            reasons['requested_depth_unreached'] += 1; continue
        if wave and int(np.count_nonzero(selector & (base[nearest] > 1.) &
                    (base[nearest] >= np.maximum.reduce(profiles)[nearest] if profiles else True))) < 16:
            reasons['wave_not_independently_effective'] += 1; continue
        regions = []
        for center, width, peak, reached, extra in zip(centers, notch_widths, notch_peaks, actual,extra_counts):
            covered = np.abs(_signed_arc_distance(arc, center, perimeter)) <= width/2.+2.
            regions.append(dict(source_points_rc=dense[covered].tolist(),
                center_arc_px=float(center), support_length_px=float(width),
                requested_peak_depth_px=float(peak), applied_max_depth_px=reached,
                independently_removed_pixels=extra))
        return new, dict(applied=True, attempts=attempt+1, attempt_reasons=dict(reasons),
            wave=wave, wave_range_px=[float(v)*wave_share for v in wave_range] if wave else None,
            max_combined_depth_px=30., applied_max_depth_px=float(depth_inside[selector].max()),
            notch_count=len(regions), requested_peak_depths_px=list(map(float, notch_peaks)),
            requested_support_lengths_px=list(map(float, notch_widths)),
            removed_area_px=int(selector.sum()), removed_fraction=float(selector.sum()/mask.sum()),
            component_count=1, ignore_source_regions=regions,
            heterogeneous_strong=heterogeneous_strong,near_windows=near_windows,
            survival_island=survival,
            notch_combination='additive_capped_30px' if heterogeneous_strong and wave else 'maximum',
            notch_independently_removed_pixels=extra_counts,
            eligible_fraction=float(np.sum(edge[eligible])/perimeter),
            positive_scope='inherited_train_seam' if not negative else None,
            negative_scope='random_arcs_anywhere_on_own_outer_contour' if negative else None)
    return None, dict(applied=False, reason='weather_rejected', attempt_reasons=dict(reasons))


def damaged_pair(sample, rng, plan, eligible_fractions=None):
    views, details, fractions = {}, {}, {}
    for side in 'ab':
        mask = np.asarray(getattr(sample, 'mask_'+side)[0], bool)
        active = side in plan['endpoints']
        if active:
            dense, _ = extract_ordered_outer_contour(mask, cap=mask.size, smoothing_sigma=0.)
            edge = np.linalg.norm(np.roll(dense, -1, axis=0)-dense, axis=1)
            arc = np.r_[0., np.cumsum(edge[:-1])]; perimeter = edge.sum()
            if sample.label:
                guards=plan.get('positive_cut_guards')
                eligible = eligible_seam(sample, side, dense,plan.get('dense_weather_scope',False),
                    cut_guard=None if guards is None else guards[side],
                    cut_guard_radius=plan.get('positive_cut_guard_radius',8.),
                    contact_tolerance=plan.get('positive_contact_tolerance',3.),
                    bridge_short_gaps_px=plan.get('bridge_short_gaps_px',0.))
            else:
                fraction = float(np.clip(eligible_fractions[side], .03, .8))
                center = rng.uniform(0., perimeter)
                eligible = np.abs(_signed_arc_distance(arc, center, perimeter)) <= perimeter*fraction/2.
            fractions[side] = float(edge[eligible].sum()/perimeter)
            new, info = weather_fragment(mask, rng, eligible=eligible,
                wave=plan['wave'], notch_widths=plan['widths'][side],
                notch_peaks=plan['peaks'][side], wave_share=plan['wave_share'], negative=not sample.label,
                wave_range=plan.get('wave_range',(3.,10.)),
                heterogeneous_strong=plan.get('heterogeneous_strong',False),
                survival_points=plan.get('survival_guide_points',{}).get(side) if sample.label else None,
                survival_width=plan.get('negative_survival_widths',{}).get(side) if not sample.label else None)
            if new is None:
                return None, dict(reason=info.get('reason'), side=side, detail=info), fractions
        else:
            new, info = mask, dict(applied=False, reason='unselected_endpoint')
        recipe = 'local' if info.get('ignore_source_regions') else 'wave'
        views[side] = _changed_view(sample, side, new, info, recipe)
        details[side] = info
    ta, tb = inherit_pair_targets(sample, views['a'], views['b'])
    if sample.label and int((ta >= 0).sum()) < 4:
        return None, dict(reason='fewer_than_four_inherited_correspondences'), fractions
    updates = dict(target_a=_readonly(ta, np.int64), target_b=_readonly(tb, np.int64))
    for side, view in views.items():
        nearest = getattr(Image, 'Resampling', Image).NEAREST
        coarse = np.asarray(Image.fromarray(np.uint8(view.mask)*255).resize((128, 128), nearest)) > 0
        updates.update({'mask_'+side:_readonly(view.mask[None], np.float32),
            'coarse_mask_'+side:_readonly(coarse[None], np.float32),
            'points_rc_'+side:view.points, 'contour_valid_'+side:view.valid})
    output = replace(sample, **updates)
    if area_ratio(output) < min(.125, area_ratio(sample)):
        return None, dict(reason='weather_created_extreme_area_ratio'), fractions
    return output, details, fractions


def augment_group(positive, negative, recipe, bank, seed):
    if not positive.label or negative.label or recipe not in RECIPE_PERCENT:
        raise ValueError('one TRAIN positive/negative group and a registered recipe required')
    rng = rng_for(seed, positive.pair_id, negative.pair_id, recipe)
    originals = (positive, negative)
    current = originals
    partial = None
    if recipe.startswith('partial'):
        current, partial = partial_group(*current, bank, rng)
        if current is None:
            return None, dict(stage='partial', detail=partial)
    preweather = current
    weather = recipe not in ('clean_anchor', 'partial_curve')
    details = [{}, {}]
    if weather:
        endpoints = ('a', 'b', 'ab')[int(rng.choice(3, p=[.4, .4, .2]))]
        wave = 'wave' in recipe
        gaps = 'gaps' in recipe or recipe == 'seam_local_deep'
        k = int(rng.integers(1, 6)) if 'gaps' in recipe else (int(rng.integers(1, 4)) if gaps else 0)
        widths = rng.uniform(15., 50., size=k) if 'gaps' in recipe else rng.uniform(30., 90., size=k)
        peaks = rng.uniform(10., 30., size=k)
        assignment = rng.choice(list(endpoints), size=k)
        plan = dict(endpoints=endpoints, wave=wave, wave_share=1./len(endpoints),
            widths={s:widths[assignment == s].tolist() for s in 'ab'},
            peaks={s:peaks[assignment == s].tolist() for s in 'ab'})
        if not wave:
            plan['endpoints'] = ''.join(s for s in endpoints if len(plan['widths'][s]))
        pos, dp, fractions = damaged_pair(current[0], rng, plan)
        if pos is None:
            return None, dict(stage='positive_damage', detail=dp)
        neg, dn, _ = damaged_pair(current[1], rng, plan, fractions)
        if neg is None:
            return None, dict(stage='negative_damage', detail=dn)
        current, details = (pos, neg), (dp, dn)
    outputs = []
    for ordinal, (old, new, detail) in enumerate(zip(originals, current, details)):
        report = changed_report(old, new, recipe)
        report['schema_version'] = SCHEMA
        report['compound'] = dict(recipe=recipe, partial=partial, damage=detail,
            stages_applied=(int(partial is not None)+int('wave' in recipe)+int('gaps' in recipe or recipe=='seam_local_deep')),
            inherited_target_count=int((new.target_a >= 0).sum()), label_coupled_acceptance=True)
        for side in 'ab':
            info = detail.get(side, {})
            report['side_'+side].update({k:v for k,v in info.items()
                if k not in ('removed_area_px', 'removed_fraction')})
            if info:
                report['side_'+side]['weather_only_removed_area_px'] = info.get('removed_area_px', 0)
        outputs.append((new, report, preweather[ordinal]))
    return outputs, dict(applied=True)
