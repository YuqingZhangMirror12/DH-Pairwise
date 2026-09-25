"""v13: source structure/scale/mirror first, then one exclusive damage recipe.

Partial is a damage recipe, NEVER an independent Boolean enhancement. Its sole
exception is the final1-3px light layer, not primary wave/local/notch weather.
Historical v12 generation is left on its original explicit branch.
"""
from collections import Counter,defaultdict
from dataclasses import replace
import numpy as np

from ..s7_compound_v1.geometry import rng_for, partial_group, masks
from ..seam_context_v3.augmentation import paired_mirror, mirror_schedule
from ..distribution_audit_20260923.measure import outline, pair_metrics
from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import changed_report
from .conservative_weather import parts, weather_group
from .scale import pair_shared_scale


SCHEMA='s7-layered-damage/13'


def canonical_mirrored_contours(sample):
    """Reindex reflected loops CCW for source-arc inheritance, preserving edges.

The model can canonicalize winding later, but offline weather inheritance runs
first and rejects clockwise arc projections. Coordinates and masks do NOT change
here; both endpoints' target indices are permuted by the exact inverse maps.
"""
    from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import _cartesian_signed_area
    changes={};permutations={};inverse={}
    for side in 'ab':
        points=getattr(sample,'points_rc_'+side);valid=getattr(sample,'contour_valid_'+side)
        ids=np.flatnonzero(valid);order=np.arange(len(points))
        if _cartesian_signed_area(points[ids])<0:order[ids]=ids[::-1]
        permutations[side]=order;inverse[side]=np.argsort(order)
        changes['points_rc_'+side]=np.ascontiguousarray(points[order])
        changes['contour_valid_'+side]=np.ascontiguousarray(valid[order])
    for side,other in (('a','b'),('b','a')):
        target=getattr(sample,'target_'+side)[permutations[side]].copy();matched=target>=0
        target[matched]=inverse[other][target[matched]];changes['target_'+side]=target
    return replace(sample,**changes)


def pilot_sources(plan,n,seed):
    """Exact pilot negative strata, retaining complete three-negative anchors.

400 negatives cannot split half into triples: use198(66 complete groups),
not200 incomplete anchors. The full12K source plan is never changed.
"""
    if n==len(plan['positive']):return plan['positive'],plan['negative']
    if n!=400:raise ValueError('layered review pilot must be400 groups')
    rng=rng_for(seed,SCHEMA,'source-pilot')
    positive=[]
    for key,count in (('native_positive',300),('gen5_partition_positive',40),('union_positive_tiny',60)):
        pool=[e for e in plan['positive'] if e['source_stratum']==key]
        positive.extend(pool[i] for i in rng.permutation(len(pool))[:count])
    quotas={'cross_gen':140,'cross_parent_same_gen':140,'same_parent_nonadjacent':120}
    negative=[];gen5=[e for e in plan['negative'] if e['mode']=='gen5']
    keys=list(quotas);counts=_allocate(40,[sum(e['kind']==k for e in gen5) for k in keys])
    for k,count in zip(keys,counts):
        pool=[e for e in gen5 if e['kind']==k]
        chosen=[pool[i] for i in rng.permutation(len(pool))[:count]]
        negative.extend(chosen);quotas[k]-=len(chosen)
    anchors=defaultdict(list)
    for e in plan['negative']:
        if e.get('anchor_group_id'):anchors[e['anchor_group_id']].append(e)
    groups=list(anchors.values());kept=0
    for i in rng.permutation(len(groups)):
        group=groups[i];cost=Counter(e['kind'] for e in group)
        if len(group)!=3 or any(cost[k]>quotas[k] for k in cost):continue
        negative.extend(group)
        for k in cost:quotas[k]-=cost[k]
        kept+=1
        if kept==66:break
    if kept!=66:raise ValueError('not enough complete negative anchor groups')
    for k,count in quotas.items():
        pool=[e for e in plan['negative'] if e['kind']==k and e['mode']=='native' and not e.get('anchor_group_id')]
        if len(pool)<count:raise ValueError('negative pilot pool exhausted')
        negative.extend(pool[i] for i in rng.permutation(len(pool))[:count])
    assert len(positive)==len(negative)==400
    assert len({e['pair_id'] for e in negative})==400
    return ([positive[i] for i in rng.permutation(n)],[negative[i] for i in rng.permutation(n)])


def corrosion_layer(recipe):
    if recipe=='partial':return 'partial'
    if recipe=='clean':return 'none'
    if recipe=='mild':return 'weak'
    if recipe.startswith('gaps'):return 'notch'
    if recipe.startswith(('wave','local_')):return 'strong'
    raise ValueError('unregistered damage recipe: '+recipe)


def exact_counts(proportions,n):
    if abs(sum(proportions.values())-100)>1e-8:
        raise ValueError('exclusive damage percentages must sum to100')
    result={key:int(round(n*value/100)) for key,value in proportions.items()}
    if sum(result.values())!=n or any(abs(result[k]-n*v/100)>1e-8 for k,v in proportions.items()):
        raise ValueError('population cannot exactly represent requested quotas')
    return result


def _allocate(n,weights):
    raw=n*np.asarray(weights,float)/sum(weights);counts=np.floor(raw).astype(int)
    order=np.argsort(-(raw-counts),kind='stable')
    counts[order[:n-int(counts.sum())]]+=1
    return counts


def layered_schedules(n,seed,profile):
    if not profile.get('layered_damage_exclusive'):
        raise ValueError('explicit layered profile required')
    counts=exact_counts(profile['corrosion_percent'],n)
    if counts.get('partial',0)!=round(n*profile['partial_percent']/100):
        raise ValueError('Partial is a mutually exclusive recipe, not an independent axis')
    rng=rng_for(seed,SCHEMA,'schedule')
    recipes=np.array([k for k,v in counts.items() for _ in range(v)],object);rng.shuffle(recipes)
    partial=recipes=='partial'
    # Preserve the overall25/50/25 length strata. Partial retains the older
    # conditional short/medium/long25:40:5 allocation (integer rounding exposed).
    # No Partial crop is secretly applied to weather cases to fill short slots.
    totals=_allocate(n,[25,50,25]);pcounts=_allocate(int(partial.sum()),[25,40,5])
    pcounts=np.minimum(pcounts,totals)
    for _ in range(int(partial.sum())-int(pcounts.sum())):
        pcounts[int(np.argmax(totals-pcounts))]+=1
    bins=np.empty(n,object)
    for flag,numbers in ((True,pcounts),(False,totals-pcounts)):
        ids=rng.permutation(np.flatnonzero(partial==flag))
        values=[key for key,count in zip(('short','medium','long'),numbers) for _ in range(count)]
        if len(values)!=len(ids):raise ValueError('joint length allocation mismatch')
        bins[ids]=values
    mirrors=mirror_schedule(n,profile.get('mirror_percent',15)/100,seed,0)
    return recipes.tolist(),partial.tolist(),bins.tolist(),mirrors.tolist()


def augment_layered_group(positive,negative,recipe,partial,bank,seed,profile,
                          target_area=None,quota_bin=None,mirror_axis=None,partial_mode=None):
    if (not positive.label or negative.label or recipe not in profile['corrosion_percent']
            or bool(partial)!=(recipe=='partial')):
        raise ValueError('Partial cannot coexist with weak, strong, notch or clean recipes')
    rng=rng_for(seed,SCHEMA,positive.pair_id,negative.pair_id,recipe)
    weights=np.asarray(profile['area_bin_probabilities'],float)
    k=int(rng.choice(len(weights),p=weights/weights.sum()))
    bins=profile['area_bins_px2']
    area=float(rng.uniform(bins[k],bins[k+1])) if target_area is None else float(target_area)
    try:
        if profile.get('preserve_original_scale_for_heldout'):
            # Original S7 heldout pairing/scale, with the new damage only.
            # Never apply this switch implicitly to the approved TRAIN profile.
            transformed=[(s,dict(common_scale=1.,requested_scale=1.,
                requested_mean_area_px2=float(np.mean([m.sum() for m in masks(s)])),
                before_mean_area_px2=float(np.mean([m.sum() for m in masks(s)])),
                after_mean_area_px2=float(np.mean([m.sum() for m in masks(s)])),
                offsets_rc=[[0.,0.],[0.,0.]],same_scale_for_both_fragments=True,
                heldout_original_geometry_unchanged=True)) for s in (positive,negative)]
        else:
            transformed=[pair_shared_scale(s,area,topology_backoff=profile.get('scale_topology_backoff',False),
                identity_fallback=profile.get('scale_identity_fallback',False)) for s in (positive,negative)]
    except ValueError as error:
        return None,dict(stage='fragment_scale',detail=dict(reason=str(error)))
    current=tuple(x[0] for x in transformed);scale_info=[x[1] for x in transformed]
    if mirror_axis:
        current=tuple(canonical_mirrored_contours(paired_mirror(s,mirror_axis)) for s in current)
    fragment_base=current
    partial_info=None;partial_arrays=[{},{}]
    if partial:
        if profile.get('partial_min_smaller_perimeter_fraction'):
            from .partial_v14 import partial_group_v14
            current,partial_info=partial_group_v14(*current,bank,rng,profile,partial_mode)
        else:current,partial_info=partial_group(*current,bank,rng)
        if current is None:return None,dict(stage='partial_only',detail=partial_info)
        partial_arrays=partial_info.pop('_arrays',[{},{}])
    # For Partial use its retained seam for the length quota. For all weather
    # types, this is still the untouched seam after fragment-level transforms.
    support_base=current
    length=pair_metrics(*[outline(m) for m in masks(current[0])],current[0].translation_a_to_b_rc)['d20_length_px']
    actual='short' if length<256 else 'medium' if length<512 else 'long'
    from .geometry import length_quota_applies
    if length_quota_applies(profile,partial) and quota_bin and (not 32<=length<=800 or actual!=quota_bin):
        return None,dict(stage='layered_length_quota',detail=dict(reason='length_outside_quota'))
    guards={s:np.empty((0,2),float) for s in 'ab'}
    details=[{},{}];weather_plan=None
    if recipe not in ('clean','partial'):
        # This is the ONLY call site for weather; Partial never reaches it.
        current,details,weather_plan,failure=weather_group(current,recipe,rng,profile,guards,scale_info)
        if current is None:return None,failure
    micro_details=[{},{}];micro_arrays=[None,None]
    if recipe!='clean' and profile.get('background_untouched_fraction'):
        from .background_recession import group as background_group
        current,micro_details,micro_arrays,failure=background_group(fragment_base,current,
            [d.get('_arrays',{}) for d in details],rng,profile)
        if current is None:return None,failure
    major,weak=parts(recipe)
    types=['partial'] if partial else ([major] if major else [])+(['smooth_weak'] if weak else [])
    if partial and (major is not None or weak):raise AssertionError('Partial weather leakage')
    result=[]
    for i,(before,new,detail) in enumerate(zip(fragment_base,current,details)):
        arrays=detail.pop('_arrays',None)
        report=changed_report(before,new,recipe)
        report['schema_version']=SCHEMA
        report['background_degradation']=micro_details[i]
        report['compound']=dict(recipe=recipe,corrosion_category=corrosion_layer(recipe),
            corrosion_types=types,corrosion_stage_count=len(types),partial=partial_info,
            damage=detail,stages_applied=len(types),inherited_target_count=int((new.target_a>=0).sum()),
            label_coupled_acceptance=True,major_corrosion_count=int(major is not None),
            weak_overlay=weak,major_type=major,no_combined_major_corrosion=True,
            partial_exclusive=True,corrosion_input='seamless_post_fragment_augmentation',
            quota_length_px=float(length) if i==0 else None,
            quota_length_basis='retained_after_partial' if partial else 'seamless_before_weather')
        if weather_plan is not None:report['compound']['weather_plan_selection']=weather_plan
        report['augmentation_layers']=dict(
            order=['source_structure','common_scale','paired_mirror','exclusive_damage'],
            fragments=dict(offline_paired_mirror=mirror_axis,source_pair_id=before.pair_id,
                common_scale=scale_info[i]['common_scale']),
            corrosion=dict(recipe=recipe,category=corrosion_layer(recipe),partial=partial,weak_overlay=weak,
                background_light=bool(micro_details[i])),
            no_primary_weather_after_partial=True,mirror_before_damage=True,
            contour_winding_canonicalized_before_damage=bool(mirror_axis))
        report['distribution_profile']=profile['name'];report['pair_shared_scale']=scale_info[i]
        report['_weather_arrays']=dict(arrays or {},**partial_arrays[i])
        report['_micro_weather_arrays']=micro_arrays[i]
        report['_offline_mirror_applied_before_damage']=True
        report['_weather_coordinate_frame']='post_fragment_pre_damage_800px'
        if new.label and profile.get('record_latent_seam'):
            from .latent_seam import source_band,measure
            scale=scale_info[0]['common_scale']
            bands=source_band(support_base[0],guards,max(6.,8.*scale),max(3.,3.*scale),
                profile.get('bridge_short_gaps_px',0.))
            latent,latent_arrays=measure(support_base[0],new,bands)
            if latent is None:return None,dict(stage='latent_seam',detail=dict(reason='no_supported_source_band'))
            latent['layered_basis']='retained_seam_after_partial' if partial else 'seamless_before_weather'
            report['latent_seam']=latent;report['_latent_arrays']=latent_arrays
        for side in 'ab':
            info=detail.get(side,{})
            report['side_'+side].update({k:v for k,v in info.items() if k not in ('removed_area_px','removed_fraction')})
            report['side_'+side]['weather_only_removed_area_px']=info.get('removed_area_px',0)
        report['fragment_stage_area_px']={s:int(getattr(before,'mask_'+s).sum()) for s in 'ab'}
        report['damage_stage_area_px']={s:int(getattr(new,'mask_'+s).sum()) for s in 'ab'}
        result.append((new,report,before))
    return result,dict(applied=True,layered=True)
