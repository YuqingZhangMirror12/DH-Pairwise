"""Separate corrosion quotas from structural partial-seam augmentation."""
from collections import Counter
import numpy as np
from ..s7_compound_v1.geometry import (
    rng_for, partial_group, damaged_pair, masks, EIGHT)
from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import changed_report
from ..seam_context_v3.augmentation import mirror_schedule

SCHEMA='s7-balanced-training/2'
CORROSION_PERCENT={'clean':30,'wave':15,'local_deep':10,'gaps':10,
                   'wave_gaps':20,'wave_local_gaps':15}


def length_quota_applies(profile,partial):
    return not profile or (profile.get('enforce_length_quotas',True) and
                           (not partial or profile.get('enforce_partial_length_quotas',True)))


def schedules(n,seed,profile=None):
    if n<=0 or n%200:
        raise ValueError('group count must be a positive multiple of200')
    if profile and profile.get('layered_damage_exclusive'):
        from .layered_geometry import layered_schedules
        return layered_schedules(n,seed,profile)
    rng=rng_for(seed,SCHEMA,'schedule')
    proportions=CORROSION_PERCENT if profile is None else profile['corrosion_percent']
    recipes=np.array([k for k,p in proportions.items() for _ in range(n*p//100)],object)
    rng.shuffle(recipes)
    partial=np.zeros(n,bool)
    for recipe in proportions:
        slots=np.flatnonzero(recipes==recipe)
        partial[rng.permutation(slots)[:len(slots)*7//10]]=True
    bins=np.empty(n,object)
    for enabled,parts in ((True,{'short':25,'medium':40,'long':5}),
                          (False,{'medium':10,'long':20})):
        slots=rng.permutation(np.flatnonzero(partial==enabled))
        values=[b for b,p in parts.items() for _ in range(n*p//100)]
        if len(slots)!=len(values):
            raise ValueError('partial/length quota mismatch')
        bins[slots]=values
    return recipes.tolist(),partial.tolist(),bins.tolist(),mirror_schedule(n,.15,seed,0).tolist()


def category(recipe):
    if recipe=='partial':return 'partial'
    if recipe=='mild':return 'weak_only'
    return 'clean' if recipe=='clean' else 'combined' if recipe in ('wave_gaps','wave_local','wave_local_gaps') else 'single'


def _placement_capacity(sample,profile,cut_guards,scale_info):
    """Placement-only geometry; never create or revise correspondence labels."""
    from ..s7_compound_v1.geometry import extract_ordered_outer_contour,eligible_seam,_eligible_runs
    result={};scale=scale_info[0]['common_scale']
    for side in 'ab':
        mask=getattr(sample,'mask_'+side)[0]
        dense,_=extract_ordered_outer_contour(mask,cap=mask.size,smoothing_sigma=0.)
        edge=np.linalg.norm(np.roll(dense,-1,axis=0)-dense,axis=1)
        arc=np.r_[0.,np.cumsum(edge[:-1])]
        eligible=eligible_seam(sample,side,dense,profile.get('dense_gt_weather_scope',False),
            cut_guard=cut_guards[side],cut_guard_radius=max(6.,8.*scale),
            contact_tolerance=max(3.,3.*scale),bridge_short_gaps_px=profile.get('bridge_short_gaps_px',0.))
        result[side]=(_eligible_runs(eligible,arc,edge),float(edge.sum()))
    return result


def _weather_group(current,recipe,rng,profile,cut_guards,scale_info):
    """Retry weather plans on a valid partial crop, not the crop itself.

    The old single-plan path is preserved by default. Conditional plans retain
    the same K/width/depth ranges and full positive+negative acceptance tests.
    A non-fitting plan is redrawn, never shrunk, clipped to fit, or called clean.
    """
    if profile and profile.get('conservative_v12'):
        from .conservative_weather import weather_group
        return weather_group(current,recipe,rng,profile,cut_guards,scale_info)
    limit=int((profile or {}).get('damage_plan_attempts',1))
    if not 1<=limit<=128:raise ValueError('damage_plan_attempts outside1..128')
    capacity=None
    if profile and profile.get('notch_placement_preflight'):
        if not profile.get('explicit_partial_cut_guard'):raise ValueError('preflight requires explicit cut guard')
        capacity=_placement_capacity(current[0],profile,cut_guards,scale_info)
    rejected=Counter();last_failure=None
    for attempt in range(limit):
        endpoints=('a','b','ab')[int(rng.choice(3,p=[.4,.4,.2]))]
        wave='wave' in recipe;local='local' in recipe;gaps='gaps' in recipe
        kl=int(rng.integers(1,4)) if local else 0
        kg=int(rng.integers(1,6)) if gaps else 0
        widths=np.r_[rng.uniform(30.,90.,kl),rng.uniform(15.,50.,kg)]
        peaks=rng.uniform(10.,30.,kl+kg)
        assignment=rng.choice(list(endpoints),size=kl+kg)
        plan=dict(endpoints=endpoints,wave=wave,wave_share=1./len(endpoints),
                  widths={s:widths[assignment==s].tolist() for s in 'ab'},
                  peaks={s:peaks[assignment==s].tolist() for s in 'ab'})
        if profile is not None and wave:
            plan['wave_range']=profile['wave_ranges_px'][int(rng.choice(len(profile['wave_ranges_px']),p=profile['wave_range_probabilities']))]
        if profile:
            plan['dense_weather_scope']=profile.get('dense_gt_weather_scope',False)
            plan['heterogeneous_strong']=profile.get('heterogeneous_strong',False)
            plan['bridge_short_gaps_px']=profile.get('bridge_short_gaps_px',0.)
            if plan['heterogeneous_strong']:plan['wave_share']=1.
        if profile and profile.get('explicit_partial_cut_guard'):
            scale=scale_info[0]['common_scale']
            plan.update(positive_cut_guards=cut_guards,
                positive_cut_guard_radius=max(6.,8.*scale),positive_contact_tolerance=max(3.,3.*scale))
        if profile and profile.get('preserve_inherited_near_island') and wave:
            # Select a short group of already-labelled reciprocal TRAIN anchors.
            # No new match is inferred across a notch or an artificial cut.
            sample=current[0];ids=np.flatnonzero(sample.target_a>=0)
            k=min(6,len(ids))
            if k<4:
                last_failure=dict(stage='damage_plan',detail=dict(reason='insufficient_source_anchors'))
                rejected['insufficient_source_anchors']+=1;continue
            windows=[np.take(ids,np.arange(i,i+k)%len(ids)) for i in range(len(ids))]
            span=np.array([np.linalg.norm(np.diff(sample.points_rc_a[v],axis=0),axis=1).sum() for v in windows])
            # Exclude windows crossing the opposite, nonseam part of the walk.
            options=np.flatnonzero(span<=min(120.,np.median(span)))
            if not len(options):
                last_failure=dict(stage='damage_plan',detail=dict(reason='no_compact_source_anchors'))
                rejected['no_compact_source_anchors']+=1;continue
            chosen=windows[int(rng.choice(options))]
            plan['survival_guide_points']={
                'a':sample.points_rc_a[chosen].tolist(),
                'b':sample.points_rc_b[sample.target_a[chosen]].tolist()}
        if not wave:plan['endpoints']=''.join(s for s in endpoints if plan['widths'][s])
        if capacity is not None:
            from ..s7_compound_v1.geometry import _place_bumps
            fits=True
            for side in plan['endpoints']:
                runs,perimeter=capacity[side]
                if not runs or (plan['widths'][side] and
                        _place_bumps(plan['widths'][side],runs,perimeter,rng,5.) is None):
                    fits=False;break
            if not fits:
                last_failure=dict(stage='damage_plan',detail=dict(reason='requested_notches_do_not_fit'))
                rejected['placement_infeasible']+=1;continue
        pos,dp,fractions=damaged_pair(current[0],rng,plan)
        if pos is None:
            last_failure=dict(stage='positive_damage',detail=dp)
            rejected['positive:'+str(dp.get('reason'))]+=1;continue
        negative_plan=dict(plan)
        if plan.get('survival_guide_points'):
            negative_plan['negative_survival_widths']={s:dp[s]['survival_island']['core_width_px']
                for s in plan['endpoints'] if dp[s].get('survival_island')}
        neg,dn,_=damaged_pair(current[1],rng,negative_plan,fractions)
        if neg is None:
            last_failure=dict(stage='negative_damage',detail=dn)
            rejected['negative:'+str(dn.get('reason'))]+=1;continue
        for detail in (dp,dn):
            for side in 'ab':
                detail[side]['notch_kinds']=np.array(['local']*kl+['gap']*kg,dtype=object)[assignment==side].tolist()
        return (pos,neg),[dp,dn],dict(attempts=attempt+1,max_attempts=limit,
            rejected=dict(rejected),conditional_on_retained_crop=limit>1),None
    return None,None,dict(attempts=limit,rejected=dict(rejected)),last_failure


def augment_group(positive,negative,recipe,partial,bank,seed,profile=None,target_area=None,quota_bin=None,mirror_axis=None,partial_mode=None):
    if profile and profile.get('layered_damage_exclusive'):
        from .layered_geometry import augment_layered_group
        return augment_layered_group(positive,negative,recipe,partial,bank,seed,profile,target_area,quota_bin,mirror_axis,partial_mode)
    proportions=CORROSION_PERCENT if profile is None else profile['corrosion_percent']
    if not positive.label or negative.label or recipe not in proportions:
        raise ValueError('registered recipe and positive/negative group required')
    rng=rng_for(seed,SCHEMA,positive.pair_id,negative.pair_id,recipe,partial)
    originals=(positive,negative);current=originals;partial_info=None
    if partial:
        current,partial_info=partial_group(*current,bank,rng)
        if current is None:
            return None,dict(stage='partial',detail=partial_info)
    cut_guards={s:np.empty((0,2),float) for s in 'ab'}
    if profile and profile.get('explicit_partial_cut_guard') and partial:
        from scipy import ndimage
        side=partial_info['sides'][0]
        old=getattr(positive,'mask_'+side)[0].astype(bool)
        kept=getattr(current[0],'mask_'+side)[0].astype(bool)
        cut_guards[side]=np.argwhere(kept & ndimage.binary_dilation(old & ~kept,
            structure=ndimage.generate_binary_structure(2,1))).astype(float)
        if not len(cut_guards[side]):
            raise ValueError('partial applied without a recorded cut boundary')
    scale_info=None
    if profile is not None:
        from .scale import pair_shared_scale
        bins=profile['area_bins_px2'];weights=np.asarray(profile['area_bin_probabilities'],float)
        k=int(rng.choice(len(weights),p=weights/weights.sum()))
        area=float(rng.uniform(bins[k],bins[k+1])) if target_area is None else float(target_area)
        try:
            transformed=[pair_shared_scale(s,area,topology_backoff=profile.get('scale_topology_backoff',False),
                identity_fallback=profile.get('scale_identity_fallback',False)) for s in current]
        except ValueError as exc:
            return None,dict(stage='scale',detail=dict(reason=str(exc)))
        current=tuple(x[0] for x in transformed);scale_info=[x[1] for x in transformed]
        for side,offset in zip('ab',scale_info[0]['offsets_rc']):
            cut_guards[side]=scale_info[0]['common_scale']*cut_guards[side]+offset
        # Report material deletion in the scaled partial frame, not translations
        # or resized pixels as if they were corrosion. Partial has its own ledger.
        originals=current
    preweather=current;details=[{},{}]
    if quota_bin and profile and profile.get('length_quota_basis')=='pre_weather_near_length':
        from ..distribution_audit_20260923.measure import outline,pair_metrics
        # Same acceptance predicate as the final materializer check, evaluated
        # before costly weathering. Per-attempt RNG seeds preserve accepted
        # samples; no quota, strength, or label is relaxed.
        length=pair_metrics(*[outline(m) for m in masks(current[0])],current[0].translation_a_to_b_rc)['d20_length_px']
        actual='short' if length<256. else 'medium' if length<512. else 'long'
        if not 32<=length<=800 or actual!=quota_bin:
            return None,dict(stage='preweather_quota',detail=dict(reason='length_outside_quota'))
    weather_plan=None
    if recipe!='clean':
        current,details,weather_plan,failure=_weather_group(current,recipe,rng,profile,cut_guards,scale_info)
        if current is None:return None,failure
    result=[]
    for i,(old,new,detail) in enumerate(zip(originals,current,details)):
        weather_arrays=detail.pop('_arrays',None)
        report=changed_report(old,new,recipe)
        if profile is not None and partial:
            for side in 'ab':
                if side==partial_info['sides'][i]:report['changed_'+side]=True
            report['changed_pair']=True;report['pose_supervision_enabled']=False
        report['schema_version']=SCHEMA
        types=[] if recipe=='clean' else [k for k,present in
              (('wave','wave' in recipe),('local','local' in recipe),('gaps','gaps' in recipe)) if present]
        if profile and profile.get('conservative_v12'):
            from .conservative_weather import parts
            major,weak=parts(recipe)
            types=([major] if major else [])+(['smooth_weak'] if weak else [])
        report['compound']=dict(recipe=recipe,corrosion_category=category(recipe),
            corrosion_types=types,corrosion_stage_count=len(types),partial=partial_info,
            damage=detail,stages_applied=len(types)+int(partial),
            inherited_target_count=int((new.target_a>=0).sum()),label_coupled_acceptance=True)
        if weather_plan is not None:report['compound']['weather_plan_selection']=weather_plan
        if profile and profile.get('conservative_v12'):
            report['compound'].update(major_corrosion_count=int(major is not None),weak_overlay=weak,
                major_type=major,no_combined_major_corrosion=True)
            report['_weather_arrays']=weather_arrays or {}
        if profile and profile.get('explicit_partial_cut_guard'):
            report['compound']['placement_scope']=dict(
                explicit_cut_boundary=True,cut_guard_point_counts={s:len(cut_guards[s]) for s in 'ab'},
                ambiguous_seam_tokens_are_not_artificial_cuts=True,
                adds_correspondence_labels=False)
        if profile is not None:
            report['distribution_profile']=profile['name']
            report['pair_shared_scale']=scale_info[i]
        if profile and profile.get('record_latent_seam') and new.label:
            from .latent_seam import source_band,measure
            scale=scale_info[0]['common_scale']
            bands=source_band(preweather[0],cut_guards,max(6.,8.*scale),max(3.,3.*scale),
                              profile.get('bridge_short_gaps_px',0.))
            latent,arrays=measure(preweather[0],new,bands)
            if latent is None:return None,dict(stage='latent_seam',detail=dict(reason='no_supported_source_band'))
            if profile.get('heterogeneous_strong') and 'wave' in recipe and not latent['has_near_and_far']:
                return None,dict(stage='latent_seam',detail=dict(reason='strong_wave_missing_near_and_far_parts'))
            report['latent_seam']=latent
            # Transient arrays are archived separately by materialize; never
            # fed to either training loader or invented correspondence targets.
            report['_latent_arrays']=arrays
        for side in 'ab':
            info=detail.get(side,{})
            report['side_'+side].update({k:v for k,v in info.items()
                                        if k not in ('removed_area_px','removed_fraction')})
            if info:
                report['side_'+side]['weather_only_removed_area_px']=info.get('removed_area_px',0)
        result.append((new,report,preweather[i]))
    return result,dict(applied=True)
