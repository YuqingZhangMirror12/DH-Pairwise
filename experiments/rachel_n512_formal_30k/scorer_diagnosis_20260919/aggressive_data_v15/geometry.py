"""Trim original supported seam ends, then replay exclusive damage and light.

All geometry stays in the original800px canvas/GT. This is a paired review
construction conditioned on the existing TRAIN recipe, not a new GT matcher.
Negative masks receive matched material-retention cuts with NO common-seam GT.
"""
from dataclasses import replace
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from PIL import Image
from ..s7_balanced_v2.latent_seam import source_band,measure
from ..s7_balanced_v2.partial_v14 import retained_support
from ..s7_balanced_v2.conservative_weather import contour,parts
from ..s7_balanced_v2.background_recession import pair as light_pair
from .fixed_gap import measure_fixed
from ..s7_compound_v1.geometry import EIGHT,masks,area_ratio
from staging.pairwise_v0_2.pairwise_data.rachel_partial_seam_dataset import PartialSeamConfig,crop_training_pair
from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import _changed_view,changed_report
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import inherit_pair_targets
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import _readonly


def weighted_quantile(v,w,q):
    order=np.argsort(v,kind='stable');v=np.asarray(v)[order];w=np.asarray(w)[order]
    if not len(v) or not w.sum()>0:raise ValueError('empty weighted curve')
    return np.interp(q,np.r_[0.,np.cumsum(w)/w.sum()],np.r_[v[0],v])


def topology(original,new):
    if not new.any() or np.any(new&~original):raise ValueError('empty/added material')
    if ndimage.label(new,EIGHT)[1]!=1:raise ValueError('disconnected fragment')
    if np.any(ndimage.binary_fill_holes(new,structure=EIGHT)&~new&original):raise ValueError('new enclosed hole')
    if new.sum()<max(64,.25*original.sum()):raise ValueError('less than25% material remains')


def crop(sample,side,mask):
    old=getattr(sample,'mask_'+side)[0].astype(bool)
    if np.array_equal(old,mask):return sample
    topology(old,mask)
    result=crop_training_pair(sample,side=side,retained_mask=mask,
        config=PartialSeamConfig().crop_config,topology_connectivity=2)
    if not result.accepted:raise ValueError('crop:'+result.reason)
    return result.sample


def apply_partial(trimmed,old_primary_masks):
    current=trimmed
    for side,mask in zip('ab',old_primary_masks):
        current=crop(current,side,getattr(current,'mask_'+side)[0].astype(bool)&mask)
    return current


def plan_positive_trim(base,reference_primary,rng,mode):
    """Use the v14 retained source arc for a paired20–30% shortening target.

The trim is executed BEFORE the same partial/weather masks. Planning from
known TRAIN source arcs is permitted; new cut edges never acquire match GT.
"""
    bands=source_band(base,bridge=0.)
    ref,proof=retained_support(base,reference_primary,bands)
    # Either fragment may lose the end(s). Always cropping the larger one can
    # unnecessarily destroy a K-notch footprint confined to that fragment.
    # Source-arc shortening, connectivity and aggregate25% limits still apply.
    side='a' if rng.random()<.5 else 'b'
    points=proof[side+'_source_points'];w=proof[side+'_source_weights']
    valid=proof[side+'_physically_retained']
    points,w=points[valid],w[valid]
    if len(points)<16:raise ValueError('too little original supported arc')
    center=np.average(points,axis=0,weights=w)
    covariance=(points-center).T@((points-center)*w[:,None])/w.sum()
    direction=np.linalg.eigh(covariance)[1][:,-1]
    # Bounded orientation perturbation avoids making all new cuts identical.
    angle=float(rng.uniform(-.12,.12));c,s=np.cos(angle),np.sin(angle)
    direction=np.array([[c,-s],[s,c]])@direction
    if rng.random()<.5:direction=-direction
    remove=float(rng.uniform(.20,.30))
    fractions=(remove/2,1-remove/2) if mode=='both' else (0.,1-remove)
    projections=points@direction
    lo,hi=weighted_quantile(projections,w,fractions)
    if mode=='one':lo=-np.inf
    mask=getattr(base,'mask_'+side)[0].astype(bool);rr,cc=np.indices(mask.shape)
    coordinate=rr*direction[0]+cc*direction[1]
    keep=(coordinate>=lo)&(coordinate<=hi)
    trimmed=crop(base,side,mask&keep)
    reference_after=replace(reference_primary,**{'mask_'+side:(getattr(reference_primary,'mask_'+side)[0].astype(bool)&keep)[None].astype(np.float32)})
    post,_=retained_support(base,reference_after,bands)
    ratio=post['common_retained_length_px']/ref['common_retained_length_px']
    if not .68<=ratio<=.82:raise ValueError('realized seam shortening outside18–32% raster allowance')
    return trimmed,dict(mode=mode,side=side,direction_rc=direction.tolist(),
        lower=None if not np.isfinite(lo) else float(lo),upper=float(hi),
        target_removed_fraction=remove,common_length_before_px=ref['common_retained_length_px'],
        common_length_after_px=post['common_retained_length_px'],retained_fraction=ratio,
        material_retained_fraction=float(getattr(trimmed,'mask_'+side).sum()/getattr(base,'mask_'+side).sum()),
        new_cut_exclusion_px=8.,trim_applied_before_primary=True,planning_basis='existing v14 TRAIN-supported surviving primary arc'),bands


def plan_negative_trim(base,positive_plan,rng):
    side='a' if base.mask_a.sum()>=base.mask_b.sum() else 'b'
    mask=getattr(base,'mask_'+side)[0].astype(bool);pixels=np.argwhere(mask)
    angle=float(rng.uniform(-np.pi,np.pi));direction=np.array([np.cos(angle),np.sin(angle)])
    projections=pixels@direction
    fraction=positive_plan['material_retained_fraction'];removed=1-fraction
    lo,hi=np.quantile(projections,[removed/2,1-removed/2] if positive_plan['mode']=='both' else [0,1-removed])
    if positive_plan['mode']=='one':lo=-np.inf
    rr,cc=np.indices(mask.shape);value=rr*direction[0]+cc*direction[1]
    new=mask&(value>=lo)&(value<=hi)
    trimmed=crop(base,side,new)
    actual=float(new.sum()/mask.sum())
    if abs(actual-fraction)>.015:raise ValueError('negative material retention differs')
    return trimmed,dict(mode=positive_plan['mode'],side=side,material_retained_fraction=actual,
        direction_rc=direction.tolist(),lower=None if not np.isfinite(lo) else float(lo),upper=float(hi),
        gt_seam_used=False,common_length_before_px=None,common_length_after_px=None,retained_fraction=None,
        trim_applied_before_primary=True,new_cut_exclusion_px=8.)


def reconstruct(original,points,field):
    # Same original-boundary EDT coordinates as v14, with a variable field.
    fill=ndimage.binary_fill_holes(np.pad(original,1),structure=EIGHT)
    inward=ndimage.distance_transform_edt(fill)[1:-1,1:-1]-.5
    pixels=np.argwhere(original&(inward<=float(np.max(field,initial=0.))))
    result=original.copy()
    if len(pixels):
        nearest=cKDTree(points).query(pixels)[1]
        removed=inward[tuple(pixels.T)]<=field[nearest]
        result[tuple(pixels[removed].T)]=False
    return result


def weather(trimmed,base,old_fields,old_damage,recipe,peak):
    """Keep the source footprint/K/weak profile; increase only major depth.

peak is a construction search parameter, NOT a claimed final two-side gap.
Final positive gap is measured after BOTH masks' background degradation.
"""
    major,weak=parts(recipe);updates={};views={};details={};fields={}
    for side in 'ab':
        original=getattr(base,'mask_'+side)[0].astype(bool)
        current=getattr(trimmed,'mask_'+side)[0].astype(bool)
        if side+'_points' in old_fields:
            p=old_fields[side+'_points'];oldbase=old_fields[side+'_major'];weakfield=old_fields[side+'_weak']
            gain=peak/max(float(oldbase.max()),1e-9) if major else 1.
            total=np.minimum(peak,oldbase*gain+weakfield) if major else old_fields[side+'_total_exact']
            alive=current[tuple(np.rint(p).astype(int).T)]
            eligible=old_fields[side+'_eligible']&alive
            intact=eligible&np.roll(eligible,-1)
            active=(oldbase>0)&eligible
            denom=float(old_fields[side+'_edge'][intact].sum())
            fraction=float(old_fields[side+'_edge'][intact&active&np.roll(active,-1)].sum()/denom) if denom else 0.
            if major and fraction>.50+1e-6:raise ValueError('primary footprint exceeds50% after seam shortening')
            new=current&reconstruct(original,p,total)
            topology(current,new)
            info=dict(old_damage[side]);removed=current&~new
            fill=ndimage.binary_fill_holes(np.pad(original,1),structure=EIGHT)
            inward=ndimage.distance_transform_edt(fill)[1:-1,1:-1]-.5
            info.update(max_combined_depth_px=float(total.max()),
                applied_max_depth_px=float(inward[removed].max(initial=0.)),v15_field_search_peak_px=peak,
                requested_peak_depths_px=[],notch_independently_removed_pixels=[],
                final_gap_not_equal_single_side_depth=True,removed_area_px=int(removed.sum()),
                removed_fraction=float(removed.sum()/current.sum()),weak_overlay_capped_to_total9=False,
                wave_range_px=[0.,float(total.max())] if major=='wave' else None,
                old_depth_measurements_replaced=True)
            info['post_trim_major_affected_fraction']=fraction
            # Every requested major region must remain effective after trimming;
            # a removed end cannot silently reduce K or erase the whole damage.
            region_receipts=[]
            removed_pixels=np.argwhere(removed)
            nearest_removed=cKDTree(p).query(removed_pixels)[1]
            for region in info.get('ignore_source_regions',[]):
                rp=np.asarray(region['source_points_rc'])
                cells=np.rint(rp).astype(int)
                if current[tuple(cells.T)].sum()<max(4,.55*len(cells)):
                    raise ValueError('trim erased a primary damage region')
                indices=cKDTree(p).query(rp)[1];belongs=np.isin(nearest_removed,indices)
                if int(belongs.sum())<8:raise ValueError('major region no longer effective')
                requested=float(total[indices].max());actual=float(inward[tuple(removed_pixels[belongs].T)].max(initial=0.))
                region_receipts.append(dict(region,requested_peak_depth_px=requested,applied_max_depth_px=actual,
                    independently_removed_pixels=int(belongs.sum())))
                info['requested_peak_depths_px'].append(requested)
                info['notch_independently_removed_pixels'].append(int(belongs.sum()))
            info['ignore_source_regions']=region_receipts
            if info['removed_area_px']<4:raise ValueError('weather lost after trim')
            fields.update({side+'_'+k:old_fields[side+'_'+k] for k in ('points','edge','arc','eligible')})
            fields.update({side+'_major':oldbase*gain,side+'_weak':weakfield,
                side+'_total':total.astype(np.float32),side+'_total_exact':total})
        else:new=current;info=dict(applied=False)
        view=_changed_view(trimmed,side,new,info,'local' if info.get('ignore_source_regions') else 'wave')
        views[side]=view;details[side]=info
        coarse=np.asarray(Image.fromarray(np.uint8(view.mask)*255).resize((128,128),Image.Resampling.NEAREST))>0
        updates.update({'mask_'+side:_readonly(view.mask[None],np.float32),
            'coarse_mask_'+side:_readonly(coarse[None],np.float32),
            'points_rc_'+side:view.points,'contour_valid_'+side:view.valid})
    ta,tb=inherit_pair_targets(trimmed,views['a'],views['b'])
    updates.update(target_a=_readonly(ta,np.int64),target_b=_readonly(tb,np.int64))
    new=replace(trimmed,**updates)
    if new.label and (new.target_a>=0).sum()<4:raise ValueError('fewer than4 inherited correspondences')
    return new,details,fields


def gap_check(base,final,bands,trimmed,primary_fields,major):
    """Report cropped-out support separately; never turn missing rays into0."""
    # Keep only old supported arc that physically survives structural trimming.
    _,proof=retained_support(base,trimmed,bands)
    # Freeze pre-cut partners from retained_support. Rematching only to the
    # cropped opposite band can invent large gaps at structural-cut endpoints.
    summary,arrays=measure_fixed(base,final,proof)
    if summary is None:raise ValueError('no measurable source projection')
    damaged=[];weights=[]
    for side in 'ab':
        p=arrays[side+'_source_points'];active=np.zeros(len(p),bool)
        # Final two-sided projected gap; the primary footprint may be on
        # either side. Convert its original coordinates into this side frame.
        for source in 'ab':
            if source+'_points' not in primary_fields:continue
            shift=(base.translation_a_to_b_rc if side=='a' and source=='b' else
                -base.translation_a_to_b_rc if side=='b' and source=='a' else np.zeros(2))
            d,index=cKDTree(primary_fields[source+'_points']).query(p+shift)
            active|=(d<=3.5)&(primary_fields[source+'_total_exact'][index]>0)
        ok=active&arrays[side+'_valid']
        damaged.extend(arrays[side+'_gap'][ok]);weights.extend(arrays[side+'_source_weight'][ok])
        arrays[side+'_primary_affected']=active
    if major:
        if not damaged:raise ValueError('primary gap footprint missing')
        maximum=float(np.max(damaged))
        if not 5.<=maximum<=15.:raise ValueError('actual final primary gap peak outside5–15px:'+str(maximum))
        summary['primary_gap_peak_px']=maximum
        summary['primary_gap_quantiles_px']=np.quantile(damaged,[0,.1,.25,.5,.75,.9,1]).tolist()
    else:summary.update(primary_gap_peak_px=None,primary_gap_quantiles_px=None)
    return summary,arrays


def augment(base_pair,old_primary,old_fields,old_reports,recipe,rng,mode,planners=None):
    positive_planner,negative_planner=planners or (plan_positive_trim,plan_negative_trim)
    pos_trim,plan,bands=positive_planner(base_pair[0],old_primary[0],rng,mode)
    neg_trim,neg_plan=negative_planner(base_pair[1],plan,rng)
    trimmed=(pos_trim,neg_trim);plans=(plan,neg_plan);major,_=parts(recipe)
    target_peak=float(rng.uniform(6.,13.)) if major else 0.
    current=[];details=[];fields=[];primary=[]
    for index in range(2):
        base=base_pair[index];now=trimmed[index]
        if recipe=='partial':
            now=apply_partial(now,masks(old_primary[index]));detail={};field={}
            if index==0:
                ratio,_=retained_support(base,now,bands)
                if ratio['common_over_smaller_perimeter']<.15:raise ValueError('Partial15% floor violated')
                plan['partial_support_constraint']=ratio
        elif recipe=='clean':detail={};field={}
        else:now,detail,field=weather(now,base,old_fields[index],old_reports[index]['compound']['damage'],recipe,target_peak)
        primary.append(now)
        if recipe!='clean':
            # Original base explicitly excludes BOTH old Partial cuts and the
            # newly exposed trim cuts from the70% light-degradation denominator.
            final,light,light_arrays=light_pair(base,now,field,rng,.70,.02)
            if final is None:raise ValueError('background:'+light['reason'])
        else:final=now;light={};light_arrays={}
        if final.label and (final.target_a>=0).sum()<4:raise ValueError('final fewer than4 inherited targets')
        if area_ratio(final)<min(.125,area_ratio(base))*.95:raise ValueError('extreme area ratio introduced')
        # Per-stage retention is insufficient: several individually valid cuts
        # can jointly leave less than25% of the original fragment. Reject the
        # attempt here, BEFORE returning/writing either member of the pair.
        for side in 'ab':
            topology(getattr(base,'mask_'+side)[0].astype(bool),
                     getattr(final,'mask_'+side)[0].astype(bool))
        detail=dict(primary_damage=detail,background=light,trim=plans[index],target_field_peak_px=target_peak)
        current.append(final);details.append(detail);fields.append(dict(primary=field,light=light_arrays))
    gap,ga=gap_check(base_pair[0],current[0],bands,
        primary[0] if recipe=='partial' else trimmed[0],fields[0]['primary'],major)
    details[0]['gap']=gap;details[1]['gap']=None
    return tuple(current),details,fields,ga,trimmed,tuple(primary)
