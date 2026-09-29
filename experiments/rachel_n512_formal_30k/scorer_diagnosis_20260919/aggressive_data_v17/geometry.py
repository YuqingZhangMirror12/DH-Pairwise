"""Re-use accepted curved cuts; regenerate only depth/K and dependent light."""
from dataclasses import replace
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from ..aggressive_data_v16.geometry import replay, selected_side, visible_curve_nonlinearity
from ..aggressive_data_v15.geometry import crop, topology, apply_partial, reconstruct
from ..aggressive_data_v15.fixed_gap import measure_fixed
from ..s7_balanced_v2.latent_seam import source_band
from ..s7_balanced_v2.partial_v14 import retained_support
from ..s7_balanced_v2.background_recession import pair as light_pair
from ..s7_compound_v1.geometry import masks, area_ratio, EIGHT
from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import _changed_view
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import inherit_pair_targets
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import _readonly
from .weather import weather_fragment, contour, parts


def fixed_trim(base, previous, proof):
    plan=previous['detail']['trim'];side=plan['side']
    if selected_side(base,plan['size_class'])!=side:raise ValueError('cut side identity changed')
    newmask=replay(getattr(base,'mask_'+side)[0].astype(bool),plan)
    for s in 'ab':
        expected=newmask if s==side else getattr(base,'mask_'+s)[0].astype(bool)
        actual=np.unpackbits(proof['packed_trim_'+s],axis=1).astype(bool)
        if not np.array_equal(expected,actual):raise ValueError('approved cut replay changed')
    return crop(base,side,newmask)


def weather_pair(sample,base,old_fields,recipe,rng,k):
    major,weak=parts(recipe);details={};fields={};views={};updates={}
    for side in 'ab':
        original=getattr(base,'mask_'+side)[0].astype(bool)
        current=getattr(sample,'mask_'+side)[0].astype(bool)
        if side+'_points' in old_fields:
            points,edge,arc=contour(original)
            if not np.allclose(points,old_fields[side+'_points'],atol=1e-4):raise ValueError('source contour mismatch')
            alive=current[tuple(np.rint(points).astype(int).T)]
            eligible=old_fields[side+'_eligible']&alive
            _,info,field=weather_fragment(original,rng,eligible,major,weak,k)
            if field is None:raise ValueError('new depth sampling:'+info['reason'])
            # Execute structural cut FIRST; material from a cut cannot return.
            new=current&reconstruct(original,points,field['total_exact'])
            topology(current,new)
            removed=current&~new
            total_active=(field['total_exact']>0)&eligible
            intact=eligible&np.roll(eligible,-1);denom=float(edge[intact].sum())
            affected=float(edge[intact&total_active&np.roll(total_active,-1)].sum()/denom) if denom else 0.
            if not 0<affected<=.50+1e-6:raise ValueError('primary layer exceeds50% surviving source arc')
            # Recompute actual contribution after the fixed structural cut.
            rp=np.argwhere(removed);nearest=cKDTree(points).query(rp)[1]
            if len(rp)<4:raise ValueError('no effective primary erosion')
            from scipy import ndimage
            inward=ndimage.distance_transform_edt(ndimage.binary_fill_holes(np.pad(original,1),structure=EIGHT))[1:-1,1:-1]-.5
            contributions=[]
            for region in info.get('ignore_source_regions',[]):
                indices=cKDTree(points).query(np.asarray(region['source_points_rc']))[1]
                belongs=np.isin(nearest,indices)
                actual=float(inward[tuple(rp[belongs].T)].max(initial=0.))
                if belongs.sum()<8 or actual<region['requested_peak_depth_px']-1.5:
                    raise ValueError('requested region no longer effective after fixed cut')
                region.update(independently_removed_pixels=int(belongs.sum()),applied_max_depth_px=actual)
                contributions.append(int(belongs.sum()))
            if major=='gaps' and len(contributions)!=k:raise ValueError('K actual regions mismatch')
            info.update(post_trim_major_affected_fraction=affected,requested_gap_count=k if major=='gaps' else 0,
                removed_area_px=int(removed.sum()),removed_fraction=float(removed.sum()/current.sum()),
                applied_max_depth_px=float(inward[removed].max()),notch_independently_removed_pixels=contributions,
                final_gap_not_equal_single_side_depth=True)
            if weak:
                base_only=current&reconstruct(original,points,field['major'])
                effective=int(np.count_nonzero(base_only&~new))
                if effective<4:raise ValueError('weak overlay not independently effective')
                info['weak_independently_removed_pixels']=effective
            fields.update({side+'_'+key:value for key,value in field.items()})
        else:new=current;info=dict(applied=False)
        view=_changed_view(sample,side,new,info,'local' if info.get('ignore_source_regions') else 'wave')
        views[side]=view;details[side]=info
        coarse=np.asarray(Image.fromarray(np.uint8(view.mask)*255).resize((128,128),Image.Resampling.NEAREST))>0
        updates.update({'mask_'+side:_readonly(view.mask[None],np.float32),
            'coarse_mask_'+side:_readonly(coarse[None],np.float32),
            'points_rc_'+side:view.points,'contour_valid_'+side:view.valid})
    ta,tb=inherit_pair_targets(sample,views['a'],views['b'])
    updates.update(target_a=_readonly(ta,np.int64),target_b=_readonly(tb,np.int64))
    return replace(sample,**updates),details,fields


def gap_check(base,final,bands,retained,fields,primary_present):
    _,proof=retained_support(base,retained,bands)
    summary,arrays=measure_fixed(base,final,proof)
    if summary is None:raise ValueError('no measurable source projection')
    damaged=[]
    for side in 'ab':
        p=arrays[side+'_source_points'];active=np.zeros(len(p),bool)
        for source in 'ab':
            if source+'_points' not in fields:continue
            shift=(base.translation_a_to_b_rc if side=='a' and source=='b' else
                -base.translation_a_to_b_rc if side=='b' and source=='a' else np.zeros(2))
            distance,index=cKDTree(fields[source+'_points']).query(p+shift)
            active|=(distance<=3.5)&(fields[source+'_total_exact'][index]>0)
        valid=active&arrays[side+'_valid'];damaged.extend(arrays[side+'_gap'][valid])
        arrays[side+'_primary_affected']=active
    if primary_present:
        if not damaged:raise ValueError('primary footprint has no resolved gap')
        peak=float(np.max(damaged))
        if not 5<=peak<=25:raise ValueError('final primary peak outside5–25px')
        summary.update(primary_gap_peak_px=peak,primary_gap_quantiles_px=np.quantile(damaged,[0,.1,.25,.5,.75,.9,1]).tolist())
    else:summary.update(primary_gap_peak_px=None,primary_gap_quantiles_px=None)
    return summary,arrays


def augment(base_pair,old_primary,old_fields,previous_pair,proofs,recipe,rng,k):
    trimmed=tuple(fixed_trim(base,previous,proof) for base,previous,proof in zip(base_pair,previous_pair,proofs))
    bands=source_band(base_pair[0],bridge=0.);current=[];details=[];fields=[];primary=[]
    for index in range(2):
        base=base_pair[index];now=trimmed[index]
        if recipe=='partial':
            now=apply_partial(now,masks(old_primary[index]));detail={};field={}
            if index==0:
                support,_=retained_support(base,now,bands)
                if support['common_over_smaller_perimeter']<.15:raise ValueError('Partial15% floor')
        elif recipe=='clean':detail={};field={}
        else:now,detail,field=weather_pair(now,base,old_fields[index],recipe,rng,k)
        primary.append(now)
        if recipe!='clean':
            final,light,light_arrays=light_pair(base,now,field,rng,.70,.02)
            if final is None:raise ValueError('background:'+light['reason'])
        else:final=now;light={};light_arrays={}
        if final.label and (final.target_a>=0).sum()<4:raise ValueError('fewer than4 inherited targets')
        if area_ratio(final)<min(.125,area_ratio(base))*.95:raise ValueError('extreme area ratio')
        for side in 'ab':topology(getattr(base,'mask_'+side)[0].astype(bool),getattr(final,'mask_'+side)[0].astype(bool))
        current.append(final);fields.append(dict(primary=field,light=light_arrays))
        details.append(dict(primary_damage=detail,background=light,trim=previous_pair[index]['detail']['trim'],
            requested_gap_count=k if recipe.startswith('gaps') else 0,depth_revision='v17-3to8-5to15-k1to4'))
    gap,arrays=gap_check(base_pair[0],current[0],bands,primary[0] if recipe=='partial' else trimmed[0],
        fields[0]['primary'],recipe not in ('clean','partial'))
    details[0]['gap']=gap;details[1]['gap']=None
    return tuple(current),details,fields,arrays,trimmed,tuple(primary)
