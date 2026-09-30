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
from .light import pair as light_pair
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


def weather_pair(sample,base,old_fields,recipe,rng,k,spec):
    major,weak=parts(recipe);details={};fields={};views={};updates={}
    for side in 'ab':
        original=getattr(base,'mask_'+side)[0].astype(bool)
        current=getattr(sample,'mask_'+side)[0].astype(bool)
        if side+'_points' in old_fields:
            points,edge,arc=contour(original)
            if not np.allclose(points,old_fields[side+'_points'],atol=1e-4):raise ValueError('source contour mismatch')
            alive=current[tuple(np.rint(points).astype(int).T)]
            eligible=old_fields[side+'_eligible']&alive
            _,info,field=weather_fragment(original,rng,eligible,major,weak,k,spec)
            if field is None:raise ValueError('new depth sampling:'+info['reason'])
            # Execute structural cut FIRST; material from a cut cannot return.
            new=current&reconstruct(original,points,field['total_exact'])
            topology(current,new)
            removed=current&~new
            total_active=(field['total_exact']>0)&eligible
            intact=eligible&np.roll(eligible,-1);denom=float(edge[intact].sum())
            affected=float(edge[intact&total_active&np.roll(total_active,-1)].sum()/denom) if denom else 0.
            if not spec['main_coverage'][0]-1e-6<=affected<=spec['main_coverage'][1]+1e-6:
                raise ValueError('primary footprint outside version eligible surviving arc bounds')
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
