"""Independent review augmentation. GT and archived datasets are never edited."""
from dataclasses import replace
import numpy as np
from scipy.spatial import cKDTree
from .trim import positive_trim,negative_trim
from .primary import weather_pair
from .light import pair as light_pair
from ..aggressive_data_v15.geometry import apply_partial,topology
from ..aggressive_data_v15.fixed_gap import measure_fixed
from ..s7_balanced_v2.latent_seam import source_band
from ..s7_balanced_v2.partial_v14 import retained_support
from ..s7_compound_v1.geometry import masks,area_ratio
from .evidence import source_islands
from .pristine import measure as measure_pristine,require as require_pristine,guard_fractions,negative_guards
from .crop_floor import evaluate as crop_evaluate
from .seam_contract import require_original, evaluate as connectable_evaluate, require as require_connectable


def gap_check(base,final,bands,retained,fields,primary_present,spec,primary):
    _,proof=retained_support(base,retained,bands)
    summary,arrays=measure_fixed(base,final,proof)
    if summary is None:raise ValueError('no measurable source projection')
    damaged=[];coverage=[];raster_coverages=[]
    for side in 'ab':
        p=arrays[side+'_source_points'];active=np.zeros(len(p),bool);changed=active.copy()
        for source in 'ab':
            if source+'_points' not in fields:continue
            shift=(base.translation_a_to_b_rc if side=='a' and source=='b' else
                -base.translation_a_to_b_rc if side=='b' and source=='a' else np.zeros(2))
            distance,index=cKDTree(fields[source+'_points']).query(p+shift)
            member=(distance<=3.5)&(fields[source+'_total_exact'][index]>0)
            active|=member
            q=np.rint(fields[source+'_points'][index]).astype(int)
            actual=(getattr(retained,'mask_'+source)[0,q[:,0],q[:,1]]>0)&(getattr(primary,'mask_'+source)[0,q[:,0],q[:,1]]==0)
            changed|=member&actual
        valid=active&arrays[side+'_valid'];damaged.extend(arrays[side+'_gap'][valid])
        arrays[side+'_primary_affected']=active
        arrays[side+'_primary_raster_removed']=changed
        w=arrays[side+'_source_weight'];coverage.append(float(w[active].sum()/w.sum()))
        raster_coverages.append(float(w[changed].sum()/w.sum()))
    if primary_present:
        if not damaged:raise ValueError('primary footprint has no resolved gap')
        peak=float(np.max(damaged))
        if not spec['gap_floor']<=peak<=spec['gap_cap']:raise ValueError('final primary peak outside version bounds')
        # Denominator is the ORIGINAL common source seam remaining after trim.
        if not all(spec['main_coverage'][0]<=x<=spec['main_coverage'][1] for x in coverage):
            raise ValueError('primary footprint outside version bilateral remaining common seam')
        summary.update(primary_gap_peak_px=peak,
            primary_gap_quantiles_px=np.quantile(damaged,[0,.1,.25,.5,.75,.9,1]).tolist())
    else:summary.update(primary_gap_peak_px=None,primary_gap_quantiles_px=None)
    summary.update(primary_field_coverage_by_side=coverage,primary_raster_coverage_by_side=raster_coverages,
        coverage_denominator='original GT-supported common seam remaining after the new end trim (before main damage)',
        primary_peak_limit_px=spec['gap_cap'])
    return summary,arrays


def augment(pair,rng,mode,bank,size_class,k,spec,trim_target):
    bases=tuple(row['original'] for row in pair);recipe=pair[0]['entry']['corrosion_recipe']
    require_original(bases[0])
    pos,pt,bands=positive_trim(bases[0],pair[0]['old_primary'],rng,mode,bank,size_class,trim_target)
    neg,nt=negative_trim(bases[1],pt,rng)
    trimmed=(pos,neg);current=[];details=[];fields=[];primary=[];positive_guard_fractions=None
    partial_applied=False;partial_skip_reason=None
    protect_pristine=bool(spec.get('pristine_protection_enabled',False))
    if not protect_pristine and spec['pristine_min_fraction']!=0:
        raise ValueError('disabled pristine protection must not hide a minimum quota')
    for index in range(2):
        base=bases[index];now=trimmed[index]
        if recipe=='partial':
            if index==0:
                if pt['crop_floor']['originally_short']:
                    partial_skip_reason='original_common_seam_below20'
                else:
                    proposed=apply_partial(now,masks(pair[index]['old_primary']))
                    try:crop_evaluate(base,proposed,bands)
                    except ValueError:partial_skip_reason='legacy_partial_would_cross_crop20_floor'
                    else:now=proposed;partial_applied=True
            elif partial_applied:now=apply_partial(now,masks(pair[index]['old_primary']))
            detail={};field={}
        elif recipe=='clean':detail={};field={}
        else:now,detail,field=weather_pair(now,base,pair[index]['fields'],recipe,rng,k,spec)
        primary.append(now)
        # For Partial, the primary stage is itself a structural cut, not
        # erosion. For all corrosion recipes the gate sees `trimmed` only.
        crop_contract=(crop_evaluate(base,now if recipe=='partial' else trimmed[index],bands)
            if index==0 else None)
        protected=None;control=None
        if index==0 and protect_pristine:
            pristine_before,_,protected=measure_pristine(base,now,bands)
            require_pristine(pristine_before,spec['pristine_min_fraction'])
            positive_guard_fractions=guard_fractions(base,protected)
            control=None
        elif protect_pristine:
            protected,control=negative_guards(base,now,positive_guard_fractions,rng)
        if recipe!='clean':
            final,light,light_arrays=light_pair(base,now,field,rng,.70,.02,peaks=spec['light'],
                protected=protected,guard_px=spec['light_protection_guard_px'],
                whole_contour=spec.get('light_scope')=='whole_postprimary_contour')
            if final is None:raise ValueError('background:'+light['reason'])
        else:final=now;light={};light_arrays={}
        if final.label and (final.target_a>=0).sum()<4:raise ValueError('fewer than4 inherited targets')
        if area_ratio(final)<min(.125,area_ratio(base))*.95:raise ValueError('extreme area ratio')
        for side in 'ab':topology(getattr(base,'mask_'+side)[0].astype(bool),getattr(final,'mask_'+side)[0].astype(bool))
        pristine=None;pristine_arrays={}
        if index==0:
            pristine,pristine_arrays,_=measure_pristine(base,final,bands)
            if protect_pristine:require_pristine(pristine,spec['pristine_min_fraction'])
            if protect_pristine and pristine['pristine_length_by_side_px']!=pristine_before['pristine_length_by_side_px']:
                raise ValueError('light invaded strictly protected original contact')
            pristine.update(enforced_minimum_fraction=spec['pristine_min_fraction'],
                protection_enabled=protect_pristine,measurement_only=not protect_pristine)
        connectable=None;connectable_arrays={}
        if index==0:
            connectable,connectable_arrays=connectable_evaluate(base,now,final,bands)
            require_connectable(connectable)
        current.append(final);fields.append(dict(primary=field,light=light_arrays,pristine=pristine_arrays,
                                                connectable=connectable_arrays))
        details.append(dict(primary_damage=detail,background=light,trim=(pt,nt)[index],
            crop_only_seam_floor=crop_contract,partial_crop_applied=partial_applied if recipe=='partial' else False,
            partial_crop_skip_reason=partial_skip_reason,
            pristine_seam=pristine,negative_preservation_control=control,
            connectable_seam=connectable,
            requested_gap_count=k if recipe.startswith('gaps') else 0,depth_revision=spec['version'],
            spec=spec,source_reference='same pre-additional-cut v14 fragment; no sequential v17/v18 cut'))
    gap,arrays=gap_check(bases[0],current[0],bands,primary[0] if recipe=='partial' else trimmed[0],
        fields[0]['primary'],recipe not in ('clean','partial'),spec,primary[0])
    details[0]['gap']=gap;details[1]['gap']=None
    details[0]['evidence_islands']=source_islands(bases[0],current[0],arrays)
    details[1]['evidence_islands']=None
    return tuple(current),details,fields,arrays,trimmed,tuple(primary)
