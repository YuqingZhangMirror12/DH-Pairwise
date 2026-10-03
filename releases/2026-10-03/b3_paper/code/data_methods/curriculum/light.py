"""v13 final light layer, measured on surviving untouched original contours.

The70% is an arc-length quota per fragment, NOT a Bernoulli probability.
Artificial Partial/notch cuts and already damaged arcs are excluded. Smooth
fields are built on whole arc intervals; there is no independent pixel noise.
"""
from collections import Counter
from dataclasses import replace
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from ..s7_balanced_v2.conservative_weather import contour, smooth_profile
from ..s7_compound_v1.geometry import EIGHT, _eligible_runs, _signed_arc_distance, _changed_view, area_ratio
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import _readonly
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import inherit_pair_targets


def untouched_contour(original,current,primary=None,protected=None,guard_px=4.):
    old,_,_=contour(original);points,edge,arc=contour(current)
    eligible=cKDTree(old).query(points)[0]<.25
    if primary:
        distance,index=cKDTree(primary['points']).query(points)
        eligible &= (distance>.25)|(primary['total'][index]<=0.)
    if protected is not None and len(protected):
        eligible &= cKDTree(protected).query(points)[0]>guard_px
    # New cut edges are absent from the original outer contour, not labelled
    # as untouched simply because the primary depth field has no value there.
    return points,edge,arc,eligible


def make_field(arc,edge,eligible,boundary_depth,rng,coverage=.70,tolerance=.02,peaks=(1.,3.)):
    perimeter=float(edge.sum());intact=eligible&np.roll(eligible,-1)
    denominator=float(edge[intact].sum())
    runs=[r for r in _eligible_runs(eligible,arc,edge) if r[1]>=24.]
    if not runs or denominator<40.:return None
    plans=[(start,length,full,float(rng.random()),float(rng.uniform(*peaks)),
            float(rng.uniform(-np.pi,np.pi))) for start,length,full in runs]

    def at(fraction):
        field=np.zeros(len(arc))
        for start,length,full,position,peak,phase in plans:
            width=length*fraction
            if width<12.:continue
            center=(position*perimeter if full else start+width/2+position*(length-width))%perimeter
            u=_signed_arc_distance(arc,center,perimeter)/(width/2.)
            # Smooth shoulders with gradual low-frequency variation inside.
            shoulder=np.sin(np.pi*np.minimum(np.maximum(1.-np.abs(u),0.)/.12,1.)/2)**2
            wave=1.+(peak-1.)*np.sin(np.pi*np.maximum(1.-np.abs(u),0.)/2)**2*(.9+.1*np.cos(2*np.pi*u+phase))
            value=np.where(np.abs(u)<1.,wave*shoulder,0.)*eligible
            field=np.maximum(field,value)
        changed=(field>=boundary_depth)&eligible
        actual=float(edge[intact&changed&np.roll(changed,-1)].sum()/denominator)
        return field,changed,actual
    low,high=0.,1.;best=None
    for _ in range(26):
        fraction=(low+high)/2;field,changed,actual=at(fraction)
        if best is None or abs(actual-coverage)<abs(best[2]-coverage):best=(field,changed,actual)
        if actual<coverage:low=fraction
        else:high=fraction
    if best is None or abs(best[2]-coverage)>tolerance:return None
    field,changed,actual=best
    field_active=field>0.
    return field,dict(eligible_length_px=denominator,actual_affected_length_px=actual*denominator,
        actual_affected_fraction=actual,target_fraction=coverage,tolerance=tolerance,
        field_support_fraction=float(edge[intact&field_active&np.roll(field_active,-1)].sum()/denominator),
        contiguous_interval_count=len(plans),requested_peak_depths_px=[p[4] for p in plans])


def fragment(original,current,primary,rng,coverage=.70,tolerance=.02,peaks=(1.,3.),protected=None,guard_px=4.,whole_contour=False):
    current=np.asarray(current,bool)
    if whole_contour:
        if protected is not None:raise ValueError('whole-contour final light cannot hide protected contact')
        points,edge,arc=contour(current);eligible=np.ones(len(points),bool)
        unprotected_eligible=eligible.copy()
    else:
        points,edge,arc,eligible=untouched_contour(original,current,primary,protected,guard_px)
        _,_,_,unprotected_eligible=untouched_contour(original,current,primary)
    old_denominator=float(edge[unprotected_eligible&np.roll(unprotected_eligible,-1)].sum())
    filled=ndimage.binary_fill_holes(np.pad(current,1),structure=EIGHT)
    inward=ndimage.distance_transform_edt(filled)[1:-1,1:-1]-.5
    points_i=np.rint(points).astype(int)
    boundary_depth=inward[tuple(points_i.T)]
    band=np.argwhere(current&(inward<=peaks[1]));index=cKDTree(points).query(band)[1]
    depth=inward[tuple(band.T)];reasons=Counter()
    for attempt in range(24):
        result=make_field(arc,edge,eligible,boundary_depth,rng,coverage,tolerance,peaks)
        if result is None:reasons['coverage_cannot_fit_smooth_intervals']+=1;continue
        field,info=result;removed=depth<=field[index]
        new=current.copy();new[tuple(band[removed].T)]=False
        if not removed.any():reasons['no_effect']+=1;continue
        if ndimage.label(new,EIGHT)[1]!=1:reasons['disconnected']+=1;continue
        if new.sum()<max(64.,.25*current.sum()):reasons['retained_area']+=1;continue
        if np.any(ndimage.binary_fill_holes(new,structure=EIGHT)&~new&current):
            reasons['new_enclosed_hole']+=1;continue
        # Recompute coverage from actual raster output, not just requested field.
        changed=current[tuple(points_i.T)]&~new[tuple(points_i.T)]
        intact=eligible&np.roll(eligible,-1)
        actual=float(edge[intact&changed&np.roll(changed,-1)].sum()/info['eligible_length_px'])
        if abs(actual-coverage)>tolerance:reasons['raster_coverage_mismatch']+=1;continue
        info.update(applied=True,background_light=True,peak_limit_px=float(peaks[1]),requested_peak_range_px=list(peaks),
            whole_postprimary_contour=whole_contour,
            actual_affected_fraction=actual,actual_affected_length_px=actual*info['eligible_length_px'],
            applied_max_depth_px=float(depth[removed].max()),removed_area_px=int(removed.sum()),
            component_count=1,attempts=attempt+1,attempt_reasons=dict(reasons),
            denominator=('surviving original uncorroded outer-contour edges; excludes new cut edges AND protected contact/control arcs'
                if protected is not None else 'surviving original uncorroded outer-contour edges; excludes new cut edges and primary damage; no contact protection'),
            before_protection_eligible_length_px=old_denominator,
            excluded_protection_length_px=old_denominator-info['eligible_length_px'],
            fraction_of_all_undamaged_contour=actual*info['eligible_length_px']/old_denominator if old_denominator else None,
            actual_fraction_of_full_postprimary_contour=float(edge[changed&np.roll(changed,-1)].sum()/edge.sum()),
            protection_guard_px=guard_px if protected is not None else 0.,
            contact_protection_enabled=protected is not None,
            gt_used_for_placement=False,field_mode='continuous_progressive_peak_range')
        arrays=dict(points=points.astype(np.float32),edge=edge.astype(np.float32),arc=arc.astype(np.float32),
            eligible=eligible,total=field.astype(np.float32),total_exact=field.copy(),
            boundary_depth=boundary_depth.astype(np.float32))
        if whole_contour:info['denominator']='entire post-primary outer contour, including eroded seam and new cut edges; no protected interval'
        return new,info,arrays
    return None,dict(reason='background_recession_rejected',attempt_reasons=dict(reasons)),None


def light_area_ratio_floor(before_ratio,minimum_relative=.95):
    """Bound tiny raster area drift; do not turn an old1:8 boundary into a cliff.

    The light layer removes real material on BOTH fragments. Uniform pixel
    depths naturally remove a greater fraction from the smaller fragment.
    Keep the previous1:8 protection, with an explicit maximum5% relative
    tolerance at that boundary (and for sources already more unequal).
    """
    if not 0<minimum_relative<=1:raise ValueError('invalid relative area-ratio guard')
    return min(.125,float(before_ratio))*minimum_relative


def pair(original,current,main_fields,rng,coverage,tolerance,minimum_ratio_relative=.95,peaks=(1.,3.),protected=None,guard_px=4.,whole_contour=False):
    views={};details={};arrays={};updates={}
    for side in 'ab':
        old=getattr(original,'mask_'+side)[0];before=getattr(current,'mask_'+side)[0]
        primary=({key:main_fields[side+'_'+key] for key in ('points','total')}
                 if side+'_points' in main_fields else None)
        new,info,fields=fragment(old,before,primary,rng,coverage,tolerance,peaks,
            None if protected is None else protected[side],guard_px,whole_contour=whole_contour)
        if new is None:return None,info,None
        # Inherit from the post-primary sample, retaining ignored new cut labels.
        view=_changed_view(current,side,new,info,'wave');views[side]=view;details[side]=info
        coarse=np.asarray(Image.fromarray(np.uint8(view.mask)*255).resize((128,128),Image.Resampling.NEAREST))>0
        updates.update({'mask_'+side:_readonly(view.mask[None],np.float32),
            'coarse_mask_'+side:_readonly(coarse[None],np.float32),
            'points_rc_'+side:view.points,'contour_valid_'+side:view.valid})
        arrays.update({side+'_'+key:value for key,value in fields.items()})
        arrays[side+'_protected_points']=np.zeros((0,2)) if protected is None else protected[side]
        arrays['packed_before_'+side]=np.packbits(before.astype(bool),axis=1)
    ta,tb=inherit_pair_targets(current,views['a'],views['b'])
    if current.label and int((ta>=0).sum())<4:
        return None,dict(reason='fewer_than_four_inherited_correspondences'),None
    updates.update(target_a=_readonly(ta,np.int64),target_b=_readonly(tb,np.int64))
    out=replace(current,**updates)
    before_ratio=area_ratio(current);after_ratio=area_ratio(out)
    floor=light_area_ratio_floor(before_ratio,minimum_ratio_relative)
    if after_ratio<floor:
        return None,dict(reason='background_created_extreme_area_ratio',
            before_ratio=before_ratio,after_ratio=after_ratio,minimum_ratio=floor),None
    for info in details.values():
        info.update(pair_area_ratio_before=before_ratio,pair_area_ratio_after=after_ratio,
            pair_area_ratio_floor=floor,minimum_ratio_relative=minimum_ratio_relative)
    return out,details,arrays
