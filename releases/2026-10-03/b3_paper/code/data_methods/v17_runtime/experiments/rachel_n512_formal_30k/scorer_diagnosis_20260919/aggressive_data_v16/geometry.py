"""TRAIN donor curves; 20% source-seam reduction, <=20% area per chosen side.

Side size is measured before this additional cut. The caller fixes the 70/30
side class before retries; geometry rejection never switches the chosen class.
"""
from dataclasses import replace
from collections import Counter
import hashlib
import numpy as np
from scipy import ndimage
from ..aggressive_data_v15.geometry import crop,topology,augment as damage_augment
from ..s7_balanced_v2.latent_seam import source_band
from ..s7_balanced_v2.partial_v14 import retained_support
from staging.pairwise_v0_2.pairwise_data.rachel_guided_partial_dataset import _weighted_offset
from .endpoints import audit_endpoints

TARGET=.20
RASTER_TOLERANCE=.01
AREA_CAP=.20


def selected_side(sample,size_class):
    if size_class not in ('smaller','larger'):raise ValueError('unknown size class')
    small='a' if sample.mask_a.sum()<=sample.mask_b.sum() else 'b'
    return small if size_class=='smaller' else ('b' if small=='a' else 'a')


def curve_field(mask,profile,angle_deg,flip,reverse):
    """Use the same TRAIN chord-normal profile as GuidedPartial, all rotations."""
    profile=np.asarray(profile,float)
    if profile.shape!=(129,) or not np.isfinite(profile).all():raise ValueError('129 finite donor samples required')
    # A zero/affine profile would be a disguised straight cut, never accepted.
    x=np.linspace(0,1,129);residual=profile-np.polyval(np.polyfit(x,profile,1),x)
    if np.std(residual)<.005:raise ValueError('donor curve too close to a straight line')
    if reverse:profile=profile[::-1]
    if flip:profile=-profile
    pixels=np.argwhere(mask);lo=pixels.min(0);hi=pixels.max(0)
    corners=np.array([[lo[0],lo[1]],[lo[0],hi[1]],[hi[0],lo[1]],[hi[0],hi[1]]])
    theta=np.deg2rad(angle_deg);tangent=np.array([np.cos(theta),np.sin(theta)])
    normal=np.array([-tangent[1],tangent[0]])
    tmin=float((corners@tangent).min());span=max(1.,float(np.ptp(corners@tangent)))
    def scores(points):
        points=np.asarray(points,float)
        return points@normal-np.interp((points@tangent-tmin)/span,x,profile)*span
    return pixels,scores,dict(angle_deg=float(angle_deg),flip=bool(flip),reverse=bool(reverse),
        tangent_span_px=span,tangent_origin_px=tmin)


def mask_from_bounds(mask,pixels,values,lower,upper):
    alive=np.ones(len(pixels),bool)
    if lower is not None:alive&=values>=lower
    if upper is not None:alive&=values<=upper
    result=np.zeros_like(mask);result[tuple(pixels[alive].T)]=True
    return result


def area_check(original,new):
    loss=float(1-new.sum()/original.sum())
    if loss<=0 or loss>AREA_CAP+1e-12:raise ValueError('additional cut area loss must be >0 and <=20% on either size')
    return loss


def replay(mask,plan):
    pixels,scores,_=curve_field(mask,plan['profile'],**{k:plan[k] for k in ('angle_deg','flip','reverse')})
    return mask_from_bounds(mask,pixels,scores(pixels),plan['lower'],plan['upper'])


def visible_curve_nonlinearity(original,new,plan):
    # Only the actual new boundary, away from old boundary, must exhibit the
    # donor's curvature. Prevent almost-linear subwindows of a curved profile.
    interior=ndimage.distance_transform_edt(original)>2
    boundary=new&~ndimage.binary_erosion(new)
    p=np.argwhere(boundary&interior)
    if len(p)<8:raise ValueError('too little visible new donor boundary')
    theta=np.deg2rad(plan['angle_deg']);t=np.array([np.cos(theta),np.sin(theta)])
    n=np.array([-t[1],t[0]])
    values=p@n;coordinate=p@t
    # Parallel curves in both-end mode are measured separately.
    _,scores,_=curve_field(original,plan['profile'],plan['angle_deg'],plan['flip'],plan['reverse'])
    distances=[np.abs(scores(p)-v) for v in (plan['lower'],plan['upper']) if v is not None]
    assignment=np.argmin(distances,axis=0);stats=[]
    for k in range(len(distances)):
        group=assignment==k
        if group.sum()<8:raise ValueError('one requested end has no visible curved boundary')
        xx=coordinate[group];yy=values[group]
        fit=np.polyval(np.polyfit(xx,yy,1),xx);rms=float(np.sqrt(np.mean((yy-fit)**2)))
        stats.append(dict(pixels=int(group.sum()),rms_from_best_line_px=rms))
        if rms<.60:raise ValueError('actual donor cut segment is nearly straight')
    return stats


def positive_trim(base,reference,rng,mode,bank,size_class):
    bands=source_band(base,bridge=0.);ref,proof=retained_support(base,reference,bands)
    side=selected_side(base,size_class);mask=getattr(base,'mask_'+side)[0].astype(bool)
    valid=proof[side+'_physically_retained'];points=proof[side+'_source_points'][valid]
    weights=proof[side+'_source_weights'][valid]
    if len(points)<16 or ref['common_retained_length_px']<=0:raise ValueError('too little original supported arc')
    cells=np.rint(points).astype(int);reasons=Counter()
    for proposal in range(48):
        donor=int(rng.integers(len(bank.profiles)));profile=bank.profiles[donor]
        try:
            pixels,scores,frame=curve_field(mask,profile,float(rng.uniform(0,360)),bool(rng.integers(2)),bool(rng.integers(2)))
            values=scores(pixels);sp=scores(cells)
            if mode=='both':
                lower=_weighted_offset(sp,weights,.90,False)
                upper=_weighted_offset(sp,weights,.90,True)
            elif mode=='one':
                low=bool(rng.integers(2));offset=_weighted_offset(sp,weights,.80,low)
                lower,upper=(None,offset) if low else (offset,None)
            else:raise ValueError('one or both ends required')
            kept=mask_from_bounds(mask,pixels,values,lower,upper);loss=area_check(mask,kept)
            plan=dict(**frame,lower=lower,upper=upper,profile=profile.tolist(),donor_index=donor,
                donor=bank.metadata['arcs'][donor],profile_sha256=hashlib.sha256(profile.tobytes()).hexdigest())
            postref=replace(reference,**{'mask_'+side:(getattr(reference,'mask_'+side)[0].astype(bool)&kept)[None].astype(np.float32)})
            after,after_proof=retained_support(base,postref,bands)
            ratio=after['common_retained_length_px']/ref['common_retained_length_px']
            if abs((1-ratio)-TARGET)>RASTER_TOLERANCE+1e-12:raise ValueError('actual source seam reduction outside19–21%')
            end_check=audit_endpoints(bands,proof,after_proof,mode)
            nonlinear=visible_curve_nonlinearity(mask,kept,plan)
            topology(mask,kept)
            trimmed=crop(base,side,kept)
            plan.update(mode=mode,side=side,size_class=size_class,target_removed_fraction=TARGET,
                raster_tolerance=RASTER_TOLERANCE,common_length_before_px=ref['common_retained_length_px'],
                common_length_after_px=after['common_retained_length_px'],retained_fraction=ratio,
                material_retained_fraction=1-loss,material_removed_fraction=loss,area_cap=AREA_CAP,
                original_areas_px={s:int(getattr(base,'mask_'+s).sum()) for s in 'ab'},
                new_cut_exclusion_px=8.,trim_applied_before_primary=True,gt_seam_used=True,
                planning_basis='v14 TRAIN-supported primary-surviving source arc',
                curve_origin='existing TRAIN manuscript outline bank; never REAL eval masks',
                endpoint_audit=end_check,endpoint_contract='only original common seam one/both ends; no interior removal',
                visible_curve=nonlinear,proposals=proposal+1,rejected_proposals=dict(reasons))
            return trimmed,plan,bands
        except ValueError as error:reasons[str(error)]+=1
    raise ValueError('no admissible TRAIN curve: '+repr(dict(reasons)))


def negative_trim(base,positive_plan,rng):
    # Same side-size recipe/profile/orientation, independently placed using
    # only this negative's area. There is no invented shared seam or Layout GT.
    side=selected_side(base,positive_plan['size_class']);mask=getattr(base,'mask_'+side)[0].astype(bool)
    params={k:positive_plan[k] for k in ('profile','angle_deg','flip','reverse')}
    pixels,scores,frame=curve_field(mask,**params);values=scores(pixels)
    removed=positive_plan['material_removed_fraction']
    if positive_plan['mode']=='both':qs=[removed/2,1-removed/2]
    elif positive_plan['lower'] is None:qs=[0.,1-removed]
    else:qs=[removed,1.]
    # Quantiles between pixels; ties stay together. Area cap is checked after.
    lower=None if qs[0]==0 else float(np.quantile(values,qs[0]))
    upper=None if qs[1]==1 else float(np.quantile(values,qs[1]))
    kept=mask_from_bounds(mask,pixels,values,lower,upper);loss=area_check(mask,kept)
    if abs(loss-removed)>.005:raise ValueError('negative matched area differs by >0.5pp')
    plan=dict(positive_plan,**frame,lower=lower,upper=upper,side=side,
        material_retained_fraction=1-loss,material_removed_fraction=loss,
        original_areas_px={s:int(getattr(base,'mask_'+s).sum()) for s in 'ab'},
        gt_seam_used=False,common_length_before_px=None,common_length_after_px=None,retained_fraction=None,
        endpoint_audit=None,endpoint_contract='negative has no common-seam/end GT',
        planning_basis='independent own-mask foreground quantile; no GT seam',
        visible_curve=visible_curve_nonlinearity(mask,kept,dict(positive_plan,**frame,lower=lower,upper=upper)))
    return crop(base,side,kept),plan


def augment(base_pair,old_primary,old_fields,old_reports,recipe,rng,mode,bank,size_class):
    return damage_augment(base_pair,old_primary,old_fields,old_reports,recipe,rng,mode,
        planners=(lambda b,p,r,m:positive_trim(b,p,r,m,bank,size_class),negative_trim))
