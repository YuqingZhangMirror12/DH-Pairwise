"""Pair-shared raster scale augmentation with the exact affine GT transform."""
from dataclasses import replace
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image


def pair_shared_scale(sample,target_mean_area,*,minimum=.55,maximum=1.8,topology_backoff=False,identity_fallback=False):
    masks=[np.asarray(getattr(sample,'mask_'+s)[0],bool) for s in 'ab']
    coordinates=[np.argwhere(m) for m in masks]
    boxes=[(c.min(0),c.max(0)+1) for c in coordinates]
    before=np.mean([m.sum() for m in masks])
    requested=float(np.sqrt(target_mean_area/before))
    fit=min(780./float(max(hi-lo)) for lo,hi in boxes)
    initial=float(np.clip(requested,minimum,min(maximum,fit)))
    identity=float(np.clip(1.,minimum,min(maximum,fit)))
    candidates=np.linspace(initial,identity,11) if topology_backoff else [initial]
    rejected=[];rendered=None
    for candidate in candidates:
        rendered=[];offsets=[]
        for mask,(lo,hi) in zip(masks,boxes):
            center=(lo+hi-1)/2
            offset=np.array([399.5,399.5])-candidate*center
            affine=np.array([[candidate,0,offset[1]],[0,candidate,offset[0]]],float)
            new=cv2.warpAffine(np.uint8(mask),affine,(800,800),flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
            if not new.any() or ndimage.label(new,np.ones((3,3),bool))[1]!=1:
                rendered=None;break
            rendered.append(new);offsets.append(offset)
        if rendered is not None:
            scale=float(candidate);break
        rejected.append(float(candidate))
    untouched=False
    if rendered is None and identity_fallback:
        # A >780px source cannot reach scale1 under the optional10px margin.
        # Preserve the exact original800px raster rather than force a destructive
        # shrink, fill a thin bridge, discard material or change source identity.
        if all(m.shape==(800,800) and m.any() and ndimage.label(m,np.ones((3,3),bool))[1]==1 for m in masks):
            rendered=[m.copy() for m in masks];offsets=[np.zeros(2),np.zeros(2)]
            scale=1.;untouched=True
    if rendered is None:
        raise ValueError('scale would empty or disconnect material')
    changes={}
    for side,new,offset in zip('ab',rendered,offsets):
        points=getattr(sample,'points_rc_'+side).copy()
        valid=getattr(sample,'contour_valid_'+side)
        points[valid]=scale*points[valid]+offset
        if (points[valid]<0).any() or (points[valid]>799).any():
            raise ValueError('scaled point outside canvas')
        coarse=np.asarray(Image.fromarray(np.uint8(new)*255).resize((128,128),Image.Resampling.NEAREST))>0
        changes.update({'mask_'+side:np.ascontiguousarray(new[None],np.float32),
                        'coarse_mask_'+side:np.ascontiguousarray(coarse[None],np.float32),
                        'points_rc_'+side:points})
    t=sample.translation_a_to_b_rc.copy()
    if sample.translation_valid:t=scale*t+offsets[1]-offsets[0]
    changes['translation_a_to_b_rc']=np.asarray(t,np.float32)
    changes['translation_a_to_b_xy_cartesian']=np.array([t[1],-t[0]],np.float32)
    after=np.mean([changes['mask_'+s].sum() for s in 'ab'])
    return replace(sample,**changes),dict(requested_mean_area_px2=float(target_mean_area),
        before_mean_area_px2=float(before),after_mean_area_px2=float(after),
        common_scale=scale,requested_scale=requested,clipped=abs(requested-scale)>1e-8,
        topology_backoff_used=bool(rejected),topology_rejected_scales=rejected,
        untouched_identity_fallback=untouched,
        topology_rule='tested common scales toward identity; optional exact original raster when all fail; no closing/filling/part removal',
        offsets_rc=[o.tolist() for o in offsets],same_scale_for_both_fragments=True,
        point_identity_and_correspondence_preserved=True,
        point_smoothing_radius_scaled_with_geometry=True,no_rotation=True)
