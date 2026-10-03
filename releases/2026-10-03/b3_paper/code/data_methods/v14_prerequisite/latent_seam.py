"""Keep a fixed pre-corrosion seam denominator; never truncate by final gap.

This is a diagnostic projection, not new correspondence supervision. The source
band is supported by TRAIN GT anchors after partial cutting and common scaling.
Post-corrosion locations are first remaining material on each old inward normal.
Unresolved rays are counted, not silently dropped or converted to perfect gaps.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from ..s7_compound_v1.geometry import eligible_seam
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour

EDGES=np.array([0,2,4,8,15,20,30,40,60,100,np.inf],float)


def contour(mask):
    p,_=extract_ordered_outer_contour(mask,cap=mask.size,smoothing_sigma=0.)
    p=np.asarray(p,float)
    return p,np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)


def source_band(sample,guards=None,radius=8.,tolerance=3.,bridge=0.):
    bands={}
    for side in 'ab':
        p,w=contour(getattr(sample,'mask_'+side)[0])
        good=eligible_seam(sample,side,p,True,cut_guard=[] if guards is None else guards[side],
            cut_guard_radius=radius,contact_tolerance=tolerance,bridge_short_gaps_px=bridge)
        bands[side]=(p,w,good)
    return bands


def ray_project(oldmask,newmask,points):
    smooth=ndimage.gaussian_filter(np.asarray(oldmask,float),2.)
    grad=np.stack(np.gradient(smooth),axis=-1)
    ij=np.clip(np.rint(points).astype(int),0,np.array(oldmask.shape)-1)
    n=grad[ij[:,0],ij[:,1]]
    norm=np.linalg.norm(n,axis=1)
    n=n/np.maximum(norm[:,None],1e-9)
    depths=np.arange(0.,64.01,.5)
    rays=points[:,None,:]+depths[None,:,None]*n[:,None,:]
    hit=ndimage.map_coordinates(np.asarray(newmask,float),
        [rays[:,:,0],rays[:,:,1]],order=0,mode='constant',cval=0)>.5
    valid=hit.any(1)&(norm>1e-6)
    first=hit.argmax(1)
    return rays[np.arange(len(points)),first],depths[first],valid


def measure(before,after,bands=None):
    if not before.label:return None,None
    bands=source_band(before) if bands is None else bands
    records={};per_side=[];weights=[];gaps=[]
    for side,other,shift in (('a','b',before.translation_a_to_b_rc),
                              ('b','a',-before.translation_a_to_b_rc)):
        p,w,bits=bands[side];q,_,qbits=bands[other]
        p=p[bits];w=w[bits];q=q[qbits]
        if not len(p) or not len(q):return None,None
        j=cKDTree(q).query(p+shift)[1];q=q[j]
        pa,da,va=ray_project(getattr(before,'mask_'+side)[0],getattr(after,'mask_'+side)[0],p)
        pb,db,vb=ray_project(getattr(before,'mask_'+other)[0],getattr(after,'mask_'+other)[0],q)
        valid=va&vb
        distance=np.linalg.norm(pa+shift-pb,axis=1)
        source_distance=np.linalg.norm(p+shift-q,axis=1)
        for name,value in dict(source_points=p,partner_source_points=q,projected_points=pa,
                partner_projected_points=pb,gap=distance,source_gap=source_distance,
                source_weight=w,valid=valid,recession=da,partner_recession=db).items():
            records[side+'_'+name]=value.astype(np.float32) if value.dtype!=bool else value
        per_side.append(dict(side=side,source_length_px=float(w.sum()),
            unresolved_length_px=float(w[~valid].sum()),source_points=len(p)))
        gaps.append(distance[valid]);weights.append(w[valid])
    v=np.concatenate(gaps);w=np.concatenate(weights)
    if not len(v):return None,None
    counts=np.histogram(v,EDGES,weights=w)[0];share=counts/counts.sum()
    lengths=[x['source_length_px'] for x in per_side]
    summary=dict(definition='fixed pre-weather TRAIN-GT-supported band; inward-normal projection; no final-gap cutoff; not correspondence labels',
        source_length_px=float(np.mean(lengths)),sides=per_side,
        ray_resolved_fraction=float(w.sum()/sum(lengths)),
        gap_mean_px=float(np.average(v,weights=w)),gap_min_px=float(v.min()),gap_max_px=float(v.max()),
        gap_p10_px=float(np.quantile(v,.1)),gap_p50_px=float(np.median(v)),gap_p90_px=float(np.quantile(v,.9)),
        gap_bins=[None if not np.isfinite(x) else float(x) for x in EDGES],
        gap_arc_length_counts=counts.tolist(),gap_share=share.tolist(),
        fraction_under5=float(w[v<=5].sum()/w.sum()),fraction_over15=float(w[v>=15].sum()/w.sum()),
        has_near_and_far=bool(w[v<=5].sum()>=5 and w[v>=15].sum()>=10))
    return summary,records
