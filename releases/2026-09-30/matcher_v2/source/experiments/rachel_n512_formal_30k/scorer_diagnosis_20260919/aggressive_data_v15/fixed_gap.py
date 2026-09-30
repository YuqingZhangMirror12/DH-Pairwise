"""Future generator measurement: no partner rematching after structural trim.

This local preparation patch does not alter bound source03 or pilot03 records.
The independently reconstructed projection_audit remains the review evidence.
"""
import numpy as np
from ..s7_balanced_v2.latent_seam import EDGES,ray_project


def source_pairs(proof,side,shift):
    keep=proof[side+'_physically_retained']
    p=proof[side+'_source_points'][keep];q=proof[side+'_partner_points'][keep]
    w=proof[side+'_source_weights'][keep]
    good=np.linalg.norm(p+shift-q,axis=1)<=3.+1e-6
    return p,q,w,good


def measure_fixed(before,after,proof):
    arrays={};sides=[];values=[];weights=[];ambiguous=0.;ray_fail=0.
    for side,other,shift in [('a','b',before.translation_a_to_b_rc),('b','a',-before.translation_a_to_b_rc)]:
        p,q,w,source_valid=source_pairs(proof,side,shift)
        if not len(p):return None,None
        pa,da,va=ray_project(getattr(before,'mask_'+side)[0],getattr(after,'mask_'+side)[0],p)
        pb,db,vb=ray_project(getattr(before,'mask_'+other)[0],getattr(after,'mask_'+other)[0],q)
        valid=source_valid&va&vb;gap=np.linalg.norm(pa+shift-pb,axis=1)
        arrays.update({side+'_'+k:v.astype(np.float32) if v.dtype!=bool else v for k,v in dict(
            source_points=p,partner_source_points=q,source_weight=w,source_valid=source_valid,
            projected_points=pa,partner_projected_points=pb,gap=gap,valid=valid,
            source_gap=np.linalg.norm(p+shift-q,axis=1),recession=da,partner_recession=db).items()})
        sides.append(dict(side=side,source_length_px=float(w.sum()),source_points=len(p),
            unresolved_length_px=float(w[~valid].sum())))
        ambiguous+=float(w[~source_valid].sum());ray_fail+=float(w[source_valid&~(va&vb)].sum())
        values.extend(gap[valid]);weights.extend(w[valid])
    if not values:return None,None
    v=np.asarray(values);w=np.asarray(weights);denom=sum(s['source_length_px'] for s in sides)
    hist=np.histogram(v,EDGES,weights=w)[0]
    summary=dict(definition='pre-cut original TRAIN source partners frozen; source ambiguity and failed rays separately unresolved; no final-gap truncation',
        source_length_px=denom/2,sides=sides,ray_resolved_fraction=float(w.sum()/denom),
        source_ambiguous_fraction=ambiguous/denom,ray_unresolved_fraction=ray_fail/denom,
        gap_mean_px=float(np.average(v,weights=w)),gap_min_px=float(v.min()),gap_max_px=float(v.max()),
        gap_p10_px=float(np.quantile(v,.1)),gap_p50_px=float(np.median(v)),gap_p90_px=float(np.quantile(v,.9)),
        gap_bins=[None if not np.isfinite(x) else float(x) for x in EDGES],
        gap_arc_length_counts=hist.tolist(),gap_share=(hist/hist.sum()).tolist(),
        fraction_under5=float(w[v<=5].sum()/w.sum()),fraction_over15=float(w[v>=15].sum()/w.sum()),
        has_near_and_far=bool(w[v<=5].sum()>=5 and w[v>=15].sum()>=10))
    return summary,arrays
