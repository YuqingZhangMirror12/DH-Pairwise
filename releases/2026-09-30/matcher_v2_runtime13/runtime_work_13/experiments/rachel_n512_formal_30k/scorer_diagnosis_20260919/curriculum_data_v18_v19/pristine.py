"""Paired unaltered ORIGINAL seam: no post-crop or correspondence denominator.

A segment counts only when both endpoints, their frozen original partners and
their 3x3 raster neighbourhoods are unchanged. No ray projection or tolerance
for erosion is used to claim pristine contact. Raster contact itself is defined
by the original GT-supported common curve (not by a new zero-gap fit).
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from ..s7_balanced_v2.latent_seam import source_band
from .weather import contour
from ..s7_compound_v1.geometry import _eligible_runs, _signed_arc_distance


def unchanged_vertices(original, final, points, radius=1):
    changed = np.asarray(original, bool) != np.asarray(final, bool)
    nearby = ndimage.maximum_filter(changed, size=2*radius+1, mode='constant')
    ij = np.rint(points).astype(int)
    return ~nearby[tuple(ij.T)]


def measure(base, final, bands=None, radius=1):
    if not base.label:
        raise ValueError('negative has no common-seam denominator')
    bands = source_band(base, bridge=0.) if bands is None else bands
    arrays = {}; original_lengths = {}; lengths = {}; ratios = {}; guards = {}
    for side, other, shift in [('a','b',base.translation_a_to_b_rc),
                               ('b','a',-base.translation_a_to_b_rc)]:
        p,w,valid = bands[side]; q,_,qvalid = bands[other]
        if not valid.any() or not qvalid.any():
            raise ValueError('original common curve missing')
        partner = q[qvalid][cKDTree(q[qvalid]).query(p+shift)[1]]
        # Denominator never depends on crop, corrosion, surviving labels, or
        # resolved rays. Missing/altered members count against the numerator.
        source_edges = valid & np.roll(valid,-1)
        keep = unchanged_vertices(getattr(base,'mask_'+side)[0],getattr(final,'mask_'+side)[0],p,radius)
        keep &= unchanged_vertices(getattr(base,'mask_'+other)[0],getattr(final,'mask_'+other)[0],partner,radius)
        keep &= np.linalg.norm(p+shift-partner,axis=1)<=3.+1e-6
        edges = source_edges & keep & np.roll(keep,-1)
        original_lengths[side] = float(w[source_edges].sum())
        lengths[side] = float(w[edges].sum())
        if original_lengths[side]<=0:raise ValueError('zero original seam')
        ratios[side] = lengths[side]/original_lengths[side]
        endpoints = edges | np.roll(edges,1)
        guards.setdefault(side,[]).append(p[endpoints])
        guards.setdefault(other,[]).append(partner[endpoints])
        arrays.update({side+'_points':p,side+'_partner_points':partner,side+'_edge':w,
            side+'_source_edges':source_edges,side+'_pristine_edges':edges})
    guards={s:np.unique(np.concatenate(v),axis=0) for s,v in guards.items()}
    summary=dict(original_common_length_px=min(original_lengths.values()),
        pristine_common_length_px=min(lengths.values()),
        pristine_common_fraction=min(lengths.values())/min(original_lengths.values()),
        original_length_by_side_px=original_lengths,pristine_length_by_side_px=lengths,
        pristine_fraction_by_side=ratios,conservative_fraction=min(ratios.values()),
        definition='original pre-additional-crop GT-supported arc; both endpoints and original partners unchanged on BOTH sides',
        exact_neighborhood_radius_px=radius,denominator_reduced_by_crop_or_erosion=False)
    return summary,arrays,guards


def require(summary,minimum=.25):
    if summary['conservative_fraction']<minimum or summary['pristine_common_fraction']<minimum:
        raise ValueError('strict paired pristine arc below25% original pre-crop seam')


def guard_fractions(base,guards):
    fractions={}
    for s in 'ab':
        p,w,_=contour(getattr(base,'mask_'+s)[0])
        membership=cKDTree(guards[s]).query(p)[0]<.25 if len(guards[s]) else np.zeros(len(p),bool)
        fractions[s]=float(w[membership&np.roll(membership,-1)].sum()/w.sum())
    return fractions


def negative_guards(base,current,fractions,rng):
    """Comparable untouched contour mass, independent random positions, no GT.

    Do not make the positive-only protected-contact rule a trivial global
    corrosion-amount label cue. No partners or seam correctness exist here.
    """
    guards={};info={}
    for s in 'ab':
        old=getattr(base,'mask_'+s)[0];new=getattr(current,'mask_'+s)[0]
        p,w,arc=contour(old);available=unchanged_vertices(old,new,p)
        runs=_eligible_runs(available,arc,w)
        total=sum(r[1] for r in runs);target=float(fractions[s]*w.sum())
        if total<target+4:raise ValueError('negative independent preservation control cannot fit')
        bits=np.zeros(len(p),bool)
        for start,length,circular in runs:
            width=target*length/total
            if width<2:continue
            center=((rng.uniform(0,w.sum()) if circular else start+width/2+rng.uniform(0,length-width)))%w.sum()
            bits|=(np.abs(_signed_arc_distance(arc,center,float(w.sum())))<=width/2)&available
        guards[s]=p[bits]
        info[s]=dict(target_original_perimeter_fraction=fractions[s],
            actual_original_perimeter_fraction=float(w[bits&np.roll(bits,-1)].sum()/w.sum()),
            gt_seam_used=False,placement='independent own-contour intervals; morphology control only')
    return guards,info


def audit_exact(base,final,bands,recorded,arrays,minimum=.25):
    """Independent offset-by-offset pixel test (not the generator's filter)."""
    originals={};retained={};ratios={}
    for s,other,shift in [('a','b',base.translation_a_to_b_rc),('b','a',-base.translation_a_to_b_rc)]:
        p,w,valid=bands[s];q,_,qvalid=bands[other]
        partner=q[qvalid][cKDTree(q[qvalid]).query(p+shift)[1]]
        good=np.linalg.norm(p+shift-partner,axis=1)<=3.+1e-6
        for side,points in [(s,p),(other,partner)]:
            old=np.asarray(getattr(base,'mask_'+side)[0],bool);new=np.asarray(getattr(final,'mask_'+side)[0],bool)
            ij=np.rint(points).astype(int)
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    shifted=ij+np.array([dr,dc]);inside=(shifted>=0).all(1)&(shifted<np.array(old.shape)).all(1)
                    indices=shifted[inside]
                    good[inside]&=(old[tuple(indices.T)]==new[tuple(indices.T)])
        denominator=valid&np.roll(valid,-1);numerator=denominator&good&np.roll(good,-1)
        for name,expected in [('points',p),('partner_points',partner),('edge',w),('source_edges',denominator),('pristine_edges',numerator)]:
            if not np.array_equal(arrays[s+'_'+name],expected):raise AssertionError('pristine proof independently disagrees:'+s+name)
        originals[s]=float(w[denominator].sum());retained[s]=float(w[numerator].sum());ratios[s]=retained[s]/originals[s]
    if min(ratios.values())<minimum:raise AssertionError('less than25% original seam is unmodified on both sides')
    for key,expected in [('original_length_by_side_px',originals),('pristine_length_by_side_px',retained),('pristine_fraction_by_side',ratios)]:
        if expected!=recorded[key]:raise AssertionError('pristine summary mismatch:'+key)
    return dict(status='passed',original_length_by_side_px=originals,pristine_length_by_side_px=retained,
        fraction_by_side=ratios,minimum=min(ratios.values()),independent_raster_neighborhood_check=True,
        negative_gt_not_fabricated=True)
