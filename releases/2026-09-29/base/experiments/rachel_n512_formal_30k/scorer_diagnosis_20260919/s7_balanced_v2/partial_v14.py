"""Partial crops with a retained-original-seam / smaller-fragment perimeter gate.

The primary-crop measurement is BEFORE the allowed1-3px background layer.
Only surviving TRAIN-supported original curve cells count; neither cut edges
nor the distance across a removed middle section are credited as seam length.
Negative crops have no seam/pose labels and share morphology, not invented GT.
"""
from collections import Counter
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..s7_compound_v1.geometry import partial_group, masks, area_ratio, rng_for, EIGHT
from staging.pairwise_v0_2.pairwise_data.rachel_partial_seam_dataset import (
    PartialSeamConfig, crop_training_pair, _ignore_ambiguous_nonmatches)
from staging.pairwise_v0_2.pairwise_data.rachel_strong_weathering import _eligible_runs
from .conservative_weather import contour
from .latent_seam import source_band


def mode_schedule(recipes, seed, profile, bins=None):
    result=[None]*len(recipes)
    if not profile.get('partial_min_smaller_perimeter_fraction'):
        return result
    ids=np.flatnonzero(np.asarray(recipes)=='partial')
    proportions=profile['partial_mode_percent']
    counts={k:len(ids)*v/100 for k,v in proportions.items()}
    if sum(proportions.values())!=100 or any(v!=int(v) for v in counts.values()):
        raise ValueError('Partial subtype quotas must be exactly representable')
    values=[k for k,v in counts.items() for _ in range(int(v))]
    rng=rng_for(seed,'partial-v14-modes')
    if bins is not None and proportions=={'end':50,'middle':50}:
        # Match subtype proportions inside the existing length strata to within
        # one pair, so a rare long stratum cannot end up entirely one subtype.
        strata={key:ids[np.asarray(bins)[ids]==key] for key in sorted(set(bins))}
        odd=[key for key,v in strata.items() if len(v)%2]
        extras=int(counts['end'])-sum(len(v)//2 for v in strata.values())
        end_extra=set(rng.permutation(odd)[:extras])
        for key,ii in strata.items():
            end=len(ii)//2+int(key in end_extra)
            for j,index in enumerate(rng.permutation(ii)):
                result[int(index)]='end' if j<end else 'middle'
        return result
    for i,value in zip(rng.permutation(ids),values):result[int(i)]=value
    return result


def retained_support(original, cropped, bands=None):
    if not original.label or not cropped.label:
        raise ValueError('retained seam has no definition for negative pairs')
    bands=source_band(original,bridge=0.) if bands is None else bands
    lengths={};original_lengths={};arrays={}
    for side,other,shift in (('a','b',original.translation_a_to_b_rc),
                              ('b','a',-original.translation_a_to_b_rc)):
        p,w,valid=bands[side];q,_,qvalid=bands[other]
        q=q[qvalid]
        if not valid.any() or not len(q):raise ValueError('no original supported curve')
        partner=q[cKDTree(q).query(p+shift)[1]]
        ip=np.rint(p).astype(int);iq=np.rint(partner).astype(int)
        keep=(getattr(cropped,'mask_'+side)[0,ip[:,0],ip[:,1]]>.5)&(
              getattr(cropped,'mask_'+other)[0,iq[:,0],iq[:,1]]>.5)
        # Credit an original contour segment only when BOTH its endpoints and
        # their partners survive. Never credit an edge ending in the cut.
        supported_edges=valid&np.roll(valid,-1)
        kept_edges=supported_edges&keep&np.roll(keep,-1)
        lengths[side]=float(w[kept_edges].sum());original_lengths[side]=float(w[supported_edges].sum())
        arrays.update({side+'_source_points':p[valid],side+'_source_weights':w[valid],
                       side+'_partner_points':partner[valid],side+'_physically_retained':kept_edges[valid]})
    areas={s:int(getattr(cropped,'mask_'+s).sum()) for s in 'ab'}
    perimeters={s:float(contour(getattr(cropped,'mask_'+s)[0])[1].sum()) for s in 'ab'}
    smaller=min('ab',key=lambda s:(areas[s],s))
    common=min(lengths.values())
    return dict(stage='after_primary_partial_before_background_light',
        numerator='minimum of bilateral surviving original TRAIN-supported curve lengths; no gap filling',
        denominator='full new outer perimeter of the smaller-by-pixel-area fragment, including new cut',
        common_retained_length_px=common,retained_length_by_side_px=lengths,
        original_length_by_side_px=original_lengths,fragment_area_px=areas,
        full_perimeter_px=perimeters,smaller_fragment=smaller,
        common_over_smaller_perimeter=common/perimeters[smaller]),arrays


def _archive_original(original):
    return {'partial_original_'+name:np.array(getattr(original,name),copy=True)
            for name in ('points_rc_a','points_rc_b','contour_valid_a','contour_valid_b','target_a','target_b')}


def _recess_frame(mask,dense):
    filled=ndimage.binary_fill_holes(np.pad(mask,1),structure=EIGHT)
    inward=ndimage.distance_transform_edt(filled)[1:-1,1:-1]-.5
    pixels=np.argwhere(mask)
    return pixels,inward[tuple(pixels.T)],cKDTree(dense).query(pixels)[1]


def _shape(arc,perimeter,center,width,profile):
    u=((arc-center+perimeter/2)%perimeter-perimeter/2)/(width/2)
    donor=np.asarray(profile,float)
    span=float(np.ptp(donor))
    donor=(donor-donor.min())/span if span>1e-9 else np.full_like(donor,.5)
    modulation=.70+.30*np.interp(np.clip((u+1)/2,0,1),np.linspace(0,1,len(donor)),donor)
    # A curved, open middle cut, not an enclosed hole or two separated fragments.
    taper=np.sin(np.pi*np.clip(1-np.abs(u),0,1)/2)**2
    return taper*modulation


def _at_depth(mask,frame,field,peak):
    pixels,inward,nearest=frame
    result=mask.copy();remove=inward<=peak*field[nearest]
    result[tuple(pixels[remove].T)]=False
    return result


def _enclosed_material_loss(original,cropped):
    """Reject rather than fill even a raster-sized enclosed crop artefact."""
    return any(np.any(ndimage.binary_fill_holes(new,structure=EIGHT)&~new&old)
               for old,new in zip(masks(original),masks(cropped)))


def _middle_group(positive,negative,bank,rng,minimum,max_attempts):
    config=PartialSeamConfig().crop_config
    pair=(positive,negative);reasons=Counter()
    choose_large=bool(rng.random()<.7)
    sides=[]
    for sample in pair:
        a,b=(m.sum() for m in masks(sample));small='a' if a<=b else 'b'
        sides.append(('b' if small=='a' else 'a') if choose_large else small)
    bands=source_band(positive,bridge=0.)
    pd,pe,eligible=bands[sides[0]];pa=np.r_[0.,np.cumsum(pe[:-1])];pp=float(pe.sum())
    runs=[r for r in _eligible_runs(eligible,pa,pe) if not r[2] and r[1]>=100]
    if not runs:return None,dict(reason='no_source_run_for_middle',attempts=0)
    selected=[getattr(s,'mask_'+side)[0].astype(bool) for s,side in zip(pair,sides)]
    nd,ne,na=contour(selected[1]);np_=float(ne.sum())
    frames=[_recess_frame(mask,dense) for mask,dense in zip(selected,(pd,nd))]
    for attempt in range(max_attempts):
        start,length,_=runs[int(rng.integers(len(runs)))]
        width=float(rng.uniform(.35,.50)*length)
        center=float((start+rng.uniform(.45,.55)*length)%pp)
        donor_index=int(rng.integers(len(bank.profiles)));donor_profile=bank.profiles[donor_index]
        field=_shape(pa,pp,center,width,donor_profile)*eligible
        # Macro Partial material removal, separate from the5-9px corrosion type.
        peak=float(np.clip(rng.uniform(.5,1.)*np.max(frames[0][1]),12.,96.))
        pm=_at_depth(selected[0],frames[0],field,peak)
        material=float(pm.sum()/selected[0].sum())
        nwidth=max(12.,min(.45*np_,width/pp*np_));ncenter=float(rng.uniform(0,np_))
        nfield=_shape(na,np_,ncenter,nwidth,donor_profile)
        lo,hi=0.,max(12.,float(frames[1][1].max())*4)
        for _ in range(14):
            middle=(lo+hi)/2
            test=_at_depth(selected[1],frames[1],nfield,middle)
            if test.sum()/selected[1].sum()>material:lo=middle
            else:hi=middle
        nm=_at_depth(selected[1],frames[1],nfield,hi)
        if abs(nm.sum()/selected[1].sum()-material)>.02:
            reasons['negative_material_not_matched']+=1;continue
        results=[crop_training_pair(s,side=side,retained_mask=m,config=config,topology_connectivity=2)
                 for s,side,m in zip(pair,sides,(pm,nm))]
        if not all(r.accepted for r in results):
            for i,r in enumerate(results):
                if not r.accepted:reasons[('positive:' if not i else 'negative:')+r.reason]+=1
            continue
        outputs=tuple(_ignore_ambiguous_nonmatches(old,r.sample,side,None)
                      for old,r,side in zip(pair,results,sides))
        if any(_enclosed_material_loss(old,new) for old,new in zip(pair,outputs)):
            reasons['new_enclosed_primary_loss']+=1;continue
        if any(area_ratio(new)<min(.125,area_ratio(old)) for old,new in zip(pair,outputs)):
            reasons['would_create_new_extreme_area_ratio']+=1;continue
        receipt,proof=retained_support(positive,outputs[0],bands)
        if receipt['common_over_smaller_perimeter']<minimum:
            reasons['common_curve_below15percent_smaller_perimeter']+=1;continue
        # Verify that this really removed an INTERIOR segment, with original
        # supported material surviving on both flanks, not just a recipe name.
        pixels=np.rint(pd).astype(int);survive=pm[tuple(pixels.T)]
        u=(pa-start)%pp;inrun=eligible&(u<length)
        relative=((pa-center+pp/2)%pp-pp/2)
        flank=[float(pe[inrun&survive&(relative<-width/2)].sum()),
               float(pe[inrun&survive&(relative>width/2)].sum())]
        removed=float(pe[inrun&~survive].sum())
        if min(flank)<max(16.,.08*length) or removed<.20*length:
            reasons['middle_not_effective_or_missing_flank']+=1;continue
        donor=bank.metadata['arcs'][donor_index]
        info=dict(applied=True,mode='middle',attempts=attempt+1,sides=sides,choose_large=choose_large,
            source_seam_retention=receipt['retained_length_by_side_px'][sides[0]]/receipt['original_length_by_side_px'][sides[0]],
            source_seam_length_px=receipt['original_length_by_side_px'][sides[0]],
            material_retention=[material,float(nm.sum()/selected[1].sum())],
            proposal=dict(kind='interior_open_curve_cut',run_start_arc_px=float(start),center_arc_px=center,width_px=width,peak_px=peak,
                          negative_center_arc_px=ncenter,negative_width_px=nwidth,negative_peak_px=hi),
            retained_flanks_px=flank,removed_middle_arc_px=removed,selected_run_length_px=float(length),
            donor_index=donor_index,donor_split=donor.get('split','train'),donor_lineage=donor['lineage'],donor_family=donor['family'],
            artificial_cut_exclusion_px=8.,attempt_reasons=dict(reasons),support_constraint=receipt)
        arrays=[_archive_original(s) for s in pair]
        for i,(d,field_i,peak_i) in enumerate(((pd,field,peak),(nd,nfield,hi))):
            arrays[i].update(partial_cut_points=d,partial_cut_field=field_i*peak_i)
        return outputs,dict(**info,_arrays=arrays)
    return None,dict(reason='middle_partial_rejected',attempts=max_attempts,attempt_reasons=dict(reasons))


def partial_group_v14(positive,negative,bank,rng,profile,mode,max_attempts=24):
    minimum=float(profile['partial_min_smaller_perimeter_fraction'])
    if not 0<minimum<1:raise ValueError('invalid retained curve fraction')
    if mode=='middle':return _middle_group(positive,negative,bank,rng,minimum,max_attempts)
    if mode!='end':raise ValueError('explicit end/middle Partial mode required')
    reasons=Counter()
    bands=source_band(positive,bridge=0.)
    for attempt in range(max_attempts):
        outputs,info=partial_group(positive,negative,bank,rng,max_attempts=1)
        if outputs is None:
            reasons.update(info.get('attempt_reasons',{}));continue
        if any(_enclosed_material_loss(old,new) for old,new in zip((positive,negative),outputs)):
            reasons['new_enclosed_primary_loss']+=1;continue
        receipt,_=retained_support(positive,outputs[0],bands)
        if receipt['common_over_smaller_perimeter']<minimum:
            reasons['common_curve_below15percent_smaller_perimeter']+=1;continue
        info.update(mode='end',attempts=attempt+1,support_constraint=receipt,
                    outer_attempt_reasons=dict(reasons),_arrays=[_archive_original(s) for s in (positive,negative)])
        return outputs,info
    return None,dict(reason='end_partial_rejected',attempts=max_attempts,attempt_reasons=dict(reasons))
