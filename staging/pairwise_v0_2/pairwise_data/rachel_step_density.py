"""S5 paired shared-arc-step inputs with unchanged v4 ownership supervision.

This experimental module never monkeypatches or changes a legacy sampler.
Physical masks, coarse inputs, labels and pose GT are preserved. Both sides
share a target-blind step, enlarged only when either complete loop hits cap.
"""
from dataclasses import replace
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .rachel_preprocess import _dense_external_contour, _arc_resample_closed
from .rachel_source_density import (source_boundary, shared_source_seams,
    confirm_fixed_e1_mask, project_to_source, inherit_dense_source_targets)
from .rachel_density_ownership_v2 import MAX_PROJECTION_RADIUS, EXCLUSION_MARGIN
from .rachel_density_ownership_v3 import (_segmented_seams, SOURCE_TRIM_RADIUS,
    TOKEN_IGNORE_RADIUS)
from .rachel_density_ownership_v4 import (_digest, _project_certified,
    certify_residual_outer_interface, PROJECTION_PROTOCOL)
from .rachel_clean_density import SCHEMA as CLEAN_REPORT_SCHEMA
from .rachel_training_dataset import _readonly

SAMPLING_SCHEMA = 'rachel-paired-shared-arc-step/1'
TARGET_SCHEMA = 'rachel-step-source-targets/1'
STEP_PX = 3.
CAP = 2048
SMOOTHING_SIGMA = 3.
PRESERVED_FIELDS = ('mask_a','mask_b','coarse_mask_a','coarse_mask_b','label',
    'translation_a_to_b_rc','translation_a_to_b_xy_cartesian','translation_valid')


def _settings(step_px, cap):
    if type(cap) is not int or cap != CAP or float(step_px) != STEP_PX:
        raise ValueError('S5 is predeclared step3px/cap2048; no implicit variant')


def _smooth(mask):
    array = np.asarray(mask).squeeze()
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError('finite 2D physical mask required')
    dense = _dense_external_contour(array.astype(bool))
    return ndimage.gaussian_filter1d(dense, sigma=SMOOTHING_SIGMA, axis=0, mode='wrap')


def sample_pair_step_contours(mask_a, mask_b, *, step_px=STEP_PX, cap=CAP):
    """Return (unpadded A, unpadded B, JSON-safe sampling receipt).

    For each closed loop L, N=ceil(L/delta); delta=max(3,L_A/2048,L_B/2048).
    Equal-arc interpolation makes the closing interval equal to all others.
    No pose/label/point-correspondence input is accepted by this API.
    """
    _settings(step_px, cap)
    dense = {'a':_smooth(mask_a), 'b':_smooth(mask_b)}
    lengths = {s:float(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).sum())
               for s,p in dense.items()}
    if any(not math.isfinite(v) or v <= 0 for v in lengths.values()):
        raise ValueError('zero or nonfinite smoothed contour perimeter')
    delta = max(float(step_px), *(v/cap for v in lengths.values()))
    points, sides = {}, {}
    for s in 'ab':
        ratio = lengths[s]/delta
        requested = max(4, int(math.ceil(ratio)))
        # Only floating rounding at the mathematically bounded ratio may clamp.
        if requested > cap and ratio > cap + 1e-9:
            raise ValueError('shared delta failed to satisfy cap')
        count = min(cap, requested)
        p = _readonly(_arc_resample_closed(dense[s], count), np.float32)
        if not np.isfinite(p).all() or len(np.unique(p,axis=0)) != count:
            raise ValueError('nonfinite or duplicated step-sampled points')
        points[s] = p
        sides[s] = dict(smoothed_perimeter_px=lengths[s], dense_points=len(dense[s]),
            count=count, requested_at_base_step=max(4,int(math.ceil(lengths[s]/step_px))),
            actual_arc_step_px=lengths[s]/count, cap_reached=count==cap,
            base_step_would_exceed_cap=lengths[s]>step_px*cap)
    detail = dict(schema_version=SAMPLING_SCHEMA, mode='paired_shared_arc_step',
        target_step_px=float(step_px), contour_cap=cap, smoothing_sigma=SMOOTHING_SIGMA,
        shared_delta_px=delta, shared_delta_enlarged=delta>step_px,
        cap_trigger_sides=[s for s in 'ab' if sides[s]['base_step_would_exceed_cap']],
        cap_policy='increase shared delta; preserve both complete closed loops',
        physical_masks_unchanged=True, physical_patch_windows_px=[7,16,32,64],
        GT_used=False, sides=sides)
    return points['a'], points['b'], detail


class StepInputContourResampler:
    """Target-blind REAL/OOD RachelBatch adapter; no source or labels consulted."""
    def __init__(self, step_px=STEP_PX, cap=CAP):
        _settings(step_px,cap)
        self.step_px, self.cap = float(step_px),cap
        self.last_details = []

    def __call__(self,batch):
        count=len(batch.pair_ids);updates={};receipts=[]
        for s in 'ab':
            updates['points_rc_'+s]=np.zeros((count,self.cap,2),np.float32)
            updates['contour_valid_'+s]=np.zeros((count,self.cap),bool)
            updates['target_'+s]=np.full((count,self.cap),-2,np.int64)
        for i in range(count):
            pa,pb,detail=sample_pair_step_contours(batch.mask_a[i],batch.mask_b[i],
                step_px=self.step_px,cap=self.cap)
            for s,p in zip('ab',(pa,pb)):
                updates['points_rc_'+s][i,:len(p)]=p
                updates['contour_valid_'+s][i,:len(p)]=True
            receipts.append(detail)
        self.last_details=receipts
        return replace(batch,**updates)


def _ignore(view,unsafe):
    view['component'][unsafe]=-1
    view['seam_s'][unsafe]=np.nan
    view['trusted'][unsafe]=False
    view['nonseam'][unsafe]=False


def assert_preserved(original,result):
    for field in PRESERVED_FIELDS:
        if not np.array_equal(getattr(original,field),getattr(result,field),equal_nan=True):
            raise ValueError('S5 modified physical/label/pose field: '+field)


def resample_source_step(sample,report,source,offsets,clean,*,step_px=STEP_PX,cap=CAP,
                         ownership_allowance_px=0.,origin_receipts=None):
    """Same v4 source resolver/projection policies, new paired-step point set.

    Existing v4 source proofs (including successful v1/v2/v3 paths) are used
    explicitly. This is not the legacy resampler under a mutable global hook.
    """
    _settings(step_px,cap)
    if report.get('schema_version') not in ('rachel-weathered-source-arc-e1/v1',CLEAN_REPORT_SCHEMA):
        raise ValueError('unknown source report; new-edge ancestry required')
    if report['schema_version']==CLEAN_REPORT_SCHEMA and (report.get('changed_pair') or
            any(report['side_'+s]['effective_applied'] for s in 'ab')):
        raise ValueError('clean evaluation cannot claim weathering')
    proofs=[v for v in (2,3,4) if report.get('source_ownership_v'+str(v)) is not None]
    if len(proofs)>1 or (proofs and not sample.label):
        raise ValueError('ownership proof must be exclusive and pair-positive')
    version=proofs[0] if proofs else 1
    proof=report.get('source_ownership_v'+str(version))
    if proof and proof.get('schema_version')!='density-local-unknown-ownership/'+str(version):
        raise ValueError('ownership proof schema mismatch')
    if version==3 and proof.get('mode')!='actual_pair_local_trim':
        raise ValueError('unknown v3 proof mode')
    if sample.label and np.any(np.asarray(source['a'],bool)&np.asarray(source['b'],bool)):
        raise ValueError('source known ownership is not exclusive')
    replay={s:confirm_fixed_e1_mask(clean[s],getattr(sample,'mask_'+s),report['side_'+s],
                (origin_receipts or {}).get(s)) for s in 'ab'}
    boundaries={s:source_boundary(source[s]) for s in 'ab'}
    raw_shared=set(boundaries['a'].keys)&set(boundaries['b'].keys) if sample.label else set()
    upper,coords=None,None
    if version in (3,4):
        retained={tuple(tuple(int(v) for v in p) for p in e) for e in proof['retained_source_edges']}
        seams,components,topology=_segmented_seams(boundaries,retained)
        if set(seams)!=retained or not components:
            raise ValueError('retained source edges differ from ownership proof')
        coords=np.asarray(proof['unknown_coordinates_parent_rc'],int)
        if version==4:
            upper={s:np.asarray(source[s],bool).copy() for s in 'ab'}
            for s in 'ab':
                if _digest(source[s])!=proof['lower_mask_sha256'][s]:raise ValueError('v4 lower mask differs')
                for r,c in proof['claimant_coordinates_by_side'][s]:upper[s][r,c]=True
                if _digest(upper[s])!=proof['upper_mask_sha256'][s]:raise ValueError('v4 upper mask differs')
            if certify_residual_outer_interface(source,upper,coords,SOURCE_TRIM_RADIUS)!=proof['outer_certificate']:
                raise ValueError('v4 residual certificate differs')
    elif sample.label:
        seams,components,topology=shared_source_seams(boundaries['a'],boundaries['b'])
    else:
        seams,components={},[]
        topology=dict(shared_cell_edges=0,accepted_edges=0,rejected_components=[],
            source_interface_not_computed_for_negative=True)
    if sample.label and not components:raise ValueError('positive pair has no trusted source seam')
    pa,pb,sampling=sample_pair_step_contours(sample.mask_a,sample.mask_b,step_px=step_px,cap=cap)
    updates,views,suppression,unsafe_v2={},{},{},{}
    for s,p in zip('ab',(pa,pb)):
        if version==4:
            view=_project_certified(p,offsets[s],boundaries[s],seams,source[s],upper[s],coords,
                radius=6.+replay[s]['max_depth_px']+ownership_allowance_px)
            dropped=np.asarray([j>=0 and boundaries[s].keys[j] in raw_shared-set(seams)
                                for j in view['source_edge_index']],bool)
            _ignore(view,dropped)
            view['reason_masks']['dropped_shared_edge']=dropped
            suppression[s]={k:int(v.sum()) for k,v in view['reason_masks'].items()}
        else:
            view=project_to_source(p,offsets[s],boundaries[s],seams,
                max_depth_px=replay[s]['max_depth_px'],ownership_allowance_px=ownership_allowance_px)
            if version in (2,3):
                radius=view['projection_radius_px']
                if radius>MAX_PROJECTION_RADIUS:raise ValueError('proof does not cover damage radius')
                tree=cKDTree(np.asarray(proof['unknown_coordinates_parent_rc'],float)+.5)
                parent=p+.5-np.asarray(offsets[s])
                ignore_radius=TOKEN_IGNORE_RADIUS if version==3 else radius+EXCLUSION_MARGIN
                unsafe=tree.query(parent)[0]<=ignore_radius
                if version==3:
                    ids=view['source_edge_index']
                    centers=(boundaries[s].starts[ids]+boundaries[s].ends[ids])*.5
                    unsafe|=tree.query(centers)[0]<=SOURCE_TRIM_RADIUS
                suppression[s]=dict(tokens=int(unsafe.sum()),radius_px=ignore_radius)
                if version==2:
                    # The v2 proof permits only uncertainty away from an
                    # existing positive descendant. Preserve that rejection
                    # rule, before the symmetric ignore is applied.
                    unsafe_v2[s]=unsafe
                else:
                    _ignore(view,unsafe)
        views[s]=view
        updates['points_rc_'+s]=p
        updates['contour_valid_'+s]=_readonly(np.ones(len(p),bool),np.bool_)
    if version==2:
        initial_a,initial_b,_=inherit_dense_source_targets(views['a'],views['b'],components)
        for s,target in zip('ab',(initial_a,initial_b)):
            prior=int(np.count_nonzero((target>=0)&unsafe_v2[s]))
            suppression[s]['prior_positive_tokens']=prior
            if prior:raise ValueError('density_unknown_owner_reaches_positive_descendant')
            _ignore(views[s],unsafe_v2[s])
    if sample.label:
        ta,tb,chosen=inherit_dense_source_targets(views['a'],views['b'],components)
        if not chosen:raise ValueError('positive pair has zero residual S5 correspondence; never silently relabel/drop')
    else:
        ta=np.full(len(pa),-1,np.int64);tb=np.full(len(pb),-1,np.int64);chosen=[]
    updates.update(target_a=_readonly(ta,np.int64),target_b=_readonly(tb,np.int64))
    result=replace(sample,**updates);assert_preserved(sample,result)
    detail=dict(schema_version=TARGET_SCHEMA,cap=cap,sampling=sampling,
        physical_masks_unchanged=True,gt_translation_used_for_matching=False,
        weathered_cross_fragment_nearest_neighbor=False,source_ownership_path=version,
        ownership_protocol='density-local-unknown-ownership/4',
        target_source='v4 resolved source-cell ancestry; new paired-step descendants',
        topology=topology,components=components,replay=replay,new_match_count=len(chosen),
        chosen_source_cells=[list(x) for x in chosen],uncertain_token_exclusion=suppression,
        projection_protocol=PROJECTION_PROTOCOL if version==4 else 'legacy-own-source-boundary-projection',
        sides={s:dict(points=len(views[s]['component']),trusted=int(views[s]['trusted'].sum()),
            ignored=int(np.count_nonzero(getattr(result,'target_'+s)==-2)),
            source_seam_tokens=int(np.count_nonzero(views[s]['component']>=0)),
            mean_step_px=views[s]['mean_step_px'],ambiguous_count=views[s]['ambiguous_count']) for s in 'ab'})
    derived=dict(report,density=detail,step_sampling=sampling,
        effective_supervised_match_count=len(chosen),inherited_match_count=len(chosen),
        ignored_token_count=int(np.count_nonzero(ta==-2)+np.count_nonzero(tb==-2)),
        inheritance_rule='v4 own-source ancestry; new paired shared-step descendants; no GT pose')
    return result,derived,views
