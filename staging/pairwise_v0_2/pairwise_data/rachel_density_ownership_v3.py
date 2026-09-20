"""Density v3: ignore a bounded ambiguous part of an actual native seam.

Every successful v2 sample takes the exact v2 path. Only the explicit
touches-actual-pair refusal adds a conservative source-edge trim and symmetric
token ignore. All retained source edges must be identical under every existing
owner assignment. Neither model masks/GT nor the common normalizer is changed.
"""
from dataclasses import replace
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .rachel_density_ownership_v2 import (
    SourceDensityRecordResolverV2,CleanSourceDensityV2Dataset,local_unknown_ownership,
    resample_source_density_v2,MAX_UNKNOWN_PIXELS,MAX_OWNER_ASSIGNMENTS,
    MAX_PROJECTION_RADIUS,EXCLUSION_MARGIN)
from .rachel_source_density import (SCHEMA as DENSITY_SCHEMA,SourceBoundary,source_boundary,
    shared_source_seams,confirm_fixed_e1_mask,project_to_source,inherit_dense_source_targets)
from .rachel_preprocess import extract_ordered_outer_contour
from .rachel_source_density_records import _mask
from .rachel_union_augmentation import UnionGeometryRejected
from .rachel_clean_density import SCHEMA as CLEAN_REPORT_SCHEMA
from .rachel_materialized_dataset import load_sample,save_sample
from .rachel_paired_density_dataset import file_sha256,validate_density_sample
from .rachel_training_dataset import _readonly

OWNERSHIP_PROTOCOL='density-local-unknown-ownership/3'
TRAIN_PIPELINE='paired-source-cell-density-materialization/3'
CLEAN_SCHEMA='rachel-clean-source-density-eval/3'
SOURCE_TRIM_RADIUS=MAX_PROJECTION_RADIUS+EXCLUSION_MARGIN  #15px
TOKEN_IGNORE_RADIUS=SOURCE_TRIM_RADIUS+MAX_PROJECTION_RADIUS+EXCLUSION_MARGIN  #30px, both sides


def _retained_keys(boundaries,tree):
    common=set(boundaries['a'].keys)&set(boundaries['b'].keys)
    if not common:return set()
    keys=sorted(common);centers=np.asarray([(np.asarray(e[0])+np.asarray(e[1]))*.5 for e in keys])
    keep=tree.query(centers)[0]>SOURCE_TRIM_RADIUS
    return {key for key,valid in zip(keys,keep) if valid}


def _segmented_seams(boundaries,retained):
    filtered={}
    for s,b in boundaries.items():
        idx=[i for i,k in enumerate(b.keys) if k in retained]
        filtered[s]=SourceBoundary(b.starts[idx],b.ends[idx],tuple(b.keys[i] for i in idx))
    return shared_source_seams(filtered['a'],filtered['b'])


def trimmed_actual_pair_ownership(parents,ab):
    """Only recover the exact v2 actual-pair refusal, never another error."""
    try:
        local_unknown_ownership(parents,ab)
    except UnionGeometryRejected as error:
        if error.reason!='density_unknown_owner_touches_actual_pair':raise
    else:raise ValueError('v3 seam trimming must not replace a successful v2 path')
    ids=sorted(parents);stack=np.stack([np.asarray(parents[k],bool) for k in ids])
    overlap=stack.sum(0)>1;fraction=float(overlap.sum()/max(1,stack.any(0).sum()))
    if fraction>.02:raise UnionGeometryRejected('ownership_overlap_fraction_exceeds_2_percent')
    depths=np.stack([ndimage.distance_transform_edt(m) for m in stack])
    second=np.sort(depths,axis=0)[-2];unknown=overlap&(second>3.)
    locations=np.argwhere(unknown);choices=[np.flatnonzero(stack[:,r,c]).tolist() for r,c in locations]
    count=int(np.prod([len(x) for x in choices]))
    if not 1<=len(locations)<=MAX_UNKNOWN_PIXELS or count>MAX_OWNER_ASSIGNMENTS:
        raise UnionGeometryRejected('density_v3_unknown_owner_proof_budget_exceeded')
    known=stack.copy();safe=overlap&~unknown
    known[:,safe]=np.arange(len(ids))[:,None]==depths[:,safe].argmax(0)[None,:]
    known[:,unknown]=False
    if not np.array_equal(known.any(0),stack.any(0)&~unknown) or not np.array_equal(known[:,~overlap],stack[:,~overlap]):
        raise ValueError('v3 known ownership changed exclusive source support')
    masks={k:known[i] for i,k in enumerate(ids)}
    boundaries={s:source_boundary(masks[k]) for s,k in zip('ab',ab)}
    tree=cKDTree(locations+.5);retained=_retained_keys(boundaries,tree)
    if not retained:raise UnionGeometryRejected('density_v3_no_unaffected_source_edges')
    for owners in itertools.product(*choices):
        variant=known.copy()
        for (r,c),owner in zip(locations,owners):variant[owner,r,c]=True
        candidate={s:source_boundary(variant[ids.index(k)]) for s,k in zip('ab',ab)}
        if _retained_keys(candidate,tree)!=retained:
            raise UnionGeometryRejected('density_v3_retained_interface_not_owner_invariant')
    seams,components,topology=_segmented_seams(boundaries,retained)
    if not components:raise UnionGeometryRejected('density_v3_zero_residual_continuous_seam')
    accepted=set(seams)
    proof=dict(schema_version=OWNERSHIP_PROTOCOL,mode='actual_pair_local_trim',actual_pair_members=list(ab),
        unknown_coordinates_parent_rc=locations.tolist(),unknown_pixels=len(locations),
        possible_existing_owner_assignments=count,all_assignments_preserve_retained_interface=True,
        maximum_group_overlap_fraction=.02,maximum_second_owner_depth_px=3.,
        observed_overlap_fraction=fraction,observed_second_depth_max_px=float(second[unknown].max()),
        max_unknown_pixels=MAX_UNKNOWN_PIXELS,max_owner_assignments=MAX_OWNER_ASSIGNMENTS,
        maximum_projection_radius_px=MAX_PROJECTION_RADIUS,exclusion_margin_px=EXCLUSION_MARGIN,
        source_trim_radius_px=SOURCE_TRIM_RADIUS,symmetric_token_ignore_radius_px=TOKEN_IGNORE_RADIUS,
        known_interface_edges_before_trim=len(set(boundaries['a'].keys)&set(boundaries['b'].keys)),
        retained_raw_edges=len(retained),retained_accepted_edges=len(accepted),
        retained_source_edges=[[list(a),list(b)] for a,b in sorted(accepted)],
        residual_components=components,topology=topology,
        unknown_owner_assigned=False,physical_mask_modified=False,GT_used=False)
    metadata=dict(method=OWNERSHIP_PROTOCOL,applied=True,exclusive_pixels_unchanged=True,
        reference_unknown_support_pixels=int(unknown.sum()),total_physical_support_unchanged=True,
        source_reference_has_explicit_unknowns=True)
    return masks,metadata,proof


def resample_source_density_v3(sample,report,source,offsets,clean,*,cap,ownership_allowance_px=0.,origin_receipts=None):
    proof=report.get('source_ownership_v3')
    if proof is None:
        return resample_source_density_v2(sample,report,source,offsets,clean,cap=cap,
            ownership_allowance_px=ownership_allowance_px,origin_receipts=origin_receipts)
    if (not sample.label or proof.get('schema_version')!=OWNERSHIP_PROTOCOL
            or proof.get('mode')!='actual_pair_local_trim' or report.get('source_ownership_v2') is not None):
        raise ValueError('v3 trimming requires exclusive verified actual-pair proof')
    if cap not in (512,1024) or report.get('schema_version') not in ('rachel-weathered-source-arc-e1/v1',CLEAN_REPORT_SCHEMA):
        raise ValueError('unknown density cap or augmentation source')
    if report['schema_version']==CLEAN_REPORT_SCHEMA and (report.get('changed_pair') or any(report['side_'+s]['effective_applied'] for s in 'ab')):
        raise ValueError('clean source cannot claim weathering')
    if np.any(np.asarray(source['a'],bool)&np.asarray(source['b'],bool)):
        raise ValueError('v3 source reference must have exclusive known ownership')
    replay={s:confirm_fixed_e1_mask(clean[s],getattr(sample,'mask_'+s),report['side_'+s],(origin_receipts or {}).get(s)) for s in 'ab'}
    boundaries={s:source_boundary(source[s]) for s in 'ab'}
    retained={tuple(tuple(int(x) for x in v) for v in edge) for edge in proof['retained_source_edges']}
    seams,components,topology=_segmented_seams(boundaries,retained)
    if set(seams)!=retained or not components:raise ValueError('v3 retained source proof differs from source geometry')
    tree=cKDTree(np.asarray(proof['unknown_coordinates_parent_rc'],float)+.5)
    updates,views,suppression={},{},{}
    for s in 'ab':
        points,valid=extract_ordered_outer_contour(np.asarray(getattr(sample,'mask_'+s)).squeeze().astype(bool),cap=cap,smoothing_sigma=3.)
        if len(np.unique(points,axis=0))!=len(points):raise ValueError('duplicated density points')
        updates['points_rc_'+s]=points;updates['contour_valid_'+s]=valid
        view=project_to_source(points,offsets[s],boundaries[s],seams,
            max_depth_px=replay[s]['max_depth_px'],ownership_allowance_px=ownership_allowance_px)
        if view['projection_radius_px']>MAX_PROJECTION_RADIUS:
            raise ValueError('v3 proof excludes this damage depth')
        parent_points=points+.5-np.asarray(offsets[s]);unsafe=tree.query(parent_points)[0]<=TOKEN_IGNORE_RADIUS
        ids=view['source_edge_index'];edgecenters=(boundaries[s].starts[ids]+boundaries[s].ends[ids])*.5
        projects_to_trim=tree.query(edgecenters)[0]<=SOURCE_TRIM_RADIUS
        unsafe|=projects_to_trim
        suppression[s]=dict(tokens=int(unsafe.sum()),radius_px=TOKEN_IGNORE_RADIUS,
            projects_into_source_trim=int(projects_to_trim.sum()),
            candidate_seam_tokens_inside_exclusion=int(np.count_nonzero((view['component']>=0)&unsafe)))
        view['component'][unsafe]=-1;view['seam_s'][unsafe]=np.nan
        view['trusted'][unsafe]=False;view['nonseam'][unsafe]=False
        views[s]=view
    ta,tb,chosen=inherit_dense_source_targets(views['a'],views['b'],components)
    if not chosen:raise UnionGeometryRejected('density_v3_zero_residual_positive_correspondence')
    updates.update(target_a=_readonly(ta,np.int64),target_b=_readonly(tb,np.int64));result=replace(sample,**updates)
    detail=dict(schema_version=DENSITY_SCHEMA,cap=cap,physical_masks_unchanged=True,
        gt_translation_used_for_matching=False,weathered_cross_fragment_nearest_neighbor=False,
        target_source='owner-invariant residual source-cell segments; local ambiguous seam and tokens ignored',
        topology=topology,components=components,replay=replay,new_match_count=len(chosen),
        old_match_count=int(np.count_nonzero(sample.target_a>=0)),chosen_source_cells=[list(x) for x in chosen],
        source_ownership_v3=proof,uncertain_token_exclusion=suppression,
        sides={s:dict(points=len(views[s]['component']),trusted=int(views[s]['trusted'].sum()),
            ignored=int(np.count_nonzero(getattr(result,'target_'+s)==-2)),
            source_seam_tokens=int(np.count_nonzero(views[s]['component']>=0)),mean_step_px=views[s]['mean_step_px'],
            projection_radius_px=views[s]['projection_radius_px'],ambiguous_count=views[s]['ambiguous_count']) for s in 'ab'})
    fields=('effective_supervised_match_count','inherited_match_count','ignored_token_count','inheritance_rule')
    derived=dict(report,density=detail,original_e1_target_summary={k:report[k] for k in fields},
        effective_supervised_match_count=len(chosen),inherited_match_count=len(chosen),
        ignored_token_count=int(np.count_nonzero(ta==-2)+np.count_nonzero(tb==-2)),
        inheritance_rule='owner-invariant residual clean source-cell IDs; symmetric ambiguous-neighborhood ignore')
    return result,derived,views


def _recover_native_v3(resolver,row,sample,report,root):
    parts=row['fragment_a']['fragment_token'].split('/');group,path=resolver._group(parts[1],parts[2])
    fragments={s:next(f for f in group['fragments'] if f['fragment_token']==row['fragment_'+s]['fragment_token']) for s in 'ab'}
    parents=resolver._parents(group);ab=[fragments[s]['fragment_id'] for s in 'ab']
    masks,metadata,proof=trimmed_actual_pair_ownership(parents,ab)
    offsets={s:fragments[s]['target_audit']['parent_to_model_offset_rc'] for s in 'ab'}
    clean={s:_mask(Path(root)/row['fragment_'+s]['model_mask_path']) for s in 'ab'}
    for s in 'ab':resolver._verify_centerpad(parents[fragments[s]['fragment_id']],clean[s],offsets[s])
    return dict(report,source_ownership_v3=proof),{s:masks[fragments[s]['fragment_id']] for s in 'ab'},offsets,clean,metadata,str(path)


class SourceDensityRecordResolverV3(SourceDensityRecordResolverV2):
    def resolve(self,index):
        try:return super().resolve(index)
        except UnionGeometryRejected as error:
            entry=self.entries[index]
            if error.reason!='density_unknown_owner_touches_actual_pair' or entry['source_stratum']!='native_positive' or not entry['label']:raise
        entry=self.entries[index];row=entry['source_row'];sample,report=load_sample(self.artifact_root/entry['artifact_path'])
        report,source,offsets,clean,metadata,path=_recover_native_v3(self,row,sample,report,entry['source_root'])
        provenance=dict(index=index,pair_id=sample.pair_id,label=True,source_stratum=entry['source_stratum'],
            fixed_artifact=str(self.artifact_root/entry['artifact_path']),source_root=entry['source_root'],
            groups=[path,path],ownership_reference_only=True,ownership_normalization=metadata,
            source_ownership_v3=report['source_ownership_v3'],parent_to_model_offsets=offsets)
        return sample,report,source,offsets,clean,3.,provenance

    def sample_at(self,index,cap):
        sample,report,source,offsets,clean,allowance,provenance=self.resolve(index)
        result,derived,views=resample_source_density_v3(sample,report,source,offsets,clean,cap=cap,ownership_allowance_px=allowance)
        derived['source_resolution']=provenance
        return result,derived,views


class CleanSourceDensityV3Dataset(CleanSourceDensityV2Dataset):
    def __init__(self,root,split,contour_cap=512,cache_dir=None):
        super().__init__(root,split,contour_cap,cache_dir=None)
        protocol={k:v for k,v in self.protocol.items() if k!='identity_sha256'}
        protocol.update(schema_version=CLEAN_SCHEMA,source_density_version='v3',ownership_protocol=OWNERSHIP_PROTOCOL,
            density_v3_module_sha256=file_sha256(__file__))
        self.identity=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.protocol=dict(protocol,identity_sha256=self.identity);self.cache_dir=Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir:
            if self.cache_dir==self.root or self.root in self.cache_dir.parents:raise ValueError('v3 cache must not modify original release')
            self.cache_dir.mkdir(parents=True,exist_ok=True);path=self.cache_dir/'cache_identity.json'
            if path.exists():
                if json.loads(path.read_text())!=self.protocol:raise ValueError('v3 cache split/cap/source identity differs')
            else:
                temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(self.protocol,indent=2)+'\n');os.replace(temp,path)

    def _resolve_clean(self,index):
        try:return super()._resolve_clean(index)
        except UnionGeometryRejected as error:
            row=self.rows[index]
            if error.reason!='density_unknown_owner_touches_actual_pair' or not row['label']:raise
        sample=self.base[index];count=int(np.count_nonzero(sample.target_a>=0))
        report=dict(schema_version=CLEAN_REPORT_SCHEMA,pair_id=sample.pair_id,source_split=self.split,
            changed_pair=False,changed_a=False,changed_b=False,weathering_applied=False,
            side_a={'effective_applied':False},side_b={'effective_applied':False},pose_supervision_enabled=True,
            original_gt_translation_preserved=True,effective_supervised_match_count=count,inherited_match_count=count,
            ignored_token_count=int(np.count_nonzero(sample.target_a==-2)+np.count_nonzero(sample.target_b==-2)),
            inheritance_rule='original clean512 targets used only as an audit count; never reindexed')
        report,source,offsets,clean,metadata,path=_recover_native_v3(self,row,sample,report,self.root)
        report['source_resolution']=dict(groups=[path,path],parent_to_model_offsets=offsets,ownership_reference_only=True,
            ownership_normalization=metadata,label_origin=row['label_origin'],negative_origin=row.get('negative_origin'),
            source_ownership_v3=report['source_ownership_v3'])
        return sample,report,source,offsets,clean,3.

    def weathered(self,index):
        row=self.rows[index];path=self.cache_dir/(hashlib.sha256(row['pair_id'].encode()).hexdigest()+'.npz') if self.cache_dir else None
        if path and path.exists():
            sample,report=load_sample(path)
            if sample.pair_id!=row['pair_id'] or bool(sample.label)!=bool(row['label']):raise ValueError('v3 cached labels differ')
            validate_density_sample(sample,report,self.contour_cap,self.identity);return sample,report
        original,report,source,offsets,clean,allowance=self._resolve_clean(index)
        sample,derived,_=resample_source_density_v3(original,report,source,offsets,clean,cap=self.contour_cap,ownership_allowance_px=allowance)
        derived['paired_density']=dict(identity_sha256=self.identity,source_pair_id=sample.pair_id,
            source_manifest=str(self.manifest_path),source_split=self.split)
        for field in ('mask_a','mask_b','coarse_mask_a','coarse_mask_b','label','translation_a_to_b_rc','translation_a_to_b_xy_cartesian','translation_valid'):
            if not np.array_equal(getattr(original,field),getattr(sample,field),equal_nan=True):raise ValueError('v3 original field changed: '+field)
        validate_density_sample(sample,derived,self.contour_cap,self.identity)
        if path:save_sample(path,sample,derived)
        return sample,derived
