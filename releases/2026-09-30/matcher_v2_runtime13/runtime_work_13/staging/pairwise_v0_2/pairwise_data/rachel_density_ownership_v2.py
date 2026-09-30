"""Version2 density-only isolation of remote, locally unknown source owners.

The common union normalizer is untouched. Its normal path is returned exactly;
only native positives may recover a bounded3px-depth failure if every possible
existing owner preserves the relevant A/B interface and that interface is far
away. Physical model masks/GT/labels remain unchanged. Union stays strict.
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

from .rachel_source_density_records import SourceDensityRecordResolver,_mask
from .rachel_source_density import (source_boundary,shared_source_seams,
    resample_source_density,inherit_dense_source_targets)
from .rachel_union_augmentation import normalize_boundary_ownership,UnionGeometryRejected
from .rachel_clean_density import CleanSourceDensityDataset,SCHEMA as CLEAN_REPORT_SCHEMA
from .rachel_materialized_dataset import load_sample,save_sample
from .rachel_paired_density_dataset import file_sha256,validate_density_sample
from .rachel_training_dataset import _readonly

OWNERSHIP_PROTOCOL='density-local-unknown-ownership/2'
TRAIN_PIPELINE='paired-source-cell-density-materialization/2'
CLEAN_SCHEMA='rachel-clean-source-density-eval/2'
MAX_UNKNOWN_PIXELS=6
MAX_OWNER_ASSIGNMENTS=64
MAX_PROJECTION_RADIUS=13.  #6 base+4px fixedE1+3 ownership allowance
EXCLUSION_MARGIN=2.
MIN_INTERFACE_DISTANCE=2*(MAX_PROJECTION_RADIUS+EXCLUSION_MARGIN)


def _interface(masks,ab):
    return shared_source_seams(source_boundary(masks[ab[0]]),source_boundary(masks[ab[1]]))


def local_unknown_ownership(parents,ab):
    """Return the unchanged old normalization or a proved local-unknown view."""
    try:
        masks,metadata=normalize_boundary_ownership(parents)
        return masks,metadata,None
    except UnionGeometryRejected as error:
        if error.reason!='ownership_second_depth_exceeds_3px':raise
    ids=sorted(parents);ab=tuple(ab)
    if len(ab)!=2 or ab[0]==ab[1] or any(k not in ids for k in ab):
        raise ValueError('two existing distinct source members required')
    stack=np.stack([np.asarray(parents[k],bool) for k in ids]);overlap=stack.sum(0)>1
    fraction=float(overlap.sum()/max(1,stack.any(0).sum()))
    if fraction>.02:raise UnionGeometryRejected('ownership_overlap_fraction_exceeds_2_percent')
    depths=np.stack([ndimage.distance_transform_edt(m) for m in stack])
    second=np.sort(depths,axis=0)[-2];unknown=overlap&(second>3.)
    locations=np.argwhere(unknown)
    if not 1<=len(locations)<=MAX_UNKNOWN_PIXELS:
        raise UnionGeometryRejected('density_unknown_owner_region_exceeds_bounded_protocol')
    if np.any(unknown&stack[ids.index(ab[0])]&stack[ids.index(ab[1])]):
        raise UnionGeometryRejected('density_unknown_owner_touches_actual_pair')
    choices=[np.flatnonzero(stack[:,r,c]).tolist() for r,c in locations]
    count=int(np.prod([len(x) for x in choices]))
    if count>MAX_OWNER_ASSIGNMENTS:
        raise UnionGeometryRejected('density_unknown_owner_proof_budget_exceeded')
    known=stack.copy();safe=overlap&~unknown
    known[:,safe]=np.arange(len(ids))[:,None]==depths[:,safe].argmax(0)[None,:]
    known[:,unknown]=False
    if (not np.array_equal(known.any(0),stack.any(0)&~unknown)
            or not np.array_equal(known[:,~overlap],stack[:,~overlap])):
        raise ValueError('known source ownership changed exclusive material')
    masks={k:known[i] for i,k in enumerate(ids)}
    seam,_,topology=_interface(masks,ab)
    if not seam:raise UnionGeometryRejected('density_unknown_owner_no_unaffected_interface')
    midpoint=np.asarray([(np.asarray(k[0])+np.asarray(k[1]))*.5 for k in seam])
    distance=cKDTree(midpoint).query(locations+.5)[0]
    if float(distance.min())<=MIN_INTERFACE_DISTANCE:
        raise UnionGeometryRejected('density_unknown_owner_too_close_to_actual_interface')
    # Actual existing claimant assignments only. No GT pose or free new owner.
    for owners in itertools.product(*choices):
        variant=known.copy()
        for (r,c),owner in zip(locations,owners):variant[owner,r,c]=True
        current,_,_=_interface({k:variant[i] for i,k in enumerate(ids)},ab)
        if set(current)!=set(seam):
            raise UnionGeometryRejected('density_unknown_owner_changes_actual_interface')
    proof=dict(schema_version=OWNERSHIP_PROTOCOL,actual_pair_members=list(ab),
        unknown_coordinates_parent_rc=locations.tolist(),unknown_pixels=len(locations),
        possible_existing_owner_assignments=count,all_assignments_preserve_interface=True,
        unknown_to_interface_min_distance_px=float(distance.min()),
        required_interface_distance_strictly_greater_px=MIN_INTERFACE_DISTANCE,
        accepted_interface_edges=topology['accepted_edges'],
        maximum_group_overlap_fraction=.02,maximum_second_owner_depth_px=3.,
        observed_overlap_fraction=fraction,observed_second_depth_max_px=float(second[unknown].max()),
        max_unknown_pixels=MAX_UNKNOWN_PIXELS,max_owner_assignments=MAX_OWNER_ASSIGNMENTS,
        maximum_projection_radius_px=MAX_PROJECTION_RADIUS,exclusion_margin_px=EXCLUSION_MARGIN,
        unknown_owner_assigned=False,physical_mask_modified=False,GT_used=False)
    metadata=dict(method=OWNERSHIP_PROTOCOL,applied=True,exclusive_pixels_unchanged=True,
        reference_unknown_support_pixels=int(unknown.sum()),total_physical_support_unchanged=True,
        source_reference_has_explicit_unknowns=True)
    return masks,metadata,proof


def resample_source_density_v2(sample,report,source,offsets,clean,*,cap,ownership_allowance_px=0.,origin_receipts=None):
    result,derived,views=resample_source_density(sample,report,source,offsets,clean,cap=cap,
        ownership_allowance_px=ownership_allowance_px,origin_receipts=origin_receipts)
    proof=report.get('source_ownership_v2')
    if proof is None:return result,derived,views  #exact original normal/negative output
    if not sample.label or proof.get('schema_version')!=OWNERSHIP_PROTOCOL:
        raise ValueError('unknown ownership permitted only for verified positive source ancestry')
    coordinates=np.asarray(proof['unknown_coordinates_parent_rc'],float)+.5
    tree=cKDTree(coordinates);suppression={}
    for s in 'ab':
        radius=views[s]['projection_radius_px']
        if radius>MAX_PROJECTION_RADIUS:raise ValueError('unknown ownership protocol does not cover this damage depth')
        parent_points=getattr(result,'points_rc_'+s)+.5-np.asarray(offsets[s])
        unsafe=tree.query(parent_points)[0]<=radius+EXCLUSION_MARGIN
        suppression[s]=dict(tokens=int(unsafe.sum()),
            prior_positive_tokens=int(np.count_nonzero((getattr(result,'target_'+s)>=0)&unsafe)),
            radius_px=radius+EXCLUSION_MARGIN)
        if suppression[s]['prior_positive_tokens']:
            raise UnionGeometryRejected('density_unknown_owner_reaches_positive_descendant')
        views[s]['component'][unsafe]=-1;views[s]['seam_s'][unsafe]=np.nan
        views[s]['trusted'][unsafe]=False;views[s]['nonseam'][unsafe]=False
    ta,tb,chosen=inherit_dense_source_targets(views['a'],views['b'],derived['density']['components'])
    if not chosen:raise ValueError('unknown ownership cannot convert a positive to all-ignore')
    result=replace(result,target_a=_readonly(ta,np.int64),target_b=_readonly(tb,np.int64))
    derived['density'].update(new_match_count=len(chosen),chosen_source_cells=[list(x) for x in chosen],
        source_ownership_v2=proof,uncertain_token_exclusion=suppression)
    for s in 'ab':
        derived['density']['sides'][s].update(trusted=int(views[s]['trusted'].sum()),
            ignored=int(np.count_nonzero(getattr(result,'target_'+s)==-2)),
            source_seam_tokens=int(np.count_nonzero(views[s]['component']>=0)))
    derived.update(effective_supervised_match_count=len(chosen),inherited_match_count=len(chosen),
        ignored_token_count=int(np.count_nonzero(ta==-2)+np.count_nonzero(tb==-2)))
    return result,derived,views


def _recover_native(resolver,row,sample,report,root):
    # Caller reaches here only after the original resolver verified lineage,
    # A/B masks, original labels and offsets and then rejected group ownership.
    parts=row['fragment_a']['fragment_token'].split('/')
    group,path=resolver._group(parts[1],parts[2])
    fragments={s:next(f for f in group['fragments'] if f['fragment_token']==row['fragment_'+s]['fragment_token']) for s in 'ab'}
    parents=resolver._parents(group);ab=[fragments[s]['fragment_id'] for s in 'ab']
    masks,metadata,proof=local_unknown_ownership(parents,ab)
    if proof is None:raise ValueError('recovery expected explicit locally unknown source')
    offsets={s:fragments[s]['target_audit']['parent_to_model_offset_rc'] for s in 'ab'}
    clean={s:_mask(Path(root)/row['fragment_'+s]['model_mask_path']) for s in 'ab'}
    for s in 'ab':resolver._verify_centerpad(parents[fragments[s]['fragment_id']],clean[s],offsets[s])
    return dict(report,source_ownership_v2=proof),{s:masks[fragments[s]['fragment_id']] for s in 'ab'},offsets,clean,metadata,str(path)


class SourceDensityRecordResolverV2(SourceDensityRecordResolver):
    def resolve(self,index):
        try:return super().resolve(index)
        except UnionGeometryRejected as error:
            entry=self.entries[index]
            if (error.reason!='ownership_second_depth_exceeds_3px'
                    or entry['source_stratum']!='native_positive' or not entry['label']):raise
        entry=self.entries[index];row=entry['source_row']
        sample,report=load_sample(self.artifact_root/entry['artifact_path'])
        report,source,offsets,clean,metadata,path=_recover_native(self,row,sample,report,entry['source_root'])
        provenance=dict(index=index,pair_id=sample.pair_id,label=True,source_stratum=entry['source_stratum'],
            fixed_artifact=str(self.artifact_root/entry['artifact_path']),source_root=entry['source_root'],
            groups=[path,path],ownership_reference_only=True,ownership_normalization=metadata,
            source_ownership_v2=report['source_ownership_v2'],parent_to_model_offsets=offsets)
        return sample,report,source,offsets,clean,3.,provenance

    def sample_at(self,index,cap):
        sample,report,source,offsets,clean,allowance,provenance=self.resolve(index)
        result,derived,views=resample_source_density_v2(sample,report,source,offsets,clean,cap=cap,ownership_allowance_px=allowance)
        derived['source_resolution']=provenance
        return result,derived,views


class CleanSourceDensityV2Dataset(CleanSourceDensityDataset):
    def __init__(self,root,split,contour_cap=512,cache_dir=None):
        super().__init__(root,split,contour_cap,cache_dir=None)
        protocol={k:v for k,v in self.protocol.items() if k!='identity_sha256'}
        protocol.update(schema_version=CLEAN_SCHEMA,source_density_version='v2',ownership_protocol=OWNERSHIP_PROTOCOL,
            density_v2_module_sha256=file_sha256(__file__))
        self.identity=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.protocol=dict(protocol,identity_sha256=self.identity)
        self.cache_dir=Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir:
            if self.cache_dir==self.root or self.root in self.cache_dir.parents:
                raise ValueError('evaluation cache must not modify original release')
            self.cache_dir.mkdir(parents=True,exist_ok=True);path=self.cache_dir/'cache_identity.json'
            if path.exists():
                if json.loads(path.read_text())!=self.protocol:raise ValueError('v2 cache split/cap/source identity differs')
            else:
                temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(self.protocol,indent=2)+'\n')
                os.replace(temporary,path)

    def _resolve_clean(self,index):
        try:return super()._resolve_clean(index)
        except UnionGeometryRejected as error:
            row=self.rows[index]
            if error.reason!='ownership_second_depth_exceeds_3px' or not row['label']:raise
        sample=self.base[index];count=int(np.count_nonzero(sample.target_a>=0))
        report=dict(schema_version=CLEAN_REPORT_SCHEMA,pair_id=sample.pair_id,source_split=self.split,
            changed_pair=False,changed_a=False,changed_b=False,weathering_applied=False,
            side_a={'effective_applied':False},side_b={'effective_applied':False},pose_supervision_enabled=True,
            original_gt_translation_preserved=True,effective_supervised_match_count=count,inherited_match_count=count,
            ignored_token_count=int(np.count_nonzero(sample.target_a==-2)+np.count_nonzero(sample.target_b==-2)),
            inheritance_rule='original clean512 targets used only as an audit count; never reindexed')
        report,source,offsets,clean,metadata,path=_recover_native(self,row,sample,report,self.root)
        report['source_resolution']=dict(groups=[path,path],parent_to_model_offsets=offsets,
            ownership_reference_only=True,ownership_normalization=metadata,label_origin=row['label_origin'],
            negative_origin=row.get('negative_origin'),source_ownership_v2=report['source_ownership_v2'])
        return sample,report,source,offsets,clean,3.

    def weathered(self,index):
        row=self.rows[index]
        path=self.cache_dir/(hashlib.sha256(row['pair_id'].encode()).hexdigest()+'.npz') if self.cache_dir else None
        if path and path.exists():
            result,report=load_sample(path)
            if result.pair_id!=row['pair_id'] or bool(result.label)!=bool(row['label']):raise ValueError('cached evaluation labels changed')
            validate_density_sample(result,report,self.contour_cap,self.identity)
            return result,report
        original,report,source,offsets,clean,allowance=self._resolve_clean(index)
        result,derived,_=resample_source_density_v2(original,report,source,offsets,clean,cap=self.contour_cap,ownership_allowance_px=allowance)
        derived['paired_density']=dict(identity_sha256=self.identity,source_pair_id=result.pair_id,
            source_manifest=str(self.manifest_path),source_split=self.split)
        for field in ('mask_a','mask_b','coarse_mask_a','coarse_mask_b','label','translation_a_to_b_rc',
                'translation_a_to_b_xy_cartesian','translation_valid'):
            if not np.array_equal(getattr(original,field),getattr(result,field),equal_nan=True):
                raise ValueError('clean original field changed: '+field)
        validate_density_sample(result,derived,self.contour_cap,self.identity)
        if path:save_sample(path,result,derived)
        return result,derived
