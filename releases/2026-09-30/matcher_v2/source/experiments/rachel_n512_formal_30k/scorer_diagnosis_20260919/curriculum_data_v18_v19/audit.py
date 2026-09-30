"""Independent pixel checks and export of committed paired review artifacts.

No generation, label repair, threshold tuning or training occurs here.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2.latent_seam import source_band,ray_project,measure
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2.partial_v14 import retained_support
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2.conservative_weather import parts
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2 import materialize
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2.scale import pair_shared_scale
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2.layered_geometry import canonical_mirrored_contours
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.seam_context_v3.augmentation import paired_mirror
from ..aggressive_data_v16.geometry import replay,selected_side,visible_curve_nonlinearity
from ..aggressive_data_full_v17.source import STATE, baseline
from ..aggressive_data_v16.endpoints import audit_endpoints
from .weather import smooth_profile
from ..s7_compound_v1.geometry import _signed_arc_distance
from .pristine import audit_exact
from .crop_floor import evaluate as crop_evaluate
from .seam_contract import evaluate as connectable_evaluate, require as require_connectable


EIGHT=np.ones((3,3),int)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(z,name):return np.unpackbits(z[name],axis=1).astype(bool)
def need(value,message):
    if not value:raise AssertionError(message)


def field_apply(mask,points,field):
    filled=ndimage.binary_fill_holes(np.pad(mask,1),structure=EIGHT)
    depth=ndimage.distance_transform_edt(filled)[1:-1,1:-1]-.5
    pixels=np.argwhere(mask&(depth<=float(field.max(initial=0))))
    result=mask.copy()
    if len(pixels):
        index=cKDTree(points).query(pixels)[1]
        removed=depth[tuple(pixels.T)]<=field[index]
        result[tuple(pixels[removed].T)]=False
    return result


def sample_stages(sample,z):
    return {stage:replace(sample,**{'mask_'+s:unpack(z,'packed_'+stage+'_'+s)[None].astype(np.float32) for s in 'ab'})
            for stage in ('fragment','trim','primary','final')}


def audit_depth(record,side,z,base,trim,primary):
    info=record['detail']['primary_damage'][side];plan=info['field_plan']
    spec=record['detail']['spec']
    major,weak=parts(record['recipe']);prefix='primary_'+side+'_'
    arc=z[prefix+'arc_exact'];edge=z[prefix+'edge_exact'];eligible=z[prefix+'eligible']
    expected_major=np.zeros(len(arc));expected_weak=expected_major.copy()
    if weak:need(spec['weak'][0]<=plan['weak_peak']<spec['weak'][1],'requested weak peak matches version')
    else:need(plan['weak_peak']==0,'no undeclared weak erosion')
    if major=='gaps':
        k=record['requested_gap_count'];need(k in spec['notch_counts'] and len(plan['regions'])==k,'requested K separate regions')
        need(len(info['ignore_source_regions'])==k,'all requested notches retained')
    for i,region in enumerate(plan['regions']):
        u=_signed_arc_distance(arc,region['center_arc_px'],float(edge.sum()))/(region['support_length_px']/2)
        if major:
            peak=region['requested_peak_depth_px'];bounds=spec['notch'] if major=='gaps' else spec['major'];need(bounds[0]<=peak<bounds[1],'each major requested peak matches version')
            mode={'wave':'wave','local_abrupt':'abrupt','local_gradual':'gradual','gaps':'gap'}[major]
            expected_major=np.maximum(expected_major,smooth_profile(u,peak,mode,plan['phase'])*eligible)
        if weak and i==plan['weak_region']:
            expected_weak=np.maximum(expected_weak,smooth_profile(u,plan['weak_peak'],plan['weak_mode'],plan['phase'])*eligible)
    need(np.allclose(expected_major,z[prefix+'major'],atol=1e-6,rtol=1e-6),'approved continuous major shape')
    need(np.allclose(expected_weak,z[prefix+'weak'],atol=1e-6,rtol=1e-6),'unchanged v14 smooth weak shape')
    need(np.array_equal(np.minimum(spec["primary_cap"],expected_major+expected_weak),z[prefix+'total_exact']),'single-side version cap')
    need(float(z[prefix+'total_exact'].max())<=spec['primary_cap']+1e-9,'actual single-side version cap')
    removed=np.argwhere(trim&~primary);p=z[prefix+'points'];nearest=cKDTree(p).query(removed)[1]
    fill=ndimage.binary_fill_holes(np.pad(base,1),structure=EIGHT)
    depth=ndimage.distance_transform_edt(fill)[1:-1,1:-1]-.5
    for region in info['ignore_source_regions']:
        indices=cKDTree(p).query(np.asarray(region['source_points_rc']))[1];member=np.isin(nearest,indices)
        need(member.sum()>=8 and int(member.sum())==region['independently_removed_pixels'],'each region really removes >=8 pixels')
        actual=float(depth[tuple(removed[member].T)].max(initial=0.))
        need(abs(actual-region['applied_max_depth_px'])<1e-6 and actual>=region['requested_peak_depth_px']-1.5,'effective region depth after trim')
    if weak:
        base_only=trim&field_apply(base,p,z[prefix+'major']);extra=int(np.count_nonzero(base_only&~primary))
        need(extra>=4 and extra==info['weak_independently_removed_pixels'],'weak not just a label')


def audit_record(record,baseline_root):
    if record.get('v14_fallback'):
        raise AssertionError('review hard examples cannot silently fall back')
    spec=record['detail']['spec']
    path=Path(record['sample_path']);proof=Path(record['proof_path'])
    need(sha(path)==record['sample_sha256'],'sample sha');need(sha(proof)==record['proof_sha256'],'proof sha')
    sample,report=load_sample(path);old,old_report=load_sample(record['baseline_sample_path'])
    need(sample.label==old.label==record['label'],'label unchanged')
    need(np.array_equal(sample.translation_a_to_b_rc,old.translation_a_to_b_rc),'GT unchanged')
    slot=record['baseline_slot'];pair_index=record['baseline_ordinal']
    entry=read(baseline_root/'groups'/f'{slot:05d}.json')['entries'][pair_index]
    with np.load(baseline_root/entry['weather_artifact'],allow_pickle=False) as data:
        weather={k:data[k] for k in data.files}
    with np.load(proof,allow_pickle=False) as data:z={k:data[k] for k in data.files}
    stages=sample_stages(sample,z);coverages=[]
    for side in 'ab':
        masks=[getattr(stages[x],'mask_'+side)[0].astype(bool) for x in stages]
        base,trim,primary,final=masks
        need(np.array_equal(base,np.unpackbits(weather['packed_preweather_'+side],axis=1).astype(bool)),'fragment stage matches v14')
        need(np.array_equal(final,getattr(sample,'mask_'+side)[0]>0),'final proof matches actual archive')
        for a,b in zip(masks,masks[1:]):need(not np.any(b&~a),'no added material')
        for m in masks[1:]:
            need(ndimage.label(m,EIGHT)[1]==1,'single connected fragment')
            need(not np.any(ndimage.binary_fill_holes(m,structure=EIGHT)&~m&base),'no new enclosed hole')
        need(final.sum()>=max(64,.25*base.sum()),'aggregate area retained')
        plan=record['detail']['trim'];expected=base.copy()
        if side==plan['side'] and plan.get('applied',True):
            need(selected_side(stages['fragment'],plan['size_class'])==side,'side selected by actual original area')
            need(0<1-trim.sum()/base.sum()<=.20+1e-12,'additional cut loses at most20% area on BOTH size classes')
            need(abs(trim.sum()/base.sum()-plan['material_retained_fraction'])<1e-12,'actual cut area matches metadata')
            bank=materialize.STATE['bank'];donor=plan['donor_index']
            need(np.array_equal(bank.profiles[donor],np.array(plan['profile'],np.float32)),'actual TRAIN donor curve binding')
            need(bank.metadata['arcs'][donor]==plan['donor'] and plan['donor']['split']==STATE['split'],'same-fold donor provenance')
            need(hashlib.sha256(bank.profiles[donor].tobytes()).hexdigest()==plan['profile_sha256'],'donor hash')
            expected=replay(base,plan)
            visible_curve_nonlinearity(base,trim,plan)
        need(np.array_equal(trim,expected),'actual structural cut equals declared plan')
        if not plan.get('applied',True):
            need(plan['material_removed_fraction']==0 and np.array_equal(base,trim),'skipped cut preserves all pixels')
        if record.get('previous_proof_path'):
            need(sha(record['previous_proof_path'])==record['previous_proof_sha256'],'previous approved trim source hash')
            with np.load(record['previous_proof_path'],allow_pickle=False) as old_z:
                need(np.array_equal(trim,unpack(old_z,'packed_trim_'+side)),'approved cut pixels completely unchanged')
        if 'primary_'+side+'_points' in z:
            audit_depth(record,side,z,base,trim,primary)
            expected=trim&field_apply(base,z['primary_'+side+'_points'],z['primary_'+side+'_total_exact'])
            need(np.array_equal(primary,expected),'primary field raster replay')
            p=z['primary_'+side+'_points'];eligible=z['primary_'+side+'_eligible']&trim[tuple(np.rint(p).astype(int).T)]
            edge=z['primary_'+side+'_edge'];intact=eligible&np.roll(eligible,-1)
            active=(z['primary_'+side+'_total_exact']>0)&eligible;denom=float(edge[intact].sum())
            fraction=float(edge[intact&active&np.roll(active,-1)].sum()/denom) if denom else 0.
            need(spec['main_coverage'][0]-1e-6<=fraction<=spec['main_coverage'][1]+1e-6,'major footprint version bounds eligible seam')
        elif record['recipe']=='clean':need(np.array_equal(trim,primary),'clean no primary erosion')
        if 'light_'+side+'_points' in z:
            need(np.array_equal(primary,unpack(z,'light_packed_before_'+side)),'light input actual primary')
            expected=field_apply(primary,z['light_'+side+'_points'],z['light_'+side+'_total_exact'])
            need(np.array_equal(final,expected),'background field raster replay')
            p=np.rint(z['light_'+side+'_points']).astype(int);eligible=z['light_'+side+'_eligible']
            if spec.get('light_scope')=='whole_postprimary_contour':
                need(eligible.all(),'final light includes entire current contour, including prior corrosion/cut edges')
                need(record['detail']['background'][side]['whole_postprimary_contour'],'whole-contour light declared')
            guard=z['light_'+side+'_protected_points']
            if not spec.get('pristine_protection_enabled',False):
                need(len(guard)==0 and spec['pristine_min_fraction']==0.,'no hidden contact protection or pristine quota')
            if len(guard):
                near=cKDTree(guard).query(z['light_'+side+'_points'])[0]<=spec['light_protection_guard_px']
                need(not np.any(eligible&near),'protected contour excluded from light eligibility')
                need(not np.any(z['light_'+side+'_total_exact'][near]),'no light field on protected contour')
            w=z['light_'+side+'_edge'];intact=eligible&np.roll(eligible,-1)
            changed=primary[tuple(p.T)]&~final[tuple(p.T)]
            coverage=float(w[intact&changed&np.roll(changed,-1)].sum()/w[intact].sum())
            need(.68-1e-6<=coverage<=.72+1e-6,'70percent actual untouched contour coverage')
            need(abs(coverage-record['detail']['background'][side]['actual_affected_fraction'])<1e-5,'coverage metadata')
            need(z['light_'+side+'_total_exact'].max()<=spec['light'][1]+1e-6,'light field version depth cap')
            need(all(spec['light'][0]<=v<=spec['light'][1] for v in record['detail']['background'][side]['requested_peak_depths_px']),'light requested peaks')
            depth=ndimage.distance_transform_edt(ndimage.binary_fill_holes(np.pad(primary,1),structure=EIGHT))[1:-1,1:-1]-.5
            removed=primary&~final;actual_peak=float(depth[removed].max(initial=0.))
            need(actual_peak<=spec['light'][1]+1e-6,'actual light raster depth within approved ceiling')
            need(abs(actual_peak-record['detail']['background'][side]['applied_max_depth_px'])<1e-6,'actual light depth metadata')
            coverages.append(coverage)
        else:need(np.array_equal(primary,final),'no undeclared background erosion')
    need(int((sample.target_a>=0).sum())==record['inherited_correspondences'],'target count')
    if sample.label:
        need(record['inherited_correspondences']>=4,'at least4 inherited matches')
        plan=record['detail']['trim']
        actual_removed=1-plan['retained_fraction'];target_removed=plan['target_removed_fraction']
        need(.25<=target_removed<=.40,'sampled crop target stays25–40%')
        if plan.get('applied',True):
            need(.25<=actual_removed<=.40,'actual applied shortening stays25–40%')
            need(abs(actual_removed-target_removed)<=.01+1e-12,'actual shortening within1pp of sampled target')
        else:need(actual_removed==0 and bool(plan['skip_reason']),'no hidden shortening in skipped case')
        reconstructed=baseline(slot,0)['original']
        for s in 'ab':need(np.array_equal(getattr(reconstructed,'mask_'+s),getattr(stages['fragment'],'mask_'+s)),'source reconstruction')
        reference=reconstructed
        bands=source_band(reconstructed,bridge=0.)
        pristine_audit=audit_exact(reconstructed,sample,bands,record['detail']['pristine_seam'],
            {k.removeprefix('pristine_'):v for k,v in z.items() if k.startswith('pristine_')},
            spec['pristine_min_fraction'])
        before,before_proof=retained_support(reconstructed,reference,bands)
        cropped_reference=replace(reference,**{'mask_'+s:(getattr(reference,'mask_'+s)*getattr(stages['trim'],'mask_'+s)) for s in 'ab'})
        after,after_proof=retained_support(reconstructed,cropped_reference,bands)
        if plan.get('applied',True):
            end_check=audit_endpoints(bands,before_proof,after_proof,plan['mode'])
            need(end_check==plan['endpoint_audit'],'original bilateral arc endpoint-only audit')
        for key,actual in [('common_length_before_px',before['common_retained_length_px']),
                           ('common_length_after_px',after['common_retained_length_px'])]:
            need(abs(actual-record['detail']['trim'][key])<1e-4,'measured original support length')
        trim_floor=crop_evaluate(reconstructed,stages['trim'],bands)
        need(trim_floor==plan['crop_floor'],'crop floor independently recomputed from archived stage pixels')
        crop_stage=stages['primary'] if record['recipe']=='partial' else stages['trim']
        crop_floor=crop_evaluate(reconstructed,crop_stage,bands)
        need(crop_floor==record['detail']['crop_only_seam_floor'],'20% measured BEFORE erosion and light, including structural Partial')
        connectable,connectable_arrays=connectable_evaluate(reconstructed,stages['primary'],sample,bands)
        require_connectable(connectable)
        need(connectable==record['detail']['connectable_seam'],'original20/final30 recomputed from stage pixels')
        for key,value in connectable_arrays.items():
            need(np.array_equal(value,z['connectable_'+key]),'connectable original edges match archived proof')
        _,original_proof=retained_support(reconstructed,stages['primary'] if record['recipe']=='partial' else stages['trim'],bands)
        active_gaps=[];all_gaps=[];all_weights=[];unresolved=0.;denominator=0.
        for side,other,shift in [('a','b',sample.translation_a_to_b_rc),('b','a',-sample.translation_a_to_b_rc)]:
            p=z['gap_'+side+'_source_points'];q=z['gap_'+side+'_partner_source_points']
            pa,_,va=ray_project(getattr(stages['fragment'],'mask_'+side)[0],getattr(sample,'mask_'+side)[0],p)
            pb,_,vb=ray_project(getattr(stages['fragment'],'mask_'+other)[0],getattr(sample,'mask_'+other)[0],q)
            keep=original_proof[side+'_physically_retained']
            need(np.allclose(p,original_proof[side+'_source_points'][keep],atol=1e-4),'frozen original source points')
            need(np.allclose(q,original_proof[side+'_partner_points'][keep],atol=1e-4),'frozen partners, no rematching after cut')
            source_valid=np.linalg.norm(p+shift-q,axis=1)<=3.+1e-6
            valid=source_valid&va&vb;dist=np.linalg.norm(pa+shift-pb,axis=1)
            need(np.array_equal(valid,z['gap_'+side+'_valid']),'normal projection resolved mask')
            need(np.allclose(dist[valid],z['gap_'+side+'_gap'][valid],atol=1e-4),'actual two-sided gap')
            affected=z['gap_'+side+'_primary_affected'];w=z['gap_'+side+'_source_weight']
            expected_active=np.zeros(len(p),bool)
            for source in 'ab':
                prefix='primary_'+source+'_'
                if prefix+'points' not in z:continue
                shift2=(sample.translation_a_to_b_rc if side=='a' and source=='b' else -sample.translation_a_to_b_rc if side=='b' and source=='a' else np.zeros(2))
                distance,index=cKDTree(z[prefix+'points']).query(p+shift2)
                expected_active|=(distance<=3.5)&(z[prefix+'total_exact'][index]>0)
            need(np.array_equal(affected,expected_active),'independent primary footprint membership')
            if record['recipe'] not in ('clean','partial'):
                need(spec['main_coverage'][0]<=float(w[affected].sum()/w.sum())<=spec['main_coverage'][1],
                    'version coverage of remaining COMMON source seam')
            active_gaps.extend(dist[valid&affected]);all_gaps.extend(dist[valid]);all_weights.extend(w[valid])
            denominator+=w.sum();unresolved+=w[~valid].sum()
        major,_=parts(record['recipe'])
        has_primary=record['recipe'] not in ('clean','partial')
        if has_primary:need(5<=max(active_gaps)<=spec['gap_cap']+1e-4,'final PRIMARY footprint version ceiling including weak')
        if record['recipe']=='partial' and not record['detail']['partial_crop_applied']:
            for s in 'ab':need(np.array_equal(getattr(stages['trim'],'mask_'+s),getattr(stages['primary'],'mask_'+s)),
                'skipped legacy Partial keeps crop-stage masks unchanged')
        need(abs(unresolved/denominator-(1-record['detail']['gap']['ray_resolved_fraction']))<1e-5,'unresolved accounted separately')
        # Measure v14 over exactly the same retained source locations. This is
        # a paired gap comparison, not a claim that L40 equals ancestral seam.
        baseline_gaps=[]
        for side,other,shift in [('a','b',sample.translation_a_to_b_rc),('b','a',-sample.translation_a_to_b_rc)]:
            p=z['gap_'+side+'_source_points'];q=z['gap_'+side+'_partner_source_points']
            pa,_,va=ray_project(getattr(stages['fragment'],'mask_'+side)[0],getattr(old,'mask_'+side)[0],p)
            pb,_,vb=ray_project(getattr(stages['fragment'],'mask_'+other)[0],getattr(old,'mask_'+other)[0],q)
            affected=z['gap_'+side+'_primary_affected'];valid=va&vb&affected&(np.linalg.norm(p+shift-q,axis=1)<=3.+1e-6)
            baseline_gaps.extend(np.linalg.norm(pa+shift-pb,axis=1)[valid])
        paired=dict(new_primary_gap_peak_px=max(active_gaps) if has_primary and active_gaps else None,
            old_primary_gap_peak_px=max(baseline_gaps) if has_primary and baseline_gaps else None,
            resolved_arc_gap_p10_p50_p90=np.quantile(all_gaps,[.1,.5,.9]).tolist(),
            unresolved_fraction=float(unresolved/denominator),
            arc_fraction_gap5to_cap=float(np.sum(np.array(all_weights)*((np.array(all_gaps)>=5)&(np.array(all_gaps)<=spec['gap_cap'])))/np.sum(all_weights)))
    else:
        need(record['detail']['gap'] is None and record['detail']['trim']['gt_seam_used'] is False,'no fabricated negative gap/GT')
        need(record['detail']['pristine_seam'] is None,'no fabricated negative pristine seam')
        need(record['detail']['connectable_seam'] is None,'no fabricated negative connectable seam')
        control=record['detail']['negative_preservation_control']
        if spec.get('pristine_protection_enabled',False):
            need(all(not v['gt_seam_used'] for v in control.values()),'negative protection independent of nonexistent common GT')
        else:need(control is None,'no hidden negative preservation control')
        pristine_audit=None;crop_floor=None;connectable=None
        paired=None
    from ..aggressive_data_full_v17.supervision import audit_supervision
    supervision=audit_supervision(record,baseline(slot,pair_index)['original'],sample,report,z)
    h=hashlib.sha256()
    for name in ('mask_a','mask_b','points_rc_a','points_rc_b','target_a','target_b','translation_a_to_b_rc'):
        value=getattr(sample,name);h.update(name.encode());h.update(value.tobytes())
    return dict(id=record['id'],status='passed',light_coverages=coverages,paired_gap=paired,pristine_seam=pristine_audit,
                crop_only_seam_floor=crop_floor,
                connectable_seam=connectable,
                supervision=supervision,sample_sha256=record['sample_sha256'],proof_sha256=record['proof_sha256'],
                model_input_sha256=h.hexdigest())
