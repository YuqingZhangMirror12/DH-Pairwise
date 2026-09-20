"""Version2: materialize paired source density with bounded unknown isolation.

No training/GPU. Default selection is the complete input manifest; --pair-ids
is an explicit small smoke/subset selection, stored distinctly in identity.
Errors are recorded, neither cap silently drops pairs, and incomplete runs
cannot be loaded as complete datasets. Existing live data are never modified.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import save_sample,load_sample
from staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_v2 import resample_source_density_v2 as resample_source_density
from staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_v2 import SourceDensityRecordResolverV2 as SourceDensityRecordResolver
from staging.pairwise_v0_2.pairwise_data.rachel_paired_density_dataset import (
    SCHEMA,PAIR_SCHEMA,file_sha256,validate_density_sample)

PIPELINE='paired-source-cell-density-materialization/2'


def canonical_digest(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save_json(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(record,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def pipeline_identity():
    from staging.pairwise_v0_2.pairwise_data import (
        rachel_source_density,rachel_source_density_records,rachel_preprocess,rachel_union_augmentation,
        rachel_edge_weathering,rachel_paired_density_dataset,rachel_density_ownership_v2)
    modules=(rachel_source_density,rachel_source_density_records,rachel_preprocess,rachel_union_augmentation,
             rachel_edge_weathering,rachel_paired_density_dataset,rachel_density_ownership_v2)
    return dict(version=PIPELINE,source_density_version='v2',ownership_protocol='density-local-unknown-ownership/2',code_sha256={m.__name__:file_sha256(m.__file__) for m in modules},
        materializer_sha256=file_sha256(__file__),python=platform.python_version(),
        numpy=np.__version__,scipy=scipy.__version__,caps=[512,1024],smoothing_sigma=3.,
        physical_input='original fixed E1 masks unchanged',
        positive_target='exact clean shared cell IDs via independent own-boundary ancestry',
        negative_target='all real points dustbin from original fixed nonpair identity',
        pair_GT_translation_used_for_targets=False,positive_zero_matches_allowed=False)


def selection_record(resolver,pair_ids,pipeline=None):
    entries=resolver.entries;by_id={e['pair_id']:i for i,e in enumerate(entries)}
    if len(by_id)!=len(entries):raise ValueError('duplicate source pair IDs')
    if pair_ids is None:
        selected=[e['pair_id'] for e in entries];mode='full_source_manifest'
    else:
        selected=list(pair_ids);mode='explicit_subset'
        if not selected or len(set(selected))!=len(selected):raise ValueError('empty / duplicate explicit selection')
        if any(pid not in by_id for pid in selected):raise ValueError('unknown selected source pair ID')
    identity=dict(schema_version=PIPELINE,source_manifest=str(resolver.manifest_path),
        source_manifest_sha256=file_sha256(resolver.manifest_path),canonical_root=str(resolver.canonical_root),
        selection_mode=mode,selected_pair_ids=selected,
        source_indices=[by_id[pid] for pid in selected],pipeline=pipeline or pipeline_identity())
    return dict(identity,identity_sha256=canonical_digest(identity))


def validate_preserved(source,result,source_report,derived):
    for name in ('mask_a','mask_b','coarse_mask_a','coarse_mask_b','translation_a_to_b_rc',
                 'translation_a_to_b_xy_cartesian','translation_valid','label'):
        if not np.array_equal(getattr(source,name),getattr(result,name),equal_nan=True):
            raise ValueError('source field changed: '+name)
    if source_report['pose_supervision_enabled']!=derived['pose_supervision_enabled']:
        raise ValueError('pose-supervision eligibility changed')


def pair_folder(pair_id):return hashlib.sha256(pair_id.encode()).hexdigest()


def make_pair(resolver,index,output,identity):
    entry=resolver.entries[index];pid=entry['pair_id']
    directory=output/'pairs'/pair_folder(pid);directory.mkdir(parents=True,exist_ok=True)
    fixed=resolver.artifact_root/entry['artifact_path'];source_sha=file_sha256(fixed)
    marker=directory/'complete.json'
    if marker.exists():
        complete=json.loads(marker.read_text())
        if (complete.get('schema_version')!=PAIR_SCHEMA or complete.get('status')!='complete'
                or complete.get('identity_sha256')!=identity or complete.get('pair_id')!=pid
                or complete.get('source_archive_sha256')!=source_sha):
            raise ValueError('resume pair identity/source differs')
        for cap in (512,1024):
            receipt=complete['caps'][str(cap)];path=output/receipt['artifact_path']
            if file_sha256(path)!=receipt['sha256']:raise ValueError('completed paired artifact changed')
            sample,report=load_sample(path);validate_density_sample(sample,report,cap,identity)
        return complete,True
    sample,report,source,offsets,clean,allowance,provenance=resolver.resolve(index)
    complete=dict(schema_version=PAIR_SCHEMA,status='writing',pair_id=pid,index=int(index),
        identity_sha256=identity,source_archive_sha256=source_sha,source_resolution=provenance,
        source_label=bool(sample.label),original_pose_supervision_enabled=report['pose_supervision_enabled'],caps={})
    cap_samples={}
    for cap in (512,1024):
        changed,derived,_=resample_source_density(sample,report,source,offsets,clean,
            cap=cap,ownership_allowance_px=allowance)
        derived['paired_density']=dict(identity_sha256=identity,source_archive_sha256=source_sha,
                                       source_pair_id=pid,source_manifest=str(resolver.manifest_path))
        derived['source_resolution']=provenance
        validate_preserved(sample,changed,report,derived)
        validate_density_sample(changed,derived,cap,identity)
        # This assertion stays in the materializer, outside model inputs.
        canary,_,_=resample_source_density(replace(sample,
            translation_a_to_b_rc=sample.translation_a_to_b_rc+1000.),report,source,offsets,clean,
            cap=cap,ownership_allowance_px=allowance)
        if not np.array_equal(changed.target_a,canary.target_a) or not np.array_equal(changed.target_b,canary.target_b):
            raise ValueError('GT translation leaked into dense correspondence')
        relative=Path('pairs')/directory.name/('n'+str(cap)+'.npz');path=output/relative
        save_sample(path,changed,derived)
        reread,rereport=load_sample(path);validate_density_sample(reread,rereport,cap,identity)
        validate_preserved(sample,reread,report,rereport)
        cap_samples[cap]=changed
        complete['caps'][str(cap)]=dict(artifact_path=str(relative),sha256=file_sha256(path),
            real_points={s:len(getattr(changed,'points_rc_'+s)) for s in 'ab'},
            positive_matches=derived['density']['new_match_count'],
            ignored={s:derived['density']['sides'][s]['ignored'] for s in 'ab'})
    complete['new_distinct_coordinates_at1024']={}
    for s in 'ab':
        small=getattr(cap_samples[512],'points_rc_'+s);large=getattr(cap_samples[1024],'points_rc_'+s)
        new=len(set(map(tuple,large))-set(map(tuple,small)))
        if len(large)>len(small) and not new:raise ValueError('larger cap only duplicates old coordinates')
        complete['new_distinct_coordinates_at1024'][s]=new
    complete.update(status='complete',gt_translation_canary_passed=True,
                    source_fields_preserved=True,completed_at_unix=time.time())
    save_json(marker,complete)
    failure=directory/'failure.json'
    if failure.exists():failure.unlink()
    return complete,False


def manifests(output,resolver,selection,completions,failures):
    complete=len(completions)==len(selection['selected_pair_ids']) and not failures
    for cap in (512,1024):
        rows=[]
        for index in selection['source_indices']:
            original=resolver.entries[index];pid=original['pair_id']
            if pid not in completions:continue
            receipt=completions[pid]['caps'][str(cap)]
            rows.append(dict(pair_id=pid,label=bool(original['label']),source_row=original['source_row'],
                source_root=original['source_root'],source_stratum=original['source_stratum'],
                source_index=index,artifact_path=receipt['artifact_path'],artifact_sha256=receipt['sha256']))
        record=dict(schema_version=SCHEMA,status='complete' if complete else 'incomplete',split='train',
            contour_cap=cap,artifact_root=str(output),identity_sha256=selection['identity_sha256'],
            selected_pair_count=len(selection['selected_pair_ids']),completed_pair_count=len(rows),
            failed_pair_count=len(failures),failures=failures,entries=rows,
            stats=dict(positive=sum(row['label'] for row in rows),negative=sum(not row['label'] for row in rows)),
            protocol=dict(selection_mode=selection['selection_mode'],source_manifest=selection['source_manifest'],
                source_manifest_sha256=selection['source_manifest_sha256'],pipeline=selection['pipeline'],
                companion_manifest='train_n'+str(1024 if cap==512 else 512)+'.json',
                paired512_control_required=True,live_dataset_modified=False))
        save_json(output/('train_n'+str(cap)+'.json'),record)
    return complete


def materialize(resolver,output,pair_ids=None,resume=False,stop_after_pairs=None,pipeline=None):
    """stop_after_pairs is interruption smoke support, never an implicit subset."""
    output=Path(output).resolve();selection=selection_record(resolver,pair_ids,pipeline)
    if output==resolver.artifact_root or resolver.artifact_root in output.parents:
        raise ValueError('derivative output must not be inside original fixed dataset')
    if resume:
        saved=json.loads((output/'source_selection.json').read_text())
        if saved!=selection:raise ValueError('resume selection / source / pipeline identity differs')
    else:
        output.mkdir(parents=True,exist_ok=False);save_json(output/'source_selection.json',selection)
    # Exclusive writer for this new output only; no lock is taken on live data.
    import fcntl
    with (output/'.writer.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        completions,failures={},[];created,resumed=0,0
        save_json(output/'run_state.json',dict(status='running',identity_sha256=selection['identity_sha256']))
        for count,index in enumerate(selection['source_indices']):
            if stop_after_pairs is not None and count>=stop_after_pairs:break
            pid=resolver.entries[index]['pair_id']
            try:
                complete,was_resumed=make_pair(resolver,index,output,selection['identity_sha256'])
                completions[pid]=complete;resumed+=int(was_resumed);created+=int(not was_resumed)
            except Exception as error:
                failure=dict(pair_id=pid,source_index=index,error_type=type(error).__name__,error=str(error))
                failures.append(failure);save_json(output/'pairs'/pair_folder(pid)/'failure.json',failure)
            save_json(output/'run_state.json',dict(status='running',identity_sha256=selection['identity_sha256'],
                selected_pairs=len(selection['selected_pair_ids']),visited_pairs=count+1,
                complete_pairs=len(completions),failed_pairs=len(failures),new_pairs=created,resumed_pairs=resumed))
        complete=manifests(output,resolver,selection,completions,failures)
        state=dict(schema_version=PIPELINE,status='complete' if complete else 'incomplete',
            identity_sha256=selection['identity_sha256'],selected_pairs=len(selection['selected_pair_ids']),
            complete_pairs=len(completions),failed_pairs=len(failures),new_pairs=created,resumed_pairs=resumed,
            full_source_manifest=selection['selection_mode']=='full_source_manifest',
            gpu_used=False,live_dataset_modified=False,failures=failures)
        save_json(output/'run_state.json',state)
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',required=True);parser.add_argument('--canonical-root',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--pair-ids',help='explicit JSON array of IDs; omit for full manifest')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-pairs',type=int,help='bounded interruption test; leaves incomplete selection')
    args=parser.parse_args()
    if args.stop_after_pairs is not None and args.stop_after_pairs<1:parser.error('--stop-after-pairs must be positive')
    ids=json.loads(Path(args.pair_ids).read_text()) if args.pair_ids else None
    resolver=SourceDensityRecordResolver(args.source_manifest,args.canonical_root)
    state=materialize(resolver,args.output,ids,args.resume,args.stop_after_pairs)
    print(json.dumps(state,indent=2),flush=True)
    return 0 if state['status']=='complete' else 2


if __name__=='__main__':sys.exit(main())
