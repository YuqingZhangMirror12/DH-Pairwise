"""Offline S5 step3/cap2048 materialization, paired with existing v4 N512.

No training/GPU. --diagnostic-only --smoke-count64 reads 64 stratified TRAIN
samples and prints input-length statistics without creating any files. Normal
production writes only a new --output root; no original/cache is modified.
"""
import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import (
    MaterializedRachelDataset,load_sample,save_sample)
from staging.pairwise_v0_2.pairwise_data.rachel_paired_density_dataset import (
    PairedSourceDensityDataset,file_sha256)
from staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_v4 import (
    SourceDensityRecordResolverV4,CleanSourceDensityV4Dataset)
from staging.pairwise_v0_2.pairwise_data.rachel_step_density import (
    sample_pair_step_contours,resample_source_step,assert_preserved,CAP)
from staging.pairwise_v0_2.pairwise_data.rachel_step_dataset import (
    SCHEMA,PAIR_SCHEMA,digest,validate_step_sample)


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def stratified_indices(entries,count,seed):
    """Round-robin deterministic stratum coverage; return original order."""
    if count is None:return list(range(len(entries)))
    if not 1<=count<=len(entries):raise ValueError('invalid diagnostic/subset count')
    groups={}
    for i,e in enumerate(entries):
        row=e.get('source_row',e)
        key=e.get('source_stratum') or row.get('label_origin') or str(bool(e['label']))
        groups.setdefault(str(key),[]).append(i)
    rng=np.random.RandomState(seed)
    for values in groups.values():rng.shuffle(values)
    selected=[]
    while len(selected)<count:
        for key in sorted(groups):
            if groups[key] and len(selected)<count:selected.append(groups[key].pop())
    return sorted(selected)


def summarize_sampling(details):
    counts=[d['sides'][s]['count'] for d in details for s in 'ab']
    arcsteps=[d['sides'][s]['actual_arc_step_px'] for d in details for s in 'ab']
    perimeters=[d['sides'][s]['smoothed_perimeter_px'] for d in details for s in 'ab']
    def stats(values):
        return dict(zip(('min','p50','p90','p95','max'),map(float,np.percentile(values,[0,50,90,95,100])))) if values else None
    return dict(pair_count=len(details),fragment_count=len(counts),points=stats(counts),
        arc_step_px=stats(arcsteps),perimeter_px=stats(perimeters),
        shared_delta_enlarged_pairs=sum(d['shared_delta_enlarged'] for d in details),
        cap_reached_fragments=sum(d['sides'][s]['cap_reached'] for d in details for s in 'ab'))


def diagnostic(args):
    if args.split!='train' or not args.fixed_e1_manifest or args.smoke_count is None:
        raise ValueError('diagnostic requires TRAIN fixed manifest and explicit smoke count')
    dataset=MaterializedRachelDataset(args.fixed_e1_manifest)
    indices=stratified_indices(dataset.entries,args.smoke_count,args.seed)
    records=[]
    for i in indices:
        sample,_=dataset.weathered(i)
        _,_,detail=sample_pair_step_contours(sample.mask_a,sample.mask_b)
        records.append(dict(pair_id=sample.pair_id,index=i,stratum=dataset.entries[i].get('source_stratum'),sampling=detail))
    result=dict(schema_version='rachel-step-input-diagnostic/1',status='complete',split='train',
        full_split=False,source_pair_count=len(dataset),source_manifest_sha256=file_sha256(args.fixed_e1_manifest),
        seed=args.seed,model_inference=False,gpu_used=False,files_written=False,
        summary=summarize_sampling([r['sampling'] for r in records]),records=records)
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
    return 0


def _source(args):
    if args.split=='train':
        if not args.fixed_e1_manifest or not args.paired512_manifest:
            raise ValueError('TRAIN requires fixed E1 manifest and existing paired512 manifest')
        resolver=SourceDensityRecordResolverV4(args.fixed_e1_manifest,args.canonical_root)
        control=PairedSourceDensityDataset(args.paired512_manifest)
        if control.contour_cap!=512 or control.protocol['pipeline'].get('source_density_version')!='v4':
            raise ValueError('control must be source-rebuilt v4 N512')
        sha=file_sha256(args.fixed_e1_manifest)
        if control.protocol.get('source_manifest_sha256')!=sha:raise ValueError('TRAIN source differs from control')
        if [e['pair_id'] for e in resolver.entries]!=[e['pair_id'] for e in control.entries]:
            raise ValueError('TRAIN original and N512 membership/order differ')
        meta=dict(kind='paired_train_manifest',manifest_path=str(control.manifest_path),
            artifact_root=str(control.root),identity_sha256=control.identity,
            source_manifest_sha256=sha,source_density_version='v4',contour_cap=512,
            protocol=control.protocol)
        def resolve(index):return resolver.resolve(index)
        def existing(index):
            entry=control.entries[index];p=control.root/entry['artifact_path']
            sample,report=control.weathered(index)
            return sample,report,p,file_sha256(p)
        return resolver.entries,resolver.manifest_path,meta,resolve,existing
    if not args.fixed512_cache:raise ValueError('clean split requires existing v4 N512 cache')
    resolver=CleanSourceDensityV4Dataset(args.canonical_root,args.split,512,cache_dir=None)
    cache=Path(args.fixed512_cache).resolve(strict=True)
    identity=json.loads((cache/'cache_identity.json').read_text())
    if identity!=resolver.protocol:
        raise ValueError('N512 clean source/version/runtime/cache identity differs; do not regenerate')
    receipt=json.loads((cache.parent/('clean_'+args.split+'_preparation.json')).read_text())
    if (receipt.get('status')!='complete' or receipt.get('full_split') is not True or
            receipt.get('source_density_version')!='v4' or receipt.get('selected_count')!=3000 or
            receipt.get('ownership_protocol')!='density-local-unknown-ownership/4' or receipt.get('failures')):
        raise ValueError('N512 clean control must have complete original3000 receipt')
    meta=dict(kind='clean_cache',artifact_root=str(cache),identity_sha256=resolver.identity,
        source_manifest_sha256=file_sha256(resolver.manifest_path),source_density_version='v4',
        contour_cap=512,protocol=identity)
    def resolve(index):
        sample,report,source,offsets,clean,allowance=resolver._resolve_clean(index)
        return sample,report,source,offsets,clean,allowance,report.get('source_resolution',{})
    def existing(index):
        row=resolver.rows[index];p=cache/(hashlib.sha256(row['pair_id'].encode()).hexdigest()+'.npz')
        sample,report=load_sample(p)
        if (sample.pair_id!=row['pair_id'] or bool(sample.label)!=bool(row['label']) or
                report.get('paired_density',{}).get('identity_sha256')!=resolver.identity or
                report.get('density',{}).get('cap')!=512):raise ValueError('clean N512 record differs')
        return sample,report,p,file_sha256(p)
    return resolver.rows,resolver.manifest_path,meta,resolve,existing


def run(args):
    if args.diagnostic_only:return diagnostic(args)
    if not args.output or not args.canonical_root:raise ValueError('materialization needs canonical root and output')
    entries,manifest,control_meta,resolve,existing=_source(args)
    expected=24000 if args.split=='train' else 3000
    if len(entries)!=expected:raise ValueError('source must retain full original24K/3K population')
    indices=stratified_indices(entries,args.smoke_count,args.seed)
    output=Path(args.output).resolve();canonical=Path(args.canonical_root).resolve()
    control_root=Path(control_meta['artifact_root']).resolve()
    forbidden=[canonical,control_root]
    if args.fixed_e1_manifest:
        forbidden.append(Path(json.loads(Path(args.fixed_e1_manifest).read_text())['artifact_root']).resolve())
    if any(output==p or p in output.parents for p in forbidden):raise ValueError('S5 output would modify existing source')
    from staging.pairwise_v0_2.pairwise_data import (rachel_step_density,rachel_step_dataset,
        rachel_source_density,rachel_density_ownership_v2,rachel_density_ownership_v3,
        rachel_density_ownership_v4,rachel_preprocess)
    protocol=dict(schema_version='rachel-step-source-materialization/1',split=args.split,
        sampling_mode='paired_shared_arc_step',step_px=3.,contour_cap=CAP,smoothing_sigma=3.,
        source_density_version='v4',ownership_protocol='density-local-unknown-ownership/4',
        source_manifest=str(manifest),source_manifest_sha256=file_sha256(manifest),
        source_selection_sha256=digest([entries[i]['pair_id'] for i in indices]),
        selected_pair_ids=[entries[i]['pair_id'] for i in indices],source_indices=indices,
        full_split=args.smoke_count is None,control512=control_meta,
        physical_masks_unchanged=True,physical_patch_windows_px=[7,16,32,64],
        pair_labels='unchanged fixed source labels',pose_targets='unchanged fixed source GT',
        assignment_targets='fresh v4 source-cell ancestry; paired shared-arc-step descendants',
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        code_sha256={m.__name__:file_sha256(m.__file__) for m in (rachel_step_density,rachel_step_dataset,
            rachel_source_density,rachel_density_ownership_v2,rachel_density_ownership_v3,
            rachel_density_ownership_v4,rachel_preprocess)},
        materializer_sha256=file_sha256(__file__),
        gt_translation_canary_policy='every smoke pair' if args.smoke_count else 'first 3 selected pairs only',
        gt_translation_canary_source_indices=indices if args.smoke_count else indices[:3])
    identity=digest(protocol)
    if output.exists():
        if not args.resume:raise ValueError('existing S5 output requires explicit --resume')
        p=output/'protocol.json'
        if not p.exists() or json.loads(p.read_text())!=protocol:raise ValueError('existing S5 output identity differs')
    else:output.mkdir(parents=True)
    with (output/'.writer.lock').open('a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError('another S5 materializer owns this output') from error
        return _produce(args,entries,indices,output,protocol,identity,resolve,existing)


def _produce(args,entries,indices,output,protocol,identity,resolve,existing):
    save_json(output/'protocol.json',protocol)
    records=[];failures=[];details=[]
    for index in indices:
        entry=entries[index];pid=entry['pair_id'];directory=output/'pairs'/hashlib.sha256(pid.encode()).hexdigest()
        path=directory/'step3.npz';marker=directory/'complete.json'
        try:
            control,control_report,control_path,control_sha=existing(index)
            if marker.exists():
                done=json.loads(marker.read_text())
                canary_expected=index in protocol['gt_translation_canary_source_indices']
                if (done.get('schema_version')!=PAIR_SCHEMA or done.get('status')!='complete' or
                        done.get('gt_translation_canary_performed')!=canary_expected or
                        done.get('gt_translation_canary_passed')!=(True if canary_expected else None) or
                        done.get('physical_fields_preserved') is not True or done.get('identity_sha256')!=identity or
                        done.get('pair_id')!=pid or done.get('control_artifact_sha256')!=control_sha or
                        file_sha256(path)!=done.get('artifact_sha256')):raise ValueError('S5 resume archive/control differs')
                result,derived=load_sample(path);validate_step_sample(result,derived,identity)
            else:
                sample,report,source,offsets,clean,allowance,provenance=resolve(index)
                assert_preserved(sample,control)
                if sample.pair_id!=pid or bool(sample.label)!=bool(entry['label']):raise ValueError('resolved pair differs')
                result,derived,_=resample_source_step(sample,report,source,offsets,clean,ownership_allowance_px=allowance)
                canary_performed=index in protocol['gt_translation_canary_source_indices']
                if canary_performed:
                    canary,_,_=resample_source_step(replace(sample,translation_a_to_b_rc=sample.translation_a_to_b_rc+1000),
                        report,source,offsets,clean,ownership_allowance_px=allowance)
                    if not np.array_equal(result.target_a,canary.target_a) or not np.array_equal(result.target_b,canary.target_b):
                        raise ValueError('GT translation leaked into S5 matching targets')
                derived['step_density']=dict(identity_sha256=identity,source_pair_id=pid,source_split=args.split)
                derived['source_resolution']=provenance
                validate_step_sample(result,derived,identity);save_sample(path,result,derived)
                reread,rereport=load_sample(path);validate_step_sample(reread,rereport,identity);assert_preserved(sample,reread)
                done=dict(schema_version=PAIR_SCHEMA,status='complete',identity_sha256=identity,pair_id=pid,
                    artifact_sha256=file_sha256(path),control_artifact_sha256=control_sha,
                    gt_translation_canary_performed=canary_performed,
                    gt_translation_canary_passed=True if canary_performed else None,physical_fields_preserved=True)
                save_json(marker,done)
            row=entry.get('source_row',entry)
            records.append(dict(pair_id=pid,label=bool(entry['label']),source_row=row,source_index=index,
                source_stratum=entry.get('source_stratum'),artifact_path=str(path.relative_to(output)),
                artifact_sha256=done['artifact_sha256'],control_artifact_path=str(control_path.resolve()),
                control_artifact_sha256=control_sha))
            details.append(derived['step_sampling'])
        except Exception as error:
            failures.append(dict(pair_id=pid,index=index,error_type=type(error).__name__,error=str(error)))
        if len(records)%100==0:print(json.dumps(dict(split=args.split,completed=len(records),failed=len(failures))),flush=True)
    complete=len(records)==len(indices) and not failures
    record=dict(schema_version=SCHEMA,status='complete' if complete else 'incomplete',split=args.split,
        full_split=args.smoke_count is None,original_count=len(entries),selected_count=len(indices),
        completed_count=len(records),failed_count=len(failures),failures=failures,
        artifact_root=str(output),identity_sha256=identity,protocol=protocol,entries=records,
        stats=dict(positive=sum(e['label'] for e in records),negative=sum(not e['label'] for e in records)),
        sampling_summary=summarize_sampling(details),model_inference=False,gpu_used=False,
        gt_translation_canary_policy=protocol['gt_translation_canary_policy'],
        gt_translation_canary_planned_count=len(protocol['gt_translation_canary_source_indices']),
        source_density_version='v4',ownership_protocol='density-local-unknown-ownership/4',
        model_or_threshold_selected=False,completed_at_unix=time.time())
    save_json(output/'manifest.json',record)
    save_json(output/'status.json',{k:v for k,v in record.items() if k not in ('entries','protocol')})
    print(json.dumps({k:v for k,v in record.items() if k not in ('entries','protocol')},indent=2),flush=True)
    return 0 if complete else 2


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=('train','val','test'),required=True)
    p.add_argument('--canonical-root');p.add_argument('--output')
    p.add_argument('--fixed-e1-manifest');p.add_argument('--paired512-manifest')
    p.add_argument('--fixed512-cache');p.add_argument('--smoke-count',type=int)
    p.add_argument('--seed',type=int,default=20260914);p.add_argument('--diagnostic-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    return p


if __name__=='__main__':sys.exit(run(parser().parse_args()))
