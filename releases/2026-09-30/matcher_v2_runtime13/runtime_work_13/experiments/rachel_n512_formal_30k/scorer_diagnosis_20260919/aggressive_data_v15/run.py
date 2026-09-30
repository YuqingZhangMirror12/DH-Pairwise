"""Bounded CPU-only paired pilot; never expands to full training data."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np

from ..s7_balanced_v2 import materialize as old
from ..s7_balanced_v2.scale import pair_shared_scale
from ..s7_balanced_v2.layered_geometry import canonical_mirrored_contours
from ..s7_balanced_v2.conservative_weather import parts
from ..seam_context_v3.augmentation import paired_mirror
from ..s7_compound_v1.materialize import save_json,read,digest
from ..s7_compound_v1.geometry import rng_for,masks
from ..distribution_audit_20260923.measure import outline,pair_metrics
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample,save_sample
from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import changed_report
from .geometry import augment,topology

STATE={}
GROUP_NAMES={
 'clean':'无腐蚀（新增缩短裁切后，剩余接缝不腐蚀）','mild':'连续弱腐蚀1–4px',
 'wave':'起伏退蚀','wave_weak':'起伏＋连续弱腐蚀','local_abrupt':'局部突变腐蚀',
 'local_abrupt_weak':'局部突变＋连续弱腐蚀','local_gradual':'局部渐进腐蚀',
 'local_gradual_weak':'局部渐进＋连续弱腐蚀','gaps':'1–3处缺口',
 'gaps_weak':'缺口＋连续弱腐蚀','partial_end':'Partial：端部裁切',
 'partial_middle':'Partial：中段切除','mirror_h':'水平镜像','mirror_v':'垂直镜像',
 'native':'原生相邻碎片','gen5':'Gen5分组合并／丢片','union_tiny':'Gen4/Gen5合并与大小碎片',
 'unequal':'大小悬殊（原v14面积比≤1:4）','negative_cross_gen':'负例：跨生成组',
 'negative_cross_parent_same_gen':'负例：同生成组跨写卷','negative_same_parent_nonadjacent':'负例：同写卷不相邻',
 'trim_one':'新增：接缝一端缩短','trim_both':'新增：接缝两端缩短'}


def tags(entry,stats):
    recipe=entry['corrosion_recipe']
    result=[('partial_'+stats['augmentation']['partial']['mode']) if recipe=='partial' else recipe]
    if entry['label']:
        mirror=entry['offline_paired_mirror']
        if mirror:result.append('mirror_h' if mirror=='horizontal' else 'mirror_v')
        kind=entry['source_stratum']
        if kind=='native_positive':result.append('native')
        if kind=='gen5_partition_positive':result.append('gen5')
        if kind=='union_positive_tiny':result.append('union_tiny')
        if stats['area_ratio']<=.25:result.append('unequal')
    else:result.append('negative_'+entry['negative_kind'])
    return result


def initialize(args):
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import cv2,torch
    cv2.setNumThreads(1);torch.set_num_threads(1)
    root=Path(args['baseline']);protocol=read(root/'protocol.json');options=protocol['options'].copy()
    old.initialize(options)
    STATE.update(args=args,root=root,profile=protocol['distribution_revision'],
        negative={e['pair_id']:e for e in old.STATE['plan']['negative']})


def reused_group(previous,out,slot):
    """Reuse ONLY fully committed, byte-verified groups from the failed pilot.

The repair only moves the already mandatory aggregate topology check inside
the rejection loop. Committed groups passed that exact check previously.
Partial sample/proof files without a group receipt are never imported.
"""
    path=previous/'groups'/f'{slot:05d}.json'
    if not path.is_file():return None
    group=read(path)
    if len(group['records'])!=2:raise ValueError('incomplete group receipt')
    for record in group['records']:
        for field,sha_field,folder in [('sample_path','sample_sha256','samples'),('proof_path','proof_sha256','proof')]:
            src=Path(record[field]);expected=previous/folder/src.name
            if src!=expected or digest(src)!=record[sha_field]:raise ValueError('reused artifact identity mismatch')
            dst=out/folder/src.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst)
            record[field]=str(dst)
        record['reused_from_verified_group']=str(path)
    group['reused_from']=str(path)
    save_json(out/'groups'/path.name,group)
    return dict(slot=slot,status='passed',records=group['records'],attempts=group['attempts'],
                reasons=group['rejections'],reused_from=str(path))


def process(slot):
    root=STATE['root'];out=Path(STATE['args']['out']);group=read(root/'groups'/f'{slot:05d}.json')
    entries=group['entries'];baseline=[];reports=[];fields=[];primary=[];base=[]
    originals=(old.clean_positive(group['source_positive_pair_id']),
               old.negative_source(STATE['negative'][group['source_negative_pair_id']])[0])
    for i,(entry,original) in enumerate(zip(entries,originals)):
        sample,report=load_sample(root/entry['artifact_path']);baseline.append(sample);reports.append(report)
        scale=report['pair_shared_scale'];before,_=pair_shared_scale(original,scale['requested_mean_area_px2'],
            topology_backoff=True,identity_fallback=True)
        if entry['offline_paired_mirror']:
            before=canonical_mirrored_contours(paired_mirror(before,entry['offline_paired_mirror']))
        with np.load(root/entry['weather_artifact'],allow_pickle=False) as z:
            main={k:z[k].copy() for k in z.files}
        for side in 'ab':
            expected=np.unpackbits(main['packed_preweather_'+side],axis=1).astype(bool)
            if not np.array_equal(expected,getattr(before,'mask_'+side)[0]>0):
                raise ValueError('cannot reproduce exact v14 fragment-stage mask')
        if entry.get('background_artifact'):
            with np.load(root/entry['background_artifact'],allow_pickle=False) as z:
                changes={'mask_'+s:np.unpackbits(z['packed_before_'+s],axis=1).astype(np.float32)[None] for s in 'ab'}
            primary.append(replace(before,**changes))
        else:primary.append(sample)
        fields.append(main);base.append(before)
    reasons=Counter();recipe=entries[0]['corrosion_recipe'];mode='one' if slot%2 else 'both'
    result=None
    for attempt in range(24):
        rng=rng_for(26092751,'paired-aggressive',slot,attempt)
        try:
            result=augment(tuple(base),tuple(primary),fields,reports,recipe,rng,mode)
            break
        except ValueError as error:reasons[str(error)]+=1
    if result is None:
        return dict(slot=slot,status='rejected',attempts=24,reasons=dict(reasons))
    final,details,newfields,gap_arrays,trimmed,newprimary=result
    records=[]
    for i,(entry,sample,detail) in enumerate(zip(entries,final,details)):
        pid=f'aggressive-v15-{slot:05d}-{i}'
        sample=replace(sample,pair_id=pid,fragment_a_token=sample.fragment_a_token+'@v15',fragment_b_token=sample.fragment_b_token+'@v15')
        report=changed_report(base[i],sample,recipe)
        report.update(schema_version='paired-v15-review/1',recipe=recipe,base_v14_pair_id=entry['pair_id'],
            compound=dict(recipe=recipe,damage=detail['primary_damage'],partial=reports[i]['compound']['partial']),
            paired_review=detail,not_full_training_dataset=True)
        for side in 'ab':
            report['side_'+side].update(detail['primary_damage'].get(side,{}))
            topology(getattr(base[i],'mask_'+side)[0].astype(bool),getattr(sample,'mask_'+side)[0].astype(bool))
        relative=f'samples/{slot:05d}_{i}.npz';save_sample(out/relative,sample,report)
        loaded,_=load_sample(out/relative)
        for name in ('mask_a','mask_b','target_a','target_b','points_rc_a','points_rc_b'):
            if not np.array_equal(getattr(sample,name),getattr(loaded,name)):raise AssertionError('archive roundtrip')
        arrays={}
        for stage,ss in [('fragment',base[i]),('trim',trimmed[i]),('primary',newprimary[i]),('final',sample)]:
            for side in 'ab':arrays[f'packed_{stage}_{side}']=np.packbits(getattr(ss,'mask_'+side)[0].astype(bool),axis=1)
        for kind,values in newfields[i].items():
            arrays.update({kind+'_'+k:v for k,v in values.items()})
        if i==0:arrays.update({'gap_'+k:v for k,v in gap_arrays.items()})
        proof=out/'proof'/f'{slot:05d}_{i}.npz';proof.parent.mkdir(exist_ok=True)
        np.savez_compressed(proof,**arrays)
        beforemetrics=pair_metrics(*[outline(m) for m in masks(baseline[i])],baseline[i].translation_a_to_b_rc if sample.label else None)
        aftermetrics=pair_metrics(*[outline(m) for m in masks(sample)],sample.translation_a_to_b_rc if sample.label else None)
        groups=tags(entry,group['statistics'][i])+['trim_'+mode]
        records.append(dict(id=pid,pair_id=pid,label=bool(sample.label),recipe=recipe,
            sample_path=str(out/relative),artifact_path=relative,baseline_sample_path=str(root/entry['artifact_path']),
            baseline_pair_id=entry['pair_id'],source_pair_id=entry['source_pair_id'],source_stratum=entry['source_stratum'],
            source_row=entry['source_row'],negative_kind=entry.get('negative_kind'),
            mirror=entry['offline_paired_mirror'],groups=groups,detail=detail,
            baseline_metrics=beforemetrics,new_metrics=aftermetrics,
            inherited_correspondences=int((sample.target_a>=0).sum()),
            sample_sha256=digest(out/relative),proof_path=str(proof),proof_sha256=digest(proof),
            stage_order=['existing_structure_scale_mirror','additional_end_trim','one_exclusive_primary','allowed_light_70percent'],
            image=None))
    save_json(out/'groups'/f'{slot:05d}.json',dict(slot=slot,records=records,attempts=attempt+1,rejections=dict(reasons)))
    return dict(slot=slot,status='passed',records=records,attempts=attempt+1,reasons=dict(reasons))


def planned_slots(root,maximum):
    manifest=read(root/'train_s7b_24k.json');entries=manifest['entries']
    metrics={r['pair_id']:r for r in (json.loads(line) for line in (root/'pair_metrics.jsonl').open())}
    buckets={k:[] for k in GROUP_NAMES if not k.startswith('trim_')}
    for entry in entries:
        slot=int(Path(entry['artifact_path']).stem.split('_')[0])
        for tag in tags(entry,metrics[entry['pair_id']]):
            if slot not in buckets[tag]:buckets[tag].append(slot)
    rng=np.random.default_rng(26092751)
    for k,v in buckets.items():
        if len(v)<10:raise ValueError('insufficient existing v14 source group: '+k)
        rng.shuffle(v)
    result=[];seen=set()
    # Interleaving predeclared recipe/structure/negative-source strata, not
    # selecting visually attractive examples. All rejects remain in audit.
    for rank in range(max(map(len,buckets.values()))):
        for pool in buckets.values():
            if rank<len(pool) and pool[rank] not in seen:
                result.append(pool[rank]);seen.add(pool[rank])
                if len(result)>=maximum:return result
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--max-groups',type=int,default=240)
    parser.add_argument('--per-type',type=int,default=10);parser.add_argument('--pilot-groups',type=int,default=0)
    parser.add_argument('--reuse-committed-groups')
    a=parser.parse_args();root=Path(a.baseline).resolve();out=Path(a.out).resolve()
    if a.workers>4 or a.max_groups>400 or a.per_type!=10:raise ValueError('bounded review pilot only; no full generation')
    if out.exists():raise ValueError('fresh output only; no automatic restart')
    out.mkdir(parents=True);(out/'samples').mkdir();(out/'groups').mkdir();start=time.time()
    slots=planned_slots(root,a.max_groups)
    if a.pilot_groups:slots=slots[:a.pilot_groups]
    save_json(out/'protocol.json',dict(schema='aggressive-v15-paired-review/1',baseline=str(root),
        baseline_manifest_sha256=digest(root/'train_s7b_24k.json'),
        purpose='stratified review pilot, not training-proportion sample',planned_slots=slots,
        target_seam_shortening=[.20,.30],raster_acceptance=[.18,.32],
        strong_peak_final_two_sided_gap_px=[5,15],weak_and_light_unchanged=True,
        no_full_generation_authorized=True,training_started=False,original_data_modified=False,
        workers=a.workers,maximum_groups=a.max_groups,
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}))
    previous=Path(a.reuse_committed_groups).resolve() if a.reuse_committed_groups else None
    if previous:
        prior=read(previous/'protocol.json')
        if prior['baseline_manifest_sha256']!=digest(root/'train_s7b_24k.json') or prior['planned_slots']!=slots:
            raise ValueError('cannot reuse a different baseline or slot plan')
        if not (previous/'failure.json').is_file():raise ValueError('repair reuse requires preserved original failure')
    completed=[];records=[];counts=Counter();failure=None
    options=dict(baseline=str(root),out=str(out))
    try:
        with ProcessPoolExecutor(a.workers,initializer=initialize,initargs=(options,)) as pool:
            for offset in range(0,len(slots),a.workers):
                batch=slots[offset:offset+a.workers]
                reused={slot:reused_group(previous,out,slot) for slot in batch} if previous else {}
                fresh=iter(pool.map(process,[slot for slot in batch if not reused.get(slot)]))
                for slot in batch:
                    result=reused.get(slot) or next(fresh)
                    completed.append({k:v for k,v in result.items() if k!='records'})
                    if result['status']=='passed':
                        records+=result['records']
                        for r in result['records']:counts.update(r['groups'])
                    save_json(out/'attempts'/f'{slot:05d}.json',completed[-1])
                    save_json(out/'status.json',dict(status='generating',processed=len(completed),pairs=len(records),
                        groups=dict(counts),elapsed_seconds=time.time()-start,training_started=False))
                if all(counts[k]>=a.per_type for k in GROUP_NAMES):break
        missing={k:a.per_type-counts[k] for k in GROUP_NAMES if counts[k]<a.per_type}
        save_json(out/'generation_audit.json',dict(attempts=completed,counts=dict(counts),missing=missing))
        save_json(out/'manifest.json',dict(artifact_root=str(out),entries=records))
        if missing and not a.pilot_groups:raise ValueError('bounded pilot insufficient type coverage:'+repr(missing))
        status='probe_complete' if a.pilot_groups else 'generated_pending_pixel_audit_and_review'
        save_json(out/'generation_complete.json',dict(status=status,pairs=len(records),accepted_groups=len(records)//2,
            processed_groups=len(completed),elapsed_seconds=time.time()-start,missing=missing,training_started=False))
        save_json(out/'status.json',dict(status=status,processed=len(completed),pairs=len(records),
            groups=dict(counts),elapsed_seconds=time.time()-start,training_started=False))
        print(json.dumps(dict(status=status,pairs=len(records),elapsed_seconds=time.time()-start)))
    except BaseException as error:
        save_json(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),pairs=len(records),
            training_started=False,no_automatic_retry=True));raise

if __name__=='__main__':main()
