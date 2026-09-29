"""Bounded revision of the 120 already reviewed groups, never 30K expansion."""
import argparse, json, os, time, traceback, shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import numpy as np
from ..aggressive_data_v16.run import GROUP_NAMES as OLD_NAMES, initialize as base_initialize, STATE
from ..s7_balanced_v2 import materialize as old
from ..s7_balanced_v2.scale import pair_shared_scale
from ..s7_balanced_v2.layered_geometry import canonical_mirrored_contours
from ..seam_context_v3.augmentation import paired_mirror
from ..s7_compound_v1.materialize import save_json,read,digest
from ..s7_compound_v1.geometry import rng_for,masks
from ..distribution_audit_20260923.measure import outline,pair_metrics
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample,save_sample
from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import changed_report
from .geometry import augment

GROUP_NAMES=dict(OLD_NAMES,mild='连续弱腐蚀3–8px',gaps='1–4处缺口（每处5–15px）',
    gaps_weak='1–4处缺口＋连续弱腐蚀3–8px',notch_k4='专项：实际4处有效缺口')


def initialize(args):
    base_initialize(args)


def process(task):
    previous_pair,k=task;out=Path(STATE['args']['out']);root=STATE['root']
    slot=int(Path(previous_pair[0]['sample_path']).stem.split('_')[0]);recipe=previous_pair[0]['recipe']
    try:
        # Non-depth recipes must remain byte-identical in all numeric arrays.
        if recipe in ('clean','partial'):
            records=[]
            for r in previous_pair:
                if digest(r['sample_path'])!=r['sample_sha256'] or digest(r['proof_path'])!=r['proof_sha256']:
                    raise ValueError('previous sample/proof changed')
                new=dict(r)
                new['previous_review_id']=r['id'];new['unchanged_non_depth_recipe']=True
                for field,folder in [('sample_path','samples'),('proof_path','proof')]:
                    dst=out/folder/Path(r[field]).name;dst.parent.mkdir(exist_ok=True);shutil.copy2(r[field],dst);new[field]=str(dst)
                records.append(new)
            save_json(out/'groups'/f'{slot:05d}.json',dict(slot=slot,records=records,attempts=0,rejections={},unchanged_control=True))
            return dict(slot=slot,status='passed',records=records,attempts=0,reasons={})
        group=read(root/'groups'/f'{slot:05d}.json');entries=group['entries']
        originals=(old.clean_positive(group['source_positive_pair_id']),old.negative_source(STATE['negative'][group['source_negative_pair_id']])[0])
        base=[];primary=[];fields=[];proofs=[];old_reports=[]
        for entry,original,r in zip(entries,originals,previous_pair):
            prior,report=load_sample(root/entry['artifact_path']);old_reports.append(report)
            before,_=pair_shared_scale(original,report['pair_shared_scale']['requested_mean_area_px2'],topology_backoff=True,identity_fallback=True)
            if entry['offline_paired_mirror']:before=canonical_mirrored_contours(paired_mirror(before,entry['offline_paired_mirror']))
            if digest(r['proof_path'])!=r['proof_sha256']:raise ValueError('reviewed cut proof hash changed')
            with np.load(r['proof_path'],allow_pickle=False) as z:proof={key:z[key].copy() for key in z.files}
            for side in 'ab':
                if not np.array_equal(np.unpackbits(proof['packed_fragment_'+side],axis=1),getattr(before,'mask_'+side)[0]):
                    raise ValueError('fragment base changed')
            with np.load(root/entry['weather_artifact'],allow_pickle=False) as z:main={key:z[key].copy() for key in z.files}
            if entry.get('background_artifact'):
                with np.load(root/entry['background_artifact'],allow_pickle=False) as z:
                    prior=replace(before,**{'mask_'+s:np.unpackbits(z['packed_before_'+s],axis=1).astype(np.float32)[None] for s in 'ab'})
            primary.append(prior);base.append(before);fields.append(main);proofs.append(proof)
        reasons=Counter();result=None
        for attempt in range(48):
            try:
                result=augment(tuple(base),tuple(primary),fields,previous_pair,proofs,recipe,
                    rng_for(26092771,'v17-depth-only',slot,attempt),k);break
            except ValueError as error:reasons[str(error)]+=1
        if result is None:return dict(slot=slot,status='failed',attempts=48,reasons=dict(reasons),requested_k=k)
        final,details,newfields,gap,trimmed,newprimary=result;records=[]
        for i,(r,sample,detail) in enumerate(zip(previous_pair,final,details)):
            pid=r['id'].replace('curved-v16','depth-v17');sample=replace(sample,pair_id=pid)
            report=changed_report(base[i],sample,recipe)
            report.update(schema_version='depth-v17-paired-review/1',recipe=recipe,base_v14_pair_id=r['baseline_pair_id'],
                compound=dict(recipe=recipe,damage=detail['primary_damage'],partial=old_reports[i]['compound']['partial']),
                paired_review=detail,not_full_training_dataset=True)
            for side in 'ab':report['side_'+side].update(detail['primary_damage'].get(side,{}))
            path=out/'samples'/f'{slot:05d}_{i}.npz';save_sample(path,sample,report)
            loaded,_=load_sample(path)
            for field in ('mask_a','mask_b','target_a','target_b','points_rc_a','points_rc_b'):
                if not np.array_equal(getattr(sample,field),getattr(loaded,field)):raise ValueError('archive roundtrip changed')
            arrays={}
            for stage,ss in [('fragment',base[i]),('trim',trimmed[i]),('primary',newprimary[i]),('final',sample)]:
                for side in 'ab':arrays['packed_'+stage+'_'+side]=np.packbits(getattr(ss,'mask_'+side)[0].astype(bool),axis=1)
            for kind,values in newfields[i].items():arrays.update({kind+'_'+key:value for key,value in values.items()})
            if i==0:arrays.update({'gap_'+key:value for key,value in gap.items()})
            proof=out/'proof'/path.name;proof.parent.mkdir(exist_ok=True);np.savez_compressed(proof,**arrays)
            groups=list(r['groups'])
            if recipe.startswith('gaps') and k==4:groups.append('notch_k4')
            record=dict(r,id=pid,pair_id=pid,previous_review_id=r['id'],previous_sample_path=r['sample_path'],
                previous_proof_path=r['proof_path'],previous_proof_sha256=r['proof_sha256'],
                sample_path=str(path),proof_path=str(proof),sample_sha256=digest(path),proof_sha256=digest(proof),
                detail=detail,groups=groups,new_metrics=pair_metrics(*[outline(m) for m in masks(sample)],sample.translation_a_to_b_rc if sample.label else None),
                inherited_correspondences=int((sample.target_a>=0).sum()),requested_gap_count=k if recipe.startswith('gaps') else 0)
            records.append(record)
        save_json(out/'groups'/f'{slot:05d}.json',dict(slot=slot,records=records,attempts=attempt+1,rejections=dict(reasons)))
        return dict(slot=slot,status='passed',records=records,attempts=attempt+1,reasons=dict(reasons),requested_k=k)
    except BaseException as error:
        return dict(slot=slot,status='failed',error=repr(error),traceback=traceback.format_exc(),requested_k=k)


def tasks_from_review(previous):
    review=read(previous/'review_bundle_01/manifest.json');audit=read(previous/'review_bundle_01/pixel_audit.json')
    if audit['status']!='passed' or audit['errors'] or audit['pairs']!=len(review['entries']):raise ValueError('previous complete audited review required')
    if audit['pilot_manifest_sha256']!=digest(previous/'review_bundle_01/manifest.json'):raise ValueError('previous audited manifest changed')
    groups={}
    for r in review['entries']:groups.setdefault(Path(r['sample_path']).stem.split('_')[0],[]).append(r)
    if len(groups)!=120 or len(review['entries'])!=240:raise ValueError('exact approved-cut population required')
    for pair in groups.values():
        pair.sort(key=lambda r:not r['label'])
        if len(pair)!=2 or [bool(r['label']) for r in pair]!=[True,False]:raise ValueError('reviewed positive/negative group pairing changed')
    notch=[(slot,pair) for slot,pair in groups.items() if pair[0]['recipe'].startswith('gaps')]
    def length(item):return min(v['eligible_length_px'] for r in item[1] for v in r['detail']['primary_damage'].values() if v.get('applied'))
    four={slot for slot,_ in sorted(notch,key=lambda item:(-length(item),item[0]))[:5]}
    remaining={slot:1+i%3 for i,(slot,_) in enumerate((item for item in notch if item[0] not in four))}
    return [(pair,4 if slot in four else remaining.get(slot,1)) for slot,pair in groups.items()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--previous',required=True)
    p.add_argument('--out',required=True);p.add_argument('--probe',action='store_true');a=p.parse_args()
    out=Path(a.out);previous=Path(a.previous)
    if out.exists():raise ValueError('fresh output only')
    tasks=tasks_from_review(previous)
    if a.probe:
        selected={};four=[]
        for task in tasks:
            recipe=task[0][0]['recipe'];selected.setdefault(recipe,task)
            if recipe.startswith('gaps') and task[1]==4:four.append(task)
        tasks=list(selected.values())+[t for t in four[:1] if t not in selected.values()]
    out.mkdir(parents=True);(out/'samples').mkdir();(out/'groups').mkdir();start=time.time()
    save_json(out/'protocol.json',dict(schema='aggressive-v17-depth-review/1',baseline=a.baseline,previous=a.previous,
        previous_manifest_sha256=digest(previous/'review_bundle_01/manifest.json'),
        weak_peak_px=[3,8],major_peak_px=[5,15],major_high_exclusive=True,combined_depth_cap_px=15,
        notch_count=[1,4],strong_peak_final_two_sided_gap_px=[5,25],gap_gate_includes_standalone_weak=True,
        background_peak_px=[1,3],background_coverage=.70,unchanged_approved_cuts=True,
        purpose='stratified per-type review, not training-frequency sample',no_full_generation_authorized=True,
        training_started=False,probe=a.probe,workers=4,source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}))
    records=[];results=[]
    with ProcessPoolExecutor(4,initializer=initialize,initargs=(dict(baseline=a.baseline,out=a.out),)) as pool:
        for result in pool.map(process,tasks):
            records.extend(result.get('records',[]));receipt={k:v for k,v in result.items() if k!='records'};results.append(receipt)
            save_json(out/'attempts'/f'{result["slot"]:05d}.json',receipt)
            save_json(out/'status.json',dict(status='generating',processed=len(results),groups=len(tasks),pairs=len(records),
                failed=sum(x['status']!='passed' for x in results),training_started=False))
    counts=Counter(k for r in records for k in r['groups']);missing={k:10-counts[k] for k in GROUP_NAMES if counts[k]<10}
    failures=[r for r in results if r['status']!='passed']
    save_json(out/'manifest.json',dict(artifact_root=str(out),entries=records))
    save_json(out/'generation_audit.json',dict(attempts=results,counts=dict(counts),missing=missing))
    if failures or (missing and not a.probe):
        save_json(out/'failure.json',dict(failures=failures,missing=missing,training_started=False,no_automatic_retry=True))
        raise ValueError('review incomplete; keep all failures, do not lower K or drop groups')
    complete=dict(status='probe_complete' if a.probe else 'generated_pending_pixel_audit_and_review',pairs=len(records),
        accepted_groups=len(records)//2,processed_groups=len(results),elapsed_seconds=time.time()-start,missing=missing,training_started=False)
    save_json(out/'generation_complete.json',complete);save_json(out/'status.json',complete);print(json.dumps(complete))

if __name__=='__main__':main()
