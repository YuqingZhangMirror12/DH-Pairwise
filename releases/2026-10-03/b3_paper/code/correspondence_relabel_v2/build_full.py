"""One bounded CPU relabel build for admitted B3 TRAIN only; review, not train."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace,asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from catalog import source_catalog,expand,sidecars,sha,bound
from relabel import run,EvidenceError,POLICY,dense,sample_mask,state_counts

def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def initialize():
    import torch
    torch.set_num_threads(1)

def metrics(z,arrays):
    if 'near_distance_a' in arrays:d=arrays['near_distance_a']
    else:d=cKDTree(dense(sample_mask(z,'b'))[0]).query(z['points_rc_a'].astype(float)+z['translation_a_to_b_rc'])[0]
    result={}
    for mode in ('old','full','tight'):
        ta=z['target_a'] if mode=='old' else arrays[mode+'_a'];tb=z['target_b'] if mode=='old' else arrays[mode+'_b']
        i=np.flatnonzero(ta>=0);j=ta[i]
        gaps=np.linalg.norm(z['points_rc_a'][i].astype(float)+z['translation_a_to_b_rc']-z['points_rc_b'][j],axis=1)
        hist={}
        for name,bits in [('le3',d<=3),('3to8',(d>3)&(d<=8)),('8to15',(d>8)&(d<=15)),('gt15',d>15)]:
            near=bits&z['contour_valid_a']
            hist[name]=dict(match=int((near&(ta>=0)).sum()),ignore=int((near&(ta==-2)).sum()),unmatched=int((near&(ta==-1)).sum()))
        result[mode]=dict(a=state_counts(ta,z['contour_valid_a']),b=state_counts(tb,z['contour_valid_b']),
            near_contour_bins_a=hist,match_distances=gaps.tolist(),count=int(len(i)),
            max_px=float(gaps.max()) if len(i) else None,gt8=int((gaps>8+1e-6).sum()),between5and8=int(((gaps>5)&(gaps<=8+1e-6)).sum()))
    return result

def process(job):
    row,out=job;row=expand(row);a,e=row['admission'],row['record'];pid=a['pair_id']
    start=time.monotonic();sample_path=Path(a['sample_path'])
    source_bindings=[];files={}
    for k,(p,expected) in sidecars(row).items():
        source_bindings.append(dict(kind=k,**bound(p,expected)))
        with np.load(p,allow_pickle=False) as z:files[k]={key:z[key].copy() for key in z.files}
    z=files['sample']
    if str(z['pair_id'].item())!=pid or bool(z['label'])!=a['label']:raise ValueError('source class/ID mismatch')
    if not a['label']:
        if np.any(z['target_a']>=0) or np.any(z['target_b']>=0):raise ValueError('original negative carries correspondence')
        return dict(entry=dict(a,recipe=e['recipe'],label_overlay=None),summary=dict(pair_id=pid,stage=a['stage'],recipe=e['recipe'],label=False,status='negative_unchanged'))
    try:
        arrays,info=run(z,files['proof'],e,files.get('preweather'),files.get('latent'))
    except EvidenceError as exc:
        arrays={mode+'_'+s:np.full_like(z['target_'+s],-2) for mode in ('full','tight') for s in 'ab'}
        arrays.update({'old_'+s:z['target_'+s].copy() for s in 'ab'})
        info=dict(status='evidence_quarantined_all_ignore',error=str(exc),pairs=[],training_admitted=False)
    # Exercise the existing CPU loader/collator/target adapters on both variants.
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.data import collate
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.check_data_compatibility import check_batch
    sample,report=load_sample(sample_path)
    checks={}
    for mode in ('full','tight'):
        new=replace(sample,target_a=arrays[mode+'_a'],target_b=arrays[mode+'_b'])
        for field in sample.__dataclass_fields__:
            if field not in ('target_a','target_b'):
                if not np.array_equal(getattr(new,field),getattr(sample,field)):raise ValueError('nonlabel array modified')
        items=[(new,report,e)];checks[mode]=check_batch(items,collate(items))[0]
    # The original archive is immutable, not overwritten by a reconstructed npz.
    if sha(sample_path)!=a['sample_sha256']:raise ValueError('source changed while labeling')
    dest=Path(out)/'labels'/pid;dest.mkdir(parents=True)
    np.savez_compressed(dest/'labels.npz',**arrays)
    mm=metrics(z,arrays)
    info.update(pair_id=pid,stage=a['stage'],recipe=e['recipe'],label=True,v14_fallback=bool(e.get('v14_fallback')),
        original_source_pair_id=e.get('source_pair_id'),source_bindings=source_bindings,
        all_nonlabel_arrays_identical=True,original_archive_unchanged=True,loader_checks=checks,metrics=mm,
        seconds=time.monotonic()-start)
    write(dest/'audit.json',info)
    entry=dict(a,recipe=e['recipe'],label_overlay=bound(dest/'labels.npz'),label_audit=bound(dest/'audit.json'))
    summary={k:v for k,v in info.items() if k not in ('pairs','counts','source_bindings','loader_checks','policy','policy_notes')}
    return dict(entry=entry,summary=summary)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-new',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();out=args.out_new
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA must be explicitly disabled')
    if not 1<=args.workers<=8:raise ValueError('bounded CPU workers required')
    if out.exists():raise ValueError('new output required; no blind retry/resume')
    if os.statvfs(out.parent).f_bavail*os.statvfs(out.parent).f_frsize<4_000_000_000:raise ValueError('insufficient free disk')
    out.mkdir();(out/'labels').mkdir();started=time.time()
    sources=[bound(p) for p in Path(__file__).parent.glob('*.py')]
    try:
        rows,bindings=source_catalog();write(out/'protocol.json',dict(schema='B3-label-overlay-review/1',policy=asdict(POLICY),
            source_bindings=bindings,implementation=sources,workers=args.workers,expected=len(rows),
            source_catalog_only='exact B3 TRAIN catalog, no SELECT/CAL/TEST',training_admitted=False,
            old_data_modified=False,model_forwards=0,optimizer_updates=0,started_unix=started))
        summaries=[];entries=[]
        with (out/'row_results.jsonl').open('x') as stream:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
                for result in pool.map(process,[(r,str(out)) for r in rows],chunksize=4):
                    entries.append(result['entry']);summaries.append(result['summary'])
                    stream.write(json.dumps(result['summary'],ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    if len(entries)%500==0:
                        print(json.dumps(dict(done=len(entries),total=len(rows),seconds=time.time()-started)),flush=True)
        for b in bindings+sources:
            if sha(b['path'])!=b['sha256']:raise ValueError('bound source changed during full build')
        for mode in ('full','tight'):
            write(out/('manifest_'+mode+'.json'),dict(schema='B3-label-overlay-review/1',mode=mode,training_admitted=False,
                admission_required='Human visual approval, then independent new training admission; not old admission replacement',entries=entries))
        grouped=defaultdict(list)
        for r in summaries:
            if r['label']:
                for key in [r['stage']+'/ALL',r['stage']+'/'+r['recipe']]:grouped[key].append(r)
        stats={}
        for key,group in grouped.items():
            record=dict(rows=len(group),quarantined=sum(r['status'].startswith('evidence_quarantined') for r in group),modes={})
            for mode in ('old','full','tight'):
                gaps=np.array([d for r in group for d in r['metrics'][mode]['match_distances']])
                modes=dict(correspondence_pairs=len(gaps),max_px=float(gaps.max()) if len(gaps) else None,
                    p50_px=float(np.median(gaps)) if len(gaps) else None,p90_px=float(np.quantile(gaps,.9)) if len(gaps) else None,
                    gt8=int((gaps>8+1e-6).sum()),between5and8=int(((gaps>5)&(gaps<=8+1e-6)).sum()),
                    zero_match_samples=sum(r['metrics'][mode]['count']==0 for r in group),
                    fewer_than4_samples=sum(r['metrics'][mode]['count']<4 for r in group),
                    token_counts={s:{k:sum(r['metrics'][mode][s][k] for r in group) for k in ('match','ignore','unmatched','padding')} for s in 'ab'},
                    near_contour_bins_a={b:{k:sum(r['metrics'][mode]['near_contour_bins_a'][b][k] for r in group)
                        for k in ('match','ignore','unmatched')} for b in ('le3','3to8','8to15','gt15')})
                record['modes'][mode]=modes
            stats[key]=record
        write(out/'statistics.json',stats)
        quarantines=[{k:r.get(k) for k in ('pair_id','stage','recipe','error')} for r in summaries if r['status'].startswith('evidence_quarantined')]
        write(out/'quarantine.json',dict(rows=quarantines,count=len(quarantines),policy='all labels ignore; no new correspondence'))
        write(out/'complete.json',dict(status='review_only_label_build_complete',rows=len(entries),
            positives=sum(e['label'] for e in entries),negatives=sum(not e['label'] for e in entries),
            negatives_unchanged=True,all_nonlabel_arrays_unchanged=True,old_archives_unchanged=True,
            full_gt8=sum(v['modes']['full']['gt8'] for k,v in stats.items() if k.endswith('/ALL')),
            quarantined=len(quarantines),training_admitted=False,human_visual_review_pending=True,
            model_forwards=0,optimizer_updates=0,seconds=time.time()-started,
            files={name:bound(out/name) for name in ('manifest_full.json','manifest_tight.json','statistics.json','quarantine.json','row_results.jsonl','protocol.json')}))
    except Exception:
        write(out/'failure.json',dict(status='failed',traceback=traceback.format_exc(),training_admitted=False));raise

if __name__=='__main__':main()
