"""One bounded type-targeted supplement after the documented240-group pilot.

Uses untried existing v14 source groups, not new torn fragments. Never retries
the same rejected slot, relaxes damage limits, or starts full generation.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import traceback
import numpy as np
from . import run


def footprint_check(record):
    with np.load(record['proof_path']) as z:
        for side in 'ab':
            if 'primary_'+side+'_major' not in z:continue
            p=z['primary_'+side+'_points'];m=np.unpackbits(z['packed_trim_'+side],axis=1).astype(bool)
            eligible=z['primary_'+side+'_eligible']&m[tuple(np.rint(p).astype(int).T)]
            edge=z['primary_'+side+'_edge'];active=(z['primary_'+side+'_major']>0)&eligible
            intact=eligible&np.roll(eligible,-1);denom=float(edge[intact].sum())
            fraction=float(edge[intact&active&np.roll(active,-1)].sum()/denom) if denom else 0.
            if fraction>.5+1e-6:return False
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--out',required=True)
    p.add_argument('--previous',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();base=Path(a.baseline).resolve();out=Path(a.out).resolve();prev=Path(a.previous).resolve()
    if out.exists() or a.workers>4:raise ValueError('new bounded CPU output only')
    prior=run.read(prev/'generation_audit.json');prior_protocol=run.read(prev/'protocol.json')
    if len(prior['attempts'])!=240 or not (prev/'failure.json').exists():raise ValueError('expected documented240-group pilot failure')
    if prior_protocol['baseline_manifest_sha256']!=run.digest(base/'train_s7b_24k.json'):raise ValueError('baseline changed')
    out.mkdir(parents=True);(out/'groups').mkdir();(out/'samples').mkdir();start=time.time()
    tried={r['slot'] for r in prior['attempts']};records=[];counts=Counter();reused=[];excluded=[];attempts=[]
    for path in sorted((prev/'groups').glob('*.json')):
        group=run.read(path)
        if all(footprint_check(r) for r in group['records']):
            got=run.reused_group(prev,out,group['slot']);records+=got['records'];reused.append(group['slot'])
            for r in got['records']:counts.update(r['groups'])
        else:excluded.append(dict(slot=group['slot'],reason='major footprint exceeds50% of remaining eligible arc'))
    entries=run.read(base/'train_s7b_24k.json')['entries']
    metrics={r['pair_id']:r for r in (json.loads(line) for line in (base/'pair_metrics.jsonl').open())}
    tags={}
    for entry in entries:
        slot=int(Path(entry['artifact_path']).stem.split('_')[0])
        tags.setdefault(slot,set()).update(run.tags(entry,metrics[entry['pair_id']]))
        tags[slot].add('trim_one' if slot%2 else 'trim_both')
    order=run.planned_slots(base,12000)
    protocol=dict(prior_protocol,schema='aggressive-v15-paired-review/3',
        maximum_total_groups=400,maximum_additional_groups=160,previous=str(prev),
        previous_failure_sha256=run.digest(prev/'failure.json'),reused_committed_slots=reused,
        excluded_committed_groups=excluded,planned_slots=[],
        repair='condition one/both-end trim on either fragment; keep K, connectivity,25%area,original target inheritance; explicit remaining-major-footprint<=50%',
        selection='deterministic next untried source group belonging to currently incomplete review type; not visual selection',
        source_sha256={p.name:run.digest(p) for p in Path(__file__).parent.glob('*.py')})
    run.save_json(out/'protocol.json',protocol)
    try:
        with ProcessPoolExecutor(a.workers,initializer=run.initialize,initargs=(dict(baseline=str(base),out=str(out)),)) as pool:
            while not all(counts[k]>=10 for k in run.GROUP_NAMES) and len(attempts)<160:
                missing={k for k in run.GROUP_NAMES if counts[k]<10}
                batch=[s for s in order if s not in tried and tags[s]&missing][:min(a.workers,160-len(attempts))]
                if not batch:break
                tried.update(batch);protocol['planned_slots']+=batch;run.save_json(out/'protocol.json',protocol)
                for result in pool.map(run.process,batch):
                    attempts.append({k:v for k,v in result.items() if k!='records'})
                    run.save_json(out/'attempts'/f'{result["slot"]:05d}.json',attempts[-1])
                    if result['status']=='passed':
                        records+=result['records']
                        for r in result['records']:counts.update(r['groups'])
                    run.save_json(out/'status.json',dict(status='bounded_supplement',pairs=len(records),
                        additional_groups_processed=len(attempts),groups=dict(counts),elapsed_seconds=time.time()-start))
        missing={k:10-counts[k] for k in run.GROUP_NAMES if counts[k]<10}
        run.save_json(out/'generation_audit.json',dict(previous_attempts=prior['attempts'],attempts=attempts,
            reused=reused,excluded=excluded,counts=dict(counts),missing=missing))
        run.save_json(out/'manifest.json',dict(artifact_root=str(out),entries=records))
        if missing:raise ValueError('400-group bound reached without type coverage:'+repr(missing))
        complete=dict(status='generated_pending_pixel_audit_and_review',pairs=len(records),
            accepted_groups=len(records)//2,previous_processed_groups=240,processed_groups=len(attempts),
            elapsed_seconds=time.time()-start,missing={},training_started=False)
        run.save_json(out/'generation_complete.json',complete);run.save_json(out/'status.json',complete)
        print(json.dumps(complete))
    except BaseException as e:
        run.save_json(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),
            pairs=len(records),automatic_retry=False,training_started=False));raise

if __name__=='__main__':main()
