"""Single-GPU, single-launch frozen inference. Never retries a failed output."""
import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from .io import read,sha,checked,ref,save,loaded_sources
from .core import combine,annotate,rotate_vectors
from .inference import prediction_without_timing,PREFIX
from .runtime import load,Population


def verify_sources(protocol):
    here=Path(__file__).parent
    if {p.name:sha(p) for p in here.glob('*.py')}!=protocol['source_python']:raise ValueError('new inference source changed')
    inventory=checked(protocol['source_binding'])
    for p,h in inventory.items():
        if sha(Path(protocol['runtime'])/p)!=h:raise ValueError('frozen runtime source changed: '+p)


def verify_smoke(engine,inputs,ids,angles,rows,population,protocol,out):
    repeat={a:engine.predict_view(inputs,ids,a) for a in angles}
    for a in angles:
        if [prediction_without_timing(r) for r in rows[a]]!=[prediction_without_timing(r) for r in repeat[a]]:
            raise ValueError('same-view inference is not bitwise identical at '+str(a))
    targets=population.targets(protocol,ids);checked_poses=0;sensitivity=[]
    for i,pid in enumerate(ids):
        target=targets[pid]['target'];v={a:rows[a][i] for a in angles}
        if target is not None:
            for a in angles:
                for c in v[a]['candidates']:
                    original=float(np.linalg.norm(np.asarray(c['translation'])-target))
                    rotated=float(np.linalg.norm(np.asarray(c['translation_view'])-rotate_vectors(target,a)))
                    if abs(original-rotated)>1e-9 or (original<=20)!=(rotated<=20):raise ValueError('GT frame restoration differs')
                    checked_poses+=1
        base=annotate(combine({0:v[0]},head='q'),targets[pid]['label'],target)
        entry=dict(pair_id=pid,gt_known=target is not None,views={})
        for a in angles:
            # combine deliberately requires identity first; single-view
            # rotation sensitivity uses the native winning candidate directly.
            cs=v[a]['candidates'];w=max(cs,key=lambda c:(c['q_sum'],-c['index'])) if cs else None
            entry['views'][a]=dict(top_q=None if w is None else w['q_sum'],pose=None if w is None else w['translation'],
                layout=None if target is None else w is not None and np.linalg.norm(np.asarray(w['translation'])-target)<=20,
                coverage=None if target is None else any(np.linalg.norm(np.asarray(c['translation'])-target)<=20 for c in cs))
            entry['views'][a]['layout']=None if target is None else bool(entry['views'][a]['layout'])
        sensitivity.append(entry)
    if not checked_poses:raise ValueError('smoke batch has no evaluable GT candidates')
    changed=sum(any(e['views'][a]['top_q']!=e['views'][0]['top_q'] or e['views'][a]['pose']!=e['views'][0]['pose'] for a in angles if a) for e in sensitivity)
    if len(angles)>1 and not changed:raise ValueError('direction sensitivity not reproduced; inspect before full sweep')
    save(out/'smoke.json',dict(status='passed',pair_ids=ids,repeat_bitwise=True,native_identity_parity=True,
        gt_inverse_pose_checks=checked_poses,direction_sensitive_pairs=changed,angles=angles,results=sensitivity))


def run(root,matcher_key,phase):
    import torch
    protocol_path=root/'protocol.json';protocol=read(protocol_path);verify_sources(protocol)
    angles=protocol['angles'];heads=['patch_sim','stats_sim','patch_real','stats_real']
    default=None
    if phase=='test':
        default=read(root/'default.json')
        checked(default['development_report'])
        if default['protocol_sha256']!=sha(protocol_path) or matcher_key!=default['setting']['matcher']:raise ValueError('TEST requires frozen development choice')
        angles=default['setting']['angles'];heads=[default['setting']['head']]
    out=root/phase/matcher_key
    if out.exists():raise ValueError('output exists; no implicit retry')
    out.mkdir(parents=True)
    save(out/'launch.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        started_unix=time.time(),argv=sys.argv,protocol=ref(protocol_path),phase=phase,matcher=matcher_key,heads=heads,
        angles=angles,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),default=None if default is None else ref(root/'default.json')))
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    engine=load(protocol,matcher_key,heads);before=engine.state_hashes();receipts={};started=time.time()
    names=['dunhuang_cv','turufan','sim_cal'] if phase=='development' else ['dunhuang_cv','turufan']
    for name in names:
        population=Population(protocol,name,phase);predpath=out/(name+'.jsonl');seen=[];smoke=False
        with predpath.open('x') as f:
            for start in range(0,len(population.pairs),protocol['physical_microbatch']):
                ids,inputs=population.batch(start,protocol['physical_microbatch'])
                check=phase=='development' and name=='dunhuang_cv' and start==0
                rows={a:engine.predict_view(inputs,ids,a,verify_native=check and a==0) for a in angles}
                for i,pid in enumerate(ids):f.write(json.dumps(dict(pair_id=pid,views={a:rows[a][i] for a in angles}),allow_nan=False)+'\n')
                f.flush();os.fsync(f.fileno());seen.extend(ids)
                if check:
                    verify_smoke(engine,inputs,ids,angles,rows,population,protocol,out);smoke=True
                    deadline=time.monotonic()+1200
                    while not (root/'smoke_admitted.json').exists():
                        if (root/'controller_failure.json').exists() or time.monotonic()>deadline:raise RuntimeError('smoke admission not granted')
                        time.sleep(2)
                    admission=read(root/'smoke_admitted.json')
                    if admission['smokes'][matcher_key]!=ref(out/'smoke.json'):raise ValueError('smoke admission changed')
                save(out/'status.json',dict(state='running_inference',dataset=name,processed=len(seen),total=len(population.pairs),
                    pair_views=len(seen)*len(angles),elapsed_seconds=time.time()-started,last_update_unix=time.time()),replace=True)
                if start%80==0:print(matcher_key,phase,name,len(seen),'/',len(population.pairs),flush=True)
        if seen!=[p['pair_id'] for p in population.pairs]:raise ValueError('prediction row identity/order differs')
        target_path=out/(name+'_targets.json');save(target_path,population.targets(protocol))
        receipts[name]=dict(predictions=ref(predpath),targets=ref(target_path),pairs=len(seen),angles=angles)
        save(out/(name+'_complete.json'),receipts[name])
    after=engine.state_hashes()
    if before!=after:raise ValueError('inference changed model state')
    verify_sources(protocol)
    save(out/'complete.json',dict(status='complete',protocol=ref(protocol_path),matcher=matcher_key,phase=phase,heads=heads,
        angles=angles,datasets=receipts,models_before=before,models_after=after,elapsed_seconds=time.time()-started,
        native_loaded_sources=loaded_sources(protocol['runtime'],PREFIX)))
    save(out/'status.json',dict(state='complete',elapsed_seconds=time.time()-started,last_update_unix=time.time()),replace=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--matcher',choices=['matcher_sim','matcher_endpoint'],required=True)
    p.add_argument('--phase',choices=['development','test'],required=True);a=p.parse_args()
    try:run(a.root,a.matcher,a.phase)
    except BaseException as e:
        out=a.root/a.phase/a.matcher
        save(out/'failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),time_unix=time.time()))
        raise


if __name__=='__main__':main()
