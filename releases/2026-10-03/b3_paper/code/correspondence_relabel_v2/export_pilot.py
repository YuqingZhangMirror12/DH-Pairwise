"""Deterministic TRAIN-only evidence export, writes tar to stdout, remote read-only."""
from collections import Counter
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from catalog import source_catalog, expand, sidecars, sha

def order(rows, salt):
    return sorted(rows, key=lambda r:hashlib.sha256((salt+r['admission']['pair_id']).encode()).hexdigest())

def main():
    rows, bindings = source_catalog()
    chosen = []
    for stage in sorted({r['admission']['stage'] for r in rows}):
        pool = [r for r in rows if r['admission']['stage']==stage and r['admission']['label']]
        if stage=='v17_filtered': pool = [r for r in pool if not r['record'].get('v14_fallback')]
        taken = {}
        for recipe in sorted({r['record']['recipe'] for r in pool}):
            for r in order([r for r in pool if r['record']['recipe']==recipe], 'recipe-v1')[:10]:
                taken[r['admission']['pair_id']]=r
        for r in order(pool, 'fill-v1'):
            if len(taken)>=120: break
            taken[r['admission']['pair_id']]=r
        chosen.extend(taken.values())
        if stage=='v17_filtered':
            chosen.extend(order([r for r in rows if r['admission']['stage']==stage and r['admission']['label']
                                 and r['record'].get('v14_fallback')], 'fallback-v1')[:22])
        chosen.extend(order([r for r in rows if r['admission']['stage']==stage and not r['admission']['label']], 'negative-v1')[:3])
    cases = [expand(r) for r in chosen]
    blobs = {}
    for case in cases:
        pid=case['admission']['pair_id']; case['files']={}
        for key, (path, expected) in sidecars(case).items():
            data=path.read_bytes(); digest=hashlib.sha256(data).hexdigest()
            if expected and digest!=expected: raise ValueError('Source SHA mismatch: '+str(path))
            rel='cases/'+pid+'/'+key+'.npz'
            blobs[rel]=data
            case['files'][key]=dict(path=rel, remote=str(path), sha256=digest, registered_sha256=expected)
    # Base-label audit uses declared identities, not a closest-looking fragment search.
    unique={}
    for r in order([r for r in rows if r['admission']['label'] and r['admission']['stage']!='straight_seam'], 'base-v1'):
        e=r['record']; sr=e['source_row']; key=e['source_root']+'::'+e['source_pair_id']
        if key not in unique and 'correspondence_path' in sr: unique[key]=e
        if len(unique)==146: break
    stats=[]
    for key,e in unique.items():
        root=Path(e['source_root']); sr=e['source_row']
        files={k:root/sr['fragment_'+k]['contour_path'] for k in 'ab'}
        files['correspondence']=root/sr['correspondence_path']
        with np.load(files['a']) as z: pa=z['points_rc'].astype(float); va=z['valid']
        with np.load(files['b']) as z: pb=z['points_rc'].astype(float); vb=z['valid']
        with np.load(files['correspondence']) as z: pairs=z['correspondence_indices']; t=z['translation_a_to_b_rc']
        valid_ids=np.flatnonzero(vb)
        d,j=cKDTree(pb[vb]).query(pa+t); j=valid_ids[j]
        aid=np.flatnonzero(va); back=aid[cKDTree((pa+t)[va]).query(pb)[1]]
        touch=va&(d<=3); matched=np.zeros(len(pa),bool); matched[pairs[:,0]]=True
        dist=np.linalg.norm(pa[pairs[:,0]]+t-pb[pairs[:,1]],axis=1)
        stats.append(dict(source_identity=key, files={k:dict(path=str(p),sha256=sha(p)) for k,p in files.items()},
            touching=int(touch.sum()), touching_matched=int((touch&matched).sum()),
            touching_unmatched=int((touch&~matched).sum()),
            unmatched_mnn_possible=int((touch&~matched&(back[j]==np.arange(len(pa)))).sum()),
            match_distances=dist.tolist()))
    manifest=dict(schema='correspondence-relabel-pilot-evidence/1', training_admitted=False,
                  source_bindings=bindings, cases=cases, base_audit=stats,
                  selection='120 positives/stage, >=10 per recipe when available, plus 22 v14-fallback controls, 3 negatives/stage. SHA256 ordering, no model results.',
                  population=dict(Counter(r['admission']['stage'] for r in rows)))
    blobs['pilot.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as tar:
        for name,data in blobs.items():
            info=tarfile.TarInfo(name); info.size=len(data); info.mtime=0
            tar.addfile(info,io.BytesIO(data))
    print(json.dumps(dict(cases=len(cases),base_pairs=len(stats),bytes=sum(map(len,blobs.values())))),file=sys.stderr)

if __name__=='__main__': main()
