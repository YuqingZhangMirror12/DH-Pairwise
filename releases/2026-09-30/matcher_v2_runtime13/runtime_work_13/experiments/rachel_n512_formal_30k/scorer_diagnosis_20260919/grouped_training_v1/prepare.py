"""One positive + three source-disjoint negatives sharing the exact anchor.

Two shape-matched negatives and one random negative, all drawn within the
same S7 recipe as the positive. No Scorer scores or real-data labels are read.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from support import *

def family(s):
    # Dataset's original source identity, not arbitrary pair/fragment IDs.
    return str(s)

def build_groups(sources, root, split):
    rng=np.random.default_rng(SEED+(0 if split=='train' else 1)); groups=[]; pairs=[]
    index={}
    for i,s in enumerate(sources): index.setdefault(s['recipe'],[]).append(i)
    for g,s in enumerate(sources):
        # Alternate anchors: include both orientations without rotating/resizing.
        a='a' if g%2==0 else 'b'; b='b' if a=='a' else 'a'
        pool=np.array([i for i in index[s['recipe']] if family(sources[i]['source'])!=family(s['source'])])
        if len(pool)<10: raise ValueError('too few independent negative sources')
        feat=np.array([sources[i]['shape_'+b] for i in pool])
        target=np.array(s['shape_'+b]); distance=((feat-target)**2).sum(1)
        near=pool[np.argsort(distance,kind='stable')[:min(80,len(pool))]]
        selected=[];used={s['source']}
        for mode in ('shape_matched','shape_matched','random'):
            candidates=near if mode=='shape_matched' else pool
            candidates=np.array([i for i in candidates if sources[i]['source'] not in used])
            if not len(candidates): candidates=np.array([i for i in pool if sources[i]['source'] not in used])
            i=int(rng.choice(candidates)); selected.append((i,mode));used.add(sources[i]['source'])
        members=[(g,'positive')]+selected
        ids=[]
        for partner,kind in members:
            item=dict(ordinal=len(pairs),pair_id=f'{split}-anchor-{g:05d}-{kind}-{partner:05d}',
                      group_id=g,anchor_source=s['source'],partner_source=sources[partner]['source'],
                      anchor_path=s['path'],anchor_side=a,partner_path=sources[partner]['path'],partner_side=b,
                      label=float(kind=='positive'),negative_kind=kind,recipe=s['recipe'],
                      original_positive_pair_id=s['pair_id'])
            if not item['label'] and item['anchor_source']==item['partner_source']: raise ValueError('false negative source')
            ids.append(len(pairs));pairs.append(item)
        rng.shuffle(ids);groups.append(ids)
    save(root/'groups'/f'{split}_pairs.json',pairs)
    np.save(root/'groups'/f'{split}_groups.npy',np.asarray(groups,dtype=np.int64))
    return dict(groups=len(groups),pairs=len(pairs),positive=len(groups),negative=3*len(groups),
                original_sources=len({s['source'] for s in sources}),recipes=dict(Counter(s['recipe'] for s in sources)))

def make(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);(root/'groups').mkdir()
    raw=read(TRAIN_MANIFEST); entries=[e for e in raw['entries'] if e['label']]
    assert len(entries)==12000
    def train_source(e):
        shape={}
        # Actual damaged mask statistics, not the undamaged source metadata.
        with np.load(Path(raw['artifact_root'])/e['artifact_path'],allow_pickle=False) as z:
            for side in 'ab':
                packed=z['mask_'+side+'_packed']; h,w=map(int,z['mask_'+side+'_shape'][-2:])
                mask=np.unpackbits(packed,axis=-1,count=w).reshape(h,w).astype(bool)
                rr,cc=np.nonzero(mask)
                shape['shape_'+side]=[float(np.log(max(1,len(rr)))),float(np.log((np.ptp(cc)+1)/(np.ptp(rr)+1)))]
        return dict(path=str(Path(raw['artifact_root'])/e['artifact_path']),recipe=e['s7_recipe'],
                    pair_id=e['pair_id'],source=e['source_row']['fragment_a']['split_unit_id'],**shape)
    with ThreadPoolExecutor(max_workers=8) as pool: training=list(pool.map(train_source,entries))
    # Existing independent clean simulation VAL, not a split of TRAIN groups.
    origin=cache.source_checkpoint()['resume_identity']
    _,val,cap,_=cache.old.make_populations(SimpleNamespace(sampling='original512',
        train_materialized_manifest=str(TRAIN_MANIFEST),dataset=str(Path(origin['populations']['val']['manifest']).parents[1])))
    rows=[json.loads(s) for s in Path(origin['populations']['val']['manifest']).read_text().splitlines() if s]
    lookup={r['pair_id']:r for r in rows}; positive_ids=[i for i,r in enumerate(rows) if r['label']]
    folder=root/'groups'/'val_sources';folder.mkdir();validation=[]
    loader=cache.old.make_ablation_loader(val,positive_ids,batch_size=1,num_workers=0,seed=cache.old.SEED,contour_cap=cap)
    for ordinal,wrapped in enumerate(loader):
        batch=getattr(wrapped,'batch',wrapped); pid=str(batch.pair_ids[0]); row=lookup[pid]; arrays={}; shapes={}
        for side in 'ab':
            mask=np.asarray(getattr(batch,'mask_'+side)[0],dtype=bool)
            arrays['mask_'+side+'_packed']=np.packbits(mask,axis=-1)
            arrays['mask_'+side+'_shape']=np.array(mask.shape,dtype=np.int64)
            arrays['points_rc_'+side]=np.asarray(getattr(batch,'points_rc_'+side)[0],np.float32)
            arrays['contour_valid_'+side]=np.asarray(getattr(batch,'contour_valid_'+side)[0],bool)
            rr,cc=np.nonzero(mask[0]);shapes['shape_'+side]=[float(np.log(max(1,len(rr)))),float(np.log((np.ptp(cc)+1)/(np.ptp(rr)+1)))]
        path=folder/f'{ordinal:05d}.npz';np.savez_compressed(path,**arrays)
        validation.append(dict(path=str(path),recipe='clean_val',pair_id=pid,
                               source=row['fragment_a']['split_unit_id'],**shapes))
    assert len(validation)==1500
    overlap={s['source'] for s in training}&{s['source'] for s in validation}
    if overlap: raise ValueError('TRAIN/VAL source overlap: '+str(overlap))
    stats={s:build_groups(src,root,s) for s,src in (('train',training),('val',validation))}
    for split in ('train','val'):
        folder=root/'cache'/split;folder.mkdir(parents=True)
        pairs=read(root/'groups'/f'{split}_pairs.json');n=len(pairs)
        for key,(dtype,tail) in cache.ARRAYS.items():
            out=np.lib.format.open_memmap(folder/(key+'.npy'),mode='w+',dtype=dtype,shape=(n,*tail))
            if key=='ready':out[:]=False
            if key=='label':out[:]=[p['label'] for p in pairs]
            out.flush();del out
        save(folder/'protocol.json',dict(schema=SCHEMA,matcher_sha=cache.SOURCE_SHA,status='allocated',pairs=n))
    save(root/'plan.json',dict(schema=SCHEMA,status='prepared',train_manifest=str(TRAIN_MANIFEST),
        matcher_sha=cache.SOURCE_SHA,source_code=str(SOURCE),stats=stats,source_overlap=0,
        group_size=4,epochs=EPOCHS,microbatch_pairs=GROUP_SIZE*GROUP_BATCH,effective_batch_pairs=GROUP_SIZE*GROUP_BATCH,
        groups_per_update=GROUP_BATCH,arms=ARMS,gpus=GPUS,
        negative_policy='same-recipe donor masks: two area/aspect-near, one random, three distinct other source images',
        caveat='not newly synthesized damage; new recombinations of existing S7 masks; no within-image negatives in this first controlled comparison',
        loss='GS0 class-balanced PairBCE; GS1 same BCE + 0.5 group softmax CE, temperature1 on logits',
        selection='fixed C16; C8 retained; no real data for checkpoint selection',
        inference='independent sigmoid(logit); no mandatory winner; group softmax used only in training',
        thresholds='fixed0.30 primary; SIMVAL and separate source-fivefold bounded0.20–0.80 reported additionally'))
    print(json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));make(p.parse_args().root)
