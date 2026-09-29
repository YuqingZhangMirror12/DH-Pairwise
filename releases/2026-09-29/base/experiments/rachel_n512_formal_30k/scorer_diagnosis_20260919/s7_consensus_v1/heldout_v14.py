"""Source-isolated existing Rachel pairs with approved v14 damage, never new tears.

User-authorized cross-manuscript negatives fill the unavailable within-parent
quota. Original SIMTEST families never enter CAL/SELECT. No RGB, real cases,
model scores, TRAIN donor profiles, or freshly generated base fragments are used.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from ..seam_context_v3.prepare import family, sources, rows, row_dataset, read, save, sha
from ..s7_balanced_v2.layered_geometry import _allocate

ROOT = Path('/root/autodl-tmp')
RELEASE = ROOT/'dataset_rachel_pairwise_n512_v1'
S7 = ROOT/'rachel_score_design_20260913_001/s6_s7_20260915/preparation/data/train_s7_24k.json'
FIELDS = ('fragment_token', 'model_mask_path', 'contour_path', 'split_unit_id',
          'foreground_area', 'bbox_aspect_ratio', 'parent_group_id', 'generator')
WORK = {}


def split_sources(val_sources, test_sources, forbidden, positive_counts, seed):
    val = set(val_sources)-set(forbidden)
    test = set(test_sources)-set(forbidden)-set(val_sources)
    if len(val)<4 or len(test)<2:
        raise ValueError('not enough original heldout source families')
    # Deterministic balance by source inventory, not model performance.
    tie = lambda name: hashlib.sha256(f'{seed}:{name}'.encode()).digest()
    ordered = sorted(val, key=lambda s:(-positive_counts.get(s, 0), tie(s)))
    cal, select = set(), set()
    for i, name in enumerate(ordered):
        (cal if i%2==0 else select).add(name)
    return dict(cal=cal, select=select, test=test)


def schedule(n, profile, seed):
    """Largest-remainder integer quotas; no hidden rounding or extra views."""
    rng = np.random.default_rng(seed)
    macro = {'clean':15, 'partial':25, 'mild':15, 'wave':15, 'local':15, 'gaps':15}
    counts = dict(zip(macro, map(int, _allocate(n, list(macro.values())))))
    recipes = []
    for key, count in counts.items():
        sub = ({'wave':['wave','wave_weak'],
                'local':['local_abrupt','local_abrupt_weak','local_gradual','local_gradual_weak'],
                'gaps':['gaps','gaps_weak']}.get(key, [key]))
        for name, amount in zip(sub, _allocate(count, [1]*len(sub))):
            recipes.extend([name]*int(amount))
    rng.shuffle(recipes)
    modes = [None]*n
    ids = np.flatnonzero(np.asarray(recipes)=='partial')
    for i, index in enumerate(rng.permutation(ids)):
        modes[int(index)] = 'end' if i<(len(ids)+1)//2 else 'middle'
    mirrors = np.repeat([0,1,2], _allocate(n, [85,7.5,7.5])).tolist()
    rng.shuffle(mirrors)
    return dict(recipes=recipes, partial=[r=='partial' for r in recipes],
                partial_modes=modes, bins=['native']*n, mirrors=mirrors,
                integer_macro_counts=counts,
                rounding='largest remainder; subgroup deviations below one pair; actual counts reported')


def cross_negatives(fragments, selected_sources, count, seed, release_split):
    rng = np.random.default_rng(seed)
    pool = [f for f in fragments.values() if family(f['split_unit_id']) in selected_sources]
    result = [];seen = set()
    for _ in range(max(10000, count*500)):
        if len(result)==count:
            return result
        a, b = (pool[int(i)] for i in rng.integers(len(pool),size=2))
        if family(a['split_unit_id'])==family(b['split_unit_id']):
            continue
        key = tuple(sorted((a['fragment_token'],b['fragment_token'])))
        if key in seen:
            continue
        ratios = {k:max(float(a[k]),float(b[k]))/min(float(a[k]),float(b[k]))
                  for k in ('foreground_area','bbox_aspect_ratio')}
        if max(ratios.values())>2.:
            continue
        seen.add(key)
        pid = 'v14-heldout-cross-'+hashlib.sha256('|'.join(key).encode()).hexdigest()[:24]
        result.append(dict(pair_id=pid,label=False,split=release_split,
            fragment_a=a,fragment_b=b,correspondence_path=None,
            negative_origin='cross_folder_scale_matched',
            label_origin='distinct_canonical_manuscript_families',scale_match=ratios))
    raise ValueError('unique same-fold scale-matched cross negative pool exhausted')


def eligible_sample(release, row):
    from scipy import ndimage
    try:
        sample = row_dataset(release, [row], row['split'])[0]
        valid = all(ndimage.label(getattr(sample,'mask_'+s)[0],np.ones((3,3),bool))[1]==1
                    for s in 'ab')
        if sample.label:
            valid = valid and int((sample.target_a>=0).sum())>=4
        return valid
    except (ValueError, OSError):
        return False


def make_bank(fragments, families, split, release, out, seed):
    """Profiles from this fold's existing masks only; not a TRAIN OutlineBank."""
    from PIL import Image
    from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import _dense_external_contour
    from staging.pairwise_v0_2.pairwise_data.rachel_curve_cut import profile_from_arc
    rng = np.random.default_rng(seed);profiles=[];arcs=[]
    by_family = defaultdict(list)
    for f in fragments.values():
        src = family(f['split_unit_id'])
        if src in families:
            by_family[src].append(f)
    for src in sorted(by_family):
        accepted=0;pool=by_family[src]
        for ix in rng.permutation(len(pool))[:16]:
            f=pool[int(ix)]
            with Image.open(release/f['model_mask_path']) as image:
                dense=_dense_external_contour(np.asarray(image)>127)
            for _ in range(40):
                if len(dense)<194:
                    break
                start=int(rng.integers(len(dense)))
                length=int(rng.integers(96,min(321,len(dense)//2+1)))
                found=profile_from_arc(dense[(start+np.arange(length))%len(dense)])
                if found is None:
                    continue
                profile, geometry=found;profiles.append(profile)
                arcs.append(dict(split=split,lineage=f['split_unit_id'],family=src,
                    fragment_token=f['fragment_token'],model_mask_path=f['model_mask_path'],
                    contour_start=start,contour_point_count=length,**geometry))
                accepted+=1
                if accepted>=8:
                    break
            if accepted>=8:
                break
    if not profiles:
        raise ValueError('no existing contour-derived donor profiles in '+split)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'profiles.npz',profiles=np.stack(profiles))
    save(out/'bank.json',dict(schema='s7-heldout-outline-bank/1',split=split,
        arcs=arcs,source_families=sorted({r['family'] for r in arcs}),
        new_base_fragments_generated=False,train_profiles_used=False))
    return dict(path=str(out),profile_count=len(arcs),source_count=len({r['family'] for r in arcs}),
                metadata_sha256=sha(out/'bank.json'),profiles_sha256=sha(out/'profiles.npz'))


def build_plan(out, profile_path, seed):
    import cv2, torch
    cv2.setNumThreads(1);torch.set_num_threads(1)
    out=Path(out).resolve()
    if (out/'sources.json').exists():
        raise ValueError('preserve existing source plan')
    original_train=read(S7)
    forbidden=set().union(*(sources(e['source_row']) for e in original_train['entries']))
    val=set().union(*(sources(r) for r in rows(RELEASE/'pairs/val.jsonl')))
    test=set().union(*(sources(r) for r in rows(RELEASE/'pairs/test.jsonl')))
    available=(val|test)-forbidden
    fragments={r['fragment_token']:{k:r[k] for k in FIELDS}
               for r in rows(RELEASE/'manifests/fragments.jsonl') if family(r['split_unit_id']) in available}
    within=[];positive_counts=Counter()
    for r in rows(RELEASE/'manifests/within_candidates.jsonl'):
        if not r.get('main_training_eligible'):
            continue
        a,b=r['fragment_a_token'],r['fragment_b_token']
        if a not in fragments or b not in fragments:
            continue
        within.append(dict(pair_id=r['pair_id'],label=bool(r['label']),fragment_a=fragments[a],
            fragment_b=fragments[b],correspondence_path=r['correspondence_path'] if r['label'] else None,
            label_origin=r['label_origin'],negative_origin=r.get('negative_origin')))
        if r['label']:
            positive_counts[family(fragments[a]['split_unit_id'])]+=1
    assignment=split_sources(val,test,forbidden,positive_counts,seed)
    approved=read(profile_path);profile=dict(approved)
    profile.update(name=approved['name']+' — original heldout scale',
        preserve_original_scale_for_heldout=True,enforce_length_quotas=False,
        enforce_partial_length_quotas=False,conditional_area_deciles_px2=None,
        heldout_policy='Original S7 scale; v14 damage unchanged. No real-area or seam-length quota imposed on validation.',
        new_training_authorized=False)
    save(out/'profile.json',profile)
    result=dict(schema='s7-v14-heldout-source-plan/1',seed=seed,dataset_root=str(RELEASE),
        historical_train_manifest=str(S7),historical_train_sha256=sha(S7),
        forbidden_sources=sorted(forbidden),approved_profile_sha256=sha(profile_path),
        profile_path=str(out/'profile.json'),profile_sha256=sha(out/'profile.json'),splits={},
        dunhuang_source_review=dict(found=True,groups=5000,
            path='/root/autodl-tmp/dunhuang_pairwise_v02/incoming/pairwise_mask_subset.zip',
            used=False,reason='No reliable manuscript lineage map; numeric group IDs are not source isolation.'),
        fallback_authorized=True,no_new_base_tearing=True,no_rgb_or_real_data=True,
        family_caveat='Known recto/verso/seite/detail/total aliases, not a complete external catalogue.',
        test_families_excluded_due_to_val_overlap=sorted((test-forbidden)&val))
    for si,(split,families) in enumerate(assignment.items()):
        total=3000 if split=='test' else 1500;n=total//2;release_split='test' if split=='test' else 'val'
        base=[dict(r,split=release_split) for r in within if sources(r)<=families]
        pos=[];neg=[];rejected=[]
        for r in base:
            if eligible_sample(RELEASE,r):
                (pos if r['label'] else neg).append(r)
            else:
                rejected.append(r['pair_id'])
        if not pos:
            raise ValueError('no valid positive base pairs for '+split)
        rng=np.random.default_rng(seed+si)
        same=[neg[int(i)] for i in rng.permutation(len(neg))[:n//2]]
        # Never repeat a scarce negative Pair hundreds of times to claim50%.
        cross=cross_negatives(fragments,families,n-len(same),seed+10+si,release_split)
        cross=[r for r in cross if eligible_sample(RELEASE,r)]
        if len(same)+len(cross)!=n:
            raise ValueError('cross source geometry failed; explicitly rebuild plan, never silently drop')
        chosen=same+cross;rng.shuffle(chosen)
        positive=[pos[int(i)] for i in np.resize(rng.permutation(len(pos)),n)]
        source_entries=[dict(pair_id=r['pair_id'],source_stratum='native_positive') for r in positive]
        negatives=[dict(mode='native',pair_id=r['pair_id'],row=r,
            source_stratum='native_negative',anchor_group_id=None,
            kind='same_parent_nonadjacent' if r['negative_origin']=='same_folder_hard' else 'cross_manuscript') for r in chosen]
        # Existing original labels define same-parent nonadjacency, never lack of a model match.
        for e in negatives:
            r=e['row'];a,b=r['fragment_a'],r['fragment_b']
            if e['kind']=='same_parent_nonadjacent':
                assert a['parent_group_id']==b['parent_group_id'] and len(sources(r))==1
            else:
                assert len(sources(r))==2
        spec=dict(source_families=sorted(families),release_split=release_split,pairs=total,
            positive=source_entries,positive_pool=pos,negative=negatives,
            negative_counts=dict(Counter(e['kind'] for e in negatives)),
            requested_negative_counts=dict(same_parent_nonadjacent=n//2,cross_manuscript=n-n//2),
            fallback_cross_count=max(0,n//2-len(same)),
            base_positive_count=len(pos),base_within_negative_count=len(neg),
            source_geometry_exclusions=rejected,
            schedule=schedule(n,profile,seed+100+si))
        spec['donor_bank']=make_bank(fragments,families,split,RELEASE,out/'banks'/split,seed+200+si)
        result['splits'][split]=spec
    save(out/'sources.json',result)
    print(json.dumps({s:{k:v[k] for k in ('pairs','base_positive_count','base_within_negative_count','negative_counts','fallback_cross_count')}
                      for s,v in result['splits'].items()}),flush=True)


def initialize(options):
    import cv2,torch
    from ..s7_balanced_v2 import materialize as material
    cv2.setNumThreads(1);torch.set_num_threads(1)
    plan=read(options['sources']);spec=plan['splits'][options['split']]
    root=plan['dataset_root'];profile=read(plan['profile_path'])
    if sha(plan['profile_path'])!=plan['profile_sha256']:
        raise ValueError('heldout profile changed')
    bank_path=Path(spec['donor_bank']['path'])
    if (sha(bank_path/'bank.json')!=spec['donor_bank']['metadata_sha256'] or
            sha(bank_path/'profiles.npz')!=spec['donor_bank']['profiles_sha256']):
        raise ValueError('heldout donor bank changed after source registration')
    metadata=read(bank_path/'bank.json')
    if metadata['split']!=options['split'] or any(a['split']!=options['split'] for a in metadata['arcs']):
        raise ValueError('cross-fold donor bank')
    if set(metadata['source_families'])-set(spec['source_families']):
        raise ValueError('donor source leak')
    with np.load(bank_path/'profiles.npz',allow_pickle=False) as z:
        profiles=z['profiles'].copy()
    pool=spec['positive_pool']
    positives={r['pair_id']:dict(pair_id=r['pair_id'],label=True,source_root=root,
        source_row=r,source_stratum='native_positive') for r in pool}
    material.STATE.clear();material.clean_positive.cache_clear()
    material.STATE.update(options=dict(options,groups=spec['pairs']//2),
        plan=dict(dataset_root=root),profile=profile,positive=spec['positive'],
        negative_plan=spec['negative'],pos_entries=positives,
        clean=row_dataset(root,pool,spec['release_split']),
        clean_lookup={r['pair_id']:i for i,r in enumerate(pool)},
        gen5_lookup={},negative=row_dataset(root,[e['row'] for e in spec['negative']],spec['release_split']),
        negative_lookup={e['pair_id']:i for i,e in enumerate(spec['negative'])},
        buckets={'native_positive':[dict(pair_id=r['pair_id'],source_stratum='native_positive') for r in pool]},
        bank=SimpleNamespace(profiles=profiles,metadata=metadata),
        **{k:spec['schedule'][k] for k in ('recipes','partial','bins','mirrors','partial_modes')})
    WORK.update(options=options,spec=spec)


def slot(index):
    from ..s7_balanced_v2 import materialize as material
    result=material.slot(index)
    # Persist actual donor identities as part of cross-split leakage checks.
    for entry,stat in zip(result['entries'],result['statistics']):
        partial=stat['augmentation'].get('partial') or {}
        entry['augmentation_donor_sources']=([partial['donor_lineage']] if partial else [])
        entry['heldout_split']=WORK['options']['split']
    save(Path(WORK['options']['out'])/'groups'/('%05d.json'%index),result)
    return result


def materialize(sources_path, split, out, workers, seed, gate=False):
    from ..s7_balanced_v2 import materialize as material
    from ..s7_balanced_v2.audit_layered import one
    from .check_data_compatibility import check_batch
    from .data import Dataset,collate
    import torch
    torch.set_num_threads(1)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists():
        raise ValueError('new output required; do not overwrite prior generation')
    plan=read(sources_path);spec=plan['splits'][split]
    options=dict(sources=str(Path(sources_path).resolve()),split=split,out=str(out),
        seed=seed,attempts=1024)
    selected=list(range(spec['pairs']//2))
    if gate:
        strata={}
        for i,(recipe,mode) in enumerate(zip(spec['schedule']['recipes'],spec['schedule']['partial_modes'])):
            strata.setdefault((recipe,mode),i)
        selected=sorted(strata.values())
    start=time.time();result=[]
    save(out/'status.json',dict(status='running',stage='generate',pid=os.getpid(),planned_groups=len(selected)))
    try:
        with ProcessPoolExecutor(workers,initializer=initialize,initargs=(options,)) as pool:
            futures=[pool.submit(slot,i) for i in selected]
            try:
                for f in as_completed(futures):
                    result.append(f.result())
                    if len(result)%25==0 or len(result)==len(selected):
                        save(out/'status.json',dict(status='running',stage='generate',pid=os.getpid(),
                            completed_groups=len(result),planned_groups=len(selected),elapsed_seconds=time.time()-start))
            except BaseException as exc:
                save(out/'failure.json',dict(status='failed',error=repr(exc),pid=os.getpid(),
                    completed_groups=len(result),elapsed_seconds=time.time()-start))
                for future in futures:
                    future.cancel()
                raise
        result.sort(key=lambda r:r['slot']);entries=[e for r in result for e in r['entries']]
        record=dict(entries=[dict(pair_id=e['pair_id'],sample_path=str(out/e['artifact_path']),
            target_metadata=e['target_metadata'],label=e['label'],source_pair_id=e['source_pair_id'],
            source_row=e['source_row'],recipe=e['corrosion_recipe'],
            augmentation_donor_sources=e['augmentation_donor_sources']) for e in entries])
        manifest=out/(split+'.json');save(manifest,record)
        save(out/'archive_manifest.json',dict(entries=entries,artifact_root=str(out),
            source_plan_sha256=sha(sources_path),split=split,gate_only=gate))
        stats=material.summarize(result);save(out/'summary.json',stats)
        with ProcessPoolExecutor(workers) as pool:
            audit=list(pool.map(one,[(str(out),e) for e in entries],chunksize=2))
        save(out/'independent_pixel_audit.json',dict(status='passed',pairs=len(entries),
            all_actual_masks_reconstructed=True,topology_checked_all=True,rows=audit))
        data=Dataset(manifest);checked=[]
        for offset in range(0,len(data),2):
            items=[data[j] for j in range(offset,min(offset+2,len(data)))]
            checked.extend(check_batch(items,collate(items)))
        realized=Counter(e['corrosion_recipe'] for e in entries if e['label'])
        if not gate:
            assert realized==Counter(spec['schedule']['recipes']) and len(entries)==spec['pairs']
        assert len({e['pair_id'] for e in entries})==len(entries)
        assert all(sources(e['source_row'])<=set(spec['source_families']) for e in entries)
        assert all({family(s) for s in e['augmentation_donor_sources']}<=set(spec['source_families']) for e in entries)
        save(out/'validation.json',dict(status='passed',pairs=len(entries),gate_only=gate,
            actual_loader_and_supervision_checked_all=True,source_isolation_checked=True,
            negative_counts=stats['negative_kind'],recipe_counts=dict(realized),
            unique_base_pairs=stats['unique_base_pairs'],rows=checked))
        save(out/'status.json',dict(status='complete',gate_only=gate,pairs=len(entries),
            elapsed_seconds=time.time()-start,manifest=str(manifest),source_plan_sha256=sha(sources_path)))
        print(json.dumps(dict(status='complete',split=split,gate_only=gate,pairs=len(entries),elapsed=time.time()-start)),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(status='failed',error=repr(exc),pid=os.getpid(),elapsed_seconds=time.time()-start))
        save(out/'status.json',dict(status='failed',error=repr(exc)))
        raise


def finish(sources_path, root):
    root=Path(root).resolve();plan=read(sources_path);specs={}
    for split in ('cal','select','test'):
        directory=root/split;status=read(directory/'status.json')
        if status.get('status')!='complete' or status.get('gate_only') or status['pairs']!=plan['splits'][split]['pairs']:
            raise ValueError('full heldout split not complete: '+split)
        manifest=directory/(split+'.json');valid=directory/'validation.json';audit=directory/'independent_pixel_audit.json'
        specs[split]=dict(manifest=str(manifest),manifest_sha256=sha(manifest),pair_count=status['pairs'],
            validation_path=str(valid),validation_sha256=sha(valid),pixel_audit_path=str(audit),pixel_audit_sha256=sha(audit))
    output=root/'validation_plan.json'
    save(output,dict(schema='s7-consensus-validation-plan/3',kind='single_mixed',status='passed',
        physical_samples=6000,splits=specs,source_plan=str(Path(sources_path).resolve()),
        source_plan_sha256=sha(sources_path)))
    from .prepare_data import load_validation_plan
    load_validation_plan(output,set(plan['forbidden_sources']))
    print(json.dumps(dict(status='passed',physical_samples=6000,plan=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('plan','materialize','finish'))
    p.add_argument('--out',required=True);p.add_argument('--profile');p.add_argument('--sources')
    p.add_argument('--split',choices=('cal','select','test'));p.add_argument('--workers',type=int,default=12)
    p.add_argument('--seed',type=int,default=26092471);p.add_argument('--gate',action='store_true')
    a=p.parse_args()
    if a.action=='plan':build_plan(a.out,a.profile,a.seed)
    elif a.action=='materialize':materialize(a.sources,a.split,a.out,a.workers,a.seed,a.gate)
    else:finish(a.sources,a.out)
