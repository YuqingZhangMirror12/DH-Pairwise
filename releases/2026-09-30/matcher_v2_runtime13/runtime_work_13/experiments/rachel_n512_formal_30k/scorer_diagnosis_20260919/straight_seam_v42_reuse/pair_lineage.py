"""Read-only original-pair registry and cross-release straight-pair admission.

The old catalog is NOT rewritten: only newly added rows can be rejected.
Original source masks (or clean Gen5 materializations) are fingerprinted before
any v14/v17/v17.5/v18 damage. No model, GT ranking, or test metric is used.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

if __package__:
    from . import generate as g, supervision as s
    from .pair_identity import base_pair_sha256
else:
    import generate as g
    import supervision as s
    from pair_identity import base_pair_sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(value, message):
    if not value:raise ValueError(message)


def source_token_key(row):
    """A/B-invariant original fragment identities, independent of damage recipe."""
    source=row['source_row']
    tokens=[source['fragment_'+side]['fragment_token'] for side in 'ab']
    require(all(isinstance(t,str) and t for t in tokens) and len(set(tokens))==2,
            'Two original fragment tokens are required')
    return digest(dict(root=str(Path(row['source_root']).resolve()),fragments=sorted(tokens)))


def original_rows(base, archives, read):
    """Join exact bound admitted rows to original provenance; no fuzzy joining."""
    result=[]
    require(digest(base['catalog'])==base['catalog_sha256'],'Original catalog changed')
    for stage,path in archives.items():
        require(str(path) in base['bound_inputs'],'Unbound original archive')
        archive=read(path,base['bound_inputs'][str(path)])
        rows={r['pair_id']:r for r in archive['entries']}
        require(len(rows)==len(archive['entries']),'Duplicate old Pair ID')
        spec=base['training_manifests'][stage];manifest=read(spec['path'],spec['sha256'])
        for entry in manifest['entries']:
            row=dict(rows[entry['pair_id']])
            if stage=='v17_filtered':
                row['artifact_path']=str((path.parent/row['artifact_path']).resolve())
            require(digest(row)==entry['source_entry_sha256'],'Old provenance binding differs')
            key=str(row['source_root'])+'::'+row['source_pair_id']
            require(key==entry['source_base_key'] and row['label']==entry['label'],
                    'Old original pair/label differs')
            result.append(dict(stage=stage,entry=entry,original=row))
    expected={(r['stage'],r['pair_id'],r['source_base_key']) for r in base['catalog']}
    actual={(r['stage'],r['entry']['pair_id'],r['entry']['source_base_key']) for r in result}
    require(actual==expected and len(actual)==len(result),'Incomplete original catalog join')
    return result


def resolve_masks(row, gen5, gen5_root, load_mask, load_sample, bind):
    source=row['source_row'];root=Path(row['source_root']).resolve()
    paths=[source['fragment_'+side].get('model_mask_path') for side in 'ab']
    if all(paths):
        masks=[]
        for relative in paths:
            path=(root/relative).resolve()
            require(path.is_relative_to(root) and path.suffix=='.png'
                    and Path(relative).parts[:2]==('model','masks_800'),'Unsafe original mask path')
            bind(path);masks.append(load_mask(path))
        return masks,'original_model_masks'
    require(not any(paths) and row['source_pair_id'] in gen5,'Unresolved original pair')
    clean=gen5[row['source_pair_id']]
    require(clean['label']==row['label'] and source_token_key(clean)==source_token_key(row),
            'Clean Gen5 pair does not match original provenance')
    path=(gen5_root/clean['artifact_path']).resolve()
    require(path.is_relative_to(gen5_root),'Unsafe clean Gen5 artifact')
    bind(path);sample,report=load_sample(path)
    require(sample.pair_id==row['source_pair_id'] and bool(sample.label)==row['label']
            and report.get('physical_damage_applied') is False,'Gen5 reference is not pre-damage')
    require(sorted((sample.fragment_a_token,sample.fragment_b_token))==sorted(
        source['fragment_'+side]['fragment_token'] for side in 'ab'),'Gen5 model token mismatch')
    return [sample.mask_a[0],sample.mask_b[0]],'clean_gen5_materialization'


def build_registry(base_path,archives,gen5_path,official_code,out):
    bound={}
    def bind(path,expected=None):
        path=str(Path(path).resolve());value=g.sha(path)
        require(expected is None or value==expected,'Bound file changed: '+path)
        require(path not in bound or bound[path]==value,'File changed during audit')
        bound[path]=value;return value
    def read(path,expected=None):
        bind(path,expected);return json.loads(Path(path).read_text())
    base=read(base_path);require(base['status']=='passed','Base admission failed')
    rows=original_rows(base,archives,read);pool=read(gen5_path)
    require(pool['split']=='train','Gen5 pool is not TRAIN')
    clean={r['pair_id']:r for r in pool['entries']}
    require(len(clean)==len(pool['entries']),'Duplicate clean Gen5 Pair ID')
    sys.path.insert(0,str(official_code))
    from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import _load_mask,RachelDatasetConfig
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
    config=RachelDatasetConfig();records={};methods=Counter();started=time.time()
    for i,selected in enumerate(rows):
        row=selected['original'];key=selected['entry']['source_base_key'];token=source_token_key(row)
        if key not in records:
            masks,method=resolve_masks(row,clean,Path(pool['artifact_root']).resolve(),
                lambda p:_load_mask(p,config)[0][0],load_sample,bind)
            records[key]=dict(source_base_key=key,source_token_key=token,
                pre_damage_pair_sha256=base_pair_sha256(*masks),label=bool(row['label']),
                method=method,stages=[],admitted_pair_ids=[])
            methods[method]+=1
        record=records[key]
        require(record['source_token_key']==token and record['label']==row['label'],
                'Original pair identity has conflicting provenance')
        record['stages']=sorted(set(record['stages']+[selected['stage']]))
        record['admitted_pair_ids'].append(selected['entry']['pair_id'])
        if (i+1)%1000==0:print(json.dumps(dict(checked=i+1,total=len(rows),original_pairs=len(records))),flush=True)
    for path,value in bound.items():require(g.sha(path)==value,'Source changed during registry build')
    result=dict(schema='original-pair-registry/1',status='passed',base_admission=dict(path=str(base_path),sha256=g.sha(base_path)),
        admitted_rows=len(rows),original_pairs=len(records),records=sorted(records.values(),key=lambda r:r['source_base_key']),
        methods=dict(methods),bound_inputs=bound,seconds=time.time()-started,unresolved_pairs=0,
        scope='All original positive and negative pairs in the admitted v17/v17.5/v18 catalog.',
        exact_identity='Before damage; independent centering, A/B swap and common D4 transforms removed.',
        limitation='Not an approximate manuscript-similarity or arbitrary-rescale identity classifier.',
        old_exposures_removed=0,training_admitted=False,gpu=False)
    g.save(out/'registry.json',result)
    g.save(out/'complete.json',dict(status='complete',registry_sha256=g.sha(out/'registry.json'),
        admitted_rows=len(rows),original_pairs=len(records),unresolved_pairs=0,training_admitted=False))
    return result


def compare_records(original,new):
    old_hashes={r['pre_damage_pair_sha256'] for r in original}
    old_keys={r['source_token_key'] for r in original}
    seen={};failures=[]
    for row in new:
        h=row['pre_damage_pair_sha256'];name=row['pair_id']
        if h in old_hashes:failures.append(dict(pair_id=name,reason='duplicates_old_pre_damage_pair'))
        if row.get('source_token_key') in old_keys:failures.append(dict(pair_id=name,reason='duplicates_old_source_pair'))
        if h in seen:failures.append(dict(pair_id=name,reason='repeated_new_pre_damage_pair',other=seen[h]))
        seen[h]=name
    return failures


def compare_release(registry_path,generation_path,out):
    registry=json.loads(registry_path.read_text());gen=json.loads(generation_path.read_text())
    require(registry['status']=='passed' and registry['unresolved_pairs']==0,'Original registry unresolved')
    require(gen['status']=='complete_generation_integrity_supervision','Full strict release not complete')
    for path,value in registry['bound_inputs'].items():require(g.sha(path)==value,'Original lineage changed')
    records=[];split_specs={};negative_count=0
    for split,spec in gen['datasets'].items():
        require(g.sha(spec['manifest_path'])==spec['manifest_sha256']
                and g.sha(spec['audit_path'])==spec['audit_sha256'],'Generated membership changed')
        manifest=json.loads(Path(spec['manifest_path']).read_text());audit=json.loads(Path(spec['audit_path']).read_text())
        checked={r['pair_id']:r for r in audit['records']}
        require(audit['status']=='passed_integrity_and_supervision' and audit['rows']==len(manifest['entries']),
                'Incomplete strict independent audit')
        for entry in manifest['entries']:
            if not entry['label']:
                negative_count+=1;continue
            require(g.sha(entry['proof_path'])==entry['proof_sha256'],'Pre-damage proof changed')
            with np.load(entry['proof_path']) as f:proof={k:f[k] for k in ('shape','cut_a','cut_b')}
            identity=base_pair_sha256(s.unpack(proof,'cut_a'),s.unpack(proof,'cut_b'))
            require(identity==entry['pre_damage_pair_sha256']==checked[entry['pair_id']]['pre_damage_pair_sha256'],
                    'Independent pre-damage identity differs')
            records.append(dict(pair_id=entry['pair_id'],split=split,pre_damage_pair_sha256=identity,
                original_cut_event=dict(seed=entry['generation_seed'],id=entry['id'],accepted_attempt=entry['tries']),
                source_operation='Fresh common cut of a single synthetic parent; not a damaged variant of an old two-fragment pair.'))
        split_specs[split]=dict(manifest_sha256=spec['manifest_sha256'],audit_sha256=spec['audit_sha256'],rows=len(manifest['entries']))
    failures=compare_records(registry['records'],records)
    result=dict(schema='straight-pair-lineage-audit/1',status='passed' if not failures else 'failed',failures=failures,
        original_registry=dict(path=str(registry_path),sha256=g.sha(registry_path)),base_admission=registry['base_admission'],
        full_generation_complete=dict(path=str(generation_path),sha256=g.sha(generation_path)),
        original_pairs_checked=registry['original_pairs'],positive_pre_damage_pairs_checked=len(records),
        negative_rows=negative_count,negative_policy='Independently sourced nonmatching pairs, no common seam GT; full-input dedup in canonical preparation.',
        splits=split_specs,records=records,old_exposures_removed=0,gpu=False,training_admitted=False)
    g.save(out/'lineage_audit.json',result)
    if not failures:g.save(out/'complete.json',dict(status='complete',lineage_audit_sha256=g.sha(out/'lineage_audit.json')))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['registry','compare'])
    for key in ('base-admission','v17-archive','v175-archive','v18-archive','gen5-manifest','official-code','registry','generation-complete'):
        p.add_argument('--'+key,type=Path)
    p.add_argument('--output-new',type=Path,required=True);a=p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU-only execution required')
    out=a.output_new.resolve();out.mkdir(parents=True,exist_ok=False)
    g.save(out/'launch.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        started_unix=time.time(),mode=a.mode,code_sha256=g.sha(__file__),gpu=False))
    try:
        if a.mode=='registry':
            result=build_registry(a.base_admission,dict(v17_filtered=a.v17_archive,**{'v17.5':a.v175_archive,'v18':a.v18_archive}),
                a.gen5_manifest,a.official_code,out)
        else:result=compare_release(a.registry,a.generation_complete,out)
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','bound_inputs','failures','splits')}),flush=True)
        return 0 if result['status']=='passed' else 2
    except Exception as exc:
        g.save(out/'failure.json',dict(status='failed',error=repr(exc),training_admitted=False));raise


if __name__=='__main__':raise SystemExit(main())
