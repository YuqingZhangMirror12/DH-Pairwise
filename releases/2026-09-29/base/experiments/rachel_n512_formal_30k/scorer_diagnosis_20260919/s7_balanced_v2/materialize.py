"""Materialize S7-B v2 in a separate directory, without changing any training."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

from staging.pairwise_v0_2.pairwise_data.rachel_composite_training import CompositeRachelPairDataset, _RowsDataset
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelDatasetConfig
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import (
    SCHEMA as ARCHIVE_SCHEMA, MaterializedWeatheredDataset, save_sample, load_sample)
from staging.pairwise_v0_2.pairwise_data.rachel_curve_cut import OutlineBank
from ..s7_compound_v1.materialize import read,save_json,digest,length_bin
from ..s7_compound_v1.geometry import masks,rng_for
from ..distribution_audit_20260923.measure import outline,pair_metrics
from ..seam_context_v3.augmentation import paired_mirror
from .geometry import SCHEMA,CORROSION_PERCENT,schedules,augment_group,category,length_quota_applies
from .source_search import SourceSearch

STATE={}


def initialize(options):
    import cv2,torch
    cv2.setNumThreads(1);torch.set_num_threads(1)
    plan=read(options['sources']);old=read(plan['s7_manifest']);o=old['protocol']['options']
    reference=read(o['reference_manifest']);composite_path=reference['protocol']['source_manifest']
    composite=read(composite_path)
    clean=CompositeRachelPairDataset(composite_path)
    gen5=MaterializedWeatheredDataset(o['gen5_manifest'])
    pos_entries={e['pair_id']:e for e in old['entries'] if e['label']}
    profile=read(options['profile']) if options.get('profile') else None
    if profile and profile.get('layered_damage_exclusive'):
        from .layered_geometry import pilot_sources
        positive,negative_plan=pilot_sources(plan,options['groups'],options['seed'])
    else:positive,negative_plan=plan['positive'][:options['groups']],plan['negative']
    # Alternatives preserve structural strata; negative source and anchor stay fixed.
    buckets=defaultdict(list)
    for e in plan['positive']:
        buckets[e['source_stratum']].append(e)
    native=[e for e in plan['negative'] if e['mode']=='native']
    negatives=_RowsDataset(plan['dataset_root'],[e['row'] for e in native],RachelDatasetConfig())
    recipes,partial,bins,mirrors=schedules(options['groups'],options['seed'],profile)
    from .partial_v14 import mode_schedule
    partial_modes=mode_schedule(recipes,options['seed'],profile or {},bins)
    STATE.update(options=options,plan=plan,positive=positive,negative_plan=negative_plan,pos_entries=pos_entries,
        clean=clean,clean_lookup={e['row']['pair_id']:i for i,e in enumerate(composite['entries'])},
        gen5=gen5,gen5_lookup={e['pair_id']:i for i,e in enumerate(gen5.entries)},
        negative=negatives,negative_lookup={e['pair_id']:i for i,e in enumerate(native)},
        buckets=buckets,recipes=recipes,partial=partial,bins=bins,mirrors=mirrors,
        bank=OutlineBank(o['outline_bank']),profile=profile,partial_modes=partial_modes)


@lru_cache(maxsize=12)
def clean_positive(pid):
    if pid in STATE['gen5_lookup']:
        sample,report=STATE['gen5'][STATE['gen5_lookup'][pid]]
        if report['changed_pair']:
            raise ValueError('structural source already weathered')
    else:
        sample=STATE['clean'][STATE['clean_lookup'][pid]]
    if not sample.label or sample.pair_id!=pid:
        raise ValueError('positive source mismatch')
    return sample


def negative_source(e):
    if e['mode']=='gen5':
        ix=STATE['gen5_lookup'][e['pair_id']]
        sample,report=STATE['gen5'][ix]
        if report['changed_pair']:
            raise ValueError('negative source already weathered')
        original=STATE['gen5'].entries[ix]
    else:
        sample=STATE['negative'][STATE['negative_lookup'][e['pair_id']]]
        original=dict(pair_id=e['pair_id'],label=False,source_root=STATE['plan']['dataset_root'],
                      source_row=e['row'],source_stratum=e['source_stratum'])
    if sample.label or sample.pair_id!=e['pair_id']:
        raise ValueError('negative source mismatch')
    return sample,original


def slot(index):
    from ..seam_context_v3.targets import known_gap_links
    s=STATE;o=s['options'];root=Path(o['out']);path=root/'groups'/('%05d.json'%index)
    recipe,partial,wanted,code=(s[key][index] for key in ('recipes','partial','bins','mirrors'))
    layered=bool(s['profile'] and s['profile'].get('layered_damage_exclusive'))
    axis=(None,'horizontal','vertical')[code]
    if path.exists():
        r=read(path)
        if (r['recipe'],r['partial'],r['length_bin'],r['mirror_code'])!=(recipe,partial,wanted,code):
            raise ValueError('resume schedule mismatch')
        if any(not (root/e['artifact_path']).exists() or not Path(e['target_metadata']).exists() for e in r['entries']):
            raise ValueError('partial archive group')
        return r
    neg_plan=s['negative_plan'][index];negative,negative_entry=negative_source(neg_plan)
    original=s['positive'][index];alternatives=s['buckets'][original['source_stratum']]
    rng=rng_for(o['seed'],SCHEMA,'replacement',index);failures=Counter()
    area=None
    if s['profile'] and s['profile'].get('conditional_area_deciles_px2'):
        arng=rng_for(o['seed'],SCHEMA,'fixed-area',index)
        quantiles=s['profile']['conditional_area_deciles_px2'][wanted]
        area=float(np.interp(arng.random(),np.linspace(0,1,len(quantiles)),quantiles))
    search=SourceSearch(original,alternatives,rng,o['attempts'],
        enabled=layered and not partial and length_quota_applies(s['profile'],False))
    while (draw:=search.draw()) is not None:
        attempt,source=draw
        positive=clean_positive(source['pair_id'])
        seed=int(rng_for(o['seed'],SCHEMA,index,attempt).integers(0,2**31))
        rows,detail=augment_group(positive,negative,recipe,partial,s['bank'],seed,s['profile'],area,
            quota_bin=wanted,mirror_axis=axis if layered else None,partial_mode=s['partial_modes'][index])
        if rows is None:
            search.rejected(source,static_length=detail.get('stage')=='layered_length_quota')
            failures[detail.get('stage','')+':'+str(detail.get('detail',{}).get('reason','rejected'))]+=1
            nested=detail.get('detail',{}).get('detail',detail.get('detail',{}))
            for reason,count in nested.get('attempt_reasons',{}).items():
                failures['inner:'+detail.get('stage','')+':'+reason]+=count
            continue
        measured=pair_metrics(*[outline(m) for m in masks(rows[0][0])],rows[0][0].translation_a_to_b_rc)
        quota_pre=bool(s['profile'] and s['profile'].get('length_quota_basis')=='pre_weather_near_length')
        before_metrics=pair_metrics(*[outline(m) for m in masks(rows[0][2])],rows[0][2].translation_a_to_b_rc) if quota_pre else measured
        length=rows[0][1]['compound']['quota_length_px'] if layered else before_metrics['d20_length_px']
        enforce_length=length_quota_applies(s['profile'],partial)
        if enforce_length and (not 32<=length<=800 or length_bin(length)!=wanted):
            search.rejected(source)
            failures['measured_length_outside_quota']+=1;continue
        entries=[];statistics=[]
        for ordinal,((sample,report,before_weather),entry) in enumerate(zip(rows,(s['pos_entries'][source['pair_id']],negative_entry))):
            latent_arrays=report.pop('_latent_arrays',None)
            weather_arrays=report.pop('_weather_arrays',None)
            micro_arrays=report.pop('_micro_weather_arrays',None)
            mirror_done=report.pop('_offline_mirror_applied_before_damage',False)
            coordinate_frame=report.pop('_weather_coordinate_frame','pre_mirror_800px')
            metadata=known_gap_links(sample,before_weather,report)
            if axis and not mirror_done:
                sample=paired_mirror(sample,axis)
            if layered:
                assert mirror_done and coordinate_frame=='post_fragment_pre_damage_800px'
                report['augmentation_layers']['fragments']['source_stratum']=entry['source_stratum']
            pid='s7b2-'+hashlib.sha256(f'{o["seed"]}:{index}:{ordinal}:{sample.pair_id}'.encode()).hexdigest()[:24]
            sample=replace(sample,pair_id=pid,
                fragment_a_token=sample.fragment_a_token+'@s7b2-'+str(index)+'-a',
                fragment_b_token=sample.fragment_b_token+'@s7b2-'+str(index)+'-b')
            report.update(source_pair_id=entry['pair_id'],offline_paired_mirror=axis,
                source_slot=index,generation_attempts=attempt+1,generation_rejections=dict(failures),
                generation_damage_attempts=search.damage_attempts+1,
                static_length_sources_excluded=len(search.excluded),
                negative_kind=neg_plan['kind'] if ordinal else None,
                anchor_group_id=neg_plan['anchor_group_id'] if ordinal else None)
            target=root/'targets'/('%05d_%d.npz'%(index,ordinal));target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('wb') as stream:
                np.savez_compressed(stream,**metadata)
            relative='samples/%05d_%d.npz'%(index,ordinal)
            weather_path=None
            if weather_arrays is not None:
                weather_path='weather/%05d_%d.npz'%(index,ordinal)
                (root/'weather').mkdir(exist_ok=True)
                with (root/weather_path).open('wb') as stream:
                    np.savez_compressed(stream,**weather_arrays,
                        packed_preweather_a=np.packbits(before_weather.mask_a[0].astype(bool),axis=1),
                        packed_preweather_b=np.packbits(before_weather.mask_b[0].astype(bool),axis=1),
                        translation_a_to_b_rc=before_weather.translation_a_to_b_rc,
                        offline_mirror_code=np.array(code),coordinate_frame=np.array(coordinate_frame))
            latent_path=None
            micro_path=None
            if micro_arrays is not None:
                micro_path='micro_weather/%05d_%d.npz'%(index,ordinal)
                (root/'micro_weather').mkdir(exist_ok=True)
                with (root/micro_path).open('wb') as stream:
                    np.savez_compressed(stream,**micro_arrays,coordinate_frame=np.array('post_primary_pre_background_800px'))
            if latent_arrays is not None:
                latent_path='latent/%05d_%d.npz'%(index,ordinal)
                (root/'latent').mkdir(exist_ok=True)
                with (root/latent_path).open('wb') as stream:
                    np.savez_compressed(stream,**latent_arrays,
                        packed_preweather_a=np.packbits(before_weather.mask_a[0].astype(bool),axis=1),
                        packed_preweather_b=np.packbits(before_weather.mask_b[0].astype(bool),axis=1),
                        translation_a_to_b_rc=before_weather.translation_a_to_b_rc,
                        offline_mirror_code=np.array(code),coordinate_frame=np.array(coordinate_frame))
            save_sample(root/relative,sample,report)
            loaded,receipt=load_sample(root/relative)
            if loaded.pair_id!=pid or receipt['compound']['recipe']!=recipe:
                raise ValueError('archive roundtrip failed')
            output=dict(entry,pair_id=pid,source_pair_id=entry['pair_id'],artifact_path=relative,
                target_metadata=str(target),source_row=dict(entry['source_row'],pair_id=pid),
                corrosion_recipe=recipe,corrosion_category=report['compound']['corrosion_category'],
                partial_applied=partial,offline_paired_mirror=axis,changed_pair=report['changed_pair'],
                negative_kind=neg_plan['kind'] if ordinal else None,
                anchor_group_id=neg_plan['anchor_group_id'] if ordinal else None,
                anchor_fragment_token=neg_plan.get('anchor_fragment_token') if ordinal else None,
                latent_seam_artifact=latent_path,
                weather_artifact=weather_path,
                background_artifact=micro_path,
                inherited_match_count=int((sample.target_a>=0).sum()))
            if layered:
                output['augmentation_layers']=report['augmentation_layers']
            entries.append(output)
            metrics=measured if not ordinal else pair_metrics(*[outline(m) for m in masks(sample)])
            statistics.append(dict(pair_id=pid,source_pair_id=entry['pair_id'],label=bool(sample.label),
                recipe=recipe,corrosion_category=report['compound']['corrosion_category'],partial=partial,
                mirror=axis,negative_kind=output['negative_kind'],anchor_group_id=output['anchor_group_id'],
                target_length_bin=wanted if enforce_length else None,area_reference_bin=wanted,
                actual_length_bin=length_bin(length) if not ordinal else None,
                augmentation=report['compound'],matched_tokens=output['inherited_match_count'],**metrics))
            statistics[-1]['pair_shared_scale']=report.get('pair_shared_scale')
            statistics[-1]['latent_seam']=report.get('latent_seam')
            statistics[-1]['quota_length_px']=length if not ordinal else None
            statistics[-1]['quota_length_basis']='pre_weather_near_length' if quota_pre else 'post_weather_near_length'
            if layered:
                statistics[-1].update(augmentation_layers=report['augmentation_layers'],
                    background_degradation=report['background_degradation'],
                    fragment_stage_area_px=report['fragment_stage_area_px'],damage_stage_area_px=report['damage_stage_area_px'],
                    quota_length_basis=report['compound']['quota_length_basis'])
        result=dict(slot=index,recipe=recipe,partial=partial,length_bin=wanted,mirror_code=code,
            source_positive_pair_id=source['pair_id'],original_positive_pair_id=original['pair_id'],
            source_negative_pair_id=neg_plan['pair_id'],attempts=attempt+1,rejections=dict(failures),
            damage_attempts=search.damage_attempts+1,static_length_sources_excluded=len(search.excluded),
            entries=entries,statistics=statistics)
        save_json(path,result)
        return result
    raise RuntimeError(json.dumps(dict(slot=index,recipe=recipe,partial=partial,negative=neg_plan['pair_id'],
        failures=dict(failures),damage_attempts=search.damage_attempts,source_draws=search.draws,
        static_length_sources_excluded=len(search.excluded),damage_attempt_budget=o['attempts'],
        source_pool_exhausted=not bool(search.pool))))


def summarize(result):
    entries=[e for r in result for e in r['entries']];rows=[v for r in result for v in r['statistics']]
    positive=[r for r in rows if r['label']];negative=[e for e in entries if not e['label']]
    metrics={}
    for label in (True,False):
        rr=[r for r in rows if r['label']==label];dm={}
        for key in ('d20_length_px','d40_gap_mean_px','d10_breaks_mean','mean_fragment_area_px','area_ratio','matched_tokens'):
            values=np.array([r[key] for r in rr if r.get(key) is not None],float)
            dm[key]=dict(n=len(values),mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                p10=float(np.quantile(values,.1)) if len(values) else None,
                p90=float(np.quantile(values,.9)) if len(values) else None)
        metrics[str(label)]=dm
    return dict(sample_count=len(entries),positive_count=len(positive),negative_count=len(negative),
        corrosion_recipe_counts=dict(Counter(r['recipe'] for r in positive)),
        corrosion_category_counts=dict(Counter(r['corrosion_category'] for r in positive)),
        partial_fraction=sum(r['partial'] for r in positive)/len(positive),
        mirror_counts=dict(Counter(str(r['mirror']) for r in positive)),
        negative_kind=dict(Counter(e['negative_kind'] for e in negative)),
        grouped_negative_pairs=sum(bool(e['anchor_group_id']) for e in negative),
        anchor_groups=dict(Counter(e['anchor_group_id'] for e in negative if e['anchor_group_id'])),
        source_strata=dict(Counter(('positive:' if e['label'] else 'negative:')+e['source_stratum'] for e in entries)),
        unique_base_pairs=len(set(e['source_pair_id'] for e in entries)),
        unique_negative_base_pairs=len(set(e['source_pair_id'] for e in negative)),
        positive_source_replacements=sum(r['source_positive_pair_id']!=r['original_positive_pair_id'] for r in result),
        max_positive_reuse=max(Counter(r['source_positive_pair_id'] for r in result).values()),
        rejections=dict(sum((Counter(r['rejections']) for r in result),Counter())),metrics=metrics)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',required=True);p.add_argument('--out',required=True)
    p.add_argument('--groups',type=int,default=12000);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--seed',type=int,default=26092332);p.add_argument('--attempts',type=int,default=2048)
    p.add_argument('--profile',help='Explicit distribution revision; absent preserves the v2 generator')
    a=p.parse_args()
    if a.groups<=0 or a.groups>12000 or a.groups%200 or not 1<=a.workers<=96:
        p.error('groups multiple200 within12000; workers1..96')
    root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True);options=vars(a);options['out']=str(root)
    lock=root/'producer.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    start=time.time();plan=read(a.sources)
    profile=read(a.profile) if a.profile else None
    layered=bool(profile and profile.get('layered_damage_exclusive'))
    if layered:
        from .layered_geometry import corrosion_layer
        recipe_category=corrosion_layer
    else:recipe_category=category
    protocol=dict(schema=SCHEMA,options=options,source_plan_sha256=digest(a.sources),
        distribution_revision=profile,
        recipe_percent=profile['corrosion_percent'] if profile else CORROSION_PERCENT,
        corrosion_categories=dict(sum((Counter({recipe_category(k):v}) for k,v in
            (profile['corrosion_percent'] if profile else CORROSION_PERCENT).items()),Counter())),
        independent_partial_percent=0 if layered else 70,
        partial_percent=profile['partial_percent'] if layered else 70,
        partial_policy='exclusive_corrosion_recipe' if layered else 'independent_axis',
        augmentation_order=profile['layer_order'] if layered else ['source_structure','partial','scale','weather','mirror'],
        offline_mirror_percent=15,
        negative_source_percent={'cross_gen':35,'cross_parent_same_gen':35,'same_parent_nonadjacent':30},
        source_plan_stats=plan['stats'],
        length_quota={'32to256':25,'256to512':50,'512to800':25} if length_quota_applies(profile,True) else None,
        length_quota_enforced=length_quota_applies(profile,True),
        nonpartial_length_quota_enforced=length_quota_applies(profile,False),
        partial_length_quota_enforced=length_quota_applies(profile,True),
        length_quota_scope='all' if length_quota_applies(profile,True) else 'nonpartial_only' if length_quota_applies(profile,False) else 'none',
        area_reference_strata={'short':25,'medium':50,'long':25},
        length_quota_basis=profile.get('length_quota_basis','post_weather_near_length') if profile else 'post_weather_near_length',
        original_data_or_training_modified=False,training_started=False,
        heldout_masks_used=False,real_aggregate_statistics_used=bool(profile),mirror_replaces_not_adds_to_online_mirror=True,
        source_search_rule='Only proven-infeasible non-Partial source/area/length combinations are excluded; original1024 damage attempts retained, no quota relaxation',
        generator_sha256={str(f.relative_to(Path(__file__).parent.parent)):digest(f) for f in
            (Path(__file__),Path(__file__).with_name('geometry.py'),Path(__file__).with_name('prepare.py'),
             Path(__file__).with_name('source_search.py'),
             Path(__file__).with_name('scale.py'),Path(__file__).with_name('latent_seam.py'),
             Path(__file__).parent.parent/'s7_compound_v1/geometry.py',
             Path(__file__).with_name('conservative_weather.py'),Path(__file__).with_name('layered_geometry.py'),
             Path(__file__).with_name('background_recession.py'),Path(__file__).with_name('partial_v14.py'))})
    if protocol['length_quota_scope']=='nonpartial_only':
        _,flags,bins,_=schedules(a.groups,a.seed,profile)
        protocol['nonpartial_length_counts']=dict(Counter(b for flag,b in zip(flags,bins) if not flag))
    if (root/'reuse_receipt.json').exists():
        receipt=read(root/'reuse_receipt.json')
        protocol['recovery']=dict(receipt_sha256=digest(root/'reuse_receipt.json'),
            retained_samples=receipt['samples'],source=receipt['source'],
            fresh_seed_replay_equivalence_claim=False,original_source_modified=False)
    try:
        if (root/'protocol.json').exists() and read(root/'protocol.json')!=protocol:
            raise ValueError('different existing generation protocol')
        save_json(root/'protocol.json',protocol);result=[]
        with ProcessPoolExecutor(a.workers,initializer=initialize,initargs=(options,)) as pool:
            pending=[pool.submit(slot,i) for i in range(a.groups)]
            try:
                for future in as_completed(pending):
                    result.append(future.result())
                    if len(result)%25==0 or len(result)==a.groups:
                        status=dict(status='running',pid=os.getpid(),completed_groups=len(result),planned_groups=a.groups,elapsed_seconds=time.time()-start)
                        save_json(root/'status.json',status);print(json.dumps(status),flush=True)
            except BaseException as exc:
                # Publish the original worker failure before shutdown waits for
                # other in-flight slots. Completed archives are kept for diagnosis.
                save_json(root/'failure.json',dict(status='failed',error=repr(exc),
                    completed_groups=len(result),pid=os.getpid(),elapsed_seconds=time.time()-start))
                print(json.dumps(dict(status='worker_failed',error=repr(exc))),flush=True)
                for future in pending:future.cancel()
                raise
        # Completion order must not change the fixed training manifest order.
        result.sort(key=lambda r:r['slot'])
        entries=[e for r in result for e in r['entries']];stats=summarize(result)
        if len({e['pair_id'] for e in entries})!=2*a.groups:
            raise ValueError('duplicate final IDs')
        manifest=root/'train_s7b_24k.json'
        save_json(manifest,dict(schema_version=ARCHIVE_SCHEMA,split='train',artifact_root=str(root),entries=entries,protocol=protocol,stats=stats))
        save_json(root/'train.json',dict(entries=[dict(pair_id=e['pair_id'],sample_path=str(root/e['artifact_path']),
            target_metadata=e['target_metadata'],label=e['label'],source_pair_id=e['source_pair_id'],
            source_row=e['source_row'],recipe=e['corrosion_recipe'],anchor_group_id=e['anchor_group_id']) for e in entries],protocol=protocol,stats=stats))
        with (root/'pair_metrics.jsonl').open('w') as stream:
            for r in result:
                for value in r['statistics']:
                    stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
        save_json(root/'summary.json',stats)
        save_json(root/'status.json',dict(status='complete' if a.groups==12000 else 'pilot_complete',
            pid=os.getpid(),manifest=str(manifest),sample_count=len(entries),elapsed_seconds=time.time()-start))
        print(json.dumps(dict(status='complete' if a.groups==12000 else 'pilot_complete',sample_count=len(entries))),flush=True)
    except BaseException as exc:
        save_json(root/'status.json',dict(status='failed',pid=os.getpid(),error=repr(exc),elapsed_seconds=time.time()-start));raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
