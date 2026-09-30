"""Build a new balanced TRAIN dataset; original S7/F/I paths are read-only."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from staging.pairwise_v0_2.pairwise_data.rachel_composite_training import CompositeRachelPairDataset
from staging.pairwise_v0_2.pairwise_data.rachel_curve_cut import OutlineBank
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import (
    SCHEMA as ARCHIVE_SCHEMA, MaterializedWeatheredDataset, load_sample, save_sample)
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.distribution_audit_20260923.measure import outline, pair_metrics
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.seam_context_v3.augmentation import paired_mirror, mirror_schedule
from .geometry import (SCHEMA, RECIPE_PERCENT, RECIPE_LENGTH_PERCENT, augment_group,
                       rng_for, schedule, length_schedule, masks)

STATE = {}


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(options):
    import cv2
    import torch
    cv2.setNumThreads(1); torch.set_num_threads(1)
    old = read(options['s7_manifest']); old_options = old['protocol']['options']
    reference = read(old_options['reference_manifest'])
    composite_path = reference['protocol']['source_manifest']
    clean = CompositeRachelPairDataset(composite_path)
    composite = read(composite_path)
    clean_lookup = {e['row']['pair_id']:i for i, e in enumerate(composite['entries'])}
    gen5 = MaterializedWeatheredDataset(old_options['gen5_manifest'])
    gen5_lookup = {e['pair_id']:i for i, e in enumerate(gen5.entries)}
    groups = defaultdict(list)
    for entry in old['entries']:
        groups[entry['s7_group_index']].append(entry)
    groups = [sorted(groups[g], key=lambda e:not e['label']) for g in sorted(groups)]
    if len(groups) != 12000 or any(len(g) != 2 or not g[0]['label'] or g[1]['label'] for g in groups):
        raise ValueError('expected original S7 12K paired positive/negative groups')
    buckets = defaultdict(list)
    for g, entries in enumerate(groups):
        buckets[tuple(e.get('source_stratum') for e in entries)].append(g)
    count = options['groups']
    group_order = rng_for(options['seed'], 'initial-source-order').permutation(len(groups))[:count]
    recipes = schedule(count, options['seed'])
    bins = length_schedule(recipes, options['seed'])
    STATE.update(options=options, clean=clean, clean_lookup=clean_lookup, gen5=gen5,
        gen5_lookup=gen5_lookup, bank=OutlineBank(old_options['outline_bank']),
        groups=groups, buckets=buckets, group_order=group_order,
        recipes=recipes, bins=bins,
        mirrors=mirror_schedule(count, .30, options['seed'], 0))


@lru_cache(maxsize=12)
def clean_group(index):
    result=[]
    for entry in STATE['groups'][index]:
        pid=entry['pair_id']
        if pid in STATE['gen5_lookup']:
            sample, report=STATE['gen5'][STATE['gen5_lookup'][pid]]
            if report['changed_pair']:
                raise ValueError('Gen5 source must be clean structural data')
        else:
            sample=STATE['clean'][STATE['clean_lookup'][pid]]
        if sample.pair_id != pid or bool(sample.label) != bool(entry['label']):
            raise ValueError('clean source mismatch')
        result.append(sample)
    return tuple(result)


def length_bin(value):
    return 'short' if value < 256. else 'medium' if value < 512. else 'long'


def materialize_slot(slot):
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.seam_context_v3.targets import known_gap_links
    state = STATE; options=state['options']; root=Path(options['out'])
    path=root/'groups'/('%05d.json'%slot)
    recipe=state['recipes'][slot]; wanted_bin=state['bins'][slot]
    if path.exists():
        result=read(path)
        if result['recipe'] != recipe or result['target_length_bin'] != wanted_bin:
            raise ValueError('resume recipe mismatch')
        for e in result['entries']:
            if not (root/e['artifact_path']).is_file() or not Path(e['target_metadata']).is_file():
                raise ValueError('incomplete saved group')
        return result
    original_group=int(state['group_order'][slot])
    strata=tuple(e.get('source_stratum') for e in state['groups'][original_group])
    alternatives=state['buckets'][strata]
    rng=rng_for(options['seed'], 'replacements', slot)
    failures=Counter()
    for attempt in range(options['attempts']):
        # Keep a source for four independent placements before using another
        # TRAIN pair in exactly the same positive/negative source strata.
        group_index=original_group if attempt < 4 else int(rng.choice(alternatives))
        original=clean_group(group_index)
        seed=int(rng_for(options['seed'], slot, attempt).integers(0, 2**31))
        rows, status=augment_group(*original, recipe, state['bank'], seed)
        if rows is None:
            failures[status.get('stage', '')+':'+status.get('detail', {}).get('reason', 'rejected')] += 1
            continue
        positive=rows[0][0]
        measured=pair_metrics(*[outline(m) for m in masks(positive)], positive.translation_a_to_b_rc)
        if measured['d20_length_px'] < 32.:
            failures['less_than32px_near_contact_proxy'] += 1; continue
        if measured['d20_length_px'] > 800.:
            failures['more_than800px_long_contact_tail'] += 1; continue
        if options['length_quotas'] and length_bin(measured['d20_length_px']) != wanted_bin:
            failures['outside_target_length_bin'] += 1; continue
        code=int(state['mirrors'][slot]); axis=(None, 'horizontal', 'vertical')[code]
        entries=[]; statistics=[]
        for ordinal, ((sample, report, before_weather), source) in enumerate(zip(rows, state['groups'][group_index])):
            # Gap metadata is computed in its documented pre-weather source;
            # reflections keep token IDs and metadata indices unchanged.
            metadata=known_gap_links(sample, before_weather, report)
            if axis:
                sample=paired_mirror(sample, axis)
            key=hashlib.sha256((str(slot)+':'+sample.pair_id+':'+str(options['seed'])).encode()).hexdigest()[:24]
            sample=replace(sample, pair_id='s7c1-'+key,
                fragment_a_token=sample.fragment_a_token+'@s7c1-'+str(slot)+'-a',
                fragment_b_token=sample.fragment_b_token+'@s7c1-'+str(slot)+'-b')
            report.update(source_pair_id=source['pair_id'], offline_paired_mirror=axis,
                source_group_index=group_index, compound_slot=slot,
                generation_attempts=attempt+1, generation_rejections=dict(failures))
            relative='samples/%05d_%d.npz'%(slot, ordinal)
            target_path=root/'targets'/('%05d_%d.npz'%(slot, ordinal))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open('wb') as stream:
                np.savez_compressed(stream, **metadata)
            save_sample(root/relative, sample, report)
            # Loading checks every final archive, including reciprocal targets.
            loaded, rr=load_sample(root/relative)
            if loaded.pair_id != sample.pair_id or rr['compound']['recipe'] != recipe:
                raise ValueError('roundtrip failed')
            entry=dict(source, source_pair_id=source['pair_id'], pair_id=sample.pair_id,
                artifact_path=relative, target_metadata=str(target_path),
                source_row=dict(source['source_row'], pair_id=sample.pair_id),
                changed_pair=report['changed_pair'], compound_recipe=recipe,
                compound_slot=slot, source_group_index=group_index,
                offline_paired_mirror=axis,
                inherited_match_count=int((sample.target_a >= 0).sum()))
            entries.append(entry)
            metrics=measured if ordinal == 0 else pair_metrics(*[outline(m) for m in masks(sample)])
            statistics.append(dict(pair_id=sample.pair_id, source_pair_id=source['pair_id'],
                label=bool(sample.label), recipe=recipe, mirror=axis,
                target_length_bin=wanted_bin, actual_length_bin=length_bin(metrics['d20_length_px']) if ordinal==0 else None,
                augmentation=report['compound'], matched_tokens=entry['inherited_match_count'],
                **metrics))
        result=dict(slot=slot, recipe=recipe, target_length_bin=wanted_bin,
            source_group_index=group_index, original_group_index=original_group,
            attempts=attempt+1, rejections=dict(failures), entries=entries, statistics=statistics)
        save_json(path, result)
        return result
    # No silent clean fallback: an unfinished quota is not a ready dataset.
    raise RuntimeError(json.dumps(dict(slot=slot, recipe=recipe, target_length_bin=wanted_bin,
        original_group_index=original_group, failures=dict(failures), attempts=options['attempts'])))


def summarize(records):
    entries=[e for r in records for e in r['entries']]
    rows=[s for r in records for s in r['statistics']]
    positive=[r for r in rows if r['label']]
    metrics={}
    for key in ('d20_length_px', 'd20_longest_px', 'd40_gap_mean_px', 'd10_breaks_mean',
                'mean_fragment_area_px', 'area_ratio', 'matched_tokens'):
        values=np.array([r[key] for r in positive if r.get(key) is not None], float)
        metrics[key]=dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
            p10=float(np.quantile(values,.1)), p25=float(np.quantile(values,.25)),
            p75=float(np.quantile(values,.75)), p90=float(np.quantile(values,.9)))
    return dict(sample_count=len(entries), positive_count=len(positive), negative_count=len(rows)-len(positive),
        recipe_counts=dict(Counter(r['recipe'] for r in records)),
        changed_counts=dict(Counter(('positive:' if e['label'] else 'negative:')+e['compound_recipe']
            for e in entries if e['changed_pair'])),
        actual_partial_fraction=sum(r['recipe'].startswith('partial') for r in records)/len(records),
        actual_compound_fraction=sum(r['statistics'][0]['augmentation']['stages_applied']>=2 for r in records)/len(records),
        mirror_counts=dict(Counter(str(r['mirror']) for r in rows)),
        source_strata=dict(Counter(('positive:' if e['label'] else 'negative:')+str(e.get('source_stratum')) for e in entries)),
        unique_base_pair_count=len(set(e['source_pair_id'] for e in entries)),
        max_base_pair_reuse=max(Counter(e['source_pair_id'] for e in entries).values()),
        source_replacement_groups=sum(r['source_group_index']!=r['original_group_index'] for r in records),
        generation_rejections=dict(sum((Counter(r['rejections']) for r in records), Counter())),
        positive_metrics=metrics,
        positive_break_fraction=float(np.mean([r['d10_breaks_mean']>0 for r in positive])),
        positive_gap2to10_fraction=float(np.mean([r['d40_gap_mean_px'] is not None and 2<=r['d40_gap_mean_px']<=10 for r in positive])),
        positive_area_below_quarter=float(np.mean([r['area_ratio']<.25 for r in positive])),
        positive_area_below_eighth=float(np.mean([r['area_ratio']<.125 for r in positive])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--s7-manifest', required=True); p.add_argument('--out', required=True)
    p.add_argument('--groups', type=int, default=12000); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=26092317); p.add_argument('--attempts', type=int, default=512)
    p.add_argument('--no-length-quotas', action='store_true', help='pilot diagnostics only')
    a=p.parse_args(); options=vars(a); options['length_quotas']=not options.pop('no_length_quotas')
    if a.groups <= 0 or a.groups > 12000 or a.groups%100 or not 1<=a.workers<=32:
        p.error('groups must be a multiple of100 within12000; workers1..32')
    if a.groups==12000 and not options['length_quotas']:
        p.error('formal data requires measured length quotas')
    root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True);options['out']=str(root)
    lock=root/'producer.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.time()
    protocol=dict(schema=SCHEMA, options=options, recipe_percent=RECIPE_PERCENT,
        source_manifest_sha256=digest(a.s7_manifest), contour_cap=512, canvas_size=800,
        inherited_correspondences_only=True, artificial_cuts_and_notches='ignore',
        positive_weather_scope='TRAIN GT seam; never real/test contours',
        negative_weather_scope='random locations on own whole outer contour',
        geometry_length_quota={'32to256':.25,'256to512':.50,'512to800':.25},
        joint_recipe_length_percent=RECIPE_LENGTH_PERCENT,
        generator_sha256={p.name:digest(p) for p in (Path(__file__), Path(__file__).with_name('geometry.py'))},
        length_metric='GT geometric near-contact proxy d20 with opposing normals',
        offline_mirror_fraction=.30, correspondence_identity_preserved=True,
        source_replacement='same positive and negative strata; independently augmented variants, not independent originals',
        real_statistics_used_for_recipe_design=True, turufan_seams_used=False,
        existing_training_val_test_files_modified=False, training_started=False)
    try:
        if (root/'protocol.json').exists() and read(root/'protocol.json')!=protocol:
            raise ValueError('existing output belongs to another recipe')
        save_json(root/'protocol.json',protocol)
        records=[]
        with ProcessPoolExecutor(a.workers,initializer=initialize,initargs=(options,)) as pool:
            for record in pool.map(materialize_slot,range(a.groups),chunksize=1):
                records.append(record)
                if len(records)%25==0 or len(records)==a.groups:
                    status=dict(status='running',pid=os.getpid(),completed_groups=len(records),
                        planned_groups=a.groups,elapsed_seconds=time.time()-started)
                    save_json(root/'status.json',status);print(json.dumps(status),flush=True)
        entries=[e for r in records for e in r['entries']]
        if len({e['pair_id'] for e in entries})!=2*a.groups:
            raise ValueError('duplicate generated pair IDs')
        stats=summarize(records)
        manifest=root/('train_s7c_24k.json' if a.groups==12000 else 'pilot_train.json')
        save_json(manifest,dict(schema_version=ARCHIVE_SCHEMA,split='train',artifact_root=str(root),
            entries=entries,protocol=protocol,stats=stats))
        v3entries=[dict(pair_id=e['pair_id'],sample_path=str(root/e['artifact_path']),
            target_metadata=e['target_metadata'],label=e['label'],source_pair_id=e['source_pair_id'],
            source_row=e['source_row'],recipe=e['compound_recipe']) for e in entries]
        save_json(root/'train.json',dict(entries=v3entries,protocol=protocol,stats=stats))
        with (root/'pair_metrics.jsonl').open('w') as f:
            for r in records:
                for s in r['statistics']:
                    f.write(json.dumps(s,ensure_ascii=False,allow_nan=False)+'\n')
        save_json(root/'summary.json',stats)
        status=dict(status='complete' if a.groups==12000 else 'pilot_complete',pid=os.getpid(),
            manifest=str(manifest),elapsed_seconds=time.time()-started,**stats)
        save_json(root/'status.json',status);print(json.dumps(status),flush=True)
    except BaseException as exc:
        save_json(root/'status.json',dict(status='failed',pid=os.getpid(),error=repr(exc),elapsed_seconds=time.time()-started))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
