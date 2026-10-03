"""Transactional per-group generation, bounded retries, no training mutation."""
from collections import Counter
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import hashlib, os, time, traceback
import numpy as np
from . import REVISION, SEED
from .spec import SPECS
from ..aggressive_data_v16.run import tags
from .audit import audit_record
from ..aggressive_data_full_v17.source import STATE, baseline
from .geometry import augment
from ..s7_compound_v1.materialize import read, save_json, digest
from ..s7_compound_v1.geometry import rng_for
from ..seam_context_v3.targets import known_gap_links
from ..seam_context_v3.prepare import family
from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import changed_report
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import save_sample, load_sample


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as f: np.savez_compressed(f, **arrays)
    temp.replace(path)


def verify_group(path, task, plan_sha):
    r = read(path)
    if r['task'] != task or r['plan_sha256'] != plan_sha or r['status'] != 'committed':
        raise ValueError('resume task/source binding changed')
    if len(r['records']) != 2:
        raise ValueError('incomplete committed group')
    for e in r['records']:
        for key in ('sample', 'proof', 'target_metadata'):
            p = e[key+'_path'] if key != 'target_metadata' else e[key]
            if digest(p) != e[key+'_sha256']:
                raise ValueError('committed artifact changed:'+key)
    return r


def attempt_mode(task, attempt):
    """Both approved end modes are admissible; neither has a user-set quota.

    Side, recipe, notch count and all pixel gates remain fixed. This prevents
    an arbitrary slot parity from making a valid 70/30 quota impossible.
    """
    if set(task.get('allowed_modes', [])) != {'one','both'}:
        raise ValueError('only the two approved endpoint modes may be sampled')
    first=task['mode']
    if first not in ('one','both'):raise ValueError('invalid preferred endpoint mode')
    return first if attempt % 2 == 0 else ('both' if first == 'one' else 'one')


def process(task):
    start = time.monotonic(); config = STATE['config']; split = STATE['split']; slot = task['slot']
    version=task['version']; spec=SPECS[version]
    out = Path(config['out'])/version; commit = out/'groups'/f'{slot:05d}.json'
    plan_sha = config['plan_sha256']
    if commit.exists():
        group = verify_group(commit, task, plan_sha)
        return dict(slot=slot, status='committed', cached=True, seconds=group['seconds'],records=group['records'],audit_rows=group['audit_rows'])
    reasons = Counter(); attempt_log = []; result = None
    negative = baseline(slot, 1)
    if task['positive_candidates'] != [slot]:raise ValueError('user forbids forced positive replacement')
    for candidate_index, candidate in enumerate(task['positive_candidates']):
        positive = baseline(candidate, 0); pair = (positive, negative)
        for attempt in range(int(config.get('geometry_attempts',10))):
            try:
                rng = rng_for(SEED, REVISION, version, slot, candidate_index, attempt)
                result = augment(pair, rng, attempt_mode(task,attempt), STATE['bank'], task['size_class'], task['k'] or 1,spec,task['trim_target'])
                break
            except ValueError as error:
                reason = str(error); reasons[reason] += 1
                attempt_log.append(dict(candidate=candidate, attempt=attempt, mode=attempt_mode(task,attempt), reason=reason))
        if result is not None: break
    if result is None:
        return dict(slot=slot,version=version,size_class=task['size_class'],status='rejected',
            reasons=dict(reasons),attempts=attempt_log,seconds=time.monotonic()-start,
            original_v14_preserved=True,counts_as_hard_review=False)
    fallback=False
    final,details,fields,gap,trimmed,primary=result
    source_group=read(STATE['root']/'groups'/f'{slot:05d}.json')
    records = []
    for ordinal, (row, sample, detail) in enumerate(zip(pair, final, details)):
        entry = row['entry']; base = row['original']; recipe = task['recipe']
        purpose='full' if config.get('full_generation_authorized',False) else 'probe'
        pid = f'{version}-r6-{purpose}-{slot:05d}-{ordinal}'
        sample = replace(sample, pair_id=pid, fragment_a_token=sample.fragment_a_token+'@'+pid,
                         fragment_b_token=sample.fragment_b_token+'@'+pid)
        report = deepcopy(row['old_report']) if fallback else changed_report(base, sample, recipe)
        report.update(schema_version='curriculum-review/1', recipe=recipe,
            source_pair_id=entry['source_pair_id'], base_v14_pair_id=entry['pair_id'],
            augmentation_revision=REVISION, paired_review=detail, not_full_training_dataset=not config.get('full_generation_authorized',False),v14_fallback=fallback,
            compound=deepcopy(row['old_report']['compound']) if fallback else dict(recipe=recipe, damage=detail['primary_damage'],
                partial=row['old_report']['compound']['partial'] if detail['partial_crop_applied'] else None),
            pair_shared_scale=row['old_report']['pair_shared_scale'])
        for side in 'ab': report['side_'+side].update(detail['primary_damage'].get(side, {}))
        path = out/'samples'/f'{slot:05d}_{ordinal}.npz'; save_sample(path, sample, report)
        loaded, _ = load_sample(path)
        for key in ('mask_a','mask_b','target_a','target_b','points_rc_a','points_rc_b'):
            if not np.array_equal(getattr(loaded,key), getattr(sample,key)):
                raise ValueError('archive roundtrip changed '+key)
        arrays = {}
        for stage, ss in [('fragment',base),('trim',trimmed[ordinal]),('primary',primary[ordinal]),('final',sample)]:
            for side in 'ab': arrays['packed_'+stage+'_'+side] = np.packbits(getattr(ss,'mask_'+side)[0].astype(bool),axis=1)
        for kind, values in fields[ordinal].items(): arrays.update({kind+'_'+key:value for key,value in values.items()})
        if ordinal == 0: arrays.update({'gap_'+key:value for key,value in gap.items()})
        proof = out/'proof'/path.name; atomic_npz(proof, **arrays)
        latent = None
        if ordinal == 0:
            latent = out/'latent'/path.name; atomic_npz(latent, **gap)
        target = out/'targets'/path.name; atomic_npz(target, **known_gap_links(sample, base, report))
        source_row = dict(entry['source_row'], pair_id=pid)
        donors = [detail['trim']['donor']['lineage']] if detail['trim'].get('applied',True) else []
        partial = report['compound']['partial'] or {}
        if partial.get('donor_lineage'): donors.append(partial['donor_lineage'])
        r = dict(version=version,id=pid, pair_id=pid, label=bool(sample.label), recipe=recipe, corrosion_recipe=recipe,
            corrosion_category=entry['corrosion_category'], partial_applied=detail['partial_crop_applied'],
            source_pair_id=entry['source_pair_id'], source_root=entry['source_root'], source_row=source_row,
            source_stratum=entry['source_stratum'], negative_kind=entry['negative_kind'],
            offline_paired_mirror=entry['offline_paired_mirror'], mirror=entry['offline_paired_mirror'],
            sample_path=str(path), artifact_path=str(path.relative_to(out)), proof_path=str(proof),
            sample_sha256=digest(path), proof_sha256=digest(proof), target_metadata=str(target),
            target_metadata_sha256=digest(target),
            latent_seam_artifact=str(latent.relative_to(out)) if latent else None,
            latent_sha256=digest(latent) if latent else None,
            detail=detail, requested_gap_count=0 if fallback else task['k'], augmentation_revision=REVISION,
            v14_fallback=fallback,planned_size_class=task['size_class'],planned_gap_count=task['k'],
            source_base_key=task['base_keys'][ordinal],
            baseline_slot=row['slot'], baseline_ordinal=ordinal,
            baseline_sample_path=str(STATE['root']/entry['artifact_path']), baseline_pair_id=entry['pair_id'],
            inherited_correspondences=int((sample.target_a>=0).sum()),
            inherited_match_count=int((sample.target_a>=0).sum()),
            changed_pair=report['changed_pair'], augmentation_donor_sources=donors,
            source_families=sorted({family(source_row['fragment_'+s]['split_unit_id']) for s in 'ab'}),
            donor_families=sorted({family(s) for s in donors}))
        r['groups']=tags(entry,source_group['statistics'][ordinal])
        if recipe=='partial' and not detail['partial_crop_applied']:
            r['groups']=[x for x in r['groups'] if not x.startswith('partial_')]+['partial_skipped_crop20']
        r['groups']+=['trim_'+detail['trim']['mode']]
        if detail['trim'].get('applied',True):r['groups'].append('crop_'+detail['trim']['size_class'])
        if recipe!='clean':r['groups'].append('light70')
        if recipe.startswith('gaps'):r['groups'].append('notches_'+str(task['k']))
        if sample.label and detail['evidence_islands']['two_to_four_bilateral']:
            r['groups'].append('evidence_2to4')
        if sample.label and detail['gap']['primary_gap_peak_px'] is not None and detail['gap']['primary_gap_peak_px']>=20.:
            r['groups'].append('gap_ge20')
        records.append(r)
    receipt = dict(status='committed', split=split, slot=slot, task=task, plan_sha256=plan_sha,
        actual_endpoint_mode=None if fallback else details[0]['trim']['mode'],v14_fallback=fallback,
        actual_positive_slot=candidate, positive_replaced=candidate!=slot, records=records,
        rejections=dict(reasons), attempts=attempt_log, seconds=time.monotonic()-start)
    # Independent reconstruction, actual target-loader and pixels BEFORE commit.
    receipt['audit_rows']=[audit_record(r,STATE['root']) for r in records]
    receipt['status']='committed'
    save_json(commit, receipt)
    return dict(slot=slot,version=version,size_class=task['size_class'],status='passed',records=records,audit_rows=receipt['audit_rows'],seconds=time.monotonic()-start,reasons=dict(reasons),attempts=len(attempt_log)+1)
