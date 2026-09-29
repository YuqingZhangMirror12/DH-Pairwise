"""Transactional per-group generation, bounded retries, no training mutation."""
from collections import Counter
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import hashlib, os, time, traceback
import numpy as np
from . import REVISION, SEED
from .source import STATE, baseline
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
    out = Path(config['out'])/split; commit = out/'groups'/f'{slot:05d}.json'
    plan_sha = config['plan_hashes'][split]
    if commit.exists():
        group = verify_group(commit, task, plan_sha)
        return dict(slot=slot, status='committed', cached=True, seconds=group['seconds'])
    reasons = Counter(); attempt_log = []; result = None
    negative = baseline(slot, 1)
    if task['positive_candidates'] != [slot]:raise ValueError('user forbids forced positive replacement')
    for candidate_index, candidate in enumerate(task['positive_candidates']):
        positive = baseline(candidate, 0); pair = (positive, negative)
        for attempt in range(12 if candidate_index == 0 else 8):
            try:
                rng = rng_for(SEED, REVISION, split, slot, candidate_index, attempt)
                result = augment(pair, rng, attempt_mode(task,attempt), STATE['bank'], task['size_class'], task['k'] or 1)
                break
            except ValueError as error:
                reason = str(error); reasons[reason] += 1
                attempt_log.append(dict(candidate=candidate, attempt=attempt, mode=attempt_mode(task,attempt), reason=reason))
        if result is not None: break
    fallback=result is None
    if fallback:
        final=tuple(row['old_sample'] for row in pair)
        details=[dict(v14_fallback=True,reason='no admissible v17 result for original pair; user-authorized retention',
            trim=None,gap=None,primary_damage={},background={},planned_size_class=task['size_class'],
            planned_gap_count=task['k'],actual_new_cut_applied=False) for _ in pair]
        fields=[dict(primary={},light={}) for _ in pair]
        trimmed=tuple(row['original'] for row in pair);primary=tuple(row['old_primary'] for row in pair)
        gap={}
        lp=pair[0]['entry'].get('latent_seam_artifact')
        if lp:
            with np.load(STATE['root']/lp,allow_pickle=False) as z:gap={k:z[k].copy() for k in z.files}
    else:
        final, details, fields, gap, trimmed, primary = result
    records = []
    for ordinal, (row, sample, detail) in enumerate(zip(pair, final, details)):
        entry = row['entry']; base = row['original']; recipe = task['recipe']
        pid = f'v17g35-{split}-{slot:05d}-{ordinal}'
        sample = replace(sample, pair_id=pid, fragment_a_token=sample.fragment_a_token+'@'+pid,
                         fragment_b_token=sample.fragment_b_token+'@'+pid)
        report = deepcopy(row['old_report']) if fallback else changed_report(base, sample, recipe)
        report.update(schema_version='aggressive-v17-full/1', recipe=recipe,
            source_pair_id=entry['source_pair_id'], base_v14_pair_id=entry['pair_id'],
            augmentation_revision=REVISION, paired_review=detail, not_full_training_dataset=False,v14_fallback=fallback,
            compound=deepcopy(row['old_report']['compound']) if fallback else dict(recipe=recipe, damage=detail['primary_damage'], partial=row['old_report']['compound']['partial']),
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
        donors = [] if fallback else [detail['trim']['donor']['lineage']]
        partial = report['compound']['partial'] or {}
        if partial.get('donor_lineage'): donors.append(partial['donor_lineage'])
        r = dict(id=pid, pair_id=pid, label=bool(sample.label), recipe=recipe, corrosion_recipe=recipe,
            corrosion_category=entry['corrosion_category'], partial_applied=recipe=='partial',
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
            baseline_slot=row['slot'], baseline_ordinal=ordinal,
            baseline_sample_path=str(STATE['root']/entry['artifact_path']), baseline_pair_id=entry['pair_id'],
            inherited_correspondences=int((sample.target_a>=0).sum()),
            inherited_match_count=int((sample.target_a>=0).sum()),
            changed_pair=report['changed_pair'], augmentation_donor_sources=donors,
            source_families=sorted({family(source_row['fragment_'+s]['split_unit_id']) for s in 'ab'}),
            donor_families=sorted({family(s) for s in donors}))
        records.append(r)
    receipt = dict(status='committed', split=split, slot=slot, task=task, plan_sha256=plan_sha,
        actual_endpoint_mode=None if fallback else details[0]['trim']['mode'],v14_fallback=fallback,
        actual_positive_slot=candidate, positive_replaced=candidate!=slot, records=records,
        rejections=dict(reasons), attempts=attempt_log, seconds=time.monotonic()-start)
    save_json(commit, receipt)
    return dict(slot=slot, status='committed', seconds=receipt['seconds'], positive_replaced=candidate!=slot,
                attempts=len(attempt_log)+1)
