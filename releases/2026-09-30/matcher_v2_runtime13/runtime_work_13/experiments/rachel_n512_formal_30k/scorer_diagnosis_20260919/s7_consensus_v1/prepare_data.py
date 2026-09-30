"""Bind completed strong TRAIN and source-disjoint synthetic CAL/SELECT.

This records a new experiment contract. It never regenerates or augments data,
changes labels, reads real examples, or silently drops overlapping sources.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ..seam_context_v3.prepare import sources,family
from .matcher import SOURCE_TRAIN_SHA


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_set(entries):
    result=set()
    for entry in entries:
        values=(sources(entry['source_row']) if 'source_row' in entry
                else {family(s) for s in entry['sources']})
        if not values or '' in values:
            raise ValueError('missing source family identity')
        values=values|{family(s) for s in entry.get('augmentation_donor_sources',[])}
        result.update(values)
    return result


def load_validation_plan(path, forbidden_sources):
    """Inspect an explicit completed6000-instance plan; never generate data."""
    path=Path(path).resolve();plan=read(path)
    has_test=plan.get('schema')=='s7-consensus-validation-plan/3'
    expected_splits={'cal','select','test'} if has_test else {'cal','select'}
    if (plan.get('schema') not in ('s7-consensus-validation-plan/2','s7-consensus-validation-plan/3')
            or plan.get('kind')!='single_mixed' or plan.get('status')!='passed'
            or plan.get('physical_samples')!=6000 or set(plan.get('splits',{}))!=expected_splits):
        raise ValueError('explicit completed6K validation plan required')
    manifests,lineage,all_ids={},{},set()
    for split,spec in plan['splits'].items():
        manifest=(path.parent/spec['manifest']).resolve()
        if sha(manifest)!=spec['manifest_sha256']:
            raise ValueError('validation manifest changed after generation audit')
        entries=read(manifest)['entries'];expected=spec['pair_count']
        if has_test and expected!={'cal':1500,'select':1500,'test':3000}[split]:
            raise ValueError('new30K protocol requires1500CAL/1500SELECT/3000TEST')
        ids=[e['pair_id'] for e in entries]
        if type(expected) is not int or expected<=0 or len(ids)!=expected or len(set(ids))!=expected:
            raise ValueError('missing/duplicate validation instances: '+split)
        if all_ids&set(ids):raise ValueError('CAL and SELECT share Pair IDs')
        all_ids.update(ids)
        if expected%2 or Counter(int(e['label']) for e in entries)!={0:expected//2,1:expected//2}:
            raise ValueError('each validation split must retain balanced positive/negative instances')
        if any(e.get('source_row',{}).get('split')=='train' for e in entries):
            raise ValueError('validation contains a TRAIN release row')
        base_ids={e['source_pair_id'] for e in entries}
        if '' in base_ids or None in base_ids:
            raise ValueError('validation base Pair identity missing')
        name=split+'_mixed';lineage[name]=source_set(entries)
        overlap=lineage[name]&forbidden_sources
        if overlap:raise ValueError('validation source overlaps training: '+repr(sorted(overlap)))
        audits={}
        for key in ('validation','pixel_audit'):
            audit_path=(path.parent/spec[key+'_path']).resolve()
            if sha(audit_path)!=spec[key+'_sha256']:
                raise ValueError('validation audit receipt changed')
            audit=read(audit_path)
            if audit.get('status')!='passed' or audit.get('pairs')!=expected:
                raise ValueError('incomplete validation audit: '+split+' '+key)
            if key=='pixel_audit':
                if (audit.get('all_actual_masks_reconstructed') is not True
                        or audit.get('topology_checked_all') is not True
                        or len(audit.get('rows',[]))!=expected
                        or {r['pair_id'] for r in audit['rows']}!=set(ids)):
                    raise ValueError('pixel audit does not cover the actual validation population')
            audits[key]=dict(path=str(audit_path),sha256=sha(audit_path))
        manifests[name]=dict(path=str(manifest),sha256=sha(manifest),pair_count=expected,
            source_count=len(lineage[name]),base_pair_count=len(base_ids),audits=audits)
    if len(all_ids)!=6000:
        raise ValueError('validation plan does not contain6000 distinct instances')
    if lineage['cal_mixed']&lineage['select_mixed']:
        raise ValueError('CAL and SELECT share source families')
    if has_test and lineage['test_mixed']&(lineage['cal_mixed']|lineage['select_mixed']):
        raise ValueError('TEST shares source families with CAL/SELECT')
    design=dict(kind='single_mixed',physical_samples=3000 if has_test else 6000,
        total_heldout_samples=6000,test_physical_samples=3000 if has_test else 0,
        plan_path=str(path),plan_sha256=sha(path),
        selection='single mixed SELECT population; recipe diagnostics do not reweight selection',
        independent_unit='original manuscript family; generated instances and base pairs reported separately',
        source_counts={split:len(lineage[split+'_mixed']) for split in ('cal','select')})
    return manifests,lineage,design


def prepare(train_root, validation_root, s7_training_manifest, out,validation_plan=None):
    train_root, validation_root, out = map(Path, (train_root, validation_root, out))
    if out.exists():
        raise ValueError('do not overwrite an experiment data contract')
    status = read(train_root/'status.json')
    validation = read(train_root/'validation.json')
    pipeline = read(train_root/'pipeline_status.json')
    if ((train_root/'failure.json').exists() or status.get('status') != 'complete'
            or pipeline.get('status') != 'complete' or pipeline.get('stage') != 'data_ready'
            or status.get('sample_count') != 24000 or validation.get('status') != 'passed'
            or not validation.get('source_identity_and_anchor_quotas_checked')):
        raise ValueError('full24K must pass generation, full quotas, validation and measurement')
    if sha(s7_training_manifest) != SOURCE_TRAIN_SHA:
        raise ValueError('historical S7 M12 TRAIN binding differs')
    records = {'train': read(train_root/'train.json')}
    profile=records['train'].get('protocol',{}).get('distribution_revision',{})
    if profile.get('partial_min_smaller_perimeter_fraction') and validation_plan is None:
        raise ValueError('approved v14 experiment requires the new6K validation plan, not old82/370')
    old = read(s7_training_manifest)
    train = records['train']['entries']
    if len(train) != 24000 or Counter(int(e['label']) for e in train) != {0:12000,1:12000}:
        raise ValueError('unexpected training population')
    if len({e['pair_id'] for e in train}) != len(train):
        raise ValueError('duplicate training pair IDs')
    if any(e['source_row']['split'] != 'train' for e in train):
        raise ValueError('a TRAIN example came from a non-TRAIN release row')
    lineage = {'train': source_set(train), 'historical_s7_train': source_set(old['entries'])}
    manifests = {};design=None
    if validation_plan:
        manifests,validation_lineage,design=load_validation_plan(validation_plan,
            lineage['train']|lineage['historical_s7_train'])
        lineage.update(validation_lineage)
    for name in (() if validation_plan else ('cal_clean', 'cal_hard', 'select_clean', 'select_hard')):
        path = validation_root/(name+'.json')
        record = read(path)
        entries = record['entries']
        expected = 82 if name.startswith('cal_') else 370
        if len(entries) != expected or len({e['pair_id'] for e in entries}) != expected:
            raise ValueError('changed or duplicate CAL/SELECT population: '+name)
        lineage[name] = source_set(entries)
        overlap = lineage[name] & (lineage['train'] | lineage['historical_s7_train'])
        if overlap:
            raise ValueError('validation source overlaps training: '+name+' '+repr(sorted(overlap)))
        records[name] = record
        manifests[name] = dict(path=str(path.resolve()), sha256=sha(path), pair_count=expected,
                               source_count=len(lineage[name]))
    if not validation_plan and ((lineage['cal_clean'] | lineage['cal_hard']) &
            (lineage['select_clean'] | lineage['select_hard'])):
        raise ValueError('CAL and SELECT share source families')
    for stem in (() if validation_plan else ('cal', 'select')):
        a = records[stem+'_clean']['entries']
        b = records[stem+'_hard']['entries']
        if [e['pair_id'] for e in a] != [e['pair_id'] for e in b]:
            raise ValueError('clean/hard pair order differs; use explicit paired views')
    test=manifests.pop('test_mixed',None)
    result = dict(status='passed', schema='s7-consensus-data-contract/'+('3' if test else '2' if validation_plan else '1'),
        train=dict(path=str((train_root/'train.json').resolve()), sha256=sha(train_root/'train.json'),
                   archive_manifest=str((train_root/'train_s7b_24k.json').resolve()),
                   archive_manifest_sha256=sha(train_root/'train_s7b_24k.json'),
                   pairs=24000, positives=12000, negatives=12000,
                   source_count=len(lineage['train']), validation_sha256=sha(train_root/'validation.json')),
        historical_s7_train=dict(path=str(Path(s7_training_manifest).resolve()), sha256=SOURCE_TRAIN_SHA,
                                 source_count=len(lineage['historical_s7_train'])),
        validation=manifests, online_mirror_probability=0., offline_mirror_probability=.15,
        views_are_paired_not_independent=not bool(validation_plan), source_disjoint=True,
        validation_hard_caveat=('6000 augmented instances are not6000 independent source manuscripts.' if validation_plan
            else 'Reused frozen v3 hard views; some documented augmentation fallbacks remain.'),
        no_real_images_or_predictions_read=True,
        development_caveat='Dunhuang aggregate area/gap statistics informed TRAIN generation; do not call it an untouched benchmark.',
        translation_note='Corroded positive archives retain valid global GT; old pose_supervision_enabled only disabled the old boundary-residual loss, not candidate-quality GT.',
        source_families={k:sorted(v) for k,v in lineage.items()})
    if design is not None:result['validation_design']=design
    if test is not None:
        result['test']=dict(mixed=test)
        result['test_policy']='Frozen final evaluation only; never read by training validation, threshold calibration or epoch selection.'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','source_disjoint','views_are_paired_not_independent')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train-root','validation-root','s7-training-manifest','out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--validation-plan',help='Explicit new6K completed/audited CAL-SELECT manifest plan')
    a = parser.parse_args()
    prepare(a.train_root, a.validation_root, a.s7_training_manifest, a.out,a.validation_plan)
