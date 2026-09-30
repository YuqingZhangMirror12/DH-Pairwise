"""Retain the same original v14 pair after a rejected v17 pixel audit.

No augmentation search, re-pairing, label change or tolerance change occurs.
The rejected artifacts are preserved before the atomic group commit changes.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil

GAP_ERROR = 'final PRIMARY footprint peak5–35px including weak'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def eligible(error, group):
    # Unexpected corruption, label/source mismatches etc. must still stop.
    return (type(error) is AssertionError and str(error) == GAP_ERROR
            and group.get('status') == 'committed'
            and group.get('v14_fallback') is False
            and len(group.get('records', [])) == 2)


def check_pair(group):
    slot = group['slot']
    if (group['task']['slot'] != slot or group['actual_positive_slot'] != slot
            or group['positive_replaced'] or len(group['records']) != 2):
        raise ValueError('only the original paired group may be retained')
    if [r['label'] for r in group['records']] != [True, False]:
        raise ValueError('paired positive/negative labels differ')
    for ordinal, record in enumerate(group['records']):
        if record['baseline_slot'] != slot or record['baseline_ordinal'] != ordinal:
            raise ValueError('baseline identity differs')


def artifact_paths(group, split_root):
    result = set()
    split_root = Path(split_root).resolve()
    for record in group['records']:
        for key in ('sample_path', 'proof_path', 'target_metadata'):
            path = Path(record[key]).resolve()
            path.relative_to(split_root)
            result.add(path)
        if record.get('latent_seam_artifact'):
            path = (split_root / record['latent_seam_artifact']).resolve()
            path.relative_to(split_root)
            result.add(path)
    return sorted(result)


def archive(group_path, history):
    group_path, history = Path(group_path).resolve(), Path(history).resolve()
    group = read(group_path)
    check_pair(group)
    history.mkdir(parents=True, exist_ok=False)
    split_root = group_path.parent.parent
    files = [group_path] + artifact_paths(group, split_root)
    preserved = {}
    for path in files:
        rel = path.relative_to(split_root)
        target = history / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest = sha(path)
        if sha(target) != digest:
            raise ValueError('rejected artifact archive differs')
        preserved[str(rel)] = digest
    save(history / 'preserved.json', preserved)
    return preserved


def build(group, out_split):
    """Use the original v14 arrays and original supervision, not v17 masks."""
    import numpy as np
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import save_sample
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.aggressive_data_full_v17 import generate as gen
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.aggressive_data_full_v17.source import STATE, baseline
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.seam_context_v3.targets import known_gap_links
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.seam_context_v3.prepare import family

    check_pair(group)
    out_split = Path(out_split).resolve()
    if out_split.exists():
        raise ValueError('fresh fallback staging directory required')
    out_split.mkdir(parents=True)
    updated = deepcopy(group)
    updated.update(v14_fallback=True, actual_endpoint_mode=None,
                   independent_audit_fallback=dict(reason=GAP_ERROR, new_geometry_attempts=0,
                       same_original_pair=True, labels_unchanged=True))
    for ordinal, record in enumerate(updated['records']):
        row = baseline(group['slot'], ordinal)
        old, original = row['old_sample'], row['original']
        if str(STATE['root'] / row['entry']['artifact_path']) != record['baseline_sample_path']:
            raise ValueError('bound original archive changed')
        pid = record['id']
        sample = replace(old, pair_id=pid, fragment_a_token=old.fragment_a_token + '@' + pid,
                         fragment_b_token=old.fragment_b_token + '@' + pid)
        detail = dict(v14_fallback=True, reason='independent v17 pixel audit rejected; retain same original v14 pair',
                      trim=None, gap=None, primary_damage={}, background={},
                      planned_size_class=group['task']['size_class'], planned_gap_count=group['task']['k'],
                      actual_new_cut_applied=False)
        report = deepcopy(row['old_report'])
        report.update(schema_version='aggressive-v17-full/1', recipe=record['recipe'],
                      source_pair_id=record['source_pair_id'], base_v14_pair_id=record['baseline_pair_id'],
                      augmentation_revision=record['augmentation_revision'], paired_review=detail,
                      not_full_training_dataset=False, v14_fallback=True)
        stem = f'{group["slot"]:05d}_{ordinal}.npz'
        sample_path = out_split / 'samples' / stem
        sample_path.parent.mkdir(exist_ok=True)
        save_sample(sample_path, sample, report)
        arrays = {}
        for stage, value in [('fragment', original), ('trim', original),
                             ('primary', row['old_primary']), ('final', sample)]:
            for side in 'ab':
                arrays['packed_' + stage + '_' + side] = np.packbits(getattr(value, 'mask_' + side)[0].astype(bool), axis=1)
        latent = None
        if ordinal == 0:
            gap = {}
            previous = row['entry'].get('latent_seam_artifact')
            if previous:
                with np.load(STATE['root'] / previous, allow_pickle=False) as data:
                    gap = {k: data[k].copy() for k in data.files}
            arrays.update({'gap_' + k: v for k, v in gap.items()})
            latent = out_split / 'latent' / stem
            gen.atomic_npz(latent, **gap)
        proof = out_split / 'proof' / stem
        gen.atomic_npz(proof, **arrays)
        target = out_split / 'targets' / stem
        gen.atomic_npz(target, **known_gap_links(sample, original, report))
        partial = report['compound']['partial'] or {}
        donors = [partial['donor_lineage']] if partial.get('donor_lineage') else []
        record.update(detail=detail, v14_fallback=True, requested_gap_count=0,
                      sample_path=str(sample_path), artifact_path=str(sample_path.relative_to(out_split)),
                      sample_sha256=sha(sample_path), proof_path=str(proof), proof_sha256=sha(proof),
                      target_metadata=str(target), target_metadata_sha256=sha(target),
                      latent_seam_artifact=str(latent.relative_to(out_split)) if latent else None,
                      latent_sha256=sha(latent) if latent else None,
                      augmentation_donor_sources=donors, donor_families=sorted({family(x) for x in donors}),
                      inherited_correspondences=int((sample.target_a >= 0).sum()),
                      inherited_match_count=int((sample.target_a >= 0).sum()), changed_pair=report['changed_pair'])
    return updated


def promote(group_path, staged, staged_split, history):
    """A failed interruption remains diagnosable; never silently re-run."""
    group_path, staged_split, history = Path(group_path).resolve(), Path(staged_split).resolve(), Path(history).resolve()
    split_root = group_path.parent.parent
    old_hash = read(history / 'preserved.json')['groups/' + group_path.name]
    if sha(group_path) != old_hash:
        raise ValueError('group changed since fallback archive')
    group = deepcopy(staged)
    for source in artifact_paths(staged, staged_split):
        target = split_root / source.relative_to(staged_split)
        temp = target.with_name(target.name + '.fallback.' + str(os.getpid()))
        shutil.copy2(source, temp)
        if sha(temp) != sha(source):
            raise ValueError('fallback promotion copy differs')
        os.replace(temp, target)
    for record in group['records']:
        for key in ('sample_path', 'proof_path', 'target_metadata'):
            record[key] = str(split_root / Path(record[key]).resolve().relative_to(staged_split))
    save(group_path, group)
    return group
