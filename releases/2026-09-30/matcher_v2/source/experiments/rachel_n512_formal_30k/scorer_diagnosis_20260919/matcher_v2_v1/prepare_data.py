"""Compute official canonical hashes after full pixel audit; do not admit/train.

All TRAIN/SELECT/TEST files are checked for numerical identity and cross-split
collisions. TEST receives no model inference or geometry calibration. A later
bound population-acceptance receipt is mandatory to promote this preparation.
"""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from ..curriculum_training_v1.catalog import MATCHER_INPUTS, TARGETS, tensor_digest, legacy_numerical_digest
from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.model_adapter import bound_module, require
from .additive_exposure import StraightSampleRef
from .data_runtime import read_bound, straight_rows


def canonical_entry(entry, audit, split, seed, loader):
    path = Path(entry['sample_path'])
    require(path.is_absolute() and file_sha(path) == entry['sample_sha256'] == audit['sample_sha256'],
            'full-data sample file differs from its audit')
    sample, report = loader(path)
    require(sample.pair_id == entry['pair_id'] == audit['pair_id'] and float(sample.label) in (0., 1.)
            and bool(sample.label) == entry['label'] == audit['positive']
            and bool(report['pose_supervision_enabled']) == (bool(sample.label) and not report['changed_pair']),
            'official loader sample identity or pose policy differs')
    raw = hashlib.sha256()
    for name in MATCHER_INPUTS:
        raw.update(name.encode()); raw.update(np.ascontiguousarray(getattr(sample, name)).tobytes())
    require(raw.hexdigest() == audit['model_input_sha256'], 'generator raw input hash differs')
    require(entry['recipe'] in ('straight_M', 'straight_J', 'straight_R'), 'unexpected straight recipe')
    if bool(sample.label):
        na = int((sample.target_a >= 0).sum()); nb = int((sample.target_b >= 0).sum())
        require(na == nb == audit['target_audit']['correspondence_count'] >= 8,
                'healthy supervision count differs from independent audit')
    else:
        require(np.all(sample.target_a == -1) and np.all(sample.target_b == -1) and not sample.translation_valid,
                'negative correspondence/pose supervision changed')
    return dict(pair_id=sample.pair_id, label=bool(sample.label), recipe=entry['recipe'],
        sample_path=str(path), sample_sha256=entry['sample_sha256'],
        source_base_key=f'v42-parent:{split}:{seed}:{sample.pair_id}',
        original_donor_references=entry['attempted_donor_references'],
        actual_matcher_input_sha256=tensor_digest(sample, MATCHER_INPUTS),
        generator_raw_input_sha256=raw.hexdigest(),
        effective_training_target_sha256=digest(dict(materialized_targets=tensor_digest(sample, TARGETS), precise_recipe=False)),
        legacy_pixel_audit_numerical_sha256=legacy_numerical_digest(sample))


def prepare(full_spec, base_spec, source_root, loader=None):
    generation = read_bound(full_spec); base = read_bound(base_spec)
    require(generation.get('status') == 'complete_generation_integrity_supervision'
            and generation.get('total_rows') == 7800 and generation.get('files_unchanged') is True
            and generation.get('cross_split_model_input_overlap') == generation.get('cross_split_source_family_overlap') == 0,
            'full generation and independent audits must complete first')
    require(base.get('schema') == 'curriculum-data-admission/1' and base.get('status') == 'passed'
            and digest(base['catalog']) == base['catalog_sha256'], 'bound original admission required')
    loader = loader or bound_module('staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset', source_root).load_sample
    seen_ids = {r['pair_id'] for r in base['catalog']}; seen_inputs = {r['model_input_sha256'] for r in base['catalog']}
    train = []; populations = {}
    for split, count in [('train', 6000), ('select', 900), ('test', 900)]:
        spec = generation['datasets'][split]
        manifest = read_bound(dict(path=spec['manifest_path'], sha256=spec['manifest_sha256']))
        audit = read_bound(dict(path=spec['audit_path'], sha256=spec['audit_sha256']))
        require(manifest.get('split') == split and not manifest.get('failed')
                and len(manifest['entries']) == audit['rows'] == len(audit['records']) == count
                and audit.get('status') == 'passed_integrity_and_supervision'
                and audit.get('source_manifest_sha256') == spec['manifest_sha256'], 'split population/audit changed')
        lookup = {r['pair_id']: r for r in audit['records']}; rows = []
        require(len(lookup) == count, 'duplicate audited identities')
        for entry in manifest['entries']:
            row = canonical_entry(entry, lookup[entry['pair_id']], split, manifest['seed'], loader)
            require(row['pair_id'] not in seen_ids and row['actual_matcher_input_sha256'] not in seen_inputs,
                    'new data duplicates original/cross-split identity or actual inputs; do not silently drop exposures')
            seen_ids.add(row['pair_id']); seen_inputs.add(row['actual_matcher_input_sha256'])
            rows.append(row)
        populations[split] = dict(count=count, positives=sum(r['label'] for r in rows),
            manifest_sha256=spec['manifest_sha256'], audit_sha256=spec['audit_sha256'],
            canonical_row_identity_sha256=digest(rows), model_inferred=False)
        if split == 'train':train = sorted(rows, key=lambda r: r['pair_id'])
    refs = [StraightSampleRef(r['pair_id'], r['source_base_key'], r['actual_matcher_input_sha256'],
                            r['sample_path'], r['sample_sha256'], r['label']) for r in train]
    return dict(schema='straight-seam-canonical-preparation/1', status='ready_for_population_acceptance',
        training_admitted=False, full_generation_complete=full_spec, base_admission=base_spec,
        split='train', entries=train, catalog_sha256=digest([asdict(r) for r in refs]),
        populations=populations, canonical_cross_split_collisions=0, canonical_base_collisions=0,
        gpu_used=False, test_inferred=False, online_augmentation=False)


def promote(preparation_spec, acceptance_spec, out):
    """New output only; never mutate the candidate preparation or generated data."""
    row = read_bound(preparation_spec)
    require(row.get('schema') == 'straight-seam-canonical-preparation/1'
            and row.get('status') == 'ready_for_population_acceptance' and row.get('training_admitted') is False,
            'canonical preparation required before promotion')
    acceptance = read_bound(acceptance_spec)
    require(acceptance.get('canonical_preparation_sha256') == preparation_spec['sha256'],
            'population acceptance belongs to different canonical data')
    result = dict(row, schema='straight-seam-training-admission/1', status='passed', training_admitted=True,
                  canonical_preparation=preparation_spec, acceptance=acceptance_spec)
    # Validate the proposed receipt before writing a passed admission file.
    # All nested source files are still read and hash-checked normally.
    from .data_runtime import check_straight_record
    check_straight_record(result)
    write_json(Path(out), result)
    return result


def combine(base_spec, straight_spec, out):
    base = read_bound(base_spec); straight, refs = straight_rows(straight_spec)
    require(base.get('schema') == 'curriculum-data-admission/1' and base.get('status') == 'passed'
            and digest(base['catalog']) == base['catalog_sha256']
            and straight['base_admission'] == base_spec, 'original admission binding differs')
    catalog = list(base['catalog']) + [asdict(r) for r in refs]
    require(len({r['pair_id'] for r in catalog}) == len(catalog)
            and len({r['model_input_sha256'] for r in catalog}) == len(catalog), 'combined catalog has duplicates')
    result = dict(schema='matcher-v2-data-admission/1', status='passed', training_admitted=True,
        base_admission=base_spec, straight_admission=straight_spec, original_exposures_removed=0,
        catalog=catalog, catalog_sha256=digest(catalog), gpu_used=False)
    write_json(Path(out), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation-complete', type=Path, required=True)
    p.add_argument('--base-admission', type=Path, required=True)
    p.add_argument('--output-new', type=Path, required=True)
    args = p.parse_args()
    bind = lambda path: dict(path=str(path.resolve()), sha256=file_sha(path))
    result = prepare(bind(args.generation_complete), bind(args.base_admission), Path(__file__).resolve().parents[4])
    write_json(args.output_new, result)
    print({k: v for k, v in result.items() if k not in ('entries', 'populations')})


if __name__ == '__main__':main()
