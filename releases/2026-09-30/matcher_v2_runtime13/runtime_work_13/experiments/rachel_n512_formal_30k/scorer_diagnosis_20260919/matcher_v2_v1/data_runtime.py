"""Immutable base-plus-straight loader; never infer admission from generation.

The generator's raw-byte input hash is not the curriculum tensor hash. The
straight training admission must explicitly contain canonical six-input and
effective-target hashes produced by the official loader before this is usable.
No augmentation, new targets, GT filtering or source repair happens at loading.
"""
from dataclasses import asdict
import json
from pathlib import Path

from ..curriculum_training_v1.catalog import MATCHER_INPUTS, TARGETS, tensor_digest, legacy_numerical_digest
from ..curriculum_training_v1.checkpoint_io import file_sha
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.model_adapter import AdmittedDataset, bound_module, require
from .additive_exposure import StraightSampleRef
from .admission_constraints import population_counts, verify_lineage


def read_bound(spec):
    require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}, 'explicit file/SHA binding required')
    path = Path(spec['path'])
    require(path.is_absolute() and file_sha(path) == spec['sha256'], 'admitted artifact changed')
    return json.loads(path.read_text())


def straight_rows(spec):
    record = read_bound(spec)
    return check_straight_record(record)


def check_straight_record(record):
    require(record.get('schema') == 'straight-seam-training-admission/1'
            and record.get('status') == 'passed' and record.get('training_admitted') is True
            and record.get('gpu_used') is False and record.get('split') == 'train'
            and record.get('online_augmentation') is False,
            'completed straight training admission required, not generation/integrity alone')
    acceptance = read_bound(record['acceptance'])
    require(acceptance.get('schema') == 'straight-seam-population-acceptance/1'
            and acceptance.get('status') == 'passed'
            and acceptance.get('calibration_split') == 'select'
            and acceptance.get('test_used_for_selection') is False
            and acceptance.get('known_real_donors_excluded') is True
            and acceptance.get('source_split_overlap') == 0
            and acceptance.get('full_generation_complete_sha256') == record['full_generation_complete']['sha256'],
            'bound SELECT calibration/source/population acceptance required')
    generation = read_bound(record['full_generation_complete'])
    counts = population_counts(generation)
    require(generation.get('status') == 'complete_generation_integrity_supervision'
            and generation.get('cross_split_model_input_overlap') == 0
            and generation.get('cross_split_source_family_overlap') == 0
            and generation.get('cross_split_pre_damage_pair_overlap') == 0,
            'all three generated splits must have completed their independent audit')
    lineage = verify_lineage(record, acceptance, generation, read_bound)
    train = generation['datasets']['train']
    manifest = read_bound(dict(path=train['manifest_path'], sha256=train['manifest_sha256']))
    audit = read_bound(dict(path=train['audit_path'], sha256=train['audit_sha256']))
    require(manifest.get('split') == 'train' and not manifest.get('failed')
            and audit.get('status') == 'passed_integrity_and_supervision'
            and audit.get('source_manifest_sha256') == train['manifest_sha256'],
            'original straight manifest/audit identity changed')
    require(len(record['entries']) == len(manifest['entries']) == len(audit['records']) == counts['train']
            and audit.get('authorized_omissions') == train['omitted'],
            'straight TRAIN must contain every explicitly admitted row')
    original = {r['pair_id']: r for r in manifest['entries']}
    audited = {r['pair_id']: r for r in audit['records']}
    refs = []; seen = set(); hashes = set()
    for row in record['entries']:
        name = row['pair_id']; require(name not in seen and name in original and name in audited, 'row membership changed')
        seen.add(name); source = original[name]; proof = audited[name]
        require(row['label'] == source['label'] and type(row['label']) is bool
                and row['sample_path'] == source['sample_path']
                and row['sample_sha256'] == source['sample_sha256'] == proof['sample_sha256']
                and row['recipe'] == source['recipe'] and row['recipe'] in ('straight_M', 'straight_J', 'straight_R')
                and row['generator_raw_input_sha256'] == proof['model_input_sha256'],
                'canonical admission differs from independently audited sample')
        require(not row['label'] or proof['target_audit']['correspondence_count'] >= 8,
                'fewer than eight healthy corresponding points')
        if row['label']:
            identity = source.get('pre_damage_pair_sha256')
            require(identity is not None and identity == row.get('pre_damage_pair_sha256')
                    == proof.get('pre_damage_pair_sha256') == lineage.get(name, {}).get('pre_damage_pair_sha256')
                    and lineage[name]['split'] == 'train' and row['source_base_key'] == 'pre-damage-pair:' + identity,
                    'Original pre-damage pairing is missing or changed')
        require(row['actual_matcher_input_sha256'] not in hashes, 'duplicated canonical straight inputs')
        hashes.add(row['actual_matcher_input_sha256'])
        refs.append(StraightSampleRef(name, row['source_base_key'], row['actual_matcher_input_sha256'],
                                     row['sample_path'], row['sample_sha256'], row['label']))
    refs = tuple(sorted(refs, key=lambda r: r.pair_id))
    require({r['pair_id'] for r in record['entries'] if r['label']} ==
            {name for name, r in lineage.items() if r['split'] == 'train'}, 'Incomplete TRAIN pair-lineage coverage')
    require(digest([asdict(r) for r in refs]) == record['catalog_sha256'], 'canonical straight catalog changed')
    return record, refs


def combined_rows(spec, base_spec, base_ledger):
    combined = read_bound(spec)
    require(combined.get('schema') == 'matcher-v2-data-admission/1'
            and combined.get('status') == 'passed' and combined.get('training_admitted') is True
            and combined.get('gpu_used') is False and combined.get('base_admission') == base_spec
            and combined.get('original_exposures_removed') == 0,
            'combined admission must preserve the exact original admission')
    original = read_bound(base_spec)
    require(digest(original['catalog']) == original['catalog_sha256']
            and original['catalog'] == [asdict(r) for r in base_ledger.catalog],
            'original admitted catalog differs')
    straight, refs = straight_rows(combined['straight_admission'])
    catalog = [asdict(r) for r in base_ledger.catalog + refs]
    require(combined['catalog'] == catalog and combined['catalog_sha256'] == digest(catalog),
            'combined catalog is not exact original prefix followed by straight TRAIN')
    require(len({r['pair_id'] for r in catalog}) == len(catalog)
            and len({r['model_input_sha256'] for r in catalog}) == len(catalog),
            'new data collides with original identities or actual six inputs')
    return combined, straight, refs


class CombinedDataset:
    def __init__(self, combined_spec, base_spec, base_ledger, runtime_ledger, source_root, loader=None):
        combined, straight, refs = combined_rows(combined_spec, base_spec, base_ledger)
        require(runtime_ledger.catalog == base_ledger.catalog + refs, 'runtime ledger/catalog identity differs')
        self.base = AdmittedDataset(base_spec['path'], base_spec['sha256'], base_ledger, source_root, loader=loader)
        self.loader = loader or bound_module('staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset', source_root).load_sample
        lookup = {row['pair_id']: row for row in straight['entries']}
        self.entries = [lookup[ref.pair_id] for ref in refs]
        self.verified = set(); self.admission_sha256 = combined_spec['sha256']

    def __len__(self):return len(self.base) + len(self.entries)

    def __getitem__(self, index):
        require(type(index) is int and 0 <= index < len(self), 'dataset index outside immutable catalog')
        if index < len(self.base):return self.base[index]
        offset = index-len(self.base); row = self.entries[offset]; path = Path(row['sample_path'])
        require(path.is_absolute(), 'absolute materialized sample path required')
        if offset not in self.verified:
            require(file_sha(path) == row['sample_sha256'], 'straight sample changed')
        sample, report = self.loader(path)
        require(sample.pair_id == row['pair_id'] and float(sample.label) in (0., 1.)
                and bool(sample.label) == row['label']
                and bool(report['pose_supervision_enabled']) == (bool(sample.label) and not report['changed_pair']),
                'straight loader identity/pose contract differs')
        if offset not in self.verified:
            require(tensor_digest(sample, MATCHER_INPUTS) == row['actual_matcher_input_sha256'],
                    'actual six straight inputs differ from canonical admission')
            require(digest(dict(materialized_targets=tensor_digest(sample, TARGETS), precise_recipe=False))
                    == row['effective_training_target_sha256']
                    and legacy_numerical_digest(sample) == row['legacy_pixel_audit_numerical_sha256'],
                    'actual straight supervision differs from admission')
            self.verified.add(offset)
        return sample, report, row
