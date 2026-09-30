"""Frozen common heldouts plus separately reported v4.2 SELECT/TEST.

Every B arm uses the same heldouts, including B2 which does not train on the
straight data. No population is filtered according to model scores or GT.
"""
from pathlib import Path

from ..curriculum_training_v1 import matcher_population as legacy
from ..curriculum_training_v1.checkpoint_io import file_sha
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.model_adapter import BASE, bound_module, require
from .data_runtime import read_bound
from .prepare_data import canonical_entry

STRAIGHT = ('sim_straight_select', 'sim_straight_test')
SPLITS = legacy.SPLITS + STRAIGHT
INPUTS = legacy.INPUTS


def straight_manifest(preparation, split):
    require(split in STRAIGHT, 'unregistered straight heldout')
    canonical = read_bound(preparation)
    require(canonical.get('schema') == 'straight-seam-canonical-preparation/1'
            and canonical.get('status') == 'ready_for_population_acceptance'
            and canonical.get('test_inferred') is False
            and canonical.get('canonical_cross_split_collisions') == canonical.get('canonical_base_collisions') == 0,
            'complete canonical three-split identity preparation required')
    generation = read_bound(canonical['full_generation_complete']); role = split[len('sim_straight_'):]
    require(generation.get('status') == 'complete_generation_integrity_supervision'
            and generation.get('total_rows') == 7800, 'complete independently audited generation required')
    binding = generation['datasets'][role]
    manifest = read_bound(dict(path=binding['manifest_path'], sha256=binding['manifest_sha256']))
    audit = read_bound(dict(path=binding['audit_path'], sha256=binding['audit_sha256']))
    require(manifest['split'] == role and not manifest.get('failed')
            and len(manifest['entries']) == audit['rows'] == len(audit['records']) == 900
            and audit['status'] == 'passed_integrity_and_supervision'
            and audit['source_manifest_sha256'] == binding['manifest_sha256'], 'straight heldout integrity differs')
    population = canonical['populations'][role]
    require(population['count'] == 900 and population['positives'] == 450
            and population['manifest_sha256'] == binding['manifest_sha256']
            and population['audit_sha256'] == binding['audit_sha256'], 'canonical population binding differs')
    return canonical, manifest, audit, binding


def fixed_straight_examples(entries):
    """Ten positives per type in pre-inference ID order, not failure mining."""
    result = []
    for recipe in ('straight_M', 'straight_J', 'straight_R'):
        ids = sorted(r['pair_id'] for r in entries if r['recipe'] == recipe and r['label'])
        require(len(ids) >= 10, 'missing a straight diagnostic category')
        result.extend(ids[:10])
    return result


def freeze_plan(spec_path, canonical_spec, case_plan, source_root):
    spec_path = Path(spec_path).resolve()
    spec = read_bound(dict(path=str(spec_path), sha256=file_sha(spec_path)))
    require(spec['schema'] == 'matcher-v2-execution/1' and spec['locked'] is True, 'locked new-arm execution required')
    legacy_plan = legacy.freeze_plan(spec['base_execution']['path'], case_plan, source_root)
    result = dict(legacy_plan, schema='matcher-v2-population-plan/1', arm=spec['arm'], module=spec['module'],
        execution_manifest_sha256=file_sha(spec_path), base_execution=spec['base_execution'],
        canonical_straight=canonical_spec, splits=list(SPLITS), scorer_used=spec['module'] != 'matcher',
        real_used_for_matcher_selection=False, turufan_used_for_selection=False,
        test_used_for_selection=False, threshold_fitting=False)
    result['pair_counts'] = dict(legacy_plan['pair_counts'], **{s:900 for s in STRAIGHT})
    result['straight'] = {}
    for split in STRAIGHT:
        canonical, manifest, _, binding = straight_manifest(canonical_spec, split)
        require(canonical['base_admission'] == read_bound(spec['base_execution'])['admission'],
                'straight collision audit belongs to a different original catalog')
        ids = [r['pair_id'] for r in manifest['entries']]
        require(len(ids) == len(set(ids)) == 900, 'duplicate straight heldout identities')
        result['straight'][split] = dict(binding, pair_ids_sha256=digest(ids),
            canonical_rows_sha256=canonical['populations'][split[len('sim_straight_'):]]['canonical_row_identity_sha256'],
            fixed_diagnostic_ids=fixed_straight_examples(manifest['entries']))
    return result


def validate_plan(plan, spec_path, source_root):
    require(plan.get('schema') == 'matcher-v2-population-plan/1'
            and plan == freeze_plan(spec_path, plan['canonical_straight'], plan['case_plan']['path'], source_root),
            'fixed population plan changed')


def population_groups(rows, split, roles):
    if split not in STRAIGHT:return legacy.population_groups(rows, split, roles)
    require(rows and len({r['pair_id'] for r in rows}) == len(rows)
            and all(r['recipe'] in ('straight_M', 'straight_J', 'straight_R') for r in rows),
            'straight population metadata differs')
    result = {'all':rows}
    for recipe in ('straight_M', 'straight_J', 'straight_R'):
        group = [r for r in rows if r['recipe'] == recipe]
        require(group, 'missing straight subgroup'); result[recipe] = group
    return result


class StraightPopulation:
    def __init__(self, entries, audit, role, seed, expected_canonical, source_root):
        self.entries = entries; self.audited = {r['pair_id']:r for r in audit['records']}
        require(len(self.audited) == len(entries), 'duplicate/missing audited straight IDs')
        self.loader = bound_module('staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset', source_root).load_sample
        # Integrity/identity only. No neural inference, threshold or GT-based
        # population selection occurs during this complete canonical recount.
        rows = [canonical_entry(e, self.audited[e['pair_id']], role, seed, self.loader) for e in entries]
        require(digest(rows) == expected_canonical, 'actual heldout canonical inputs/targets differ')

    def __len__(self):return len(self.entries)

    def __getitem__(self, i):
        entry = self.entries[i]
        require(file_sha(entry['sample_path']) == entry['sample_sha256'], 'heldout sample changed')
        sample, report = self.loader(entry['sample_path'])
        require(sample.pair_id == entry['pair_id'] and bool(sample.label) == bool(entry['label']), 'heldout identity differs')
        return sample, report, entry


def load_population(split, plan, source_root):
    require(split in SPLITS, 'unregistered heldout population')
    if split not in STRAIGHT:return legacy.load_population(split, plan, source_root)
    canonical, manifest, audit, _ = straight_manifest(plan['canonical_straight'], split)
    view = plan['straight'][split]; role = split[len('sim_straight_'):]
    require(digest([r['pair_id'] for r in manifest['entries']]) == view['pair_ids_sha256'], 'heldout membership changed')
    dataset = StraightPopulation(manifest['entries'], audit, role, manifest['seed'],
        canonical['populations'][role]['canonical_row_identity_sha256'], source_root)
    api = bound_module(BASE+'s7_consensus_v1.data', source_root)
    def batches():
        for start in range(0, len(dataset), 8):
            indices = range(start, min(start+8, len(dataset)))
            yield [dataset.entries[i] for i in indices], api.collate([dataset[i] for i in indices])
    source = dict(manifest=view['manifest_path'], manifest_sha256=view['manifest_sha256'],
        canonical_rows_sha256=view['canonical_rows_sha256'], preprocessing='unchanged v4.2 synthetic heldout',
        simulation_revision='v42-reuse-full', actual_real_donor_used=False)
    return dict(pairs=dataset.entries), batches(), source, dataset


def targets_after_prediction(meta, split, plan, dataset=None):
    if split not in STRAIGHT:return legacy.targets_after_prediction(meta, split, plan, dataset)
    # Same target semantics as the original simulation population. This is
    # invoked only after durable prediction completion by both evaluators.
    return legacy.targets_after_prediction(meta, 'sim_test', plan, dataset)


def wanted_ids(plan, split):
    cases = read_bound(plan['case_plan'])
    return ({r['pair_id'] for r in cases['cases'] if r['split'] == split}
            | set(plan.get('straight', {}).get(split, {}).get('fixed_diagnostic_ids', [])))
