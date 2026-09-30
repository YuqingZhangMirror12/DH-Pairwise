"""User-authorized small shortages and mandatory original-pair dedup receipts."""
from ..curriculum_training_v1.model_adapter import require


REQUESTED = dict(train=6000, select=900, test=900)


def population_counts(generation):
    require(generation.get('requested_rows') == 7800
            and set(generation['datasets']) == set(REQUESTED), 'All three requested populations required')
    counts = {}; omitted = 0
    for split, requested in REQUESTED.items():
        spec = generation['datasets'][split]
        actual, missing = spec.get('rows'), spec.get('omitted')
        require(type(actual) is int and type(missing) is int and 0 <= missing <= 20
                and spec.get('requested_rows') == requested and actual + missing == requested,
                'Only an explicitly recorded small population shortage is permitted')
        counts[split] = actual; omitted += missing
    require(omitted <= 20 and generation.get('authorized_omissions') == omitted
            and generation.get('total_rows') == sum(counts.values()) == 7800 - omitted,
            'Population shortage exceeds the user-authorized twenty-row total')
    return counts


def verify_lineage(record, acceptance, generation, read_bound):
    require('pair_lineage_audit' in acceptance, 'Original-pair lineage audit is mandatory')
    lineage = read_bound(acceptance['pair_lineage_audit'])
    require(lineage.get('schema') == 'straight-pair-lineage-audit/1'
            and lineage.get('status') == 'passed' and lineage.get('failures') == []
            and lineage.get('full_generation_complete') == record['full_generation_complete']
            and lineage.get('base_admission') == record['base_admission']
            and lineage.get('old_exposures_removed') == 0,
            'Original-pair lineage audit is failed, stale, or changes old exposures')
    registry = read_bound(lineage['original_registry'])
    require(registry.get('schema') == 'original-pair-registry/1' and registry.get('status') == 'passed'
            and registry.get('unresolved_pairs') == 0 and registry.get('base_admission') == record['base_admission'],
            'Original-pair registry is incomplete or belongs to another base catalog')
    original = {r['pre_damage_pair_sha256'] for r in registry['records']}
    observed = set(); lookup = {}
    for row in lineage['records']:
        identity = row['pre_damage_pair_sha256']
        require(identity not in original and identity not in observed and row['pair_id'] not in lookup,
                'Repeated pre-damage pair across old/new samples or corruption types')
        observed.add(identity); lookup[row['pair_id']] = row
    require(len(observed) == lineage['positive_pre_damage_pairs_checked']
            and len(observed) + lineage['negative_rows'] == generation['total_rows'],
            'Incomplete positive-pair lineage census')
    for split, spec in generation['datasets'].items():
        checked = lineage['splits'][split]
        require(checked['manifest_sha256'] == spec['manifest_sha256']
                and checked['audit_sha256'] == spec['audit_sha256'] and checked['rows'] == spec['rows'],
                'Original-pair audit is not for the complete generated split')
    return lookup
