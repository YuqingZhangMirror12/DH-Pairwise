"""Bind the full release to the pixel-based original20 census, without changing quotas."""
from copy import deepcopy
import math
from pathlib import Path
from ..s7_compound_v1.materialize import read, digest


def restrict(prepared, rows):
    expected = {(v, t['slot']): t for v, tasks in prepared['tasks'].items() for t in tasks}
    observed = {}
    for row in rows:
        key = (row['version'], row['slot'])
        if key in observed or key not in expected:
            raise ValueError('duplicate or unknown census view')
        task = expected[key]
        if row['base_keys'] != task['base_keys'] or row['recipe'] != task['recipe']:
            raise ValueError('census source identity or recipe changed')
        ratio = row['original_measurement']['common_over_smaller_perimeter']
        if not math.isfinite(ratio) or type(row['eligible']) is not bool or row['eligible'] != (ratio >= .20):
            raise ValueError('census eligibility disagrees with original20 measurement')
        observed[key] = row
    if set(observed) != set(expected):
        raise ValueError('incomplete original20 census')
    result = deepcopy(prepared)
    result['tasks'] = {v: [t for t in tasks if observed[v, t['slot']]['eligible']]
                       for v, tasks in prepared['tasks'].items()}
    result['original20_excluded_views'] = {
        v: len(prepared['tasks'][v]) - len(result['tasks'][v]) for v in prepared['tasks']}
    return result


def load(root, prepared, source):
    root = Path(root); source = Path(source)
    if (root/'failure.json').exists():
        raise ValueError('admission census has a failure receipt')
    complete = read(root/'complete.json'); protocol = read(root/'protocol.json')
    if complete['status'] != 'complete' or not complete['necessary_capacity_passed']:
        raise ValueError('insufficient original20 eligible unique bases under fixed quotas; do not force expansion')
    if digest(root/'rows.json') != complete['rows_sha256'] or digest(root/'protocol.json') != complete['protocol_sha256']:
        raise ValueError('admission census hash mismatch')
    if digest(root/'plan.json') != protocol['plan_sha256'] or read(root/'plan.json') != prepared:
        raise ValueError('census plan/targets differ from current full plan')
    census_source = Path(protocol['source'])
    if digest(census_source/'source_binding.json') != protocol['source_binding_sha256']:
        raise ValueError('census source binding changed')
    # Only the full-release driver and its launcher changed since the census;
    # every existing geometry, reconstruction, dependency and data module must match.
    suffix = 'experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/curriculum_data_v18_v19/'
    allowed = {suffix+'full.py', suffix+'launch_full.py'}
    for name, sha in read(census_source/'source_binding.json').items():
        if digest(census_source/name) != sha:
            raise ValueError('immutable census dependency changed: '+name)
        if name not in allowed and digest(source/name) != sha:
            raise ValueError('generation differs from admission measurement source: '+name)
    result = restrict(prepared, read(root/'rows.json'))
    result['admission'] = dict(root=str(root), complete_sha256=digest(root/'complete.json'),
        rows_sha256=complete['rows_sha256'], measured_original_pixels=True,
        fixed_source_partition_and_recipe_quotas=True,
        necessary_capacity_only_final30_still_audited_per_sample=True)
    return result
