"""Explicit paired-view versus single-population simulation validation.

This module never creates data, selects sources, or guesses a new protocol from
filenames. Historical paired clean/hard metrics and new mixed-population metrics
have different denominators and must be declared in the bound contract.
"""
from collections import Counter


def layout(contract):
    design = contract.get('validation_design')
    if design is None:
        if contract.get('schema') != 's7-consensus-data-contract/1':
            raise ValueError('explicit validation design required for a new contract')
        kind = 'paired_clean_hard'
    else:
        kind = design.get('kind')
    if kind == 'paired_clean_hard':
        views = ('clean', 'hard')
    elif kind == 'single_mixed':
        views = ('mixed',)
    else:
        raise ValueError('unsupported validation design')
    expected = {split+'_'+view for split in ('cal', 'select') for view in views}
    if set(contract['validation']) != expected:
        raise ValueError('validation manifests differ from the declared design')
    for name, record in contract['validation'].items():
        if type(record.get('pair_count')) is not int or record['pair_count'] <= 0:
            raise ValueError('invalid declared Pair count: '+name)
    return kind, views


def check_rows(data, contract):
    kind, views = layout(contract)
    if set(data) != {'cal', 'select'}:
        raise ValueError('CAL and SELECT must both be evaluated')
    counts, physical, ids_by_split = {}, {}, {}
    for split in ('cal', 'select'):
        if set(data[split]) != set(views):
            raise ValueError('prediction views differ from contract')
        ids_by_view = []
        for view in views:
            rows = data[split][view]
            ids = [r['pair_id'] for r in rows]
            expected = contract['validation'][split+'_'+view]['pair_count']
            if len(ids) != expected or len(set(ids)) != expected:
                raise ValueError('missing or duplicate validation Pair: '+split+'_'+view)
            if set(Counter(bool(r['label']) for r in rows)) != {False, True}:
                raise ValueError('validation population requires both labels')
            if any(r['label'] and not r['gt_known'] for r in rows):
                raise ValueError('simulation positives require known layout GT')
            ids_by_view.append(ids)
        if kind == 'paired_clean_hard':
            if ids_by_view[0] != ids_by_view[1]:
                raise ValueError('paired clean/hard Pair order differs')
            if [r['label'] for r in data[split]['clean']] != [r['label'] for r in data[split]['hard']]:
                raise ValueError('paired view labels differ')
        ids_by_split[split] = set().union(*(set(ids) for ids in ids_by_view))
        counts[split] = len(ids_by_split[split])
        physical[split] = sum(map(len, ids_by_view))
    if ids_by_split['cal'] & ids_by_split['select']:
        raise ValueError('CAL and SELECT contain the same Pair ID')
    design = contract.get('validation_design') or {}
    if design.get('physical_samples') is not None and sum(physical.values()) != design['physical_samples']:
        raise ValueError('actual validation population differs from declared total')
    # Distinct augmented Pair IDs are not independent manuscript families.
    return dict(validation_design=kind, clean_hard_are_paired_views=kind=='paired_clean_hard',
        cal_distinct_pair_ids=counts['cal'], select_distinct_pair_ids=counts['select'],
        cal_physical_samples=physical['cal'], select_physical_samples=physical['select'],
        validation_physical_samples=sum(physical.values()),
        population_note='Pair IDs/augmented instances, not a count of independent manuscripts')


def learning_curve_csv(curve):
    views = sorted(set().union(*(set(row['selected']) for row in curve)))
    base = ['epoch', 'updates', 'exposures', 'selection_value', 'threshold']
    lines = [','.join(base+[view+'_layout20' for view in views])]
    for row in curve:
        values = [row[key] for key in base]
        values += [row['selected'].get(view, {}).get('layout20', '') for view in views]
        lines.append(','.join(map(str, values)))
    return '\n'.join(lines)+'\n'
