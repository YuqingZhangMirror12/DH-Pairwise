"""Simulation-only frozen-checkpoint selection; no training or real-data input.

Reports must contain full per-pair matching loss and either its unreduced
components or the registered frozen-native physical-microbatch-one provenance.
This checks arithmetic/provenance declarations, not whether inference actually
ran: the caller must bind actual process returns and tensor/checkpoint hashes.
"""
import math

from .protocol import (LOSS_SEMANTICS, RULE_ID, SELECTION_KIND, STAGE_WEIGHTS,
                       canonical, digest, is_sha, require, validate_protocol)

REPORT_SCHEMA = 'model-selection-checkpoint-report/2'
RESULT_SCHEMA = 'model-selection-result/2'
LOSS_COMPONENTS = ('match_nll', 'dustbin_nll', 'translation_smooth_l1', 'sinkhorn_residual')
REPORT_FIELDS = {'schema', 'protocol_sha256', 'update', 'role', 'manifest_sha256',
                 'status', 'returncode', 'checkpoint_sha256', 'model_state_sha256', 'rows'}
ROW_FIELDS = {'pair_id', 'sample_sha256', 'stage', 'generator', 'label', 'gt_known',
              'layout20', 'candidate_coverage', 'numeric_valid', 'has_candidate',
              'matcher_loss', 'loss_semantics'}
NATIVE_MICROBATCH1 = 'frozen_native_evaluate_view_microbatch1/1'


def _number(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'finite nonnegative ' + name + ' required')
    return float(value)


def _check_row(row, expected):
    require(isinstance(row, dict) and (set(row) == ROW_FIELDS | {'loss_components'}
            or set(row) == ROW_FIELDS | {'loss_implementation', 'physical_microbatch'}),
            'exact per-pair report fields required; batch loss surrogates are forbidden')
    require(all(row[key] == expected[key] for key in ('pair_id', 'sample_sha256', 'stage', 'generator', 'label')),
            'prediction row/sample/stratum identity differs')
    require(all(type(row[key]) is bool for key in ('label', 'gt_known', 'layout20',
            'candidate_coverage', 'numeric_valid', 'has_candidate')), 'Boolean prediction decisions required')
    # The frozen objective masks numeric-invalid examples to zero. Such a
    # checkpoint must fail closed, not win the loss tie-break with invalid zeros.
    require(row['numeric_valid'], 'numeric-invalid inference invalidates the checkpoint report')
    require(not row['label'] or row['gt_known'], 'positive simulation layout GT required')
    require(not row['layout20'] or (row['label'] and row['gt_known'] and row['numeric_valid']
            and row['has_candidate'] and row['candidate_coverage']), 'impossible correct-layout declaration')
    require(not row['candidate_coverage'] or (row['label'] and row['gt_known']
            and row['numeric_valid'] and row['has_candidate']), 'impossible candidate-coverage declaration')
    require(row['loss_semantics'] == LOSS_SEMANTICS, 'genuine full per-pair matcher loss semantics required')
    actual = _number(row['matcher_loss'], 'per-pair matcher loss')
    if 'loss_components' not in row:
        require(row['loss_implementation'] == NATIVE_MICROBATCH1
                and type(row['physical_microbatch']) is int and row['physical_microbatch'] == 1,
                'registered frozen-native microbatch1 loss provenance required')
        return
    components = row['loss_components']
    require(isinstance(components, dict) and set(components) == set(LOSS_COMPONENTS),
            'unreduced full matching-loss components required')
    parts = {key: _number(value, key) for key, value in components.items()}
    full = .5 * (parts['match_nll'] + parts['dustbin_nll']) + .5 * parts['translation_smooth_l1'] + .05 * parts['sinkhorn_residual']
    require(math.isclose(actual, full, rel_tol=1e-5, abs_tol=1e-6),
            'full per-pair matcher loss does not equal its unreduced components')


def summarize_report(plan, report):
    validate_protocol(plan)
    require(isinstance(report, dict) and set(report) == REPORT_FIELDS
            and report['schema'] == REPORT_SCHEMA, 'complete checkpoint report required')
    require(report['protocol_sha256'] == plan['sha256'] and report['role'] == 'select',
            'Matcher selection accepts this frozen simulation SELECT only; CAL/TEST/real forbidden')
    require(type(report['update']) is int and report['update'] in plan['candidate_updates'],
            'checkpoint outside the frozen trained candidate inventory')
    require(report['status'] == 'completed' and type(report['returncode']) is int and report['returncode'] == 0
            and is_sha(report['checkpoint_sha256']) and is_sha(report['model_state_sha256']),
            'successful actual evaluation and checkpoint/model identity required')
    binding = plan['manifest_bindings']['select']
    require(report['manifest_sha256'] == binding['sha256'], 'SELECT manifest hash differs')
    expected = {row['pair_id']: row for row in binding['manifest']['entries']}
    rows = report['rows']
    require(isinstance(rows, list) and len(rows) == binding['pair_count'], 'prediction row count differs')
    require(all(isinstance(row, dict) and isinstance(row.get('pair_id'), str) for row in rows),
            'prediction Pair IDs required')
    require(len({row['pair_id'] for row in rows}) == len(rows)
            and {row['pair_id'] for row in rows} == set(expected), 'missing/duplicate/unexpected prediction rows')
    for row in rows:
        _check_row(row, expected[row['pair_id']])
    stages = {}
    for stage in STAGE_WEIGHTS:
        strata = {}
        for generator in plan['required_generators'][stage]:
            selected = [row for row in rows if (row['stage'], row['generator']) == (stage, generator)]
            positives = [row for row in selected if row['label']]
            strata[generator] = dict(pairs=len(selected), positives=len(positives),
                negatives=len(selected) - len(positives),
                layout=math.fsum(int(r['layout20']) for r in positives) / len(positives),
                loss=math.fsum(r['matcher_loss'] for r in selected) / len(selected),
                candidate_coverage=math.fsum(int(r['candidate_coverage']) for r in positives) / len(positives))
        stages[stage] = dict(strata=strata, **{name: math.fsum(r[name] for r in strata.values()) / len(strata)
                                             for name in ('layout', 'loss', 'candidate_coverage')})
    totals = {name: math.fsum(STAGE_WEIGHTS[s] * values[name] for s, values in stages.items())
              for name in ('layout', 'loss', 'candidate_coverage')}
    return canonical(dict(update=report['update'], macro_layout=totals['layout'], macro_loss=totals['loss'],
        diagnostic_candidate_coverage=totals['candidate_coverage'], stages=stages,
        report_sha256=digest(report), checkpoint_sha256=report['checkpoint_sha256'],
        model_state_sha256=report['model_state_sha256'], protocol_sha256=plan['sha256']))


def select_checkpoint(plan, reports):
    """Select only after every frozen candidate has a complete valid report."""
    validate_protocol(plan)
    require(isinstance(reports, list) and all(isinstance(r, dict) for r in reports),
            'explicit checkpoint report list required')
    updates = [report.get('update') for report in reports]
    require(all(type(n) is int for n in updates) and len(updates) == len(set(updates))
            and sorted(updates) == plan['candidate_updates'],
            'candidate report set incomplete, duplicated, or outside the frozen inventory')
    summaries = sorted((summarize_report(plan, report) for report in reports), key=lambda r: r['update'])
    maximum = max(row['macro_layout'] for row in summaries)
    cutoff = maximum - plan['rule']['absolute_layout_tolerance']
    eligible = [row for row in summaries if row['macro_layout'] >= cutoff]
    winner = min(eligible, key=lambda row: (row['macro_loss'], -row['macro_layout'], row['update']))
    result = canonical(dict(schema=RESULT_SCHEMA, status='selected_from_complete_frozen_inventory',
        rule_id=RULE_ID, selection_kind=SELECTION_KIND, protocol_sha256=plan['sha256'],
        candidate_updates=plan['candidate_updates'], maximum_macro_layout=maximum,
        minimum_eligible_macro_layout=cutoff, eligible_updates=[r['update'] for r in eligible],
        selected_update=winner['update'], selected=winner, candidates=summaries,
        role='select', real_used=False, test_used=False, cal_used_for_matcher_selection=False,
        coverage_affects_selection=False, matcher_retrained=False,
        export_created=False, end_to_end_training_ready=False))
    result['sha256'] = digest(result)
    return result
