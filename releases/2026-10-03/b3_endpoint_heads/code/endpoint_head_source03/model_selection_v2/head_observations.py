"""Adopt selected-head validation rows without repeating GPU inference.

This is CPU postprocessing, not a substitute for successful training or held-out
evaluation. The public entry first requires the complete matched-head training
chain. Saved CAL/SELECT and Dunhuang development predictions are then bound to
the selected update, checked against population identities, and summarized with
the unchanged native policy. No TEST/Turufan rows may enter this artifact.
"""
import argparse
import math
from pathlib import Path

from . import head_terminal as terminal, posthoc_export as ex
from .checkpoint_scan import module, save
from .protocol import canonical, digest, require
from .released_protocol import checked

SCHEMA = 'mixed-select-head-validation-reuse/1'
# Same immutable GT used in the completed original B3 and endpoint evaluation.
# Do not accept the bytes currently at a mutable path as historical ground truth.
REAL_GT_SHA = 'f1184d437c51f66ade0eef006e2ceb9731bc19a3825612083920122dd6a0f57a'


def receipt_only(ref):
    return {key: ref[key] for key in ('path', 'sha256')}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_rows(rows, entries, *, recipes=False):
    """Identity and per-row invariants; not a claim of rerunning the model."""
    require(isinstance(rows, list) and isinstance(entries, list) and rows and entries,
            'nonempty predictions and explicit source entries required')
    expected = {r['pair_id']: r for r in entries}; actual = {r['pair_id']: r for r in rows}
    require(len(expected) == len(entries) == len(actual) == len(rows) and set(expected) == set(actual),
            'missing, duplicated or foreign validation Pair')
    flags = ('label', 'gt_known', 'numeric_valid', 'has_candidate', 'layout20', 'candidate_coverage')
    for pair_id, row in actual.items():
        source = expected[pair_id]
        require(type(source['label']) is bool and all(type(row.get(k)) is bool for k in flags)
                and row['label'] == source['label'] and row['gt_known'] == row['label'],
                'validation label/GT/boolean identity differs')
        require(finite(row.get('score')) and 0 <= row['score'] <= 1, 'nonfinite or invalid head score')
        has = row['has_candidate']; pose = row.get('translation'); error = row.get('error_px')
        require((isinstance(pose, list) and len(pose) == 2 and all(map(finite, pose))) if has else pose is None,
                'candidate/translation identity differs')
        require((finite(error) and error >= 0) if row['gt_known'] and has else error is None,
                'layout error presence/finite range differs')
        require(row['layout20'] == (error is not None and error <= 20)
                and (not row['layout20'] or row['candidate_coverage'])
                and (row['gt_known'] or not row['candidate_coverage']), 'layout/coverage flags disagree')
        if recipes:
            require(isinstance(source.get('recipe'), str) and row.get('recipe') == source['recipe'],
                    'validation recipe identity differs')
    return sorted(rows, key=lambda r: r['pair_id'])


def dunhuang_entries(role_plan, meta, roles):
    """Use declared fold identities, not a hand-picked count or a name prefix."""
    require(role_plan.get('schema') == 'threshold-joint-real-split/1'
            and role_plan.get('source_disjoint') is True
            and role_plan.get('role_folds') == {'real_cal': [1], 'real_select': [2, 3, 4], 'real_test': [0]},
            'source-isolated registered real folds required')
    spec = role_plan['datasets']['dunhuang_cv']; rows = meta['pairs']
    lookup = {r['pair_id']: r for r in rows}
    require(len(lookup) == len(rows), 'duplicated real source Pair')
    excluded = set(spec['excluded_gt_pair_ids']); seen = set(excluded); groups = {}
    # Validate all membership metadata, including the withheld role; no TEST
    # predictions or pixel arrays are loaded and no TEST labels tune thresholds.
    for role, folds in role_plan['role_folds'].items():
        group = [r for r in rows if r['fold'] in folds and r['pair_id'] not in excluded]
        ids = [r['pair_id'] for r in group]; record = spec['roles'][role]
        require(ids == record['pair_ids'] and len(ids) == record['pairs']
                and not seen.intersection(ids), 'real fold membership differs/overlaps')
        require(all(type(r['label']) is bool for r in group)
                and sum(r['label'] for r in group) == record['positive']
                and sum(not r['label'] for r in group) == record['negative'], 'real fold labels/counts differ')
        seen.update(ids); groups[role] = group
    require(seen == set(lookup) and set(roles) == {'real_cal', 'real_select'},
            'only complete Dunhuang development roles may be reused')
    return {role: groups[role] for role in roles}


def verify_real_errors(rows, entries, gt):
    """Recompute selected-pose errors, not unverifiable full candidate clouds."""
    import numpy as np
    records = gt['positive_pairs']; by_gt = {r['pair_id']: r for r in records}
    require(len(records) == len(by_gt), 'duplicated layout GT')
    by_id = {r['pair_id']: r for r in entries}
    for row in rows:
        if not row['label']:
            continue
        source = by_id[row['pair_id']]; target = by_gt[row['pair_id']]
        require((source['fragment_a_id'], source['fragment_b_id']) ==
                (target['fragment_a_token'], target['fragment_b_token']), 'GT endpoint order differs')
        if row['has_candidate']:
            # Native validation uses FP32 torch norms. Allow only small CPU/GPU
            # rounding differences, not different GT, units or translations.
            error = float(np.linalg.norm(np.asarray(row['translation'], dtype=np.float32)
                - np.asarray(target['translation_gt_a_to_b_rc'], dtype=np.float32)))
            require(math.isclose(error, row['error_px'], rel_tol=1e-6, abs_tol=1e-4),
                    'saved Dunhuang pose error does not match the bound GT')


def without_timing(value):
    return canonical({k: v for k, v in value.items() if k != 'elapsed_seconds'})


def audit_selected(saved, origin, contract, real_ref, gt_ref, runtime):
    """Audit one already-admitted export's observation; performs zero forwards."""
    require(origin.get('schema') == terminal.ORIGIN_SCHEMA
            and origin.get('module') == saved['module']
            and origin.get('selected_updates') == saved['updates']
            and origin.get('validation_contract_sha256') == digest(contract)
            and origin.get('task3_overlay_applied') is False, 'selected head/validation identity differs')
    observation = saved['observation']; binding = saved['binding']
    require(observation.get('schema') == 'curriculum-observation/1'
            and observation.get('update') == saved['updates'] > 0
            and observation.get('module') == saved['module']
            and observation.get('common_plan_sha256') == binding['common_plan_sha256']
            and observation.get('selection_eligible') is True and observation.get('test_used') is False,
            'selected trained observation required')
    module('curriculum_training_v1.runtime_io', runtime).check_observations(
        [dict(update=saved['updates'], report=observation)], binding)
    artifact = checked(observation['artifact']); rows = artifact['rows']
    require(set(rows) == {'simulation', 'real_development'}
            and set(rows['simulation']) == {'cal', 'select'}
            and set(rows['real_development']) == {'dunhuang_cv'}
            and set(rows['real_development']['dunhuang_cv']) == {'real_cal', 'real_select'},
            'CAL/SELECT and Dunhuang development only; TEST/Turufan are forbidden')
    sim = rows['simulation']; real = rows['real_development']
    groups = {}
    for role in ('cal', 'select'):
        require(set(sim[role]) == {'mixed'}, 'new single mixed validation required')
        ref = contract['validation'][role+'_mixed']; manifest = checked(receipt_only(ref))
        require(manifest['split'] == role and len(manifest['entries']) == ref['pair_count'],
                'new SIM manifest role/count differs')
        groups['new_sim_'+role] = check_rows(sim[role]['mixed'], manifest['entries'], recipes=True)
    role_plan = checked(real_ref); spec = role_plan['datasets']['dunhuang_cv']
    meta = checked(dict(path=spec['remote_manifest'], sha256=spec['manifest_sha256']))
    entries = dunhuang_entries(role_plan, meta, real['dunhuang_cv'])
    require(gt_ref['path'] == role_plan['gt_path'], 'GT path differs from the fixed real plan')
    gt = checked(gt_ref)
    for role in ('real_cal', 'real_select'):
        group = check_rows(real['dunhuang_cv'][role], entries[role])
        verify_real_errors(group, entries[role], gt)
        groups['dunhuang_'+role] = group
    metric = module('s7_consensus_v1.metrics', runtime).summarize
    config = module('s7_consensus_v1.config', runtime).TrainingConfig(
        scorer_variant=binding['model_spec']['scorer_variant'])
    sim_replay = module('s7_consensus_v1.evaluation', runtime).summarize_validation(sim, contract, 'scorer', config)
    real_replay = module('matcher_v2_v1.validation', runtime).select_dunhuang(real, config, metric)
    require(without_timing(observation['simulation']) == canonical(sim_replay)
            and without_timing(observation['real_development']) == canonical(real_replay),
            'saved selected metrics/threshold do not replay from their actual rows')
    thresholds, threshold_origins = module('matcher_v2_v1.terminal', runtime).thresholds_for_export(saved)
    require(thresholds == origin['thresholds'] and threshold_origins == origin['threshold_origins'],
            'terminal thresholds changed after selection')
    summaries = {}
    for name, group in groups.items():
        threshold = thresholds['dunhuang_cv' if name.startswith('dunhuang_') else 'sim_test']
        summaries[name] = dict(primary=metric(group, threshold), fixed03=metric(group, .30),
                              pair_ids_sha256=digest(sorted(r['pair_id'] for r in group)))
    combined = groups['dunhuang_real_cal'] + groups['dunhuang_real_select']
    summaries['dunhuang_all_development'] = dict(primary=metric(combined, thresholds['dunhuang_cv']),
        fixed03=metric(combined, .30), pair_ids_sha256=digest(sorted(r['pair_id'] for r in combined)))
    return canonical(dict(schema=SCHEMA, status='selected_validation_audited_no_new_inference',
        origin=origin, observation=observation['artifact'], validation_contract=contract,
        real_split=real_ref, gt=gt_ref, summaries=summaries, replayed_simulation=sim_replay,
        replayed_real_development=real_replay, model_forward_calls=0, test_predictions_read=False,
        test_used_for_selection=False, terminal_evaluation_complete=False,
        limitations=['Saved Dunhuang rows have no full candidate/Q/MLP trace; coverage flags are not independently reconstructed.',
                    'SIM pose errors are taken from the bound validation artifact, not recomputed from pixels.',
                    'Reusing validated rows is not new GPU inference or a claim of blind development data.'])), groups


def adopt(driver_root, spec_path, selection_kind, output):
    output = Path(output).resolve()
    require(not output.exists(), 'preserve existing validation adoption')
    saved, origin = terminal.verified_export(driver_root, spec_path, selection_kind)
    spec = ex.read(spec_path); original = checked(spec['original_execution'])
    base = checked(original['base_execution']); real_ref = base['real_split']; runtime = spec['runtime']
    require(real_ref['sha256'] == module('curriculum_training_v1.matcher_population', runtime).REAL_PLAN_SHA,
            'only the registered real role plan is admitted')
    role_plan = checked(real_ref); gt_ref = dict(path=role_plan['gt_path'], sha256=REAL_GT_SHA)
    contract = saved['binding']['mixed_sim_head_experiment']['validation_contract']
    audit, groups = audit_selected(saved, origin, contract, real_ref, gt_ref, runtime)
    output.mkdir(parents=True)
    rows_ref = save(output/'rows.json', groups); audit_ref = save(output/'audit.json', audit)
    return save(output/'complete.json', dict(schema=SCHEMA, status='validation_rows_adopted',
        execution=ex.receipt(spec_path), selected_export=dict(path=origin['checkpoint'], sha256=origin['checkpoint_sha256']),
        actual_training_return=origin['actual_formal_return'], rows=rows_ref, audit=audit_ref,
        pairs={k: len(v) for k, v in groups.items()}, gpu_inference=False, terminal_evaluation_complete=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver-root', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--selection-kind', choices=('sim_best', 'real_best', 'equal_budget_endpoint'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    adopt(args.driver_root, args.spec, args.selection_kind, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
