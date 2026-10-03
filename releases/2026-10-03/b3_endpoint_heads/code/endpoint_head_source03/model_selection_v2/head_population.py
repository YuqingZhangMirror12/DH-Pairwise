"""Terminal populations for newly matched heads: reuse development, infer TEST.

No Matcher/Scorer choice, threshold fit, task restart or CUDA allocation happens
here. Dunhuang development rows have already been admitted separately; only its
withheld 161 pairs reach a terminal model forward. Turufan keeps its full 602
classification-only pairs and declared folds. Original simulation TEST and
strict-straight TEST remain unchanged, including their point labels.
"""
import argparse
from pathlib import Path

from . import head_execution as execution, head_observations as observations
from . import head_terminal as terminal, posthoc_export as ex
from .checkpoint_scan import module, save
from .protocol import canonical, digest, require
from .released_protocol import COUNTS, checked

SCHEMA = 'mixed-select-matched-head-populations/1'
SPLITS = ('sim_test', 'dunhuang_test', 'turufan', 'sim_straight_test')
COUNTS_TEST = dict(sim_test=3000, dunhuang_test=161, turufan=602, sim_straight_test=900)
INPUTS = ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b', 'contour_valid_a', 'contour_valid_b')


def reuse_artifacts(ref, spec_path):
    complete = checked(ref)
    require(complete.get('schema') == observations.SCHEMA
            and complete.get('status') == 'validation_rows_adopted'
            and complete.get('execution') == ex.receipt(spec_path)
            and complete.get('gpu_inference') is False
            and complete.get('terminal_evaluation_complete') is False, 'bound selected validation reuse required')
    audit = checked(complete['audit']); rows = checked(complete['rows']); origin = audit['origin']
    require(audit.get('schema') == observations.SCHEMA and origin.get('schema') == terminal.ORIGIN_SCHEMA
            and complete['selected_export'] == dict(path=origin['checkpoint'], sha256=origin['checkpoint_sha256'])
            and complete['actual_training_return'] == origin['actual_formal_return']
            and audit.get('model_forward_calls') == 0 and audit.get('test_predictions_read') is False
            and audit.get('test_used_for_selection') is False and audit.get('terminal_evaluation_complete') is False,
            'validation reuse/model ancestry differs')
    role_plan = checked(audit['real_split'])
    roles = role_plan['datasets']['dunhuang_cv']['roles']
    expected = dict(new_sim_cal=COUNTS['cal'][0], new_sim_select=COUNTS['select'][0],
        **{'dunhuang_'+role: roles[role]['pairs'] for role in ('real_cal', 'real_select')})
    # Use the frozen role metadata, not an assumed equal five-way split.
    require(complete['pairs'] == expected and set(rows) == set(expected)
            and {k: len(v) for k, v in rows.items()} == expected, 'reused population counts differ')
    all_ids = set()
    for name, group in rows.items():
        ids = [r['pair_id'] for r in group]
        require(len(set(ids)) == len(ids) and not all_ids.intersection(ids)
                and audit['summaries'][name]['pair_ids_sha256'] == digest(sorted(ids)),
                'reused validation identities differ or overlap')
        all_ids.update(ids)
    return complete, audit, rows


def build_plan(spec_path, reuse_ref, native_plan):
    complete, audit, rows = reuse_artifacts(reuse_ref, spec_path)
    role_plan = checked(native_plan['real_split']); role_spec = role_plan['datasets']['dunhuang_cv']
    require(native_plan['real_split'] == audit['real_split']
            and native_plan['real_binding']['gt_sha256'] == audit['gt']['sha256'] == observations.REAL_GT_SHA,
            'original real role/GT binding differs from admitted validation')
    meta = checked(dict(path=role_spec['remote_manifest'], sha256=role_spec['manifest_sha256']))
    entries = observations.dunhuang_entries(role_plan, meta, ('real_cal', 'real_select'))
    for role in ('real_cal', 'real_select'):
        observations.check_rows(rows['dunhuang_'+role], entries[role])
    test_ids = role_spec['roles']['real_test']['pair_ids']
    require(len(test_ids) == len(set(test_ids)) == COUNTS_TEST['dunhuang_test']
            and native_plan['pair_counts']['sim_test'] == COUNTS_TEST['sim_test']
            and native_plan['pair_counts']['turufan'] == COUNTS_TEST['turufan']
            and native_plan['pair_counts']['sim_straight_test'] == COUNTS_TEST['sim_straight_test'],
            'unchanged terminal population sizes required')
    seen = {r['pair_id'] for group in rows.values() for r in group}
    require(not seen.intersection(test_ids), 'a Dunhuang TEST Pair was already in validation')
    return canonical(dict(schema=SCHEMA, status='frozen_before_terminal_inference',
        execution=ex.receipt(spec_path), validation_reuse=reuse_ref, origin=audit['origin'],
        legacy_population=native_plan, splits=list(SPLITS), pair_counts=COUNTS_TEST,
        dunhuang_test_ids=test_ids, real_split=native_plan['real_split'], case_plan=native_plan['case_plan'],
        simulation_revision=native_plan['simulation_revision'], no_repeat_validation_inference=True,
        reused_counts=complete['pairs'], test_used_for_selection=False, threshold_refitting=False,
        task3_overlay_applied=False, turufan_layout_gt_available=False,
        dunhuang_full800_policy='639 admitted saved development predictions + 161 new TEST predictions',
        strict_select_policy='new mixed SELECT already includes its frozen strict stratum; no old900 SELECT rerun'))


def freeze(spec_path, reuse_ref, canonical_ref, case_plan, runtime):
    spec = ex.read(spec_path)
    require(spec.get('schema') == execution.SCHEMA and spec['runtime'] == runtime,
            'new matched-head execution required')
    execution.verify_sources(runtime, spec['native_inventory'], spec['external_python'])
    native = module('matcher_v2_v1.population', runtime)
    plan = native.freeze_plan(spec['original_execution']['path'], canonical_ref, case_plan, runtime)
    return build_plan(spec_path, reuse_ref, plan)


def verify_plan(plan, runtime):
    require(plan.get('schema') == SCHEMA and plan.get('status') == 'frozen_before_terminal_inference'
            and plan.get('splits') == list(SPLITS) and plan.get('pair_counts') == COUNTS_TEST
            and plan.get('test_used_for_selection') is False and plan.get('threshold_refitting') is False
            and plan.get('task3_overlay_applied') is False, 'registered new-head terminal plan required')
    legacy = plan['legacy_population']; spec_path = plan['execution']['path']
    checked(plan['execution'])
    require(plan == freeze(spec_path, plan['validation_reuse'], legacy['canonical_straight'],
                           legacy['case_plan']['path'], runtime), 'terminal plan/data identity changed')


def only_registered_batches(batches, pair_ids):
    """Filter before tensor transfer/model.forward, never after inference."""
    require(len(pair_ids) == len(set(pair_ids)), 'unique requested TEST IDs required')
    wanted = set(pair_ids); emitted = []; seen = set()
    for items, batch in batches:
        ids = [r['pair_id'] for r in items]
        require(len(ids) == len(set(ids)) and not seen.intersection(ids), 'duplicate source input batch Pair')
        seen.update(ids)
        indices = [i for i, pair_id in enumerate(ids) if pair_id in wanted]
        if not indices:
            continue
        require(all(name in batch and len(batch[name]) == len(items) for name in INPUTS),
                'six native input tensors and original batch dimensions required')
        selected = [items[i] for i in indices]; emitted += [r['pair_id'] for r in selected]
        yield selected, {name: batch[name][indices] for name in INPUTS}
    require(emitted == pair_ids, 'missing or out-of-order withheld TEST input Pair')


def load_population(split, plan, runtime):
    require(split in SPLITS and plan.get('schema') == SCHEMA, 'registered terminal-only split required')
    native = module('matcher_v2_v1.population', runtime)
    legacy = plan['legacy_population']; mapped = 'dunhuang_cv' if split == 'dunhuang_test' else split
    meta, batches, source, dataset = native.load_population(mapped, legacy, runtime)
    if split != 'dunhuang_test':
        return meta, batches, source, dataset
    wanted = plan['dunhuang_test_ids']; by_id = {r['pair_id']: r for r in meta['pairs']}
    require(len(by_id) == len(meta['pairs']) and set(wanted) <= set(by_id)
            and all(by_id[p]['fold'] == 0 for p in wanted), 'actual input manifest is not withheld Dunhuang TEST')
    selected = [by_id[p] for p in wanted]
    source = dict(source, inference_role='real_test_only', inference_pair_ids_sha256=digest(wanted),
                  development_inference_repeated=False, original_pairs=len(meta['pairs']))
    return dict(meta, pairs=selected), only_registered_batches(batches, wanted), source, dataset


def groups(rows, split, role_plan, runtime):
    require(split in SPLITS, 'terminal-only population required')
    if split == 'dunhuang_test':
        ids = [r['pair_id'] for r in rows]
        require(ids == role_plan['datasets']['dunhuang_cv']['roles']['real_test']['pair_ids']
                and len(ids) == len(set(ids)), 'wrong withheld real group')
        return {'real_test': rows}
    return module('matcher_v2_v1.population', runtime).population_groups(rows, split, role_plan)


def targets_after_prediction(meta, split, plan, dataset, runtime):
    require(split in SPLITS, 'terminal-only target population required')
    mapped = 'dunhuang_cv' if split == 'dunhuang_test' else split
    return module('matcher_v2_v1.population', runtime).targets_after_prediction(
        meta, mapped, plan['legacy_population'], dataset)


def diagnostic_ids(plan, split, runtime):
    require(split in SPLITS, 'terminal-only diagnostic population required')
    native = module('matcher_v2_v1.population', runtime)
    mapped = 'dunhuang_cv' if split == 'dunhuang_test' else split
    wanted = native.wanted_ids(plan['legacy_population'], mapped)
    return wanted.intersection(plan['dunhuang_test_ids']) if split == 'dunhuang_test' else wanted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--validation-reuse', type=Path, required=True)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--case-plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); spec = ex.read(args.spec)
    require(not args.out.exists(), 'preserve existing terminal population plan')
    result = freeze(args.spec, ex.receipt(args.validation_reuse), ex.receipt(args.canonical),
                    args.case_plan, spec['runtime'])
    save(args.out, result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
