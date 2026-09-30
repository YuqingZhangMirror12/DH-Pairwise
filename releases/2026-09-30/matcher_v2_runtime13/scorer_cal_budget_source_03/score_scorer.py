"""CAL-negative budgets for frozen B0--B3 Scorer predictions, without inference.

The original SIM threshold is retained as a separate reference. Only the locked
639 CAL/SELECT identities enter new calculations; TEST is integrity-checked by
the original evaluator but never enters calibration, selection or new metrics.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919'
MODULES = ('scorer_patch', 'scorer_stats')
METRIC_SHA = '8f59fae15b0ad5833f167501b10bd00060a949e01831744b5ad6b52bbaa8bb2e'
GROUPS = ('J', 'R', 'curved', 'unmeasurable')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def inventory(root):
    return {p.name: sha(p) for p in sorted(Path(root).glob('*.py'))}


def development_rows(frozen, strata, original_threshold):
    """Extract the same winners; never rerank, drop failures, or consult TEST."""
    require(math.isfinite(original_threshold) and 0 <= original_threshold <= 1, 'invalid original threshold')
    by_id = {row['pair_id']: row for row in frozen}
    require(len(by_id) == len(frozen), 'duplicate frozen prediction identity')
    ids = [row['pair_id'] for row in strata]
    require(ids and len(set(ids)) == len(ids) and set(ids) <= set(by_id), 'missing/duplicate development membership')
    result = []
    for item in strata:
        role, fold = item['role'], item['fold']
        require((role == 'real_cal' and fold == 1) or (role == 'real_select' and fold in (2, 3, 4)),
                'TEST or wrong CAL/SELECT role')
        row = by_id[item['pair_id']]
        require(type(item['label']) is bool and row['label'] is item['label'] and row['fold'] == fold,
                'frozen labels/folds differ from locked strata')
        require(all(type(row[k]) is bool for k in ('numeric_valid', 'has_candidate', 'accepted')),
                'explicit prediction booleans required')
        score = row['score']
        require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
                'finite Scorer probability required')
        candidates = row['candidates']; winner = row['selected_cluster_id']
        require(type(row['candidate_count']) is int and row['candidate_count'] == len(candidates),
                'candidate count differs')
        require(len({c['cluster_id'] for c in candidates}) == len(candidates), 'duplicate cluster identity')
        require(all(type(c['selected']) is bool for c in candidates), 'explicit selected flag required')
        if row['has_candidate']:
            selected = [c for c in candidates if c['selected']]
            require(type(winner) is int and len(selected) == 1 and selected[0]['cluster_id'] == winner,
                    'winner is not the frozen selected cluster')
            chosen = selected[0]
            require(chosen['score'] == score and chosen['refined_translation'] == row['translation'],
                    'selected score/pose changed')
            require(all(type(c['logit']) in (int, float) and math.isfinite(c['logit']) for c in candidates)
                    and chosen['logit'] == max(c['logit'] for c in candidates), 'Scorer winner was reranked')
        else:
            require(not candidates and winner in (-1, None) and score == 0 and row['translation'] is None,
                    'invalid no-candidate record')
        expected = row['numeric_valid'] and row['has_candidate'] and score >= original_threshold
        require(row['accepted'] is expected, 'original acceptance differs from frozen SIM threshold')
        if item['label']:
            require(row['gt_known'] is True and type(row['layout20']) is bool
                    and type(row['candidate_coverage']) is bool, 'known Dunhuang positive layout/coverage required')
            require(not row['layout20'] or (row['numeric_valid'] and row['has_candidate']
                    and row['candidate_coverage']), 'correct winner without valid/covered candidate')
            require(item['seam_group'] in GROUPS, 'unknown mask-only positive group')
        else:
            # The original posthoc join stores False for unknown-GT negatives.
            # Normalize them to null here instead of calling them wrong poses.
            require(row['gt_known'] is False and row.get('target_translation_rc') is None
                    and row['layout20'] in (False, None) and row['candidate_coverage'] in (False, None),
                    'negative examples have no Layout GT')
        result.append(dict(pair_id=item['pair_id'], role=role, fold=fold, label=item['label'],
            numeric_valid=row['numeric_valid'], has_candidate=row['has_candidate'], score=score,
            selected_cluster_id=winner, frozen_accepted=row['accepted'],
            layout20=row['layout20'] if item['label'] else None,
            candidate_coverage=row['candidate_coverage'] if item['label'] else None,
            seam_group=item['seam_group'] if item['label'] else None))
    return result


def fixed_threshold_readout(rows, threshold, positive_groups):
    """Independent set-count recount; no CAL schema or fitting implied here."""
    require(rows and len({r['pair_id'] for r in rows}) == len(rows)
            and all(r['role'] == 'real_select' for r in rows), 'unique SELECT-only rows required')
    require(math.isfinite(threshold) and threshold >= 0, 'finite nonnegative threshold required')
    positive = {r['pair_id'] for r in rows if r['label']}
    negative = {r['pair_id'] for r in rows if not r['label']}
    require(positive and negative and set(positive_groups) == positive, 'complete positive groups/both classes required')
    require(all(g in GROUPS for g in positive_groups.values()), 'unknown group')
    accepted = {r['pair_id'] for r in rows if r['numeric_valid'] and r['has_candidate'] and r['score'] >= threshold}
    correct = {r['pair_id'] for r in rows if r['label'] and r['layout20']}
    covered = {r['pair_id'] for r in rows if r['label'] and r['candidate_coverage']}
    tp, fp, fn, tn = len(accepted & positive), len(accepted & negative), len(positive - accepted), len(negative - accepted)
    groups = {}
    for group in GROUPS:
        members = {i for i, kind in positive_groups.items() if kind == group}
        groups[group] = dict(positive_count=len(members), covered=len(members & covered),
            layout_correct=len(members & correct), layout_correct_and_accepted=len(members & correct & accepted))
    return dict(threshold=threshold, pair_count=len(rows), positive_count=len(positive), negative_count=len(negative),
        tp=tp, fp=fp, tn=tn, fn=fn, accuracy=(tp+tn)/len(rows), pair_f1=2*tp/(2*tp+fp+fn),
        observed_select_fpr=fp/len(negative), positive_groups=groups,
        layout_correct=len(correct), candidate_coverage=len(covered),
        layout_correct_and_accepted=len(correct & accepted), correct_layout_rejected=len(correct-accepted),
        wrong_layout_accepted=len((accepted & positive)-correct),
        numeric_invalid=sum(not r['numeric_valid'] for r in rows),
        no_candidate=sum(not r['has_candidate'] for r in rows))


def compute(rows, original_threshold, metrics):
    calibration = [r for r in rows if r['role'] == 'real_cal']
    selected = [r for r in rows if r['role'] == 'real_select']
    require(len(calibration)+len(selected) == len(rows), 'forbidden analysis role')
    groups = {r['pair_id']: r['seam_group'] for r in selected if r['label']}
    budgets = {}
    for fraction in (.01, .02, .05):
        cal = metrics.calibrate_negative_budget(calibration, fraction)
        result = metrics.score_at_calibrated_budget(selected, cal, groups)
        recount = fixed_threshold_readout(selected, cal['threshold'], groups)
        for key in ('pair_count', 'tp', 'fp', 'tn', 'fn', 'accuracy', 'pair_f1', 'observed_select_fpr',
                    'positive_groups', 'layout_correct', 'layout_correct_and_accepted'):
            require(result[key] == recount[key], 'independent SELECT recount differs: '+key)
        cal_fp = len({r['pair_id'] for r in calibration if not r['label'] and r['has_candidate']
                      and r['numeric_valid'] and r['score'] >= cal['threshold']})
        require(cal_fp == cal['actual_false_positives'] <= cal['allowed_false_positives'], 'independent CAL recount differs')
        budgets[str(fraction)] = dict(result, independent_set_recount=recount)
    return dict(original_sim_threshold=dict(schema='select-at-frozen-sim-threshold/1',
        threshold_origin='SIM-CAL at the frozen SIM-selected update; no recalibration',
        **fixed_threshold_readout(selected, original_threshold, groups)), cal_negative_budgets=budgets)


def verify_origin(origin, arm, module, real_split_sha):
    require(arm in ('B0', 'B1', 'B2', 'B3') and module in MODULES, 'registered Scorer experiment required')
    require((arm == 'B0' and origin['schema'] == 'curriculum-scorer-evaluation-origin/1'
             and origin.get('arm', 'B0') == 'B0') or
            (arm != 'B0' and origin['schema'] == 'matcher-v2-terminal-origin/1'), 'wrong experiment origin schema')
    require(origin['module'] == module and origin['selection_kind'] == 'sim_best'
            and origin['selection_on_real'] is False and origin['selection_on_test'] is False
            and origin['threshold_refitted'] is False and origin['gt_used_for_prediction'] is False,
            'unchanged SIM-selected Scorer required')
    require(origin['matcher_updated_during_training'] is False and origin['old_head_imported'] is False
            and origin['local_conflict_head_present'] is False, 'registered frozen-Matcher new lightweight head required')
    require(origin['real_plan_sha256'] == real_split_sha and origin['split'] == 'dunhuang_cv', 'wrong real roles/population')
    require(origin['total_completed_updates'] == (24000 if arm in ('B0', 'B2') else 31667)
            and 0 < origin['selected_updates'] <= origin['total_completed_updates'], 'incomplete/different training budget')
    require(arm == 'B0' or origin['arm'] == arm, 'model belongs to another arm')
    require(origin['threshold'] == origin['thresholds']['sim_test'] == origin['thresholds']['dunhuang_cv'],
            'frozen SIM threshold changed')


def verify_output(out, arm, module, expected_code):
    """Reopen saved inputs/results after the child exited; no neural model."""
    out = Path(out)
    require(not (out/'failure.json').exists(), 'failure precedes budget completion')
    complete = read(out/'complete.json')
    require(complete['schema'] == 'scorer-cal-budgets-complete/1' and complete['status'] == 'complete'
            and complete['arm'] == arm and complete['module'] == module
            and complete['development_pairs'] == 639 and complete['negative_cal'] == 102
            and complete['negative_select'] == 304 and complete['gpu_used'] is False
            and complete['model_inference'] is False and complete['source_sha256'] == expected_code,
            'wrong Scorer budget completion')
    require(set(complete['files']) == {'summary.json', 'development.jsonl'}
            and all(sha(out/name) == value for name, value in complete['files'].items()), 'budget output changed')
    summary = read(out/'summary.json')
    require(summary['schema'] == 'matcher-v2-scorer-cal-budgets/1' and summary['arm'] == arm
            and summary['module'] == module and summary['source_sha256'] == expected_code
            and summary['test_used_for_analysis'] is False and summary['inference_repeated'] is False
            and summary['candidates_modified'] is False and summary['independent_set_recount_passed'] is True,
            'wrong budget analysis protocol')
    for field in ('strata', 'frozen_source', 'evaluation_controller_complete'):
        bound = summary[field]
        require(binding(bound['path']) == bound, 'completed analysis input changed: '+field)
    plan = read(summary['strata']['path'])
    verify_origin(summary['origin'], arm, module, plan['inputs']['real_split']['sha256'])
    normalized = development_rows(read_rows(summary['frozen_source']['path']), plan['rows'], summary['origin']['threshold'])
    require(normalized == read_rows(out/'development.jsonl'), 'saved development rows differ from frozen inputs')
    metrics = importlib.import_module('diagnostic_metrics')
    require(sha(metrics.__file__) == METRIC_SHA, 'wrong metric verifier')
    require(compute(normalized, summary['origin']['threshold'], metrics) == summary['metrics'], 'saved metrics failed recount')
    return dict(status='passed', arm=arm, module=module, complete_sha256=sha(out/'complete.json'),
        summary_sha256=sha(out/'summary.json'), development_sha256=sha(out/'development.jsonl'),
        actual_saved_rows_recounted=True, model_inference=False, pairs=len(normalized))


def baseline_for_b0(root, entry, package):
    """B0 deployed only adapters; restore its original bound baseline namespace."""
    values = read(Path(root)/'sim_best_dunhuang_cv_launch.json')['command']
    def flag(name):
        require(values.count(name) == 1 and values.index(name)+1 < len(values), 'missing/duplicate '+name)
        return Path(values[values.index(name)+1])
    spec = read(flag('--spec'))
    execution = importlib.import_module(PACKAGE+'.curriculum_training_v1.execution')
    baseline, _ = execution.check_sources(spec['baseline'])
    require(flag('--common-source').resolve() == (package/'s7_consensus_eval_v14').resolve()
            and flag('--binary-source').resolve() == (package/'binary_eval_v1').resolve(), 'B0 evaluator sources differ')
    entry.check_preparation(flag('--preparation'), package/'s7_consensus_eval_v14', package/'binary_eval_v1', baseline)
    entry.bind_baseline(baseline)


def load_verifier(source, root, arm):
    source = Path(source).resolve(); sys.path.insert(0, str(source))
    entry = importlib.import_module(PACKAGE+'.curriculum_scorer_eval_v1.entry')
    package = source/PACKAGE.replace('.', '/')
    require(source in Path(entry.__file__).resolve().parents, 'wrong frozen evaluator entry')
    if arm == 'B0':baseline_for_b0(root, entry, package)
    entry.bind_evaluation(package/'s7_consensus_eval_v14', package/'binary_eval_v1')
    kind = 'curriculum_scorer_eval_v1' if arm == 'B0' else 'matcher_v2_v1'
    name = 'controller' if arm == 'B0' else 'evaluation_controller'
    controller = importlib.import_module(PACKAGE+'.'+kind+'.'+name)
    auditor = importlib.import_module(PACKAGE+'.curriculum_scorer_eval_v1.audit')
    require(source in Path(controller.__file__).resolve().parents and source in Path(auditor.__file__).resolve().parents,
            'wrong frozen evaluator implementation')
    return controller


def run(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only posthoc analysis required')
    require(not args.out.exists(), 'exclusive fresh analysis output required')
    code = inventory(Path(__file__).parent)
    dependencies = inventory(args.diagnostics_source)
    require(dependencies['diagnostic_metrics.py'] == METRIC_SHA, 'metric implementation changed')
    sys.path.insert(0, str(args.diagnostics_source.resolve()))
    import diagnostic_metrics as metrics
    import ridge_runner
    require(Path(metrics.__file__).resolve().parent == args.diagnostics_source.resolve()
            and Path(ridge_runner.__file__).resolve().parent == args.diagnostics_source.resolve(), 'wrong metric dependency import')
    plan, _ = ridge_runner.verify_strata(args.strata)
    root = args.evaluation_root.resolve()
    controller = load_verifier(args.source_root, root, args.arm)
    for bad in (root/'failure.json', root/'controller_failure.json', root.parent/'failure.json',
                root.parent/'training/controller_failure.json', root.parent/'training/formal/failure.json'):
        require(not bad.exists(), 'failure precedes completion: '+str(bad))
    completion = read(root/'evaluation_complete.json')
    require(completion['status'] == 'complete' and completion['module'] == args.module
            and completion['job_count'] == (6 if args.arm == 'B0' else 12), 'all required terminal jobs must be complete')
    job = dict(name='sim_best_dunhuang_cv', selection='sim_best', split='dunhuang_cv')
    proof = controller.verify_job(root, job) if args.arm == 'B0' else controller.verify_job(root, job, args.module)
    require(proof in completion['jobs'], 'verified job missing from full completion')
    origin = proof['provenance']; verify_origin(origin, args.arm, args.module, plan['inputs']['real_split']['sha256'])
    require(sha(origin['checkpoint']) == origin['checkpoint_sha256'], 'selected checkpoint changed')
    training = Path(read(root/'controller_launch.json')['training_controller_root'])
    require(sha(training/'controller_complete.json') == origin['controller_complete_sha256'], 'training terminal changed')
    frozen = root/job['name']/'case_diagnostics.jsonl'; before = binding(frozen)
    rows = development_rows(read_rows(frozen), plan['rows'], origin['threshold'])
    cal = [r for r in rows if r['role'] == 'real_cal']; selected = [r for r in rows if r['role'] == 'real_select']
    require(len(rows) == 639 and len(cal) == 160 and len(selected) == 479
            and sum(r['label'] for r in cal) == 58 and sum(r['label'] for r in selected) == 175,
            'complete predeclared development population required')
    output = compute(rows, origin['threshold'], metrics)
    require(before == binding(frozen) and code == inventory(Path(__file__).parent)
            and dependencies == inventory(args.diagnostics_source), 'source/predictions changed during analysis')
    import torch
    require(not torch.cuda.is_initialized(), 'unexpected CUDA initialization')
    args.out.mkdir(parents=True)
    summary = dict(schema='matcher-v2-scorer-cal-budgets/1', arm=args.arm, module=args.module,
        origin=origin, proof=proof, strata=binding(args.strata), frozen_source=before,
        evaluation_controller_complete=binding(root/'evaluation_complete.json'), metrics=output,
        source_sha256=code, diagnostic_source_sha256=dependencies,
        test_used_for_analysis=False, inference_repeated=False, candidates_modified=False,
        original_sim_threshold_results_overwritten=False, independent_set_recount_passed=True,
        caveats=['CAL target FPR does not guarantee SELECT FPR.',
                 'J/R/curved are positive-only mask/GT strata, not per-group negative populations.',
                 'These frozen predictions retain their original device numerics; no CPU/GPU rerun or sensitivity claim.'])
    save(args.out/'summary.json', summary)
    with (args.out/'development.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    complete = dict(schema='scorer-cal-budgets-complete/1', status='complete', arm=args.arm, module=args.module,
        development_pairs=639, negative_cal=102, negative_select=304, gpu_used=False, model_inference=False,
        files={p.name: sha(p) for p in args.out.iterdir() if p.is_file()},
        source_sha256=code, process_return_must_be_verified_separately=True)
    save(args.out/'complete.json', complete)
    print(json.dumps(complete), flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostics-source', 'source-root', 'evaluation-root', 'strata', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=('B0', 'B1', 'B2', 'B3'), required=True)
    parser.add_argument('--module', choices=MODULES, required=True)
    args = parser.parse_args(); existed = args.out.exists()
    try:
        run(args)
    except BaseException as error:
        if not existed:
            args.out.mkdir(parents=True, exist_ok=True)
            if not (args.out/'failure.json').exists():
                save(args.out/'failure.json', dict(status='failed', error=repr(error),
                    traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
