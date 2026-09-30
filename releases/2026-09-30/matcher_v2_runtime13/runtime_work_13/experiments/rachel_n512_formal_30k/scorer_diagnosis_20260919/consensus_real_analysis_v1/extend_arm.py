"""Append one completed frozen research arm without changing historical results.

CPU-only postprocessing. Reuses registered source-disjoint real folds and the
prespecified classification threshold grid; never selects geometry or epochs.
Original 803 Dunhuang records calibrate folds before the 3 GT exclusions.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time

from .analyze import (DIAG, OLD_MODELS, POLICIES, audit_job, compact, close,
                      paired, policy_count, read, require, rows, sha)
from ..bounded_real_calibration_v2.common import crossfit


PRIOR_ARMS = {
    'threshold_m12': ('threshold', 'm12'),
    'threshold_scratch_fixed': ('threshold', 'scratch_fixed'),
    'simple_m12': ('simple', 'm12'),
    'simple_scratch_fixed': ('simple', 'scratch_fixed'),
}


def prior_bindings(base):
    summary = base['summary']
    require(summary['status'] == 'complete' and summary['all_source_hashes_unchanged'],
            'previous complete comparison required')
    result = {}
    for mapping in [summary['inputs']] + [x['input_sha256'] for x in summary.get('extensions', [])]:
        for path, value in mapping.items():
            path = str(Path(path).resolve())
            require(path not in result or result[path] == value, 'prior source binding conflict')
            result[path] = value
    return result


def prior_rows(key, entry, split, imports, bindings, evidence):
    """Load a named prior arm only from the prior analysis's immutable files."""
    require(key in PRIOR_ARMS and key in imports, 'prior evaluation directory required: ' + key)
    variant, arm = PRIOR_ARMS[key]
    folder = Path(imports[key]) / (arm + '_' + split)
    paths = [folder / name for name in ('protocol.json', 'case_diagnostics.jsonl')]
    for path in paths:
        expected = bindings.get(str(path.resolve()))
        require(expected is not None and sha(path) == expected, 'unbound or changed prior file: ' + str(path))
        evidence[str(path)] = expected
    protocol = read(paths[0])
    require(protocol['status'] == 'complete' and protocol['variant'] == variant
            and protocol['arm'] == arm and protocol['split'] == split,
            'prior frozen arm identity differs')
    require(protocol['checkpoint_sha256'] == entry['checkpoint_sha256']
            and protocol['selected_epoch'] == entry['epoch']
            and protocol['threshold'] == entry['sim_threshold'], 'prior selected model differs')
    return [compact(r) for r in rows(paths[1])]


def check_completion(terminal, arm):
    require(terminal['status'] == 'complete' and terminal['arm'] == arm,
            'wrong or incomplete arm')
    require(terminal.get('all_three_populations_verified') is True and
            terminal.get('cases_exported') == 11, 'not three complete populations/11 cases')
    jobs = terminal['jobs']
    require(len(jobs) == 3 and {j['split'] for j in jobs} ==
            {'sim_test_v14', 'dunhuang_cv', 'turufan'}, 'missing/duplicate evaluation population')
    require(all(j['status'] == 'complete' and j['returncode'] == 0 for j in jobs),
            'unsuccessful frozen evaluation')
    require(not terminal.get('training_modified', False), 'training changed during evaluation')


def append_policies(meta, rr, sim_threshold, excluded, layout_available):
    """CAL/OOF uses the original cohort; corrected reporting never changes it."""
    ids = [r['pair_id'] for r in meta['pairs']]
    require(ids == [r['pair_id'] for r in rr], 'prediction population/order differs')
    for a, b in zip(meta['pairs'], rr):
        require(a['label'] == b['label'] and a['fold'] == b['fold'], 'fold or label mismatch')
    policies, decisions, out_of_fold = {}, {}, {}
    for policy, threshold in [('sim_frozen', sim_threshold), ('fixed03', .30)]:
        dd = [r['decision_valid'] and r['score'] >= threshold for r in rr]
        decisions[policy] = dd
        policies[policy] = dict(threshold=threshold,
            **policy_count(rr, dd, layout_available, excluded))
    for policy in POLICIES:
        cv, heldout = crossfit(meta, rr, policy)
        by_id = {r['pair_id']: r for r in heldout}
        ordered = [by_id[x] for x in ids]
        dd = [r['accepted'] for r in ordered]
        decisions[policy] = dd
        out_of_fold[policy] = ordered
        policies[policy] = dict(folds=cv['folds'],
            threshold_median=cv['threshold_median'], threshold_min=cv['threshold_min'],
            threshold_max=cv['threshold_max'], **policy_count(rr, dd, layout_available, excluded))
    return policies, decisions, out_of_fold


def run(args):
    out = Path(args.out)
    require(not out.exists(), 'preserve previous analysis; choose a new output')
    root = Path(args.evaluation)
    base = read(args.base)
    bindings = prior_bindings(base)
    prior_imports = {}
    for item in getattr(args, 'prior_evaluation', None) or []:
        name, directory = item.split('=', 1)
        require(name in PRIOR_ARMS and name not in prior_imports, 'unknown or duplicate prior arm')
        prior_imports[name] = directory
    evidence = {str(Path(args.base)): sha(args.base)}
    terminal = read(root / 'evaluation_complete.json')
    evidence[str(root / 'evaluation_complete.json')] = sha(root / 'evaluation_complete.json')
    check_completion(terminal, args.arm)
    case_path = DIAG / 's7_consensus_eval_v14/case_plan.json'
    case_plan = read(case_path)
    evidence[str(case_path)] = sha(case_path)
    new_data, protocols, audits = {}, {}, []
    for job in terminal['jobs']:
        p, rr, audit = audit_job(root, job, case_plan, evidence)
        require(p['arm'] == args.arm and p['variant'] == args.variant, 'model identity differs')
        require(p['checkpoint_sha256'] == args.checkpoint_sha256, 'unexpected selected model')
        protocols[p['split']] = p
        new_data[p['split']] = rr
        audits.append(audit)
    key = args.variant + '_' + args.arm
    result = copy.deepcopy(base)
    legacy_path = Path(args.historical) / 'report_with_probes.json'
    legacy = {r['pair_id']: r for r in read(legacy_path)['queries']['independent_cases']['rows']}
    evidence[str(legacy_path)] = sha(legacy_path)
    for split, meta_name in [('dunhuang_cv', 'real'), ('turufan', 'ood')]:
        target = result['summary']['splits'][split]
        require(key not in target['models'], 'arm already included')
        mpath = DIAG / ('real_domain_calibration_v1/results/' + meta_name + '/manifest.json')
        meta = read(mpath)
        evidence[str(mpath)] = sha(mpath)
        p, rr = protocols[split], new_data[split]
        require(p['source']['manifest_sha256'] == evidence[str(mpath)], 'changed real cohort')
        ids = [r['pair_id'] for r in meta['pairs']]
        excluded = set(target['excluded_pair_ids'])
        expected_excluded = set(case_plan['user_confirmed_gt_exclusions']) if meta_name == 'real' else set()
        require(excluded == expected_excluded, 'changed GT exclusions')
        available = meta_name == 'real'
        policies, decisions, oof = append_policies(meta, rr, p['threshold'], excluded, available)
        target['models'][key] = dict(epoch=p['selected_epoch'],
            checkpoint_sha256=p['checkpoint_sha256'], sim_threshold=p['threshold'],
            variant=p['variant'], evidence_mode=p['evidence_mode'], source=p['source'], policies=policies)
        result['out_of_fold'][split][key] = oof
        for old_key, old_entry in base['summary']['splits'][split]['models'].items():
            if old_key in OLD_MODELS:
                old_rows = []
                for item in meta['pairs']:
                    raw = legacy[item['pair_id']]['models'][old_key]
                    old_rows.append(dict(pair_id=item['pair_id'], label=bool(item['label']),
                        fold=item['fold'], score=raw['score'], decision_valid=raw['decision_valid'],
                        layout_good_20=raw['layout20'] if available and item['label'] else None))
            elif old_key in ('mergefix_m12', 'mergefix_scratch'):
                old_arm = old_key.removeprefix('mergefix_')
                pold = Path(args.mergefix_evaluation) / (old_arm + '_' + split) / 'case_diagnostics.jsonl'
                evidence[str(pold)] = sha(pold)
                old_rows = [compact(r) for r in rows(pold)]
            else:
                old_rows = prior_rows(old_key, old_entry, split, prior_imports, bindings, evidence)
            require([r['pair_id'] for r in old_rows] == ids, 'historical order changed')
            comparisons = {}
            for policy in ('sim_frozen', 'fixed03') + POLICIES:
                reference = old_entry['policies'][policy]
                if policy in POLICIES:
                    lookup = {r['pair_id']: r for r in base['out_of_fold'][split][old_key][policy]}
                    old_decisions = [lookup[x]['accepted'] for x in ids]
                    for x in old_rows:
                        close(x['score'], lookup[x['pair_id']]['score'], 'historical frozen score')
                else:
                    old_decisions = [x['decision_valid'] and x['score'] >= reference['threshold'] for x in old_rows]
                counts = policy_count(old_rows, old_decisions, available, excluded)
                for scope in ('original', 'corrected'):
                    for name, value in counts[scope].items():
                        close(value, reference[scope][name], 'historical summary changed')
                comparisons[policy] = paired(rr, old_rows, decisions[policy], old_decisions, available, excluded)
            target['paired_corrected'][key + '_vs_' + old_key] = comparisons
        for name, original in base['summary']['splits'][split]['models'].items():
            require(target['models'][name] == original, 'append modified old model result')
    for name, value in evidence.items():
        require(sha(name) == value, 'input changed during postprocessing: ' + name)
    result['summary']['extensions'] = result['summary'].get('extensions', []) + [dict(
        arm=key, generated_unix=time.time(), jobs_verified=audits,
        fixed_cases_verified=11, neural_inference_repeated=False, input_sha256=evidence,
        original_results_preserved=True, annotations_modified=False,
        only_classification_thresholds_crossfit=True, epochs_geometry_or_budget_selected_on_real=False)]
    result['summary']['generated_unix'] = time.time()
    result['summary']['all_source_hashes_unchanged'] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(path=str(out), sha256=sha(out), model=key,
        model_count=len(result['summary']['splits']['dunhuang_cv']['models']),
        results={k: result['summary']['splits'][k]['models'][key]['policies']['bounded_max_f1']
                 for k in ('dunhuang_cv','turufan')})))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('evaluation','base','historical','mergefix-evaluation','checkpoint-sha256','out'):
        p.add_argument('--'+field, required=True)
    p.add_argument('--arm', choices=['m12','scratch_fixed'], required=True)
    p.add_argument('--variant', choices=['threshold','simple'], required=True)
    p.add_argument('--prior-evaluation', action='append', metavar='MODEL=DIRECTORY',
                   help='Immutable outputs for already included threshold/simple arms')
    run(p.parse_args())
