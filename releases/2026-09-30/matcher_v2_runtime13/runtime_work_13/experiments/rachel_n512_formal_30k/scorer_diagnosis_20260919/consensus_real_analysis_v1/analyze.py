"""Reconcile completed mergefix predictions, then reuse the registered real CV.

This is CPU-only arithmetic on immutable outputs. It does not select epochs,
alter scores, infer new candidates, modify annotations, or run a neural model.
The original 803-pair folds select thresholds; GT exclusions are applied only
after those decisions have been frozen, identically to the historical report.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from ..bounded_real_calibration_v2.common import crossfit, GRID, metrics

DIAG = Path(__file__).resolve().parents[1]
OLD_MODELS = ('S7_M12_matched_C16', 'S7_H', 'v3_B22',
              'frozen_features', 'independent_features')
POLICIES = ('bounded_max_f1', 'bounded_recall95')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, name):
    if a is None or b is None:
        require(a is None and b is None, name + ': unknown is not zero')
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10),
                name + ': ' + repr((a, b)))
    else:
        require(a == b, name + ': mismatch')


def independently_count(rr, decisions, layout_available):
    """Explicit set-count confusion matrix, including wrong-pose acceptance."""
    require(len(rr) == len(decisions) and bool(rr), 'metric population mismatch')
    positive = {i for i, r in enumerate(rr) if r['label']}
    negative = set(range(len(rr))) - positive
    accepted = {i for i, a in enumerate(decisions) if a}
    tp, fp = len(positive & accepted), len(negative & accepted)
    fn, tn = len(positive - accepted), len(negative - accepted)
    result = dict(n=len(rr), positive=len(positive), negative=len(negative),
        tp=tp, fp=fp, fn=fn, tn=tn, accuracy=(tp + tn) / len(rr),
        precision=tp / max(1, tp + fp), recall=tp / max(1, len(positive)),
        f1=2 * tp / max(1, 2 * tp + fp + fn),
        false_positive_rate=fp / max(1, len(negative)))
    shared = metrics([r['label'] for r in rr], decisions, [r['score'] for r in rr])
    for key in result:
        close(result[key], shared[key], 'independent metric ' + key)
    result['auroc'] = shared.get('auroc')
    layout_keys = ('layout_correct_total', 'layout_correct_accepted',
        'layout_correct_rejected', 'layout_accuracy', 'wrong_pose_accepted',
        'joint_tp', 'joint_fp', 'joint_fn', 'joint_precision', 'joint_recall', 'joint_f1')
    if not layout_available:
        result.update({key: None for key in layout_keys})
        return result
    require(all(rr[i]['layout_good_20'] is not None for i in positive), 'missing positive GT')
    good = {i for i in positive if rr[i]['layout_good_20']}
    jt, wrong = len(good & accepted), len((positive - good) & accepted)
    jf, jn = fp + wrong, len(positive) - jt
    result.update(layout_correct_total=len(good), layout_correct_accepted=jt,
        layout_correct_rejected=len(good - accepted), layout_accuracy=len(good) / len(positive),
        wrong_pose_accepted=wrong, joint_tp=jt, joint_fp=jf, joint_fn=jn,
        joint_precision=jt / max(1, jt + jf), joint_recall=jt / len(positive),
        joint_f1=2 * jt / max(1, 2 * jt + jf + jn))
    return result


def compact(r):
    return dict(pair_id=r['pair_id'], label=bool(r['label']), score=r['score'],
        decision_valid=bool(r['has_candidate'] and r['numeric_valid']),
        layout_good_20=bool(r['layout20']) if r['label'] and r['gt_known'] else None,
        layout_error_px=r['error_px'], translation=r['translation'], fold=r['fold'])


def policy_count(rr, decisions, layout_available, excluded):
    keep = [i for i, r in enumerate(rr) if r['pair_id'] not in excluded]
    return dict(original=independently_count(rr, decisions, layout_available),
        corrected=independently_count([rr[i] for i in keep], [decisions[i] for i in keep],
                                      layout_available))


def audit_sidecar(folder, item, evidence):
    """The index hashes arrays.npz, not its descriptive evidence.json."""
    path = folder / item['evidence']
    meta = read(path)
    evidence[str(path)] = sha(path)
    require(meta['pair_id'] == item['pair_id'], 'case evidence identity differs')
    sidecar = path.parent / meta['sidecar']['path']
    require(sidecar.parent.resolve() == path.parent.resolve(), 'sidecar escapes evidence directory')
    evidence[str(sidecar)] = sha(sidecar)
    require(evidence[str(sidecar)] == meta['sidecar']['sha256'] == item['sidecar_sha256'],
            'case array sidecar checksum differs')
    require(sidecar.stat().st_size == meta['sidecar']['bytes'], 'case array sidecar size differs')


def audit_job(root, job, case_plan, evidence):
    folder = root / job['task']
    def load(name):
        p = folder / name
        evidence[str(p)] = sha(p)
        return read(p)
    protocol, summary = load('protocol.json'), load('summary.json')
    complete, status = load('prediction_complete.json'), load('status.json')
    require(job['status'] == 'complete' and job['returncode'] == 0, 'job not complete')
    require(protocol['status'] == summary['status'] == status['status'] == 'complete',
            'incomplete protocol/summary/status')
    require(complete['status'] == 'all_predictions_frozen' and complete['model_state_unchanged'],
            'predictions not frozen or model changed')
    require(not protocol['model_selection_on_test_or_real'] and not protocol['threshold_refitted']
            and not protocol['gt_used_for_prediction'], 'evaluation used forbidden selection/input')
    require(evidence[str(folder / 'summary.json')] == job['verified']['summary_sha256'],
            'downloaded summary checksum differs')
    predictions_path = folder / 'pair_predictions.jsonl'
    prediction_sha = sha(predictions_path)
    require(prediction_sha == complete['sha256'] == job['verified']['predictions_sha256'],
            'downloaded predictions checksum differs')
    evidence[str(predictions_path)] = prediction_sha
    diagnostic_path = folder / 'case_diagnostics.jsonl'
    evidence[str(diagnostic_path)] = sha(diagnostic_path)
    pred, detailed = rows(predictions_path), rows(diagnostic_path)
    n = protocol['total_pairs']
    require(n == len(pred) == len(detailed) == complete['pairs'] == job['verified']['pairs'],
            'incomplete row counts')
    require(len({r['pair_id'] for r in pred}) == n, 'duplicate prediction Pair IDs')
    forbidden = {'label', 'fold', 'target_translation_rc', 'gt_known', 'layout20', 'error_px'}
    for p, d in zip(pred, detailed):
        require(not forbidden.intersection(p), 'posthoc labels leaked into prediction records')
        require(all(d[k] == v for k, v in p.items()), 'posthoc join changed raw predictions')
        require(math.isfinite(d['score']) and 0 <= d['score'] <= 1, 'invalid frozen score')
        valid = d['has_candidate'] and d['numeric_valid']
        require(d['accepted'] == bool(valid and d['score'] >= protocol['threshold']),
                'frozen decision differs from declared threshold')
        if d['gt_known'] and d['label']:
            error = math.dist(d['translation'], d['target_translation_rc']) if valid else None
            close(error, d['error_px'], 'winning pose error')
            require(d['layout20'] == bool(error is not None and error <= 20), 'GT20 label mismatch')
            errors = [math.dist(c['refined_translation'], d['target_translation_rc'])
                      for c in d['candidates']]
            for a, b in zip(errors, d['candidate_errors_px']):
                close(a, b, 'candidate pose error')
            require(len(errors) == len(d['candidate_errors_px']), 'candidate array length differs')
            require(d['candidate_coverage'] == any(e <= 20 for e in errors), 'coverage mismatch')
    compact_rows = [compact(r) for r in detailed]
    excluded = set(case_plan['user_confirmed_gt_exclusions']) if protocol['split'] == 'dunhuang_cv' else set()
    for group_name, reference in summary['groups'].items():
        subset = [r for r in compact_rows if group_name != 'gt_corrected_800' or r['pair_id'] not in excluded]
        for name, stored in reference.items():
            decisions = [r['decision_valid'] and r['score'] >= stored['threshold'] for r in subset]
            actual = independently_count(subset, decisions, summary['layout_gt_available'])
            aliases = dict(n='pairs', positive='positives', negative='negatives',
                           layout_correct_total='layout20_count', layout_accuracy='layout20',
                           layout_correct_accepted='joint_tp', layout_correct_rejected='winner_correct_but_rejected')
            for k, v in actual.items():
                target = aliases.get(k, k)
                if target in stored:
                    close(v, stored[target], job['task'] + '/' + group_name + '/' + name + '/' + target)
    index = load('diagnostic_index.json')
    planned = {r['pair_id'] for r in case_plan['cases'] if r['split'] == protocol['split']}
    require(index['selected_by_new_results'] is False, 'diagnostic cases selected after results')
    require({r['pair_id'] for r in index['cases']} == planned, 'fixed case plan differs')
    require(len(index['cases']) == len(planned), 'duplicate fixed diagnostic case')
    for item in index['cases']:
        audit_sidecar(folder, item, evidence)
        audit_path = folder / item['numerical_audit']
        audit = read(audit_path)
        require(audit['status'] == item['numerical_audit_status'] == 'passed', 'case audit failed')
        require(audit['pair_id'] == item['pair_id'], 'case audit identity differs')
        evidence[str(audit_path)] = sha(audit_path)
    return protocol, compact_rows, dict(job=job['task'], pairs=n, cases=len(index['cases']),
        checksums_passed=True, row_join_preserved=True, metrics_independently_recounted=True)


def paired(left, right, decisions_left, decisions_right, layout_available, excluded):
    require([r['pair_id'] for r in left] == [r['pair_id'] for r in right], 'paired IDs differ')
    selected = [i for i, r in enumerate(left) if r['pair_id'] not in excluded]
    classification = Counter()
    layout = Counter()
    joint = Counter()
    for i in selected:
        require(left[i]['label'] == right[i]['label'], 'paired labels differ')
        y = left[i]['label']
        classification[('new_correct' if decisions_left[i] == y else 'new_wrong') + '__' +
                       ('old_correct' if decisions_right[i] == y else 'old_wrong')] += 1
        if layout_available and y:
            a, b = left[i]['layout_good_20'], right[i]['layout_good_20']
            layout[('new_good' if a else 'new_bad') + '__' + ('old_good' if b else 'old_bad')] += 1
            joint[('new_pass' if a and decisions_left[i] else 'new_fail') + '__' +
                  ('old_pass' if b and decisions_right[i] else 'old_fail')] += 1
    return dict(classification=dict(classification), layout=dict(layout) if layout_available else None,
                accepted_correct_layout=dict(joint) if layout_available else None)


def run(args):
    root, old_root = Path(args.evaluation), Path(args.historical)
    output = Path(args.out)
    require(not output.exists(), 'Preserve previous analysis; output already exists')
    evidence = {}
    case_path = DIAG / 's7_consensus_eval_v14/case_plan.json'
    case_plan = read(case_path); evidence[str(case_path)] = sha(case_path)
    terminal = read(root / 'evaluation_complete.json')
    evidence[str(root / 'evaluation_complete.json')] = sha(root / 'evaluation_complete.json')
    require(terminal['status'] == 'complete' and terminal['all_six_populations_verified']
            and terminal['cases_exported'] == 22 and len(terminal['jobs']) == 6, 'not six completed evaluations')
    audited, new_data, new_protocols = [], {}, {}
    for job in terminal['jobs']:
        p, rr, audit = audit_job(root, job, case_plan, evidence)
        audited.append(audit); new_data[job['task']] = rr; new_protocols[job['task']] = p
    historical = read(old_root / 'report_with_probes.json')
    old_comparison = read(old_root / 'comparison.json')
    old_oof = read(old_root / 'oof_predictions.json')
    for name in ('report_with_probes.json', 'comparison.json', 'oof_predictions.json'):
        evidence[str(old_root / name)] = sha(old_root / name)
    legacy_rows = historical['queries']['independent_cases']['rows']
    legacy_by_id = {r['pair_id']: r for r in legacy_rows}
    require(len(legacy_rows) == len(legacy_by_id) == 1405, 'historical cohort duplicates/missing')
    result = dict(schema='frozen-real-comparison/1', status='complete', generated_unix=time.time(),
        protocol=dict(real_threshold_grid=list(GRID), original_source_folds_reused=True,
            exclusions_after_calibration_only=True, epochs_or_geometry_selected_on_real=False,
            scores_rescaled=False, neural_inference_repeated=False, training_replicates=1,
            real_data_prior_design_exposure=True, negative_labels='source-constructed, not all human verified',
            turufan_layout_gt_available=False, annotations_modified=False),
        download_verification=audited, splits={}, inputs=evidence)
    all_rows = {}
    for split, meta_name in (('dunhuang_cv', 'real'), ('turufan', 'ood')):
        mpath = DIAG / ('real_domain_calibration_v1/results/' + meta_name + '/manifest.json')
        meta = read(mpath); evidence[str(mpath)] = sha(mpath)
        population = meta['pairs']; ids = [p['pair_id'] for p in population]
        require(len(ids) == len(set(ids)), 'manifest IDs not unique')
        layout_available = meta_name == 'real'
        excluded = set(case_plan['user_confirmed_gt_exclusions']) if layout_available else set()
        require(excluded <= set(ids), 'exclusion not in cohort')
        models, normalized, decisions = {}, {}, {}
        all_rows[split] = {}
        for arm in ('m12', 'scratch'):
            job = arm + '_' + split; key = 'mergefix_' + arm
            p, rr = new_protocols[job], new_data[job]
            require(p['source']['manifest_sha256'] == evidence[str(mpath)], 'different real manifest bytes')
            require([r['pair_id'] for r in rr] == ids, 'different prediction population/order')
            for a, b in zip(rr, population):
                require(a['label'] == b['label'] and a['fold'] == b['fold'], 'label/fold mismatch')
            normalized[key] = rr; decisions[key] = {}; all_rows[split][key] = {}
            entry = dict(epoch=p['selected_epoch'], checkpoint_sha256=p['checkpoint_sha256'],
                         sim_threshold=p['threshold'], source=p['source'], policies={})
            for policy, threshold in (('sim_frozen', p['threshold']), ('fixed03', .30)):
                dd = [r['decision_valid'] and r['score'] >= threshold for r in rr]
                decisions[key][policy] = dd
                entry['policies'][policy] = dict(threshold=threshold,
                    **policy_count(rr, dd, layout_available, excluded))
            for policy in POLICIES:
                cv, heldout = crossfit(meta, rr, policy)
                by_id = {r['pair_id']: r for r in heldout}
                ordered = [by_id[pid] for pid in ids]
                dd = [r['accepted'] for r in ordered]; decisions[key][policy] = dd
                all_rows[split][key][policy] = ordered
                entry['policies'][policy] = dict(folds=cv['folds'], threshold_median=cv['threshold_median'],
                    threshold_min=cv['threshold_min'], threshold_max=cv['threshold_max'],
                    **policy_count(rr, dd, layout_available, excluded))
            models[key] = entry
        for key in OLD_MODELS:
            rr = []
            for p in population:
                row = legacy_by_id[p['pair_id']]; raw = row['models'][key]
                require(bool(row['label']) == p['label'] and row['fold'] == p['fold'], 'historical fold/label differs')
                rr.append(dict(pair_id=p['pair_id'], label=bool(p['label']), fold=p['fold'],
                    score=raw['score'], decision_valid=raw['decision_valid'],
                    layout_good_20=raw['layout20'] if layout_available and p['label'] else None,
                    layout_error_px=raw['layout_error_px'], translation=raw['translation']))
            normalized[key] = rr; decisions[key] = {}; all_rows[split][key] = {}
            ref = old_comparison['splits'][split]['models'][key]
            entry = dict(historical_outputs_preserved=True, policies={})
            for policy in ('sim_frozen', 'fixed03'):
                t = ref[policy]['threshold']; dd = [r['decision_valid'] and r['score'] >= t for r in rr]
                decisions[key][policy] = dd
                counts = policy_count(rr, dd, layout_available, excluded)
                for k, v in ref[policy].items():
                    if k in counts['original']:
                        close(counts['original'][k], v, key + '/' + policy + '/' + k)
                entry['policies'][policy] = dict(threshold=t, **counts)
            for policy in POLICIES:
                saved = old_oof[split][key][policy]
                lookup = {r['pair_id']: r for r in saved}
                require(len(saved) == len(lookup) == len(ids) and set(lookup) == set(ids), 'historical OOF cohort differs')
                ordered = [lookup[pid] for pid in ids]
                for r, frozen in zip(rr, ordered):
                    close(r['score'], frozen['score'], 'historical score changed')
                    require(frozen['label'] == r['label'] and frozen['fold'] == r['fold'], 'historical OOF fold differs')
                    require(frozen['threshold'] in GRID, 'historical threshold outside grid')
                    require(frozen['accepted'] == bool(r['decision_valid'] and r['score'] >= frozen['threshold']), 'historical OOF decision changed')
                dd = [r['accepted'] for r in ordered]; decisions[key][policy] = dd
                all_rows[split][key][policy] = ordered
                counts = policy_count(rr, dd, layout_available, excluded)
                for k, v in ref['cv'][policy]['pooled_out_of_fold'].items():
                    if k in counts['original']:
                        close(counts['original'][k], v, key + '/OOF/' + k)
                folds = ref['cv'][policy]['folds']
                for f in folds:
                    require({r['threshold'] for r in ordered if r['fold'] == f['fold']} == {f['threshold']}, 'fold threshold metadata differs')
                entry['policies'][policy] = dict(folds=folds,
                    threshold_median=statistics.median(f['threshold'] for f in folds),
                    threshold_min=min(f['threshold'] for f in folds),
                    threshold_max=max(f['threshold'] for f in folds), **counts)
            models[key] = entry
        comparisons = {}
        for left in ('mergefix_m12', 'mergefix_scratch'):
            for right in OLD_MODELS + (('mergefix_m12',) if left == 'mergefix_scratch' else ()):
                comparisons[left + '_vs_' + right] = {
                    policy: paired(normalized[left], normalized[right], decisions[left][policy],
                                   decisions[right][policy], layout_available, excluded)
                    for policy in ('sim_frozen', 'fixed03') + POLICIES}
        result['splits'][split] = dict(original_pairs=len(ids),
            corrected_pairs=len(ids) - len(excluded), excluded_pair_ids=sorted(excluded),
            models=models, paired_corrected=comparisons)
    for p in list(evidence):
        require(sha(p) == evidence[p], 'analysis modified a source file: ' + p)
    result['all_source_hashes_unchanged'] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(dict(summary=result, out_of_fold=all_rows), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(output), sha256=sha(output), jobs_verified=6,
                          fixed_cases_verified=22, compared_models=7)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--historical', required=True)
    p.add_argument('--out', required=True)
    run(p.parse_args())
