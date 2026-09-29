"""Independent, fail-closed analysis of completed frozen decoder controls.

This is postprocessing only: no torch, model loading, inference, optimizer,
training-root writes, or TEST predictions. It does not modify the live sweep.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


MODELS = ('threshold_scratch_fixed', 'binary_patch', 'binary_stats',
          'aggressive_binary_patch')
SEARCHES = ('baseline', 'top3_only', 'all_modes_only', 'seeds32_only', 'combined')
POPULATION = {'dunhuang_cv': 639, 'turufan': 480, 'sim_select': 1500}
BASELINE = ('baseline', 'baseline')
POOLING_DIAGNOSTICS = {'zero_mean', 'zero_max', 'zero_mean_max'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sigmoid(value):
    return 1. / (1. + math.exp(-max(-700., min(700., value))))


def expected_readouts(model):
    extra = (('no_overlap_penalty', 'no_conflict_penalty', 'no_conflict_or_overlap')
             if model == 'threshold_scratch_fixed' else
             ('zero_overlap_feature',) + (() if model == 'binary_stats' else
               ('zero_mean', 'zero_max', 'zero_mean_max')))
    return {'baseline', 'raw_sum_q_rank', 'raw_max_q_rank', *extra}


def check_roles(plan):
    require(plan['role_folds'] == {'real_cal': [1], 'real_select': [2, 3, 4],
                                   'real_test': [0]}, 'registered source roles changed')
    require(plan['source_disjoint'], 'source isolation not established')
    output = {}
    for split in ('dunhuang_cv', 'turufan'):
        item = plan['datasets'][split]
        sets = {role: set(value['pair_ids']) for role, value in item['roles'].items()}
        seen = set()
        for role, ids in sets.items():
            require(len(ids) == len(item['roles'][role]['pair_ids']), 'duplicate role ID')
            require(not seen & ids, 'CAL/SELECT/TEST overlap')
            require(not ids & set(item['excluded_gt_pair_ids']), 'excluded GT entered role')
            seen |= ids
        output[split] = {pid: role for role in ('real_cal', 'real_select') for pid in sets[role]}
        require(len(output[split]) == POPULATION[split], 'unexpected development population')
    return output


def check_record(row, model, roles, sim_labels, real_labels):
    split, pid, role = row['split'], row['pair_id'], row['role']
    require(split in POPULATION, 'unknown or TEST split')
    require(isinstance(row['label'], bool), 'label must be boolean')
    if split == 'sim_select':
        require(role == 'sim_select' and pid in sim_labels, 'wrong SIM SELECT identity')
        expected_label = sim_labels[pid]
    else:
        require(roles[split].get(pid) == role, 'TEST/excluded/incorrect real role entered')
        expected_label = real_labels[split][pid]
    require(row['label'] == bool(expected_label), 'prediction/manifest label mismatch')
    target = row['target']
    if split == 'turufan' or not row['label']:
        require(target is None, 'invented Turufan/negative Layout GT')
    else:
        require(isinstance(target, list) and len(target) == 2 and
                all(math.isfinite(x) for x in target), 'missing positive Layout GT')
    require(set(row['predictions']) == set(SEARCHES), 'missing/extra search control')
    for search, prediction in row['predictions'].items():
        candidates = prediction['candidates']
        require(len(candidates) <= 8, 'final candidate budget changed')
        require(isinstance(prediction['numeric_valid'], bool), 'numeric validity absent')
        require(math.isfinite(prediction['seconds']) and prediction['seconds'] >= 0,
                'invalid elapsed time')
        require(prediction['search_audit']['gt_used'] is False, 'GT used in search')
        expected = {'row_column_topk': 3 if search in ('top3_only', 'combined') else 2,
                    'mode_limit': None if search in ('all_modes_only', 'combined') else 128,
                    'initial_seeds': 32 if search in ('seeds32_only', 'combined') else 16}
        require(prediction['search_audit']['policy'] == expected, 'confounded search control')
        for candidate in candidates:
            require(len(candidate['translation']) == 2 and
                    all(math.isfinite(x) for x in candidate['translation']), 'invalid candidate pose')
            require(candidate['pairs'] > 0, 'empty candidate union')
            require(0 <= candidate['max_q'] <= candidate['sum_q'] + 1e-7,
                    'inconsistent candidate Q')
            require(math.isfinite(candidate['sum_q']), 'nonfinite Q')
            require(abs(sigmoid(candidate['logit']) - candidate['score']) <= 5e-7,
                    'candidate score/logit disagree')
        require(set(prediction['readouts']) == expected_readouts(model), 'missing/extra readout')
        for kind, value in prediction['readouts'].items():
            winner = value['winner']
            if kind in ('raw_sum_q_rank', 'raw_max_q_rank'):
                field = 'sum_q' if kind == 'raw_sum_q_rank' else 'max_q'
                chosen = max(range(len(candidates)), key=lambda i: candidates[i][field]) if candidates else -1
                score = candidates[chosen]['score'] if chosen >= 0 else 0.
            else:
                logits = value['logits']
                require(len(logits) == len(candidates) and all(math.isfinite(x) for x in logits),
                        'bad logit population')
                chosen = max(range(len(logits)), key=logits.__getitem__) if logits else -1
                score = sigmoid(logits[chosen]) if chosen >= 0 else 0.
                if kind == 'baseline':
                    require(all(abs(a - b['logit']) <= 1e-7 for a, b in zip(logits, candidates)),
                            'baseline logit mismatch')
            require(winner == chosen and abs(score - value['score']) <= 5e-7,
                    'winner/score inconsistent with recorded readout')


def outcome(row, search, readout):
    prediction = row['predictions'][search]
    value = prediction['readouts'][readout]
    candidates, winner, target = prediction['candidates'], value['winner'], row['target']
    errors = ([math.dist(candidate['translation'], target) for candidate in candidates]
              if target is not None else [])
    candidate = candidates[winner] if winner >= 0 else None
    valid = prediction['numeric_valid'] and candidate is not None
    return dict(pair_id=row['pair_id'], label=row['label'], known=target is not None,
                valid=valid, score=value['score'], winner=winner,
                pose=candidate['translation'] if candidate else None,
                error=errors[winner] if errors and winner >= 0 else None,
                good=bool(valid and errors and errors[winner] <= 20),
                covered=bool(errors and min(errors) <= 20),
                pairs=candidate['pairs'] if candidate else 0,
                sum_q=candidate['sum_q'] if candidate else 0.,
                max_q=candidate['max_q'] if candidate else 0.,
                candidate_count=len(candidates), seconds=prediction['seconds'])


def accepted(row, threshold):
    return row['valid'] and row['score'] >= threshold


def average_precision(rows):
    positives = sum(row['label'] for row in rows)
    if not positives:
        return 0.
    by_score = defaultdict(list)
    for row in rows:
        by_score[row['score'] if row['valid'] else 0.].append(row['label'])
    total = true = 0
    value = 0.
    for score in sorted(by_score, reverse=True):
        new_true = sum(by_score[score]); true += new_true; total += len(by_score[score])
        value += new_true * true / total
    return value / positives


def metrics(rows, threshold, *, layout_available):
    require(bool(rows), 'empty metric population')
    positives = sum(row['label'] for row in rows); negatives = len(rows) - positives
    tp = sum(row['label'] and accepted(row, threshold) for row in rows)
    fp = sum(not row['label'] and accepted(row, threshold) for row in rows)
    fn, tn = positives - tp, negatives - fp
    out = dict(pairs=len(rows), positives=positives, negatives=negatives, threshold=threshold,
               tp=tp, fp=fp, fn=fn, tn=tn, accuracy=(tp + tn) / len(rows),
               precision=tp / max(1, tp + fp), recall=tp / max(1, positives),
               f1=2 * tp / max(1, 2 * tp + fp + fn), ap=average_precision(rows),
               false_positive_rate=fp / negatives if negatives else None,
               no_candidate_or_invalid=sum(not row['valid'] for row in rows))
    pscore = [row['score'] if row['valid'] else 0. for row in rows if row['label']]
    nscore = [row['score'] if row['valid'] else 0. for row in rows if not row['label']]
    out['auroc'] = (sum((p > n) + .5 * (p == n) for p in pscore for n in nscore) /
                    (len(pscore) * len(nscore))) if pscore and nscore else None
    fields = ('known_positive_layouts', 'layout20_count', 'layout20', 'candidate_coverage_count',
              'candidate_coverage', 'covered_but_winner_wrong', 'winner_correct_but_rejected',
              'positive_no_correct_candidate', 'wrong_pose_accepted', 'joint_tp', 'joint_fp',
              'joint_fn', 'joint_precision', 'joint_recall', 'joint_f1')
    if not layout_available:
        require(not any(row['known'] for row in rows), 'layout GT on no-GT domain')
        out.update({key: None for key in fields})
    else:
        require(all(row['known'] == row['label'] for row in rows), 'incomplete Layout GT')
        good = sum(row['good'] for row in rows); covered = sum(row['covered'] for row in rows)
        joint = sum(row['good'] and accepted(row, threshold) for row in rows)
        wrong = sum(row['known'] and not row['good'] and accepted(row, threshold) for row in rows)
        jfp, jfn = fp + wrong, positives - joint
        out.update(known_positive_layouts=positives, layout20_count=good,
                   layout20=good / positives if positives else None,
                   candidate_coverage_count=covered,
                   candidate_coverage=covered / positives if positives else None,
                   covered_but_winner_wrong=sum(row['covered'] and not row['good'] for row in rows),
                   winner_correct_but_rejected=good - joint,
                   positive_no_correct_candidate=positives - covered,
                   wrong_pose_accepted=wrong, joint_tp=joint, joint_fp=jfp, joint_fn=jfn,
                   joint_precision=joint / max(1, joint + jfp) if positives else None,
                   joint_recall=joint / positives if positives else None,
                   joint_f1=2 * joint / max(1, 2 * joint + jfp + jfn) if positives else None)
    return out


def calibrate(cal, split, role='real_cal'):
    require(role == 'real_cal', 'threshold fitting restricted to REAL-CAL')
    require(split in ('dunhuang_cv', 'turufan'), 'no SELECT threshold calibration')
    require(any(row['label'] for row in cal) and not all(row['label'] for row in cal),
            'CAL needs both classes')
    target = 'joint_f1' if split == 'dunhuang_cv' else 'f1'
    precision = 'joint_precision' if split == 'dunhuang_cv' else 'precision'
    choices = []
    for integer in range(20, 81):
        threshold = integer / 100.
        value = metrics(cal, threshold, layout_available=split != 'turufan')
        choices.append(((value[target], -abs(threshold - .30), value[precision], threshold), value))
    # This is the already registered real-development rule, not a new objective.
    return max(choices, key=lambda item: item[0])[1]


def transitions(baseline, alternative, base_threshold, threshold, *, layout_available):
    old = {row['pair_id']: row for row in baseline}
    require(len(old) == len(baseline) == len(alternative) and
            set(old) == {row['pair_id'] for row in alternative}, 'unpaired comparison')
    keys = ('classification_gained', 'classification_lost', 'positive_recovered', 'positive_lost',
            'false_positive_added', 'false_positive_removed', 'layout_gained', 'layout_lost',
            'coverage_gained', 'coverage_lost', 'correct_and_accepted_gained',
            'correct_and_accepted_lost', 'winner_pose_changed')
    ids = {key: [] for key in keys}
    for row in alternative:
        before = old[row['pair_id']]; pid = row['pair_id']
        require(before['label'] == row['label'] and before['known'] == row['known'], 'paired truth changed')
        was, now = accepted(before, base_threshold), accepted(row, threshold)
        correct_was, correct_now = was == row['label'], now == row['label']
        if correct_now and not correct_was: ids['classification_gained'].append(pid)
        if correct_was and not correct_now: ids['classification_lost'].append(pid)
        if row['label']:
            if now and not was: ids['positive_recovered'].append(pid)
            if was and not now: ids['positive_lost'].append(pid)
        else:
            if now and not was: ids['false_positive_added'].append(pid)
            if was and not now: ids['false_positive_removed'].append(pid)
        if row['known']:
            for source, prefix in (('good', 'layout'), ('covered', 'coverage')):
                if row[source] and not before[source]: ids[prefix + '_gained'].append(pid)
                if before[source] and not row[source]: ids[prefix + '_lost'].append(pid)
            if row['good'] and now and not (before['good'] and was):
                ids['correct_and_accepted_gained'].append(pid)
            if before['good'] and was and not (row['good'] and now):
                ids['correct_and_accepted_lost'].append(pid)
        changed = ((row['pose'] is None) != (before['pose'] is None) or
                   (row['pose'] is not None and before['pose'] is not None and
                    math.dist(row['pose'], before['pose']) > 1e-5))
        if changed: ids['winner_pose_changed'].append(pid)
    counts = {key: len(value) for key, value in ids.items()}
    if not layout_available:
        for key in keys:
            if key.startswith(('layout_', 'coverage_', 'correct_and_accepted_')):
                counts[key] = None; ids[key] = None
    return dict(counts=counts, pair_ids=ids)


def quantiles(values):
    values = sorted(values)
    if not values:
        return dict(count=0, mean=None, p10=None, p25=None, p50=None, p75=None, p90=None)
    def percentile(p):
        x = (len(values) - 1) * p; lo = int(x); hi = math.ceil(x)
        return values[lo] * (hi - x) + values[hi] * (x - lo) if hi != lo else values[lo]
    return dict(count=len(values), mean=sum(values) / len(values),
                **{'p' + str(p): percentile(p / 100.) for p in (10, 25, 50, 75, 90)})


def distributions(rows):
    return {name: {field: quantiles([row[field] for row in group])
                   for field in ('pairs', 'sum_q', 'max_q', 'candidate_count', 'score')}
            for name, group in (
                ('all_positive_winners', [row for row in rows if row['label']]),
                ('all_negative_winners', [row for row in rows if not row['label']]),
                ('correct_positive_winners', [row for row in rows if row['good']]))}


def load_verified(root, model, plan, source_plan, sim_manifest, real_manifests):
    root = Path(root); folder = root / 'full_development_01' / model
    require(not (root / 'controller_failure.json').exists() and not (folder / 'failure.json').exists(),
            'failure takes precedence over stale running/complete')
    complete, protocol = read(folder / 'complete.json'), read(folder / 'protocol.json')
    require(complete['status'] == 'complete' and complete['model_unchanged'] is True and
            complete['pairs'] == 2619 and complete['population'] == POPULATION,
            'full development evidence incomplete; pilot not accepted')
    require(read(root / ('full_development_01_' + model + '_exit.json'))['returncode'] == 0,
            'worker did not exit successfully')
    require(complete['records_sha256'] == sha(folder / 'records.jsonl') and
            complete['summary_sha256'] == sha(folder / 'summary.json'), 'output SHA mismatch')
    require(protocol['pilot_limit'] is None and protocol['model'] == model and
            protocol['source_plan_sha256'] == sha(source_plan) and
            protocol['real_roles_sha256'] == sha(plan), 'wrong source/role plan or pilot')
    for name, key in (('run_factor_sweep.py', 'script_sha256'), ('decoder.py', 'decoder_sha256')):
        require(protocol[key] == sha(root / 'source_01' / name), 'bound sweep source changed')
    require(all(protocol[key] is False for key in ('threshold_refitted', 'training_performed',
                                                   'gpu_used', 'test_used')), 'not frozen CPU controls')
    require(.20 <= protocol['threshold'] <= .80, 'threshold outside protocol')
    roles = check_roles(read(plan)); population = defaultdict(int); seen = set()
    sim = read(sim_manifest)['entries']
    sim_labels = {row['pair_id']: row['label'] for row in sim}
    require(len(sim_labels) == len(sim) == 1500, 'SIM SELECT manifest incomplete/duplicated')
    real_labels = {}
    for split, path in real_manifests.items():
        require(sha(path) == read(plan)['datasets'][split]['manifest_sha256'], 'real manifest changed')
        rows = read(path)['pairs']; real_labels[split] = {r['pair_id']: r['label'] for r in rows}
        require(len(real_labels[split]) == len(rows), 'duplicate real source ID')
    groups = defaultdict(list); equivalence = defaultdict(lambda: {'strict': 0, 'sensitive': 0, 'sensitive_ids': []})
    with (folder / 'records.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line); check_record(row, model, roles, sim_labels, real_labels)
            split, pid, role = row['split'], row['pair_id'], row['role']
            require((split, pid) not in seen, 'duplicate prediction identity')
            seen.add((split, pid)); population[split] += 1
            for search, prediction in row['predictions'].items():
                for kind in prediction['readouts']:
                    groups[(split, role, search, kind)].append(outcome(row, search, kind))
            eq = row['reference_equivalence']
            if split != 'sim_select':
                require(eq is not None, 'CPU/GPU equivalence evidence absent')
                key = 'strict' if eq['strict'] else 'sensitive'; equivalence[split][key] += 1
                if not eq['strict']: equivalence[split]['sensitive_ids'].append(pid)
    require(dict(population) == POPULATION, 'row population differs from complete receipt')
    require({pid for split, pid in seen if split == 'sim_select'} == set(sim_labels), 'SIM SELECT IDs differ')
    for split, mapping in roles.items():
        require({pid for ds, pid in seen if ds == split} == set(mapping), 'real development IDs differ')
    return groups, protocol, dict(equivalence)


def analyze_groups(groups, protocol, model, reference_equivalence=None):
    rows = []; frozen = protocol['threshold']; choices = {}
    for split in ('dunhuang_cv', 'turufan'):
        for search in SEARCHES:
            for kind in expected_readouts(model):
                choices[(split, search, kind)] = calibrate(groups[(split, 'real_cal', search, kind)], split)
    for (split, role, search, kind), values in sorted(groups.items()):
        baseline = groups[(split, role, *BASELINE)]
        comparisons = [('original_frozen_cal', frozen, frozen)]
        if split != 'sim_select':
            comparisons.append(('separate_real_cal', choices[(split, search, kind)]['threshold'],
                                choices[(split, *BASELINE)]['threshold']))
        for mode, threshold, baseline_threshold in comparisons:
            current = metrics(values, threshold, layout_available=split != 'turufan')
            original = metrics(baseline, baseline_threshold, layout_available=split != 'turufan')
            paired = transitions(baseline, values, baseline_threshold, threshold, layout_available=split != 'turufan')
            delta = {key: current[key] - original[key] if current[key] is not None else None
                     for key in ('accuracy', 'f1', 'fp', 'layout20_count', 'candidate_coverage_count',
                                 'joint_tp', 'joint_f1', 'winner_correct_but_rejected')}
            require(paired['counts']['classification_gained'] - paired['counts']['classification_lost'] ==
                    current['tp'] + current['tn'] - original['tp'] - original['tn'], 'paired count arithmetic failed')
            strict = None
            if reference_equivalence is not None and split != 'sim_select':
                excluded = set(reference_equivalence[split]['sensitive_ids'])
                strict_values = [row for row in values if row['pair_id'] not in excluded]
                strict_baseline = [row for row in baseline if row['pair_id'] not in excluded]
                strict = dict(pairs=len(strict_values), excluded_pairs=len(values) - len(strict_values),
                    threshold_refitted_on_subset=False,
                    metrics=(metrics(strict_values, threshold, layout_available=split != 'turufan')
                             if strict_values else None),
                    paired=(transitions(strict_baseline, strict_values, baseline_threshold, threshold,
                                        layout_available=split != 'turufan') if strict_values else None))
            rows.append(dict(split=split, role=role, search=search, readout=kind, threshold_mode=mode,
                             mechanism_only=kind in POOLING_DIAGNOSTICS or kind == 'zero_overlap_feature',
                             not_a_calibrated_q_classifier=kind.startswith('raw_'),
                             metrics=current, baseline_threshold=baseline_threshold, delta=delta,
                             paired=paired, distributions=distributions(values),
                             strict_cpu_gpu_reproduction_subset=strict,
                             runtime_seconds=quantiles([row['seconds'] for row in values])))
    # Development ranking is NOT heldout validation or automatic promotion.
    ranking = []
    for search in SEARCHES:
        for kind in sorted(expected_readouts(model) - POOLING_DIAGNOSTICS):
            selected = [row for row in rows if row['search'] == search and row['readout'] == kind and
                        row['role'] == 'real_select' and row['threshold_mode'] == 'separate_real_cal']
            by_domain = {row['split']: row for row in selected}
            d, t = by_domain['dunhuang_cv'], by_domain['turufan']
            sim = next(row for row in rows if row['search'] == search and row['readout'] == kind
                       and row['role'] == 'sim_select')
            ranking.append(dict(search=search, readout=kind,
                development_value=(d['metrics']['joint_f1'] + t['metrics']['f1']) / 2,
                baseline_delta=(d['delta']['joint_f1'] + t['delta']['f1']) / 2,
                dunhuang_layout_delta=d['delta']['layout20_count'],
                false_positive_delta={'dunhuang_cv': d['delta']['fp'], 'turufan': t['delta']['fp']},
                sim_select_delta=sim['delta'],
                sim_select_dataset='v17' if model == 'aggressive_binary_patch' else 'v14',
                production_retraining_required=kind == 'zero_overlap_feature',
                scope='development candidate, not deployment or heldout improvement'))
    ranking.sort(key=lambda row: (-row['development_value'], row['search'], row['readout']))
    return dict(model=model, rows=rows, development_ranking=ranking,
                new_pooling_heads_included=False, automatic_production_change=False)


def collect_analysis(root, role_plan, source_plan, sim_manifests, real_manifests, models=None):
    """Admit complete models without calling a running whole sweep complete.

    Default/all-model mode retains the original controller completion gate.
    An explicit subset still requires each selected model's complete 2619-row
    evidence and successful worker exit through load_verified. It never consumes
    partial rows or infers completion from a status count.
    """
    selected = tuple(MODELS if models is None else models)
    require(selected and len(set(selected)) == len(selected) and set(selected) <= set(MODELS),
            'invalid, duplicated or empty requested model set')
    require(set(sim_manifests) == set(selected), 'exact model-specific SIM SELECT manifests required')
    require(set(real_manifests) == {'dunhuang_cv', 'turufan'}, 'both actual real manifests required')
    root = Path(root)
    require(not (root / 'controller_failure.json').exists(), 'controller failure takes precedence')
    completion_path = root / 'complete.json'
    completion_observed = completion_path.exists()
    if completion_observed:
        complete = read(completion_path)
        require(complete['status'] == 'complete' and complete['pairs'] == 10476 and
                set(complete['models']) == set(MODELS), 'controller complete receipt invalid')
    all_requested = set(selected) == set(MODELS)
    require(not all_requested or completion_observed, 'controller not complete')
    output = dict(schema='decoder-controls-independent-analysis/1',
                  status='complete' if all_requested else 'model_complete',
                  requested_models=list(selected), all_models_analyzed=all_requested,
                  controller_completion_observed=completion_observed,
                  test_used=False, training_performed=False, model_modified=False,
                  historical_real_exposure=True, analysis_source_sha256=sha(__file__),
                  controller_complete_sha256=sha(completion_path) if completion_observed else None,
                  real_role_plan_sha256=sha(role_plan), models={})
    for model in selected:
        groups, protocol, eq = load_verified(root, model, role_plan, source_plan,
                                             sim_manifests[model], real_manifests)
        output['models'][model] = dict(analyze_groups(groups, protocol, model, eq),
            protocol=protocol, reference_equivalence=eq,
            model_complete_sha256=sha(root / 'full_development_01' / model / 'complete.json'),
            sim_select_manifest_sha256=sha(sim_manifests[model]))
    return output


def _assignments(values):
    pairs = [item.split('=', 1) for item in values]
    require(all(len(item) == 2 and all(item) for item in pairs), 'expected NAME=path')
    result = dict(pairs)
    require(len(result) == len(pairs), 'duplicate manifest assignment')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--role-plan', required=True)
    parser.add_argument('--source-plan', required=True)
    parser.add_argument('--sim-manifest', action='append', required=True, help='MODEL=SELECT manifest')
    parser.add_argument('--real-manifest', action='append', required=True, help='SPLIT=manifest')
    parser.add_argument('--model', action='append', choices=MODELS,
                        help='Explicit completed-model subset; omitted means the full four-model sweep')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    sim = _assignments(args.sim_manifest); real = _assignments(args.real_manifest)
    output = collect_analysis(args.root, args.role_plan, args.source_plan, sim, real, args.model)
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=output['status'], output=str(path), sha256=sha(path),
                         models=len(output['models']), all_models_analyzed=output['all_models_analyzed'])))


if __name__ == '__main__':
    main()
