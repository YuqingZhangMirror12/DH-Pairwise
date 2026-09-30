"""Audit raw union evidence versus trained scoring from frozen outputs only.

No network forward, parameter change, new candidate, or model selection. Pure
sum(Q) is a secondary audit restricted to pairs with EVERY candidate aligned
to the frozen cache. Main M uses all final saved candidates, Q times observed
arc length before learned support/conflict gates. Ranking comparisons keep
the final poses fixed, so they do not credit refinement to candidate ranking.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, kendalltau


ROOT = Path('artifacts/threshold_m12_evaluation_20260927')
BASE = Path('artifacts/threshold_diagnosis_20260927')
OUT = Path('artifacts/matcher_evidence_gap_20260927/ranking')
MASS = 'observed_mass_length_px'
MODES = (MASS, 'logit', 'perfect_local_score', 'no_conflict_score', 'no_overlap_score')
JOBS = ('m12_sim_test_v14', 'm12_dunhuang_cv', 'm12_turufan')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qs(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                **{k: float(np.quantile(x, p)) if len(x) else None for k, p in
                   [('p10', .1), ('p25', .25), ('median', .5), ('p75', .75), ('p90', .9)]})


def best(row, field):
    # Saved candidate order, also the model's torch.argmax tie order.
    return max(row['candidates'], key=lambda c: c[field]) if row['candidates'] else None


def auc(labels, scores):
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    assert np.isfinite(s).all() and y.any() and (~y).any()
    r = rankdata(s, method='average')
    p, n = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - p*(p+1)/2)/(p*n))


def cut_at_fp(negative_scores, budget):
    """Negative-only retrospective cut; reject an entire boundary tie."""
    values = sorted(negative_scores, reverse=True)
    return float(np.nextafter(values[budget], np.inf)) if budget < len(values) else -math.inf


def correct(candidate):
    return bool(candidate and candidate.get('gt_error_px') is not None and candidate['gt_error_px'] <= 20)


def outcome(row, field):
    c = best(row, field)
    return dict(pair_id=row['pair_id'], positive=bool(row['label']),
                candidate=c['cluster_id'] if c else None,
                # All candidates are numerically valid in this archive; sentinel
                # merely preserves no-candidate cases in ROC denominators.
                value=float(c[field]) if c and row['numeric_valid'] else -1e10,
                valid=bool(c and row['numeric_valid']), correct=correct(c),
                proposal_correct=bool(c and row['gt_known'] and
                    np.linalg.norm(np.asarray(c['proposal_translation'])-row['target_translation_rc']) <= 20))


def operating(rows, field, budget, layout):
    ds = [outcome(r, field) for r in rows]
    threshold = cut_at_fp([d['value'] for d in ds if not d['positive'] and d['valid']], budget)
    tp = fp = lc = joint = 0
    for d in ds:
        accepted = d['valid'] and d['value'] >= threshold
        tp += int(accepted and d['positive'])
        fp += int(accepted and not d['positive'])
        lc += int(d['positive'] and d['correct'])
        joint += int(d['positive'] and d['correct'] and accepted)
    p = sum(d['positive'] for d in ds)
    assert fp <= budget
    return dict(fp_budget=budget, threshold=threshold, tp=tp, fp=fp,
                fn=p-tp, tn=len(ds)-p-fp, pair_f1=2*tp/(tp+fp+p),
                layout_correct=lc if layout else None, joint_accepted=joint if layout else None,
                retrospective=True, positive_labels_used_to_choose_cut=False)


def rank_comparison(rows, field, layout):
    multi = [r for r in rows if len(r['candidates']) > 1 and r['numeric_valid']]
    agreement, correlations, perfect_order, ties, transitions, detail = Counter(), [], 0, Counter(), Counter(), []
    label_correlations = {'positive': [], 'negative': []}
    for r in multi:
        a, b = best(r, field), best(r, 'logit')
        same = a['cluster_id'] == b['cluster_id']
        group = 'positive' if r['label'] else 'negative'
        agreement[group+'_n'] += 1
        agreement[group+'_same_top'] += int(same)
        x = [c[field] for c in r['candidates']]
        y = [c['logit'] for c in r['candidates']]
        ties['raw_top_ties'] += int(sum(v == max(x) for v in x) > 1)
        ties['scorer_top_ties'] += int(sum(v == max(y) for v in y) > 1)
        if len(set(x)) > 1 and len(set(y)) > 1:
            correlation = (float(spearmanr(x, y).statistic), float(kendalltau(x, y).statistic))
            correlations.append(correlation)
            label_correlations[group].append(correlation)
        perfect_order += int(np.array_equal(rankdata(x), rankdata(y)))
    for r in rows:
        a, b = best(r, field), best(r, 'logit')
        if not r['label'] or not layout:
            continue
        ac, bc = correct(a), correct(b)
        transition = ('correct' if ac else 'wrong')+'_to_'+('correct' if bc else 'wrong')
        transitions[transition] += 1
        if a and b and a['cluster_id'] != b['cluster_id']:
            keys = ('cluster_id', MASS, 'score', 'logit', 'positive_evidence_px',
                    'conflict_evidence_px', 'support_fraction', 'conflict_fraction',
                    'overlap_penalty', 'gt_error_px', 'proposal_translation', 'refined_translation')
            detail.append(dict(pair_id=r['pair_id'], transition=transition,
                raw={k:a[k] for k in keys}, scorer={k:b[k] for k in keys}))
    return dict(multi_pairs=len(multi), same_top=sum(agreement[k] for k in ('positive_same_top','negative_same_top')),
                changed_top=sum(agreement[k] for k in ('positive_n','negative_n'))-
                    sum(agreement[k] for k in ('positive_same_top','negative_same_top')),
                by_label=dict(agreement), completely_same_order=perfect_order,
                ties=dict(ties), spearman=qs([s for s,k in correlations]),
                kendall=qs([k for s,k in correlations]),
                correlation_by_label={g:dict(spearman=qs([s for s,k in v]),
                    kendall=qs([k for s,k in v])) for g,v in label_correlations.items()},
                positive_layout_transitions=dict(transitions) if layout else None,
                examples=detail)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [BASE/'pair_diagnosis.json', BASE/'summary.json',
             Path('artifacts/threshold_rootcause_20260927/joined_candidate_counts.json')]
    sources = {str(p): sha(p) for p in paths}
    derived = read(paths[0]); previous = read(paths[1]); joined = read(paths[2])
    results = {}; all_decisions = []
    for job in JOBS:
        directory = ROOT/job
        complete = read(directory/'prediction_complete.json')
        assert complete['model_state_unchanged'] is True
        assert complete['checkpoint_sha256'] == previous['checkpoint_sha256']
        prediction_path = directory/'pair_predictions.jsonl'
        assert sha(prediction_path) == complete['sha256']
        original = {r['pair_id']:r for r in map(json.loads, prediction_path.read_text().splitlines())}
        rows = [r for r in derived if r['dataset'] == job]
        assert len({r['pair_id'] for r in rows}) == len(rows)
        for r in rows:
            p = original[r['pair_id']]
            assert p['selected_cluster_id'] == r['selected_cluster_id']
            assert len(p['candidates']) == len(r['candidates'])
            for c, d in zip(p['candidates'], r['candidates']):
                assert all(c[k] == d[k] for k in c), 'derived values disagree with immutable prediction'
            if r['candidates']:
                assert best(r, 'logit')['cluster_id'] == r['selected_cluster_id']
        for file in ('pair_predictions.jsonl', 'case_diagnostics.jsonl', 'prediction_complete.json'):
            sources[str(directory/file)] = sha(directory/file)
            if str(directory/file) in previous['sources']:
                assert sources[str(directory/file)] == previous['sources'][str(directory/file)]
        layout = job != 'm12_turufan'
        p = sum(r['label'] for r in rows); n = len(rows)-p
        result = dict(pairs=len(rows), positives=p, negatives=n,
            candidate_counts=dict(Counter(len(r['candidates']) for r in rows)),
            numeric_invalid=sum(not r['numeric_valid'] for r in rows),
            raw_mass_first_matches_builder_order=sum(bool(r['candidates']) and
                best(r, MASS)['cluster_id'] == r['candidates'][0]['cluster_id'] for r in rows),
            ranking=rank_comparison(rows, MASS, layout), methods={})
        for field in MODES:
            ds = [outcome(r, field) for r in rows]
            result['methods'][field] = dict(
                auc=auc([d['positive'] for d in ds], [d['value'] for d in ds]),
                layout_correct_final_poses=sum(d['correct'] and d['positive'] for d in ds) if layout else None,
                layout_correct_proposal_poses=sum(d['proposal_correct'] and d['positive'] for d in ds) if layout else None,
                matched_fp=[operating(rows, field, b, layout) for b in sorted({1, 5, round(n*32/508)})])
            all_decisions.extend(dict(dataset=job, method=field, **d) for d in ds)
        # Cross-pair candidate accuracy: what the head gates on correct versus
        # incorrect native layouts, excluding classification of negative pairs.
        groups = {}
        for name, predicate in (
                ('gt_correct_positive', lambda r,c: r['label'] and correct(c)),
                ('gt_wrong_positive', lambda r,c: r['label'] and not correct(c)),
                ('negative', lambda r,c: not r['label'])):
            if not layout and name != 'negative': continue
            cs = [c for r in rows for c in r['candidates'] if predicate(r,c)]
            groups[name] = {k: qs([c[k] for c in cs if c[k] is not None]) for k in
                           (MASS,'support_fraction','conflict_fraction','score','overlap_penalty')}
            groups[name]['support_below_10pct'] = sum(c['support_fraction'] is not None and
                c['support_fraction'] < .1 for c in cs)
            groups[name]['support_below_50pct'] = sum(c['support_fraction'] is not None and
                c['support_fraction'] < .5 for c in cs)
        result['local_gate_distributions'] = groups
        if layout:
            result['final_pose_coverage'] = sum(r['label'] and any(correct(c) for c in r['candidates']) for r in rows)
        # A second audit uses unweighted sum(Q), not M. Only whole-pair-complete
        # alignment is allowed. Never choose a winner after dropping mismatches.
        cache = {(r['pair_id'], r['cluster_id']): r for r in joined if r['dataset']==job}
        if cache:
            aligned, excluded = [], []
            for r in rows:
                if all((r['pair_id'], c['cluster_id']) in cache for c in r['candidates']):
                    copy = dict(r, candidates=[dict(c, q_sum=cache[(r['pair_id'],c['cluster_id'])]['q_sum'])
                                               for c in r['candidates']])
                    aligned.append(copy)
                else:
                    excluded.append(r['pair_id'])
            result['pure_q_sum_secondary'] = dict(pairs=len(aligned), excluded_pair_ids=excluded,
                ranking=rank_comparison(aligned,'q_sum',layout))
        results[job] = result
    assert [(results[j]['pairs'],results[j]['positives']) for j in JOBS] == [(3000,1500),(800,292),(602,301)]
    assert results['m12_dunhuang_cv']['methods']['logit']['layout_correct_final_poses'] == 212
    result = dict(status='complete', schema='frozen-scorer-ranking/1',
        checkpoint_sha256=previous['checkpoint_sha256'], datasets=results, sources=sources,
        baseline_definition='M = sum of deduplicated union Q weighted by observed arc length, before learned gates',
        scope='Saved E8 outputs only. Ranking holds all final refined poses fixed; no network forward.',
        caveats=['At most eight clusters, not eight Sinkhorn passes.',
            'Matched-FP cuts are retrospective development diagnostics, not deployed or newly selected models.',
            'Raw M at final poses retains the existing neural refinement; not an end-to-end no-head system.',
            'Pure sum(Q) secondary results exclude whole pairs with any cache alignment mismatch.',
            'Candidate rows and repeated manuscript families are not independent statistical samples.',
            'Turufan has no layout GT; never infer correct layouts there.'])
    for name, value in [('summary.json',result),('pair_decisions.json',all_decisions)]:
        (OUT/name).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    assert all(sha(p)==h for p,h in sources.items())
    for name,r in results.items():
        print(name, json.dumps({k:v for k,v in r.items() if k not in ('ranking','local_gate_distributions','pure_q_sum_secondary')},ensure_ascii=False))
        print('RANK', json.dumps({k:v for k,v in r['ranking'].items() if k!='examples'},ensure_ascii=False))
        if 'pure_q_sum_secondary' in r:
            q=r['pure_q_sum_secondary']; print('QSUM',q['pairs'],len(q['excluded_pair_ids']),
                json.dumps({k:v for k,v in q['ranking'].items() if k!='examples'},ensure_ascii=False))


if __name__ == '__main__':
    main()
