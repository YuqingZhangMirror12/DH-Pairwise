"""Separate input evidence, local gating, readout penalties, and operating point.

No network execution, new training, geometric tuning, or deployed score change.
Matched-FP thresholds use negative scores only, on already exposed development
data. They are descriptive diagnostics, NOT held-out/calibrated performance.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .analyze import read, sha, quantiles, sigmoid, DIAG

MODES = ('score', 'no_overlap_score', 'no_conflict_score', 'perfect_local_score')
NAMES = {'m12_sim_test_v14': '仿真 TEST', 'm12_dunhuang_cv': '敦煌', 'm12_turufan': 'Turufan'}


def decisions(rows, mode, threshold, has_layout):
    result = []
    for r in rows:
        c = max(r['candidates'], key=lambda x: x[mode]) if r['candidates'] else None
        valid = bool(r['numeric_valid'] and c)
        correct = bool(c and c['gt_error_px'] <= 20) if has_layout and r['label'] else None
        result.append(dict(pair_id=r['pair_id'], positive=r['label'], valid=valid,
            winner=c['cluster_id'] if c else None, score=c[mode] if c else None,
            accepted=bool(valid and c[mode] >= threshold), correct=correct))
    return result


def metrics(ds, threshold, has_layout):
    counts = Counter(('tp' if d['positive'] else 'fp') if d['accepted']
                     else ('fn' if d['positive'] else 'tn') for d in ds)
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    return dict(**{k: counts[k] for k in ('tp', 'fp', 'fn', 'tn')},
        layout_correct=sum(d['correct'] is True for d in ds) if has_layout else None,
        layout_correct_accepted=sum(d['correct'] is True and d['accepted'] for d in ds) if has_layout else None,
        threshold=threshold, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        false_positive_rate=fp/(counts['fp']+counts['tn']) if counts['fp']+counts['tn'] else None)


def matched_threshold(rows, mode, budget):
    """Lowest cut yielding <= budget false positives, without consulting positives.

    A tie at the boundary is conservatively rejected as a whole. No arbitrary
    tie-breaking by pair ID/label and no maximizing positive recall on this set.
    """
    negative_scores = sorted([max(c[mode] for c in r['candidates']) for r in rows
        if not r['label'] and r['numeric_valid'] and r['candidates']], reverse=True)
    return float(np.nextafter(negative_scores[budget], np.inf)) if budget < len(negative_scores) else 0.


def crossing(before, after, joint=False):
    def ok(d):
        return d['accepted'] and (d['correct'] is True if joint else d['positive'])
    gains, losses = [], []
    for a, b in zip(before, after):
        assert a['pair_id'] == b['pair_id']
        if not ok(a) and ok(b): gains.append(a['pair_id'])
        if ok(a) and not ok(b): losses.append(a['pair_id'])
    return dict(gains=len(gains), losses=len(losses), net=len(gains)-len(losses),
                gained_ids=gains, lost_ids=losses)


def compact(c):
    return {k: v for k, v in c.items() if k not in ('edge_ids', 'q_values', 'widths')}


def main():
    base = Path('artifacts/threshold_diagnosis_20260927')
    out = Path('artifacts/threshold_rootcause_20260927')
    files = [base/'pair_diagnosis.json', base/'summary.json', base/'reviewed_c10.json',
             *[out/(k+'.json') for k in ('sim_select', 'dunhuang_cv', 'turufan')], out/'complete.json']
    sources = {str(p): sha(p) for p in files}
    pairs = read(base/'pair_diagnosis.json'); summary = read(base/'summary.json')
    raw = {k: read(out/(k+'.json')) for k in ('sim_select', 'dunhuang_cv', 'turufan')}
    assert read(out/'complete.json')['pairs'] == 2905
    by_job = {j: [r for r in pairs if r['dataset'] == j] for j in NAMES}
    assert [len(by_job[j]) for j in NAMES] == [3000, 800, 602]
    w = summary['readout_weights']
    # Independent formula reconstruction, including saved counterfactual values.
    max_error = 0.
    for r in pairs:
        for c in r['candidates']:
            p, n, m = (c[k] for k in ('positive_evidence_px', 'conflict_evidence_px', 'observed_mass_length_px'))
            o = c['overlap']['fraction_min_area']
            pos = w['positive']*math.log1p(p/32)
            neg = w['conflict']*math.log1p(n/32)
            ov = w['overlap']*o
            score_values = [sigmoid(w['bias']+pos-neg-ov), sigmoid(w['bias']+pos-neg),
                            sigmoid(w['bias']+pos-ov), sigmoid(w['bias']+w['positive']*math.log1p(m/32)-ov)]
            for mode, value in zip(MODES, score_values):
                max_error = max(max_error, abs(c[mode]-value))
                assert abs(c[mode]-value) < 2e-6
    operating, sensitivity, transitions, detail = [], [], {}, []
    for job, rows in by_job.items():
        has_layout = job != 'm12_turufan'
        nneg = sum(not r['label'] for r in rows)
        budget = round(32/508*nneg)
        ds_by = {}
        for mode in MODES:
            threshold = matched_threshold(rows, mode, budget)
            for condition, t in [('original_0.21', .21), ('matched_fp_budget', threshold)]:
                ds = decisions(rows, mode, t, has_layout)
                ds_by[(mode, condition)] = ds
                m = metrics(ds, t, has_layout)
                if condition == 'matched_fp_budget': assert m['fp'] <= budget
                else:
                    original = summary['jobs'][job]['counterfactual_readouts'][mode]
                    for k in ('tp', 'fp', 'fn', 'tn', 'layout_correct', 'layout_correct_accepted'):
                        assert m[k] == original[k]
                operating.append(dict(dataset=NAMES[job], job=job, mode=mode, condition=condition,
                    fp_budget=budget, negative_pairs=nneg, positive_pairs=len(rows)-nneg,
                    retrospective=condition == 'matched_fp_budget',
                    within_registered_cal_range=.20 <= t <= .80, **m))
                if job == 'm12_dunhuang_cv':
                    detail.extend(dict(mode=mode, condition=condition, threshold=t, **d) for d in ds)
            for b in (0, 1, 5, 16, 32, 50, 100):
                t = matched_threshold(rows, mode, b)
                sensitivity.append(dict(job=job, dataset=NAMES[job], mode=mode, fp_budget=b,
                    **metrics(decisions(rows, mode, t, has_layout), t, has_layout)))
        if has_layout:
            transitions[job] = {mode: crossing(ds_by[('score','matched_fp_budget')],
                ds_by[(mode,'matched_fp_budget')], joint=True) for mode in MODES[1:]}
    # Quantify cache-to-E8 agreement BEFORE joining per-edge counts to head outputs.
    alignment = {}; joined = []
    for split in ('dunhuang_cv','turufan'):
        cache = {r['pair_id']:r for r in raw[split]}
        failures=[]; number=0; aligned=0; positive_failures=0
        for r in by_job['m12_'+split]:
            rc = cache[r['pair_id']]; cc={c['cluster_id']:c for c in rc['clusters']}
            for c in r['candidates']:
                number += 1; q=cc.get(c['cluster_id'])
                dm=abs(q['mass']-c['observed_mass_length_px']) if q else None
                dt=float(np.linalg.norm(np.array(q['proposal_translation'])-c['proposal_translation'])) if q else None
                agrees = q is not None and dm <= 1e-4 and dt <= 1e-3
                if not agrees:
                    failures.append(dict(pair_id=r['pair_id'], label=r['label'], cluster_id=c['cluster_id'],
                        mass_abs_error=dm, proposal_distance_px=dt)); positive_failures += int(r['label']); continue
                aligned += 1
                joined.append(dict(dataset='m12_'+split, pair_id=r['pair_id'], label=r['label'],
                    **compact(q), **{k:c[k] for k in ('selected','score','support_fraction','conflict_fraction',
                        'positive_evidence_px','conflict_evidence_px','observed_mass_length_px',
                        'gt_error_px','perfect_local_score','no_conflict_score','overlap_penalty')}))
        alignment[split]=dict(actual_candidates=number, aligned_candidates=aligned,
            positive_candidate_mismatches=positive_failures,excluded_mismatches=failures,
            tolerances=dict(mass_abs=1e-4,proposal_distance_px=1e-3),
            note='Earlier frozen-M12 cache; only verified matching candidates joined. No full inference rerun.')
    # Compare pre-head correct hypotheses, not a population selected by Scorer success.
    distributions=[]; groups={}; covered={}
    fields=('n','q_mean','q_sum','mass','width_q_weighted','effective_edges','endpoints_a','endpoints_b')
    for split,rows in raw.items():
        valid=[r for r in rows if not r['gt_excluded']]
        group=defaultdict(list); cov=Counter()
        for r in valid:
            if r['label'] and r['usable_gt']:
                cov['positives'] += 1
                good=[c for c in r['clusters'] if c['proposal_gt_error']<=20]
                if good:
                    cov['with_correct_proposal'] += 1
                    group['best_mass_correct_prehead'].append(max(good,key=lambda c:c['mass']))
            if not r['label'] and r['clusters']:
                group['max_mass_negative_prehead'].append(max(r['clusters'],key=lambda c:c['mass']))
        groups[split]=group;covered[split]=dict(cov)
        for name,cs in group.items():
            distributions.append(dict(dataset=split, group=name, samples=len(cs),
                measures={k:quantiles([c[k] for c in cs]) for k in fields}))
    # Exact factorization for each candidate, then geometric-mean domain ratios.
    factors={}
    for split in ('sim_select','dunhuang_cv'):
        cs=groups[split]['best_mass_correct_prehead']
        for c in cs: assert abs(c['n']*c['q_mean']*c['width_q_weighted']-c['mass']) < 1e-8
        factors[split]={k:float(np.exp(np.mean(np.log([c[k] for c in cs])))) for k in
                       ('n','q_mean','width_q_weighted','mass')}
    ratios={k:factors['dunhuang_cv'][k]/factors['sim_select'][k] for k in factors['sim_select']}
    assert abs(ratios['n']*ratios['q_mean']*ratios['width_q_weighted']-ratios['mass'])<1e-10
    # Same-pose evidence bottlenecks in the 70 rejected correct E8 winners.
    jby={(r['dataset'],r['pair_id'],r['cluster_id']):r for r in joined}
    rejected=defaultdict(list)
    for r in by_job['m12_dunhuang_cv']:
        if r['label'] and r['layout20'] and not r['accepted']:
            c=next(c for c in r['candidates'] if c['selected'])
            j=jby[(r['dataset'],r['pair_id'],c['cluster_id'])]
            key='all_support_recovers_at_021' if c['perfect_local_score']>=.21 else 'mass_or_overlap_blocks_at_021'
            rejected[key].append(j)
    rejection={k:dict(cases=len(cs), statistics={f:quantiles([r[f] for r in cs]) for f in
        (*fields,'support_fraction','conflict_fraction','score')},pair_ids=[r['pair_id'] for r in cs]) for k,cs in rejected.items()}
    assert sum(v['cases'] for v in rejection.values())==70
    # Fixed C10 examples: cross-check candidate membership against actual saved network evidence.
    aliases={c['pair_id']:c['alias'] for c in read(DIAG/'s7_consensus_eval_v14/case_plan.json')['cases']}
    examples=[]
    for case in read(base/'reviewed_c10.json')['cases']:
        job='m12_'+case['provenance']['split']
        for c in case['clusters']:
            key=(job,case['pair_id'],c['cluster_id'])
            if key not in jby:continue
            j=jby[key]; saved=c['proposal'].get('original_union_edge_ids',c['proposal']['edge_ids'])
            rr=next(r for r in raw[case['provenance']['split']] if r['pair_id']==case['pair_id'])
            q=next(x for x in rr['clusters'] if x['cluster_id']==c['cluster_id'])
            assert sorted(map(tuple,saved))==list(map(tuple,q['edge_ids']))
            probabilities=[p for side in ('a','b') for p in c['points']['final'][side] if p['evidence_present']]
            examples.append(dict(alias=aliases[case['pair_id']],**j,
                active_endpoints=len(probabilities),
                endpoint_support_probability=quantiles([p['support_probability'] for p in probabilities]),
                endpoint_conflict_probability=quantiles([p['conflict_probability'] for p in probabilities]),
                actual_saved_membership_verified=True))
    result=dict(schema='threshold-rootcause/1',status='complete',checkpoint_sha256=summary['checkpoint_sha256'],
        scope='Read-only frozen E8 diagnosis; no deployment/training/candidate/refinement change',
        operating_points=operating,fp_curves=sensitivity,matched_joint_transitions=transitions,
        input_distributions=distributions,prehead_coverage=covered,cache_alignment=alignment,
        mass_factorization=dict(formula='M=N*meanQ*Q-weighted observation width',geometric_means=factors,
            dun_over_sim_ratios=ratios,causal_attribution=False,
            population='largest-M correct pre-head cluster per covered positive; SIM SELECT750 vs Dun253'),
        correct_rejected_subgroups=rejection,c10_examples=examples,
        zero_evidence_score=sigmoid(w['bias']),readout_weights=w,
        readout_reconstruction_max_score_error=max_error,sources=sources,
        caveats=['All-support is GT-free fixed-readout ablation, not a global recall upper bound or retrained model.',
            'Matched-FP cuts are retrospective, negative-score-only, outside .20-.80 for some conditions; not deployed.',
            'SIM input counts use SELECT1500; score tradeoffs use independent saved TEST3000. Do not conflate.',
            '20px GT residual is a geometric proxy, not exact human correspondence GT.',
            'Count/Q ratios are descriptive; point density, visible overlap and sample composition are not controlled.',
            'Turufan has no layout GT; correct-layout and joint counts are null.'])
    for name,data in [('rootcause.json',result),('joined_candidate_counts.json',joined),('dun_operating_decisions.json',detail)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    assert all(sha(p)==h for p,h in sources.items())
    print(json.dumps(dict(status='complete',operating_points=[r for r in operating if r['job']=='m12_dunhuang_cv'],
        factors=result['mass_factorization'],alignment={k:{kk:vv for kk,vv in v.items() if kk!='excluded_mismatches'} for k,v in alignment.items()},
        examples=[{k:r[k] for k in ('alias','n','q_mean','mass','support_fraction','score','perfect_local_score')} for r in examples if r['selected']]),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
