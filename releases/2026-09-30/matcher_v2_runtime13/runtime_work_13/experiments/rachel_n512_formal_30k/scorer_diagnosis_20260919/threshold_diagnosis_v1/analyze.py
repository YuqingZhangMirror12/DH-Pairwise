"""Reconstruct every saved candidate score; never train or alter predictions.

The all-support ablation is an analytic score ceiling at the saved pose and Q,
not a trained replacement or a claim that every union member is a correct match.
It is GT-free (P=M, C=0 for every candidate), not a GT oracle. It is NOT a global
layout/recall ceiling: reranking and operating-point tradeoffs must be measured.
Counterfactual readouts preserve candidates/poses/local probabilities except
for the explicitly changed term; they do not rerun candidate selection/refit.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

DIAG = Path(__file__).resolve().parents[1]
EXPECTED_SHA = '367d2b496ea789902e18847760303387035992701d455dab0f499ee3a6c1ca72'
JOBS = ('m12_sim_test_v14', 'm12_dunhuang_cv', 'm12_turufan')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(values):
        return dict(n=0, minimum=None, p10=None, median=None, p90=None, maximum=None, mean=None)
    assert np.isfinite(values).all()
    return dict(n=len(values), minimum=float(values.min()), p10=float(np.quantile(values, .1)),
                median=float(np.median(values)), p90=float(np.quantile(values, .9)),
                maximum=float(values.max()), mean=float(values.mean()))


def sigmoid(x):
    return float(1 / (1 + math.exp(-x)))


def decompose(c, w):
    p, n, m = (c[k] for k in ('positive_evidence_px', 'conflict_evidence_px', 'observed_mass_length_px'))
    o = c['overlap']['fraction_min_area']
    assert o is not None, 'do not replace unavailable overlap with zero'
    assert p >= 0 and n >= 0 and m >= 0 and p+n <= m+2e-4
    positive = w['positive'] * math.log1p(p / w['length_scale_px'])
    conflict = w['conflict'] * math.log1p(n / w['length_scale_px'])
    overlap = w['overlap'] * o
    logit = w['bias'] + positive - conflict - overlap
    perfect = w['bias'] + w['positive'] * math.log1p(m / w['length_scale_px']) - overlap
    return dict(bias=w['bias'], positive_logit=positive, conflict_penalty=conflict,
                overlap_penalty=overlap, reconstructed_logit=logit,
                reconstruction_abs=abs(logit-c['logit']),
                reconstructed_score=sigmoid(logit), no_overlap_score=sigmoid(logit+overlap),
                no_conflict_score=sigmoid(logit+conflict),
                perfect_local_score=sigmoid(perfect),
                perfect_local_no_overlap_score=sigmoid(perfect+overlap),
                support_fraction=p/m if m else None, conflict_fraction=n/m if m else None,
                unknown_fraction=max(0., (m-p-n)/m) if m else None)


def count_decisions(rows, score_key, threshold, has_layout):
    counts = Counter()
    for r in rows:
        cc = r['candidates']
        winner = max(cc, key=lambda c: c[score_key]) if cc else None
        accepted = bool(winner and r['numeric_valid'] and winner[score_key] >= threshold)
        positive = r['label']
        counts['tp' if accepted and positive else 'fp' if accepted else 'fn' if positive else 'tn'] += 1
        if positive and has_layout:
            correct = winner is not None and winner['gt_error_px'] <= 20
            counts['layout_correct'] += int(correct)
            counts['layout_correct_accepted'] += int(correct and accepted)
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    return dict({key:counts[key] for key in ('tp','fp','fn','tn')},
                layout_correct=counts['layout_correct'] if has_layout else None,
                layout_correct_accepted=counts['layout_correct_accepted'] if has_layout else None,
                f1=2*tp/(2*tp+fp+fn), threshold=threshold,
                local_class_oracle=score_key.startswith('perfect'), layout_available=has_layout)


def run(root, checkpoint, out):
    root, checkpoint, out = Path(root), Path(checkpoint), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert sha(checkpoint) == EXPECTED_SHA
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert cp['epoch'] == 8 and cp['stage'] == 'scorer' and cp['threshold'] == .21
    state = cp['model']
    w = {'bias':float(state['head.bias']), 'length_scale_px':float(state['head.length_scale_px'])}
    w.update({k:float(torch.nn.functional.softplus(state['head.'+k+'_scale']))
              for k in ('positive', 'conflict', 'overlap')})
    layers = []
    for key, tensor in state.items():
        if not key.startswith('head.'):
            continue
        t = tensor.detach().double()
        sample = t if t.ndim == 0 else t[:8] if t.ndim == 1 else t[:8, :8]
        layers.append(dict(name=key, shape=list(t.shape), count=t.numel(),
            mean=float(t.mean()), std=float(t.std(unbiased=False)), minimum=float(t.min()),
            maximum=float(t.max()), l2=float(t.norm()), sample=sample.tolist(),
            sample_note='first8 entries per dimension, not averaged or selected by magnitude'))
    excluded = set(read(DIAG/'s7_consensus_eval_v14/case_plan.json')['user_confirmed_gt_exclusions'])
    result = dict(schema='threshold-frozen-diagnosis/1', status='complete',
        checkpoint_sha256=sha(checkpoint), epoch=8, threshold=.21, readout_weights=w,
        readout_formula='bias + positive*log1p(P/32) - conflict*log1p(C/32) - overlap*O',
        scope='saved candidates and selected E8 only; no Matcher or full evaluation rerun',
        confidence_note='analytic score decomposition is exact; domain-shift causality needs controlled training',
        minimum_mass_at_zero_overlap={str(t):w['length_scale_px']*math.expm1((math.log(t/(1-t))-w['bias'])/w['positive']) for t in (.2,.21,.3)},
        jobs={}, sources={str(checkpoint):sha(checkpoint)}, counterfactuals_are_deployed=False)
    candidates_out, pairs_out = [], []
    for job in JOBS:
        directory = root/job
        protocol, complete, status = [read(directory/f) for f in ('protocol.json','prediction_complete.json','status.json')]
        assert status['status'] == 'complete' and complete['model_state_unchanged'] is True
        assert protocol['checkpoint_sha256'] == EXPECTED_SHA and protocol['threshold'] == .21
        assert sha(directory/'pair_predictions.jsonl') == complete['sha256']
        raw = [json.loads(x) for x in (directory/'case_diagnostics.jsonl').read_text().splitlines()]
        assert len(raw) == len({r['pair_id'] for r in raw})
        has_layout = job != 'm12_turufan'
        keep = [r for r in raw if not (job == 'm12_dunhuang_cv' and r['pair_id'] in excluded)]
        groups = defaultdict(list)
        funnel = Counter()
        maximum_error = 0.
        for r in keep:
            assert r['candidate_count'] == len(r['candidates']) == len(r['candidate_errors_px'])
            for i, c in enumerate(r['candidates']):
                c.update(decompose(c,w))
                maximum_error = max(maximum_error, c['reconstruction_abs'])
                assert abs(c['reconstructed_score']-c['score']) < 2e-6
                c['gt_error_px'] = r['candidate_errors_px'][i] if r['gt_known'] else None
                if r['gt_known']:
                    actual = float(np.linalg.norm(np.asarray(c['refined_translation'])-r['target_translation_rc']))
                    assert abs(actual-c['gt_error_px']) < 1e-4
                candidates_out.append(dict(dataset=job,pair_id=r['pair_id'],label=r['label'],**c))
                if r['gt_known']:
                    groups['correct_candidates' if c['gt_error_px'] <= 20 else 'wrong_candidates'].append(c)
                elif not r['label']:
                    groups['negative_candidates'].append(c)
            selected = [c for c in r['candidates'] if c['selected']]
            assert len(selected) == int(r['has_candidate'])
            winner = selected[0] if selected else None
            if winner:
                assert winner['score'] == r['score'] and winner['score'] == max(c['score'] for c in r['candidates'])
                groups['positive_winners' if r['label'] else 'negative_winners'].append(winner)
                if r['label'] and has_layout and r['layout20']:
                    groups['correct_winners'].append(winner)
                    groups['correct_winners_accepted' if r['accepted'] else 'correct_winners_rejected'].append(winner)
                elif r['label'] and has_layout:
                    groups['wrong_winners'].append(winner)
            if r['label'] and has_layout:
                correct = [c for c in r['candidates'] if c['gt_error_px'] <= 20]
                bucket = ('no_correct_candidate' if not correct else 'correct_candidate_wrong_winner' if not r['layout20']
                          else 'correct_winner_rejected' if not r['accepted'] else 'correct_winner_accepted')
                funnel[bucket] += 1
                if correct:
                    groups['best_correct_per_positive'].append(max(correct,key=lambda c:c['score']))
            pairs_out.append(dict(dataset=job,**r))
        fields = ('score','observed_mass_length_px','positive_evidence_px','conflict_evidence_px',
                  'support_fraction','conflict_fraction','unknown_fraction','positive_logit',
                  'conflict_penalty','overlap_penalty','perfect_local_score')
        distributions = {g:{k:quantiles([c[k] for c in cc]) for k in fields} for g,cc in groups.items()}
        rejection = {}
        for group in ('correct_winners_rejected','best_correct_per_positive','positive_winners','negative_winners'):
            cc = groups[group]
            rr = [c for c in cc if c['score'] < .21]
            rejection[group] = dict(total=len(cc), below_threshold=len(rr),
                no_overlap_recovers=sum(c['no_overlap_score'] >= .21 for c in rr),
                no_conflict_recovers=sum(c['no_conflict_score'] >= .21 for c in rr),
                perfect_local_recovers=sum(c['perfect_local_score'] >= .21 for c in rr),
                impossible_even_perfect_local=sum(c['perfect_local_score'] < .21 for c in rr),
                impossible_even_perfect_local_no_overlap=sum(c['perfect_local_no_overlap_score'] < .21 for c in rr))
        pair_features = {}
        for group, subset in [('all_positive',[r for r in keep if r['label']]),
                              ('all_negative',[r for r in keep if not r['label']]),
                              ('correct_winners',[r for r in keep if r['label'] and has_layout and r['layout20']])]:
            pair_features[group] = {key:quantiles([r[key] for r in subset]) for key in
                ('absolute_q_mass','unmatched_a_mean','candidate_count')}
        result['jobs'][job] = dict(original_pairs=len(raw), pairs=len(keep), positives=sum(r['label'] for r in keep),
            candidates=sum(len(r['candidates']) for r in keep), funnel=dict(funnel),
            reconstruction_max_abs=maximum_error, distributions=distributions, rejection=rejection,
            pair_features=pair_features,
            counterfactual_readouts={key:count_decisions(keep,key,.21,has_layout) for key in
                ('score','no_overlap_score','no_conflict_score','perfect_local_score')},
            sensitivity={str(t):count_decisions(keep,'score',t,has_layout) for t in (.2,.21,.3)})
        assert maximum_error < 3e-6
        if has_layout:
            assert sum(funnel.values()) == result['jobs'][job]['positives']
        for file in ('case_diagnostics.jsonl','pair_predictions.jsonl','protocol.json','prediction_complete.json'):
            result['sources'][str(directory/file)] = sha(directory/file)
    assert [result['jobs'][j]['pairs'] for j in JOBS] == [3000,800,602]
    for name, data in [('summary.json',result),('layer_weights.json',dict(checkpoint_sha256=EXPECTED_SHA,readout=w,layers=layers)),
                       ('candidate_decomposition.json',candidates_out),('pair_diagnosis.json',pairs_out)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    assert all(sha(path) == value for path,value in result['sources'].items())
    print(json.dumps(dict(weights=w,minimum_mass=result['minimum_mass_at_zero_overlap'],
        jobs={k:{z:v[z] for z in ('pairs','funnel','rejection','counterfactual_readouts','reconstruction_max_abs')} for k,v in result['jobs'].items()}),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--evaluation',default='artifacts/threshold_m12_evaluation_20260927')
    p.add_argument('--checkpoint',default='artifacts/threshold_diagnosis_20260927/threshold_m12_best_joint.pt')
    p.add_argument('--out',default='artifacts/threshold_diagnosis_20260927')
    args=p.parse_args();run(args.evaluation,args.checkpoint,args.out)
