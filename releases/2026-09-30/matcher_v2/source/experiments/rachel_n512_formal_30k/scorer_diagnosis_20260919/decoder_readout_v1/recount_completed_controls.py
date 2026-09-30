"""Independent counts from raw predictions, with no analyzer/model imports.

Reads frozen local files only. Verifies headline classification/Layout counts,
CAL-only threshold choice, paired IDs and every selected seed for the 128-limit
control. Never changes a model, reruns inference, or writes to existing outputs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def observations(records, search, readout, threshold):
    out = {}
    for r in records:
        p = r['predictions'][search]; v = p['readouts'][readout]
        i = v['winner']; cs = p['candidates']; label = r['label']
        accept = bool(p['numeric_valid'] and i >= 0 and v['score'] >= threshold)
        errors = [math.dist(c['translation'], r['target']) for c in cs] if r['target'] is not None else []
        good = bool(p['numeric_valid'] and i >= 0 and errors and errors[i] <= 20)
        coverage = bool(errors and min(errors) <= 20)
        out[r['pair_id']] = (label, accept, good, coverage)
    return out


def counts(obs, layout):
    p = sum(v[0] for v in obs.values()); n = len(obs)-p
    tp = sum(y and a for y, a, _, _ in obs.values())
    fp = sum(not y and a for y, a, _, _ in obs.values())
    result = dict(pairs=len(obs), positives=p, negatives=n, tp=tp, fp=fp, fn=p-tp, tn=n-fp,
                  accuracy=(tp+n-fp)/len(obs), f1=2*tp/max(1, p+tp+fp), precision=tp/max(1,tp+fp))
    if layout:
        good = sum(v[2] for v in obs.values()); cover = sum(v[3] for v in obs.values())
        joint = sum(a and g for _, a, g, _ in obs.values())
        wrong = sum(y and a and not g for y, a, g, _ in obs.values())
        result.update(layout20_count=good, candidate_coverage_count=cover, joint_tp=joint,
            joint_fp=fp+wrong, joint_fn=p-joint, joint_f1=2*joint/max(1,p+joint+fp+wrong),
            joint_precision=joint/max(1,joint+fp+wrong), winner_correct_but_rejected=good-joint,
            covered_but_winner_wrong=cover-good, positive_no_correct_candidate=p-cover)
    return result


def paired_ids(before, after):
    tests = {
        'classification': lambda v: v[0] == v[1],
        'layout': lambda v: v[2],
        'coverage': lambda v: v[3],
        'correct_and_accepted': lambda v: v[1] and v[2],
    }
    result = {}
    for name, fn in tests.items():
        result[name+'_gained'] = sorted(k for k in before if not fn(before[k]) and fn(after[k]))
        result[name+'_lost'] = sorted(k for k in before if fn(before[k]) and not fn(after[k]))
    result['false_positive_added'] = sorted(k for k in before if not after[k][0] and after[k][1] and not before[k][1])
    result['false_positive_removed'] = sorted(k for k in before if not after[k][0] and not after[k][1] and before[k][1])
    return result


def verify(root, analysis):
    verified = {}
    for model, result in analysis['models'].items():
        path = root/'full_development_01'/model/'records.jsonl'
        records = [json.loads(line) for line in path.open()]
        assert len(records) == 2619 and len({(r['split'],r['pair_id']) for r in records}) == 2619
        groups = defaultdict(list)
        for r in records: groups[(r['split'], r['role'])].append(r)
        assert {k:len(v) for k,v in groups.items()} == {('dunhuang_cv','real_cal'):160,
            ('dunhuang_cv','real_select'):479,('turufan','real_cal'):120,
            ('turufan','real_select'):360,('sim_select','sim_select'):1500}
        equal = dict(selected_modes=0, candidates=0, all_readouts=0, out_of_128=0, max_selected_mass_rank=0)
        for r in records:
            b, a = (r['predictions'][k] for k in ('baseline','all_modes_only'))
            equal['selected_modes'] += b['search_audit']['selected_modes'] == a['search_audit']['selected_modes']
            equal['candidates'] += b['candidates'] == a['candidates']
            equal['all_readouts'] += b['readouts'] == a['readouts']
            ranks = [m['mass_rank'] for m in a['search_audit']['selected_modes']]
            equal['out_of_128'] += any(v > 128 for v in ranks)
            equal['max_selected_mass_rank'] = max(equal['max_selected_mass_rank'], max(ranks,default=0))
        thresholds = {}
        for row in result['rows']:
            if row['threshold_mode'] != 'separate_real_cal' or row['role'] != 'real_cal': continue
            split, search, readout = (row[k] for k in ('split','search','readout'))
            choices = []
            for integer in range(20,81):
                t = integer/100
                m = counts(observations(groups[(split,'real_cal')], search, readout, t), split!='turufan')
                target, prec = ('joint_f1','joint_precision') if split!='turufan' else ('f1','precision')
                choices.append((m[target],-abs(t-.30),m[prec],t))
            t = max(choices)[3]
            assert t == row['metrics']['threshold'], (model,split,search,readout,'CAL mismatch')
            thresholds[(split,search,readout)] = t
        checked_metrics = checked_paired = 0
        by_key = {(r['split'],r['role'],r['search'],r['readout'],r['threshold_mode']):r for r in result['rows']}
        for row in result['rows']:
            split, role, search, readout, mode = (row[k] for k in ('split','role','search','readout','threshold_mode'))
            threshold = result['protocol']['threshold'] if mode=='original_frozen_cal' else thresholds[(split,search,readout)]
            assert threshold == row['metrics']['threshold']
            population = groups[(split,role)]
            after = observations(population,search,readout,threshold)
            m = counts(after,split!='turufan')
            for k,v in m.items():
                assert abs(v-row['metrics'][k]) <= 1e-10, (model,split,role,search,readout,mode,k)
                checked_metrics += 1
            baseline = by_key[(split,role,'baseline','baseline',mode)]
            before = observations(population,'baseline','baseline',baseline['metrics']['threshold'])
            ids = paired_ids(before,after)
            for k,v in ids.items():
                expected = row['paired']['pair_ids'][k]
                if expected is None:
                    assert split=='turufan' and k.startswith(('layout_','coverage_','correct_and_accepted_'))
                else: assert sorted(expected)==v, (model,split,role,search,readout,k)
                checked_paired += 1
        verified[model] = dict(records=len(records),records_sha256=digest(path),
            independently_checked_metric_values=checked_metrics, paired_id_sets=checked_paired,
            cal_thresholds_checked=len(thresholds),all_modes_vs_baseline=equal)
    return dict(schema='decoder-controls-independent-recount/1',status='passed',models=verified,
                test_used=False,inference_rerun=False, training_modified=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); value=verify(a.root,json.loads(a.analysis.read_text()))
    value['analysis_sha256']=digest(a.analysis);value['recount_source_sha256']=digest(Path(__file__))
    with a.out.open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__': main()
