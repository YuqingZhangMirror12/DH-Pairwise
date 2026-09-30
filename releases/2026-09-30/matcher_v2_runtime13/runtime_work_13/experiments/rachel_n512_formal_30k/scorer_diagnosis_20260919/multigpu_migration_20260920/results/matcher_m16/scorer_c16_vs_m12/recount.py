"""Independent raw-row recount; no endpoint comparator metric functions."""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent.parent
ROOTS = {'M12': RESULTS/'s7_direct/all_tokens',
         'M16': OUT.parent/'scorer_c16'}
OPS = ('max_f1', 'recall_95')


def read(path):
    return json.loads(path.read_text())


def good(row):
    value = row['layouts']['full_top2_mode']
    error = value.get('translation_l2_px')
    return value.get('valid') is True and isinstance(error, (int,float)) and math.isfinite(error) and error <= 20


def yes(row, threshold):
    return row['decision_valid'] and row['classification']['fused'] >= threshold


def ids(rows):
    return sorted(r['pair_id'] for r in rows)


result = {'splits': {}, 'training': {}, 'threshold_mode': 'each own fixed C16 clean-SIMVAL thresholds'}
for split in ('test', 'real', 'ood'):
    endpoints = {name: root/'evaluation/c16'/split for name,root in ROOTS.items()}
    rows, thresholds = {}, {}
    for name, path in endpoints.items():
        protocol, summary = read(path/'protocol.json'), read(path/'summary.json')
        assert protocol['status'] == summary['status'] == 'complete'
        values = [json.loads(x) for x in (path/'pair_results.jsonl').read_text().splitlines()]
        rows[name] = {r['pair_id']:r for r in values}
        assert len(rows[name]) == len(values) == protocol['sample_count']
        thresholds[name] = summary['model']['operating_points']['thresholds']
    a,b = rows['M12'],rows['M16']
    assert a.keys() == b.keys()
    fields = ('label','fragment_a','fragment_b','target_translation_rc')
    if split == 'real':
        fields += ('review_status','strict_member')
    assert all(all(a[i][key] == b[i][key] for key in fields) for i in a)
    positive = [r for r in a.values() if r['label']]
    groups = {'all':list(a.values())}
    if split == 'real':
        groups.update(kept_positive=[r for r in positive if r['review_status']=='keep'],
            strict39=[r for r in a.values() if not r['label'] and r['strict_member']],
            distractor469=[r for r in a.values() if not r['label'] and not r['strict_member']])
        assert len(groups['strict39']) == 39 and len(groups['distractor469']) == 469
    entry = dict(count=len(a), identity_equal=True, groups={})
    for group, selected in groups.items():
        keys=ids(selected); positives=[i for i in keys if a[i]['label']]
        detail=dict(count=len(keys), positive_count=len(positives), operating_points={})
        if split != 'ood':
            detail['layout20'] = {x+'_to_'+y:[i for i in positives
                if good(a[i]) == (x=='good') and good(b[i]) == (y=='good')]
                for x in ('good','bad') for y in ('good','bad')}
        for op in OPS:
            ta,tb=thresholds['M12'][op],thresholds['M16'][op]
            q=dict(thresholds={'M12':ta,'M16':tb},
                   accepted={'M12':sum(yes(a[i],ta) for i in keys),
                             'M16':sum(yes(b[i],tb) for i in keys)})
            q['rescued_positive']=[i for i in positives if not yes(a[i],ta) and yes(b[i],tb)]
            q['lost_positive']=[i for i in positives if yes(a[i],ta) and not yes(b[i],tb)]
            if split != 'ood':
                q['correct_layout_accepted']={name:sum(good(rows[name][i]) and yes(rows[name][i],thresholds[name][op])
                    for i in positives) for name in ROOTS}
                q['correct_layout_rejected']={name:sum(good(rows[name][i]) and not yes(rows[name][i],thresholds[name][op])
                    for i in positives) for name in ROOTS}
                q['joint_success_gained']=[i for i in positives if not (good(a[i]) and yes(a[i],ta)) and good(b[i]) and yes(b[i],tb)]
                q['joint_success_lost']=[i for i in positives if good(a[i]) and yes(a[i],ta) and not (good(b[i]) and yes(b[i],tb))]
                q['both_layout_good_classification_rescued']=[i for i in q['rescued_positive'] if good(a[i]) and good(b[i])]
                q['both_layout_good_classification_lost']=[i for i in q['lost_positive'] if good(a[i]) and good(b[i])]
            detail['operating_points'][op]=q
        entry['groups'][group]=detail
    entry['different_final_valid_or_translation']=[i for i in a if any(
        a[i]['layouts']['full_top2_mode'].get(k) != b[i]['layouts']['full_top2_mode'].get(k)
        for k in ('valid','translation_rc'))]
    result['splits'][split]=entry

controls = ('arm','model','initial_state_sha256','physical_microbatch','effective_batch','accumulation_steps',
    'head_seed','data_seed','classifier_epochs','optimizer','lr_by_head_epoch','weight_decay','grad_clip_norm',
    'precision','loss','training_count','validation_count','selection_population')
identities={}
for name,root in ROOTS.items():
    training=root/'training';protocol=read(training/'protocol.json')
    identity=protocol['identity'];identities[name]=identity
    curve=[]
    for epoch in range(1,17):
        val=read(training/('validation_head_%03d.json'%epoch))
        values=val['operating_points']['validation']
        curve.append(dict(head_epoch=epoch,mean_loss=val['mean_loss'],
            max_f1=values['max_f1']['f1'],AP=values['max_f1']['auprc'],
            precision_at_recall95=values['recall_95']['precision']))
    result['training'][name]=dict(status=read(training/'status.json'),validation_curve=curve,
        matcher_epoch=identity['source_matcher_epochs'],initial_state_sha256=identity['initial_state_sha256'])
assert all(identities['M12'][k] == identities['M16'][k] for k in controls)
assert all(identities['M12']['cache_bindings'][s]['population'] ==
           identities['M16']['cache_bindings'][s]['population'] for s in ('train','val'))
result['equal_budget_controls'] = list(controls)+['train/val exact population metadata']
result['interpretation_caveat'] = 'Fixed Matcher changes and fresh equal-budget head retraining; not a causal attribution to a specific internal feature or threshold.'
(OUT/'independent_recount.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
for split,x in result['splits'].items():
    print(split,'different final valid/translation',len(x['different_final_valid_or_translation']))
    for group,g in x['groups'].items():
        print(group,'layout', {k:len(v) for k,v in g.get('layout20',{}).items()})
        for op,q in g['operating_points'].items():
            print(op, {k:({s:len(v) if isinstance(v,list) else v for s,v in x.items()} if isinstance(x,dict)
                         else len(x) if isinstance(x,list) else x) for k,x in q.items()})
