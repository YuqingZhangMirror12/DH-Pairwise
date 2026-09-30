"""Retain existing SIMVAL R99 and independently count negative/layout transitions."""
import json
from pathlib import Path
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.endpoint_compare_v1 import compare as metrics

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'s7_direct'
ARMS=('all_tokens','matched_tokens','edge_seed')
OPS=('max_f1','recall_95','recall_99')


def yes(row,threshold):
    return row['decision_valid'] and row['classification']['fused']>=threshold


def good(row):
    value=row['layouts']['full_top2_mode'];error=value.get('translation_l2_px')
    return value['valid'] and isinstance(error,(int,float)) and 0<=error<=20


result=dict(status='complete',threshold_source='unchanged fixed-C16 clean SIMVAL; R99 already frozen, no refit',
    primary_ops=['max_f1','recall_95'],additional_diagnostic_op='recall_99',splits={},
    caveats=['SIMVAL Recall95/99 targets do not guarantee real-domain Recall95/99.',
        'edge_seed uses one best-support pre-refinement candidate group, not a single matched edge.',
        'Versus matched_tokens, edge_seed changes correspondence encoding, proposal stage and adds18720 parameters.',
        'Only future seed/multi/final matched_edges contrasts share the edge encoder capacity.',
        'No final layout is altered in these scorer-only comparisons.'])
for split in ('test','real','ood'):
    endpoints={arm:metrics.load_endpoint(BASE/arm/'evaluation/c16'/split,split) for arm in ARMS}
    assert all(e['status']=='complete' for e in endpoints.values())
    old=endpoints['all_tokens']['rows']
    for arm,e in endpoints.items():
        assert metrics.identity_difference(old,e['rows'],split)['equal']
        assert all(old[i]['layouts']==e['rows'][i]['layouts'] for i in old)
    entry=dict(all_layout_objects_identical=True,count=len(old),models={},paired={})
    for arm,e in endpoints.items():
        protocol=json.loads((BASE/arm/'evaluation/c16'/split/'protocol.json').read_text())
        val=json.loads((BASE/arm/'training/validation_head_016.json').read_text())
        thresholds=e['summary']['model']['operating_points']['thresholds']
        assert all(thresholds[op]==protocol['model']['operating_points']['thresholds'][op]
                   ==val['operating_points']['thresholds'][op] for op in OPS)
        entry['models'][arm]={}
        for group,rows in metrics.populations(e['rows'].values(),split).items():
            entry['models'][arm][group]={op:metrics.metrics(rows,thresholds[op],split) for op in OPS}
        if split=='real':
            for name,predicate in [('strict39',lambda r:r['strict_member']),('distractor469',lambda r:not r['strict_member'])]:
                rows=[r for r in e['rows'].values() if not r['label'] and predicate(r)]
                assert len(rows)==(39 if name=='strict39' else 469)
                entry['models'][arm][name]={op:dict(count=len(rows),false_positive_count=sum(yes(r,thresholds[op]) for r in rows),
                    false_positive_ids=sorted(r['pair_id'] for r in rows if yes(r,thresholds[op]))) for op in OPS}
    for reference in ('all_tokens','matched_tokens'):
        a,b=endpoints[reference],endpoints['edge_seed'];entry['paired'][reference]={}
        for group,rows in metrics.populations(a['rows'].values(),split).items():
            ids=[r['pair_id'] for r in rows if r['label']]
            entry['paired'][reference][group]={}
            for op in OPS:
                ta=a['summary']['model']['operating_points']['thresholds'][op]
                tb=b['summary']['model']['operating_points']['thresholds'][op]
                gained=[i for i in ids if not yes(a['rows'][i],ta) and yes(b['rows'][i],tb)]
                lost=[i for i in ids if yes(a['rows'][i],ta) and not yes(b['rows'][i],tb)]
                record=dict(rescued_positive_ids=sorted(gained),lost_positive_ids=sorted(lost))
                if split!='ood':
                    record.update(correct_layout_gained_ids=sorted(i for i in gained if good(a['rows'][i])),
                                  correct_layout_lost_ids=sorted(i for i in lost if good(a['rows'][i])))
                entry['paired'][reference][group][op]=record
    result['splits'][split]=entry
(OUT/'frozen_ops_recount.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for arm in ARMS:
    real=result['splits']['real']['models'][arm]
    for op in OPS:
        m=real['keep803'][op];ood=result['splits']['ood']['models'][arm]['ood301'][op]
        print(arm,op,'th',m['threshold'],'TP/FP',m['tp'],m['fp'],'accepted_good',m['layout20']['accepted_correct'],
            'strict/distractor',real['strict39'][op]['false_positive_count'],real['distractor469'][op]['false_positive_count'],
            'OOD',ood['accepted_positive_count'])
for op,r in result['splits']['real']['paired']['matched_tokens']['keep803'].items():
    print('matched_tokens->edge_seed',op,{k:len(v) for k,v in r.items()})
