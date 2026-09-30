"""Package baseline and transitions from saved measurements, no new inference."""
import argparse
from pathlib import Path

from measure import read, save, quant, CV


def run(root, phase1):
    root=Path(root);phase1=Path(phase1)
    protocol=read(phase1/'protocol.json');summary=read(root/'summary.json')
    compact=read(root/'pair_summary.json')
    baseline={};fixed=[];transitions={};old_by_id={}
    for split,n in protocol['expected'].items():
        rows=[read(phase1/split/(str(i).zfill(5)+'.json')) for i in range(n)]
        old_by_id.update({r['pair_id']:r for r in rows})
        valid=[r for r in rows if r['label'] and r['gt_known'] and not r['gt_excluded']]
        complete=0;strict_mixed=0
        for r in valid:
            correct={h['index'] for h in r['hypotheses'] if h['edge_ids'] and h['gt_error_px']<=20}
            wrong={h['index'] for h in r['hypotheses'] if h['edge_ids'] and h['gt_error_px']>20}
            touched=[c for c in r['clusters'] if c['retained'] and set(c['hypothesis_ids'])&correct]
            complete+=bool(correct and len(touched)==1 and correct<=set(touched[0]['hypothesis_ids']) and touched[0]['gt_error_px']<=20 and not touched[0]['contains_gt40_wrong_hypothesis'])
            strict_mixed+=sum(bool(set(c['hypothesis_ids'])&correct and set(c['hypothesis_ids'])&wrong) for c in r['clusters'])
        negative_max=[max((c['support_mass_px'] for c in r['clusters'] if c['retained']),default=0.)
            for r in rows if not r['label']]
        baseline[split]=dict(pairs=n,gt_valid=len(valid),complete_correct_count=complete,
            complete_correct_fraction=complete/len(valid) if valid else None,strict_mixed20=strict_mixed,
            negative_largest_sparse_mass_px=quant(negative_max))
    meta=read(CV/'real'/'manifest.json')
    identity={p['pair_id']:p for p in meta['pairs']}
    for variant,rows in compact.items():
        transitions[variant]={}
        for metric in ['coverage_prebudget','coverage_retained','top_cluster_correct']:
            gained=[];lost=[]
            for r in rows:
                if r['split']!='dunhuang_cv' or not r['label'] or r['gt_excluded']:continue
                old=old_by_id[r['pair_id']]['gt_diagnostic'][metric]
                if r[metric]!=old:
                    (gained if r[metric] else lost).append(identity[r['pair_id']])
            transitions[variant][metric]=dict(gained=gained,lost=lost)
    for case in protocol['cases']['cases']:
        old=old_by_id[case['pair_id']]
        new={name:read(root/name/old['split']/(str(old['index']).zfill(5)+'.json')) for name in summary['groups']}
        fixed.append(dict(alias=case['alias'],pair_id=case['pair_id'],baseline=old,variants=new))
    changed={x['pair_id'] for group in transitions.values() for metric in group.values() for side in metric.values() for x in side}
    changed_records=[]
    for pid in sorted(changed):
        old=old_by_id[pid]
        changed_records.append(dict(identity=identity[pid],baseline=old,
            variants={name:read(root/name/'dunhuang_cv'/(str(old['index']).zfill(5)+'.json')) for name in ('s10','s16')}))
    save(root/'comparison.json',dict(baseline=baseline,transitions=transitions))
    save(root/'fixed_cases.json',fixed)
    save(root/'changed_cases.json',changed_records)
    print(baseline)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--phase1',default='/root/autodl-tmp/consensus_strictness_20260925/results_phase1')
    a=p.parse_args();run(a.root,a.phase1)
