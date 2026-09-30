"""Offline candidate-pose diagnostics; never change deployed layouts or fit thresholds.

Run with --endpoint edge_seed:test=/complete/test (and real/ood), --output NEWDIR.
Only stdlib and the existing complete-endpoint/cohort reader are imported.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from ..endpoint_compare_v1 import compare as endpoints

SCHEMA = 'candidate-pose-alignment/1'
OPS = ('max_f1', 'recall_99')
POSE_EQUAL_ATOL_PX = 1e-4  # Saved candidates are FP32; production decoder is FP64.


def vector(value):
    return isinstance(value, list) and len(value)==2 and all(endpoints.finite(x) for x in value)


def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def cohort(row, split):
    if split=='ood':return 'ood_positive301'
    if split=='test':return 'test_positive1500' if row['label'] else 'test_negative1500'
    if row['label']:
        return 'real_kept_positive295' if row['review_status']=='keep' else 'real_excluded_positive213'
    return 'real_original_negative39' if row['strict_member'] else 'real_constructed_negative469'


def inspect_row(row, split, thresholds):
    """Scorer selection is computed first without labels/GT; oracle is diagnostic only."""
    d=row['candidate_details']
    names=('group_logits','group_ranks','group_eligible','group_present','group_translation_rc')
    values=[d[k] for k in names]
    if not all(isinstance(v,list) for v in values) or len({len(v) for v in values})!=1:
        raise ValueError('inconsistent group array shapes: '+row['pair_id'])
    groups=[]
    for slot,(logit,rank,eligible,present,pose) in enumerate(zip(*values)):
        if type(eligible) is not bool or type(present) is not bool or eligible and not present:
            raise ValueError('invalid group presence/eligibility')
        if present and (type(rank) is not int or rank<=0 or not vector(pose)):
            raise ValueError('present group needs positive rank and finite pose')
        if eligible and not endpoints.finite(logit):raise ValueError('eligible group needs finite logit')
        groups.append(dict(slot=slot,proposal_rank=rank,present=present,eligible=eligible,
            logit=logit if eligible else None,translation_rc=pose if present else None))
    present_ranks=[g['proposal_rank'] for g in groups if g['present']]
    if len(present_ranks)!=len(set(present_ranks)):raise ValueError('duplicate present proposal rank')
    eligible=[g for g in groups if g['eligible']]
    # Python stable sort matches first-slot torch.argmax on tied maximum logits.
    ordered=sorted(eligible,key=lambda g:-g['logit'])
    for rank,g in enumerate(ordered,1):g['scorer_order_1based']=rank
    highest=ordered[0] if ordered else None
    expected_rank=highest['proposal_rank'] if highest else 0
    if (d['selected_group_rank']!=expected_rank
            or d['has_selected_candidate_group'] is not bool(highest)
            or d['used_fallback'] is not (highest is None)):
        raise ValueError('saved selected group/fallback does not match score argmax')
    saved_pose=d['selected_group_translation_rc']
    if highest and (not vector(saved_pose) or distance(saved_pose,highest['translation_rc'])>POSE_EQUAL_ATOL_PX):
        raise ValueError('saved selected pose disagrees with highest eligible group')
    if not highest and vector(saved_pose):raise ValueError('no eligible group has a finite selected pose')

    production=row['layouts']['full_top2_mode']
    if type(production.get('valid')) is not bool:raise ValueError('production validity missing')
    production_pose=production.get('translation_rc')
    if production['valid'] and not vector(production_pose):raise ValueError('valid production pose missing')
    delta=distance(production_pose,highest['translation_rc']) if highest and production['valid'] else None
    result=dict(pair_id=row['pair_id'],split=split,cohort=cohort(row,split),label=bool(row['label']),
        fragment_a=row.get('fragment_a'),fragment_b=row.get('fragment_b'),
        score=row['classification']['fused'],decision_valid=row['decision_valid'],
        accepted={op:endpoints.accepted(row,thresholds[op]) for op in OPS},
        fallback=d['used_fallback'],present_group_count=len(present_ranks),eligible_group_count=len(eligible),
        selected_group_rank=expected_rank,selected_slot=None if not highest else highest['slot'],
        production_valid=production['valid'],production_translation_rc=production_pose if vector(production_pose) else None,
        highest_scorer_translation_rc=None if not highest else highest['translation_rc'],
        pose_difference_l2_px=delta,same_translation=None if delta is None else delta<=POSE_EQUAL_ATOL_PX,
        groups=groups,gt_diagnostic=None)
    # Negative rows have no correct-join pose; OOD never has annotated pose GT.
    # Even if a malformed negative/OOD contains coordinates, they are not used.
    target=row.get('target_translation_rc')
    if split!='ood' and row['label'] and vector(target):
        pe=distance(production_pose,target) if production['valid'] else None
        recorded=production.get('translation_l2_px')
        if production['valid'] and (not endpoints.finite(recorded)
                or not math.isclose(pe,recorded,rel_tol=1e-7,abs_tol=1e-5)):
            raise ValueError('recomputed production GT error disagrees with saved error')
        for g in eligible:g['gt_error_l2_px']=distance(g['translation_rc'],target)
        best=min(eligible,key=lambda g:g['gt_error_l2_px']) if eligible else None
        he=None if highest is None else highest['gt_error_l2_px']
        pg=production['valid'] and pe<=20
        hg=highest is not None and he<=20
        result['gt_diagnostic']=dict(target_translation_rc=target,production_error_l2_px=pe,
            production_layout20=pg,highest_scorer_error_l2_px=he,highest_scorer_layout20=hg,
            gain_not_deployed=not pg and hg,loss_not_deployed=pg and not hg,
            gt_best_eligible_group_rank=None if best is None else best['proposal_rank'],
            gt_best_eligible_scorer_order_1based=None if best is None else best['scorer_order_1based'],
            gt_best_eligible_error_l2_px=None if best is None else best['gt_error_l2_px'],
            gt_best_eligible_layout20_coverage=best is not None and best['gt_error_l2_px']<=20,
            gt_best_role='diagnostic coverage only; never deployed or threshold-selected')
    return result


def quantiles(values):
    values=sorted(values)
    def q(f):
        z=(len(values)-1)*f;i=int(z);j=min(i+1,len(values)-1)
        return values[i]+(values[j]-values[i])*(z-i)
    return None if not values else dict(n=len(values),mean=statistics.mean(values),
        p10=q(.1),p50=q(.5),p90=q(.9),maximum=values[-1])


def histogram(values):
    return dict(sorted(Counter(str(x) for x in values).items()))


def aggregate(rows, thresholds):
    diffs=[r['pose_difference_l2_px'] for r in rows if r['pose_difference_l2_px'] is not None]
    result=dict(n=len(rows),decision_valid=sum(r['decision_valid'] for r in rows),
        fallback=sum(r['fallback'] for r in rows),no_eligible_group=sum(not r['eligible_group_count'] for r in rows),
        selected_proposal_rank_counts=histogram(r['selected_group_rank'] for r in rows),
        eligible_group_count_histogram=histogram(r['eligible_group_count'] for r in rows),
        pose_comparison=dict(comparable=len(diffs),same_within_atol=sum(x<=POSE_EQUAL_ATOL_PX for x in diffs),
            different=sum(x>POSE_EQUAL_ATOL_PX for x in diffs),difference_gt20=sum(x>20 for x in diffs),
            difference_l2_px=quantiles(diffs)),operating_points={})
    gt=[r for r in rows if r['gt_diagnostic'] is not None]
    for op,t in thresholds.items():
        accepted=[r for r in rows if r['accepted'][op]]
        result['operating_points'][op]=dict(threshold=t,accepted=len(accepted),rejected=len(rows)-len(accepted),
            selected_proposal_rank_counts=histogram(r['selected_group_rank'] for r in accepted),
            pose_difference_gt20=sum(r['pose_difference_l2_px'] is not None and r['pose_difference_l2_px']>20 for r in accepted))
        if gt:
            a=[r['gt_diagnostic'] for r in gt if r['accepted'][op]]
            result['operating_points'][op].update(accepted_gt_positive_count=len(a),
                deployed_production_layout20_accepted=sum(g['production_layout20'] for g in a),
                hypothetical_highest_scorer_layout20_accepted=sum(g['highest_scorer_layout20'] for g in a),
                hypothetical_pose_gain=sum(g['gain_not_deployed'] for g in a),
                hypothetical_pose_loss=sum(g['loss_not_deployed'] for g in a))
    if gt:
        gs=[r['gt_diagnostic'] for r in gt]
        result['positive_gt_pose']=dict(denominator=len(rows),known_gt_count=len(gt),missing_gt_count=len(rows)-len(gt),
            deployed_production_layout20=sum(g['production_layout20'] for g in gs),
            hypothetical_highest_scorer_layout20=sum(g['highest_scorer_layout20'] for g in gs),
            hypothetical_gain=sum(g['gain_not_deployed'] for g in gs),hypothetical_loss=sum(g['loss_not_deployed'] for g in gs),
            gain_pair_ids=[r['pair_id'] for r in gt if r['gt_diagnostic']['gain_not_deployed']],
            loss_pair_ids=[r['pair_id'] for r in gt if r['gt_diagnostic']['loss_not_deployed']],
            diagnostic_gt_best_eligible_layout20_coverage=sum(g['gt_best_eligible_layout20_coverage'] for g in gs),
            diagnostic_gt_best_proposal_rank_counts=histogram(g['gt_best_eligible_group_rank'] for g in gs),
            diagnostic_gt_best_scorer_order_counts=histogram(g['gt_best_eligible_scorer_order_1based'] for g in gs),
            diagnostic_covered_gt_best_scorer_order_counts=histogram(g['gt_best_eligible_scorer_order_1based'] for g in gs if g['gt_best_eligible_layout20_coverage']))
    else:
        result['layout_gt_status']='not_applicable_negative_or_unavailable_GT; no pose accuracy/coverage claim'
    return result


def load_and_summarize(path, split):
    loaded=endpoints.load_endpoint(Path(path),split)
    if loaded['status']!='complete':raise ValueError(loaded.get('reason','incomplete endpoint'))
    summary=loaded['summary'];model=summary['model']
    if model.get('architecture') not in ('fresh_matched_only_ca:edge_seed','fresh_matched_only_ca:edge_multi'):
        raise ValueError('only saved edge_seed/edge_multi endpoints supported')
    protocol=json.loads((Path(path)/'protocol.json').read_text())
    thresholds={k:model['operating_points']['thresholds'][k] for k in OPS}
    for k,t in thresholds.items():
        if not endpoints.finite(t) or not 0<=t<=1 or t!=protocol['model']['operating_points']['thresholds'][k]:
            raise ValueError('unsafe/mismatched own frozen SIMVAL operating point')
    if protocol.get('ground_truth_used_to_select_candidates') is not False:
        raise ValueError('missing target-blind source provenance')
    raw=list(loaded['rows'].values())
    if split=='real':
        for name,values in endpoints.populations(raw,split).items():
            counts=(len(values),sum(r['label'] for r in values),sum(not r['label'] for r in values))
            if counts!=endpoints.REAL_COUNTS[name]:raise ValueError('REAL review/cohort count mismatch')
    rows=[inspect_row(r,split,thresholds) for r in raw]
    groups={name:aggregate([r for r in rows if r['cohort']==name],thresholds)
        for name in sorted({r['cohort'] for r in rows}) if name!='real_excluded_positive213'}
    # Independently recomputed production pose/gating must match the existing
    # evaluator's primary population; never validate a hypothetical pose as deployed.
    if split in ('test','real'):
        key='test_positive1500' if split=='test' else 'real_kept_positive295'
        official=summary['groups']['all' if split=='test' else 'kept_plus_all_negative']['layout']['max_f1']['20']
        g=groups[key]
        if (g['positive_gt_pose']['deployed_production_layout20']!=official['raw_correct']
                or g['operating_points']['max_f1']['deployed_production_layout20_accepted']!=official['accepted_correct']):
            raise ValueError('production Layout20 recount disagrees with endpoint summary')
    return dict(status='complete',split=split,model=loaded['model'],architecture=model['architecture'],
        thresholds=thresholds,sources=loaded['sources'],groups=groups,
        row_count=len(rows),excluded_review_positive_count=sum(r['cohort']=='real_excluded_positive213' for r in rows)),rows


def run(inputs, output):
    if Path(output).exists():raise ValueError('new output directory required; do not overwrite prior results')
    names=sorted({model for model,split in inputs})
    if not names or any((m,s) not in inputs for m in names for s in ('test','real','ood')):
        raise ValueError('every model requires complete TEST/REAL/OOD inputs')
    summaries={};cases=[]
    for model in names:
        identity=None
        for split in ('test','real','ood'):
            report,rows=load_and_summarize(inputs[model,split],split)
            current=(report['model']['checkpoint_sha256'],report['thresholds'])
            if identity is not None and current!=identity:raise ValueError('checkpoint/threshold differs between domains')
            identity=current;summaries[model+':'+split]=report
            cases.extend(dict(model=model,**r) for r in rows)
    result=dict(schema=SCHEMA,status='complete',created_at_utc=datetime.now(timezone.utc).isoformat(),
        no_inference=True,no_threshold_fit=True,no_remote_access=True,no_deployment_change=True,
        cases_count=len(cases),endpoints=summaries,implementation_sha256={
            str(Path(__file__).resolve()):sha(__file__),str(Path(endpoints.__file__).resolve()):sha(endpoints.__file__)},
        definitions=dict(deployed_layout='unchanged layouts.full_top2_mode, valid and L2(target)<=20px',
            highest_scorer_pose='maximum saved logit among present AND eligible groups; first array slot resolves ties; diagnostic, NOT deployed',
            gt_best='minimum GT L2 among eligible groups only; diagnostic coverage and scorer-order rank, never deployed',
            group_rank='saved target-blind proposal rank, not scorer rank',
            coordinates='A-to-B row/column translation; p_b=p_a+translation; no sign flip',
            pose_same_atol_px=POSE_EQUAL_ATOL_PX,
            no_eligible='no proposed pose; counted as not Layout20 with known positive GT, not as correct fallback',
            accept='original decision_valid AND original fused probability >= own frozen SIMVAL threshold',
            real='review_status==keep positive295; strict_member negative39; remaining constructed cross-case negative469',
            ood='301 positive only; acceptance/rank/pose disagreement, no GT layout accuracy'),
        caveats=['Highest-group pose does not replace the production decoder in the saved adapter.',
            'edge_seed records a pre-refinement seed; production uses refined final translation.',
            'GT-best is an unavailable-at-inference diagnostic, not a deployable oracle or performance result.',
            'R99 is pre-existing SIMVAL recall_99, not REAL/OOD fitted; target recall is only guaranteed empirically on SIMVAL.',
            'Differences describe saved outputs, not proof of architectural cause or a promotion decision.'])
    output=Path(output);output.mkdir(parents=True)
    (output/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (output/'cases.jsonl').open('w') as f:
        for case in cases:f.write(json.dumps(case,ensure_ascii=False,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',action='append',required=True,help='MODEL:SPLIT=/complete/endpoint')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();inputs={}
    for value in args.endpoint:
        identity,path=value.split('=',1);model,split=identity.rsplit(':',1)
        if split not in ('test','real','ood') or (model,split) in inputs:raise ValueError('invalid/duplicate endpoint')
        inputs[model,split]=Path(path).resolve()
    result=run(inputs,args.output)
    print(json.dumps(dict(status=result['status'],cases_count=result['cases_count'],output=str(Path(args.output).resolve()))))


if __name__=='__main__':main()
