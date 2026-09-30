"""GT-correspondence geometry diagnostic on ALL S7 TRAIN12K positive pairs.

Targets deliberately supply correspondence evidence to the unchanged production
decoder. This is NOT deployment, a performance upper bound, or a learned-model
evaluation: material lost between inherited corresponding arcs can bias the
zero-gap translation even when every correspondence label is correct.
No model forward, optimizer, GPU computation, threshold fitting or queue edits.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from staging.pairwise_v0_2.models import translation_layout as decoder
from .diagnose import SOURCE_SHA, TRAIN_SHA, sha

SCHEMA = "s7-training-gt-correspondence-geometry/1"
CONFIG = decoder.TranslationLayoutConfig(correspondence_mode='topk_union',top_k=2,
    max_candidates=512,min_inliers=3,inlier_radius_px=10.)


def save(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def oracle_evidence(points_a,points_b,valid_a,valid_b,target_a,target_b,gt_translation):
    pa,pb=np.asarray(points_a),np.asarray(points_b)
    va,vb=np.asarray(valid_a),np.asarray(valid_b)
    ta,tb=np.asarray(target_a),np.asarray(target_b)
    gt=np.asarray(gt_translation,dtype=float)
    if (pa.shape!=(len(va),2) or pb.shape!=(len(vb),2) or va.dtype!=np.bool_ or vb.dtype!=np.bool_
            or ta.shape!=va.shape or tb.shape!=vb.shape or gt.shape!=(2,) or not np.isfinite(gt).all()):
        raise ValueError('unaligned final contour/target/GT arrays')
    ii=np.flatnonzero(va & (ta>=0))
    jj=ta[ii]
    if (not np.issubdtype(ta.dtype,np.integer) or not np.issubdtype(tb.dtype,np.integer)
            or (jj>=len(vb)).any() or not vb[jj].all() or not np.array_equal(tb[jj],ii)
            or not np.array_equal(np.sort(jj),np.flatnonzero(vb & (tb>=0)))):
        raise ValueError('final inherited correspondence targets must be reciprocal on valid endpoints')
    # Keep original array indices. Removing zero-confidence endpoints from the
    # valid masks only avoids sorting the padded zero matrix; positive edges and
    # production tie-breaking identities are unchanged.
    oracle_a=np.zeros(len(va),bool);oracle_a[ii]=True
    oracle_b=np.zeros(len(vb),bool);oracle_b[jj]=True
    confidence=np.zeros((len(pa),len(pb)),np.float32)
    confidence[ii,jj]=1.
    result=decoder.estimate_translation_layout(pa,pb,confidence,oracle_a,oracle_b,config=CONFIG)
    errors=np.linalg.norm(pb[jj]-pa[ii]-gt,axis=1)
    if not np.isfinite(errors).all():
        raise ValueError('nonfinite GT-edge geometric displacement')
    error=float(np.linalg.norm(result.t_a_to_b_rc-gt)) if result.valid else None
    return dict(gt_edge_count=len(ii),
        gt_edge_error_median_px=float(np.median(errors)) if len(errors) else None,
        gt_edge_error_p90_px=float(np.quantile(errors,.9)) if len(errors) else None,
        oracle_layout_valid=bool(result.valid),oracle_layout20_success=bool(error is not None and error<=20.),
        oracle_error_px=error,oracle_translation_a_to_b_rc=result.t_a_to_b_rc.tolist() if result.valid else None,
        oracle_candidate_count=int(result.candidate_count),oracle_inlier_count=int(result.inlier_count),
        oracle_residual_px=result.residual_px,oracle_reason=result.reason)


def summarize(rows):
    count=len(rows)
    joint={name:0 for name in ('both_success','model_fail_oracle_success','model_success_oracle_fail','both_fail')}
    for row in rows:
        m,o=bool(row['model_layout20_success']),bool(row['oracle_layout20_success'])
        name=('both_success' if m and o else 'model_fail_oracle_success' if o
              else 'model_success_oracle_fail' if m else 'both_fail')
        joint[name]+=1
    metrics={}
    for name in ('gt_edge_count','gt_edge_error_median_px','gt_edge_error_p90_px',
                 'oracle_error_px','model_raw_error_px','oracle_residual_px'):
        values=np.array([r[name] for r in rows if r[name] is not None],float)
        metrics[name]=dict(available_count=len(values),mean=float(values.mean()) if len(values) else None,
            median=float(np.median(values)) if len(values) else None,
            p90=float(np.quantile(values,.9)) if len(values) else None)
    model_success=sum(r['model_layout20_success'] for r in rows)
    oracle_success=sum(r['oracle_layout20_success'] for r in rows)
    return dict(count=count,positive_denominator=count,model_layout20_success_count=model_success,
        model_layout20_rate=model_success/count if count else None,
        oracle_layout20_success_count=oracle_success,oracle_layout20_rate=oracle_success/count if count else None,
        oracle_invalid_count=sum(not r['oracle_layout_valid'] for r in rows),
        fewer_than_3_target_edges_count=sum(r['gt_edge_count']<3 for r in rows),
        joint=joint,metrics=metrics,
        aggregation='pair-weighted; edge_error summaries aggregate per-pair statistics, not pooled edges')


def run(args):
    manifest=Path(args.manifest).resolve(strict=True)
    model_path=Path(args.model_rows).resolve(strict=True)
    model_summary_path=model_path.parent/'summary.json'
    model_summary=json.loads(model_summary_path.read_text())
    if (sha(manifest)!=TRAIN_SHA or model_summary.get('status')!='complete'
            or model_summary.get('overall',{}).get('count')!=24000
            or model_summary.get('overall',{}).get('positive_count')!=12000
            or model_summary.get('sources',{}).get('manifest_sha256')!=TRAIN_SHA
            or model_summary.get('sources',{}).get('source_checkpoint_sha256')!=SOURCE_SHA
            or sha(model_path)!=model_summary.get('rows_sha256')):
        raise ValueError('requires full registered S7 TRAIN24K manifest and completed Matcher diagnosis')
    payload=json.loads(manifest.read_text())
    entries=payload['entries']
    model_rows=[json.loads(line) for line in model_path.read_text().splitlines() if line]
    by_id={row['pair_id']:row for row in model_rows}
    if (len(entries)!=24000 or len(model_rows)!=24000 or len(by_id)!=24000
            or len({e['pair_id'] for e in entries})!=24000
            or set(by_id)!={e['pair_id'] for e in entries}
            or any(e['label']!=by_id[e['pair_id']]['label'] for e in entries)):
        raise ValueError('full source/model pair membership or labels differ')
    positives=[e for e in entries if e['label']==1]
    if len(positives)!=12000:
        raise ValueError('all12000 positives required, never a failure-only selection')
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    protocol=dict(schema=SCHEMA,status='running',positive_count=12000,completed_count=0,
        decoder_config=asdict(CONFIG),decoder_source_sha256=sha(decoder.__file__),
        implementation_sha256=sha(__file__),manifest=str(manifest),manifest_sha256=TRAIN_SHA,
        model_rows=str(model_path),model_rows_sha256=sha(model_path),
        model_summary_sha256=sha(model_summary_path),source_checkpoint_sha256=SOURCE_SHA,
        targets_used_to_construct_correspondence=True,model_forward=False,GPU_computation=False,
        threshold_fit=False,queue_modified=False,performance_upper_bound_claimed=False)
    save(out/'protocol.json',protocol)
    rows=[]
    try:
        with (out/'rows.jsonl').open('x') as stream:
            for entry in positives:
                model=by_id[entry['pair_id']]
                with np.load(Path(payload['artifact_root'])/entry['artifact_path'],allow_pickle=False) as archive:
                    if (str(archive['pair_id'].item())!=entry['pair_id'] or float(archive['label'])!=1.
                            or not bool(archive['translation_valid'])):
                        raise ValueError('final positive artifact identity/GT differs')
                    report=json.loads(str(archive['report_json'].item()))
                    changed=bool(report['changed_pair'])
                    if (changed!=bool(model['actual_changed_pair']) or changed!=bool(entry['changed_pair'])
                            or entry['s7_recipe']!=model['s7_recipe']):
                        raise ValueError('recipe/applied-state differs from actual final artifact')
                    metrics=oracle_evidence(archive['points_rc_a'],archive['points_rc_b'],
                        archive['contour_valid_a'],archive['contour_valid_b'],
                        archive['target_a'],archive['target_b'],archive['translation_a_to_b_rc'])
                    if metrics['gt_edge_count']!=model['final_target_matches']:
                        raise ValueError('final target population differs from Matcher diagnosis')
                row=dict(pair_id=entry['pair_id'],s7_recipe=entry['s7_recipe'],actual_changed_pair=changed,
                    fallback_reason=report.get('fallback_reason'),model_raw_error_px=model['raw_layout_error_px'],
                    model_layout_valid=bool(model['layout_valid']),model_layout20_success=bool(model['raw_layout20_success']),
                    **metrics)
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%1000==0:
                    protocol['completed_count']=len(rows);save(out/'protocol.json',protocol)
        grouped=defaultdict(list)
        for row in rows:
            grouped[str(row['s7_recipe'])+'|actual_changed_pair='+str(row['actual_changed_pair']).lower()].append(row)
        summary=dict(schema=SCHEMA,status='complete',scope='TRAIN in-sample GT-correspondence geometry diagnostic',
            decoder_config=asdict(CONFIG),overall=summarize(rows),
            by_recipe_and_actual_changed={key:summarize(values) for key,values in sorted(grouped.items())},
            rows_sha256=sha(out/'rows.jsonl'),elapsed_s=time.monotonic()-started,
            limitations=['GT correspondences are used, so this is neither deployment nor a learned-model accuracy result.',
                'Correct inherited source-arc labels may retain nonzero material gaps; zero-gap placement can be biased.',
                'Not a performance upper bound: equal GT edge weights and one displacement mode need not dominate learned evidence.',
                'Every positive remains in the Layout20 denominator, including fewer than3 targets and invalid decodes.',
                'Recipe/applied-state comparisons are descriptive, not randomized augmentation causal estimates.'])
        save(out/'summary.json',summary)
        protocol.update(status='complete',completed_count=len(rows),elapsed_s=time.monotonic()-started)
        return summary
    except BaseException as error:
        protocol.update(status='failed',completed_count=len(rows),error=repr(error),elapsed_s=time.monotonic()-started)
        raise
    finally:
        save(out/'protocol.json',protocol)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','model-rows','output'):
        parser.add_argument('--'+name,required=True)
    result=run(parser.parse_args())
    print(json.dumps(dict(status=result['status'],count=result['overall']['count'],elapsed_s=result['elapsed_s'])))
