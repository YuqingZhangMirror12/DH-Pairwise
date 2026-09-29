"""Complete native Matcher population inference with immutable evidence/GT order.

This reusable runner has no scheduler, selection or retry. The command entry
is responsible for source, terminal and actual population admission first.
"""
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .checkpoint_io import file_sha, tree_sha, write_json
from .exposure import digest
from .matcher_diagnostics import summarize
from .matcher_evaluation import annotate_population, predict_batch, prediction_sha
from .matcher_population import tensor_inputs
from .model_adapter import require
from .runtime_io import read


def audit_edges(row):
    """Recompute captured raw union evidence; not an attention/importance map."""
    require('contour_points' in row, 'fixed example contour coordinates missing')
    points=row['contour_points'];a=np.asarray(points['a'],dtype=float).reshape(-1,2)
    b=np.asarray(points['b'],dtype=float).reshape(-1,2)
    oa=np.asarray(points['original_a']);ob=np.asarray(points['original_b'])
    require(len(a)==len(oa)==row['valid_points_a'] and len(b)==len(ob)==row['valid_points_b']
            and np.isfinite(a).all() and np.isfinite(b).all(), 'fixed contour record differs')
    total=0
    for candidate in row['candidates']:
        edge=candidate['edges']; raw=np.asarray(edge['compact_indices'])
        require(raw.ndim==2 and raw.shape[1]==2 and raw.dtype.kind in 'iu', 'integer compact edge IDs required')
        ids=raw.astype(np.int64); n=len(ids)
        require(n==candidate['unique_edge_count'] and len({tuple(x) for x in ids})==n
                and n>0 and ids.min()>=0 and ids[:,0].max()<len(a) and ids[:,1].max()<len(b),
                'edge count, duplication or range differs')
        expected_original=np.stack([oa[ids[:,0]],ob[ids[:,1]]],axis=1)
        require(np.array_equal(np.asarray(edge['original_indices']),expected_original), 'original edge mapping differs')
        q=np.asarray(edge['raw_q'],float); arc=np.asarray(edge['arc_px'],float); mass=np.asarray(edge['q_arc'],float)
        residual=np.asarray(edge['residual_px'],float)
        require(all(v.shape==(n,) and np.isfinite(v).all() for v in (q,arc,mass,residual))
                and (q>=0).all() and (arc>0).all(), 'nonfinite/negative raw evidence')
        require(np.allclose(mass,q*arc,rtol=2e-6,atol=1e-7)
                and np.isclose(q.sum(),candidate['q_sum'],rtol=2e-6,atol=1e-7)
                and np.isclose(mass.sum(),candidate['q_arc_mass_px'],rtol=2e-6,atol=1e-7), 'absolute Q/arc mass differs')
        error=np.linalg.norm(b[ids[:,1]]-a[ids[:,0]]-np.asarray(candidate['translation_rc']),axis=1)
        require(np.allclose(residual,error,rtol=2e-5,atol=2e-4), 'saved correspondence residual differs')
        require(candidate['unique_endpoints_a']==len(np.unique(ids[:,0]))
                and candidate['unique_endpoints_b']==len(np.unique(ids[:,1])), 'endpoint counts differ')
        total+=n
    return dict(status='passed',pair_id=row['pair_id'],candidates=len(row['candidates']),
        edges_across_candidates=total,duplicate_edge_within_cluster=False,
        native_prediction_sha256=prediction_sha(row),scorer_used=False,attention_present=False)


def make_summary(rows, groups, provenance):
    by_id={r['pair_id']:r for r in rows}
    require(len(by_id)==len(rows) and groups, 'unique prediction population and groups required')
    main='all' if provenance['split'].startswith('sim_') else 'real_test'
    require(main in groups, 'main evaluation group missing')
    result={}
    for name,ids in groups.items():
        require(ids and len(ids)==len(set(ids)) and set(ids)<=set(by_id), 'invalid group membership: '+name)
        result[name]=summarize([by_id[i] for i in ids])
    return dict(schema='curriculum-matcher-population-summary/1',status='complete',provenance=provenance,
        groups=result,main_group=main,
        classification_accuracy=None,joint_f1=None,scorer_used=False,threshold_fitting=False,
        real_test_is_historically_unseen=False,
        interpretation='Native Matcher evidence and fixed T16 poses; not Scorer classification. Real roles are source-isolated developmental analysis.')


def write_jsonl(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def read_jsonl(path):
    with Path(path).open() as stream:return [json.loads(line) for line in stream if line.strip()]


def run_population(matcher, geometry, source_root, meta, batches, source, *, out, provenance,
                   wanted_ids, groups, targets_callback, device):
    """targets_callback is invoked only AFTER prediction_complete is durable."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.time()
    before=tree_sha(matcher.state_dict());ids=[p['pair_id'] for p in meta['pairs']]
    try:
        require(ids and len(set(ids))==len(ids) and set(wanted_ids)<=set(ids), 'unique complete population/fixed cases required')
        require(provenance['scorer_used'] is False and provenance['real_used_for_selection'] is False,
                'native preselected Matcher origin required')
        population=dict(source=source,pair_ids=ids,groups=groups,fixed_diagnostic_ids=sorted(wanted_ids))
        origin=dict(provenance,evaluation_population_sha256=digest(population),total_pairs=len(ids),
                    model_input_fields=['mask_a','mask_b','points_rc_a','points_rc_b','contour_valid_a','contour_valid_b'],
                    microbatch=8,gt_used_for_prediction=False,model_state_before=before)
        write_json(out/'population.json',population);write_json(out/'protocol.json',dict(status='inference',**origin))
        rows=[];audits=[]
        with (out/'pair_predictions.jsonl').open('x') as stream,torch.no_grad():
            for items,batch in batches:
                batch_ids=[r['pair_id'] for r in items]
                require(0<len(batch_ids)<=8 and batch_ids==ids[len(rows):len(rows)+len(items)], 'batch order/size differs')
                inputs=tensor_inputs(batch,device)
                predicted=predict_batch(matcher,geometry,source_root,inputs,batch_ids,wanted_ids)
                for row in predicted:
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');rows.append(row)
                    if row['pair_id'] in wanted_ids:audits.append(audit_edges(row))
                stream.flush()
                write_json(out/'status.json',dict(status='inference',processed=len(rows),total=len(ids),
                    elapsed_seconds=time.time()-started,pid=os.getpid()),replace=True)
            stream.flush();os.fsync(stream.fileno())
        require([r['pair_id'] for r in rows]==ids and {a['pair_id'] for a in audits}==set(wanted_ids),
                'missing predictions or fixed diagnostics')
        require(tree_sha(matcher.state_dict())==before, 'frozen inference changed model state')
        frozen=dict(schema='curriculum-native-prediction-complete/1',status='all_predictions_frozen',
            pairs=len(rows),prediction_sha256=file_sha(out/'pair_predictions.jsonl'),provenance=origin,
            model_state_unchanged=True,targets_joined=False)
        write_json(out/'prediction_complete.json',frozen)
        targets=targets_callback()
        labelled=annotate_population(rows,targets)
        write_jsonl(out/'case_diagnostics.jsonl',labelled)
        summary=make_summary(labelled,groups,origin)
        write_json(out/'summary.json',summary);write_json(out/'diagnostic_audit.json',dict(status='passed',cases=audits))
        write_json(out/'protocol.json',dict(status='complete',**origin),replace=True)
        write_json(out/'status.json',dict(status='complete',pairs=len(rows),elapsed_seconds=time.time()-started,pid=os.getpid()),replace=True)
        files=('population.json','protocol.json','pair_predictions.jsonl','prediction_complete.json',
               'case_diagnostics.jsonl','summary.json','diagnostic_audit.json','status.json')
        complete=dict(schema='curriculum-native-evaluation-complete/1',status='complete',provenance=origin,
            pairs=len(rows),files={name:file_sha(out/name) for name in files},
            model_state_unchanged=True,gt_joined_after_predictions=True,scorer_used=False,
            process_success_not_yet_certified=True)
        write_json(out/'evaluation_complete.json',complete)
        return complete
    except BaseException as error:
        write_json(out/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),
            processed=len(locals().get('rows',[])),elapsed_seconds=time.time()-started,automatic_retry=False))
        raise


def verify_population(out):
    """Reopen artifacts, recount/recompute metrics and evidence after success."""
    out=Path(out);require(not (out/'failure.json').exists(), 'failure precedes stale evaluation completion')
    complete=read(out/'evaluation_complete.json')
    require(complete['schema']=='curriculum-native-evaluation-complete/1' and complete['status']=='complete'
            and complete['model_state_unchanged'] is True and complete['gt_joined_after_predictions'] is True
            and complete['scorer_used'] is False, 'not a completed native population')
    expected={'population.json','protocol.json','pair_predictions.jsonl','prediction_complete.json',
              'case_diagnostics.jsonl','summary.json','diagnostic_audit.json','status.json'}
    require(set(complete['files'])==expected and all(file_sha(out/name)==sha for name,sha in complete['files'].items()),
            'completed artifact changed or missing')
    population=read(out/'population.json');frozen=read(out/'prediction_complete.json');origin=complete['provenance']
    require(origin['evaluation_population_sha256']==digest(population) and frozen['provenance']==origin
            and frozen['prediction_sha256']==file_sha(out/'pair_predictions.jsonl')
            and frozen['model_state_unchanged'] is True and frozen['targets_joined'] is False,
            'prediction completion/population binding differs')
    require(read(out/'protocol.json')==dict(status='complete',**origin), 'completed protocol differs')
    raw=read_jsonl(out/'pair_predictions.jsonl');labelled=read_jsonl(out/'case_diagnostics.jsonl')
    require(len(raw)==len(labelled)==complete['pairs']==frozen['pairs']==origin['total_pairs']
            and [r['pair_id'] for r in raw]==[r['pair_id'] for r in labelled]==population['pair_ids'], 'population rows differ')
    targets=[dict(pair_id=r['pair_id'],label=r['label'],gt_pose=r['target_translation_rc']) for r in labelled]
    require(annotate_population(raw,targets)==labelled, 'labelled metrics changed evidence/targets')
    require(make_summary(labelled,population['groups'],origin)==read(out/'summary.json'), 'summary differs from actual rows')
    wanted=set(population['fixed_diagnostic_ids']);audits=[audit_edges(r) for r in raw if r['pair_id'] in wanted]
    require({r['pair_id'] for r in audits}==wanted and read(out/'diagnostic_audit.json')==dict(status='passed',cases=audits),
            'fixed evidence audit differs')
    require(read(out/'status.json')['status']=='complete', 'missing terminal status')
    return dict(status='passed',pairs=len(raw),diagnostic_cases=len(audits),
        evaluation_complete_sha256=file_sha(out/'evaluation_complete.json'),
        prediction_sha256=frozen['prediction_sha256'],model_state_unchanged=True,scorer_used=False,
        process_return_must_be_checked_separately=True)
