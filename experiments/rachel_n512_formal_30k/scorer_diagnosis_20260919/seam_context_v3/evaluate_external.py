"""Frozen v3 inference on the unchanged real caches or original SIM TEST.

Only six input tensors reach forward. Labels and pose GT are joined after the
complete prediction file is closed. This adapter never changes model/decoder.
"""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from .config import Config
from .model import SeamContextModel
from .prepare import read, save, sha, sources
from .seam_proposals import propose
from .validate import metrics

ROOT = Path('/root/autodl-tmp')
CV = ROOT/'rachel_score_design_20260913_001/real_domain_calibration_v1_20260921'
REAL = ROOT/'rachel_layout_v2_20260906_001/real_preparation/prepared'
GT = ROOT/'rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json'
RELEASE = ROOT/'dataset_rachel_pairwise_n512_v1'
NAMES = ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b', 'contour_valid_a', 'contour_valid_b')


def input_batches(meta, arrays, size):
    index = {x: i for i, x in enumerate(meta['fragment_ids'])}
    for start in range(0, len(meta['pairs']), size):
        pairs = meta['pairs'][start:start+size]
        batch = {}
        for side in 'ab':
            indices = [index[p['fragment_'+side+'_id']] for p in pairs]
            batch['mask_'+side] = np.unpackbits(arrays['packed_masks'][indices], axis=-1).astype(np.float32)[:, None]
            batch['points_rc_'+side] = arrays['points'][indices].astype(np.float32)
            batch['contour_valid_'+side] = arrays['valid'][indices].astype(bool)
        yield pairs, batch


def load_inputs(split, batch_size):
    if split == 'sim_test':
        from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset, collate_rachel_pairs
        manifest = RELEASE/'pairs/test.jsonl'
        pairs = [json.loads(x) for x in manifest.read_text().splitlines() if x]
        dataset = RachelPairDataset(RELEASE, 'test')
        def batches():
            for start in range(0, len(dataset), batch_size):
                batch = collate_rachel_pairs([dataset[i] for i in range(start, min(start+batch_size, len(dataset)))]).as_dict()
                yield pairs[start:start+batch_size], {k: batch[k] for k in NAMES}
        return dict(pairs=pairs), batches(), dict(manifest=str(manifest), manifest_sha256=sha(manifest)), dataset
    manifest = REAL/'manifest.json' if split == 'dunhuang' else CV/('real' if split == 'dunhuang_cv' else 'ood')/'manifest.json'
    meta = read(manifest)
    prepared = Path(meta.get('prepared', REAL))
    with np.load(prepared/'inputs.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in ('packed_masks', 'points', 'valid')}
    n = len(meta['fragment_ids'])
    if arrays['packed_masks'].shape != (n,800,100) or arrays['points'].shape != (n,512,2) or arrays['valid'].shape != (n,512):
        raise ValueError('prepared inputs changed shape')
    expected = {'dunhuang': (1016,508), 'dunhuang_cv': (803,295), 'turufan': (602,301)}[split]
    if (len(meta['pairs']),sum(bool(p['label']) for p in meta['pairs'])) != expected:
        raise ValueError('frozen population changed')
    return meta, input_batches(meta,arrays,batch_size), dict(manifest=str(manifest),manifest_sha256=sha(manifest),
        prepared=str(prepared),inputs_sha256=sha(prepared/'inputs.npz'),preprocessing='unchanged existing 800px cache; original512 sampling'), None


def array(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def describe_prediction(o, b, pair, cfg):
    rec, verified = o.records[b], o.verified[b]
    points_a, points_b = array(o.ga.points[b]), array(o.gb.points[b])
    edge = rec.edges.detach().cpu().numpy()
    q0, q1 = array(o.ot0.real_transport[b]), array(o.ot1.real_transport[b])
    candidates=[]
    for k,c in enumerate(o.candidates[b]):
        ij=edge[c.edge_ids]
        t=array(verified.translations[k])
        residual=np.linalg.norm(points_b[ij[:,1]]-points_a[ij[:,0]]-t,axis=1)
        candidates.append(dict(index=k,score=float(verified.logits[k].sigmoid()),
            quality_logit=float(verified.quality[k]),null_logit=float(verified.null),
            proposal_translation_rc=c.translation.tolist(),translation_rc=t.tolist(),
            structural_score=c.structural_score,edge_count=len(ij),
            unique_endpoints_a=len(np.unique(ij[:,0])),unique_endpoints_b=len(np.unique(ij[:,1])),
            arc_point_counts=list(verified.arc_point_counts[k]),evidence=array(verified.evidence[k]).tolist(),
            endpoint_points_a_rc=points_a[ij[:,0]].tolist(),endpoint_points_b_rc=points_b[ij[:,1]].tolist(),
            q0=q0[ij[:,0],ij[:,1]].tolist(),q1=q1[ij[:,0],ij[:,1]].tolist(),
            residual_px=residual.tolist(),residual_median_px=float(np.median(residual))))
    z=propose(rec,o.ot0.real_transport[b],o.ga,o.gb,b,cfg)
    winner=verified.winner
    t=array(verified.translations[winner]) if verified.has_candidate else None
    score=float(verified.score)
    valid=bool(t is not None and np.isfinite(t).all() and np.isfinite(score))
    return dict(pair_id=pair['pair_id'],score=score,numeric_valid=valid,has_candidate=bool(candidates),
        translation=t.tolist() if valid else None,winner_index=winner,candidates=candidates,
        q0_candidates=[dict(translation_rc=c.translation.tolist(),structural_score=c.structural_score) for c in z],
        candidate_count=len(candidates),valid_points_a=int(o.ga.counts[b]),valid_points_b=int(o.gb.counts[b]),
        q0_mass=float(q0.sum()),q1_mass=float(q1.sum()),
        affinity_delta_abs_max=float(rec.delta.abs().max()) if len(edge) else 0.,
        ot_residual=float(torch.maximum(o.ot1.diagnostics.row_residual_max[b],o.ot1.diagnostics.col_residual_max[b])))


def attach_targets(predictions, meta, split, dataset=None):
    """Post-inference only. Unknown Turufan pose must remain unknown."""
    ground_truth={}
    if split.startswith('dunhuang'):
        ground_truth={r['pair_id']:r for r in read(GT)['positive_pairs']}
    kept={r['pair_id'] for r in read(CV/'real/manifest.json')['pairs'] if r['label']} if split=='dunhuang' else set()
    rows=[]
    for i,(prediction,pair) in enumerate(zip(predictions,meta['pairs'])):
        if prediction['pair_id']!=pair['pair_id']:
            raise ValueError('pair order differs')
        gt=None
        if split=='sim_test':
            sample=dataset[i]
            label=bool(sample.label)
            if sample.translation_valid: gt=sample.translation_a_to_b_rc
        else:
            label=bool(pair['label'])
            if split.startswith('dunhuang') and label:
                target=ground_truth[pair['pair_id']]
                if (target['fragment_a_token'],target['fragment_b_token']) != (pair['fragment_a_id'],pair['fragment_b_id']):
                    raise ValueError('GT endpoint order differs')
                gt=target['translation_gt_a_to_b_rc']
        r=dict(prediction,label=label,gt_known=gt is not None,target_translation_rc=np.asarray(gt).tolist() if gt is not None else None)
        def error(t):
            return float(np.linalg.norm(np.asarray(t)-gt)) if t is not None and gt is not None else None
        r['error_px']=error(r['translation'])
        r['layout20']=bool(r['error_px'] is not None and r['error_px']<=20)
        for c in r['candidates']:
            c['proposal_error_px']=error(c['proposal_translation_rc']);c['error_px']=error(c['translation_rc'])
        for c in r['q0_candidates']: c['error_px']=error(c['translation_rc'])
        r['coverage8']=any(c['proposal_error_px'] is not None and c['proposal_error_px']<=20 for c in r['candidates'])
        r['refined_coverage8']=any(c['error_px'] is not None and c['error_px']<=20 for c in r['candidates'])
        r['q0_coverage8']=any(c['error_px'] is not None and c['error_px']<=20 for c in r['q0_candidates'])
        r['review_keep']=r['pair_id'] in kept if split=='dunhuang' and label else None
        r['strict_member']=pair.get('strict')
        r['case_cluster']=pair.get('case_cluster')
        r['fold']=pair.get('fold')
        rows.append(r)
    return rows


def summarize(rows,threshold):
    result=metrics(rows,threshold)
    labels=np.array([r['label'] for r in rows],bool)
    positive=[r for r in rows if r['label'] and r['gt_known']]
    result['negatives']=len(rows)-int(labels.sum())
    result['pair_tn']=result['negatives']-result['pair_fp']
    if labels.any() and (~labels).any():
        from sklearn.metrics import roc_auc_score
        result['pair_auroc']=float(roc_auc_score(labels,[r['score'] for r in rows]))
    else: result['pair_auroc']=None
    if not positive:
        for k in list(result):
            if k.startswith('joint_') or k in ('layout20','candidate_coverage','q0_candidate_coverage','wrong_pose_accepted'):
                result[k]=None
    else:
        accepted=lambda r:r['numeric_valid'] and r['has_candidate'] and r['score']>=threshold
        result['layout_correct_count']=sum(r['layout20'] for r in positive)
        result['correct_layout_rejected']=sum(r['layout20'] and not accepted(r) for r in positive)
        result['refined_candidate_coverage']=sum(r['refined_coverage8'] for r in positive)/len(positive)
        result['failure_groups']={
            'no_correct_refined_candidate':sum(not r['refined_coverage8'] for r in positive),
            'correct_candidate_but_winner_wrong':sum(r['refined_coverage8'] and not r['layout20'] for r in positive),
            'correct_winner_but_rejected':sum(r['layout20'] and not accepted(r) for r in positive),
            'correct_winner_and_accepted':sum(r['layout20'] and accepted(r) for r in positive)}
    return result


def run(a):
    torch.set_num_threads(1)
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False);selection=read(a.selection)
    if cp['stage']!='B' or selection['binding']!=cp['binding'] or selection['epoch']!=cp['selection_state']['best']['epoch']:
        raise ValueError('not the selected joint checkpoint')
    cfg=Config(**{f.name:cp['config'][f.name] for f in fields(Config)})
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed)
    # Match the formal train/validation runtime. CUDA cumsum has no strict
    # deterministic kernel; do not add a restriction absent from training.
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    meta,batches,source,dataset=load_inputs(a.split,a.microbatch)
    protocol=dict(status='running',split=a.split,checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),
        selection=str(a.selection),selected_epoch=selection['epoch'],threshold=selection['threshold'],
        fixed_secondary_threshold=.3,config=cp['config'],source=source,sample_count=len(meta['pairs']),
        microbatch=a.microbatch,precision='fp32',pid=os.getpid(),model_input_fields=list(NAMES),
        gt_used_to_generate_candidates=False,real_used_to_select_checkpoint=False,threshold_fitting=False,
        historical_real_design_exposure=True,script_sha256=sha(__file__),
        deterministic_note='same seeded runtime as training; CUDA reductions not guaranteed bitwise identical')
    save(out/'protocol.json',protocol)
    try:
        device=torch.device('cuda:0')
        model=SeamContextModel(cfg).to(device);model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False)
        predictions=[]
        with (out/'pair_predictions.jsonl').open('x') as stream,torch.no_grad():
            for pairs,batch in batches:
                tensors=[torch.as_tensor(np.array(batch[k],copy=True),device=device,dtype=torch.bool if k.startswith('contour_valid') else torch.float32) for k in NAMES]
                o=model(*tensors,decode=True,verify=True)
                for b,p in enumerate(pairs):
                    row=describe_prediction(o,b,p,cfg)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');predictions.append(row)
                stream.flush()
                if len(predictions)%64<a.microbatch or len(predictions)==len(meta['pairs']):
                    save(out/'status.json',dict(status='inference',processed=len(predictions),total=len(meta['pairs']),seconds=time.time()-started,pid=os.getpid()))
                    print(json.dumps(dict(processed=len(predictions),total=len(meta['pairs']),seconds=round(time.time()-started,2))),flush=True)
            os.fsync(stream.fileno())
        if [r['pair_id'] for r in predictions] != [p['pair_id'] for p in meta['pairs']]:
            raise ValueError('missing or duplicate predictions')
        save(out/'prediction_complete.json',dict(status='all_predictions_frozen',count=len(predictions),
            checkpoint_sha256=protocol['checkpoint_sha256'],predictions_sha256=sha(out/'pair_predictions.jsonl')))
        rows=attach_targets(predictions,meta,a.split,dataset)
        with (out/'case_diagnostics.jsonl').open('x') as stream:
            for r in rows:stream.write(json.dumps(r,allow_nan=False)+'\n')
        groups={'all':rows}
        if a.split=='dunhuang':
            groups.update(reviewed295_with_original_negatives=[r for r in rows if not r['label'] or r['review_keep']],
                strict547=[r for r in rows if r['strict_member']])
        if a.split=='turufan':groups['positive301']=[r for r in rows if r['label']]
        summary=dict(status='complete',split=a.split,threshold=selection['threshold'],
            groups={k:dict(primary=summarize(v,selection['threshold']),fixed03=summarize(v,.3)) for k,v in groups.items()},
            checkpoint_sha256=protocol['checkpoint_sha256'],layout_gt_available=a.split!='turufan')
        if a.split=='sim_test':
            from .prepare import TRAIN
            train_sources=set().union(*(sources(e['source_row']) for e in read(TRAIN)['entries']))
            overlap={p['pair_id'] for p in meta['pairs'] if sources(p)&train_sources}
            summary['train_source_overlap_pairs']=len(overlap)
            independent=[r for r in rows if r['pair_id'] not in overlap]
            summary['source_independent_test_count']=len(independent)
            if independent:summary['groups']['train_source_disjoint']=dict(primary=summarize(independent,selection['threshold']),fixed03=summarize(independent,.3))
        save(out/'summary.json',summary)
        protocol.update(status='complete',seconds=time.time()-started,max_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        save(out/'protocol.json',protocol);save(out/'status.json',dict(status='complete',count=len(rows),seconds=time.time()-started))
        print(json.dumps(dict(status='complete',split=a.split,seconds=time.time()-started)),flush=True)
    except Exception as error:
        save(out/'failure.json',dict(error=repr(error),pid=os.getpid()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--selection',required=True)
    p.add_argument('--split',choices=('dunhuang','dunhuang_cv','turufan','sim_test'),required=True)
    p.add_argument('--out',required=True);p.add_argument('--microbatch',type=int,default=8)
    run(p.parse_args())
