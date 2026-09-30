"""Evaluate only the completed SIM-selected S7-H on unchanged real inputs."""
import argparse
import json
import os
from pathlib import Path
import random
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from ..s7_hard_finetune.model import load_origin
from ..s7_hard_finetune.launch import MATCHER, HEAD
from ..matched_only.inference import FrozenMatchedInference
from ..seam_context_v3.evaluate_external import load_inputs, NAMES, GT
from ..seam_context_v3.prepare import read, save, sha


def check_selection(status, selection, cp):
    if (status.get('status')!='training_complete' or
        status.get('stop_reason')!='prespecified_M8_C8_finetune_budget_completed'):
        raise ValueError('formal fine-tuning has not completed')
    for phase in ('matcher','scorer'):
        s=status[phase]
        if (s['epochs'],s['updates'],s['exposures'])!=(8,984,62656):
            raise ValueError('stage budget differs from approved run')
    if (cp['phase']!='scorer' or cp['binding']!=selection['binding'] or
        cp['best']!=status['scorer']['best'] or
        cp['best']!={k:selection[k] for k in ('epoch','key','threshold')}):
        raise ValueError('not the SIM-selected Scorer checkpoint')
    if not .2<=selection['threshold']<=.8:
        raise ValueError('unexpected SIM threshold')


def load_selected(root):
    cp=torch.load(root/'best_scorer.pt',map_location='cpu',weights_only=False)
    selection=read(root/'scorer_selection.json')
    check_selection(read(root/'status.json'),selection,cp)
    model=load_origin(MATCHER,HEAD)
    model.load_state_dict(cp['model'],strict=True)
    frozen=FrozenMatchedInference(model.base_model,model.score_head)
    return frozen,cp,selection


def prediction(o,pair):
    s=o.score_details['selection'];t=s.translation_a_to_b_rc[0].detach().cpu().numpy()
    valid=bool(s.layout_valid[0])
    return dict(pair_id=pair['pair_id'],score=float(o.fused_probability[0]),
        decision_valid=bool(o.decision_valid[0]),layout_valid=valid,
        translation=t.tolist() if valid and np.isfinite(t).all() else None,
        endpoints_a=int(s.mask_a[0].sum()),endpoints_b=int(s.mask_b[0].sum()),
        inlier_count=int(s.candidate_inliers[0].sum()),
        used_fallback=bool(o.score_details['used_fallback'][0]))


def attach(predictions,meta,split):
    gt={r['pair_id']:r for r in read(GT)['positive_pairs']} if split=='dunhuang_cv' else {}
    rows=[]
    for r,p in zip(predictions,meta['pairs']):
        if r['pair_id']!=p['pair_id']:raise ValueError('target join order differs')
        target=None
        if p['label'] and split=='dunhuang_cv':
            g=gt[p['pair_id']]
            if (g['fragment_a_token'],g['fragment_b_token'])!=(p['fragment_a_id'],p['fragment_b_id']):
                raise ValueError('GT fragment order differs')
            target=g['translation_gt_a_to_b_rc']
        error=float(np.linalg.norm(np.array(r['translation'])-target)) if r['translation'] is not None and target is not None else None
        rows.append(dict(r,label=int(p['label']),fold=p['fold'],target_translation_rc=target,
            layout_error_px=error,layout_good_20=bool(error is not None and error<=20) if target is not None else None))
    return rows


def run(a):
    torch.set_num_threads(1)
    random.seed(260913);np.random.seed(260913);torch.manual_seed(260913);torch.cuda.manual_seed_all(260913)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root,out=Path(a.training),Path(a.out)
    model,cp,selection=load_selected(root)
    out.mkdir(parents=True,exist_ok=False);started=time.time()
    meta,batches,source,_=load_inputs(a.split,1)
    protocol=dict(status='inference',split=a.split,source=source,model='S7-H matched_tokens',
        selected_matcher_epoch=read(root/'matcher_selection.json')['epoch'],
        selected_scorer_epoch=selection['epoch'],threshold=selection['threshold'],
        checkpoint_sha256=sha(root/'best_scorer.pt'),training_binding=cp['binding'],
        real_used_to_select_checkpoint=False,threshold_fitting=False,microbatch=1,
        network_input_fields=list(NAMES),gt_used_in_forward=False,
        historical_real_design_exposure=True,layout_gt_available=a.split=='dunhuang_cv',pid=os.getpid())
    save(out/'protocol.json',protocol)
    try:
        model=model.cuda();predictions=[]
        with (out/'pair_predictions.jsonl').open('x') as f:
            for pairs,batch in batches:
                values=[torch.as_tensor(np.array(batch[k],copy=True),device='cuda',
                    dtype=torch.bool if k.startswith('contour_valid') else torch.float32) for k in NAMES]
                row=prediction(model(*values),pairs[0]);predictions.append(row)
                f.write(json.dumps(row,allow_nan=False)+'\n')
                if len(predictions)%64==0:
                    f.flush();save(out/'status.json',dict(status='inference',count=len(predictions),seconds=time.time()-started))
            f.flush();os.fsync(f.fileno())
        if [r['pair_id'] for r in predictions]!=[p['pair_id'] for p in meta['pairs']]:raise ValueError('prediction population differs')
        save(out/'prediction_complete.json',dict(status='all_predictions_frozen',count=len(predictions),
            checkpoint_sha256=protocol['checkpoint_sha256'],predictions_sha256=sha(out/'pair_predictions.jsonl')))
        detailed=attach(predictions,meta,a.split)
        with (out/'case_diagnostics.jsonl').open('x') as f:
            for r in detailed:f.write(json.dumps(r,allow_nan=False)+'\n')
        protocol.update(status='complete',seconds=time.time()-started,count=len(detailed))
        save(out/'protocol.json',protocol);save(out/'status.json',dict(status='complete',count=len(detailed),seconds=time.time()-started))
        print(json.dumps(protocol),flush=True)
    except Exception as error:
        save(out/'failure.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--out',required=True)
    p.add_argument('--split',choices=('dunhuang_cv','turufan'),required=True);run(p.parse_args())
