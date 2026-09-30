"""Source-isolated real epoch selection; never evaluate role real_test here."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from .evidence import PairEvidence
from .matcher import INPUTS
from .metrics import summarize

ROLES=('real_cal','real_select')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def bind_plan(path):
    plan=read(path)
    if (plan.get('schema')!='threshold-joint-real-split/1' or not plan.get('source_disjoint')
            or plan['role_folds']!={'real_cal':[1],'real_select':[2,3,4],'real_test':[0]}):
        raise ValueError('registered source-disjoint development plan required')
    binding=dict(plan_sha256=sha(path),gt_sha256=sha(plan['gt_path']),sources={})
    for ds,spec in plan['datasets'].items():
        meta=read(spec['remote_manifest'])
        if sha(spec['remote_manifest'])!=spec['manifest_sha256']:
            raise ValueError('real source manifest changed')
        seen=set();fragment_sets={};group_sets={}
        for role,description in spec['roles'].items():
            expected=[r for r in meta['pairs'] if r['fold'] in plan['role_folds'][role]
                      and r['pair_id'] not in spec['excluded_gt_pair_ids']]
            if [r['pair_id'] for r in expected]!=description['pair_ids']:
                raise ValueError('real role identity changed')
            ids=set(description['pair_ids'])
            if seen&ids:raise ValueError('duplicate development/test pairs')
            seen|=ids
            fragment_sets[role]={r[k] for r in expected for k in ('fragment_a_id','fragment_b_id')}
            group_sets[role]={meta['fragment_source_group'][f] for f in fragment_sets[role]}
            for previous in fragment_sets:
                if previous!=role and (fragment_sets[role]&fragment_sets[previous] or group_sets[role]&group_sets[previous]):
                    raise ValueError('source/fragment leakage')
        if len(seen)!=len(meta['pairs'])-len(spec['excluded_gt_pair_ids']):raise ValueError('missing real role rows')
        prepared=Path(spec['prepared'])/'inputs.npz'
        binding['sources'][ds]=dict(manifest_sha256=spec['manifest_sha256'],inputs_sha256=sha(prepared))
    return binding


def domain_metric(ds,rows,threshold):
    metrics=summarize(rows,threshold)
    if ds=='turufan':
        for k in list(metrics):
            if k.startswith('joint_') or k in ('layout20','layout20_count','candidate_coverage',
                    'candidate_coverage_count','covered_but_winner_wrong','winner_correct_but_rejected',
                    'positive_no_correct_candidate','wrong_pose_accepted'):
                metrics[k]=None
    return metrics


def select_from_development(rows, config):
    if set(rows)!=set(('dunhuang_cv','turufan')):raise ValueError('both real domains required')
    output={}
    for ds,data in rows.items():
        if set(data)!=set(ROLES):raise ValueError('TEST cannot enter real epoch selection')
        cal=data['real_cal'];select=data['real_select']
        if ds=='dunhuang_cv' and any(r['label'] and not r['gt_known'] for r in cal+select):
            raise ValueError('Dunhuang selection requires all positive layout GT')
        if ds=='turufan' and any(r['gt_known'] for r in cal+select):
            raise ValueError('Turufan must not have invented layout GT')
        if {r['pair_id'] for r in cal}&{r['pair_id'] for r in select}:raise ValueError('calibration/select overlap')
        if any(not any(r['label'] for r in rr) or all(r['label'] for r in rr) for rr in (cal,select)):
            raise ValueError('real calibration and selection each require both classes')
        target='joint_f1' if ds=='dunhuang_cv' else 'f1';choices=[]
        for integer in range(20,81):
            t=integer/100;metrics=domain_metric(ds,cal,t)
            precision=metrics['joint_precision' if ds=='dunhuang_cv' else 'precision']
            choices.append(((metrics[target],-abs(t-config.threshold_tie_preference),precision,t),t,metrics))
        _,t,cal_metrics=max(choices,key=lambda x:x[0])
        output[ds]=dict(threshold=t,calibration=cal_metrics,select=domain_metric(ds,select,t))
    a=output['dunhuang_cv']['select'];b=output['turufan']['select']
    primary=(a['joint_f1']+b['f1'])/2
    return dict(status='development_selection',thresholds={k:v['threshold'] for k,v in output.items()},
        key=[primary,a['layout20'],(a['f1']+b['f1'])/2],selection_value=primary,domains=output,
        test_used=False,real_used=True,development_evaluation=True,gradients_used=False)


class RealDevelopment:
    def __init__(self,path,binding):
        self.plan=read(path)
        if sha(path)!=binding['plan_sha256']:raise ValueError('development plan changed')
        if sha(self.plan['gt_path'])!=binding['gt_sha256']:raise ValueError('layout GT changed')
        self.gt={r['pair_id']:r for r in read(self.plan['gt_path'])['positive_pairs']}
        self.data={}
        for ds,spec in self.plan['datasets'].items():
            if sha(spec['remote_manifest'])!=binding['sources'][ds]['manifest_sha256']:raise ValueError('real manifest changed')
            prepared=Path(spec['prepared'])/'inputs.npz'
            if sha(prepared)!=binding['sources'][ds]['inputs_sha256']:raise ValueError('real image/contour cache changed')
            meta=read(spec['remote_manifest'])
            with np.load(prepared,allow_pickle=False) as z:
                arrays={k:z[k] for k in ('packed_masks','points','valid')}
            n=len(meta['fragment_ids'])
            if (arrays['packed_masks'].shape!=(n,800,100) or arrays['points'].shape!=(n,512,2)
                    or arrays['valid'].shape!=(n,512)):
                raise ValueError('real preprocessing shapes changed')
            self.data[ds]=(meta,arrays,{f:i for i,f in enumerate(meta['fragment_ids'])})

    @torch.no_grad()
    def evaluate(self,model,device,config):
        start=time.time();model.eval();data={}
        rank=dist.get_rank() if dist.is_initialized() else 0
        world=dist.get_world_size() if dist.is_initialized() else 1
        for ds,(meta,arrays,lookup) in self.data.items():
            data[ds]={};by_id={r['pair_id']:r for r in meta['pairs']}
            for role in ROLES:
                ids=self.plan['datasets'][ds]['roles'][role]['pair_ids']
                local=[by_id[i] for i in ids[rank::world]];rows=[]
                for start_idx in range(0,len(local),config.microbatch):
                    pairs=local[start_idx:start_idx+config.microbatch];batch={}
                    for side in 'ab':
                        index=[lookup[r['fragment_'+side+'_id']] for r in pairs]
                        batch['mask_'+side]=torch.from_numpy(np.unpackbits(arrays['packed_masks'][index],axis=-1).astype(np.float32)[:,None]).to(device)
                        batch['points_rc_'+side]=torch.from_numpy(arrays['points'][index].astype(np.float32)).to(device)
                        batch['contour_valid_'+side]=torch.from_numpy(arrays['valid'][index].astype(bool)).to(device)
                    output=model.matcher(*(batch[k] for k in INPUTS))
                    for i,item in enumerate(pairs):
                        pair=PairEvidence.from_matcher(output,i,batch['mask_a'],batch['mask_b'])
                        pred=model.score_pair(pair)  # No frozen-weight candidate cache.
                        target=None
                        if ds=='dunhuang_cv' and item['label']:
                            record=self.gt[item['pair_id']]
                            if (record['fragment_a_token'],record['fragment_b_token'])!=(item['fragment_a_id'],item['fragment_b_id']):raise ValueError('real GT endpoint order differs')
                            target=pair.q.new_tensor(record['translation_gt_a_to_b_rc'])
                        error=float((pred.translation_a_to_b_rc-target).norm()) if target is not None and pred.has_candidate else None
                        rows.append(dict(pair_id=item['pair_id'],label=bool(item['label']),gt_known=target is not None,
                            score=float(pred.score),has_candidate=pred.has_candidate,numeric_valid=pred.numeric_valid,
                            translation=None if not pred.has_candidate else pred.translation_a_to_b_rc.cpu().tolist(),
                            layout20=bool(error is not None and error<=20),error_px=error,
                            candidate_coverage=bool(target is not None and any(float((c.translation-target).norm())<=20 for c in pred.clusters))))
                if world>1:
                    groups=[None]*world;dist.all_gather_object(groups,rows);rows=[r for g in groups for r in g]
                if len(rows)!=len(ids) or {r['pair_id'] for r in rows}!=set(ids):raise ValueError('missing/duplicate real-development predictions')
                data[ds][role]=sorted(rows,key=lambda r:r['pair_id'])
        result=select_from_development(data,config);result['elapsed_seconds']=time.time()-start
        return result,data
