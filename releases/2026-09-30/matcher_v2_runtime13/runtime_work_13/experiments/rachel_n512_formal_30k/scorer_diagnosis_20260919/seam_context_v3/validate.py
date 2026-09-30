"""Source-disjoint simulation threshold calibration and checkpoint selection."""
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from .data import Dataset, collate, to_device, INPUTS
from .seam_proposals import propose
from .targets import compact_targets


def average_precision_score(label,score):
    """Non-interpolated AP with equal-score groups (same convention as sklearn)."""
    order=np.argsort(-score,kind='stable');y=label[order];s=score[order]
    if not y.any():return 0.
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    tp=np.cumsum(y)[ends];precision=tp/(ends+1)
    return float((np.diff(np.r_[0,tp])*precision).sum()/y.sum())


def metrics(rows, threshold):
    label = np.array([r['label'] for r in rows], bool)
    known = np.array([r['gt_known'] for r in rows], bool)
    score = np.array([r['score'] for r in rows])
    valid = np.array([r['numeric_valid'] and r['has_candidate'] for r in rows], bool)
    good = np.array([r['layout20'] for r in rows], bool)
    accepted = valid & (score >= threshold)
    tp = int((label & accepted).sum()); fp = int((~label & accepted).sum()); fn = int((label & ~accepted).sum())
    jtp = int((label & known & accepted & good).sum())
    wrong_pose_accept = int((label & known & accepted & ~good).sum())
    jfp = fp+wrong_pose_accept; jfn = int((label & known).sum())-jtp
    return dict(threshold=float(threshold), count=len(rows), positives=int(label.sum()),
        positive_pose_unknown=int((label & ~known).sum()), pair_tp=tp, pair_fp=fp, pair_fn=fn,
        pair_accuracy=float((label == accepted).mean()), pair_precision=tp/max(tp+fp,1),
        pair_recall=tp/max(tp+fn,1), pair_f1=2*tp/max(2*tp+fp+fn,1),
        pair_ap=float(average_precision_score(label,score)) if label.any() else 0.,
        joint_tp=jtp, joint_fp=jfp, joint_fn=jfn, wrong_pose_accepted=wrong_pose_accept,
        joint_f1=2*jtp/max(2*jtp+jfp+jfn,1), joint_precision=jtp/max(jtp+jfp,1),
        joint_recall=jtp/max(jtp+jfn,1), layout20=int((label & known & good).sum())/max(int((label & known).sum()),1),
        candidate_coverage=int(sum(r['label'] and r['gt_known'] and r['coverage8'] for r in rows))/max(int((label & known).sum()),1),
        q0_candidate_coverage=int(sum(r['label'] and r['gt_known'] and r['q0_coverage8'] for r in rows))/max(int((label & known).sum()),1))


def select_threshold(cal, cfg):
    grid=np.arange(cfg.threshold_min,cfg.threshold_max+cfg.threshold_step/2,cfg.threshold_step)
    candidates=[]
    for threshold in grid:
        m=[metrics(cal[v],threshold) for v in ('clean','hard')]
        candidates.append((float(np.mean([x['joint_f1'] for x in m])), -abs(float(threshold)-.3),float(threshold),m))
    selected=max(candidates,key=lambda x:(x[0],x[1]))
    eligible=[x for x in candidates if all(m['joint_precision']>=.95 and m['joint_tp']>0 for m in x[3])]
    high=max(eligible,key=lambda x:(np.mean([m['joint_recall'] for m in x[3]]),x[1]))[2] if eligible else None
    return selected[2], high


@torch.no_grad()
def evaluate_view(model, manifest, stage, device, microbatch=4, workers=2):
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    dataset=Dataset(manifest)
    subset=Subset(dataset,list(range(rank,len(dataset),world)))
    loader=DataLoader(subset,batch_size=microbatch,collate_fn=collate,num_workers=workers,pin_memory=True)
    result=[];model.eval()
    for batch in loader:
        batch=to_device(batch,device)
        o=model(*(batch[k] for k in INPUTS),decode=True,verify=stage=='B')
        targets=compact_targets(batch,o)
        for b,candidates in enumerate(o.candidates):
            gt=batch['translation_a_to_b_rc'][b].cpu().numpy()
            label=bool(batch['labels'][b]);known=bool(batch['translation_valid'][b])
            errors=[float(np.linalg.norm(c.translation-gt)) for c in candidates]
            zero=propose(o.records[b],o.ot0.real_transport[b],o.ga,o.gb,b,model.cfg)
            if stage=='B' and o.verified[b].has_candidate:
                v=o.verified[b];best=v.winner
                t=v.translations[best].float().cpu().numpy()
                score=float(v.score);valid=bool(np.isfinite(t).all() and np.isfinite(score))
            elif candidates:
                t=candidates[0].translation;score=0.;valid=bool(np.isfinite(t).all())
            else:
                t=None;score=0.;valid=False
            error=float(np.linalg.norm(t-gt)) if valid and known else None
            matching=targets['a'][b]>=0
            gt_ids=torch.nonzero(matching).flatten()
            match_nll=float(-o.ot1.real_transport[b,gt_ids,targets['a'][b,gt_ids]].clamp_min(1e-12).log().mean()) if len(gt_ids) else None
            result.append(dict(pair_id=batch['pair_ids'][b],label=label,gt_known=known,score=score,
                has_candidate=bool(candidates),numeric_valid=valid,translation=t.tolist() if valid else None,
                error_px=error,layout20=bool(error is not None and error<=20),
                coverage8=bool(known and any(e<=20 for e in errors)),
                q0_coverage8=bool(known and any(np.linalg.norm(c.translation-gt)<=20 for c in zero)),
                candidate_count=len(candidates),valid_points_a=int(o.ga.counts[b]),valid_points_b=int(o.gb.counts[b]),
                gt_match_nll=match_nll,
                affinity_delta_abs_max=float(o.records[b].delta.abs().max()) if len(o.records[b].delta) else 0.,
                ot_residual=float(torch.maximum(o.ot1.diagnostics.row_residual_max[b],o.ot1.diagnostics.col_residual_max[b]))))
    if world>1:
        buckets=[None]*world;dist.all_gather_object(buckets,result);result=[r for part in buckets for r in part]
    if len(result)!=len(dataset) or len({r['pair_id'] for r in result})!=len(result):
        raise RuntimeError('validation duplicate/missing Pair: no distributed padding allowed')
    return sorted(result,key=lambda x:x['pair_id'])


def validate(model,root,stage,device,microbatch=4,workers=2):
    data={}
    for partition in ('cal','select'):
        data[partition]={v:evaluate_view(model,root/(partition+'_'+v+'.json'),stage,device,microbatch,workers)
                         for v in ('clean','hard')}
    threshold,high=select_threshold(data['cal'],model.cfg) if stage=='B' else (.3,None)
    selected={v:metrics(data['select'][v],threshold) for v in ('clean','hard')}
    primary=float(np.mean([m['joint_f1' if stage=='B' else 'candidate_coverage'] for m in selected.values()]))
    correspondence=[r['gt_match_nll'] for view in data['select'].values() for r in view if r['gt_match_nll'] is not None]
    tiebreak=(float(np.mean([m['layout20'] for m in selected.values()])),
              float(np.mean([m['pair_ap'] for m in selected.values()])) if stage=='B' else -float(np.mean(correspondence)))
    return dict(stage=stage,threshold=threshold,selection_value=primary,tiebreak=list(tiebreak),
        selected=selected,fixed03={v:metrics(data['select'][v],.3) for v in ('clean','hard')},
        high_precision_cal_threshold=high,
        high_precision_transfer={v:metrics(data['select'][v],high) for v in ('clean','hard')} if high is not None else None),data
