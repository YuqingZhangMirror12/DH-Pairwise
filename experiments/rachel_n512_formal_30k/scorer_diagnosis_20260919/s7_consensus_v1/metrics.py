"""Pair classification and joint acceptance/correct-layout are separate."""
import numpy as np


def average_precision(labels,scores):
    labels=np.asarray(labels,dtype=bool);scores=np.asarray(scores,dtype=float)
    if not labels.any():
        return 0.
    order=np.argsort(-scores,kind='stable');y=labels[order];s=scores[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    true=np.cumsum(y)[ends]
    return float((np.diff(np.r_[0,true])*(true/(ends+1))).sum()/y.sum())


def summarize(rows,threshold):
    if not rows:
        raise ValueError('empty evaluation population')
    label=np.array([r['label'] for r in rows],bool)
    known=np.array([r['gt_known'] for r in rows],bool)&label
    scores=np.array([r['score'] for r in rows])
    valid=np.array([r['numeric_valid'] and r['has_candidate'] for r in rows],bool)
    good=np.array([r['layout20'] for r in rows],bool)&known
    covered=np.array([r['candidate_coverage'] for r in rows],bool)&known
    accepted=valid&(scores>=threshold)
    tp=int((label&accepted).sum());fp=int((~label&accepted).sum());fn=int((label&~accepted).sum())
    jt=int((good&accepted).sum());wrong=int((known&~good&accepted).sum());count=int(known.sum())
    jf=fp+wrong;jn=count-jt
    return dict(threshold=float(threshold),pairs=len(rows),positives=int(label.sum()),negatives=int((~label).sum()),
        known_positive_layouts=count,accuracy=float((accepted==label).mean()),tp=tp,fp=fp,fn=fn,
        precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),
        ap=average_precision(label,np.where(valid,scores,0.)),
        layout20_count=int(good.sum()),layout20=int(good.sum())/count if count else None,
        candidate_coverage_count=int(covered.sum()),candidate_coverage=int(covered.sum())/count if count else None,
        covered_but_winner_wrong=int((covered&~good).sum()),
        winner_correct_but_rejected=int((good&~accepted).sum()),
        positive_no_correct_candidate=int((known&~covered).sum()),
        wrong_pose_accepted=wrong,negative_pair_false_positives=fp,
        joint_tp=jt,joint_fp=jf,joint_fn=jn,
        joint_precision=jt/max(1,jt+jf) if count else None,
        joint_recall=jt/count if count else None,
        joint_f1=2*jt/max(1,2*jt+jf+jn) if count else None,
        no_candidate=int((~valid).sum()))


def choose_threshold(cal,config,*,paired_views=True):
    views=('clean','hard') if paired_views else ('mixed',)
    if set(cal)!=set(views):
        raise ValueError('CAL views differ from the explicit validation protocol')
    if paired_views and [r['pair_id'] for r in cal['clean']]!=[r['pair_id'] for r in cal['hard']]:
        raise ValueError('CAL view Pair IDs/order differ')
    # Preserve the declared decimal grid at >= boundaries (not .59+3e-16).
    grid=np.round(np.arange(config.threshold_minimum,
        config.threshold_maximum+config.threshold_step/2,config.threshold_step),12)
    choices=[]
    for threshold in grid:
        metrics=[summarize(cal[v],threshold) for v in views]
        if any(m['joint_f1'] is None for m in metrics):
            raise ValueError('cannot calibrate joint score without GT layouts')
        choices.append((float(np.mean([m['joint_f1'] for m in metrics])),
            -abs(float(threshold)-config.threshold_tie_preference),float(threshold)))
    return max(choices,key=lambda x:x[:2])[2]
