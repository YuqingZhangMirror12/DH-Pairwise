"""Threshold fitting is explicit and separate from retained TEST reporting."""
import math
import numpy as np


def accepted(row,threshold):return row['numeric_valid'] and row['has_candidate'] and row['score']>=threshold


def auc(rows):
    positive=[r['score'] if r['numeric_valid'] and r['has_candidate'] else 0. for r in rows if r['label']]
    negative=[r['score'] if r['numeric_valid'] and r['has_candidate'] else 0. for r in rows if not r['label']]
    if not positive or not negative:return None
    # Exact Mann-Whitney via sorted negative scores, including ties.
    n=np.sort(negative);p=np.asarray(positive)
    return float((np.searchsorted(n,p,side='left')+np.searchsorted(n,p,side='right')).sum())/(2*len(p)*len(n))


def summarize(rows,threshold):
    if not rows:raise ValueError('empty metric population')
    pos=[r for r in rows if r['label']];neg=[r for r in rows if not r['label']];known=[r for r in pos if r['gt_known']]
    tp=sum(accepted(r,threshold) for r in pos);fp=sum(accepted(r,threshold) for r in neg);fn=len(pos)-tp
    jt=sum(accepted(r,threshold) and r['layout20'] for r in known);wrong=sum(accepted(r,threshold) and not r['layout20'] for r in known)
    has_gt=bool(known)
    return dict(pairs=len(rows),positive=len(pos),negative=len(neg),layout_gt_positives=len(known),threshold=threshold,
        auc=auc(rows),pair_tp=tp,pair_fp=fp,pair_fn=fn,pair_tn=len(neg)-fp,
        pair_f1=2*tp/max(1,2*tp+fp+fn),accuracy=(tp+len(neg)-fp)/len(rows),
        actual_fpr=fp/len(neg) if neg else None,
        layout_correct=sum(r['layout20'] for r in known) if has_gt else None,
        candidate_coverage=sum(r['candidate_coverage'] for r in known) if has_gt else None,
        deduplicated_coverage=sum(r['deduplicated_coverage'] for r in known) if has_gt else None,
        correct_and_accepted=jt if has_gt else None,
        correct_but_rejected=sum(r['layout20'] and not accepted(r,threshold) for r in known) if has_gt else None,
        wrong_pose_accepted=wrong if has_gt else None,
        joint_f1=2*jt/max(1,2*jt+fp+wrong+len(known)-jt) if has_gt else None,
        invalid=sum(not r['numeric_valid'] for r in rows),no_candidate=sum(not r['has_candidate'] for r in rows))


def fpr_threshold(rows,rate):
    """Lowest threshold with empirical negative FPR <= budget, ties not split.

    For operating thresholds the caller supplies CAL rows. Same-population
    negatives may be used only for explicitly labelled development ROC plots.
    """
    if rate not in (.01,.02,.05):raise ValueError('unregistered FPR target')
    neg=[r for r in rows if not r['label']]
    if not neg:raise ValueError('negative CAL population required')
    scores=sorted((r['score'] for r in neg if r['numeric_valid'] and r['has_candidate']),reverse=True)
    budget=math.floor(rate*len(neg)+1e-12)
    if len(scores)<=budget:return 0.
    return float(np.nextafter(float(scores[budget]),np.inf))


def fit_threshold(rows,head):
    # Preserve historical Scorer grid and .30 tie rule. Raw Q has no probability
    # grid: its diagnostic default uses CAL negatives at FPR 2%.
    if head=='q':return fpr_threshold(rows,.02)
    choices=[]
    for i in range(20,81):
        t=i/100;v=summarize(rows,t)
        if v['joint_f1'] is None:raise ValueError('CAL joint threshold needs layout GT')
        choices.append((v['joint_f1'],-abs(t-.3),t))
    return max(choices,key=lambda x:x[:2])[2]


def evaluate_setting(rows,cal_rows,head,diagnostic_roc=True):
    t=fit_threshold(cal_rows,head)
    return dict(primary=summarize(rows,t),calibrated_on_pairs=len(cal_rows),
        frozen_cal_fpr={str(rate):summarize(rows,fpr_threshold(cal_rows,rate)) for rate in (.01,.02,.05)},
        development_empirical_roc=({str(rate):summarize(rows,fpr_threshold(rows,rate)) for rate in (.01,.02,.05)}
                                  if diagnostic_roc else None))
