"""GT-free rotation and candidate selection. No pose averaging or Q rescaling."""
import math
import numpy as np

INPUTS = ('mask_a','mask_b','points_rc_a','points_rc_b','contour_valid_a','contour_valid_b')
VIEW_SETS = ((0,), (0,180), (0,90), (0,90,180), (0,90,180,270))
METHODS = ('A','B','C')
REVISION = 'common-rotation-ensemble/1'


def quarter_turns(angle):
    if type(angle) is not int or angle not in (0,90,180,270):
        raise ValueError('only registered lossless common rotations are supported')
    return angle//90


def rotate_vectors(vectors, angle, inverse=False):
    k=quarter_turns(angle); x=np.asarray(vectors).copy()
    if x.shape[-1:]!=(2,) or not np.isfinite(x).all():raise ValueError('finite row,column vectors required')
    for _ in range((-k if inverse else k)%4):x=np.stack((-x[...,1],x[...,0]),axis=-1)
    return x


def rotate_inputs(inputs, angle):
    if set(inputs)!=set(INPUTS):raise ValueError('only six model inputs; targets forbidden')
    k=quarter_turns(angle); result={}
    for side in 'ab':
        m=np.asarray(inputs['mask_'+side]);p=np.asarray(inputs['points_rc_'+side]);v=np.asarray(inputs['contour_valid_'+side])
        if m.ndim!=4 or m.shape[1]!=1 or m.shape[-2:]!=(800,800) or p.shape!=(len(m),512,2) or v.shape!=(len(m),512):
            raise ValueError('unchanged 800/N512 input geometry required')
        if m.dtype!=np.float32 or p.dtype!=np.float32 or v.dtype!=bool:raise ValueError('FP32/boolean inputs required')
        q=p.copy()
        for _ in range(k):q=np.stack((799-q[...,1],q[...,0]),axis=-1)
        # Preserve the padding contract. Contour start/order is NOT re-extracted.
        if k:q=np.where(v[...,None],q,0).astype(np.float32)
        result['mask_'+side]=np.rot90(m,k,axes=(-2,-1)).copy()
        result['points_rc_'+side]=q
        result['contour_valid_'+side]=v.copy()
    return result


def candidate_key(c, head):
    value=c['q_sum'] if head=='q' else c['heads'][head]['logit']
    return (value,-c['angle'],-c['index'])


def candidate_score(c,head):return c['q_sum'] if head=='q' else c['heads'][head]['score']


def deduplicate(candidates,head,radius=16.):
    """Score-first complete-link groups, retaining one ACTUAL candidate per group.

    All members must lie within radius of every other member, so a chain of
    near poses cannot join two distant modes. Group score is not a sum.
    """
    groups=[]
    for c in sorted(candidates,key=lambda c:candidate_key(c,head),reverse=True):
        target=next((g for g in groups if all(np.linalg.norm(np.asarray(c['translation'])-m['translation'])<=radius for m in g)),None)
        if target is None:groups.append([c])
        else:target.append(c)
    return groups


def combine(views,angles=(0,),method='B',head='q',dedup_radius=16.):
    """Pure selection from serialized, unlabelled per-view candidates.

    A: select each view's own head winner (native Q winner for q-only), then
       choose the view whose winner has greatest raw sum(Q).
    B: pooled actual candidate with maximum head score, pose-near duplicates
       represented by their highest-scoring actual member.
    C: same pose as B, acceptance score is the mean of per-view top scores.
    A/B/C at {0} all reproduce that head's single-view decision.
    """
    angles=tuple(angles)
    if not angles or len(set(angles))!=len(angles) or angles[0]!=0 or method not in METHODS:
        raise ValueError('unique registered views including identity and explicit method required')
    for a in angles:quarter_turns(a)
    if set(angles)-set(views):raise ValueError('missing requested view')
    if dedup_radius!=16.:raise ValueError('preregistered merge radius is 16 pixels')
    candidates=[]; winners=[]
    for a in angles:
        row=views[a]
        if 'label' in row or 'target_translation_rc' in row:raise ValueError('targets must not enter ensemble selection')
        cs=row['candidates'] if row['numeric_valid'] else []
        for c in cs:
            if c['angle']!=a or not np.isfinite(c['translation']).all():raise ValueError('candidate frame differs')
            if not all(math.isfinite(x) for x in (c['q_sum'],candidate_score(c,head),candidate_key(c,head)[0])):raise ValueError('nonfinite candidate score')
        candidates.extend(cs)
        winners.append(max(cs,key=lambda c:candidate_key(c,head)) if cs else None)
    if not candidates:
        return dict(has_candidate=False,numeric_valid=all(views[a]['numeric_valid'] for a in angles),
                    translation=None,score=0.,winner=None,union_candidates=[],deduplicated_candidates=[],
                    agreement_views=0,views=list(angles),method=method,head=head)
    groups=deduplicate(candidates,head,dedup_radius)
    winner=(max((w for w in winners if w is not None),key=lambda c:(c['q_sum'],-c['angle'],-c['index']))
            if method=='A' else groups[0][0])
    score=(sum(0. if w is None else candidate_score(w,head) for w in winners)/len(angles)
           if method=='C' else candidate_score(winner,head))
    return dict(has_candidate=True,numeric_valid=True,translation=list(winner['translation']),score=score,
        winner=dict(angle=winner['angle'],index=winner['index']),
        union_candidates=[c['translation'] for c in candidates],
        deduplicated_candidates=[g[0]['translation'] for g in groups],
        agreement_views=int(sum(w is not None and np.linalg.norm(np.asarray(w['translation'])-winner['translation'])<=16 for w in winners)),
        candidate_count=len(candidates),deduplicated_count=len(groups),
        views=list(angles),method=method,head=head)


def annotate(prediction,label,target=None):
    if type(label) is not bool or (target is not None and not label):raise ValueError('positive-only layout targets')
    if target is not None and (np.asarray(target).shape!=(2,) or not np.isfinite(target).all()):raise ValueError('invalid GT')
    def error(p):return None if target is None or p is None else float(np.linalg.norm(np.asarray(p)-target))
    e=error(prediction['translation']);known=target is not None
    return dict(prediction,label=label,gt_known=known,error_px=e,
        target_translation_rc=None if target is None else list(target),
        layout20=None if not known else bool(prediction['numeric_valid'] and e is not None and e<=20),
        candidate_coverage=None if not known else any(error(p)<=20 for p in prediction['union_candidates']),
        deduplicated_coverage=None if not known else any(error(p)<=20 for p in prediction['deduplicated_candidates']))
