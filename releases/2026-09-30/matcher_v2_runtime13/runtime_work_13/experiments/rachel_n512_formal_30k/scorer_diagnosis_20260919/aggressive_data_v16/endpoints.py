"""Audit end-only shortening along the *original* common contour order.

The contour seam is unwrapped once across its largest unsupported gap. It is
never re-ordered by a proposed cutting curve, nor split into convenient new
seams after cropping. Existing unsupported/eroded gaps receive no length.
"""
import numpy as np


def source_order(weights, supported):
    weights=np.asarray(weights,float);supported=np.asarray(supported,bool)
    ids=np.flatnonzero(supported)
    if len(ids)<2:raise ValueError('no open common seam to shorten')
    arc=np.r_[0.,np.cumsum(weights[:-1])];perimeter=float(weights.sum())
    forward=(arc[np.roll(ids,-1)]-arc[ids])%perimeter
    gaps=np.maximum(0.,forward-weights[ids])
    # A completely closed shared outline has no well-defined seam ends.
    if float(gaps.max())<2:raise ValueError('closed common seam has no declared ends')
    start=(int(np.argmax(gaps))+1)%len(ids)
    return np.roll(ids,-start)


def ordered_end_check(kept,weights,mode):
    kept=np.asarray(kept,bool);weights=np.asarray(weights,float)
    where=np.flatnonzero(kept)
    if not len(where):raise ValueError('no retained seam')
    first,last=int(where[0]),int(where[-1])
    if not np.all(kept[first:last+1]):raise ValueError('additional cut removes an interior seam segment')
    left=float(weights[:first].sum());right=float(weights[last+1:].sum())
    ends=int(left>0)+int(right>0)
    if mode not in ('one','both'):raise ValueError('explicit end mode required')
    if ends!=(1 if mode=='one' else 2):raise ValueError('cut does not remove the requested seam end(s)')
    return dict(mode=mode,removed_start_px=left,removed_end_px=right,
        retained_middle_px=float(weights[kept].sum()),interior_removed_px=0.,
        original_order_preserved=True)


def audit_endpoints(bands,before,after,mode):
    checks={}
    for side in 'ab':
        p,w,valid=bands[side]
        order=source_order(w,valid&np.roll(valid,-1))
        # retained_support proof arrays contain only `valid` source points.
        rank=np.cumsum(valid)-1
        proof_order=rank[order]
        active=before[side+'_physically_retained'][proof_order]
        ids=proof_order[active];ww=w[order[active]]
        kept=after[side+'_physically_retained'][ids]
        checks[side]=ordered_end_check(kept,ww,mode)
    return checks
