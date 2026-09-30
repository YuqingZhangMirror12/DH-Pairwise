"""Pre-damage unordered fragment-pair identity, not an augmentation/file ID.

Invariant to independent canvas centering, paired quarter turns/reflections and
A/B exchange. Source-pair lineage must ALSO be checked: this exact pixel key is
not an approximate same-manuscript or arbitrary-rescaling identity classifier.
"""
import hashlib
import json

import numpy as np


def cropped(mask):
    value=np.asarray(mask,dtype=bool)
    if value.ndim!=2 or not value.any():raise ValueError('Nonempty 2D base mask required')
    rr,cc=np.nonzero(value)
    return value[rr.min():rr.max()+1,cc.min():cc.max()+1]


def fragment_bytes(mask):
    value=np.ascontiguousarray(mask,dtype=bool)
    return json.dumps(list(value.shape),separators=(',',':')).encode()+b'\0'+np.packbits(value).tobytes()


def base_pair_sha256(a,b):
    a,b=cropped(a),cropped(b)
    variants=[]
    for flip in (False,True):
        aa,bb=(a[:,::-1],b[:,::-1]) if flip else (a,b)
        for k in range(4):
            fragments=sorted((fragment_bytes(np.rot90(aa,k)),fragment_bytes(np.rot90(bb,k))))
            payload=b'pre-damage-fragment-pair/1\0'+b''.join(len(x).to_bytes(8,'little')+x for x in fragments)
            variants.append(hashlib.sha256(payload).hexdigest())
    return min(variants)
