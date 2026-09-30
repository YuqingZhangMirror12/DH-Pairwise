"""Pure CPU acceleration; fixed indices only, no autograd/GT in this search."""
import ctypes
from pathlib import Path
import numpy as np

_LIBRARY = None


def beam_paths(ordered, neighbors, admissible, lp, da, db, node, stop, perimeter_b, width):
    global _LIBRARY
    if _LIBRARY is None:
        path=Path(__file__).with_name('libseam_beam.so')
        if not path.exists():
            raise RuntimeError('build beam_kernel.cpp before decoding; no slow silent fallback')
        _LIBRARY=ctypes.CDLL(str(path))
        _LIBRARY.seam_beam.argtypes=[ctypes.c_int64]*4+[ctypes.c_void_p]*6+[ctypes.c_double]+[ctypes.c_void_p]*4
        _LIBRARY.seam_beam.restype=None
    n,k=neighbors.shape
    gap=np.maximum(da,db)>32
    transition=np.log(np.take_along_axis(lp,gap[...,None].astype(np.int64),-1)[...,0]+1e-8)
    arrays=[np.ascontiguousarray(ordered,np.int64),np.ascontiguousarray(neighbors,np.int64),
            np.ascontiguousarray(admissible,np.uint8),np.ascontiguousarray(transition,np.float64),
            np.ascontiguousarray(node,np.float64),np.ascontiguousarray(db,np.float64)]
    scores=np.empty((n,width),np.float64);travel=np.empty_like(scores)
    parents=np.empty((n,width),np.int64);ranks=np.empty_like(parents)
    outputs=[scores,travel,parents,ranks]
    _LIBRARY.seam_beam(n,width,k,len(ordered),*[ctypes.c_void_p(a.ctypes.data) for a in arrays],
                      perimeter_b,*[ctypes.c_void_p(a.ctypes.data) for a in outputs])
    final=scores+np.log(stop[:,None]+1e-8)
    result=[]
    for idx in np.argsort(-final.ravel(),kind='stable')[:4]:
        e,r=divmod(int(idx),width)
        if not np.isfinite(final[e,r]):continue
        value=float(final[e,r]);chain=[]
        while e>=0:
            chain.append(e);e,r=int(parents[e,r]),int(ranks[e,r])
            if len(chain)>n:raise RuntimeError('cyclic beam predecessor')
        result.append((value,tuple(reversed(chain))))
    return result
