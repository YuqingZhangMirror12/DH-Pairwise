"""Two disjoint GPU shards; exact fixed Matcher, FP32, no model updates."""
import argparse
import os
import time
from functools import lru_cache
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from support import *
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local.model import select_predicted_inliers

@lru_cache(maxsize=96)
def fragment(path,side):
    with np.load(path,allow_pickle=False) as z:
        shape=tuple(z['mask_'+side+'_shape'])
        raw_points=z['points_rc_'+side];raw_valid=z['contour_valid_'+side]
        if not 4<=len(raw_points)<=512 or raw_valid.shape!=(len(raw_points),):
            raise ValueError('invalid source contour length')
        points=np.zeros((512,2),np.float32);valid=np.zeros(512,bool)
        points[:len(raw_points)]=raw_points;valid[:len(raw_points)]=raw_valid
        return (np.unpackbits(z['mask_'+side+'_packed'],axis=-1,count=int(shape[-1])).reshape(shape).astype(np.float32),
                points,valid)

def run(root,shard,batch_size=8):
    torch.set_num_threads(2)
    if torch.cuda.device_count()!=1: raise ValueError('one assigned GPU required')
    base=cache.old.load_decoupled_checkpoint(cache.source_checkpoint()).base_model.eval().requires_grad_(False).to('cuda:0')
    root=Path(root); start=time.time();done=0
    for split in ('train','val'):
        rows=read(root/'groups'/f'{split}_pairs.json');folder=root/'cache'/split
        arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r+',allow_pickle=False) for k in cache.ARRAYS}
        # Group ownership: no two writers ever touch the same pair or anchor group.
        indices=[i for i,r in enumerate(rows) if r['group_id']%2==shard and not arrays['ready'][i]]
        for offset in range(0,len(indices),batch_size):
            ids=indices[offset:offset+batch_size]
            a=[fragment(rows[i]['anchor_path'],rows[i]['anchor_side']) for i in ids]
            b=[fragment(rows[i]['partner_path'],rows[i]['partner_side']) for i in ids]
            packed=[np.stack([v[k] for v in values]) for k in range(3) for values in (a,b)]
            # mask_a, mask_b, points_a, points_b, valid_a, valid_b
            args=[torch.from_numpy(x).to('cuda:0') for x in packed]
            with torch.inference_mode():
                o=base(*args)
                s=select_predicted_inliers(o.assignment,args[2],args[3],args[4],args[5])
                ij=s.candidate_indices.clamp_min(0);bi=torch.arange(len(ids),device='cuda:0')[:,None]
                weights=torch.where(s.candidate_valid,o.assignment[bi,ij[...,0],ij[...,1]],0.)
            values=dict(features_a=o.token_features_a,features_b=o.token_features_b,
                        valid_a=args[4],valid_b=args[5],points_a=args[2],points_b=args[3],candidate_weights=weights,
                        training_valid=o.training_valid,decision_valid=o.decision_valid)
            for name in ('mask_a','mask_b','candidate_indices','candidate_valid','candidate_inliers',
                         'translation_a_to_b_rc','layout_valid'):values[name]=getattr(s,name)
            for name,value in values.items():
                x=value.detach().cpu().numpy()
                if name!='translation_a_to_b_rc' and x.dtype.kind=='f' and not np.isfinite(x).all():
                    raise ValueError('nonfinite evidence '+name)
                arrays[name][ids]=x
            arrays['ready'][ids]=True;done+=len(ids)
            if offset==0 or (offset+batch_size)%512==0:
                save(root/'cache'/f'shard_{shard}_status.json',dict(status='precomputing',split=split,
                    shard=shard,completed_this_process=done,split_remaining=max(0,len(indices)-offset-len(ids)),
                    seconds=time.time()-start,microbatch=batch_size,matcher_frozen=True,
                    peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20))
        for value in arrays.values():value.flush()
    save(root/'cache'/f'shard_{shard}_status.json',dict(status='complete',shard=shard,
        completed_this_process=done,seconds=time.time()-start,microbatch=batch_size,matcher_frozen=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--shard',type=int,choices=(0,1),required=True)
    p.add_argument('--batch-size',type=int,default=8);a=p.parse_args();run(a.root,a.shard,a.batch_size)
