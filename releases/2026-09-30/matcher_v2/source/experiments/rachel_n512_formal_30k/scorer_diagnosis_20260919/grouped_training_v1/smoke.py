"""One CPU integration batch; never a checkpoint or formal training result."""
from support import *
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.local_evidence_v2.model import make

if __name__=='__main__':
    torch.set_num_threads(2);ds=GroupCache(ROOT,'train',require_complete=False)
    ready=np.asarray(ds.arrays['ready'])[ds.groups].all(1);groups=ds.groups[np.flatnonzero(ready)[:12]]
    assert len(groups)==12
    for ids in groups:
        records=[ds.records[i] for i in ids]
        assert len({(r['anchor_path'],r['anchor_side']) for r in records})==1
        p=np.asarray(ds.arrays['points_a'][ids]);assert np.array_equal(p,np.repeat(p[:1],4,axis=0))
    b=ds.batch(groups.ravel(),'cpu');net=make('reference512_h4');o=net(*b.model_args,**b.model_kwargs)
    loss,bce,rank=group_loss(o.logit,b.labels,b.training_valid,ARMS[1]);loss.backward()
    gradients=[p.grad for p in net.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients)>0
    report=dict(status='passed',formal_training=False,device='cpu',groups=12,pair_batch=48,
        loss=float(loss.detach()),bce=float(bce.detach()),rank=float(rank.detach()),
        same_anchor_within_group=True,finite_nonzero_gradients=True,model=net.metadata())
    save(ROOT/'smoke.json',report);print(json.dumps({k:v for k,v in report.items() if k!='model'}))
