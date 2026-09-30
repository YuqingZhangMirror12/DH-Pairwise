"""Same-data 48-pair batches: independent PairBCE versus group competition."""
import argparse
import fcntl
import os
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from support import *
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.local_evidence_v2.model import make
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only import train as prior

@torch.inference_mode()
def group_eval(net,dataset):
    net.eval();n=len(dataset);scores=np.empty(n);z=np.empty(n);valid=np.empty(n,bool)
    for start in range(0,n,48):
        ids=np.arange(start,min(n,start+48));b=dataset.batch(ids,'cuda:0');o=net(*b.model_args,**b.model_kwargs)
        z[ids]=o.logit.cpu().numpy();valid[ids]=b.decision_valid.cpu().numpy()
        scores[ids]=o.logit.sigmoid().cpu().numpy()
    ids=dataset.groups;labels=np.asarray(dataset.arrays['label']);logits=z[ids]
    target=labels[ids].argmax(1);pos=logits[np.arange(len(ids)),target]
    neg=np.where(labels[ids]>0,-np.inf,logits);ranks=1+(neg>=pos[:,None]).sum(1)
    accepted=(scores>=.3)&valid;neg_ids=ids[labels[ids]==0].reshape(len(ids),3)
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
    return dict(groups=len(ids),group_top1=float((ranks==1).mean()),group_mrr=float((1/ranks).mean()),
        mean_positive_minus_hardest_negative=float((pos-neg.max(1)).mean()),
        negative_only_group_any_accept_at03=float(accepted[neg_ids].any(1).mean()),
        fixed03=dict(accuracy=accuracy_score(labels,accepted),precision=precision_score(labels,accepted,zero_division=0),
            recall=recall_score(labels,accepted),f1=f1_score(labels,accepted),auroc=roc_auc_score(labels,scores)))

def run(root,arm,resume=False):
    root=Path(root);out=root/'training'/arm;out.mkdir(parents=True,exist_ok=True)
    if torch.cuda.device_count()!=1:raise ValueError('one assigned GPU required')
    torch.set_num_threads(2);prior.runner._set_determinism(260914)
    dataset=GroupCache(root,'train');gv=GroupCache(root,'val');sv=data.FormalCache(VAL_CACHE,'val')
    with (root/'cache'/'.commit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for view in (dataset,gv):
            p=read(view.root/'protocol.json')
            if p['status']!='complete':
                p.update(status='complete',completed_pairs=len(view),precompute_device='two assigned CUDA GPUs',
                    precision='fp32',microbatch=8,contour_padding='same zero-coordinate/false-valid padding to512 as original collate')
                save(view.root/'protocol.json',p)
    net=make('reference512_h4').to('cuda:0');opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
    identity=dict(schema=SCHEMA,arm=arm,model=net.metadata(),matcher_sha=cache.SOURCE_SHA,matcher_frozen=True,
        train=dataset.binding,group_val=gv.binding,clean_val=sv.binding,seed=260914,data_seed=SEED,
        epochs=EPOCHS,group_size=4,group_batch=12,physical_microbatch=48,effective_batch=48,accumulation=1,
        pair_exposures_per_epoch=len(dataset),positive_exposures_per_epoch=12000,negative_exposures_per_epoch=36000,
        loss='0.5 pos BCE + 0.5 mean neg BCE'+(' + 0.5 group CE(logits,T=1)' if arm==ARMS[1] else ''),
        code_sha={n:data.sha(Path(__file__).with_name(n)) for n in ('train.py','support.py')},
        selection='fixed C16; C8 also retained; never use REAL/OOD for epoch selection')
    completed=0;last=out/'last.pt'
    if last.exists():
        if not resume:raise ValueError('existing checkpoint requires --resume')
        checkpoint=torch.load(last,map_location='cpu',weights_only=False)
        if checkpoint['identity']!=identity:raise ValueError('resume identity differs')
        net.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        prior.restore_rng_state(checkpoint['rng']);completed=checkpoint['completed_segments']
    save(out/'protocol.json',identity);prior.BATCH=48;start=time.time()
    save(out/'status.json',dict(status='training',arm=arm,completed_segments=completed,
        actual_pair_batch=48,groups_per_update=12,matcher_frozen=True))
    print(json.dumps(dict(event='formal_training_started',arm=arm,actual_pair_batch=48,
                          groups_per_update=12,completed_segments=completed)),flush=True)
    for epoch in range(1,EPOCHS+1):
        if epoch*4<=completed:continue
        rng=np.random.default_rng(SEED+epoch);order=rng.permutation(len(dataset.groups))
        for param in opt.param_groups:param['lr']=prior.learning_rate(epoch)
        for segment in range(4):
            number=(epoch-1)*4+segment+1
            if number<=completed:continue
            net.train();began=time.time();sums=np.zeros(4);count=0
            groups=order[segment*3000:(segment+1)*3000];torch.cuda.reset_peak_memory_stats()
            for offset in range(0,3000,GROUP_BATCH):
                ids=dataset.groups[groups[offset:offset+GROUP_BATCH]].ravel();batch=dataset.batch(ids,'cuda:0')
                opt.zero_grad(set_to_none=True);o=net(*batch.model_args,**batch.model_kwargs)
                loss,bce,rank=group_loss(o.logit,batch.labels,batch.training_valid,arm)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
                sums[:3]+=np.array([loss.item(),bce.item(),rank.item()]);sums[3]+=float(o.used_fallback.float().mean());count+=1
            report=dict(epoch=epoch,segment=segment+1,groups=3000,pairs=12000,updates=count,
                loss=float(sums[0]/count),pair_bce=float(sums[1]/count),group_ce=float(sums[2]/count),
                fallback_fraction=float(sums[3]/count),seconds=time.time()-began,microbatch=48,effective_batch=48,
                peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
            save(out/f'segment_{number:03d}.json',report)
            if segment==3:
                val,rows=prior.evaluate(net,sv,torch.device('cuda:0'))
                save(out/f'validation_{epoch:03d}.json',val)
                if epoch in (8,16):save(out/f'validation_{epoch:03d}_rows.json',rows)
                save(out/f'group_validation_{epoch:03d}.json',group_eval(net,gv))
            checkpoint=dict(schema=SCHEMA,identity=identity,model={k:v.detach().cpu() for k,v in net.state_dict().items()},
                optimizer=opt.state_dict(),rng=prior.capture_rng_state(),completed_segments=number,matcher_updated=False)
            prior.runner._atomic_torch_save(last,checkpoint)
            if segment==3 and epoch in (8,16):prior.runner._atomic_torch_save(out/f'head_epoch_{epoch:03d}.pt',checkpoint)
            completed=number;save(out/'status.json',dict(status='training',arm=arm,completed_segments=number,
                epoch=epoch,seconds=time.time()-start,last_segment=report))
            print(json.dumps(dict(event='segment_complete',arm=arm,**report)),flush=True)
    head=out/'head_epoch_016.pt';val=read(out/'validation_016.json')
    save(out/'freeze.json',dict(schema=SCHEMA,status='complete',identity=identity,checkpoint=head.name,
        checkpoint_sha256=data.sha(head),operating_points=val['operating_points'],real_ood_used_for_fit=False))
    save(out/'status.json',dict(status='training_complete',arm=arm,epochs=16,updates=16000,
        training_pair_exposures=768000,matcher_updated=False,seconds=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:run(a.root,a.arm,a.resume)
    except BaseException as e:
        save(Path(a.root)/'training'/a.arm/'failure.json',dict(error=repr(e),recovery='resume last completed segment'));raise
