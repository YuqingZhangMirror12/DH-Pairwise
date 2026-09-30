"""Bounded 8-pair real-shape gradient/isolation and memory check, not training."""
import argparse
import torch
from .independent_features import from_checkpoint, ScorerTrainingSystem
from .data import Dataset,collate,to_device,INPUTS
from .model import SeamContextModel
from .prepare import save


def run(a):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    ds=Dataset(a.data+'/train.json')
    # Mix both labels and long/short recipes; take reproducible spaced samples.
    indices=(0,1,3000,6000,9000,12000,16000,20000,1000,4000,7000,10000,13000,17000,21000,23000)
    batch=to_device(collate([ds[i] for i in indices[:a.microbatch]]),'cuda')
    report={};reference=None
    for arm in ('frozen_features','independent_features'):
        torch.cuda.reset_peak_memory_stats();m=from_checkpoint(a.base,arm).cuda()
        with torch.no_grad():
            o=m(*(batch[k] for k in INPUTS),decode=True,verify=True)
            scores=torch.stack([v.score for v in o.verified]).cpu()
            trans=[v.translations.detach().cpu() for v in o.verified]
            q=o.ot1.real_transport.cpu()
            if reference is None:reference=(scores,trans,q)
            else:
                torch.testing.assert_close(scores,reference[0],atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(q,reference[2],atol=0,rtol=0)
                for x,y in zip(trans,reference[1]):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
        del o
        if arm=='independent_features':
            for part in ('patch_encoder','scale_gate','arc_context'):
                original=dict(getattr(m,part).named_parameters());copy=dict(getattr(m.scorer_features,part).named_parameters())
                for key in original:assert original[key].data_ptr()!=copy[key].data_ptr()
        frozen={k:v.detach().clone() for k,v in m.named_parameters() if not v.requires_grad}
        sys=ScorerTrainingSystem(m);sys.train();loss,parts,counts=sys(batch);loss.backward()
        assert torch.isfinite(loss)
        bad=[k for k,p in m.named_parameters() if not p.requires_grad and p.grad is not None]
        assert not bad,bad
        grad={}
        for part in ('verifier','scorer_features.patch_encoder','scorer_features.arc_context'):
            pp=[p for k,p in m.named_parameters() if k.startswith(part) and p.grad is not None]
            if pp:
                norm=sum(float(p.grad.square().sum()) for p in pp)**.5;assert norm>0 and all(torch.isfinite(p.grad).all() for p in pp)
                grad[part]=norm
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5)
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],5.)
        opt.step()
        assert all(torch.equal(p,dict(m.named_parameters())[k]) for k,p in frozen.items())
        report[arm]=dict(loss=float(loss),counts=counts,gradient_norms=grad,matcher_unchanged=True,
            peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
            parameters=sum(p.numel() for p in m.parameters()),trainable=sum(p.numel() for p in m.parameters() if p.requires_grad))
        del sys,m,opt,loss,parts,frozen,pp,q,trans,scores;torch.cuda.empty_cache()
    save(a.out,dict(passed=True,microbatch=a.microbatch,training_started=False,initial_output_parity=True,arms=report))
    print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--microbatch',type=int,choices=(8,16),default=8)
    run(p.parse_args())
