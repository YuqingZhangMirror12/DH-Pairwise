"""Two disposable full-batch optimizer steps; never counted as formal epochs."""
import argparse
import json
import os
import time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from ..matched_only import data as cached, train as prior
from .model import make
from .data import TrainingEvidence
from .train import BATCH, training_loss
from .runtime import lease


def run(args):
    torch.set_num_threads(2)
    prior.runner._set_determinism(260914)
    if args.arm=="gcn_shredding_h4":
        torch.use_deterministic_algorithms(True,warn_only=True)
    source=cached.FormalCache(args.train_cache,"train")
    evidence=TrainingEvidence(args.train_diagnostics,source) if args.arm=="joint_D_h4" else None
    if evidence:
        _,ledger=evidence.order(1)
        print(json.dumps(dict(event="D_sampler",summary=evidence.summary,ledger=ledger)),flush=True)
    model=make(args.arm).to("cuda:0").train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    start=time.perf_counter()
    for step in range(2):
        ids=range(step*BATCH,(step+1)*BATCH)
        batch=source.batch(ids,"cuda:0")
        optimizer.zero_grad(set_to_none=True)
        result=model(*batch.model_args,**batch.model_kwargs)
        loss,_,_=training_loss(result,batch,evidence,ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("smoke loss")
    torch.cuda.synchronize()
    print(json.dumps(dict(status="smoke_passed",arm=args.arm,batch=BATCH,steps=2,
        seconds=time.perf_counter()-start,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,loss=float(loss),counted_as_training=False)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--arm",required=True)
    p.add_argument("--train-cache",required=True)
    p.add_argument("--train-diagnostics",required=True)
    p.add_argument("--gpu-uuid",required=True)
    p.add_argument("--lock-root",required=True)
    args=p.parse_args()
    with lease(args.gpu_uuid,args.lock_root):
        run(args)
