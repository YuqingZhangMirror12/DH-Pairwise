"""Explicit four-GPU execution continuation of the sealed edge_multi trainer.

Only train_segment uses a Tensor-only DataParallel adapter. Original base,
single optimizer, global16 loss/clip, epoch order, validation and endpoints stay.
Import has no CUDA/process effects. No forced device-count or fabricated proof.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import device_runtime as devices


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def gpu_list(value):
    values=value.split(',')
    if len(values)!=4 or len(set(values))!=4 or any(not devices.UUID_RE.fullmatch(x) for x in values):
        raise ValueError('exactly four distinct complete GPU UUIDs required')
    return values


@contextmanager
def leases(uuids, root):
    """Adopt parent-owned open descriptions or take new locks for a smoke."""
    root=Path(root).resolve()
    inherited=os.environ.get('RACHEL_DP_LEASE_FDS_JSON')
    owned=[]
    if inherited:
        rows=json.loads(inherited)
        if [x['uuid'] for x in rows]!=uuids:
            raise ValueError('inherited GPU lease order differs from visible device order')
        for row in rows:
            expected=root/(row['uuid']+'.lock')
            fd=row['fd']
            if type(fd)!=int or fd<3 or Path(row['path']).resolve()!=expected:
                raise ValueError('invalid inherited lease descriptor')
            actual=os.fstat(fd); target=expected.stat()
            if (actual.st_dev,actual.st_ino)!=(target.st_dev,target.st_ino):
                raise ValueError('inherited lease inode differs')
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    else:
        root.mkdir(parents=True,exist_ok=True)
        try:
            for uuid in sorted(uuids):
                handle=(root/(uuid+'.lock')).open('a+')
                owned.append(handle)
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:
            for handle in owned:
                handle.close()
            raise
    try:
        inventory={row['uuid'] for row in devices.inventory()}
        if not set(uuids)<=inventory:
            raise ValueError('GPU UUID missing from actual inventory')
        for uuid in uuids:
            occupied=devices.foreign_pids(uuid)
            if occupied:
                raise RuntimeError('GPU occupied before four-card entry: '+str((uuid,occupied)))
        yield
    finally:
        # Inherited locks belong to the parent open description; never unlock
        # them here while that parent is recording child completion.
        for handle in owned:
            handle.close()


def run(args):
    uuids=gpu_list(args.gpus)
    receipt=Path(args.receipt).resolve()
    if receipt.exists():
        raise ValueError('unique execution receipt required; no automatic retry')
    original_args=list(args.arguments)
    if original_args[:1]==['--']:
        original_args=original_args[1:]
    if args.module!='matched_only.train':
        raise ValueError('only sealed matched_only.train supported')
    record=dict(schema='edge-multi-four-gpu-runtime/1',status='starting',mode=args.mode,
        pid=os.getpid(),gpus=uuids,global_batch=16,per_gpu_batch=4,
        device_count=4,precision='fp32',optimizer_count=1,learning_rate_unchanged=True,
        layer_count_unchanged=True,loss_unchanged=True,training_budget_unchanged=True,
        validation='original base on source cuda:0, original order and thresholds',
        numerical_equivalence='same objective; GPU reductions are not bitwise equivalent',
        cuda_rng_policy='legacy logical source RNG retained; full four-device RNG also saved',
        started_unix=time.time(),runtime_sha256=sha(__file__),module=args.module)
    save(receipt,record)
    try:
        with leases(uuids,args.lock_root):
            os.environ.update(CUDA_VISIBLE_DEVICES=','.join(uuids),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            import torch
            import multigpu_scorer_adapter as adapter
            train=importlib.import_module(args.module)
            if platform.system()!='Linux' or not torch.cuda.is_available() or torch.cuda.device_count()!=4:
                raise RuntimeError('remote Linux and four actual visible CUDA devices required')
            torch.set_num_threads(1)
            torch.set_default_dtype(torch.float32)
            train.runner._set_determinism(train.HEAD_SEED)
            inner=train.parser().parse_args(original_args)
            if inner.arm!='edge_multi' or inner.stop_after_head_epoch!=16:
                raise ValueError('only original edge_multi C16 budget')
            training=train.data.CandidateStageCache(train.data.FormalCache(inner.train_cache,'train'),
                inner.train_stage_cache,'edge_multi')
            validation=train.data.CandidateStageCache(train.data.FormalCache(inner.val_cache,'val'),
                inner.val_stage_cache,'edge_multi')
            base=train.make_scorer(inner.arm,seed=train.HEAD_SEED)
            ident=train.identity(inner.arm,base,training,validation)
            base=base.to(device='cuda:0',dtype=torch.float32)
            optimizer=train.create_optimizer(base)
            record.update(status='running',adapter_sha256=sha(adapter.__file__),
                          original_training_source_sha256=sha(train.__file__),
                          initial_protocol_identity=ident)
            save(receipt,record)
            if args.mode=='benchmark':
                if not args.checkpoint or not args.output:
                    raise ValueError('benchmark requires an immutable source snapshot and separate output')
                source=Path(args.checkpoint).resolve()
                source_sha=sha(source)
                loaded=torch.load(source,map_location='cpu',weights_only=False)
                completed,_=train.restore(base,optimizer,loaded,ident)
                if sha(source)!=source_sha:
                    raise ValueError('benchmark checkpoint changed during read')
                next_segment=train.plan()[completed]
                order=train.runner.epoch_indices(len(training),seed=train.DATA_SEED,
                    epoch=next_segment['absolute_epoch'],limit=None)
                offset=next_segment['offset']
                indices=order[offset:offset+16*4]
                result=adapter.benchmark(base,optimizer,training,indices,device_ids=(0,1,2,3))
                result.update(checkpoint_sha256=source_sha,completed_segments=completed,
                              source_checkpoint=str(source),formal_training_counted=False,
                              runtime_sha256=record['runtime_sha256'],adapter_sha256=record['adapter_sha256'])
                if Path(args.output).exists():
                    raise ValueError('benchmark output already exists')
                save(args.output,result)
                record.update(status='complete',benchmark_result=str(Path(args.output).resolve()),
                              finished_unix=time.time())
                save(receipt,record)
                return result

            if not inner.resume:
                raise ValueError('formal mode must continue the original checkpoint, not restart')
            root=Path(inner.output).resolve(strict=True)
            parallel=adapter.make_parallel(base,device_ids=(0,1,2,3))
            original_segment=train.train_segment
            original_payload=train.checkpoint_payload
            original_restore=train.restore
            original_save=train.save
            topology={k:v for k,v in record.items() if k not in ('initial_protocol_identity','status')}
            topology['runtime_receipt']=str(receipt)

            def payload(*a,**kw):
                p=original_payload(*a,**kw)
                p['parallel_cuda_rng_state']=p['rng_state']['cuda']
                p['rng_state']['cuda']=p['rng_state']['cuda'][:1]
                p['execution_topology']=topology
                return p

            def restore(*a,**kw):
                result=original_restore(*a,**kw)
                saved=a[2]
                full=saved.get('parallel_cuda_rng_state')
                if full is not None:
                    if len(full)!=4:
                        raise ValueError('parallel RNG checkpoint has wrong replica count')
                    torch.cuda.set_rng_state_all([v.cpu() for v in full])
                return result

            def annotated_save(path,value):
                if Path(path).name in ('protocol.json','status.json'):
                    value=dict(value,execution_topology=topology)
                return original_save(path,value)

            train.train_segment=lambda model,dataset,indices,opt,device: adapter.train_segment(
                model,dataset,indices,opt,device,parallel=parallel,device_ids=(0,1,2,3))
            train.checkpoint_payload=payload
            train.restore=restore
            train.save=annotated_save
            try:
                with train.run_lock(root):
                    loaded=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
                    record['resumed_from_segments']=loaded['completed_segments']
                    record['resume_checkpoint_sha256']=sha(root/'last.pt')
                    save(receipt,record)
                    result=train.execute(inner,base,optimizer,training,validation,ident,torch.device('cuda:0'))
            finally:
                train.train_segment=original_segment
                train.checkpoint_payload=original_payload
                train.restore=original_restore
                train.save=original_save
            if result.get('status')!='complete' or result.get('completed_segments')!=64:
                raise RuntimeError('registered full C16 training did not finish')
            record.update(status='complete',training_result=result,finished_unix=time.time())
        save(receipt,record)
        return result
    except BaseException as error:
        record.update(status='failed',error=repr(error),finished_unix=time.time())
        save(receipt,record)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',required=True)
    p.add_argument('--lock-root',required=True)
    p.add_argument('--receipt',required=True)
    p.add_argument('--mode',choices=('benchmark','resume'),required=True)
    p.add_argument('--module',default='matched_only.train')
    p.add_argument('--checkpoint')
    p.add_argument('--output')
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    return p


if __name__=='__main__':
    print(json.dumps(run(parser().parse_args()),ensure_ascii=False))
