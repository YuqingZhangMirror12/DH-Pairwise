"""Explicit, data-free CPU/CUDA parity gate for the external Attention recorder.

No dataset/checkpoint/experiment root is accepted or opened. This is not a
training or evaluation launcher. A CUDA run must be scheduled by the caller
only after checking that its chosen GPU has been released by training.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn

from .attention_trace import AttentionTrace
from .audit import validate_arrays
from .snapshot import snapshot_prediction
from .frozen import TrainingConfig, registered_protocol
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1 import consensus_head, model as model_module
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evidence import PairEvidence
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.geometry import compact_contour


def state_digest(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(str((tuple(value.shape),value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def assert_equal(expected,actual,label):
    if (not torch.isfinite(expected).all() or not torch.isfinite(actual).all()
            or not torch.equal(expected,actual)):
        raise ValueError('capture changed or invalidated '+label)


def synthetic_pair(device,empty=False):
    a=torch.tensor([[0.,0.],[0.,8.],[5.,8.],[10.,8.],[10.,0.]],device=device)
    b=torch.tensor([[0.,13.],[0.,20.],[10.,20.],[10.,13.],[5.,13.]],device=device)
    valid=torch.ones(1,5,dtype=torch.bool,device=device)
    ga,gb=compact_contour(a[None],valid),compact_contour(b[None],valid)
    q=torch.zeros(5,5,device=device)
    if not empty: q[2,4]=.12; q[2,3]=.4
    features=[torch.randn(5,96,device=device) for _ in range(4)]
    ids=torch.arange(5,device=device)
    return PairEvidence(*features,q,1-q.sum(1),1-q.sum(0),ga,gb,ids,ids)


def rng_state(device):
    return [torch.get_rng_state().clone()]+(
        [torch.cuda.get_rng_state(device).clone()] if device.type=='cuda' else [])


def check_rng(before,device):
    if not all(torch.equal(a,b) for a,b in zip(before,rng_state(device))):
        raise ValueError('capture changed RNG state')


def numeric_settings():
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
        tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        float32_matmul_precision=torch.get_float32_matmul_precision())


@contextmanager
def numeric_protocol():
    """Match frozen evaluate.run FP32 settings, restore the caller afterwards."""
    before=numeric_settings()
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        yield numeric_settings()
    finally:
        torch.use_deterministic_algorithms(before['deterministic_algorithms'],
            warn_only=before['deterministic_warn_only'])
        torch.set_float32_matmul_precision(before['float32_matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32=before['tf32_matmul']
        torch.backends.cudnn.allow_tf32=before['tf32_cudnn']
        torch.backends.cudnn.benchmark=before['cudnn_benchmark']
        torch.backends.cudnn.deterministic=before['cudnn_deterministic']


@torch.no_grad()
def run(device_name):
    device=torch.device(device_name)
    if device.type not in ('cpu','cuda') or (device.type=='cuda' and device.index is None):
        raise ValueError('use cpu or an explicitly indexed cuda:N device')
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('requested CUDA unavailable; do not silently fall back to CPU')
    if device.type=='cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting the CUDA gate')
    devices=[] if device.type=='cpu' else [device.index]
    # Restore caller RNG; this synthetic gate does not borrow trained weights.
    with numeric_protocol() as executed_numeric_settings,torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(26092611)
        if device.type=='cuda':
            with torch.cuda.device(device): torch.cuda.manual_seed(26092611)
        operator=nn.ModuleDict({'attention':consensus_head.MeasureAttention()}).to(device).eval()
        identity=state_digest(operator)
        cases=[]
        for batch,n,m in ((1,3,2),(3,31,47),(2,129,193),(1,0,3),(1,3,0)):
            q=torch.randn(batch,n,96,device=device); k=torch.randn(batch,m,96,device=device)
            qv=torch.ones(batch,n,dtype=torch.bool,device=device)
            kv=torch.ones(batch,m,dtype=torch.bool,device=device)
            if n: qv[:,-1]=False
            if batch>1: kv[-1]=False
            measure=torch.exp(-torch.rand(batch,m,device=device)*40.)
            bias=-torch.rand(batch,n,m,device=device)*30.
            expected=operator['attention'](q,k,qv,kv,measure,bias)
            rng=rng_state(device)
            with AttentionTrace(operator) as trace:
                actual=operator['attention'](q,k,qv,kv,measure,bias)
            assert_equal(expected,actual,'operator output'); check_rng(rng,device)
            if len(trace.records)!=1: raise ValueError('wrong operator call count')
            record=trace.records[0]
            row_target=(qv & kv.any(-1)[:,None]).float().cpu()
            if not torch.allclose(record['effective_head_mean'].sum(-1),row_target,rtol=0,atol=2e-6):
                raise ValueError('effective Attention row mass differs')
            cases.append(dict(batch=batch,queries=n,keys=m,bitwise_output_equal=True,
                reconstruction_max_abs=record['reconstruction_max_abs']))
        if state_digest(operator)!=identity: raise ValueError('operator parameters changed')

        model=model_module.S7Consensus(None,CompatibilityConfig(.5,.5,.5,.5,1.,30.),
            head=consensus_head.ConsensusEvidenceHead()).to(device).eval()
        identity=state_digest(model)
        protocol=registered_protocol(TrainingConfig().record())
        pairs=[]
        for empty in (False,True):
            pair=synthetic_pair(device,empty)
            expected=model.score_pair(pair,capture_diagnostics=True)
            rng=rng_state(device)
            with AttentionTrace(model.head) as trace:
                actual=model.score_pair(pair,capture_diagnostics=True)
            check_rng(rng,device)
            if (expected.selected_cluster_id!=actual.selected_cluster_id or
                    len(expected.clusters)!=len(actual.clusters)):
                raise ValueError('capture changed candidate identity')
            assert_equal(expected.score,actual.score,'final score')
            if actual.has_candidate:
                assert_equal(expected.translation_a_to_b_rc,actual.translation_a_to_b_rc,'final translation')
            for lhs,rhs in zip(expected.clusters,actual.clusters):
                for stage in ('initial_encoded','encoded'):
                    for side in ('a','b'):
                        left=getattr(getattr(lhs,stage),side);right=getattr(getattr(rhs,stage),side)
                        for attr in ('state','local_logits','local_probabilities','localization_reliability'):
                            assert_equal(getattr(left,attr),getattr(right,attr),stage+'/'+side+'/'+attr)
            meta,arrays=snapshot_prediction('SYNTHETIC-ONLY',pair,actual,threshold=.5,
                provenance=dict(variant=protocol['variant'],evidence_mode=protocol['evidence_mode']),
                attention_trace=trace)
            validate_arrays(meta,arrays)
            pairs.append(dict(empty_q=empty,candidates=len(actual.clusters),attention_calls=len(trace.records),
                production_head_feature_dim=96,bitwise_outputs_equal=True,audit='passed'))
        if state_digest(model)!=identity: raise ValueError('model parameters changed')
        if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
            raise ValueError('capture left hooks installed')
    files=[Path(consensus_head.__file__),Path(model_module.__file__),Path(__file__).with_name('attention_trace.py')]
    return dict(status='passed',verified_at_utc=datetime.now(timezone.utc).isoformat(),
        device=str(device),torch_version=torch.__version__,protocol=protocol,
        device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        numeric_settings_executed=executed_numeric_settings,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        operators=cases,synthetic_model_cases=pairs,parameters_unchanged=True,rng_unchanged=True,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        trained_checkpoint_or_dataset_opened=False,training_started=False,
        limitation='Synthetic numerical parity only; no trained-model, real-data or quality claim')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    out=Path(args.out)
    if out.exists(): raise FileExistsError('refuse to overwrite gate receipt')
    result=run(args.device)
    # Use exclusive creation; the directory must already be chosen by caller.
    with out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],device=result['device'],receipt=str(out))))


if __name__=='__main__': main()
