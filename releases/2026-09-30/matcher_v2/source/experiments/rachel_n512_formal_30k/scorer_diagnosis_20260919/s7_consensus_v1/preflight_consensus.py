"""Isolated actual-TRAIN forward/backward gates; discard every updated weight."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import collate_rachel_pairs
from . import matcher as matcher_module
from .compatibility import CompatibilityConfig
from .evidence import PairEvidence
from .losses import batch_loss
from .matcher import INPUTS, S7MatcherAdapter
from .model import S7Consensus
from .preflight_matcher import digest, state_digest
from .targets import PairLabels


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def run(args):
    out=Path(args.output)
    if out.exists():
        raise ValueError('do not overwrite earlier preflight')
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(26092403)
    device=torch.device(args.device)
    adapter=S7MatcherAdapter.from_s7_m12(args.checkpoint).to(device)
    calibration=json.loads(Path(args.calibration).read_text())
    config=CompatibilityConfig.from_calibration(calibration)
    model=S7Consensus(adapter,config).to(device).train()
    before=state_digest(adapter)
    head_before=state_digest(model.head)
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=1e-4)
    manifest=json.loads(Path(args.manifest).read_text())
    artifact_root=Path(manifest.get('artifact_root',Path(args.manifest).parent))
    no_update=bool(getattr(args,'no_update',False))
    seen=set();entries=[]
    for entry in manifest['entries']:
        key=(entry.get('recipe',entry.get('s7_recipe')),int(entry['label']))
        if key not in seen:
            seen.add(key);entries.append(entry)
    result=dict(status='running',scope='actual-TRAIN implementation gates, NOT formal training',
        checkpoint_sha256=digest(args.checkpoint),manifest_sha256=digest(args.manifest),
        calibration_sha256=digest(args.calibration),device=str(device),
        batch_size=args.batch_size,config=asdict(config),batches=[],optimizer_updates=0,
        no_optimizer_update_requested=no_update,
        updated_weights_discarded=True,code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    def save():
        out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    save();begin=time.time()
    try:
        for start in range(0,len(entries),args.batch_size):
            selected=entries[start:start+args.batch_size]
            samples=[]
            for entry in selected:
                path=entry.get('sample_path') or str(artifact_root/entry['artifact_path'])
                sample,_=load_sample(path)
                if sample.pair_id!=entry['pair_id']:
                    raise AssertionError('identity mismatch')
                samples.append(sample)
            batch=collate_rachel_pairs(samples).as_dict()
            inputs=[torch.from_numpy(np.array(batch[k],copy=True)).to(device) for k in INPUTS]
            labels=[PairLabels.from_sample(s,e.get('recipe',e.get('s7_recipe')),device)
                    for s,e in zip(samples,selected)]
            if device.type=='cuda':
                torch.cuda.reset_peak_memory_stats(device)
            sync(device);t=time.time()
            with patch.object(matcher_module,'dustbin_sinkhorn',wraps=matcher_module.dustbin_sinkhorn) as sinkhorn:
                output=adapter(*inputs)
                if sinkhorn.call_count!=1:
                    raise AssertionError('not exactly one Sinkhorn per batched Matcher forward')
            pairs=[PairEvidence.from_matcher(output,i,inputs[0],inputs[1]) for i in range(len(samples))]
            sync(device);matcher_seconds=time.time()-t;t=time.time()
            proposals=[model.builder(pair) for pair in pairs]
            sync(device);proposal_seconds=time.time()-t;t=time.time()
            predictions=[model.score_pair(pair,proposals=p) for pair,p in zip(pairs,proposals)]
            flags=[bool(s.label) for s in samples]
            loss,items=batch_loss(model,predictions,labels,extension_flags=flags)
            sync(device);head_seconds=time.time()-t;t=time.time()
            if not torch.isfinite(loss):
                raise AssertionError('nonfinite loss')
            loss.backward()
            grads={n:float(p.grad.detach().norm()) for n,p in model.head.named_parameters() if p.grad is not None}
            if not grads or not all(np.isfinite(v) for v in grads.values()) or not any(v>0 for v in grads.values()):
                raise AssertionError('missing/nonfinite scorer gradients')
            if any(p.grad is not None for p in adapter.parameters()):
                raise AssertionError('frozen Matcher has gradients')
            for pred in predictions:
                for cluster in pred.clusters:
                    if cluster.translation is not cluster.encoded.evidence.pose:
                        raise AssertionError('final-pose score mismatch')
            with torch.no_grad():
                first=next((c for p in predictions for c in p.clusters),None)
                if first is not None:
                    dense=model.head.dense_reference(first.encoded.evidence)
                    penalty=first.overlap['fraction_min_area'] if first.overlap['available'] else 0.
                    reference=model.head.readout(dense,penalty).logit
                    torch.testing.assert_close(reference,first.readout.logit,atol=2e-5,rtol=2e-5)
            torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.)
            if not no_update:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sync(device);backward_seconds=time.time()-t
            receipt=dict(pair_ids=[s.pair_id for s in samples],labels=[int(s.label) for s in samples],
                recipes=[e.get('recipe',e.get('s7_recipe')) for e in selected],
                matcher_seconds=matcher_seconds,proposal_seconds=proposal_seconds,
                head_and_loss_seconds=head_seconds,backward_seconds=backward_seconds,
                loss=float(loss.detach()),grad_norms=grads,
                max_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                max_reserved_mb=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None,
                candidates=[len(p.clusters) for p in predictions],
                nonzero_queries=[[(int(c.encoded.evidence.a.valid.sum()),int(c.encoded.evidence.b.valid.sum()))
                                  for c in p.clusters] for p in predictions],
                proposals_merged=[len(p.merge_trace) for p in proposals],
                counts=[x.counts for x in items],sinkhorn_calls=1)
            result['batches'].append(receipt);result['optimizer_updates']+=int(not no_update)
            save()
            print(json.dumps({k:receipt[k] for k in ('candidates','loss','max_allocated_mb','matcher_seconds','proposal_seconds','head_and_loss_seconds','backward_seconds')},allow_nan=False),flush=True)
            del predictions,proposals,pairs,output,loss,items,inputs,batch,labels
            gc.collect()
        result.update(status='passed',elapsed_seconds=time.time()-begin,
            matcher_unchanged=state_digest(adapter)==before,
            scorer_parameters_changed=state_digest(model.head)!=head_before)
        if (not result['matcher_unchanged']
                or result['scorer_parameters_changed']==no_update):
            raise AssertionError('parameter change contract failed')
        save()
    except Exception as error:
        result.update(status='failed',elapsed_seconds=time.time()-begin,
            failure_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        save()
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','manifest','calibration','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--no-update',action='store_true',help='check actual TRAIN gradients without changing model weights')
    run(parser.parse_args())
