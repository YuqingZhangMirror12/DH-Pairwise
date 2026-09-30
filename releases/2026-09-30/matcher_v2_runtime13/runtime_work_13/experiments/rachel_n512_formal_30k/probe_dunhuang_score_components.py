"""Frozen Dunhuang score/evidence distributions, with explicit reviewed membership.

No training, threshold fitting, candidate-policy change, or REAL-based selection.
Historical E1 and Full24 are separate frozen checkpoints. GT is attached only
after the input-only model forward; review status is never an input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

from experiments.rachel_n512_formal_30k.probe_local_score_domains import (
    Probe, FIELDS, save, sealed, RachelPairDataset, make_ablation_loader,
    load_winner, np, torch)
from experiments.rachel_n512_formal_30k.run_real_contiguous_seam_ablation import load_prepared_cache
from experiments.rachel_n512_formal_30k.run_real_layout_decoder_experiment import input_batches
from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint


def run(args):
    dest=Path(args.output); dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); sealed._set_determinism(260912)
    if args.kind=='full24':
        model,identity,thresholds=load_winner(args.training_run)
    else:
        path=Path(args.training_run)/'winner.pt'
        ckpt=sealed._torch_load_checkpoint(path)
        model=load_rachel_checkpoint(ckpt)
        protocol=json.loads(Path(args.original_protocol).read_text())
        sha=sealed._sha256_file(path)
        if sha!=protocol['checkpoint_sha256']:
            raise ValueError('Historical checkpoint differs from original evaluated checkpoint')
        identity=protocol; thresholds=protocol['branch_validation_thresholds']
    device=torch.device('cuda:0'); model=model.to(device).eval().requires_grad_(False)
    p=Probe(model,device,dest)
    metadata,arrays=load_prepared_cache(args.prepared)
    keep=set(json.loads(Path(args.keep_ids).read_text())['kept_positive_pair_ids'])
    real_by_id={r['pair_id']:r for r in metadata['pairs']}
    positives={r['pair_id'] for r in metadata['pairs'] if r['label']}
    if len(keep)!=295 or not keep<=positives: raise ValueError('Review membership mismatch')
    protocol=dict(kind=args.kind,model=identity,thresholds=thresholds,status='running',training=False,
        input_cache=args.prepared,real_pairs=len(metadata['pairs']),real_positive=len(positives),
        kept_positive=len(keep),sim_selection='128 positive/64 negative in TEST, sha256(pair_id) order',
        canonical_results_changed=False,review_used_for_model_input=False,layout_threshold_px=20,
        original_protocol=args.original_protocol,counterfactuals_are_diagnostics_not_new_models=True)
    save(dest/'protocol.json',protocol); start=time.monotonic()
    for batch in input_batches(metadata,arrays,8):
        p.batch('dunhuang',batch.pair_ids,[getattr(batch,n) for n in FIELDS])
    # Open GT after every real prediction, rather than using it to choose candidates.
    gt={r['pair_id']:r for r in json.loads(Path(args.real_gt).read_text())['positive_pairs']}
    if set(gt)!=positives: raise ValueError('GT membership mismatch')
    for r in p.rows:
        source=real_by_id[r['pair_id']]
        r.update(label=bool(source['label']),strict_member=bool(source['strict']),
                 case_cluster=source['case_cluster'],fragment_a=source['fragment_a_id'],
                 fragment_b=source['fragment_b_id'],
                 review_status=('keep' if r['pair_id'] in keep else 'exclude') if source['label'] else 'not_reviewed_negative')
        target=gt.get(r['pair_id']); t=None
        if target:
            if target['fragment_a_token']!=source['fragment_a_id'] or target['fragment_b_token']!=source['fragment_b_id']:
                raise ValueError('GT endpoint mismatch')
            t=np.asarray(target['translation_gt_a_to_b_rc'],float)
        r['target_translation_rc']=t.tolist() if t is not None else None
        r['layout']['error_px']=float(np.linalg.norm(np.asarray(r['layout']['translation_rc'])-t)) if t is not None and r['layout']['valid'] else None
    manifest=[json.loads(x) for x in (Path(args.dataset)/'pairs/test.jsonl').read_text().splitlines() if x]
    key=lambda i:hashlib.sha256(manifest[i]['pair_id'].encode()).hexdigest()
    ix=[]
    for positive,n in [(True,128),(False,64)]:
        ix+=sorted([i for i,r in enumerate(manifest) if bool(r['label'])==positive],key=key)[:n]
    ds=RachelPairDataset(args.dataset,'test')
    for batch in make_ablation_loader(ds,ix,batch_size=8,num_workers=2,seed=260912,contour_cap=512):
        targets=[batch.translation_a_to_b_rc[i] if batch.translation_valid[i] else None for i in range(len(batch.pair_ids))]
        p.batch('sim_test',batch.pair_ids,[getattr(batch,n) for n in FIELDS],batch.labels,targets)
    p.counterfactual()
    save(dest/'results.json',dict(rows=p.rows))
    protocol.update(status='complete',rows=len(p.rows),elapsed_seconds=time.monotonic()-start)
    save(dest/'protocol.json',protocol)
    print(json.dumps(dict(status='complete',kind=args.kind,rows=len(p.rows),elapsed=protocol['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--kind',choices=['full24','historical_e1'],required=True)
    for k in ('training-run','original-protocol','prepared','real-gt','keep-ids','dataset','output'):
        a.add_argument('--'+k,required=True)
    run(a.parse_args())
