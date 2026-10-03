"""Real immutable full heldout identity/batch audit; no model inference/GPU."""
import argparse
import os
from pathlib import Path

from common import api, read, bound, save, require, check_preparation, source_map


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('spec','preparation','repair-preparation','canonical-straight','case-plan','out'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not args.out.exists(),'exclusive CPU preflight required')
    check_preparation(args.repair_preparation);before=source_map()
    import torch
    import entry
    inputs=api('matcher_v2_v1.runtime_inputs').load_inputs(read(args.spec));source=inputs['source']
    entry.check_preparation(args.preparation,source)
    pop=api('matcher_v2_v1.population')
    plan=pop.freeze_plan(args.spec,bound(args.canonical_straight),args.case_plan,source)
    pop.validate_plan(plan,args.spec,source)
    records=[]
    for split in pop.STRAIGHT:
        meta,batches,data_source,dataset=entry.load_population(split,plan,source)
        # Constructor has recomputed canonical hashes over every actual NPZ;
        # also exercise original inherited item loading + unchanged collator.
        items,batch=next(batches)
        require(len(meta['pairs'])==len(dataset)==900 and len(items)==8,'full heldout/batch identity differs')
        require([r['pair_id'] for r in items]==[r['pair_id'] for r in meta['pairs'][:8]],'batch membership changed')
        shapes={k:list(batch[k].shape) for k in pop.INPUTS}
        require(all(shape[0]==8 for shape in shapes.values()),'full eight-example official batch required')
        records.append(dict(split=split,pairs=len(dataset),canonical_rows_sha256=data_source['canonical_rows_sha256'],
            first_batch_shapes=shapes,actual_inputs_and_targets_hash_verified=True,
            no_inference=True,no_selection=True))
    require(not torch.cuda.is_initialized() and before==source_map(),'preflight changed sources or initialized CUDA')
    save(args.out,dict(schema='matcher-v2-terminal-repair-live-preflight/1',status='passed',
        execution=bound(args.spec),repair_preparation=bound(args.repair_preparation),records=records,
        cuda_initialized=False,model_inference_performed=False,source_files_unchanged=True))
    print('Complete SELECT/TEST identity and official batch preflight passed; no inference or CUDA')


if __name__=='__main__':main()
