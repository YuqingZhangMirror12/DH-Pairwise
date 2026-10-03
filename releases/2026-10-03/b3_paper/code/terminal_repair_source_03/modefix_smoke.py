"""Bounded actual-export, full800/N512 inference check; no training/selection."""
import argparse
import os
from pathlib import Path
import torch
from common import api, read, bound, save, require, check_preparation, source_map
import entry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('spec','preparation','repair-preparation','controller-root','population-plan','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=('cuda:0','cpu'),required=True)
    args=p.parse_args()
    require(not args.out.exists(),'exclusive smoke receipt required')
    check_preparation(args.repair_preparation); sources=source_map()
    inputs=entry.load_inputs(read(args.spec)); source=inputs['source']
    entry.check_preparation(args.preparation,source)
    tree_sha=api('curriculum_training_v1.checkpoint_io').tree_sha
    predict=api('curriculum_training_v1.matcher_evaluation').predict_batch
    plan=read(args.population_plan);entry.population.validate_plan(plan,args.spec,source)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    meta,batches,_,_=entry.load_population('sim_select',plan,source)
    items,batch=next(batches);ids=[r['pair_id'] for r in items]
    names=api('s7_consensus_v1.matcher').INPUTS
    values={k:batch[k].to(args.device) for k in names}
    require(len(ids)==8 and values['mask_a'].shape[-1]==800 and values['points_rc_a'].shape[1]==512,
            'official full-size eight-pair batch required')
    records=[]
    for choice in ('sim_best','equal_budget_endpoint'):
        saved,origin=entry.verified_export(args.controller_root,args.spec,inputs['plan'],choice)
        old,geometry,_=entry.original.load_model(saved,origin,source)
        require(old.training and old.frozen and not any(m.training for m in list(old.modules())[1:])
                and not any(p.requires_grad for p in old.parameters()),'actual diagnosed root-flag defect changed')
        old.to(args.device)
        try:
            predict(old,geometry,source,values,ids)
        except ValueError as error:
            require(str(error)=='frozen eval Matcher required','unexpected original inference failure')
        else:
            raise ValueError('original actual-export failure no longer reproduced')
        expected=tree_sha(old.state_dict());old.eval()
        with torch.no_grad():reference=old(**values)
        model,geometry,_=entry.load_model(saved,origin,source);model.to(args.device)
        with torch.no_grad():actual=model(**values)
        require(torch.equal(reference.assignment,actual.assignment)
                and torch.equal(reference.affinity,actual.affinity), 'mode repair changed affinity or Sinkhorn Q')
        require(bool(actual.numeric_valid.all()) and bool(torch.isfinite(actual.assignment).all()),
                'actual repaired inference is numerically invalid')
        rows=predict(model,geometry,source,values,ids)
        require(len(rows)==8 and tree_sha(model.state_dict())==expected,'prediction/weights changed')
        records.append(dict(selection=choice,checkpoint_sha256=origin['checkpoint_sha256'],
            updates=origin['updates'],training_success_verified=True,old_failure_reproduced=True,
            root_training_after=model.training,parameters_trainable=False,model_unchanged=True,
            same_affinity_and_q_as_explicit_eval=True,actual_pairs=len(rows),
            candidates=[r['retained_count'] for r in rows]))
        del old,model,reference,actual
    require(sources==source_map(),'smoke changed repair sources')
    save(args.out,dict(schema='matcher-v2-root-eval-mode-smoke/1',status='passed',
        execution=bound(args.spec),preparation=bound(args.preparation),repair_preparation=bound(args.repair_preparation),
        source_files=sources,records=records,device=args.device,training_performed=False,
        performance_claim=False,gt_used_for_prediction=False,weights_modified=False))
    print('Actual selected/endpoint weights: original error reproduced, repaired native inference passed; no training')


if __name__=='__main__':main()
