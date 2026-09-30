"""Dedicated terminal native Matcher evaluation, no training/Scorer calls."""
import argparse
import os
from pathlib import Path
import random
import traceback

import numpy as np
import torch

from .checkpoint_io import file_sha, write_json
from .execution import load_inputs
from .matcher_population import (SPLITS, load_population, population_groups,
                                 targets_after_prediction, validate_plan)
from .matcher_run import run_population, verify_population
from .matcher_terminal import load_matcher, verified_export
from .model_adapter import require
from .runtime_io import read
from .verify_validation_preparation import bind_baseline


def check_preparation(path):
    receipt=read(path);root=Path(__file__).parent
    current={p.name:file_sha(p) for p in root.glob('*.py')}
    require(receipt.get('status')=='passed' and receipt.get('population_runner_tested') is True
            and receipt.get('failures')==receipt.get('errors')==receipt.get('skipped')==0
            and receipt.get('source_sha256')==current and receipt.get('gpu_used') is False,
            'this exact native population runner needs passed CPU preparation')
    return receipt


def run(args):
    for key in ('spec','population_plan','preparation','controller_root','out'):
        setattr(args,key,Path(getattr(args,key)).resolve())
    check_preparation(args.preparation)
    spec=read(args.spec); inputs=load_inputs(spec)
    require(inputs['plan'].record['module']=='matcher', 'native Matcher-only evaluation required')
    # Terminal is checked BEFORE any heldout loader is invoked.
    saved,origin=verified_export(args.controller_root,args.spec,inputs['plan'],args.order,args.selection)
    bind_baseline(inputs['baseline'])
    population_plan=read(args.population_plan);validate_plan(population_plan,args.spec,inputs['baseline'])
    matcher,geometry,origin=load_matcher(saved,origin,inputs['baseline'])
    require(matcher.base.config.canvas_size==800 and matcher.base.config.contour_cap==512,
            'formal evaluation requires the full800/N512 configuration')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    seed=inputs['plan'].record['seed']; random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device(args.device);matcher.to(device)
    meta,batches,source,dataset=load_population(args.split,population_plan,inputs['baseline'])
    roles=read(population_plan['real_split']['path']);cases=read(population_plan['case_plan']['path'])
    grouped=population_groups([dict(pair_id=p['pair_id']) for p in meta['pairs']],args.split,roles)
    groups={name:[p['pair_id'] for p in rows] for name,rows in grouped.items()}
    wanted={p['pair_id'] for p in cases['cases'] if p['split']==args.split}
    origin.update(split=args.split,population_plan_sha256=file_sha(args.population_plan),
        preparation_sha256=file_sha(args.preparation),source=source,
        evaluation_source_sha256={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')},
        real_split_sha256=population_plan['real_split']['sha256'],
        simulation_revision=population_plan['simulation_revision'])
    result=run_population(matcher,geometry,inputs['baseline'],meta,batches,source,out=args.out,
        provenance=origin,wanted_ids=wanted,groups=groups,device=device,
        targets_callback=lambda:targets_after_prediction(meta,args.split,population_plan,dataset))
    audit=verify_population(args.out);write_json(Path(args.out)/'independent_artifact_audit.json',audit)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('spec','population-plan','preparation','controller-root','out'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--order',choices=('curriculum','mixed'),required=True)
    parser.add_argument('--selection',choices=('sim_best','equal_budget_endpoint'),required=True)
    parser.add_argument('--split',choices=SPLITS,required=True)
    parser.add_argument('--device',choices=('cuda:0','cpu'),default='cuda:0')
    args=parser.parse_args();existed=args.out.exists()
    try:run(args)
    except BaseException as error:
        if not existed and args.out.is_dir() and not (args.out/'failure.json').exists():
            write_json(args.out/'failure.json',dict(status='failed',error=repr(error),
                traceback=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':main()
