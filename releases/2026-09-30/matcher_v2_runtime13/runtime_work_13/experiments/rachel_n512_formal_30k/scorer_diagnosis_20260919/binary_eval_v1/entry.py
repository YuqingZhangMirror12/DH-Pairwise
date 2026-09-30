"""Fresh-process entry, binding the binary training source before imports."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import types

PACKAGE='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1'
RELATIVE=Path(*PACKAGE.split('.'))

def bootstrap(source,common_source):
    if any(n=='experiments' or n.startswith('experiments.') for n in sys.modules):
        raise ValueError('binary evaluation requires a fresh bound-source process')
    source=Path(source).resolve();sys.path.insert(0,str(source))
    training=importlib.import_module(PACKAGE+'.train')
    if Path(training.__file__).resolve()!=source/RELATIVE/'train.py':raise ValueError('wrong training source imported')
    for name,path in [('consensus_binary_eval_common',common_source),
                      ('consensus_binary_eval_adapter',Path(__file__).parent)]:
        module=types.ModuleType(name);module.__path__=[str(Path(path).resolve())];sys.modules[name]=module
    helper=importlib.import_module(PACKAGE+'.real_development')
    return training,helper

def validate_preparation(args):
    # Import data-only helpers using the local file, not the experiment tree.
    from contracts import read,sha,inventory,PLAN_SHA
    r=read(args.preparation)
    if (r.get('schema')!='binary-evaluation-preparation/1' or r.get('status')!='cpu_preparation_passed'
            or r.get('errors') or r.get('failures') or r.get('verified_variants')!=['patch','stats']
            or r.get('real_inference_performed') is not False or r.get('source_files_unchanged') is not True
            or r.get('real_checkpoints_opened') is not False or r.get('gpu_preflight') is not False
            or r.get('formal_training_started') is not False or r.get('skipped')!=0
            or not isinstance(r.get('tests'),int) or r['tests']<=0):
        raise ValueError('both binary CPU evaluation preparations required')
    runs=r.get('results',[])
    if ([x.get('variant') for x in runs]!=['patch','stats']
            or any(x.get('status')!='passed' or x.get('returncode')!=0 or x.get('tests',0)<=0
                   or x.get('errors')!=0 or x.get('failures')!=0 or x.get('skipped')!=0 for x in runs)
            or sum(x['tests'] for x in runs)!=r['tests']):
        raise ValueError('both actual CPU test runs required')
    if any(not r.get(k) for k in ('adapter_python_sha256','common_python_sha256','training_source_sha256')):
        raise ValueError('nonempty source bindings required')
    if (inventory(Path(__file__).parent)!=r['adapter_python_sha256']
            or inventory(args.common_source)!=r['common_python_sha256']
            or inventory(Path(args.root)/'source',True)!=r['training_source_sha256']
            or sha(args.real_plan)!=r['real_plan_sha256'] or sha(args.real_plan)!=PLAN_SHA
            or sha(args.case_plan)!=r['fixed_case_plan_sha256']):
        raise ValueError('evaluation/training/source-plan binding changed')
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','reference','out','preparation','common-source','real-plan','case-plan'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--variant',choices=('patch','stats'),required=True)
    p.add_argument('--selection',choices=('sim','real'),required=True)
    p.add_argument('--split',choices=('sim_test_v14','dunhuang_cv','turufan'),required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    validate_preparation(a)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    training,helper=bootstrap(Path(a.root)/'source',a.common_source)
    import random
    import numpy as np
    import torch
    seed=training.TrainingConfig(scorer_variant=a.variant).data_seed
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    importlib.import_module('consensus_binary_eval_adapter.evaluate').run(a,helper)

if __name__=='__main__':main()
