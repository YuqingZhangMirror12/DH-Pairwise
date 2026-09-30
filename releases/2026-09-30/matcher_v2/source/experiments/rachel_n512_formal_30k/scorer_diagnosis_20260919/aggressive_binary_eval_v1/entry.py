"""Fresh-process frozen experiment3 evaluation; never imports running sources."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1'
RELATIVE = Path(*PACKAGE.split('.'))
PLAN_SHA = '0f8c13bbc4277052ecd6d21e335e97bf58dc8522174e2a11341c3221d50450f6'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def inventory(path, recursive=False):
    root=Path(path)
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*.py') if recursive else root.glob('*.py'))}


def bootstrap(source, common_source, binary_source):
    if any(n=='experiments' or n.startswith('experiments.') for n in sys.modules):
        raise ValueError('aggressive evaluation requires a fresh bound-source process')
    source=Path(source).resolve();sys.path.insert(0,str(source))
    training=importlib.import_module(PACKAGE+'.train')
    if Path(training.__file__).resolve()!=source/RELATIVE/'train.py':
        raise ValueError('wrong training source imported')
    if training.TrainingConfig().schema!='aggressive-binary-training/1':
        raise ValueError('not the independently prepared experiment3 source')
    for name,path in (('consensus_binary_eval_common',common_source),
                      ('consensus_binary_eval_adapter',binary_source),
                      ('consensus_aggressive_eval_adapter',Path(__file__).parent)):
        module=types.ModuleType(name);module.__path__=[str(Path(path).resolve())];sys.modules[name]=module
    return training,importlib.import_module(PACKAGE+'.real_development')


def validate_preparation(args):
    from pathlib import Path
    r=json.loads(Path(args.preparation).read_text())
    if (r.get('schema')!='aggressive-binary-evaluation-preparation/1'
            or r.get('status')!='cpu_preparation_passed' or r.get('verified_variants')!=['patch']
            or type(r.get('tests')) is not int or r['tests']<=0
            or any(r.get(k)!=0 for k in ('errors','failures','skipped'))
            or r.get('source_files_unchanged') is not True
            or any(r.get(k) is not False for k in
                   ('real_inference_performed','real_checkpoints_opened','gpu_preflight','formal_training_started'))):
        raise ValueError('completed CPU experiment3 evaluator preparation required')
    runs=r.get('results',[])
    if (len(runs)!=1 or runs[0].get('variant')!='patch' or runs[0].get('status')!='passed'
            or runs[0].get('returncode')!=0 or runs[0].get('tests')!=r['tests']
            or any(runs[0].get(k)!=0 for k in ('errors','failures','skipped'))):
        raise ValueError('actual Patch CPU evaluator tests required')
    actual=dict(adapter_python_sha256=inventory(Path(__file__).parent),
        binary_python_sha256=inventory(args.binary_source),common_python_sha256=inventory(args.common_source),
        training_source_sha256=inventory(Path(args.root)/'source',True),real_plan_sha256=sha(args.real_plan),
        fixed_case_plan_sha256=sha(args.case_plan))
    if (any(not v or r.get(k)!=v for k,v in actual.items())
            or actual['real_plan_sha256']!=PLAN_SHA):
        raise ValueError('experiment3 evaluator/source/plan binding changed')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference','out','preparation','common-source','binary-source','real-plan','case-plan'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--selection',choices=('sim','real'),required=True)
    p.add_argument('--split',choices=('sim_test_aggressive','dunhuang_cv','turufan'),required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    validate_preparation(args)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    training,helper=bootstrap(Path(args.root)/'source',args.common_source,args.binary_source)
    import random
    import numpy as np
    import torch
    seed=training.TrainingConfig().data_seed
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    importlib.import_module('consensus_aggressive_eval_adapter.evaluate').run(args,helper)


if __name__=='__main__':main()
