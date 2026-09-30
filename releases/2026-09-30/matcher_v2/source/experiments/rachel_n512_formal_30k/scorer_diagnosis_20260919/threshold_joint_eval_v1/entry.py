"""Isolated entry: bind one immutable training implementation before imports."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1'
RELATIVE = Path(*PACKAGE.split('.'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(path):
    return {p.name: digest(p) for p in sorted(Path(path).glob('*.py'))}


def validate_preparation(args):
    receipt = json.loads(Path(args.preparation).read_text())
    if (receipt.get('status') != 'cpu_preparation_passed'
            or receipt.get('schema') != 'threshold-joint-evaluation-preparation/1'
            or receipt.get('errors') or receipt.get('failures')
            or not receipt.get('both_implementations_import_verified')
            or receipt.get('real_inference_performed') is not False):
        raise ValueError('complete independent evaluation preparation required')
    if inventory(Path(__file__).parent) != receipt['adapter_python_sha256']:
        raise ValueError('evaluation adapter changed since CPU verification')
    if inventory(args.common_source) != receipt['common_python_sha256']:
        raise ValueError('shared frozen inference utilities changed')
    implementation = Path(args.root)/'source'/RELATIVE
    if inventory(implementation) != receipt['implementations'][args.model_kind]:
        raise ValueError('selected bound implementation differs from CPU verification')
    helper_path = Path(args.joint_source)/RELATIVE/'real_development.py'
    if digest(helper_path) != receipt['real_development_helper_sha256']:
        raise ValueError('source-isolated real selection helper changed')
    if digest(args.real_plan) != receipt['real_plan_sha256']:
        raise ValueError('real source split changed')
    if (getattr(args,'operation',None)=='evaluate'
            and digest(args.case_plan)!=receipt['fixed_case_plan_sha256']):
        raise ValueError('fixed diagnostic cases changed')
    return receipt


def bootstrap(training_source, common_source, joint_source):
    """No monkeypatch of a training source or of the old frozen-Matcher guard."""
    if any(name == 'experiments' or name.startswith('experiments.') for name in sys.modules):
        raise ValueError('use a fresh process for one bound model implementation')
    sys.path.insert(0, str(Path(training_source).resolve()))
    training = importlib.import_module(PACKAGE+'.train')
    if Path(training.__file__).resolve() != (Path(training_source)/RELATIVE/'train.py').resolve():
        raise ValueError('wrong training implementation imported')
    common = types.ModuleType('consensus_joint_eval_common')
    common.__path__ = [str(Path(common_source).resolve())]
    sys.modules[common.__name__] = common
    adapter = types.ModuleType('consensus_joint_eval_adapter')
    adapter.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[adapter.__name__] = adapter
    # For the unchanged frozen control, bind only the NEW data/selection helper
    # to the imported control classes; no joint model/optimizer is imported.
    name = PACKAGE+'.joint_real_development_evaluation'
    helper_path = Path(joint_source)/RELATIVE/'real_development.py'
    spec = importlib.util.spec_from_file_location(name, helper_path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[name] = helper
    spec.loader.exec_module(helper)
    return training, helper


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('evaluate', 'reselect-control'))
    for key in ('root', 'reference', 'out', 'preparation', 'common-source', 'joint-source', 'real-plan'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--model-kind', choices=('joint', 'frozen'), required=True)
    p.add_argument('--selection', choices=('sim', 'real'), default='real')
    p.add_argument('--real-selection', help='independent frozen-control REAL selection receipt')
    p.add_argument('--split', choices=('sim_test_v14', 'dunhuang_cv', 'turufan'))
    p.add_argument('--case-plan')
    p.add_argument('--device', default='cuda:0')
    return p


def main():
    args = parser().parse_args()
    if args.operation == 'reselect-control' and args.model_kind != 'frozen':
        raise ValueError('only the original frozen control needs retrospective REAL selection')
    if args.operation == 'evaluate':
        if not args.split or not args.case_plan:
            raise ValueError('registered split and fixed diagnostic case plan required')
        if args.model_kind == 'frozen' and (args.selection != 'real' or not args.real_selection):
            raise ValueError('reuse existing frozen SIM evaluation; new control evaluation is REAL-selected only')
    validate_preparation(args)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    training, helper = bootstrap(Path(args.root)/'source', args.common_source, args.joint_source)
    import random
    import numpy as np
    import torch
    seed=training.TrainingConfig().data_seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    module = 'reselect' if args.operation == 'reselect-control' else 'evaluate'
    importlib.import_module('consensus_joint_eval_adapter.'+module).run(args, helper)


if __name__ == '__main__':
    main()
