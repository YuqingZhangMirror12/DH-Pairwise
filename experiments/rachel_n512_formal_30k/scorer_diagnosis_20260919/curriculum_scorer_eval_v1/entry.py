"""Explicit terminal evaluator; no GPU search, training, retries or calibration."""
import argparse
import importlib
import os
from pathlib import Path
import random
import sys
import types

from ..curriculum_training_v1.checkpoint_io import file_sha
from ..curriculum_training_v1.execution import load_inputs
from ..curriculum_training_v1.model_adapter import require
from ..curriculum_training_v1.runtime_io import read
from ..curriculum_training_v1.verify_validation_preparation import bind_baseline
from .terminal import CHOICES, MODULES, verified_export, load_model
from .population import SPLITS, validate_plan


def inventory(root):
    root = Path(root).resolve()
    return {p.name:file_sha(p) for p in sorted(root.glob('*.py'))}


def alias(name, root):
    require(name not in sys.modules, 'fresh evaluator alias required: ' + name)
    module = types.ModuleType(name); module.__path__ = [str(Path(root).resolve())]; sys.modules[name] = module


def bind_evaluation(common_source, binary_source):
    alias('consensus_binary_eval_common', common_source)
    alias('consensus_binary_eval_adapter', binary_source)


def check_preparation(path, common_source, binary_source, baseline):
    from .. import curriculum_training_v1 as curriculum
    receipt = read(path)
    require(receipt.get('schema') == 'curriculum-scorer-evaluation-preparation/1'
            and receipt.get('status') == 'passed' and receipt.get('scope') == 'complete'
            and receipt.get('verified_variants') == ['patch', 'stats']
            and type(receipt.get('tests')) is int and receipt['tests'] > 0
            and receipt.get('errors') == receipt.get('failures') == receipt.get('skipped') == 0
            and receipt.get('gpu_used') is False and receipt.get('real_inference_performed') is False
            and receipt.get('source_files_unchanged') is True, 'completed CPU Scorer evaluation preparation required')
    actual = dict(adapter_python_sha256=inventory(Path(__file__).parent),
        curriculum_python_sha256=inventory(Path(curriculum.__file__).parent),
        common_python_sha256=inventory(common_source), binary_python_sha256=inventory(binary_source),
        baseline_python_sha256={str(p.relative_to(baseline)):file_sha(p) for p in Path(baseline).rglob('*.py')})
    require(all(value and receipt.get(key) == value for key,value in actual.items()),
            'tested evaluation/training/baseline source changed')
    return receipt


def run(args):
    import numpy as np
    import torch
    spec = read(args.spec); inputs = load_inputs(spec)
    require(inputs['plan'].record['module'] in MODULES, 'only the two trained curriculum heads are registered')
    check_preparation(args.preparation, args.common_source, args.binary_source, inputs['baseline'])
    # A prior run's stale running JSON, a selected intermediate checkpoint,
    # and a vanished PID are all insufficient to enter heldout evaluation.
    saved, origin = verified_export(args.controller_root, args.spec, inputs['plan'], args.selection)
    bind_baseline(inputs['baseline']); bind_evaluation(args.common_source, args.binary_source)
    population = read(args.population_plan); validate_plan(population, args.spec, inputs['baseline'])
    model, origin = load_model(saved, origin, inputs['baseline'])
    require(model.matcher.base.config.canvas_size == 800 and model.matcher.base.config.contour_cap == 512,
            'formal final evaluation requires full800/N512 inputs')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed = inputs['plan'].record['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = torch.device(args.device); model.to(device)
    origin.update(population_plan_sha256=file_sha(args.population_plan), preparation_sha256=file_sha(args.preparation))
    evaluator = importlib.import_module(__package__+'.evaluate')
    result = evaluator.run_population(model, inputs['baseline'], population, args.split, origin, args.out, device)
    auditor = importlib.import_module(__package__+'.audit')
    from ..curriculum_training_v1.checkpoint_io import write_json
    try:
        proof = auditor.verify_population(args.out)
        write_json(Path(args.out)/'independent_artifact_audit.json', proof)
    except BaseException as error:
        write_json(Path(args.out)/'failure.json',dict(status='failed',error=repr(error),
            phase='independent_artifact_audit',automatic_retry=False))
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('spec', 'population-plan', 'preparation', 'controller-root', 'out', 'common-source', 'binary-source'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--selection', choices=CHOICES, required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
