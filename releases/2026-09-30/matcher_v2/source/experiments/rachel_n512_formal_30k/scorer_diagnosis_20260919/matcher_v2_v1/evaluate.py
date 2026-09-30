"""Terminal native/Scorer evaluation for explicitly reconstructed B1--B3.

No training, parameter search, retry or threshold fitting. The heldout loader
is reached only after actual training return, checkpoint and export verification.
"""
import argparse
import importlib
import os
from pathlib import Path
import random
import traceback

import numpy as np
import torch

from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1 import matcher_run
from ..curriculum_training_v1.model_adapter import require
from ..curriculum_training_v1.runtime_io import read
from ..curriculum_scorer_eval_v1.entry import bind_evaluation
from .runtime_inputs import load_inputs
from .terminal import verified_export, load_model
from . import population


def check_preparation(path, root):
    record = read(path); root = Path(root)
    require(record.get('schema') == 'matcher-v2-runtime-cpu-preparation/1'
            and record.get('status') == 'passed' and record.get('tests', 0) > 0
            and record.get('errors') == record.get('failures') == record.get('skipped') == 0
            and record.get('cuda_initialized') is False and record.get('source_files_unchanged') is True
            and record.get('binding_sha256') == file_sha(root/'source_binding.json')
            and record.get('baseline_composition_sha256') == file_sha(root/'baseline_composition.json')
            and record.get('asset_binding_sha256') == file_sha(root/'asset_binding.json'),
            'complete tested v2 source/assets required')
    assets = read(root/'asset_binding.json')
    require(assets and all(file_sha(root/name) == signature for name, signature in assets.items()), 'tested assets changed')
    return record


def run(args):
    spec = read(args.spec); inputs = load_inputs(spec); source = inputs['source']
    check_preparation(args.preparation, source)
    saved, origin = verified_export(args.controller_root, args.spec, inputs['plan'], args.selection)
    plan = read(args.population_plan); population.validate_plan(plan, args.spec, source)
    model, geometry, origin = load_model(saved, origin, source)
    native = spec['module'] == 'matcher'; matcher = model if native else model.matcher
    require(matcher.base.config.canvas_size == 800 and matcher.base.config.contour_cap == 512,
            'full800/N512 final inference required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed = inputs['plan'].record['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = torch.device(args.device); model.to(device)
    origin.update(population_plan_sha256=file_sha(args.population_plan), preparation_sha256=file_sha(args.preparation),
        simulation_revision=plan['simulation_revision'], real_split_sha256=plan['real_split']['sha256'])
    wanted = population.wanted_ids(plan, args.split)
    if native:
        meta, batches, data_source, dataset = population.load_population(args.split, plan, source)
        roles = read(plan['real_split']['path'])
        groups = {name:[p['pair_id'] for p in rows] for name, rows in
            population.population_groups(meta['pairs'], args.split, roles).items()}
        origin.update(split=args.split, source=data_source)
        result = matcher_run.run_population(model, geometry, source, meta, batches, data_source,
            out=args.out, provenance=origin, wanted_ids=wanted, groups=groups, device=device,
            targets_callback=lambda:population.targets_after_prediction(meta, args.split, plan, dataset))
        proof = matcher_run.verify_population(args.out)
    else:
        package = Path(__file__).resolve().parent.parent
        bind_evaluation(package/'s7_consensus_eval_v14', package/'binary_eval_v1')
        for split in ('sim_select', *population.STRAIGHT):
            origin['thresholds'][split] = origin['thresholds']['sim_test']
            origin['threshold_origins'][split] = origin['threshold_origins']['sim_test']
        evaluator = importlib.import_module(__package__.rsplit('.', 1)[0]+'.curriculum_scorer_eval_v1.evaluate')
        result = evaluator.run_population(model, source, plan, args.split, origin, args.out, device,
            registered_splits=population.SPLITS, population_loader=population.load_population,
            group_builder=population.population_groups, target_loader=population.targets_after_prediction,
            extra_diagnostic_ids=wanted)
        auditor = importlib.import_module(__package__.rsplit('.', 1)[0]+'.curriculum_scorer_eval_v1.audit')
        proof = auditor.verify_population(args.out)
    write_json(Path(args.out)/'independent_artifact_audit.json', proof)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('spec', 'population-plan', 'preparation', 'controller-root', 'out'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--selection', choices=('sim_best', 'real_best', 'equal_budget_endpoint'), required=True)
    parser.add_argument('--split', choices=population.SPLITS, required=True)
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    args = parser.parse_args(); existed = args.out.exists()
    try:run(args)
    except BaseException as error:
        if not existed and args.out.is_dir() and not (args.out/'failure.json').exists():
            write_json(args.out/'failure.json', dict(status='failed', error=repr(error),
                traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':main()
