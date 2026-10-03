"""Original frozen final evaluator with the seed-only population bridge.

No training/selection/threshold changes; six model inputs and targets retain
the admitted canonical hashes. Non-straight heldouts use the original loader.
"""
import argparse
import importlib
import os
from pathlib import Path
import random
import traceback

import numpy as np
import torch

from common import PACKAGE, api, check_preparation as check_repair_preparation
from strict_population_bridge import load_population as strict_load_population

original = api('matcher_v2_v1.evaluate')
file_sha, write_json, require, read = original.file_sha, original.write_json, original.require, original.read
load_inputs, verified_export = original.load_inputs, original.verified_export
check_preparation, bind_evaluation = original.check_preparation, original.bind_evaluation
matcher_run, population = original.matcher_run, original.population


def load_model(saved, origin, source):
    """Repair only the verified frozen-export root mode, never training tensors.

    source13 stays immutable. Do not hide trainable weights, wrong adapters or
    a genuinely training child module by calling eval indiscriminately.
    """
    model, geometry, loaded = original.load_model(saved, origin, source)
    if saved['module'] == 'matcher':
        matcher_api = api('s7_consensus_v1.matcher')
        require(isinstance(model, matcher_api.S7MatcherAdapter) and model.frozen
                and not any(p.requires_grad for p in model.parameters())
                and not any(m.training for m in list(model.modules())[1:]),
                'only a fully frozen export with inactive children can receive mode repair')
        tree_sha = api('curriculum_training_v1.checkpoint_io').tree_sha
        before = tree_sha(model.state_dict())
        model.eval()
        require(not any(m.training for m in model.modules())
                and tree_sha(model.state_dict()) == before,
                'export eval mode changed weights or left training modules')
    return model, geometry, loaded


def load_population(split, plan, source):
    if split in population.STRAIGHT:
        return strict_load_population(population, split, plan, source)
    return population.load_population(split, plan, source)


def run(args):
    # This body is mechanically checked against source13 evaluate.run. The
    # only differences are loader dispatch and locating the original package.
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
        meta, batches, data_source, dataset = load_population(args.split, plan, source)
        roles = read(plan['real_split']['path'])
        groups = {name:[p['pair_id'] for p in rows] for name, rows in
            population.population_groups(meta['pairs'], args.split, roles).items()}
        origin.update(split=args.split, source=data_source)
        result = matcher_run.run_population(model, geometry, source, meta, batches, data_source,
            out=args.out, provenance=origin, wanted_ids=wanted, groups=groups, device=device,
            targets_callback=lambda:population.targets_after_prediction(meta, args.split, plan, dataset))
        proof = matcher_run.verify_population(args.out)
    else:
        package = Path(original.__file__).resolve().parent.parent
        bind_evaluation(package/'s7_consensus_eval_v14', package/'binary_eval_v1')
        for split in ('sim_select', *population.STRAIGHT):
            origin['thresholds'][split] = origin['thresholds']['sim_test']
            origin['threshold_origins'][split] = origin['threshold_origins']['sim_test']
        evaluator = importlib.import_module(PACKAGE+'.curriculum_scorer_eval_v1.evaluate')
        result = evaluator.run_population(model, source, plan, args.split, origin, args.out, device,
            registered_splits=population.SPLITS, population_loader=load_population,
            group_builder=population.population_groups, target_loader=population.targets_after_prediction,
            extra_diagnostic_ids=wanted)
        auditor = importlib.import_module(PACKAGE+'.curriculum_scorer_eval_v1.audit')
        proof = auditor.verify_population(args.out)
    write_json(Path(args.out)/'independent_artifact_audit.json', proof)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('spec', 'population-plan', 'preparation', 'controller-root', 'out', 'repair-preparation'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--selection', choices=('sim_best', 'real_best', 'equal_budget_endpoint'), required=True)
    parser.add_argument('--split', choices=population.SPLITS, required=True)
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    args = parser.parse_args(); existed = args.out.exists()
    check_repair_preparation(args.repair_preparation)
    require(not existed, 'exclusive evaluation output required')
    try:
        run(args)
    except BaseException as error:
        if not existed and args.out.is_dir() and not (args.out/'failure.json').exists():
            write_json(args.out/'failure.json', dict(status='failed', error=repr(error),
                traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
