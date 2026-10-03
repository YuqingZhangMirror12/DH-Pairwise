"""One-shot frozen terminal inference for a newly trained matched B3 head.

No training, retries, GPU discovery, re-selection or calibration. An actual
parent process must record this worker's return; JSON completion alone is not
process success. CAL/SELECT and Dunhuang development are reused, never inferred
by this entry. It delegates prediction and numeric audits to the frozen native
evaluator, with an explicit new-type model admission and TEST-only population.
"""
import argparse
import os
from pathlib import Path
import random

from . import head_execution as execution, head_population as population
from . import head_terminal as terminal, posthoc_export as ex
from .checkpoint_scan import module, save
from .protocol import require


def terminal_origin(origin, plan, plan_ref, loaded):
    require(plan['origin'] == origin and plan['schema'] == population.SCHEMA,
            'population belongs to another selected head/export')
    thresholds = dict(origin['thresholds']); sources = dict(origin['threshold_origins'])
    for target, original in [('dunhuang_test', 'dunhuang_cv'), ('sim_straight_test', 'sim_test')]:
        thresholds[target] = thresholds[original]; sources[target] = sources[original]
    return dict(loaded, thresholds=thresholds, threshold_origins=sources,
        population_plan=plan_ref, population_plan_sha256=plan_ref['sha256'],
        validation_reuse=plan['validation_reuse'], matched_fresh_head=True,
        development_inference_repeated=False, rotation_ensemble=False,
        new_sim_calibration_already_frozen=True)


def run(args):
    import numpy as np
    import torch
    require(args.split in population.SPLITS and args.device in ('cuda:0', 'cpu'),
            'explicit registered split and device required')
    out = Path(args.out).resolve()
    require(not out.exists(), 'preserve previous terminal output; no automatic retry')
    spec = ex.read(args.spec); runtime = spec['runtime']
    require(spec.get('schema') == execution.SCHEMA, 'new matched-head execution required')
    execution.verify_sources(runtime, spec['native_inventory'], spec['external_python'])
    plan = ex.read(args.population_plan); population.verify_plan(plan, runtime)
    require(plan['execution'] == ex.receipt(args.spec), 'population and execution differ')
    saved, origin = terminal.verified_export(args.driver_root, args.spec, args.selection_kind)
    require(plan['origin'] == origin, 'selected head differs from pre-inference population lock')
    model, _, loaded = terminal.load_model(saved, origin, runtime)
    architecture = loaded['architecture']
    require(architecture['canvas_size'] == 800 and architecture['contour_cap'] == 512,
            'full800/N512 frozen architecture required')
    seed = saved['binding']['common_plan']['seed']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = torch.device(args.device); model.to(device)
    package = Path(runtime)/'experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919'
    module('curriculum_scorer_eval_v1.entry', runtime).bind_evaluation(
        package/'s7_consensus_eval_v14', package/'binary_eval_v1')
    provenance = terminal_origin(origin, plan, ex.receipt(args.population_plan), loaded)
    evaluator = module('curriculum_scorer_eval_v1.evaluate', runtime)
    result = evaluator.run_population(model, runtime, plan, args.split, provenance, out, device,
        registered_splits=population.SPLITS, population_loader=population.load_population,
        group_builder=lambda rows, split, roles: population.groups(rows, split, roles, runtime),
        target_loader=lambda meta, split, current, dataset: population.targets_after_prediction(
            meta, split, current, dataset, runtime),
        extra_diagnostic_ids=population.diagnostic_ids(plan, args.split, runtime))
    try:
        proof = module('curriculum_scorer_eval_v1.audit', runtime).verify_population(out)
        require(proof['pairs'] == population.COUNTS_TEST[args.split], 'terminal prediction count differs')
        save(out/'independent_artifact_audit.json', proof)
    except BaseException as error:
        save(out/'failure.json', dict(status='failed', phase='independent_artifact_audit',
             error=repr(error), automatic_retry=False))
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'driver-root', 'population-plan', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--selection-kind', choices=('sim_best', 'real_best', 'equal_budget_endpoint'), required=True)
    parser.add_argument('--split', choices=population.SPLITS, required=True)
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    run(parser.parse_args())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
