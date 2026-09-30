"""Compile one new immutable execution from the unchanged original budget.

Compilation does not allocate GPUs or start a process. A head additionally
requires its own arm's completed, SIM-selected Matcher export.
"""
import argparse
from pathlib import Path

from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.model_adapter import require
from .additive_exposure import build_additive_ledger
from .data_runtime import read_bound, combined_rows
from .runtime_inputs import WINDOWS, MODULES, verify_composed_source, base_inputs, derived_base_plan, load_inputs, registered_topology
from .runtime_io import selected_matcher
from .runtime_schedule import compile_plan


def bound(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=file_sha(path))


def compile_execution(base_execution, arm, module, out, combined_admission=None, selected=None, *, matcher_microbatch=8):
    require(arm in ('B1', 'B2', 'B3') and module in MODULES, 'unregistered experiment')
    root = Path(__file__).resolve().parents[4]; out = Path(out).resolve()
    require(not out.exists(), 'new locked-plan directory required')
    matcher = module == 'matcher'
    require((selected is None) == matcher, 'head requires its own completed Matcher')
    topology = registered_topology(module, matcher_microbatch)
    spec = dict(schema='matcher-v2-execution/1', locked=True, arm=arm, module=module,
        base_execution=bound(base_execution), source_binding=bound(root/'source_binding.json'),
        baseline_composition=bound(root/'baseline_composition.json'),
        combined_admission=None if combined_admission is None else bound(combined_admission),
        runtime_plan=None, runtime_schedule=None,
        topology=topology, selected_matcher=selected)
    base = read_bound(spec['base_execution']); _, baseline_sha = verify_composed_source(spec, base)
    _, ledger, original, _, _ = base_inputs(base, baseline_sha)
    original = derived_base_plan(original, ledger, module)
    additive = None; admission_sha = None
    if arm == 'B2':require(combined_admission is None, 'B2 has no added straight TRAIN')
    else:
        require(combined_admission is not None, 'B1/B3 require admitted straight TRAIN')
        _, _, refs = combined_rows(spec['combined_admission'], base['admission'], ledger)
        additive = build_additive_ledger(ledger, refs, WINDOWS, seed=ledger.seed)
        admission_sha = spec['combined_admission']['sha256']
    if not matcher:
        require(set(selected) == {'export_root', 'process_return', 'common_plan_sha256'}, 'completed Matcher source required')
        selected_matcher(selected['export_root'], selected['process_return'], selected['common_plan_sha256'], arm)
    _, plan, schedule = compile_plan(original, ledger, arm, additive=additive, admitted_data_sha256=admission_sha)
    out.mkdir(parents=True)
    write_json(out/'runtime_plan.json', plan.record); write_json(out/'runtime_schedule.json', schedule)
    spec.update(runtime_plan=bound(out/'runtime_plan.json'), runtime_schedule=bound(out/'runtime_schedule.json'))
    load_inputs(spec)  # Exact independent reconstruction before publishing the executable manifest.
    write_json(out/'execution.json', spec)
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-execution', type=Path, required=True)
    p.add_argument('--arm', choices=('B1', 'B2', 'B3'), required=True)
    p.add_argument('--module', choices=MODULES, required=True)
    p.add_argument('--out-new', type=Path, required=True)
    p.add_argument('--combined-admission', type=Path)
    p.add_argument('--selected-matcher', type=Path)
    p.add_argument('--matcher-microbatch', type=int, choices=(8, 16), default=8,
                   help='Matcher only: two GPUs, global batch32; each new configuration requires its GPU gate.')
    args = p.parse_args()
    selected = None if args.selected_matcher is None else read_bound(bound(args.selected_matcher))
    spec = compile_execution(args.base_execution, args.arm, args.module, args.out_new, args.combined_admission, selected,
                             matcher_microbatch=args.matcher_microbatch)
    print(dict(arm=spec['arm'], module=spec['module'], execution=bound(args.out_new/'execution.json'), gpu_started=False))


if __name__ == '__main__':main()
