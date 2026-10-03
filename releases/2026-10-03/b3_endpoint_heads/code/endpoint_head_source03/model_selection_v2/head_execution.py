"""Independent new-SELECT matched-head entry; never modifies the native runtime.

Compilation and admission are CPU-only. Training reuses the original B3 update
loop with a scoped preparation dependency, not a mutation of module globals.
The original TRAIN ledger/labels/geometry/budget are preserved. Only the frozen
Matcher identity and the SIM SELECT/CAL population change. No Task3 overlays,
automatic GPU allocation, retries, waiting process, or old-head imports exist.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import types

from . import head_bridge as bridge, posthoc_export as ex
from .checkpoint_scan import module, save
from .protocol import canonical, digest, require
from .released_protocol import checked

SCHEMA = 'mixed-select-head-execution/1'
FIELDS = {'schema', 'locked', 'arm', 'module', 'original_execution', 'scan_request',
          'selected_matcher_adoption', 'validation_contract', 'runtime_plan',
          'runtime_schedule', 'topology', 'runtime', 'native_inventory',
          'external_python', 'combined_admission', 'task3_overlay_applied'}
HEADS = ('scorer_patch', 'scorer_stats')


def external_inventory():
    return {p.name: ex.file_sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def verify_sources(runtime, native_inventory, external_python):
    require(external_inventory() == external_python, 'independent head source changed')
    runtime = Path(runtime).resolve(strict=True)
    expected = checked(native_inventory)
    actual = {str(p.relative_to(runtime)): ex.file_sha(p) for p in runtime.rglob('*.py')}
    require(actual == expected, 'frozen native runtime changed')
    return runtime


def derive_head(values, original_spec, name, runtime, adoption=None):
    """Use the original compiler; translate no clocks or budgets a second way."""
    require(name in HEADS and original_spec['arm'] == 'B3'
            and original_spec['module'] == 'matcher'
            and original_spec['selected_matcher'] is None, 'original B3 Matcher execution required')
    inputs = module('matcher_v2_v1.runtime_inputs', runtime)
    base_record = checked(values['base_execution']['runtime_plan'])
    plans = module('curriculum_training_v1.runtime_plan', runtime)
    original = plans.lock_record(base_record, values['base_ledger'])
    base = inputs.derived_base_plan(original, values['base_ledger'], name)
    ledger, plan, schedule = module('matcher_v2_v1.runtime_schedule', runtime).compile_plan(
        base, values['base_ledger'], 'B3', additive=values['ledger'].ledger,
        admitted_data_sha256=original_spec['combined_admission']['sha256'])
    strip = lambda v: {k: v for k, v in v.items() if k not in ('module', 'selection_rule')}
    require(strip(plan.record) == strip(values['plan'].record)
            and ledger == values['ledger'], 'head TRAIN ledger, optimizer settings or budget changed')
    topology = inputs.registered_topology(name)
    if adoption is not None and 'head_topology' in adoption:
        from . import head_endpoint
        head_endpoint.check(adoption)
        topology = dict(adoption['head_topology'])
        require(topology['world_size'] * topology['microbatch'] * topology['accumulate']
                == plan.record['effective_batch'] == 32, 'two GPUs must preserve global batch32')
    return plan, schedule, topology


def original_inputs(adoption):
    bridge.check_terminal_adoption(adoption)
    request = checked(adoption['scan_request'])
    runtime = str(Path(adoption['runtime']).resolve(strict=True))
    require(request['runtime'] == runtime and request['arm'] == 'B3'
            and ex.receipt(request['execution_spec']) == adoption['original_training']['execution_spec'],
            'scan and original execution ancestry differ')
    spec = checked(adoption['original_training']['execution_spec'])
    values = module('matcher_v2_v1.runtime_inputs', runtime).load_inputs(spec)
    require(str(values['source']) == runtime
            and values['plan'].record == adoption['source_binding']['common_plan']
            and values['plan'].sha256 == adoption['source_binding']['common_plan_sha256'],
            'original B3 training plan differs from the selected Matcher')
    return request, runtime, spec, values


def compile_specs(scan_root, output):
    """Write new immutable execution specs only after the complete scan returns0."""
    output = Path(output).resolve()
    require(not output.exists(), 'new compilation directory required; preserve previous outputs')
    adoption = bridge.verify_completed_scan(scan_root)
    return compile_adopted(adoption, output)


def compile_endpoint(request_path, output):
    """Honor the explicit endpoint request without waiting for/selecting from the scan."""
    from . import head_endpoint
    output = Path(output).resolve()
    require(not output.exists(), 'new compilation directory required; preserve previous outputs')
    return compile_adopted(head_endpoint.adopt(request_path), output)


def compile_adopted(adoption, output):
    output = Path(output).resolve()
    require(not output.exists(), 'new compilation directory required; preserve previous outputs')
    request, runtime, original, values = original_inputs(adoption)
    external = external_inventory()
    verify_sources(runtime, request['native_inventory'], external)
    contract = bridge.bind_head_validation(adoption)
    # All validations/derivations precede output creation. No process is launched.
    derived = {name: derive_head(values, original, name, runtime, adoption) for name in HEADS}
    output.mkdir(parents=True)
    adoption_ref = save(output/'selected_matcher_adoption.json', adoption)
    contract_ref = save(output/'validation_contract.json', contract)
    specs = {}
    for name, (plan, schedule, topology) in derived.items():
        root = output/name
        spec = dict(schema=SCHEMA, locked=True, arm='B3', module=name,
            original_execution=adoption['original_training']['execution_spec'],
            scan_request=adoption['scan_request'], selected_matcher_adoption=adoption_ref,
            validation_contract=contract_ref, runtime_plan=save(root/'runtime_plan.json', plan.record),
            runtime_schedule=save(root/'runtime_schedule.json', schedule), topology=topology,
            runtime=runtime, native_inventory=request['native_inventory'], external_python=external,
            combined_admission=original['combined_admission'], task3_overlay_applied=False)
        specs[name] = save(root/'execution.json', canonical(spec))
    result = dict(schema='mixed-select-head-compilation/1', status='compiled_no_training',
        selected_matcher_adoption=adoption_ref, validation_contract=contract_ref, specs=specs,
        fresh_heads=True, original_train_and_labels_preserved=True,
        task3_overlay_applied=False, training_started=False, gpu_gate_required=True,
        matcher_choice=adoption.get('selection_kind'), matcher_update=adoption.get('update'))
    save(output/'compiled.json', result)
    return result


def load_inputs(spec_path):
    spec = ex.read(spec_path)
    require(set(spec) == FIELDS and spec['schema'] == SCHEMA and spec['locked'] is True
            and spec['arm'] == 'B3' and spec['module'] in HEADS
            and spec['task3_overlay_applied'] is False, 'exact independent matched-head manifest required')
    runtime = str(verify_sources(spec['runtime'], spec['native_inventory'], spec['external_python']))
    adoption = checked(spec['selected_matcher_adoption'])
    require(adoption['runtime'] == runtime and adoption['scan_request'] == spec['scan_request']
            and adoption['original_training']['execution_spec'] == spec['original_execution'],
            'head execution lost its completed scan binding')
    request, loaded_runtime, original, values = original_inputs(adoption)
    require(loaded_runtime == runtime and request['native_inventory'] == spec['native_inventory']
            and spec['combined_admission'] == original['combined_admission'],
            'original TRAIN/source ancestry changed')
    plan, schedule, topology = derive_head(values, original, spec['module'], runtime, adoption)
    require(checked(spec['runtime_plan']) == plan.record and checked(spec['runtime_schedule']) == schedule
            and spec['topology'] == topology
            and all(type(v) is int for v in spec['topology'].values()), 'head plan/schedule/topology differs')
    contract = checked(spec['validation_contract'])
    require(contract == bridge.bind_head_validation(adoption), 'new SELECT/CAL population differs')
    values = dict(values, plan=plan, schedule=schedule, contract=contract,
                  selected_matcher_adoption=adoption)
    return spec, values


def prepare(args):
    """Native preparation with an explicitly typed new-Matcher/validation import."""
    spec, values = load_inputs(args.spec)
    runtime = spec['runtime']; source = values['source']
    require(args.mode in ('gate', 'formal')
            and ((type(args.gate_stop) is int and args.gate_stop in (1, 12))
                 if args.mode == 'gate' else args.gate_stop is None),
            'discarded gate requires1/12 actual updates; formal uses the original full budget')
    require(args.mode != 'gate' or args.gate_receipt is None, 'gate cannot import a formal receipt')
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    topology = module('curriculum_training_v1.training_core', runtime).Topology(rank=rank, **spec['topology'])
    require(world == topology.world_size and world in (1, 2) and rank == local
            and 0 <= rank < world, 'actual torchrun ranks must match the registered head topology')
    os.environ['CURRICULUM_BASELINE_SOURCE'] = str(source)
    base = values['base_execution']
    architecture = module('curriculum_training_v1.verify_validation_preparation', runtime).reference_architecture(
        base['reference_checkpoint']['path'])
    require(architecture.canvas_size == 800 and architecture.contour_cap == 512 and architecture.feature_dim == 96,
            'original full800/N512 architecture required')
    geometry = module('s7_consensus_v1.compatibility', runtime).CompatibilityConfig.from_calibration(values['geometry'])
    parts = bridge.make_components(values['plan'], values['schedule'], topology, source, architecture,
        geometry, values['selected_matcher_adoption'], values['contract'])
    formal = dict(parts['binding'], execution_manifest_sha256=ex.file_sha(args.spec))
    names = sorted(name for name, p in parts['module'].named_parameters() if p.requires_grad)
    require(bool(names), 'fresh head has no trainable parameters')
    gate = None
    if args.mode == 'formal':
        require(args.gate_receipt is not None, 'formal training requires the actual matching GPU resume gate')
        gate = module('curriculum_training_v1.execution', runtime).check_gate(args.gate_receipt, formal)
        proof = ex.read(args.gate_receipt)
        for name in ('gradient_uninterrupted', 'gradient_resumed_from_update1'):
            module('matcher_v2_v1.gradient_gate', runtime).check_gradient_receipt(proof[name], formal, names, world)
    dataset = module('matcher_v2_v1.data_runtime', runtime).CombinedDataset(
        spec['combined_admission'], base['admission'], values['base_ledger'], values['ledger'], source)
    return spec, values, topology, local, parts, formal, gate, dataset


def scoped_native_run(native_run, args):
    """Keep the native function and module globals untouched, including on error.

    This single explicit dependency substitution changes CPU admission only.
    The compiled code for updates, AdamW, RNG, committed checkpoints, validation,
    pause handling, gradient receipts and exports remains the frozen run body.
    """
    require(isinstance(native_run, types.FunctionType)
            and 'prepare' in native_run.__code__.co_names, 'expected frozen B3 update entry required')
    namespace = dict(native_run.__globals__, prepare=prepare)
    entry = types.FunctionType(native_run.__code__, namespace, native_run.__name__,
                               native_run.__defaults__, native_run.__closure__)
    entry.__kwdefaults__ = deepcopy(native_run.__kwdefaults__)
    return entry(args)


def prepare_cpu(spec_path, output):
    """Construct the actual full-resolution selected Matcher and fresh head once.

    This is an admission receipt, not a successful GPU gate or training result.
    No synthetic model, parameter-count-only shortcut, or GPU forward is used.
    """
    import torch
    from types import SimpleNamespace
    require(not torch.cuda.is_initialized(), 'CPU preparation must not own a CUDA context')
    output = Path(output).resolve()
    require(not output.exists(), 'preserve existing CPU preparation')
    args = SimpleNamespace(spec=Path(spec_path), mode='gate', gate_stop=12, gate_receipt=None)
    spec, values, topology, local, parts, formal, gate, dataset = prepare(args)
    require(not torch.cuda.is_initialized() and gate is None and local == 0,
            'CPU preparation touched CUDA or imported a gate')
    model = parts['model']; hashing = module('curriculum_training_v1.checkpoint_io', spec['runtime'])
    require(not any(p.requires_grad for p in model.matcher.parameters())
            and not any(m.training for m in model.matcher.modules()), 'Matcher is not fully frozen/eval')
    result = dict(schema='mixed-select-head-cpu-preparation/1', status='passed',
        execution=ex.receipt(spec_path), external_python=spec['external_python'],
        native_inventory=spec['native_inventory'], formal_binding_sha256=digest(formal),
        model_state_sha256=hashing.tree_sha(model.state_dict()),
        matcher_state_sha256=hashing.tree_sha(model.matcher.state_dict()),
        head_state_sha256=hashing.tree_sha(model.head.state_dict()),
        trainable_parameter_names=sorted(n for n, p in parts['module'].named_parameters() if p.requires_grad),
        original_train_ledger_sha256=values['ledger'].sha256,
        original_total_updates=values['ledger'].total_updates,
        full800_n512=True, cuda_initialized=False, model_forward_calls=0, optimizer_updates=0,
        task3_overlay_applied=False, gpu_gate_passed=False, training_started=False)
    save(output, result)
    return result


def run(args):
    spec = ex.read(args.spec)
    # Refuse changed code before importing the original CUDA/update module.
    runtime = str(verify_sources(spec['runtime'], spec['native_inventory'], spec['external_python']))
    native = module('matcher_v2_v1.execution', runtime)
    require(Path(native.__file__).resolve().is_relative_to(Path(runtime)), 'wrong native execution source')
    return scoped_native_run(native.run, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('compile')
    build.add_argument('--scan-root', type=Path, required=True)
    build.add_argument('--out', type=Path, required=True)
    endpoint = sub.add_parser('compile-endpoint')
    endpoint.add_argument('--request', type=Path, required=True)
    endpoint.add_argument('--out', type=Path, required=True)
    cpu = sub.add_parser('prepare-cpu')
    cpu.add_argument('--spec', type=Path, required=True)
    cpu.add_argument('--out', type=Path, required=True)
    train = sub.add_parser('train')
    train.add_argument('--spec', type=Path, required=True)
    train.add_argument('--out', type=Path, required=True)
    train.add_argument('--mode', choices=('gate', 'formal'), required=True)
    train.add_argument('--gate-stop', type=int, choices=(1, 12))
    train.add_argument('--gate-receipt', type=Path)
    train.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'compile':
        compile_specs(args.scan_root, args.out)
        return 0
    if args.action == 'compile-endpoint':
        compile_endpoint(args.request, args.out)
        return 0
    if args.action == 'prepare-cpu':
        prepare_cpu(args.spec, args.out)
        return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
