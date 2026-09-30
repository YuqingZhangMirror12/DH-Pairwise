"""Reconstruct registered B1--B3 execution inputs before any CUDA allocation."""
import copy
from pathlib import Path

from ..curriculum_training_v1.checkpoint_io import file_sha
from ..curriculum_training_v1.exposure import STAGES, SampleRef, build_ledger, digest
from ..curriculum_training_v1.model_adapter import require
from ..curriculum_training_v1.runtime_plan import lock_record
from .additive_exposure import build_additive_ledger
from .data_runtime import combined_rows, read_bound
from .runtime_schedule import compile_plan
from .validation import RULES, validate_rule

WINDOWS = ((0, 1500, 167), (1500, 15000, 4500), (15000, 21000, 2000), (21000, 24000, 1000))
MODULES = ('matcher', 'scorer_patch', 'scorer_stats')


def registered_topology(module, matcher_microbatch=8):
    """Two measured Matcher layouts; preserve global batch32 and old default."""
    require(module in MODULES, 'unregistered module')
    require(type(matcher_microbatch) is int and matcher_microbatch in (8, 16),
            'registered Matcher microbatch is 8 or 16')
    if module == 'matcher':
        return dict(world_size=2, microbatch=matcher_microbatch,
                    accumulate=16 // matcher_microbatch, workers=4)
    require(matcher_microbatch == 8, 'Matcher microbatch option is not a head setting')
    return dict(world_size=1, microbatch=32, accumulate=1, workers=4)


def verify_composed_source(spec, base_spec):
    source = Path(spec['source_binding']['path']).resolve().parent
    require(source == Path(__file__).resolve().parents[4], 'execute the bound composed source, not another checkout')
    sources = read_bound(spec['source_binding']); composition = read_bound(spec['baseline_composition'])
    require(composition['execution_sha256'] == spec['base_execution']['sha256']
            and composition['python_sha256'] == base_spec['baseline']['python_sha256'],
            'copied production baseline is not the original B0 source')
    actual_files = {str(p.relative_to(source)) for p in source.rglob('*.py')}
    require(actual_files == set(sources), 'unbound or missing Python source in execution snapshot')
    for name, expected in sources.items():
        path = (source/name).resolve(strict=True)
        require(path.is_relative_to(source) and file_sha(path) == expected, 'executing source changed: '+name)
    for name, expected in composition['python_sha256'].items():
        require(sources.get(name) == expected, 'new code changed the preserved production baseline')
    return source, digest(composition['python_sha256'])


def base_inputs(base, baseline_sha):
    require(base.get('schema') == 'curriculum-execution/1' and base.get('locked') is True
            and base.get('selected_matcher') is None, 'original locked random-start Matcher execution required')
    admission = read_bound(base['admission']); record = read_bound(base['runtime_plan'])
    require(admission.get('schema') == 'curriculum-data-admission/1' and admission.get('status') == 'passed'
            and admission.get('gpu_used') is False and record['module'] == 'matcher', 'original data not admitted')
    require(record['total_updates'] == 24000 and record['effective_batch'] == 32
            and [record['stage_updates'][s] for s in STAGES] == [15000, 6000, 3000], 'registered original budget differs')
    require(record['learning_rate_knots'] == [[0, 1e-4], [15000, 5e-5], [21000, 2.5e-5]]
            and record['validation_updates'] == list(range(0, 24001, 1500)), 'original LR/validation protocol differs')
    ledger = build_ledger([SampleRef(**r) for r in admission['catalog']], record['stage_updates'],
                          record['seed'], effective_batch=32)
    original = lock_record(record, ledger)
    require(record['data_admission_sha256'] == base['admission']['sha256']
            and record['geometry_sha256'] == base['geometry']['sha256']
            and record['baseline_sources_sha256'] == baseline_sha, 'original plan bindings differ')
    geometry = read_bound(base['geometry'])
    require(geometry.get('status') == 'complete' and geometry.get('schema') == 's7-consensus-train-geometry/2'
            and geometry.get('curriculum_training_catalog_sha256') == admission['catalog_sha256']
            and geometry.get('curriculum_data_admission_sha256') == base['admission']['sha256']
            and geometry.get('real_used') is False and geometry.get('test_used') is False,
            'preserve original TRAIN-only geometry calibration')
    contract = read_bound(base['simulation_contract'])
    require(contract.get('status') == 'passed' and contract.get('source_disjoint') is True
            and set(contract['validation']) == {'cal_mixed', 'select_mixed'}, 'original fixed SIM contract required')
    for name, split in [('cal_mixed', 'cal'), ('select_mixed', 'select')]:
        item = contract['validation'][name]
        view = read_bound(dict(path=item['path'], sha256=item['sha256']))
        require(view['split'] == split and len(view['entries']) == item['pair_count'], 'SIM validation membership changed')
    require(Path(base['reference_checkpoint']['path']).is_absolute()
            and file_sha(base['reference_checkpoint']['path']) == base['reference_checkpoint']['sha256'],
            'architecture-only reference changed')
    read_bound(base['real_split'])  # Bind the plan only; never open Turufan here.
    return admission, ledger, original, geometry, contract


def derived_base_plan(original, ledger, module):
    require(module in MODULES, 'unregistered module')
    record = copy.deepcopy(original.record)
    # Only module-specific selection policy changes; data, optimizer, seed,
    # exposure order, LR and total original budget are copied verbatim.
    record['module'] = module
    record['selection_rule']['id'] = RULES[module]
    return lock_record(record, ledger)


def load_inputs(spec):
    fields = {'schema', 'locked', 'arm', 'module', 'base_execution', 'source_binding',
              'baseline_composition', 'combined_admission', 'runtime_plan', 'runtime_schedule',
              'topology', 'selected_matcher'}
    require(set(spec) == fields and spec['schema'] == 'matcher-v2-execution/1' and spec['locked'] is True,
            'complete locked v2 execution manifest required')
    require(spec['arm'] in ('B1', 'B2', 'B3') and spec['module'] in MODULES, 'unregistered arm/module')
    base = read_bound(spec['base_execution']); source, baseline_sha = verify_composed_source(spec, base)
    admission, base_ledger, original, geometry, contract = base_inputs(base, baseline_sha)
    original = derived_base_plan(original, base_ledger, spec['module'])
    additive = None; combined_sha = None
    if spec['arm'] == 'B2':
        require(spec['combined_admission'] is None, 'B2 may not use straight data')
    else:
        _, _, refs = combined_rows(spec['combined_admission'], base['admission'], base_ledger)
        additive = build_additive_ledger(base_ledger, refs, WINDOWS, seed=base_ledger.seed)
        combined_sha = spec['combined_admission']['sha256']
    ledger, plan, schedule = compile_plan(original, base_ledger, spec['arm'], additive=additive,
                                         admitted_data_sha256=combined_sha)
    require(read_bound(spec['runtime_plan']) == plan.record and read_bound(spec['runtime_schedule']) == schedule,
            'provided runtime plan is not the exact original-exposure-preserving compilation')
    validate_rule(plan)
    matcher = spec['module'] == 'matcher'
    allowed = [registered_topology(spec['module'])]
    if matcher:allowed.append(registered_topology('matcher', 16))
    require(spec['topology'] in allowed and all(type(v) is int for v in spec['topology'].values()),
            'registered training topology differs')
    require((spec['selected_matcher'] is None) == matcher, 'only new heads import their own completed Matcher')
    return dict(base_execution=base, source=source, admission=admission, base_ledger=base_ledger,
                ledger=ledger, plan=plan, schedule=schedule, geometry=geometry, contract=contract)
