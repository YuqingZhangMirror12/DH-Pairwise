"""Fresh model-only exports selected after an unchanged completed training run.

This adapter never rewrites a historical binding, old validation observation,
selection, or model export. It never starts training or allocates a GPU.
The production backend delegates completion and committed-state checks to the
existing strict terminal/checkpoint primitives. Tests can inject CPU fakes.
"""
import copy
import hashlib
import json
from pathlib import Path

EXPORT_SCHEMA = 'mixed-sim-matcher-export/2'
SELECTION_KIND = 'mixed_sim_v2_best'
CANDIDATE_SCHEMA = 'model-selection-candidate-manifest/2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def receipt(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=file_sha(path))


def read_bound(item):
    require(set(item) == {'path', 'sha256'}, 'an exact path/SHA receipt is required')
    path = Path(item['path'])
    require(path.is_absolute() and file_sha(path) == item['sha256'], 'bound artifact changed')
    return read(path)


class ProductionBackend:
    """Lazy imports permit no-Torch CPU tests without weakening production checks."""

    def verify_original(self, controller_root, spec_path, arm):
        from ..curriculum_training_v1.runtime_plan import RuntimePlan
        root = Path(controller_root)
        binding = read(root / 'formal/training_complete.json')['binding']
        plan = RuntimePlan(binding['common_plan'], binding['common_plan_sha256'])
        if arm in ('B0', 'MIX'):
            from ..curriculum_training_v1.matcher_terminal import verified_export
            order = 'mixed' if arm == 'MIX' else 'curriculum'
            saved, origin = verified_export(root, spec_path, plan, order, 'equal_budget_endpoint')
            require('matcher_v2_experiment' not in saved['binding'], 'old-architecture arm binding differs')
        else:
            from ..matcher_v2_v1.terminal import verified_export
            require(arm in ('B1', 'B2', 'B3'), 'unregistered experiment arm')
            saved, origin = verified_export(root, spec_path, plan, 'equal_budget_endpoint')
            require(saved['binding']['matcher_v2_experiment']['arm'] == arm, 'original arm differs')
        return saved, origin

    def checkpoint_at(self, root, update, binding):
        from ..curriculum_training_v1.runtime_io import checkpoint_at
        return checkpoint_at(root, update, binding)

    def model_only(self, state, binding):
        from ..curriculum_training_v1.runtime_io import model_only
        return model_only(state, binding)

    def save_model(self, path, record):
        from ..curriculum_training_v1.runtime_io import save_model
        return save_model(path, record)

    def tree_sha(self, model):
        from ..curriculum_training_v1.checkpoint_io import tree_sha
        return tree_sha(model)


def original_training_evidence(controller_root, spec_path, arm, backend=None):
    """Certify actual successful training, not a model file or completion flag alone."""
    backend = backend or ProductionBackend()
    require(arm in ('B0', 'MIX', 'B1', 'B2', 'B3'), 'unregistered original arm')
    root = Path(controller_root).resolve(strict=True)
    spec = Path(spec_path).resolve(strict=True)
    for name in ('controller_failure.json', 'failure.json', 'formal/failure.json', 'formal/exports/failure.json'):
        require(not (root / name).exists(), 'failure precedes stale training completion')
    require(not list((root / 'formal').glob('failure_attempt_*.json')), 'unresolved formal failure')
    saved, origin = backend.verify_original(root, spec, arm)
    binding = saved['binding']; plan = binding['common_plan']
    require(binding['module'] == plan['module'] == saved['module'] == 'matcher'
            and binding['run_mode'] == 'formal'
            and digest(plan) == binding['common_plan_sha256']
            and saved['selection_kind'] == 'equal_budget_endpoint'
            and saved['updates'] == saved['total_completed_updates'] == plan['total_updates']
            and saved['optimizer_imported'] is False and saved['training_rng_included'] is False,
            'original endpoint/budget/binding must be verified')
    require(binding['order'] == ('mixed' if arm == 'MIX' else 'curriculum'), 'original presentation order differs')
    paths = ['formal_launch.json', 'formal_return.json', 'controller_complete.json',
             'export_process_return.json', 'gpu_gate.json', 'formal/training_complete.json',
             'formal/exports/training_complete.json', 'formal/exports/selection.json']
    evidence = {name: receipt(root / name) for name in paths}
    returned = read_bound(evidence['formal_return.json'])
    require(returned.get('phase') == 'formal' and type(returned.get('returncode')) is int
            and returned['returncode'] == 0
            and returned.get('launch_sha256') == evidence['formal_launch.json']['sha256'],
            'actual successful formal process return required')
    return dict(arm=arm, controller_root=str(root), execution_spec=receipt(spec),
                source_binding=copy.deepcopy(binding), source_origin=origin, artifacts=evidence)


def _verified_evaluation(item, candidate, protocol, manifest_sha, native_bindings):
    from .native_eval import IMPLEMENTATION, bind_materialized_select
    require(set(item) == {'report', 'launch', 'process_return'}, 'complete actual evaluation receipts required')
    report, launch, returned = (read_bound(item[key]) for key in ('report', 'launch', 'process_return'))
    require(launch.get('schema') == 'model-selection-evaluation-launch/2'
            and launch.get('phase') == 'matcher_select' and launch.get('role') == 'select'
            and launch.get('protocol_sha256') == protocol['sha256']
            and launch.get('candidate_manifest_sha256') == manifest_sha
            and launch.get('update') == candidate['update']
            and launch.get('checkpoint_sha256') == candidate['checkpoint_file']['sha256']
            and launch.get('model_state_sha256') == candidate['model_state_sha256']
            and launch.get('report_path') == item['report']['path'],
            'evaluation launch must bind this candidate/protocol/report')
    command = launch.get('command')
    require(isinstance(command, list) and bool(command)
            and all(isinstance(part, str) and part for part in command), 'actual evaluation command required')
    require(launch.get('native_evaluator_revision') == IMPLEMENTATION,
            'bound frozen native evaluator revision required')
    for key in ('materialized_select', 'source_inventory', 'evaluation_config'):
        ref = launch.get(key)
        require(isinstance(ref, dict), 'native evaluation input path/SHA receipts required')
        read_bound(ref)
        native_bindings['receipts'][(ref['path'], ref['sha256'])] = ref
    signature = tuple(launch[key]['sha256'] for key in
                      ('materialized_select', 'source_inventory', 'evaluation_config'))
    require(native_bindings.setdefault('input_signature', signature) == signature,
            'all candidates must use identical materialized SELECT/source/config bytes')
    materialized = launch['materialized_select']
    identity = (materialized['path'], materialized['sha256'])
    if identity not in native_bindings['materialized']:
        files = bind_materialized_select(protocol, *identity)
        native_bindings['materialized'][identity] = files
    require(returned.get('schema') == 'model-selection-evaluation-return/2'
            and returned.get('phase') == 'matcher_select' and type(returned.get('returncode')) is int
            and returned['returncode'] == 0 and returned.get('launch_sha256') == item['launch']['sha256']
            and returned.get('report_path') == item['report']['path']
            and returned.get('report_sha256') == item['report']['sha256'],
            'successful actual evaluation return must bind launch and exact report bytes')
    require(report.get('checkpoint_sha256') == candidate['checkpoint_file']['sha256']
            and report.get('model_state_sha256') == candidate['model_state_sha256']
            and report.get('update') == candidate['update'], 'reported candidate tensor identity differs')
    return report


def export_reselected_matcher(output, controller_root, spec_path, arm,
                              protocol_receipt, candidate_manifest_receipt,
                              evaluations, *, backend=None):
    """Replay all new reports, verify committed tensors, then publish a fresh export.

    ``evaluations`` maps every positive candidate update to immutable report,
    actual launch, and actual return receipts. A report's own returncode is not
    accepted as evidence that the evaluator returned successfully.
    ``checkpoint_sha256`` always means the actual committed rank_00.pt bytes;
    ``model_state_sha256`` means unchanged runtime_io.model_only tensors.
    This writes only ``output``, which must be new and outside the old run.
    """
    from .protocol import validate_protocol
    from .selection import select_checkpoint
    backend = backend or ProductionBackend()
    output = Path(output).resolve(); root = Path(controller_root).resolve(strict=True)
    require(not output.exists() and root not in output.parents, 'fresh output outside original training required')
    protocol = read_bound(protocol_receipt); validate_protocol(protocol, verify_files=True)
    candidates = read_bound(candidate_manifest_receipt)
    original = original_training_evidence(root, spec_path, arm, backend)
    binding = original['source_binding']; plan = binding['common_plan']
    require(set(candidates) == {'schema', 'arm', 'source_controller_root', 'source_binding_sha256', 'candidates'}
            and candidates['schema'] == CANDIDATE_SCHEMA and candidates['arm'] == arm
            and candidates['source_controller_root'] == str(root)
            and candidates['source_binding_sha256'] == digest(binding), 'candidate inventory original binding differs')
    items = candidates['candidates']
    require(isinstance(items, list) and all(isinstance(item, dict) for item in items), 'candidate inventory missing')
    require([item.get('update') for item in items] == protocol['candidate_updates']
            and isinstance(evaluations, dict) and sorted(evaluations) == protocol['candidate_updates'],
            'complete frozen candidate inventory and evaluation set required')
    # Replay and validate the entire report population before loading large
    # checkpoint tensors. Receipts are consumed, never synthesized here.
    reports = []; native_bindings = dict(receipts={}, materialized={})
    for candidate in items:
        require(set(candidate) == {'update', 'committed_checkpoint', 'checkpoint_file', 'model_state_sha256'},
                'exact committed candidate receipt fields required')
        update = candidate['update']
        require(type(update) is int and 0 < update <= plan['total_updates'], 'candidate outside trained budget')
        reports.append(_verified_evaluation(evaluations[update], candidate, protocol,
                                             candidate_manifest_receipt['sha256'], native_bindings))
    result = select_checkpoint(protocol, reports)
    require(result['selection_kind'] == SELECTION_KIND and result['role'] == 'select'
            and result['real_used'] is False and result['test_used'] is False
            and result['cal_used_for_matcher_selection'] is False, 'simulation SELECT-only result required')
    chosen = result['selected_update']
    selected_model = selected_commit = None
    # Every candidate remains mandatory, including non-winners. Release its
    # optimizer/RNG and model tensors immediately after verification; retain
    # only the winner's model-only state and committed receipt.
    for candidate in items:
        update = candidate['update']
        checkpoint_path = root / 'formal/checkpoints' / ('update_%06d' % update) / 'rank_00.pt'
        require(candidate['checkpoint_file'] == receipt(checkpoint_path), 'rank_00 checkpoint identity differs')
        state, committed = backend.checkpoint_at(root / 'formal/checkpoints', update, binding)
        require(committed == candidate['committed_checkpoint'], 'committed checkpoint receipt changed')
        require(state['binding'] == binding and state['sampling']['completed_updates'] == update,
                'loaded committed checkpoint binding or update differs')
        model = backend.model_only(state, binding)
        require(backend.tree_sha(model) == candidate['model_state_sha256'], 'actual candidate model tensor hash differs')
        if update == chosen:
            selected_model, selected_commit = model, committed
        del state, model, committed
    require(selected_model is not None and selected_commit is not None, 'selected committed model missing')
    record = dict(schema=EXPORT_SCHEMA, module='matcher', stage='matcher', order=binding['order'],
        arm=arm, selection_kind=SELECTION_KIND, updates=chosen,
        exposures=chosen * plan['effective_batch'], total_completed_updates=plan['total_updates'],
        binding=copy.deepcopy(binding), model=selected_model, committed_checkpoint=selected_commit,
        original_training=original, posthoc_selection=dict(protocol=protocol_receipt,
            candidate_manifest=candidate_manifest_receipt,
            evaluations={str(k): v for k, v in evaluations.items()}, result=result),
        optimizer_imported=False, training_rng_included=False, matcher_retrained=False,
        old_observations_rewritten=False, real_used_for_selection=False, test_used_for_selection=False)
    # Recheck every non-tensor input immediately before publication. Do not
    # synthesize or reuse a historical observation for the newly selected point.
    for ref in [protocol_receipt, candidate_manifest_receipt, original['execution_spec'],
                *original['artifacts'].values(),
                *native_bindings['receipts'].values(),
                *(ref for item in evaluations.values() for ref in item.values())]:
        require(file_sha(ref['path']) == ref['sha256'], 'input changed before new export publication')
    validate_protocol(protocol, verify_files=True)
    from .native_eval import bind_materialized_select
    for identity, files in native_bindings['materialized'].items():
        require(files == bind_materialized_select(protocol, *identity),
                'materialized SELECT binding changed before publication')
    output.mkdir(parents=True, exist_ok=False)
    exported = backend.save_model(output / (SELECTION_KIND + '.pt'), record)
    complete = dict(schema='model-selection-posthoc-export-complete/2', status='export_complete',
        selection_kind=SELECTION_KIND, arm=arm, selected_update=chosen, export=exported,
        original_training=original, protocol=protocol_receipt,
        candidate_manifest=candidate_manifest_receipt, selection=result,
        matcher_retrained=False, head_training_completed=False, real_evaluation_completed=False,
        test_used_for_selection=False, real_used_for_selection=False)
    with (output / 'complete.json').open('x') as stream:
        json.dump(complete, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return complete
