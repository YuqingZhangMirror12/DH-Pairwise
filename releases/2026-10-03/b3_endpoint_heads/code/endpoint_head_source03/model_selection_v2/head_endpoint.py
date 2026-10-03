"""User-designated B3 endpoint admission for fresh Scorer training.

The original endpoint stays an equal_budget_endpoint export. It is neither an
old SIM winner nor a winner of the in-flight new SELECT scan. Only the scan's
already-frozen source/data registry is reused; no scan results are read.
"""
from pathlib import Path

from . import posthoc_export as ex
from .checkpoint_scan import FrozenBackend, module
from .protocol import canonical, digest, require, validate_protocol
from .released_protocol import checked

SCHEMA = 'user-fixed-b3-endpoint-head-adoption/1'
REQUEST_SCHEMA = 'user-fixed-b3-endpoint-head-request/1'
KIND = 'user_fixed_b3_endpoint'
UPDATE = 31667
MODEL_SHA = '01c7070242c943861c1d44a6c5df4a5b463bb369a26439f9fbb053ea1faf6ac1'
INITIALIZATION = 'user_fixed_b3_endpoint_fresh_head'
DUAL_TOPOLOGY = dict(world_size=2, microbatch=16, accumulate=1, workers=4)


def check_request(request):
    fields = {'schema', 'arm', 'matcher_update', 'matcher_file_sha256',
            'source_registry', 'validation_population_protocol', 'user_instruction',
            'train_labels_unchanged', 'task3_overlay_applied'}
    require(set(request) in (fields, fields | {'head_topology'})
        and request['schema'] == REQUEST_SCHEMA and request['arm'] == 'B3'
        and type(request['matcher_update']) is int and request['matcher_update'] == UPDATE
        and request['matcher_file_sha256'] == MODEL_SHA
        and isinstance(request['user_instruction'], str) and bool(request['user_instruction'])
        and request['train_labels_unchanged'] is True and request['task3_overlay_applied'] is False,
        'exact user-designated B3 endpoint request required')
    if 'head_topology' in request:
        require(request['head_topology'] == DUAL_TOPOLOGY
                and all(type(v) is int for v in request['head_topology'].values()),
                'explicit two-GPU effective32 head topology required')


def adopt(request_path):
    request_ref = ex.receipt(request_path); request = checked(request_ref); check_request(request)
    registry = checked(request['source_registry'])
    protocol = checked(request['validation_population_protocol']); validate_protocol(protocol, verify_files=True)
    require(registry['arm'] == 'B3' and registry['publication'] == protocol['publication'],
            'endpoint and new validation population registry differ')
    # Actual original training success, full budget and endpoint bytes are
    # admitted once here. This is not another terminal inference or re-training.
    original = ex.original_training_evidence(registry['training_root'], registry['execution_spec'],
                                            'B3', FrozenBackend(registry['runtime']))
    origin = original['source_origin']; binding = original['source_binding']
    require(origin['selection_kind'] == 'equal_budget_endpoint'
            and origin['selected_updates'] == origin['total_completed_updates'] == UPDATE
            and binding['common_plan']['total_updates'] == UPDATE
            and origin['checkpoint_sha256'] == MODEL_SHA == ex.file_sha(origin['checkpoint']),
            'requested endpoint identity/update differs')
    model = dict(path=origin['checkpoint'], sha256=MODEL_SHA, update=UPDATE,
                 model_state_sha256=origin['model_state_sha256'])
    value = dict(schema=SCHEMA, arm='B3', selection_kind=KIND, source_binding=binding,
        original_training=original, model=model, update=UPDATE,
        runtime=registry['runtime'], scan_request=request['source_registry'],
        protocol=request['validation_population_protocol'], endpoint_request=request_ref,
        population_protocol_role='validation_population_only_not_matcher_selection',
        matcher_choice_basis='user_requested_endpoint_after_development_observations',
        user_designated_endpoint=True, development_informed_choice=True,
        scan_results_used=False, matcher_retrained=False, task3_overlay_applied=False,
        head_training_started=False, test_used_by_this_admission=False)
    if 'head_topology' in request:
        value['head_topology'] = request['head_topology']
    result = canonical(dict(value, sha256=digest(value)))
    check_terminal(result)
    return result


def check(adoption):
    require(adoption.get('schema') == SCHEMA and adoption.get('arm') == 'B3'
            and adoption.get('selection_kind') == KIND
            and adoption.get('sha256') == digest({k: v for k, v in adoption.items() if k != 'sha256'}),
            'explicit user endpoint adoption required; not a SELECT winner')
    require(type(adoption.get('update')) is int and adoption['update'] == UPDATE
            and adoption['model']['update'] == UPDATE and adoption['model']['sha256'] == MODEL_SHA
            and adoption['source_binding']['common_plan']['total_updates'] == UPDATE,
            'endpoint adoption update/model differs')
    require(adoption.get('user_designated_endpoint') is True and adoption.get('development_informed_choice') is True
            and adoption.get('matcher_choice_basis') == 'user_requested_endpoint_after_development_observations'
            and adoption.get('population_protocol_role') == 'validation_population_only_not_matcher_selection',
            'manual development-informed endpoint choice must remain explicit')
    for key in ('scan_results_used', 'matcher_retrained', 'task3_overlay_applied',
                'head_training_started', 'test_used_by_this_admission'):
        require(adoption.get(key) is False, 'unapproved endpoint change: '+key)
    if 'head_topology' in adoption:
        require(adoption['head_topology'] == DUAL_TOPOLOGY
                and all(type(v) is int for v in adoption['head_topology'].values()),
                'endpoint head topology differs')
    require(not any(key in adoption for key in ('scan_complete', 'scan_actual_return', 'selection',
                                                'candidate_manifest', 'evaluations')),
            'endpoint adoption must not claim scan-result selection')


def check_terminal(adoption):
    check(adoption)
    request = checked(adoption['endpoint_request']); check_request(request)
    require(request['source_registry'] == adoption['scan_request']
            and request['validation_population_protocol'] == adoption['protocol']
            and request.get('head_topology') == adoption.get('head_topology'), 'endpoint request binding differs')
    original = adoption['original_training']; registry = checked(adoption['scan_request'])
    require(original['arm'] == registry['arm'] == 'B3'
            and original['controller_root'] == registry['training_root']
            and original['execution_spec'] == ex.receipt(registry['execution_spec'])
            and original['source_binding'] == adoption['source_binding']
            and registry['runtime'] == adoption['runtime'], 'endpoint original ancestry differs')
    records = {key: checked(ref) for key, ref in original['artifacts'].items()}
    returned = records['formal_return.json']; launch = records['formal_launch.json']
    require(type(returned.get('returncode')) is int and returned['returncode'] == 0
            and returned.get('phase') == launch.get('phase') == 'formal'
            and returned.get('launch_sha256') == original['artifacts']['formal_launch.json']['sha256'],
            'actual original successful endpoint return required')
    command = launch.get('command', [])
    require('experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_v2_v1.execution' in command,
            'dedicated original Matcher command required')
    for flag, value in [('--spec', registry['execution_spec']), ('--mode', 'formal'),
                        ('--out', str(Path(registry['training_root'])/'formal'))]:
        require(command.count(flag) == 1 and command[command.index(flag)+1] == value,
                'original endpoint launch differs: '+flag)
    complete = records['formal/exports/training_complete.json']
    require(complete['binding'] == adoption['source_binding']
            and complete['exports']['equal_budget_endpoint'] == adoption['model'],
            'endpoint export differs from actual completed training')
    require(ex.file_sha(adoption['model']['path']) == MODEL_SHA, 'original endpoint model bytes changed')


def verify_model(saved, adoption, tree_sha):
    check(adoption)
    binding, spec = module('matcher_v2_v1.model_runtime', adoption['runtime']).check_export(saved)
    require(saved.get('selection_kind') == 'equal_budget_endpoint' and saved.get('module') == 'matcher'
            and saved.get('binding') == adoption['source_binding']
            and saved.get('updates') == saved.get('total_completed_updates') == UPDATE
            and saved.get('exposures') == UPDATE*binding['common_plan']['effective_batch']
            and saved.get('optimizer_imported') is False and saved.get('training_rng_included') is False,
            'original endpoint export identity/budget differs')
    require(tree_sha(saved['model']) == adoption['model']['model_state_sha256'],
            'actual endpoint tensor identity differs')
    return binding, spec
