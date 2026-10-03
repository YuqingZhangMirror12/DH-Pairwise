"""Typed CPU bridge from a verified B3 Matcher choice to fresh B3 heads.

This is not a trainer or scheduler. It does not patch the frozen runtime,
relabel an export as historical sim_best, or admit unreviewed Task3 labels.
Both the completed mixed-SELECT winner and a separately authorized fixed
endpoint have distinct, fail-closed admission paths and provenance.
The eventual training entry must bind this source and pass the existing
discarded 12-update/resume gate before starting either formal head.
"""
from dataclasses import asdict, replace
from pathlib import Path

from . import posthoc_export as ex
from .checkpoint_scan import UPDATES, module
from .protocol import canonical, digest, require, validate_protocol
from .released_protocol import checked, COUNTS
from .selection import select_checkpoint

ADOPTION_SCHEMA = 'mixed-select-matcher-head-adoption/1'
HEAD_SCHEMA = 'mixed-select-matched-head-binding/1'
CONTRACT_SCHEMA = 'mixed-select-matched-head-validation/1'


def verify_completed_scan(root):
    """Require actual outer/worker returns and replay the frozen selection.

    Does not re-evaluate pixels or open original training checkpoints. Original
    successful-training evidence is inherited byte-for-byte from the strict
    posthoc export. Its cited receipts remain mandatory, not recomputed success.
    """
    root = Path(root).resolve(strict=True)
    failures = list(root.glob('failure*.json')) + list(root.glob('evaluations/*/failure*.json'))
    require(not failures, 'scan failure precedes any completion')
    returned = ex.read(root/'actual_return.json')
    require(type(returned.get('returncode')) is int and returned['returncode'] == 0,
            'actual successful scan process return required')
    require(returned.get('complete') == ex.receipt(root/'complete.json'), 'scan return/complete binding differs')
    launch = checked(returned['launch'])
    cmd = launch.get('command', [])
    require(len(cmd) == 8 and cmd[1:4] == ['-m', 'model_selection_v2.checkpoint_scan', 'controller']
            and cmd[4:6] == ['--root', str(root)] and cmd[6] == '--gpus', 'wrong completed controller command')
    completed = checked(returned['complete'])
    require(completed.get('status') == 'new_select_checkpoint_scan_and_export_complete'
            and completed.get('actual_completed_candidates') == len(UPDATES)
            and completed.get('new_select_pairs') == COUNTS['select'][0]
            and completed.get('no_training') is True, 'full registered scan required')
    require(returned['finished_unix'] >= completed['finished_unix'] >= launch['started_unix'],
            'scan completion precedes its launch or follows its return')
    prepared_ref = ex.receipt(root/'prepared.json')
    controller = ex.read(root/'controller_launch.json')
    require(controller.get('prepared') == prepared_ref, 'controller used a different preparation')
    prepared = checked(prepared_ref)
    refs = prepared['refs']; data = {k: checked(v) for k, v in refs.items()}
    request, protocol = data['request'], data['protocol']
    validate_protocol(protocol, verify_files=True)
    require(request['arm'] == 'B3' and request['updates'] == protocol['candidate_updates'] == UPDATES
            and prepared['task3_overlay_applied'] is False, 'only the original B3 mixed-SELECT scan is admitted')
    exported = checked(completed['export_complete'])
    require(exported.get('schema') == 'model-selection-posthoc-export-complete/2'
            and exported.get('status') == 'export_complete' and exported.get('arm') == 'B3'
            and exported.get('selection_kind') == ex.SELECTION_KIND
            and exported.get('protocol') == refs['protocol']
            and exported.get('candidate_manifest') == refs['candidates']
            and exported.get('original_training') == data['original'], 'posthoc export ancestry differs')
    candidates = data['candidates']
    require(candidates['schema'] == ex.CANDIDATE_SCHEMA and candidates['arm'] == 'B3'
            and candidates['source_controller_root'] == request['training_root']
            and candidates['source_binding_sha256'] == digest(data['original']['source_binding'])
            and [c['update'] for c in candidates['candidates']] == UPDATES, 'candidate inventory differs')
    evaluations = ex.read(root/'evaluations.json')
    require(set(evaluations) == {str(u) for u in UPDATES}, 'incomplete candidate evaluations')
    cache = dict(receipts={}, materialized={}); reports = []
    for candidate in candidates['candidates']:
        item = evaluations[str(candidate['update'])]
        reports.append(ex._verified_evaluation(item, candidate, protocol, refs['candidates']['sha256'], cache))
    result = select_checkpoint(protocol, reports)
    require(result == exported['selection'] and result['selected_update'] == completed['selected_update']
            == exported['selected_update'] == exported['export']['update'], 'replayed SELECT winner differs')
    require(ex.file_sha(exported['export']['path']) == exported['export']['sha256'], 'selected model bytes changed')
    for ref in [data['original']['execution_spec'], *data['original']['artifacts'].values()]:
        checked(ref)
    candidate = next(c for c in candidates['candidates'] if c['update'] == result['selected_update'])
    require(candidate['model_state_sha256'] == exported['export']['model_state_sha256'], 'winner tensor identity differs')
    value = dict(schema=ADOPTION_SCHEMA, arm='B3', selection_kind=ex.SELECTION_KIND,
        source_binding=data['original']['source_binding'], original_training=data['original'],
        model=exported['export'], update=result['selected_update'], candidate=candidate,
        protocol=refs['protocol'], candidate_manifest=refs['candidates'], evaluations=evaluations,
        selection=result, scan_complete=returned['complete'], scan_actual_return=ex.receipt(root/'actual_return.json'),
        export_complete=completed['export_complete'], runtime=request['runtime'], scan_request=refs['request'],
        matcher_retrained=False, task3_overlay_applied=False, head_training_started=False,
        real_used_for_matcher_selection=False, test_used_for_matcher_selection=False)
    return canonical(dict(value, sha256=digest(value)))


def check_adoption(adoption):
    from . import head_endpoint
    if adoption.get('schema') == head_endpoint.SCHEMA:
        return head_endpoint.check(adoption)
    require(adoption.get('sha256') == digest({k: v for k, v in adoption.items() if k != 'sha256'})
            and adoption.get('schema') == ADOPTION_SCHEMA and adoption.get('arm') == 'B3'
            and adoption.get('selection_kind') == ex.SELECTION_KIND,
            'explicit mixed-SELECT adoption required; never relabel as sim_best')
    for key in ('matcher_retrained', 'task3_overlay_applied', 'head_training_started',
                'real_used_for_matcher_selection', 'test_used_for_matcher_selection'):
        require(adoption.get(key) is False, 'unapproved change in selected-Matcher adoption: '+key)
    require(adoption['update'] == adoption['model']['update'] == adoption['candidate']['update']
            == adoption['selection']['selected_update'], 'selected update binding differs')


def check_terminal_adoption(adoption):
    """A self-digested metadata dictionary alone cannot authorize an import."""
    from . import head_endpoint
    if adoption.get('schema') == head_endpoint.SCHEMA:
        return head_endpoint.check_terminal(adoption)
    check_adoption(adoption)
    returned = checked(adoption['scan_actual_return'])
    require(type(returned.get('returncode')) is int and returned['returncode'] == 0
            and returned.get('complete') == adoption['scan_complete'], 'successful scan return required for head import')
    complete = checked(adoption['scan_complete']); exported = checked(adoption['export_complete'])
    require(complete.get('export_complete') == adoption['export_complete']
            and complete.get('status') == 'new_select_checkpoint_scan_and_export_complete'
            and complete.get('selected_update') == adoption['update']
            and exported.get('export') == adoption['model']
            and exported.get('selection') == adoption['selection']
            and exported.get('protocol') == adoption['protocol']
            and exported.get('candidate_manifest') == adoption['candidate_manifest']
            and exported.get('original_training') == adoption['original_training'], 'head import terminal ancestry differs')


def verify_model_record(saved, adoption, tree_sha):
    """Validate the new schema as itself, without manufacturing legacy metadata."""
    import torch
    from . import head_endpoint
    if adoption.get('schema') == head_endpoint.SCHEMA:
        return head_endpoint.verify_model(saved, adoption, tree_sha)
    check_adoption(adoption)
    binding = adoption['source_binding']; plan = binding['common_plan']
    require(saved.get('schema') == ex.EXPORT_SCHEMA and saved.get('selection_kind') == ex.SELECTION_KIND
            and saved.get('module') == saved.get('stage') == 'matcher' and saved.get('arm') == 'B3'
            and saved.get('binding') == binding and saved.get('order') == binding['order'] == 'curriculum'
            and saved.get('updates') == adoption['update']
            and saved.get('total_completed_updates') == plan['total_updates']
            and saved.get('exposures') == adoption['update']*plan['effective_batch'], 'new export identity differs')
    require(binding['module'] == plan['module'] == 'matcher' and binding.get('run_mode') == 'formal'
            and digest(plan) == binding['common_plan_sha256'], 'original formal Matcher binding required')
    require(saved.get('original_training') == adoption['original_training']
            and saved.get('committed_checkpoint') == adoption['candidate']['committed_checkpoint'],
            'original training or selected checkpoint ancestry differs')
    expected = dict(protocol=adoption['protocol'], candidate_manifest=adoption['candidate_manifest'],
                    evaluations=adoption['evaluations'], result=adoption['selection'])
    require(saved.get('posthoc_selection') == expected, 'selected export lost its new rule/protocol')
    for key in ('optimizer_imported', 'training_rng_included', 'matcher_retrained',
                'old_observations_rewritten', 'real_used_for_selection', 'test_used_for_selection'):
        require(saved.get(key) is False, 'new export must be a SELECT-only model: '+key)
    require(not any(k in saved for k in ('optimizer', 'rng', 'observation')), 'unexpected optimizer/RNG/old observation')
    state = saved.get('model', {})
    require(bool(state) and all(k.startswith(('matcher.', 'head.')) for k in state)
            and all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all()) for v in state.values()),
            'finite, explicit Matcher state required')
    require(tree_sha(state) == adoption['model']['model_state_sha256']
            == adoption['candidate']['model_state_sha256'], 'actual exported tensor identity differs')
    return binding, binding['model_spec']


def bind_head_validation(adoption):
    """Replace validation population only; no TRAIN/TEST writes or inference."""
    check_adoption(adoption)
    protocol = checked(adoption['protocol']); validate_protocol(protocol, verify_files=True)
    publication = checked(protocol['publication'])
    validation = {}
    for role, counts in COUNTS.items():
        ref = publication['outputs'][role]['manifest']; data = checked(ref)
        require(data.get('split') == role and data.get('real_used') is False and data.get('test_used') is False,
                'head validation requires the exact new SIM role')
        expected = {r['pair_id']: r for r in protocol['manifest_bindings'][role]['manifest']['entries']}
        entries = data['entries']; ids = [r['pair_id'] for r in entries]
        require(len(ids) == len(set(ids)) == counts[0] and set(ids) == set(expected), 'head validation identity differs')
        for row in entries:
            target = expected[row['pair_id']]
            require(type(row['label']) is bool and row['label'] == target['label']
                    and isinstance(row.get('recipe'), str) and bool(row['recipe']), 'head validation label/recipe differs')
            path = row.get('sample_path')
            if not path:
                require(bool(row.get('artifact_path')), 'materialized sample path required')
                path = Path(data.get('artifact_root', Path(ref['path']).parent))/row['artifact_path']
            require(Path(path).is_absolute() and ex.file_sha(path) == target['sample_sha256'],
                    'materialized validation bytes differ from published lineage')
        validation[role+'_mixed'] = dict(ref, pair_count=counts[0])
    from . import head_endpoint
    fixed_endpoint = adoption.get('schema') == head_endpoint.SCHEMA
    contract = dict(schema=CONTRACT_SCHEMA, status='passed', source_disjoint=True,
        validation_design=dict(kind='single_mixed', physical_samples=sum(c[0] for c in COUNTS.values())),
        validation=validation, publication=protocol['publication'],
        matcher_selection_protocol=None if fixed_endpoint else adoption['protocol'],
        head_epoch_selection='new_SIM_SELECT', head_threshold_calibration='new_SIM_CAL',
        head_rule='joint_f1_layout_ap+dunhuang_joint_only/1',
        train_changed=False, test_used=False, task3_overlay_applied=False,
        population_note='Augmentation views are correlated; counts are not independent manuscripts.')
    if fixed_endpoint:
        contract.update(validation_population_protocol=adoption['protocol'],
                        matcher_choice_basis=adoption['matcher_choice_basis'],
                        development_informed_matcher_choice=True)
    return canonical(contract)


def initialization_kind(adoption):
    from . import head_endpoint
    return (head_endpoint.INITIALIZATION if adoption.get('schema') == head_endpoint.SCHEMA
            else 'mixed_select_arm_matcher_new_head')


def make_components(plan, schedule, topology, runtime, architecture, geometry, adoption, validation_contract):
    """Import all selected Matcher tensors and initialize a fresh trainable head."""
    import torch
    check_terminal_adoption(adoption)
    policy = module('matcher_v2_v1.model_runtime', runtime)
    require(module('s7_consensus_v1.pose_consensus', runtime).REVISION == policy.PROPOSAL_REVISION,
            'original registered T16 implementation required')
    module('matcher_v2_v1.validation', runtime).validate_rule(plan)
    policy.check_schedule(plan, schedule)
    require(adoption['runtime'] == str(Path(runtime).resolve()) and schedule['arm'] == 'B3'
            and plan.record['module'] in ('scorer_patch', 'scorer_stats'), 'own-arm new head required')
    require(topology.world_size*topology.microbatch*topology.accumulate == plan.record['effective_batch'],
            'head effective batch differs')
    source_plan = adoption['source_binding']['common_plan']
    clean = lambda r: {k: v for k, v in r.items() if k not in ('module', 'selection_rule')}
    require(clean(plan.record) == clean(source_plan), 'head TRAIN, labels, seed, budget, loss or LR differs')
    require(validation_contract == bind_head_validation(adoption), 'head mixed SELECT/CAL contract differs')
    path = adoption['model']['path']
    require(ex.file_sha(path) == adoption['model']['sha256'], 'selected model bytes changed')
    hashing = module('curriculum_training_v1.checkpoint_io', runtime)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    _, spec = verify_model_record(saved, adoption, hashing.tree_sha)
    experiment = adoption['source_binding'].get('matcher_v2_experiment', {})
    require(experiment.get('schema') == policy.EXPERIMENT_SCHEMA and experiment.get('arm') == 'B3'
            and experiment.get('evaluation_policy') == policy.EVALUATION_POLICY, 'registered B3 architecture required')
    implementation = spec['matcher_implementation']
    require(spec['architecture'] == canonical(asdict(architecture))
            and spec['geometry'] == canonical(asdict(geometry))
            and implementation['architecture'] == spec['architecture']
            and implementation['model_seed'] == plan.record['model_seed']
            and implementation['matcher_v2'] == canonical(asdict(policy.config_for_arm('B3')))
            and spec['proposal_revision'] == policy.PROPOSAL_REVISION, 'head architecture/geometry differs')
    state = {k[len('matcher.'):]: v for k, v in saved['model'].items() if k.startswith('matcher.')}
    require(bool(state), 'selected Matcher tensors missing')
    spec_api = module('matcher_v2_v1.spec', runtime)
    matcher = spec_api.from_model_spec(implementation, frozen=True, state=state)
    cls = module('matcher_v2_v1.adapter', runtime).MatcherV2Adapter
    require(isinstance(matcher, cls) and not any(p.requires_grad for p in matcher.parameters())
            and not any(m.training for n, m in matcher.named_modules() if n), 'exact frozen Matcher import required')
    matcher.eval()
    require(hashing.tree_sha(matcher.state_dict()) == hashing.tree_sha(state), 'Matcher import changed tensors')
    variant = 'patch' if plan.record['module'] == 'scorer_patch' else 'stats'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan.record['head_seed'])
        head = module('binary_scorer_v1.head', runtime).BinaryClusterHead(variant)
    head.requires_grad_(True)
    model = module('binary_scorer_v1.model', runtime).BinaryConsensus(matcher, geometry, head=head)
    config = module('s7_consensus_v1.config', runtime).TrainingConfig(scorer_variant=variant,
        matcher_seed=plan.record['model_seed'], head_seed=plan.record['head_seed'], data_seed=plan.record['seed'],
        microbatch=topology.microbatch, world_size=topology.world_size, accumulate=topology.accumulate,
        workers_per_rank=topology.workers, learning_rate=plan.learning_rate_knots[0][1],
        weight_decay=plan.record['weight_decay'], gradient_clip_norm=plan.record['gradient_clip_norm'])
    wrapper = module('s7_consensus_v1.train', runtime).TrainModule(model, 'scorer', config)
    validation_config = replace(config, microbatch=min(8, config.microbatch))
    binding = module('curriculum_training_v1.runtime_plan', runtime).experiment_binding(
        plan, 'curriculum', adoption['model']['sha256'])
    binding['model_spec'] = canonical(dict(architecture=asdict(architecture), geometry=asdict(geometry),
        matcher_implementation=spec_api.model_spec(matcher, plan.record['model_seed']),
        proposal_revision=policy.PROPOSAL_REVISION, scorer_variant=variant,
        validation_microbatch=validation_config.microbatch,
        initial_matcher_state_sha256=hashing.tree_sha(matcher.state_dict()),
        initial_head_state_sha256=hashing.tree_sha(head.state_dict()), matcher_trainable_parameters=0,
        head_trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
        initialization=initialization_kind(adoption), selected_matcher=adoption,
        optimizer_imported=False, old_head_imported=False))
    binding['matcher_v2_experiment'] = canonical(dict(schema=policy.EXPERIMENT_SCHEMA, arm='B3',
        schedule=schedule, evaluation_policy=policy.EVALUATION_POLICY))
    binding['mixed_sim_head_experiment'] = canonical(dict(schema=HEAD_SCHEMA,
        selected_matcher_adoption_sha256=adoption['sha256'], validation_contract=validation_contract,
        validation_contract_sha256=digest(validation_contract), original_train_and_labels_preserved=True,
        task3_overlay_applied=False, rotation_ensemble_in_training=False))
    return dict(model=model, module=wrapper, config=config, validation_config=validation_config,
        collate=module('s7_consensus_v1.data', runtime).collate, binding=binding)
