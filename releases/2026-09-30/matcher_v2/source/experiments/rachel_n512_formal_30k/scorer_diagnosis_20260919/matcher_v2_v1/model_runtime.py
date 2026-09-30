"""Architecture-aware construction for B1--B3 and their two fresh light heads.

Uses the existing losses, T16 builder and Scorer interfaces. Export loading here
checks tensor/architecture identity, NOT process success: the launcher must first
verify the complete training/selection/return receipts before final evaluation.
"""
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ..curriculum_training_v1.checkpoint_io import file_sha, tree_sha
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.model_adapter import BASE, bound_module, require
from ..curriculum_training_v1.runtime_plan import experiment_binding
from ..curriculum_training_v1.validation_adapter import canonical
from .network import MatcherV2Config
from .validation import validate_rule

EXPERIMENT_SCHEMA = 'matcher-v2-production-arm/1'
PROPOSAL_REVISION = 'native-hypothesis-complete-link-union/1-diameter16'
EVALUATION_POLICY = 'sim-native+dunhuang-development+turufan-final-only/1'


def config_for_arm(arm):
    require(arm in ('B1', 'B2', 'B3'), 'only B1/B2/B3 are registered; ablations require a later decision')
    return MatcherV2Config(enabled=arm != 'B1')


def check_schedule(plan, schedule):
    config_for_arm(schedule.get('arm'))
    require(schedule.get('schema') == 'matcher-v2-runtime-schedule/1'
            and schedule.get('module') == plan.record['module']
            and schedule.get('runtime_plan_sha256') == plan.sha256
            and schedule.get('runtime_ledger_sha256') == plan.record['ledger_sha256']
            and schedule.get('actual_updates') == plan.record['total_updates']
            and schedule.get('actual_learning_rate_knots') == plan.record['learning_rate_knots']
            and schedule.get('actual_validation_updates') == plan.record['validation_updates'],
            'compiled arm/schedule binding differs from the runtime plan')
    require(schedule.get('original_exposures_and_order_preserved') is True
            and schedule.get('source_geometry_unchanged') is True,
            'original exposure/geometry contract required')


def check_export(saved):
    require(saved.get('schema') == 'curriculum-model-export/1', 'bound model-only export required')
    binding = saved['binding']; plan = binding['common_plan']
    require(digest(plan) == binding['common_plan_sha256']
            and saved['module'] == binding['module'] == plan['module']
            and saved['order'] == binding['order'] == 'curriculum'
            and type(saved['updates']) is int and saved['updates'] > 0
            and saved['updates'] <= plan['total_updates'], 'export identity/update differs')
    experiment = binding.get('matcher_v2_experiment', {})
    require(experiment.get('schema') == EXPERIMENT_SCHEMA
            and experiment.get('evaluation_policy') == EVALUATION_POLICY,
            'explicit new-arm architecture/evaluation binding required')
    cfg = config_for_arm(experiment.get('arm'))
    spec = binding['model_spec']; serialized = spec.get('matcher_implementation', {})
    require(serialized.get('matcher_v2') == canonical(asdict(cfg))
            and serialized.get('model_seed') == plan['model_seed']
            and serialized.get('architecture') == spec['architecture']
            and spec['proposal_revision'] == PROPOSAL_REVISION,
            'serialized Matcher architecture differs from its registered arm')
    schedule = experiment['schedule']
    require(schedule['arm'] == experiment['arm']
            and schedule['runtime_plan_sha256'] == binding['common_plan_sha256']
            and schedule['runtime_ledger_sha256'] == plan['ledger_sha256'],
            'export schedule belongs to another experiment')
    require(bool(saved.get('model'))
            and all(k.startswith(('matcher.', 'head.')) for k in saved['model']),
            'unexpected model component or missing model state')
    for value in saved['model'].values():
        require(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()),
                'non-finite/non-tensor export state')
    return binding, spec


def load_export_matcher(saved, source_root):
    """Explicit v2 constructor for native evaluation and selected-head imports."""
    _, spec = check_export(saved)
    api = bound_module(BASE + 'matcher_v2_v1.spec', source_root)
    state = {k[len('matcher.'):]: value for k, value in saved['model'].items() if k.startswith('matcher.')}
    require(bool(state), 'selected export has no Matcher state')
    matcher = api.from_model_spec(spec['matcher_implementation'], frozen=True, state=state)
    require(tree_sha(matcher.state_dict()) == tree_sha(state), 'Matcher tensor import differs')
    return matcher


def load_export_scorer(saved, source_root):
    """Frozen full Scorer; never load a random/inactive head from a Matcher run."""
    binding, spec = check_export(saved)
    require(binding['module'] in ('scorer_patch', 'scorer_stats'), 'trained Scorer export required')
    variant = 'patch' if binding['module'] == 'scorer_patch' else 'stats'
    require(spec['scorer_variant'] == variant, 'trained Scorer variant differs')
    geometry_api = bound_module(BASE + 's7_consensus_v1.compatibility', source_root)
    head_api = bound_module(BASE + 'binary_scorer_v1.head', source_root)
    model_api = bound_module(BASE + 'binary_scorer_v1.model', source_root)
    matcher = load_export_matcher(saved, source_root)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(binding['common_plan']['head_seed'])
        head = head_api.BinaryClusterHead(variant)
    model = model_api.BinaryConsensus(matcher, geometry_api.CompatibilityConfig(**spec['geometry']), head=head)
    model.load_state_dict(saved['model'], strict=True)
    require(tree_sha(model.state_dict()) == tree_sha(saved['model']), 'full Scorer tensor import differs')
    model.requires_grad_(False); model.eval()
    return model


def make_components(plan, schedule, topology, source_root, architecture, geometry, selected=None, cache=None):
    """Construct on CPU; no CUDA allocation, data evaluation or optimizer import."""
    validate_rule(plan); check_schedule(plan, schedule)
    arm = schedule['arm']; cfg = config_for_arm(arm)
    require(topology.world_size * topology.microbatch * topology.accumulate == plan.record['effective_batch'],
            'registered global batch differs from topology')
    require(architecture.feature_dim == 96, 'registered Matcher width is 96')
    prefix = BASE + 's7_consensus_v1.'
    config_api = bound_module(prefix + 'config', source_root)
    train_api = bound_module(prefix + 'train', source_root)
    data_api = bound_module(prefix + 'data', source_root)
    policy = bound_module(prefix + 'pose_consensus', source_root)
    model_api = bound_module(BASE + 'binary_scorer_v1.model', source_root)
    head_api = bound_module(BASE + 'binary_scorer_v1.head', source_root)
    matcher_api = bound_module(BASE + 'matcher_v2_v1.adapter', source_root)
    spec_api = bound_module(BASE + 'matcher_v2_v1.spec', source_root)
    require(policy.REVISION == PROPOSAL_REVISION, 'registered T16 complete-link builder required')
    training_matcher = plan.record['module'] == 'matcher'
    require((selected is None) == training_matcher, 'random Matcher / selected frozen head start differs')
    require(not training_matcher or cache is None, 'updating Matcher cannot use a frozen candidate cache')
    binding = experiment_binding(plan, 'curriculum', None if selected is None else selected['sha256'])
    imported = None
    if training_matcher:
        matcher = matcher_api.fresh_matcher_v2(architecture, config=cfg, seed=plan.record['model_seed'])
    else:
        path = Path(selected['path'])
        require(file_sha(path) == selected['sha256'], 'selected Matcher export changed')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        source, source_spec = check_export(saved)
        require(source == selected['source_binding'] and source['module'] == 'matcher'
                and saved['selection_kind'] == 'sim_best' and saved['updates'] == selected['updates']
                and source['matcher_v2_experiment']['arm'] == arm,
                'head must use the SIM-selected Matcher from its own arm')
        for key in ('data_admission_sha256', 'geometry_sha256', 'baseline_sources_sha256', 'model_seed',
                    'head_seed', 'ledger_sha256', 'total_updates', 'seed', 'learning_rate_knots'):
            require(source['common_plan'][key] == plan.record[key], 'head source differs: ' + key)
        require(source_spec['architecture'] == canonical(asdict(architecture))
                and source_spec['geometry'] == canonical(asdict(geometry)), 'head architecture/geometry differs')
        matcher = load_export_matcher(saved, source_root)
        imported = canonical(selected)
    variant = 'stats' if plan.record['module'] == 'scorer_stats' else 'patch'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan.record['head_seed'])
        head = head_api.BinaryClusterHead(variant)
    matcher.set_frozen(not training_matcher); head.requires_grad_(not training_matcher)
    model = model_api.BinaryConsensus(matcher, geometry, head=head)
    config = config_api.TrainingConfig(scorer_variant=variant,
        matcher_seed=plan.record['model_seed'], head_seed=plan.record['head_seed'], data_seed=plan.record['seed'],
        microbatch=topology.microbatch, world_size=topology.world_size, accumulate=topology.accumulate,
        workers_per_rank=topology.workers, learning_rate=plan.learning_rate_knots[0][1],
        weight_decay=plan.record['weight_decay'], gradient_clip_norm=plan.record['gradient_clip_norm'])
    wrapper = train_api.TrainModule(model, 'matcher' if training_matcher else 'scorer', config, cache=cache)
    validation_config = replace(config, microbatch=min(8, config.microbatch))
    binding['model_spec'] = canonical(dict(architecture=asdict(architecture), geometry=asdict(geometry),
        matcher_implementation=spec_api.model_spec(matcher, plan.record['model_seed']),
        proposal_revision=policy.REVISION, scorer_variant=variant,
        validation_microbatch=validation_config.microbatch,
        initial_matcher_state_sha256=tree_sha(matcher.state_dict()), initial_head_state_sha256=tree_sha(head.state_dict()),
        matcher_trainable_parameters=sum(p.numel() for p in matcher.parameters() if p.requires_grad),
        head_trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
        initialization='shared_random_seed' if training_matcher else 'selected_arm_matcher_new_head',
        selected_matcher=imported, optimizer_imported=False, old_head_imported=False))
    binding['matcher_v2_experiment'] = canonical(dict(schema=EXPERIMENT_SCHEMA, arm=arm,
        schedule=schedule, evaluation_policy=EVALUATION_POLICY))
    return dict(model=model, module=wrapper, config=config, validation_config=validation_config,
                collate=data_api.collate, binding=binding)
