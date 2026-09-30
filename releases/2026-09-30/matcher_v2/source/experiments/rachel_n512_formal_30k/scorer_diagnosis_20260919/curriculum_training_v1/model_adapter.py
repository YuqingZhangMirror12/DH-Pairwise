"""Actual immutable loader/model adapters for the update-based curriculum loop.

No fixed 24K epoch engine, old optimizer, or old Scorer is imported. Dedicated
launch, geometry/data admission and genuine GPU gates remain separate.
"""
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path

import torch

from .catalog import MATCHER_INPUTS, TARGETS, tensor_digest, legacy_numerical_digest
from .checkpoint_io import file_sha, tree_sha
from .exposure import STAGES, SampleRef, canonical_catalog, digest
from .runtime_plan import experiment_binding
from .validation_adapter import canonical, validate_rule

BASE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_module(name, source_root):
    module = importlib.import_module(name)
    require(Path(source_root).resolve() in Path(module.__file__).resolve().parents,
            'imported a different model/loader source: ' + name)
    return module


class AdmittedDataset:
    """One canonical index space shared by C/M and every loader rank.

    Hash actual materialized tensors on first read per worker. Do not retain
    image tensors in RAM or transform/mirror them online.
    """
    def __init__(self, admission_path, expected_sha256, ledger, source_root, loader=None):
        self.path = Path(admission_path)
        require(file_sha(self.path) == expected_sha256, 'admission file changed')
        admission = json.loads(self.path.read_text())
        require(admission['schema'] == 'curriculum-data-admission/1'
                and admission['status'] == 'passed' and admission['gpu_used'] is False,
                'completed data admission required')
        catalog = canonical_catalog(SampleRef(**r) for r in admission['catalog'])
        require(catalog == ledger.catalog and digest([asdict(r) for r in catalog]) == admission['catalog_sha256'],
                'dataset/ledger index order or membership differs')
        all_rows = {}
        for stage in STAGES:
            spec = admission['training_manifests'][stage]; path = Path(spec['path'])
            require(file_sha(path) == spec['sha256'], 'admitted manifest changed')
            record = json.loads(path.read_text())
            require(record['schema'] == 'curriculum-admitted-train/1' and record['stage'] == stage
                    and record['split'] == 'train' and record['online_augmentation'] is False,
                    'wrong stage/online augmentation in admitted manifest')
            for row in record['entries']:
                key = (stage, row['pair_id'])
                require(key not in all_rows, 'duplicate admitted dataset index')
                all_rows[key] = row
        require(set(all_rows) == {(r.stage, r.pair_id) for r in catalog}, 'manifest/catalog membership differs')
        self.entries = []
        for ref in catalog:
            row = all_rows[(ref.stage, ref.pair_id)]
            require(row['label'] == ref.label and type(row['label']) is bool
                    and row['sample_path'] == ref.sample_path and row['sample_sha256'] == ref.sample_sha256
                    and row['source_base_key'] == ref.source_base_key
                    and row['actual_matcher_input_sha256'] == ref.model_input_sha256
                    and isinstance(row['recipe'], str) and row['recipe'], 'admitted reference differs from ledger')
            self.entries.append(row)
        self.loader = loader or bound_module(
            'staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset', source_root).load_sample
        self.verified = set(); self.admission_sha256 = expected_sha256

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]; path = Path(entry['sample_path'])
        require(path.is_absolute(), 'absolute materialized path required')
        if index not in self.verified:
            require(file_sha(path) == entry['sample_sha256'], 'actual admitted sample changed')
        sample, report = self.loader(path)
        require(sample.pair_id == entry['pair_id'] and float(sample.label) in (0., 1.)
                and bool(sample.label) == entry['label'], 'actual loader identity/label differs')
        if index not in self.verified:
            require(tensor_digest(sample, MATCHER_INPUTS) == entry['actual_matcher_input_sha256'],
                    'actual six Matcher inputs differ from admission')
            target = digest(dict(materialized_targets=tensor_digest(sample, TARGETS),
                                 precise_recipe=entry['recipe'] == 'clean'))
            require(target == entry['effective_training_target_sha256']
                    and legacy_numerical_digest(sample) == entry['legacy_pixel_audit_numerical_sha256'],
                    'actual training supervision differs from admission')
            self.verified.add(index)
        return sample, report, entry


def make_components(plan, order, topology, source_root, architecture, geometry, selected=None, cache=None):
    """Construct actual Matcher/head/TrainModule, without moving it to a GPU."""
    validate_rule(plan)
    require(topology.world_size * topology.microbatch * topology.accumulate == plan.record['effective_batch'],
            'registered global batch differs from topology')
    require(architecture.feature_dim == 96, 'registered Matcher width is96')
    prefix = BASE + 's7_consensus_v1.'
    config_api = bound_module(prefix + 'config', source_root)
    matcher_api = bound_module(prefix + 'scratch_matcher', source_root)
    train_api = bound_module(prefix + 'train', source_root)
    data_api = bound_module(prefix + 'data', source_root)
    policy = bound_module(prefix + 'pose_consensus', source_root)
    model_api = bound_module(BASE + 'binary_scorer_v1.model', source_root)
    head_api = bound_module(BASE + 'binary_scorer_v1.head', source_root)
    require(policy.REVISION == 'native-hypothesis-complete-link-union/1-diameter16',
            'registered fixed16 complete-link union builder required')
    training_matcher = plan.record['module'] == 'matcher'
    require((selected is None) == training_matcher, 'Matcher random start / selected frozen head start differs')
    require(not training_matcher or cache is None, 'updating Matcher cannot use frozen candidate cache')
    binding = experiment_binding(plan, order, None if selected is None else selected['sha256'])
    matcher = matcher_api.fresh_matcher(architecture, plan.record['model_seed'])
    imported = None
    if not training_matcher:
        path = Path(selected['path'])
        require(file_sha(path) == selected['sha256'], 'selected curriculum Matcher export changed')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        source = selected['source_binding']
        require(saved['schema'] == 'curriculum-model-export/1' and saved['binding'] == source
                and source['module'] == 'matcher' and source['order'] == 'curriculum'
                and saved['selection_kind'] == 'sim_best' and saved['updates'] == selected['updates'] > 0,
                'not the selected curriculum Matcher')
        for key in ('data_admission_sha256', 'geometry_sha256', 'baseline_sources_sha256', 'model_seed'):
            require(source['common_plan'][key] == plan.record[key], 'head source differs: ' + key)
        require(source['model_spec']['architecture'] == canonical(asdict(architecture))
                and source['model_spec']['geometry'] == canonical(asdict(geometry)),
                'head architecture/geometry differs from selected Matcher')
        state = {k[len('matcher.'):]: v for k, v in saved['model'].items() if k.startswith('matcher.')}
        require(bool(state), 'selected export has no Matcher state')
        matcher.load_state_dict(state, strict=True)
        require(tree_sha(matcher.state_dict()) == tree_sha(state), 'Matcher tensor import differs')
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
    stage = 'matcher' if training_matcher else 'scorer'
    wrapper = train_api.TrainModule(model, stage, config, cache=cache)
    # The single-GPU light heads may train micro32, but the registered frozen
    # inference/validation path stays micro8. Do not accidentally enlarge it.
    validation_config = replace(config, microbatch=min(8, config.microbatch))
    binding['model_spec'] = canonical(dict(architecture=asdict(architecture), geometry=asdict(geometry),
        proposal_revision=policy.REVISION, scorer_variant=variant,
        validation_microbatch=validation_config.microbatch,
        initial_matcher_state_sha256=tree_sha(matcher.state_dict()),
        initial_head_state_sha256=tree_sha(head.state_dict()),
        matcher_trainable_parameters=sum(p.numel() for p in matcher.parameters() if p.requires_grad),
        head_trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
        initialization='shared_random_seed' if training_matcher else 'selected_curriculum_matcher_new_head',
        selected_matcher=imported, optimizer_imported=False, old_head_imported=False))
    return dict(model=model, module=wrapper, config=config, validation_config=validation_config,
                collate=data_api.collate, binding=binding)
