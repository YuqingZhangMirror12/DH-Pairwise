"""Frozen G0/G1 endpoints through the original TEST/REAL/OOD evaluator.

Only completed C16 training is eligible; C8 is a retained fixed-budget endpoint.
Both operating points come from the bound clean SIMVAL report. No target-domain
threshold fitting or candidate reranking is added. One original Matcher forward
supplies the unchanged assignment/Layout; the independent Scorer supplies scores.
"""
from dataclasses import asdict, dataclass
import functools
import json
from pathlib import Path

import torch

from . import train, model as architecture, data as auxiliary_data
from ..matched_only import cache as source_cache, data as cache_data
from ..pair_grid_readout import model as grid_readout
from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as original
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation_depth_controls import continue_depth_controls as private


@dataclass(frozen=True)
class EndpointOutput(architecture.FeatureAdaptationOutput):
    score_details: dict


class EndpointModel(architecture.ScorerFeatureAdaptationModel):
    """Expose raw Scorer evidence without changing the architecture/state keys."""
    def forward(self, *inputs):
        output = super().forward(*inputs)
        return EndpointOutput(**vars(output), score_details=dict(
            raw_similarity=output.raw_similarity, calibrated_logit=output.calibrated_logit,
            final_logit=output.fused_logit, training_valid=output.training_valid))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def load_frozen_model(root, selection, *, budget=16):
    if budget not in (8, 16) or selection not in original.SELECTIONS:
        raise ValueError('registered C8/C16 and SIMVAL selections only')
    root = Path(root).resolve(strict=True)
    status = json.loads((root / 'status.json').read_text())
    freeze_path = root / 'freezes' / ('c%d.json' % budget)
    freeze = json.loads(freeze_path.read_text())
    ident = freeze.get('identity', {})
    negatives = ident.get('negative_counts', [])
    if (status.get('status') != 'complete' or status.get('completed_segments') != 64
            or status.get('completed_head_epochs') != 16
            or canonical(status.get('identity')) != canonical(ident)
            or freeze.get('schema') != train.SCHEMA or freeze.get('status') != 'complete_endpoint'
            or freeze.get('budget_head_epochs') != budget or freeze.get('real_ood_used') is not False
            or freeze.get('selection_population') != 'clean SIMVAL3000 only'
            or freeze.get('primary_selection') != 'fixed_endpoint'
            or ident.get('schema') != train.SCHEMA or ident.get('arm') not in ('G0', 'G1')
            or ident.get('source_checkpoint_sha256') != source_cache.SOURCE_SHA
            or ident.get('source_matcher_epochs') != 12 or ident.get('head_epochs') != 16
            or ident.get('ordinary_pairs_per_epoch') != 24000 or ident.get('ordinary_physical_batch') != 16
            or ident.get('auxiliary_groups_per_epoch') != 1470 or ident.get('auxiliary_pairs_per_epoch') != 3028
            or ident.get('auxiliary_bce') is not False or ident.get('real_ood_used') is not False
            or len(negatives) != 1470 or any(type(n) is not int or n not in (1, 2) for n in negatives)
            or sum(negatives) != 1558
            or ident.get('implementation_sha256') != train.implementation_binding()):
        raise ValueError('requires completed unchanged G0/G1 C16 training and SIMVAL-only freeze')
    if status.get('exposures') != train.exposure_ledger(64, negatives):
        raise ValueError('completed ordinary/auxiliary exposure ledger differs')
    key = 'fixed_endpoint' if selection == 'fixed_epoch' else selection
    chosen = freeze['selections'][key]
    epoch = chosen.get('head_epoch')
    if (type(epoch) is not int or not 1 <= epoch <= budget
            or selection == 'fixed_epoch' and epoch != budget):
        raise ValueError('checkpoint outside declared head-training budget')
    checkpoint = root / ('head_epoch_%03d.pt' % epoch)
    if chosen.get('checkpoint') != checkpoint.name or cache_data.sha(checkpoint) != chosen.get('checkpoint_sha256'):
        raise ValueError('checkpoint ownership/hash differs from SIMVAL freeze')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (saved.get('schema') != train.SCHEMA or canonical(saved.get('identity')) != canonical(ident)
            or saved.get('completed_segments') != 4 * epoch
            or saved.get('exposures') != train.exposure_ledger(4 * epoch, negatives)
            or saved.get('role') != 'epoch_anchor' or saved.get('matcher_updated') is not False
            or saved.get('phase') != 'classifier' or saved.get('formal_training_counted') is not True):
        raise ValueError('selected checkpoint identity/ordinary/auxiliary budget is inconsistent')
    validation_path = root / ('validation_head_%03d.json' % epoch)
    if (chosen.get('validation') != validation_path.name
            or chosen.get('validation_sha256') != cache_data.sha(validation_path)):
        raise ValueError('selected SIMVAL report differs from frozen report hash')
    report = json.loads(validation_path.read_text())
    operating = report['operating_points']
    op = 'recall_95' if selection == 'recall95' else 'max_f1'
    if (report.get('sample_count') != 3000 or report.get('positive_count') != 1500
            or chosen.get('primary_pair_threshold') != operating['thresholds'][op]
            or canonical(chosen.get('operating_points')) != canonical(operating)):
        raise ValueError('SIMVAL operating point differs from frozen selection')

    source = source_cache.source_checkpoint()
    source_model = train.source_training.load_decoupled_checkpoint(source)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(ident['head_seed'])
        model = EndpointModel(source_model, feature_trainable=ident['arm'] == 'G1')
    del source_model, source
    initial_frozen = train.frozen_digests(model)
    if (initial_frozen != ident.get('frozen_digests')
            or canonical(model.metadata()) != canonical(saved.get('model_metadata'))
            or canonical(model.metadata()) != canonical(ident.get('model'))):
        raise ValueError('Scorer architecture or original frozen Matcher/G0 stem differs')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    if train.frozen_digests(model) != initial_frozen:
        raise ValueError('loaded checkpoint changed frozen original Matcher/G0 stem')
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('nonfinite trained Scorer state')
    model.eval().requires_grad_(False)
    thresholds = {name: operating['thresholds']['max_f1'] for name in original.core.BRANCHES}
    receipt = dict(training_run=str(root), budget=12+budget, head_budget=budget,
        selection=selection, epoch=12+epoch, head_epoch=epoch, seed=ident['data_seed'],
        freeze_path=str(freeze_path), freeze_sha256=cache_data.sha(freeze_path),
        checkpoint_path=str(checkpoint), checkpoint_sha256=chosen['checkpoint_sha256'],
        source_matcher_checkpoint=str(source_cache.SOURCE), source_matcher_sha256=source_cache.SOURCE_SHA,
        model_config=asdict(model.config), architecture='independent_scorer_ca_lme:' + ident['arm'],
        sampling='original512', classifier_thresholds=thresholds, operating_points=operating,
        winner_record=chosen, training_identity=ident, model_design=model.metadata(),
        classifier_only_pair_bce=False, classifier_loss='ordinary PairBCE plus .3 same-anchor raw-similarity ranking',
        auxiliary_bce=False, exposures=saved['exposures'], coarse_is_untrained_diagnostic=True,
        local_and_fused_are_same_single_classifier=True, test_or_real_used_for_fit=False,
        ood_used_for_fit=False, endpoint_only=True, matcher_layout_unchanged=True,
        score_details_fields=['raw_similarity', 'calibrated_logit', 'final_logit', 'training_valid'],
        evaluation_adapter_sha256=cache_data.sha(__file__))
    return model, receipt


def inference_runtime(device):
    runtime = original.inference_runtime(device)
    runtime['source_sha256'] = dict(runtime['source_sha256'])
    for module in (architecture, auxiliary_data, train):
        runtime['source_sha256']['scorer_feature_adaptation_v1/' + Path(module.__file__).name] = cache_data.sha(module.__file__)
    runtime['source_sha256']['scorer_feature_adaptation_v1/evaluate.py'] = cache_data.sha(__file__)
    runtime['source_sha256']['pair_grid_readout/model.py'] = cache_data.sha(grid_readout.__file__)
    runtime['feature_adaptation_runtime_binding'] = train.SCHEMA
    return runtime


def run(args):
    if args.device != 'cuda:0':
        raise ValueError('formal endpoint CLI requires the guarded remote cuda:0')
    evaluate = private.private_function(original.run,
        load_frozen_model=functools.partial(load_frozen_model, budget=args.head_budget),
        inference_runtime=inference_runtime)
    with train.prior.lock_owner.gpu_lock():
        return evaluate(args)


if __name__ == '__main__':
    parser = original.parser()
    parser.add_argument('--head-budget', type=int, choices=(8, 16), default=16)
    run(parser.parse_args())
