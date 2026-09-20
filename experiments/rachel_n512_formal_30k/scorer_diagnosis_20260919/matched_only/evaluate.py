"""Frozen five-arm endpoints, reusing the original TEST/REAL/OOD evaluator.

No threshold fitting on held-out populations. Fixed C8/C16 and auxiliary
SIMVAL-selected checkpoints can only be read after full C16 training completes.
The selected candidate group's pose is a diagnostic, never a silent replacement
for the unchanged production decoder. Original evaluator records score_details.
"""
from dataclasses import asdict
import functools
import json
from pathlib import Path

import torch

from . import cache, data, inference, model as architecture, train, stage_cache, candidate_groups
from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as original
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation_depth_controls import continue_depth_controls as private


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def load_frozen_model(root, selection, *, budget=16):
    if budget not in (8, 16) or selection not in original.SELECTIONS:
        raise ValueError('registered C8/C16 and SIMVAL selections only')
    root = Path(root).resolve(strict=True)
    status = json.loads((root / 'status.json').read_text())
    path = root / 'freezes' / ('c%d.json' % budget)
    freeze = json.loads(path.read_text())
    ident = freeze.get('identity', {})
    if (status.get('status') != 'complete' or status.get('completed_segments') != 64
            or freeze.get('schema') != train.SCHEMA or freeze.get('status') != 'complete_endpoint'
            or freeze.get('budget_head_epochs') != budget or freeze.get('real_ood_used') is not False
            or freeze.get('selection_population') != 'clean SIMVAL3000 only'
            or freeze.get('primary_selection') != 'fixed_endpoint'
            or ident.get('source_checkpoint_sha256') != cache.SOURCE_SHA
            or ident.get('matcher_frozen') is not True or ident.get('arm') not in train.ARMS
            or ident.get('classifier_epochs') != 16 or ident.get('training_count') != 24000
            or ident.get('validation_count') != 3000 or ident.get('real_ood_used') is not False
            or ident.get('implementation_sha256') != train.implementation_binding()):
        raise ValueError('requires completed unchanged five-arm C16 training and SIMVAL-only freeze')
    key = 'fixed_endpoint' if selection == 'fixed_epoch' else selection
    chosen = freeze['selections'][key]
    epoch = chosen.get('head_epoch')
    if (type(epoch) is not int or not 1 <= epoch <= budget
            or selection == 'fixed_epoch' and epoch != budget):
        raise ValueError('checkpoint outside declared head-training budget')
    checkpoint = root / ('head_epoch_%03d.pt' % epoch)
    if chosen.get('checkpoint') != checkpoint.name or data.sha(checkpoint) != chosen.get('checkpoint_sha256'):
        raise ValueError('checkpoint ownership/hash differs from SIMVAL freeze')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (saved.get('schema') != train.SCHEMA or canonical(saved.get('identity')) != canonical(ident)
            or saved.get('head_epoch') != epoch or saved.get('completed_segments') != 4 * epoch
            or saved.get('classifier_pair_exposures') != 24000 * epoch
            or saved.get('optimizer_updates') != 1500 * epoch
            or saved.get('matcher_updated') is not False or saved.get('phase') != 'classifier'
            or saved.get('formal_training_counted') is not True):
        raise ValueError('selected head checkpoint identity/budget is inconsistent')
    validation_path = root / ('validation_head_%03d.json' % epoch)
    if (chosen.get('validation') != validation_path.name
            or chosen.get('validation_sha256') != data.sha(validation_path)):
        raise ValueError('selected SIMVAL report differs from frozen report hash')
    report = json.loads(validation_path.read_text())
    operating = report['operating_points']
    op = 'recall_95' if selection == 'recall95' else 'max_f1'
    if (report.get('sample_count') != 3000 or report.get('positive_count') != 1500
            or chosen.get('primary_pair_threshold') != operating['thresholds'][op]
            or canonical(chosen.get('operating_points')) != canonical(operating)):
        raise ValueError('SIMVAL operating point differs from frozen selection')
    head = train.make_scorer(ident['arm'], seed=ident['head_seed'])
    if canonical(head.metadata()) != canonical(saved.get('model_metadata')):
        raise ValueError('Scorer architecture differs from saved head')
    head.load_state_dict(saved['model_state_dict'], strict=True)
    if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
        raise ValueError('nonfinite trained head state')
    source = cache.source_checkpoint()
    base = cache.old.load_decoupled_checkpoint(source).base_model
    adapter = inference.FrozenMatchedInference(base, head)
    # Historical summaries expect three branch thresholds; only the single new
    # fused score is reported as a competing classifier by original.run.
    thresholds = {name: operating['thresholds']['max_f1'] for name in original.core.BRANCHES}
    receipt = dict(training_run=str(root), budget=12+budget, head_budget=budget,
        selection=selection, epoch=12+epoch, head_epoch=epoch, seed=ident['data_seed'],
        freeze_path=str(path), freeze_sha256=data.sha(path), checkpoint_path=str(checkpoint),
        checkpoint_sha256=chosen['checkpoint_sha256'], source_matcher_checkpoint=str(cache.SOURCE),
        source_matcher_sha256=cache.SOURCE_SHA, model_config=asdict(base.config),
        architecture='fresh_matched_only_ca:' + ident['arm'], sampling='original512',
        classifier_thresholds=thresholds, operating_points=operating, winner_record=chosen,
        training_identity=ident, model_design=adapter.metadata(), classifier_only_pair_bce=True,
        coarse_is_untrained_diagnostic=True, local_and_fused_are_same_single_classifier=True,
        test_or_real_used_for_fit=False, ood_used_for_fit=False, endpoint_only=True,
        selected_group_pose_role='diagnostic only; original production decoder unchanged',
        evaluation_adapter_sha256=data.sha(__file__))
    return adapter, receipt


def inference_runtime(device):
    runtime = original.inference_runtime(device)
    runtime['source_sha256'] = dict(runtime['source_sha256'])
    modules = (inference, architecture, train, stage_cache, candidate_groups, data, cache)
    runtime['source_sha256'].update({'matched_only/' + Path(m.__file__).name: data.sha(m.__file__) for m in modules})
    runtime['source_sha256']['matched_only/evaluate.py'] = data.sha(__file__)
    runtime['matched_runtime_binding'] = inference.SCHEMA
    return runtime


def run(args):
    if args.device != 'cuda:0':
        raise ValueError('formal endpoint CLI requires the guarded remote cuda:0')
    evaluate = private.private_function(original.run,
        load_frozen_model=functools.partial(load_frozen_model, budget=args.head_budget),
        inference_runtime=inference_runtime)
    with train.lock_owner.gpu_lock():
        return evaluate(args)


if __name__ == '__main__':
    parser = original.parser()
    parser.add_argument('--head-budget', type=int, choices=(8, 16), default=16)
    run(parser.parse_args())
