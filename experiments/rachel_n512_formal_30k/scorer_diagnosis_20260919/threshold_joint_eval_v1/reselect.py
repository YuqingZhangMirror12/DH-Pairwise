"""Read archived frozen-control weights; only real_cal/real_select reach forward."""
from pathlib import Path
import time
import traceback

import torch

from consensus_joint_eval_common import frozen as base
from .contracts import (PLAN_SHA, read, sha, save, validate_archive, matcher_change,
                        choose_real_best)
from .loading import load_frozen_control, freeze_model


def run(args, helper):
    model, _, provenance, selection = load_frozen_control(args.root, args.reference)
    if sha(args.real_plan) != PLAN_SHA:
        raise ValueError('registered real split required')
    development_binding = helper.bind_plan(args.real_plan)
    development = helper.RealDevelopment(args.real_plan, development_binding)
    stage = Path(args.root)/'formal_scratch_fixed/scorer'
    epochs = list(range(0, selection['actual_epochs'] + 1, 2))
    # Register ALL epoch inputs before any score inspection. Missing archives
    # fail closed; never replace them with only the surviving/best checkpoints.
    files = []
    matcher_origin = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                      if k.startswith('matcher.')}
    for epoch in epochs:
        weights = stage/f'epoch_{epoch:03d}_weights.pt'
        validation = stage/f'epoch_{epoch:03d}_validation.json'
        cp = torch.load(weights, map_location='cpu', weights_only=False)
        validate_archive(cp, read(validation), selection['binding'], epoch)
        matcher_change(matcher_origin, cp['model'], expect_updated=False)
        files.append(dict(epoch=epoch, checkpoint=str(weights), checkpoint_sha256=sha(weights),
                          validation=str(validation), validation_sha256=sha(validation)))
        del cp
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    identity = dict(schema='frozen-e32-real-reselection/1',
        original_selection_sha256=provenance['selection_sha256'],
        terminal_receipt_sha256=provenance['terminal_receipt_sha256'],
        development_binding=development_binding, test_used=False, real_used=True,
        gradients_used=False, optimizer_updates=0, original_training_outputs_unchanged=True,
        historical_real_development_exposure=True, epochs=epochs,
        postprocess_preparation_sha256=sha(args.preparation), source_inputs=files)
    save(out/'protocol.json', dict(status='running', **identity))
    model.to(args.device)
    curve = []
    try:
        for entry in files:
            if (sha(entry['checkpoint']) != entry['checkpoint_sha256']
                    or sha(entry['validation']) != entry['validation_sha256']):
                raise ValueError('archived input changed during reselection')
            cp = torch.load(entry['checkpoint'], map_location='cpu', weights_only=False)
            freeze_model(model, cp['model'])
            before = base.training.state_digest(model)
            report, predictions = development.evaluate(model, torch.device(args.device), base.TrainingConfig())
            if base.training.state_digest(model) != before:
                raise ValueError('read-only reselection changed model tensors')
            epoch = entry['epoch']
            pred_path = out/f'epoch_{epoch:03d}_development_predictions.json'
            save(pred_path, predictions)
            row = dict(**entry, synthetic_cal_threshold=cp['threshold'], real_report=report,
                       predictions=pred_path.name, predictions_sha256=sha(pred_path), model_state_unchanged=True)
            save(out/f'epoch_{epoch:03d}_development.json', row)
            curve.append(row)
            save(out/'real_curve.json', curve)
            save(out/'status.json', dict(status='development_reselection', completed_epochs=len(curve),
                total_epochs=len(files), last_epoch=epoch, elapsed_seconds=time.time()-started))
        chosen = choose_real_best(curve)
        # Terminal files are read-only, not a new training/selection writeback.
        if (sha(stage/'selection.json') != identity['original_selection_sha256']
                or sha(stage.parent/'training_complete.json') != identity['terminal_receipt_sha256']):
            raise ValueError('original terminal receipts changed')
        record = dict(status='complete', **identity, best=chosen,
                      curve_sha256=sha(out/'real_curve.json'), elapsed_seconds=time.time()-started)
        save(out/'real_selection.json', record)
        save(out/'status.json', dict(status='complete', completed_epochs=len(curve),
                                    elapsed_seconds=time.time()-started))
    except BaseException as error:
        save(out/'failure.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc()))
        raise
