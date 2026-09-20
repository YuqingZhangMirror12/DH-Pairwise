"""Frozen evaluation for new S3/S4/S5; no test-driven fitting or reranking.

The historical S3 staged evaluator and the live S0/S1/S2 loader are untouched.
Only the new classifier score is a decision branch; coarse is untrained and
retained solely as an explicitly labelled diagnostic for file compatibility.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.rachel_n512_formal_30k import evaluate_score_design as core
from experiments.rachel_n512_formal_30k import decoupled_prediction_reuse as reuse

SCHEMA = 'rachel-score-decoupled-evaluation/1'
SELECTIONS = ('fixed_epoch', 'max_f1', 'recall95')


def inference_runtime(device):
    """Small code/runtime identity; actual input bytes are checked per batch."""
    package_root = Path(__file__).resolve().parents[2]
    files = list((package_root / 'staging/pairwise_v0_2/models').glob('*.py'))
    files.extend(Path(module.__file__) for module in
        (core, core.fixed, core.common, core.real, core.sealed, reuse))
    files.extend((Path(__file__), Path(core.pair_size_metadata.__code__.co_filename)))
    return dict(device=str(device), torch_version=str(torch.__version__),
        numpy_version=np.__version__, cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        source_sha256={str(path.resolve().relative_to(package_root)): core.sealed._sha256_file(path)
            for path in sorted(set(files))})


def load_frozen_model(root, selection):
    from experiments.rachel_n512_formal_30k.train_score_decoupled import load_decoupled_checkpoint
    root = Path(root).resolve(strict=True)
    path = root / 'classifier_freezes/freeze.json'
    freeze = json.loads(path.read_text())
    if (freeze.get('schema_version') != 'rachel-score-decoupled-training/1'
            or freeze.get('status') != 'complete' or freeze.get('held_out_used_for_fit') is not False
            or freeze.get('eligible_epoch_range') != [13, 20]):
        raise ValueError('requires completed C13..20 SIM-VAL-only classifier freeze')
    selected = freeze['selections'][selection]
    epoch = selected['selected_epoch']
    if not 13 <= epoch <= 20 or (selection == 'fixed_epoch' and epoch != 20):
        raise ValueError('checkpoint outside registered classifier budget')
    checkpoint = root / ('epoch_%03d.pt' % epoch)
    if Path(selected['checkpoint']).resolve() != checkpoint:
        raise ValueError('checkpoint must belong to this training run')
    digest = core.sealed._sha256_file(checkpoint)
    if digest != selected['checkpoint_sha256']:
        raise ValueError('checkpoint changed after SIM VAL freeze')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    training_identity = payload.get('resume_identity', {})
    identity_digest = hashlib.sha256(json.dumps(training_identity, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    # JSON freezes normalize tuples to lists; compare their canonical JSON
    # identities rather than rejecting a legitimate Torch-vs-JSON container.
    frozen_identity_digest = hashlib.sha256(json.dumps(freeze.get('resume_identity'), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (identity_digest != frozen_identity_digest
            or identity_digest != freeze.get('resume_identity_sha256')
            or payload.get('seed') != training_identity.get('seed')
            or payload.get('epoch') != epoch or payload.get('phase') != 'classifier'
            or payload.get('global_exposure') != epoch * 24000
            or selected.get('test_or_real_or_ood_used_for_fit') is not False):
        raise ValueError('selected checkpoint differs from frozen training identity or budget')
    model = load_decoupled_checkpoint(payload)
    identity = dict(training_run=str(root), budget=20, selection=selection,
        freeze_path=str(path), freeze_sha256=core.sealed._sha256_file(path),
        checkpoint_path=str(checkpoint), checkpoint_sha256=digest,
        epoch=epoch, seed=payload['seed'], model_config=asdict(model.config),
        architecture=payload['resume_identity']['head_kind'],
        sampling=payload['resume_identity']['sampling'],
        classifier_thresholds=selected['classifier_thresholds'],
        operating_points=selected['operating_points'], winner_record=selected,
        training_identity=payload['resume_identity'], model_design=model.metadata(),
        classifier_only_pair_bce=True, coarse_is_untrained_diagnostic=True,
        local_and_fused_are_same_single_classifier=True,
        test_or_real_used_for_fit=False, ood_used_for_fit=False)
    return model.eval().requires_grad_(False), identity


def run(args):
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError('invalid evaluation batch/workers')
    if args.split == 'real' and not args.keep_ids:
        raise ValueError('REAL requires unchanged keep IDs')
    torch.set_num_threads(1)
    model, identity = load_frozen_model(args.training_run, args.selection)
    core.sealed._set_determinism(identity['seed'])
    device = torch.device(args.device)
    model_on_device = False
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    sampling = identity['sampling']
    mode = sampling if isinstance(sampling, str) else sampling['mode']
    resampler = None
    if mode == 'step3':
        from staging.pairwise_v0_2.pairwise_data.rachel_step_density import StepInputContourResampler
        resampler = StepInputContourResampler(step_px=3., cap=2048)
    elif mode == 'paired512':
        resampler = core.InputContourResampler(512)
    elif mode != 'original512':
        raise ValueError('unknown registered sampling mode')
    targets, metadata, source_units = {}, None, None
    if args.split == 'test':
        dataset_root = Path(args.dataset)
        manifest_path = dataset_root / 'pairs/test.jsonl'
        manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
        if len(manifest) != 3000 or sum(row['label'] for row in manifest) != 1500:
            raise ValueError('requires complete balanced TEST3000')
        expected_ids = [row['pair_id'] for row in manifest]
        source_units = {row['pair_id']: sorted({row['fragment_' + side]['split_unit_id']
                                             for side in 'ab'}) for row in manifest}
        # Evaluation is label-free resampling of the SAME original masks;
        # classifier/layout never sees correspondence supervision or GT.
        ds = core.RachelPairDataset(dataset_root, 'test')
        batches = core.make_ablation_loader(ds, tuple(range(len(ds))), batch_size=args.batch_size,
            num_workers=args.workers, seed=identity['seed'], contour_cap=512)
        source = dict(dataset_root=str(dataset_root), manifest_sha256=core.sealed._sha256_file(manifest_path))
    else:
        cache = Path(args.prepared_cache if args.split == 'real' else args.ood_prepared)
        if args.split == 'real':
            metadata, arrays = core.load_prepared_cache(cache)
            if len(metadata['pairs']) != 1016 or sum(row['label'] for row in metadata['pairs']) != 508:
                raise ValueError('requires unchanged Dunhuang1016')
        else:
            metadata = json.loads((cache / 'manifest.json').read_text())
            if (len(metadata['pairs']) != 301 or len(metadata['fragment_ids']) != 602
                    or metadata.get('layout_gt_provided') is not False
                    or metadata.get('negative_pairs_constructed') is not False):
                raise ValueError('requires original positive-only Turufan301')
            with np.load(cache / 'inputs.npz', allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in ('packed_masks', 'points', 'valid')}
        expected_ids = [row['pair_id'] for row in metadata['pairs']]
        batches = core.real.input_batches(metadata, arrays, args.batch_size)
        source = dict(prepared_cache=str(cache), manifest_sha256=core.sealed._sha256_file(cache / 'manifest.json'))
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError('duplicate evaluation IDs')
    protocol = dict(schema_version=SCHEMA, status='running', split=args.split, model=identity,
        **source, sample_count=len(expected_ids), batch_size=args.batch_size, precision='fp32',
        decoder=core.fixed.DECODER_NAME, decoder_config=asdict(core.fixed.TOP2_CONFIG),
        decoder_design_unchanged=True, resampling=mode, model_input_fields=list(core.FIELDS),
        ground_truth_used_to_select_candidates=False, gt_attached_after_complete_prediction_freeze=True,
        thresholds_fitted=False, test_or_real_used_for_fit=False, ood_used_for_fit=False,
        script_sha256=core.sealed._sha256_file(Path(__file__)),
        inference_runtime=inference_runtime(device),
        prediction_reuse_schema=reuse.SCHEMA, input_digest_schema=reuse.INPUT_DIGEST_SCHEMA)
    prediction_reuse = reuse.PredictionReuse(destination, protocol, expected_ids)
    core.save(destination / 'protocol.json', protocol)
    predictions, started = [], time.monotonic()
    try:
        with (destination / 'pair_predictions.jsonl').open('x') as stream, \
                (destination / 'prediction_batches.jsonl').open('x') as batch_stream:
            for batch_index, batch in enumerate(batches):
                # Save TEST labels separately before inference-only resampler clears targets.
                if args.split == 'test':
                    for i, pid in enumerate(batch.pair_ids):
                        targets[pid] = dict(label=bool(batch.labels[i]),
                            translation_rc=core.common.clean(batch.translation_a_to_b_rc[i])
                            if batch.translation_valid[i] else None, source_unit_ids=source_units[pid])
                if resampler is not None:
                    batch = resampler(batch) if mode == 'step3' else resampler.resample_batch(batch)
                # Reuse only a completed sibling selection's target-blind
                # predictions, with identical weights/runtime and actual inputs.
                # A cache miss follows precisely the original inference path.
                input_record = reuse.batch_input_record(batch)
                rows = prediction_reuse.lookup(batch_index, input_record)
                if rows is None:
                    if not model_on_device:
                        model = model.to(device).eval().requires_grad_(False)
                        model_on_device = True
                    rows = core.predict_batch(model, batch, device)
                batch_stream.write(json.dumps(input_record, ensure_ascii=False, allow_nan=False) + '\n')
                predictions.extend(rows)
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            batch_stream.flush()
            os.fsync(batch_stream.fileno())
        if [row['pair_id'] for row in predictions] != expected_ids:
            raise ValueError('predictions do not match complete ordered population')
        protocol['prediction_files_sha256'] = {name: core.sealed._sha256_file(destination / name)
            for name in ('pair_predictions.jsonl', 'prediction_batches.jsonl')}
        protocol['prediction_reuse'] = prediction_reuse.summary()
        core.save(destination / 'prediction_complete.json', dict(status='all_predictions_frozen',
            sample_count=len(predictions), real_gt_opened=False, review_labels_opened=False,
            checkpoint_sha256=identity['checkpoint_sha256']))
        if args.split == 'real':
            rows = core.real.attach_ground_truth(predictions, metadata['pairs'], args.translation_gt_json)
            kept = set(json.loads(Path(args.keep_ids).read_text())['kept_positive_pair_ids'])
            if len(kept) != 295 or not kept <= {r['pair_id'] for r in rows if r['label']}:
                raise ValueError('keep cohort differs from reviewed295')
            for row in rows:
                row['review_status'] = ('keep' if row['pair_id'] in kept else 'exclude') if row['label'] else 'not_reviewed_negative'
            protocol['keep_ids_sha256'] = core.sealed._sha256_file(Path(args.keep_ids))
        elif args.split == 'test':
            rows = core.fixed.attach_test_targets(predictions, targets)
        else:
            rows = predictions
            for row in rows:
                row.update(label=True, target_translation_rc=None, layout_gt_available=False)
                row['layouts'][core.fixed.DECODER_NAME]['translation_l2_px'] = None
        core.fixed.write_rows(destination / 'pair_results.jsonl', rows)
        groups = core.summarize_populations(rows, identity, args.split)
        # Do not present the untrained coarse diagnostic as a competing method.
        for group in groups.values():
            if isinstance(group, dict):
                for field in ('classification', 'score_distributions', 'extreme_score_cases'):
                    if isinstance(group.get(field), dict):
                        group[field] = {key: value for key, value in group[field].items() if key == 'fused'}
                group['decision_branch'] = 'fused (single new Pair classifier)'
                group['coarse_is_untrained_diagnostic'] = True
        summary = dict(status='complete', split=args.split, model=identity, groups=groups,
            selection_on_this_population=False, threshold_fitting_performed=False,
            prediction_reuse=protocol['prediction_reuse'])
        if args.baseline_evaluation:
            summary['paired_baseline_layout'] = core.compare_baseline(rows, args.baseline_evaluation, args.split)
        core.save(destination / 'summary.json', summary)
        protocol.update(status='complete', elapsed_seconds=time.monotonic() - started)
        core.save(destination / 'protocol.json', protocol)
        print(json.dumps(dict(status='complete', split=args.split, count=len(rows), output=str(destination))))
        return summary
    except Exception as error:
        protocol.update(status='failed', error=repr(error))
        core.save(destination / 'protocol.json', protocol)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run', required=True)
    p.add_argument('--selection', choices=SELECTIONS, default='fixed_epoch')
    p.add_argument('--split', choices=('test', 'real', 'ood'), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dataset', default='/root/autodl-tmp/dataset_rachel_pairwise_n512_v1')
    p.add_argument('--prepared-cache', default=str(core.fixed.DEFAULT_PREPARED_CACHE))
    p.add_argument('--ood-prepared', default=str(core.DEFAULT_OOD))
    p.add_argument('--translation-gt-json', default=str(core.real.DEFAULT_TRANSLATION_GT))
    p.add_argument('--keep-ids')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--baseline-evaluation')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
