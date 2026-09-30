"""CPU-only reusable S7 M12 token/selected-edge cache for Scorer training.

No labels or GT enter Matcher/selector. Store FP32 contextual tokens and the
production decoder's predicted candidates, not full 512x512 Q. All arms share
these exact inputs; CPU numerical provenance is explicit. Incomplete caches
cannot be used for formal training. This module never starts GPU work.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from experiments.rachel_n512_formal_30k import train_score_decoupled as old
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local.model import (
    select_predicted_inliers, DECODER_CONFIG,
)

ROOT = Path('/root/autodl-tmp/rachel_score_design_20260913_001')
SOURCE = ROOT / 's6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt'
SOURCE_SHA = 'd8a93af1eb5f3b02baaf7d42b9d8675242a446a1b11e43cde0561ba89e670e07'
TRAIN_SHA = '79a9e959f32ef9899116e299425d6350b17f6a04bd5070b5aa9a730319447c36'
SCHEMA = 's7-m12-frozen-token-edge-cache/1'
INPUTS = ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b', 'contour_valid_a', 'contour_valid_b')
COUNTS = {'train': 24000, 'val': 3000}
ARRAYS = {
    'features_a': ('float32', (512, 96)), 'features_b': ('float32', (512, 96)),
    'valid_a': ('bool', (512,)), 'valid_b': ('bool', (512,)),
    'points_a': ('float32', (512, 2)), 'points_b': ('float32', (512, 2)),
    'mask_a': ('bool', (512,)), 'mask_b': ('bool', (512,)),
    'candidate_indices': ('int64', (512, 2)), 'candidate_valid': ('bool', (512,)),
    'candidate_inliers': ('bool', (512,)), 'candidate_weights': ('float32', (512,)),
    'translation_a_to_b_rc': ('float32', (2,)), 'layout_valid': ('bool', ()),
    'training_valid': ('bool', ()), 'decision_valid': ('bool', ()),
    'label': ('float32', ()), 'ready': ('bool', ()),
}
_BASE = None
_THREADPOOL = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def source_checkpoint(path=SOURCE):
    if sha(path) != SOURCE_SHA:
        raise ValueError('requires the original resumable S7 M12 checkpoint')
    # Own previously generated project checkpoint, not the colleague archive.
    source = torch.load(path, map_location='cpu', weights_only=False)
    if (source['epoch'] != 12 or source['completed_segments'] != 48 or source['phase'] != 'matcher'
            or source['resume_identity']['populations']['train']['manifest_sha256'] != TRAIN_SHA):
        raise ValueError('S7 M12/source TRAIN identity differs')
    return source


def initialize(path):
    global _BASE, _THREADPOOL
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from threadpoolctl import threadpool_limits
    _THREADPOOL = threadpool_limits(limits=1)
    source = source_checkpoint(path)
    _BASE = old.load_decoupled_checkpoint(source).base_model.eval().requires_grad_(False)
    if _BASE.config.contour_cap != 512 or _BASE.config.feature_dim != 96:
        raise ValueError('requires S7 original512 features96')


def compute(item):
    """The worker input contains no labels, targets or GT translation."""
    ordinal, arrays = item
    start = time.monotonic()
    args = [torch.as_tensor(arrays[k])[None] for k in INPUTS]
    with torch.inference_mode():
        output = _BASE(*args)
        selected = select_predicted_inliers(output.assignment, args[2], args[3], args[4], args[5])
    candidate_indices = selected.candidate_indices[0]
    present = selected.candidate_valid[0]
    weight = torch.zeros(512)
    ij = candidate_indices[present]
    weight[present] = output.assignment[0, ij[:, 0], ij[:, 1]]
    result = dict(features_a=output.token_features_a[0].numpy(), features_b=output.token_features_b[0].numpy(),
        valid_a=arrays['contour_valid_a'], valid_b=arrays['contour_valid_b'],
        points_a=arrays['points_rc_a'], points_b=arrays['points_rc_b'],
        candidate_weights=weight.numpy(), training_valid=output.training_valid[0].numpy(),
        decision_valid=output.decision_valid[0].numpy())
    for name in ('mask_a', 'mask_b', 'candidate_indices', 'candidate_valid', 'candidate_inliers',
                 'translation_a_to_b_rc', 'layout_valid'):
        result[name] = getattr(selected, name)[0].numpy()
    if not all(np.isfinite(result[name]).all() for name in ('features_a', 'features_b', 'candidate_weights')):
        raise ValueError('nonfinite model evidence')
    # Invalid predicted layout may contain NaN displacement; retain, never fake GT.
    return ordinal, result, dict(seconds=time.monotonic()-start, layout_reason=selected.reasons[0])


def bounded(items, workers, source_path):
    if workers == 1:
        initialize(source_path)
        for item in items:
            yield compute(item)
        return
    iterator = iter(items)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(source_path,)) as pool:
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while len(pending) < 2*workers and not exhausted:
                try:
                    pending.add(pool.submit(compute, next(iterator)))
                except StopIteration:
                    exhausted = True
            if pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()


def items(loader, count, outputs, records):
    seen = set()
    for ordinal, wrapped in enumerate(loader):
        if ordinal >= count:
            break
        batch = getattr(wrapped, 'batch', wrapped)
        pair_id = str(batch.pair_ids[0])
        if pair_id in seen:
            raise ValueError('duplicate pair ID')
        seen.add(pair_id)
        label = float(batch.labels[0])
        if label not in (0., 1.):
            raise ValueError('requires binary pair labels')
        arrays = {k: np.asarray(getattr(batch, k)[0], dtype=np.bool_ if k.startswith('contour_valid')
                                else np.float32) for k in INPUTS}
        fingerprint = hashlib.sha256()
        for k, v in arrays.items():
            fingerprint.update(k.encode()); fingerprint.update(v.tobytes())
        records[ordinal] = dict(ordinal=ordinal, pair_id=pair_id, label=label, input_sha256=fingerprint.hexdigest())
        outputs['label'][ordinal] = label
        yield ordinal, arrays


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('explicit CUDA_VISIBLE_DEVICES empty is required for CPU preparation')
    if not 1 <= args.workers <= 4:
        raise ValueError('bounded CPU workers1..4')
    torch.set_num_threads(1)
    source = source_checkpoint(args.source)
    origin = source['resume_identity']
    train, val, cap, populations = old.make_populations(SimpleNamespace(sampling='original512',
        train_materialized_manifest=origin['populations']['train']['manifest'],
        dataset=str(Path(origin['populations']['val']['manifest']).parents[1])))
    if old.canonical_digest(populations) != old.canonical_digest(origin['populations']):
        raise ValueError('source population drift')
    split, root = args.split, Path(args.output).resolve()
    count = COUNTS[split] if args.limit is None else args.limit
    if not 0 < count <= COUNTS[split]:
        raise ValueError('invalid finite pair limit')
    root.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema=SCHEMA, status='running', pid=os.getpid(), split=split,
        source_checkpoint=str(args.source), source_checkpoint_sha256=SOURCE_SHA,
        population=populations[split], pair_count=count, expected_full_count=COUNTS[split],
        formal_training_eligible=args.limit is None, precompute_device='cpu', features_dtype='float32',
        matcher_frozen=True, no_online_augmentation=True, selector_gt_free=True,
        decoder=asdict(DECODER_CONFIG), workers=args.workers,
        arrays={k: dict(dtype=d, shape=[count, *s]) for k,(d,s) in ARRAYS.items()},
        source_model_config=origin['base_model_config'], implementation_sha256=sha(__file__),
        training_or_validation_selection=False, gpu_jobs_started=False, completed_pairs=0)
    save(root/'protocol.json', protocol)
    outputs = {k: np.lib.format.open_memmap(root/(k+'.npy'), mode='w+', dtype=d, shape=(count,*s))
               for k,(d,s) in ARRAYS.items()}
    outputs['ready'][:] = False
    factory = old.make_weathering_loader if split == 'train' else old.make_ablation_loader
    loader = factory(train if split == 'train' else val, list(range(count)), batch_size=1,
        num_workers=0, seed=old.SEED, contour_cap=cap)
    records, timings = [None]*count, [None]*count
    start = time.monotonic()
    try:
        for ordinal, result, timing in bounded(items(loader, count, outputs, records), args.workers, str(args.source)):
            for name, value in result.items():
                outputs[name][ordinal] = value
            outputs['ready'][ordinal] = True
            timings[ordinal] = timing
            protocol['completed_pairs'] += 1
            if protocol['completed_pairs'] % 512 == 0:
                protocol['elapsed_s'] = time.monotonic()-start
                save(root/'protocol.json', protocol)
        if not outputs['ready'].all() or any(r is None for r in records):
            raise ValueError('cache population incomplete')
        for array in outputs.values():
            array.flush()
        save(root/'pairs.json', records)
        protocol.update(status='complete', elapsed_s=time.monotonic()-start,
            positive_count=int(outputs['label'].sum()),
            training_valid_count=int(outputs['training_valid'].sum()),
            decision_valid_count=int(outputs['decision_valid'].sum()),
            predicted_layout_valid_count=int(outputs['layout_valid'].sum()),
            worker_pair_seconds=sum(t['seconds'] for t in timings),
            pairs_sha256=sha(root/'pairs.json'),
            output_bytes=sum((root/(k+'.npy')).stat().st_size for k in ARRAYS))
    except BaseException as error:
        protocol.update(status='failed', error=repr(error), elapsed_s=time.monotonic()-start)
        raise
    finally:
        save(root/'protocol.json', protocol)
    print(json.dumps(protocol, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--split', choices=tuple(COUNTS), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int, help='CPU timing/debug probe only; never a formal training cache')
    run(p.parse_args())
