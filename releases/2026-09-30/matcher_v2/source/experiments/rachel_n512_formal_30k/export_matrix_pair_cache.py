"""One frozen FP32 forward pass shared by classifier and assignment ablations."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader


def save_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def run(args):
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sealed._set_determinism(args.seed)
    checkpoint_path = Path(args.checkpoint)
    sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    model = load_rachel_checkpoint(sealed._torch_load_checkpoint(checkpoint_path)).to(args.device).eval()
    thresholds = {'coarse': args.coarse_threshold, 'local': args.local_threshold, 'fused': args.fused_threshold}
    manifest = dict(schema_version='rachel-matrix-pair-cache/v1', status='running', split=args.split,
        matcher_checkpoint_id=sha, matcher_checkpoint_path=str(checkpoint_path), precision='fp32',
        original_thresholds=thresholds, sample_count=0, chunks=[], model_config=vars(model.config),
        coarse_cosine_definition='cosine of existing BCE-trained coarse embeddings; NOT PairingNet InfoNCE',
        transport_storage='float32', affinity_storage='float16', seed=args.seed,
        real_translation_gt_opened=False, probe_only=bool(args.limit),
        exact_model_probability_fields=True)
    save_json(dest / 'manifest.json', manifest)
    pair_metadata = None
    if args.split == 'real':
        from experiments.rachel_n512_formal_30k.run_real_contiguous_seam_ablation import load_prepared_cache
        from experiments.rachel_n512_formal_30k.run_real_layout_decoder_experiment import input_batches
        metadata, arrays = load_prepared_cache(args.prepared_cache)
        pair_metadata = {r['pair_id']: r for r in metadata['pairs']}
        loader = input_batches(metadata, arrays, args.batch_size)
        total = len(pair_metadata)
    else:
        dataset = RachelPairDataset(Path(args.dataset), args.split)
        indices = tuple(range(min(len(dataset), args.limit) if args.limit else len(dataset)))
        loader = make_ablation_loader(dataset, indices, batch_size=args.batch_size,
            num_workers=args.workers, seed=args.seed, contour_cap=model.config.contour_cap)
        total = len(indices)
    started = time.perf_counter()
    pending = []
    def flush():
        if not pending:
            return
        data = {k: np.concatenate([r[k] for r in pending], axis=0) for k in pending[0]}
        path = dest / ('chunk_%04d.npz' % len(manifest['chunks']))
        np.savez(path, **data)
        count = len(data['label'])
        manifest['chunks'].append({'path': path.name, 'sample_count': count})
        manifest['sample_count'] += count
        manifest['elapsed_seconds'] = time.perf_counter() - started
        save_json(dest / 'manifest.json', manifest)
        pending.clear()
        print(json.dumps(dict(processed=manifest['sample_count'], total=total,
                              elapsed_seconds=round(manifest['elapsed_seconds'], 2))), flush=True)
    with torch.inference_mode():
        for batch in loader:
            tensors = [sealed._tensor(getattr(batch, name), torch.device(args.device), dtype) for name, dtype in (
                ('mask_a', torch.float32), ('mask_b', torch.float32),
                ('points_rc_a', torch.float32), ('points_rc_b', torch.float32),
                ('contour_valid_a', torch.bool), ('contour_valid_b', torch.bool))]
            output = model(*tensors)
            data = {name: getattr(output, name).detach().float().cpu().numpy()
                    for name in ('affinity', 'unmatched_a', 'unmatched_b', 'coarse_logit', 'local_logit', 'fused_logit',
                                 'coarse_probability', 'local_probability', 'fused_probability')}
            data['affinity'] = data['affinity'].astype(np.float16)
            data['real_transport'] = output.assignment.detach().float().cpu().numpy()
            data['coarse_cosine'] = F.cosine_similarity(output.coarse.embedding_a,
                output.coarse.embedding_b, dim=-1).detach().float().cpu().numpy()
            data.update(points_a_rc=np.asarray(batch.points_rc_a, np.float32),
                        points_b_rc=np.asarray(batch.points_rc_b, np.float32),
                        valid_a=np.asarray(batch.contour_valid_a, bool),
                        valid_b=np.asarray(batch.contour_valid_b, bool),
                        pair_id=np.asarray(batch.pair_ids, dtype='U'),
                        fragment_a=np.asarray(batch.fragment_a_tokens, dtype='U'),
                        fragment_b=np.asarray(batch.fragment_b_tokens, dtype='U'))
            if pair_metadata is not None:
                info = [pair_metadata[x] for x in batch.pair_ids]
                data.update(label=np.array([r['label'] for r in info], bool),
                            strict_member=np.array([r['strict'] for r in info], bool),
                            case_id=np.array([r['case_cluster'] for r in info], dtype='U'),
                            target_translation_rc=np.full((len(info), 2), np.nan, np.float32))
            else:
                targets = np.asarray(batch.translation_a_to_b_rc, np.float32).copy()
                targets[~np.asarray(batch.translation_valid, bool)] = np.nan
                data.update(label=np.asarray(batch.labels, bool), target_translation_rc=targets,
                            target_a=np.asarray(batch.target_a), target_b=np.asarray(batch.target_b))
            for key in ('real_transport', 'coarse_logit', 'local_logit', 'fused_logit'):
                if not np.isfinite(data[key]).all():
                    raise ValueError('nonfinite model output: ' + key)
            pending.append(data)
            if sum(len(x['label']) for x in pending) >= args.chunk_size:
                flush()
            if args.limit and manifest['sample_count'] >= args.limit:
                break
    flush()
    if manifest['sample_count'] != total:
        raise ValueError('cache count differs from requested population')
    manifest.update(status='complete', elapsed_seconds=time.perf_counter() - started)
    save_json(dest / 'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', default='/root/autodl-tmp/dataset_rachel_pairwise_n512_v1')
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=('train', 'val', 'test', 'real'), required=True)
    p.add_argument('--prepared-cache', default='/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--chunk-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=260909)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--coarse-threshold', type=float, default=0.6713946461677551)
    p.add_argument('--local-threshold', type=float, default=0.46360623836517334)
    p.add_argument('--fused-threshold', type=float, default=0.896569013595581)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
