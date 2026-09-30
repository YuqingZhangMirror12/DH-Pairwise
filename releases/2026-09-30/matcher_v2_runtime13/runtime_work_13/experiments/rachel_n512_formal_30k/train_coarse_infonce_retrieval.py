"""CPU InfoNCE projection on frozen coarse features; not full PairingNet.

Positive TRAIN pairs supervise a new projection. Each batch contains at most one
pair per original source image: other sources supply in-batch negatives, while
unlabelled same-source pairs are never assumed negative. The frozen CNN retains
its BCE pretraining. Evaluation is pair-list classification, not gallery recall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_coarse_gate_ablation import fit_f1_threshold
from analyze_layout_v2_results import classification


FEATURE_SCHEMA = 'rachel-coarse-retrieval-features/v1'
SCHEMA = 'rachel-coarse-infonce-projection/v1'
CONFIG = dict(epochs=10, batch_size=64, temperature=0.07, hidden_dim=128,
              output_dim=128, learning_rate=1e-3, weight_decay=1e-4,
              seed=260909, cpu_threads=2)


def write_json(path, data, exclusive=True):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read_features(path):
    path = Path(path)
    if path.is_dir():
        path /= 'manifest.json'
    m = json.loads(path.read_text())
    if m.get('schema_version') != FEATURE_SCHEMA or m.get('status') != 'complete':
        raise ValueError('complete coarse feature cache required')
    with np.load(path.parent / m['feature_file'], allow_pickle=False) as f:
        data = {k: f[k] for k in ('embeddings', 'fragment_id', 'source_id')}
    with np.load(path.parent / m['pair_file'], allow_pickle=False) as f:
        for key in ('pair_id', 'index_a', 'index_b', 'label'):
            data[key] = f[key]
        if m['split'] == 'real':
            for key in ('strict_member', 'case_id'):
                data[key] = f[key]
    n, d = data['embeddings'].shape
    p = len(data['label'])
    if (n, d, p) != (m['fragment_count'], m['feature_dim'], m['sample_count']):
        raise ValueError('feature population differs from manifest')
    if data['embeddings'].dtype != np.float32 or not np.isfinite(data['embeddings']).all():
        raise ValueError('finite FP32 embeddings required')
    if len(data['source_id']) != n or len(set(data['fragment_id'].tolist())) != n:
        raise ValueError('fragment metadata invalid')
    if len(set(data['pair_id'].tolist())) != p or data['label'].dtype != np.bool_:
        raise ValueError('unique pair IDs and explicit bool labels required')
    for key in ('index_a', 'index_b'):
        if data[key].shape != (p,) or data[key].dtype.kind not in 'iu':
            raise ValueError('integer pair endpoints required')
        if np.any(data[key] < 0) or np.any(data[key] >= n):
            raise ValueError('pair index outside fragment population')
    if not p or not data['label'].any() or data['label'].all():
        raise ValueError('both classes required for classification evaluation')
    return m, data


class RetrievalProjection(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, output_dim=128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, output_dim))

    def forward(self, features):
        return F.normalize(self.network(F.normalize(features, dim=-1)), dim=-1)


def positive_source_batches(data, rng, batch_size=64):
    """Use each positive once, at most one pair/source/batch; singleton is logged.

    Source sampling is weighted by remaining pairs. No pair is silently relabelled
    or duplicated to fill the final batch. A singleton cannot supply negatives.
    """
    if batch_size < 2:
        raise ValueError('InfoNCE requires at least two source groups')
    groups = {}
    for index in np.flatnonzero(data['label']):
        a, b = data['index_a'][index], data['index_b'][index]
        source = str(data['source_id'][a])
        if not source or source != str(data['source_id'][b]):
            raise ValueError('positive endpoints need the same source lineage')
        groups.setdefault(source, []).append(int(index))
    for group in groups.values():
        rng.shuffle(group)
    while groups:
        keys = list(groups)
        count = min(batch_size, len(keys))
        weights = np.array([len(groups[k]) for k in keys], dtype=float)
        selected = rng.choice(len(keys), count, replace=False, p=weights / weights.sum())
        batch = []
        for position in selected:
            key = keys[int(position)]
            batch.append(groups[key].pop())
            if not groups[key]:
                del groups[key]
        yield np.asarray(batch, dtype=np.int64)


def symmetric_infonce(a, b, temperature=0.07):
    if a.ndim != 2 or a.shape != b.shape or len(a) < 2 or temperature <= 0:
        raise ValueError('matched batches of >=2 projected features required')
    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


@torch.no_grad()
def pair_cosines(data, projection=None):
    x = torch.from_numpy(data['embeddings'])
    parts = []
    if projection is not None:
        projection.eval()
    for chunk in x.split(4096):
        parts.append(F.normalize(chunk, dim=-1) if projection is None else projection(chunk))
    z = torch.cat(parts)
    return (z[data['index_a']] * z[data['index_b']]).sum(-1).clamp(-1, 1).numpy()


def evaluate_scores(data, values, threshold=None):
    if threshold is None:
        threshold = fit_f1_threshold(data['label'], values)
    return classification(data['label'], values, threshold)


def assert_fit_inputs(tm, train, vm, val):
    if (tm['split'], tm['sample_count'], vm['split'], vm['sample_count']) != ('train', 24000, 'val', 3000):
        raise ValueError('formal fit requires TRAIN24000 and VAL3000 only')
    if tm['matcher_checkpoint_id'] != vm['matcher_checkpoint_id'] or tm['feature_dim'] != vm['feature_dim']:
        raise ValueError('feature checkpoint/dimension mismatch')
    if set(train['source_id'].tolist()) & set(val['source_id'].tolist()):
        raise ValueError('TRAIN/VAL original-source overlap')


def fit(train_path, val_path, output):
    tm, train = read_features(train_path)
    vm, val = read_features(val_path)
    assert_fit_inputs(tm, train, vm, val)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(CONFIG['cpu_threads'])
    torch.manual_seed(CONFIG['seed'])
    rng = np.random.default_rng(CONFIG['seed'])
    model = RetrievalProjection(tm['feature_dim'], CONFIG['hidden_dim'], CONFIG['output_dim'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                                  weight_decay=CONFIG['weight_decay'])
    features = torch.from_numpy(train['embeddings'])
    identity = evaluate_scores(val, pair_cosines(val))
    protocol = dict(schema_version=SCHEMA, status='running', config=CONFIG,
                    matcher_checkpoint_id=tm['matcher_checkpoint_id'], feature_dim=tm['feature_dim'],
                    selected_on='val', selection_rule='VAL equal-pair F1; AP then earlier epoch',
                    encoder_frozen=True, encoder_pretraining='original BCE pair classifier',
                    loss='bidirectional in-batch InfoNCE on labelled TRAIN positive pairs',
                    negatives='different original source images only; no same-source assumed negatives',
                    hard_negative_pair_rows_used_in_loss=False,
                    test_or_real_used_for_fit=False, device='cpu', gallery_retrieval_evaluated=False,
                    paper_reproduction=False, train_count=tm['sample_count'], val_count=vm['sample_count'],
                    feature_layer=tm.get('feature_layer'),
                    limitation='Frozen coarse CNN projection ablation, not end-to-end PairingNet or proof against contrastive retrieval')
    write_json(output / 'protocol.json', protocol)
    best_key = (-1., -1.)
    history = []
    total_updates = 0
    total_pairs = 0
    for epoch in range(1, CONFIG['epochs'] + 1):
        start = time.monotonic()
        model.train()
        samples, skipped, weighted_loss, updates = 0, 0, 0., 0
        for batch in positive_source_batches(train, rng, CONFIG['batch_size']):
            if len(batch) < 2:
                skipped += len(batch)
                continue
            optimizer.zero_grad(set_to_none=True)
            a = model(features[train['index_a'][batch]])
            b = model(features[train['index_b'][batch]])
            loss = symmetric_infonce(a, b, CONFIG['temperature'])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite InfoNCE loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            weighted_loss += float(loss.detach()) * len(batch)
            samples += len(batch)
            updates += 1
        if not samples or samples + skipped != int(train['label'].sum()):
            raise ValueError('positive pair exposure accounting mismatch')
        metrics = evaluate_scores(val, pair_cosines(val, model))
        total_updates += updates
        total_pairs += samples
        event = dict(epoch=epoch, positive_pairs_trained=samples, singleton_pairs_skipped=skipped,
                     optimizer_updates=updates, mean_loss=weighted_loss / samples,
                     seconds=time.monotonic() - start, validation=metrics)
        history.append(event)
        write_json(output / ('epoch_%02d.json' % epoch), event)
        key = (metrics['f1'], metrics['auprc'])
        if key > best_key:
            best_key = key
            winner = dict(epoch=epoch, validation=metrics)
            torch.save(dict(state_dict=model.state_dict(), feature_dim=tm['feature_dim'], config=CONFIG,
                            matcher_checkpoint_id=tm['matcher_checkpoint_id']), output / 'projection.pt')
        write_json(output / 'status.json', dict(status='running', epoch=epoch,
                    completed_optimizer_updates=total_updates, selected_epoch=winner['epoch']), exclusive=False)
        print(json.dumps(dict(event='infonce_epoch_complete', **event)), flush=True)
    freeze = dict(**{k: v for k, v in protocol.items() if k != 'status'}, status='complete',
                  completed_epochs=CONFIG['epochs'], completed_positive_pair_exposures=total_pairs,
                  completed_optimizer_updates=total_updates, selected_epoch=winner['epoch'],
                  validation=winner['validation'], thresholds=dict(infonce=winner['validation']['threshold'],
                  frozen_feature_cosine=identity['threshold']), baseline_validation=identity,
                  projection_file='projection.pt', history=history)
    write_json(output / 'validation_freeze.json', freeze)
    write_json(output / 'status.json', dict(status='complete', epoch=CONFIG['epochs'],
               completed_optimizer_updates=total_updates, selected_epoch=winner['epoch']), exclusive=False)
    return freeze


def load_projection(freeze_path, manifest):
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    if (freeze.get('schema_version') != SCHEMA or freeze.get('status') != 'complete' or
            freeze.get('selected_on') != 'val' or freeze.get('completed_epochs') != CONFIG['epochs'] or
            freeze.get('test_or_real_used_for_fit') is not False):
        raise ValueError('completed VAL-only projection freeze required')
    if manifest['matcher_checkpoint_id'] != freeze['matcher_checkpoint_id'] or manifest['feature_dim'] != freeze['feature_dim']:
        raise ValueError('evaluation feature provenance differs')
    saved = torch.load(freeze_path.parent / freeze['projection_file'], map_location='cpu')
    if saved['matcher_checkpoint_id'] != freeze['matcher_checkpoint_id'] or saved['config'] != freeze['config']:
        raise ValueError('projection provenance differs')
    model = RetrievalProjection(saved['feature_dim'], saved['config']['hidden_dim'], saved['config']['output_dim'])
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    return freeze, model


def evaluate(feature_path, freeze_path, output):
    # Validate freeze before opening evaluation feature data.
    header = json.loads(Path(freeze_path).read_text())
    if header.get('status') != 'complete' or header.get('selected_on') != 'val':
        raise ValueError('freeze must precede external evaluation')
    manifest, data = read_features(feature_path)
    if manifest['split'] not in ('test', 'real'):
        raise ValueError('external evaluation accepts TEST/REAL only')
    freeze, model = load_projection(freeze_path, manifest)
    torch.set_num_threads(CONFIG['cpu_threads'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scores = dict(infonce=pair_cosines(data, model), frozen_feature_cosine=pair_cosines(data))
    def population(include):
        subset = dict(label=data['label'][include])
        return dict(sample_count=int(include.sum()), positive_count=int(subset['label'].sum()),
                    negative_count=int((~subset['label']).sum()), policies={name:
                    evaluate_scores(subset, values[include], freeze['thresholds'][name])
                    for name, values in scores.items()})
    metrics = dict(status='complete', split=manifest['split'],
                   matcher_checkpoint_id=manifest['matcher_checkpoint_id'], selected_epoch=freeze['selected_epoch'],
                   classification=population(np.ones(len(data['label']), bool)),
                   score_type='cosine [-1,1], not calibrated pairability probability',
                   test_or_real_used_for_fit=False, pose_evaluated=False, gallery_retrieval_evaluated=False,
                   decision_coverage=1., encoder_frozen=True, projection_trained_with_infonce=True)
    if manifest['split'] == 'real':
        metrics['strict_classification'] = population(data['strict_member'])
    with (output / 'pair_scores.jsonl').open('x') as stream:
        for i, pair_id in enumerate(data['pair_id']):
            row = dict(pair_id=str(pair_id), label=bool(data['label'][i]),
                       scores={name: float(values[i]) for name, values in scores.items()})
            if manifest['split'] == 'real':
                row.update(strict_member=bool(data['strict_member'][i]), case_id=str(data['case_id'][i]))
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    write_json(output / 'metrics.json', metrics)
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('fit')
    train.add_argument('--train-features', required=True)
    train.add_argument('--val-features', required=True)
    train.add_argument('--output', required=True)
    test = commands.add_parser('evaluate')
    test.add_argument('--features', required=True)
    test.add_argument('--freeze', required=True)
    test.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        fit(args.train_features, args.val_features, args.output)
    else:
        evaluate(args.features, args.freeze, args.output)


if __name__ == '__main__':
    main()
