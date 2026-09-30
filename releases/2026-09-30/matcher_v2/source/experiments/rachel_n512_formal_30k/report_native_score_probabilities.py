"""Correct baseline reporting precision without changing frozen models/policies.

New matrix heads already save native FP32 network probabilities. The baseline
reader instead reconstructed FP64 sigmoid from logits, unintentionally resolving
FP32 saturation ties. Replace only original-network baseline scores with exact
probabilities already present in TEST/REAL caches. Retain old artifacts, all new
policy score definitions, selected weights/epochs and frozen thresholds.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_matrix_pair_head import population_metrics


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def run(args):
    source, output, cache = Path(args.source), Path(args.output), Path(args.cache)
    cache = cache / 'manifest.json' if cache.is_dir() else cache
    manifest = load_json(cache)
    metrics = load_json(source / 'metrics.json')
    freeze = load_json(args.freeze)
    if manifest['status'] != 'complete' or metrics['status'] != 'complete' or freeze['status'] != 'complete':
        raise ValueError('completed upstream data/fit/metrics required')
    if manifest['split'] not in ('test', 'real') or metrics['split'] != manifest['split']:
        raise ValueError('native reporting correction is TEST/REAL only')
    identity = manifest['matcher_checkpoint_id']
    if freeze['matcher_checkpoint_id'] != identity:
        raise ValueError('frozen classifier and input cache use different matcher checkpoints')
    metric_identity = metrics.get('matcher_checkpoint_id', metrics.get('cache', {}).get('matcher_checkpoint_id'))
    if metric_identity != identity:
        raise ValueError('score artifact and cache use different matcher checkpoints')
    matrix = freeze.get('schema_version') == 'rachel-matrix-pair-head/v1'
    if not matrix and freeze.get('schema_version') != 'rachel-cached-score-fusion/v1':
        raise ValueError('unsupported frozen policy format')
    rows = [json.loads(line) for line in (source / 'pair_scores.jsonl').read_text().splitlines() if line]
    if len(rows) != manifest['sample_count'] or len({r['pair_id'] for r in rows}) != len(rows):
        raise ValueError('prediction population count or uniqueness differs')
    native = {}
    for chunk in manifest['chunks']:
        with np.load(cache.parent / chunk['path'], allow_pickle=False) as z:
            ids, labels = z['pair_id'], z['label']
            values = {b: z[b + '_probability'] for b in ('coarse', 'local', 'fused')}
            for i, pair_id in enumerate(ids):
                if str(pair_id) in native:
                    raise ValueError('duplicate native prediction pair ID')
                native[str(pair_id)] = dict(label=bool(labels[i]), **{b: float(v[i]) for b, v in values.items()})
    if set(native) != {r['pair_id'] for r in rows}:
        raise ValueError('native and score pair IDs differ; no inner-join repair')
    changed = {b: 0 for b in ('coarse', 'local', 'fused')}
    saturated = {b: 0 for b in changed}
    for row in rows:
        reference = native[row['pair_id']]
        if row['label'] != reference['label']:
            raise ValueError('labels differ across frozen predictions')
        for branch in changed:
            key = 'existing_' + branch if matrix else branch
            changed[branch] += int(row['scores'][key] != reference[branch])
            saturated[branch] += int(reference[branch] in (0., 1.))
            row['scores'][key] = reference[branch]
        if not matrix:
            for name, policy in freeze['policies'].items():
                if policy['kind'] == 'original_branch':
                    row['scores'][name] = reference[policy['branch']]
    # Normalize only the reporting interface; policy thresholds are untouched.
    policies = freeze['policies'] if matrix else {name: dict(branch=name, threshold=p['threshold'], gate_threshold=None)
                                                  for name, p in freeze['policies'].items()}
    predictions = dict(labels=np.array([r['label'] for r in rows], bool),
                       scores={k: np.array([r['scores'][k] for r in rows], float) for k in rows[0]['scores']})
    corrected = population_metrics(predictions, policies)
    decision_changes = {name: {k: value[k] - metrics['classification']['policies'][name][k]
                              for k in ('tp', 'fp', 'fn', 'tn')}
                        for name, value in corrected['policies'].items()}
    result = dict(metrics, classification=corrected, native_probability_reporting=True,
        probability_reporting_status='original_network_branches_from_exact_cached_FP32_probabilities',
        original_uncorrected_metrics=str(source / 'metrics.json'), native_cache=str(cache),
        frozen_policy_source=str(Path(args.freeze)), thresholds_modified=False, weights_modified=False,
        model_executed=False, new_affine_or_average_policy_definitions_unchanged=True,
        baseline_probability_changed_row_counts=changed, native_probability_saturated_row_counts=saturated,
        decision_confusion_count_deltas=decision_changes,
        correction_reason='Remove unintended FP64 sigmoid reconstruction from original-network baselines; original artifacts retained.')
    if manifest['split'] == 'real':
        strict = np.array([r['strict_member'] for r in rows], bool)
        if (int(strict.sum()), int(predictions['labels'][strict].sum())) != (547, 508):
            raise ValueError('strict population changed')
        result['strict_classification'] = population_metrics(predictions, policies, strict)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metrics.json', result)
    with (output / 'pair_scores.jsonl').open('x') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(dict(status='complete', output=str(output), confusion_changes=decision_changes,
                         old_fused_ap=metrics['classification']['policies']['existing_fused_original_frozen' if matrix else 'fused_original_frozen']['auprc'],
                         native_fused_ap=corrected['policies']['existing_fused_original_frozen' if matrix else 'fused_original_frozen']['auprc'])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--freeze', required=True)
    p.add_argument('--output', required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
