"""TRAIN-fitted late fusion of frozen coarse/local logits, scalar CPU work only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_coarse_gate_ablation import fit_f1_threshold
from analyze_layout_v2_results import classification


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def read_scalars(path):
    path = Path(path)
    if path.is_dir():
        path /= 'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 'rachel-matrix-pair-cache/v1' or manifest.get('status') != 'complete':
        raise ValueError('complete frozen matcher cache required')
    fields = ['pair_id', 'label', 'coarse_logit', 'local_logit', 'fused_logit']
    if manifest['split'] == 'real':
        fields += ['strict_member', 'case_id']
    rows = {k: [] for k in fields}
    for chunk in manifest['chunks']:
        with np.load(path.parent / chunk['path'], allow_pickle=False) as z:
            # Never load correspondence matrices, points or GT translations.
            for key in fields:
                rows[key].append(z[key])
    data = {k: np.concatenate(v) for k, v in rows.items()}
    if any(len(v) != manifest['sample_count'] for v in data.values()):
        raise ValueError('scalar population differs from complete cache')
    if len(set(data['pair_id'].tolist())) != manifest['sample_count']:
        raise ValueError('duplicate pair identifiers')
    if data['label'].dtype != np.bool_ or any(not np.isfinite(data[k]).all() for k in fields if k.endswith('_logit')):
        raise ValueError('invalid classifier scalars')
    data['x'] = np.column_stack([data['coarse_logit'], data['local_logit']]).astype(float)
    return manifest, data


def affine_objective(parameters, x, y, regularization=1e-4):
    z = x @ parameters[:2] + parameters[2]
    loss = np.mean(np.logaddexp(0, z) - y * z) + regularization * np.sum(parameters[:2] ** 2) / 2
    error = expit(z) - y
    gradient = np.r_[x.T @ error / len(y) + regularization * parameters[:2], error.mean()]
    return loss, gradient


def fit_affine(x, labels, nonnegative):
    bounds = [(0, None), (0, None), (None, None)] if nonnegative else None
    result = minimize(affine_objective, np.array([.5, .5, 0.]), args=(x, labels.astype(float)),
                      method='L-BFGS-B', jac=True, bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('affine fusion fit did not converge: ' + result.message)
    return dict(kind='affine_logits', coefficients=result.x[:2].tolist(), intercept=float(result.x[2]),
                nonnegative_weights=nonnegative, regularization=1e-4, train_objective=float(result.fun),
                iterations=int(result.nit), fit_population='train24000', optimizer='L-BFGS-B')


def scores(data, policies):
    output = {}
    for name, policy in policies.items():
        if policy['kind'] == 'affine_logits':
            value = expit(data['x'] @ np.asarray(policy['coefficients']) + policy['intercept'])
        elif policy['kind'] == 'probability_mean':
            value = expit(data['x']).mean(1)
        elif policy['kind'] == 'logit_mean':
            value = expit(data['x'].mean(1))
        else:
            value = expit(data[policy['branch'] + '_logit'].astype(float))
        output[name] = value
    return output


def population(data, values, policies, include=None):
    include = np.ones(len(data['label']), bool) if include is None else include
    y = data['label'][include]
    return dict(sample_count=int(len(y)), positive_count=int(y.sum()), negative_count=int((~y).sum()),
                policies={name: classification(y, value[include], policies[name]['threshold']) for name, value in values.items()})


def save_population(root, manifest, data, freeze):
    values = scores(data, freeze['policies'])
    result = dict(status='complete', split=manifest['split'], matcher_checkpoint_id=manifest['matcher_checkpoint_id'],
        classification=population(data, values, freeze['policies']), no_model_forward=True, pose_evaluated=False,
        score_source='FP32 cached logits; CPU float64 sigmoid, not bitwise old GPU probability preservation',
        test_or_real_used_for_fit=False, coefficients_fit='TRAIN', thresholds_fit='VAL equal-row F1')
    if manifest['split'] == 'real':
        if (int(data['strict_member'].sum()), int(data['label'][data['strict_member']].sum())) != (547, 508):
            raise ValueError('strict population differs')
        result['strict_classification'] = population(data, values, freeze['policies'], data['strict_member'])
    write_json(root / 'metrics.json', result)
    with (root / 'pair_scores.jsonl').open('x') as f:
        for i, pair_id in enumerate(data['pair_id']):
            row = dict(pair_id=str(pair_id), label=bool(data['label'][i]), scores={k: float(v[i]) for k, v in values.items()})
            if manifest['split'] == 'real':
                row.update(strict_member=bool(data['strict_member'][i]), case_id=str(data['case_id'][i]))
            f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(dict(status='complete', split=manifest['split'], metrics=str(root / 'metrics.json'),
                          f1={k: v['f1'] for k, v in result['classification']['policies'].items()})), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-cache')
    p.add_argument('--val-cache')
    p.add_argument('--freeze')
    p.add_argument('--evaluate-cache')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    if a.train_cache:
        if not a.val_cache or a.freeze or a.evaluate_cache:
            raise ValueError('fit accepts only TRAIN and VAL')
        tm, train = read_scalars(a.train_cache)
        vm, val = read_scalars(a.val_cache)
        if (tm['split'], tm['sample_count'], vm['split'], vm['sample_count']) != ('train', 24000, 'val', 3000):
            raise ValueError('formal fusion needs TRAIN24000/VAL3000')
        if tm['matcher_checkpoint_id'] != vm['matcher_checkpoint_id'] or tm['precision'] != vm['precision']:
            raise ValueError('frozen matcher mismatch')
        policies = {name: dict(kind='original_branch', branch=name, threshold_fit='VAL equal-row F1')
                    for name in ('coarse', 'local', 'fused')}
        policies.update(mean_probability=dict(kind='probability_mean'), mean_logit=dict(kind='logit_mean'),
                        affine_free=fit_affine(train['x'], train['label'], False),
                        affine_nonnegative=fit_affine(train['x'], train['label'], True))
        values = scores(val, policies)
        for name, value in values.items():
            policies[name]['threshold'] = fit_f1_threshold(val['label'], value)
        for name, threshold in vm['original_thresholds'].items():
            policies[name + '_original_frozen'] = dict(kind='original_branch', branch=name,
                threshold=float(threshold), threshold_fit='original matcher VAL freeze')
        freeze = dict(schema_version='rachel-cached-score-fusion/v1', status='complete', created_unix=time.time(),
            matcher_checkpoint_id=tm['matcher_checkpoint_id'], precision=tm['precision'], policies=policies,
            train_count=24000, val_count=3000, test_or_real_used_for_fit=False,
            model_selection='no REAL winner selected; fixed controls with separate VAL thresholds',
            feature_order=['coarse_logit', 'local_logit'], classifiers_frozen=True, matrix_head_changed=False)
        write_json(output / 'validation_freeze.json', freeze)
        save_population(output, vm, val, freeze)
    else:
        if not a.freeze or not a.evaluate_cache or a.val_cache:
            raise ValueError('evaluation needs frozen receipt and TEST/REAL cache only')
        freeze = json.loads(Path(a.freeze).read_text())
        if freeze.get('schema_version') != 'rachel-cached-score-fusion/v1' or freeze.get('status') != 'complete':
            raise ValueError('completed fit required')
        em, data = read_scalars(a.evaluate_cache)
        if em['split'] not in ('test', 'real') or em['matcher_checkpoint_id'] != freeze['matcher_checkpoint_id'] or em['precision'] != freeze['precision']:
            raise ValueError('held-out frozen matcher mismatch')
        save_population(output, em, data, freeze)


if __name__ == '__main__':
    main()
