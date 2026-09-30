"""Export verified binary-head evidence, without loading a model or using CUDA.

The evaluation queue already independently replayed the actual MLP tensors.
This consumer verifies those exact artifacts and prepares display data. It does
not repeat inference, fit thresholds, interpret pooling as Attention, or infer
causal importance from parameter magnitudes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_PAIRS = {
    'sim_test_v14': 3000, 'sim_test_aggressive': 3000,
    'dunhuang_cv': 803, 'turufan': 602,
}
EXPECTED_CASES = {
    'sim_test_v14': 0, 'sim_test_aggressive': 0,
    'dunhuang_cv': 10, 'turufan': 1,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError('nonfinite JSON number: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'artifact path escapes its directory')
    return path


def tensor_reference(sidecar, key, arrays):
    value = arrays[key]
    out = dict(path=str(sidecar), entry=key, shape=list(value.shape), dtype=value.dtype.str)
    if value.size:
        out['range'] = dict(min=float(value.min()), max=float(value.max()),
                            mean=float(value.mean()), std=float(value.std()))
    else:
        out['range'] = None
    return out


def export_case(evidence_path, audit_path, verified_case):
    """The caller supplies the queue's hash-bound fixed-case verification."""
    evidence_path = Path(evidence_path).resolve()
    audit_path = Path(audit_path).resolve()
    require(sha(evidence_path) == verified_case['evidence_sha256'],
            'evidence differs from the queue-verified case')
    meta, audit = read(evidence_path), read(audit_path)
    require(meta.get('schema') == 'binary-cluster-evidence/1', 'not binary evidence')
    require(meta.get('pair_id') == verified_case['pair_id'], 'case identity differs')
    variant = meta.get('variant')
    require(variant in ('patch', 'stats'), 'unknown light-Scorer variant')
    require(audit.get('status') == 'passed' and audit.get('errors') == []
            and audit.get('variant') == variant
            and audit.get('clusters') == len(meta['clusters'])
            and audit.get('raw_union_q_pooling_and_mlp_replayed') is True
            and audit.get('no_external_model_or_gt_opened') is True,
            'passed independent binary numeric audit required')
    require(all(meta['semantics'].get(k) is False for k in (
        'attention_present', 'local_conflict_classifier', 'learned_refinement')),
        'binary evidence cannot declare complex-head modules')
    sidecar = child(evidence_path.parent, meta['sidecar']['path'])
    require(sha(sidecar) == meta['sidecar']['sha256'] == verified_case['sidecar_sha256'],
            'sidecar differs from the queue-verified case')
    with np.load(sidecar, allow_pickle=False) as source:
        arrays = {k: source[k].copy() for k in source.files}
    require(set(arrays) == set(meta['arrays']), 'array membership differs')
    for key, value in arrays.items():
        spec = meta['arrays'][key]
        require(list(value.shape) == spec['shape'] and value.dtype.str == spec['dtype']
                and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'],
                'array content differs: ' + key)
        require(value.dtype.kind in 'biuf' and np.isfinite(value).all(),
                'nonfinite or unsupported tensor: ' + key)

    pair = meta['pair']
    pa = arrays[pair['geometry_a']['points']]
    pb = arrays[pair['geometry_b']['points']]
    q_matrix = arrays[pair['q']]
    result = dict(
        schema='binary-report-case/1', pair_id=meta['pair_id'], variant=variant,
        provenance=meta['provenance'], has_candidate=meta['has_candidate'],
        numeric_valid=meta['numeric_valid'], selected_cluster_id=meta['selected_cluster_id'],
        score=meta['score'], accepted=meta['accepted'], threshold=meta['threshold'],
        translation_a_to_b_rc=meta['translation'],
        coordinate_convention='row,column; add each cluster pose to A, leave B unchanged',
        points_a_rc=pa.tolist(), points_b_rc=pb.tolist(), clusters=[],
        tensor_sources={k: None if pair[k] is None else tensor_reference(sidecar, pair[k], arrays)
                        for k in ('q', 'mask_a', 'mask_b', 'unmatched_a', 'unmatched_b')},
        semantics=dict(attention_present=False, local_conflict_classifier=False,
            learned_refinement=False, feature_pooling_present=variant == 'patch',
            pooling_weight_role=('feature_mean_and_weighted_geometry' if variant == 'patch'
                                 else 'weighted_geometry_only'),
            pooling_weights_are_attention=False, layer_values_are_causal_importance=False,
            weights='original Q is retained per deduplicated (i,j); Q times observed arc supplies mass',
            normalization='conditional mean-pooling only; count, sum Q and mass remain separate MLP inputs',
            candidates='each cluster is scored independently by the same head; select the highest logit'),
        verification=dict(queue_verified_evidence_sha256=sha(evidence_path),
            sidecar_sha256=sha(sidecar), numeric_audit_sha256=sha(audit_path),
            imported_numeric_audit_status=audit['status'],
            this_export_repeated_model_inference=False, this_export_replayed_mlp=False),
    )
    for cluster in meta['clusters']:
        x = {key: None if ref is None else arrays[ref]
             for key, ref in cluster['inputs'].items()}
        ids = x['edge_ids']
        require(ids.ndim == 2 and ids.shape[1] == 2 and len(ids) > 0
                and ids.dtype.kind in 'iu' and np.array_equal(ids, np.unique(ids, axis=0)),
                'display requires the exact deduplicated union')
        i, j = ids.T
        require((i >= 0).all() and (j >= 0).all()
                and (i < len(pa)).all() and (j < len(pb)).all(), 'edge index outside contour')
        require(np.array_equal(x['q'], q_matrix[i, j]), 'display Q differs from online Q')
        require(np.allclose(x['mass_weights'], x['q'] * x['arc_px'], rtol=2e-5, atol=2e-5),
                'display mass differs from Q times arc')
        total = x['mass_weights'].astype(np.float64).sum()
        require(total > 0 and np.allclose(x['normalized_weights'],
                x['mass_weights'] / total, rtol=2e-5, atol=2e-5), 'display pooling weights differ')
        require(len(meta['scalar_names']) == len(x['statistics']), 'scalar input order differs')
        pose = x['pose']
        placed_a = pa[i] + pose
        residual = pb[j] - placed_a
        layers = []
        for layer in cluster['layers']:
            layers.append(dict(name=layer['name'], kind=layer['kind'],
                tensors={field: tensor_reference(sidecar, ref, arrays)
                         for field, ref in layer.items() if field not in ('name', 'kind')}))
        result['clusters'].append(dict(
            cluster_id=cluster['cluster_id'], selected=cluster['selected'],
            pose_a_to_b_rc=pose.tolist(), logit=cluster['logit'], score=cluster['score'],
            original_member_poses_rc=arrays[cluster['proposal']['member_translations_rc']].tolist(),
            counts=dict(unique_pairs=len(ids), unique_a=len(np.unique(i)), unique_b=len(np.unique(j))),
            absolute_q_sum=float(x['q'].sum()), mass_px_sum=float(x['mass_weights'].sum()),
            edges=[dict(a=int(ai), b=int(bj), a_placed_rc=ar.tolist(), b_rc=br.tolist(),
                        q=float(q), observed_arc_px=float(arc), mass_px=float(mass),
                        conditional_pooling_weight=float(cw), residual_rc=dr.tolist(),
                        residual_px=float(np.linalg.norm(dr)))
                   for ai, bj, ar, br, q, arc, mass, cw, dr in zip(
                       i, j, placed_a, pb[j], x['q'], x['arc_px'], x['mass_weights'],
                       x['normalized_weights'], residual)],
            scalar_inputs=dict(zip(meta['scalar_names'], x['statistics'].tolist())),
            patch_context=(None if x['patch_context'] is None else
                           tensor_reference(sidecar, cluster['inputs']['patch_context'], arrays)),
            pooled_features=(None if cluster['pooled_features'] is None else
                             tensor_reference(sidecar, cluster['pooled_features'], arrays)),
            layers=layers,
        ))
    json.dumps(result, allow_nan=False)
    return result


def export_job(directory, verification_path):
    """Import one terminal evaluation job, not declare an experiment complete."""
    directory = Path(directory).resolve()
    verification_path = Path(verification_path).resolve()
    require(not (directory / 'failure.json').exists(), 'failure overrides a stale completed summary')
    verified = read(verification_path)
    summary, protocol, status, frozen, index = [read(directory / name) for name in (
        'summary.json', 'protocol.json', 'status.json', 'prediction_complete.json', 'diagnostic_index.json')]
    split = verified.get('split')
    require(split in EXPECTED_PAIRS, 'unregistered evaluation population')
    require(verified.get('status') == 'passed' and verified.get('model_state_unchanged') is True
            and verified.get('real_inference_performed') is False, 'completed queue verification required')
    variant = verified.get('variant')
    choice = verified.get('selection_kind')
    require(variant in ('patch', 'stats') and choice in ('sim', 'real'), 'explicit arm and selection required')
    identity = dict(variant='binary_' + variant, selection_kind=choice, split=split,
                    total_pairs=EXPECTED_PAIRS[split], selected_epoch=verified['selected_epoch'],
                    checkpoint_sha256=verified['checkpoint_sha256'])
    require(all(all(value.get(k) == v for k, v in identity.items())
                for value in (summary, protocol, frozen)), 'frozen result identity differs')
    require(all(value.get('execution_continuation') == protocol.get('execution_continuation')
                for value in (summary, frozen)), 'frozen execution continuation differs')
    require(status.get('status') == summary.get('status') == protocol.get('status') == 'complete'
            and frozen.get('status') == 'all_predictions_frozen'
            and frozen.get('model_state_unchanged') is True
            and status.get('pairs') == frozen.get('pairs') == verified.get('pairs') == EXPECTED_PAIRS[split],
            'not a complete, immutable evaluation job')
    require(sha(directory / 'summary.json') == verified['summary_sha256']
            and sha(directory / 'pair_predictions.jsonl') == verified['predictions_sha256'] == frozen['sha256'],
            'summary or predictions differ from queue verification')
    require(summary.get('threshold_refitting') is False, 'export must not refit a threshold')
    require(summary.get('threshold') == protocol.get('threshold') == frozen.get('threshold'),
            'recorded classification threshold differs')
    experiment = protocol.get('experiment_variant', protocol['variant'])
    require(all(value.get('experiment_variant', value['variant']) == experiment
                for value in (summary, frozen)), 'experiment identity differs')
    require(experiment in ('binary_patch', 'binary_stats', 'aggressive_binary_patch'),
            'unregistered experiment identity')
    require((split != 'sim_test_aggressive' or experiment == 'aggressive_binary_patch')
            and (experiment != 'aggressive_binary_patch' or split != 'sim_test_v14'),
            'new-data and v14 TEST must remain separate')
    if split.startswith('sim_test_'):
        require(summary.get('main_group') == 'all', 'SIM main group differs')
    else:
        require(summary.get('main_group') == 'real_test'
                and summary.get('real_test_is_historically_unseen') is False,
                'REAL main result must be the source-held-out developmental TEST role')
    if split == 'turufan':
        require(summary.get('layout_gt_available') is False, 'Turufan has no Layout GT')
        for group in summary['groups'].values():
            for policy in group.values():
                require(all(value is None for key, value in policy.items()
                            if key.startswith('joint_') or key in ('layout20', 'layout20_count',
                                'candidate_coverage', 'candidate_coverage_count', 'covered_but_winner_wrong',
                                'winner_correct_but_rejected', 'positive_no_correct_candidate', 'wrong_pose_accepted')),
                        'invented Turufan Layout metric')
    cases = index['cases']
    verified_cases = {c['pair_id']: c for c in verified['fixed_cases']}
    require(index.get('selected_by_new_results') is False
            and len(cases) == len(verified_cases) == len(verified['fixed_cases']) == EXPECTED_CASES[split]
            and {c['pair_id'] for c in cases} == set(verified_cases)
            and summary.get('diagnostic_cases') == cases, 'fixed diagnostic case set differs')
    exported = []
    for case in cases:
        result = export_case(child(directory, case['evidence']), child(directory, case['numerical_audit']),
                             verified_cases[case['pair_id']])
        require(result['variant'] == variant and all(result['provenance'].get(k) == v
                for k, v in identity.items()), 'case provenance differs from evaluation job')
        require(result['provenance'].get('execution_continuation') == protocol.get('execution_continuation'),
                'case execution continuation differs from evaluation job')
        require(result['threshold'] == summary['threshold'], 'case threshold differs')
        exported.append(result)
    return dict(schema='binary-report-job/1', status='verified_single_evaluation_job',
                experiment=experiment, selection_kind=choice, split=split,
                checkpoint_sha256=verified['checkpoint_sha256'], selected_epoch=verified['selected_epoch'],
                threshold=summary['threshold'], main_group=summary['main_group'],
                groups=summary['groups'], cases=exported, complete_experiment_claimed=False,
                inference_repeated=False, threshold_refitted=False,
                summary_metrics='copied from the hash-bound frozen evaluation; not re-estimated by this exporter',
                source=dict(job=str(directory), verification_sha256=sha(verification_path),
                            summary_sha256=verified['summary_sha256'],
                            predictions_sha256=verified['predictions_sha256']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = export_job(args.job, args.verification)
    # Never overwrite an existing report or write into the frozen evidence job.
    target = args.out.resolve()
    require(not target.is_relative_to(args.job.resolve()), 'output cannot modify frozen evaluation inputs')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
