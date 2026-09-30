"""Audited C10 arrays -> bounded, source-backed rows for the existing Data app.

This is display preparation, NEVER inference. No model call, new threshold,
geometric fit, GT correction or candidate selection belongs here. Matrix bins
retain every entry; bounded link lists explicitly disclose their omitted mass.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .audit import audit_snapshot, validate_arrays
from .frozen import read, sha


SCHEMA = 's7-consensus-reviewed-case/1'
QUANTITIES = {
    'q': ('原始 Matcher Q', 'absolute transport mass'),
    'initial_recalled': ('精修前实际评分证据', 'absolute transport mass'),
    'final_recalled': ('最终位姿实际评分证据', 'absolute transport mass'),
    'initial_directional_diagnostic': ('精修前方向核 Q×K（仅诊断）', 'absolute transport mass'),
    'final_directional_diagnostic': ('最终方向核 Q×K（仅诊断）', 'absolute transport mass'),
    'union_membership': ('精确并集成员（0/1）', 'binary membership'),
    'localization': ('实际精修定位权重', 'mass-weighted px'),
    'compatibility_fit': ('实际精修相容约束权重', 'mass-weighted px'),
    'support': ('最终评分支持贡献', 'mass-weighted px'),
    'unknown': ('最终证据不足贡献', 'mass-weighted px'),
    'conflict': ('最终冲突贡献', 'mass-weighted px'),
}


def matrix_view(value, bins=48):
    """All-cell arithmetic means for heatmaps, not per-image max normalization."""
    value = np.asarray(value)
    if value.ndim != 2 or type(bins) is not int or not 1 <= bins <= 128:
        raise ValueError('matrix display requires2D values and1..128 bins')
    n, m = value.shape
    result = dict(shape=[n, m], nonfinite_count=int((~np.isfinite(value)).sum()),
        aggregation='arithmetic mean of ALL entries in each compact-index bin',
        normalization='none', display_field='mean', rows=[])
    if result['nonfinite_count']:
        return dict(result, available=False, total=None, reason='nonfinite source matrix')
    if (value < 0).any():
        raise ValueError('negative evidence cannot use an unsigned heatmap')
    result.update(available=True, total=float(value.sum(dtype=np.float64)),
                  nonzero_count=int(np.count_nonzero(value)))
    if not n or not m:
        return result
    ar = np.linspace(0, n, min(n, bins) + 1, dtype=int)
    br = np.linspace(0, m, min(m, bins) + 1, dtype=int)
    for ai, (a0, a1) in enumerate(zip(ar[:-1], ar[1:])):
        for bi, (b0, b1) in enumerate(zip(br[:-1], br[1:])):
            block = value[a0:a1, b0:b1].astype(np.float64)
            result['rows'].append(dict(a_bin=ai, b_bin=bi,
                a_start=int(a0), a_end_exclusive=int(a1),
                b_start=int(b0), b_end_exclusive=int(b1), cells=int(block.size),
                total=float(block.sum()), mean=float(block.mean()),
                maximum=float(block.max()), nonzero_count=int(np.count_nonzero(block))))
    return result


def mask_runs(mask):
    """Exact original-resolution binary raster runs; never contour smoothing."""
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError('expected an exact binary2D mask')
    runs = []
    for row, values in enumerate(mask.astype(bool)):
        changes = np.diff(np.pad(values.astype(np.int8), (1, 1)))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            runs.append([row, int(start), int(end)])
    return dict(height=mask.shape[0], width=mask.shape[1], foreground_pixels=int(mask.sum()),
                encoding='row,start-column,end-column-exclusive; original pixels', runs=runs)


def bounded_links(matrix, a, b, original_a, original_b, pose, limit=128, eligible=None):
    """Top display links only; retain omitted counts/mass, and correct B=-t sign."""
    matrix = np.asarray(matrix, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    if type(limit) is not int or not 1 <= limit <= 512 or pose.shape != (2,):
        raise ValueError('invalid display link bound or pose')
    if (matrix.shape != (len(a), len(b)) or not np.isfinite(matrix).all()
            or (matrix < 0).any() or not np.isfinite(pose).all()):
        raise ValueError('invalid link evidence')
    keep = matrix > 0
    if eligible is not None:
        if np.asarray(eligible).shape != matrix.shape:
            raise ValueError('link eligibility shape differs')
        keep &= np.asarray(eligible, dtype=bool)
    flat = np.flatnonzero(keep)
    order = np.argsort(-matrix.ravel()[flat], kind='stable')
    chosen = flat[order[:limit]]
    rows = []
    for index in chosen:
        i, j = map(int, np.unravel_index(index, matrix.shape))
        rows.append(dict(a_compact=i, b_compact=j,
            a_original=int(original_a[i]), b_original=int(original_b[j]),
            ax=float(a[i, 1]), ay=float(a[i, 0]),
            bx=float(b[j, 1] - pose[1]), by=float(b[j, 0] - pose[0]),
            weight=float(matrix[i, j])))
    total = float(matrix.ravel()[flat].sum())
    displayed = float(matrix.ravel()[chosen].sum())
    return dict(rows=rows, eligible_edges=len(flat), displayed_edges=len(rows),
        omitted_edges=len(flat)-len(rows), eligible_weight=total, displayed_weight=displayed,
        omitted_weight=float(matrix.ravel()[flat[order[limit:]]].sum()),
        display_fraction=displayed/total if total else None,
        pose_rc=pose.tolist(), limit=limit, selection='largest weight; compact-index tie order',
        used_to_limit_model_input=False)


def _resolved(value, arrays):
    if isinstance(value, str) and value in arrays:
        return arrays[value].tolist()
    if isinstance(value, dict):
        return {k: _resolved(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolved(v, arrays) for v in value]
    return value


def _number(value):
    return float(value) if np.isfinite(value) else None


def _side_rows(meta, arrays, stage, side, pose):
    pair = meta['pair']; geometry = pair['geometry_'+side]
    get = lambda key: arrays[key]
    points = get(geometry['points']); original = get(pair['original_'+side])
    inp, output = stage[side]['input'], stage[side]['output']
    shift = np.zeros(2) if side == 'a' else -np.asarray(pose)
    probability = get(output['local_probabilities'])
    measure = get(inp['mass']).astype(np.float64)*get(inp['observed_arc_px'])
    rows = []
    for i, point in enumerate(points):
        rows.append(dict(compact_id=i, original_id=int(original[i]),
            x=float(point[1]+shift[1]), y=float(point[0]+shift[0]),
            source_x=float(point[1]), source_y=float(point[0]),
            arc_px=float(get(geometry['arc_px'])[i]), cell_px=float(get(geometry['cell_px'])[i]),
            observed_arc_px=float(get(inp['observed_arc_px'])[i]),
            arc_interval_start_px=float(get(inp['arc_intervals'])[i, 0]),
            arc_interval_end_px=float(get(inp['arc_intervals'])[i, 1]),
            evidence_present=bool(get(inp['valid'])[i]), mass=float(get(inp['mass'])[i]),
            unmatched=float(get(inp['unmatched'])[i]), other_pose_mass=float(get(inp['other_mass'])[i]),
            support_probability=float(probability[i, 0]), unknown_probability=float(probability[i, 1]),
            conflict_probability=float(probability[i, 2]),
            support_weight_px=float(measure[i]*probability[i, 0]),
            unknown_weight_px=float(measure[i]*probability[i, 1]),
            conflict_weight_px=float(measure[i]*probability[i, 2]),
            localizer_output=float(get(output['localization_reliability'])[i]),
            conditional_residual_mean_rc=get(inp['residual_mean'])[i].tolist(),
            conditional_residual_variance_rc=get(inp['residual_variance'])[i].tolist()))
    return rows


def build_case(meta, arrays, *, bins=48, link_limit=128):
    audit = validate_arrays(meta, arrays)
    pair = meta['pair']; get = lambda key: arrays[key]
    points = {s: get(pair['geometry_'+s]['points']) for s in 'ab'}
    original = {s: get(pair['original_'+s]) for s in 'ab'}
    q = get(pair['q'])
    result = dict(schema=SCHEMA, pair_id=meta['pair_id'], provenance=meta['provenance'],
        selected_cluster_id=meta['selected_cluster_id'], has_candidate=meta['has_candidate'],
        numeric_valid=meta['numeric_valid'], score=meta['score'], threshold=meta['threshold'],
        accepted=meta['accepted'], translation_a_to_b_rc=meta['translation_a_to_b_rc'],
        canvas_b_shift_rc=meta['canvas_b_shift_rc'], pose_uncertainty=meta['pose_uncertainty'],
        audit=audit, attention_capture=meta.get('attention_capture'),
        masks={s: mask_runs(None if pair['mask_'+s] is None else get(pair['mask_'+s])) for s in 'ab'},
        geometry={s: [dict(compact_id=i, original_id=int(original[s][i]),
            x=float(p[1]), y=float(p[0]), arc_px=float(get(pair['geometry_'+s]['arc_px'])[i]),
            cell_px=float(get(pair['geometry_'+s]['cell_px'])[i]),
            q_mass=_number(q.sum(1 if s == 'a' else 0, dtype=np.float64)[i]),
            unmatched=_number(get(pair['unmatched_'+s])[i])) for i, p in enumerate(points[s])] for s in 'ab'},
        q=matrix_view(q, bins), clusters=[], trace=_resolved(meta['proposals'], arrays),
        semantics=dict(meta['semantics'], matrix_display='all-entry bin means; never per-image renormalized',
            links='bounded display sample ONLY; all omitted counts/weights are reported',
            class_probabilities='learned support/unknown/conflict outputs, NOT ground-truth labels',
            gradient_attribution_available=False,
            attention_weights_available=meta['semantics'].get('attention_weights_exported',False),
            score_contributions_are_causal_attribution=False), quantities=QUANTITIES)
    for cluster in meta['clusters']:
        before, final = cluster['stages']['initial'], cluster['stages']['final']
        initial_pose, final_pose = get(before['pose']), get(final['pose'])
        ids = get(cluster['correspondence_ids'])
        added = np.zeros_like(q, dtype=bool)
        added[ids[:, 0], ids[:, 1]] = get(cluster['added_to_sparse_proposal'])
        matrices = dict(initial_recalled=get(before['weights']), final_recalled=get(final['weights']),
            localization=get(cluster['refinement']['localization_weights']),
            compatibility_fit=get(cluster['refinement']['compatibility_weights']),
            **{name: get(key) for name, key in cluster['edge_contributions'].items()})
        if meta['semantics'].get('evidence_mode') == 'exact_union_q':
            matrices.update(initial_directional_diagnostic=get(before['directional_q_diagnostic']),
                final_directional_diagnostic=get(final['directional_q_diagnostic']),
                union_membership=get(final['kernels']))
        def links(name, pose, eligible=None):
            return bounded_links(matrices[name], points['a'], points['b'], original['a'], original['b'],
                                 pose, link_limit, eligible)
        result['clusters'].append(dict(cluster_id=cluster['cluster_id'], selected=cluster['selected'],
            proposal=_resolved(cluster['proposal'], arrays),
            refined_translation_rc=final_pose.tolist(), initial_translation_rc=initial_pose.tolist(),
            refinement_steps=_resolved(cluster['refinement']['steps'], arrays),
            underconstrained=cluster['refinement']['underconstrained'],
            overlap=cluster['overlap'], readout=_resolved(cluster['readout'], arrays),
            joint_constraint=cluster.get('joint_constraint'),
            endpoint_diagnostics=_resolved(cluster['endpoint_diagnostics'], arrays),
            points={name: {s: _side_rows(meta, arrays, stage, s, pose) for s in 'ab'}
                for name, stage, pose in [('initial', before, initial_pose), ('final', final, final_pose)]},
            heatmaps={name: matrix_view(matrix, bins) for name, matrix in matrices.items()},
            attention={name:{label:dict(layer=item['layer'],kind=item['kind'],heads=item['heads'],
                    query_side=item['query_side'],key_side=item['key_side'],
                    query_compact_ids=get(item['query_compact_ids']).tolist(),
                    key_compact_ids=get(item['key_compact_ids']).tolist(),
                    query_original_ids=original[item['query_side']][get(item['query_compact_ids'])].tolist(),
                    key_original_ids=original[item['key_side']][get(item['key_compact_ids'])].tolist(),
                    matrix=matrix_view(get(item['head_mean']),bins),
                    key_measure=get(item['key_measure']).tolist(),
                    reconstruction_max_abs=item['reconstruction_max_abs'],
                    meaning='head mean of executed feature attention; binned means do not re-normalize rows',
                    geometry_bias_exported=item['geometry_bias'] is not None)
                for label,item in stage.get('attention',{}).items()}
                for name,stage in [('initial',before),('final',final)]},
            links=dict(initial_localization=links('localization', initial_pose),
                initial_compatibility=links('compatibility_fit', initial_pose),
                final_support=links('support', final_pose), final_added_support=links('support', final_pose, added),
                final_conflict=links('conflict', final_pose)),
            added_correspondence_count=cluster['added_correspondence_count'],
            added_support_contribution_px=float(matrices['support'][added].sum(dtype=np.float64))))
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def shared_scales(cases):
    """A single absolute display range per quantity across ALL compared cases."""
    values = {name: [] for name in QUANTITIES}
    for case in cases:
        for name, view in [('q', case['q'])]+[(name, view) for c in case['clusters']
                for name, view in c['heatmaps'].items()]:
            if view['available']:
                values[name].extend(row['mean'] for row in view['rows'])
    result = {key: dict(minimum=0., maximum=max(items) if items else None,
        observed=bool(items), field='mean', transformation='linear',
        shared_across='all supplied arms/cases/clusters; not separately normalized',
        label=QUANTITIES[key][0], unit=QUANTITIES[key][1]) for key, items in values.items()}
    has_attention=any(c['semantics'].get('attention_weights_available') for c in cases)
    result['scorer_attention']=dict(minimum=0.,maximum=1. if has_attention else None,
        observed=has_attention,field='mean',transformation='linear',
        shared_across='all captured stages/layers/arms/cases; no independent renormalization',
        label='Scorer四头平均Attention，不是Matcher Q',unit='row-normalized feature attention')
    return result


def load_evaluation(directory, *, bins=48, link_limit=128):
    """Read only a complete frozen-evaluation result; no partial report data."""
    directory = Path(directory).resolve()
    protocol = read(directory/'protocol.json')
    complete = read(directory/'prediction_complete.json')
    index = read(directory/'diagnostic_index.json')
    if ((directory/'failure.json').exists() or read(directory/'status.json')['status'] != 'complete'
            or protocol['status'] != 'complete' or complete['status'] != 'all_predictions_frozen'
            or complete['model_state_unchanged'] is not True
            or protocol['model_selection_on_test_or_real'] is not False
            or protocol['threshold_refitted'] is not False
            or index['selected_by_new_results'] is not False
            or complete['sha256'] != sha(directory/'pair_predictions.jsonl')):
        raise ValueError('complete frozen inference and fixed-case evidence required')
    targets = {r['pair_id']: r for r in (json.loads(line) for line in
        (directory/'case_diagnostics.jsonl').read_text().splitlines())}
    cases = []
    for item in index['cases']:
        path = (directory/item['evidence']).resolve()
        if directory not in path.parents:
            raise ValueError('evidence path escapes result directory')
        audit_snapshot(path)
        meta = read(path)
        for key in ('arm', 'split', 'checkpoint_sha256', 'selected_epoch', 'threshold',
                    'selection_sha256', 'data_contract_sha256', 'geometry_calibration_sha256'):
            if meta['provenance'][key] != protocol[key] or complete[key] != protocol[key]:
                raise ValueError('case/protocol frozen identity differs: '+key)
        for key in ('variant', 'evidence_mode'):
            if key in protocol and (meta['provenance'].get(key) != protocol[key]
                    or complete.get(key) != protocol[key] or meta['semantics'].get(key) != protocol[key]):
                raise ValueError('case/protocol evidence meaning differs: '+key)
        if (meta['pair_id'] != item['pair_id'] or meta['sidecar']['sha256'] != item['sidecar_sha256']
                or meta['threshold'] != protocol['threshold']):
            raise ValueError('diagnostic index differs from actual snapshot')
        with np.load(path.parent/meta['sidecar']['path'], allow_pickle=False) as values:
            case = build_case(meta, {k: values[k] for k in values.files}, bins=bins, link_limit=link_limit)
        target = targets[case['pair_id']]
        for key in ('score', 'accepted', 'selected_cluster_id', 'has_candidate', 'numeric_valid'):
            if target[key] != case[key]:
                raise ValueError('posthoc target row differs from frozen prediction')
        if target['translation'] != case['translation_a_to_b_rc']:
            raise ValueError('posthoc layout differs from frozen prediction')
        expected_error = (float(np.linalg.norm(np.asarray(target['translation'])-target['target_translation_rc']))
                          if target['gt_known'] and target['translation'] is not None else None)
        if expected_error is None:
            if target['error_px'] is not None:
                raise ValueError('unknown layout GT must not have a numerical error')
        elif not np.isclose(expected_error, target['error_px'], rtol=1e-7, atol=1e-6):
            raise ValueError('posthoc layout error differs from coordinates')
        case['posthoc_evaluation'] = {key: target[key] for key in
            ('label', 'gt_known', 'target_translation_rc', 'error_px', 'layout20', 'candidate_errors_px')}
        if not target['gt_known']:
            case['posthoc_evaluation']['layout20'] = None
        case['source'] = dict(evidence_json=str(path), evidence_sha256=sha(path),
            sidecar_sha256=meta['sidecar']['sha256'], protocol_sha256=sha(directory/'protocol.json'),
            predictions_sha256=complete['sha256'], case_diagnostics_sha256=sha(directory/'case_diagnostics.jsonl'))
        cases.append(case)
    return cases


def write_review(evaluations, out, *, bins=48, link_limit=128):
    cases = [case for directory in evaluations for case in load_evaluation(directory, bins=bins, link_limit=link_limit)]
    keys = [(c['provenance']['checkpoint_sha256'], c['provenance']['arm'],
             c['provenance']['split'], c['pair_id']) for c in cases]
    if not cases or len(keys) != len(set(keys)):
        raise ValueError('empty or duplicate reviewed case population')
    result = dict(schema='s7-consensus-reviewed-cases/1', cases=cases, scales=shared_scales(cases),
        case_selection='registered guide cases, not selected by current results',
        model_inference_performed=False, layout_correctness_inferred_from_heatmap=False,
        bins=bins, display_link_limit=link_limit)
    with Path(out).open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        handle.write('\n')
    return dict(path=str(out), sha256=sha(out), cases=len(cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', action='append', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--bins', type=int, default=48)
    parser.add_argument('--link-limit', type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(write_review(args.evaluation, args.out, bins=args.bins, link_limit=args.link_limit)))
