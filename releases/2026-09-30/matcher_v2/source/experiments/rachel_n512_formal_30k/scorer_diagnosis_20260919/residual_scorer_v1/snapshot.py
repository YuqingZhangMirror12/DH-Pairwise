"""Residual evidence snapshot and independent arithmetic audit.

Common union/Q/statistics equations intentionally mirror binary_eval_v1. The
graph/schema are separate: a skip addition must not be replayed as an MLP chain.
Only same-forward arrays are saved. Pooling weights are not Attention weights.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..binary_scorer_v1.head import SCALAR_NAMES
from ..s7_consensus_v1.diagnostics import _Arrays, _proposal, _json_value, write_snapshot
from .head import ARCHITECTURE, MODULE, architecture_record
from .trace import LAYER_KINDS


SCHEMA = 'residual-cluster-evidence/1'
VARIANT = 'binary_patch_residual'


def snapshot_prediction(pair_id, pair, prediction, *, threshold, provenance, trace):
    if provenance.get('variant') != VARIANT or provenance.get('architecture') != ARCHITECTURE:
        raise ValueError('explicit residual provenance, not an old binary label, required')
    if len(trace.clusters) != len(prediction.clusters) or trace.current:
        raise ValueError('one complete same-forward trace per cluster required')
    a = _Arrays()
    pr = {}
    for name in ('q', 'unmatched_a', 'unmatched_b', 'original_a', 'original_b',
                 'mask_a', 'mask_b', 'local_a', 'local_b', 'context_a', 'context_b'):
        value = getattr(pair, name)
        pr[name] = None if value is None else a.add('pair/' + name, value)
    for side in 'ab':
        geometry = getattr(pair, 'g' + side)
        n = pair.q.shape[0 if side == 'a' else 1]
        pr['geometry_' + side] = {name: a.add('pair/' + side + '/' + name, getattr(geometry, name)[0, :n])
                                 for name in ('points', 'next_step_px', 'arc_px', 'outward_normal_rc', 'normal_reliability')}
        pr['geometry_' + side]['perimeter_px'] = float(geometry.perimeter_px[0])
    saved = []
    for index, (cluster, layers) in enumerate(zip(prediction.clusters, trace.clusters)):
        prefix = 'cluster_%03d' % index
        r, x = cluster.readout, cluster.readout.inputs
        inputs = {name: a.add(prefix + '/input/' + name, getattr(x, name))
                  for name in ('edge_ids', 'q', 'arc_px', 'mass_weights', 'normalized_weights',
                               'edge_geometry', 'statistics', 'patch_context', 'pose')}
        values = [dict(name=name, kind=layer['kind'], **{
            field: a.add(prefix + '/layer/' + name + '/' + field, value)
            for field, value in layer.items() if field != 'kind'}) for name, layer in layers.items()]
        saved.append(dict(cluster_id=index, selected=index == prediction.selected_cluster_id,
            proposal=_proposal(a, prefix + '/proposal', cluster.proposal, pair), inputs=inputs,
            pooled_features=a.add(prefix + '/pooled_features', r.pooled_features),
            logit=float(r.logit), score=float(r.score), layers=values))
    meta = dict(schema=SCHEMA, pair_id=pair_id, provenance=provenance, variant=VARIANT,
        architecture=architecture_record(trace.head), scalar_names=list(SCALAR_NAMES),
        threshold=threshold, has_candidate=prediction.has_candidate, numeric_valid=prediction.numeric_valid,
        selected_cluster_id=prediction.selected_cluster_id, score=float(prediction.score),
        accepted=prediction.accepted, translation=_json_value(prediction.translation_a_to_b_rc),
        pair=pr, clusters=saved, arrays=a.manifest,
        semantics=dict(attention_present=False, local_conflict_classifier=False, learned_refinement=False,
            fixed_conflict_penalty=False, batch_norm=False,
            correspondence='deduplicated exact union; original online Q; no second Sinkhorn',
            mass_weights='Q times observed arc length; not learned Attention',
            pooling='conditional weighted mean plus max; absolute Q/count/mass remain separate inputs',
            residual='h + F(h), including actual branch and sum; no activation after addition',
            selection='one shared head separately scores all clusters; argmax logit',
            layers='detached hooks from the actual forward; no rerun or synthesized attention'))
    return meta, a.values


def audit_snapshot(path):
    """No checkpoint, inference, labels or external GT are used in the replay."""
    from torch.nn import functional as F
    path = Path(path)
    m = json.loads(path.read_text())
    if m.get('schema') != SCHEMA or m.get('variant') != VARIANT:
        raise ValueError('explicit residual evidence schema required')
    if m.get('architecture', {}).get('architecture') != ARCHITECTURE:
        raise ValueError('wrong residual architecture')
    side = path.parent / m['sidecar']['path']
    if hashlib.sha256(side.read_bytes()).hexdigest() != m['sidecar']['sha256']:
        raise ValueError('residual sidecar hash differs')
    with np.load(side, allow_pickle=False) as source:
        data = {k: source[k].copy() for k in source.files}
    if set(data) != set(m['arrays']):
        raise ValueError('residual array membership differs')
    for name, value in data.items():
        declaration = m['arrays'][name]
        if (list(value.shape) != declaration['shape'] or value.dtype.str != declaration['dtype']
                or hashlib.sha256(value.tobytes()).hexdigest() != declaration['sha256']):
            raise ValueError('residual array identity differs: ' + name)
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    def close(x, y, message):
        check(np.shape(x) == np.shape(y) and np.allclose(x, y, rtol=2e-5, atol=2e-5, equal_nan=False), message)

    def get(reference):
        return data[reference]

    p = m['pair']
    q = get(p['q'])
    ga, gb = p['geometry_a'], p['geometry_b']
    points_a, points_b = get(ga['points']), get(gb['points'])

    def arcs(geometry):
        step = get(geometry['next_step_px'])
        return np.minimum(.5 * step, 3.5) + np.minimum(.5 * np.roll(step, 1), 3.5)

    aa, ab = arcs(ga), arcs(gb)
    for name in ('attention_present', 'local_conflict_classifier', 'learned_refinement', 'fixed_conflict_penalty', 'batch_norm'):
        check(m['semantics'].get(name) is False, 'unexpected residual semantic: ' + name)
    check(m['scalar_names'] == list(SCALAR_NAMES), 'scalar order differs')
    width = m['architecture']['feature_dim']
    shapes = {'edge_mlp.0': (64, 4 * width + 8), 'edge_mlp.2': (32, 64),
              'cluster_mlp.0': (64, 80), 'cluster_mlp.2.down': (32, 64),
              'cluster_mlp.2.up': (64, 32), 'cluster_mlp.3': (1, 64)}
    expected_parameters = sum(a * b + a for a, b in shapes.values())
    check(m['architecture']['parameter_count'] == expected_parameters, 'parameter count differs')
    check(m['architecture']['module'] == MODULE, 'experiment identity differs')
    check(m['architecture']['activation_after_addition'] is False, 'unexpected post-add activation')
    shared_parameters = {}
    for index, c in enumerate(m['clusters']):
        check(c['cluster_id'] == index, 'cluster IDs differ')
        check(c['selected'] == (index == m['selected_cluster_id']), 'cluster selected flag differs')
        x = {name: get(reference) for name, reference in c['inputs'].items()}
        ids = x['edge_ids']
        if ids.ndim != 2 or ids.shape[1] != 2 or len(ids) == 0 or ids.dtype.kind not in 'iu':
            raise ValueError('nonempty integer edge union required')
        i, j = ids.T
        if np.any(ids < 0) or np.any(i >= q.shape[0]) or np.any(j >= q.shape[1]):
            raise ValueError('edge index outside Q')
        check(np.array_equal(ids, np.unique(ids, axis=0)), 'union duplicated or unsorted')
        proposal = c['proposal']
        check(np.array_equal(ids, np.unique(get(proposal['edge_ids']), axis=0)), 'Scorer did not receive full proposal union')
        check(np.array_equal(x['q'], q[i, j]), 'online Q differs')
        check(np.isfinite(x['q']).all() and (x['q'] >= 0).all(), 'invalid Q')
        close(x['arc_px'], (aa[i] + ab[j]) / 2, 'observed arc differs')
        close(x['mass_weights'], x['q'] * x['arc_px'], 'Q arc mass differs')
        w = x['mass_weights']
        if not np.isfinite(w).all() or w.astype(np.float64).sum() <= 0:
            raise ValueError('positive finite mass required')
        cw = (w.astype(np.float64) / w.astype(np.float64).sum()).astype(w.dtype)
        close(x['normalized_weights'], cw, 'conditional pooling differs')
        check(np.array_equal(x['pose'], get(proposal['translation'])), 'pose differs from common builder fit')
        distance = np.linalg.norm(points_b[j] - points_a[i] - x['pose'], axis=-1)
        stats = x['statistics']
        close(stats[:5], np.array([np.log1p(len(ids)), np.log1p(x['q'].sum()), np.log1p(w.sum()),
                                  x['q'].mean(), x['q'].max()]), 'absolute statistics differ')
        poses = get(proposal['member_translations_rc'])
        diameter = np.linalg.norm(poses[:, None] - poses[None, :], axis=-1).max()
        check(diameter <= 16.0001, 'original candidate diameter exceeds fixed16')
        row, col = np.zeros(q.shape[0], q.dtype), np.zeros(q.shape[1], q.dtype)
        np.add.at(row, i, x['q'])
        np.add.at(col, j, x['q'])
        ua, ub = get(p['unmatched_a'])[i], get(p['unmatched_b'])[j]
        outside = (np.maximum(q.sum(1) - row, 0)[i] + np.maximum(q.sum(0) - col, 0)[j]) / 2
        unmatched = (ua + ub) / 2
        edge = np.stack([np.log1p(x['q'] * 100), x['q'], distance / 20, (distance / 20) ** 2,
                         unmatched, np.abs(ua - ub), outside, np.log1p(x['arc_px']) / 4], axis=-1)
        close(x['edge_geometry'], edge, 'edge geometry differs')
        coverage = np.array([aa[np.unique(i)].sum() / max(1, ga['perimeter_px']),
                             ab[np.unique(j)].sum() / max(1, gb['perimeter_px'])])
        overlap = 0.
        if p['mask_a'] is not None and p['mask_b'] is not None:
            ma, mb = get(p['mask_a']), get(p['mask_b'])
            dr, dc = np.rint(x['pose']).astype(int)
            r0, c0 = max(0, -dr), max(0, -dc)
            r1, c1 = min(ma.shape[0], mb.shape[0] - dr), min(ma.shape[1], mb.shape[1] - dc)
            if r1 > r0 and c1 > c0:
                overlap = float((ma[r0:r1, c0:c1] * mb[r0 + dr:r1 + dr, c0 + dc:c1 + dc]).sum()) / max(1, min(ma.sum(), mb.sum()))
        tail = np.array([1 / (np.square(cw).sum() * len(cw)), coverage.mean(), coverage.min(),
            (cw * distance).sum() / 20, np.sqrt((cw * distance ** 2).sum()) / 20, distance.max() / 20,
            (cw * unmatched).sum(), (cw * outside).sum(), overlap, diameter / 16,
            np.log1p(len(proposal['merged_hypothesis_ids']))])
        close(stats[5:], tail, 'remaining statistics differ')
        layers = c['layers']
        if [layer['name'] for layer in layers] != list(LAYER_KINDS):
            raise ValueError('incomplete, repeated or reordered residual graph')
        lookup = {layer['name']: layer for layer in layers}
        for layer in layers:
            name = layer['name']
            if layer['kind'] != LAYER_KINDS[name]:
                raise ValueError('wrong residual layer kind: ' + name)
            inp = torch.from_numpy(get(layer['input']))
            if layer['kind'] == 'linear':
                weight, bias = get(layer['weight']), get(layer['bias'])
                if weight.shape != shapes[name] or bias.shape != (shapes[name][0],):
                    raise ValueError('wrong residual parameter shape: ' + name)
                for field, array in (('weight', weight), ('bias', bias)):
                    key = name + '/' + field
                    if index:
                        check(np.array_equal(array, shared_parameters[key]), 'different head weights across clusters: ' + key)
                    else:
                        shared_parameters[key] = array
                result = F.linear(inp, torch.from_numpy(weight), torch.from_numpy(bias))
            elif layer['kind'] == 'gelu':
                result = F.gelu(inp)
            else:
                result = inp + torch.from_numpy(get(layer['branch_output']))
            close(result.numpy(), get(layer['output']), 'layer replay differs: ' + name)
        chains = [('edge_mlp.0', 'edge_mlp.1', 'edge_mlp.2', 'edge_mlp.3'),
                  ('cluster_mlp.0', 'cluster_mlp.1', 'cluster_mlp.2.down', 'cluster_mlp.2.activation', 'cluster_mlp.2.up')]
        for chain in chains:
            for before, after in zip(chain, chain[1:]):
                close(get(lookup[before]['output']), get(lookup[after]['input']), 'graph edge differs: ' + before + ' -> ' + after)
        add = lookup['cluster_mlp.2']
        close(get(add['input']), get(lookup['cluster_mlp.1']['output']), 'skip input differs')
        close(get(add['branch_output']), get(lookup['cluster_mlp.2.up']['output']), 'branch input differs')
        close(get(add['output']), get(lookup['cluster_mlp.3']['input']), 'residual sum is not final projection input')
        features = np.concatenate([(get(p['local_a'])[i] + get(p['local_b'])[j]) / 2,
            np.abs(get(p['local_a'])[i] - get(p['local_b'])[j]),
            (get(p['context_a'])[i] + get(p['context_b'])[j]) / 2,
            np.abs(get(p['context_a'])[i] - get(p['context_b'])[j])], axis=-1)
        close(x['patch_context'], features, 'patch/context inputs differ')
        close(get(lookup['edge_mlp.0']['input']), np.concatenate([features, edge], axis=-1), 'edge encoder input differs')
        h = get(lookup['edge_mlp.3']['output'])
        pooled = np.concatenate([(h * cw[:, None]).sum(0), h.max(0)])
        close(get(c['pooled_features']), pooled, 'mean/max pooling differs')
        close(get(lookup['cluster_mlp.0']['input']), np.concatenate([pooled, stats]), 'cluster input differs')
        logit = float(get(lookup['cluster_mlp.3']['output']).reshape(-1)[0])
        close(logit, c['logit'], 'final logit differs')
        close(float(torch.tensor(logit).sigmoid()), c['score'], 'final sigmoid differs')
    if m['clusters']:
        winner = int(np.argmax([c['logit'] for c in m['clusters']]))
        check(winner == m['selected_cluster_id'], 'winner is not shared Scorer argmax')
        close(m['score'], m['clusters'][winner]['score'], 'winning score differs')
        close(m['translation'], get(m['clusters'][winner]['inputs']['pose']), 'winning pose differs')
    else:
        check(m['selected_cluster_id'] == -1 and m['translation'] is None and m['score'] == 0., 'empty candidate invented output')
    check(m['has_candidate'] == bool(m['clusters']), 'candidate presence differs')
    check(m['accepted'] == bool(m['has_candidate'] and m['numeric_valid'] and m['score'] >= m['threshold']), 'acceptance differs')
    return dict(status='passed' if not errors else 'failed', errors=errors, clusters=len(m['clusters']),
        variant=VARIANT, residual_addition_replayed=True, attention_present=False,
        raw_union_q_pooling_and_graph_replayed=True, no_external_model_or_gt_opened=True)
