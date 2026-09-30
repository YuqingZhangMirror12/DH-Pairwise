"""Lossless, opt-in C10 evidence snapshots of an ALREADY computed prediction.

No model call, proposal regeneration, Sinkhorn, GT, or visualization cutoff is
used here. Large arrays go to an NPZ sidecar; JSON keeps meanings and index maps.
Localization weights belong to the actual PRE-refinement head pass, whereas
score support belongs to the actual re-encoded FINAL pose. They are not attention
maps, and a numerical nonzero observation is not a confirmed seam.
"""
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


SCHEMA = 's7-consensus-evidence/1'


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


class _Arrays:
    def __init__(self):
        self.values = {}
        self.manifest = {}

    def add(self, name, value):
        if name in self.values:
            raise ValueError('duplicate evidence array: ' + name)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.array(value, copy=True, order='C')
        if value.dtype.kind not in 'biuf':
            raise ValueError('evidence arrays must be numeric, not pickled objects')
        self.values[name] = value
        self.manifest[name] = dict(shape=list(value.shape), dtype=value.dtype.str,
            sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest(),
            nonfinite_count=int((~np.isfinite(value)).sum()))
        return name


def _same_tensor(a, b, message):
    if a is None or b is None:
        if a is not b:
            raise ValueError(message)
        return
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.shape != b.shape or not torch.allclose(a, b, rtol=0, atol=0, equal_nan=True):
        raise ValueError(message)


def union_arc_length(intervals, perimeter):
    """Union of existing observed intervals on a closed arc, with NO gap fill."""
    intervals = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(intervals).all() or not math.isfinite(perimeter) or perimeter < 0:
        raise ValueError('arc geometry must be finite with nonnegative perimeter')
    if not perimeter or not len(intervals):
        return 0.
    parts = []
    for start, end in intervals:
        length = float(end - start)
        if length < 0:
            raise ValueError('raw intervals must be unwrapped with end >= start')
        if length >= perimeter:
            return float(perimeter)
        if not length:
            continue
        start = float(start % perimeter)
        end = start + length
        parts.append((start, min(end, perimeter)))
        if end > perimeter:
            parts.append((0., end - perimeter))
    if not parts:
        return 0.
    parts.sort()
    start, end = parts[0]
    total = 0.
    for left, right in parts[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return float(total + end - start)


def _original_ids(pair, ids):
    a = pair.original_a[ids[:, 0].to(pair.original_a.device)].detach().cpu()
    b = pair.original_b[ids[:, 1].to(pair.original_b.device)].detach().cpu()
    return torch.stack((a, b), -1)


def _proposal(arrays, prefix, proposal, pair):
    result = {}
    for f in fields(proposal):
        value = getattr(proposal, f.name)
        result[f.name] = (arrays.add(prefix + '/' + f.name, value)
                          if isinstance(value, torch.Tensor) else _json_value(value))
    result['edge_original_ids'] = arrays.add(prefix + '/edge_original_ids',
                                            _original_ids(pair, proposal.edge_ids))
    return result


def _encoded(arrays, prefix, encoded):
    evidence = encoded.evidence
    result = {name: arrays.add(prefix + '/' + name, getattr(evidence, name))
              for name in ('pose', 'weights', 'kernels', 'localization_kernels')}
    for side in 'ab':
        result[side] = {}
        for section, value in (('input', getattr(evidence, side)),
                               ('output', getattr(encoded, side))):
            result[side][section] = {f.name: arrays.add(
                f'{prefix}/{side}/{section}/{f.name}', getattr(value, f.name)) for f in fields(value)}
    return result


def snapshot_prediction(pair_id, pair, prediction, *, threshold, provenance):
    """Return detached metadata and exact arrays, not another model prediction.

    Call score_pair(..., capture_diagnostics=True) for nonempty predictions.
    Caller-supplied provenance should identify frozen checkpoint, input, source,
    geometry calibration and threshold origin; this function cannot verify those
    external files. No real-data labels or GT are required or consumed.
    """
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError('pair_id must be a nonempty string')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be a finite probability')
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError('explicit caller-supplied provenance is required')
    if prediction.has_candidate != bool(prediction.clusters):
        raise ValueError('candidate presence is inconsistent')
    if prediction.has_candidate:
        if not 0 <= prediction.selected_cluster_id < len(prediction.clusters):
            raise ValueError('winner index is out of range')
        winner = prediction.clusters[prediction.selected_cluster_id]
        _same_tensor(prediction.translation_a_to_b_rc, winner.translation,
                     'reported pose does not belong to the selected cluster')
        _same_tensor(prediction.score, winner.readout.score,
                     'reported score does not belong to the selected cluster')
    elif prediction.selected_cluster_id != -1 or prediction.translation_a_to_b_rc is not None:
        raise ValueError('empty candidates must not invent a selected pose')
    expected_accept = bool(prediction.has_candidate and prediction.numeric_valid
                           and float(prediction.score.detach()) >= threshold)
    if expected_accept != prediction.accepted:
        raise ValueError('recorded acceptance does not match the supplied threshold')
    arrays = _Arrays()
    pair_record = {}
    for name in ('local_a', 'local_b', 'context_a', 'context_b', 'q', 'unmatched_a',
                 'unmatched_b', 'original_a', 'original_b', 'mask_a', 'mask_b'):
        value = getattr(pair, name)
        pair_record[name] = None if value is None else arrays.add('pair/' + name, value)
    for side in 'ab':
        g = getattr(pair, 'g' + side)
        n = len(getattr(pair, 'local_' + side))
        pair_record['geometry_' + side] = {name: arrays.add(f'pair/geometry_{side}/{name}',
            getattr(g, name)[0, :n]) for name in ('points', 'arc_px', 'next_step_px',
            'cell_px', 'outward_normal_rc', 'normal_reliability')}
        pair_record['geometry_' + side]['perimeter_px'] = float(g.perimeter_px[0])
    proposals = prediction.proposals
    cloud = {f.name: (arrays.add('proposal_cloud/' + f.name, getattr(proposals.cloud, f.name))
                     if isinstance(getattr(proposals.cloud, f.name), torch.Tensor)
                     else _json_value(getattr(proposals.cloud, f.name))) for f in fields(proposals.cloud)}
    hypotheses = [_proposal(arrays, f'hypothesis_{i:03d}', p, pair)
                  for i, p in enumerate(proposals.hypotheses)]
    clusters = []
    for i, cluster in enumerate(prediction.clusters):
        prefix = f'cluster_{i:03d}'
        before, final = cluster.initial_encoded, cluster.encoded
        if before is None:
            raise ValueError('initial pass was not captured; use capture_diagnostics=True')
        if before.evidence.pair is not pair or final.evidence.pair is not pair:
            raise ValueError('evidence does not belong to the supplied pair')
        _same_tensor(before.evidence.pose, cluster.proposal.translation,
                     'captured initial head pass is not at the proposal pose')
        _same_tensor(final.evidence.pose, cluster.translation,
                     'final score input does not match refined/output pose')
        w = final.evidence.weights
        ids = (w > 0).nonzero(as_tuple=False)
        proposed = {tuple(x) for x in cluster.proposal.edge_ids.detach().cpu().tolist()}
        added = torch.tensor([tuple(x) not in proposed for x in ids.detach().cpu().tolist()],
                             dtype=torch.bool, device=ids.device)
        stages = dict(initial=_encoded(arrays, prefix + '/initial', before),
                      final=_encoded(arrays, prefix + '/final', final))
        readout = {f.name: arrays.add(prefix + '/readout/' + f.name,
                    getattr(cluster.readout, f.name)) for f in fields(cluster.readout)}
        refinement = {name: arrays.add(prefix + '/refinement/' + name,
            getattr(cluster.refinement, name)) for name in ('translation', 'localization_weights',
            'compatibility_weights', 'information', 'uncertainty_eigenvalues')}
        steps = cluster.refinement.steps
        refinement['steps'] = arrays.add(prefix + '/refinement/steps',
            torch.stack(steps) if steps else w.new_zeros((0, 2)))
        refinement['underconstrained'] = cluster.refinement.underconstrained
        diagnostics = {}
        for side in 'ab':
            inputs = getattr(final.evidence, side)
            outputs = getattr(final, side)
            measure = inputs.mass * inputs.observed_arc_px
            support = getattr(cluster.readout, 'support_weights_' + side)
            supported = support.detach().cpu().numpy() > 0
            intervals = inputs.arc_intervals.detach().cpu().numpy()[supported]
            perimeter = float(getattr(pair, 'g' + side).perimeter_px[0])
            diagnostics[side] = dict(
                support_arc_intervals=arrays.add(prefix + f'/diagnostic/{side}/support_arc_intervals', intervals),
                unique_supported_arc_length_px=union_arc_length(intervals, perimeter),
                observed_mass_length_px=float(measure.detach().sum()),
                weighted_support_px=float(support.detach().sum()),
                weighted_conflict_px=float((measure * outputs.local_probabilities[:, 2]).detach().sum()),
                weighted_unknown_px=float((measure * outputs.local_probabilities[:, 1]).detach().sum()),
                unmatched=stages['final'][side]['input']['unmatched'],
                other_pose_mass=stages['final'][side]['input']['other_mass'],
                partition_total=arrays.add(prefix + f'/diagnostic/{side}/partition_total',
                    inputs.mass + inputs.other_mass + inputs.unmatched))
        # A/B views of one observation are averaged, never counted twice.
        edge_contributions = {}
        for k, label in enumerate(('support', 'unknown', 'conflict')):
            sa = final.evidence.a.observed_arc_px * final.a.local_probabilities[:, k]
            sb = final.evidence.b.observed_arc_px * final.b.local_probabilities[:, k]
            edge_contributions[label] = arrays.add(prefix + '/edge_contribution/' + label,
                                                   .5 * w * (sa[:, None] + sb[None]))
        clusters.append(dict(cluster_id=i, selected=i == prediction.selected_cluster_id,
            proposal=_proposal(arrays, prefix + '/proposal', cluster.proposal, pair),
            refinement=refinement, stages=stages, readout=readout, overlap=cluster.overlap,
            correspondence_ids=arrays.add(prefix + '/correspondence_ids', ids),
            correspondence_original_ids=arrays.add(prefix + '/correspondence_original_ids', _original_ids(pair, ids)),
            added_to_sparse_proposal=arrays.add(prefix + '/added_to_sparse_proposal', added),
            added_correspondence_count=int(added.sum()), edge_contributions=edge_contributions,
            endpoint_diagnostics=diagnostics))
    result = dict(schema=SCHEMA, pair_id=pair_id, provenance=provenance,
        provenance_verified_by_exporter=False, threshold=float(threshold),
        has_candidate=prediction.has_candidate, numeric_valid=prediction.numeric_valid,
        selected_cluster_id=prediction.selected_cluster_id,
        translation_a_to_b_rc=_json_value(prediction.translation_a_to_b_rc),
        canvas_b_shift_rc=_json_value(None if prediction.translation_a_to_b_rc is None
                                      else -prediction.translation_a_to_b_rc),
        score=_json_value(prediction.score), accepted=prediction.accepted,
        pose_uncertainty=prediction.pose_uncertainty, pair=pair_record,
        proposals=dict(cloud=cloud, seeds=arrays.add('seeds', proposals.seeds),
            hypotheses=hypotheses, merge_trace=proposals.merge_trace), clusters=clusters,
        semantics=dict(coordinates='row,column; p_b ~= p_a + t; place B on A with -t',
            q='complete compact Matcher Q, unnormalized absolute mass, no diagnostic TopK',
            kernels='soft damage compatibility K, NOT attention',
            weights='absolute recalled mass Q*K, NOT normalized attention',
            correspondence_ids='all numerically nonzero FINAL Q*K entries, no display cutoff',
            edge_contributions='absolute Q*K times mean A/B observed-arc local-class support; sums equal readout evidence',
            arc_union='union of existing intervals with numerical nonzero support; NOT confirmed seam length or gap fill',
            refinement_weights='actual INITIAL head-pass localization/compatibility weights used for the joint fit',
            final_reliability='final re-encoded localizer output; NOT retroactively used in refinement',
            uncertainty='curvature proxy from last fit iteration, NOT calibrated error probability',
            attention_weights_exported=False, gradient_attribution_exported=False,
            per_layer_hidden_states_exported=False, gt_used_by_exporter=False), arrays=arrays.manifest)
    # Also checks caller metadata can be represented without silently emitting NaN.
    result = _json_value(result)
    json.dumps(result, allow_nan=False)
    return result, arrays.values


def write_snapshot(directory, metadata, arrays):
    """Write a new snapshot directory. Never overwrite prior evidence/results."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'arrays.npz'
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    metadata = dict(metadata, sidecar=dict(path=path.name, sha256=digest.hexdigest(),
                                         bytes=path.stat().st_size, allow_pickle=False))
    with (directory / 'evidence.json').open('x', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')
    return metadata
