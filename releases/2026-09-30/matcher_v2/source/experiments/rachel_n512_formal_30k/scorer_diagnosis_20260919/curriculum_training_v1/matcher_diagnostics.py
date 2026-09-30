"""Scorer-free inspection of native threshold unions and raw Matcher evidence.

GT, when available, labels already-built hypotheses only. No thresholds,
proposals, poses, Q or network parameters are changed here. Represented contour
arc is not claimed to be a ground-truth-correct seam correspondence measure.
"""
import math

import numpy as np
import torch

from .exposure import digest


def distribution(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError('nonfinite diagnostic values')
    if not len(array):
        return dict(count=0, mean=None, quantiles=None)
    q = np.quantile(array, [0., .05, .1, .25, .5, .75, .9, .95, 1.])
    return dict(count=len(array), mean=float(array.mean()),
                quantiles=dict(zip(('min', 'p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'max'),
                                   map(float, q))))


def identity(cluster):
    edges = torch.unique(cluster.edge_ids.detach().cpu().long(), dim=0)
    return digest(dict(edges=edges.tolist(), translation=cluster.translation.detach().cpu().tolist()))


@torch.no_grad()
def inspect_pair(pair_id, pair, proposals, *, label, gt_pose=None, all_clusters=None,
                 seam_reference=None, capture_edges=False):
    """Return all candidate statistics, plus explicit raw-Q and Q*arc winners.

    all_clusters must be the builder's prebudget list for THIS invocation.
    Omission means budget loss is unknown, not zero. seam_reference is optional
    independently audited metadata: {'length_px': ..., 'definition': ...}; its
    ratio is an observed-arc normalization, never a GT accuracy assertion.
    """
    from ..binary_scorer_v1.head import inputs_for_cluster
    from ..s7_consensus_v1.evidence import observed_arc_cells
    if not isinstance(pair_id, str) or not pair_id or (label is not None and type(label) is not bool):
        raise ValueError('explicit pair identity and binary label (or unlabelled None) required')
    if gt_pose is not None and label is not True:
        raise ValueError('negative/unlabelled pairs have no common layout GT')
    target = None if gt_pose is None else torch.as_tensor(gt_pose, device=pair.q.device, dtype=pair.q.dtype)
    if target is not None and (target.shape != (2,) or not bool(torch.isfinite(target).all())):
        raise ValueError('finite 2D positive layout GT required')
    if seam_reference is not None:
        length = seam_reference.get('length_px')
        if (type(length) not in (int, float) or not math.isfinite(length) or length <= 0
                or not isinstance(seam_reference.get('definition'), str) or not seam_reference['definition']):
            raise ValueError('positive independently defined seam length required')
    retained = tuple(proposals.clusters)
    clusters = retained if all_clusters is None else tuple(all_clusters)
    if (len(clusters) < len(retained)
            or [identity(c) for c in clusters[:len(retained)]] != [identity(c) for c in retained]):
        raise ValueError('prebudget candidates do not match this pair retained prefix')
    if not pair.numeric_valid and clusters:
        raise ValueError('invalid pair cannot reuse stale builder candidates')
    if not retained and all_clusters is not None and clusters:
        raise ValueError('nonempty prebudget list with empty retained list is stale or has zero budget')
    aa, _ = observed_arc_cells(pair.ga); ab, _ = observed_arc_cells(pair.gb)
    records = []
    for index, cluster in enumerate(clusters):
        x = inputs_for_cluster(pair, cluster, include_features=False)
        i, j = x.edge_ids.unbind(1)
        q = x.q.detach().double().cpu().numpy()
        arc_a = float(aa[i.unique()].sum()); arc_b = float(ab[j.unique()].sum())
        represented = min(arc_a, arc_b)
        qsum = float(x.q.double().sum()); mass = float(x.mass_weights.double().sum())
        residual = (pair.points_b[j] - pair.points_a[i] - x.pose).norm(dim=1)
        error = None if target is None else float((x.pose - target).norm())
        item = dict(index=index, retained=index < len(retained), candidate_sha256=identity(cluster),
            translation_rc=x.pose.cpu().tolist(), unique_edge_count=len(x.edge_ids),
            unique_endpoints_a=len(i.unique()), unique_endpoints_b=len(j.unique()),
            q_sum=qsum, q_mean=float(q.mean()), q_distribution=distribution(q),
            q_arc_mass_px=mass, q_effective_count=qsum*qsum/max(float(np.square(q).sum()), 1e-30),
            represented_arc_a_px=arc_a, represented_arc_b_px=arc_b,
            represented_bilateral_arc_px=represented,
            represented_arc_over_reference=None if seam_reference is None else represented / seam_reference['length_px'],
            residual_px=distribution(residual.cpu().numpy()),
            raw_q_weighted_residual_px=float((x.q.double() * residual.double()).sum()) / max(qsum, 1e-30),
            original_hypothesis_count=len(cluster.merged_hypothesis_ids),
            original_pose_diameter_px=float(cluster.actual_diameter_px),
            pose_error_px=error, layout20=None if error is None else error <= 20.)
        if capture_edges:
            item['edges'] = dict(compact_indices=x.edge_ids.cpu().tolist(),
                original_indices=torch.stack((pair.original_a[i], pair.original_b[j]), dim=1).cpu().tolist(),
                raw_q=x.q.cpu().tolist(), arc_px=x.arc_px.cpu().tolist(),
                q_arc=x.mass_weights.cpu().tolist(), residual_px=residual.cpu().tolist())
        records.append(item)
    admitted = records[:len(retained)]

    def winner(rows, field):
        return None if not rows else max(rows, key=lambda row: (row[field], -row['index']))['index']

    def correct(rows):
        return None if target is None else any(row['layout20'] for row in rows)

    raw = winner(admitted, 'q_sum'); mass = winner(admitted, 'q_arc_mass_px')
    good = [row for row in admitted if row['layout20'] is True]
    precoverage = None if all_clusters is None else correct(records)
    coverage = correct(admitted)
    return dict(schema='curriculum-matcher-pair/1', pair_id=pair_id, label=label,
        gt_known=target is not None, numeric_valid=bool(pair.numeric_valid),
        valid_points_a=len(pair.q), valid_points_b=pair.q.shape[1],
        retained_count=len(retained), prebudget_count=None if all_clusters is None else len(clusters),
        native_seed_count=len(proposals.seeds), native_hypothesis_count=len(proposals.hypotheses),
        candidates=records, q_sum_winner=raw, q_arc_winner=mass,
        builder_winner=0 if admitted else None,
        q_sum_winner_layout20=None if raw is None or target is None else records[raw]['layout20'],
        q_arc_winner_layout20=None if mass is None or target is None else records[mass]['layout20'],
        q_sum_winner_error_px=None if raw is None else records[raw]['pose_error_px'],
        q_arc_winner_error_px=None if mass is None else records[mass]['pose_error_px'],
        retained_correct_coverage=coverage, prebudget_correct_coverage=precoverage,
        budget_lost_correct=None if precoverage is None else bool(precoverage and not coverage),
        best_correct_by_q_arc=winner(good, 'q_arc_mass_px'),
        negative_max_q_sum=None if label is not False or raw is None else records[raw]['q_sum'],
        negative_max_q_arc_px=None if label is not False or mass is None else records[mass]['q_arc_mass_px'],
        seam_reference=None if seam_reference is None else dict(seam_reference),
        seam_normalization_is_gt_correspondence_accuracy=False,
        scorer_used=False, gt_used_in_proposal=False, q_modified=False)


def summarize(rows):
    """Pair-grain distributions, retaining both classes and all denominators."""
    if not rows or len({row['pair_id'] for row in rows}) != len(rows):
        raise ValueError('nonempty unique pair population required')
    if any(type(row['label']) is not bool or row['schema'] != 'curriculum-matcher-pair/1' or row['scorer_used'] is not False
           or row['q_modified'] is not False or row['gt_used_in_proposal'] is not False for row in rows):
        raise ValueError('native Matcher-only diagnostic rows required')
    positive = [row for row in rows if row['label']]
    negative = [row for row in rows if not row['label']]
    known = [row for row in positive if row['gt_known']]
    preknown = [row for row in known if row['budget_lost_correct'] is not None]
    correct = [row['candidates'][row['best_correct_by_q_arc']] for row in known
               if row['best_correct_by_q_arc'] is not None]
    fields = ('unique_edge_count', 'unique_endpoints_a', 'unique_endpoints_b', 'q_sum', 'q_mean',
              'q_arc_mass_px', 'q_effective_count', 'represented_bilateral_arc_px',
              'represented_arc_over_reference', 'pose_error_px')
    errors = lambda name: distribution([row[name] for row in known if row[name] is not None])
    count = lambda name: sum(row[name] is True for row in known) if known else None
    q_layout = count('q_sum_winner_layout20'); mass_layout = count('q_arc_winner_layout20')
    return dict(schema='curriculum-matcher-summary/1', pairs=len(rows), positives=len(positive), negatives=len(negative),
        positive_layout_gt_count=len(known), invalid_pairs=sum(not row['numeric_valid'] for row in rows),
        positive_no_candidate=sum(row['retained_count'] == 0 for row in positive),
        negative_no_candidate=sum(row['retained_count'] == 0 for row in negative),
        correct_coverage_count=count('retained_correct_coverage'),
        q_sum_layout20_count=q_layout, q_sum_layout20=None if not known else q_layout / len(known),
        q_arc_layout20_count=mass_layout, q_arc_layout20=None if not known else mass_layout / len(known),
        q_sum_error_px=errors('q_sum_winner_error_px'), q_arc_error_px=errors('q_arc_winner_error_px'),
        budget_audited_positive_count=len(preknown),
        budget_lost_correct_count=None if not preknown else sum(row['budget_lost_correct'] for row in preknown),
        correct_cluster_population=len(correct),
        correct_cluster_pair_distributions={field: distribution([row[field] for row in correct if row[field] is not None])
                                            for field in fields},
        negative_max_q_sum=distribution([row['negative_max_q_sum'] for row in negative
                                         if row['negative_max_q_sum'] is not None]),
        negative_max_q_arc_px=distribution([row['negative_max_q_arc_px'] for row in negative
                                           if row['negative_max_q_arc_px'] is not None]),
        no_candidate_negative_mass_imputed=False,
        scorer_used=False, classification_accuracy=None, joint_f1=None,
        distribution_grain='one strongest-correct cluster per GT-covered positive pair; other positives counted separately',
        observed_arc_caveat='unique sampled contour arc, not proof each correspondence lies on the true seam; missing reference excluded with denominator')
