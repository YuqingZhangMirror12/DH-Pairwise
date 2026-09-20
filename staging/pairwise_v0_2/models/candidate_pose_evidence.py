"""Target-blind evidence at a supplied translation, plus TRAIN-only proposals.

``translation_rc`` maps A to B: ``point_b = point_a + translation_rc``.
B is therefore placed at ``-translation_rc`` on A's canvas. No target enters
``geometry_at_translation``; even correspondence-free proposals retain their
measurable contour contact and filled-mask overlap. The separate, explicitly
TRAIN-only sampler uses targets to propose supervised examples, never features.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .physical_translation_layout import (
    _filled_mask, outward_contour_normals, translated_intersection_area,
)
from .seam_geometry_pair_head import GEOMETRY_NAMES, TOP2_CONFIG, arc_evidence
from .translation_layout import _candidates, _validate_inputs


def geometry_at_translation(points_a, points_b, assignment, valid_a, valid_b,
                            mask_a, mask_b, translation_rc):
    """Return finite float32 ``GEOMETRY_NAMES`` evidence at exactly the given t.

    The fixed Top2-union correspondence extractor is reused, but neither the
    decoder's winning pose nor its inliers/support are used. Inliers are edges
    with ``norm((b - a) - t) <= 10``. ``layout_valid`` denotes at least three
    such edges, not target correctness or acceptance. Support uses the same
    maximum-normalized correspondence weights as the existing Top2 decoder.

    Runner-up support is recomputed relative to t: the strongest edge-centered
    mode over 20px from t, divided by t's own support (zero if absent). Residual
    is weighted inlier RMS / 10. With no inliers it is the nearest edge distance
    / 10, or the neutral radius value 1 if there are no candidate edges. Missing
    normal evidence has availability 0 and neutral complementarity .5. Contour
    contact and mask overlap are always recomputed, including invalid layouts.
    Inputs and the supplied translation are never modified or refined.
    """
    t = np.asarray(translation_rc, dtype=np.float64)
    if t.shape != (2,) or not np.isfinite(t).all():
        raise ValueError("translation_rc must be a finite [2] A-to-B translation")
    return geometry_at_translations(points_a, points_b, assignment, valid_a, valid_b,
        mask_a, mask_b, t[None, :])[0]


def geometry_at_translations(points_a, points_b, assignment, valid_a, valid_b,
                             mask_a, mask_b, translations_rc):
    """Target-blind float32 [K,22] evidence, sharing one pair's preprocessing.

    Rows follow the supplied translations in order and are identical to calling
    ``geometry_at_translation`` separately. Top2 candidate extraction, filled
    mask erosion, outward normals, edge-normal votes, the A-contour tree and
    candidate-centered support are reused within the call. No cross-pair cache,
    mutation, refinement, decoder winner, target or supervision is involved.
    ``translations_rc`` must be finite [K,2]; K=0 returns an empty [0,22] array.
    """
    a, b, matrix, va, vb = _validate_inputs(
        points_a, points_b, assignment, valid_a, valid_b)
    translations = np.asarray(translations_rc, dtype=np.float64)
    if translations.shape == (0,):
        translations = translations.reshape(0, 2)
    if (translations.ndim != 2 or translations.shape[1:] != (2,)
            or not np.isfinite(translations).all()):
        raise ValueError("translations_rc must be finite [K,2] A-to-B translations")
    ma, mb = _filled_mask(mask_a, "mask_a"), _filled_mask(mask_b, "mask_b")
    area = min(int(ma.sum()), int(mb.sum()))
    if area <= 0:
        raise ValueError("candidate geometry requires nonempty filled masks")
    if not len(translations):
        return np.empty((0, len(GEOMETRY_NAMES)), dtype=np.float32)

    indices, weight, _ = _candidates(a, b, matrix, va, vb, TOP2_CONFIG)
    count = len(indices)
    radius = TOP2_CONFIG.inlier_radius_px
    delta = b[indices[:, 1]] - a[indices[:, 0]]
    total_weight = float(weight.sum())
    ia, ib = np.flatnonzero(va), np.flatnonzero(vb)
    tree_a = cKDTree(a[ia]) if len(ia) and len(ib) else None
    row, col = np.ogrid[-2:3, -2:3]
    disk = row * row + col * col <= 4
    ea = ndimage.binary_erosion(ma, structure=disk, border_value=0)
    eb = ndimage.binary_erosion(mb, structure=disk, border_value=0)
    mode_support, edge_normals = None, None
    features = np.empty((len(translations), len(GEOMETRY_NAMES)), dtype=np.float32)
    for index, t in enumerate(translations):
        distance = np.linalg.norm(delta - t, axis=1)
        if not np.isfinite(distance).all():
            raise ValueError("candidate displacement residuals must be finite")
        inlier = distance <= radius
        selected, inlier_weight = indices[inlier], weight[inlier]
        support = float(inlier_weight.sum())
        inlier_count = int(inlier.sum())
        norm_weight = inlier_weight / support if support else np.empty(0)
        residual = (float(np.sqrt(np.sum(norm_weight * distance[inlier] ** 2)))
                    if support else (float(distance.min()) if count else radius))
        runner_ratio = 0.
        separate = distance > 2 * radius
        if support and separate.any():
            if mode_support is None:
                squared = np.sum((delta[:, None, :] - delta[None, :, :]) ** 2, axis=2)
                mode_support = (squared <= radius * radius) @ weight
            runner_ratio = float(mode_support[separate].max()) / support

        covered_a, longest_a = arc_evidence(a, va, selected[:, 0])
        covered_b, longest_b = arc_evidence(b, vb, selected[:, 1])
        wa, wb = np.zeros(len(a)), np.zeros(len(b))
        np.maximum.at(wa, selected[:, 0], inlier_weight)
        np.maximum.at(wb, selected[:, 1], inlier_weight)
        independent = min(float(wa.sum()), float(wb.sum())) / support if support else 0.

        near = [0.] * 4
        if tree_a is not None:
            shifted_b = b[ib] - t
            da = cKDTree(shifted_b).query(a[ia])[0]
            db = tree_a.query(shifted_b)[0]
            near = []
            for contact_radius in (3., 5.):
                ca = arc_evidence(a, va, ia[da <= contact_radius])[0]
                cb = arc_evidence(b, vb, ib[db <= contact_radius])[0]
                near.extend((min(ca, cb), max(ca, cb)))

        normal_available, complementary = 0., .5
        if support:
            if edge_normals is None:
                na, good_a = outward_contour_normals(a, va, arc_half_length_px=8.)
                nb, good_b = outward_contour_normals(b, vb, arc_half_length_px=8.)
                good_edge = good_a[indices[:, 0]] & good_b[indices[:, 1]]
                dot = np.sum(na[indices[:, 0]] * nb[indices[:, 1]], axis=1)
                edge_normals = good_edge, (1 - np.clip(dot, -1, 1)) * .5
            good_edge, normal_vote = edge_normals
            good = good_edge[inlier]
            normal_available = float(norm_weight[good].sum())
            if good.any():
                complementary = float(np.sum(norm_weight[good] * normal_vote[inlier][good])
                                      / normal_available)

        overlap = translated_intersection_area(ma, mb, t) / area
        deep = translated_intersection_area(ea, eb, t) / area
        features[index] = (float(inlier_count >= TOP2_CONFIG.min_inliers),
            count / TOP2_CONFIG.max_candidates, inlier_count / count if count else 0.,
            support / total_weight if count else 0., np.log1p(support), runner_ratio,
            residual / radius, independent, min(covered_a, covered_b), max(covered_a, covered_b),
            min(longest_a, longest_b), max(longest_a, longest_b),
            float(norm_weight[distance[inlier] <= 3].sum()),
            float(norm_weight[distance[inlier] <= 5].sum()),
            *near, normal_available, complementary, overlap, deep)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite candidate geometry evidence")
    return features


def generate_train_candidates(gt_translation_rc, predicted_translation_rc, *, seed,
                              correct_count=2, near_wrong_count=2, far_wrong_count=2):
    """TRAIN ONLY: propose translations and compute labels from actual L2 error.

    Return JSON-serializable records with ``kind``, ``translation_rc``,
    ``error_px`` and Boolean ``label`` (L2 <= 10px). The first ``correct``
    proposal is the GT itself; further correct proposals sample uniform radius
    0--5px and uniform angle. ``near_wrong`` uses 12--30px and ``far_wrong``
    40--160px. The unchanged finite prediction is appended as ``predicted``;
    its label is measured, not inferred from the name. None/nonfinite predicted
    translations mean no valid prediction and are omitted, never fabricated.

    Duplicates are intentionally retained as distinct proposal sources. This
    helper must never run on VAL/TEST/REAL or be used to build inference inputs.
    Only each record's translation goes into ``geometry_at_translation``;
    kind, GT, label and error are supervision/provenance, not feature channels.
    """
    gt = np.asarray(gt_translation_rc, dtype=np.float64)
    if gt.shape != (2,) or not np.isfinite(gt).all():
        raise ValueError("TRAIN GT translation must be finite [2]")
    counts = (correct_count, near_wrong_count, far_wrong_count)
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0
           for n in counts):
        raise ValueError("TRAIN candidate counts must be nonnegative integers")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("TRAIN candidate seed must be a nonnegative integer")
    predicted = None
    if predicted_translation_rc is not None:
        predicted = np.asarray(predicted_translation_rc, dtype=np.float64)
        if predicted.shape != (2,):
            raise ValueError("predicted translation must have shape [2] or be None")
    rng = np.random.default_rng(seed)
    records = []

    def append(kind, translation):
        error = float(np.linalg.norm(translation - gt))
        records.append(dict(kind=kind, translation_rc=translation.tolist(),
            error_px=error, label=bool(error <= 10.)))

    for kind, count, low, high in (("correct", correct_count, 0., 5.),
            ("near_wrong", near_wrong_count, 12., 30.),
            ("far_wrong", far_wrong_count, 40., 160.)):
        for index in range(count):
            if kind == "correct" and index == 0:
                proposal = gt.copy()
            else:
                radius, angle = rng.uniform(low, high), rng.uniform(0., 2 * np.pi)
                proposal = gt + radius * np.asarray([np.cos(angle), np.sin(angle)])
            append(kind, proposal)
    if predicted is not None and np.isfinite(predicted).all():
        append("predicted", predicted)
    return records
