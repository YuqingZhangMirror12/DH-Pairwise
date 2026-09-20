import numpy as np
import pytest

from staging.pairwise_v0_2.models.translation_layout import (
    TranslationLayoutConfig,
    estimate_translation_layout,
)


def _diagonal_problem(displacements, weights=None):
    delta = np.asarray(displacements, dtype=np.float64)
    a = np.column_stack((np.arange(len(delta)) * 101.0, np.arange(len(delta)) * 37.0))
    return a, a + delta, np.diag(np.ones(len(delta)) if weights is None else weights)


def test_mode_chooses_dominant_cluster_instead_of_global_average():
    a, b, score = _diagonal_problem([[12, -7]] * 7 + [[-90, 160]] * 5 + [[310, 420]])
    result = estimate_translation_layout(a, b, score)
    assert result.valid
    np.testing.assert_allclose(result.t_a_to_b_rc, [12, -7])
    assert result.inlier_count == 7
    assert result.candidate_count == 13
    assert result.inlier_fraction == pytest.approx(7 / 13)
    assert result.runner_up_support_ratio == pytest.approx(5 / 7)
    assert result.residual_px == 0


def test_mode_support_is_confidence_weighted_and_reports_residual():
    delta = [[10, 5], [11, 5], [9, 5]] + [[-80, 60]] * 5
    a, b, score = _diagonal_problem(delta, [1, 1, 1] + [.1] * 5)
    result = estimate_translation_layout(a, b, score)
    np.testing.assert_allclose(result.t_a_to_b_rc, [10, 5])
    assert result.valid and result.inlier_count == 3
    assert result.residual_px == pytest.approx(np.sqrt(2 / 3))
    assert result.weighted_inlier_fraction == pytest.approx(3 / 3.5)


@pytest.mark.parametrize("decoder", ["mode_consensus", "median_cauchy"])
@pytest.mark.parametrize("correspondence_mode", ["reciprocal_top1", "topk_union"])
@pytest.mark.parametrize("refinement", ["weighted_mean", "weighted_median"])
def test_exchanging_fragments_negates_translation(decoder, correspondence_mode, refinement):
    a, b, score = _diagonal_problem([[12, -7], [13, -7], [11, -7], [12, -6], [12, -8], [90, 70]])
    config = TranslationLayoutConfig(decoder=decoder, correspondence_mode=correspondence_mode, refinement=refinement)
    forward = estimate_translation_layout(a, b, score, config=config)
    reverse = estimate_translation_layout(b, a, score.T, config=config)
    assert forward.valid and reverse.valid
    np.testing.assert_allclose(forward.t_a_to_b_rc, -reverse.t_a_to_b_rc, atol=1e-12)
    assert forward.inlier_count == reverse.inlier_count
    assert forward.runner_up_support_ratio == pytest.approx(reverse.runner_up_support_ratio)


def test_sparse_cauchy_handles_outliers_and_half_mass_median_symmetrically():
    a, b, score = _diagonal_problem([[5, -4], [7, -4], [5, -6], [7, -6], [400, 300], [-400, -300]])
    config = TranslationLayoutConfig(decoder="median_cauchy")
    result = estimate_translation_layout(a, b, score, config=config)
    assert result.valid and result.inlier_count == 4
    np.testing.assert_allclose(result.t_a_to_b_rc, [6, -5])


def test_zero_evidence_and_insufficient_correspondences_are_invalid():
    points = np.zeros((4, 2))
    result = estimate_translation_layout(points, points, np.zeros((4, 4)))
    assert not result.valid and result.reason == "no_candidates"
    assert result.candidate_count == 0 and np.isnan(result.t_a_to_b_rc).all()
    result = estimate_translation_layout(points[:2], points[:2], np.eye(2))
    assert not result.valid and result.reason == "insufficient_inliers"
    assert result.inlier_count == 2


def test_noncontiguous_padding_ignores_nan_and_large_padded_scores():
    a, b, score = _diagonal_problem([[15, 20]] * 3)
    a = np.insert(a, 1, [np.nan, np.inf], axis=0)
    b = np.insert(b, 2, [np.nan, np.inf], axis=0)
    padded = np.full((4, 4), np.nan)
    valid_a = np.array([True, False, True, True])
    valid_b = np.array([True, True, False, True])
    padded[np.ix_(valid_a, valid_b)] = score
    padded[1, :] = 1e100
    result = estimate_translation_layout(a, b, padded, valid_a, valid_b)
    assert result.valid and result.candidate_count == 3
    np.testing.assert_allclose(result.t_a_to_b_rc, [15, 20])
    assert 1 not in result.candidate_indices[:, 0]
    assert 2 not in result.candidate_indices[:, 1]


def test_tiny_transport_is_not_rejected_by_an_absolute_dustbin_threshold():
    a, b, score = _diagonal_problem([[15, 20]] * 4)
    result = estimate_translation_layout(a, b, score * 1e-20)
    assert result.valid and result.candidate_count == 4
    np.testing.assert_allclose(result.t_a_to_b_rc, [15, 20])


def test_topk_union_and_candidate_cap_are_symmetric():
    a, b, score = _diagonal_problem([[10, 20]] * 8, np.linspace(1, 2, 8))
    score += .001
    config = TranslationLayoutConfig(correspondence_mode="topk_union", top_k=2, max_candidates=5)
    forward = estimate_translation_layout(a, b, score, config=config)
    reverse = estimate_translation_layout(b, a, score.T, config=config)
    assert forward.candidate_count == reverse.candidate_count == 5
    np.testing.assert_allclose(forward.t_a_to_b_rc, -reverse.t_a_to_b_rc)
    assert set(map(tuple, forward.candidate_indices)) == set(map(tuple, reverse.candidate_indices[:, ::-1]))


def test_dual_softmax_accepts_raw_negative_affinity_and_is_stable():
    a, b, _ = _diagonal_problem([[10, 20]] * 4)
    logits = np.full((4, 4), -10000.)
    np.fill_diagonal(logits, -9990.)
    config = TranslationLayoutConfig(score_mode="dual_softmax", affinity_temperature=.25)
    result = estimate_translation_layout(a, b, logits, config=config)
    assert result.valid
    np.testing.assert_allclose(result.t_a_to_b_rc, [10, 20])
    reverse = estimate_translation_layout(b, a, logits.T, config=config)
    np.testing.assert_allclose(result.t_a_to_b_rc, -reverse.t_a_to_b_rc)


def test_solver_does_not_fit_rotation():
    a = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=np.float64)
    rotation = np.array([[0, -1], [1, 0]])
    b = a @ rotation.T + [30, 20]
    result = estimate_translation_layout(a, b, np.eye(4))
    assert not result.valid and result.inlier_count == 1


@pytest.mark.parametrize("case", ["nan_point", "nan_score", "negative_confidence", "mask_dtype", "score_shape"])
def test_active_inputs_are_strictly_validated(case):
    a, b, score = _diagonal_problem([[10, 20]] * 4)
    valid_a = None
    if case == "nan_point":
        a[0, 0] = np.nan
    elif case == "nan_score":
        score[0, 0] = np.nan
    elif case == "negative_confidence":
        score[0, 0] = -1
    elif case == "mask_dtype":
        valid_a = np.ones(4, dtype=np.int64)
    else:
        score = score[:3]
    with pytest.raises(ValueError):
        estimate_translation_layout(a, b, score, valid_a)


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"max_candidates": -1}, {"inlier_radius_px": 0},
                                       {"decoder": "ransac"}, {"score_mode": "unknown"},
                                       {"affinity_temperature": float("nan")}])
def test_config_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        TranslationLayoutConfig(**kwargs)
