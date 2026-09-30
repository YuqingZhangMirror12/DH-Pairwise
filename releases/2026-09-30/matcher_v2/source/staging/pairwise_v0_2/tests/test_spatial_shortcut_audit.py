import numpy as np
import pytest

from staging.pairwise_v0_2.preflight.audit_spatial_shortcuts_v0_2 import (
    mask_spatial_feature,
    spatial_scores,
    weighted_auroc,
)


def test_spatial_scores_use_canvas_coordinates_only():
    left = np.zeros((20, 20), dtype=bool)
    touching = np.zeros_like(left)
    distant = np.zeros_like(left)
    left[5:10, 3:8] = True
    touching[5:10, 8:13] = True
    distant[14:19, 14:19] = True
    close_score = spatial_scores(
        mask_spatial_feature(left), mask_spatial_feature(touching)
    )
    far_score = spatial_scores(mask_spatial_feature(left), mask_spatial_feature(distant))
    assert (
        close_score["negative_foreground_centroid_distance_over_canvas_diagonal"]
        > far_score["negative_foreground_centroid_distance_over_canvas_diagonal"]
    )
    assert (
        close_score["negative_axis_aligned_bbox_gap_over_canvas_diagonal"]
        > far_score["negative_axis_aligned_bbox_gap_over_canvas_diagonal"]
    )
    assert close_score["bbox_intersection_over_union"] == 0.0


def test_weighted_auroc_handles_ties_and_cluster_weights():
    assert weighted_auroc(
        [1.0, 0.8, 0.2, 0.0],
        [True, True, False, False],
        [0.25, 0.25, 0.25, 0.25],
    ) == pytest.approx(1.0)
    assert weighted_auroc(
        [0.5, 0.5], [True, False], [0.5, 0.5]
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="both classes"):
        weighted_auroc([0.2, 0.3], [True, True], [0.5, 0.5])


def test_empty_mask_fails_closed():
    with pytest.raises(ValueError, match="empty mask"):
        mask_spatial_feature(np.zeros((8, 8), dtype=bool))
