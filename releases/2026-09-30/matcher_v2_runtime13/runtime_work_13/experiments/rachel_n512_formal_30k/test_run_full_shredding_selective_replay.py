from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from experiments.rachel_n512_formal_30k.run_full_shredding_selective_replay import (
    ROUTE_ARTIFACT_SCHEMA_VERSION,
    ROUTE_ARTIFACT_STATUS,
    SelectiveReplayError,
    _canonical_sha256,
    load_route_artifact,
    route_full_scores,
    validate_cached_replay,
)


def test_route_full_scores_and_frozen_artifact_gate(tmp_path) -> None:
    selected = route_full_scores(
        [0.1, 0.2, 0.3, float("nan")],
        [True, True, False, True],
        0.2,
    )
    assert np.array_equal(selected, [False, True, True, True])

    artifact = {
        "schema_version": ROUTE_ARTIFACT_SCHEMA_VERSION,
        "status": ROUTE_ARTIFACT_STATUS,
        "route": {
            "score": "full_n512.fused_probability",
            "rule": "fail_open_if_invalid_else_probability_greater_than_or_equal_to_threshold",
            "threshold": 0.2,
            "threshold_fit_source": "validation_only",
            "test_or_real_parameter_fit": False,
            "top_k_or_budget_cap": None,
        },
        "classification": {
            "score": "full_n512.fused_probability",
            "threshold": 0.9,
            "shreddingnet_score_used": False,
        },
        "pose": {
            "compute_pose_for_shreddingnet_rejected_pairs": True,
            "unrouted_pairs_are_unconditional_pose_failures": True,
        },
        "protocol": {
            "test_or_real_read_by_this_command": False,
            "single_primary_operating_point": True,
            "threshold_sweep_on_test_or_real_forbidden": True,
        },
        "frozen_checkpoints_sha256": {
            "full_n512": "a" * 64,
            "shreddingnet_adapted": {
                "coarse": "b" * 64,
                "matching": "c" * 64,
                "classify": "d" * 64,
            },
        },
    }
    artifact["content_sha256"] = _canonical_sha256(artifact)
    path = tmp_path / "route_artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert load_route_artifact(path)["route"]["threshold"] == 0.2

    artifact["route"]["test_or_real_parameter_fit"] = True
    artifact.pop("content_sha256")
    artifact["content_sha256"] = _canonical_sha256(artifact)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(SelectiveReplayError, match="protocol fields differ"):
        load_route_artifact(path)


def _fixtures():
    records = [
        {
            "pair_id": "a",
            "routed_to_shreddingnet": True,
            "full": {
                "fused_probability": 0.25,
                "decision_valid": True,
                "translation_hat_rc": [1.0, 2.0],
            },
            "shreddingnet": {
                "pair_probability": 0.75,
                "translation_valid": True,
                "translation_hat_rc": [3.0, 4.0],
            },
        },
        {
            "pair_id": "b",
            "routed_to_shreddingnet": False,
            "full": {
                "fused_probability": 0.01,
                "decision_valid": True,
                "translation_hat_rc": [5.0, 6.0],
            },
            "shreddingnet": None,
        },
    ]
    full = [
        {
            "pair_id": "a",
            "scores": {"fused": {"probability": 0.25}},
            "decision": {"valid": True},
            "geometry": {"translation_hat_rc": [1.0, 2.0]},
        },
        {
            "pair_id": "b",
            "scores": {"fused": {"probability": 0.01}},
            "decision": {"valid": True},
            "geometry": {"translation_hat_rc": [5.0, 6.0]},
        },
    ]
    shred = [
        {
            "pair_id": "a",
            "scores": {"pair_probability": {"probability": 0.75}},
            "geometry": {
                "translation_prediction_valid": True,
                "translation_hat_rc": [3.0, 4.0],
            },
        },
        {
            "pair_id": "b",
            "scores": {"pair_probability": {"probability": 0.1}},
            "geometry": {
                "translation_prediction_valid": False,
                "translation_hat_rc": None,
            },
        },
    ]
    return records, full, shred


def test_cache_parity_checks_only_selected_shredding_pose_and_fails_mismatch() -> None:
    records, full, shred = _fixtures()
    result = validate_cached_replay(
        records,
        full,
        shred,
        score_atol=1e-6,
        pose_atol_px=1e-6,
        relative_tolerance=0.0,
    )
    assert result["status"] == "passed"
    assert result["full_pairs_checked"] == 2
    assert result["shreddingnet_selected_pairs_checked"] == 1

    altered = copy.deepcopy(shred)
    altered[0]["geometry"]["translation_hat_rc"][0] = 30.0
    with pytest.raises(SelectiveReplayError, match="pose cache mismatch"):
        validate_cached_replay(
            records,
            full,
            altered,
            score_atol=1e-6,
            pose_atol_px=1e-6,
            relative_tolerance=0.0,
        )
