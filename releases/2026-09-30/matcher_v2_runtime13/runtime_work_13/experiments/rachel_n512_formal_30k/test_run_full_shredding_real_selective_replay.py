from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.rachel_n512_formal_30k import (
    run_full_shredding_real_selective_replay as subject,
)


def _records_and_cache():
    records = []
    cached = []
    for index in range(subject.EXPECTED_PAIRS):
        selected = index % 4 != 0
        full_score = 0.25 if selected else 0.1
        pose_valid = index % 7 != 0
        shred_pose = [3.0, -4.0] if pose_valid else None
        records.append(
            {
                "pair_id": "p{:04d}".format(index),
                "label": index < 508,
                "case_cluster": "c{:04d}".format(index),
                "route_threshold": 0.2,
                "routed_to_shreddingnet": selected,
                "full": {
                    "fused_probability": full_score,
                    "decision_valid": True,
                    "decision_at_frozen_threshold": False,
                    "translation_hat_rc": [1.0, 2.0],
                },
                "shreddingnet": (
                    {
                        "pair_probability": 0.75,
                        "translation_valid": pose_valid,
                        "translation_hat_rc": shred_pose,
                    }
                    if selected
                    else None
                ),
            }
        )
        cached.append(
            {
                "pair_id": "p{:04d}".format(index),
                "label": index < 508,
                "case_cluster": "c{:04d}".format(index),
                "methods": {
                    "full_n512": {
                        "probability": full_score,
                        "valid": True,
                        "decision_at_frozen_validation_threshold": False,
                        "translation_hat_rc_unsupervised": [1.0, 2.0],
                    },
                    "shreddingnet_adapted": {
                        "probability": 0.75,
                        "valid": True,
                        "translation_hat_rc_unsupervised": shred_pose,
                    },
                },
            }
        )
    return records, cached


def test_cached_real_parity_checks_full_and_only_selected_shredding() -> None:
    records, cached = _records_and_cache()
    result = subject.validate_cached_real_replay(
        records,
        cached,
        score_atol=0.0,
        pose_atol_px=0.0,
        relative_tolerance=0.0,
    )
    assert result["status"] == "complete_cross_node_runtime_parity_diagnostics"
    assert result["full_pairs_checked"] == 1016
    assert result["shreddingnet_selected_pairs_checked"] == 762


def test_cached_real_parity_rejects_order_change() -> None:
    records, cached = _records_and_cache()
    cached[0], cached[1] = cached[1], cached[0]
    with pytest.raises(subject.RealSelectiveReplayError, match="pair order"):
        subject.validate_cached_real_replay(
            records,
            cached,
            score_atol=0.0,
            pose_atol_px=0.0,
            relative_tolerance=0.0,
        )


def test_unused_full_pose_difference_is_reported_not_gated() -> None:
    records, cached = _records_and_cache()
    records[0]["full"]["translation_hat_rc"] = [11.0, 2.0]
    result = subject.validate_cached_real_replay(
        records,
        cached,
        score_atol=0.0,
        pose_atol_px=0.0,
        relative_tolerance=0.0,
    )
    assert result["status"] == "complete_cross_node_runtime_parity_diagnostics"
    assert result["route_set_exact"] is True
    assert result["full_pose_role"] == "unused_diagnostic_only"
    assert result["full_unused_pose_tolerance_mismatch_count"] == 1


def test_numeric_drift_is_reported_when_discrete_runtime_outputs_match() -> None:
    records, cached = _records_and_cache()
    records[0]["full"]["fused_probability"] = 0.11
    records[1]["shreddingnet"]["pair_probability"] = 0.8
    records[1]["shreddingnet"]["translation_hat_rc"] = [3.1, -4.0]
    result = subject.validate_cached_real_replay(
        records,
        cached,
        score_atol=0.0,
        pose_atol_px=0.0,
        relative_tolerance=0.0,
    )
    assert result["route_set_exact"] is True
    assert result["numeric_tolerance_mismatch_count"] == {
        "full_fused_probability": 1,
        "full_translation_component_unused": 0,
        "shreddingnet_pair_probability_diagnostic": 1,
        "shreddingnet_translation_component": 1,
    }


def test_shredding_pose_validity_drift_is_published_as_diagnostic() -> None:
    records, cached = _records_and_cache()
    records[1]["shreddingnet"]["translation_valid"] = False
    records[1]["shreddingnet"]["translation_hat_rc"] = None
    result = subject.validate_cached_real_replay(
        records,
        cached,
        score_atol=0.0,
        pose_atol_px=0.0,
        relative_tolerance=0.0,
    )
    assert result["shreddingnet_translation_validity_exact"] is False
    assert result["discrete_mismatch_count"][
        "shreddingnet_translation_validity"
    ] == 1


def test_summary_has_balanced_population_and_route_counts() -> None:
    records, _ = _records_and_cache()
    result = subject.summarize_records(records)
    assert result["population"] == {"total": 1016, "positive": 508, "negative": 508}
    assert result["routing"]["selected_count"] == 762


def test_runner_releases_full_before_moving_shredding_to_cuda() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    body = source[source.index("def run_real_selective_replay") :]
    release = body.index('full.model.to("cpu")')
    move_shred = body.index("_move_shredding_to_cuda(frozen_shred, device)")
    assert release < move_shred


def test_move_shredding_updates_inner_runtime_device() -> None:
    class Module:
        def __init__(self) -> None:
            self.device = None

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    predictor = SimpleNamespace(
        coarse=Module(), classify=Module(), device=subject.torch.device("cpu")
    )
    frozen = SimpleNamespace(
        _predictor=predictor, _device=subject.torch.device("cpu")
    )
    target = subject.torch.device("cuda:0")

    subject._move_shredding_to_cuda(frozen, target)

    assert predictor.device == target
    assert frozen._device == target
    assert predictor.coarse.device == target
    assert predictor.classify.device == target


def test_config_rejects_non_cuda_device(tmp_path: Path) -> None:
    values = {name: tmp_path / name for name in (
        "n512_run_directory",
        "shreddingnet_freeze_path",
        "route_artifact",
        "real_manifest",
        "real_local_receipt",
        "real_main_root",
        "real_supp_root",
        "cached_real_pair_only",
        "output_root",
    )}
    with pytest.raises(ValueError, match="CUDA"):
        subject.RealSelectiveReplayConfig(**values, device="cpu")
