from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from staging.pairwise_v0_2.baselines import (
    exact_seam_corrosion_robustness as robustness,
)
from staging.pairwise_v0_2.baselines import (
    exact_seam_post_pilot_real_evaluation as post_pilot,
)
from staging.pairwise_v0_2.baselines.real_dunhuang_evaluation import (
    RealBatchPrediction,
    load_strict_real_pair_dataset,
)
from staging.pairwise_v0_2.tests.test_exact_seam_post_pilot_real_evaluation import (
    _fake_loaded_pair,
)
from staging.pairwise_v0_2.tests.test_real_dunhuang_evaluation import (
    _fixture_documents,
    _geometry_config,
)
from staging.pairwise_v0_2.training.geometry_batch import DATA_DIRECTION_TO_INDEX
from staging.pairwise_v0_2.training.geometry_cache import GeometryArtifactCache


def _square_mask() -> np.ndarray:
    mask = np.zeros((80, 80), dtype=np.bool_)
    mask[8:72, 8:72] = True
    return mask


def _condition(name: str) -> robustness.CorrosionCondition:
    return next(value for value in robustness.FIXED_CONDITIONS if value.name == name)


def test_fixed_profile_has_seven_nested_deterministic_mask_conditions() -> None:
    assert [value.name for value in robustness.FIXED_CONDITIONS] == [
        "clean",
        "erosion_r2",
        "erosion_r4",
        "erosion_r8",
        "break_k1_r8",
        "break_k2_r8",
        "break_k4_r8",
    ]
    mask = _square_mask()
    outputs = {
        condition.name: robustness.degrade_mask(
            mask,
            fragment_id="fragment/fixed-a",
            condition=condition,
            seed=260830,
        )
        for condition in robustness.FIXED_CONDITIONS
    }
    repeated = robustness.degrade_mask(
        mask,
        fragment_id="fragment/fixed-a",
        condition=_condition("break_k4_r8"),
        seed=260830,
    )

    assert np.array_equal(outputs["break_k4_r8"], repeated)
    assert all(value.dtype == np.bool_ and not value.flags.writeable for value in outputs.values())
    assert np.all(outputs["erosion_r8"] <= outputs["erosion_r4"])
    assert np.all(outputs["erosion_r4"] <= outputs["erosion_r2"])
    assert np.all(outputs["erosion_r2"] <= outputs["clean"])
    assert np.all(outputs["break_k4_r8"] <= outputs["break_k2_r8"])
    assert np.all(outputs["break_k2_r8"] <= outputs["break_k1_r8"])
    assert np.all(outputs["break_k1_r8"] <= outputs["clean"])
    assert outputs["erosion_r8"].sum() < outputs["erosion_r2"].sum()
    assert outputs["break_k4_r8"].sum() < outputs["break_k1_r8"].sum()


def test_evaluator_shares_blinded_geometry_and_emits_two_fixed_curves(
    tmp_path: Path,
) -> None:
    manifest, receipt = _fixture_documents(tmp_path)
    dataset = load_strict_real_pair_dataset(
        manifest,
        receipt,
        target_long_side=96,
        expected_case_count=2,
        expected_pair_count=4,
        expected_positive_count=3,
        expected_negative_count=1,
    )
    labels = {record.pair_id: record.label for record in dataset.records}
    directions = {
        record.pair_id: (
            -1
            if record.direction_b_wrt_a is None
            else DATA_DIRECTION_TO_INDEX[record.direction_b_wrt_a]
        )
        for record in dataset.records
    }
    calls = []

    def scorer(loaded, prepared):
        payload = prepared.payload
        calls.append(
            {
                "arm": loaded.arm.name.value,
                "prepared_id": id(prepared),
                "pair_id": payload.sample_ids[0],
                "labels": payload.labels.clone(),
                "direction_valid": payload.direction_target_valid.clone(),
                "has_exact": payload.exact_loss_targets() is not None,
            }
        )
        pair_id = payload.sample_ids[0]
        is_exact = loaded.arm.name is post_pilot._METHOD_ARM[post_pilot.EXACT_METHOD]
        probability = (
            0.95 if is_exact else 0.9
        ) if labels[pair_id] else (0.05 if is_exact else 0.1)
        direction = max(0, directions[pair_id])
        return RealBatchPrediction(
            probability=torch.tensor([probability], dtype=torch.float32),
            valid=torch.tensor([True], dtype=torch.bool),
            best_direction_index=torch.tensor([direction], dtype=torch.long),
        )

    expected = {
        method: {
            "sample_count": 4.0,
            "positive_count": 3.0,
            "negative_count": 1.0,
            "auroc": 1.0,
            "auprc": 1.0,
        }
        for method in (robustness.WEAK_METHOD, robustness.EXACT_METHOD)
    }
    models = _fake_loaded_pair()
    try:
        result = robustness.evaluate_exact_seam_corrosion_robustness(
            records=dataset.records,
            base_mask_loader=dataset.mask_loader,
            geometry_config=_geometry_config(),
            geometry_cache=GeometryArtifactCache(tmp_path / "corrosion-cache"),
            models=models,
            expected_clean_metrics=expected,
            seed=models.seed,
            arm_scorer=scorer,
            expected_pair_count=4,
        )
    finally:
        dataset.mask_loader.close()

    assert result["clean_reproduction_gate"]["passed"] is True
    assert result["pilot"]["corruption_seed"] == models.seed
    assert result["preprocessing"]["corruption_seed"] == models.seed
    assert result["validation"]["population_reconstruction"][
        "validation_fraction"
    ] == pytest.approx(0.2)
    assert result["condition_order"] == [
        condition.name for condition in robustness.FIXED_CONDITIONS
    ]
    assert set(result["family_fixed_intersection_curves"]) == {"erosion", "break"}
    assert result["processing"]["pair_condition_count"] == 28
    assert result["processing"]["planned_weak_exact_pair_forwards"] == 56
    assert len(result["pairs"]) == 4
    assert [row["pair_id"] for row in result["pairs"]] == [
        record.pair_id for record in dataset.records
    ]
    assert len(calls) == 56
    for weak, exact in zip(calls[::2], calls[1::2]):
        assert weak["prepared_id"] == exact["prepared_id"]
        assert weak["pair_id"] == exact["pair_id"]
        assert not weak["labels"].any()
        assert not exact["labels"].any()
        assert not weak["direction_valid"].any()
        assert not exact["direction_valid"].any()
        assert weak["has_exact"] is False
        assert exact["has_exact"] is False
    for condition in result["conditions"].values():
        assert condition["common_valid_count"] == 4
        assert condition["methods"]["weak"]["common_valid_ranking"]["row"][
            "auroc"
        ] == pytest.approx(1.0)
        assert condition["methods"]["exact"]["direction_on_common_valid"][
            "accuracy_invalid_as_incorrect"
        ] == pytest.approx(1.0)
    for curve in result["family_fixed_intersection_curves"].values():
        assert curve["fixed_pair_count"] == 4
        assert len(curve["points"]) == 4


def test_family_curve_uses_one_valid_intersection_across_all_family_levels() -> None:
    labels = (True, False, True, False)
    clusters = ("a", "a", "b", "b")
    targets = (1, -1, 3, -1)
    values = {}
    for condition in robustness.FIXED_CONDITIONS:
        values[condition.name] = {}
        for method in (robustness.WEAK_METHOD, robustness.EXACT_METHOD):
            valid = [True, True, True, True]
            if condition.name == "erosion_r8" and method == robustness.EXACT_METHOD:
                valid[0] = False
            if condition.name == "break_k4_r8" and method == robustness.WEAK_METHOD:
                valid[1] = False
            values[condition.name][method] = {
                "probability": [0.9, 0.1, 0.8, 0.2],
                "valid": valid,
                "direction": [1, 0, 3, 0],
            }

    erosion = robustness._fixed_intersection_curve(
        "erosion",
        values,
        labels=labels,
        clusters=clusters,
        direction_targets=targets,
    )
    breaking = robustness._fixed_intersection_curve(
        "break",
        values,
        labels=labels,
        clusters=clusters,
        direction_targets=targets,
    )

    assert erosion["fixed_pair_count"] == 3
    assert erosion["fixed_positive_count"] == 1
    assert breaking["fixed_pair_count"] == 3
    assert breaking["fixed_negative_count"] == 1
    assert all(
        point["methods"]["weak"]["ranking"]["sample_count"] == 3
        for point in erosion["points"]
    )
    assert all(
        point["methods"]["exact"]["ranking"]["sample_count"] == 3
        for point in breaking["points"]
    )


def test_clean_reproduction_gate_rejects_a_metric_mismatch() -> None:
    observed = {
        method: {
            "sample_count": 200.0,
            "positive_count": 100.0,
            "negative_count": 100.0,
            "auroc": 0.7,
            "auprc": 0.6,
        }
        for method in (robustness.WEAK_METHOD, robustness.EXACT_METHOD)
    }
    expected = {
        method: dict(row) for method, row in observed.items()
    }
    expected[robustness.EXACT_METHOD]["auroc"] = 0.71

    with pytest.raises(
        robustness.CorrosionRobustnessError,
        match="did not reproduce",
    ):
        robustness._clean_reproduction_gate(
            observed,
            expected,
            tolerance=1e-6,
        )


def test_clean_gate_stops_before_any_corrupted_condition_forward(
    tmp_path: Path,
) -> None:
    manifest, receipt = _fixture_documents(tmp_path)
    dataset = load_strict_real_pair_dataset(
        manifest,
        receipt,
        target_long_side=96,
        expected_case_count=2,
        expected_pair_count=4,
        expected_positive_count=3,
        expected_negative_count=1,
    )
    calls = []

    def uninformative_scorer(loaded, prepared):
        calls.append((loaded.arm.name.value, prepared.payload.sample_ids[0]))
        return RealBatchPrediction(
            probability=torch.tensor([0.5], dtype=torch.float32),
            valid=torch.tensor([True], dtype=torch.bool),
            best_direction_index=torch.tensor([0], dtype=torch.long),
        )

    expected = {
        method: {
            "sample_count": 4.0,
            "positive_count": 3.0,
            "negative_count": 1.0,
            "auroc": 1.0,
            "auprc": 1.0,
        }
        for method in (robustness.WEAK_METHOD, robustness.EXACT_METHOD)
    }
    models = _fake_loaded_pair()
    try:
        with pytest.raises(
            robustness.CorrosionRobustnessError,
            match="did not reproduce",
        ):
            robustness.evaluate_exact_seam_corrosion_robustness(
                records=dataset.records,
                base_mask_loader=dataset.mask_loader,
                geometry_config=_geometry_config(),
                geometry_cache=GeometryArtifactCache(tmp_path / "early-gate-cache"),
                models=models,
                expected_clean_metrics=expected,
                seed=models.seed,
                arm_scorer=uninformative_scorer,
                expected_pair_count=4,
            )
    finally:
        dataset.mask_loader.close()

    # Four clean pairs times two arms.  The other six conditions never start.
    assert len(calls) == 8


def test_rebuild_uses_explicit_fixed_point_two_fraction(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def population_builder(config, *, record_filter):
        del record_filter
        captured["config"] = config
        return SimpleNamespace(validation_records=tuple(range(200)))

    monkeypatch.setattr(
        robustness,
        "build_exact_seam_pilot_population",
        population_builder,
    )
    summary_config = {
        "max_pairs": 1000,
        "train_pairs": 800,
        "validation_pairs": 200,
        "epochs": 3,
        "batch_size": 8,
        "seed": 260830,
        "generator": "gen4voronoi",
        "exact_loss_weight": 0.25,
    }

    records, _ = robustness._rebuild_validation_population(
        summary_path=tmp_path / "run" / "exact_seam_pilot_summary.json",
        summary_config=summary_config,
        mask_root=tmp_path / "no_erode",
        pilot_cache=GeometryArtifactCache(tmp_path / "pilot-cache"),
        geometry_config=_geometry_config(),
    )

    assert len(records) == 200
    assert captured["config"].max_pairs == 1000
    assert captured["config"].train_pair_count == 800
    assert captured["config"].validation_pair_count == 200
    assert captured["config"].validation_fraction == pytest.approx(0.2)


def test_rebuild_rejects_nondefault_counts_instead_of_inferring_fraction(
    tmp_path: Path,
) -> None:
    summary_config = {
        # The previous implementation inferred 200/1250 == 0.16 and silently
        # reconstructed a different population.  v0.1 supports only 0.2.
        "max_pairs": 1250,
        "train_pairs": 1050,
        "validation_pairs": 200,
        "epochs": 3,
        "batch_size": 8,
        "seed": 260830,
        "generator": "gen4voronoi",
        "exact_loss_weight": 0.25,
    }

    with pytest.raises(
        robustness.CorrosionRobustnessError,
        match="only supports the fixed default pilot run",
    ):
        robustness._rebuild_validation_population(
            summary_path=tmp_path / "exact_seam_pilot_summary.json",
            summary_config=summary_config,
            mask_root=tmp_path / "no_erode",
            pilot_cache=GeometryArtifactCache(tmp_path / "pilot-cache"),
            geometry_config=_geometry_config(),
        )


def test_corruption_seed_must_equal_loaded_pilot_seed(tmp_path: Path) -> None:
    models = _fake_loaded_pair()

    with pytest.raises(
        robustness.CorrosionRobustnessError,
        match="must equal the completed pilot seed",
    ):
        robustness.evaluate_exact_seam_corrosion_robustness(
            records=(),
            base_mask_loader=lambda reference: np.zeros((8, 8), dtype=np.bool_),
            geometry_config=_geometry_config(),
            geometry_cache=GeometryArtifactCache(tmp_path / "seed-cache"),
            models=models,
            expected_clean_metrics={},
            seed=models.seed + 1,
            expected_pair_count=0,
        )


def test_cli_is_fixed_to_pair_local_inference() -> None:
    parsed = robustness._parser().parse_args(
        [
            "--pilot-summary",
            "pilot/exact_seam_pilot_summary.json",
            "--mask-root",
            "data/gen4voronoi/no_erode",
            "--pilot-cache-dir",
            "pilot/cache",
            "--geometry-cache-dir",
            "robustness/cache",
            "--output",
            "robustness/result.json",
        ]
    )
    config = robustness.CorrosionRobustnessConfig(
        pilot_summary=parsed.pilot_summary,
        mask_root=parsed.mask_root,
        pilot_cache_dir=parsed.pilot_cache_dir,
        geometry_cache_dir=parsed.geometry_cache_dir,
        output=parsed.output,
        device=parsed.device,
        batch_size=parsed.batch_size,
        clean_tolerance=parsed.clean_tolerance,
    )

    assert config.device == "cuda"
    assert config.batch_size == 1
    assert config.clean_tolerance == pytest.approx(1e-6)
    with pytest.raises(ValueError, match="batch_size=1"):
        robustness.CorrosionRobustnessConfig(
            pilot_summary=Path("pilot.json"),
            mask_root=Path("masks"),
            pilot_cache_dir=Path("pilot-cache"),
            geometry_cache_dir=Path("cache"),
            output=Path("result.json"),
            batch_size=2,
        )
