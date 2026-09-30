from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from staging.pairwise_v0_2.baselines import mm_validation_comparison as comparison
from staging.pairwise_v0_2.baselines import real_dunhuang_evaluation as real_evaluation
from staging.pairwise_v0_2.baselines.historical_mm_siamese import (
    HISTORICAL_MM_BASELINE_ID,
)
from staging.pairwise_v0_2.baselines.real_dunhuang_evaluation import (
    MATCHED_ROUTE_A_SIAMESE_ID,
    RealBatchPrediction,
    RealDunhuangEvaluationError,
    evaluate_real_dunhuang_four_arms,
    load_strict_real_pair_dataset,
    prepare_real_geometry_batch,
    score_loaded_winner_direct,
)
from staging.pairwise_v0_2.geometry import CandidateBuilderConfig
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.geometry_cache import GeometryArtifactCache
from staging.pairwise_v0_2.training.local_q1_backend import (
    LocalQ1Backend,
    LocalQ1BackendMode,
)
from staging.pairwise_v0_2.training.short_ablation import AblationArmName


def _alpha_sha(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask, dtype=np.bool_).tobytes(order="C")
    ).hexdigest()


def _write_rgba(path: Path, mask: np.ndarray) -> None:
    value = np.zeros((*mask.shape, 4), dtype=np.uint8)
    value[..., :3] = np.array((90, 70, 50), dtype=np.uint8)
    value[..., 3] = mask.astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="RGBA").save(path)


def _mask(kind: int) -> np.ndarray:
    value = np.zeros((96, 96), dtype=np.bool_)
    if kind == 0:
        value[12:84, 18:48] = True
    elif kind == 1:
        value[12:84, 48:78] = True
    else:
        value[18:48, 18:78] = True
    return value


def _fixture_documents(tmp_path: Path):
    main = tmp_path / "main"
    supp = tmp_path / "supp"
    manifest_sha = "a" * 64
    cases = []
    local_cases = {}

    def add_case(
        uid: str,
        group: str,
        masks,
        boxes,
        pair_labels,
        *,
        category="ground_truth_simple",
        disposition="eligible",
    ):
        directory_name = (
            "Ground Truth Simple"
            if category == "ground_truth_simple"
            else ("Small" if category == "small" else "Issue")
        )
        group_dir = main / directory_name / group
        fragments = []
        paths = []
        for index, (mask, bbox) in enumerate(zip(masks, boxes), start=1):
            path = group_dir / "{}.png".format(index)
            _write_rgba(path, mask)
            paths.append(str(path))
            fragments.append(
                {
                    "fragment_id": index,
                    "has_alpha": True,
                    "alpha_mask_sha256": _alpha_sha(mask),
                    "bbox_xyxy": list(bbox),
                }
            )
        occurrence_uid = "occ-" + uid
        cases.append(
            {
                "case_uid": uid,
                "canonical_collection": "main",
                "canonical_category": category,
                "observed_categories": [category],
                "disposition": disposition,
                "fragments": fragments,
                "pair_labels": list(pair_labels),
                "occurrences": [
                    {
                        "occurrence_uid": occurrence_uid,
                        "collection": "main",
                        "category": category,
                    }
                ],
            }
        )
        local_cases[uid] = {
            "occurrences": [
                {
                    "occurrence_uid": occurrence_uid,
                    "collection": "main",
                    "category": category,
                    "fragment_paths": paths,
                }
            ]
        }

    # The two alpha masks are identical while their GT origins differ.  This
    # makes the no-bbox-origin model-input assertion directly testable.
    identical = _mask(0)
    add_case(
        "dhcase-fixture-positive",
        "1",
        (identical, identical.copy()),
        ((10, 20, 106, 116), (220, 20, 316, 116)),
        ({"fragment_a": 1, "fragment_b": 2, "label": "positive"},),
    )
    add_case(
        "dhcase-fixture-mixed",
        "2",
        (_mask(0), _mask(1), _mask(2)),
        (
            (0, 0, 96, 96),
            (110, 0, 206, 96),
            (110, 120, 206, 216),
        ),
        (
            {"fragment_a": 1, "fragment_b": 2, "label": "positive"},
            {"fragment_a": 1, "fragment_b": 3, "label": "negative"},
            {"fragment_a": 2, "fragment_b": 3, "label": "positive"},
        ),
        category="small",
    )
    # These are present but must never enter the strict adapter.
    add_case(
        "dhcase-fixture-issue",
        "3",
        (_mask(0), _mask(1)),
        ((0, 0, 96, 96), (110, 0, 206, 96)),
        ({"fragment_a": 1, "fragment_b": 2, "label": "positive"},),
        category="issue",
    )
    manifest = {
        "schema_version": "pairwise-v0.2-real-external-test/0.1",
        "manifest_sha256": manifest_sha,
        "cases": cases,
    }
    receipt = {
        "schema_version": "pairwise-v0.2-real-external-test-local-receipt/0.1",
        "portable_manifest_sha256": manifest_sha,
        "dataset_roots": {"main": str(main), "supp": str(supp)},
        "cases": local_cases,
    }
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "paths.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return manifest_path, receipt_path


def _geometry_config() -> GeometryBatchConfig:
    geometry = CandidateBuilderConfig(
        window_scale_fractions=(0.12,),
        window_min_px=2.0,
        window_max_px=128.0,
        output_size=(12, 14),
        min_run_length_fraction=0.0,
        min_run_length_px=2.0,
        side_resample_count=24,
    )
    return GeometryBatchConfig(
        geometry=geometry,
        coarse_output_size=(48, 56),
        max_batch_size=8,
        max_candidates_per_sample=256,
        max_candidates_per_batch=512,
        max_sequence_length=1024,
        max_local_tensor_elements=20_000_000,
    )


def test_strict_adapter_is_alpha_only_and_never_encodes_bbox_origin(tmp_path):
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
    try:
        assert dataset.case_count == 2
        assert len(dataset.records) == 4
        assert dataset.positive_count == 3
        assert dataset.negative_count == 1
        assert {
            record.provenance["canonical_category"] for record in dataset.records
        } == {
            "ground_truth_simple",
            "small",
        }
        first = dataset.records[0]
        mask_a = dataset.mask_loader(first.fragment_a)
        mask_b = dataset.mask_loader(first.fragment_b)
        assert np.array_equal(mask_a, mask_b)
        assert mask_a.flags.writeable is False
        assert first.direction_b_wrt_a == "right"
    finally:
        dataset.mask_loader.close()


def test_alpha_hash_change_fails_before_geometry(tmp_path):
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
    source_path = tmp_path / "main/Ground Truth Simple/1/1.png"
    changed = np.zeros((96, 96), dtype=np.bool_)
    changed[5:15, 5:15] = True
    _write_rgba(source_path, changed)
    try:
        with pytest.raises(RealDunhuangEvaluationError, match="differs from manifest"):
            dataset.mask_loader(dataset.records[0].fragment_a)
    finally:
        dataset.mask_loader.close()


def test_four_arm_fixture_reports_pair_metrics_and_direction_accuracy(tmp_path):
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
    config = _geometry_config()
    cache = GeometryArtifactCache(tmp_path / "geometry-cache")
    backend = LocalQ1Backend(device="cpu", mode=LocalQ1BackendMode.FORMAL)

    def winner_loader(name):
        arm = comparison._arm_from_backend(backend, AblationArmName(name))
        return comparison.LoadedArmWinner(
            arm=arm,
            session=object(),
            runner_mm_report={},
        )

    def perfect_scorer(loaded, prepared):
        del loaded
        payload = prepared.payload
        probability = torch.where(
            payload.labels,
            torch.full(payload.labels.shape, 0.9, dtype=torch.float32),
            torch.full(payload.labels.shape, 0.1, dtype=torch.float32),
        )
        best = torch.where(
            payload.direction_target_valid,
            payload.direction_target,
            torch.zeros_like(payload.direction_target),
        )
        return RealBatchPrediction(
            probability=probability,
            valid=torch.ones_like(payload.labels),
            best_direction_index=best,
        )

    try:
        result = evaluate_real_dunhuang_four_arms(
            dataset=dataset,
            geometry_config=config,
            geometry_cache=cache,
            arm_winner_loader=winner_loader,
            arm_scorer=perfect_scorer,
            batch_size=2,
        )
    finally:
        dataset.mask_loader.close()

    assert result["pair_count"] == 4
    assert result["cluster_ids"] == [
        record.component_id for record in dataset.records
    ]
    assert result["positive_count"] == 3
    assert result["negative_count"] == 1
    assert result["common_valid_count"] == 4
    assert result["preprocessing"] == {
        "pixel_source": "fragment_png_alpha_only_threshold_ge_128",
        "rgb_used": False,
        "gt_composite_used": False,
        "bbox_origin_or_canvas_coordinate_exposed_to_model": False,
        "bbox_used_for": "positive_direction_target_only",
        "normalization": (
            "tight_alpha_crop_then_common_scale_within_case_from_alpha_"
            "fragment_dimensions_only"
        ),
        "target_long_side": 96,
        "rotation_search": False,
    }
    assert len(result["methods"]) == 4
    for row in result["methods"].values():
        assert row["common_valid_metrics"]["row"]["auroc"] == pytest.approx(1.0)
        assert row["common_valid_metrics"]["row"]["auprc"] == pytest.approx(1.0)
        assert row["direction_common_valid"][
            "accuracy_invalid_as_incorrect"
        ] == pytest.approx(1.0)


class _OrderedHistoricalProbability(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.register_buffer("probability", torch.tensor(probability, dtype=torch.float32))
        self.offset = 0
        self.observed_shapes = []

    def forward(self, input_a, input_b):
        assert input_a.shape[1:] == (1, 64, 64)
        assert input_b.shape[1:] == (1, 64, 64)
        self.observed_shapes.append((tuple(input_a.shape), tuple(input_b.shape)))
        count = input_a.shape[0]
        start = self.offset
        self.offset += count
        return self.probability[start : start + count, None]


@pytest.mark.parametrize(
    (
        "include_historical",
        "include_matched",
        "method_count",
        "population",
        "status_fragment",
    ),
    (
        (True, False, 5, "five", "plus_historical_mm"),
        (False, True, 5, "five", "plus_matched_siamese"),
        (True, True, 6, "six", "plus_two_siamese_controls"),
    ),
)
def test_optional_siamese_controls_share_real_pair_ids_labels_and_alpha(
    tmp_path,
    include_historical,
    include_matched,
    method_count,
    population,
    status_fragment,
):
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
    config = _geometry_config()
    cache = GeometryArtifactCache(tmp_path / "historical-geometry-cache")
    backend = LocalQ1Backend(device="cpu", mode=LocalQ1BackendMode.FORMAL)

    def winner_loader(name):
        return comparison.LoadedArmWinner(
            arm=comparison._arm_from_backend(backend, AblationArmName(name)),
            session=object(),
            runner_mm_report={},
        )

    def perfect_scorer(loaded, prepared):
        del loaded
        payload = prepared.payload
        return RealBatchPrediction(
            probability=torch.where(
                payload.labels,
                torch.full(payload.labels.shape, 0.9, dtype=torch.float32),
                torch.full(payload.labels.shape, 0.1, dtype=torch.float32),
            ),
            valid=torch.ones_like(payload.labels),
            best_direction_index=torch.where(
                payload.direction_target_valid,
                payload.direction_target,
                torch.zeros_like(payload.direction_target),
            ),
        )

    historical = (
        _OrderedHistoricalProbability((0.9, 0.8, 0.1, 0.7))
        if include_historical
        else None
    )
    matched = (
        _OrderedHistoricalProbability((0.8, 0.7, 0.2, 0.6))
        if include_matched
        else None
    )
    try:
        result = evaluate_real_dunhuang_four_arms(
            dataset=dataset,
            geometry_config=config,
            geometry_cache=cache,
            arm_winner_loader=winner_loader,
            arm_scorer=perfect_scorer,
            batch_size=2,
            historical_model=historical,
            historical_device="cpu",
            historical_batch_size=3,
            matched_model=matched,
            matched_device="cpu",
            matched_batch_size=2,
        )
    finally:
        dataset.mask_loader.close()

    assert status_fragment in result["status"]
    assert result["pair_ids"] == [record.pair_id for record in dataset.records]
    assert result["labels"] == [record.label for record in dataset.records]
    assert result["same_pair_ids_all_methods"] is True
    assert result["same_labels_all_methods"] is True
    assert result["primary_metric_population"] == (
        "intersection_valid_across_all_{}_methods".format(population)
    )
    assert len(result["methods"]) == method_count
    assert result["historical_mm_checkpoint_evaluated"] is include_historical
    assert result["matched_route_a_siamese_checkpoint_evaluated"] is include_matched
    controls = (
        (
            HISTORICAL_MM_BASELINE_ID,
            historical,
            "historical_mm_30k_checkpoint",
            "ordered_A_then_B_historical_checkpoint_join_probability",
        ),
        (
            MATCHED_ROUTE_A_SIAMESE_ID,
            matched,
            "route_a_matched_training_winner",
            "ordered_A_then_B_route_a_matched_checkpoint_join_probability",
        ),
    )
    for method_id, model, role, semantics in controls:
        if model is None:
            assert method_id not in result["methods"]
            continue
        row = result["methods"][method_id]
        assert row["native_valid_count"] == 4
        assert row["valid"] == [True] * 4
        assert row["native_metrics"]["row"]["auroc"] == pytest.approx(1.0)
        assert row["native_metrics"]["row"]["auprc"] == pytest.approx(1.0)
        assert row["common_valid_metrics"]["row"]["auroc"] == pytest.approx(1.0)
        assert row["common_valid_metrics"]["row"]["auprc"] == pytest.approx(1.0)
        assert row["direction_common_valid"] is None
        assert row["checkpoint_role"] == role
        assert row["score_semantics"] == semantics
        assert row["model_input"] == "same_strict_real_alpha_masks_as_four_local_arms"
    if historical is not None:
        assert historical.observed_shapes == [
            ((3, 1, 64, 64), (3, 1, 64, 64)),
            ((1, 1, 64, 64), (1, 1, 64, 64)),
        ]
    if matched is not None:
        assert matched.observed_shapes == [
            ((2, 1, 64, 64), (2, 1, 64, 64)),
            ((2, 1, 64, 64), (2, 1, 64, 64)),
        ]


def test_cli_keeps_historical_checkpoint_optional():
    old_arguments = [
        "--manifest",
        "manifest.json",
        "--local-path-receipt",
        "paths.json",
        "--route-config",
        "route.json",
        "--run-directory",
        "run",
        "--geometry-cache-dir",
        "cache",
        "--output",
        "result.json",
    ]
    parsed_old = real_evaluation._parser().parse_args(old_arguments)
    assert parsed_old.historical_checkpoint is None
    assert parsed_old.historical_batch_size == 256
    assert parsed_old.matched_checkpoint is None
    assert parsed_old.matched_batch_size == 256

    parsed_new = real_evaluation._parser().parse_args(
        old_arguments
        + [
            "--historical-checkpoint",
            "mobilenet.pt",
            "--historical-batch-size",
            "64",
            "--matched-checkpoint",
            "matched.pt",
            "--matched-batch-size",
            "32",
        ]
    )
    assert parsed_new.historical_checkpoint == Path("mobilenet.pt")
    assert parsed_new.historical_batch_size == 64
    assert parsed_new.matched_checkpoint == Path("matched.pt")
    assert parsed_new.matched_batch_size == 32


def test_direct_forward_exposes_best_direction_without_runner_change(tmp_path):
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
    backend = LocalQ1Backend(device="cpu", mode=LocalQ1BackendMode.FORMAL)
    arm = comparison._arm_from_backend(backend, AblationArmName.LOCAL_DUAL_SOFTMAX)
    session = backend.create_session(arm, seed=123)
    prepared = prepare_real_geometry_batch(
        dataset.records[:1],
        mask_loader=dataset.mask_loader,
        geometry_config=_geometry_config(),
        geometry_cache=GeometryArtifactCache(tmp_path / "direct-cache"),
        arm=arm,
    )
    try:
        prediction = score_loaded_winner_direct(
            comparison.LoadedArmWinner(
                arm=arm,
                session=session,
                runner_mm_report={},
            ),
            prepared,
        )
    finally:
        dataset.mask_loader.close()
    assert prediction.probability.shape == (1,)
    assert prediction.valid.shape == (1,)
    assert prediction.best_direction_index.shape == (1,)
    assert prediction.best_direction_index.dtype == torch.long
