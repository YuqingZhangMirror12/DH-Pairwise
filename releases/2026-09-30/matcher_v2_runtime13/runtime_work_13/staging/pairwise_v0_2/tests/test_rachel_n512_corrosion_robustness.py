from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from staging.pairwise_v0_2.baselines import (
    rachel_n512_corrosion_robustness as corrosion,
)
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import (
    extract_ordered_outer_contour,
)


def _square(size: int = 96) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.bool_)
    mask[16:80, 18:78] = True
    return mask


def _prediction(
    pair_id: str,
    probability: float,
    *,
    valid: bool = True,
    translation=(0.0, 0.0),
) -> corrosion.CorrosionPrediction:
    return corrosion.CorrosionPrediction(
        pair_id=pair_id,
        probability=probability,
        valid=valid,
        translation_hat_rc=translation,
        translation_dispersion_px=1.0 if valid else None,
    )


def test_fixed_conditions_are_exactly_clean_erosion_and_local_bites() -> None:
    assert [value.name for value in corrosion.FIXED_CONDITIONS] == [
        "clean",
        "erosion_r2",
        "erosion_r4",
        "erosion_r8",
        "local_bites_k1_r8",
        "local_bites_k2_r8",
        "local_bites_k4_r8",
    ]
    assert [value.erosion_radius_px for value in corrosion.FIXED_CONDITIONS[1:4]] == [
        2,
        4,
        8,
    ]
    assert [value.bite_count for value in corrosion.FIXED_CONDITIONS[4:]] == [1, 2, 4]
    assert all(value.bite_radius_px == 8 for value in corrosion.FIXED_CONDITIONS[4:])


def test_erosion_is_binary_monotone_and_clean_is_a_readonly_copy() -> None:
    mask = _square()
    clean = corrosion.degrade_mask(
        mask,
        fragment_id="fragment-a",
        condition=corrosion.FIXED_CONDITIONS[0],
        seed=260831,
    )
    eroded = corrosion.degrade_mask(
        mask,
        fragment_id="fragment-a",
        condition=corrosion.FIXED_CONDITIONS[2],
        seed=260831,
    )
    assert clean.dtype == np.bool_
    assert not clean.flags.writeable
    assert not eroded.flags.writeable
    assert np.array_equal(clean, mask)
    assert clean is not mask
    assert np.all(~eroded | mask)
    assert 0 < np.count_nonzero(eroded) < np.count_nonzero(mask)


def test_local_bites_are_fragment_condition_deterministic_and_pair_blind() -> None:
    mask = _square()
    condition = corrosion.FIXED_CONDITIONS[-1]
    first = corrosion.degrade_mask(
        mask,
        fragment_id="model/masks_800/f17.png",
        condition=condition,
        seed=260831,
    )
    repeated_in_another_pair = corrosion.degrade_mask(
        mask,
        fragment_id="model/masks_800/f17.png",
        condition=condition,
        seed=260831,
    )
    other_condition = corrosion.degrade_mask(
        mask,
        fragment_id="model/masks_800/f17.png",
        condition=corrosion.FIXED_CONDITIONS[-2],
        seed=260831,
    )
    assert np.array_equal(first, repeated_in_another_pair)
    assert np.count_nonzero(first) < np.count_nonzero(mask)
    assert not np.array_equal(first, other_condition)
    assert np.all(~first | other_condition)


def test_corrupted_fragment_cache_reextracts_contour_from_corrupted_mask(
    tmp_path: Path,
) -> None:
    mask = np.zeros((96, 96), dtype=np.bool_)
    mask[8:88, 10:86] = True
    mask[30:66, 50:86] = False
    path = tmp_path / "mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)
    condition = corrosion.CorrosionCondition(
        "test_erosion_r4", "erosion", 4, erosion_radius_px=4
    )
    cache = corrosion._ConditionFragmentCache(
        condition=condition,
        seed=260831,
        canvas_size=96,
        contour_cap=64,
        capacity=2,
    )
    fragment = cache.get("fragment-a", path)
    expected_mask = corrosion.degrade_mask(
        mask, fragment_id="fragment-a", condition=condition, seed=260831
    )
    expected_points, _ = extract_ordered_outer_contour(
        expected_mask, cap=64, smoothing_sigma=3.0
    )
    clean_points, _ = extract_ordered_outer_contour(mask, cap=64, smoothing_sigma=3.0)
    assert np.array_equal(fragment.mask, expected_mask)
    np.testing.assert_allclose(
        fragment.points_rc[: len(expected_points)], expected_points, rtol=0.0, atol=0.0
    )
    assert not np.array_equal(fragment.points_rc[: len(clean_points)], clean_points)
    assert fragment.geometry_valid
    assert int(fragment.contour_valid.sum()) == len(expected_points)
    assert cache.get("fragment-a", path) is fragment
    assert cache.receipt()["cache_hit_count"] == 1


def test_erosion_split_dumbbell_keeps_mask_but_invalidates_only_n512_geometry(
    tmp_path: Path,
) -> None:
    mask = np.zeros((96, 96), dtype=np.bool_)
    mask[18:46, 8:36] = True
    mask[18:46, 60:88] = True
    mask[31, 36:60] = True
    path = tmp_path / "dumbbell.png"
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)
    cache = corrosion._ConditionFragmentCache(
        condition=corrosion.FIXED_CONDITIONS[1],
        seed=260831,
        canvas_size=96,
        contour_cap=64,
        capacity=2,
    )
    fragment = cache.get("dumbbell", path)
    assert fragment.component_count_8_connected == 2
    assert fragment.geometry_valid is False
    assert fragment.geometry_failure == "component_count_8_connected_2_not_exactly_one"
    assert np.count_nonzero(fragment.mask) > 0
    assert fragment.historical_mask.shape == (1, 64, 64)
    assert fragment.benchmark_geometry_valid is True
    assert int(fragment.benchmark_contour_valid.sum()) >= 4
    audit = cache.fragment_audit()["dumbbell"]
    assert audit["component_count_8_connected"] == 2
    assert audit["n512_geometry_valid"] is False


def test_shared_clean_source_authority_rejects_cross_condition_source_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mask.png"
    first = _square()
    Image.fromarray(first.astype(np.uint8) * 255, mode="L").save(path)
    authority = corrosion._CleanSourceAuthority()
    clean_cache = corrosion._ConditionFragmentCache(
        condition=corrosion.FIXED_CONDITIONS[0],
        seed=260831,
        canvas_size=96,
        contour_cap=64,
        capacity=2,
        clean_source_authority=authority,
    )
    clean_cache.get("fragment-a", path)
    authority_row = authority.rows()[0]
    assert authority_row["fragment_id"] == "fragment-a"
    assert len(authority_row["source_png_sha256"]) == 64
    assert len(authority_row["clean_semantic_mask_sha256"]) == 64
    changed = np.roll(first, 1, axis=0)
    Image.fromarray(changed.astype(np.uint8) * 255, mode="L").save(path)
    erosion_cache = corrosion._ConditionFragmentCache(
        condition=corrosion.FIXED_CONDITIONS[1],
        seed=260831,
        canvas_size=96,
        contour_cap=64,
        capacity=2,
        clean_source_authority=authority,
    )
    with pytest.raises(corrosion.RachelN512CorrosionError, match="changed across"):
        erosion_cache.get("fragment-a", path)


def test_clean_parity_checks_validity_probability_and_translation() -> None:
    observed = {
        method: (
            _prediction(
                "p0",
                0.7,
                translation=(3.0, -2.0) if method == "full_n512" else None,
            ),
        )
        for method in corrosion.METHODS
    }
    expected = {
        method: (
            corrosion._ExpectedCleanPrediction(
                pair_id="p0",
                label=True,
                cluster_id="c0",
                probability=0.7,
                valid=True,
                translation_hat_rc=(3.0, -2.0) if method == "full_n512" else None,
            ),
        )
        for method in corrosion.METHODS
    }
    receipt = corrosion._clean_parity_gate(observed, expected, tolerance=1e-6)
    assert receipt["passed"] is True
    changed = dict(observed)
    changed["coarse_only"] = (_prediction("p0", 0.70001),)
    with pytest.raises(corrosion.RachelN512CorrosionError, match="did not reproduce"):
        corrosion._clean_parity_gate(changed, expected, tolerance=1e-6)

    benchmark_observed = dict(observed)
    benchmark_expected = dict(expected)
    benchmark_observed["pairingnet_adapted"] = (
        corrosion.CorrosionPrediction(
            "p0", float("nan"), False, (3.0, -2.0), translation_valid=True
        ),
    )
    benchmark_expected["pairingnet_adapted"] = (
        corrosion._ExpectedCleanPrediction(
            pair_id="p0",
            label=True,
            cluster_id="c0",
            probability=float("nan"),
            valid=False,
            translation_hat_rc=(3.0, -2.0),
            translation_valid=True,
        ),
    )
    assert corrosion._clean_parity_gate(
        benchmark_observed, benchmark_expected, tolerance=1e-6
    )["passed"] is True
    benchmark_observed["pairingnet_adapted"] = (
        corrosion.CorrosionPrediction(
            "p0", float("nan"), False, (3.01, -2.0), translation_valid=True
        ),
    )
    with pytest.raises(corrosion.RachelN512CorrosionError, match="did not reproduce"):
        corrosion._clean_parity_gate(
            benchmark_observed, benchmark_expected, tolerance=1e-6
        )


def test_summary_uses_one_all_method_all_condition_fixed_common_population() -> None:
    pair_ids = tuple("p{}".format(index) for index in range(6))
    labels = (False, True, False, True, False, True)
    clusters = ("c0", "c0", "c1", "c1", "c2", "c2")
    predictions = {}
    for condition_index, condition in enumerate(corrosion.FIXED_CONDITIONS):
        methods = {}
        for method_index, method in enumerate(corrosion.METHODS):
            rows = []
            for index, pair_id in enumerate(pair_ids):
                probability = (0.15 if not labels[index] else 0.85) - 0.01 * condition_index
                valid = not (
                    condition.name == "erosion_r8"
                    and method == "full_n512"
                    and index == 0
                )
                rows.append(
                    _prediction(
                        pair_id,
                        probability + method_index * 1e-4,
                        valid=valid,
                        translation=(float(condition_index), 0.0),
                    )
                )
            methods[method] = tuple(rows)
        predictions[condition.name] = methods
    summary = corrosion.summarize_corrosion_predictions(
        predictions,
        labels=labels,
        clusters=clusters,
        thresholds={method: 0.5 for method in corrosion.METHODS},
        pair_ids=pair_ids,
    )
    assert summary["primary_population"]["pair_count"] == 5
    assert summary["primary_population"]["coverage"] == pytest.approx(5 / 6)
    for condition in corrosion.FIXED_CONDITIONS:
        for method in corrosion.METHODS:
            view = summary["conditions"][condition.name]["methods"][method][
                "fixed_all_method_all_condition_common_valid"
            ]
            assert view["coverage"]["valid_count"] == 5
            assert view["primary_threshold_free_ranking"] is not None
            assert (
                view["secondary_frozen_validation_threshold"][
                    "fit_performed_during_robustness_evaluation"
                ]
                is False
            )
    stability = summary["conditions"]["erosion_r4"][
        "full_n512_translation_stability"
    ]
    assert stability["selected_pair_count"] == 3
    assert stability["median_l2_change_px"] == pytest.approx(2.0)
    assert stability["positive_pairs_only_negative_pairs_excluded"] is True
    assert summary["geometry_interpretation"]["correspondence_accuracy_reported"] is False


def test_parse_clean_reference_ignores_additive_source_unit_ids(tmp_path: Path) -> None:
    path = tmp_path / "pair_scores.jsonl"
    row = {
        "schema_version": "rachel-n512-sealed-test-pair/1.0",
        "arm": "coarse_only",
        "pair_id": "p0",
        "label": True,
        "cluster_id": "c0",
        "source_unit_ids": ["u0"],
        "scores": {"coarse": {"probability": 0.75, "valid": True}},
        "main_score": "coarse",
        "decision": {
            "validation_threshold": 0.5,
            "valid": True,
            "predicted_label": True,
        },
    }
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(payload)
    parsed = corrosion._parse_expected_pair_scores(
        path,
        method="coarse_only",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_count=1,
    )
    assert parsed[0].pair_id == "p0"
    assert parsed[0].probability == 0.75


def test_module_has_no_legacy_or_real_evaluator_dependency() -> None:
    source = Path(corrosion.__file__).read_text(encoding="utf-8")
    assert "staging.pairwise_v0_1" not in source
    assert "rachel_n512_real_external" not in source
    assert "real_dunhuang" not in source


def test_run_freezes_and_gates_every_winner_before_any_test_open() -> None:
    source = inspect.getsource(corrosion.run_rachel_n512_corrosion_robustness)
    freeze_n512 = source.index("sealed._freeze_completed_winners")
    gate_n512 = source.index("sealed._require_formal_convergence")
    freeze_matched = source.index("freeze_matched_mm_winners")
    gate_matched = source.index("sealed._require_matched_training_alignment")
    hash_gate_matched = source.index("sealed._matched_training_hash_evidence")
    freeze_benchmarks = source.index("benchmark_adapter.freeze_same_data_benchmarks")
    gate_benchmark_manifests = source.index(
        "sealed._benchmark_training_manifest_evidence"
    )
    resolve_sealed = source.index("config.sealed_test_directory.resolve")
    reject_output = source.index("_reject_output_inside_sources")
    open_clean_reference = source.index("_load_sealed_reference_after_freeze")
    open_test_manifest = source.index("sealed._test_manifest_rows")
    assert freeze_n512 < gate_n512 < freeze_matched < gate_matched
    assert gate_matched < hash_gate_matched < freeze_benchmarks
    assert freeze_benchmarks < gate_benchmark_manifests < resolve_sealed
    assert resolve_sealed < reject_output < open_clean_reference
    assert open_clean_reference < open_test_manifest
    score_predictions = source.index("_score_condition")
    open_targets = source.index("_load_positive_translation_targets_after_predictions")
    assert score_predictions < open_targets
    score_parameters = inspect.signature(corrosion._score_condition).parameters
    assert not {"labels", "clusters", "targets"} & set(score_parameters)


def test_formal_default_requires_exact_six_authorities_and_legacy_is_explicit(
    tmp_path: Path,
) -> None:
    base = {
        "run_directory": tmp_path,
        "matched_mm_run_directory": tmp_path,
        "sealed_test_directory": tmp_path,
        "output_root": tmp_path,
    }
    with pytest.raises(ValueError, match="requires PairingNet and ShreddingNet"):
        corrosion.RachelN512CorrosionConfig(**base)
    legacy = corrosion.RachelN512CorrosionConfig(
        **base, compatibility_mode=True
    )
    assert legacy.compatibility_mode is True
    with pytest.raises(ValueError, match="forbids benchmark authorities"):
        corrosion.RachelN512CorrosionConfig(
            **base,
            pairingnet_run_directory=tmp_path,
            shreddingnet_freeze_path=tmp_path / "train_val_freeze.json",
            compatibility_mode=True,
        )


def test_exact_six_score_uses_one_common_mask_and_one_forward_per_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    masks = []
    for index in range(3):
        mask = np.zeros((800, 800), dtype=np.bool_)
        mask[100 + index : 700 - index, 120:680] = True
        path = tmp_path / "mask-{}.png".format(index)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)
        masks.append(path)
    pairs = (
        corrosion.BlindSyntheticPair("p0", "f0", "f1", masks[0], masks[1]),
        corrosion.BlindSyntheticPair("p1", "f1", "f2", masks[1], masks[2]),
    )

    class _Model:
        def __init__(self):
            self.calls = 0
            self.seen = []

        def to(self, _device):
            return self

        def eval(self):
            return self

    class _Coarse(_Model):
        def __call__(self, first, second):
            self.calls += 1
            self.seen.append((first.detach().cpu().numpy(), second.detach().cpu().numpy()))
            count = len(first)
            return SimpleNamespace(
                probability=torch.full((count,), 0.6),
                valid_problem=torch.ones(count, dtype=torch.bool),
            )

    class _Full(_Model):
        def __call__(self, first, second, *_contours):
            self.calls += 1
            self.seen.append((first.detach().cpu().numpy(), second.detach().cpu().numpy()))
            count = len(first)
            return SimpleNamespace(
                fused_probability=torch.full((count,), 0.7),
                decision_valid=torch.ones(count, dtype=torch.bool),
                translation_hat_rc=torch.zeros((count, 2)),
                translation_dispersion_px=torch.ones(count),
            )

    class _Matched(_Model):
        def __call__(self, first, second):
            self.calls += 1
            self.seen.append((first.detach().cpu().numpy(), second.detach().cpu().numpy()))
            return torch.full((len(first), 1), 0.55)

    class _Benchmark:
        def __init__(self, method):
            self.method = method
            self.calls = 0
            self.seen = []

        def to(self, _device):
            return self

        def eval(self):
            return self

        def predict_batch(self, batch, *, return_correspondence):
            assert return_correspondence is False
            self.calls += 1
            self.seen.append((batch.mask_a.copy(), batch.mask_b.copy()))
            count = len(batch.pair_ids)
            return SimpleNamespace(
                method_key=self.method,
                pair_ids=tuple(batch.pair_ids),
                pair_probability=np.full(count, 0.65, dtype=np.float32),
                decision_valid=np.ones(count, dtype=np.bool_),
                translation_hat_rc=np.zeros((count, 2), dtype=np.float32),
                translation_valid=np.ones(count, dtype=np.bool_),
            )

    coarse = _Coarse()
    full = _Full()
    monkeypatch.setattr(corrosion, "RachelN512Pairwise", _Full)
    model_config = SimpleNamespace(coarse_size=128)
    n512 = {
        "coarse_only": SimpleNamespace(model=coarse, model_config=model_config),
        "full_n512": SimpleNamespace(model=full, model_config=model_config),
    }
    matched_models = {method: _Matched() for method in corrosion.MATCHED_METHODS}
    matched = {
        method: SimpleNamespace(model=matched_models[method])
        for method in corrosion.MATCHED_METHODS
    }
    benchmark_models = {
        method: _Benchmark(method) for method in corrosion.BENCHMARK_METHODS
    }
    benchmarks = {
        method: SimpleNamespace(
            model=benchmark_models[method],
            predict_batch=benchmark_models[method].predict_batch,
        )
        for method in corrosion.BENCHMARK_METHODS
    }
    cache = corrosion._ConditionFragmentCache(
        condition=corrosion.FIXED_CONDITIONS[0],
        seed=19,
        canvas_size=800,
        contour_cap=512,
        capacity=8,
    )
    predictions, audit = corrosion._score_condition(
        pairs,
        fragment_cache=cache,
        n512_winners=n512,
        matched_winners=matched,
        benchmark_winners=benchmarks,
        methods=corrosion.METHODS,
        device=torch.device("cpu"),
        precision="fp32",
        batch_size=2,
    )
    assert tuple(predictions) == corrosion.METHODS
    assert audit["pair_forward_count_by_method"] == {
        method: 2 for method in corrosion.METHODS
    }
    assert audit["each_method_forwarded_each_pair_exactly_once"] is True
    assert coarse.calls == full.calls == 1
    assert all(model.calls == 1 for model in matched_models.values())
    assert all(model.calls == 1 for model in benchmark_models.values())
    np.testing.assert_array_equal(full.seen[0][0], benchmark_models[corrosion.BENCHMARK_METHODS[0]].seen[0][0])
    np.testing.assert_array_equal(
        benchmark_models[corrosion.BENCHMARK_METHODS[0]].seen[0][0],
        benchmark_models[corrosion.BENCHMARK_METHODS[1]].seen[0][0],
    )


def test_positive_translation_metrics_exclude_negatives_and_use_frozen_threshold() -> None:
    predictions = (
        _prediction("positive-0", 0.9, translation=(1.0, 0.0)),
        _prediction("negative-0", 0.99, translation=(999.0, 999.0)),
        _prediction("positive-1", 0.4, translation=(6.0, 0.0)),
        _prediction("negative-1", 0.01, translation=(-999.0, -999.0)),
    )
    result = corrosion._positive_translation_metric_view(
        predictions,
        labels=(True, False, True, False),
        targets=((0.0, 0.0), None, (0.0, 0.0), None),
        selected=(True, True, True, True),
        threshold=0.5,
    )
    assert result["eligible_positive_count"] == 2
    assert result["median_l2_px"] == pytest.approx(3.5)
    assert result["p90_l2_px"] == pytest.approx(5.5)
    assert result["recall_at_2px"] == pytest.approx(0.5)
    assert result["recall_at_5px"] == pytest.approx(0.5)
    assert result["recall_at_8px"] == pytest.approx(1.0)
    assert result["joint_frozen_threshold_and_translation_recall_at_8px"] == pytest.approx(
        0.5
    )


def test_direct_geometry_reports_te_assembly_pairing_registration_and_na_corr() -> None:
    predictions = (
        corrosion.CorrosionPrediction(
            "p0", 0.9, True, (3.0, 4.0), translation_valid=True
        ),
        corrosion.CorrosionPrediction(
            "p1", 0.9, True, None, translation_valid=False
        ),
        corrosion.CorrosionPrediction("n0", 0.8, True),
        corrosion.CorrosionPrediction("n1", 0.1, True),
    )
    result = corrosion._direct_geometry_metric_view(
        predictions,
        labels=(True, True, False, False),
        targets=((0.0, 0.0), (6.0, 8.0), None, None),
        selected=(True, True, True, True),
        threshold=0.5,
        contour_area_sums=(100.0, 200.0, 100.0, 100.0),
    )
    assert result["translation"]["te_px_conditioned_on_valid_pose"] == {
        "count": 1,
        "median": 5.0,
        "p90": 5.0,
    }
    assert result["translation"]["unconditional_positive_recall"]["at_5px"] == 0.5
    assembly = result["assembly_edge"]["by_tolerance"]["at_5px"]
    assert assembly["true_positive_count"] == 1
    assert assembly["predicted_count"] == 3
    assert assembly["target_count"] == 2
    registration = result["pairingnet_style_registration"]
    assert registration["identity_fallback_count"] == 1
    assert registration["mean_symmetric_hausdorff_px"] == pytest.approx(7.5)
    assert registration["mean_normalized_translation_error"] == pytest.approx(0.05)
    assert result["correspondence"]["status"] == "not_applicable"


def test_parse_exact_six_benchmark_clean_reference_translation_validity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pairingnet.jsonl"
    row = {
        "schema_version": "rachel-n512-sealed-test-pair/1.0",
        "arm": "pairingnet_adapted",
        "method_id": "fixture",
        "pair_id": "p0",
        "label": True,
        "cluster_id": "c0",
        "source_unit_ids": ["u0"],
        "scores": {
            "pair_probability": {"probability": 0.75, "valid": True}
        },
        "main_score": "pair_probability",
        "decision": {
            "validation_threshold": 0.5,
            "valid": True,
            "predicted_label": True,
        },
        "geometry": {
            "translation_prediction_valid": True,
            "translation_hat_rc": [3.0, -4.0],
        },
    }
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(payload)
    parsed = corrosion._parse_expected_pair_scores(
        path,
        method="pairingnet_adapted",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_count=1,
    )
    assert parsed[0].translation_valid is True
    assert parsed[0].translation_hat_rc == (3.0, -4.0)


def test_post_prediction_target_loader_reads_positive_translation_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "release"
    target = root / "targets" / "pairs" / "fixture" / "p0.npz"
    target.parent.mkdir(parents=True)
    np.savez_compressed(
        target,
        correspondence_indices=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        translation_a_to_b_rc=np.asarray([3.0, -4.0], dtype=np.float32),
        translation_a_to_b_xy_cartesian=np.asarray([-4.0, -3.0], dtype=np.float32),
    )
    manifest = root / "pairs" / "test.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = (
        {
            "pair_id": "p0",
            "split": "test",
            "label": True,
            "correspondence_path": "targets/pairs/fixture/p0.npz",
        },
        {
            "pair_id": "n0",
            "split": "test",
            "label": False,
            "correspondence_path": None,
        },
    )
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    translations, receipt = (
        corrosion._load_positive_translation_targets_after_predictions(
            root,
            manifest,
            pair_ids=("p0", "n0"),
            labels=(True, False),
            expected_manifest_sha256=manifest_sha,
        )
    )
    assert translations == ((3.0, -4.0), None)
    assert receipt["raw_positive_target_archive_count"] == 1
    assert receipt["negative_target_archive_count"] == 0
    assert receipt["used_for_model_forward_checkpoint_selection_or_threshold_fit"] is False


def test_pair_rows_emit_source_units_and_endpoint_component_audit() -> None:
    manifest = SimpleNamespace(
        pair_id="p0",
        label=True,
        cluster_id="unit:u0",
        source_unit_ids=("u0",),
    )
    pair_input = corrosion.BlindSyntheticPair(
        "p0",
        "fa",
        "fb",
        Path("a.png"),
        Path("b.png"),
    )
    predictions = {
        method: (_prediction("p0", 0.75, translation=(0.0, 0.0)),)
        for method in corrosion.METHODS
    }
    audit = {
        fragment_id: {
            "component_count_8_connected": count,
            "n512_geometry_valid": count == 1,
            "n512_geometry_failure": None if count == 1 else "split",
        }
        for fragment_id, count in (("fa", 1), ("fb", 2))
    }
    row = corrosion._condition_pair_rows(
        corrosion.FIXED_CONDITIONS[0],
        predictions,
        manifest_rows=(manifest,),
        pair_inputs=(pair_input,),
        fragment_audit=audit,
        fixed_common=(False,),
        thresholds={method: 0.5 for method in corrosion.METHODS},
    )[0]
    assert corrosion.SCHEMA_VERSION == "rachel-n512-corrosion-robustness/2.0"
    assert row["schema_version"] == "rachel-n512-corrosion-pair/2.0"
    assert list(row["methods"]) == list(corrosion.METHODS)
    assert row["source_unit_ids"] == ["u0"]
    assert row["fragment_morphology"]["a"]["component_count_8_connected"] == 1
    assert row["fragment_morphology"]["b"]["component_count_8_connected"] == 2


def test_disclosure_contains_exact_prior_morphology_event_and_target_timing() -> None:
    source = Path(corrosion.__file__).read_text(encoding="utf-8")
    required = {
        '"prior_convergence_time_synthetic_test_mask_morphology_review": True',
        '"prior_convergence_time_synthetic_test_mask_sample_count": 500',
        '"prior_convergence_time_synthetic_test_labels_read": False',
        '"prior_convergence_time_synthetic_test_model_scores_read": False',
        '"prior_convergence_time_synthetic_test_activity_used_for_training_checkpoint_or_threshold_selection": False',
        '"prior_convergence_time_synthetic_test_activity_influenced_fixed_corrosion_conditions": False',
        '"prior_convergence_time_synthetic_test_activity_used_for_corrosion_runtime_and_representation_QA": True',
        '"real_data_accessed_in_that_activity": False',
        '"raw_target_archives_opened_only_after_all_predictions": True',
        '"sealed_gt_derived_fields_used_for_model_or_selection": False',
    }
    assert all(value in source for value in required)
    truthful_e5 = {
        '"prior_epoch5_synthetic_test_human_visible_before_continuation": True',
        '"prior_epoch5_synthetic_test_used_by_automated_checkpoint_selection": False',
        '"prior_epoch5_synthetic_test_used_by_automated_threshold_fitting": False',
        '"prior_epoch5_synthetic_test_used_by_automated_early_stopping": False',
        '"prior_epoch5_synthetic_test_used_by_automated_scheduler": False',
        '"claim_no_human_cognitive_influence": False',
    }
    assert all(value in source for value in truthful_e5)
    assert "prior_epoch5_synthetic_test_used_for_continuation_or_selection" not in source
    assert "prior_epoch5_real_activity_used_for_training_or_selection" not in source


def test_endpoint_pigeonhole_bootstrap_is_shared_deterministic_and_paired() -> None:
    pair_ids = tuple("p{}".format(index) for index in range(6))
    labels = (True, False, True, False, True, False)
    source_units = (
        ("u0",),
        ("u0", "u1"),
        ("u1",),
        ("u1", "u2"),
        ("u2",),
        ("u0", "u2"),
    )
    targets = (
        (0.0, 0.0),
        None,
        (0.0, 0.0),
        None,
        (0.0, 0.0),
        None,
    )
    predictions = {}
    for condition_index, condition in enumerate(corrosion.FIXED_CONDITIONS):
        methods = {}
        for method_index, method in enumerate(corrosion.METHODS):
            rows = []
            for index, (pair_id, label) in enumerate(zip(pair_ids, labels)):
                base = 0.82 if label else 0.18
                probability = base - condition_index * 0.015 + method_index * 0.005
                rows.append(
                    _prediction(
                        pair_id,
                        probability,
                        translation=(float(condition_index + index), 0.0),
                    )
                )
            methods[method] = tuple(rows)
        predictions[condition.name] = methods
    arguments = {
        "labels": labels,
        "source_unit_ids": source_units,
        "translation_targets": targets,
        "full_threshold": 0.5,
        "fixed_common": (True,) * 6,
        "replicates": 64,
        "seed": 91,
    }
    first = corrosion.endpoint_pigeonhole_corrosion_bootstrap(
        predictions, **arguments
    )
    second = corrosion.endpoint_pigeonhole_corrosion_bootstrap(
        predictions, **arguments
    )
    assert first == second
    assert first["sampling_dependency_unit_count"] == 3
    assert first[
        "shared_draws_across_all_methods_conditions_and_geometry_metrics"
    ] is True
    clean_delta = first["ranking"]["condition_minus_clean_same_method"]["clean"][
        "full_n512"
    ]["auroc"]
    assert clean_delta["point_estimate"] == pytest.approx(0.0)
    assert clean_delta["percentile_95_ci"] == pytest.approx([0.0, 0.0])
    translation = first["positive_only_translation_gt_and_joint"]
    assert translation["negative_pairs_excluded"] is True
    assert translation["by_condition"]["erosion_r8"]["median_l2_px"][
        "valid_replicates"
    ] == first["valid_replicates"]
    direct = corrosion._positive_translation_metric_view(
        predictions["erosion_r8"]["full_n512"],
        labels=labels,
        targets=targets,
        selected=(True,) * 6,
        threshold=0.5,
    )
    assert translation["by_condition"]["erosion_r8"]["median_l2_px"][
        "point_estimate"
    ] == direct["median_l2_px"]
    assert translation["by_condition"]["erosion_r8"]["p90_l2_px"][
        "point_estimate"
    ] == direct["p90_l2_px"]


def test_weighted_quantile_matches_numpy_linear_expansion() -> None:
    values = np.asarray([1.0, 6.0], dtype=np.float64)
    unit = np.ones(2, dtype=np.float64)
    assert corrosion._weighted_quantile(values, unit, 0.5) == pytest.approx(3.5)
    assert corrosion._weighted_quantile(values, unit, 0.9) == pytest.approx(5.5)

    rng = np.random.default_rng(260901)
    for _ in range(64):
        values = rng.normal(size=9)
        weights = rng.integers(0, 6, size=9, dtype=np.int64)
        if not weights.any():
            weights[0] = 1
        expanded = np.repeat(values, weights)
        for quantile in (0.0, 0.1, 0.5, 0.73, 0.9, 1.0):
            observed = corrosion._weighted_quantile(
                values, weights.astype(np.float64), quantile
            )
            expected = float(np.quantile(expanded, quantile))
            assert observed == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_corrosion_file_publication_is_no_replace(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    corrosion._atomic_json(target, {"value": 1})
    with pytest.raises(corrosion.RachelN512CorrosionError, match="overwrite"):
        corrosion._atomic_json(target, {"value": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}


def test_config_rejects_non_strict_clean_tolerance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="clean_tolerance"):
        corrosion.RachelN512CorrosionConfig(
            run_directory=tmp_path,
            matched_mm_run_directory=tmp_path,
            sealed_test_directory=tmp_path,
            output_root=tmp_path,
            clean_tolerance=1e-3,
        )
