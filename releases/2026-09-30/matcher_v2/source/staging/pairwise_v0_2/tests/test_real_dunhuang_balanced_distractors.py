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
from staging.pairwise_v0_2.baselines import (
    real_dunhuang_balanced_distractors as balanced,
)
from staging.pairwise_v0_2.baselines.historical_mm_siamese import (
    HISTORICAL_MM_BASELINE_ID,
)
from staging.pairwise_v0_2.baselines.matched_route_a_siamese import (
    MATCHED_ROUTE_A_SIAMESE_ID,
)
from staging.pairwise_v0_2.baselines.real_dunhuang_evaluation import (
    RealBatchPrediction,
    load_strict_real_pair_dataset,
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
    return hashlib.sha256(mask.tobytes(order="C")).hexdigest()


def _write_alpha(path: Path, mask: np.ndarray) -> None:
    value = np.zeros((*mask.shape, 4), dtype=np.uint8)
    value[..., :3] = np.asarray((80, 60, 40), dtype=np.uint8)
    value[..., 3] = mask.astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="RGBA").save(path)


def _rectangle(shape, top, left, height, width):
    value = np.zeros(shape, dtype=np.bool_)
    value[top : top + height, left : left + width] = True
    return value


def _four_case_documents(tmp_path: Path):
    root = tmp_path / "main"
    manifest_sha = "c" * 64
    # The first small alpha occurs in two different cases.  It is legal as an
    # occurrence, but the constructed matcher must never pair those exact masks.
    small = _rectangle((64, 64), 8, 10, 32, 20)
    medium = _rectangle((72, 72), 7, 8, 42, 30)
    large = _rectangle((88, 88), 6, 10, 62, 42)
    wide = _rectangle((80, 96), 9, 7, 40, 70)
    masks_by_case = (
        (small, large),
        (small.copy(), wide),
        (medium, large.copy()),
        (medium.copy(), wide.copy()),
    )
    cases = []
    local_cases = {}
    for case_index, masks in enumerate(masks_by_case):
        case_uid = "dhcase-balanced-fixture-{}".format(case_index)
        group = root / "Ground Truth Simple" / str(case_index)
        fragments = []
        paths = []
        for fragment_id, mask in enumerate(masks, start=1):
            path = group / "{}.png".format(fragment_id)
            _write_alpha(path, mask)
            paths.append(str(path))
            fragments.append(
                {
                    "fragment_id": fragment_id,
                    "has_alpha": True,
                    "alpha_mask_sha256": _alpha_sha(mask),
                    "bbox_xyxy": [
                        100 * fragment_id,
                        0,
                        100 * fragment_id + mask.shape[1],
                        mask.shape[0],
                    ],
                }
            )
        occurrence_uid = "occ-" + case_uid
        cases.append(
            {
                "case_uid": case_uid,
                "canonical_collection": "main",
                "canonical_category": "ground_truth_simple",
                "observed_categories": ["ground_truth_simple"],
                "disposition": "eligible",
                "fragments": fragments,
                "pair_labels": [
                    {"fragment_a": 1, "fragment_b": 2, "label": "positive"}
                ],
                "occurrences": [
                    {
                        "occurrence_uid": occurrence_uid,
                        "collection": "main",
                        "category": "ground_truth_simple",
                    }
                ],
            }
        )
        local_cases[case_uid] = {
            "occurrences": [
                {
                    "occurrence_uid": occurrence_uid,
                    "collection": "main",
                    "category": "ground_truth_simple",
                    "fragment_paths": paths,
                }
            ]
        }
    manifest = {
        "schema_version": "pairwise-v0.2-real-external-test/0.1",
        "manifest_sha256": manifest_sha,
        "cases": cases,
    }
    receipt = {
        "schema_version": "pairwise-v0.2-real-external-test-local-receipt/0.1",
        "portable_manifest_sha256": manifest_sha,
        "dataset_roots": {"main": str(root), "supp": str(tmp_path / "supp")},
        "cases": local_cases,
    }
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "receipt.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return manifest_path, receipt_path


def _strict_fixture(tmp_path: Path):
    manifest, receipt = _four_case_documents(tmp_path)
    return load_strict_real_pair_dataset(
        manifest,
        receipt,
        target_long_side=96,
        expected_case_count=4,
        expected_pair_count=4,
        expected_positive_count=4,
        expected_negative_count=0,
    )


def _construction_config() -> balanced.BalancedDistractorConfig:
    return balanced.BalancedDistractorConfig(
        seed="balanced-fixture-fixed-seed",
        strict_positive_count=4,
        strict_negative_count=0,
        unique_fragment_occurrence_count=8,
        constructed_distractor_count=4,
        max_fragments_per_case=2,
        # This deliberately tiny fixture repeats four very different masks and
        # exercises identity/case constraints rather than the production scale
        # distribution.  The formal 938-occurrence population uses the frozen
        # default 2x scale bounds.
        max_alpha_dimension_ratio=8.0,
        max_alpha_foreground_area_ratio=8.0,
    )


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


def test_constructed_plan_is_deterministic_cross_case_and_non_gt(tmp_path):
    strict = _strict_fixture(tmp_path)
    config = _construction_config()
    try:
        first = balanced.build_balanced_real_pair_dataset(strict, config=config)
        second = balanced.build_balanced_real_pair_dataset(strict, config=config)
        assert len(first.dataset.records) == 8
        assert first.dataset.positive_count == 4
        assert first.dataset.negative_count == 4
        assert first.dataset.records[:4] == strict.records
        assert [record.pair_id for record in first.constructed_records] == [
            record.pair_id for record in second.constructed_records
        ]
        assert first.receipt["content_sha256"] == second.receipt["content_sha256"]
        assert first.receipt["construction"]["constructed_selection_sha256"] == (
            second.receipt["construction"]["constructed_selection_sha256"]
        )

        fragment_use = {}
        case_pairs = set()
        for record in first.constructed_records:
            assert record.label is False
            assert record.direction_b_wrt_a is None
            assert record.label_origin == balanced.BALANCED_DISTRACTOR_LABEL_ORIGIN
            assert record.provenance["constructed_distractor"] is True
            assert record.provenance["constructed_is_ground_truth_negative"] is False
            case_a = record.provenance["source_case_uid_a"]
            case_b = record.provenance["source_case_uid_b"]
            assert case_a != case_b
            case_pair = tuple(sorted((case_a, case_b)))
            assert case_pair not in case_pairs
            case_pairs.add(case_pair)
            assert record.fragment_a.content_sha256 != record.fragment_b.content_sha256
            for fragment_id in record.canonical_pair_key:
                fragment_use[fragment_id] = fragment_use.get(fragment_id, 0) + 1
            # Rebinding pair provenance must not change loader-visible alpha bytes.
            rebound = first.dataset.mask_loader(record.fragment_a)
            original = next(
                reference
                for strict_record in strict.records
                for reference in (strict_record.fragment_a, strict_record.fragment_b)
                if reference.fragment_id == record.fragment_a.fragment_id
            )
            assert np.array_equal(rebound, first.dataset.mask_loader(original))
        assert set(fragment_use.values()) == {1}
        assert len(fragment_use) == 8
        assert first.receipt["negative_semantics"] == {
            "manifest_negative_count": 0,
            "manifest_negatives_are_ground_truth_labelled": True,
            "constructed_distractor_count": 4,
            "constructed_distractors_are_ground_truth_negatives": False,
            "required_label": (
                "constructed cross-case scale-matched distractor; not GT-negative"
            ),
            "scientific_use": (
                "external evaluation only; never training, threshold fitting, or tuning"
            ),
        }
    finally:
        strict.mask_loader.close()


def test_default_contract_is_exactly_balanced_and_scale_bounded():
    config = balanced.BalancedDistractorConfig()
    assert config.strict_positive_count == 508
    assert config.strict_negative_count == 39
    assert config.unique_fragment_occurrence_count == 938
    assert config.constructed_distractor_count == 469
    assert config.balanced_positive_count == 508
    assert config.balanced_negative_count == 508
    assert config.max_alpha_dimension_ratio == pytest.approx(2.0)
    assert config.max_alpha_foreground_area_ratio == pytest.approx(2.0)


class _OrderedControl(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.register_buffer("probability", torch.tensor(probability, dtype=torch.float32))
        self.offset = 0
        self.observed = []

    def forward(self, input_a, input_b):
        self.observed.append((input_a.detach().cpu().clone(), input_b.detach().cpu().clone()))
        count = input_a.shape[0]
        start = self.offset
        self.offset += count
        return self.probability[start : start + count, None]


def test_six_methods_share_balanced_pair_order_and_report_origin_strata(tmp_path):
    strict = _strict_fixture(tmp_path)
    config = _construction_config()
    geometry = _geometry_config()
    cache = GeometryArtifactCache(tmp_path / "geometry-cache")
    backend = LocalQ1Backend(device="cpu", mode=LocalQ1BackendMode.FORMAL)

    def winner_loader(name):
        return comparison.LoadedArmWinner(
            arm=comparison._arm_from_backend(backend, AblationArmName(name)),
            session=object(),
            runner_mm_report={},
        )

    prepared_by_representation = {}

    def perfect_scorer(loaded, prepared):
        representation = prepared.payload.candidate_representation
        identity = (
            prepared.record_sequence_sha256,
            prepared.prepared_input_sha256,
            prepared.local_candidate_sha256,
        )
        previous = prepared_by_representation.setdefault(representation, identity)
        assert identity == previous
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

    historical = _OrderedControl((0.9,) * 4 + (0.1,) * 4)
    matched = _OrderedControl((0.8,) * 4 + (0.2,) * 4)
    try:
        result = balanced.evaluate_balanced_distractor_six_methods(
            strict_dataset=strict,
            geometry_config=geometry,
            geometry_cache=cache,
            arm_winner_loader=winner_loader,
            arm_scorer=perfect_scorer,
            historical_model=historical,
            matched_model=matched,
            historical_batch_size=4,
            matched_batch_size=4,
            construction_config=config,
            batch_size=8,
        )
    finally:
        strict.mask_loader.close()

    assert result["strict_547_result_replaced_or_modified"] is False
    assert result["pair_count"] == 8
    assert result["positive_count"] == 4
    assert result["negative_count"] == 4
    assert result["pair_ids"][:4] == [record.pair_id for record in strict.records]
    assert set(result["methods"]) == {
        AblationArmName.LOCAL_DUAL_SOFTMAX.value,
        AblationArmName.LOCAL_DUSTBIN_SINKHORN.value,
        AblationArmName.KEYPOINT_DUAL_SOFTMAX.value,
        AblationArmName.KEYPOINT_DUSTBIN_SINKHORN.value,
        HISTORICAL_MM_BASELINE_ID,
        MATCHED_ROUTE_A_SIAMESE_ID,
    }
    assert result["fairness"] == {
        "same_ordered_pair_ids_all_six_methods": True,
        "same_labels_all_six_methods": True,
        "same_source_alpha_masks_all_six_methods": True,
        "same_prepared_tensors_dual_vs_sinkhorn_within_representation": True,
        "architecture_specific_tensorization": {
            "local_methods": (
                "same cached contour-patch tensor within each representation"
            ),
            "siamese_controls": (
                "same bool alpha masks and same deterministic 64x64 transform"
            ),
        },
    }
    for row in result["methods"].values():
        assert row["common_valid_metrics"]["row"]["auroc"] == pytest.approx(1.0)
        assert row["common_valid_metrics"]["row"]["auprc"] == pytest.approx(1.0)
        strata = row["origin_strata_common_six_method"]
        assert strata["strict_manifest_positive"]["count"] == 4
        assert strata["strict_manifest_negative"]["count"] == 0
        assert strata["constructed_cross_case_distractor_not_gt_negative"][
            "count"
        ] == 4
        assert strata["constructed_cross_case_distractor_not_gt_negative"][
            "false_positive_rate_at_0_5"
        ] == pytest.approx(0.0)
    assert len(historical.observed) == len(matched.observed) == 2
    for historical_batch, matched_batch in zip(historical.observed, matched.observed):
        assert torch.equal(historical_batch[0], matched_batch[0])
        assert torch.equal(historical_batch[1], matched_batch[1])


def test_balanced_cli_requires_both_siamese_controls():
    parser = balanced._parser()
    required = [
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
        "--historical-checkpoint",
        "historical.pt",
        "--matched-checkpoint",
        "matched.pt",
        "--output",
        "balanced.json",
    ]
    parsed = parser.parse_args(required)
    assert parsed.historical_checkpoint == Path("historical.pt")
    assert parsed.matched_checkpoint == Path("matched.pt")
    with pytest.raises(SystemExit):
        parser.parse_args(required[:-4] + required[-2:])
