from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from staging.pairwise_v0_2.baselines import mm_validation_comparison as comparison
from staging.pairwise_v0_2.baselines.coarse_local_fusion_evaluation import (
    CALIBRATION_PHASE,
    LOCAL_METHOD_IDS,
    REAL_PHASE,
    REPORT_PHASE,
    PairScoreTable,
    apply_coarse_local_fusions,
    build_fusion_evaluation,
    fit_coarse_local_fusions,
    fused_method_id,
    real_score_table,
    score_route_a_phase,
)
from staging.pairwise_v0_2.baselines.matched_route_a_siamese import (
    MATCHED_ROUTE_A_SIAMESE_ID,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MM_CANONICAL_BINDING,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.local_q1_backend import (
    LocalQ1Backend,
    LocalQ1BackendMode,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArmName,
    PredictionBatch,
    PreparedAblationBatch,
    record_sequence_fingerprint,
)


def _table(phase, prefix, labels, *, invalidate=None):
    labels = np.asarray(labels, dtype=np.bool_)
    count = len(labels)
    coarse = np.where(labels, 0.88, 0.12).astype(np.float64)
    probability = {MATCHED_ROUTE_A_SIAMESE_ID: coarse}
    valid = {MATCHED_ROUTE_A_SIAMESE_ID: np.ones(count, dtype=np.bool_)}
    for index, method in enumerate(LOCAL_METHOD_IDS):
        if index == 0:
            scores = np.full(count, 0.5, dtype=np.float64)
        else:
            positive = 0.70 + index * 0.04
            scores = np.where(labels, positive, 1.0 - positive).astype(np.float64)
        probability[method] = scores
        usable = np.ones(count, dtype=np.bool_)
        if invalidate is not None and method == invalidate[0]:
            usable[invalidate[1]] = False
        valid[method] = usable
    return PairScoreTable(
        phase=phase,
        pair_ids=tuple("{}-pair-{}".format(prefix, index) for index in range(count)),
        labels=labels,
        cluster_ids=tuple("{}-cluster-{}".format(prefix, index // 2) for index in range(count)),
        dataset_ids=tuple("mm" if index < count // 2 else "eccv" for index in range(count)),
        probability=probability,
        valid=valid,
    )


def test_fusion_fit_uses_calibration_only_and_application_ignores_later_labels():
    labels = (False, True, False, True, False, True, False, True)
    calibration = _table(CALIBRATION_PHASE, "cal", labels)
    models = fit_coarse_local_fusions(calibration)
    portable_before = {
        method: model.portable_dict() for method, model in models.items()
    }

    report_a = _table(REPORT_PHASE, "report", labels)
    report_b = PairScoreTable(
        phase=REPORT_PHASE,
        pair_ids=report_a.pair_ids,
        labels=np.asarray(tuple(not value for value in labels), dtype=np.bool_),
        cluster_ids=report_a.cluster_ids,
        dataset_ids=report_a.dataset_ids,
        probability=report_a.probability,
        valid=report_a.valid,
    )
    probability_a, valid_a = apply_coarse_local_fusions(report_a, models)
    probability_b, valid_b = apply_coarse_local_fusions(report_b, models)

    assert set(models) == set(LOCAL_METHOD_IDS)
    assert models[LOCAL_METHOD_IDS[0]].coarse_weight == pytest.approx(1.0)
    assert portable_before == {
        method: model.portable_dict() for method, model in models.items()
    }
    for method in probability_a:
        assert np.array_equal(probability_a[method], probability_b[method])
        assert np.array_equal(valid_a[method], valid_b[method])


def test_fixed_models_report_native_common_and_real_metrics_without_refit():
    labels = (False, True, False, True, False, True, False, True)
    calibration = _table(CALIBRATION_PHASE, "cal", labels)
    models = fit_coarse_local_fusions(calibration)
    report = _table(
        REPORT_PHASE,
        "report",
        labels,
        invalidate=(LOCAL_METHOD_IDS[-1], 0),
    )
    real = _table(
        REAL_PHASE,
        "real",
        labels,
        invalidate=(LOCAL_METHOD_IDS[-1], 0),
    )

    result = build_fusion_evaluation(
        calibration=calibration,
        models=models,
        report=report,
        real=real,
    )

    assert result["no_leakage"] == {
        "parameter_fit_reads_only": CALIBRATION_PHASE,
        "validation_report_labels_used_for": "metrics_after_parameters_frozen_only",
        "real_labels_used_for": "metrics_after_parameters_frozen_only",
        "calibration_report_pair_ids_disjoint": True,
    }
    assert result["validation_report"]["common_valid_count"] == 7
    assert result["real_dunhuang"]["common_valid_count"] == 7
    for local_method in LOCAL_METHOD_IDS:
        fused = fused_method_id(local_method)
        row = result["validation_report"]["methods"][fused]
        assert row["kind"] == "calibration_frozen_fusion"
        assert row["common_valid_metrics"]["row"]["auroc"] == pytest.approx(1.0)
        assert row["common_valid_metrics"]["row"]["auprc"] == pytest.approx(1.0)


def test_real_score_adapter_consumes_existing_pair_rows_and_cluster_ids():
    labels = [False, True, False, True]
    methods = {}
    for method in (MATCHED_ROUTE_A_SIAMESE_ID,) + LOCAL_METHOD_IDS:
        methods[method] = {
            "probability": [0.1, 0.9, 0.2, 0.8],
            "valid": [True, True, True, True],
        }
    table = real_score_table(
        {
            "dataset_id": "real_dunhuang_strict_alpha_v0_1",
            "pair_count": 4,
            "pair_ids": ["real-{}".format(index) for index in range(4)],
            "labels": labels,
            "cluster_ids": ["case-0", "case-0", "case-1", "case-1"],
            "methods": methods,
        }
    )

    assert table.phase == REAL_PHASE
    assert table.pair_count == 4
    assert table.cluster_ids == ("case-0", "case-0", "case-1", "case-1")
    assert set(table.probability) == set(methods)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(index, label):
    component = "component-{}".format(index // 2)

    def fragment(side):
        return MaskMemberRef(
            binding=MM_CANONICAL_BINDING,
            archive_member="fusion/{}/{}.png".format(index, side),
            fragment_id="fragment-{}-{}".format(index, side),
            dataset_id="mm_augmented",
            canonical_group_id="group-{}".format(index),
            component_id=component,
            split="val",
            threshold_rule="binary_brighter_value",
        )

    first = fragment("a")
    second = fragment("b")
    return TrainingPairRecord(
        fragment_a=first,
        fragment_b=second,
        label=label,
        direction_b_wrt_a="right" if label else None,
        dataset_id="mm_augmented",
        canonical_group_id="group-{}".format(index),
        component_id=component,
        split="val",
        canonical_pair_key=tuple(sorted((first.fragment_id, second.fragment_id))),
        label_origin="fixture",
    )


class _Plan:
    def __init__(self, batches):
        self.batches = batches

    def phase_batches(self, phase):
        assert phase == CALIBRATION_PHASE
        return tuple({"ordinal": index} for index in range(len(self.batches)))


class _Provider:
    def __init__(self, batches):
        self.batches = batches

    def planned_records(self, phase, batch_ordinal):
        assert phase == CALIBRATION_PHASE
        return self.batches[batch_ordinal]

    def prepare(self, records, *, arm, phase):
        assert phase == CALIBRATION_PHASE
        sequence = record_sequence_fingerprint(records)
        representation = arm.candidate_representation
        return PreparedAblationBatch(
            payload=tuple(records),
            sample_count=len(records),
            record_sequence_sha256=sequence,
            prepared_input_sha256=_sha("prepared:" + representation + sequence),
            local_candidate_sha256=_sha("local:" + representation + sequence),
            coarse_preprocessing_sha256="0" * 64,
            geometry_config_sha256="1" * 64,
            processing_counts={
                "coarse_preprocess_count": 2 * len(records),
                "geometry_build_count": 0,
                "geometry_cache_read_count": len(records),
                "geometry_cache_write_count": 0,
                "local_candidate_count": len(records),
                "mask_load_count": 2 * len(records),
            },
            candidate_representation=representation,
        )


class _Session:
    def predict_batch(self, batch, *, evidence):
        del evidence
        labels = torch.tensor([record.label for record in batch.payload])
        return PredictionBatch(
            probability=torch.where(
                labels,
                torch.full(labels.shape, 0.8),
                torch.full(labels.shape, 0.2),
            ),
            valid=torch.ones_like(labels),
        )


class _MeanSiamese(nn.Module):
    def forward(self, input_a, input_b):
        del input_b
        return input_a.mean((1, 2, 3))[:, None]


def test_phase_scorer_uses_complete_frozen_batches_and_all_five_sources():
    records = tuple(_record(index, index % 2 == 1) for index in range(4))
    batches = (records[:2], records[2:])
    masks = {}
    for record in records:
        mask = np.full((20, 20), record.label, dtype=np.bool_)
        masks[record.fragment_a.archive_member] = mask
        masks[record.fragment_b.archive_member] = mask
    backend = LocalQ1Backend(device="cpu", mode=LocalQ1BackendMode.FORMAL)

    def winner_loader(name):
        return comparison.LoadedArmWinner(
            arm=comparison._arm_from_backend(backend, AblationArmName(name)),
            session=_Session(),
            runner_mm_report={},
        )

    table = score_route_a_phase(
        phase=CALIBRATION_PHASE,
        plan=_Plan(batches),
        provider=_Provider(batches),
        arm_winner_loader=winner_loader,
        matched_model=_MeanSiamese(),
        matched_mask_loader=lambda reference: masks[reference.archive_member],
        matched_batch_size=3,
    )

    assert table.pair_ids == tuple(record.pair_id for record in records)
    assert table.labels.tolist() == [False, True, False, True]
    assert set(table.probability) == set(LOCAL_METHOD_IDS) | {
        MATCHED_ROUTE_A_SIAMESE_ID
    }
    assert all(value.tolist() == [True] * 4 for value in table.valid.values())


def test_cli_exposes_direct_remote_fusion_inputs():
    from staging.pairwise_v0_2.baselines import (
        coarse_local_fusion_evaluation as fusion,
    )

    help_text = fusion._parser().format_help()
    for flag in (
        "--batch-plan",
        "--run-directory",
        "--matched-checkpoint",
        "--real-evaluation",
        "--output",
    ):
        assert flag in help_text
