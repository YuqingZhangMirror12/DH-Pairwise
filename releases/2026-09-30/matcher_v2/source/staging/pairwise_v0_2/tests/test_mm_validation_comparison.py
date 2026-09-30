from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from staging.pairwise_v0_2.baselines import mm_validation_comparison as comparison
from staging.pairwise_v0_2.baselines.historical_mm_siamese import (
    HISTORICAL_MM_BASELINE_ID,
)
from staging.pairwise_v0_2.baselines.matched_route_a_siamese import (
    MATCHED_ROUTE_A_SIAMESE_ID,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.evaluation import evaluate_pairwise
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


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(index: int, *, dataset: str, label: bool) -> TrainingPairRecord:
    binding = MM_CANONICAL_BINDING if dataset == "mm_augmented" else ECCV_CANONICAL_BINDING
    component = "component-{}-{}".format(dataset, index // 2)
    group = "group-{}-{}".format(dataset, index)

    def fragment(side: str) -> MaskMemberRef:
        return MaskMemberRef(
            binding=binding,
            archive_member="fixture/{}/{}-{}.png".format(dataset, side, index),
            fragment_id="{}-{}-{}".format(dataset, side, index),
            dataset_id=dataset,
            canonical_group_id=group,
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
        dataset_id=dataset,
        canonical_group_id=group,
        component_id=component,
        split="val",
        canonical_pair_key=tuple(sorted((first.fragment_id, second.fragment_id))),
        label_origin="fixture",
    )


class _Plan:
    def __init__(self, batch_count):
        self._entries = tuple({"ordinal": index} for index in range(batch_count))

    def phase_batches(self, phase):
        assert phase == "validation_report"
        return self._entries


class _Provider:
    def __init__(self, batches, *, poison_arm=None):
        self.batches = tuple(tuple(batch) for batch in batches)
        self.poison_arm = poison_arm

    def planned_records(self, phase, batch_ordinal):
        assert phase == "validation_report"
        return self.batches[batch_ordinal]

    def prepare(self, records, *, arm, phase):
        assert phase == "validation_report"
        sequence = record_sequence_fingerprint(records)
        representation = arm.candidate_representation
        poison = ""
        if arm.name == self.poison_arm:
            poison = arm.name.value
        prepared = _sha("prepared:" + representation + ":" + sequence + poison)
        local = _sha("local:" + representation + ":" + sequence + poison)
        return PreparedAblationBatch(
            payload=tuple(records),
            sample_count=len(records),
            record_sequence_sha256=sequence,
            prepared_input_sha256=prepared,
            local_candidate_sha256=local,
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
    def __init__(self, score_by_pair, invalid_pairs=()):
        self.score_by_pair = score_by_pair
        self.invalid_pairs = frozenset(invalid_pairs)

    def predict_batch(self, batch, *, evidence):
        del evidence
        records = batch.payload
        return PredictionBatch(
            probability=torch.tensor(
                [self.score_by_pair.get(record.pair_id, 0.5) for record in records],
                dtype=torch.float32,
            ),
            valid=torch.tensor(
                [record.pair_id not in self.invalid_pairs for record in records],
                dtype=torch.bool,
            ),
        )


class _HistoricalMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, input_a, input_b):
        del input_b
        return input_a.mean(dim=(1, 2, 3), keepdim=False)[:, None]


def _fixture(poison_arm=None):
    mm = tuple(
        _record(index, dataset="mm_augmented", label=index % 2 == 0)
        for index in range(4)
    )
    eccv = tuple(
        _record(index + 10, dataset="eccv_1113data", label=index % 2 == 0)
        for index in range(2)
    )
    batches = ((mm[0], eccv[0], mm[1]), (mm[2], mm[3], eccv[1]))
    provider = _Provider(batches, poison_arm=poison_arm)
    masks = {}
    for record, fraction in zip(mm, (0.9, 0.1, 0.8, 0.2)):
        mask = np.zeros(400, dtype=np.bool_)
        mask[: round(400 * fraction)] = True
        mask = mask.reshape(20, 20)
        masks[record.fragment_a.archive_member] = mask
        masks[record.fragment_b.archive_member] = mask
    return mm, batches, provider, masks


def _winner_loader(mm, *, invalid_arm=None, invalid_pair=None):
    backend = LocalQ1Backend(device="cpu", mode=LocalQ1BackendMode.FORMAL)
    labels = [record.label for record in mm]
    clusters = [record.component_id for record in mm]

    def load(name):
        name = AblationArmName(name)
        arm = comparison._arm_from_backend(backend, name)
        scores = {
            record.pair_id: (0.92 - index * 0.03 if record.label else 0.08 + index * 0.03)
            for index, record in enumerate(mm)
        }
        invalid = {invalid_pair} if name == invalid_arm and invalid_pair else set()
        valid = [record.pair_id not in invalid for record in mm]
        report = evaluate_pairwise(
            [scores[record.pair_id] for record in mm],
            labels,
            valid,
            clusters,
            threshold=0.5,
        )
        return comparison.LoadedArmWinner(
            arm=arm,
            session=_Session(scores, invalid),
            runner_mm_report=report,
        )

    return load


def test_five_methods_use_identical_mm_report_pair_ids_and_common_valid_rows():
    mm, batches, provider, masks = _fixture()
    invalid_arm = AblationArmName.KEYPOINT_DUSTBIN_SINKHORN
    loader = _winner_loader(
        mm,
        invalid_arm=invalid_arm,
        invalid_pair=mm[0].pair_id,
    )

    result = comparison.compare_mm_validation_report(
        plan=_Plan(len(batches)),
        provider=provider,
        arm_winner_loader=loader,
        historical_model=_HistoricalMean(),
        historical_mask_loader=lambda reference: masks[reference.archive_member],
    )

    assert result["pair_ids"] == [record.pair_id for record in mm]
    assert result["pair_count"] == 4
    assert result["common_valid_count"] == 3
    assert result["same_pair_ids_all_methods"] is True
    assert len(result["methods"]) == 5
    assert HISTORICAL_MM_BASELINE_ID in result["methods"]
    for name in comparison._FOUR_ARMS:
        assert result["methods"][name.value]["runner_ranking_metrics_reproduced"] is True
        assert result["methods"][name.value]["common_valid_metrics"]["row"][
            "auroc"
        ] == pytest.approx(1.0)


def test_optional_matched_siamese_forms_six_method_same_pair_comparison():
    mm, batches, provider, masks = _fixture()
    observed_members = []

    def shared_mask_loader(reference):
        observed_members.append(reference.archive_member)
        return masks[reference.archive_member]

    result = comparison.compare_mm_validation_report(
        plan=_Plan(len(batches)),
        provider=provider,
        arm_winner_loader=_winner_loader(mm),
        historical_model=_HistoricalMean(),
        historical_mask_loader=shared_mask_loader,
        matched_model=_HistoricalMean(),
        matched_batch_size=3,
    )

    expected_members = [
        reference.archive_member
        for record in mm
        for reference in (record.fragment_a, record.fragment_b)
    ]
    assert observed_members == expected_members * 2
    assert result["status"] == "complete_same_mm_pair_ids_six_method_comparison"
    assert result["pair_ids"] == [record.pair_id for record in mm]
    assert result["labels"] == [record.label for record in mm]
    assert result["common_valid_count"] == 4
    assert result["primary_metric_population"] == (
        "intersection_valid_across_all_six_methods"
    )
    assert result["matched_route_a_siamese_checkpoint_evaluated"] is True
    assert len(result["methods"]) == 6
    matched = result["methods"][MATCHED_ROUTE_A_SIAMESE_ID]
    assert matched["native_valid_count"] == 4
    assert matched["native_metrics"]["row"]["auroc"] == pytest.approx(1.0)
    assert matched["native_metrics"]["row"]["auprc"] == pytest.approx(1.0)
    assert matched["common_valid_metrics"]["row"]["auroc"] == pytest.approx(1.0)
    assert matched["common_valid_metrics"]["row"]["auprc"] == pytest.approx(1.0)


def test_matched_cli_arguments_are_optional():
    actions = {action.dest: action for action in comparison._parser()._actions}
    assert actions["matched_checkpoint"].required is False
    assert actions["matched_checkpoint"].default is None
    assert actions["matched_batch_size"].required is False
    assert actions["matched_batch_size"].default == 256


def test_rejects_different_dual_and_sinkhorn_tensors_within_representation():
    poisoned = AblationArmName.LOCAL_DUSTBIN_SINKHORN
    mm, batches, provider, masks = _fixture(poison_arm=poisoned)

    with pytest.raises(
        comparison.MMValidationComparisonError,
        match="inputs differ within one representation",
    ):
        comparison.compare_mm_validation_report(
            plan=_Plan(len(batches)),
            provider=provider,
            arm_winner_loader=_winner_loader(mm),
            historical_model=_HistoricalMean(),
            historical_mask_loader=lambda reference: masks[reference.archive_member],
        )
