from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from staging.pairwise_v0_2.baselines.matched_route_a_siamese import (
    HistoricalProbabilityFocalLoss,
    MatchedPhasePopulation,
    MatchedSiameseContract,
    MatchedSiameseError,
    evaluate_matched_records,
    fit_cluster_balanced_f1_threshold,
    load_matched_phase_population,
    make_random_historical_mm_model,
    preload_mask_tensors,
    run_matched_siamese_training,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.local_q1_cache_builder import LocalQ1Population


def _record(
    index: int,
    *,
    split: str,
    dataset: str,
    label: bool,
    component: str,
) -> TrainingPairRecord:
    binding = MM_CANONICAL_BINDING if dataset == "mm_augmented" else (
        ECCV_CANONICAL_BINDING
    )
    group = "{}-group-{}".format(split, index)

    def reference(side: str) -> MaskMemberRef:
        return MaskMemberRef(
            binding=binding,
            archive_member="fixture/{}/{}-{}.png".format(split, index, side),
            fragment_id="{}-{}".format(index, side),
            dataset_id=dataset,
            canonical_group_id=group,
            component_id=component,
            split=split,
            threshold_rule="binary_brighter_value",
        )

    return TrainingPairRecord(
        fragment_a=reference("a"),
        fragment_b=reference("b"),
        label=label,
        direction_b_wrt_a="right" if label else None,
        dataset_id=dataset,
        canonical_group_id=group,
        component_id=component,
        split=split,
        canonical_pair_key=tuple(sorted(("{}-a".format(index), "{}-b".format(index)))),
        label_origin="fixture",
    )


def _records():
    train = tuple(
        _record(
            index,
            split="train",
            dataset="mm_augmented" if index < 2 else "eccv_1113data",
            label=index % 2 == 0,
            component="train-component-{}".format(index),
        )
        for index in range(4)
    )
    validation = []
    for phase_index in range(3):
        for offset in range(4):
            index = 100 + phase_index * 4 + offset
            validation.append(
                _record(
                    index,
                    split="val",
                    dataset="mm_augmented" if offset < 2 else "eccv_1113data",
                    label=offset % 2 == 0,
                    component="val-component-{}-{}".format(phase_index, offset),
                )
            )
    return train, tuple(validation)


def _fake_population(train, validation):
    # The unit test exercises the plan-to-record mapping in isolation.  The
    # production CLI obtains a fully validated instance from Route-A replay.
    population = object.__new__(LocalQ1Population)
    object.__setattr__(population, "training_records", tuple(train))
    object.__setattr__(population, "validation_records", tuple(validation))
    object.__setattr__(population, "freeze_file_sha256", "1" * 64)
    object.__setattr__(population, "freeze_content_sha256", "2" * 64)
    return population


def _phase_row(ordinals):
    return {
        "record_count": len(ordinals),
        "batch_count": 1,
        "batches": [{"ordinal": 0, "record_ordinals": list(ordinals)}],
    }


def _write_plan(path: Path, *, duplicate_validation: bool = False) -> None:
    report = [8, 9, 10, 11]
    if duplicate_validation:
        report[-1] = 0
    document = {
        "schema_version": "dunhuang-local-q1-batch-plan/0.4",
        "external_locks": {
            "freeze_file_sha256": "1" * 64,
            "freeze_content_sha256": "2" * 64,
        },
        "phases": {
            "train": _phase_row([3, 1, 0, 2]),
            "validation_select": _phase_row([0, 1, 2, 3]),
            "validation_calibration": _phase_row([4, 5, 6, 7]),
            "validation_report": _phase_row(report),
        },
    }
    payload = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    document["content_sha256"] = hashlib.sha256(payload).hexdigest()
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _phases() -> MatchedPhasePopulation:
    train, validation = _records()
    return MatchedPhasePopulation(
        train=train,
        validation_select=validation[:4],
        validation_calibration=validation[4:8],
        validation_report=validation[8:],
        plan_file_sha256="3" * 64,
        plan_content_sha256="4" * 64,
    )


def _mask_loader_factory(records):
    masks = {}
    for record in records:
        for reference in (record.fragment_a, record.fragment_b):
            mask = np.zeros((20, 30), dtype=np.bool_)
            if record.label:
                mask[2:18, 2:27] = True
            else:
                mask[8:12, 12:18] = True
            masks[reference.archive_member] = mask

    class Loader:
        def __call__(self, reference):
            return masks[reference.archive_member]

        def close(self):
            return None

    return Loader


def test_plan_uses_all_train_and_strict_disjoint_validation_roles(tmp_path):
    train, validation = _records()
    population = _fake_population(train, validation)
    path = tmp_path / "plan.json"
    _write_plan(path)

    phases = load_matched_phase_population(path, population)

    assert tuple(row.pair_id for row in phases.train) == tuple(
        train[index].pair_id for index in (3, 1, 0, 2)
    )
    assert len(phases.validation_select) == 4
    assert len(phases.validation_calibration) == 4
    assert len(phases.validation_report) == 4
    component_sets = [
        {row.component_id for row in phases.records(phase)}
        for phase in (
            "validation_select",
            "validation_calibration",
            "validation_report",
        )
    ]
    assert not component_sets[0].intersection(component_sets[1])
    assert not component_sets[0].intersection(component_sets[2])
    assert not component_sets[1].intersection(component_sets[2])


def test_plan_rejects_validation_ordinal_overlap(tmp_path):
    train, validation = _records()
    path = tmp_path / "plan.json"
    _write_plan(path, duplicate_validation=True)

    with pytest.raises(MatchedSiameseError, match="disjoint and exhaustive"):
        load_matched_phase_population(path, _fake_population(train, validation))


def test_historical_focal_matches_companion_probability_formula():
    probability = torch.tensor([0.9, 0.2, 0.4], dtype=torch.float32)
    target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    observed = HistoricalProbabilityFocalLoss(alpha=0.25, gamma=2.0)(
        probability, target
    )
    bce = torch.nn.functional.binary_cross_entropy(
        probability, target, reduction="none"
    )
    expected = (0.25 * (1.0 - torch.exp(-bce)).pow(2.0) * bce).mean()
    assert torch.equal(observed, expected)


def test_historical_random_initialization_is_seeded_and_replays_linear_rule():
    first = make_random_historical_mm_model(29, device=torch.device("cpu"))
    second = make_random_historical_mm_model(29, device=torch.device("cpu"))

    assert torch.equal(first.fc[0].weight, second.fc[0].weight)
    assert torch.equal(first.fc[2].weight, second.fc[2].weight)
    assert torch.allclose(first.fc[0].bias, torch.full_like(first.fc[0].bias, 0.01))
    assert torch.allclose(first.fc[2].bias, torch.full_like(first.fc[2].bias, 0.01))


def test_preload_is_mask_only_unique_and_64_square():
    phases = _phases()
    records = phases.train + phases.validation_select
    tensors = preload_mask_tensors(
        records,
        _mask_loader_factory(records),
        workers=2,
    )

    unique = {
        reference.archive_member
        for record in records
        for reference in (record.fragment_a, record.fragment_b)
    }
    assert len(tensors) == len(unique)
    assert all(value.dtype == torch.float32 for value in tensors.values())
    assert all(value.shape == (1, 64, 64) for value in tensors.values())


def test_metrics_and_calibration_are_dataset_cluster_aware():
    rows = _phases().validation_calibration
    probability = torch.tensor([0.9, 0.1, 0.8, 0.2], dtype=torch.float64)

    report = evaluate_matched_records(rows, probability, threshold=0.5)
    threshold = fit_cluster_balanced_f1_threshold(rows, probability)

    assert report["equal_dataset_macro_cluster"]["auroc"] == pytest.approx(1.0)
    assert report["equal_dataset_macro_cluster"]["auprc"] == pytest.approx(1.0)
    assert set(report["by_dataset"]) == {"mm_augmented", "eccv_1113data"}
    assert report["coverage"]["valid_count"] == 4
    assert report["direction"]["accuracy"] is None
    assert threshold["cluster_balanced_f1"] == pytest.approx(1.0)


class _TinyOrderedSiamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_a, input_b):
        value = (input_a.mean((1, 2, 3)) + input_b.mean((1, 2, 3))) / 2.0
        return torch.sigmoid(self.weight * value + self.bias)[:, None]


def test_end_to_end_winner_uses_select_then_calibration_then_report(tmp_path):
    phases = _phases()
    all_records = (
        phases.train
        + phases.validation_select
        + phases.validation_calibration
        + phases.validation_report
    )
    output = tmp_path / "run"

    receipt = run_matched_siamese_training(
        phases=phases,
        loader_factory=_mask_loader_factory(all_records),
        output_dir=output,
        contract=MatchedSiameseContract(
            epochs=2,
            seed=17,
            train_batch_size=2,
            eval_batch_size=3,
            preprocess_workers=2,
        ),
        device=torch.device("cpu"),
        model_factory=_TinyOrderedSiamese,
    )

    assert receipt["status"] == "complete_matched_route_a_whole_mask_siamese"
    assert len(receipt["training"]["epochs"]) == 2
    assert receipt["training"]["total_pair_presentations"] == 8
    assert receipt["winner"]["epoch"] in {1, 2}
    assert receipt["validation_report"]["coverage"]["record_count"] == 4
    assert receipt["validation_report"]["direction"]["accuracy"] is None
    assert (output / "matched_historical_mm_siamese_winner.pt").is_file()
    assert (output / "matched_siamese_receipt.json").is_file()
    assert (output / "validation_report_scores.json").is_file()
