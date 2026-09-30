from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from staging.pairwise_v0_2.baselines.historical_mm_siamese import (
    HistoricalMMBaselineError,
    preprocess_historical_mm_mask,
    score_historical_mm_validation,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MM_CANONICAL_BINDING,
    MaskMemberRef,
    TrainingPairRecord,
)


def _record(index: int, label: bool, *, dataset: str = "mm_augmented", split: str = "val"):
    component = "component-{}".format(index // 2)
    fragment_a = MaskMemberRef(
        binding=MM_CANONICAL_BINDING,
        archive_member="fixture/a-{}.png".format(index),
        fragment_id="a-{}".format(index),
        dataset_id=dataset,
        canonical_group_id="group-{}".format(index),
        component_id=component,
        split=split,
        threshold_rule="binary_brighter_value",
    )
    fragment_b = MaskMemberRef(
        binding=MM_CANONICAL_BINDING,
        archive_member="fixture/b-{}.png".format(index),
        fragment_id="b-{}".format(index),
        dataset_id=dataset,
        canonical_group_id="group-{}".format(index),
        component_id=component,
        split=split,
        threshold_rule="binary_brighter_value",
    )
    return TrainingPairRecord(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        label=label,
        direction_b_wrt_a="right" if label else None,
        dataset_id=dataset,
        canonical_group_id="group-{}".format(index),
        component_id=component,
        split=split,
        canonical_pair_key=tuple(sorted(("a-{}".format(index), "b-{}".format(index)))),
        label_origin="fixture",
    )


class _MeanProbability(nn.Module):
    def forward(self, input_a, input_b):
        probability = (input_a.mean(dim=(1, 2, 3)) + input_b.mean(dim=(1, 2, 3))) / 2
        return probability[:, None]


def test_preprocess_reproduces_legacy_torchvision_transform():
    mask = np.zeros((37, 91), dtype=np.bool_)
    mask[3:29, 17:68] = True
    mask[13:20, 68:89] = True
    source = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    legacy = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )(source)

    observed = preprocess_historical_mm_mask(mask)

    assert observed.dtype == torch.float32
    assert observed.shape == (1, 64, 64)
    assert torch.equal(observed, legacy)


def test_scores_exact_supplied_order_and_returns_auroc_auprc():
    records = tuple(_record(index, index % 2 == 0) for index in range(4))
    masks = {}
    foreground_fractions = (0.9, 0.1, 0.8, 0.2)
    for record, fraction in zip(records, foreground_fractions):
        height = 20
        width = 20
        count = round(height * width * fraction)
        mask = np.zeros((height * width,), dtype=np.bool_)
        mask[:count] = True
        mask = mask.reshape(height, width)
        masks[record.fragment_a.archive_member] = mask
        masks[record.fragment_b.archive_member] = mask

    result = score_historical_mm_validation(
        _MeanProbability(),
        records,
        lambda reference: masks[reference.archive_member],
        batch_size=3,
    )

    assert result.pair_ids == tuple(record.pair_id for record in records)
    assert result.cluster_ids == tuple(record.component_id for record in records)
    assert result.label.tolist() == [True, False, True, False]
    assert result.pair_count == 4
    assert result.metrics["row"]["auroc"] == pytest.approx(1.0)
    assert result.metrics["row"]["auprc"] == pytest.approx(1.0)
    assert result.summary()["score_semantics"].startswith("ordered_A_then_B")


@pytest.mark.parametrize(
    ("dataset", "split", "message"),
    [
        ("eccv_1113data", "val", "mm_augmented"),
        ("mm_augmented", "train", "validation"),
    ],
)
def test_rejects_non_mm_or_non_validation_records(dataset, split, message):
    record = _record(0, True, dataset=dataset, split=split)
    mask = np.zeros((16, 16), dtype=np.bool_)
    mask[2:14, 2:14] = True

    with pytest.raises(HistoricalMMBaselineError, match=message):
        score_historical_mm_validation(
            _MeanProbability(),
            [record],
            lambda _reference: mask,
        )


def test_rejects_non_boolean_mask_before_model_execution():
    record = _record(0, True)
    mask = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(HistoricalMMBaselineError, match="2D bool mask"):
        score_historical_mm_validation(
            _MeanProbability(),
            [record],
            lambda _reference: mask,
        )
