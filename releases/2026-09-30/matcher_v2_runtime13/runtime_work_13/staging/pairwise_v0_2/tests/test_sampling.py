import pytest

from staging.pairwise_v0_2.pairwise_data.sampling import (
    BalancedPairSampler,
    BalancedSamplingConfig,
    SamplingError,
    iter_frozen_validation,
    validation_stream_fingerprint,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)


BINDING = ArchiveBinding("local_asset://sampling_fixture", "zip", "1" * 64)


def _record(index, label, *, split="train", hard_score=None):
    common = dict(
        binding=BINDING,
        dataset_id="fixture",
        canonical_group_id="fixture/group/{}".format(index),
        component_id="fixture/component/{}".format(index),
        split=split,
        threshold_rule="grayscale_uint8_gt_127",
        content_sha256="2" * 64,
    )
    first_id = "fixture/{}/a".format(index)
    second_id = "fixture/{}/b".format(index)
    first = MaskMemberRef(
        archive_member="masks/{}/a.png".format(index),
        fragment_id=first_id,
        **common,
    )
    second = MaskMemberRef(
        archive_member="masks/{}/b.png".format(index),
        fragment_id=second_id,
        **common,
    )
    return TrainingPairRecord(
        fragment_a=first,
        fragment_b=second,
        label=label,
        direction_b_wrt_a="right" if label else None,
        dataset_id="fixture",
        canonical_group_id=common["canonical_group_id"],
        component_id=common["component_id"],
        split=split,
        canonical_pair_key=tuple(sorted((first_id, second_id))),
        label_origin="fixture",
        static_hard_negative_score=hard_score,
    )


def test_balanced_sampler_exact_ratios_hard_negatives_and_determinism():
    records = (
        [_record(index, True) for index in range(10)]
        + [_record(100 + index, False, hard_score=0.8) for index in range(4)]
        + [_record(200 + index, False, hard_score=0.0) for index in range(10)]
    )
    config = BalancedSamplingConfig(
        epoch_size=12,
        positive_fraction=0.5,
        hard_negative_fraction=0.5,
        seed="fixture",
        replacement_on_shortfall=False,
    )
    first_sampler = BalancedPairSampler(config)
    second_sampler = BalancedPairSampler(config)
    first = list(first_sampler.sample_epoch(records, epoch=3))
    second = list(second_sampler.sample_epoch(records, epoch=3))

    assert [item.pair_id for item in first] == [item.pair_id for item in second]
    assert len(first) == 12
    assert sum(item.label for item in first) == 6
    assert (
        sum(
            (not item.label) and (item.static_hard_negative_score or 0) > 0
            for item in first
        )
        == 3
    )
    assert first_sampler.last_audit.output_hard_negative_count == 3
    assert first_sampler.last_audit.replacement_count == 0


def test_sampler_can_use_model_mined_pair_scores():
    positive = [_record(index, True) for index in range(3)]
    negatives = [_record(10 + index, False, hard_score=None) for index in range(3)]
    mined = {negatives[0].pair_id: 0.9}
    sampler = BalancedPairSampler(
        BalancedSamplingConfig(
            epoch_size=4,
            positive_fraction=0.5,
            hard_negative_fraction=0.5,
            hard_negative_threshold=0.5,
            seed="mined",
        )
    )

    output = list(
        sampler.sample_epoch(positive + negatives, epoch=0, mined_negative_scores=mined)
    )

    assert len(output) == 4
    assert sum(item.label for item in output) == 2
    assert sum(item.pair_id == negatives[0].pair_id for item in output) == 1


def test_train_sampler_rejects_validation_records():
    sampler = BalancedPairSampler(BalancedSamplingConfig(epoch_size=2))
    with pytest.raises(SamplingError, match="train-only"):
        list(sampler.sample_epoch([_record(1, True, split="val")], epoch=0))


def test_validation_is_frozen_unbalanced_and_fingerprinted():
    records = [
        _record(0, False, split="val", hard_score=0.9),
        _record(1, True, split="val"),
        _record(2, False, split="val", hard_score=0.0),
    ]

    frozen = list(iter_frozen_validation(records))
    first = validation_stream_fingerprint(records)
    second = validation_stream_fingerprint(records)

    assert frozen == records
    assert first == second
    assert first["count"] == 3
    assert first["positive_count"] == 1
    assert first["negative_count"] == 2


def test_validation_stream_rejects_train_records():
    with pytest.raises(SamplingError, match="only val"):
        list(iter_frozen_validation([_record(0, True)]))
