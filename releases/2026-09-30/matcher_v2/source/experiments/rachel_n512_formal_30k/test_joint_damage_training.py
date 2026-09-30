from collections import Counter
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from experiments.rachel_n512_formal_30k.train_joint_damage import (
    build_random_model, learning_rate, parse_smoke, state_digest)
from staging.pairwise_v0_2.pairwise_data.rachel_staged_damage_dataset import (
    StagedDamageDataset, exposure_schedule, schedule_audit)
from staging.pairwise_v0_2.tests.test_rachel_weathered_dataset import sample
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig


def test_exact_multiset_and_stable_label_blind_schedule():
    ids = ["pair-%d" % i for i in range(100)]
    for pid in ids:
        assert Counter(exposure_schedule(pid, arm="joint")) == Counter((0, 0, 0, 1, 2, 3))
        assert exposure_schedule(pid, arm="staged") == (0, 0, 0, 1, 2, 3)
    a, b = (schedule_audit(ids, arm=arm) for arm in ("staged", "joint"))
    assert a["total_slots"] == b["total_slots"] == {"0": 300, "1": 100, "2": 100, "3": 100}
    assert a["pair_id_order_sha256"] == b["pair_id_order_sha256"]
    assert a["schedule_sha256"] != b["schedule_sha256"]


def test_real_weather_arrays_same_multiset_and_clean_exact_identity():
    original = sample()
    class Train:
        split = "train"
        def __len__(self): return 1
        def __getitem__(self, index): return original
    datasets = [StagedDamageDataset(Train(), arm=arm) for arm in ("staged", "joint")]
    outputs = []
    for dataset in datasets:
        slots = {}
        for epoch in range(1, 7):
            dataset.set_epoch(epoch)
            value, report = dataset[0]
            slot = report["damage_schedule"]["weather_epoch"]
            if slot is None:
                assert value is original
                assert not report["changed_pair"]
            else:
                slots[slot] = value
        outputs.append(slots)
    for slot in (1, 2, 3):
        for name, value in vars(outputs[0][slot]).items():
            other = getattr(outputs[1][slot], name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, other)
            else:
                assert value == other


def test_source_weights_never_loaded_and_same_random_initialization():
    source = dict(model_kind="full", model_options={},
        model_config=asdict(RachelN512Config(window_sizes_px=(7., 16., 32., 64.))),
        loss_config=asdict(RachelN512LossConfig()),
        model_state_dict={"deliberately_invalid": torch.tensor(float("nan"))})
    first, _, _, digest = build_random_model(source)
    torch.randn(111)
    second, _, _, other = build_random_model(source)
    assert digest == other == state_digest(second)
    assert all(torch.isfinite(p).all() for p in first.parameters())


def test_shared_learning_rate_and_smoke_budget():
    assert [learning_rate(e) for e in range(1, 7)] == [1e-4] * 3 + [2e-5] * 3
    assert parse_smoke("128/8") == (128, 8)
    assert parse_smoke("128") == (128, 8)
    with pytest.raises(Exception): parse_smoke("128/7")
    with pytest.raises(ValueError): learning_rate(0)
