from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from staging.pairwise_v0_2.training import exact_seam_pilot as pilot_module
from staging.pairwise_v0_2.training.checkpoint import CheckpointReceipt
from staging.pairwise_v0_2.training.exact_seam_pilot import (
    ExactSeamPilotConfig,
    ExactSeamPilotError,
    _load_scalar_mask,
    _parser,
    _records_for_group,
    _select_arm_winner,
    discover_exact_mask_groups,
)
from staging.pairwise_v0_2.training.local_q1_backend import LocalQ1Session


def _write_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255, mode="L").save(path)


def _four_fragment_partition(group: Path) -> None:
    # Fragment 0 touches 1 and 2; fragment 1 touches 0 and 3; 0/3 and 1/2
    # are exact negatives.  All masks remain in one aligned parent canvas.
    masks = []
    for row_slice, column_slice in (
        (slice(0, 8), slice(0, 8)),
        (slice(0, 8), slice(8, 16)),
        (slice(8, 16), slice(0, 8)),
        (slice(8, 16), slice(8, 16)),
    ):
        mask = np.zeros((16, 16), dtype=np.bool_)
        mask[row_slice, column_slice] = True
        masks.append(mask)
    for index, mask in enumerate(masks):
        _write_mask(group / (str(index) + ".png"), mask)


def test_pilot_discovers_only_requested_no_erode_generator(tmp_path: Path) -> None:
    selected = tmp_path / "gen4voronoi" / "no_erode" / "10"
    ignored = tmp_path / "gen5voronoi" / "no_erode" / "11"
    _four_fragment_partition(selected)
    _four_fragment_partition(ignored)

    groups = discover_exact_mask_groups(tmp_path, generator="gen4voronoi", seed=17)

    assert groups == (selected,)


def test_group_adapter_emits_exact_positive_negative_mask_pairs(tmp_path: Path) -> None:
    group = tmp_path / "gen4voronoi" / "no_erode" / "10"
    _four_fragment_partition(group)

    records = _records_for_group(tmp_path, group, split="train", seed=19)

    assert len(records) == 6
    assert sum(value.label for value in records) == 4
    assert sum(not value.label for value in records) == 2
    assert all(
        value.direction_b_wrt_a is not None
        if value.label
        else value.direction_b_wrt_a is None
        for value in records
    )
    assert all(value.provenance["mask_only"] is True for value in records)
    assert all(value.provenance["rotation_augmentation"] is False for value in records)


def test_group_eligibility_matches_legacy_single_pair_and_skips_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group = tmp_path / "gen4voronoi" / "no_erode" / "10"
    _four_fragment_partition(group)
    records = _records_for_group(tmp_path, group, split="train", seed=19)
    loader = pilot_module._DirectoryMaskLoader(tmp_path)
    cache = pilot_module.GeometryArtifactCache(tmp_path / "cache")
    geometry_config = pilot_module.GeometryBatchConfig()
    digest_calls = []

    def fake_digests(batch):
        digest_calls.append(batch)
        return "1" * 64, "2" * 64

    monkeypatch.setattr(pilot_module, "local_q1_prepared_digests", fake_digests)
    legacy = []
    for record in records:
        try:
            pilot_module._prepared_batch(
                (record,),
                loader=loader,
                cache=cache,
                geometry_config=geometry_config,
            )
        except (ExactSeamPilotError, pilot_module.GeometryBatchError):
            legacy.append(False)
        else:
            legacy.append(True)
    digest_count_after_legacy = len(digest_calls)

    grouped = pilot_module.exact_seam_group_eligibility(
        records,
        loader=loader,
        cache=cache,
        geometry_config=geometry_config,
    )

    assert grouped == tuple(legacy)
    assert len(digest_calls) == digest_count_after_legacy


def test_group_eligibility_isolates_one_record_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group = tmp_path / "gen4voronoi" / "no_erode" / "10"
    _four_fragment_partition(group)
    records = _records_for_group(tmp_path, group, split="train", seed=19)
    failed_pair_id = records[2].pair_id
    calls = []

    class FakeTargets:
        def __init__(self, values) -> None:
            self.sample_index = torch.arange(len(values), dtype=torch.long)
            self.assignment_target_a = torch.tensor(
                [[0] if value.label else [-1] for value in values],
                dtype=torch.long,
            )

    class FakeBatch:
        def __init__(self, values) -> None:
            self.labels = torch.tensor(
                [value.label for value in values], dtype=torch.bool
            )
            self.geometry_valid = torch.ones(len(values), dtype=torch.bool)
            self.targets = FakeTargets(values)

        def exact_loss_targets(self):
            return self.targets

        def model_inputs(self):
            return {}

    def fake_build(values, *_args, **_kwargs):
        calls.append(tuple(value.pair_id for value in values))
        if any(value.pair_id == failed_pair_id for value in values):
            raise pilot_module.GeometryBatchError("fixture failure")
        return FakeBatch(values)

    monkeypatch.setattr(pilot_module, "build_geometry_batch", fake_build)

    observed = pilot_module.exact_seam_group_eligibility(
        records,
        loader=object(),
        cache=object(),
        geometry_config=pilot_module.GeometryBatchConfig(),
    )

    assert observed == tuple(value.pair_id != failed_pair_id for value in records)
    assert calls[0] == tuple(value.pair_id for value in records)
    assert (failed_pair_id,) in calls


def test_group_filter_preserves_legacy_population_order_quota_and_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_group = tmp_path / "gen4voronoi" / "no_erode" / "train"
    validation_group = tmp_path / "gen4voronoi" / "no_erode" / "val"
    _four_fragment_partition(train_group)
    _four_fragment_partition(validation_group)

    class TinyPilotConfig(ExactSeamPilotConfig):
        @property
        def train_pair_count(self) -> int:
            return 4

        @property
        def validation_pair_count(self) -> int:
            return 4

    config = TinyPilotConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        max_pairs=500,
    )
    monkeypatch.setattr(
        pilot_module,
        "discover_exact_mask_groups",
        lambda *_args, **_kwargs: (train_group, validation_group),
    )
    monkeypatch.setattr(
        pilot_module,
        "_split_for_group",
        lambda relative, *_args: "val" if relative.endswith("/val") else "train",
    )

    legacy = pilot_module.build_exact_seam_pilot_population(
        config,
        record_filter=lambda _record: True,
    )
    group_calls = []

    def accept_group(rows):
        group_calls.append(tuple(value.pair_id for value in rows))
        return tuple(True for _ in rows)

    grouped = pilot_module.build_exact_seam_pilot_population(
        config,
        group_record_filter=accept_group,
    )

    assert grouped == legacy
    assert [len(values) for values in group_calls] == [6, 6]


def test_pilot_rejects_rgb_input_and_cpu_training(tmp_path: Path) -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    with pytest.raises(ExactSeamPilotError, match="scalar masks only"):
        _load_scalar_mask(rgb_path)

    with pytest.raises(ValueError, match="CUDA"):
        ExactSeamPilotConfig(
            mask_root=tmp_path,
            output_root=tmp_path / "output",
            cache_root=tmp_path / "cache",
            device="cpu",
        )


def test_pilot_defaults_to_balanced_1000_pair_three_epoch_run(tmp_path: Path) -> None:
    config = ExactSeamPilotConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
    )

    assert config.max_pairs == 1000
    assert config.train_pair_count == 800
    assert config.validation_pair_count == 200
    assert config.epochs == 3
    assert config.exact_loss_weight == 0.25
    assert config.exact_loss_schedule is None
    assert config.resolved_exact_loss_schedule == (0.25, 0.25, 0.25)
    assert config.initialization_seed is None
    assert config.resolved_initialization_seed == config.seed
    assert config.population_workers == 1
    assert config.device == "cuda"


def test_pilot_accepts_balanced_6964_pair_scale_and_cli(tmp_path: Path) -> None:
    config = ExactSeamPilotConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        max_pairs=6964,
    )

    assert config.max_pairs == 6964
    assert config.train_pair_count == 5570
    assert config.validation_pair_count == 1394
    assert config.train_pair_count // 2 == 2785
    assert config.validation_pair_count // 2 == 697

    parsed = _parser().parse_args(
        [
            "--mask-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "cli-output"),
            "--max-pairs",
            "6964",
            "--population-workers",
            "4",
        ]
    )
    assert parsed.max_pairs == 6964
    assert parsed.population_workers == 4


def test_pilot_accepts_balanced_30000_pair_scale_snapshot_and_cli(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "population-30000.json"
    config = ExactSeamPilotConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        max_pairs=30000,
        population_snapshot=snapshot_path,
    )

    assert config.train_pair_count == 24000
    assert config.validation_pair_count == 6000
    assert config.train_pair_count // 2 == 12000
    assert config.validation_pair_count // 2 == 3000
    assert config.population_snapshot == snapshot_path

    parsed = _parser().parse_args(
        [
            "--mask-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "cli-output"),
            "--max-pairs",
            "30000",
            "--population-snapshot",
            str(snapshot_path),
        ]
    )
    assert parsed.max_pairs == 30000
    assert parsed.population_snapshot == snapshot_path

    with pytest.raises(ValueError, match="max_pairs"):
        ExactSeamPilotConfig(
            mask_root=tmp_path,
            output_root=tmp_path / "too-large",
            cache_root=tmp_path / "cache",
            max_pairs=30002,
        )


def test_initialization_seed_is_optional_validated_and_exposed_by_cli(
    tmp_path: Path,
) -> None:
    parsed = _parser().parse_args(
        [
            "--mask-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "cli-output"),
            "--seed",
            "260830",
            "--initialization-seed",
            "41",
        ]
    )
    assert parsed.seed == 260830
    assert parsed.initialization_seed == 41

    config = ExactSeamPilotConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        seed=260830,
        initialization_seed=41,
    )
    assert config.resolved_initialization_seed == 41

    with pytest.raises(ValueError, match="initialization_seed"):
        ExactSeamPilotConfig(
            mask_root=tmp_path,
            output_root=tmp_path / "bad-output",
            cache_root=tmp_path / "bad-cache",
            initialization_seed=-1,
        )


def test_explicit_schedule_and_cli_require_one_weight_per_epoch(
    tmp_path: Path,
) -> None:
    config = ExactSeamPilotConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        epochs=5,
        exact_loss_schedule=(0.25, 0.05, 0.0, 0.0, 0.0),
    )
    assert config.resolved_exact_loss_schedule == (0.25, 0.05, 0.0, 0.0, 0.0)

    parsed = _parser().parse_args(
        [
            "--mask-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "cli-output"),
            "--epochs",
            "5",
            "--exact-loss-schedule",
            "0.25",
            "0.05",
            "0",
            "0",
            "0",
        ]
    )
    assert parsed.exact_loss_schedule == [0.25, 0.05, 0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="length must equal epochs"):
        ExactSeamPilotConfig(
            mask_root=tmp_path,
            output_root=tmp_path / "bad-output",
            cache_root=tmp_path / "bad-cache",
            epochs=5,
            exact_loss_schedule=(0.25, 0.05),
        )


def test_winner_policy_uses_auroc_then_auprc_then_earlier_epoch() -> None:
    epochs = [
        {
            "epoch": 1,
            "validation": {"weak": {"auroc": 0.7, "auprc": 0.8}},
        },
        {
            "epoch": 2,
            "validation": {"weak": {"auroc": 0.7, "auprc": 0.85}},
        },
        {
            "epoch": 3,
            "validation": {"weak": {"auroc": 0.7, "auprc": 0.85}},
        },
    ]

    assert _select_arm_winner(epochs, "weak")["epoch"] == 2


def test_dynamic_exact_weight_does_not_replace_optimizer() -> None:
    session = object.__new__(LocalQ1Session)
    optimizer = object()
    object.__setattr__(session, "_uses_exact_assignment", True)
    object.__setattr__(session, "_exact_assignment_loss_weight", 0.25)
    session.optimizer = optimizer

    session.set_exact_assignment_loss_weight(0.05)
    assert session.exact_assignment_loss_weight == 0.05
    assert session.optimizer is optimizer

    session.set_exact_assignment_loss_weight(0.0)
    assert session.exact_assignment_loss_weight == 0.0
    assert session.optimizer is optimizer


def test_five_epoch_schedule_saves_each_checkpoint_and_selects_arm_winners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = (
        SimpleNamespace(pair_id="positive", label=True),
        SimpleNamespace(pair_id="negative", label=False),
    )
    population = SimpleNamespace(
        training_records=records,
        validation_records=records,
        discovered_group_count=2,
        consumed_group_count=2,
    )

    class FakeSession:
        def __init__(self, name: str) -> None:
            self.name = name
            self.model = torch.nn.Linear(1, 1)
            self.optimizer = object()
            self.initial_model_state_sha256 = "0" * 64
            self.weights = []

        def set_exact_assignment_loss_weight(self, value: float) -> None:
            self.weights.append(float(value))

        def train_batch(self, _prepared: object) -> SimpleNamespace:
            exact = self.name == "exact"
            return SimpleNamespace(
                loss=1.0,
                valid_count=2,
                diagnostics={
                    "exact_assignment_loss": 3.0 if exact else 0.0,
                    "exact_supervised_match_count": 1 if exact else 0,
                    "exact_supervised_dustbin_a_count": 4 if exact else 0,
                    "exact_supervised_dustbin_b_count": 4 if exact else 0,
                },
            )

    sessions = {name: FakeSession(name) for name in ("weak", "exact")}

    session_seeds = []

    class FakeBackend:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def create_session(self, arm: SimpleNamespace, *, seed: int) -> FakeSession:
            session_seeds.append(seed)
            return sessions[arm.pilot_name]

    metric_rows = [
        {
            "weak": {"auroc": 0.70, "auprc": 0.80},
            "exact": {"auroc": 0.60, "auprc": 0.70},
        },
        {
            "weak": {"auroc": 0.70, "auprc": 0.85},
            "exact": {"auroc": 0.80, "auprc": 0.80},
        },
        {
            "weak": {"auroc": 0.70, "auprc": 0.85},
            "exact": {"auroc": 0.80, "auprc": 0.82},
        },
        {
            "weak": {"auroc": 0.69, "auprc": 0.84},
            "exact": {"auroc": 0.80, "auprc": 0.82},
        },
        {
            "weak": {"auroc": 0.68, "auprc": 0.83},
            "exact": {"auroc": 0.79, "auprc": 0.81},
        },
    ]

    def fake_save_checkpoint(
        path: Path,
        _model: object,
        *,
        config: object,
        epoch: int,
        optimizer: object,
        metrics: object,
        provenance: object,
    ) -> CheckpointReceipt:
        del config, optimizer, metrics, provenance
        path.write_bytes(b"checkpoint")
        token = "{:064d}".format(epoch)
        return CheckpointReceipt(
            path=str(path),
            file_sha256=token,
            config_hash="1" * 64,
            epoch=epoch,
            model_state_sha256="2" * 64,
            optimizer_state_sha256="3" * 64,
            canonical_content_sha256="4" * 64,
        )

    monkeypatch.setattr(pilot_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pilot_module, "GeometryArtifactCache", lambda _path: object())
    monkeypatch.setattr(pilot_module, "_DirectoryMaskLoader", lambda _path: object())
    monkeypatch.setattr(
        pilot_module,
        "build_exact_seam_pilot_population",
        lambda *_args, **_kwargs: population,
    )
    monkeypatch.setattr(pilot_module, "LocalQ1Backend", FakeBackend)
    monkeypatch.setattr(
        pilot_module,
        "_arm",
        lambda _backend, name: SimpleNamespace(
            pilot_name=("exact" if "exact_seam" in name.value else "weak"),
            model_config={"name": name.value},
        ),
    )
    monkeypatch.setattr(
        pilot_module, "_prepared_batch", lambda *_args, **_kwargs: object()
    )
    epoch_shuffle_seeds = []

    def fake_epoch_order(records, *, seed: int, epoch: int):
        del epoch
        epoch_shuffle_seeds.append(seed)
        return tuple(records)

    monkeypatch.setattr(pilot_module, "_epoch_order", fake_epoch_order)
    monkeypatch.setattr(
        pilot_module,
        "_validation_metrics",
        lambda **_kwargs: metric_rows.pop(0),
    )
    monkeypatch.setattr(pilot_module, "save_checkpoint", fake_save_checkpoint)

    result = pilot_module.run_exact_seam_pilot(
        ExactSeamPilotConfig(
            mask_root=tmp_path,
            output_root=tmp_path / "output",
            cache_root=tmp_path / "cache",
            max_pairs=500,
            epochs=5,
            seed=260830,
            initialization_seed=41,
            exact_loss_schedule=(0.25, 0.05, 0.0, 0.0, 0.0),
        )
    )

    assert sessions["weak"].weights == [0.0] * 5
    assert sessions["exact"].weights == [0.25, 0.05, 0.0, 0.0, 0.0]
    assert len(result["epoch_checkpoints"]["weak"]) == 5
    assert len(result["epoch_checkpoints"]["exact"]) == 5
    assert result["winner_selection"]["winners"]["weak"]["epoch"] == 2
    assert result["winner_selection"]["winners"]["exact"]["epoch"] == 3
    assert result["checkpoints"]["weak"]["epoch"] == 2
    assert result["checkpoints"]["exact"]["epoch"] == 3
    assert result["continuous_session_and_optimizer_per_arm"] is True
    assert session_seeds == [41, 41]
    assert epoch_shuffle_seeds == [41] * 5
    assert result["config"]["seed"] == 260830
    assert result["config"]["initialization_seed"] == 41
    assert result["config"]["initialization_seed_source"] == "explicit"
    assert all(
        Path(receipt["path"]).is_file()
        for rows in result["epoch_checkpoints"].values()
        for receipt in rows
    )
