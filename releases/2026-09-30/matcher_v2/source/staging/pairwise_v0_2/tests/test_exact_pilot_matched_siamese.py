from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from staging.pairwise_v0_2.baselines import exact_pilot_matched_siamese as control
from staging.pairwise_v0_2.baselines.matched_route_a_siamese import (
    MatchedSiameseContract,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training import exact_seam_pilot
from staging.pairwise_v0_2.training.exact_seam_pilot import (
    ExactSeamPilotPopulation,
)
from staging.pairwise_v0_2.training.short_ablation import (
    record_sequence_fingerprint,
)


_BINDING = ArchiveBinding(
    logical_id="canonical://fixture/exact-pilot-siamese",
    archive_format="zip",
    sha256="1" * 64,
)


def _record(index: int, *, split: str, label: bool) -> TrainingPairRecord:
    group = "{}-group-{}".format(split, index)

    def reference(side: str) -> MaskMemberRef:
        return MaskMemberRef(
            binding=_BINDING,
            archive_member="fixture/{}/{}-{}.png".format(split, index, side),
            fragment_id="{}-{}-{}".format(split, index, side),
            dataset_id="shredding_pipeline_exact_aligned_masks",
            canonical_group_id=group,
            component_id=group,
            split=split,
            threshold_rule="grayscale_uint8_gt_127",
        )

    first = reference("a")
    second = reference("b")
    return TrainingPairRecord(
        fragment_a=first,
        fragment_b=second,
        label=label,
        direction_b_wrt_a="right" if label else None,
        dataset_id=first.dataset_id,
        canonical_group_id=group,
        component_id=group,
        split=split,
        canonical_pair_key=tuple(sorted((first.fragment_id, second.fragment_id))),
        label_origin="fixture",
    )


def _population() -> ExactSeamPilotPopulation:
    return ExactSeamPilotPopulation(
        training_records=tuple(
            _record(index, split="train", label=index % 2 == 0) for index in range(4)
        ),
        validation_records=(
            _record(100, split="val", label=True),
            _record(101, split="val", label=False),
        ),
        discovered_group_count=6,
        consumed_group_count=6,
    )


def _mask_loader_factory(records):
    masks = {}
    for record_index, record in enumerate(records):
        for side_index, reference in enumerate((record.fragment_a, record.fragment_b)):
            mask = np.zeros((18, 22), dtype=np.bool_)
            height = 4 + ((record_index + side_index) % 10)
            width = 5 + ((2 * record_index + side_index) % 12)
            mask[1 : 1 + height, 2 : 2 + width] = True
            masks[reference.archive_member] = mask

    class Loader:
        def __call__(self, reference):
            return masks[reference.archive_member]

        def close(self):
            return None

    return Loader


class _ShapeCheckingSiamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_a, input_b):
        assert input_a.ndim == 4 and tuple(input_a.shape[1:]) == (1, 64, 64)
        assert input_b.ndim == 4 and tuple(input_b.shape[1:]) == (1, 64, 64)
        feature = input_a.mean((1, 2, 3)) + input_b.mean((1, 2, 3))
        return torch.sigmoid(self.weight * feature + self.bias)[:, None]


def test_defaults_are_the_same_1000_pair_exact_pilot_contract(tmp_path: Path):
    config = control.ExactPilotMatchedSiameseConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
    )

    assert config.max_pairs == 1000
    assert config.train_pair_count == 800
    assert config.validation_pair_count == 200
    assert config.epochs == 3
    assert config.batch_size == 8
    assert config.seed == 260830
    assert config.generator == "gen4voronoi"
    assert config.population_config.train_pair_count == 800
    assert config.initialization_seed is None
    assert config.resolved_initialization_seed == config.seed
    assert config.initialization_seed_source == "legacy_seed_fallback"
    assert config.population_workers == 1
    assert config.population_config.population_workers == 1
    assert config.training_contract.seed == config.seed
    assert config.training_contract.train_batch_size == 8
    assert config.training_contract.eval_batch_size == 8


def test_matched_siamese_accepts_balanced_6964_pair_scale_and_cli(
    tmp_path: Path,
):
    config = control.ExactPilotMatchedSiameseConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        max_pairs=6964,
    )

    assert config.max_pairs == 6964
    assert config.train_pair_count == 5570
    assert config.validation_pair_count == 1394
    assert config.population_config.train_pair_count // 2 == 2785
    assert config.population_config.validation_pair_count // 2 == 697

    parsed = control._parser().parse_args(
        [
            "--mask-root",
            str(tmp_path / "masks"),
            "--output-root",
            str(tmp_path / "cli-output"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--max-pairs",
            "6964",
            "--population-workers",
            "4",
        ]
    )
    assert parsed.max_pairs == 6964
    assert parsed.population_workers == 4


def test_matched_siamese_accepts_30000_scale_and_shared_snapshot_cli(
    tmp_path: Path,
):
    snapshot_path = tmp_path / "population-30000.json"
    config = control.ExactPilotMatchedSiameseConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        max_pairs=30000,
        population_snapshot=snapshot_path,
        population_workers=8,
    )

    assert config.train_pair_count == 24000
    assert config.validation_pair_count == 6000
    assert config.population_config.train_pair_count // 2 == 12000
    assert config.population_config.validation_pair_count // 2 == 3000
    assert config.population_config.population_snapshot == snapshot_path
    assert config.population_config.population_workers == 8

    parsed = control._parser().parse_args(
        [
            "--mask-root",
            str(tmp_path / "masks"),
            "--output-root",
            str(tmp_path / "cli-output"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--max-pairs",
            "30000",
            "--population-snapshot",
            str(snapshot_path),
            "--population-workers",
            "8",
        ]
    )
    assert parsed.max_pairs == 30000
    assert parsed.population_snapshot == snapshot_path
    assert parsed.population_workers == 8


def test_matched_siamese_separates_population_and_initialization_seed(
    tmp_path: Path,
):
    config = control.ExactPilotMatchedSiameseConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        seed=260830,
        initialization_seed=260831,
    )

    assert config.population_config.seed == 260830
    assert config.training_contract.seed == 260831
    assert config.resolved_initialization_seed == 260831
    assert config.initialization_seed_source == "explicit"

    parsed = control._parser().parse_args(
        [
            "--mask-root",
            str(tmp_path / "masks"),
            "--output-root",
            str(tmp_path / "cli-output"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--seed",
            "260830",
            "--initialization-seed",
            "260831",
        ]
    )
    assert parsed.seed == 260830
    assert parsed.initialization_seed == 260831

    with pytest.raises(ValueError, match="initialization_seed"):
        control.ExactPilotMatchedSiameseConfig(
            mask_root=tmp_path,
            output_root=tmp_path / "bad-output",
            cache_root=tmp_path / "cache",
            initialization_seed=-1,
        )


def test_population_reuses_exact_builder_and_group_geometry_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    population = _population()
    captured = {}

    class FakeLoader:
        def __init__(self, root):
            captured["loader_root"] = Path(root)

    def fake_builder(config, *, group_record_filter):
        captured["config"] = config
        captured["group_record_filter"] = group_record_filter
        return population

    monkeypatch.setattr(control, "GeometryArtifactCache", lambda root: ("cache", root))
    monkeypatch.setattr(exact_seam_pilot, "_DirectoryMaskLoader", FakeLoader)
    monkeypatch.setattr(
        exact_seam_pilot,
        "exact_seam_group_eligibility",
        lambda records, **kwargs: tuple(True for _ in records),
    )
    monkeypatch.setattr(
        exact_seam_pilot, "build_exact_seam_pilot_population", fake_builder
    )
    config = control.ExactPilotMatchedSiameseConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
    )

    observed = control.build_exact_pilot_matched_siamese_population(config)

    assert observed is population
    assert captured["config"].max_pairs == 1000
    assert captured["config"].seed == 260830
    assert captured["config"].generator == "gen4voronoi"
    records = population.training_records[:2]
    assert captured["group_record_filter"](records) == (True, True)
    assert captured["loader_root"] == tmp_path


def test_training_uses_exact_epoch_order_once_and_whole_64_square_masks(
    tmp_path: Path,
):
    population = _population()
    records = population.training_records + population.validation_records
    contract = MatchedSiameseContract(
        epochs=2,
        seed=17,
        train_batch_size=2,
        eval_batch_size=2,
        preprocess_workers=1,
    )

    receipt = control.run_exact_population_matched_siamese_training(
        population=population,
        loader_factory=_mask_loader_factory(records),
        output_root=tmp_path / "run",
        contract=contract,
        device=torch.device("cpu"),
        model_factory=_ShapeCheckingSiamese,
        population_seed=260830,
        initialization_seed_source="explicit",
    )

    expected_fingerprints = [
        record_sequence_fingerprint(
            exact_seam_pilot._epoch_order(
                population.training_records,
                seed=contract.seed,
                epoch=epoch,
            )
        )
        for epoch in (1, 2)
    ]
    assert receipt["status"] == "complete"
    assert receipt["population"]["train_pair_count"] == 4
    assert receipt["population"]["validation_pair_count"] == 2
    assert [row["train"]["sample_count"] for row in receipt["epochs"]] == [4, 4]
    assert [
        row["train"]["pair_sequence_sha256"] for row in receipt["epochs"]
    ] == expected_fingerprints
    assert receipt["fairness"]["train_pair_presentations_per_epoch"] == 1
    assert receipt["fairness"]["same_train_pair_order_as_exact_seam_pilot"] is True
    assert receipt["seeds"] == {
        "population_seed": 260830,
        "initialization_seed": 17,
        "initialization_seed_source": "explicit",
    }
    assert receipt["winner"]["epoch"] in {1, 2}
    assert (tmp_path / "run" / "exact_pilot_matched_siamese_winner.pt").is_file()
    assert (tmp_path / "run" / "exact_pilot_matched_siamese_summary.json").is_file()
    assert (
        tmp_path / "run" / "exact_pilot_matched_siamese_validation_scores.json"
    ).is_file()


def test_cli_exposes_only_direct_experiment_inputs(tmp_path: Path):
    args = control._parser().parse_args(
        [
            "--mask-root",
            str(tmp_path / "masks"),
            "--output-root",
            str(tmp_path / "output"),
            "--cache-root",
            str(tmp_path / "cache"),
        ]
    )

    assert args.max_pairs == 1000
    assert args.epochs == 3
    assert args.batch_size == 8
    assert args.seed == 260830
    assert args.initialization_seed is None
    assert args.generator == "gen4voronoi"
    assert args.device == "cuda"
