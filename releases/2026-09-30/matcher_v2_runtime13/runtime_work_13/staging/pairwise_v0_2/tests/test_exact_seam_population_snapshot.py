from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from staging.pairwise_v0_2.training import exact_seam_pilot as exact
from staging.pairwise_v0_2.training import exact_seam_population_snapshot as snapshot


def _write_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255, mode="L").save(path)


def _four_fragment_partition(group: Path) -> None:
    for index, (row_slice, column_slice) in enumerate(
        (
            (slice(0, 8), slice(0, 8)),
            (slice(0, 8), slice(8, 16)),
            (slice(8, 16), slice(0, 8)),
            (slice(8, 16), slice(8, 16)),
        )
    ):
        mask = np.zeros((16, 16), dtype=np.bool_)
        mask[row_slice, column_slice] = True
        _write_mask(group / (str(index) + ".png"), mask)


def _two_split_groups(mask_root: Path, *, seed: int) -> None:
    selected = {}
    for index in range(1000):
        relative = "gen4voronoi/no_erode/group-{:04d}".format(index)
        split = exact._split_for_group(relative, seed, 0.2)
        selected.setdefault(split, relative)
        if set(selected) == {"train", "val"}:
            break
    assert set(selected) == {"train", "val"}
    for relative in selected.values():
        _four_fragment_partition(mask_root / relative)


class _TinyPilotConfig(exact.ExactSeamPilotConfig):
    @property
    def train_pair_count(self) -> int:
        return 4

    @property
    def validation_pair_count(self) -> int:
        return 4


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    train_group = tmp_path / "gen4voronoi" / "no_erode" / "train"
    validation_group = tmp_path / "gen4voronoi" / "no_erode" / "val"
    unused_group = tmp_path / "gen4voronoi" / "no_erode" / "unused"
    for group in (train_group, validation_group, unused_group):
        _four_fragment_partition(group)
    groups = (train_group, validation_group, unused_group)
    monkeypatch.setattr(
        exact,
        "discover_exact_mask_groups",
        lambda *_args, **_kwargs: groups,
    )
    monkeypatch.setattr(
        exact,
        "_split_for_group",
        lambda relative, *_args: "val" if relative.endswith("/val") else "train",
    )
    config = _TinyPilotConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        max_pairs=500,
    )
    return config, groups


def _accept_all_collector(config, tasks, **_kwargs):
    values = []
    for task in tasks:
        records = exact._records_for_group(
            config.mask_root,
            config.mask_root / task.relative_group,
            split=task.split,
            seed=config.seed,
        )
        values.append(
            snapshot._GroupResult(
                index=task.index,
                relative_group=task.relative_group,
                split=task.split,
                pair_ids=tuple(record.pair_id for record in records),
                labels=tuple(record.label for record in records),
                eligible=tuple(True for _record in records),
            )
        )
    return tuple(values)


def test_freeze_replays_original_order_quota_and_writes_lean_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _groups = _fixture(tmp_path, monkeypatch)
    expected = exact.build_exact_seam_pilot_population(
        config, record_filter=lambda _record: True
    )
    monkeypatch.setattr(snapshot, "_collect_results", _accept_all_collector)
    output = tmp_path / "snapshot.json"

    payload = snapshot.freeze_exact_seam_population_snapshot(
        config,
        {
            "consumed_group_count": 2,
            "discovered_group_count": 3,
            "train_positive_count": 2,
            "train_negative_count": 2,
            "validation_positive_count": 2,
            "validation_negative_count": 2,
        },
        3,
        output,
    )

    assert payload["schema_version"] == snapshot.SNAPSHOT_SCHEMA_VERSION
    assert payload["pilot"] == {
        "seed": config.seed,
        "generator": config.generator,
        "max_pairs": config.max_pairs,
        "validation_fraction": config.validation_fraction,
        "consumed_group_count": 2,
    }
    assert payload["population"]["training_pair_ids"] == [
        record.pair_id for record in expected.training_records
    ]
    assert payload["population"]["validation_pair_ids"] == [
        record.pair_id for record in expected.validation_records
    ]
    assert "group_decisions" not in payload["population"]
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_load_lean_snapshot_reconstructs_order_without_geometry_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _groups = _fixture(tmp_path, monkeypatch)
    expected = exact.build_exact_seam_pilot_population(
        config, record_filter=lambda _record: True
    )
    payload = {
        "schema_version": snapshot.SNAPSHOT_SCHEMA_VERSION,
        # Production snapshots may carry these build-envelope fields.  The
        # population loader intentionally accepts them without conversion.
        "status": "complete",
        "workers": 8,
        "runtime_seconds": 123.0,
        "pilot": {
            "seed": config.seed,
            "generator": config.generator,
            "max_pairs": config.max_pairs,
            "validation_fraction": config.validation_fraction,
            "consumed_group_count": 2,
        },
        "population": {
            "training_pair_ids": [
                record.pair_id for record in expected.training_records
            ],
            "validation_pair_ids": [
                record.pair_id for record in expected.validation_records
            ],
            "train_pairs": 4,
            "validation_pairs": 4,
            "train_positive_count": 2,
            "train_negative_count": 2,
            "validation_positive_count": 2,
            "validation_negative_count": 2,
            "group_disjoint": True,
        },
    }
    path = tmp_path / "lean.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        exact,
        "exact_seam_group_eligibility",
        lambda *_args, **_kwargs: pytest.fail("eligibility must not run on load"),
    )

    observed = snapshot.load_exact_seam_population_snapshot(path, config)

    assert [record.pair_id for record in observed.training_records] == [
        record.pair_id for record in expected.training_records
    ]
    assert [record.pair_id for record in observed.validation_records] == [
        record.pair_id for record in expected.validation_records
    ]
    assert observed.discovered_group_count == 3
    assert observed.consumed_group_count == 2


def test_runtime_first_use_writes_snapshot_and_reuse_skips_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _groups = _fixture(tmp_path, monkeypatch)
    expected = exact.build_exact_seam_pilot_population(
        config, record_filter=lambda _record: True
    )
    path = tmp_path / "direct.json"

    replay_config = _TinyPilotConfig(
        mask_root=config.mask_root,
        output_root=config.output_root,
        cache_root=config.cache_root,
        max_pairs=config.max_pairs,
        population_snapshot=path,
    )
    build_calls = []

    def fake_build(config_value, *, group_record_filter):
        build_calls.append((config_value, group_record_filter))
        return expected

    monkeypatch.setattr(exact, "build_exact_seam_pilot_population", fake_build)

    first = exact.build_or_load_exact_seam_pilot_population(
        replay_config,
        group_record_filter=lambda rows: tuple(True for _ in rows),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    def fail_eligibility(_records):
        pytest.fail("eligibility must not run when snapshot already exists")

    observed = exact.build_or_load_exact_seam_pilot_population(
        replay_config,
        group_record_filter=fail_eligibility,
    )

    assert payload["population"]["train_pairs"] == 4
    assert payload["population"]["validation_pairs"] == 4
    assert first == expected
    assert observed == expected
    assert len(build_calls) == 1


def test_population_workers_do_not_change_the_legacy_no_snapshot_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _groups = _fixture(tmp_path, monkeypatch)
    expected = exact.build_exact_seam_pilot_population(
        config, record_filter=lambda _record: True
    )
    no_snapshot = _TinyPilotConfig(
        mask_root=config.mask_root,
        output_root=config.output_root,
        cache_root=config.cache_root,
        max_pairs=config.max_pairs,
        population_workers=4,
    )
    calls = []

    def fake_build(config_value, *, group_record_filter):
        calls.append((config_value, group_record_filter))
        return expected

    monkeypatch.setattr(exact, "build_exact_seam_pilot_population", fake_build)
    monkeypatch.setattr(
        snapshot,
        "ProcessPoolExecutor",
        lambda *_args, **_kwargs: pytest.fail(
            "the legacy no-snapshot path must not create a worker pool"
        ),
    )

    observed = exact.build_or_load_exact_seam_pilot_population(
        no_snapshot,
        group_record_filter=lambda rows: tuple(True for _ in rows),
    )

    assert observed is expected
    assert len(calls) == 1


def test_first_snapshot_workers_1_2_4_are_byte_and_population_identical(
    tmp_path: Path,
) -> None:
    mask_root = tmp_path / "masks"
    _two_split_groups(mask_root, seed=260830)
    cache_root = tmp_path / "cache"
    snapshots = []
    populations = []
    for workers in (1, 2, 4):
        path = tmp_path / "population-workers-{}.json".format(workers)
        config = _TinyPilotConfig(
            mask_root=mask_root,
            output_root=tmp_path / "output-{}".format(workers),
            cache_root=cache_root,
            max_pairs=500,
            population_snapshot=path,
            population_workers=workers,
        )
        loader = exact._DirectoryMaskLoader(mask_root)
        cache = exact.GeometryArtifactCache(cache_root)

        def eligibility(records):
            return exact.exact_seam_group_eligibility(
                records,
                loader=loader,
                cache=cache,
                geometry_config=exact.GeometryBatchConfig(),
            )

        populations.append(
            exact.build_or_load_exact_seam_pilot_population(
                config,
                group_record_filter=eligibility,
            )
        )
        snapshots.append(path.read_bytes())

    expected_train = [record.pair_id for record in populations[0].training_records]
    expected_validation = [
        record.pair_id for record in populations[0].validation_records
    ]
    assert snapshots[0] == snapshots[1] == snapshots[2]
    for population in populations[1:]:
        assert population == populations[0]
        assert [record.pair_id for record in population.training_records] == (
            expected_train
        )
        assert [record.pair_id for record in population.validation_records] == (
            expected_validation
        )
        assert population.consumed_group_count == populations[0].consumed_group_count


def test_parallel_first_snapshot_wraps_worker_failure(tmp_path: Path) -> None:
    mask_root = tmp_path / "masks"
    group = mask_root / "gen4voronoi" / "no_erode" / "bad"
    _four_fragment_partition(group)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(group / "0.png")
    config = _TinyPilotConfig(
        mask_root=mask_root,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        max_pairs=500,
        population_snapshot=tmp_path / "population.json",
        population_workers=2,
    )

    with pytest.raises(
        snapshot.ExactSeamPopulationSnapshotError,
        match="parallel eligibility worker failed for group",
    ) as captured:
        snapshot.build_exact_seam_pilot_population_parallel(config, workers=2)

    assert isinstance(captured.value.__cause__, exact.ExactSeamPilotError)


def test_python_module_class_identity_can_write_and_load_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _groups = _fixture(tmp_path, monkeypatch)
    expected = exact.build_exact_seam_pilot_population(
        config, record_filter=lambda _record: True
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        alternate = runpy.run_module(
            exact.__name__,
            run_name="__exact_seam_pilot_module_smoke__",
            alter_sys=True,
        )
    alternate_config_type = type(
        "ExactSeamPilotConfig",
        (alternate["ExactSeamPilotConfig"],),
        {
            "train_pair_count": property(lambda _self: 4),
            "validation_pair_count": property(lambda _self: 4),
        },
    )
    alternate_config = alternate_config_type(
        mask_root=config.mask_root,
        output_root=config.output_root,
        cache_root=config.cache_root,
        max_pairs=config.max_pairs,
        population_snapshot=tmp_path / "alternate.json",
        population_workers=4,
    )
    alternate_population = alternate["ExactSeamPilotPopulation"](
        training_records=expected.training_records,
        validation_records=expected.validation_records,
        discovered_group_count=expected.discovered_group_count,
        consumed_group_count=expected.consumed_group_count,
    )

    parallel_config = alternate_config_type(
        mask_root=config.mask_root,
        output_root=config.output_root,
        cache_root=config.cache_root,
        max_pairs=config.max_pairs,
        population_snapshot=tmp_path / "alternate-parallel.json",
        population_workers=2,
    )
    parallel = alternate["build_or_load_exact_seam_pilot_population"](
        parallel_config,
        group_record_filter=lambda _rows: pytest.fail(
            "parallel first-build must run concrete worker eligibility"
        ),
    )
    assert [record.pair_id for record in parallel.training_records] == [
        record.pair_id for record in expected.training_records
    ]
    assert [record.pair_id for record in parallel.validation_records] == [
        record.pair_id for record in expected.validation_records
    ]

    snapshot.write_exact_seam_population_snapshot(
        alternate_config,
        alternate_population,
        alternate_config.population_snapshot,
    )
    monkeypatch.setattr(
        snapshot,
        "ProcessPoolExecutor",
        lambda *_args, **_kwargs: pytest.fail(
            "snapshot replay must not create a worker pool"
        ),
    )
    observed = alternate["build_or_load_exact_seam_pilot_population"](
        alternate_config,
        group_record_filter=lambda _rows: pytest.fail(
            "module-mode replay must not call eligibility"
        ),
    )

    assert [record.pair_id for record in observed.training_records] == [
        record.pair_id for record in expected.training_records
    ]
    assert [record.pair_id for record in observed.validation_records] == [
        record.pair_id for record in expected.validation_records
    ]


def test_population_snapshot_true_python_m_validate_smoke(tmp_path: Path) -> None:
    mask_root = tmp_path / "masks"
    for index in range(200):
        _four_fragment_partition(mask_root / "gen4voronoi" / "no_erode" / str(index))
    config = exact.ExactSeamPilotConfig(
        mask_root=mask_root,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        max_pairs=500,
    )
    population = exact.build_exact_seam_pilot_population(
        config,
        record_filter=lambda _record: True,
    )
    path = tmp_path / "subprocess.json"
    snapshot.write_exact_seam_population_snapshot(config, population, path)
    workspace = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(workspace), environment.get("PYTHONPATH", "")) if value
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "staging.pairwise_v0_2.training.exact_seam_population_snapshot",
            "validate",
            "--mask-root",
            str(mask_root),
            "--max-pairs",
            "500",
            "--validation-fraction",
            "0.2",
            "--seed",
            "260830",
            "--generator",
            "gen4voronoi",
            "--snapshot",
            str(path),
        ],
        cwd=workspace,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert "loaded train=400 validation=100" in completed.stdout


def test_freeze_rejects_a_summary_consumed_count_after_actual_quota_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _groups = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(snapshot, "_collect_results", _accept_all_collector)

    with pytest.raises(
        snapshot.ExactSeamPopulationSnapshotError,
        match="replayed consumed_group_count 2 != summary 3",
    ):
        snapshot.freeze_exact_seam_population_snapshot(
            config,
            {"consumed_group_count": 3},
            2,
            tmp_path / "bad.json",
        )


def test_load_rejects_config_identity_and_pair_id_outside_consumed_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, groups = _fixture(tmp_path, monkeypatch)
    expected = exact.build_exact_seam_pilot_population(
        config, record_filter=lambda _record: True
    )
    train_ids = [record.pair_id for record in expected.training_records]
    validation_ids = [record.pair_id for record in expected.validation_records]
    later_records = exact._records_for_group(
        config.mask_root, groups[2], split="train", seed=config.seed
    )
    train_ids[0] = later_records[0].pair_id
    payload = {
        "schema_version": snapshot.SNAPSHOT_SCHEMA_VERSION,
        "pilot": {
            "seed": config.seed,
            "generator": config.generator,
            "max_pairs": config.max_pairs,
            "validation_fraction": config.validation_fraction,
            "consumed_group_count": 2,
        },
        "population": {
            "training_pair_ids": train_ids,
            "validation_pair_ids": validation_ids,
            "train_pairs": 4,
            "validation_pairs": 4,
            "train_positive_count": 2,
            "train_negative_count": 2,
            "validation_positive_count": 2,
            "validation_negative_count": 2,
            "group_disjoint": True,
        },
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        snapshot.ExactSeamPopulationSnapshotError,
        match="not found in the first consumed groups",
    ):
        snapshot.load_exact_seam_population_snapshot(path, config)

    payload["pilot"]["seed"] = config.seed + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        snapshot.ExactSeamPopulationSnapshotError,
        match="pilot seed does not match config",
    ):
        snapshot.load_exact_seam_population_snapshot(path, config)


def test_scale_6964_config_has_expected_snapshot_counts(tmp_path: Path) -> None:
    config = exact.ExactSeamPilotConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        max_pairs=6964,
    )

    assert config.train_pair_count == 5570
    assert config.validation_pair_count == 1394
    snapshot._validate_summary_counts(
        {
            "train_pairs": 5570,
            "validation_pairs": 1394,
            "train_positive_count": 2785,
            "train_negative_count": 2785,
            "validation_positive_count": 697,
            "validation_negative_count": 697,
        },
        config,
    )


def test_scale_30000_config_has_expected_snapshot_counts(tmp_path: Path) -> None:
    config = exact.ExactSeamPilotConfig(
        mask_root=tmp_path,
        output_root=tmp_path / "output",
        cache_root=tmp_path / "cache",
        max_pairs=30000,
    )

    assert config.train_pair_count == 24000
    assert config.validation_pair_count == 6000
    snapshot._validate_summary_counts(
        {
            "train_pairs": 24000,
            "validation_pairs": 6000,
            "train_positive_count": 12000,
            "train_negative_count": 12000,
            "validation_positive_count": 3000,
            "validation_negative_count": 3000,
        },
        config,
    )
