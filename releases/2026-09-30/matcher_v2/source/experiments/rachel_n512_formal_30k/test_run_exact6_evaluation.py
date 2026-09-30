from __future__ import annotations

import os
from pathlib import Path

import pytest

from experiments.rachel_n512_formal_30k import run_exact6_evaluation as launcher


def test_missing_shredding_freeze_fails_before_output_or_sealed_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = Path(launcher.__file__).resolve().parents[2]
    sealed_root = tmp_path / "opaque-sealed"
    real_root = tmp_path / "opaque-real"
    output_root = tmp_path / "must-not-exist"
    config = launcher.ExactSixEvaluationConfig(
        source_root=source_root,
        n512_run_directory=tmp_path / "n512",
        matched_mm_run_directory=tmp_path / "matched",
        pairingnet_run_directory=tmp_path / "pairing",
        shreddingnet_freeze_path=tmp_path / "shredding" / "train_val_freeze.json",
        dataset_root=sealed_root,
        real_manifest_path=real_root / "control" / "real_test_manifest.json",
        real_local_receipt_path=real_root / "control" / "local_path_receipt.json",
        real_main_root=real_root / "main",
        real_supp_root=real_root / "supp",
        output_root=output_root,
        preflight_only=True,
    )

    calls = []

    def reject_missing_shredding(path: Path, *, device: str) -> None:
        calls.append((Path(path), device))
        raise launcher.benchmark_adapter.RachelBenchmarkEvalAdapterError(
            "ShreddingNet authority must be train_val_freeze.json"
        )

    def forbidden_downstream(*args, **kwargs):
        raise AssertionError("a downstream authority or evaluator was reached")

    monkeypatch.setattr(
        launcher.benchmark_adapter,
        "freeze_shreddingnet_benchmark",
        reject_missing_shredding,
    )
    monkeypatch.setattr(
        launcher.benchmark_adapter,
        "freeze_pairingnet_benchmark",
        forbidden_downstream,
    )
    monkeypatch.setattr(
        launcher.sealed, "_freeze_completed_winners", forbidden_downstream
    )
    monkeypatch.setattr(
        launcher, "freeze_matched_mm_winners", forbidden_downstream
    )
    monkeypatch.setattr(launcher, "_prepare_output_root", forbidden_downstream)
    monkeypatch.setattr(
        launcher.sealed, "run_sealed_synthetic_test", forbidden_downstream
    )
    monkeypatch.setattr(
        launcher.corrosion,
        "run_rachel_n512_corrosion_robustness",
        forbidden_downstream,
    )
    monkeypatch.setattr(
        launcher.real_external,
        "evaluate_strict_real_external",
        forbidden_downstream,
    )
    monkeypatch.setattr(
        launcher.real_translation,
        "evaluate_real_translation_gt_postprediction",
        forbidden_downstream,
    )

    forbidden_roots = tuple(
        os.path.abspath(str(path)) for path in (sealed_root, real_root)
    )

    def assert_not_sealed(path: object) -> None:
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        candidate = os.path.abspath(os.fsdecode(path))
        for forbidden in forbidden_roots:
            if os.path.commonpath((candidate, forbidden)) == forbidden:
                raise AssertionError("sealed or real path was accessed")

    original_open = Path.open
    original_iterdir = Path.iterdir
    original_listdir = os.listdir
    original_scandir = os.scandir

    def guarded_open(path: Path, *args, **kwargs):
        assert_not_sealed(path)
        return original_open(path, *args, **kwargs)

    def guarded_iterdir(path: Path):
        assert_not_sealed(path)
        return original_iterdir(path)

    def guarded_listdir(path: object = "."):
        assert_not_sealed(path)
        return original_listdir(path)

    def guarded_scandir(path: object = "."):
        assert_not_sealed(path)
        return original_scandir(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    monkeypatch.setattr(os, "listdir", guarded_listdir)
    monkeypatch.setattr(os, "scandir", guarded_scandir)

    with pytest.raises(
        launcher.ExactSixEvaluationGateError,
        match="ShreddingNet authority must be train_val_freeze.json",
    ):
        launcher.run_exact_six_evaluation(config)

    assert calls == [(config.shreddingnet_freeze_path, "cpu")]
    assert not output_root.exists()


def test_resume_state_accepts_canonical_json_key_order(tmp_path: Path) -> None:
    config_record = {
        "schema_version": launcher.SCHEMA_VERSION,
        "config": {"output_root": str(tmp_path)},
        "authority_content_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }
    config = launcher.ExactSixEvaluationConfig(
        source_root=Path(launcher.__file__).resolve().parents[2],
        n512_run_directory=tmp_path / "n512",
        matched_mm_run_directory=tmp_path / "matched",
        pairingnet_run_directory=tmp_path / "pairing",
        shreddingnet_freeze_path=tmp_path / "shredding" / "train_val_freeze.json",
        dataset_root=tmp_path / "sealed",
        real_manifest_path=tmp_path / "real-control" / "manifest.json",
        real_local_receipt_path=tmp_path / "real-control" / "receipt.json",
        real_main_root=tmp_path / "real-main",
        real_supp_root=tmp_path / "real-supp",
        output_root=tmp_path / "output",
        resume=True,
    )
    config.output_root.mkdir()
    launcher._write_json_new(
        config.output_root / "launcher_config.json", config_record
    )
    launcher._write_json_new(
        config.output_root / "launcher_state.json",
        launcher._initial_state("b" * 64),
    )

    _, state = launcher._prepare_output_root(config, config_record)

    assert set(state["stages"]) == set(launcher.STAGES)
