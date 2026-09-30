from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from staging.pairwise_v0_2.preflight import build_c0_run_plan as builder
from staging.pairwise_v0_2.training import c0_runner as runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "staging"
    for package in ("pairwise_v0_1", "pairwise_v0_2"):
        (root / package / "runtime").mkdir(parents=True)
        (root / package / "runtime/dependency.py").write_text(
            "VALUE = {!r}\n".format(package), encoding="utf-8"
        )
    runner_path = root / "pairwise_v0_2/training/c0_runner.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text(
        'PRODUCTION_RUN_PLAN_FILE_SHA256 = "{}"\n'
        'PRODUCTION_RUN_PLAN_CONTENT_SHA256 = "{}"\n'.format("1" * 64, "2" * 64),
        encoding="utf-8",
    )
    return root


def _write_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[runner.ProductionRunPlanBinding, Path]:
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}\n", encoding="utf-8")
    bundle = _source_bundle(tmp_path)
    monkeypatch.setattr(
        runner,
        "__file__",
        str(bundle / "pairwise_v0_2/training/c0_runner.py"),
    )
    specs = MappingProxyType(
        {
            "freeze_receipt": runner._role("artifact://fixture/freeze", "metadata"),
            "source_bundle": runner._role(
                "source://fixture/runtime",
                "source_bundle",
                runner.SOURCE_BUNDLE_HASH_MODE,
            ),
        }
    )
    monkeypatch.setattr(runner, "PRODUCTION_RUN_PLAN_ROLE_SPECS", specs)
    manifest = runner.c0_source_bundle_manifest(bundle)
    roles = [
        {
            "role": "freeze_receipt",
            "logical_id": "artifact://fixture/freeze",
            "kind": "metadata",
            "hash_mode": "sha256_bytes",
            "bytes": freeze.stat().st_size,
            "sha256": _sha256(freeze),
        },
        {
            "role": "source_bundle",
            "logical_id": "source://fixture/runtime",
            "kind": "source_bundle",
            "hash_mode": runner.SOURCE_BUNDLE_HASH_MODE,
            "bytes": manifest["total_bytes"],
            "sha256": manifest["manifest_sha256"],
        },
    ]
    plan = {
        "schema_version": runner.RUN_PLAN_SCHEMA_VERSION,
        "status": "frozen_no_execution",
        "contract": dict(runner.C0RunnerContract().portable_dict()),
        "freeze_content_sha256": "3" * 64,
        "identity_index": {
            "schema_version": runner.IDENTITY_INDEX_CONTENT_SCHEMA,
            "member_count": 7,
            "content_sha256": "4" * 64,
        },
        "roles": roles,
        "source_bundle": manifest,
        "scope": {
            "archive_bytes_hashed": True,
            "archive_members_opened": False,
            "historical_test_access": dict(runner.HISTORICAL_TEST_ACCESS_EVIDENCE),
            "mask_pixels_decoded": False,
            "model_or_backend_created": False,
            "sealed_real_read": False,
        },
    }
    plan["content_sha256"] = runner._content_sha256(plan)
    plan_path = tmp_path / "c0_run_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(runner, "PRODUCTION_RUN_PLAN_FILE_SHA256", _sha256(plan_path))
    monkeypatch.setattr(
        runner,
        "PRODUCTION_RUN_PLAN_CONTENT_SHA256",
        plan["content_sha256"],
    )
    return (
        runner.ProductionRunPlanBinding(
            plan_path=plan_path,
            role_paths={"freeze_receipt": freeze, "source_bundle": bundle},
        ),
        bundle,
    )


def test_exact_source_bundle_transitive_tamper_fails_before_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, bundle = _write_plan(tmp_path, monkeypatch)
    runner.verify_c0_production_plan(binding)
    monkeypatch.setattr(runner, "PRODUCTION_FREEZE_FILE_SHA256", "5" * 64)
    monkeypatch.setattr(runner, "PRODUCTION_FREEZE_CONTENT_SHA256", "6" * 64)
    transitive = bundle / "pairwise_v0_2/runtime/dependency.py"
    transitive.write_text("VALUE = 'tampered'\n", encoding="utf-8")
    events: list[str] = []
    with pytest.raises(
        runner.C0RunnerError, match="source bundle|source_bundle|role SHA"
    ):
        runner.run_c0_n_q1(
            run_plan=runner.C0RunnerContract(),
            freeze_receipt=runner.FrozenReceiptLock(
                path=binding.role_paths["freeze_receipt"],
                file_sha256="5" * 64,
                content_sha256="6" * 64,
            ),
            selected_train_records=(),
            validation_records=(),
            provider_factory=lambda: events.append("provider"),
            backend_factory=lambda: events.append("backend"),
            output_dir=tmp_path / "output",
            production_plan=binding,
        )
    assert events == []


def test_source_bundle_must_be_the_executing_runner_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, _bundle = _write_plan(tmp_path, monkeypatch)
    other = tmp_path / "other-staging"
    other.mkdir()
    role_paths = dict(binding.role_paths)
    role_paths["source_bundle"] = other
    wrong = runner.ProductionRunPlanBinding(
        plan_path=binding.plan_path,
        role_paths=role_paths,
    )
    with pytest.raises(runner.C0RunnerError, match="executing runner tree"):
        runner.verify_c0_production_plan(wrong)


def test_builder_is_byte_deterministic_and_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}\n", encoding="utf-8")
    bundle = _source_bundle(tmp_path)
    monkeypatch.setattr(
        builder,
        "_validate_freeze",
        lambda *_args: {"content_sha256": "3" * 64},
    )

    class _Index:
        content_sha256 = "4" * 64
        identity_count = 7

    monkeypatch.setattr(
        builder.HistoricalIdentityIndex,
        "from_files",
        classmethod(lambda _cls, **_kwargs: _Index()),
    )
    paths = {
        "freeze_receipt": freeze,
        "source_bundle": bundle,
    }
    for role in (
        "mm_archive",
        "eccv_archive",
        "mm_fingerprint_cache",
        "eccv_fingerprint_cache",
        "historical_split",
    ):
        path = tmp_path / (role + ".bin")
        path.write_bytes(role.encode("utf-8"))
        paths[role] = path
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    builder.build_c0_run_plan(role_paths=paths, output=first)
    builder.build_c0_run_plan(role_paths=paths, output=second)
    assert first.read_bytes() == second.read_bytes()
    text = first.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    payload = json.loads(text)
    assert payload["scope"] == {
        "archive_bytes_hashed": True,
        "archive_members_opened": False,
        "historical_test_access": dict(runner.HISTORICAL_TEST_ACCESS_EVIDENCE),
        "mask_pixels_decoded": False,
        "model_or_backend_created": False,
        "sealed_real_read": False,
    }


def test_identity_lock_shape_fails_before_both_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, _bundle = _write_plan(tmp_path, monkeypatch)
    payload = json.loads(binding.plan_path.read_text(encoding="utf-8"))
    payload["identity_index"] = {"content_sha256": "4" * 64}
    payload.pop("content_sha256")
    payload["content_sha256"] = runner._content_sha256(payload)
    binding.plan_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        runner, "PRODUCTION_RUN_PLAN_FILE_SHA256", _sha256(binding.plan_path)
    )
    monkeypatch.setattr(
        runner,
        "PRODUCTION_RUN_PLAN_CONTENT_SHA256",
        payload["content_sha256"],
    )
    monkeypatch.setattr(runner, "PRODUCTION_FREEZE_FILE_SHA256", "5" * 64)
    monkeypatch.setattr(runner, "PRODUCTION_FREEZE_CONTENT_SHA256", "6" * 64)
    events: list[str] = []
    with pytest.raises(runner.C0RunnerError, match="identity-index lock schema"):
        runner.run_c0_n_q1(
            run_plan=runner.C0RunnerContract(),
            freeze_receipt=runner.FrozenReceiptLock(
                path=binding.role_paths["freeze_receipt"],
                file_sha256="5" * 64,
                content_sha256="6" * 64,
            ),
            selected_train_records=(),
            validation_records=(),
            provider_factory=lambda: events.append("provider"),
            backend_factory=lambda: events.append("backend"),
            output_dir=tmp_path / "output",
            production_plan=binding,
        )
    assert events == []


def test_source_bundle_rejects_python_symlink(tmp_path: Path) -> None:
    bundle = _source_bundle(tmp_path)
    target = bundle / "pairwise_v0_1/runtime/dependency.py"
    link = bundle / "pairwise_v0_1/runtime/linked.py"
    link.symlink_to(target)
    with pytest.raises(runner.C0RunnerError, match="symlink"):
        runner.c0_source_bundle_manifest(bundle)


def test_every_runtime_role_rejects_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, _bundle = _write_plan(tmp_path, monkeypatch)
    link = tmp_path / "freeze-link.json"
    link.symlink_to(binding.role_paths["freeze_receipt"])
    role_paths = dict(binding.role_paths)
    role_paths["freeze_receipt"] = link
    linked = runner.ProductionRunPlanBinding(
        plan_path=binding.plan_path,
        role_paths=role_paths,
    )
    with pytest.raises(runner.C0RunnerError, match="cannot be a symlink"):
        runner.verify_c0_production_plan(linked)


def test_plan_builder_rejects_any_role_symlink(tmp_path: Path) -> None:
    paths = {}
    for role in runner.PRODUCTION_RUN_PLAN_ROLE_SPECS:
        path = tmp_path / role
        if role == "source_bundle":
            path.mkdir()
        else:
            path.write_bytes(role.encode("utf-8"))
        paths[role] = path
    target = paths["mm_archive"]
    link = tmp_path / "mm-archive-link"
    link.symlink_to(target)
    paths["mm_archive"] = link
    with pytest.raises(runner.C0RunnerError, match="cannot be a symlink"):
        builder.build_c0_run_plan(role_paths=paths, output=tmp_path / "plan.json")
