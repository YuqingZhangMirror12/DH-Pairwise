from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from staging.pairwise_v0_2.preflight import build_local_q1_run_plan as builder
from staging.pairwise_v0_2.training import local_q1_cache_builder as cache_builder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "staging"
    first = root / "pairwise_v0_1/runtime/dependency.py"
    first.parent.mkdir(parents=True)
    first.write_text("VALUE = 'v0.1'\n", encoding="utf-8")
    second = root / "pairwise_v0_2/runtime/dependency.py"
    second.parent.mkdir(parents=True)
    second.write_text("VALUE = 'v0.2'\n", encoding="utf-8")
    cache_source = root / "pairwise_v0_2/training/local_q1_cache_builder.py"
    cache_source.parent.mkdir(parents=True)
    cache_source.write_text(
        'PRODUCTION_LOCAL_Q1_RUN_PLAN_FILE_SHA256 = ("{}")\n'
        'PRODUCTION_LOCAL_Q1_RUN_PLAN_CONTENT_SHA256 = ("{}")\n'.format(
            "1" * 64, "2" * 64
        ),
        encoding="utf-8",
    )
    return root


def _role_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for index, role in enumerate(cache_builder.LOCAL_Q1_RUN_PLAN_ROLE_SPECS):
        path = tmp_path / "roles" / (role + ".bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((role + ":" + str(index)).encode("utf-8"))
        paths[role] = path
    return paths


def _write_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[cache_builder.LocalQ1ProductionBinding, Path, dict[str, Path]]:
    source_bundle = _source_bundle(tmp_path)
    role_paths = _role_paths(tmp_path)
    monkeypatch.setattr(
        builder,
        "_validate_freeze",
        lambda *_args: {
            "content_sha256": cache_builder.PRODUCTION_LOCAL_Q1_FREEZE_CONTENT_SHA256,
            "locks": {
                "historical_identity_index": {
                    "member_count": (
                        cache_builder.PRODUCTION_LOCAL_Q1_IDENTITY_MEMBER_COUNT
                    ),
                    "content_sha256": (
                        cache_builder.PRODUCTION_LOCAL_Q1_IDENTITY_CONTENT_SHA256
                    ),
                }
            },
        },
    )
    plan_path = tmp_path / "local_q1_run_plan.json"
    plan = builder.build_local_q1_run_plan(
        role_paths=role_paths,
        source_bundle=source_bundle,
        output=plan_path,
    )
    monkeypatch.setattr(
        cache_builder,
        "PRODUCTION_LOCAL_Q1_RUN_PLAN_FILE_SHA256",
        _sha256(plan_path),
    )
    monkeypatch.setattr(
        cache_builder,
        "PRODUCTION_LOCAL_Q1_RUN_PLAN_CONTENT_SHA256",
        plan["content_sha256"],
    )
    monkeypatch.setattr(
        cache_builder,
        "__file__",
        str(source_bundle / "pairwise_v0_2/training/local_q1_cache_builder.py"),
    )
    return (
        cache_builder.LocalQ1ProductionBinding(
            plan_path=plan_path,
            source_bundle=source_bundle,
            role_paths=role_paths,
        ),
        source_bundle,
        role_paths,
    )


def test_builder_is_byte_deterministic_path_free_and_exact_eight_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, source_bundle, role_paths = _write_plan(tmp_path, monkeypatch)
    first = binding.plan_path.read_bytes()
    second_path = tmp_path / "second.json"
    builder.build_local_q1_run_plan(
        role_paths=role_paths,
        source_bundle=source_bundle,
        output=second_path,
    )
    assert first == second_path.read_bytes()
    assert str(tmp_path) not in first.decode("utf-8")

    plan, locks, authority = cache_builder.verify_local_q1_production_plan(binding)
    assert len(plan["roles"]) == len(locks) == 8
    assert set(locks) == set(cache_builder.LOCAL_Q1_RUN_PLAN_ROLE_SPECS)
    assert authority.run_plan_file_sha256 == _sha256(binding.plan_path)
    assert authority.run_plan_content_sha256 == plan["content_sha256"]
    assert (
        authority.source_bundle_manifest_sha256
        == plan["source_bundle"]["manifest_sha256"]
    )
    assert plan["scope"] == {
        "archive_bytes_hashed": True,
        "archive_members_opened": False,
        "historical_test_access": {
            "archive_members_opened": False,
            "assignment_or_pair_records_parsed": False,
            "mask_pixels_decoded": False,
            "pair_stream_read": False,
            "split_container_bytes_hashed_without_parsing": True,
        },
        "mask_pixels_decoded": False,
        "model_or_backend_created": False,
        "sealed_real_read": False,
    }


def test_transitive_source_tamper_fails_exact_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, source_bundle, _role_paths_value = _write_plan(tmp_path, monkeypatch)
    cache_builder.verify_local_q1_production_plan(binding)
    dependency = source_bundle / "pairwise_v0_1/runtime/dependency.py"
    dependency.write_text("VALUE = 'tampered'\n", encoding="utf-8")
    with pytest.raises(
        cache_builder.LocalQ1CacheBuilderError, match="source-bundle exact manifest"
    ):
        cache_builder.verify_local_q1_production_plan(binding)


def test_production_build_rejects_source_tamper_before_population_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, source_bundle, _role_paths_value = _write_plan(tmp_path, monkeypatch)
    dependency = source_bundle / "pairwise_v0_2/runtime/dependency.py"
    dependency.write_text("VALUE = 'tampered-before-build'\n", encoding="utf-8")
    events = []

    def forbidden_identity(*_args, **_kwargs):
        events.append("identity")
        raise AssertionError("source verification must precede population rebuild")

    monkeypatch.setattr(
        cache_builder.HistoricalIdentityIndex,
        "from_files",
        classmethod(forbidden_identity),
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(
        cache_builder.LocalQ1CacheBuilderError, match="source-bundle exact manifest"
    ):
        cache_builder.build_production_local_q1_cache(
            binding=binding,
            output_dir=output,
            producer_workers=1,
        )
    assert events == []
    assert not output.exists()


def test_every_production_loader_factory_rechecks_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, source_bundle, role_paths = _write_plan(tmp_path, monkeypatch)
    _plan, _locks, authority = cache_builder.verify_local_q1_production_plan(binding)
    events = []

    def fake_loader(sources):
        events.append(tuple(sorted(sources)))
        return object()

    monkeypatch.setattr(cache_builder, "LazyMaskArchiveLoader", fake_loader)
    factory = cache_builder.ProductionMaskLoaderFactory(
        mm_archive=role_paths["mm_archive"],
        eccv_archive=role_paths["eccv_archive"],
        synthetic_archive=role_paths["synthetic_archive"],
        source_bundle=source_bundle,
        expected_source_bundle_manifest_sha256=(
            authority.source_bundle_manifest_sha256
        ),
    )
    factory()
    assert len(events) == 1
    dependency = source_bundle / "pairwise_v0_1/runtime/dependency.py"
    dependency.write_text("VALUE = 'late-tamper'\n", encoding="utf-8")
    with pytest.raises(
        cache_builder.LocalQ1CacheBuilderError,
        match="source bundle changed before loader construction",
    ):
        factory()
    assert len(events) == 1


def test_role_tamper_fails_against_plan_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, _source_bundle_value, role_paths = _write_plan(tmp_path, monkeypatch)
    role_paths["synthetic_archive"].write_bytes(b"tampered archive bytes")
    with pytest.raises(
        cache_builder.LocalQ1CacheBuilderError,
        match="production input role changed: synthetic_archive",
    ):
        cache_builder.verify_local_q1_production_plan(binding)


def test_self_resigned_plan_cannot_replace_external_file_or_content_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, _source_bundle_value, _role_paths_value = _write_plan(
        tmp_path, monkeypatch
    )
    plan = json.loads(binding.plan_path.read_text(encoding="utf-8"))
    plan["scope"]["model_or_backend_created"] = True
    plan.pop("content_sha256")
    plan["content_sha256"] = hashlib.sha256(
        json.dumps(
            plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    binding.plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        cache_builder.LocalQ1CacheBuilderError, match="external file hash"
    ):
        cache_builder.verify_local_q1_production_plan(binding)


def test_source_bundle_must_be_executing_cache_builder_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, _source_bundle_value, role_paths = _write_plan(tmp_path, monkeypatch)
    other = tmp_path / "other-staging"
    other.mkdir()
    wrong = cache_builder.LocalQ1ProductionBinding(
        plan_path=binding.plan_path,
        source_bundle=other,
        role_paths=role_paths,
    )
    with pytest.raises(
        cache_builder.LocalQ1CacheBuilderError, match="executing cache-builder tree"
    ):
        cache_builder.verify_local_q1_production_plan(wrong)


def test_anchor_literal_changes_do_not_change_source_manifest(
    tmp_path: Path,
) -> None:
    source_bundle = _source_bundle(tmp_path)
    before = cache_builder.local_q1_source_bundle_manifest(source_bundle)
    source = source_bundle / "pairwise_v0_2/training/local_q1_cache_builder.py"
    source.write_text(
        'PRODUCTION_LOCAL_Q1_RUN_PLAN_FILE_SHA256 = ("{}")\n'
        'PRODUCTION_LOCAL_Q1_RUN_PLAN_CONTENT_SHA256 = ("{}")\n'.format(
            "a" * 64, "b" * 64
        ),
        encoding="utf-8",
    )
    after = cache_builder.local_q1_source_bundle_manifest(source_bundle)
    assert after == before


def test_source_bundle_rejects_bytecode_cache(tmp_path: Path) -> None:
    source_bundle = _source_bundle(tmp_path)
    cache = source_bundle / "pairwise_v0_1/runtime/__pycache__"
    cache.mkdir()
    (cache / "dependency.cpython-38.pyc").write_bytes(b"untrusted-bytecode")
    with pytest.raises(cache_builder.LocalQ1CacheBuilderError, match="bytecode cache"):
        cache_builder.local_q1_source_bundle_manifest(source_bundle)


def test_unfrozen_compiled_plan_anchors_fail_before_role_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, _source_bundle_value, role_paths = _write_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cache_builder, "PRODUCTION_LOCAL_Q1_RUN_PLAN_FILE_SHA256", "0" * 64
    )
    role_paths["mm_archive"].unlink()
    with pytest.raises(
        cache_builder.LocalQ1CacheBuilderError, match="has not been frozen"
    ):
        cache_builder.verify_local_q1_production_plan(binding)
