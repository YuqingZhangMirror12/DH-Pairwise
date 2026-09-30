from __future__ import annotations

from pathlib import Path

import pytest

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    IDENTITY_INDEX_CONTENT_SCHEMA,
)
from staging.pairwise_v0_2.training import c0_production as production
from staging.pairwise_v0_2.training.c0_runner import (
    C0RunnerError,
    PRODUCTION_RUN_PLAN_ROLE_SPECS,
    ProductionRunPlanBinding,
)


def test_identity_mismatch_precedes_stream_archive_and_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    role_paths = {}
    for role in PRODUCTION_RUN_PLAN_ROLE_SPECS:
        path = tmp_path / role
        if role == "source_bundle":
            path.mkdir()
        else:
            path.write_text("{}\n", encoding="utf-8")
        role_paths[role] = path
    binding = ProductionRunPlanBinding(
        plan_path=tmp_path / "canonical-plan.json",
        role_paths=role_paths,
    )
    plan = {
        "identity_index": {
            "schema_version": IDENTITY_INDEX_CONTENT_SCHEMA,
            "member_count": 10,
            "content_sha256": "a" * 64,
        }
    }
    events: list[str] = []
    monkeypatch.setattr(
        production,
        "verify_c0_production_plan",
        lambda _binding: (plan, ()),
    )

    class _Index:
        identity_count = 9
        content_sha256 = "b" * 64

    monkeypatch.setattr(
        production.HistoricalIdentityIndex,
        "from_files",
        classmethod(lambda _cls, **_kwargs: _Index()),
    )
    monkeypatch.setattr(
        production,
        "_historical_records",
        lambda *_args, **_kwargs: events.append("archive_member_stream"),
    )
    monkeypatch.setattr(
        production,
        "C0CoarseProvider",
        lambda **_kwargs: events.append("provider_factory"),
    )
    monkeypatch.setattr(
        production,
        "C0CoarseBackend",
        lambda *_args: events.append("backend_factory"),
    )
    with pytest.raises(C0RunnerError, match="identity index differs"):
        production.run_c0_production(
            binding=binding,
            output_dir=tmp_path / "output",
            device="cpu",
        )
    assert events == []
