#!/usr/bin/env python3
"""Production entrypoint for the preregistered C0-N-Q1 run.

The CLI accepts only local role paths, output directory, and device.  All
expected hashes, counts, and policies come from the canonical run plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    HistoricalIdentityIndex,
    freeze_c0_n_q1,
)
from staging.pairwise_v0_2.pairwise_data.lazy_mask_loader import (
    ArchiveSourceSpec,
    LazyMaskArchiveLoader,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ECCV_CANONICAL_BINDING,
    MM_CANONICAL_BINDING,
    TrainingPairRecord,
    iter_historical_pair_records,
)
from staging.pairwise_v0_2.training.c0_coarse_backend import C0CoarseBackend
from staging.pairwise_v0_2.training.c0_coarse_provider import C0CoarseProvider
from staging.pairwise_v0_2.training.c0_runner import (
    C0RunArtifacts,
    C0RunnerContract,
    C0RunnerError,
    FrozenReceiptLock,
    PRODUCTION_FREEZE_CONTENT_SHA256,
    PRODUCTION_FREEZE_FILE_SHA256,
    PRODUCTION_RUN_PLAN_ROLE_SPECS,
    ProductionRunPlanBinding,
    run_c0_n_q1,
    verify_c0_production_plan,
)
from staging.pairwise_v0_2.training.c0_runtime_adapter import (
    C0CoarseBackendAdapter,
    C0CoarseProviderAdapter,
)


CANONICAL_PLAN_PATH = Path(__file__).resolve().parents[1] / "preflight/c0_run_plan.json"


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise C0RunnerError("verified C0 metadata is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise C0RunnerError("verified C0 metadata root must be an object")
    return value


def _historical_records(
    role_paths: Mapping[str, Path], split: str
) -> Iterable[TrainingPairRecord]:
    return iter_historical_pair_records(
        split_manifest=role_paths["historical_split"],
        split=split,
        mm_archive=role_paths["mm_archive"],
        eccv_archive=role_paths["eccv_archive"],
    )


def _rebuild_frozen_records(
    *,
    role_paths: Mapping[str, Path],
    plan: Mapping[str, Any],
) -> tuple[
    HistoricalIdentityIndex,
    Mapping[str, Any],
    tuple[TrainingPairRecord, ...],
    tuple[TrainingPairRecord, ...],
]:
    freeze = _load_json(role_paths["freeze_receipt"])
    index = HistoricalIdentityIndex.from_files(
        mm_cache_path=role_paths["mm_fingerprint_cache"],
        eccv_cache_path=role_paths["eccv_fingerprint_cache"],
        split_path=role_paths["historical_split"],
    )
    identity_lock = plan["identity_index"]
    if (
        index.identity_count != identity_lock["member_count"]
        or index.content_sha256 != identity_lock["content_sha256"]
    ):
        raise C0RunnerError("rebuilt historical identity index differs from plan")

    validation_raw = tuple(_historical_records(role_paths, "val"))
    training_raw = _historical_records(role_paths, "train")
    rebuilt = freeze_c0_n_q1(
        identity_index=index,
        validation_records=validation_raw,
        training_records=training_raw,
        archive_paths={
            "mm_augmented": role_paths["mm_archive"],
            "eccv_1113data": role_paths["eccv_archive"],
        },
    )
    if rebuilt.receipt != freeze:
        raise C0RunnerError("metadata reconstruction differs from canonical freeze")
    validation = tuple(index.verify_record(record).record for record in validation_raw)
    return index, freeze, rebuilt.selected_train_records, validation


def run_c0_production(
    *,
    binding: ProductionRunPlanBinding,
    output_dir: Path,
    device: str,
) -> C0RunArtifacts:
    """Rebuild all historical inputs, then execute the verified C0 runner."""

    # The first action validates every preregistered external file/source byte.
    plan, _verified_roles = verify_c0_production_plan(binding)
    index, freeze, selected_train, validation = _rebuild_frozen_records(
        role_paths=binding.role_paths,
        plan=plan,
    )
    loader_holder: Dict[str, LazyMaskArchiveLoader] = {}

    def provider_factory() -> C0CoarseProviderAdapter:
        if loader_holder:
            raise C0RunnerError("production provider factory called more than once")
        loader = LazyMaskArchiveLoader(
            {
                MM_CANONICAL_BINDING.logical_id: ArchiveSourceSpec(
                    binding=MM_CANONICAL_BINDING,
                    source=binding.role_paths["mm_archive"],
                ),
                ECCV_CANONICAL_BINDING.logical_id: ArchiveSourceSpec(
                    binding=ECCV_CANONICAL_BINDING,
                    source=binding.role_paths["eccv_archive"],
                ),
            }
        )
        loader_holder["loader"] = loader
        provider = C0CoarseProvider(
            identity_index=index,
            mask_loader=loader,
            freeze_receipt=freeze,
            expected_freeze_content_sha256=PRODUCTION_FREEZE_CONTENT_SHA256,
            expected_identity_index_content_sha256=plan["identity_index"][
                "content_sha256"
            ],
        )
        return C0CoarseProviderAdapter(
            provider,
            archive_locks=freeze["locks"]["archives"],
            expected_freeze_content_sha256=PRODUCTION_FREEZE_CONTENT_SHA256,
            expected_identity_index_content_sha256=plan["identity_index"][
                "content_sha256"
            ],
        )

    def backend_factory() -> C0CoarseBackendAdapter:
        return C0CoarseBackendAdapter(C0CoarseBackend(device), runner_seed="260828")

    try:
        return run_c0_n_q1(
            run_plan=C0RunnerContract(),
            freeze_receipt=FrozenReceiptLock(
                path=binding.role_paths["freeze_receipt"],
                file_sha256=PRODUCTION_FREEZE_FILE_SHA256,
                content_sha256=PRODUCTION_FREEZE_CONTENT_SHA256,
            ),
            selected_train_records=selected_train,
            validation_records=validation,
            provider_factory=provider_factory,
            backend_factory=backend_factory,
            output_dir=output_dir,
            production_plan=binding,
        )
    finally:
        loader = loader_holder.get("loader")
        if loader is not None:
            loader.close()


def _parse_binding(value: str) -> tuple[str, Path]:
    role, separator, path = value.partition("=")
    if not separator or not role or not path:
        raise argparse.ArgumentTypeError("binding must be ROLE=PATH")
    if role not in PRODUCTION_RUN_PLAN_ROLE_SPECS:
        raise argparse.ArgumentTypeError("unknown C0 plan role: " + role)
    return role, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bind",
        action="append",
        type=_parse_binding,
        required=True,
        metavar="ROLE=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    pairs = list(args.bind)
    role_paths = dict(pairs)
    if len(role_paths) != len(pairs) or set(role_paths) != set(
        PRODUCTION_RUN_PLAN_ROLE_SPECS
    ):
        raise C0RunnerError("CLI requires each canonical role exactly once")
    artifacts = run_c0_production(
        binding=ProductionRunPlanBinding(
            plan_path=CANONICAL_PLAN_PATH,
            role_paths=role_paths,
        ),
        output_dir=args.output_dir,
        device=args.device,
    )
    print(artifacts.receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CANONICAL_PLAN_PATH", "run_c0_production"]
