"""Fail-closed, resumable launcher for the frozen Rachel exact-six evaluation.

The launcher deliberately accepts sealed synthetic and real paths only as
opaque strings until every train/validation authority has been restored and
cross-checked.  It then runs, in order, the sealed synthetic evaluation,
corrosion evaluation, target-blind real evaluation, and translation-GT
post-evaluation.  Evaluation artifacts are never overwritten; a resumed
failed stage receives a new numbered attempt path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple

from staging.pairwise_v0_2.baselines import (
    rachel_n512_corrosion_robustness as corrosion,
)
from staging.pairwise_v0_2.baselines import rachel_n512_real_external as real_external
from staging.pairwise_v0_2.baselines import (
    rachel_n512_real_translation_gt as real_translation,
)
from staging.pairwise_v0_2.baselines import (
    rachel_same_data_benchmark_eval_adapter as benchmark_adapter,
)
from staging.pairwise_v0_2.baselines.rachel_matched_mm_evaluation import (
    MATCHED_METHODS,
    freeze_matched_mm_winners,
)
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


SCHEMA_VERSION = "rachel-exact-six-gated-evaluation-launcher/1.0"
STATE_SCHEMA_VERSION = "rachel-exact-six-gated-evaluation-state/1.0"
PREFIX_IMPORT_SCHEMA_VERSION = "rachel-exact-six-prefix-continuation/1.0"
STAGES = ("synthetic", "corrosion", "real_pair_only", "real_translation_gt")
PREFIX_IMPORT_STAGES_BY_STATUS = {
    "complete_verified_synthetic_corrosion_prefix_import": STAGES[:2],
    "complete_verified_synthetic_corrosion_real_pair_only_prefix_import": STAGES[
        :3
    ],
}
FORMAL_METHODS = (
    "coarse_only",
    "full_n512",
    "matched_mm_converged",
    "matched_mm_same_exposure_epoch5",
    benchmark_adapter.PAIRINGNET_METHOD_KEY,
    benchmark_adapter.SHREDDINGNET_METHOD_KEY,
)

DEFAULT_SOURCE_ROOT = Path(
    "/root/autodl-tmp/rachel_pairwise_eval_source_exact6_20260905_001"
)
DEFAULT_N512_RUN = Path(
    "/root/autodl-tmp/rachel_n512_convergence_20260901_001/"
    "run-convergence-50a8cffb0ae92614"
)
DEFAULT_MATCHED_RUN = Path(
    "/root/autodl-tmp/rachel_matched_mm_convergence_20260901_001/"
    "run-52250e6f1d31ca7b"
)
DEFAULT_PAIRINGNET_RUN = Path(
    "/root/autodl-tmp/rachel_same_data_benchmark_direct_20260904_001/"
    "pairingnet_train"
)
DEFAULT_SHREDDINGNET_FREEZE = Path(
    "/root/autodl-tmp/rachel_same_data_benchmark_direct_20260904_001/"
    "shreddingnet_train/train_val_freeze.json"
)
DEFAULT_DATASET_ROOT = Path(
    "/root/autodl-tmp/dataset_rachel_pairwise_n512_v1"
)
DEFAULT_REAL_CONTROL_ROOT = Path(
    "/root/autodl-tmp/rachel_pairwise_source_immutable_20260901_001/"
    "staging/pairwise_v0_2/pairwise_data/real_test_v0_1"
)
DEFAULT_REAL_AUTHORITY_ROOT = Path(
    "/root/autodl-tmp/dunhuang_pairwise_v02/"
    "real_dunhuang_strict_alpha_20260830_001"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260905_001"
)


class ExactSixEvaluationGateError(RuntimeError):
    """A pre-sealed gate, resume contract, or stage validation failed."""


@dataclass(frozen=True)
class ExactSixEvaluationConfig:
    source_root: Path = DEFAULT_SOURCE_ROOT
    n512_run_directory: Path = DEFAULT_N512_RUN
    matched_mm_run_directory: Path = DEFAULT_MATCHED_RUN
    pairingnet_run_directory: Path = DEFAULT_PAIRINGNET_RUN
    shreddingnet_freeze_path: Path = DEFAULT_SHREDDINGNET_FREEZE
    dataset_root: Path = DEFAULT_DATASET_ROOT
    real_manifest_path: Path = DEFAULT_REAL_CONTROL_ROOT / "real_test_manifest.json"
    real_local_receipt_path: Path = DEFAULT_REAL_CONTROL_ROOT / "local_path_receipt.json"
    real_main_root: Path = DEFAULT_REAL_AUTHORITY_ROOT / "Dunhuang Dataset"
    real_supp_root: Path = DEFAULT_REAL_AUTHORITY_ROOT / "Dunhuang Dataset Supp"
    output_root: Path = DEFAULT_OUTPUT_ROOT
    device: str = "cuda:0"
    synthetic_batch_size: int = 16
    synthetic_num_workers: int = 8
    corrosion_batch_size: int = 16
    corrosion_fragment_cache_size: int = 384
    corrosion_bootstrap_replicates: int = 20_000
    corrosion_bootstrap_seed: int = 260_901
    real_batch_size: int = 1
    real_translation_bootstrap_repetitions: int = 20_000
    resume: bool = False
    preflight_only: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_root",
            "n512_run_directory",
            "matched_mm_run_directory",
            "pairingnet_run_directory",
            "shreddingnet_freeze_path",
            "dataset_root",
            "real_manifest_path",
            "real_local_receipt_path",
            "real_main_root",
            "real_supp_root",
            "output_root",
        ):
            value = Path(getattr(self, name)).expanduser()
            if not value.is_absolute():
                raise ValueError(name + " must be absolute")
            object.__setattr__(self, name, Path(os.path.abspath(str(value))))
        for name in (
            "synthetic_batch_size",
            "synthetic_num_workers",
            "corrosion_batch_size",
            "corrosion_fragment_cache_size",
            "corrosion_bootstrap_replicates",
            "real_batch_size",
            "real_translation_bootstrap_repetitions",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        if (
            type(self.corrosion_bootstrap_seed) is not int  # noqa: E721
            or self.corrosion_bootstrap_seed < 0
        ):
            raise ValueError("corrosion_bootstrap_seed must be non-negative")
        if self.synthetic_batch_size != self.corrosion_batch_size:
            raise ValueError(
                "synthetic and corrosion batch sizes must match for clean parity"
            )
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        if type(self.resume) is not bool or type(self.preflight_only) is not bool:  # noqa: E721
            raise TypeError("resume and preflight_only must be bool")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> Dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExactSixEvaluationGateError(
            description + " is not readable strict JSON"
        ) from error
    if not isinstance(value, dict):
        raise ExactSixEvaluationGateError(description + " root must be an object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_new(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ExactSixEvaluationGateError(
            "refusing to overwrite launcher artifact: " + str(path)
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        _fsync_directory(path.parent)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".tmp-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _paths_overlap(first: Path, second: Path) -> bool:
    first_string = os.path.abspath(str(first))
    second_string = os.path.abspath(str(second))
    try:
        common = os.path.commonpath((first_string, second_string))
    except ValueError:
        return False
    return common in {first_string, second_string}


def _validate_namespaces(config: ExactSixEvaluationConfig) -> None:
    protected = (
        config.source_root,
        config.n512_run_directory,
        config.matched_mm_run_directory,
        config.pairingnet_run_directory,
        config.shreddingnet_freeze_path.parent,
        config.dataset_root,
        config.real_manifest_path.parent,
        config.real_local_receipt_path.parent,
        config.real_main_root,
        config.real_supp_root,
    )
    for value in protected:
        if _paths_overlap(config.output_root, value):
            raise ExactSixEvaluationGateError(
                "output root overlaps a protected source namespace: " + str(value)
            )


def _source_hashes() -> Mapping[str, str]:
    files = (
        Path(__file__),
        Path(sealed.__file__),
        Path(benchmark_adapter.__file__),
        Path(corrosion.__file__),
        Path(real_external.__file__),
        Path(real_translation.__file__),
        Path(real_translation.formal_stats.__file__),
    )
    root = Path(__file__).resolve().parents[2]
    return {
        path.resolve(strict=True).relative_to(root).as_posix(): _sha256_file(path)
        for path in files
    }


def _release_models(
    n512_winners: Sequence[object],
    matched_winners: Mapping[str, object],
    benchmark_winners: Mapping[str, object],
) -> None:
    for winner in n512_winners:
        model = getattr(winner, "model", None)
        if model is not None:
            model.to("cpu")
    for winner in matched_winners.values():
        model = getattr(winner, "model", None)
        if model is not None:
            model.to("cpu")
    for winner in benchmark_winners.values():
        release = getattr(winner, "release_to_cpu", None)
        if release is not None:
            release()


def freeze_all_training_authorities(
    config: ExactSixEvaluationConfig,
) -> Mapping[str, object]:
    """Restore every train/val authority without accepting a sealed path."""

    n512_winners: Tuple[object, ...] = ()
    matched_winners: Mapping[str, object] = {}
    benchmark_winners: MutableMapping[str, object] = {}
    try:
        actual_source_root = Path(__file__).resolve().parents[2]
        if config.source_root.resolve(strict=True) != actual_source_root:
            raise ExactSixEvaluationGateError(
                "launcher is not executing from the configured isolated source root"
            )
        # ShreddingNet is intentionally first.  While its formal freeze is
        # absent, no other authority, output, synthetic, or real path is read.
        shredding = benchmark_adapter.freeze_shreddingnet_benchmark(
            config.shreddingnet_freeze_path, device="cpu"
        )
        benchmark_winners[benchmark_adapter.SHREDDINGNET_METHOD_KEY] = shredding
        pairing = benchmark_adapter.freeze_pairingnet_benchmark(
            config.pairingnet_run_directory, device="cpu"
        )
        benchmark_winners = {
            benchmark_adapter.PAIRINGNET_METHOD_KEY: pairing,
            benchmark_adapter.SHREDDINGNET_METHOD_KEY: shredding,
        }
        if pairing.training_manifest_sha256 != shredding.training_manifest_sha256:
            raise ExactSixEvaluationGateError(
                "PairingNet and ShreddingNet train/val manifests differ"
            )

        n512_receipt, n512_receipt_sha, raw_n512_winners = (
            sealed._freeze_completed_winners(config.n512_run_directory)
        )
        sealed._require_formal_convergence(n512_receipt, raw_n512_winners)
        n512_winners = tuple(raw_n512_winners)

        matched_receipt, matched_receipt_sha, matched_winners = (
            freeze_matched_mm_winners(
                config.matched_mm_run_directory, device="cpu"
            )
        )
        if tuple(matched_winners) != MATCHED_METHODS:
            raise ExactSixEvaluationGateError("matched-MM winner inventory differs")
        aligned_dataset_root = sealed._require_matched_training_alignment(
            n512_receipt, matched_receipt
        )
        if aligned_dataset_root.resolve(strict=True) != config.dataset_root.resolve(
            strict=True
        ):
            raise ExactSixEvaluationGateError(
                "configured dataset differs from the aligned train/val authority"
            )
        matched_alignment = sealed._matched_training_hash_evidence(
            n512_receipt,
            matched_receipt,
            n512_run_directory=config.n512_run_directory,
            matched_run_directory=config.matched_mm_run_directory,
        )
        benchmark_alignment = sealed._benchmark_training_manifest_evidence(
            aligned_dataset_root, benchmark_winners
        )

        authority: Dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "all_train_validation_authorities_frozen_before_sealed_open",
            "source_sha256": dict(_source_hashes()),
            "n512": {
                "run_directory": str(config.n512_run_directory),
                "receipt_sha256": n512_receipt_sha,
                "winners": {
                    winner.arm: {
                        "checkpoint_sha256": winner.checkpoint_sha256,
                        "threshold_sha256": winner.threshold.content_sha256,
                    }
                    for winner in n512_winners
                },
            },
            "matched_mm": {
                "run_directory": str(config.matched_mm_run_directory),
                "receipt_sha256": matched_receipt_sha,
                "winners": {
                    method: {
                        "checkpoint_sha256": matched_winners[
                            method
                        ].checkpoint_sha256,
                        "threshold_sha256": matched_winners[
                            method
                        ].threshold.content_sha256,
                    }
                    for method in MATCHED_METHODS
                },
                "alignment": matched_alignment,
            },
            "same_data_benchmarks": {
                "methods": {
                    method: benchmark_winners[method].provenance()
                    for method in benchmark_adapter.BENCHMARK_METHODS
                },
                "alignment": benchmark_alignment,
            },
            "formal_method_order": list(FORMAL_METHODS),
            "sealed_synthetic_path_opened": False,
            "real_path_opened": False,
            "output_created": False,
        }
        authority["content_sha256"] = _canonical_sha256(authority)
        return authority
    except ExactSixEvaluationGateError:
        raise
    except Exception as error:
        raise ExactSixEvaluationGateError(
            "train/validation authority freeze failed: " + str(error)
        ) from error
    finally:
        _release_models(n512_winners, matched_winners, benchmark_winners)


def _config_record(
    config: ExactSixEvaluationConfig, authority: Mapping[str, object]
) -> Dict[str, object]:
    value = asdict(config)
    value.pop("resume")
    value.pop("preflight_only")
    normalized = {
        key: str(item) if isinstance(item, Path) else item
        for key, item in value.items()
    }
    record: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "config": normalized,
        "authority_content_sha256": authority["content_sha256"],
    }
    record["content_sha256"] = _canonical_sha256(record)
    return record


def _initial_state(config_sha256: str) -> Dict[str, object]:
    value: Dict[str, object] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "running",
        "config_sha256": config_sha256,
        "stages": {
            stage: {"status": "pending", "attempts": 0, "output": None}
            for stage in STAGES
        },
    }
    value["content_sha256"] = _canonical_sha256(value)
    return value


def _stamp_state(state: MutableMapping[str, object]) -> None:
    state.pop("content_sha256", None)
    state["content_sha256"] = _canonical_sha256(state)


def _prepare_output_root(
    config: ExactSixEvaluationConfig,
    config_record: Mapping[str, object],
) -> Tuple[Path, Dict[str, object]]:
    root = config.output_root
    config_path = root / "launcher_config.json"
    state_path = root / "launcher_state.json"
    if root.exists() or root.is_symlink():
        if not config.resume:
            raise ExactSixEvaluationGateError(
                "output root exists; use --resume only for this exact launcher run"
            )
        if root.is_symlink() or not root.is_dir():
            raise ExactSixEvaluationGateError("output root must be a real directory")
        if not config_path.is_file():
            raise ExactSixEvaluationGateError(
                "existing output root is not owned by this launcher"
            )
        observed_config = _read_json(config_path, "launcher config")
        if observed_config != dict(config_record):
            raise ExactSixEvaluationGateError(
                "resume configuration or frozen authority differs"
            )
        if state_path.exists():
            state = _read_json(state_path, "launcher state")
        else:
            state = _initial_state(str(config_record["content_sha256"]))
            _write_json_new(state_path, state)
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(root, 0o700)
        _fsync_directory(root.parent)
        _write_json_new(config_path, config_record)
        state = _initial_state(str(config_record["content_sha256"]))
        _write_json_new(state_path, state)
    if (
        state.get("schema_version") != STATE_SCHEMA_VERSION
        or state.get("config_sha256") != config_record["content_sha256"]
        or not isinstance(state.get("stages"), Mapping)
        or set(state["stages"]) != set(STAGES)
    ):
        raise ExactSixEvaluationGateError("launcher resume state differs")
    return state_path, state


@contextmanager
def _exclusive_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".launcher.lock"
    descriptor = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ExactSixEvaluationGateError(
                "another launcher process owns this output root"
            ) from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _stage_row(state: MutableMapping[str, object], stage: str) -> MutableMapping[str, object]:
    stages = state.get("stages")
    if not isinstance(stages, MutableMapping):
        raise ExactSixEvaluationGateError("launcher stages are malformed")
    row = stages.get(stage)
    if not isinstance(row, MutableMapping):
        raise ExactSixEvaluationGateError(stage + " launcher state is malformed")
    return row


def _begin_attempt(
    state_path: Path, state: MutableMapping[str, object], stage: str
) -> int:
    row = _stage_row(state, stage)
    attempts = row.get("attempts")
    if type(attempts) is not int or attempts < 0:  # noqa: E721
        raise ExactSixEvaluationGateError(stage + " attempt count is invalid")
    attempts += 1
    row.update({"status": "running", "attempts": attempts, "output": None})
    _stamp_state(state)
    _write_json_atomic(state_path, state)
    return attempts


def _finish_attempt(
    state_path: Path,
    state: MutableMapping[str, object],
    stage: str,
    output: Path,
) -> None:
    row = _stage_row(state, stage)
    row.update({"status": "complete", "output": str(output)})
    _stamp_state(state)
    _write_json_atomic(state_path, state)


def _fail_attempt(
    state_path: Path,
    state: MutableMapping[str, object],
    stage: str,
    error: BaseException,
) -> None:
    row = _stage_row(state, stage)
    row.update(
        {
            "status": "failed",
            "output": None,
            "last_error": type(error).__name__ + ":" + str(error),
        }
    )
    _stamp_state(state)
    _write_json_atomic(state_path, state)


def _output_member(root: Path, value: object, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExactSixEvaluationGateError(description + " output path is missing")
    path = Path(value).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ExactSixEvaluationGateError(
            description + " output escapes launcher root"
        ) from error
    return path


def _validate_stage_output(root: Path, stage: str, value: object) -> Path:
    output = _output_member(root, value, stage)
    if stage == "synthetic":
        receipt = _read_json(output / "test_receipt.json", "synthetic receipt")
        expected_schema = sealed.SCHEMA_VERSION
        expected_status = sealed.FORMAL_COMPLETE_STATUS
    elif stage == "corrosion":
        receipt = _read_json(
            output / "robustness_receipt.json", "corrosion receipt"
        )
        expected_schema = corrosion.SCHEMA_VERSION
        expected_status = "complete_formal_exact_six_frozen_synthetic_corrosion_only"
    elif stage == "real_pair_only":
        receipt = _read_json(output, "real pair-only result")
        expected_schema = real_external.SCHEMA_VERSION
        expected_status = real_external.FORMAL_COMBINED_STATUS
    else:
        assert stage == "real_translation_gt"
        receipt = _read_json(output, "real translation-GT result")
        expected_schema = real_translation.SCHEMA_VERSION
        expected_status = "complete_postprediction_real_positive_translation_gt"
    if receipt.get("schema_version") != expected_schema or receipt.get(
        "status"
    ) != expected_status:
        raise ExactSixEvaluationGateError(stage + " completion identity differs")
    if stage == "synthetic":
        protocol = receipt.get("protocol")
        if (
            not isinstance(protocol, Mapping)
            or protocol.get("formal_method_inventory") != list(FORMAL_METHODS)
            or protocol.get("formal_exact_six_frozen_before_test_open") is not True
        ):
            raise ExactSixEvaluationGateError(
                "synthetic exact-six completion contract differs"
            )
    elif stage == "corrosion":
        protocol = receipt.get("protocol")
        if (
            not isinstance(protocol, Mapping)
            or protocol.get("formal_method_inventory") != list(FORMAL_METHODS)
            or protocol.get("formal_exact_six") is not True
        ):
            raise ExactSixEvaluationGateError(
                "corrosion exact-six completion contract differs"
            )
    elif stage == "real_pair_only":
        forward = receipt.get("forward_contract")
        if not isinstance(forward, Mapping) or forward.get("methods") != list(
            FORMAL_METHODS
        ):
            raise ExactSixEvaluationGateError("real exact-six method order differs")
    else:
        source = receipt.get("source_pair_only_evaluation")
        if not isinstance(source, Mapping) or source.get("exact_method_order") != list(
            FORMAL_METHODS
        ):
            raise ExactSixEvaluationGateError(
                "translation-GT exact-six method order differs"
            )
    return output


def _reuse_if_complete(
    root: Path, state: MutableMapping[str, object], stage: str
) -> Optional[Path]:
    row = _stage_row(state, stage)
    if row.get("status") != "complete":
        return None
    return _validate_stage_output(root, stage, row.get("output"))


def _output_tree_manifest_sha256(path: Path) -> str:
    root = Path(path)
    if root.is_symlink():
        raise ExactSixEvaluationGateError(
            "imported output must not be a symlink: " + str(root)
        )
    if root.is_file():
        rows = (
            {
                "path": root.name,
                "size": root.stat().st_size,
                "sha256": _sha256_file(root),
            },
        )
    elif root.is_dir():
        values = []
        for value in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if value.is_symlink():
                raise ExactSixEvaluationGateError(
                    "imported output contains a symlink: " + str(value)
                )
            if value.is_dir():
                continue
            if not value.is_file():
                raise ExactSixEvaluationGateError(
                    "imported output contains a special member: " + str(value)
                )
            values.append(
                {
                    "path": value.relative_to(root).as_posix(),
                    "size": value.stat().st_size,
                    "sha256": _sha256_file(value),
                }
            )
        if not values:
            raise ExactSixEvaluationGateError("imported output tree is empty")
        rows = tuple(values)
    else:
        raise ExactSixEvaluationGateError(
            "imported output is not a file or directory: " + str(root)
        )
    return _canonical_sha256(rows)


def _validate_prefix_import(
    root: Path, state: MutableMapping[str, object]
) -> Optional[Mapping[str, object]]:
    raw_path = state.get("prefix_import_receipt")
    if raw_path is None:
        return None
    path = _output_member(root, raw_path, "prefix import receipt")
    if not path.is_file():
        raise ExactSixEvaluationGateError("prefix import receipt is not a file")
    receipt = _read_json(path, "prefix import receipt")
    expected_content_sha = receipt.get("content_sha256")
    body = dict(receipt)
    body.pop("content_sha256", None)
    status = receipt.get("status")
    expected_stages = (
        PREFIX_IMPORT_STAGES_BY_STATUS.get(status)
        if isinstance(status, str)
        else None
    )
    if (
        receipt.get("schema_version") != PREFIX_IMPORT_SCHEMA_VERSION
        or expected_stages is None
        or receipt.get("destination_root") != str(root)
        or not isinstance(expected_content_sha, str)
        or expected_content_sha != _canonical_sha256(body)
        or state.get("prefix_import_receipt_content_sha256")
        != expected_content_sha
    ):
        raise ExactSixEvaluationGateError("prefix import receipt identity differs")
    imported = receipt.get("imported_stages")
    if not isinstance(imported, Mapping) or set(imported) != set(expected_stages):
        raise ExactSixEvaluationGateError("prefix import stage inventory differs")
    declared_order = receipt.get("imported_stage_order")
    if declared_order is not None and declared_order != list(expected_stages):
        raise ExactSixEvaluationGateError("prefix import stage order differs")
    if len(expected_stages) == 3 and declared_order != list(expected_stages):
        raise ExactSixEvaluationGateError("prefix import stage order is missing")
    for stage in expected_stages:
        evidence = imported.get(stage)
        row = _stage_row(state, stage)
        if not isinstance(evidence, Mapping) or row.get("status") != "complete":
            raise ExactSixEvaluationGateError(stage + " import evidence is malformed")
        output = _output_member(root, row.get("output"), stage)
        expected_tree_sha = evidence.get("tree_manifest_sha256")
        if (
            evidence.get("destination_output") != str(output)
            or row.get("tree_manifest_sha256") != expected_tree_sha
            or not isinstance(expected_tree_sha, str)
            or _output_tree_manifest_sha256(output) != expected_tree_sha
        ):
            raise ExactSixEvaluationGateError(stage + " imported bytes differ")
    return {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "content_sha256": expected_content_sha,
        "source_prefix_root": receipt.get("source_prefix_root"),
        "imported_stage_order": list(expected_stages),
        "old_authority_content_sha256": receipt.get(
            "old_authority_content_sha256"
        ),
    }


def _run_stage(
    root: Path,
    state_path: Path,
    state: MutableMapping[str, object],
    stage: str,
    callback,
) -> Path:
    reused = _reuse_if_complete(root, state, stage)
    if reused is not None:
        return reused
    attempt = _begin_attempt(state_path, state, stage)
    try:
        output = Path(callback(attempt))
        validated = _validate_stage_output(root, stage, str(output))
    except BaseException as error:
        _fail_attempt(state_path, state, stage, error)
        raise
    _finish_attempt(state_path, state, stage, validated)
    return validated


def run_exact_six_evaluation(config: ExactSixEvaluationConfig) -> Mapping[str, object]:
    """Run the gated four-stage protocol, or only its preflight when requested."""

    if not isinstance(config, ExactSixEvaluationConfig):
        raise TypeError("config must be ExactSixEvaluationConfig")
    _validate_namespaces(config)

    # Critical boundary: no output root or sealed path operation precedes this.
    authority = freeze_all_training_authorities(config)
    if config.preflight_only:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "preflight_complete_no_output_or_sealed_access",
            "authority_content_sha256": authority["content_sha256"],
        }

    config_record = _config_record(config, authority)
    state_path, state = _prepare_output_root(config, config_record)
    root = config.output_root
    with _exclusive_lock(root):
        prefix_import = _validate_prefix_import(root, state)
        synthetic_output = _run_stage(
            root,
            state_path,
            state,
            "synthetic",
            lambda attempt: sealed.run_sealed_synthetic_test(
                sealed.RachelN512SealedTestConfig(
                    run_directory=config.n512_run_directory,
                    output_root=root
                    / "synthetic"
                    / "attempt-{:03d}".format(attempt),
                    dataset_root=config.dataset_root,
                    matched_mm_run_directory=config.matched_mm_run_directory,
                    pairingnet_run_directory=config.pairingnet_run_directory,
                    shreddingnet_freeze_path=config.shreddingnet_freeze_path,
                    batch_size=config.synthetic_batch_size,
                    num_workers=config.synthetic_num_workers,
                    device=config.device,
                )
            ),
        )
        corrosion_output = _run_stage(
            root,
            state_path,
            state,
            "corrosion",
            lambda attempt: corrosion.run_rachel_n512_corrosion_robustness(
                corrosion.RachelN512CorrosionConfig(
                    run_directory=config.n512_run_directory,
                    matched_mm_run_directory=config.matched_mm_run_directory,
                    sealed_test_directory=synthetic_output,
                    output_root=root
                    / "corrosion"
                    / "attempt-{:03d}".format(attempt),
                    pairingnet_run_directory=config.pairingnet_run_directory,
                    shreddingnet_freeze_path=config.shreddingnet_freeze_path,
                    dataset_root=config.dataset_root,
                    device=config.device,
                    batch_size=config.corrosion_batch_size,
                    fragment_cache_size=config.corrosion_fragment_cache_size,
                    bootstrap_replicates=config.corrosion_bootstrap_replicates,
                    bootstrap_seed=config.corrosion_bootstrap_seed,
                )
            ),
        )
        real_pair_output = _run_stage(
            root,
            state_path,
            state,
            "real_pair_only",
            lambda attempt: _run_real_pair_only(config, root, attempt),
        )
        real_translation_output = _run_stage(
            root,
            state_path,
            state,
            "real_translation_gt",
            lambda attempt: _run_real_translation_gt(
                config, root, attempt, real_pair_output
            ),
        )
        terminal_prefix_import = _validate_prefix_import(root, state)
        if terminal_prefix_import != prefix_import:
            raise ExactSixEvaluationGateError(
                "prefix import evidence changed during continuation"
            )
        terminal_path = root / "launcher_receipt.json"
        terminal: Dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_exact_six_synthetic_corrosion_real",
            "config_sha256": config_record["content_sha256"],
            "authority_content_sha256": authority["content_sha256"],
            "formal_method_order": list(FORMAL_METHODS),
            "outputs": {
                "synthetic": str(synthetic_output),
                "corrosion": str(corrosion_output),
                "real_pair_only": str(real_pair_output),
                "real_translation_gt": str(real_translation_output),
            },
        }
        if terminal_prefix_import is not None:
            terminal["prefix_import"] = terminal_prefix_import
        terminal["content_sha256"] = _canonical_sha256(terminal)
        if terminal_path.exists():
            if _read_json(terminal_path, "launcher terminal receipt") != terminal:
                raise ExactSixEvaluationGateError(
                    "existing launcher terminal receipt differs"
                )
        else:
            _write_json_new(terminal_path, terminal)
        state["status"] = "complete"
        state["terminal_receipt"] = str(terminal_path)
        _stamp_state(state)
        _write_json_atomic(state_path, state)
        return terminal


def _run_real_pair_only(
    config: ExactSixEvaluationConfig, root: Path, attempt: int
) -> Path:
    output = root / "real" / "pair-only-attempt-{:03d}.json".format(attempt)
    real_external.evaluate_strict_real_external(
        config.n512_run_directory,
        config.real_manifest_path,
        config.real_local_receipt_path,
        matched_mm_run_directory=config.matched_mm_run_directory,
        pairingnet_run_directory=config.pairingnet_run_directory,
        shreddingnet_freeze_path=config.shreddingnet_freeze_path,
        main_root=config.real_main_root,
        supp_root=config.real_supp_root,
        device=config.device,
        batch_size=config.real_batch_size,
        output_path=output,
        include_balanced_1016=True,
        compatibility_mode=False,
    )
    return output


def _run_real_translation_gt(
    config: ExactSixEvaluationConfig,
    root: Path,
    attempt: int,
    real_pair_output: Path,
) -> Path:
    output = root / "real" / "translation-gt-attempt-{:03d}.json".format(
        attempt
    )
    real_translation.evaluate_real_translation_gt_postprediction(
        real_pair_output,
        config.real_manifest_path,
        config.real_local_receipt_path,
        output_path=output,
        main_root=config.real_main_root,
        supp_root=config.real_supp_root,
        bootstrap_repetitions=config.real_translation_bootstrap_repetitions,
        compatibility_mode=False,
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--n512-run-directory", type=Path, default=DEFAULT_N512_RUN)
    parser.add_argument(
        "--matched-mm-run-directory", type=Path, default=DEFAULT_MATCHED_RUN
    )
    parser.add_argument(
        "--pairingnet-run-directory", type=Path, default=DEFAULT_PAIRINGNET_RUN
    )
    parser.add_argument(
        "--shreddingnet-freeze-path",
        type=Path,
        default=DEFAULT_SHREDDINGNET_FREEZE,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--real-manifest",
        type=Path,
        default=DEFAULT_REAL_CONTROL_ROOT / "real_test_manifest.json",
    )
    parser.add_argument(
        "--real-local-receipt",
        type=Path,
        default=DEFAULT_REAL_CONTROL_ROOT / "local_path_receipt.json",
    )
    parser.add_argument(
        "--real-main-root",
        type=Path,
        default=DEFAULT_REAL_AUTHORITY_ROOT / "Dunhuang Dataset",
    )
    parser.add_argument(
        "--real-supp-root",
        type=Path,
        default=DEFAULT_REAL_AUTHORITY_ROOT / "Dunhuang Dataset Supp",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--synthetic-batch-size", type=int, default=16)
    parser.add_argument("--synthetic-num-workers", type=int, default=8)
    parser.add_argument("--corrosion-batch-size", type=int, default=16)
    parser.add_argument("--corrosion-fragment-cache-size", type=int, default=384)
    parser.add_argument(
        "--corrosion-bootstrap-replicates", type=int, default=20_000
    )
    parser.add_argument("--corrosion-bootstrap-seed", type=int, default=260_901)
    parser.add_argument("--real-batch-size", type=int, default=1)
    parser.add_argument(
        "--real-translation-bootstrap-repetitions", type=int, default=20_000
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = run_exact_six_evaluation(
            ExactSixEvaluationConfig(
                source_root=arguments.source_root,
                n512_run_directory=arguments.n512_run_directory,
                matched_mm_run_directory=arguments.matched_mm_run_directory,
                pairingnet_run_directory=arguments.pairingnet_run_directory,
                shreddingnet_freeze_path=arguments.shreddingnet_freeze_path,
                dataset_root=arguments.dataset_root,
                real_manifest_path=arguments.real_manifest,
                real_local_receipt_path=arguments.real_local_receipt,
                real_main_root=arguments.real_main_root,
                real_supp_root=arguments.real_supp_root,
                output_root=arguments.output_root,
                device=arguments.device,
                synthetic_batch_size=arguments.synthetic_batch_size,
                synthetic_num_workers=arguments.synthetic_num_workers,
                corrosion_batch_size=arguments.corrosion_batch_size,
                corrosion_fragment_cache_size=(
                    arguments.corrosion_fragment_cache_size
                ),
                corrosion_bootstrap_replicates=(
                    arguments.corrosion_bootstrap_replicates
                ),
                corrosion_bootstrap_seed=arguments.corrosion_bootstrap_seed,
                real_batch_size=arguments.real_batch_size,
                real_translation_bootstrap_repetitions=(
                    arguments.real_translation_bootstrap_repetitions
                ),
                resume=arguments.resume,
                preflight_only=arguments.preflight_only,
            )
        )
    except (ExactSixEvaluationGateError, ValueError) as error:
        print("exact-six evaluation refused: " + str(error), file=os.sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
