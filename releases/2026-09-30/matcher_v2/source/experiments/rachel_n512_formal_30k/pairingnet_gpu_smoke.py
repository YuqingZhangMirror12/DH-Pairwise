"""One-effective-batch, train-only CUDA smoke for the Rachel PairingNet adapter.

This executable is deliberately separate from the formal benchmark runner.  It
constructs the real adapter, decodes exactly one formal-size batch from the
Rachel training manifest, and performs one optimizer step.  It never opens a
sealed-test or real-data entry point and it never creates a formal run folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import Mapping, Optional, Sequence

import torch

from staging.pairwise_v0_2.baselines import rachel_pairingnet_benchmark as pairing
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import (
    RachelDatasetConfig,
    RachelPairDataset,
)
from staging.pairwise_v0_2.training.rachel_n512_runner import epoch_indices


SCHEMA_VERSION = "rachel-pairingnet-gpu-smoke/1.0"
FORMAL_BATCH_SIZE = 20


class PairingNetSmokeError(RuntimeError):
    """The train-only smoke contract was not satisfied."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_bound(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result["content_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def _lexical_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.normpath(str(expanded)))


def _assert_no_symlink_components(path: Path) -> None:
    absolute = _lexical_absolute(path)
    components = (Path(absolute.anchor),) if absolute.anchor else ()
    current = components[0] if components else Path()
    for part in absolute.parts[1:] if absolute.anchor else absolute.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise PairingNetSmokeError(
                "path traverses a symlink component: " + str(current)
            )


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _write_json_new(path: Path, value: Mapping[str, object]) -> dict[str, object]:
    destination = _lexical_absolute(path)
    _assert_no_symlink_components(destination)
    if os.path.lexists(destination):
        raise PairingNetSmokeError("smoke receipt already exists")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(parent)
    payload_value = _content_bound(value)
    payload = json.dumps(
        payload_value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + destination.name + ".tmp-", dir=str(parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise PairingNetSmokeError("smoke receipt appeared before publish") from error
        directory_descriptor = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return payload_value


def run_smoke(
    *,
    dataset_root: Path,
    official_source_root: Path,
    output: Path,
    device: str,
    precision: str,
    num_workers: int,
) -> dict[str, object]:
    if precision not in {"fp32", "bf16"}:
        raise PairingNetSmokeError("precision must be fp32 or bf16")
    if type(num_workers) is not int or num_workers < 0:  # noqa: E721
        raise PairingNetSmokeError("num_workers must be a non-negative integer")
    device_value = torch.device(device)
    if device_value.type != "cuda" or not torch.cuda.is_available():
        raise PairingNetSmokeError("the formal smoke requires an available CUDA device")

    destination = _lexical_absolute(output)
    _assert_no_symlink_components(destination)
    if os.path.lexists(destination):
        raise PairingNetSmokeError("smoke output must be fresh")

    dataset = Path(dataset_root).expanduser().resolve(strict=True)
    official = Path(official_source_root).expanduser().resolve(strict=True)
    source_root = Path(__file__).resolve(strict=True).parents[2]
    canonical_destination = destination.resolve(strict=False)
    if any(
        _paths_overlap(canonical_destination, protected)
        for protected in (dataset, official, source_root)
    ):
        raise PairingNetSmokeError("smoke receipt overlaps a protected input namespace")
    population = pairing.audit_train_val_population(dataset, require_formal_counts=True)
    official_audit = pairing.audit_official_source(official)
    adapter_source_sha256, adaptation_contract_sha256 = (
        pairing._adapter_identity_hashes()
    )

    dataset_config = RachelDatasetConfig(
        mask_size=pairing.PairingNetRachelModelConfig().canvas_size,
        contour_cap=pairing.PairingNetRachelModelConfig().contour_cap,
    )
    train_dataset = RachelPairDataset(dataset, "train", dataset_config)
    if len(train_dataset) != pairing.FORMAL_TRAIN_ROWS:
        raise PairingNetSmokeError("formal train population changed after audit")
    train_metadata = pairing._read_manifest(dataset, "train")
    if len(train_metadata) != len(train_dataset):
        raise PairingNetSmokeError("train metadata/dataset row count differs")
    order = epoch_indices(
        len(train_dataset), pairing.FORMAL_SEED, 1, None
    )[:FORMAL_BATCH_SIZE]
    expected_pair_ids = tuple(train_metadata[index].pair_id for index in order)
    expected_labels = tuple(bool(train_metadata[index].label) for index in order)
    if not any(expected_labels) or all(expected_labels):
        raise PairingNetSmokeError(
            "the first formal batch must expose both positive and negative examples"
        )
    loader = pairing._loader(
        train_dataset,
        order,
        batch_size=FORMAL_BATCH_SIZE,
        num_workers=num_workers,
        seed=pairing.FORMAL_SEED + 1,
    )

    pairing._set_determinism(pairing.FORMAL_SEED)
    # Some CUDA builds reject an explicit ``torch.device`` in the memory
    # statistics APIs even though model placement accepts it.  Select the
    # requested device once, then query/reset statistics for the current
    # device through the no-argument API.
    torch.cuda.set_device(device_value)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model_config = pairing.PairingNetRachelModelConfig()
    run_config = pairing.PairingNetRachelRunConfig(
        dataset_root=dataset,
        output_root=destination.parent / "never-created-formal-output",
        official_source_root=official,
        device=device,
        precision=precision,
        num_workers=num_workers,
    )
    model = pairing.PairingNetRachelAdapted(model_config).to(device_value).train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    batch = next(iter(loader))
    if len(batch.pair_ids) != FORMAL_BATCH_SIZE:
        raise PairingNetSmokeError("smoke did not obtain one full formal batch")
    observed_labels = tuple(bool(float(value) >= 0.5) for value in batch.labels)
    if tuple(batch.pair_ids) != expected_pair_ids or observed_labels != expected_labels:
        raise PairingNetSmokeError("smoke batch pair/label order differs")
    inputs, targets = pairing._batch_tensors(batch, device_value)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=precision == "bf16",
    ):
        model_output = model(*inputs)
        loss = pairing.compute_pairingnet_loss(
            model_output,
            targets[0],
            targets[1],
            pair_weight=run_config.pair_loss_weight,
            matching_weight=run_config.matching_loss_weight,
        )
    loss.total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), run_config.gradient_clip_norm, error_if_nonfinite=True
    )
    optimizer.step()
    torch.cuda.synchronize(device_value)
    scalar_values = {
        "total": float(loss.total.detach().float().cpu().item()),
        "matching_focal": float(loss.matching_focal.detach().float().cpu().item()),
        "pair_bce": float(loss.pair_bce.detach().float().cpu().item()),
        "gradient_norm": float(gradient_norm.detach().float().cpu().item()),
    }
    if not all(math.isfinite(value) for value in scalar_values.values()):
        raise PairingNetSmokeError("smoke produced a non-finite scalar")
    batch_order_rows = [
        {"ordinal": ordinal, "pair_id": pair_id, "label": label}
        for ordinal, (pair_id, label) in enumerate(
            zip(expected_pair_ids, expected_labels)
        )
    ]
    batch_order = {
        "schema_version": "rachel-pairingnet-gpu-smoke-batch-order/1.0",
        "seed": pairing.FORMAL_SEED,
        "epoch_one_based": 1,
        "rows": batch_order_rows,
        "ordered_pair_ids_sha256": hashlib.sha256(
            _canonical_bytes(list(expected_pair_ids))
        ).hexdigest(),
        "ordered_pair_labels_sha256": hashlib.sha256(
            _canonical_bytes(batch_order_rows)
        ).hexdigest(),
        "positive_pair_ids": [
            pair_id
            for pair_id, label in zip(expected_pair_ids, expected_labels)
            if label
        ],
        "negative_pair_ids": [
            pair_id
            for pair_id, label in zip(expected_pair_ids, expected_labels)
            if not label
        ],
        "contains_positive_and_negative": True,
    }

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_train_only_one_formal_batch_optimizer_step",
        "method_id": pairing.METHOD_ID,
        "official_commit": pairing.OFFICIAL_COMMIT,
        "adapter_source_sha256": adapter_source_sha256,
        "adaptation_contract_sha256": adaptation_contract_sha256,
        "dataset_manifest_sha256": population["manifests"],
        "batch_size": FORMAL_BATCH_SIZE,
        "device": str(device_value),
        "device_name": torch.cuda.get_device_name(device_value),
        "precision": precision,
        "num_workers": num_workers,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "scalars": scalar_values,
        "batch_order": batch_order,
        "official_source_audit": official_audit,
        "scope": {
            "train_manifest_used_for_tensor_loading": True,
            "validation_manifest_metadata_audited": True,
            "validation_tensor_loading": False,
            "sealed_synthetic_or_real_opened": False,
            "formal_training_started": False,
            "random_smoke_model_not_checkpoint": True,
            "optimizer_steps": 1,
        },
    }
    return _write_json_new(destination, receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="one-formal-batch train-only PairingNet CUDA smoke"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--official-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    parsed = _parser().parse_args(arguments)
    receipt = run_smoke(
        dataset_root=parsed.dataset_root,
        official_source_root=parsed.official_source_root,
        output=parsed.output,
        device=parsed.device,
        precision=parsed.precision,
        num_workers=parsed.num_workers,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
