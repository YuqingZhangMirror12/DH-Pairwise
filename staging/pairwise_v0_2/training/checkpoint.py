"""Config-bound, atomic checkpoint helpers for trusted experiment files."""

from __future__ import annotations

import dataclasses
import enum
import hmac
import hashlib
import inspect
import json
import math
import os
import re
import struct
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, Mapping, Optional, Tuple
from urllib.parse import parse_qsl, urlsplit

import torch
from torch import Tensor, nn


CHECKPOINT_VERSION = "dunhuang-pairwise-checkpoint/0.3"
CHECKPOINT_CONTENT_SCHEMA = "dunhuang-pairwise-checkpoint-content/0.3"
_SAFE_FLOAT_TAG = "__dunhuang_pairwise_safe_float_hex_v0_2__"
_SAFE_TUPLE_TAG = "__dunhuang_pairwise_safe_tuple_v0_2__"
_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "api_token",
    "apikey",
    "authorization",
    "bearer_token",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
    "ssh_password",
    "token",
}
_SENSITIVE_QUERY_KEYS = _SENSITIVE_KEYS.union(
    {
        "auth",
        "key",
        "signature",
        "sig",
        "x_amz_credential",
        "x_amz_security_token",
        "x_amz_signature",
    }
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _normalized(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, enum.Enum):
        return _normalized(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalized(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("config/metadata cannot contain NaN or infinity")
        return value
    raise TypeError("unsupported config/metadata type: {}".format(type(value).__name__))


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _is_local_path(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.casefold()
    return (
        stripped.startswith(("/", "~/", "\\\\"))
        or bool(_WINDOWS_ABSOLUTE_RE.match(stripped))
        or lowered.startswith("file:")
    )


def _assert_safe_metadata(value: Any, location: str = "metadata") -> None:
    """Reject credentials and host-specific paths in portable metadata."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = _normalized_key(key)
            if normalized_key in _SENSITIVE_KEYS:
                raise ValueError("sensitive key is forbidden in {}".format(location))
            _assert_safe_metadata(item, "{}.{}".format(location, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_safe_metadata(item, "{}[{}]".format(location, index))
    elif isinstance(value, str):
        if _is_local_path(value):
            raise ValueError("absolute/file path is forbidden in {}".format(location))
        try:
            parsed = urlsplit(value)
        except ValueError as exc:
            raise ValueError(
                "malformed URL is forbidden in {}".format(location)
            ) from exc
        if parsed.password is not None or parsed.username is not None:
            raise ValueError("URL credentials are forbidden in {}".format(location))
        for query_key, _ in parse_qsl(parsed.query, keep_blank_values=True):
            if _normalized_key(query_key) in _SENSITIVE_QUERY_KEYS:
                raise ValueError(
                    "sensitive URL query is forbidden in {}".format(location)
                )


def _normalized_safe(value: Any, location: str) -> Any:
    normalized = _normalized(value)
    _assert_safe_metadata(normalized, location)
    return normalized


def canonical_config_hash(config: Any) -> str:
    normalized = _normalized(config)
    encoded = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_frame(digest: Any, tag: bytes, payload: bytes = b"") -> None:
    """Write an unambiguous typed frame into ``digest``."""

    digest.update(struct.pack(">Q", len(tag)))
    digest.update(tag)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _canonical_tensor_bytes(value: Tensor) -> Tuple[str, Tuple[int, ...], bytes]:
    if value.device.type == "meta":
        raise ValueError("canonical tensor cannot use the meta device")
    if value.layout != torch.strided:
        raise ValueError("canonical tensor must be dense strided")
    if value.is_quantized:
        raise ValueError("canonical tensor cannot be quantized")
    supported_dtypes = {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }
    if value.dtype not in supported_dtypes:
        raise TypeError("unsupported canonical tensor dtype: {}".format(value.dtype))
    cpu = value.detach().cpu().contiguous()
    if (cpu.is_floating_point() or cpu.is_complex()) and not bool(
        torch.isfinite(cpu).all().item()
    ):
        raise ValueError("canonical tensor cannot contain NaN or infinity")
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    if sys.byteorder == "big" and cpu.element_size() > 1:
        # Canonical tensor bytes are little-endian.  Complex numbers consist
        # of two independently endian-sensitive floating components.
        unit = cpu.element_size() // 2 if cpu.is_complex() else cpu.element_size()
        raw = b"".join(
            raw[offset : offset + unit][::-1] for offset in range(0, len(raw), unit)
        )
    return str(cpu.dtype), tuple(int(item) for item in cpu.shape), raw


def canonical_tensor_tree_sha256(value: Any) -> str:
    """Hash a restricted tensor/primitive tree independently of serialization.

    Mappings are ordered by their string keys.  Lists and tuples are distinct,
    floats use exact hexadecimal encoding, and tensors bind dtype, shape and
    exact contiguous CPU bytes in canonical little-endian order.  Optimizer
    state is hashed through the schema-specific adapter below because native
    PyTorch optimizer ``state`` mappings use integer parameter IDs.
    """

    digest = hashlib.sha256()
    active_containers = set()

    def visit(item: Any) -> None:
        if isinstance(item, Tensor):
            dtype, shape, raw = _canonical_tensor_bytes(item)
            _canonical_frame(digest, b"tensor:start")
            _canonical_frame(digest, b"tensor:dtype", dtype.encode("ascii"))
            _canonical_frame(
                digest,
                b"tensor:shape",
                json.dumps(shape, separators=(",", ":")).encode("ascii"),
            )
            _canonical_frame(digest, b"tensor:byte-order", b"little")
            _canonical_frame(digest, b"tensor:data", raw)
            _canonical_frame(digest, b"tensor:end")
            return
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active_containers:
                raise ValueError("canonical tree cannot contain a cycle")
            keys = list(item.keys())
            if any(type(key) is not str for key in keys):  # noqa: E721
                raise TypeError("canonical mapping keys must be built-in strings")
            active_containers.add(identity)
            try:
                _canonical_frame(digest, b"dict:start", str(len(keys)).encode("ascii"))
                for key in sorted(keys):
                    _canonical_frame(digest, b"dict:key", key.encode("utf-8"))
                    visit(item[key])
                _canonical_frame(digest, b"dict:end")
            finally:
                active_containers.remove(identity)
            return
        if type(item) in (list, tuple):
            identity = id(item)
            if identity in active_containers:
                raise ValueError("canonical tree cannot contain a cycle")
            active_containers.add(identity)
            tag = b"list" if type(item) is list else b"tuple"  # noqa: E721
            try:
                _canonical_frame(
                    digest, tag + b":start", str(len(item)).encode("ascii")
                )
                for child in item:
                    visit(child)
                _canonical_frame(digest, tag + b":end")
            finally:
                active_containers.remove(identity)
            return
        if item is None:
            _canonical_frame(digest, b"none")
            return
        if type(item) is bool:  # noqa: E721
            _canonical_frame(digest, b"bool", b"1" if item else b"0")
            return
        if type(item) is int:  # noqa: E721
            _canonical_frame(digest, b"int", str(item).encode("ascii"))
            return
        if type(item) is float:  # noqa: E721
            if not math.isfinite(item):
                raise ValueError("canonical scalar cannot be NaN or infinity")
            _canonical_frame(digest, b"float", item.hex().encode("ascii"))
            return
        if type(item) is str:  # noqa: E721
            _canonical_frame(digest, b"str", item.encode("utf-8"))
            return
        raise TypeError(
            "unsupported canonical tree type: {}".format(type(item).__name__)
        )

    visit(value)
    return digest.hexdigest()


def _canonical_optimizer_view(value: Any) -> Mapping[str, Any]:
    """Represent native optimizer integer parameter IDs without key coercion."""

    if not isinstance(value, Mapping) or set(value) != {"state", "param_groups"}:
        raise ValueError("optimizer state must contain state and param_groups")
    state = value["state"]
    param_groups = value["param_groups"]
    if not isinstance(state, Mapping):
        raise ValueError("optimizer state field must be a mapping")
    entries = []
    for parameter_index, parameter_state in state.items():
        if type(parameter_index) is not int or parameter_index < 0:  # noqa: E721
            raise TypeError(
                "optimizer parameter IDs must be non-negative built-in ints"
            )
        entries.append({"parameter_index": parameter_index, "state": parameter_state})
    entries.sort(key=lambda item: item["parameter_index"])
    return {"param_groups": param_groups, "state_entries": entries}


def _checkpoint_content_fields(
    *,
    epoch: int,
    config_hash: str,
    model_state: Mapping[str, Any],
    optimizer_state: Optional[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Optional[str]]:
    model_hash = canonical_tensor_tree_sha256(model_state)
    optimizer_hash = (
        None
        if optimizer_state is None
        else canonical_tensor_tree_sha256(_canonical_optimizer_view(optimizer_state))
    )
    metrics_hash = canonical_tensor_tree_sha256(metrics)
    provenance_hash = canonical_tensor_tree_sha256(provenance)
    manifest = {
        "checkpoint_content_schema": CHECKPOINT_CONTENT_SCHEMA,
        "checkpoint_version": CHECKPOINT_VERSION,
        "config_hash": config_hash,
        "epoch": epoch,
        "metrics_sha256": metrics_hash,
        "model_state_sha256": model_hash,
        "optimizer_state_sha256": optimizer_hash,
        "provenance_sha256": provenance_hash,
    }
    return {
        "model_state_sha256": model_hash,
        "optimizer_state_sha256": optimizer_hash,
        "metrics_sha256": metrics_hash,
        "provenance_sha256": provenance_hash,
        "canonical_content_sha256": canonical_tensor_tree_sha256(manifest),
    }


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _weights_only_safe_tree(value: Any) -> Any:
    """Encode opcodes unsupported by the Torch 1.13 restricted unpickler.

    Torch 1.13's ``weights_only`` unpickler rejects Python's ``BINFLOAT``
    opcode.  Float hex strings preserve values exactly while leaving tensors
    untouched.  Tuples are tagged as lists to keep the serialized primitive
    surface deliberately small and are restored after restricted loading.
    """

    if isinstance(value, Tensor):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint cannot contain NaN or infinity")
        return {_SAFE_FLOAT_TAG: value.hex()}
    if isinstance(value, Mapping):
        if _SAFE_FLOAT_TAG in value or _SAFE_TUPLE_TAG in value:
            raise ValueError("checkpoint mapping uses a reserved safe-format key")
        return {key: _weights_only_safe_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_weights_only_safe_tree(item) for item in value]
    if isinstance(value, tuple):
        return {_SAFE_TUPLE_TAG: [_weights_only_safe_tree(item) for item in value]}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(
        "unsupported checkpoint state type: {}".format(type(value).__name__)
    )


def _restore_weights_only_safe_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, Mapping):
        if set(value) == {_SAFE_FLOAT_TAG}:
            encoded = value[_SAFE_FLOAT_TAG]
            if not isinstance(encoded, str):
                raise ValueError("safe-format float payload is invalid")
            try:
                result = float.fromhex(encoded)
            except ValueError as exc:
                raise ValueError("safe-format float payload is invalid") from exc
            if not math.isfinite(result):
                raise ValueError("safe-format float must be finite")
            return result
        if set(value) == {_SAFE_TUPLE_TAG}:
            items = value[_SAFE_TUPLE_TAG]
            if not isinstance(items, list):
                raise ValueError("safe-format tuple payload is invalid")
            return tuple(_restore_weights_only_safe_tree(item) for item in items)
        if _SAFE_FLOAT_TAG in value or _SAFE_TUPLE_TAG in value:
            raise ValueError("safe-format reserved key is ambiguous")
        return {
            key: _restore_weights_only_safe_tree(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_restore_weights_only_safe_tree(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise ValueError(
        "restricted checkpoint produced unsupported type: {}".format(
            type(value).__name__
        )
    )


def _sha256_stream(stream: BinaryIO) -> Tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise ValueError("checkpoint stream must yield bytes")
        digest.update(chunk)
        byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)[0]


@contextmanager
def _verified_checkpoint_stream(
    path: Path, expected_file_sha256: str
) -> Iterator[Tuple[BinaryIO, str]]:
    """Verify and rewind the exact file descriptor used by ``torch.load``."""

    if not isinstance(expected_file_sha256, str) or not _SHA256_RE.fullmatch(
        expected_file_sha256
    ):
        raise ValueError("expected_file_sha256 must be 64 lowercase hex characters")
    source = Path(path)
    if not source.is_file():
        raise ValueError("checkpoint path must be a regular file")
    with source.open("rb") as handle:
        observed, _ = _sha256_stream(handle)
        if not hmac.compare_digest(observed, expected_file_sha256):
            raise ValueError("checkpoint file SHA-256 does not match expected value")
        handle.seek(0, os.SEEK_SET)
        yield handle, observed


@dataclass(frozen=True)
class CheckpointReceipt:
    path: str
    file_sha256: str
    config_hash: str
    epoch: int
    model_state_sha256: str
    optimizer_state_sha256: Optional[str]
    canonical_content_sha256: str
    checkpoint_version: str = CHECKPOINT_VERSION


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    config: Any,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> CheckpointReceipt:
    """Atomically save CPU state and a canonical config hash."""

    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    normalized_config = _normalized_safe(config, "config")
    normalized_metrics = _normalized_safe(metrics or {}, "metrics")
    normalized_provenance = _normalized_safe(provenance or {}, "provenance")
    config_hash = canonical_config_hash(normalized_config)
    model_state = _cpu_tree(model.state_dict())
    optimizer_state = None if optimizer is None else _cpu_tree(optimizer.state_dict())
    content_fields = _checkpoint_content_fields(
        epoch=int(epoch),
        config_hash=config_hash,
        model_state=model_state,
        optimizer_state=optimizer_state,
        metrics=normalized_metrics,
        provenance=normalized_provenance,
    )
    payload: Dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_content_schema": CHECKPOINT_CONTENT_SCHEMA,
        "config": normalized_config,
        "config_hash": config_hash,
        "epoch": int(epoch),
        "metrics": normalized_metrics,
        "provenance": normalized_provenance,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        **content_fields,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        # PyTorch uses the output filename stem as the ZIP archive's internal
        # root.  A random NamedTemporaryFile basename therefore makes otherwise
        # identical checkpoints byte-different.  Serialize inside a random
        # private directory but keep the *basename* stable, then atomically
        # replace the destination.
        temporary_directory = tempfile.mkdtemp(
            prefix=".checkpoint-write-", dir=str(destination.parent)
        )
        temporary_name = os.path.join(temporary_directory, destination.name)
        # Protocol 2 plus the safe-tree encoder is accepted by both the early
        # Torch 1.13 restricted unpickler and current Torch 2.5.  No loading
        # path in this module invokes unrestricted pickle.
        torch.save(
            _weights_only_safe_tree(payload),
            temporary_name,
            pickle_protocol=2,
        )
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
        if "temporary_directory" in locals():
            try:
                os.rmdir(temporary_directory)
            except FileNotFoundError:
                pass
    return CheckpointReceipt(
        path=str(destination),
        file_sha256=_sha256(destination),
        config_hash=config_hash,
        epoch=epoch,
        model_state_sha256=str(content_fields["model_state_sha256"]),
        optimizer_state_sha256=content_fields["optimizer_state_sha256"],
        canonical_content_sha256=str(content_fields["canonical_content_sha256"]),
    )


def load_trusted_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    expected_config: Any,
    expected_file_sha256: str,
    expected_canonical_content_sha256: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
    trusted: bool = False,
) -> Dict[str, Any]:
    """Load a hash-pinned checkpoint with restricted PyTorch deserialization.

    ``torch.save`` is a pickle-backed container, so a matching digest alone is
    not a trust decision.  Callers must obtain ``expected_file_sha256`` from an
    independently controlled receipt and explicitly set ``trusted=True``.  The
    loader requires PyTorch's restricted ``weights_only=True`` unpickler; there
    is intentionally no fallback to unrestricted legacy pickle.  Historical
    unknown checkpoints are outside this boundary and must not be loaded here.
    """

    if not trusted:
        raise ValueError("checkpoint deserialization requires trusted=True")
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RuntimeError(
            "this PyTorch lacks restricted weights_only loading; unsafe fallback disabled"
        )
    with _verified_checkpoint_stream(Path(path), expected_file_sha256) as (
        stream,
        observed_file_sha256,
    ):
        encoded_payload = torch.load(
            stream,
            map_location=map_location,
            weights_only=True,
        )
    payload = _restore_weights_only_safe_tree(encoded_payload)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("checkpoint version is unsupported")
    stored_config = payload.get("config")
    stored_hash = payload.get("config_hash")
    normalized_stored_config = _normalized_safe(stored_config, "config")
    normalized_expected_config = _normalized_safe(expected_config, "expected_config")
    normalized_metrics = _normalized_safe(payload.get("metrics", {}), "metrics")
    normalized_provenance = _normalized_safe(
        payload.get("provenance", {}), "provenance"
    )
    if stored_hash != canonical_config_hash(normalized_stored_config):
        raise ValueError("checkpoint config hash is internally inconsistent")
    if stored_hash != canonical_config_hash(normalized_expected_config):
        raise ValueError("checkpoint config does not match expected_config")
    epoch = payload.get("epoch")
    if type(epoch) is not int or epoch < 0:  # noqa: E721
        raise ValueError("checkpoint epoch must be a non-negative int")
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint model state must be a mapping")
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer_state is not None and not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint optimizer state must be a mapping or null")
    if payload.get("checkpoint_content_schema") != CHECKPOINT_CONTENT_SCHEMA:
        raise ValueError("checkpoint content schema is unsupported")
    observed_content = _checkpoint_content_fields(
        epoch=epoch,
        config_hash=stored_hash,
        model_state=model_state,
        optimizer_state=optimizer_state,
        metrics=normalized_metrics,
        provenance=normalized_provenance,
    )
    for field, observed in observed_content.items():
        stored = payload.get(field)
        if stored != observed:
            raise ValueError("checkpoint {} is internally inconsistent".format(field))
    canonical_content_sha256 = str(observed_content["canonical_content_sha256"])
    if expected_canonical_content_sha256 is not None:
        if not isinstance(
            expected_canonical_content_sha256, str
        ) or not _SHA256_RE.fullmatch(expected_canonical_content_sha256):
            raise ValueError(
                "expected_canonical_content_sha256 must be 64 lowercase hex characters"
            )
        if not hmac.compare_digest(
            canonical_content_sha256, expected_canonical_content_sha256
        ):
            raise ValueError(
                "checkpoint canonical content does not match expected value"
            )
    model.load_state_dict(model_state, strict=True)
    if optimizer is not None:
        if optimizer_state is None:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(optimizer_state)
    return {
        "checkpoint_version": payload["checkpoint_version"],
        "config": normalized_stored_config,
        "config_hash": stored_hash,
        "epoch": epoch,
        "metrics": normalized_metrics,
        "provenance": normalized_provenance,
        "file_sha256": observed_file_sha256,
        **observed_content,
    }


__all__ = [
    "CHECKPOINT_VERSION",
    "CHECKPOINT_CONTENT_SCHEMA",
    "CheckpointReceipt",
    "canonical_config_hash",
    "canonical_tensor_tree_sha256",
    "load_trusted_checkpoint",
    "save_checkpoint",
]
