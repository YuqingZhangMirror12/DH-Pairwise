"""Build and verify the isolated Rachel paper-benchmark Python environment.

The builder deliberately has no dataset argument and performs no CUDA work.  It
creates one fresh venv from an already provisioned Torch runtime, but every
runtime subprocess starts with ``-I -S -B`` and receives only content-bound,
held-directory site roots.  It downloads exactly three pre-reviewed wheels,
installs them with hashes and ``--no-deps``, and publishes the queue-compatible
receipt only after all checks pass.  ``verify`` is read-only and rehashes the
imported package closure.

This module remains compatible with Python 3.8 because the remote base runtime
may be older than the development workstation.
"""

from __future__ import annotations

import argparse
import base64
import csv
import ctypes
from email.parser import Parser
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit
import zipfile


ENVIRONMENT_SCHEMA = "rachel-benchmark-isolated-python-environment/1.0"
STATUS = "complete_isolated_benchmark_environment"
OFFICIAL_SHREDDINGNET_COMMIT = "0ae3b544ca4e910732f3f459b39aa15cdc62dbcb"
OFFICIAL_ENV_YAML_SHA256 = (
    "2dfb583a57ec6346cba6f42be02ea35ff05b67484157ed51b61aad0c204d9e08"
)
EXPECTED_TORCH_MODULE_VERSION = "2.5.1+cu124"
EXPECTED_TORCH_CUDA_VERSION = "12.4"
EXPECTED_NUMPY_ADAPTATION_VERSION = "2.1.3"
OFFICIAL_NUMPY_VERSION = "2.2.3"
EXPECTED_TORCHVISION_BASE_VERSION = "0.20.1"
EXPECTED_PILLOW_VERSION = "11.0.0"
WHEEL_LOCK_SCHEMA = "rachel-benchmark-reviewed-wheel-lock/1.0"
REVIEWED_WHEEL_LOCK_RELATIVE_PATH = (
    "experiments/rachel_n512_formal_30k/rachel_benchmark_reviewed_wheel_lock.json"
)
REVIEWED_WHEEL_LOCK_FILE_SHA256 = (
    "c0e2861e4148174fc25c5e14be539e29470065ed93f06814a3e15cd7540e0406"
)
ADDITIONS = (
    ("torch-geometric", "2.6.1", "torch_geometric"),
    ("opencv-python", "4.10.0.84", "cv2"),
    ("scipy", "1.14.1", "scipy"),
)
INHERITED_DISTRIBUTIONS = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("Pillow", "PIL"),
)
TRACKED_DISTRIBUTIONS = INHERITED_DISTRIBUTIONS + tuple(
    (distribution, module) for distribution, _version, module in ADDITIONS
)
PIP_RUNTIME_DISTRIBUTION = (("pip", "pip"),)
BASE_TRACKED_DISTRIBUTIONS = INHERITED_DISTRIBUTIONS + PIP_RUNTIME_DISTRIBUTION
TARGET_TRACKED_DISTRIBUTIONS = TRACKED_DISTRIBUTIONS + PIP_RUNTIME_DISTRIBUTION
RECEIPT_NAME = "environment_receipt.json"
INVENTORY_RELATIVE_PATH = ".environment_build_evidence/import_inventory.json"
INSTALL_REPORT_RELATIVE_PATH = ".environment_build_evidence/pip_install_report.json"
WHEELHOUSE_RELATIVE_PATH = ".environment_build_evidence/wheelhouse"
LOCK_RELATIVE_PATH = ".environment_build_evidence/requirements.sha256.txt"
ACTIVE_SITE_RELATIVE_PATH = ".environment_build_evidence/active_site_audit.json"
EXPLICIT_SITE_BOOTSTRAP_SCHEMA = "rachel-benchmark-explicit-site-bootstrap/3.0"
RUNTIME_NO_SITE_SCHEMA = "rachel-benchmark-runtime-no-site-evidence/2.0"
TORCHVISION_DYNAMIC_ADMISSION_SCHEMA = (
    "rachel-benchmark-torchvision-dynamic-module-admission/1.0"
)
TORCHVISION_DYNAMIC_AUTHORITIES = {
    "torch_version": "2.5.1+cu124",
    "torchvision_version": "0.20.1+cu124",
    "instantiator_relative_path": (
        "torch/distributed/nn/jit/instantiator.py"
    ),
    "instantiator_module": "torch.distributed.nn.jit.instantiator",
    "instantiator_source_sha256": (
        "440a619c764e4133564d7956ba060a7223e94664854b94a4a2074d095756db7e"
    ),
    "instantiator_append_line": 23,
    "instantiator_append_source": (
        "sys.path.append(INSTANTIATED_TEMPLATE_DIR_PATH)"
    ),
    "remote_module_relative_path": (
        "torch/distributed/nn/api/remote_module.py"
    ),
    "remote_module_module": "torch.distributed.nn.api.remote_module",
    "remote_module_source_sha256": (
        "55c9c44ba25a2b5edf105fbd740ceff771f937147d7d6a9d6232f05681e7eeaf"
    ),
    "template_relative_path": (
        "torch/distributed/nn/jit/templates/remote_module_template.py"
    ),
    "template_module": (
        "torch.distributed.nn.jit.templates.remote_module_template"
    ),
    "template_source_sha256": (
        "0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc"
    ),
    "generated_module_name": "_remote_module_non_scriptable",
    "generated_module_filename": "_remote_module_non_scriptable.py",
    "generated_module_size": 2355,
    "generated_module_sha256": (
        "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
    ),
    "preseed_origin": "<rachel-pinned-_remote_module_non_scriptable>",
}
TORCHVISION_DYNAMIC_ADMISSION_POLICY = {
    "controlled_sys_path_default_deny": True,
    "single_pinned_instantiator_append": True,
    "captured_directory_fd_held_through_payload": True,
    "captured_directory_identity": "regular_directory,euid_owner,mode_0700,stable_dev_inode",
    "captured_file_identity": "regular_file,euid_owner,mode_0644,nlink_1,stable_fd_hash",
    "captured_path_import_denied": True,
    "generated_write_fd_anchored": True,
    "preseed_module_memory_only": True,
    "captured_origin_compile_and_exec_denied": True,
    "pinned_modules_loaded_from_held_source_bytes": True,
    "pinned_module_bytecode_cache_disabled": True,
    "reviewed_sys_path_restored_before_payload": True,
}


class EnvironmentBuildError(RuntimeError):
    """A fail-closed environment invariant was violated."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _torchvision_dynamic_admission_configuration(
    enabled: bool,
) -> Dict[str, object]:
    if not isinstance(enabled, bool):
        raise EnvironmentBuildError("dynamic admission enable flag differs")
    return {
        "schema_version": TORCHVISION_DYNAMIC_ADMISSION_SCHEMA,
        "enabled": enabled,
        "authorities": dict(TORCHVISION_DYNAMIC_AUTHORITIES),
        "policy": dict(TORCHVISION_DYNAMIC_ADMISSION_POLICY),
    }


def _successful_torchvision_dynamic_admission_evidence() -> Dict[str, object]:
    configured = _torchvision_dynamic_admission_configuration(True)
    return {
        "schema_version": configured["schema_version"],
        "enabled": True,
        "authorities": configured["authorities"],
        "policy": configured["policy"],
        "observed": {
            "append_caller_validated": True,
            "append_count": 1,
            "captured_compile_count": 0,
            "captured_directory_final_entries": [
                TORCHVISION_DYNAMIC_AUTHORITIES["generated_module_filename"]
            ],
            "captured_directory_initially_empty": True,
            "captured_directory_revalidated_after_payload": True,
            "captured_exec_count": 0,
            "captured_finder_count": 0,
            "captured_temp_origin_module_count": 0,
            "generated_file_sha256": TORCHVISION_DYNAMIC_AUTHORITIES[
                "generated_module_sha256"
            ],
            "generated_file_size": TORCHVISION_DYNAMIC_AUTHORITIES[
                "generated_module_size"
            ],
            "preseed_compile_count": 1,
            "preseed_exec_count": 1,
            "pinned_source_compile_count": {
                TORCHVISION_DYNAMIC_AUTHORITIES["instantiator_module"]: 1,
                TORCHVISION_DYNAMIC_AUTHORITIES["remote_module_module"]: 1,
                TORCHVISION_DYNAMIC_AUTHORITIES["template_module"]: 1,
            },
            "pinned_source_exec_count": {
                TORCHVISION_DYNAMIC_AUTHORITIES["instantiator_module"]: 1,
                TORCHVISION_DYNAMIC_AUTHORITIES["remote_module_module"]: 1,
                TORCHVISION_DYNAMIC_AUTHORITIES["template_module"]: 1,
            },
            "pinned_source_find_count": {
                TORCHVISION_DYNAMIC_AUTHORITIES["instantiator_module"]: 1,
                TORCHVISION_DYNAMIC_AUTHORITIES["remote_module_module"]: 1,
                TORCHVISION_DYNAMIC_AUTHORITIES["template_module"]: 1,
            },
            "pinned_source_loader_identity_validated": True,
            "pinned_source_pyc_read_count": 0,
            "remote_module_identity_is_preseed": True,
            "reviewed_sys_path_restored": True,
            "template_compile_count": 1,
            "template_exec_count": 1,
        },
    }


def _validate_torchvision_dynamic_admission_evidence(
    value: object, label: str
) -> Dict[str, object]:
    evidence = _require_mapping(value, label)
    expected = _successful_torchvision_dynamic_admission_evidence()
    if evidence != expected:
        raise EnvironmentBuildError(label + " differs")
    return evidence


def _stat_identity(metadata: os.stat_result) -> Tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _inode_identity(metadata: os.stat_result) -> Tuple[int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
    )


def _open_absolute_nofollow(path: Path, *, directory: bool, label: str) -> int:
    absolute = _lexical_absolute(str(path), label)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    cloexec_flag = getattr(os, "O_CLOEXEC", 0)
    nonblock_flag = getattr(os, "O_NONBLOCK", 0)
    current = os.open(
        absolute.anchor,
        os.O_RDONLY | directory_flag | cloexec_flag,
    )
    try:
        parts = absolute.parts[1:]
        for index, part in enumerate(parts):
            final = index == len(parts) - 1
            flags = os.O_RDONLY | nofollow_flag | cloexec_flag | nonblock_flag
            if not final or directory:
                flags |= directory_flag
            next_descriptor = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_descriptor
        metadata = os.fstat(current)
        if directory and not stat.S_ISDIR(metadata.st_mode):
            raise EnvironmentBuildError(label + " is not a directory")
        if not directory and not stat.S_ISREG(metadata.st_mode):
            raise EnvironmentBuildError(label + " is not a regular file")
        return current
    except OSError as error:
        os.close(current)
        raise EnvironmentBuildError(
            label + " cannot be opened without symlinks"
        ) from error
    except BaseException:
        os.close(current)
        raise


def _revalidate_fd_path(
    path: Path, descriptor: int, *, directory: bool, label: str
) -> None:
    before = _inode_identity(os.fstat(descriptor))
    reopened = _open_absolute_nofollow(path, directory=directory, label=label)
    try:
        if _inode_identity(os.fstat(reopened)) != before:
            raise EnvironmentBuildError(label + " path identity changed")
    finally:
        os.close(reopened)


def _read_bytes_file(path: Path, label: str) -> bytes:
    descriptor = _open_absolute_nofollow(path, directory=False, label=label)
    try:
        before = _stat_identity(os.fstat(descriptor))
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        if _stat_identity(os.fstat(descriptor)) != before:
            raise EnvironmentBuildError(label + " changed while being read")
        _revalidate_fd_path(path, descriptor, directory=False, label=label)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_bytes_file(path, "SHA-256 input"))


def _read_text_file(path: Path, label: str) -> str:
    try:
        return _read_bytes_file(path, label).decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise EnvironmentBuildError(label + " is not valid UTF-8") from error


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _lexical_absolute(value: str, label: str) -> Path:
    if not value:
        raise EnvironmentBuildError(label + " must be non-empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise EnvironmentBuildError(label + " must be absolute")
    return Path(os.path.normpath(str(path)))


def _lstat(path: Path) -> Optional[os.stat_result]:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _assert_no_symlink_components(path: Path, label: str) -> None:
    absolute = _lexical_absolute(str(path), label)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        metadata = _lstat(current)
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise EnvironmentBuildError(label + " traverses a symlink: " + str(current))


def _assert_no_symlink_parent_components(path: Path, label: str) -> None:
    absolute = _lexical_absolute(str(path), label)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:-1]:
        current = current / part
        metadata = _lstat(current)
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise EnvironmentBuildError(
                label + " traverses a parent symlink: " + str(current)
            )


def _require_python_launcher(path: Path, label: str) -> Tuple[Path, Dict[str, object]]:
    """Bind a regular launcher or a controlled leaf-symlink chain.

    Conda commonly exposes ``bin/python`` as a leaf symlink.  The lexical path
    must be executed so Python discovers that Conda environment's prefix, while
    every link and the final regular executable remain content-bound.
    """

    launcher = _lexical_absolute(str(path), label)
    chain: List[Dict[str, object]] = []
    visited = set()
    current = launcher
    for _depth in range(16):
        _assert_no_symlink_parent_components(current, label)
        if str(current) in visited:
            raise EnvironmentBuildError(label + " symlink chain contains a cycle")
        visited.add(str(current))
        metadata = _lstat(current)
        if metadata is None:
            raise EnvironmentBuildError(label + " does not exist: " + str(current))
        if stat.S_ISREG(metadata.st_mode):
            if not metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                raise EnvironmentBuildError(label + " target is not executable")
            resolved_sha = _sha256_file(current)
            confirmed = _lstat(current)
            if confirmed is None or _stat_identity(confirmed) != _stat_identity(
                metadata
            ):
                raise EnvironmentBuildError(label + " executable identity changed")
            return launcher, {
                "launcher": str(launcher),
                "launcher_kind": "leaf_symlink_chain" if chain else "regular_file",
                "symlink_chain": chain,
                "resolved_executable": str(current),
                "resolved_executable_sha256": resolved_sha,
                "resolved_executable_device": int(metadata.st_dev),
                "resolved_executable_inode": int(metadata.st_ino),
            }
        if not stat.S_ISLNK(metadata.st_mode):
            raise EnvironmentBuildError(
                label + " must resolve to a regular executable file"
            )
        before_link = _stat_identity(metadata)
        raw_target = os.readlink(str(current))
        after_link = _lstat(current)
        if after_link is None or _stat_identity(after_link) != before_link:
            raise EnvironmentBuildError(label + " symlink identity changed")
        if not raw_target:
            raise EnvironmentBuildError(label + " has an empty symlink target")
        target = Path(raw_target)
        if not target.is_absolute():
            target = current.parent / target
        target = _lexical_absolute(str(target), label + " symlink target")
        chain.append(
            {
                "path": str(current),
                "link_target": raw_target,
                "next_path": str(target),
                "device": int(metadata.st_dev),
                "inode": int(metadata.st_ino),
                "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
            }
        )
        current = target
    raise EnvironmentBuildError(label + " symlink chain is too deep")


def _launcher_sha256(path: Path, label: str) -> str:
    _launcher, identity = _require_python_launcher(path, label)
    value = identity.get("resolved_executable_sha256")
    if not _is_sha256(value):
        raise EnvironmentBuildError(label + " lacks a resolved SHA-256")
    return str(value)


def _require_regular_file(path: Path, label: str) -> Path:
    lexical = _lexical_absolute(str(path), label)
    descriptor = _open_absolute_nofollow(lexical, directory=False, label=label)
    try:
        _revalidate_fd_path(lexical, descriptor, directory=False, label=label)
    finally:
        os.close(descriptor)
    return lexical


def _require_read_only_file(path: Path, label: str) -> Path:
    regular = _require_regular_file(path, label)
    descriptor = _open_absolute_nofollow(regular, directory=False, label=label)
    try:
        if os.fstat(descriptor).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise EnvironmentBuildError(label + " must be read-only")
        _revalidate_fd_path(regular, descriptor, directory=False, label=label)
    finally:
        os.close(descriptor)
    return regular


def _require_directory(path: Path, label: str) -> Path:
    lexical = _lexical_absolute(str(path), label)
    descriptor = _open_absolute_nofollow(lexical, directory=True, label=label)
    try:
        _revalidate_fd_path(lexical, descriptor, directory=True, label=label)
    finally:
        os.close(descriptor)
    return lexical


def _require_read_only_directory(path: Path, label: str) -> Path:
    directory = _require_directory(path, label)
    descriptor = _open_absolute_nofollow(directory, directory=True, label=label)
    try:
        if os.fstat(descriptor).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise EnvironmentBuildError(label + " must be read-only")
        _revalidate_fd_path(directory, descriptor, directory=True, label=label)
    finally:
        os.close(descriptor)
    return directory


def _require_exact_mode(
    path: Path, expected: int, *, directory: bool, label: str
) -> None:
    descriptor = _open_absolute_nofollow(path, directory=directory, label=label)
    try:
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != expected:
            raise EnvironmentBuildError(label + " mode differs")
        _revalidate_fd_path(path, descriptor, directory=directory, label=label)
    finally:
        os.close(descriptor)


def _chmod_nofollow(path: Path, mode: int, *, directory: bool, label: str) -> None:
    descriptor = _open_absolute_nofollow(path, directory=directory, label=label)
    try:
        before = _inode_identity(os.fstat(descriptor))
        os.fchmod(descriptor, mode)
        after = os.fstat(descriptor)
        if (
            _inode_identity(after)[:2] != before[:2]
            or stat.S_IMODE(after.st_mode) != mode
        ):
            raise EnvironmentBuildError(label + " mode sealing failed")
        _revalidate_fd_path(path, descriptor, directory=directory, label=label)
    finally:
        os.close(descriptor)


def _directory_identity(path: Path, label: str) -> Dict[str, int]:
    descriptor = _open_absolute_nofollow(path, directory=True, label=label)
    try:
        metadata = os.fstat(descriptor)
        _revalidate_fd_path(path, descriptor, directory=True, label=label)
        return {"device": int(metadata.st_dev), "inode": int(metadata.st_ino)}
    finally:
        os.close(descriptor)


def _revalidate_directory_identity(
    path: Path, expected: Mapping[str, object], label: str
) -> None:
    if _directory_identity(path, label) != dict(expected):
        raise EnvironmentBuildError(label + " identity changed")


class _ExecutionAnchor:
    """Hold one directory inode across every child that consumes it.

    A lexical path can be renamed and replaced after it has been checked.  Child
    processes therefore receive a descriptor-backed path and inherit the held
    descriptor explicitly.  The lexical name is still revalidated before and
    after each child so a rename is fail-closed even though the child never
    follows the replacement.
    """

    def __init__(self, root: Path, label: str) -> None:
        self.root = _require_directory(root, label)
        self.label = label
        self.descriptor = _open_absolute_nofollow(
            self.root, directory=True, label=label
        )
        self.owner_pid = os.getpid()
        self.identity = _inode_identity(os.fstat(self.descriptor))
        self.closed = False

    def validate(self) -> None:
        if self.closed:
            raise EnvironmentBuildError(self.label + " execution anchor is closed")
        if os.getpid() != self.owner_pid:
            raise EnvironmentBuildError(
                self.label + " execution anchor crossed a process boundary"
            )
        try:
            metadata = os.fstat(self.descriptor)
        except OSError as error:
            raise EnvironmentBuildError(
                self.label + " held descriptor is unavailable"
            ) from error
        if _inode_identity(metadata) != self.identity:
            raise EnvironmentBuildError(self.label + " held inode changed")
        _revalidate_fd_path(
            self.root,
            self.descriptor,
            directory=True,
            label=self.label + " execution anchor",
        )

    def child_path(self, path: Path, label: str) -> Path:
        target = _lexical_absolute(str(path), label)
        try:
            relative = target.relative_to(self.root)
        except ValueError as error:
            raise EnvironmentBuildError(
                label + " is outside the execution anchor"
            ) from error
        descriptor_namespace = (
            "/proc/{}/fd".format(self.owner_pid)
            if sys.platform.startswith("linux")
            else "/dev/fd"
        )
        descriptor_root = Path(descriptor_namespace) / str(self.descriptor)
        return descriptor_root / relative

    def close(self) -> None:
        if not self.closed:
            os.close(self.descriptor)
            self.closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass


def _read_fd_payload(descriptor: int, label: str) -> Tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise EnvironmentBuildError(label + " changed while being read")
    return b"".join(chunks), after


def _replace_dirfd_file(
    directory_descriptor: int,
    name: str,
    payload: bytes,
    mode: int,
    label: str,
) -> None:
    if name in {"", ".", ".."} or "/" in name:
        raise EnvironmentBuildError(label + " has an invalid entry name")
    token = hashlib.sha256(
        (name + "\0" + str(os.getpid())).encode("utf-8")
    ).hexdigest()[:20]
    temporary = ".rachel-env-rewrite-" + token
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            temporary, flags, stat.S_IMODE(mode), dir_fd=directory_descriptor
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise EnvironmentBuildError(label + " replacement write stalled")
            offset += written
        os.fchmod(descriptor, stat.S_IMODE(mode))
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    except OSError as error:
        raise EnvironmentBuildError(label + " cannot be replaced safely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass


def _normalize_and_audit_environment_scripts(
    root: Path,
    execution_anchor: _ExecutionAnchor,
    *,
    rewrite_descriptor_paths: bool,
    descriptor_path_replacements: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    """Remove ephemeral FD paths from generated scripts, then bind them.

    ``venv`` and pip may embed the descriptor-backed construction path in
    console-script shebangs or activation helpers.  These files are generated
    locally, so they are rewritten through the held ``bin`` dirfd before any
    inventory is frozen.  No descriptor namespace may remain afterwards.
    """

    execution_anchor.validate()
    bin_path = root / "bin"
    try:
        bin_descriptor = os.open(
            "bin",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=execution_anchor.descriptor,
        )
    except OSError as error:
        raise EnvironmentBuildError(
            "environment bin cannot be opened through the held root"
        ) from error
    entries = []
    replacements = {
        str(
            execution_anchor.child_path(root, "environment script descriptor prefix")
        ): str(root),
    }
    if descriptor_path_replacements is not None:
        for source, destination in descriptor_path_replacements.items():
            if (
                re.fullmatch(r"/(?:proc/(?:self|[0-9]+)|dev)/fd/[0-9]+(?:/.*)?", source)
                is None
                or not Path(destination).is_absolute()
            ):
                raise EnvironmentBuildError(
                    "environment script descriptor replacement is malformed"
                )
            replacements[source] = destination
    try:
        for name in sorted(os.listdir(bin_descriptor)):
            if name in {"", ".", ".."} or "/" in name:
                raise EnvironmentBuildError("environment bin contains an invalid entry")
            metadata = os.stat(name, dir_fd=bin_descriptor, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(name, dir_fd=bin_descriptor)
                if re.search(r"/(?:proc/(?:self|[0-9]+)|dev)/fd/[0-9]+", target):
                    raise EnvironmentBuildError(
                        "environment bin symlink retains an ephemeral descriptor path"
                    )
                entries.append(
                    {
                        "path": str(bin_path / name),
                        "kind": "symlink",
                        "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
                        "link_target": target,
                    }
                )
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise EnvironmentBuildError("environment bin contains a non-file entry")
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=bin_descriptor,
                )
            except OSError as error:
                raise EnvironmentBuildError(
                    "environment bin entry cannot be opened without symlinks"
                ) from error
            try:
                payload, confirmed = _read_fd_payload(
                    descriptor, "environment bin entry"
                )
            finally:
                os.close(descriptor)
            rewritten = payload
            if rewrite_descriptor_paths:
                for source, destination in replacements.items():
                    rewritten = rewritten.replace(
                        source.encode("utf-8"), destination.encode("utf-8")
                    )
                if rewritten != payload:
                    _replace_dirfd_file(
                        bin_descriptor,
                        name,
                        rewritten,
                        confirmed.st_mode,
                        "environment generated script",
                    )
                    payload = rewritten
                    confirmed = os.stat(
                        name, dir_fd=bin_descriptor, follow_symlinks=False
                    )
            if re.search(rb"/(?:proc/(?:self|[0-9]+)|dev)/fd/[0-9]+", payload):
                raise EnvironmentBuildError(
                    "environment script retains an ephemeral descriptor path"
                )
            entries.append(
                {
                    "path": str(bin_path / name),
                    "kind": "file",
                    "mode": format(stat.S_IMODE(confirmed.st_mode), "04o"),
                    "size": len(payload),
                    "sha256": _sha256_bytes(payload),
                    "executable": bool(
                        confirmed.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    ),
                }
            )
        execution_anchor.validate()
    except OSError as error:
        raise EnvironmentBuildError("environment bin audit failed") from error
    finally:
        os.close(bin_descriptor)
    return _content_bound(
        {
            "schema_version": "rachel-benchmark-bin-script-inventory/1.0",
            "environment_root": str(root),
            "entries": entries,
            "ephemeral_descriptor_paths_present": False,
        }
    )


def _normalize_target_pyvenv_configuration(
    root: Path,
    execution_anchor: _ExecutionAnchor,
    descriptor_path_replacements: Optional[Mapping[str, str]] = None,
) -> None:
    """Replace the ephemeral construction path in generated pyvenv.cfg."""

    execution_anchor.validate()
    try:
        descriptor = os.open(
            "pyvenv.cfg",
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=execution_anchor.descriptor,
        )
    except OSError as error:
        raise EnvironmentBuildError(
            "target pyvenv.cfg cannot be opened through the held root"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise EnvironmentBuildError("target pyvenv.cfg is not a regular file")
        payload, confirmed = _read_fd_payload(descriptor, "target pyvenv.cfg")
    finally:
        os.close(descriptor)
    replacements = {
        str(execution_anchor.child_path(root, "target pyvenv descriptor prefix")): str(
            root
        )
    }
    if descriptor_path_replacements is not None:
        replacements.update(descriptor_path_replacements)
    normalized = payload
    for source, destination in replacements.items():
        if (
            re.fullmatch(r"/(?:proc/(?:self|[0-9]+)|dev)/fd/[0-9]+(?:/.*)?", source)
            is None
            or not Path(destination).is_absolute()
        ):
            raise EnvironmentBuildError(
                "target pyvenv descriptor replacement is malformed"
            )
        normalized = normalized.replace(
            source.encode("utf-8"), destination.encode("utf-8")
        )
    if normalized != payload:
        _replace_dirfd_file(
            execution_anchor.descriptor,
            "pyvenv.cfg",
            normalized,
            confirmed.st_mode,
            "target pyvenv.cfg",
        )
    if re.search(rb"/(?:proc/(?:self|[0-9]+)|dev)/fd/[0-9]+", normalized):
        raise EnvironmentBuildError(
            "target pyvenv.cfg retains an ephemeral descriptor path"
        )
    execution_anchor.validate()


def _list_regular_files(path: Path, label: str) -> List[Path]:
    directory = _require_directory(path, label)
    descriptor = _open_absolute_nofollow(directory, directory=True, label=label)
    result = []
    try:
        before = _inode_identity(os.fstat(descriptor))
        for name in sorted(os.listdir(descriptor)):
            if name in {"", ".", ".."} or "/" in name:
                raise EnvironmentBuildError(label + " contains an invalid entry")
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=descriptor,
            )
            try:
                if not stat.S_ISREG(os.fstat(child).st_mode):
                    raise EnvironmentBuildError(label + " contains a non-regular entry")
            finally:
                os.close(child)
            result.append(directory / name)
        if _inode_identity(os.fstat(descriptor)) != before:
            raise EnvironmentBuildError(label + " changed while being listed")
        _revalidate_fd_path(directory, descriptor, directory=True, label=label)
    except OSError as error:
        raise EnvironmentBuildError(label + " cannot be listed safely") from error
    finally:
        os.close(descriptor)
    return result


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _mkdir_new_anchored(path: Path, label: str, mode: int = 0o755) -> Path:
    target = _lexical_absolute(str(path), label)
    parent = _require_directory(target.parent, label + " parent")
    parent_descriptor = _open_absolute_nofollow(
        parent, directory=True, label=label + " parent"
    )
    created = False
    try:
        _revalidate_fd_path(
            parent, parent_descriptor, directory=True, label=label + " parent"
        )
        try:
            os.stat(target.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise EnvironmentBuildError(label + " must be fresh")
        try:
            os.mkdir(target.name, mode=mode, dir_fd=parent_descriptor)
            created = True
        except FileExistsError as error:
            raise EnvironmentBuildError(label + " appeared during claim") from error
        child_descriptor = os.open(
            target.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            if not stat.S_ISDIR(os.fstat(child_descriptor).st_mode):
                raise EnvironmentBuildError(label + " claim is not a directory")
            _revalidate_fd_path(target, child_descriptor, directory=True, label=label)
        finally:
            os.close(child_descriptor)
        _revalidate_fd_path(
            parent, parent_descriptor, directory=True, label=label + " parent"
        )
        return target
    except BaseException:
        if created:
            try:
                os.rmdir(target.name, dir_fd=parent_descriptor)
            except OSError:
                pass
        raise
    finally:
        os.close(parent_descriptor)


def _command_environment() -> Dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("PIP_") or name.startswith("PYTHON"):
            environment.pop(name, None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_NO_INPUT": "1",
        }
    )
    environment.pop("PYTHONPATH", None)
    return environment


def _run_checked(
    arguments: Sequence[str],
    *,
    timeout: int = 600,
    label: str,
    stdin_text: Optional[str] = None,
    execution_anchor: Optional[_ExecutionAnchor] = None,
    additional_execution_anchors: Sequence[_ExecutionAnchor] = (),
) -> subprocess.CompletedProcess:
    anchors = list(additional_execution_anchors)
    if execution_anchor is not None:
        anchors.insert(0, execution_anchor)
    unique_anchors = []
    seen_descriptors = set()
    for anchor in anchors:
        if anchor.descriptor not in seen_descriptors:
            anchor.validate()
            unique_anchors.append(anchor)
            seen_descriptors.add(anchor.descriptor)
    pass_fds = tuple(anchor.descriptor for anchor in unique_anchors)
    run_arguments: Dict[str, object] = {
        "check": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "timeout": timeout,
        "env": _command_environment(),
        "pass_fds": pass_fds,
    }
    if stdin_text is not None:
        run_arguments["input"] = stdin_text
    try:
        completed = subprocess.run(tuple(arguments), **run_arguments)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        for anchor in unique_anchors:
            anchor.validate()
        raise EnvironmentBuildError(label + " failed") from error
    for anchor in unique_anchors:
        anchor.validate()
    return completed


QUEUE_LIVE_PROBE_CODE = r"""import importlib, importlib.metadata as m, json, os, site, sys
names = ("torch", "torch-geometric", "opencv-python", "opencv-python-headless", "numpy", "scipy")
packages = {}
for name in names:
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: packages[name] = None
imports = {}
for name in ("torch", "torch_geometric", "cv2", "numpy", "scipy"):
    try:
        importlib.import_module(name)
        imports[name] = "ok"
    except Exception as error:
        imports[name] = type(error).__name__ + ":" + str(error)
print(json.dumps({
    "executable": os.path.realpath(sys.executable),
    "prefix": os.path.realpath(sys.prefix),
    "base_prefix": os.path.realpath(sys.base_prefix),
    "python_version": sys.version.split()[0],
    "user_site_enabled": bool(site.ENABLE_USER_SITE),
    "packages": packages,
    "imports": imports,
}, sort_keys=True))"""


SAFE_RUNTIME_PROBE_CODE = r"""import json, os, platform, sys
print(json.dumps({
    "executable": os.path.realpath(sys.executable),
    "prefix": os.path.realpath(sys.prefix),
    "base_prefix": os.path.realpath(sys.base_prefix),
    "python_version": sys.version.split()[0],
    "isolated": bool(sys.flags.isolated),
    "no_site": bool(sys.flags.no_site),
    "dont_write_bytecode": bool(sys.dont_write_bytecode),
    "platform_system": platform.system(),
    "platform_machine": platform.machine(),
    "sys_path": [os.path.realpath(value) for value in sys.path if value],
}, sort_keys=True))"""


DETAILED_PROBE_CODE = r"""import importlib, importlib.metadata as m, json, os, platform, re, site, sys
pairs = json.loads(sys.argv[1])
details = {}
for distribution_name, module_name in pairs:
    module = importlib.import_module(module_name)
    distribution = m.distribution(distribution_name)
    origin = getattr(module, "__file__", None)
    if not origin:
        raise RuntimeError("module lacks __file__: " + module_name)
    details[distribution_name] = {
        "distribution_version": distribution.version,
        "distribution_location": os.path.realpath(str(distribution.locate_file(""))),
        "module": module_name,
        "module_version": str(getattr(module, "__version__", distribution.version)),
        "import_origin": os.path.realpath(origin),
    }
local = []
prefix = os.path.realpath(sys.prefix)
for distribution in m.distributions():
    location = os.path.realpath(str(distribution.locate_file("")))
    try:
        inside = os.path.commonpath((location, prefix)) == prefix
    except ValueError:
        inside = False
    if inside:
        name = distribution.metadata.get("Name") or ""
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        local.append({"name": normalized, "version": distribution.version, "location": location})
try:
    headless = m.version("opencv-python-headless")
except m.PackageNotFoundError:
    headless = None
torch = importlib.import_module("torch")
print(json.dumps({
    "executable": os.path.realpath(sys.executable),
    "prefix": prefix,
    "base_prefix": os.path.realpath(sys.base_prefix),
    "python_version": sys.version.split()[0],
    "user_site_enabled": bool(site.ENABLE_USER_SITE),
    "sitecustomize_loaded": "sitecustomize" in sys.modules,
    "usercustomize_loaded": "usercustomize" in sys.modules,
    "sys_path": [os.path.realpath(value) for value in sys.path if value],
    "platform_system": platform.system(),
    "platform_machine": platform.machine(),
    "torch_module_version": str(torch.__version__),
    "torch_cuda_build_version": str(torch.version.cuda),
    "opencv_python_headless_version": headless,
    "tracked": details,
    "local_distributions": sorted(local, key=lambda value: (value["name"], value["version"], value["location"])),
    "torchvision_dynamic_module_admission": globals().get("__rachel_torchvision_dynamic_admission__"),
}, sort_keys=True))"""


FUNCTIONAL_CPU_PROBE_CODE = r"""import json, sys
import cv2
import numpy as np
from PIL import Image
import scipy.optimize
import torch
import torchvision
from torch_geometric.nn import DeepGCNLayer, GATConv

if str(torch.__version__) != "2.5.1+cu124":
    raise RuntimeError("unexpected Torch module version")
if str(torch.version.cuda) != "12.4":
    raise RuntimeError("unexpected Torch CUDA build version")
torch.random.default_generator.manual_seed(260831)
x = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 20.0
edge_index = torch.tensor([[0, 1, 2, 3, 4, 0, 2, 4], [1, 2, 3, 4, 0, 2, 4, 1]], dtype=torch.long)
gat = GATConv(4, 8, heads=1, concat=True, dropout=0.0).cpu().eval()
with torch.no_grad():
    y = gat(x, edge_index)
deep = DeepGCNLayer(
    conv=GATConv(8, 8, heads=1, concat=True, dropout=0.0),
    norm=torch.nn.LayerNorm(8),
    act=torch.nn.ReLU(inplace=False),
    block="res+",
    dropout=0.0,
    ckpt_grad=False,
).cpu().eval()
with torch.no_grad():
    z = deep(y, edge_index)
if list(y.shape) != [5, 8] or list(z.shape) != [5, 8] or not torch.isfinite(z).all():
    raise RuntimeError("PyG CPU forward failed")
mask = np.zeros((16, 16), dtype=np.uint8)
mask[3:13, 4:12] = 255
contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if len(contours) != 1 or float(cv2.contourArea(contours[0])) <= 0.0:
    raise RuntimeError("OpenCV contour check failed")
source = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
target = source + np.asarray([2, 3], dtype=np.float32)
matrix, inliers = cv2.estimateAffinePartial2D(source, target)
if matrix is None or inliers is None:
    raise RuntimeError("OpenCV affine check failed")
rows, columns = scipy.optimize.linear_sum_assignment(np.asarray([[3.0, 1.0], [1.0, 3.0]]))
if rows.tolist() != [0, 1] or columns.tolist() != [1, 0]:
    raise RuntimeError("SciPy assignment check failed")
pil_image = Image.fromarray(mask)
if pil_image.size != (16, 16):
    raise RuntimeError("Pillow image check failed")
normalized = torchvision.transforms.functional.normalize(
    torch.ones((3, 2, 2), dtype=torch.float32), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
)
if list(normalized.shape) != [3, 2, 2] or float(normalized.sum()) != 12.0:
    raise RuntimeError("torchvision CPU check failed")
print(json.dumps({
    "cuda_tensor_created": False,
    "device": "cpu",
    "torch_linear_shape": [5, 8],
    "pyg_gat_shape": list(y.shape),
    "pyg_deepgcn_shape": list(z.shape),
    "pyg_output_checksum": format(float(z.sum()), ".9g"),
    "opencv_contour_count": len(contours),
    "opencv_affine_translation": [format(float(matrix[0, 2]), ".9g"), format(float(matrix[1, 2]), ".9g")],
    "scipy_assignment": columns.tolist(),
    "pillow_size": list(pil_image.size),
    "torchvision_version": str(torchvision.__version__),
    "torchvision_dynamic_module_admission": globals().get("__rachel_torchvision_dynamic_admission__"),
}, sort_keys=True))"""


INVENTORY_PROBE_CODE = r"""import base64, csv, hashlib, importlib, importlib.metadata as m, io, json, os, stat, sys
pairs = json.loads(sys.argv[1])
environment_root = os.path.realpath(sys.argv[2])
hash_cache = {}

def open_regular(path):
    real = os.path.abspath(path)
    if not os.path.isabs(real):
        raise RuntimeError("package hash path is not absolute")
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    cloexec_flag = getattr(os, "O_CLOEXEC", 0)
    nonblock_flag = getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(os.path.sep, os.O_RDONLY | directory_flag | cloexec_flag)
    try:
        parts = [part for part in real.split(os.path.sep) if part]
        for index, part in enumerate(parts):
            final = index == len(parts) - 1
            flags = os.O_RDONLY | nofollow_flag | cloexec_flag | nonblock_flag
            if not final:
                flags |= directory_flag
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("package hash input is not regular: " + real)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise

def sha_file(path):
    real = os.path.realpath(path)
    descriptor = open_regular(real)
    before = os.fstat(descriptor)
    signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    cached = hash_cache.get(real)
    if cached is not None and cached[0] == signature:
        os.close(descriptor)
        return cached[1]
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk: break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if signature != after_signature:
        raise RuntimeError("package file changed during hashing: " + real)
    confirmed = open_regular(real)
    try:
        reopened = os.fstat(confirmed)
        if (reopened.st_dev, reopened.st_ino, reopened.st_mode) != (before.st_dev, before.st_ino, before.st_mode):
            raise RuntimeError("package file path identity changed: " + real)
    finally:
        os.close(confirmed)
    value = digest.hexdigest()
    hash_cache[real] = (signature, value)
    return value

def read_text(path):
    real = os.path.realpath(path)
    descriptor = open_regular(real)
    chunks = []
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError("package text file changed during read: " + real)
    finally:
        os.close(descriptor)
    sha_file(real)
    return b"".join(chunks).decode("utf-8", errors="strict")

def entry(path, relative):
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        resolved = os.path.realpath(path)
        if not os.path.isfile(resolved):
            raise RuntimeError("package symlink target is not a regular file: " + path)
        return {"path": relative, "kind": "symlink", "link_target": target, "resolved": resolved, "size": os.path.getsize(resolved), "sha256": sha_file(resolved)}
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("non-regular package entry: " + path)
    return {"path": relative, "kind": "file", "size": metadata.st_size, "sha256": sha_file(path)}

def tree(root):
    entries = []
    if os.path.isfile(root) or os.path.islink(root):
        entries.append(entry(root, os.path.basename(root)))
    else:
        for current, directories, files in os.walk(root, topdown=True, followlinks=False):
            kept = []
            for name in sorted(directories):
                full = os.path.join(current, name)
                relative = os.path.relpath(full, root).replace(os.sep, "/")
                if os.path.islink(full):
                    entries.append(entry(full, relative))
                else:
                    kept.append(name)
            directories[:] = kept
            for name in sorted(files):
                full = os.path.join(current, name)
                relative = os.path.relpath(full, root).replace(os.sep, "/")
                entries.append(entry(full, relative))
    entries.sort(key=lambda value: value["path"])
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"entries": entries, "file_count": len(entries), "byte_count": sum(value["size"] for value in entries), "content_sha256": hashlib.sha256(payload).hexdigest()}

tracked = {}
for distribution_name, module_name in pairs:
    module = importlib.import_module(module_name)
    origin = os.path.realpath(module.__file__)
    package_root = os.path.dirname(origin) if hasattr(module, "__path__") else origin
    distribution = m.distribution(distribution_name)
    distribution_location = os.path.realpath(str(distribution.locate_file("")))
    record_candidates = []
    for item in distribution.files or []:
        normalized = str(item).replace(os.sep, "/")
        if normalized.endswith(".dist-info/RECORD"):
            record_candidates.append(os.path.realpath(str(distribution.locate_file(item))))
    if len(record_candidates) != 1 or not os.path.isfile(record_candidates[0]):
        raise RuntimeError("distribution has no unique RECORD: " + distribution_name)
    record_path = record_candidates[0]
    verified_hashes = 0
    declared_hashes = 0
    mismatched_declared_hashes = 0
    site_package_entries = []
    rows = 0
    with io.StringIO(read_text(record_path), newline="") as stream:
        for relative, declared, declared_size in csv.reader(stream):
            rows += 1
            if not declared:
                continue
            declared_hashes += 1
            algorithm, encoded = declared.split("=", 1)
            if algorithm != "sha256":
                raise RuntimeError("non-SHA256 RECORD entry: " + distribution_name)
            candidate = os.path.realpath(os.path.join(os.path.dirname(record_path), "..", relative))
            if not os.path.isfile(candidate):
                raise RuntimeError("missing RECORD file: " + candidate)
            try:
                inside_site_packages = os.path.commonpath((candidate, distribution_location)) == distribution_location
            except ValueError:
                inside_site_packages = False
            observed_hex = sha_file(candidate)
            observed = base64.urlsafe_b64encode(bytes.fromhex(observed_hex)).decode("ascii").rstrip("=")
            size_matches = not declared_size or int(declared_size) == os.path.getsize(candidate)
            matches = observed == encoded and size_matches
            if matches:
                verified_hashes += 1
            else:
                mismatched_declared_hashes += 1
                if (
                    inside_site_packages
                    and os.path.commonpath((record_path, environment_root))
                    == environment_root
                ):
                    raise RuntimeError("locally installed RECORD content differs: " + candidate)
            if inside_site_packages:
                site_package_entries.append({"path": os.path.relpath(candidate, distribution_location).replace(os.sep, "/"), "size": os.path.getsize(candidate), "sha256": observed_hex, "record_hash_matches": matches})
    site_package_entries.sort(key=lambda value: value["path"])
    site_payload = json.dumps(site_package_entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    tracked[distribution_name] = {
        "distribution_version": distribution.version,
        "distribution_location": distribution_location,
        "module": module_name,
        "import_origin": origin,
        "package_root": package_root,
        "module_tree": tree(package_root),
        "dist_info_root": os.path.dirname(record_path),
        "record_path": record_path,
        "record_sha256": sha_file(record_path),
        "record_rows": rows,
        "record_declared_sha256_rows": declared_hashes,
        "record_verified_sha256_rows": verified_hashes,
        "record_mismatched_sha256_rows": mismatched_declared_hashes,
        "site_package_record_tree": {"entries": site_package_entries, "file_count": len(site_package_entries), "byte_count": sum(value["size"] for value in site_package_entries), "content_sha256": hashlib.sha256(site_payload).hexdigest()},
    }
inventory = {
    "schema_version": "rachel-benchmark-import-inventory/1.0",
    "environment_root": environment_root,
    "python": os.path.realpath(sys.executable),
    "python_sha256": sha_file(os.path.realpath(sys.executable)),
    "prefix": os.path.realpath(sys.prefix),
    "base_prefix": os.path.realpath(sys.base_prefix),
    "tracked": tracked,
    "torchvision_dynamic_module_admission": globals().get("__rachel_torchvision_dynamic_admission__"),
}
inventory["content_sha256"] = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
print(json.dumps(inventory, sort_keys=True, separators=(",", ":"), allow_nan=False))"""


EXPLICIT_SITE_DESCRIPTOR_GUARD_CODE = r"""def _held_site_descriptor(binding):
    descriptor_path = binding["descriptor_path"]
    if not isinstance(descriptor_path, str) or not descriptor_path:
        raise RuntimeError("explicit-site descriptor path is malformed")
    if sys.platform.startswith("linux"):
        pieces = descriptor_path.split("/")
        if len(pieces) != 5 or pieces[:2] != ["", "proc"] or pieces[3] != "fd":
            raise RuntimeError("explicit-site descriptor is not a reviewed procfs FD path")
        owner_text, descriptor_text = pieces[2], pieces[4]
        if (
            not owner_text.isdigit()
            or str(int(owner_text)) != owner_text
            or int(owner_text) != os.getppid()
        ):
            raise RuntimeError("explicit-site descriptor owner PID differs")
    elif sys.platform == "darwin":
        pieces = descriptor_path.split("/")
        if len(pieces) != 4 or pieces[:3] != ["", "dev", "fd"]:
            raise RuntimeError("explicit-site descriptor is not a reviewed devfs FD path")
        descriptor_text = pieces[3]
    else:
        raise RuntimeError("explicit-site descriptor platform is unsupported")
    if (
        not descriptor_text.isdigit()
        or str(int(descriptor_text)) != descriptor_text
        or int(descriptor_text) < 3
    ):
        raise RuntimeError("explicit-site descriptor FD differs")
    descriptor_number = int(descriptor_text)
    try:
        inherited = os.fstat(descriptor_number)
    except OSError as error:
        raise RuntimeError("explicit-site descriptor FD is not inherited") from error
    if (
        not stat.S_ISDIR(inherited.st_mode)
        or inherited.st_dev != binding["device"]
        or inherited.st_ino != binding["inode"]
    ):
        raise RuntimeError("explicit-site inherited directory identity differs")
    try:
        descriptor_leaf = os.stat(descriptor_path, follow_symlinks=False)
        if sys.platform.startswith("linux") and not stat.S_ISLNK(descriptor_leaf.st_mode):
            raise RuntimeError("explicit-site procfs descriptor is not a magic link")
        followed = os.stat(descriptor_path, follow_symlinks=True)
    except OSError as error:
        raise RuntimeError("explicit-site descriptor target cannot be inspected") from error
    if not stat.S_ISDIR(followed.st_mode) or followed.st_ino != inherited.st_ino:
        raise RuntimeError("explicit-site descriptor target differs from inherited FD")
    if sys.platform.startswith("linux") and followed.st_dev != inherited.st_dev:
        raise RuntimeError("explicit-site procfs target device differs from inherited FD")
    canonical_path = binding["canonical_path"]
    if (
        not isinstance(canonical_path, str)
        or not os.path.isabs(canonical_path)
        or os.path.normpath(canonical_path) != canonical_path
    ):
        raise RuntimeError("explicit-site canonical path is malformed")
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    cloexec_flag = getattr(os, "O_CLOEXEC", 0)
    canonical_descriptor = os.open(
        os.path.sep, os.O_RDONLY | directory_flag | cloexec_flag
    )
    try:
        for part in [part for part in canonical_path.split(os.path.sep) if part]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | directory_flag | nofollow_flag | cloexec_flag,
                dir_fd=canonical_descriptor,
            )
            os.close(canonical_descriptor)
            canonical_descriptor = next_descriptor
        canonical = os.fstat(canonical_descriptor)
    except OSError as error:
        raise RuntimeError(
            "explicit-site canonical directory cannot be opened without symlinks"
        ) from error
    finally:
        os.close(canonical_descriptor)
    if (
        not stat.S_ISDIR(canonical.st_mode)
        or canonical.st_dev != inherited.st_dev
        or canonical.st_ino != inherited.st_ino
    ):
        raise RuntimeError("explicit-site canonical directory identity differs")
    return descriptor_path
"""


EXPLICIT_SITE_STDIN_LAUNCHER_CODE = r"""import sys
if len(sys.argv) < 5:
    raise RuntimeError("explicit-site stdin launcher arguments are missing")
count_text = sys.argv[1]
if not count_text.isdigit() or str(int(count_text)) != count_text:
    raise RuntimeError("explicit-site no-site path count is malformed")
count = int(count_text)
if count < 1 or count > 64 or len(sys.argv) < count + 6:
    raise RuntimeError("explicit-site no-site path count differs")
expected_path = tuple(sys.argv[2:2 + count])
if any(not value or not value.startswith("/") for value in expected_path):
    raise RuntimeError("explicit-site reviewed startup path is malformed")
reviewed_path_sha256 = sys.argv[2 + count]
bootstrap_sha256 = sys.argv[3 + count]
remaining = sys.argv[4 + count:]
# Only the built-in sys module has been touched.  Replace, rather than filter,
# the launcher surface before any filesystem-backed import can occur.
sys.path[:] = expected_path
if tuple(sys.path) != expected_path:
    raise RuntimeError("explicit-site launcher could not construct reviewed path")
source = sys.stdin.buffer.read()
import hashlib, json
if tuple(sys.path) != expected_path:
    raise RuntimeError("explicit-site launcher import path changed")
observed_path_sha256 = hashlib.sha256(json.dumps(list(expected_path), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
if observed_path_sha256 != reviewed_path_sha256:
    raise RuntimeError("explicit-site reviewed startup path SHA-256 differs")
if hashlib.sha256(source).hexdigest() != bootstrap_sha256:
    raise RuntimeError("explicit-site stdin bootstrap SHA-256 differs")
sys.argv = ["<rachel-explicit-site-bootstrap>", *remaining]
namespace = {"__name__": "__main__", "__file__": "<rachel-explicit-site-bootstrap>"}
exec(compile(source, namespace["__file__"], "exec"), namespace, namespace)
"""


TORCHVISION_DYNAMIC_ADMISSION_BOOTSTRAP_CODE = r"""
_dynamic_configuration = configuration["torchvision_dynamic_admission"]
_expected_dynamic_authorities = {
    "torch_version": "2.5.1+cu124",
    "torchvision_version": "0.20.1+cu124",
    "instantiator_relative_path": "torch/distributed/nn/jit/instantiator.py",
    "instantiator_module": "torch.distributed.nn.jit.instantiator",
    "instantiator_source_sha256": "440a619c764e4133564d7956ba060a7223e94664854b94a4a2074d095756db7e",
    "instantiator_append_line": 23,
    "instantiator_append_source": "sys.path.append(INSTANTIATED_TEMPLATE_DIR_PATH)",
    "remote_module_relative_path": "torch/distributed/nn/api/remote_module.py",
    "remote_module_module": "torch.distributed.nn.api.remote_module",
    "remote_module_source_sha256": "55c9c44ba25a2b5edf105fbd740ceff771f937147d7d6a9d6232f05681e7eeaf",
    "template_relative_path": "torch/distributed/nn/jit/templates/remote_module_template.py",
    "template_module": "torch.distributed.nn.jit.templates.remote_module_template",
    "template_source_sha256": "0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc",
    "generated_module_name": "_remote_module_non_scriptable",
    "generated_module_filename": "_remote_module_non_scriptable.py",
    "generated_module_size": 2355,
    "generated_module_sha256": "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8",
    "preseed_origin": "<rachel-pinned-_remote_module_non_scriptable>",
}
_expected_dynamic_policy = {
    "controlled_sys_path_default_deny": True,
    "single_pinned_instantiator_append": True,
    "captured_directory_fd_held_through_payload": True,
    "captured_directory_identity": "regular_directory,euid_owner,mode_0700,stable_dev_inode",
    "captured_file_identity": "regular_file,euid_owner,mode_0644,nlink_1,stable_fd_hash",
    "captured_path_import_denied": True,
    "generated_write_fd_anchored": True,
    "preseed_module_memory_only": True,
    "captured_origin_compile_and_exec_denied": True,
    "pinned_modules_loaded_from_held_source_bytes": True,
    "pinned_module_bytecode_cache_disabled": True,
    "reviewed_sys_path_restored_before_payload": True,
}
if (
    not isinstance(_dynamic_configuration, dict)
    or set(_dynamic_configuration) != {"schema_version", "enabled", "authorities", "policy"}
    or _dynamic_configuration["schema_version"] != "rachel-benchmark-torchvision-dynamic-module-admission/1.0"
    or not isinstance(_dynamic_configuration["enabled"], bool)
    or _dynamic_configuration["authorities"] != _expected_dynamic_authorities
    or _dynamic_configuration["policy"] != _expected_dynamic_policy
):
    raise RuntimeError("torchvision dynamic admission configuration differs")

_dynamic_state = {
    "phase": "preload" if _dynamic_configuration["enabled"] else "locked",
    "append_count": 0,
    "restore_count": 0,
    "captured": None,
    "captured_finder_count": 0,
    "captured_compile_count": 0,
    "captured_exec_count": 0,
    "template_compile_count": 0,
    "template_exec_count": 0,
    "preseed_compile_count": 0,
    "preseed_exec_count": 0,
    "pinned_source_find_count": {},
    "pinned_source_compile_count": {},
    "pinned_source_exec_count": {},
    "pinned_source_loaders": {},
    "pinned_source_pyc_read_count": 0,
    "preseed": None,
    "pinned_sources": None,
}

def _dynamic_descriptor_number(binding):
    value = binding["descriptor_path"].rsplit("/", 1)[-1]
    if not value.isdigit() or str(int(value)) != value or int(value) < 3:
        raise RuntimeError("dynamic source site descriptor differs")
    descriptor = int(value)
    metadata = os.fstat(descriptor)
    if metadata.st_dev != binding["device"] or metadata.st_ino != binding["inode"]:
        raise RuntimeError("dynamic source site descriptor identity differs")
    return descriptor

def _dynamic_read_relative(binding, relative_path):
    if (
        not isinstance(relative_path, str)
        or relative_path.startswith("/")
        or "\\" in relative_path
        or any(part in {"", ".", ".."} for part in relative_path.split("/"))
    ):
        raise RuntimeError("dynamic source relative path differs")
    descriptor = os.dup(_dynamic_descriptor_number(binding))
    try:
        parts = relative_path.split("/")
        for index, part in enumerate(parts):
            final = index == len(parts) - 1
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            if not final:
                flags |= getattr(os, "O_DIRECTORY", 0)
            try:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                return None
            os.close(descriptor)
            descriptor = next_descriptor
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("dynamic pinned source is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError("dynamic pinned source changed during read")
        return {
            "bytes": b"".join(chunks),
            "descriptor_path": binding["descriptor_path"] + "/" + relative_path,
            "canonical_path": binding["canonical_path"] + "/" + relative_path,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mode": before.st_mode,
            "size": before.st_size,
        }
    finally:
        os.close(descriptor)

def _dynamic_read_unique_pinned(relative_path, expected_sha256):
    matches = []
    for binding in configuration["sites"]:
        item = _dynamic_read_relative(binding, relative_path)
        if item is not None:
            matches.append(item)
    if len(matches) != 1:
        raise RuntimeError("dynamic pinned source is not unique: " + relative_path)
    item = matches[0]
    if hashlib.sha256(item["bytes"]).hexdigest() != expected_sha256:
        raise RuntimeError("dynamic pinned source SHA-256 differs: " + relative_path)
    return item

def _dynamic_origin_is_captured(filename):
    captured = _dynamic_state["captured"]
    if captured is None or not isinstance(filename, str):
        return False
    for root in (captured["descriptor_path"], captured["lexical_path"]):
        try:
            if os.path.commonpath((os.path.abspath(filename), root)) == root:
                return True
        except (OSError, ValueError):
            pass
    try:
        parent = os.stat(os.path.dirname(filename), follow_symlinks=False)
    except OSError:
        return False
    return (parent.st_dev, parent.st_ino) == (captured["device"], captured["inode"])

def _dynamic_is_pinned_bytecode_path(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    try:
        path = os.fsdecode(value)
    except (TypeError, ValueError):
        return False
    normalized = os.path.normpath(path)
    for binding in configuration["sites"]:
        for relative_key in (
            "instantiator_relative_path",
            "template_relative_path",
            "remote_module_relative_path",
        ):
            relative = _expected_dynamic_authorities[relative_key]
            parent = os.path.dirname(relative)
            stem = os.path.basename(relative)[:-3]
            for root in (binding["descriptor_path"], binding["canonical_path"]):
                legacy = os.path.normpath(os.path.join(root, parent, stem + ".pyc"))
                cache = os.path.normpath(os.path.join(root, parent, "__pycache__"))
                if normalized == legacy:
                    return True
                try:
                    if (
                        os.path.commonpath((normalized, cache)) == cache
                        and os.path.basename(normalized).startswith(stem + ".")
                        and os.path.basename(normalized).endswith(".pyc")
                    ):
                        return True
                except ValueError:
                    pass
    return False

def _dynamic_assert_path_surface():
    if sys.path is not _controlled_import_path:
        raise RuntimeError("explicit-site sys.path object was replaced")
    expected = reviewed_import_path
    if _dynamic_state["phase"] == "preload" and _dynamic_state["append_count"] == 1:
        expected = reviewed_import_path + (_dynamic_state["captured"]["descriptor_path"],)
    if tuple(sys.path) != expected:
        raise RuntimeError("explicit-site payload changed the reviewed import path")

def _dynamic_audit(event, arguments):
    if event == "open" and arguments and _dynamic_is_pinned_bytecode_path(arguments[0]):
        _dynamic_state["pinned_source_pyc_read_count"] += 1
        raise RuntimeError("dynamic pinned module bytecode read denied")
    if event == "import":
        _dynamic_assert_path_surface()
    elif event == "compile" and len(arguments) >= 2:
        source, filename = arguments[0], arguments[1]
        raw = source if isinstance(source, bytes) else str(source).encode("utf-8")
        if filename == "<rachel-pinned-remote-module-template>":
            _dynamic_state["template_compile_count"] += 1
            if hashlib.sha256(raw).hexdigest() != _expected_dynamic_authorities["template_source_sha256"]:
                raise RuntimeError("dynamic template pseudo compile bytes differ")
        elif filename == _expected_dynamic_authorities["preseed_origin"]:
            _dynamic_state["preseed_compile_count"] += 1
            if len(raw) != _expected_dynamic_authorities["generated_module_size"] or hashlib.sha256(raw).hexdigest() != _expected_dynamic_authorities["generated_module_sha256"]:
                raise RuntimeError("dynamic preseed compile bytes differ")
        elif _dynamic_origin_is_captured(filename):
            _dynamic_state["captured_compile_count"] += 1
            raise RuntimeError("compile from captured dynamic directory denied")
    elif event == "exec" and arguments:
        code = arguments[0]
        filename = getattr(code, "co_filename", None)
        if filename == "<rachel-pinned-remote-module-template>":
            _dynamic_state["template_exec_count"] += 1
        elif filename == _expected_dynamic_authorities["preseed_origin"]:
            _dynamic_state["preseed_exec_count"] += 1
            if getattr(code, "co_name", None) != "<module>":
                raise RuntimeError("dynamic preseed code object differs")
        elif _dynamic_origin_is_captured(filename):
            _dynamic_state["captured_exec_count"] += 1
            raise RuntimeError("exec from captured dynamic directory denied")

class _ControlledImportPath(list):
    def _deny(self, *_args, **_kwargs):
        raise RuntimeError("unreviewed sys.path mutation")

    def append(self, value):
        if _dynamic_state["phase"] != "preload" or _dynamic_state["append_count"] != 0:
            raise RuntimeError("unreviewed sys.path append")
        caller = sys._getframe(1)
        pinned = _dynamic_state["pinned_sources"]["instantiator"]
        caller_filename = caller.f_code.co_filename
        if (
            caller.f_globals.get("__name__") != "torch.distributed.nn.jit.instantiator"
            or caller.f_code.co_name != "<module>"
            or caller.f_lineno != _expected_dynamic_authorities["instantiator_append_line"]
            or (
                caller_filename != pinned["descriptor_path"]
                and os.path.realpath(caller_filename) != os.path.realpath(pinned["canonical_path"])
            )
            or caller.f_globals.get("INSTANTIATED_TEMPLATE_DIR_PATH") != value
            or getattr(caller.f_globals.get("_TEMP_DIR"), "name", None) != value
        ):
            raise RuntimeError("unreviewed sys.path append caller")
        if not isinstance(value, str) or not os.path.isabs(value) or os.path.normpath(value) != value:
            raise RuntimeError("dynamic generated directory path differs")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(value, flags)
        try:
            opened = os.fstat(descriptor)
            lexical = os.stat(value, follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(lexical.st_mode)
                or (opened.st_dev, opened.st_ino) != (lexical.st_dev, lexical.st_ino)
                or opened.st_uid != os.geteuid()
                or stat.S_IMODE(opened.st_mode) != 0o700
                or os.listdir(descriptor) != []
            ):
                raise RuntimeError("dynamic generated directory identity/mode differs")
            if sys.platform.startswith("linux"):
                descriptor_path = "/proc/{}/fd/{}".format(os.getpid(), descriptor)
            elif sys.platform == "darwin":
                descriptor_path = "/dev/fd/{}".format(descriptor)
            else:
                raise RuntimeError("dynamic generated directory platform differs")
            followed = os.stat(descriptor_path)
            if (followed.st_dev, followed.st_ino) != (opened.st_dev, opened.st_ino):
                raise RuntimeError("dynamic generated directory FD path differs")
            _dynamic_state["captured"] = {
                "lexical_path": value,
                "descriptor_path": descriptor_path,
                "descriptor": descriptor,
                "device": opened.st_dev,
                "inode": opened.st_ino,
                "mode": opened.st_mode,
                "uid": opened.st_uid,
            }
            caller.f_globals["INSTANTIATED_TEMPLATE_DIR_PATH"] = descriptor_path
            _dynamic_state["append_count"] = 1
            list.append(self, descriptor_path)
            _dynamic_assert_path_surface()
            descriptor = -1
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    extend = _deny
    insert = _deny
    pop = _deny
    remove = _deny
    clear = _deny
    reverse = _deny
    sort = _deny
    __delitem__ = _deny
    __setitem__ = _deny
    __iadd__ = _deny
    __imul__ = _deny

    def restore_exact(self):
        if _dynamic_state["phase"] != "preload" or _dynamic_state["restore_count"] != 0:
            raise RuntimeError("dynamic reviewed path restore differs")
        list.__setitem__(self, slice(None), reviewed_import_path)
        _dynamic_state["restore_count"] = 1
        _dynamic_state["phase"] = "locked"
        _dynamic_assert_path_surface()

def _dynamic_pinned_source_key(fullname):
    mapping = {
        _expected_dynamic_authorities["instantiator_module"]: "instantiator",
        _expected_dynamic_authorities["template_module"]: "template",
        _expected_dynamic_authorities["remote_module_module"]: "remote_module",
    }
    return mapping.get(fullname)

class _PinnedSourceLoader:
    def __init__(self, fullname, source_key):
        self.fullname = fullname
        self.source_key = source_key

    def create_module(self, _specification):
        return None

    def exec_module(self, module):
        source = _dynamic_state["pinned_sources"][self.source_key]
        counts = _dynamic_state["pinned_source_exec_count"]
        if counts.get(self.fullname, 0) != 0:
            raise RuntimeError("dynamic pinned source executed more than once")
        compile_counts = _dynamic_state["pinned_source_compile_count"]
        compile_counts[self.fullname] = compile_counts.get(self.fullname, 0) + 1
        if compile_counts[self.fullname] != 1:
            raise RuntimeError("dynamic pinned source compiled more than once")
        code = compile(
            source["bytes"],
            source["descriptor_path"],
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )
        module.__file__ = source["descriptor_path"]
        module.__cached__ = None
        module.__loader__ = self
        counts[self.fullname] = 1
        exec(code, module.__dict__, module.__dict__)

class _RejectCapturedOrigin:
    def find_spec(self, fullname, path=None, target=None):
        pinned_key = _dynamic_pinned_source_key(fullname)
        if pinned_key is not None:
            if _dynamic_state["pinned_sources"] is None:
                raise ImportError("dynamic pinned source requested before admission")
            find_counts = _dynamic_state["pinned_source_find_count"]
            find_counts[fullname] = find_counts.get(fullname, 0) + 1
            if find_counts[fullname] != 1:
                raise ImportError("dynamic pinned source resolved more than once")
            loader = _dynamic_state["pinned_source_loaders"].get(fullname)
            if loader is None:
                loader = _PinnedSourceLoader(fullname, pinned_key)
                _dynamic_state["pinned_source_loaders"][fullname] = loader
            source = _dynamic_state["pinned_sources"][pinned_key]
            return importlib.machinery.ModuleSpec(
                fullname,
                loader,
                origin=source["descriptor_path"],
            )
        specification = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        origin = None if specification is None else specification.origin
        if _dynamic_origin_is_captured(origin):
            _dynamic_state["captured_finder_count"] += 1
            raise ImportError("captured dynamic directory module origin denied: " + fullname)
        return specification

def _dynamic_validate_captured_file():
    captured = _dynamic_state["captured"]
    if captured is None:
        raise RuntimeError("dynamic generated directory was not captured")
    directory = os.fstat(captured["descriptor"])
    lexical = os.stat(captured["lexical_path"], follow_symlinks=False)
    descriptor_target = os.stat(captured["descriptor_path"])
    identity = (captured["device"], captured["inode"])
    if (
        not stat.S_ISDIR(directory.st_mode)
        or not stat.S_ISDIR(lexical.st_mode)
        or (directory.st_dev, directory.st_ino) != identity
        or (lexical.st_dev, lexical.st_ino) != identity
        or (descriptor_target.st_dev, descriptor_target.st_ino) != identity
        or directory.st_uid != os.geteuid()
        or stat.S_IMODE(directory.st_mode) != 0o700
    ):
        raise RuntimeError("dynamic generated directory changed")
    expected_filename = _expected_dynamic_authorities["generated_module_filename"]
    names = sorted(os.listdir(captured["descriptor"]))
    if names != [expected_filename]:
        raise RuntimeError("dynamic generated directory file set differs")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(expected_filename, flags, dir_fd=captured["descriptor"])
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        not stat.S_ISREG(before.st_mode)
        or (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) != 0o644
        or before.st_nlink != 1
        or len(raw) != _expected_dynamic_authorities["generated_module_size"]
        or hashlib.sha256(raw).hexdigest() != _expected_dynamic_authorities["generated_module_sha256"]
    ):
        raise RuntimeError("dynamic written generated source differs")
    return names

def _dynamic_temp_origin_module_count():
    count = 0
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        origin_values = [getattr(module, "__file__", None)]
        specification = getattr(module, "__spec__", None)
        origin_values.append(None if specification is None else getattr(specification, "origin", None))
        if any(_dynamic_origin_is_captured(value) for value in origin_values):
            count += 1
    return count

def _dynamic_origin_under_reviewed_sites(value):
    if not isinstance(value, str) or not value or value.startswith("<"):
        return False
    absolute = os.path.abspath(value)
    resolved = os.path.realpath(absolute)
    for binding in configuration["sites"]:
        for root in (binding["descriptor_path"], binding["canonical_path"]):
            try:
                if os.path.commonpath((absolute, root)) == root or os.path.commonpath((resolved, os.path.realpath(root))) == os.path.realpath(root):
                    return True
            except (OSError, ValueError):
                pass
    return False

def _dynamic_validate_pinned_loaders():
    expected = {
        _expected_dynamic_authorities["instantiator_module"]: "instantiator",
        _expected_dynamic_authorities["template_module"]: "template",
        _expected_dynamic_authorities["remote_module_module"]: "remote_module",
    }
    expected_counts = {name: 1 for name in expected}
    if (
        _dynamic_state["pinned_source_find_count"] != expected_counts
        or _dynamic_state["pinned_source_compile_count"] != expected_counts
        or _dynamic_state["pinned_source_exec_count"] != expected_counts
        or set(_dynamic_state["pinned_source_loaders"]) != set(expected)
        or _dynamic_state["pinned_source_pyc_read_count"] != 0
    ):
        raise RuntimeError("dynamic pinned source loader counts differ")
    for fullname, source_key in expected.items():
        module = sys.modules.get(fullname)
        loader = _dynamic_state["pinned_source_loaders"][fullname]
        specification = None if module is None else getattr(module, "__spec__", None)
        source = _dynamic_state["pinned_sources"][source_key]
        if (
            module is None
            or module.__loader__ is not loader
            or specification is None
            or specification.loader is not loader
            or specification.origin != source["descriptor_path"]
            or getattr(module, "__file__", None) != source["descriptor_path"]
            or getattr(module, "__cached__", None) is not None
        ):
            raise RuntimeError("dynamic pinned source loader identity differs: " + fullname)
    parent_names = {
        "torch",
        "torch.distributed",
        "torch.distributed.nn",
        "torch.distributed.nn.jit",
        "torch.distributed.nn.jit.templates",
        "torch.distributed.nn.api",
    }
    for fullname in parent_names:
        module = sys.modules.get(fullname)
        origins = []
        if module is not None:
            file_origin = getattr(module, "__file__", None)
            if isinstance(file_origin, str):
                origins.append(file_origin)
            package_path = getattr(module, "__path__", None)
            if package_path is not None:
                origins.extend(str(value) for value in package_path)
        if not origins or any(not _dynamic_origin_under_reviewed_sites(value) for value in origins):
            raise RuntimeError("dynamic pinned parent package origin differs: " + fullname)

def _dynamic_revalidate_pinned_sources():
    for name, relative_key, hash_key in (
        ("instantiator", "instantiator_relative_path", "instantiator_source_sha256"),
        ("template", "template_relative_path", "template_source_sha256"),
        ("remote_module", "remote_module_relative_path", "remote_module_source_sha256"),
    ):
        observed = _dynamic_read_unique_pinned(
            _expected_dynamic_authorities[relative_key],
            _expected_dynamic_authorities[hash_key],
        )
        original = _dynamic_state["pinned_sources"][name]
        if (
            observed["device"],
            observed["inode"],
            observed["mode"],
            observed["size"],
            observed["bytes"],
        ) != (
            original["device"],
            original["inode"],
            original["mode"],
            original["size"],
            original["bytes"],
        ):
            raise RuntimeError("dynamic pinned source identity changed: " + name)

def _dynamic_success_evidence():
    return {
        "schema_version": _dynamic_configuration["schema_version"],
        "enabled": True,
        "authorities": _dynamic_configuration["authorities"],
        "policy": _dynamic_configuration["policy"],
        "observed": {
            "append_caller_validated": True,
            "append_count": _dynamic_state["append_count"],
            "captured_compile_count": _dynamic_state["captured_compile_count"],
            "captured_directory_final_entries": [_expected_dynamic_authorities["generated_module_filename"]],
            "captured_directory_initially_empty": True,
            "captured_directory_revalidated_after_payload": True,
            "captured_exec_count": _dynamic_state["captured_exec_count"],
            "captured_finder_count": _dynamic_state["captured_finder_count"],
            "captured_temp_origin_module_count": _dynamic_temp_origin_module_count(),
            "generated_file_sha256": _expected_dynamic_authorities["generated_module_sha256"],
            "generated_file_size": _expected_dynamic_authorities["generated_module_size"],
            "preseed_compile_count": _dynamic_state["preseed_compile_count"],
            "preseed_exec_count": _dynamic_state["preseed_exec_count"],
            "pinned_source_compile_count": dict(_dynamic_state["pinned_source_compile_count"]),
            "pinned_source_exec_count": dict(_dynamic_state["pinned_source_exec_count"]),
            "pinned_source_find_count": dict(_dynamic_state["pinned_source_find_count"]),
            "pinned_source_loader_identity_validated": True,
            "pinned_source_pyc_read_count": _dynamic_state["pinned_source_pyc_read_count"],
            "remote_module_identity_is_preseed": True,
            "reviewed_sys_path_restored": tuple(sys.path) == reviewed_import_path,
            "template_compile_count": _dynamic_state["template_compile_count"],
            "template_exec_count": _dynamic_state["template_exec_count"],
        },
    }

def _dynamic_preload_torchvision():
    authorities = _expected_dynamic_authorities
    pinned = {
        "instantiator": _dynamic_read_unique_pinned(authorities["instantiator_relative_path"], authorities["instantiator_source_sha256"]),
        "template": _dynamic_read_unique_pinned(authorities["template_relative_path"], authorities["template_source_sha256"]),
        "remote_module": _dynamic_read_unique_pinned(authorities["remote_module_relative_path"], authorities["remote_module_source_sha256"]),
    }
    lines = pinned["instantiator"]["bytes"].decode("utf-8").splitlines()
    expected_line = authorities["instantiator_append_line"]
    expected_source = authorities["instantiator_append_source"]
    if len(lines) < expected_line or lines[expected_line - 1].strip() != expected_source or sum(line.strip() == expected_source for line in lines) != 1:
        raise RuntimeError("dynamic instantiator append source/line differs")
    _dynamic_state["pinned_sources"] = pinned
    for fullname in (
        authorities["instantiator_module"],
        authorities["template_module"],
        authorities["remote_module_module"],
    ):
        if fullname in sys.modules:
            raise RuntimeError("dynamic pinned source unexpectedly preloaded: " + fullname)
    template_origin = "<rachel-pinned-remote-module-template>"
    template_namespace = {"__name__": "_rachel_pinned_remote_template", "__file__": template_origin}
    exec(compile(pinned["template"]["bytes"], template_origin, "exec"), template_namespace, template_namespace)
    generated_text = template_namespace["get_remote_module_template"](True).format(
        assign_module_interface_cls="module_interface_cls = None",
        args="*args",
        kwargs="**kwargs",
        arg_types="*args, **kwargs",
        arrow_and_return_type="",
        arrow_and_future_return_type="",
        jit_script_decorator="",
    )
    generated = generated_text.encode("utf-8")
    if len(generated) != authorities["generated_module_size"] or hashlib.sha256(generated).hexdigest() != authorities["generated_module_sha256"]:
        raise RuntimeError("dynamic generated remote-module bytes differ")
    module_name = authorities["generated_module_name"]
    if module_name in sys.modules:
        raise RuntimeError("dynamic generated module unexpectedly preloaded")
    preseed = types.ModuleType(module_name)
    preseed.__file__ = authorities["preseed_origin"]
    preseed.__package__ = ""
    preseed.__loader__ = None
    preseed.__cached__ = None
    preseed.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None, origin=authorities["preseed_origin"])
    sys.modules[module_name] = preseed
    _dynamic_state["preseed"] = preseed
    previous_umask = os.umask(0o022)
    try:
        exec(compile(generated, authorities["preseed_origin"], "exec"), preseed.__dict__, preseed.__dict__)
        if not isinstance(preseed.__dict__.get("_generated_methods"), list) or len(preseed.__dict__["_generated_methods"]) != 2:
            raise RuntimeError("dynamic preseed generated methods differ")
        torchvision = importlib.import_module("torchvision")
        remote_module = importlib.import_module("torch.distributed.nn.api.remote_module")
        instantiator = importlib.import_module("torch.distributed.nn.jit.instantiator")
    except BaseException:
        if sys.modules.get(module_name) is preseed:
            del sys.modules[module_name]
        raise
    finally:
        os.umask(previous_umask)
        if sys.path is _controlled_import_path and _dynamic_state["phase"] == "preload" and _dynamic_state["append_count"] == 1:
            _controlled_import_path.restore_exact()
    if (
        str(getattr(sys.modules.get("torch"), "__version__", "")) != authorities["torch_version"]
        or str(getattr(torchvision, "__version__", "")) != authorities["torchvision_version"]
        or _dynamic_state["append_count"] != 1
        or _dynamic_state["restore_count"] != 1
        or remote_module._NON_SCRIPTABLE_REMOTE_MODULE_MODULE is not preseed
        or instantiator.INSTANTIATED_TEMPLATE_DIR_PATH != _dynamic_state["captured"]["descriptor_path"]
        or instantiator.__file__ != pinned["instantiator"]["descriptor_path"]
        or remote_module.__file__ != pinned["remote_module"]["descriptor_path"]
    ):
        raise RuntimeError("dynamic pinned module identity differs")
    _dynamic_validate_captured_file()
    if (
        _dynamic_state["captured_finder_count"] != 0
        or _dynamic_state["captured_compile_count"] != 0
        or _dynamic_state["captured_exec_count"] != 0
        or _dynamic_state["template_compile_count"] != 1
        or _dynamic_state["template_exec_count"] != 1
        or _dynamic_state["preseed_compile_count"] != 1
        or _dynamic_state["preseed_exec_count"] != 1
        or _dynamic_temp_origin_module_count() != 0
    ):
        raise RuntimeError("dynamic admission event counts differ")
    _dynamic_validate_pinned_loaders()
    _dynamic_revalidate_pinned_sources()
    return _dynamic_success_evidence()

def _dynamic_finalize():
    _dynamic_assert_path_surface()
    if not _dynamic_configuration["enabled"]:
        if _dynamic_state["append_count"] != 0 or _dynamic_state["captured"] is not None:
            raise RuntimeError("disabled dynamic admission observed activity")
        return
    if sys.modules.get(_expected_dynamic_authorities["generated_module_name"]) is not _dynamic_state["preseed"]:
        raise RuntimeError("dynamic preseed module identity changed")
    remote_module = sys.modules.get("torch.distributed.nn.api.remote_module")
    instantiator = sys.modules.get("torch.distributed.nn.jit.instantiator")
    if (
        remote_module is None
        or remote_module._NON_SCRIPTABLE_REMOTE_MODULE_MODULE is not _dynamic_state["preseed"]
        or instantiator is None
        or instantiator.INSTANTIATED_TEMPLATE_DIR_PATH != _dynamic_state["captured"]["descriptor_path"]
        or sys.meta_path.count(_dynamic_finder) != 1
        or sys.meta_path.index(_dynamic_finder) >= sys.meta_path.index(importlib.machinery.PathFinder)
    ):
        raise RuntimeError("dynamic runtime admission guard identity changed")
    _dynamic_validate_captured_file()
    _dynamic_validate_pinned_loaders()
    _dynamic_revalidate_pinned_sources()
    if _dynamic_success_evidence() != __rachel_torchvision_dynamic_admission__:
        raise RuntimeError("dynamic admission evidence changed")
    os.close(_dynamic_state["captured"]["descriptor"])
    _dynamic_state["captured"]["descriptor"] = -1

_controlled_import_path = _ControlledImportPath(sys.path)
sys.path = _controlled_import_path
if sys.meta_path.count(importlib.machinery.PathFinder) != 1:
    raise RuntimeError("dynamic PathFinder surface differs")
_dynamic_finder = _RejectCapturedOrigin()
sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder), _dynamic_finder)
sys.addaudithook(_dynamic_audit)
__rachel_torchvision_dynamic_admission__ = None
if _dynamic_configuration["enabled"]:
    __rachel_torchvision_dynamic_admission__ = _dynamic_preload_torchvision()
_dynamic_assert_path_surface()
"""


EXPLICIT_SITE_BOOTSTRAP_CODE = (
    r"""import hashlib, importlib, importlib.machinery, json, os, site as _site, stat, sys, types
"""
    + EXPLICIT_SITE_DESCRIPTOR_GUARD_CODE
    + r"""
configuration = json.loads(sys.argv[1])
payload_code = sys.argv[2]
payload_arguments = sys.argv[3:]
declared = configuration.pop("content_sha256", None)
observed = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
if declared != observed:
    raise RuntimeError("explicit-site configuration hash differs")
if set(configuration) != {"schema_version", "startup", "runtime", "no_site_sys_path", "sites", "torchvision_dynamic_admission"} or configuration["schema_version"] != "rachel-benchmark-explicit-site-bootstrap/3.0":
    raise RuntimeError("explicit-site configuration schema differs")
if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
    raise RuntimeError("explicit-site bootstrap lacks -I -S -B")
if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
    raise RuntimeError("startup customization loaded before explicit-site bootstrap")
def _blocked_site_startup(*_args, **_kwargs):
    raise RuntimeError("automatic site/PTH processing is disabled")
for _name in ("addpackage", "addsitedir", "main", "execsitecustomize", "execusercustomize"):
    if hasattr(_site, _name):
        setattr(_site, _name, _blocked_site_startup)
_site.ENABLE_USER_SITE = False
startup = configuration["startup"]
runtime = configuration["runtime"]
if set(startup) != {"executable", "prefix", "base_prefix"} or set(runtime) != {"prefix", "base_prefix"}:
    raise RuntimeError("explicit-site runtime identity schema differs")
if os.path.realpath(sys.executable) != startup["executable"] or os.path.realpath(sys.prefix) != startup["prefix"] or os.path.realpath(sys.base_prefix) != startup["base_prefix"]:
    raise RuntimeError("explicit-site startup runtime identity differs")
if runtime["base_prefix"] != startup["base_prefix"] or not os.path.isabs(runtime["prefix"]):
    raise RuntimeError("explicit-site logical runtime identity differs")
current = [os.path.realpath(value) for value in sys.path if value]
if current != configuration["no_site_sys_path"]:
    raise RuntimeError("no-site sys.path differs before explicit injection")
seen = set(current)
for binding in configuration["sites"]:
    if set(binding) != {"canonical_path", "descriptor_path", "device", "inode", "role", "audit_content_sha256"}:
        raise RuntimeError("explicit-site binding schema differs")
    descriptor_path = _held_site_descriptor(binding)
    canonical = binding["canonical_path"]
    if canonical in seen:
        raise RuntimeError("explicit-site canonical path differs")
    sys.path.append(descriptor_path)
    seen.add(canonical)
reviewed_import_path = tuple(sys.path)
sys.prefix = runtime["prefix"]
sys.exec_prefix = runtime["prefix"]
if os.path.realpath(sys.base_prefix) != runtime["base_prefix"]:
    raise RuntimeError("explicit-site base prefix changed")
"""
    + TORCHVISION_DYNAMIC_ADMISSION_BOOTSTRAP_CODE
    + r"""
sys.argv = ["<rachel-explicit-site-payload>", *payload_arguments]
namespace = {"__name__": "__main__", "__file__": "<rachel-explicit-site-payload>", "__rachel_explicit_site_bindings__": tuple(configuration["sites"]), "__rachel_torchvision_dynamic_admission__": __rachel_torchvision_dynamic_admission__}
exec(compile(payload_code, namespace["__file__"], "exec"), namespace, namespace)
_dynamic_finalize()
"""
)


PIP_NO_SITE_RUN_CODE = r"""import importlib.util, os, runpy, sys
bindings = globals().get("__rachel_explicit_site_bindings__")
if not isinstance(bindings, tuple):
    raise RuntimeError("pip lacks reviewed explicit-site bindings")
specification = importlib.util.find_spec("pip")
origin = None if specification is None else specification.origin
if not origin:
    raise RuntimeError("pip import origin is missing")
real_origin = os.path.realpath(origin)
matches = []
for binding in bindings:
    expected_site = os.path.realpath(binding["descriptor_path"])
    if os.path.commonpath((real_origin, expected_site)) == expected_site:
        matches.append(binding)
if len(matches) != 1 or matches[0].get("role") != "audit_1":
    raise RuntimeError("pip is not imported from the reviewed resolved-base site")
sys.argv = ["pip", *sys.argv[1:]]
runpy.run_module("pip", run_name="__main__", alter_sys=True)
"""


def _validate_pip_runtime_audits(
    audits: Sequence[Mapping[str, object]], label: str
) -> None:
    if len(audits) != 2:
        raise EnvironmentBuildError(
            label + " requires exactly target and resolved-base site audits"
        )
    prefixes = []
    for index, audit in enumerate(audits):
        _validate_content_sha(audit, label + " site audit")
        prefix = audit.get("prefix")
        sites = audit.get("site_directories")
        if (
            not isinstance(prefix, str)
            or not Path(prefix).is_absolute()
            or not isinstance(sites, list)
            or len(sites) != 1
        ):
            raise EnvironmentBuildError(label + " site audit differs")
        if (
            audit.get("runtime_no_site") is not True
            or audit.get("pth_execution_policy") != "present_but_never_executed"
            or audit.get("activated_paths") != []
        ):
            raise EnvironmentBuildError(label + " no-site audit differs")
        prefixes.append(prefix)
    if prefixes[0] == prefixes[1]:
        raise EnvironmentBuildError(label + " target and pip prefixes overlap")


def _canonical_site_bindings(
    audits: Sequence[Mapping[str, object]], label: str
) -> List[Dict[str, object]]:
    bindings = []
    canonical_seen = set()
    for audit_index, audit in enumerate(audits):
        _validate_content_sha(audit, label + " site audit")
        prefix_value = audit.get("prefix")
        directories = audit.get("site_directories")
        scanned = audit.get("startup_search_directories")
        if (
            not isinstance(prefix_value, str)
            or not isinstance(directories, list)
            or not isinstance(scanned, list)
        ):
            raise EnvironmentBuildError(label + " site audit is malformed")
        prefix = _lexical_absolute(prefix_value, label + " site prefix")
        scanned_by_path = {
            str(item.get("path")): item for item in scanned if isinstance(item, dict)
        }
        for directory_value in directories:
            directory = _lexical_absolute(
                str(directory_value), label + " site directory"
            )
            if str(directory) in canonical_seen:
                continue
            if not _is_within(directory, prefix):
                raise EnvironmentBuildError(label + " site directory escaped prefix")
            recorded = scanned_by_path.get(str(directory))
            if not isinstance(recorded, dict):
                raise EnvironmentBuildError(
                    label + " site directory lacks audited identity"
                )
            identity = _directory_identity(directory, label + " site directory")
            if identity != {
                "device": recorded.get("device"),
                "inode": recorded.get("inode"),
            }:
                raise EnvironmentBuildError(label + " site directory identity drifted")
            bindings.append(
                {
                    "canonical_path": str(directory),
                    "prefix": str(prefix),
                    "relative_path": directory.relative_to(prefix).as_posix(),
                    "device": identity["device"],
                    "inode": identity["inode"],
                    "role": "audit_" + str(audit_index),
                    "audit_content_sha256": audit["content_sha256"],
                }
            )
            canonical_seen.add(str(directory))
    if not bindings:
        raise EnvironmentBuildError(label + " has no explicit site bindings")
    return bindings


def _run_explicit_site_payload(
    python: Path,
    code: str,
    arguments: Sequence[str],
    *,
    safe_probe: Mapping[str, object],
    audits: Sequence[Mapping[str, object]],
    timeout: int,
    label: str,
    primary_anchor: Optional[_ExecutionAnchor] = None,
    preload_torchvision: bool = False,
) -> subprocess.CompletedProcess:
    if not isinstance(preload_torchvision, bool):
        raise EnvironmentBuildError(label + " dynamic preload flag differs")
    no_site_sys_path = safe_probe.get("sys_path")
    if not isinstance(no_site_sys_path, list) or any(
        not isinstance(path, str) for path in no_site_sys_path
    ):
        raise EnvironmentBuildError(label + " no-site sys.path is malformed")
    startup_executable = safe_probe.get("executable")
    startup_prefix = safe_probe.get("prefix")
    startup_base_prefix = safe_probe.get("base_prefix")
    if any(
        not isinstance(value, str) or not Path(value).is_absolute()
        for value in (startup_executable, startup_prefix, startup_base_prefix)
    ):
        raise EnvironmentBuildError(label + " no-site runtime identity is malformed")
    owned_anchors: List[_ExecutionAnchor] = []
    anchors_by_prefix: Dict[str, _ExecutionAnchor] = {}
    site_anchors: List[_ExecutionAnchor] = []

    def anchor_for(prefix: Path) -> _ExecutionAnchor:
        key = str(prefix)
        existing = anchors_by_prefix.get(key)
        if existing is not None:
            return existing
        if primary_anchor is not None and primary_anchor.root == prefix:
            anchor = primary_anchor
        else:
            anchor = _ExecutionAnchor(prefix, label + " prefix")
            owned_anchors.append(anchor)
        anchors_by_prefix[key] = anchor
        return anchor

    try:
        logical_prefix = _lexical_absolute(
            str(python.parent.parent), label + " executable prefix"
        )
        launcher, launcher_identity = _require_python_launcher(
            python, label + " Python"
        )
        if launcher != python:
            raise EnvironmentBuildError(label + " Python launcher path differs")
        resolved_executable = _lexical_absolute(
            str(launcher_identity["resolved_executable"]),
            label + " resolved executable",
        )
        resolved_prefix = _lexical_absolute(
            str(resolved_executable.parent.parent), label + " resolved prefix"
        )
        anchor_for(logical_prefix)
        executable_anchor = anchor_for(resolved_prefix)
        executable = executable_anchor.child_path(
            resolved_executable, label + " executable"
        )
        sites = []
        for binding in _canonical_site_bindings(audits, label):
            prefix = Path(str(binding["prefix"]))
            directory = Path(str(binding["canonical_path"]))
            anchor_for(prefix)
            site_anchor = _ExecutionAnchor(directory, label + " held site directory")
            owned_anchors.append(site_anchor)
            site_anchors.append(site_anchor)
            sites.append(
                {
                    "canonical_path": binding["canonical_path"],
                    "descriptor_path": str(
                        site_anchor.child_path(directory, label + " site directory")
                    ),
                    "device": binding["device"],
                    "inode": binding["inode"],
                    "role": binding["role"],
                    "audit_content_sha256": binding["audit_content_sha256"],
                }
            )
        configuration = _content_bound(
            {
                "schema_version": EXPLICIT_SITE_BOOTSTRAP_SCHEMA,
                "startup": {
                    "executable": startup_executable,
                    "prefix": startup_prefix,
                    "base_prefix": startup_base_prefix,
                },
                "runtime": {
                    "prefix": str(logical_prefix),
                    "base_prefix": startup_base_prefix,
                },
                "no_site_sys_path": list(no_site_sys_path),
                "sites": sites,
                "torchvision_dynamic_admission": (
                    _torchvision_dynamic_admission_configuration(
                        preload_torchvision
                    )
                ),
            }
        )
        bootstrap_sha256 = _sha256_bytes(EXPLICIT_SITE_BOOTSTRAP_CODE.encode("utf-8"))
        no_site_path_sha256 = _sha256_bytes(_canonical_bytes(list(no_site_sys_path)))
        completed = _run_checked(
            (
                str(executable),
                "-I",
                "-S",
                "-B",
                "-c",
                EXPLICIT_SITE_STDIN_LAUNCHER_CODE,
                str(len(no_site_sys_path)),
                *no_site_sys_path,
                no_site_path_sha256,
                bootstrap_sha256,
                json.dumps(configuration, sort_keys=True, separators=(",", ":")),
                code,
                *arguments,
            ),
            timeout=timeout,
            label=label,
            stdin_text=EXPLICIT_SITE_BOOTSTRAP_CODE,
            execution_anchor=executable_anchor,
            additional_execution_anchors=(
                *tuple(anchors_by_prefix.values()),
                *tuple(site_anchors),
            ),
        )
        if _require_python_launcher(python, label + " Python")[1] != launcher_identity:
            raise EnvironmentBuildError(label + " Python launcher changed")
        return completed
    finally:
        for anchor in reversed(owned_anchors):
            anchor.close()


def _run_json_explicit_sites(
    python: Path,
    code: str,
    arguments: Sequence[str] = (),
    *,
    safe_probe: Mapping[str, object],
    audits: Sequence[Mapping[str, object]],
    timeout: int = 600,
    label: str,
    primary_anchor: Optional[_ExecutionAnchor] = None,
    preload_torchvision: bool = False,
) -> Dict[str, object]:
    completed = _run_explicit_site_payload(
        python,
        code,
        arguments,
        safe_probe=safe_probe,
        audits=audits,
        timeout=timeout,
        label=label,
        primary_anchor=primary_anchor,
        preload_torchvision=preload_torchvision,
    )
    try:
        value = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise EnvironmentBuildError(label + " returned malformed JSON") from error
    if not isinstance(value, dict):
        raise EnvironmentBuildError(label + " did not return a JSON object")
    return value


def _run_json_no_site(
    python: Path,
    code: str,
    arguments: Sequence[str] = (),
    *,
    timeout: int = 600,
    label: str,
    execution_anchor: Optional[_ExecutionAnchor] = None,
) -> Dict[str, object]:
    executable = (
        execution_anchor.child_path(python, label + " executable")
        if execution_anchor is not None
        else python
    )
    completed = _run_checked(
        (str(executable), "-I", "-S", "-B", "-c", code, *arguments),
        timeout=timeout,
        label=label,
        execution_anchor=execution_anchor,
    )
    try:
        value = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise EnvironmentBuildError(label + " returned malformed JSON") from error
    if not isinstance(value, dict):
        raise EnvironmentBuildError(label + " did not return a JSON object")
    return value


def _safe_runtime_probe(
    python: Path, execution_anchor: Optional[_ExecutionAnchor] = None
) -> Dict[str, object]:
    return _run_json_no_site(
        python,
        SAFE_RUNTIME_PROBE_CODE,
        label="no-site runtime identity probe",
        execution_anchor=execution_anchor,
    )


def _anchored_safe_runtime_probe(python: Path, prefix: Path) -> Dict[str, object]:
    launcher, launcher_identity = _require_python_launcher(
        python, "no-site runtime Python"
    )
    if launcher != python:
        raise EnvironmentBuildError("no-site runtime Python launcher path differs")
    resolved_executable = Path(str(launcher_identity["resolved_executable"]))
    resolved_prefix = resolved_executable.parent.parent
    formal_anchor = _ExecutionAnchor(prefix, "no-site formal runtime prefix")
    resolved_anchor = (
        formal_anchor
        if resolved_prefix == prefix
        else _ExecutionAnchor(resolved_prefix, "no-site resolved runtime prefix")
    )
    try:
        executable = resolved_anchor.child_path(
            resolved_executable, "no-site resolved runtime executable"
        )
        completed = _run_checked(
            (
                str(executable),
                "-I",
                "-S",
                "-B",
                "-c",
                SAFE_RUNTIME_PROBE_CODE,
            ),
            label="no-site runtime identity probe",
            execution_anchor=resolved_anchor,
            additional_execution_anchors=(formal_anchor,),
        )
        try:
            value = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise EnvironmentBuildError(
                "no-site runtime identity probe returned malformed JSON"
            ) from error
        if not isinstance(value, dict):
            raise EnvironmentBuildError(
                "no-site runtime identity probe did not return a JSON object"
            )
        if (
            _require_python_launcher(python, "no-site runtime Python")[1]
            != launcher_identity
        ):
            raise EnvironmentBuildError("no-site runtime Python launcher changed")
        return value
    finally:
        if resolved_anchor is not formal_anchor:
            resolved_anchor.close()
        formal_anchor.close()


def _queue_live_probe(
    python: Path,
    *,
    safe_probe: Mapping[str, object],
    audits: Sequence[Mapping[str, object]],
    execution_anchor: Optional[_ExecutionAnchor] = None,
) -> Dict[str, object]:
    return _run_json_explicit_sites(
        python,
        QUEUE_LIVE_PROBE_CODE,
        safe_probe=safe_probe,
        audits=audits,
        label="queue live probe",
        primary_anchor=execution_anchor,
        preload_torchvision=True,
    )


def _detailed_probe(
    python: Path,
    *,
    safe_probe: Mapping[str, object],
    audits: Sequence[Mapping[str, object]],
    execution_anchor: Optional[_ExecutionAnchor] = None,
) -> Dict[str, object]:
    return _run_json_explicit_sites(
        python,
        DETAILED_PROBE_CODE,
        (json.dumps(TARGET_TRACKED_DISTRIBUTIONS),),
        safe_probe=safe_probe,
        audits=audits,
        label="detailed dependency probe",
        primary_anchor=execution_anchor,
        preload_torchvision=True,
    )


def _base_probe(
    python: Path,
    *,
    safe_probe: Mapping[str, object],
    audits: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    return _run_json_explicit_sites(
        python,
        DETAILED_PROBE_CODE,
        (json.dumps(BASE_TRACKED_DISTRIBUTIONS),),
        safe_probe=safe_probe,
        audits=audits,
        label="base runtime dependency probe",
        preload_torchvision=True,
    )


def _functional_cpu_probe(
    python: Path,
    *,
    safe_probe: Mapping[str, object],
    audits: Sequence[Mapping[str, object]],
    execution_anchor: Optional[_ExecutionAnchor] = None,
) -> Dict[str, object]:
    return _run_json_explicit_sites(
        python,
        FUNCTIONAL_CPU_PROBE_CODE,
        safe_probe=safe_probe,
        audits=audits,
        timeout=180,
        label="real CPU dependency forward probe",
        primary_anchor=execution_anchor,
        preload_torchvision=True,
    )


def _import_inventory(
    python: Path,
    root: Path,
    *,
    safe_probe: Mapping[str, object],
    audits: Sequence[Mapping[str, object]],
    execution_anchor: Optional[_ExecutionAnchor] = None,
) -> Dict[str, object]:
    inventory_root = (
        execution_anchor.child_path(root, "inventory environment root")
        if execution_anchor is not None
        else root
    )
    return _run_json_explicit_sites(
        python,
        INVENTORY_PROBE_CODE,
        (json.dumps(TARGET_TRACKED_DISTRIBUTIONS), str(inventory_root)),
        safe_probe=safe_probe,
        audits=audits,
        timeout=1800,
        label="imported package content inventory",
        primary_anchor=execution_anchor,
        preload_torchvision=True,
    )


def _content_bound(value: Mapping[str, object]) -> Dict[str, object]:
    result = dict(value)
    result["content_sha256"] = _sha256_bytes(_canonical_bytes(result))
    return result


def _validate_content_sha(value: Mapping[str, object], label: str) -> None:
    declared = value.get("content_sha256")
    if not _is_sha256(declared):
        raise EnvironmentBuildError(label + " lacks a content SHA-256")
    copied = dict(value)
    copied.pop("content_sha256", None)
    if _sha256_bytes(_canonical_bytes(copied)) != declared:
        raise EnvironmentBuildError(label + " content SHA-256 differs")


def _write_bytes_new_immutable(path: Path, payload: bytes, label: str) -> None:
    target = _lexical_absolute(str(path), label)
    if target.name in {"", ".", ".."}:
        raise EnvironmentBuildError(label + " has an invalid leaf name")
    parent = _require_directory(target.parent, label + " parent")
    parent_descriptor = _open_absolute_nofollow(
        parent, directory=True, label=label + " parent"
    )
    temporary_name = ".{}.tmp-{}-{}".format(
        target.name, os.getpid(), os.urandom(12).hex()
    )
    temporary_descriptor: Optional[int] = None
    published = False
    try:
        _revalidate_fd_path(
            parent,
            parent_descriptor,
            directory=True,
            label=label + " parent",
        )
        try:
            os.stat(target.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileExistsError as error:
            raise EnvironmentBuildError("refusing to overwrite " + label) from error
        except FileNotFoundError:
            pass
        else:
            raise EnvironmentBuildError("refusing to overwrite " + label)
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        written = 0
        while written < len(payload):
            count = os.write(temporary_descriptor, payload[written:])
            if count <= 0:
                raise EnvironmentBuildError(label + " temporary write stalled")
            written += count
        os.fsync(temporary_descriptor)
        os.fchmod(temporary_descriptor, 0o444)
        temporary_identity = _inode_identity(os.fstat(temporary_descriptor))
        os.close(temporary_descriptor)
        temporary_descriptor = None
        _revalidate_fd_path(
            parent,
            parent_descriptor,
            directory=True,
            label=label + " parent",
        )
        _rename_noreplace(parent_descriptor, temporary_name, target.name, label=label)
        published = True
        destination_descriptor = os.open(
            target.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            destination_metadata = os.fstat(destination_descriptor)
            if (
                not stat.S_ISREG(destination_metadata.st_mode)
                or _inode_identity(destination_metadata) != temporary_identity
            ):
                raise EnvironmentBuildError(label + " publish identity differs")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(destination_descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            if digest.hexdigest() != _sha256_bytes(payload):
                raise EnvironmentBuildError(label + " published content differs")
            _revalidate_fd_path(
                target,
                destination_descriptor,
                directory=False,
                label=label,
            )
        finally:
            os.close(destination_descriptor)
        os.fsync(parent_descriptor)
        _revalidate_fd_path(
            parent,
            parent_descriptor,
            directory=True,
            label=label + " parent",
        )
    except BaseException:
        if published:
            try:
                os.unlink(target.name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except OSError:
                pass
        raise
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def _rename_noreplace(
    directory_descriptor: int, source: str, destination: str, *, label: str
) -> None:
    """Atomically rename inside one held dirfd without replacing a peer."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    result: Optional[int] = None
    if sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is not None:
            function.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            function.restype = ctypes.c_int
            result = int(
                function(
                    directory_descriptor,
                    source_bytes,
                    directory_descriptor,
                    destination_bytes,
                    1,  # RENAME_NOREPLACE
                )
            )
        else:
            syscall = getattr(library, "syscall", None)
            if syscall is not None:
                syscall.restype = ctypes.c_long
                result = int(
                    syscall(
                        ctypes.c_long(316),  # __NR_renameat2 on formal x86_64 Linux
                        ctypes.c_int(directory_descriptor),
                        ctypes.c_char_p(source_bytes),
                        ctypes.c_int(directory_descriptor),
                        ctypes.c_char_p(destination_bytes),
                        ctypes.c_uint(1),
                    )
                )
    elif sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is not None:
            function.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            function.restype = ctypes.c_int
            result = int(
                function(
                    directory_descriptor,
                    source_bytes,
                    directory_descriptor,
                    destination_bytes,
                    0x00000004,  # RENAME_EXCL
                )
            )
    if result is None:
        raise EnvironmentBuildError(label + " platform lacks atomic no-clobber rename")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise EnvironmentBuildError(label + " appeared during publication")
        raise EnvironmentBuildError(label + " atomic publication failed") from OSError(
            error_number, os.strerror(error_number)
        )


def _write_json_new_immutable(
    path: Path, value: Mapping[str, object], label: str, *, bind_content: bool = False
) -> Dict[str, object]:
    payload_value = _content_bound(value) if bind_content else dict(value)
    payload = _formatted_json_bytes(payload_value)
    _write_bytes_new_immutable(path, payload, label)
    return payload_value


def _formatted_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _load_json(path: Path, label: str) -> Dict[str, object]:
    value, _sha256 = _load_json_with_sha(path, label)
    return value


def _load_json_with_sha(path: Path, label: str) -> Tuple[Dict[str, object], str]:
    try:
        payload = _read_bytes_file(path, label)
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EnvironmentBuildError(label + " is not readable JSON") from error
    if not isinstance(value, dict):
        raise EnvironmentBuildError(label + " must be a JSON object")
    return value, _sha256_bytes(payload)


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _python_major_minor(version: object, label: str) -> str:
    if not isinstance(version, str):
        raise EnvironmentBuildError(label + " Python version is missing")
    pieces = version.split(".")
    if len(pieces) < 2 or not pieces[0].isdigit() or not pieces[1].isdigit():
        raise EnvironmentBuildError(label + " Python version is malformed")
    return pieces[0] + "." + pieces[1]


def _validate_safe_runtime_probe(
    value: Mapping[str, object], launcher: Path, label: str
) -> None:
    launcher_value, launcher_identity = _require_python_launcher(launcher, label)
    if launcher_value != launcher or value.get("executable") != str(
        launcher_identity["resolved_executable"]
    ):
        raise EnvironmentBuildError(label + " no-site executable identity differs")
    if (
        value.get("isolated") is not True
        or value.get("no_site") is not True
        or value.get("dont_write_bytecode") is not True
    ):
        raise EnvironmentBuildError(label + " no-site probe flags differ")
    if value.get("platform_system") != "Linux" or value.get("platform_machine") not in {
        "x86_64",
        "AMD64",
    }:
        raise EnvironmentBuildError(label + " formal wheel target is not Linux x86_64")
    if _python_major_minor(value.get("python_version"), label) != "3.12":
        raise EnvironmentBuildError(label + " formal wheel target must be Python 3.12")
    safe_prefix = value.get("prefix")
    safe_base_prefix = value.get("base_prefix")
    if (
        not isinstance(safe_prefix, str)
        or not isinstance(safe_base_prefix, str)
        or not Path(safe_prefix).is_absolute()
        or not Path(safe_base_prefix).is_absolute()
        or safe_prefix != os.path.realpath(safe_prefix)
        or safe_base_prefix != os.path.realpath(safe_base_prefix)
        or safe_prefix != safe_base_prefix
    ):
        raise EnvironmentBuildError(label + " no-site prefix/base-prefix differs")
    sys_path = value.get("sys_path")
    if (
        not isinstance(sys_path, list)
        or not sys_path
        or any(not isinstance(path, str) or not path for path in sys_path)
        or len(sys_path) != len(set(sys_path))
        or any(
            "site-packages" in Path(path).parts or "dist-packages" in Path(path).parts
            for path in sys_path
        )
    ):
        raise EnvironmentBuildError(label + " no-site sys.path is malformed")


def _active_site_directories(prefix: Path, python_version: object) -> List[Path]:
    major_minor = _python_major_minor(python_version, "active-site")
    prefix = _require_directory(prefix, "active-site prefix")
    library_roots = []
    identities = set()
    lib_identity: Optional[Tuple[int, int]] = None
    lib = prefix / "lib"
    if _lstat(lib) is not None:
        lib = _require_directory(lib, "active-site lib directory")
        observed_identity = _directory_identity(lib, "active-site lib directory")
        identity = (observed_identity["device"], observed_identity["inode"])
        lib_identity = identity
        library_roots.append(lib)
        identities.add(identity)
    lib64 = prefix / "lib64"
    lib64_metadata = _lstat(lib64)
    if lib64_metadata is not None:
        if stat.S_ISLNK(lib64_metadata.st_mode):
            before = _stat_identity(lib64_metadata)
            raw_target = os.readlink(str(lib64))
            after = _lstat(lib64)
            if after is None or _stat_identity(after) != before:
                raise EnvironmentBuildError("active-site lib64 alias changed")
            target = Path(raw_target)
            if not target.is_absolute():
                target = prefix / target
            target = _lexical_absolute(str(target), "active-site lib64 alias")
            if target != prefix / "lib" or lib_identity is None:
                raise EnvironmentBuildError(
                    "active-site lib64 symlink is not lib alias"
                )
            prefix_descriptor = _open_absolute_nofollow(
                prefix, directory=True, label="active-site prefix"
            )
            try:
                followed = os.open(
                    "lib64",
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=prefix_descriptor,
                )
                try:
                    followed_metadata = os.fstat(followed)
                    if (
                        int(followed_metadata.st_dev),
                        int(followed_metadata.st_ino),
                    ) != lib_identity:
                        raise EnvironmentBuildError(
                            "active-site lib64 alias inode differs from lib"
                        )
                finally:
                    os.close(followed)
                confirmed_alias = os.stat(
                    "lib64", dir_fd=prefix_descriptor, follow_symlinks=False
                )
                if _stat_identity(confirmed_alias) != before:
                    raise EnvironmentBuildError("active-site lib64 alias changed")
                _revalidate_fd_path(
                    prefix,
                    prefix_descriptor,
                    directory=True,
                    label="active-site prefix",
                )
            except OSError as error:
                raise EnvironmentBuildError(
                    "active-site lib64 alias cannot be identity-checked"
                ) from error
            finally:
                os.close(prefix_descriptor)
        elif stat.S_ISDIR(lib64_metadata.st_mode):
            lib64 = _require_directory(lib64, "active-site lib64 directory")
            observed_identity = _directory_identity(
                lib64, "active-site lib64 directory"
            )
            identity = (observed_identity["device"], observed_identity["inode"])
            if identity not in identities:
                library_roots.append(lib64)
                identities.add(identity)
        else:
            raise EnvironmentBuildError("active-site lib64 is unsupported")
    result = []
    site_identities = set()
    for library_root in library_roots:
        candidate = library_root / ("python" + major_minor) / "site-packages"
        metadata = _lstat(candidate)
        if metadata is None:
            continue
        directory = _require_directory(candidate, "active site directory")
        observed_identity = _directory_identity(directory, "active site directory")
        identity = (observed_identity["device"], observed_identity["inode"])
        if identity not in site_identities:
            result.append(directory)
            site_identities.add(identity)
    if not result:
        raise EnvironmentBuildError("no active site-packages directory was found")
    return sorted(set(result), key=str)


def _parse_pyvenv_configuration(
    prefix: Path,
    *,
    required: bool,
    require_system_site_packages: bool = False,
) -> Optional[Dict[str, object]]:
    path = prefix / "pyvenv.cfg"
    metadata = _lstat(path)
    if metadata is None:
        if required:
            raise EnvironmentBuildError("target pyvenv.cfg is missing")
        return None
    path = _require_regular_file(path, "pyvenv.cfg")
    payload = _read_bytes_file(path, "pyvenv.cfg")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise EnvironmentBuildError("pyvenv.cfg is not valid UTF-8") from error
    if re.search(r"/(?:proc/(?:self|[0-9]+)|dev)/fd/[0-9]+", text):
        raise EnvironmentBuildError("pyvenv.cfg contains an ephemeral descriptor path")
    descriptor = _open_absolute_nofollow(path, directory=False, label="pyvenv.cfg")
    try:
        metadata = os.fstat(descriptor)
        _revalidate_fd_path(path, descriptor, directory=False, label="pyvenv.cfg")
    finally:
        os.close(descriptor)
    allowed = {
        "home",
        "include-system-site-packages",
        "version",
        "executable",
        "command",
        "prompt",
    }
    parsed: Dict[str, str] = {}
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if "=" not in raw_line:
            raise EnvironmentBuildError("pyvenv.cfg contains a malformed line")
        key, value = (part.strip() for part in raw_line.split("=", 1))
        normalized = key.lower()
        if normalized not in allowed or normalized in parsed or not value:
            raise EnvironmentBuildError("pyvenv.cfg contains an unknown/duplicate key")
        parsed[normalized] = value
    if (
        require_system_site_packages
        and parsed.get("include-system-site-packages", "").lower() != "true"
    ):
        raise EnvironmentBuildError("venv does not enable system site packages")
    if _sha256_file(path) != _sha256_bytes(payload):
        raise EnvironmentBuildError("pyvenv.cfg changed while being audited")
    return {
        "path": str(path),
        "sha256": _sha256_bytes(payload),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
        "keys": parsed,
    }


def _audit_no_site_path(path_value: str) -> Dict[str, object]:
    path = _lexical_absolute(path_value, "no-site sys.path entry")
    metadata = _lstat(path)
    if metadata is None:
        return {"path": str(path), "kind": "missing"}
    if stat.S_ISDIR(metadata.st_mode):
        directory = _require_directory(path, "no-site sys.path directory")
        descriptor = _open_absolute_nofollow(
            directory, directory=True, label="no-site sys.path directory"
        )
        try:
            names = sorted(os.listdir(descriptor))
            observed = os.fstat(descriptor)
            _revalidate_fd_path(
                directory,
                descriptor,
                directory=True,
                label="no-site sys.path directory",
            )
        finally:
            os.close(descriptor)
        customization = [
            name
            for name in names
            if name.lower().startswith("sitecustomize")
            or name.lower().startswith("usercustomize")
        ]
        if customization:
            raise EnvironmentBuildError(
                "no-site path contains a customization module: "
                + ", ".join(customization)
            )
        return {
            "path": str(directory),
            "kind": "directory",
            "device": int(observed.st_dev),
            "inode": int(observed.st_ino),
        }
    if stat.S_ISREG(metadata.st_mode):
        file_path = _require_regular_file(path, "no-site sys.path archive")
        if file_path.suffix.lower() == ".egg":
            raise EnvironmentBuildError("legacy .egg sys.path archives are forbidden")
        payload = _read_bytes_file(file_path, "no-site sys.path archive")
        try:
            with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
                names = archive.namelist()
        except zipfile.BadZipFile as error:
            raise EnvironmentBuildError(
                "regular no-site sys.path entry is not a valid ZIP"
            ) from error
        if any(
            name.split("/", 1)[0].lower().startswith("sitecustomize")
            or name.split("/", 1)[0].lower().startswith("usercustomize")
            for name in names
        ):
            raise EnvironmentBuildError(
                "no-site sys.path archive contains a customization module"
            )
        return {
            "path": str(file_path),
            "kind": "file",
            "sha256": _sha256_bytes(payload),
            "size": len(payload),
        }
    raise EnvironmentBuildError("no-site sys.path contains an unsupported entry")


def _audit_pth_file(path: Path, site_directory: Path) -> Dict[str, object]:
    payload = _read_bytes_file(path, "site .pth file")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise EnvironmentBuildError("site .pth file is not valid UTF-8") from error
    directives = []
    declared_paths = []
    for ordinal, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("import ") or line.startswith("import\t"):
            directives.append(
                {
                    "line": ordinal,
                    "kind": "executable_present_but_never_executed",
                    "line_sha256": _sha256_bytes(raw_line.encode("utf-8")),
                }
            )
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = site_directory / candidate
        candidate = _lexical_absolute(str(candidate), ".pth path directive")
        metadata = _lstat(candidate)
        if metadata is None:
            binding: Dict[str, object] = {
                "path": str(candidate),
                "kind": "missing",
            }
        elif stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode):
            binding = _audit_no_site_path(str(candidate))
        else:
            raise EnvironmentBuildError(".pth path directive is unsupported")
        declared_paths.append(str(candidate))
        directives.append(
            {"line": ordinal, "kind": "path", "text": raw_line, "binding": binding}
        )
    return {
        "path": str(path),
        "kind": "pth",
        "sha256": _sha256_bytes(payload),
        "execution_policy": "present_but_never_executed",
        "directives": directives,
        "declared_paths": sorted(set(declared_paths)),
        "activated_paths": [],
    }


def _audit_active_site(
    prefix: Path,
    *,
    safe_probe: Mapping[str, object],
    require_pyvenv: bool,
    require_system_site_packages: bool = False,
) -> Dict[str, object]:
    prefix = _require_directory(prefix, "active-site prefix")
    python_version = safe_probe.get("python_version")
    site_directories = _active_site_directories(prefix, python_version)
    major_minor = _python_major_minor(python_version, "active-site")
    stdlib = _require_directory(
        prefix / "lib" / ("python" + major_minor), "active stdlib directory"
    )
    startup_artifacts = []
    declared_pth_paths = []
    scanned = []
    for directory in [stdlib] + site_directories:
        descriptor = _open_absolute_nofollow(
            directory, directory=True, label="active startup directory"
        )
        try:
            names = sorted(os.listdir(descriptor))
            directory_metadata = os.fstat(descriptor)
            _revalidate_fd_path(
                directory,
                descriptor,
                directory=True,
                label="active startup directory",
            )
        finally:
            os.close(descriptor)
        scanned.append(
            {
                "path": str(directory),
                "device": int(directory_metadata.st_dev),
                "inode": int(directory_metadata.st_ino),
            }
        )
        for name in names:
            lowered = name.lower()
            artifact_path = directory / name
            if lowered.startswith("sitecustomize") or lowered.startswith(
                "usercustomize"
            ):
                raise EnvironmentBuildError(
                    "active Python customization modules are forbidden: "
                    + str(artifact_path)
                )
            if lowered.endswith(".egg-link") or lowered.endswith(".egg"):
                raise EnvironmentBuildError(
                    "legacy active-site injection artifacts are forbidden: "
                    + str(artifact_path)
                )
            if lowered.endswith(".pth"):
                artifact = _audit_pth_file(
                    _require_regular_file(artifact_path, "site .pth file"), directory
                )
                startup_artifacts.append(artifact)
                declared_pth_paths.extend(artifact["declared_paths"])
    safe_sys_path = safe_probe.get("sys_path")
    if not isinstance(safe_sys_path, list):
        raise EnvironmentBuildError("active-site no-site sys.path is malformed")
    no_site_bindings = [
        _audit_no_site_path(str(path)) for path in sorted(safe_sys_path)
    ]
    value = {
        "schema_version": "rachel-benchmark-active-site-audit/1.0",
        "prefix": str(prefix),
        "python_version": str(python_version),
        "site_directories": [str(path) for path in site_directories],
        "startup_search_directories": scanned,
        "startup_artifacts": sorted(
            startup_artifacts, key=lambda item: str(item["path"])
        ),
        "declared_pth_paths": sorted(set(declared_pth_paths)),
        "activated_paths": [],
        "runtime_no_site": True,
        "pth_execution_policy": "present_but_never_executed",
        "no_site_sys_path": list(safe_sys_path),
        "no_site_sys_path_bindings": no_site_bindings,
        "forbidden_startup_artifacts": [],
        "pyvenv_configuration": _parse_pyvenv_configuration(
            prefix,
            required=require_pyvenv,
            require_system_site_packages=require_system_site_packages,
        ),
        "user_site_enabled_contract": False,
        "pythonpath_pythonhome_contract": "cleared",
    }
    return _content_bound(value)


def _audit_base_site_surface(
    formal_prefix: Path, safe_probe: Mapping[str, object]
) -> Dict[str, object]:
    """Audit both the formal environment and its resolved no-site runtime.

    A pyvenv/Conda launcher can identify a formal environment while ``-S``
    falls back to the resolved base interpreter.  Both roots are enumerated,
    but only their held, reviewed site directories are injected; Python's
    automatic site startup remains disabled.
    """

    formal_prefix = _require_directory(formal_prefix, "formal base prefix")
    safe_prefix_value = safe_probe.get("prefix")
    if not isinstance(safe_prefix_value, str):
        raise EnvironmentBuildError("base no-site resolved prefix is missing")
    resolved_prefix = _require_directory(
        _lexical_absolute(safe_prefix_value, "base no-site resolved prefix"),
        "base no-site resolved prefix",
    )
    distinct = formal_prefix != resolved_prefix
    prefix_audits = [
        _audit_active_site(
            formal_prefix,
            safe_probe=safe_probe,
            require_pyvenv=distinct,
            require_system_site_packages=distinct,
        )
    ]
    if distinct:
        prefix_audits.append(
            _audit_active_site(
                resolved_prefix,
                safe_probe=safe_probe,
                require_pyvenv=False,
                require_system_site_packages=False,
            )
        )
    site_directories = sorted(
        {
            str(path)
            for audit in prefix_audits
            for path in audit.get("site_directories", [])
        }
    )
    activated_paths = sorted(
        {
            str(path)
            for audit in prefix_audits
            for path in audit.get("activated_paths", [])
        }
    )
    return _content_bound(
        {
            "schema_version": "rachel-benchmark-base-site-surface/1.0",
            "formal_prefix": str(formal_prefix),
            "resolved_no_site_prefix": str(resolved_prefix),
            "distinct_prefixes": distinct,
            "prefix_audits": prefix_audits,
            # Only explicit site directories are injected.  PTH directives are
            # content-bound above but never interpreted as startup inputs.
            "site_directories": site_directories,
            "activated_paths": activated_paths,
            "runtime_no_site": True,
            "pth_execution_policy": "present_but_never_executed",
        }
    )


def _resolved_base_audit(
    base_site_surface: Mapping[str, object],
) -> Dict[str, object]:
    """Select only the resolved/base site inherited by a nested target venv."""

    _validate_content_sha(base_site_surface, "base site surface")
    if base_site_surface.get("schema_version") != (
        "rachel-benchmark-base-site-surface/1.0"
    ):
        raise EnvironmentBuildError("base site surface schema differs")
    resolved_prefix = base_site_surface.get("resolved_no_site_prefix")
    audits = base_site_surface.get("prefix_audits")
    if not isinstance(resolved_prefix, str) or not isinstance(audits, list):
        raise EnvironmentBuildError("base site surface is malformed")
    matches = [
        audit
        for audit in audits
        if isinstance(audit, dict) and audit.get("prefix") == resolved_prefix
    ]
    if len(matches) != 1:
        raise EnvironmentBuildError("base site surface lacks one resolved-prefix audit")
    _validate_content_sha(matches[0], "resolved base site audit")
    return dict(matches[0])


def _base_prefix_audits(
    base_site_surface: Mapping[str, object],
) -> Tuple[Dict[str, object], ...]:
    _validate_content_sha(base_site_surface, "base site surface")
    audits = base_site_surface.get("prefix_audits")
    if not isinstance(audits, list) or not audits:
        raise EnvironmentBuildError("base site surface prefix audits are malformed")
    result = []
    for audit in audits:
        if not isinstance(audit, dict):
            raise EnvironmentBuildError("base site surface prefix audit is malformed")
        _validate_content_sha(audit, "base site prefix audit")
        result.append(dict(audit))
    return tuple(result)


def _validate_probe_startup_surface(
    detailed: Mapping[str, object],
    safe_probes: Sequence[Mapping[str, object]],
    audits: Sequence[Mapping[str, object]],
    label: str,
) -> None:
    if (
        detailed.get("sitecustomize_loaded") is not False
        or detailed.get("usercustomize_loaded") is not False
    ):
        raise EnvironmentBuildError(label + " loaded a customization module")
    allowed_paths = set()
    for probe in safe_probes:
        paths = probe.get("sys_path")
        if not isinstance(paths, list):
            raise EnvironmentBuildError(label + " no-site sys.path is malformed")
        allowed_paths.update(str(path) for path in paths)
    for audit in audits:
        _validate_content_sha(audit, label + " active-site audit")
        directories = audit.get("site_directories")
        activated = audit.get("activated_paths")
        if (
            not isinstance(directories, list)
            or activated != []
            or audit.get("runtime_no_site") is not True
            or audit.get("pth_execution_policy") != "present_but_never_executed"
        ):
            raise EnvironmentBuildError(label + " active-site audit is malformed")
        allowed_paths.update(str(path) for path in directories)
    sys_path = detailed.get("sys_path")
    if not isinstance(sys_path, list) or any(
        not isinstance(path, str) for path in sys_path
    ):
        raise EnvironmentBuildError(label + " sys.path probe is malformed")
    if set(sys_path) != allowed_paths:
        raise EnvironmentBuildError(label + " active sys.path surface differs")


def _validate_official_environment(path: Path) -> Dict[str, object]:
    path = _require_regular_file(path, "official ShreddingNet env.yaml")
    payload = _read_bytes_file(path, "official ShreddingNet env.yaml")
    observed_sha = _sha256_bytes(payload)
    if observed_sha != OFFICIAL_ENV_YAML_SHA256:
        raise EnvironmentBuildError("official ShreddingNet env.yaml SHA-256 differs")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise EnvironmentBuildError(
            "official ShreddingNet env.yaml is not valid UTF-8"
        ) from error
    pins: Dict[str, str] = {}
    for match in re.finditer(
        r"^\s*-\s*([A-Za-z0-9_.-]+)==([^\s#]+)\s*$", text, flags=re.MULTILINE
    ):
        pins[_canonical_distribution_name(match.group(1))] = match.group(2)
    required = {
        "torch": "2.5.1",
        "torch-geometric": "2.6.1",
        "opencv-python": "4.10.0.84",
        "torchvision": "0.20.1",
        "numpy": "2.2.3",
        "scipy": "1.14.1",
        "pillow": "11.0.0",
    }
    if any(pins.get(name) != version for name, version in required.items()):
        raise EnvironmentBuildError("official ShreddingNet dependency pins differ")
    return {
        "commit": OFFICIAL_SHREDDINGNET_COMMIT,
        "env_yaml": str(path),
        "env_yaml_sha256": observed_sha,
        "audited_pins": required,
    }


def _validate_base_probe(
    value: Mapping[str, object],
    python: Path,
    *,
    safe_probe: Mapping[str, object],
    active_site_audit: Mapping[str, object],
) -> None:
    if value.get("executable") != os.path.realpath(str(python)):
        raise EnvironmentBuildError("base Python executable identity differs")
    if value.get("prefix") != os.path.realpath(str(python.parent.parent)):
        raise EnvironmentBuildError(
            "base Python sys.prefix differs from its declared environment root"
        )
    if (
        value.get("python_version") != safe_probe.get("python_version")
        or value.get("platform_system") != "Linux"
        or value.get("platform_machine") not in {"x86_64", "AMD64"}
    ):
        raise EnvironmentBuildError("base runtime platform identity differs")
    version = value.get("python_version")
    if not isinstance(version, str):
        raise EnvironmentBuildError("base Python version is missing")
    try:
        major, minor = (int(part) for part in version.split(".")[:2])
    except (TypeError, ValueError) as error:
        raise EnvironmentBuildError("base Python version is malformed") from error
    if (major, minor) < (3, 8):
        raise EnvironmentBuildError("base Python must be at least Python 3.8")
    if value.get("user_site_enabled") is not False:
        raise EnvironmentBuildError("base Python isolated probe enables user site")
    if value.get("torch_module_version") != EXPECTED_TORCH_MODULE_VERSION:
        raise EnvironmentBuildError("base Torch must be exactly 2.5.1+cu124")
    if value.get("torch_cuda_build_version") != EXPECTED_TORCH_CUDA_VERSION:
        raise EnvironmentBuildError("base Torch must be the CUDA 12.4 build")
    tracked = value.get("tracked")
    if not isinstance(tracked, dict) or set(tracked) != {
        name for name, _module in BASE_TRACKED_DISTRIBUTIONS
    }:
        raise EnvironmentBuildError("base inherited dependency set differs")
    for name, _module in INHERITED_DISTRIBUTIONS + PIP_RUNTIME_DISTRIBUTION:
        item = tracked.get(name)
        if not isinstance(item, dict):
            raise EnvironmentBuildError("base dependency probe is malformed: " + name)
        for field in ("distribution_version", "distribution_location", "import_origin"):
            if not isinstance(item.get(field), str) or not item.get(field):
                raise EnvironmentBuildError(
                    "base dependency lacks " + field + ": " + name
                )
    pip_item = tracked.get("pip")
    if not isinstance(pip_item, dict):
        raise EnvironmentBuildError("base pip runtime identity is malformed")
    for field in ("distribution_version", "distribution_location", "import_origin"):
        if not isinstance(pip_item.get(field), str) or not pip_item.get(field):
            raise EnvironmentBuildError("base pip runtime lacks " + field)
    numpy_item = tracked.get("numpy")
    if (
        not isinstance(numpy_item, dict)
        or numpy_item.get("distribution_version") != EXPECTED_NUMPY_ADAPTATION_VERSION
    ):
        raise EnvironmentBuildError(
            "base NumPy must be the intentional 2.1.3 Rachel adaptation"
        )
    exact_inherited = {
        "torch": EXPECTED_TORCH_MODULE_VERSION,
        "numpy": EXPECTED_NUMPY_ADAPTATION_VERSION,
        "Pillow": EXPECTED_PILLOW_VERSION,
    }
    for name, expected in exact_inherited.items():
        item = tracked.get(name)
        if not isinstance(item, dict) or item.get("distribution_version") != expected:
            raise EnvironmentBuildError("base inherited version differs: " + name)
    torchvision_item = tracked.get("torchvision")
    if not isinstance(torchvision_item, dict) or torchvision_item.get(
        "distribution_version"
    ) not in {
        EXPECTED_TORCHVISION_BASE_VERSION,
        EXPECTED_TORCHVISION_BASE_VERSION + "+cu124",
    }:
        raise EnvironmentBuildError("base torchvision version differs")
    _validate_probe_startup_surface(
        value, (safe_probe,), (active_site_audit,), "base runtime"
    )


def _validate_target_probes(
    *,
    root: Path,
    python: Path,
    base_probe: Mapping[str, object],
    inherited_base_probe: Mapping[str, object],
    safe_probes: Sequence[Mapping[str, object]],
    active_site_audits: Sequence[Mapping[str, object]],
    live: Mapping[str, object],
    detailed: Mapping[str, object],
) -> None:
    if live.get("executable") != os.path.realpath(str(python)) or live.get(
        "prefix"
    ) != str(root):
        raise EnvironmentBuildError("target queue live-probe identity differs")
    if live.get("user_site_enabled") is not False:
        raise EnvironmentBuildError("target Python enables user site")
    imports = live.get("imports")
    if imports != {
        "torch": "ok",
        "torch_geometric": "ok",
        "cv2": "ok",
        "numpy": "ok",
        "scipy": "ok",
    }:
        raise EnvironmentBuildError("target queue dependency imports failed")
    packages = live.get("packages")
    if not isinstance(packages, dict):
        raise EnvironmentBuildError("target queue package probe is malformed")
    if packages.get("torch-geometric") != "2.6.1":
        raise EnvironmentBuildError("target torch-geometric version differs")
    if packages.get("opencv-python") != "4.10.0.84":
        raise EnvironmentBuildError("target opencv-python version differs")
    if packages.get("scipy") != "1.14.1":
        raise EnvironmentBuildError("target scipy version differs")
    if packages.get("torch") != EXPECTED_TORCH_MODULE_VERSION:
        raise EnvironmentBuildError("target Torch distribution version differs")
    if packages.get("numpy") != EXPECTED_NUMPY_ADAPTATION_VERSION:
        raise EnvironmentBuildError("target NumPy adaptation version differs")
    if packages.get("opencv-python-headless") is not None:
        raise EnvironmentBuildError("conflicting opencv-python-headless is visible")
    if detailed.get("executable") != os.path.realpath(str(python)) or detailed.get(
        "prefix"
    ) != str(root):
        raise EnvironmentBuildError("target detailed-probe identity differs")
    if detailed.get("torch_module_version") != EXPECTED_TORCH_MODULE_VERSION:
        raise EnvironmentBuildError("target Torch module version differs")
    if detailed.get("torch_cuda_build_version") != EXPECTED_TORCH_CUDA_VERSION:
        raise EnvironmentBuildError("target Torch CUDA build version differs")
    if detailed.get("opencv_python_headless_version") is not None:
        raise EnvironmentBuildError("target has conflicting OpenCV distribution")
    target_tracked = detailed.get("tracked")
    base_tracked = inherited_base_probe.get("tracked")
    if not isinstance(target_tracked, dict) or set(target_tracked) != {
        name for name, _module in TARGET_TRACKED_DISTRIBUTIONS
    }:
        raise EnvironmentBuildError("target tracked dependency set differs")
    if not isinstance(base_tracked, dict):
        raise EnvironmentBuildError("base tracked dependency set is malformed")
    for name, _module in INHERITED_DISTRIBUTIONS + PIP_RUNTIME_DISTRIBUTION:
        target_item = target_tracked.get(name)
        base_item = base_tracked.get(name)
        if not isinstance(target_item, dict) or not isinstance(base_item, dict):
            raise EnvironmentBuildError("inherited dependency probe differs: " + name)
        for field in ("distribution_version", "distribution_location", "import_origin"):
            if target_item.get(field) != base_item.get(field):
                raise EnvironmentBuildError(
                    "inherited dependency drifted from base: " + name + " " + field
                )
    for name, expected, _module in ADDITIONS:
        item = target_tracked.get(name)
        if not isinstance(item, dict) or item.get("distribution_version") != expected:
            raise EnvironmentBuildError("added dependency version differs: " + name)
        for field in ("distribution_location", "import_origin"):
            value = item.get(field)
            if not isinstance(value, str) or not _is_within(Path(value), root):
                raise EnvironmentBuildError("added dependency escaped venv: " + name)
    local = detailed.get("local_distributions")
    if not isinstance(local, list):
        raise EnvironmentBuildError("target local distribution inventory is malformed")
    allowed_local = {"torch-geometric", "opencv-python", "scipy"}
    local_names = {item.get("name") for item in local if isinstance(item, dict)}
    if local_names != allowed_local:
        raise EnvironmentBuildError("target venv lacks required local distributions")
    source_formal_value = base_probe.get("prefix")
    resolved_base_value = base_probe.get("base_prefix")
    if not isinstance(source_formal_value, str) or not isinstance(
        resolved_base_value, str
    ):
        raise EnvironmentBuildError("base runtime prefix identity is malformed")
    source_formal = Path(source_formal_value)
    resolved_base = Path(resolved_base_value)
    if source_formal != resolved_base:
        for audit in active_site_audits:
            for field in ("site_directories", "activated_paths"):
                paths = audit.get(field)
                if not isinstance(paths, list):
                    raise EnvironmentBuildError("target active-site audit is malformed")
                if any(
                    _is_within(Path(str(path)), source_formal)
                    and not _is_within(Path(str(path)), resolved_base)
                    for path in paths
                ):
                    raise EnvironmentBuildError(
                        "target runtime inherited the source formal site"
                    )
    _validate_probe_startup_surface(
        detailed, safe_probes, active_site_audits, "target runtime"
    )


def _exact_keys(value: Mapping[str, object], expected: set, label: str) -> None:
    if set(value) != expected:
        raise EnvironmentBuildError(label + " keys differ")


def _validate_wheel_lock(
    path: Path,
    expected_file_sha256: object,
    safe_probe: Mapping[str, object],
) -> Dict[str, object]:
    lock_path = _require_read_only_file(path, "reviewed wheel lock")
    value, observed_file_sha256 = _load_json_with_sha(lock_path, "reviewed wheel lock")
    if (
        expected_file_sha256 != REVIEWED_WHEEL_LOCK_FILE_SHA256
        or observed_file_sha256 != REVIEWED_WHEEL_LOCK_FILE_SHA256
    ):
        raise EnvironmentBuildError("reviewed wheel lock file SHA-256 differs")
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "target",
            "official_index",
            "packages",
            "review",
            "content_sha256",
        },
        "reviewed wheel lock",
    )
    _validate_content_sha(value, "reviewed wheel lock")
    if (
        value.get("schema_version") != WHEEL_LOCK_SCHEMA
        or value.get("status") != "reviewed_frozen_wheel_authority"
    ):
        raise EnvironmentBuildError("reviewed wheel lock status/schema differs")
    target = value.get("target")
    if target != {
        "python_major_minor": "3.12",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
    }:
        raise EnvironmentBuildError("reviewed wheel lock target differs")
    if (
        _python_major_minor(safe_probe.get("python_version"), "wheel target") != "3.12"
        or safe_probe.get("platform_system") != "Linux"
        or safe_probe.get("platform_machine") not in {"x86_64", "AMD64"}
    ):
        raise EnvironmentBuildError("base runtime differs from wheel lock target")
    official_index = value.get("official_index")
    review = value.get("review")
    if not isinstance(official_index, dict) or not isinstance(review, dict):
        raise EnvironmentBuildError("reviewed wheel lock authorities are malformed")
    _exact_keys(
        official_index,
        {"name", "download_host", "project_json_urls", "simple_json_urls"},
        "wheel lock official index",
    )
    if (
        official_index.get("name") != "PyPI"
        or official_index.get("download_host") != "files.pythonhosted.org"
        or review
        != {
            "frozen_before_any_wheel_download": True,
            "source": "official PyPI project-version JSON",
            "test_or_real_data_accessed": False,
            "independent_authority_cross_check": {
                "accept": "application/vnd.pypi.simple.v1+json",
                "result": "all_three_filename_size_sha256_equal",
                "source": "official PyPI PEP 691 Simple JSON",
            },
        }
    ):
        raise EnvironmentBuildError("reviewed wheel lock review authority differs")
    packages = value.get("packages")
    if not isinstance(packages, list) or len(packages) != len(ADDITIONS):
        raise EnvironmentBuildError("reviewed wheel lock package count differs")
    expected_identities = {(name, version) for name, version, _module in ADDITIONS}
    observed_identities = set()
    normalized_packages = []
    project_urls = official_index.get("project_json_urls")
    simple_urls = official_index.get("simple_json_urls")
    if not isinstance(project_urls, dict) or not isinstance(simple_urls, dict):
        raise EnvironmentBuildError("reviewed wheel lock JSON sources are malformed")
    for package in packages:
        if not isinstance(package, dict):
            raise EnvironmentBuildError("reviewed wheel lock package is malformed")
        _exact_keys(
            package,
            {
                "distribution",
                "version",
                "filename",
                "filename_tags",
                "sha256",
                "size",
                "url",
                "source_json_url",
                "source_simple_json_url",
                "upload_time_iso_8601",
            },
            "reviewed wheel lock package",
        )
        name = package.get("distribution")
        version = package.get("version")
        filename = package.get("filename")
        tags = package.get("filename_tags")
        url = package.get("url")
        if (
            not isinstance(name, str)
            or not isinstance(version, str)
            or (name, version) not in expected_identities
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".whl")
            or not _is_sha256(package.get("sha256"))
            or type(package.get("size")) is not int
            or int(package["size"]) <= 0
            or not isinstance(tags, dict)
            or set(tags) != {"python", "abi", "platform"}
            or any(not isinstance(tags.get(key), str) for key in tags)
            or not isinstance(url, str)
        ):
            raise EnvironmentBuildError("reviewed wheel lock package fields differ")
        stem_parts = filename[:-4].rsplit("-", 3)
        identity_parts = stem_parts[0].rsplit("-", 1) if stem_parts else []
        if (
            len(stem_parts) != 4
            or len(identity_parts) != 2
            or _canonical_distribution_name(identity_parts[0]) != name
            or identity_parts[1] != version
            or stem_parts[1:] != [tags["python"], tags["abi"], tags["platform"]]
        ):
            raise EnvironmentBuildError("reviewed wheel filename tags differ")
        parsed_url = urlsplit(url)
        if (
            parsed_url.scheme != "https"
            or parsed_url.netloc != "files.pythonhosted.org"
            or Path(unquote(parsed_url.path)).name != filename
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise EnvironmentBuildError("reviewed wheel URL differs")
        json_url = "https://pypi.org/pypi/{}/{}/json".format(name, version)
        simple_url = "https://pypi.org/simple/{}/".format(name)
        if (
            package.get("source_json_url") != json_url
            or package.get("source_simple_json_url") != simple_url
            or project_urls.get(name) != json_url
            or simple_urls.get(name) != simple_url
            or not isinstance(package.get("upload_time_iso_8601"), str)
            or not str(package["upload_time_iso_8601"]).endswith("Z")
        ):
            raise EnvironmentBuildError("reviewed PyPI JSON provenance differs")
        observed_identities.add((name, version))
        normalized_packages.append(dict(package))
    if observed_identities != expected_identities:
        raise EnvironmentBuildError("reviewed wheel lock identities differ")
    normalized_packages.sort(key=lambda item: str(item["distribution"]))
    return {
        "path": str(lock_path),
        "file_sha256": REVIEWED_WHEEL_LOCK_FILE_SHA256,
        "content_sha256": value["content_sha256"],
        "packages": normalized_packages,
        "target": dict(target),
        "official_index": dict(official_index),
        "review": dict(review),
    }


def _inspect_wheel(
    path: Path, authority: Optional[Mapping[str, object]] = None
) -> Dict[str, object]:
    path = _require_regular_file(path, "downloaded wheel")
    try:
        wheel_bytes = _read_bytes_file(path, "downloaded wheel")
        with zipfile.ZipFile(io.BytesIO(wheel_bytes), "r") as archive:
            names = archive.namelist()
            file_names = [name for name in names if not name.endswith("/")]
            if len(file_names) != len(set(file_names)) or any(
                not name
                or name.startswith("/")
                or "\\" in name
                or ".." in Path(name).parts
                for name in file_names
            ):
                raise EnvironmentBuildError("wheel archive paths are unsafe")
            for info in archive.infolist():
                unix_mode = (int(info.external_attr) >> 16) & 0o170000
                if unix_mode == stat.S_IFLNK:
                    raise EnvironmentBuildError("wheel archive contains a symlink")
            metadata_candidates = [
                name for name in file_names if name.endswith(".dist-info/METADATA")
            ]
            wheel_candidates = [
                name for name in file_names if name.endswith(".dist-info/WHEEL")
            ]
            record_candidates = [
                name for name in file_names if name.endswith(".dist-info/RECORD")
            ]
            if (
                len(metadata_candidates) != 1
                or len(wheel_candidates) != 1
                or len(record_candidates) != 1
            ):
                raise EnvironmentBuildError("wheel lacks unique dist-info authorities")
            metadata_name = metadata_candidates[0]
            wheel_name = wheel_candidates[0]
            record_name = record_candidates[0]
            dist_info = metadata_name.rsplit("/", 1)[0]
            if (
                "/" in dist_info
                or wheel_name.rsplit("/", 1)[0] != dist_info
                or record_name.rsplit("/", 1)[0] != dist_info
            ):
                raise EnvironmentBuildError("wheel dist-info roots differ")
            metadata_bytes = archive.read(metadata_name)
            wheel_metadata_bytes = archive.read(wheel_name)
            record_bytes = archive.read(record_name)
            metadata = Parser().parsestr(
                metadata_bytes.decode("utf-8", errors="strict")
            )
            wheel_metadata = Parser().parsestr(
                wheel_metadata_bytes.decode("utf-8", errors="strict")
            )
            record_rows = list(
                csv.reader(io.StringIO(record_bytes.decode("utf-8", errors="strict")))
            )
            if len(record_rows) != len(file_names):
                raise EnvironmentBuildError("wheel RECORD membership differs")
            record_paths = set()
            verified_rows = 0
            for row in record_rows:
                if len(row) != 3:
                    raise EnvironmentBuildError("wheel RECORD row is malformed")
                relative, declared, declared_size = row
                if relative in record_paths or relative not in file_names:
                    raise EnvironmentBuildError("wheel RECORD path set differs")
                record_paths.add(relative)
                member = archive.read(relative)
                if relative == record_name:
                    if declared or declared_size:
                        raise EnvironmentBuildError(
                            "wheel RECORD self-entry must be unhashed"
                        )
                    continue
                if not declared.startswith("sha256=") or not declared_size:
                    raise EnvironmentBuildError("wheel RECORD lacks SHA-256/size")
                encoded = (
                    base64.urlsafe_b64encode(hashlib.sha256(member).digest())
                    .decode("ascii")
                    .rstrip("=")
                )
                if declared.split("=", 1)[1] != encoded or int(declared_size) != len(
                    member
                ):
                    raise EnvironmentBuildError("wheel RECORD content differs")
                verified_rows += 1
            if record_paths != set(file_names):
                raise EnvironmentBuildError("wheel RECORD omits archive members")
    except (
        OSError,
        UnicodeError,
        ValueError,
        csv.Error,
        zipfile.BadZipFile,
        KeyError,
    ) as error:
        raise EnvironmentBuildError("cannot inspect downloaded wheel") from error
    names = metadata.get_all("Name") or []
    versions = metadata.get_all("Version") or []
    name = names[0] if len(names) == 1 else None
    version = versions[0] if len(versions) == 1 else None
    if (
        not name
        or not version
        or not metadata.get("Metadata-Version")
        or not wheel_metadata.get("Wheel-Version")
        or wheel_metadata.get("Root-Is-Purelib") not in {"true", "false"}
    ):
        raise EnvironmentBuildError("downloaded wheel metadata is incomplete")
    dist_info_identity = dist_info[: -len(".dist-info")].rsplit("-", 1)
    filename_identity = path.name[:-4].rsplit("-", 3)[0].rsplit("-", 1)
    if (
        len(dist_info_identity) != 2
        or len(filename_identity) != 2
        or _canonical_distribution_name(dist_info_identity[0])
        != _canonical_distribution_name(name)
        or _canonical_distribution_name(filename_identity[0])
        != _canonical_distribution_name(name)
        or dist_info_identity[1] != version
        or filename_identity[1] != version
    ):
        raise EnvironmentBuildError("wheel filename/dist-info identity differs")
    tags = wheel_metadata.get_all("Tag") or []
    result: Dict[str, object] = {
        "filename": path.name,
        "distribution": _canonical_distribution_name(name),
        "version": version,
        "sha256": _sha256_bytes(wheel_bytes),
        "size": len(wheel_bytes),
        "wheel_version": wheel_metadata.get("Wheel-Version"),
        "wheel_tags": sorted(tags),
        "metadata_sha256": _sha256_bytes(metadata_bytes),
        "wheel_metadata_sha256": _sha256_bytes(wheel_metadata_bytes),
        "record_sha256": _sha256_bytes(record_bytes),
        "record_rows": len(record_rows),
        "record_verified_sha256_rows": verified_rows,
    }
    if authority is not None:
        for field in ("filename", "distribution", "version", "sha256", "size"):
            if result[field] != authority.get(field):
                raise EnvironmentBuildError("wheel differs from reviewed " + field)
        filename_tags = authority.get("filename_tags")
        if not isinstance(filename_tags, dict):
            raise EnvironmentBuildError("reviewed wheel tags are malformed")
        expected_tags = sorted(
            "{}-{}-{}".format(python_tag, abi_tag, platform_tag)
            for python_tag in str(filename_tags["python"]).split(".")
            for abi_tag in str(filename_tags["abi"]).split(".")
            for platform_tag in str(filename_tags["platform"]).split(".")
        )
        if result["wheel_version"] != "1.0" or result["wheel_tags"] != expected_tags:
            raise EnvironmentBuildError("wheel WHEEL tags differ from filename tags")
    return result


def _download_and_lock_wheels(
    python: Path,
    root: Path,
    reviewed_lock: Mapping[str, object],
    execution_anchor: _ExecutionAnchor,
    safe_probe: Mapping[str, object],
    audits: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    _validate_pip_runtime_audits(audits, "pinned wheel download")
    _mkdir_new_anchored(
        root / ".environment_build_evidence", "environment build evidence"
    )
    wheelhouse = _mkdir_new_anchored(root / WHEELHOUSE_RELATIVE_PATH, "wheelhouse")
    packages = reviewed_lock.get("packages")
    if not isinstance(packages, list) or len(packages) != len(ADDITIONS):
        raise EnvironmentBuildError("reviewed wheel authority is malformed")
    authorities = {}
    urls = []
    for item in packages:
        if not isinstance(item, dict):
            raise EnvironmentBuildError("reviewed wheel authority entry is malformed")
        filename = item.get("filename")
        url = item.get("url")
        if not isinstance(filename, str) or not isinstance(url, str):
            raise EnvironmentBuildError("reviewed wheel authority fields are malformed")
        authorities[filename] = item
        urls.append(url)
    download_destination = execution_anchor.child_path(
        wheelhouse, "pinned download destination"
    )
    download = _run_explicit_site_payload(
        python,
        PIP_NO_SITE_RUN_CODE,
        (
            "--isolated",
            "--disable-pip-version-check",
            "download",
            "--no-cache-dir",
            "--no-deps",
            "--only-binary=:all:",
            "--dest",
            str(download_destination),
            *urls,
        ),
        safe_probe=safe_probe,
        audits=audits,
        timeout=3600,
        label="pinned wheel download",
        primary_anchor=execution_anchor,
    )
    wheel_paths = _list_regular_files(wheelhouse, "downloaded wheelhouse")
    if len(wheel_paths) != len(ADDITIONS):
        raise EnvironmentBuildError(
            "pinned download did not produce exactly three wheels"
        )
    if {path.name for path in wheel_paths} != set(authorities):
        raise EnvironmentBuildError("downloaded wheel filenames differ")
    wheels = [_inspect_wheel(path, authorities[path.name]) for path in wheel_paths]
    wheels.sort(key=lambda value: str(value["distribution"]))
    lock_lines = []
    for authority in sorted(packages, key=lambda value: str(value["distribution"])):
        lock_lines.append(
            str(authority["distribution"])
            + "=="
            + str(authority["version"])
            + " --hash=sha256:"
            + str(authority["sha256"])
        )
    lock_payload = ("\n".join(lock_lines) + "\n").encode("utf-8")
    lock_path = root / LOCK_RELATIVE_PATH
    _write_bytes_new_immutable(lock_path, lock_payload, "hashed requirements lock")
    for wheel in wheel_paths:
        _chmod_nofollow(wheel, 0o444, directory=False, label="downloaded wheel")
    _chmod_nofollow(wheelhouse, 0o555, directory=True, label="wheelhouse")
    return wheels, {
        "argv": _download_evidence_argv(root, urls),
        "argv_policy": (
            "fd_paths_canonicalized_to_lexical_paths_static_payloads_recorded_by_sha256"
        ),
        "stdout_sha256": _sha256_bytes(download.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(download.stderr.encode("utf-8")),
        "requirements_lock": str(lock_path),
        "requirements_lock_sha256": _sha256_file(lock_path),
    }


def _normalize_install_report_paths(
    report_path: Path,
    root: Path,
    reviewed_lock: Mapping[str, object],
    *,
    execution_anchor: _ExecutionAnchor,
) -> Dict[str, object]:
    """Canonicalize descriptor-backed wheel URLs in pip's install report."""

    root = _lexical_absolute(str(root), "pip report environment root")
    report_path = _lexical_absolute(str(report_path), "pip install report")
    if (
        execution_anchor.root != root
        or report_path != root / INSTALL_REPORT_RELATIVE_PATH
    ):
        raise EnvironmentBuildError("pip install report anchor/path identity differs")
    execution_anchor.validate()
    evidence_descriptor: Optional[int] = None
    report_descriptor: Optional[int] = None
    try:
        evidence_descriptor = os.open(
            ".environment_build_evidence",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=execution_anchor.descriptor,
        )
        report_descriptor = os.open(
            "pip_install_report.json",
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=evidence_descriptor,
        )
        metadata = os.fstat(report_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise EnvironmentBuildError("pip install report is not regular")
        payload, confirmed = _read_fd_payload(report_descriptor, "pip install report")
    except OSError as error:
        raise EnvironmentBuildError(
            "pip install report cannot be opened through the held root"
        ) from error
    finally:
        if report_descriptor is not None:
            os.close(report_descriptor)
    try:
        report = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as error:
        if evidence_descriptor is not None:
            os.close(evidence_descriptor)
        raise EnvironmentBuildError(
            "pip install report is not readable JSON"
        ) from error
    if not isinstance(report, dict):
        if evidence_descriptor is not None:
            os.close(evidence_descriptor)
        raise EnvironmentBuildError("pip install report must be a JSON object")
    packages = reviewed_lock.get("packages")
    install = report.get("install")
    if not isinstance(packages, list) or not isinstance(install, list):
        if evidence_descriptor is not None:
            os.close(evidence_descriptor)
        raise EnvironmentBuildError("pip install report package set is malformed")
    authorities = {
        str(item["filename"]): item for item in packages if isinstance(item, dict)
    }
    normalized = []
    seen = set()
    for item in install:
        if not isinstance(item, dict):
            if evidence_descriptor is not None:
                os.close(evidence_descriptor)
            raise EnvironmentBuildError("pip install report entry is malformed")
        download_info = item.get("download_info")
        url = download_info.get("url") if isinstance(download_info, dict) else None
        if not isinstance(url, str):
            if evidence_descriptor is not None:
                os.close(evidence_descriptor)
            raise EnvironmentBuildError("pip install report wheel URL is malformed")
        parsed = urlsplit(url)
        try:
            source_path = Path(unquote(parsed.path, errors="strict"))
        except UnicodeError as error:
            if evidence_descriptor is not None:
                os.close(evidence_descriptor)
            raise EnvironmentBuildError(
                "pip install report wheel URL is malformed"
            ) from error
        filename = source_path.name
        authority = authorities.get(filename)
        if authority is None or filename in seen:
            if evidence_descriptor is not None:
                os.close(evidence_descriptor)
            raise EnvironmentBuildError(
                "pip install report wheel filename is not uniquely reviewed"
            )
        canonical_path = root / WHEELHOUSE_RELATIVE_PATH / filename
        anchored_path = execution_anchor.child_path(
            canonical_path, "pip report anchored wheel path"
        )
        allowed_urls = {anchored_path.as_uri(), canonical_path.as_uri()}
        if (
            parsed.scheme != "file"
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or url not in allowed_urls
            or source_path not in {anchored_path, canonical_path}
        ):
            if evidence_descriptor is not None:
                os.close(evidence_descriptor)
            raise EnvironmentBuildError(
                "pip install report wheel URL escaped the held wheelhouse"
            )
        canonical_url = canonical_path.as_uri()
        download_info["url"] = canonical_url
        normalized.append(
            {
                "filename": filename,
                "source_url": url,
                "canonical_path": str(canonical_path),
                "canonical_url": canonical_url,
            }
        )
        seen.add(filename)
    if seen != set(authorities):
        if evidence_descriptor is not None:
            os.close(evidence_descriptor)
        raise EnvironmentBuildError("pip install report reviewed wheel set differs")
    normalization = _content_bound(
        {
            "schema_version": "rachel-benchmark-pip-report-path-normalization/1.0",
            "policy": "execution-anchor-file-url-to-canonical-lexical-wheel-path",
            "entries": sorted(normalized, key=lambda item: str(item["filename"])),
        }
    )
    assert evidence_descriptor is not None
    try:
        _replace_dirfd_file(
            evidence_descriptor,
            report_path.name,
            _formatted_json_bytes(report),
            confirmed.st_mode,
            "pip install report path normalization",
        )
    finally:
        os.close(evidence_descriptor)
    execution_anchor.validate()
    return normalization


def _install_locked_wheels(
    python: Path,
    root: Path,
    reviewed_lock: Mapping[str, object],
    execution_anchor: _ExecutionAnchor,
    safe_probe: Mapping[str, object],
    audits: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    _validate_pip_runtime_audits(audits, "hash-locked wheel install")
    report_path = root / INSTALL_REPORT_RELATIVE_PATH
    lock_path = root / LOCK_RELATIVE_PATH
    wheelhouse = root / WHEELHOUSE_RELATIVE_PATH
    anchored_root = execution_anchor.child_path(root, "hash-locked install prefix")
    anchored_report = execution_anchor.child_path(
        report_path, "pip install report output"
    )
    anchored_wheelhouse = execution_anchor.child_path(
        wheelhouse, "hash-locked wheelhouse input"
    )
    anchored_lock = execution_anchor.child_path(lock_path, "hashed requirements input")
    install = _run_explicit_site_payload(
        python,
        PIP_NO_SITE_RUN_CODE,
        (
            "--isolated",
            "--disable-pip-version-check",
            "install",
            "--no-index",
            "--no-deps",
            "--require-hashes",
            "--ignore-installed",
            "--prefix",
            str(anchored_root),
            "--find-links",
            str(anchored_wheelhouse),
            "--report",
            str(anchored_report),
            "-r",
            str(anchored_lock),
        ),
        safe_probe=safe_probe,
        audits=audits,
        timeout=3600,
        label="hash-locked wheel install",
        primary_anchor=execution_anchor,
    )
    report_path_normalization = _normalize_install_report_paths(
        report_path,
        root,
        reviewed_lock,
        execution_anchor=execution_anchor,
    )
    report_path = _require_regular_file(report_path, "pip install report")
    installed = _validate_install_report(report_path, root, reviewed_lock)
    _chmod_nofollow(report_path, 0o444, directory=False, label="pip install report")
    return {
        "argv": _install_evidence_argv(root),
        "argv_policy": (
            "fd_paths_canonicalized_to_lexical_paths_static_payloads_recorded_by_sha256"
        ),
        "stdout_sha256": _sha256_bytes(install.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(install.stderr.encode("utf-8")),
        "report": str(report_path),
        "report_sha256": _sha256_file(report_path),
        "report_path_normalization": report_path_normalization,
        "installed": installed,
    }


def _download_evidence_argv(root: Path, urls: Sequence[str]) -> List[str]:
    return [
        str(root / "bin/python"),
        "-I",
        "-S",
        "-B",
        "-c",
        "<explicit-site-stdin-launcher-sha256="
        + _sha256_bytes(EXPLICIT_SITE_STDIN_LAUNCHER_CODE.encode("utf-8"))
        + ">",
        "<content-sha256-bound-exact-no-site-sys-path-count-and-entries>",
        "<stdin-explicit-site-bootstrap-sha256="
        + _sha256_bytes(EXPLICIT_SITE_BOOTSTRAP_CODE.encode("utf-8"))
        + ">",
        "<content-bound-explicit-site-configuration>",
        "<pip-no-site-run-sha256="
        + _sha256_bytes(PIP_NO_SITE_RUN_CODE.encode("utf-8"))
        + ">",
        "--isolated",
        "--disable-pip-version-check",
        "download",
        "--no-cache-dir",
        "--no-deps",
        "--only-binary=:all:",
        "--dest",
        str(root / WHEELHOUSE_RELATIVE_PATH),
        *urls,
    ]


def _install_evidence_argv(root: Path) -> List[str]:
    return [
        str(root / "bin/python"),
        "-I",
        "-S",
        "-B",
        "-c",
        "<explicit-site-stdin-launcher-sha256="
        + _sha256_bytes(EXPLICIT_SITE_STDIN_LAUNCHER_CODE.encode("utf-8"))
        + ">",
        "<content-sha256-bound-exact-no-site-sys-path-count-and-entries>",
        "<stdin-explicit-site-bootstrap-sha256="
        + _sha256_bytes(EXPLICIT_SITE_BOOTSTRAP_CODE.encode("utf-8"))
        + ">",
        "<content-bound-explicit-site-configuration>",
        "<pip-no-site-run-sha256="
        + _sha256_bytes(PIP_NO_SITE_RUN_CODE.encode("utf-8"))
        + ">",
        "--isolated",
        "--disable-pip-version-check",
        "install",
        "--no-index",
        "--no-deps",
        "--require-hashes",
        "--ignore-installed",
        "--prefix",
        str(root),
        "--find-links",
        str(root / WHEELHOUSE_RELATIVE_PATH),
        "--report",
        str(root / INSTALL_REPORT_RELATIVE_PATH),
        "-r",
        str(root / LOCK_RELATIVE_PATH),
    ]


def _validate_install_report_path_normalization(
    value: object,
    root: Path,
    reviewed_lock: Mapping[str, object],
) -> None:
    if not isinstance(value, dict):
        raise EnvironmentBuildError("pip report path normalization is malformed")
    _validate_content_sha(value, "pip report path normalization")
    _exact_keys(
        value,
        {"schema_version", "policy", "entries", "content_sha256"},
        "pip report path normalization",
    )
    if value.get("schema_version") != (
        "rachel-benchmark-pip-report-path-normalization/1.0"
    ) or value.get("policy") != (
        "execution-anchor-file-url-to-canonical-lexical-wheel-path"
    ):
        raise EnvironmentBuildError("pip report path normalization policy differs")
    packages = reviewed_lock.get("packages")
    entries = value.get("entries")
    if not isinstance(packages, list) or not isinstance(entries, list):
        raise EnvironmentBuildError("pip report path normalization entries differ")
    filenames = {str(item["filename"]) for item in packages if isinstance(item, dict)}
    observed = set()
    for item in entries:
        if not isinstance(item, dict):
            raise EnvironmentBuildError(
                "pip report path normalization entry is malformed"
            )
        _exact_keys(
            item,
            {"filename", "source_url", "canonical_path", "canonical_url"},
            "pip report path normalization entry",
        )
        filename = item.get("filename")
        if not isinstance(filename, str) or filename not in filenames:
            raise EnvironmentBuildError(
                "pip report path normalization filename differs"
            )
        canonical_path = root / WHEELHOUSE_RELATIVE_PATH / filename
        canonical_url = canonical_path.as_uri()
        source_url = item.get("source_url")
        if (
            item.get("canonical_path") != str(canonical_path)
            or item.get("canonical_url") != canonical_url
            or not isinstance(source_url, str)
        ):
            raise EnvironmentBuildError(
                "pip report canonical wheel path binding differs"
            )
        if source_url != canonical_url:
            parsed = urlsplit(source_url)
            if (
                parsed.scheme != "file"
                or parsed.netloc
                or parsed.query
                or parsed.fragment
                or re.fullmatch(
                    r"/(?:proc/(?:self|[0-9]+)|dev)/fd/[0-9]+/"
                    + re.escape(WHEELHOUSE_RELATIVE_PATH)
                    + "/"
                    + re.escape(filename),
                    unquote(parsed.path),
                )
                is None
            ):
                raise EnvironmentBuildError(
                    "pip report source wheel URL binding differs"
                )
        if filename in observed:
            raise EnvironmentBuildError(
                "pip report path normalization filename is duplicated"
            )
        observed.add(filename)
    if observed != filenames:
        raise EnvironmentBuildError("pip report path normalization set differs")


def _validate_install_report(
    path: Path,
    root: Path,
    reviewed_lock: Mapping[str, object],
    expected_file_sha256: Optional[str] = None,
) -> List[Dict[str, str]]:
    report, observed_file_sha256 = _load_json_with_sha(path, "pip install report")
    if (
        expected_file_sha256 is not None
        and observed_file_sha256 != expected_file_sha256
    ):
        raise EnvironmentBuildError("pip install report file SHA-256 differs")
    install = report.get("install")
    if not isinstance(install, list) or len(install) != len(ADDITIONS):
        raise EnvironmentBuildError("pip install report package set differs")
    wheelhouse = _require_directory(root / WHEELHOUSE_RELATIVE_PATH, "wheelhouse")
    package_values = reviewed_lock.get("packages")
    if not isinstance(package_values, list):
        raise EnvironmentBuildError("reviewed wheel authority is malformed")
    authorities = {
        (str(item["distribution"]), str(item["version"])): item
        for item in package_values
        if isinstance(item, dict)
    }
    wheel_paths = _list_regular_files(wheelhouse, "wheelhouse")
    wheels = {}
    for wheel_path in wheel_paths:
        matching = [
            item
            for item in authorities.values()
            if item.get("filename") == wheel_path.name
        ]
        if len(matching) != 1:
            raise EnvironmentBuildError("wheelhouse filename is not reviewed")
        inspected = _inspect_wheel(wheel_path, matching[0])
        wheels[(str(inspected["distribution"]), str(inspected["version"]))] = (
            inspected,
            matching[0],
            wheel_path,
        )
    observed: List[Dict[str, str]] = []
    for item in install:
        if not isinstance(item, dict):
            raise EnvironmentBuildError("pip install report entry is malformed")
        metadata = item.get("metadata")
        download_info = item.get("download_info")
        if not isinstance(metadata, dict) or not isinstance(download_info, dict):
            raise EnvironmentBuildError("pip install report lacks metadata")
        name_value = metadata.get("name")
        version_value = metadata.get("version")
        if not isinstance(name_value, str) or not isinstance(version_value, str):
            raise EnvironmentBuildError("pip install report identity is malformed")
        name = _canonical_distribution_name(name_value)
        wheel_binding = wheels.get((name, version_value))
        if wheel_binding is None:
            raise EnvironmentBuildError("pip install report names an unexpected wheel")
        wheel, authority, wheel_path = wheel_binding
        if set(download_info) != {"url", "archive_info"}:
            raise EnvironmentBuildError("pip install report download fields differ")
        url = download_info.get("url")
        if not isinstance(url, str):
            raise EnvironmentBuildError("pip install report wheel URL is malformed")
        parsed = urlsplit(url)
        try:
            url_path = Path(unquote(parsed.path, errors="strict"))
        except UnicodeError as error:
            raise EnvironmentBuildError(
                "pip install report wheel URL is malformed"
            ) from error
        if (
            parsed.scheme != "file"
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or url != wheel_path.as_uri()
            or url_path != wheel_path
            or url_path.name != authority.get("filename")
        ):
            raise EnvironmentBuildError("pip install report wheel path differs")
        archive = download_info.get("archive_info")
        if not isinstance(archive, dict):
            raise EnvironmentBuildError("pip install report lacks archive hash")
        hashes = archive.get("hashes")
        sha = hashes.get("sha256") if isinstance(hashes, dict) else None
        if sha is None:
            single = archive.get("hash")
            if isinstance(single, str) and single.startswith("sha256="):
                sha = single.split("=", 1)[1]
        if sha != wheel["sha256"]:
            raise EnvironmentBuildError("pip install report wheel SHA-256 differs")
        observed.append(
            {
                "distribution": name,
                "version": version_value,
                "wheel_sha256": str(sha),
                "wheel_filename": str(authority["filename"]),
                "wheel_url": url,
            }
        )
    observed.sort(key=lambda value: value["distribution"])
    if {(item["distribution"], item["version"]) for item in observed} != {
        (name, version) for name, version, _module in ADDITIONS
    }:
        raise EnvironmentBuildError("pip install report exact pins differ")
    return observed


def _pip_freeze(
    python: Path,
    *,
    safe_probe: Mapping[str, object],
    audits: Sequence[Mapping[str, object]],
    execution_anchor: Optional[_ExecutionAnchor] = None,
) -> Dict[str, object]:
    _validate_pip_runtime_audits(audits, "pip freeze")
    completed = _run_explicit_site_payload(
        python,
        PIP_NO_SITE_RUN_CODE,
        (
            "--isolated",
            "--disable-pip-version-check",
            "freeze",
            "--all",
        ),
        safe_probe=safe_probe,
        audits=audits,
        timeout=600,
        label="pip freeze",
        primary_anchor=execution_anchor,
    )
    lines = [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "lines": lines,
        "canonical_sha256": _sha256_bytes(_canonical_bytes(lines)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="create one fresh environment")
    build.add_argument("--base-python", required=True)
    build.add_argument("--environment-root", required=True)
    build.add_argument("--official-shreddingnet-env", required=True)
    build.add_argument("--wheel-lock", required=True)
    build.add_argument("--expected-wheel-lock-sha256", required=True)
    verify = subparsers.add_parser("verify", help="read-only full receipt verification")
    verify.add_argument("--environment-root", required=True)
    verify.add_argument("--receipt")
    verify.add_argument("--expected-receipt-sha256")
    return parser


def _require_safe_controller_runtime() -> None:
    if (
        not bool(sys.flags.isolated)
        or not bool(sys.flags.no_site)
        or not bool(sys.dont_write_bytecode)
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise EnvironmentBuildError(
            "builder/verifier controller must start with Python -I -S -B"
        )


def _validate_inventory_identity(
    inventory: Mapping[str, object], root: Path, python: Path
) -> None:
    _validate_content_sha(inventory, "import inventory")
    if (
        inventory.get("environment_root") != str(root)
        or inventory.get("python") != os.path.realpath(str(python))
        or inventory.get("python_sha256")
        != _launcher_sha256(python, "benchmark Python")
    ):
        raise EnvironmentBuildError("import inventory environment identity differs")


def _pth_never_executed_bindings(
    audits: Sequence[Mapping[str, object]], label: str
) -> List[Dict[str, object]]:
    result = []
    seen = set()
    for audit in audits:
        _validate_content_sha(audit, label + " active-site audit")
        if (
            audit.get("runtime_no_site") is not True
            or audit.get("pth_execution_policy") != "present_but_never_executed"
            or audit.get("activated_paths") != []
        ):
            raise EnvironmentBuildError(label + " PTH execution policy differs")
        artifacts = audit.get("startup_artifacts")
        if not isinstance(artifacts, list):
            raise EnvironmentBuildError(label + " startup artifacts are malformed")
        for artifact in artifacts:
            if not isinstance(artifact, dict) or artifact.get("kind") != "pth":
                raise EnvironmentBuildError(label + " startup artifact is malformed")
            path = artifact.get("path")
            directives = artifact.get("directives")
            declared_paths = artifact.get("declared_paths")
            if (
                not isinstance(path, str)
                or path in seen
                or not _is_sha256(artifact.get("sha256"))
                or artifact.get("execution_policy") != "present_but_never_executed"
                or artifact.get("activated_paths") != []
                or not isinstance(directives, list)
                or not isinstance(declared_paths, list)
            ):
                raise EnvironmentBuildError(label + " PTH binding is malformed")
            executable_line_hashes = []
            for directive in directives:
                if not isinstance(directive, dict):
                    raise EnvironmentBuildError(label + " PTH directive is malformed")
                if directive.get("kind") == "executable_present_but_never_executed":
                    line_sha256 = directive.get("line_sha256")
                    if not _is_sha256(line_sha256):
                        raise EnvironmentBuildError(
                            label + " executable PTH directive lacks SHA-256"
                        )
                    executable_line_hashes.append(str(line_sha256))
                elif directive.get("kind") != "path":
                    raise EnvironmentBuildError(label + " PTH directive kind differs")
            result.append(
                {
                    "path": path,
                    "sha256": artifact["sha256"],
                    "execution_policy": "present_but_never_executed",
                    "directive_count": len(directives),
                    "directives_content_sha256": _sha256_bytes(
                        _canonical_bytes(directives)
                    ),
                    "executable_line_sha256s": executable_line_hashes,
                    "declared_paths": list(declared_paths),
                    "audit_content_sha256": audit["content_sha256"],
                }
            )
            seen.add(path)
    return sorted(result, key=lambda item: str(item["path"]))


def _tracked_import_origin_bindings(
    probe: Mapping[str, object], label: str
) -> List[Dict[str, str]]:
    tracked = probe.get("tracked")
    if not isinstance(tracked, dict) or not tracked:
        raise EnvironmentBuildError(label + " tracked import origins are malformed")
    result = []
    for distribution in sorted(tracked):
        item = tracked[distribution]
        if not isinstance(distribution, str) or not isinstance(item, dict):
            raise EnvironmentBuildError(label + " tracked import origin is malformed")
        fields = {
            "distribution": distribution,
            "distribution_version": item.get("distribution_version"),
            "distribution_location": item.get("distribution_location"),
            "module": item.get("module"),
            "import_origin": item.get("import_origin"),
        }
        if any(not isinstance(value, str) or not value for value in fields.values()):
            raise EnvironmentBuildError(label + " tracked import origin is incomplete")
        result.append(fields)  # type: ignore[arg-type]
    return result


def _runtime_context_evidence(
    *,
    safe_probe: Mapping[str, object],
    audits: Sequence[Mapping[str, object]],
    probe: Mapping[str, object],
    label: str,
) -> Dict[str, object]:
    safe_identity = {
        "executable": safe_probe.get("executable"),
        "prefix": safe_probe.get("prefix"),
        "base_prefix": safe_probe.get("base_prefix"),
    }
    no_site_sys_path = safe_probe.get("sys_path")
    logical_identity = {
        "prefix": probe.get("prefix"),
        "base_prefix": probe.get("base_prefix"),
    }
    if (
        any(not isinstance(value, str) or not value for value in safe_identity.values())
        or any(
            not isinstance(value, str) or not value
            for value in logical_identity.values()
        )
        or logical_identity["base_prefix"] != safe_identity["base_prefix"]
        or not isinstance(no_site_sys_path, list)
        or any(not isinstance(path, str) for path in no_site_sys_path)
    ):
        raise EnvironmentBuildError(label + " runtime identity evidence is malformed")
    bindings = _canonical_site_bindings(audits, label)
    if bindings[0]["prefix"] != logical_identity["prefix"]:
        raise EnvironmentBuildError(label + " primary injected site prefix differs")
    dynamic_admission = _validate_torchvision_dynamic_admission_evidence(
        probe.get("torchvision_dynamic_module_admission"),
        label + " torchvision dynamic module admission",
    )
    return _content_bound(
        {
            "runtime_no_site": True,
            "safe_startup_identity": safe_identity,
            "logical_runtime_identity": logical_identity,
            "no_site_sys_path": list(no_site_sys_path),
            "injected_site_directories": bindings,
            "import_origins": _tracked_import_origin_bindings(probe, label),
            "torchvision_dynamic_module_admission": dynamic_admission,
            "pth_files_present_but_never_executed": (
                _pth_never_executed_bindings(audits, label)
            ),
        }
    )


def _runtime_no_site_evidence(
    *,
    base_safe_probe: Mapping[str, object],
    base_audits: Sequence[Mapping[str, object]],
    base_probe: Mapping[str, object],
    resolved_base_safe_probe: Mapping[str, object],
    resolved_base_audits: Sequence[Mapping[str, object]],
    resolved_base_probe: Mapping[str, object],
    target_safe_probe: Mapping[str, object],
    target_audits: Sequence[Mapping[str, object]],
    target_probe: Mapping[str, object],
) -> Dict[str, object]:
    return _content_bound(
        {
            "schema_version": RUNTIME_NO_SITE_SCHEMA,
            "runtime_no_site": True,
            "interpreter_flags": ["-I", "-S", "-B"],
            "bootstrap": {
                "configuration_schema": EXPLICIT_SITE_BOOTSTRAP_SCHEMA,
                "launcher_source_sha256": _sha256_bytes(
                    EXPLICIT_SITE_STDIN_LAUNCHER_CODE.encode("utf-8")
                ),
                "source_sha256": _sha256_bytes(
                    EXPLICIT_SITE_BOOTSTRAP_CODE.encode("utf-8")
                ),
                "torchvision_dynamic_admission_source_sha256": _sha256_bytes(
                    TORCHVISION_DYNAMIC_ADMISSION_BOOTSTRAP_CODE.encode("utf-8")
                ),
                "torchvision_dynamic_admission_configuration": (
                    _torchvision_dynamic_admission_configuration(True)
                ),
                "torchvision_dynamic_admission_runtime_only": True,
                "pip_dynamic_preload_disabled": True,
                "source_transport": "sha256_bound_stdin",
                "launcher_sys_path_policy": (
                    "content_sha256_bound_exact_replacement_before_filesystem_import"
                ),
                "pip_runner_source_sha256": _sha256_bytes(
                    PIP_NO_SITE_RUN_CODE.encode("utf-8")
                ),
                "payload_source_sha256": {
                    "queue_live_probe": _sha256_bytes(
                        QUEUE_LIVE_PROBE_CODE.encode("utf-8")
                    ),
                    "detailed_probe": _sha256_bytes(
                        DETAILED_PROBE_CODE.encode("utf-8")
                    ),
                    "functional_cpu_probe": _sha256_bytes(
                        FUNCTIONAL_CPU_PROBE_CODE.encode("utf-8")
                    ),
                    "import_inventory": _sha256_bytes(
                        INVENTORY_PROBE_CODE.encode("utf-8")
                    ),
                },
                "configuration_content_sha256_bound": True,
                "preimport_sys_path_exact": True,
                "payload_import_sys_path_audited": True,
                "site_directories_fd_anchored": True,
                "site_main_and_pth_processing_blocked": True,
                "sitecustomize_and_usercustomize_never_imported": True,
            },
            "base": _runtime_context_evidence(
                safe_probe=base_safe_probe,
                audits=base_audits,
                probe=base_probe,
                label="base no-site runtime",
            ),
            "resolved_base": _runtime_context_evidence(
                safe_probe=resolved_base_safe_probe,
                audits=resolved_base_audits,
                probe=resolved_base_probe,
                label="resolved base no-site runtime",
            ),
            "target": _runtime_context_evidence(
                safe_probe=target_safe_probe,
                audits=target_audits,
                probe=target_probe,
                label="target no-site runtime",
            ),
        }
    )


def _validate_runtime_no_site_evidence(value: object) -> Dict[str, object]:
    evidence = _require_mapping(value, "runtime no-site evidence")
    _validate_content_sha(evidence, "runtime no-site evidence")
    _exact_keys(
        evidence,
        {
            "schema_version",
            "runtime_no_site",
            "interpreter_flags",
            "bootstrap",
            "base",
            "resolved_base",
            "target",
            "content_sha256",
        },
        "runtime no-site evidence",
    )
    if (
        evidence.get("schema_version") != RUNTIME_NO_SITE_SCHEMA
        or evidence.get("runtime_no_site") is not True
        or evidence.get("interpreter_flags") != ["-I", "-S", "-B"]
    ):
        raise EnvironmentBuildError("runtime no-site evidence contract differs")
    expected_bootstrap = {
        "configuration_schema": EXPLICIT_SITE_BOOTSTRAP_SCHEMA,
        "launcher_source_sha256": _sha256_bytes(
            EXPLICIT_SITE_STDIN_LAUNCHER_CODE.encode("utf-8")
        ),
        "source_sha256": _sha256_bytes(EXPLICIT_SITE_BOOTSTRAP_CODE.encode("utf-8")),
        "torchvision_dynamic_admission_source_sha256": _sha256_bytes(
            TORCHVISION_DYNAMIC_ADMISSION_BOOTSTRAP_CODE.encode("utf-8")
        ),
        "torchvision_dynamic_admission_configuration": (
            _torchvision_dynamic_admission_configuration(True)
        ),
        "torchvision_dynamic_admission_runtime_only": True,
        "pip_dynamic_preload_disabled": True,
        "source_transport": "sha256_bound_stdin",
        "launcher_sys_path_policy": (
            "content_sha256_bound_exact_replacement_before_filesystem_import"
        ),
        "pip_runner_source_sha256": _sha256_bytes(PIP_NO_SITE_RUN_CODE.encode("utf-8")),
        "payload_source_sha256": {
            "queue_live_probe": _sha256_bytes(QUEUE_LIVE_PROBE_CODE.encode("utf-8")),
            "detailed_probe": _sha256_bytes(DETAILED_PROBE_CODE.encode("utf-8")),
            "functional_cpu_probe": _sha256_bytes(
                FUNCTIONAL_CPU_PROBE_CODE.encode("utf-8")
            ),
            "import_inventory": _sha256_bytes(INVENTORY_PROBE_CODE.encode("utf-8")),
        },
        "configuration_content_sha256_bound": True,
        "preimport_sys_path_exact": True,
        "payload_import_sys_path_audited": True,
        "site_directories_fd_anchored": True,
        "site_main_and_pth_processing_blocked": True,
        "sitecustomize_and_usercustomize_never_imported": True,
    }
    if evidence.get("bootstrap") != expected_bootstrap:
        raise EnvironmentBuildError("runtime no-site bootstrap identity differs")
    for name in ("base", "resolved_base", "target"):
        context = _require_mapping(evidence.get(name), name + " runtime context")
        _validate_content_sha(context, name + " runtime context")
        _exact_keys(
            context,
            {
                "runtime_no_site",
                "safe_startup_identity",
                "logical_runtime_identity",
                "no_site_sys_path",
                "injected_site_directories",
                "import_origins",
                "torchvision_dynamic_module_admission",
                "pth_files_present_but_never_executed",
                "content_sha256",
            },
            name + " runtime context",
        )
        if context.get("runtime_no_site") is not True:
            raise EnvironmentBuildError(name + " runtime no-site flag differs")
        _validate_torchvision_dynamic_admission_evidence(
            context.get("torchvision_dynamic_module_admission"),
            name + " torchvision dynamic module admission",
        )
        safe_identity = _require_mapping(
            context.get("safe_startup_identity"), name + " safe startup identity"
        )
        logical_identity = _require_mapping(
            context.get("logical_runtime_identity"),
            name + " logical runtime identity",
        )
        _exact_keys(
            safe_identity,
            {"executable", "prefix", "base_prefix"},
            name + " safe startup identity",
        )
        _exact_keys(
            logical_identity,
            {"prefix", "base_prefix"},
            name + " logical runtime identity",
        )
        if (
            any(
                not isinstance(value, str) or not Path(value).is_absolute()
                for value in safe_identity.values()
            )
            or any(
                not isinstance(value, str) or not Path(value).is_absolute()
                for value in logical_identity.values()
            )
            or logical_identity["base_prefix"] != safe_identity["base_prefix"]
        ):
            raise EnvironmentBuildError(name + " runtime identity differs")
        no_site_sys_path = context.get("no_site_sys_path")
        if (
            not isinstance(no_site_sys_path, list)
            or not no_site_sys_path
            or any(
                not isinstance(path, str) or not Path(path).is_absolute()
                for path in no_site_sys_path
            )
            or len(no_site_sys_path) != len(set(no_site_sys_path))
        ):
            raise EnvironmentBuildError(name + " no-site sys.path differs")
        injected = context.get("injected_site_directories")
        if not isinstance(injected, list) or not injected:
            raise EnvironmentBuildError(name + " injected sites differ")
        injected_paths = []
        for item in injected:
            binding = _require_mapping(item, name + " injected site binding")
            _exact_keys(
                binding,
                {
                    "canonical_path",
                    "prefix",
                    "relative_path",
                    "device",
                    "inode",
                    "role",
                    "audit_content_sha256",
                },
                name + " injected site binding",
            )
            canonical_path = binding.get("canonical_path")
            prefix = binding.get("prefix")
            relative_path = binding.get("relative_path")
            if (
                not isinstance(canonical_path, str)
                or not isinstance(prefix, str)
                or not isinstance(relative_path, str)
                or not Path(canonical_path).is_absolute()
                or not Path(prefix).is_absolute()
                or Path(prefix) / relative_path != Path(canonical_path)
                or not isinstance(binding.get("device"), int)
                or not isinstance(binding.get("inode"), int)
                or re.fullmatch(r"audit_[0-9]+", str(binding.get("role"))) is None
                or not _is_sha256(binding.get("audit_content_sha256"))
            ):
                raise EnvironmentBuildError(name + " injected site binding differs")
            injected_paths.append(canonical_path)
        if (
            len(injected_paths) != len(set(injected_paths))
            or injected[0].get("prefix") != logical_identity["prefix"]
        ):
            raise EnvironmentBuildError(name + " injected site order differs")
        origins = context.get("import_origins")
        if not isinstance(origins, list) or not origins:
            raise EnvironmentBuildError(name + " import origins differ")
        expected_distributions = (
            {distribution for distribution, _module in TARGET_TRACKED_DISTRIBUTIONS}
            if name == "target"
            else {distribution for distribution, _module in BASE_TRACKED_DISTRIBUTIONS}
        )
        observed_distributions = []
        for item in origins:
            origin = _require_mapping(item, name + " import origin binding")
            _exact_keys(
                origin,
                {
                    "distribution",
                    "distribution_version",
                    "distribution_location",
                    "module",
                    "import_origin",
                },
                name + " import origin binding",
            )
            if any(
                not isinstance(value, str) or not value for value in origin.values()
            ) or any(
                not Path(str(origin[field])).is_absolute()
                for field in ("distribution_location", "import_origin")
            ):
                raise EnvironmentBuildError(name + " import origin binding differs")
            observed_distributions.append(str(origin["distribution"]))
        if (
            observed_distributions != sorted(observed_distributions)
            or set(observed_distributions) != expected_distributions
        ):
            raise EnvironmentBuildError(name + " import origin set differs")
        pth_files = context.get("pth_files_present_but_never_executed")
        if not isinstance(pth_files, list):
            raise EnvironmentBuildError(name + " runtime PTH evidence differs")
        pth_paths = []
        for item in pth_files:
            pth = _require_mapping(item, name + " runtime PTH binding")
            _exact_keys(
                pth,
                {
                    "path",
                    "sha256",
                    "execution_policy",
                    "directive_count",
                    "directives_content_sha256",
                    "executable_line_sha256s",
                    "declared_paths",
                    "audit_content_sha256",
                },
                name + " runtime PTH binding",
            )
            executable_hashes = pth.get("executable_line_sha256s")
            declared_paths = pth.get("declared_paths")
            if (
                not isinstance(pth.get("path"), str)
                or not Path(str(pth["path"])).is_absolute()
                or not _is_sha256(pth.get("sha256"))
                or pth.get("execution_policy") != "present_but_never_executed"
                or not isinstance(pth.get("directive_count"), int)
                or int(pth["directive_count"]) < 0
                or not _is_sha256(pth.get("directives_content_sha256"))
                or not isinstance(executable_hashes, list)
                or any(not _is_sha256(item) for item in executable_hashes)
                or not isinstance(declared_paths, list)
                or any(
                    not isinstance(path, str) or not Path(path).is_absolute()
                    for path in declared_paths
                )
                or not _is_sha256(pth.get("audit_content_sha256"))
            ):
                raise EnvironmentBuildError(name + " runtime PTH binding differs")
            pth_paths.append(str(pth["path"]))
        if pth_paths != sorted(pth_paths) or len(pth_paths) != len(set(pth_paths)):
            raise EnvironmentBuildError(name + " runtime PTH order differs")
    return evidence


def _active_site_evidence(
    *,
    base_safe_probe: Mapping[str, object],
    base_audit: Mapping[str, object],
    target_safe_probe: Mapping[str, object],
    target_audit: Mapping[str, object],
) -> Dict[str, object]:
    return _content_bound(
        {
            "schema_version": "rachel-benchmark-active-site-evidence/1.0",
            "policy": {
                "controller_started_with_no_site": True,
                "pth_files_content_bound": True,
                "pth_files_present_but_never_executed": True,
                "executable_pth_directives_present_but_never_executed": True,
                "egg_and_egg_link_rejected": True,
                "sitecustomize_and_usercustomize_rejected": True,
                "payload_subprocesses_started_with_no_site": True,
                "site_directories_explicitly_injected_by_held_descriptor": True,
                "unknown_runtime_sys_path_rejected": True,
            },
            "base": {
                "safe_probe": dict(base_safe_probe),
                "audit": dict(base_audit),
            },
            "target": {
                "safe_probe": dict(target_safe_probe),
                "audit": dict(target_audit),
            },
        }
    )


def _load_active_site_evidence(
    root: Path, binding: Mapping[str, object]
) -> Dict[str, object]:
    path = _require_read_only_file(
        Path(str(binding.get("path"))), "stored active-site evidence"
    )
    if path != root / ACTIVE_SITE_RELATIVE_PATH:
        raise EnvironmentBuildError("stored active-site evidence path differs")
    value, observed_file_sha256 = _load_json_with_sha(
        path, "stored active-site evidence"
    )
    if observed_file_sha256 != binding.get("file_sha256"):
        raise EnvironmentBuildError("stored active-site evidence file drifted")
    _validate_content_sha(value, "stored active-site evidence")
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise EnvironmentBuildError("stored active-site evidence binding differs")
    _exact_keys(
        value,
        {"schema_version", "policy", "base", "target", "content_sha256"},
        "stored active-site evidence",
    )
    if value.get(
        "schema_version"
    ) != "rachel-benchmark-active-site-evidence/1.0" or value.get("policy") != {
        "controller_started_with_no_site": True,
        "pth_files_content_bound": True,
        "pth_files_present_but_never_executed": True,
        "executable_pth_directives_present_but_never_executed": True,
        "egg_and_egg_link_rejected": True,
        "sitecustomize_and_usercustomize_rejected": True,
        "payload_subprocesses_started_with_no_site": True,
        "site_directories_explicitly_injected_by_held_descriptor": True,
        "unknown_runtime_sys_path_rejected": True,
    }:
        raise EnvironmentBuildError("stored active-site policy differs")
    return value


def build_environment(
    *,
    base_python: Path,
    environment_root: Path,
    official_env_yaml: Path,
    wheel_lock_path: Path,
    expected_wheel_lock_sha256: str,
) -> Dict[str, object]:
    """Build a fresh venv and atomically publish its completion receipt."""

    _require_safe_controller_runtime()
    base_python, base_launcher = _require_python_launcher(base_python, "base Python")
    base_prefix = _lexical_absolute(
        str(base_python.parent.parent), "base Python prefix"
    )
    base_safe_probe = _anchored_safe_runtime_probe(base_python, base_prefix)
    _validate_safe_runtime_probe(base_safe_probe, base_python, "base Python")
    base_active_site = _audit_base_site_surface(base_prefix, base_safe_probe)
    reviewed_lock = _validate_wheel_lock(
        wheel_lock_path, expected_wheel_lock_sha256, base_safe_probe
    )
    official = _validate_official_environment(official_env_yaml)
    base_python_sha = str(base_launcher["resolved_executable_sha256"])
    base_probe = _base_probe(
        base_python,
        safe_probe=base_safe_probe,
        audits=_base_prefix_audits(base_active_site),
    )
    _validate_base_probe(
        base_probe,
        base_python,
        safe_probe=base_safe_probe,
        active_site_audit=base_active_site,
    )
    resolved_base_python = Path(str(base_launcher["resolved_executable"]))
    resolved_base_active_site = _resolved_base_audit(base_active_site)
    resolved_base_probe = _base_probe(
        resolved_base_python,
        safe_probe=base_safe_probe,
        audits=(resolved_base_active_site,),
    )
    _validate_base_probe(
        resolved_base_probe,
        resolved_base_python,
        safe_probe=base_safe_probe,
        active_site_audit=resolved_base_active_site,
    )
    if _require_python_launcher(base_python, "base Python")[1] != base_launcher:
        raise EnvironmentBuildError("base Python launcher changed during preflight")

    environment_root = _lexical_absolute(str(environment_root), "environment root")
    _assert_no_symlink_components(environment_root, "environment root")
    safe_base_prefix = _lexical_absolute(
        str(base_safe_probe["prefix"]), "resolved base Python prefix"
    )
    for inherited_prefix in {base_prefix, safe_base_prefix}:
        if _is_within(environment_root, inherited_prefix) or _is_within(
            inherited_prefix, environment_root
        ):
            raise EnvironmentBuildError(
                "environment root overlaps an inherited base environment"
            )
    root = _mkdir_new_anchored(environment_root, "environment root")
    root_identity = _directory_identity(root, "environment root")
    execution_anchor = _ExecutionAnchor(root, "environment root")
    try:
        return _finish_environment_build(
            base_python=base_python,
            base_launcher=base_launcher,
            base_safe_probe=base_safe_probe,
            base_prefix=base_prefix,
            base_active_site=base_active_site,
            reviewed_lock=reviewed_lock,
            official=official,
            base_python_sha=base_python_sha,
            base_probe=base_probe,
            resolved_base_probe=resolved_base_probe,
            root=root,
            root_identity=root_identity,
            official_env_yaml=official_env_yaml,
            execution_anchor=execution_anchor,
        )
    finally:
        execution_anchor.close()


def _finish_environment_build(
    *,
    base_python: Path,
    base_launcher: Mapping[str, object],
    base_safe_probe: Mapping[str, object],
    base_prefix: Path,
    base_active_site: Mapping[str, object],
    reviewed_lock: Mapping[str, object],
    official: Mapping[str, object],
    base_python_sha: str,
    base_probe: Mapping[str, object],
    resolved_base_probe: Mapping[str, object],
    root: Path,
    root_identity: Mapping[str, object],
    official_env_yaml: Path,
    execution_anchor: _ExecutionAnchor,
) -> Dict[str, object]:
    # ``venv`` creates directories by resolving ``env_dir`` relative to its
    # parent.  A procfs/devfs descriptor path names the already-open root, but
    # is not a usable creation path for the stdlib venv builder.  Hold and
    # revalidate both the fresh root and its parent around the child while
    # passing the verified lexical target only for this creation step.
    target_parent_anchor = _ExecutionAnchor(
        root.parent, "environment root parent"
    )
    base_execution_anchor = _ExecutionAnchor(base_prefix, "base Python prefix")
    resolved_base_executable = Path(str(base_launcher["resolved_executable"]))
    resolved_base_prefix = resolved_base_executable.parent.parent
    resolved_base_anchor = (
        base_execution_anchor
        if resolved_base_prefix == base_prefix
        else _ExecutionAnchor(resolved_base_prefix, "resolved base Python prefix")
    )
    construction_descriptor_replacements = {
        str(
            base_execution_anchor.child_path(
                base_prefix, "formal base descriptor prefix"
            )
        ): str(base_prefix),
        str(
            resolved_base_anchor.child_path(
                resolved_base_prefix, "resolved base descriptor prefix"
            )
        ): str(resolved_base_prefix),
    }
    try:
        anchored_base_python = resolved_base_anchor.child_path(
            resolved_base_executable, "venv creation base Python"
        )
        _run_checked(
            (
                str(anchored_base_python),
                "-I",
                "-S",
                "-B",
                "-m",
                "venv",
                "--without-pip",
                "--copies",
                "--system-site-packages",
                str(root),
            ),
            timeout=600,
            label="fresh benchmark venv creation",
            execution_anchor=execution_anchor,
            additional_execution_anchors=(
                target_parent_anchor,
                base_execution_anchor,
                resolved_base_anchor,
            ),
        )
    finally:
        if resolved_base_anchor is not base_execution_anchor:
            resolved_base_anchor.close()
        base_execution_anchor.close()
        target_parent_anchor.close()
    _revalidate_directory_identity(root, root_identity, "environment root")
    _normalize_target_pyvenv_configuration(
        root,
        execution_anchor,
        descriptor_path_replacements=construction_descriptor_replacements,
    )
    _normalize_and_audit_environment_scripts(
        root,
        execution_anchor,
        rewrite_descriptor_paths=True,
        descriptor_path_replacements=construction_descriptor_replacements,
    )
    python, target_launcher = _require_python_launcher(
        root / "bin/python", "benchmark Python"
    )
    if python != root / "bin/python":
        raise EnvironmentBuildError("benchmark Python launcher path differs")
    if target_launcher.get("launcher_kind") != "regular_file":
        raise EnvironmentBuildError("benchmark Python must be a copied regular file")
    target_safe_probe = _safe_runtime_probe(python, execution_anchor)
    _validate_safe_runtime_probe(target_safe_probe, python, "benchmark Python")
    target_active_site_initial = _audit_active_site(
        root,
        safe_probe=target_safe_probe,
        require_pyvenv=True,
        require_system_site_packages=True,
    )

    target_runtime_audits = (
        target_active_site_initial,
        _resolved_base_audit(base_active_site),
    )
    wheels, download_evidence = _download_and_lock_wheels(
        python,
        root,
        reviewed_lock,
        execution_anchor,
        target_safe_probe,
        target_runtime_audits,
    )
    _revalidate_directory_identity(root, root_identity, "environment root")
    install_evidence = _install_locked_wheels(
        python,
        root,
        reviewed_lock,
        execution_anchor,
        target_safe_probe,
        target_runtime_audits,
    )
    _revalidate_directory_identity(root, root_identity, "environment root")
    script_inventory = _normalize_and_audit_environment_scripts(
        root, execution_anchor, rewrite_descriptor_paths=True
    )

    target_safe_probe_final = _safe_runtime_probe(python, execution_anchor)
    _validate_safe_runtime_probe(target_safe_probe_final, python, "benchmark Python")
    target_active_site = _audit_active_site(
        root,
        safe_probe=target_safe_probe_final,
        require_pyvenv=True,
        require_system_site_packages=True,
    )
    if (
        target_safe_probe_final != target_safe_probe
        or target_active_site != target_active_site_initial
    ):
        raise EnvironmentBuildError("target startup surface changed during install")
    confirmed_base_safe = _anchored_safe_runtime_probe(base_python, base_prefix)
    _validate_safe_runtime_probe(confirmed_base_safe, base_python, "base Python")
    confirmed_base_active = _audit_base_site_surface(base_prefix, confirmed_base_safe)
    if (
        confirmed_base_safe != base_safe_probe
        or confirmed_base_active != base_active_site
    ):
        raise EnvironmentBuildError("base startup surface changed during build")

    confirmed_base_probe = _base_probe(
        base_python,
        safe_probe=base_safe_probe,
        audits=_base_prefix_audits(base_active_site),
    )
    _validate_base_probe(
        confirmed_base_probe,
        base_python,
        safe_probe=base_safe_probe,
        active_site_audit=base_active_site,
    )
    if confirmed_base_probe != base_probe:
        raise EnvironmentBuildError("base dependency probe changed during build")
    confirmed_resolved_base_probe = _base_probe(
        resolved_base_executable,
        safe_probe=base_safe_probe,
        audits=(_resolved_base_audit(base_active_site),),
    )
    _validate_base_probe(
        confirmed_resolved_base_probe,
        resolved_base_executable,
        safe_probe=base_safe_probe,
        active_site_audit=_resolved_base_audit(base_active_site),
    )
    if confirmed_resolved_base_probe != resolved_base_probe:
        raise EnvironmentBuildError(
            "resolved base dependency probe changed during build"
        )
    target_runtime_audits = (
        target_active_site,
        _resolved_base_audit(base_active_site),
    )
    live_probe = _queue_live_probe(
        python,
        safe_probe=target_safe_probe,
        audits=target_runtime_audits,
        execution_anchor=execution_anchor,
    )
    detailed_probe = _detailed_probe(
        python,
        safe_probe=target_safe_probe,
        audits=target_runtime_audits,
        execution_anchor=execution_anchor,
    )
    _validate_target_probes(
        root=root,
        python=python,
        base_probe=base_probe,
        inherited_base_probe=resolved_base_probe,
        safe_probes=(target_safe_probe, base_safe_probe),
        active_site_audits=(
            target_active_site,
            _resolved_base_audit(base_active_site),
        ),
        live=live_probe,
        detailed=detailed_probe,
    )
    functional_probe = _functional_cpu_probe(
        python,
        safe_probe=target_safe_probe,
        audits=target_runtime_audits,
        execution_anchor=execution_anchor,
    )
    inventory = _import_inventory(
        python,
        root,
        safe_probe=target_safe_probe,
        audits=target_runtime_audits,
        execution_anchor=execution_anchor,
    )
    _validate_inventory_identity(inventory, root, python)
    freeze = _pip_freeze(
        python,
        safe_probe=target_safe_probe,
        audits=target_runtime_audits,
        execution_anchor=execution_anchor,
    )
    confirmed_inventory = _import_inventory(
        python,
        root,
        safe_probe=target_safe_probe,
        audits=target_runtime_audits,
        execution_anchor=execution_anchor,
    )
    _validate_inventory_identity(confirmed_inventory, root, python)
    if confirmed_inventory != inventory:
        raise EnvironmentBuildError(
            "imported package content changed during environment build"
        )
    inventory_path = root / INVENTORY_RELATIVE_PATH
    _write_json_new_immutable(
        inventory_path, inventory, "import inventory", bind_content=False
    )
    active_site_value = _active_site_evidence(
        base_safe_probe=base_safe_probe,
        base_audit=base_active_site,
        target_safe_probe=target_safe_probe,
        target_audit=target_active_site,
    )
    active_site_path = root / ACTIVE_SITE_RELATIVE_PATH
    _write_json_new_immutable(
        active_site_path,
        active_site_value,
        "active-site evidence",
        bind_content=False,
    )
    runtime_no_site_value = _runtime_no_site_evidence(
        base_safe_probe=base_safe_probe,
        base_audits=_base_prefix_audits(base_active_site),
        base_probe=base_probe,
        resolved_base_safe_probe=base_safe_probe,
        resolved_base_audits=(_resolved_base_audit(base_active_site),),
        resolved_base_probe=resolved_base_probe,
        target_safe_probe=target_safe_probe,
        target_audits=target_runtime_audits,
        target_probe=detailed_probe,
    )

    builder_source = _require_regular_file(Path(__file__), "environment builder source")
    build_evidence = {
        "download": download_evidence,
        "install": install_evidence,
        "wheels": wheels,
        "reviewed_wheel_lock": reviewed_lock,
        "target_launcher_identity": target_launcher,
        "bin_scripts": script_inventory,
        "inventory": {
            "path": str(inventory_path),
            "file_sha256": _sha256_file(inventory_path),
            "content_sha256": inventory["content_sha256"],
        },
        "active_site": {
            "path": str(active_site_path),
            "file_sha256": _sha256_file(active_site_path),
            "content_sha256": active_site_value["content_sha256"],
        },
        "runtime_no_site": runtime_no_site_value,
    }
    _chmod_nofollow(
        root / ".environment_build_evidence",
        0o555,
        directory=True,
        label="environment build evidence",
    )
    _verify_build_evidence(
        root, build_evidence, base_safe_probe, execution_anchor=execution_anchor
    )
    if (
        _load_active_site_evidence(root, build_evidence["active_site"])
        != active_site_value
    ):
        raise EnvironmentBuildError("active-site evidence changed before receipt")
    final_inventory = _import_inventory(
        python,
        root,
        safe_probe=target_safe_probe,
        audits=target_runtime_audits,
        execution_anchor=execution_anchor,
    )
    _validate_inventory_identity(final_inventory, root, python)
    if final_inventory != inventory:
        raise EnvironmentBuildError("final imported package inventory drifted")
    if (
        _normalize_and_audit_environment_scripts(
            root, execution_anchor, rewrite_descriptor_paths=False
        )
        != script_inventory
    ):
        raise EnvironmentBuildError("final environment bin script inventory drifted")
    final_base_safe = _anchored_safe_runtime_probe(base_python, base_prefix)
    final_target_safe = _safe_runtime_probe(python, execution_anchor)
    _validate_safe_runtime_probe(final_base_safe, base_python, "base Python")
    _validate_safe_runtime_probe(final_target_safe, python, "benchmark Python")
    if (
        final_base_safe != base_safe_probe
        or final_target_safe != target_safe_probe
        or _audit_base_site_surface(base_prefix, final_base_safe) != base_active_site
        or _audit_active_site(
            root,
            safe_probe=final_target_safe,
            require_pyvenv=True,
            require_system_site_packages=True,
        )
        != target_active_site
    ):
        raise EnvironmentBuildError("final Python startup surface drifted")
    final_runtime_no_site = _runtime_no_site_evidence(
        base_safe_probe=final_base_safe,
        base_audits=_base_prefix_audits(base_active_site),
        base_probe=base_probe,
        resolved_base_safe_probe=final_base_safe,
        resolved_base_audits=(_resolved_base_audit(base_active_site),),
        resolved_base_probe=resolved_base_probe,
        target_safe_probe=final_target_safe,
        target_audits=target_runtime_audits,
        target_probe=detailed_probe,
    )
    if final_runtime_no_site != runtime_no_site_value:
        raise EnvironmentBuildError("final explicit no-site runtime evidence drifted")
    if _validate_official_environment(official_env_yaml) != official:
        raise EnvironmentBuildError("official environment authority drifted")
    _revalidate_directory_identity(root, root_identity, "environment root")
    if _require_python_launcher(python, "benchmark Python")[1] != target_launcher:
        raise EnvironmentBuildError("benchmark Python launcher changed during build")
    if _require_python_launcher(base_python, "base Python")[1] != base_launcher:
        raise EnvironmentBuildError("base Python launcher changed during build")
    receipt_value: Dict[str, object] = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "status": STATUS,
        "environment_root": str(root),
        "python": str(python),
        "python_sha256": str(target_launcher["resolved_executable_sha256"]),
        "isolated_from_primary_training_environment": True,
        "runtime_no_site": True,
        "sealed_test_or_real_accessed": False,
        "live_probe": live_probe,
        "isolation_model": (
            "fresh_venv_with_fd_anchored_explicit_sites_and_no_site_startup"
        ),
        "application_distributions_installed": {
            name: version for name, version, _module in ADDITIONS
        },
        "inherited_distributions_verified": [
            name for name, _module in INHERITED_DISTRIBUTIONS
        ],
        "official_shreddingnet_environment": official,
        "base_runtime": {
            "python": str(base_python),
            "python_sha256": base_python_sha,
            "launcher_identity": base_launcher,
            "safe_probe": base_safe_probe,
            "active_site_audit": base_active_site,
            "probe": base_probe,
            "resolved_probe": resolved_base_probe,
            "numpy_adaptation": {
                "distribution": "numpy",
                "official_shreddingnet_version": OFFICIAL_NUMPY_VERSION,
                "inherited_base_version": EXPECTED_NUMPY_ADAPTATION_VERSION,
                "intentional": True,
                "reason": "Rachel benchmark compatibility with the provisioned Torch 2.5.1+cu124 runtime",
            },
        },
        "detailed_probe": detailed_probe,
        "functional_cpu_probe": functional_probe,
        "pip_freeze": freeze,
        "build_evidence": build_evidence,
        "builder_source": {
            "path": str(builder_source),
            "sha256": _sha256_file(builder_source),
        },
    }
    receipt_value = _content_bound(receipt_value)
    _validate_receipt_value(
        receipt_value,
        root=root,
        python=python,
        expected_receipt_sha256=None,
        receipt_file_sha256=None,
    )
    if _sha256_file(builder_source) != receipt_value["builder_source"]["sha256"]:
        raise EnvironmentBuildError("environment builder source changed before receipt")
    receipt_path = root / RECEIPT_NAME
    if _lstat(receipt_path) is not None:
        raise EnvironmentBuildError("environment completion receipt already exists")
    receipt_payload = _formatted_json_bytes(receipt_value)
    receipt_sha256 = _sha256_bytes(receipt_payload)
    _revalidate_directory_identity(root, root_identity, "environment root")
    # This atomic RENAME_NOREPLACE is the final filesystem mutation on success.
    _write_json_new_immutable(
        receipt_path,
        receipt_value,
        "environment completion receipt",
        bind_content=False,
    )
    return {
        "schema_version": ENVIRONMENT_SCHEMA,
        "status": STATUS,
        "environment_root": str(root),
        "python": str(python),
        "python_sha256": receipt_value["python_sha256"],
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "receipt_content_sha256": receipt_value["content_sha256"],
        "import_inventory_content_sha256": inventory["content_sha256"],
        "runtime_no_site": True,
    }


def verify_environment(
    *,
    environment_root: Path,
    receipt_path: Optional[Path] = None,
    expected_receipt_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Read-only, content-level verification suitable before every queue stage."""

    _require_safe_controller_runtime()
    root = _require_directory(environment_root, "environment root")
    if root != _lexical_absolute(str(environment_root), "environment root"):
        raise EnvironmentBuildError("environment root identity differs")
    execution_anchor = _ExecutionAnchor(root, "environment root")
    try:
        return _verify_environment_anchored(
            root=root,
            receipt_path=receipt_path,
            expected_receipt_sha256=expected_receipt_sha256,
            execution_anchor=execution_anchor,
        )
    finally:
        execution_anchor.close()


def _verify_environment_anchored(
    *,
    root: Path,
    receipt_path: Optional[Path],
    expected_receipt_sha256: Optional[str],
    execution_anchor: _ExecutionAnchor,
) -> Dict[str, object]:
    receipt = receipt_path if receipt_path is not None else root / RECEIPT_NAME
    receipt = _require_read_only_file(receipt, "environment completion receipt")
    if receipt != root / RECEIPT_NAME:
        raise EnvironmentBuildError(
            "environment receipt is not the canonical root receipt"
        )
    value, receipt_sha = _load_json_with_sha(receipt, "environment completion receipt")
    python, target_launcher = _require_python_launcher(
        root / "bin/python", "benchmark Python"
    )
    _validate_receipt_value(
        value,
        root=root,
        python=python,
        expected_receipt_sha256=expected_receipt_sha256,
        receipt_file_sha256=receipt_sha,
    )

    official_value = value["official_shreddingnet_environment"]
    assert isinstance(official_value, dict)
    observed_official = _validate_official_environment(
        Path(str(official_value["env_yaml"]))
    )
    if observed_official != official_value:
        raise EnvironmentBuildError("official environment authority drifted")

    base_runtime = value["base_runtime"]
    assert isinstance(base_runtime, dict)
    base_python, base_launcher = _require_python_launcher(
        Path(str(base_runtime["python"])), "base Python"
    )
    if base_launcher != base_runtime.get("launcher_identity") or base_launcher.get(
        "resolved_executable_sha256"
    ) != base_runtime.get("python_sha256"):
        raise EnvironmentBuildError("base Python binary drifted")
    base_prefix = _lexical_absolute(
        str(base_python.parent.parent), "base Python prefix"
    )
    current_base_safe = _anchored_safe_runtime_probe(base_python, base_prefix)
    _validate_safe_runtime_probe(current_base_safe, base_python, "base Python")
    current_target_safe = _safe_runtime_probe(python, execution_anchor)
    _validate_safe_runtime_probe(current_target_safe, python, "benchmark Python")
    current_base_active = _audit_base_site_surface(base_prefix, current_base_safe)
    current_target_active = _audit_active_site(
        root,
        safe_probe=current_target_safe,
        require_pyvenv=True,
        require_system_site_packages=True,
    )
    build_evidence = value["build_evidence"]
    assert isinstance(build_evidence, dict)
    active_binding = _require_mapping(
        build_evidence.get("active_site"), "active-site evidence binding"
    )
    stored_active = _load_active_site_evidence(root, active_binding)
    expected_active = _active_site_evidence(
        base_safe_probe=current_base_safe,
        base_audit=current_base_active,
        target_safe_probe=current_target_safe,
        target_audit=current_target_active,
    )
    if stored_active != expected_active:
        raise EnvironmentBuildError("active Python startup surface drifted")
    if (
        current_base_safe != base_runtime.get("safe_probe")
        or current_base_active != base_runtime.get("active_site_audit")
        or target_launcher != build_evidence.get("target_launcher_identity")
    ):
        raise EnvironmentBuildError("runtime launcher/startup authority drifted")
    _verify_build_evidence(
        root,
        build_evidence,
        current_base_safe,
        execution_anchor=execution_anchor,
    )

    current_base_probe = _base_probe(
        base_python,
        safe_probe=current_base_safe,
        audits=_base_prefix_audits(current_base_active),
    )
    _validate_base_probe(
        current_base_probe,
        base_python,
        safe_probe=current_base_safe,
        active_site_audit=current_base_active,
    )
    if current_base_probe != base_runtime.get("probe"):
        raise EnvironmentBuildError("base runtime dependency probe drifted")
    resolved_base_python = Path(str(base_launcher["resolved_executable"]))
    current_resolved_base_probe = _base_probe(
        resolved_base_python,
        safe_probe=current_base_safe,
        audits=(_resolved_base_audit(current_base_active),),
    )
    _validate_base_probe(
        current_resolved_base_probe,
        resolved_base_python,
        safe_probe=current_base_safe,
        active_site_audit=_resolved_base_audit(current_base_active),
    )
    if current_resolved_base_probe != base_runtime.get("resolved_probe"):
        raise EnvironmentBuildError("resolved base dependency probe drifted")

    current_target_audits = (
        current_target_active,
        _resolved_base_audit(current_base_active),
    )
    live_probe = _queue_live_probe(
        python,
        safe_probe=current_target_safe,
        audits=current_target_audits,
        execution_anchor=execution_anchor,
    )
    detailed_probe = _detailed_probe(
        python,
        safe_probe=current_target_safe,
        audits=current_target_audits,
        execution_anchor=execution_anchor,
    )
    _validate_target_probes(
        root=root,
        python=python,
        base_probe=current_base_probe,
        inherited_base_probe=current_resolved_base_probe,
        safe_probes=(current_target_safe, current_base_safe),
        active_site_audits=(
            current_target_active,
            _resolved_base_audit(current_base_active),
        ),
        live=live_probe,
        detailed=detailed_probe,
    )
    if live_probe != value.get("live_probe"):
        raise EnvironmentBuildError("queue live probe drifted")
    if detailed_probe != value.get("detailed_probe"):
        raise EnvironmentBuildError("detailed dependency probe drifted")
    expected_runtime_no_site = _runtime_no_site_evidence(
        base_safe_probe=current_base_safe,
        base_audits=_base_prefix_audits(current_base_active),
        base_probe=current_base_probe,
        resolved_base_safe_probe=current_base_safe,
        resolved_base_audits=(_resolved_base_audit(current_base_active),),
        resolved_base_probe=current_resolved_base_probe,
        target_safe_probe=current_target_safe,
        target_audits=current_target_audits,
        target_probe=detailed_probe,
    )
    if expected_runtime_no_site != build_evidence.get("runtime_no_site"):
        raise EnvironmentBuildError("explicit no-site runtime evidence drifted")
    functional_probe = _functional_cpu_probe(
        python,
        safe_probe=current_target_safe,
        audits=current_target_audits,
        execution_anchor=execution_anchor,
    )
    if functional_probe != value.get("functional_cpu_probe"):
        raise EnvironmentBuildError("functional CPU probe drifted")
    freeze = _pip_freeze(
        python,
        safe_probe=current_target_safe,
        audits=current_target_audits,
        execution_anchor=execution_anchor,
    )
    if freeze != value.get("pip_freeze"):
        raise EnvironmentBuildError("pip freeze drifted")

    inventory_binding = build_evidence.get("inventory")
    if not isinstance(inventory_binding, dict):
        raise EnvironmentBuildError("receipt inventory binding is malformed")
    inventory_path = _require_read_only_file(
        Path(str(inventory_binding.get("path"))), "stored import inventory"
    )
    if inventory_path != root / INVENTORY_RELATIVE_PATH:
        raise EnvironmentBuildError("stored import inventory path differs")
    stored_inventory, inventory_file_sha256 = _load_json_with_sha(
        inventory_path, "stored import inventory"
    )
    if inventory_file_sha256 != inventory_binding.get("file_sha256"):
        raise EnvironmentBuildError("stored import inventory file drifted")
    _validate_content_sha(stored_inventory, "stored import inventory")
    if stored_inventory.get("content_sha256") != inventory_binding.get(
        "content_sha256"
    ):
        raise EnvironmentBuildError("stored import inventory binding differs")
    current_inventory = _import_inventory(
        python,
        root,
        safe_probe=current_target_safe,
        audits=current_target_audits,
        execution_anchor=execution_anchor,
    )
    _validate_inventory_identity(current_inventory, root, python)
    if current_inventory != stored_inventory:
        raise EnvironmentBuildError("imported package content inventory drifted")

    builder_source = value["builder_source"]
    assert isinstance(builder_source, dict)
    current_builder = _require_regular_file(
        Path(__file__), "environment builder source"
    )
    if builder_source.get("path") != str(current_builder) or builder_source.get(
        "sha256"
    ) != _sha256_file(current_builder):
        raise EnvironmentBuildError("environment verifier source identity differs")
    return {
        "schema_version": ENVIRONMENT_SCHEMA,
        "status": "verified_isolated_benchmark_environment",
        "environment_root": str(root),
        "python": str(python),
        "python_sha256": value["python_sha256"],
        "receipt": str(receipt),
        "receipt_sha256": receipt_sha,
        "receipt_content_sha256": value["content_sha256"],
        "import_inventory_content_sha256": stored_inventory["content_sha256"],
        "runtime_no_site": True,
        "sealed_test_or_real_accessed": False,
    }


def _require_mapping(value: object, label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise EnvironmentBuildError(label + " must be a JSON object")
    return value


def _validate_receipt_value(
    value: Mapping[str, object],
    *,
    root: Path,
    python: Path,
    expected_receipt_sha256: Optional[str],
    receipt_file_sha256: Optional[str],
) -> None:
    _validate_content_sha(value, "environment completion receipt")
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "environment_root",
            "python",
            "python_sha256",
            "isolated_from_primary_training_environment",
            "runtime_no_site",
            "sealed_test_or_real_accessed",
            "live_probe",
            "isolation_model",
            "application_distributions_installed",
            "inherited_distributions_verified",
            "official_shreddingnet_environment",
            "base_runtime",
            "detailed_probe",
            "functional_cpu_probe",
            "pip_freeze",
            "build_evidence",
            "builder_source",
            "content_sha256",
        },
        "environment completion receipt",
    )
    if expected_receipt_sha256 is not None:
        if not _is_sha256(expected_receipt_sha256):
            raise EnvironmentBuildError("expected receipt SHA-256 is malformed")
        if receipt_file_sha256 != expected_receipt_sha256:
            raise EnvironmentBuildError("environment receipt file SHA-256 differs")
    required_identity = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "status": STATUS,
        "environment_root": str(root),
        "python": str(python),
        "python_sha256": _launcher_sha256(python, "benchmark Python"),
        "isolated_from_primary_training_environment": True,
        "runtime_no_site": True,
        "sealed_test_or_real_accessed": False,
    }
    for field, expected in required_identity.items():
        if value.get(field) != expected:
            raise EnvironmentBuildError("environment receipt field differs: " + field)
    for field in (
        "live_probe",
        "official_shreddingnet_environment",
        "base_runtime",
        "detailed_probe",
        "functional_cpu_probe",
        "pip_freeze",
        "build_evidence",
        "builder_source",
    ):
        _require_mapping(value.get(field), "environment receipt " + field)
    if value.get("isolation_model") != (
        "fresh_venv_with_fd_anchored_explicit_sites_and_no_site_startup"
    ):
        raise EnvironmentBuildError("environment isolation model differs")
    if value.get("application_distributions_installed") != {
        name: version for name, version, _module in ADDITIONS
    }:
        raise EnvironmentBuildError("installed application distribution set differs")
    if value.get("inherited_distributions_verified") != [
        name for name, _module in INHERITED_DISTRIBUTIONS
    ]:
        raise EnvironmentBuildError("inherited distribution set differs")
    base_runtime = _require_mapping(value.get("base_runtime"), "base runtime")
    _exact_keys(
        base_runtime,
        {
            "python",
            "python_sha256",
            "launcher_identity",
            "safe_probe",
            "active_site_audit",
            "probe",
            "resolved_probe",
            "numpy_adaptation",
        },
        "base runtime",
    )
    if base_runtime.get("numpy_adaptation") != {
        "distribution": "numpy",
        "official_shreddingnet_version": OFFICIAL_NUMPY_VERSION,
        "inherited_base_version": EXPECTED_NUMPY_ADAPTATION_VERSION,
        "intentional": True,
        "reason": "Rachel benchmark compatibility with the provisioned Torch 2.5.1+cu124 runtime",
    }:
        raise EnvironmentBuildError("intentional NumPy adaptation disclosure differs")
    builder_source = _require_mapping(
        value.get("builder_source"), "builder source identity"
    )
    _exact_keys(builder_source, {"path", "sha256"}, "builder source identity")


def _verify_build_evidence(
    root: Path,
    value: Mapping[str, object],
    base_safe_probe: Mapping[str, object],
    execution_anchor: Optional[_ExecutionAnchor] = None,
) -> None:
    _exact_keys(
        value,
        {
            "download",
            "install",
            "wheels",
            "reviewed_wheel_lock",
            "target_launcher_identity",
            "bin_scripts",
            "inventory",
            "active_site",
            "runtime_no_site",
        },
        "environment build evidence",
    )
    _validate_runtime_no_site_evidence(value.get("runtime_no_site"))
    owned_anchor = execution_anchor is None
    if execution_anchor is None:
        execution_anchor = _ExecutionAnchor(root, "environment root")
    try:
        observed_scripts = _normalize_and_audit_environment_scripts(
            root, execution_anchor, rewrite_descriptor_paths=False
        )
    finally:
        if owned_anchor:
            execution_anchor.close()
    if observed_scripts != value.get("bin_scripts"):
        raise EnvironmentBuildError("environment bin script inventory drifted")
    evidence_root = _require_read_only_directory(
        root / ".environment_build_evidence", "environment build evidence"
    )
    _require_exact_mode(
        evidence_root,
        0o555,
        directory=True,
        label="environment build evidence",
    )
    descriptor = _open_absolute_nofollow(
        evidence_root, directory=True, label="environment build evidence"
    )
    try:
        evidence_names = set(os.listdir(descriptor))
        _revalidate_fd_path(
            evidence_root,
            descriptor,
            directory=True,
            label="environment build evidence",
        )
    finally:
        os.close(descriptor)
    if evidence_names != {
        "active_site_audit.json",
        "import_inventory.json",
        "pip_install_report.json",
        "requirements.sha256.txt",
        "wheelhouse",
    }:
        raise EnvironmentBuildError("environment evidence file set differs")

    lock_binding = _require_mapping(
        value.get("reviewed_wheel_lock"), "reviewed wheel lock binding"
    )
    reviewed_lock = _validate_wheel_lock(
        Path(str(lock_binding.get("path"))),
        lock_binding.get("file_sha256"),
        base_safe_probe,
    )
    if reviewed_lock != lock_binding:
        raise EnvironmentBuildError("reviewed wheel lock binding drifted")
    packages = reviewed_lock["packages"]
    assert isinstance(packages, list)
    authorities = {
        str(item["filename"]): item for item in packages if isinstance(item, dict)
    }
    wheels = value.get("wheels")
    if not isinstance(wheels, list) or len(wheels) != len(ADDITIONS):
        raise EnvironmentBuildError("wheel evidence is malformed")
    wheelhouse = _require_read_only_directory(
        root / WHEELHOUSE_RELATIVE_PATH, "wheelhouse"
    )
    _require_exact_mode(wheelhouse, 0o555, directory=True, label="wheelhouse")
    paths = _list_regular_files(wheelhouse, "wheelhouse")
    if {path.name for path in paths} != set(authorities):
        raise EnvironmentBuildError("wheelhouse reviewed filename set differs")
    for path in paths:
        _require_read_only_file(path, "downloaded wheel")
        _require_exact_mode(path, 0o444, directory=False, label="downloaded wheel")
    observed = [_inspect_wheel(path, authorities[path.name]) for path in paths]
    observed.sort(key=lambda item: str(item["distribution"]))
    if observed != wheels:
        raise EnvironmentBuildError("downloaded wheel evidence drifted")
    download = _require_mapping(value.get("download"), "download evidence")
    _exact_keys(
        download,
        {
            "argv",
            "argv_policy",
            "stdout_sha256",
            "stderr_sha256",
            "requirements_lock",
            "requirements_lock_sha256",
        },
        "download evidence",
    )
    lock = _require_read_only_file(
        root / LOCK_RELATIVE_PATH, "hashed requirements lock"
    )
    _require_exact_mode(lock, 0o444, directory=False, label="hashed requirements lock")
    expected_lock = (
        "\n".join(
            str(item["distribution"])
            + "=="
            + str(item["version"])
            + " --hash=sha256:"
            + str(item["sha256"])
            for item in sorted(packages, key=lambda row: str(row["distribution"]))
        )
        + "\n"
    ).encode("utf-8")
    observed_lock_bytes = _read_bytes_file(lock, "hashed requirements lock")
    if (
        download.get("requirements_lock") != str(lock)
        or download.get("requirements_lock_sha256")
        != _sha256_bytes(observed_lock_bytes)
        or observed_lock_bytes != expected_lock
    ):
        raise EnvironmentBuildError("hashed requirements lock drifted")
    expected_download_argv = _download_evidence_argv(
        root, [str(item["url"]) for item in packages]
    )
    if (
        download.get("argv_policy")
        != (
            "fd_paths_canonicalized_to_lexical_paths_static_payloads_recorded_by_sha256"
        )
        or download.get("argv") != expected_download_argv
        or any(
            not _is_sha256(download.get(field))
            for field in ("stdout_sha256", "stderr_sha256")
        )
    ):
        raise EnvironmentBuildError("wheel download command evidence differs")
    install = _require_mapping(value.get("install"), "install evidence")
    _exact_keys(
        install,
        {
            "argv",
            "argv_policy",
            "stdout_sha256",
            "stderr_sha256",
            "report",
            "report_sha256",
            "report_path_normalization",
            "installed",
        },
        "install evidence",
    )
    report = _require_read_only_file(
        root / INSTALL_REPORT_RELATIVE_PATH, "pip install report"
    )
    _require_exact_mode(report, 0o444, directory=False, label="pip install report")
    if (
        install.get("argv_policy")
        != "fd_paths_canonicalized_to_lexical_paths_static_payloads_recorded_by_sha256"
        or install.get("argv") != _install_evidence_argv(root)
        or any(
            not _is_sha256(install.get(field))
            for field in ("stdout_sha256", "stderr_sha256")
        )
        or install.get("report") != str(report)
        or not _is_sha256(install.get("report_sha256"))
        or install.get("installed")
        != _validate_install_report(
            report, root, reviewed_lock, str(install["report_sha256"])
        )
    ):
        raise EnvironmentBuildError("pip install report drifted")
    _validate_install_report_path_normalization(
        install.get("report_path_normalization"), root, reviewed_lock
    )

    inventory_binding = _require_mapping(
        value.get("inventory"), "import inventory binding"
    )
    _exact_keys(
        inventory_binding,
        {"path", "file_sha256", "content_sha256"},
        "import inventory binding",
    )
    inventory_path = _require_read_only_file(
        Path(str(inventory_binding.get("path"))), "stored import inventory"
    )
    _require_exact_mode(
        inventory_path, 0o444, directory=False, label="stored import inventory"
    )
    stored_inventory, inventory_file_sha256 = _load_json_with_sha(
        inventory_path, "stored import inventory"
    )
    _validate_content_sha(stored_inventory, "stored import inventory")
    if (
        inventory_path != root / INVENTORY_RELATIVE_PATH
        or inventory_file_sha256 != inventory_binding.get("file_sha256")
        or stored_inventory.get("content_sha256")
        != inventory_binding.get("content_sha256")
    ):
        raise EnvironmentBuildError("stored import inventory evidence drifted")
    active_binding = _require_mapping(
        value.get("active_site"), "active-site evidence binding"
    )
    _exact_keys(
        active_binding,
        {"path", "file_sha256", "content_sha256"},
        "active-site evidence binding",
    )
    active_path = root / ACTIVE_SITE_RELATIVE_PATH
    _require_exact_mode(
        active_path, 0o444, directory=False, label="stored active-site evidence"
    )
    _load_active_site_evidence(root, active_binding)
    target_launcher = _require_mapping(
        value.get("target_launcher_identity"), "target launcher identity"
    )
    if (
        _require_python_launcher(root / "bin/python", "benchmark Python")[1]
        != target_launcher
    ):
        raise EnvironmentBuildError("target Python launcher evidence drifted")


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "build":
            result = build_environment(
                base_python=_lexical_absolute(arguments.base_python, "base Python"),
                environment_root=_lexical_absolute(
                    arguments.environment_root, "environment root"
                ),
                official_env_yaml=_lexical_absolute(
                    arguments.official_shreddingnet_env,
                    "official ShreddingNet env.yaml",
                ),
                wheel_lock_path=_lexical_absolute(
                    arguments.wheel_lock, "reviewed wheel lock"
                ),
                expected_wheel_lock_sha256=arguments.expected_wheel_lock_sha256,
            )
        else:
            receipt = (
                _lexical_absolute(arguments.receipt, "environment receipt")
                if arguments.receipt
                else None
            )
            result = verify_environment(
                environment_root=_lexical_absolute(
                    arguments.environment_root, "environment root"
                ),
                receipt_path=receipt,
                expected_receipt_sha256=arguments.expected_receipt_sha256,
            )
    except EnvironmentBuildError as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
