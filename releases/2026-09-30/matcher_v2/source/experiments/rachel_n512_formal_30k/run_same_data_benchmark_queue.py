"""Fail-closed queue for the Rachel PairingNet and ShreddingNet benchmarks.

The controller is intentionally train/validation-only.  It waits for the
already-running N=512 and matched-MM convergence authorities, then runs the
same-data paper adaptations in this fixed order:

    PairingNet audit -> one-batch GPU smoke -> formal train
    ShreddingNet audit -> one-effective-step-per-stage GPU smoke -> formal train

Every executable source, review decision, official checkout, and isolated
Python environment is hash/identity checked before a formal trainer can start.
The controller has no sealed-test, corrosion, or real-data command.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import unquote, urlsplit


CONFIG_SCHEMA = "rachel-same-data-benchmark-queue-config/2.0"
REVIEW_SCHEMA = "rachel-benchmark-runner-review/1.0"
QUEUE_REVIEW_SCHEMA = "rachel-same-data-benchmark-queue-review/1.0"
TRAIN_VAL_ASSET_REVIEW_SCHEMA = "rachel-train-val-asset-freeze-review/1.0"
ENVIRONMENT_SCHEMA = "rachel-benchmark-isolated-python-environment/1.0"
CONTROLLER_SCHEMA = "rachel-same-data-benchmark-queue/2.0"

PAIRINGNET_COMMIT = "e878b781b2b2065a4b7da09d2f639e8f0a35e97a"
SHREDDINGNET_COMMIT = "0ae3b544ca4e910732f3f459b39aa15cdc62dbcb"
PAIRINGNET_OFFICIAL_REPOSITORY = "https://github.com/zhourixin/PairingNet"
PAIRINGNET_OFFICIAL_KEY_FILE_SHA256 = {
    "PairingNet Code/run.py": (
        "bac51a2eb26832a1f6819361ccdb3dfb823cbe071ae6e5fb12ca3a1b99c0eeee"
    ),
    "PairingNet Code/utils/config.py": (
        "b0d74cd5d76feb1018eceea32e5e6c02acd8e2b25d1f9e8c6c44368f66ef25ac"
    ),
    "PairingNet Code/utils/pipeline.py": (
        "2201b0f2345355fd529ceccdbb63e5891367f5633793d369fa83bba0a16766af"
    ),
    "PairingNet Code/utils/encoder.py": (
        "277b29352a73eb0ab083c3ead814d26b359aec15531ec3e253cbe98f69845106"
    ),
    "PairingNet Code/utils/loss.py": (
        "a084522f67409e4a755be78d1cc59c8c260302757c7f94d27008068db6d45e07"
    ),
    "PairingNet Code/utils/evaluation.py": (
        "5e135d5875355af91250337850260251816d18907fd80a31348bd1d1414a70ac"
    ),
    "PairingNet Code/utils/ransac.py": (
        "2a657035810c1a17f5983a0c079d9cbd4b561e9fbc5399531af83076f9494e0f"
    ),
    "PairingNet Code/PairingNet_train_val_test.py": (
        "b8bd21d882bd2e61ff892f5024e2349ce852a53b77861ebfe1d52c105888ef82"
    ),
    "PairingNet Code/matching_test.py": (
        "01a19d723a2a1643992e7be510899efdc153fe0bd04a95050e935679eba65bec"
    ),
}

METHODS = {
    "pairingnet": {
        "method_id": "pairingnet_rachel_mask_n512_upright_translation_v1",
        "runner_relative_path": (
            "staging/pairwise_v0_2/baselines/rachel_pairingnet_benchmark.py"
        ),
        "adaptation_contract_relative_path": (
            "staging/pairwise_v0_2/baselines/PAIRINGNET_RACHEL_ADAPTATION.md"
        ),
        "smoke_source_relative_path": (
            "experiments/rachel_n512_formal_30k/pairingnet_gpu_smoke.py"
        ),
        "runner_module": (
            "staging.pairwise_v0_2.baselines.rachel_pairingnet_benchmark"
        ),
        "smoke_module": (
            "experiments.rachel_n512_formal_30k.pairingnet_gpu_smoke"
        ),
        "official_commit": PAIRINGNET_COMMIT,
    },
    "shreddingnet": {
        "method_id": (
            "rachel_shreddingnet_maskonly_n512_upright_release0ae3b544_v1"
        ),
        "runner_relative_path": (
            "staging/pairwise_v0_2/baselines/rachel_shreddingnet_benchmark.py"
        ),
        "adaptation_contract_relative_path": (
            "staging/pairwise_v0_2/baselines/RACHEL_SHREDDINGNET_BENCHMARK.md"
        ),
        "runner_module": (
            "staging.pairwise_v0_2.baselines.rachel_shreddingnet_benchmark"
        ),
        "official_commit": SHREDDINGNET_COMMIT,
    },
}

SHREDDINGNET_RELEASE_RECIPE = {
    "seed": 260831,
    "official_release_seed": 1024,
    "contour_cap": 512,
    "patch_size": 7,
    "feature_dim": 64,
    "resgcn_blocks": 14,
    "cycle_radius": 8,
    "decoder_blocks": 2,
    "attention_heads": 8,
    "coarse_epochs": 128,
    "matching_epochs": 128,
    "classify_epochs": 128,
    "coarse_batch_size": 54,
    "matching_batch_size": 36,
    "classify_batch_size": 20,
    "coarse_lr": 1e-4,
    "matching_lr": 1e-3,
    "classify_lr": 1e-4,
    "weight_decay": 5e-4,
    "infonce_temperature": 0.12,
    "focal_alpha": 0.55,
    "focal_gamma": 8.0,
    "correspondence_threshold": 0.006,
    "pair_score_threshold": 0.5,
    "official_top_k": 20,
    "rachel_top_k": 2,
}
SHREDDINGNET_EFFECTIVE_BATCH = {
    "coarse": 54,
    "matching": 36,
    "classify": 20,
}
SHREDDINGNET_METRIC_KEYS = {
    "coarse": {"loss", "infor_loss", "top1_recall", "top5_recall"},
    "matching": {"loss", "positive_loss"},
    "classify": {"loss", "accuracy", "precision", "recall", "f1"},
}

SOURCE_INVENTORY_SCOPE = "whole_immutable_root_python_and_shell_execution_surface"
SOURCE_EXECUTABLE_SUFFIXES = {".py", ".sh"}
SOURCE_FORBIDDEN_SUFFIXES = {
    ".dll",
    ".dylib",
    ".egg",
    ".pth",
    ".pyc",
    ".pyd",
    ".pyo",
    ".so",
    ".whl",
    ".zip",
}
SOURCE_FORBIDDEN_NAMES = {"sitecustomize.py", "usercustomize.py"}
SOURCE_IGNORED_VCS_DIRECTORIES = {".git", ".hg", ".svn"}
IMMUTABLE_SOURCE_FILE_MODE = 0o444
IMMUTABLE_SOURCE_DIRECTORY_MODE = 0o555
CONTROLLER_RELATIVE_PATH = (
    "experiments/rachel_n512_formal_30k/run_same_data_benchmark_queue.py"
)
PAIRINGNET_SMOKE_RELATIVE_PATH = (
    "experiments/rachel_n512_formal_30k/pairingnet_gpu_smoke.py"
)
TRAIN_VAL_ASSET_FREEZE_RELATIVE_PATH = (
    "experiments/rachel_n512_formal_30k/rachel_train_val_asset_freeze.py"
)
TRAIN_VAL_ASSET_FREEZE_MODULE = (
    "experiments.rachel_n512_formal_30k.rachel_train_val_asset_freeze"
)
TRAIN_VAL_ASSET_FREEZE_TEST_RELATIVE_PATH = (
    "experiments/rachel_n512_formal_30k/test_rachel_train_val_asset_freeze.py"
)
TRAIN_VAL_ASSET_FREEZE_SHA256 = (
    "9de93177106e7af86554755ac7186decd19b67d2ce8319d72815c656b5298522"
)
TRAIN_VAL_ASSET_FREEZE_TEST_SHA256 = (
    "e672f6ce60288d5591b7caeb5cab9d9577afb69aed7d210aba73b784ce72482a"
)
STRICT_VERIFIER_RELATIVE_PATH = (
    "experiments/rachel_n512_formal_30k/benchmark_queue_strict_verifier.py"
)
STRICT_VERIFIER_MODULE = (
    "experiments.rachel_n512_formal_30k.benchmark_queue_strict_verifier"
)
ENVIRONMENT_BUILDER_RELATIVE_PATH = (
    "experiments/rachel_n512_formal_30k/build_rachel_benchmark_environment.py"
)
ENVIRONMENT_BUILDER_MODULE = (
    "experiments.rachel_n512_formal_30k.build_rachel_benchmark_environment"
)
ENVIRONMENT_BUILDER_SHA256 = (
    "b4814505925fac49accf4546f1675711160e84c615d451ce7d08d7a38c58f53f"
)
ENVIRONMENT_BUILDER_TEST_RELATIVE_PATH = (
    "experiments/rachel_n512_formal_30k/"
    "test_build_rachel_benchmark_environment.py"
)
ENVIRONMENT_BUILDER_TEST_SHA256 = (
    "7710d5f71b71ce922c4e14f344dd2d2e7acafa33c0ce64f56175a6dacb33a7f5"
)
ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SCHEMA = (
    "rachel-benchmark-explicit-site-bootstrap/3.0"
)
ENVIRONMENT_RUNTIME_NO_SITE_SCHEMA = (
    "rachel-benchmark-runtime-no-site-evidence/2.0"
)
ENVIRONMENT_TORCHVISION_DYNAMIC_ADMISSION_SCHEMA = (
    "rachel-benchmark-torchvision-dynamic-module-admission/1.0"
)
ENVIRONMENT_EXPLICIT_SITE_STDIN_LAUNCHER_SHA256 = (
    "64b34393434875ee75527d60a146eeb9c9d7c9beb9087b1e37a7b203ffb23177"
)
ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SHA256 = (
    "26ec23abd4a82124ff9c2ed66c146079b076764761df755fe2790b0e2319c58c"
)
ENVIRONMENT_TORCHVISION_DYNAMIC_ADMISSION_SOURCE_SHA256 = (
    "c28611ebd267d12b23b19eb312473f0861a21aa5b8e63cab535071c92f9a2fdc"
)
ENVIRONMENT_PIP_NO_SITE_RUNNER_SHA256 = (
    "bb7c18bdd008168dc4cc952c0173159ad5b66555e91efd6a905f6d860dfcbcf6"
)
ENVIRONMENT_RUNTIME_PAYLOAD_SHA256 = {
    "queue_live_probe": (
        "5a207655ff47e30bb78eb2a0451d97b480a6e93cc997108a09b053f0682482f5"
    ),
    "detailed_probe": (
        "0e7502b68a5443597d74325524d0f8506d28bb1a4bddc08604b6b2fa93b6cffc"
    ),
    "functional_cpu_probe": (
        "1766dd63d9a5479b29d92015a5ea8eea7ef554d4fcff095642313e35b97297fe"
    ),
    "import_inventory": (
        "9c7afb4bf0edb27b54d2683f4ed9dc8c0234d5ad2d271c2dfed349e142986039"
    ),
}
ENVIRONMENT_TORCHVISION_DYNAMIC_AUTHORITIES = {
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
ENVIRONMENT_TORCHVISION_DYNAMIC_POLICY = {
    "controlled_sys_path_default_deny": True,
    "single_pinned_instantiator_append": True,
    "captured_directory_fd_held_through_payload": True,
    "captured_directory_identity": (
        "regular_directory,euid_owner,mode_0700,stable_dev_inode"
    ),
    "captured_file_identity": (
        "regular_file,euid_owner,mode_0644,nlink_1,stable_fd_hash"
    ),
    "captured_path_import_denied": True,
    "generated_write_fd_anchored": True,
    "preseed_module_memory_only": True,
    "captured_origin_compile_and_exec_denied": True,
    "pinned_modules_loaded_from_held_source_bytes": True,
    "pinned_module_bytecode_cache_disabled": True,
    "reviewed_sys_path_restored_before_payload": True,
}
ENVIRONMENT_TORCHVISION_PINNED_MODULES = (
    "torch.distributed.nn.api.remote_module",
    "torch.distributed.nn.jit.instantiator",
    "torch.distributed.nn.jit.templates.remote_module_template",
)
ENVIRONMENT_REVIEW_SCHEMA = "rachel-benchmark-environment-review/1.0"
ENVIRONMENT_REVIEW_RELATIVE_PATH = (
    "experiments/rachel_n512_formal_30k/reviews/"
    "benchmark_environment_review_GO.json"
)
ENVIRONMENT_REVIEW_SHA256 = (
    "e34a1ce9deae4927e9c0717c543b308affe17cf704ffa06184381acdffea2dfd"
)
ENVIRONMENT_REVIEWED_WHEEL_LOCK_RELATIVE_PATH = (
    "experiments/rachel_n512_formal_30k/"
    "rachel_benchmark_reviewed_wheel_lock.json"
)
ENVIRONMENT_REVIEWED_WHEEL_LOCK_SHA256 = (
    "c0e2861e4148174fc25c5e14be539e29470065ed93f06814a3e15cd7540e0406"
)
ENVIRONMENT_EVIDENCE_DIRECTORY = ".environment_build_evidence"
ENVIRONMENT_WHEELHOUSE_RELATIVE_PATH = (
    ENVIRONMENT_EVIDENCE_DIRECTORY + "/wheelhouse"
)
ENVIRONMENT_REQUIREMENTS_LOCK_RELATIVE_PATH = (
    ENVIRONMENT_EVIDENCE_DIRECTORY + "/requirements.sha256.txt"
)
ENVIRONMENT_INSTALL_REPORT_RELATIVE_PATH = (
    ENVIRONMENT_EVIDENCE_DIRECTORY + "/pip_install_report.json"
)
ENVIRONMENT_PIP_ARGV_POLICY = (
    "fd_paths_canonicalized_to_lexical_paths_static_payloads_recorded_by_sha256"
)
QUEUE_CHILD_MODULES = {
    "experiments.rachel_n512_formal_30k.benchmark_queue_strict_verifier",
    "experiments.rachel_n512_formal_30k.build_rachel_benchmark_environment",
    "experiments.rachel_n512_formal_30k.pairingnet_gpu_smoke",
    "experiments.rachel_n512_formal_30k.rachel_train_val_asset_freeze",
    "staging.pairwise_v0_2.baselines.rachel_pairingnet_benchmark",
    "staging.pairwise_v0_2.baselines.rachel_shreddingnet_benchmark",
}
QUEUE_CHILD_TRANSITIVE_SOURCE_RELATIVE_PATHS = {
    "experiments/__init__.py",
    "experiments/rachel_n512_formal_30k/__init__.py",
    STRICT_VERIFIER_RELATIVE_PATH,
    PAIRINGNET_SMOKE_RELATIVE_PATH,
    TRAIN_VAL_ASSET_FREEZE_RELATIVE_PATH,
    ENVIRONMENT_BUILDER_RELATIVE_PATH,
    "staging/__init__.py",
    "staging/pairwise_v0_2/__init__.py",
    "staging/pairwise_v0_2/baselines/__init__.py",
    "staging/pairwise_v0_2/baselines/rachel_matched_mm_evaluation.py",
    "staging/pairwise_v0_2/baselines/rachel_matched_mm_siamese.py",
    "staging/pairwise_v0_2/baselines/rachel_pairingnet_benchmark.py",
    "staging/pairwise_v0_2/baselines/"
    "rachel_same_data_benchmark_eval_adapter.py",
    "staging/pairwise_v0_2/baselines/rachel_shreddingnet_benchmark.py",
    "staging/pairwise_v0_2/models/__init__.py",
    "staging/pairwise_v0_2/models/coarse.py",
    "staging/pairwise_v0_2/models/frozen_transport_matrix_readout.py",
    "staging/pairwise_v0_2/models/local_matcher.py",
    "staging/pairwise_v0_2/models/optimal_transport.py",
    "staging/pairwise_v0_2/models/pairwise.py",
    "staging/pairwise_v0_2/models/rachel_n512.py",
    "staging/pairwise_v0_2/pairwise_data/__init__.py",
    "staging/pairwise_v0_2/pairwise_data/rachel_training_dataset.py",
    "staging/pairwise_v0_2/training/__init__.py",
    "staging/pairwise_v0_2/training/evaluation.py",
    "staging/pairwise_v0_2/training/rachel_n512_loss.py",
    "staging/pairwise_v0_2/training/rachel_n512_runner.py",
    "staging/pairwise_v0_2/training/rachel_n512_sealed_test.py",
}
SOURCE_AUTHORITY_RELATIVE_PATHS = {
    CONTROLLER_RELATIVE_PATH,
    TRAIN_VAL_ASSET_FREEZE_TEST_RELATIVE_PATH,
    ENVIRONMENT_BUILDER_TEST_RELATIVE_PATH,
}
REQUIRED_SOURCE_RELATIVE_PATHS = (
    QUEUE_CHILD_TRANSITIVE_SOURCE_RELATIVE_PATHS
    | SOURCE_AUTHORITY_RELATIVE_PATHS
)

ISOLATED_MODULE_BOOTSTRAP = r'''import importlib.machinery, importlib.util, os, pathlib, stat, sys
if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
    raise SystemExit("isolated/no-site/no-bytecode bootstrap flags are required")
if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
    raise SystemExit("site/user customization was imported before bootstrap")
if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
    raise SystemExit("Python path/home injection is forbidden")
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
module = sys.argv[2]
arguments = sys.argv[3:]
if not module or any(not part.isidentifier() for part in module.split(".")):
    raise SystemExit("invalid reviewed module name")
parts = module.split(".")
if len(parts) < 2 or parts[0] not in {"experiments", "staging"}:
    raise SystemExit("reviewed module is outside the frozen package roots")

def exact_existing_path(value, expected, label):
    if not isinstance(value, (str, bytes, os.PathLike)):
        raise SystemExit(label + " is missing")
    try:
        observed = pathlib.Path(value).resolve(strict=True)
    except OSError as error:
        raise SystemExit(label + " cannot be resolved: " + str(error))
    if observed != expected:
        raise SystemExit(label + " resolved outside immutable source")

def exact_package_directory(path, label):
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SystemExit(label + " cannot be opened: " + str(error))
    if not stat.S_ISDIR(metadata.st_mode) or path.resolve(strict=True) != path:
        raise SystemExit(label + " is not a physical immutable-source directory")

def exact_regular_file(path, label):
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SystemExit(label + " cannot be opened: " + str(error))
    if not stat.S_ISREG(metadata.st_mode) or path.resolve(strict=True) != path:
        raise SystemExit(label + " is not a physical immutable-source file")

def exact_search_locations(value, expected, label):
    if value is None:
        raise SystemExit(label + " has no package search locations")
    try:
        locations = list(value)
    except TypeError as error:
        raise SystemExit(label + " search locations are malformed: " + str(error))
    if len(locations) != 1:
        raise SystemExit(label + " search locations are not uniquely frozen")
    exact_existing_path(locations[0], expected, label + " search location")

source_entries = [
    value
    for value in sys.path
    if isinstance(value, str) and os.path.realpath(value) == str(root)
]
if len(source_entries) != 1:
    raise SystemExit("immutable source is not one reviewed held import root")
search_locations = [str(root)]
for depth in range(1, len(parts)):
    package_name = ".".join(parts[:depth])
    package_directory = root.joinpath(*parts[:depth])
    package_init = package_directory / "__init__.py"
    exact_package_directory(package_directory, package_name)
    exact_regular_file(package_init, package_name + " initializer")
    if package_name in sys.modules:
        raise SystemExit(package_name + " was imported before reviewed bootstrap")
    package_spec = importlib.machinery.PathFinder.find_spec(
        package_name, search_locations
    )
    if package_spec is None or package_spec.loader is None:
        raise SystemExit(package_name + " cannot be resolved as a regular package")
    exact_existing_path(
        package_spec.origin, package_init, package_name + " spec origin"
    )
    exact_search_locations(
        package_spec.submodule_search_locations,
        package_directory,
        package_name + " spec",
    )
    get_filename = getattr(package_spec.loader, "get_filename", None)
    if get_filename is None:
        raise SystemExit(package_name + " loader has no frozen filename")
    exact_existing_path(
        get_filename(package_name), package_init, package_name + " loader filename"
    )
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package
    try:
        package_spec.loader.exec_module(package)
    except BaseException:
        sys.modules.pop(package_name, None)
        raise
    exact_existing_path(
        getattr(package, "__file__", None),
        package_init,
        package_name + " __file__",
    )
    loaded_spec = getattr(package, "__spec__", None)
    if loaded_spec is None:
        raise SystemExit(package_name + " has no loaded spec")
    exact_existing_path(
        loaded_spec.origin, package_init, package_name + " loaded spec origin"
    )
    exact_search_locations(
        getattr(package, "__path__", None),
        package_directory,
        package_name + " __path__",
    )
    exact_search_locations(
        loaded_spec.submodule_search_locations,
        package_directory,
        package_name + " loaded spec",
    )
    if depth > 1:
        parent_name = ".".join(parts[: depth - 1])
        setattr(sys.modules[parent_name], parts[depth - 1], package)
    search_locations = [str(package_directory)]

expected = root.joinpath(*parts).with_suffix(".py")
exact_regular_file(expected, "reviewed module")
if module in sys.modules:
    raise SystemExit("reviewed module was imported before bootstrap")
spec = importlib.machinery.PathFinder.find_spec(module, search_locations)
if spec is None or spec.loader is None or spec.submodule_search_locations is not None:
    raise SystemExit("reviewed module cannot be resolved")
exact_existing_path(spec.origin, expected, "reviewed module spec origin")
get_filename = getattr(spec.loader, "get_filename", None)
get_code = getattr(spec.loader, "get_code", None)
if get_filename is None or get_code is None:
    raise SystemExit("reviewed module loader cannot provide frozen source code")
exact_existing_path(get_filename(module), expected, "reviewed module loader filename")
code = get_code(module)
if code is None:
    raise SystemExit("reviewed module loader returned no code")
sys.argv = [str(expected), *arguments]
namespace = {
    "__name__": "__main__",
    "__file__": str(expected),
    "__cached__": None,
    "__loader__": spec.loader,
    "__package__": module.rpartition(".")[0],
    "__spec__": spec,
}
exec(code, namespace, namespace)
'''
TERMINATION_SIGNALS = {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}
THREAT_MODEL_LIMITATIONS = (
    "single-user non-adversarial server: source, official, upstream, and "
    "original-dataset lexical roots are fully revalidated before/after each "
    "child but are not all held by persistent directory descriptors across "
    "the final revalidation-to-Popen interval",
)


class QueueContractError(RuntimeError):
    """A fail-closed queue invariant was violated."""


class QueueSignal(BaseException):
    """A termination signal interrupted the queue."""

    def __init__(self, signum: int):
        self.signum = int(signum)
        super().__init__(signal.Signals(signum).name)


def _child_unblock_termination_signals() -> None:
    """Undo the parent's atomic-spawn mask in the just-forked child."""

    signal.pthread_sigmask(signal.SIG_UNBLOCK, TERMINATION_SIGNALS)


def _open_anchored_environment_python(
    environment: Mapping[str, object],
) -> tuple[int, str, str]:
    """Hold the verified environment inode across one child process lifetime."""

    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required_flags):
        raise QueueContractError("directory-FD environment anchoring is unsupported")
    root = Path(str(environment["root"]))
    descriptor = os.open(
        str(root),
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or int(metadata.st_dev) != environment["root_device"]
            or int(metadata.st_ino) != environment["root_inode"]
        ):
            raise QueueContractError("benchmark environment root identity changed")
        proc_descriptor_root = Path("/proc") / str(os.getpid()) / "fd" / str(
            descriptor
        )
        if proc_descriptor_root.is_dir():
            anchored_root = proc_descriptor_root
        elif sys.platform == "darwin":
            # Darwin's /dev/fd directory descriptors are not traversable or
            # executable.  CUDA formal execution is Linux-only; retain the FD
            # and strict post-child identity check for local fixture coverage.
            anchored_root = root
        else:
            raise QueueContractError(
                "Linux controller-PID /proc FD environment anchoring is unavailable"
            )
        anchored_python = anchored_root / str(environment["python_relative_path"])
        python_metadata = anchored_python.stat()
        if (
            not stat.S_ISREG(python_metadata.st_mode)
            or int(python_metadata.st_dev) != environment["python_device"]
            or int(python_metadata.st_ino) != environment["python_inode"]
        ):
            raise QueueContractError("anchored benchmark Python identity changed")
        return descriptor, str(anchored_root), str(anchored_python)
    except BaseException:
        os.close(descriptor)
        raise


def _open_absolute_directory_nofollow(path: Path, label: str) -> int:
    """Open one absolute directory without following any component symlink."""

    if not path.is_absolute() or path != Path(os.path.normpath(str(path))):
        raise QueueContractError(label + " path is not normalized absolute")
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required_flags):
        raise QueueContractError(label + " nofollow directory open is unsupported")
    descriptor = os.open(
        os.path.sep, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    )
    try:
        for part in path.parts[1:]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise QueueContractError(label + " is not a directory")
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise QueueContractError(
            label + " cannot be opened without symlink traversal"
        ) from error
    except BaseException:
        os.close(descriptor)
        raise


def _site_descriptor_path(descriptor: int) -> str:
    if sys.platform.startswith("linux"):
        path = "/proc/{}/fd/{}".format(os.getpid(), descriptor)
    elif sys.platform == "darwin":
        path = "/dev/fd/{}".format(descriptor)
    else:
        raise QueueContractError("explicit-site FD platform is unsupported")
    followed = os.stat(path, follow_symlinks=True)
    held = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(followed.st_mode)
        or int(followed.st_ino) != int(held.st_ino)
        or (
            sys.platform.startswith("linux")
            and int(followed.st_dev) != int(held.st_dev)
        )
    ):
        raise QueueContractError("explicit-site FD magic-link identity differs")
    return path


def _open_anchored_source_binding(
    source_root: Path, source_inventory_sha256: str
) -> tuple[int, dict[str, object]]:
    descriptor = _open_absolute_directory_nofollow(
        source_root, "immutable source execution root"
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != IMMUTABLE_SOURCE_DIRECTORY_MODE
        ):
            raise QueueContractError("immutable source execution root mode differs")
        return descriptor, {
            "canonical_path": str(source_root),
            "descriptor_path": _site_descriptor_path(descriptor),
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "role": "immutable_source",
            "audit_content_sha256": _require_sha256(
                source_inventory_sha256, "immutable source inventory"
            ),
        }
    except BaseException:
        os.close(descriptor)
        raise


def _close_and_revalidate_source_binding(
    descriptor: int, binding: Mapping[str, object]
) -> None:
    error: Optional[BaseException] = None
    try:
        held = os.fstat(descriptor)
        lexical_descriptor = _open_absolute_directory_nofollow(
            Path(str(binding["canonical_path"])),
            "immutable source lexical revalidation",
        )
        try:
            lexical = os.fstat(lexical_descriptor)
        finally:
            os.close(lexical_descriptor)
        expected = (int(binding["device"]), int(binding["inode"]))
        if (
            (int(held.st_dev), int(held.st_ino)) != expected
            or (int(lexical.st_dev), int(lexical.st_ino)) != expected
        ):
            raise QueueContractError(
                "immutable source lexical/held directory identity changed"
            )
    except BaseException as observed:
        error = observed
    finally:
        os.close(descriptor)
    if error is not None:
        raise error


def _open_anchored_environment_sites(
    environment: Mapping[str, object],
) -> tuple[dict[str, list[dict[str, object]]], tuple[int, ...]]:
    """Hold every reviewed base/resolved/target site directory for one child."""

    evidence = environment.get("runtime_no_site")
    if not isinstance(evidence, dict):
        raise QueueContractError("environment runtime no-site evidence is missing")
    held: dict[str, list[dict[str, object]]] = {}
    descriptors: list[int] = []
    try:
        for context_name in ("base", "resolved_base", "target"):
            context = evidence.get(context_name)
            bindings = (
                context.get("injected_site_directories")
                if isinstance(context, dict)
                else None
            )
            if not isinstance(bindings, list) or not bindings:
                raise QueueContractError(
                    context_name + " explicit-site evidence is missing"
                )
            rows = []
            for index, binding_value in enumerate(bindings):
                if not isinstance(binding_value, dict):
                    raise QueueContractError("explicit-site binding is malformed")
                canonical_path = Path(str(binding_value["canonical_path"]))
                descriptor = _open_absolute_directory_nofollow(
                    canonical_path,
                    "{} explicit site {}".format(context_name, index),
                )
                descriptors.append(descriptor)
                metadata = os.fstat(descriptor)
                if (
                    int(metadata.st_dev) != binding_value["device"]
                    or int(metadata.st_ino) != binding_value["inode"]
                ):
                    raise QueueContractError(
                        context_name + " explicit-site identity changed"
                    )
                rows.append(
                    {
                        "canonical_path": str(canonical_path),
                        "descriptor_path": _site_descriptor_path(descriptor),
                        "device": int(metadata.st_dev),
                        "inode": int(metadata.st_ino),
                        "role": binding_value["role"],
                        "audit_content_sha256": binding_value[
                            "audit_content_sha256"
                        ],
                    }
                )
            held[context_name] = rows
        return held, tuple(descriptors)
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _close_and_revalidate_environment_sites(
    descriptors: Sequence[int],
    held_sites: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    """Reopen each lexical site without symlinks, compare, then close all FDs."""

    rows = [
        row
        for context_name in ("base", "resolved_base", "target")
        for row in held_sites[context_name]
    ]
    error: Optional[BaseException] = None
    try:
        if len(rows) != len(descriptors):
            raise QueueContractError("held explicit-site descriptor count differs")
        for descriptor, row in zip(descriptors, rows):
            held = os.fstat(descriptor)
            lexical_descriptor = _open_absolute_directory_nofollow(
                Path(str(row["canonical_path"])), "explicit-site lexical revalidation"
            )
            try:
                lexical = os.fstat(lexical_descriptor)
            finally:
                os.close(lexical_descriptor)
            expected = (int(row["device"]), int(row["inode"]))
            if (
                (int(held.st_dev), int(held.st_ino)) != expected
                or (int(lexical.st_dev), int(lexical.st_ino)) != expected
            ):
                raise QueueContractError(
                    "explicit-site lexical/held directory identity changed"
                )
    except BaseException as observed:
        error = observed
    finally:
        for descriptor in reversed(tuple(descriptors)):
            os.close(descriptor)
    if error is not None:
        raise error


def _explicit_site_child_configuration(
    environment: Mapping[str, object],
    held_sites: Mapping[str, Sequence[Mapping[str, object]]],
    source_binding: Mapping[str, object],
) -> dict[str, object]:
    evidence = environment["runtime_no_site"]
    assert isinstance(evidence, dict)
    target = evidence["target"]
    assert isinstance(target, dict)
    value: dict[str, object] = {
        "schema_version": ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SCHEMA,
        "startup": target["safe_startup_identity"],
        "runtime": target["logical_runtime_identity"],
        "no_site_sys_path": target["no_site_sys_path"],
        "sites": [*list(held_sites["target"]), dict(source_binding)],
        "torchvision_dynamic_admission": (
            _torchvision_dynamic_admission_configuration(True)
        ),
    }
    value["content_sha256"] = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    return value


def _close_and_revalidate_environment_anchor(
    descriptor: int, environment: Mapping[str, object]
) -> None:
    """Prove the held root and its lexical name still identify the same inode."""

    error: Optional[BaseException] = None
    try:
        held = os.fstat(descriptor)
        lexical = os.stat(str(environment["root"]), follow_symlinks=False)
        if (
            not stat.S_ISDIR(held.st_mode)
            or not stat.S_ISDIR(lexical.st_mode)
            or int(held.st_dev) != environment["root_device"]
            or int(held.st_ino) != environment["root_inode"]
            or (int(lexical.st_dev), int(lexical.st_ino))
            != (int(held.st_dev), int(held.st_ino))
        ):
            raise QueueContractError(
                "benchmark environment lexical/held root identity changed"
            )
    except BaseException as observed:
        error = observed
    finally:
        os.close(descriptor)
    if error is not None:
        raise error


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reviewed_environment_runtime_sources(builder: Path) -> dict[str, str]:
    """Extract reviewed bootstrap strings without executing builder top-level code."""

    try:
        payload = builder.read_bytes()
    except OSError as error:
        raise QueueContractError(
            "environment builder bootstrap source cannot be read"
        ) from error
    if hashlib.sha256(payload).hexdigest() != ENVIRONMENT_BUILDER_SHA256:
        raise QueueContractError("environment builder source SHA-256 differs")
    try:
        tree = ast.parse(payload.decode("utf-8"), filename=str(builder))
    except (UnicodeError, SyntaxError) as error:
        raise QueueContractError(
            "environment builder bootstrap source cannot be parsed"
        ) from error
    values: dict[str, str] = {}

    def evaluate(node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return evaluate(node.left) + evaluate(node.right)
        raise QueueContractError(
            "environment builder bootstrap assignment is not static string composition"
        )

    wanted = {
        "EXPLICIT_SITE_DESCRIPTOR_GUARD_CODE",
        "EXPLICIT_SITE_STDIN_LAUNCHER_CODE",
        "TORCHVISION_DYNAMIC_ADMISSION_BOOTSTRAP_CODE",
        "EXPLICIT_SITE_BOOTSTRAP_CODE",
    }
    seen = set()
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in wanted
        ):
            name = statement.targets[0].id
            if name in seen:
                raise QueueContractError(
                    "environment builder bootstrap assignment is duplicated"
                )
            values[name] = evaluate(statement.value)
            seen.add(name)
    if seen != wanted:
        raise QueueContractError("environment builder bootstrap assignments differ")
    observed = {
        "launcher": hashlib.sha256(
            values["EXPLICIT_SITE_STDIN_LAUNCHER_CODE"].encode("utf-8")
        ).hexdigest(),
        "bootstrap": hashlib.sha256(
            values["EXPLICIT_SITE_BOOTSTRAP_CODE"].encode("utf-8")
        ).hexdigest(),
        "dynamic_admission": hashlib.sha256(
            values["TORCHVISION_DYNAMIC_ADMISSION_BOOTSTRAP_CODE"].encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    expected = {
        "launcher": ENVIRONMENT_EXPLICIT_SITE_STDIN_LAUNCHER_SHA256,
        "bootstrap": ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SHA256,
        "dynamic_admission": (
            ENVIRONMENT_TORCHVISION_DYNAMIC_ADMISSION_SOURCE_SHA256
        ),
    }
    if observed != expected:
        raise QueueContractError("environment builder bootstrap source identity differs")
    return {
        "launcher": values["EXPLICIT_SITE_STDIN_LAUNCHER_CODE"],
        "bootstrap": values["EXPLICIT_SITE_BOOTSTRAP_CODE"],
        "dynamic_admission": values[
            "TORCHVISION_DYNAMIC_ADMISSION_BOOTSTRAP_CODE"
        ],
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise QueueContractError(label + " must be one supplied lowercase SHA-256")
    return str(value)


def _lexical_absolute(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise QueueContractError(label + " must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise QueueContractError(label + " must be absolute")
    return Path(os.path.normpath(str(path)))


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise QueueContractError(label + " must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise QueueContractError(label + " is not a confined relative path")
    normalized = path.as_posix()
    if normalized != value:
        raise QueueContractError(label + " is not normalized")
    return normalized


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
            raise QueueContractError(label + " traverses a symlink: " + str(current))


def _require_directory(path: Path, label: str) -> Path:
    _assert_no_symlink_components(path, label)
    metadata = _lstat(path)
    if metadata is None or not stat.S_ISDIR(metadata.st_mode):
        raise QueueContractError(label + " is not an existing directory")
    return path.resolve(strict=True)


def _require_regular_file(path: Path, label: str) -> Path:
    _assert_no_symlink_components(path, label)
    metadata = _lstat(path)
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        raise QueueContractError(label + " is not a regular non-symlink file")
    return path.resolve(strict=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    regular = _require_regular_file(path, label)
    try:
        value = json.loads(regular.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QueueContractError(label + " is not readable JSON") from error
    if not isinstance(value, dict):
        raise QueueContractError(label + " must contain one JSON object")
    return value


def _content_bound(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result["content_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def _validate_content_sha(value: Mapping[str, object], label: str) -> None:
    declared = value.get("content_sha256")
    if not _is_sha256(declared):
        raise QueueContractError(label + " lacks a content SHA-256")
    canonical = dict(value)
    canonical.pop("content_sha256", None)
    if hashlib.sha256(_canonical_bytes(canonical)).hexdigest() != declared:
        raise QueueContractError(label + " content SHA-256 differs")


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _positive_shape_tree(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for shape in value:
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            return False
    return True


def _validate_pairingnet_smoke_batch_order(value: object) -> None:
    if not isinstance(value, dict):
        raise QueueContractError("PairingNet smoke batch order is not an object")
    _exact_keys(
        value,
        {
            "schema_version",
            "seed",
            "epoch_one_based",
            "rows",
            "ordered_pair_ids_sha256",
            "ordered_pair_labels_sha256",
            "positive_pair_ids",
            "negative_pair_ids",
            "contains_positive_and_negative",
        },
        "PairingNet smoke batch order",
    )
    rows = value.get("rows")
    if (
        value.get("schema_version")
        != "rachel-pairingnet-gpu-smoke-batch-order/1.0"
        or value.get("seed") != 260831
        or value.get("epoch_one_based") != 1
        or not isinstance(rows, list)
        or len(rows) != 20
    ):
        raise QueueContractError("PairingNet smoke batch order identity differs")
    normalized_rows = []
    pair_ids = []
    labels = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"ordinal", "pair_id", "label"}:
            raise QueueContractError("PairingNet smoke batch row fields differ")
        pair_id = row.get("pair_id")
        label = row.get("label")
        if (
            row.get("ordinal") != ordinal
            or not isinstance(pair_id, str)
            or not pair_id
            or type(label) is not bool  # noqa: E721
        ):
            raise QueueContractError("PairingNet smoke batch row identity differs")
        normalized_rows.append(dict(row))
        pair_ids.append(pair_id)
        labels.append(label)
    positive = [pair_id for pair_id, label in zip(pair_ids, labels) if label]
    negative = [pair_id for pair_id, label in zip(pair_ids, labels) if not label]
    if (
        len(set(pair_ids)) != len(pair_ids)
        or not positive
        or not negative
        or value.get("positive_pair_ids") != positive
        or value.get("negative_pair_ids") != negative
        or value.get("contains_positive_and_negative") is not True
        or value.get("ordered_pair_ids_sha256")
        != hashlib.sha256(_canonical_bytes(pair_ids)).hexdigest()
        or value.get("ordered_pair_labels_sha256")
        != hashlib.sha256(_canonical_bytes(normalized_rows)).hexdigest()
    ):
        raise QueueContractError("PairingNet smoke batch class/order binding differs")


def _validate_shreddingnet_smoke_stage(
    *,
    stage: str,
    value: object,
    configured_microbatch: int,
    amp: bool,
) -> None:
    if not isinstance(value, dict):
        raise QueueContractError("ShreddingNet " + stage + " smoke row is not an object")
    _exact_keys(
        value,
        {
            "effective_batch_size",
            "configured_release_effective_batch_size",
            "gpu_microbatch_size",
            "nominal_accumulation_steps",
            "amp_enabled",
            "elapsed_seconds",
            "effective_pairs_per_second",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "metrics",
        },
        "ShreddingNet " + stage + " smoke row",
    )
    effective_batch = SHREDDINGNET_EFFECTIVE_BATCH[stage]
    if (
        value.get("effective_batch_size") != effective_batch
        or value.get("configured_release_effective_batch_size") != effective_batch
        or value.get("gpu_microbatch_size") != configured_microbatch
        or configured_microbatch > effective_batch
        or value.get("nominal_accumulation_steps")
        != math.ceil(effective_batch / configured_microbatch)
        or value.get("amp_enabled") is not amp
    ):
        raise QueueContractError(
            "ShreddingNet " + stage + " effective/microbatch exposure differs"
        )
    elapsed = value.get("elapsed_seconds")
    throughput = value.get("effective_pairs_per_second")
    allocated = value.get("peak_allocated_bytes")
    reserved = value.get("peak_reserved_bytes")
    if (
        not _finite_number(elapsed)
        or float(elapsed) <= 0.0
        or not _finite_number(throughput)
        or float(throughput) <= 0.0
        or type(allocated) is not int  # noqa: E721
        or allocated <= 0
        or type(reserved) is not int  # noqa: E721
        or reserved < allocated
    ):
        raise QueueContractError("ShreddingNet " + stage + " resource metrics differ")
    metrics = value.get("metrics")
    expected_metric_keys = SHREDDINGNET_METRIC_KEYS[stage]
    if (
        not isinstance(metrics, dict)
        or set(metrics) != expected_metric_keys
        or any(not _finite_number(metric) for metric in metrics.values())
        or any(
            float(metrics[name]) < 0.0
            for name in expected_metric_keys
            if name in {"loss", "infor_loss", "positive_loss"}
        )
        or any(
            not 0.0 <= float(metrics[name]) <= 1.0
            for name in expected_metric_keys
            if name in {"top1_recall", "top5_recall", "accuracy", "precision", "recall", "f1"}
        )
    ):
        raise QueueContractError("ShreddingNet " + stage + " smoke metrics differ")


def _write_json_new(path: Path, value: Mapping[str, object]) -> dict[str, object]:
    _assert_no_symlink_components(path, "JSON output")
    if os.path.lexists(path):
        raise QueueContractError("refusing to overwrite JSON output: " + str(path))
    parent = _require_directory(path.parent, "JSON output parent")
    payload_value = _content_bound(value)
    payload = json.dumps(
        payload_value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".tmp-", dir=str(parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise QueueContractError("JSON output appeared during publish") from error
        directory_descriptor = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return payload_value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QueueContractError(label + " must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, object], keys: set[str], label: str) -> None:
    observed = set(value)
    if observed != keys:
        missing = sorted(keys - observed)
        extra = sorted(observed - keys)
        raise QueueContractError(
            "{} keys differ (missing={}, extra={})".format(label, missing, extra)
        )


def load_config(path: Path) -> dict[str, Any]:
    config_path = _require_regular_file(path, "queue config")
    value = _load_json(config_path, "queue config")
    _exact_keys(
        value,
        {
            "schema_version",
            "controller",
            "train_val_asset_freeze_authority",
            "immutable_source_root",
            "dataset_root",
            "output_root",
            "benchmark_environment",
            "upstreams",
            "official_sources",
            "methods",
            "runtime",
            "wait",
        },
        "queue config",
    )
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise QueueContractError("queue config schema differs")
    value["_config_path"] = str(config_path)
    value["_config_file_sha256"] = _sha256_file(config_path)
    return value


def _source_file(
    root: Path, relative: object, expected_sha256: object, label: str
) -> tuple[Path, str, str]:
    relative_value = _relative_path(relative, label + " relative path")
    expected = _require_sha256(expected_sha256, label + " SHA-256")
    path = _require_regular_file(root / relative_value, label)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise QueueContractError(label + " escaped immutable source root") from error
    observed = _sha256_file(path)
    if observed != expected:
        raise QueueContractError(label + " SHA-256 differs")
    return path, relative_value, observed


def _source_inventory(root: Path) -> list[dict[str, object]]:
    """Inventory the complete import/launch surface below the immutable root."""

    rows: list[dict[str, object]] = []
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        kept_directories = []
        for name in sorted(directory_names):
            child = directory_path / name
            metadata = child.lstat()
            relative = child.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise QueueContractError(
                    "symlink forbidden in immutable source: " + relative
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise QueueContractError("non-directory source traversal member")
            if name == "__pycache__":
                raise QueueContractError(
                    "bytecode cache forbidden in immutable source: " + relative
                )
            if name not in SOURCE_IGNORED_VCS_DIRECTORIES:
                kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            child = directory_path / name
            metadata = child.lstat()
            relative = child.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise QueueContractError(
                    "symlink forbidden in immutable source: " + relative
                )
            if not stat.S_ISREG(metadata.st_mode):
                raise QueueContractError(
                    "non-regular member forbidden in immutable source: " + relative
                )
            suffix = child.suffix.lower()
            if name.lower() in SOURCE_FORBIDDEN_NAMES:
                raise QueueContractError(
                    "site/user customization forbidden in immutable source: " + relative
                )
            if suffix in SOURCE_FORBIDDEN_SUFFIXES:
                raise QueueContractError(
                    "compiled/path/archive execution surface forbidden: " + relative
                )
            mode = stat.S_IMODE(metadata.st_mode)
            if mode & 0o111 and suffix not in SOURCE_EXECUTABLE_SUFFIXES:
                raise QueueContractError(
                    "unexpected executable surface in immutable source: " + relative
                )
            if suffix in SOURCE_EXECUTABLE_SUFFIXES:
                rows.append(
                    {
                        "path": relative,
                        "bytes": int(metadata.st_size),
                        "mode": mode,
                        "sha256": _sha256_file(child),
                    }
                )
    rows.sort(key=lambda row: str(row["path"]))
    if len({str(row["path"]) for row in rows}) != len(rows):
        raise QueueContractError("immutable source inventory contains duplicate paths")
    return rows


def _source_inventory_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(_canonical_bytes(list(rows))).hexdigest()


def _complete_source_inventory(root: Path) -> list[dict[str, object]]:
    """Hash every member of the intentionally minimal immutable source root."""

    rows: list[dict[str, object]] = []
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        kept_directories = []
        for name in sorted(directory_names):
            child = directory_path / name
            metadata = child.lstat()
            relative = child.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise QueueContractError(
                    "symlink forbidden in immutable source: " + relative
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise QueueContractError(
                    "non-directory member forbidden in immutable source: " + relative
                )
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            child = directory_path / name
            metadata = child.lstat()
            relative = child.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise QueueContractError(
                    "symlink forbidden in immutable source: " + relative
                )
            if not stat.S_ISREG(metadata.st_mode):
                raise QueueContractError(
                    "non-regular member forbidden in immutable source: " + relative
                )
            rows.append(
                {
                    "path": relative,
                    "bytes": int(metadata.st_size),
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "sha256": _sha256_file(child),
                }
            )
    rows.sort(key=lambda row: str(row["path"]))
    return rows


def _validate_minimal_execution_source_paths(
    inventory: Sequence[Mapping[str, object]],
) -> None:
    inventory_paths = {str(row.get("path")) for row in inventory}
    missing_required = sorted(REQUIRED_SOURCE_RELATIVE_PATHS - inventory_paths)
    if missing_required:
        raise QueueContractError(
            "immutable source inventory lacks required reviewed modules: "
            + ", ".join(missing_required)
        )
    unexpected_sources = sorted(inventory_paths - REQUIRED_SOURCE_RELATIVE_PATHS)
    if unexpected_sources:
        raise QueueContractError(
            "immutable source contains Python/shell outside the minimal reviewed "
            "execution whitelist: " + ", ".join(unexpected_sources)
        )


def _validate_immutable_source_permissions(
    root: Path, complete_inventory: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    directory_count = 0
    for directory, directory_names, _file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        metadata = directory_path.lstat()
        relative = (
            "."
            if directory_path == root
            else directory_path.relative_to(root).as_posix()
        )
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != IMMUTABLE_SOURCE_DIRECTORY_MODE
        ):
            raise QueueContractError(
                "immutable source directory mode must be 0555: " + relative
            )
        directory_count += 1
        directory_names.sort()
    bad_files = sorted(
        str(row.get("path"))
        for row in complete_inventory
        if row.get("mode") != IMMUTABLE_SOURCE_FILE_MODE
    )
    if bad_files:
        raise QueueContractError(
            "immutable source regular-file mode must be 0444: "
            + ", ".join(bad_files)
        )
    return {
        "regular_file_mode_octal": "0444",
        "directory_mode_octal": "0555",
        "regular_file_count": len(complete_inventory),
        "directory_count": directory_count,
    }


def _validate_controller_authority(
    source_root: Path,
    value: Mapping[str, object],
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    _exact_keys(
        value,
        {
            "relative_path",
            "sha256",
            "review_marker_relative_path",
            "review_marker_sha256",
        },
        "controller authority",
    )
    if value.get("relative_path") != CONTROLLER_RELATIVE_PATH:
        raise QueueContractError("controller relative path differs")
    controller, controller_relative, controller_sha = _source_file(
        source_root,
        value.get("relative_path"),
        value.get("sha256"),
        "queue controller",
    )
    if controller != Path(__file__).resolve(strict=True):
        raise QueueContractError("running controller differs from external authority")
    marker, marker_relative, marker_sha = _source_file(
        source_root,
        value.get("review_marker_relative_path"),
        value.get("review_marker_sha256"),
        "queue review marker",
    )
    smoke_path = _require_regular_file(
        source_root / PAIRINGNET_SMOKE_RELATIVE_PATH,
        "PairingNet GPU smoke source",
    )
    _validate_minimal_execution_source_paths(inventory)
    inventory_sha = _source_inventory_sha256(inventory)
    review = _load_json(marker, "queue review marker")
    _exact_keys(
        review,
        {
            "schema_version",
            "status",
            "p0_count",
            "p1_count",
            "review_scope_complete",
            "controller",
            "pairingnet_smoke",
            "source_inventory",
            "isolated_bootstrap_reviewed",
            "forbidden_execution_surfaces_absent",
            "sealed_synthetic_accessed",
            "real_data_accessed",
        },
        "queue review marker",
    )
    if (
        review.get("schema_version") != QUEUE_REVIEW_SCHEMA
        or review.get("status") != "GO"
        or review.get("p0_count") != 0
        or review.get("p1_count") != 0
        or review.get("review_scope_complete") is not True
        or review.get("controller")
        != {"relative_path": controller_relative, "sha256": controller_sha}
        or review.get("pairingnet_smoke")
        != {
            "relative_path": PAIRINGNET_SMOKE_RELATIVE_PATH,
            "sha256": _sha256_file(smoke_path),
        }
        or review.get("source_inventory")
        != {
            "scope": SOURCE_INVENTORY_SCOPE,
            "file_count": len(inventory),
            "sha256": inventory_sha,
        }
        or review.get("isolated_bootstrap_reviewed") is not True
        or review.get("forbidden_execution_surfaces_absent") is not True
        or review.get("sealed_synthetic_accessed") is not False
        or review.get("real_data_accessed") is not False
    ):
        raise QueueContractError("queue review marker is not an exact unconditional GO")
    return {
        "path": str(controller),
        "relative_path": controller_relative,
        "sha256": controller_sha,
        "review_marker": {
            "path": str(marker),
            "relative_path": marker_relative,
            "sha256": marker_sha,
        },
        "source_inventory_sha256": inventory_sha,
        "source_inventory_file_count": len(inventory),
    }


def _validate_train_val_asset_authority(
    source_root: Path, value: Mapping[str, object]
) -> dict[str, object]:
    """Require an external GO for the exact v3 frozen-view implementation/tests."""

    _exact_keys(
        value,
        {
            "module_relative_path",
            "module_sha256",
            "test_relative_path",
            "test_sha256",
            "review_marker_relative_path",
            "review_marker_sha256",
        },
        "train/val asset-freeze authority",
    )
    if (
        value.get("module_relative_path") != TRAIN_VAL_ASSET_FREEZE_RELATIVE_PATH
        or value.get("module_sha256") != TRAIN_VAL_ASSET_FREEZE_SHA256
        or value.get("test_relative_path")
        != TRAIN_VAL_ASSET_FREEZE_TEST_RELATIVE_PATH
        or value.get("test_sha256") != TRAIN_VAL_ASSET_FREEZE_TEST_SHA256
    ):
        raise QueueContractError("train/val asset-freeze reviewed source/test differs")
    module, module_relative, module_sha = _source_file(
        source_root,
        value.get("module_relative_path"),
        value.get("module_sha256"),
        "train/val asset-freeze module",
    )
    test, test_relative, test_sha = _source_file(
        source_root,
        value.get("test_relative_path"),
        value.get("test_sha256"),
        "train/val asset-freeze focused tests",
    )
    marker, marker_relative, marker_sha = _source_file(
        source_root,
        value.get("review_marker_relative_path"),
        value.get("review_marker_sha256"),
        "train/val asset-freeze review marker",
    )
    review = _load_json(marker, "train/val asset-freeze review marker")
    _exact_keys(
        review,
        {
            "schema_version",
            "status",
            "p0_count",
            "p1_count",
            "review_scope_complete",
            "module",
            "focused_tests",
            "contracts",
            "same_freeze_root_fd_lifetime_reviewed",
            "final_complete_tree_revalidation_reviewed",
            "regular_file_link_count_one_reviewed",
            "root_replacement_race_tests_passed",
            "train_val_only",
            "sealed_test_accessed",
            "real_data_accessed",
        },
        "train/val asset-freeze review marker",
    )
    expected_contracts = {
        "inventory_schema_version": (
            "rachel-train-val-semantic-asset-inventory/3.0"
        ),
        "receipt_schema_version": "rachel-train-val-semantic-asset-freeze/3.0",
        "verification_schema_version": (
            "rachel-train-val-semantic-asset-verification/3.0"
        ),
        "frozen_view_schema_version": (
            "rachel-frozen-train-val-dataset-view/2.0"
        ),
    }
    if (
        review.get("schema_version") != TRAIN_VAL_ASSET_REVIEW_SCHEMA
        or review.get("status") != "GO"
        or review.get("p0_count") != 0
        or review.get("p1_count") != 0
        or review.get("review_scope_complete") is not True
        or review.get("module")
        != {"relative_path": module_relative, "sha256": module_sha}
        or review.get("focused_tests")
        != {"relative_path": test_relative, "sha256": test_sha}
        or review.get("contracts") != expected_contracts
        or review.get("same_freeze_root_fd_lifetime_reviewed") is not True
        or review.get("final_complete_tree_revalidation_reviewed") is not True
        or review.get("regular_file_link_count_one_reviewed") is not True
        or review.get("root_replacement_race_tests_passed") is not True
        or review.get("train_val_only") is not True
        or review.get("sealed_test_accessed") is not False
        or review.get("real_data_accessed") is not False
    ):
        raise QueueContractError(
            "train/val asset-freeze review marker is not an exact unconditional GO"
        )
    return {
        "module": {
            "path": str(module),
            "relative_path": module_relative,
            "sha256": module_sha,
        },
        "focused_tests": {
            "path": str(test),
            "relative_path": test_relative,
            "sha256": test_sha,
        },
        "review_marker": {
            "path": str(marker),
            "relative_path": marker_relative,
            "sha256": marker_sha,
        },
        "contracts": expected_contracts,
    }


def _git_output(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise QueueContractError("cannot inspect official git checkout") from error
    return result.stdout.strip()


def _validate_official_source(
    value: Mapping[str, object], method: str
) -> dict[str, object]:
    _exact_keys(value, {"root", "commit"}, method + " official source")
    root = _require_directory(
        _lexical_absolute(value.get("root"), method + " official source root"),
        method + " official source root",
    )
    expected = str(METHODS[method]["official_commit"])
    if value.get("commit") != expected:
        raise QueueContractError(method + " config does not name the exact official commit")
    observed = _git_output(root, "rev-parse", "HEAD")
    if observed != expected:
        raise QueueContractError(method + " official checkout commit differs")
    if _git_output(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise QueueContractError(method + " official checkout is dirty")
    return {"root": str(root), "commit": observed, "clean": True}


def _environment_probe(python: Path) -> dict[str, object]:
    """Compatibility probe source mirrored by the environment builder tests.

    Formal queue stages use the stronger full ``verify`` command below; this
    helper remains the single canonical lightweight probe definition shared by
    the builder and controller.
    """

    code = r'''import importlib, importlib.metadata as m, json, os, site, sys
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
}, sort_keys=True))'''
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    try:
        result = subprocess.run(
            (str(python), "-I", "-c", code),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            env=environment,
        )
        probe = json.loads(result.stdout)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as error:
        raise QueueContractError("isolated benchmark Python probe failed") from error
    if not isinstance(probe, dict):
        raise QueueContractError("isolated benchmark Python probe is malformed")
    return probe


def _torchvision_dynamic_admission_configuration(
    enabled: bool,
) -> dict[str, object]:
    if type(enabled) is not bool:  # noqa: E721
        raise QueueContractError("dynamic admission enable flag differs")
    return {
        "schema_version": ENVIRONMENT_TORCHVISION_DYNAMIC_ADMISSION_SCHEMA,
        "enabled": enabled,
        "authorities": dict(ENVIRONMENT_TORCHVISION_DYNAMIC_AUTHORITIES),
        "policy": dict(ENVIRONMENT_TORCHVISION_DYNAMIC_POLICY),
    }


def _successful_torchvision_dynamic_admission_evidence() -> dict[str, object]:
    return {
        "schema_version": ENVIRONMENT_TORCHVISION_DYNAMIC_ADMISSION_SCHEMA,
        "enabled": True,
        "authorities": dict(ENVIRONMENT_TORCHVISION_DYNAMIC_AUTHORITIES),
        "policy": dict(ENVIRONMENT_TORCHVISION_DYNAMIC_POLICY),
        "observed": {
            "append_caller_validated": True,
            "append_count": 1,
            "captured_compile_count": 0,
            "captured_directory_final_entries": [
                ENVIRONMENT_TORCHVISION_DYNAMIC_AUTHORITIES[
                    "generated_module_filename"
                ]
            ],
            "captured_directory_initially_empty": True,
            "captured_directory_revalidated_after_payload": True,
            "captured_exec_count": 0,
            "captured_finder_count": 0,
            "captured_temp_origin_module_count": 0,
            "generated_file_sha256": (
                ENVIRONMENT_TORCHVISION_DYNAMIC_AUTHORITIES[
                    "generated_module_sha256"
                ]
            ),
            "generated_file_size": ENVIRONMENT_TORCHVISION_DYNAMIC_AUTHORITIES[
                "generated_module_size"
            ],
            "preseed_compile_count": 1,
            "preseed_exec_count": 1,
            "pinned_source_compile_count": {
                module: 1 for module in ENVIRONMENT_TORCHVISION_PINNED_MODULES
            },
            "pinned_source_exec_count": {
                module: 1 for module in ENVIRONMENT_TORCHVISION_PINNED_MODULES
            },
            "pinned_source_find_count": {
                module: 1 for module in ENVIRONMENT_TORCHVISION_PINNED_MODULES
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
) -> dict[str, object]:
    evidence = _mapping(value, label)
    _exact_keys(
        evidence,
        {"schema_version", "enabled", "authorities", "policy", "observed"},
        label,
    )
    authorities = _mapping(evidence.get("authorities"), label + " authorities")
    policy = _mapping(evidence.get("policy"), label + " policy")
    observed = _mapping(evidence.get("observed"), label + " observed")
    _exact_keys(
        authorities,
        set(ENVIRONMENT_TORCHVISION_DYNAMIC_AUTHORITIES),
        label + " authorities",
    )
    _exact_keys(
        policy,
        set(ENVIRONMENT_TORCHVISION_DYNAMIC_POLICY),
        label + " policy",
    )
    expected = _successful_torchvision_dynamic_admission_evidence()
    expected_observed = expected["observed"]
    assert isinstance(expected_observed, dict)
    _exact_keys(observed, set(expected_observed), label + " observed")
    for field in (
        "pinned_source_compile_count",
        "pinned_source_exec_count",
        "pinned_source_find_count",
    ):
        counts = _mapping(observed.get(field), label + " " + field)
        _exact_keys(
            counts,
            set(ENVIRONMENT_TORCHVISION_PINNED_MODULES),
            label + " " + field,
        )
    if _canonical_bytes(evidence) != _canonical_bytes(expected):
        raise QueueContractError(label + " differs")
    return dict(evidence)


def _validate_runtime_no_site_evidence(
    value: object, environment_root: Path
) -> dict[str, object]:
    """Deep-check the builder's explicit-site receipt before opening any site."""

    evidence = _mapping(value, "environment runtime no-site evidence")
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
        "environment runtime no-site evidence",
    )
    _validate_content_sha(evidence, "environment runtime no-site evidence")
    expected_bootstrap = {
        "configuration_schema": ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SCHEMA,
        "launcher_source_sha256": (
            ENVIRONMENT_EXPLICIT_SITE_STDIN_LAUNCHER_SHA256
        ),
        "source_sha256": ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SHA256,
        "torchvision_dynamic_admission_source_sha256": (
            ENVIRONMENT_TORCHVISION_DYNAMIC_ADMISSION_SOURCE_SHA256
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
        "pip_runner_source_sha256": ENVIRONMENT_PIP_NO_SITE_RUNNER_SHA256,
        "payload_source_sha256": ENVIRONMENT_RUNTIME_PAYLOAD_SHA256,
        "configuration_content_sha256_bound": True,
        "preimport_sys_path_exact": True,
        "payload_import_sys_path_audited": True,
        "site_directories_fd_anchored": True,
        "site_main_and_pth_processing_blocked": True,
        "sitecustomize_and_usercustomize_never_imported": True,
    }
    if (
        evidence.get("schema_version") != ENVIRONMENT_RUNTIME_NO_SITE_SCHEMA
        or evidence.get("runtime_no_site") is not True
        or evidence.get("interpreter_flags") != ["-I", "-S", "-B"]
        or evidence.get("bootstrap") != expected_bootstrap
    ):
        raise QueueContractError("environment runtime no-site contract differs")
    expected_distributions = {
        "base": {"torch", "torchvision", "numpy", "Pillow", "pip"},
        "resolved_base": {"torch", "torchvision", "numpy", "Pillow", "pip"},
        "target": {
            "torch",
            "torchvision",
            "numpy",
            "Pillow",
            "pip",
            "torch-geometric",
            "opencv-python",
            "scipy",
        },
    }
    normalized: dict[str, object] = dict(evidence)
    for name in ("base", "resolved_base", "target"):
        context = _mapping(evidence.get(name), name + " runtime context")
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
        _validate_content_sha(context, name + " runtime context")
        _validate_torchvision_dynamic_admission_evidence(
            context.get("torchvision_dynamic_module_admission"),
            name + " torchvision dynamic module admission",
        )
        safe_identity = _mapping(
            context.get("safe_startup_identity"), name + " safe startup identity"
        )
        logical_identity = _mapping(
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
        identity_values = tuple(safe_identity.values()) + tuple(
            logical_identity.values()
        )
        no_site_sys_path = context.get("no_site_sys_path")
        if (
            context.get("runtime_no_site") is not True
            or any(
                not isinstance(item, str) or not Path(item).is_absolute()
                for item in identity_values
            )
            or logical_identity.get("base_prefix")
            != safe_identity.get("base_prefix")
            or not isinstance(no_site_sys_path, list)
            or not no_site_sys_path
            or any(
                not isinstance(path, str) or not Path(path).is_absolute()
                for path in no_site_sys_path
            )
            or len(no_site_sys_path) != len(set(no_site_sys_path))
        ):
            raise QueueContractError(name + " runtime identity/sys.path differs")
        injected = context.get("injected_site_directories")
        if not isinstance(injected, list) or not injected:
            raise QueueContractError(name + " injected-site evidence differs")
        injected_paths = []
        for index, binding_value in enumerate(injected):
            binding = _mapping(
                binding_value,
                "{} injected-site binding {}".format(name, index),
            )
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
                "{} injected-site binding {}".format(name, index),
            )
            canonical_path = binding.get("canonical_path")
            prefix = binding.get("prefix")
            relative_path = binding.get("relative_path")
            role = binding.get("role")
            if (
                not isinstance(canonical_path, str)
                or not isinstance(prefix, str)
                or not isinstance(relative_path, str)
                or not Path(canonical_path).is_absolute()
                or not Path(prefix).is_absolute()
                or _relative_path(
                    relative_path, name + " injected-site relative path"
                )
                != relative_path
                or Path(prefix) / relative_path != Path(canonical_path)
                or type(binding.get("device")) is not int  # noqa: E721
                or type(binding.get("inode")) is not int  # noqa: E721
                or not isinstance(role, str)
                or not role.startswith("audit_")
                or not role[6:].isdigit()
                or str(int(role[6:])) != role[6:]
                or not _is_sha256(binding.get("audit_content_sha256"))
            ):
                raise QueueContractError(name + " injected-site binding differs")
            directory = _require_directory(
                _lexical_absolute(canonical_path, name + " injected site"),
                name + " injected site",
            )
            metadata = directory.stat()
            if (
                int(metadata.st_dev) != binding["device"]
                or int(metadata.st_ino) != binding["inode"]
            ):
                raise QueueContractError(name + " injected-site identity differs")
            injected_paths.append(canonical_path)
        if (
            len(injected_paths) != len(set(injected_paths))
            or injected[0].get("prefix") != logical_identity.get("prefix")
        ):
            raise QueueContractError(name + " injected-site order differs")
        origins = context.get("import_origins")
        if not isinstance(origins, list) or not origins:
            raise QueueContractError(name + " import-origin evidence differs")
        observed_distributions = []
        for origin_value in origins:
            origin = _mapping(origin_value, name + " import-origin binding")
            _exact_keys(
                origin,
                {
                    "distribution",
                    "distribution_version",
                    "distribution_location",
                    "module",
                    "import_origin",
                },
                name + " import-origin binding",
            )
            if any(not isinstance(item, str) or not item for item in origin.values()):
                raise QueueContractError(name + " import-origin binding differs")
            if any(
                not Path(str(origin[field])).is_absolute()
                for field in ("distribution_location", "import_origin")
            ):
                raise QueueContractError(name + " import-origin path differs")
            observed_distributions.append(str(origin["distribution"]))
        if (
            observed_distributions != sorted(observed_distributions)
            or set(observed_distributions) != expected_distributions[name]
        ):
            raise QueueContractError(name + " import-origin set differs")
        pth_files = context.get("pth_files_present_but_never_executed")
        if not isinstance(pth_files, list):
            raise QueueContractError(name + " PTH evidence differs")
        pth_paths = []
        for pth_value in pth_files:
            pth = _mapping(pth_value, name + " PTH binding")
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
                name + " PTH binding",
            )
            executable_hashes = pth.get("executable_line_sha256s")
            declared_paths = pth.get("declared_paths")
            if (
                not isinstance(pth.get("path"), str)
                or not Path(str(pth["path"])).is_absolute()
                or not _is_sha256(pth.get("sha256"))
                or pth.get("execution_policy") != "present_but_never_executed"
                or type(pth.get("directive_count")) is not int  # noqa: E721
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
                raise QueueContractError(name + " PTH binding differs")
            pth_paths.append(str(pth["path"]))
        if pth_paths != sorted(pth_paths) or len(pth_paths) != len(set(pth_paths)):
            raise QueueContractError(name + " PTH binding order differs")
        normalized[name] = dict(context)
    target = normalized["target"]
    assert isinstance(target, dict)
    logical_target = target["logical_runtime_identity"]
    assert isinstance(logical_target, dict)
    if logical_target.get("prefix") != str(environment_root):
        raise QueueContractError("target no-site runtime prefix differs from environment")
    return normalized


def _environment_pip_evidence_argv_prefix(root: Path) -> list[str]:
    return [
        str(root / "bin/python"),
        "-I",
        "-S",
        "-B",
        "-c",
        "<explicit-site-stdin-launcher-sha256="
        + ENVIRONMENT_EXPLICIT_SITE_STDIN_LAUNCHER_SHA256
        + ">",
        "<content-sha256-bound-exact-no-site-sys-path-count-and-entries>",
        "<stdin-explicit-site-bootstrap-sha256="
        + ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SHA256
        + ">",
        "<content-bound-explicit-site-configuration>",
        "<pip-no-site-run-sha256="
        + ENVIRONMENT_PIP_NO_SITE_RUNNER_SHA256
        + ">",
    ]


def _expected_environment_download_argv(
    root: Path, package_urls: Sequence[str]
) -> list[str]:
    return [
        *_environment_pip_evidence_argv_prefix(root),
        "--isolated",
        "--disable-pip-version-check",
        "download",
        "--no-cache-dir",
        "--no-deps",
        "--only-binary=:all:",
        "--dest",
        str(root / ENVIRONMENT_WHEELHOUSE_RELATIVE_PATH),
        *package_urls,
    ]


def _expected_environment_install_argv(root: Path) -> list[str]:
    return [
        *_environment_pip_evidence_argv_prefix(root),
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
        str(root / ENVIRONMENT_WHEELHOUSE_RELATIVE_PATH),
        "--report",
        str(root / ENVIRONMENT_INSTALL_REPORT_RELATIVE_PATH),
        "-r",
        str(root / ENVIRONMENT_REQUIREMENTS_LOCK_RELATIVE_PATH),
    ]


def _validate_environment_report_path_normalization(
    value: object,
    root: Path,
    package_filenames: Sequence[str],
) -> dict[str, object]:
    normalization = _mapping(
        value, "environment receipt pip-report path normalization"
    )
    _exact_keys(
        normalization,
        {"schema_version", "policy", "entries", "content_sha256"},
        "environment receipt pip-report path normalization",
    )
    _validate_content_sha(
        normalization, "environment receipt pip-report path normalization"
    )
    entries = normalization.get("entries")
    filenames = set(package_filenames)
    if (
        normalization.get("schema_version")
        != "rachel-benchmark-pip-report-path-normalization/1.0"
        or normalization.get("policy")
        != "execution-anchor-file-url-to-canonical-lexical-wheel-path"
        or not isinstance(entries, list)
        or len(entries) != len(filenames)
    ):
        raise QueueContractError(
            "environment receipt pip-report path normalization differs"
        )
    observed = set()
    descriptor_pattern = re.compile(
        r"/(?:proc/(?:self|[0-9]+)|dev)/fd/[0-9]+/"
        + re.escape(ENVIRONMENT_WHEELHOUSE_RELATIVE_PATH)
        + r"/([^/]+)"
    )
    for index, entry_value in enumerate(entries):
        entry = _mapping(
            entry_value,
            "environment receipt pip-report normalization entry " + str(index),
        )
        _exact_keys(
            entry,
            {"filename", "source_url", "canonical_path", "canonical_url"},
            "environment receipt pip-report normalization entry " + str(index),
        )
        filename = entry.get("filename")
        if not isinstance(filename, str) or filename not in filenames:
            raise QueueContractError(
                "environment receipt pip-report normalization filename differs"
            )
        canonical_path = root / ENVIRONMENT_WHEELHOUSE_RELATIVE_PATH / filename
        canonical_url = canonical_path.as_uri()
        source_url = entry.get("source_url")
        source_path = None
        if isinstance(source_url, str):
            parsed = urlsplit(source_url)
            if (
                parsed.scheme == "file"
                and not parsed.netloc
                and not parsed.query
                and not parsed.fragment
            ):
                source_path = unquote(parsed.path)
        descriptor_match = (
            descriptor_pattern.fullmatch(source_path)
            if isinstance(source_path, str)
            else None
        )
        if (
            entry.get("canonical_path") != str(canonical_path)
            or entry.get("canonical_url") != canonical_url
            or (
                source_url != canonical_url
                and (
                    descriptor_match is None
                    or descriptor_match.group(1) != filename
                )
            )
            or filename in observed
        ):
            raise QueueContractError(
                "environment receipt pip-report normalization path differs"
            )
        observed.add(filename)
    if observed != filenames:
        raise QueueContractError(
            "environment receipt pip-report normalization package set differs"
        )
    return dict(normalization)


def _validate_environment_pip_evidence(
    build_evidence: Mapping[str, object],
    root: Path,
    reviewed_wheel_lock: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    packages = reviewed_wheel_lock.get("packages")
    if not isinstance(packages, list) or not packages:
        raise QueueContractError("environment reviewed wheel package set differs")
    package_urls = []
    package_filenames = []
    for index, package_value in enumerate(packages):
        package = _mapping(
            package_value, "environment reviewed wheel package " + str(index)
        )
        url = package.get("url")
        filename = package.get("filename")
        if (
            not isinstance(url, str)
            or not url
            or not isinstance(filename, str)
            or not filename
            or "/" in filename
            or "\\" in filename
        ):
            raise QueueContractError(
                "environment reviewed wheel package URL/filename differs"
            )
        package_urls.append(url)
        package_filenames.append(filename)
    if len(set(package_filenames)) != len(package_filenames):
        raise QueueContractError("environment reviewed wheel filenames differ")

    download = _mapping(
        build_evidence.get("download"), "environment receipt download evidence"
    )
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
        "environment receipt download evidence",
    )
    if (
        download.get("argv")
        != _expected_environment_download_argv(root, package_urls)
        or download.get("argv_policy") != ENVIRONMENT_PIP_ARGV_POLICY
        or download.get("requirements_lock")
        != str(root / ENVIRONMENT_REQUIREMENTS_LOCK_RELATIVE_PATH)
        or any(
            not _is_sha256(download.get(field))
            for field in (
                "stdout_sha256",
                "stderr_sha256",
                "requirements_lock_sha256",
            )
        )
    ):
        raise QueueContractError(
            "environment receipt download launcher/command evidence differs"
        )

    install = _mapping(
        build_evidence.get("install"), "environment receipt install evidence"
    )
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
        "environment receipt install evidence",
    )
    if (
        install.get("argv") != _expected_environment_install_argv(root)
        or install.get("argv_policy") != ENVIRONMENT_PIP_ARGV_POLICY
        or install.get("report")
        != str(root / ENVIRONMENT_INSTALL_REPORT_RELATIVE_PATH)
        or any(
            not _is_sha256(install.get(field))
            for field in (
                "stdout_sha256",
                "stderr_sha256",
                "report_sha256",
            )
        )
        or not isinstance(install.get("installed"), list)
    ):
        raise QueueContractError(
            "environment receipt install launcher/command evidence differs"
        )
    normalization = _validate_environment_report_path_normalization(
        install.get("report_path_normalization"), root, package_filenames
    )
    return dict(download), dict(install), normalization


def _validate_environment(value: Mapping[str, object]) -> dict[str, object]:
    _exact_keys(
        value,
        {
            "root",
            "python",
            "python_sha256",
            "receipt",
            "receipt_sha256",
            "receipt_content_sha256",
            "import_inventory_content_sha256",
            "builder_source_relative_path",
            "builder_source_sha256",
            "builder_test_relative_path",
            "builder_test_sha256",
            "reviewed_wheel_lock_relative_path",
            "reviewed_wheel_lock_sha256",
            "review_marker_relative_path",
            "review_marker_sha256",
        },
        "benchmark environment",
    )
    root = _require_directory(
        _lexical_absolute(value.get("root"), "benchmark environment root"),
        "benchmark environment root",
    )
    python = _require_regular_file(
        _lexical_absolute(value.get("python"), "benchmark Python"),
        "benchmark Python",
    )
    try:
        python.relative_to(root)
    except ValueError as error:
        raise QueueContractError("benchmark Python is not inside isolated env root") from error
    python_sha = _require_sha256(value.get("python_sha256"), "benchmark Python")
    if _sha256_file(python) != python_sha:
        raise QueueContractError("benchmark Python SHA-256 differs")
    root_metadata = root.stat()
    python_metadata = python.stat()
    receipt_path = _require_regular_file(
        _lexical_absolute(value.get("receipt"), "environment receipt"),
        "environment receipt",
    )
    receipt_sha = _require_sha256(
        value.get("receipt_sha256"), "environment receipt"
    )
    if _sha256_file(receipt_path) != receipt_sha:
        raise QueueContractError("environment receipt SHA-256 differs")
    receipt = _load_json(receipt_path, "environment receipt")
    _exact_keys(
        receipt,
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
        "environment receipt",
    )
    _validate_content_sha(receipt, "environment receipt")
    receipt_content_sha = _require_sha256(
        value.get("receipt_content_sha256"), "environment receipt content"
    )
    inventory_content_sha = _require_sha256(
        value.get("import_inventory_content_sha256"),
        "environment import inventory content",
    )
    builder_relative = _relative_path(
        value.get("builder_source_relative_path"), "environment builder source"
    )
    builder_sha = _require_sha256(
        value.get("builder_source_sha256"), "environment builder source"
    )
    builder_test_relative = _relative_path(
        value.get("builder_test_relative_path"), "environment builder tests"
    )
    builder_test_sha = _require_sha256(
        value.get("builder_test_sha256"), "environment builder tests"
    )
    wheel_lock_relative = _relative_path(
        value.get("reviewed_wheel_lock_relative_path"),
        "environment reviewed wheel lock",
    )
    wheel_lock_sha = _require_sha256(
        value.get("reviewed_wheel_lock_sha256"),
        "environment reviewed wheel lock",
    )
    review_marker_relative = _relative_path(
        value.get("review_marker_relative_path"), "environment review marker"
    )
    review_marker_sha = _require_sha256(
        value.get("review_marker_sha256"), "environment review marker"
    )
    if builder_relative != ENVIRONMENT_BUILDER_RELATIVE_PATH:
        raise QueueContractError("environment builder source path differs")
    if (
        builder_test_relative != ENVIRONMENT_BUILDER_TEST_RELATIVE_PATH
        or builder_test_sha != ENVIRONMENT_BUILDER_TEST_SHA256
    ):
        raise QueueContractError("environment builder-test authority differs")
    if (
        wheel_lock_relative != ENVIRONMENT_REVIEWED_WHEEL_LOCK_RELATIVE_PATH
        or wheel_lock_sha != ENVIRONMENT_REVIEWED_WHEEL_LOCK_SHA256
    ):
        raise QueueContractError("environment reviewed wheel-lock authority differs")
    build_evidence = receipt.get("build_evidence")
    if isinstance(build_evidence, dict):
        _exact_keys(
            build_evidence,
            {
                "download",
                "install",
                "wheels",
                "reviewed_wheel_lock",
                "target_launcher_identity",
                "inventory",
                "active_site",
                "bin_scripts",
                "runtime_no_site",
            },
            "environment receipt build evidence",
        )
    inventory_binding = (
        build_evidence.get("inventory") if isinstance(build_evidence, dict) else None
    )
    reviewed_wheel_lock = (
        build_evidence.get("reviewed_wheel_lock")
        if isinstance(build_evidence, dict)
        else None
    )
    bin_scripts = (
        build_evidence.get("bin_scripts") if isinstance(build_evidence, dict) else None
    )
    runtime_no_site = (
        _validate_runtime_no_site_evidence(build_evidence.get("runtime_no_site"), root)
        if isinstance(build_evidence, dict)
        else None
    )
    if isinstance(reviewed_wheel_lock, dict):
        _exact_keys(
            reviewed_wheel_lock,
            {
                "path",
                "file_sha256",
                "content_sha256",
                "packages",
                "target",
                "official_index",
                "review",
            },
            "environment receipt reviewed wheel lock",
        )
    download_evidence = None
    install_evidence = None
    report_path_normalization = None
    if isinstance(build_evidence, dict) and isinstance(reviewed_wheel_lock, dict):
        (
            download_evidence,
            install_evidence,
            report_path_normalization,
        ) = _validate_environment_pip_evidence(
            build_evidence, root, reviewed_wheel_lock
        )
    if isinstance(bin_scripts, dict):
        _exact_keys(
            bin_scripts,
            {
                "schema_version",
                "environment_root",
                "entries",
                "ephemeral_descriptor_paths_present",
                "content_sha256",
            },
            "environment receipt bin-script inventory",
        )
    receipt_builder = receipt.get("builder_source")
    if (
        receipt.get("schema_version") != ENVIRONMENT_SCHEMA
        or receipt.get("status") != "complete_isolated_benchmark_environment"
        or receipt.get("environment_root") != str(root)
        or receipt.get("python") != str(python)
        or receipt.get("python_sha256") != python_sha
        or receipt.get("isolated_from_primary_training_environment") is not True
        or receipt.get("runtime_no_site") is not True
        or receipt.get("sealed_test_or_real_accessed") is not False
        or receipt.get("content_sha256") != receipt_content_sha
        or receipt.get("isolation_model")
        != "fresh_venv_with_fd_anchored_explicit_sites_and_no_site_startup"
        or receipt.get("application_distributions_installed")
        != {
            "torch-geometric": "2.6.1",
            "opencv-python": "4.10.0.84",
            "scipy": "1.14.1",
        }
        or receipt.get("inherited_distributions_verified")
        != ["torch", "torchvision", "numpy", "Pillow"]
        or not isinstance(receipt.get("live_probe"), dict)
        or not isinstance(receipt.get("detailed_probe"), dict)
        or not isinstance(receipt.get("functional_cpu_probe"), dict)
        or not isinstance(receipt.get("pip_freeze"), dict)
        or not isinstance(inventory_binding, dict)
        or inventory_binding.get("content_sha256") != inventory_content_sha
        or not _is_sha256(inventory_binding.get("file_sha256"))
        or not isinstance(reviewed_wheel_lock, dict)
        or reviewed_wheel_lock.get("file_sha256") != wheel_lock_sha
        or not _is_sha256(reviewed_wheel_lock.get("content_sha256"))
        or not isinstance(reviewed_wheel_lock.get("packages"), list)
        or not isinstance(download_evidence, dict)
        or not isinstance(install_evidence, dict)
        or not isinstance(report_path_normalization, dict)
        or report_path_normalization.get("schema_version")
        != "rachel-benchmark-pip-report-path-normalization/1.0"
        or report_path_normalization.get("policy")
        != "execution-anchor-file-url-to-canonical-lexical-wheel-path"
        or not isinstance(report_path_normalization.get("entries"), list)
        or len(report_path_normalization["entries"])
        != len(reviewed_wheel_lock["packages"])
        or not isinstance(bin_scripts, dict)
        or bin_scripts.get("schema_version")
        != "rachel-benchmark-bin-script-inventory/1.0"
        or bin_scripts.get("environment_root") != str(root)
        or not isinstance(bin_scripts.get("entries"), list)
        or bin_scripts.get("ephemeral_descriptor_paths_present") is not False
        or not _is_sha256(bin_scripts.get("content_sha256"))
        or not isinstance(receipt_builder, dict)
        or receipt_builder.get("sha256") != builder_sha
        or not isinstance(runtime_no_site, dict)
    ):
        raise QueueContractError("environment receipt identity/scope differs")
    return {
        "root": str(root),
        "python": str(python),
        "python_sha256": python_sha,
        "python_relative_path": python.relative_to(root).as_posix(),
        "root_device": int(root_metadata.st_dev),
        "root_inode": int(root_metadata.st_ino),
        "python_device": int(python_metadata.st_dev),
        "python_inode": int(python_metadata.st_ino),
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "receipt_content_sha256": receipt_content_sha,
        "import_inventory_content_sha256": inventory_content_sha,
        "builder_source_relative_path": builder_relative,
        "builder_source_sha256": builder_sha,
        "builder_source": dict(receipt_builder),
        "builder_test_relative_path": builder_test_relative,
        "builder_test_sha256": builder_test_sha,
        "reviewed_wheel_lock_relative_path": wheel_lock_relative,
        "reviewed_wheel_lock_sha256": wheel_lock_sha,
        "reviewed_wheel_lock": dict(reviewed_wheel_lock),
        "review_marker_relative_path": review_marker_relative,
        "review_marker_sha256": review_marker_sha,
        "runtime_no_site": runtime_no_site,
    }


def _validate_environment_source_authority(
    source_root: Path, environment: Mapping[str, object]
) -> dict[str, object]:
    builder, builder_relative, builder_sha = _source_file(
        source_root,
        environment.get("builder_source_relative_path"),
        environment.get("builder_source_sha256"),
        "environment builder source",
    )
    tests, tests_relative, tests_sha = _source_file(
        source_root,
        environment.get("builder_test_relative_path"),
        environment.get("builder_test_sha256"),
        "environment builder focused tests",
    )
    wheel_lock, wheel_relative, wheel_sha = _source_file(
        source_root,
        environment.get("reviewed_wheel_lock_relative_path"),
        environment.get("reviewed_wheel_lock_sha256"),
        "environment reviewed wheel lock",
    )
    marker, marker_relative, marker_sha = _source_file(
        source_root,
        environment.get("review_marker_relative_path"),
        environment.get("review_marker_sha256"),
        "environment review marker",
    )
    if (
        builder_relative != ENVIRONMENT_BUILDER_RELATIVE_PATH
        or builder_sha != ENVIRONMENT_BUILDER_SHA256
        or tests_relative != ENVIRONMENT_BUILDER_TEST_RELATIVE_PATH
        or tests_sha != ENVIRONMENT_BUILDER_TEST_SHA256
        or wheel_relative != ENVIRONMENT_REVIEWED_WHEEL_LOCK_RELATIVE_PATH
        or wheel_sha != ENVIRONMENT_REVIEWED_WHEEL_LOCK_SHA256
        or marker_relative != ENVIRONMENT_REVIEW_RELATIVE_PATH
        or marker_sha != ENVIRONMENT_REVIEW_SHA256
    ):
        raise QueueContractError("environment reviewed source authority differs")
    decision = _load_json(marker, "environment review marker")
    _exact_keys(
        decision,
        {
            "schema_version",
            "status",
            "p0_count",
            "p1_count",
            "review_scope_complete",
            "builder",
            "focused_tests",
            "reviewed_wheel_lock",
            "contracts",
            "verification",
            "sealed_test_accessed",
            "real_data_accessed",
        },
        "environment review marker",
    )
    expected_contracts = {
        "bin_script_inventory_and_no_ephemeral_paths_reviewed": True,
        "content_bound_stdin_launcher_and_preimport_sys_path_reviewed": True,
        "dual_site_and_prefix_audit_reviewed": True,
        "environment_receipt_published_last_no_success_on_failure": True,
        "executable_pth_present_but_never_executed_reviewed": True,
        "fd_anchored_subprocess_execution_reviewed": True,
        "lib64_alias_inode_deduplication_reviewed": True,
        "linux_proc_fd_magic_link_binding_reviewed": True,
        "pip_report_path_normalization_reviewed": True,
        "runtime_no_site_explicit_fd_site_injection_reviewed": True,
        "scipy_installed_into_local_environment": True,
        "torchvision_dynamic_module_admission_reviewed": True,
        "pinned_torchvision_source_pyc_never_read_reviewed": True,
        "venv_without_pip_and_resolved_base_pip_reviewed": True,
    }
    expected_verification = {
        "fd_anchor_cross_tests_passed": 2,
        "focused_python38_tests_passed": 61,
        "py_compile_clean": True,
        "ruff_clean": True,
    }
    if (
        decision.get("schema_version") != ENVIRONMENT_REVIEW_SCHEMA
        or decision.get("status") != "GO"
        or decision.get("p0_count") != 0
        or decision.get("p1_count") != 0
        or decision.get("review_scope_complete") is not True
        or decision.get("builder")
        != {"relative_path": builder_relative, "sha256": builder_sha}
        or decision.get("focused_tests")
        != {"relative_path": tests_relative, "sha256": tests_sha}
        or decision.get("reviewed_wheel_lock")
        != {"relative_path": wheel_relative, "sha256": wheel_sha}
        or decision.get("contracts") != expected_contracts
        or decision.get("verification") != expected_verification
        or decision.get("sealed_test_accessed") is not False
        or decision.get("real_data_accessed") is not False
    ):
        raise QueueContractError("environment review marker is not an exact GO")
    wheel_authority = _load_json(wheel_lock, "environment reviewed wheel lock")
    _exact_keys(
        wheel_authority,
        {
            "schema_version",
            "status",
            "content_sha256",
            "official_index",
            "packages",
            "review",
            "target",
        },
        "environment reviewed wheel lock",
    )
    _validate_content_sha(wheel_authority, "environment reviewed wheel lock")
    expected_receipt_wheel = {
        "path": str(wheel_lock),
        "file_sha256": wheel_sha,
        "content_sha256": wheel_authority.get("content_sha256"),
        "packages": wheel_authority.get("packages"),
        "target": wheel_authority.get("target"),
        "official_index": wheel_authority.get("official_index"),
        "review": wheel_authority.get("review"),
    }
    receipt_builder = environment.get("builder_source")
    receipt_wheel = environment.get("reviewed_wheel_lock")
    if (
        receipt_builder != {"path": str(builder), "sha256": builder_sha}
        or receipt_wheel != expected_receipt_wheel
    ):
        raise QueueContractError(
            "environment receipt does not consume the reviewed source authority"
        )
    return {
        "builder": {
            "path": str(builder),
            "relative_path": builder_relative,
            "sha256": builder_sha,
        },
        "focused_tests": {
            "path": str(tests),
            "relative_path": tests_relative,
            "sha256": tests_sha,
        },
        "reviewed_wheel_lock": {
            "path": str(wheel_lock),
            "relative_path": wheel_relative,
            "sha256": wheel_sha,
            "content_sha256": receipt_wheel["content_sha256"],
        },
        "review_marker": {
            "path": str(marker),
            "relative_path": marker_relative,
            "sha256": marker_sha,
        },
    }


def _validate_review_method(
    source_root: Path, method: str, value: Mapping[str, object]
) -> dict[str, object]:
    expected_keys = {
        "method_id",
        "runner_relative_path",
        "runner_sha256",
        "adaptation_contract_relative_path",
        "adaptation_contract_sha256",
        "review_marker_relative_path",
        "review_marker_sha256",
    }
    if method == "pairingnet":
        expected_keys |= {"smoke_source_relative_path", "smoke_source_sha256"}
    _exact_keys(value, expected_keys, method + " method config")
    specification = METHODS[method]
    if value.get("method_id") != specification["method_id"]:
        raise QueueContractError(method + " method ID differs")
    for field in (
        "runner_relative_path",
        "adaptation_contract_relative_path",
    ):
        if value.get(field) != specification[field]:
            raise QueueContractError(method + " " + field + " differs")
    runner, runner_relative, runner_sha = _source_file(
        source_root,
        value.get("runner_relative_path"),
        value.get("runner_sha256"),
        method + " runner",
    )
    contract, contract_relative, contract_sha = _source_file(
        source_root,
        value.get("adaptation_contract_relative_path"),
        value.get("adaptation_contract_sha256"),
        method + " adaptation contract",
    )
    smoke: Optional[Path] = None
    smoke_relative: Optional[str] = None
    smoke_sha: Optional[str] = None
    if method == "pairingnet":
        if value.get("smoke_source_relative_path") != specification[
            "smoke_source_relative_path"
        ]:
            raise QueueContractError("pairingnet smoke source path differs")
        smoke, smoke_relative, smoke_sha = _source_file(
            source_root,
            value.get("smoke_source_relative_path"),
            value.get("smoke_source_sha256"),
            "pairingnet smoke source",
        )

    marker, marker_relative, marker_sha = _source_file(
        source_root,
        value.get("review_marker_relative_path"),
        value.get("review_marker_sha256"),
        method + " review marker",
    )
    decision = _load_json(marker, method + " review marker")
    if (
        decision.get("schema_version") != REVIEW_SCHEMA
        or decision.get("status") != "GO"
        or decision.get("method_id") != specification["method_id"]
        or decision.get("p0_count") != 0
        or decision.get("p1_count") != 0
        or decision.get("review_scope_complete") is not True
        or decision.get("sealed_synthetic_accessed") is not False
        or decision.get("real_data_accessed") is not False
    ):
        raise QueueContractError(method + " review marker is not an unconditional GO")
    reviewed_runner = decision.get("runner")
    reviewed_contract = decision.get("adaptation_contract")
    if reviewed_runner != {"relative_path": runner_relative, "sha256": runner_sha}:
        raise QueueContractError(method + " review does not bind the exact runner")
    if reviewed_contract != {
        "relative_path": contract_relative,
        "sha256": contract_sha,
    }:
        raise QueueContractError(
            method + " review does not bind the exact adaptation contract"
        )
    if method == "pairingnet" and decision.get("smoke_source") != {
        "relative_path": smoke_relative,
        "sha256": smoke_sha,
    }:
        raise QueueContractError("pairingnet review does not bind the smoke source")
    return {
        "method_id": specification["method_id"],
        "runner": {
            "path": str(runner),
            "relative_path": runner_relative,
            "sha256": runner_sha,
        },
        "adaptation_contract": {
            "path": str(contract),
            "relative_path": contract_relative,
            "sha256": contract_sha,
        },
        "smoke_source": (
            {
                "path": str(smoke),
                "relative_path": smoke_relative,
                "sha256": smoke_sha,
            }
            if smoke is not None
            else None
        ),
        "review_marker": {
            "path": str(marker),
            "relative_path": marker_relative,
            "sha256": marker_sha,
        },
    }


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


def _validate_upstream_config(value: Mapping[str, object], label: str) -> dict[str, str]:
    _exact_keys(value, {"root", "receipt_glob", "expected_schema"}, label)
    root = _require_directory(
        _lexical_absolute(value.get("root"), label + " root"), label + " root"
    )
    receipt_glob = value.get("receipt_glob")
    if receipt_glob != "run-*/run_receipt.json":
        raise QueueContractError(label + " receipt glob is not the frozen one-level form")
    expected_schema = value.get("expected_schema")
    if not isinstance(expected_schema, str) or not expected_schema:
        raise QueueContractError(label + " expected schema is missing")
    return {
        "root": str(root),
        "receipt_glob": str(receipt_glob),
        "expected_schema": expected_schema,
    }


def validate_config(config: Mapping[str, object]) -> dict[str, Any]:
    source_root = _require_directory(
        _lexical_absolute(config.get("immutable_source_root"), "immutable source root"),
        "immutable source root",
    )
    expected_controller = source_root / CONTROLLER_RELATIVE_PATH
    current_controller = Path(__file__).resolve(strict=True)
    if current_controller != expected_controller.resolve(strict=True):
        raise QueueContractError(
            "controller must execute directly from the configured immutable source root"
        )
    inventory = _source_inventory(source_root)
    if CONTROLLER_RELATIVE_PATH not in {str(row["path"]) for row in inventory}:
        raise QueueContractError("controller is absent from immutable source inventory")
    controller = _validate_controller_authority(
        source_root,
        _mapping(config.get("controller"), "controller authority"),
        inventory,
    )
    train_val_asset_freeze_authority = _validate_train_val_asset_authority(
        source_root,
        _mapping(
            config.get("train_val_asset_freeze_authority"),
            "train/val asset-freeze authority",
        ),
    )
    dataset_root = _require_directory(
        _lexical_absolute(config.get("dataset_root"), "Rachel dataset root"),
        "Rachel dataset root",
    )
    output_root = _lexical_absolute(config.get("output_root"), "benchmark output root")
    _assert_no_symlink_components(output_root, "benchmark output root")
    if os.path.lexists(output_root):
        raise QueueContractError("benchmark output root must be fresh")
    output_parent = _require_directory(output_root.parent, "benchmark output parent")

    environment = _validate_environment(
        _mapping(config.get("benchmark_environment"), "benchmark environment")
    )
    environment_source_authority = _validate_environment_source_authority(
        source_root, environment
    )
    upstream_values = _mapping(config.get("upstreams"), "upstreams")
    _exact_keys(upstream_values, {"n512", "matched_mm"}, "upstreams")
    upstreams = {
        key: _validate_upstream_config(
            _mapping(upstream_values.get(key), key + " upstream"), key + " upstream"
        )
        for key in ("n512", "matched_mm")
    }

    official_values = _mapping(config.get("official_sources"), "official sources")
    _exact_keys(official_values, set(METHODS), "official sources")
    official = {
        method: _validate_official_source(
            _mapping(official_values.get(method), method + " official source"), method
        )
        for method in METHODS
    }

    method_values = _mapping(config.get("methods"), "methods")
    _exact_keys(method_values, set(METHODS), "methods")
    methods = {
        method: _validate_review_method(
            source_root,
            method,
            _mapping(method_values.get(method), method + " method config"),
        )
        for method in METHODS
    }

    allowed_source_members = set(REQUIRED_SOURCE_RELATIVE_PATHS)
    allowed_source_members.add(
        str(controller["review_marker"]["relative_path"])
    )
    allowed_source_members.add(
        str(train_val_asset_freeze_authority["review_marker"]["relative_path"])
    )
    allowed_source_members.add(ENVIRONMENT_REVIEWED_WHEEL_LOCK_RELATIVE_PATH)
    allowed_source_members.add(
        str(environment_source_authority["review_marker"]["relative_path"])
    )
    for method in METHODS:
        allowed_source_members.add(
            str(methods[method]["adaptation_contract"]["relative_path"])
        )
        allowed_source_members.add(
            str(methods[method]["review_marker"]["relative_path"])
        )
    complete_source_inventory = _complete_source_inventory(source_root)
    observed_source_members = {
        str(row["path"]) for row in complete_source_inventory
    }
    if observed_source_members != allowed_source_members:
        missing = sorted(allowed_source_members - observed_source_members)
        extra = sorted(observed_source_members - allowed_source_members)
        raise QueueContractError(
            "minimal immutable source member whitelist differs "
            "(missing={}, extra={})".format(missing, extra)
        )
    source_permissions = _validate_immutable_source_permissions(
        source_root, complete_source_inventory
    )

    runtime = _mapping(config.get("runtime"), "runtime")
    _exact_keys(runtime, {"device", "pairingnet", "shreddingnet"}, "runtime")
    device = runtime.get("device")
    if not isinstance(device, str) or not (
        device == "cuda"
        or (
            device.startswith("cuda:")
            and device[5:].isdigit()
            and str(int(device[5:])) == device[5:]
        )
    ):
        raise QueueContractError("formal benchmark device must be CUDA")
    pairing_runtime = _mapping(runtime.get("pairingnet"), "pairingnet runtime")
    _exact_keys(pairing_runtime, {"precision", "num_workers"}, "pairingnet runtime")
    if pairing_runtime.get("precision") not in {"fp32", "bf16"}:
        raise QueueContractError("pairingnet precision differs")
    if (
        type(pairing_runtime.get("num_workers")) is not int
        or int(pairing_runtime["num_workers"]) < 0
        or int(pairing_runtime["num_workers"]) > 64
    ):
        raise QueueContractError("pairingnet num_workers must be in [0,64]")
    shredding_runtime = _mapping(runtime.get("shreddingnet"), "shreddingnet runtime")
    _exact_keys(
        shredding_runtime,
        {
            "num_workers",
            "coarse_microbatch",
            "matching_microbatch",
            "classify_microbatch",
            "amp",
        },
        "shreddingnet runtime",
    )
    for field in (
        "num_workers",
        "coarse_microbatch",
        "matching_microbatch",
        "classify_microbatch",
    ):
        value = shredding_runtime.get(field)
        minimum = 0 if field == "num_workers" else 1
        maximum = 64 if field == "num_workers" else SHREDDINGNET_EFFECTIVE_BATCH[
            field[: -len("_microbatch")]
        ]
        if (  # noqa: E721
            type(value) is not int
            or int(value) < minimum
            or int(value) > maximum
        ):
            raise QueueContractError("shreddingnet " + field + " is invalid")
    if type(shredding_runtime.get("amp")) is not bool:  # noqa: E721
        raise QueueContractError("shreddingnet amp must be boolean")

    wait = _mapping(config.get("wait"), "wait")
    _exact_keys(wait, {"poll_seconds", "timeout_seconds"}, "wait")
    poll_seconds = wait.get("poll_seconds")
    timeout_seconds = wait.get("timeout_seconds")
    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, (int, float))
        or not math.isfinite(float(poll_seconds))
        or not 0.1 <= float(poll_seconds) <= 60.0
    ):
        raise QueueContractError("wait poll_seconds must be in [0.1, 60]")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) < 0.0
    ):
        raise QueueContractError("wait timeout_seconds must be non-negative")

    protected = [
        source_root,
        dataset_root,
        Path(environment["root"]),
        *(Path(value["root"]) for value in upstreams.values()),
        *(Path(value["root"]) for value in official.values()),
    ]
    canonical_output = (output_parent / output_root.name).resolve(strict=False)
    for path in protected:
        if _paths_overlap(canonical_output, path.resolve(strict=True)):
            raise QueueContractError(
                "benchmark output overlaps a protected input namespace"
            )
    environment_root = Path(environment["root"])
    for path in (source_root, dataset_root, *(Path(v["root"]) for v in official.values())):
        if _paths_overlap(environment_root, path):
            raise QueueContractError("isolated benchmark environment overlaps an input")

    if methods["pairingnet"]["smoke_source"]["sha256"] != _sha256_file(
        source_root / PAIRINGNET_SMOKE_RELATIVE_PATH
    ):
        raise QueueContractError("queue/method reviews disagree on PairingNet smoke")
    return {
        "config_path": config["_config_path"],
        "config_file_sha256": config["_config_file_sha256"],
        "source_root": str(source_root),
        "controller": controller,
        "train_val_asset_freeze_authority": train_val_asset_freeze_authority,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "environment": environment,
        "environment_source_authority": environment_source_authority,
        "upstreams": upstreams,
        "official_sources": official,
        "methods": methods,
        "runtime": {
            "device": device,
            "pairingnet": dict(pairing_runtime),
            "shreddingnet": dict(shredding_runtime),
        },
        "wait": {
            "poll_seconds": float(poll_seconds),
            "timeout_seconds": float(timeout_seconds),
        },
        "source_inventory": inventory,
        "source_permissions": source_permissions,
        "complete_source_inventory": complete_source_inventory,
        "complete_source_inventory_sha256": _source_inventory_sha256(
            complete_source_inventory
        ),
    }


def _validate_upstream_receipt(path: Path, key: str, schema: str) -> dict[str, object]:
    receipt = _load_json(path, key + " upstream receipt")
    if (
        receipt.get("schema_version") != schema
        or receipt.get("status") != "complete_train_validation_only"
        or receipt.get("test_accessed") is not False
        or receipt.get("real_external_test_accessed") is not False
    ):
        raise QueueContractError(key + " upstream completion/scope differs")
    if key == "n512":
        arms = receipt.get("arm_results")
        if (
            not isinstance(arms, list)
            or not arms
            or any(
                not isinstance(row, dict)
                or row.get("stop_reason") != "validation_early_stop"
                or row.get("convergence_claim")
                != "validation_plateau_under_declared_rule"
                for row in arms
            )
        ):
            raise QueueContractError("N512 upstream did not demonstrate validation plateau")
    else:
        if (
            receipt.get("stop_reason") != "validation_early_stop"
            or receipt.get("convergence_claim")
            != "validation_plateau_under_declared_rule"
        ):
            raise QueueContractError(
                "matched-MM upstream did not demonstrate validation plateau"
            )
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256_file(path),
        "schema_version": schema,
        "status": receipt["status"],
        "convergence_demonstrated": True,
        "sealed_test_or_real_accessed": False,
    }


def _discover_upstream(value: Mapping[str, str], key: str) -> Optional[dict[str, object]]:
    root = _require_directory(Path(value["root"]), key + " upstream root")
    candidates: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        metadata = child.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise QueueContractError(key + " upstream root contains a symlink")
        if child.name.startswith("run-") and stat.S_ISDIR(metadata.st_mode):
            receipt = child / "run_receipt.json"
            if not os.path.lexists(receipt):
                raise QueueContractError(key + " finalized run lacks a receipt")
            candidates.append(_require_regular_file(receipt, key + " upstream receipt"))
        elif child.name.startswith(".partial-") and stat.S_ISDIR(metadata.st_mode):
            failure = child / "failure.json"
            if os.path.lexists(failure):
                raise QueueContractError(key + " upstream has a terminal failure receipt")
    if not candidates:
        return None
    if len(candidates) != 1:
        raise QueueContractError(key + " upstream has ambiguous finalized runs")
    return _validate_upstream_receipt(candidates[0], key, value["expected_schema"])


def _current_train_val_manifest_binding(dataset_root: Path) -> dict[str, str]:
    """Hash exactly the two authorized manifests, never test or real data."""

    return {
        split: _sha256_file(
            _require_regular_file(
                dataset_root / "pairs" / (split + ".jsonl"),
                "Rachel " + split + " manifest",
            )
        )
        for split in ("train", "val")
    }


def _train_manifest_labels(dataset_root: Path) -> dict[str, bool]:
    path = _require_regular_file(
        dataset_root / "pairs/train.jsonl", "Rachel train manifest"
    )
    labels: dict[str, bool] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                pair_id = row.get("pair_id") if isinstance(row, dict) else None
                label = row.get("label") if isinstance(row, dict) else None
                if (
                    not isinstance(pair_id, str)
                    or not pair_id
                    or type(label) is not bool  # noqa: E721
                    or pair_id in labels
                ):
                    raise QueueContractError(
                        "Rachel train manifest pair/label differs at line "
                        + str(line_number)
                    )
                labels[pair_id] = label
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QueueContractError("Rachel train manifest cannot bind smoke order") from error
    if len(labels) != 24_000:
        raise QueueContractError("Rachel train manifest row count differs")
    return labels


class BenchmarkQueue:
    def __init__(self, authority: Mapping[str, Any]):
        self.authority = dict(authority)
        self.output_root = Path(authority["output_root"])
        self.control_root = self.output_root / "control"
        self.log_root = self.control_root / "logs"
        self.terminal_receipt = self.control_root / "terminal_receipt.json"
        self.stage = "validated_prewrite_authority"
        self.stage_history: list[dict[str, object]] = []
        self.upstream_receipts: dict[str, object] = {}
        self.upstream_strict_verification: Optional[dict[str, object]] = None
        self.environment_verification: Optional[dict[str, object]] = None
        self.source_dataset_binding: Optional[dict[str, str]] = None
        self.benchmark_dataset_root: Optional[Path] = None
        self.dataset_binding: Optional[dict[str, str]] = None
        self.asset_freeze_verification: Optional[dict[str, object]] = None
        self.pairingnet_smoke_receipt_sha256: Optional[str] = None
        self.pairingnet_smoke_content_sha256: Optional[str] = None
        self.shreddingnet_smoke_receipt_sha256: Optional[str] = None
        self.shreddingnet_smoke_content_sha256: Optional[str] = None
        self.training_started = {"pairingnet": False, "shreddingnet": False}
        self.active_child: Optional[subprocess.Popen[bytes]] = None
        self.active_process_group: Optional[int] = None
        self.failure_receipt_enabled = False
        self.terminal_written = False
        self._handling_signal = False
        self._signal_notification_error: Optional[str] = None

    def _install_signals(self) -> None:
        for signum in TERMINATION_SIGNALS:
            signal.signal(signum, self._signal_handler)

    def _signal_handler(self, signum: int, _frame: object) -> None:
        if self._handling_signal:
            os._exit(128 + int(signum))
        self._handling_signal = True
        self.stage = self.stage + ":signal_" + signal.Signals(signum).name
        # Do not call Popen.poll()/wait() here.  A signal can interrupt
        # Popen.wait() while its internal waitpid lock is held, so re-entering
        # those methods from the Python signal handler can make a dead child
        # appear alive until timeout.  The main QueueSignal handler performs
        # the synchronous reap after this async-safe group notification.
        process_group = self.active_process_group
        if process_group is not None:
            try:
                os.killpg(process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as error:
                self._signal_notification_error = repr(error)
        raise QueueSignal(signum)

    @staticmethod
    def _process_group_exists(process_group: int) -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        return True

    def _terminate_active_child(self) -> None:
        child = self.active_child
        process_group = self.active_process_group
        if child is None:
            self.active_child = None
            self.active_process_group = None
            return
        if process_group is None or process_group != child.pid:
            raise QueueContractError("active child process-group identity is missing")
        if self._process_group_exists(process_group):
            try:
                os.killpg(process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5.0
        while self._process_group_exists(process_group) and time.monotonic() < deadline:
            child.poll()
            time.sleep(0.05)
        if self._process_group_exists(process_group):
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            child.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            raise QueueContractError("active child process group could not be reaped")
        self.active_child = None
        self.active_process_group = None

    def _acquire_output(self) -> None:
        self.stage = "acquire_fresh_output"
        _assert_no_symlink_components(self.output_root, "benchmark output root")
        try:
            os.mkdir(self.output_root, 0o700)
        except FileExistsError as error:
            raise QueueContractError("fresh output namespace appeared before acquire") from error
        try:
            os.mkdir(self.control_root, 0o700)
            self.failure_receipt_enabled = True
            self._install_signals()
            os.mkdir(self.log_root, 0o700)
        except FileExistsError as error:
            raise QueueContractError("fresh control namespace appeared before acquire") from error
        _write_json_new(
            self.control_root / "launch_authority.json",
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "validated_before_upstream_wait_or_benchmark_process",
                "config_path": self.authority["config_path"],
                "config_file_sha256": self.authority["config_file_sha256"],
                "controller": self.authority["controller"],
                "train_val_asset_freeze_authority": self.authority[
                    "train_val_asset_freeze_authority"
                ],
                "source_root": self.authority["source_root"],
                "complete_source_inventory_sha256": self.authority[
                    "complete_source_inventory_sha256"
                ],
                "complete_source_file_count": len(
                    self.authority["complete_source_inventory"]
                ),
                "source_permissions": self.authority["source_permissions"],
                "dataset_root": self.authority["dataset_root"],
                "environment": self.authority["environment"],
                "environment_source_authority": self.authority[
                    "environment_source_authority"
                ],
                "official_sources": self.authority["official_sources"],
                "methods": self.authority["methods"],
                "runtime": self.authority["runtime"],
                "scope": {
                    "train_val_only": True,
                    "resume_supported": False,
                    "sealed_test_corrosion_real_commands_present": False,
                    "formal_training_started": False,
                },
                "threat_model_limitations": list(THREAT_MODEL_LIMITATIONS),
            },
        )
        _write_json_new(
            self.control_root / "source_code_freeze.json",
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "immutable_source_inventoried_before_wait_or_benchmark",
                "source_root": self.authority["source_root"],
                "scope": SOURCE_INVENTORY_SCOPE,
                "executable_suffixes": sorted(SOURCE_EXECUTABLE_SUFFIXES),
                "forbidden_suffixes": sorted(SOURCE_FORBIDDEN_SUFFIXES),
                "site_or_user_customization_forbidden": True,
                "source_inventory_sha256": self.authority["controller"][
                    "source_inventory_sha256"
                ],
                "complete_source_inventory_sha256": self.authority[
                    "complete_source_inventory_sha256"
                ],
                "source_permissions": self.authority["source_permissions"],
                "train_val_asset_freeze_authority": self.authority[
                    "train_val_asset_freeze_authority"
                ],
                "environment_source_authority": self.authority[
                    "environment_source_authority"
                ],
                "files": self.authority["source_inventory"],
                "complete_files": self.authority["complete_source_inventory"],
                "sealed_test_or_real_opened": False,
            },
        )

    def _verify_source_and_reviews(self) -> None:
        root = _require_directory(Path(self.authority["source_root"]), "source root")
        inventory = _source_inventory(root)
        if inventory != self.authority["source_inventory"]:
            raise QueueContractError("immutable source inventory changed")
        complete_inventory = _complete_source_inventory(root)
        if complete_inventory != self.authority["complete_source_inventory"]:
            raise QueueContractError("complete immutable source inventory changed")
        permissions = _validate_immutable_source_permissions(root, complete_inventory)
        if permissions != self.authority["source_permissions"]:
            raise QueueContractError("immutable source permission authority changed")
        config = _load_json(Path(self.authority["config_path"]), "queue config")
        if _sha256_file(Path(self.authority["config_path"])) != self.authority[
            "config_file_sha256"
        ]:
            raise QueueContractError("queue config changed")
        observed_controller = _validate_controller_authority(
            root,
            _mapping(config.get("controller"), "controller authority"),
            inventory,
        )
        if observed_controller != self.authority["controller"]:
            raise QueueContractError("controller review/source authority changed")
        observed_assets = _validate_train_val_asset_authority(
            root,
            _mapping(
                config.get("train_val_asset_freeze_authority"),
                "train/val asset-freeze authority",
            ),
        )
        if observed_assets != self.authority["train_val_asset_freeze_authority"]:
            raise QueueContractError(
                "train/val asset-freeze review/source authority changed"
            )
        observed_environment_source = _validate_environment_source_authority(
            root, self.authority["environment"]
        )
        if observed_environment_source != self.authority[
            "environment_source_authority"
        ]:
            raise QueueContractError("environment review/source authority changed")
        method_values = _mapping(config.get("methods"), "methods")
        for method in METHODS:
            observed = _validate_review_method(
                root,
                method,
                _mapping(method_values.get(method), method + " method config"),
            )
            if observed != self.authority["methods"][method]:
                raise QueueContractError(method + " review/source authority changed")

    def _verify_environment_and_official(self) -> None:
        config = _load_json(Path(self.authority["config_path"]), "queue config")
        environment = _validate_environment(
            _mapping(config.get("benchmark_environment"), "benchmark environment")
        )
        if environment != self.authority["environment"]:
            raise QueueContractError("isolated benchmark environment changed")
        official_values = _mapping(config.get("official_sources"), "official sources")
        for method in METHODS:
            observed = _validate_official_source(
                _mapping(official_values.get(method), method + " official source"), method
            )
            if observed != self.authority["official_sources"][method]:
                raise QueueContractError(method + " official checkout changed")

    def _verify_dataset_binding(self) -> None:
        if self.source_dataset_binding is not None:
            observed_source = _current_train_val_manifest_binding(
                Path(self.authority["dataset_root"])
            )
            if observed_source != self.source_dataset_binding:
                raise QueueContractError(
                    "original Rachel train/val source manifests changed"
                )
        if self.dataset_binding is not None:
            if self.benchmark_dataset_root is None:
                raise QueueContractError("frozen runner dataset root is missing")
            observed_frozen = _current_train_val_manifest_binding(
                self.benchmark_dataset_root
            )
            if observed_frozen != self.dataset_binding:
                raise QueueContractError(
                    "frozen runner train/val manifests changed between methods"
                )

    def _runner_dataset_root(self) -> str:
        if self.benchmark_dataset_root is None or self.dataset_binding is None:
            raise QueueContractError(
                "self-contained frozen runner dataset was not established"
            )
        return str(self.benchmark_dataset_root)

    def _stage_stdout_json(self, stage: str, label: str) -> dict[str, object]:
        path = _require_regular_file(
            self.log_root / (stage + ".stdout.log"), label + " stdout"
        )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise QueueContractError(label + " did not emit one JSON object") from error
        if not isinstance(value, dict):
            raise QueueContractError(label + " stdout is not one JSON object")
        return value

    def _validate_asset_freeze_verification(
        self,
        value: Mapping[str, object],
        *,
        receipt_sha256: str,
        semantic_sha256: str,
    ) -> dict[str, object]:
        _exact_keys(
            value,
            {
                "schema_version",
                "status",
                "dataset_root",
                "freeze_dir",
                "frozen_dataset_root",
                "freeze_receipt_sha256",
                "freeze_receipt_size_bytes",
                "inventory_file_sha256",
                "inventory_file_size_bytes",
                "semantic_inventory_content_sha256",
                "source_asset_file_count",
                "source_asset_total_size_bytes",
                "frozen_asset_file_count",
                "frozen_asset_total_size_bytes",
                "manifest_content_sha256",
                "ordered_pair_ids_sha256",
                "canonical_ordered_pairs_sha256",
                "freeze_root_device",
                "freeze_root_inode",
                "view_root_device",
                "view_root_inode",
                "rehash_all_frozen_dataset_bytes",
                "source_snapshot_reverified",
                "runner_dataset_root_is_self_contained_frozen_view",
                "same_freeze_root_fd_held_for_full_verification",
                "final_complete_tree_revalidated",
                "frozen_view_regular_file_link_count_required",
                "sealed_test_manifest_opened",
                "sealed_test_assets_opened",
                "real_data_opened",
                "rgb_opened",
            },
            "train/val asset-freeze verification",
        )
        expected_frozen_root = str(
            self.control_root / "train_val_asset_freeze" / "frozen_data"
        )
        manifests = value.get("manifest_content_sha256")
        ordered = value.get("ordered_pair_ids_sha256")
        canonical_ordered = value.get("canonical_ordered_pairs_sha256")
        if (
            value.get("schema_version")
            != "rachel-train-val-semantic-asset-verification/3.0"
            or value.get("status")
            != "verified_train_val_semantic_assets_unchanged"
            or value.get("dataset_root") != self.authority["dataset_root"]
            or value.get("freeze_dir")
            != str(self.control_root / "train_val_asset_freeze")
            or value.get("frozen_dataset_root") != expected_frozen_root
            or value.get("freeze_receipt_sha256") != receipt_sha256
            or value.get("semantic_inventory_content_sha256") != semantic_sha256
            or not _is_sha256(value.get("inventory_file_sha256"))
            or type(value.get("freeze_receipt_size_bytes")) is not int  # noqa: E721
            or int(value["freeze_receipt_size_bytes"]) <= 0
            or type(value.get("inventory_file_size_bytes")) is not int  # noqa: E721
            or int(value["inventory_file_size_bytes"]) <= 0
            or type(value.get("source_asset_file_count")) is not int  # noqa: E721
            or int(value["source_asset_file_count"]) <= 0
            or type(value.get("source_asset_total_size_bytes")) is not int  # noqa: E721
            or int(value["source_asset_total_size_bytes"]) <= 0
            or type(value.get("frozen_asset_file_count")) is not int  # noqa: E721
            or int(value["frozen_asset_file_count"]) <= 0
            or type(value.get("frozen_asset_total_size_bytes")) is not int  # noqa: E721
            or int(value["frozen_asset_total_size_bytes"]) <= 0
            or not isinstance(manifests, dict)
            or set(manifests) != {"train", "val"}
            or any(not _is_sha256(manifests.get(split)) for split in ("train", "val"))
            or not isinstance(ordered, dict)
            or set(ordered) != {"train", "val"}
            or any(not _is_sha256(ordered.get(split)) for split in ("train", "val"))
            or not isinstance(canonical_ordered, dict)
            or set(canonical_ordered) != {"train", "val"}
            or any(
                not _is_sha256(canonical_ordered.get(split))
                for split in ("train", "val")
            )
            or type(value.get("freeze_root_device")) is not int  # noqa: E721
            or int(value["freeze_root_device"]) < 0
            or type(value.get("freeze_root_inode")) is not int  # noqa: E721
            or int(value["freeze_root_inode"]) <= 0
            or type(value.get("view_root_device")) is not int  # noqa: E721
            or int(value["view_root_device"]) < 0
            or type(value.get("view_root_inode")) is not int  # noqa: E721
            or int(value["view_root_inode"]) <= 0
            or value.get("rehash_all_frozen_dataset_bytes") is not True
            or value.get("source_snapshot_reverified") is not True
            or value.get("runner_dataset_root_is_self_contained_frozen_view")
            is not True
            or value.get("same_freeze_root_fd_held_for_full_verification")
            is not True
            or value.get("final_complete_tree_revalidated") is not True
            or type(value.get("frozen_view_regular_file_link_count_required"))
            is not int  # noqa: E721
            or value.get("frozen_view_regular_file_link_count_required") != 1
            or value.get("sealed_test_manifest_opened") is not False
            or value.get("sealed_test_assets_opened") is not False
            or value.get("real_data_opened") is not False
            or value.get("rgb_opened") is not False
        ):
            raise QueueContractError("train/val asset-freeze verification differs")
        return dict(value)

    def _create_train_val_asset_freeze(self) -> None:
        freeze_dir = self.control_root / "train_val_asset_freeze"
        create_stage = "train_val_asset_freeze_create"
        self._run_child(
            create_stage,
            (
                "-m",
                TRAIN_VAL_ASSET_FREEZE_MODULE,
                "create",
                "--dataset-root",
                self.authority["dataset_root"],
                "--output-dir",
                str(freeze_dir),
            ),
        )
        receipt_path = _require_regular_file(
            freeze_dir / "train_val_semantic_asset_freeze_receipt.json",
            "train/val semantic asset-freeze receipt",
        )
        receipt = _load_json(receipt_path, "train/val semantic asset-freeze receipt")
        if (
            set(receipt)
            != {
                "schema_version",
                "status",
                "dataset_root",
                "frozen_dataset",
                "inventory",
                "publication_contract",
                "scope",
            }
            or receipt.get("schema_version")
            != "rachel-train-val-semantic-asset-freeze/3.0"
            or receipt.get("status")
            != "complete_self_contained_train_val_semantic_asset_freeze"
            or receipt.get("dataset_root") != self.authority["dataset_root"]
            or receipt.get("scope")
            != {
                "opened_pair_manifests": ["pairs/train.jsonl", "pairs/val.jsonl"],
                "frozen_runner_dataset_root": str(freeze_dir / "frozen_data"),
                "sealed_test_manifest_opened": False,
                "sealed_test_assets_opened": False,
                "real_data_opened": False,
                "rgb_opened": False,
            }
        ):
            raise QueueContractError("train/val semantic asset-freeze receipt differs")
        inventory_binding = receipt.get("inventory")
        frozen_dataset = receipt.get("frozen_dataset")
        if (
            not isinstance(inventory_binding, dict)
            or set(inventory_binding)
            != {
                "filename",
                "sha256",
                "size_bytes",
                "semantic_inventory_content_sha256",
            }
            or not isinstance(frozen_dataset, dict)
            or set(frozen_dataset)
            != {
                "relative_path",
                "absolute_path",
                "runner_dataset_root_required",
                "freeze_root_device",
                "freeze_root_inode",
                "view_root_device",
                "view_root_inode",
                "manifest_content_sha256",
                "ordered_pair_ids_sha256",
                "canonical_ordered_pairs_sha256",
                "file_count",
                "total_size_bytes",
                "file_mode_octal",
                "directory_mode_octal",
            }
        ):
            raise QueueContractError("train/val asset inventory binding is missing")
        if receipt.get("publication_contract") != {
            "fresh_directory_required": True,
            "all_output_operations_anchored_by_parent_and_output_dir_fds": True,
            "atomic_file_publication": (
                "same_parent_dir_fd_temp_fsync_then_hard_link_no_replace"
            ),
            "source_assets_materialized_by_independent_byte_copy": True,
            "source_hardlinks_used": False,
            "frozen_view_read_only": True,
            "overwrite_allowed": False,
            "receipt_published_last_as_commit_marker": True,
        }:
            raise QueueContractError("train/val asset publication contract differs")
        semantic_sha = _require_sha256(
            inventory_binding.get("semantic_inventory_content_sha256"),
            "train/val semantic asset inventory",
        )
        inventory_path = _require_regular_file(
            freeze_dir / str(inventory_binding.get("filename")),
            "train/val semantic asset inventory",
        )
        if (
            inventory_binding.get("filename")
            != "train_val_semantic_asset_inventory.json"
            or _sha256_file(inventory_path) != inventory_binding.get("sha256")
            or inventory_path.stat().st_size != inventory_binding.get("size_bytes")
        ):
            raise QueueContractError("train/val semantic inventory file differs")
        inventory = _load_json(inventory_path, "train/val semantic asset inventory")
        if set(inventory) != {
            "semantic_inventory",
            "semantic_inventory_content_sha256",
        } or inventory.get("semantic_inventory_content_sha256") != semantic_sha:
            raise QueueContractError("train/val semantic inventory envelope differs")
        semantic = inventory.get("semantic_inventory")
        if (
            not isinstance(semantic, dict)
            or hashlib.sha256(_canonical_bytes(semantic)).hexdigest() != semantic_sha
        ):
            raise QueueContractError("train/val semantic inventory content differs")
        source_manifests = semantic.get("manifest_content_sha256")
        if (
            not isinstance(source_manifests, dict)
            or set(source_manifests) != {"train", "val"}
            or source_manifests
            != _current_train_val_manifest_binding(
                Path(self.authority["dataset_root"])
            )
        ):
            raise QueueContractError("original train/val manifest authority differs")
        receipt_sha = _sha256_file(receipt_path)
        verification = self._verify_train_val_asset_freeze(
            "initial_before_pairingnet", receipt_sha, semantic_sha
        )
        self.asset_freeze_verification = verification
        if (
            frozen_dataset.get("relative_path") != "frozen_data"
            or frozen_dataset.get("absolute_path")
            != verification["frozen_dataset_root"]
            or frozen_dataset.get("runner_dataset_root_required") is not True
            or any(
                type(frozen_dataset.get(field)) is not int  # noqa: E721
                for field in (
                    "freeze_root_device",
                    "freeze_root_inode",
                    "view_root_device",
                    "view_root_inode",
                )
            )
            or frozen_dataset.get("freeze_root_device")
            != verification["freeze_root_device"]
            or frozen_dataset.get("freeze_root_inode")
            != verification["freeze_root_inode"]
            or frozen_dataset.get("view_root_device")
            != verification["view_root_device"]
            or frozen_dataset.get("view_root_inode")
            != verification["view_root_inode"]
            or frozen_dataset.get("manifest_content_sha256")
            != verification["manifest_content_sha256"]
            or frozen_dataset.get("ordered_pair_ids_sha256")
            != verification["ordered_pair_ids_sha256"]
            or frozen_dataset.get("canonical_ordered_pairs_sha256")
            != verification["canonical_ordered_pairs_sha256"]
            or frozen_dataset.get("file_count")
            != verification["frozen_asset_file_count"]
            or frozen_dataset.get("total_size_bytes")
            != verification["frozen_asset_total_size_bytes"]
            or frozen_dataset.get("file_mode_octal") != "0444"
            or frozen_dataset.get("directory_mode_octal") != "0555"
        ):
            raise QueueContractError("self-contained frozen runner view differs")
        self.source_dataset_binding = dict(source_manifests)
        self.benchmark_dataset_root = _require_directory(
            Path(str(verification["frozen_dataset_root"])),
            "self-contained frozen runner dataset",
        )
        self.dataset_binding = dict(verification["manifest_content_sha256"])
        self._verify_dataset_binding()
        _write_json_new(
            self.control_root / "train_val_asset_freeze_gate.json",
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "complete_train_val_semantic_assets_frozen_before_benchmarks",
                "freeze_dir": str(freeze_dir.relative_to(self.output_root)),
                "freeze_receipt_sha256": receipt_sha,
                "semantic_inventory_content_sha256": semantic_sha,
                "original_dataset_root": self.authority["dataset_root"],
                "frozen_runner_dataset_root": self._runner_dataset_root(),
                "source_asset_file_count": verification["source_asset_file_count"],
                "source_asset_total_size_bytes": verification[
                    "source_asset_total_size_bytes"
                ],
                "frozen_asset_file_count": verification["frozen_asset_file_count"],
                "frozen_asset_total_size_bytes": verification[
                    "frozen_asset_total_size_bytes"
                ],
                "source_manifest_content_sha256": self.source_dataset_binding,
                "frozen_manifest_content_sha256": verification[
                    "manifest_content_sha256"
                ],
                "ordered_pair_ids_sha256": verification[
                    "ordered_pair_ids_sha256"
                ],
                "source_assets_reverified": True,
                "runner_dataset_root_is_self_contained_frozen_view": True,
                "all_referenced_train_val_assets_rehashed": True,
                "sealed_test_or_real_opened": False,
            },
        )

    def _verify_train_val_asset_freeze(
        self,
        label: str,
        receipt_sha256: str,
        semantic_sha256: str,
        *,
        raw_spawn: bool = False,
    ) -> dict[str, object]:
        stage = "train_val_asset_freeze_verify_" + label
        execute = self._spawn_child if raw_spawn else self._run_child
        execute(
            stage,
            (
                "-m",
                TRAIN_VAL_ASSET_FREEZE_MODULE,
                "verify",
                "--dataset-root",
                self.authority["dataset_root"],
                "--freeze-dir",
                str(self.control_root / "train_val_asset_freeze"),
                "--expected-freeze-receipt-sha256",
                receipt_sha256,
                "--expected-semantic-inventory-content-sha256",
                semantic_sha256,
            ),
        )
        value = self._stage_stdout_json(stage, "train/val asset-freeze verifier")
        return self._validate_asset_freeze_verification(
            value,
            receipt_sha256=receipt_sha256,
            semantic_sha256=semantic_sha256,
        )

    def _reverify_train_val_assets(
        self, label: str, *, raw_spawn: bool = False
    ) -> None:
        baseline = self.asset_freeze_verification
        if not isinstance(baseline, dict):
            raise QueueContractError("train/val asset freeze was not established")
        observed = self._verify_train_val_asset_freeze(
            label,
            str(baseline["freeze_receipt_sha256"]),
            str(baseline["semantic_inventory_content_sha256"]),
            raw_spawn=raw_spawn,
        )
        if observed != baseline:
            raise QueueContractError("train/val semantic asset authority changed")
        _write_json_new(
            self.control_root / ("train_val_asset_reverification_" + label + ".json"),
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "train_val_semantic_assets_reverified_unchanged",
                "checkpoint": label,
                "verification": observed,
                "sealed_test_or_real_opened": False,
            },
        )

    def _validate_upstream_strict_verification(
        self, value: Mapping[str, object]
    ) -> dict[str, object]:
        _exact_keys(
            value,
            {
                "schema_version",
                "status",
                "n512",
                "matched_mm",
                "canonical_config_authority",
                "matched_training_hash_evidence",
                "dataset_root",
                "formal_convergence_verified",
                "winner_checkpoints_strict_loaded_cpu",
                "test_or_real_opened",
            },
            "strict upstream verification",
        )
        n512 = value.get("n512")
        matched = value.get("matched_mm")
        if not isinstance(n512, dict) or not isinstance(matched, dict):
            raise QueueContractError("strict upstream method records are missing")
        _exact_keys(
            n512,
            {
                "run_root",
                "receipt_sha256",
                "schema_version",
                "status",
                "fingerprint_sha256",
                "config",
                "population",
                "winners",
            },
            "strict N512 verification",
        )
        _exact_keys(
            matched,
            {
                "run_root",
                "receipt_sha256",
                "schema_version",
                "status",
                "fingerprint_sha256",
                "config",
                "population",
                "stop_reason",
                "convergence_claim",
                "winners",
            },
            "strict matched-MM verification",
        )
        expected_roots = {
            key: str(Path(self.upstream_receipts[key]["path"]).parent)
            for key in ("n512", "matched_mm")
        }
        n_winners = n512.get("winners")
        m_winners = matched.get("winners")
        if (
            value.get("schema_version")
            != "rachel-benchmark-queue-upstream-verification/1.0"
            or value.get("status")
            != "verified_formal_upstream_winners_and_alignment"
            or value.get("dataset_root") != self.authority["dataset_root"]
            or value.get("formal_convergence_verified") is not True
            or value.get("winner_checkpoints_strict_loaded_cpu") is not True
            or value.get("test_or_real_opened") is not False
            or n512.get("run_root") != expected_roots["n512"]
            or matched.get("run_root") != expected_roots["matched_mm"]
            or n512.get("receipt_sha256")
            != self.upstream_receipts["n512"]["sha256"]
            or matched.get("receipt_sha256")
            != self.upstream_receipts["matched_mm"]["sha256"]
            or n512.get("schema_version")
            != self.authority["upstreams"]["n512"]["expected_schema"]
            or matched.get("schema_version")
            != self.authority["upstreams"]["matched_mm"]["expected_schema"]
            or n512.get("status") != "complete_train_validation_only"
            or matched.get("status") != "complete_train_validation_only"
            or not _is_sha256(n512.get("fingerprint_sha256"))
            or not _is_sha256(matched.get("fingerprint_sha256"))
            or matched.get("stop_reason") != "validation_early_stop"
            or matched.get("convergence_claim")
            != "validation_plateau_under_declared_rule"
            or not isinstance(n512.get("config"), dict)
            or not isinstance(n512.get("population"), dict)
            or not isinstance(matched.get("config"), dict)
            or not isinstance(matched.get("population"), dict)
            or not isinstance(n_winners, dict)
            or set(n_winners) != {"coarse_only", "full_n512"}
            or not isinstance(m_winners, dict)
            or set(m_winners)
            != {"matched_mm_converged", "matched_mm_same_exposure_epoch5"}
        ):
            raise QueueContractError("strict upstream identity/convergence differs")
        for label, winners, require_plateau in (
            ("N512", n_winners, True),
            ("matched-MM", m_winners, False),
        ):
            for row in winners.values():
                expected = {
                    "epoch",
                    "checkpoint_sha256",
                    "threshold_content_sha256",
                }
                if require_plateau:
                    expected |= {"stop_reason", "convergence_claim"}
                if not isinstance(row, dict):
                    raise QueueContractError(label + " strict winner is malformed")
                _exact_keys(row, expected, label + " strict winner")
                if (
                    type(row.get("epoch")) is not int  # noqa: E721
                    or int(row["epoch"]) <= 0
                    or not _is_sha256(row.get("checkpoint_sha256"))
                    or not _is_sha256(row.get("threshold_content_sha256"))
                    or (
                        require_plateau
                        and (
                            row.get("stop_reason") != "validation_early_stop"
                            or row.get("convergence_claim")
                            != "validation_plateau_under_declared_rule"
                        )
                    )
                ):
                    raise QueueContractError(label + " strict winner differs")
        config_authority = value.get("canonical_config_authority")
        evidence = value.get("matched_training_hash_evidence")
        if (
            not isinstance(config_authority, dict)
            or set(config_authority)
            != {
                "comparison",
                "model_config",
                "model_config_sha256",
                "loss_config",
                "loss_config_sha256",
                "model_config_sha256_by_arm",
                "loss_config_sha256_by_arm",
                "all_winner_configs_exactly_equal",
            }
            or config_authority.get("comparison")
            != "complete_dataclass_field_mapping_exact_equality"
            or config_authority.get("all_winner_configs_exactly_equal") is not True
            or not _is_sha256(config_authority.get("model_config_sha256"))
            or not _is_sha256(config_authority.get("loss_config_sha256"))
            or not isinstance(config_authority.get("model_config"), dict)
            or not config_authority.get("model_config")
            or not isinstance(config_authority.get("loss_config"), dict)
            or not config_authority.get("loss_config")
            or hashlib.sha256(
                _canonical_bytes(config_authority["model_config"])
            ).hexdigest()
            != config_authority.get("model_config_sha256")
            or hashlib.sha256(
                _canonical_bytes(config_authority["loss_config"])
            ).hexdigest()
            != config_authority.get("loss_config_sha256")
            or config_authority.get("model_config_sha256_by_arm")
            != {
                arm: config_authority["model_config_sha256"]
                for arm in ("coarse_only", "full_n512")
            }
            or config_authority.get("loss_config_sha256_by_arm")
            != {
                arm: config_authority["loss_config_sha256"]
                for arm in ("coarse_only", "full_n512")
            }
            or not isinstance(evidence, dict)
            or set(evidence)
            != {
                "claim_level",
                "canonical_dataset_root",
                "canonical_seed",
                "train_manifest",
                "validation_manifest",
                "training_fingerprints_recomputed",
                "validation_pair_order",
                "limitations",
            }
            or evidence.get("claim_level")
            != (
                "same_frozen_train_val_manifest_content_and_exact_validation_pair_order;"
                "per_epoch_training_presentation_order_not_provable"
            )
            or evidence.get("canonical_dataset_root")
            != self.authority["dataset_root"]
            or evidence.get("canonical_seed") != n512["config"].get("seed")
        ):
            raise QueueContractError("strict upstream config/hash authority differs")
        train_evidence = evidence.get("train_manifest")
        val_evidence = evidence.get("validation_manifest")
        fingerprints = evidence.get("training_fingerprints_recomputed")
        order_evidence = evidence.get("validation_pair_order")
        limitations = evidence.get("limitations")
        current = _current_train_val_manifest_binding(
            Path(self.authority["dataset_root"])
        )
        if (
            not isinstance(train_evidence, dict)
            or set(train_evidence)
            != {
                "path",
                "content_sha256",
                "bound_to_both_training_fingerprints",
                "pair_count",
                "pair_order_fingerprint_sha256",
            }
            or not isinstance(val_evidence, dict)
            or set(val_evidence)
            != {
                "path",
                "content_sha256",
                "bound_to_both_training_fingerprints",
                "pair_count",
                "pair_order_fingerprint_sha256",
                "exact_order_equal_to_all_four_validation_score_artifacts",
            }
            or train_evidence.get("path")
            != str(Path(self.authority["dataset_root"]) / "pairs/train.jsonl")
            or val_evidence.get("path")
            != str(Path(self.authority["dataset_root"]) / "pairs/val.jsonl")
            or train_evidence.get("content_sha256") != current["train"]
            or val_evidence.get("content_sha256") != current["val"]
            or not _is_sha256(train_evidence.get("pair_order_fingerprint_sha256"))
            or not _is_sha256(val_evidence.get("pair_order_fingerprint_sha256"))
            or train_evidence.get("pair_count") != 24_000
            or val_evidence.get("pair_count") != 3_000
            or train_evidence.get("bound_to_both_training_fingerprints") is not True
            or val_evidence.get("bound_to_both_training_fingerprints") is not True
            or val_evidence.get(
                "exact_order_equal_to_all_four_validation_score_artifacts"
            )
            is not True
            or not isinstance(fingerprints, dict)
            or fingerprints
            != {
                "rachel_n512": n512["fingerprint_sha256"],
                "matched_mm": matched["fingerprint_sha256"],
            }
            or not isinstance(order_evidence, dict)
            or set(order_evidence)
            != {
                "exact_order_equal_across_four_frozen_thresholds",
                "pair_count",
                "pair_order_fingerprint_sha256",
                "sources",
            }
            or order_evidence.get(
                "exact_order_equal_across_four_frozen_thresholds"
            )
            is not True
            or order_evidence.get("pair_count") != 3_000
            or order_evidence.get("pair_order_fingerprint_sha256")
            != val_evidence.get("pair_order_fingerprint_sha256")
            or not isinstance(order_evidence.get("sources"), dict)
            or set(order_evidence["sources"])
            != {
                "coarse_only",
                "full_n512",
                "matched_mm_converged",
                "matched_mm_same_exposure_epoch5",
            }
            or not all(
                isinstance(row, dict) and row
                for row in order_evidence["sources"].values()
            )
            or limitations
            != {
                "exact_per_epoch_training_pair_presentation_order_saved": False,
                "exact_per_epoch_training_pair_presentation_order_claimed_equal": False,
            }
        ):
            raise QueueContractError("strict upstream manifest/order evidence differs")
        nconfig = n512["config"]
        npopulation = n512["population"]
        mconfig = matched["config"]
        mpopulation = matched["population"]
        if (
            nconfig.get("seed") != 260831
            or nconfig.get("epochs") != 128
            or nconfig.get("batch_size") != 16
            or not isinstance(nconfig.get("continuation"), dict)
            or nconfig["continuation"].get("min_total_epochs") != 20
            or nconfig["continuation"].get("patience") != 12
            or nconfig["continuation"].get("early_stop_reads") != ["val"]
            or nconfig["continuation"].get("checkpoint_selection_reads")
            != ["val"]
            or npopulation.get("train_total") != 24_000
            or npopulation.get("val_total") != 3_000
            or npopulation.get("train_rows_per_epoch") != 24_000
            or npopulation.get("val_rows") != 3_000
            or mconfig.get("seed") != nconfig.get("seed")
            or mconfig.get("batch_size") != 16
            or mconfig.get("max_total_epochs") != 128
            or mconfig.get("min_total_epochs") != 20
            or mconfig.get("patience") != 12
            or mconfig.get("source_splits_opened") != ["train", "val"]
            or mconfig.get("test_accessed") is not False
            or mconfig.get("real_external_test_accessed") is not False
            or mpopulation
            != {
                "train_total": 24_000,
                "val_total": 3_000,
                "train_positive": 12_000,
                "val_positive": 1_500,
            }
        ):
            raise QueueContractError("strict upstream config/population differs")
        return dict(value)

    def _strict_verify_upstreams(self, label: str) -> None:
        rediscovered = {
            key: _discover_upstream(self.authority["upstreams"][key], key)
            for key in ("n512", "matched_mm")
        }
        if rediscovered != self.upstream_receipts:
            raise QueueContractError("upstream finalized-run authority changed")
        stage = "strict_upstream_verify_" + label
        self._run_child(
            stage,
            (
                "-m",
                STRICT_VERIFIER_MODULE,
                "upstreams",
                "--n512-run",
                str(Path(self.upstream_receipts["n512"]["path"]).parent),
                "--matched-run",
                str(Path(self.upstream_receipts["matched_mm"]["path"]).parent),
                "--dataset-root",
                self.authority["dataset_root"],
            ),
        )
        observed = self._validate_upstream_strict_verification(
            self._stage_stdout_json(stage, "strict upstream verifier")
        )
        if self.upstream_strict_verification is None:
            self.upstream_strict_verification = observed
        elif observed != self.upstream_strict_verification:
            raise QueueContractError("strict upstream authority changed")
        _write_json_new(
            self.control_root / ("strict_upstream_verification_" + label + ".json"),
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "strict_upstream_authority_verified",
                "checkpoint": label,
                "verification": observed,
                "verifier_stdout_sha256": _sha256_file(
                    self.log_root / (stage + ".stdout.log")
                ),
                "sealed_test_or_real_opened": False,
            },
        )

    def _wait_for_upstreams(self) -> None:
        self.stage = "waiting_for_n512_and_matched_mm_plateau_receipts"
        started = time.monotonic()
        while True:
            observed = {
                key: _discover_upstream(self.authority["upstreams"][key], key)
                for key in ("n512", "matched_mm")
            }
            if all(value is not None for value in observed.values()):
                self.upstream_receipts = observed
                break
            timeout = self.authority["wait"]["timeout_seconds"]
            if timeout and time.monotonic() - started >= timeout:
                raise QueueContractError("timed out waiting for both upstream plateaus")
            time.sleep(self.authority["wait"]["poll_seconds"])
        self._verify_source_and_reviews()
        self._verify_environment_and_official()
        self._strict_verify_upstreams("initial_before_benchmarks")
        _write_json_new(
            self.control_root / "upstream_plateau_gate.json",
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "both_upstream_validation_plateaus_complete",
                "upstreams": self.upstream_receipts,
                "strict_verification": self.upstream_strict_verification,
                "benchmark_training_started": dict(self.training_started),
                "sealed_test_or_real_opened": False,
            },
        )

    def _child_environment(self, anchored_environment_root: str) -> dict[str, str]:
        environment_root = Path(anchored_environment_root)
        return {
            "PATH": os.pathsep.join(
                (str(environment_root / "bin"), "/usr/local/bin", "/usr/bin", "/bin")
            ),
            "HOME": str(environment_root),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }

    def _spawn_child(self, stage: str, arguments: Sequence[str]) -> None:
        self.stage = stage
        self._verify_source_and_reviews()
        self._verify_environment_and_official()
        self._verify_dataset_binding()
        if len(arguments) < 2 or arguments[0] != "-m":
            raise QueueContractError("child command is not one reviewed Python module")
        module = str(arguments[1])
        if module not in {
            str(METHODS["pairingnet"]["runner_module"]),
            str(METHODS["pairingnet"]["smoke_module"]),
            str(METHODS["shreddingnet"]["runner_module"]),
            TRAIN_VAL_ASSET_FREEZE_MODULE,
            STRICT_VERIFIER_MODULE,
            ENVIRONMENT_BUILDER_MODULE,
        }:
            raise QueueContractError("child module is outside the reviewed command set")
        interpreter_flags = ["-I", "-S", "-B", "-u", "-c"]
        stdout_path = self.log_root / (stage + ".stdout.log")
        stderr_path = self.log_root / (stage + ".stderr.log")
        if os.path.lexists(stdout_path) or os.path.lexists(stderr_path):
            raise QueueContractError("stage log path already exists")
        runtime_sources = _reviewed_environment_runtime_sources(
            Path(
                str(
                    self.authority["environment_source_authority"]["builder"][
                        "path"
                    ]
                )
            )
        )
        started = time.time()
        environment_descriptor, anchored_root, anchored_python = (
            _open_anchored_environment_python(self.authority["environment"])
        )
        try:
            held_sites, site_descriptors = _open_anchored_environment_sites(
                self.authority["environment"]
            )
            try:
                source_descriptor, source_binding = _open_anchored_source_binding(
                    Path(str(self.authority["source_root"])),
                    str(self.authority["controller"]["source_inventory_sha256"]),
                )
                try:
                    site_configuration = _explicit_site_child_configuration(
                        self.authority["environment"], held_sites, source_binding
                    )
                    no_site_path = site_configuration["no_site_sys_path"]
                    assert isinstance(no_site_path, list)
                    command = [
                        anchored_python,
                        *interpreter_flags,
                        runtime_sources["launcher"],
                        str(len(no_site_path)),
                        *no_site_path,
                        hashlib.sha256(_canonical_bytes(no_site_path)).hexdigest(),
                        ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SHA256,
                        json.dumps(
                            site_configuration,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                        ISOLATED_MODULE_BOOTSTRAP,
                        self.authority["source_root"],
                        module,
                        *arguments[2:],
                    ]
                    with stdout_path.open("xb") as stdout, stderr_path.open(
                        "xb"
                    ) as stderr:
                        previous_mask = signal.pthread_sigmask(
                            signal.SIG_BLOCK, TERMINATION_SIGNALS
                        )
                        try:
                            child = subprocess.Popen(
                                command,
                                cwd=self.control_root,
                                env=self._child_environment(anchored_root),
                                stdin=subprocess.PIPE,
                                stdout=stdout,
                                stderr=stderr,
                                start_new_session=True,
                                preexec_fn=_child_unblock_termination_signals,
                                pass_fds=(
                                    environment_descriptor,
                                    *site_descriptors,
                                    source_descriptor,
                                ),
                            )
                            process_group = os.getpgid(child.pid)
                            if process_group != child.pid:
                                try:
                                    child.kill()
                                finally:
                                    child.wait()
                                raise QueueContractError(
                                    "reviewed child did not become its own "
                                    "process-group leader"
                                )
                            self.active_child = child
                            self.active_process_group = process_group
                            if stage == "pairingnet_formal_train":
                                self.training_started["pairingnet"] = True
                            elif stage == "shreddingnet_formal_train":
                                self.training_started["shreddingnet"] = True
                        finally:
                            signal.pthread_sigmask(
                                signal.SIG_SETMASK, previous_mask
                            )
                        try:
                            if child.stdin is None:
                                raise QueueContractError(
                                    "reviewed child stdin pipe is unavailable"
                                )
                            child.stdin.write(runtime_sources["bootstrap"].encode())
                            child.stdin.close()
                            return_code = child.wait()
                        except BaseException:
                            if child.poll() is None:
                                os.killpg(process_group, signal.SIGKILL)
                            child.wait()
                            raise
                        finally:
                            if not self._handling_signal:
                                self.active_child = None
                                self.active_process_group = None
                finally:
                    _close_and_revalidate_source_binding(
                        source_descriptor, source_binding
                    )
            finally:
                _close_and_revalidate_environment_sites(
                    site_descriptors, held_sites
                )
        finally:
            _close_and_revalidate_environment_anchor(
                environment_descriptor, self.authority["environment"]
            )
        row = {
            "stage": stage,
            "started_unix_time": started,
            "finished_unix_time": time.time(),
            "return_code": return_code,
            "stdout": str(stdout_path.relative_to(self.output_root)),
            "stdout_sha256": _sha256_file(stdout_path),
            "stderr": str(stderr_path.relative_to(self.output_root)),
            "stderr_sha256": _sha256_file(stderr_path),
            "formal_training_stage": stage.endswith("_formal_train"),
        }
        self.stage_history.append(row)
        if return_code != 0:
            raise QueueContractError(stage + " exited with code " + str(return_code))
        self._verify_source_and_reviews()
        self._verify_environment_and_official()
        self._verify_dataset_binding()

    def _verify_environment_live(self, label: str) -> None:
        stage = "environment_verify_" + label
        environment = self.authority["environment"]
        self._spawn_child(
            stage,
            (
                "-m",
                ENVIRONMENT_BUILDER_MODULE,
                "verify",
                "--environment-root",
                environment["root"],
                "--receipt",
                environment["receipt"],
                "--expected-receipt-sha256",
                environment["receipt_sha256"],
            ),
        )
        value = self._stage_stdout_json(stage, "isolated environment verifier")
        _exact_keys(
            value,
            {
                "schema_version",
                "status",
                "environment_root",
                "python",
                "python_sha256",
                "receipt",
                "receipt_sha256",
                "receipt_content_sha256",
                "import_inventory_content_sha256",
                "runtime_no_site",
                "sealed_test_or_real_accessed",
            },
            "isolated environment verification",
        )
        if (
            value.get("schema_version") != ENVIRONMENT_SCHEMA
            or value.get("status") != "verified_isolated_benchmark_environment"
            or value.get("environment_root") != environment["root"]
            or value.get("python") != environment["python"]
            or value.get("python_sha256") != environment["python_sha256"]
            or value.get("receipt") != environment["receipt"]
            or value.get("receipt_sha256") != environment["receipt_sha256"]
            or value.get("receipt_content_sha256")
            != environment["receipt_content_sha256"]
            or value.get("import_inventory_content_sha256")
            != environment["import_inventory_content_sha256"]
            or value.get("runtime_no_site") is not True
            or value.get("sealed_test_or_real_accessed") is not False
        ):
            raise QueueContractError("isolated environment verification differs")
        if self.environment_verification is None:
            self.environment_verification = dict(value)
        elif value != self.environment_verification:
            raise QueueContractError("isolated environment authority changed")
        _write_json_new(
            self.control_root / ("environment_verification_" + label + ".json"),
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "isolated_benchmark_environment_reverified",
                "checkpoint": label,
                "verification": value,
                "verifier_stdout_sha256": _sha256_file(
                    self.log_root / (stage + ".stdout.log")
                ),
                "sealed_test_or_real_opened": False,
            },
        )

    def _run_child(self, stage: str, arguments: Sequence[str]) -> None:
        if len(arguments) < 2 or arguments[0] != "-m":
            raise QueueContractError("child command is not one reviewed Python module")
        module = str(arguments[1])
        if module == ENVIRONMENT_BUILDER_MODULE:
            raise QueueContractError(
                "environment verifier may only run through the internal live gate"
            )
        self._verify_environment_live("before_" + stage)
        method_execution = module in {
            str(METHODS["pairingnet"]["runner_module"]),
            str(METHODS["pairingnet"]["smoke_module"]),
            str(METHODS["shreddingnet"]["runner_module"]),
        } or (
            module == STRICT_VERIFIER_MODULE
            and len(arguments) >= 3
            and arguments[2] in {"pairingnet", "shreddingnet"}
        )
        if method_execution and self.asset_freeze_verification is not None:
            self._reverify_train_val_assets(
                "immediate_before_" + stage, raw_spawn=True
            )
        self._spawn_child(stage, arguments)
        self._verify_environment_live("after_" + stage)

    def _pairing_audit(self) -> None:
        self._reverify_train_val_assets("before_pairingnet_audit")
        official = self.authority["official_sources"]["pairingnet"]["root"]
        self._run_child(
            "pairingnet_audit",
            (
                "-m",
                str(METHODS["pairingnet"]["runner_module"]),
                "--dataset-root",
                self._runner_dataset_root(),
                "--official-source-root",
                official,
                "--audit-only",
            ),
        )
        stdout_path = self.log_root / "pairingnet_audit.stdout.log"
        try:
            report = json.loads(stdout_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise QueueContractError("PairingNet audit stdout is not one JSON object") from error
        official_audit = report.get("official_source_audit")
        population = report.get("population_audit")
        counts = population.get("population") if isinstance(population, dict) else None
        expected_counts = {
            "train": {"rows": 24_000, "positive": 12_000, "negative": 12_000},
            "val": {"rows": 3_000, "positive": 1_500, "negative": 1_500},
        }
        counts_valid = isinstance(counts, dict) and set(counts) == set(expected_counts)
        if counts_valid:
            for split, expected in expected_counts.items():
                row = counts.get(split)
                if (
                    not isinstance(row, dict)
                    or any(row.get(key) != value for key, value in expected.items())
                    or type(row.get("split_unit_count")) is not int  # noqa: E721
                    or row["split_unit_count"] <= 0
                ):
                    counts_valid = False
                    break
        manifests = population.get("manifests") if isinstance(population, dict) else None
        manifest_binding_valid = (
            isinstance(manifests, dict)
            and set(manifests) == {"train_sha256", "val_sha256"}
            and _is_sha256(manifests.get("train_sha256"))
            and _is_sha256(manifests.get("val_sha256"))
        )
        if (
            report.get("schema_version") != "rachel-pairingnet-preflight-audit/1.0"
            or not isinstance(official_audit, dict)
            or official_audit.get("commit") != PAIRINGNET_COMMIT
            or not isinstance(population, dict)
            or population.get("status") != "train_val_population_audited"
            or population.get("formal_population_required") is not True
            or not counts_valid
            or population.get("parent_lineage_disjoint") is not True
            or population.get("sealed_synthetic_accessed") is not False
            or population.get("real_data_accessed") is not False
            or not manifest_binding_valid
        ):
            raise QueueContractError("PairingNet audit receipt identity differs")
        audit_binding = {
            "train": manifests["train_sha256"],
            "val": manifests["val_sha256"],
        }
        if self.dataset_binding is None or audit_binding != self.dataset_binding:
            raise QueueContractError(
                "PairingNet audit manifests differ from frozen train/val assets"
            )
        self._verify_dataset_binding()
        _write_json_new(
            self.control_root / "pairingnet_audit_gate.json",
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "pairingnet_train_val_audit_complete",
                "audit_stdout_sha256": _sha256_file(stdout_path),
                "runner_sha256": self.authority["methods"]["pairingnet"]["runner"]
                ["sha256"],
                "official_commit": PAIRINGNET_COMMIT,
                "train_val_manifest_sha256": self.dataset_binding,
                "sealed_test_or_real_opened": False,
            },
        )

    def _pairing_smoke(self) -> None:
        self._reverify_train_val_assets("before_pairingnet_gpu_smoke")
        receipt_path = self.control_root / "pairingnet_gpu_smoke.json"
        runtime = self.authority["runtime"]["pairingnet"]
        self._verify_environment_and_official()
        self._run_child(
            "pairingnet_gpu_smoke",
            (
                "-m",
                str(METHODS["pairingnet"]["smoke_module"]),
                "--dataset-root",
                self._runner_dataset_root(),
                "--official-source-root",
                self.authority["official_sources"]["pairingnet"]["root"],
                "--output",
                str(receipt_path),
                "--device",
                self.authority["runtime"]["device"],
                "--precision",
                runtime["precision"],
                "--num-workers",
                str(runtime["num_workers"]),
            ),
        )
        receipt = _load_json(receipt_path, "PairingNet GPU smoke receipt")
        _validate_content_sha(receipt, "PairingNet GPU smoke receipt")
        scope = receipt.get("scope")
        method = self.authority["methods"]["pairingnet"]
        scalars = receipt.get("scalars")
        official_audit = receipt.get("official_source_audit")
        batch_order = receipt.get("batch_order")
        if (
            set(receipt)
            != {
                "schema_version",
                "status",
                "method_id",
                "official_commit",
                "adapter_source_sha256",
                "adaptation_contract_sha256",
                "dataset_manifest_sha256",
                "batch_size",
                "device",
                "device_name",
                "precision",
                "num_workers",
                "elapsed_seconds",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "scalars",
                "batch_order",
                "official_source_audit",
                "scope",
                "content_sha256",
            }
            or receipt.get("schema_version")
            != "rachel-pairingnet-gpu-smoke/1.0"
            or receipt.get("status")
            != "complete_train_only_one_formal_batch_optimizer_step"
            or receipt.get("method_id") != METHODS["pairingnet"]["method_id"]
            or receipt.get("official_commit") != PAIRINGNET_COMMIT
            or receipt.get("adapter_source_sha256") != method["runner"]["sha256"]
            or receipt.get("adaptation_contract_sha256")
            != method["adaptation_contract"]["sha256"]
            or receipt.get("batch_size") != 20
            or receipt.get("device") != self.authority["runtime"]["device"]
            or not isinstance(receipt.get("device_name"), str)
            or not receipt.get("device_name")
            or receipt.get("precision") != runtime["precision"]
            or receipt.get("num_workers") != runtime["num_workers"]
            or receipt.get("dataset_manifest_sha256")
            != {
                "train_sha256": self.dataset_binding["train"],
                "val_sha256": self.dataset_binding["val"],
            }
            or not _finite_number(receipt.get("elapsed_seconds"))
            or float(receipt["elapsed_seconds"]) <= 0.0
            or type(receipt.get("peak_allocated_bytes")) is not int  # noqa: E721
            or int(receipt["peak_allocated_bytes"]) <= 0
            or type(receipt.get("peak_reserved_bytes")) is not int  # noqa: E721
            or int(receipt["peak_reserved_bytes"])
            < int(receipt["peak_allocated_bytes"])
            or not isinstance(scalars, dict)
            or set(scalars)
            != {"total", "matching_focal", "pair_bce", "gradient_norm"}
            or any(not _finite_number(value) for value in scalars.values())
            or any(float(value) < 0.0 for value in scalars.values())
            or not isinstance(official_audit, dict)
            or set(official_audit)
            != {
                "schema_version",
                "status",
                "checkout_root",
                "repository",
                "origin_reported",
                "commit",
                "clean",
                "key_file_sha256",
            }
            or official_audit.get("schema_version")
            != "pairingnet-official-source-audit/1.0"
            or official_audit.get("status") != "exact_clean_checkout_verified"
            or official_audit.get("checkout_root")
            != self.authority["official_sources"]["pairingnet"]["root"]
            or official_audit.get("repository") != PAIRINGNET_OFFICIAL_REPOSITORY
            or not isinstance(official_audit.get("origin_reported"), str)
            or not official_audit.get("origin_reported")
            or official_audit.get("commit") != PAIRINGNET_COMMIT
            or official_audit.get("clean") is not True
            or official_audit.get("key_file_sha256")
            != PAIRINGNET_OFFICIAL_KEY_FILE_SHA256
            or scope
            != {
                "train_manifest_used_for_tensor_loading": True,
                "validation_manifest_metadata_audited": True,
                "validation_tensor_loading": False,
                "sealed_synthetic_or_real_opened": False,
                "formal_training_started": False,
                "random_smoke_model_not_checkpoint": True,
                "optimizer_steps": 1,
            }
        ):
            raise QueueContractError("PairingNet GPU smoke receipt differs")
        _validate_pairingnet_smoke_batch_order(batch_order)
        train_labels = _train_manifest_labels(Path(self._runner_dataset_root()))
        assert isinstance(batch_order, dict)
        if any(
            train_labels.get(row["pair_id"]) is not row["label"]
            for row in batch_order["rows"]
        ):
            raise QueueContractError("PairingNet smoke pair/label order differs from train")
        receipt_file_sha256 = _sha256_file(receipt_path)
        self.pairingnet_smoke_receipt_sha256 = receipt_file_sha256
        self.pairingnet_smoke_content_sha256 = str(receipt["content_sha256"])
        _write_json_new(
            self.control_root / "pairingnet_gpu_smoke_gate.json",
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "pairingnet_gpu_smoke_accepted_before_formal_train",
                "receipt": str(receipt_path.relative_to(self.output_root)),
                "receipt_file_sha256": receipt_file_sha256,
                "receipt_content_sha256": receipt["content_sha256"],
                "batch_size": 20,
                "ordered_pair_ids_sha256": batch_order[
                    "ordered_pair_ids_sha256"
                ],
                "ordered_pair_labels_sha256": batch_order[
                    "ordered_pair_labels_sha256"
                ],
                "positive_and_negative_present": True,
                "resource_metrics_finite": True,
                "formal_training_started": False,
                "sealed_test_or_real_opened": False,
            },
        )

    def _pairing_train(self) -> None:
        self.stage = "pairingnet_pretrain_revalidation"
        self._verify_source_and_reviews()
        self._verify_environment_and_official()
        self._reverify_train_val_assets("before_pairingnet_formal_train")
        self._strict_verify_upstreams("before_pairingnet_formal_train")
        smoke_path = _require_regular_file(
            self.control_root / "pairingnet_gpu_smoke.json",
            "accepted PairingNet GPU smoke receipt",
        )
        smoke = _load_json(smoke_path, "accepted PairingNet GPU smoke receipt")
        _validate_content_sha(smoke, "accepted PairingNet GPU smoke receipt")
        gate = _load_json(
            self.control_root / "pairingnet_gpu_smoke_gate.json",
            "PairingNet GPU smoke acceptance gate",
        )
        _validate_content_sha(gate, "PairingNet GPU smoke acceptance gate")
        if (
            not _is_sha256(self.pairingnet_smoke_receipt_sha256)
            or _sha256_file(smoke_path) != self.pairingnet_smoke_receipt_sha256
            or smoke.get("content_sha256") != self.pairingnet_smoke_content_sha256
            or gate.get("status")
            != "pairingnet_gpu_smoke_accepted_before_formal_train"
            or gate.get("receipt_file_sha256")
            != self.pairingnet_smoke_receipt_sha256
            or gate.get("receipt_content_sha256")
            != self.pairingnet_smoke_content_sha256
        ):
            raise QueueContractError(
                "PairingNet GPU smoke receipt/gate changed before formal train"
            )
        if os.path.lexists(self.output_root / "pairingnet"):
            raise QueueContractError("PairingNet formal output is not fresh")
        runtime = self.authority["runtime"]["pairingnet"]
        self._run_child(
            "pairingnet_formal_train",
            (
                "-m",
                str(METHODS["pairingnet"]["runner_module"]),
                "--dataset-root",
                self._runner_dataset_root(),
                "--official-source-root",
                self.authority["official_sources"]["pairingnet"]["root"],
                "--output-root",
                str(self.output_root / "pairingnet"),
                "--device",
                self.authority["runtime"]["device"],
                "--precision",
                runtime["precision"],
                "--num-workers",
                str(runtime["num_workers"]),
            ),
        )
        receipt_path = self.output_root / "pairingnet" / "completion_receipt.json"
        receipt = _load_json(receipt_path, "PairingNet completion receipt")
        method = self.authority["methods"]["pairingnet"]
        population = receipt.get("population_audit")
        if (
            receipt.get("status") != "train_validation_complete"
            or receipt.get("method_id") != METHODS["pairingnet"]["method_id"]
            or receipt.get("official_commit") != PAIRINGNET_COMMIT
            or receipt.get("adapter_source_sha256") != method["runner"]["sha256"]
            or receipt.get("adaptation_contract_sha256")
            != method["adaptation_contract"]["sha256"]
            or receipt.get("convergence_demonstrated") is not True
            or receipt.get("sealed_synthetic_accessed") is not False
            or receipt.get("real_data_accessed") is not False
            or not isinstance(population, dict)
            or population.get("manifests")
            != {
                "train_sha256": self.dataset_binding["train"],
                "val_sha256": self.dataset_binding["val"],
            }
        ):
            raise QueueContractError("PairingNet formal completion receipt differs")
        self._reverify_train_val_assets(
            "before_pairingnet_completion_strict_loader"
        )
        strict = self._strict_verify_pairingnet_completion()
        _write_json_new(
            self.control_root / "pairingnet_completion_gate.json",
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "pairingnet_formal_train_validation_complete",
                "receipt": str(receipt_path.relative_to(self.output_root)),
                "receipt_sha256": _sha256_file(receipt_path),
                "convergence_demonstrated": True,
                "strict_verification": strict,
                "sealed_test_or_real_opened": False,
            },
        )

    def _strict_verify_pairingnet_completion(self) -> dict[str, object]:
        stage = "pairingnet_completion_strict_verify"
        runtime = self.authority["runtime"]["pairingnet"]
        self._run_child(
            stage,
            (
                "-m",
                STRICT_VERIFIER_MODULE,
                "pairingnet",
                "--output-root",
                str(self.output_root / "pairingnet"),
                "--dataset-root",
                self._runner_dataset_root(),
                "--official-source-root",
                self.authority["official_sources"]["pairingnet"]["root"],
                "--device",
                self.authority["runtime"]["device"],
                "--precision",
                runtime["precision"],
                "--num-workers",
                str(runtime["num_workers"]),
            ),
        )
        value = self._stage_stdout_json(stage, "PairingNet completion verifier")
        _exact_keys(
            value,
            {
                "schema_version",
                "status",
                "output_root",
                "completion_receipt_sha256",
                "artifacts",
                "epochs_completed",
                "winner_epoch",
                "plateau",
                "stop_reason",
                "validation_threshold",
                "manifest_content_sha256",
                "winner_and_last_strict_loaded_cpu",
                "threshold_replayed",
                "inference_contract_verified",
                "validation_report_verified",
                "test_or_real_opened",
            },
            "PairingNet completion verification",
        )
        artifacts = value.get("artifacts")
        if (
            value.get("schema_version")
            != "rachel-benchmark-queue-pairingnet-verification/1.0"
            or value.get("status")
            != "verified_pairingnet_train_val_completion"
            or value.get("output_root") != str(self.output_root / "pairingnet")
            or value.get("completion_receipt_sha256")
            != _sha256_file(
                self.output_root / "pairingnet" / "completion_receipt.json"
            )
            or not isinstance(artifacts, dict)
            or set(artifacts)
            != {
                "winner",
                "last",
                "threshold",
                "inference_contract",
                "validation_report",
                "validation_predictions",
            }
            or any(not _is_sha256(item) for item in artifacts.values())
            or type(value.get("epochs_completed")) is not int  # noqa: E721
            or not 20 <= int(value["epochs_completed"]) <= 128
            or type(value.get("winner_epoch")) is not int  # noqa: E721
            or not 1 <= int(value["winner_epoch"]) <= int(value["epochs_completed"])
            or type(value.get("plateau")) is not int  # noqa: E721
            or int(value["plateau"]) < 12
            or value.get("stop_reason") != "validation_plateau"
            or not _finite_number(value.get("validation_threshold"))
            or not 0.0 <= float(value["validation_threshold"]) <= 1.0
            or value.get("manifest_content_sha256") != self.dataset_binding
            or value.get("winner_and_last_strict_loaded_cpu") is not True
            or value.get("threshold_replayed") is not True
            or value.get("inference_contract_verified") is not True
            or value.get("validation_report_verified") is not True
            or value.get("test_or_real_opened") is not False
        ):
            raise QueueContractError("PairingNet strict completion verification differs")
        return dict(value)

    def _shredding_runtime_arguments(self) -> tuple[str, ...]:
        runtime = self.authority["runtime"]["shreddingnet"]
        values = [
            "--num-workers",
            str(runtime["num_workers"]),
            "--coarse-microbatch",
            str(runtime["coarse_microbatch"]),
            "--matching-microbatch",
            str(runtime["matching_microbatch"]),
            "--classify-microbatch",
            str(runtime["classify_microbatch"]),
        ]
        if not runtime["amp"]:
            values.append("--no-amp")
        return tuple(values)

    def _shredding_audit(self) -> None:
        self._reverify_train_val_assets("before_shreddingnet_audit")
        receipt_path = self.control_root / "shreddingnet_audit.json"
        self._run_child(
            "shreddingnet_audit",
            (
                "-m",
                str(METHODS["shreddingnet"]["runner_module"]),
                "audit",
                "--dataset-root",
                self._runner_dataset_root(),
                "--official-repo",
                self.authority["official_sources"]["shreddingnet"]["root"],
                "--output",
                str(receipt_path),
            ),
        )
        receipt = _load_json(receipt_path, "ShreddingNet audit receipt")
        _validate_content_sha(receipt, "ShreddingNet audit receipt")
        method = self.authority["methods"]["shreddingnet"]
        adapter = receipt.get("adapter_identity")
        official = receipt.get("official_source")
        dataset = receipt.get("dataset")
        if (
            receipt.get("status") != "audit_complete_train_val_only"
            or receipt.get("method_id") != METHODS["shreddingnet"]["method_id"]
            or not isinstance(adapter, dict)
            or adapter.get("adapter_source_sha256") != method["runner"]["sha256"]
            or adapter.get("adaptation_document_sha256")
            != method["adaptation_contract"]["sha256"]
            or not isinstance(official, dict)
            or official.get("commit") != SHREDDINGNET_COMMIT
            or not isinstance(dataset, dict)
            or dataset.get("counts")
            != {
                "train": {"rows": 24_000, "positive": 12_000, "negative": 12_000},
                "val": {"rows": 3_000, "positive": 1_500, "negative": 1_500},
            }
            or dataset.get("formal_counts_required") is not True
            or dataset.get("lineage_disjoint") is not True
            or dataset.get("opened_manifests")
            != ["pairs/train.jsonl", "pairs/val.jsonl"]
            or dataset.get("sealed_test_manifest_opened") is not False
            or dataset.get("real_data_opened") is not False
            or dataset.get("manifest_content_sha256") != self.dataset_binding
        ):
            raise QueueContractError("ShreddingNet audit receipt identity differs")

    def _shredding_smoke(self) -> None:
        self._reverify_train_val_assets("before_shreddingnet_gpu_smoke")
        receipt_path = self.control_root / "shreddingnet_gpu_smoke.json"
        self._verify_environment_and_official()
        self._run_child(
            "shreddingnet_gpu_smoke",
            (
                "-m",
                str(METHODS["shreddingnet"]["runner_module"]),
                "gpu-smoke",
                "--dataset-root",
                self._runner_dataset_root(),
                "--official-repo",
                self.authority["official_sources"]["shreddingnet"]["root"],
                "--output",
                str(receipt_path),
                "--device",
                self.authority["runtime"]["device"],
                *self._shredding_runtime_arguments(),
            ),
        )
        receipt = _load_json(receipt_path, "ShreddingNet GPU smoke receipt")
        _validate_content_sha(receipt, "ShreddingNet GPU smoke receipt")
        method = self.authority["methods"]["shreddingnet"]
        adapter = receipt.get("adapter_identity")
        scope = receipt.get("scope")
        dataset_binding = receipt.get("dataset_binding")
        ordered_binding = (
            dataset_binding.get("ordered_pair_ids_sha256")
            if isinstance(dataset_binding, dict)
            else None
        )
        provenance = receipt.get("train_val_provenance")
        preflight = receipt.get("runtime_preflight")
        stage_shapes = (
            preflight.get("stage_output_shapes")
            if isinstance(preflight, dict)
            else None
        )
        expected_runtime = {
            "coarse_microbatch": self.authority["runtime"]["shreddingnet"]
            ["coarse_microbatch"],
            "matching_microbatch": self.authority["runtime"]["shreddingnet"]
            ["matching_microbatch"],
            "classify_microbatch": self.authority["runtime"]["shreddingnet"]
            ["classify_microbatch"],
            "amp": self.authority["runtime"]["shreddingnet"]["amp"],
        }
        expected_adapter = {
            "adapter_source_filename": Path(
                method["runner"]["relative_path"]
            ).name,
            "adapter_source_sha256": method["runner"]["sha256"],
            "adaptation_document_filename": Path(
                method["adaptation_contract"]["relative_path"]
            ).name,
            "adaptation_document_sha256": method["adaptation_contract"]["sha256"],
        }
        if (
            set(receipt)
            != {
                "schema_version",
                "status",
                "method_id",
                "device",
                "device_name",
                "recipe",
                "runtime_batch_config",
                "adapter_identity",
                "dataset_binding",
                "train_val_provenance",
                "runtime_preflight",
                "stages",
                "dataset_manifest_sha256",
                "official_commit",
                "scope",
                "numeric_contract",
                "content_sha256",
            }
            or receipt.get("schema_version") != "rachel-shreddingnet-gpu-smoke/1.0"
            or receipt.get("status")
            != "complete_train_only_release_effective_step_smoke"
            or receipt.get("method_id") != METHODS["shreddingnet"]["method_id"]
            or receipt.get("official_commit") != SHREDDINGNET_COMMIT
            or receipt.get("device") != self.authority["runtime"]["device"]
            or not isinstance(receipt.get("device_name"), str)
            or not receipt.get("device_name")
            or receipt.get("recipe") != SHREDDINGNET_RELEASE_RECIPE
            or receipt.get("runtime_batch_config") != expected_runtime
            or adapter != expected_adapter
            or not isinstance(dataset_binding, dict)
            or set(dataset_binding)
            != {"manifest_content_sha256", "ordered_pair_ids_sha256"}
            or dataset_binding.get("manifest_content_sha256")
            != self.dataset_binding
            or not isinstance(ordered_binding, dict)
            or set(ordered_binding) != {"train", "val"}
            or any(not _is_sha256(ordered_binding.get(split)) for split in ("train", "val"))
            or receipt.get("dataset_manifest_sha256") != self.dataset_binding
            or provenance
            != {
                "adapter_source_sha256": expected_adapter["adapter_source_sha256"],
                "adaptation_document_sha256": expected_adapter[
                    "adaptation_document_sha256"
                ],
                "train_manifest_content_sha256": self.dataset_binding["train"],
                "val_manifest_content_sha256": self.dataset_binding["val"],
                "train_ordered_pair_ids_sha256": ordered_binding["train"],
                "val_ordered_pair_ids_sha256": ordered_binding["val"],
            }
            or not isinstance(preflight, dict)
            or set(preflight)
            != {
                "preflight_kind",
                "device",
                "torch_version",
                "torch_geometric_version",
                "real_stage_model_constructed_and_forwarded",
                "stage_output_shapes",
                "train_probe_pair_ids_sha256",
                "train_positive_and_negative_artifacts_decoded",
                "train_val_manifests_reverified_after_decode",
                "output_created_or_written",
            }
            or preflight.get("preflight_kind")
            != "rachel_shreddingnet_real_models_train_probe"
            or preflight.get("device") != self.authority["runtime"]["device"]
            or not isinstance(preflight.get("torch_version"), str)
            or not preflight.get("torch_version")
            or not isinstance(preflight.get("torch_geometric_version"), str)
            or not preflight.get("torch_geometric_version")
            or preflight.get("real_stage_model_constructed_and_forwarded")
            != {"coarse": True, "matching": True, "classify": True}
            or not isinstance(stage_shapes, dict)
            or set(stage_shapes) != set(SHREDDINGNET_EFFECTIVE_BATCH)
            or any(not _positive_shape_tree(stage_shapes[stage]) for stage in stage_shapes)
            or not _is_sha256(preflight.get("train_probe_pair_ids_sha256"))
            or preflight.get("train_positive_and_negative_artifacts_decoded") is not True
            or preflight.get("train_val_manifests_reverified_after_decode") is not True
            or preflight.get("output_created_or_written") is not False
            or scope
            != {
                "train_manifest_only_for_tensor_loading": True,
                "validation_tensor_loading": False,
                "sealed_test_or_real_opened": False,
                "random_smoke_models_not_checkpoints": True,
                "formal_training_started": False,
            }
            or receipt.get("numeric_contract")
            != {
                "dual_softmax_probability_compute_dtype": "float32",
                "amp_logits_promoted_before_mask_and_softmax": True,
                "focal_logarithm_epsilon_semantics": (
                    "log_p_plus_1e-9_and_log_1_minus_p_plus_1e-9"
                ),
            }
        ):
            raise QueueContractError("ShreddingNet GPU smoke receipt differs")
        stages = receipt.get("stages")
        if not isinstance(stages, dict) or set(stages) != set(
            SHREDDINGNET_EFFECTIVE_BATCH
        ):
            raise QueueContractError(
                "ShreddingNet smoke stages must be exact coarse/matching/classify dict"
            )
        for stage in ("coarse", "matching", "classify"):
            _validate_shreddingnet_smoke_stage(
                stage=stage,
                value=stages[stage],
                configured_microbatch=expected_runtime[stage + "_microbatch"],
                amp=expected_runtime["amp"],
            )
        receipt_file_sha256 = _sha256_file(receipt_path)
        self.shreddingnet_smoke_receipt_sha256 = receipt_file_sha256
        self.shreddingnet_smoke_content_sha256 = str(receipt["content_sha256"])
        _write_json_new(
            self.control_root / "shreddingnet_gpu_smoke_gate.json",
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "shreddingnet_gpu_smoke_accepted_before_formal_train",
                "receipt": str(receipt_path.relative_to(self.output_root)),
                "receipt_file_sha256": receipt_file_sha256,
                "receipt_content_sha256": receipt["content_sha256"],
                "effective_batch_size": dict(SHREDDINGNET_EFFECTIVE_BATCH),
                "runtime_batch_config": expected_runtime,
                "metrics_finite_and_complete": True,
                "formal_training_started": False,
                "sealed_test_or_real_opened": False,
            },
        )

    def _shredding_train(self) -> None:
        self.stage = "shreddingnet_pretrain_revalidation"
        self._verify_source_and_reviews()
        self._verify_environment_and_official()
        self._reverify_train_val_assets("before_shreddingnet_formal_train")
        self._strict_verify_upstreams("before_shreddingnet_formal_train")
        smoke_path = _require_regular_file(
            self.control_root / "shreddingnet_gpu_smoke.json",
            "accepted ShreddingNet GPU smoke receipt",
        )
        gate = _load_json(
            self.control_root / "shreddingnet_gpu_smoke_gate.json",
            "ShreddingNet GPU smoke acceptance gate",
        )
        _validate_content_sha(gate, "ShreddingNet GPU smoke acceptance gate")
        smoke = _load_json(smoke_path, "accepted ShreddingNet GPU smoke receipt")
        _validate_content_sha(smoke, "accepted ShreddingNet GPU smoke receipt")
        if (
            not _is_sha256(self.shreddingnet_smoke_receipt_sha256)
            or _sha256_file(smoke_path)
            != self.shreddingnet_smoke_receipt_sha256
            or smoke.get("content_sha256")
            != self.shreddingnet_smoke_content_sha256
            or gate.get("status")
            != "shreddingnet_gpu_smoke_accepted_before_formal_train"
            or gate.get("receipt_file_sha256")
            != self.shreddingnet_smoke_receipt_sha256
            or gate.get("receipt_content_sha256")
            != self.shreddingnet_smoke_content_sha256
        ):
            raise QueueContractError(
                "ShreddingNet GPU smoke receipt changed before formal train"
            )
        destination = self.output_root / "shreddingnet"
        if os.path.lexists(destination):
            raise QueueContractError("ShreddingNet formal output is not fresh")
        self._run_child(
            "shreddingnet_formal_train",
            (
                "-m",
                str(METHODS["shreddingnet"]["runner_module"]),
                "train",
                "--dataset-root",
                self._runner_dataset_root(),
                "--official-repo",
                self.authority["official_sources"]["shreddingnet"]["root"],
                "--output-root",
                str(destination),
                "--device",
                self.authority["runtime"]["device"],
                *self._shredding_runtime_arguments(),
            ),
        )
        freeze_path = destination / "train_val_freeze.json"
        freeze = _load_json(freeze_path, "ShreddingNet train/val freeze")
        _validate_content_sha(freeze, "ShreddingNet train/val freeze")
        method = self.authority["methods"]["shreddingnet"]
        adapter = freeze.get("adapter_identity")
        scope = freeze.get("scope")
        dataset_binding = freeze.get("dataset_binding")
        if (
            freeze.get("schema_version")
            != "rachel-shreddingnet-train-val-freeze/1.0"
            or freeze.get("checkpoint_kind")
            != "rachel_shreddingnet_train_val_freeze"
            or freeze.get("status") != "complete_train_val_frozen_no_test_or_real"
            or freeze.get("method_id") != METHODS["shreddingnet"]["method_id"]
            or freeze.get("official_commit") != SHREDDINGNET_COMMIT
            or not isinstance(adapter, dict)
            or adapter.get("adapter_source_sha256") != method["runner"]["sha256"]
            or adapter.get("adaptation_document_sha256")
            != method["adaptation_contract"]["sha256"]
            or not isinstance(scope, dict)
            or scope.get("sealed_test_or_real_opened") is not False
            or scope.get("train_val_only") is not True
            or not isinstance(dataset_binding, dict)
            or dataset_binding.get("manifest_content_sha256")
            != self.dataset_binding
        ):
            raise QueueContractError("ShreddingNet train/val freeze identity differs")
        self._reverify_train_val_assets(
            "before_shreddingnet_completion_strict_loader"
        )
        strict = self._strict_verify_shreddingnet_completion()
        _write_json_new(
            self.control_root / "shreddingnet_completion_gate.json",
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "shreddingnet_formal_train_validation_complete",
                "freeze": str(freeze_path.relative_to(self.output_root)),
                "freeze_sha256": _sha256_file(freeze_path),
                "release_three_stage_schedule_complete": True,
                "strict_verification": strict,
                "sealed_test_or_real_opened": False,
            },
        )

    def _strict_verify_shreddingnet_completion(self) -> dict[str, object]:
        stage = "shreddingnet_completion_strict_verify"
        self._run_child(
            stage,
            (
                "-m",
                STRICT_VERIFIER_MODULE,
                "shreddingnet",
                "--output-root",
                str(self.output_root / "shreddingnet"),
                "--dataset-root",
                self._runner_dataset_root(),
                "--official-source-root",
                self.authority["official_sources"]["shreddingnet"]["root"],
            ),
        )
        value = self._stage_stdout_json(stage, "ShreddingNet completion verifier")
        _exact_keys(
            value,
            {
                "schema_version",
                "status",
                "output_root",
                "freeze_file_sha256",
                "freeze_content_sha256",
                "run_contract_file_sha256",
                "validation_report_file_sha256",
                "validation_report_content_sha256",
                "stage_artifacts",
                "manifest_content_sha256",
                "three_stage_winners_strict_loaded_cpu",
                "three_stage_progress_checkpoints_strict_loaded_cpu",
                "threshold_score_and_fit_replayed",
                "inference_contract_verified",
                "validation_report_verified",
                "test_or_real_opened",
            },
            "ShreddingNet completion verification",
        )
        stage_artifacts = value.get("stage_artifacts")
        valid_stages = isinstance(stage_artifacts, dict) and set(stage_artifacts) == {
            "coarse",
            "matching",
            "classify",
        }
        if valid_stages:
            for stage_name, row in stage_artifacts.items():
                if not isinstance(row, dict) or set(row) != {
                    "completion_file_sha256",
                    "completion_content_sha256",
                    "winner_sha256",
                    "progress_sha256",
                    "completed_epochs",
                    "winner_epoch_zero_based",
                }:
                    valid_stages = False
                    break
                epochs = SHREDDINGNET_RELEASE_RECIPE[stage_name + "_epochs"]
                if (
                    any(
                        not _is_sha256(row.get(key))
                        for key in (
                            "completion_file_sha256",
                            "completion_content_sha256",
                            "winner_sha256",
                            "progress_sha256",
                        )
                    )
                    or row.get("completed_epochs") != epochs
                    or type(row.get("winner_epoch_zero_based")) is not int  # noqa: E721
                    or not 0 <= int(row["winner_epoch_zero_based"]) < epochs
                ):
                    valid_stages = False
                    break
        if (
            value.get("schema_version")
            != "rachel-benchmark-queue-shreddingnet-verification/1.0"
            or value.get("status")
            != "verified_shreddingnet_train_val_completion"
            or value.get("output_root") != str(self.output_root / "shreddingnet")
            or value.get("freeze_file_sha256")
            != _sha256_file(
                self.output_root / "shreddingnet" / "train_val_freeze.json"
            )
            or not _is_sha256(value.get("freeze_content_sha256"))
            or not _is_sha256(value.get("run_contract_file_sha256"))
            or not _is_sha256(value.get("validation_report_file_sha256"))
            or not _is_sha256(value.get("validation_report_content_sha256"))
            or not valid_stages
            or value.get("manifest_content_sha256") != self.dataset_binding
            or value.get("three_stage_winners_strict_loaded_cpu") is not True
            or value.get("three_stage_progress_checkpoints_strict_loaded_cpu")
            is not True
            or value.get("threshold_score_and_fit_replayed") is not True
            or value.get("inference_contract_verified") is not True
            or value.get("validation_report_verified") is not True
            or value.get("test_or_real_opened") is not False
        ):
            raise QueueContractError(
                "ShreddingNet strict completion verification differs"
            )
        return dict(value)

    def _artifact_inventory(self, *, require_complete: bool) -> dict[str, object]:
        candidates = {
            "launch_authority": self.control_root / "launch_authority.json",
            "source_code_freeze": self.control_root / "source_code_freeze.json",
            "train_val_asset_freeze_gate": self.control_root
            / "train_val_asset_freeze_gate.json",
            "train_val_asset_freeze_receipt": self.control_root
            / "train_val_asset_freeze/train_val_semantic_asset_freeze_receipt.json",
            "upstream_plateau_gate": self.control_root / "upstream_plateau_gate.json",
            "pairingnet_audit_gate": self.control_root / "pairingnet_audit_gate.json",
            "pairingnet_gpu_smoke": self.control_root / "pairingnet_gpu_smoke.json",
            "pairingnet_gpu_smoke_gate": self.control_root
            / "pairingnet_gpu_smoke_gate.json",
            "pairingnet_completion_gate": self.control_root
            / "pairingnet_completion_gate.json",
            "pairingnet_completion_receipt": self.output_root
            / "pairingnet/completion_receipt.json",
            "shreddingnet_audit": self.control_root / "shreddingnet_audit.json",
            "shreddingnet_gpu_smoke": self.control_root
            / "shreddingnet_gpu_smoke.json",
            "shreddingnet_gpu_smoke_gate": self.control_root
            / "shreddingnet_gpu_smoke_gate.json",
            "shreddingnet_completion_gate": self.control_root
            / "shreddingnet_completion_gate.json",
            "shreddingnet_train_val_freeze": self.output_root
            / "shreddingnet/train_val_freeze.json",
            "content_sha256_inventory": self.control_root
            / "content_sha256_inventory.json",
        }
        result: dict[str, object] = {}
        missing = []
        for label, path in candidates.items():
            if not os.path.lexists(path):
                if require_complete:
                    missing.append(label)
                continue
            try:
                regular = _require_regular_file(path, label + " artifact")
            except QueueContractError:
                if require_complete:
                    raise
                result[label] = {
                    "path": str(path.relative_to(self.output_root)),
                    "status": "present_but_unsafe_or_nonregular_not_opened",
                }
                continue
            result[label] = {
                "path": str(regular.relative_to(self.output_root)),
                "sha256": _sha256_file(regular),
            }
        if missing:
            raise QueueContractError(
                "successful queue lacks required artifacts: " + ", ".join(missing)
            )
        return result

    def _scan_recursive_output_content(self) -> dict[str, object]:
        excluded = {
            "control/content_sha256_inventory.json",
            "control/terminal_receipt.json",
        }
        files = []
        directory_count = 0

        def visit(directory: Path) -> None:
            nonlocal directory_count
            directory_count += 1
            try:
                entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
            except OSError as error:
                raise QueueContractError(
                    "cannot recursively inventory benchmark output"
                ) from error
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(self.output_root).as_posix()
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError as error:
                    raise QueueContractError(
                        "cannot stat recursive output member: " + relative
                    ) from error
                if stat.S_ISLNK(metadata.st_mode):
                    raise QueueContractError(
                        "recursive output inventory rejects symlink: " + relative
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    visit(path)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise QueueContractError(
                        "recursive output inventory rejects special file: " + relative
                    )
                if relative in excluded:
                    continue
                files.append(
                    {
                        "relative_path": relative,
                        "size_bytes": metadata.st_size,
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "sha256": _sha256_file(path),
                    }
                )

        visit(self.output_root)
        return {
            "files": files,
            "file_count": len(files),
            "directory_count": directory_count,
            "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        }

    def _publish_recursive_output_inventory(self) -> dict[str, object]:
        path = self.control_root / "content_sha256_inventory.json"
        first = self._scan_recursive_output_content()
        value = _write_json_new(
            path,
            {
                "schema_version": CONTROLLER_SCHEMA,
                "status": "complete_recursive_output_content_inventory",
                "scope": "all_regular_output_files_excluding_inventory_and_terminal",
                **first,
                "symlinks_or_special_files_present": False,
                "sealed_test_or_real_opened": False,
            },
        )
        if self._scan_recursive_output_content() != first:
            raise QueueContractError(
                "benchmark outputs changed while publishing recursive inventory"
            )
        return {
            "path": str(path.relative_to(self.output_root)),
            "file_sha256": _sha256_file(path),
            "content_sha256": value["content_sha256"],
            "file_count": first["file_count"],
            "total_size_bytes": first["total_size_bytes"],
        }

    def _write_terminal(
        self,
        *,
        success: bool,
        exit_code: int,
        termination: str,
        error: Optional[BaseException] = None,
    ) -> None:
        if not self.failure_receipt_enabled or self.terminal_written:
            return
        recursive_inventory = (
            self._publish_recursive_output_inventory() if success else None
        )
        value = {
            "schema_version": CONTROLLER_SCHEMA,
            "status": (
                "complete_same_data_benchmark_queue"
                if success
                else "failed_same_data_benchmark_queue"
            ),
            "stage": self.stage,
            "exit_code": int(exit_code),
            "termination": termination,
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error) if error is not None else None,
            "training_started": dict(self.training_started),
            "upstream_receipts": dict(self.upstream_receipts),
            "original_dataset_root": self.authority["dataset_root"],
            "frozen_runner_dataset_root": (
                str(self.benchmark_dataset_root)
                if self.benchmark_dataset_root is not None
                else None
            ),
            "source_train_val_manifest_sha256": self.source_dataset_binding,
            "frozen_train_val_manifest_sha256": self.dataset_binding,
            "stage_history": list(self.stage_history),
            "artifact_inventory": self._artifact_inventory(
                require_complete=success
            ),
            "recursive_output_content_inventory": recursive_inventory,
            "source_inventory_sha256": hashlib.sha256(
                _canonical_bytes(self.authority["source_inventory"])
            ).hexdigest(),
            "complete_source_inventory_sha256": self.authority[
                "complete_source_inventory_sha256"
            ],
            "source_permissions": self.authority["source_permissions"],
            "config_file_sha256": self.authority["config_file_sha256"],
            "official_commits": {
                method: self.authority["official_sources"][method]["commit"]
                for method in METHODS
            },
            "sealed_test_corrosion_real_opened": False,
            "test_or_real_metric_used_for_selection": False,
            "resume_supported": False,
            "threat_model_limitations": list(THREAT_MODEL_LIMITATIONS),
        }
        _write_json_new(self.terminal_receipt, value)
        self.terminal_written = True

    def run(self) -> None:
        self._acquire_output()
        self._wait_for_upstreams()
        self._create_train_val_asset_freeze()
        self._pairing_audit()
        self._pairing_smoke()
        self._pairing_train()
        self._reverify_train_val_assets("between_methods_before_shreddingnet")
        self._shredding_audit()
        self._shredding_smoke()
        self._shredding_train()
        self._reverify_train_val_assets("after_all_benchmark_training")
        self._strict_verify_upstreams("after_all_benchmark_training")
        self._reverify_train_val_assets("final_after_upstream_reverification")
        self.stage = "post_train_authority_revalidation"
        self._verify_source_and_reviews()
        self._verify_environment_and_official()
        self.stage = "complete"
        self._write_terminal(success=True, exit_code=0, termination="normal")


def _parse_arguments(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="fail-closed queue for Rachel same-data paper benchmarks"
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(arguments)


def _require_no_site_controller_runtime() -> None:
    """Fail before config/output access unless the controller used -I -S -B."""

    if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
        raise QueueContractError(
            "queue controller itself requires Python -I -S -B"
        )
    if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise QueueContractError(
            "site/user customization loaded before queue controller"
        )
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        raise QueueContractError("queue controller Python path/home injection is forbidden")


def main(arguments: Optional[Sequence[str]] = None) -> int:
    try:
        _require_no_site_controller_runtime()
    except BaseException as error:
        print("same-data benchmark queue: " + str(error), file=sys.stderr)
        return 1
    parsed = _parse_arguments(arguments)
    queue: Optional[BenchmarkQueue] = None
    try:
        config = load_config(parsed.config)
        authority = validate_config(config)
        queue = BenchmarkQueue(authority)
        queue.run()
        return 0
    except QueueSignal as error:
        if queue is not None:
            terminal_error: BaseException = error
            if queue._signal_notification_error is not None:
                terminal_error = QueueContractError(
                    signal.Signals(error.signum).name
                    + " received; initial child process-group notification failed: "
                    + queue._signal_notification_error
                )
            try:
                queue._terminate_active_child()
            except BaseException as cleanup_error:
                terminal_error = QueueContractError(
                    signal.Signals(error.signum).name
                    + " received; child process-group cleanup failed: "
                    + str(cleanup_error)
                )
            queue._write_terminal(
                success=False,
                exit_code=128 + error.signum,
                termination=signal.Signals(error.signum).name,
                error=terminal_error,
            )
        return 128 + error.signum
    except BaseException as error:
        if queue is not None:
            queue._terminate_active_child()
            queue._write_terminal(
                success=False,
                exit_code=1,
                termination="nonzero_exit",
                error=error,
            )
        print("same-data benchmark queue: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
