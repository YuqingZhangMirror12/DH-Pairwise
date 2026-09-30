from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import pytest

from experiments.rachel_n512_formal_30k import run_same_data_benchmark_queue as queue

if __name__ != "fixture_runtime":
    from experiments.rachel_n512_formal_30k import pairingnet_gpu_smoke


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = Path(__file__).with_name("run_same_data_benchmark_queue.py")
TEMPLATE = Path(__file__).with_name(
    "same_data_benchmark_queue_config.template.json"
)

PAIR_COMMIT = "e878b781b2b2065a4b7da09d2f639e8f0a35e97a"
SHRED_COMMIT = "0ae3b544ca4e910732f3f459b39aa15cdc62dbcb"
PAIR_METHOD = "pairingnet_rachel_mask_n512_upright_translation_v1"
SHRED_METHOD = "rachel_shreddingnet_maskonly_n512_upright_release0ae3b544_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_source(module: str) -> tuple[Path, bool] | None:
    candidate = ROOT.joinpath(*module.split("."))
    module_file = candidate.with_suffix(".py")
    package_file = candidate / "__init__.py"
    if module_file.is_file():
        return module_file, False
    if package_file.is_file():
        return package_file, True
    return None


def _relative_source_module(relative: str) -> str:
    path = Path(relative)
    parts = list(path.parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    else:
        parts[-1] = Path(parts[-1]).stem
    return ".".join(parts)


def _static_queue_child_source_closure() -> set[str]:
    pending = list(queue.QUEUE_CHILD_MODULES)
    observed: set[str] = set()
    while pending:
        module = pending.pop(0)
        source = _module_source(module)
        if source is None:
            continue
        path, is_package = source
        relative = path.relative_to(ROOT).as_posix()
        if relative in observed:
            continue
        observed.add(relative)
        parts = module.split(".")
        for depth in range(1, len(parts) + int(is_package)):
            parent = ".".join(parts[:depth])
            parent_source = _module_source(parent)
            if parent_source is not None:
                pending.append(parent)
        package = module if is_package else module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = package.split(".")
                    keep = len(package_parts) - (node.level - 1)
                    prefix = ".".join(package_parts[:keep])
                    base = prefix + ("." + node.module if node.module else "")
                else:
                    base = node.module or ""
                if base:
                    targets.append(base)
                    targets.extend(
                        base + "." + alias.name
                        for alias in node.names
                        if alias.name != "*"
                    )
            for target in targets:
                if target == "experiments" or target == "staging" or target.startswith(
                    ("experiments.", "staging.")
                ):
                    if _module_source(target) is not None:
                        pending.append(target)
    return observed


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _seal_source_tree(root: Path) -> None:
    for directory, directory_names, file_names in os.walk(
        root, topdown=False, followlinks=False
    ):
        directory_path = Path(directory)
        for name in file_names:
            path = directory_path / name
            if path.is_file() and not path.is_symlink():
                path.chmod(queue.IMMUTABLE_SOURCE_FILE_MODE)
        for name in directory_names:
            path = directory_path / name
            if path.is_dir() and not path.is_symlink():
                path.chmod(queue.IMMUTABLE_SOURCE_DIRECTORY_MODE)
        directory_path.chmod(queue.IMMUTABLE_SOURCE_DIRECTORY_MODE)


def _unseal_source_tree(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        directory_path.chmod(0o755)
        for name in directory_names:
            path = directory_path / name
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
        for name in file_names:
            path = directory_path / name
            if path.is_file() and not path.is_symlink():
                path.chmod(0o644)


def _execution_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "readonly-execution-snapshot"
    for relative in sorted(queue.REQUIRED_SOURCE_RELATIVE_PATHS):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    _seal_source_tree(snapshot)
    return snapshot


def _review_marker(
    *,
    method_id: str,
    runner_relative: str,
    runner_sha: str,
    contract_relative: str,
    contract_sha: str,
    smoke_relative: str | None = None,
    smoke_sha: str | None = None,
) -> dict[str, object]:
    marker: dict[str, object] = {
        "schema_version": "rachel-benchmark-runner-review/1.0",
        "status": "GO",
        "method_id": method_id,
        "p0_count": 0,
        "p1_count": 0,
        "review_scope_complete": True,
        "runner": {"relative_path": runner_relative, "sha256": runner_sha},
        "adaptation_contract": {
            "relative_path": contract_relative,
            "sha256": contract_sha,
        },
        "sealed_synthetic_accessed": False,
        "real_data_accessed": False,
    }
    if smoke_relative is not None:
        marker["smoke_source"] = {
            "relative_path": smoke_relative,
            "sha256": smoke_sha,
        }
    return marker


def _legacy_fake_benchmark_python(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def write_bound(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    payload["content_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

def option(values, name):
    return values[values.index(name) + 1]

def manifest_binding(values):
    root = Path(option(values, "--dataset-root"))
    return {
        split: hashlib.sha256((root / "pairs" / (split + ".jsonl")).read_bytes()).hexdigest()
        for split in ("train", "val")
    }

def ordered_binding():
    return {
        split: hashlib.sha256(("fixture-" + split + "-ordered-pair-ids").encode()).hexdigest()
        for split in ("train", "val")
    }

log = Path(os.environ["FAKE_QUEUE_LOG"])
with log.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\n")

if sys.argv[1:3] == ["-I", "-c"]:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps({
        "executable": str(Path(__file__).resolve()),
        "prefix": str(root),
        "base_prefix": "/fixture/base",
        "python_version": "3.fixture",
        "user_site_enabled": False,
        "packages": {
            "torch": "2.5.1+cu124",
            "torch-geometric": "2.fixture",
            "opencv-python": "4.fixture",
            "opencv-python-headless": None,
            "numpy": "2.fixture",
            "scipy": "1.14.1",
        },
        "imports": {
            "torch": "ok",
            "torch_geometric": "ok",
            "cv2": "ok",
            "numpy": "ok",
            "scipy": "ok",
        },
    }, sort_keys=True))
    raise SystemExit(0)

arguments = sys.argv[1:]
assert arguments.pop(0) == "-u"
assert arguments.pop(0) == "-m"
module = arguments.pop(0)
pair_runner = "staging.pairwise_v0_2.baselines.rachel_pairingnet_benchmark"
pair_smoke = "experiments.rachel_n512_formal_30k.pairingnet_gpu_smoke"
shred_runner = "staging.pairwise_v0_2.baselines.rachel_shreddingnet_benchmark"
pair_sha = os.environ["RACHEL_QUEUE_PAIRINGNET_RUNNER_SHA256"]
pair_doc = os.environ["RACHEL_QUEUE_PAIRINGNET_CONTRACT_SHA256"]
shred_sha = os.environ["RACHEL_QUEUE_SHREDDINGNET_RUNNER_SHA256"]
shred_doc = os.environ["RACHEL_QUEUE_SHREDDINGNET_CONTRACT_SHA256"]

if module == pair_runner and "--audit-only" in arguments:
    manifests = manifest_binding(arguments)
    print(json.dumps({
        "schema_version": "rachel-pairingnet-preflight-audit/1.0",
        "population_audit": {
            "status": "train_val_population_audited",
            "formal_population_required": True,
            "population": {
                "train": {"rows": 24000, "positive": 12000, "negative": 12000, "split_unit_count": 10},
                "val": {"rows": 3000, "positive": 1500, "negative": 1500, "split_unit_count": 5},
            },
            "manifests": {
                "train_sha256": manifests["train"],
                "val_sha256": manifests["val"],
            },
            "parent_lineage_disjoint": True,
            "sealed_synthetic_accessed": False,
            "real_data_accessed": False,
        },
        "official_source_audit": {"commit": "e878b781b2b2065a4b7da09d2f639e8f0a35e97a"},
    }))
elif module == pair_smoke:
    if os.environ.get("FAKE_FAIL_STAGE") == "pairing_smoke":
        raise SystemExit(17)
    write_bound(option(arguments, "--output"), {
        "schema_version": "rachel-pairingnet-gpu-smoke/1.0",
        "status": "complete_train_only_one_formal_batch_optimizer_step",
        "method_id": "pairingnet_rachel_mask_n512_upright_translation_v1",
        "official_commit": "e878b781b2b2065a4b7da09d2f639e8f0a35e97a",
        "adapter_source_sha256": pair_sha,
        "adaptation_contract_sha256": pair_doc,
        "dataset_manifest_sha256": {
            "train_sha256": manifest_binding(arguments)["train"],
            "val_sha256": manifest_binding(arguments)["val"],
        },
        "batch_size": 20,
        "precision": option(arguments, "--precision"),
        "scope": {
            "formal_training_started": False,
            "sealed_synthetic_or_real_opened": False,
            "optimizer_steps": 1,
        },
    })
    print("{}")
elif module == pair_runner:
    output = Path(option(arguments, "--output-root"))
    output.mkdir()
    _ = (output / "completion_receipt.json").write_text(json.dumps({
        "status": "train_validation_complete",
        "method_id": "pairingnet_rachel_mask_n512_upright_translation_v1",
        "official_commit": "e878b781b2b2065a4b7da09d2f639e8f0a35e97a",
        "adapter_source_sha256": pair_sha,
        "adaptation_contract_sha256": pair_doc,
        "population_audit": {
            "manifests": {
                "train_sha256": manifest_binding(arguments)["train"],
                "val_sha256": manifest_binding(arguments)["val"],
            },
        },
        "convergence_demonstrated": True,
        "sealed_synthetic_accessed": False,
        "real_data_accessed": False,
    }), encoding="utf-8")
    if os.environ.get("FAKE_MUTATE_AFTER_PAIR_TRAIN") == "1":
        dataset = Path(option(arguments, "--dataset-root"))
        with (dataset / "pairs/val.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("mutated-after-pair-train\n")
    print("{}")
elif module == shred_runner and arguments[0] == "audit":
    write_bound(option(arguments, "--output"), {
        "schema_version": "rachel-shreddingnet-maskonly-benchmark/1.0",
        "status": "audit_complete_train_val_only",
        "method_id": "rachel_shreddingnet_maskonly_n512_upright_release0ae3b544_v1",
        "adapter_identity": {
            "adapter_source_sha256": shred_sha,
            "adaptation_document_sha256": shred_doc,
        },
        "dataset": {
            "counts": {
                "train": {"rows": 24000, "positive": 12000, "negative": 12000},
                "val": {"rows": 3000, "positive": 1500, "negative": 1500},
            },
            "formal_counts_required": True,
            "lineage_disjoint": True,
            "opened_manifests": ["pairs/train.jsonl", "pairs/val.jsonl"],
            "sealed_test_manifest_opened": False,
            "real_data_opened": False,
            "manifest_content_sha256": manifest_binding(arguments),
        },
        "official_source": {"commit": "0ae3b544ca4e910732f3f459b39aa15cdc62dbcb"},
    })
    print("{}")
elif module == shred_runner and arguments[0] == "gpu-smoke":
    runtime = {
        "coarse_microbatch": int(option(arguments, "--coarse-microbatch")),
        "matching_microbatch": int(option(arguments, "--matching-microbatch")),
        "classify_microbatch": int(option(arguments, "--classify-microbatch")),
        "amp": "--no-amp" not in arguments,
    }
    manifests = manifest_binding(arguments)
    ordered = ordered_binding()

    def stage_row(stage, effective_batch_size, metrics):
        microbatch = runtime[stage + "_microbatch"]
        return {
            "effective_batch_size": effective_batch_size,
            "configured_release_effective_batch_size": effective_batch_size,
            "gpu_microbatch_size": microbatch,
            "nominal_accumulation_steps": (
                effective_batch_size + microbatch - 1
            ) // microbatch,
            "amp_enabled": runtime["amp"],
            "elapsed_seconds": 1.0,
            "effective_pairs_per_second": float(effective_batch_size),
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
            "metrics": metrics,
        }

    stages = {
        "coarse": stage_row("coarse", 54, {
            "loss": 1.0,
            "infor_loss": 1.0,
            "top1_recall": 0.1,
            "top5_recall": 0.2,
        }),
        "matching": stage_row("matching", 36, {
            "loss": 1.0,
            "positive_loss": 1.0,
        }),
        "classify": stage_row("classify", 20, {
            "loss": 1.0,
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        }),
    }
    defect = os.environ.get("FAKE_SHREDDING_SMOKE_DEFECT")
    if defect == "list_schema":
        stages = [dict({"stage": name}, **row) for name, row in stages.items()]
    elif defect == "bad_batch":
        stages["matching"]["effective_batch_size"] = 35
    elif defect == "nonfinite_metric":
        stages["classify"]["metrics"]["f1"] = "nan"

    smoke_receipt = {
        "schema_version": "rachel-shreddingnet-gpu-smoke/1.0",
        "status": "complete_train_only_release_effective_step_smoke",
        "method_id": "rachel_shreddingnet_maskonly_n512_upright_release0ae3b544_v1",
        "device": option(arguments, "--device"),
        "device_name": "fixture-cuda",
        "recipe": {
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
        },
        "official_commit": "0ae3b544ca4e910732f3f459b39aa15cdc62dbcb",
        "runtime_batch_config": runtime,
        "dataset_binding": {
            "manifest_content_sha256": manifests,
            "ordered_pair_ids_sha256": ordered,
        },
        "adapter_identity": {
            "adapter_source_filename": "rachel_shreddingnet_benchmark.py",
            "adapter_source_sha256": shred_sha,
            "adaptation_document_filename": "RACHEL_SHREDDINGNET_BENCHMARK.md",
            "adaptation_document_sha256": shred_doc,
        },
        "train_val_provenance": {
            "adapter_source_sha256": shred_sha,
            "adaptation_document_sha256": shred_doc,
            "train_manifest_content_sha256": manifests["train"],
            "val_manifest_content_sha256": manifests["val"],
            "train_ordered_pair_ids_sha256": ordered["train"],
            "val_ordered_pair_ids_sha256": ordered["val"],
        },
        "runtime_preflight": {
            "preflight_kind": "rachel_shreddingnet_real_models_train_probe",
            "device": option(arguments, "--device"),
            "torch_version": "2.fixture",
            "torch_geometric_version": "2.fixture",
            "real_stage_model_constructed_and_forwarded": {
                "coarse": True,
                "matching": True,
                "classify": True,
            },
            "stage_output_shapes": {
                "coarse": [[54, 64], [54, 64]],
                "matching": [[36, 512, 512]],
                "classify": [[20], [20], [20]],
            },
            "train_probe_pair_ids_sha256": hashlib.sha256(
                b"fixture-train-probe-pair-ids"
            ).hexdigest(),
            "train_positive_and_negative_artifacts_decoded": True,
            "train_val_manifests_reverified_after_decode": True,
            "output_created_or_written": False,
        },
        "scope": {
            "train_manifest_only_for_tensor_loading": True,
            "validation_tensor_loading": False,
            "formal_training_started": False,
            "sealed_test_or_real_opened": False,
            "random_smoke_models_not_checkpoints": True,
        },
        "stages": stages,
        "dataset_manifest_sha256": manifests,
        "numeric_contract": {
            "dual_softmax_probability_compute_dtype": "float32",
            "amp_logits_promoted_before_mask_and_softmax": True,
            "focal_logarithm_epsilon_semantics": (
                "log_p_plus_1e-9_and_log_1_minus_p_plus_1e-9"
            ),
        },
    }
    if defect == "bad_adapter_sha":
        smoke_receipt["adapter_identity"]["adapter_source_sha256"] = "0" * 64
    write_bound(option(arguments, "--output"), smoke_receipt)
    if defect == "bad_content_sha":
        output_path = Path(option(arguments, "--output"))
        written = json.loads(output_path.read_text(encoding="utf-8"))
        written["content_sha256"] = "0" * 64
        output_path.write_text(json.dumps(written, sort_keys=True), encoding="utf-8")
    print("{}")
elif module == shred_runner and arguments[0] == "train":
    output = Path(option(arguments, "--output-root"))
    output.mkdir()
    write_bound(output / "train_val_freeze.json", {
        "schema_version": "rachel-shreddingnet-train-val-freeze/1.0",
        "checkpoint_kind": "rachel_shreddingnet_train_val_freeze",
        "status": "complete_train_val_frozen_no_test_or_real",
        "method_id": "rachel_shreddingnet_maskonly_n512_upright_release0ae3b544_v1",
        "official_commit": "0ae3b544ca4e910732f3f459b39aa15cdc62dbcb",
        "adapter_identity": {
            "adapter_source_sha256": shred_sha,
            "adaptation_document_sha256": shred_doc,
        },
        "dataset_binding": {
            "manifest_content_sha256": manifest_binding(arguments),
        },
        "scope": {"sealed_test_or_real_opened": False, "train_val_only": True},
    })
    print("{}")
else:
    raise SystemExit("unexpected fixture command: " + repr((module, arguments)))
''',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _bound(value: dict[str, object]) -> dict[str, object]:
    payload = dict(value)
    payload["content_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def _fixture_runtime_no_site(
    environment_root: Path, benchmark_python: Path
) -> dict[str, object]:
    base_root = environment_root.parent / "fixture-base-runtime"
    resolved_root = environment_root.parent / "fixture-resolved-runtime"
    sites = {
        "base": base_root / "lib/python3.8/site-packages",
        "resolved_base": resolved_root / "lib/python3.8/site-packages",
        "target": environment_root / "lib/python3.8/site-packages",
    }
    for site_directory in sites.values():
        site_directory.mkdir(parents=True, exist_ok=True)

    distributions = {
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

    def binding(path: Path, prefix: Path, role: str) -> dict[str, object]:
        metadata = path.stat()
        return {
            "canonical_path": str(path.resolve()),
            "prefix": str(prefix.resolve()),
            "relative_path": path.resolve().relative_to(prefix.resolve()).as_posix(),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "role": role,
            "audit_content_sha256": hashlib.sha256(
                (str(path) + role).encode()
            ).hexdigest(),
        }

    bindings = {
        "base": [binding(sites["base"], base_root, "audit_0")],
        "resolved_base": [
            binding(sites["resolved_base"], resolved_root, "audit_0")
        ],
        "target": [
            binding(sites["target"], environment_root, "audit_0"),
            binding(sites["resolved_base"], resolved_root, "audit_1"),
        ],
    }
    contexts = {}
    for name in ("base", "resolved_base", "target"):
        logical_root = {
            "base": base_root,
            "resolved_base": resolved_root,
            "target": environment_root,
        }[name]
        origins = []
        for distribution in sorted(distributions[name]):
            module = distribution.replace("-", "_")
            site = sites["target"] if name == "target" else sites[name]
            origins.append(
                {
                    "distribution": distribution,
                    "distribution_version": "fixture",
                    "distribution_location": str(site),
                    "module": module,
                    "import_origin": str(site / module / "__init__.py"),
                }
            )
        contexts[name] = _bound(
            {
                "runtime_no_site": True,
                "safe_startup_identity": {
                    "executable": str(benchmark_python.resolve()),
                    "prefix": str(logical_root.resolve()),
                    "base_prefix": str(base_root.resolve()),
                },
                "logical_runtime_identity": {
                    "prefix": str(logical_root.resolve()),
                    "base_prefix": str(base_root.resolve()),
                },
                "no_site_sys_path": [str(base_root.resolve() / "fixture-stdlib")],
                "injected_site_directories": bindings[name],
                "import_origins": origins,
                "torchvision_dynamic_module_admission": (
                    queue._successful_torchvision_dynamic_admission_evidence()
                ),
                "pth_files_present_but_never_executed": [],
            }
        )
    return _bound(
        {
            "schema_version": queue.ENVIRONMENT_RUNTIME_NO_SITE_SCHEMA,
            "runtime_no_site": True,
            "interpreter_flags": ["-I", "-S", "-B"],
            "bootstrap": {
                "configuration_schema": (
                    queue.ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SCHEMA
                ),
                "launcher_source_sha256": (
                    queue.ENVIRONMENT_EXPLICIT_SITE_STDIN_LAUNCHER_SHA256
                ),
                "source_sha256": (
                    queue.ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SHA256
                ),
                "torchvision_dynamic_admission_source_sha256": (
                    queue.ENVIRONMENT_TORCHVISION_DYNAMIC_ADMISSION_SOURCE_SHA256
                ),
                "torchvision_dynamic_admission_configuration": (
                    queue._torchvision_dynamic_admission_configuration(True)
                ),
                "torchvision_dynamic_admission_runtime_only": True,
                "pip_dynamic_preload_disabled": True,
                "source_transport": "sha256_bound_stdin",
                "launcher_sys_path_policy": (
                    "content_sha256_bound_exact_replacement_before_filesystem_import"
                ),
                "pip_runner_source_sha256": (
                    queue.ENVIRONMENT_PIP_NO_SITE_RUNNER_SHA256
                ),
                "payload_source_sha256": queue.ENVIRONMENT_RUNTIME_PAYLOAD_SHA256,
                "configuration_content_sha256_bound": True,
                "preimport_sys_path_exact": True,
                "payload_import_sys_path_audited": True,
                "site_directories_fd_anchored": True,
                "site_main_and_pth_processing_blocked": True,
                "sitecustomize_and_usercustomize_never_imported": True,
            },
            **contexts,
        }
    )


@contextmanager
def _explicit_site_subprocess(
    payload: str,
    arguments: list[str],
    explicit_site_directories: list[Path] | None = None,
):
    safe = json.loads(
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                (
                    "import json,os,sys;print(json.dumps({"
                    "'executable':os.path.realpath(sys.executable),"
                    "'prefix':os.path.realpath(sys.prefix),"
                    "'base_prefix':os.path.realpath(sys.base_prefix),"
                    "'sys_path':[os.path.realpath(v) for v in sys.path if v]}))"
                ),
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
    )
    no_site_paths = set(safe["sys_path"])
    site_directories = []
    seen = set()
    candidates = (
        explicit_site_directories
        if explicit_site_directories is not None
        else [Path(entry) for entry in sys.path if entry]
    )
    argument_root = (
        Path(arguments[0]).resolve()
        if arguments and Path(arguments[0]).is_dir()
        else None
    )
    if argument_root is not None:
        candidates = [argument_root, *candidates]
    for entry in candidates:
        path = entry.resolve()
        text = str(path)
        if not path.is_dir() or text in no_site_paths or text in seen:
            continue
        if (
            explicit_site_directories is None
            and path != argument_root
            and "site-packages" not in path.parts
        ):
            continue
        site_directories.append(path)
        seen.add(text)
    if not site_directories:
        raise AssertionError("fixture interpreter has no explicit site directories")
    descriptors = []
    bindings = []
    try:
        for index, site_directory in enumerate(site_directories):
            descriptor = os.open(
                str(site_directory),
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            descriptors.append(descriptor)
            metadata = os.fstat(descriptor)
            descriptor_path = (
                "/proc/{}/fd/{}".format(os.getpid(), descriptor)
                if sys.platform.startswith("linux")
                else "/dev/fd/{}".format(descriptor)
            )
            bindings.append(
                {
                    "canonical_path": str(site_directory),
                    "descriptor_path": descriptor_path,
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "role": "audit_" + str(index),
                    "audit_content_sha256": hashlib.sha256(
                        str(site_directory).encode()
                    ).hexdigest(),
                }
            )
        configuration = _bound(
            {
                "schema_version": queue.ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SCHEMA,
                "startup": {
                    "executable": safe["executable"],
                    "prefix": safe["prefix"],
                    "base_prefix": safe["base_prefix"],
                },
                "runtime": {
                    "prefix": safe["prefix"],
                    "base_prefix": safe["base_prefix"],
                },
                "no_site_sys_path": safe["sys_path"],
                "sites": bindings,
                "torchvision_dynamic_admission": (
                    queue._torchvision_dynamic_admission_configuration(False)
                ),
            }
        )
        sources = queue._reviewed_environment_runtime_sources(
            ROOT / queue.ENVIRONMENT_BUILDER_RELATIVE_PATH
        )
        bootstrap = sources["bootstrap"]
        if sys.platform == "darwin":
            # Darwin directory FDs are identity-checkable but not traversable.
            # Formal CUDA execution remains exact Linux proc-FD execution.
            bootstrap = bootstrap.replace(
                "sys.path.append(descriptor_path)",
                "sys.path.append(canonical)",
                1,
            )
        yield (
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                sources["launcher"],
                str(len(safe["sys_path"])),
                *safe["sys_path"],
                hashlib.sha256(_canonical(safe["sys_path"])).hexdigest(),
                hashlib.sha256(bootstrap.encode("utf-8")).hexdigest(),
                json.dumps(configuration, sort_keys=True, separators=(",", ":")),
                payload,
                *arguments,
            ],
            tuple(descriptors),
            bootstrap,
        )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _write_plain(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_bound(path: Path, value: dict[str, object]) -> dict[str, object]:
    payload = _bound(value)
    _write_plain(path, payload)
    return payload


def _option(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def _manifest_hashes(dataset_root: Path) -> dict[str, str]:
    return {
        split: _sha(dataset_root / "pairs" / (split + ".jsonl"))
        for split in ("train", "val")
    }


def _ordered_pair_hashes(dataset_root: Path) -> dict[str, str]:
    result = {}
    for split in ("train", "val"):
        rows = [
            json.loads(line)
            for line in (dataset_root / "pairs" / (split + ".jsonl"))
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        result[split] = hashlib.sha256(
            _canonical([row["pair_id"] for row in rows])
        ).hexdigest()
    return result


def _fake_asset_create(arguments: list[str]) -> None:
    source = Path(_option(arguments, "--dataset-root"))
    freeze = Path(_option(arguments, "--output-dir"))
    frozen = freeze / "frozen_data"
    (frozen / "pairs").mkdir(parents=True)
    for split in ("train", "val"):
        shutil.copyfile(
            source / "pairs" / (split + ".jsonl"),
            frozen / "pairs" / (split + ".jsonl"),
        )
    dummy = frozen / "model/masks_800/fixture.bin"
    dummy.parent.mkdir(parents=True)
    dummy.write_bytes(b"fixture-semantic-mask-bytes")
    source_manifests = _manifest_hashes(source)
    frozen_manifests = _manifest_hashes(frozen)
    ordered = _ordered_pair_hashes(frozen)
    files = [
        frozen / "pairs/train.jsonl",
        frozen / "pairs/val.jsonl",
        dummy,
    ]
    semantic = {
        "manifest_content_sha256": source_manifests,
        "ordered_pair_ids_sha256": ordered,
        "canonical_ordered_pairs_sha256": ordered,
        "fixture_dummy_sha256": _sha(dummy),
    }
    semantic_sha = hashlib.sha256(_canonical(semantic)).hexdigest()
    inventory_path = freeze / "train_val_semantic_asset_inventory.json"
    _write_plain(
        inventory_path,
        {
            "semantic_inventory": semantic,
            "semantic_inventory_content_sha256": semantic_sha,
        },
    )
    total_size = sum(path.stat().st_size for path in files)
    receipt = {
        "schema_version": "rachel-train-val-semantic-asset-freeze/3.0",
        "status": "complete_self_contained_train_val_semantic_asset_freeze",
        "dataset_root": str(source),
        "frozen_dataset": {
            "relative_path": "frozen_data",
            "absolute_path": str(frozen),
            "runner_dataset_root_required": True,
            "freeze_root_device": freeze.stat().st_dev,
            "freeze_root_inode": freeze.stat().st_ino,
            "view_root_device": frozen.stat().st_dev,
            "view_root_inode": frozen.stat().st_ino,
            "manifest_content_sha256": frozen_manifests,
            "ordered_pair_ids_sha256": ordered,
            "canonical_ordered_pairs_sha256": ordered,
            "file_count": len(files),
            "total_size_bytes": total_size,
            "file_mode_octal": "0444",
            "directory_mode_octal": "0555",
        },
        "inventory": {
            "filename": inventory_path.name,
            "sha256": _sha(inventory_path),
            "size_bytes": inventory_path.stat().st_size,
            "semantic_inventory_content_sha256": semantic_sha,
        },
        "publication_contract": {
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
        },
        "scope": {
            "opened_pair_manifests": ["pairs/train.jsonl", "pairs/val.jsonl"],
            "frozen_runner_dataset_root": str(frozen),
            "sealed_test_manifest_opened": False,
            "sealed_test_assets_opened": False,
            "real_data_opened": False,
            "rgb_opened": False,
        },
    }
    _write_plain(
        freeze / "train_val_semantic_asset_freeze_receipt.json", receipt
    )
    print(json.dumps({"status": "created"}, sort_keys=True))


def _fake_asset_verify(arguments: list[str], controls: dict[str, object]) -> None:
    source = Path(_option(arguments, "--dataset-root"))
    freeze = Path(_option(arguments, "--freeze-dir"))
    frozen = freeze / "frozen_data"
    receipt_path = freeze / "train_val_semantic_asset_freeze_receipt.json"
    inventory_path = freeze / "train_val_semantic_asset_inventory.json"
    _ = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    semantic = inventory["semantic_inventory"]
    dummy = frozen / "model/masks_800/fixture.bin"
    if _sha(dummy) != semantic["fixture_dummy_sha256"]:
        raise SystemExit("fixture semantic asset changed")
    if controls.get("replace_pair_smoke_after_gate"):
        smoke = freeze.parent / "pairingnet_gpu_smoke.json"
        gate = freeze.parent / "pairingnet_gpu_smoke_gate.json"
        if smoke.is_file() and gate.is_file():
            value = json.loads(smoke.read_text(encoding="utf-8"))
            value.pop("content_sha256")
            value["elapsed_seconds"] = float(value["elapsed_seconds"]) + 0.25
            _write_bound(smoke, value)
    manifests = _manifest_hashes(frozen)
    ordered = _ordered_pair_hashes(frozen)
    files = [
        frozen / "pairs/train.jsonl",
        frozen / "pairs/val.jsonl",
        dummy,
    ]
    result = {
        "schema_version": "rachel-train-val-semantic-asset-verification/3.0",
        "status": "verified_train_val_semantic_assets_unchanged",
        "dataset_root": str(source),
        "freeze_dir": str(freeze),
        "frozen_dataset_root": str(frozen),
        "freeze_receipt_sha256": _sha(receipt_path),
        "freeze_receipt_size_bytes": receipt_path.stat().st_size,
        "inventory_file_sha256": _sha(inventory_path),
        "inventory_file_size_bytes": inventory_path.stat().st_size,
        "semantic_inventory_content_sha256": inventory[
            "semantic_inventory_content_sha256"
        ],
        "source_asset_file_count": len(files),
        "source_asset_total_size_bytes": sum(path.stat().st_size for path in files),
        "frozen_asset_file_count": len(files),
        "frozen_asset_total_size_bytes": sum(path.stat().st_size for path in files),
        "manifest_content_sha256": manifests,
        "ordered_pair_ids_sha256": ordered,
        "canonical_ordered_pairs_sha256": ordered,
        "freeze_root_device": freeze.stat().st_dev,
        "freeze_root_inode": freeze.stat().st_ino,
        "view_root_device": frozen.stat().st_dev,
        "view_root_inode": frozen.stat().st_ino,
        "rehash_all_frozen_dataset_bytes": True,
        "source_snapshot_reverified": True,
        "runner_dataset_root_is_self_contained_frozen_view": True,
        "same_freeze_root_fd_held_for_full_verification": True,
        "final_complete_tree_revalidated": True,
        "frozen_view_regular_file_link_count_required": 1,
        "sealed_test_manifest_opened": False,
        "sealed_test_assets_opened": False,
        "real_data_opened": False,
        "rgb_opened": False,
    }
    if result["freeze_receipt_sha256"] != _option(
        arguments, "--expected-freeze-receipt-sha256"
    ):
        raise SystemExit("fixture freeze receipt pin changed")
    if result["semantic_inventory_content_sha256"] != _option(
        arguments, "--expected-semantic-inventory-content-sha256"
    ):
        raise SystemExit("fixture semantic inventory pin changed")
    print(json.dumps(result, sort_keys=True))


def _fake_upstream_summary(arguments: list[str]) -> dict[str, object]:
    dataset = Path(_option(arguments, "--dataset-root"))
    nroot = Path(_option(arguments, "--n512-run"))
    mroot = Path(_option(arguments, "--matched-run"))
    manifests = _manifest_hashes(dataset)
    sha = hashlib.sha256
    winner = {
        "epoch": 21,
        "checkpoint_sha256": sha(b"winner").hexdigest(),
        "threshold_content_sha256": sha(b"threshold").hexdigest(),
    }
    n_winner = {
        **winner,
        "stop_reason": "validation_early_stop",
        "convergence_claim": "validation_plateau_under_declared_rule",
    }
    model_config = {"contour_cap": 512, "upright_translation_only": True}
    loss_config = {"translation_supervision": True}
    model_sha = sha(_canonical(model_config)).hexdigest()
    loss_sha = sha(_canonical(loss_config)).hexdigest()
    order_sha = sha(b"fixture-order").hexdigest()
    return {
        "schema_version": "rachel-benchmark-queue-upstream-verification/1.0",
        "status": "verified_formal_upstream_winners_and_alignment",
        "n512": {
            "run_root": str(nroot),
            "receipt_sha256": _sha(nroot / "run_receipt.json"),
            "schema_version": "rachel-n512-train-run/1.0",
            "status": "complete_train_validation_only",
            "fingerprint_sha256": sha(b"n512-fingerprint").hexdigest(),
            "config": {
                "arms": ["coarse_only", "full_n512"],
                "seed": 260831,
                "epochs": 128,
                "batch_size": 16,
                "continuation": {
                    "min_total_epochs": 20,
                    "patience": 12,
                    "early_stop_reads": ["val"],
                    "checkpoint_selection_reads": ["val"],
                },
            },
            "population": {
                "train_total": 24000,
                "val_total": 3000,
                "train_rows_per_epoch": 24000,
                "val_rows": 3000,
            },
            "winners": {
                "coarse_only": dict(n_winner),
                "full_n512": dict(n_winner),
            },
        },
        "matched_mm": {
            "run_root": str(mroot),
            "receipt_sha256": _sha(mroot / "run_receipt.json"),
            "schema_version": "rachel-matched-historical-mm-train/1.0",
            "status": "complete_train_validation_only",
            "fingerprint_sha256": sha(b"matched-fingerprint").hexdigest(),
            "config": {
                "seed": 260831,
                "batch_size": 16,
                "max_total_epochs": 128,
                "min_total_epochs": 20,
                "patience": 12,
                "source_splits_opened": ["train", "val"],
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
            "population": {
                "train_total": 24000,
                "val_total": 3000,
                "train_positive": 12000,
                "val_positive": 1500,
            },
            "stop_reason": "validation_early_stop",
            "convergence_claim": "validation_plateau_under_declared_rule",
            "winners": {
                "matched_mm_converged": dict(winner),
                "matched_mm_same_exposure_epoch5": {**winner, "epoch": 5},
            },
        },
        "canonical_config_authority": {
            "comparison": "complete_dataclass_field_mapping_exact_equality",
            "model_config": model_config,
            "model_config_sha256": model_sha,
            "loss_config": loss_config,
            "loss_config_sha256": loss_sha,
            "model_config_sha256_by_arm": {
                "coarse_only": model_sha,
                "full_n512": model_sha,
            },
            "loss_config_sha256_by_arm": {
                "coarse_only": loss_sha,
                "full_n512": loss_sha,
            },
            "all_winner_configs_exactly_equal": True,
        },
        "matched_training_hash_evidence": {
            "claim_level": (
                "same_frozen_train_val_manifest_content_and_exact_validation_pair_order;"
                "per_epoch_training_presentation_order_not_provable"
            ),
            "canonical_dataset_root": str(dataset),
            "canonical_seed": 260831,
            "train_manifest": {
                "path": str(dataset / "pairs/train.jsonl"),
                "content_sha256": manifests["train"],
                "bound_to_both_training_fingerprints": True,
                "pair_count": 24000,
                "pair_order_fingerprint_sha256": order_sha,
            },
            "validation_manifest": {
                "path": str(dataset / "pairs/val.jsonl"),
                "content_sha256": manifests["val"],
                "bound_to_both_training_fingerprints": True,
                "pair_count": 3000,
                "pair_order_fingerprint_sha256": order_sha,
                "exact_order_equal_to_all_four_validation_score_artifacts": True,
            },
            "training_fingerprints_recomputed": {
                "rachel_n512": sha(b"n512-fingerprint").hexdigest(),
                "matched_mm": sha(b"matched-fingerprint").hexdigest(),
            },
            "validation_pair_order": {
                "exact_order_equal_across_four_frozen_thresholds": True,
                "pair_count": 3000,
                "pair_order_fingerprint_sha256": order_sha,
                "sources": {
                    name: {"artifact_sha256": hashlib.sha256(name.encode()).hexdigest()}
                    for name in (
                        "coarse_only",
                        "full_n512",
                        "matched_mm_converged",
                        "matched_mm_same_exposure_epoch5",
                    )
                },
            },
            "limitations": {
                "exact_per_epoch_training_pair_presentation_order_saved": False,
                "exact_per_epoch_training_pair_presentation_order_claimed_equal": False,
            },
        },
        "dataset_root": str(dataset),
        "formal_convergence_verified": True,
        "winner_checkpoints_strict_loaded_cpu": True,
        "test_or_real_opened": False,
    }


def _fake_pair_smoke(arguments: list[str], source_root: Path, controls: dict[str, object]) -> None:
    if controls.get("fail_stage") == "pairing_smoke":
        raise SystemExit(17)
    dataset = Path(_option(arguments, "--dataset-root"))
    train_rows = [
        json.loads(line)
        for line in (dataset / "pairs/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[:20]
    ]
    rows = [
        {"ordinal": ordinal, "pair_id": row["pair_id"], "label": row["label"]}
        for ordinal, row in enumerate(train_rows)
    ]
    pair_ids = [row["pair_id"] for row in rows]
    positives = [row["pair_id"] for row in rows if row["label"]]
    negatives = [row["pair_id"] for row in rows if not row["label"]]
    official = Path(_option(arguments, "--official-source-root"))
    receipt = {
        "schema_version": "rachel-pairingnet-gpu-smoke/1.0",
        "status": "complete_train_only_one_formal_batch_optimizer_step",
        "method_id": PAIR_METHOD,
        "official_commit": PAIR_COMMIT,
        "adapter_source_sha256": _sha(
            source_root / queue.METHODS["pairingnet"]["runner_relative_path"]
        ),
        "adaptation_contract_sha256": _sha(
            source_root
            / queue.METHODS["pairingnet"]["adaptation_contract_relative_path"]
        ),
        "dataset_manifest_sha256": {
            "train_sha256": _manifest_hashes(dataset)["train"],
            "val_sha256": _manifest_hashes(dataset)["val"],
        },
        "batch_size": 20,
        "device": _option(arguments, "--device"),
        "device_name": "fixture-cuda",
        "precision": _option(arguments, "--precision"),
        "num_workers": int(_option(arguments, "--num-workers")),
        "elapsed_seconds": 1.0,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        "scalars": {
            "total": 1.0,
            "matching_focal": 0.3,
            "pair_bce": 0.7,
            "gradient_norm": 1.1,
        },
        "batch_order": {
            "schema_version": "rachel-pairingnet-gpu-smoke-batch-order/1.0",
            "seed": 260831,
            "epoch_one_based": 1,
            "rows": rows,
            "ordered_pair_ids_sha256": hashlib.sha256(_canonical(pair_ids)).hexdigest(),
            "ordered_pair_labels_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
            "positive_pair_ids": positives,
            "negative_pair_ids": negatives,
            "contains_positive_and_negative": True,
        },
        "official_source_audit": {
            "schema_version": "pairingnet-official-source-audit/1.0",
            "status": "exact_clean_checkout_verified",
            "checkout_root": str(official),
            "repository": queue.PAIRINGNET_OFFICIAL_REPOSITORY,
            "origin_reported": queue.PAIRINGNET_OFFICIAL_REPOSITORY,
            "commit": PAIR_COMMIT,
            "clean": True,
            "key_file_sha256": queue.PAIRINGNET_OFFICIAL_KEY_FILE_SHA256,
        },
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
    defect = controls.get("pair_smoke_defect")
    if defect == "device":
        receipt["device"] = "cpu"
    elif defect == "workers":
        receipt["num_workers"] = 9
    elif defect == "nonfinite":
        receipt["scalars"]["total"] = "nan"
    elif defect == "vram":
        receipt["peak_reserved_bytes"] = 50
    elif defect == "official_audit":
        receipt["official_source_audit"]["clean"] = False
    elif defect == "order":
        receipt["batch_order"]["contains_positive_and_negative"] = False
    _write_bound(Path(_option(arguments, "--output")), receipt)
    print("{}")


def _fake_shred_smoke(arguments: list[str], source_root: Path, controls: dict[str, object]) -> None:
    dataset = Path(_option(arguments, "--dataset-root"))
    runtime = {
        "coarse_microbatch": int(_option(arguments, "--coarse-microbatch")),
        "matching_microbatch": int(_option(arguments, "--matching-microbatch")),
        "classify_microbatch": int(_option(arguments, "--classify-microbatch")),
        "amp": "--no-amp" not in arguments,
    }

    def stage_row(stage: str, effective: int, metrics: dict[str, float]) -> dict[str, object]:
        microbatch = runtime[stage + "_microbatch"]
        return {
            "effective_batch_size": effective,
            "configured_release_effective_batch_size": effective,
            "gpu_microbatch_size": microbatch,
            "nominal_accumulation_steps": (effective + microbatch - 1) // microbatch,
            "amp_enabled": runtime["amp"],
            "elapsed_seconds": 1.0,
            "effective_pairs_per_second": float(effective),
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
            "metrics": metrics,
        }

    stages = {
        "coarse": stage_row(
            "coarse", 54, {"loss": 1.0, "infor_loss": 1.0, "top1_recall": 0.1, "top5_recall": 0.2}
        ),
        "matching": stage_row("matching", 36, {"loss": 1.0, "positive_loss": 1.0}),
        "classify": stage_row(
            "classify", 20, {"loss": 1.0, "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}
        ),
    }
    defect = controls.get("shred_smoke_defect")
    if defect == "list_schema":
        stages = [dict({"stage": name}, **row) for name, row in stages.items()]
    elif defect == "bad_batch":
        stages["matching"]["effective_batch_size"] = 35
    elif defect == "nonfinite_metric":
        stages["classify"]["metrics"]["f1"] = "nan"
    method = queue.METHODS["shreddingnet"]
    adapter_sha = _sha(source_root / method["runner_relative_path"])
    document_sha = _sha(source_root / method["adaptation_contract_relative_path"])
    manifests = _manifest_hashes(dataset)
    ordered = _ordered_pair_hashes(dataset)
    receipt = {
        "schema_version": "rachel-shreddingnet-gpu-smoke/1.0",
        "status": "complete_train_only_release_effective_step_smoke",
        "method_id": SHRED_METHOD,
        "device": _option(arguments, "--device"),
        "device_name": "fixture-cuda",
        "recipe": queue.SHREDDINGNET_RELEASE_RECIPE,
        "runtime_batch_config": runtime,
        "adapter_identity": {
            "adapter_source_filename": "rachel_shreddingnet_benchmark.py",
            "adapter_source_sha256": adapter_sha,
            "adaptation_document_filename": "RACHEL_SHREDDINGNET_BENCHMARK.md",
            "adaptation_document_sha256": document_sha,
        },
        "dataset_binding": {
            "manifest_content_sha256": manifests,
            "ordered_pair_ids_sha256": ordered,
        },
        "train_val_provenance": {
            "adapter_source_sha256": adapter_sha,
            "adaptation_document_sha256": document_sha,
            "train_manifest_content_sha256": manifests["train"],
            "val_manifest_content_sha256": manifests["val"],
            "train_ordered_pair_ids_sha256": ordered["train"],
            "val_ordered_pair_ids_sha256": ordered["val"],
        },
        "runtime_preflight": {
            "preflight_kind": "rachel_shreddingnet_real_models_train_probe",
            "device": _option(arguments, "--device"),
            "torch_version": "2.fixture",
            "torch_geometric_version": "2.fixture",
            "real_stage_model_constructed_and_forwarded": {"coarse": True, "matching": True, "classify": True},
            "stage_output_shapes": {"coarse": [[54, 64]], "matching": [[36, 512, 512]], "classify": [[20]]},
            "train_probe_pair_ids_sha256": hashlib.sha256(b"probe").hexdigest(),
            "train_positive_and_negative_artifacts_decoded": True,
            "train_val_manifests_reverified_after_decode": True,
            "output_created_or_written": False,
        },
        "stages": stages,
        "dataset_manifest_sha256": manifests,
        "official_commit": SHRED_COMMIT,
        "scope": {
            "train_manifest_only_for_tensor_loading": True,
            "validation_tensor_loading": False,
            "sealed_test_or_real_opened": False,
            "random_smoke_models_not_checkpoints": True,
            "formal_training_started": False,
        },
        "numeric_contract": {
            "dual_softmax_probability_compute_dtype": "float32",
            "amp_logits_promoted_before_mask_and_softmax": True,
            "focal_logarithm_epsilon_semantics": "log_p_plus_1e-9_and_log_1_minus_p_plus_1e-9",
        },
    }
    if defect == "bad_adapter_sha":
        receipt["adapter_identity"]["adapter_source_sha256"] = "0" * 64
    output = Path(_option(arguments, "--output"))
    _write_bound(output, receipt)
    if defect == "bad_content_sha":
        value = json.loads(output.read_text(encoding="utf-8"))
        value["content_sha256"] = "0" * 64
        _write_plain(output, value)
    print("{}")


def _fake_runtime_main(argv: list[str]) -> int:
    executable = Path(argv[0]).resolve()
    fixture_root = executable.parents[2]
    controls_path = fixture_root / "fake-controls.json"
    controls = (
        json.loads(controls_path.read_text(encoding="utf-8"))
        if controls_path.is_file()
        else {}
    )
    if argv[1:6] == ["-I", "-S", "-B", "-u", "-c"] and len(argv) >= 15:
        launcher_flags = argv[1:6]
        path_count = int(argv[7])
        configuration_index = 10 + path_count
        explicit_site_configuration = json.loads(argv[configuration_index])
        if explicit_site_configuration.get("content_sha256") != hashlib.sha256(
            _canonical(
                {
                    key: value
                    for key, value in explicit_site_configuration.items()
                    if key != "content_sha256"
                }
            )
        ).hexdigest():
            raise SystemExit("fixture explicit-site configuration hash differs")
        if hashlib.sha256(sys.stdin.buffer.read()).hexdigest() != argv[
            9 + path_count
        ]:
            raise SystemExit("fixture stdin bootstrap hash differs")
        if explicit_site_configuration.get("schema_version") != (
            queue.ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SCHEMA
        ) or explicit_site_configuration.get(
            "torchvision_dynamic_admission"
        ) != queue._torchvision_dynamic_admission_configuration(True):
            raise SystemExit("fixture dynamic admission configuration differs")
        source_root = Path(argv[configuration_index + 2])
        module = argv[configuration_index + 3]
        arguments = argv[configuration_index + 4 :]
    else:
        raise SystemExit("fixture expected isolated reviewed-module bootstrap")
    log_path = fixture_root / "fake-python.jsonl"
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "module": module,
                    "arguments": arguments,
                    "launcher_flags": launcher_flags,
                    "runtime_no_site": True,
                    "dynamic_admission_enabled": explicit_site_configuration[
                        "torchvision_dynamic_admission"
                    ]["enabled"],
                    "explicit_site_roles": [
                        binding["role"]
                        for binding in explicit_site_configuration["sites"]
                    ],
                    "environment_home": os.environ.get("HOME"),
                    "environment_path_first": os.environ.get("PATH", "").split(
                        os.pathsep
                    )[0],
                },
                sort_keys=True,
            )
            + "\n"
        )
    if module == queue.ENVIRONMENT_BUILDER_MODULE:
        summary = dict(controls["environment_summary"])
        if controls.get("environment_drift_after_pair_smoke") and (
            fixture_root / "benchmark-output/control/pairingnet_gpu_smoke.json"
        ).is_file():
            summary["import_inventory_content_sha256"] = "0" * 64
        print(json.dumps(summary, sort_keys=True))
        return 0
    if module == queue.TRAIN_VAL_ASSET_FREEZE_MODULE:
        if arguments[0] == "create":
            _fake_asset_create(arguments)
        else:
            _fake_asset_verify(arguments, controls)
        return 0
    if module == queue.STRICT_VERIFIER_MODULE:
        command = arguments[0]
        if command == "upstreams":
            print(json.dumps(_fake_upstream_summary(arguments), sort_keys=True))
        elif command == "pairingnet":
            output = Path(_option(arguments, "--output-root"))
            manifests = _manifest_hashes(Path(_option(arguments, "--dataset-root")))
            artifacts = {
                name: hashlib.sha256(name.encode()).hexdigest()
                for name in (
                    "winner", "last", "threshold", "inference_contract",
                    "validation_report", "validation_predictions",
                )
            }
            print(json.dumps({
                "schema_version": "rachel-benchmark-queue-pairingnet-verification/1.0",
                "status": "verified_pairingnet_train_val_completion",
                "output_root": str(output),
                "completion_receipt_sha256": _sha(output / "completion_receipt.json"),
                "artifacts": artifacts,
                "epochs_completed": 32,
                "winner_epoch": 20,
                "plateau": 12,
                "stop_reason": "validation_plateau",
                "validation_threshold": 0.5,
                "manifest_content_sha256": manifests,
                "winner_and_last_strict_loaded_cpu": True,
                "threshold_replayed": True,
                "inference_contract_verified": True,
                "validation_report_verified": True,
                "test_or_real_opened": False,
            }, sort_keys=True))
        else:
            output = Path(_option(arguments, "--output-root"))
            manifests = _manifest_hashes(Path(_option(arguments, "--dataset-root")))
            stages = {
                stage: {
                    "completion_file_sha256": hashlib.sha256((stage + "completion").encode()).hexdigest(),
                    "completion_content_sha256": hashlib.sha256((stage + "content").encode()).hexdigest(),
                    "winner_sha256": hashlib.sha256((stage + "winner").encode()).hexdigest(),
                    "progress_sha256": hashlib.sha256((stage + "progress").encode()).hexdigest(),
                    "completed_epochs": 128,
                    "winner_epoch_zero_based": 100,
                }
                for stage in ("coarse", "matching", "classify")
            }
            print(json.dumps({
                "schema_version": "rachel-benchmark-queue-shreddingnet-verification/1.0",
                "status": "verified_shreddingnet_train_val_completion",
                "output_root": str(output),
                "freeze_file_sha256": _sha(output / "train_val_freeze.json"),
                "freeze_content_sha256": hashlib.sha256(b"freeze").hexdigest(),
                "run_contract_file_sha256": hashlib.sha256(b"contract").hexdigest(),
                "validation_report_file_sha256": hashlib.sha256(b"report").hexdigest(),
                "validation_report_content_sha256": hashlib.sha256(b"report-content").hexdigest(),
                "stage_artifacts": stages,
                "manifest_content_sha256": manifests,
                "three_stage_winners_strict_loaded_cpu": True,
                "three_stage_progress_checkpoints_strict_loaded_cpu": True,
                "threshold_score_and_fit_replayed": True,
                "inference_contract_verified": True,
                "validation_report_verified": True,
                "test_or_real_opened": False,
            }, sort_keys=True))
        return 0
    pair_runner = queue.METHODS["pairingnet"]["runner_module"]
    pair_smoke = queue.METHODS["pairingnet"]["smoke_module"]
    shred_runner = queue.METHODS["shreddingnet"]["runner_module"]
    if module == pair_runner and "--audit-only" in arguments:
        if controls.get("sleep_pair_audit"):
            grandchild = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"]
            )
            (fixture_root / "grandchild.pid").write_text(str(grandchild.pid), encoding="ascii")
            time.sleep(60)
        dataset = Path(_option(arguments, "--dataset-root"))
        manifests = _manifest_hashes(dataset)
        print(json.dumps({
            "schema_version": "rachel-pairingnet-preflight-audit/1.0",
            "population_audit": {
                "status": "train_val_population_audited",
                "formal_population_required": True,
                "population": {
                    "train": {"rows": 24000, "positive": 12000, "negative": 12000, "split_unit_count": 10},
                    "val": {"rows": 3000, "positive": 1500, "negative": 1500, "split_unit_count": 5},
                },
                "manifests": {"train_sha256": manifests["train"], "val_sha256": manifests["val"]},
                "parent_lineage_disjoint": True,
                "sealed_synthetic_accessed": False,
                "real_data_accessed": False,
            },
            "official_source_audit": {"commit": PAIR_COMMIT},
        }, sort_keys=True))
        return 0
    if module == pair_smoke:
        _fake_pair_smoke(arguments, source_root, controls)
        return 0
    if module == pair_runner:
        output = Path(_option(arguments, "--output-root"))
        output.mkdir()
        dataset = Path(_option(arguments, "--dataset-root"))
        manifests = _manifest_hashes(dataset)
        _write_plain(output / "completion_receipt.json", {
            "status": "train_validation_complete",
            "method_id": PAIR_METHOD,
            "official_commit": PAIR_COMMIT,
            "adapter_source_sha256": _sha(source_root / queue.METHODS["pairingnet"]["runner_relative_path"]),
            "adaptation_contract_sha256": _sha(source_root / queue.METHODS["pairingnet"]["adaptation_contract_relative_path"]),
            "population_audit": {"manifests": {"train_sha256": manifests["train"], "val_sha256": manifests["val"]}},
            "convergence_demonstrated": True,
            "sealed_synthetic_accessed": False,
            "real_data_accessed": False,
        })
        if controls.get("mutate_frozen_asset_after_pair"):
            dummy = dataset / "model/masks_800/fixture.bin"
            dummy.chmod(0o644)
            dummy.write_bytes(b"mutated")
        print("{}")
        return 0
    if module == shred_runner and arguments[0] == "audit":
        dataset = Path(_option(arguments, "--dataset-root"))
        manifests = _manifest_hashes(dataset)
        method = queue.METHODS["shreddingnet"]
        _write_bound(Path(_option(arguments, "--output")), {
            "schema_version": "rachel-shreddingnet-maskonly-benchmark/1.0",
            "status": "audit_complete_train_val_only",
            "method_id": SHRED_METHOD,
            "adapter_identity": {
                "adapter_source_sha256": _sha(source_root / method["runner_relative_path"]),
                "adaptation_document_sha256": _sha(source_root / method["adaptation_contract_relative_path"]),
            },
            "dataset": {
                "counts": {
                    "train": {"rows": 24000, "positive": 12000, "negative": 12000},
                    "val": {"rows": 3000, "positive": 1500, "negative": 1500},
                },
                "formal_counts_required": True,
                "lineage_disjoint": True,
                "opened_manifests": ["pairs/train.jsonl", "pairs/val.jsonl"],
                "sealed_test_manifest_opened": False,
                "real_data_opened": False,
                "manifest_content_sha256": manifests,
            },
            "official_source": {"commit": SHRED_COMMIT},
        })
        print("{}")
        return 0
    if module == shred_runner and arguments[0] == "gpu-smoke":
        _fake_shred_smoke(arguments, source_root, controls)
        return 0
    if module == shred_runner and arguments[0] == "train":
        output = Path(_option(arguments, "--output-root"))
        output.mkdir()
        dataset = Path(_option(arguments, "--dataset-root"))
        method = queue.METHODS["shreddingnet"]
        _write_bound(output / "train_val_freeze.json", {
            "schema_version": "rachel-shreddingnet-train-val-freeze/1.0",
            "checkpoint_kind": "rachel_shreddingnet_train_val_freeze",
            "status": "complete_train_val_frozen_no_test_or_real",
            "method_id": SHRED_METHOD,
            "official_commit": SHRED_COMMIT,
            "adapter_identity": {
                "adapter_source_sha256": _sha(source_root / method["runner_relative_path"]),
                "adaptation_document_sha256": _sha(source_root / method["adaptation_contract_relative_path"]),
            },
            "dataset_binding": {"manifest_content_sha256": _manifest_hashes(dataset)},
            "scope": {"sealed_test_or_real_opened": False, "train_val_only": True},
        })
        if controls.get("inject_output_symlink"):
            (output / "forbidden-link").symlink_to(output / "train_val_freeze.json")
        print("{}")
        return 0
    raise SystemExit("unexpected fixture command: " + repr((module, arguments)))


def _fake_benchmark_python(path: Path) -> None:
    import inspect

    path.parent.mkdir(parents=True)
    runtime_path = path.parents[2] / "fixture_runtime.py"
    functions = (
        _sha,
        _canonical,
        _bound,
        _write_plain,
        _write_bound,
        _option,
        _manifest_hashes,
        _ordered_pair_hashes,
        _fake_asset_create,
        _fake_asset_verify,
        _fake_upstream_summary,
        _fake_pair_smoke,
        _fake_shred_smoke,
        _fake_runtime_main,
    )
    runtime_path.write_text(
        "from __future__ import annotations\n"
        "import hashlib, json, os, shutil, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, {!r})\n".format(str(ROOT))
        + "from experiments.rachel_n512_formal_30k import "
        "run_same_data_benchmark_queue as queue\n"
        "PAIR_COMMIT = {!r}\nSHRED_COMMIT = {!r}\n"
        "PAIR_METHOD = {!r}\nSHRED_METHOD = {!r}\n".format(
            PAIR_COMMIT, SHRED_COMMIT, PAIR_METHOD, SHRED_METHOD
        )
        + "\n\n".join(inspect.getsource(function) for function in functions)
        + "\n",
        encoding="utf-8",
    )
    path.write_text(
        "#!{}\n".format(sys.executable)
        + "import runpy, sys\n"
        + "values = runpy.run_path({!r}, run_name='fixture_runtime')\n".format(
            str(runtime_path)
        )
        + "raise SystemExit(values['_fake_runtime_main']([{!r}, *sys.argv[1:]]))\n".format(
            str(path)
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _legacy_fixture(
    tmp_path: Path, *, completed_upstreams: bool = True
) -> tuple[Path, Path, Path]:
    source = tmp_path / "immutable-source"
    controller = source / (
        "experiments/rachel_n512_formal_30k/run_same_data_benchmark_queue.py"
    )
    controller.parent.mkdir(parents=True)
    controller.write_bytes(CONTROLLER.read_bytes())

    files = {
        "staging/pairwise_v0_2/baselines/rachel_pairingnet_benchmark.py": "# pair runner\n",
        "staging/pairwise_v0_2/baselines/PAIRINGNET_RACHEL_ADAPTATION.md": "pair contract\n",
        "experiments/rachel_n512_formal_30k/pairingnet_gpu_smoke.py": "# pair smoke\n",
        "staging/pairwise_v0_2/baselines/rachel_shreddingnet_benchmark.py": "# shred runner\n",
        "staging/pairwise_v0_2/baselines/RACHEL_SHREDDINGNET_BENCHMARK.md": "shred contract\n",
    }
    paths: dict[str, Path] = {}
    for relative, payload in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        paths[relative] = path

    pair_runner_rel = "staging/pairwise_v0_2/baselines/rachel_pairingnet_benchmark.py"
    pair_doc_rel = "staging/pairwise_v0_2/baselines/PAIRINGNET_RACHEL_ADAPTATION.md"
    pair_smoke_rel = "experiments/rachel_n512_formal_30k/pairingnet_gpu_smoke.py"
    shred_runner_rel = "staging/pairwise_v0_2/baselines/rachel_shreddingnet_benchmark.py"
    shred_doc_rel = "staging/pairwise_v0_2/baselines/RACHEL_SHREDDINGNET_BENCHMARK.md"
    pair_marker_rel = "experiments/rachel_n512_formal_30k/reviews/pairingnet.json"
    shred_marker_rel = "experiments/rachel_n512_formal_30k/reviews/shreddingnet.json"
    pair_marker = source / pair_marker_rel
    shred_marker = source / shred_marker_rel
    _write_json(
        pair_marker,
        _review_marker(
            method_id=PAIR_METHOD,
            runner_relative=pair_runner_rel,
            runner_sha=_sha(paths[pair_runner_rel]),
            contract_relative=pair_doc_rel,
            contract_sha=_sha(paths[pair_doc_rel]),
            smoke_relative=pair_smoke_rel,
            smoke_sha=_sha(paths[pair_smoke_rel]),
        ),
    )
    _write_json(
        shred_marker,
        _review_marker(
            method_id=SHRED_METHOD,
            runner_relative=shred_runner_rel,
            runner_sha=_sha(paths[shred_runner_rel]),
            contract_relative=shred_doc_rel,
            contract_sha=_sha(paths[shred_doc_rel]),
        ),
    )
    asset_marker_relative = (
        "experiments/rachel_n512_formal_30k/reviews/train_val_asset_freeze.json"
    )
    asset_marker = source / asset_marker_relative
    _write_json(
        asset_marker,
        {
            "schema_version": queue.TRAIN_VAL_ASSET_REVIEW_SCHEMA,
            "status": "GO",
            "p0_count": 0,
            "p1_count": 0,
            "review_scope_complete": True,
            "module": {
                "relative_path": queue.TRAIN_VAL_ASSET_FREEZE_RELATIVE_PATH,
                "sha256": queue.TRAIN_VAL_ASSET_FREEZE_SHA256,
            },
            "focused_tests": {
                "relative_path": queue.TRAIN_VAL_ASSET_FREEZE_TEST_RELATIVE_PATH,
                "sha256": queue.TRAIN_VAL_ASSET_FREEZE_TEST_SHA256,
            },
            "contracts": {
                "inventory_schema_version": (
                    "rachel-train-val-semantic-asset-inventory/3.0"
                ),
                "receipt_schema_version": (
                    "rachel-train-val-semantic-asset-freeze/3.0"
                ),
                "verification_schema_version": (
                    "rachel-train-val-semantic-asset-verification/3.0"
                ),
                "frozen_view_schema_version": (
                    "rachel-frozen-train-val-dataset-view/2.0"
                ),
            },
            "same_freeze_root_fd_lifetime_reviewed": True,
            "final_complete_tree_revalidation_reviewed": True,
            "regular_file_link_count_one_reviewed": True,
            "root_replacement_race_tests_passed": True,
            "train_val_only": True,
            "sealed_test_accessed": False,
            "real_data_accessed": False,
        },
    )

    environment_root = tmp_path / "benchmark-env"
    benchmark_python = environment_root / "bin/python"
    _fake_benchmark_python(benchmark_python)
    fake_git = tmp_path / "fake-bin/git"
    fake_git.parent.mkdir()
    fake_git.write_text(
        "#!{}\n".format(sys.executable)
        + "import pathlib, sys\n"
        + "root = pathlib.Path(sys.argv[sys.argv.index('-C') + 1]).name\n"
        + "if 'rev-parse' in sys.argv:\n"
        + " print({!r} if root == 'PairingNet' else {!r})\n".format(
            PAIR_COMMIT, SHRED_COMMIT
        ),
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    probe = {
        "executable": str(benchmark_python.resolve()),
        "prefix": str(environment_root.resolve()),
        "base_prefix": "/fixture/base",
        "python_version": "3.fixture",
        "user_site_enabled": False,
        "packages": {
            "torch": "2.5.1+cu124",
            "torch-geometric": "2.fixture",
            "opencv-python": "4.fixture",
            "opencv-python-headless": None,
            "numpy": "2.fixture",
            "scipy": "1.14.1",
        },
        "imports": {
            "torch": "ok",
            "torch_geometric": "ok",
            "cv2": "ok",
            "numpy": "ok",
            "scipy": "ok",
        },
    }
    env_receipt = environment_root / "environment_receipt.json"
    _write_json(
        env_receipt,
        {
            "schema_version": "rachel-benchmark-isolated-python-environment/1.0",
            "status": "complete_isolated_benchmark_environment",
            "environment_root": str(environment_root.resolve()),
            "python": str(benchmark_python.resolve()),
            "python_sha256": _sha(benchmark_python),
            "isolated_from_primary_training_environment": True,
            "sealed_test_or_real_accessed": False,
            "live_probe": probe,
        },
    )

    dataset = tmp_path / "dataset"
    (dataset / "pairs").mkdir(parents=True)
    (dataset / "pairs/train.jsonl").write_text("fixture-train\n", encoding="utf-8")
    (dataset / "pairs/val.jsonl").write_text("fixture-val\n", encoding="utf-8")
    n512 = tmp_path / "n512"
    matched = tmp_path / "matched"
    n512.mkdir()
    matched.mkdir()
    if completed_upstreams:
        _write_json(
            n512 / "run-fixture/run_receipt.json",
            {
                "schema_version": "rachel-n512-train-run/1.0",
                "status": "complete_train_validation_only",
                "arm_results": [
                    {
                        "stop_reason": "validation_early_stop",
                        "convergence_claim": "validation_plateau_under_declared_rule",
                    }
                ],
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
        )
        _write_json(
            matched / "run-fixture/run_receipt.json",
            {
                "schema_version": "rachel-matched-historical-mm-train/1.0",
                "status": "complete_train_validation_only",
                "stop_reason": "validation_early_stop",
                "convergence_claim": "validation_plateau_under_declared_rule",
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
        )

    output = tmp_path / "benchmark-output"
    config = {
        "schema_version": "rachel-same-data-benchmark-queue-config/1.0",
        "immutable_source_root": str(source.resolve()),
        "dataset_root": str(dataset.resolve()),
        "output_root": str(output),
        "benchmark_environment": {
            "root": str(environment_root.resolve()),
            "python": str(benchmark_python.resolve()),
            "python_sha256": _sha(benchmark_python),
            "receipt": str(env_receipt.resolve()),
            "receipt_sha256": _sha(env_receipt),
        },
        "upstreams": {
            "n512": {
                "root": str(n512.resolve()),
                "receipt_glob": "run-*/run_receipt.json",
                "expected_schema": "rachel-n512-train-run/1.0",
            },
            "matched_mm": {
                "root": str(matched.resolve()),
                "receipt_glob": "run-*/run_receipt.json",
                "expected_schema": "rachel-matched-historical-mm-train/1.0",
            },
        },
        "official_sources": {
            "pairingnet": {
                "root": str((ROOT / "tmp/benchmarks/PairingNet").resolve()),
                "commit": PAIR_COMMIT,
            },
            "shreddingnet": {
                "root": str((ROOT / "tmp/benchmarks/shreddingnet").resolve()),
                "commit": SHRED_COMMIT,
            },
        },
        "methods": {
            "pairingnet": {
                "method_id": PAIR_METHOD,
                "runner_relative_path": pair_runner_rel,
                "runner_sha256": _sha(paths[pair_runner_rel]),
                "adaptation_contract_relative_path": pair_doc_rel,
                "adaptation_contract_sha256": _sha(paths[pair_doc_rel]),
                "smoke_source_relative_path": pair_smoke_rel,
                "smoke_source_sha256": _sha(paths[pair_smoke_rel]),
                "review_marker_relative_path": pair_marker_rel,
                "review_marker_sha256": _sha(pair_marker),
            },
            "shreddingnet": {
                "method_id": SHRED_METHOD,
                "runner_relative_path": shred_runner_rel,
                "runner_sha256": _sha(paths[shred_runner_rel]),
                "adaptation_contract_relative_path": shred_doc_rel,
                "adaptation_contract_sha256": _sha(paths[shred_doc_rel]),
                "review_marker_relative_path": shred_marker_rel,
                "review_marker_sha256": _sha(shred_marker),
            },
        },
        "runtime": {
            "device": "cuda:0",
            "pairingnet": {"precision": "fp32", "num_workers": 0},
            "shreddingnet": {
                "num_workers": 0,
                "coarse_microbatch": 1,
                "matching_microbatch": 1,
                "classify_microbatch": 1,
                "amp": True,
            },
        },
        "wait": {"poll_seconds": 0.1, "timeout_seconds": 3},
    }
    config_path = tmp_path / "queue-config.json"
    _write_json(config_path, config)
    return controller, config_path, output


def _fixture(
    tmp_path: Path, *, completed_upstreams: bool = True
) -> tuple[Path, Path, Path]:
    source = tmp_path / "immutable-source"
    builder_relative = queue.ENVIRONMENT_BUILDER_RELATIVE_PATH
    builder_source = ROOT / builder_relative
    builder_sha = _sha(builder_source)
    for relative in sorted(queue.REQUIRED_SOURCE_RELATIVE_PATHS):
        source_path = ROOT / relative
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
    wheel_lock_relative = queue.ENVIRONMENT_REVIEWED_WHEEL_LOCK_RELATIVE_PATH
    wheel_lock = source / wheel_lock_relative
    wheel_lock.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / wheel_lock_relative, wheel_lock)
    for method in queue.METHODS.values():
        relative = str(method["adaptation_contract_relative_path"])
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    controller = source / queue.CONTROLLER_RELATIVE_PATH
    controller_text = controller.read_text(encoding="utf-8")
    if queue.ENVIRONMENT_BUILDER_SHA256 != builder_sha:
        controller_text = controller_text.replace(
            queue.ENVIRONMENT_BUILDER_SHA256, builder_sha
        )
        controller.write_text(controller_text, encoding="utf-8")

    pair = queue.METHODS["pairingnet"]
    shred = queue.METHODS["shreddingnet"]
    pair_marker_relative = (
        "experiments/rachel_n512_formal_30k/reviews/pairingnet.json"
    )
    shred_marker_relative = (
        "experiments/rachel_n512_formal_30k/reviews/shreddingnet.json"
    )
    pair_marker = source / pair_marker_relative
    shred_marker = source / shred_marker_relative
    _write_json(
        pair_marker,
        _review_marker(
            method_id=PAIR_METHOD,
            runner_relative=str(pair["runner_relative_path"]),
            runner_sha=_sha(source / str(pair["runner_relative_path"])),
            contract_relative=str(pair["adaptation_contract_relative_path"]),
            contract_sha=_sha(
                source / str(pair["adaptation_contract_relative_path"])
            ),
            smoke_relative=str(pair["smoke_source_relative_path"]),
            smoke_sha=_sha(source / str(pair["smoke_source_relative_path"])),
        ),
    )
    _write_json(
        shred_marker,
        _review_marker(
            method_id=SHRED_METHOD,
            runner_relative=str(shred["runner_relative_path"]),
            runner_sha=_sha(source / str(shred["runner_relative_path"])),
            contract_relative=str(shred["adaptation_contract_relative_path"]),
            contract_sha=_sha(
                source / str(shred["adaptation_contract_relative_path"])
            ),
        ),
    )
    asset_marker_relative = (
        "experiments/rachel_n512_formal_30k/reviews/"
        "train_val_asset_freeze_review_GO.json"
    )
    asset_marker = source / asset_marker_relative
    asset_marker.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / asset_marker_relative, asset_marker)
    environment_marker_relative = queue.ENVIRONMENT_REVIEW_RELATIVE_PATH
    environment_marker = source / environment_marker_relative
    environment_marker.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / environment_marker_relative, environment_marker)
    for relative in queue.REQUIRED_SOURCE_RELATIVE_PATHS:
        (source / relative).chmod(queue.IMMUTABLE_SOURCE_FILE_MODE)
    inventory = queue._source_inventory(source)
    queue_marker_relative = (
        "experiments/rachel_n512_formal_30k/reviews/queue.json"
    )
    queue_marker = source / queue_marker_relative
    _write_json(
        queue_marker,
        {
            "schema_version": queue.QUEUE_REVIEW_SCHEMA,
            "status": "GO",
            "p0_count": 0,
            "p1_count": 0,
            "review_scope_complete": True,
            "controller": {
                "relative_path": queue.CONTROLLER_RELATIVE_PATH,
                "sha256": _sha(controller),
            },
            "pairingnet_smoke": {
                "relative_path": queue.PAIRINGNET_SMOKE_RELATIVE_PATH,
                "sha256": _sha(source / queue.PAIRINGNET_SMOKE_RELATIVE_PATH),
            },
            "source_inventory": {
                "scope": queue.SOURCE_INVENTORY_SCOPE,
                "file_count": len(inventory),
                "sha256": queue._source_inventory_sha256(inventory),
            },
            "isolated_bootstrap_reviewed": True,
            "forbidden_execution_surfaces_absent": True,
            "sealed_synthetic_accessed": False,
            "real_data_accessed": False,
        },
    )

    environment_root = tmp_path / "benchmark-env"
    benchmark_python = environment_root / "bin/python"
    _fake_benchmark_python(benchmark_python)
    runtime_no_site = _fixture_runtime_no_site(environment_root, benchmark_python)
    inventory_sha = hashlib.sha256(b"fixture-import-inventory").hexdigest()
    receipt_content_sha = hashlib.sha256(b"placeholder").hexdigest()
    environment_receipt = environment_root / "environment_receipt.json"
    reviewed_wheel_lock = json.loads(wheel_lock.read_text(encoding="utf-8"))
    wheelhouse = environment_root / ".environment_build_evidence/wheelhouse"
    report_path_normalization = _bound(
        {
            "schema_version": (
                "rachel-benchmark-pip-report-path-normalization/1.0"
            ),
            "policy": (
                "execution-anchor-file-url-to-canonical-lexical-wheel-path"
            ),
            "entries": [
                {
                    "filename": package["filename"],
                    "source_url": (wheelhouse / package["filename"]).as_uri(),
                    "canonical_path": str(wheelhouse / package["filename"]),
                    "canonical_url": (wheelhouse / package["filename"]).as_uri(),
                }
                for package in reviewed_wheel_lock["packages"]
            ],
        }
    )
    environment_value = {
        "schema_version": queue.ENVIRONMENT_SCHEMA,
        "status": "complete_isolated_benchmark_environment",
        "environment_root": str(environment_root.resolve()),
        "python": str(benchmark_python.resolve()),
        "python_sha256": _sha(benchmark_python),
        "isolated_from_primary_training_environment": True,
        "runtime_no_site": True,
        "sealed_test_or_real_accessed": False,
        "live_probe": {"fixture": True},
        "isolation_model": (
            "fresh_venv_with_fd_anchored_explicit_sites_and_no_site_startup"
        ),
        "application_distributions_installed": {
            "torch-geometric": "2.6.1",
            "opencv-python": "4.10.0.84",
            "scipy": "1.14.1",
        },
        "inherited_distributions_verified": [
            "torch",
            "torchvision",
            "numpy",
            "Pillow",
        ],
        "official_shreddingnet_environment": {},
        "base_runtime": {},
        "detailed_probe": {"fixture": True},
        "functional_cpu_probe": {"fixture": True},
        "pip_freeze": {"fixture": True},
        "build_evidence": {
            "download": {
                "argv": queue._expected_environment_download_argv(
                    environment_root.resolve(),
                    [package["url"] for package in reviewed_wheel_lock["packages"]],
                ),
                "argv_policy": queue.ENVIRONMENT_PIP_ARGV_POLICY,
                "stdout_sha256": hashlib.sha256(
                    b"fixture-download-out"
                ).hexdigest(),
                "stderr_sha256": hashlib.sha256(
                    b"fixture-download-err"
                ).hexdigest(),
                "requirements_lock": str(
                    environment_root.resolve()
                    / queue.ENVIRONMENT_REQUIREMENTS_LOCK_RELATIVE_PATH
                ),
                "requirements_lock_sha256": hashlib.sha256(
                    b"fixture-requirements-lock"
                ).hexdigest(),
            },
            "install": {
                "argv": queue._expected_environment_install_argv(
                    environment_root.resolve()
                ),
                "argv_policy": queue.ENVIRONMENT_PIP_ARGV_POLICY,
                "stdout_sha256": hashlib.sha256(b"fixture-install-out").hexdigest(),
                "stderr_sha256": hashlib.sha256(b"fixture-install-err").hexdigest(),
                "report": str(
                    environment_root
                    / ".environment_build_evidence/pip_install_report.json"
                ),
                "report_sha256": hashlib.sha256(b"fixture-report").hexdigest(),
                "report_path_normalization": report_path_normalization,
                "installed": [],
            },
            "wheels": [],
            "inventory": {
                "path": "fixture-import-inventory.json",
                "file_sha256": hashlib.sha256(b"fixture-file").hexdigest(),
                "content_sha256": inventory_sha,
            },
            "active_site": {"fixture": True},
            "target_launcher_identity": {"fixture": True},
            "reviewed_wheel_lock": {
                "path": str(wheel_lock),
                "file_sha256": _sha(wheel_lock),
                "content_sha256": reviewed_wheel_lock["content_sha256"],
                "packages": reviewed_wheel_lock["packages"],
                "target": reviewed_wheel_lock["target"],
                "official_index": reviewed_wheel_lock["official_index"],
                "review": reviewed_wheel_lock["review"],
            },
            "bin_scripts": {
                "schema_version": "rachel-benchmark-bin-script-inventory/1.0",
                "environment_root": str(environment_root.resolve()),
                "entries": [],
                "ephemeral_descriptor_paths_present": False,
                "content_sha256": hashlib.sha256(b"fixture-bin-scripts").hexdigest(),
            },
            "runtime_no_site": runtime_no_site,
        },
        "builder_source": {
            "path": str(source / builder_relative),
            "sha256": builder_sha,
        },
    }
    environment_value["content_sha256"] = hashlib.sha256(
        _canonical(environment_value)
    ).hexdigest()
    receipt_content_sha = environment_value["content_sha256"]
    _write_plain(environment_receipt, environment_value)
    environment_summary = {
        "schema_version": queue.ENVIRONMENT_SCHEMA,
        "status": "verified_isolated_benchmark_environment",
        "environment_root": str(environment_root.resolve()),
        "python": str(benchmark_python.resolve()),
        "python_sha256": _sha(benchmark_python),
        "receipt": str(environment_receipt.resolve()),
        "receipt_sha256": _sha(environment_receipt),
        "receipt_content_sha256": receipt_content_sha,
        "import_inventory_content_sha256": inventory_sha,
        "runtime_no_site": True,
        "sealed_test_or_real_accessed": False,
    }

    dataset = tmp_path / "dataset"
    (dataset / "pairs").mkdir(parents=True)
    for split, count in (("train", 24_000), ("val", 3_000)):
        with (dataset / "pairs" / (split + ".jsonl")).open(
            "w", encoding="utf-8"
        ) as stream:
            for index in range(count):
                stream.write(
                    json.dumps(
                        {
                            "pair_id": "{}-{:05d}".format(split, index),
                            "label": index % 2 == 0,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    n512 = tmp_path / "n512"
    matched = tmp_path / "matched"
    n512.mkdir()
    matched.mkdir()
    if completed_upstreams:
        _write_json(
            n512 / "run-fixture/run_receipt.json",
            {
                "schema_version": "rachel-n512-train-run/1.0",
                "status": "complete_train_validation_only",
                "arm_results": [
                    {
                        "stop_reason": "validation_early_stop",
                        "convergence_claim": (
                            "validation_plateau_under_declared_rule"
                        ),
                    }
                ],
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
        )
        _write_json(
            matched / "run-fixture/run_receipt.json",
            {
                "schema_version": "rachel-matched-historical-mm-train/1.0",
                "status": "complete_train_validation_only",
                "stop_reason": "validation_early_stop",
                "convergence_claim": "validation_plateau_under_declared_rule",
                "test_accessed": False,
                "real_external_test_accessed": False,
            },
        )

    output = tmp_path / "benchmark-output"
    config = {
        "schema_version": queue.CONFIG_SCHEMA,
        "controller": {
            "relative_path": queue.CONTROLLER_RELATIVE_PATH,
            "sha256": _sha(controller),
            "review_marker_relative_path": queue_marker_relative,
            "review_marker_sha256": _sha(queue_marker),
        },
        "train_val_asset_freeze_authority": {
            "module_relative_path": queue.TRAIN_VAL_ASSET_FREEZE_RELATIVE_PATH,
            "module_sha256": queue.TRAIN_VAL_ASSET_FREEZE_SHA256,
            "test_relative_path": queue.TRAIN_VAL_ASSET_FREEZE_TEST_RELATIVE_PATH,
            "test_sha256": queue.TRAIN_VAL_ASSET_FREEZE_TEST_SHA256,
            "review_marker_relative_path": asset_marker_relative,
            "review_marker_sha256": _sha(asset_marker),
        },
        "immutable_source_root": str(source.resolve()),
        "dataset_root": str(dataset.resolve()),
        "output_root": str(output),
        "benchmark_environment": {
            "root": str(environment_root.resolve()),
            "python": str(benchmark_python.resolve()),
            "python_sha256": _sha(benchmark_python),
            "receipt": str(environment_receipt.resolve()),
            "receipt_sha256": _sha(environment_receipt),
            "receipt_content_sha256": receipt_content_sha,
            "import_inventory_content_sha256": inventory_sha,
            "builder_source_relative_path": builder_relative,
            "builder_source_sha256": builder_sha,
            "builder_test_relative_path": (
                queue.ENVIRONMENT_BUILDER_TEST_RELATIVE_PATH
            ),
            "builder_test_sha256": queue.ENVIRONMENT_BUILDER_TEST_SHA256,
            "reviewed_wheel_lock_relative_path": wheel_lock_relative,
            "reviewed_wheel_lock_sha256": _sha(wheel_lock),
            "review_marker_relative_path": environment_marker_relative,
            "review_marker_sha256": _sha(environment_marker),
        },
        "upstreams": {
            "n512": {
                "root": str(n512.resolve()),
                "receipt_glob": "run-*/run_receipt.json",
                "expected_schema": "rachel-n512-train-run/1.0",
            },
            "matched_mm": {
                "root": str(matched.resolve()),
                "receipt_glob": "run-*/run_receipt.json",
                "expected_schema": "rachel-matched-historical-mm-train/1.0",
            },
        },
        "official_sources": {
            "pairingnet": {
                "root": str((ROOT / "tmp/benchmarks/PairingNet").resolve()),
                "commit": PAIR_COMMIT,
            },
            "shreddingnet": {
                "root": str((ROOT / "tmp/benchmarks/shreddingnet").resolve()),
                "commit": SHRED_COMMIT,
            },
        },
        "methods": {
            "pairingnet": {
                "method_id": PAIR_METHOD,
                "runner_relative_path": pair["runner_relative_path"],
                "runner_sha256": _sha(source / pair["runner_relative_path"]),
                "adaptation_contract_relative_path": pair[
                    "adaptation_contract_relative_path"
                ],
                "adaptation_contract_sha256": _sha(
                    source / pair["adaptation_contract_relative_path"]
                ),
                "smoke_source_relative_path": pair["smoke_source_relative_path"],
                "smoke_source_sha256": _sha(
                    source / pair["smoke_source_relative_path"]
                ),
                "review_marker_relative_path": pair_marker_relative,
                "review_marker_sha256": _sha(pair_marker),
            },
            "shreddingnet": {
                "method_id": SHRED_METHOD,
                "runner_relative_path": shred["runner_relative_path"],
                "runner_sha256": _sha(source / shred["runner_relative_path"]),
                "adaptation_contract_relative_path": shred[
                    "adaptation_contract_relative_path"
                ],
                "adaptation_contract_sha256": _sha(
                    source / shred["adaptation_contract_relative_path"]
                ),
                "review_marker_relative_path": shred_marker_relative,
                "review_marker_sha256": _sha(shred_marker),
            },
        },
        "runtime": {
            "device": "cuda:0",
            "pairingnet": {"precision": "fp32", "num_workers": 0},
            "shreddingnet": {
                "num_workers": 0,
                "coarse_microbatch": 1,
                "matching_microbatch": 1,
                "classify_microbatch": 1,
                "amp": True,
            },
        },
        "wait": {"poll_seconds": 0.1, "timeout_seconds": 3},
    }
    config_path = tmp_path / "queue-config.json"
    _write_json(config_path, config)
    _write_json(tmp_path / "fake-controls.json", {"environment_summary": environment_summary})
    return controller, config_path, output


def _run(
    controller: Path,
    config: Path,
    log: Path,
    *,
    timeout: float = 90,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    controls_path = config.parent / "fake-controls.json"
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    if extra_environment:
        translations = {
            "FAKE_FAIL_STAGE": "fail_stage",
            "FAKE_SHREDDING_SMOKE_DEFECT": "shred_smoke_defect",
            "FAKE_MUTATE_AFTER_PAIR_TRAIN": "mutate_frozen_asset_after_pair",
        }
        for key, value in extra_environment.items():
            controls[translations.get(key, key)] = (
                value == "1" if value in {"0", "1"} else value
            )
    _write_json(controls_path, controls)
    assert log == config.parent / "fake-python.jsonl"
    source = Path(
        json.loads(config.read_text(encoding="utf-8"))["immutable_source_root"]
    )
    _seal_source_tree(source)
    try:
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(controller),
                "--config",
                str(config),
            ],
            env={
                **os.environ,
                "PATH": str(config.parent / "fake-bin")
                + os.pathsep
                + os.environ.get("PATH", ""),
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    finally:
        _unseal_source_tree(source)


def test_template_binds_frozen_shreddingnet_authority_and_keeps_pair_placeholders() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    value = json.loads(text)
    assert value["official_sources"]["pairingnet"]["commit"] == PAIR_COMMIT
    assert value["official_sources"]["shreddingnet"]["commit"] == SHRED_COMMIT
    assert value["methods"]["shreddingnet"]["runner_sha256"] == (
        "40193df524cbb066cd9585788049a7449fc2c5a48a417d185ff8d56783c2b770"
    )
    assert value["methods"]["shreddingnet"]["adaptation_contract_sha256"] == (
        "96c59b4b0693fc8c031a94b03b64888465fc1fba202c367d58bb347b2a844f85"
    )
    assert value["methods"]["shreddingnet"]["review_marker_sha256"] == (
        "f794ae02ccc42d494e3d9d4286a5873c53af7ced3817a71bd30f0da4dd7e9d51"
    )
    assert value["methods"]["pairingnet"]["runner_sha256"].startswith(
        "FILL_EXACT_"
    )
    assert value["train_val_asset_freeze_authority"]["module_sha256"] == (
        queue.TRAIN_VAL_ASSET_FREEZE_SHA256
    )
    assert value["train_val_asset_freeze_authority"]["test_sha256"] == (
        queue.TRAIN_VAL_ASSET_FREEZE_TEST_SHA256
    )
    assert value["train_val_asset_freeze_authority"][
        "review_marker_sha256"
    ] == _sha(
        ROOT
        / value["train_val_asset_freeze_authority"][
            "review_marker_relative_path"
        ]
    )
    assert value["benchmark_environment"]["builder_test_sha256"] == (
        queue.ENVIRONMENT_BUILDER_TEST_SHA256
    )
    assert value["benchmark_environment"]["builder_source_sha256"] == (
        "b4814505925fac49accf4546f1675711160e84c615d451ce7d08d7a38c58f53f"
    )
    assert value["benchmark_environment"]["review_marker_sha256"] == (
        "e34a1ce9deae4927e9c0717c543b308affe17cf704ffa06184381acdffea2dfd"
    )
    assert value["benchmark_environment"]["review_marker_sha256"] == _sha(
        ROOT / queue.ENVIRONMENT_REVIEW_RELATIVE_PATH
    )
    assert value["benchmark_environment"][
        "reviewed_wheel_lock_relative_path"
    ] == queue.ENVIRONMENT_REVIEWED_WHEEL_LOCK_RELATIVE_PATH
    assert value["benchmark_environment"][
        "reviewed_wheel_lock_sha256"
    ] == queue.ENVIRONMENT_REVIEWED_WHEEL_LOCK_SHA256
    assert "sealed" not in " ".join(value.keys()).lower()


def test_environment_builder_stdin_launcher_and_pip_evidence_exactly_match() -> None:
    from experiments.rachel_n512_formal_30k import (
        build_rachel_benchmark_environment as environment_builder,
    )

    def digest(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    assert queue.ENVIRONMENT_BUILDER_SHA256 == _sha(
        ROOT / queue.ENVIRONMENT_BUILDER_RELATIVE_PATH
    )
    assert queue.ENVIRONMENT_BUILDER_TEST_SHA256 == _sha(
        ROOT / queue.ENVIRONMENT_BUILDER_TEST_RELATIVE_PATH
    )
    assert queue.ENVIRONMENT_EXPLICIT_SITE_STDIN_LAUNCHER_SHA256 == digest(
        environment_builder.EXPLICIT_SITE_STDIN_LAUNCHER_CODE
    )
    assert queue.ENVIRONMENT_EXPLICIT_SITE_BOOTSTRAP_SHA256 == digest(
        environment_builder.EXPLICIT_SITE_BOOTSTRAP_CODE
    )
    assert queue.ENVIRONMENT_TORCHVISION_DYNAMIC_ADMISSION_SOURCE_SHA256 == (
        digest(environment_builder.TORCHVISION_DYNAMIC_ADMISSION_BOOTSTRAP_CODE)
    )
    assert queue._torchvision_dynamic_admission_configuration(True) == (
        environment_builder._torchvision_dynamic_admission_configuration(True)
    )
    assert queue._torchvision_dynamic_admission_configuration(False) == (
        environment_builder._torchvision_dynamic_admission_configuration(False)
    )
    assert queue._successful_torchvision_dynamic_admission_evidence() == (
        environment_builder._successful_torchvision_dynamic_admission_evidence()
    )
    assert queue.ENVIRONMENT_PIP_NO_SITE_RUNNER_SHA256 == digest(
        environment_builder.PIP_NO_SITE_RUN_CODE
    )
    assert queue.ENVIRONMENT_RUNTIME_PAYLOAD_SHA256 == {
        "queue_live_probe": digest(environment_builder.QUEUE_LIVE_PROBE_CODE),
        "detailed_probe": digest(environment_builder.DETAILED_PROBE_CODE),
        "functional_cpu_probe": digest(
            environment_builder.FUNCTIONAL_CPU_PROBE_CODE
        ),
        "import_inventory": digest(environment_builder.INVENTORY_PROBE_CODE),
    }
    root = Path("/reviewed/environment-root")
    wheel_lock = json.loads(
        (ROOT / queue.ENVIRONMENT_REVIEWED_WHEEL_LOCK_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
    )
    urls = [package["url"] for package in wheel_lock["packages"]]
    assert queue._expected_environment_download_argv(root, urls) == (
        environment_builder._download_evidence_argv(root, urls)
    )
    assert queue._expected_environment_install_argv(root) == (
        environment_builder._install_evidence_argv(root)
    )
    marker = json.loads(
        (ROOT / queue.ENVIRONMENT_REVIEW_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    assert marker["contracts"][
        "content_bound_stdin_launcher_and_preimport_sys_path_reviewed"
    ] is True
    assert marker["contracts"][
        "torchvision_dynamic_module_admission_reviewed"
    ] is True
    assert marker["contracts"][
        "pinned_torchvision_source_pyc_never_read_reviewed"
    ] is True
    assert marker["verification"]["focused_python38_tests_passed"] == 61


@pytest.mark.parametrize(
    "attack",
    ("authority", "temp_origin_count", "pyc_read_count", "restore_false"),
)
def test_dynamic_admission_validator_rejects_nested_tampering(attack: str) -> None:
    evidence = queue._successful_torchvision_dynamic_admission_evidence()
    if attack == "authority":
        evidence["authorities"]["generated_module_size"] += 1
    elif attack == "temp_origin_count":
        evidence["observed"]["captured_temp_origin_module_count"] = 1
    elif attack == "pyc_read_count":
        evidence["observed"]["pinned_source_pyc_read_count"] = 1
    else:
        evidence["observed"]["reviewed_sys_path_restored"] = False
    with pytest.raises(queue.QueueContractError, match="differs"):
        queue._validate_torchvision_dynamic_admission_evidence(
            evidence, "fixture dynamic admission"
        )


def test_controller_rejects_isolated_without_no_site_before_config_access(
    tmp_path: Path,
) -> None:
    missing_config = tmp_path / "must-not-be-opened.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(CONTROLLER),
            "--config",
            str(missing_config),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "requires Python -I -S -B" in result.stderr
    assert not missing_config.exists()


@pytest.mark.parametrize("attack", ("none", "wrong_fd"))
def test_explicit_site_runner_never_executes_pth_and_rejects_wrong_fd(
    tmp_path: Path, attack: str
) -> None:
    site_directory = tmp_path / "reviewed-site-packages"
    site_directory.mkdir()
    pth_marker = tmp_path / "PTH_EXECUTED"
    payload_marker = tmp_path / "PAYLOAD_EXECUTED"
    (site_directory / "queue_probe_dependency.py").write_text(
        "VALUE = 'reviewed-explicit-site'\n", encoding="utf-8"
    )
    (site_directory / "executable_attack.pth").write_text(
        "import pathlib; pathlib.Path({!r}).write_text('attack', encoding='utf-8')\n".format(
            str(pth_marker)
        ),
        encoding="utf-8",
    )
    payload = (
        "import pathlib, site, sys\n"
        "if 'sitecustomize' in sys.modules or 'usercustomize' in sys.modules: "
        "raise RuntimeError('customization loaded')\n"
        "try:\n"
        "    site.addsitedir(sys.argv[1])\n"
        "except RuntimeError:\n"
        "    pass\n"
        "else:\n"
        "    raise RuntimeError('site.addsitedir was not blocked')\n"
        "pathlib.Path(sys.argv[2]).write_text('payload', encoding='utf-8')\n"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with _explicit_site_subprocess(
        payload,
        [str(site_directory), str(payload_marker)],
        [site_directory],
    ) as (command, pass_fds, bootstrap):
        if attack == "wrong_fd":
            path_count = int(command[6])
            configuration_index = 9 + path_count
            configuration = json.loads(command[configuration_index])
            bad_descriptor = (
                "/proc/{}/fd/999999".format(os.getpid())
                if sys.platform.startswith("linux")
                else "/dev/fd/999999"
            )
            for binding in configuration["sites"]:
                binding["descriptor_path"] = bad_descriptor
            configuration.pop("content_sha256")
            configuration = _bound(configuration)
            command[configuration_index] = json.dumps(
                configuration, sort_keys=True, separators=(",", ":")
            )
        result = subprocess.run(
            command,
            pass_fds=pass_fds,
            check=False,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            input=bootstrap,
            text=True,
            timeout=30,
        )
    assert not pth_marker.exists()
    if attack == "none":
        assert result.returncode == 0, result.stderr
        assert payload_marker.read_text(encoding="utf-8") == "payload"
    else:
        assert result.returncode != 0
        assert "descriptor FD is not inherited" in result.stderr
        assert not payload_marker.exists()


def test_false_runtime_no_site_receipt_field_fails_before_output_or_child(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt_path = Path(config["benchmark_environment"]["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["runtime_no_site"] = False
    receipt.pop("content_sha256")
    receipt = _bound(receipt)
    _write_plain(receipt_path, receipt)
    config["benchmark_environment"]["receipt_sha256"] = _sha(receipt_path)
    config["benchmark_environment"]["receipt_content_sha256"] = receipt[
        "content_sha256"
    ]
    _write_json(config_path, config)
    log = tmp_path / "fake-python.jsonl"
    result = _run(controller, config_path, log)
    assert result.returncode != 0
    assert "environment receipt identity/scope differs" in result.stderr
    assert not output.exists()
    assert not log.exists()


def test_previous_environment_builder_pin_fails_before_output_or_child(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["benchmark_environment"]["builder_source_sha256"] = (
        "d3be9f5437306dcd5b25cf3158bef9f2350e9e65b14f14964ce74795cecedfc6"
    )
    _write_json(config_path, config)
    log = tmp_path / "fake-python.jsonl"
    result = _run(controller, config_path, log)
    assert result.returncode != 0
    assert "environment receipt identity/scope differs" in result.stderr
    assert not output.exists()
    assert not log.exists()


def test_previous_environment_review_marker_fails_before_output_or_child(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["benchmark_environment"]["review_marker_sha256"] = (
        "56f9b615554a9eb7ae708fbf815d3bf6a04cff60d60a85bd3b333ca59783d023"
    )
    _write_json(config_path, config)
    log = tmp_path / "fake-python.jsonl"
    result = _run(controller, config_path, log)
    assert result.returncode != 0
    assert "environment review marker SHA-256 differs" in result.stderr
    assert not output.exists()
    assert not log.exists()


@pytest.mark.parametrize(
    "field",
    (
        "launcher_source_sha256",
        "source_transport",
        "launcher_sys_path_policy",
        "preimport_sys_path_exact",
        "payload_import_sys_path_audited",
    ),
)
def test_missing_stdin_launcher_receipt_field_fails_before_output_or_child(
    tmp_path: Path, field: str
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt_path = Path(config["benchmark_environment"]["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    runtime = receipt["build_evidence"]["runtime_no_site"]
    runtime["bootstrap"].pop(field)
    runtime.pop("content_sha256")
    receipt["build_evidence"]["runtime_no_site"] = _bound(runtime)
    receipt.pop("content_sha256")
    receipt = _bound(receipt)
    _write_plain(receipt_path, receipt)
    config["benchmark_environment"]["receipt_sha256"] = _sha(receipt_path)
    config["benchmark_environment"]["receipt_content_sha256"] = receipt[
        "content_sha256"
    ]
    _write_json(config_path, config)
    log = tmp_path / "fake-python.jsonl"
    result = _run(controller, config_path, log)
    assert result.returncode != 0
    assert "environment runtime no-site contract differs" in result.stderr
    assert not output.exists()
    assert not log.exists()


def test_temp_path_in_pip_launcher_evidence_fails_before_output_or_child(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt_path = Path(config["benchmark_environment"]["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    download_argv = receipt["build_evidence"]["download"]["argv"]
    download_argv[5] = "/tmp/unreviewed-explicit-site-launcher.py"
    receipt.pop("content_sha256")
    receipt = _bound(receipt)
    _write_plain(receipt_path, receipt)
    config["benchmark_environment"]["receipt_sha256"] = _sha(receipt_path)
    config["benchmark_environment"]["receipt_content_sha256"] = receipt[
        "content_sha256"
    ]
    _write_json(config_path, config)
    log = tmp_path / "fake-python.jsonl"
    result = _run(controller, config_path, log)
    assert result.returncode != 0
    assert "download launcher/command evidence differs" in result.stderr
    assert not output.exists()
    assert not log.exists()


def test_dynamic_admission_receipt_attacks_all_fail_before_output_or_child(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    original_config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt_path = Path(original_config["benchmark_environment"]["receipt"])
    original_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    log = tmp_path / "fake-python.jsonl"
    for attack in (
        "missing_context",
        "tampered_authority",
        "old_schema",
        "temp_origin_count",
        "pyc_read_count",
        "restore_false",
    ):
        config = json.loads(json.dumps(original_config))
        receipt = json.loads(json.dumps(original_receipt))
        runtime = receipt["build_evidence"]["runtime_no_site"]
        target = runtime["target"]
        if attack == "missing_context":
            target.pop("torchvision_dynamic_module_admission")
        elif attack == "old_schema":
            runtime["schema_version"] = (
                "rachel-benchmark-runtime-no-site-evidence/1.0"
            )
            runtime["bootstrap"]["configuration_schema"] = (
                "rachel-benchmark-explicit-site-bootstrap/2.0"
            )
        else:
            admission = target["torchvision_dynamic_module_admission"]
            if attack == "tampered_authority":
                admission["authorities"]["generated_module_size"] += 1
            elif attack == "temp_origin_count":
                admission["observed"]["captured_temp_origin_module_count"] = 1
            elif attack == "pyc_read_count":
                admission["observed"]["pinned_source_pyc_read_count"] = 1
            else:
                admission["observed"]["reviewed_sys_path_restored"] = False
        if attack != "old_schema":
            target.pop("content_sha256")
            runtime["target"] = _bound(target)
        runtime.pop("content_sha256")
        receipt["build_evidence"]["runtime_no_site"] = _bound(runtime)
        receipt.pop("content_sha256")
        receipt = _bound(receipt)
        _write_plain(receipt_path, receipt)
        config["benchmark_environment"]["receipt_sha256"] = _sha(receipt_path)
        config["benchmark_environment"]["receipt_content_sha256"] = receipt[
            "content_sha256"
        ]
        _write_json(config_path, config)
        result = _run(controller, config_path, log)
        assert result.returncode != 0, attack
        assert any(
            token in result.stderr.lower() for token in ("runtime", "dynamic")
        ), (attack, result.stderr)
        assert not output.exists(), attack
        assert not log.exists(), attack
    _write_plain(receipt_path, original_receipt)
    _write_json(config_path, original_config)


@pytest.mark.parametrize(
    "attack",
    ("bytecode", "sitecustomize", "unexpected_executable", "symlink"),
)
def test_whole_root_inventory_rejects_unreviewed_execution_surfaces(
    tmp_path: Path, attack: str
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    if attack == "bytecode":
        (root / "payload.pyc").write_bytes(b"bytecode")
        match = "compiled/path/archive"
    elif attack == "sitecustomize":
        (root / "sitecustomize.py").write_text("pass\n", encoding="utf-8")
        match = "site/user customization"
    elif attack == "unexpected_executable":
        path = root / "launcher.txt"
        path.write_text("payload\n", encoding="utf-8")
        path.chmod(0o755)
        match = "unexpected executable"
    else:
        target = tmp_path / "outside.py"
        target.write_text("pass\n", encoding="utf-8")
        (root / "alias.py").symlink_to(target)
        match = "symlink forbidden"
    with pytest.raises(queue.QueueContractError, match=match):
        queue._source_inventory(root)


def test_whole_root_inventory_digest_changes_for_extra_python(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "reviewed.py").write_text("pass\n", encoding="utf-8")
    baseline = queue._source_inventory(root)
    (root / "late_payload.py").write_text("pass\n", encoding="utf-8")
    observed = queue._source_inventory(root)
    assert observed != baseline
    assert queue._source_inventory_sha256(observed) != (
        queue._source_inventory_sha256(baseline)
    )


@pytest.mark.parametrize(
    "package_parts",
    (
        ("staging", "pairwise_v0_2"),
        ("experiments", "rachel_n512_formal_30k"),
    ),
)
@pytest.mark.parametrize("missing_initializer_depth", (None, 1, 2))
def test_isolated_bootstrap_never_executes_or_accepts_same_named_site_parent(
    tmp_path: Path,
    package_parts: tuple[str, str],
    missing_initializer_depth: int | None,
) -> None:
    source = tmp_path / "immutable-source"
    fake_site = tmp_path / "fake-site-packages"
    attack_marker = tmp_path / "SITE_PARENT_EXECUTED"
    target_marker = tmp_path / "REVIEWED_TARGET_EXECUTED"
    module = ".".join((*package_parts, "bootstrap_probe"))

    for root in (source, fake_site):
        current = root
        for depth, part in enumerate(package_parts, start=1):
            current /= part
            current.mkdir(parents=True, exist_ok=True)
            initializer = current / "__init__.py"
            if root == source and missing_initializer_depth == depth:
                continue
            if root == fake_site:
                initializer.write_text(
                    "from pathlib import Path\nPath({!r}).write_text("
                    "'executed', encoding='utf-8')\n".format(
                        str(attack_marker)
                    ),
                    encoding="utf-8",
                )
            else:
                initializer.write_text("# reviewed package\n", encoding="utf-8")

    target = source.joinpath(*package_parts, "bootstrap_probe.py")
    target.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text('reviewed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with _explicit_site_subprocess(
        queue.ISOLATED_MODULE_BOOTSTRAP,
        [str(source), module, str(target_marker)],
        explicit_site_directories=[fake_site, source],
    ) as (command, pass_fds, bootstrap):
        result = subprocess.run(
            command,
            pass_fds=pass_fds,
            check=False,
            env=environment,
            input=bootstrap,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    assert not attack_marker.exists()
    if missing_initializer_depth is None:
        assert result.returncode == 0, result.stderr
        assert target_marker.read_text(encoding="utf-8") == "reviewed"
    else:
        assert result.returncode != 0
        assert "initializer" in result.stderr
        assert not target_marker.exists()


def test_static_queue_child_import_closure_matches_frozen_whitelist() -> None:
    observed = _static_queue_child_source_closure()
    assert observed == queue.QUEUE_CHILD_TRANSITIVE_SOURCE_RELATIVE_PATHS
    assert {
        "staging/pairwise_v0_2/baselines/"
        "rachel_same_data_benchmark_eval_adapter.py",
        "staging/pairwise_v0_2/models/coarse.py",
        "staging/pairwise_v0_2/models/frozen_transport_matrix_readout.py",
        "staging/pairwise_v0_2/models/local_matcher.py",
        "staging/pairwise_v0_2/models/optimal_transport.py",
        "staging/pairwise_v0_2/models/pairwise.py",
    } <= observed


def test_readonly_exact_closure_imports_and_all_child_help_entries_succeed(
    tmp_path: Path,
) -> None:
    snapshot = _execution_snapshot(tmp_path)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["HOME"] = str(tmp_path / "isolated-home")
    Path(environment["HOME"]).mkdir()
    try:
        complete = queue._complete_source_inventory(snapshot)
        permissions = queue._validate_immutable_source_permissions(
            snapshot, complete
        )
        assert permissions["regular_file_mode_octal"] == "0444"
        assert permissions["directory_mode_octal"] == "0555"
        modules = [
            (_relative_source_module(relative), relative)
            for relative in sorted(
                queue.QUEUE_CHILD_TRANSITIVE_SOURCE_RELATIVE_PATHS
            )
        ]
        import_probe = r'''import importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
modules = json.loads(sys.argv[2])
if len([value for value in sys.path if pathlib.Path(value).resolve() == root]) != 1:
    raise SystemExit("reviewed source import root differs")
for module_name, relative in modules:
    module = importlib.import_module(module_name)
    expected = (root / relative).resolve(strict=True)
    origin = getattr(module, "__file__", None)
    if origin is None or pathlib.Path(origin).resolve(strict=True) != expected:
        raise SystemExit("module origin differs: " + module_name)
print(json.dumps({"status": "all_reviewed_modules_imported", "count": len(modules)}))
'''
        with _explicit_site_subprocess(
            import_probe, [str(snapshot), json.dumps(modules)]
        ) as (command, pass_fds, bootstrap):
            imported = subprocess.run(
                command,
                pass_fds=pass_fds,
                check=False,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                input=bootstrap,
                text=True,
                timeout=90,
            )
        assert imported.returncode == 0, imported.stderr
        assert json.loads(imported.stdout) == {
            "status": "all_reviewed_modules_imported",
            "count": len(modules),
        }

        help_entries = [
            (module, ["--help"])
            for module in sorted(
                queue.QUEUE_CHILD_MODULES
                | {"staging.pairwise_v0_2.training.rachel_n512_sealed_test"}
            )
        ]
        help_entries.extend(
            [
                (
                    str(queue.METHODS["shreddingnet"]["runner_module"]),
                    [subcommand, "--help"],
                )
                for subcommand in ("audit", "gpu-smoke", "train")
            ]
        )
        for module, help_arguments in help_entries:
            with _explicit_site_subprocess(
                queue.ISOLATED_MODULE_BOOTSTRAP,
                [str(snapshot), module, *help_arguments],
            ) as (command, pass_fds, bootstrap):
                result = subprocess.run(
                    command,
                    pass_fds=pass_fds,
                    check=False,
                    cwd=tmp_path,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    input=bootstrap,
                    text=True,
                    timeout=90,
                )
            assert result.returncode == 0, (module, help_arguments, result.stderr)
    finally:
        _unseal_source_tree(snapshot)


def test_deleting_any_transitive_closure_member_fails_source_preflight(
    tmp_path: Path,
) -> None:
    snapshot = _execution_snapshot(tmp_path)
    stash = tmp_path / "closure-stash"
    stash.mkdir()
    try:
        baseline = queue._source_inventory(snapshot)
        queue._validate_minimal_execution_source_paths(baseline)
        for index, relative in enumerate(
            sorted(queue.QUEUE_CHILD_TRANSITIVE_SOURCE_RELATIVE_PATHS)
        ):
            path = snapshot / relative
            stored = stash / ("{:02d}-".format(index) + path.name)
            path.parent.chmod(0o755)
            path.rename(stored)
            path.parent.chmod(queue.IMMUTABLE_SOURCE_DIRECTORY_MODE)
            try:
                observed = queue._source_inventory(snapshot)
                with pytest.raises(queue.QueueContractError) as raised:
                    queue._validate_minimal_execution_source_paths(observed)
                assert relative in str(raised.value)
            finally:
                path.parent.chmod(0o755)
                stored.rename(path)
                path.parent.chmod(queue.IMMUTABLE_SOURCE_DIRECTORY_MODE)
        assert queue._source_inventory(snapshot) == baseline
        queue._validate_immutable_source_permissions(
            snapshot, queue._complete_source_inventory(snapshot)
        )
    finally:
        _unseal_source_tree(snapshot)


def test_source_permission_preflight_rejects_writable_file_or_directory(
    tmp_path: Path,
) -> None:
    snapshot = _execution_snapshot(tmp_path)
    target = snapshot / queue.CONTROLLER_RELATIVE_PATH
    directory = snapshot / "staging/pairwise_v0_2/models"
    try:
        target.chmod(0o644)
        with pytest.raises(
            queue.QueueContractError, match="regular-file mode must be 0444"
        ):
            queue._validate_immutable_source_permissions(
                snapshot, queue._complete_source_inventory(snapshot)
            )
        target.chmod(queue.IMMUTABLE_SOURCE_FILE_MODE)
        directory.chmod(0o755)
        with pytest.raises(
            queue.QueueContractError, match="directory mode must be 0555"
        ):
            queue._validate_immutable_source_permissions(
                snapshot, queue._complete_source_inventory(snapshot)
            )
        directory.chmod(queue.IMMUTABLE_SOURCE_DIRECTORY_MODE)
    finally:
        _unseal_source_tree(snapshot)


def test_reviewed_wheel_lock_drift_fails_before_output_or_child(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    wheel_lock = (
        Path(config["immutable_source_root"])
        / queue.ENVIRONMENT_REVIEWED_WHEEL_LOCK_RELATIVE_PATH
    )
    with wheel_lock.open("ab") as stream:
        stream.write(b"\n")
    result = _run(controller, config_path, tmp_path / "fake-python.jsonl")
    assert result.returncode != 0
    assert not output.exists()
    assert not (tmp_path / "fake-python.jsonl").exists()
    assert "wheel lock" in result.stderr


def test_extra_nonexecution_source_member_fails_complete_root_whitelist(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = Path(config["immutable_source_root"])
    (source / "unreviewed-note.txt").write_text("extra\n", encoding="utf-8")
    result = _run(controller, config_path, tmp_path / "fake-python.jsonl")
    assert result.returncode != 0
    assert not output.exists()
    assert "minimal immutable source member whitelist" in result.stderr


def test_review_marker_cannot_authorize_python_outside_minimal_whitelist(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = Path(config["immutable_source_root"])
    (source / "late_payload.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    marker = source / config["controller"]["review_marker_relative_path"]
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    inventory = queue._source_inventory(source)
    marker_value["source_inventory"] = {
        "scope": queue.SOURCE_INVENTORY_SCOPE,
        "file_count": len(inventory),
        "sha256": queue._source_inventory_sha256(inventory),
    }
    _write_json(marker, marker_value)
    config["controller"]["review_marker_sha256"] = _sha(marker)
    _write_json(config_path, config)
    result = _run(controller, config_path, tmp_path / "fake-python.jsonl")
    assert result.returncode != 0
    assert not output.exists()
    assert "minimal reviewed execution whitelist" in result.stderr


def test_environment_python_fd_anchor_never_executes_swapped_outside_root(
    tmp_path: Path,
) -> None:
    _controller, config_path, _output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    environment = queue._validate_environment(config["benchmark_environment"])
    descriptor, anchored_root, anchored_python = (
        queue._open_anchored_environment_python(environment)
    )
    environment_root = Path(environment["root"])
    held_root = tmp_path / "held-benchmark-environment"
    outside = tmp_path / "outside-environment"
    outside_python = outside / "bin/python"
    outside_python.parent.mkdir(parents=True)
    outside_marker = outside / "EXECUTED_OUTSIDE"
    outside_python.write_text(
        "#!/bin/sh\n: > " + str(outside_marker) + "\nexit 113\n",
        encoding="utf-8",
    )
    outside_python.chmod(0o755)
    environment_root.rename(held_root)
    environment_root.symlink_to(outside, target_is_directory=True)
    try:
        if anchored_root.startswith("/proc/") and "/fd/" in anchored_root:
            result = subprocess.run(
                [anchored_python, "--intentionally-invalid-fixture-probe"],
                pass_fds=(descriptor,),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert result.returncode != 113
        else:
            assert sys.platform == "darwin"
        assert not outside_marker.exists()
        with pytest.raises(
            queue.QueueContractError,
            match="lexical/held root identity changed",
        ):
            queue._close_and_revalidate_environment_anchor(
                descriptor, environment
            )
    finally:
        if environment_root.is_symlink():
            environment_root.unlink()
        if held_root.exists():
            held_root.rename(environment_root)


def test_spawn_uses_held_linux_environment_for_python_home_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = {
        "output_root": str(tmp_path / "output"),
        "source_root": "/immutable/source",
        "environment": {"root": "/lexical/environment"},
        "controller": {"source_inventory_sha256": "c" * 64},
        "environment_source_authority": {
            "builder": {
                "path": str(ROOT / queue.ENVIRONMENT_BUILDER_RELATIVE_PATH)
            }
        },
    }
    controller = queue.BenchmarkQueue(authority)
    controller.output_root.mkdir()
    controller.control_root.mkdir()
    controller.log_root.mkdir()
    monkeypatch.setattr(controller, "_verify_source_and_reviews", lambda: None)
    monkeypatch.setattr(
        controller, "_verify_environment_and_official", lambda: None
    )
    monkeypatch.setattr(controller, "_verify_dataset_binding", lambda: None)
    monkeypatch.setattr(
        queue,
        "_open_anchored_environment_python",
        lambda _environment: (
            77,
            "/proc/31337/fd/77",
            "/proc/31337/fd/77/bin/python",
        ),
    )
    held_sites = {
        name: [
            {
                "canonical_path": "/held/" + name,
                "descriptor_path": "/proc/1/fd/" + str(80 + index),
                "device": 1,
                "inode": 2 + index,
                "role": "audit_0",
                "audit_content_sha256": "a" * 64,
            }
        ]
        for index, name in enumerate(("base", "resolved_base", "target"))
    }
    monkeypatch.setattr(
        queue,
        "_open_anchored_environment_sites",
        lambda _environment: (held_sites, (80, 81, 82)),
    )
    source_binding = {
        "canonical_path": authority["source_root"],
        "descriptor_path": "/proc/1/fd/83",
        "device": 1,
        "inode": 5,
        "role": "immutable_source",
        "audit_content_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        queue,
        "_open_anchored_source_binding",
        lambda _source_root, _inventory_sha256: (83, source_binding),
    )
    configuration_calls = []

    def fixture_site_configuration(
        environment: object, sites: object, source: object
    ) -> dict[str, object]:
        configuration_calls.append((environment, sites, source))
        return {
            "schema_version": "fixture",
            "no_site_sys_path": [],
            "content_sha256": "b" * 64,
        }

    monkeypatch.setattr(
        queue,
        "_explicit_site_child_configuration",
        fixture_site_configuration,
    )
    close_order = []
    closed = []
    monkeypatch.setattr(
        queue,
        "_close_and_revalidate_environment_anchor",
        lambda descriptor, environment: (
            closed.append((descriptor, environment)),
            close_order.append("environment"),
        ),
    )
    closed_sites = []
    monkeypatch.setattr(
        queue,
        "_close_and_revalidate_environment_sites",
        lambda descriptors, sites: (
            closed_sites.append((descriptors, sites)),
            close_order.append("sites"),
        ),
    )
    closed_source = []
    monkeypatch.setattr(
        queue,
        "_close_and_revalidate_source_binding",
        lambda descriptor, binding: (
            closed_source.append((descriptor, binding)),
            close_order.append("source"),
        ),
    )
    monkeypatch.setattr(queue.os, "getpgid", lambda pid: pid)
    observed = {}

    class FakeStdin:
        def write(self, value: bytes) -> int:
            observed["stdin"] = value
            return len(value)

        def close(self) -> None:
            observed["stdin_closed"] = True

    class FakeChild:
        pid = 4242
        stdin = FakeStdin()

        @staticmethod
        def wait(timeout: object = None) -> int:
            del timeout
            return 0

        @staticmethod
        def kill() -> None:
            raise AssertionError("valid fixture child must not be killed")

        @staticmethod
        def poll() -> int:
            return 0

    def fake_popen(command: object, **kwargs: object) -> FakeChild:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return FakeChild()

    monkeypatch.setattr(queue.subprocess, "Popen", fake_popen)
    controller._spawn_child(
        "fixture_environment_verify",
        ("-m", queue.ENVIRONMENT_BUILDER_MODULE, "verify"),
    )
    assert observed["command"][0] == "/proc/31337/fd/77/bin/python"
    kwargs = observed["kwargs"]
    assert kwargs["pass_fds"] == (77, 80, 81, 82, 83)
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["HOME"] == "/proc/31337/fd/77"
    assert kwargs["env"]["PATH"].split(os.pathsep)[0] == (
        "/proc/31337/fd/77/bin"
    )
    assert configuration_calls == [
        (authority["environment"], held_sites, source_binding)
    ]
    assert observed["stdin"] == queue._reviewed_environment_runtime_sources(
        ROOT / queue.ENVIRONMENT_BUILDER_RELATIVE_PATH
    )["bootstrap"].encode()
    assert observed["stdin_closed"] is True
    assert closed_source == [(83, source_binding)]
    assert closed == [(77, authority["environment"])]
    assert closed_sites == [((80, 81, 82), held_sites)]
    assert close_order == ["source", "sites", "environment"]


def test_fixture_runs_fixed_audit_smoke_train_order_and_completes(
    tmp_path: Path,
) -> None:
    controller, config, output = _fixture(tmp_path)
    log = tmp_path / "fake-python.jsonl"
    result = _run(controller, config, log)
    assert result.returncode == 0, (result.stdout, result.stderr)
    invocations = [json.loads(line) for line in log.read_text().splitlines()]
    observed = []
    method_dataset_roots = []
    for row in invocations:
        module, arguments = row["module"], row["arguments"]
        if "pairingnet_gpu_smoke" in module:
            observed.append("pairing_smoke")
        elif "rachel_pairingnet_benchmark" in module and "--audit-only" in arguments:
            observed.append("pairing_audit")
        elif "rachel_pairingnet_benchmark" in module:
            observed.append("pairing_train")
        elif arguments and arguments[0] == "audit":
            observed.append("shredding_audit")
        elif arguments and arguments[0] == "gpu-smoke":
            observed.append("shredding_smoke")
        elif arguments and arguments[0] == "train":
            observed.append("shredding_train")
        if module in {
            queue.METHODS["pairingnet"]["runner_module"],
            queue.METHODS["pairingnet"]["smoke_module"],
            queue.METHODS["shreddingnet"]["runner_module"],
        }:
            method_dataset_roots.append(_option(arguments, "--dataset-root"))
    assert observed == [
        "pairing_audit",
        "pairing_smoke",
        "pairing_train",
        "shredding_audit",
        "shredding_smoke",
        "shredding_train",
    ]
    frozen_root = output / "control/train_val_asset_freeze/frozen_data"
    assert method_dataset_roots
    assert set(method_dataset_roots) == {str(frozen_root)}
    serialized_commands = json.dumps(invocations).lower()
    assert all(
        forbidden not in serialized_commands
        for forbidden in ("evaluate-val", "sealed-test", "real-external", "corrosion")
    )
    assert all(
        row["launcher_flags"] == ["-I", "-S", "-B", "-u", "-c"]
        and row["runtime_no_site"] is True
        and row["dynamic_admission_enabled"] is True
        and row["explicit_site_roles"]
        == ["audit_0", "audit_1", "immutable_source"]
        for row in invocations
    )
    if sys.platform.startswith("linux"):
        assert all(
            row["environment_home"].startswith("/proc/")
            and "/fd/" in row["environment_home"]
            and row["environment_path_first"].startswith("/proc/")
            and "/fd/" in row["environment_path_first"]
            for row in invocations
        )
    else:
        configured_environment = str(
            json.loads(config.read_text(encoding="utf-8"))["benchmark_environment"][
                "root"
            ]
        )
        assert all(
            row["environment_home"] == configured_environment
            and row["environment_path_first"]
            == str(Path(configured_environment) / "bin")
            for row in invocations
        )
    terminal = json.loads(
        (output / "control/terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "complete_same_data_benchmark_queue"
    assert terminal["training_started"] == {
        "pairingnet": True,
        "shreddingnet": True,
    }
    assert terminal["sealed_test_corrosion_real_opened"] is False
    smoke = json.loads(
        (output / "control/shreddingnet_gpu_smoke.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(smoke["stages"]) == {"coarse", "matching", "classify"}
    assert {
        stage: row["effective_batch_size"]
        for stage, row in smoke["stages"].items()
    } == {"coarse": 54, "matching": 36, "classify": 20}
    gate_path = output / "control/shreddingnet_gpu_smoke_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == (
        "shreddingnet_gpu_smoke_accepted_before_formal_train"
    )
    assert gate["receipt_file_sha256"] == _sha(
        output / "control/shreddingnet_gpu_smoke.json"
    )
    assert "shreddingnet_gpu_smoke_gate" in terminal["artifact_inventory"]


def test_missing_review_sha_fails_before_output_and_before_any_benchmark(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["methods"]["shreddingnet"]["runner_sha256"] = (
        "FILL_EXACT_REVIEWED_SHREDDINGNET_RUNNER_SHA256"
    )
    _write_json(config_path, config)
    log = tmp_path / "fake-python.jsonl"
    result = _run(controller, config_path, log)
    assert result.returncode != 0
    assert "supplied lowercase SHA-256" in result.stderr
    assert not output.exists()
    assert not log.exists()


def test_missing_asset_review_sha_fails_before_output_or_child(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["train_val_asset_freeze_authority"]["review_marker_sha256"] = (
        "FILL_EXACT_TRAIN_VAL_ASSET_FREEZE_REVIEW_MARKER_SHA256"
    )
    _write_json(config_path, config)
    result = _run(controller, config_path, tmp_path / "fake-python.jsonl")
    assert result.returncode != 0
    assert "supplied lowercase SHA-256" in result.stderr
    assert not output.exists()
    assert not (tmp_path / "fake-python.jsonl").exists()


def test_missing_controller_review_sha_fails_before_output_or_child(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["controller"]["review_marker_sha256"] = (
        "FILL_EXACT_QUEUE_REVIEW_MARKER_SHA256"
    )
    _write_json(config_path, config)
    result = _run(controller, config_path, tmp_path / "fake-python.jsonl")
    assert result.returncode != 0
    assert "supplied lowercase SHA-256" in result.stderr
    assert not output.exists()
    assert not (tmp_path / "fake-python.jsonl").exists()


def test_output_symlink_component_is_rejected_before_write_or_benchmark(
    tmp_path: Path,
) -> None:
    controller, config_path, _ = _fixture(tmp_path)
    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    alias = tmp_path / "output-parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["output_root"] = str(alias / "must-not-exist")
    _write_json(config_path, config)
    log = tmp_path / "fake-python.jsonl"
    result = _run(controller, config_path, log)
    assert result.returncode != 0
    assert "traverses a symlink" in result.stderr
    assert not (real_parent / "must-not-exist").exists()
    assert not log.exists()


def test_missing_environment_review_sha_fails_before_output_or_child(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["benchmark_environment"]["review_marker_sha256"] = "0" * 64
    _write_json(config_path, config)
    log = tmp_path / "fake-python.jsonl"
    result = _run(controller, config_path, log)
    assert result.returncode != 0
    assert "environment review marker SHA-256 differs" in result.stderr
    assert not output.exists()
    assert not log.exists()


def test_existing_output_is_never_reused_or_clobbered(tmp_path: Path) -> None:
    controller, config_path, output = _fixture(tmp_path)
    output.mkdir()
    sentinel = output / "user-owned.txt"
    sentinel.write_text("preserve-me", encoding="utf-8")
    log = tmp_path / "fake-python.jsonl"
    result = _run(controller, config_path, log)
    assert result.returncode != 0
    assert "must be fresh" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve-me"
    assert not (output / "control").exists()
    assert not log.exists()


def test_nonplateau_upstream_writes_terminal_failure_without_benchmark_start(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    n512_receipt = (
        Path(config["upstreams"]["n512"]["root"])
        / "run-fixture/run_receipt.json"
    )
    receipt = json.loads(n512_receipt.read_text(encoding="utf-8"))
    receipt["arm_results"][0]["stop_reason"] = "hard_cap_reached"
    receipt["arm_results"][0]["convergence_claim"] = (
        "not_established_before_hard_cap"
    )
    _write_json(n512_receipt, receipt)
    log = tmp_path / "fake-python.jsonl"
    result = _run(controller, config_path, log)
    assert result.returncode != 0
    terminal = json.loads(
        (output / "control/terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "failed_same_data_benchmark_queue"
    assert terminal["training_started"] == {
        "pairingnet": False,
        "shreddingnet": False,
    }
    assert not log.exists()


def test_gpu_smoke_failure_cannot_leave_success_or_start_formal_train(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    log = tmp_path / "fake-python.jsonl"
    result = _run(
        controller,
        config_path,
        log,
        extra_environment={"FAKE_FAIL_STAGE": "pairing_smoke"},
    )
    assert result.returncode != 0
    terminal = json.loads(
        (output / "control/terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "failed_same_data_benchmark_queue"
    assert terminal["stage"] == "pairingnet_gpu_smoke"
    assert terminal["training_started"] == {
        "pairingnet": False,
        "shreddingnet": False,
    }
    assert not (output / "control/pairingnet_gpu_smoke.json").exists()
    assert not (output / "pairingnet").exists()
    invocations = [json.loads(line) for line in log.read_text().splitlines()]
    method_rows = [
        row
        for row in invocations
        if row["module"]
        in {
            queue.METHODS["pairingnet"]["runner_module"],
            queue.METHODS["pairingnet"]["smoke_module"],
            queue.METHODS["shreddingnet"]["runner_module"],
        }
    ]
    assert len(method_rows) == 2
    assert "--audit-only" in method_rows[0]["arguments"]
    assert "pairingnet_gpu_smoke" in method_rows[1]["module"]


@pytest.mark.parametrize(
    "defect", ("device", "workers", "nonfinite", "vram", "official_audit", "order")
)
def test_pairing_smoke_contract_defect_blocks_acceptance_and_formal_train(
    tmp_path: Path, defect: str
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    result = _run(
        controller,
        config_path,
        tmp_path / "fake-python.jsonl",
        extra_environment={"pair_smoke_defect": defect},
    )
    assert result.returncode != 0
    terminal = json.loads(
        (output / "control/terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "failed_same_data_benchmark_queue"
    assert terminal["stage"].endswith("pairingnet_gpu_smoke")
    assert terminal["training_started"] == {
        "pairingnet": False,
        "shreddingnet": False,
    }
    assert (output / "control/pairingnet_gpu_smoke.json").is_file()
    assert not (output / "control/pairingnet_gpu_smoke_gate.json").exists()
    assert not (output / "pairingnet").exists()


def test_pairing_smoke_no_replace_gate_detects_valid_receipt_replacement(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    result = _run(
        controller,
        config_path,
        tmp_path / "fake-python.jsonl",
        extra_environment={"replace_pair_smoke_after_gate": "1"},
    )
    assert result.returncode != 0
    terminal = json.loads(
        (output / "control/terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert terminal["training_started"] == {
        "pairingnet": False,
        "shreddingnet": False,
    }
    assert "changed before formal train" in terminal["error"]
    assert (output / "control/pairingnet_gpu_smoke_gate.json").is_file()
    assert not (output / "pairingnet").exists()


def test_environment_drift_after_smoke_blocks_acceptance_and_formal_train(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    result = _run(
        controller,
        config_path,
        tmp_path / "fake-python.jsonl",
        extra_environment={"environment_drift_after_pair_smoke": "1"},
    )
    assert result.returncode != 0
    terminal = json.loads(
        (output / "control/terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert terminal["training_started"] == {
        "pairingnet": False,
        "shreddingnet": False,
    }
    assert terminal["stage"] == "environment_verify_after_pairingnet_gpu_smoke"
    assert not (output / "control/pairingnet_gpu_smoke_gate.json").exists()
    assert not (output / "pairingnet").exists()


def test_pairing_smoke_cuda_preflight_failure_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(
        pairingnet_gpu_smoke.torch.cuda, "is_available", lambda: False
    )
    with pytest.raises(
        pairingnet_gpu_smoke.PairingNetSmokeError,
        match="available CUDA device",
    ):
        pairingnet_gpu_smoke.run_smoke(
            dataset_root=tmp_path / "must-not-open-dataset",
            official_source_root=tmp_path / "must-not-open-official",
            output=output,
            device="cuda:0",
            precision="fp32",
            num_workers=0,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "defect",
    [
        "list_schema",
        "bad_batch",
        "nonfinite_metric",
        "bad_adapter_sha",
        "bad_content_sha",
    ],
)
def test_shredding_smoke_contract_defect_blocks_formal_train_and_acceptance_gate(
    tmp_path: Path, defect: str
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    log = tmp_path / "fake-python.jsonl"
    result = _run(
        controller,
        config_path,
        log,
        extra_environment={"FAKE_SHREDDING_SMOKE_DEFECT": defect},
    )
    assert result.returncode != 0
    terminal = json.loads(
        (output / "control/terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "failed_same_data_benchmark_queue"
    assert terminal["stage"].endswith("shreddingnet_gpu_smoke")
    assert terminal["training_started"] == {
        "pairingnet": True,
        "shreddingnet": False,
    }
    assert not (output / "control/shreddingnet_gpu_smoke_gate.json").exists()
    assert not (output / "shreddingnet").exists()
    invocations = [json.loads(line) for line in log.read_text().splitlines()]
    shredding_commands = [
        row["arguments"][0]
        for row in invocations
        if row["module"] == queue.METHODS["shreddingnet"]["runner_module"]
    ]
    assert shredding_commands == ["audit", "gpu-smoke"]


def test_frozen_semantic_asset_mutation_stops_queue_before_second_method(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    log = tmp_path / "fake-python.jsonl"
    result = _run(
        controller,
        config_path,
        log,
        extra_environment={"FAKE_MUTATE_AFTER_PAIR_TRAIN": "1"},
    )
    assert result.returncode != 0
    terminal = json.loads(
        (output / "control/terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "failed_same_data_benchmark_queue"
    assert terminal["training_started"] == {
        "pairingnet": True,
        "shreddingnet": False,
    }
    assert "asset-freeze verifier" in terminal["error"] or (
        "train_val_asset_freeze_verify" in terminal["stage"]
    )
    invocations = [json.loads(line) for line in log.read_text().splitlines()]
    modules = [row["module"] for row in invocations]
    assert all("shreddingnet" not in module for module in modules)


def test_recursive_terminal_inventory_rejects_late_output_symlink(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    result = _run(
        controller,
        config_path,
        tmp_path / "fake-python.jsonl",
        extra_environment={"inject_output_symlink": "1"},
    )
    assert result.returncode != 0
    terminal = json.loads(
        (output / "control/terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "failed_same_data_benchmark_queue"
    assert terminal["training_started"] == {
        "pairingnet": True,
        "shreddingnet": True,
    }
    assert "symlink" in terminal["error"]
    assert not (output / "control/content_sha256_inventory.json").exists()


def test_sigterm_while_waiting_writes_atomic_terminal_receipt(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(
        tmp_path, completed_upstreams=False
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["wait"]["timeout_seconds"] = 0
    _write_json(config_path, config)
    log = tmp_path / "fake-python.jsonl"
    source = Path(config["immutable_source_root"])
    _seal_source_tree(source)
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(controller),
            "--config",
            str(config_path),
        ],
        env={**os.environ, "FAKE_QUEUE_LOG": str(log)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    launch = output / "control/launch_authority.json"
    deadline = time.monotonic() + 15
    try:
        while not launch.is_file() and process.poll() is None:
            if time.monotonic() >= deadline:
                raise AssertionError("controller did not enter upstream wait")
            time.sleep(0.02)
        assert launch.is_file()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 143, (stdout, stderr)
        terminal = json.loads(
            (output / "control/terminal_receipt.json").read_text(encoding="utf-8")
        )
        assert terminal["status"] == "failed_same_data_benchmark_queue"
        assert terminal["termination"] == "SIGTERM"
        assert terminal["training_started"] == {
            "pairingnet": False,
            "shreddingnet": False,
        }
        assert terminal["sealed_test_corrosion_real_opened"] is False
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        _unseal_source_tree(source)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="formal process-group signal semantics are Linux-only",
)
def test_sigterm_during_child_kills_registered_process_group_and_writes_terminal(
    tmp_path: Path,
) -> None:
    controller, config_path, output = _fixture(tmp_path)
    controls_path = tmp_path / "fake-controls.json"
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    controls["sleep_pair_audit"] = True
    _write_json(controls_path, controls)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = Path(config["immutable_source_root"])
    _seal_source_tree(source)
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(controller),
            "--config",
            str(config_path),
        ],
        env={
            **os.environ,
            "PATH": str(tmp_path / "fake-bin")
            + os.pathsep
            + os.environ.get("PATH", ""),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pid_path = tmp_path / "grandchild.pid"
    deadline = time.monotonic() + 30
    try:
        while not pid_path.is_file() and process.poll() is None:
            if time.monotonic() >= deadline:
                raise AssertionError("controller did not register the sleeping child")
            time.sleep(0.02)
        assert pid_path.is_file()
        grandchild_pid = int(pid_path.read_text(encoding="ascii"))
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 143, (stdout, stderr)
        terminal = json.loads(
            (output / "control/terminal_receipt.json").read_text(encoding="utf-8")
        )
        assert terminal["status"] == "failed_same_data_benchmark_queue"
        assert terminal["termination"] == "SIGTERM"
        assert terminal["training_started"] == {
            "pairingnet": False,
            "shreddingnet": False,
        }
        child_deadline = time.monotonic() + 5
        while time.monotonic() < child_deadline:
            status = subprocess.run(
                ["/bin/ps", "-p", str(grandchild_pid), "-o", "stat="],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout.strip()
            if not status or status.startswith("Z"):
                break
            time.sleep(0.05)
        else:
            raise AssertionError("registered child process group survived SIGTERM")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        _unseal_source_tree(source)
