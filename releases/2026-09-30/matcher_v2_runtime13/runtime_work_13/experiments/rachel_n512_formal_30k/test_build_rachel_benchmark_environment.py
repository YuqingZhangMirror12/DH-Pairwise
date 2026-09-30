from __future__ import annotations

import ast
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Dict, Mapping, Optional, Tuple
import zipfile

import pytest

from experiments.rachel_n512_formal_30k import (
    build_rachel_benchmark_environment as subject,
)


SOURCE = Path(subject.__file__).resolve()
REVIEWED_LOCK = SOURCE.with_name("rachel_benchmark_reviewed_wheel_lock.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_probe(
    python: Path, execution_anchor: Optional[subject._ExecutionAnchor] = None
) -> Dict[str, object]:
    return {
        "executable": str(python.resolve()),
        "prefix": str(python.parent.parent.resolve()),
        "base_prefix": "/fixture/system",
        "python_version": "3.12.4",
        "isolated": True,
        "no_site": True,
        "dont_write_bytecode": True,
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "sys_path": ["/fixture/runtime/lib/python3.12"],
    }


def _tracked(names: Tuple[Tuple[str, str], ...]) -> Dict[str, Dict[str, str]]:
    result = {}
    for distribution, module in names:
        version = "fixture-" + distribution
        if distribution == "numpy":
            version = subject.EXPECTED_NUMPY_ADAPTATION_VERSION
        result[distribution] = {
            "distribution_version": version,
            "distribution_location": "/fixture/base/site-packages",
            "module": module,
            "module_version": version,
            "import_origin": "/fixture/base/site-packages/" + module + "/__init__.py",
        }
    return result


def _base_probe(base_python: Path) -> Dict[str, object]:
    tracked = _tracked(subject.BASE_TRACKED_DISTRIBUTIONS)
    tracked["torch"]["distribution_version"] = "2.5.1+cu124"
    tracked["torch"]["module_version"] = "2.5.1+cu124"
    return {
        "executable": str(base_python.resolve()),
        "prefix": str(base_python.parent.parent.resolve()),
        "base_prefix": "/fixture/system",
        "python_version": "3.12.4",
        "user_site_enabled": False,
        "sitecustomize_loaded": False,
        "usercustomize_loaded": False,
        "sys_path": [
            "/fixture/runtime/lib/python3.12",
            "/fixture/base/site-packages",
        ],
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "torch_module_version": "2.5.1+cu124",
        "torch_cuda_build_version": "12.4",
        "opencv_python_headless_version": None,
        "tracked": tracked,
        "local_distributions": [],
        "torchvision_dynamic_module_admission": (
            subject._successful_torchvision_dynamic_admission_evidence()
        ),
    }


def _live_probe(root: Path, python: Path) -> Dict[str, object]:
    return {
        "executable": str(python.resolve()),
        "prefix": str(root),
        "base_prefix": "/fixture/system",
        "python_version": "3.12.4",
        "user_site_enabled": False,
        "packages": {
            "torch": "2.5.1+cu124",
            "torch-geometric": "2.6.1",
            "opencv-python": "4.10.0.84",
            "opencv-python-headless": None,
            "numpy": subject.EXPECTED_NUMPY_ADAPTATION_VERSION,
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


def _detailed_probe(
    root: Path, python: Path, base: Mapping[str, object]
) -> Dict[str, object]:
    base_tracked = base["tracked"]
    assert isinstance(base_tracked, dict)
    tracked = json.loads(json.dumps(base_tracked))
    local = []
    for distribution, version, module in subject.ADDITIONS:
        tracked[distribution] = {
            "distribution_version": version,
            "distribution_location": str(root / "lib/python3.12/site-packages"),
            "module": module,
            "module_version": version,
            "import_origin": str(
                root / "lib/python3.12/site-packages" / module / "__init__.py"
            ),
        }
        local.append({"name": distribution, "version": version, "location": str(root)})
    return {
        "executable": str(python.resolve()),
        "prefix": str(root),
        "base_prefix": "/fixture/system",
        "python_version": "3.12.4",
        "user_site_enabled": False,
        "sitecustomize_loaded": False,
        "usercustomize_loaded": False,
        "sys_path": [
            "/fixture/runtime/lib/python3.12",
            str(root / "lib/python3.12/site-packages"),
            "/fixture/base/site-packages",
        ],
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "torch_module_version": "2.5.1+cu124",
        "torch_cuda_build_version": "12.4",
        "opencv_python_headless_version": None,
        "tracked": tracked,
        "local_distributions": local,
        "torchvision_dynamic_module_admission": (
            subject._successful_torchvision_dynamic_admission_evidence()
        ),
    }


def _inventory(root: Path, python: Path) -> Dict[str, object]:
    return subject._content_bound(
        {
            "schema_version": "rachel-benchmark-import-inventory/1.0",
            "environment_root": str(root),
            "python": str(python.resolve()),
            "python_sha256": _sha(python.resolve()),
            "prefix": str(root),
            "base_prefix": "/fixture/system",
            "tracked": {"fixture": {"module_tree": {"content_sha256": "a" * 64}}},
        }
    )


def _official(path: Path) -> Dict[str, object]:
    return {
        "commit": subject.OFFICIAL_SHREDDINGNET_COMMIT,
        "env_yaml": str(path),
        "env_yaml_sha256": "f" * 64,
        "audited_pins": {
            "torch": "2.5.1",
            "torch-geometric": "2.6.1",
            "opencv-python": "4.10.0.84",
            "torchvision": "0.20.1",
            "numpy": "2.2.3",
            "scipy": "1.14.1",
            "pillow": "11.0.0",
        },
    }


def _fixture_audit(prefix: Path) -> Dict[str, object]:
    stdlib = prefix / "lib/python3.12"
    site_packages = stdlib / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    stdlib_metadata = stdlib.stat()
    site_metadata = site_packages.stat()
    return subject._content_bound(
        {
            "schema_version": "rachel-benchmark-active-site-audit/1.0",
            "prefix": str(prefix),
            "site_directories": [str(site_packages)],
            "startup_search_directories": [
                {
                    "path": str(stdlib),
                    "device": stdlib_metadata.st_dev,
                    "inode": stdlib_metadata.st_ino,
                },
                {
                    "path": str(site_packages),
                    "device": site_metadata.st_dev,
                    "inode": site_metadata.st_ino,
                },
            ],
            "startup_artifacts": [],
            "declared_pth_paths": [],
            "activated_paths": [],
            "runtime_no_site": True,
            "pth_execution_policy": "present_but_never_executed",
        }
    )


def _reviewed_lock_binding(path: Path) -> Dict[str, object]:
    return {
        "path": str(path),
        "file_sha256": subject.REVIEWED_WHEEL_LOCK_FILE_SHA256,
        "content_sha256": "1" * 64,
        "packages": [
            {
                "distribution": distribution,
                "version": version,
                "filename": distribution + ".whl",
                "sha256": str(index + 1) * 64,
                "url": "https://files.pythonhosted.org/" + distribution + ".whl",
            }
            for index, (distribution, version, _module) in enumerate(subject.ADDITIONS)
        ],
        "target": {
            "python_major_minor": "3.12",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
        },
        "official_index": {},
        "review": {},
    }


def _descriptor_guard(platform: str, os_module=os):
    namespace = {
        "os": os_module,
        "stat": stat,
        "sys": SimpleNamespace(platform=platform),
    }
    exec(subject.EXPLICIT_SITE_DESCRIPTOR_GUARD_CODE, namespace, namespace)
    return namespace["_held_site_descriptor"]


def _install_build_mocks(monkeypatch, tmp_path: Path):
    base_python = tmp_path / "base" / "bin/python"
    base_python.parent.mkdir(parents=True)
    base_python.write_bytes(b"fixture-base-python")
    base_python.chmod(0o755)
    official_yaml = tmp_path / "official-env.yaml"
    official_yaml.write_text("fixture\n", encoding="utf-8")
    root = tmp_path / "benchmark-env"
    base = _base_probe(base_python)
    lock_binding = _reviewed_lock_binding(REVIEWED_LOCK)

    monkeypatch.setattr(subject, "_require_safe_controller_runtime", lambda: None)
    monkeypatch.setattr(subject, "_safe_runtime_probe", _safe_probe)
    monkeypatch.setattr(
        subject,
        "_anchored_safe_runtime_probe",
        lambda python, prefix: _safe_probe(python),
    )
    monkeypatch.setattr(subject, "_validate_safe_runtime_probe", lambda *a, **k: None)
    monkeypatch.setattr(
        subject,
        "_audit_active_site",
        lambda prefix, **kwargs: _fixture_audit(prefix),
    )
    monkeypatch.setattr(
        subject,
        "_validate_wheel_lock",
        lambda path, expected, safe: lock_binding,
    )
    monkeypatch.setattr(
        subject, "_validate_official_environment", lambda path: _official(path)
    )
    monkeypatch.setattr(subject, "_base_probe", lambda python, **kwargs: base)
    monkeypatch.setattr(subject, "_validate_base_probe", lambda *a, **k: None)

    def subprocess_run(arguments, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        if "venv" in arguments:
            assert arguments[1:4] == ("-I", "-S", "-B")
            assert "--without-pip" in arguments
            assert "--system-site-packages" in arguments
            (root / "bin").mkdir(parents=True)
            (root / "bin/python").write_bytes(b"fixture-target-python")
            (root / "bin/python").chmod(0o755)
            (root / "lib/python3.12/site-packages").mkdir(parents=True)
            (root / "pyvenv.cfg").write_text(
                "include-system-site-packages = true\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(subject.subprocess, "run", subprocess_run)

    def download(
        python,
        environment_root,
        reviewed_lock,
        execution_anchor=None,
        *args,
        **kwargs,
    ):
        evidence = environment_root / ".environment_build_evidence"
        evidence.mkdir()
        wheelhouse = evidence / "wheelhouse"
        wheelhouse.mkdir()
        lock = evidence / "requirements.sha256.txt"
        lock.write_text("fixture-lock\n", encoding="utf-8")
        lock.chmod(0o444)
        wheelhouse.chmod(0o555)
        wheels = [
            {
                "filename": distribution + ".whl",
                "distribution": distribution,
                "version": version,
                "sha256": str(index + 1) * 64,
                "size": index + 1,
            }
            for index, (distribution, version, _module) in enumerate(subject.ADDITIONS)
        ]
        return wheels, {
            "argv": ["pip", "download"],
            "argv_policy": (
                "fd_paths_canonicalized_to_lexical_paths_static_payloads_recorded_by_sha256"
            ),
            "stdout_sha256": "c" * 64,
            "stderr_sha256": "d" * 64,
            "requirements_lock": str(lock),
            "requirements_lock_sha256": _sha(lock),
        }

    def install(
        python,
        environment_root,
        reviewed_lock,
        execution_anchor=None,
        *args,
        **kwargs,
    ):
        report = environment_root / subject.INSTALL_REPORT_RELATIVE_PATH
        report.write_text("{}\n", encoding="utf-8")
        report.chmod(0o444)
        normalization = subject._content_bound(
            {
                "schema_version": (
                    "rachel-benchmark-pip-report-path-normalization/1.0"
                ),
                "policy": ("execution-anchor-file-url-to-canonical-lexical-wheel-path"),
                "entries": [
                    {
                        "filename": item["filename"],
                        "source_url": (
                            environment_root
                            / subject.WHEELHOUSE_RELATIVE_PATH
                            / item["filename"]
                        ).as_uri(),
                        "canonical_path": str(
                            environment_root
                            / subject.WHEELHOUSE_RELATIVE_PATH
                            / item["filename"]
                        ),
                        "canonical_url": (
                            environment_root
                            / subject.WHEELHOUSE_RELATIVE_PATH
                            / item["filename"]
                        ).as_uri(),
                    }
                    for item in reviewed_lock["packages"]
                ],
            }
        )
        return {
            "argv": ["pip", "install"],
            "argv_policy": (
                "fd_paths_canonicalized_to_lexical_paths_static_payloads_recorded_by_sha256"
            ),
            "stdout_sha256": "e" * 64,
            "stderr_sha256": "f" * 64,
            "report": str(report),
            "report_sha256": _sha(report),
            "report_path_normalization": normalization,
            "installed": [],
        }

    monkeypatch.setattr(subject, "_download_and_lock_wheels", download)
    monkeypatch.setattr(subject, "_install_locked_wheels", install)
    monkeypatch.setattr(subject, "_validate_target_probes", lambda **kwargs: None)
    monkeypatch.setattr(
        subject,
        "_functional_cpu_probe",
        lambda python, **kwargs: {"device": "cpu"},
    )
    monkeypatch.setattr(
        subject,
        "_pip_freeze",
        lambda python, **kwargs: {
            "lines": [],
            "canonical_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(
        subject,
        "_queue_live_probe",
        lambda python, **kwargs: _live_probe(root, python),
    )
    monkeypatch.setattr(
        subject,
        "_detailed_probe",
        lambda python, **kwargs: _detailed_probe(root, python, base),
    )
    monkeypatch.setattr(
        subject,
        "_import_inventory",
        lambda python, environment_root, **kwargs: _inventory(environment_root, python),
    )

    evidence_verifications = {"count": 0}

    def verify_evidence(environment_root, evidence, safe_probe, execution_anchor=None):
        evidence_verifications["count"] += 1
        if evidence_verifications["count"] == 1:
            assert not (environment_root / subject.RECEIPT_NAME).exists()
        evidence_root = environment_root / ".environment_build_evidence"
        assert stat.S_IMODE(evidence_root.stat().st_mode) == 0o555
        assert (
            stat.S_IMODE(
                (environment_root / subject.INVENTORY_RELATIVE_PATH).stat().st_mode
            )
            == 0o444
        )
        assert (
            stat.S_IMODE(
                (environment_root / subject.ACTIVE_SITE_RELATIVE_PATH).stat().st_mode
            )
            == 0o444
        )

    monkeypatch.setattr(subject, "_verify_build_evidence", verify_evidence)
    return base_python, official_yaml, root, base, lock_binding


def _build(monkeypatch, tmp_path: Path):
    base_python, official, root, base, lock_binding = _install_build_mocks(
        monkeypatch, tmp_path
    )
    result = subject.build_environment(
        base_python=base_python,
        environment_root=root,
        official_env_yaml=official,
        wheel_lock_path=REVIEWED_LOCK,
        expected_wheel_lock_sha256=subject.REVIEWED_WHEEL_LOCK_FILE_SHA256,
    )
    return result, base_python, official, root, base, lock_binding


def test_venv_creation_uses_lexical_target_with_held_root_and_parent(
    monkeypatch, tmp_path: Path
) -> None:
    base_python, official, root, _base, _lock = _install_build_mocks(
        monkeypatch, tmp_path
    )
    mocked_run = subject.subprocess.run
    observed = {}

    def inspect_run(arguments, **kwargs):
        if "venv" in arguments:
            observed["arguments"] = tuple(arguments)
            observed["pass_fds"] = tuple(kwargs["pass_fds"])
        return mocked_run(arguments, **kwargs)

    monkeypatch.setattr(subject.subprocess, "run", inspect_run)
    subject.build_environment(
        base_python=base_python,
        environment_root=root,
        official_env_yaml=official,
        wheel_lock_path=REVIEWED_LOCK,
        expected_wheel_lock_sha256=subject.REVIEWED_WHEEL_LOCK_FILE_SHA256,
    )

    arguments = observed["arguments"]
    descriptor_path = re.compile(
        r"^/(?:proc/(?:self|[0-9]+)|dev)/fd/[0-9]+(?:/|$)"
    )
    assert arguments[-1] == str(root)
    assert descriptor_path.match(arguments[-1]) is None
    assert descriptor_path.match(arguments[0]) is not None
    assert len(observed["pass_fds"]) >= 3


def _make_wheel(
    path: Path,
    distribution: str,
    version: str,
    *,
    tags: Tuple[str, ...] = ("py3-none-any",),
    bad_record: bool = False,
) -> None:
    dist_info_name = distribution.replace("-", "_") + "-" + version + ".dist-info"
    members = {
        distribution.replace("-", "_") + "/__init__.py": b"__version__ = 'fixture'\n",
        dist_info_name + "/METADATA": (
            "Metadata-Version: 2.1\nName: "
            + distribution
            + "\nVersion: "
            + version
            + "\n"
        ).encode("utf-8"),
        dist_info_name + "/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\n"
            + "".join("Tag: " + tag + "\n" for tag in tags)
        ).encode("utf-8"),
    }
    record_name = dist_info_name + "/RECORD"
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    for index, (name, payload) in enumerate(sorted(members.items())):
        digest = hashlib.sha256(payload).digest()
        if bad_record and index == 0:
            digest = b"\0" * 32
        encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        writer.writerow((name, "sha256=" + encoded, str(len(payload))))
    writer.writerow((record_name, "", ""))
    members[record_name] = stream.getvalue().encode("utf-8")
    with zipfile.ZipFile(str(path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(name, payload)


def _wheel_authority(path: Path, distribution: str, version: str) -> Dict[str, object]:
    stem = path.name[:-4].rsplit("-", 3)
    return {
        "filename": path.name,
        "distribution": distribution,
        "version": version,
        "sha256": _sha(path),
        "size": path.stat().st_size,
        "filename_tags": {
            "python": stem[1],
            "abi": stem[2],
            "platform": stem[3],
        },
    }


def test_source_is_python38_syntax_and_compiles_with_real_python38() -> None:
    ast.parse(SOURCE.read_text(encoding="utf-8"), feature_version=(3, 8))
    python38 = Path("/Users/yuqingzhang/opt/anaconda3/bin/python3.8")
    if python38.is_file():
        subprocess.run((str(python38), "-m", "py_compile", str(SOURCE)), check=True)


def test_live_probe_source_exactly_matches_queue_controller() -> None:
    controller = SOURCE.with_name("run_same_data_benchmark_queue.py")
    tree = ast.parse(controller.read_text(encoding="utf-8"))
    queue_probe: Optional[str] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_environment_probe":
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "code"
                    for target in statement.targets
                ):
                    queue_probe = ast.literal_eval(statement.value)
    assert queue_probe == subject.QUEUE_LIVE_PROBE_CODE


def test_all_runtime_wrappers_enable_pinned_torchvision_preload(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(subject, "_run_json_explicit_sites", observed)
    common = {
        "safe_probe": {},
        "audits": (),
    }
    python = tmp_path / "python"
    subject._queue_live_probe(python, **common)
    subject._detailed_probe(python, **common)
    subject._base_probe(python, **common)
    subject._functional_cpu_probe(python, **common)
    subject._import_inventory(python, tmp_path, **common)
    assert len(calls) == 5
    assert all(call.get("preload_torchvision") is True for call in calls)


def test_run_checked_uses_argument_vector_and_clears_python_pip_injection(
    monkeypatch,
) -> None:
    observed = {}
    monkeypatch.setenv("PIP_INDEX_URL", "https://untrusted.invalid/simple")
    monkeypatch.setenv("PYTHONSTARTUP", "/untrusted.py")

    def fake_run(arguments, **kwargs):
        observed["arguments"] = arguments
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(arguments, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    result = subject._run_checked(("python", "-I", "-c", "pass"), label="probe")
    assert result.stdout == "ok"
    assert observed["arguments"] == ("python", "-I", "-c", "pass")
    assert "shell" not in observed["kwargs"]
    environment = observed["kwargs"]["env"]
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in environment
    assert "PYTHONSTARTUP" not in environment
    assert "PIP_INDEX_URL" not in environment
    assert environment["PIP_CONFIG_FILE"] == os.devnull


def test_run_checked_converts_subprocess_failure(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(9, args[0])

    monkeypatch.setattr(subject.subprocess, "run", fail)
    with pytest.raises(subject.EnvironmentBuildError, match="probe failed"):
        subject._run_checked(("python",), label="probe")


def test_subprocess_anchor_survives_temporary_root_swap_without_outside_write(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    moved = tmp_path / "environment-moved"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    anchor = subject._ExecutionAnchor(root, "fixture environment")
    anchored_output = anchor.child_path(root / "child-output", "child output")
    code = r"""import os, pathlib, sys
root, moved, outside, output, descriptor = sys.argv[1:]
os.rename(root, moved)
os.symlink(outside, root)
if sys.platform.startswith("linux"):
    pathlib.Path(output).write_text("held-inode", encoding="utf-8")
else:
    child = os.open("child-output", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=int(descriptor))
    os.write(child, b"held-inode")
    os.close(child)
os.unlink(root)
os.rename(moved, root)
"""
    try:
        result = subject._run_checked(
            (
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                code,
                str(root),
                str(moved),
                str(outside),
                str(anchored_output),
                str(anchor.descriptor),
            ),
            label="temporary root-swap fixture",
            execution_anchor=anchor,
        )
        assert result.returncode == 0
    finally:
        anchor.close()
    assert (root / "child-output").read_text(encoding="utf-8") == "held-inode"
    assert list(outside.iterdir()) == []


def test_linux_subprocess_anchor_path_and_pass_fds_contract(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "environment"
    root.mkdir()
    anchor = subject._ExecutionAnchor(root, "fixture environment")
    observed = {}

    def fake_run(arguments, **kwargs):
        observed["arguments"] = arguments
        observed["pass_fds"] = kwargs["pass_fds"]
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(subject.sys, "platform", "linux")
    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    anchored_python = anchor.child_path(root / "bin/python", "fixture Python")
    try:
        subject._run_checked(
            (str(anchored_python), "-I", "-S", "-B", "-c", "pass"),
            label="Linux anchor contract",
            execution_anchor=anchor,
        )
        assert str(anchored_python) == (
            "/proc/"
            + str(anchor.owner_pid)
            + "/fd/"
            + str(anchor.descriptor)
            + "/bin/python"
        )
        assert observed["pass_fds"] == (anchor.descriptor,)
    finally:
        anchor.close()


def test_subprocess_anchor_fails_closed_on_persistent_root_symlink_swap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    moved = tmp_path / "environment-moved"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    anchor = subject._ExecutionAnchor(root, "fixture environment")
    anchored_output = anchor.child_path(root / "child-output", "child output")
    code = r"""import os, pathlib, sys
root, moved, outside, output, descriptor = sys.argv[1:]
os.rename(root, moved)
os.symlink(outside, root)
if sys.platform.startswith("linux"):
    pathlib.Path(output).write_text("held-inode", encoding="utf-8")
else:
    child = os.open("child-output", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=int(descriptor))
    os.write(child, b"held-inode")
    os.close(child)
"""
    try:
        with pytest.raises(subject.EnvironmentBuildError, match="symlink|identity"):
            subject._run_checked(
                (
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    code,
                    str(root),
                    str(moved),
                    str(outside),
                    str(anchored_output),
                    str(anchor.descriptor),
                ),
                label="persistent root-swap fixture",
                execution_anchor=anchor,
            )
    finally:
        anchor.close()
    assert (moved / "child-output").read_text(encoding="utf-8") == "held-inode"
    assert list(outside.iterdir()) == []


def test_generated_script_descriptor_shebang_is_rewritten_and_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    (root / "bin").mkdir(parents=True)
    anchor = subject._ExecutionAnchor(root, "fixture environment")
    descriptor_root = anchor.child_path(root, "fixture descriptor root")
    script = root / "bin/tool"
    script.write_text(
        "#!" + str(descriptor_root / "bin/python") + "\nprint('ok')\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    try:
        inventory = subject._normalize_and_audit_environment_scripts(
            root, anchor, rewrite_descriptor_paths=True
        )
        confirmed = subject._normalize_and_audit_environment_scripts(
            root, anchor, rewrite_descriptor_paths=False
        )
    finally:
        anchor.close()
    assert inventory == confirmed
    assert script.read_text(encoding="utf-8").startswith(
        "#!" + str(root / "bin/python") + "\n"
    )
    assert inventory["ephemeral_descriptor_paths_present"] is False


def test_generated_pyvenv_descriptor_path_is_rewritten_and_rejected_if_left(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    root.mkdir()
    anchor = subject._ExecutionAnchor(root, "fixture environment")
    descriptor_root = anchor.child_path(root, "fixture descriptor root")
    configuration = root / "pyvenv.cfg"
    configuration.write_text(
        "home = /fixture/base/bin\n"
        "include-system-site-packages = true\n"
        "command = /fixture/base/bin/python -m venv " + str(descriptor_root) + "\n",
        encoding="utf-8",
    )
    try:
        subject._normalize_target_pyvenv_configuration(root, anchor)
    finally:
        anchor.close()
    parsed = subject._parse_pyvenv_configuration(
        root, required=True, require_system_site_packages=True
    )
    assert parsed is not None
    assert parsed["keys"]["command"].endswith(str(root))

    configuration.write_text(
        "include-system-site-packages = true\ncommand = /proc/self/fd/9\n",
        encoding="utf-8",
    )
    with pytest.raises(subject.EnvironmentBuildError, match="ephemeral descriptor"):
        subject._parse_pyvenv_configuration(
            root, required=True, require_system_site_packages=True
        )


def test_controller_requires_isolated_no_site_no_bytecode(monkeypatch) -> None:
    flags = type("Flags", (), {"isolated": 1, "no_site": 0, "dont_write_bytecode": 1})()
    monkeypatch.setattr(subject.sys, "flags", flags)
    with pytest.raises(subject.EnvironmentBuildError, match="-I -S -B"):
        subject._require_safe_controller_runtime()


def test_atomic_publication_is_read_only_and_no_clobber(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    subject._write_json_new_immutable(target, {"status": "complete"}, "receipt")
    original = target.read_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    with pytest.raises(subject.EnvironmentBuildError, match="overwrite"):
        subject._write_json_new_immutable(target, {"status": "changed"}, "receipt")
    assert target.read_bytes() == original


def test_atomic_publication_rejects_symlink_component(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(subject.EnvironmentBuildError, match="symlink"):
        subject._write_json_new_immutable(
            link / "receipt.json", {"status": "complete"}, "receipt"
        )


def test_atomic_publication_detects_parent_rename_and_symlink_race(
    monkeypatch, tmp_path: Path
) -> None:
    parent = tmp_path / "publish"
    moved = tmp_path / "moved"
    attacker = tmp_path / "attacker"
    parent.mkdir()
    attacker.mkdir()
    original = subject._rename_noreplace

    def race(directory_descriptor, source, destination, *, label):
        parent.rename(moved)
        parent.symlink_to(attacker, target_is_directory=True)
        original(directory_descriptor, source, destination, label=label)

    monkeypatch.setattr(subject, "_rename_noreplace", race)
    with pytest.raises(subject.EnvironmentBuildError, match="symlink|identity"):
        subject._write_bytes_new_immutable(parent / "receipt.json", b"ok", "receipt")
    assert not (attacker / "receipt.json").exists()
    assert not (moved / "receipt.json").exists()


def test_atomic_publication_destination_race_never_clobbers(
    monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "receipt.json"
    original = subject._rename_noreplace

    def race(directory_descriptor, source, destination, *, label):
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
            dir_fd=directory_descriptor,
        )
        os.write(descriptor, b"attacker")
        os.close(descriptor)
        original(directory_descriptor, source, destination, label=label)

    monkeypatch.setattr(subject, "_rename_noreplace", race)
    with pytest.raises(subject.EnvironmentBuildError, match="appeared"):
        subject._write_bytes_new_immutable(target, b"receipt", "receipt")
    assert target.read_bytes() == b"attacker"


def test_fd_hash_detects_path_replacement(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "payload"
    target.write_bytes(b"original")
    original = subject._revalidate_fd_path
    replaced = {"done": False}

    def race(path, descriptor, *, directory, label):
        if path == target and not replaced["done"]:
            replaced["done"] = True
            path.unlink()
            path.write_bytes(b"replacement")
        original(path, descriptor, directory=directory, label=label)

    monkeypatch.setattr(subject, "_revalidate_fd_path", race)
    with pytest.raises(subject.EnvironmentBuildError, match="identity"):
        subject._sha256_file(target)


def test_controlled_python_leaf_symlink_is_content_bound(tmp_path: Path) -> None:
    binary = tmp_path / "miniconda" / "bin/python3.12"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"python-binary")
    binary.chmod(0o755)
    launcher = tmp_path / "conda-env" / "bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(binary)
    observed_path, identity = subject._require_python_launcher(launcher, "base Python")
    assert observed_path == launcher
    assert identity["launcher_kind"] == "leaf_symlink_chain"
    assert identity["symlink_chain"][0]["link_target"] == str(binary)
    assert identity["resolved_executable"] == str(binary)
    assert identity["resolved_executable_sha256"] == _sha(binary)
    assert isinstance(identity["resolved_executable_inode"], int)


def test_python_launcher_rejects_parent_symlink(tmp_path: Path) -> None:
    real_environment = tmp_path / "real-env"
    (real_environment / "bin").mkdir(parents=True)
    (real_environment / "bin/python").write_bytes(b"python")
    alias = tmp_path / "alias-env"
    alias.symlink_to(real_environment, target_is_directory=True)
    with pytest.raises(subject.EnvironmentBuildError, match="parent symlink"):
        subject._require_python_launcher(alias / "bin/python", "base Python")


def test_no_site_probe_allows_resolved_base_prefix_and_audits_both_sites(
    tmp_path: Path,
) -> None:
    resolved = tmp_path / "miniconda"
    resolved_python = resolved / "bin/python3.12"
    resolved_python.parent.mkdir(parents=True)
    resolved_python.write_bytes(b"python")
    resolved_python.chmod(0o755)
    resolved_stdlib = resolved / "lib/python3.12"
    (resolved_stdlib / "site-packages").mkdir(parents=True)

    formal = tmp_path / "formal-env"
    launcher = formal / "bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(resolved_python)
    formal_stdlib = formal / "lib/python3.12"
    (formal_stdlib / "site-packages").mkdir(parents=True)
    (formal / "pyvenv.cfg").write_text(
        "home = " + str(resolved / "bin") + "\n"
        "include-system-site-packages = true\n"
        "version = 3.12.4\n",
        encoding="utf-8",
    )
    safe = {
        "executable": str(resolved_python),
        "prefix": str(resolved),
        "base_prefix": str(resolved),
        "python_version": "3.12.4",
        "isolated": True,
        "no_site": True,
        "dont_write_bytecode": True,
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "sys_path": [str(resolved_stdlib)],
    }
    subject._validate_safe_runtime_probe(safe, launcher, "fixture base Python")
    audit = subject._audit_base_site_surface(formal, safe)
    assert audit["distinct_prefixes"] is True
    assert len(audit["prefix_audits"]) == 2
    assert set(audit["site_directories"]) == {
        str(formal_stdlib / "site-packages"),
        str(resolved_stdlib / "site-packages"),
    }
    resolved_audit = subject._resolved_base_audit(audit)
    assert resolved_audit["prefix"] == str(resolved)
    assert resolved_audit["site_directories"] == [
        str(resolved_stdlib / "site-packages")
    ]


def test_base_dual_site_requires_reviewed_system_site_pyvenv(tmp_path: Path) -> None:
    formal, _formal_stdlib, _formal_site, safe = _site_fixture(tmp_path)
    resolved = tmp_path / "resolved"
    resolved_stdlib = resolved / "lib/python3.12"
    (resolved_stdlib / "site-packages").mkdir(parents=True)
    safe = dict(safe)
    safe["prefix"] = str(resolved)
    safe["base_prefix"] = str(resolved)
    safe["sys_path"] = [str(resolved_stdlib)]
    with pytest.raises(subject.EnvironmentBuildError, match="pyvenv.cfg is missing"):
        subject._audit_base_site_surface(formal, safe)
    (formal / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n", encoding="utf-8"
    )
    with pytest.raises(subject.EnvironmentBuildError, match="system site packages"):
        subject._audit_base_site_surface(formal, safe)


@pytest.mark.skipif(
    sys.version_info[:2] > (3, 13),
    reason="Python 3.14 changed -S pyvenv prefix initialization",
)
def test_real_stdlib_pyvenv_no_site_prefix_falls_back_to_base(tmp_path: Path) -> None:
    environment = tmp_path / "real-pyvenv-fixture"
    subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-m",
            "venv",
            "--without-pip",
            "--copies",
            str(environment),
        ),
        check=True,
        env=subject._command_environment(),
    )
    python = environment / "bin/python"
    code = (
        "import json,os,sys;print(json.dumps({"
        "'prefix':os.path.realpath(sys.prefix),"
        "'base_prefix':os.path.realpath(sys.base_prefix)}))"
    )
    normal = json.loads(
        subprocess.run(
            (str(python), "-I", "-B", "-c", code),
            check=True,
            capture_output=True,
            text=True,
            env=subject._command_environment(),
        ).stdout
    )
    no_site = json.loads(
        subprocess.run(
            (str(python), "-I", "-S", "-B", "-c", code),
            check=True,
            capture_output=True,
            text=True,
            env=subject._command_environment(),
        ).stdout
    )
    assert normal["prefix"] == str(environment.resolve())
    assert no_site["prefix"] == no_site["base_prefix"]
    assert no_site["prefix"] != str(environment.resolve())
    configuration = subject._parse_pyvenv_configuration(
        environment, required=True, require_system_site_packages=False
    )
    assert configuration is not None
    assert configuration["keys"]["include-system-site-packages"] == "false"


@pytest.mark.skipif(
    sys.version_info[:2] > (3, 13),
    reason="Python 3.14 changed -S pyvenv prefix initialization",
)
def test_real_nested_pyvenv_target_surface_excludes_source_formal_site(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-formal"
    target = tmp_path / "target-benchmark"
    environment = subject._command_environment()
    subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-m",
            "venv",
            "--without-pip",
            "--copies",
            "--system-site-packages",
            str(source),
        ),
        check=True,
        env=environment,
    )
    subprocess.run(
        (
            str(source / "bin/python"),
            "-I",
            "-S",
            "-B",
            "-m",
            "venv",
            "--without-pip",
            "--copies",
            "--system-site-packages",
            str(target),
        ),
        check=True,
        env=environment,
    )
    probe_code = r"""import fixture_explicit_site, json, os, pip, site, sys
bindings = globals()["__rachel_explicit_site_bindings__"]
def logical_path(value):
    absolute = os.path.abspath(value)
    for binding in bindings:
        descriptor = binding["descriptor_path"]
        if absolute == descriptor or absolute.startswith(descriptor + os.path.sep):
            return binding["canonical_path"] + absolute[len(descriptor):]
    return os.path.realpath(absolute)
print(json.dumps({
    "prefix": os.path.realpath(sys.prefix),
    "base_prefix": os.path.realpath(sys.base_prefix),
    "fixture_value": fixture_explicit_site.VALUE,
    "pip_origin": logical_path(pip.__file__),
    "user_site_enabled": bool(site.ENABLE_USER_SITE),
    "sitecustomize_loaded": "sitecustomize" in sys.modules,
    "usercustomize_loaded": "usercustomize" in sys.modules,
    "sys_path": [logical_path(value) for value in sys.path if value],
}, sort_keys=True))
"""
    if not sys.platform.startswith("linux"):
        # macOS /dev/fd directory descriptors are identity-checkable but are
        # not traversable by importlib.  Exercise the same inherited FDs with
        # openat; the mocked Linux test below covers procfs magic-link import
        # traversal itself.
        probe_code = r"""import json, os, site, sys
bindings = globals()["__rachel_explicit_site_bindings__"]
def descriptor_number(binding):
    return int(binding["descriptor_path"].rsplit("/", 1)[1])
def read_relative(binding, relative):
    descriptor = os.open(relative, os.O_RDONLY, dir_fd=descriptor_number(binding))
    try:
        return os.read(descriptor, 4096).decode("utf-8")
    finally:
        os.close(descriptor)
def logical_path(value):
    absolute = os.path.abspath(value)
    for binding in bindings:
        descriptor = binding["descriptor_path"]
        if absolute == descriptor or absolute.startswith(descriptor + os.path.sep):
            return binding["canonical_path"] + absolute[len(descriptor):]
    return os.path.realpath(absolute)
fixture_source = read_relative(bindings[0], "fixture_explicit_site.py")
read_relative(bindings[1], "pip/__init__.py")
print(json.dumps({
    "prefix": os.path.realpath(sys.prefix),
    "base_prefix": os.path.realpath(sys.base_prefix),
    "fixture_value": 73 if "VALUE = 73" in fixture_source else None,
    "pip_origin": bindings[1]["canonical_path"] + "/pip/__init__.py",
    "user_site_enabled": bool(site.ENABLE_USER_SITE),
    "sitecustomize_loaded": "sitecustomize" in sys.modules,
    "usercustomize_loaded": "usercustomize" in sys.modules,
    "sys_path": [logical_path(value) for value in sys.path if value],
}, sort_keys=True))
"""
    safe_code = r"""import json, os, sys
print(json.dumps({
    "executable": os.path.realpath(sys.executable),
    "prefix": os.path.realpath(sys.prefix),
    "base_prefix": os.path.realpath(sys.base_prefix),
    "sys_path": [os.path.realpath(value) for value in sys.path if value],
}, sort_keys=True))
"""
    target_python = target / "bin/python"
    safe = json.loads(
        subprocess.run(
            (str(target_python), "-I", "-S", "-B", "-c", safe_code),
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout
    )
    version = "python" + str(sys.version_info[0]) + "." + str(sys.version_info[1])
    source_site = str((source / "lib" / version / "site-packages").resolve())
    target_site = str((target / "lib" / version / "site-packages").resolve())
    resolved_site = str(
        (Path(safe["prefix"]) / "lib" / version / "site-packages").resolve()
    )
    assert safe["prefix"] == safe["base_prefix"]
    assert Path(resolved_site).is_dir()
    marker = tmp_path / "pth-execution-marker"
    for name in (
        "distutils-precedence.pth",
        "ruamel.yaml-0.17.21-py3.12-nspkg.pth",
    ):
        (Path(target_site) / name).write_text(
            "import pathlib; pathlib.Path({!r}).write_text({!r})\n".format(
                str(marker), name
            ),
            encoding="utf-8",
        )
    (Path(target_site) / "fixture_explicit_site.py").write_text(
        "VALUE = 73\n", encoding="utf-8"
    )
    audit_safe = {
        "python_version": "{}.{}.{}".format(*sys.version_info[:3]),
        "sys_path": list(safe["sys_path"]),
    }
    audited_target = subject._audit_active_site(
        target,
        safe_probe=audit_safe,
        require_pyvenv=True,
        require_system_site_packages=True,
    )
    assert {
        Path(item["path"]).name for item in audited_target["startup_artifacts"]
    } == {
        "distutils-precedence.pth",
        "ruamel.yaml-0.17.21-py3.12-nspkg.pth",
    }
    assert all(
        item["execution_policy"] == "present_but_never_executed"
        and subject._is_sha256(item["sha256"])
        for item in audited_target["startup_artifacts"]
    )
    target_audit = subject._content_bound(
        {
            "prefix": str(target.resolve()),
            "site_directories": [target_site],
            "startup_search_directories": [
                {
                    "path": target_site,
                    "device": Path(target_site).stat().st_dev,
                    "inode": Path(target_site).stat().st_ino,
                }
            ],
            "activated_paths": [],
            "runtime_no_site": True,
            "pth_execution_policy": "present_but_never_executed",
        }
    )
    resolved_audit = subject._content_bound(
        {
            "prefix": safe["prefix"],
            "site_directories": [resolved_site],
            "startup_search_directories": [
                {
                    "path": resolved_site,
                    "device": Path(resolved_site).stat().st_dev,
                    "inode": Path(resolved_site).stat().st_ino,
                }
            ],
            "activated_paths": [],
            "runtime_no_site": True,
            "pth_execution_policy": "present_but_never_executed",
        }
    )
    site_descriptors = []
    sites = []
    try:
        for index, (path, audit) in enumerate(
            ((target_site, target_audit), (resolved_site, resolved_audit))
        ):
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            site_descriptors.append(descriptor)
            metadata = os.fstat(descriptor)
            descriptor_path = (
                "/proc/{}/fd/{}".format(os.getpid(), descriptor)
                if sys.platform.startswith("linux")
                else "/dev/fd/{}".format(descriptor)
            )
            sites.append(
                {
                    "canonical_path": path,
                    "descriptor_path": descriptor_path,
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "role": "audit_" + str(index),
                    "audit_content_sha256": audit["content_sha256"],
                }
            )
        configuration = subject._content_bound(
            {
                "schema_version": subject.EXPLICIT_SITE_BOOTSTRAP_SCHEMA,
                "startup": {
                    "executable": safe["executable"],
                    "prefix": safe["prefix"],
                    "base_prefix": safe["base_prefix"],
                },
                "runtime": {
                    "prefix": str(target.resolve()),
                    "base_prefix": safe["base_prefix"],
                },
                "no_site_sys_path": list(safe["sys_path"]),
                "sites": sites,
                "torchvision_dynamic_admission": (
                    subject._torchvision_dynamic_admission_configuration(False)
                ),
            }
        )
        completed = subprocess.run(
            (
                str(target_python),
                "-I",
                "-S",
                "-B",
                "-c",
                subject.EXPLICIT_SITE_STDIN_LAUNCHER_CODE,
                str(len(safe["sys_path"])),
                *safe["sys_path"],
                subject._sha256_bytes(subject._canonical_bytes(list(safe["sys_path"]))),
                hashlib.sha256(
                    subject.EXPLICIT_SITE_BOOTSTRAP_CODE.encode("utf-8")
                ).hexdigest(),
                json.dumps(configuration, sort_keys=True, separators=(",", ":")),
                probe_code,
            ),
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            pass_fds=tuple(site_descriptors),
            input=subject.EXPLICIT_SITE_BOOTSTRAP_CODE,
        )
    finally:
        for descriptor in site_descriptors:
            os.close(descriptor)
    detailed = json.loads(completed.stdout)
    assert detailed["prefix"] == str(target.resolve())
    assert detailed["base_prefix"] == safe["base_prefix"]
    assert detailed["fixture_value"] == 73
    assert os.path.commonpath((detailed["pip_origin"], resolved_site)) == resolved_site
    assert os.path.commonpath((detailed["pip_origin"], target_site)) != target_site
    assert detailed["user_site_enabled"] is False
    assert detailed["sys_path"] == list(safe["sys_path"]) + [
        target_site,
        resolved_site,
    ]
    assert source_site not in detailed["sys_path"]
    assert not marker.exists()
    subject._validate_probe_startup_surface(
        detailed,
        (safe,),
        (target_audit, resolved_audit),
        "real nested target",
    )
    formal_audit = subject._content_bound(
        {
            "site_directories": [source_site],
            "activated_paths": [],
            "runtime_no_site": True,
            "pth_execution_policy": "present_but_never_executed",
        }
    )
    with pytest.raises(subject.EnvironmentBuildError, match="sys.path"):
        subject._validate_probe_startup_surface(
            detailed,
            (safe,),
            (target_audit, resolved_audit, formal_audit),
            "real nested target with source formal site",
        )


def test_reviewed_lock_is_exact_read_only_dual_pypi_authority(tmp_path: Path) -> None:
    lock = tmp_path / "reviewed.json"
    lock.write_bytes(REVIEWED_LOCK.read_bytes())
    lock.chmod(0o444)
    binding = subject._validate_wheel_lock(
        lock,
        subject.REVIEWED_WHEEL_LOCK_FILE_SHA256,
        {
            "python_version": "3.12.4",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
        },
    )
    assert binding["file_sha256"] == subject.REVIEWED_WHEEL_LOCK_FILE_SHA256
    assert len(binding["packages"]) == 3
    assert set(binding["official_index"]) == {
        "name",
        "download_host",
        "project_json_urls",
        "simple_json_urls",
    }
    assert binding["review"]["independent_authority_cross_check"]["result"] == (
        "all_three_filename_size_sha256_equal"
    )


def test_reviewed_lock_rejects_any_file_hash_or_mode_drift(tmp_path: Path) -> None:
    lock = tmp_path / "reviewed.json"
    lock.write_bytes(REVIEWED_LOCK.read_bytes() + b" ")
    lock.chmod(0o444)
    safe = {
        "python_version": "3.12.4",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
    }
    with pytest.raises(subject.EnvironmentBuildError, match="file SHA-256"):
        subject._validate_wheel_lock(
            lock, subject.REVIEWED_WHEEL_LOCK_FILE_SHA256, safe
        )
    lock.chmod(0o644)
    with pytest.raises(subject.EnvironmentBuildError, match="read-only"):
        subject._validate_wheel_lock(
            lock, subject.REVIEWED_WHEEL_LOCK_FILE_SHA256, safe
        )


def test_wheel_inspection_binds_filename_metadata_wheel_and_record(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "torch_geometric-2.6.1-py3-none-any.whl"
    _make_wheel(wheel, "torch-geometric", "2.6.1")
    authority = _wheel_authority(wheel, "torch-geometric", "2.6.1")
    observed = subject._inspect_wheel(wheel, authority)
    assert observed["distribution"] == "torch-geometric"
    assert observed["version"] == "2.6.1"
    assert observed["sha256"] == _sha(wheel)
    assert observed["wheel_tags"] == ["py3-none-any"]
    assert observed["record_rows"] == observed["record_verified_sha256_rows"] + 1
    assert all(
        subject._is_sha256(observed[field])
        for field in ("metadata_sha256", "wheel_metadata_sha256", "record_sha256")
    )


def test_wheel_inspection_rejects_record_or_tag_drift(tmp_path: Path) -> None:
    bad_record = tmp_path / "scipy-1.14.1-py3-none-any.whl"
    _make_wheel(bad_record, "scipy", "1.14.1", bad_record=True)
    with pytest.raises(subject.EnvironmentBuildError, match="RECORD"):
        subject._inspect_wheel(bad_record)

    bad_tag = tmp_path / "opencv_python-4.10.0.84-py3-none-any.whl"
    _make_wheel(
        bad_tag,
        "opencv-python",
        "4.10.0.84",
        tags=("cp312-cp312-manylinux_2_17_x86_64",),
    )
    authority = _wheel_authority(bad_tag, "opencv-python", "4.10.0.84")
    with pytest.raises(subject.EnvironmentBuildError, match="tags"):
        subject._inspect_wheel(bad_tag, authority)


def test_pip_report_binds_all_three_exact_file_urls_and_hashes(tmp_path: Path) -> None:
    root = tmp_path / "environment"
    wheelhouse = root / subject.WHEELHOUSE_RELATIVE_PATH
    wheelhouse.mkdir(parents=True)
    packages = []
    report_install = []
    for filename_name, metadata_name, version in (
        ("torch_geometric", "torch-geometric", "2.6.1"),
        ("opencv_python", "opencv-python", "4.10.0.84"),
        ("scipy", "scipy", "1.14.1"),
    ):
        wheel = wheelhouse / (filename_name + "-" + version + "-py3-none-any.whl")
        _make_wheel(wheel, metadata_name, version)
        authority = _wheel_authority(wheel, metadata_name, version)
        packages.append(authority)
        report_install.append(
            {
                "metadata": {"name": metadata_name, "version": version},
                "download_info": {
                    "url": wheel.as_uri(),
                    "archive_info": {"hashes": {"sha256": _sha(wheel)}},
                },
            }
        )
    reviewed = {"packages": packages}
    report = root / subject.INSTALL_REPORT_RELATIVE_PATH
    report.write_text(json.dumps({"install": report_install}), encoding="utf-8")
    observed = subject._validate_install_report(report, root, reviewed)
    assert {(item["distribution"], item["version"]) for item in observed} == {
        ("torch-geometric", "2.6.1"),
        ("opencv-python", "4.10.0.84"),
        ("scipy", "1.14.1"),
    }
    report_install[0]["download_info"]["url"] = (
        "https://files.pythonhosted.org/unreviewed.whl"
    )
    report.write_text(json.dumps({"install": report_install}), encoding="utf-8")
    with pytest.raises(subject.EnvironmentBuildError, match="wheel path"):
        subject._validate_install_report(report, root, reviewed)


def test_descriptor_guard_accepts_only_mocked_parent_procfs_magic_fd(
    tmp_path: Path,
) -> None:
    site = tmp_path / "site-packages"
    site.mkdir()
    descriptor = os.open(site, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    metadata = os.fstat(descriptor)
    owner_pid = 424242
    descriptor_path = "/proc/{}/fd/{}".format(owner_pid, descriptor)

    class FakeLinuxOS:
        path = os.path

        def __init__(self):
            self.follow_calls = []

        def __getattr__(self, name):
            return getattr(os, name)

        def getppid(self):
            return owner_pid

        def stat(self, path, *, follow_symlinks=True):
            if path == descriptor_path:
                self.follow_calls.append(follow_symlinks)
                if follow_symlinks:
                    return metadata
                return SimpleNamespace(st_mode=stat.S_IFLNK | 0o700)
            return os.stat(path, follow_symlinks=follow_symlinks)

    fake_os = FakeLinuxOS()
    guard = _descriptor_guard("linux", fake_os)
    binding = {
        "canonical_path": str(site),
        "descriptor_path": descriptor_path,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }
    try:
        assert guard(binding) == descriptor_path
        assert fake_os.follow_calls == [False, True]

        wrong = dict(binding, descriptor_path="/proc/424243/fd/{}".format(descriptor))
        with pytest.raises(RuntimeError, match="owner PID"):
            guard(wrong)

        wrong = dict(binding, descriptor_path=str(site))
        with pytest.raises(RuntimeError, match="procfs FD path"):
            guard(wrong)

        wrong = dict(
            binding,
            descriptor_path="/proc/{}/fd/{}".format(owner_pid, descriptor + 100000),
        )
        with pytest.raises(RuntimeError, match="not inherited"):
            guard(wrong)

        with pytest.raises(RuntimeError, match="inherited directory identity"):
            guard(dict(binding, device=metadata.st_dev + 1))
        with pytest.raises(RuntimeError, match="inherited directory identity"):
            guard(dict(binding, inode=metadata.st_ino + 1))
    finally:
        os.close(descriptor)


def test_descriptor_guard_rejects_canonical_root_swap_without_outside_access(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    moved = tmp_path / "environment-moved"
    outside = tmp_path / "outside"
    site = root / "lib/python3.12/site-packages"
    outside_site = outside / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    outside_site.mkdir(parents=True)
    descriptor = os.open(site, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    metadata = os.fstat(descriptor)
    descriptor_path = "/dev/fd/{}".format(descriptor)
    guard = _descriptor_guard("darwin")
    binding = {
        "canonical_path": str(site),
        "descriptor_path": descriptor_path,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }
    root.rename(moved)
    root.symlink_to(outside, target_is_directory=True)
    try:
        with pytest.raises(RuntimeError, match="without symlinks"):
            guard(binding)
        assert list(outside_site.iterdir()) == []
    finally:
        os.close(descriptor)


def test_stdin_launcher_keeps_exact_path_and_blocks_temp_only_module(
    tmp_path: Path,
) -> None:
    reviewed_site = tmp_path / "reviewed-site"
    unreviewed_temp = tmp_path / "temporary-script-directory"
    reviewed_site.mkdir()
    unreviewed_temp.mkdir()
    marker = tmp_path / "malicious-import-marker"
    launcher_marker = tmp_path / "launcher-import-marker"
    (unreviewed_temp / "hashlib.py").write_text(
        "open({!r}, 'w').write('executed')\n".format(str(launcher_marker)),
        encoding="utf-8",
    )
    (unreviewed_temp / "malicious_temp_only.py").write_text(
        "from pathlib import Path\nPath({!r}).write_text('executed')\n".format(
            str(marker)
        ),
        encoding="utf-8",
    )
    safe = subject._safe_runtime_probe(Path(sys.executable))
    descriptor = os.open(reviewed_site, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    metadata = os.fstat(descriptor)
    descriptor_path = (
        "/proc/{}/fd/{}".format(os.getpid(), descriptor)
        if sys.platform.startswith("linux")
        else "/dev/fd/{}".format(descriptor)
    )
    configuration = subject._content_bound(
        {
            "schema_version": subject.EXPLICIT_SITE_BOOTSTRAP_SCHEMA,
            "startup": {
                "executable": safe["executable"],
                "prefix": safe["prefix"],
                "base_prefix": safe["base_prefix"],
            },
            "runtime": {
                "prefix": safe["prefix"],
                "base_prefix": safe["base_prefix"],
            },
            "no_site_sys_path": list(safe["sys_path"]),
            "sites": [
                {
                    "canonical_path": str(reviewed_site),
                    "descriptor_path": descriptor_path,
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "role": "audit_0",
                    "audit_content_sha256": "a" * 64,
                }
            ],
            "torchvision_dynamic_admission": (
                subject._torchvision_dynamic_admission_configuration(False)
            ),
        }
    )
    bootstrap_sha = hashlib.sha256(
        subject.EXPLICIT_SITE_BOOTSTRAP_CODE.encode("utf-8")
    ).hexdigest()

    def command(payload: str):
        return (
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            subject.EXPLICIT_SITE_STDIN_LAUNCHER_CODE,
            str(len(safe["sys_path"])),
            *safe["sys_path"],
            subject._sha256_bytes(subject._canonical_bytes(list(safe["sys_path"]))),
            bootstrap_sha,
            json.dumps(configuration, sort_keys=True, separators=(",", ":")),
            payload,
        )

    try:
        benign_command = list(command("import json,sys;print(json.dumps(sys.path))"))
        benign_command[5] = (
            "import sys;sys.path.insert(0,sys.argv.pop(1));"
            "exec(compile({!r},'<reviewed-stdin-launcher>','exec'),globals(),globals())"
        ).format(subject.EXPLICIT_SITE_STDIN_LAUNCHER_CODE)
        benign_command.insert(6, str(unreviewed_temp))
        benign = subprocess.run(
            benign_command,
            check=True,
            capture_output=True,
            text=True,
            input=subject.EXPLICIT_SITE_BOOTSTRAP_CODE,
            pass_fds=(descriptor,),
            env=subject._command_environment(),
        )
        assert json.loads(benign.stdout) == list(safe["sys_path"]) + [descriptor_path]
        assert not launcher_marker.exists()
        malicious = subprocess.run(
            command(
                "import sys;sys.path.insert(0,{!r});import malicious_temp_only".format(
                    str(unreviewed_temp)
                )
            ),
            check=False,
            capture_output=True,
            text=True,
            input=subject.EXPLICIT_SITE_BOOTSTRAP_CODE,
            pass_fds=(descriptor,),
            env=subject._command_environment(),
        )
    finally:
        os.close(descriptor)
    assert malicious.returncode != 0
    assert "unreviewed sys.path mutation" in malicious.stderr
    assert not marker.exists()


def _dynamic_admission_fixture_sources(attack: str = "none"):
    generated = "_generated_methods = [object(), object()]\n"
    template = (
        "def get_remote_module_template(enable_moving_cpu_tensors_to_cuda):\n"
        "    return {!r}\n".format(generated)
    )
    before_append = [
        "import atexit, importlib, importlib.util, os, sys, tempfile",
        "_TEMP_DIR = tempfile.TemporaryDirectory()",
        "INSTANTIATED_TEMPLATE_DIR_PATH = _TEMP_DIR.name",
        "GENERATED = {!r}".format(
            generated + ("# generated drift\n" if attack == "generated_bytes" else "")
        ),
        "from torch.distributed.nn.jit.templates import remote_module_template",
    ]
    if attack == "directory_symlink":
        before_append.extend(
            [
                "os.rmdir(INSTANTIATED_TEMPLATE_DIR_PATH)",
                "_OUTSIDE_DIR = INSTANTIATED_TEMPLATE_DIR_PATH + '_outside'",
                "os.mkdir(_OUTSIDE_DIR, 0o700)",
                "os.symlink(_OUTSIDE_DIR, INSTANTIATED_TEMPLATE_DIR_PATH)",
                "atexit.register(lambda: (os.unlink(INSTANTIATED_TEMPLATE_DIR_PATH) if os.path.islink(INSTANTIATED_TEMPLATE_DIR_PATH) else None, os.rmdir(_OUTSIDE_DIR) if os.path.isdir(_OUTSIDE_DIR) else None))",
            ]
        )
    if attack == "wrong_caller":
        while len(before_append) < 20:
            before_append.append("# pinned fixture padding")
        before_append.extend(
            [
                "def _wrong_caller():",
                "    # pinned fixture caller padding",
                "    sys.path.append(INSTANTIATED_TEMPLATE_DIR_PATH)",
                "_wrong_caller()",
            ]
        )
    else:
        while len(before_append) < 22:
            before_append.append("# pinned fixture padding")
        before_append.append(
            "sys.path.append(INSTANTIATED_TEMPLATE_DIR_PATH)"
        )
        if attack == "second_append":
            before_append.append(
                "sys.path.append(INSTANTIATED_TEMPLATE_DIR_PATH)"
            )
    body = [
        "def instantiate_non_scriptable_remote_module_template():",
        "    output = os.path.join(INSTANTIATED_TEMPLATE_DIR_PATH, '_remote_module_non_scriptable.py')",
    ]
    if attack == "file_symlink":
        body.append("    os.symlink(__file__, output)")
    elif attack == "file_hardlink":
        body.append("    os.link(__file__, output)")
    else:
        body.extend(
            [
                "    with open(output, 'w', encoding='utf-8') as stream:",
                "        stream.write(GENERATED)",
            ]
        )
        if attack == "extra_file":
            body.extend(
                [
                    "    with open(os.path.join(INSTANTIATED_TEMPLATE_DIR_PATH, 'extra.py'), 'w', encoding='utf-8') as stream:",
                    "        stream.write('extra')",
                ]
            )
        elif attack == "file_mode":
            body.append("    os.chmod(output, 0o600)")
        elif attack == "captured_compile":
            body.extend(
                [
                    "    sys.modules.pop('_remote_module_non_scriptable', None)",
                    "    specification = importlib.util.spec_from_file_location('_remote_module_non_scriptable', output)",
                    "    module = importlib.util.module_from_spec(specification)",
                    "    specification.loader.exec_module(module)",
                ]
            )
    body.extend(
        [
            "    importlib.invalidate_caches()",
            "    return importlib.import_module('_remote_module_non_scriptable')",
        ]
    )
    instantiator = "\n".join(before_append + body) + "\n"
    if attack != "wrong_caller":
        assert instantiator.splitlines()[22].strip() == (
            "sys.path.append(INSTANTIATED_TEMPLATE_DIR_PATH)"
        )
    remote = (
        "from torch.distributed.nn.jit import instantiator\n"
        "_NON_SCRIPTABLE_REMOTE_MODULE_MODULE = "
        "instantiator.instantiate_non_scriptable_remote_module_template()\n"
    )
    return {
        "torch/distributed/nn/jit/instantiator.py": instantiator.encode("utf-8"),
        "torch/distributed/nn/jit/templates/remote_module_template.py": (
            template.encode("utf-8")
        ),
        "torch/distributed/nn/api/remote_module.py": remote.encode("utf-8"),
        "torch/__init__.py": b"__version__ = 'fixture-torch'\n",
        "torchvision/__init__.py": (
            b"__version__ = 'fixture-vision'\n"
            b"from torch.distributed.nn.api import remote_module\n"
        ),
    }, generated.encode("utf-8")


def _run_dynamic_admission_fixture(
    tmp_path: Path,
    *,
    attack: str = "none",
    payload: str = (
        "import json,sys;print(json.dumps({"
        "'admission':__rachel_torchvision_dynamic_admission__,"
        "'sys_path':list(sys.path)}))"
    ),
    source_drift: bool = False,
    malicious_pyc_marker: Optional[Path] = None,
    malicious_pyc_relative_path: str = (
        "torch/distributed/nn/jit/instantiator.py"
    ),
) -> subprocess.CompletedProcess:
    reviewed_site = tmp_path / ("dynamic-site-" + attack)
    reviewed_site.mkdir(parents=True)
    sources, generated = _dynamic_admission_fixture_sources(attack)
    for relative, content in sources.items():
        path = reviewed_site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for package in (
        "torch/distributed",
        "torch/distributed/nn",
        "torch/distributed/nn/jit",
        "torch/distributed/nn/jit/templates",
        "torch/distributed/nn/api",
    ):
        initializer = reviewed_site / package / "__init__.py"
        if not initializer.exists():
            initializer.write_bytes(b"")
    if malicious_pyc_marker is not None:
        import importlib._bootstrap_external as bootstrap_external
        import importlib.util

        source_path = reviewed_site / malicious_pyc_relative_path
        metadata = source_path.stat()
        malicious_code = compile(
            "open({!r}, 'w').write('executed')\n".format(
                str(malicious_pyc_marker)
            ),
            str(source_path),
            "exec",
        )
        cache_path = Path(importlib.util.cache_from_source(str(source_path)))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(
            bootstrap_external._code_to_timestamp_pyc(
                malicious_code,
                int(metadata.st_mtime),
                metadata.st_size,
            )
        )
    authorities = dict(subject.TORCHVISION_DYNAMIC_AUTHORITIES)
    authorities.update(
        {
            "torch_version": "fixture-torch",
            "torchvision_version": "fixture-vision",
            "instantiator_source_sha256": hashlib.sha256(
                sources["torch/distributed/nn/jit/instantiator.py"]
            ).hexdigest(),
            "template_source_sha256": hashlib.sha256(
                sources[
                    "torch/distributed/nn/jit/templates/remote_module_template.py"
                ]
            ).hexdigest(),
            "remote_module_source_sha256": hashlib.sha256(
                sources["torch/distributed/nn/api/remote_module.py"]
            ).hexdigest(),
            "generated_module_size": len(generated),
            "generated_module_sha256": hashlib.sha256(generated).hexdigest(),
        }
    )
    dynamic_source = subject.TORCHVISION_DYNAMIC_ADMISSION_BOOTSTRAP_CODE
    start = dynamic_source.index("_expected_dynamic_authorities = {")
    end = dynamic_source.index("\n_expected_dynamic_policy =", start)
    dynamic_source = (
        dynamic_source[:start]
        + "_expected_dynamic_authorities = "
        + repr(authorities)
        + dynamic_source[end:]
    )
    if not sys.platform.startswith("linux"):
        # Darwin directory FDs are identity-checkable but cannot be traversed
        # as /dev/fd/N/child.  The production Linux branch remains unmodified;
        # this local fixture exercises the admission state machine using the
        # already identity-checked lexical directories.
        dynamic_source = dynamic_source.replace(
            'elif sys.platform == "darwin":\n'
            '                descriptor_path = "/dev/fd/{}".format(descriptor)',
            'elif sys.platform == "darwin":\n'
            "                descriptor_path = value",
        )
    bootstrap = subject.EXPLICIT_SITE_BOOTSTRAP_CODE.replace(
        subject.TORCHVISION_DYNAMIC_ADMISSION_BOOTSTRAP_CODE,
        dynamic_source,
    )
    if not sys.platform.startswith("linux"):
        bootstrap = bootstrap.replace(
            "    sys.path.append(descriptor_path)\n    seen.add(canonical)",
            "    sys.path.append(canonical)\n    seen.add(canonical)",
            1,
        )
    if source_drift:
        source = reviewed_site / authorities["instantiator_relative_path"]
        source.write_bytes(source.read_bytes() + b"# drift\n")
    safe = subject._safe_runtime_probe(Path(sys.executable))
    descriptor = os.open(
        reviewed_site, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    metadata = os.fstat(descriptor)
    descriptor_path = (
        "/proc/{}/fd/{}".format(os.getpid(), descriptor)
        if sys.platform.startswith("linux")
        else "/dev/fd/{}".format(descriptor)
    )
    configuration = subject._content_bound(
        {
            "schema_version": subject.EXPLICIT_SITE_BOOTSTRAP_SCHEMA,
            "startup": {
                "executable": safe["executable"],
                "prefix": safe["prefix"],
                "base_prefix": safe["base_prefix"],
            },
            "runtime": {
                "prefix": safe["prefix"],
                "base_prefix": safe["base_prefix"],
            },
            "no_site_sys_path": list(safe["sys_path"]),
            "sites": [
                {
                    "canonical_path": str(reviewed_site),
                    "descriptor_path": descriptor_path,
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "role": "audit_0",
                    "audit_content_sha256": "a" * 64,
                }
            ],
            "torchvision_dynamic_admission": {
                "schema_version": subject.TORCHVISION_DYNAMIC_ADMISSION_SCHEMA,
                "enabled": True,
                "authorities": authorities,
                "policy": dict(subject.TORCHVISION_DYNAMIC_ADMISSION_POLICY),
            },
        }
    )
    try:
        return subprocess.run(
            (
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                subject.EXPLICIT_SITE_STDIN_LAUNCHER_CODE,
                str(len(safe["sys_path"])),
                *safe["sys_path"],
                subject._sha256_bytes(
                    subject._canonical_bytes(list(safe["sys_path"]))
                ),
                hashlib.sha256(bootstrap.encode("utf-8")).hexdigest(),
                json.dumps(configuration, sort_keys=True, separators=(",", ":")),
                payload,
            ),
            check=False,
            capture_output=True,
            text=True,
            input=bootstrap,
            pass_fds=(descriptor,),
            env=subject._command_environment(),
        )
    finally:
        os.close(descriptor)


def test_dynamic_preseed_admission_has_zero_captured_code_execution_and_restores_path(
    tmp_path: Path,
) -> None:
    completed = _run_dynamic_admission_fixture(tmp_path)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    observed = result["admission"]["observed"]
    assert observed["append_count"] == 1
    assert observed["captured_finder_count"] == 0
    assert observed["captured_compile_count"] == 0
    assert observed["captured_exec_count"] == 0
    assert observed["captured_temp_origin_module_count"] == 0
    assert observed["template_compile_count"] == 1
    assert observed["template_exec_count"] == 1
    assert observed["preseed_compile_count"] == 1
    assert observed["preseed_exec_count"] == 1
    assert set(observed["pinned_source_find_count"].values()) == {1}
    assert set(observed["pinned_source_compile_count"].values()) == {1}
    assert set(observed["pinned_source_exec_count"].values()) == {1}
    assert observed["pinned_source_loader_identity_validated"] is True
    assert observed["pinned_source_pyc_read_count"] == 0
    assert observed["remote_module_identity_is_preseed"] is True
    assert observed["reviewed_sys_path_restored"] is True
    assert not any("tmp" in path for path in result["sys_path"] if path.startswith("/tmp/"))


@pytest.mark.parametrize(
    "relative_path",
    [
        "torch/distributed/nn/jit/instantiator.py",
        "torch/distributed/nn/jit/templates/remote_module_template.py",
        "torch/distributed/nn/api/remote_module.py",
    ],
)
def test_dynamic_pinned_source_loader_ignores_valid_header_malicious_pyc(
    tmp_path: Path, relative_path: str
) -> None:
    marker = tmp_path / "malicious-pyc-marker"
    completed = _run_dynamic_admission_fixture(
        tmp_path,
        malicious_pyc_marker=marker,
        malicious_pyc_relative_path=relative_path,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)["admission"]["observed"]
    assert observed["pinned_source_pyc_read_count"] == 0
    assert observed["pinned_source_loader_identity_validated"] is True
    assert not marker.exists()


@pytest.mark.parametrize(
    "attack, message",
    [
        ("wrong_caller", "unreviewed sys.path append caller"),
        ("second_append", "append source/line differs"),
        ("directory_symlink", "Too many levels|Not a directory|generated directory"),
        ("generated_bytes", "written generated source differs"),
        ("extra_file", "directory file set differs"),
        ("file_symlink", "written generated source|Too many levels"),
        ("file_hardlink", "written generated source differs|No such file"),
        ("file_mode", "written generated source differs"),
        ("captured_compile", "compile from captured dynamic directory denied"),
    ],
)
def test_dynamic_preseed_admission_rejects_caller_tree_and_execution_attacks(
    tmp_path: Path, attack: str, message: str
) -> None:
    completed = _run_dynamic_admission_fixture(tmp_path, attack=attack)
    assert completed.returncode != 0
    assert re.search(message, completed.stderr), completed.stderr


def test_dynamic_preseed_admission_rejects_source_drift_and_post_restore_mutation(
    tmp_path: Path,
) -> None:
    drift = _run_dynamic_admission_fixture(
        tmp_path / "drift", source_drift=True
    )
    assert drift.returncode != 0
    assert "pinned source SHA-256 differs" in drift.stderr
    mutation = _run_dynamic_admission_fixture(
        tmp_path / "mutation",
        payload="import sys;list.append(sys.path, '/unreviewed-after-restore')",
    )
    assert mutation.returncode != 0
    assert "changed the reviewed import path" in mutation.stderr


def test_dynamic_admission_evidence_is_exact_and_rejects_authority_or_count_drift(
) -> None:
    evidence = subject._successful_torchvision_dynamic_admission_evidence()
    subject._validate_torchvision_dynamic_admission_evidence(evidence, "fixture")
    for mutate in ("authority", "count", "restored", "owner_policy"):
        drifted = json.loads(json.dumps(evidence))
        if mutate == "authority":
            drifted["authorities"]["template_source_sha256"] = "0" * 64
        elif mutate == "count":
            drifted["observed"]["captured_exec_count"] = 1
        elif mutate == "restored":
            drifted["observed"]["reviewed_sys_path_restored"] = False
        else:
            drifted["policy"]["captured_file_identity"] = "unreviewed-owner"
        with pytest.raises(subject.EnvironmentBuildError, match="differs"):
            subject._validate_torchvision_dynamic_admission_evidence(
                drifted, "fixture"
            )


def test_install_argv_anchors_all_root_paths_and_normalizes_report_under_swap(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "environment"
    moved = tmp_path / "environment-moved"
    outside = tmp_path / "outside"
    evidence = root / ".environment_build_evidence"
    wheelhouse = root / subject.WHEELHOUSE_RELATIVE_PATH
    evidence.mkdir(parents=True)
    wheelhouse.mkdir()
    python = root / "bin/python"
    python.parent.mkdir()
    python.write_bytes(b"fixture-target-python")
    python.chmod(0o755)
    audit = _fixture_audit(root)
    resolved_base = tmp_path / "resolved-base"
    resolved_base_audit = _fixture_audit(resolved_base)
    safe = _safe_probe(python)
    (root / subject.LOCK_RELATIVE_PATH).write_text("fixture-lock\n", encoding="utf-8")
    outside.mkdir()
    reviewed = _reviewed_lock_binding(REVIEWED_LOCK)
    anchor = subject._ExecutionAnchor(root, "fixture environment")
    observed = {}

    def fake_run(arguments, **kwargs):
        observed["arguments"] = tuple(arguments)
        observed["pass_fds"] = kwargs["pass_fds"]
        observed["stdin"] = kwargs.get("input")
        root.rename(moved)
        root.symlink_to(outside, target_is_directory=True)
        report = {
            "install": [
                {
                    "metadata": {
                        "name": item["distribution"],
                        "version": item["version"],
                    },
                    "download_info": {
                        "url": anchor.child_path(
                            moved.parent
                            / root.name
                            / subject.WHEELHOUSE_RELATIVE_PATH
                            / item["filename"],
                            "fixture anchored wheel",
                        ).as_uri(),
                        "archive_info": {"hashes": {"sha256": item["sha256"]}},
                    },
                }
                for item in reviewed["packages"]
            ]
        }
        descriptor = os.open(
            subject.INSTALL_REPORT_RELATIVE_PATH,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
            dir_fd=anchor.descriptor,
        )
        os.write(descriptor, json.dumps(report).encode("utf-8"))
        os.close(descriptor)
        root.unlink()
        moved.rename(root)
        return subprocess.CompletedProcess(arguments, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    monkeypatch.setattr(subject, "_validate_install_report", lambda *a, **k: [])
    try:
        result = subject._install_locked_wheels(
            python,
            root,
            reviewed,
            anchor,
            safe,
            (audit, resolved_base_audit),
        )
        descriptor_root = anchor.child_path(root, "fixture descriptor root")
    finally:
        anchor.close()
    arguments = observed["arguments"]
    assert arguments[0] == str(descriptor_root / "bin/python")
    assert arguments[1:4] == ("-I", "-S", "-B")
    assert arguments[4] == "-c"
    assert arguments[5] == subject.EXPLICIT_SITE_STDIN_LAUNCHER_CODE
    path_count = int(arguments[6])
    assert arguments[7 : 7 + path_count] == tuple(safe["sys_path"])
    path_sha_index = 7 + path_count
    assert arguments[path_sha_index] == subject._sha256_bytes(
        subject._canonical_bytes(list(safe["sys_path"]))
    )
    bootstrap_index = path_sha_index + 1
    assert (
        arguments[bootstrap_index]
        == hashlib.sha256(
            subject.EXPLICIT_SITE_BOOTSTRAP_CODE.encode("utf-8")
        ).hexdigest()
    )
    configuration = json.loads(arguments[bootstrap_index + 1])
    subject._validate_content_sha(configuration, "fixture explicit-site configuration")
    assert configuration["schema_version"] == subject.EXPLICIT_SITE_BOOTSTRAP_SCHEMA
    assert configuration["torchvision_dynamic_admission"] == (
        subject._torchvision_dynamic_admission_configuration(False)
    )
    assert configuration["sites"][0]["canonical_path"] == str(
        root / "lib/python3.12/site-packages"
    )
    assert configuration["sites"][1]["canonical_path"] == str(
        resolved_base / "lib/python3.12/site-packages"
    )
    assert configuration["sites"][1]["role"] == "audit_1"
    assert (
        configuration["sites"][0]["descriptor_path"]
        != configuration["sites"][0]["canonical_path"]
    )
    assert (
        re.fullmatch(
            r"/(?:proc/[0-9]+/fd|dev/fd)/[0-9]+",
            configuration["sites"][0]["descriptor_path"],
        )
        is not None
    )
    assert arguments[bootstrap_index + 2] == subject.PIP_NO_SITE_RUN_CODE
    assert (
        str(root / "lib/python3.12/site-packages")
        not in arguments[bootstrap_index + 3 :]
    )
    assert observed["stdin"] == subject.EXPLICIT_SITE_BOOTSTRAP_CODE
    assert arguments[arguments.index("--prefix") + 1] == str(descriptor_root)
    assert arguments[arguments.index("--find-links") + 1] == str(
        descriptor_root / subject.WHEELHOUSE_RELATIVE_PATH
    )
    assert arguments[arguments.index("--report") + 1] == str(
        descriptor_root / subject.INSTALL_REPORT_RELATIVE_PATH
    )
    assert arguments[arguments.index("-r") + 1] == str(
        descriptor_root / subject.LOCK_RELATIVE_PATH
    )
    assert observed["pass_fds"]
    assert list(outside.iterdir()) == []
    normalized_report = json.loads(
        (root / subject.INSTALL_REPORT_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    assert {item["download_info"]["url"] for item in normalized_report["install"]} == {
        (root / subject.WHEELHOUSE_RELATIVE_PATH / item["filename"]).as_uri()
        for item in reviewed["packages"]
    }
    subject._validate_install_report_path_normalization(
        result["report_path_normalization"], root, reviewed
    )


def _site_fixture(tmp_path: Path):
    prefix = tmp_path / "runtime"
    stdlib = prefix / "lib/python3.12"
    site_packages = stdlib / "site-packages"
    site_packages.mkdir(parents=True)
    safe = {"python_version": "3.12.4", "sys_path": [str(stdlib)]}
    return prefix, stdlib, site_packages, safe


def test_active_site_directories_deduplicate_controlled_lib64_alias(
    tmp_path: Path,
) -> None:
    prefix, _stdlib, site_packages, safe = _site_fixture(tmp_path)
    (prefix / "lib64").symlink_to("lib", target_is_directory=True)
    assert subject._active_site_directories(prefix, safe["python_version"]) == [
        site_packages
    ]
    audit = subject._audit_active_site(prefix, safe_probe=safe, require_pyvenv=False)
    assert audit["site_directories"] == [str(site_packages)]
    scanned_sites = [
        item
        for item in audit["startup_search_directories"]
        if item["path"] == str(site_packages)
    ]
    assert len(scanned_sites) == 1


def test_active_site_directories_reject_unknown_lib64_symlink(
    tmp_path: Path,
) -> None:
    prefix, _stdlib, _site_packages, safe = _site_fixture(tmp_path)
    outside = tmp_path / "outside-lib"
    outside.mkdir()
    (prefix / "lib64").symlink_to(outside, target_is_directory=True)
    with pytest.raises(subject.EnvironmentBuildError, match="not lib alias"):
        subject._active_site_directories(prefix, safe["python_version"])


def test_base_site_surface_audits_formal_and_resolved_with_lib64_alias(
    tmp_path: Path,
) -> None:
    formal = tmp_path / "formal"
    resolved = tmp_path / "resolved"
    formal_stdlib = formal / "lib/python3.12"
    formal_site = formal_stdlib / "site-packages"
    resolved_stdlib = resolved / "lib/python3.12"
    resolved_site = resolved_stdlib / "site-packages"
    formal_site.mkdir(parents=True)
    resolved_site.mkdir(parents=True)
    (formal / "lib64").symlink_to("lib", target_is_directory=True)
    (formal / "pyvenv.cfg").write_text(
        "home = {}\ninclude-system-site-packages = true\nversion = 3.12.4\n".format(
            resolved
        ),
        encoding="utf-8",
    )
    marker = tmp_path / "must-not-run"
    for name in (
        "distutils-precedence.pth",
        "ruamel.yaml-0.17.21-py3.12-nspkg.pth",
    ):
        (resolved_site / name).write_text(
            "import pathlib; pathlib.Path({!r}).write_text({!r})\n".format(
                str(marker), name
            ),
            encoding="utf-8",
        )
    safe = {
        "python_version": "3.12.4",
        "prefix": str(resolved),
        "base_prefix": str(resolved),
        "sys_path": [str(resolved_stdlib)],
    }
    surface = subject._audit_base_site_surface(formal, safe)
    assert surface["distinct_prefixes"] is True
    assert surface["site_directories"] == sorted([str(formal_site), str(resolved_site)])
    assert [audit["prefix"] for audit in surface["prefix_audits"]] == [
        str(formal),
        str(resolved),
    ]
    resolved_audit = subject._resolved_base_audit(surface)
    assert {
        Path(item["path"]).name for item in resolved_audit["startup_artifacts"]
    } == {
        "distutils-precedence.pth",
        "ruamel.yaml-0.17.21-py3.12-nspkg.pth",
    }
    assert resolved_audit["activated_paths"] == []
    pth_bindings = subject._pth_never_executed_bindings(
        (resolved_audit,), "fixture resolved base"
    )
    assert {Path(item["path"]).name for item in pth_bindings} == {
        "distutils-precedence.pth",
        "ruamel.yaml-0.17.21-py3.12-nspkg.pth",
    }
    assert all(
        item["execution_policy"] == "present_but_never_executed"
        and subject._is_sha256(item["sha256"])
        and item["executable_line_sha256s"]
        for item in pth_bindings
    )
    assert not marker.exists()


def test_active_site_audit_binds_pth_without_activating_its_runtime_path(
    tmp_path: Path,
) -> None:
    prefix, stdlib, site_packages, safe = _site_fixture(tmp_path)
    extra = prefix / "extra"
    extra.mkdir()
    (site_packages / "fixture.pth").write_text(str(extra) + "\n", encoding="utf-8")
    audit = subject._audit_active_site(prefix, safe_probe=safe, require_pyvenv=False)
    assert audit["activated_paths"] == []
    assert audit["declared_pth_paths"] == [str(extra)]
    assert audit["pth_execution_policy"] == "present_but_never_executed"
    detailed = {
        "sitecustomize_loaded": False,
        "usercustomize_loaded": False,
        "sys_path": [str(stdlib), str(site_packages)],
    }
    subject._validate_probe_startup_surface(detailed, (safe,), (audit,), "fixture")
    detailed["sys_path"].append(str(extra))
    with pytest.raises(subject.EnvironmentBuildError, match="sys.path"):
        subject._validate_probe_startup_surface(detailed, (safe,), (audit,), "fixture")


@pytest.mark.parametrize(
    "directive", ["import fixture_bootstrap\n", "  import\tfixture_bootstrap\n"]
)
def test_active_site_audit_records_executable_pth_as_never_executed(
    directive: str, tmp_path: Path
) -> None:
    prefix, _stdlib, site_packages, safe = _site_fixture(tmp_path)
    pth = site_packages / "fixture.pth"
    pth.write_text(directive, encoding="utf-8")
    audit = subject._audit_active_site(prefix, safe_probe=safe, require_pyvenv=False)
    assert audit["activated_paths"] == []
    assert audit["startup_artifacts"] == [
        {
            "path": str(pth),
            "kind": "pth",
            "sha256": _sha(pth),
            "execution_policy": "present_but_never_executed",
            "directives": [
                {
                    "line": 1,
                    "kind": "executable_present_but_never_executed",
                    "line_sha256": hashlib.sha256(
                        directive.rstrip("\n").encode("utf-8")
                    ).hexdigest(),
                }
            ],
            "declared_paths": [],
            "activated_paths": [],
        }
    ]


@pytest.mark.parametrize("name", ["legacy.egg-link", "legacy.egg"])
def test_active_site_audit_rejects_legacy_injection(name: str, tmp_path: Path) -> None:
    prefix, _stdlib, site_packages, safe = _site_fixture(tmp_path)
    path = site_packages / name
    if name.endswith(".egg"):
        path.mkdir()
    else:
        path.write_text("/untrusted\n", encoding="utf-8")
    with pytest.raises(subject.EnvironmentBuildError, match="legacy"):
        subject._audit_active_site(prefix, safe_probe=safe, require_pyvenv=False)


def test_active_site_audit_rejects_customization_and_unknown_pyvenv_key(
    tmp_path: Path,
) -> None:
    prefix, _stdlib, site_packages, safe = _site_fixture(tmp_path)
    (site_packages / "sitecustomize.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(subject.EnvironmentBuildError, match="customization"):
        subject._audit_active_site(prefix, safe_probe=safe, require_pyvenv=False)
    (site_packages / "sitecustomize.py").unlink()
    (prefix / "pyvenv.cfg").write_text(
        "include-system-site-packages = true\nunknown = value\n", encoding="utf-8"
    )
    with pytest.raises(subject.EnvironmentBuildError, match="unknown/duplicate"):
        subject._audit_active_site(prefix, safe_probe=safe, require_pyvenv=True)


def test_build_failure_publishes_zero_success_receipts(
    monkeypatch, tmp_path: Path
) -> None:
    base_python, official, root, _base, _lock = _install_build_mocks(
        monkeypatch, tmp_path
    )

    def fail(*args, **kwargs):
        raise subject.EnvironmentBuildError("fixture download failed")

    monkeypatch.setattr(subject, "_download_and_lock_wheels", fail)
    with pytest.raises(subject.EnvironmentBuildError, match="download failed"):
        subject.build_environment(
            base_python=base_python,
            environment_root=root,
            official_env_yaml=official,
            wheel_lock_path=REVIEWED_LOCK,
            expected_wheel_lock_sha256=subject.REVIEWED_WHEEL_LOCK_FILE_SHA256,
        )
    assert root.is_dir()
    assert not (root / subject.RECEIPT_NAME).exists()
    assert list(root.rglob(subject.RECEIPT_NAME)) == []


def test_success_receipt_schema_scipy_numpy_disclosure_and_publication_last(
    monkeypatch, tmp_path: Path
) -> None:
    labels = []
    original = subject._write_json_new_immutable

    def observed_writer(path, value, label, **kwargs):
        labels.append(label)
        return original(path, value, label, **kwargs)

    monkeypatch.setattr(subject, "_write_json_new_immutable", observed_writer)
    result, base_python, official, root, _base, _lock = _build(monkeypatch, tmp_path)
    receipt_path = root / subject.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert labels[-1] == "environment completion receipt"
    assert result["receipt_sha256"] == _sha(receipt_path)
    assert set(receipt) == {
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
    }
    assert receipt["application_distributions_installed"]["scipy"] == "1.14.1"
    assert receipt["runtime_no_site"] is True
    runtime_evidence = receipt["build_evidence"]["runtime_no_site"]
    assert runtime_evidence["schema_version"] == subject.RUNTIME_NO_SITE_SCHEMA
    assert runtime_evidence["runtime_no_site"] is True
    assert runtime_evidence["interpreter_flags"] == ["-I", "-S", "-B"]
    assert runtime_evidence["bootstrap"]["site_main_and_pth_processing_blocked"] is True
    assert runtime_evidence["target"]["injected_site_directories"][0]["prefix"] == str(
        root
    )
    assert runtime_evidence["target"]["import_origins"]
    assert {
        item["distribution"] for item in runtime_evidence["target"]["import_origins"]
    } == {
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "torch-geometric",
        "opencv-python",
        "scipy",
        "pip",
    }
    subject._validate_runtime_no_site_evidence(runtime_evidence)
    assert "scipy" not in receipt["inherited_distributions_verified"]
    assert (
        receipt["build_evidence"]["install"]["report_path_normalization"]["policy"]
        == "execution-anchor-file-url-to-canonical-lexical-wheel-path"
    )
    assert receipt["base_runtime"]["numpy_adaptation"] == {
        "distribution": "numpy",
        "official_shreddingnet_version": "2.2.3",
        "inherited_base_version": "2.1.3",
        "intentional": True,
        "reason": "Rachel benchmark compatibility with the provisioned Torch 2.5.1+cu124 runtime",
    }
    subject._validate_content_sha(receipt, "receipt")
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    with pytest.raises(subject.EnvironmentBuildError, match="fresh"):
        subject.build_environment(
            base_python=base_python,
            environment_root=root,
            official_env_yaml=official,
            wheel_lock_path=REVIEWED_LOCK,
            expected_wheel_lock_sha256=subject.REVIEWED_WHEEL_LOCK_FILE_SHA256,
        )


def test_verify_summary_exact_and_rechecks_expected_receipt_hash(
    monkeypatch, tmp_path: Path
) -> None:
    result, _base_python, _official, root, _base, _lock = _build(monkeypatch, tmp_path)
    verified = subject.verify_environment(
        environment_root=root,
        expected_receipt_sha256=result["receipt_sha256"],
    )
    assert set(verified) == {
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
    }
    assert verified["status"] == "verified_isolated_benchmark_environment"
    with pytest.raises(subject.EnvironmentBuildError, match="receipt file SHA-256"):
        subject.verify_environment(
            environment_root=root,
            expected_receipt_sha256="0" * 64,
        )


def test_verify_fails_on_imported_package_content_drift(
    monkeypatch, tmp_path: Path
) -> None:
    result, _base_python, _official, root, _base, _lock = _build(monkeypatch, tmp_path)
    drifted = _inventory(root, root / "bin/python")
    drifted["tracked"] = {"tampered": True}
    drifted = subject._content_bound(
        {key: value for key, value in drifted.items() if key != "content_sha256"}
    )
    monkeypatch.setattr(
        subject,
        "_import_inventory",
        lambda python, root, **kwargs: drifted,
    )
    with pytest.raises(
        subject.EnvironmentBuildError, match="content inventory drifted"
    ):
        subject.verify_environment(
            environment_root=root,
            expected_receipt_sha256=result["receipt_sha256"],
        )


def test_verify_rejects_writable_or_symlink_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    _result, _base_python, _official, root, _base, _lock = _build(monkeypatch, tmp_path)
    receipt = root / subject.RECEIPT_NAME
    receipt.chmod(0o644)
    with pytest.raises(subject.EnvironmentBuildError, match="read-only"):
        subject.verify_environment(environment_root=root)
    receipt.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    receipt.symlink_to(outside)
    with pytest.raises(subject.EnvironmentBuildError, match="symlink"):
        subject.verify_environment(environment_root=root)


def test_cli_failure_returns_nonzero_and_prints_no_success_json(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        subject,
        "verify_environment",
        lambda **kwargs: (_ for _ in ()).throw(
            subject.EnvironmentBuildError("fixture")
        ),
    )
    code = subject.main(("verify", "--environment-root", "/fixture/root"))
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err == "ERROR: fixture\n"
