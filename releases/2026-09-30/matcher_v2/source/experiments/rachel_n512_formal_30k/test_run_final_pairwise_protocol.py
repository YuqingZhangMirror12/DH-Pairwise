from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("run_final_pairwise_protocol.sh")
PROVENANCE = Path(__file__).with_name("rachel_data_provenance_pretest.json")


def _bash3_safety_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Mirror only the controller authority needed for pre-evaluation fixtures."""

    source_root = tmp_path / "fixture-source"
    controller = (
        source_root
        / "experiments/rachel_n512_formal_30k/run_final_pairwise_protocol.sh"
    )
    controller.parent.mkdir(parents=True)
    payload = SCRIPT.read_text(encoding="utf-8")
    old_gate = "if (( BASH_VERSINFO[0] < 4 )); then"
    assert payload.count(old_gate) == 1
    # The production file retains its Bash>=4 gate.  This fixture-only copy
    # lowers just that gate so macOS Bash 3.2 can exercise pre-evaluation paths.
    controller.write_text(payload.replace(old_gate, "if (( BASH_VERSINFO[0] < 3 )); then", 1), encoding="utf-8")
    controller.chmod(0o755)
    required = (
        "experiments/rachel_n512_formal_30k/paired_cluster_bootstrap.py",
        "staging/pairwise_v0_2/training/rachel_n512_sealed_test.py",
        "staging/pairwise_v0_2/baselines/rachel_same_data_benchmark_eval_adapter.py",
        "staging/pairwise_v0_2/baselines/rachel_pairingnet_benchmark.py",
        "staging/pairwise_v0_2/baselines/rachel_shreddingnet_benchmark.py",
        "staging/pairwise_v0_2/baselines/rachel_n512_corrosion_robustness.py",
        "staging/pairwise_v0_2/baselines/rachel_n512_real_external.py",
        "staging/pairwise_v0_2/baselines/rachel_n512_real_translation_gt.py",
    )
    for relative in required:
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# local safety fixture; never imported for evaluation\n", encoding="utf-8")
    return source_root, controller


def _shell_function(source: str, name: str, next_name: str) -> str:
    start = source.index(name + "() {")
    stop = source.index(next_name + "() {", start)
    return source[start:stop]


def test_controller_has_valid_bash_syntax_and_frozen_stage_order() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text(encoding="utf-8")
    lines = source.splitlines()
    heredoc_count = 0
    index = 0
    while index < len(lines):
        if "<<'PY'" in lines[index]:
            end = index + 1
            while end < len(lines) and lines[end] != "PY":
                end += 1
            assert end < len(lines)
            ast.parse("\n".join(lines[index + 1 : end]) + "\n")
            heredoc_count += 1
            index = end
        index += 1
    assert heredoc_count >= 15
    commands = (
        "-m staging.pairwise_v0_2.training.rachel_n512_sealed_test",
        "-m staging.pairwise_v0_2.baselines.rachel_n512_corrosion_robustness",
        "-m staging.pairwise_v0_2.baselines.rachel_n512_real_external",
        "-m experiments.rachel_n512_formal_30k.paired_cluster_bootstrap \\\n  --synthetic-receipt",
        "-m experiments.rachel_n512_formal_30k.paired_cluster_bootstrap \\\n  --real-json",
        "-m staging.pairwise_v0_2.baselines.rachel_n512_real_translation_gt",
    )
    positions = [source.index(command) for command in commands]
    assert positions == sorted(positions)
    assert source.count(commands[0]) == 1
    assert source.count(commands[1]) == 1
    assert source.count(commands[2]) == 1
    assert source.count(commands[3]) == 1
    assert source.count(commands[4]) == 1
    assert source.count(commands[5]) == 1
    assert source.count('--pairingnet-run-directory "$PAIRINGNET_RUN_DIRECTORY"') == 3
    assert source.count('--shreddingnet-freeze-path "$SHREDDINGNET_FREEZE_PATH"') == 3
    assert 'readonly BENCHMARK_QUEUE_GATE="$CONTROL_ROOT/same_data_benchmark_queue_gate.json"' in source
    assert 'CURRENT_STAGE="waiting_for_completed_same_data_benchmark_queue"' in source
    assert 'CURRENT_STAGE="same_data_benchmark_queue_terminal_and_inventory_validation"' in source
    assert source.index("verify_completed_benchmark_queue() {") < source.index(
        'CURRENT_STAGE="train_validation_freeze_gate"'
    )
    assert source.index('CURRENT_STAGE="train_validation_freeze_gate"') < source.index(
        'CURRENT_STAGE="sealed_synthetic"'
    )
    assert '"formal_exact_six":True' in source
    assert '"native_cm_fm_se_or_ga_claimed":False' in source
    assert '"automatic_performance_pass_fail_applied":False' in source
    assert "legacy_four_method_corrosion_stage" not in source
    assert "corrosion_clean_reference_four_method_projection" not in source
    assert "formal_exact_six_seven_condition_corrosion_complete_and_verified" in source
    corrosion_command = source[
        source.index(commands[1]) : source.index('CURRENT_STAGE="corrosion_receipt_validation"')
    ]
    assert '--sealed-test-directory "$SEALED_TEST_DIRECTORY"' in corrosion_command
    assert '--pairingnet-run-directory "$PAIRINGNET_RUN_DIRECTORY"' in corrosion_command
    assert '--shreddingnet-freeze-path "$SHREDDINGNET_FREEZE_PATH"' in corrosion_command
    assert '"same_corrupted_masks_supplied_to_all_six_methods"' in source
    assert '"each_method_forwarded_each_pair_exactly_once_per_condition"' in source
    assert '"benchmark_correspondence_not_applicable_under_corrosion": True' in source
    assert '"reviewed_GO_markers": reviewed_go_markers' in source
    for review_name in (
        "queue_controller",
        "train_val_asset_freeze",
        "benchmark_environment",
        "pairingnet_runner",
        "shreddingnet_runner",
    ):
        assert '"{}"'.format(review_name) in source

    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert "export PYTHONDONTWRITEBYTECODE=1" in source
    first_python_call = source.index('"$PYTHON"')
    for hardening in (
        "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "export PYTHONDONTWRITEBYTECODE=1",
        "export PYTHONNOUSERSITE=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export OMP_NUM_THREADS=1",
        'export PYTHONPATH="$IMMUTABLE_SOURCE_ROOT"',
    ):
        assert source.index(hardening) < first_python_call
    assert source.index("BASH_VERSINFO[0] < 4") < source.index("run_logged() {")
    assert 'readonly LOG_ROOT="$CONTROL_ROOT/logs"' in source
    assert "trap 'on_signal TERM 143' TERM" in source
    assert "os.link(" in source
    assert "os.fsync(" in source
    assert "complete_final_pairwise_protocol" in source
    assert "content_sha256_inventory.json" in source
    assert "terminal_receipt.json" in source
    assert "RACHEL_DATA_PROVENANCE" in source
    assert '"canonical_model_and_loss_config_authority"' in source
    assert '"rachel_data_provenance"' in source
    assert (
        "receipts_winner_checkpoints_configs_and_data_provenance_reverified_after_all_evaluations"
        in source
    )
    for relative in (
        "preprocess_receipt.json",
        "qa/preprocess_summary.json",
        "pairs/train.jsonl",
        "pairs/val.jsonl",
    ):
        assert relative in source
    assert '"same_folder_hard": 6000' in source
    assert '"cross_folder_scale_matched": 6000' in source
    assert '"rachel-pairwise-n512-preprocessing/1.0"' in source
    assert '"complete_rachel_pairwise_n512_30k"' in source
    assert '"rachel-pairwise-30k-selection/1.0"' in source
    assert '"preprocess_summary_exactly_embedded_in_receipt": True' in source
    assert (
        'get("declared_line_count_from_preprocess_receipt") != 3000' in source
    )
    assert source.count("terminate_active_process_tree || true") >= 3
    assert "cleanup_launcher_ready_file || true" in source
    assert "ssh " not in source
    assert "scp " not in source
    assert "rm -" not in source


def test_pretest_provenance_declares_only_complete_sha256_digests() -> None:
    document = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    digests = []

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "sha256":
                    digests.append(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    assert len(digests) == 4
    assert all(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
        for value in digests
    )


def test_formal_corrosion_gate_is_exact_six_and_has_no_legacy_headline() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    method_literal = (
        '["coarse_only", "full_n512", "matched_mm_converged", '
        '"matched_mm_same_exposure_epoch5", "pairingnet_adapted", '
        '"shreddingnet_adapted"]'
    )
    assert "rachel-n512-corrosion-robustness/2.0" in source
    assert method_literal in source
    assert 'bootstrap.get("formal_exact_six") is not True' in source
    assert 'bootstrap.get("method_order") != corrosion_methods' in source
    assert 'list(direct) != translation_methods' in source
    assert 'audit.get("pair_forward_count_by_method")' in source
    assert 'audit.get("batch_forward_count_by_method")' in source
    assert 'audit.get("same_common_corrupted_mask_arrays_used_by_all_methods")' in source
    assert 'set(by_condition[condition]) != direct_metric_names' in source
    assert 'primary.get("definition") != "all_six_methods_valid_at_all_seven_conditions"' in source
    assert '"formal_exact_six": True' in source
    assert '"six_methods_verified": True' in source
    assert "clean_reference_projection" not in source


def test_production_controller_rejects_bash_older_than_four(tmp_path: Path) -> None:
    major = int(subprocess.check_output(["bash", "-c", "printf %s \"${BASH_VERSINFO[0]}\""]).decode())
    if major >= 4:
        assert "if (( BASH_VERSINFO[0] < 4 )); then" in SCRIPT.read_text(encoding="utf-8")
        return
    final_root = tmp_path / "must-not-exist"
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, "FINAL_ROOT": str(final_root)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert "requires Bash >= 4" in result.stderr
    assert not final_root.exists()


def test_nonformal_replicate_override_fails_before_output_or_data_access(
    tmp_path: Path,
) -> None:
    source_root, controller = _bash3_safety_fixture(tmp_path)
    final_root = tmp_path / "must-not-exist"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": sys.executable,
            "IMMUTABLE_SOURCE_ROOT": str(source_root),
            "FINAL_ROOT": str(final_root),
            "STATS_REPLICATES": "19999",
        }
    )
    result = subprocess.run(
        ["bash", str(controller)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "exactly 20000" in result.stderr
    assert not final_root.exists()


def test_default_formal_statistics_seed_passes_seed_gate_before_output(
    tmp_path: Path,
) -> None:
    source_root, controller = _bash3_safety_fixture(tmp_path)
    final_root = tmp_path / "must-not-exist"
    environment = os.environ.copy()
    environment.pop("STATS_SEED", None)
    environment.update(
        {
            "PYTHON": sys.executable,
            "IMMUTABLE_SOURCE_ROOT": str(source_root),
            "FINAL_ROOT": str(final_root),
            "DATASET_ROOT": "deliberately-relative-after-seed-gate",
        }
    )
    result = subprocess.run(
        ["bash", str(controller)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "formal statistics require exactly seed" not in result.stderr
    assert "all configured filesystem paths must be absolute" in result.stderr
    assert not final_root.exists()


def test_wrong_formal_statistics_seed_fails_preopen_and_preoutput(
    tmp_path: Path,
) -> None:
    source_root, controller = _bash3_safety_fixture(tmp_path)
    final_root = tmp_path / "must-not-exist"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": sys.executable,
            "IMMUTABLE_SOURCE_ROOT": str(source_root),
            "FINAL_ROOT": str(final_root),
            "DATASET_ROOT": "would-fail-if-the-seed-gate-were-late",
            "STATS_SEED": "260901",
        }
    )
    result = subprocess.run(
        ["bash", str(controller)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "formal statistics require exactly seed 20260901" in result.stderr
    assert "all configured filesystem paths must be absolute" not in result.stderr
    assert not final_root.exists()

    source = SCRIPT.read_text(encoding="utf-8")
    seed_gate = source.index("(( STATS_SEED == 20260901 ))")
    assert seed_gate < source.index("for value in \"$IMMUTABLE_SOURCE_ROOT\"")
    assert seed_gate < source.index('"$PYTHON" - "$FINAL_ROOT"')
    assert seed_gate < source.index('mkdir "$FINAL_ROOT"')
    assert seed_gate < source.index('CURRENT_STAGE="synthetic_statistics"')


def test_output_inside_protected_input_fails_before_first_mkdir(
    tmp_path: Path,
) -> None:
    source_root, controller = _bash3_safety_fixture(tmp_path)
    protected = tmp_path / "protected-dataset"
    protected.mkdir()
    final_root = protected / "must-not-be-created"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": sys.executable,
            "IMMUTABLE_SOURCE_ROOT": str(source_root),
            "FINAL_ROOT": str(final_root),
            "DATASET_ROOT": str(protected),
            "CONVERGENCE_ROOT": str(tmp_path / "convergence"),
            "MATCHED_ROOT": str(tmp_path / "matched"),
            "REAL_AUTHORITY_ROOT": str(tmp_path / "real"),
            "REAL_MAIN_ROOT": str(tmp_path / "real/main"),
            "REAL_SUPP_ROOT": str(tmp_path / "real/supp"),
            "REAL_MANIFEST": str(tmp_path / "real-manifest/manifest.json"),
            "REAL_LOCAL_RECEIPT": str(tmp_path / "real-manifest/receipt.json"),
        }
    )
    result = subprocess.run(
        ["bash", str(controller)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert "protected input namespace" in result.stderr
    assert not final_root.exists()


def test_sigterm_while_waiting_writes_atomic_failure_receipt_without_data_open(
    tmp_path: Path,
) -> None:
    source_root, controller_path = _bash3_safety_fixture(tmp_path)
    convergence = tmp_path / "convergence"
    matched = tmp_path / "matched"
    convergence.mkdir()
    matched.mkdir()
    final_root = tmp_path / "final"

    sleeper = subprocess.Popen(["sleep", "60"])
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": sys.executable,
            "IMMUTABLE_SOURCE_ROOT": str(source_root),
            "CONVERGENCE_ROOT": str(convergence),
            "MATCHED_ROOT": str(matched),
            "FINAL_ROOT": str(final_root),
            "MATCHED_CONTROLLER_PID": str(sleeper.pid),
            "MATCHED_CONTROLLER_CMD_PATTERN": "sleep",
            "WAIT_INTERVAL_SECONDS": "1",
            "WAIT_TIMEOUT_SECONDS": "0",
            # These paths must remain unopened because no training gate exists.
            "DATASET_ROOT": str(tmp_path / "forbidden-synthetic"),
            "REAL_AUTHORITY_ROOT": str(tmp_path / "forbidden-real"),
        }
    )
    controller = subprocess.Popen(
        ["bash", str(controller_path)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    failure = final_root / "control/terminal_receipt.json"
    source_freeze = final_root / "control/source_code_freeze.json"
    deadline = time.monotonic() + 20.0
    try:
        while time.monotonic() < deadline:
            if source_freeze.is_file():
                break
            if controller.poll() is not None:
                break
            time.sleep(0.05)
        assert source_freeze.is_file(), controller.communicate(timeout=1)
        controller.send_signal(signal.SIGTERM)
        stdout, stderr = controller.communicate(timeout=10)
        assert controller.returncode == 143, (stdout, stderr)
        assert failure.is_file()
        receipt = json.loads(failure.read_text(encoding="utf-8"))
        assert receipt["status"] == "failed_final_pairwise_protocol"
        assert receipt["termination"] == "TERM"
        assert "signal_TERM" in receipt["stage"]
        assert receipt["metric_artifacts_read_for_integrity_validation"] is False
        assert (
            receipt[
                "metric_values_used_for_adaptive_model_checkpoint_threshold_or_condition_selection"
            ]
            is False
        )
        assert receipt["metric_values_emitted_to_controller_stdout"] is False
        assert not (tmp_path / "forbidden-synthetic").exists()
        assert not (tmp_path / "forbidden-real").exists()
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5)
        sleeper.terminate()
        sleeper.wait(timeout=5)


def test_missing_benchmark_queue_fails_before_test_or_real_open(
    tmp_path: Path,
) -> None:
    source_root, controller = _bash3_safety_fixture(tmp_path)
    convergence = tmp_path / "convergence"
    matched = tmp_path / "matched"
    for root, schema in (
        (convergence, "rachel-n512-train-run/1.0"),
        (matched, "rachel-matched-historical-mm-train/1.0"),
    ):
        run = root / "run-fixture"
        run.mkdir(parents=True)
        (run / "run_receipt.json").write_text(
            json.dumps(
                {
                    "schema_version": schema,
                    "status": "complete_train_validation_only",
                    "test_accessed": False,
                    "real_external_test_accessed": False,
                }
            ),
            encoding="utf-8",
        )
    final_root = tmp_path / "late-failure"
    result = subprocess.run(
        ["bash", str(controller)],
        env={
            **os.environ,
            "PYTHON": sys.executable,
            "IMMUTABLE_SOURCE_ROOT": str(source_root),
            "CONVERGENCE_ROOT": str(convergence),
            "MATCHED_ROOT": str(matched),
            "DATASET_ROOT": str(tmp_path / "dataset"),
            "REAL_AUTHORITY_ROOT": str(tmp_path / "real"),
            "BENCHMARK_QUEUE_ROOT": str(tmp_path / "missing-benchmark-queue"),
            "BENCHMARK_QUEUE_PID": "",
            "FINAL_ROOT": str(final_root),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    terminal = final_root / "control/terminal_receipt.json"
    receipt = json.loads(terminal.read_text(encoding="utf-8"))
    declared = receipt.pop("content_sha256")
    canonical = json.dumps(
        receipt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert declared == hashlib.sha256(canonical).hexdigest()
    assert receipt["status"] == "failed_final_pairwise_protocol"
    assert receipt["stage"] == "waiting_for_completed_same_data_benchmark_queue"
    assert receipt["metric_artifacts_read_for_integrity_validation"] is False
    assert not (tmp_path / "dataset").exists()
    assert not (tmp_path / "real").exists()


def test_tampered_benchmark_queue_terminal_fails_before_test_or_real_open(
    tmp_path: Path,
) -> None:
    source_root, controller = _bash3_safety_fixture(tmp_path)
    convergence = tmp_path / "convergence"
    matched = tmp_path / "matched"
    for root, schema in (
        (convergence, "rachel-n512-train-run/1.0"),
        (matched, "rachel-matched-historical-mm-train/1.0"),
    ):
        run = root / "run-fixture"
        run.mkdir(parents=True)
        (run / "run_receipt.json").write_text(
            json.dumps(
                {
                    "schema_version": schema,
                    "status": "complete_train_validation_only",
                    "test_accessed": False,
                    "real_external_test_accessed": False,
                }
            ),
            encoding="utf-8",
        )
    queue_root = tmp_path / "benchmark-queue"
    terminal = queue_root / "control/terminal_receipt.json"
    terminal.parent.mkdir(parents=True)
    terminal.write_text(
        json.dumps(
            {
                "schema_version": "rachel-same-data-benchmark-queue/2.0",
                "status": "GO_complete_same_data_benchmark_queue",
                "content_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    final_root = tmp_path / "tampered-queue-failure"
    result = subprocess.run(
        ["bash", str(controller)],
        env={
            **os.environ,
            "PYTHON": sys.executable,
            "IMMUTABLE_SOURCE_ROOT": str(source_root),
            "CONVERGENCE_ROOT": str(convergence),
            "MATCHED_ROOT": str(matched),
            "BENCHMARK_QUEUE_ROOT": str(queue_root),
            "DATASET_ROOT": str(tmp_path / "forbidden-synthetic"),
            "REAL_AUTHORITY_ROOT": str(tmp_path / "forbidden-real"),
            "FINAL_ROOT": str(final_root),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    receipt = json.loads(
        (final_root / "control/terminal_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "failed_final_pairwise_protocol"
    assert receipt["stage"] == "same_data_benchmark_queue_terminal_and_inventory_validation"
    assert receipt["metric_artifacts_read_for_integrity_validation"] is True
    assert not (tmp_path / "forbidden-synthetic").exists()
    assert not (tmp_path / "forbidden-real").exists()


def test_term_ignoring_process_group_is_killed_and_ready_file_removed(
    tmp_path: Path,
) -> None:
    ready = tmp_path / ".launcher-ready-fixture"
    child_ready = tmp_path / "child-ready"
    launcher = tmp_path / "stubborn_group.py"
    launcher.write_text(
        """\
import pathlib
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
child_code = '''
import pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text(str(__import__('os').getpid()), encoding='utf-8')
while True: time.sleep(1)
'''
child = subprocess.Popen([sys.executable, '-c', child_code, sys.argv[2]])
deadline = time.monotonic() + 5
while not pathlib.Path(sys.argv[2]).is_file():
    if time.monotonic() >= deadline: raise SystemExit('child handshake timeout')
    time.sleep(0.01)
pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')
while True: time.sleep(1)
""",
        encoding="utf-8",
    )
    leader = subprocess.Popen(
        [sys.executable, str(launcher), str(ready), str(child_ready)],
        start_new_session=True,
    )
    deadline = time.monotonic() + 10
    while not ready.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.is_file()

    source = SCRIPT.read_text(encoding="utf-8")
    functions = _shell_function(
        source, "cleanup_launcher_ready_file", "terminate_active_process_tree"
    ) + _shell_function(source, "terminate_active_process_tree", "on_signal")
    harness = tmp_path / "process_group_harness.sh"
    harness.write_text(
        "set -u\n"
        + functions
        + "\nterminate_active_process_tree\n"
        + "cleanup_launcher_ready_file\n",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "ACTIVE_CHILD": str(leader.pid),
        "ACTIVE_PROCESS_GROUP": str(leader.pid),
        "STARTING_CHILD": "0",
        "LAUNCH_READY_FILE": str(ready),
    }
    harness_process = subprocess.Popen(
        ["bash", str(harness)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        leader.wait(timeout=12)
        stdout, stderr = harness_process.communicate(timeout=5)
        assert harness_process.returncode == 0, (stdout, stderr)
        assert not ready.exists()
        with pytest.raises(ProcessLookupError):
            os.killpg(leader.pid, 0)
    finally:
        if leader.poll() is None:
            os.killpg(leader.pid, signal.SIGKILL)
            leader.wait(timeout=5)
        if harness_process.poll() is None:
            harness_process.kill()
            harness_process.wait(timeout=5)


def test_final_checkpoint_verifier_rejects_every_symlink_component(
    tmp_path: Path,
) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    lines = source.splitlines()
    function_source = None
    index = 0
    while index < len(lines):
        if "<<'PY'" not in lines[index]:
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end] != "PY":
            end += 1
        block = "\n".join(lines[index + 1 : end]) + "\n"
        if "def resolve_without_symlink_components" in block:
            tree = ast.parse(block)
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "resolve_without_symlink_components"
            )
            function_source = compile(
                ast.Module(body=[function], type_ignores=[]),
                filename="embedded_symlink_verifier",
                mode="exec",
            )
            break
        index = end + 1
    assert function_source is not None
    namespace = {"Path": Path, "os": os, "stat": __import__("stat")}
    exec(function_source, namespace)
    verifier = namespace["resolve_without_symlink_components"]

    real = tmp_path / "real"
    real.mkdir()
    checkpoint = real / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen")
    assert verifier(str(checkpoint), "checkpoint") == checkpoint.resolve()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(SystemExit, match="traverses a symlink"):
        verifier(str(alias / checkpoint.name), "checkpoint")
