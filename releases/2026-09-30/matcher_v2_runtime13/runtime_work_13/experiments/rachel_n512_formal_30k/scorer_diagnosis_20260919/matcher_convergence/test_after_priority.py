"""Finite dependency ledger tests with fake processes, never a live dispatch."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import after_priority as a


def fixture():
    owner = dict(pid=2468, startticks=987654, argv=["python", "-m", "priority", "--execute"])
    stages = [dict(name="stage-%02d" % i, command=["python", "task.py", str(i)],
        cwd="/fixture/source", env={"CUDA_VISIBLE_DEVICES": "0"},
        completion="/fixture/output/stage-%02d/status.json" % i,
        completion_expect=dict(status="complete")) for i in range(60)]
    spec = dict(process=owner, status_path="/fixture/priority/status.json",
        schema_field="schema", schema="fixture-priority/1", transaction_id="fixture-transaction",
        completed_stages=60, stage_ledger=stages)
    receipt = dict(schema=spec["schema"], pid=owner["pid"], transaction_id=spec["transaction_id"],
        status="complete", completed_stages=60, active_pid=None, active_name=None,
        stages=[dict(s, status="complete", returncode=0) for s in deepcopy(stages)])
    return receipt, dict(owner, state="S"), spec


class AfterPriorityTests(unittest.TestCase):
    def test_complete_exact_ledger_needs_observed_terminal_process(self):
        receipt, process, spec = fixture()
        for terminal in (None, dict(process, state="Z", argv=[]), dict(process, state="X", argv=[])):
            with self.subTest(process=terminal):
                self.assertEqual(a.dependency_state(receipt, terminal, spec), "ready")

    def test_live_predecessor_waits_even_after_complete_receipt(self):
        receipt, process, spec = fixture()
        for status in ("waiting_current_queue", "running", "complete"):
            with self.subTest(status=status):
                self.assertEqual(a.dependency_state(dict(receipt, status=status), process, spec), "waiting")

    def test_failed_or_missing_running_process_refuses_dispatch(self):
        receipt, process, spec = fixture()
        for status in ("failed", "interrupted", "unknown"):
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                a.dependency_state(dict(receipt, status=status), process, spec)
        for status in ("waiting_current_queue", "running"):
            for terminal in (None, dict(process, state="Z", argv=[])):
                with self.subTest(status=status, process=terminal), self.assertRaises(RuntimeError):
                    a.dependency_state(dict(receipt, status=status), terminal, spec)

    def test_complete_receipt_cannot_hide_incomplete_stage_or_active_child(self):
        receipt, _, spec = fixture()
        for update in (dict(completed_stages=59), dict(active_pid=555), dict(active_name="still-running")):
            with self.subTest(update=update), self.assertRaises(ValueError):
                a.dependency_state(dict(receipt, **update), None, spec)
        for update in (dict(status="running"), dict(returncode=1), dict(returncode=None)):
            bad = deepcopy(receipt)
            bad["stages"][-1].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                a.dependency_state(bad, None, spec)

    def test_registered_declarations_population_and_receipt_owner_are_exact(self):
        receipt, process, spec = fixture()
        bad_receipts = [dict(receipt, stages=receipt["stages"][:-1]),
            dict(receipt, schema="different/1"), dict(receipt, transaction_id="other"),
            dict(receipt, pid=2469)]
        for key, value in (('command', ['python', 'another.py']),
                ('env', {"CUDA_VISIBLE_DEVICES": "1"}), ('completion_expect', {"status": "paused"})):
            bad = deepcopy(receipt)
            bad["stages"][10][key] = value
            bad_receipts.append(bad)
        for index, bad in enumerate(bad_receipts):
            with self.subTest(index=index), self.assertRaises(ValueError):
                a.dependency_state(bad, process, spec)

    def test_reused_pid_changed_command_and_stopped_process_are_not_ready(self):
        receipt, process, spec = fixture()
        for update in (dict(pid=2469), dict(startticks=987655),
                dict(startticks=987655, state="Z", argv=[]),
                dict(argv=["unrelated", "process"]), dict(state="T"), dict(state="t")):
            with self.subTest(update=update), self.assertRaises(RuntimeError):
                a.dependency_state(receipt, dict(process, **update), spec)

    def test_wait_deadline_records_failure_without_spawn_or_signals(self):
        receipt, process, spec = fixture()
        receipt["status"] = "running"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = dict(output_root=str(root), dependency=spec, deadline_hours=1/3600,
                stages=[], required_receipts=[])
            def process_snapshot(pid):
                return process if pid == spec["process"]["pid"] else dict(pid=pid, startticks=123, state="S", argv=[])
            with patch.object(a, "load_plan", return_value=plan), \
                    patch.object(a, "read", return_value=receipt), \
                    patch.object(a, "process_identity", side_effect=process_snapshot), \
                    patch.object(a.os, "getsid", return_value=a.os.getpid()), \
                    patch.object(a.time, "monotonic", side_effect=[0., 2.]), \
                    patch.object(a.time, "sleep") as sleep, \
                    patch.object(a.subprocess, "Popen") as spawn, \
                    patch.object(a.os, "kill") as kill, patch.object(a.os, "killpg") as killpg:
                with self.assertRaisesRegex(TimeoutError, "predecessor untouched"):
                    a.execute(root/"fake-plan.json")
            sleep.assert_not_called()
            spawn.assert_not_called()
            kill.assert_not_called()
            killpg.assert_not_called()
            status = json.loads((root/"status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["completed_stages"], 0)
        self.assertFalse(status["active_child_may_still_run"])
        self.assertFalse(status["existing_queue_modified"])


if __name__ == "__main__":
    unittest.main()
