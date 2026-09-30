"""Finite parent/child command tests. No future freeze, model or GPU is run."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.rachel_n512_formal_30k import prepare_score_after050_inputs as tail


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, status="complete", source_density_version="v3"):
    paths = tail.locations(root, source_density_version)
    for source, entries in (("source", ("run_recall_benchmark_queue",)),
            ("input_source", ("prepare_score_input_queue", "train_score_input_variant", "evaluate_score_input_variant")),
            ("density_source", ("prepare_score_density_queue", "train_score_density", "evaluate_score_density"))):
        for name in entries:
            path = paths[source] / "experiments/rachel_n512_formal_30k" / (name + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    path = paths["density_source"] / "staging/pairwise_v0_2/pairwise_data" / ("rachel_density_ownership_" + source_density_version + ".py")
    path.parent.mkdir(parents=True)
    path.touch()
    if source_density_version == "v4":
        path.with_name("rachel_density_outer_certificate.py").touch()
    tail_config = tail.after020_config(root, 99999990)
    configs = [("after020_s3_to050", 99999991, tail_config)]
    # Rebase the actual saved producer configs; no actual data/predictions needed.
    report = Path(__file__).resolve().parents[2] / "reports/rachel_score_design_20260913_001"
    for name, pid in (("train_density_preparation_v3", 99999992), ("clean_density_preparation_v3", 99999993)):
        raw = (report / ("queue_" + name + ".json")).read_text()
        raw = raw.replace("v3", source_density_version)
        name = name.replace("v3", source_density_version)
        config = json.loads(raw.replace("/root/autodl-tmp/rachel_score_design_20260913_001", str(root)))
        configs.append((name, pid, config))
    for name, pid, config in configs:
        dump(root / "queues" / name / "config.json", config)
        dump(root / "queues" / name / "queue_state.json", dict(config=config, pid=pid, status=status))
    return dict(root=root, after020_queue_pid=99999991, train_density_queue_pid=99999992,
        clean_density_queue_pid=99999993, s3_comparison=paths["comparison"], source_density_version=source_density_version)


class InputTailTests(unittest.TestCase):
    def test_v4_changes_only_density_source_and_owns_new_parent(self):
        kw = dict(s3_comparison="/experiment/s3/candidate_dual/paired_freeze.json")
        old = tail.build_config("/experiment", 11, 12, 13, **kw)
        new = tail.build_config("/experiment", 11, 22, 23, source_density_version="v4", **kw)
        self.assertEqual(new["root"], "/experiment/queues/after050_inputs020_v4")
        self.assertEqual(new["dependencies"][0], old["dependencies"][0])
        self.assertEqual([x["name"] for x in new["dependencies"]][1:],
            ["train_density_preparation_v4", "clean_density_preparation_v4"])
        for i in range(16):
            if i not in (4, 5):
                self.assertEqual(new["stages"][i], old["stages"][i])
        density = new["stages"][4]["command"]
        self.assertEqual(density[density.index("--source-density-version") + 1], "v4")
        self.assertIn("/experiment/paired_source_density_full24_v4/data", density)
        self.assertIn("/experiment/paired_density_clean_eval_v4/data", density)
        self.assertEqual(density[density.index("--budget") + 1], "20")

    def test_v4_prepares_only_after_real_v4_handles_and_stopped_v3_parent(self):
        for live, started in ((True, False), (False, True), (False, False)):
            with self.subTest(live=live, started=started), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                kw = fixture(root, source_density_version="v4")
                previous = tail.build_config(root, 99999991, 99999982, 99999983,
                    s3_comparison=kw["s3_comparison"])
                state = dict(config=previous, pid=99999984, status="waiting_for_dependency", active_child=None,
                    stages=[dict(name="prepare_coarse256", status="running" if started else "queued")])
                dump(Path(previous["root"]) / "config.json", previous)
                dump(Path(previous["root"]) / "queue_state.json", state)
                before = (Path(previous["root"]) / "config.json").read_bytes()
                with patch.object(tail, "dependency_live", side_effect=lambda pid, path: pid == 99999984 and live):
                    if live or started:
                        with self.assertRaises(ValueError):
                            tail.prepare(**kw)
                    else:
                        with contextlib.redirect_stdout(io.StringIO()):
                            tail.prepare(**kw)
                        receipt = tail.read(root / "queues/after050_inputs020_v4/preparation.json")
                        self.assertEqual(receipt["paired_density_version"], "v4")
                        self.assertTrue(receipt["v3_parent_retirement_observation"]["no_child_work_started"])
                        self.assertEqual(receipt["formal_training_runs"], 9)
                        self.assertFalse(receipt["future_child_queues_prepared"])
                self.assertEqual((Path(previous["root"]) / "config.json").read_bytes(), before)

    def test_v4_refuses_wrong_producer_and_missing_outer_certificate(self):
        for mistake in ("producer", "certificate"):
            with self.subTest(mistake=mistake), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                kw = fixture(root, source_density_version="v4")
                if mistake == "producer":
                    directory = root / "queues/clean_density_preparation_v4"
                    state = tail.read(directory / "queue_state.json")
                    state["config"]["stages"][0]["command"][2] = tail.MODULE_ROOT + "prepare_clean_density_v3"
                    dump(directory / "config.json", state["config"])
                    dump(directory / "queue_state.json", state)
                else:
                    (tail.locations(root, "v4")["density_source"] /
                        "staging/pairwise_v0_2/pairwise_data/rachel_density_outer_certificate.py").unlink()
                with self.assertRaises(ValueError):
                    tail.prepare(**kw)
                self.assertFalse(tail.locations(root, "v4")["queue"].exists())

    def test_sixteen_ordered_steps_fixed_design_and_owned_children(self):
        config = tail.build_config("/experiment", 11, 12, 13,
            s3_comparison="/experiment/s3/candidate_dual/paired_freeze.json")
        self.assertEqual(config["root"], "/experiment/queues/after050_inputs020")
        self.assertEqual(config["source"], "/experiment/source")
        self.assertEqual([d["pid"] for d in config["dependencies"]], [11, 12, 13])
        self.assertEqual([d["name"] for d in config["dependencies"]],
            ["after020_s3_to050", "train_density_preparation_v3", "clean_density_preparation_v3"])
        self.assertEqual([s["name"] for s in config["stages"]], [prefix + axis
            for axis in ("coarse256", "coarse512", "density_v3", "single7", "single16", "single32", "single64", "post")
            for prefix in ("prepare_", "run_")])
        for index, stage in enumerate(config["stages"]):
            if index % 2:
                self.assertEqual(stage["command"][2], tail.MODULE_ROOT + "run_recall_benchmark_queue")
                self.assertEqual(stage["command"][-1], stage["marker"])
                self.assertEqual(stage["resume_arguments"], ["--resume"])
                self.assertEqual(stage["completion_statuses"], ["complete"])
            else:
                cmd = stage["command"]
                source = "/experiment/density_runtime_source" if index == 4 else "/experiment/input_source"
                self.assertEqual(cmd[:4], ["/usr/bin/env", "-C", source, "PYTHONPATH=" + source])
                self.assertEqual(cmd[cmd.index("--architecture") + 1], "candidate_dual")
                self.assertEqual(cmd[cmd.index("--budget") + 1], "20")
                self.assertNotIn("resume_arguments", stage)
                self.assertNotIn("extra_pythonpath", stage)
                self.assertEqual(stage["completion_statuses"], ["prepared_not_launched"])

    def test_actual_preparer_and_train_eval_parsers_72_child_steps_nine_formal_runs(self):
        from experiments.rachel_n512_formal_30k import prepare_score_input_queue as inputs
        from experiments.rachel_n512_formal_30k import prepare_score_density_queue as density
        from experiments.rachel_n512_formal_30k import train_score_input_variant as ti, evaluate_score_input_variant as ei
        from experiments.rachel_n512_formal_30k import train_score_density as td, evaluate_score_density as ed
        parent = tail.build_config("/experiment", 11, 12, 13,
            s3_comparison="/experiment/s3/candidate_dual/paired_freeze.json")
        counts = dict(steps=0, smoke=0, formal=0, evaluation=0)
        for axis_index, stage in enumerate(parent["stages"][::2]):
            cmd = stage["command"]
            is_density = axis_index == 2
            preparer = density if is_density else inputs
            args = preparer.parser().parse_args(cmd[7:])
            if is_density:
                child = density.build_config(args.root, args.architecture, args.budget, args.run_name,
                    train_root=args.train_root, clean_root=args.clean_root, source_density_version=args.source_density_version)
                self.assertEqual(args.source_density_version, "v3")
                self.assertEqual(child["source"], "/experiment/density_runtime_source")
                self.assertEqual(len(child["stages"]), 16)
            else:
                spec = inputs.input_spec(args)
                self.assertEqual(len(spec.changed_axes()), 1)
                child = inputs.build_config(args.root, args.architecture, args.budget, spec, args.run_name)
                self.assertEqual(child["source"], "/experiment/input_source")
                self.assertEqual(len(child["stages"]), 8)
                if args.single_window is not None:
                    self.assertEqual(spec.coarse_size, 128)
                    self.assertEqual(spec.transport_fusion, "early")
                if spec.transport_fusion == "post":
                    self.assertEqual(spec.coarse_size, 128)
                    self.assertEqual(tuple(spec.window_sizes_px), (7, 16, 32, 64))
            self.assertEqual(parent["stages"][2 * axis_index + 1]["command"][-1], child["root"] + "/config.json")
            counts["steps"] += len(child["stages"])
            for child_stage in child["stages"]:
                argv = child_stage["command"]
                training = ".train_score_" in argv[2]
                selected = (td if is_density else ti) if training else (ed if is_density else ei)
                parsed = selected.parser().parse_args(argv[3:])
                if training:
                    self.assertEqual(parsed.architecture, "candidate_dual")
                    self.assertEqual(parsed.stop_after_epoch, 20)
                    self.assertEqual(selected.MAX_EPOCHS, 50)  # Fixed timeline, not a caller-adjustable CLI flag.
                    self.assertFalse(parsed.resume)
                    if parsed.smoke:
                        self.assertEqual(parsed.smoke, 32)
                        counts["smoke"] += 1
                    else:
                        counts["formal"] += 1
                        self.assertEqual(child_stage["completion_statuses"], ["budget_complete"])
                        self.assertEqual(child_stage["resume_arguments"], ["--resume"])
                else:
                    counts["evaluation"] += 1
                    self.assertEqual(parsed.budget, 20)
                    self.assertNotIn("resume_arguments", child_stage)
                    if not is_density:
                        self.assertIn("/candidate_dual/evaluation/budget_020/", parsed.baseline_evaluation)
                    elif "density1024" in child_stage["name"]:
                        self.assertIn("/n512/evaluation/budget_020/", parsed.baseline_evaluation)
        self.assertEqual(counts, dict(steps=72, smoke=9, formal=9, evaluation=54))

    def test_prepare_has_no_future_prerequisites_or_process_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            kw = fixture(root)
            with patch.object(subprocess, "Popen", side_effect=AssertionError("no launch")):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = tail.prepare(**kw)
            self.assertEqual(result["status"], "prepared_not_launched")
            receipt = tail.read(root / "queues/after050_inputs020/preparation.json")
            for field in ("axes_accumulated", "smoke_replaces_formal_training", "held_out_metrics_used_to_choose_next_action",
                          "future_child_queues_prepared", "future_freezes_prepared", "launches_processes"):
                self.assertFalse(receipt[field])
            self.assertEqual(receipt["formal_exposures_per_run"], 480000)
            self.assertEqual(receipt["formal_training_runs"], 9)
            self.assertFalse(kw["s3_comparison"].exists())
            self.assertFalse(tail.locations(root)["train_root"].exists())
            self.assertEqual([p for p in tail.future_outputs(root) if p.exists()], [tail.locations(root)["queue"]])
            with self.assertRaises(FileExistsError):
                tail.prepare(**kw)

    def test_live_dependency_required_unless_complete_and_all_handles_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            kw = fixture(root, status="running")
            with patch.object(tail, "dependency_live", return_value=False):
                with self.assertRaisesRegex(ValueError, "process is stopped"):
                    tail.prepare(**kw)
            with patch.object(tail, "dependency_live", return_value=True) as live:
                with contextlib.redirect_stdout(io.StringIO()):
                    tail.prepare(**kw)
                self.assertEqual([call.args[0] for call in live.call_args_list], [99999991, 99999992, 99999993])

    def test_dependency_identity_version_subset_and_output_collisions_reject(self):
        for problem in ("pid", "failed", "wrong_config", "wrong_v2", "subset", "wrong_tail", "missing_source", "occupied"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                kw = fixture(root)
                directory = root / "queues/train_density_preparation_v3"
                state = tail.read(directory / "queue_state.json")
                if problem == "pid":
                    state["pid"] += 1
                elif problem == "failed":
                    state["status"] = "failed"
                elif problem == "wrong_config":
                    state["config"]["source"] = "/wrong/source"
                elif problem in ("wrong_v2", "subset"):
                    if problem == "wrong_v2":
                        state["config"]["stages"][0]["command"][2] = tail.MODULE_ROOT + "materialize_source_density_v2"
                    else:
                        state["config"]["stages"][0]["command"] += ["--stop-after-pairs", "12"]
                    dump(directory / "config.json", state["config"])
                elif problem == "wrong_tail":
                    other = root / "queues/after020_s3_to050"
                    wrong = tail.read(other / "queue_state.json")
                    wrong["config"]["stages"][0]["command"][-1] = "candidate_pair"
                    dump(other / "config.json", wrong["config"])
                    dump(other / "queue_state.json", wrong)
                elif problem == "missing_source":
                    (root / "input_source/experiments/rachel_n512_formal_30k/prepare_score_input_queue.py").unlink()
                else:
                    (root / "density_variants/dual_joint020_density_v3").mkdir(parents=True)
                dump(directory / "queue_state.json", state)
                with self.assertRaises((ValueError, FileExistsError)):
                    tail.prepare(**kw)
                self.assertFalse((root / "queues/after050_inputs020").exists())

    def test_invalid_explicit_identity_and_s3_path(self):
        for root, pids, comparison in (("relative", (1, 2, 3), "/x"),
                ("/experiment", (True, 2, 3), "/experiment/s3/candidate_dual/paired_freeze.json"),
                ("/experiment", (1, 0, 3), "/experiment/s3/candidate_dual/paired_freeze.json"),
                ("/experiment", (1, 2, 3), "/experiment/s3/candidate_pair/paired_freeze.json")):
            with self.assertRaises(ValueError):
                tail.build_config(root, *pids, s3_comparison=comparison)

    def test_actual_cli_prepares_only_sixteen_stage_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            kw = fixture(root)
            cmd = [sys.executable, "-m", tail.MODULE_ROOT + "prepare_score_after050_inputs", "--root", str(root),
                "--after020-queue-pid", "99999991", "--train-density-queue-pid", "99999992",
                "--clean-density-queue-pid", "99999993", "--s3-comparison", str(kw["s3_comparison"])]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
            config = tail.read(json.loads(result.stdout)["config"])
            self.assertEqual(len(config["stages"]), 16)
            self.assertFalse((Path(config["root"]) / "queue_state.json").exists())
            self.assertFalse((root / "queues/input_dual_joint020_coarse256").exists())


if __name__ == "__main__":
    unittest.main()
