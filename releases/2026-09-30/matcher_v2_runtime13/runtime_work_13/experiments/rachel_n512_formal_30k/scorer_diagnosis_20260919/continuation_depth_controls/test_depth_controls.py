"""CPU fixtures only: no formal data, checkpoint, network, CUDA or deployment."""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from experiments.rachel_n512_formal_30k import decoupled_prediction_reuse as reuse
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation_depth_controls import continue_depth_controls as ctrl
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation_depth_controls import evaluate_depth_controls as ev
from experiments.rachel_n512_formal_30k.test_train_score_decoupled import model, identity, tiny_loader
from experiments.rachel_n512_formal_30k.test_train_score_staged import tensor_tree_equal
from staging.pairwise_v0_2.models.rachel_decoupled_score import build_decoupled_score_model


@contextmanager
def fixture(arm="s4_d1"):
    """Synthetic tiny models explicitly replace registry pins only inside tests."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        origin_root = root / "original"
        origin_root.mkdir()
        net = build_decoupled_score_model(model("cross_attention").config, "cross_attention",
            model_options={"cross_attention_depth": ctrl.ARMS[arm][1]}).set_phase("classifier")
        origin = identity(net)
        origin.update(contour_cap=512, workers=4, model_options=net.model_options,
            matcher_checkpoint_sha256=ctrl.MATCHER_CHECKPOINT_SHA,
            populations={"train": {"manifest_sha256": ctrl.EXPECTED_TRAIN_MANIFEST_SHA[arm],
                                    "manifest": str(root / "unit_train.json")},
                         "val": {"manifest_sha256": ctrl.VAL_SHA,
                                  "manifest": str(root / "pairs/val.jsonl")}})
        optimizer = ctrl.old.create_optimizer(net)
        for name, parameter in net.score_head.named_parameters():
            if name != "no_evidence_logit":
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for state in optimizer.state.values():
            state["step"].fill_(12000)
        for group in optimizer.param_groups:
            group["lr"] = 2e-5
        receipt = ctrl.old.matcher_receipt(net, origin)
        runtime = ctrl.old.prepare_runtime_batching(SimpleNamespace(physical_microbatch=16),
            origin, 48, source_checkpoint_sha256="a" * 64)
        source = ctrl.old.checkpoint_payload(net, optimizer, identity=origin,
            loss_config=ctrl.old.RachelN512LossConfig(), completed=80, receipt=receipt,
            winners={}, role="unit-only", runtime_batching=runtime)
        source_path = origin_root / "epoch_020.pt"
        torch.save(source, source_path)
        expected = dict(checkpoint_sha256=ctrl.old._sha256(source_path),
            resume_identity_sha256=ctrl.old.canonical_digest(origin),
            checkpoint_bytes=source_path.stat().st_size)
        with patch.dict(ctrl.EXPECTED, {arm: expected}), patch.object(ctrl, "MATCHER_STATE_SHA", receipt["base_state_sha256"]):
            args = SimpleNamespace(arm=arm, restore_mode="exact")
            ident = ctrl.make_identity(args, source, source_path, origin["populations"])
            ctrl.old.save_json(origin_root / "status.json", dict(status="complete", epoch=20,
                completed_segments=80, global_exposure=480000, optimizer_updates=30000))
            ctrl.old.save_json(origin_root / "protocol.json", dict(runtime_batching=runtime))
            ctrl.old.save_json(origin_root / "classifier_freezes/freeze.json", dict(
                status="complete", resume_identity=origin, selections={"fixed_epoch": dict(
                    selected_epoch=20, checkpoint_sha256=expected["checkpoint_sha256"])}))
            yield SimpleNamespace(root=root, source_root=origin_root, path=source_path,
                net=net, optimizer=optimizer, source=source, ident=ident, arm=arm)


class DepthControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_budget_and_public_code_isolation(self):
        rows = ctrl.plan()
        self.assertEqual((len(rows), rows[0]["number"], rows[-1]["number"]), (32, 81, 112))
        self.assertEqual(sum(r["count"] for r in rows), 192000)
        self.assertEqual(sum(r["count"] // 16 for r in rows), 12000)
        self.assertTrue(all(r["learning_rate"] == 2e-5 and r["phase"] == "classifier" for r in rows))
        self.assertEqual(tuple(ctrl.base.ARMS), ("s6_d2", "s7"))
        self.assertNotEqual(ctrl.SCHEMA, ctrl.base.SCHEMA)
        self.assertIs(ctrl._run.__code__, ctrl.base.run.__code__)
        self.assertIsNot(ctrl._run.__globals__, ctrl.base.run.__globals__)
        self.assertIs(ctrl.base.run.__globals__["validate_source"], ctrl.base.validate_source)
        self.assertIs(ctrl._run.__globals__["payload"], ctrl.payload)
        self.assertIs(ctrl._run.__globals__["restore"], ctrl.restore)
        self.assertIs(ctrl._restore.__globals__["check_optimizer"], ctrl.check_optimizer)
        self.assertIs(ev._load_frozen_model.__globals__["train"], ctrl)
        self.assertIs(ev.base.load_frozen_model.__globals__["train"], ctrl.base)
        self.assertIs(ev._run.__globals__["inference_runtime"], ev.inference_runtime)
        self.assertIs(ev.base.evaluator.run.__globals__["inference_runtime"], ev.base.evaluator.inference_runtime)
        self.assertIs(ctrl.train_segment, ctrl.base.train_segment)

    def test_both_depths_source_and_complete_adam_state_pass(self):
        for arm, depth in (("s4_d1", 1), ("s6_d4", 4)):
            with self.subTest(arm=arm), fixture(arm) as f:
                loaded = ctrl.validate_source(f.source, arm)
                self.assertEqual(loaded.score_head.depth, depth)
                self.assertTrue(tensor_tree_equal(loaded.state_dict(), f.net.state_dict()))
                self.assertEqual(ctrl.check_optimizer(f.source, loaded, expected_head_step=12000)["head_step"], 12000)

    def test_source_depth_population_sampling_and_matcher_are_pinned(self):
        with fixture() as f:
            for field, value in (("sampling", "step3"), ("contour_cap", 2048),
                    ("model_options", {"cross_attention_depth": 4}),
                    ("matcher_checkpoint_sha256", "b" * 64)):
                corrupt = deepcopy(f.source)
                corrupt["resume_identity"][field] = value
                with self.assertRaisesRegex(ValueError, "pinned"):
                    ctrl.validate_source(corrupt, f.arm)
            corrupt = deepcopy(f.source)
            corrupt["matcher_pretraining_receipt"]["base_state_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "shared M12"):
                ctrl.validate_source(corrupt, f.arm)

    def test_inference_only_earlier_epoch_and_physical_change_rejected(self):
        with fixture() as f:
            corrupt = deepcopy(f.source)
            corrupt["inference_only"] = True
            with self.assertRaises(ValueError):
                ctrl.validate_source(corrupt, f.arm)
            corrupt = deepcopy(f.source)
            corrupt["completed_segments"] = 76
            with self.assertRaises(ValueError):
                ctrl.validate_source(corrupt, f.arm)
            corrupt = deepcopy(f.source)
            corrupt["runtime_batching"]["physical_microbatch"] = 8
            with self.assertRaises(ValueError):
                ctrl.validate_source(corrupt, f.arm)

    def test_no_missing_moments_rng_or_wrong_adam_recipe(self):
        with fixture("s6_d4") as f:
            for field in ("optimizer_state_dict", "rng_state"):
                corrupt = deepcopy(f.source)
                del corrupt[field]
                with self.assertRaises(ValueError):
                    ctrl.validate_source(corrupt, f.arm)
            for field, value in (("lr", 1e-4), ("weight_decay", 0), ("betas", (.8, .99)),
                                 ("eps", 1e-6), ("maximize", True)):
                corrupt = deepcopy(f.source)
                corrupt["optimizer_state_dict"]["param_groups"][1][field] = value
                with self.assertRaises(ValueError):
                    ctrl.check_optimizer(corrupt, f.net, expected_head_step=12000)
            key = next(iter(f.source["optimizer_state_dict"]["state"]))
            for value in (11999, 12000.5, float("nan")):
                corrupt = deepcopy(f.source)
                corrupt["optimizer_state_dict"]["state"][key]["step"].fill_(value)
                with self.assertRaises((ValueError, OverflowError)):
                    ctrl.check_optimizer(corrupt, f.net, expected_head_step=12000)
            corrupt = deepcopy(f.source)
            del corrupt["optimizer_state_dict"]["state"][key]
            with self.assertRaisesRegex(ValueError, "missing active"):
                ctrl.check_optimizer(corrupt, f.net, expected_head_step=12000)

    def test_source_file_hash_and_warm_start_rejected(self):
        with fixture() as f:
            args = SimpleNamespace(arm=f.arm, restore_mode="warm-start")
            with self.assertRaisesRegex(ValueError, "exact"):
                ctrl.make_identity(args, f.source, f.path, f.source["resume_identity"]["populations"])
            args.restore_mode = "exact"
            with patch.dict(ctrl.EXPECTED[f.arm], {"checkpoint_sha256": "b" * 64}):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    ctrl.make_identity(args, f.source, f.path, f.source["resume_identity"]["populations"])

    def test_exact_restore_and_one_cpu_update_for_both_depths(self):
        for arm in ctrl.ARMS:
            with self.subTest(arm=arm), fixture(arm) as f:
                net = ctrl.validate_source(f.source, arm)
                optimizer = ctrl.old.create_optimizer(net)
                ctrl.restore(net, optimizer, deepcopy(f.source), f.ident, initial=True)
                self.assertTrue(tensor_tree_equal(f.source["optimizer_state_dict"], optimizer.state_dict()))
                self.assertTrue(torch.equal(f.source["rng_state"]["torch"], ctrl.old.capture_rng_state()["torch"]))
                before = ctrl.old.state_digest(net.base_model)
                args = SimpleNamespace(output=str(f.root), microbatch=1, physical_microbatch=16,
                    effective_batch=16, runtime_effective_batch=None, log_every=1000)
                report = ctrl.train_segment(net, tiny_loader(16, 16), optimizer,
                    ctrl.old.RachelN512LossConfig(), torch.device("cpu"), args, 21)
                self.assertEqual((report["samples"], report["optimizer_updates"]), (16, 1))
                self.assertEqual(report["loss_components"]["total"], report["loss_components"]["fused_pair_bce"])
                self.assertEqual(before, ctrl.old.state_digest(net.base_model))
                self.assertFalse(net.base_model.training)
                self.assertTrue(all(not p.requires_grad for p in net.base_model.parameters()))
                self.assertTrue(all(int(s["step"]) == 12001 for s in optimizer.state.values()))

    def test_continuation_ledger_schema_runtime_and_arm_binding(self):
        with fixture() as f:
            for number in (80, 81, 84, 112):
                checkpoint = ctrl.payload(f.net, f.optimizer, f.source, f.ident, number, {}, "unit")
                self.assertEqual(ctrl.validate_continuation(checkpoint, f.ident), number)
                self.assertEqual(checkpoint["epoch"], (number + 3) // 4)
                with self.assertRaises(ValueError):
                    ctrl.base.validate_continuation(checkpoint, f.ident)
                corrupt = deepcopy(checkpoint)
                corrupt["continuation_pair_exposures"] += 1
                with self.assertRaisesRegex(ValueError, "ledger"):
                    ctrl.validate_continuation(corrupt, f.ident)
                corrupt = deepcopy(checkpoint)
                corrupt["runtime_batching"]["history"][0]["override_source"] = "changed"
                with self.assertRaisesRegex(ValueError, "runtime batching changed"):
                    ctrl.validate_continuation(corrupt, f.ident)
            wrong = deepcopy(f.ident)
            wrong["arm"] = "s6_d4"
            with self.assertRaises(ValueError):
                ctrl.validate_identity(wrong)

    def test_resumed_next_update_matches_uninterrupted_for_d4(self):
        with fixture("s6_d4") as f:
            checkpoint = ctrl.payload(f.net, f.optimizer, f.source, f.ident, 80, {}, "unit")
            clone = ctrl.validate_source(f.source, f.arm)
            optimizer = ctrl.old.create_optimizer(clone)
            ctrl.restore(clone, optimizer, deepcopy(checkpoint), f.ident)
            args = SimpleNamespace(output=str(f.root), microbatch=1, physical_microbatch=16,
                effective_batch=16, runtime_effective_batch=None, log_every=1000)
            for net, opt in ((f.net, f.optimizer), (clone, optimizer)):
                ctrl.old.restore_rng_state(checkpoint["rng_state"])
                net.set_phase("classifier").train()
                ctrl.train_segment(net, tiny_loader(16, 16), opt, ctrl.old.RachelN512LossConfig(),
                    torch.device("cpu"), args, 21)
            self.assertTrue(tensor_tree_equal(f.net.state_dict(), clone.state_dict()))
            self.assertTrue(tensor_tree_equal(f.optimizer.state_dict(), optimizer.state_dict()))

    def test_metadata_preflight_reads_no_tensors_and_does_not_execute(self):
        with fixture() as f, patch.object(ctrl.base.torch, "load", side_effect=AssertionError("no tensor reads")):
            args = ctrl.parser().parse_args(["--arm", f.arm, "--source-training", str(f.source_root)])
            with patch.object(ctrl, "_run", side_effect=AssertionError("no training")):
                result = ctrl.run(args)
            self.assertTrue(result["metadata_only"])
            self.assertFalse(result["tensor_payloads_read"])
            self.assertFalse(result["launches_training"])
            self.assertFalse(result["gpu_used"])
            self.assertEqual(result["optimizer_source"], str(f.path.resolve()) + ":optimizer_state_dict")

    def test_full_entry_cpu_discard32_smoke_for_both_depths(self):
        for arm in ctrl.ARMS:
            with self.subTest(arm=arm), fixture(arm) as f:
                output = f.root / "smoke32"
                args = ctrl.parser().parse_args(["--arm", arm, "--source-training", str(f.source_root),
                    "--output", str(output), "--execute", "--smoke", "32"])
                # Only the device facade and formal-data loader are replaced.
                # Reused runner, source checks, optimizer/RNG restore and actual
                # classification loss/backprop/update all run on tiny CPU data.
                cpu_torch = SimpleNamespace(set_num_threads=torch.set_num_threads,
                    load=torch.load, device=lambda _: torch.device("cpu"),
                    __version__=torch.__version__, version=SimpleNamespace(cuda=None),
                    cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                        reset_peak_memory_stats=lambda _: None))
                cpu_platform = SimpleNamespace(system=lambda: "Linux", python_version=lambda: "unit-cpu")
                records = f.source["resume_identity"]["populations"]
                with patch.dict(ctrl._run.__globals__, torch=cpu_torch, platform=cpu_platform), \
                     patch.object(ctrl.old, "make_populations", return_value=([], [], 16, records)), \
                     patch.object(ctrl.old, "make_weathering_loader", return_value=tiny_loader(32, 16)), \
                     patch.object(ctrl, "GPU_LOCK", f.root / "heatmap-gpu.lock"), \
                     patch.object(ctrl.subprocess, "check_output", return_value=""):
                    result = ctrl.run(args)
                self.assertEqual(result["status"], "smoke_complete")
                self.assertEqual((result["pair_exposures"], result["optimizer_updates"]), (32, 2))
                self.assertFalse(result["formal_training_counted"])
                self.assertTrue(result["weights_discarded"])
                self.assertFalse(list(output.glob("*.pt")))
                self.assertEqual(ctrl.old._sha256(f.path), f.ident["source_checkpoint_sha256"])
                updates = json.loads((output / "first_updates.json").read_text())["updates"]
                self.assertEqual([u["head_adam_steps"] for u in updates], [[12001], [12002]])

    def test_preflight_rejects_incomplete_or_changed_source_receipts(self):
        with fixture() as f:
            status = f.source_root / "status.json"
            data = json.loads(status.read_text())
            data["status"] = "running"
            ctrl.old.save_json(status, data)
            with self.assertRaisesRegex(ValueError, "completed C8"):
                ctrl.preflight(f.arm, f.source_root)

    def test_busy_shared_lock_exits_before_gpu_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heatmap-gpu.lock"
            with path.open("a+") as held, patch.object(ctrl, "GPU_LOCK", path):
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(ctrl.subprocess, "check_output", side_effect=AssertionError("lock first")):
                    with self.assertRaisesRegex(RuntimeError, "lock busy"):
                        with ctrl.gpu_lock():
                            self.fail("must never enter")

    def test_occupied_gpu_rejected_and_lock_released(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heatmap-gpu.lock"
            with patch.object(ctrl, "GPU_LOCK", path):
                with patch.object(ctrl.subprocess, "check_output", return_value="456\n"):
                    with self.assertRaisesRegex(RuntimeError, "compute processes"):
                        with ctrl.gpu_lock():
                            self.fail("must not enter")
                with patch.object(ctrl.subprocess, "check_output", return_value=""):
                    with ctrl.gpu_lock():
                        pass

    def test_endpoint_only_loader_accepts_both_depths_and_historical_auxiliary(self):
        for arm in ctrl.ARMS:
            with self.subTest(arm=arm), fixture(arm) as f:
                output = f.root / "continued"
                output.mkdir()
                checkpoint = ctrl.payload(f.net, f.optimizer, f.source, f.ident, 112, {}, "unit")
                torch.save(checkpoint, output / "epoch_028.pt")
                report = dict(decision_coverage=1., thresholds={}, selection_key=[.9])
                points = dict(thresholds={"max_f1": .5, "recall_95": .2}, selection_key=[.8])
                winners = {name: ctrl.old.winner_record(f.source_root if name == "max_f1" else output,
                    20 if name == "max_f1" else 28, report, points, name)
                    for name in ("fixed_epoch", "max_f1", "recall95")}
                ctrl.publish_freeze(output, f.ident, winners)
                ctrl.old.save_json(output / "status.json", dict(status="running", completed_segments=108))
                with self.assertRaisesRegex(ValueError, "prohibited before"):
                    ev.load_frozen_model(output, "fixed_epoch")
                ctrl.old.save_json(output / "status.json", dict(status="complete", completed_segments=112))
                for selection, epoch in (("fixed_epoch", 28), ("max_f1", 20), ("recall95", 28)):
                    net, receipt = ev.load_frozen_model(output, selection)
                    self.assertEqual(receipt["epoch"], epoch)
                    self.assertEqual(receipt["budget"], 28)
                    self.assertEqual(net.score_head.depth, ctrl.ARMS[arm][1])
                    self.assertTrue(receipt["endpoint_only"])
                    self.assertTrue(tensor_tree_equal(net.state_dict(), f.net.state_dict()))
                freeze_path = output / "classifier_freezes/freeze.json"
                freeze = json.loads(freeze_path.read_text())
                freeze["continuation_identity"]["restore_mode"] = "warm-start"
                ctrl.old.save_json(freeze_path, freeze)
                with self.assertRaisesRegex(ValueError, "contract differs"):
                    ev.load_frozen_model(output, "fixed_epoch")

    def test_changed_adapter_source_invalidates_prediction_reuse(self):
        def protocol(runtime):
            result = {key: None for key in reuse.PROTOCOL_FIELDS}
            result.update(status="complete", schema_version=reuse.EVALUATION_SCHEMA,
                split="real", sample_count=1, batch_size=1, precision="fp32",
                inference_runtime=runtime, prediction_reuse_schema=reuse.SCHEMA,
                input_digest_schema=reuse.INPUT_DIGEST_SCHEMA,
                prepared_cache="/synthetic/prepared",
                model={key: None for key in reuse.MODEL_FIELDS})
            return result
        with patch.object(ev.base.evaluator, "inference_runtime", side_effect=lambda _: {"source_sha256": {}}):
            baseline = ev.inference_runtime(torch.device("cpu"))
            sha = ctrl.old._sha256
            for target in (ctrl.__file__, ev.__file__, ev.base.__file__, ctrl.base.__file__, ctrl.old.__file__):
                def changed(path):
                    return "a" * 64 if Path(path).resolve() == Path(target).resolve() else sha(path)
                with patch.object(ctrl.old, "_sha256", side_effect=changed):
                    different = ev.inference_runtime(torch.device("cpu"))
                self.assertFalse(reuse._compatible(protocol(baseline), protocol(different)))
            self.assertTrue(reuse._compatible(protocol(baseline), protocol(deepcopy(baseline))))

    def test_endpoint_rejects_changed_training_source_before_tensor_load(self):
        with fixture() as f:
            root = f.root / "frozen"
            ctrl.old.save_json(root / "classifier_freezes/freeze.json", {"continuation_identity": f.ident})
            sha = ctrl.old._sha256
            for target in (ctrl.__file__, ctrl.base.__file__, ctrl.old.__file__):
                def changed(path):
                    return "b" * 64 if Path(path).resolve() == Path(target).resolve() else sha(path)
                with patch.object(ctrl.old, "_sha256", side_effect=changed), \
                     patch.object(ctrl.base.torch, "load", side_effect=AssertionError("no weight reads")):
                    with self.assertRaisesRegex(ValueError, "implementation changed"):
                        ev.load_frozen_model(root, "fixed_epoch")


if __name__ == "__main__":
    unittest.main()
