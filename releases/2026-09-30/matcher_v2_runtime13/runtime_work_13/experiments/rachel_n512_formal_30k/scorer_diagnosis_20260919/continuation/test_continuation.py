"""CPU-only unittest checks; no formal data, network or CUDA required."""
from copy import deepcopy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation import continue_classifier as cont
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation import evaluate_continuation as ev
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation import run_continuation_queue as queue
from experiments.rachel_n512_formal_30k.test_train_score_decoupled import model, identity, tiny_loader
from experiments.rachel_n512_formal_30k.test_train_score_staged import tensor_tree_equal


def fixture():
    net = model("cross_attention").set_phase("classifier")
    origin = identity(net)
    optimizer = cont.old.create_optimizer(net)
    for name, parameter in net.base_model.named_parameters():
        if not name.startswith(("coarse.", "local_head.", "fusion.")):
            parameter.grad = torch.ones_like(parameter)
    # Real Adam moments for all used head parameters, no state for unused fallback.
    for name, parameter in net.score_head.named_parameters():
        if name != "no_evidence_logit":
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for item in optimizer.state.values():
        item["step"].fill_(12000)
    for parameter in net.base_model.parameters():
        if parameter in optimizer.state:
            optimizer.state[parameter]["step"].fill_(18000)
    for group in optimizer.param_groups:
        group["lr"] = 2e-5
    receipt = cont.old.matcher_receipt(net, origin)
    runtime = cont.old.prepare_runtime_batching(SimpleNamespace(physical_microbatch=16), origin, 80,
        source_checkpoint_sha256="a" * 64)
    source = cont.old.checkpoint_payload(net, optimizer, identity=origin,
        loss_config=cont.old.RachelN512LossConfig(), completed=80, receipt=receipt,
        winners={}, role="unit-only", runtime_batching=runtime)
    ident = dict(schema_version=cont.SCHEMA, source_resume_identity_sha256=cont.old.canonical_digest(origin),
        restore_mode="exact", warm_start_seed=None)
    return net, optimizer, source, ident


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_schedule_and_public_functions_unchanged(self):
        rows = cont.plan()
        self.assertEqual((len(rows), rows[0]["number"], rows[-1]["number"]), (32, 81, 112))
        self.assertEqual(sum(r["count"] for r in rows), 192000)
        self.assertTrue(all(r["phase"] == "classifier" and r["learning_rate"] == 2e-5 for r in rows))
        self.assertIs(cont.train_segment.__code__, cont.old.train_segment.__code__)
        self.assertIsNot(cont.train_segment.__globals__, cont.old.train_segment.__globals__)
        self.assertEqual(cont.old.TOTAL_EPOCHS, 20)
        with self.assertRaises(ValueError):
            cont.old.phase_for_epoch(21)
        with self.assertRaises(ValueError):
            cont.phase_for_epoch(20)
        self.assertEqual(cont.phase_for_epoch(28), "classifier")
        self.assertIs(ev.run.__code__, ev.evaluator.run.__code__)
        self.assertIs(ev.evaluator.run.__globals__["load_frozen_model"], ev.evaluator.load_frozen_model)

    def test_actual_optimizer_required_except_explicit_unused_parameter(self):
        net, _, source, _ = fixture()
        cont.check_optimizer(source, net, expected_head_step=12000)
        for field in ("optimizer_state_dict", "rng_state"):
            corrupt = deepcopy(source)
            del corrupt[field]
            with self.assertRaises(ValueError):
                cont.check_optimizer(corrupt, net, expected_head_step=12000)
        corrupt = deepcopy(source)
        key = corrupt["optimizer_state_dict"]["param_groups"][1]["params"][1]
        del corrupt["optimizer_state_dict"]["state"][key]
        with self.assertRaisesRegex(ValueError, "missing active"):
            cont.check_optimizer(corrupt, net, expected_head_step=12000)
        corrupt = deepcopy(source)
        corrupt["optimizer_state_dict"]["state"][key]["step"].fill_(11999)
        with self.assertRaisesRegex(ValueError, "step differs"):
            cont.check_optimizer(corrupt, net, expected_head_step=12000)

    def test_restore_preserves_every_weight_optimizer_and_rng(self):
        net, _, source, ident = fixture()
        clone = model("cross_attention")
        opt = cont.old.create_optimizer(clone)
        self.assertEqual(cont.restore(clone, opt, source, ident, initial=True), 80)
        self.assertTrue(tensor_tree_equal(net.state_dict(), clone.state_dict()))
        self.assertTrue(tensor_tree_equal(source["optimizer_state_dict"], opt.state_dict()))
        restored_rng = cont.old.capture_rng_state()
        self.assertEqual(source["rng_state"]["python"], restored_rng["python"])
        self.assertTrue(torch.equal(source["rng_state"]["torch"], restored_rng["torch"]))
        self.assertTrue(np.array_equal(source["rng_state"]["numpy"][1], restored_rng["numpy"][1]))
        self.assertEqual(source["rng_state"]["numpy"][2:], restored_rng["numpy"][2:])
        self.assertFalse(clone.base_model.training)
        self.assertTrue(all(not p.requires_grad for p in clone.base_model.parameters()))

    def test_warm_start_is_explicit_and_records_reset(self):
        _, _, source, ident = fixture()
        source.pop("optimizer_state_dict")
        source.pop("rng_state")
        net = model("cross_attention")
        optimizer = cont.old.create_optimizer(net)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            cont.restore(net, optimizer, source, ident, initial=True)
        ident.update(restore_mode="warm-start", warm_start_seed=260919)
        self.assertEqual(cont.restore(net, optimizer, source, ident, initial=True), 80)
        self.assertFalse(optimizer.state)
        p = cont.payload(net, optimizer, source, ident, 80, {}, "unit")
        self.assertIn("warm-start", p["initialization"])
        self.assertEqual(p["continuation_pair_exposures"], 0)

    def test_new_payload_never_fakes_epoch20(self):
        net, optimizer, source, ident = fixture()
        for number in (80, 81, 84, 112):
            p = cont.payload(net, optimizer, source, ident, number, {}, "unit")
            self.assertEqual(cont.validate_continuation(p, ident), number)
            self.assertEqual(p["epoch"], (number + 3) // 4)
            self.assertEqual(p["continuation_optimizer_updates"], (number - 80) * 375)
            if number > 80:
                with self.assertRaises(ValueError):
                    cont.old.validate_checkpoint_progress(p, p["resume_identity"])
            corrupt = deepcopy(p)
            corrupt["optimizer_updates"] -= 1
            with self.assertRaisesRegex(ValueError, "ledger"):
                cont.validate_continuation(corrupt, ident)

    def test_exact_reused_training_loss_and_frozen_base(self):
        with tempfile.TemporaryDirectory() as directory:
            net, optimizer, source, ident = fixture()
            cont.restore(net, optimizer, source, ident, initial=True)
            before = cont.old.state_digest(net.base_model)
            args = SimpleNamespace(output=directory, microbatch=1, physical_microbatch=16,
                effective_batch=16, runtime_effective_batch=None, log_every=1000)
            report = cont.train_segment(net, tiny_loader(count=16, micro=16), optimizer,
                cont.old.RachelN512LossConfig(), torch.device("cpu"), args, 21)
            self.assertEqual((report["samples"], report["optimizer_updates"]), (16, 1))
            self.assertEqual(before, cont.old.state_digest(net.base_model))
            losses = report["loss_components"]
            self.assertEqual(losses["total"], losses["fused_pair_bce"])
            self.assertTrue(all(v == 0 for k, v in losses.items() if k not in ("total", "fused_pair_bce")))
            self.assertTrue(all(int(optimizer.state[p]["step"]) == 12001
                for p in net.score_head.parameters() if p in optimizer.state))
            self.assertTrue(all(int(optimizer.state[p]["step"]) == 18000
                for p in net.base_model.parameters() if p in optimizer.state))

    def test_resume_continues_same_next_update(self):
        with tempfile.TemporaryDirectory() as directory:
            net, optimizer, source, ident = fixture()
            anchor = cont.payload(net, optimizer, source, ident, 80, {}, "unit")
            clone = model("cross_attention")
            opt = cont.old.create_optimizer(clone)
            # Model an independent deserialization. Recent PyTorch may reuse
            # same-device moment tensors when load_state_dict sees live objects.
            cont.restore(clone, opt, deepcopy(anchor), ident)
            args = SimpleNamespace(output=directory, microbatch=1, physical_microbatch=16,
                effective_batch=16, runtime_effective_batch=None, log_every=1000)
            rng = deepcopy(anchor["rng_state"])
            for m, o in ((net, optimizer), (clone, opt)):
                cont.old.restore_rng_state(rng)
                m.set_phase("classifier").train()
                cont.train_segment(m, tiny_loader(16, 16), o, cont.old.RachelN512LossConfig(),
                                   torch.device("cpu"), args, 21)
            self.assertTrue(tensor_tree_equal(net.state_dict(), clone.state_dict()))
            self.assertTrue(tensor_tree_equal(optimizer.state_dict(), opt.state_dict()))

    def test_primary_only28_auxiliary_preserves_original_tie_rules(self):
        winners = {"max_f1": {"selection_key": [0.9], "selected_epoch": 20},
                   "recall95": {"selection_key": [0.8], "selected_epoch": 19}}
        report = {"decision_coverage": 1.}
        def record(root, epoch, r, p, selection):
            return {"selection_key": [0.9 if selection == "max_f1" else 0.8], "selected_epoch": epoch}
        with patch.object(cont.old, "winner_record", side_effect=record):
            result = cont.update_winners(winners, Path("new"), 21, report, {})
            self.assertEqual(result, winners)
            result = cont.update_winners(winners, Path("new"), 28, report, {})
            self.assertEqual(result["fixed_epoch"]["selected_epoch"], 28)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "before fixed C16"):
                cont.publish_freeze(Path(directory), {}, winners)
            with self.assertRaises(FileNotFoundError):
                ev.load_frozen_model(directory, "fixed_epoch")

    def test_endpoint_loader_accepts_honest28_and_historical_auxiliary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin, output = root / "old", root / "new"
            origin.mkdir()
            output.mkdir()
            net, optimizer, source, ident = fixture()
            ident["source_checkpoint"] = str(origin / "epoch_020.pt")
            torch.save(source, origin / "epoch_020.pt")
            p = cont.payload(net, optimizer, source, ident, 112, {}, "unit")
            torch.save(p, output / "epoch_028.pt")
            report = dict(decision_coverage=1., thresholds={}, selection_key=[.9])
            points = dict(thresholds={"max_f1": .5, "recall_95": .2}, selection_key=[.8])
            winners = {name: cont.old.winner_record(origin if name == "max_f1" else output,
                20 if name == "max_f1" else 28, report, points, name)
                for name in ("fixed_epoch", "max_f1", "recall95")}
            cont.publish_freeze(output, ident, winners)
            cont.old.save_json(output / "status.json", dict(status="running", completed_segments=108))
            with self.assertRaisesRegex(ValueError, "prohibited before"):
                ev.load_frozen_model(output, "fixed_epoch")
            cont.old.save_json(output / "status.json", dict(status="complete", completed_segments=112))
            for selection, expected_epoch in (("fixed_epoch", 28), ("max_f1", 20), ("recall95", 28)):
                loaded, receipt = ev.load_frozen_model(output, selection)
                self.assertEqual(receipt["budget"], 28)
                self.assertEqual(receipt["epoch"], expected_epoch)
                self.assertTrue(receipt["endpoint_only"])
                self.assertFalse(loaded.training)
                self.assertTrue(tensor_tree_equal(loaded.state_dict(), net.state_dict()))

    def test_data_identity_cannot_change_or_read_test_for_selection(self):
        _, _, source, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch_020.pt"
            torch.save(source, path)
            args = SimpleNamespace(arm="s7", restore_mode="exact")
            wrong = deepcopy(source["resume_identity"]["populations"])
            wrong["val"]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "SIMVAL identity differs"):
                cont.make_identity(args, source, path, wrong)

    def test_finite_two_arm_queue_and_endpoint_only_order(self):
        rows = queue.stage_plan(Path("/new"), "python")
        self.assertEqual(len(rows), 20)
        self.assertEqual([r["kind"] for r in rows].count("training"), 2)
        self.assertEqual(rows[0]["name"], "s6_d2_C9_C16")
        self.assertEqual(rows[10]["name"], "s7_C9_C16")
        self.assertTrue(all(r["kind"] == "endpoint_evaluation" for i,r in enumerate(rows) if i not in (0,10)))
        self.assertTrue(all("--keep-ids" in r["command"] for r in rows if r["name"].endswith("_real")))

    def test_first_update_hook_is_bounded_and_nonmutating(self):
        net, optimizer, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            hook = cont.install_first_update_receipt(optimizer, net, path)
            for i in range(3):
                optimizer.step()
            import json
            receipt = json.loads((path / "first_updates.json").read_text())
            self.assertEqual(len(receipt["updates"]), 2)
            self.assertTrue(receipt["updates"][0]["matcher_frozen"])
            self.assertEqual(receipt["updates"][0]["head_adam_steps"], [12000])
            hook.remove()


if __name__ == "__main__":
    unittest.main()
