"""CPU synthetic entrypoint smoke; no network, real checkpoints or GPU runs."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local import train, evaluate, queue, model as local
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local.test_candidate_local import source_checkpoint
from experiments.rachel_n512_formal_30k.test_train_score_decoupled import tiny_loader
from experiments.rachel_n512_formal_30k.test_train_score_staged import tensor_tree_equal


def fixture():
    source = source_checkpoint()
    net = train.old.load_decoupled_checkpoint(source)
    source["resume_identity"]["populations"]["train"]["manifest"] = "/synthetic/pairs/train.jsonl"
    source["resume_identity"]["populations"]["val"]["manifest"] = "/synthetic/pairs/val.jsonl"
    source["matcher_pretraining_receipt"] = train.old.matcher_receipt(net, source["resume_identity"])
    source["runtime_batching"] = train.old.prepare_runtime_batching(
        SimpleNamespace(physical_microbatch=16), source["resume_identity"], 80,
        source_checkpoint_sha256=train.SOURCE_SHA)
    return source


def context(source):
    return patch.object(train, "MATCHER_SHA", source["matcher_pretraining_receipt"]["base_state_sha256"])


def ident(source, arm="c1"):
    return train.make_identity(arm, source, Path("synthetic_epoch020.pt"), source["resume_identity"]["populations"])


class EntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_source_sha_and_strict_s6_source_guard(self):
        source = fixture()
        with tempfile.TemporaryDirectory() as directory, context(source):
            path = Path(directory) / "epoch_020.pt"
            torch.save(source, path)
            with self.assertRaisesRegex(ValueError, "pinned"):
                train.read_source(path)
            with patch.object(train.old, "_sha256", return_value=train.SOURCE_SHA), \
                    patch.object(train.cont, "validate_source", return_value=None) as validation:
                loaded = train.read_source(path)
                validation.assert_called_once()
                self.assertEqual(validation.call_args.args[1], "s6_d2")
                self.assertEqual(loaded["epoch"], 20)
            with patch.object(train.old, "_sha256", return_value=train.SOURCE_SHA):
                # Synthetic cap16/source manifest is not the registered S6:
                # even a matching hash stub cannot bypass configuration checks.
                with self.assertRaises(ValueError):
                    train.read_source(path)

    def test_payload_exact_optimizer_rng_roundtrip_and_not_old_schema(self):
        source = fixture()
        with context(source):
            net = local.build_from_s6_epoch20(source)
            optimizer, _ = local.restore_source_optimizer(source, net)
            identity = ident(source)
            saved = deepcopy(train.payload(net, optimizer, source, identity, 80, {}))
            rng = train.old.capture_rng_state()
            loaded = train.load_model(saved)
            self.assertTrue(torch.equal(rng["torch"], torch.get_rng_state()))
            self.assertTrue(tensor_tree_equal(net.state_dict(), loaded.state_dict()))
            self.assertNotIn("decoupled_training_schema", saved)
            clone, _ = local.restore_source_optimizer(source, loaded)
            clone.load_state_dict(saved["optimizer_state_dict"])
            self.assertTrue(tensor_tree_equal(clone.state_dict(), optimizer.state_dict()))
            train.restore_rng(saved, torch.device("cpu"))
            got = train.old.capture_rng_state()
            self.assertEqual(saved["rng_state"]["python"], got["python"])
            self.assertTrue(np.array_equal(saved["rng_state"]["numpy"][1], got["numpy"][1]))
            self.assertTrue(torch.equal(saved["rng_state"]["torch"], got["torch"]))
            broken = deepcopy(saved)
            broken["optimizer_state_dict"]["param_groups"].pop()
            with self.assertRaises(ValueError):
                train.check_optimizer(broken, net)
            broken = deepcopy(saved)
            broken["additional_pair_exposures"] += 1
            with self.assertRaisesRegex(ValueError, "ledger"):
                train.validate_payload(broken, identity)

    def test_full_training_entry_cpu_disposable_smoke32_both_arms(self):
        source = fixture()
        with tempfile.TemporaryDirectory() as directory, context(source):
            root = Path(directory)
            (root / "immutable_source").mkdir()
            path = root / "immutable_source/source.pt"
            torch.save(source, path)
            results = []
            for arm in train.MODES:
                args = SimpleNamespace(arm=arm, source=str(path), output=str(root / arm), resume=False, smoke=32)
                populations = (object(), object(), 16, source["resume_identity"]["populations"])
                loader = tiny_loader(32, 16)
                with patch.object(train, "read_source", return_value=deepcopy(source)), \
                        patch.object(train.old, "make_populations", return_value=populations), \
                        patch.object(train.old, "make_weathering_loader", return_value=loader), \
                        patch.object(train.old, "evaluate_pair_validation", side_effect=AssertionError("smoke must not evaluate")):
                    result = train.run_locked(args, device=torch.device("cpu"))
                self.assertEqual(result["status"], "smoke_complete")
                self.assertEqual(result["optimizer_updates"], 2)
                self.assertFalse(list((root / arm).glob("*.pt")))
                self.assertEqual(result["training"]["candidate_decoder_cost"]["pairs"], 32)
                self.assertEqual(result["training"]["candidate_decoder_cost"]["calls"], 2)
                self.assertGreaterEqual(result["training"]["candidate_decoder_cost"]["elapsed_s"], 0)
                steps = json.loads((root / arm / "first_updates.json").read_text())["updates"]
                self.assertIn(12002, steps[1]["head_adam_steps"])
                self.assertIn(2, steps[1]["head_adam_steps"])
                results.append(result)
            self.assertEqual(results[0]["candidate_identity"]["populations"], results[1]["candidate_identity"]["populations"])

    def test_shared_gpu_lock_is_nonblocking_and_no_gpu_probe_if_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.lock"
            with path.open("a+") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(train.subprocess, "check_output") as probe:
                    with self.assertRaises(BlockingIOError):
                        with train.gpu_lock(path):
                            self.fail("cannot enter occupied lock")
                    probe.assert_not_called()
            with patch.object(train.subprocess, "check_output", return_value="999999\n"):
                with self.assertRaisesRegex(RuntimeError, "occupied"):
                    with train.gpu_lock(path):
                        self.fail("cannot enter occupied GPU")

    def test_evaluation_loader_requires_complete_own_epoch_freeze(self):
        source = fixture()
        with tempfile.TemporaryDirectory() as directory, context(source):
            root = Path(directory)
            identity = ident(source)
            net = local.build_from_s6_epoch20(source)
            optimizer, _ = local.restore_source_optimizer(source, net)
            for p in net.score_head.global_head.parameters():
                if p in optimizer.state:
                    optimizer.state[p]["step"].fill_(24000)
            saved = train.payload(net, optimizer, source, identity, 112, {})
            train.old.runner._atomic_torch_save(root / "epoch_028.pt", saved)
            record = dict(selected_epoch=28, checkpoint=str(root / "epoch_028.pt"),
                test_or_real_or_ood_used_for_fit=False, classifier_thresholds={"fused": .5}, operating_points={})
            winners = {key: dict(record) for key in ("fixed_epoch", "max_f1", "recall95")}
            train.publish_freeze(root, identity, winners)
            train.old.save_json(root / "status.json", dict(status="running", completed_segments=108))
            with self.assertRaisesRegex(ValueError, "completed"):
                evaluate.load_frozen_model(root, "fixed_epoch")
            train.old.save_json(root / "status.json", dict(status="complete", completed_segments=112))
            loaded, receipt = evaluate.load_frozen_model(root, "fixed_epoch")
            self.assertTrue(tensor_tree_equal(loaded.state_dict(), net.state_dict()))
            self.assertEqual(receipt["epoch"], 28)
            self.assertTrue(receipt["endpoint_only"])
            self.assertIs(evaluate._run.__code__, evaluate.original.run.__code__)
            self.assertIs(evaluate.original.run.__globals__["load_frozen_model"], evaluate.original.load_frozen_model)
            winners["max_f1"]["selected_epoch"] = 20
            with self.assertRaises(ValueError):
                train.publish_freeze(root, identity, winners)

    def test_finite_queue_two_arms_no_extra_training_or_live_c0_outputs(self):
        rows = queue.stage_plan(Path("/tmp/unit_candidate_only"), "python")
        self.assertEqual(len(rows), 20)
        self.assertEqual([r["name"] for r in rows if r["kind"] == "training"], ["c1_C9_C16", "c2_C9_C16"])
        self.assertTrue(all("candidate_local." in r["command"][2] for r in rows))
        self.assertEqual(sum(r["kind"] == "endpoint_evaluation" for r in rows), 18)
        self.assertFalse(any("--resume" in r["command"] for r in rows))
        self.assertEqual(sum(s["count"] for s in train.cont.plan()), 192000)
        self.assertEqual(train.cont.plan()[-1]["epoch"], 28)


if __name__ == "__main__":
    unittest.main()
