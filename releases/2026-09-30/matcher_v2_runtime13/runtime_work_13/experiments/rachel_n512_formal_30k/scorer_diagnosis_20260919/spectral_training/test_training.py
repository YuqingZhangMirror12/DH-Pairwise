"""Synthetic CPU cache/trainer/loader tests, never real-data/GPU experiments."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_training import cache, model, train, evaluate, queue
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local.test_entrypoints import fixture
from experiments.rachel_n512_formal_30k.test_train_score_decoupled import tiny_loader
from experiments.rachel_n512_formal_30k.test_train_score_staged import tensor_tree_equal


def make_bundle(root, source, source_path, count=32):
    splits = {}
    for split in ("train", "val"):
        # CPU fixture has same images/IDs for convenience, but independent cache
        # identities: no production split-overlap claim is made by this fixture.
        splits[split] = cache.prepare_split(root, split, tiny_loader(count, 1),
            source["resume_identity"]["populations"][split], source, source_path=source_path, expected_count=count)
    train_record = splits["train"]
    c = cache.f.load_cache(train_record["path"], train_record["identity"], train_record["sha256"])
    bundle = dict(schema_version=cache.SCHEMA, status="complete", source_checkpoint_sha256=train.shared.SOURCE_SHA,
        matcher_state_sha256=train.shared.MATCHER_SHA, splits=splits, normalizer=cache.f.fit_train_statistics(c))
    path = root / "bundle.json"
    train.old.save_json(path, bundle)
    return path, cache.f.file_sha256(path), bundle


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = fixture()
        # Add only fixture manifest checksums to the synthetic source protocol.
        for split in ("train", "val"):
            self.source["resume_identity"]["populations"][split]["manifest_sha256"] = ("1" if split == "train" else "2") * 64
        self.source["matcher_pretraining_receipt"]["matcher_contract_sha256"] = train.old.canonical_digest(
            train.old.matcher_contract(self.source["resume_identity"]))
        self.source["runtime_batching"] = train.old.prepare_runtime_batching(
            SimpleNamespace(physical_microbatch=16), self.source["resume_identity"], 80,
            source_checkpoint_sha256=train.shared.SOURCE_SHA)
        (self.root / "immutable_source").mkdir()
        self.source_path = self.root / "immutable_source/epoch_020.pt"
        torch.save(self.source, self.source_path)
        self.matcher_patch = patch.object(train.shared, "MATCHER_SHA", self.source["matcher_pretraining_receipt"]["base_state_sha256"])
        self.matcher_patch.start()
        self.count_patch = patch.dict(cache.SPLIT_COUNTS, train=32, val=32)
        self.count_patch.start()
        (self.root / "cache").mkdir()
        self.bundle_path, self.bundle_sha, self.bundle = make_bundle(self.root / "cache", self.source, self.source_path)
        _, self.caches = cache.load_bundle(self.bundle_path, self.bundle_sha)

    def tearDown(self):
        self.count_patch.stop()
        self.matcher_patch.stop()
        self.temp.cleanup()

    def args(self, variant, output):
        return SimpleNamespace(variant=variant, source=str(self.source_path), cache_bundle=str(self.bundle_path),
            cache_bundle_sha=self.bundle_sha, output=str(output), resume=False, smoke=32)

    def test_cache_full_input_binding_split_and_train_only_shared_normalizer(self):
        batch = next(iter(tiny_loader(32, 16))).batch
        hashes = cache.input_hashes(batch)
        self.assertEqual(len(hashes), 16)
        summary = self.caches["train"].batch(list(batch.pair_ids), hashes)
        self.assertEqual(summary["values"].shape, (16, 10))
        changed = replace(batch, mask_a=batch.mask_a.copy())
        changed.mask_a[0, 0, 0, 0] = 1. - changed.mask_a[0, 0, 0, 0]
        with self.assertRaises(ValueError):
            self.caches["train"].batch(list(batch.pair_ids), cache.input_hashes(changed))
        for c in (self.caches["val"],):
            with self.assertRaises(ValueError):
                cache.f.fit_train_statistics(c)
        broken = deepcopy(self.bundle)
        broken["normalizer"]["fit_split"] = "val"
        path = self.root / "broken.json"
        train.old.save_json(path, broken)
        with self.assertRaisesRegex(ValueError, "normalizer"):
            cache.load_bundle(path, cache.f.file_sha256(path))

    def test_three_variants_initial_exact_score_same193_and_no_runtime_svd(self):
        source_model = train.old.load_decoupled_checkpoint(self.source).eval()
        batch = next(iter(tiny_loader(32, 16))).batch
        inputs, _ = train.old.runner._full_batch(batch, torch.device("cpu"))
        with torch.no_grad():
            expected = source_model(*inputs)
        before = torch.get_rng_state().clone()
        residuals = []
        with patch.object(np.linalg, "svd", side_effect=AssertionError("no online SVD")):
            for variant in train.VARIANTS:
                m = model.build(self.source, self.bundle["normalizer"], variant).eval()
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                residuals.append(deepcopy(m.score_head.residual.state_dict()))
                self.assertEqual(sum(p.numel() for p in m.score_head.residual.parameters()), 193)
                m.bind_batch(batch, self.caches["train"])
                got = m(*inputs)
                self.assertTrue(torch.equal(got.fused_logit, expected.fused_logit))
                self.assertTrue(torch.equal(got.assignment, expected.assignment))
                self.assertEqual(m.binding_cost["svd_calls"], 0)
                with self.assertRaisesRegex(ValueError, "fresh"):
                    m(*inputs)
        self.assertTrue(all(tensor_tree_equal(residuals[0], value) for value in residuals))

    def test_real_cpu_smoke32_all_three_with_frozen_matcher_and_exact_source_optimizer(self):
        for variant in train.VARIANTS:
            args = self.args(variant, self.root / variant)
            with patch.object(train.shared, "read_source", return_value=deepcopy(self.source)), \
                    patch.object(cache, "populations", return_value=(object(), object(), 16, self.source["resume_identity"]["populations"])), \
                    patch.object(train.old, "make_weathering_loader", return_value=tiny_loader(32, 16)), \
                    patch.object(train.old, "evaluate_pair_validation", side_effect=AssertionError("smoke no VAL")), \
                    patch.object(np.linalg, "svd", side_effect=AssertionError("smoke no SVD")):
                result = train.run_locked(args, device=torch.device("cpu"))
            self.assertEqual(result["optimizer_updates"], 2)
            self.assertTrue(result["frozen_base_unchanged"])
            self.assertEqual(result["training"]["cached_summary_binding"]["pairs"], 32)
            self.assertFalse(list((self.root / variant).glob("*.pt")))
            steps = json.loads((self.root / variant / "first_updates.json").read_text())["updates"]
            self.assertEqual(steps[-1]["head_adam_steps"], [2, 12002])

    def test_model_optimizer_rng_schema_roundtrip_and_wrong_variant_rejected(self):
        args = self.args("mass_spectral", self.root / "run")
        identity = train.identity(args, self.source, self.bundle, self.source["resume_identity"]["populations"])
        m = model.build(self.source, self.bundle["normalizer"], "mass_spectral")
        optimizer = model.restore_optimizer(self.source, m)
        saved = deepcopy(train.payload(m, optimizer, self.source, identity, 80, {}))
        loaded = train.load_model(saved)
        # helper handles dict-valued PyTorch extra_state in scorer/normalizer.
        self.assertTrue(tensor_tree_equal(train.cpu_state(m), train.cpu_state(loaded)))
        opt = model.restore_optimizer(self.source, loaded)
        opt.load_state_dict(saved["optimizer_state_dict"])
        self.assertTrue(tensor_tree_equal(opt.state_dict(), optimizer.state_dict()))
        train.shared.restore_rng(saved, torch.device("cpu"))
        self.assertTrue(torch.equal(saved["rng_state"]["torch"], torch.get_rng_state()))
        broken = deepcopy(saved)
        broken["spectral_identity"]["variant"] = "mass"
        with self.assertRaises(ValueError):
            train.load_model(broken)

    def test_finite_queue30_and_no_svd_or_cache_generation_in_training_plan(self):
        registry = {s: dict(bundle=s + ".json", sha256="3" * 64) for s in ("test", "real", "ood")}
        rows = queue.stage_plan(Path("/tmp/spectral-only"), self.bundle_path, self.bundle_sha, registry, "python")
        self.assertEqual(len(rows), 30)
        self.assertEqual(sum(r["kind"] == "training" for r in rows), 3)
        self.assertTrue(all("spectral_training." in r["command"][2] for r in rows))
        self.assertFalse(any(r["command"][2].endswith("cache") for r in rows))
        self.assertEqual(sum(x["count"] for x in train.cont.plan()), 192000)

    def test_cpu_multiprocess_cache_records_equal_single_worker(self):
        items = list(cache.pair_items(tiny_loader(2, 1)))
        one = sorted(cache.bounded_compute(iter(items), 1, str(self.source_path)), key=lambda x: x[0])
        two = sorted(cache.bounded_compute(iter(items), 2, str(self.source_path)), key=lambda x: x[0])
        self.assertEqual([v[1] for v in one], [v[1] for v in two])

    def test_endpoint_gate_and_original_predict_function_kept(self):
        args = self.args("mass", self.root / "endpoint")
        identity = train.identity(args, self.source, self.bundle, self.source["resume_identity"]["populations"])
        m = model.build(self.source, self.bundle["normalizer"], "mass")
        opt = model.restore_optimizer(self.source, m)
        # Synthetic final state for loader testing only, not experiment output.
        for p in m.score_head.residual.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for p in m.score_head.ca.parameters():
            if p in opt.state:
                opt.state[p]["step"].fill_(24000)
        for p in m.score_head.residual.parameters():
            opt.state[p]["step"].fill_(12000)
        saved = train.payload(m, opt, self.source, identity, 112, {})
        root = Path(args.output)
        root.mkdir()
        path = root / "epoch_028.pt"
        train.old.runner._atomic_torch_save(path, saved)
        record = dict(selected_epoch=28, checkpoint=str(path), test_or_real_or_ood_used_for_fit=False,
            classifier_thresholds={"fused": .5}, operating_points={})
        winners = {s: deepcopy(record) for s in ("fixed_epoch", "max_f1", "recall95")}
        train.publish_freeze(root, identity, winners)
        train.old.save_json(root / "status.json", dict(status="running", completed_segments=108))
        with self.assertRaisesRegex(ValueError, "complete"):
            evaluate.load_frozen_model(root, "fixed_epoch")
        train.old.save_json(root / "status.json", dict(status="complete", completed_segments=112))
        loaded, receipt = evaluate.load_frozen_model(root, "fixed_epoch")
        self.assertEqual(receipt["epoch"], 28)
        self.assertTrue(tensor_tree_equal(train.cpu_state(m), train.cpu_state(loaded)))
        loaded.endpoint_cache = self.caches["val"]
        batch = next(iter(tiny_loader(1, 1))).batch
        with patch.object(evaluate.original.core, "predict_batch", return_value=["same original decoder path"]) as call:
            result = evaluate.bind_and_predict(loaded, batch, torch.device("cpu"))
        self.assertEqual(result, ["same original decoder path"])
        call.assert_called_once_with(loaded, batch, torch.device("cpu"))
        self.assertIsNotNone(loaded._pending_summary)


if __name__ == "__main__":
    unittest.main()
