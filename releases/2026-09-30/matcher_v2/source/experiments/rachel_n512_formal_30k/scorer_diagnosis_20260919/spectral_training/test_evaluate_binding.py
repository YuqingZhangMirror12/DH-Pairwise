"""CPU-only regression for the real spectral endpoint PredictionReuse contract."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from experiments.rachel_n512_formal_30k import decoupled_prediction_reuse as reuse
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_training import evaluate as ev


def protocol(runtime):
    value = {key: None for key in reuse.PROTOCOL_FIELDS}
    value.update(status="complete", schema_version=reuse.EVALUATION_SCHEMA, split="ood",
        sample_count=301, batch_size=1, precision="fp32", prepared_cache="/synthetic/prepared",
        inference_runtime=runtime, prediction_reuse_schema=reuse.SCHEMA,
        input_digest_schema=reuse.INPUT_DIGEST_SCHEMA, model={key: None for key in reuse.MODEL_FIELDS})
    return value


class EvaluationBindingTests(unittest.TestCase):
    def test_changed_summary_bundle_cache_or_identity_misses_actual_reuse_comparator(self):
        record = dict(sha256="a" * 64, identity={"split": "ood", "source": "pinned"})
        with patch.object(ev.original, "inference_runtime", side_effect=lambda _: {"source_sha256": {}}):
            baseline = ev.inference_runtime(torch.device("cpu"), "b" * 64, record)
            variants = [("c" * 64, record), ("b" * 64, dict(record, sha256="d" * 64)),
                        ("b" * 64, dict(record, identity={"split": "ood", "source": "changed"}))]
            self.assertTrue(reuse._compatible(protocol(baseline), protocol(deepcopy(baseline))))
            for bundle_sha, changed in variants:
                runtime = ev.inference_runtime(torch.device("cpu"), bundle_sha, changed)
                self.assertFalse(reuse._compatible(protocol(baseline), protocol(runtime)))

    def test_all_spectral_implementation_changes_miss_actual_reuse_comparator(self):
        record = dict(sha256="a" * 64, identity={"split": "ood"})
        with patch.object(ev.original, "inference_runtime", side_effect=lambda _: {"source_sha256": {}}):
            baseline = ev.inference_runtime(torch.device("cpu"), "b" * 64, record)
            sha = ev.train.old._sha256
            for target in (*ev.training_implementation_sha256(), ev.__file__):
                def changed(path):
                    return "c" * 64 if Path(path).resolve() == Path(target).resolve() else sha(path)
                with patch.object(ev.train.old, "_sha256", side_effect=changed):
                    runtime = ev.inference_runtime(torch.device("cpu"), "b" * 64, record)
                self.assertFalse(reuse._compatible(protocol(baseline), protocol(runtime)))

    def test_endpoint_rejects_changed_training_source_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = dict(implementation_sha256=ev.training_implementation_sha256())
            ev.train.old.save_json(root / "classifier_freezes/freeze.json", dict(
                schema_version=ev.train.SCHEMA, status="complete", budget_epochs=28,
                eligible_epoch_range=[21, 28], held_out_used_for_fit=False,
                spectral_identity=identity, spectral_identity_sha256=ev.train.f.digest_json(identity)))
            ev.train.old.save_json(root / "status.json", dict(status="complete", completed_segments=112))
            sha = ev.train.old._sha256
            for target in identity["implementation_sha256"]:
                def changed(path):
                    return "d" * 64 if Path(path).resolve() == Path(target).resolve() else sha(path)
                with patch.object(ev.train.old, "_sha256", side_effect=changed), \
                     patch.object(ev.torch, "load", side_effect=AssertionError("must reject before tensor reads")):
                    with self.assertRaisesRegex(ValueError, "implementation changed"):
                        ev.load_frozen_model(root, "fixed_epoch")

    def test_real_run_passes_summary_bound_runtime_to_private_evaluator(self):
        record = dict(sha256="a" * 64, identity={"split": "ood"})
        bundle = dict(splits={"ood": record})
        caches = {"ood": SimpleNamespace(records=[None] * 301)}
        args = SimpleNamespace(summary_bundle="synthetic", summary_bundle_sha="b" * 64,
                               split="ood", device="cuda:0")
        captured = {}
        def private(function, **overrides):
            captured.update(overrides)
            self.assertIs(function, ev.original.run)
            return lambda arguments: "unit-complete"
        public_runtime = ev.original.run.__globals__["inference_runtime"]
        with patch.object(ev.cache, "load_bundle", return_value=(bundle, caches)), \
             patch.object(ev.train.cont, "private_function", side_effect=private), \
             patch.object(ev.train.shared, "gpu_lock", return_value=nullcontext()), \
             patch.object(ev.original, "inference_runtime", side_effect=lambda _: {"source_sha256": {}}):
            self.assertEqual(ev.run(args), "unit-complete")
            runtime = captured["inference_runtime"](torch.device("cpu"))
        self.assertEqual(runtime["spectral_summary_binding"]["bundle_sha256"], args.summary_bundle_sha)
        self.assertEqual(runtime["spectral_summary_binding"]["cache_sha256"], record["sha256"])
        self.assertIs(ev.original.run.__globals__["inference_runtime"], public_runtime)
        self.assertIs(captured["core"].predict_batch, ev.bind_and_predict)


if __name__ == "__main__":
    unittest.main()
