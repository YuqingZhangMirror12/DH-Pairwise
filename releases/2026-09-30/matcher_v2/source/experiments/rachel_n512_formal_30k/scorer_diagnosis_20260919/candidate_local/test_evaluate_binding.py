"""CPU-only guards for candidate endpoint source identity and prediction reuse."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.rachel_n512_formal_30k import decoupled_prediction_reuse as reuse
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local import evaluate, train, model
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local.test_entrypoints import fixture, context, ident


def protocol(runtime):
    value = {key: None for key in reuse.PROTOCOL_FIELDS}
    value.update(status="complete", schema_version=reuse.EVALUATION_SCHEMA,
        split="real", sample_count=1, batch_size=1, precision="fp32",
        inference_runtime=runtime, prediction_reuse_schema=reuse.SCHEMA,
        input_digest_schema=reuse.INPUT_DIGEST_SCHEMA, prepared_cache="/synthetic/prepared",
        model={key: None for key in reuse.MODEL_FIELDS})
    return value


def endpoint(root, source):
    identity = ident(source)
    net = model.build_from_s6_epoch20(source)
    optimizer, _ = model.restore_source_optimizer(source, net)
    for parameter in net.score_head.global_head.parameters():
        if parameter in optimizer.state:
            optimizer.state[parameter]["step"].fill_(24000)
    saved = train.payload(net, optimizer, source, identity, 112, {})
    train.old.runner._atomic_torch_save(root / "epoch_028.pt", saved)
    record = dict(selected_epoch=28, checkpoint=str(root / "epoch_028.pt"),
        test_or_real_or_ood_used_for_fit=False, classifier_thresholds={"fused": .5}, operating_points={})
    train.publish_freeze(root, identity, {key: dict(record) for key in ("fixed_epoch", "max_f1", "recall95")})
    train.old.save_json(root / "status.json", dict(status="complete", completed_segments=112))


class EvaluationBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_private_runtime_binding_does_not_patch_original_evaluator(self):
        self.assertIs(evaluate._run.__code__, evaluate.original.run.__code__)
        self.assertIs(evaluate._run.__globals__["inference_runtime"], evaluate.inference_runtime)
        self.assertIs(evaluate._run.__globals__["load_frozen_model"], evaluate.load_frozen_model)
        self.assertIs(evaluate.original.run.__globals__["inference_runtime"], evaluate.original.inference_runtime)
        self.assertIs(evaluate.original.run.__globals__["load_frozen_model"], evaluate.original.load_frozen_model)

    def test_runtime_binds_candidate_and_decoder_files_and_changed_hash_misses_reuse(self):
        package_root = Path(evaluate.original.__file__).resolve().parents[2]
        required = (Path(model.__file__), Path(train.__file__), Path(evaluate.__file__),
                    Path(model.estimate_translation_layout.__code__.co_filename))
        # Leave the authoritative public helper intact; isolate only its
        # baseline payload so the test measures the added source bindings.
        with patch.object(evaluate.original, "inference_runtime", side_effect=lambda _device: {"source_sha256": {"base": "0" * 64}}):
            baseline = evaluate.inference_runtime(torch.device("cpu"))
            for path in required:
                self.assertEqual(baseline["source_sha256"][str(path.resolve().relative_to(package_root))], train.old._sha256(path))
            original_sha = train.old._sha256
            for changed_path in required:
                def changed_sha(path):
                    return "f" * 64 if Path(path).resolve() == changed_path.resolve() else original_sha(path)
                with patch.object(train.old, "_sha256", side_effect=changed_sha):
                    changed = evaluate.inference_runtime(torch.device("cpu"))
                self.assertFalse(reuse._compatible(protocol(baseline), protocol(changed)))
            self.assertTrue(reuse._compatible(protocol(baseline), protocol(deepcopy(baseline))))

    def test_loader_rejects_candidate_source_change_before_constructing_model(self):
        source = fixture()
        with tempfile.TemporaryDirectory() as directory, context(source):
            root = Path(directory)
            endpoint(root, source)
            original_sha = train.old._sha256
            for changed_path in (Path(train.__file__), Path(model.__file__),
                                 Path(train.cont.__file__), Path(train.old.__file__)):
                def changed_sha(path):
                    return "e" * 64 if Path(path).resolve() == changed_path.resolve() else original_sha(path)
                with patch.object(train.old, "_sha256", side_effect=changed_sha), \
                        patch.object(train, "load_model", side_effect=AssertionError("changed source must be rejected before model construction")):
                    with self.assertRaisesRegex(ValueError, "implementation changed"):
                        evaluate.load_frozen_model(root, "fixed_epoch")
            loaded, receipt = evaluate.load_frozen_model(root, "fixed_epoch")
            self.assertIsInstance(loaded, model.FrozenCandidateLocalModel)
            self.assertEqual(receipt["architecture"], "candidate_local_ca_residual")


if __name__ == "__main__":
    unittest.main()
