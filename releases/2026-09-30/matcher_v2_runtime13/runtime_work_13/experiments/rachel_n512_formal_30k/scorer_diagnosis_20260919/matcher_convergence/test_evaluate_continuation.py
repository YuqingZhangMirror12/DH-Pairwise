"""Fixed-endpoint reader wiring only; no inference, optimizer, GPU or metrics."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import evaluate_continuation as ev


def metadata(epoch):
    result = dict(epoch=epoch, completed_segments=4*epoch, phase="matcher",
        global_exposure=24000*epoch, optimizer_updates=1500*epoch,
        checkpoint_role="epoch_anchor", formal_training_counted=True)
    if epoch > 12:
        result.update(rng_mode="exact_gpu", continuation_identity=dict(rng_mode="exact_gpu"))
    return result


def inputs(root, count=3000):
    checkpoint = root/"checkpoints"/"epoch.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"not a real checkpoint; preflight must not load it")
    dataset = root/"dataset"
    (dataset/"pairs").mkdir(parents=True)
    manifest = dataset/"pairs"/"val.jsonl"
    manifest.write_text("\n".join(json.dumps(dict(pair_id="p%04d"%i,label=int(i%2==0)))
                                  for i in range(count))+"\n")
    args = ev.parser().parse_args(["--checkpoint",str(checkpoint),"--epoch","16",
        "--dataset",str(dataset),"--output",str(root/"evaluation")])
    return args,manifest


class ContinuationEvaluationTests(unittest.TestCase):
    def test_accepts_exact_m12_m16_m20_metadata(self):
        for epoch in (12,16,20):
            with self.subTest(epoch=epoch):
                expected = ev.validate_endpoint_metadata(metadata(epoch),epoch)
                self.assertEqual(expected["phase"],"matcher")
                self.assertEqual(expected["completed_segments"],4*epoch)
                self.assertEqual(expected["optimizer_updates"],1500*epoch)

    def test_rejects_classifier_partial_cpu_and_invalid_epoch(self):
        for update in (dict(phase="classifier"),dict(completed_segments=79),
                dict(checkpoint_role="recovery"),dict(formal_training_counted=False),
                dict(inference_only=True),dict(rng_mode="cpu_only_probe"),
                dict(continuation_identity=dict(rng_mode="cpu_only_probe"))):
            with self.subTest(update=update), self.assertRaises(ValueError):
                ev.validate_endpoint_metadata(dict(metadata(20),**update),20)
        for epoch in (8,13,15,21,None,True,16.,"16"):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                ev.validate_endpoint_metadata(metadata(16),epoch)

    def test_default_preflight_reads_manifest_only_without_model_or_output(self):
        with TemporaryDirectory() as tmp:
            args,manifest = inputs(Path(tmp))
            self.assertFalse(args.execute)
            with patch.object(ev.original,"sha256",return_value=ev.original.VAL_HASH) as digest, \
                    patch.object(ev,"load_endpoint",side_effect=AssertionError("must not load a model")) as load, \
                    patch.object(ev,"execute",side_effect=AssertionError("must not execute")) as execute, \
                    patch.object(torch,"load",side_effect=AssertionError("must not open checkpoint")) as torch_load:
                result = ev.preflight(args)
            digest.assert_called_once_with(manifest.resolve())
            load.assert_not_called();execute.assert_not_called();torch_load.assert_not_called()
            self.assertEqual(result["status"],"preflight_only")
            self.assertEqual((result["count"],result["positive_count"],result["negative_count"]),(3000,1500,1500))
            self.assertEqual(len(result["pair_ids"]),3000)
            self.assertFalse(result["classification_metrics"])
            self.assertFalse(result["selection_performed"])
            self.assertFalse(result["execute_requested"])
            self.assertFalse(args.output.exists())

    def test_preflight_rejects_actual_tiny_json_manifest_even_with_matching_hash(self):
        with TemporaryDirectory() as tmp:
            args,_ = inputs(Path(tmp),count=2)
            with patch.object(ev.original,"sha256",return_value=ev.original.VAL_HASH), \
                    patch.object(torch,"load",side_effect=AssertionError("must not open checkpoint")), \
                    self.assertRaisesRegex(ValueError,"balanced clean SIMVAL3000"):
                ev.preflight(args)
            self.assertFalse(args.output.exists())

    def test_m16_loader_calls_typed_continuation_validation_and_freezes_base(self):
        from . import continuation_core as core
        from staging.pairwise_v0_2.models import rachel_decoupled_score as wrapper_module
        payload = metadata(16)
        identity = payload["continuation_identity"]
        identity.update(source_frozen_digests=dict(head="frozen"), source_checkpoint_sha256="source",
            implementation_bindings=dict(core="implementation"),
            source_resume_identity=dict(populations=dict(val=dict(manifest_sha256=ev.original.VAL_HASH,
                                                                  count=3000,split="val"))))
        payload.update(current_base_state_sha256="base",loss_config={})
        base = torch.nn.Linear(2,2)
        wrapper = SimpleNamespace(base_model=base)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"endpoint.pt"
            path.write_bytes(b"mocked checkpoint")
            with patch.object(torch,"load",return_value=payload) as load, \
                    patch.object(core,"validate_progress",return_value=64) as progress, \
                    patch.object(core,"validate_optimizer") as optimizer, \
                    patch.object(core,"validate_source",side_effect=AssertionError("not an M12 import")), \
                    patch.object(core,"frozen_digests",return_value=deepcopy(identity["source_frozen_digests"])), \
                    patch.object(core.old,"state_digest",return_value="base"), \
                    patch.object(wrapper_module,"load_decoupled_score_checkpoint",return_value=wrapper) as typed:
                model,config,info = ev.load_endpoint(path,16)
            load.assert_called_once_with(path.resolve(),map_location="cpu",weights_only=False)
            progress.assert_called_once_with(payload,identity,rng_mode="exact_gpu")
            optimizer.assert_called_once_with(payload,wrapper,expected_updates=24000)
            typed.assert_called_once_with(payload)
            self.assertIs(model,base)
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertTrue(config.validate_runtime_targets)
            self.assertEqual(info["current_matcher_epochs"],16)
            self.assertFalse(info["trained_scorer_head_called"])
            self.assertFalse(info["trained_classifier_evaluated"])


if __name__=="__main__":
    unittest.main()
