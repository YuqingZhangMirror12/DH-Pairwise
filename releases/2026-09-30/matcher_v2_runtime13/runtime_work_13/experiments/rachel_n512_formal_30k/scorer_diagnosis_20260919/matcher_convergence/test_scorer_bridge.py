"""CPU bridge wiring/parity tests; no real checkpoint, GPU or formal training."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import scorer_bridge as b
from ..matched_only.test_training import make_cache, TinyData, TinyModel, same_tree


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def bridge(self, root, epoch):
        path = root / ("m%d.pt" % epoch)
        source = dict(epoch=epoch,resume_identity=dict(populations=dict(
            train=dict(count=24000,manifest_sha256=b.core.TRAIN_SHA256),
            val=dict(count=3000,manifest_sha256=b.core.VAL_SHA256))))
        torch.save(source,path)
        digest = b.core.SOURCE_SHA256 if epoch == 12 else b.data.sha(path)
        info = dict(checkpoint_sha256=digest,matcher_state_sha256=str(epoch)*32)
        with patch.object(b.endpoint_loader,"load_endpoint",return_value=(None,None,info)) as loader:
            bridge = b.EndpointBridge(path,epoch)
            loader.assert_called_once_with(path.resolve(),epoch)
        return bridge

    def test_original_modules_unchanged_and_training_kernel_bytecode_shared(self):
        originals = (b.train.SCHEMA,b.cache.SOURCE_SHA,b.train.run,b.train.plan,b.data.FormalCache)
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp),20)
            self.assertEqual(bridge.train.SCHEMA,b.TRAIN_SCHEMA)
            self.assertEqual(bridge.train.ARMS,("all_tokens",))
            self.assertIs(bridge.train.train_segment.__code__,b.train.train_segment.__code__)
            self.assertIsNot(bridge.train.train_segment.__globals__,b.train.train_segment.__globals__)
            self.assertIs(bridge.cache.compute.__code__,b.cache.compute.__code__)
        self.assertEqual(originals,(b.train.SCHEMA,b.cache.SOURCE_SHA,b.train.run,b.train.plan,b.data.FormalCache))

    def test_three_matchers_have_identical_fresh_d2_and_shuffle_budget(self):
        states=[]
        with tempfile.TemporaryDirectory() as tmp:
            for epoch in (12,16,20):
                bridge=self.bridge(Path(tmp),epoch)
                model=bridge.train.make_scorer("all_tokens",seed=b.train.HEAD_SEED)
                self.assertEqual((model.head.depth,model.head.cross_attention.num_heads),(2,4))
                states.append(deepcopy(model.state_dict()))
                plan=bridge.train.plan()
                self.assertEqual(len(plan),64)
                self.assertEqual(sum(r["count"] for r in plan),384000)
                self.assertEqual((plan[0]["absolute_epoch"],plan[-1]["absolute_epoch"]),(epoch+1,epoch+16))
                self.assertEqual((plan[0]["head_data_shuffle_epoch"],plan[-1]["head_data_shuffle_epoch"]),(13,28))
                for e in (1,8,16):
                    actual=bridge.train.runner.epoch_indices(24000,seed=260913,epoch=epoch+e,limit=32)
                    expected=b.train.runner.epoch_indices(24000,seed=260913,epoch=12+e,limit=32)
                    self.assertEqual(actual,expected)
            for name,value in states[0].items():
                self.assertTrue(all(torch.equal(value,state[name]) for state in states[1:]),name)

    def test_checkpoint_explicit_clocks_and_same_original_resume_kernel(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge=self.bridge(Path(tmp),20)
            model=bridge.train.make_scorer("all_tokens",seed=b.train.HEAD_SEED)
            optimizer=bridge.train.create_optimizer(model)
            identity=dict(unit="strict unchanged")
            saved=bridge.train.checkpoint_payload(model,optimizer,identity,0,{},"initial")
            self.assertEqual((saved["matcher_epoch"],saved["head_epoch"],saved["absolute_epoch"],saved["head_data_shuffle_epoch"]),(20,0,20,12))
            n,winners=bridge.train.restore(model,optimizer,saved,identity)
            self.assertEqual((n,winners),(0,{}))
            for key in ("matcher_epoch","absolute_epoch","head_data_shuffle_epoch"):
                bad=dict(saved);bad[key]+=1
                with self.assertRaisesRegex(ValueError,"clock"):
                    bridge.train.restore(model,optimizer,bad,identity)

    def test_e1_e8_e16_identical_data_and_numerical_optimizer_steps(self):
        dataset=TinyData(24000,"train_")
        with tempfile.TemporaryDirectory() as tmp:
            bridges=[self.bridge(Path(tmp),epoch) for epoch in (12,16,20)]
            for head_epoch in (1,8,16):
                snapshots=[]
                for bridge in bridges:
                    torch.manual_seed(260914)
                    model=TinyModel()
                    optimizer=bridge.train.create_optimizer(model)
                    optimizer.param_groups[0]["lr"]=bridge.train.learning_rate(head_epoch)
                    ids=bridge.train.runner.epoch_indices(24000,seed=260913,
                        epoch=bridge.epoch+head_epoch,limit=32)
                    report=bridge.train.train_segment(model,dataset,ids,optimizer,torch.device("cpu"))
                    self.assertEqual(report["optimizer_updates"],2)
                    snapshots.append((ids,deepcopy(model.state_dict()),deepcopy(optimizer.state_dict()),
                        report["mean_loss"],torch.get_rng_state().clone()))
                same_tree(self,snapshots[0],snapshots[1])
                same_tree(self,snapshots[0],snapshots[2])

    def test_initial_partial_and_epoch_anchor_roundtrip_exact_state(self):
        # Unit-only segment16 keeps true Adam update counts without375 CPU
        # updates per segment. The public factory/CLI remains fixed6000.
        dataset=TinyData(24000,"train_")
        with tempfile.TemporaryDirectory() as tmp:
            for matcher_epoch in (12,16,20):
                bridge=self.bridge(Path(tmp),matcher_epoch)
                bridge.train.SEGMENT_SIZE=16
                for completed in (0,1,4,5):
                    torch.manual_seed(260914)
                    model=TinyModel();optimizer=bridge.train.create_optimizer(model)
                    for number in range(1,completed+1):
                        head_epoch=(number+3)//4
                        optimizer.param_groups[0]["lr"]=bridge.train.learning_rate(head_epoch)
                        ids=bridge.train.runner.epoch_indices(24000,seed=260913,
                            epoch=matcher_epoch+head_epoch,limit=16)
                        bridge.train.train_segment(model,dataset,ids,optimizer,torch.device("cpu"))
                    identity=dict(unit=True,matcher_endpoint=bridge.endpoint)
                    saved=bridge.train.checkpoint_payload(model,optimizer,identity,completed,{},"unit_recovery")
                    h=(completed+3)//4
                    self.assertEqual((saved["head_epoch"],saved["absolute_epoch"],saved["head_data_shuffle_epoch"]),
                        (h,matcher_epoch+h,12+h))
                    path=Path(tmp)/"last.pt";torch.save(saved,path)
                    saved=torch.load(path,map_location="cpu",weights_only=False)
                    expected_rng=torch.rand(5)
                    restored=TinyModel();other=bridge.train.create_optimizer(restored)
                    n,_=bridge.train.restore(restored,other,saved,identity)
                    self.assertEqual(n,completed)
                    same_tree(self,restored.state_dict(),model.state_dict())
                    same_tree(self,other.state_dict(),optimizer.state_dict())
                    self.assertTrue(torch.equal(torch.rand(5),expected_rng))
                    bad=deepcopy(saved);bad["identity"]["matcher_endpoint"]["matcher_epoch"]=99
                    with self.assertRaisesRegex(ValueError,"identity"):
                        bridge.train.restore(restored,other,bad,identity)

    def test_new_cache_source_separation_and_no_labels_in_model_args(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(b.data,"FULL_COUNTS",dict(train=4,val=4)):
            root=Path(tmp); bridge=self.bridge(root,16)
            protocol=make_cache(root/"cache")
            protocol.update(schema=b.CACHE_SCHEMA,source_checkpoint_sha256=bridge.endpoint["matcher_checkpoint_sha256"],
                matcher_endpoint=bridge.endpoint,bridge_implementation_sha256=b.data.sha(b.__file__),
                numeric_cache_implementation_sha256=b.data.sha(b.cache.__file__))
            b.train.save(root/"cache/protocol.json",protocol)
            dataset=bridge.data.FormalCache(root/"cache","train")
            batch=dataset.batch([0,1])
            self.assertEqual(set(batch.model_kwargs),{"candidate_weights","points_a_rc","points_b_rc"})
            self.assertEqual(len(batch.model_args),5)
            other=self.bridge(root,20)
            with self.assertRaisesRegex(ValueError,"different Matcher"):
                other.data.FormalCache(root/"cache","train")

    def test_legacy_m12_cache_only_reused_by_m12_with_original_numeric_code(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(b.data,"FULL_COUNTS",dict(train=4,val=4)):
            root=Path(tmp); bridge=self.bridge(root,12)
            protocol=make_cache(root/"cache")
            protocol["implementation_sha256"]=b.data.sha(b.cache.__file__)
            b.train.save(root/"cache/protocol.json",protocol)
            self.assertTrue(bridge.data.FormalCache(root/"cache","train").binding["legacy_m12_cache_reused"])
            other=self.bridge(root,16)
            with self.assertRaises(ValueError):other.data.FormalCache(root/"cache","train")
            protocol["implementation_sha256"]="changed"
            b.train.save(root/"cache/protocol.json",protocol)
            with self.assertRaisesRegex(ValueError,"implementation"):
                bridge.data.FormalCache(root/"cache","train")

    def test_endpoint_cache_protocol_records_current_matcher_not_m12(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bridge=self.bridge(root,20)
            bridge.cache.save(root/"protocol.json",dict(status="complete"))
            protocol=json.loads((root/"protocol.json").read_text())
            self.assertEqual(protocol["matcher_endpoint"]["matcher_epoch"],20)
            self.assertEqual(protocol["bridge_implementation_sha256"],b.data.sha(b.__file__))

    def test_bridge_source_is_in_prediction_reuse_compared_runtime_field(self):
        from experiments.rachel_n512_formal_30k import decoupled_prediction_reuse as reuse
        with tempfile.TemporaryDirectory() as tmp:
            a=self.bridge(Path(tmp),16);c=self.bridge(Path(tmp),20)
            with patch.object(b.evaluate.original,"inference_runtime",side_effect=lambda d:{"source_sha256":{}}):
                ra=a.evaluate.inference_runtime(torch.device("cpu"))
                rc=c.evaluate.inference_runtime(torch.device("cpu"))
            self.assertIn("inference_runtime",reuse.PROTOCOL_FIELDS)
            self.assertNotEqual(ra["source_sha256"]["matcher_checkpoint"],rc["source_sha256"]["matcher_checkpoint"])
            self.assertEqual(ra["source_sha256"]["scorer_bridge"],b.data.sha(b.__file__))

    def test_existing_m12_all_tokens_head_reuse_still_calls_original_frozen_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bridge=self.bridge(root,12)
            head=b.train.make_scorer("all_tokens",seed=b.train.HEAD_SEED)
            ident=dict(arm="all_tokens",head_seed=b.train.HEAD_SEED,data_seed=b.train.DATA_SEED,
                initial_state_sha256=b.train.state_digest(head))
            b.train.save(root/"freezes/c16.json",dict(schema=b.train.SCHEMA,identity=ident))
            adapter=object()
            with patch.object(b.evaluate,"load_frozen_model",return_value=(adapter,dict(head_epoch=16))) as inherited:
                model,receipt=bridge.evaluate.load_frozen_model(root,"fixed_epoch")
            inherited.assert_called_once_with(root,"fixed_epoch",budget=16)
            self.assertIs(model,adapter)
            self.assertEqual((receipt["matcher_epoch"],receipt["head_epoch"],receipt["epoch"]),(12,16,28))
            self.assertTrue(receipt["legacy_m12_head_reused"])

    def test_frozen_loader_rejects_head_bound_to_another_matcher_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bridge=self.bridge(root,20)
            head=b.train.make_scorer("all_tokens",seed=b.train.HEAD_SEED)
            ident=dict(arm="all_tokens",head_seed=b.train.HEAD_SEED,data_seed=b.train.DATA_SEED,
                initial_state_sha256=b.train.state_digest(head),
                matcher_endpoint=dict(bridge.endpoint,matcher_epoch=16))
            b.train.save(root/"freezes/c16.json",dict(schema=b.TRAIN_SCHEMA,identity=ident))
            with patch.object(b.endpoint_loader,"load_endpoint",side_effect=AssertionError("not reached")), \
                    self.assertRaisesRegex(ValueError,"another fixed Matcher"):
                bridge.evaluate.load_frozen_model(root,"fixed_epoch")

    def test_m20_c16_frozen_loader_composes_actual_typed_base_and_head(self):
        from ..matched_only.test_evaluate import LoaderTests
        from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config, RachelN512Pairwise
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bridge=self.bridge(root,20)
            frozen=LoaderTests().fixture(root,arm="all_tokens")
            head=b.train.make_scorer("all_tokens",seed=b.train.HEAD_SEED)
            frozen["schema"]=b.TRAIN_SCHEMA
            frozen["identity"].update(source_checkpoint_sha256=bridge.endpoint["matcher_checkpoint_sha256"],
                matcher_endpoint=deepcopy(bridge.endpoint),source_matcher_epochs=20,
                initial_state_sha256=b.train.state_digest(head),
                implementation_sha256=bridge.train.implementation_binding())
            checkpoint=root/"head_epoch_016.pt"
            saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
            saved.update(schema=b.TRAIN_SCHEMA,identity=deepcopy(frozen["identity"]),
                matcher_epoch=20,absolute_epoch=36,head_data_shuffle_epoch=28)
            torch.save(saved,checkpoint)
            for chosen in frozen["selections"].values():
                chosen["checkpoint_sha256"]=b.data.sha(checkpoint)
            b.train.save(root/"freezes/c16.json",frozen)
            base=RachelN512Pairwise(RachelN512Config())
            with patch.object(b.endpoint_loader,"load_endpoint",return_value=(base,None,
                    dict(checkpoint_sha256=bridge.endpoint["matcher_checkpoint_sha256"]))) as typed:
                adapter,receipt=bridge.evaluate.load_frozen_model(root,"fixed_epoch")
            typed.assert_called_once_with(bridge.path,20)
            self.assertIs(adapter.base_model,base)
            self.assertFalse(any(p.requires_grad for p in adapter.parameters()))
            self.assertEqual((receipt["matcher_epoch"],receipt["head_epoch"],receipt["epoch"],receipt["budget"]),(20,16,36,36))
            self.assertFalse(receipt["legacy_m12_head_reused"])
            self.assertEqual(adapter.metadata()["matcher_endpoint"],bridge.endpoint)


if __name__=="__main__":
    unittest.main()
