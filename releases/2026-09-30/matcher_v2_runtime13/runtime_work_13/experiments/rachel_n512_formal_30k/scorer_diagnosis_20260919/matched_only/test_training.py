"""Small synthetic CPU tests, not a formal training/inference run.

The scheduling test replaces ONLY sample counts/segment size and the neural
model with a tiny dropout net; it exercises the real checkpoint/resume loop.
Cache tests use complete tiny fixtures with patched expected counts, never a
formal CLI bypass. No remote, checkpoint downloads, REAL or OOD data.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import cache, data, model, train


def make_cache(root, split="train", count=4):
    root.mkdir()
    arrays = {name: np.zeros((count, *tail), dtype=dtype)
              for name, (dtype, tail) in cache.ARRAYS.items()}
    arrays["features_a"][:] = np.random.default_rng(8).normal(size=arrays["features_a"].shape)
    arrays["features_b"][:] = np.random.default_rng(9).normal(size=arrays["features_b"].shape)
    arrays["valid_a"][:, :4] = arrays["valid_b"][:, :4] = True
    arrays["points_a"][:, :4] = [[0., 0.], [30., 0.], [60., 0.], [100., 100.]]
    arrays["points_b"][:] = arrays["points_a"] + [10., 5.]
    arrays["candidate_indices"][:] = -1
    arrays["translation_a_to_b_rc"][:] = np.nan
    for i in range(0, count, 2):
        arrays["candidate_indices"][i, :3] = [[0, 0], [1, 1], [2, 2]]
        arrays["candidate_valid"][i, :3] = arrays["candidate_inliers"][i, :3] = True
        arrays["mask_a"][i, :3] = arrays["mask_b"][i, :3] = True
        arrays["candidate_weights"][i, :3] = [.9, .8, .7]
        arrays["translation_a_to_b_rc"][i] = [10., 5.]
        arrays["layout_valid"][i] = True
    arrays["label"][:] = np.arange(count) % 2
    arrays["training_valid"][:] = arrays["decision_valid"][:] = arrays["ready"][:] = True
    for name, array in arrays.items():
        np.save(root / (name+".npy"), array, allow_pickle=False)
    records = [dict(ordinal=i, pair_id=split+"_%d" % i, label=float(i % 2), input_sha256="f"*64)
               for i in range(count)]
    train.save(root / "pairs.json", records)
    protocol = dict(schema=cache.SCHEMA, status="complete", formal_training_eligible=True,
        split=split, pair_count=count, expected_full_count=count, completed_pairs=count,
        source_checkpoint_sha256=cache.SOURCE_SHA,
        population=dict(count=count, split=split, sampling="original512", contour_cap=512,
            manifest_sha256=cache.TRAIN_SHA if split=="train" else data.VAL_SHA),
        matcher_frozen=True, selector_gt_free=True, no_online_augmentation=True,
        precompute_device="cpu", features_dtype="float32", decoder=asdict(model.DECODER_CONFIG),
        source_model_config=dict(canvas_size=800, contour_cap=512, feature_dim=96, num_heads=4,
            window_sizes_px=[7.,16.,32.,64.], patch_size=16),
        arrays={name: dict(dtype=dtype, shape=[count,*tail]) for name,(dtype,tail) in cache.ARRAYS.items()},
        pairs_sha256=data.sha(root / "pairs.json"), positive_count=count//2,
        implementation_sha256="a"*64)
    train.save(root / "protocol.json", protocol)
    return protocol


class TinyData:
    def __init__(self, count, prefix):
        self.count, self.prefix = count, prefix
    def __len__(self):
        return self.count
    def batch(self, indices, device="cpu"):
        ids = torch.as_tensor(list(indices), device=device)
        x = torch.stack(((ids.float()-self.count/2)/self.count, (ids % 3).float()/3), dim=1)
        return data.CacheBatch((x,), {}, (ids % 2).float(), ids % 3 != 0,
            torch.ones_like(ids, dtype=torch.bool), tuple(self.prefix+str(int(i)) for i in ids))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.dropout = nn.Dropout(.25)
    def forward(self, x):
        z = self.linear(self.dropout(x)).squeeze(-1)
        return SimpleNamespace(logit=z, used_fallback=torch.zeros_like(z, dtype=torch.bool),
            has_decoded_candidate=torch.ones_like(z, dtype=torch.bool))
    def metadata(self):
        return dict(tiny_test_only=True)


def same_tree(test, a, b):
    if torch.is_tensor(a):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a:
            same_tree(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(type(a), type(b)); test.assertEqual(len(a), len(b))
        for aa, bb in zip(a,b):
            same_tree(test, aa, bb)
    else:
        test.assertEqual(a,b)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # This local Torch environment has no sklearn. Test-only substitute
        # for AP uses the project's exact tie-grouped dependency-light AP,
        # not invented metrics. On the formal server sklearn is used unchanged.
        cls.test_only_ap_substitute = importlib.util.find_spec("sklearn") is None
        if cls.test_only_ap_substitute:
            from staging.pairwise_v0_2.training.metrics import _ranking_metrics
            sk, skm = ModuleType("sklearn"), ModuleType("sklearn.metrics")
            skm.average_precision_score = lambda y,s: _ranking_metrics(list(s),list(y))[1]
            sk.metrics = skm
            with patch.dict(sys.modules,{"sklearn":sk,"sklearn.metrics":skm}):
                module = importlib.import_module("experiments.rachel_n512_formal_30k.recall_operating_points")
            # patch.dict restores the module table, including imports made
            # inside its scope. Retain only this test's original helper module.
            sys.modules[module.__name__] = module
            cls.addClassCleanup(lambda: sys.modules.pop(module.__name__,None))

    def test_formal_plan_batch_and_lr(self):
        p = train.plan()
        self.assertEqual(len(p),64)
        self.assertEqual((p[0]["absolute_epoch"], p[-1]["absolute_epoch"]),(13,28))
        self.assertEqual(sum(x["count"] for x in p),384000)
        self.assertEqual(train.BATCH,16)
        self.assertEqual(train.learning_rate(3),1e-4)
        self.assertEqual(train.learning_rate(4),2e-5)
        self.assertEqual(sum(x["epoch_complete"] for x in p),16)
        self.assertEqual(p[32]["head_epoch"],9)
        for e in range(1,17):
            order = train.runner.epoch_indices(24000, seed=train.DATA_SEED, epoch=12+e, limit=None)
            self.assertEqual(len(np.unique(order)),24000)
        with self.assertRaises(ValueError):
            train.learning_rate(0)

    def test_loss_means_whole_batch_and_keeps_invalid_zero_grad(self):
        logits = torch.tensor([0., 0.], requires_grad=True)
        value = train.pair_loss(logits, torch.tensor([1.,0.]), torch.tensor([True,False]))
        self.assertAlmostEqual(float(value),np.log(2)/2, places=7)
        value.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([-.25,0.]))
        zero = torch.zeros(16,requires_grad=True)
        train.pair_loss(zero, torch.zeros(16),torch.zeros(16,dtype=torch.bool)).backward()
        self.assertTrue(torch.equal(zero.grad,torch.zeros(16)))

    def test_incomplete_probe_and_wrong_source_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"cache"
            p = make_cache(root)
            # The genuine production expectations reject tiny counts, even if
            # an ineligible flag were falsely changed to true.
            with self.assertRaises(ValueError):
                data.FormalCache(root,"train")
            with patch.dict(data.FULL_COUNTS,{"train":4}):
                for key, value in (("formal_training_eligible",False),("status","running"),
                                   ("source_checkpoint_sha256","wrong"),("completed_pairs",3)):
                    changed = dict(p, **{key:value}); train.save(root/"protocol.json",changed)
                    with self.assertRaises(ValueError):
                        data.FormalCache(root,"train")
                train.save(root/"protocol.json",p)
                self.assertEqual(len(data.FormalCache(root,"train")),4)
            with self.assertRaises(ValueError):
                data.FormalCache(root,"real")

    def test_cache_adapter_gt_blind_readonly_and_all_arms_forward(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(data.FULL_COUNTS,{"train":4}):
            root = Path(tmp)/"cache"; make_cache(root)
            ds = data.FormalCache(root,"train")
            batch = ds.batch([0,1])
            self.assertIsInstance(ds.arrays["features_a"],np.memmap)
            self.assertFalse(ds.arrays["features_a"].flags.writeable)
            self.assertEqual(set(batch.model_kwargs), {"candidate_weights","points_a_rc","points_b_rc"})
            self.assertIsInstance(batch.model_args[-1].reasons,tuple)
            for arm in model.ARMS:
                net = model.make_fresh_scorer(arm, seed=train.HEAD_SEED)
                out = net(*batch.model_args,**batch.model_kwargs)
                self.assertTrue(torch.isfinite(out.logit).all())
                train.pair_loss(out.logit,batch.labels,batch.training_valid).backward()
                self.assertTrue(any(p.grad is not None for p in net.parameters()))
                before = train.state_digest(net)
                optimizer = train.create_optimizer(net)
                segment = train.train_segment(net,ds,np.tile(np.arange(4),4),optimizer,"cpu")
                self.assertEqual(segment["samples"],16)
                self.assertEqual(segment["optimizer_updates"],1)
                self.assertEqual(segment["physical_microbatch"],16)
                self.assertNotEqual(before,train.state_digest(net))
                self.assertTrue(all(p.dtype == torch.float32 for p in net.parameters()))
            # Label mutation affects the separate loss target, never selector
            # or features. Unknown GT metadata is also not forwarded.
            ds.records[0]["label"] = 1.
            ds.records[0]["gt_translation"] = [900.,900.]
            again = ds.batch([0,1])
            self.assertEqual(again.model_kwargs.keys(),batch.model_kwargs.keys())
            for a,b in zip(again.model_args[:4],batch.model_args[:4]):
                self.assertTrue(torch.equal(a,b))
            self.assertEqual(again.labels.tolist(),[0.,1.])
            with self.assertRaises(ValueError):
                ds.batch([4])

    def test_cache_pair_hash_ready_and_array_header_checks(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(data.FULL_COUNTS,{"train":4}):
            root=Path(tmp)/"cache"; make_cache(root)
            ready=np.load(root/"ready.npy",mmap_mode="r+"); ready[2]=False; ready.flush()
            with self.assertRaises(ValueError): data.FormalCache(root,"train")
            ready[2]=True; ready.flush()
            np.save(root/"label.npy",np.zeros((4,1),np.float32),allow_pickle=False)
            with self.assertRaises(ValueError): data.FormalCache(root,"train")

    def test_aux_selection_earliest_tie_coverage_and_no_real(self):
        model_=TinyModel().eval()
        report, rows = train.evaluate(model_,TinyData(18,"val"),"cpu")
        self.assertEqual(report["sample_count"],18)
        self.assertEqual(len(rows),18)
        self.assertEqual(rows[-1]["pair_id"],"val17")
        self.assertEqual(rows[0]["logit"],0.)
        self.assertEqual(rows[0]["score"],.5)
        winners = train.update_winners({},1,report)
        self.assertEqual(train.update_winners(winners,2,report),winners)
        invalid = dict(report,decision_valid_count=17,max_f1_selection_key=[10.,10.])
        self.assertEqual(train.update_winners(winners,2,invalid),winners)
        args=train.parser().parse_args(["--arm","all_tokens","--train-cache","a","--val-cache","b","--output","c"])
        self.assertEqual(args.stop_after_head_epoch,16)
        self.assertFalse(hasattr(args,"real")); self.assertFalse(hasattr(args,"ood"))

    def test_restore_rejects_identity_ledger_and_moments_tampering(self):
        m=TinyModel(); opt=train.create_optimizer(m)
        ident={"test_only":True}
        saved=train.checkpoint_payload(m,opt,ident,0,{},"initial")
        for key,value in (("identity",{"changed":True}),("classifier_pair_exposures",1),
                          ("optimizer_parameter_steps",{}),("matcher_updated",True)):
            broken=deepcopy(saved); broken[key]=value
            target=TinyModel()
            with self.assertRaises(ValueError):
                train.restore(target,train.create_optimizer(target),broken,ident)
        target=TinyModel(); optimizer=train.create_optimizer(target)
        self.assertEqual(train.restore(target,optimizer,saved,ident),(0,{}))

    def test_conditional_parameters_need_not_reach_global_adam_counter(self):
        class Conditional(nn.Module):
            def __init__(self):
                super().__init__(); self.a=nn.Parameter(torch.tensor(1.)); self.b=nn.Parameter(torch.tensor(2.))
            def metadata(self): return {"conditional_test":True}
        with patch.object(train,"SEGMENT_SIZE",32):
            m=Conditional(); opt=train.create_optimizer(m)
            for name in ("a","b"):
                opt.zero_grad(set_to_none=True); getattr(m,name).square().backward(); opt.step()
            saved=train.checkpoint_payload(m,opt,{"test":True},1,{},"segment_recovery")
            self.assertEqual(list(saved["optimizer_parameter_steps"].values()),[1.,1.])
            copied=Conditional(); other=train.create_optimizer(copied)
            self.assertEqual(train.restore(copied,other,saved,{"test":True})[0],1)
            bad=deepcopy(saved)
            state=next(iter(bad["optimizer_state_dict"]["state"].values()))
            state["exp_avg_sq"]=torch.tensor(float("nan"))
            with self.assertRaises(ValueError): train.restore(copied,other,bad,{"test":True})

    def test_exact_resume_pause_c8_final_c16_and_repair_publication(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            for name,value in (("SEGMENT_SIZE",16),("TRAIN_COUNT",64),("VAL_COUNT",18)):
                stack.enter_context(patch.object(train,name,value))
            training,validation=TinyData(64,"train"),TinyData(18,"val")
            ident={"tiny_schedule_test":True}
            def launch(root,stop,resume=False):
                train.runner._set_determinism(817)
                model_=TinyModel(); opt=train.create_optimizer(model_)
                root.mkdir(exist_ok=True)
                args=SimpleNamespace(output=str(root),arm="all_tokens",resume=resume,stop_after_head_epoch=stop)
                with redirect_stdout(io.StringIO()):
                    result=train.execute(args,model_,opt,training,validation,ident,"cpu")
                return result,torch.load(root/"last.pt",weights_only=False)
            full=Path(tmp)/"full"; split=Path(tmp)/"split"
            result,expected=launch(full,16)
            self.assertEqual(result["status"],"complete")
            paused,anchor=launch(split,8)
            self.assertEqual(paused["status"],"paused")
            self.assertEqual(anchor["completed_segments"],32)
            self.assertEqual(anchor["absolute_epoch"],20)
            (split/"freezes/c8.json").unlink()  # simulated post-commit publication interruption
            resumed,actual=launch(split,16,True)
            self.assertEqual(resumed["status"],"complete")
            self.assertTrue((split/"freezes/c8.json").is_file())
            for name in ("model_state_dict","optimizer_state_dict","rng_state","optimizer_parameter_steps","winners"):
                same_tree(self,expected[name],actual[name])
            self.assertEqual(actual["head_epoch"],16)
            self.assertEqual(actual["absolute_epoch"],28)
            self.assertEqual(actual["optimizer_updates"],64)
            c8=json.loads((split/"freezes/c8.json").read_text())
            c16=json.loads((split/"freezes/c16.json").read_text())
            self.assertLessEqual(c8["selections"]["max_f1"]["head_epoch"],8)
            self.assertEqual(c16["selections"]["fixed_endpoint"]["head_epoch"],16)
            self.assertEqual(c16["primary_selection"],"fixed_endpoint")
            (split/"freezes/c16.json").unlink()
            repaired,again=launch(split,16,True)
            self.assertEqual(repaired["status"],"complete")
            same_tree(self,actual["model_state_dict"],again["model_state_dict"])

    def test_failed_segment_never_claims_completion_and_resumes_last_commit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(train,"SEGMENT_SIZE",16):
            root=Path(tmp); net=TinyModel(); opt=train.create_optimizer(net)
            args=SimpleNamespace(output=str(root),arm="all_tokens",resume=False,stop_after_head_epoch=8)
            real=train.train_segment
            calls=[]
            def fail_second(*args):
                calls.append(1)
                if len(calls)==2: raise RuntimeError("synthetic interrupted segment")
                return real(*args)
            with patch.object(train,"train_segment",side_effect=fail_second):
                with self.assertRaises(RuntimeError),redirect_stdout(io.StringIO()):
                    train.execute(args,net,opt,TinyData(64,"train"),TinyData(18,"val"),{"test":True},"cpu")
            saved=torch.load(root/"last.pt",weights_only=False)
            self.assertEqual(saved["completed_segments"],1)
            self.assertEqual(json.loads((root/"protocol.json").read_text())["status"],"failed")
            self.assertFalse((root/"head_epoch_001.pt").exists())

    def test_shared_gpu_lock_is_nonblocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock=Path(tmp)/"heatmap-gpu.lock"
            with train.lock_owner.gpu_lock(lock,check_gpu=False):
                with self.assertRaises(BlockingIOError):
                    with train.lock_owner.gpu_lock(lock,check_gpu=False):
                        self.fail("busy GPU lock was acquired")
            with train.lock_owner.gpu_lock(lock,check_gpu=False):
                pass


if __name__=="__main__":
    unittest.main()
