"""Synthetic CPU checks only; formal CLI has no reduced-budget/CPU bypass."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import train
from .data import AuxiliaryBatch
from .test_data import entry, sample
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import collate_rachel_pairs


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(4,1)
        self.scale=nn.Parameter(torch.tensor(.7))
        self.bias=nn.Parameter(torch.tensor(-.2))
    def forward(self,x):
        raw=self.linear(x).squeeze(-1).tanh()
        return SimpleNamespace(raw_similarity=raw,calibrated_logit=self.scale.exp()*raw+self.bias)


class TinyModel(nn.Module):
    def __init__(self,feature_trainable=True,dropout=0.):
        super().__init__()
        self.base_model=nn.Linear(2,2).requires_grad_(False)
        self.scorer_stem=nn.Linear(2,4).requires_grad_(feature_trainable)
        self.score_head=TinyHead()
        self.dropout=nn.Dropout(dropout)
        self.feature_trainable=feature_trainable
    def scorer_forward(self,x):
        return self.score_head(self.scorer_stem(self.dropout(x)))
    def metadata(self):
        return dict(tiny_test_only=True,feature_trainable=self.feature_trainable)


class TinyRaw:
    def __init__(self,count,prefix="train"):
        self.count,self.prefix=count,prefix
    def __len__(self):return self.count
    def batch(self,indices,device="cpu"):
        ids=torch.tensor(list(indices),device=device)
        x=torch.stack(((ids.float()-self.count/2)/self.count,(ids%3).float()/3),1)
        return train.RawBatch((x,),(ids%2).float(),ids%3!=0,torch.ones_like(ids,dtype=torch.bool),
            tuple(self.prefix+str(int(i)) for i in ids))


class TinyAux:
    counts=(1,2,1)
    def __len__(self):return len(self.counts)
    def __getitem__(self,index):
        n=1+self.counts[index]
        x=torch.tensor([[.5,-.2],[.7,.3],[-.9,.8]])[:n]+index*.1
        return AuxiliaryBatch((x,),torch.tensor([1.]+[0.]*(n-1)),
            torch.tensor([[0,i] for i in range(1,n)],dtype=torch.long),("anchor",)*n,True,{})


def same_tree(test,a,b):
    if torch.is_tensor(a):test.assertTrue(torch.equal(a,b))
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:same_tree(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):same_tree(test,x,y)
    else:test.assertEqual(a,b)


def tiny_settings():
    stack=ExitStack()
    for key,value in dict(BATCH=2,TRAIN_COUNT=8,SEGMENT_SIZE=4,HEAD_EPOCHS=2,
                           SEGMENTS_PER_EPOCH=2,TOTAL_SEGMENTS=4).items():
        stack.enter_context(patch.object(train,key,value))
    return stack


def fake_evaluate(model,validation,device):
    # Exercise save/restore around validation that happens to consume RNG.
    random.random();np.random.rand();torch.rand(2)
    score=float(next(model.score_head.parameters()).detach().sum())
    return dict(sample_count=8,decision_valid_count=8,max_f1_selection_key=[score,.4],
        recall95_selection_key=[score,.4],operating_points=dict(thresholds=dict(max_f1=.5,recall_95=.3))),[]


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_full_plan_aux_slots_and_exact_exposures(self):
        self.assertEqual(len(train.plan()),64)
        self.assertEqual(sum(p["count"] for p in train.plan()),384000)
        counts=[1]*1382+[2]*88
        for e in range(1,17):
            slots=train.auxiliary_schedule(e,1470)
            self.assertEqual(len(slots),1500)
            self.assertEqual(slots.count(None),30)
            self.assertEqual(sorted(x for x in slots if x is not None),list(range(1470)))
            self.assertEqual(tuple(x for s in range(4) for x in slots[s*375:(s+1)*375]),slots)
        for segments,epochs in ((4,1),(32,8),(64,16)):
            ledger=train.exposure_ledger(segments,counts)
            self.assertEqual(ledger,dict(ordinary_pairs=24000*epochs,auxiliary_groups=1470*epochs,
                auxiliary_pairs=3028*epochs,pair_forwards=27028*epochs,
                no_aux_updates=30*epochs,optimizer_updates=1500*epochs))

    def test_optimizer_only_scorer_and_stem_lr_ratio(self):
        for adapt in (False,True):
            model=TinyModel(adapt)
            optimizer=train.create_optimizer(model)
            self.assertEqual([g["name"] for g in optimizer.param_groups],["head","stem"] if adapt else ["head"])
            self.assertFalse(any(p is q for g in optimizer.param_groups for p in g["params"]
                                 for q in model.base_model.parameters()))
            for e,rate in ((3,1e-4),(4,2e-5),(16,2e-5)):
                train.set_learning_rate(optimizer,e)
                self.assertEqual(optimizer.param_groups[0]["lr"],rate)
                if adapt:self.assertEqual(optimizer.param_groups[1]["lr"],rate*.1)

    def test_sequential_backward_matches_exact_combined_formula(self):
        torch.manual_seed(71)
        actual=TinyModel();expected=deepcopy(actual)
        dataset,aux=TinyRaw(16),TinyAux()
        optimizer=torch.optim.SGD((p for p in actual.parameters() if p.requires_grad),lr=.02)
        other=torch.optim.SGD((p for p in expected.parameters() if p.requires_grad),lr=.02)
        report=train.train_segment(actual,dataset,aux,list(range(16)),[1],optimizer,"cpu")
        ordinary=dataset.batch(range(16));group=aux[1]
        out=expected.scorer_forward(*ordinary.inputs)
        rank=expected.scorer_forward(*group.inputs).raw_similarity
        loss=train.pair_loss(out.calibrated_logit,ordinary.labels,ordinary.training_valid)+.3*torch.relu(.15-rank[0]+rank[1:].max())
        loss.backward();torch.nn.utils.clip_grad_norm_((p for p in expected.parameters() if p.requires_grad),5.)
        other.step()
        same_tree(self,actual.state_dict(),expected.state_dict())
        self.assertEqual((report["ordinary_pairs"],report["auxiliary_pairs"],report["optimizer_updates"]),(16,3,1))
        self.assertAlmostEqual(report["mean_objective_per_update"],float(loss),places=6)
        self.assertTrue(all(p.grad is None for p in actual.base_model.parameters()))

    def test_no_aux_has_only_ordinary_bce_and_raw_rank_ignores_affine(self):
        model=TinyModel();aux=TinyAux()
        opt=train.create_optimizer(model)
        report=train.train_segment(model,TinyRaw(16),aux,list(range(16)),[None],opt,"cpu")
        self.assertEqual((report["auxiliary_pairs"],report["auxiliary_groups"],report["no_aux_updates"]),(0,0,1))
        self.assertEqual(report["rank_sum"],0.)
        group=aux[1];out=model.scorer_forward(*group.inputs)
        raw=train.raw_group_rank(out.raw_similarity,group)
        with torch.no_grad():model.score_head.bias.add_(100)
        self.assertTrue(torch.equal(raw,train.raw_group_rank(model.scorer_forward(*group.inputs).raw_similarity,group)))
        torch.testing.assert_close(raw,train.raw_group_rank(out.raw_similarity+3.,group),atol=1e-6,rtol=0)

    def test_pair_loss_preserves_all16_denominator(self):
        z=torch.zeros(16,requires_grad=True);valid=torch.zeros(16,dtype=torch.bool);valid[0]=True
        value=train.pair_loss(z,torch.ones(16),valid)
        value.backward()
        self.assertAlmostEqual(float(value),np.log(2)/16,places=7)
        self.assertEqual(z.grad.tolist(),[-.5/16]+[0.]*15)

    def test_validation_recomputes_features_and_calibrates_all_simval_rows(self):
        # Test-only AP substitute when this local CPU runtime lacks sklearn;
        # use the project's exact tie-aware AP, never invented metric values.
        with ExitStack() as stack:
            if importlib.util.find_spec("sklearn") is None:
                from staging.pairwise_v0_2.training.metrics import _ranking_metrics
                sklearn,skmetrics=ModuleType("sklearn"),ModuleType("sklearn.metrics")
                skmetrics.average_precision_score=lambda y,s:_ranking_metrics(list(s),list(y))[1]
                sklearn.metrics=skmetrics
                stack.enter_context(patch.dict(sys.modules,{"sklearn":sklearn,"sklearn.metrics":skmetrics}))
            model=TinyModel()
            with patch.object(model.base_model,"forward",side_effect=AssertionError("Matcher invoked")):
                report,rows=train.evaluate(model,TinyRaw(18,"val"),"cpu")
            self.assertEqual(report["sample_count"],18)
            self.assertEqual(report["positive_count"],9)
            self.assertEqual(report["training_valid_count"],12)
            self.assertEqual(rows[0]["score"],.5)
            self.assertEqual(rows[0]["logit"],0.)
            self.assertIn("raw_similarity",rows[0])
            self.assertIn("recall_95",report["operating_points"]["thresholds"])

    def _new(self):
        random.seed(91);np.random.seed(92);torch.manual_seed(93)
        m=TinyModel(dropout=.25)
        ident=dict(frozen_digests=train.frozen_digests(m),negative_counts=list(TinyAux.counts),tiny_test_only=True)
        return m,train.create_optimizer(m),ident

    def test_committed_resume_is_bitwise_equal_after_uncommitted_updates(self):
        with tempfile.TemporaryDirectory() as directory,tiny_settings(),patch.object(train,"evaluate",side_effect=fake_evaluate),redirect_stdout(io.StringIO()):
            root=Path(directory);full=root/"full";interrupted=root/"interrupted";full.mkdir();interrupted.mkdir()
            dataset,validation,aux=TinyRaw(8),TinyRaw(8,"val"),TinyAux()
            m,opt,ident=self._new()
            args=SimpleNamespace(output=str(full),resume=False,stop_after_head_epoch=2,arm="G1")
            train.execute(args,m,opt,dataset,validation,aux,ident,"cpu")
            expected_model=deepcopy(m.state_dict());expected_adam=deepcopy(opt.state_dict())
            expected_rng=deepcopy(train.capture_rng_state())
            m,opt,ident2=self._new();self.assertEqual(ident,ident2)
            args.output=str(interrupted)
            original=train.train_segment;calls=[]
            def interrupt_after_updates(*a,**kw):
                report=original(*a,**kw);calls.append(1)
                if len(calls)==2:raise RuntimeError("injected after optimizer update before commit")
                return report
            with patch.object(train,"train_segment",side_effect=interrupt_after_updates),self.assertRaisesRegex(RuntimeError,"injected"):
                train.execute(args,m,opt,dataset,validation,aux,ident,"cpu")
            saved=torch.load(interrupted/"last.pt",weights_only=False)
            self.assertEqual(saved["completed_segments"],1)
            m,opt,ident3=self._new();args.resume=True
            random.random();np.random.rand();torch.rand(3) # restored only after initialization
            result=train.execute(args,m,opt,dataset,validation,aux,ident3,"cpu")
            self.assertEqual(result["status"],"complete")
            same_tree(self,expected_model,m.state_dict());same_tree(self,expected_adam,opt.state_dict())
            same_tree(self,expected_rng,train.capture_rng_state())
            # Already-complete resume does no updates and still publishes valid status.
            result=train.execute(args,m,opt,dataset,validation,aux,ident3,"cpu")
            self.assertEqual(result["exposures"]["ordinary_pairs"],16)

    def test_restore_rejects_frozen_weights_parameter_order_and_moments(self):
        with tiny_settings():
            model,optimizer,ident=self._new()
            train.train_segment(model,TinyRaw(8),TinyAux(),list(range(4)),
                train.auxiliary_schedule(1,3)[:2],optimizer,"cpu")
            payload=deepcopy(train.checkpoint_payload(model,optimizer,ident,1,{},"segment_recovery"))
            cases=[]
            broken=deepcopy(payload);broken["model_state_dict"]["base_model.weight"][0,0]+=1;cases.append(broken)
            broken=deepcopy(payload);broken["optimizer_state_dict"]["param_groups"][0]["params"].reverse();cases.append(broken)
            broken=deepcopy(payload);broken["optimizer_state_dict"]["param_groups"][1]["lr"]*=10;cases.append(broken)
            broken=deepcopy(payload);next(iter(broken["optimizer_state_dict"]["state"].values()))["exp_avg"].fill_(float("nan"));cases.append(broken)
            broken=deepcopy(payload);broken["exposures"]["auxiliary_pairs"]+=1;cases.append(broken)
            for bad in cases:
                target,opt,_=self._new()
                with self.assertRaises(ValueError):train.restore(target,opt,bad,ident)
            g0=TinyModel(False);ident0=dict(frozen_digests=train.frozen_digests(g0),negative_counts=list(TinyAux.counts))
            payload0=train.checkpoint_payload(g0,train.create_optimizer(g0),ident0,0,{},"initial")
            payload0["model_state_dict"]["scorer_stem.weight"][0,0]+=1
            with self.assertRaisesRegex(ValueError,"frozen"):
                train.restore(g0,train.create_optimizer(g0),payload0,ident0)

    def test_raw_loader_uses_only_validity_and_checks_six_input_identity(self):
        samples=[sample(entry("p","A","B",True)),sample(entry("n","A","C",False))]
        class Dataset:
            split="train"
            def __len__(self):return len(samples)
            def __getitem__(self,i):return samples[i]
        batch=collate_rachel_pairs(samples)
        records=[]
        for i,pair_id in enumerate(batch.pair_ids):
            h=hashlib.sha256()
            for k in train.source_cache.INPUTS:
                h.update(k.encode());h.update(getattr(batch,k)[i].tobytes())
            records.append(dict(pair_id=pair_id,label=float(batch.labels[i]),input_sha256=h.hexdigest()))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"train.json";path.write_text("fixture manifest")
            class ValidOnly(dict):
                def __getitem__(self,k):
                    if k not in ("training_valid","decision_valid"):raise AssertionError("cached feature accessed")
                    return super().__getitem__(k)
            cache=SimpleNamespace(split="train",records=records,
                arrays=ValidOnly(training_valid=np.array([True,False]),decision_valid=np.array([True,False])),
                binding=dict(population=dict(manifest_sha256=train.cache_data.sha(path))))
            raw=train.RawPopulation(Dataset(),cache,path)
            out=raw.batch([1,0])
            self.assertEqual(len(out.inputs),6)
            self.assertEqual(out.labels.tolist(),[0.,1.]);self.assertEqual(out.training_valid.tolist(),[False,True])
            self.assertFalse(hasattr(raw,"arrays"))
            wrong=deepcopy(cache);wrong.records[0]["input_sha256"]="f"*64
            with self.assertRaisesRegex(ValueError,"fingerprint"):
                train.RawPopulation(Dataset(),wrong,path).batch([0])

    def test_cli_smoke_and_formal_status_contract(self):
        args=train.parser().parse_args(["--arm","G1","--train-cache","t","--val-cache","v",
            "--availability","a","--output","o","--smoke","32"])
        self.assertEqual(args.smoke,32);self.assertEqual(args.stop_after_head_epoch,16)
        self.assertFalse(hasattr(args,"real"));self.assertFalse(hasattr(args,"ood"))
        args.resume=True
        with self.assertRaisesRegex(ValueError,"discard"):
            train.run(args)


if __name__=="__main__":unittest.main()
