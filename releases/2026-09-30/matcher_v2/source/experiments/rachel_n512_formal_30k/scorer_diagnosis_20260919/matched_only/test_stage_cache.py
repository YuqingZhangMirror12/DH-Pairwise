"""CPU-only stage evidence/scorer integration; tiny synthetic caches only."""
from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import cache,data,model,stage_cache,train
from . import test_training as training_tests


def put_geometry(root,row,offsets,weights=None):
    offsets=np.asarray(offsets,np.float32)
    if offsets.ndim==1: offsets=np.column_stack([offsets,np.zeros(len(offsets),np.float32)])
    n=len(offsets)
    a=torch.zeros(1,512,2)
    a[0,:n]=torch.tensor(np.column_stack([np.arange(n)*50,np.arange(n)*70]),dtype=torch.float32)
    b=a.clone(); b[0,:n]+=torch.tensor(offsets)
    valid=torch.zeros(1,512,dtype=torch.bool);valid[:,:n]=True
    q=torch.zeros(1,512,512)
    q[0,torch.arange(n),torch.arange(n)]=torch.tensor(weights if weights is not None else np.ones(n),dtype=torch.float32)
    selected=model.select_predicted_inliers(q,a,b,valid,valid)
    values=dict(points_a=a[0],points_b=b[0],valid_a=valid[0],valid_b=valid[0])
    for name in ("candidate_indices","candidate_valid","candidate_inliers","mask_a","mask_b",
                 "translation_a_to_b_rc","layout_valid"):
        values[name]=getattr(selected,name)[0]
    ij=selected.candidate_indices[0].clamp_min(0)
    values["candidate_weights"]=torch.where(selected.candidate_valid[0],q[0,ij[:,0],ij[:,1]],torch.zeros(512))
    for name,value in values.items():
        arr=np.load(root/(name+".npy"),mmap_mode="r+");arr[row]=value.numpy();arr.flush()
    return selected


class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Same documented test-only dependency-light AP substitution if local
        # sklearn is unavailable; formal server always uses original sklearn.
        training_tests.TrainingTests.setUpClass.__func__(cls)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.count_patch=patch.dict(data.FULL_COUNTS,{"train":4,"val":4})
        self.count_patch.start();self.addCleanup(self.count_patch.stop)
        self.base=self.root/"base";training_tests.make_cache(self.base)

    def prepare(self):
        source=data.FormalCache(self.base,"train")
        side=self.root/"stages"
        stage_cache.prepare(source,side)
        return source,side

    def test_one_time_side_cache_gt_free_no_neural_forward_and_final_replay(self):
        put_geometry(self.base,0,[-1,0,1,59,60,61,119,120,121],
                     [1.,.9,.8,.7,.6,.5,.4,.3,.2])
        # Explicitly prohibit ANY model forward during side-cache generation.
        with patch.object(model.FreshMatchedScorer,"forward",side_effect=AssertionError("no NN")):
            source,side=self.prepare()
        loaded=stage_cache.StageCache(side,source)
        groups=loaded.batch([0,1],"edge_multi")
        self.assertEqual(groups.eligible[0].sum().item(),3)
        self.assertEqual(groups.eligible[1].sum().item(),0)
        torch.testing.assert_close(groups.candidate_inliers[0,0],torch.from_numpy(np.array(source.arrays["candidate_inliers"][0])))
        torch.testing.assert_close(groups.translation_rc[0,0],torch.from_numpy(np.array(source.arrays["translation_a_to_b_rc"][0])))
        fields=inspect.signature(stage_cache.candidate_groups.build_candidate_groups).parameters
        self.assertFalse(set(fields)&{"gt","labels","label","features_a","assignment"})
        protocol=json.loads((side/"protocol.json").read_text())
        self.assertFalse(protocol["labels_used"]);self.assertFalse(protocol["neural_inference"])
        self.assertEqual(len(protocol["array_sha256"]),len(stage_cache.ARRAYS))

    def test_seed_is_unrefined_not_first_mode(self):
        put_geometry(self.base,0,[-9,0,9,-11],[2,1,1,.2])
        source,side=self.prepare(); loaded=stage_cache.StageCache(side,source)
        seed,multi=loaded.batch([0],"edge_seed"),loaded.batch([0],"edge_multi")
        torch.testing.assert_close(seed.translation_rc[0,0],torch.tensor([0.,0.]))
        self.assertFalse(torch.equal(seed.candidate_inliers,multi.candidate_inliers[:,:1]))
        self.assertGreater(float(torch.linalg.vector_norm(seed.translation_rc[0,0]-multi.translation_rc[0,0])),6.)

    def test_three_edge_arms_same_initialization_parameters_and_first_mode_logit(self):
        put_geometry(self.base,0,[-1,0,1,59,60,61],[1,.9,.8,.5,.4,.3])
        source,side=self.prepare()
        final=model.make_fresh_scorer("matched_edges",seed=train.HEAD_SEED).eval()
        seed=train.make_scorer("edge_seed").eval(); multi=train.make_scorer("edge_multi").eval()
        for obj in (seed,multi):
            self.assertEqual(obj.metadata()["parameters"],final.metadata()["parameters"])
            for key,value in final.state_dict().items():
                self.assertTrue(torch.equal(value,obj.edge_head.state_dict()[key]),key)
        batch=source.batch([0,1])
        reference=final(*batch.model_args,**batch.model_kwargs)
        grouped=data.CandidateStageCache(source,side,"edge_multi").batch([0,1])
        actual=multi(*grouped.model_args,**grouped.model_kwargs)
        torch.testing.assert_close(actual.group_logits[0,0],reference.logit[0])
        self.assertTrue(actual.used_fallback[1]);self.assertFalse(actual.used_fallback[0])
        self.assertEqual(actual.selected_group_rank[1].item(),0)

    def test_shared_ca_max_readout_bce_only_winning_group_and_negative_not_forced_positive(self):
        put_geometry(self.base,0,[-1,0,1,59,60,61,119,120,121],[1,.9,.8,.7,.6,.5,.4,.3,.2])
        source,side=self.prepare()
        batch=data.CandidateStageCache(source,side,"edge_multi").batch([0])
        net=train.make_scorer("edge_multi")
        out=net(*batch.model_args,**batch.model_kwargs)
        out.group_logits.retain_grad()
        valid_logits=out.group_logits[0,out.group_eligible[0]]
        self.assertEqual(float(out.logit[0]),float(valid_logits.max()))
        self.assertEqual(batch.labels.tolist(),[0.]) # wrong proposals do not rewrite negative label
        train.pair_loss(out.logit,batch.labels,batch.training_valid).backward()
        tied=valid_logits.detach()==valid_logits.detach().max()
        got=out.group_logits.grad[0,out.group_eligible[0]]
        self.assertTrue((got[~tied]==0).all())
        self.assertTrue((got[tied]>0).all()) # negative label pushes max DOWN
        for p in net.edge_head.head.classifier.parameters():
            p.data.zero_()
        net.edge_head.head.classifier[-1].bias.data.fill_(-10.)
        rejected=net(*batch.model_args,**batch.model_kwargs)
        self.assertTrue(rejected.has_decoded_candidate[0])
        self.assertLess(float(rejected.logit.sigmoid()),.001)
        rejected.group_logits.retain_grad()
        train.pair_loss(rejected.logit,batch.labels,batch.training_valid).backward()
        eligible=rejected.group_eligible[0]
        expected=rejected.logit.sigmoid().detach()/eligible.sum()
        torch.testing.assert_close(rejected.group_logits.grad[0,eligible],expected.expand(int(eligible.sum())))

    def test_empty_and_insufficient_fallback_learns_without_global_rescue(self):
        put_geometry(self.base,0,[5,6],[1,.9])
        source,side=self.prepare()
        for arm in ("edge_seed","edge_multi"):
            batch=data.CandidateStageCache(source,side,arm).batch([0,1])
            net=train.make_scorer(arm)
            out=net(*batch.model_args,**batch.model_kwargs)
            self.assertTrue(out.used_fallback.all())
            if arm=="edge_seed":
                self.assertTrue(out.group_present[0,0])
                self.assertEqual(stage_cache.STATUS_NAMES[int(out.group_status_code[0,0])],"insufficient_inliers")
            # Different pair labels remain separate. Use positive weighted
            # target to make the scalar's training direction unambiguous.
            train.pair_loss(out.logit,torch.ones(2),batch.training_valid).backward()
            self.assertLess(float(net.edge_head.head.no_evidence_logit.grad),0.)
            self.assertTrue(all(p.grad is None for name,p in net.named_parameters()
                                if name!="edge_head.head.no_evidence_logit"))

    def test_invalid_final_can_have_secondary_group_without_relabeling(self):
        put_geometry(self.base,0,[0,1,80,81,82],[1,1,.1,.1,.1])
        source,side=self.prepare()
        batch=data.CandidateStageCache(source,side,"edge_multi").batch([0])
        out=train.make_scorer("edge_multi")(*batch.model_args,**batch.model_kwargs)
        self.assertFalse(out.has_decoded_candidate[0])
        self.assertTrue(out.has_selected_candidate_group[0]);self.assertFalse(out.used_fallback[0])
        self.assertEqual(batch.labels.tolist(),[0.])

    def test_stage_batched_training_and_evaluation_have_group_ranks_and_nullable_logits(self):
        put_geometry(self.base,0,[-1,0,1,59,60,61],[1,.9,.8,.5,.4,.3])
        source,side=self.prepare()
        for arm in ("edge_seed","edge_multi"):
            ds=data.CandidateStageCache(source,side,arm)
            net=train.make_scorer(arm);opt=train.create_optimizer(net)
            report=train.train_segment(net,ds,np.tile(np.arange(4),4),opt,"cpu")
            self.assertEqual(report["optimizer_updates"],1)
            validation,rows=train.evaluate(net,ds,"cpu")
            self.assertEqual(validation["sample_count"],4)
            self.assertIn("groups",rows[0]);self.assertGreater(rows[0]["selected_group_rank"],0)
            self.assertEqual(rows[1]["selected_group_rank"],0)
            self.assertTrue(all(row["logit"] is None for row in rows[1]["groups"]))
            json.dumps(rows,allow_nan=False)
            with patch.object(train,"SEGMENT_SIZE",16):
                saved=train.checkpoint_payload(net,opt,{"stage_test":arm},1,{},"segment_recovery")
                copied=train.make_scorer(arm);other=train.create_optimizer(copied)
                self.assertEqual(train.restore(copied,other,saved,{"stage_test":arm})[0],1)

    def test_no_per_epoch_proposals_and_labels_never_enter_model(self):
        source,side=self.prepare()
        ds=data.CandidateStageCache(source,side,"edge_multi")
        with patch.object(stage_cache.candidate_groups,"build_candidate_groups",side_effect=AssertionError("epoch replay forbidden")):
            batch=ds.batch([0,1]);net=train.make_scorer("edge_multi")
            net(*batch.model_args,**batch.model_kwargs)
        with self.assertRaises(TypeError):
            net(*batch.model_args,**dict(batch.model_kwargs,label=batch.labels))
        bad=replace(batch.model_kwargs["groups"],eligible=torch.ones(2,5,dtype=torch.bool))
        with self.assertRaises(ValueError):
            net(*batch.model_args,**dict(batch.model_kwargs,groups=bad))

    def test_partial_probes_source_or_array_mutations_fail_closed(self):
        source=data.FormalCache(self.base,"train")
        probe=self.root/"probe";stage_cache.prepare(source,probe,limit=2)
        with self.assertRaises(ValueError):stage_cache.StageCache(probe,source)
        side=self.root/"stages";stage_cache.prepare(source,side)
        mutable=np.load(side/"ranks.npy",mmap_mode="r+");mutable[0,0]=3;mutable.flush()
        with self.assertRaisesRegex(ValueError,"hash"):stage_cache.StageCache(side,source)
        with self.assertRaises(ValueError):stage_cache.prepare(source,self.base/"nested")

    def test_all_existing_arms_and_stage_identifiers_are_explicit(self):
        self.assertEqual(train.ARMS,("all_tokens","matched_tokens","matched_edges","edge_seed","edge_multi"))
        for arm in model.ARMS:
            self.assertIsInstance(train.make_scorer(arm),model.FreshMatchedScorer)
        self.assertEqual(train.make_scorer("edge_multi").metadata()["parameters"]["total"],197699)
        self.assertEqual(train.make_scorer("edge_seed").metadata()["groups"],1)


if __name__=="__main__":unittest.main()
