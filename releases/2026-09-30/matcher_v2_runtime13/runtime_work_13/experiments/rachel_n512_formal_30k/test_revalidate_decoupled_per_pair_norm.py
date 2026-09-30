"""CPU synthetic contract tests; no actual C epochs, model inference, or GPU."""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.rachel_n512_formal_30k import revalidate_decoupled_per_pair_norm as revalidate
from experiments.rachel_n512_formal_30k import train_score_decoupled as trainer
from experiments.rachel_n512_formal_30k.test_train_score_decoupled import model, identity, tiny_loader
from experiments.rachel_n512_formal_30k.run_layout_decoder_experiment import classification, fit_threshold
from experiments.rachel_n512_formal_30k.evaluate_score_decoupled import load_frozen_model
from staging.pairwise_v0_2.models.rachel_decoupled_score import build_decoupled_score_model
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig


@contextmanager
def raises(exception, text):
    with unittest.TestCase().assertRaisesRegex(exception,text):yield


def tensor_tree_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(tensor_tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(tensor_tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def fixture(root):
    source=root/"source_v2";source.mkdir()
    dataset=root/"dataset";(dataset/"pairs").mkdir(parents=True)
    # IDs/counts are protocol fixtures, not a fabricated experimental result.
    expected=[dict(pair_id="unit-val-%04d"%i,label=i<1500) for i in range(3000)]
    manifest=dataset/"pairs/val.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in expected)+"\n")
    tiny=model()
    net=build_decoupled_score_model(replace(tiny.config,contour_cap=512),head_kind="matrix_cnn",
        matrix_head_revision="bn_relu_pool_v2").set_phase("classifier")
    ident=identity(net);ident["contour_cap"]=512
    ident["populations"]["val"]={"manifest_sha256":revalidate.sha(manifest)}
    receipt=trainer.matcher_receipt(net,ident)
    for epoch in revalidate.EPOCHS:
        # Make each epoch's copied learned parameters identifiable, without training.
        with torch.no_grad():net.score_head.fc.bias.fill_(epoch/100.)
        payload=trainer.checkpoint_payload(net,trainer.create_optimizer(net),identity=ident,
            loss_config=RachelN512LossConfig(),completed=4*epoch,receipt=receipt,
            winners={"old_threshold_must_not_survive":.999999},role="synthetic_fixture_only")
        torch.save(payload,source/("epoch_%03d.pt"%epoch))
    trainer.save_json(source/"classifier_freezes/freeze.json",dict(schema_version=trainer.SCHEMA,status="complete",
        eligible_epoch_range=[13,20],held_out_used_for_fit=False,resume_identity=ident,
        resume_identity_sha256=trainer.canonical_digest(ident),selections={"old_wrong":True}))
    plan=revalidate.make_plan(source,root/"converted_v3",dataset,batch_size=1,workers=0,device="cpu")
    return plan,expected


def evaluated_rows(expected,epoch):
    labels=np.asarray([r["label"] for r in expected],bool)
    # Epoch17 is perfect; all other epochs lose increasing positives. This tests
    # genuinely new selection, never the source's deliberately invalid winners.
    scores=np.full(3000,.3)
    scores[:1500]=.8
    if epoch!=17:scores[:(epoch-12)*10]=.2
    rows=[dict(**r,decision_valid=True,classification=dict(coarse=.5,local=float(score),fused=float(score)))
        for r,score in zip(expected,scores)]
    thresholds={k:fit_threshold(labels,np.asarray([r["classification"][k] for r in rows])) for k in ("coarse","local","fused")}
    methods={k:classification(labels,np.asarray([r["classification"][k] for r in rows]),thresholds[k]) for k in thresholds}
    report=dict(sample_count=3000,positive_count=1500,negative_count=1500,decision_coverage=1.,
        methods=methods,thresholds=thresholds,selection_key=[methods["fused"]["f1"],methods["fused"]["auprc"]],
        threshold_grain="equal pair rows; no cluster reweighting",pose_used_for_selection=False)
    return report,rows


def fixture_operating_points(labels,scores):
    """Dependency-light test double; production calls registered sklearn-backed fit."""
    labels,scores=np.asarray(labels,bool),np.asarray(scores,float)
    thresholds={"max_f1":fit_threshold(labels,scores)}
    for recall in (.9,.95,.98,.99):
        thresholds["recall_%d"%int(recall*100)]=float(np.sort(scores[labels])[::-1][int(np.ceil(recall*labels.sum()))-1])
    thresholds["recall_first"]=min(thresholds["max_f1"],thresholds["recall_99"])
    validation={k:classification(labels,scores,t) for k,t in thresholds.items()}
    return dict(thresholds=thresholds,validation=validation,
        selection_key=[validation["recall_95"]["precision"],validation["recall_95"]["auprc"]])


def test_explicit_conversion_preserves_all_parameters_and_archives_training_state(tmp_path):
    plan,_=fixture(Path(tmp_path));src=plan["provenance"]["source_checkpoints"]["13"]
    original=torch.load(src["path"],map_location="cpu",weights_only=False)
    saved=deepcopy(original)
    converted=revalidate.convert_payload(original,source_path=src["path"],source_sha256=src["sha256"],identity=plan["identity"])
    assert tensor_tree_equal(original,saved)
    before=trainer.load_decoupled_checkpoint(original);after=trainer.load_decoupled_checkpoint(converted)
    assert revalidate.parameter_digest(before)==revalidate.parameter_digest(after)
    assert len(converted["conversion_provenance"]["removed_buffers"])==9
    assert converted["inference_only"] and converted["new_training_pair_exposures"]==0
    assert converted["global_exposure"]==original["global_exposure"]==13*24000
    assert converted["executed_pair_exposures"]==0 and converted["winners"]=={}
    assert "optimizer_state_dict" not in converted and "rng_state" not in converted and "runtime_batching" not in converted
    assert tensor_tree_equal(converted["source_training_state"]["optimizer_state_dict"],original["optimizer_state_dict"])


def test_all_eight_VAL_reselected_resume_reuses_only_new_receipts_and_existing_evaluator_loads(tmp_path):
    torch.set_num_threads(1)
    plan,expected=fixture(Path(tmp_path));source=Path(plan["source_root"])
    before={str(p.relative_to(source)):revalidate.sha(p) for p in source.rglob("*") if p.is_file()}
    calls=[]
    def evaluate(payload,epoch):
        assert payload["inference_only"] and payload["winners"]=={}
        assert payload["decoupled_score"]["matrix_head_revision"]=="per_pair_norm_v3"
        calls.append(epoch);return evaluated_rows(expected,epoch)
    with patch.object(revalidate,"fit_operating_points",fixture_operating_points):
        first=revalidate.execute_epochs(plan,evaluate,stop_after_epoch=15)
    assert first["status"]=="stopped_at_epoch" and calls==[13,14,15]
    with patch.object(revalidate,"fit_operating_points",fixture_operating_points):
        second=revalidate.execute_epochs(plan,evaluate,resume=True)
    assert calls==list(range(13,21)) and second["reused_validation_epochs"]==[13,14,15]
    assert second["status"]=="complete" and second["new_training_pair_exposures"]==0
    assert second["frozen_selections"]==dict(max_f1=17,recall95=17,fixed_epoch=20)
    for selection,epoch in second["frozen_selections"].items():
        net,meta=load_frozen_model(plan["output"],selection)
        assert meta["epoch"]==epoch and net.matrix_head_revision=="per_pair_norm_v3"
        assert meta["training_identity"]["normalization_conversion"]["new_training_pair_exposures"]==0
    assert before=={str(p.relative_to(source)):revalidate.sha(p) for p in source.rglob("*") if p.is_file()}
    again=revalidate.execute_epochs(plan,lambda *_: (_ for _ in ()).throw(AssertionError("must not reevaluate")),resume=True)
    assert again["reused_validation_epochs"]==list(range(13,21))
    report=Path(plan["output"])/"validation_013_rows.json"
    report.write_text("[]")
    with raises(ValueError,"receipt"):
        revalidate.execute_epochs(plan,evaluate,resume=True)


def test_cli_and_complete_population_mismatch_refuse(tmp_path):
    args=revalidate.parser().parse_args(["--source-training-run","s","--output","n"])
    assert not args.resume and args.batch_size==1 and args.stop_after_epoch==20
    plan,expected=fixture(Path(tmp_path));report,rows=evaluated_rows(expected,13)
    with raises(ValueError,"complete SIM VAL"):
        revalidate.validate_rows(report,rows[:-1],plan)
    with raises(FileExistsError,"new output"):
        revalidate.execute_epochs({**plan,"output":plan["source_root"]},lambda *_:None)
    rows[0]["decision_valid"]=False;report["decision_coverage"]=2999/3000
    revalidate.validate_rows(report,rows,plan)


def test_shared_one_base_scores_exactly_match_original_core_per_head_validation():
    torch.set_num_threads(1)
    v2=model(); nets={}
    for epoch in (13,20):
        net=build_decoupled_score_model(v2.config,head_kind="matrix_cnn",phase="classifier",
            matrix_head_revision="per_pair_norm_v3")
        net.base_model.load_state_dict(v2.base_model.state_dict(),strict=True)
        with torch.no_grad():net.score_head.fc.bias.fill_(epoch/10.)
        nets[epoch]=net
    batches=[wrapped.batch for wrapped in tiny_loader(4,micro=2,balanced=True)]
    expected={epoch:trainer.evaluate_pair_validation(net,batches,torch.device("cpu")) for epoch,net in nets.items()}
    counts=[]
    handle=nets[13].base_model.register_forward_hook(lambda *_:counts.append(1))
    actual=revalidate.evaluate_shared_validation(nets,batches,torch.device("cpu"));handle.remove()
    assert len(counts)==len(batches)
    assert actual==expected
    with torch.no_grad():next(nets[20].base_model.parameters()).add_(.1)
    with raises(ValueError,"different frozen bases"):
        revalidate.evaluate_shared_validation(nets,batches,torch.device("cpu"))
