"""Explicit S3 v2 trajectory conversion and fresh SIM VAL selection, no training.

C13..20 learned parameters are copied exactly into per_pair_norm_v3; only the
nine head BN running buffers are removed. Old evaluation scores, thresholds,
and winners are never reused. Converted checkpoints are inference-only. Their
legacy epoch/exposure fields identify the inherited source trajectory, not new
optimization. Original optimizer/RNG/runtime state is retained only as source
provenance, never as active resumable training state.

Use a NEW --output; --resume reuses only this runner's completed, hash-bound
VAL receipts. No TEST, REAL, OOD, pose selection, training, or queue mutation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import numpy as np

from experiments.rachel_n512_formal_30k import train_score_decoupled as trainer

SCHEMA = "rachel-decoupled-per-pair-norm-revalidation/1"
SOURCE_REVISION, TARGET_REVISION = "bn_relu_pool_v2", "per_pair_norm_v3"
EPOCHS = tuple(range(13, 21))


def fit_operating_points(labels, scores):
    # Remote formal environment provides sklearn; conversion/CPU shape checks
    # do not need to import that optional analytical runtime at module import.
    from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points as fit
    return fit(labels,scores)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_source_identity(identity):
    if (identity.get("head_kind") != "matrix_cnn" or identity.get("matrix_head_revision") != SOURCE_REVISION
            or identity.get("sampling") != "original512" or identity.get("contour_cap") != 512):
        raise ValueError("only the registered original512 S3 matrix bn_relu_pool_v2 trajectory is accepted")


def make_plan(source_root, output, dataset_root, *, batch_size=1, workers=4, device="cuda:0"):
    source_root, output = Path(source_root).resolve(strict=True), Path(output).resolve()
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError("source and new output must be disjoint; source artifacts are read-only")
    freeze_path = source_root / "classifier_freezes/freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if (freeze.get("schema_version") != trainer.SCHEMA or freeze.get("status") != "complete"
            or freeze.get("eligible_epoch_range") != [13, 20] or freeze.get("held_out_used_for_fit") is not False):
        raise ValueError("requires the complete source C13..20 freeze")
    identity = freeze["resume_identity"]
    validate_source_identity(identity)
    if trainer.canonical_digest(identity) != freeze["resume_identity_sha256"]:
        raise ValueError("source freeze identity digest differs")
    manifest = Path(dataset_root).resolve(strict=True) / "pairs/val.jsonl"
    expected = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    ids = [r["pair_id"] for r in expected]
    if (len(expected) != 3000 or len(set(ids)) != 3000 or sum(bool(r["label"]) for r in expected) != 1500
            or sha(manifest) != identity["populations"]["val"]["manifest_sha256"]):
        raise ValueError("requires unchanged complete balanced original SIM VAL3000")
    sources = {str(epoch): dict(path=str(source_root / ("epoch_%03d.pt" % epoch)),
        sha256=sha(source_root / ("epoch_%03d.pt" % epoch))) for epoch in EPOCHS}
    from staging.pairwise_v0_2.models import rachel_decoupled_score as model_module
    provenance = dict(schema_version=SCHEMA, source_revision=SOURCE_REVISION, target_revision=TARGET_REVISION,
        source_training_run=str(source_root), source_freeze_sha256=sha(freeze_path),
        source_resume_identity_sha256=trainer.canonical_digest(identity), source_checkpoints=sources,
        conversion="same learned parameters; remove only nine head BN running buffers",
        train_forward_gradient_equivalence="per-pair BN batch-stat path preserved; no numerical training rerun claimed",
        new_training_pair_exposures=0, new_optimizer_updates=0,
        inherited_training_epochs=[13,20], inference_only=True,
        source_thresholds_or_winners_reused=False, fresh_selection_population="clean SIM VAL3000 only",
        held_out_used_for_fit=False, GT_layout_used_for_selection=False)
    new_identity = deepcopy(identity)
    new_identity.update(matrix_head_revision=TARGET_REVISION, inference_only=True,
        normalization_conversion=provenance, new_training_pair_exposures=0, new_optimizer_updates=0)
    return dict(schema_version=SCHEMA, source_root=str(source_root), output=str(output),
        source_identity=identity, identity=new_identity, provenance=provenance,
        validation=dict(dataset_root=str(Path(dataset_root).resolve()), manifest=str(manifest),
            manifest_sha256=sha(manifest), count=3000, positive_count=1500,
            pair_labels=[dict(pair_id=r["pair_id"],label=bool(r["label"])) for r in expected]),
        runtime=dict(batch_size=batch_size, workers=workers, device=device, seed=identity["seed"],
            validation_strategy="one shared frozen base pass for unfinished heads; no saved Q cache",
            torch_version=str(torch.__version__), runner_sha256=sha(__file__),
            trainer_sha256=sha(trainer.__file__), model_sha256=sha(model_module.__file__)))


def convert_payload(source, *, source_path, source_sha256, identity):
    """No inference or optimization. Strictly load old/new models and compare state."""
    from staging.pairwise_v0_2.models.rachel_decoupled_score import (
        build_decoupled_score_model, convert_matrix_head_revision_state)
    if (source.get("epoch") not in EPOCHS or source.get("phase") != "classifier"
            or source.get("completed_segments") != source["epoch"] * 4 or source.get("inference_only")):
        raise ValueError("requires an original completed C13..20 trained checkpoint")
    if trainer.canonical_digest(source.get("resume_identity")) != identity["normalization_conversion"]["source_resume_identity_sha256"]:
        raise ValueError("source checkpoint identity differs from original frozen trajectory")
    old = trainer.load_decoupled_checkpoint(source)
    if old.metadata()["matrix_head_revision"] != SOURCE_REVISION:
        raise ValueError("conversion requires explicit v2 source")
    with torch.random.fork_rng(devices=[]):
        new = build_decoupled_score_model(identity["base_model_config"], head_kind="matrix_cnn", matrix_threshold=old.matrix_threshold,
            phase="classifier", matrix_head_revision=TARGET_REVISION)
    new.base_model.load_state_dict(old.base_model.state_dict(), strict=True)
    head_state = convert_matrix_head_revision_state(old.score_head.state_dict(),
        source_revision=SOURCE_REVISION, target_revision=TARGET_REVISION)
    new.score_head.load_state_dict(head_state, strict=True)
    before, after = old.state_dict(), new.state_dict()
    removed = sorted(set(before)-set(after))
    if len(removed) != 9 or set(after)-set(before) or any(not torch.equal(value,before[key]) for key,value in after.items()):
        raise ValueError("conversion changed something beyond nine running buffers")
    if parameter_digest(old) != parameter_digest(new):
        raise ValueError("conversion changed learned parameters")
    converted = deepcopy(source)
    state = {key: converted.pop(key) for key in ("optimizer_state_dict","rng_state","runtime_batching") if key in converted}
    converted.pop("winners",None)
    converted.update(resume_identity=deepcopy(identity), decoupled_score=new.metadata(),
        model_config=deepcopy(identity["base_model_config"]),
        model_state_dict={k:v.detach().cpu().clone() for k,v in after.items()}, winners={},
        inference_only=True, checkpoint_role="normalization_converted_inference_only",
        formal_training_counted=False, source_resume_identity=deepcopy(source["resume_identity"]),
        source_training_state=state, source_runtime_batching=deepcopy(source.get("runtime_batching")),
        source_checkpoint_sha256=source_sha256, source_checkpoint_path=str(source_path),
        inherited_training_pair_exposures=source["global_exposure"],
        source_executed_pair_exposures=source.get("executed_pair_exposures"), executed_pair_exposures=0,
        new_training_pair_exposures=0, new_optimizer_updates=0,
        conversion_provenance=dict(schema_version=SCHEMA, source_revision=SOURCE_REVISION,
            target_revision=TARGET_REVISION, source_checkpoint_sha256=source_sha256,
            source_epoch=source["epoch"], learned_parameters_sha256=parameter_digest(new),
            removed_buffers=removed, learned_parameters_bitwise_equal=True,
            retained_buffers_bitwise_equal=True, old_scores_thresholds_winners_reused=False,
            epoch_exposure_update_fields="inherited source trajectory only, not new training",
            new_training_pair_exposures=0,new_optimizer_updates=0))
    # Existing typed inference loader remains strict and unchanged.
    reloaded = trainer.load_decoupled_checkpoint(converted)
    if parameter_digest(reloaded) != parameter_digest(old):
        raise ValueError("typed new checkpoint roundtrip changed parameters")
    return converted


def validate_rows(report, rows, plan):
    expected=plan["validation"]["pair_labels"]
    if ([dict(pair_id=r["pair_id"],label=bool(r["label"])) for r in rows] != expected
            or report.get("sample_count") != 3000 or report.get("positive_count") != 1500
            or report.get("negative_count") != 1500
            or not all(type(r.get("decision_valid")) is bool for r in rows)
            or report.get("decision_coverage") != sum(r["decision_valid"] for r in rows)/3000):
        raise ValueError("fresh validation did not cover exactly the bound complete SIM VAL3000")


def ensure_converted_checkpoint(plan, epoch):
    source=plan["provenance"]["source_checkpoints"][str(epoch)]
    if sha(source["path"])!=source["sha256"]:raise ValueError("source checkpoint changed after binding")
    path=Path(plan["output"])/("epoch_%03d.pt"%epoch)
    if path.exists():
        payload=torch.load(path,map_location="cpu",weights_only=False)
        if (payload.get("resume_identity")!=plan["identity"] or payload.get("source_checkpoint_sha256")!=source["sha256"]
                or payload.get("epoch")!=epoch or payload.get("inference_only") is not True):
            raise ValueError("uncommitted converted checkpoint differs")
        trainer.load_decoupled_checkpoint(payload)
    else:
        original=torch.load(source["path"],map_location="cpu",weights_only=False)
        payload=convert_payload(original,source_path=source["path"],source_sha256=source["sha256"],identity=plan["identity"])
        trainer.runner._atomic_torch_save(path,payload)
    return payload


def report_from_rows(rows):
    """Exact core VAL calibration/report formula; no pose or auxiliary score."""
    from experiments.rachel_n512_formal_30k.train_realism_data_ablation import (
        fit_threshold, classification, validation_selection_key)
    if len({row["pair_id"] for row in rows})!=len(rows):raise ValueError("VAL IDs are not unique")
    labels=np.asarray([r["label"] for r in rows],bool)
    if not labels.any() or labels.all():raise ValueError("VAL must contain both classes")
    thresholds,methods={},{}
    for branch in ("coarse","local","fused"):
        scores=np.asarray([row["classification"][branch] for row in rows])
        if not np.isfinite(scores).all():raise ValueError("nonfinite validation pair scores")
        thresholds[branch]=fit_threshold(labels,scores)
        methods[branch]=classification(labels,scores,thresholds[branch])
    return dict(sample_count=len(rows),positive_count=int(labels.sum()),negative_count=int((~labels).sum()),
        methods=methods,thresholds=thresholds,decision_coverage=float(np.mean([r["decision_valid"] for r in rows])),
        selection_key=list(validation_selection_key(methods["fused"])),
        threshold_grain="equal pair rows; no cluster reweighting",pose_used_for_selection=False)


def evaluate_shared_validation(models, loader, device):
    """One unchanged base forward per batch, exact C-wrapper semantics per head."""
    if not models:raise ValueError("at least one unfinished head is required")
    base_digests={trainer.state_digest(model.base_model) for model in models.values()}
    if len(base_digests)!=1:raise ValueError("cannot share inference across different frozen bases")
    if any(m.head_kind!="matrix_cnn" or m.phase!="classifier" or m.matrix_head_revision!=TARGET_REVISION for m in models.values()):
        raise ValueError("shared inference accepts only converted v3 classifier heads")
    base=next(iter(models.values())).base_model.to(device).eval().requires_grad_(False)
    heads={epoch:m.score_head.to(device).eval().requires_grad_(False) for epoch,m in models.items()}
    rows={epoch:[] for epoch in models}
    with torch.inference_mode():
        for batch in loader:
            inputs,_=trainer.runner._full_batch(batch,device)
            original=base(*inputs)
            for epoch,head in heads.items():
                score=head(original.assignment,inputs[4],inputs[5])
                finite=original.training_valid & torch.isfinite(score)
                score=torch.where(finite,score,torch.zeros_like(score))
                probability=score.sigmoid()
                decision=finite & original.transport.diagnostics.converged
                values=dict(coarse=original.coarse_probability.detach().cpu().numpy(),
                    local=probability.detach().cpu().numpy(),fused=probability.detach().cpu().numpy())
                if any(not np.isfinite(v).all() for v in values.values()):raise ValueError("nonfinite validation pair scores")
                for index,pid in enumerate(batch.pair_ids):
                    rows[epoch].append(dict(pair_id=pid,label=bool(batch.labels[index]),
                        classification={k:float(v[index]) for k,v in values.items()},
                        decision_valid=bool(decision[index].item())))
    return {epoch:(report_from_rows(values),values) for epoch,values in rows.items()}


def execute_epochs(plan, evaluate_epoch, *, resume=False, stop_after_epoch=20):
    """Callback evaluates new models only; injection supports small CPU unit fixtures."""
    if stop_after_epoch not in EPOCHS:raise ValueError("stop-after-epoch must be13..20")
    root=Path(plan["output"])
    if root.exists() and not resume:raise FileExistsError("new output required, or explicit --resume")
    root.mkdir(parents=True,exist_ok=resume)
    plan_path=root/"revalidation_plan.json"
    plan_sha=trainer.canonical_digest(plan)
    if plan_path.exists():
        if trainer.canonical_digest(json.loads(plan_path.read_text())) != plan_sha:
            raise ValueError("resume conversion/source/VAL/runtime identity changed")
    else:trainer.save_json(plan_path,plan)
    winners={}; completed=[]; reused=[]; started=time.monotonic()
    with trainer.run_lock(root):
        trainer.save_json(root/"protocol.json",dict(schema_version=SCHEMA,status="running",
            plan_sha256=plan_sha,conversion=plan["provenance"],new_training_pair_exposures=0,
            active_training_state_available=False,inference_only=True))
        try:
            for epoch in EPOCHS:
                if epoch>stop_after_epoch:break
                source=plan["provenance"]["source_checkpoints"][str(epoch)]
                if sha(source["path"])!=source["sha256"]:raise ValueError("source checkpoint changed after binding")
                checkpoint_path=root/("epoch_%03d.pt"%epoch)
                receipt_path=root/("validation_%03d_complete.json"%epoch)
                rows_path=root/("validation_%03d_rows.json"%epoch)
                report_path=root/("validation_%03d.json"%epoch)
                trainer.save_json(root/"status.json",dict(status="running",phase="SIM_VAL_revalidation",
                    epoch=epoch,completed_epochs=completed,new_training_pair_exposures=0,pid=os.getpid()))
                if receipt_path.exists():
                    receipt=json.loads(receipt_path.read_text())
                    if (receipt.get("status")!="complete" or receipt.get("plan_sha256")!=plan_sha
                            or receipt.get("source_checkpoint_sha256")!=source["sha256"]
                            or any(sha(root/name)!=digest for name,digest in receipt["artifacts_sha256"].items())
                            or set(receipt["artifacts_sha256"])!={checkpoint_path.name,rows_path.name,report_path.name}):
                        raise ValueError("completed VAL receipt or artifact identity changed")
                    bundle=json.loads(report_path.read_text());rows=json.loads(rows_path.read_text())
                    report,points=bundle["validation"],bundle["operating_points"]
                    validate_rows(report,rows,plan);reused.append(epoch)
                else:
                    payload=ensure_converted_checkpoint(plan,epoch)
                    report,rows=evaluate_epoch(payload,epoch)
                    validate_rows(report,rows,plan)
                    points=fit_operating_points([r["label"] for r in rows],[r["classification"]["fused"] for r in rows])
                    if report["thresholds"]["fused"]!=points["thresholds"]["max_f1"]:
                        raise ValueError("core validation and fresh operating points disagree")
                    trainer.save_json(rows_path,rows)
                    trainer.save_json(report_path,dict(epoch=epoch,validation=report,operating_points=points,
                        selection_eligible=True,global_exposure=epoch*24000,inherited_exposure_only=True,
                        new_training_pair_exposures=0,source_checkpoint_sha256=source["sha256"]))
                    trainer.save_json(receipt_path,dict(schema_version=SCHEMA,status="complete",epoch=epoch,
                        plan_sha256=plan_sha,source_checkpoint_sha256=source["sha256"],
                        new_training_pair_exposures=0,validation_count=3000,held_out_used_for_fit=False,
                        artifacts_sha256={p.name:sha(p) for p in (checkpoint_path,rows_path,report_path)}))
                    del payload
                winners=trainer.update_winners(winners,root=root,epoch=epoch,report=report,points=points)
                completed.append(epoch)
                trainer.publish_freezes(root,epoch,winners,plan["identity"])
            status="complete" if completed==list(EPOCHS) else "stopped_at_epoch"
            result=dict(schema_version=SCHEMA,status=status,phase="SIM_VAL_revalidation",completed_epochs=completed,
                reused_validation_epochs=reused,new_training_pair_exposures=0,new_optimizer_updates=0,
                inherited_source_budget="M12+C8; no retraining",inference_only=True,
                frozen_selections={k:v["selected_epoch"] for k,v in winners.items()},
                elapsed_seconds=time.monotonic()-started,held_out_used_for_fit=False)
            trainer.save_json(root/"status.json",result)
            trainer.save_json(root/"protocol.json",dict(**result,plan_sha256=plan_sha,conversion=plan["provenance"]))
            return result
        except BaseException as error:
            trainer.save_json(root/"status.json",dict(schema_version=SCHEMA,status="failed",error=repr(error),
                completed_epochs=completed,new_training_pair_exposures=0))
            raise


def run(args):
    if args.batch_size<1 or args.workers<0:raise ValueError("invalid VAL batch/workers")
    torch.set_num_threads(1)
    plan=make_plan(args.source_training_run,args.output,args.dataset,
        batch_size=args.batch_size,workers=args.workers,device=args.device)
    dataset=trainer.RachelPairDataset(args.dataset,"val")
    if len(dataset)!=3000 or dataset.split!="val":raise ValueError("complete SIM VAL only")
    device=torch.device(args.device)
    pending_results={}
    def evaluate(payload,epoch):
        if epoch not in pending_results:
            trainer.runner._set_determinism(plan["identity"]["seed"])
            epochs=[e for e in EPOCHS if e<=args.stop_after_epoch
                and not (Path(plan["output"])/("validation_%03d_complete.json"%e)).exists()]
            models={e:trainer.load_decoupled_checkpoint(ensure_converted_checkpoint(plan,e)) for e in epochs}
            loader=trainer.make_ablation_loader(dataset,list(range(3000)),batch_size=args.batch_size,
                num_workers=args.workers,seed=plan["identity"]["seed"],contour_cap=512)
            pending_results.update(evaluate_shared_validation(models,loader,device))
            del models,loader
        return pending_results.pop(epoch)
    return execute_epochs(plan,evaluate,resume=args.resume,stop_after_epoch=args.stop_after_epoch)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-training-run",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--dataset",default="/root/autodl-tmp/dataset_rachel_pairwise_n512_v1")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--stop-after-epoch",type=int,choices=EPOCHS,default=20)
    return p


if __name__=="__main__":
    print(json.dumps(run(parser().parse_args()),ensure_ascii=False))
