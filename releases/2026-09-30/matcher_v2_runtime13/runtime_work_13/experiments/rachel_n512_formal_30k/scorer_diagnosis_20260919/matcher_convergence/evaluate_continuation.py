"""Fixed S7 M12/M16/M20 SIMVAL evaluation, without Scorer or model selection.

Uses the original matcher loss and production decoder definitions. The default
CLI only checks inputs; --execute acquires the shared GPU lock nonblocking.
No checkpoint or threshold fitting, no TEST/REAL/OOD, no training or queue edits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence import evaluate_matcher_simval as original

SCHEMA="s7-matcher-continuation-fixed-simval/1"
EPOCHS=(12,16,20)


def validate_endpoint_metadata(payload,epoch):
    if type(epoch) is not int or epoch not in EPOCHS:
        raise ValueError("only fixed M12/M16/M20 endpoints are registered")
    expected=dict(epoch=epoch,completed_segments=epoch*4,phase="matcher",
        global_exposure=epoch*24000,optimizer_updates=epoch*1500,checkpoint_role="epoch_anchor")
    if any(payload.get(key)!=value for key,value in expected.items()):
        raise ValueError("requires exact committed Matcher epoch anchor, not a C checkpoint")
    if payload.get("formal_training_counted") is not True or payload.get("inference_only"):
        raise ValueError("discard/warm-start/inference-only checkpoint is not a formal endpoint")
    if epoch>12:
        if payload.get("rng_mode")!="exact_gpu" or payload.get("continuation_identity",{}).get("rng_mode")!="exact_gpu":
            raise ValueError("CPU-only continuation probe cannot be a formal endpoint")
    return expected


def load_endpoint(checkpoint,epoch):
    import torch
    from . import continuation_core as core
    from staging.pairwise_v0_2.models.rachel_decoupled_score import load_decoupled_score_checkpoint
    path=Path(checkpoint).resolve(strict=True)
    digest=original.sha256(path)
    payload=torch.load(path,map_location="cpu",weights_only=False)
    validate_endpoint_metadata(payload,epoch)
    if epoch==12:
        wrapper=core.validate_source(payload,digest)
        current_digest=core.SOURCE_BASE_SHA256
        origin=payload["resume_identity"]
        bindings=dict(source_checkpoint_sha256=digest,source_epoch=12)
    else:
        identity=payload["continuation_identity"]
        if core.validate_progress(payload,identity,rng_mode="exact_gpu")!=epoch*4:
            raise ValueError("committed continuation progress differs")
        wrapper=load_decoupled_score_checkpoint(payload)
        core.validate_optimizer(payload,wrapper,expected_updates=epoch*1500)
        if core.frozen_digests(wrapper)!=identity["source_frozen_digests"]:
            raise ValueError("frozen classifier/diagnostic branches changed")
        origin=identity["source_resume_identity"]
        current_digest=payload["current_base_state_sha256"]
        bindings=dict(source_checkpoint_sha256=identity["source_checkpoint_sha256"],source_epoch=12,
            continuation_identity_sha256=core.old.canonical_digest(identity),
            training_implementation_bindings=identity["implementation_bindings"])
    if core.old.state_digest(wrapper.base_model)!=current_digest:
        raise ValueError("Matcher state differs from committed checkpoint")
    record=origin["populations"]["val"]
    if record.get("manifest_sha256")!=original.VAL_HASH or record.get("count")!=3000 or record.get("split")!="val":
        raise ValueError("requires unchanged SIMVAL3000")
    base=wrapper.base_model.eval().requires_grad_(False)
    config=replace(core.old.RachelN512LossConfig(**payload["loss_config"]),validate_runtime_targets=True)
    info=dict(epoch=epoch,checkpoint=str(path),checkpoint_sha256=digest,
        matcher_state_sha256=current_digest,origin=bindings,source_base_loss_config=asdict(config),
        current_matcher_epochs=epoch,trained_classifier_evaluated=False,
        trained_scorer_head_called=False,formal_source=True,
        base_diagnostic_branches="coarse/local/fusion forward only; BCE weights zero; no classification gate")
    return base,config,info


def preflight(args):
    if args.epoch not in EPOCHS or args.workers!=4:
        raise ValueError("fixed M12/M16/M20, original workers4")
    checkpoint=Path(args.checkpoint).resolve(strict=True)
    dataset=Path(args.dataset).resolve(strict=True)
    manifest=dataset/"pairs/val.jsonl"
    if original.sha256(manifest)!=original.VAL_HASH:
        raise ValueError("SIMVAL manifest differs")
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line]
    ids=original.validate_manifest(rows)
    out=Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("new evaluation output required; never overwrite")
    if out==dataset or dataset in out.parents or out==checkpoint.parent or checkpoint.parent in out.parents:
        raise ValueError("evaluation output must be separate from data and checkpoint directory")
    return dict(schema=SCHEMA,status="preflight_only",epoch=args.epoch,
        checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
        dataset=str(dataset),manifest=str(manifest),manifest_sha256=original.VAL_HASH,
        count=3000,positive_count=1500,negative_count=1500,pair_ids=ids,
        pair_order_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        output=str(out),batch_size=1,workers=4,execute_requested=args.execute,
        classification_metrics=False,selection_performed=False,threshold_fit=False,
        held_out_real_ood_used=False)


def execute(args,plan):
    # Existing lock is held before CUDA setup/model loading; never overlap train.
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local.train import gpu_lock
    with gpu_lock():
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
            os.environ[name]="1"
        import torch
        from . import continuation_core as core
        from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
        from experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint import TOP2_CONFIG
        from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError("requires exactly one visible CUDA GPU")
        torch.set_num_threads(1)
        core.old.runner._set_determinism(core.old.SEED)
        device=torch.device("cuda:0")
        model,config,identity=load_endpoint(plan["checkpoint"],plan["epoch"])
        model.to(device)
        dataset=RachelPairDataset(plan["dataset"],"val")
        if len(dataset)!=3000 or dataset.split!="val":
            raise ValueError("reader must be unchanged SIMVAL3000")
        loader=make_ablation_loader(dataset,tuple(range(3000)),batch_size=1,num_workers=4,
            seed=core.old.SEED,contour_cap=512)
        destination=Path(plan["output"])
        destination.mkdir(parents=True,exist_ok=False)
        protocol=dict(plan,status="running",pid=os.getpid(),model=identity,
            evaluator_sha256=original.sha256(__file__),
            original_metric_implementation_sha256=original.sha256(original.__file__),
            decoder_config=asdict(TOP2_CONFIG),weights_frozen=True,classifier_gate=False,
            success_criterion="raw decoded translation L2 <=20px, all positive pairs in denominator",
            loss_semantics="All-pair means include zero invalid/unsupervised terms; interpret together with training-valid counts and conditional-supervision denominators.",
            endpoint_selection="fixed M12/M16/M20; primary M20; no best-epoch fitting",
            limitations=["Clean SIMVAL alone cannot establish difficult S7-domain convergence.",
                "New Matcher features require equally trained Scorer heads before comparing classification."],
            completed_count=0)
        original.save(destination/"protocol.json",protocol)
        rows,started=[],time.monotonic()
        try:
            with (destination/"pair_metrics.jsonl").open("x") as stream:
                for batch in loader:
                    current=original.evaluate_batch(model,batch,config,device)
                    for row in current:
                        stream.write(json.dumps(row,allow_nan=False)+"\n")
                    rows.extend(current)
                    if len(rows)%250==0:
                        protocol["completed_count"]=len(rows)
                        original.save(destination/"protocol.json",protocol)
            if [r["pair_id"] for r in rows]!=plan["pair_ids"]:
                raise ValueError("SIMVAL population/order changed")
            if core.old.state_digest(model)!=identity["matcher_state_sha256"]:
                raise ValueError("evaluation modified frozen Matcher")
            result=dict(schema=SCHEMA,status="complete",model=identity,
                metrics=original.summarize(rows),count=3000,
                effective_matcher_objective_weights=dict(assignment_nll=.5,translation_smooth_l1=.5,
                    sinkhorn_residual=.05,all_pair_bce=0.),
                metrics_source_sha256=original.sha256(destination/"pair_metrics.jsonl"),
                classification_metrics_reported=False,checkpoint_selection_performed=False,
                elapsed_seconds=time.monotonic()-started)
            original.save(destination/"summary.json",result)
            protocol.update(status="complete",completed_count=3000,elapsed_seconds=time.monotonic()-started)
        except BaseException as error:
            protocol.update(status="failed",completed_count=len(rows),error=repr(error),elapsed_seconds=time.monotonic()-started)
            raise
        finally:
            original.save(destination/"protocol.json",protocol)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--epoch",type=int,choices=EPOCHS,required=True)
    p.add_argument("--dataset",type=Path,default=Path("/root/autodl-tmp/dataset_rachel_pairwise_n512_v1"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,choices=(4,),default=4)
    p.add_argument("--execute",action="store_true")
    return p


if __name__=="__main__":
    args=parser().parse_args()
    plan=preflight(args)
    if args.execute:
        execute(args,plan)
    else:
        print(json.dumps({k:v for k,v in plan.items() if k!="pair_ids"},indent=2))
