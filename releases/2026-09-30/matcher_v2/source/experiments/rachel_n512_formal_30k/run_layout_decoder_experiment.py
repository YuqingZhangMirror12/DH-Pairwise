"""Frozen coarse/sliding classification with independent translation decoders.

Validation selects geometry only. Test never fits a decoder or pair threshold.
The original Full forward and its dispersion remain untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import Subset

from staging.pairwise_v0_2.models.translation_layout import (
    TranslationLayoutConfig, estimate_translation_layout,
)
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed
from staging.pairwise_v0_2.training.metrics import _ranking_metrics

DEFAULT_RUN = "/root/autodl-tmp/rachel_n512_convergence_20260901_001/run-convergence-50a8cffb0ae92614"
DEFAULT_DATA = "/root/autodl-tmp/dataset_rachel_pairwise_n512_v1"


def decoder_configs():
    common = dict(max_candidates=512, min_inliers=3, inlier_radius_px=10.0)
    return {
        "full_sparse_cauchy": TranslationLayoutConfig(**common, decoder="median_cauchy"),
        "full_reciprocal_mode": TranslationLayoutConfig(**common),
        "full_top2_mode": TranslationLayoutConfig(**common, correspondence_mode="topk_union", top_k=2),
        "full_affinity_mode": TranslationLayoutConfig(**common, score_mode="dual_softmax", affinity_temperature=0.25),
        "full_reciprocal_mode_r5": TranslationLayoutConfig(max_candidates=512, min_inliers=3, inlier_radius_px=5.0),
    }


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def classification(labels, scores, threshold):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    selected = np.isfinite(scores) & (scores >= threshold)
    tp, fp = int((selected & labels).sum()), int((selected & ~labels).sum())
    fn, tn = int((~selected & labels).sum()), int((~selected & ~labels).sum())
    result = dict(threshold=float(threshold), accuracy=(tp+tn)/len(labels),
                  precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn),
                  f1=2*tp/max(1,2*tp+fp+fn), tp=tp, fp=fp, fn=fn, tn=tn)
    if len(np.unique(labels)) == 2:
        auroc,auprc=_ranking_metrics(scores.tolist(),labels.tolist())
        result.update(auroc=auroc,auprc=auprc)
    return result


def fit_threshold(labels, scores):
    labels,scores=np.asarray(labels,bool),np.asarray(scores,float)
    order=np.argsort(-scores,kind="stable")
    ranked=scores[order]
    last=np.r_[np.flatnonzero(ranked[:-1]!=ranked[1:]),len(ranked)-1]
    tp=np.cumsum(labels[order])[last]
    f1=2*tp/np.maximum(last+1+labels.sum(),1)
    return float(ranked[last[np.argmax(f1)]])


def summarize(rows, threshold, branch_thresholds=None):
    labels = np.array([r["label"] for r in rows], bool)
    scores = {b:np.array([r["classification"][b] for r in rows]) for b in ("coarse", "local", "fused")}
    selected = scores["fused"] >= threshold
    result = {"sample_count":len(rows), "positive_count":int(labels.sum()),
              "classification":{}, "layout":{}}
    for branch, values in scores.items():
        result["classification"][branch] = {"at_0_5":classification(labels,values,0.5)}
        if branch_thresholds:
            result["classification"][branch]["at_validation_row_f1_threshold"] = classification(labels,values,branch_thresholds[branch])
    result["classification"]["fused"]["at_original_frozen_threshold"] = classification(labels,scores["fused"],threshold)
    for name in rows[0]["layouts"]:
        errors = np.array([r["layouts"][name]["translation_l2_px"] if r["layouts"][name]["translation_l2_px"] is not None else np.inf for r in rows])
        valid = np.array([r["layouts"][name]["valid"] for r in rows],bool)
        positive_errors = errors[labels & valid & np.isfinite(errors)]
        m = {"positive_pose_coverage":float((labels & valid).sum()/max(1,labels.sum())),
             "median_px_conditional":float(np.median(positive_errors)) if len(positive_errors) else None,
             "p90_px_conditional":float(np.quantile(positive_errors,.9)) if len(positive_errors) else None,
             "recall":{}, "assembly":{}}
        for tolerance in (2,5,8,10):
            correct = labels & valid & (errors <= tolerance)
            tp = int((selected & correct).sum())
            fp, fn = int(selected.sum())-tp, int(labels.sum())-tp
            m["recall"][str(tolerance)] = float(correct.sum()/max(1,labels.sum()))
            m["assembly"][str(tolerance)] = dict(precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),tp=tp,fp=fp,fn=fn)
        result["layout"][name] = m
    return result


def run(args):
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    receipt, receipt_sha, winners = sealed._freeze_completed_winners(Path(args.run))
    full = next(w for w in winners if w.arm == "full_n512")
    seed = int(receipt["config"]["seed"])
    sealed._set_determinism(seed)
    device = torch.device(args.device)
    model = full.model.to(device).eval()
    frozen_threshold = float(full.threshold.threshold)
    authority = None
    if args.freeze:
        authority = json.loads(Path(args.freeze).read_text())
        if authority.get("probe_only") or authority.get("source_split") != "validation" or authority.get("test_or_real_used_for_fit"):
            raise ValueError("test requires a full validation-only decoder freeze")
        if authority["checkpoint_sha256"] != full.checkpoint_sha256:
            raise ValueError("validation geometry freeze references another Full checkpoint")
        configs = {name:TranslationLayoutConfig(**value) for name,value in authority["decoders"].items()}
    else:
        if args.split != "val":
            raise ValueError("test requires an existing validation freeze")
        configs = decoder_configs()
    dataset = RachelPairDataset(Path(args.dataset),args.split)
    with (Path(args.dataset)/"pairs"/(args.split+".jsonl")).open(encoding="utf-8") as manifest_stream:
        manifest_rows=[json.loads(line) for line in manifest_stream if line.strip()]
    source_units={row["pair_id"]:sorted({row["fragment_a"]["split_unit_id"],row["fragment_b"]["split_unit_id"]}) for row in manifest_rows}
    if args.limit:
        dataset = Subset(dataset,list(range(min(args.limit,len(dataset)))))
    loader = sealed._test_loader(dataset,batch_size=args.batch_size,num_workers=args.workers,seed=seed)
    head = None
    if args.shred_freeze:
        from staging.pairwise_v0_2.models.shredding_layout_head import load_shredding_layout_head
        head = load_shredding_layout_head(Path(args.shred_freeze),device=str(device),checkpoint_stage="classify",microbatch_size=args.shred_microbatch)
    protocol = {"status":"running", "split":args.split, "population_size":len(dataset),
                "checkpoint_sha256":full.checkpoint_sha256,"checkpoint_epoch":full.epoch,
                "original_fused_threshold":frozen_threshold,"decoders":{n:asdict(c) for n,c in configs.items()},
                "routing_used":False,"rotation_estimated":False,"classifier_modified":False,
                "test_previously_observed_in_prior_experiments":True,
                "shred_layout_head_used":bool(head),"shred_coarse_or_classifier_executed":False}
    write_json(destination/"protocol.json",protocol)
    rows=[]
    started=time.perf_counter()
    with (destination/"pair_results.jsonl").open("x",encoding="utf-8") as stream, torch.inference_mode():
        for batch_number,batch in enumerate(loader):
            tensors=[sealed._tensor(getattr(batch,n),device,t) for n,t in (
                ("mask_a",torch.float32),("mask_b",torch.float32),("points_rc_a",torch.float32),
                ("points_rc_b",torch.float32),("contour_valid_a",torch.bool),("contour_valid_b",torch.bool))]
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=receipt["config"]["precision"]=="bf16"):
                output=model(*tensors)
            values={name:getattr(output,name).detach().float().cpu().numpy() for name in (
                "fused_probability","coarse_probability","local_probability","assignment","affinity","translation_hat_rc")}
            shred_out=head.predict_batch(batch,return_correspondence=False) if head else None
            for i,pair_id in enumerate(batch.pair_ids):
                layouts={"full_original":{"translation_rc":values["translation_hat_rc"][i],"valid":bool(output.decision_valid[i].item())}}
                for name,config in configs.items():
                    matrix=values["affinity" if config.score_mode=="dual_softmax" else "assignment"][i]
                    estimate=estimate_translation_layout(batch.points_rc_a[i],batch.points_rc_b[i],matrix,
                        batch.contour_valid_a[i],batch.contour_valid_b[i],config=config)
                    d=asdict(estimate)
                    d.pop("candidate_indices",None)
                    d.pop("inlier_mask",None)
                    layouts[name]={"translation_rc":estimate.t_a_to_b_rc,"valid":bool(estimate.valid),"diagnostics":d}
                if shred_out is not None:
                    layouts["full_with_shred_matching_layout"]={"translation_rc":shred_out.translation_hat_rc[i],"valid":bool(shred_out.translation_valid[i]),
                        "diagnostics":{"candidate_count":int(shred_out.correspondence_count[i]),"inlier_count":int(shred_out.inlier_count[i])}}
                # Supervision is read only after all target-blind decoders have finished.
                gt=np.asarray(batch.translation_a_to_b_rc[i],float)
                target_valid=bool(batch.translation_valid[i])
                for layout in layouts.values():
                    t=np.asarray(layout["translation_rc"],float)
                    layout["offset_b_in_a_rc"]=-t
                    layout["translation_l2_px"]=float(np.linalg.norm(t-gt)) if target_valid and layout["valid"] else None
                row=clean({"pair_id":pair_id,"source_unit_ids":source_units[pair_id],"fragment_a":batch.fragment_a_tokens[i],"fragment_b":batch.fragment_b_tokens[i],
                     "label":bool(batch.labels[i]),"target_translation_rc":gt if target_valid else None,
                     "classification":{b:float(values[b+"_probability"][i]) for b in ("coarse","local","fused")},"layouts":layouts})
                rows.append(row)
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
            stream.flush()
            if batch_number%10==0 or len(rows)==len(dataset):
                print(json.dumps({"processed":len(rows),"total":len(dataset),"elapsed_s":round(time.perf_counter()-started,2)}),flush=True)
    branch_thresholds=authority["branch_validation_thresholds"] if authority else {
        b:fit_threshold([r["label"] for r in rows],[r["classification"][b] for r in rows]) for b in ("coarse","local","fused")}
    summary=summarize(rows,frozen_threshold,branch_thresholds)
    if not authority:
        candidates=list(configs)
        selected=max(candidates,key=lambda n:(np.mean(list(summary["layout"][n]["recall"].values())),summary["layout"][n]["assembly"]["10"]["f1"],-summary["layout"][n]["p90_px_conditional"] if summary["layout"][n]["p90_px_conditional"] is not None else -np.inf))
        authority={"source_split":"validation","sample_count":len(rows),"selection_rule":"maximize mean unconditional R@2/5/8/10, then assembly F1@10, then minimize conditional P90", "selected_full_decoder":selected,
            "checkpoint_sha256":full.checkpoint_sha256,"decoders":{n:asdict(c) for n,c in configs.items()},
            "branch_validation_thresholds":branch_thresholds,"original_fused_threshold":frozen_threshold,
            "test_or_real_used_for_fit":False,"probe_only":bool(args.limit)}
        write_json(destination/"validation_freeze.json",authority)
    summary.update(status="complete",split=args.split,selected_full_decoder=authority["selected_full_decoder"],elapsed_s=time.perf_counter()-started,
        original_classification_preserved=True,rotation_estimated=False,routing_used=False)
    write_json(destination/"summary.json",summary)
    print(json.dumps(clean(summary),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default=DEFAULT_RUN)
    parser.add_argument("--dataset",default=DEFAULT_DATA)
    parser.add_argument("--split",choices=("val","test"),default="val")
    parser.add_argument("--output",required=True)
    parser.add_argument("--freeze")
    parser.add_argument("--shred-freeze")
    parser.add_argument("--shred-microbatch",type=int,default=1)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--limit",type=int,default=0)
    run(parser.parse_args())


if __name__=="__main__":
    main()
