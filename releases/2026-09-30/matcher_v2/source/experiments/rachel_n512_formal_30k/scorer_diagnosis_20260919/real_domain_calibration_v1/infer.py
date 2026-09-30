"""Infer score-blind newly sampled negatives; reuse frozen original rows."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from .prepare import ARMS, read, rows, save
from ..local_evidence_v2.evaluate import load_frozen_model
from ..local_evidence_v2.runtime import lease
from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluator
from experiments.rachel_n512_formal_30k.run_real_layout_decoder_experiment import input_batches


def small(row):
    layout = row.get("layouts", {}).get("full_top2_mode", {})
    err = layout.get("translation_l2_px")
    return dict(pair_id=row["pair_id"], score=float(row["classification"]["fused"]),
        decision_valid=bool(row["decision_valid"]),
        layout_good_20=None if err is None else bool(layout.get("valid") and err<=20),
        layout_error_px=err)


def infer(root, arm):
    root = Path(root); base = Path(read(root/"protocol.json")["base"])
    destination=root/"predictions"/arm
    destination.mkdir(parents=True,exist_ok=False)
    started=time.time()
    torch.set_num_threads(1)
    evaluator.core.sealed._set_determinism(260913)
    if arm == "gcn_shredding_h4":
        torch.use_deterministic_algorithms(True,warn_only=True)
    model, receipt=load_frozen_model(base/"training"/arm,"fixed_epoch")
    expected=read(root/"model_freezes.json")[arm]["checkpoint_sha256"]
    if receipt["checkpoint_sha256"] != expected:
        raise ValueError("model changed since split freeze")
    device=torch.device("cuda:0")
    model=model.to(device).eval().requires_grad_(False)
    for split in ("real","ood"):
        meta=read(root/split/"manifest.json")
        original={r["pair_id"]:r for r in rows(base/"evaluation"/arm/split/"pair_results.jsonl")}
        reused=[]
        for r in meta["pairs"]:
            if r["reuse_original_score"]:
                old=original[r["pair_id"]]
                if (old["fragment_a"],old["fragment_b"])!=(r["fragment_a_id"],r["fragment_b_id"]):
                    raise ValueError("reused score endpoints differ")
                reused.append(dict(small(old),prediction_source="existing_frozen_output"))
        del original
        new_rows=[r for r in meta["pairs"] if not r["reuse_original_score"]]
        # Only named input fields are read by input_batches. Labels/folds are
        # not forwarded to the model or used to select any Layout candidate.
        inputs=dict(fragment_ids=meta["fragment_ids"],pairs=new_rows)
        with np.load(Path(meta["prepared"])/"inputs.npz",allow_pickle=False) as f:
            arrays={k:f[k] for k in ("packed_masks","points","valid")}
        predictions=[]
        with torch.inference_mode():
            for i,batch in enumerate(input_batches(inputs,arrays,4)):
                output=evaluator.core.predict_batch(model,batch,device)
                predictions.extend(dict(small(r),prediction_source="new_cross_source_inference") for r in output)
                if i%20==0:
                    save(destination/"status.json",dict(status="inference",arm=arm,split=split,
                        new_done=len(predictions),new_total=len(new_rows),batch=4,training=False))
        merged={r["pair_id"]:r for r in reused+predictions}
        if set(merged)!={r["pair_id"] for r in meta["pairs"]}:
            raise ValueError("inference incomplete")
        for r in merged.values():
            if not np.isfinite(r["score"]) or not 0<=r["score"]<=1:
                raise ValueError("invalid score")
        save(destination/(split+".json"),dict(status="complete",arm=arm,split=split,
            checkpoint_sha256=expected,trained=False,thresholds_fitted=False,
            reused_count=len(reused),new_inferred_count=len(predictions),
            rows=[merged[r["pair_id"]] for r in meta["pairs"]]))
    save(destination/"status.json",dict(status="complete",arm=arm,training=False,
        elapsed_seconds=time.time()-started))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--gpu-uuid",required=True)
    a=p.parse_args()
    with lease(a.gpu_uuid,Path(a.root)/"gpu_locks"):
        infer(a.root,a.arm)
