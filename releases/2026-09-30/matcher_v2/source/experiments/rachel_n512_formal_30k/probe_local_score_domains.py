"""Read-only Full24 inference diagnostics; all counterfactual scores are non-canonical.

No fitting, checkpoint selection, threshold changes, or training. Model inputs
never include labels/GT. Synthetic GT is attached after each prediction. OOD
has pair-positive labels but no layout GT. Evidence replacements are sensitivity
probes, not a validated alternative classifier.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from experiments.rachel_n512_formal_30k.evaluate_recall_data_volume import load_winner
from experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint import TOP2_CONFIG
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from staging.pairwise_v0_2.models.translation_layout import estimate_translation_layout
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed

GLOBALS = ["coverage_min", "coverage_mean", "coverage_difference", "weighted_affinity",
           "max_affinity", "continuity_mean", "continuity_difference", "soft_dispersion_normalized"]
FIELDS = ("mask_a", "mask_b", "points_rc_a", "points_rc_b", "contour_valid_a", "contour_valid_b")


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def save(path, obj):
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def overlap(a, b, t):
    a, b = np.asarray(a).squeeze().astype(bool), np.asarray(b).squeeze().astype(bool)
    dr, dc = np.rint(-np.asarray(t)).astype(int)
    ra, rb = max(0, dr), max(0, -dr)
    ca, cb = max(0, dc), max(0, -dc)
    h, w = min(a.shape[0]-ra, b.shape[0]-rb), min(a.shape[1]-ca, b.shape[1]-cb)
    area = int(np.count_nonzero(a[ra:ra+h,ca:ca+w] & b[rb:rb+h,cb:cb+w])) if h > 0 and w > 0 else 0
    return area / max(1, min(a.sum(), b.sum()))


def shape_info(mask, points, valid):
    rr, cc = np.where(np.asarray(mask).squeeze() > .5)
    pts = points[valid]
    steps = np.linalg.norm(np.roll(pts, -1, axis=0)-pts, axis=1)
    return dict(area=len(rr), equivalent_side_px=float(np.sqrt(len(rr))),
                bbox_hw=[int(rr.max()-rr.min()+1), int(cc.max()-cc.min()+1)],
                contour_points=len(pts), contour_length_px=float(steps.sum()),
                mean_contour_step_px=float(steps.mean()))


def tensors_of(arrays, device):
    return [torch.as_tensor(x, device=device, dtype=torch.bool if i >= 4 else torch.float32)
            for i, x in enumerate(arrays)]


def sym(a, b):
    return torch.cat((torch.abs(a-b), a*b, .5*(a+b), torch.maximum(a,b)), dim=1)


def pool_variant(seq, valid, k):
    values = seq * valid[:, None]
    if k == 0:
        return values.sum(2) / valid.sum(1)[:, None].clamp_min(1)
    masked = torch.where(valid[:, None], values, torch.full_like(values, -1e4))
    return torch.stack([masked[i].topk(min(k,int(valid[i].sum())), dim=1).values.mean(1)
                        for i in range(len(valid))])


def fusion(model, coarse, local):
    x = torch.stack((coarse, local, coarse*local, torch.abs(coarse-local)), dim=1)
    return model.fusion(x).squeeze(1)


class Probe:
    def __init__(self, model, device, output):
        self.model, self.device, self.output = model, device, Path(output)
        self.rows, self.xs, self.stash, self.preview = [], [], {}, {}
        self.h1 = model.local_head.head.register_forward_pre_hook(self.capture_x)
        self.h2 = model.local_head.sequence.register_forward_hook(self.capture_sequence)

    def capture_x(self, module, inputs): self.stash["x"] = inputs[0].detach().clone()
    def capture_sequence(self, module, inputs, output): self.stash.setdefault("seq", []).append(output.detach())

    @torch.inference_mode()
    def batch(self, domain, ids, arrays, labels=None, targets=None, extra=None):
        tensors = tensors_of(arrays, self.device)
        self.stash = {}
        out = self.model(*tensors)
        x = self.stash["x"].clone()
        seq_a, seq_b = self.stash["seq"]
        variants = {}
        for name,k in (("mean_only",0),("max_only",1),("top8_mean",8),("top32_mean",32)):
            z = torch.cat((sym(pool_variant(seq_a,tensors[4],k),pool_variant(seq_b,tensors[5],k)),x[:,-8:]),1)
            ll = self.model.local_head.head(z).squeeze(1)
            variants[name] = (ll.cpu().numpy(), fusion(self.model,out.coarse_logit,ll).cpu().numpy())
        q_all, aff_all = out.assignment.cpu().numpy(), out.affinity.cpu().numpy()
        soft_t = out.translation_hat_rc.cpu().numpy()
        for i,pair_id in enumerate(ids):
            ma,mb,pa,pb,va,vb = [np.asarray(v[i]) for v in arrays]
            va,vb=va.astype(bool),vb.astype(bool)
            q,aff=q_all[i],aff_all[i]
            est=estimate_translation_layout(pa,pb,q,va,vb,config=TOP2_CONFIG)
            ii=est.candidate_indices
            abs_w=q[ii[:,0],ii[:,1]] if len(ii) else np.empty(0)
            ins=ii[est.inlier_mask]
            mode_abs=float(abs_w[est.inlier_mask].sum()) if len(ii) else 0.
            delta=pb[None,:,:]-pa[:,None,:]
            mode_res=np.linalg.norm(delta-est.t_a_to_b_rc,axis=2)
            robust_q=q/(1+(mode_res/8.)**2)
            mode_disp=float(np.sqrt((robust_q*mode_res**2).sum()/max(float(robust_q.sum()),1e-12))) if est.valid else None
            row_mass,col_mass=q.sum(1),q.sum(0)
            shape_a,shape_b=shape_info(ma,pa,va),shape_info(mb,pb,vb)
            row=dict(pair_id=pair_id,domain=domain,
                scores={name:float(getattr(out,name+"_probability")[i]) for name in ("coarse","local","fused")},
                logits={name:float(getattr(out,name+"_logit")[i]) for name in ("coarse","local","fused")},
                global_evidence=dict(zip(GLOBALS,x[i,-8:].cpu().tolist())),
                decision_valid=bool(out.decision_valid[i]),
                transport=dict(total_mass=float(q.sum()),max_entry=float(q.max()),
                    row_mass_p10_p50_p90=np.quantile(row_mass[va],[.1,.5,.9]),
                    col_mass_p10_p50_p90=np.quantile(col_mass[vb],[.1,.5,.9]),
                    unmatched_mean=.5*float(out.unmatched_a[i][tensors[4][i]].mean()+out.unmatched_b[i][tensors[5][i]].mean()),
                    rows_mass_ge_half=int((row_mass[va]>=.5).sum()), rows_mass_ge_point1=int((row_mass[va]>=.1).sum())),
                shape_a=shape_a,shape_b=shape_b,
                area_ratio=min(shape_a["area"],shape_b["area"])/max(shape_a["area"],shape_b["area"]),
                layout=dict(translation_rc=est.t_a_to_b_rc,valid=est.valid,residual_px=est.residual_px,
                    inlier_count=est.inlier_count,unique_inliers_a=len(np.unique(ins[:,0])) if len(ins) else 0,
                    unique_inliers_b=len(np.unique(ins[:,1])) if len(ins) else 0,
                    weighted_inlier_fraction=est.weighted_inlier_fraction,
                    runner_up_support_ratio=est.runner_up_support_ratio,
                    absolute_inlier_mass=mode_abs,absolute_candidate_mass=float(abs_w.sum()),
                    absolute_mode_fraction_of_total_mass=mode_abs/max(float(q.sum()),1e-15),
                    soft_translation_rc=soft_t[i],soft_mode_distance_px=float(np.linalg.norm(soft_t[i]-est.t_a_to_b_rc)),
                    soft_dispersion_px=float(out.translation_dispersion_px[i]),
                    mode_centered_soft_dispersion_px=mode_disp,
                    overlap_small_fraction=overlap(ma,mb,est.t_a_to_b_rc) if est.valid else None),
                pooling_counterfactual={name:dict(local_logit=float(v[0][i]),fused_logit=float(v[1][i])) for name,v in variants.items()})
            if labels is not None: row["label"]=bool(labels[i])
            if targets is not None:
                gt=targets[i]
                row["target_translation_rc"]=gt
                row["layout"]["error_px"]=float(np.linalg.norm(est.t_a_to_b_rc-gt)) if gt is not None and est.valid else None
            if extra is not None: row["extra"]=extra[i]
            self.rows.append(clean(row)); self.xs.append(x[i].cpu().numpy())
            # Small deterministic preview cache; no full-matrix dump.
            if domain == "ood" or (domain == "sim_test" and labels is not None and labels[i]):
                self.preview[pair_id]=(np.packbits(ma.astype(bool)),np.packbits(mb.astype(bool)),pa,pb,ins,q[ins[:,0],ins[:,1]] if len(ins) else np.empty(0))
        with (self.output/"rows.jsonl").open("a") as f:
            for row in self.rows[-len(ids):]: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        del out,tensors

    @torch.inference_mode()
    def counterfactual(self):
        x=np.stack(self.xs)
        positive=np.array([r["domain"]=="sim_test" and r.get("label",False) for r in self.rows])
        if not positive.any(): return
        ref=np.median(x[positive],axis=0)
        tensor=torch.tensor(x,device=self.device)
        coarse=torch.tensor([r["logits"]["coarse"] for r in self.rows],device=self.device)
        groups={"coverage":slice(96,99),"strength":slice(99,101),"continuity":slice(101,103),
                "soft_dispersion":slice(103,104),"all_global":slice(96,104),"sequence":slice(0,96)}
        for name,sl in groups.items():
            z=tensor.clone(); z[:,sl]=torch.tensor(ref[sl],device=self.device)
            ll=self.model.local_head.head(z).squeeze(1); fl=fusion(self.model,coarse,ll)
            for i,row in enumerate(self.rows):
                row.setdefault("evidence_counterfactual",{})[name]=dict(local_logit=float(ll[i]),fused_logit=float(fl[i]))
        for field in ("residual_px","mode_centered_soft_dispersion_px"):
            z=tensor.clone()
            z[:,103]=torch.tensor([r["layout"][field]/800 if r["layout"]["valid"] and r["layout"][field] is not None else float(x[i,103]) for i,r in enumerate(self.rows)],device=self.device)
            ll=self.model.local_head.head(z).squeeze(1); fl=fusion(self.model,coarse,ll)
            for i,row in enumerate(self.rows):
                row["evidence_counterfactual"][field+"_in_dispersion_slot"]=dict(local_logit=float(ll[i]),fused_logit=float(fl[i]),valid=row["layout"]["valid"])
        save(self.output/"counterfactual_reference.json",dict(scope="diagnostic-only SIM TEST positive median; not fitting or deployed scores",global_medians=dict(zip(GLOBALS,ref[-8:])),positive_count=int(positive.sum())))
        np.savez_compressed(self.output/"head_inputs.npz",x=x)

    def save_previews(self):
        chosen=[]
        ood=[r for r in self.rows if r["domain"]=="ood"]
        for group in ([r for r in ood if r["scores"]["coarse"]>.9 and r["scores"]["local"]<.01],
                      [r for r in ood if r["scores"]["local"]<.01 and r["layout"]["overlap_small_fraction"] is not None and r["layout"]["overlap_small_fraction"]<.1],
                      sorted(ood,key=lambda r:-r["scores"]["local"])):
            for r in group[:2]:
                if r["pair_id"] not in chosen: chosen.append(r["pair_id"])
        chosen += [r["pair_id"] for r in self.rows if r["domain"]=="sim_test" and r.get("label")][:2]
        dest=self.output/"previews"; dest.mkdir(exist_ok=True)
        for k in chosen:
            ma,mb,pa,pb,ins,w=self.preview[k]
            name=hashlib.sha256(k.encode()).hexdigest()[:12]+".npz"
            np.savez_compressed(dest/name,mask_a=ma,mask_b=mb,points_a=pa,points_b=pb,inliers=ins,weights=w)
            next(r for r in self.rows if r["pair_id"]==k)["preview_file"]="previews/"+name


def run(args):
    dest=Path(args.output); dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); sealed._set_determinism(260912)
    model,identity,thresholds=load_winner(args.training_run)
    device=torch.device(args.device); model=model.to(device).eval().requires_grad_(False)
    probe=Probe(model,device,dest)
    protocol=dict(status="running",model=identity,thresholds=thresholds,training=False,
        canonical_scores_changed=False,selection="SIM TEST hash-order by pair_id, separately positive/negative; all eligible OOD",seed=260912,
        ood=str(args.ood),dataset=str(args.dataset),counterfactual_scores_are_not_performance_claims=True)
    save(dest/"protocol.json",protocol)
    started=time.monotonic()
    root=Path(args.dataset)
    manifest=[json.loads(x) for x in (root/"pairs/test.jsonl").read_text().splitlines() if x.strip()]
    key=lambda i:hashlib.sha256(manifest[i]["pair_id"].encode()).hexdigest()
    indices=sorted([i for i,r in enumerate(manifest) if r["label"]],key=key)[:args.sim_positive]
    indices+=sorted([i for i,r in enumerate(manifest) if not r["label"]],key=key)[:args.sim_negative]
    dataset=RachelPairDataset(root,"test")
    loader=make_ablation_loader(dataset,indices,batch_size=args.batch_size,num_workers=2,seed=260912,contour_cap=512)
    damage_seeds=[]
    for batch in loader:
        arr=[getattr(batch,n) for n in FIELDS]
        targets=[batch.translation_a_to_b_rc[i] if batch.translation_valid[i] else None for i in range(len(batch.pair_ids))]
        probe.batch("sim_test",batch.pair_ids,arr,batch.labels,targets)
        for i,label in enumerate(batch.labels):
            if label and len(damage_seeds)<args.damage_pairs:
                damage_seeds.append((batch.pair_ids[i],[x[i:i+1].copy() for x in arr],targets[i]))
    print(json.dumps(dict(stage="sim_complete",rows=len(probe.rows),elapsed=time.monotonic()-started)),flush=True)
    if damage_seeds:
        from staging.pairwise_v0_2.pairwise_data.rachel_edge_weathering import weather_fragment_edges,EdgeWeatheringConfig
        from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour
        for pair_id,orig,gt in damage_seeds:
            for depth in (2.,4.):
                arr=[x.copy() for x in orig]; details=[]
                for side in (0,1):
                    mask,detail=weather_fragment_edges(arr[side][0].squeeze().astype(bool),seed=260912,
                        fragment_id=pair_id+str(side),config=EdgeWeatheringConfig(max_depth_px=depth))
                    pts,valid=extract_ordered_outer_contour(mask,cap=512,smoothing_sigma=3.)
                    arr[side]=mask[None,None].astype(np.float32)
                    arr[2+side]=np.zeros((1,512,2),np.float32);arr[4+side]=np.zeros((1,512),bool)
                    arr[2+side][0,:len(pts)]=pts;arr[4+side][0,:len(valid)]=valid
                    details.append(detail)
                probe.batch("sim_damage",[pair_id+"__depth"+str(int(depth))],arr,[True],[gt],
                    [dict(base_pair_id=pair_id,max_depth_px=depth,damage=details)])
    meta=json.loads((Path(args.ood)/"manifest.json").read_text())
    ids=meta["fragment_ids"];lookup={k:i for i,k in enumerate(ids)}
    with np.load(Path(args.ood)/"inputs.npz") as f:
        packed,points,valid=f["packed_masks"],f["points"],f["valid"]
    pairs=meta["pairs"][:args.ood_limit] if args.ood_limit else meta["pairs"]
    for start in range(0,len(pairs),args.batch_size):
        part=pairs[start:start+args.batch_size]
        a=np.array([lookup[p["fragment_a_id"]] for p in part]);b=np.array([lookup[p["fragment_b_id"]] for p in part])
        arr=[np.unpackbits(packed[a],axis=2)[:,None].astype(np.float32),np.unpackbits(packed[b],axis=2)[:,None].astype(np.float32),points[a],points[b],valid[a],valid[b]]
        probe.batch("ood",[p["pair_id"] for p in part],arr,[True]*len(part))
    print(json.dumps(dict(stage="ood_complete",rows=len(probe.rows),elapsed=time.monotonic()-started)),flush=True)
    probe.counterfactual();probe.save_previews()
    save(dest/"results.json",dict(rows=probe.rows,counterfactuals_diagnostic_only=True))
    protocol.update(status="complete",total_rows=len(probe.rows),elapsed_seconds=time.monotonic()-started,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else None)
    save(dest/"protocol.json",protocol)
    print(json.dumps(dict(status="complete",rows=len(probe.rows),elapsed=time.monotonic()-started)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run",required=True);p.add_argument("--dataset",required=True)
    p.add_argument("--ood",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--sim-positive",type=int,default=128);p.add_argument("--sim-negative",type=int,default=64)
    p.add_argument("--damage-pairs",type=int,default=16);p.add_argument("--ood-limit",type=int,default=0)
    run(p.parse_args())
