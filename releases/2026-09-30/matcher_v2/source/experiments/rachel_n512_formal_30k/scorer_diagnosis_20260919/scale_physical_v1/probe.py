"""CPU-only, frozen S7 physical-scale intervention on the existing fixed40.

R: original patches/features and coordinates.
A: scaled raster/coordinates, unchanged window spans.
B: scaled raster/coordinates, compensated window spans.
C: B encoded features, original context coordinates (counterfactual).
D: R encoded features, scaled context coordinates (counterfactual).

This does NOT run Sinkhorn/Layout, fit thresholds, or estimate accuracy. The
R/D/C/B factorial separates position input from raster/local-feature changes;
its interaction must not be interpreted as additive independent mechanisms.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True
import numpy as np
from scipy import ndimage
import torch

try:
    from .geometry import cloned_sampler, downscale_pair
except ImportError:  # Direct detached CLI execution.
    from geometry import cloned_sampler, downscale_pair

CHECKPOINT_SHA = "7c1212e2d9d62c25954457add3f9319b03dfe8fbc42aa96116e40caae0dc2c37"
CASES_SHA = "7bab8e29a348e1ea62607bcf45376e6fe6e8dac2f2b7802d6b7415463d8544cf"
SELECTION_SHA = "071d14418a3b7cafee408b17d29a0edba175992206ce3a8be20048c03c259d37"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def difference(reference, current):
    if (reference.shape != current.shape or reference.numel() == 0
            or not torch.isfinite(reference).all() or not torch.isfinite(current).all()):
        raise ValueError("feature comparison requires aligned, finite, nonempty inputs")
    norm = torch.linalg.vector_norm(reference).item()
    delta = torch.linalg.vector_norm(current-reference).item()
    return dict(relative_l2=delta/norm if norm else None,
                absolute_l2=delta, max_abs=float((current-reference).abs().max()))


def compare_snapshots(reference, current, valid):
    """All comparisons exclude padded tokens; patch mismatch is per scale."""
    result = {}
    for i, side in enumerate("ab"):
        keep = valid[i]
        ref_p, cur_p = reference["patches"][i][keep], current["patches"][i][keep]
        if ref_p.shape != cur_p.shape or ref_p.ndim != 5 or len(ref_p) == 0:
            raise ValueError("bad valid patch shape")
        result[side] = dict(patch_difference_rate_by_scale=(ref_p != cur_p).float().mean((0,2,3,4)).tolist(),
            encoded=difference(reference["encoded"][i][keep], current["encoded"][i][keep]),
            context=difference(reference["context"][i][keep], current["context"][i][keep]))
    return result


def boundary_distance(mask, points, valid):
    binary = mask[0,0].cpu().numpy().astype(bool)
    boundary = binary & ~ndimage.binary_erosion(binary, border_value=0)
    if not boundary.any():
        raise ValueError("scaled raster has no material boundary")
    distance = ndimage.distance_transform_edt(~boundary)
    coords = points[valid].cpu().numpy()
    values = ndimage.map_coordinates(distance, coords.T, order=1, mode="nearest")
    return dict(mean_px=float(values.mean()), p95_px=float(np.quantile(values,.95)),
                max_px=float(values.max()), count=len(values))


@torch.inference_mode()
def encode(base, sampler, masks, points, valid):
    patches = tuple(sampler(masks[i], points[i], valid[i]) for i in range(2))
    encoded = tuple(base._encode_patches(patches[i], valid[i]) for i in range(2))
    return dict(patches=patches, encoded=encoded)


@torch.inference_mode()
def score(model, evidence, points, valid):
    context = model.base_model.context(*evidence["encoded"], *valid, *points, 800)
    logit = model.score_head(*context, *valid).reshape(-1)
    if logit.numel() != 1 or not torch.isfinite(logit).all():
        raise ValueError("expected one finite raw Scorer logit")
    snapshot = dict(evidence, context=context)
    return dict(logit=float(logit[0]), probability=float(logit[0].sigmoid())), snapshot


def factorial(rows):
    return rows["B"]["logit"]-rows["C"]["logit"]-rows["D"]["logit"]+rows["R"]["logit"]


def read_cases(root):
    protocol = json.loads((root/"protocol.json").read_text())
    if (protocol.get("status") != "complete" or protocol.get("sample_count") != 40
            or protocol.get("completed_count") != 40 or protocol.get("selection_json_sha256") != SELECTION_SHA
            or protocol.get("cases_sha256") != CASES_SHA or sha(root/"cases.json") != CASES_SHA
            or protocol["model"]["checkpoint_sha256"] != CHECKPOINT_SHA):
        raise ValueError("requires exact complete fixed40 S7 C8 archived probe")
    cases = json.loads((root/"cases.json").read_text())
    index = {r["pair_id"]: r for r in cases}
    chosen = protocol["selected_pairs"]
    ids = [r["pair_id"] for r in chosen]
    if len(cases) != 40 or len(index) != 40 or len(ids) != 40 or len(set(ids)) != 40 or set(ids) != set(index):
        raise ValueError("changed or duplicated case membership")
    return protocol, [(r,index[r["pair_id"]]) for r in chosen]


def summarize(rows):
    out = dict(case_count=len(rows), metrics="score sensitivity only; not accuracy", scales={})
    for scale in (.5,.75):
        measures = [r["scales"][str(scale)] for r in rows]
        ref = np.array([r["R"]["logit"] for r in rows])
        group = {}
        for branch in "ABCD":
            logits = np.array([r["branches"][branch]["logit"] for r in measures])
            drift = logits-ref
            group[branch] = dict(mean_signed_logit_drift=float(drift.mean()),
                mean_absolute_logit_drift=float(np.abs(drift).mean()),
                median_absolute_logit_drift=float(np.median(np.abs(drift))),
                max_absolute_logit_drift=float(np.abs(drift).max()))
        errors = {b: np.abs(np.array([r["branches"][b]["logit"] for r in measures])-ref) for b in "ABCD"}
        inter = np.array([r["factorial_logit_interaction"] for r in measures])
        out["scales"][str(scale)] = dict(branches=group,
            B_closer_to_R_than_A_count=int((errors["B"]<errors["A"]).sum()),
            C_closer_to_R_than_B_count=int((errors["C"]<errors["B"]).sum()),
            mean_signed_interaction=float(inter.mean()),mean_absolute_interaction=float(np.abs(inter).mean()))
    return out


def run(args):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(Path(args.source_root).resolve(strict=True)))
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as ev
    root, out = Path(args.probe_root).resolve(strict=True), Path(args.output).resolve()
    source, selected_cases = read_cases(root)
    model, identity = ev.load_frozen_model(source["model"]["training_run"], source["model"]["selection"])
    if identity["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("wrong checkpoint")
    model.cpu().eval().requires_grad_(False)
    base = model.base_model
    if (base.config.canvas_size != 800 or base.config.contour_cap != 512
            or list(base.patch_sampler.window_sizes_px) != [7.,16.,32.,64.]):
        raise ValueError("unexpected original physical input settings")
    samplers = {s: cloned_sampler(base.patch_sampler,s) for s in (1.,.5,.75)}
    initial = state_digest(model)
    out.mkdir(parents=True,exist_ok=False)
    protocol = dict(schema="frozen-s7-physical-scale/1",status="running",model=identity,
        probe_root=str(root),cases_sha256=CASES_SHA,selection_sha256=SELECTION_SHA,
        source_protocol_sha256=sha(root/"protocol.json"),source_root=str(Path(args.source_root).resolve()),
        script_sha256=sha(__file__),geometry_sha256=sha(Path(__file__).with_name("geometry.py")),
        original_config=asdict(base.config),initial_state_sha256=initial,
        scales=[.5,.75],window_sizes={str(s):list(v.window_sizes_px) for s,v in samplers.items()},
        common_center_rc=[399.5,399.5],raster_sampling="inverse nearest, align_corners=True, zero outside",
        point_sampling="same original512 identities, order, validity and padding; no contour re-extraction",
        branches=dict(R="original encoded features and coordinates",
            A="scaled raster/points; original window spans",
            B="scaled raster/points; window offsets multiplied by scale",
            C="B encoded features; original context coordinates (counterfactual)",
            D="R encoded features; scaled context coordinates (counterfactual)"),
        cpu_threads=1,GPU_used=False,training=False,threshold_fit=False,GT_used=False,layout_evaluated=False,
        limitations=["Existing40 score-stratified diagnostic pairs, not a representative benchmark.",
            "C/D are off-manifold interventions, not deployable scale normalization.",
            "Residual R-to-C includes binary raster quantization and local sampling changes.",
            "Only .5/.75 shrinking, fixed token identities, frozen S7C8 scorer path; no accuracy claim."],
        completed_count=0)
    save(out/"protocol.json",protocol)
    rows, started = [], time.monotonic()
    try:
        with torch.inference_mode():
            for selected, case in selected_cases:
                path = (root/case["arrays_path"]).resolve(strict=True)
                if root not in path.parents or sha(path) != case["arrays_sha256"]:
                    raise ValueError("unexpected archived input")
                with np.load(path,allow_pickle=False) as ar:
                    masks = tuple(torch.from_numpy(ar["mask_"+s][None,None].astype(np.float32)) for s in "ab")
                    points = tuple(torch.from_numpy(ar["points_rc_"+s][None].astype(np.float32)) for s in "ab")
                    valid = tuple(torch.from_numpy(ar["valid_"+s][None].copy()) for s in "ab")
                if any(p.shape != (1,512,2) for p in points):
                    raise ValueError("original token identities changed")
                evidence_r = encode(base,base.patch_sampler,masks,points,valid)
                row_r, snapshot_r = score(model,evidence_r,points,valid)
                stored_delta = abs(row_r["logit"]-case["raw_head_logit"])
                if stored_delta > 2e-4:
                    raise ValueError("original scorer pipeline does not reproduce archived logit")
                if not rows:
                    identity_pair = downscale_pair(*masks,*points,*valid,scale=1.)
                    for x,y in zip((*masks,*points,*valid),identity_pair.inputs):
                        torch.testing.assert_close(x,y,rtol=0,atol=0)
                    ev1 = encode(base,samplers[1.],identity_pair.inputs[:2],identity_pair.inputs[2:4],valid)
                    result1,snapshot1 = score(model,ev1,identity_pair.inputs[2:4],valid)
                    for name in ("patches","encoded","context"):
                        for x,y in zip(snapshot_r[name],snapshot1[name]):
                            torch.testing.assert_close(x,y,rtol=0,atol=0)
                    if result1 != row_r:
                        raise ValueError("s=1 scorer identity failed")
                    protocol["identity_scale1_verified"] = True
                row = dict(pair_id=case["pair_id"],dataset=case["dataset"],name=selected["name"],
                    stratum=selected["stratum"],arrays_sha256=case["arrays_sha256"],R=row_r,
                    stored_raw_logit_delta=stored_delta,
                    original_boundary_distance={s:boundary_distance(masks[i],points[i],valid[i]) for i,s in enumerate("ab")},
                    scales={})
                for scale in (.5,.75):
                    changed = downscale_pair(*masks,*points,*valid,scale=scale)
                    new_masks, new_points = changed.inputs[:2],changed.inputs[2:4]
                    positional_error = []
                    inverse_window_error = []
                    for i in range(2):
                        keep = valid[i]
                        expected = scale*(points[i][keep]*2./799.-1.)
                        actual = new_points[i][keep]*2./799.-1.
                        positional_error.append(float((expected-actual).abs().max()))
                        original_locations = points[i][keep][:,None,None,None,:]+base.patch_sampler.offsets_rc[None]
                        new_locations = new_points[i][keep][:,None,None,None,:]+samplers[scale].offsets_rc[None]
                        inverse = 399.5+(new_locations-399.5)/scale
                        err = float((original_locations-inverse).abs().max())
                        if err > 2e-4:
                            raise ValueError("compensated sampling physical coordinates do not coincide")
                        inverse_window_error.append(err)
                    evidence_a = encode(base,base.patch_sampler,new_masks,new_points,valid)
                    evidence_b = encode(base,samplers[scale],new_masks,new_points,valid)
                    branch_rows, snapshots = {},{}
                    for name,evidence,coords in (("A",evidence_a,new_points),("B",evidence_b,new_points),
                                                 ("C",evidence_b,points),("D",evidence_r,new_points)):
                        branch_rows[name],snapshots[name] = score(model,evidence,coords,valid)
                        branch_rows[name]["difference_from_R"] = compare_snapshots(snapshot_r,snapshots[name],valid)
                    row["scales"][str(scale)] = dict(geometry=changed.diagnostics,
                        boundary_distance={s:boundary_distance(new_masks[i],new_points[i],valid[i]) for i,s in enumerate("ab")},
                        normalized_position_scale_identity_max_error=positional_error,
                        compensated_inverse_window_max_error_px=inverse_window_error,
                        branches=branch_rows,
                        factorial_logit_interaction=factorial(dict(branch_rows,R=row_r)))
                rows.append(row)
                save(out/"cases.json",rows)
                protocol.update(completed_count=len(rows),elapsed_seconds=time.monotonic()-started)
                save(out/"protocol.json",protocol)
                print(json.dumps(dict(completed=len(rows),pair_id=row["pair_id"])),flush=True)
        final = state_digest(model)
        if final != initial:
            raise ValueError("original model state mutated")
        save(out/"summary.json",summarize(rows))
        protocol.update(status="complete",final_state_sha256=final,original_state_unchanged=True,
                        cases_output_sha256=sha(out/"cases.json"),elapsed_seconds=time.monotonic()-started)
        save(out/"protocol.json",protocol)
    except BaseException as exc:
        protocol.update(status="failed",error=f"{type(exc).__name__}: {exc}",completed_count=len(rows),
                        elapsed_seconds=time.monotonic()-started)
        save(out/"protocol.json",protocol)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root",required=True)
    parser.add_argument("--probe-root",required=True)
    parser.add_argument("--output",required=True)
    run(parser.parse_args())
