"""Finite CPU-only input interventions on the completed matched_edges head.

Weights, selected endpoints, edge identities, predicted displacement and frozen
features never change. Scaling input channels is not a separately trained
ablation, and zero residual means ideal fit rather than missing information.
"""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

VARIANTS = {
    "noop": (1., 1., False),
    "q_zero": (0., 1., False),
    "q_half": (.5, 1., False),
    "q_double": (2., 1., False),
    "residual_zero": (1., 0., False),
    "residual_half": (1., .5, False),
    "residual_double": (1., 2., False),
    "both_zero": (0., 0., False),
    "mate_zero": (1., 1., True),
}


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


@contextmanager
def intervention(model, variant):
    """Post-selection metadata scaling preserves projection bias and symmetry."""
    qs, rs, mate_zero = VARIANTS[variant]
    def meta_hook(module, inputs):
        meta = inputs[0].clone()
        meta[..., 0] *= qs
        meta[..., 1] *= rs
        return (meta,)
    def mate_hook(module, inputs, output):
        return torch.zeros_like(output)
    handles = [model.edge_projection.register_forward_pre_hook(meta_hook)]
    if mate_zero:
        handles.append(model.mate_projection.register_forward_hook(mate_hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def dist(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                min=float(values.min()), max=float(values.max())) if len(values) else dict(n=0)


@torch.inference_mode()
def probe(model, args, kwargs, metadata, thresholds):
    baseline = model(*args, **kwargs)
    if not bool(baseline.has_decoded_candidate[0]) or bool(baseline.used_fallback[0]):
        return dict(**metadata, skipped_intervention=True,
                    reason="no decoded evidence; do not fabricate edge input", baseline_logit=float(baseline.logit[0]))
    snapshots = {}
    def capture(module, inputs, output):
        snapshots["meta"] = inputs[0].clone()
    h = model.edge_projection.register_forward_hook(capture)
    try:
        replay = model(*args, **kwargs)
    finally:
        h.remove()
    torch.testing.assert_close(replay.logit, baseline.logit, rtol=0, atol=0)
    selection = args[4]
    active = selection.candidate_valid & selection.candidate_inliers & selection.layout_valid[:, None]
    meta = snapshots["meta"][active]
    base = float(baseline.logit[0])
    values = {}
    for name in VARIANTS:
        with intervention(model, name):
            output = model(*args, **kwargs)
        logit = float(output.logit[0])
        if not np.isfinite(logit):
            raise ValueError("nonfinite intervention")
        values[name] = dict(logit=logit, probability=float(output.logit[0].sigmoid()),
                           delta_logit=logit-base,
                           accepted={k: float(output.logit[0].sigmoid()) >= v for k, v in thresholds.items()})
    if abs(values["noop"]["delta_logit"]) > 1e-6:
        raise ValueError("no-op replay failed")
    # An exactly zero input channel removes only its linear contribution.
    w = model.edge_projection.weight
    q_effect = meta[:, :1] * w[:, 0][None, :]
    r_effect = meta[:, 1:] * w[:, 1][None, :]
    def rms(x):
        return float(x.square().mean().sqrt())
    edges = selection.candidate_indices[active]
    ea, eb = args[0][0, edges[:, 0]], args[1][0, edges[:, 1]]
    return dict(**metadata, skipped_intervention=False, baseline_logit=base,
                baseline_probability=float(baseline.logit[0].sigmoid()),
                accepted={k: float(baseline.logit[0].sigmoid()) >= v for k, v in thresholds.items()},
                unique_a=int(selection.mask_a.sum()), unique_b=int(selection.mask_b.sum()),
                inlier_edges=int(active.sum()), fixed_translation_rc=selection.translation_a_to_b_rc[0].tolist(),
                q=dist(meta[:, 0].tolist()), residual_norm=dist(meta[:, 1].tolist()),
                projection_rms=dict(q=q_effect.square().mean().sqrt().item(), residual=rms(r_effect),
                    bias=rms(model.edge_projection.bias), self_a=rms(model.self_projection(ea)),
                    mate_b=rms(model.mate_projection(eb))), interventions=values)


def select_val(cache):
    minimum = np.minimum(cache.arrays["mask_a"].sum(1), cache.arrays["mask_b"].sum(1))
    selected, availability = [], {}
    for label in (0., 1.):
        for name, lo, hi in (("le32", 1, 32), ("33_64", 33, 64), ("65_128", 65, 128), ("129_plus", 129, 1024)):
            ids = [i for i, r in enumerate(cache.records) if r["label"] == label
                   and lo <= minimum[i] <= hi and cache.arrays["layout_valid"][i]]
            ids.sort(key=lambda i: cache.records[i]["pair_id"])
            group = "val_%s_%s" % ("positive" if label else "negative", name)
            availability[group] = dict(available=len(ids), selected=min(8, len(ids)))
            selected.extend(dict(ordinal=i, pair_id=cache.records[i]["pair_id"], label=label,
                                 group=group, split="val") for i in ids[:8])
    return selected, availability


def summarize(cases):
    groups = {}
    for group in sorted({r["group"] for r in cases}):
        rows = [r for r in cases if r["group"] == group and not r["skipped_intervention"]]
        variants = {}
        for name in VARIANTS:
            variants[name] = dict(delta_logit=dist([r["interventions"][name]["delta_logit"] for r in rows]),
                increased=sum(r["interventions"][name]["delta_logit"] > 1e-5 for r in rows),
                decreased=sum(r["interventions"][name]["delta_logit"] < -1e-5 for r in rows),
                accepted={k: sum(r["interventions"][name]["accepted"][k] for r in rows)
                          for k in ("max_f1", "recall_99")})
        groups[group] = dict(n=len(rows), baseline_logit=dist([r["baseline_logit"] for r in rows]),
            baseline_accepted={k: sum(r["accepted"][k] for r in rows) for k in ("max_f1", "recall_99")},
            q_mean=dist([r["q"]["mean"] for r in rows]),
            residual_mean=dist([r["residual_norm"]["mean"] for r in rows]), variants=variants)
    return groups


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.dont_write_bytecode = True
    from matched_only import data, evaluate
    from edge_metadata_cases import select_real_cases, iter_real_inputs
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    adapter, receipt = evaluate.load_frozen_model(args.training_run, "fixed_epoch", budget=16)
    adapter.cpu().eval().requires_grad_(False)
    model = adapter.score_head
    if model.arm != "matched_edges":
        raise ValueError("requires completed matched_edges")
    thresholds = {k: receipt["operating_points"]["thresholds"][k] for k in ("max_f1", "recall_99")}
    cache = data.FormalCache(args.cache, "val")
    selected, availability = select_val(cache)
    real_selected = (json.loads(Path(args.selection_json).read_text()) if args.selection_json
                     else select_real_cases(Path(args.results_root)))
    if len(real_selected) != 42 or len({r["pair_id"] for r in real_selected}) != 42:
        raise ValueError("expected the fixed unique42 diagnostic REAL cases")
    if any(r["checkpoint_sha256"] != receipt["checkpoint_sha256"]
           or r["source_matcher_sha256"] != receipt["source_matcher_sha256"] for r in real_selected):
        raise ValueError("saved REAL source model differs from replay model")
    protocol = dict(schema="frozen-edge-metadata-interventions/1", status="running", device="cpu", threads=1,
        checkpoint_sha256=receipt["checkpoint_sha256"], matcher_sha256=receipt["source_matcher_sha256"],
        variants=VARIANTS, thresholds=thresholds, val_selected=selected, real_selected=real_selected,
        availability=availability, selection_committed_before_scoring=True,
        selection_json_sha256=data.sha(args.selection_json) if args.selection_json else None,
        training=False, checkpoint_writes=False, no_layout_redecode_during_interventions=True,
        caveats=["Inference interventions are not separately trained ablations or deployment recommendations.",
                 "Zero residual represents artificially perfect fit, not absent evidence.",
                 "Q/residual scaling and removing mate projection can be out of training distribution.",
                 "REAL groups are outcome-selected diagnostic cases, not unbiased metric estimates.",
                 "Frozen SIMVAL thresholds are not recalibrated for altered inputs."])
    save(out / "protocol.json", protocol)
    cases = []
    try:
        with (out / "cases.jsonl").open("x") as stream:
            def append(row):
                cases.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            for item in selected:
                batch = cache.batch([item["ordinal"]], "cpu")
                append(probe(model, batch.model_args, batch.model_kwargs, item, thresholds))
            for item, inputs in iter_real_inputs(real_selected, device="cpu"):
                captured = {}
                def capture(module, model_args, model_kwargs):
                    captured.update(args=model_args, kwargs=model_kwargs)
                handle = model.register_forward_pre_hook(capture, with_kwargs=True)
                try:
                    with torch.inference_mode():
                        output = adapter(*inputs)
                finally:
                    handle.remove()
                row = probe(model, captured["args"], captured["kwargs"], item, thresholds)
                if row["skipped_intervention"]:
                    raise ValueError("saved valid REAL layout lost its evidence on replay: " + item["pair_id"])
                row["online_training_valid"] = bool(output.training_valid[0])
                row["online_decision_valid"] = bool(output.decision_valid[0])
                row["saved_logit_replay_error"] = abs(float(output.fused_logit[0])-item["expected_logit"])
                row["saved_translation_replay_error_px"] = float(np.linalg.norm(
                    np.asarray(row["fixed_translation_rc"])-np.asarray(item["expected_translation_rc"])))
                if (row["saved_logit_replay_error"] > 1e-3 or row["saved_translation_replay_error_px"] > .01
                        or not row["online_training_valid"]):
                    raise ValueError("CPU vs saved formal prediction differs: " + item["pair_id"])
                append(row)
        protocol.update(status="complete", cases=len(cases), elapsed_s=time.monotonic()-start)
        save(out / "summary.json", dict(status="complete", cases=len(cases), groups=summarize(cases),
                                         elapsed_s=protocol["elapsed_s"]))
        save(out / "protocol.json", protocol)
        print(json.dumps(dict(status="complete", cases=len(cases), elapsed_s=protocol["elapsed_s"])))
    except BaseException as exc:
        protocol.update(status="failed", error=repr(exc), completed_cases=len(cases), elapsed_s=time.monotonic()-start)
        save(out / "protocol.json", protocol)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-run", "cache", "output"):
        parser.add_argument("--"+name, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-root")
    source.add_argument("--selection-json")
    run(parser.parse_args())
