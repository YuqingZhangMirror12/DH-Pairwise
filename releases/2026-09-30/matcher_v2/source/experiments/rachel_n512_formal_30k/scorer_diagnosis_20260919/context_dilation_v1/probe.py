"""Four preselected exact-anchor cases, frozen S6D2 C8, CPU1, no accuracy claims."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import torch
from functional_context import context_forward

SOURCE_SHA = "56d3a4949e9d7e10f6ab5bdadc0b8a50d17192a7130584c3b5d22e21ce4d2076"
EXPECTED_IDS = {
    "pair/sha256/d1d9bf23b0040e4a85fc1d4fb409c17f212642cd14242241cf4e5d28b230595b",
    "pair/sha256/38fe27eb76a312c5c02c3b25663eb4e74844e8a9760c5af2c2485b7f71ad0130",
    "pair/sha256/7eec1bf63820ba371eba29259afe54948b6a000cb2bbde352f567fb16c5e2e70",
    "rachel-within-1103d0e4862ced74dc3059da",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def difference(reference, value):
    return dict(max_abs=float((reference - value).abs().max()),
        relative_l2=float(torch.linalg.vector_norm(reference - value) / torch.linalg.vector_norm(reference)),
        mean_cosine=float(torch.nn.functional.cosine_similarity(reference, value, dim=-1).mean()))


def head_score(head, features):
    valid = [torch.ones(value.shape[:2], dtype=torch.bool) for value in features]
    z = head(*features, *valid)[0]
    if not torch.isfinite(z):
        raise ValueError("nonfinite frozen-head result")
    return dict(logit=float(z), probability=float(z.sigmoid()), counts=[value.shape[1] for value in features])


def capture(base, masks, points):
    snapshot = {}
    def before(module, inputs):
        snapshot["inputs"] = tuple(x.clone() if torch.is_tensor(x) else x for x in inputs)
    def after(module, inputs, outputs):
        snapshot["outputs"] = tuple(x.clone() for x in outputs)
    handles = [base.context.register_forward_pre_hook(before), base.context.register_forward_hook(after)]
    tensors = [torch.tensor(masks[s][None, None], dtype=torch.float32) for s in "ab"]
    tensors += [torch.tensor(points[s][None], dtype=torch.float32) for s in "ab"]
    tensors += [torch.ones(1, len(points[s]), dtype=torch.bool) for s in "ab"]
    try:
        output = base(*tensors)
    finally:
        for handle in handles:
            handle.remove()
    if not bool(output.training_valid[0]):
        raise ValueError("invalid base forward")
    for side, feature in zip("ab", snapshot["outputs"]):
        torch.testing.assert_close(feature, getattr(output, "token_features_" + side), rtol=0, atol=0)
    return snapshot


def run(args):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU only")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    sys.path.insert(0, args.source_root)
    sys.path.insert(0, args.sampling_source)
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as ev
    from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour
    # Explicitly load the already reviewed sampling helper without shadowing this script.
    import runpy
    anchor_indices = runpy.run_path(str(Path(args.sampling_source) / "probe.py"))["anchor_indices"]
    root, out = Path(args.probe_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    previous = json.loads(Path(args.count_results).read_text())
    selected = [r for r in previous["rows"] if all(v["maximum_distance_px"] == 0 and v["unique_current_anchors"] == 512
        for v in next(x for x in r["variants"] if x["variant"] == "resampled1024")["anchor_512"]["sides"].values())]
    if len(selected) != 4 or {r["pair_id"] for r in selected} != EXPECTED_IDS:
        raise ValueError("only the four predeclared exact-anchor cases are allowed")
    if previous["protocol"]["status"] != "complete" or previous["protocol"]["source_checkpoint_sha256"] != SOURCE_SHA:
        raise ValueError("prior completed count experiment/source differs")
    source = json.loads((root / "protocol.json").read_text())
    cases = {r["pair_id"]: r for r in json.loads((root / "cases.json").read_text())}
    model, identity = ev.load_frozen_model(source["model"]["training_run"], source["model"]["selection"])
    if identity["checkpoint_sha256"] != SOURCE_SHA:
        raise ValueError("not original S6D2 C8 checkpoint")
    model.cpu().eval().requires_grad_(False)
    base = model.base_model
    original_config = asdict(base.config)
    before_state = state_digest(model)
    model.config = base.config = replace(base.config, contour_cap=1024)
    config_before = [(m.padding, m.dilation) for m in base.context.modules() if isinstance(m, torch.nn.Conv1d)]
    protocol = dict(status="running", schema_version="fixed-context-convolution-dilation/1",
        source_checkpoint_sha256=SOURCE_SHA, source_root=args.source_root, probe_root=str(root),
        count_results_sha256=sha(args.count_results), cases_sha256=sha(root / "cases.json"),
        sampling_helper_sha256=sha(Path(args.sampling_source) / "probe.py"),
        original_context_source_sha256=sha(sys.modules[type(base.context).__module__].__file__),
        original_head_source_sha256=sha(sys.modules[type(model.score_head).__module__].__file__),
        script_sha256=sha(__file__), functional_context_sha256=sha(Path(__file__).with_name("functional_context.py")),
        original_config=original_config, cpu_threads=1, torch_version=str(torch.__version__), GPU_used=False, GT_read_by_experiment=False,
        training_or_threshold_fit=False, checkpoint_selection=False, original_parameters_sha256=before_state,
        intervention="only functional circular Conv1d dilation2 and matching padding at cap1024; GN/position/landmarkCA/head unchanged",
        selection="same four previously selected cases having exact512 anchors on both1024 sides; not a representative population")
    save(out / "protocol.json", protocol)
    rows, started = [], time.monotonic()
    try:
        with torch.inference_mode():
            for chosen in selected:
                case = cases[chosen["pair_id"]]
                path = root / case["arrays_path"]
                if sha(path) != case["arrays_sha256"]:
                    raise ValueError("saved masks changed")
                with np.load(path, allow_pickle=False) as archive:
                    masks = {s: archive["mask_" + s].copy() for s in "ab"}
                if any(not np.isin(masks[s], (0, 1)).all() for s in "ab"):
                    raise ValueError("physical masks must be binary")
                points, captured = {}, {}
                d1_checks = {}
                for cap in (512, 1024):
                    points[cap] = {s: extract_ordered_outer_contour(masks[s].astype(bool), cap=cap, smoothing_sigma=3.)[0] for s in "ab"}
                    if any(len(points[cap][s]) != cap for s in "ab"):
                        raise ValueError("both sides must have exact requested counts")
                    captured[cap] = capture(base, masks, points[cap])
                    d1 = context_forward(base.context, *captured[cap]["inputs"], dilation=1)
                    d1_checks[str(cap)] = [difference(a, b) for a, b in zip(captured[cap]["outputs"], d1)]
                    for a, b in zip(captured[cap]["outputs"], d1):
                        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
                indices, sides = {}, {}
                dilated = context_forward(base.context, *captured[1024]["inputs"], dilation=2)
                for i, s in enumerate("ab"):
                    ids, detail = anchor_indices(points[512][s], points[1024][s])
                    if detail["maximum_distance_px"] != 0 or detail["unique_current_anchors"] != 512:
                        raise ValueError("expected exact unique physical anchors")
                    for offset in (-2, -1, 0, 1, 2):
                        np.testing.assert_array_equal(points[1024][s][(ids + 2 * offset) % 1024],
                            points[512][s][(np.arange(512) + offset) % 512])
                    indices[s] = ids
                    a, b = captured[512]["inputs"][i], captured[1024]["inputs"][i][:, ids]
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    sides[s] = dict(anchors=detail, physical_convolution_neighbors_exact=True,
                        encoded=difference(a, b), normal1024_context=difference(captured[512]["outputs"][i], captured[1024]["outputs"][i][:, ids]),
                        dilated1024_context=difference(captured[512]["outputs"][i], dilated[i][:, ids]))
                baseline = head_score(model.score_head, captured[512]["outputs"])
                variant_rows = []
                for name, features in (("normal1024", captured[1024]["outputs"]), ("dilated1024", dilated)):
                    anchors = tuple(features[i][:, indices[s]] for i, s in enumerate("ab"))
                    variant_rows.append(dict(variant=name, full=head_score(model.score_head, features),
                        anchors512=head_score(model.score_head, anchors)))
                old512 = next(v for v in chosen["variants"] if v["variant"] == "resampled512")
                old1024 = next(v for v in chosen["variants"] if v["variant"] == "resampled1024")
                replay = dict(baseline512_logit_abs=abs(baseline["logit"] - old512["logit"]),
                    normal1024_full_logit_abs=abs(variant_rows[0]["full"]["logit"] - old1024["logit"]),
                    normal1024_anchors_logit_abs=abs(variant_rows[0]["anchors512"]["logit"] - old1024["anchor_512"]["logit"]))
                if max(replay.values()) > 2e-4:
                    raise ValueError("prior sampling-count score replay differs")
                row = dict(name=chosen["name"], pair_id=chosen["pair_id"], arrays_sha256=case["arrays_sha256"],
                    baseline512=baseline, variants=variant_rows, sides=sides, dilation1_replay=d1_checks,
                    prior_score_replay=replay)
                rows.append(row)
                save(out / "partial_results.json", rows)
                print(json.dumps(dict(name=row["name"], baseline=baseline["probability"],
                    variants={r["variant"]:dict(full=r["full"]["probability"], anchors=r["anchors512"]["probability"]) for r in variant_rows})), flush=True)
        if state_digest(model) != before_state or config_before != [(m.padding, m.dilation) for m in base.context.modules() if isinstance(m, torch.nn.Conv1d)]:
            raise ValueError("production module state/config mutated")
        if torch.cuda.is_initialized():
            raise RuntimeError("unexpected GPU initialization")
        protocol.update(status="complete", completed_cases=4, elapsed_seconds=time.monotonic() - started,
            parameters_unchanged=True, original_conv_configuration_unchanged=True)
        save(out / "results.json", dict(protocol=protocol, rows=rows,
            limitations=["Four fixed diagnostic cases; not accuracy or generalization evidence.",
                "Dilation preserves circular physical convolution neighbors at exact anchors, not the full512 computation.",
                "GroupNorm still normalizes1024 tokens and landmark cross-attention still sees1024; neither is isolated.",
                "Full-versus-anchor head scores change both attention and pooling; not a pooling-only test.",
                "No density retraining and no selected production configuration."]))
    except BaseException as error:
        protocol.update(status="failed", completed_cases=len(rows), error=repr(error))
        raise
    finally:
        save(out / "protocol.json", protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "sampling-source", "probe-root", "count-results", "output"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
