"""Fixed S7 / all predeclared 40 pairs: R original, A arc512, B arc1024,
C arc1024 with functional dilation2. CPU1 only; no training or threshold fit.

--source-root is the existing sealed project source; --context-helper is the
unchanged context_dilation_v1/functional_context.py file. No source is patched.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True
import numpy as np
from scipy import ndimage
import torch

CHECKPOINT_SHA = "7c1212e2d9d62c25954457add3f9319b03dfe8fbc42aa96116e40caae0dc2c37"
CASES_SHA = "7bab8e29a348e1ea62607bcf45376e6fe6e8dac2f2b7802d6b7415463d8544cf"
SELECTION_SHA = "071d14418a3b7cafee408b17d29a0edba175992206ce3a8be20048c03c259d37"


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


def difference(reference, current):
    if reference.shape != current.shape or not torch.isfinite(reference).all() or not torch.isfinite(current).all():
        raise ValueError("unaligned or nonfinite feature comparison")
    absolute = torch.linalg.vector_norm(reference-current).item()
    norm = torch.linalg.vector_norm(reference).item()
    return dict(max_abs=float((reference-current).abs().max()), absolute_l2=absolute,
                reference_l2=norm, relative_l2=absolute/norm if norm else None)


def uniform_sets(masks, dense_extractor, arc_resampler):
    points = {512: {}, 1024: {}}
    detail = {}
    for side in "ab":
        dense = dense_extractor(masks[side].astype(bool))
        smooth = ndimage.gaussian_filter1d(dense, sigma=3., axis=0, mode="wrap")
        perimeter = float(np.linalg.norm(np.roll(smooth, -1, axis=0)-smooth, axis=1).sum())
        points[1024][side] = arc_resampler(smooth, 1024).astype(np.float32)
        # Nested grid by construction, independent of floating operation order.
        points[512][side] = points[1024][side][::2].copy()
        np.testing.assert_allclose(points[512][side], arc_resampler(smooth, 512), rtol=0, atol=5e-5)
        # Unlike original512, this diagnostic always resamples, including short contours.
        np.testing.assert_array_equal(points[512][side], points[1024][side][::2])
        for offset in (-2, -1, 0, 1, 2):
            np.testing.assert_array_equal(points[512][side][(np.arange(512)+offset) % 512],
                points[1024][side][(2*np.arange(512)+2*offset) % 1024])
        detail[side] = dict(dense_count=len(dense), smoothed_perimeter_px=perimeter,
            arc_step512_px=perimeter/512, arc_step1024_px=perimeter/1024,
            exact_even_anchors=True, exact_dilation2_neighbors=True)
    return points, detail


def head_score(head, features, valid):
    logit = head(*features, *valid)[0]
    if not torch.isfinite(logit):
        raise ValueError("nonfinite raw Scorer logit")
    return dict(logit=float(logit), probability=float(logit.sigmoid()),
                valid_counts=[int(v.sum()) for v in valid])


@torch.inference_mode()
def forward(model, masks, points, valid, functional_context, decoder, decoder_config,
            *, dilation=1, verify_d1=False):
    """The hook replaces context BEFORE the original Matcher continues to OT.

    Patch and encoded snapshots are transient; outputs are only compact numbers.
    R retains the full stored point tensors/padding rather than compacting them.
    """
    base = model.base_model
    snapshot = {"patches": []}

    def capture_patches(module, inputs, output):
        snapshot["patches"].append(output.detach().clone())

    def replace_context(module, inputs, output):
        snapshot["encoded"] = (inputs[0].detach().clone(), inputs[1].detach().clone())
        if verify_d1:
            replay = functional_context(module, *inputs, dilation=1)
            for expected, actual in zip(output, replay):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        selected = functional_context(module, *inputs, dilation=2) if dilation == 2 else output
        snapshot["context"] = tuple(x.detach().clone() for x in selected)
        return selected

    handles = [base.patch_sampler.register_forward_hook(capture_patches),
               base.context.register_forward_hook(replace_context)]
    tensors = [torch.from_numpy(masks[s][None, None].astype(np.float32)) for s in "ab"]
    tensors += [torch.from_numpy(points[s][None].astype(np.float32)) for s in "ab"]
    tensors += [torch.from_numpy(valid[s][None].copy()) for s in "ab"]
    try:
        output = base(*tensors)
    finally:
        for handle in handles:
            handle.remove()
    if len(snapshot["patches"]) != 2 or not bool(output.training_valid[0]):
        raise ValueError("unexpected patch call count or invalid Matcher forward; never drop a case")
    features = (output.token_features_a, output.token_features_b)
    for expected, actual in zip(snapshot["context"], features):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    pose = decoder(points["a"], points["b"], output.assignment[0].cpu().numpy(),
                   valid["a"], valid["b"], config=decoder_config)
    row = dict(full=head_score(model.score_head, features, tensors[-2:]),
        layout=dict(valid=bool(pose.valid), candidate_count=int(pose.candidate_count),
            inlier_count=int(pose.inlier_count), residual_px=pose.residual_px,
            translation_a_to_b_rc=pose.t_a_to_b_rc.tolist() if pose.valid else None,
            reason=pose.reason), sinkhorn_converged=bool(output.transport.diagnostics.converged[0]),
        functional_dilation1_verified=verify_d1)
    return row, snapshot


def compare_anchors(head, reference, current):
    error = {}
    for i, side in enumerate("ab"):
        error[side] = {name: difference(reference[name][i], current[name][i][:, ::2])
                       for name in ("patches", "encoded", "context")}
        # Identical physical coordinates must produce identical local evidence.
        for name in ("patches", "encoded"):
            torch.testing.assert_close(reference[name][i], current[name][i][:, ::2], rtol=0, atol=0)
    features = tuple(v[:, ::2] for v in current["context"])
    valid = [torch.ones(v.shape[:2], dtype=torch.bool) for v in features]
    return dict(score=head_score(head, features, valid), errors_to_A=error)


def read_cases(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if (protocol.get("status") != "complete" or protocol.get("sample_count") != 40
            or protocol.get("completed_count") != 40 or protocol.get("selection_json_sha256") != SELECTION_SHA
            or protocol.get("cases_sha256") != CASES_SHA or sha(root / "cases.json") != CASES_SHA
            or protocol["model"]["checkpoint_sha256"] != CHECKPOINT_SHA):
        raise ValueError("requires the complete, exact predeclared40 S7 C8 heatmap snapshot")
    cases = json.loads((root / "cases.json").read_text())
    indexed = {r["pair_id"]: r for r in cases}
    chosen = protocol["selected_pairs"]
    ids = [r["pair_id"] for r in chosen]
    if len(cases) != 40 or len(indexed) != 40 or len(ids) != 40 or len(set(ids)) != 40 or set(ids) != set(indexed):
        raise ValueError("do not subset, duplicate or change the existing40 cases")
    return protocol, [(r, indexed[r["pair_id"]]) for r in chosen]


def run(args):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only diagnostic")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    sys.path.insert(0, str(Path(args.source_root).resolve(strict=True)))
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as ev
    from staging.pairwise_v0_2.models.translation_layout import estimate_translation_layout
    from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import _dense_external_contour, _arc_resample_closed
    helper_path = Path(args.context_helper).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("density_v2_functional_context", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    root, out = Path(args.probe_root).resolve(strict=True), Path(args.output).resolve()
    source, chosen = read_cases(root)
    model, identity = ev.load_frozen_model(source["model"]["training_run"], source["model"]["selection"])
    if identity["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("wrong S7 checkpoint")
    model.cpu().eval().requires_grad_(False)
    base = model.base_model
    original_config = base.config
    if (original_config.canvas_size != 800 or original_config.coarse_size != 128
            or original_config.contour_cap != 512 or list(original_config.window_sizes_px) != [7., 16., 32., 64.]):
        raise ValueError("unexpected S7 physical input/patch configuration")
    decoder_config = ev.core.fixed.TOP2_CONFIG
    if asdict(decoder_config) != source["decoder"]:
        raise ValueError("production layout decoder differs from original snapshot")
    initial_digest = state_digest(model)
    conv_config = [(m.padding, m.dilation) for m in base.context.modules() if isinstance(m, torch.nn.Conv1d)]
    base.config = model.config = replace(original_config, contour_cap=1024)
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema="density-physical-context/2", status="running", model=identity,
        case_count=40, cases_sha256=CASES_SHA, selection_json_sha256=SELECTION_SHA,
        source_protocol_sha256=sha(root / "protocol.json"), context_helper_sha256=sha(helper_path),
        script_sha256=sha(__file__), source_root=str(Path(args.source_root).resolve()),
        context_source_sha256=sha(sys.modules[type(base.context).__module__].__file__),
        head_source_sha256=sha(sys.modules[type(model.score_head).__module__].__file__),
        decoder_config=asdict(decoder_config), original_config=asdict(original_config),
        parameter_sha256=initial_digest, cpu_threads=1, GPU_used=False, training=False,
        threshold_fit=False, GT_used=False, physical_masks_resized=False, coarse_size_changed=False,
        branches=dict(R="original stored points/valid, including padding; original context",
            A="always equal-arc512; dilation1 (bridge, NOT production original512)",
            B="same smooth loop equal-arc1024; dilation1",
            C="same B points; functional dilation2 in both circular Conv1d per block"),
        selection="all existing40 score-stratified cases, fixed membership; no new case selection",
        limitations=["Per-fragment512 physical convolution spacing is preserved, not a universal fixed3px neighborhood.",
            "A intentionally upsamples/resamples short contours; R-to-A bridge must be shown separately.",
            "GN and landmark CA still see1024 tokens; not separately isolated.",
            "Anchor-only Scorer changes both attention and pooling, not pooling alone.",
            "Frozen512-trained S7, selected40 diagnostics; no population accuracy/generalization claim."])
    save(out / "protocol.json", protocol)
    rows, started = [], time.monotonic()
    try:
        with torch.inference_mode():
            for index, (selected, case) in enumerate(chosen):
                archive_path = (root / case["arrays_path"]).resolve(strict=True)
                if root not in archive_path.parents or sha(archive_path) != case["arrays_sha256"]:
                    raise ValueError("case arrays path/SHA differs")
                with np.load(archive_path, allow_pickle=False) as archive:
                    masks = {s: archive["mask_"+s].copy() for s in "ab"}
                    original_points = {s: archive["points_rc_"+s].copy() for s in "ab"}
                    original_valid = {s: archive["valid_"+s].copy() for s in "ab"}
                if any(masks[s].shape != (800, 800) or not np.isin(masks[s], (0, 1)).all()
                       or original_points[s].shape != (512, 2) or original_valid[s].shape != (512,)
                       or original_valid[s].dtype != np.bool_ for s in "ab"):
                    raise ValueError("saved physical mask/point/valid shape differs")
                points, sampling = uniform_sets(masks, _dense_external_contour, _arc_resample_closed)
                branch, snapshots = {}, {}
                for name, count, dilation in (("R", None, 1), ("A", 512, 1), ("B", 1024, 1), ("C", 1024, 2)):
                    ps = original_points if count is None else points[count]
                    valid = original_valid if count is None else {s: np.ones(count, bool) for s in "ab"}
                    measured, snapshot = forward(model, masks, ps, valid, helper.context_forward,
                        estimate_translation_layout, decoder_config, dilation=dilation,
                        verify_d1=(index == 0 and dilation == 1))
                    branch[name] = measured
                    if name == "R":
                        measured["historical_logit_abs_error"] = abs(measured["full"]["logit"]-case["raw_head_logit"])
                        if measured["historical_logit_abs_error"] > 2e-4:
                            raise ValueError("original padded512 historical Scorer replay failed")
                    elif name == "A":
                        snapshots["A"] = snapshot
                    else:
                        measured["anchors512"] = compare_anchors(model.score_head, snapshots["A"], snapshot)
                row = dict(pair_id=case["pair_id"], dataset=case["dataset"], name=selected.get("name"),
                    stratum=selected.get("stratum"), fragment_a=case["fragment_a"], fragment_b=case["fragment_b"],
                    arrays_sha256=case["arrays_sha256"], sampling=sampling, branches=branch,
                    bridge_A_minus_R=dict(logit=branch["A"]["full"]["logit"]-branch["R"]["full"]["logit"],
                        probability=branch["A"]["full"]["probability"]-branch["R"]["full"]["probability"]))
                rows.append(row)
                save(out / "partial_results.json", rows)
                print(json.dumps(dict(completed_cases=len(rows), pair_id=row["pair_id"],
                    probabilities={k: v["full"]["probability"] for k, v in branch.items()})), flush=True)
        if state_digest(model) != initial_digest or conv_config != [(m.padding, m.dilation) for m in base.context.modules() if isinstance(m, torch.nn.Conv1d)]:
            raise ValueError("original weights/conv configuration changed")
        if torch.cuda.is_initialized():
            raise RuntimeError("unexpected GPU initialization")
        protocol.update(status="complete", completed_cases=40, parameters_unchanged=True,
                        original_conv_configuration_unchanged=True, elapsed_seconds=time.monotonic()-started)
        save(out / "results.json", dict(protocol=protocol, rows=rows))
    except BaseException as error:
        protocol.update(status="failed", completed_cases=len(rows), error=repr(error))
        raise
    finally:
        base.config = model.config = original_config
        save(out / "protocol.json", protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source-root", "context-helper", "probe-root", "output"):
        parser.add_argument("--"+field, required=True)
    run(parser.parse_args())
