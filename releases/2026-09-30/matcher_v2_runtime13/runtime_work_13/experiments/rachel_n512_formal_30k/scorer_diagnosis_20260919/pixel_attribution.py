"""Supplementary sensitivity of the raw Scorer logit to binary-mask pixels.

No training, checkpoint mutation, RGB inference, integrated-gradients baseline,
or physical-damage claim. The mask is continuously relaxed ONLY to differentiate
at the observed binary input; contour coordinates and validity stay fixed.

The exact Scorer ancestry is patch sampling -> patch encoder -> matcher context
-> CrossAttentionPairHead. Replaying that ancestry bypasses the inference
wrapper's no_grad without building a discarded Sinkhorn backward graph. Every
case must reproduce the original complete frozen forward's raw head logit.

CLI defaults to eight predetermined strata from the existing 40-case selection:
REAL TP-good/FN-good/TP-bad/FP, and the first OOD case in each S6 score quartile.
Case choice is completed before gradients are calculated. This remains a small
post-hoc diagnostic, not a representative dataset or a causal region analysis.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from staging.pairwise_v0_2.models.rachel_decoupled_score import CrossAttentionPairHead, DecoupledScoreModel
from staging.pairwise_v0_2.models.rachel_n512 import ContourPatchSampler, RachelN512Pairwise
from staging.pairwise_v0_2.models.translation_layout import estimate_translation_layout
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.heatmap_probe import (
    FIELDS, FORWARD_ATOL, FORWARD_RTOL, _assert_close, _clean, _json, _numpy,
    attach_targets, selected_batches,
)

SCHEMA = "frozen-mask-pixel-sensitivity/1"
DEFAULT_STRATA = (
    ("real", "real_TP_good"), ("real", "real_FN_good"),
    ("real", "real_TP_bad"), ("real", "real_FP"),
    ("ood", "ood_s6_score_quartile_1"), ("ood", "ood_s6_score_quartile_2"),
    ("ood", "ood_s6_score_quartile_3"), ("ood", "ood_s6_score_quartile_4"),
)
INTERPRETATION = dict(
    target="raw pre-sigmoid CrossAttentionPairHead logit, before decision-validity gating",
    signed_gradient="d(raw_logit)/d(mask_pixel_value), evaluated at the original binary input; positive is the local increase direction where differentiable. At tied MaxPool or other nondifferentiable activations this is the framework-selected subgradient, not necessarily a unique directional derivative",
    magnitude="absolute pixel derivative; sensitivity, not a probability, additive contribution or causal effect",
    input_times_gradient="mask_pixel * derivative; foreground-only proxy, necessarily zero in background and NOT a substitute for background sensitivity",
    coordinates="exact prepared mask pixels, row/column; no RGB registration or RGB gradients",
    scope="complete patch/context/scorer derivative, with model weights, sampled contours and padding fixed",
    sampling="the original nearest-neighbor, zero-padded, align_corners=True mask sampler is replayed without its no_grad guard; only sampled mask values have a derivative, and repeated/overlapping pixel samples accumulate their downstream sensitivities",
    excluded="derivative through contour re-extraction, binarization, rescaling, layout selection, Sinkhorn, or classifier threshold",
    no_integrated_gradients="zero-mask baseline would mismatch the fixed contour; no IG baseline is used",
    no_physical_ablation="no default image deletion: keeping old contour after deleting paper is not a self-consistent physical input",
    finite_difference="small continuous central perturbations verify derivatives, may leave [0,1], and are NOT physical perturbations or additional real examples",
    nondifferentiability="binary patches can produce tied MaxPool values; central differences can disagree with the autograd-selected subgradient. Such discrepancies remain needs_review, not silently accepted by weakening tolerances",
    numerical_nondeterminism="CUDA grid_sample backward may use atomic accumulation; explicit opt-in can relax deterministic algorithms ONLY during gradient calculation, then restores the original setting",
)


def select_pixel_cases(rows, all_selected=False):
    if (not isinstance(rows, list) or not rows or any(not isinstance(row, dict)
            or row.get("dataset") not in ("real", "ood", "test")
            or not isinstance(row.get("pair_id"), str) for row in rows)):
        raise ValueError("requires an existing nonempty [{dataset,pair_id,stratum,...}] selection")
    if len({(row["dataset"], row["pair_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate dataset/pair IDs in selection")
    if all_selected:
        return list(rows)
    chosen = []
    for dataset, stratum in DEFAULT_STRATA:
        candidates = [row for row in rows if row["dataset"] == dataset and row.get("stratum") == stratum]
        if not candidates:
            raise ValueError("default eight-case stratum missing: %s/%s" % (dataset, stratum))
        chosen.append(candidates[0])
    return chosen


def differentiable_patches(sampler, masks, points_rc, valid):
    """Exact ContourPatchSampler operations, except for its internal no_grad.

    We do NOT change nearest sampling to bilinear: that would be a different
    function at the original fractional coordinates. Only mask pixels, not
    contour points or sampling coordinates, are differentiated by this probe.
    Each case separately compares these values to the original sampler.
    """
    if type(sampler) is not ContourPatchSampler:
        raise ValueError("unregistered patch sampler; refusing an assumed differentiable replay")
    batch, token_count = points_rc.shape[:2]
    centers = points_rc.to(dtype=torch.float32)[:, :, None, None, None, :]
    offsets = sampler.offsets_rc.to(device=points_rc.device, dtype=torch.float32)[None, None]
    sample_rc = centers + offsets
    denominator = float(sampler.canvas_size - 1)
    grid_x = sample_rc[..., 1] * (2.0 / denominator) - 1.0
    grid_y = sample_rc[..., 0] * (2.0 / denominator) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
        batch, token_count * sampler.scale_count * sampler.patch_size,
        sampler.patch_size, 2)
    sampled = F.grid_sample(masks.to(dtype=torch.float32), grid,
        mode="nearest", padding_mode="zeros", align_corners=True)
    sampled = sampled.reshape(batch, 1, token_count, sampler.scale_count,
        sampler.patch_size, sampler.patch_size).permute(0, 2, 3, 1, 4, 5)
    return sampled * valid[:, :, None, None, None, None]


def raw_mask_logit(model, mask_a, mask_b, points_a, points_b, valid_a, valid_b):
    """Exact score dependency path; differentiable without training any weight."""
    base = model.base_model
    encoded_a = base._encode_patches(differentiable_patches(base.patch_sampler, mask_a, points_a, valid_a), valid_a)
    encoded_b = base._encode_patches(differentiable_patches(base.patch_sampler, mask_b, points_b, valid_b), valid_b)
    a, b = base.context(encoded_a, encoded_b, valid_a, valid_b, points_a, points_b, base.config.canvas_size)
    return model.score_head(a, b, valid_a, valid_b)


@contextmanager
def gradient_determinism(device, allow_nondeterministic_gradient):
    """Never silently relax an evaluation/training process's global setting."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if device.type == "cuda" and not allow_nondeterministic_gradient:
        raise ValueError("CUDA mask gradients require explicit --allow-nondeterministic-gradient "
                         "because grid_sample backward may be nondeterministic; use CPU otherwise")
    changed = device.type == "cuda" and enabled
    if changed:
        torch.use_deterministic_algorithms(False)
    try:
        yield dict(original_deterministic_algorithms=enabled, original_warn_only=warn_only,
            relaxed_for_backward=changed, cuda_gradient_may_be_nondeterministic=device.type == "cuda")
    finally:
        if changed:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def finite_difference_check(model, masks, fixed, gradients, epsilon=1e-3, pixel_count=128,
                            reference_logit=None):
    """Two central-difference steps along one well-conditioned sensitivity direction.

    Largest-gradient pixel directions improve float32 signal-to-roundoff; they
    validate a derivative, NOT independently chosen scientific evidence. The
    tested direction is saved in the receipt as exact side/pixel coordinates.
    No re-estimation of contour coordinates takes place in this check.
    """
    if not np.isfinite(epsilon) or epsilon <= 0 or type(pixel_count) is not int or pixel_count < 1:
        raise ValueError("finite-difference epsilon and pixel_count must be positive")
    directions, selected = [], {}
    for side, gradient in zip("ab", gradients):
        flat = gradient.detach().reshape(-1)
        count = min(pixel_count, int((flat != 0).sum()))
        # Stable ties are resolved by original flattened row/column order.
        ids = torch.argsort(flat.abs(), descending=True, stable=True)[:count]
        direction = torch.zeros_like(flat)
        direction[ids] = flat[ids].sign()
        directions.append(direction.reshape_as(gradient))
        width = gradient.shape[-1]
        selected[side] = dict(pixel_rc=torch.stack((ids // width, ids % width), dim=1),
            direction=direction[ids])
    analytic = float(sum((g * d).sum() for g, d in zip(gradients, directions)))
    if reference_logit is None:
        with torch.no_grad(), torch.autocast(device_type=masks[0].device.type, enabled=False):
            reference_logit = raw_mask_logit(model, *masks, *fixed)[0]
    reference_logit = reference_logit.detach()
    evaluations = []
    for step in (epsilon, epsilon / 2):
        with torch.no_grad(), torch.autocast(device_type=masks[0].device.type, enabled=False):
            high = raw_mask_logit(model, *(m + step * d for m, d in zip(masks, directions)), *fixed)[0]
            low = raw_mask_logit(model, *(m - step * d for m, d in zip(masks, directions)), *fixed)[0]
        derivative = float((high - low) / (2 * step))
        error = abs(derivative - analytic)
        evaluations.append(dict(epsilon=step, central_derivative=derivative,
            forward_directional_slope=float((high - reference_logit) / step),
            backward_directional_slope=float((reference_logit - low) / step),
            absolute_error=error, relative_error=error / abs(analytic) if analytic else None))
    best = min(evaluations, key=lambda row: row["absolute_error"])
    tolerance = 1e-3 + .10 * abs(analytic)
    low_signal = abs(analytic) < 1e-3
    return _clean(dict(direction="up to %d largest-magnitude pixels per mask, aligned with derivative sign" % pixel_count,
        selected=selected, analytic_directional_derivative=analytic, checks=evaluations,
        absolute_tolerance=1e-3, relative_tolerance=.10,
        status="low_signal" if low_signal else "passed" if best["absolute_error"] <= tolerance else "needs_review",
        note="numerical direction check only, not an independent causal deletion test; mismatch may reflect nondifferentiable tied activations, finite-step effects or floating-point error and is retained for review"))


def summarize_gradient(mask, gradient):
    flat = gradient.reshape(-1)
    count = min(20, len(flat))
    ids = np.argsort(-np.abs(flat), kind="stable")[:count]
    nonzero = np.argwhere(gradient != 0)
    width = gradient.shape[-1]
    return dict(shape=list(mask.shape), pixel_count=int(mask.size), nonzero_gradient_pixels=int(len(nonzero)),
        signed_min=float(gradient.min()), signed_max=float(gradient.max()),
        absolute_max=float(np.abs(gradient).max()), absolute_sum=float(np.abs(gradient).sum()),
        foreground_absolute_sum=float(np.abs(gradient[mask > .5]).sum()),
        background_absolute_sum=float(np.abs(gradient[mask <= .5]).sum()),
        support_bbox_xyxy=[int(nonzero[:, 1].min()), int(nonzero[:, 0].min()),
            int(nonzero[:, 1].max()) + 1, int(nonzero[:, 0].max()) + 1] if len(nonzero) else None,
        top_magnitude_pixels=[dict(row=int(i // width), column=int(i % width), mask_value=float(mask.flat[i]),
            signed_gradient=float(flat[i]), magnitude=float(abs(flat[i]))) for i in ids])


@torch.inference_mode(False)
@torch.enable_grad()
def probe_pixels(model, tensors, *, allow_nondeterministic_gradient=False, finite_difference_epsilon=1e-3):
    if (type(model) is not DecoupledScoreModel or model.head_kind != "cross_attention"
            or type(model.base_model) is not RachelN512Pairwise
            or type(model.score_head) is not CrossAttentionPairHead
            or model.phase != "classifier" or model.training):
        raise ValueError("requires frozen classifier-phase decoupled cross-attention model in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("freeze every model parameter before probing; this function never changes weights/flags")
    if len(tensors) != 6 or any(len(tensor) != 1 for tensor in tensors):
        raise ValueError("requires exactly one pair of six original model tensors")
    device = tensors[0].device
    if device.type not in ("cpu", "cuda") or any(tensor.device != device for tensor in tensors):
        raise ValueError("all inputs must share one supported CPU/CUDA device")
    if any(tensor.dtype != torch.float32 for tensor in tensors[:4]) or any(tensor.dtype != torch.bool for tensor in tensors[4:]):
        raise ValueError("frozen FP32 masks/coordinates and bool validity are required")
    if any(not ((mask == 0) | (mask == 1)).all() for mask in tensors[:2]):
        raise ValueError("observed masks must remain exactly finite binary inputs")
    if device.type == "cuda" and not allow_nondeterministic_gradient:
        raise ValueError("use CPU or explicitly pass --allow-nondeterministic-gradient for grid_sample backward")
    captured = []
    hook = model.base_model.register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            original = model(*tensors)
            base = captured[0]
            raw_reference = model.score_head(base.token_features_a, base.token_features_b, tensors[4], tensors[5])[0]
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError("the original full forward must execute the base once")
    masks = [tensor.detach().clone().requires_grad_(True) for tensor in tensors[:2]]
    fixed = [tensor.detach().clone() for tensor in tensors[2:]]
    patch_errors = {}
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for side, mask, points, valid in zip("ab", masks, fixed[:2], fixed[2:]):
            expected = model.base_model.patch_sampler(mask, points, valid)
            replayed = differentiable_patches(model.base_model.patch_sampler, mask, points, valid)
            if not torch.equal(expected, replayed):
                raise ValueError("%s original and differentiable sampled patches differ" % side)
            patch_errors[side] = float((expected - replayed).abs().max())
    with torch.autocast(device_type=device.type, enabled=False):
        raw = raw_mask_logit(model, *masks, *fixed)[0]
    _assert_close(raw.detach(), raw_reference, "mask-ancestry raw head logit")
    effective = torch.where(base.training_valid[0], raw.detach(), torch.zeros_like(raw.detach()))
    _assert_close(effective, original.fused_logit[0], "mask-ancestry wrapper logit")
    with gradient_determinism(device, allow_nondeterministic_gradient) as deterministic_receipt:
        gradients = torch.autograd.grad(raw, masks, allow_unused=False)
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError("nonfinite mask gradient; refusing misleading sensitivity output")
    numerical = finite_difference_check(model, masks, fixed, gradients,
        epsilon=finite_difference_epsilon, reference_logit=raw.detach())
    # Private intermediate is consumed by the CLI decoder, not JSON-serialized.
    arrays, summaries = {"_original_assignment_for_layout": _numpy(base.assignment[0])}, {}
    for side, mask, gradient in zip("ab", masks, gradients):
        m, g = _numpy(mask[0, 0]), _numpy(gradient[0, 0])
        arrays.update({"mask_" + side: m.astype(np.uint8), "signed_gradient_" + side: g,
            "absolute_gradient_" + side: np.abs(g), "input_times_gradient_" + side: m * g,
            "points_rc_" + side: _numpy(fixed[0 if side == "a" else 1][0]),
            "valid_" + side: _numpy(fixed[2 if side == "a" else 3][0])})
        summaries[side] = summarize_gradient(m, g)
    result = dict(schema_version=SCHEMA, score=float(original.fused_probability[0]),
        raw_head_logit=float(raw_reference), raw_head_probability=float(raw_reference.sigmoid()),
        decision_valid=bool(original.decision_valid[0]), training_valid=bool(original.training_valid[0]),
        original_raw_logit_error=float((raw.detach() - raw_reference).abs()),
        wrapper_logit_error=float((effective - original.fused_logit[0]).abs()),
        original_patch_max_abs_error=patch_errors,
        forward_equivalence_tolerance=dict(atol=FORWARD_ATOL, rtol=FORWARD_RTOL),
        pixel_sensitivity=summaries, finite_difference=numerical,
        determinism=deterministic_receipt, interpretation=INTERPRETATION)
    return _clean(result), arrays


def run(args):
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluation
    source = Path(args.selection_json)
    chosen = select_pixel_cases(json.loads(source.read_text()), args.all_selected)
    selection_by_id = {(row["dataset"], row["pair_id"]): row for row in chosen}
    torch.set_num_threads(1)
    model, identity = evaluation.load_frozen_model(args.training_run, args.selection)
    evaluation.core.sealed._set_determinism(identity["seed"])
    device = torch.device(args.device)
    if device.type == "cuda" and not args.allow_nondeterministic_gradient:
        raise ValueError("CUDA mask derivatives require explicit --allow-nondeterministic-gradient; no run was started")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device).eval().requires_grad_(False)
    protocol = dict(schema_version=SCHEMA, status="running", model=identity,
        selected_cases=chosen, sample_count=len(chosen),
        selection_policy="explicit complete supplied list" if args.all_selected else "first preselected case in each fixed default stratum; no gradient-based case selection",
        selection_json_sha256=evaluation.core.sealed._sha256_file(source),
        script_sha256=evaluation.core.sealed._sha256_file(Path(__file__)),
        runtime=dict(torch_version=str(torch.__version__), device=str(device), precision="fp32", batch_size=1),
        parameters_fitted=False, checkpoint_changed=False, thresholds_fitted=False,
        interpretation=INTERPRETATION)
    _json(output / "protocol.json", protocol)
    rows = []
    try:
        for dataset, batch in selected_batches(args, chosen, identity):
            tensors = [evaluation.core.sealed._tensor(getattr(batch, key), device,
                torch.bool if key.startswith("contour_valid") else torch.float32) for key in FIELDS]
            pair_started = time.perf_counter()
            row, arrays = probe_pixels(model, tensors,
                allow_nondeterministic_gradient=args.allow_nondeterministic_gradient,
                finite_difference_epsilon=args.finite_difference_epsilon)
            assignment = arrays.pop("_original_assignment_for_layout")
            layout = estimate_translation_layout(arrays["points_rc_a"], arrays["points_rc_b"], assignment,
                arrays["valid_a"], arrays["valid_b"], config=evaluation.core.fixed.TOP2_CONFIG)
            row["layout"] = _clean(asdict(layout))
            selected = selection_by_id[(dataset, batch.pair_ids[0])]
            row.update(dataset=dataset, pair_id=batch.pair_ids[0], fragment_a=batch.fragment_a_tokens[0],
                fragment_b=batch.fragment_b_tokens[0], name=selected.get("name"),
                stratum=selected.get("stratum"),
                stratum_reference="S6-depth2 original SIMVAL operating point, not necessarily this model's classification",
                probe_elapsed_seconds=time.perf_counter() - pair_started)
            key = hashlib.sha256((dataset + "\0" + row["pair_id"]).encode()).hexdigest()[:20]
            row["arrays_path"] = key + ".npz"
            np.savez_compressed(output / row["arrays_path"], **arrays)
            row["arrays_sha256"] = evaluation.core.sealed._sha256_file(output / row["arrays_path"])
            if dataset == "test":
                row.update(_test_label=bool(batch.labels[0]), _test_target=_clean(batch.translation_a_to_b_rc[0])
                    if batch.translation_valid[0] else None)
            rows.append(row)
            _json(output / "partial_cases.json", [{k: v for k, v in item.items() if not k.startswith("_test_")} for item in rows])
            print(json.dumps(dict(dataset=dataset, pair_id=row["pair_id"], completed=len(rows), total=len(chosen),
                numerical_check=row["finite_difference"]["status"])), flush=True)
        if {(row["dataset"], row["pair_id"]) for row in rows} != {(row["dataset"], row["pair_id"]) for row in chosen}:
            raise ValueError("pixel probe did not cover the selected IDs")
        attach_targets(rows, args)
        _json(output / "cases.json", rows)
        needs_review = [row["pair_id"] for row in rows if row["finite_difference"]["status"] != "passed"]
        protocol.update(status="needs_review" if needs_review else "complete", completed_count=len(rows),
            numerical_review_pair_ids=needs_review, cases_sha256=evaluation.core.sealed._sha256_file(output / "cases.json"))
    except Exception as error:
        protocol.update(status="failed", error=repr(error), completed_count=len(rows))
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        protocol["elapsed_seconds"] = time.perf_counter() - started
        protocol["gpu_memory"] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            scope="this process/device since loading model onto GPU; excludes other processes") if device.type == "cuda" else None
        _json(output / "protocol.json", protocol)
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run", required=True)
    p.add_argument("--selection", choices=("fixed_epoch", "max_f1", "recall95"), default="fixed_epoch")
    p.add_argument("--selection-json", required=True)
    p.add_argument("--all-selected", action="store_true", help="Use an explicitly supplied finite subset instead of the default eight strata")
    p.add_argument("--output", required=True)
    p.add_argument("--prepared-cache", required=True)
    p.add_argument("--ood-prepared", required=True)
    p.add_argument("--dataset", default="/root/autodl-tmp/dataset_rachel_pairwise_n512_v1")
    p.add_argument("--translation-gt-json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-nondeterministic-gradient", action="store_true")
    p.add_argument("--finite-difference-epsilon", type=float, default=1e-3)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
