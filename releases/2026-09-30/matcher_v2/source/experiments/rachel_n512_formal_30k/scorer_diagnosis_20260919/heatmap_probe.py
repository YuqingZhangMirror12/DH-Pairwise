"""Post-hoc, frozen CrossAttentionPairHead probes; no fitting or model edits.

Run with ``python -m experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.heatmap_probe``.
The selection JSON is a list of {"dataset": "real"|"ood"|"test", "pair_id": ...}.
Only selected inputs are loaded/inferred. All arrays use the prepared 800px
mask coordinate system; token indices always refer to the original padded
contour, while stored attention matrices index the explicitly saved valid IDs.

Gradient x activation is a local sensitivity, NOT a causal explanation or an
additive decomposition of the logit. Attention/pool weights are associations,
not causal contributions. Zero/removal perturbations are off-manifold head
interventions with a fixed matcher; they are not physical fragment edits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from staging.pairwise_v0_2.models.rachel_decoupled_score import (
    SCHEMA as MODEL_SCHEMA, CrossAttentionPairHead, DecoupledScoreModel,
)
from staging.pairwise_v0_2.models.translation_layout import estimate_translation_layout

SCHEMA = "frozen-cross-attention-heatmap-probe/1"
FORWARD_ATOL = 2e-5
FORWARD_RTOL = 2e-5
FIELDS = ("mask_a", "mask_b", "points_rc_a", "points_rc_b", "contour_valid_a", "contour_valid_b")
NOTES = dict(
    attribution="pre-sigmoid raw head logit sensitivity; channel sum of activation*gradient, not causal or additive",
    absolute_attribution="sum of absolute channel-wise activation*gradient, NOT absolute value of the signed sum",
    attention="mean across attention heads; non-causal diagnostic, independently recomputed without altering the score path",
    pool="learned attentive-pool weights; max-pool channel winners are a separate path; neither is causal",
    perturbation="fixed-matcher token zero/removal intervention only; off manifold, not an edit of the physical mask",
    alignment="best decoder inliers are predicted correspondences, NOT a GT seam or proof of adjacency",
    target="raw head logit before wrapper validity gating; no sigmoid-saturation gradient",
    signed_sign="positive means locally increasing that token's activation scale raises the raw logit; negative means it lowers it; this is not a finite deletion effect",
    token_coordinates="original padded contour indices, coordinates are row/column in prepared mask pixels; not a pixel-level attribution",
)


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _clean(value):
    if torch.is_tensor(value):
        value = _numpy(value)
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(path, value):
    Path(path).write_text(json.dumps(_clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _assert_close(actual, reference, what, atol=FORWARD_ATOL, rtol=FORWARD_RTOL):
    if not torch.allclose(actual, reference, atol=atol, rtol=rtol):
        error = float((actual - reference).abs().max())
        raise ValueError("probe differs from original %s (maximum error %.9g)" % (what, error))


def _interaction(layer, query, source):
    """Keep the exact need_weights=False score graph; collect weights separately."""
    q, s = query[None], source[None]
    nq, ns = layer.norm(q), layer.norm(s)
    update, _ = layer.cross_attention(nq, ns, ns, need_weights=False)
    residual = q + update
    decoded = residual + layer.ffn(layer.output_norm(residual))
    with torch.no_grad():
        _, weights = layer.cross_attention(nq.detach(), ns.detach(), ns.detach(),
            need_weights=True, average_attn_weights=True)
    return residual[0], decoded[0], weights[0]


@torch.inference_mode(False)
@torch.enable_grad()
def trace_head(head, features_a, features_b, valid_a, valid_b):
    """Trace one pair, detached from the frozen base, without changing parameters.

    Returns JSON-compatible token summaries plus NPZ-ready exact feature,
    gradient and attention arrays. Input features are [Na,D]/[Nb,D]. Padding
    may be non-contiguous. Raises for unsupported heads or missing evidence.
    """
    if type(head) is not CrossAttentionPairHead or head.depth not in (1, 2, 4):
        raise ValueError("only the registered 1/2/4-layer CrossAttentionPairHead is supported")
    if head.training:
        raise ValueError("probe requires eval mode")
    if (features_a.ndim != 2 or features_b.ndim != 2 or
            features_a.shape[1] != features_b.shape[1] or
            valid_a.dtype != torch.bool or valid_b.dtype != torch.bool or
            valid_a.shape != features_a.shape[:1] or valid_b.shape != features_b.shape[:1]):
        raise ValueError("feature/validity schema differs from a single contour pair")
    ia, ib = torch.where(valid_a)[0], torch.where(valid_b)[0]
    if not len(ia) or not len(ib):
        raise ValueError("empty contour has a no-evidence bias, not attributable token evidence")
    a = features_a.index_select(0, ia).detach().clone().requires_grad_(True)
    b = features_b.index_select(0, ib).detach().clone().requires_grad_(True)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("nonfinite context features")
    records = [("context_input", a, b)]
    attention = {}
    # Both directions read the SAME previous-layer states, not an updated A.
    for index, layer in enumerate([head] + list(head.extra_layers), 1):
        ra, aa, wab = _interaction(layer, a, b)
        rb, bb, wba = _interaction(layer, b, a)
        # Record 2D view tensors that actually feed subsequent computation.
        # residual[0] itself is a sibling view of decoded; see _interaction:
        # record the decoded state, which is the exact next-layer parent.
        records.append(("decoder_%d_output" % index, aa, bb))
        attention["decoder_%d_a_to_b_attention_mean" % index] = _numpy(wab)
        attention["decoder_%d_b_to_a_attention_mean" % index] = _numpy(wba)
        attention["decoder_%d_a_attention_residual" % index] = _numpy(ra)
        attention["decoder_%d_b_attention_residual" % index] = _numpy(rb)
        a, b = aa, bb
    pa, pb = head._pool(a), head._pool(b)
    logit = head.classifier(torch.cat((.5 * (pa + pb), torch.abs(pa - pb)))).reshape(())
    with torch.no_grad():
        original = head(features_a[None], features_b[None], valid_a[None], valid_b[None])[0]
    _assert_close(logit.detach(), original, "head logit")
    states = [state for _, a_state, b_state in records for state in (a_state, b_state)]
    gradients = torch.autograd.grad(logit, states, allow_unused=False)
    if any(not torch.isfinite(value).all() for value in gradients):
        raise ValueError("nonfinite attribution gradient; refusing a misleading heatmap")
    arrays = dict(attention)
    arrays.update(valid_indices_a=_numpy(ia), valid_indices_b=_numpy(ib))
    layers = []
    for index, (name, aa, bb) in enumerate(records):
        entry = {"name": name}
        for offset, (side, value, original_indices) in enumerate((("a", aa, ia), ("b", bb, ib))):
            gradient = gradients[2 * index + offset]
            product = value.detach() * gradient.detach()
            entry[side] = dict(token_indices=_numpy(original_indices),
                signed=product.sum(-1), absolute=product.abs().sum(-1),
                activation_l2=torch.linalg.vector_norm(value.detach(), dim=-1),
                gradient_l2=torch.linalg.vector_norm(gradient.detach(), dim=-1))
            arrays[name + "_" + side + "_features"] = _numpy(value)
            arrays[name + "_" + side + "_gradients"] = _numpy(gradient)
        layers.append(_clean(entry))
    pool = {}
    with torch.no_grad():
        for side, value, ids in (("a", a, ia), ("b", b, ib)):
            gate_logits = head.pool_gate(value).squeeze(-1)
            maximum = value.amax(0)
            ties = value == maximum[None]
            # Fractionally credit tied maxima; each channel contributes 1 total.
            max_share = (ties / ties.sum(0, keepdim=True)).sum(1) / value.shape[1]
            pool[side] = dict(token_indices=ids, gate_logits=gate_logits,
                weights=torch.softmax(gate_logits, dim=0), max_pool_channel_share=max_share)
    return dict(schema_version=SCHEMA, raw_head_logit=float(logit.detach()),
        raw_head_probability=float(logit.detach().sigmoid()), depth=head.depth,
        original_head_max_abs_error=float((logit.detach() - original).abs()),
        layers=layers, pool=_clean(pool), interpretation=NOTES), arrays


def perturb_tokens(head, features_a, features_b, valid_a, valid_b, selected_a, selected_b, mode):
    """Evaluate a frozen head under explicit original-index token interventions."""
    if mode not in ("zero", "mask"):
        raise ValueError("intervention must be zero or mask")
    a, b, va, vb = (value.detach().clone() for value in (features_a, features_b, valid_a, valid_b))
    for value, valid, selected in ((a, va, selected_a), (b, vb, selected_b)):
        ids = torch.as_tensor(selected, dtype=torch.long, device=value.device)
        if ids.ndim != 1 or len(ids) != len(torch.unique(ids)) or (
                len(ids) and ((ids < 0).any() or (ids >= len(valid)).any() or not valid[ids].all())):
            raise ValueError("perturbation indices must be unique valid original contour indices")
        if mode == "zero":
            value[ids] = 0
        else:
            valid[ids] = False
    with torch.no_grad():
        logit = head(a[None], b[None], va[None], vb[None])[0]
    return dict(raw_head_logit=float(logit), raw_head_probability=float(logit.sigmoid()),
        a_indices=list(map(int, selected_a)), b_indices=list(map(int, selected_b)),
        reaches_no_evidence_bias=not bool(va.any() and vb.any()), mode=mode)


def _alignment(layers, inlier_a, inlier_b):
    result = []
    for layer in layers:
        row = {"layer": layer["name"]}
        for side, inliers in (("a", inlier_a), ("b", inlier_b)):
            values = layer[side]
            membership = np.isin(values["token_indices"], inliers)
            absolute = np.asarray(values["absolute"])
            signed = np.asarray(values["signed"])
            total = float(absolute.sum())
            share = float(absolute[membership].sum() / total) if total > 0 else None
            fraction = float(membership.mean())
            row[side] = dict(predicted_inlier_token_fraction=fraction,
                absolute_attribution_share_on_inliers=share,
                enrichment_over_uniform=(share / fraction if share is not None and fraction else None),
                signed_sum_on_inliers=float(signed[membership].sum()), signed_sum_total=float(signed.sum()))
        result.append(row)
    return result


def probe_pair(model, tensors, *, decoder_config, ablation_fraction=.1, seed=0):
    """Frozen one-pair original inference + differentiable head-only replay."""
    if (type(model) is not DecoupledScoreModel or model.head_kind != "cross_attention" or
            model.phase != "classifier" or model.metadata().get("schema_version") != MODEL_SCHEMA):
        raise ValueError("requires a registered classifier-phase decoupled cross-attention checkpoint")
    if model.training or len(tensors) != 6 or any(len(value) != 1 for value in tensors):
        raise ValueError("probe requires eval mode and one pair of six model tensors")
    if not 0 <= ablation_fraction < 1:
        raise ValueError("ablation fraction must be in [0,1)")
    captured = []
    handle = model.base_model.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        # Exactly the official frozen evaluator's inference/autocast policy.
        # Its captured context features are inference tensors: trace_head must
        # explicitly leave inference mode before cloning differentiable leaves.
        with torch.inference_mode(), torch.autocast(device_type=tensors[0].device.type, enabled=False):
            original = model(*tensors)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("original forward did not execute exactly one frozen base")
    base = captured[0]
    va, vb = tensors[-2][0], tensors[-1][0]
    fa, fb = base.token_features_a[0], base.token_features_b[0]
    with torch.enable_grad(), torch.autocast(device_type=tensors[0].device.type, enabled=False):
        result, arrays = trace_head(model.score_head, fa, fb, va, vb)
    raw_logit = torch.tensor(result["raw_head_logit"], device=original.fused_logit.device,
                             dtype=original.fused_logit.dtype)
    effective_logit = torch.where(base.training_valid[0] & torch.isfinite(raw_logit), raw_logit, raw_logit * 0)
    _assert_close(effective_logit, original.fused_logit[0], "wrapper logit")
    _assert_close(effective_logit.sigmoid(), original.fused_probability[0], "wrapper probability")
    points_a, points_b = _numpy(tensors[2][0]), _numpy(tensors[3][0])
    assignment = _numpy(base.assignment[0])
    estimate = estimate_translation_layout(points_a, points_b, assignment, _numpy(va), _numpy(vb),
                                          config=decoder_config)
    candidate_pairs = estimate.candidate_indices
    inlier_pairs = candidate_pairs[estimate.inlier_mask]
    inlier_a = np.unique(inlier_pairs[:, 0]) if len(inlier_pairs) else np.zeros(0, dtype=np.int64)
    inlier_b = np.unique(inlier_pairs[:, 1]) if len(inlier_pairs) else np.zeros(0, dtype=np.int64)
    layout = asdict(estimate)
    layout.update(inlier_token_indices_a=inlier_a, inlier_token_indices_b=inlier_b,
        candidate_sinkhorn_mass=assignment[candidate_pairs[:, 0], candidate_pairs[:, 1]],
        offset_b_in_a_rc=-estimate.t_a_to_b_rc,
        decoder_config=asdict(decoder_config),
        minimum_arc_length_gate_present=False, pairability_test_present=False,
        warning="valid only meets the decoder's inlier-count/ambiguity gate; NOT proof of adjacency or a reliable seam")
    arrays.update(mask_a=_numpy(tensors[0][0, 0]).astype(np.uint8),
        mask_b=_numpy(tensors[1][0, 0]).astype(np.uint8),
        points_rc_a=points_a, points_rc_b=points_b,
        valid_a=_numpy(va), valid_b=_numpy(vb),
        sinkhorn_assignment=assignment, matcher_affinity=_numpy(base.affinity[0]),
        matcher_unmatched_a=_numpy(base.unmatched_a[0]), matcher_unmatched_b=_numpy(base.unmatched_b[0]))
    result.update(score=float(original.fused_probability[0]), logit=float(original.fused_logit[0]),
        training_valid=bool(original.training_valid[0]), decision_valid=bool(original.decision_valid[0]),
        wrapper_max_abs_error=float((effective_logit - original.fused_logit[0]).abs()),
        attribution_is_valid_decision=bool(original.decision_valid[0]),
        forward_equivalence_tolerance=dict(atol=FORWARD_ATOL, rtol=FORWARD_RTOL),
        points_rc_a=points_a, points_rc_b=points_b, valid_a=_numpy(va), valid_b=_numpy(vb),
        layout=_clean(layout), layer_layout_alignment=_alignment(result["layers"], inlier_a, inlier_b))
    if ablation_fraction:
        rng = np.random.default_rng(seed)
        subsets = {}
        for side, valid, seam in (("a", va, inlier_a), ("b", vb, inlier_b)):
            attribution = result["layers"][0][side]
            ids = np.asarray(attribution["token_indices"], dtype=np.int64)
            count = min(len(ids) - 1, max(1, int(math.ceil(len(ids) * ablation_fraction))))
            top = ids[np.argsort(-np.asarray(attribution["signed"]), kind="stable")[:count]]
            absolute_top = ids[np.argsort(-np.asarray(attribution["absolute"]), kind="stable")[:count]]
            subsets[side] = dict(top_signed=top, top_absolute=absolute_top,
                random_for_top=rng.choice(ids, count, replace=False), layout_inliers=seam,
                random_for_layout=rng.choice(ids, len(seam), replace=False))
        result["perturbations"] = []
        for name in subsets["a"]:
            for mode in ("zero", "mask"):
                row = perturb_tokens(model.score_head, fa, fb, va, vb,
                    subsets["a"][name], subsets["b"][name], mode)
                row.update(selection=name, baseline_raw_head_logit=result["raw_head_logit"],
                    delta_logit=row["raw_head_logit"] - result["raw_head_logit"],
                    delta_probability=row["raw_head_probability"] - result["raw_head_probability"],
                    interpretation=NOTES["perturbation"])
                result["perturbations"].append(row)
    return _clean(result), arrays


def selected_batches(args, selections, identity):
    """Existing evaluator's input contract, narrowed BEFORE mask unpack/inference."""
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluation
    core = evaluation.core
    mode = identity["sampling"]
    mode = mode if isinstance(mode, str) else mode["mode"]
    if mode == "step3":
        from staging.pairwise_v0_2.pairwise_data.rachel_step_density import StepInputContourResampler
        transform = StepInputContourResampler(step_px=3., cap=2048)
    elif mode == "paired512":
        transform = core.InputContourResampler(512).resample_batch
    elif mode == "original512":
        transform = lambda batch: batch
    else:
        raise ValueError("unknown frozen sampling schema: " + str(mode))
    for dataset in ("real", "ood", "test"):
        requested = [row for row in selections if row["dataset"] == dataset]
        if not requested:
            continue
        if dataset == "test":
            root = Path(args.dataset)
            manifest = [json.loads(line) for line in (root / "pairs/test.jsonl").read_text().splitlines() if line]
            lookup = {row["pair_id"]: index for index, row in enumerate(manifest)}
            if len(lookup) != len(manifest):
                raise ValueError("duplicate TEST pair IDs")
            missing = {row["pair_id"] for row in requested} - set(lookup)
            if missing:
                raise ValueError("selected TEST IDs absent: " + str(sorted(missing)))
            ds = core.RachelPairDataset(root, "test")
            # Keep deterministic input order. No train/val/test selector is fitted.
            batches = core.make_ablation_loader(ds, [lookup[row["pair_id"]] for row in requested],
                batch_size=1, num_workers=0, seed=identity["seed"], contour_cap=512)
        else:
            cache = Path(args.prepared_cache if dataset == "real" else args.ood_prepared)
            metadata = json.loads((cache / "manifest.json").read_text())
            if dataset == "real" and metadata.get("schema") != "real-layout-prepared-v1":
                raise ValueError("unknown REAL prepared-cache schema")
            if dataset == "ood" and (metadata.get("layout_gt_provided") is not False or
                    metadata.get("negative_pairs_constructed") is not False):
                raise ValueError("OOD must be the positive-only no-layout-GT cache")
            lookup = {row["pair_id"]: row for row in metadata["pairs"]}
            if len(lookup) != len(metadata["pairs"]):
                raise ValueError("duplicate prepared pair IDs")
            missing = {row["pair_id"] for row in requested} - set(lookup)
            if missing:
                raise ValueError("selected %s IDs absent: %s" % (dataset, sorted(missing)))
            narrowed = dict(metadata, pairs=[lookup[row["pair_id"]] for row in requested])
            with np.load(cache / "inputs.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in ("packed_masks", "points", "valid")}
            batches = core.real.input_batches(narrowed, arrays, 1)
        for batch in batches:
            yield dataset, transform(batch)


def attach_targets(rows, args):
    """Open pose ground truth only AFTER all selected frozen predictions exist."""
    targets = {}
    if any(row["dataset"] == "real" for row in rows) and args.translation_gt_json:
        source = json.loads(Path(args.translation_gt_json).read_text())
        if not isinstance(source.get("positive_pairs"), list):
            raise ValueError("unknown real pose target schema")
        targets = {row["pair_id"]: row for row in source["positive_pairs"]}
        if len(targets) != len(source["positive_pairs"]):
            raise ValueError("duplicate real pose target IDs")
    real_pairs = {}
    if any(row["dataset"] == "real" for row in rows):
        manifest = json.loads((Path(args.prepared_cache) / "manifest.json").read_text())
        real_pairs = {row["pair_id"]: row for row in manifest["pairs"]}
    for row in rows:
        dataset, pair_id = row["dataset"], row["pair_id"]
        target = None
        if dataset == "real":
            pair = real_pairs[pair_id]
            row["label"] = bool(pair["label"])
            row["case_cluster"] = pair.get("case_cluster")
            if args.translation_gt_json and row["label"] and pair_id not in targets:
                raise ValueError("selected REAL positive lacks pose GT in provided target file")
            if not row["label"] and pair_id in targets:
                raise ValueError("negative REAL pair unexpectedly has positive pose GT")
            if pair_id in targets:
                gt = targets[pair_id]
                if (gt["fragment_a_token"] != row["fragment_a"] or gt["fragment_b_token"] != row["fragment_b"]):
                    raise ValueError("ground-truth endpoints differ from selected input")
                target = gt["translation_gt_a_to_b_rc"]
        elif dataset == "test":
            target = row.pop("_test_target")
            row["label"] = row.pop("_test_label")
        else:
            row["label"] = True
        if target is not None:
            target = np.asarray(target, dtype=float)
            if target.shape != (2,) or not np.isfinite(target).all():
                raise ValueError("pose GT must be finite [2]")
        row["target_translation_rc"] = _clean(target)
        row["layout_gt_available"] = target is not None
        row["layout"]["translation_l2_px"] = (float(np.linalg.norm(np.asarray(row["layout"]["t_a_to_b_rc"]) - target))
            if target is not None and row["layout"]["valid"] else None)


def run(args):
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluation
    selections = json.loads(Path(args.selection_json).read_text())
    if (not isinstance(selections, list) or not selections or any(not isinstance(row, dict) or
            row.get("dataset") not in ("real", "ood", "test") or not isinstance(row.get("pair_id"), str)
            for row in selections)):
        raise ValueError("selection must be a nonempty [{dataset,pair_id,...}] JSON list")
    keys = [(row["dataset"], row["pair_id"]) for row in selections]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate selected dataset/pair IDs")
    torch.set_num_threads(1)
    model, identity = evaluation.load_frozen_model(args.training_run, args.selection)
    if model.head_kind != "cross_attention":
        raise ValueError("requested checkpoint is not a cross-attention classifier")
    evaluation.core.sealed._set_determinism(identity["seed"])
    device = torch.device(args.device)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device).eval().requires_grad_(False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema_version=SCHEMA, status="running", model=identity,
        selection_json_sha256=evaluation.core.sealed._sha256_file(Path(args.selection_json)),
        selected_pairs=selections, sample_count=len(selections),
        decoder=asdict(evaluation.core.fixed.TOP2_CONFIG), script_sha256=evaluation.core.sealed._sha256_file(Path(__file__)),
        runtime=dict(torch_version=str(torch.__version__), device=str(device),
            cuda_version=torch.version.cuda, precision="fp32", batch_size=1,
            device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            original_forward_mode="torch.inference_mode; autocast disabled",
            attribution_forward_mode="inference_mode(False); enable_grad; autocast disabled"),
        fitted_parameters=False, thresholds_fitted=False, posthoc_selected_case_diagnostic=True,
        not_an_independent_test=True, interpretation=NOTES)
    _json(output / "protocol.json", protocol)
    rows = []
    try:
        for dataset, batch in selected_batches(args, selections, identity):
            tensors = [evaluation.core.sealed._tensor(getattr(batch, key), device,
                torch.bool if key.startswith("contour_valid") else torch.float32) for key in FIELDS]
            pair_id = batch.pair_ids[0]
            key = hashlib.sha256((dataset + "\0" + pair_id).encode()).hexdigest()[:20]
            pair_started = time.perf_counter()
            row, arrays = probe_pair(model, tensors, decoder_config=evaluation.core.fixed.TOP2_CONFIG,
                ablation_fraction=args.ablation_fraction, seed=int(key[:8], 16))
            row.update(dataset=dataset, pair_id=pair_id, fragment_a=batch.fragment_a_tokens[0],
                fragment_b=batch.fragment_b_tokens[0], arrays_path=key + ".npz",
                probe_elapsed_seconds=time.perf_counter() - pair_started)
            if dataset == "test":
                row.update(_test_label=bool(batch.labels[0]),
                    _test_target=_clean(batch.translation_a_to_b_rc[0]) if batch.translation_valid[0] else None)
            np.savez_compressed(output / row["arrays_path"], **arrays)
            row["arrays_sha256"] = evaluation.core.sealed._sha256_file(output / row["arrays_path"])
            rows.append(row)
            # Checkpoint partial probes without claiming completed population.
            _json(output / "predictions_without_attached_gt.json",
                  [{key: value for key, value in item.items() if not key.startswith("_test_")} for item in rows])
            print(json.dumps(dict(dataset=dataset, pair_id=pair_id, completed=len(rows), total=len(keys))), flush=True)
        if {(row["dataset"], row["pair_id"]) for row in rows} != set(keys):
            raise ValueError("probe population differs from requested IDs")
        attach_targets(rows, args)
        _json(output / "cases.json", rows)
        protocol.update(status="complete", completed_count=len(rows),
            cases_sha256=evaluation.core.sealed._sha256_file(output / "cases.json"))
    except Exception as error:
        protocol.update(status="failed", error=repr(error), completed_count=len(rows))
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            protocol["gpu_memory"] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                scope="this process/device since loading model onto GPU; excludes other processes")
        else:
            protocol["gpu_memory"] = None
        protocol["elapsed_seconds"] = time.perf_counter() - started
        _json(output / "protocol.json", protocol)
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run", required=True)
    p.add_argument("--selection", choices=("fixed_epoch", "max_f1", "recall95"), default="fixed_epoch")
    p.add_argument("--selection-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prepared-cache", required=True)
    p.add_argument("--ood-prepared", required=True)
    p.add_argument("--dataset", default="/root/autodl-tmp/dataset_rachel_pairwise_n512_v1")
    p.add_argument("--translation-gt-json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ablation-fraction", type=float, default=.1)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
