"""Two existing cases: CPU max-branch explanation, never attribution approval.

Keeps checkpoints, sampled points, mask inputs and the old numerical tolerance.
Original MaxPool2d winners are frozen by gather; Scorer amax tied winners retain
equal derivative weights. Temporary instance methods are always restored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919 import pixel_attribution as pixel

SCHEMA = "fixed-mask-max-branch-counterfactual/1"
SELECTED_IDS = (
    "pair/sha256/d1d9bf23b0040e4a85fc1d4fb409c17f212642cd14242241cf4e5d28b230595b",
    "pair/sha256/b6909377cdcb1b5a2ecbe7e371ae3d1b1a234cbdc0f6a8ffbaf8c677c72a16fb",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def pair2(value):
    return (value, value) if isinstance(value, int) else tuple(value)


class MaxBranches:
    """Scope only explicit, input-dependent max reductions in score ancestry."""
    def __init__(self, model):
        self.model = model
        self.pools = [(name, module) for name, module in model.base_model.patch_encoder.named_modules()
                      if isinstance(module, nn.MaxPool2d)]
        if len(self.pools) != 3:
            raise ValueError("requires exactly three registered patch MaxPool2d layers")
        for _, module in self.pools:
            if (type(module) is not nn.MaxPool2d or pair2(module.kernel_size) != (2, 2)
                    or pair2(module.stride) != (2, 2) or pair2(module.padding) != (0, 0)
                    or pair2(module.dilation) != (1, 1) or module.ceil_mode or module.return_indices):
                raise ValueError("unregistered pool configuration; coverage not established")
        if type(model.score_head) is not pixel.CrossAttentionPairHead:
            raise ValueError("unregistered Scorer pooling")
        self.baseline = []
        self.saved_methods = []

    def __enter__(self):
        for name, module in self.pools:
            self._replace(module, "forward", lambda module, value, name=name: self.pool2d(name, value))
        self._replace(self.model.score_head, "_pool", lambda head, value: self.scorer_pool(head, value))
        return self

    def _replace(self, module, key, function):
        self.saved_methods.append((module, key, key in module.__dict__, module.__dict__.get(key)))
        setattr(module, key, types.MethodType(function, module))

    def __exit__(self, *unused):
        for module, key, existed, value in reversed(self.saved_methods):
            if existed:
                setattr(module, key, value)
            else:
                delattr(module, key)

    def entry(self, kind, name, value):
        index = self.cursor
        self.cursor += 1
        reference = None if self.mode == "capture" else self.baseline[index]
        if reference is not None and (reference["kind"] != kind or reference["name"] != name
                                      or reference["input_shape"] != tuple(value.shape)):
            raise ValueError("max-call sequence/shape changed")
        return index, reference

    def pool2d(self, name, value):
        index, reference = self.entry("MaxPool2d", name, value)
        if self.mode == "frozen":
            indices = reference["indices"]
            return value.flatten(2).gather(2, indices.flatten(2)).reshape_as(indices)
        pooled, indices = F.max_pool2d(value, 2, 2, return_indices=True)
        windows = value.unfold(2, 2, 2).unfold(3, 2, 2)
        tied_count = (windows == pooled[..., None, None]).sum((-1, -2))
        tied = tied_count > 1
        record = dict(call_index=index, kind="MaxPool2d", name=name,
            input_shape=list(value.shape), outputs=indices.numel(), tied_outputs=int(tied.sum()),
            tied_fraction=float(tied.float().mean()), maximum_tie_multiplicity=int(tied_count.max()))
        if self.mode == "capture":
            self.baseline.append(dict(kind="MaxPool2d", name=name, input_shape=tuple(value.shape),
                indices=indices.detach().clone(), tied=tied.detach().clone()))
        else:
            changed = indices != reference["indices"]
            record.update(winner_changed_outputs=int(changed.sum()),
                changed_at_baseline_ties=int((changed & reference["tied"]).sum()),
                changed_at_baseline_unique=int((changed & ~reference["tied"]).sum()),
                tie_status_changed_outputs=int((tied != reference["tied"]).sum()))
        self.records.append(record)
        return pooled

    def scorer_pool(self, head, value):
        index, reference = self.entry("Scorer_amax", "scorer._pool.amax", value)
        weights = torch.softmax(head.pool_gate(value).squeeze(-1), dim=0)
        attentive = (weights[:, None] * value).sum(0)
        if self.mode == "frozen":
            # Anchored affine form is baseline-exact, and its derivative is the
            # same equal-weight subgradient as torch.amax at tied maxima.
            maximum = reference["maximum"] + ((value - reference["value"]) * reference["weights"]).sum(0)
        else:
            maximum = value.amax(0)
            winners = value == maximum[None]
            count = winners.sum(0)
            record = dict(call_index=index, kind="Scorer_amax", name="scorer._pool.amax",
                input_shape=list(value.shape), outputs=value.shape[1], tied_outputs=int((count > 1).sum()),
                tied_fraction=float((count > 1).float().mean()), maximum_tie_multiplicity=int(count.max()))
            if self.mode == "capture":
                self.baseline.append(dict(kind="Scorer_amax", name="scorer._pool.amax", input_shape=tuple(value.shape),
                    maximum=maximum.detach().clone(), value=value.detach().clone(),
                    weights=(winners.to(value.dtype) / count[None]).detach().clone(),
                    winners=winners.detach().clone(), tied=(count > 1).detach().clone()))
            else:
                changed = (winners != reference["winners"]).any(0)
                record.update(winner_set_changed_outputs=int(changed.sum()),
                    changed_at_baseline_ties=int((changed & reference["tied"]).sum()),
                    changed_at_baseline_unique=int((changed & ~reference["tied"]).sum()))
            self.records.append(record)
        pooled = torch.cat((attentive, maximum))
        self.pooled.append(pooled.detach())
        return pooled

    def run(self, mode, masks, fixed):
        if mode not in ("capture", "original", "frozen") or (mode == "capture" and self.baseline):
            raise ValueError("invalid max audit mode")
        self.mode, self.cursor, self.records, self.pooled = mode, 0, [], []
        result = pixel.raw_mask_logit(self.model, *masks, *fixed)[0]
        if self.cursor != 8 or len(self.pooled) != 2:
            raise ValueError("expected six patch MaxPool calls and two Scorer amax calls")
        return result, dict(max_operations=self.records,
            symmetric_abs_zero_components=int((self.pooled[0] == self.pooled[1]).sum()))


def check_close(actual, expected, name, atol=2e-6, rtol=2e-5):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(name + " is not equivalent; cannot interpret frozen-branch test")
    return dict(max_abs_error=float((actual - expected).abs().max()),
                l2_relative_error=float((actual - expected).norm() / expected.norm().clamp_min(1e-12)))


def directions_from_receipt(row, gradients):
    directions = []
    for side, gradient in zip("ab", gradients):
        selected = row["finite_difference"]["selected"][side]
        rc = torch.as_tensor(selected["pixel_rc"], dtype=torch.long)
        signs = torch.as_tensor(selected["direction"], dtype=torch.float32)
        if not torch.equal(gradient[0, 0, rc[:, 0], rc[:, 1]].sign(), signs):
            raise ValueError("saved finite-difference direction no longer matches gradient signs")
        direction = torch.zeros_like(gradient)
        direction[0, 0, rc[:, 0], rc[:, 1]] = signs
        directions.append(direction)
    return directions


def probe_case(model, row, arrays):
    masks = [torch.from_numpy(arrays["mask_" + side].astype(np.float32)).reshape(1, 1, 800, 800).requires_grad_(True)
             for side in "ab"]
    fixed = [torch.from_numpy(arrays["points_rc_" + side].copy())[None] for side in "ab"]
    fixed += [torch.from_numpy(arrays["valid_" + side].copy())[None] for side in "ab"]
    original = pixel.raw_mask_logit(model, *masks, *fixed)[0]
    original_g = torch.autograd.grad(original, masks)
    replay = check_close(original.detach(), torch.tensor(row["raw_head_logit"]), "saved original raw logit")
    saved_gradient = {}
    for side, gradient in zip("ab", original_g):
        saved_gradient[side] = check_close(gradient[0, 0], torch.from_numpy(arrays["signed_gradient_" + side]), "saved gradient " + side)
    directions = directions_from_receipt(row, original_g)
    analytic = float(sum((g * d).sum() for g, d in zip(original_g, directions)))
    with MaxBranches(model) as branches:
        captured, trace = branches.run("capture", masks, fixed)
        captured_g = torch.autograd.grad(captured, masks)
        captured_error = check_close(captured.detach(), original.detach(), "instrumented original logit")
        for actual, expected in zip(captured_g, original_g):
            check_close(actual, expected, "instrumented original gradient")
        frozen, frozen_trace = branches.run("frozen", masks, fixed)
        frozen_g = torch.autograd.grad(frozen, masks)
        frozen_error = check_close(frozen.detach(), original.detach(), "frozen baseline logit")
        gradient_errors = {side: check_close(g, ref, "frozen baseline gradient " + side)
                           for side, g, ref in zip("ab", frozen_g, original_g)}
        frozen_analytic = float(sum((g * d).sum() for g, d in zip(frozen_g, directions)))
        checks = []
        for mode, base, derivative in (("original", original, analytic), ("frozen", frozen, frozen_analytic)):
            for epsilon in (.001, .0005):
                with torch.no_grad():
                    high, high_trace = branches.run(mode, [m + epsilon * d for m, d in zip(masks, directions)], fixed)
                    low, low_trace = branches.run(mode, [m - epsilon * d for m, d in zip(masks, directions)], fixed)
                central = float((high - low) / (2 * epsilon))
                error = abs(central - derivative)
                checks.append(dict(mode=mode, epsilon=epsilon, analytic_directional_derivative=derivative,
                    high_logit=float(high), low_logit=float(low), central_derivative=central,
                    forward_slope=float((high - base.detach()) / epsilon),
                    backward_slope=float((base.detach() - low) / epsilon),
                    absolute_error=error, relative_error=error / abs(derivative),
                    within_original_tolerance=error <= .001 + .1 * abs(derivative),
                    high_trace=high_trace, low_trace=low_trace))
    restored = pixel.raw_mask_logit(model, *masks, *fixed)[0]
    check_close(restored.detach(), original.detach(), "restored instance methods")
    return dict(dataset=row["dataset"], pair_id=row["pair_id"], name=row["name"], stratum=row["stratum"],
        raw_head_logit=float(original), historical_fd_status=row["finite_difference"]["status"],
        history_status_unchanged=True, saved_forward_equivalence=replay, saved_gradient_equivalence=saved_gradient,
        instrumented_forward_equivalence=captured_error, frozen_forward_equivalence=frozen_error,
        frozen_gradient_equivalence=gradient_errors, baseline_trace=trace,
        checks=checks, original_tolerance=dict(absolute=.001, relative=.1),
        interpretation="frozen max winners define a different nearby function with baseline-equivalent value/gradient; explanatory contrast only, never automatic validation of original pixel attribution",
        coverage="six explicit patch MaxPool2d calls plus two Scorer amax calls; softmax internal stabilizing maxima not frozen (softmax is smooth); mask-independent landmark count clamp unchanged; symmetric abs zero components recorded")


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only fresh process required")
    os.nice(max(0, 10 - os.nice(0)))
    random.seed(260913)
    np.random.seed(260913)
    torch.random.default_generator.manual_seed(260913)
    torch.use_deterministic_algorithms(True)
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluation
    root, output = Path(args.pixel_root), Path(args.output)
    protocol = json.loads((root / "protocol.json").read_text())
    if sha(root / "cases.json") != protocol["cases_sha256"] or sha(pixel.__file__) != protocol["script_sha256"]:
        raise ValueError("source pixel results/code changed")
    cases = json.loads((root / "cases.json").read_text())
    cases = [next(row for row in cases if row["pair_id"] == pair_id and row["dataset"] == "real") for pair_id in SELECTED_IDS]
    model, identity = evaluation.load_frozen_model(protocol["model"]["training_run"], protocol["model"]["selection"])
    if identity["checkpoint_sha256"] != protocol["model"]["checkpoint_sha256"]:
        raise ValueError("checkpoint changed")
    model = model.cpu().eval().requires_grad_(False)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema_version=SCHEMA, status="running", parameters_fitted=False,
        cpu_threads=1, nice=os.nice(0), device="cpu", torch_version=str(torch.__version__),
        source_cases_sha256=sha(root / "cases.json"), source_protocol_sha256=sha(root / "protocol.json"),
        source_pixel_script_sha256=sha(pixel.__file__), script_sha256=sha(__file__),
        checkpoint_sha256=identity["checkpoint_sha256"], selected_pair_ids=list(SELECTED_IDS),
        historical_status_writes=False, image_generation=False, inputs=[])
    save(output / "protocol.json", receipt)
    started, results = time.perf_counter(), []
    try:
        for row in cases:
            path = root / row["arrays_path"]
            if sha(path) != row["arrays_sha256"]:
                raise ValueError("saved mask/gradient arrays changed")
            with np.load(path, allow_pickle=False) as arrays:
                result = probe_case(model, row, arrays)
            results.append(result)
            receipt["inputs"].append(dict(pair_id=row["pair_id"], arrays_sha256=sha(path)))
            save(output / "partial_results.json", results)
            print(json.dumps(dict(name=row["name"], completed=len(results), total=2)), flush=True)
        save(output / "results.json", results)
        receipt.update(status="complete_explanatory_control", completed_count=2, results_sha256=sha(output / "results.json"))
    except BaseException as exc:
        receipt.update(status="failed", error=repr(exc), completed_count=len(results))
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        save(output / "protocol.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel-root", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
