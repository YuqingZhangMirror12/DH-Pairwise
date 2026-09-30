"""Frozen-head interventions on saved contextual tokens, not physical masks.

Test whether non-inlier context suppresses a predicted seam's score. Preserve
the exact Matcher output; never use a GT seam or layout error to pick tokens.
Includes same-count random deletion and a token-permutation sanity control.
All repetitions are deterministic and case selection precedes this probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

SCHEMA = "frozen-scorer-token-dilution/1"
FRACTIONS = (0., .25, .5, .75, 1.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def compact_membership(original_ids, inlier_ids):
    ids, inliers = np.asarray(original_ids), np.asarray(inlier_ids)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or inliers.ndim != 1:
        raise ValueError("indices must be unique one-dimensional original IDs")
    if not np.isin(inliers, ids).all():
        raise ValueError("predicted inlier endpoint is not a valid saved token")
    return np.isin(ids, inliers)


def intervention_masks(inlier_a, inlier_b, *, repeats=16, seed=260919):
    if repeats < 2:
        raise ValueError("at least two random repetitions required")
    members = [np.asarray(inlier_a, dtype=bool), np.asarray(inlier_b, dtype=bool)]
    if any(m.ndim != 1 or not len(m) for m in members):
        raise ValueError("nonempty 1-D token sets required")
    rng = np.random.default_rng(seed)
    result = []
    # Retained fractions share a random ordering within a repetition. Thus the
    # added context is nested, rather than replacing all context at each dose.
    for repeat in range(repeats):
        orders = [rng.permutation(np.where(~m)[0]) for m in members]
        for fraction in FRACTIONS:
            if repeat and fraction in (0., 1.):
                continue
            masks = []
            for member, order in zip(members, orders):
                mask = member.copy()
                mask[order[:int(np.ceil(fraction * len(order)))]] = True
                masks.append(mask)
            result.append(dict(kind="retain_inliers_add_context", repeat=repeat,
                               noninlier_fraction=fraction, masks=masks))
    result.append(dict(kind="remove_inliers", repeat=0, masks=[~m for m in members]))
    for repeat in range(repeats):
        masks = []
        for member in members:
            mask = np.ones(len(member), dtype=bool)
            mask[rng.choice(len(member), int(member.sum()), replace=False)] = False
            masks.append(mask)
        result.append(dict(kind="remove_random_same_count", repeat=repeat, masks=masks))
    return result


@torch.inference_mode()
def run_head_probe(head, a, b, member_a, member_b, *, repeats=16, seed=260919):
    if head.training or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("frozen eval head and two compact contextual feature arrays required")
    if a.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("this bounded diagnostic is CPU-only")
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("nonfinite saved context")
    def evaluate(ma, mb):
        va, vb = torch.as_tensor(ma, dtype=torch.bool), torch.as_tensor(mb, dtype=torch.bool)
        z = head(a[None], b[None], va[None], vb[None])[0]
        if not torch.isfinite(z):
            raise ValueError("nonfinite intervention score")
        return dict(logit=float(z), probability=float(z.sigmoid()),
                    count_a=int(va.sum()), count_b=int(vb.sum()),
                    no_evidence=not bool(va.any() and vb.any()))
    full_a, full_b = np.ones(len(a), bool), np.ones(len(b), bool)
    baseline = evaluate(full_a, full_b)
    rng = np.random.default_rng(seed ^ 0xBA51)
    pa, pb = rng.permutation(len(a)), rng.permutation(len(b))
    perm = head(a[pa][None], b[pb][None], torch.ones(1, len(a), dtype=torch.bool),
                torch.ones(1, len(b), dtype=torch.bool))[0]
    error = abs(float(perm) - baseline["logit"])
    if error > 2e-5 + 2e-5 * abs(baseline["logit"]):
        raise ValueError("unexpected scorer sensitivity to compact-token permutation")
    rows = []
    for spec in intervention_masks(member_a, member_b, repeats=repeats, seed=seed):
        ma, mb = spec.pop("masks")
        row = {**spec, **evaluate(ma, mb)}
        row.update(delta_logit=row["logit"] - baseline["logit"],
                   retained_inliers_a=int((ma & member_a).sum()),
                   retained_inliers_b=int((mb & member_b).sum()))
        rows.append(row)
    return dict(baseline=baseline, permutation_logit_error=error, interventions=rows,
                inlier_count_a=int(np.asarray(member_a).sum()),
                inlier_count_b=int(np.asarray(member_b).sum()))


def run(args):
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluation
    torch.set_num_threads(1)
    root, output = Path(args.probe_root), Path(args.output)
    selection_path = Path(args.selection_json)
    selected = json.loads(selection_path.read_text())
    strata = {(r["dataset"], r["pair_id"]): r for r in selected}
    if len(strata) != len(selected):
        raise ValueError("duplicate fixed-selection IDs")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = dict(schema_version=SCHEMA, status="running", models=args.models,
        repeats=args.repeats, fractions=list(FRACTIONS), selected_count=len(selected),
        selection_sha256=sha(selection_path), source_sha256=sha(__file__),
        torch_version=str(torch.__version__), device="cpu", cpu_threads=1,
        parameters_fitted=False, thresholds_fitted=False, GT_used_for_token_selection=False,
        scope="off-manifold intervention AFTER frozen matcher context; neither mask nor Sinkhorn recomputed",
        controls="same-count random deletion; nested context doses; independent A/B permutation",
        caveat="predicted inliers can be wrong; dose-response does not prove physical shape causality or a new classifier's generalization",
        inputs=[])
    write_json(output / "protocol.json", protocol)
    results = []
    try:
        for name in args.models:
            folder = root / name
            source = json.loads((folder / "protocol.json").read_text())
            cases = json.loads((folder / "cases.json").read_text())
            if source["status"] != "complete" or len(cases) != len(selected):
                raise ValueError("heatmap source not complete for fixed selection")
            if {(r["dataset"], r["pair_id"]) for r in cases} != set(strata):
                raise ValueError("source and fixed selection disagree")
            original = source["model"]
            model, identity = evaluation.load_frozen_model(original["training_run"], original["selection"])
            if identity["checkpoint_sha256"] != original["checkpoint_sha256"]:
                raise ValueError("source checkpoint changed")
            head = model.score_head.cpu().eval().requires_grad_(False)
            protocol["inputs"].append(dict(model=name, checkpoint_sha256=identity["checkpoint_sha256"],
                cases_sha256=sha(folder / "cases.json"), protocol_sha256=sha(folder / "protocol.json")))
            for case in cases:
                path = folder / case["arrays_path"]
                if sha(path) != case["arrays_sha256"]:
                    raise ValueError("saved feature arrays changed")
                with np.load(path, allow_pickle=False) as arrays:
                    a, b = (torch.from_numpy(arrays["context_input_" + side + "_features"].copy()) for side in "ab")
                    members = [compact_membership(arrays["valid_indices_" + side],
                        case["layout"]["inlier_token_indices_" + side]) for side in "ab"]
                key = case["dataset"] + "\0" + case["pair_id"]
                seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                measured = run_head_probe(head, a, b, *members, repeats=args.repeats, seed=seed)
                error = abs(measured["baseline"]["logit"] - case["raw_head_logit"])
                if error > 2e-5 + 2e-5 * abs(case["raw_head_logit"]):
                    raise ValueError("saved input/head replay differs from original inference")
                results.append(dict(model=name, dataset=case["dataset"], pair_id=case["pair_id"],
                    name=strata[(case["dataset"], case["pair_id"])].get("name"),
                    stratum=strata[(case["dataset"], case["pair_id"])]["stratum"],
                    label=case["label"], layout_valid=case["layout"]["valid"],
                    layout_error_px=case["layout"].get("translation_l2_px"),
                    threshold=identity["operating_points"]["thresholds"]["max_f1"],
                    saved_forward_error=error, arrays_sha256=case["arrays_sha256"], **measured))
                write_json(output / "partial_results.json", results)
            print(json.dumps(dict(model=name, cases_complete=len(cases), total_complete=len(results))), flush=True)
        write_json(output / "results.json", results)
        protocol.update(status="complete", completed_count=len(results), results_sha256=sha(output / "results.json"))
    except Exception as exc:
        protocol.update(status="failed", error=repr(exc), completed_count=len(results))
        raise
    finally:
        protocol["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "protocol.json", protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", default=["s4", "s6", "s6_depth4", "s7"])
    parser.add_argument("--repeats", type=int, default=16)
    run(parser.parse_args())
