"""Evaluate a SIM-VAL-frozen score-design budget, without held-out selection.

The old Top2 translation decoder is fixed, not its predictions after training.
Save target-blind scores, P/R candidate diagnostics and layouts first; attach
REAL GT and the user's keep/exclude membership only after prediction freeze.
Turufan has only 301 positive labels and no pose GT: never report its accuracy,
precision, F1, AP/AUROC or layout accuracy. No thresholds are fitted here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from experiments.rachel_n512_formal_30k import evaluate_realism_checkpoint as fixed
from experiments.rachel_n512_formal_30k import run_layout_decoder_experiment as common
from experiments.rachel_n512_formal_30k import run_real_layout_decoder_experiment as real
from experiments.rachel_n512_formal_30k.fragment_size_strata import pair_size_metadata
from experiments.rachel_n512_formal_30k.score_design_checkpoint_io import load_owned_epoch_checkpoint
from experiments.rachel_n512_formal_30k.resampled_input_support import (
    InputContourResampler, make_ablation_loader,
)
from experiments.rachel_n512_formal_30k.run_real_contiguous_seam_ablation import load_prepared_cache
from staging.pairwise_v0_2.models.translation_layout import estimate_translation_layout
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed

SCHEMA = "rachel-score-design-evaluation/1"
BRANCHES = ("coarse", "local", "fused")
FIELDS = ("mask_a", "mask_b", "points_rc_a", "points_rc_b", "contour_valid_a", "contour_valid_b")
DEFAULT_OOD = Path("/root/autodl-tmp/turufan_ood_pairwise_20260912_001/prepared")


def save(path, value):
    """Write a run-owned artifact; never overwrite canonical inputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(common.clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_frozen_model(training_run, budget, selection):
    """Consume exactly train_score_design's completed budget-freeze contract."""
    # Import late: analytical helpers remain usable while the trainer module
    # is being installed and never accidentally load an old model factory.
    from experiments.rachel_n512_formal_30k.train_score_design import load_score_checkpoint
    root = Path(training_run).resolve(strict=True)
    path = root / "budget_freezes" / f"{budget:03d}" / "freeze.json"
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if (freeze.get("schema_version") != "rachel-score-design-training/1"
            or freeze.get("status") != "frozen_at_budget"
            or freeze.get("held_out_used_for_fit") is not False
            or freeze.get("selection_population") != "cleanVAL3000 only"):
        raise ValueError("requires a complete SIM-VAL-only budget freeze")
    if (freeze.get("budget_epochs") != budget or freeze.get("budget_exposures") != budget * 24000
            or freeze.get("eligible_epoch_range") != [5, budget]):
        raise ValueError("budget freeze does not match requested training budget")
    selected = freeze["winners"][selection]
    if selected.get("test_or_real_or_ood_used_for_fit") is not False or selected.get("selection") != selection:
        raise ValueError("winner has not been frozen under the requested SIM-VAL selection rule")
    checkpoint_path = Path(selected["checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    digest = sealed._sha256_file(checkpoint_path)
    if digest != selected["checkpoint_sha256"]:
        raise ValueError("budget winner checkpoint differs from its frozen identity")
    payload = load_owned_epoch_checkpoint(root, checkpoint_path, selected["selected_epoch"], digest)
    if (payload["epoch"] != selected["selected_epoch"] or not 5 <= payload["epoch"] <= budget
            or payload["global_exposure"] != selected["selected_global_exposure"]):
        raise ValueError("checkpoint epoch is not eligible for this budget winner")
    model = load_score_checkpoint(payload)
    branch_thresholds = selected["classifier_thresholds"]
    operating = selected["operating_points"]["thresholds"]
    if set(branch_thresholds) != set(BRANCHES) or not {"max_f1", "recall_95"} <= set(operating):
        raise ValueError("freeze lacks branch thresholds or fused max-F1/recall95 workpoints")
    for value in list(branch_thresholds.values()) + list(operating.values()):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("frozen thresholds must be finite probabilities")
    config = asdict(model.config)
    identity = dict(training_run=str(root), budget=budget, selection=selection,
        freeze_path=str(path), freeze_sha256=sealed._sha256_file(path),
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=digest,
        epoch=payload["epoch"], seed=int(payload["seed"]),
        architecture=payload["score_design"]["architecture"], model_config=config,
        candidate_config=asdict(model.candidate_config) if hasattr(model, "candidate_config") else None,
        score_design_metadata=payload["score_design"],
        resample_contour_cap=payload.get("resample_contour_cap"),
        classifier_thresholds=branch_thresholds, operating_points=selected["operating_points"],
        winner_record=selected, test_or_real_used_for_fit=False)
    return model.eval().requires_grad_(False), identity


def overlap_small_fraction(mask_a, mask_b, translation_a_to_b_rc):
    """Clipping-safe native masks, nearest-integer B placement, full-area denom.

    t is A->B; B placed in A's frame uses -t. Canvas overlap is computed on
    their shared support, but the denominator is min(original A,B areas), not
    the visible/cropped part. Quantization error affects boundary pixels.
    """
    a = np.asarray(mask_a, bool).squeeze()
    b = np.asarray(mask_b, bool).squeeze()
    shift = np.rint(-np.asarray(translation_a_to_b_rc)).astype(np.int64)
    low = np.maximum(0, shift)
    high = np.minimum(a.shape, shift + np.asarray(b.shape))
    if np.any(high <= low):
        return 0.0
    alo = tuple(slice(int(x), int(y)) for x, y in zip(low, high))
    blo = tuple(slice(int(x), int(y)) for x, y in zip(low - shift, high - shift))
    return float(np.count_nonzero(a[alo] & b[blo]) / max(1, min(a.sum(), b.sum())))


def _candidate_details(output, index):
    details = getattr(output, "score_details", None)
    if details is None:
        return None
    # Preserve named network outputs, not evaluator-created labels. Unknown
    # non-tensor metadata is skipped instead of guessing its batch dimension.
    result = {}
    for key, value in details.items():
        if torch.is_tensor(value) and value.ndim >= 1:
            result[key] = value[index].detach().cpu().numpy()
    for source, target in (("candidate_logits", "candidate_correct_probability"),
                           ("candidate_pair_logits", "candidate_pair_probability")):
        if source in details:
            result[target] = torch.sigmoid(details[source][index]).detach().float().cpu().numpy()
    return common.clean(result)


def predict_batch(model, batch, device):
    tensors = [sealed._tensor(getattr(batch, key), device,
               torch.bool if key.startswith("contour_valid") else torch.float32) for key in FIELDS]
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = model(*tensors)
    scores = {b: getattr(output, b + "_probability").detach().float().cpu().numpy() for b in BRANCHES}
    count = len(batch.pair_ids)
    if any(s.shape != (count,) or not np.isfinite(s).all() or np.any((s < 0) | (s > 1))
           for s in scores.values()):
        raise ValueError("invalid score probabilities")
    assignments = output.assignment.detach().float().cpu().numpy()
    rows = []
    for i, pair_id in enumerate(batch.pair_ids):
        estimate = estimate_translation_layout(batch.points_rc_a[i], batch.points_rc_b[i],
            assignments[i], batch.contour_valid_a[i], batch.contour_valid_b[i], config=fixed.TOP2_CONFIG)
        diagnostics = asdict(estimate)
        for key in ("t_a_to_b_rc", "candidate_indices", "inlier_mask"):
            diagnostics.pop(key, None)
        layout = dict(translation_rc=estimate.t_a_to_b_rc, offset_b_in_a_rc=-estimate.t_a_to_b_rc,
            valid=bool(estimate.valid), diagnostics=diagnostics,
            overlap_small_fraction=overlap_small_fraction(batch.mask_a[i], batch.mask_b[i],
                estimate.t_a_to_b_rc) if estimate.valid else None)
        rows.append(common.clean(dict(pair_id=pair_id, fragment_a=batch.fragment_a_tokens[i],
            fragment_b=batch.fragment_b_tokens[i], decision_valid=bool(output.decision_valid[i]),
            classification={b: float(scores[b][i]) for b in BRANCHES},
            candidate_details=_candidate_details(output, i),
            layouts={fixed.DECODER_NAME: layout}, **pair_size_metadata(batch.mask_a[i], batch.mask_b[i]))))
    return rows


def positive_layout(rows, threshold, tolerance):
    positive = [r for r in rows if r["label"]]
    good = [r for r in positive if r["layouts"][fixed.DECODER_NAME]["valid"]
            and r["layouts"][fixed.DECODER_NAME]["translation_l2_px"] is not None
            and r["layouts"][fixed.DECODER_NAME]["translation_l2_px"] <= tolerance]
    accepted = lambda r: r["decision_valid"] and r["classification"]["fused"] >= threshold
    good_ids = {r["pair_id"] for r in good}
    return dict(positive_count=len(positive), tolerance_px=tolerance, raw_correct=len(good),
        raw_recall=len(good) / len(positive) if positive else None,
        accepted_correct=sum(accepted(r) for r in good),
        classification_FN_but_layout_correct=sum(not accepted(r) for r in good),
        accepted_positive_bad_layout=sum(accepted(r) and r["pair_id"] not in good_ids for r in positive),
        end_to_end_positive_recall=sum(accepted(r) for r in good) / len(positive) if positive else None,
        accepted_negative_count=sum(accepted(r) for r in rows if not r["label"]))


def summarize_group(rows, identity, *, pose_gt_available):
    """Never fabricate full binary metrics on positive-only/negative-only groups."""
    if not rows:
        return dict(sample_count=0)
    labels = np.array([r["label"] for r in rows], bool)
    valid = np.array([r["decision_valid"] for r in rows], bool)
    result = dict(sample_count=len(rows), positive_count=int(labels.sum()),
        negative_count=int((~labels).sum()), decision_valid_count=int(valid.sum()),
        classification={}, score_distributions={}, extreme_score_cases={})
    for branch in BRANCHES:
        scores = np.array([r["classification"][branch] for r in rows], float)
        thresholds = {"max_f1": identity["classifier_thresholds"][branch]}
        if branch == "fused":
            thresholds.update({k: identity["operating_points"]["thresholds"][k] for k in ("max_f1", "recall_95")})
        result["classification"][branch] = {}
        for name, threshold in thresholds.items():
            accepted = valid & (scores >= threshold)
            if labels.all():
                metrics = dict(threshold=threshold, accepted_positive_count=int(accepted.sum()),
                    false_negative_count=int((~accepted).sum()), positive_recall=float(accepted.mean()))
            elif (~labels).all():
                metrics = dict(threshold=threshold, false_positive_count=int(accepted.sum()),
                    true_negative_count=int((~accepted).sum()), false_positive_rate=float(accepted.mean()))
            else:
                # Invalid decisions operationally reject and rank below every
                # valid probability; source scores remain unchanged on disk.
                metrics = common.classification(labels, np.where(valid, scores, -1), threshold)
            result["classification"][branch][name] = metrics
        result["score_distributions"][branch] = dict(
            quantile_levels=[0, .1, .25, .5, .75, .9, 1],
            quantiles=np.quantile(scores, [0, .1, .25, .5, .75, .9, 1]).tolist(),
            mean=float(scores.mean()), exact_zero_count=int((scores == 0).sum()), exact_one_count=int((scores == 1).sum()))
        low_ids = [r["pair_id"] for r in rows if r["label"] and r["classification"][branch] <= .01]
        high_ids = [r["pair_id"] for r in rows if not r["label"] and r["classification"][branch] >= .9]
        result["extreme_score_cases"][branch] = dict(positive_score_le_001_count=len(low_ids),
            positive_score_le_001_pair_ids=low_ids, negative_score_ge_09_count=len(high_ids),
            negative_score_ge_09_pair_ids=high_ids)
    if pose_gt_available:
        result["layout"] = {op: {str(tol): positive_layout(rows, threshold, tol) for tol in (10, 20)}
            for op, threshold in identity["operating_points"]["thresholds"].items()
            if op in ("max_f1", "recall_95")}
    else:
        result["layout_metrics_unavailable"] = "No layout ground truth; decoded poses are visual diagnostics only."
    if len(set(labels.tolist())) < 2:
        result["binary_metrics_unavailable"] = "A single-class population cannot establish accuracy, precision, F1, AP or AUROC."
    return result


def summarize_populations(rows, identity, split):
    groups = {"all": rows}
    if split == "real":
        positive = [r for r in rows if r["label"]]
        negative = [r for r in rows if not r["label"]]
        keep = [r for r in positive if r["review_status"] == "keep"]
        exclude = [r for r in positive if r["review_status"] == "exclude"]
        strict = [r for r in negative if r["strict_member"]]
        constructed = [r for r in negative if not r["strict_member"]]
        groups.update(kept_positive=keep, excluded_positive=exclude, negative_all=negative,
            negative_strict=strict, negative_constructed=constructed,
            kept_plus_all_negative=keep + negative, kept_plus_strict_negative=keep + strict,
            kept_plus_constructed_negative=keep + constructed)
    elif split == "test":
        groups.update(positive=[r for r in rows if r["label"]], negative=[r for r in rows if not r["label"]])
    return {name: summarize_group(group, identity, pose_gt_available=split != "ood") for name, group in groups.items()}


def compare_baseline(rows, baseline_evaluation, split):
    """Descriptive paired regressions; never select models or choose a fallback."""
    root = Path(baseline_evaluation)
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["status"] != "complete" or protocol["split"] != split:
        raise ValueError("baseline must be a completed evaluation of the same split")
    previous = {r["pair_id"]: r for r in map(json.loads, (root / "pair_results.jsonl").read_text().splitlines())}
    if set(previous) != {r["pair_id"] for r in rows}:
        raise ValueError("baseline evaluation population differs")
    result = dict(source=str(root.resolve()), selection_changed=False, fallback_applied=False)
    if split == "ood":
        result["pose_comparison_unavailable"] = "No OOD pose GT."
        return result
    good = lambda r, tol: r["layouts"][fixed.DECODER_NAME]["valid"] and (
        r["layouts"][fixed.DECODER_NAME]["translation_l2_px"] is not None) and (
        r["layouts"][fixed.DECODER_NAME]["translation_l2_px"] <= tol)
    groups = {"all_positive": [r for r in rows if r["label"]]}
    if split == "real":
        groups["kept_positive"] = [r for r in rows if r.get("review_status") == "keep"]
    for group, selected in groups.items():
        result[group] = {}
        for tol in (10, 20):
            gained = [r["pair_id"] for r in selected if good(r, tol) and not good(previous[r["pair_id"]], tol)]
            lost = [r["pair_id"] for r in selected if not good(r, tol) and good(previous[r["pair_id"]], tol)]
            result[group][str(tol)] = dict(n=len(selected), gained=len(gained), lost=len(lost),
                gained_pair_ids=gained, lost_pair_ids=lost)
    return result


def run(args):
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers nonnegative")
    if args.split == "real" and (not args.keep_ids or not Path(args.keep_ids).is_file()):
        raise ValueError("real evaluation requires the existing --keep-ids export")
    torch.set_num_threads(1)
    model, identity = load_frozen_model(args.training_run, args.budget, args.selection)
    sealed._set_determinism(identity["seed"])
    device = torch.device(args.device)
    model = model.to(device).eval().requires_grad_(False)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    cap = int(identity["model_config"]["contour_cap"])
    resample_cap = identity["resample_contour_cap"]
    if resample_cap is not None and resample_cap != cap:
        raise ValueError("training resampling cap differs from the model input cap")
    # A paired re-extracted512 control explicitly records512, whereas the
    # untouched baseline recordsNone. Do not silently mix these protocols.
    resampler = InputContourResampler(cap) if cap != 512 or resample_cap is not None else None
    source, targets, metadata = {}, {}, None
    if args.split == "test":
        root = Path(args.dataset)
        manifest_path = root / "pairs/test.jsonl"
        manifest = [json.loads(x) for x in manifest_path.read_text().splitlines() if x]
        if len(manifest) != 3000 or sum(r["label"] for r in manifest) != 1500:
            raise ValueError("requires the complete original balanced TEST3000")
        expected_ids = [r["pair_id"] for r in manifest]
        source_units = {r["pair_id"]: sorted({r["fragment_" + side]["split_unit_id"] for side in "ab"}) for r in manifest}
        ds = RachelPairDataset(root, "test")
        batches = make_ablation_loader(ds, tuple(range(len(ds))), batch_size=args.batch_size,
            num_workers=args.workers, seed=identity["seed"], contour_cap=512)
        source = dict(dataset_root=str(root), manifest_sha256=sealed._sha256_file(manifest_path))
    else:
        cache = Path(args.prepared_cache if args.split == "real" else args.ood_prepared)
        if args.split == "real":
            metadata, arrays = load_prepared_cache(cache)
            if (len(metadata["pairs"]) != 1016 or sum(r["label"] for r in metadata["pairs"]) != 508
                    or sum(r["label"] and r["strict"] for r in metadata["pairs"]) != 508):
                raise ValueError("requires the unchanged Dunhuang1016 population")
        else:
            metadata = json.loads((cache / "manifest.json").read_text())
            if (len(metadata["pairs"]) != 301 or len(metadata["fragment_ids"]) != 602
                    or metadata.get("layout_gt_provided") is not False
                    or metadata.get("negative_pairs_constructed") is not False):
                raise ValueError("requires the original positive-only Turufan301 population")
            with np.load(cache / "inputs.npz", allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in ("packed_masks", "points", "valid")}
        expected_ids = [r["pair_id"] for r in metadata["pairs"]]
        batches = real.input_batches(metadata, arrays, args.batch_size)
        source = dict(prepared_cache=str(cache), manifest_sha256=sealed._sha256_file(cache / "manifest.json"))
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate evaluation pair ID")
    protocol = dict(schema_version=SCHEMA, status="running", split=args.split, model=identity,
        **source, sample_count=len(expected_ids), batch_size=args.batch_size, precision="fp32",
        decoder=fixed.DECODER_NAME, decoder_config=asdict(fixed.TOP2_CONFIG),
        decoder_design_unchanged=True, layout_predictions_may_change_with_trained_weights=True,
        resampling="original prepared512" if resampler is None else "mask-only original canvas; Gaussian sigma3",
        model_input_fields=list(FIELDS), ground_truth_used_to_select_candidates=False,
        gt_attached_after_complete_prediction_freeze=True, thresholds_fitted=False,
        test_or_real_used_for_fit=False, ood_used_for_fit=False,
        overlap_definition="intersection/min(original areas), B placement=-t, nearest integer raster",
        invalid_decision_policy="reject; rank score below all valid probabilities",
        script_sha256=sealed._sha256_file(Path(__file__)))
    save(destination / "protocol.json", protocol)
    predictions = []
    started = time.monotonic()
    try:
        with (destination / "pair_predictions.jsonl").open("x", encoding="utf-8") as stream:
            for batch in batches:
                if resampler is not None:
                    batch = resampler.resample_batch(batch)
                batch_rows = predict_batch(model, batch, device)
                if args.split == "test":
                    for i, pair_id in enumerate(batch.pair_ids):
                        targets[pair_id] = dict(label=bool(batch.labels[i]),
                            translation_rc=common.clean(batch.translation_a_to_b_rc[i]) if batch.translation_valid[i] else None,
                            source_unit_ids=source_units[pair_id])
                predictions.extend(batch_rows)
                for row in batch_rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if [r["pair_id"] for r in predictions] != expected_ids:
            raise ValueError("predictions do not match the complete ordered evaluation population")
        save(destination / "prediction_complete.json", dict(status="all_predictions_frozen",
            sample_count=len(predictions), real_gt_opened=False, review_labels_opened=False,
            checkpoint_sha256=identity["checkpoint_sha256"]))
        if args.split == "real":
            rows = real.attach_ground_truth(predictions, metadata["pairs"], args.translation_gt_json)
            kept = set(json.loads(Path(args.keep_ids).read_text())["kept_positive_pair_ids"])
            positives = {r["pair_id"] for r in rows if r["label"]}
            if len(kept) != 295 or not kept <= positives:
                raise ValueError("keep export differs from the reviewed295 positive cohort")
            for row in rows:
                row["review_status"] = ("keep" if row["pair_id"] in kept else "exclude") if row["label"] else "not_reviewed_negative"
            protocol["keep_ids_sha256"] = sealed._sha256_file(Path(args.keep_ids))
        elif args.split == "test":
            rows = fixed.attach_test_targets(predictions, targets)
        else:
            rows = predictions
            for row in rows:
                row.update(label=True, target_translation_rc=None, layout_gt_available=False)
                row["layouts"][fixed.DECODER_NAME]["translation_l2_px"] = None
        fixed.write_rows(destination / "pair_results.jsonl", rows)
        summary = dict(status="complete", split=args.split, model=identity,
            groups=summarize_populations(rows, identity, args.split),
            selection_on_this_population=False, threshold_fitting_performed=False)
        if args.baseline_evaluation:
            summary["paired_baseline_layout"] = compare_baseline(rows, args.baseline_evaluation, args.split)
        save(destination / "summary.json", summary)
        protocol.update(status="complete", elapsed_seconds=time.monotonic() - started)
        save(destination / "protocol.json", protocol)
        print(json.dumps(dict(status="complete", split=args.split, sample_count=len(rows),
                              output=str(destination), elapsed_seconds=protocol["elapsed_seconds"])), flush=True)
        return summary
    except Exception as error:
        protocol.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic() - started)
        save(destination / "protocol.json", protocol)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run", required=True)
    p.add_argument("--budget", required=True, type=int, choices=(5, 10, 20, 30, 50))
    p.add_argument("--selection", required=True, choices=("max_f1", "recall95"))
    p.add_argument("--split", required=True, choices=("test", "real", "ood"))
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="/root/autodl-tmp/dataset_rachel_pairwise_n512_v1")
    p.add_argument("--prepared-cache", default=str(fixed.DEFAULT_PREPARED_CACHE))
    p.add_argument("--ood-prepared", default=str(DEFAULT_OOD))
    p.add_argument("--translation-gt-json", default=str(real.DEFAULT_TRANSLATION_GT))
    p.add_argument("--keep-ids")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--baseline-evaluation")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
