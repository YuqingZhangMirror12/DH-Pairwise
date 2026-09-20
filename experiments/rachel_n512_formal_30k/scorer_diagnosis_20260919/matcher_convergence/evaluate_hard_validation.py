"""CPU-only, frozen typed S7 Matcher diagnostics on a fixed hard-VAL manifest.

No training, Scorer, thresholds, REAL/OOD, GPU lock or checkpoint selection.
Default is manifest/stat preflight; --execute is explicit. Model inputs remain
the standard six tensors. The separate materialization report controls only
supervised loss eligibility, never Matcher forward or the production decoder.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import FunctionType

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence import evaluate_matcher_simval as original

INPUT_SCHEMA = "s7-hard-simval-diagnostic/1"
SCHEMA = "s7-hard-simval-matcher-evaluation/1"
RECIPES = ("clean", "wave", "local", "seam_gaps", "partial_curve")
EPOCHS = (12, 16, 20)


def validate_manifest(record, source_rows, *, allow_pilot=False):
    allowed = ("complete", "pilot_complete") if allow_pilot else ("complete",)
    if (record.get("schema") != INPUT_SCHEMA or record.get("split") != "val"
            or record.get("status") not in allowed):
        raise ValueError("requires complete hard-SIMVAL diagnostic manifest")
    protocol = record.get("protocol", {})
    if (protocol.get("diagnostic_only") is not True
            or protocol.get("source_val_manifest_sha256") != original.VAL_HASH
            or type(protocol.get("seed")) is not int):
        raise ValueError("diagnostic/source VAL/seed binding differs")
    source = {r["pair_id"]: r for r in source_rows}
    if not source or len(source) != len(source_rows):
        raise ValueError("source VAL IDs must be nonempty and unique")
    entries = record.get("entries", [])
    ids, keys, counts = set(), set(), Counter()
    for entry in entries:
        pair, source_id, recipe = (entry.get(k) for k in ("pair_id", "source_pair_id", "recipe"))
        if (not isinstance(pair, str) or not pair or pair == source_id or pair in ids
                or recipe not in RECIPES or (source_id, recipe) in keys):
            raise ValueError("novel unique pair IDs and unique source/recipe required")
        if (source_id not in source or entry.get("label") not in (0, 1)
                or entry["label"] != source[source_id]["label"]
                or type(entry.get("changed_pair")) is not bool):
            raise ValueError("source ID/label or actual changed flag differs")
        if recipe == "clean" and entry["changed_pair"]:
            raise ValueError("clean reference cannot be changed")
        if "source_family_overlap" in entry and type(entry["source_family_overlap"]) is not bool:
            raise ValueError("optional source_family_overlap must be an explicit boolean")
        ids.add(pair); keys.add((source_id, recipe)); counts[recipe, bool(entry["label"])] += 1
    if not entries or "clean" not in {r for r, _ in counts}:
        raise ValueError("nonempty manifest and paired clean references required")
    for source_id, recipe in keys:
        if recipe != "clean" and (source_id, "clean") not in keys:
            raise ValueError("stress row is missing its same-source clean reference")
    covered = {source_id for source_id, _ in keys}
    source_counts = Counter(source_id for source_id, _ in keys)
    for source_id in covered:
        if source_counts[source_id] != 2 or (source_id, "clean") not in keys:
            raise ValueError("each included source requires exactly one clean and one variant")
    for recipe in {r for r, _ in counts}:
        if counts[recipe, True] != counts[recipe, False]:
            raise ValueError("each requested recipe must be positive/negative balanced")
    if record["status"] == "complete":
        if (len(source) != 3000 or covered != set(source) or len(entries) != 6000
                or counts["clean", True] != 1500 or counts["clean", False] != 1500
                or any(counts[r, label] != 375 for r in RECIPES[1:] for label in (True, False))):
            raise ValueError("formal complete requires all original3000 sources,6000 rows,clean3000 and750 per recipe")
    return {r: dict(count=counts[r, True] + counts[r, False],
                    positive_count=counts[r, True], negative_count=counts[r, False])
            for r in RECIPES if counts[r, True] + counts[r, False]}


def preflight(args):
    if args.workers not in (1, 2, 3, 4) or args.epoch not in EPOCHS:
        raise ValueError("CPU workers1..4 and fixed M12/M16/M20 only")
    manifest = Path(args.manifest).resolve(strict=True)
    source_manifest = Path(args.source_val_manifest).resolve(strict=True)
    if original.sha256(source_manifest) != original.VAL_HASH:
        raise ValueError("original clean VAL manifest SHA differs")
    source_rows = [json.loads(s) for s in source_manifest.read_text().splitlines() if s]
    original.validate_manifest(source_rows)
    record = json.loads(manifest.read_text())
    counts = validate_manifest(record, source_rows, allow_pilot=args.allow_pilot)
    root = Path(record["artifact_root"]).resolve(strict=True)
    for entry in record["entries"]:
        path = (root / entry["artifact_path"]).resolve(strict=True)
        if root not in path.parents or not path.is_file():
            raise ValueError("artifact must be a file inside the materialized root")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("new output directory required; no overwrite/resume")
    for protected in (root, source_manifest.parent, checkpoint.parent):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("evaluation output must be separate from data/checkpoints")
    ids = [e["pair_id"] for e in record["entries"]]
    return dict(schema=SCHEMA, status="preflight_only", manifest=str(manifest),
        manifest_sha256=original.sha256(manifest), source_val_manifest=str(source_manifest),
        source_val_manifest_sha256=original.VAL_HASH, source_status=record["status"],
        source_protocol=record["protocol"], artifact_root=str(root),
        count=len(ids), by_recipe=counts, pair_ids=ids,
        pair_order_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        checkpoint=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
        epoch=args.epoch, output=str(output), workers=args.workers, cpu_threads_per_worker=1,
        device="cpu", batch_size=1, execute_requested=args.execute, gpu_lock_used=False)


def measure_losses(output, targets, config, pose_flags):
    """Original samplewise M loss, with S7's explicit damage eligibility mask."""
    import torch
    from experiments.rachel_n512_formal_30k.decoupled_samplewise_loss import samplewise_phase_terms
    pose = torch.as_tensor(pose_flags, dtype=torch.bool, device=targets[4].device)
    values = samplewise_phase_terms(output, targets, pose, config, "matcher")
    valid = output.training_valid
    matched = (targets[1] >= 0) & valid[:, None]
    return ({key: values[key].detach().cpu().tolist() for key in original.TERM_NAMES},
            matched.sum(1).detach().cpu().tolist(),
            (pose & targets[4] & valid).detach().cpu().tolist())


def evaluate_batch(model, batch, config, device, pose_flags):
    # Private globals: retain byte-identical forward/layout/metric code without
    # mutating the clean evaluator or the GT translation-valid target.
    context = dict(original.evaluate_batch.__globals__)
    context["measure_losses"] = lambda out, targets, cfg: measure_losses(out, targets, cfg, pose_flags)
    private = FunctionType(original.evaluate_batch.__code__, context,
                           original.evaluate_batch.__name__, original.evaluate_batch.__defaults__,
                           original.evaluate_batch.__closure__)
    return private(model, batch, config, device)


def load_entry(root, entry):
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
    path = (Path(root) / entry["artifact_path"]).resolve(strict=True)
    if Path(root).resolve() not in path.parents:
        raise ValueError("artifact outside materialized root")
    sample, report = load_sample(path)
    if (sample.pair_id != entry["pair_id"] or bool(sample.label) != entry["label"]
            or type(report.get("changed_pair")) is not bool
            or report["changed_pair"] != entry["changed_pair"]
            or type(report.get("pose_supervision_enabled")) is not bool
            or report["pose_supervision_enabled"] != (bool(sample.label) and not report["changed_pair"])):
        raise ValueError("archive ID/label/change/pose sidecar differs from manifest")
    for key in ("source_pair_id", "pose_supervision_enabled", "fallback_reason"):
        if key in entry and key in report and entry[key] != report[key]:
            raise ValueError("archive sidecar differs: " + key)
    if report.get("hard_val_recipe", entry["recipe"]) != entry["recipe"]:
        raise ValueError("archive recipe differs")
    return sample, report, path


def evaluate_chunk(work):
    # Executed in bounded spawn workers, each owning only one CPU frozen model.
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence.evaluate_matcher_simval_cpu import configure_cpu
    torch = configure_cpu(1)
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence.evaluate_continuation import load_endpoint
    from experiments.rachel_n512_formal_30k.train_joint_damage import state_digest
    from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import collate_rachel_pairs
    started = time.monotonic()
    model, config, identity = load_endpoint(work["checkpoint"], work["epoch"])
    rows, timings = [], []
    for ordinal, entry in work["entries"]:
        tick = time.monotonic()
        sample, report, path = load_entry(work["artifact_root"], entry)
        batch = collate_rachel_pairs([sample], contour_cap=512)
        row = evaluate_batch(model, batch, config, torch.device("cpu"),
                             [report["pose_supervision_enabled"]])[0]
        row.update(ordinal=ordinal, source_pair_id=entry["source_pair_id"], recipe=entry["recipe"],
            changed_pair=report["changed_pair"], pose_supervision_enabled=report["pose_supervision_enabled"],
            fallback_reason=report.get("fallback_reason"), artifact_path=entry["artifact_path"],
            artifact_sha256=original.sha256(path))
        if "source_family_overlap" in entry:
            row["source_family_overlap"] = entry["source_family_overlap"]
        rows.append(row)
        timings.append(dict(pair_id=row["pair_id"], seconds=time.monotonic()-tick))
    if state_digest(model) != identity["matcher_state_sha256"]:
        raise ValueError("evaluation changed frozen Matcher")
    return dict(rows=rows, timings=timings, model=identity, elapsed_seconds=time.monotonic()-started,
                runtime=dict(torch=str(torch.__version__), threads=torch.get_num_threads(), device="cpu"))


def summarize(rows):
    """Recipe and same-source paired summaries; source copies are not iid."""
    result = dict(all_entries=original.summarize(rows), by_recipe={}, paired_clean={})
    clean = {r["source_pair_id"]: r for r in rows if r["recipe"] == "clean"}
    for recipe in RECIPES:
        subset = [r for r in rows if r["recipe"] == recipe]
        if not subset:
            continue
        metrics = original.summarize(subset)
        changed = [r for r in subset if r["changed_pair"]]
        metrics.update(changed_count=len(changed),
            changed_positive_count=sum(r["label"] for r in changed),
            fallback_reasons=dict(Counter(r["fallback_reason"] for r in subset if r["fallback_reason"])),
            actually_changed=original.summarize(changed) if changed else None)
        result["by_recipe"][recipe] = metrics
        if recipe != "clean":
            references = [clean[r["source_pair_id"]] for r in subset]
            positives = [(a, b) for a, b in zip(references, subset) if b["label"]]
            deltas = [b["raw_translation_l2_px"] - a["raw_translation_l2_px"] for a, b in positives
                      if a["raw_translation_l2_px"] is not None and b["raw_translation_l2_px"] is not None]
            result["paired_clean"][recipe] = dict(count=len(subset),
                matched_clean=original.summarize(references), requested_variant=original.summarize(subset),
                positive_count=len(positives),
                positive_raw_layout20_transitions=dict(Counter(
                    ("correct" if a["raw_layout20_correct"] else "failed") + "_to_" +
                    ("correct" if b["raw_layout20_correct"] else "failed") for a, b in positives)),
                raw_translation_delta_px_mean_when_both_decoders_valid=original.mean_or_none(deltas),
                both_decoder_valid_positive_count=len(deltas))
    if any("source_family_overlap" in r for r in rows):
        disjoint = [r for r in rows if r.get("source_family_overlap") is False]
        changed = [r for r in disjoint if r["changed_pair"]]
        clean_disjoint = [r for r in disjoint if r["recipe"] == "clean"]
        result["optional_source_family_disjoint"] = dict(count=len(disjoint),
            overlapping_count=sum(r.get("source_family_overlap") is True for r in rows),
            unknown_count=sum("source_family_overlap" not in r for r in rows),
            clean=original.summarize(clean_disjoint) if clean_disjoint else None,
            actually_changed=original.summarize(changed) if changed else None,
            by_recipe={recipe: original.summarize([r for r in disjoint if r["recipe"] == recipe])
                       for recipe in RECIPES if any(r["recipe"] == recipe for r in disjoint)},
            interpretation="Optional conservative source-family stratum; all rows remain in primary summaries.")
    return result


def execute(args, plan):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("set CUDA_VISIBLE_DEVICES='' before this CPU-only invocation")
    if original.sha256(plan["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("manifest changed after preflight")
    from experiments.rachel_n512_formal_30k import decoupled_samplewise_loss
    from experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint import TOP2_CONFIG
    record = json.loads(Path(plan["manifest"]).read_text())
    entries = list(enumerate(record["entries"]))
    workers = min(plan["workers"], len(entries))
    work = [dict(checkpoint=plan["checkpoint"], epoch=plan["epoch"], artifact_root=plan["artifact_root"],
                 entries=entries[i*len(entries)//workers:(i+1)*len(entries)//workers]) for i in range(workers)]
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = dict(plan, status="running", pid=os.getpid(), completed_count=0,
        evaluator_sha256=original.sha256(__file__), loss_source_sha256=original.sha256(decoupled_samplewise_loss.__file__),
        original_evaluator_sha256=original.sha256(original.__file__), decoder_config=asdict(TOP2_CONFIG),
        weights_frozen=True, precision="fp32", training_performed=False,
        classifier_metrics_reported=False, thresholds_fitted=False, checkpoint_selection_performed=False,
        real_or_ood_read=False, cuda_visible_devices="", actual_cpu_workers=workers,
        loss_semantics="Canonical samplewise M objective; pose mask is explicit materialized report. Damaged positive translation loss is zero. Report all-pair and conditional denominators separately.",
        raw_layout_semantics="Production full_top2_mode against inherited source GT on all positive rows; pose-disabled positives remain in layout denominator; no classification gate.",
        limitations=["Equal-weight requested stress families are not the S7 TRAIN mixture.",
            "Clean/stress copies share source pairs; pooled entries are not independent examples.",
            "Paired comparisons use only the same source IDs for each requested recipe; fallbacks remain included.",
            "Pose-disabled differentiable translation error is null, not zero or a newly supervised target.",
            "CPU FP32 is not bitwise CUDA equivalent; these fixed diagnostics alone do not prove convergence."])
    original.save(output / "protocol.json", protocol)
    rows, timings, identity = [], [], None
    try:
        if workers == 1:
            results = [evaluate_chunk(work[0])]
        else:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                results = list(pool.map(evaluate_chunk, work))
        for chunk in results:
            if identity is not None and chunk["model"] != identity:
                raise ValueError("CPU workers loaded different Matcher identities")
            identity = chunk["model"]
            rows.extend(chunk["rows"]); timings.extend(chunk["timings"])
        if [r["pair_id"] for r in rows] != plan["pair_ids"]:
            raise ValueError("evaluated population/order differs from manifest")
        if original.sha256(plan["manifest"]) != plan["manifest_sha256"]:
            raise ValueError("source manifest changed during evaluation")
        with (output / "pair_metrics.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        original.save(output / "timing_pairs.json", dict(pairs=timings,
            elapsed_seconds=time.monotonic()-started, workers=workers, cpu_threads_per_worker=1))
        status = "pilot_complete" if plan["source_status"] == "pilot_complete" else "complete"
        summary = dict(schema=SCHEMA, status=status, count=len(rows), model=identity,
            manifest_sha256=plan["manifest_sha256"], metrics=summarize(rows),
            pair_metrics_sha256=original.sha256(output / "pair_metrics.jsonl"),
            diagnostic_only=True, training_eligible=False, classifier_metrics_reported=False,
            effective_matcher_objective_weights=dict(assignment_nll=.5, translation_smooth_l1=.5,
                sinkhorn_residual=.05, all_pair_bce=0.), limitations=protocol["limitations"])
        original.save(output / "summary.json", summary)
        protocol.update(status=status, model=identity, completed_count=len(rows),
            summary_sha256=original.sha256(output / "summary.json"))
    except BaseException as error:
        protocol.update(status="failed", completed_count=len(rows), error=repr(error))
        raise
    finally:
        protocol["elapsed_seconds"] = time.monotonic()-started
        original.save(output / "protocol.json", protocol)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source-val-manifest", type=Path,
        default=Path("/root/autodl-tmp/dataset_rachel_pairwise_n512_v1/pairs/val.jsonl"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--epoch", type=int, choices=EPOCHS, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-pilot", action="store_true", help="explicitly accept pilot_complete; never relabel it formal complete")
    p.add_argument("--execute", action="store_true")
    return p


def main():
    args = parser().parse_args()
    plan = preflight(args)
    if args.execute:
        execute(args, plan)
    else:
        print(json.dumps({k: v for k, v in plan.items() if k != "pair_ids"}, indent=2))


if __name__ == "__main__":
    main()
