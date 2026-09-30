"""Materialize one fixed balanced S7 TRAIN population; never read held-out data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import (
    SCHEMA, MaterializedWeatheredDataset, save_sample)
from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import (
    SCHEMA as RECIPE_SCHEMA, RECIPE_PERCENT, recipe_schedule, changed_report, strong_group)
from staging.pairwise_v0_2.pairwise_data.rachel_guided_partial_dataset import GuidedPartialDataset
from staging.pairwise_v0_2.pairwise_data.rachel_partial_seam_dataset import PartialSeamConfig
from staging.pairwise_v0_2.pairwise_data.rachel_strong_weathering import StrongWeatheringConfig
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import make_training_dataset


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                   allow_nan=False, indent=2) + "\n")
    os.replace(temporary, path)


STATE = {}


def side_statistics(side):
    """Retain measured strong depth separately from older requested profiles."""
    applied = bool(side.get("effective_applied", side.get("applied", False)))
    measured = side.get("applied_max_depth_px")
    legacy_profile = side.get("max_sampled_depth_px", side.get("max_depth_px"))
    actual = float(measured) if applied and measured is not None else (0. if not applied else None)
    return dict(applied=applied, removed_fraction=side.get("removed_fraction"),
        applied_max_depth_px=actual,
        max_depth_px=actual if actual is not None else legacy_profile,
        depth_statistic=("measured_removed_pixel_inward_distance" if measured is not None else
                         "identity_zero" if not applied else "legacy_profile_depth_not_measured_displacement"),
        requested_depth_range_px=side.get("requested_depth_range_px"),
        requested_peak_depths_px=side.get("requested_peak_depths_px"),
        actual_gap_count=int(side.get("gap_k_applied", side.get("actual_gap_count", 0)) or 0) if applied else 0,
        gap_k_requested=side.get("gap_k_requested", 0),
        gap_k_applied=int(side.get("gap_k_applied", 0)) if applied else 0,
        local_notch_count_requested=side.get("local_notch_count_requested", 0),
        local_notch_count_applied=int(side.get("local_notch_count_applied", 0)) if applied else 0,
        component_count=side.get("component_count"))


def initialize(options):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    fixed = MaterializedWeatheredDataset(options["reference_manifest"])
    source_manifest = fixed.protocol["source_manifest"]
    clean, _ = make_training_dataset(options["dataset_root"], source_manifest)
    metadata = [(e["pair_id"], int(e["label"])) for e in fixed.entries]
    if len(clean) != len(fixed):
        raise ValueError("clean/fixed TRAIN population mismatch")
    guided = GuidedPartialDataset(clean, pair_metadata=metadata,
        bank=options["outline_bank"], seed=options["seed"], epoch=1,
        config=PartialSeamConfig(probability=1., max_attempts=12))
    protected = [i for i, (p, _) in enumerate(guided._groups)
                 if fixed.entries[p].get("source_stratum") == "union_positive_tiny"]
    recipes = recipe_schedule(len(guided._groups), options["seed"], protected_groups=protected)
    gen5 = None
    pool_indices = None
    if not options["skip_gen5"]:
        gen5 = MaterializedWeatheredDataset(options["gen5_manifest"])
        pool_indices = {label: [i for i, e in enumerate(gen5.entries) if bool(e["label"]) == label]
                        for label in (True, False)}
        count = recipes.count("gen5_partition")
        if any(len(indices) < count for indices in pool_indices.values()):
            raise ValueError("Gen5 pool is insufficient; cannot replace with unrelated data")
    gen5_rank = {g: rank for rank, g in enumerate(i for i, r in enumerate(recipes)
                                                if r == "gen5_partition")}
    STATE.update(options=options, fixed=fixed, clean=clean, guided=guided,
                 recipes=recipes, gen5=gen5, pool_indices=pool_indices, gen5_rank=gen5_rank)


def materialize_group(group_index):
    state, options = STATE, STATE["options"]
    root = Path(options["output_root"])
    receipt = root / "groups" / (str(group_index).zfill(5) + ".json")
    recipe = state["recipes"][group_index]
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved.get("group_index") != group_index or saved.get("recipe") != recipe:
            raise ValueError("saved group receipt does not match this fixed recipe slot")
        if saved.get("pilot_skipped"):
            if recipe != "gen5_partition" or not options["skip_gen5"]:
                raise ValueError("pilot-skipped Gen5 group cannot count as a completed formal group")
        elif len(saved.get("entries", [])) != 2 or any(
                not (root / entry["artifact_path"]).is_file() for entry in saved.get("entries", [])):
            raise ValueError("saved group receipt has missing sample artifacts; refusing false resume completion")
        return saved
    indices = state["guided"]._groups[group_index]
    entries = [state["fixed"].entries[i] for i in indices]
    if recipe == "gen5_partition":
        if options["skip_gen5"]:
            result = dict(group_index=group_index, recipe=recipe, pilot_skipped=True, entries=[])
            write_json(receipt, result)
            return result
        rank = state["gen5_rank"][group_index]
        selected = [state["pool_indices"][label][rank] for label in (True, False)]
        rows = [state["gen5"][i] for i in selected]
        entries = [state["gen5"].entries[i] for i in selected]
    elif recipe == "reference_e1":
        rows = [state["fixed"][i] for i in indices]
    else:
        originals = [state["clean"][i] for i in indices]
        if any(s.pair_id != e["pair_id"] or bool(s.label) != bool(e["label"])
               for s, e in zip(originals, entries)):
            raise ValueError("clean/fixed order or label mismatch")
        if recipe == "partial_curve":
            samples, detail = state["guided"]._get_group(group_index)
            rows = [(sample, changed_report(old, sample, recipe, detail=detail,
                      fallback_reason=None if detail["applied"] else detail["reason"]))
                    for old, sample in zip(originals, samples)]
        else:
            rows = strong_group(*originals, mode=recipe, seed=options["seed"])
    output, summaries = [], []
    for ordinal, ((sample, report), source) in enumerate(zip(rows, entries)):
        report = dict(report)
        report["s7_assignment"] = dict(schema=RECIPE_SCHEMA, recipe=recipe,
            seed=options["seed"], group_index=group_index,
            original_slot_pair_id=state["fixed"].entries[indices[ordinal]]["pair_id"])
        relative = "samples/%05d_%d.npz" % (group_index, ordinal)
        save_sample(root / relative, sample, report)
        entry = dict(source, artifact_path=relative, pair_id=sample.pair_id,
            label=bool(sample.label), changed_pair=bool(report["changed_pair"]),
            s7_recipe=recipe, s7_group_index=group_index)
        entry["source_row"] = dict(source["source_row"], pair_id=sample.pair_id,
                                   label=bool(sample.label), split="train")
        output.append(entry)
        summaries.append(dict(label=bool(sample.label), changed=bool(report["changed_pair"]),
            fallback_reason=report.get("fallback_reason"),
            matched_tokens=int((sample.target_a >= 0).sum()),
            side_a=side_statistics(report.get("side_a", {})),
            side_b=side_statistics(report.get("side_b", {}))))
    result = dict(group_index=group_index, recipe=recipe, entries=output, summaries=summaries)
    write_json(receipt, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset-root", "reference-manifest", "outline-bank", "output-root"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--gen5-manifest")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260915)
    parser.add_argument("--group-limit", type=int, help="pilot only; never emits complete readiness")
    parser.add_argument("--skip-gen5", action="store_true", help="pilot only")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("use 1..4 CPU workers so GPU training is not starved")
    if args.skip_gen5 and not args.group_limit:
        parser.error("Gen5 can only be skipped in an explicitly bounded pilot")
    if not args.skip_gen5 and not args.gen5_manifest:
        parser.error("formal materialization requires the real Gen5 pool")
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "producer.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    started = time.time()
    options = vars(args)
    options["output_root"] = str(root)
    protocol = dict(schema_version=RECIPE_SCHEMA, options=options, recipe_percent=RECIPE_PERCENT,
        reference_manifest_sha256=hashlib.sha256(Path(args.reference_manifest).read_bytes()).hexdigest(),
        clean_source_only_for_new_damage=True, frozen_real_val_test=True,
        inherited_source_arc_targets=True, artificial_notches="ignore, not positive",
        damaged_pose_loss="disabled, identical to E1", dimensions_px=[800,800], contour_cap=512,
        strong_weathering_config=asdict(StrongWeatheringConfig(topology_connectivity=8, seam_short_gap_bridge_px=8.)),
        source_arc_short_gap_bridge="placement only; never additional positive targets",
        group_coupled_acceptance=True, no_weakening_when_infeasible=True)
    try:
        protocol_path = root / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("output exists with a different frozen recipe")
        write_json(protocol_path, protocol)
        initialize(options)
        count = len(STATE["recipes"])
        if count != 12000:
            raise ValueError("S7 formal reference must be 24K balanced TRAIN")
        if args.group_limit:
            count = min(count, args.group_limit)
        status = dict(status="running", producer_pid=os.getpid(), started_at=started,
                      completed_groups=0, planned_groups=count, pilot=bool(args.group_limit))
        write_json(root / "status.json", status)
        records = []
        if args.workers == 1:
            iterator = map(materialize_group, range(count))
            executor = None
        else:
            executor = ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(options,))
            iterator = executor.map(materialize_group, range(count), chunksize=1)
        try:
            for record in iterator:
                records.append(record)
                if len(records) % 50 == 0 or len(records) == count:
                    status.update(completed_groups=len(records), elapsed_seconds=time.time()-started)
                    write_json(root / "status.json", status)
                    print(json.dumps(status), flush=True)
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)
        entries = [e for record in records for e in record["entries"]]
        ids = [e["pair_id"] for e in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("S7 population contains duplicate pair IDs")
        counts, applied, reasons = Counter(), Counter(), Counter()
        for record in records:
            for s in record.get("summaries", []):
                key = record["recipe"] + (":positive" if s["label"] else ":negative")
                counts[key] += 1
                applied[key] += s["changed"]
                if s["fallback_reason"]:
                    reasons[record["recipe"] + ":" + s["fallback_reason"]] += 1
        positive = sum(bool(e["label"]) for e in entries)
        stats = dict(sample_count=len(entries), positive_count=positive,
            negative_count=len(entries)-positive, requested_counts=dict(counts),
            changed_counts=dict(applied), fallback_reasons=dict(reasons),
            elapsed_seconds=time.time()-started)
        manifest = root / ("pilot_train.json" if args.group_limit else "train_s7_24k.json")
        write_json(manifest, dict(schema_version=SCHEMA, split="train", artifact_root=str(root),
            entries=entries, protocol=protocol, stats=stats))
        if not args.group_limit and (len(entries) != 24000 or positive != 12000):
            raise ValueError("cannot publish incomplete or unbalanced S7 TRAIN")
        write_json(root / "summary.json", stats)
        write_json(root / "status.json", dict(status="pilot_complete" if args.group_limit else "complete",
            producer_pid=os.getpid(), manifest=str(manifest), **stats))
        print(json.dumps(dict(manifest=str(manifest), **stats)), flush=True)
    except BaseException as exc:
        write_json(root / "status.json", dict(status="failed", producer_pid=os.getpid(),
            error=type(exc).__name__ + ": " + str(exc), elapsed_seconds=time.time()-started))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
