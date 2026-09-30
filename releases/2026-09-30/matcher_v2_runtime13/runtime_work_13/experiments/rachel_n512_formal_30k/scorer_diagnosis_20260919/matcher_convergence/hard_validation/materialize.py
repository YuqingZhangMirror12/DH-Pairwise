"""Fixed clean/strong-damage SIMVAL stress controls, never a training dataset.

All 3000 existing VAL pairs get a clean copy and ONE assigned damage variant.
The four recipe groups have equal positive/negative requested counts; coupled
acceptance and fallbacks are retained, not filtered. Original TRAIN-only loaders
and their split guards remain unchanged. GT is used only to construct inherited
targets/partial cuts, never as a model input or a prediction-selection rule.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import save_sample
from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import strong_group, changed_report
from staging.pairwise_v0_2.pairwise_data.rachel_partial_seam_dataset import PartialSeamConfig, PartialSeamDataset
from staging.pairwise_v0_2.pairwise_data.rachel_guided_partial_dataset import GuidedPartialDataset
from staging.pairwise_v0_2.pairwise_data.rachel_curve_cut import OutlineBank, manuscript_family
from staging.pairwise_v0_2.pairwise_data.rachel_strong_weathering import StrongWeatheringConfig

SCHEMA = "s7-hard-simval-diagnostic/1"
SOURCE_SHA = "daa6ccdd7686e93ba91ddfb1452c987145c26898a1917d2ac7d3180e199a8af8"
TRAIN_SHA = "79a9e959f32ef9899116e299425d6350b17f6a04bd5070b5aa9a730319447c36"
SEED = 26092071
RECIPES = ("wave", "local", "seam_gaps", "partial_curve")
STATE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def source_rows(path):
    if sha(path) != SOURCE_SHA:
        raise ValueError("requires the unchanged original clean VAL3000")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if (len(rows) != 3000 or len({r["pair_id"] for r in rows}) != 3000
            or sum(r["label"] == 1 for r in rows) != 1500
            or any(r["label"] not in (0, 1) or r.get("split") != "val" for r in rows)):
        raise ValueError("VAL IDs, binary labels or source split differ")
    return rows


def group_plan(rows, seed=SEED):
    """Deterministic independent ordinal (+,-) coupling; NOT same-anchor mining."""
    positives = [i for i, r in enumerate(rows) if r["label"] == 1]
    negatives = [i for i, r in enumerate(rows) if r["label"] == 0]
    if not positives or len(positives) != len(negatives) or len(positives) % 4:
        raise ValueError("equal classes and positive count divisible by4 required")
    # This local RNG does not touch any training process RNG or source order.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(positives))
    recipe_for_group = {}
    for rank, group in enumerate(order):
        recipe_for_group[int(group)] = RECIPES[rank % 4]
    return [dict(group_index=i, source_indices=[a, b], recipe=recipe_for_group[i])
            for i, (a, b) in enumerate(zip(positives, negatives))]


def source_provenance(rows, train_manifest):
    """Record existing source aliases; do not remove or relabel validation rows."""
    if sha(train_manifest) != TRAIN_SHA:
        raise ValueError("requires the fixed S7 TRAIN source manifest")
    train = [entry["source_row"] for entry in json.loads(Path(train_manifest).read_text())["entries"]]
    sides = ("fragment_a", "fragment_b")
    train_lineages = {r[s]["split_unit_id"] for r in train for s in sides}
    val_lineages = {r[s]["split_unit_id"] for r in rows for s in sides}
    train_tokens = {r[s]["fragment_token"] for r in train for s in sides}
    val_tokens = {r[s]["fragment_token"] for r in rows for s in sides}
    if train_lineages & val_lineages or train_tokens & val_tokens:
        raise ValueError("exact file lineage or fragment overlaps TRAIN and VAL")
    families = {manuscript_family(n) for n in train_lineages}
    flags = {r["pair_id"]: any(manuscript_family(r[s]["split_unit_id"]) in families for s in sides)
             for r in rows}
    return flags, dict(train_manifest=str(train_manifest), train_manifest_sha256=TRAIN_SHA,
        exact_file_lineage_overlap=0, exact_fragment_token_overlap=0,
        train_lineage_count=len(train_lineages), val_lineage_count=len(val_lineages),
        family_alias_overlap=sorted(families & {manuscript_family(n) for n in val_lineages}),
        flagged_source_pairs=sum(flags.values()),
        flagged_positive_pairs=sum(flags[r["pair_id"]] for r in rows if r["label"]),
        flagged_negative_pairs=sum(flags[r["pair_id"]] for r in rows if not r["label"]),
        flag_definition="source_family_overlap uses conservative manuscript_family aliases; not proof of pixel duplication")


class ValidationCurvePair:
    """Geometry-only adapter, not a TRAIN dataset or disguised split.

    Reuses the existing S7 proposal/materialization numerical methods on an
    explicit pair of validation samples. It does not inherit dataset __init__,
    expose __getitem__, or register these samples with any training loader.
    """
    _rng = PartialSeamDataset._rng
    proposal = GuidedPartialDataset.proposal
    materialize = GuidedPartialDataset._materialize

    def __init__(self, positive, negative, bank, seed):
        if not positive.label or negative.label:
            raise ValueError("one original positive and one original negative required")
        self.base = (positive, negative)
        self._groups = ((0, 1),)
        self.pair_metadata = tuple((s.pair_id, int(s.label)) for s in self.base)
        self.bank, self.seed, self.epoch = bank, seed, 1
        self.config = PartialSeamConfig(probability=1., max_attempts=12)
        self.split = "val"


def initialize(options):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    STATE["threadpool"] = threadpool_limits(limits=1)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only materialization requires CUDA hidden")
    dataset = RachelPairDataset(options["dataset"], "val")
    if len(dataset) != 3000 or dataset.split != "val":
        raise ValueError("requires clean validation reader")
    STATE.update(dataset=dataset, bank=OutlineBank(options["outline_bank"]), options=options)


def validate_variant(old, new, report):
    if (old.pair_id != new.pair_id or old.label != new.label
            or old.fragment_a_token != new.fragment_a_token or old.fragment_b_token != new.fragment_b_token
            or old.translation_valid != new.translation_valid
            or not np.array_equal(old.translation_a_to_b_rc, new.translation_a_to_b_rc, equal_nan=True)
            or not np.array_equal(old.translation_a_to_b_xy_cartesian, new.translation_a_to_b_xy_cartesian, equal_nan=True)):
        raise ValueError("augmentation changed source identity/adjacency/original-frame GT")
    changed = False
    for side in "ab":
        before, after = (np.asarray(getattr(s, "mask_"+side), bool) for s in (old, new))
        if before.shape != after.shape or (after & ~before).any() or not after.any():
            raise ValueError("augmentation emptied/added material or rescaled/reframed mask")
        changed |= not np.array_equal(before, after)
        target = getattr(new, "target_"+side)
        valid = getattr(new, "contour_valid_"+side)
        opposite = getattr(new, "target_"+("b" if side == "a" else "a"))
        indices = np.flatnonzero(target >= 0)
        if target.shape != valid.shape or target.ndim != 1 or np.any(target < -2):
            raise ValueError("invalid inherited target shape/value")
        if len(indices) and (np.any(~valid[indices]) or np.any(target[indices] >= len(opposite))):
            raise ValueError("invalid inherited correspondence")
        if len(indices) and not np.array_equal(opposite[target[indices]], indices):
            raise ValueError("inherited correspondences are not reciprocal")
        if not new.label and np.any(target >= 0):
            raise ValueError("negative pair acquired positive correspondence targets")
    if (changed != bool(report["changed_pair"]) or bool(report["pose_supervision_enabled"])
            != (bool(new.label) and not changed)):
        raise ValueError("actual change/pose-supervision report differs")
    return changed


def materialize_group(plan):
    dataset, options = STATE["dataset"], STATE["options"]
    originals = tuple(dataset[i] for i in plan["source_indices"])
    recipe = plan["recipe"]
    if recipe == "partial_curve":
        samples, detail = ValidationCurvePair(*originals, STATE["bank"], options["seed"]).materialize(0)
        variants = tuple((new, changed_report(old, new, recipe, detail=detail,
            fallback_reason=None if detail["applied"] else detail["reason"]))
            for old, new in zip(originals, samples))
    else:
        variants = strong_group(*originals, mode=recipe, seed=options["seed"])
    result = []
    for member, (original, (variant, detail)) in enumerate(zip(originals, variants)):
        validate_variant(original, variant, detail)
        for name, sample, report in (("clean", original, changed_report(original, original, "clean")),
                                      (recipe, variant, detail)):
            source_id = sample.pair_id
            pair_id = "hardval-" + name + "::" + source_id
            sample = replace(sample, pair_id=pair_id)
            report = dict(report, pair_id=pair_id, source_pair_id=source_id, diagnostic_only=True,
                source_split="val", hard_val_recipe=name, assigned_recipe=recipe,
                source_family_overlap=options["family_flags"][source_id])
            relative = "samples/%04d_%d_%s.npz" % (plan["group_index"], member, name)
            save_sample(Path(options["output"])/relative, sample, report)
            areas = [int(np.count_nonzero(getattr(sample, "mask_"+s))) for s in "ab"]
            result.append(dict(pair_id=pair_id, source_pair_id=source_id, label=bool(sample.label),
                source_index=plan["source_indices"][member], group_index=plan["group_index"],
                recipe=name, assigned_recipe=recipe, changed_pair=bool(report["changed_pair"]),
                source_family_overlap=options["family_flags"][source_id],
                artifact_path=relative, pose_supervision_enabled=bool(report["pose_supervision_enabled"]),
                fallback_reason=report.get("fallback_reason"), matched_tokens=int((sample.target_a>=0).sum()),
                final_area_ratio=min(areas)/max(areas), final_areas_px=areas))
    save(Path(options["output"])/"groups"/("%04d.json" % plan["group_index"]),
         dict(plan=plan, entries=result))
    return result


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not 1 <= args.workers <= 4:
        raise ValueError("CPU only, at most4 bounded workers")
    rows = source_rows(Path(args.dataset)/"pairs/val.jsonl")
    family_flags, provenance = source_provenance(rows, Path(args.train_manifest).resolve())
    plans = group_plan(rows)
    if args.limit_groups is not None:
        if not 1 <= args.limit_groups <= 1500:
            raise ValueError("invalid pilot group count")
        # Interleave recipes for a pilot, independently of augmentation outcomes.
        by_recipe = [[p for p in plans if p["recipe"] == r] for r in RECIPES]
        plans = [p for group in zip(*by_recipe) for p in group][:args.limit_groups]
    output = Path(args.output).resolve()
    source_root = Path(args.dataset).resolve()
    bank_root = Path(args.outline_bank).resolve()
    if any(output == p or p in output.parents or output in p.parents for p in (source_root, bank_root)):
        raise ValueError("new diagnostic output must be outside original data/outline-bank")
    output.mkdir(parents=True, exist_ok=False)
    options = dict(dataset=str(source_root), outline_bank=str(bank_root), output=str(output), seed=SEED,
                   family_flags=family_flags)
    protocol = dict(source_val_manifest=str(source_root/"pairs/val.jsonl"), source_val_manifest_sha256=SOURCE_SHA,
        seed=SEED, diagnostic_only=True, training_eligible=False, threshold_fitting=False,
        model_selection=False, real_ood_used=False, source_population=3000,
        recipe_assignment="one of4 equally weighted stress recipes per source (+,-) group, plus matched clean copy",
        group_pairing="ordinal positive[i]/negative[i], not asserted same-anchor or same-page",
        recipe_percent={r:25 for r in RECIPES},
        limitations=["Stress-control mix is NOT the S7 TRAIN mixture: no Gen5 replacement or E1 2/4px branch.",
            "Same augmentation family is reused; this does not prove realism or unseen-damage generalization.",
            "File lineages/fragments are disjoint, but159 source pairs share conservative manuscript-family aliases with TRAIN; retain and stratify them.",
            "Coupled rejected augmentation stays as unchanged fallback; report requested and actually changed cohorts separately.",
            "Source-arc matches after wave damage need not satisfy zero-gap coordinate equality."],
        source_provenance=provenance, outline_bank=str(bank_root),
        outline_bank_sha256={name:sha(bank_root/name) for name in ("bank.json", "profiles.npz")},
        partial_config=asdict(PartialSeamConfig(probability=1., max_attempts=12)),
        strong_config=asdict(StrongWeatheringConfig(topology_connectivity=8, seam_short_gap_bridge_px=8.)),
        implementation_sha256=sha(__file__), cpu_workers=args.workers, pilot=args.limit_groups is not None)
    started = time.monotonic()
    save(output/"protocol.json", protocol)
    status = dict(status="running", pid=os.getpid(), groups=len(plans), completed_groups=0,
                  intended_rows=4*len(plans), diagnostic_only=True)
    save(output/"status.json", status)
    entries = []
    try:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize, initargs=(options,)) as pool:
            for group in pool.map(materialize_group, plans, chunksize=1):
                entries.extend(group)
                status["completed_groups"] += 1
                if status["completed_groups"] % 50 == 0:
                    save(output/"status.json", dict(status, elapsed_s=time.monotonic()-started))
        if len(entries) != 4*len(plans) or len({e["pair_id"] for e in entries}) != len(entries):
            raise ValueError("duplicate/incomplete output")
        counts = Counter((e["recipe"], bool(e["label"])) for e in entries)
        changed = Counter((e["recipe"], bool(e["label"])) for e in entries if e["changed_pair"])
        for recipe in {e["recipe"] for e in entries}:
            if counts[recipe,True] != counts[recipe,False] or changed[recipe,True] != changed[recipe,False]:
                raise ValueError("coupled requested/applied class counts differ")
        summary = dict(count=len(entries), source_count=len(entries)//2,
            by_recipe={r:dict(positive=counts[r,True],negative=counts[r,False],
                             changed_positive=changed[r,True],changed_negative=changed[r,False])
                       for r in ("clean",)+RECIPES},
            fallback_reasons=dict(Counter(e["fallback_reason"] for e in entries if e["fallback_reason"])),
            elapsed_s=time.monotonic()-started)
        final_status = "pilot_complete" if args.limit_groups is not None else "complete"
        manifest = dict(schema=SCHEMA, split="val", status=final_status, artifact_root=str(output),
                        entries=entries, protocol=protocol, summary=summary)
        save(output/"manifest.json", manifest)
        save(output/"summary.json", summary)
        save(output/"status.json", dict(status, status=final_status, **summary,
             completed_groups=len(plans), manifest=str(output/"manifest.json")))
        return summary
    except BaseException as error:
        save(output/"status.json", dict(status, status="failed", error=repr(error),
             completed_rows=len(entries), elapsed_s=time.monotonic()-started))
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="/root/autodl-tmp/dataset_rachel_pairwise_n512_v1")
    p.add_argument("--outline-bank", default="/root/autodl-tmp/rachel_curve_hardneg_20260910_001/data/outline_bank_v2")
    p.add_argument("--train-manifest", default="/root/autodl-tmp/rachel_score_design_20260913_001/s6_s7_20260915/preparation/data/train_s7_24k.json")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, choices=(1,2,3,4), default=2)
    p.add_argument("--limit-groups", type=int)
    return p


if __name__=="__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
