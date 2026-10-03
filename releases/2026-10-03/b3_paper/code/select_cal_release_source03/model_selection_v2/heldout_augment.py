"""New SELECT/CAL orchestration around frozen augmentation mathematics.

This module must be loaded beside the frozen curriculum runtime, never into a
running trainer. It does not generate TRAIN or TEST, tune a model, relax pixel
gates, repair targets, or silently substitute a v14 fallback for a v17 sample.
The frozen v14 intermediate is the common reference for all three stages.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

PREFIX = "experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919"
ROLES = ("cal", "select")
STAGES = ("v17_filtered", "v17.5", "v18")
STATE = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def tensor_identity(sample, *, include_supervision=False):
    """Content identity excludes arbitrary archive IDs/reports and NPZ headers."""
    import numpy as np
    names = [stem+side for stem in ("mask_", "coarse_mask_", "points_rc_", "contour_valid_")
             for side in "ab"]
    if include_supervision:
        names += ["target_a", "target_b", "translation_a_to_b_rc", "translation_valid", "label"]
    result = hashlib.sha256()
    for name in names:
        value = np.asarray(getattr(sample, name))
        result.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        result.update(np.ascontiguousarray(value).tobytes())
    return result.hexdigest()


def validate_task(task):
    if task["role"] not in ROLES or task["stage"] not in STAGES:
        raise ValueError("new heldout roles/stages only")
    if task["generator"] not in ("Gen2", "Gen3", "Gen4", "Gen5"):
        raise ValueError("explicit base generator required")
    if task["size_class"] not in ("smaller", "larger"):
        raise ValueError("explicit frozen size class required")
    if task["mode"] not in ("one", "both") or not 1 <= task["k"] <= 4:
        raise ValueError("invalid endpoint mode or notch count")
    if not .25 <= task["trim_target"] <= .40:
        raise ValueError("curriculum shortening range changed")
    if task["slot"] < 0 or task["master_seed"] < 0:
        raise ValueError("invalid slot/seed")


def rng_key(task, attempt):
    """All roles and provenance enter the random domain, not just a slot."""
    validate_task(task)
    return ("mixed-sim-heldout/1", task["master_seed"], task["role"],
            task["stage"], task["generator"], task["slot"],
            tuple(task["base_pair_ids"]), attempt)


def verify_commit(path, task, plan_sha):
    receipt = json.loads(Path(path).read_text())
    if receipt["task"] != task or receipt["plan_sha256"] != plan_sha:
        raise ValueError("resume binding changed")
    if receipt["status"] != "committed" or len(receipt["records"]) != 2:
        raise ValueError("not a full admitted pair group")
    for row in receipt["records"]:
        for path_key, hash_key in (("sample_path", "sample_sha256"),
                                   ("proof_path", "proof_sha256"),
                                   ("target_metadata", "target_metadata_sha256"),
                                   ("unaugmented_sample_path", "unaugmented_sample_sha256")):
            if digest(row[path_key]) != row[hash_key]:
                raise ValueError("committed output changed: " + path_key)
        if row.get("latent_seam_artifact") and digest(row["latent_seam_artifact"]) != row["latent_sha256"]:
            raise ValueError("committed latent proof changed")
        if digest(row["baseline_sample_path"]) != row["baseline_sample_sha256"]:
            raise ValueError("committed baseline sample changed")
        if digest(row["baseline_group_path"]) != row["baseline_group_sha256"]:
            raise ValueError("committed baseline provenance changed")
    return receipt


def process(task):
    """Generate one fixed base/recipe pair, then independently replay pixels.

    Rejections are data, not permission to replace the selected base or change
    a recipe. A caller may register a new deterministic attempt domain before
    continuing, but cannot count a rejected group toward its quota.
    """
    validate_task(task)
    if task["role"] != STATE["role"]:
        raise ValueError("worker role mismatch")
    import numpy as np
    from staging.pairwise_v0_2.pairwise_data.rachel_s7_dataset import changed_report
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import save_sample, load_sample
    common = importlib.import_module(PREFIX + ".s7_compound_v1.geometry")
    targets = importlib.import_module(PREFIX + ".seam_context_v3.targets")
    frozen = importlib.import_module(PREFIX + ".aggressive_data_full_v17.generate")
    old = task["stage"] == "v17_filtered"
    package = "aggressive_data_full_v17" if old else "curriculum_data_v18_v19"
    geometry = importlib.import_module(PREFIX + "." + package + ".geometry")
    audit = importlib.import_module(PREFIX + "." + package + ".audit")
    source = STATE["source"]
    config = STATE["config"]
    out = Path(config["out"]) / task["role"] / task["stage"]
    commit = out / "groups" / f'{task["slot"]:05d}.json'
    if commit.exists():
        return verify_commit(commit, task, config["generation_plan_sha256"])
    start = time.monotonic()
    pair = tuple(source.baseline(task["slot"], ordinal) for ordinal in (0, 1))
    if [p["entry"]["source_pair_id"] for p in pair] != task["base_pair_ids"]:
        raise ValueError("actual base pairs differ from preregistration")
    if any(p["entry"]["corrosion_recipe"] != task["recipe"] for p in pair):
        raise ValueError("recipe differs from common baseline")
    result = None
    rejections = []
    for attempt in range(config.get("geometry_attempts", 24)):
        rng = common.rng_for(*rng_key(task, attempt))
        mode = task["mode"] if attempt % 2 == 0 else ("both" if task["mode"] == "one" else "one")
        try:
            arguments = (pair, rng, mode, source.STATE["bank"], task["size_class"], task["k"])
            if old:
                result = geometry.augment(*arguments)
            else:
                specs = importlib.import_module(PREFIX + ".curriculum_data_v18_v19.spec").SPECS
                result = geometry.augment(*arguments, specs[task["stage"]], task["trim_target"])
            break
        except ValueError as error:
            rejections.append(dict(attempt=attempt, mode=mode, reason=str(error)))
    if result is None:
        receipt = dict(status="rejected", task=task, reasons=dict(Counter(r["reason"] for r in rejections)),
                       attempts=rejections, seconds=time.monotonic()-start,
                       v14_fallback=False, counts_toward_final=False)
        save_json(out / "rejections" / commit.name, receipt)
        return receipt
    final, details, fields, gap, trimmed, primary = result
    records = []
    for ordinal, (row, sample, detail) in enumerate(zip(pair, final, details)):
        entry, base = row["entry"], row["original"]
        pid = f'mixsim2-{task["role"]}-{task["stage"]}-{task["generator"]}-{task["slot"]:05d}-{ordinal}'
        sample = replace(sample, pair_id=pid, fragment_a_token=sample.fragment_a_token+"@"+pid,
                         fragment_b_token=sample.fragment_b_token+"@"+pid)
        partial = row["old_report"]["compound"]["partial"]
        partial_applied = task["recipe"] == "partial" if old else detail["partial_crop_applied"]
        report = changed_report(base, sample, task["recipe"])
        report.update(schema_version="mixed-sim-heldout-archive/1", data_role=task["role"],
                      recipe=task["recipe"], source_pair_id=entry["source_pair_id"],
                      base_v14_pair_id=entry["pair_id"], paired_review=detail, v14_fallback=False,
                      not_full_training_dataset=True,
                      compound=dict(recipe=task["recipe"], damage=detail["primary_damage"],
                                    partial=partial if partial_applied else None),
                      pair_shared_scale=deepcopy(row["old_report"]["pair_shared_scale"]))
        for side in "ab":
            report["side_"+side].update(detail["primary_damage"].get(side, {}))
        path = out / "samples" / f'{task["slot"]:05d}_{ordinal}.npz'
        save_sample(path, sample, report)
        loaded, _ = load_sample(path)
        for key in ("mask_a", "mask_b", "target_a", "target_b", "points_rc_a", "points_rc_b"):
            if not np.array_equal(getattr(loaded, key), getattr(sample, key)):
                raise ValueError("roundtrip altered " + key)
        clean_path = Path(config["out"]) / task["role"] / "unaugmented" / path.name
        if not clean_path.exists():
            save_sample(clean_path, base, changed_report(base, base, "clean"))
        else:
            saved_base, _ = load_sample(clean_path)
            for key in ("mask_a", "mask_b", "target_a", "target_b"):
                if not np.array_equal(getattr(saved_base, key), getattr(base, key)):
                    raise ValueError("common preaugmentation reference changed")
        arrays = {}
        for name, value in (("fragment", base), ("trim", trimmed[ordinal]),
                            ("primary", primary[ordinal]), ("final", sample)):
            for side in "ab":
                arrays["packed_"+name+"_"+side] = np.packbits(getattr(value,"mask_"+side)[0].astype(bool), axis=1)
        for kind, values in fields[ordinal].items():
            arrays.update({kind+"_"+key: value for key, value in values.items()})
        if ordinal == 0:
            arrays.update({"gap_"+key:value for key,value in gap.items()})
        proof = out / "proof" / path.name
        frozen.atomic_npz(proof, **arrays)
        target = out / "targets" / path.name
        frozen.atomic_npz(target, **targets.known_gap_links(sample, base, report))
        latent = None
        if ordinal == 0:
            latent = out / "latent" / path.name
            frozen.atomic_npz(latent, **gap)
        donors = [detail["trim"]["donor"]["lineage"]] if detail["trim"].get("applied", True) else []
        if partial_applied and partial and partial.get("donor_lineage"):
            donors.append(partial["donor_lineage"])
        record = dict(pair_id=pid, id=pid, label=bool(sample.label), data_role=task["role"],
                      stage=task["stage"], generator=task["generator"], version=task["stage"],
                      recipe=task["recipe"], corrosion_recipe=task["recipe"],
                      sample_path=str(path), sample_sha256=digest(path),
                      proof_path=str(proof), proof_sha256=digest(proof),
                      target_metadata=str(target), target_metadata_sha256=digest(target),
                      latent_seam_artifact=str(latent) if latent else None,
                      latent_sha256=digest(latent) if latent else None,
                      unaugmented_sample_path=str(clean_path), unaugmented_sample_sha256=digest(clean_path),
                      model_tensors_sha256=tensor_identity(sample),
                      supervised_tensors_sha256=tensor_identity(sample,include_supervision=True),
                      unaugmented_model_tensors_sha256=tensor_identity(base),
                      baseline_sample_path=str(source.STATE["root"]/entry["artifact_path"]),
                      baseline_sample_sha256=digest(source.STATE["root"]/entry["artifact_path"]),
                      baseline_group_path=str(source.STATE["root"]/"groups"/f'{task["slot"]:05d}.json'),
                      baseline_group_sha256=digest(source.STATE["root"]/"groups"/f'{task["slot"]:05d}.json'),
                      baseline_slot=task["slot"], baseline_ordinal=ordinal,
                      source_pair_id=entry["source_pair_id"], source_row=entry["source_row"],
                      source_root=entry["source_root"], negative_kind=entry["negative_kind"],
                      offline_paired_mirror=entry["offline_paired_mirror"],
                      augmentation_donor_sources=donors, detail=detail, requested_gap_count=task["k"],
                      inherited_correspondences=int((sample.target_a >= 0).sum()),
                      partial_applied=partial_applied, v14_fallback=False)
        records.append(record)
    # The original auditor uses actual NPZ pixels, same-fold bank, original
    # source reconstruction and target loader. Never replace it with counters.
    audit_rows = [audit.audit_record(record, source.STATE["root"]) for record in records]
    receipt = dict(status="committed", task=task, plan_sha256=config["generation_plan_sha256"],
                   records=records, audit_rows=audit_rows, rejections=rejections,
                   seconds=time.monotonic()-start, source_replaced=False, v14_fallback=False)
    save_json(commit, receipt)
    return receipt
