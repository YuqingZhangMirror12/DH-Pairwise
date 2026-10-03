"""Bounded, CPU-only new heldout build; no training subprocesses.

Run as a standalone module next to heldout_augment/heldout_plan with the frozen
curriculum source root on PYTHONPATH. Existing runtimes are imported read-only.
Only this new build directory receives intermediate samples and receipts.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

try:
    from . import heldout_augment as augmentation
except ImportError:
    import heldout_augment as augmentation

PREFIX = augmentation.PREFIX


def read(path):
    return json.loads(Path(path).read_text())


def bind_loaded_sources():
    rows = {}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and (name.startswith(PREFIX) or name.startswith("staging.pairwise_v0_2")):
            path = str(Path(path).resolve())
            if Path(path).suffix == ".py":
                rows[path] = augmentation.digest(path)
    return rows


def check_sources(rows):
    for path, sha in rows.items():
        if augmentation.digest(path) != sha:
            raise ValueError("frozen source changed during build: " + path)


def validate_loaded_inventory(bound, inventory):
    root = Path(inventory["root"]).resolve()
    for path, actual in bound.items():
        target = Path(path).resolve()
        if not target.is_relative_to(root):
            raise ValueError("native dependency outside bound frozen runtime: " + path)
        if inventory["files"].get(str(target.relative_to(root))) != actual:
            raise ValueError("native dependency differs from preregistered bytes: " + path)


def validate_tasks(plan):
    expected = {(role, gen, stage, j) for role in ("cal", "select")
                for gen in ("Gen2", "Gen3", "Gen4", "Gen5")
                for stage in augmentation.STAGES for j in range(120)}
    buckets = defaultdict(list)
    for task in plan["tasks"]:
        augmentation.validate_task(task)
        buckets[(task["role"], task["generator"], task["stage"], task["quota_slot"])].append(task)
    if set(buckets) != expected or any(sorted(t["reserve_index"] for t in tasks) != [0,1,2]
                                       for tasks in buckets.values()):
        raise ValueError("all two-fold Gen/stage/120 quotas and three registered reserves required")
    if (plan["desired_pairs_per_role"], plan["desired_pairs_per_curriculum_stage"],
        plan["desired_strict_pairs_per_role"]) != (3200,960,320):
        raise ValueError("fixed population quotas changed")


def expected_baseline_exhaustion(error, task):
    """Only original bounded-geometry exhaustion, never OOM or arbitrary errors."""
    try:
        value = json.loads(str(error))
    except (ValueError, TypeError):
        return False
    keys = {"slot", "recipe", "partial", "negative", "failures", "damage_attempts",
            "source_draws", "static_length_sources_excluded", "damage_attempt_budget", "source_pool_exhausted"}
    return (isinstance(value, dict) and set(value) == keys and value["slot"] == task["slot"]
            and value["recipe"] == task["recipe"] and value["negative"] == task["base_pair_ids"][1]
            and isinstance(value["failures"], dict) and value["damage_attempt_budget"] == 1024
            and 0 <= value["damage_attempts"] <= 1024)


def quota_groups(tasks, role, pilot=False):
    buckets = defaultdict(list)
    for task in tasks:
        if task["role"] == role:
            buckets[(task["generator"],task["quota_slot"],task["stage"])].append(task)
    keys = sorted(buckets)
    if pilot:
        chosen = {}
        for key in keys:
            task = buckets[key][0]
            # Representative uncorroded/strong compound gates across every
            # generator, stage and side. Full build audits every remaining row.
            if task["recipe"] not in ("clean", "gaps_weak"):
                continue
            cell = (task["generator"],task["stage"],task["recipe"],task["size_class"])
            chosen.setdefault(cell,key)
        keys = sorted(chosen.values())
    return [(key, sorted(buckets[key],key=lambda t:t["reserve_index"])) for key in keys]


def run_role(args):
    root, role, pilot = args
    root = Path(root).resolve()
    plan_path = root/"plan"/"generation_plan.json"
    plan = read(plan_path)
    validate_tasks(plan)
    inventory = read(root/"frozen_source_inventory.json")
    if augmentation.digest(plan["source_plan_path"]) != plan["source_plan_sha256"]:
        raise ValueError("source plan changed")
    source_plan = read(plan["source_plan_path"])
    out = root/"augmented"
    receipt_root = root/("pilot_receipts" if pilot else "build_receipts")/role
    receipt_root.mkdir(parents=True,exist_ok=True)
    complete = receipt_root/"complete.json"
    if complete.exists():
        raise ValueError("preserve prior completed role; inspect its receipt instead")
    start = time.time()
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        import torch
        import cv2
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        heldout = importlib.import_module(PREFIX+".s7_consensus_v1.heldout_v14")
        material = importlib.import_module(PREFIX+".s7_balanced_v2.materialize")
        base_audit = importlib.import_module(PREFIX+".s7_balanced_v2.audit_layered")
        source = importlib.import_module(PREFIX+".aggressive_data_full_v17.source")
        baseline_root = root/"baseline"/role
        baseline_root.mkdir(parents=True,exist_ok=True)
        options = dict(sources=plan["source_plan_path"],split=role,out=str(baseline_root),
                       seed=plan["master_seed"]+(10001 if role=="cal" else 20002),attempts=1024)
        heldout.initialize(options)
        config=dict(out=str(out),generation_plan_sha256=augmentation.digest(plan_path),geometry_attempts=24)
        source.STATE.clear()
        source.baseline.cache_clear()
        source.STATE.update(config=config,split=role,root=baseline_root,groups={},
                            bank=material.STATE["bank"],
                            negative={e["pair_id"]:e for e in material.STATE["negative_plan"]})
        augmentation.STATE.clear()
        augmentation.STATE.update(config=config,role=role,source=source)
        bound = bind_loaded_sources()
        validate_loaded_inventory(bound, inventory)
        completed, failures, base_receipts = [], [], {}
        seen_inputs = set()
        quotas = quota_groups(plan["tasks"],role,pilot)
        for key, tasks in quotas:
            accepted = None
            for task in tasks:
                slot = task["slot"]
                if slot not in source.STATE["groups"]:
                    # Independent frozen v14 materialization and pixel replay.
                    # A rejected baseline is retained as a rejection record and
                    # cannot be silently used as an accepted final example.
                    try:
                        baseline = heldout.slot(slot)
                    except RuntimeError as error:
                        if not expected_baseline_exhaustion(error, task):
                            raise
                        failures.append(dict(task=task,phase="baseline_geometry_exhausted",error=str(error)))
                        continue
                    # Independent audit failures are never ordinary geometry
                    # exclusions: stop and retain evidence instead of filtering
                    # a corrupt proof/target out of the population silently.
                    audits = [base_audit.one((str(baseline_root),entry)) for entry in baseline["entries"]]
                    base_receipts[str(slot)] = dict(audit_rows=audits,group_sha256=augmentation.digest(baseline_root/"groups"/f"{slot:05d}.json"))
                    source.STATE["groups"][slot] = {i:e for i,e in enumerate(baseline["entries"])}
                result = augmentation.process(task)
                if result["status"] == "committed":
                    hashes = [r["model_tensors_sha256"] for r in result["records"]]
                    if len(set(hashes)) != len(hashes) or set(hashes) & seen_inputs:
                        failures.append(dict(task=task, phase="duplicate_model_input",
                                             error="archived audit retained; not admitted to release"))
                        continue
                    seen_inputs.update(hashes)
                    accepted = result
                    break
                failures.append(dict(task=task,phase="augmentation",error=result.get("reasons")))
            if accepted is not None:
                completed.append(accepted)
            else:
                failures.append(dict(quota=list(key),phase="quota_exhausted",error="all registered candidates rejected"))
            progress=dict(status="running",role=role,pid=os.getpid(),
                finished_quotas=len(completed)+sum(f["phase"]=="quota_exhausted" for f in failures),
                planned_quotas=len(quotas),admitted_pairs=2*len(completed),elapsed_seconds=time.time()-start)
            augmentation.save_json(receipt_root/"status.json",progress)
            if progress["finished_quotas"] % 20 == 0 or progress["finished_quotas"] == len(quotas):
                print(json.dumps(dict(progress,pilot=pilot)),flush=True)
        later = bind_loaded_sources()
        for path, value in later.items():
            if path in bound and bound[path] != value:
                raise ValueError("source binding changed during generation: " + path)
            bound[path] = value
        check_sources(bound)
        validate_loaded_inventory(bound, inventory)
        records=[record for group in completed for record in group["records"]]
        exhausted=[f for f in failures if f["phase"]=="quota_exhausted"]
        if not pilot and not exhausted:
            if len(records) != 2880 or Counter((r["stage"], r["generator"], r["label"]) for r in records) != Counter({
                (stage,gen,label):120 for stage in augmentation.STAGES
                for gen in ("Gen2","Gen3","Gen4","Gen5") for label in (False,True)}):
                raise ValueError("actual complete population is not the registered 2880 rows")
        receipt=dict(schema="mixed-sim-heldout-role-build/1",role=role,
                     status="pilot_passed" if pilot and not exhausted else "complete" if not exhausted else "shortfall",
                     pilot_only=pilot, planned_quotas=len(quotas),admitted_pairs=len(records),
                     records=records,actual_base_groups=len(base_receipts),base_audit=base_receipts,
                     frozen_source_sha256=bound,gpu_used=False,model_outputs_used=False,
                     failures=failures,plan_sha256=augmentation.digest(plan_path),
                     elapsed_seconds=time.time()-start)
        augmentation.save_json(complete,receipt)
        augmentation.save_json(receipt_root/"status.json",{k:receipt[k] for k in
                              ("role","status","pilot_only","planned_quotas","admitted_pairs","elapsed_seconds")})
        return {k:receipt[k] for k in ("role","status","admitted_pairs","elapsed_seconds")}
    except BaseException as error:
        augmentation.save_json(receipt_root/"failure.json",dict(status="failed",error=repr(error),
                              traceback=traceback.format_exc(),pid=os.getpid(),elapsed_seconds=time.time()-start))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--role",choices=("cal","select","both"),default="both")
    args=parser.parse_args()
    roles=("cal","select") if args.role=="both" else (args.role,)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_role,(args.root,role,args.pilot)) for role in roles]
        for future in as_completed(futures):
            print(json.dumps(future.result(),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
