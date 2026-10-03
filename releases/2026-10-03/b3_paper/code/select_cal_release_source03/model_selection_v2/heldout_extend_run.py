"""Append-only source-reserve extension; retain source03 evidence unchanged.

This is data construction, not an experiment runner. The original pixel,
geometry, source, target and duplicate-input gates remain in force. Successful
old pilot records are admitted by reference, never relabelled as fresh output.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import os
from pathlib import Path
import time
import traceback

try:
    from . import heldout_run as original, heldout_augment as augmentation
    from .heldout_extend_plan import validate_extension
    from .heldout_adopt import check_admission
except ImportError:
    import heldout_run as original
    import heldout_augment as augmentation
    from heldout_extend_plan import validate_extension
    from heldout_adopt import check_admission


def read(path):
    return json.loads(Path(path).read_text())


def task_key(task):
    return json.dumps(task,sort_keys=True,separators=(",",":"))


def admit_commit(imported, task, plan_sha, output):
    """A new wrapper references, rather than overwrites, original success."""
    if set(imported)!={"commit","old_plan","old_pilot_complete"}:
        raise ValueError("exact imported proof bindings required")
    for ref in imported.values():
        if augmentation.digest(ref["path"])!=ref["sha256"]:
            raise ValueError("old imported receipt bytes changed")
    old=augmentation.verify_commit(imported["commit"]["path"],task,imported["old_plan"]["sha256"])
    receipt=dict(status="committed",task=task,plan_sha256=plan_sha,
                 records=old["records"],audit_rows=old["audit_rows"],
                 imported_from=imported,old_pixels_regenerated=False,
                 source_replaced=False,v14_fallback=False)
    output=Path(output)
    if output.exists():
        if read(output)!=receipt:
            raise ValueError("existing imported wrapper differs")
    else:
        augmentation.save_json(output,receipt)
    return receipt


def run_role(args):
    root,role,pilot=args
    root=Path(root).resolve()
    plan_path=root/"plan"/"generation_plan.json"
    plan,sources=validate_extension(plan_path)
    admission=read(root/"extension_admission.json")
    check_admission(admission)
    adopted=admission["roles"][role]
    inventory=read(root/"frozen_source_inventory.json")
    receipt_root=root/("pilot_receipts" if pilot else "build_receipts")/role
    complete=receipt_root/"complete.json"
    if complete.exists():
        raise ValueError("completed invocation cannot be repeated")
    receipt_root.mkdir(parents=True,exist_ok=True)
    start=time.time()
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        os.nice(10)
        import torch
        import cv2
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        heldout=importlib.import_module(original.PREFIX+".s7_consensus_v1.heldout_v14")
        material=importlib.import_module(original.PREFIX+".s7_balanced_v2.materialize")
        audit=importlib.import_module(original.PREFIX+".s7_balanced_v2.audit_layered")
        source=importlib.import_module(original.PREFIX+".aggressive_data_full_v17.source")
        baseline_root=root/"baseline"/role
        baseline_root.mkdir(parents=True,exist_ok=True)
        heldout.initialize(dict(sources=plan["source_plan_path"],split=role,out=str(baseline_root),
             seed=plan["master_seed"]+(10001 if role=="cal" else 20002),attempts=1024))
        config=dict(out=str(root/"augmented"),generation_plan_sha256=augmentation.digest(plan_path),geometry_attempts=24)
        source.STATE.clear()
        source.baseline.cache_clear()
        source.STATE.update(config=config,split=role,root=baseline_root,groups={},bank=material.STATE["bank"],
            negative={e["pair_id"]:e for e in material.STATE["negative_plan"]})
        augmentation.STATE.clear()
        augmentation.STATE.update(config=config,role=role,source=source)
        bound=original.bind_loaded_sources()
        original.validate_loaded_inventory(bound,inventory)
        completed,failures,base_receipts,base_roots=[],[],{},{}
        seen=set()
        quotas=original.quota_groups(plan["tasks"],role,pilot)
        for key,tasks in quotas:
            accepted=None
            for task in tasks:
                slot=task["slot"]
                prior_failure=adopted["rejections"].get(task_key(task))
                if prior_failure is not None:
                    failures.append(dict(task=task,phase=prior_failure["phase"],error=prior_failure["error"],
                         imported_rejection_from=prior_failure["old_pilot_complete"],replayed=False))
                    continue
                group_name=f'{task["stage"]}:{slot}'
                imported=adopted["commits"].get(group_name)
                commit=root/"augmented"/role/task["stage"]/"groups"/f"{slot:05d}.json"
                if imported is not None:
                    result=admit_commit(imported,task,config["generation_plan_sha256"],commit)
                    # All baseline files behind the imported commit remain
                    # pinned by the immutable old-pilot admission receipt.
                    oldbase=adopted["baselines"][str(slot)]
                    base_receipts[str(slot)]=dict(audit_rows=oldbase["audit_rows"],
                        group_sha256=oldbase["group"]["sha256"],imported_from=oldbase["group"],root=oldbase["root"])
                else:
                    if slot not in source.STATE["groups"]:
                        oldbase=adopted["baselines"].get(str(slot))
                        if oldbase is not None:
                            baseline=read(oldbase["group"]["path"])
                            base_roots[slot]=Path(oldbase["root"])
                            base_receipts[str(slot)]=dict(audit_rows=oldbase["audit_rows"],
                                group_sha256=oldbase["group"]["sha256"],imported_from=oldbase["group"],root=oldbase["root"])
                        else:
                            try:
                                baseline=heldout.slot(slot)
                            except RuntimeError as error:
                                if not original.expected_baseline_exhaustion(error,task):raise
                                failures.append(dict(task=task,phase="baseline_geometry_exhausted",error=str(error)))
                                continue
                            audits=[audit.one((str(baseline_root),entry)) for entry in baseline["entries"]]
                            base_roots[slot]=baseline_root
                            base_receipts[str(slot)]=dict(audit_rows=audits,
                                group_sha256=augmentation.digest(baseline_root/"groups"/f"{slot:05d}.json"),root=str(baseline_root))
                        source.STATE["groups"][slot]={i:e for i,e in enumerate(baseline["entries"])}
                    source.STATE["root"]=base_roots[slot]
                    result=augmentation.process(task)
                if result["status"]=="committed":
                    hashes=[r["model_tensors_sha256"] for r in result["records"]]
                    if len(set(hashes))!=2 or set(hashes)&seen:
                        failures.append(dict(task=task,phase="duplicate_model_input",error="actual input duplicate; original proof retained"))
                        continue
                    seen.update(hashes)
                    accepted=result
                    break
                failures.append(dict(task=task,phase="augmentation",error=result.get("reasons")))
            if accepted is None:
                failures.append(dict(quota=list(key),phase="quota_exhausted",error="all twelve preregistered candidates rejected"))
            else:completed.append(accepted)
            progress=dict(status="running",role=role,pid=os.getpid(),
                finished_quotas=len(completed)+sum(f["phase"]=="quota_exhausted" for f in failures),
                planned_quotas=len(quotas),admitted_pairs=2*len(completed),elapsed_seconds=time.time()-start)
            augmentation.save_json(receipt_root/"status.json",progress)
            if progress["finished_quotas"]%20==0 or progress["finished_quotas"]==len(quotas):
                print(json.dumps(dict(progress,pilot=pilot)),flush=True)
        later=original.bind_loaded_sources()
        original.check_sources(bound)
        original.validate_loaded_inventory(later,inventory)
        bound.update(later)
        check_admission(admission)
        records=[r for c in completed for r in c["records"]]
        exhausted=[f for f in failures if f["phase"]=="quota_exhausted"]
        if not pilot and not exhausted:
            expected={(stage,gen,label):120 for stage in augmentation.STAGES for gen in ("Gen2","Gen3","Gen4","Gen5") for label in (False,True)}
            if len(records)!=2880 or Counter((r["stage"],r["generator"],r["label"]) for r in records)!=Counter(expected):
                raise ValueError("exact full source population failed")
        receipt=dict(schema="mixed-sim-heldout-role-build/2",role=role,pilot_only=pilot,
             status="pilot_passed" if pilot and not exhausted else "complete" if not exhausted else "shortfall",
             planned_quotas=len(quotas),admitted_pairs=len(records),records=records,
             actual_base_groups=len(base_receipts),base_audit=base_receipts,failures=failures,
             frozen_source_sha256=bound,gpu_used=False,model_outputs_used=False,
             plan_sha256=augmentation.digest(plan_path),elapsed_seconds=time.time()-start,
             extension_admission=dict(path=str(root/"extension_admission.json"),sha256=augmentation.digest(root/"extension_admission.json")),
             imported_groups=sum("imported_from" in c for c in completed),reserve_count=12)
        augmentation.save_json(complete,receipt)
        augmentation.save_json(receipt_root/"status.json",{k:receipt[k] for k in ("status","role","planned_quotas","admitted_pairs","elapsed_seconds")})
        return {k:receipt[k] for k in ("status","role","planned_quotas","admitted_pairs","elapsed_seconds","imported_groups")}
    except BaseException as error:
        augmentation.save_json(receipt_root/"failure.json",dict(status="failed",error=repr(error),traceback=traceback.format_exc(),pid=os.getpid()))
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--pilot",action="store_true")
    args=p.parse_args()
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(run_role,(args.root,role,args.pilot)) for role in ("cal","select")]
        for future in as_completed(tasks):print(json.dumps(future.result()),flush=True)


if __name__=="__main__":main()
