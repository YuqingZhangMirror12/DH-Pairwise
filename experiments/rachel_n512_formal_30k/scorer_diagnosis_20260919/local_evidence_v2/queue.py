"""Four-GPU work-conserving finite queue: train + three held-out evaluations.

Each worker owns one allowed physical UUID; no outside GPU is used. No daemon
restarts completed legacy work. A failed job is recorded and the next ready job
can proceed; failures are not silently counted as completion.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .model import CONFIGS
from .train import save

PREFIX="experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.local_evidence_v2."
ORDER=("reference512_h4","gcn_pairing_h4","cap128_h4","cap256_h4",
       "joint_D_h4","stable_h4","cap512_h8","gcn_shredding_h4")


@contextmanager
def locked(root):
    with (root/"dispatch.lock").open("a+") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def update(root,name,**updates):
    with locked(root):
        plan=json.loads((root/"queue.json").read_text())
        record=next(r for r in plan["jobs"] if r["arm"]==name)
        record.update(updates,updated_unix=time.time())
        save(root/"queue.json",plan)


def claim(root,gpu):
    with locked(root):
        plan=json.loads((root/"queue.json").read_text())
        for record in plan["jobs"]:
            if record["status"]=="pending":
                record.update(status="running",stage="train",gpu_uuid=gpu,worker_pid=os.getpid(),started_unix=time.time())
                save(root/"queue.json",plan)
                return record["arm"],plan
    return None,None


def worker(root,gpu):
    root=Path(root).resolve()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER="PCI_BUS_ID",
        OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONUNBUFFERED="1")
    while True:
        arm,plan=claim(root,gpu)
        if arm is None:
            return
        paths=plan["paths"]; training=root/"training"/arm
        shared=["--gpu-uuid",gpu,"--lock-root",str(root/"gpu_locks")]
        train=[sys.executable,"-m",PREFIX+"train","--arm",arm,
            "--train-cache",paths["train_cache"],"--val-cache",paths["val_cache"],
            "--train-diagnostics",paths["train_diagnostics"],"--output",str(training),*shared]
        commands=[("train",train)]
        for split in ("test","real","ood"):
            commands.append((split,[sys.executable,"-m",PREFIX+"evaluate",
                "--training-run",str(training),"--selection","fixed_epoch","--split",split,
                "--output",str(root/"evaluation"/arm/split),"--batch-size","4","--workers","2",
                "--keep-ids",paths["keep_ids"],"--prepared-cache",paths["prepared_cache"],
                "--ood-prepared",paths["ood_prepared"],"--translation-gt-json",paths["translation_gt"],*shared]))
        error=None
        for stage,command in commands:
            update(root,arm,stage=stage)
            log=root/"logs"/(arm+"_"+stage+".log")
            with log.open("a") as stream:
                child=subprocess.Popen(command,env=env,cwd=paths["source"],stdout=stream,stderr=subprocess.STDOUT)
                update(root,arm,child_pid=child.pid,command=command,log=str(log))
                code=child.wait()
            if code:
                error=dict(stage=stage,returncode=code,log=str(log))
                break
        update(root,arm,status="failed" if error else "complete",error=error,
            finished_unix=time.time(),stage="failed" if error else "train_val_test_real_ood_complete")


def launch(root,source,research_root):
    root,source,research_root=Path(root).resolve(),Path(source).resolve(),Path(research_root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    if (root/"queue.json").exists():
        raise ValueError("queue exists; do not launch duplicate workers")
    rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"],text=True)
    gpus=[tuple(v.strip() for v in line.split(",")) for line in rows.splitlines() if line.strip()]
    allowed=[uuid for index,uuid in gpus if index in ("0","1","2","3")]
    if len(allowed)!=4:
        raise ValueError("exactly the four assigned GPUs0..3 must be available")
    diagnostic=research_root/"scorer_diagnosis_20260919"
    paths=dict(source=str(source),train_cache=str(diagnostic/"matched_only_cache_v1/formal_v1/train"),
        val_cache=str(diagnostic/"matched_only_cache_v1/formal_v1/val"),
        train_diagnostics=str(diagnostic/"matched_support_diagnosis_v2/rows.jsonl"),
        keep_ids=str(research_root/"keep_ids.json"),
        prepared_cache="/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared",
        ood_prepared="/root/autodl-tmp/turufan_ood_pairwise_20260912_001/prepared",
        translation_gt="/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json")
    for name,path in paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(name+": "+path)
    (root/"logs").mkdir(exist_ok=True)
    save(root/"queue.json",dict(schema="local-evidence-v2-finite-four-gpu/1",started_unix=time.time(),
        allowed_gpu_uuids=allowed,paths=paths,jobs=[dict(arm=arm,status="pending") for arm in ORDER],
        per_job_completion="C16 train + every-epoch SIMVAL + final SIMTEST/REAL/OOD",
        matcher_frozen=True,automatic_next_pending=True))
    workers=[]
    for gpu in allowed:
        log=root/"logs"/(gpu+"_worker.log")
        with log.open("a") as stream:
            process=subprocess.Popen([sys.executable,"-m",PREFIX+"queue","worker","--root",str(root),"--gpu",gpu],
                cwd=source,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONPATH=str(source),PYTHONUNBUFFERED="1"))
        workers.append(dict(gpu_uuid=gpu,pid=process.pid,log=str(log)))
    save(root/"workers.json",workers)
    print(json.dumps(dict(status="launched",workers=workers,jobs=list(ORDER))),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("launch");a.add_argument("--root",required=True);a.add_argument("--source",required=True);a.add_argument("--research-root",required=True)
    a=sub.add_parser("worker");a.add_argument("--root",required=True);a.add_argument("--gpu",required=True)
    args=p.parse_args()
    if args.action=="launch":launch(args.root,args.source,args.research_root)
    else:worker(args.root,args.gpu)
