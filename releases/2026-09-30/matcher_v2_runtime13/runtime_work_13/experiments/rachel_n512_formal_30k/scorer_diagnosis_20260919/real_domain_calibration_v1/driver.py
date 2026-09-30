"""Finite four-GPU inference queue, then CPU out-of-fold threshold analysis."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from .prepare import ARMS, save
PREFIX="experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.real_domain_calibration_v1."


def run(root):
    root=Path(root)
    raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"],text=True)
    gpus=[line.split(",")[1].strip() for line in raw.splitlines() if line.split(",")[0].strip() in ("0","1","2","3")]
    if len(gpus)!=4:raise RuntimeError("four assigned GPUs required")
    jobs=queue.Queue()
    for arm in ("gcn_pairing_h4","gcn_shredding_h4",*(a for a in ARMS if not a.startswith("gcn_"))):jobs.put(arm)
    states={a:dict(status="pending") for a in ARMS}; lock=threading.Lock()
    def update(arm,**fields):
        with lock:
            states[arm].update(fields)
            save(root/"inference_queue.json",states)
    def worker(gpu):
        while True:
            try:arm=jobs.get_nowait()
            except queue.Empty:return
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONUNBUFFERED="1")
            command=[sys.executable,"-m",PREFIX+"infer","--root",str(root),"--arm",arm,"--gpu-uuid",gpu]
            log=root/"logs"/(arm+".log")
            with log.open("a") as stream:
                process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
                update(arm,status="running",gpu_uuid=gpu,pid=process.pid,started_unix=time.time(),log=str(log))
                result=process.wait()
            update(arm,status="complete" if result==0 else "failed",exit_code=result,finished_unix=time.time())
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,gpus))
    if any(r["status"]!="complete" for r in states.values()):
        save(root/"driver_status.json",dict(status="inference_failed",jobs=states))
        return 1
    save(root/"driver_status.json",dict(status="calibrating_cpu",training=False))
    result=subprocess.run([sys.executable,"-m",PREFIX+"calibrate","--root",str(root)]).returncode
    save(root/"driver_status.json",dict(status="complete" if result==0 else "calibration_failed",training=False,exit_code=result))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--launch",action="store_true")
    a=p.parse_args();root=Path(a.root).resolve();(root/"logs").mkdir(exist_ok=True)
    if a.launch:
        if (root/"driver_launch.json").exists():raise RuntimeError("driver already launched; do not duplicate")
        with (root/"logs/driver.log").open("a") as stream:
            child=subprocess.Popen([sys.executable,"-m",PREFIX+"driver","--root",str(root)],stdout=stream,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED="1"))
        save(root/"driver_launch.json",dict(pid=child.pid,root=str(root),started_unix=time.time()))
        print(json.dumps(dict(status="launched",pid=child.pid,training=False)),flush=True)
    else:sys.exit(run(root))
