"""Low-priority CPU-only job; reserve resources for the six existing GPU ranks."""
from pathlib import Path
import os, shutil, time

GIB=1024**3


def snapshot(root):
    cpu=Path('/sys/fs/cgroup/cpu.max'); quota=cpu.read_text().split() if cpu.exists() else ['max','100000']
    cores=float(quota[0])/float(quota[1]) if quota[0]!='max' else float(os.cpu_count())
    cur=Path('/sys/fs/cgroup/memory.current'); limit=Path('/sys/fs/cgroup/memory.max')
    memory=int(cur.read_text()) if cur.exists() else 0
    cap=int(limit.read_text()) if limit.exists() and limit.read_text().strip()!='max' else 10**16
    return dict(cpu_quota_cores=cores,load1=os.getloadavg()[0],memory_bytes=memory,
        memory_limit_bytes=cap,disk_free_bytes=shutil.disk_usage(root).free,
        checked_unix=time.time())


def dispatch_allowed(s):
    return (s['disk_free_bytes']>=50*GIB and
            s['memory_bytes']<=min(.75*s['memory_limit_bytes'],s['memory_limit_bytes']-60*GIB) and
            s['load1']<.80*s['cpu_quota_cores'])


def worker_cap(s, requested):
    # At least half of the CPU entitlement remains outside our worker pool.
    return max(1,min(32,int(requested),int(s['cpu_quota_cores']//2)))


def low_priority():
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    if hasattr(os,'nice'):
        current=os.getpriority(os.PRIO_PROCESS,0)
        if current<15:os.nice(15-current)

