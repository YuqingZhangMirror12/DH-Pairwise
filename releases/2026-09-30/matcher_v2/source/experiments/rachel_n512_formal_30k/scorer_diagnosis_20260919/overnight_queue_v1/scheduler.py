"""Local server event dispatcher; independent of the desktop's two-hour reports."""
import argparse,fcntl,sys,time,traceback
from common import *

def next_task(tasks):
    for name in ORDER:
        if tasks[name]['status']=='pending':return name
        if tasks[name]['status']=='failed':return None
    return None
def old_status(lane):
    root=old_root(lane);arm=lane['arm'];formal=root/('formal_'+arm)
    if any((root/n).exists() for n in ('failure_'+arm+'_controller.json','failure_'+arm+'.json')) or any((formal/n).exists() for n in ('failure.json','scorer/failure.json')):
        return 'failed'
    driver=read(root/('driver_'+arm+'.json'));launch=read(root/('formal_launch_'+arm+'.json'))
    if driver.get('status')=='training_complete_evaluation_pending' and (root/(arm+'_formal_exit.json')).exists():
        if read(root/(arm+'_formal_exit.json')).get('returncode')!=0:return 'failed'
        if (formal/'training_complete.json').exists() and not live(launch['job']):return 'terminal'
    if live(launch['job']) or live(launch['controller']):return 'running'
    return 'unexpected_exit'
def start_worker(lane,operation,release=None):
    work=ROOT/'work'/(lane['id']+'_'+operation)
    command=[PYTHON,str(ROOT/'source/worker.py'),'--lane',lane['id'],'--operation',operation,'--work',str(work)]
    if release:command+=['--release',release]
    log=ROOT/'logs'/(lane['id']+'_'+operation+'.log')
    with log.open('xb') as f:
        proc=subprocess.Popen(command,env=env(),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    record=dict(operation=operation,work=str(work),identity=identity(proc.pid),started_unix=time.time(),command=command)
    save(ROOT/'launches'/(lane['id']+'_'+operation+'.json'),record)
    return proc,record
def driver():
    lock=(ROOT/'scheduler.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(ROOT/'plan.json');authorization=read(ROOT/'authorization.json')
    if plan['authorization_sha256']!=sha(ROOT/'authorization.json') or authorization['order']!=ORDER:raise ValueError('authorization binding changed')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/'source'/name)!=digest:raise ValueError('scheduler source changed')
    if (ROOT/'driver_status.json').exists():raise ValueError('preserve prior state; no implicit scheduler restart')
    tasks={name:dict(status='pending') for name in ORDER}
    lanes={x['id']:dict(config=x,status='existing_training') for x in LANES};active={};problems=[]
    for folder in ('work','logs','launches'):(ROOT/folder).mkdir(exist_ok=True)
    while True:
        for lane_id,(proc,record) in list(active.items()):
            if proc.poll() is None:continue
            lane=lanes[lane_id];work=Path(record['work']);op=record['operation']
            if proc.returncode or (work/'failure.json').exists() or not (work/'complete.json').exists():
                lane['status']='failed';problems.append(dict(lane=lane_id,operation=op,returncode=proc.returncode,work=str(work)))
                if op in tasks:tasks[op].update(status='failed')
            else:
                result=read(work/'complete.json');lane['status']='released'
                if op=='release_old':lane['release']=result['release']
                else:tasks[op].update(status='complete',result=result)
            del active[lane_id]
        for lane_id,lane in lanes.items():
            if lane['status']=='existing_training':
                state=old_status(lane['config'])
                if state in ('failed','unexpected_exit'):
                    lane['status']='failed';problems.append(dict(lane=lane_id,operation='existing_training',reason=state))
                elif state=='terminal' and free(lane['config']['gpus']):
                    active[lane_id]=start_worker(lane['config'],'release_old');lane['status']='frozen_evaluation'
        for lane_id,lane in lanes.items():
            if lane['status']!='released' or not free(lane['config']['gpus']):continue
            task=next_task(tasks)
            if task is None:continue
            if task in ('binary_patch','binary_stats'):
                active[lane_id]=start_worker(lane['config'],task,lane['release']);lane['status']=task
                tasks[task].update(status='running',lane=lane_id,started_unix=time.time())
            else:
                # Later stage admission is deliberately explicit. A pending
                # unverified training/evaluation adapter cannot be launched.
                ready=ROOT/'admissions'/(task+'.json')
                if not ready.exists():
                    tasks[task]['waiting_for']='verified dedicated controller and new-data admission' if task=='aggressive_scratch' else 'earlier experiment launch plus verified joint continuation'
                    continue
                record=read(ready)
                if record.get('status')!='ready' or record.get('task')!=task:raise ValueError('invalid future job admission')
                for file,digest in record['files_sha256'].items():
                    if sha(file)!=digest:raise ValueError('future controller source changed')
                command=[str(x).replace('{gpus}',lane['config']['gpus']).replace('{release}',lane['release']) for x in record['command']]
                log=ROOT/'logs'/(task+'.log')
                with log.open('xb') as f:
                    proc=subprocess.Popen(command,env=env(),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                work=Path(record['work']);launch=dict(operation=task,work=str(work),identity=identity(proc.pid),command=command,started_unix=time.time())
                save(ROOT/'launches'/(task+'.json'),launch);active[lane_id]=(proc,launch)
                lane['status']=task;tasks[task].update(status='running',lane=lane_id)
        save(ROOT/'driver_status.json',dict(status='running_with_attention' if problems else 'running',controller=identity(os.getpid()),
            tasks=tasks,lanes=lanes,active={k:r for k,(_,r) in active.items()},problems=problems,
            updated_unix=time.time(),automatic_retries=0,training_sources_modified=False,poll_seconds=60))
        if all(t['status']=='complete' for t in tasks.values()):
            save(ROOT/'queue_complete.json',dict(status='complete',tasks=tasks,finished_unix=time.time()));return
        time.sleep(60)
def main():
    p=argparse.ArgumentParser();p.add_argument('--driver',action='store_true');a=p.parse_args()
    if a.driver:
        try:driver()
        except BaseException as error:save(ROOT/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),automatic_retries=0));raise
    else:
        if (ROOT/'controller_launch.json').exists():raise ValueError('scheduler already registered')
        with (ROOT/'controller.log').open('xb') as f:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--driver'],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env(),start_new_session=True)
        result=dict(controller=identity(child.pid),created_unix=time.time(),automatic_restarts=0)
        save(ROOT/'controller_launch.json',result);print(json.dumps(result))
if __name__=='__main__':main()
