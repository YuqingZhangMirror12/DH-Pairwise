"""A released lane evaluates its old head once, then may train a new head."""
import argparse,importlib.util,sys,time,traceback
from pathlib import Path
from common import *

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def release_old(lane,work):
    root=old_root(lane);arm=lane['arm'];post=root/'postprocess_arms_v1'
    control=module(post/'launch_arm_evaluation.py','arm_evaluator')
    control.training_terminal(root,arm)
    if not free(lane['gpus']):raise ValueError('lane still occupied')
    gate=post/'gates';gate.mkdir(exist_ok=True)
    cpu=gate/('cpu_'+arm+'.json');cuda=gate/('cuda_'+arm+'.json')
    launch_wait([PYTHON,str(post/'prepare_arm_remote.py'),'--root',str(root),'--postprocess',str(post),
        '--reference',REFERENCE,'--arm',arm,'--out',str(cpu)],work/'cpu_gate.log',env())
    launch_wait([PYTHON,'-m','s7_consensus_eval_v14.attention_gate','--device','cuda:0','--out',str(cuda)],
        work/'cuda_gate.log',env(lane['gpus'].split(',')[0],str(post)+':'+str(root/'source')))
    out=root/('evaluation_'+arm+'_20260928')
    command=[PYTHON,str(post/'launch_arm_evaluation.py'),'--root',str(root),'--postprocess',str(post),
        '--reference',REFERENCE,'--arm',arm,'--out',str(out),'--gpus',lane['gpus'],'--cpu-receipt',str(cpu),'--cuda-receipt',str(cuda)]
    launch_wait(command,work/'launch_evaluation.log',env())
    registry=read(out/'controller_launch.json');controller=registry.get('controller',registry)
    finished=wait_terminal(out/'evaluation_complete.json',controller,
        [out/'failure.json',out/'controller_failure.json'],'complete')
    checked=control.gates(root,post,arm,cpu,cuda);jobs=[]
    for job in finished['jobs']:
        if job.get('returncode')!=0 or job.get('status')!='complete':raise ValueError('evaluation job did not complete')
        control.verify_result(Path(job['out']),arm,job['split'],checked['selected'][arm])
        jobs.append(dict(root=job['out'],returncode=0))
    result=dict(schema='binary-released-lane/2',status='complete',authorization_sha256=sha(ROOT/'authorization.json'),
        branches=[dict(formal_root=str(root/('formal_'+arm)),evaluations=jobs)],gpus=lane['gpus'],
        terminal_sha256=sha(root/('formal_'+arm)/'training_complete.json'),evaluation_sha256=sha(out/'evaluation_complete.json'))
    control_binary=module(BINARY/'controller_source_03/launch_training.py','binary_controller')
    save(work/'release.json',result);control_binary.validate_queue_release(work/'release.json')
    return dict(status='complete',operation='release_old',release=str(work/'release.json'),gpus=lane['gpus'])
def binary(task,lane,work,release):
    variant=task.removeprefix('binary_')
    if not free(lane['gpus']):raise ValueError('released lane is occupied')
    launcher=BINARY/'controller_source_03/launch_training.py'
    launch_wait([PYTHON,str(launcher),'--root',str(BINARY_ROOT),'--variant',variant,
        '--gpus',lane['gpus'],'--queue-release',release],work/'launch_training.log',env())
    controller=read(BINARY_ROOT/('controller_launch_'+variant+'.json'))
    result=wait_terminal(BINARY_ROOT/('driver_'+variant+'.json'),controller,
        [BINARY_ROOT/('failure_'+variant+'_controller.json')],'training_complete_evaluation_pending')
    if not free(lane['gpus']):raise ValueError('training terminal published but GPUs not released')
    launch_wait([PYTHON,str(BINARY/'binary_eval_queue_v1/launch_evaluation.py'),'--root',str(BINARY_ROOT),
        '--prepared',str(BINARY),'--variant',variant,'--gpus',lane['gpus']],work/'launch_evaluation.log',env())
    out=BINARY_ROOT/('evaluation_'+variant+'_01');record=read(out/'controller_launch.json')['controller']
    evaluation=wait_terminal(out/'evaluation_complete.json',record,
        [out/'controller_failure.json',out/'failure.json',out/'launch_failure.json'],'complete')
    if (len(evaluation.get('jobs',[]))!=6 or not evaluation.get('all_six_populations_verified')
            or any(j.get('status')!='complete' or j.get('returncode')!=0 for j in evaluation['jobs'])):
        raise ValueError('six frozen evaluations required')
    return dict(status='complete',operation=task,gpus=lane['gpus'],training_root=str(BINARY_ROOT/('formal_'+variant)),
        training_terminal_sha256=sha(BINARY_ROOT/('formal_'+variant)/'training_complete.json'),
        evaluation_complete=str(out/'evaluation_complete.json'),evaluation_sha256=sha(out/'evaluation_complete.json'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--lane',required=True);p.add_argument('--operation',required=True)
    p.add_argument('--work',required=True);p.add_argument('--release');a=p.parse_args()
    lane=next(x for x in LANES if x['id']==a.lane);work=Path(a.work);work.mkdir(exist_ok=False)
    try:
        result=release_old(lane,work) if a.operation=='release_old' else binary(a.operation,lane,work,a.release)
        save(work/'complete.json',dict(result,completed_unix=time.time()))
    except BaseException as error:
        save(work/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),automatic_retries=0));raise
if __name__=='__main__':main()
