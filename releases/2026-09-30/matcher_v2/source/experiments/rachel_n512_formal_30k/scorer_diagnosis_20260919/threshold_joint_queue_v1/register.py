"""One-shot CPU admission for the last task; does not launch/reserve GPUs."""
from pathlib import Path
import time
from contracts import PREPARED,POST,FORMAL,CONTROL,QUEUE,REL,read,save,sha,inventory
from run_queued import load,TASK

def verify_prepared():
    train=read(PREPARED/'training_preparation_v02_remote.json')
    if (train.get('status')!='cpu_preparation_passed' or train.get('tests')!=116
            or train.get('external_control_and_split_tests')!=10
            or any(train.get(k) for k in ('errors','failures','external_errors','external_failures'))
            or train.get('dedicated_launcher_pending') is not False):raise ValueError('joint training preparation missing')
    source=PREPARED/'training_source_02'
    if {str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}!=train['source_inventory_sha256']:
        raise ValueError('joint training source changed')
    if inventory(PREPARED/'threshold_joint_v1')!=train['external_python_sha256']:
        raise ValueError('joint external controller changed')
    old=read(PREPARED/'evaluation_preparation_remote_01/preparation.json')
    for directory,key in ((PREPARED/'threshold_joint_eval_v1','adapter_python_sha256'),(PREPARED/'s7_consensus_eval_v14','common_python_sha256')):
        if inventory(directory)!=old[key]:raise ValueError('training-controller preparation inputs changed')
    evaluation=read(POST/'preparation.json')
    if (evaluation.get('schema')!='threshold-joint-evaluation-preparation/1'
            or evaluation.get('status')!='cpu_preparation_passed' or evaluation.get('tests',0)<192
            or any(evaluation.get(k)!=0 for k in ('errors','failures','skipped'))
            or evaluation.get('both_implementations_import_verified') is not True
            or evaluation.get('real_inference_performed') is not False):raise ValueError('updated independent evaluator not verified')
    for directory,key in ((POST/'threshold_joint_eval_v1','adapter_python_sha256'),(POST/'s7_consensus_eval_v14','common_python_sha256')):
        if inventory(directory)!=evaluation[key]:raise ValueError('updated evaluation files changed')
    if (inventory(source/REL)!=evaluation['implementations']['joint']
            or inventory(CONTROL/'source'/REL)!=evaluation['implementations']['frozen']
            or sha(PREPARED/'real_split.json')!=evaluation['real_plan_sha256']
            or sha(POST/'s7_consensus_eval_v14/case_plan.json')!=evaluation['fixed_case_plan_sha256']):
        raise ValueError('two implementations/cases/real split differ')
    queue=read(PREPARED/'queued_preparation_remote_01.json')
    if (queue.get('status')!='passed' or queue.get('tests',0)<15
            or any(queue.get(k)!=0 for k in ('errors','failures','skipped'))
            or queue.get('source_sha256')!=inventory(Path(__file__).parent)
            or queue.get('source_unchanged') is not True or queue.get('gpu_tasks_started') is not False):
        raise ValueError('full joint lifecycle CPU preparation missing')
    return train,evaluation,queue

def main():
    common=load(QUEUE/'source/common.py','night_queue_io')
    authorization=read(QUEUE/'authorization.json')
    if (authorization.get('order')!=common.ORDER or authorization.get('status')!='user_authorized'
            or authorization.get('keep_existing_training_unchanged') is not True):raise ValueError('latest authorized priority differs')
    output=QUEUE/'admissions'/(TASK+'.json')
    if output.exists() or (FORMAL/'controller_launch_scratch_joint.json').exists():raise ValueError('already registered; never overwrite/restart')
    verify_prepared()
    paths=[QUEUE/'authorization.json',QUEUE/'source/common.py',PREPARED/'training_preparation_v02_remote.json',
        PREPARED/'evaluation_preparation_remote_01/preparation.json',PREPARED/'queued_preparation_remote_01.json',
        POST/'preparation.json',PREPARED/'real_split.json',POST/'s7_consensus_eval_v14/case_plan.json',
        Path('/root/autodl-tmp/binary_scorer_20260927/controller_source_03/launch_training.py')]
    for folder in (PREPARED/'training_source_02',PREPARED/'threshold_joint_v1',PREPARED/'threshold_joint_eval_v1',
                   PREPARED/'s7_consensus_eval_v14',POST/'threshold_joint_eval_v1',POST/'s7_consensus_eval_v14',Path(__file__).parent):
        paths.extend(folder.rglob('*.py'))
    work=QUEUE/'work'/TASK
    save(output,dict(schema='verified-future-gpu-task/1',status='ready',task=TASK,work=str(work),
        command=[common.PYTHON,str(Path(__file__).with_name('run_queued.py')),'--gpus','{gpus}',
            '--release','{release}','--work',str(work)],files_sha256={str(p):sha(p) for p in paths},
        formal_root=str(FORMAL),gpu_jobs_started=False,order='after both lightweight heads and new-data experiment dispatch',
        frozen_control_real_reselection_included=True,registered_unix=time.time()))
    print('Registered last-priority E32 joint lifecycle; no GPU job started by registration.')

if __name__=='__main__':main()
