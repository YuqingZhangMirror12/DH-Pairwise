"""Admit experiment3 only after completed30K and tested full lifecycle adapters.

One CPU-only invocation, not a polling service. Missing data completion simply
leaves admission absent; it never reserves GPUs or creates a formal run.
"""
from pathlib import Path
import time
from run_queued import PREPARED,FORMAL,QUEUE,TASK,load

REL=Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919')
DATA=Path('/root/autodl-tmp/aggressive_data_v17_full30k_20260927/dataset_03')


def ready_data(common):
    if (DATA/'pipeline_failure.json').exists():raise ValueError('new data has a preserved failure')
    done=common.read(DATA/'pipeline_complete.json')
    if (done.get('status')!='complete' or done.get('pairs')!=30000
            or done.get('data_contract_sha256')!=common.sha(DATA/'data_contract.json')
            or done.get('geometry_sha256')!=common.sha(DATA/'geometry_calibration_v2/geometry_calibration.json')):
        raise ValueError('new30K pipeline has not completely audited/calibrated')
    admission=load(PREPARED/'training_source_03'/REL/'aggressive_binary_v1/admission.py','new_data_admission')
    proof=admission.validate_data(DATA/'data_contract.json',DATA/'geometry_calibration_v2/geometry_calibration.json',DATA/'human_approval.json')
    return proof


def main():
    common=load(QUEUE/'source/common.py','night_queue_io')
    authorization=common.read(QUEUE/'authorization.json')
    if authorization.get('order')!=common.ORDER or not authorization.get('keep_existing_training_unchanged'):
        raise ValueError('latest user queue authorization differs')
    output=QUEUE/'admissions'/(TASK+'.json')
    if output.exists():raise ValueError('admission already registered; never overwrite')
    if (FORMAL/'controller_launch.json').exists():raise ValueError('training already registered; inspect, do not relaunch')
    source=PREPARED/'training_source_03'
    control=load(source/REL/'aggressive_binary_v1/launch_training.py','aggressive_training_controller')
    cpu=common.read(PREPARED/'cpu_tests_remote_03.json');control.validate_cpu(cpu,source)
    entry=load(PREPARED/'aggressive_binary_eval_v1/entry.py','aggressive_evaluation_entry')
    # validate_preparation expects a root containing source; supply its source
    # binding here directly, without constructing a fake formal directory.
    evaluation=common.read(PREPARED/'evaluation_preparation_remote.json')
    if (evaluation.get('status')!='cpu_preparation_passed' or evaluation.get('verified_variants')!=['patch']
            or evaluation.get('training_source_sha256')!=cpu['source_sha256']
            or evaluation.get('adapter_python_sha256')!=entry.inventory(PREPARED/'aggressive_binary_eval_v1')
            or evaluation.get('binary_python_sha256')!=entry.inventory(PREPARED/'binary_eval_v1')
            or evaluation.get('common_python_sha256')!=entry.inventory(PREPARED/'s7_consensus_eval_v14')
            or any(evaluation.get(k)!=0 for k in ('errors','failures','skipped'))
            or evaluation.get('tests',0)<=0):
        raise ValueError('CPU evaluation bindings changed or incomplete')
    queue=common.read(PREPARED/'queue_preparation_remote.json')
    if (queue.get('schema')!='aggressive-binary-evaluation-queue-preparation/1' or queue.get('status')!='passed'
            or queue.get('verified_variants')!=['patch'] or queue.get('tests',0)<=0
            or any(queue.get(k)!=0 for k in ('errors','failures','skipped'))
            or queue.get('queue_python_sha256')!=entry.inventory(PREPARED/'aggressive_binary_eval_queue_v1')
            or any(queue['source_bindings'].get(k)!=evaluation.get(k) for k in queue['source_bindings'])):
        raise ValueError('CPU six-population evaluation queue not prepared')
    wrapper=common.read(PREPARED/'queued_preparation_remote_02.json')
    if (wrapper.get('status')!='passed' or wrapper.get('tests',0)<=0
            or any(wrapper.get(k)!=0 for k in ('errors','failures','skipped'))
            or wrapper.get('source_sha256')!=entry.inventory(Path(__file__).parent)):
        raise ValueError('night queue wrapper not prepared')
    proof=ready_data(common)
    paths=[QUEUE/'authorization.json',QUEUE/'source/common.py',control.CONTROL,
        PREPARED/'cpu_tests_remote_03.json',PREPARED/'evaluation_preparation_remote.json',
        PREPARED/'queue_preparation_remote.json',PREPARED/'queued_preparation_remote_02.json',
        PREPARED/'real_split.json',PREPARED/'s7_consensus_eval_v14/case_plan.json',
        DATA/'pipeline_complete.json',DATA/'data_contract.json',DATA/'human_approval.json',
        DATA/'geometry_calibration_v2/geometry_calibration.json']
    for folder in (source,PREPARED/'aggressive_binary_eval_v1',PREPARED/'binary_eval_v1',
                   PREPARED/'s7_consensus_eval_v14',PREPARED/'aggressive_binary_eval_queue_v1',Path(__file__).parent):
        paths.extend(folder.rglob('*.py'))
    work=QUEUE/'work'/TASK
    result=dict(schema='verified-future-gpu-task/1',status='ready',task=TASK,work=str(work),
        command=[common.PYTHON,str(Path(__file__).with_name('run_queued.py')),'--gpus','{gpus}',
            '--release','{release}','--work',str(work)],files_sha256={str(p):common.sha(p) for p in paths},
        data_admission=proof,formal_root=str(FORMAL),gpu_jobs_started=False,registered_unix=time.time())
    common.save(output,result);print('Registered experiment3 after full data admission; no GPU job started by registration.')


if __name__=='__main__':main()
