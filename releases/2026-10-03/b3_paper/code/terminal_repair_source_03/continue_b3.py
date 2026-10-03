"""B3-only continuation, armed for the verified strict identity failure.

Never kills/preempts/restarts training. Waits for the old owner to exit, verifies
the complete training and exact known evaluation failure, then repairs missing
evaluation and starts the original two new heads under repaired terminal flow.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import os
from pathlib import Path
import time
import traceback

from common import HERE, PACKAGE, api, require, read, sha, bound, save, check_preparation, same_process, environment
import pipeline

ROOT = Path('/root/autodl-tmp/matcher_v2_20260930')
SOURCE = ROOT/'runtime_work_13'
OLD = ROOT/'b3_matcher_pipeline_01'
OLD_QUEUE = ROOT/'b3_head_queue_01'
OUT = ROOT/'b3_head_queue_03'
MATCHER = ROOT/'b3_matcher_terminal_repair_02'
REPAIR_PREP = ROOT/'terminal_modefix_cpu_remote_20261001_02.json'
PREVIOUS = ROOT/'b3_head_queue_02'
OLD_SCRIPT = ROOT/'head_queue_source_01/continue_b3_heads.py'
SCRIPT_SHA = 'bc21621f2d5833cee4f7d40a81b7ce5c3d14753e46755f406039a08225b22a2f'


def old_helper():
    path = OLD_SCRIPT
    require(sha(path)==SCRIPT_SHA, 'original B3 wait contract changed')
    spec = importlib.util.spec_from_file_location('original_b3_contract',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.check_bindings()
    return module


def require_heads_unstarted():
    for module in ('scorer_patch','scorer_stats'):
        for suffix in ('locked_01','pipeline_01','locked_02','pipeline_02'):
            require(not (ROOT/('b3_'+module+'_'+suffix)).exists(), 'B3 head was already prepared/started; no takeover')
        require(not (OLD_QUEUE/module).exists(), 'old B3 head lane already started')
    require(not (OLD_QUEUE/'verified_matcher.json').exists(), 'old owner already admitted Matcher')


def pending_state(identity):
    for path in [OLD/'training/controller_failure.json',OLD/'training/failure.json',
                 OLD/'training/formal/failure.json',*list((OLD/'training/formal').glob('failure_attempt_*.json'))]:
        require(not path.exists(), 'training failure cannot be handled by identity repair')
    if (OLD/'complete.json').exists() and not (OLD/'failure.json').exists():
        return 'original_completed_no_takeover'
    if not (OLD/'failure.json').exists(): return 'waiting_original_matcher'
    for path,key in ((OLD/'pipeline_launch.json','controller'),(OLD/'evaluation_launch.json','process'),
                     (OLD/'evaluation/controller_launch.json','controller'),(OLD_QUEUE/'launch.json','controller')):
        if path.exists() and same_process(read(path).get(key),identity):
            return 'waiting_old_owners_to_exit'
    require((OLD_QUEUE/'failure.json').is_file(), 'old head waiter has no terminal failure receipt')
    require(read(OLD_QUEUE/'failure.json').get('error') ==
            "ValueError('Matcher failure overrides completion; no automatic retry')", 'old head waiter failed for another reason')
    require_heads_unstarted()
    return 'ready_for_exact_failure_verification'


def task_args(old, spec, out, gpus, adopt=None):
    args = old.task_args(spec,out,gpus)
    args.repair_preparation = REPAIR_PREP; args.adopt_pipeline = adopt
    return args


def child_command(args):
    values = [str(args.python),str(HERE/'pipeline.py')]
    for name in ('spec','preparation','canonical_straight','case_plan','python','out','repair_preparation'):
        values += ['--'+name.replace('_','-'),str(getattr(args,name))]
    values += ['--gpus',*map(str,args.gpus)]
    if args.adopt_pipeline: values += ['--adopt-pipeline',str(args.adopt_pipeline)]
    return values


def execute_task(lane,args):
    launcher = api('curriculum_training_v1.launcher'); lane.mkdir()
    values = child_command(args)
    launcher.execute(lane,'pipeline',values,environment(SOURCE,args.gpus),SOURCE)
    api('matcher_v2_v1.pipeline').verify_child(lane,'pipeline',values)
    result = pipeline.verify_complete(args.out)
    save(lane/'complete.json',dict(status='complete',pipeline=bound(args.out/'complete.json'),
                                  returned=bound(lane/'pipeline_return.json')))
    return result


def run_head(item,old,selected):
    module,gpu = item; old.check_bindings(); check_preparation(REPAIR_PREP)
    locked = ROOT/('b3_'+module+'_locked_02')
    spec = api('matcher_v2_v1.compile_execution').compile_execution(old.BASE,'B3',module,locked,
            combined_admission=old.COMBINED,selected=selected)
    require(spec['topology']['world_size']==1 and spec['topology']['microbatch']==32,
            'head topology changed')
    args = task_args(old,locked/'execution.json',ROOT/('b3_'+module+'_pipeline_02'),[gpu])
    return execute_task(OUT/module,args)


def preflight():
    check_preparation(REPAIR_PREP); old = old_helper()
    launcher = api('curriculum_training_v1.launcher')
    require(sha(PREVIOUS/'failure.json') ==
            'eb7e0d2f0f7fcaf034bdeb32b93aab37d717488d7ff818b4b0c6d2bdf32e98f2',
            'previous failed continuation identity differs')
    require(not same_process(read(PREVIOUS/'launch.json').get('controller'), launcher.identity),
            'previous continuation is still live')
    for name in ('scorer_patch', 'scorer_stats', 'verified_matcher.json', 'complete.json'):
        require(not (PREVIOUS/name).exists(), 'previous continuation already started or admitted heads')
    api('matcher_v2_v1.evaluate').check_preparation(old.PREPARATION,SOURCE)
    inputs = api('matcher_v2_v1.runtime_inputs').load_inputs(read(old.SPEC))
    require(inputs['plan'].record['total_updates']==31667 and read(old.SPEC)['arm']=='B3'
            and read(old.SPEC)['module']=='matcher', 'not the complete B3 main experiment')
    require_heads_unstarted()
    require(not OUT.exists() and not MATCHER.exists(), 'exclusive B3 repair queue required')
    return old, inputs


def run():
    old,inputs = preflight(); launcher = api('curriculum_training_v1.launcher')
    OUT.mkdir(); lock = (OUT/'controller.lock').open('x')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'launch.json',dict(controller=launcher.identity(os.getpid()),script=bound(__file__),
        repair_preparation=bound(REPAIR_PREP),original_waiter_launch=bound(OLD_QUEUE/'launch.json'),
        original_pipeline_launch=bound(OLD/'pipeline_launch.json'),
        lanes=[dict(module=m,gpu=g) for m,g in old.LANES],automatic_retry=False,
        strict_known_failure_only=True,training_restarts=False,started_unix=time.time()))
    try:
        deadline = time.monotonic()+48*3600
        while True:
            state = pending_state(launcher.identity)
            api('curriculum_training_v1.checkpoint_io').write_json(OUT/'status.json',dict(status=state,
                gpu_allocated=False,automatic_retry=False,observed_unix=time.time()),replace=True)
            if state=='original_completed_no_takeover':
                # Unexpected healthy old completion: original owner remains sole
                # owner. Do not start duplicate heads or mark their work done.
                save(OUT/'not_needed.json',dict(status=state,head_owner=str(OLD_QUEUE),heads_claimed_complete=False))
                return
            if state=='ready_for_exact_failure_verification': break
            require(time.monotonic()<deadline,'bounded B3 completion wait expired')
            time.sleep(60)
        old.check_bindings(); check_preparation(REPAIR_PREP)
        args = task_args(old,old.SPEC,MATCHER,[0,5],OLD)
        completed = execute_task(OUT/'matcher_repair',args)
        training = Path(completed['effective_training_controller_root'])
        receipt_path = training/'export_process_return.json'; returned = read(receipt_path)
        require(returned['returncode']==0,'actual Matcher export return required')
        exported = read(Path(returned['export_root'])/'training_complete.json')
        selected = dict(export_root=returned['export_root'],process_return=str(receipt_path),
                        common_plan_sha256=exported['binding']['common_plan_sha256'])
        require(selected['common_plan_sha256']==inputs['plan'].sha256,'selected Matcher plan changed')
        save(OUT/'verified_matcher.json',dict(repaired_pipeline=bound(MATCHER/'complete.json'),selected=selected))
        require_heads_unstarted(); launcher.check_free([0,5],2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda item:run_head(item,old,selected),old.LANES))
        save(OUT/'complete.json',dict(status='B3_both_heads_and_repaired_terminal_evaluation_complete',
            results=results,automatic_retry=False,report_delivery_still_required=True,completed_unix=time.time()))
    except BaseException as error:
        save(OUT/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.preflight:
        preflight(); print('B3 continuation preflight passed; no CUDA/training launch')
    else: run()
