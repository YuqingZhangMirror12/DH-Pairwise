"""One independent CPU-only posthoc queue; never edits or starts GPU work."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import score_scorer as adapter

ROOT = Path('/root/autodl-tmp/matcher_v2_20260930')
B0_ROOT = Path('/root/autodl-tmp/s7_curriculum_v17_v175_v18_20260929')
B0_SOURCE = Path('/root/autodl-tmp/curriculum_training_20260928/execution_preparation_10')
SOURCE13_SHA = '9558af206eb70de7f9b62689ad4c9baa57587f875cc8fa309c0d3fa5719cde6c'
STRATA_SHA = '5f0d7a7be2f39857a92935a262c078f540890d36794dbea5ee91cebc17ebc357'


def registered_jobs(root=ROOT, b0=B0_ROOT):
    return [dict(name=arm+'_'+module, arm=arm, module=module,
        pipeline=str(b0/(module+'_01') if arm == 'B0' else root/(arm.lower()+'_'+module+'_pipeline_01')),
        source=str(B0_SOURCE if arm == 'B0' else root/'runtime_work_13'))
        for arm in ('B0', 'B3', 'B1', 'B2') for module in adapter.MODULES]


def upstream(job, root=ROOT):
    path = Path(job['pipeline']); arm = job['arm']
    files = [path/'failure.json', path/'pipeline_failure.json', path/'training/controller_failure.json', path/'training/formal/failure.json',
             path/'evaluation/controller_failure.json']
    files += sorted((path/'training/formal').glob('failure_attempt_*.json'))
    if arm == 'B3':files += [root/'b3_head_queue_01/failure.json', root/'b3_head_queue_01'/job['module']/'failure.json']
    elif arm in ('B1', 'B2'):
        files += [root/'b12_control_queue_01/failure.json', root/'b12_control_queue_01'/arm/'failure.json']
    failure = next((p for p in files if p.exists()), None)
    if failure is not None:
        return dict(state='failed', failure=adapter.binding(failure), detail=adapter.read(failure))
    terminal = path/('pipeline_complete.json' if arm == 'B0' else 'complete.json')
    if not terminal.is_file():return dict(state='waiting')
    receipt = adapter.read(terminal)
    expected = 'training_and_required_evaluation_complete' if arm == 'B0' else 'complete'
    adapter.require(receipt.get('status') == expected and receipt.get('module') == job['module'],
                    'malformed/wrong-module upstream terminal')
    return dict(state='ready', terminal=adapter.binding(terminal))


def identity(pid):
    path = Path('/proc')/str(pid); stat = (path/'stat').read_text().split(') ', 1)[1].split()
    return dict(pid=pid, starttime=int(stat[19]), state=stat[0],
                cmdline=(path/'cmdline').read_bytes().replace(b'\0', b' ').decode().strip())


def replace_json(path, value):
    tmp = Path(str(path)+'.tmp'); adapter.save(tmp, value); os.replace(tmp, path)


def command(job, out, python, root=ROOT):
    return [str(python), str(Path(__file__).parent/'score_scorer.py'), '--arm', job['arm'], '--module', job['module'],
        '--source-root', job['source'], '--evaluation-root', str(Path(job['pipeline'])/'evaluation'),
        '--diagnostics-source', str(root/'diagnostics_source_03'), '--strata', str(root/'development_strata_01/plan.json'),
        '--out', str(out)]


def check_static(root=ROOT):
    adapter.require(adapter.sha(root/'runtime_work_13/source_binding.json') == SOURCE13_SHA, 'frozen source13 changed')
    adapter.require(adapter.sha(root/'development_strata_01/plan.json') == STRATA_SHA, 'fixed development membership changed')
    adapter.require(adapter.sha(root/'diagnostics_source_03/diagnostic_metrics.py') == adapter.METRIC_SHA, 'metric code changed')


def execute_job(job, queue, python, expected_code, *, root=ROOT, build=command, verify=adapter.verify_output):
    out = queue/job['name']; adapter.require(not out.exists(), 'preserve prior attempt')
    values = build(job, out, python, root)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    started = time.time()
    with (queue/(job['name']+'.log')).open('xb') as log:
        child = subprocess.Popen(values, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        process = dict(pid=child.pid)
        if Path('/proc').exists():
            try:process = identity(child.pid)
            except FileNotFoundError:process['already_exited'] = True
        launch = queue/(job['name']+'_launch.json')
        adapter.save(launch, dict(job=job, command=values, process=process, started_unix=started,
            gpu_used=False, model_inference=False, automatic_retry=False))
        code = child.wait()
    returned = queue/(job['name']+'_return.json')
    adapter.save(returned, dict(returncode=code, launch_sha256=adapter.sha(launch),
        elapsed_seconds=time.time()-started, automatic_retry=False))
    adapter.require(code == 0, 'posthoc child failed: '+str(code))
    audit = verify(out, job['arm'], job['module'], expected_code)
    adapter.save(queue/(job['name']+'_audit.json'), audit)
    return dict(job=job['name'], return_sha256=adapter.sha(returned), audit=audit)


def run(args):
    source = Path(__file__).resolve().parent
    adapter.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not args.out.exists(), 'CPU-only fresh queue required')
    check_static(); proof = adapter.read(args.preparation); code = adapter.inventory(source)
    adapter.require(proof['status'] == 'passed' and proof['source_sha256'] == code and proof['source_unchanged'] is True
            and proof['tests'] >= 27 and proof['errors'] == proof['failures'] == proof['skipped'] == 0
            and proof['gpu_used'] is False, 'matching passed CPU preparation required')
    sys.path.insert(0, str(ROOT/'diagnostics_source_03'))
    import diagnostic_metrics
    adapter.require(adapter.sha(diagnostic_metrics.__file__) == adapter.METRIC_SHA, 'wrong diagnostic import')
    pending = registered_jobs(); completed = []; failures = []
    args.out.mkdir(parents=True); os.nice(10)
    adapter.save(args.out/'launch.json', dict(controller=identity(os.getpid()), jobs=pending.copy(), source_sha256=code,
        preparation=adapter.binding(args.preparation), cpu_workers=1, niceness=10, gpu_used=False,
        model_inference=False, automatic_retry=False, wait_seconds=60))
    try:
        while pending:
            adapter.require(adapter.inventory(source) == code, 'immutable queue source changed')
            ready = None
            for job in list(pending):
                state = upstream(job)
                if state['state'] == 'failed':
                    failure = dict(job=job['name'], upstream=state, automatic_retry=False)
                    adapter.save(args.out/(job['name']+'_failure.json'), failure)
                    failures.append(failure); pending.remove(job)
                elif state['state'] == 'ready' and ready is None:ready = job
            replace_json(args.out/'status.json', dict(status='running_with_failure' if failures else 'waiting_or_running',
                pending=[j['name'] for j in pending], completed=completed, failures=failures,
                active=ready['name'] if ready else None, gpu_used=False))
            if ready is None:
                if pending:time.sleep(60)
                continue
            pending.remove(ready)
            try:
                check_static()
                completed.append(execute_job(ready, args.out, args.python, code))
            except Exception as error:
                failure = dict(job=ready['name'], error=repr(error), traceback=traceback.format_exc(), automatic_retry=False)
                adapter.save(args.out/(ready['name']+'_failure.json'), failure); failures.append(failure)
        result = dict(status='failed' if failures else 'complete', completed=completed, failures=failures,
                      automatic_retry=False, source_sha256=code, gpu_used=False, model_inference=False)
        adapter.save(args.out/('failure.json' if failures else 'complete.json'), result)
        replace_json(args.out/'status.json', dict(result, pending=[], active=None))
    except BaseException as error:
        if not (args.out/'failure.json').exists():
            adapter.save(args.out/'failure.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
                automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'python', 'preparation'):parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':main()
