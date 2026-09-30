"""One-shot low-priority CPU admission then completed B0 reference evaluation.

No GPU allocator, live training changes, model selection, or automatic retries.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from admit_reference_select import ROOT, binding, read, require, save, sha

SOURCE = Path(__file__).resolve().parent
B0_SOURCE = Path('/root/autodl-tmp/curriculum_training_20260928/execution_preparation_10')
B0_SPEC = Path('/root/autodl-tmp/curriculum_training_20260928/locked_plan_02/matcher_execution.json')
B0_SPEC_SHA = 'b846706db28403a3e0a145daa64c4ceb7e00a034d8032433efa5da5c64e5134f'
B0_TRAINING = Path('/root/autodl-tmp/s7_curriculum_v17_v175_v18_20260929/curriculum_matcher_02/training')


def cpu_environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def identity(pid):
    path = Path('/proc')/str(pid)
    stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttime=int(stat[19]), state=stat[0],
        cmdline=(path/'cmdline').read_bytes().replace(b'\0', b' ').decode().strip())


def commands(out):
    admission = out/'admission'
    return [sys.executable, str(SOURCE/'admit_reference_select.py'), '--source-root', str(ROOT/'runtime_work_13'),
            '--out', str(admission)], admission


def evaluation_command(out, admission):
    return [sys.executable, str(SOURCE/'evaluate_reference_select.py'), '--source-root', str(B0_SOURCE),
        '--spec', str(B0_SPEC), '--spec-sha', B0_SPEC_SHA, '--training-root', str(B0_TRAINING),
        '--admission', str(admission), '--admission-sha', sha(admission/'complete.json'),
        '--out', str(out/'b0_native'), '--arm', 'B0', '--selection', 'sim_best']


def child(command, name, out):
    started = time.time()
    with (out/(name+'.log')).open('x') as stream:
        process = subprocess.Popen(command, env=cpu_environment(), stdout=stream, stderr=subprocess.STDOUT)
        save(out/(name+'_launch.json'), dict(command=command, process=identity(process.pid),
            cuda_visible_devices='', started_unix=started, automatic_retry=False))
        code = process.wait()
        stream.flush(); os.fsync(stream.fileno())
    save(out/(name+'_return.json'), dict(returncode=code, elapsed_seconds=time.time()-started,
        log=binding(out/(name+'.log')), completed_unix=time.time(), automatic_retry=False))
    require(code == 0, name + ' child failed')


def run(out, preparation):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only controller required')
    require(not out.exists(), 'exclusive new pipeline root required')
    prep = read(preparation)
    code = {p.name: sha(p) for p in SOURCE.glob('*.py')}
    require(prep['status'] == 'passed' and prep['tests'] >= 18
            and prep['errors'] == prep['failures'] == prep['skipped'] == 0
            and prep['source_unchanged'] and not prep['cuda_initialized']
            and prep['source_sha256'] == code, 'unchanged tested CPU companion required')
    require(sha(B0_SPEC) == B0_SPEC_SHA, 'B0 execution changed')
    out.mkdir()
    save(out/'controller.json', dict(status='running', process=identity(os.getpid()),
        preparation=binding(preparation), source_sha256=code, automatic_retry=False,
        device='cpu', workers=1, nice=os.getpriority(os.PRIO_PROCESS, 0)))
    try:
        command, admission = commands(out)
        child(command, 'admission', out)
        from evaluate_reference_select import verify_admission
        verify_admission(admission, sha(admission/'complete.json'))
        child(evaluation_command(out, admission), 'b0_native', out)
        result = read(out/'b0_native/independent_artifact_audit.json')
        require(result['status'] == 'passed' and result['pairs'] == 900
                and result['diagnostic_cases'] == 30 and result['model_state_unchanged']
                and result['rng_unchanged'] and not result['cuda_initialized'], 'incomplete B0 native reference evaluation')
        end = read(out/'b0_native/evaluation_complete.json')
        require(not (out/'b0_native/failure.json').exists()
                and all(sha(out/'b0_native'/name) == signature for name, signature in end['files'].items()),
                'B0 native terminal output changed')
        require(code == {p.name: sha(p) for p in SOURCE.glob('*.py')}, 'CPU companion changed while running')
        save(out/'complete.json', dict(status='complete', pairs=900, gpu_used=False,
            admission_complete=binding(admission/'complete.json'),
            native_evaluation_complete=binding(out/'b0_native/evaluation_complete.json'),
            independent_audit=binding(out/'b0_native/independent_artifact_audit.json'),
            child_returns={n: binding(out/(n+'_return.json')) for n in ('admission', 'b0_native')},
            classification_accuracy=None, scorer_used=False, automatic_retry=False, completed_unix=time.time()))
    except BaseException as error:
        save(out/'failure.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    args = parser.parse_args()
    os.nice(10)
    run(args.out, args.preparation)


if __name__ == '__main__':
    main()
