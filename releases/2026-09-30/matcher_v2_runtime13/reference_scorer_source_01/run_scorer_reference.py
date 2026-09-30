"""At most two CPU children, completed frozen heads only; no GPU scheduling."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from evaluate_reference_scorer import (MODULES, PACKAGE, READOUT_CODE, REFERENCE_CODE,
    SPLIT, binding, read, require, save, sha, verify_code)

SOURCE = Path(__file__).resolve().parent
JOB_FIELDS = {'arm', 'module', 'source_root', 'engine_root', 'spec', 'preparation', 'training_root',
              'reference_source', 'admission', 'selection'}


def cpu_environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def identity(pid):
    path = Path('/proc')/str(pid)
    stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttime=int(stat[19]), state=stat[0],
        cmdline=(path/'cmdline').read_bytes().replace(b'\0', b' ').decode().strip())


def validate_config(config):
    require(set(config) == {'schema', 'cpu_workers', 'jobs'}
            and config['schema'] == 'reference-scorer-cpu-pipeline/1' and config['cpu_workers'] == 2,
            'registered two-CPU companion configuration required')
    jobs = config['jobs']
    require(len(jobs) == 2 and {job['module'] for job in jobs} == set(MODULES)
            and len({job['arm'] for job in jobs}) == 1 and len({job['selection'] for job in jobs}) == 1,
            'one Patch and one Stats of the same arm and selection policy required')
    for job in jobs:
        require(set(job) == JOB_FIELDS and job['arm'] in ('B0', 'B1', 'B2', 'B3')
                and job['selection'] in ('sim_best', 'equal_budget_endpoint'),
                'registered arm, frozen selection and exact job fields required')
        for field in ('spec', 'preparation'):
            require(set(job[field]) == {'path', 'sha256'} and Path(job[field]['path']).is_absolute(), 'bound '+field+' required')
        require(set(job['admission']) == {'root','complete_sha256'} and Path(job['admission']['root']).is_absolute(),
                'immutable admission required')
        for field in ('source_root','engine_root','training_root','reference_source'):
            require(Path(job[field]).is_absolute(), 'explicit absolute '+field+' required')
    return jobs


def command(job, out):
    values = [sys.executable, str(SOURCE/'evaluate_reference_scorer.py')]
    for key in ('arm','module','source_root','engine_root','training_root','reference_source','selection'):
        values += ['--'+key.replace('_','-'), str(job[key])]
    for key in ('spec','preparation'):
        values += ['--'+key, job[key]['path']]
    return values+['--spec-sha', job['spec']['sha256'], '--admission', job['admission']['root'],
        '--admission-sha', job['admission']['complete_sha256'], '--out', str(out)]


def verify_inputs(job):
    for name in ('spec','preparation'):
        require(binding(job[name]['path']) == job[name], 'bound '+name+' changed')
    require(sha(Path(job['admission']['root'])/'complete.json') == job['admission']['complete_sha256'], 'admission changed')
    verify_code(job['reference_source'], REFERENCE_CODE)
    verify_code(Path(job['engine_root'])/PACKAGE.replace('.', '/')/'curriculum_scorer_eval_v1', READOUT_CODE)


def verify_completed(job, out):
    out = Path(out); require(not (out/'failure.json').exists(), 'evaluation failure precedes completion')
    complete = read(out/'evaluation_complete.json'); audit = read(out/'independent_artifact_audit.json')
    expected_files = {'population.json','protocol.json','pair_predictions.jsonl','prediction_complete.json',
                      'case_diagnostics.jsonl','summary.json','diagnostic_index.json','status.json'}
    require(complete.get('schema') == 'curriculum-scorer-evaluation-complete/1'
            and set(complete.get('files', {})) == expected_files, 'complete artifact membership required')
    require(complete['status'] == 'evaluation_complete' and complete['pairs'] == 900
            and complete['model_state_unchanged'] is True
            and audit['status'] == 'passed' and audit['pairs'] == 900 and audit['diagnostic_cases'] == 30
            and audit['actual_rows_and_numeric_evidence_recomputed'] is True
            and audit['model_state_unchanged'] is True
            and audit['rng_unchanged'] is True and audit['cuda_initialized'] is False
            and audit['scorer_used'] is True and audit['threshold_refitted'] is False,
            'complete independently recomputed900/30 Scorer artifacts required')
    require(all(sha(out/name) == signature for name,signature in complete['files'].items())
            and audit['evaluation_complete_sha256'] == sha(out/'evaluation_complete.json')
            and audit['prediction_sha256'] == sha(out/'pair_predictions.jsonl'), 'terminal output changed')
    origin = complete['provenance']; summary = read(out/'summary.json'); population = read(out/'population.json')
    require(origin['arm'] == audit['arm'] == job['arm'] and origin['module'] == audit['module'] == job['module']
            and origin['selection_kind'] == job['selection'] and origin['split'] == SPLIT
            and origin['source_runtime'] == job['source_root'] and origin['readout_runtime'] == job['engine_root']
            and origin['threshold'] == origin['thresholds']['sim_test']
            and summary['threshold_refitting'] is False, 'reference model/threshold origin changed')
    require(sha(origin['checkpoint']) == origin['checkpoint_sha256'], 'selected model file changed during evaluation')
    raw = [json.loads(line) for line in (out/'pair_predictions.jsonl').read_text().splitlines() if line.strip()]
    labeled = [json.loads(line) for line in (out/'case_diagnostics.jsonl').read_text().splitlines() if line.strip()]
    require(len(raw) == len(labeled) == len(set(population['pair_ids'])) == 900
            and [row['pair_id'] for row in raw] == [row['pair_id'] for row in labeled] == population['pair_ids'],
            'complete unreordered population required')
    require(all('label' not in row and 'gt_known' not in row for row in raw), 'labels entered prediction artifact')
    for group, count in (('all',900),('straight_M',200),('straight_J',400),('straight_R',300)):
        require(len(population['groups'][group]) == summary['groups'][group]['primary']['pairs'] == count,
                'original reference group counts changed')
    return dict(status='complete', arm=job['arm'], module=job['module'],
        evaluation_complete=binding(out/'evaluation_complete.json'), independent_audit=binding(out/'independent_artifact_audit.json'),
        pairs=900, threshold=origin['threshold'], selected_updates=origin['selected_updates'],
        checkpoint_sha256=origin['checkpoint_sha256'], gpu_used=False, automatic_retry=False)


def run_job(job, root):
    work = root/job['module']; work.mkdir()
    try:
        verify_inputs(job); out = work/'evaluation'; values = command(job, out); began = time.time()
        with (work/'evaluate.log').open('xb') as stream:
            child = subprocess.Popen(values, env=cpu_environment(), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT)
            save(work/'launch.json', dict(command=values, process=identity(child.pid), started_unix=began,
                cuda_visible_devices='', automatic_retry=False))
            code = child.wait()
        save(work/'return.json', dict(returncode=code, launch_sha256=sha(work/'launch.json'),
            elapsed_seconds=time.time()-began, automatic_retry=False))
        require(code == 0, 'Scorer reference child failed; preserve output and do not retry')
        verify_inputs(job); result = verify_completed(job, out)
        result['actual_child_return'] = binding(work/'return.json')
        save(work/'complete.json', result)
        return result
    except BaseException as error:
        save(work/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


def verify_preparation(path):
    proof = read(path)
    require(proof['status'] == 'passed' and proof['tests'] >= 20
            and proof['errors'] == proof['failures'] == proof['skipped'] == 0
            and proof['cuda_initialized'] is False and proof['source_unchanged'] is True
            and proof['source_sha256'] == {p.name:sha(p) for p in SOURCE.glob('*.py')}
            and proof['readout_code_sha256'] == READOUT_CODE and proof['reference_code_sha256'] == REFERENCE_CODE,
            'complete source-bound CPU preparation required')


def run(config_path, out, preparation):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only supplemental controller required')
    jobs = validate_config(read(config_path)); verify_preparation(preparation)
    require(not out.exists(), 'exclusive output; no duplicate or retry')
    for job in jobs:verify_inputs(job)
    out.mkdir()
    save(out/'launch.json', dict(controller=identity(os.getpid()), config=binding(config_path),
        preparation=binding(preparation), jobs=jobs, cpu_workers=2, cuda_visible_devices='',
        nice=os.getpriority(os.PRIO_PROCESS,0), automatic_retry=False))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_job, job, out) for job in jobs]
            results = [future.result() for future in futures]
        verify_preparation(preparation)
        save(out/'complete.json', dict(status='complete', results=results, gpu_used=False,
            training_changed=False, threshold_refitted=False, automatic_retry=False, completed_unix=time.time()))
    except BaseException as error:
        save(out/'failure.json', dict(status='failed',error=repr(error),traceback=traceback.format_exc(),
            automatic_retry=False, other_children_not_signalled=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','out','preparation'):parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args(); os.nice(10)
    run(args.config,args.out,args.preparation)


if __name__ == '__main__':main()
