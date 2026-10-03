"""Complete the registered twelve jobs, reusing verified successful old jobs.

Accepted old failures are the strict-heldout seed mismatch and the diagnosed
native frozen-adapter root training flag, before predictions. No blind retries.
"""
import argparse
from pathlib import Path
import traceback

from common import (HERE, PACKAGE, api, read, sha, bound, save, require,
                    check_preparation, check_binding, same_process, environment, source_map, ensure_evaluation_binding)

SEED_ERROR = 'ValueError: actual heldout canonical inputs/targets differ'
MODE_ERROR = 'ValueError: frozen eval Matcher required'


def command(args, population, out, job):
    return [str(args.python), str(HERE/'entry.py'), '--spec', str(args.spec),
        '--population-plan', str(population), '--preparation', str(args.preparation),
        '--controller-root', str(args.controller_root), '--selection', job['selection'],
        '--split', job['split'], '--device', 'cuda:0', '--out', str(out),
        '--repair-preparation', str(args.repair_preparation)]


def exact_flag(command, flag, expected):
    require(command.count(flag) == 1 and command[command.index(flag)+1] == str(expected),
            'command identity differs: '+flag)


def classify_old_job(root, job, verify, identity):
    """Return success / known pre-inference failure / genuinely unattempted."""
    root = Path(root); name = job['name']; out = root/name
    launch_path = root/(name+'_launch.json'); return_path = root/(name+'_return.json')
    if not launch_path.exists():
        require(not return_path.exists() and not out.exists() and not (root/(name+'.log')).exists(),
                'orphaned old evaluation output; manual investigation required')
        return dict(job=job, state='unattempted')
    launch = read(launch_path)
    require(not same_process(launch.get('process'), identity), 'old evaluation process still active')
    require(return_path.is_file(), 'actual old evaluation return missing')
    returned = read(return_path)
    require(returned['launch_sha256'] == sha(launch_path) and launch['job'] == job,
            'old launch/return identity differs')
    if returned['returncode'] == 0:
        return dict(job=job, state='reused_success', verified=verify(root, job))
    require(returned['returncode'] == 1,
            'unrelated or signal-related evaluation failure; do not retry')
    text = (root/(name+'.log')).read_text()
    seed_failure = (job['split'] in ('sim_straight_select','sim_straight_test')
                    and text.rstrip().endswith(SEED_ERROR))
    mode_failure = (job['split'] in ('sim_select','sim_test')
                    and text.rstrip().endswith(MODE_ERROR)
                    and 'curriculum_training_v1/matcher_evaluation.py' in text
                    and 'in predict_batch' in text)
    require(seed_failure or mode_failure, 'not a known canonical seed or frozen eval-mode error')
    # An error after any durable prediction is not the diagnosed constructor bug.
    for path in out.rglob('*') if out.exists() else []:
        # The native runner opens this stream before calling the first batch.
        # Only an empty stream with its exact processed=0 failure is admissible.
        if mode_failure and path.name == 'pair_predictions.jsonl' and path.is_file() and path.stat().st_size == 0:
            failure = read(out/'failure.json')
            require(failure.get('processed') == 0 and failure.get('error') ==
                    "ValueError('frozen eval Matcher required')", 'not a zero-prediction mode failure')
            continue
        require(not (path.is_file() and ('prediction' in path.name or path.name in ('summary.json','evaluation_complete.json'))),
                'failed job already produced predictions; do not replay')
    return dict(job=job, state='known_seed_failure' if seed_failure else 'known_eval_mode_failure', launch=bound(launch_path),
                returned=bound(return_path), log=bound(root/(name+'.log')))


def inspect_old(args, inputs, selected, plan):
    """Failure precedence is relaxed only for the exact audited identity defect."""
    original = api('matcher_v2_v1.evaluation_controller')
    pipeline = api('matcher_v2_v1.pipeline'); launcher = api('curriculum_training_v1.launcher')
    old = Path(args.reuse_evaluation); task = old.parent; module = inputs['plan'].record['module']
    require(old.name == 'evaluation' and task/'training' == args.controller_root,
            'old evaluator must belong to this actual training controller')
    require((task/'failure.json').is_file() and (old/'controller_failure.json').is_file()
            and not (task/'complete.json').exists() and not (old/'evaluation_complete.json').exists(),
            'only a terminal failed old pipeline may be repaired')
    old_launch = read(task/'pipeline_launch.json'); ctrl = read(old/'controller_launch.json')
    require(not same_process(old_launch.get('controller'), launcher.identity)
            and not same_process(ctrl.get('controller'), launcher.identity), 'old controllers still active')
    old_args = argparse.Namespace(**vars(args)); old_args.out = task
    sequence = pipeline.commands(old_args, task, len(args.gpus))
    require(old_launch['commands'] == [list(x) for x in sequence]
            and old_launch['execution_sha256'] == sha(args.spec), 'old pipeline command/binding differs')
    pipeline.verify_child(task, 'training', sequence[0])
    ev_launch = read(task/'evaluation_launch.json'); returned = read(task/'evaluation_return.json')
    require(ev_launch['phase'] == returned['phase'] == 'evaluation'
            and ev_launch['command'] == sequence[1] and returned['returncode'] == 1
            and returned['launch_sha256'] == sha(task/'evaluation_launch.json'), 'old evaluator return differs')
    require(read(task/'failure.json').get('error') ==
            "ChildFailure('evaluation exited1; preserve output and inspect before any explicit recovery')",
            'old pipeline failed outside the known evaluation phase')
    require(not same_process(ev_launch.get('process'), launcher.identity), 'old evaluation child active')
    require(ctrl['execution'] == bound(args.spec) and ctrl['selected_models'] == selected
            and ctrl['training_controller_root'] == str(args.controller_root)
            and ctrl['preparation_sha256'] == sha(args.preparation)
            and ctrl['population_plan_sha256'] == sha(old/'population_plan.json')
            and ctrl['jobs'] == original.jobs(module)
            and read(old/'population_plan.json') == plan, 'old selected models/population changed')
    ledger = []
    for job in original.jobs(module):
        path = old/(job['name']+'_launch.json')
        if path.exists():
            expected = original.command(args.python, args, old/'population_plan.json', old/job['name'], job, module)
            require(read(path)['command'] == expected, 'old evaluator implementation/command differs')
        ledger.append(classify_old_job(old, job, lambda r,j: original.verify_job(r,j,module), launcher.identity))
    diagnosed = ('known_seed_failure', 'known_eval_mode_failure')
    require(any(x['state'] in diagnosed for x in ledger), 'no diagnosed evaluation failure to repair')
    if any(x['state'] == 'known_eval_mode_failure' for x in ledger):
        require(module == 'matcher', 'root eval-mode repair is native Matcher only')
    failures = [dict(job=x['job']['name'],returncode=1) for x in ledger if x['state'] in diagnosed]
    error = read(old/'controller_failure.json').get('error','')
    # execute_queue failure ordering follows actual completion ordering, not
    # necessarily job order. Parse its literal failure list, never eval it.
    import ast
    prefix = 'Scorer evaluation queue incomplete: '
    require(error.startswith('ValueError('), 'unexpected old controller exception type')
    message = ast.literal_eval(error[len('ValueError('):-1])
    require(message.startswith(prefix), 'old controller failed outside its drained job queue')
    observed = ast.literal_eval(message[len(prefix):])
    require(sorted(observed,key=lambda x:x['job'])==sorted(failures,key=lambda x:x['job']),
            'old controller recorded an additional failure')
    # A controller exception with surviving handles must never release the GPU.
    if (old/'live_children_after_controller_error.json').exists():
        raise ValueError('old controller recorded undrained children; manual inspection required')
    return dict(old_root=str(old), pipeline_failure=bound(task/'failure.json'),
        controller_failure=bound(old/'controller_failure.json'), evaluation_return=bound(task/'evaluation_return.json'),
        ledger=ledger, preserved_all_old_outputs=True, successful_jobs_not_repeated=True)


def verify_job(root, job, module):
    root = Path(root); out = root/job['name']; controller = read(root/'controller_launch.json')
    require(job in api('matcher_v2_v1.evaluation_controller').jobs(module), 'unregistered job')
    launch_path = root/(job['name']+'_launch.json'); launch = read(launch_path)
    returned = read(root/(job['name']+'_return.json'))
    require(launch['job'] == job and returned['returncode'] == 0
            and returned['launch_sha256'] == sha(launch_path), 'successful actual repaired child return required')
    args = argparse.Namespace(**controller['arguments'])
    require(launch['command'] == command(args, root/'population_plan.json', out, job),
            'repaired evaluator exact command differs')
    auditor = api('curriculum_training_v1.matcher_run' if module == 'matcher' else 'curriculum_scorer_eval_v1.audit')
    audit = auditor.verify_population(out)
    complete = read(out/'evaluation_complete.json'); origin = complete['provenance']
    wanted = controller['selected_models'][job['selection']]
    for field in ('checkpoint_sha256','model_state_sha256','matcher_state_sha256','selected_updates','arm','module'):
        require(origin[field] == wanted[field], 'actual evaluated model differs: '+field)
    require(origin['selection_kind'] == job['selection'] and origin['split'] == job['split']
            and origin['population_plan_sha256'] == controller['population_plan_sha256']
            and origin['preparation_sha256'] == controller['preparation_sha256']
            and origin['total_pairs'] == read(root/'population_plan.json')['pair_counts'][job['split']],
            'actual evaluation population differs')
    require(read(out/'independent_artifact_audit.json') == audit, 'independent evaluation audit differs')
    return dict(job=job, return_sha256=sha(root/(job['name']+'_return.json')), audit=audit, provenance=origin)


def verify_complete(root):
    root = Path(root)
    require(not (root/'controller_failure.json').exists(), 'repair failure precedes completion')
    receipt = read(root/'evaluation_complete.json'); launch = read(root/'controller_launch.json')
    args = argparse.Namespace(**launch['arguments'])
    for key in ('spec','preparation','controller_root','canonical_straight','case_plan','out','python','repair_preparation','reuse_evaluation'):
        value = getattr(args, key, None)
        if value is not None: setattr(args, key, Path(value))
    check_preparation(args.repair_preparation)
    inputs = api('matcher_v2_v1.runtime_inputs').load_inputs(read(args.spec))
    api('matcher_v2_v1.evaluate').check_preparation(args.preparation, inputs['source'])
    module = inputs['plan'].record['module']; original = api('matcher_v2_v1.evaluation_controller')
    if module != 'matcher': ensure_evaluation_binding(inputs['source'])
    choices = dict.fromkeys(j['selection'] for j in original.jobs(module))
    selected = {c:api('matcher_v2_v1.terminal').verified_export(args.controller_root,args.spec,inputs['plan'],c)[1]
                for c in choices}
    require(launch['selected_models'] == selected and launch['source_files'] == source_map()
            and launch['repair_preparation'] == bound(args.repair_preparation), 'repair/model preparation changed')
    plan = read(root/'population_plan.json')
    api('matcher_v2_v1.population').validate_plan(plan, args.spec, inputs['source'])
    require(launch['population_plan_sha256'] == sha(root/'population_plan.json')
            and receipt['controller_launch_sha256'] == sha(root/'controller_launch.json')
            and receipt['population_plan_sha256'] == launch['population_plan_sha256']
            and receipt['schema'] == 'matcher-v2-seed-repaired-evaluation/1' and receipt['status'] == 'complete'
            and receipt['module'] == module and receipt['arm'] == read(args.spec)['arm'], 'repair receipt differs')
    old = inspect_old(args, inputs, selected, plan) if args.reuse_evaluation else None
    require(launch['old_evaluation'] == old, 'old immutable evaluation evidence changed')
    reused = {x['job']['name']:x['verified'] for x in old['ledger'] if x['state'] == 'reused_success'} if old else {}
    items = receipt['jobs']; require(len(items) == len(original.jobs(module)) == receipt['job_count'], 'missing jobs')
    require([x['job'] for x in items] == original.jobs(module), 'ordered complete job registry differs')
    for item in items:
        job = item['job']; is_reused = job['name'] in reused
        expected = reused[job['name']] if is_reused else verify_job(root,job,module)
        require(item == dict(job=job, reused=is_reused, root=str(args.reuse_evaluation if is_reused else root),
                             result=expected), 'job artifact/return evidence changed')
    return receipt


def run(args):
    for key in ('spec','preparation','controller_root','canonical_straight','case_plan','out','python','repair_preparation','reuse_evaluation'):
        value = getattr(args,key,None)
        if value is not None: setattr(args,key,Path(value).resolve())
    check_preparation(args.repair_preparation)
    inputs = api('matcher_v2_v1.runtime_inputs').load_inputs(read(args.spec)); source = inputs['source']
    original = api('matcher_v2_v1.evaluation_controller'); launcher = api('curriculum_training_v1.launcher')
    original.check_preparation(args.preparation,source); module = inputs['plan'].record['module']
    if module != 'matcher': ensure_evaluation_binding(source)
    planned = original.jobs(module)
    selected = {c:original.verified_export(args.controller_root,args.spec,inputs['plan'],c)[1]
                for c in dict.fromkeys(j['selection'] for j in planned)}
    plan = original.freeze_plan(args.spec,bound(args.canonical_straight),args.case_plan,source)
    old = inspect_old(args,inputs,selected,plan) if args.reuse_evaluation else None
    require(args.python.is_file() and len(args.gpus) in (1,2) and not args.out.exists(), 'explicit fresh evaluator required')
    devices = launcher.check_free(args.gpus,len(args.gpus))
    root = args.out; root.mkdir(parents=True)
    original.write_json(root/'population_plan.json',plan)
    arguments = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    save(root/'controller_launch.json', dict(controller=launcher.identity(__import__('os').getpid()),
        arguments=arguments,module=module,execution=bound(args.spec),selected_models=selected,gpu_devices=devices,
        population_plan_sha256=sha(root/'population_plan.json'),preparation_sha256=sha(args.preparation),
        source_files=source_map(),repair_preparation=bound(args.repair_preparation),old_evaluation=old,
        automatic_retry=False,threshold_fitting=False,training_changes=False))
    try:
        reused = {x['job']['name']:x['verified'] for x in old['ledger'] if x['state']=='reused_success'} if old else {}
        pending = [j for j in planned if j['name'] not in reused]
        results = original.execute_queue(root,args.gpus,environment(source),
            lambda j,out:command(args,root/'population_plan.json',out,j), registered_jobs=pending,
            verify=lambda r,j:verify_job(r,j,module)) if pending else []
        fresh = {x['job']['name']:x for x in results}
        receipt = dict(schema='matcher-v2-seed-repaired-evaluation/1',status='complete',arm=read(args.spec)['arm'],
            module=module,job_count=len(planned),controller_launch_sha256=sha(root/'controller_launch.json'),
            population_plan_sha256=sha(root/'population_plan.json'),automatic_retry=False,threshold_fitting=False,
            jobs=[dict(job=j,reused=j['name'] in reused,root=str(args.reuse_evaluation if j['name'] in reused else root),
                       result=reused[j['name']] if j['name'] in reused else fresh[j['name']]) for j in planned])
        save(root/'evaluation_complete.json',receipt)
        verify_complete(root)
        original.write_json(root/'driver_status.json',dict(status='complete',completed=len(planned),active={},pending=[]),replace=True)
        return receipt
    except BaseException as error:
        save(root/'controller_failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('spec','preparation','controller-root','canonical-straight','case-plan','out','python','repair-preparation'):
        p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--reuse-evaluation',type=Path)
    p.add_argument('--gpus',type=int,nargs='+',required=True)
    run(p.parse_args())


if __name__ == '__main__': main()
