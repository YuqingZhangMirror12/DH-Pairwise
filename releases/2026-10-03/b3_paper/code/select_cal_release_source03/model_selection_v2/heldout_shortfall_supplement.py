"""One user-authorized, bounded supplement; never edits the running build.

Only CAL/Gen3/quota19/v17_filtered is eligible. Four preregistered extra
candidate groups run on four CPU threads/processes, each for at most 15 min.
Geometry remains native. Expected rejection/time exhaustion may leave the
explicitly authorized two-pair deficit; unexpected failures never qualify.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = Path('/root/autodl-tmp/model_selection_v2_20261002')
BUILD = BASE / 'curriculum_build_05'
ROOT = BASE / 'shortfall_cal_gen3_q19_01'
FROZEN = BASE / 'curriculum_source_07/model_selection_v2'
TARGET = ('cal', 'Gen3', 19, 'v17_filtered')
SHARD_SHA = '598c11de5065ebd0ad9163927ef7c3b336f2348a6356bd0bc98bf43fe9894300'
MAX_CANDIDATES = 4
TIMEOUT = 900
THREAD_ENV = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                  PYTHONDONTWRITEBYTECODE='1')
SCHEDULE_FIELDS = ('recipes', 'partial', 'partial_modes', 'bins', 'mirrors')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve(strict=True)), sha256=sha(path))


def checked(receipt):
    require(sha(receipt['path']) == receipt['sha256'], 'registered input changed: '+receipt['path'])
    return read(receipt['path'])


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, starttime=int(stat[19]), state=stat[0],
                    cmdline=proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode().strip())
    except FileNotFoundError:
        return dict(pid=pid, exited_before_identity_capture=True)


def make_extension(generation, sources, shortfall, extension):
    require(shortfall['status'] == 'shortfall' and shortfall['role'] == 'cal'
            and shortfall['shard']['name'] == 'cal_Gen3_02'
            and shortfall['planned_quotas'] == 45 and shortfall['admitted_pairs'] == 88,
            'only the approved 88/90 CAL shard may be supplemented')
    require(Counter(r['label'] for r in shortfall['records']) == Counter({True:44, False:44}),
            'original admitted positive/negative count differs')
    exhausted = [f['quota'] for f in shortfall['failures'] if f['phase'] == 'quota_exhausted']
    require(exhausted == [['Gen3', 19, 'v17_filtered']], 'different or multiple exhausted quotas')
    tasks = sorted((t for t in generation['tasks'] if
                    (t['role'], t['generator'], t['quota_slot'], t['stage']) == TARGET),
                   key=lambda t:t['reserve_index'])
    require([t['reserve_index'] for t in tasks] == list(range(12)), 'old 12 reserves required')
    fixed = dict(role='cal', generator='Gen3', quota_slot=19, stage='v17_filtered',
                 recipe='partial', size_class='smaller', mode='one', k=4)
    require(all(all(t[k] == v for k, v in fixed.items()) for t in tasks), 'approved quota attributes differ')
    failed = [f['task'] for f in shortfall['failures'] if f.get('task') in tasks]
    require(sorted(failed, key=lambda t:t['reserve_index']) == tasks, 'not all registered candidates rejected')
    spec = sources['splits']['cal']
    n = len(spec['positive'])
    require(len(spec['negative']) == len(spec['slot_generators']) == n and spec['pairs'] == 2*n
            and all(len(spec['schedule'][name]) == n for name in SCHEDULE_FIELDS),
            'original CAL slot arrays differ')
    require(all(t['master_seed'] == generation['master_seed'] and t['base_pair_ids'] ==
                [spec['positive'][t['slot']]['pair_id'],spec['negative'][t['slot']]['pair_id']]
                for t in tasks), 'old task seed or source binding differs')
    cal_families = set(spec['source_families'])
    forbidden = set(sources['forbidden_sources']) | set(sources['splits']['select']['source_families'])
    require(not cal_families & forbidden, 'original CAL source families are not isolated')
    positive = extension._pool(spec['positive_pool'], 'Gen3', True)
    negative_rows = extension._pool([e['row'] for e in spec['negative']], 'Gen3', False)
    negative = {e['pair_id']:e for e in spec['negative'] if e['pair_id'] in negative_rows}
    old_pairs = [tuple(t['base_pair_ids']) for t in tasks]
    extra = extension.append_candidates(positive, negative, old_pairs,
        seed=generation['master_seed'], role='cal', gen='Gen3', quota=19, count=MAX_CANDIDATES)
    expanded = copy.deepcopy(sources)
    target = expanded['splits']['cal']
    template = tasks[0]
    added = []
    for index, draw in enumerate(extra):
        p, n = draw['positive_pair_id'], draw['negative_pair_id']
        require((p, n) not in old_pairs and (p, n) not in [tuple(t['base_pair_ids']) for t in added],
                'only unused same-pool candidate combinations allowed')
        slot = len(target['positive'])
        target['positive'].append(dict(pair_id=p, source_stratum='native_positive'))
        target['negative'].append(copy.deepcopy(negative[n]))
        target['slot_generators'].append('Gen3')
        for name in SCHEDULE_FIELDS:
            target['schedule'][name].append(copy.deepcopy(spec['schedule'][name][template['slot']]))
        task = dict(template, slot=slot, reserve_index=12+index, base_pair_ids=[p, n],
                    trim_target=extension.trim_target(generation['master_seed'], 'cal', 'v17_filtered', p))
        draw['reserve_index'] = task['reserve_index']
        draw['slot'] = slot
        added.append(task)
    target['pairs'] = 2*len(target['positive'])
    target['generator_counts']['Gen3']['candidate_groups'] += MAX_CANDIDATES
    registration = dict(schema='one-quota-heldout-supplement/1', target=list(TARGET), tasks=added,
        draws=extra, max_candidates=MAX_CANDIDATES, per_candidate_seconds=TIMEOUT,
        requested_added_pairs=2, positive_added_limit=1, negative_added_limit=1,
        user_allows_unfilled_target=True, approved_missing_pairs=2,
        missing_policy='Only this quota may remain missing after the bounded batch; report actual counts.',
        source_pool_unchanged=True, old_slots_unchanged=True, geometry_gates_unchanged=True,
        original_baseline_attempts=1024, original_augmentation_attempts=24,
        selection_order='first registered accepted candidate, independent of completion order',
        final_cross_fold_dedup_and_release_pending=True, train_test_or_model_outputs_used=False)
    return expanded, registration


def modules(runtime):
    sys.path[:0] = [str(FROZEN), str(runtime)]
    return (importlib.import_module('heldout_extend_plan'),
            importlib.import_module('heldout_reduce_run'),
            importlib.import_module('heldout_run'),
            importlib.import_module('heldout_augment'))


def verify_context(root):
    context = read(root/'context.json')
    require(context['script'] == ref(__file__), 'supplement source changed')
    for path, expected in context['external_sources'].items():
        require(sha(path) == expected, 'frozen source07 file changed')
    for receipt in context['inputs'].values():
        checked(receipt)
    registration = checked(context['registration'])
    source_plan = checked(context['sources'])
    require(registration['target'] == list(TARGET) and len(registration['tasks']) == MAX_CANDIDATES
            and registration['per_candidate_seconds'] == TIMEOUT, 'supplement boundary changed')
    return context, registration, source_plan


def prepare(root):
    require(root == ROOT and not root.exists(), 'unique new supplement directory required; no retry')
    launch_ref = ref(BUILD/'controller_launch.json')
    launch = checked(launch_ref)
    projection = checked(launch['reduction_plan'])
    gen_ref = projection['original_generation_plan']
    generation = checked(gen_ref)
    sources_ref = dict(path=generation['source_plan_path'], sha256=generation['source_plan_sha256'])
    sources = checked(sources_ref)
    short_ref = dict(path=str(BUILD/'shard_receipts/cal_Gen3_02/complete.json'), sha256=SHARD_SHA)
    shortfall = checked(short_ref)
    external = {str(p):sha(p) for p in FROZEN.glob('*.py')}
    require(len(external) == 9, 'unexpected source07 inventory')
    for path, actual in external.items():
        if path in launch['bindings']:
            require(actual == launch['bindings'][path], 'source07 differs from original launch')
    extension, _, _, _ = modules(launch['frozen_runtime'])
    expanded, registration = make_extension(generation, sources, shortfall, extension)
    cpu = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    quota = float(cpu[0])/int(cpu[1]) if cpu[0] != 'max' else len(os.sched_getaffinity(0))
    memory = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    free_memory = int(memory)-int(Path('/sys/fs/cgroup/memory.current').read_text()) if memory != 'max' else 10**12
    fs = os.statvfs(BASE)
    require(quota >= 48 and free_memory >= 8*1024**3 and fs.f_bavail*fs.f_frsize >= 5*1024**3,
            'insufficient safe headroom for four CPU-only candidates')
    root.mkdir()
    save(root/'sources.json', expanded)
    save(root/'registration.json', registration)
    save(root/'context.json', dict(script=ref(__file__), runtime=launch['frozen_runtime'],
        external_sources=external, registration=ref(root/'registration.json'), sources=ref(root/'sources.json'),
        inputs=dict(launch=launch_ref, reduction=launch['reduction_plan'], generation=gen_ref,
                    sources=sources_ref, shortfall=short_ref, inventory=launch['frozen_source_inventory']),
        prepared_unix=time.time(), resource_admission=dict(cpu_quota=quota, free_memory=free_memory,
            added_cpu_workers=4, nice=10, cuda_visible_devices=''),
        old_build_unmodified=True, other_quota_waivers_authorized=False))
    return root


def candidate(root, index):
    os.environ.update(THREAD_ENV)
    os.nice(10)
    context, registration, sources = verify_context(root)
    require(0 <= index < MAX_CANDIDATES, 'candidate index out of scope')
    task = registration['tasks'][index]
    _, reduced, original, aug = modules(context['runtime'])
    prefix = original.PREFIX
    heldout = importlib.import_module(prefix+'.s7_consensus_v1.heldout_v14')
    material = importlib.import_module(prefix+'.s7_balanced_v2.materialize')
    audit = importlib.import_module(prefix+'.s7_balanced_v2.audit_layered')
    source = importlib.import_module(prefix+'.aggressive_data_full_v17.source')
    out = root/'candidates'/str(index)
    baseline_root = out/'baseline/cal'
    baseline_root.mkdir(parents=True)
    inventory = checked(context['inputs']['inventory'])
    bound = original.bind_loaded_sources()
    original.validate_loaded_inventory(bound, inventory)
    heldout.initialize(dict(sources=context['sources']['path'], split='cal', out=str(baseline_root),
        seed=task['master_seed']+10001, attempts=1024))
    try:
        baseline = reduced.fixed_source_slot(heldout, material, task['slot'])
    except RuntimeError as error:
        if not original.expected_baseline_exhaustion(error, task):
            raise
        result = dict(status='rejected', phase='baseline_geometry_exhausted', error=str(error))
    else:
        require(reduced.baseline_identity(baseline, task, sources['splits']['cal']), 'registered positive replaced')
        rows = [audit.one((str(baseline_root), e)) for e in baseline['entries']]
        save(out/'baseline_admission.json', dict(root=str(baseline_root),
            group=ref(baseline_root/'groups'/f"{task['slot']:05d}.json"), audit_rows=rows))
        config = dict(out=str(out/'augmented'), generation_plan_sha256=context['registration']['sha256'],
                      geometry_attempts=24)
        source.STATE.clear()
        source.baseline.cache_clear()
        source.STATE.update(config=config, split='cal', root=baseline_root,
            groups={task['slot']:{i:e for i,e in enumerate(baseline['entries'])}}, bank=material.STATE['bank'],
            negative={e['pair_id']:e for e in material.STATE['negative_plan']})
        aug.STATE.clear()
        aug.STATE.update(config=config, role='cal', source=source)
        generated = aug.process(task)
        if generated['status'] == 'committed':
            require(Counter(r['label'] for r in generated['records']) == Counter({True:1, False:1})
                    and len({r['model_tensors_sha256'] for r in generated['records']}) == 2,
                    'supplement is not a unique balanced pair group')
            result = dict(status='candidate_ready_pending_release',
                commit=ref(out/'augmented/cal/v17_filtered/groups'/f"{task['slot']:05d}.json"),
                baseline_admission=ref(out/'baseline_admission.json'))
        else:
            require(generated['status'] == 'rejected', 'unexpected augmentation result')
            result = dict(status='rejected', phase='augmentation', reasons=generated['reasons'])
    original.check_sources(bound)
    later = original.bind_loaded_sources()
    original.validate_loaded_inventory(later, inventory)
    verify_context(root)
    save(out/'outcome.json', dict(result, task=task, registration=context['registration'],
        frozen_sources=later, finished_unix=time.time(), gpu_used=False, geometry_relaxed=False))
    return 0


def run_candidate(root, index):
    out = root/'candidates'/str(index)
    out.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), '--candidate', str(index), '--root', str(root)]
    with (out/'process.log').open('x') as log:
        began = time.time()
        process = subprocess.Popen(command, env=dict(os.environ, **THREAD_ENV), stdout=log, stderr=log)
        save(out/'launch.json', dict(identity(process.pid), command=command, started_unix=began,
                                    timeout_seconds=TIMEOUT, context=ref(root/'context.json')))
        timed_out = False
        try:
            rc = process.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()  # Still-owned, unreaped child; never a PID lookup or broad kill.
            rc = process.wait()
        actual = dict(command=command, returncode=rc, started_unix=began, finished_unix=time.time(),
                      timed_out=timed_out, timeout_seconds=TIMEOUT)
        save(out/'actual_return.json', actual)
    if timed_out:
        save(out/'timeout.json', dict(status='bounded_time_budget_exhausted',
            partial_outputs_preserved=True, partial_outputs_admitted=False, automatic_retry=False))
        return dict(index=index, status='bounded_time_budget_exhausted', actual_return=ref(out/'actual_return.json'))
    if rc != 0:
        return dict(index=index, status='unexpected_failure', actual_return=ref(out/'actual_return.json'))
    outcome = read(out/'outcome.json')
    require(outcome['status'] in ('rejected', 'candidate_ready_pending_release'), 'unknown terminal result')
    return dict(index=index, status=outcome['status'], actual_return=ref(out/'actual_return.json'),
                outcome=ref(out/'outcome.json'))


def summarize(results):
    require(len(results) == MAX_CANDIDATES and sorted(r['index'] for r in results) == list(range(MAX_CANDIDATES)),
            'all bounded candidate attempts must have a real return')
    results = sorted(results, key=lambda r:r['index'])
    require(not any(r['status'] == 'unexpected_failure' for r in results),
            'unexpected failure is not an authorized geometry shortfall')
    require(all(r['status'] in ('rejected', 'bounded_time_budget_exhausted', 'candidate_ready_pending_release')
                for r in results), 'unrecognized candidate status')
    ready = [r for r in results if r['status'] == 'candidate_ready_pending_release']
    return dict(status='supplement_ready_pending_release' if ready else 'bounded_batch_exhausted_shortfall_authorized',
        candidates=results, candidate_priority=[r['index'] for r in ready],
        provisional_selected_index=ready[0]['index'] if ready else None,
        added_pairs_limit=2, unresolved_missing_pairs=0 if ready else 2,
        final_cross_fold_dedup_and_release_pending=True, automatic_retry=False)


def batch(root):
    verify_context(root)
    with ThreadPoolExecutor(max_workers=MAX_CANDIDATES) as pool:
        results = list(pool.map(lambda index:run_candidate(root, index), range(MAX_CANDIDATES)))
    verify_context(root)
    save(root/'batch_result.json', dict(summarize(results), finished_unix=time.time()))
    return 0


def controller(root):
    verify_context(root)
    save(root/'controller_identity.json', dict(identity(os.getpid()), context=ref(root/'context.json')))
    command = [sys.executable, str(Path(__file__).resolve()), '--batch', '--root', str(root)]
    began = time.time()
    rc = subprocess.run(command, env=dict(os.environ, **THREAD_ENV)).returncode
    save(root/'actual_return.json', dict(command=command, returncode=rc, started_unix=began,
        finished_unix=time.time(), automatic_retry=False))
    if rc:
        if not (root/'failure.json').exists():
            save(root/'failure.json', dict(status='supplement_failed', returncode=rc, automatic_retry=False))
        return rc
    verify_context(root)
    result = read(root/'batch_result.json')
    save(root/'complete.json', dict(result, actual_return=ref(root/'actual_return.json'),
        batch_result=ref(root/'batch_result.json'), context=ref(root/'context.json')))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--launch', action='store_true')
    modes.add_argument('--controller', action='store_true')
    modes.add_argument('--batch', action='store_true')
    modes.add_argument('--candidate', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    require(root == ROOT, 'dedicated supplement output required')
    os.environ.update(THREAD_ENV)
    if args.launch:
        prepare(root)
        command = [sys.executable, str(Path(__file__).resolve()), '--controller', '--root', str(root)]
        with (root/'controller.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True,
                                       env=dict(os.environ, **THREAD_ENV))
        launch = dict(identity(process.pid), command=command, started_unix=time.time(),
                      context=ref(root/'context.json'), restart_policy='never')
        save(root/'launch.json', launch)
        print(json.dumps(dict(root=str(root), launch=launch), ensure_ascii=False))
        return 0
    try:
        if args.controller:
            return controller(root)
        if args.batch:
            return batch(root)
        return candidate(root, args.candidate)
    except BaseException as error:
        target = root/'candidates'/str(args.candidate) if args.candidate is not None else root
        path = target/'failure.json'
        if not path.exists():
            save(path, dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
                            actual_identity=identity(os.getpid()), automatic_retry=False))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
