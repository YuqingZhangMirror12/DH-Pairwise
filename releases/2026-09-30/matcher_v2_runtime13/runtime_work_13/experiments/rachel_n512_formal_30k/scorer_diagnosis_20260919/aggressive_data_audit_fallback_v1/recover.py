"""One CPU-only recovery: audit rejection -> same-pair v14, never relaxed rules."""
import argparse
import fcntl
import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from fallback_group import read, sha, save, eligible, archive, build, promote, GAP_ERROR

BASE = Path('/root/autodl-tmp/aggressive_data_v17_full30k_20260927')
DATA = BASE / 'dataset_03'
ROOT = BASE / 'recovery_audit_fallback_01'
PKG = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.aggressive_data_full_v17'
ADMISSION = Path('/root/autodl-tmp/scorer_queue_20260928/admissions/aggressive_scratch.json')
REGISTER = Path('/root/autodl-tmp/aggressive_binary_20260927/queued_source_02/register.py')


def previous():
    if 'v17_numeric_recovery_bound' not in sys.modules:
        spec = importlib.util.spec_from_file_location('v17_numeric_recovery_bound', BASE / 'recovery_numeric_source_01/recover.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules['v17_numeric_recovery_bound']


def inventory():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def validate():
    old = previous()
    if sha(DATA / 'config.json') != old.CONFIG_SHA or sha(old.SOURCE / 'source_binding.json') != old.SOURCE_SHA:
        raise ValueError('generator config/source binding changed')
    for name, digest in read(old.SOURCE / 'source_binding.json').items():
        if sha(old.SOURCE / name) != digest:
            raise ValueError('immutable generator changed')
    old.validate_cpu()
    if (DATA / 'pipeline_complete.json').exists() or ADMISSION.exists():
        raise ValueError('already complete/admitted')
    for path in [DATA / 'pipeline_launch.json'] + list(DATA.glob('pipeline_resume_*.json')):
        if old.live(read(path)):
            raise ValueError('another data owner still alive')
    error = "AssertionError(" + repr(GAP_ERROR) + ")"
    if read(DATA / 'pipeline_failure.json').get('error') != error:
        raise ValueError('unexpected failure; no fallback recovery')
    if read(BASE / 'recovery_numeric_01/failure.json').get('error') != error:
        raise ValueError('observed audit failure not bound')
    sys.path.insert(0, str(old.SOURCE))
    return read(DATA / 'config.json')


def audit_module():
    return previous().corrected_auditor()


def preflight():
    config = validate()
    output = ROOT / 'preflight.json'
    if output.exists():
        raise ValueError('preflight already recorded')
    source = importlib.import_module(PKG + '.source')
    source.initialize(config, 'test')
    path = DATA / 'test/groups/00183.json'
    before = sha(path)
    group = read(path)
    try:
        audit_module().audit_record(group['records'][0], source.STATE['root'])
    except AssertionError as error:
        if not eligible(error, group):
            raise
    else:
        raise ValueError('observed case no longer fails')
    stage = ROOT / 'preflight_staging/test'
    retained = build(group, stage)
    rows = [audit_module().audit_record(r, source.STATE['root']) for r in retained['records']]
    if sha(path) != before:
        raise ValueError('preflight modified actual dataset')
    save(output, dict(status='passed',original_group_sha256=before,
                      original_record_id=group['records'][0]['id'], original_error=GAP_ERROR,
                      retained_rows=rows, source_sha256=inventory(),
                      observed_recomputed_gap_px=4.999999797050212,
                      original_dataset_changed=False, new_geometry_attempts=0,
                      gap_bounds_or_tolerances_changed=False, gpu_used=False))
    print('Actual rejected pair -> v14 numerical identity and both supervision audits passed.', flush=True)


def initialize(config, split):
    source = importlib.import_module(PKG + '.source')
    source.initialize(config, split)
    importlib.import_module(PKG + '.pipeline').audit_record = audit_module().audit_record


def audited_group(task):
    source = importlib.import_module(PKG + '.source')
    split = source.STATE['split']
    group_path = DATA / split / 'groups' / f'{task["slot"]:05d}.json'
    result = DATA / split / 'audits' / group_path.name
    group_hash = sha(group_path)
    if result.exists():
        record = read(result)
        if record.get('status') != 'passed' or record.get('group_sha256') != group_hash:
            raise ValueError('audit resume mismatch')
        return dict(slot=task['slot'],status='passed',cached=True)
    group = read(group_path)
    rows = [audit_module().audit_record(r,source.STATE['root']) for r in group['records']]
    save(result,dict(status='passed',group_sha256=group_hash,rows=rows))
    return dict(slot=task['slot'],status='passed')


def audit_task(task):
    source = importlib.import_module(PKG + '.source')
    split = source.STATE['split']
    path = DATA / split / 'groups' / f'{task["slot"]:05d}.json'
    group = read(path)
    try:
        return audited_group(task)
    except AssertionError as error:
        if not eligible(error, group):
            raise
    # Both members stay paired; no new v17 search and no replacement sources.
    history = ROOT / 'rejected' / split / path.stem
    preserved = archive(path, history)
    stage = ROOT / 'staging' / split / path.stem
    retained = build(group, stage)
    rows = [audit_module().audit_record(r, source.STATE['root']) for r in retained['records']]
    if not all(r.get('baseline_numerical_identity') for r in rows):
        raise ValueError('same original v14 arrays not established')
    promote(path, retained, stage, history)
    result = audited_group(task)
    save(ROOT / 'replacements' / split / path.name, dict(status='passed',
         original_group_sha256=preserved['groups/' + path.name], retained_group_sha256=sha(path),
         history=str(history), history_manifest_sha256=sha(history / 'preserved.json'),
         audit_sha256=sha(DATA / split / 'audits' / path.name),
         same_original_pair=True, new_geometry_attempts=0, bounds_or_tolerances_changed=False))
    return result


def check_preserved():
    replaced = {str(p.parent.name + '/groups/' + p.name): read(p)
                for p in (ROOT / 'replacements').glob('*/*.json')}
    for name, digest in read(ROOT / 'preexisting.json').items():
        if name in replaced:
            record = replaced[name]
            if record['original_group_sha256'] != digest or sha(DATA / name) != record['retained_group_sha256']:
                raise ValueError('recorded fallback identity differs')
            history = Path(record['history'])
            for rel, old_hash in read(history / 'preserved.json').items():
                if sha(history / rel) != old_hash:
                    raise ValueError('rejected history changed')
        elif sha(DATA / name) != digest:
            raise ValueError('unrelated existing group/audit changed: ' + name)
    return replaced


def finish(config):
    result = importlib.import_module(PKG + '.finish').finish(config)
    replacements = check_preserved()
    provenance = dict(schema='same-pair-v14-post-audit-fallback/1',
         explicit_user_fallback_authorization=True, replaced_groups=len(replacements),
         replaced_pairs=2*len(replacements), replacements=replacements,
         source_sha256=inventory(), original_generator_unchanged=True,
         accepted_prior_groups_and_audits_unchanged=True, audit_bounds_or_tolerances_changed=False,
         numeric_audit_preparation_sha256=sha(BASE / 'recovery_numeric_01/cpu_tests_remote.json'),
         independent_auditor_sha256=sha(BASE / 'recovery_numeric_source_01/audit.py'),
         preflight_sha256=sha(ROOT / 'preflight.json'))
    audit = read(DATA / 'full_audit.json')
    audit['same_pair_postaudit_fallback'] = provenance
    save(DATA / 'full_audit.json', audit)
    contract = read(DATA / 'data_contract.json')
    contract['aggressive_full_audit']['sha256'] = sha(DATA / 'full_audit.json')
    contract['same_pair_postaudit_fallback'] = provenance
    save(DATA / 'data_contract.json', contract)
    save(ROOT / 'preservation.json', dict(status='passed', **provenance))
    return result


def validate_tests():
    record = read(ROOT / 'cpu_tests_remote.json')
    if (record.get('status') != 'passed' or record.get('tests', 0) < 8
            or any(record.get(k) != 0 for k in ('failures','errors','skipped'))
            or record['source_sha256'] != inventory()):
        raise ValueError('fallback tests not bound to source')
    pre = read(ROOT / 'preflight.json')
    if pre.get('status') != 'passed' or pre['source_sha256'] != inventory():
        raise ValueError('actual pair preflight not bound')


def driver():
    with (ROOT / 'recovery.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_tests()
        config = validate()
        if (ROOT / 'history').exists():
            raise ValueError('already attempted; no automatic retry')
        save(ROOT / 'preexisting.json', previous().files_to_preserve())
        (ROOT / 'history').mkdir()
        for name in ('pipeline_failure.json','pipeline_status.json'):
            shutil.copy2(DATA / name, ROOT / 'history' / name)
            if sha(DATA / name) != sha(ROOT / 'history' / name):
                raise ValueError('failure history copy differs')
        (DATA / 'pipeline_failure.json').unlink()
        pipeline = importlib.import_module(PKG + '.pipeline')
        pipeline.audit_task = audit_task
        pipeline.initialize = initialize
        pipeline.audit_record = audit_module().audit_record
        pipeline.finish = finish
        save(ROOT / 'status.json', dict(status='resuming_unfinished_audit',controller=previous().identity(os.getpid()),gpu_used=False))
        sys.argv = ['same-pair-v14-audit-recovery','--out',str(DATA),'--resume']
        pipeline.main()
        complete = read(DATA / 'pipeline_complete.json')
        if complete.get('status') != 'complete' or complete.get('pairs') != 30000:
            raise ValueError('complete30K missing')
        check_preserved()
        save(ROOT / 'status.json',dict(status='full_data_complete_registering',gpu_used=False))
        if ADMISSION.exists():
            raise ValueError('another owner admitted data')
        with (ROOT / 'registration.log').open('xb') as log:
            job = subprocess.Popen([sys.executable,str(REGISTER)],env=previous().environment(),stdout=log,stderr=subprocess.STDOUT)
            save(ROOT / 'registration_launch.json',dict(controller=previous().identity(job.pid)))
            code = job.wait()
        save(ROOT / 'registration_exit.json',dict(returncode=code,finished_unix=time.time()))
        if code or not ADMISSION.exists():
            raise ValueError('registration failed; no automatic retry')
        done = dict(status='complete',data_complete_sha256=sha(DATA / 'pipeline_complete.json'),
                    admission_sha256=sha(ADMISSION),finished_unix=time.time(),gpu_tasks_started=False)
        save(ROOT / 'complete.json',done)
        save(ROOT / 'status.json',done)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--driver',action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only fallback recovery')
    if args.preflight:
        preflight()
    elif args.driver:
        try:
            driver()
        except BaseException as error:
            result = dict(status='failed',error=repr(error),traceback=traceback.format_exc(),automatic_retries=0)
            save(ROOT / 'failure.json',result)
            save(ROOT / 'status.json',result)
            raise
    else:
        validate_tests()
        validate()
        if any((ROOT / name).exists() for name in ('controller_launch.json','failure.json','complete.json')):
            raise ValueError('already registered; no repeat')
        command = [sys.executable,str(Path(__file__).resolve()),'--driver']
        with (ROOT / 'controller.log').open('xb') as log:
            proc = subprocess.Popen(command,env=previous().environment(),stdin=subprocess.DEVNULL,
                                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result = dict(controller=previous().identity(proc.pid),command=command,created_unix=time.time(),automatic_retries=0)
        save(ROOT / 'controller_launch.json',result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
