"""User-authorized one-shot physical-batch migration at a committed segment.

Does not alter frozen queue configuration, experimental data, or checkpoints
by hand. Every replacement queue uses its existing explicit resume interface.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def command(pid):
    path = Path('/proc') / str(pid) / 'cmdline'
    return path.read_bytes().decode().strip('\0').split('\0') if path.exists() else []


def live(pid, marker):
    return any(marker in item for item in command(pid))


def read(path):
    return json.loads(Path(path).read_text())


def wait_for(fn, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(.25)
    raise TimeoutError('condition did not become true')


def run(args):
    work = Path(args.work)
    training = Path(args.training)
    trainer_pid = args.trainer_pid
    parents = args.queue_pids
    queue_marker = 'experiments.rachel_n512_formal_30k.run_recall_benchmark_queue'
    trainer_marker = 'experiments.rachel_n512_formal_30k.train_score_decoupled'
    if len(parents) != 4 or len(set(parents + [trainer_pid])) != 5:
        raise ValueError('expected one trainer and the exact four distinct queue handles')
    if not live(trainer_pid, trainer_marker):
        raise RuntimeError('original trainer not live')
    configs = []
    for pid in parents:
        argv = command(pid)
        if queue_marker not in argv:
            raise RuntimeError('queue handle changed: ' + str(pid))
        path = Path(argv[argv.index('--config') + 1])
        config = read(path)
        state = read(Path(config['root']) / 'queue_state.json')
        if state['config'] != config or state['pid'] != pid:
            raise RuntimeError('queue source/state differs')
        configs.append((path, config))
    dependency_names = [None]
    for index in range(1, len(configs)):
        matches = [row for row in configs[index][1]['dependencies']
                   if row['marker'] == str(configs[index - 1][0])]
        if len(matches) != 1:
            raise RuntimeError('tail must have exactly one dependency on its preceding queue')
        dependency_names.append(matches[0]['name'])
    state = read(Path(configs[0][1]['root']) / 'queue_state.json')
    if state.get('active_child', {}).get('pid') != trainer_pid:
        raise RuntimeError('trainer is not the active child of the expected queue')
    initial = read(training / 'status.json')
    if initial.get('phase') != 'matcher' or initial.get('pid') != trainer_pid:
        raise RuntimeError('unexpected training phase')
    target = initial['global_exposure'] + 6000
    print(json.dumps(dict(event='waiting_for_next_committed_segment', target_exposure=target)), flush=True)
    deadline = time.monotonic() + args.max_wait_seconds
    while time.monotonic() < deadline:
        if not live(trainer_pid, trainer_marker):
            raise RuntimeError('trainer stopped before boundary; not restarting automatically')
        status = read(training / 'status.json')
        if status.get('phase') == 'matcher' and status.get('global_exposure', 0) >= target:
            break
        time.sleep(1)
    else:
        raise TimeoutError('no committed boundary in observation window; trainer unchanged')
    stopped, old_queues_terminated = [], False
    receipt = dict(schema='decoupled-physical-batch-migration/1',
        initiated_at=datetime.datetime.now().isoformat(), physical_microbatch=args.physical,
        effective_batch=16, logical_microbatch=1, original_trainer_pid=trainer_pid,
        original_queue_pids=parents, target_exposure=target,
        training_config_and_data_unchanged=True, model_optimizer_rng_restored=True,
        uncommitted_work_may_be_replayed=True)
    try:
        os.kill(trainer_pid, signal.SIGSTOP)
        stopped.append(trainer_pid)
        for pid in reversed(parents):
            os.kill(pid, signal.SIGSTOP)
            stopped.append(pid)
        subprocess.run(['cp', str(training / 'last.pt'), str(work / 'pre_migration_last.pt')], check=True)
        subprocess.run(['cp', str(training / 'protocol.json'), str(work / 'pre_migration_protocol.json')], check=True)
        import torch
        checkpoint = torch.load(work / 'pre_migration_last.pt', map_location='cpu', weights_only=False)
        if checkpoint['global_exposure'] < target or checkpoint['completed_segments'] * 6000 != checkpoint['global_exposure']:
            raise RuntimeError('checkpoint not at announced committed boundary')
        for key in ('model_state_dict', 'optimizer_state_dict', 'rng_state'):
            if key not in checkpoint:
                raise RuntimeError('missing recovery state: ' + key)
        receipt.update(committed_segments=checkpoint['completed_segments'],
                       committed_exposure=checkpoint['global_exposure'], optimizer_updates=checkpoint['optimizer_updates'])
        del checkpoint
        print(json.dumps(dict(event='checkpoint_preserved', **receipt)), flush=True)
        os.kill(trainer_pid, signal.SIGINT)
        os.kill(trainer_pid, signal.SIGCONT)
        wait_for(lambda: not live(trainer_pid, trainer_marker))
        for pid in reversed(parents):
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
        wait_for(lambda: all(not live(pid, queue_marker) for pid in parents))
        old_queues_terminated = True
        replacements = []
        for index, (config_path, config) in enumerate(configs):
            argv = [args.python, '-u', '-m', queue_marker, '--config', str(config_path), '--resume']
            if index:
                argv += ['--dependency-pid', dependency_names[index] + '=' + str(replacements[-1])]
            environment = dict(os.environ)
            environment.pop('RACHEL_SCORE_PHYSICAL_MICROBATCH_N512', None)
            if index == 0:
                environment['RACHEL_SCORE_PHYSICAL_MICROBATCH_N512'] = str(args.physical)
            with (work / ('queue_resume_%d.log' % index)).open('a') as log:
                process = subprocess.Popen(argv, cwd=config['source'], env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            replacements.append(process.pid)
            state_path = Path(config['root']) / 'queue_state.json'
            def registered():
                if process.poll() is not None:
                    raise RuntimeError('replacement queue exited; inspect ' + str(work / ('queue_resume_%d.log' % index)))
                current = read(state_path)
                return current if current.get('pid') == process.pid and current.get('status') in ('running', 'waiting_for_dependency') else None
            current = wait_for(registered)
            if index == 0:
                receipt['replacement_trainer_pid'] = current['active_child']['pid']
            receipt['replacement_queue_pids'] = list(replacements)
            (work / 'migration.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        receipt.update(status='queues_resumed', completed_at=datetime.datetime.now().isoformat())
        (work / 'migration.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
    except BaseException as error:
        receipt.update(status='migration_error', error=repr(error))
        (work / 'migration.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        if not old_queues_terminated:
            for pid in reversed(stopped):
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True)
    parser.add_argument('--training', required=True)
    parser.add_argument('--trainer-pid', type=int, required=True)
    parser.add_argument('--queue-pids', nargs=4, type=int, required=True)
    parser.add_argument('--physical', type=int, choices=(4, 8, 16), required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--max-wait-seconds', type=int, default=900)
    run(parser.parse_args())
