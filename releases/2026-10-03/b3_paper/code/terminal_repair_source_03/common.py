"""External terminal identity repair. Never modifies the frozen source/runtime."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading

PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919'
HERE = Path(__file__).resolve().parent
BRIDGE_SHA = '6ff50bcf68462aa659beb33472a2e7a14b698220336abaf6e7758c134c970456'
_alias_lock = threading.RLock()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_binding(value):
    require(sha(value['path']) == value['sha256'], 'bound file changed: '+value['path'])


def api(name):
    return importlib.import_module(PACKAGE+'.'+name)


def source_map():
    return {p.name:sha(p) for p in sorted(HERE.glob('*.py'))}


def check_preparation(path):
    row = read(path)
    require(row.get('schema') == 'matcher-v2-terminal-repair-cpu/1'
            and row.get('status') == 'passed' and row.get('tests', 0) >= 25
            and row.get('failures') == row.get('errors') == row.get('skipped') == 0
            and row.get('cuda_initialized') is False and row.get('source_files') == source_map(),
            'complete unchanged terminal repair CPU preparation required')
    require(sha(HERE/'strict_population_bridge.py') == BRIDGE_SHA, 'audited seed-only bridge changed')
    return row


def same_process(record, identity):
    if not record or record.get('already_exited'):
        return False
    try:
        current = identity(record['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return False
    # PID reuse is not a live handle for this task. Never signal either process.
    return all(current.get(k) == record.get(k) for k in ('pid', 'starttime', 'cmdline'))


def environment(source, gpus=()):
    return dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def ensure_evaluation_binding(source):
    """Read-only idempotent verification across parent/parallel head audits."""
    package = Path(source)/PACKAGE.replace('.','/')
    expected = {'consensus_binary_eval_common':package/'s7_consensus_eval_v14',
                'consensus_binary_eval_adapter':package/'binary_eval_v1'}
    with _alias_lock:
        if all(name not in sys.modules for name in expected):
            api('curriculum_scorer_eval_v1.entry').bind_evaluation(*expected.values())
        require(all(name in sys.modules and list(sys.modules[name].__path__)==[str(path.resolve())]
                    for name,path in expected.items()), 'evaluation alias belongs to a different runtime')
