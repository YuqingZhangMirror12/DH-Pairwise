"""Narrow, fail-closed admission for the diagnosed pre-update naming repair."""
import hashlib
import json
from pathlib import Path

REL = Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/s7_consensus_v1')
PREPARED = Path('/root/autodl-tmp/consensus_threshold_joint_20260927')
FAILED = Path('/root/autodl-tmp/s7_consensus_threshold_joint_e32_20260927')
ROOT = FAILED / 'recovery_20260929_01'
REVISION = 'joint-source03-snapshot-shadow-repair/1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(path):
    path = Path(path)
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*.py'))}


def expected_repair(text):
    replacements = (
        ("for snapshot in plan['reevaluate']:", "for migration_entry in plan['reevaluate']:"),
        ('saved=load_bound(snapshot,origin)', 'saved=load_bound(migration_entry,origin)'),
        ("reevaluating_epoch=snapshot['epoch']", "reevaluating_epoch=migration_entry['epoch']"),
        ("baseline=evaluate(snapshot['epoch'],migration_snapshot=snapshot)",
         "baseline=evaluate(migration_entry['epoch'],migration_snapshot=migration_entry)"),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('unexpected original scope; no broad rewrite')
        text = text.replace(old, new)
    return text


def verify_source_delta(old, new):
    old, new = Path(old), Path(new)
    before, after = inventory(old), inventory(new)
    target = str(REL / 'train.py')
    if set(before) != set(after) or {k for k in before if before[k] != after[k]} != {target}:
        raise ValueError('only train.py local variable naming may change')
    if expected_repair((old / target).read_text()) != (new / target).read_text():
        raise ValueError('repair differs from reviewed four replacements')
    return dict(before_sha256=before[target], after_sha256=after[target], changed=[target],
                model_loss_data_config_unchanged=True, source_inventory_sha256=after)


def validate_previous_failure(root=FAILED):
    root = Path(root)
    failure = read(root / 'failure_scratch_joint_controller.json')
    gate = root / 'preflight/ddp_scratch_joint_01'
    detail = read(gate / 'failure.json')
    exit_record = read(root / 'scratch_joint_gate_exit.json')
    if (failure.get('status') != 'failed' or exit_record.get('returncode') != 1
            or detail.get('type') != 'UnboundLocalError' or 'snapshot' not in detail.get('message', '')
            or failure.get('current_job', {}).get('phase') != 'gate'):
        raise ValueError('not the specifically diagnosed pre-update failure')
    if (root / 'formal_scratch_joint').exists() or (gate / 'scorer/last.pt').exists():
        raise ValueError('existing formal work/checkpoint needs separate recovery review')
    if (gate / 'scorer/status.json').exists() and read(gate / 'scorer/status.json').get('updates', 0) != 0:
        raise ValueError('failed attempt already changed parameters')
    return {str(p): sha(p) for p in (root / 'failure_scratch_joint_controller.json',
                                    gate / 'failure.json', root / 'scratch_joint_gate_exit.json')}


def validate_preparation(prepared, code):
    prepared, code = Path(prepared), Path(code)
    old_receipt = read(prepared / 'training_preparation_v02_remote.json')
    record = read(prepared / 'training_preparation_v03_remote.json')
    if (record.get('status') != 'cpu_preparation_passed' or record.get('revision') != REVISION
            or record.get('errors') or record.get('failures') or record.get('tests', 0) < 6
            or record.get('old_failure_reproduced') is not True
            or record.get('actual_run_stage_cpu_update_verified') is not True
            or record.get('formal_training_started') is not False):
        raise ValueError('missing targeted CPU repair evidence')
    if (sha(prepared / 'training_preparation_v02_remote.json') != record['inherited_preparation_sha256']
            or old_receipt.get('status') != 'cpu_preparation_passed'
            or old_receipt.get('tests') != 116 or old_receipt.get('errors') or old_receipt.get('failures')
            or old_receipt.get('external_control_and_split_tests') != 10):
        raise ValueError('original verified protocol changed')
    old, new = prepared / 'training_source_02', prepared / 'training_source_03'
    if inventory(old) != old_receipt['source_inventory_sha256']:
        raise ValueError('historical source modified')
    delta = verify_source_delta(old, new)
    if delta != record['source_delta'] or delta['source_inventory_sha256'] != record['source_inventory_sha256']:
        raise ValueError('new source differs from tested repair')
    if inventory(code) != record['recovery_python_sha256']:
        raise ValueError('recovery/evaluation controller changed after tests')
    evaluation = prepared / 'evaluation_source_03/preparation.json'
    if sha(evaluation) != record['evaluation_preparation_sha256']:
        raise ValueError('evaluation repair binding changed')
    return new, record
