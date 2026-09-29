"""Small fail-closed contracts; no model inference or optimizer in this module."""
import hashlib
import json
import math
import os
from pathlib import Path

import torch

E32_SHA = '80cac47d5bc5340df35a7a7c36ab4a3580a9eea8c99adf797ff5744cd2068b17'
PLAN_SHA = '0f8c13bbc4277052ecd6d21e335e97bf58dc8522174e2a11341c3221d50450f6'
UNUSED = ('matcher.base.coarse.', 'matcher.base.local_head.', 'matcher.base.fusion.')
STOPS = ('simulation_plateau_after_lr_reductions', 'budget_limit_not_claimed_converged')
DOMAINS = ('dunhuang_cv', 'turufan')
ROLES = ('real_cal', 'real_select', 'real_test')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def inventory(directory):
    return {p.name: sha(p) for p in sorted(Path(directory).glob('*.py'))}


def validate_budget(selection):
    epochs = selection.get('actual_epochs')
    if (type(epochs) is not int or not 16 <= epochs <= 48 or epochs % 2
            or selection.get('updates') != epochs * 750
            or selection.get('exposures') != epochs * 24000
            or selection.get('stop_reason') not in STOPS):
        raise ValueError('invalid terminal training budget')
    return epochs


def validate_threshold(value):
    if (type(value) not in (float, int) or not math.isfinite(value)
            or not .2 <= value <= .8 or abs(value * 100 - round(value * 100)) > 1e-8):
        raise ValueError('threshold outside registered .20-.80 grid')


def validate_real_report(report):
    if (report.get('status') != 'development_selection' or report.get('test_used') is not False
            or report.get('real_used') is not True or report.get('gradients_used') is not False
            or report.get('development_evaluation') is not True
            or set(report.get('thresholds', {})) != set(DOMAINS)
            or len(report.get('key', [])) != 3
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in report['key'])):
        raise ValueError('invalid development-only selection report')
    for value in report['thresholds'].values():
        validate_threshold(value)


def choose_real_best(reports):
    """Epoch0 is diagnostic only; an exact tie keeps the earlier trained epoch."""
    if not reports or len({r['epoch'] for r in reports}) != len(reports):
        raise ValueError('empty or duplicate epoch curve')
    selected = None
    for row in sorted(reports, key=lambda r: r['epoch']):
        epoch = row['epoch']
        if type(epoch) is not int or epoch < 0 or epoch % 2:
            raise ValueError('invalid validation epoch')
        validate_real_report(row['real_report'])
        if epoch > 0 and (selected is None or tuple(row['real_report']['key']) > tuple(selected['real_report']['key'])):
            selected = row
    if selected is None:
        raise ValueError('no trained REAL-SELECT checkpoint')
    return selected


def metric_identity(metrics):
    result = {k: v for k, v in metrics.items() if k != 'elapsed_seconds'}
    if isinstance(result.get('real_development'), dict):
        result['real_development'] = {k: v for k, v in result['real_development'].items()
                                      if k != 'elapsed_seconds'}
    return result


def validate_archive(cp, validation, binding, epoch):
    # An identical crash replay may rewrite timing, never model/metric identity.
    metrics = metric_identity(cp.get('metrics', {}))
    expected_metrics = metric_identity({k: v for k, v in validation.items()
                                       if k not in ('epoch', 'updates', 'exposures')})
    if (cp.get('binding') != binding or cp.get('stage') != 'scorer'
            or cp.get('epoch') != epoch or validation.get('epoch') != epoch
            or cp.get('updates') != epoch * 750 or cp.get('exposures') != epoch * 24000
            or validation.get('updates') != cp['updates'] or validation.get('exposures') != cp['exposures']
            or cp.get('threshold') != validation.get('threshold')
            or metrics != expected_metrics):
        raise ValueError('archived epoch identity/validation mismatch')
    validate_threshold(cp['threshold'])


def matcher_change(origin, selected, *, expect_updated):
    a = {k: v for k, v in origin.items() if k.startswith('matcher.')}
    b = {k: v for k, v in selected.items() if k.startswith('matcher.')}
    if not a or a.keys() != b.keys():
        raise ValueError('Matcher tensor membership differs')
    changed = []
    sq = norm = maximum = 0.
    for name, x in a.items():
        y = b[name]
        if x.shape != y.shape or x.dtype != y.dtype or not torch.isfinite(y).all():
            raise ValueError('Matcher shape/dtype/finite contract differs')
        norm += float(x.double().square().sum())
        if torch.equal(x, y):
            continue
        if name.startswith(UNUSED):
            raise ValueError('unused legacy head changed: ' + name)
        changed.append(name)
        d = y.double() - x.double()
        sq += float(d.square().sum())
        maximum = max(maximum, float(d.abs().max()))
    if expect_updated is True and not changed:
        raise ValueError('joint Matcher did not update')
    if expect_updated is False and changed:
        raise ValueError('frozen control Matcher changed')
    return dict(changed_tensors=len(changed), changed_names=changed,
                relative_l2=(sq / max(norm, 1e-300)) ** .5, max_absolute=maximum,
                unused_heads_unchanged=True)


def validate_joint_terminal(cp, selection, complete, terminal, expected_config, choice):
    binding = cp.get('binding', {})
    if (binding.get('arm') != 'scratch_joint' or binding.get('formal_training') is not True
            or binding.get('preflight_steps') != 0 or binding.get('config') != expected_config
            or cp.get('stage') != 'scorer'):
        raise ValueError('not the registered formal joint checkpoint')
    if (selection.get('status') != 'selected' or complete.get('status') != 'stage_complete'
            or terminal.get('status') != 'training_complete' or terminal.get('arm') != 'scratch_joint'
            or terminal.get('stages') != ['scorer'] or terminal.get('last_stage') != complete
            or any(record.get('binding') != binding for record in (selection, complete, terminal))
            or {k: v for k, v in selection.items() if k != 'status'} !=
               {k: v for k, v in complete.items() if k != 'status'}):
        raise ValueError('joint terminal identities differ')
    epochs = validate_budget(selection)
    if (selection.get('selection_on_real') is not True or selection.get('test_used') is not False
            or selection.get('matcher_unchanged') is not False
            or selection.get('development_evaluation') is not True
            or selection.get('matcher_change', {}).get('changed_tensors', 0) <= 0
            or selection.get('matcher_change', {}).get('unused_changed') != []):
        raise ValueError('missing joint/real selection audit')
    if choice not in ('sim', 'real'):
        raise ValueError('unregistered selection choice')
    chosen = selection['best' if choice == 'sim' else 'best_real']
    epoch = chosen['epoch']
    if (type(epoch) is not int or not (0 if choice == 'sim' else 2) <= epoch <= epochs
            or epoch % 2 or cp.get('epoch') != epoch or cp.get('metrics', {}).get('key') != chosen['key']):
        raise ValueError('not the registered selected epoch')
    if choice == 'sim':
        validate_threshold(chosen['threshold'])
        if cp.get('threshold') != chosen['threshold'] or cp['metrics'].get('threshold') != chosen['threshold']:
            raise ValueError('selected synthetic CAL threshold differs')
    else:
        validate_real_report(cp['metrics'])
        if cp.get('thresholds') != chosen['thresholds'] or cp['metrics']['thresholds'] != chosen['thresholds']:
            raise ValueError('selected real CAL thresholds differ')
    return binding


def partition_real(rows, plan, domain):
    """Full-domain context remains developmental; fold0 is the new protocol holdout."""
    spec = plan['datasets'][domain]
    by_id = {r['pair_id']: r for r in rows}
    expected = set(spec['excluded_gt_pair_ids'])
    for role in ROLES:
        expected.update(spec['roles'][role]['pair_ids'])
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise ValueError('real final population differs from registered roles')
    groups = {'all_development_context': rows}
    for role in ROLES:
        ids = spec['roles'][role]['pair_ids']
        groups[role] = [by_id[i] for i in ids]
    if domain == 'dunhuang_cv':
        excluded = set(spec['excluded_gt_pair_ids'])
        groups['gt_corrected_800_development_context'] = [r for r in rows if r['pair_id'] not in excluded]
    return groups
