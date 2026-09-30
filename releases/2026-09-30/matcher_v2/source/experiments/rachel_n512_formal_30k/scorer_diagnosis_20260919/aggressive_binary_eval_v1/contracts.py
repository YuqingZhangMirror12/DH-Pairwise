"""Experiment3 terminal checks; shared score/role rules, not the E32 origin."""
from consensus_binary_eval_adapter.contracts import (
    PLAN_SHA, DOMAINS, STOPS, read, sha, inventory, save, validate_threshold,
    metric_identity, validate_budget, validate_real_report, verify_selection_curve,
    partition_real)

SIM_SPLIT = 'sim_test_aggressive'


def validate_terminal(cp, selection, complete, terminal, config, choice):
    if choice not in ('sim', 'real'):
        raise ValueError('explicit SIM or REAL selection required')
    b = cp.get('binding', {})
    if (cp.get('stage') != 'scorer' or b.get('stage') != 'scorer'
            or b.get('arm') != 'scratch_aggressive' or b.get('formal_training') is not True
            or b.get('preflight_steps') != 0 or b.get('config') != config
            or config.get('schema') != 'aggressive-binary-training/1'
            or config.get('scorer_variant') != 'patch'):
        raise ValueError('not an experiment3 formal Patch Scorer checkpoint')
    if (selection.get('status') != 'selected' or complete.get('status') != 'stage_complete'
            or terminal.get('status') != 'training_complete' or terminal.get('arm') != 'scratch_aggressive'
            or terminal.get('stages') != ['matcher', 'scorer']
            or terminal.get('last_stage') != complete
            or terminal.get('selected_matcher') != b.get('fixed_matcher')
            or any(r.get('binding') != b for r in (selection, complete, terminal))
            or {k:v for k,v in selection.items() if k != 'status'} !=
               {k:v for k,v in complete.items() if k != 'status'}):
        raise ValueError('experiment3 terminal identities differ')
    epochs = validate_budget(selection)
    if (selection.get('matcher_unchanged') is not True or selection.get('selection_on_real') is not True
            or selection.get('test_used') is not False or selection.get('migration_origin') is not None):
        raise ValueError('the new Matcher must be frozen for all Scorer training')
    chosen = selection['best' if choice == 'sim' else 'best_real']; e = chosen['epoch']
    if (type(e) is not int or e % 2 or not (0 if choice == 'sim' else 2) <= e <= epochs
            or cp.get('epoch') != e or cp.get('metrics', {}).get('key') != chosen['key']):
        raise ValueError('Scorer checkpoint is not the selected epoch')
    if choice == 'sim':
        validate_threshold(chosen['threshold'])
        if cp.get('threshold') != chosen['threshold'] or cp['metrics'].get('threshold') != chosen['threshold']:
            raise ValueError('SIM CAL threshold differs')
    else:
        validate_real_report(cp['metrics'])
        if cp.get('thresholds') != chosen['thresholds'] or cp['metrics']['thresholds'] != chosen['thresholds']:
            raise ValueError('REAL CAL thresholds differ')
    return b
