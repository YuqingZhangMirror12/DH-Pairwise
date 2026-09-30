"""Immutable observation artifacts and exports from committed update states.

Exports contain model tensors only, never optimizer/RNG state. They do not
certify a process exit or frozen TEST/real evaluation. No training is launched.
"""
import json
import os
from pathlib import Path
import uuid

import torch

from .checkpoint_io import cpu_copy, file_sha, tree_sha, verified_state, write_json
from .exposure import digest, integer
from .validation_adapter import canonical, select_history


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


class ObservationWriter:
    """Rank0 callback; interrupted uncommitted attempts remain separate files."""
    def __init__(self, root, plan, binding):
        self.root = Path(root).resolve(); self.plan = plan; self.binding = canonical(binding)
        require(binding['common_plan_sha256'] == plan.sha256, 'writer plan differs')

    def __call__(self, completed, record, rows):
        integer(completed, 'completed update')
        require(completed in self.plan.validation_updates and record['update'] == completed
                and record['common_plan_sha256'] == self.plan.sha256
                and record['module'] == self.plan.record['module'], 'wrong observation identity')
        require(set(rows) == {'simulation', 'real_development'}, 'observation rows missing a domain')
        directory = self.root / ('update_%06d' % completed)
        directory.mkdir(parents=True, exist_ok=True)
        # uuid uses OS entropy, not any of the training RNG streams. A resumed
        # uncommitted evaluation cannot overwrite a prior attempt or its timing.
        path = directory / (uuid.uuid4().hex + '.json')
        payload = canonical(dict(schema='curriculum-validation-artifact/1',
            binding=self.binding, observation=record, rows=rows))
        write_json(path, payload)
        return dict(path=str(path), sha256=file_sha(path))


def check_observations(observations, binding):
    for row in observations:
        report = row['report']; receipt = report['artifact']; path = Path(receipt['path'])
        require(path.is_absolute() and file_sha(path) == receipt['sha256'], 'observation artifact changed')
        saved = read(path)
        require(saved.get('schema') == 'curriculum-validation-artifact/1'
                and saved['binding'] == binding
                and saved['observation'] == {k: v for k, v in report.items() if k != 'artifact'},
                'saved validation rows belong to a different observation or experiment')


def checkpoint_at(root, update, binding):
    """An uncommitted rank shard is never a selectable model."""
    root = Path(root); integer(update, 'checkpoint update')
    manifest = read(root / ('update_%06d' % update) / 'committed.json')
    require(manifest.get('schema') == 'curriculum-checkpoint-commit/1'
            and manifest['completed_updates'] == update
            and manifest['binding_sha256'] == digest(binding), 'committed update binding differs')
    receipts = manifest['ranks']; world = manifest['world_size']
    require([r['rank'] for r in receipts] == list(range(world)) and world > 0,
            'missing committed rank')
    for item in receipts:
        require(item['world_size'] == world and item['completed_updates'] == update
                and item['binding_sha256'] == manifest['binding_sha256']
                and item['sampling'] == manifest['sampling']
                and item['shared_state_sha256'] == receipts[0]['shared_state_sha256'],
                'ranks do not share the same complete update')
        expected = 'update_%06d/rank_%02d.pt' % (update, item['rank'])
        require(item['relative_path'] == expected and file_sha(root / expected) == item['file_sha256'],
                'committed rank file changed or missing')
    state = verified_state(root, receipts[0])
    require(state['binding'] == binding and state['sampling']['order'] == binding['order'],
            'checkpoint belongs to another presentation order')
    require(state['sampling']['ledger_sha256'] == binding['common_plan']['ledger_sha256'],
            'checkpoint exposure ledger differs')
    return state, dict(path=str((root / ('update_%06d' % update) / 'committed.json').resolve()),
                       sha256=file_sha(root / ('update_%06d' % update) / 'committed.json'))


def component_state(model, prefix):
    state = {k[len(prefix):]: value for k, value in model.items() if k.startswith(prefix)}
    require(bool(state), 'missing model component: ' + prefix)
    return state


def model_only(state, binding):
    require(state['model'] and all(k.startswith('model.') for k in state['model']),
            'expected the bound TrainModule model state, not an arbitrary network')
    result = {k[len('model.'):]: value for k, value in state['model'].items()}
    matcher = component_state(result, 'matcher.'); head = component_state(result, 'head.')
    spec = binding['model_spec']
    if binding['module'] == 'matcher':
        require(tree_sha(head) == spec['initial_head_state_sha256'], 'inactive Scorer changed during Matcher training')
    else:
        require(tree_sha(matcher) == spec['initial_matcher_state_sha256'], 'frozen Matcher changed during Scorer training')
    return cpu_copy(result)


def save_model(path, record):
    path = Path(path)
    require(not path.exists(), 'preserve earlier exported model')
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('xb') as stream:
        torch.save(record, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    require(tree_sha(loaded) == tree_sha(record), 'exported state did not round-trip exactly')
    return dict(path=str(path.resolve()), sha256=file_sha(path),
                model_state_sha256=tree_sha(record['model']), update=record['updates'])


def export_completed(root, checkpoints, plan, binding, *, selection_replay=select_history):
    """Export SIM-best, optional REAL-best, and the equal-budget endpoint."""
    root = Path(root); checkpoints = Path(checkpoints)
    require(not root.exists(), 'fresh export directory required')
    require(binding['common_plan_sha256'] == plan.sha256 and binding['common_plan'] == plan.record,
            'export plan binding differs')
    pointer = read(checkpoints / 'last_committed.json')
    end = plan.record['total_updates']; expected = 'update_%06d/committed.json' % end
    require(pointer['schema'] == 'curriculum-checkpoint-pointer/1'
            and pointer['completed_updates'] == end and pointer['relative_path'] == expected
            and file_sha(checkpoints / expected) == pointer['file_sha256'],
            'registered training budget is not fully committed')
    final, final_receipt = checkpoint_at(checkpoints, end, binding)
    # New isolated experiments can provide their explicitly bound selection
    # policy. Historical callers retain the exact original two-domain default.
    history = final['observations']; selection = selection_replay(plan, history, end)
    check_observations(history, binding)
    require(selection['fixed_budget_reached'] and selection['best_sim'] is not None,
            'complete native trained selection required')
    choices = dict(sim_best=selection['best_sim']['update'], equal_budget_endpoint=end)
    if binding['module'] != 'matcher':
        require(selection['best_real'] is not None, 'trained real development selection required')
        choices['real_best'] = selection['best_real']['update']
    states = {}; receipts = {}
    for update in sorted(set(choices.values())):
        state, committed = checkpoint_at(checkpoints, update, binding)
        require(state['observations'] == [r for r in history if r['update'] <= update],
                'selected checkpoint validation history differs from the final history')
        states[update] = model_only(state, binding); receipts[update] = committed
    root.mkdir(parents=True)
    exports = {}
    for name, update in choices.items():
        observation = next(r['report'] for r in history if r['update'] == update)
        record = dict(schema='curriculum-model-export/1', module=binding['module'],
            stage='matcher' if binding['module'] == 'matcher' else 'scorer', order=binding['order'],
            selection_kind=name, updates=update, exposures=update * plan.record['effective_batch'],
            total_completed_updates=end, binding=binding, model=states[update],
            observation=observation, committed_checkpoint=receipts[update],
            optimizer_imported=False, training_rng_included=False)
        exports[name] = save_model(root / (name + '.pt'), record)
    selection.update(status='selected', binding=binding, exports=exports,
        final_committed_checkpoint=final_receipt, actual_epochs=None,
        epoch_note='update-budget experiment; no invented 24K epoch count',
        final_model_evaluation_complete=False, claimed_converged=False)
    write_json(root / 'selection.json', selection)
    complete = dict(schema='curriculum-training-complete/1', status='training_complete',
        binding=binding, completed_updates=end, completed_exposures=final['sampling']['completed_exposures'],
        selection_sha256=file_sha(root / 'selection.json'), exports=exports,
        stop_reason='fixed_shared_update_budget', claimed_converged=False,
        process_success_not_yet_certified=True, frozen_evaluation_pending=True)
    write_json(root / 'training_complete.json', complete)
    return complete


def selected_curriculum_matcher(root, process_return, expected_common_plan_sha256):
    """Only a completed, successful curriculum Matcher can initialize new heads."""
    root = Path(root)
    for path in (root / 'failure.json', root.parent / 'failure.json'):
        require(not path.exists(), 'Matcher failure precedes stale completion')
    returned = read(process_return)
    complete = read(root / 'training_complete.json')
    require(returned['returncode'] == 0
            and returned['training_complete_sha256'] == file_sha(root / 'training_complete.json'),
            'Matcher process success and terminal binding required')
    require(complete['status'] == 'training_complete' and complete['stop_reason'] == 'fixed_shared_update_budget',
            'Matcher did not finish its fixed budget')
    binding = complete['binding']
    require(binding['module'] == 'matcher' and binding['order'] == 'curriculum'
            and binding['common_plan_sha256'] == expected_common_plan_sha256
            and digest(binding['common_plan']) == expected_common_plan_sha256,
            'not this completed curriculum Matcher')
    require(file_sha(root / 'selection.json') == complete['selection_sha256'], 'Matcher selection changed')
    selection = read(root / 'selection.json'); selected = selection['exports']['sim_best']
    require(selection['fixed_budget_reached'] is True and selection['best_real'] is None
            and selection['binding'] == binding and selection['exports'] == complete['exports'],
            'wrong Matcher selection policy or export set')
    path = Path(selected['path'])
    require(path.resolve() == (root / 'sim_best.pt').resolve() and file_sha(path) == selected['sha256'],
            'selected Matcher file changed or belongs to another export')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(saved['schema'] == 'curriculum-model-export/1' and saved['binding'] == binding
            and saved['selection_kind'] == 'sim_best' and saved['updates'] == selection['best_sim']['update']
            and saved['updates'] > 0 and tree_sha(saved['model']) == selected['model_state_sha256'],
            'selected Matcher tensor/observation identity differs')
    return dict(path=str(path.resolve()), sha256=selected['sha256'],
        selection_sha256=file_sha(root / 'selection.json'), terminal_sha256=file_sha(root / 'training_complete.json'),
        successful_return_sha256=file_sha(process_return), updates=saved['updates'],
        source_common_plan_sha256=expected_common_plan_sha256,
        source_binding=binding, old_head_imported=False, optimizer_imported=False)
