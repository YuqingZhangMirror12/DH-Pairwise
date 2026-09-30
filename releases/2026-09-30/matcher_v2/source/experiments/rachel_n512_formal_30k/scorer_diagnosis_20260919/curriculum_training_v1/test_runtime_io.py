import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from .checkpoint_io import DistributedCheckpoint, file_sha, tree_sha, write_json
from .runtime_io import (ObservationWriter, check_observations, checkpoint_at,
                         export_completed, selected_curriculum_matcher)
from .runtime_plan import experiment_binding
from .test_validation_adapter import setup_plan, report
from .training_core import Topology, capture_rng, run_updates
from .validation_adapter import ValidationAdapter


class TinyModule(nn.Module):
    def __init__(self, train_matcher):
        super().__init__(); self.model = nn.Module()
        self.model.matcher = nn.Linear(1, 1); self.model.head = nn.Linear(1, 1)
        self.model.matcher.requires_grad_(train_matcher)
        self.model.head.requires_grad_(not train_matcher)
        self.train_matcher = train_matcher

    def forward(self, x):
        layer = self.model.matcher if self.train_matcher else self.model.head
        loss = (layer(x) - 1.).square().mean()
        return loss, {'tiny_fixture': loss.detach()}, {'pairs': len(x)}


def train_fixture(root, module='matcher', order='curriculum', stop=None):
    torch.manual_seed(120)
    ledger, plan = setup_plan(module); raw = TinyModule(module == 'matcher')
    binding = experiment_binding(plan, order, None if module == 'matcher' else 'a' * 64)
    binding['model_spec'] = dict(initial_matcher_state_sha256=tree_sha(raw.model.matcher.state_dict()),
        initial_head_state_sha256=tree_sha(raw.model.head.state_dict()), synthetic_tiny_network=True)
    topology = Topology(0, 1, 4, 1)
    writer = ObservationWriter(root / 'validation', plan, binding)
    current = [0]
    def sim():
        score = {0: .99, 7: .7, 12: .6, 16: .5}[current[0]]
        return report(module, score)[0], {'select': [{'id': 'explicit-metric-fixture', 'score': score}]}
    def real():
        score = {0: .99, 7: .5, 12: .7, 16: .6}[current[0]]
        return report(module, score)[1], {'real_select': [{'id': 'explicit-real-fixture'}]}
    adapter = ValidationAdapter(plan, topology, sim, None if module == 'matcher' else real, writer)
    def evaluate(update):
        current[0] = update
        return adapter(update)
    optimizer = torch.optim.AdamW([p for p in raw.parameters() if p.requires_grad], lr=.0001)
    state = run_updates(raw, optimizer, [torch.tensor([.2])] * len(ledger.catalog), ledger, order,
        topology, 'cpu', plan.learning_rate_knots, plan.validation_updates, evaluate=evaluate,
        binding=binding, checkpoint_every=4, stop_after=stop,
        on_checkpoint=DistributedCheckpoint(root / 'checkpoints', 0, 1))
    return ledger, plan, binding, copy.deepcopy(state)


class RuntimeIOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger, self.plan, self.binding, self.state = train_fixture(self.root)

    def export(self):
        return export_completed(self.root / 'export', self.root / 'checkpoints', self.plan, self.binding)

    def returned(self, complete):
        path = self.root / 'return.json'
        write_json(path, dict(returncode=0,
            training_complete_sha256=file_sha(self.root / 'export' / 'training_complete.json')))
        return path

    def test_only_trained_best_and_equal_budget_endpoint_exported(self):
        complete = self.export()
        self.assertEqual(complete['exports']['sim_best']['update'], 7)
        self.assertEqual(complete['exports']['equal_budget_endpoint']['update'], 16)
        self.assertEqual(complete['completed_exposures'], 64)
        self.assertFalse(complete['claimed_converged']); self.assertTrue(complete['frozen_evaluation_pending'])
        self.assertTrue(complete['process_success_not_yet_certified'])

    def test_export_contains_all_model_tensors_but_no_optimizer_or_rng(self):
        complete = self.export()
        saved = torch.load(complete['exports']['sim_best']['path'], weights_only=False)
        state, _ = checkpoint_at(self.root / 'checkpoints', 7, self.binding)
        expected = {k[len('model.'):]: v for k, v in state['model'].items()}
        self.assertEqual(tree_sha(saved['model']), tree_sha(expected))
        self.assertNotIn('optimizer', saved); self.assertNotIn('rng', saved); self.assertNotIn('epoch', saved)

    def test_real_and_sim_selection_are_separate_from_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, plan, binding, _ = train_fixture(root, 'scorer_stats')
            result = export_completed(root / 'export', root / 'checkpoints', plan, binding)
        self.assertEqual(result['exports']['sim_best']['update'], 7)
        self.assertEqual(result['exports']['real_best']['update'], 12)
        self.assertEqual(result['exports']['equal_budget_endpoint']['update'], 16)

    def test_incomplete_budget_not_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, plan, binding, _ = train_fixture(root, stop=12)
            with self.assertRaisesRegex(ValueError, 'not fully committed'):
                export_completed(root / 'export', root / 'checkpoints', plan, binding)
            self.assertFalse((root / 'export').exists())

    def test_missing_selected_commit_not_inferred_from_rank_shard(self):
        (self.root / 'checkpoints' / 'update_000007' / 'committed.json').unlink()
        with self.assertRaises(FileNotFoundError): self.export()
        self.assertFalse((self.root / 'export').exists())

    def test_changed_observation_file_blocks_export(self):
        receipt = self.state['observations'][1]['report']['artifact']
        Path(receipt['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'observation artifact changed'): self.export()

    def test_changed_committed_rank_blocks_export(self):
        path = self.root / 'checkpoints' / 'update_000007' / 'rank_00.pt'
        path.write_bytes(path.read_bytes() + b'corrupt')
        with self.assertRaisesRegex(ValueError, 'rank file changed'): self.export()

    def test_existing_exports_never_overwritten(self):
        self.export()
        with self.assertRaisesRegex(ValueError, 'fresh export'): self.export()

    def test_selected_matcher_requires_success_return_not_training_receipt_alone(self):
        complete = self.export()
        with self.assertRaises(FileNotFoundError):
            selected_curriculum_matcher(self.root / 'export', self.root / 'return.json', self.plan.sha256)
        self.assertEqual(selected_curriculum_matcher(self.root / 'export', self.returned(complete),
                         self.plan.sha256)['updates'], 7)

    def test_wrong_return_or_other_training_terminal_rejected(self):
        self.export(); path = self.root / 'return.json'
        write_json(path, dict(returncode=1, training_complete_sha256='a' * 64))
        with self.assertRaisesRegex(ValueError, 'process success'):
            selected_curriculum_matcher(self.root / 'export', path, self.plan.sha256)

    def test_mixed_matcher_cannot_silently_initialize_curriculum_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, plan, binding, _ = train_fixture(root, order='mixed')
            export_completed(root / 'export', root / 'checkpoints', plan, binding)
            returned = root / 'return.json'
            write_json(returned, dict(returncode=0,
                training_complete_sha256=file_sha(root / 'export' / 'training_complete.json')))
            with self.assertRaisesRegex(ValueError, 'not this completed curriculum'):
                selected_curriculum_matcher(root / 'export', returned, plan.sha256)

    def test_changed_export_tensor_file_rejected(self):
        complete = self.export(); returned = self.returned(complete)
        Path(complete['exports']['sim_best']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'selected Matcher file changed'):
            selected_curriculum_matcher(self.root / 'export', returned, self.plan.sha256)

    def test_failure_takes_precedence_over_training_complete(self):
        complete = self.export(); returned = self.returned(complete)
        write_json(self.root / 'failure.json', dict(error='synthetic failure'))
        with self.assertRaisesRegex(ValueError, 'failure precedes'):
            selected_curriculum_matcher(self.root / 'export', returned, self.plan.sha256)

    def test_repeated_uncommitted_observation_preserves_both_attempts_and_rng(self):
        writer = ObservationWriter(self.root / 'extra', self.plan, self.binding)
        record = {k: v for k, v in self.state['observations'][0]['report'].items() if k != 'artifact'}
        before = tree_sha(capture_rng('cpu')); rows = dict(simulation=[], real_development=None)
        a = writer(0, record, rows); b = writer(0, record, rows)
        self.assertNotEqual(a['path'], b['path']); self.assertEqual(a['sha256'], b['sha256'])
        self.assertEqual(before, tree_sha(capture_rng('cpu')))

    def test_writer_rejects_wrong_update_or_missing_domain(self):
        writer = ObservationWriter(self.root / 'extra', self.plan, self.binding)
        record = {k: v for k, v in self.state['observations'][0]['report'].items() if k != 'artifact'}
        with self.assertRaisesRegex(ValueError, 'identity'): writer(7, record, dict(simulation=[], real_development=None))
        with self.assertRaisesRegex(ValueError, 'domain'): writer(0, record, dict(simulation=[]))

    def test_observation_binding_not_just_file_hash_checked(self):
        rows = copy.deepcopy(self.state['observations']); report_row = rows[0]['report']
        path = Path(report_row['artifact']['path']); saved = json.loads(path.read_text())
        saved['binding']['order'] = 'mixed'; path.write_text(json.dumps(saved))
        report_row['artifact']['sha256'] = file_sha(path)
        with self.assertRaisesRegex(ValueError, 'different observation'): check_observations(rows, self.binding)


if __name__ == '__main__':
    unittest.main()
