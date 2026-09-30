"""Filesystem resume and complete-rank semantics; no remote job mutation."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from . import test_training_core as fixture
from .checkpoint_io import DistributedCheckpoint, commit, load_committed, save_rank, tree_sha
from .exposure import RankMicrobatches
from .training_core import Topology, run_updates


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def rank_states(self, completed=3):
        state = fixture.experiment(stop=completed)
        ledger = fixture.prepare()[3]
        result = []
        for rank in range(2):
            row = copy.deepcopy(state); row['rank'] = rank
            row['sampling'] = RankMicrobatches(ledger, 'curriculum', 0, rank, 2, 1, 2).cursor(completed)
            row['rng']['torch'] = torch.Generator().manual_seed(800 + rank).get_state()
            result.append(row)
        return result, ledger

    def test_all_ranks_commit_despite_distinct_rank_rng(self):
        rows, ledger = self.rank_states()
        receipts = [save_rank(self.root, x) for x in rows]
        self.assertNotEqual(receipts[0]['full_state_sha256'], receipts[1]['full_state_sha256'])
        self.assertEqual(receipts[0]['shared_state_sha256'], receipts[1]['shared_state_sha256'])
        commit(self.root, receipts)
        for rank in range(2):
            loaded = load_committed(self.root, rows[rank]['binding'], ledger, 'curriculum', Topology(rank, 2, 1, 2))
            self.assertEqual(tree_sha(loaded), tree_sha(rows[rank]))

    def test_partial_later_save_does_not_advance_committed_resume(self):
        rows, ledger = self.rank_states(); commit(self.root, [save_rank(self.root, x) for x in rows])
        later, _ = self.rank_states(4); save_rank(self.root, later[0])
        loaded = load_committed(self.root, rows[0]['binding'], ledger, 'curriculum', Topology(0, 2, 1, 2))
        self.assertEqual(loaded['sampling']['completed_updates'], 3)
        self.assertTrue((self.root / 'update_000004/rank_00.pt').is_file())

    def test_missing_rank_does_not_create_pointer(self):
        rows, _ = self.rank_states()
        with self.assertRaisesRegex(ValueError, 'missing'):
            commit(self.root, [save_rank(self.root, rows[0])])
        self.assertFalse((self.root / 'last_committed.json').exists())

    def test_rank_parameter_divergence_does_not_commit(self):
        rows, _ = self.rank_states(); rows[1]['model']['frozen'] += 1
        with self.assertRaisesRegex(ValueError, 'ranks disagree'):
            commit(self.root, [save_rank(self.root, x) for x in rows])

    def test_changed_shard_not_accepted(self):
        rows, _ = self.rank_states(); receipts = [save_rank(self.root, x) for x in rows]
        path = self.root / receipts[1]['relative_path']; path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            commit(self.root, receipts)

    def test_manifest_corruption_does_not_resume(self):
        rows, ledger = self.rank_states(); pointer = commit(self.root, [save_rank(self.root, x) for x in rows])
        path = self.root / pointer['relative_path']; path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'manifest hash'):
            load_committed(self.root, rows[0]['binding'], ledger, 'curriculum', Topology(0, 2, 1, 2))

    def test_other_rank_missing_after_commit_does_not_resume(self):
        rows, ledger = self.rank_states(); receipts = [save_rank(self.root, x) for x in rows]; commit(self.root, receipts)
        (self.root / receipts[1]['relative_path']).unlink()
        with self.assertRaises(FileNotFoundError):
            load_committed(self.root, rows[0]['binding'], ledger, 'curriculum', Topology(0, 2, 1, 2))

    def test_changed_binding_or_order_or_topology_rejected(self):
        rows, ledger = self.rank_states(); commit(self.root, [save_rank(self.root, x) for x in rows])
        for binding, order, topology in [({'other':True}, 'curriculum', Topology(0, 2, 1, 2)),
                                         (rows[0]['binding'], 'mixed', Topology(0, 2, 1, 2)),
                                         (rows[0]['binding'], 'curriculum', Topology(0, 1, 2, 2))]:
            with self.subTest(order=order, topology=topology), self.assertRaises(ValueError):
                load_committed(self.root, binding, ledger, order, topology)

    def test_no_duplicate_overwrite_or_backward_commit(self):
        rows, _ = self.rank_states(); receipts = [save_rank(self.root, x) for x in rows]; commit(self.root, receipts)
        with self.assertRaises(FileExistsError):
            save_rank(self.root, rows[0])
        with self.assertRaisesRegex(ValueError, 'backward'):
            commit(self.root, receipts)

    def test_forged_rank_path_rejected(self):
        rows, _ = self.rank_states(); receipts = [save_rank(self.root, x) for x in rows]
        receipts[0]['relative_path'] = '../unrelated.pt'
        with self.assertRaisesRegex(ValueError, 'escaped'):
            commit(self.root, receipts)

    def test_tree_hash_handles_scalar_tensors_and_type_or_order(self):
        value = {'b':torch.tensor(1.), 'a':torch.tensor([2, 3])}
        self.assertEqual(tree_sha(value), tree_sha(dict(reversed(list(value.items())))))
        self.assertNotEqual(tree_sha(value), tree_sha({'b':torch.tensor(1), 'a':torch.tensor([2, 3])}))

    def test_cannot_advance_a_corrupted_previous_commit(self):
        rows, _ = self.rank_states(); value = commit(self.root, [save_rank(self.root, x) for x in rows])
        (self.root / value['relative_path']).write_text('{}')
        later, _ = self.rank_states(4)
        with self.assertRaisesRegex(ValueError, 'previous committed pointer'):
            commit(self.root, [save_rank(self.root, x) for x in later])

    def test_cannot_change_presentation_order_within_an_output_root(self):
        rows, _ = self.rank_states(); commit(self.root, [save_rank(self.root, x) for x in rows])
        later, _ = self.rank_states(4)
        for row in later:
            row['sampling']['order'] = 'mixed'
        with self.assertRaisesRegex(ValueError, 'different experiment'):
            commit(self.root, [save_rank(self.root, x) for x in later])

    def test_actual_core_disk_resume_matches_uninterrupted(self):
        full = fixture.experiment()
        partial = fixture.experiment(stop=7)
        DistributedCheckpoint(self.root, 0, 1)(partial)
        ledger = fixture.prepare()[3]
        restored = load_committed(self.root, partial['binding'], ledger, 'curriculum', Topology(0, 1, 2, 2))
        self.assertEqual(tree_sha(full), tree_sha(fixture.experiment(resume=restored)))

    def test_initial_and_last_update_callback_are_separate_commits(self):
        model, optimizer, data, ledger = fixture.prepare()
        result = run_updates(model, optimizer, data, ledger, 'curriculum', Topology(0, 1, 2, 2),
            'cpu', [(0, .001)], (), binding={'fixture':True}, stop_after=1,
            on_checkpoint=DistributedCheckpoint(self.root, 0, 1))
        loaded = load_committed(self.root, result['binding'], ledger, 'curriculum', Topology(0, 1, 2, 2))
        self.assertEqual(tree_sha(result), tree_sha(loaded))
        self.assertTrue((self.root / 'update_000000/committed.json').is_file())


if __name__ == '__main__':
    unittest.main()
