"""CPU protocol fixtures only. Synthetic CUDA metadata is NOT a GPU gate."""
import copy
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from . import execution as api
from .checkpoint_io import commit, file_sha, save_rank, tree_sha, write_json
from .exposure import STAGES, build_ledger, digest
from .runtime_io import export_completed
from .test_exposure import samples
from .test_runtime_io import train_fixture
from .test_runtime_plan import fixture
from .validation_adapter import RULES


def bound(path):
    return dict(path=str(path.resolve()), sha256=file_sha(path))


def manifest_fixture(root):
    baseline = root / 'baseline'; baseline.mkdir(); (baseline / 'fixture.py').write_text('# fixture\n')
    baseline_map = {'fixture.py': file_sha(baseline / 'fixture.py')}
    rows = samples((16, 16, 16)); ledger = build_ledger(rows, dict(zip(STAGES, (7, 5, 4))), 260928, 32)
    _, record = fixture(); record.update(ledger_sha256=ledger.sha256, seed=ledger.seed, effective_batch=32,
        baseline_sources_sha256=digest(baseline_map)); record['selection_rule']['id'] = RULES['matcher']
    catalog = [asdict(row) for row in ledger.catalog]
    admission = root / 'admission.json'; write_json(admission, dict(status='passed',
        schema='curriculum-data-admission/1', catalog=catalog, catalog_sha256=digest(catalog)))
    geometry = root / 'geometry.json'; write_json(geometry, dict(status='complete',
        schema='s7-consensus-train-geometry/2', curriculum_training_catalog_sha256=digest(catalog),
        curriculum_data_admission_sha256=file_sha(admission), real_used=False, test_used=False))
    validation = {}
    for name, split in [('cal_mixed', 'cal'), ('select_mixed', 'select')]:
        path = root / (name + '.json'); write_json(path, dict(split=split, entries=[dict(id='synthetic')]))
        validation[name] = dict(bound(path), pair_count=1)
    contract = root / 'contract.json'; write_json(contract, dict(status='passed', source_disjoint=True, validation=validation))
    record['data_admission_sha256'] = file_sha(admission); record['geometry_sha256'] = file_sha(geometry)
    plan = root / 'plan.json'; write_json(plan, record)
    real = root / 'real.json'; write_json(real, dict(synthetic_fixture=True))
    reference = root / 'reference.pt'; reference.write_bytes(b'architecture-only manifest fixture')
    own = Path(api.__file__).resolve().parent
    return dict(schema='curriculum-execution/1', locked=True, runtime_plan=bound(plan), admission=bound(admission),
        geometry=bound(geometry), simulation_contract=bound(contract), real_split=bound(real),
        reference_checkpoint=bound(reference), implementation=dict(root=str(own),
        python_sha256={str(p.relative_to(own)): file_sha(p) for p in own.rglob('*.py')}),
        baseline=dict(root=str(baseline.resolve()), python_sha256=baseline_map),
        topology=dict(world_size=2, microbatch=8, accumulate=2, workers=4), selected_matcher=None)


class ExecutionInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.spec = manifest_fixture(self.root)

    def test_manifest_protocol_only_no_cuda_or_dataset_loading(self):
        with patch.object(torch.cuda, 'set_device', side_effect=AssertionError('GPU must not be accessed')):
            inputs = api.load_inputs(self.spec)
        self.assertEqual(inputs['ledger'].total_updates, 16)
        self.assertEqual(inputs['ledger'].effective_batch, 32)

    def test_unlocked_or_extra_fields_rejected(self):
        for name, value in [('locked', False), ('automatic_retry', True)]:
            changed = copy.deepcopy(self.spec); changed[name] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'locked execution'):
                api.load_inputs(changed)

    def test_changed_source_does_not_reuse_binding(self):
        (self.root / 'baseline' / 'fixture.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'source changed'): api.load_inputs(self.spec)

    def test_other_execution_source_rejected(self):
        self.spec['implementation']['root'] = str(self.root / 'baseline')
        with self.assertRaisesRegex(ValueError, 'different curriculum source'): api.load_inputs(self.spec)

    def test_modified_file_even_same_declared_status_rejected(self):
        Path(self.spec['admission']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'file binding'): api.load_inputs(self.spec)

    def test_legacy_geometry_cannot_be_claimed_curriculum_calibration(self):
        path = Path(self.spec['geometry']['path']); value = json.loads(path.read_text()); value['real_used'] = True
        write_json(path, value, replace=True); self.spec['geometry'] = bound(path)
        p = Path(self.spec['runtime_plan']['path']); record = json.loads(p.read_text())
        record['geometry_sha256'] = file_sha(path); write_json(p, record, replace=True); self.spec['runtime_plan'] = bound(p)
        with self.assertRaisesRegex(ValueError, 'all-TRAIN'): api.load_inputs(self.spec)

    def test_cal_select_population_hash_checked(self):
        (self.root / 'cal_mixed.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'file binding'): api.load_inputs(self.spec)

    def test_unregistered_topology_rejected(self):
        self.spec['topology']['microbatch'] = 16
        with self.assertRaisesRegex(ValueError, 'topology'): api.load_inputs(self.spec)

    def test_random_matcher_cannot_import_selected_weights(self):
        self.spec['selected_matcher'] = dict(path='old-E32')
        with self.assertRaisesRegex(ValueError, 'cannot import weights'): api.load_inputs(self.spec)

    def test_updating_matcher_has_no_candidate_cache(self):
        self.assertEqual(api.candidate_caches(self.root, dict(module='matcher'), None, 0, 2, None), (None, None))

    def test_rank_zero_failure_not_silenced_in_single_process(self):
        def fail():
            raise OSError('synthetic write error')
        with self.assertRaisesRegex(RuntimeError, 'synthetic write error'): api.rank0_call(fail, 0, 1, 'init')


class ExecutionRecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.ledger, self.plan, self.binding, self.state = train_fixture(self.root, stop=7)

    def later(self):
        with tempfile.TemporaryDirectory() as tmp:
            return train_fixture(Path(tmp), stop=12)[3]

    def test_no_leftovers_does_not_create_recovery_folder(self):
        self.assertIsNone(api.archive_resume_leftovers(self.root, self.state))
        self.assertEqual(list(self.root.glob('resume_preserved_*')), [])

    def test_partial_later_shard_preserved_without_advancing_resume(self):
        later = self.later(); save_rank(self.root / 'checkpoints', later)
        before = file_sha(self.root / 'checkpoints' / 'last_committed.json')
        proof = api.archive_resume_leftovers(self.root, self.state)
        preserved = Path(proof['path']) / 'checkpoints' / 'update_000012' / 'rank_00.pt'
        self.assertEqual(tree_sha(torch.load(preserved, weights_only=False)), tree_sha(later))
        self.assertEqual(before, file_sha(self.root / 'checkpoints' / 'last_committed.json'))
        # It is now possible to replay and save the unacknowledged update again.
        commit(self.root / 'checkpoints', [save_rank(self.root / 'checkpoints', later)])

    def test_commit_before_pointer_crash_is_preserved_not_adopted(self):
        pointer = json.loads((self.root / 'checkpoints' / 'last_committed.json').read_text())
        commit(self.root / 'checkpoints', [save_rank(self.root / 'checkpoints', self.later())])
        write_json(self.root / 'checkpoints' / 'last_committed.json', pointer, replace=True)
        proof = api.archive_resume_leftovers(self.root, self.state)
        self.assertTrue((Path(proof['path']) / 'checkpoints/update_000012/committed.json').is_file())
        self.assertEqual(json.loads((self.root / 'checkpoints/last_committed.json').read_text()), pointer)

    def test_foreign_uncommitted_state_not_moved(self):
        later = self.later(); later['binding']['order'] = 'mixed'; save_rank(self.root / 'checkpoints', later)
        with self.assertRaisesRegex(ValueError, 'another experiment'): api.archive_resume_leftovers(self.root, self.state)
        self.assertTrue((self.root / 'checkpoints/update_000012/rank_00.pt').exists())

    def test_explicit_resume_archives_bound_pause_request_and_receipt(self):
        pointer = json.loads((self.root / 'checkpoints/last_committed.json').read_text())
        write_json(self.root / 'pause_request.json', dict(action='pause_after_committed_update', binding_sha256=digest(self.binding)))
        write_json(self.root / 'pause_complete.json', dict(binding_sha256=digest(self.binding), last_committed=pointer))
        before = tree_sha(self.state['rng']); proof = api.archive_resume_leftovers(self.root, self.state)
        self.assertFalse((self.root / 'pause_request.json').exists())
        self.assertTrue((Path(proof['path']) / 'pause_request.json').is_file())
        self.assertEqual(tree_sha(self.state['rng']), before)

    def test_unbound_pause_not_consumed(self):
        write_json(self.root / 'pause_request.json', dict(action='pause_after_committed_update', binding_sha256='wrong'))
        with self.assertRaisesRegex(ValueError, 'unbound pause'): api.archive_resume_leftovers(self.root, self.state)

    def test_pause_callback_commits_before_raising(self):
        request = self.root / 'pause_request.json'; write_json(request, dict(action='pause_after_committed_update', binding_sha256=digest(self.binding)))
        callback = api.CompleteCheckpoint(self.root / 'checkpoints', 0, 1, request, digest(self.binding))
        with self.assertRaises(api.CoordinatedPause): callback(self.later())
        self.assertEqual(json.loads((self.root / 'checkpoints/last_committed.json').read_text())['completed_updates'], 12)

    def test_new_export_attempt_preserves_partial_and_completed_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, plan, binding, _ = train_fixture(root)
            complete = export_completed(root / 'exports', root / 'checkpoints', plan, binding)
            old_sha = file_sha(root / 'exports/training_complete.json')
            second = export_completed(root / 'exports_attempt_fixture', root / 'checkpoints', plan, binding)
            self.assertEqual(complete['exports']['sim_best']['model_state_sha256'], second['exports']['sim_best']['model_state_sha256'])
            self.assertEqual(old_sha, file_sha(root / 'exports/training_complete.json'))


class ExecutionGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        _, _, self.formal, state = train_fixture(self.root / 'tiny_cpu', stop=12)
        self.cpu = copy.deepcopy(state); state['binding'] = dict(self.formal, run_mode='gate'); state['observations'] = []
        # Explicit synthetic CUDA metadata for parser tests ONLY, no GPU use.
        state['rng']['cuda'] = torch.Generator().manual_seed(192).get_state()
        self.synthetic = state; self.proof = dict(status='passed', formal_binding_sha256=digest(self.formal))
        for name, origin in [('uninterrupted', None), ('resumed_from_update1', 1)]:
            root = self.root / name; checkpoints = root / 'checkpoints'
            commit(checkpoints, [save_rank(checkpoints, state)])
            path = root / 'gate_update12.json'; write_json(path, api.gate_record(root, state, self.formal, origin))
            self.proof[name] = bound(path)
        self.path = self.root / 'synthetic_proof.json'; write_json(self.path, self.proof)

    def test_cpu_state_cannot_be_presented_as_real_gate(self):
        with self.assertRaisesRegex(ValueError, 'CPU state'): api.gate_record(self.root, self.cpu, self.formal, None)

    def test_synthetic_parser_fixture_checks_every_bound_checkpoint(self):
        self.assertEqual(api.check_gate(self.path, self.formal)['status'], 'passed')

    def test_gate_receipt_without_actual_checkpoint_not_sufficient(self):
        (self.root / 'uninterrupted/checkpoints/update_000012/rank_00.pt').unlink()
        with self.assertRaises(FileNotFoundError): api.check_gate(self.path, self.formal)

    def test_other_formal_binding_rejected(self):
        with self.assertRaisesRegex(ValueError, 'matching verified GPU gate'):
            api.check_gate(self.path, dict(self.formal, other=True))

    def test_gate_rejects_frozen_module_change(self):
        state = copy.deepcopy(self.synthetic); state['model']['model.head.bias'] += 1
        with self.assertRaisesRegex(ValueError, 'intended module'):
            api.gate_record(self.root / 'uninterrupted', state, self.formal, None)

    def test_gate_rejects_unchanged_active_module(self):
        formal = copy.deepcopy(self.formal)
        active = {k[len('model.matcher.'):]: v for k, v in self.synthetic['model'].items() if k.startswith('model.matcher.')}
        formal['model_spec']['initial_matcher_state_sha256'] = tree_sha(active)
        state = copy.deepcopy(self.synthetic); state['binding'] = dict(formal, run_mode='gate')
        with self.assertRaisesRegex(ValueError, 'intended module'): api.gate_record(self.root / 'uninterrupted', state, formal, None)

    def test_gate_with_validation_or_wrong_stop_rejected(self):
        state = copy.deepcopy(self.synthetic); state['observations'] = [dict(synthetic=True)]
        with self.assertRaisesRegex(ValueError, 'invalid gate state'):
            api.gate_record(self.root / 'uninterrupted', state, self.formal, None)


def gloo_worker(rank, root, rendezvous, state):
    """Actual two-process collectives; tensors/protocol are CPU fixtures."""
    torch.set_num_threads(1); root = Path(root)
    dist.init_process_group('gloo', rank=rank, world_size=2, init_method='file://' + rendezvous, timeout=timedelta(seconds=30))
    results = []
    try:
        state = copy.deepcopy(state); state['rank'] = rank
        state['sampling'].update(world_size=2, microbatch=1, accumulate=2)
        state['rng']['torch'] = torch.Generator().manual_seed(81 + rank).get_state()
        callback = api.CompleteCheckpoint(root / 'success', rank, 2); callback(state); results.append('commit')
        bad = copy.deepcopy(state)
        if rank == 1: bad['model']['model.head.bias'] += 1
        try: api.CompleteCheckpoint(root / 'diverged', rank, 2)(bad)
        except RuntimeError as error: results.append('divergence' if 'ranks disagree' in str(error) else str(error))
        # A rank-local save error must propagate to its peer without hanging.
        if rank == 1: save_rank(root / 'save_error', state)
        dist.barrier()
        try: api.CompleteCheckpoint(root / 'save_error', rank, 2)(state)
        except RuntimeError as error: results.append('save_error' if 'save failed' in str(error) else str(error))
        def fail(): raise OSError('synthetic rank0 failure')
        try: api.rank0_call(fail, rank, 2, 'init')
        except RuntimeError as error: results.append('rank0_error' if 'synthetic rank0 failure' in str(error) else str(error))
        write_json(root / ('rank%d_result.json' % rank), results)
    finally:
        dist.destroy_process_group()


class DistributedExecutionTests(unittest.TestCase):
    def test_actual_gloo_commit_and_failure_propagation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); _, _, _, state = train_fixture(root / 'tiny_cpu', stop=7)
            mp.spawn(gloo_worker, args=(str(root), str(root / 'rendezvous'), state), nprocs=2, join=True)
            for rank in range(2):
                self.assertEqual(json.loads((root / ('rank%d_result.json' % rank)).read_text()),
                    ['commit', 'divergence', 'save_error', 'rank0_error'])
            self.assertTrue((root / 'success/last_committed.json').is_file())
            self.assertFalse((root / 'diverged/last_committed.json').exists())
            self.assertFalse((root / 'save_error/last_committed.json').exists())


if __name__ == '__main__':
    unittest.main()
