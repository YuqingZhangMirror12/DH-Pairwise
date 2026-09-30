"""Admission/budget/source checks on explicit synthetic manifests; no GPUs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.exposure import STAGES, SampleRef, build_ledger, digest
from ..curriculum_training_v1.runtime_plan import lock_record
from ..curriculum_training_v1.test_execution import manifest_fixture, bound
from .runtime_schedule import compile_plan
from . import runtime_inputs as api


class RuntimeInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.base = manifest_fixture(self.root)
        admission = json.loads(Path(self.base['admission']['path']).read_text()); admission['gpu_used'] = False
        write_json(Path(self.base['admission']['path']), admission, replace=True); self.base['admission'] = bound(Path(self.base['admission']['path']))
        geometry = json.loads(Path(self.base['geometry']['path']).read_text())
        geometry['curriculum_data_admission_sha256'] = self.base['admission']['sha256']
        write_json(Path(self.base['geometry']['path']), geometry, replace=True); self.base['geometry'] = bound(Path(self.base['geometry']['path']))
        self.ledger = build_ledger([SampleRef(**r) for r in admission['catalog']], dict(zip(STAGES, (15000, 6000, 3000))), 260928, 32)
        record = json.loads(Path(self.base['runtime_plan']['path']).read_text())
        record.update(ledger_sha256=self.ledger.sha256, total_updates=24000,
            stage_updates=dict(zip(STAGES, (15000, 6000, 3000))), learning_rate_knots=[[0, 1e-4], [15000, 5e-5], [21000, 2.5e-5]],
            validation_updates=list(range(0, 24001, 1500)), data_admission_sha256=self.base['admission']['sha256'],
            geometry_sha256=self.base['geometry']['sha256'])
        write_json(Path(self.base['runtime_plan']['path']), record, replace=True); self.base['runtime_plan'] = bound(Path(self.base['runtime_plan']['path']))
        self.original = lock_record(record, self.ledger)

    def spec(self, module='matcher'):
        base_path = self.root/'execution.json'; write_json(base_path, self.base, replace=base_path.exists())
        original = api.derived_base_plan(self.original, self.ledger, module)
        _, plan, schedule = compile_plan(original, self.ledger, 'B2')
        plan_path = self.root/'v2_plan.json'; schedule_path = self.root/'schedule.json'
        write_json(plan_path, plan.record, replace=plan_path.exists()); write_json(schedule_path, schedule, replace=schedule_path.exists())
        return dict(schema='matcher-v2-execution/1', locked=True, arm='B2', module=module,
            base_execution=bound(base_path), source_binding={}, baseline_composition={}, combined_admission=None,
            runtime_plan=bound(plan_path), runtime_schedule=bound(schedule_path),
            topology=dict(world_size=2, microbatch=8, accumulate=2, workers=4) if module == 'matcher' else dict(
                world_size=1, microbatch=32, accumulate=1, workers=4),
            selected_matcher=None if module == 'matcher' else dict(explicit_fixture=True))

    def load(self, spec):
        with patch.object(api, 'verify_composed_source', return_value=(self.root, self.original.record['baseline_sources_sha256'])), \
             patch.object(torch.cuda, 'set_device', side_effect=AssertionError('must not initialize CUDA')):
            return api.load_inputs(spec)

    def test_B2_exact_original_exposures_and_new_head_policy(self):
        for module in api.MODULES:
            value = self.load(self.spec(module))
            self.assertEqual(value['ledger'].total_updates, 24000)
            self.assertEqual(value['ledger'].sequence('curriculum'), self.ledger.sequence('curriculum'))
            self.assertEqual(value['schedule']['added_straight_updates'], 0)
            self.assertEqual(value['plan'].record['selection_rule']['id'], api.RULES[module])

    def test_do_not_use_json_object_order_for_stage_budget(self):
        path = Path(self.base['runtime_plan']['path']); row = json.loads(path.read_text())
        row['stage_updates'] = dict(sorted(row['stage_updates'].items()))
        write_json(path, row, replace=True); self.base['runtime_plan'] = bound(path)
        self.assertEqual(self.load(self.spec())['ledger'].total_updates, 24000)

    def test_changed_schedule_or_budget_not_accepted(self):
        spec = self.spec(); path = Path(spec['runtime_schedule']['path']); row = json.loads(path.read_text())
        row['actual_updates'] += 1; write_json(path, row, replace=True); spec['runtime_schedule'] = bound(path)
        with self.assertRaisesRegex(ValueError, 'exact original'):self.load(spec)
        path = Path(self.base['runtime_plan']['path']); row = json.loads(path.read_text()); row['total_updates'] = 25000
        write_json(path, row, replace=True); self.base['runtime_plan'] = bound(path)
        with self.assertRaisesRegex(ValueError, 'original budget'):self.load(self.spec())

    def test_unregistered_topology_arms_and_extra_data_rejected(self):
        for key, value in [('arm', 'B4'), ('combined_admission', {'path': '/forbidden'}), ('locked', False),
                           ('topology', dict(world_size=1, microbatch=64, accumulate=1, workers=4))]:
            spec = self.spec(); spec[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):self.load(spec)

    def test_augmentation_requires_admission_not_generation(self):
        spec = self.spec(); spec['arm'] = 'B3'
        path = self.root/'generated.json'; write_json(path, dict(status='generated', training_admitted=False))
        spec['combined_admission'] = bound(path)
        with self.assertRaisesRegex(ValueError, 'combined admission'):self.load(spec)

    def test_reference_and_population_content_bound(self):
        spec = self.spec(); Path(self.base['reference_checkpoint']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'reference changed'):self.load(spec)

    def test_actual_composed_snapshot_contains_unchanged_original_baseline(self):
        source = Path(__file__).resolve().parents[4]
        composition = json.loads((source/'baseline_composition.json').read_text())
        base = dict(baseline=dict(python_sha256=composition['python_sha256']))
        spec = dict(source_binding=bound(source/'source_binding.json'), baseline_composition=bound(source/'baseline_composition.json'),
                    base_execution=dict(sha256=composition['execution_sha256']))
        root, sha = api.verify_composed_source(spec, base)
        self.assertEqual(root, source); self.assertEqual(sha, digest(composition['python_sha256']))
        bad = copy.deepcopy(base); bad['baseline']['python_sha256']['forbidden.py'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'original B0'):api.verify_composed_source(spec, bad)


if __name__ == '__main__':unittest.main()
