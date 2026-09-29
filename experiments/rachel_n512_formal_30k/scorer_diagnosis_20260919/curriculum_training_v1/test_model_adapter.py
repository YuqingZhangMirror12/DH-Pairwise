from dataclasses import asdict
import copy
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from .checkpoint_io import file_sha, tree_sha
from .exposure import STAGES, SampleRef, build_ledger, digest
from .model_adapter import AdmittedDataset, make_components
from .runtime_plan import check_light_heads, check_matched_matchers, lock_record
from .test_exposure import samples
from .test_data_admission import ReleaseFixture
from .test_validation_adapter import setup_plan
from .training_core import Topology

BASE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'


class ModelAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ['CURRICULUM_BASELINE_SOURCE'])
        cls.arch_api = importlib.import_module('staging.pairwise_v0_2.models.rachel_n512')
        cls.geometry_api = importlib.import_module(BASE + 's7_consensus_v1.compatibility')

    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.arch = self.arch_api.RachelN512Config(canvas_size=32, coarse_size=32, contour_cap=16,
            landmark_count=2, context_layers=2, activation_checkpointing=False)
        self.geometry = self.geometry_api.CompatibilityConfig(.5, .5, .5, .5, 1., 15.)
        self.topology = Topology(0, 1, 4, 1)

    def build(self, module='matcher', order='curriculum', selected=None, cache=None):
        _, plan = setup_plan(module)
        return make_components(plan, order, self.topology, self.source, self.arch, self.geometry, selected, cache)

    def selected_fixture(self):
        """Actual architecture tensors; fabricated completed-update metadata, not training evidence."""
        parts = self.build(); model = parts['model']
        with torch.no_grad():
            next(p for p in model.matcher.parameters() if p.requires_grad).add_(.01)
        path = self.root / 'synthetic-selected.pt'
        saved = dict(schema='curriculum-model-export/1', binding=parts['binding'], selection_kind='sim_best',
                     updates=7, model=copy.deepcopy(model.state_dict()))
        torch.save(saved, path)
        return dict(path=str(path), sha256=file_sha(path), updates=7, source_binding=parts['binding'])

    def test_matcher_pair_has_identical_fresh_weights_and_only_order_differs(self):
        a = self.build(); b = self.build(order='mixed')
        self.assertEqual(tree_sha(a['model'].state_dict()), tree_sha(b['model'].state_dict()))
        self.assertEqual(check_matched_matchers(a['binding'], b['binding'])['status'], 'matched')
        self.assertIsNone(a['binding']['fixed_matcher_sha256'])
        self.assertFalse(any(p.requires_grad for p in a['model'].head.parameters()))
        self.assertTrue(any(p.requires_grad for p in a['model'].matcher.parameters()))

    def test_constructor_does_not_consume_global_torch_rng(self):
        before = torch.get_rng_state().clone(); self.build()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_matcher_refuses_fixed_weights_and_frozen_proposal_cache(self):
        with self.assertRaisesRegex(ValueError, 'random start'): self.build(selected={'sha256': 'a' * 64})
        with self.assertRaisesRegex(ValueError, 'cache'): self.build(cache=object())

    def test_both_heads_import_only_same_matcher_and_initialize_expected_sizes(self):
        selected = self.selected_fixture(); a = self.build('scorer_patch', selected=selected)
        b = self.build('scorer_stats', selected=selected)
        self.assertEqual(check_light_heads(a['binding'], b['binding'])['status'], 'matched')
        self.assertEqual(tree_sha(a['model'].matcher.state_dict()), tree_sha(b['model'].matcher.state_dict()))
        self.assertEqual(sum(p.numel() for p in a['model'].head.parameters()), 34529)
        self.assertEqual(sum(p.numel() for p in b['model'].head.parameters()), 3201)
        for parts in (a, b):
            self.assertFalse(any(p.requires_grad for p in parts['model'].matcher.parameters()))
            self.assertTrue(all(p.requires_grad for p in parts['model'].head.parameters()))
            self.assertFalse(parts['binding']['model_spec']['old_head_imported'])
            self.assertFalse(parts['binding']['model_spec']['optimizer_imported'])
            parts['module'].train()
            self.assertFalse(parts['model'].matcher.base.training)

    def test_no_hidden_mixed_scorer_experiment(self):
        with self.assertRaisesRegex(ValueError, 'not registered'):
            self.build('scorer_patch', order='mixed', selected=self.selected_fixture())

    def test_selected_file_mutation_is_not_accepted(self):
        selected = self.selected_fixture(); Path(selected['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'export changed'): self.build('scorer_patch', selected=selected)

    def test_source_loader_import_must_be_bound(self):
        _, plan = setup_plan()
        with self.assertRaisesRegex(ValueError, 'different model/loader source'):
            make_components(plan, 'curriculum', self.topology, self.root, self.arch, self.geometry)

    def test_effective_batch_mismatch_rejected_before_model_construction(self):
        _, plan = setup_plan()
        with self.assertRaisesRegex(ValueError, 'global batch'):
            make_components(plan, 'curriculum', Topology(0, 1, 8, 1), self.source, self.arch, self.geometry)

    def test_head_recipe_loss_adapter_uses_existing_bound_implementations(self):
        selected = self.selected_fixture(); parts = self.build('scorer_stats', selected=selected)
        self.assertEqual(parts['module'].stage, 'scorer')
        self.assertEqual(parts['config'].scorer_variant, 'stats')
        self.assertEqual(parts['config'].effective_batch, 4)
        self.assertEqual(parts['collate'].__module__, BASE + 's7_consensus_v1.data')

    def test_single_gpu_micro32_training_keeps_micro8_validation(self):
        selected = self.selected_fixture(); _, original = setup_plan('scorer_patch')
        ledger = build_ledger(samples((16, 16, 16)), dict(zip(STAGES, (1, 1, 1))), 4, effective_batch=32)
        record = copy.deepcopy(original.record)
        record.update(ledger_sha256=ledger.sha256, total_updates=3,
            stage_updates=dict(zip(STAGES, (1, 1, 1))), effective_batch=32, seed=4,
            learning_rate_knots=[[0, .0001]], validation_updates=[0, 1, 2, 3])
        plan = lock_record(record, ledger)
        parts = make_components(plan, 'curriculum', Topology(0, 1, 32, 1),
                                self.source, self.arch, self.geometry, selected)
        self.assertEqual(parts['config'].microbatch, 32)
        self.assertEqual(parts['config'].effective_batch, 32)
        self.assertEqual(parts['validation_config'].microbatch, 8)
        self.assertEqual(parts['binding']['model_spec']['validation_microbatch'], 8)


class DatasetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.f = ReleaseFixture(self.root)
        self.admission = self.f.run()
        # Admission canonicalizes symlinks, including macOS /var -> /private/var.
        self.f.samples.update({str(Path(k).resolve()): v for k, v in self.f.samples.items()})
        refs = [SampleRef(**r) for r in self.admission['catalog']]
        self.ledger = build_ledger(refs, dict(zip(STAGES, (3, 1, 1))), 4, effective_batch=2)
        outputs = {}
        for stage, manifest in self.admission.pop('manifests').items():
            path = self.root / ('admitted-' + stage + '.json')
            path.write_text(json.dumps(manifest)); outputs[stage] = dict(path=str(path), sha256=file_sha(path))
        self.admission['training_manifests'] = outputs
        self.path = self.root / 'admission_complete.json'; self.save()
        self.loader = lambda p: (self.f.samples[str(p)], {'explicit_synthetic_fixture': True})

    def save(self):
        self.path.write_text(json.dumps(self.admission))

    def dataset(self):
        return AdmittedDataset(self.path, file_sha(self.path), self.ledger, '.', self.loader)

    def test_actual_unique_index_space_and_unchanged_loader_arrays(self):
        dataset = self.dataset(); self.assertEqual(len(dataset), 10)
        for index, ref in enumerate(self.ledger.catalog):
            sample, report, entry = dataset[index]
            self.assertEqual(sample.pair_id, ref.pair_id)
            self.assertIs(sample, self.f.samples[ref.sample_path])
            self.assertEqual(entry['recipe'], 'clean')
        self.assertEqual(len(dataset.verified), 10)

    def test_manifest_hash_binding_checked(self):
        Path(self.admission['training_manifests']['v18']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'manifest changed'): self.dataset()

    def test_actual_file_hash_on_first_read(self):
        dataset = self.dataset(); Path(dataset.entries[0]['sample_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'sample changed'): dataset[0]

    def test_actual_input_arrays_not_just_filename_checked(self):
        dataset = self.dataset(); self.f.samples[dataset.entries[0]['sample_path']].points_rc_a += 1
        with self.assertRaisesRegex(ValueError, 'six Matcher inputs'): dataset[0]

    def test_actual_supervision_arrays_checked(self):
        dataset = self.dataset(); self.f.samples[dataset.entries[0]['sample_path']].target_a[0] = -1
        with self.assertRaisesRegex(ValueError, 'supervision differs'): dataset[0]

    def test_stale_catalog_identity_rejected(self):
        self.admission['catalog'][0]['sample_sha256'] = 'a' * 64; self.save()
        with self.assertRaisesRegex(ValueError, 'index order or membership'): self.dataset()


if __name__ == '__main__':
    unittest.main()
