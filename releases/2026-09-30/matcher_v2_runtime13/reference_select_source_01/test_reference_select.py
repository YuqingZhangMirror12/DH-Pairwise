import copy
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import admit_reference_select as admission
import evaluate_reference_select as evaluator


def manifest():
    rows = []
    for recipe, count in admission.COUNTS.items():
        for label in (True, False):
            for index in range(count):
                name = f'{recipe}/{label}/{index}'
                rows.append(dict(pair_id=name, id=name, recipe=recipe, label=label,
                    split='select', generator='claude-straight-seam-v4'))
    return dict(schema='claude-straight-seam-v4/1', split='select', revision='v4.2', seed=26093015, failed=[], entries=rows)


class ManifestTests(unittest.TestCase):
    def test_preserves_full_900_and_original_order(self):
        data = manifest(); rows = admission.validate_manifest(data)
        self.assertIs(rows, data['entries'])
        self.assertEqual(sum(e['label'] for e in rows), 450)

    def test_missing_or_duplicate_rows_rejected(self):
        for mutation in ('missing', 'duplicate', 'wrong_label'):
            data = manifest()
            if mutation == 'missing':
                data['entries'].pop()
            elif mutation == 'duplicate':
                data['entries'][-1] = data['entries'][0]
            else:
                data['entries'][0]['label'] = False
            with self.assertRaises(ValueError):
                admission.validate_manifest(data)

    def test_train_test_or_incomplete_manifest_not_admitted(self):
        for key, value in (('split', 'test'), ('split', 'train'), ('failed', ['failure']), ('seed', 26093085)):
            data = manifest(); data[key] = value
            with self.assertRaises(ValueError):
                admission.validate_manifest(data)

    def test_file_binding_and_cpu_only_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'proof.json'; admission.save(path, dict(status='passed'))
            signature = admission.sha(path)
            self.assertEqual(admission.read_bound(path, signature), dict(status='passed'))
            with self.assertRaisesRegex(ValueError, 'bound input'):
                admission.read_bound(path, '0'*64)
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                admission.run(SimpleNamespace())


class DonorTests(unittest.TestCase):
    def fixture(self):
        root = Path('/synthetic_pool')
        relative = 'model/masks_800/gen2voronoi_1/10/2.png'
        source = dict(path=str(root/relative), source_family='synthetic-family', file_sha256='a'*64)
        entry = dict(source_root=str(root), recipe='straight_J', label=True,
            meta=dict(type='J', base='torn_rachel', generator_revision='v4.2',
                      source_mask=relative, source_family=source['source_family']))
        return entry, root, {source['path']: source}

    def test_registered_synthetic_source_retains_identity(self):
        entry, root, lookup = self.fixture()
        actual = admission.donor_records(entry, root, lookup)
        self.assertEqual(actual, [dict(kind='synthetic_voronoi', **next(iter(lookup.values())))])

    def test_real_unsafe_or_train_donor_path_rejected(self):
        entry, root, lookup = self.fixture()
        for path in ('/real/dunhuang/mask.png', '../../model/masks_800/gen2voronoi_1/10/2.png',
                     'model/masks_800/real/10/2.png', 'model/masks_800/gen2voronoi_1/999/2.png'):
            bad = copy.deepcopy(entry); bad['meta']['source_mask'] = path
            with self.assertRaises(ValueError):
                admission.donor_records(bad, root, lookup)

    def test_wrong_family_or_unregistered_root_rejected(self):
        entry, root, lookup = self.fixture()
        entry['meta']['source_family'] = 'train-family'
        with self.assertRaisesRegex(ValueError, 'source-family'):
            admission.donor_records(entry, root, lookup)
        entry['source_root'] = '/real'
        with self.assertRaisesRegex(ValueError, 'source root'):
            admission.donor_records(entry, root, lookup)

    def test_procedural_not_claimed_to_have_source_file(self):
        entry, root, lookup = self.fixture()
        entry['recipe'] = 'straight_R'; entry['meta'] = dict(type='R', base='strip', generator_revision='v4.2')
        self.assertEqual(admission.donor_records(entry, root, lookup), [dict(kind='procedural', base='strip')])
        entry['meta']['source_mask'] = '/real/borrowed.png'
        with self.assertRaisesRegex(ValueError, 'undeclared donor'):
            admission.donor_records(entry, root, lookup)

    def test_cross_source_negative_cannot_reuse_parent(self):
        entry, root, lookup = self.fixture(); leaf = entry['meta']
        entry['label'] = False
        entry['meta'] = dict(type='J', base='torn_rachel', generator_revision='v4.2', a=leaf, b=leaf)
        with self.assertRaisesRegex(ValueError, 'negative reuses'):
            admission.donor_records(entry, root, lookup)

    def test_existing_source_collision_and_real_match_fail(self):
        selected = [dict(path=f'/sim/{i}', source_family=f'f{i}', file_sha256=f's{i}',
                         pixel_sha256=f'p{i}', crop_pixel_sha256=f'c{i}') for i in range(150)]
        audit = dict(status='passed', errors=[], collision_counts=dict(train_select=0), source_root='/sim',
                     inventory=dict(select=selected, train=[], test=[]))
        real = dict(source_admission_sha256=admission.BOUND['sources'][1], synthetic_inventory=dict(select=dict(
            invalid_synthetic_paths=[], exact_prepared_mask_matches=[], potential_turufan_parent_aliases={})))
        self.assertEqual(len(admission.source_lookup(audit, real)[1]), 150)
        audit['inventory']['train'] = [selected[0]]
        with self.assertRaisesRegex(ValueError, 'collision'):
            admission.source_lookup(audit, real)
        audit['inventory']['train'] = []
        real['synthetic_inventory']['select']['exact_prepared_mask_matches'] = ['real']
        with self.assertRaisesRegex(ValueError, 'real donor'):
            admission.source_lookup(audit, real)


class SampleTests(unittest.TestCase):
    def make_fixture(self, tmp, positive=True):
        loader = importlib.import_module('staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset')
        catalog = importlib.import_module(admission.PACKAGE + '.curriculum_training_v1.catalog')
        model = importlib.import_module('staging.pairwise_v0_2.pairwise_data.rachel_training_dataset')
        mask = np.zeros((1, 800, 800), np.float32); mask[:, 100:300, 100:300] = 1
        meta = dict(type='M', base='rachel', shift_a=[0, 0], shift_b=[8, -6], generator_revision='v4.2')
        sample = model.RachelPairSample(pair_id='fixture', fragment_a_token='a', fragment_b_token='b',
            mask_a=mask, mask_b=mask.copy(), coarse_mask_a=np.zeros((1, 128, 128), np.float32),
            coarse_mask_b=np.zeros((1, 128, 128), np.float32),
            points_rc_a=np.zeros((512, 2), np.float32), points_rc_b=np.zeros((512, 2), np.float32),
            contour_valid_a=np.ones(512, bool), contour_valid_b=np.ones(512, bool),
            target_a=np.full(512, -1, np.int64), target_b=np.full(512, -1, np.int64),
            label=np.float32(positive), translation_valid=np.bool_(positive),
            translation_a_to_b_rc=np.array([8, -6] if positive else [0, 0], np.float32),
            translation_a_to_b_xy_cartesian=np.array([-6, -8] if positive else [0, 0], np.float32))
        path = Path(tmp)/'samples/fixture.npz'
        report = dict(split='select', pose_supervision_enabled=positive, changed_pair=False,
                      generator_meta=meta, gt_used_for_inputs=False)
        loader.save_sample(path, sample, report)
        entry = dict(pair_id='fixture', label=positive, recipe='straight_M', sample_path=str(path),
            artifact_path='samples/fixture.npz', sample_sha256=admission.sha(path), meta=meta)
        return entry, loader, catalog

    def test_actual_official_loader_full_shape_and_canonical_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry, loader, catalog = self.make_fixture(tmp)
            row = admission.inspect_entry(entry, loader.load_sample, catalog, reference_root=Path(tmp))
            sample, _ = loader.load_sample(entry['sample_path'])
            self.assertEqual(row['model_input_sha256'], catalog.tensor_digest(sample, catalog.MATCHER_INPUTS))
            self.assertNotEqual(row['target_sha256'], row['model_input_sha256'])

    def test_hash_path_label_and_gt_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry, loader, catalog = self.make_fixture(tmp)
            for field, value in (('sample_sha256', '0'*64), ('pair_id', 'other'), ('label', False),
                                 ('artifact_path', '../fixture.npz')):
                bad = copy.deepcopy(entry); bad[field] = value
                with self.assertRaises(ValueError):
                    admission.inspect_entry(bad, loader.load_sample, catalog, reference_root=Path(tmp))
            bad = copy.deepcopy(entry); bad['meta']['shift_b'] = [2, 2]
            with self.assertRaises(ValueError):
                admission.inspect_entry(bad, loader.load_sample, catalog, reference_root=Path(tmp))

    def test_exact_input_duplicates_are_separate_from_shared_sources(self):
        row = lambda p, h: dict(pair_id=p, model_input_sha256=h)
        proof = admission.duplicate_report([row('r1', 'a'), row('r2', 'a'), row('r3', 'c')],
            [row('train', 'a')], [row('new', 'c')])
        self.assertEqual(proof['within_reference'], [['r1', 'r2']])
        self.assertEqual(proof['with_training'][0]['training'], ['train'])
        self.assertEqual(proof['with_new_select'][0]['reference'], ['r3'])


class EvaluationTests(unittest.TestCase):
    def test_forward_tensor_filter_excludes_gt_and_metadata(self):
        import torch
        api = importlib.import_module(admission.PACKAGE + '.curriculum_training_v1.matcher_population')
        batch = {k: np.zeros((1, 2), bool if k.startswith('contour_valid') else np.float32) for k in api.INPUTS}
        batch.update(label='must not be sent', translation_a_to_b_rc='must not be sent', source_family='private')
        tensors = api.tensor_inputs(batch, torch.device('cpu'))
        self.assertEqual(set(tensors), set(api.INPUTS))
        self.assertTrue(all(v.device.type == 'cpu' for v in tensors.values()))

    def test_cuda_request_rejected_before_loading_model(self):
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}), \
             patch.object(evaluator, 'terminal_model', side_effect=AssertionError('not reached')):
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                evaluator.run(SimpleNamespace())

    def test_incomplete_terminal_cannot_load_population(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}), \
             patch.object(evaluator, 'terminal_model', side_effect=ValueError('not complete')) as model, \
             patch.object(evaluator, 'verify_admission', side_effect=AssertionError('must not read population')):
            with self.assertRaisesRegex(ValueError, 'not complete'):
                evaluator.run(SimpleNamespace(out=Path(tmp)/'out'))
            model.assert_called_once()

    def test_gt_join_requires_complete_frozen_predictions(self):
        class Dataset:
            def __len__(self):
                return 1
            def __getitem__(self, i):
                raise AssertionError('GT must not be read')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                evaluator.targets_after_freeze(tmp, Dataset())
            admission.save(Path(tmp)/'prediction_complete.json', dict(pairs=0))
            with self.assertRaisesRegex(ValueError, 'durable full'):
                evaluator.targets_after_freeze(tmp, Dataset())

    def test_postfreeze_gt_is_unchanged_and_negative_gt_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            admission.save(root/'pair_predictions.jsonl', dict(pair_id='a'))
            admission.save(root/'prediction_complete.json', dict(pairs=2, targets_joined=False,
                model_state_unchanged=True, prediction_sha256=admission.sha(root/'pair_predictions.jsonl')))
            dataset = [(SimpleNamespace(label=True, translation_valid=True,
                translation_a_to_b_rc=np.array([8, -6])), None, dict(pair_id='a')),
                (SimpleNamespace(label=False, translation_valid=False), None, dict(pair_id='b'))]
            self.assertEqual(evaluator.targets_after_freeze(root, dataset), [dict(pair_id='a', label=True, gt_pose=[8, -6]),
                dict(pair_id='b', label=False, gt_pose=None)])

    def test_failure_precedes_stale_admission_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            admission.save(Path(tmp)/'failure.json', dict(status='failed'))
            with self.assertRaisesRegex(ValueError, 'failure precedes'):
                evaluator.verify_admission(tmp, 'unknown')

    def test_existing_output_not_overwritten_or_retried(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}), \
             patch.object(evaluator, 'terminal_model', side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError, 'exclusive'):
                evaluator.run(SimpleNamespace(out=Path(tmp)))


class ControllerTests(unittest.TestCase):
    def test_cpu_only_environment_and_no_gpu_scheduler(self):
        import run_reference_pipeline as controller
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,5', 'OMP_NUM_THREADS': '32'}):
            env = controller.cpu_environment()
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['OMP_NUM_THREADS'], '1')
        self.assertEqual(env['MKL_NUM_THREADS'], '1')
        command, out = controller.commands(Path('/output'))
        self.assertEqual(out, Path('/output/admission'))
        self.assertIn('runtime_work_13', ' '.join(command))
        self.assertNotIn('launch_training', ' '.join(command))

    def test_native_b0_uses_terminal_not_checkpoint_path(self):
        import run_reference_pipeline as controller
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); admission.save(root/'complete.json', dict(status='complete'))
            command = controller.evaluation_command(root, root)
            self.assertIn('--training-root', command)
            self.assertNotIn('--checkpoint', command)
            self.assertEqual(command[command.index('--arm')+1], 'B0')
            self.assertEqual(command[command.index('--selection')+1], 'sim_best')
            self.assertNotIn('real_best', command)

    def test_preparation_must_bind_exact_unchanged_code(self):
        import run_reference_pipeline as controller
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}):
            root = Path(tmp); proof = root/'prep.json'
            admission.save(proof, dict(status='passed', tests=24, errors=0, failures=0, skipped=0,
                source_unchanged=True, cuda_initialized=False, source_sha256={}))
            with patch.object(controller, 'child', side_effect=AssertionError('must not run')), \
                 self.assertRaisesRegex(ValueError, 'unchanged tested'):
                controller.run(root/'out', proof)
            self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()
