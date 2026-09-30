import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import data_admission as admission
from .catalog import legacy_numerical_digest, deduplicate_basic
from .exposure import digest
from .test_catalog import fixture as sample_fixture


class ReleaseFixture:
    """Small disk receipt chain; loader tensors are explicit synthetic fixtures."""
    def __init__(self, base, duplicate=None):
        self.base = Path(base); self.root = self.base / 'hard'
        self.pipeline = self.base / 'pipeline'; self.train = self.base / 'v17' / 'train'
        self.samples = {}; self.rows = {}; self.audits = {}
        original = []
        for i in range(8):
            row, audit = self.row(self.train, 'basic-' + str(i), i % 2 == 0, i + 1, 'base-' + str(i))
            original.append(row); self.audits[row['pair_id']] = audit
        if duplicate:
            source = self.samples[original[2]['sample_path']]
            sample = copy.deepcopy(source); sample.pair_id = original[4]['pair_id']
            if duplicate == 'targets':
                sample.target_a[0] = -1
            if duplicate == 'recipe':
                original[4]['recipe'] = 'wave'
            self.samples[original[4]['sample_path']] = sample
            self.audits[sample.pair_id]['model_input_sha256'] = legacy_numerical_digest(sample)
        for i in range(0, 8, 2):
            path = self.train / 'groups' / (str(i) + '.json')
            self.write(path, dict(records=original[i:i + 2]))
            group = dict(status='passed', group_sha256=admission.file_sha(path),
                         rows=[self.audits[r['pair_id']] for r in original[i:i + 2]])
            self.write(self.train / 'audits' / path.name, group)
        manifest = dict(split='train', augmentation_revision=admission.REVISION, entries=original)
        original_path = self.train / 'train.json'; archive_path = self.train / 'archive_manifest.json'
        self.write(original_path, manifest); self.write(archive_path, manifest)
        self.write(self.train / 'full_pixel_audit.json', dict(status='passed', checked_pairs=8,
            manifest_sha256=admission.file_sha(original_path), actual_loader_and_supervision_checked_all=True,
            all_pixels_and_endpoints_checked=True,
            group_receipts=[self.spec(self.train / 'audits' / (str(i) + '.json')) for i in range(0, 8, 2)]))
        full_path = self.train.parent / 'full_audit.json'
        self.write(full_path, dict(status='passed', failures=0, source_disjoint=True,
            target_and_donor_audit=True, endpoint_and_area_pixel_audit=True,
            fallback_numerical_identity_audit=True,
            manifest_sha256=dict(train=admission.file_sha(original_path)),
            split_audits=dict(train=self.spec(self.train / 'full_pixel_audit.json'))))
        self.contract = self.train.parent / 'data_contract.json'
        self.write(self.contract, dict(status='passed', source_disjoint=True,
            augmentation_revision=admission.REVISION, online_mirror_probability=0.,
            train=dict(self.spec(original_path), archive_manifest=str(archive_path),
                archive_manifest_sha256=admission.file_sha(archive_path), pairs=8, pair_count=8,
                positives=4, negatives=4), aggressive_full_audit=self.spec(full_path)))
        hard = []; versions = {}
        for ordinal, stage in enumerate(('v17.5', 'v18')):
            rows = []; receipts = []
            for i in range(2):
                key = 'base-' + str(i) if ordinal == 0 else 'hard-only-' + str(i)
                row, receipt = self.row(self.root / stage, stage + '-' + str(i), i == 0,
                                        100 + 2 * ordinal + i, key)
                row['version'] = stage; rows.append(row); receipts.append(receipt)
            self.write(self.root / stage / 'manifest.json', dict(split='train', entries=rows))
            self.write(self.root / stage / 'pixel_audit.json', dict(status='passed', pairs=2,
                all_rows_rechecked_after_generation=True, receipts=receipts))
            versions[stage] = dict(pairs=2,
                manifest_sha256=admission.file_sha(self.root / stage / 'manifest.json'),
                audit_sha256=admission.file_sha(self.root / stage / 'pixel_audit.json'))
            hard.extend(rows)
        self.write(self.root / 'curriculum_base_exclusion.json', dict(status='full_generation_passed',
            source_pair_keys=[admission.original_key(r) for r in hard]))
        self.write(self.root / 'v17_curriculum_base_reference_manifest.json', dict(
            entries=[admission.normalize_archive_row(r, self.train) for r in original[2:]],
            archive_manifest=str(archive_path), archive_sha256=admission.file_sha(archive_path),
            original_rows=8, excluded_rows=2, training_started=False))
        complete = dict(status='complete', source_binding_sha256=digest('generation source'),
            base_exclusion_sha256=admission.file_sha(self.root / 'curriculum_base_exclusion.json'))
        self.write(self.root / 'pipeline_complete.json', complete)
        self.write(self.root / 'generation_complete.json', complete)
        complete_sha = admission.file_sha(self.root / 'pipeline_complete.json')
        self.write(self.pipeline / 'pipeline_complete.json', dict(status='complete', complete_sha256=complete_sha))
        self.write(self.pipeline / 'full_generation_return.json', dict(returncode=0))
        self.verification = self.base / 'verification.json'
        self.write(self.verification, dict(status='passed', training_started=False, gpu_used=False,
            heldout_family_overlap=[], versions=versions, pipeline_complete_sha256=complete_sha,
            source_binding_sha256=complete['source_binding_sha256']))

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))

    @staticmethod
    def spec(path):
        return dict(path=str(path), sha256=admission.file_sha(path))

    def change(self, path, fn):
        value = json.loads(path.read_text()); fn(value); self.write(path, value)

    def row(self, root, name, label, number, base_key):
        path = root / 'samples' / (name + '.npz'); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('explicit synthetic loader fixture: ' + name).encode())
        sample = sample_fixture(); sample.pair_id = name; sample.label = np.float32(label)
        sample.points_rc_a += number
        self.samples[str(path)] = sample
        row = dict(pair_id=name, label=label, recipe='clean', source_root='/original',
            source_pair_id=base_key, source_row=dict(split='train'),
            source_base_key='/original::' + base_key, artifact_path='samples/' + path.name,
            sample_path=str(path), sample_sha256=admission.file_sha(path))
        self.rows[name] = row
        audit = dict(id=name, status='passed', sample_sha256=row['sample_sha256'],
                     model_input_sha256=legacy_numerical_digest(sample))
        return row, audit

    def run(self):
        with patch.dict(admission.HARD_COUNTS, {'v17.5': 2, 'v18': 2}, clear=True):
            return admission.prepare(self.root, self.pipeline, self.verification, self.contract,
                                     lambda path: (self.samples[str(path)], {}))


class DataAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.f = ReleaseFixture(self.temp.name)

    def test_complete_release_emits_actual_counts_without_modifying_inputs(self):
        before = {str(p): admission.file_sha(p) for p in Path(self.temp.name).rglob('*') if p.is_file()}
        result = self.f.run()
        self.assertEqual(result['versions']['v17_filtered'], dict(pairs=6, positives=3, negatives=3))
        self.assertEqual(len(result['catalog']), 10)
        self.assertEqual(result['basic_rows_excluded_by_original_identity'], 2)
        self.assertEqual(result['actual_sample_files_hashed'], 10)
        self.assertFalse(result['budget_locked']); self.assertFalse(result['training_started'])
        self.assertEqual(before, {str(p): admission.file_sha(p) for p in Path(self.temp.name).rglob('*') if p.is_file()})

    def test_training_entries_keep_recipe_absolute_sample_and_separate_hashes(self):
        result = self.f.run(); row = result['manifests']['v18']['entries'][0]
        self.assertTrue(Path(row['sample_path']).is_absolute()); self.assertEqual(row['recipe'], 'clean')
        self.assertNotEqual(row['actual_matcher_input_sha256'], row['effective_training_target_sha256'])
        self.assertFalse(result['manifests']['v18']['online_augmentation'])

    def test_partial_generation_is_not_admitted(self):
        (self.f.root / 'generation_complete.json').unlink()
        with self.assertRaises(FileNotFoundError): self.f.run()

    def test_failure_wins_over_stale_complete(self):
        self.f.write(self.f.pipeline / 'pipeline_failure.json', dict(error='synthetic'))
        with self.assertRaisesRegex(ValueError, 'takes precedence'): self.f.run()

    def test_nonzero_generation_return_rejected(self):
        self.f.write(self.f.pipeline / 'full_generation_return.json', dict(returncode=1))
        with self.assertRaisesRegex(ValueError, 'successfully'): self.f.run()

    def test_final_verification_must_refer_to_this_release(self):
        self.f.change(self.f.verification, lambda v: v.update(pipeline_complete_sha256=digest('other')))
        with self.assertRaisesRegex(ValueError, 'bound artifact differs'): self.f.run()

    def test_sample_mutation_after_pixel_audit_rejected(self):
        Path(self.f.rows['v18-0']['sample_path']).write_bytes(b'mutated')
        with self.assertRaisesRegex(ValueError, 'sample file differs'): self.f.run()

    def test_bound_hard_manifest_mutation_rejected(self):
        self.f.change(self.f.root / 'v18' / 'manifest.json', lambda v: v['entries'].pop())
        with self.assertRaisesRegex(ValueError, 'bound artifact differs'): self.f.run()

    def test_original_group_mutation_rejected_even_with_passed_summary(self):
        self.f.write(self.f.train / 'groups' / '0.json', dict(records=[]))
        with self.assertRaisesRegex(ValueError, 'bound artifact differs'): self.f.run()

    def test_reference_cannot_silently_drop_extra_rows(self):
        path = self.f.root / 'v17_curriculum_base_reference_manifest.json'
        self.f.change(path, lambda v: v['entries'].pop())
        with self.assertRaisesRegex(ValueError, 'exactly the original'): self.f.run()

    def test_reference_cannot_rewrite_recipe(self):
        path = self.f.root / 'v17_curriculum_base_reference_manifest.json'
        self.f.change(path, lambda v: v['entries'][0].update(recipe='wave'))
        with self.assertRaisesRegex(ValueError, 'exactly the original'): self.f.run()

    def test_reference_cannot_restore_later_stage_base(self):
        path = self.f.root / 'v17_curriculum_base_reference_manifest.json'
        self.f.change(path, lambda v: v['entries'].append(self.f.rows['basic-0']))
        with self.assertRaisesRegex(ValueError, 'exactly the original'): self.f.run()

    def test_online_augmentation_not_added(self):
        self.f.change(self.f.contract, lambda v: v.update(online_mirror_probability=.15))
        with self.assertRaisesRegex(ValueError, 'data contract'): self.f.run()

    def test_identical_basic_inputs_deduplicated_and_class_imbalance_disclosed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ReleaseFixture(tmp, duplicate='same').run()
        self.assertEqual(result['versions']['v17_filtered'], dict(pairs=5, positives=2, negatives=3))
        self.assertEqual(len(result['exact_input_dedup']['discarded_exact_duplicates']), 1)
        self.assertEqual(result['versions']['v18']['pairs'], 2)
        self.assertTrue(result['class_balance_is_exposure_not_row_deletion'])

    def test_identical_inputs_with_conflicting_targets_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'conflicting supervision'):
                ReleaseFixture(tmp, duplicate='targets').run()

    def test_clean_only_pose_supervision_is_part_of_conflict_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'conflicting supervision'):
                ReleaseFixture(tmp, duplicate='recipe').run()

    def test_effective_target_recipe_gate_not_arbitrary_recipe_name(self):
        row = self.f.rows['basic-2']; audit = self.f.audits['basic-2']
        loader = lambda path: (self.f.samples[str(path)], {})
        clean = admission.inspect_training_row('v17_filtered', row, audit, loader)
        wave = admission.inspect_training_row('v17_filtered', dict(row, recipe='wave'), audit, loader)
        gap = admission.inspect_training_row('v17_filtered', dict(row, recipe='gaps'), audit, loader)
        self.assertNotEqual(clean.target_sha256, wave.target_sha256)
        self.assertEqual(wave.target_sha256, gap.target_sha256)
        self.assertEqual(clean.ref.model_input_sha256, wave.ref.model_input_sha256)

    def test_missing_recipe_not_silently_assumed_clean(self):
        row = dict(self.f.rows['basic-2']); row.pop('recipe')
        with self.assertRaisesRegex(ValueError, 'recipe required'):
            admission.inspect_training_row('v17_filtered', row, self.f.audits['basic-2'],
                lambda path: (self.f.samples[str(path)], {}))

    def test_duplicate_identity_helper_rejects_overwrite(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            admission.indexed([dict(id='a'), dict(id='a')], 'id', 'receipt')

    def test_metadata_bindings_detect_late_mutation(self):
        bindings = admission.Bindings(); bindings.read(self.f.contract)
        self.f.change(self.f.contract, lambda v: v.update(status='failed'))
        with self.assertRaisesRegex(ValueError, 'admission input changed'): bindings.verify()


if __name__ == '__main__':
    unittest.main()
