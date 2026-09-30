"""Synthetic file/loader fixtures, never a real-dataset admission receipt."""
import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ..curriculum_training_v1.catalog import MATCHER_INPUTS, tensor_digest
from ..curriculum_training_v1.checkpoint_io import file_sha
from ..curriculum_training_v1.exposure import digest
from ..curriculum_training_v1.test_catalog import fixture
from .additive_exposure import StraightSampleRef
from . import data_runtime as api
from .prepare_data import canonical_entry


class CanonicalLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve()/'fixture.npz'; self.path.write_bytes(b'explicit CPU loader fixture')
        self.sample = fixture()
        for side in 'ab':
            setattr(self.sample, 'points_rc_'+side, np.arange(20, dtype=np.float32).reshape(10, 2))
            setattr(self.sample, 'contour_valid_'+side, np.ones(10, dtype=bool))
            setattr(self.sample, 'target_'+side, np.array([*range(8), -2, -1], dtype=np.int64))
        self.report = dict(changed_pair=True, pose_supervision_enabled=False)
        self.entry = dict(pair_id='pair', label=True, recipe='straight_J', sample_path=str(self.path),
                          sample_sha256=file_sha(self.path), attempted_donor_references=[],pre_damage_pair_sha256=digest('precut'))
        h = hashlib.sha256()
        for name in MATCHER_INPUTS:
            h.update(name.encode()); h.update(np.ascontiguousarray(getattr(self.sample, name)).tobytes())
        self.audit = dict(pair_id='pair', positive=True, sample_sha256=file_sha(self.path),
                          model_input_sha256=h.hexdigest(), target_audit=dict(correspondence_count=8),
                          pre_damage_pair_sha256=digest('precut'))
        self.loader = lambda path: (self.sample, self.report)

    def row(self):return canonical_entry(self.entry, self.audit, 'train', 42, self.loader)

    def dataset(self):
        # Exercise actual first-read checking without fabricating a full
        # population's admission. Constructor receipts are tested separately.
        dataset = object.__new__(api.CombinedDataset); dataset.base = ['unchanged original tuple']
        dataset.entries = [self.row()]; dataset.loader = self.loader; dataset.verified = set()
        return dataset

    def test_raw_byte_hash_is_not_the_canonical_tensor_hash(self):
        row = self.row()
        self.assertNotEqual(row['actual_matcher_input_sha256'], row['generator_raw_input_sha256'])
        self.assertEqual(row['actual_matcher_input_sha256'], tensor_digest(self.sample, MATCHER_INPUTS))
        self.sample.translation_a_to_b_rc += 2
        changed = self.row()
        self.assertEqual(row['actual_matcher_input_sha256'], changed['actual_matcher_input_sha256'])
        self.assertNotEqual(row['effective_training_target_sha256'], changed['effective_training_target_sha256'])

    def test_original_prefix_and_actual_straight_tensors_not_transformed(self):
        dataset = self.dataset()
        self.assertEqual(dataset[0], 'unchanged original tuple')
        self.assertIs(dataset[1][0], self.sample)
        self.assertEqual(dataset[1][0].target_a[-2:].tolist(), [-2, -1])
        self.assertEqual(dataset.verified, {0})
        with self.assertRaises(ValueError):dataset[-1]

    def test_file_input_target_and_pose_mutation_rejected(self):
        for change, error in [(lambda: self.sample.points_rc_a.__setitem__((0, 0), 99.), 'six straight inputs'),
                              (lambda: self.sample.target_a.__setitem__(0, -2), 'supervision'),
                              (lambda: self.report.update(pose_supervision_enabled=True), 'pose contract'),
                              (lambda: self.path.write_bytes(b'changed'), 'sample changed')]:
            sample, report, content = copy.deepcopy(self.sample), dict(self.report), self.path.read_bytes()
            dataset = self.dataset(); change()
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):dataset[1]
            self.sample = sample; self.report = report; self.path.write_bytes(content)

    def test_fewer_healthy_targets_rejected(self):
        self.sample.target_a[0] = -2
        with self.assertRaisesRegex(ValueError, 'healthy'):self.row()


class AdmissionReceiptTests(unittest.TestCase):
    def fixture(self):
        # Complete-size identity-only rows, no synthetic pixels are admitted.
        entries = []; originals = []; audited = []; refs = []; lineage_rows = []
        for i in range(6000):
            name = f'fixture-{i:04d}'; signature = digest(name)
            positive=i%2 == 0;identity=digest(['precut',i]) if positive else None
            source_key='pre-damage-pair:'+identity if positive else name
            row = dict(pair_id=name, label=positive, sample_path='/synthetic/'+name+'.npz', sample_sha256=signature,
                recipe='straight_J', source_base_key=source_key, actual_matcher_input_sha256=signature,
                pre_damage_pair_sha256=identity,
                generator_raw_input_sha256=digest(['raw', i]))
            entries.append(row); originals.append(dict(row))
            audited.append(dict(pair_id=name, sample_sha256=signature, model_input_sha256=row['generator_raw_input_sha256'],
                                target_audit=dict(correspondence_count=8),pre_damage_pair_sha256=identity))
            refs.append(StraightSampleRef(name, source_key, signature, row['sample_path'], signature, row['label']))
            if positive:lineage_rows.append(dict(pair_id=name,split='train',pre_damage_pair_sha256=identity))
        for split in ('select','test'):
            lineage_rows.extend(dict(pair_id=split+str(i),split=split,pre_damage_pair_sha256=digest([split,i])) for i in range(450))
        record = dict(schema='straight-seam-training-admission/1', status='passed', training_admitted=True,
            gpu_used=False, split='train', online_augmentation=False, entries=entries,
            catalog_sha256=digest([asdict(r) for r in refs]), acceptance='acceptance',
            base_admission=dict(path='base',sha256=digest('base')),
            full_generation_complete=dict(path='complete', sha256=digest('complete')))
        acceptance = dict(schema='straight-seam-population-acceptance/1', status='passed', calibration_split='select',
            test_used_for_selection=False, known_real_donors_excluded=True, source_split_overlap=0,
            full_generation_complete_sha256=digest('complete'),pair_lineage_audit='lineage')
        populations={split:dict(rows=n,requested_rows=n,omitted=0,manifest_path=split,manifest_sha256=digest(split),
                    audit_path='audit' if split=='train' else split+'-audit',audit_sha256=digest('audit' if split=='train' else split+'-audit'))
                    for split,n in [('train',6000),('select',900),('test',900)]}
        complete = dict(status='complete_generation_integrity_supervision', total_rows=7800,
            requested_rows=7800,authorized_omissions=0,cross_split_pre_damage_pair_overlap=0,
            cross_split_model_input_overlap=0, cross_split_source_family_overlap=0,
            datasets=populations)
        lineage=dict(schema='straight-pair-lineage-audit/1',status='passed',failures=[],
            full_generation_complete=record['full_generation_complete'],base_admission=record['base_admission'],
            old_exposures_removed=0,original_registry='registry',records=lineage_rows,
            positive_pre_damage_pairs_checked=3900,negative_rows=3900,splits=populations)
        registry=dict(schema='original-pair-registry/1',status='passed',unresolved_pairs=0,
            base_admission=record['base_admission'],records=[dict(pre_damage_pair_sha256='old')])
        files = dict(acceptance=acceptance, complete=complete, train=dict(split='train', failed=[], entries=originals),
                     lineage=lineage,registry=registry,
                     audit=dict(status='passed_integrity_and_supervision', source_manifest_sha256=digest('train'),
                                records=audited,authorized_omissions=0))
        return record, files

    def check(self, record, files):
        with patch.object(api, 'read_bound', side_effect=lambda spec: files[spec if isinstance(spec, str) else spec['path']]):
            return api.check_straight_record(record)

    def test_generation_or_unreviewed_population_cannot_be_admitted(self):
        record, files = self.fixture()
        self.assertEqual(len(self.check(record, files)[1]), 6000)
        for key, value in [('training_admitted', False), ('split', 'select'), ('online_augmentation', True), ('status', 'generated')]:
            with self.subTest(key=key), self.assertRaises(ValueError):self.check(dict(record, **{key: value}), files)
        files['acceptance']['test_used_for_selection'] = True
        with self.assertRaisesRegex(ValueError, 'SELECT calibration'):self.check(record, files)

    def test_missing_row_changed_sample_or_duplicate_canonical_inputs_rejected(self):
        for mutate in (lambda r: r['entries'].pop(),
                       lambda r: r['entries'][0].update(sample_sha256='a'*64),
                       lambda r: r['entries'][0].update(actual_matcher_input_sha256=r['entries'][1]['actual_matcher_input_sha256'])):
            record, files = self.fixture(); mutate(record)
            with self.assertRaises(ValueError):self.check(record, files)

    def test_pair_level_lineage_is_mandatory(self):
        record,files=self.fixture();files['acceptance'].pop('pair_lineage_audit')
        with self.assertRaisesRegex(ValueError,'lineage'):self.check(record,files)

    def test_old_pair_reused_under_new_name_rejected(self):
        record,files=self.fixture()
        files['registry']['records'][0]['pre_damage_pair_sha256']=files['lineage']['records'][0]['pre_damage_pair_sha256']
        with self.assertRaisesRegex(ValueError,'Repeated pre-damage'):self.check(record,files)

    def test_incomplete_or_stale_lineage_rejected(self):
        for change in (lambda f:f['registry'].update(unresolved_pairs=1),
                       lambda f:f['lineage']['records'].pop(),
                       lambda f:f['lineage'].update(full_generation_complete={'path':'different','sha256':'0'*64})):
            record,files=self.fixture();change(files)
            with self.assertRaises(ValueError):self.check(record,files)

    def test_explicit_authorized_shortfall_not_missing_records(self):
        record,files=self.fixture()
        # Drop one positive from every bound artifact, not just the loader.
        record['entries'].pop(0);files['train']['entries'].pop(0);files['audit']['records'].pop(0)
        files['audit']['authorized_omissions']=1
        files['complete'].update(authorized_omissions=1,total_rows=7799)
        files['complete']['datasets']['train'].update(rows=5999,omitted=1)
        files['lineage']['records'].pop(0);files['lineage']['positive_pre_damage_pairs_checked']=3899
        record['catalog_sha256']=digest([asdict(StraightSampleRef(r['pair_id'],r['source_base_key'],r['actual_matcher_input_sha256'],
            r['sample_path'],r['sample_sha256'],r['label'])) for r in record['entries']])
        self.assertEqual(len(self.check(record,files)[1]),5999)


if __name__ == '__main__':unittest.main()
