"""Synthetic on-disk receipts only; none grants real user authorization."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .admission import COUNTS, REVIEW_GROUPS, REVISION, sha, validate_approval, validate_data


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-exp3-contract-')
        self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        ids = ['review-' + str(i) for i in range(20)]
        text='SYNTHETIC UNIT TEST ONLY'
        self.approval = dict(schema='aggressive-data-human-approval/2', status='approved',
            scope='full30k_after_full_per_type10_review', evidence_kind='explicit_user_message_in_current_thread',
            user_approval_text=text,user_approval_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            fallback_authorization_text=text,fallback_authorization_sha256=hashlib.sha256(text.encode()).hexdigest(),
            authorized_delta=dict(after=[5,35],retain_original_v14_on_failed_new_augmentation=True),
            approved_at_iso='synthetic fixture', augmentation_revision=REVISION,
            review_generation=self.put('review_generation', dict(status='generated_pending_pixel_audit_and_review',
                pairs=20, missing={})),
            review_pixel_audit=self.put('review_audit', dict(status='passed', pairs=20, errors=[],
                receipts=[dict(id=i, status='passed') for i in ids],
                positive_size_counts=dict(smaller=7, larger=3))),
            review_rendered=self.put('rendered', dict(rows=[dict(id=i) for i in ids],
                groups={k: dict(ids=ids[:10]) for k in REVIEW_GROUPS})))
        self.approval_path = self.root / 'approval.json'; self.write(self.approval_path, self.approval)
        specs = {}
        for name, count in COUNTS.items():
            specs[name] = self.put(name, dict(split=name, augmentation_revision=REVISION,
                entries=[dict(pair_id=name + '-' + str(i), label=int(i % 2 == 0)) for i in range(count)]))
        self.contract = dict(status='passed', source_disjoint=True, augmentation_revision=REVISION,
            review_approval_sha256=sha(self.approval_path), online_mirror_probability=0,
            train=dict(**specs['train'], archive_manifest=specs['train']['path'],
                       archive_manifest_sha256=specs['train']['sha256']),
            validation=dict(cal_mixed=specs['cal'], select_mixed=specs['select']), test=dict(mixed=specs['test']),
            source_families={**{k:[k + '-manuscript'] for k in COUNTS},
                             'historical_s7_train':['train-manuscript'], 'donor_train':['train-manuscript']})
        self.contract['aggressive_full_audit'] = self.put('full_audit', dict(
            schema='aggressive-v17-full-audit/1', status='passed', checked_pairs=30000, failures=0,
            manifest_sha256={k:v['sha256'] for k,v in specs.items()},
            review_approval_sha256=sha(self.approval_path), source_disjoint=True,
            endpoint_and_area_pixel_audit=True, target_and_donor_audit=True,fallback_numerical_identity_audit=True))
        self.contract_path = self.root / 'contract.json'; self.write(self.contract_path, self.contract)
        self.calibration = dict(schema='s7-consensus-train-geometry/2', status='complete', pairs=12000,
            inherited_edges=123456, contract_sha256=sha(self.contract_path),
            manifest_sha256=specs['train']['sha256'])
        self.calibration_path = self.root / 'calibration.json'; self.write(self.calibration_path, self.calibration)

    def write(self, path, data):
        path.write_text(json.dumps(data, sort_keys=True) + '\n')

    def put(self, name, value):
        path = self.root / (name + '.json'); self.write(path, value)
        return dict(path=str(path), sha256=sha(path))

    def check(self):
        return validate_data(self.contract_path, self.calibration_path, self.approval_path)

    def altered_contract(self, edit):
        edit(self.contract); self.write(self.contract_path, self.contract)
        self.calibration['contract_sha256'] = sha(self.contract_path)
        self.write(self.calibration_path, self.calibration)

    def test_complete_synthetic_contract_and_counts(self):
        receipt = self.check()
        self.assertEqual(set(receipt['manifest_sha256']), set(COUNTS))
        self.assertTrue(receipt['geometry_from_new_train_only'])
        self.assertFalse(receipt['test_used_for_training_or_selection'])

    def test_fallback_authorization_text_is_bound(self):
        self.approval['fallback_authorization_text']='changed'
        self.write(self.approval_path,self.approval)
        with self.assertRaisesRegex(ValueError,'fallback'):validate_approval(self.approval_path)

    def test_ten_pair_probe_not_full_review(self):
        self.approval['review_generation'] = self.put('probe', dict(status='probe_complete', pairs=10, missing={}))
        self.write(self.approval_path, self.approval)
        with self.assertRaisesRegex(ValueError, 'probe'):
            validate_approval(self.approval_path)

    def test_preview_acceptance_not_full_generation_authorization(self):
        self.approval['scope'] = 'ten_preview_pairs_only'
        self.write(self.approval_path, self.approval)
        with self.assertRaisesRegex(ValueError, 'explicit approval'):
            validate_approval(self.approval_path)

    def test_missing_group_or_short_group_rejected(self):
        for mode in ('missing', 'short'):
            value = json.loads(Path(self.approval['review_rendered']['path']).read_text())
            if mode == 'missing': value['groups'].pop('gaps')
            else: value['groups']['clean']['ids'] = value['groups']['clean']['ids'][:9]
            approval = copy.deepcopy(self.approval)
            approval['review_rendered'] = self.put('bad_' + mode, value)
            self.write(self.root / 'bad_approval.json', approval)
            with self.assertRaises(ValueError): validate_approval(self.root / 'bad_approval.json')

    def test_stale_audit_hash_rejected(self):
        Path(self.approval['review_pixel_audit']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'): validate_approval(self.approval_path)

    def test_failed_review_or_wrong_side_quota_rejected(self):
        original = json.loads(Path(self.approval['review_pixel_audit']['path']).read_text())
        for mode in ('failure', 'quota'):
            data = copy.deepcopy(original)
            if mode == 'failure': data['receipts'][0]['status'] = 'failed'
            else: data['positive_size_counts'] = dict(smaller=6, larger=4)
            approval = copy.deepcopy(self.approval); approval['review_pixel_audit'] = self.put(mode, data)
            self.write(self.root / 'bad.json', approval)
            with self.assertRaises(ValueError): validate_approval(self.root / 'bad.json')

    def test_v14_cannot_be_relabelled_as_experiment3(self):
        self.altered_contract(lambda c: c.pop('augmentation_revision'))
        with self.assertRaisesRegex(ValueError, 'not v14'): self.check()

    def test_cross_split_and_historical_source_leakage_rejected(self):
        original = copy.deepcopy(self.contract)
        for key in ('train', 'historical_s7_train', 'donor_train'):
            self.contract = copy.deepcopy(original)
            self.altered_contract(lambda c: c['source_families'][key].append('cal-manuscript'))
            with self.assertRaisesRegex(ValueError, 'leakage'): self.check()

    def test_manifest_counts_and_balance_checked_not_only_claims(self):
        data = json.loads(Path(self.contract['train']['path']).read_text())
        data['entries'][0]['label'] = 0
        ref = self.put('unbalanced_train', data)
        self.altered_contract(lambda c: c['train'].update(ref))
        with self.assertRaisesRegex(ValueError, 'equal positive'): self.check()

    def test_manifest_identity_checked_before_training(self):
        Path(self.contract['validation']['cal_mixed']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'): self.check()

    def test_old_geometry_contract_or_partial_train_not_admitted(self):
        for field, value in (('contract_sha256', 'old-v14'), ('pairs', 100), ('manifest_sha256', 'old-train')):
            data = dict(self.calibration, **{field:value}); self.write(self.calibration_path, data)
            with self.assertRaisesRegex(ValueError, 'recompute geometry'): self.check()

    def test_incomplete_full_audit_rejected(self):
        data = json.loads(Path(self.contract['aggressive_full_audit']['path']).read_text())
        data['checked_pairs'] = 24000
        self.altered_contract(lambda c: c.update(aggressive_full_audit=self.put('short_audit', data)))
        with self.assertRaisesRegex(ValueError, 'all 30K'): self.check()

    def test_no_extra_online_augmentation(self):
        self.altered_contract(lambda c: c.update(online_mirror_probability=.15))
        with self.assertRaisesRegex(ValueError, 'online'): self.check()


if __name__ == '__main__': unittest.main()
