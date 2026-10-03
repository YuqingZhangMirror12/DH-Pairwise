import copy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from . import protocol as p


def row(role, stage, generator, label, index=0):
    token = f'{role}:{stage}:{generator}:{label}:{index}'
    return dict(pair_id=token, sample_sha256=hashlib.sha256(token.encode()).hexdigest(),
        stage=stage, generator=generator, label=label,
        parent_ids=[token + ':parent'], base_pair_ids=[token + ':base'], fragment_ids=[token + ':fragment'],
        donor_parent_ids=[], donor_base_pair_ids=[], donor_fragment_ids=[])


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifests = {}
        for role in p.ROLES:
            rows = [row(role, s, g, label) for s in p.STAGE_WEIGHTS for g in p.DEFAULT_GENERATORS[s] for label in (False, True)]
            self.manifests[role] = dict(schema=p.MANIFEST_SCHEMA, role=role, entries=rows)

    def bindings(self):
        result = {}
        for role, manifest in self.manifests.items():
            path = self.root / (role + '.json')
            path.write_text(json.dumps(manifest, sort_keys=True))
            result[role] = p.bind_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())
        return result

    def plan(self, updates=None):
        return p.freeze_protocol(self.bindings(), [100, 200, 300] if updates is None else updates)


class ProtocolTests(Fixture):
    def test_freezes_every_manifest_policy_and_candidate(self):
        plan = self.plan()
        self.assertEqual(p.validate_protocol(plan, verify_files=True), plan)
        self.assertEqual(plan['rule']['id'], p.RULE_ID)
        self.assertEqual(plan['rule']['selection_kind'], 'mixed_sim_v2_best')
        self.assertEqual(plan['same_fold_base_pair_reuse']['select']['base_pairs_reused'], 0)
        self.assertEqual(plan['same_fold_base_pair_reuse']['select']['strata']['v18']['Gen5']['positives'], 1)
        self.assertEqual(plan['same_fold_base_pair_reuse']['select']['primary_parent_count'], 26)

    def test_file_hash_mismatch(self):
        bindings = self.bindings(); path = Path(bindings['select']['path'])
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
            p.freeze_protocol(bindings, [100])

    def test_parent_base_fragment_and_donor_exclusions(self):
        for protected in ('train', 'test', 'cal'):
            for identity in p.IDENTITIES:
                for donor in (False, True):
                    with self.subTest(protected=protected, identity=identity, donor=donor):
                        original = copy.deepcopy(self.manifests['select']['entries'][0])
                        key = ('donor_' if donor else '') + identity
                        self.manifests['select']['entries'][0][key] = self.manifests[protected]['entries'][0][identity][:]
                        with self.assertRaisesRegex(ValueError, 'leakage'):
                            self.plan()
                        self.manifests['select']['entries'][0] = original

    def test_donor_in_exclusion_catalog_is_protected(self):
        self.manifests['train']['entries'][0]['donor_parent_ids'] = self.manifests['select']['entries'][0]['parent_ids'][:]
        with self.assertRaisesRegex(ValueError, 'donor parent_ids leakage'):
            self.plan()

    def test_cross_role_sample_hash_overlap_rejected_despite_unique_identities(self):
        for left, right in itertools.combinations(p.ROLES, 2):
            if {left, right} == {'train', 'test'}:
                continue
            with self.subTest(left=left, right=right):
                item = self.manifests[right]['entries'][0]
                original = item['sample_sha256']
                item['sample_sha256'] = self.manifests[left]['entries'][0]['sample_sha256']
                with self.assertRaisesRegex(ValueError, 'sample_sha256 leakage'):
                    self.plan()
                item['sample_sha256'] = original

    def test_old_train_test_sample_overlap_is_not_recertified(self):
        self.manifests['test']['entries'][0]['sample_sha256'] = self.manifests['train']['entries'][0]['sample_sha256']
        p.validate_protocol(self.plan())

    def test_same_fold_augmentation_reuse_allowed_and_disclosed(self):
        source = self.manifests['select']['entries'][0]
        augmented = copy.deepcopy(source); augmented['pair_id'] += ':augmentation'
        augmented['sample_sha256'] = 'a' * 64
        self.manifests['select']['entries'].append(augmented)
        reuse = self.plan()['same_fold_base_pair_reuse']['select']
        self.assertEqual(reuse['base_pairs_reused'], 1)
        self.assertEqual(reuse['extra_base_pair_appearances'], 1)
        self.assertEqual(reuse['maximum_base_pair_multiplicity'], 2)

    def test_duplicate_pair_rejected(self):
        self.manifests['select']['entries'].append(copy.deepcopy(self.manifests['select']['entries'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.plan()

    def test_missing_stage_generator_or_class_rejected(self):
        original = copy.deepcopy(self.manifests['select']['entries'])
        for predicate in (lambda r: r['stage'] == 'v18', lambda r: r['generator'] == 'Gen5',
                          lambda r: r['label']):
            self.manifests['select']['entries'] = [r for r in original if not predicate(r)]
            with self.assertRaises(ValueError):self.plan()
        self.manifests['select']['entries'] = original

    def test_explicit_generator_mapping_not_pair_name_inference(self):
        for role in ('cal', 'select'):
            for item in self.manifests[role]['entries']:
                item['pair_id'] = item['pair_id'].replace('Gen2', 'misleading-Gen5')
        self.plan()

    def test_role_swap_or_real_role_rejected(self):
        bindings = self.bindings(); bindings['select'], bindings['cal'] = bindings['cal'], bindings['select']
        with self.assertRaisesRegex(ValueError, 'phase/role'):
            p.freeze_protocol(bindings, [100])
        self.manifests['select']['role'] = 'real_select'
        with self.assertRaisesRegex(ValueError, 'real roles'):
            self.plan()

    def test_candidate_list_and_plan_mutation_rejected(self):
        for updates in ([], [0], [True], [200, 100], [100, 100]):
            with self.assertRaises(ValueError):self.plan(updates)
        plan = self.plan(); plan['candidate_updates'].append(400)
        with self.assertRaisesRegex(ValueError, 'digest differs'):p.validate_protocol(plan)

    def test_modified_policy_rejected_even_if_rehashed(self):
        plan = self.plan(); plan['rule']['stage_weights']['v18'] = .8
        plan['sha256'] = p.digest({k: v for k, v in plan.items() if k != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'changed selection rule'):p.validate_protocol(plan)

    def test_missing_donor_inventory_not_silently_empty(self):
        del self.manifests['select']['entries'][0]['donor_fragment_ids']
        with self.assertRaisesRegex(ValueError, 'donor identities'):self.plan()

    def test_generator_requirement_cannot_be_relaxed(self):
        generators = {stage: ['Gen2', 'Gen3'] for stage in p.STAGE_WEIGHTS}
        with self.assertRaisesRegex(ValueError, 'cannot be relaxed'):
            p.freeze_protocol(self.bindings(), [100], generators)

    def test_strict_straight_is_not_a_fictitious_gen5(self):
        plan = self.plan()
        self.assertEqual(plan['required_generators']['strict_straight'], ['straight_strip'])
        for item in self.manifests['select']['entries']:
            if item['stage'] == 'strict_straight':item['generator'] = 'Gen5'
        with self.assertRaisesRegex(ValueError, 'stratum'):self.plan()


if __name__ == '__main__':unittest.main()
