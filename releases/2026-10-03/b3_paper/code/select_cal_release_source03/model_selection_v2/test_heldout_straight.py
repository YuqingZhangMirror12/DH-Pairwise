"""New wrapper CPU tests; fixtures never use original datasets or GPU."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from . import heldout_straight as h


class HeldoutStraightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = self.root / 'sources.json'
        self.parents = self.root / 'parents.json'
        rows = []
        folds = {}
        for role in ('cal', 'select'):
            folds[role] = []
            for i in range(2):
                family = role + str(i)
                f = self.root / (family + '.png')
                f.write_bytes(family.encode())
                rows.append(dict(role=role, source_family=family, path=str(f), file_sha256=h.file_sha(f),
                                 fragment_token='synthetic/' + family))
                folds[role].append(dict(role=role, family=family, edge_donors_must_be_from_same_role_families=True))
        self.write(self.index, dict(entries=rows))
        self.real = self.root / 'real_exclusion.json'
        self.write(self.real, dict(status='passed_source_exclusion', source_index_sha256=h.file_sha(self.index),
            checked_sources={r['fragment_token']: r['file_sha256'] for r in rows},
            potential_turufan_parent_aliases=[], exact_prepared_mask_matches=[], real_scores_read=False))
        self.write(self.parents, dict(schema='task1-evaluation-only-parent-plan/1', folds=folds,
            exclusion_families={key: ['blocked'] for key in h.EXCLUSION_KEYS}, hard_exclusions={},
            evidence={'existing_mask_source_index': {'path': str(self.index), 'sha256': h.file_sha(self.index)}}))
        # Freeze real source identities but inject only the materialization
        # backend in tests. This does not execute any reference generator.
        base = Path(__file__).resolve().parents[4] / 'deliverables/Simulation_data_handoff_20261001/code'
        self.runtime = base / 'frozen_remote/matcher_v2_20260930/v42_strict_runtime_01'
        self.reference = base / 'frozen_remote/matcher_v2_20260930/reference_v42_frozen_01'
        self.preprocess = base / 'dependency_runtime_work_13/staging/pairwise_v0_2/pairwise_data/rachel_preprocess.py'
        self.metrics = base / 'frozen_remote/matcher_v2_20260930/v42_metrics_frozen_01.py'
        self.env = patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''})
        self.env.start(); self.addCleanup(self.env.stop)

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data, sort_keys=True))

    def plan(self, role='cal', **changes):
        args = dict(parent_plan_path=self.parents, expected_parent_sha=h.file_sha(self.parents),
                    source_index_path=self.index, expected_index_sha=h.file_sha(self.index),
                    role=role, dataset_id='new-heldout', masterseeds={'cal': 202610021, 'select': 202610022},
                    output_new=self.root / 'new-heldout' / role / 'strict', frozen_runtime_dir=self.runtime,
                    reference_dir=self.reference, preprocess_path=self.preprocess, metrics_path=self.metrics,
                    real_exclusion_path=self.real, expected_real_exclusion_sha=h.file_sha(self.real))
        args.update(changes)
        return h.freeze_plan(**args)

    def test_exact_quota_role_and_reproducible_jobs(self):
        p = self.plan()
        jobs = h.jobs(p)
        self.assertEqual(320, len(jobs))
        self.assertEqual(jobs, h.jobs(p))
        self.assertEqual({('M', True): 40, ('M', False): 40, ('J', True): 75, ('J', False): 75,
                          ('R', True): 45, ('R', False): 45}, dict(h.Counter((j[0], bool(j[1])) for j in jobs)))
        self.assertTrue(all(j[3] == 'cal' for j in jobs))
        self.assertNotEqual(jobs[0][-1], h.jobs(self.plan('select'))[0][-1])

    def test_forbid_train_test_and_legacy_or_same_seed(self):
        for role in ('train', 'test', 'real'):
            with self.subTest(role=role), self.assertRaises(ValueError): self.plan(role)
        for seeds in ({'cal': 26093085, 'select': 2}, {'cal': 7, 'select': 7}, {'cal': True, 'select': 9}):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError): self.plan(masterseeds=seeds)

    def test_block_train_or_test_family_and_role_overlap(self):
        original = json.loads(self.parents.read_text())
        for mutate in (lambda p: p['exclusion_families'][h.EXCLUSION_KEYS[0]].append('cal0'),
                       lambda p: p['exclusion_families'][h.EXCLUSION_KEYS[1]].append('cal0'),
                       lambda p: p['folds']['select'][0].update(family='cal0')):
            p = copy.deepcopy(original); mutate(p); self.write(self.parents, p)
            with self.assertRaises(ValueError): self.plan()

    def test_reject_foreign_donor_changed_bytes_and_duplicate_token(self):
        p = self.plan()
        Path(p['source_rows'][0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bound file changed'): h.validate_plan(p)
        for bad in ({'path': '/outside', 'source_family': 'select0'}, p['source_rows'][0] | {'role': 'select'}):
            with self.assertRaisesRegex(ValueError, 'donor outside'):
                h.validate_entry(p, ('M', 1, 0, 'cal', p['output_new'], 8),
                                 {'failed': True, 'attempted_donor_references': [bad]})

    def test_source_index_receipt_is_bound_in_parent_admission(self):
        data = json.loads(self.index.read_text()); data['entries'][0]['fragment_token'] = 'alias'
        self.write(self.index, data)
        with self.assertRaisesRegex(ValueError, 'exact receipt'): self.plan()

    def test_real_exclusion_must_cover_cal_and_reject_aliases(self):
        value = json.loads(self.real.read_text())
        del value['checked_sources']['synthetic/cal0']; self.write(self.real, value)
        with self.assertRaisesRegex(ValueError, 'every source'): self.plan()
        value['checked_sources']['synthetic/cal0'] = h.file_sha(self.root / 'cal0.png')
        value['potential_turufan_parent_aliases'] = ['cal0']; self.write(self.real, value)
        with self.assertRaisesRegex(ValueError, 'real alias'): self.plan()

    def test_pilot_uses_separate_namespace_and_does_not_claim_320(self):
        p = self.plan(dataset_id='new-pilot', output_new=self.root / 'new-pilot' / 'cal' / 'strict')
        result = h.materialize(p, backend_factory=FakeBackend, pilot=True)
        self.assertEqual((6, 6, True), (result['generated'], result['expected'], result['pilot']))

    def test_require_explicit_exclusions_and_no_crossrole_sample_bytes(self):
        parents = json.loads(self.parents.read_text())
        parents.pop('exclusion_families'); self.write(self.parents, parents)
        with self.assertRaisesRegex(ValueError, 'explicit TRAIN/TEST'): self.plan()

    def test_duplicate_fragment_and_crossrole_source_hash_fail_closed(self):
        parents = json.loads(self.parents.read_text())
        original = json.loads(self.index.read_text())
        same_token = copy.deepcopy(original)
        same_token['entries'][1]['fragment_token'] = same_token['entries'][0]['fragment_token']
        with self.assertRaisesRegex(ValueError, 'duplicate source'):
            h._sources(parents, same_token, 'cal', False)
        same_bytes = copy.deepcopy(original)
        same_bytes['entries'][2]['file_sha256'] = same_bytes['entries'][0]['file_sha256']
        with self.assertRaisesRegex(ValueError, 'identical source bytes cross'):
            h._sources(parents, same_bytes, 'cal', False)

    def test_private_backend_loads_without_global_mutation_or_generation(self):
        import numpy  # Keep dependencies loaded before patch.dict restores sys.modules.
        import scipy
        import cv2
        p = self.plan()
        sentinel = types.ModuleType('gen_straight_seam_v3')
        sentinel.SPLIT_SEED = {'sentinel': 99}
        with patch.dict(sys.modules, {'gen_straight_seam_v3': sentinel}):
            before = sys.path[:]
            backend = h.FrozenBackend(p)
            self.assertIs(sys.modules['gen_straight_seam_v3'], sentinel)
            self.assertEqual({'sentinel': 99}, sentinel.SPLIT_SEED)
            self.assertEqual(before, sys.path)
            self.assertEqual(4, backend.reference.SPLIT_SEED['cal'])
            self.assertEqual(2, backend.reference.SPLIT_SEED['select'])
            self.assertEqual(2, len(backend.reference.SRCS))
            bases = [backend.reference.base_of('J', p['masterseeds']['cal'], 'cal', 1, i) for i in range(75)]
            self.assertEqual({'torn_rachel': 60, 'margin_fragment': 9, 'torn_strip': 6}, dict(h.Counter(bases)))
            self.assertFalse(Path(p['output_new']).exists())

    def test_private_backends_do_not_share_cal_domain_dict(self):
        first, second = h.FrozenBackend(self.plan()), h.FrozenBackend(self.plan('select'))
        self.assertIsNot(first.reference.SPLIT_SEED, second.reference.SPLIT_SEED)
        first.reference.SPLIT_SEED['cal'] = 90
        self.assertEqual(4, second.reference.SPLIT_SEED['cal'])

    def test_materialize_fixture_never_claims_independent_audit_or_production(self):
        p = self.plan()
        receipt = h.materialize(p, backend_factory=FakeBackend)
        self.assertEqual('materialized_pending_independent_audit', receipt['status'])
        self.assertEqual(320, receipt['generated'])
        self.assertFalse(receipt['production_backend'])
        with self.assertRaisesRegex(ValueError, 'production successful'):
            h.audit_materialization(p, load_sample=lambda p: None,
                                    target_builder_path=self.root / 'not-read.py', expected_target_builder_sha='0'*64)
        with self.assertRaisesRegex(ValueError, 'new output'): h.materialize(p, backend_factory=FakeBackend)

    def test_strict_positive_geometry_and_target_evidence_required(self):
        p = self.plan(); out = Path(p['output_new'])
        (out / 'samples').mkdir(parents=True); (out / 'proof').mkdir()
        job = h.jobs(p)[0]
        entry = FakeBackend(p).generate_one(job)
        entry['geometry_attempts'] = [dict(passed=False)]
        with self.assertRaisesRegex(ValueError, 'strict geometry'): h.validate_entry(p, job, entry)
        entry['geometry_attempts'] = [dict(passed=True)]; entry['target_audit'] = None
        with self.assertRaisesRegex(ValueError, 'target/damage'): h.validate_entry(p, job, entry)

    def test_failure_not_retried_or_called_complete(self):
        p = self.plan()
        class Failing(FakeBackend):
            def generate_one(self, job):
                if job[0] == 'M' and job[1] == 1 and job[2] == 0:
                    return dict(failed=True, attempted_donor_references=[], last_reason='few_healthy_corr')
                return super().generate_one(job)
        receipt = h.materialize(p, backend_factory=Failing)
        self.assertEqual(('incomplete', 319, 1), (receipt['status'], receipt['generated'], receipt['failed']))

    def test_no_cuda_no_output_overwrite_no_quota_mutation(self):
        p = self.plan(); p['counts_per_label']['J'] = 74
        with self.assertRaisesRegex(ValueError, 'quota'): h.validate_plan(p)
        p = self.plan()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), self.assertRaisesRegex(ValueError, 'CPU-only'):
            h.materialize(p, backend_factory=FakeBackend)
        with self.assertRaisesRegex(ValueError, 'namespace'): self.plan(output_new=self.root / 'cal' / 'bad')


class FakeBackend:
    def __init__(self, plan): self.plan = plan

    def generate_one(self, job):
        kind, label, i, role, root, seed = job
        name = f'{kind}_{label}_{i}'
        path = Path(root) / 'samples' / (name + '.npz')
        path.write_bytes(name.encode())
        result = dict(id=name, pair_id=role + name, split=role, label=bool(label), recipe='straight_' + kind,
                      sample_path=str(path), sample_sha256=h.file_sha(path), training_admitted=False,
                      wrapper_revision='codex-v42-geometry-explicit-targets/1', attempted_donor_references=[],
                      piece_attempts=[], finalization_attempts=[], meta={})
        if label:
            proof = Path(root) / 'proof' / (name + '.npz'); proof.write_bytes(b'proof')
            result.update(proof_path=str(proof), proof_sha256=h.file_sha(proof), target_audit={}, accepted_damage_trace={})
            result.update(geometry_attempts=[dict(passed=True)], pre_damage_pair_sha256=h.digest(name))
        return result


if __name__ == '__main__':
    unittest.main()
