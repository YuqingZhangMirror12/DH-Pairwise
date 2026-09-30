"""Synthetic fixtures only: no checkpoint, corpus, network, or CUDA access."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from consensus_binary_eval_adapter.snapshot import snapshot_prediction, write_snapshot, audit_snapshot
from consensus_binary_eval_adapter.trace import MLPTrace
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.binary_scorer_v1.test_binary import fixture
from . import export


def save(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + '\n')


def capture(variant='patch', empty=False, duplicate=False, provenance=None):
    torch.manual_seed(260928)
    model, pair, proposals, _ = fixture(variant)
    model.eval().requires_grad_(False)
    if empty:
        proposals = replace(proposals, clusters=())
    if duplicate:
        proposals = replace(proposals, clusters=tuple(
            replace(c, edge_ids=c.edge_ids.repeat(3, 1)) for c in proposals.clusters))
    with torch.no_grad(), MLPTrace(model.head) as trace:
        pred = model.score_pair(pair, proposals=proposals, threshold=.3)
    return snapshot_prediction('synthetic-case', pair, pred, threshold=.3,
                               provenance=provenance or {'variant': 'binary_' + variant}, trace=trace)


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = {v: capture(v) for v in ('patch', 'stats')}
        cls.empty = capture(empty=True)
        cls.duplicated = capture(duplicate=True)

    def case(self, root, example=None):
        meta, arrays = deepcopy(example or self.examples['patch'])
        root = Path(root) / 'case'
        write_snapshot(root, meta, arrays)
        audited = audit_snapshot(root / 'evidence.json')
        self.assertEqual(audited['status'], 'passed')
        save(root / 'audit.json', audited)
        verified = dict(pair_id=meta['pair_id'], evidence_sha256=export.sha(root / 'evidence.json'),
                        sidecar_sha256=export.sha(root / 'arrays.npz'))
        return root, verified

    def test_patch_exact_union_values_and_all_clusters_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.case(directory)
            value = export.export_case(root / 'evidence.json', root / 'audit.json', verified)
            self.assertEqual(len(value['clusters']), 2)
            self.assertFalse(value['semantics']['pooling_weights_are_attention'])
            self.assertFalse(value['semantics']['layer_values_are_causal_importance'])
            self.assertTrue(value['semantics']['feature_pooling_present'])
            with np.load(root / 'arrays.npz') as arrays:
                meta = export.read(root / 'evidence.json')
                for original, cluster in zip(meta['clusters'], value['clusters']):
                    np.testing.assert_array_equal([e['q'] for e in cluster['edges']],
                                                  arrays[original['inputs']['q']])
                    self.assertAlmostEqual(sum(e['conditional_pooling_weight'] for e in cluster['edges']), 1., places=6)
                    self.assertEqual(len(cluster['scalar_inputs']), 16)
                    for edge in cluster['edges']:
                        np.testing.assert_allclose(np.array(edge['b_rc']) - edge['a_placed_rc'], edge['residual_rc'])
                    self.assertIsNotNone(cluster['patch_context'])
                    self.assertEqual(len(cluster['layers']), 9)

    def test_stats_has_no_patch_features_or_feature_pooling(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.case(directory, self.examples['stats'])
            value = export.export_case(root / 'evidence.json', root / 'audit.json', verified)
            self.assertFalse(value['semantics']['feature_pooling_present'])
            self.assertEqual(value['semantics']['pooling_weight_role'], 'weighted_geometry_only')
            for cluster in value['clusters']:
                self.assertIsNone(cluster['patch_context'])
                self.assertIsNone(cluster['pooled_features'])
                self.assertEqual(len(cluster['layers']), 5)

    def test_empty_case_does_not_invent_pose_or_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.case(directory, self.empty)
            value = export.export_case(root / 'evidence.json', root / 'audit.json', verified)
            self.assertEqual(value['clusters'], [])
            self.assertEqual(value['selected_cluster_id'], -1)
            self.assertIsNone(value['translation_a_to_b_rc'])
            self.assertFalse(value['accepted'])

    def test_repeated_seed_edges_do_not_repeat_display_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.case(directory, self.duplicated)
            value = export.export_case(root / 'evidence.json', root / 'audit.json', verified)
            for cluster in value['clusters']:
                ids = [(e['a'], e['b']) for e in cluster['edges']]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertAlmostEqual(sum(e['q'] for e in cluster['edges']), cluster['absolute_q_sum'], places=5)

    def test_read_only_export_preserves_all_evidence_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.case(directory)
            before = {p.name: export.sha(p) for p in root.iterdir()}
            export.export_case(root / 'evidence.json', root / 'audit.json', verified)
            self.assertEqual(before, {p.name: export.sha(p) for p in root.iterdir()})

    def test_changed_evidence_rejected_even_if_sidecar_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.case(directory)
            meta = export.read(root / 'evidence.json'); meta['score'] += .1
            save(root / 'evidence.json', meta)
            with self.assertRaisesRegex(ValueError, 'queue-verified'):
                export.export_case(root / 'evidence.json', root / 'audit.json', verified)

    def test_changed_sidecar_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.case(directory)
            with (root / 'arrays.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'sidecar differs'):
                export.export_case(root / 'evidence.json', root / 'audit.json', verified)

    def test_nonpassing_or_incomplete_numeric_audit_rejected(self):
        for change in ({'status': 'failed'}, {'errors': ['wrong Q']},
                       {'raw_union_q_pooling_and_mlp_replayed': False}, {'clusters': 3}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root, verified = self.case(directory)
                record = export.read(root / 'audit.json'); record.update(change)
                save(root / 'audit.json', record)
                with self.assertRaisesRegex(ValueError, 'numeric audit'):
                    export.export_case(root / 'evidence.json', root / 'audit.json', verified)

    def test_no_complex_attention_semantics_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.case(directory)
            meta = export.read(root / 'evidence.json'); meta['semantics']['attention_present'] = True
            save(root / 'evidence.json', meta); verified['evidence_sha256'] = export.sha(root / 'evidence.json')
            with self.assertRaisesRegex(ValueError, 'complex-head'):
                export.export_case(root / 'evidence.json', root / 'audit.json', verified)

    def test_artifact_path_cannot_escape_case_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.case(directory)
            meta = export.read(root / 'evidence.json'); meta['sidecar']['path'] = '../outside.npz'
            save(root / 'evidence.json', meta); verified['evidence_sha256'] = export.sha(root / 'evidence.json')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                export.export_case(root / 'evidence.json', root / 'audit.json', verified)

    def test_resealed_array_manifest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.case(directory)
            meta = export.read(root / 'evidence.json')
            meta['arrays'][meta['pair']['q']]['shape'] = [1]
            save(root / 'evidence.json', meta); verified['evidence_sha256'] = export.sha(root / 'evidence.json')
            with self.assertRaisesRegex(ValueError, 'array content'):
                export.export_case(root / 'evidence.json', root / 'audit.json', verified)

    def job(self, root, split='sim_test_v14', choice='sim', experiment='binary_patch'):
        root = Path(root); root.mkdir(exist_ok=True)
        n = export.EXPECTED_PAIRS[split]
        identity = dict(variant='binary_patch', selection_kind=choice, split=split, total_pairs=n,
                        selected_epoch=8, checkpoint_sha256='synthetic-not-trained', threshold=.3,
                        experiment_variant=experiment)
        (root / 'pair_predictions.jsonl').write_text(''.join(
            json.dumps(dict(pair_id='synthetic-' + str(i), score=.4)) + '\n' for i in range(n)))
        save(root / 'status.json', dict(status='complete', pairs=n))
        save(root / 'protocol.json', dict(status='complete', **identity))
        save(root / 'prediction_complete.json', dict(status='all_predictions_frozen', pairs=n,
            sha256=export.sha(root / 'pair_predictions.jsonl'), model_state_unchanged=True, **identity))
        cases, verified_cases = [], []
        if split == 'turufan':
            example = deepcopy(self.examples['patch']); example[0]['provenance'] = identity
            case_root, verified = self.case(root, example)
            cases.append(dict(pair_id=verified['pair_id'], evidence='case/evidence.json',
                              numerical_audit='case/audit.json'))
            verified_cases.append(verified)
        save(root / 'diagnostic_index.json', dict(cases=cases, selected_by_new_results=False))
        real = not split.startswith('sim_test_')
        primary = dict(pairs=n, f1=.5, joint_f1=None, layout20=None, candidate_coverage=None)
        summary = dict(status='complete', threshold_refitting=False, layout_gt_available=split != 'turufan',
            main_group='real_test' if real else 'all', real_test_is_historically_unseen=False,
            groups={'real_test' if real else 'all': dict(primary=primary, fixed03=deepcopy(primary))},
            diagnostic_cases=cases, **identity)
        save(root / 'summary.json', summary)
        verification = dict(status='passed', variant='patch', selection_kind=choice, split=split, pairs=n,
            selected_epoch=8, checkpoint_sha256=identity['checkpoint_sha256'],
            model_state_unchanged=True, real_inference_performed=False, fixed_cases=verified_cases,
            summary_sha256=export.sha(root / 'summary.json'),
            predictions_sha256=export.sha(root / 'pair_predictions.jsonl'))
        save(root / 'verification.json', verification)
        return root, root / 'verification.json'

    def reseal_summary(self, root, change):
        summary = export.read(root / 'summary.json'); change(summary)
        save(root / 'summary.json', summary)
        verified = export.read(root / 'verification.json'); verified['summary_sha256'] = export.sha(root / 'summary.json')
        save(root / 'verification.json', verified)

    def test_complete_sim_job_is_not_whole_experiment_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.job(directory)
            result = export.export_job(root, verified)
            self.assertEqual(result['status'], 'verified_single_evaluation_job')
            self.assertFalse(result['complete_experiment_claimed'])
            self.assertFalse(result['threshold_refitted'])
            self.assertEqual(result['cases'], [])

    def test_real_selection_role_and_unknown_turufan_layout_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.job(directory, 'turufan', 'real')
            result = export.export_job(root, verified)
            self.assertEqual(result['selection_kind'], 'real')
            self.assertEqual(result['main_group'], 'real_test')
            self.assertIsNone(result['groups']['real_test']['primary']['joint_f1'])
            self.assertEqual(len(result['cases']), 1)

    def test_aggressive_dataset_identity_is_not_v14(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.job(directory, 'sim_test_aggressive', experiment='aggressive_binary_patch')
            result = export.export_job(root, verified)
            self.assertEqual(result['experiment'], 'aggressive_binary_patch')
            self.assertEqual(result['split'], 'sim_test_aggressive')

    def test_v14_and_aggressive_sim_test_cannot_be_swapped(self):
        for split, experiment in (('sim_test_aggressive', 'binary_patch'),
                                  ('sim_test_v14', 'aggressive_binary_patch')):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as directory:
                root, verified = self.job(directory, split, experiment=experiment)
                with self.assertRaisesRegex(ValueError, 'remain separate'):
                    export.export_job(root, verified)

    def test_failure_precedes_stale_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.job(directory); save(root / 'failure.json', {'status': 'failed'})
            with self.assertRaisesRegex(ValueError, 'failure overrides'):
                export.export_job(root, verified)

    def test_running_and_partial_jobs_cannot_publish_results(self):
        for change in ({'status': 'inference'}, {'pairs': 2999}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root, verified = self.job(directory)
                record = export.read(root / 'status.json'); record.update(change); save(root / 'status.json', record)
                with self.assertRaisesRegex(ValueError, 'not a complete'):
                    export.export_job(root, verified)

    def test_changed_summary_is_not_imported(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.job(directory)
            value = export.read(root / 'summary.json'); value['groups']['all']['primary']['f1'] = 1.
            save(root / 'summary.json', value)
            with self.assertRaisesRegex(ValueError, 'queue verification'):
                export.export_job(root, verified)

    def test_turufan_layout_cannot_be_invented_even_in_resealed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.job(directory, 'turufan')
            self.reseal_summary(root, lambda v: v['groups']['real_test']['primary'].update(joint_f1=.9))
            with self.assertRaisesRegex(ValueError, 'invented Turufan'):
                export.export_job(root, verified)

    def test_real_all_population_cannot_replace_heldout_main_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.job(directory, 'turufan')
            self.reseal_summary(root, lambda v: v.update(main_group='all_development_context'))
            with self.assertRaisesRegex(ValueError, 'source-held-out'):
                export.export_job(root, verified)

    def test_importing_exporter_does_not_import_torch_or_model(self):
        code = ('import importlib.util,sys; '
                's=importlib.util.spec_from_file_location("report_export",' + repr(export.__file__) + '); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                'assert "torch" not in sys.modules; assert not any("s7_consensus_v1" in k for k in sys.modules)')
        run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_cli_writes_separate_file_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root, verified = self.job(Path(directory) / 'input')
            out = Path(directory) / 'report.json'
            cmd = [sys.executable, export.__file__, '--job', str(root),
                   '--verification', str(verified), '--out', str(out)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            before = export.sha(out)
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(export.sha(out), before)
            cmd[-1] = str(root / 'new_report.json')
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertFalse((root / 'new_report.json').exists())


if __name__ == '__main__':
    unittest.main()
