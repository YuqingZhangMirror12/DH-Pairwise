"""Assembly -> existing exporter -> actual untrained CPU MLP evidence.

Training completion and population summaries are explicitly synthetic fixture
metadata. This tests the artifact interface, not training or real performance.
"""
from copy import deepcopy
from pathlib import Path
import unittest

from . import assemble as a
from . import export
from .test_assemble import fixture, save, reseal
from .test_export import capture
from consensus_binary_eval_adapter.snapshot import write_snapshot, audit_snapshot


def materialize(root, experiment, values):
    variant = a.EXPERIMENTS[experiment]['variant']
    example = capture(variant)
    gate = a.read(root / 'selected_models.json')
    terminal = a.read(root / 'evaluation_complete.json')
    for record in terminal['jobs']:
        task, choice, split = record['task'], record['selection_kind'], record['split']
        folder = root / task
        model = gate['selected'][choice]
        n = export.EXPECTED_PAIRS[split]
        identity = dict(model, split=split, total_pairs=n, threshold=model['thresholds'][split])
        save(folder / 'protocol.json', dict(status='complete', **identity))
        save(folder / 'status.json', dict(status='complete', pairs=n))
        with (folder / 'pair_predictions.jsonl').open('x') as stream:
            for i in range(n):
                import json
                stream.write(json.dumps(dict(pair_id='synthetic-pair-' + str(i), score=.4)) + '\n')
        prediction_sha = export.sha(folder / 'pair_predictions.jsonl')
        save(folder / 'prediction_complete.json', dict(status='all_predictions_frozen', pairs=n,
            sha256=prediction_sha, model_state_unchanged=True, **identity))
        cases, verified_cases = [], []
        for i in range(export.EXPECTED_CASES[split]):
            pair_id = 'synthetic-pair-' + str(i)
            meta, arrays = deepcopy(example)
            meta['pair_id'] = pair_id
            meta['provenance'] = dict(identity, synthetic_fixture=True)
            meta['threshold'] = identity['threshold']
            # The selected threshold changes only the deterministic final
            # acceptance comparison; the recorded MLP forward stays intact.
            meta['accepted'] = bool(meta['has_candidate'] and meta['numeric_valid']
                                    and meta['score'] >= meta['threshold'])
            destination = folder / ('case_' + str(i))
            write_snapshot(destination, meta, arrays)
            audit = audit_snapshot(destination / 'evidence.json')
            if audit['status'] != 'passed': raise ValueError(audit)
            save(destination / 'audit.json', audit)
            cases.append(dict(pair_id=pair_id, evidence=destination.name + '/evidence.json',
                              numerical_audit=destination.name + '/audit.json'))
            verified_cases.append(dict(pair_id=pair_id,
                evidence_sha256=export.sha(destination / 'evidence.json'),
                sidecar_sha256=export.sha(destination / 'arrays.npz')))
        save(folder / 'diagnostic_index.json', dict(cases=cases, selected_by_new_results=False))
        source = values[task]
        save(folder / 'summary.json', dict(status='complete', threshold_refitting=False,
            layout_gt_available=split != 'turufan', main_group=source['main_group'],
            real_test_is_historically_unseen=False, groups=source['groups'], diagnostic_cases=cases, **identity))
        proof = dict(status='passed', variant=variant, selection_kind=choice, split=split, pairs=n,
            selected_epoch=model['selected_epoch'], checkpoint_sha256=model['checkpoint_sha256'],
            model_state_unchanged=True, real_inference_performed=False, fixed_cases=verified_cases,
            summary_sha256=export.sha(folder / 'summary.json'), predictions_sha256=prediction_sha)
        record['verified'] = proof
        save(root / (task + '_verified.json'), proof)
    save(root / 'evaluation_complete.json', terminal)
    reseal(root, complete=True)


class IntegrationTests(unittest.TestCase):
    def check_experiment(self, experiment):
        with fixture(experiment) as (root, values):
            materialize(root, experiment, values)
            before = {str(p.relative_to(root)): export.sha(p) for p in root.rglob('*') if p.is_file()}
            result = a.comparison_bundle({experiment: root})
            imported = result['experiments'][0]
            self.assertEqual(sum(len(job['cases']) for job in imported['jobs']), 22)
            for job in imported['jobs']:
                for case in job['cases']:
                    self.assertFalse(case['semantics']['attention_present'])
                    self.assertEqual(case['verification']['imported_numeric_audit_status'], 'passed')
                    self.assertEqual(case['threshold'], job['threshold'])
            self.assertEqual(before, {str(p.relative_to(root)): export.sha(p) for p in root.rglob('*') if p.is_file()})
            self.assertFalse(result['all_three_frozen_evaluations_imported'])

    def test_patch_all_six_jobs_through_actual_exporter(self):
        self.check_experiment('binary_patch')

    def test_stats_all_six_jobs_through_actual_exporter(self):
        self.check_experiment('binary_stats')

    def test_aggressive_population_through_actual_exporter(self):
        self.check_experiment('aggressive_binary_patch')


if __name__ == '__main__':
    unittest.main()
