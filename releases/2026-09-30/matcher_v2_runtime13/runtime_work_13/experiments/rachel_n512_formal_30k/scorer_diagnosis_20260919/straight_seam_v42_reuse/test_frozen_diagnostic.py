import copy
import unittest
import numpy as np

from .frozen_diagnostic import EXPECTED, label_predictions, validate_population


def fixture():
    entries, records = [], []
    for (kind, label), n in EXPECTED.items():
        for i in range(n):
            pair = f'{kind}-{label}-{i}'
            base = 'strip' if kind == 'R' else 'rachel'
            entries.append(dict(pair_id=pair, recipe='straight_' + kind, label=label,
                sample_sha256='a' * 64, meta=dict(base=base)))
            records.append(dict(pair_id=pair, positive=label, kind=kind, base=base,
                sample_sha256='a' * 64, target_audit=dict(correspondence_count=8 if label else 0)))
    return (dict(split='select', entries=entries, failed=[]),
            dict(status='passed_integrity_and_supervision', rows=900, records=records,
                 source_manifest_sha256='b' * 64))


class FrozenDiagnosticTest(unittest.TestCase):
    def test_complete_population(self):
        manifest, audit = fixture()
        self.assertEqual(len(validate_population(manifest, audit, 'b' * 64)), 900)

    def test_test_split_forbidden(self):
        manifest, audit = fixture()
        manifest['split'] = 'test'
        with self.assertRaises(ValueError): validate_population(manifest, audit, 'b' * 64)

    def test_partial_population_forbidden(self):
        manifest, audit = fixture()
        manifest['entries'].pop()
        with self.assertRaises(ValueError): validate_population(manifest, audit, 'b' * 64)

    def test_hash_binding(self):
        manifest, audit = fixture()
        with self.assertRaises(ValueError): validate_population(manifest, audit, 'c' * 64)

    def test_changed_row_rejected(self):
        for field, value in [('sample_sha256', 'c' * 64), ('positive', False), ('kind', 'R'), ('base', 'other')]:
            manifest, audit = fixture()
            audit['records'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_population(manifest, audit, 'b' * 64)

    def test_duplicates_rejected(self):
        manifest, audit = fixture()
        audit['records'][1] = copy.deepcopy(audit['records'][0])
        with self.assertRaises(ValueError): validate_population(manifest, audit, 'b' * 64)

    def test_supervision_minimum(self):
        manifest, audit = fixture()
        audit['records'][0]['target_audit']['correspondence_count'] = 7
        with self.assertRaises(ValueError): validate_population(manifest, audit, 'b' * 64)

    def test_label_after_prediction(self):
        entries = [dict(pair_id='p', recipe='straight_J', label=True, meta=dict(base='torn_rachel')),
                   dict(pair_id='n', recipe='straight_J', label=False, meta=dict(base='torn_rachel'))]
        predictions = [dict(pair_id='p', winner=0, candidates=[dict(translation=[0, 0])]),
                       dict(pair_id='n', winner=-1, candidates=[])]
        snapshot = copy.deepcopy(predictions)
        rows = label_predictions(entries, predictions, {'p':np.array([10., 10.]), 'n':None})
        self.assertTrue(rows[0]['layout20'])
        self.assertTrue(rows[0]['coverage'])
        self.assertIsNone(rows[1]['layout20'])
        self.assertIsNone(rows[1]['coverage'])
        self.assertEqual(snapshot, predictions)
        with self.assertRaises(ValueError): label_predictions(entries, list(reversed(predictions)), {})

    def test_bad_winner_rejected(self):
        entries = [dict(pair_id='p', recipe='straight_J', label=True, meta=dict(base='torn_rachel'))]
        with self.assertRaises(ValueError):
            label_predictions(entries, [dict(pair_id='p', winner=1, candidates=[])], {'p':np.zeros(2)})


if __name__ == '__main__':unittest.main()
