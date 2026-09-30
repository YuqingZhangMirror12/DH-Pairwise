import json
from pathlib import Path
import tempfile
import unittest

from .analyze import audit_sidecar, independently_count, policy_count, sha


def population():
    return [dict(pair_id=str(i), label=y, score=s, layout_good_20=g)
            for i, (y, s, g) in enumerate(((True, .8, True),
                                          (True, .7, False),
                                          (False, .6, None),
                                          (False, .1, None)))]


class MetricTests(unittest.TestCase):
    def test_wrong_pose_is_classification_tp_but_joint_fp_and_fn(self):
        out = independently_count(population(), [True, True, True, False], True)
        self.assertEqual((out['tp'], out['fp'], out['fn']), (2, 1, 0))
        self.assertEqual((out['joint_tp'], out['joint_fp'], out['joint_fn']), (1, 2, 1))
        self.assertAlmostEqual(out['joint_f1'], .4)

    def test_missing_gt_is_not_a_failed_layout(self):
        out = independently_count(population(), [True, True, True, False], False)
        self.assertIsNone(out['layout_accuracy'])
        self.assertIsNone(out['joint_f1'])
        self.assertEqual(out['tp'], 2)

    def test_exclusions_do_not_recalibrate_decisions(self):
        out = policy_count(population(), [False, True, False, False], True, {'1'})
        self.assertEqual(out['original']['tp'], 1)
        self.assertEqual(out['corrected']['tp'], 0)
        self.assertEqual(out['corrected']['layout_correct_rejected'], 1)


class SidecarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        arrays = self.root / 'arrays.npz'
        arrays.write_bytes(b'fixture bytes; array parsing is tested by the exporter')
        self.item = dict(pair_id='example', evidence='evidence.json', sidecar_sha256=sha(arrays))
        meta = dict(pair_id='example', sidecar=dict(path='arrays.npz', sha256=sha(arrays),
                                                  bytes=arrays.stat().st_size))
        (self.root / 'evidence.json').write_text(json.dumps(meta))

    def tearDown(self):
        self.tmp.cleanup()

    def test_hash_refers_to_array_file_not_metadata_file(self):
        evidence = {}
        audit_sidecar(self.root, self.item, evidence)
        self.assertEqual(len(evidence), 2)
        self.assertNotEqual(evidence[str(self.root / 'evidence.json')], self.item['sidecar_sha256'])

    def test_modified_array_is_rejected(self):
        (self.root / 'arrays.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            audit_sidecar(self.root, self.item, {})


if __name__ == '__main__':
    unittest.main()
