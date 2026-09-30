"""Synthetic numerical and completed-input contract tests; no model/GPU/data scan."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("spectral_diagnostics", Path(__file__).with_name("spectral_diagnostics.py"))
spectral = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spectral)
np = spectral.np


class SpectralTests(unittest.TestCase):
    def test_rectangular_rank_one(self):
        result = spectral.spectral_metrics(np.full((3, 5), .2))
        self.assertAlmostEqual(result["raw_transport_total_mass"], 3)
        shape = result["spectral_shape"]
        self.assertAlmostEqual(shape["sigma1_over_frobenius"], 1)
        self.assertAlmostEqual(shape["effective_rank_entropy_singular"], 1)
        self.assertAlmostEqual(shape["participation_ratio_energy"], 1)
        self.assertEqual(shape["topk_effective_k"]["16"], 3)
        self.assertEqual(result["rows_select_columns"]["argmax_unique_opposite_count"], 1)
        self.assertEqual(result["rows_select_columns"]["exact_maximum_tie_axis_count"], 3)

    def test_identity_known_energy(self):
        result = spectral.spectral_metrics(np.eye(8))
        shape = result["spectral_shape"]
        self.assertAlmostEqual(shape["sigma1_over_frobenius"], 1 / np.sqrt(8))
        self.assertAlmostEqual(shape["effective_rank_entropy_singular"], 8)
        self.assertAlmostEqual(shape["participation_ratio_energy"], 8)
        self.assertAlmostEqual(shape["participation_ratio_singular"], 8)
        for k in spectral.TOPK:
            self.assertAlmostEqual(shape["topk_energy_fraction"][str(k)], min(k, 8) / 8)
        self.assertEqual(result["mutual_argmax"]["pair_count"], 8)

    def test_scale_shape_invariance_mass_changes(self):
        matrix = np.array([[.1, .2, .3], [.2, .4, .1]])
        a, b = map(spectral.spectral_metrics, (matrix, matrix * 7))
        self.assertAlmostEqual(b["raw_transport_total_mass"], 7 * a["raw_transport_total_mass"])
        self.assertAlmostEqual(b["raw_sigma1"], 7 * a["raw_sigma1"])
        for key in ("sigma1_over_frobenius", "effective_rank_entropy_singular", "participation_ratio_energy"):
            self.assertAlmostEqual(a["spectral_shape"][key], b["spectral_shape"][key])
        self.assertAlmostEqual(a["mass_normalized"]["sigma1"], b["mass_normalized"]["sigma1"])

    def test_transpose_and_permutation(self):
        matrix = np.array([[.01, .5, 0], [.6, .02, .3]])
        a, b, c = map(spectral.spectral_metrics, (matrix, matrix.T, matrix[::-1, ::-1]))
        np.testing.assert_allclose(a["singular_values"], b["singular_values"])
        np.testing.assert_allclose(a["singular_values"], c["singular_values"])
        self.assertEqual(a["rows_select_columns"], b["columns_select_rows"])
        self.assertEqual(a["mutual_argmax"]["pair_count"], 2)

    def test_spectrum_cannot_distinguish_contiguous_and_scrambled_matches(self):
        contiguous = np.eye(8)
        scrambled = np.eye(8)[[0, 4, 1, 5, 2, 6, 3, 7]]
        a, b = map(spectral.spectral_metrics, (contiguous, scrambled))
        np.testing.assert_array_equal(a["singular_values"], b["singular_values"])
        self.assertEqual(a["spectral_shape"], b["spectral_shape"])
        self.assertEqual(a["mutual_argmax"], b["mutual_argmax"])
        self.assertEqual(a["rows_select_columns"], b["rows_select_columns"])
        self.assertEqual(np.count_nonzero(np.abs(np.diff(contiguous.argmax(1))) == 1), 7)
        self.assertEqual(np.count_nonzero(np.abs(np.diff(scrambled.argmax(1))) == 1), 0)

    def test_zero_and_zero_rows(self):
        result = spectral.spectral_metrics(np.zeros((2, 3)))
        self.assertIsNone(result["spectral_shape"]["sigma1_over_frobenius"])
        self.assertIsNone(result["spectral_shape"]["topk_energy_fraction"]["1"])
        self.assertEqual(result["spectral_shape"]["effective_rank_entropy_singular"], 0)
        self.assertEqual(result["rows_select_columns"]["argmax_unique_opposite_count"], 0)
        self.assertEqual(result["mutual_argmax"]["pair_count"], 0)
        result = spectral.spectral_metrics(np.array([[0., 0.], [.5, .5]]))
        self.assertEqual(result["rows_select_columns"]["positive_mass_fraction"], .5)
        self.assertEqual(result["rows_select_columns"]["conditional_maximum_given_positive_axis_mass"]["mean"], .5)

    def test_invalid_input(self):
        for matrix in ([1, 2], np.empty((0, 2)), [[-1]], [[np.nan]], [[np.inf]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                spectral.spectral_metrics(matrix)

    def test_mask_removes_padding_not_last_valid_token(self):
        arrays = {"sinkhorn_assignment": np.arange(12).reshape(3, 4),
                  "valid_a": np.array([True, False, True]), "valid_b": np.array([False, True, False, True])}
        matrix, metadata = spectral.masked_assignment(arrays)
        np.testing.assert_array_equal(matrix, [[1, 3], [9, 11]])
        self.assertEqual(metadata["excluded_padding_mass"], 42)
        self.assertFalse(metadata["dustbins_removed_by_this_script"])
        arrays["valid_a"] = np.array([1, 0, 1])
        with self.assertRaises(ValueError):
            spectral.masked_assignment(arrays)


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.arm = self.root / "s4"
        self.arm.mkdir()
        self.output = self.root / "spectral" / "s4"
        npz = self.arm / "one.npz"
        np.savez(npz, sinkhorn_assignment=np.array([[.2, .1, .3], [.4, .1, .2]]),
                 valid_a=np.ones(2, dtype=bool), valid_b=np.ones(3, dtype=bool))
        self.case = dict(dataset="real", pair_id="p1", label=True, score=.9,
                         arrays_path=npz.name, arrays_sha256=spectral.sha256(npz))
        self.selection = dict(dataset="real", pair_id="p1", label=True, stratum="real_TP_good")
        (self.arm / "cases.json").write_text(json.dumps([self.case]))
        self.protocol = dict(status="complete", selected_pairs=[self.selection], sample_count=1,
                             completed_count=1, cases_sha256=spectral.sha256(self.arm / "cases.json"))
        self.write_protocol()

    def tearDown(self):
        self.temp.cleanup()

    def write_protocol(self):
        (self.arm / "protocol.json").write_text(json.dumps(self.protocol))

    def test_completed_roundtrip_and_no_overwrite(self):
        hashes = {p.name: spectral.sha256(p) for p in self.arm.iterdir()}
        result = spectral.run(self.arm, self.output)
        self.assertEqual(result["case_count"], 1)
        self.assertEqual(result["cases"][0]["metrics"]["shape"], [2, 3])
        self.assertTrue(result["sources"]["input_hashes_unchanged_after_run"])
        self.assertEqual(hashes, {p.name: spectral.sha256(p) for p in self.arm.iterdir()})
        self.assertEqual(json.loads((self.output / "metrics.json").read_text())["status"], "complete")
        self.assertIn("选择性示例", (self.output / "SUMMARY.md").read_text())
        with self.assertRaises(FileExistsError):
            spectral.run(self.arm, self.output)

    def test_incomplete_gate_before_npz_load(self):
        self.protocol["status"] = "running"
        self.write_protocol()
        with patch.object(spectral.np, "load", side_effect=AssertionError("must not load NPZ")):
            with self.assertRaisesRegex(ValueError, "must be complete"):
                spectral.run(self.arm, self.output)
        self.assertFalse(self.output.exists())

    def test_source_hash_gate(self):
        (self.arm / "cases.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "cases.json hash"):
            spectral.run(self.arm, self.output)
        self.assertFalse(self.output.exists())

    def test_npz_hash_gate(self):
        (self.arm / "one.npz").write_bytes(b"not the original NPZ")
        with self.assertRaisesRegex(ValueError, "NPZ hash"):
            spectral.run(self.arm, self.output)

    def test_label_and_population_gate(self):
        self.protocol["selected_pairs"][0]["label"] = False
        self.write_protocol()
        with self.assertRaisesRegex(ValueError, "label differs"):
            spectral.run(self.arm, self.output)
        self.protocol["selected_pairs"][0]["label"] = True
        self.protocol["completed_count"] = 2
        self.write_protocol()
        with self.assertRaisesRegex(ValueError, "population"):
            spectral.run(self.arm, self.output)

    def test_no_output_into_original_arm(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            spectral.run(self.arm, self.arm / "new")

    def test_path_traversal_gate(self):
        self.case["arrays_path"] = "../elsewhere.npz"
        (self.arm / "cases.json").write_text(json.dumps([self.case]))
        self.protocol["cases_sha256"] = spectral.sha256(self.arm / "cases.json")
        self.write_protocol()
        with self.assertRaisesRegex(ValueError, "escapes"):
            spectral.run(self.arm, self.output)


if __name__ == "__main__":
    unittest.main()
