import copy
import json
from pathlib import Path
import tempfile
import unittest

import decompose as d


def row(i, label, g, delta, eligible=True, error=10, valid=True):
    return {"pair_id": str(i), "label": label, "decision_valid": valid,
            "classification": {"fused": d.sigmoid(g+delta)},
            "target_translation_rc": [0., 1.],
            "layouts": {"full_top2_mode": {"valid": True, "translation_l2_px": error}},
            "candidate_details": {"global_logit": g, "local_residual_logit": delta, "local_eligible": eligible}}


class DecompositionTests(unittest.TestCase):
    def test_signed_residual_and_layout_flips(self):
        rows = [row(1, True, -1, 2), row(2, True, 1, -2),
                row(3, False, -1, 2), row(4, False, 1, -2), row(5, True, -1, 2, error=30)]
        values, diag = d.reconstruct(rows)
        self.assertFalse(diag["issues"])
        r = d.flips(rows, values, .5, "real")
        self.assertEqual(r["positive_rescued"]["pair_ids"], ["1", "5"])
        self.assertEqual(r["positive_lost"]["pair_ids"], ["2"])
        self.assertEqual(r["negative_new_false_positive"]["pair_ids"], ["3"])
        self.assertEqual(r["negative_corrected_false_positive"]["pair_ids"], ["4"])
        self.assertEqual(r["correct_layout20"]["known_positive_rescued"]["pair_ids"], ["1"])
        self.assertEqual(r["correct_layout20"]["known_positive_lost"]["pair_ids"], ["2"])

    def test_sigmoid_stability_and_boundary(self):
        self.assertEqual(d.sigmoid(1000), 1.)
        self.assertEqual(d.sigmoid(-1000), 0.)
        rows = [row(1, True, -1, 1, error=20), row(2, True, -1, 2, valid=False)]
        values, _ = d.reconstruct(rows)
        r = d.flips(rows, values, .5, "test")
        self.assertEqual(r["positive_rescued"]["pair_ids"], ["1"])
        self.assertEqual(r["correct_layout20"]["raw_correct_count"], 2)

    def test_missing_components_not_zero(self):
        rows = [row(1, True, 0, 0), row(2, True, 0, 0)]
        rows[0]["candidate_details"] = None
        del rows[1]["candidate_details"]["local_residual_logit"]
        values, diag = d.reconstruct(rows)
        self.assertFalse(values)
        self.assertEqual(diag["issues"]["candidate_details_missing"]["pair_ids"], ["1"])
        self.assertIsNone(diag["max_probability_absolute_error"])

    def test_ineligible_delta_and_reconstruction_mismatch(self):
        rows = [row(1, True, 0, 0, False), row(2, True, 0, 1, False), row(3, False, 1, 1)]
        rows[2]["classification"]["fused"] = .1
        _, diag = d.reconstruct(rows)
        self.assertEqual(diag["issues"]["ineligible_with_nonzero_applied_residual"]["pair_ids"], ["2"])
        self.assertEqual(diag["issues"]["fused_probability_reconstruction_mismatch"]["pair_ids"], ["3"])

    def test_tolerance_does_not_hide_threshold_crossing(self):
        rows = [row(1, True, -1, 1)]
        rows[0]["classification"]["fused"] = .5 - 1e-7
        values, diag = d.reconstruct(rows)
        self.assertFalse(diag["issues"])
        self.assertEqual(d.flips(rows, values, .5, "real")["status"], "unavailable")

    def test_ood_has_only_positive_no_layout_metrics(self):
        rows = [row(1, True, -1, 2), row(2, True, 1, -2)]
        values, _ = d.reconstruct(rows)
        r = d.flips(rows, values, .5, "ood")
        self.assertEqual(r["global_positive_recall"], .5)
        self.assertEqual(r["fused_positive_recall"], .5)
        for key in ("f1", "accuracy", "correct_layout20", "negative_new_false_positive"):
            self.assertNotIn(key, r)

    def test_unfinished_endpoint_not_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/"protocol.json").write_text(json.dumps({"status": "running"}))
            (path/"summary.json").write_text("half written")
            result = d.decompose_endpoint(path, "real")
            self.assertEqual(result["status"], "unavailable")
            self.assertNotIn("populations", result)


if __name__ == "__main__":
    unittest.main()
