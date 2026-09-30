import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare as c


def row(i, positive=True, score=.5, keep=True, strict=True, error=20):
    return {"pair_id": str(i), "label": positive, "fragment_a": "A"+str(i), "fragment_b": "B"+str(i),
            "review_status": "keep" if keep else "exclude", "strict_member": strict,
            "target_translation_rc": [0., 1.], "decision_valid": True, "classification": {"fused": score},
            "layouts": {"full_top2_mode": {"valid": True, "translation_l2_px": error}}}


class CompareTests(unittest.TestCase):
    def test_ties_and_invalid_rank(self):
        rows = [row(1), row(2, False)]
        self.assertEqual(c.ranking(rows), {"auroc": .5, "auprc": .5})
        rows[1]["decision_valid"] = False
        self.assertEqual(c.ranking(rows), {"auroc": 1., "auprc": 1.})

    def test_threshold_layout_boundary_and_missing(self):
        rows = [row(1), row(2, score=.49), row(3, error=20.00001), row(4, False)]
        m = c.metrics(rows, .5, "real")
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 1, 0))
        self.assertEqual((m["layout20"]["raw_correct"], m["layout20"]["accepted_correct"],
                          m["layout20"]["classification_FN_but_layout_correct"]), (2, 1, 1))
        del rows[0]["layouts"]
        m = c.metrics(rows, .5, "real")
        self.assertIsNone(m["layout20"]["raw_correct"])
        self.assertEqual(m["layout20"]["missing_pair_ids"], ["1"])

    def test_population_rules(self):
        rows = [row(1), row(2, keep=False), row(3, False), row(4, False, strict=False)]
        groups = c.populations(rows, "real")
        self.assertEqual({k: len(v) for k, v in groups.items()},
                         {"all1016": 4, "keep803": 3, "strict547": 3, "keep_strict334": 2})
        self.assertEqual({r["pair_id"] for r in groups["strict547"]}, {"1", "2", "3"})

    def test_id_alignment_not_order_and_layout_not_identity(self):
        a = {"1": row(1), "2": row(2)}
        b = {"2": copy.deepcopy(a["2"]), "1": copy.deepcopy(a["1"])}
        b["1"]["layouts"]["full_top2_mode"]["translation_l2_px"] = 21
        self.assertTrue(c.identity_difference(a, b, "real")["equal"])
        self.assertFalse(c.identity_difference(a, b, "real")["order_equal"])
        self.assertEqual(c.layout_changes(a, b)["layout20_success_changed_pair_ids"], ["1"])
        b["1"]["review_status"] = "exclude"
        self.assertFalse(c.identity_difference(a, b, "real")["equal"])
        del b["2"]
        self.assertEqual(c.identity_difference(a, b, "real")["missing_pair_ids"], ["2"])

    def test_protocol_gate_never_reads_halfwritten_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/"protocol.json").write_text('{"status":"running"}')
            (p/"summary.json").write_text('not json')
            (p/"pair_results.jsonl").write_text('not json')
            result = c.load_endpoint(p, "real")
            self.assertEqual(result["reason"], "protocol_not_complete: running")
            self.assertEqual(set(result["sources"]), {"protocol.json"})

    def test_complete_protocol_missing_row_does_not_zero_fill(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(c.EXPECTED, {"real": (2, 1, 1)}):
            p = Path(tmp)
            model = {"checkpoint_sha256": "x", "operating_points": {"thresholds": {op: .5 for op in c.OPS}}}
            protocol = {"status": "complete", "split": "real", "sample_count": 2, "model": model,
                        "thresholds_fitted": False, "test_or_real_used_for_fit": False, "ood_used_for_fit": False}
            summary = {"status": "complete", "split": "real", "model": model,
                       "selection_on_this_population": False, "threshold_fitting_performed": False}
            (p/"protocol.json").write_text(json.dumps(protocol))
            (p/"summary.json").write_text(json.dumps(summary))
            (p/"pair_results.jsonl").write_text(json.dumps(row(1))+"\n"+json.dumps(row(2, False)))
            self.assertEqual(c.load_endpoint(p, "real")["status"], "complete")
            (p/"pair_results.jsonl").write_text(json.dumps(row(1)))
            result = c.load_endpoint(p, "real")
            self.assertEqual(result["status"], "unavailable")
            self.assertIn("population count mismatch", result["reason"])

    def test_ood_never_binary_metrics(self):
        m = c.metrics([row(1), row(2, score=.4)], .5, "ood")
        self.assertEqual(m["positive_recall"], .5)
        for key in ("accuracy", "precision", "f1", "auroc", "auprc", "layout20"):
            self.assertNotIn(key, m)


if __name__ == "__main__":
    unittest.main()
