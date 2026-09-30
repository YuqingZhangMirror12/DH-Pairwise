"""Synthetic arithmetic/receipt tests only; no real outputs are fabricated."""
import copy
import math
import unittest
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.density_physical_v2 import analyze as a


def fixture():
    cases, rows = [], []
    for index in range(40):
        dataset = "real" if index < 20 else "ood" if index < 32 else "test"
        identity = dict(pair_id=f"synthetic-unit-test/{index}", dataset=dataset,
            fragment_a=f"a/{index}", fragment_b=f"b/{index}", arrays_sha256="synthetic")
        cases.append(identity.copy())
        branches = {}
        for name, z, translation in (("R", 0., [0., 0.]), ("A", .5, [0., 0.]), ("B", 1.5, [3., 4.]), ("C", .75, [0., 1.])):
            branches[name] = dict(full=dict(logit=z, probability=1/(1+math.exp(-z)), valid_counts=[512, 512]),
                layout=dict(valid=True, candidate_count=20, inlier_count=10, residual_px=2.,
                    translation_a_to_b_rc=translation, reason="ok"), sinkhorn_converged=True)
        for name, z, relative in (("B", 1., (.1, .3)), ("C", .6, (.01, .09))):
            errors = {s: {stage: dict(relative_l2=relative[i] if stage == "context" else 0., max_abs=.1 if stage == "context" else 0.)
                         for stage in ("patches", "encoded", "context")} for i, s in enumerate("ab")}
            branches[name]["anchors512"] = dict(score=dict(logit=z, probability=1/(1+math.exp(-z))), errors_to_A=errors)
        rows.append(dict(identity, branches=branches, bridge_A_minus_R=dict(logit=.5,
            probability=branches["A"]["full"]["probability"]-branches["R"]["full"]["probability"])))
    protocol = dict(schema="density-physical-context/2", status="complete", case_count=40, completed_cases=40,
        parameters_unchanged=True, original_conv_configuration_unchanged=True,
        model=dict(checkpoint_sha256=a.EXPECTED_CHECKPOINT), GPU_used=False, training=False, threshold_fit=False,
        GT_used=False, physical_masks_resized=False, coarse_size_changed=False,
        parameter_sha256="synthetic", script_sha256="synthetic", cases_sha256=a.EXPECTED_CASES,
        selection_json_sha256="synthetic", limitations=["Synthetic unit fixture; not real inference."])
    return dict(protocol=protocol, rows=rows), cases


class AnalyzeTests(unittest.TestCase):
    def test_pair_not_fragment_units_and_independent_bridge(self):
        result, cases = fixture()
        measured = a.analyze(result, cases)
        g = measured["groups"]["all"]
        self.assertEqual(g["pair_count"], 40)
        context = g["anchor_context_relative_l2"]
        self.assertEqual(context["B"]["observed_pair_count"], 40)
        self.assertAlmostEqual(context["B"]["mean"], .2)
        self.assertAlmostEqual(context["C"]["mean"], .05)
        self.assertAlmostEqual(context["ratio_of_paired_mean_drift_reduction"], .75)
        self.assertAlmostEqual(g["bridge_R_to_A"]["absolute_logit"]["mean"], .5)
        self.assertAlmostEqual(g["full_logit_absolute_drift_from_A"]["B"]["mean"], 1.)
        self.assertAlmostEqual(g["anchor_logit_absolute_drift_from_A"]["ratio_of_paired_mean_drift_reduction"], .8)
        self.assertEqual(g["layout_change_from_A"]["B"]["translation_change_px"]["mean"], 5.)
        self.assertEqual(g["layout_change_from_A"]["C"]["translation_change_px"]["mean"], 1.)
        self.assertIn("40", a.findings(measured))

    def test_recovery_zero_missing_and_negative_unclipped(self):
        self.assertIsNone(a.recovery(0., .1)["recovery_fraction"])
        self.assertEqual(a.recovery(1., 3.)["recovery_fraction"], -2.)
        self.assertIsNone(a.recovery(None, 1.)["reduction"])
        self.assertEqual(a.stats([None, 1.])["missing_pair_count"], 1)

    def test_invalid_layout_is_not_imputed_as_zero_change(self):
        result, cases = fixture()
        result["rows"][0]["branches"]["C"]["layout"].update(valid=False, translation_a_to_b_rc=None, residual_px=None, reason="invalid")
        measured = a.analyze(result, cases)
        g = measured["groups"]["all"]
        self.assertEqual(g["layouts"]["C"]["valid_pairs"], 39)
        self.assertEqual(g["layout_change_from_A"]["C"]["translation_change_px"]["observed_pair_count"], 39)
        self.assertEqual(g["layout_change_from_A"]["C"]["valid_transitions"]["1->0"], 1)

    def test_reject_incomplete_or_changed_parameters(self):
        for key, value in (("status", "running"), ("completed_cases", 39), ("parameters_unchanged", False)):
            result, cases = fixture()
            result["protocol"][key] = value
            with self.assertRaises(ValueError):
                a.analyze(result, cases)

    def test_reject_missing_duplicate_or_wrong_endpoint(self):
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.__setitem__(1, rows[0]),
                         lambda rows: rows[0].update(fragment_a="wrong")):
            result, cases = fixture()
            mutation(result["rows"])
            with self.assertRaises(ValueError):
                a.analyze(result, cases)


if __name__ == "__main__":
    unittest.main()
