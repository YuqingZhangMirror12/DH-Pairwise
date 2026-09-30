import copy
import unittest

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.analyze_token_dilution import (
    FRACTIONS, SCHEMA, aggregate_cases, analyze, quantile, sigmoid, summarize_case,
)


def fixture(repeats=16):
    # Ten valid tokens per fragment, with two selected inlier endpoints.
    def point(logit, count, retained):
        return dict(logit=logit, probability=sigmoid(logit), count_a=count,
                    count_b=count, no_evidence=count == 0, delta_logit=logit,
                    retained_inliers_a=retained, retained_inliers_b=retained)
    interventions = []
    for rep in range(repeats):
        for fraction in FRACTIONS:
            if rep and fraction in (0., 1.):
                continue
            interventions.append(dict(kind="retain_inliers_add_context", repeat=rep,
                noninlier_fraction=fraction, **point(2 * (1 - fraction), 2 + int(8 * fraction), 2)))
    interventions.append(dict(kind="remove_inliers", repeat=0, **point(-1., 8, 0)))
    for rep in range(repeats):
        interventions.append(dict(kind="remove_random_same_count", repeat=rep,
            **point(-.1 + .01 * rep, 8, 1)))
    return dict(model="s6", dataset="real", pair_id="test/pair", name="fixture",
        stratum="real_FN_good", label=True, layout_valid=True, layout_error_px=4.,
        threshold=.6123, saved_forward_error=0., permutation_logit_error=0.,
        baseline=dict(logit=0., probability=.5, count_a=10, count_b=10, no_evidence=False),
        inlier_count_a=2, inlier_count_b=2, interventions=interventions)


class AnalysisTests(unittest.TestCase):
    def test_case_first_signed_contrast_and_dose(self):
        summary = summarize_case(fixture(), 16)
        self.assertEqual(summary["random_same_count_deletion"]["delta_logit"]["n"], 16)
        self.assertAlmostEqual(summary["remove_inlier_minus_mean_random_delta_logit"], -.975)
        self.assertEqual(summary["remove_inlier_drop_larger_than_random_draws"], 16)
        self.assertEqual(summary["nested_paths_nonincreasing"], 16)
        self.assertTrue(summary["median_curve_nonincreasing"])
        self.assertEqual([p["logit"]["n"] for p in summary["dose_curve"]], [1, 16, 16, 16, 1])
        self.assertFalse(summary["baseline_accepted"])
        self.assertTrue(summary["inlier_only_accepted"])

    def test_aggregate_cases_does_not_pool_16_draws(self):
        a = summarize_case(fixture(), 16)
        b = copy.deepcopy(a)
        b["random_same_count_deletion"]["delta_logit"]["mean"] = -2.
        summary = aggregate_cases([a, b])
        self.assertEqual(summary["n_cases"], 2)
        self.assertEqual(summary["case_mean_random_delta_logit"]["n"], 2)
        self.assertAlmostEqual(summary["case_mean_random_delta_logit"]["mean"], -1.0125)

    def test_bad_counts_or_duplicate_or_sign_rejected(self):
        for mutation in ("count", "duplicate", "delta"):
            row = fixture()
            if mutation == "count":
                row["interventions"][-1]["count_a"] = 9
            elif mutation == "duplicate":
                row["interventions"].append(row["interventions"][-1])
            else:
                row["interventions"][0]["delta_logit"] = -2.
            with self.assertRaises(ValueError):
                summarize_case(row, 16)

    def test_stratum_schema_coverage_checks(self):
        row = fixture()
        selection = [{k: row[k] for k in ("pair_id", "dataset", "stratum", "label")}]
        protocol = dict(schema_version=SCHEMA, status="complete", parameters_fitted=False,
            thresholds_fitted=False, GT_used_for_token_selection=False, fractions=FRACTIONS,
            selected_count=1, completed_count=1, models=["s6"], repeats=16)
        result = analyze([row], protocol, selection)
        self.assertEqual(result["coverage"]["unique_selected_cases"], 1)
        for field, value in (("schema_version", "unknown"), ("status", "partial"),
                             ("GT_used_for_token_selection", True), ("completed_count", 2)):
            with self.assertRaises(ValueError):
                analyze([row], dict(protocol, **{field: value}), selection)
        with self.assertRaises(ValueError):
            analyze([row, row], dict(protocol, completed_count=2), selection)

    def test_no_ood_layout_claim_and_percentile(self):
        row = fixture()
        row["dataset"] = "ood"
        with self.assertRaises(ValueError):
            summarize_case(row, 16)
        self.assertEqual(quantile([0, 1, 2, 3], .25), .75)
        self.assertEqual(quantile([7], .75), 7.)


if __name__ == "__main__":
    unittest.main()
