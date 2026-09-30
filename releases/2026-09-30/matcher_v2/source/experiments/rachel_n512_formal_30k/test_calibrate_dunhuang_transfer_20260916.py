import unittest

from experiments.rachel_n512_formal_30k.calibrate_dunhuang_transfer_20260916 import fit_thresholds, metrics, auc


def rows(labels, scores):
    return [dict(label=y, score=s) for y, s in zip(labels, scores)]


class ThresholdTests(unittest.TestCase):
    def test_ties_not_split(self):
        sample = rows([True, False, True], [.8, .8, .2])
        m = metrics(sample, .8)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 1))

    def test_constant_score(self):
        sample = rows([True, False, True], [.5] * 3)
        self.assertEqual(set(fit_thresholds(sample).values()), {.5})
        self.assertEqual(auc(sample), .5)

    def test_precision_rule_not_largest_feasible_threshold(self):
        # Highest threshold with Recall>=.5 is .9: precision=.5.
        # At .8, precision improves to2/3: must prefer .8.
        sample = rows([True, False, True], [.9, .9, .8])
        self.assertEqual(fit_thresholds(sample, .5)["dunhuang_recall95_max_precision"], .8)

    def test_requires_two_classes(self):
        with self.assertRaises(ValueError):
            fit_thresholds(rows([True], [.9]))

    def test_boundary_inclusive(self):
        sample = rows([True, False], [.75, .25])
        self.assertEqual(fit_thresholds(sample)["dunhuang_max_f1"], .75)
        self.assertEqual(metrics(sample, .75)["f1"], 1.)


if __name__ == "__main__":
    unittest.main()
