"""Two targeted metric semantics checks; no endpoint files or model access."""
import unittest

from . import recount_matcher_scorers as recount


class RecountSemanticsTests(unittest.TestCase):
    def test_unknown_ood_layout_is_null_not_zero_success(self):
        row = dict(label=True, score=.8, decision_valid=True, target_translation_rc=None,
                   layout_valid=True, layout_error=None)
        result = recount.metrics([row], .5)
        self.assertEqual(result['tp'], 1)
        self.assertEqual(result['layout_positive_states'], {'unknown': 1})
        for name in ('layout20_correct', 'layout20_rate_all_positive', 'accepted_correct', 'rejected_correct'):
            self.assertIsNone(result[name])

    def test_invalid_decision_ranks_at_minus_one(self):
        rows = [dict(label=True, score=.1, decision_valid=True),
                dict(label=False, score=.9, decision_valid=False)]
        self.assertEqual(recount.auc(rows), 1.)
        rows = [dict(label=True, score=.9, decision_valid=False),
                dict(label=False, score=.1, decision_valid=True)]
        self.assertEqual(recount.auc(rows), 0.)


if __name__ == '__main__':
    unittest.main()
