import copy
from types import SimpleNamespace
import unittest

from ..s7_consensus_v1.metrics import summarize
from .terminal import thresholds_for_export
from .test_validation import rows_fixture
from .validation import select_dunhuang


class ThresholdTests(unittest.TestCase):
    def saved(self, selection):
        return dict(module='scorer_patch', selection_kind=selection, updates=1500,
            observation=dict(update=1500, selection_eligible=True, test_used=False,
                simulation=dict(stage='scorer', real_used=False, key=[.9, .8, .7], selection_value=.9, threshold=.5),
                real_development=select_dunhuang(rows_fixture(), SimpleNamespace(threshold_tie_preference=.3), summarize)))

    def test_Dun_CAL_never_recalibrates_Turufan_or_SIM(self):
        thresholds, origins = thresholds_for_export(self.saved('real_best'))
        self.assertEqual(thresholds, dict(sim_test=.5, dunhuang_cv=.31, turufan=.5))
        self.assertIn('SIM-CAL', origins['turufan']); self.assertIn('Dunhuang CAL', origins['dunhuang_cv'])
        for kind in ('sim_best', 'equal_budget_endpoint'):
            self.assertEqual(set(thresholds_for_export(self.saved(kind))[0].values()), {.5})

    def test_reject_matcher_zero_update_test_or_Turufan_selection(self):
        for mutate in (lambda r: r.update(module='matcher'), lambda r: r.update(updates=0),
                       lambda r: r['observation'].update(test_used=True),
                       lambda r: r['observation']['real_development']['thresholds'].update(turufan=.2)):
            record = self.saved('real_best'); mutate(record)
            with self.assertRaises(ValueError):thresholds_for_export(record)


if __name__ == '__main__':unittest.main()
