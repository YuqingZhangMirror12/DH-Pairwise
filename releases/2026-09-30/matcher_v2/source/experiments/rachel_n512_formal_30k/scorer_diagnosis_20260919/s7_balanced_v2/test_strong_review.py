import unittest

from .strong_review import placement_summary


class PlacementSummaryTest(unittest.TestCase):
    def test_actual_sides_and_conditional_attempts(self):
        rows=[]
        for label in (True,False):
            for width,attempts in ((20.,1),(40.,3)):
                rows.append(dict(label=label,recipe='wave_gaps',augmentation=dict(
                    weather_plan_selection=dict(attempts=attempts),
                    damage=dict(a=dict(applied=True,survival_island=dict(
                        core_width_px=width,anchor_count=6 if label else 0)),
                        b=dict(applied=False)))))
        pos,neg=placement_summary(rows)
        self.assertEqual(pos['islandSides'],2)
        self.assertEqual(pos['coreWidthPx'],neg['coreWidthPx'])
        self.assertEqual(pos['coreWidthPx']['median'],30.)
        self.assertEqual(pos['planAttempts']['mean'],2.)
        self.assertEqual(pos['anchorCounts'],{'6':2})
        self.assertEqual(neg['anchorCounts'],{'0':2})

    def test_legacy_absence_is_not_zero_width(self):
        rows=[dict(label=True,recipe='wave',augmentation=dict(
            damage=dict(a=dict(applied=True,survival_island=None))))]
        row=placement_summary(rows)[0]
        self.assertEqual(row['islandSides'],0)
        self.assertIsNone(row['coreWidthPx']['median'])
        self.assertIsNone(row['planAttempts']['mean'])


if __name__=='__main__':unittest.main()
