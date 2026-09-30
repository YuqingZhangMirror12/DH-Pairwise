import copy
import unittest

from .bind_threshold_scratch import checked_cluster


def fixture():
    matrix = dict(available=True, rows=[dict(a_bin=0, b_bin=0, a_start=0,
        a_end_exclusive=1, b_start=0, b_end_exclusive=1, mean=.600000003)],
        shape=[1, 1], normalization='none', total=.600000003, nonzero_count=1)
    cl = dict(cluster_id=0, selected=True,
        proposal=dict(original_union_edge_ids=[[0, 0]], pose_diameter_px=16,
                      actual_diameter_px=15.9, merged_hypothesis_ids=[1, 3]),
        heatmaps=dict(initial_recalled=matrix, final_recalled=copy.deepcopy(matrix)),
        readout=dict(score=.45, positive_evidence_px=.2, support_weights_a=[.2]),
        attention={}, links=dict(final_support=dict(used_to_limit_model_input=False,
            displayed_edges=1, omitted_edges=0, eligible_edges=1, rows=[dict(weight=.2)])),
        initial_translation_rc=[0, 0], refined_translation_rc=[1, 2],
        underconstrained=False, overlap={})
    audit = dict(cluster_id=0, evidence_mode='exact_union_q',
                 directional_mass_used_for_scorer=False, union_absolute_q_mass=.600000024)
    return cl, audit


class BindingTests(unittest.TestCase):
    def test_float32_audit_and_float64_display_are_consistent(self):
        cl, au = fixture()
        result = checked_cluster(cl, au, None)
        self.assertEqual(result['union_edge_count'], 1)
        self.assertEqual(result['heatmaps']['final_recalled']['means'], [[.600000003]])
        self.assertNotIn('support_weights_a', result)
        self.assertIsNone(result['error_px'])

    def test_no_duplicate_evidence(self):
        cl, au = fixture(); cl['proposal']['original_union_edge_ids'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            checked_cluster(cl, au, 3)

    def test_not_radius_or_chain(self):
        cl, au = fixture(); cl['proposal']['actual_diameter_px'] = 20
        with self.assertRaisesRegex(ValueError, 'diameter'):
            checked_cluster(cl, au, 3)

    def test_no_directional_score_substitution(self):
        cl, au = fixture(); au['directional_mass_used_for_scorer'] = True
        with self.assertRaisesRegex(ValueError, 'mode'):
            checked_cluster(cl, au, 3)

    def test_no_changed_q_sum(self):
        cl, au = fixture(); au['union_absolute_q_mass'] = .62
        with self.assertRaisesRegex(ValueError, 'sum'):
            checked_cluster(cl, au, 3)

    def test_no_changed_final_q_cells_even_equal_sum(self):
        cl, au = fixture(); cl['heatmaps']['final_recalled']['rows'][0]['mean'] = .59
        with self.assertRaisesRegex(ValueError, 'two passes'):
            checked_cluster(cl, au, 3)

    def test_display_truncation_not_input_truncation(self):
        cl, au = fixture(); cl['links']['final_support']['used_to_limit_model_input'] = True
        with self.assertRaisesRegex(ValueError, 'display limit'):
            checked_cluster(cl, au, 3)


if __name__ == '__main__':
    unittest.main()
