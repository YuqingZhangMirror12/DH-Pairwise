import unittest

import torch

from .geometry import compact_contour, compact_matrix, cyclic_delta_px, deterministic_prefix_sum


class ValidContourTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.square = torch.tensor([[2., 2.], [2., 12.], [12., 12.], [12., 2.]])

    def test_padding_maps_order_and_gather(self):
        p = torch.full((1, 9, 2), float('nan'))
        ids = torch.tensor([1, 3, 4, 8])
        p[0, ids] = self.square
        valid = torch.isfinite(p[..., 0])
        g = compact_contour(p, valid)
        torch.testing.assert_close(g.points[0], self.square)
        self.assertEqual(g.compact_to_original[0].tolist(), ids.tolist())
        self.assertEqual(g.original_to_compact[0].tolist(), [-1, 0, -1, 1, 2, -1, -1, -1, 3])
        torch.testing.assert_close(g.gather(p), self.square[None])
        torch.testing.assert_close(g.scatter(g.points)[valid], self.square)
        self.assertTrue((g.scatter(g.points)[~valid] == 0).all())

    def test_real_cycle_closure_not_storage_padding(self):
        g = compact_contour(self.square[None], torch.ones(1, 4, dtype=torch.bool))
        torch.testing.assert_close(g.next_step_px, torch.full((1, 4), 10.))
        torch.testing.assert_close(g.arc_px, torch.tensor([[0., 10., 20., 30.]]))
        torch.testing.assert_close(g.cell_px.sum(1), g.perimeter_px)

    def test_outward_normals_both_windings(self):
        for p in (self.square, self.square.flip(0)):
            g = compact_contour(p[None], torch.ones(1, 4, dtype=torch.bool))
            self.assertTrue(((g.outward_normal_rc[0] * (p - p.mean(0))).sum(-1) > 0).all())
            torch.testing.assert_close(g.points[0], p)  # No winding rewrite.

    def test_cyclic_origin_changes_only_arc_origin(self):
        a = compact_contour(self.square[None], torch.ones(1, 4, dtype=torch.bool))
        b = compact_contour(self.square.roll(1, 0)[None], torch.ones(1, 4, dtype=torch.bool))
        torch.testing.assert_close(a.outward_normal_rc.roll(1, 1), b.outward_normal_rc)
        da = cyclic_delta_px(a.arc_px[:, :, None], a.arc_px[:, None, :], a.perimeter_px[:, None, None])
        db = cyclic_delta_px(b.arc_px[:, :, None], b.arc_px[:, None, :], b.perimeter_px[:, None, None])
        torch.testing.assert_close(da.roll((1, 1), (1, 2)), db)

    def test_batch_composition_padding_does_not_change_geometry(self):
        a = compact_contour(self.square[None], torch.ones(1, 4, dtype=torch.bool))
        p = torch.zeros(2, 8, 2)
        p[0, :4] = self.square
        p[1] = self.square.repeat_interleave(2, dim=0)
        valid = torch.ones(2, 8, dtype=torch.bool)
        valid[0, 4:] = False
        p[0, 4:] = float('nan')
        b = compact_contour(p, valid)
        torch.testing.assert_close(a.cell_px[0], b.cell_px[0, :4])
        torch.testing.assert_close(a.outward_normal_rc[0], b.outward_normal_rc[0, :4])
        torch.testing.assert_close(b.cell_px.sum(1), torch.tensor([40., 40.]))

    def test_duplicate_points_do_not_add_arc_length(self):
        p = self.square.repeat_interleave(2, dim=0)
        g = compact_contour(p[None], torch.ones(1, 8, dtype=torch.bool))
        torch.testing.assert_close(g.cell_px.sum(1), torch.tensor([40.]))

    def test_matrix_maps_and_invalid_nan(self):
        p = torch.full((2, 6, 2), float('nan'))
        p[0, [0, 1, 4, 5]] = self.square
        p[1, :4] = self.square
        valid = torch.isfinite(p[..., 0])
        g = compact_contour(p, valid)
        q = torch.full((2, 6, 6), float('nan'))
        v = torch.arange(16.).reshape(4, 4)
        for b in range(2):
            ids = valid[b].nonzero().flatten()
            q[b, ids[:, None], ids[None, :]] = v
        torch.testing.assert_close(compact_matrix(q, g, g), v[None].expand(2, -1, -1))

    def test_empty_and_degenerate_do_not_invent_normals(self):
        p = torch.full((2, 4, 2), float('nan'))
        p[1] = torch.tensor([[0., 1.], [0., 2.], [0., 3.], [0., 4.]])
        valid = torch.isfinite(p[..., 0])
        g = compact_contour(p, valid)
        self.assertTrue(torch.isfinite(g.points).all())
        self.assertTrue((g.normal_reliability == 0).all())
        self.assertTrue((g.cell_px[0] == 0).all())

    def test_valid_nan_is_not_silently_hidden(self):
        p = self.square[None].clone()
        p[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'VALID'):
            compact_contour(p, torch.ones(1, 4, dtype=torch.bool))

    def test_deterministic_prefix_matches_cpu_cumsum(self):
        for n in (1, 4, 7, 512):
            x = torch.arange(n, dtype=torch.float32)[None].repeat(2, 1)
            torch.testing.assert_close(deterministic_prefix_sum(x), x.cumsum(-1))


if __name__ == '__main__':
    unittest.main()
