import unittest
from .heldout_plan import canonical_generator, native_negative_pool


def fragment(role, family, token, generator="gen3voronoi", area=1000):
    return dict(role=role,family=family,row=dict(fragment_token=token,generator=generator,
                foreground_area=area,bbox_aspect_ratio=1.0))


class HeldoutPlanTests(unittest.TestCase):
    def test_generator_subfamilies_remain_explicit_in_original_row(self):
        for name, canonical in (("gen2voronoi_2","Gen2"),("gen4voronoi_1_3","Gen4"),
                                ("gen5voronoi_1_1_3","Gen5")):
            self.assertEqual(canonical_generator(name),canonical)
        with self.assertRaises(ValueError):
            canonical_generator("straight_strip")

    def test_cross_negatives_only_same_role_same_generator_different_parent(self):
        pool=[fragment("cal","A","a"),fragment("cal","A","b"),fragment("cal","B","c"),
              fragment("select","S","s"),fragment("cal","B","wronggen","gen2voronoi_1"),
              fragment("cal","C","toosmall",area=1)]
        rows=native_negative_pool(pool,"cal","Gen3",100)
        self.assertEqual(len(rows),2)
        self.assertTrue(all(not row["label"] for row in rows))
        self.assertEqual({(r["fragment_a"]["fragment_token"],r["fragment_b"]["fragment_token"]) for r in rows},
                         {("a","c"),("b","c")})
        self.assertEqual(rows,native_negative_pool(pool,"cal","Gen3",100))
        self.assertEqual(len({r["pair_id"] for r in rows}),2)

    def test_one_parent_cannot_invent_negative(self):
        with self.assertRaisesRegex(ValueError,"no isolated"):
            native_negative_pool([fragment("cal","A","a"),fragment("cal","A","b")],"cal","Gen3",10)


if __name__ == "__main__":
    unittest.main()
