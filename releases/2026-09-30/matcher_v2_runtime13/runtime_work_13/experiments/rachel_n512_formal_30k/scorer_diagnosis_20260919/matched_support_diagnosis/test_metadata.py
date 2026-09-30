from copy import deepcopy
import unittest

from .metadata import COUNT_PROVENANCE, join_metadata


def entry(pair_id, label, a="doc-a", b="doc-a", area_a=100, area_b=400):
    return dict(pair_id=pair_id,label=label,s7_recipe="partial_curve",changed_pair=True,
        source_stratum="fixture",area_ratio_band="other",surviving_correspondences=17,
        inherited_match_count=23,source_row=dict(pair_id=pair_id,label=label,split="train",
            fragment_a=dict(split_unit_id=a,foreground_area=area_a),
            fragment_b=dict(split_unit_id=b,foreground_area=area_b)))


class MetadataTests(unittest.TestCase):
    def test_cache_order_fields_area_and_source_kinds_without_mutation(self):
        entries=[entry("positive",True),entry("same",False),entry("cross",False,b="doc-b")]
        records=[dict(pair_id=p,label=float(p=="positive")) for p in ("cross","positive","same")]
        before=deepcopy((entries,records))
        result=join_metadata(entries,records)
        self.assertEqual([r["pair_id"] for r in result],["cross","positive","same"])
        self.assertEqual([r["cache_ordinal"] for r in result],[0,1,2])
        self.assertEqual([r["negative_source_kind"] for r in result],["cross_source","positive","same_source"])
        self.assertTrue(all(r["source_area_ratio"]==.25 for r in result))
        self.assertTrue(all(r["source_area_ratio_stage"]=="preaugmentation" for r in result))
        self.assertTrue(all(r["correspondence_count_provenance"]==COUNT_PROVENANCE for r in result))
        self.assertEqual(result[1]["surviving_correspondences"],17)
        self.assertEqual(result[1]["inherited_match_count"],23)
        self.assertEqual(result[1]["s7_recipe"],"partial_curve")
        self.assertTrue(result[1]["changed_pair"])
        self.assertEqual(result[1]["source_stratum"],"fixture")
        self.assertEqual(result[1]["area_ratio_band"],"other")
        self.assertEqual((entries,records),before)

    def test_missing_area_and_metadata_remain_none(self):
        row=entry("p",True,a=None,b=None,area_a=None)
        del row["inherited_match_count"]
        result=join_metadata([row],[dict(pair_id="p",label=1)])[0]
        self.assertIsNone(result["source_area_ratio"])
        self.assertIsNone(result["inherited_match_count"])
        self.assertEqual(result["negative_source_kind"],"positive")

    def test_duplicates_on_either_side_reject(self):
        e=entry("p",True);r=dict(pair_id="p",label=True)
        for entries,records in (([e,e],[r]),([e],[r,r])):
            with self.subTest(entries=len(entries)),self.assertRaisesRegex(ValueError,"duplicate"):
                join_metadata(entries,records)

    def test_missing_or_extra_membership_reject(self):
        e=entry("p",True);r=dict(pair_id="p",label=True)
        for entries,records in (([e],[]),([],[r]),([e],[dict(pair_id="q",label=True)])):
            with self.subTest(entries=entries,records=records),self.assertRaisesRegex(ValueError,"full membership"):
                join_metadata(entries,records)

    def test_label_mismatch_and_nonbinary_labels_reject(self):
        e=entry("p",True)
        for label in (False,2,"1",None,float("nan")):
            with self.subTest(label=label),self.assertRaises(ValueError):
                join_metadata([e],[dict(pair_id="p",label=label)])
        e["source_row"]["label"]=False
        with self.assertRaisesRegex(ValueError,"source/manifest label"):
            join_metadata([e],[dict(pair_id="p",label=True)])

    def test_negative_missing_source_id_is_not_same_source(self):
        for bad in (None,"", " "):
            e=entry("negative",False,a=bad,b=bad)
            with self.subTest(bad=bad),self.assertRaisesRegex(ValueError,"missing split_unit_id"):
                join_metadata([e],[dict(pair_id="negative",label=0.)])

    def test_nontrain_and_invalid_present_area_reject(self):
        for alter in (lambda e:e["source_row"].update(split="val"),
                      lambda e:e["source_row"]["fragment_a"].update(foreground_area=0)):
            e=entry("p",True);alter(e)
            with self.assertRaises(ValueError):
                join_metadata([e],[dict(pair_id="p",label=True)])


if __name__=="__main__":
    unittest.main()
