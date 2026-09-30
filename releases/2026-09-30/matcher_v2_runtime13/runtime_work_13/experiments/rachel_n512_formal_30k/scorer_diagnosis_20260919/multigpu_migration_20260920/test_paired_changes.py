from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.multigpu_migration_20260920 import paired_changes as m


def row(i, label=True, score=.5, error=20, keep=True, strict=False):
    return dict(pair_id=str(i), label=label, fragment_a="a"+str(i), fragment_b="b"+str(i),
                target_translation_rc=[0., 1.], review_status="keep" if keep else "exclude",
                strict_member=strict, classification=dict(fused=score), decision_valid=True,
                layouts=dict(full_top2_mode=dict(valid=True, translation_rc=[0., 1.], translation_l2_px=error)))


def endpoint(root, rows, split="test", threshold=.5):
    root.mkdir()
    model = dict(checkpoint_sha256=root.name, operating_points=dict(thresholds={op:threshold for op in m.base.OPS}))
    protocol = dict(status="complete", split=split, sample_count=len(rows), model=model,
                    thresholds_fitted=False, test_or_real_used_for_fit=False, ood_used_for_fit=False)
    summary = dict(status="complete", split=split, model=model,
                   selection_on_this_population=False, threshold_fitting_performed=False)
    (root/"protocol.json").write_text(json.dumps(protocol))
    (root/"summary.json").write_text(json.dumps(summary))
    (root/"pair_results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    return root


class PairedChangesTests(unittest.TestCase):
    def test_own_threshold_rescue_loss_layout_cross_and_conservation(self):
        a = [row(1,score=.7),row(2,score=.9),row(3,score=.9),row(4,score=.1),
             row(5,False,.9),row(6,False,.1),row(7,False,.1),row(8,False,.9)]
        b = deepcopy(a)
        for r, score in zip(b,[.6,.4,.6,.1,.1,.9,.1,.9]): r["classification"]["fused"]=score
        b[0]["layouts"]["full_top2_mode"].update(translation_l2_px=21,translation_rc=[2.,1.])
        a[1].pop("layouts")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(m.base.EXPECTED,{"test":(8,4,4)}):
            root=Path(tmp)
            result=m.compare_endpoints(endpoint(root/"a",a,threshold=.8),endpoint(root/"b",b),"test")
        stat=result["populations"]["test3000"]["max_f1"]
        self.assertEqual(result["thresholds"], {"reference":{"max_f1":.8,"recall_95":.8},"new":{"max_f1":.5,"recall_95":.5}})
        self.assertTrue(stat["counts_conserved"])
        self.assertEqual([c["count"] for c in stat["positive_transitions"].values()],[1,1,1,1])
        self.assertEqual([c["count"] for c in stat["negative_transitions"].values()],[1,1,1,1])
        self.assertEqual(stat["positive_transitions"]["FN_to_TP"]["layout20_cross"]["good_to_bad"]["pair_ids"],["1"])
        self.assertEqual(stat["positive_transitions"]["TP_to_FN"]["layout20_cross"]["unknown_to_good"]["count"],1)
        self.assertEqual(stat["positive_transitions"]["TP_to_TP"]["identical_valid_final_layout"]["pair_ids"],["3"])

    def test_invalid_decision_rejects_and_missing_layout_is_not_bad(self):
        a=row(1,score=.6)
        b=deepcopy(a);b["decision_valid"]=False;b["classification"]["fused"]=.99;b.pop("layouts")
        self.assertEqual(m.classification_state(b,.5),"FN")
        self.assertEqual(m.layout_label(b,"test"),"unknown")
        self.assertFalse(m.same_final_layout(a,b,"test"))
        b["layouts"]={"full_top2_mode":{"valid":False}}
        self.assertEqual(m.layout_label(b,"test"),"bad")

    def test_real_populations_keep_all_negatives_and_strict_members(self):
        rows=[row(1),row(2,keep=False),row(3,False,strict=True),row(4,False)]
        expected={"all1016":(4,2,2),"keep803":(3,1,2),"strict547":(3,2,1),"keep_strict334":(2,1,1)}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(m.base.EXPECTED,{"real":(4,2,2)}), patch.dict(m.base.REAL_COUNTS,expected):
            root=Path(tmp)
            result=m.compare_endpoints(endpoint(root/"a",rows,"real"),endpoint(root/"b",list(reversed(rows)),"real"),"real")
        for name, counts in expected.items():
            stat=result["populations"][name]["max_f1"]
            self.assertEqual((stat["sample_count"],stat["positive_count"],stat["negative_count"]),counts)
        self.assertEqual(result["populations"]["keep803"]["max_f1"]["positive_transitions"]["TP_to_TP"]["pair_ids"],["1"])
        self.assertFalse(result["identity_check"]["order_equal"])

    def test_no_intersection_on_missing_identity_or_incomplete_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(m.base.EXPECTED,{"test":(2,1,1)}):
            root=Path(tmp); a=endpoint(root/"a",[row(1),row(2,False)]);b=endpoint(root/"b",[row(9),row(2,False)])
            with self.assertRaisesRegex(ValueError,"no intersection"):
                m.compare_endpoints(a,b,"test")
            (b/"protocol.json").write_text('{"status":"running"}')
            with self.assertRaisesRegex(ValueError,"endpoint unavailable"):
                m.compare_endpoints(a,b,"test")

    def test_ood_reports_only_recall_transitions_and_unknown_layout(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(m.base.EXPECTED,{"ood":(2,2,0)}):
            root=Path(tmp)
            result=m.compare_endpoints(endpoint(root/"a",[row(1,score=.1),row(2,score=.9)],"ood"),
                                       endpoint(root/"b",[row(1,score=.9),row(2,score=.9)],"ood"),"ood")
        stat=result["populations"]["ood301"]["max_f1"]
        self.assertEqual((stat["reference_recall"],stat["new_recall"]),(.5,1.))
        self.assertNotIn("negative_transitions",stat)
        self.assertNotIn("accuracy",stat)
        self.assertEqual(stat["positive_transitions"]["FN_to_TP"]["layout20_cross"]["unknown_to_unknown"]["pair_ids"],["1"])
        self.assertFalse(stat["layout_available"])


if __name__ == "__main__":
    unittest.main()
