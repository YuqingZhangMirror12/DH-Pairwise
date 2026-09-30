from copy import deepcopy
import math
import unittest

from . import summarize_hard_scorers as summary


def score(logit):
    return dict(logit=logit,probability=1/(1+math.exp(-logit)))


def fixtures():
    thresholds={head:dict(max_f1=.7,recall_95=.6,recall_99=.5) for head in summary.HEADS}
    rows=[]
    # Two paired sources per recipe, one positive and one negative each.
    for index, (recipe,label) in enumerate((("wave",True),("wave",False),("local",True),("local",False))):
        for view in ("clean","damaged"):
            changed=view=="damaged" and index<2
            clean_logit=2. if label else -2.
            c8_logit=(-1. if label else 1.) if changed else clean_logit
            if view=="clean": c8_logit=clean_logit
            c16_logit=c8_logit+(.4 if label else -.4)
            rows.append(dict(pair_id=f"{index}-{view}",source_pair_id=str(index),recipe="clean" if view=="clean" else recipe,
                assigned_recipe=recipe,label=label,changed_pair=changed,source_family_overlap=index==0,
                fallback_reason="coupled_rejection" if view=="damaged" and not changed else None,
                training_valid=True,decision_valid=True,raw_layout_valid=True,
                raw_translation_l2_px=(22. if changed else 5.) if label else None,
                raw_layout20_correct=bool(label and not changed),
                scores={head:score(c8_logit if head.endswith("c8") else c16_logit) for head in summary.HEADS}))
    return rows,thresholds


class HardSummaryTests(unittest.TestCase):
    def test_pair_counts_changed_fallback_and_transitions(self):
        rows,thresholds=fixtures()
        result=summary.summarize(rows,thresholds,expected_sources=4)
        self.assertEqual((result["source_count"],result["view_count"]),(4,8))
        self.assertEqual(len(result["sources"]),4)
        head=result["heads"]["matched_edges_c8"]
        self.assertEqual(head["populations"]["clean"]["n"],4)
        self.assertEqual(head["populations"]["actually_changed"]["n"],2)
        self.assertEqual(head["populations"]["unchanged_fallback"]["n"],2)
        transition=head["by_recipe"]["wave"]["paired_full_requested"]["operating_points"]["max_f1"]
        self.assertEqual(transition["positive_acceptance"]["accepted_to_rejected"],1)
        self.assertEqual(transition["negative_acceptance"]["rejected_to_accepted"],1)
        self.assertEqual(transition["positive_correct_layout_accepted"]["accepted_to_rejected"],1)
        fallback=head["by_recipe"]["local"]
        self.assertEqual(fallback["actually_changed"]["n"],0)
        self.assertIsNone(fallback["actually_changed"]["mean_pair_bce"])
        self.assertEqual(fallback["fallback_reasons"],{"coupled_rejection":2})
        self.assertEqual(head["source_family"]["overlap"]["clean"]["n"],1)
        self.assertEqual(head["source_family"]["disjoint"]["clean"]["n"],3)

    def test_empty_and_single_class_metrics_are_not_fabricated(self):
        rows,thresholds=fixtures()
        empty=summary.metric([],"all_tokens_c8",thresholds["all_tokens_c8"])
        self.assertEqual(empty["n"],0)
        for key in ("auroc","average_precision","mean_pair_bce","mean_logit"):
            self.assertIsNone(empty[key])
        self.assertIsNone(empty["operating_points"]["max_f1"]["accuracy"])
        positive=summary.metric([r for r in rows if r["label"]],"all_tokens_c8",thresholds["all_tokens_c8"])
        self.assertIsNone(positive["auroc"])
        self.assertIsNone(positive["operating_points"]["max_f1"]["f1"])
        self.assertIsNotNone(positive["operating_points"]["max_f1"]["recall"])

    def test_invalid_decision_rejects_and_masked_loss_denominator_is_all(self):
        rows,thresholds=fixtures()
        selected=[deepcopy(rows[0]),deepcopy(rows[2])]
        selected[0]["decision_valid"]=False
        selected[0]["training_valid"]=False
        result=summary.metric(selected,"all_tokens_c8",thresholds["all_tokens_c8"])
        self.assertEqual(result["operating_points"]["max_f1"]["tp"],0)
        self.assertEqual(result["operating_points"]["max_f1"]["correct_layout_rejected"],1)
        self.assertAlmostEqual(result["mean_training_pair_bce_all_rows"],summary.bce(-2,False)/2)
        self.assertEqual(result["auroc"],0.)

    def test_grouped_tie_ranking_and_bce_numerical_stability(self):
        rows,thresholds=fixtures()
        selected=[deepcopy(rows[0]),deepcopy(rows[2])]
        for row in selected: row["scores"]["all_tokens_c8"]=score(0.)
        result=summary.metric(selected,"all_tokens_c8",thresholds["all_tokens_c8"])
        self.assertEqual(result["auroc"],.5)
        self.assertEqual(result["average_precision"],.5)
        self.assertAlmostEqual(result["mean_pair_bce"],math.log(2))
        self.assertTrue(math.isfinite(summary.bce(10000,False)))
        self.assertTrue(math.isfinite(summary.bce(-10000,True)))

    def test_budget_comparison_keeps_own_thresholds(self):
        rows,thresholds=fixtures()
        thresholds["matched_tokens_c16"]["max_f1"]=.95
        result=summary.summarize(rows,thresholds,expected_sources=4)
        paired=result["c8_to_c16"]["matched_tokens"]["clean"]["operating_points"]["max_f1"]
        self.assertEqual((paired["left_threshold"],paired["right_threshold"]),(.7,.95))
        self.assertEqual(paired["positive_acceptance"]["accepted_to_rejected"],2)
        self.assertGreater(result["c8_to_c16"]["matched_tokens"]["clean"]["mean_positive_logit_delta"],0)

    def test_rejects_duplicate_missing_or_inconsistent_pair_sources(self):
        rows,thresholds=fixtures()
        mutations=[]
        duplicate=deepcopy(rows); duplicate[1]=deepcopy(duplicate[0]);mutations.append(duplicate)
        labels=deepcopy(rows);labels[1]["label"]=False;mutations.append(labels)
        overlap=deepcopy(rows);overlap[1]["source_family_overlap"]=False;mutations.append(overlap)
        clean_changed=deepcopy(rows);clean_changed[0]["changed_pair"]=True;mutations.append(clean_changed)
        for bad in mutations:
            with self.assertRaises(ValueError): summary.summarize(bad,thresholds,expected_sources=4)
        with self.assertRaises(ValueError): summary.summarize(rows[:-1],thresholds,expected_sources=4)
        with self.assertRaises(ValueError): summary.summarize(rows,thresholds)


if __name__=="__main__":
    unittest.main()
