import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from . import summarize_priority as s


def fixtures(split):
    count, positive = s.COUNTS[split]
    rows = []
    for i in range(count):
        row = dict(pair_id=split+str(i), label=i < positive, decision_valid=i != 0,
            classification=dict(fused=.8 if i % 2 == 0 else .2),
            layouts={s.DECODER:dict(valid=True, translation_l2_px=20. if i % 3 == 0 else 21.)})
        if split == "real":
            row["review_status"] = "keep" if i < 295 else "exclude" if i < positive else "not_reviewed_negative"
        if split == "ood":
            row.update(layout_gt_available=False, target_translation_rc=None)
            row["layouts"][s.DECODER]["translation_l2_px"] = None
        rows.append(row)
    selected = [r for r in rows if split != "real" or not r["label"] or r["review_status"] == "keep"]
    group = dict(sample_count=len(selected), positive_count=sum(r["label"] for r in selected),
        negative_count=sum(not r["label"] for r in selected),
        decision_valid_count=sum(r["decision_valid"] for r in selected), classification=dict(fused={}))
    if split != "ood":
        group["layout"] = {}
    for op, threshold in (("max_f1", .5), ("recall_95", .1)):
        accepted = {r["pair_id"] for r in selected if r["decision_valid"] and r["classification"]["fused"] >= threshold}
        tp = sum(r["label"] and r["pair_id"] in accepted for r in selected)
        fn = group["positive_count"]-tp
        if split == "ood":
            metrics = dict(threshold=threshold, accepted_positive_count=tp, false_negative_count=fn,
                positive_recall=tp/group["positive_count"])
        else:
            fp = sum(not r["label"] and r["pair_id"] in accepted for r in selected)
            tn = group["negative_count"]-fp
            metrics = dict(threshold=threshold, tp=tp, fn=fn, fp=fp, tn=tn,
                accuracy=(tp+tn)/len(selected), precision=tp/(tp+fp), recall=tp/group["positive_count"],
                f1=2*tp/(2*tp+fp+fn), auprc=.7)
            good = [r for r in selected if r["label"] and r["layouts"][s.DECODER]["translation_l2_px"] <= 20]
            good_ids = {r["pair_id"] for r in good}
            accepted_correct = len(good_ids & accepted)
            group["layout"][op] = {"20":dict(positive_count=group["positive_count"], tolerance_px=20,
                raw_correct=len(good), raw_recall=len(good)/group["positive_count"],
                accepted_correct=accepted_correct, classification_FN_but_layout_correct=len(good)-accepted_correct,
                accepted_positive_bad_layout=tp-accepted_correct,
                end_to_end_positive_recall=accepted_correct/group["positive_count"], accepted_negative_count=fp)}
        group["classification"]["fused"][op] = metrics
    return rows, group


def write_endpoint(root, arm, budget, split, rows=None):
    folder = "adaptation_training" if arm in ("G0", "G1") else "training"
    path = root/folder/arm/"evaluation"/("c%d"%budget)/split
    path.mkdir(parents=True)
    base, group = fixtures(split)
    rows = base if rows is None else rows
    identity = dict(selection="fixed_epoch", head_budget=budget, training_identity=dict(arm=arm),
        operating_points=dict(thresholds=dict(max_f1=.5, recall_95=.1)))
    s.save(path/"protocol.json", dict(status="complete", split=split, sample_count=len(rows)))
    s.save(path/"summary.json", dict(status="complete", split=split, model=identity,
        selection_on_this_population=False, threshold_fitting_performed=False, groups={s.COHORTS[split]:group}))
    (path/"pair_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    return path


class SummaryTests(unittest.TestCase):
    def test_complete42_and_adaptation_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in s.ARMS:
                for budget in s.BUDGETS:
                    for split in s.SPLITS:
                        write_endpoint(root, arm, budget, split)
            report = s.collect(root)
            self.assertEqual((report["status"], report["completed_endpoints"]), ("complete", 42))
            self.assertEqual(report["populations"]["real"]["selected_count"], 803)
            self.assertEqual(len(report["populations"]["real"]["excluded_positive_pair_ids"]), 213)
            self.assertTrue(any("adaptation_training/G1" in r["source"] for r in report["endpoints"]))
            self.assertIn("Layout≤20px", s.markdown(report))
            self.assertIn("No negative pairs or layout GT", s.markdown(report))

    def test_boundary_and_invalid_decision_are_rejected(self):
        with TemporaryDirectory() as tmp:
            path = write_endpoint(Path(tmp), "matched_edges", 8, "test")
            row, _ = s.endpoint(path, "matched_edges", 8, "test")
            metric = row["operating_points"]["max_f1"]
            self.assertIn("test0", metric["false_negative_pair_ids"])
            self.assertIn("test0", metric["layout_le20"]["classification_rejected_correct_pair_ids"])
            self.assertEqual(metric["layout_le20"]["raw_correct"], 500)
            self.assertEqual(metric["layout_le20"]["classification_FN_but_layout_correct"], 251)

    def test_ood_has_no_binary_or_pose_metrics(self):
        with TemporaryDirectory() as tmp:
            path = write_endpoint(Path(tmp), "G0", 16, "ood")
            row, _ = s.endpoint(path, "G0", 16, "ood")
            for metrics in row["operating_points"].values():
                self.assertEqual(set(metrics), {"threshold", "recall", "tp", "fn", "false_negative_pair_ids"})
                self.assertEqual(metrics["tp"]+metrics["fn"], 301)

    def test_missing_is_pending_and_requires_explicit_partial(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_endpoint(root, "all_tokens", 8, "ood")
            args = SimpleNamespace(root=root, output=root/"report", allow_partial=False)
            with self.assertRaisesRegex(RuntimeError, "partial report saved"):
                s.run(args)
            report = s.read(root/"report/summary.json")
            self.assertEqual((report["status"], report["completed_endpoints"], report["pending_endpoints"]), ("partial", 1, 41))
            self.assertNotIn("operating_points", report["endpoints"][0])
            args.output, args.allow_partial = root/"allowed", True
            self.assertEqual(s.run(args)["status"], "partial")

    def test_duplicate_ids_and_wrong_cohort_rejected(self):
        for split, mutate, message in (("test", lambda r:r[1].update(pair_id=r[0]["pair_id"]), "duplicate"),
                ("real", lambda r:r[295].update(review_status="keep"), "reviewed295")):
            with self.subTest(split=split), TemporaryDirectory() as tmp:
                rows, _ = fixtures(split); mutate(rows)
                path = write_endpoint(Path(tmp), "all_tokens", 16, split, rows)
                with self.assertRaisesRegex(ValueError, message):
                    s.endpoint(path, "all_tokens", 16, split)

    def test_stale_metrics_and_endpoint_budget_rejected(self):
        with TemporaryDirectory() as tmp:
            path = write_endpoint(Path(tmp), "all_tokens", 16, "test")
            summary = s.read(path/"summary.json")
            summary["groups"]["all"]["classification"]["fused"]["max_f1"]["fn"] += 1
            s.save(path/"summary.json", summary)
            with self.assertRaisesRegex(ValueError, "differs from pair rows: fn"):
                s.endpoint(path, "all_tokens", 16, "test")
            with self.assertRaisesRegex(ValueError, "registered fixed-budget"):
                s.endpoint(path, "all_tokens", 8, "test")

    def test_equal_count_changed_ids_across_arms_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_endpoint(root, "all_tokens", 16, "ood")
            rows, _ = fixtures("ood"); rows[5]["pair_id"] += "changed"
            write_endpoint(root, "matched_tokens", 16, "ood", rows)
            with self.assertRaisesRegex(ValueError, "IDs/labels/review cohort changed"):
                s.collect(root)


if __name__ == "__main__":
    unittest.main()
