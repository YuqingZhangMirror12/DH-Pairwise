"""Small numerical joins plus one synthetic full-membership receipt fixture."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from . import join_hard_oracle as j


def model_row(source, recipe, *, label=True, success=True, valid=True, changed=None, family=False):
    changed = recipe != "clean" if changed is None else changed
    pose = label and not changed
    return dict(pair_id=recipe+"::"+source, source_pair_id=source, recipe=recipe, label=label,
        changed_pair=changed, fallback_reason="rejected" if recipe != "clean" and not changed else None,
        source_family_overlap=family, training_valid=valid,
        pose_supervision_enabled=pose, pose_supervised=pose and valid,
        supervised_correspondence_count=3 if label and valid else 0,
        losses=dict(assignment_nll=3. if valid else 0., match_nll=2. if label and valid else 0.,
            dustbin_nll=1. if valid else 0., translation_smooth_l1=10. if pose and valid else 0.),
        raw_layout_valid=success, raw_translation_l2_px=1. if label and success else None,
        raw_layout20_correct=bool(label and success))


def oracle_row(model, *, success=True, invalid=False):
    return dict({key: model[key] for key in
        ("pair_id", "source_pair_id", "recipe", "changed_pair", "fallback_reason", "source_family_overlap")},
        **dict(oracle_layout_valid=not invalid, oracle_error_px=None if invalid else 1. if success else 25.,
               oracle_layout20_success=bool(success and not invalid)))


def tables():
    model, oracle = [], []
    for i, (ms, os) in enumerate(((True, True), (False, True), (True, False), (False, False))):
        for recipe in ("clean", "wave"):
            row = model_row(str(i), recipe, success=True if recipe == "clean" else ms,
                family=i >= 2)
            model.append(row)
            oracle.append(oracle_row(row, success=True if recipe == "clean" else os,
                invalid=recipe == "wave" and i == 3))
    return model, oracle


def write_full_fixture(root):
    model, oracle = [], []
    for i in range(3000):
        for recipe in ("clean", "wave"):
            row = model_row(str(i), recipe, label=i % 2 == 0)
            model.append(row)
            if row["label"]:
                oracle.append(oracle_row(row))
    md, od = root/"model", root/"oracle"
    md.mkdir(); od.mkdir()
    for directory, name, rows in ((md, "pair_metrics.jsonl", model), (od, "rows.jsonl", oracle)):
        (directory/name).write_text("".join(json.dumps(r)+"\n" for r in rows))
    common = dict(status="complete", manifest_sha256="f"*64)
    ms = dict(common, schema=j.MODEL_SCHEMA, count=6000, model=dict(epoch=12),
        pair_metrics_sha256=j.sha(md/"pair_metrics.jsonl"))
    os = dict(common, schema=j.ORACLE_SCHEMA, metrics=dict(positive_source_count=1500),
        rows_sha256=j.sha(od/"rows.jsonl"))
    (md/"summary.json").write_text(json.dumps(ms))
    (od/"summary.json").write_text(json.dumps(os))
    (md/"protocol.json").write_text(json.dumps(dict(common, schema=j.MODEL_SCHEMA,
        model=dict(epoch=12), completed_count=6000, summary_sha256=j.sha(md/"summary.json"))))
    (od/"protocol.json").write_text(json.dumps(dict(common, schema=j.ORACLE_SCHEMA, completed_count=3000)))
    return md, od


class JoinHardOracleTests(unittest.TestCase):
    def test_four_cells_are_exclusive_and_invalids_remain_failures(self):
        model, oracle = tables()
        joined = j.join_rows(model, oracle)
        result = j.cohort([r for r in joined if r["recipe"] == "wave"])
        self.assertEqual(result["positive_layout_denominator"], 4)
        self.assertEqual(result["four_cells"], dict.fromkeys(j.CELLS, 1))
        self.assertEqual(result["model_invalid_positive_count"], 2)
        self.assertEqual(result["oracle_invalid_positive_count"], 1)

    def test_loss_denominators_do_not_reward_disabled_pose_or_invent_dustbin_counts(self):
        rows = [model_row("a", "clean"), model_row("a", "wave"),
            model_row("b", "wave", valid=False), model_row("negative", "wave", label=False)]
        summary = j.loss_summary(rows)
        self.assertEqual(summary["all_rows"], 4)
        self.assertEqual(summary["training_valid_rows"], 3)
        self.assertEqual(summary["all_row_means"]["assignment_nll"], dict(denominator=4, mean=2.25))
        self.assertEqual(summary["match_supervised_pair_mean"], dict(denominator=2, mean=2.))
        self.assertEqual(summary["supervised_correspondence_tokens"], 6)
        self.assertEqual(summary["pose_supervised_smooth_l1"], dict(denominator=1, mean=10.))
        self.assertEqual(summary["pose_disabled_rows"], 3)
        self.assertIsNone(summary["dustbin_active_token_denominator"])
        self.assertNotIn("total", summary["all_row_means"])

    def test_duplicate_missing_positive_or_metadata_mismatch_is_rejected(self):
        model, oracle = tables()
        invalid_cases = [(model+[model[0]], oracle), (model, oracle+[oracle[0]]), (model, oracle[:-1])]
        changed = deepcopy(oracle); changed[0]["source_pair_id"] = "different"
        invalid_cases.append((model, changed))
        for a, b in invalid_cases:
            with self.subTest(model=len(a), oracle=len(b)), self.assertRaises(ValueError):
                j.join_rows(a, b)
        changed = deepcopy(model); changed[1]["raw_layout20_correct"] = False
        with self.assertRaisesRegex(ValueError, "validity/error/success"):
            j.join_rows(changed, oracle)

    def test_same_source_transitions_and_optional_family_strata_preserve_fallback(self):
        model, oracle = tables()
        for recipe in ("clean", "local"):
            r = model_row("fallback", recipe, changed=False)
            model.append(r); oracle.append(oracle_row(r))
        result = j.summarize(j.join_rows(model, oracle))
        all_pairs = result["paired_all"]
        self.assertEqual(all_pairs["source_count"], 5)
        self.assertEqual(all_pairs["model_clean_to_variant"]["success_to_failure"], 2)
        self.assertEqual(all_pairs["oracle_clean_to_variant"]["success_to_failure"], 2)
        self.assertEqual(all_pairs["paired_loss_delta_variant_minus_clean"]["translation_smooth_l1"]["denominator"], 1)
        self.assertEqual(result["by_requested_recipe"]["local"]["requested_variant"]["fallback_reasons"], {"rejected": 1})
        self.assertEqual(result["optional_source_family_disjoint"]["metrics"]["paired_all"]["source_count"], 3)
        self.assertIn("local|changed=false", result["by_recipe_and_actual_change"])

    def test_full_completion_manifest_hash_and_rows_hash_are_required(self):
        with TemporaryDirectory() as temporary:
            md, od = write_full_fixture(Path(temporary))
            joined, sources = j.load_sources(md, od)
            self.assertEqual(len(joined), 6000)
            self.assertEqual(sum(r["oracle"] is not None for r in joined), 3000)
            self.assertEqual(sources["model"]["epoch"], 12)
            protocol = j.read(od/"protocol.json")
            for update in (dict(status="pilot_complete"), dict(manifest_sha256="e"*64)):
                (od/"protocol.json").write_text(json.dumps(dict(protocol, **update)))
                with self.assertRaises(ValueError):
                    j.load_sources(md, od)
            (od/"protocol.json").write_text(json.dumps(protocol))
            with (od/"rows.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "hashes differ"):
                j.load_sources(md, od)


if __name__ == "__main__":
    unittest.main()
