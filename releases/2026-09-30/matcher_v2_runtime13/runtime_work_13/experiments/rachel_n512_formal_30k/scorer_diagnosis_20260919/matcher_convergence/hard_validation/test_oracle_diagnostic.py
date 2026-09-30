"""Tiny GT geometry and paired hard-VAL driver tests; no model or real data."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import save_sample
from . import oracle_diagnostic as d
from .test_materialize import sample


def geometry_sample(positive=True, count=5, band=0.):
    old = sample(positive)
    points = np.asarray([[10., 2.], [10., 12.], [10., 22.], [10., 32.], [10., 42.]])[:count]
    valid = np.ones(count, bool)
    target = np.arange(count, dtype=np.int64) if positive else np.full(count, -1, np.int64)
    gt = np.asarray([7., -3.]) if positive else np.full(2, np.nan)
    return replace(old, points_rc_a=points, points_rc_b=points + [7.+band, -3.],
        contour_valid_a=valid, contour_valid_b=valid.copy(), target_a=target,
        target_b=target.copy(), translation_a_to_b_rc=gt,
        translation_a_to_b_xy_cartesian=np.asarray([-3., -7.]) if positive else gt.copy())


def row(source, recipe, *, band=0., count=5, changed=False, fallback=None):
    s = geometry_sample(count=count, band=band)
    evidence = d.geometry.oracle_evidence(s.points_rc_a, s.points_rc_b,
        s.contour_valid_a, s.contour_valid_b, s.target_a, s.target_b, s.translation_a_to_b_rc)
    return dict(pair_id=recipe+"::"+source, source_pair_id=source, recipe=recipe,
        changed_pair=changed, actual_changed_pair=changed, fallback_reason=fallback, **evidence)


def fixture(root):
    artifacts = root/"artifacts"
    artifacts.mkdir()
    source_path = root/"source/pairs/val.jsonl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("fixture source; SHA reader is mocked\n")
    source, entries = [], []
    for i in range(4):
        positive = i % 2 == 0
        source_id = "source-%d" % i
        source.append(dict(pair_id=source_id, label=int(positive), split="val"))
        assigned = "wave" if i < 2 else "local"
        original = replace(geometry_sample(positive), pair_id=source_id)
        for recipe in ("clean", assigned):
            changed = recipe == "wave"
            new = original
            if changed:
                mask = original.mask_a.copy(); mask[0, 1, 1] = 0
                new = replace(original, mask_a=mask, points_rc_b=original.points_rc_b+[25., 0.])
            fallback = "coupled_rejection" if recipe == "local" else None
            report = d.materialize.changed_report(original, new, recipe, fallback_reason=fallback)
            pair_id = "hardval-"+recipe+"::"+source_id
            new = replace(new, pair_id=pair_id)
            report.update(pair_id=pair_id, source_pair_id=source_id, hard_val_recipe=recipe,
                          source_split="val", diagnostic_only=True)
            relative = "%d-%s.npz" % (i, recipe)
            save_sample(artifacts/relative, new, report)
            entries.append(dict(pair_id=pair_id, source_pair_id=source_id, label=positive,
                recipe=recipe, assigned_recipe=assigned, changed_pair=changed,
                fallback_reason=fallback, pose_supervision_enabled=positive and not changed,
                source_family_overlap=False, artifact_path=relative))
    record = dict(schema=d.hard.INPUT_SCHEMA, split="val", status="pilot_complete",
        artifact_root=str(artifacts), entries=entries,
        protocol=dict(source_val_manifest=str(source_path), source_val_manifest_sha256=d.hard.original.VAL_HASH,
            seed=17, diagnostic_only=True, training_eligible=False, threshold_fitting=False,
            model_selection=False, real_ood_used=False))
    manifest = artifacts/"manifest.json"
    manifest.write_text(json.dumps(record))
    args = d.parser().parse_args(["--manifest", str(manifest), "--output", str(root/"result"), "--allow-pilot"])
    return args, record, source


class OracleDiagnosticTests(unittest.TestCase):
    def test_uses_exact_production_oracle_and_missing_material_is_not_upper_bound(self):
        from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_support_diagnosis import oracle_geometry
        self.assertIs(d.geometry.oracle_evidence, oracle_geometry.oracle_evidence)
        clean, damaged = row("a", "clean"), row("a", "wave", band=25., changed=True)
        self.assertTrue(clean["oracle_layout20_success"])
        self.assertFalse(damaged["oracle_layout20_success"])
        self.assertEqual(damaged["gt_edge_count"], 5)
        self.assertEqual(damaged["oracle_residual_px"], 0.)
        self.assertEqual(damaged["oracle_error_px"], 25.)

    def test_pairwise_recipe_changed_and_fallback_strata_keep_all_sources(self):
        rows = [row("a", "clean"), row("a", "wave", band=25., changed=True),
            row("b", "clean"), row("b", "wave", fallback="rejected"),
            row("c", "clean"), row("c", "local", count=2, changed=True)]
        result = d.summarize(rows)
        self.assertEqual(result["positive_source_count"], 3)
        self.assertEqual(result["evaluated_positive_rows"], 6)
        paired = result["paired_all"]
        self.assertEqual(paired["clean"]["oracle_layout20_rate"], 1.)
        self.assertEqual(paired["requested_variant"]["positive_denominator"], 3)
        self.assertEqual(paired["requested_variant"]["oracle_layout20_rate"], 1/3)
        self.assertEqual(paired["requested_variant"]["fewer_than_3_target_edges_count"], 1)
        self.assertEqual(paired["layout20_transitions"]["success_to_failure"], 2)
        self.assertEqual(paired["error_delta_px_when_both_valid"]["available_count"], 2)
        fallback = result["by_recipe_and_actual_change"]["wave|actual_changed=false"]
        self.assertEqual(fallback["positive_source_count"], 1)
        self.assertEqual(fallback["requested_variant"]["fallback_reasons"], {"rejected": 1})
        with self.assertRaises(ValueError):
            d.summarize(rows[1:])

    def test_pilot_opt_in_and_full_population_are_enforced(self):
        with TemporaryDirectory() as temporary:
            args, record, sources = fixture(Path(temporary))
            with patch.object(d.materialize, "source_rows", return_value=sources):
                args.allow_pilot = False
                with self.assertRaises(ValueError):
                    d.preflight(args)
                args.allow_pilot = True
                self.assertEqual(len(d.preflight(args)[-1]), 4)
                record["status"] = "complete"
                Path(args.manifest).write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "formal complete"):
                    d.preflight(args)

    def test_tiny_pilot_reads_only_positive_artifacts_and_preserves_fallback_rows(self):
        with TemporaryDirectory() as temporary:
            args, record, sources = fixture(Path(temporary))
            with patch.object(d.materialize, "source_rows", return_value=sources), \
                    patch.object(d.hard, "load_entry", wraps=d.hard.load_entry) as load:
                result = d.run(args)
            self.assertEqual(load.call_count, 4)
            self.assertTrue(all(call.args[1]["label"] for call in load.call_args_list))
            rows = [json.loads(line) for line in (Path(args.output)/"rows.jsonl").read_text().splitlines()]
            protocol = json.loads((Path(args.output)/"protocol.json").read_text())
        self.assertEqual(result["status"], "pilot_complete")
        self.assertEqual(result["metrics"]["positive_source_count"], 2)
        self.assertEqual(result["metrics"]["paired_all"]["requested_variant"]["oracle_layout20_rate"], .5)
        self.assertEqual({r["pair_id"] for r in rows}, {e["pair_id"] for e in record["entries"] if e["label"]})
        self.assertEqual([r["fallback_reason"] for r in rows if r["recipe"] == "local"], ["coupled_rejection"])
        self.assertTrue(all("changed_pair" in r and "source_family_overlap" in r for r in rows))
        self.assertFalse(protocol["model_forward"])
        self.assertFalse(protocol["GPU_computation"])
        self.assertFalse(protocol["performance_upper_bound_claimed"])

    def test_paired_gt_drift_fails_without_publishing_complete_summary(self):
        with TemporaryDirectory() as temporary:
            args, _, sources = fixture(Path(temporary))
            original_load = d.hard.load_entry
            def drift(root, entry):
                s, report, path = original_load(root, entry)
                if entry["recipe"] == "wave":
                    s = replace(s, translation_a_to_b_rc=s.translation_a_to_b_rc+[1., 0.])
                return s, report, path
            with patch.object(d.materialize, "source_rows", return_value=sources), \
                    patch.object(d.hard, "load_entry", side_effect=drift), \
                    self.assertRaisesRegex(ValueError, "original-frame GT differs"):
                d.run(args)
            self.assertFalse((Path(args.output)/"summary.json").exists())
            self.assertEqual(json.loads((Path(args.output)/"protocol.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
