"""Synthetic CPU-only hard-VAL contract, pose and paired-summary tests."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence import evaluate_hard_validation as hard
from experiments.rachel_n512_formal_30k.test_decoupled_samplewise_loss import fixture, legacy_single, sliced
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig

torch.set_num_threads(1)


def manifest():
    sources = [dict(pair_id="source-"+str(i), label=i % 2 == 0) for i in range(4)]
    entries = []
    for i, source in enumerate(sources):
        for recipe in ("clean", "wave" if i < 2 else "local"):
            entries.append(dict(pair_id=recipe+"::"+source["pair_id"], source_pair_id=source["pair_id"],
                label=source["label"], recipe=recipe, changed_pair=recipe != "clean",
                artifact_path=str(len(entries))+".npz"))
    return dict(schema=hard.INPUT_SCHEMA, split="val", status="pilot_complete", artifact_root="unused",
        protocol=dict(source_val_manifest_sha256=hard.original.VAL_HASH, seed=99, diagnostic_only=True),
        entries=entries), sources


def metrics_row(entry, correct=True):
    positive = bool(entry["label"])
    return dict(entry, training_valid=True, decision_valid=False,
        losses={key: 1. for key in hard.original.TERM_NAMES},
        supervised_correspondence_count=4 if positive else 0,
        pose_supervised=positive and not entry["changed_pair"],
        differentiable_translation_l2_px=2. if positive and not entry["changed_pair"] else None,
        raw_layout_valid=correct, raw_translation_l2_px=2. if positive and correct else None,
        raw_layout20_correct=positive and correct, fallback_reason=None)


class ManifestTests(unittest.TestCase):
    def test_flexible_count_balanced_recipe_and_one_recipe_per_source(self):
        record, source = manifest()
        counts = hard.validate_manifest(record, source, allow_pilot=True)
        self.assertEqual(counts["clean"]["count"], 4)
        self.assertEqual(counts["wave"]["count"], 2)
        self.assertEqual(sum(c["count"] for c in counts.values()), 8)

    def test_identity_balance_clean_binding_and_pilot(self):
        record, source = manifest()
        for mutate in (
            lambda r: r["entries"][0].update(pair_id=r["entries"][0]["source_pair_id"]),
            lambda r: r["entries"][0].update(label=False),
            lambda r: r["entries"].pop(0),
            lambda r: r["entries"].pop(1),
            lambda r: r["entries"].append(dict(r["entries"][0])),
            lambda r: r["entries"][0].update(changed_pair=True),
        ):
            altered = deepcopy(record); mutate(altered)
            with self.assertRaises(ValueError): hard.validate_manifest(altered, source, allow_pilot=True)
        record["status"] = "pilot_complete"
        with self.assertRaises(ValueError): hard.validate_manifest(record, source)
        self.assertEqual(len(hard.validate_manifest(record, source, allow_pilot=True)), 3)

    def test_complete_requires_full6000_exact_source_coverage_and_recipe_counts(self):
        record, source = manifest()
        record["status"] = "complete"
        with self.assertRaisesRegex(ValueError, "formal complete"):
            hard.validate_manifest(record, source, allow_pilot=True)
        source = [dict(pair_id=str(i), label=i % 2 == 0) for i in range(3000)]
        record["entries"] = [dict(pair_id=recipe+"::"+s["pair_id"], source_pair_id=s["pair_id"],
            label=s["label"], recipe=recipe, changed_pair=recipe != "clean", artifact_path="unused")
            for i, s in enumerate(source) for recipe in ("clean", hard.RECIPES[1+(i//2)%4])]
        counts = hard.validate_manifest(record, source)
        self.assertEqual(counts["clean"]["count"], 3000)
        self.assertTrue(all(counts[r]["count"] == 750 for r in hard.RECIPES[1:]))
        record["entries"] = record["entries"][:-4]
        with self.assertRaisesRegex(ValueError, "formal complete"):
            hard.validate_manifest(record, source)

    def test_preflight_no_weights_and_new_output_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"artifacts").mkdir(); (root/"source").mkdir(); (root/"weights").mkdir()
            record, source = manifest()
            record["artifact_root"] = str(root/"artifacts")
            for row in record["entries"]: (root/"artifacts"/row["artifact_path"]).touch()
            source_path = root/"source/val.jsonl"
            source_path.write_text("\n".join(json.dumps(r) for r in source))
            digest = hard.original.sha256(source_path)
            record["protocol"]["source_val_manifest_sha256"] = digest
            path = root/"artifacts/manifest.json"; path.write_text(json.dumps(record))
            checkpoint = root/"weights/M12.pt"; checkpoint.write_bytes(b"not weights")
            args = SimpleNamespace(manifest=path, source_val_manifest=source_path, checkpoint=checkpoint,
                output=root/"out", epoch=12, workers=2, execute=False, allow_pilot=True)
            with patch.object(hard.original, "VAL_HASH", digest), \
                 patch.object(hard.original, "validate_manifest"), \
                 patch.object(torch, "load", side_effect=AssertionError("must not load")):
                plan = hard.preflight(args)
                self.assertEqual(plan["count"], 8)
                self.assertFalse(args.output.exists())
                args.output.mkdir()
                with self.assertRaises(FileExistsError): hard.preflight(args)


class NumericalTests(unittest.TestCase):
    def test_exact_canonical_pose_masked_loss(self):
        output, targets, pose, _ = fixture(8)
        pose = pose.clone(); pose[1] = False
        config = RachelN512LossConfig()
        values, counts, flags = hard.measure_losses(output, targets, config, pose)
        for i in range(8):
            total, pieces = legacy_single(sliced(output, i), tuple(t[i:i+1] for t in targets), pose[i:i+1], "matcher")
            self.assertAlmostEqual(values["total"][i], total.item(), places=12)
            self.assertAlmostEqual(values["translation_smooth_l1"][i], pieces["translation_smooth_l1"].item(), places=12)
        self.assertEqual(values["translation_smooth_l1"][1], 0.)
        self.assertFalse(flags[1]); self.assertGreater(counts[1], 0)
        self.assertTrue(targets[4][1])  # GT validity must remain untouched.

    def test_private_evaluator_preserves_decoder_forward_and_clean_globals(self):
        output, targets, _, _ = fixture(2)
        output.decision_valid = output.training_valid.clone()
        points = np.tile(np.array([[0,0], [10,0], [10,10], [0,10]], np.float32), (2,1,1))
        batch = SimpleNamespace(pair_ids=("a", "b"),
            mask_a=np.zeros((2,1,8,8), np.float32), mask_b=np.zeros((2,1,8,8), np.float32),
            points_rc_a=points, points_rc_b=points+np.array([2,3], np.float32),
            contour_valid_a=np.ones((2,4), bool), contour_valid_b=np.ones((2,4), bool),
            labels=targets[0].numpy(), target_a=targets[1].numpy(), target_b=targets[2].numpy(),
            translation_a_to_b_rc=targets[3].numpy(), translation_valid=targets[4].numpy())
        calls = []
        def model(*inputs):
            self.assertFalse(torch.is_grad_enabled()); calls.append(len(inputs)); return output
        before = hard.original.measure_losses
        clean = hard.original.evaluate_batch(model, batch, RachelN512LossConfig(), torch.device("cpu"))
        damaged = hard.evaluate_batch(model, batch, RachelN512LossConfig(), torch.device("cpu"), [False, False])
        self.assertIs(before, hard.original.measure_losses)
        self.assertEqual(calls, [6, 6])
        self.assertTrue(clean[1]["pose_supervised"])
        self.assertFalse(damaged[1]["pose_supervised"])
        self.assertIsNone(damaged[1]["differentiable_translation_l2_px"])
        for a, b in zip(clean, damaged):
            for key in ("raw_layout_valid", "raw_layout20_correct", "raw_translation_l2_px", "supervised_correspondence_count"):
                self.assertEqual(a[key], b[key])
            self.assertEqual(a["losses"]["assignment_nll"], b["losses"]["assignment_nll"])

    def test_archive_sidecar_binding_round_trip(self):
        from experiments.rachel_n512_formal_30k.test_train_score_decoupled import tiny_loader
        from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import save_sample
        sample, report = tiny_loader(2, balanced=True).dataset[0]
        sample = replace(sample, pair_id="wave::fixture")
        report = dict(report, changed_pair=True, pose_supervision_enabled=False,
                      source_pair_id="fixture", hard_val_recipe="wave")
        entry = dict(pair_id=sample.pair_id, source_pair_id="fixture", recipe="wave", label=True,
                     changed_pair=True, artifact_path="one.npz")
        with tempfile.TemporaryDirectory() as directory:
            save_sample(Path(directory)/"one.npz", sample, report)
            restored, sidecar, _ = hard.load_entry(directory, entry)
            self.assertTrue(restored.translation_valid)
            self.assertFalse(sidecar["pose_supervision_enabled"])
            with self.assertRaises(ValueError): hard.load_entry(directory, dict(entry, changed_pair=False))


class SummaryTests(unittest.TestCase):
    def test_pilot_execute_preserves_rows_and_never_reports_formal_complete(self):
        record, _ = manifest(); record["status"] = "pilot_complete"
        rows = [metrics_row(e) for e in record["entries"]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/"manifest.json"
            path.write_text(json.dumps(record))
            plan = dict(manifest=str(path), manifest_sha256=hard.original.sha256(path),
                workers=1, checkpoint="unused", epoch=12, artifact_root="unused", output=str(root/"out"),
                pair_ids=[r["pair_id"] for r in rows], source_status="pilot_complete")
            chunk = dict(rows=rows, timings=[], model=dict(epoch=12, fixture_only=True))
            with patch.dict(hard.os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                 patch.object(hard, "evaluate_chunk", return_value=chunk) as evaluate:
                hard.execute(SimpleNamespace(), plan)
            self.assertEqual(evaluate.call_count, 1)
            summary = json.loads((root/"out/summary.json").read_text())
            protocol = json.loads((root/"out/protocol.json").read_text())
            self.assertEqual(summary["status"], "pilot_complete")
            self.assertEqual(protocol["status"], "pilot_complete")
            self.assertFalse(summary["training_eligible"])
            saved = [json.loads(s) for s in (root/"out/pair_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(saved, rows)

    def test_source_paired_not_all_clean_and_optional_family_stratum(self):
        record, _ = manifest()
        rows = [metrics_row(e, not(e["recipe"] == "wave" and e["label"])) for e in record["entries"]]
        for row in rows: row["source_family_overlap"] = row["source_pair_id"] == "source-0"
        result = hard.summarize(rows)
        wave = result["paired_clean"]["wave"]
        self.assertEqual(wave["matched_clean"]["all_pairs"]["count"], 2)
        self.assertEqual(wave["positive_raw_layout20_transitions"], {"correct_to_failed": 1})
        self.assertEqual(result["all_entries"]["all_pairs"]["count"], 8)
        self.assertEqual(result["optional_source_family_disjoint"]["count"], 6)
        self.assertEqual(result["by_recipe"]["wave"]["changed_count"], 2)
        self.assertEqual(result["all_entries"]["raw_layout20"]["positive_count"], 4)

    def test_cpu_explicit_execution_and_cli_no_selection(self):
        args = hard.parser().parse_args(["--manifest", "m", "--checkpoint", "c", "--epoch", "12", "--output", "o"])
        self.assertFalse(args.execute); self.assertFalse(args.allow_pilot)
        names = {a.dest for a in hard.parser()._actions}
        self.assertFalse({"split", "threshold", "selection", "train", "gpu", "limit"} & names)
        with patch.dict(hard.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(RuntimeError, "CPU-only"):
                hard.execute(args, {})


if __name__ == "__main__":
    unittest.main()
