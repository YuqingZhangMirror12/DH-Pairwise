"""Synthetic metadata fixtures only: no GPU work, model build or launch."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.rachel_n512_formal_30k import prepare_score_input_queue as queue
from experiments.rachel_n512_formal_30k import train_score_input_variant, evaluate_score_input_variant
from experiments.rachel_n512_formal_30k.score_design_input_variants import InputVariantSpec, full24_reference_config
from experiments.rachel_n512_formal_30k.score_design_stages import stage_protocol
from staging.pairwise_v0_2.models.rachel_candidate_score import CandidateScoreConfig
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue.canonical(data)))


def eval_fixture(output, split, model, keep_sha, *, s3=False):
    dump(output / "protocol.json", dict(schema_version="rachel-score-staged-evaluation/1" if s3 else "rachel-score-design-evaluation/1",
        status="complete", split=split, sample_count=queue.COUNTS[split], model=model,
        decoder=queue.core.fixed.DECODER_NAME, decoder_config=asdict(queue.core.fixed.TOP2_CONFIG),
        thresholds_fitted=False, test_or_real_used_for_fit=False, ood_used_for_fit=False,
        keep_ids_sha256=keep_sha if split == "real" else None))
    dump(output / "summary.json", dict(status="complete", ignored_metric=-999))
    # The preparer only checks presence; it must not read/score held-out rows.
    (output / "pair_results.jsonl").write_text("fixture opaque predictions are not parsed\n")


def make_fixture(root, architecture="candidate_pair", budget=10, s3_arch=None):
    s3_arch = s3_arch or (architecture if architecture != "original" else "candidate_pair")
    (root / "input_source").mkdir()
    dump(root / "keep_ids.json", dict(kept_positive_pair_ids=["keep_%d" % i for i in range(295)]))
    keep_sha = queue._sha256(root / "keep_ids.json")
    dataset, materialized, metadata = root / "data", root / "train.json", root / "source_metadata.pt"
    dump(dataset / "pairs/val.jsonl", {"fixture": "VAL"})
    dump(materialized, {"fixture": "TRAIN"})
    config, loss = asdict(full24_reference_config()), asdict(RachelN512LossConfig())
    torch.save(dict(model_config=config, loss_config=loss, unused_weight=torch.ones(2)), metadata)
    identity = dict(schema_version="rachel-score-design-training/1", seed=260913, score_design=architecture,
        training_mode="joint", train_manifest_sha256=queue._sha256(materialized),
        validation_manifest_sha256=queue._sha256(dataset / "pairs/val.jsonl"),
        train_count=24000, train_split="train", validation_count=3000, validation_split="val",
        loss_config=loss, shared_base_initial_weights_sha256="a" * 64,
        candidate_config=asdict(CandidateScoreConfig()) if architecture != "original" else {},
        candidate_correctness_tolerance_px=20., candidate_correctness_weight=.5 if architecture == "candidate_dual" else 0.,
        candidate_correctness_reduction="mean valid candidates per pair, then mean supervised pairs",
        microbatch=4, effective_batch=16, optimizer="AdamW", weight_decay=1e-4,
        lr_by_epoch=[queue.learning_rate(e) for e in range(1, 51)], grad_clip_norm=5., precision="fp32",
        max_epochs=50, segment_pairs=6000, validation_every_epochs=1, min_selection_epoch=5,
        held_out_used_for_training_or_selection=False)
    run = root / architecture / "training"
    run.mkdir(parents=True)
    epoch = 5
    checkpoint = run / "epoch_005.pt"
    torch.save(dict(score_design_schema="rachel-score-design-training/1", epoch=epoch,
        model_config=config, model_state_dict={"fixture": torch.ones(2)}, source_weights_loaded=False,
        formal_training_counted=True, completed_segments=epoch * 4, global_exposure=epoch * 24000,
        optimizer_updates=epoch * 1500, optimizer_state_dict={}, rng_state={}, resume_identity=identity), checkpoint)
    if budget != 5:
        (run / ("epoch_%03d.pt" % budget)).write_bytes(b"fixture-budget-anchor-exists")
    dump(run / "validation_005_rows.json", [{"fixture": "validation"}])
    thresholds = dict(coarse=.2, local=.3, fused=.4)
    points = dict(thresholds=dict(max_f1=.4, recall_95=.1))
    winners = {s: dict(selection=s, selected_epoch=epoch, selected_global_exposure=epoch * 24000,
        checkpoint=str(checkpoint), checkpoint_sha256=queue._sha256(checkpoint),
        validation_predictions=str(run / "validation_005_rows.json"),
        classifier_thresholds=thresholds, operating_points=points,
        test_or_real_or_ood_used_for_fit=False) for s in queue.SELECTIONS}
    freeze_path = run / "budget_freezes" / ("%03d" % budget) / "freeze.json"
    dump(freeze_path, dict(schema_version="rachel-score-design-training/1", status="frozen_at_budget",
        budget_epochs=budget, budget_exposures=budget * 24000, eligible_epoch_range=[5, budget],
        selection_population="cleanVAL3000 only", held_out_used_for_fit=False,
        resume_identity_sha256=queue.digest(identity), winners=winners))
    for selection in queue.SELECTIONS:
        model = dict(training_run=str(run), budget=budget, selection=selection, architecture=architecture,
            seed=260913, epoch=epoch, checkpoint_sha256=winners[selection]["checkpoint_sha256"],
            freeze_sha256=queue._sha256(freeze_path), model_config=config,
            classifier_thresholds=thresholds, operating_points=points)
        for split in queue.SPLITS:
            eval_fixture(queue.baseline_evaluation_path(root, architecture, budget, selection, split), split, model, keep_sha)
    s3_common = dict(fixture="paired protocol only", architecture=s3_arch)
    s3_freezes = {}
    for schedule in ("joint", "staged"):
        s3_run = root / s3_arch / "training" if schedule == "joint" else root / "s3" / s3_arch / "training"
        selections = {s: dict(selection=s, selected_epoch=20, selected_global_exposure=480000,
            eligible_epoch_range=[20, 20] if s == "fixed_epoch" else [13, 20],
            test_or_real_or_ood_used_for_fit=False, checkpoint_sha256="b" * 64,
            classifier_thresholds=thresholds, operating_points=points)
            for s in ("fixed_epoch", "max_f1", "recall95")}
        s3_freeze_path = s3_run / "s3_freezes/freeze.json"
        dump(s3_freeze_path, dict(schema_version=queue.S3_SCHEMA, status="frozen_s3_epoch20",
            schedule=schedule, seed=260913, training_run=str(s3_run), architecture=s3_arch,
            budget_epochs=20, budget_exposures=480000, budget_optimizer_updates=30000,
            eligible_epoch_range=[13, 20], primary_selection="fixed_epoch", primary_epoch=20,
            selection_population="cleanVAL3000 only", held_out_used_for_fit=False, live_s0_s2_winner_imported=False,
            source_epochs=[dict(epoch=e) for e in range(13, 21)], selections=selections,
            common_training_contract=s3_common, common_training_contract_sha256=queue.digest(s3_common),
            validation_population_sha256="c" * 64, phase_protocol=stage_protocol(s3_arch, schedule=schedule)))
        s3_freezes[schedule] = str(s3_freeze_path)
        for selection, winner in selections.items():
            model = dict(training_run=str(s3_run), budget=20, selection=selection, schedule=schedule,
                architecture=s3_arch, seed=260913, epoch=20, checkpoint_sha256=winner["checkpoint_sha256"],
                freeze_sha256=queue._sha256(s3_freeze_path), classifier_thresholds=thresholds, operating_points=points)
            for split in queue.SPLITS:
                eval_fixture(s3_run.parent / "evaluation/s3_epoch020" / selection / split, split, model, keep_sha, s3=True)
    paired = root / "s3" / s3_arch / "paired_freeze.json"
    dump(paired, dict(schema_version=queue.S3_SCHEMA, status="paired_frozen", budget_epochs=20,
        primary="fixed_epoch20", auxiliary_window=[13, 20], held_out_used_for_fit=False,
        joint_freeze=s3_freezes["joint"], staged_freeze=s3_freezes["staged"],
        common_training_contract_sha256=queue.digest(s3_common)))
    return dict(s3_comparison=paired, dataset=dataset, materialized=materialized, metadata_checkpoint=metadata)


class InputQueueTests(unittest.TestCase):
    def test_exactly_eight_stages_single_axis_no_default_retraining(self):
        specs = [InputVariantSpec(coarse_size=n) for n in (256, 512)]
        specs += [InputVariantSpec(window_sizes_px=(w,)) for w in (7, 16, 32, 64)]
        specs += [InputVariantSpec(transport_fusion="post")]
        for spec in specs:
            config = queue.build_config("/experiment", "candidate_dual", 10, spec, "chosen")
            self.assertEqual(config["source"], "/experiment/input_source")
            self.assertEqual(len(config["stages"]), 8)
            smoke, train = [train_score_input_variant.parser().parse_args(s["command"][3:]) for s in config["stages"][:2]]
            self.assertEqual(smoke.smoke, 32); self.assertIsNone(train.smoke)
            self.assertFalse(train.resume)
            self.assertEqual(train.stop_after_epoch, 10)
            self.assertEqual(train_score_input_variant.input_spec(train), spec)
            for s in config["stages"][2:]:
                args = evaluate_score_input_variant.parser().parse_args(s["command"][3:])
                self.assertEqual(args.budget, 10)
                self.assertIn("/candidate_dual/evaluation/budget_010/", args.baseline_evaluation)
                self.assertEqual(args.keep_ids is not None, args.split == "real")
            self.assertEqual(len({s["marker"] for s in config["stages"]}), 8)
        with self.assertRaises(ValueError): queue.locations("/x", "original", 5, InputVariantSpec(), "bad")
        with self.assertRaises(ValueError): queue.locations("/x", "original", 6, specs[0], "bad")
        with self.assertRaises(ValueError): queue.locations("/x", "original", 5, specs[0], "../bad")

    def test_complete_metadata_prerequisites_prepare_without_model_or_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            kw = make_fixture(root)
            real_load, locations = torch.load, []
            def meta_only(*args, **kwargs):
                locations.append(kwargs.get("map_location"))
                result = real_load(*args, **kwargs)
                if "model_state_dict" in result:
                    self.assertTrue(all(t.is_meta for t in result["model_state_dict"].values()))
                return result
            with patch.object(queue.torch, "load", side_effect=meta_only), patch("subprocess.Popen", side_effect=AssertionError("launch")):
                result = queue.prepare(root, "candidate_pair", 10, InputVariantSpec(coarse_size=256), "one", **kw)
            self.assertEqual(result["status"], "prepared_not_launched")
            self.assertEqual(locations, ["meta", "meta"])
            config = queue.read(result["config"])
            self.assertEqual(len(config["stages"]), 8)
            receipt = queue.read(root / "input_variants/one/preparation.json")
            self.assertEqual(len(receipt["s3_prerequisite"]["evaluations"]), 18)
            self.assertEqual(len(receipt["baseline"]["evaluations"]), 6)
            self.assertFalse(receipt["launches_processes"])
            self.assertFalse((root / "input_variants/one/training").exists())
            with self.assertRaises(FileExistsError):
                queue.prepare(root, "candidate_pair", 10, InputVariantSpec(coarse_size=256), "one", **kw)

    def test_candidate_requires_same_architecture_s3_original_only_uses_order_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            kw = make_fixture(root, s3_arch="candidate_dual")
            with self.assertRaisesRegex(ValueError, "another architecture"):
                queue.prepare(root, "candidate_pair", 10, InputVariantSpec(coarse_size=256), "one", **kw)
            self.assertFalse((root / "input_variants").exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            kw = make_fixture(root, architecture="original")
            queue.prepare(root, "original", 10, InputVariantSpec(coarse_size=512), "original_one", **kw)
            receipt = queue.read(root / "input_variants/original_one/preparation.json")
            self.assertIn("no original staged conclusion", receipt["s3_prerequisite"]["role"])

    def test_missing_s3_eval_or_baseline_eval_or_mismatched_data_block_before_output(self):
        for failure in ("s3", "baseline", "manifest", "budget", "source"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve(); kw = make_fixture(root)
                if failure == "s3":
                    path = root / "s3/candidate_pair/evaluation/s3_epoch020/recall95/ood/summary.json"
                    dump(path, dict(status="running"))
                elif failure == "baseline":
                    (root / "candidate_pair/evaluation/budget_010/max_f1/real/pair_results.jsonl").unlink()
                elif failure == "manifest":
                    kw["materialized"].write_text("changed")
                elif failure == "budget":
                    (root / "candidate_pair/training/epoch_010.pt").unlink()
                else:
                    (root / "input_source").rmdir()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    queue.prepare(root, "candidate_pair", 10, InputVariantSpec(window_sizes_px=(7,)), "one", **kw)
                self.assertFalse((root / "input_variants").exists())

    def test_held_out_metric_values_do_not_choose_spec_or_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); kw = make_fixture(root)
            for path in root.rglob("summary.json"):
                dump(path, dict(status="complete", ignored_metric=99999999))
            queue.prepare(root, "candidate_pair", 10, InputVariantSpec(transport_fusion="post"), "selected", **kw)
            receipt = queue.read(root / "input_variants/selected/preparation.json")
            self.assertEqual(receipt["budget"], 10)
            self.assertEqual(receipt["input_spec"]["transport_fusion"], "post")
            self.assertFalse(receipt["held_out_metrics_used_to_choose_next_action"])


if __name__ == "__main__":
    unittest.main()
