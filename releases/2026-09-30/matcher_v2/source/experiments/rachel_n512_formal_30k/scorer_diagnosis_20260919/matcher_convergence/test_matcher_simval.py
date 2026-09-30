"""CPU-only synthetic tests; no remote I/O, real checkpoints, datasets or GPU."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("matcher_simval", Path(__file__).with_name("evaluate_matcher_simval.py"))
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)
from experiments.rachel_n512_formal_30k.test_decoupled_samplewise_loss import fixture, legacy_single, sliced
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig

torch.set_num_threads(1)


def payload(epoch=8):
    return dict(epoch=epoch, phase="matcher", completed_segments=epoch * 4,
                global_exposure=epoch * 24000, checkpoint_role="epoch_anchor",
                decoupled_score={"phase": "matcher"}, resume_identity={
                    "sampling": "original512", "base_model_config": {"contour_cap": 512},
                    "populations": {"val": {"manifest_sha256": evaluate.VAL_HASH, "count": 3000}}})


def record(pair_id, positive=True, valid=True, layout=True):
    return dict(pair_id=pair_id, label=positive, training_valid=valid, decision_valid=valid,
                losses={key: 2. if valid else 0. for key in evaluate.TERM_NAMES},
                supervised_correspondence_count=2 if positive and valid else 0,
                pose_supervised=positive and valid, differentiable_translation_l2_px=5. if positive and valid else None,
                raw_layout_valid=layout, raw_layout20_correct=positive and layout)


class IdentityTests(unittest.TestCase):
    def test_exact_m_epochs(self):
        for epoch in (8, 10, 12):
            evaluate.validate_payload_metadata(payload(epoch), epoch)

    def test_refuse_classifier_or_wrong_anchor(self):
        for key, value in (("phase", "classifier"), ("completed_segments", 31),
                           ("epoch", 20), ("global_exposure", 1), ("checkpoint_role", "recovery")):
            p = payload(); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluate.validate_payload_metadata(p, 8)
        with self.assertRaises(ValueError):
            evaluate.validate_payload_metadata(payload(9), 9)

    def test_refuse_wrong_sampling_and_val(self):
        for route, value in ((["sampling"], "step3"), (["base_model_config", "contour_cap"], 2048),
                             (["populations", "val", "manifest_sha256"], "wrong")):
            p = payload(); target = p["resume_identity"]
            for key in route[:-1]: target = target[key]
            target[route[-1]] = value
            with self.assertRaises(ValueError): evaluate.validate_payload_metadata(p, 8)

    def test_manifest_count_balance_duplicates(self):
        rows = [dict(pair_id=str(i), label=i < 1500) for i in range(3000)]
        self.assertEqual(len(evaluate.validate_manifest(rows)), 3000)
        for altered in (rows[:-1], rows + rows[:1], [dict(r, label=True) for r in rows]):
            with self.assertRaises(ValueError): evaluate.validate_manifest(altered)
        rows[-1]["pair_id"] = rows[0]["pair_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate.validate_manifest(rows)

    def test_cli_has_no_selection_epochs_test_or_real_flags(self):
        args = evaluate.parser().parse_args([])
        self.assertFalse(args.execute)
        self.assertEqual(args.batch_size, 1)
        names = {item.dest for item in evaluate.parser()._actions}
        self.assertFalse({"selection", "epochs", "split", "keep_ids", "ood_prepared"} & names)


class NumericalTests(unittest.TestCase):
    def test_reuses_micro1_loss_and_clean_positive_pose(self):
        out, targets, _, _ = fixture(8)
        terms, counts, pose = evaluate.measure_losses(out, targets, RachelN512LossConfig())
        references = [legacy_single(sliced(out, i), tuple(t[i:i+1] for t in targets),
                                    targets[4][i:i+1], "matcher") for i in range(8)]
        for i, (total, component) in enumerate(references):
            self.assertAlmostEqual(terms["total"][i], total.item(), places=12)
            for key in ("assignment_nll", "translation_smooth_l1", "sinkhorn_residual"):
                self.assertAlmostEqual(terms[key][i], component[key].item(), places=12)
        self.assertTrue(pose[1])  # Clean positive: no inherited train weathering pose mask.
        self.assertFalse(pose[4])  # Invalid positive remains zero, separately counted.
        self.assertEqual(counts[4], 0)
        self.assertEqual(terms["total"][4], 0)

    def test_all_pair_and_conditional_denominators(self):
        rows = [record("good"), record("invalid", valid=False, layout=False), record("neg", positive=False)]
        s = evaluate.summarize(rows)
        self.assertAlmostEqual(s["all_pairs"]["mean_losses"]["assignment_nll"], 4/3)
        self.assertEqual(s["positive_pairs"]["mean_losses"]["translation_smooth_l1"], 1)
        self.assertEqual(s["conditional_supervision"]["translation_smooth_l1_per_supervised_pair"], 2)
        self.assertEqual(s["raw_layout20"]["positive_count"], 2)
        self.assertEqual(s["raw_layout20"]["success_rate"], .5)

    def test_no_classification_gate_on_raw_layout(self):
        row = record("p"); row["decision_valid"] = False
        self.assertEqual(evaluate.summarize([row])["raw_layout20"]["correct_count"], 1)

    def test_empty_supervision_is_null_and_zero_remains_zero(self):
        s = evaluate.summarize([record("n", positive=False, valid=False, layout=False)])
        self.assertIsNone(s["conditional_supervision"]["correspondence_nll_per_supervised_pair"])
        self.assertIsNone(s["raw_layout20"]["success_rate"])
        self.assertEqual(s["all_pairs"]["mean_losses"]["total"], 0)
        with self.assertRaises(ValueError): evaluate.summarize([])
        with self.assertRaises(ValueError): evaluate.summarize([record("x"), record("x")])

    def test_small_cpu_forward_loss_decoder_without_weights(self):
        out, targets, _, _ = fixture(2)
        out.decision_valid = out.training_valid.clone()
        batch = SimpleNamespace(pair_ids=("a", "b"),
            mask_a=np.zeros((2, 1, 8, 8), dtype=np.float32), mask_b=np.zeros((2, 1, 8, 8), dtype=np.float32),
            points_rc_a=np.tile(np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32), (2, 1, 1)),
            points_rc_b=np.tile(np.array([[2, 3], [12, 3], [12, 13], [2, 13]], dtype=np.float32), (2, 1, 1)),
            contour_valid_a=np.ones((2, 4), dtype=bool), contour_valid_b=np.ones((2, 4), dtype=bool),
            labels=targets[0].numpy(), target_a=targets[1].numpy(), target_b=targets[2].numpy(),
            translation_a_to_b_rc=targets[3].numpy(), translation_valid=targets[4].numpy())
        calls = []
        def fake_model(*args):
            calls.append(len(args)); self.assertFalse(torch.is_grad_enabled()); return out
        rows = evaluate.evaluate_batch(fake_model, batch, RachelN512LossConfig(), torch.device("cpu"))
        self.assertEqual(calls, [6])
        self.assertEqual([r["pair_id"] for r in rows], ["a", "b"])
        self.assertEqual(set(rows[0]["losses"]), set(evaluate.TERM_NAMES))
        self.assertNotIn("score", rows[0])
        self.assertNotIn("threshold", rows[0])


class SafetyTests(unittest.TestCase):
    def test_shared_lock_held_refuses_before_gpu_query(self):
        import fcntl
        with tempfile.TemporaryDirectory() as d, patch.object(fcntl, "flock", side_effect=BlockingIOError), \
             patch.object(evaluate.subprocess, "check_output") as query:
            with self.assertRaisesRegex(RuntimeError, "lock is held"):
                with evaluate.exclusive_gpu(Path(d) / "lock"): self.fail("must not enter")
            query.assert_not_called()

    def test_compute_process_refuses_and_empty_gpu_allows(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(evaluate.subprocess, "check_output", return_value="1234\n"):
                with self.assertRaisesRegex(RuntimeError, "compute processes"):
                    with evaluate.exclusive_gpu(Path(d) / "lock"): self.fail("must not enter")
            with patch.object(evaluate.subprocess, "check_output", return_value=""):
                with evaluate.exclusive_gpu(Path(d) / "lock"): pass

    def test_preflight_reads_no_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "dataset/pairs").mkdir(parents=True)
            manifest = root / "dataset/pairs/val.jsonl"
            manifest.write_text("\n".join(json.dumps(dict(pair_id=str(i), label=i < 1500)) for i in range(3000)))
            run = root / "run"; run.mkdir()
            for epoch in evaluate.EPOCHS: (run / ("epoch_%03d.pt" % epoch)).write_bytes(b"not a checkpoint")
            args = SimpleNamespace(dataset=root / "dataset", arm="all", output=root / "new", batch_size=1, workers=0, execute=False)
            with patch.object(evaluate, "VAL_HASH", evaluate.sha256(manifest)), patch.object(evaluate, "ARMS", {"test": run}), \
                 patch.object(torch, "load", side_effect=AssertionError("preflight must not load weights")):
                plan = evaluate.preflight(args)
            self.assertEqual(len(plan["checkpoint_entries"]), 3)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
