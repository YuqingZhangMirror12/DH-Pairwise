"""Small CPU/config tests; no remote queues, datasets or CUDA are touched."""
from dataclasses import replace
import unittest

from experiments.rachel_n512_formal_30k import prepare_s8_attention_extension as prep


def flag(command, name):
    return command[command.index(name) + 1]


class S8AttentionPreparationTests(unittest.TestCase):
    def setUp(self):
        self.config = prep.build_config("/experiment/s5_attention", "/experiment/s5_attention/source",
                                        reference_root="/experiment")

    def test_new_head_only_and_exact_existing_step_density_sources(self):
        stages = self.config["stages"]
        self.assertEqual(len(stages), 12)
        self.assertEqual(len({s["marker"] for s in stages}), 12)
        for stage in stages[1:3]:
            command = stage["command"]
            for name, value in {"--head-kind": "cross_attention", "--cross-attention-depth": "2",
                    "--sampling": "step3", "--microbatch": "1", "--physical-microbatch": "4",
                    "--effective-batch": "16", "--stop-after-epoch": "20"}.items():
                self.assertEqual(flag(command, name), value)
            self.assertEqual(flag(command, "--matcher-checkpoint"),
                "/experiment/new_s345_20260914/s5_step3_cap2048/training/epoch_012.pt")
            self.assertEqual(flag(command, "--density-train-manifest"),
                "/experiment/new_s345_20260914/step_data/train/manifest.json")
            self.assertEqual(flag(command, "--clean-val-manifest"),
                "/experiment/new_s345_20260914/step_data/val/manifest.json")
            self.assertNotIn("--matrix-head-revision", command)
        self.assertEqual(flag(stages[1]["command"], "--smoke-phase"), "classifier")
        self.assertEqual(stages[2]["resume_arguments"], ["--resume"])
        self.assertFalse(self.config["preparation"]["matcher_retrained"])
        self.assertEqual(self.config["preparation"]["new_classifier_exposures"], 192000)

    def test_evaluations_compare_same2048_cnn_at_frozen_selections(self):
        seen = set()
        for stage in self.config["stages"][3:]:
            command = stage["command"]
            selection, split = flag(command, "--selection"), flag(command, "--split")
            seen.add((selection, split))
            self.assertEqual(flag(command, "--baseline-evaluation"),
                "/experiment/new_s345_20260914/s5_step3_cap2048/evaluation/%s/%s" % (selection, split))
            self.assertEqual("--keep-ids" in command, split == "real")
            self.assertNotIn("--threshold", command)
        self.assertEqual(seen, {(s, p) for s in prep.SELECTIONS for p in prep.SPLITS})

    def test_commands_parse_and_validate(self):
        from experiments.rachel_n512_formal_30k import train_score_decoupled as train, evaluate_score_decoupled as evaluate, benchmark_s8_attention_capacity as gate
        for stage in self.config["stages"]:
            command = stage["command"]
            module = train if command[2].endswith("train_score_decoupled") else gate if command[2].endswith("benchmark_s8_attention_capacity") else evaluate
            args = module.parser().parse_args(command[3:])
            if module is train:
                train.validate_arguments(args)

    def test_refuses_old_output_or_unregistered_batch(self):
        for root, source in [("/experiment/new_s345_20260914", "/experiment/new_s345_20260914/source"),
                ("/experiment/new_s345_20260914/new", "/experiment/new_s345_20260914/new/source"),
                ("/experiment/new", "/experiment/source")]:
            with self.assertRaises(ValueError):
                prep.build_config(root, source, reference_root="/experiment")
        for physical in (1, 2, 8, 16, 20, 36, 54):
            with self.assertRaises(ValueError):
                prep.build_config("/experiment/new", "/experiment/new/source", physical_microbatch=physical)

    def test_matrix_m12_import_to_depth2_cap2048_preserves_base_but_not_old_head(self):
        import torch
        from experiments.rachel_n512_formal_30k import train_score_decoupled as train
        from experiments.rachel_n512_formal_30k.test_train_score_decoupled import model, identity
        from experiments.rachel_n512_formal_30k.train_joint_damage import state_digest
        from staging.pairwise_v0_2.models.rachel_decoupled_score import build_decoupled_score_model
        from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
        torch.set_num_threads(1)
        config = replace(model().config, contour_cap=2048)
        matrix = build_decoupled_score_model(config, "matrix_cnn", matrix_head_revision="per_pair_norm_v3")
        ident = identity(matrix)
        # Frozen remote snapshots can carry an older default test-fixture revision.
        # The synthetic identity must describe the model this test actually built.
        ident.update(matrix_head_revision=matrix.metadata()["matrix_head_revision"])
        ident.update(sampling="step3", contour_cap=2048, populations={"train": {"density_fixture": True}, "val": {"density_fixture": True}})
        receipt = train.matcher_receipt(matrix, ident)
        payload = train.checkpoint_payload(matrix, train.create_optimizer(matrix), identity=ident,
            loss_config=RachelN512LossConfig(), completed=48, receipt=receipt, winners={}, role="synthetic_unit_fixture")
        cross = build_decoupled_score_model(config, "cross_attention", model_options={"cross_attention_depth": 2})
        other = dict(ident, head_kind="cross_attention", matrix_head_revision=None, model_options={"cross_attention_depth": 2})
        head_before = state_digest(cross.score_head)
        train.import_matcher(cross, payload, other)
        self.assertEqual(state_digest(cross.base_model), state_digest(matrix.base_model))
        self.assertEqual(state_digest(cross.score_head), head_before)
        self.assertEqual(train.matcher_contract(ident), train.matcher_contract(other))

    def test_capacity_gate_is_first_and_uses_only_existing_train_geometry_receipt(self):
        gate = self.config["stages"][0]
        command = gate["command"]
        self.assertTrue(command[2].endswith("benchmark_s8_attention_capacity"))
        self.assertEqual(flag(command, "--selection-file"),
            "/experiment/new_s345_20260914/s5_capacity_v3_20260914/cpu_plan/selection.json")
        self.assertEqual(gate["completion_statuses"], ["complete"])
        self.assertTrue(gate["completion_path"].endswith("capacity_gate/result.json"))
        self.assertNotIn("--resume", command)
        self.assertNotIn("--run-cuda", command)  # Explicit gate command itself is a one-shot CUDA stage.
        self.assertFalse(self.config["preparation"]["capacity_search"])

    def test_gpu_guard_only_queries_the_explicit_target_gpu(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from experiments.rachel_n512_formal_30k import benchmark_s8_attention_capacity as gate
        with patch.dict(gate.os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-target0"}), patch.object(gate.subprocess, "run", return_value=SimpleNamespace(stdout="")) as run:
            self.assertEqual(gate.require_isolated_idle_gpu(), "GPU-target0")
            self.assertEqual(run.call_args[0][0][:3], ["nvidia-smi", "-i", "GPU-target0"])
        with patch.dict(gate.os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            with self.assertRaisesRegex(RuntimeError, "one explicit"):
                gate.require_isolated_idle_gpu()

    def test_stress_plan_is_tied_to_train_identity_not_prior_cnn_scores(self):
        from types import SimpleNamespace
        from copy import deepcopy
        from experiments.rachel_n512_formal_30k import benchmark_s8_attention_capacity as gate
        class Dataset(SimpleNamespace):
            def __len__(self):
                return 24000
        entries = [{"pair_id": "fixture-%d" % i, "label": bool(i % 2)} for i in range(16)]
        dataset = Dataset(identity="identity", split="train", contour_cap=2048, entries=entries)
        rows = [dict(source_index=i, pair_id=e["pair_id"], label=e["label"], true_Na=100,
                     true_Nb=120, true_matrix_cells=12000) for i, e in enumerate(entries)]
        selection = dict(full_TRAIN_scanned_pair_count=24000, unique_pair_count=16, indices=list(range(16)),
                         pairs=rows, formal_collation={"cap": 2048})
        plan = dict(schema_version=gate.SELECTION_SCHEMA, status="cpu_plan_complete", dataset_identity="identity",
                    manifest_sha256="sha", selection=selection)
        self.assertEqual(gate.validate_plan(plan, dataset, "sha"), selection)
        wrong = deepcopy(plan)
        wrong["manifest_sha256"] = "different"
        with self.assertRaises(ValueError):
            gate.validate_plan(wrong, dataset, "sha")
        wrong = deepcopy(plan)
        wrong["selection"]["indices"][-1] = 0
        with self.assertRaises(ValueError):
            gate.validate_plan(wrong, dataset, "sha")


if __name__ == "__main__":
    unittest.main()
