"""Synthetic CPU contracts only; no real weights, remote data or GPU."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence import evaluate_matcher_simval_cpu as cpu
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence import test_matcher_simval as reference


class CPUContracts(unittest.TestCase):
    def test_small_default_and_balanced_systematic_indices(self):
        args = cpu.parser().parse_args([])
        self.assertEqual((args.pairs, args.epoch, args.arm, args.workers, args.cpu_threads),
                         (24, "12", "shared_s3_s4_s6", 0, 2))
        self.assertFalse(args.execute)
        rows = [dict(pair_id=str(i), label=i % 2) for i in range(3000)]
        for count in (24, 32, 3000):
            ids = cpu.selected_indices(rows, count)
            self.assertEqual(len(ids), count)
            self.assertEqual(sum(rows[i]["label"] for i in ids), count // 2)
            self.assertEqual(tuple(sorted(set(ids))), ids)
            self.assertEqual(cpu.selected_indices(rows, count), ids)
        with self.assertRaises(ValueError):
            cpu.selected_indices(rows, 8)

    def test_preflight_reuses_manifest_contract_without_load_or_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / "dataset/pairs/val.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("\n".join(json.dumps(dict(pair_id=str(i), label=i % 2)) for i in range(3000)))
            run = root / "training"
            run.mkdir()
            for epoch in (8, 10, 12):
                (run / ("epoch_%03d.pt" % epoch)).write_bytes(b"not a checkpoint")
            args = cpu.parser().parse_args(["--dataset", str(root / "dataset"), "--output", str(root / "new")])
            with patch.object(cpu.original, "VAL_HASH", cpu.original.sha256(manifest)), \
                 patch.object(cpu.original, "ARMS", {"shared_s3_s4_s6": run}), \
                 patch.object(torch, "load", side_effect=AssertionError("no checkpoint read")), \
                 patch.object(cpu.original, "exclusive_gpu", side_effect=AssertionError("no GPU lock")):
                plan = cpu.preflight(args)
            self.assertEqual(len(plan["checkpoint_entries"]), 1)
            self.assertEqual(plan["checkpoint_entries"][0]["epoch"], 12)
            self.assertEqual(plan["sample_count"], 24)
            self.assertNotIn("gpu_lock", plan)
            self.assertFalse(args.output.exists())

    def test_cpu_runtime_never_initializes_cuda(self):
        with patch.dict(os.environ), patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("no CUDA")):
            cpu.configure_cpu(1)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertFalse(torch.cuda.is_initialized())
            self.assertEqual(torch.get_num_threads(), 1)
        with patch.dict(os.environ), patch.object(torch.cuda, "is_initialized", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "initialized CUDA"):
                cpu.configure_cpu(1)

    def test_timing_excludes_two_warmups_only_from_estimate(self):
        points = [dict(data_seconds=.1, evaluate_seconds=x) for x in (9., 8., 1., 2.)]
        result = cpu.timing_summary(points)
        self.assertEqual(result["measured_pairs"], 4)
        self.assertAlmostEqual(result["steady_pair_seconds"]["median"], 1.6)
        self.assertAlmostEqual(result["rough_3000_pair_loop_seconds"], 4800.)

    def test_orchestration_uses_original_evaluator_cpu_only(self):
        from experiments.rachel_n512_formal_30k import train_score_decoupled as train
        from experiments.rachel_n512_formal_30k import resampled_input_support as inputs
        from staging.pairwise_v0_2.pairwise_data import rachel_training_dataset as data
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ):
            root = Path(d)
            checkpoint = root / "epoch_012.pt"
            checkpoint.write_bytes(b"synthetic source")
            manifest = root / "val.jsonl"
            manifest.write_text("fixed fixture")
            args = cpu.parser().parse_args(["--dataset", str(root), "--output", str(root / "new"), "--cpu-threads", "1"])
            ids = [str(i) for i in range(24)]
            entry = dict(arm="shared_s3_s4_s6", epoch=12, checkpoint=str(checkpoint))
            plan = dict(output=str(args.output), checkpoint_entries=[entry], indices=list(range(24)),
                pair_ids=ids, purpose="timing_subset_not_validation_curve", simval_manifest=str(manifest))
            payload = reference.payload(12)
            payload["resume_identity"]["populations"]["val"]["manifest_sha256"] = cpu.original.sha256(manifest)
            payload["loss_config"] = {}
            class Dataset:
                split = "val"
                def __len__(self): return 3000
            seen = []
            def evaluate(model, batch, config, device):
                seen.append(device.type)
                return [reference.record(batch, positive=int(batch) % 2 == 1)]
            with patch.object(cpu.original, "VAL_HASH", cpu.original.sha256(manifest)), \
                 patch.object(data, "RachelPairDataset", return_value=Dataset()), \
                 patch.object(inputs, "make_ablation_loader", return_value=ids), \
                 patch.object(train, "matcher_contract", return_value={"fixture": True}), \
                 patch.object(train, "load_decoupled_checkpoint", return_value=SimpleNamespace(base_model=torch.nn.Linear(1, 1))), \
                 patch.object(torch, "load", return_value=payload) as load, \
                 patch.object(cpu.original, "evaluate_batch", side_effect=evaluate), \
                 patch.object(cpu.original, "exclusive_gpu", side_effect=AssertionError("no GPU lock")), \
                 patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("no CUDA")):
                cpu.execute(args, plan)
            self.assertEqual(seen, ["cpu"] * 24)
            self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
            protocol = json.loads((args.output / "protocol.json").read_text())
            summary = json.loads((args.output / "shared_s3_s4_s6/m12/summary.json").read_text())
            self.assertEqual(protocol["status"], "complete")
            self.assertFalse(protocol["runtime"]["cuda_initialized"])
            self.assertEqual(summary["metrics"]["raw_layout20"]["positive_count"], 12)
            self.assertEqual(summary["sample_count"], 24)


if __name__ == "__main__":
    unittest.main()
