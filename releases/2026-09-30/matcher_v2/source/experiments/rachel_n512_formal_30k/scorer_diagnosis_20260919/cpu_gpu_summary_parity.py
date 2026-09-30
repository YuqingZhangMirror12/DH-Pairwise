"""Eight original S6 TEST inputs: CPU Matcher vs saved historical GPU Q.

Engineering numeric check only: never run CUDA, fit statistics, read TRAIN,
select thresholds or compare different pairs. All outputs go to a new folder.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

# Set before importing NumPy/Torch; this standalone process uses CPU1 only.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from experiments.rachel_n512_formal_30k import train_score_decoupled as trainer
from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluator
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_head import features as sf

SCHEMA = "same-input-cpu-vs-saved-gpu-spectral-summary/1"
ROOT = Path("/root/autodl-tmp/rachel_score_design_20260913_001")
CHECKPOINT = ROOT / "attention_depth_20260915/s4_cross_attention_depth2/training/epoch_020.pt"
SOURCE_SHA = "56d3a4949e9d7e10f6ab5bdadc0b8a50d17192a7130584c3b5d22e21ce4d2076"
FIELDS = sf.INPUT_FIELDS


def restore_inputs(archive):
    arrays = {}
    for side in ("a", "b"):
        mask = np.asarray(archive["mask_" + side])
        if mask.dtype != np.uint8 or not np.isin(mask, [0, 1]).all():
            raise ValueError("saved mask casting is not verified lossless binary")
        arrays["mask_" + side] = mask[None, None].astype(np.float32)
        arrays["points_rc_" + side] = np.asarray(archive["points_rc_" + side])[None]
        arrays["contour_valid_" + side] = np.asarray(archive["valid_" + side])[None]
    return arrays


def verify_inputs(saved, batch):
    checks = {}
    for key in FIELDS:
        dtype = np.bool_ if key.startswith("contour_valid") else np.float32
        raw = np.asarray(getattr(batch, key))
        actual = np.asarray(raw, dtype=dtype)
        stored = saved[key]
        same = (stored.dtype == actual.dtype and stored.shape == actual.shape
                and np.ascontiguousarray(stored).tobytes() == np.ascontiguousarray(actual).tobytes())
        if not same:
            raise ValueError("historical NPZ and original TEST loader inputs differ: " + key)
        checks[key] = dict(shape=list(actual.shape), tensor_dtype=str(actual.dtype),
            raw_loader_dtype=str(raw.dtype), bytes_identical=True)
    # Remove the batch axis, retaining the original mask channel axis.
    digest = sf.model_input_sha256({k: saved[k][0] for k in FIELDS})
    return checks, digest


def compare(cpu, gpu, valid_a, valid_b):
    if cpu.shape != gpu.shape or cpu.dtype != gpu.dtype or cpu.dtype != np.float32:
        raise ValueError("CPU/GPU real_transport shape/dtype differ")
    if not np.isfinite(cpu).all() or not np.isfinite(gpu).all():
        raise ValueError("nonfinite transport")
    error = np.abs(cpu.astype(np.float64) - gpu.astype(np.float64))
    effective = error[np.ix_(valid_a, valid_b)]
    cpu_summary = sf.compute_summary(cpu, valid_a, valid_b, matrix_kind="real_transport")
    gpu_summary = sf.compute_summary(gpu, valid_a, valid_b, matrix_kind="real_transport")
    diff = np.abs(np.asarray(cpu_summary.values) - np.asarray(gpu_summary.values))
    return dict(q_shape=list(cpu.shape), q_dtype=str(cpu.dtype), q_max_abs_error=float(error.max()),
        q_mean_abs_error=float(error.mean()), q_valid_mean_abs_error=float(effective.mean()),
        q_relative_fro_error=float(np.linalg.norm(error) / max(np.linalg.norm(gpu.astype(np.float64)), 1e-30)),
        q_cpu_mass=float(cpu.astype(np.float64).sum()), q_gpu_mass=float(gpu.astype(np.float64).sum()),
        q_bitwise_equal=bool(np.array_equal(cpu, gpu)), summary_max_abs_error=float(diff.max()),
        summary_abs_error=dict(zip(sf.FEATURE_NAMES, diff.tolist())),
        cpu_summary=dict(zip(sf.FEATURE_NAMES, cpu_summary.values)),
        saved_gpu_summary=dict(zip(sf.FEATURE_NAMES, gpu_summary.values)),
        valid_points_a=cpu_summary.n_a, valid_points_b=cpu_summary.n_b)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        from threadpoolctl import threadpool_limits
        limit = threadpool_limits(limits=1)
    except ImportError:
        limit = None  # standalone pre-import environment is pinned to1 above
    heatmaps = Path(args.heatmap_root).resolve(strict=True)
    protocol = json.loads((heatmaps / "protocol.json").read_text())
    cases = json.loads((heatmaps / "cases.json").read_text())
    selected = [row for row in cases if row["dataset"] == "test"]
    if (protocol.get("status") != "complete" or len(selected) != 8
            or protocol["model"]["checkpoint_sha256"] != SOURCE_SHA
            or protocol["runtime"]["device"] != "cuda:0"
            or protocol["runtime"]["precision"] != "fp32"
            or protocol["runtime"]["batch_size"] != 1):
        raise ValueError("requires exactly the original eight S6 TEST GPU cases")
    source = Path(args.checkpoint).resolve(strict=True)
    source_sha = sf.file_sha256(source)  # once for the single source checkpoint
    if source_sha != SOURCE_SHA:
        raise ValueError("wrong S6-D2 source checkpoint")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    model = trainer.load_decoupled_checkpoint(payload)
    base = model.base_model.cpu().eval().requires_grad_(False)
    if model.head_kind != "cross_attention" or model.score_head.depth != 2:
        raise ValueError("source architecture differs")
    del model, payload
    dataset_root = Path(args.dataset).resolve(strict=True)
    manifest_path = dataset_root / "pairs/test.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    lookup = {r["pair_id"]: i for i, r in enumerate(manifest)}
    if len(manifest) != 3000 or len(lookup) != 3000:
        raise ValueError("expected original complete TEST3000 manifest")
    dataset = evaluator.core.RachelPairDataset(dataset_root, "test")
    ids = [r["pair_id"] for r in selected]
    batches = evaluator.core.make_ablation_loader(dataset, [lookup[p] for p in ids], batch_size=1,
        num_workers=0, seed=protocol["model"]["seed"], contour_cap=512)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema_version=SCHEMA, status="running", selected_count=8, dataset="test",
        source_checkpoint=str(source), source_checkpoint_sha256=source_sha,
        historical_protocol=str(heatmaps / "protocol.json"), historical_runtime=protocol["runtime"],
        cpu_runtime=dict(torch_version=str(torch.__version__), numpy_version=np.__version__,
            device="cpu", torch_threads=1, workers=1, cuda_visible_devices="",
            deterministic_algorithms=True, autocast=False),
        source_model_config=asdict(base.config), test_manifest_sha256=sf.file_sha256(manifest_path),
        script_sha256=sf.file_sha256(__file__), spectral_features_sha256=sf.file_sha256(sf.__file__),
        all_inputs_checked_against_original_test_loader=True, gpu_executed=False,
        train_data_read=False, statistics_fitted=False, thresholds_fitted=False,
        GT_used_for_case_selection=False, selection="all8 prior S6 heatmap TEST cases, unchanged order",
        numerical_acceptance_rule="descriptive raw errors only; no TEST-tuned threshold or normalization")
    save(output / "protocol.json", receipt)
    rows = []
    try:
        for row, batch in zip(selected, batches):
            if list(batch.pair_ids) != [row["pair_id"]]:
                raise ValueError("case order differs")
            path = heatmaps / row["arrays_path"]
            if sf.file_sha256(path) != row["arrays_sha256"]:  # once per NPZ
                raise ValueError("historical arrays hash mismatch")
            with np.load(path, allow_pickle=False) as z:
                inputs = restore_inputs(z)
                gpu_q = z["sinkhorn_assignment"].copy()
            checks, input_sha = verify_inputs(inputs, batch)
            tensors = [torch.from_numpy(inputs[k]) for k in FIELDS]
            case_started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
                current = base(*tensors)
            cpu_q = current.assignment[0].numpy()
            record = compare(cpu_q, gpu_q, inputs["contour_valid_a"][0], inputs["contour_valid_b"][0])
            record.update(pair_id=row["pair_id"], arrays_path=str(path), arrays_sha256=row["arrays_sha256"],
                input_sha256=input_sha, input_checks=checks, elapsed_cpu_forward_and_summary_s=time.perf_counter() - case_started)
            rows.append(record)
            save(output / "partial_results.json", rows)
            print(json.dumps(dict(completed=len(rows), pair_id=row["pair_id"], q_max_abs_error=record["q_max_abs_error"],
                summary_max_abs_error=record["summary_max_abs_error"])), flush=True)
        if len(rows) != 8:
            raise ValueError("incomplete eight-case replay")
        result = dict(case_count=8, all_six_inputs_bytes_equal=True,
            q_max_abs_error=max(r["q_max_abs_error"] for r in rows),
            q_mean_abs_error_equal_pair_mean=float(np.mean([r["q_mean_abs_error"] for r in rows])),
            q_relative_fro_error_max=max(r["q_relative_fro_error"] for r in rows),
            summary_max_abs_error=max(r["summary_max_abs_error"] for r in rows),
            per_feature_max_abs_error={k: max(r["summary_abs_error"][k] for r in rows) for k in sf.FEATURE_NAMES},
            bitwise_equal_cases=sum(r["q_bitwise_equal"] for r in rows), rows=rows)
        save(output / "results.json", result)
        receipt.update(status="complete", completed_count=8, results_sha256=sf.file_sha256(output / "results.json"),
            summary={k: v for k, v in result.items() if k != "rows"})
    except BaseException as error:
        receipt.update(status="failed", completed_count=len(rows), error=repr(error))
        raise
    finally:
        receipt["elapsed_s"] = time.perf_counter() - started
        save(output / "protocol.json", receipt)
        if limit is not None:
            limit.restore_original_limits()
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heatmap-root", default=str(ROOT / "scorer_diagnosis_20260919/heatmaps_v1/s6"))
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--dataset", default="/root/autodl-tmp/dataset_rachel_pairwise_n512_v1")
    p.add_argument("--output", required=True)
    run(p.parse_args())
