"""CPU-only, post-hoc SVD of completed, frozen selected-case transport arrays.

No model imports, inference, fitting, threshold selection, or input mutations.
The saved assignment already excludes dustbins; validity masks remove padding.
Run this file directly with --arm-root .../heatmaps_v1/s4 --output .../spectral/s4.
Output directories must not exist and must be outside the original arm directory.
"""
from __future__ import annotations

import os

# Set before numpy import, including when imported by the test suite.
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _key in THREAD_ENV:
    os.environ[_key] = "1"

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

SCHEMA = "selected-transport-spectral-diagnostics/1"
TOPK = (1, 2, 4, 8, 16)
NOTES = {
    "population": "40 deliberately selected cases per matcher, not a random evaluation population",
    "selection": "REAL S6 decision/layout strata; OOD cross-model score disagreement and S6 score quartiles; SIM label strata",
    "interpretation": "descriptive/exploratory only; no significance, classification causality, or aggregate-performance claims",
    "matrix": "sinkhorn_assignment is non-dustbin real_transport; retain only valid_a x valid_b; never remove its last row/column as a dustbin",
    "normalization": "mass normalization divides the entire matrix by its sum, not row normalization; scalar normalization preserves spectral shape",
    "spectral": "float64 rectangular SVD; never apply eigenvalue decomposition directly to a non-square matrix",
    "effective_rank": "exp(entropy(sigma/sum(sigma))); zero matrix returns 0",
    "participation_ratio": "energy PR = (sum(sigma^2))^2/sum(sigma^4); singular PR also reported explicitly",
    "coverage": "matrix-index support/argmax coverage, NOT physical seam length or GT correspondence accuracy; exact ties use first index",
    "zero_matrix": "shape ratios and normalized maxima are null when mass/Frobenius norm is zero",
    "models": "S4 and S7 have distinct frozen matchers; same-matcher D2/D4 do not require repeat decomposition",
    "ood": "selected OOD cases are all positive and lack GT layout; no OOD classification accuracy or seam-accuracy claim",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "min": None, "q25": None, "median": None,
                "q75": None, "max": None, "mean": None}
    if not np.isfinite(values).all():
        raise ValueError("nonfinite summary input")
    result = dict(zip(("min", "q25", "median", "q75", "max"),
                      map(float, np.quantile(values, [0, .25, .5, .75, 1]))))
    result.update(count=int(values.size), mean=float(values.mean()))
    return result


def axis_metrics(matrix, total_mass):
    """Each row selects a column. Transpose for the reciprocal axis report."""
    axis_mass = matrix.sum(axis=1)
    maxima = matrix.max(axis=1)
    positive = axis_mass > 0
    winners = matrix.argmax(axis=1)
    winner_counts = np.bincount(winners[positive], minlength=matrix.shape[1])
    conditional_peaks = maxima[positive] / axis_mass[positive]
    return {
        "axis_count": int(matrix.shape[0]), "opposite_axis_count": int(matrix.shape[1]),
        "positive_mass_count": int(positive.sum()),
        "positive_mass_fraction": float(positive.mean()),
        "axis_transport_mass": distribution(axis_mass),
        "raw_maximum": distribution(maxima),
        "global_mass_normalized_maximum": distribution(maxima / total_mass) if total_mass else None,
        "conditional_maximum_given_positive_axis_mass": distribution(conditional_peaks),
        "maxima_sum_over_total_mass": float(maxima.sum() / total_mass) if total_mass else None,
        "argmax_unique_opposite_count": int(np.count_nonzero(winner_counts)),
        "argmax_opposite_coverage_fraction": float(np.count_nonzero(winner_counts) / matrix.shape[1]),
        "argmax_winner_multiplicity_including_zero": distribution(winner_counts),
        "exact_maximum_tie_axis_count": int(((matrix == maxima[:, None]).sum(axis=1)[positive] > 1).sum()),
    }


def spectral_metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("matrix must be nonempty and two-dimensional")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("transport matrix must be finite and nonnegative")
    mass = float(matrix.sum())
    # No Gram squaring: direct rectangular SVD avoids loss of small singular values.
    singular = np.linalg.svd(matrix, compute_uv=False)
    frobenius = float(np.linalg.norm(singular))
    sigma_sum = float(singular.sum())
    sigma1 = float(singular[0])
    probability = singular / sigma_sum if sigma_sum else np.zeros_like(singular)
    positive = probability > 0
    energy = (singular / frobenius) ** 2 if frobenius else np.zeros_like(singular)
    raw_frobenius = float(np.linalg.norm(matrix))
    if not np.isclose(frobenius, raw_frobenius, rtol=1e-10, atol=1e-14):
        raise ValueError("SVD Frobenius energy check failed")
    row_argmax = matrix.argmax(axis=1)
    col_argmax = matrix.argmax(axis=0)
    row_ids = np.arange(matrix.shape[0])
    mutual = (col_argmax[row_argmax] == row_ids) & (matrix[row_ids, row_argmax] > 0)
    tolerance = float(np.finfo(np.float64).eps * max(matrix.shape) * sigma1)
    return {
        "shape": list(matrix.shape),
        "raw_transport_total_mass": mass,
        "raw_frobenius_norm": frobenius,
        "raw_sigma1": sigma1,
        "singular_values": singular.tolist(),
        "mass_normalized": {
            "sigma1": sigma1 / mass if mass else None,
            "frobenius_norm": frobenius / mass if mass else None,
            "total_mass": 1.0 if mass else None,
        },
        "spectral_shape": {
            "sigma1_over_frobenius": sigma1 / frobenius if frobenius else None,
            "topk_energy_fraction": {str(k): float(energy[:k].sum()) if frobenius else None for k in TOPK},
            "topk_effective_k": {str(k): min(k, len(singular)) for k in TOPK},
            "effective_rank_entropy_singular": float(np.exp(-np.sum(probability[positive] * np.log(probability[positive])))) if sigma_sum else 0.0,
            "participation_ratio_energy": float(1 / np.sum(energy ** 2)) if frobenius else 0.0,
            "participation_ratio_singular": float(1 / np.sum(probability ** 2)) if sigma_sum else 0.0,
            "numerical_rank_float64": int((singular > tolerance).sum()),
            "numerical_rank_tolerance": tolerance,
        },
        "rows_select_columns": axis_metrics(matrix, mass),
        "columns_select_rows": axis_metrics(matrix.T, mass),
        "mutual_argmax": {"pair_count": int(mutual.sum()),
                          "row_coverage_fraction": float(mutual.sum() / matrix.shape[0]),
                          "column_coverage_fraction": float(mutual.sum() / matrix.shape[1])},
    }


def masked_assignment(arrays):
    matrix = np.asarray(arrays["sinkhorn_assignment"], dtype=np.float64)
    va, vb = np.asarray(arrays["valid_a"]), np.asarray(arrays["valid_b"])
    if matrix.ndim != 2 or va.shape != (matrix.shape[0],) or vb.shape != (matrix.shape[1],):
        raise ValueError("assignment and validity-mask shapes differ; do not guess dustbin/padding layout")
    if va.dtype != np.bool_ or vb.dtype != np.bool_:
        raise ValueError("validity masks must be boolean")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("saved matrix contains nonfinite or negative values")
    keep = va[:, None] & vb[None, :]
    metadata = {
        "saved_shape": list(matrix.shape), "valid_rows": int(va.sum()), "valid_columns": int(vb.sum()),
        "saved_total_mass_including_padding": float(matrix.sum()),
        "excluded_padding_mass": float(matrix[~keep].sum()),
        "dustbins_removed_by_this_script": False,
    }
    return matrix[np.ix_(va, vb)], metadata


SUMMARY_PATHS = {
    "transport_mass": ("raw_transport_total_mass",),
    "sigma1_over_fro": ("spectral_shape", "sigma1_over_frobenius"),
    "energy_top1": ("spectral_shape", "topk_energy_fraction", "1"),
    "energy_top4": ("spectral_shape", "topk_energy_fraction", "4"),
    "energy_top16": ("spectral_shape", "topk_energy_fraction", "16"),
    "effective_rank": ("spectral_shape", "effective_rank_entropy_singular"),
    "energy_pr": ("spectral_shape", "participation_ratio_energy"),
    "rows_argmax_column_coverage": ("rows_select_columns", "argmax_opposite_coverage_fraction"),
    "columns_argmax_row_coverage": ("columns_select_rows", "argmax_opposite_coverage_fraction"),
}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups["all_selected"].append(row)
        groups["dataset_label/" + row["dataset"] + "/" + str(int(row["label"]))].append(row)
        groups["stratum/" + row["stratum"]].append(row)
    result = {}
    for group, members in sorted(groups.items()):
        stats = {"case_count": len(members)}
        for name, path in SUMMARY_PATHS.items():
            values = []
            for row in members:
                value = row["metrics"]
                for component in path:
                    value = value[component]
                if value is not None:
                    values.append(value)
            stats[name] = distribution(values)
        result[group] = stats
    return result


def render_markdown(result):
    lines = ["# Selected40 transport 谱诊断 — " + result["arm"], "",
             "仅从已 complete 的冻结 NPZ 读取 Sinkhorn 非 dustbin、有效行列矩阵；CPU 单线程 SVD，无新推理、训练或阈值拟合。", "",
             "40 例包括 REAL 的 S6 score/布局分层、OOD 的跨模型分歧与 S6 score 四分位、SIM 正负分层；这是选择性示例，不代表总体，不能作显著性或分类因果结论。OOD 无负例且无 GT layout。", "",
             "总质量与谱形状分开：P/sum(P) 只做全矩阵质量归一化；sigma1/Fro、top-k 能量、有效秩对全局尺度不变。覆盖是矩阵索引统计，不是物理拼缝覆盖。", "",
             "下表为各组中位数。rank=exp(H(sigma/sum sigma))；PR=(sum sigma²)²/sum sigma⁴。每例完整 top-1/2/4/8/16、奇异值、行列最大值和覆盖保存在 metrics.json。", "",
             "| 选择组 | n | raw mass | sigma1/Fro | E1 | E4 | E16 | rank | energy PR | row→col覆盖 | col→row覆盖 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group, stats in result["summary"].items():
        fields = [group, str(stats["case_count"])]
        for key in SUMMARY_PATHS:
            value = stats[key]["median"]
            fields.append("NA" if value is None else f"{value:.5g}")
        lines.append("| " + " | ".join(fields) + " |")
    lines += ["", "重复分解范围：只需 S4 与 S7。D2/D4 复用同一 Matcher 时并无新的 assignment 谱；本报告未将这类重复头配置算作额外独立样本。", "",
              "输入路径：`" + result["sources"]["arm_root"] + "`。protocol/cases/每例 NPZ 的 SHA256、线程限制、矩阵/分母定义见 metrics.json。未计算相关性 p 值，也不因某谱形状宣布正/负例可分或 CrossAttention 失败。", ""]
    return "\n".join(lines)


def run(arm_root, output):
    arm_root, output = Path(arm_root).resolve(), Path(output).resolve()
    if output == arm_root or arm_root in output.parents:
        raise ValueError("output must be outside the original arm directory")
    if output.exists():
        raise FileExistsError("refusing to overwrite existing diagnostics: " + str(output))
    protocol_path, cases_path = arm_root / "protocol.json", arm_root / "cases.json"
    protocol_hash = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "complete":
        raise ValueError("arm protocol.status must be complete before any NPZ is loaded")
    cases_hash = sha256(cases_path)
    if protocol.get("cases_sha256") != cases_hash:
        raise ValueError("cases.json hash differs from completed protocol")
    cases = json.loads(cases_path.read_text())
    selected = protocol["selected_pairs"]
    keys = lambda values: [(r["dataset"], r["pair_id"]) for r in values]
    case_keys, selected_keys = keys(cases), keys(selected)
    if (not cases or len(set(case_keys)) != len(case_keys) or len(set(selected_keys)) != len(selected_keys)
            or set(case_keys) != set(selected_keys) or protocol["completed_count"] != len(cases)
            or protocol["sample_count"] != len(cases)):
        raise ValueError("inconsistent completed case population")
    selected_map = dict(zip(selected_keys, selected))
    started, rows = time.perf_counter(), []
    source_files = [(protocol_path, protocol_hash), (cases_path, cases_hash)]
    for case in cases:
        selection = selected_map[(case["dataset"], case["pair_id"])]
        if bool(case["label"]) != bool(selection["label"]):
            raise ValueError("attached label differs from frozen selection")
        npz_path = (arm_root / case["arrays_path"]).resolve()
        if arm_root not in npz_path.parents:
            raise ValueError("NPZ path escapes arm directory")
        npz_hash = sha256(npz_path)
        if npz_hash != case["arrays_sha256"]:
            raise ValueError("NPZ hash differs from completed case record")
        with np.load(npz_path, allow_pickle=False) as arrays:
            matrix, masking = masked_assignment(arrays)
        metrics = spectral_metrics(matrix)
        rows.append({"dataset": case["dataset"], "pair_id": case["pair_id"],
                     "label": bool(case["label"]), "stratum": selection["stratum"],
                     "name": selection.get("name"), "score": case["score"],
                     "decision_valid": case.get("decision_valid"),
                     "layout_gt_available": case.get("layout_gt_available"),
                     "arrays_path": str(npz_path), "arrays_sha256": npz_hash,
                     "masking": masking, "metrics": metrics})
        source_files.append((npz_path, npz_hash))
        print(json.dumps({"arm": arm_root.name, "completed": len(rows), "total": len(cases)}), flush=True)
    for path, digest in source_files:
        if sha256(path) != digest:
            raise ValueError("source changed during analysis: " + str(path))
    result = {"schema_version": SCHEMA, "status": "complete", "arm": arm_root.name,
              "case_count": len(rows), "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "sources": {"arm_root": str(arm_root), "protocol_sha256": protocol_hash,
                          "cases_sha256": cases_hash, "selection_json_sha256": protocol.get("selection_json_sha256"),
                          "model": protocol.get("model"), "script_sha256": sha256(__file__),
                          "input_hashes_unchanged_after_run": True},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "thread_environment": {key: os.environ[key] for key in THREAD_ENV},
                          "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                          "elapsed_seconds": time.perf_counter() - started, "gpu_used": False},
              "notes": NOTES, "strata_counts": dict(Counter(row["stratum"] for row in rows)),
              "summary": summarize(rows), "cases": rows}
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "SUMMARY.md").write_text(render_markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Linux: additionally pin this process to one already-allowed CPU.
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    result = run(args.arm_root, args.output)
    print(json.dumps({"status": result["status"], "arm": result["arm"],
                      "cases": result["case_count"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
