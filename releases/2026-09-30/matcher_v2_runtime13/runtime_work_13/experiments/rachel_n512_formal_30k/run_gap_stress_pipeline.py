"""Finite post-training TEST stress evaluation; never restarts a training job.

Await the already-running E0--E3 pipeline, then evaluate the same fixed TEST
pairs at three preregistered erosion severities. E3 shares E2 forward passes.
This is a one-off dependency queue, not a recurring monitor or model search.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys

from experiments.rachel_n512_formal_30k.run_all_weathering_ablations import (
    now, read_json, save_json,
)

DEPTHS = (0, 2, 4)
MODELS = ("e0", "e1", "e2")
DECODER = "full_top2_mode"
EXPECTED_PREDECESSOR_STAGES = (
    "train_e2", "evaluate_e2_test", "evaluate_e2_real",
    "fit_e3_and_evaluate", "summarize_e0_e1_e2_e3",
)


def require_complete_predecessor(root):
    root = Path(root)
    state = read_json(root / "pipeline_state.json")
    stages = state.get("stages", [])
    if (state.get("status") != "complete"
            or tuple(x.get("name") for x in stages) != EXPECTED_PREDECESSOR_STAGES
            or any(x.get("status") != "complete" or x.get("returncode") != 0 for x in stages)):
        raise RuntimeError("Existing E0--E3 pipeline must complete every required stage")
    if read_json(root / "summary" / "summary.json").get("complete") is not True:
        raise RuntimeError("Existing E0--E3 summary is not complete")
    return state


def expected_live_process(pid, root):
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().replace("\0", " ")
    except FileNotFoundError:
        return False
    return "run_all_weathering_ablations" in command and str(Path(root).resolve()) in command


def wait_for_predecessor(root):
    state = read_json(Path(root) / "pipeline_state.json")
    if state.get("status") == "complete":
        return require_complete_predecessor(root)
    if state.get("status") not in ("running", "waiting_for_e1"):
        raise RuntimeError("Existing pipeline is not running: " + str(state.get("status")))
    pid = state.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        raise RuntimeError("An exact predecessor PID is required")
    if not expected_live_process(pid, root):
        return require_complete_predecessor(root)
    if not hasattr(os, "pidfd_open"):
        subprocess.run(["tail", "--pid=" + str(pid), "--sleep-interval=10", "-f", "/dev/null"], check=True)
    else:
        try:
            descriptor = os.pidfd_open(pid)
        except ProcessLookupError:
            return require_complete_predecessor(root)
        try:
            if not expected_live_process(pid, root):
                return require_complete_predecessor(root)
            select.select([descriptor], [], [])
        finally:
            os.close(descriptor)
    return require_complete_predecessor(root)


def stage_plan(args):
    root, all_root = Path(args.root).resolve(), Path(args.all_root).resolve()
    training = {"e0": args.e0_training_run, "e1": args.e1_training_run,
                "e2": str(all_root / "e2" / "training" / "e2")}
    stages = []
    for model in MODELS:
        for depth in DEPTHS:
            name = model + "_depth" + str(depth)
            output = root / "evaluation" / model / ("depth" + str(depth))
            command = ["--training-run", training[model], "--model-label", model,
                "--dataset", args.dataset, "--depth", str(depth),
                "--output", str(output), "--cache-dir", str(root / "augmentation_cache"),
                "--device", "cuda:0", "--workers", str(args.workers)]
            if model == "e2":
                command += ["--e3-head-root", str(all_root / "e3")]
            stages.append(dict(name=name, status="queued", model_label=model, depth=depth,
                module="experiments.rachel_n512_formal_30k.evaluate_gap_stress",
                arguments=command, output=str(output)))
    return stages


def require_full_result(stage):
    root = Path(stage["output"])
    receipt, summary = read_json(root / "receipt.json"), read_json(root / "summary.json")
    if (receipt.get("status") != "complete" or summary.get("status") != "complete"
            or receipt.get("full_test") is not True
            or receipt.get("evaluated_pair_count") != 3000
            or receipt.get("total_test_pair_count") != 3000
            or receipt.get("model_label") != stage["model_label"]
            or receipt.get("depths") != [stage["depth"]]
            or summary.get("sample_count") != 3000 or summary.get("positive_count") != 1500):
        raise RuntimeError("A probe, partial run, or different population is not a full stress result")
    if not (root / "pair_results.jsonl").is_file():
        raise RuntimeError("Completed stress result requires per-pair predictions")
    return summary


def summarize_results(stages, output):
    """Keep native fused, the fixed geometry head, and its control separate."""
    rows, population = [], None
    expected = {(model, depth) for model in MODELS for depth in DEPTHS}
    if {(s["model_label"], s["depth"]) for s in stages} != expected or len(stages) != 9:
        raise RuntimeError("All nine model/depth evaluations are required")
    for stage in stages:
        result = require_full_result(stage)
        pair_path = Path(stage["output"]) / "pair_results.jsonl"
        with pair_path.open() as stream:
            pairs = [json.loads(line) for line in stream if line.strip()]
        ids = [(p["pair_id"], bool(p["label"])) for p in pairs]
        if (len(ids) != 3000 or sum(p[1] for p in ids) != 1500
                or len({p[0] for p in ids}) != 3000):
            raise RuntimeError("Stress result rows do not retain full TEST population")
        if population is None:
            population = ids
        elif ids != population:
            raise RuntimeError("Model/depth comparison must use identical ordered TEST pairs")
        method_names = ("native_fused", "score_geometry", "score_only") if stage["model_label"] == "e2" else ("native_fused",)
        for method_name in method_names:
            method = result["methods"][method_name]
            subset = result.get("actual_changed_subset", {})
            rows.append(dict(model=stage["model_label"], method=method_name, depth=stage["depth"],
                threshold=method["threshold"], classification=method["classification"],
                joint=method["joint"], layout=result["layout"][DECODER],
                changed_subset=subset.get("methods", {}).get(method_name),
                changed_subset_positive_count=subset.get("positive_count"),
                changed_subset_layout=(subset.get("layout") or {}).get(DECODER),
                stress=result.get("stress"), closing_direction_bias=result.get("closing_direction_bias"),
                mask_overlap=result.get("mask_overlap"),
                sources={name: str((Path(stage["output"]) / name).resolve())
                         for name in ("summary.json", "receipt.json", "pair_results.jsonl")}))
    for row in rows:
        clean = next(r for r in rows if (r["model"], r["method"], r["depth"]) == (row["model"], row["method"], 0))
        row["change_from_same_model_clean"] = dict(
            pair_recall=row["classification"]["recall"] - clean["classification"]["recall"],
            layout_recall10=row["layout"]["recall"]["10"] - clean["layout"]["recall"]["10"],
            joint10_tp=row["joint"]["10"]["tp"] - clean["joint"]["10"]["tp"])
    result = dict(schema_version="rachel-gap-stress-comparison/1", status="complete", complete=True,
        completed_at=now(), sample_count=3000, positive_count=1500, rows=rows,
        limitations=["Synthetic derived TEST stress test, not a new real-world holdout.",
            "Requested erosion depth is not measured gap width; actual changes/skips are reported.",
            "Closing-direction bias decomposes GT translation error; it is not gap-width accuracy.",
            "E3 heads share E2 layout; only pair acceptance can differ.",
            "All thresholds remain frozen on clean VAL; no TEST/REAL refitting or winner selection."])
    output = Path(output)
    save_json(output / "summary.json", result)
    lines = ["# 腐蚀间隙专项测试结果", "", "原 TEST 3,000 对（正例 1,500），全部条件、模型使用相同样本；阈值保持原 VAL 冻结。", "",
        "| 模型／分类头 | 最大腐蚀 px | Precision | Recall | F1 | AP | 布局 R10 | 接受且摆对 TP10 | 对干净 ΔTP10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        metric = row["classification"]
        values = [metric.get(k) for k in ("precision", "recall", "f1", "auprc")]
        values.append(row["layout"]["recall"]["10"])
        numbers = ["—" if x is None else "{:.2f}%".format(100 * x) for x in values]
        lines.append("| {} / {} | {} | {} | {} | {:+d} |".format(row["model"], row["method"], row["depth"],
            " | ".join(numbers), row["joint"]["10"]["tp"], row["change_from_same_model_clean"]["joint10_tp"]))
    lines += ["", "## 解释边界", ""] + ["- " + x for x in result["limitations"]]
    lines += ["", "实际受损子集、闭缝方向偏差与重叠诊断保存在同目录 summary.json；每项均带原始结果路径。", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def run(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "pipeline_state.json"
    if state_path.exists():
        raise FileExistsError("Refusing to duplicate an existing gap-stress pipeline")
    preflight = read_json(root / "preflight" / "e0_depth2" / "receipt.json")
    if (preflight.get("status") != "complete" or preflight.get("full_test") is not False
            or not preflight.get("evaluated_pair_count") or preflight.get("model_label") != "e0"
            or preflight.get("depths") != [2]):
        raise RuntimeError("A bounded E0/depth2 inference probe must pass before queueing")
    source = Path(__file__).resolve().parents[2]
    state = dict(schema_version="rachel-gap-stress-pipeline/1", status="waiting_for_e0_e3",
        started_at=now(), pid=os.getpid(), source=str(source), arguments=vars(args),
        active_stage="waiting_for_e0_e3", predecessor=str(Path(args.all_root) / "pipeline_state.json"),
        stages=stage_plan(args), adds_training=False, original_inputs_modified=False,
        model_or_threshold_selection_on_test_or_real=False)
    save_json(state_path, state)
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        predecessor = wait_for_predecessor(args.all_root)
        state.update(status="running", predecessor_completed_at=predecessor.get("completed_at"))
        save_json(state_path, state)
        for stage in state["stages"]:
            command = [args.python, "-u", "-m", stage["module"]] + stage["arguments"]
            stage.update(status="running", started_at=now(), command=command,
                log=str(root / (stage["name"] + ".log")))
            state["active_stage"] = stage["name"]
            save_json(state_path, state)
            with Path(stage["log"]).open("x", encoding="utf-8") as stream:
                child = subprocess.Popen(command, cwd=str(source), env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                stage["pid"] = child.pid
                save_json(state_path, state)
                code = child.wait()
            stage.update(status="complete" if code == 0 else "failed", returncode=code, completed_at=now())
            save_json(state_path, state)
            if code:
                raise RuntimeError(stage["name"] + " failed; see " + stage["log"])
            require_full_result(stage)
        state["active_stage"] = "summarize_gap_stress"
        save_json(state_path, state)
        summarize_results(state["stages"], root / "summary")
        state.update(status="complete", completed_at=now(), active_stage=None)
        save_json(state_path, state)
    except Exception as error:
        state.update(status="failed", failed_at=now(), error=repr(error))
        save_json(state_path, state)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "all-root", "e0-training-run", "e1-training-run", "dataset"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--workers", type=int, default=4)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
