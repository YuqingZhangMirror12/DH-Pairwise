"""Copy and independently summarize already-complete M8 JSON records only."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = "/root/autodl-tmp/rachel_score_design_20260913_001/scorer_diagnosis_20260919/matcher_convergence/cpu_full_v1"
EXPECTED_SUMMARY_SHA = "1bf6434309e870379450274401c586a2f020180d801bb678f53c692fcb81c392"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_files(relative_names, destination):
    for relative in relative_names:
        if (destination / Path(relative).name).exists():
            raise FileExistsError("refusing to overwrite an existing snapshot")
    command = ["scp", "-q", "-o", "ControlPath=none", "-i",
        "/Users/yuqingzhang/.ssh/id_ed25519_rachel_benchmark_20260904", "-P", "42993",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    command += ["root@connect.westb.seetacloud.com:" + REMOTE + "/" + name for name in relative_names]
    subprocess.run(command + [str(destination) + "/"], check=True)


def distribution(rows, field):
    values = [r[field] for r in rows]
    assert all(v is not None and math.isfinite(v) for v in values)
    count = sum(v <= 20 for v in values)
    return dict(field=field, positive_count=len(rows), valid_error_count=len(values),
        mean_l2_px=statistics.mean(values), median_l2_px=statistics.median(values),
        correct_at20=count, success_rate_at20=count / len(rows), tolerance_px=20.0)


def main():
    destination = ROOT / "shared_s3_s4_s6/m08"
    destination.mkdir(parents=True, exist_ok=False)
    names = ["summary.json", "pair_metrics.jsonl", "timing_pairs.json"]
    copy_files(["shared_s3_s4_s6/m08/" + name for name in names], destination)
    copy_files(["protocol.json"], ROOT)
    assert sha(destination / "summary.json") == EXPECTED_SUMMARY_SHA
    summary = json.loads((destination / "summary.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert summary["status"] == "complete" and summary["epoch"] == 8
    assert summary["purpose"] == "full_fixed_simval" and summary["sample_count"] == 3000
    evaluation = [e for e in protocol["evaluations"] if e["arm"] == "shared_s3_s4_s6" and e["epoch"] == 8]
    assert len(evaluation) == 1 and evaluation[0]["summary_sha256"] == EXPECTED_SUMMARY_SHA
    assert sha(destination / "pair_metrics.jsonl") == summary["pair_metrics_sha256"]
    with (destination / "pair_metrics.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    assert len(rows) == 3000 and len({r["pair_id"] for r in rows}) == 3000
    assert [r["pair_id"] for r in rows] == protocol["pair_ids"]
    timing = json.loads((destination / "timing_pairs.json").read_text())
    assert [r["pair_id"] for r in timing] == [r["pair_id"] for r in rows]
    positives = [r for r in rows if r["label"]]
    assert len(positives) == 1500
    assert all(r["pose_supervised"] and r["raw_layout_valid"] for r in positives)
    soft = distribution(positives, "differentiable_translation_l2_px")
    raw = distribution(positives, "raw_translation_l2_px")
    assert all(r["raw_layout20_correct"] == (r["raw_layout_valid"] and r["raw_translation_l2_px"] <= 20)
        for r in positives)
    assert raw["correct_at20"] == 1484 == summary["metrics"]["raw_layout20"]["correct_count"]
    assert abs(soft["mean_l2_px"] - summary["metrics"]["conditional_supervision"]["differentiable_translation_l2_px_mean"]) < 1e-10
    result = dict(schema_version="matcher-m08-json-readonly-check/1", arm="shared_s3_s4_s6", epoch=8,
        observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_root=REMOTE, source_summary_sha256=EXPECTED_SUMMARY_SHA,
        source_protocol_status=protocol["status"], checkpoint_sha256=summary["checkpoint_sha256"],
        fixed_simval_sha256=summary["fixed_simval_sha256"], sample_count=3000, positive_count=1500, negative_count=1500,
        same_ordered_ids_as_protocol=True, no_training_or_inference_performed=True,
        differentiable_soft_translation=soft, final_full_top2_mode_layout=raw,
        raw_minus_soft_mean_l2_px=raw["mean_l2_px"] - soft["mean_l2_px"],
        raw_vs_soft_at20=dict(both=sum(r["raw_translation_l2_px"] <= 20 and r["differentiable_translation_l2_px"] <= 20 for r in positives),
            raw_only=sum(r["raw_translation_l2_px"] <= 20 and r["differentiable_translation_l2_px"] > 20 for r in positives),
            soft_only=sum(r["raw_translation_l2_px"] > 20 and r["differentiable_translation_l2_px"] <= 20 for r in positives),
            neither=sum(r["raw_translation_l2_px"] > 20 and r["differentiable_translation_l2_px"] > 20 for r in positives)),
        timing=summary["timing"],
        copied_file_sha256={str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "protocol.json"] + [destination / n for n in names]},
        caveats=["Single M8 checkpoint baseline only; no convergence claim or checkpoint selection.",
            "Errors are positive-only pixel L2; both distributions cover the same1500 positive pairs.",
            "Raw layout is full_top2_mode with no classification threshold gate; not soft translation_hat_rc.",
            "No M10/M12 inference, waiting or polling performed by this collection."])
    with (destination / "M08_CHECK.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("source_protocol_status", "differentiable_soft_translation",
        "final_full_top2_mode_layout", "raw_vs_soft_at20")}))


if __name__ == "__main__":
    main()
