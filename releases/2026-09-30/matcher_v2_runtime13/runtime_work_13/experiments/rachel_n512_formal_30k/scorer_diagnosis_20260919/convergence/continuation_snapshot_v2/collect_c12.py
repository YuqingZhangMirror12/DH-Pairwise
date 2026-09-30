"""Fetch the confirmed C12 JSON-only result once; retain v1 unchanged."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "continuation_snapshot_v1"
REMOTE = "/root/autodl-tmp/rachel_score_design_20260913_001/scorer_diagnosis_20260919/continuation_v1/s6_d2_c16"


def main():
    raw = ROOT / "continuation_c12"
    raw.mkdir(exist_ok=False)
    names = ["protocol.json", "status.json", "validation_024.json", "validation_024_rows.json"]
    names += ["segment_%03d.json" % n for n in range(93, 97)]
    command = ["scp", "-q", "-o", "ControlPath=none", "-i",
        "/Users/yuqingzhang/.ssh/id_ed25519_rachel_benchmark_20260904", "-P", "42993",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    command += ["root@connect.westb.seetacloud.com:" + REMOTE + "/" + name for name in names]
    subprocess.run(command + [str(raw) + "/"], check=True)
    # Reuse v1's pure JSON reader and fixed-threshold confusion calculation.
    helpers = runpy.run_path(str(V1 / "summarize_snapshot.py"))
    read, fixed_metrics = helpers["read"], helpers["fixed_metrics"]
    original = read(V1 / "epoch_metrics.json")
    baseline = read(V1 / "baseline_c8/protocol.json")
    protocol = read(raw / "protocol.json")
    population = baseline["populations"]["val"]
    assert population == protocol["continuation_identity"]["populations"]["val"]
    old = read(V1 / "baseline_c8/validation_020.json")
    baseline_rows = read(V1 / "baseline_c8/validation_020_rows.json")
    rows = read(raw / "validation_024_rows.json")
    assert [(r["pair_id"], r["label"]) for r in rows] == [(r["pair_id"], r["label"]) for r in baseline_rows]
    assert len(rows) == 3000 and sum(bool(r["label"]) for r in rows) == 1500
    assert all(r["decision_valid"] for r in rows)
    val = read(raw / "validation_024.json")
    assert val["epoch"] == 24
    assert (val["validation"]["sample_count"], val["validation"]["positive_count"],
            val["validation"]["negative_count"]) == (3000, 1500, 1500)
    segments = [read(raw / ("segment_%03d.json" % n)) for n in range(93, 97)]
    assert all(s["segment"]["epoch"] == 24 and s["training"]["samples"] == 6000 for s in segments)
    assert segments[-1]["segment"]["epoch_complete"]
    assert sum(s["training"]["optimizer_updates"] for s in segments) == 1500
    best = val["validation"]["methods"]["fused"]
    check = fixed_metrics(rows, best["threshold"])
    assert all(abs(check[k] - best[k]) < 1e-12 for k in check)
    bce = 0.
    for row in rows:
        y = int(row["label"])
        p = max(1e-12, min(1 - 1e-12, float(row["classification"]["fused"])))
        bce -= y * math.log(p) + (1 - y) * math.log1p(-p)
    latest = dict(classifier_epoch=12, absolute_epoch=24,
        train_pair_bce=sum(s["training"]["loss_components"]["fused_pair_bce"] * 6000 for s in segments) / 24000,
        val_pair_bce=bce / len(rows), val_auroc=best["auroc"], val_auprc=best["auprc"],
        val_max_f1=best["f1"], val_max_f1_threshold=best["threshold"],
        fixed_c8_max_f1_threshold=fixed_metrics(rows, old["operating_points"]["thresholds"]["max_f1"]),
        fixed_c8_r95_threshold=fixed_metrics(rows, old["operating_points"]["thresholds"]["recall_95"]),
        train_samples=24000, val_samples=3000, val_positive_count=1500, val_negative_count=1500,
        saved_probability_source="continuation_c12/validation_024_rows.json")
    epochs = original["epochs"]
    for record in epochs:
        record["saved_probability_source"] = "../continuation_snapshot_v1/" + record["saved_probability_source"]
    epochs.append(latest)
    result = dict(schema_version="s6-d2-continuation-compact-epochs/2",
        observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_run=REMOTE, earlier_epochs_source="../continuation_snapshot_v1/epoch_metrics.json",
        population_check="identical manifest SHA and ordered pair IDs/labels; C12 against C8, C8-C11 verified in v1",
        val_manifest_sha256=population["manifest_sha256"], val_bce_method=original["val_bce_method"],
        train_bce_method="sample-weighted mean of four complete 6000-pair training segments",
        ranking_metrics="existing validation.methods.fused values, no new inference or threshold fit",
        source_status_snapshot=read(raw / "status.json"), no_training_or_inference_performed=True,
        epochs=epochs, raw_file_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(raw.glob("*.json"))},
        v1_summary_sha256=hashlib.sha256((V1 / "epoch_metrics.json").read_bytes()).hexdigest(),
        caveats=["Only complete C8-C12 epochs shown; C16 unfinished, no wait for C13.",
                 "SIMVAL-only single-seed preliminary trend does not establish REAL/OOD improvement.",
                 "Frozen Matcher; classification continuation does not change raw layouts."])
    with (ROOT / "epoch_metrics.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(output=str(ROOT / "epoch_metrics.json"), latest=latest,
        source_status_snapshot=result["source_status_snapshot"])))


if __name__ == "__main__":
    main()
