"""One read-only C13 collection, adapted from v2; preserve all earlier snapshots."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parent
V1, V2 = (ROOT.parent / ("continuation_snapshot_v" + n) for n in ("1", "2"))
REMOTE = "/root/autodl-tmp/rachel_score_design_20260913_001/scorer_diagnosis_20260919/continuation_v1/s6_d2_c16"
KEY = "/Users/yuqingzhang/.ssh/id_ed25519_rachel_benchmark_20260904"
HOST = "root@connect.westb.seetacloud.com"
PYTHON = "/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python"


def main():
    raw = ROOT / "continuation_c13"
    raw.mkdir(exist_ok=False)
    names = ["protocol.json", "status.json", "validation_025.json", "validation_025_rows.json"]
    names += ["segment_%03d.json" % n for n in range(97, 101)]
    command = ["scp", "-q", "-o", "ControlPath=none", "-i", KEY, "-P", "42993",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    command += [HOST + ":" + REMOTE + "/" + name for name in names]
    subprocess.run(command + [str(raw) + "/"], check=True)
    probe = '''import datetime,json
from pathlib import Path
p=Path('/proc/99022')
process=None
if p.exists():
 fields=(p/'stat').read_text().rsplit(')',1)[1].split()
 process=dict(pid=99022,state=fields[0],startticks=int(fields[19]),cmdline=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode())
print(json.dumps(dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),process=process)))
'''
    snapshot = json.loads(subprocess.run(["ssh", "-S", "none", "-i", KEY, "-p", "42993",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, PYTHON + " -"],
        input=probe, text=True, capture_output=True, check=True).stdout)
    with (ROOT / "trainer_snapshot.json").open("x") as stream:
        json.dump(snapshot, stream, indent=2)
        stream.write("\n")
    helpers = runpy.run_path(str(V1 / "summarize_snapshot.py"))
    read, fixed_metrics = helpers["read"], helpers["fixed_metrics"]
    original = read(V2 / "epoch_metrics.json")
    baseline = read(V1 / "baseline_c8/protocol.json")
    protocol = read(raw / "protocol.json")
    population = baseline["populations"]["val"]
    assert population == protocol["continuation_identity"]["populations"]["val"]
    old = read(V1 / "baseline_c8/validation_020.json")
    assert old["operating_points"]["thresholds"]["max_f1"] == 0.6123367547988892
    assert old["operating_points"]["thresholds"]["recall_95"] == 0.4257810115814209
    rows = read(raw / "validation_025_rows.json")
    ids_labels = [(r["pair_id"], r["label"]) for r in rows]
    for previous in (V1 / "baseline_c8/validation_020_rows.json", V2 / "continuation_c12/validation_024_rows.json"):
        assert ids_labels == [(r["pair_id"], r["label"]) for r in read(previous)]
    assert len(rows) == 3000 and len({r["pair_id"] for r in rows}) == 3000
    assert sum(bool(r["label"]) for r in rows) == 1500 and all(r["decision_valid"] for r in rows)
    val = read(raw / "validation_025.json")
    assert val["epoch"] == 25
    assert (val["validation"]["sample_count"], val["validation"]["positive_count"],
            val["validation"]["negative_count"]) == (3000, 1500, 1500)
    segments = [read(raw / ("segment_%03d.json" % n)) for n in range(97, 101)]
    assert all(s["segment"]["epoch"] == 25 and s["training"]["samples"] == 6000 for s in segments)
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
    latest = dict(classifier_epoch=13, absolute_epoch=25,
        train_pair_bce=sum(s["training"]["loss_components"]["fused_pair_bce"] * 6000 for s in segments) / 24000,
        val_pair_bce=bce / len(rows), val_auroc=best["auroc"], val_auprc=best["auprc"],
        val_max_f1=best["f1"], val_max_f1_threshold=best["threshold"],
        fixed_c8_max_f1_threshold=fixed_metrics(rows, old["operating_points"]["thresholds"]["max_f1"]),
        fixed_c8_r95_threshold=fixed_metrics(rows, old["operating_points"]["thresholds"]["recall_95"]),
        train_samples=24000, val_samples=3000, val_positive_count=1500, val_negative_count=1500,
        saved_probability_source="continuation_c13/validation_025_rows.json")
    prior = original["epochs"][-1]
    assert prior["classifier_epoch"] == 12
    delta = {key: latest[key] - prior[key] for key in (
        "train_pair_bce", "val_pair_bce", "val_auroc", "val_auprc", "val_max_f1")}
    for threshold in ("fixed_c8_max_f1_threshold", "fixed_c8_r95_threshold"):
        delta[threshold] = {key: latest[threshold][key] - prior[threshold][key]
            for key in ("tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1")}
    epochs = original["epochs"]
    for record in epochs:
        source = (V2 / record["saved_probability_source"]).resolve()
        record["saved_probability_source"] = "../" + str(source.relative_to(ROOT.parent))
    epochs.append(latest)
    result = dict(schema_version="s6-d2-continuation-compact-epochs/3",
        observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_run=REMOTE, earlier_epochs_source="../continuation_snapshot_v2/epoch_metrics.json",
        population_check="C13 has identical manifest SHA and ordered pair IDs/labels to C8 and C12; earlier rounds verified in v1/v2",
        val_manifest_sha256=population["manifest_sha256"], val_bce_method=original["val_bce_method"],
        train_bce_method=original["train_bce_method"], ranking_metrics=original["ranking_metrics"],
        source_status_snapshot=read(raw / "status.json"), live_trainer_snapshot=snapshot,
        no_training_or_inference_performed=True, epochs=epochs, c13_minus_c12=delta,
        raw_file_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(raw.glob("*.json"))},
        v2_summary_sha256=hashlib.sha256((V2 / "epoch_metrics.json").read_bytes()).hexdigest(),
        caveats=["Only complete C8-C13 epochs shown; no wait for C14/C16.",
                 "SIMVAL-only single-seed preliminary trend, not a convergence or REAL/OOD claim.",
                 "Frozen Matcher; classification continuation does not change raw layouts."])
    with (ROOT / "epoch_metrics.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(output=str(ROOT / "epoch_metrics.json"), latest=latest,
        c13_minus_c12=delta, source_status_snapshot=result["source_status_snapshot"], trainer=snapshot)))


if __name__ == "__main__":
    main()
