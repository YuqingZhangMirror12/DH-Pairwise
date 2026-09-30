"""Local JSON-only summary of copied complete C8-C11 records; no model imports."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = "/root/autodl-tmp/rachel_score_design_20260913_001/"
RUNS = {"baseline_c8": REMOTE + "attention_depth_20260915/s4_cross_attention_depth2/training",
        "continuation_c9_c11": REMOTE + "scorer_diagnosis_20260919/continuation_v1/s6_d2_c16"}


def read(path):
    return json.loads(path.read_text())


def fixed_metrics(rows, threshold):
    tp = sum(bool(r["label"]) and r["classification"]["fused"] >= threshold for r in rows)
    fp = sum(not r["label"] and r["classification"]["fused"] >= threshold for r in rows)
    fn = sum(bool(r["label"]) for r in rows) - tp
    tn = len(rows) - tp - fp - fn
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=(tp + tn) / len(rows), precision=tp / (tp + fp), recall=tp / (tp + fn),
        f1=2 * tp / (2 * tp + fp + fn))


def main():
    baseline = read(ROOT / "baseline_c8/protocol.json")
    continuation = read(ROOT / "continuation_c9_c11/protocol.json")
    population = baseline["populations"]["val"]
    assert population == continuation["continuation_identity"]["populations"]["val"]
    old = read(ROOT / "baseline_c8/validation_020.json")
    thresholds = {k: old["operating_points"]["thresholds"][k] for k in ("max_f1", "recall_95")}
    baseline_rows = read(ROOT / "baseline_c8/validation_020_rows.json")
    ids_labels = [(r["pair_id"], r["label"]) for r in baseline_rows]
    epochs = []
    for epoch in (20, 21, 22, 23):
        name = "baseline_c8" if epoch == 20 else "continuation_c9_c11"
        directory = ROOT / name
        val_path = directory / ("validation_%03d.json" % epoch)
        val, rows = read(val_path), read(directory / ("validation_%03d_rows.json" % epoch))
        assert [(r["pair_id"], r["label"]) for r in rows] == ids_labels
        assert len(rows) == 3000 and sum(bool(r["label"]) for r in rows) == 1500
        assert all(r["decision_valid"] for r in rows)
        assert (val["validation"]["sample_count"], val["validation"]["positive_count"],
                val["validation"]["negative_count"]) == (3000, 1500, 1500)
        segments = [read(directory / ("segment_%03d.json" % n)) for n in range((epoch - 1) * 4 + 1, epoch * 4 + 1)]
        assert all(s["segment"]["epoch"] == epoch and s["training"]["samples"] == 6000 for s in segments)
        assert segments[-1]["segment"]["epoch_complete"]
        assert sum(s["training"]["optimizer_updates"] for s in segments) == 1500
        best = val["validation"]["methods"]["fused"]
        check = fixed_metrics(rows, best["threshold"])
        assert all(abs(check[k] - best[k]) < 1e-12 for k in check)
        epochs.append(dict(classifier_epoch=epoch - 12, absolute_epoch=epoch,
            source_run=RUNS[name], validation_file=val_path.name,
            training_samples=24000, complete_segments=4,
            train_pair_bce=sum(s["training"]["loss_components"]["fused_pair_bce"] * 6000 for s in segments) / 24000,
            val_max_f1_recorded=best,
            old_c8_fixed_thresholds_from_saved_scores={k: fixed_metrics(rows, threshold) for k, threshold in thresholds.items()}))
    result = dict(schema_version="s6-d2-continuation-readonly-snapshot/1",
        summarized_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        complete_classifier_epochs=[8, 9, 10, 11], no_training_or_inference_performed=True,
        same_ordered_pair_ids_and_labels=True, val_population=population,
        val_sample_count=3000, val_positive_count=1500, val_negative_count=1500,
        source_status_snapshot=read(ROOT / "continuation_c9_c11/status.json"),
        methods=dict(train_pair_bce="sample-weighted mean of four complete 6000-pair segments",
            val_max_f1="existing per-epoch SIMVAL max-F1 threshold; no new threshold fit",
            auprc_auroc="copied from existing validation.methods.fused",
            fixed_old_thresholds="confusion counts calculated locally from already saved scores at C8 thresholds; no inference"),
        source_run_directories=RUNS, epochs=epochs,
        caveats=["C16 unfinished at snapshot; epoch24 partial training excluded.",
                 "SIMVAL-only preliminary single-seed trend, not proof of REAL/OOD improvement.",
                 "Matcher remains frozen; classification continuation cannot change raw layouts."],
        raw_file_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for name in RUNS for p in sorted((ROOT / name).glob("*.json"))})
    path = ROOT / "summary.json"
    with path.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(path=str(path), epochs=epochs), ensure_ascii=False))


if __name__ == "__main__":
    main()
