"""Compact trend input from the saved snapshot, without inference or threshold fitting."""
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / "summary.json").read_text())
metrics = []
for record in summary["epochs"]:
    epoch = record["absolute_epoch"]
    directory = "baseline_c8" if epoch == 20 else "continuation_c9_c11"
    source = root / directory / ("validation_%03d_rows.json" % epoch)
    rows = json.loads(source.read_text())
    bce = 0.
    for row in rows:
        y = int(row["label"])
        p = max(1e-12, min(1 - 1e-12, float(row["classification"]["fused"])))
        bce -= y * math.log(p) + (1 - y) * math.log1p(-p)
    best = record["val_max_f1_recorded"]
    fixed = record["old_c8_fixed_thresholds_from_saved_scores"]["max_f1"]
    metrics.append(dict(classifier_epoch=record["classifier_epoch"], absolute_epoch=epoch,
        train_pair_bce=record["train_pair_bce"], val_pair_bce=bce / len(rows),
        val_auroc=best["auroc"], val_auprc=best["auprc"], val_max_f1=best["f1"],
        val_max_f1_threshold=best["threshold"], fixed_c8_max_f1_threshold=fixed,
        fixed_c8_r95_threshold=record["old_c8_fixed_thresholds_from_saved_scores"]["recall_95"],
        train_samples=24000, val_samples=len(rows), val_positive_count=1500, val_negative_count=1500,
        saved_probability_source=str(source.relative_to(root))))
result = dict(schema_version="s6-d2-continuation-compact-epochs/1", source="summary.json",
    population_check="identical manifest SHA and identical ordered pair IDs/labels across all4 epochs",
    val_manifest_sha256=summary["val_population"]["manifest_sha256"],
    val_bce_method="mean PairBCE from saved fused probabilities; epsilon1e-12; no model inference",
    source_status_snapshot=summary["source_status_snapshot"], epochs=metrics,
    caveats=summary["caveats"])
output = root / "epoch_metrics.json"
with output.open("x") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps(dict(output=str(output), epochs=[{k: r[k] for k in (
    "classifier_epoch", "train_pair_bce", "val_pair_bce", "val_auroc", "val_auprc", "val_max_f1",
    "fixed_c8_max_f1_threshold")} for r in metrics])))
