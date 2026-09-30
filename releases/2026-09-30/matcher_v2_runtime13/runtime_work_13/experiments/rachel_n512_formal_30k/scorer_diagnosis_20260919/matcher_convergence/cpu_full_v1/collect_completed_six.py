"""Read-only remote JSON collection and independent six-checkpoint recount.

No model imports or remote writes; original shared M8 and running protocol
snapshots remain byte-identical. Completed protocol receives its own directory.
"""
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import math
from pathlib import Path
import runpy
import statistics

ROOT = Path(__file__).resolve().parent
EXPECTED_PROTOCOL_SHA = "017dbfd494555dfd42164444f577eb7a463e294838d95fcbf2a871c537eaa8ad"
ARMS, EPOCHS = ("shared_s3_s4_s6", "s7"), (8, 10, 12)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error_stats(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    assert all(math.isfinite(v) for v in values)
    return dict(field=field, positive_count=len(rows), finite_error_count=len(values),
        missing_error_count=len(rows) - len(values),
        mean_l2_px=statistics.mean(values) if values else None,
        median_l2_px=statistics.median(values) if values else None,
        at10_correct=sum(v <= 10 for v in values), at20_correct=sum(v <= 20 for v in values),
        at10_rate=sum(v <= 10 for v in values) / len(rows),
        at20_rate=sum(v <= 20 for v in values) / len(rows),
        threshold_denominator="all1500 GT positives; missing/invalid counts as failure")


def loss_means(rows):
    return {key: statistics.mean(r["losses"][key] for r in rows) for key in rows[0]["losses"]}


def successes(rows, field, threshold):
    return [r[field] is not None and r[field] <= threshold for r in rows]


def transition(before, after, field, threshold):
    left, right = successes(before, field, threshold), successes(after, field, threshold)
    result = dict(tolerance_px=threshold, both_correct=sum(a and b for a, b in zip(left, right)),
        gained=sum(not a and b for a, b in zip(left, right)),
        lost=sum(a and not b for a, b in zip(left, right)),
        both_failed=sum(not a and not b for a, b in zip(left, right)))
    result["net_gain"] = result["gained"] - result["lost"]
    assert sum(result[k] for k in ("both_correct", "gained", "lost", "both_failed")) == 1500
    return result


def main():
    helper = runpy.run_path(str(ROOT / "collect_m08.py"))
    preserved = [ROOT / "protocol.json"] + list((ROOT / "shared_s3_s4_s6/m08").iterdir())
    before = {str(p): sha(p) for p in preserved if p.is_file()}
    receipt = ROOT / "completed_receipt"
    receipt.mkdir(exist_ok=False)
    helper["copy_files"](["protocol.json"], receipt)
    assert sha(receipt / "protocol.json") == EXPECTED_PROTOCOL_SHA
    protocol = read(receipt / "protocol.json")
    assert protocol["status"] == "complete" and protocol["completed_evaluations"] == 6
    assert {(e["arm"], e["epoch"]) for e in protocol["evaluations"]} == {(a, e) for a in ARMS for e in EPOCHS}
    names = ("summary.json", "pair_metrics.jsonl", "timing_pairs.json")
    def collect(key):
        arm, epoch = key
        if key == ("shared_s3_s4_s6", 8):
            return
        relative = arm + "/m%02d/" % epoch
        destination = ROOT / relative
        destination.mkdir(parents=True, exist_ok=False)
        helper["copy_files"]([relative + name for name in names], destination)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(collect, [(a, e) for a in ARMS for e in EPOCHS]))
    assert {str(p): sha(p) for p in preserved if p.is_file()} == before
    outputs, pairs = [], {}
    first_identity = first_gt_counts = None
    for arm in ARMS:
        for epoch in EPOCHS:
            directory = ROOT / arm / ("m%02d" % epoch)
            summary = read(directory / "summary.json")
            entry = next(e for e in protocol["evaluations"] if (e["arm"], e["epoch"]) == (arm, epoch))
            assert sha(directory / "summary.json") == entry["summary_sha256"]
            assert sha(directory / "pair_metrics.jsonl") == summary["pair_metrics_sha256"]
            assert summary["status"] == "complete" and summary["purpose"] == "full_fixed_simval"
            assert summary["sample_count"] == 3000 and summary["frozen_matcher_unchanged"]
            with (directory / "pair_metrics.jsonl").open() as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            assert len(rows) == 3000 and len({r["pair_id"] for r in rows}) == 3000
            assert [r["pair_id"] for r in rows] == protocol["pair_ids"]
            timing = read(directory / "timing_pairs.json")
            assert [r["pair_id"] for r in timing] == protocol["pair_ids"]
            identity = [(r["pair_id"], r["label"]) for r in rows]
            gt_counts = [(r["supervised_correspondence_count"], r["pose_supervised"]) for r in rows]
            if first_identity is None:
                first_identity, first_gt_counts = identity, gt_counts
            assert identity == first_identity and gt_counts == first_gt_counts
            positives, negatives = [r for r in rows if r["label"]], [r for r in rows if not r["label"]]
            assert len(positives) == len(negatives) == 1500
            assert all(r["training_valid"] and r["decision_valid"] for r in rows)
            assert all(r["pose_supervised"] and r["raw_layout_valid"] for r in positives)
            means = {"all_pairs": loss_means(rows), "positive_pairs": loss_means(positives), "negative_pairs": loss_means(negatives)}
            for group, values in means.items():
                assert all(abs(value - summary["metrics"][group]["mean_losses"][term]) < 1e-10 for term, value in values.items())
            matched = [r for r in rows if r["supervised_correspondence_count"] > 0]
            counts = dict(correspondence_pair_count=len(matched),
                correspondence_token_count=sum(r["supervised_correspondence_count"] for r in rows),
                pose_pair_count=sum(r["pose_supervised"] for r in rows),
                positive_count=1500, negative_count=1500)
            conditional = summary["metrics"]["conditional_supervision"]
            assert all(counts[k] == conditional[k] for k in ("correspondence_pair_count", "correspondence_token_count", "pose_pair_count"))
            soft = error_stats(positives, "differentiable_translation_l2_px")
            raw = error_stats(positives, "raw_translation_l2_px")
            assert raw["at20_correct"] == summary["metrics"]["raw_layout20"]["correct_count"]
            assert all(r["raw_layout20_correct"] == (r["raw_layout_valid"] and r["raw_translation_l2_px"] <= 20) for r in positives)
            assert abs(soft["mean_l2_px"] - conditional["differentiable_translation_l2_px_mean"]) < 1e-10
            outputs.append(dict(arm=arm, epoch=epoch, sample_count=3000, gt_counts=counts,
                summary=str((directory / "summary.json").relative_to(ROOT)),
                summary_sha256=entry["summary_sha256"], checkpoint_sha256=summary["checkpoint_sha256"],
                matcher_state_sha256=summary["matcher_state_sha256"], fixed_simval_sha256=summary["fixed_simval_sha256"],
                soft_translation=soft, final_mode_layout=raw, mean_losses=means,
                correspondence_nll_per_supervised_pair=statistics.mean(r["losses"]["match_nll"] for r in matched),
                translation_smooth_l1_per_supervised_pair=statistics.mean(r["losses"]["translation_smooth_l1"] for r in positives),
                elapsed_loop_seconds=summary["elapsed_loop_seconds"],
                copied_file_sha256={str((directory / name).relative_to(ROOT)): sha(directory / name) for name in names}))
            pairs[(arm, epoch)] = positives
    assert len({r["fixed_simval_sha256"] for r in outputs}) == 1
    changes = []
    for arm in ARMS:
        for first, last in ((8, 10), (10, 12), (8, 12)):
            changes.append(dict(arm=arm, from_epoch=first, to_epoch=last,
                soft_translation={"at%d" % t: transition(pairs[(arm, first)], pairs[(arm, last)], "differentiable_translation_l2_px", t) for t in (10, 20)},
                final_mode_layout={"at%d" % t: transition(pairs[(arm, first)], pairs[(arm, last)], "raw_translation_l2_px", t) for t in (10, 20)}))
    result = dict(schema_version="matcher-six-checkpoint-readonly-analysis/1",
        observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_root=helper["REMOTE"], complete_protocol="completed_receipt/protocol.json",
        complete_protocol_sha256=EXPECTED_PROTOCOL_SHA, completed_evaluations=6,
        no_training_or_inference_performed=True, original_m8_and_running_protocol_preserved=True,
        identical_ordered_ids_labels_and_gt_counts=True, sample_count_each=3000, positive_count_each=1500,
        negative_count_each=1500, fixed_simval_sha256=outputs[0]["fixed_simval_sha256"],
        checkpoints=outputs, paired_checkpoint_changes=changes,
        definitions=dict(layout_success="positive-only L2<=10 or20 pixels, no classifier threshold; all GT positives denominator",
            error_mean_median="over finite positive errors; all six observed runs have all1500 valid errors",
            soft_translation="existing differentiable translation_hat_rc error, not final full_top2_mode output",
            assignment_nll="existing per-pair loss; all/positive/negative means independently recomputed from saved terms",
            correspondence_token_count="sum of evaluator supervised_correspondence_count, not a count of independent examples"),
        caveats=["Only M8/M10/M12, one existing run per arm; cannot establish full convergence or causality.",
            "Full clean SIMVAL only; no REAL/OOD inference, threshold selection or checkpoint selection.",
            "Soft and final decoder estimates differ; declining soft loss need not improve final layout monotonically."])
    with (ROOT / "SIX_CHECKPOINT_ANALYSIS.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(output=str(ROOT / "SIX_CHECKPOINT_ANALYSIS.json"), checkpoints=[dict(
        arm=r["arm"], epoch=r["epoch"], soft=r["soft_translation"], final=r["final_mode_layout"],
        nll_all=r["mean_losses"]["all_pairs"]["assignment_nll"],
        nll_positive=r["mean_losses"]["positive_pairs"]["assignment_nll"],
        gt_counts=r["gt_counts"]) for r in outputs], paired_changes=changes)))


if __name__ == "__main__":
    main()
