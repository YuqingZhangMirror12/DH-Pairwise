"""Read saved confirmation summaries; compare each candidate to the same seed.

No models, raw pairs, GT, threshold fitting, decoder selection, or new tests of
statistical significance are performed. Missing reports stay pending. This
derived summary does not establish whether an absent remote job is running.
"""
import argparse
import json
from pathlib import Path


SEEDS = (260908, 260909)
POPULATIONS = ("val", "test", "real_balanced1016", "real_strict547")
COUNTS = {"val": (3000, 1500), "test": (3000, 1500),
          "real_balanced1016": (1016, 508), "real_strict547": (547, 508)}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def extract_summary(path, freeze, seed, population):
    result = dict(status="pending", source=str(path), metrics=None, issues=[])
    if not Path(path).exists() or freeze is None:
        return result
    try:
        summary = read_json(path)
        decoder = freeze["selected_full_decoder"]
        if (freeze["seed"] != seed or freeze["source_split"] != "validation"
                or freeze["test_or_real_used_for_fit"] is not False):
            raise ValueError("freeze must belong to this seed and validation only")
        split = "real" if population.startswith("real_") else population
        if summary["status"] != "complete" or summary["split"] != split:
            raise ValueError("summary is not a completed expected split")
        for key in ("selected_full_decoder", "checkpoint_sha256", "precision"):
            if summary[key] != freeze[key]:
                raise ValueError("summary differs from its own freeze: " + key)
        selected = summary["strict_summary"] if population == "real_strict547" else summary
        n, positives = COUNTS[population]
        if (selected["sample_count"], selected["positive_count"]) != (n, positives):
            raise ValueError("unexpected population denominators")
        pair = selected["classification"]["fused"]["at_original_frozen_threshold"]
        if pair["threshold"] != freeze["original_fused_threshold"]:
            raise ValueError("pairing threshold differs from own validation freeze")
        if sum(pair[key] for key in ("tp", "fp", "fn", "tn")) != n or pair["tp"] + pair["fn"] != positives:
            raise ValueError("pairing counts disagree with population")
        layout = selected["layout"][decoder]
        metrics = {"pair_" + key: pair[key] for key in (
            "f1", "auroc", "auprc", "precision", "recall", "tp", "fp", "fn")}
        metrics.update(pose_coverage=layout["positive_pose_coverage"],
                       pose_median_px_conditional=layout["median_px_conditional"],
                       pose_p90_px_conditional=layout["p90_px_conditional"])
        for cutoff in (2, 5, 8, 10):
            recall = layout["recall"][str(cutoff)]
            count = recall * positives
            if not 0 <= recall <= 1 or abs(count - round(count)) > 1e-6:
                raise ValueError("pose recall does not recover an integer success count")
            metrics["pose_r%d" % cutoff] = recall
            metrics["pose_n%d" % cutoff] = round(count)
        metrics.update({"joint10_" + key: layout["assembly"]["10"][key]
                        for key in ("f1", "tp", "fp", "fn")})
        result.update(status="complete", metrics=metrics, selected_decoder=decoder,
                      selected_epoch=freeze["checkpoint_epoch"], threshold=pair["threshold"],
                      sample_count=n, positive_count=positives,
                      checkpoint_sha256=freeze["checkpoint_sha256"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status="needs_attention", issues=[str(exc)])
    return result


def summarize(root):
    root = Path(root).resolve()
    rows = []
    for seed in SEEDS:
        sources = {}
        for arm in ("control", "candidate"):
            directory = root / ("confirmation_controls" if arm == "control" else "confirmation_candidate") / ("seed%d" % seed)
            freeze_path = directory / "val" / "validation_freeze.json"
            try:
                freeze = read_json(freeze_path) if freeze_path.exists() else None
                sources[arm] = (directory, freeze, None)
            except (OSError, ValueError) as exc:
                sources[arm] = (directory, None, str(exc))
        for population in POPULATIONS:
            row = dict(seed=seed, population=population, delta=None)
            split = "real" if population.startswith("real_") else population
            for arm, (directory, freeze, error) in sources.items():
                result = extract_summary(directory / split / "summary.json", freeze, seed, population)
                if error:
                    result.update(status="needs_attention", issues=[error])
                row[arm] = result
            if all(row[arm]["status"] == "complete" for arm in ("control", "candidate")):
                row["delta"] = {key: value - row["control"]["metrics"][key]
                                for key, value in row["candidate"]["metrics"].items()
                                if isinstance(value, (float, int)) and isinstance(row["control"]["metrics"][key], (float, int))}
            rows.append(row)
    states = [row[arm]["status"] for row in rows for arm in ("control", "candidate")]
    status = "needs_attention" if "needs_attention" in states else "complete" if all(s == "complete" for s in states) else "partial"
    return dict(schema_version="matched-confirmation-comparison/1", status=status, rows=rows,
                delta_direction="candidate_minus_same_seed_control", seeds=list(SEEDS),
                scope="Saved own-validation-frozen summaries only; no candidate/seed reselection.",
                caveats=["Both seeds must be reported; no significance claim from two seeds.",
                         "REAL populations share the same508 positives; strict is not an independent dataset.",
                         "Median and P90 are conditional on valid poses; inspect coverage alongside them.",
                         "TEST and REAL were previously viewed; these are not untouched holdouts."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
