"""Prepare, not launch, one paired512 -> paired1024 finite comparison.

Unlike other input axes this must retrain the512 control: both arms consume
the same newly reconstructed source-seam supervision pipeline. Historical
prepared512 targets are not a single-factor control. S3 and complete paired
TRAIN/clean VAL/TEST preparation must finish before this queue is prepared.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from experiments.rachel_n512_formal_30k.prepare_score_budget_queue import (
    PYTHON, DATASET, METADATA_CHECKPOINT, ARMS, BUDGETS, SELECTIONS, SPLITS)


def build_config(root, architecture, budget, run_name, *, train_root, clean_root,
                 source_density_version="v1", python=PYTHON):
    root, train_root, clean_root = map(Path, (root, train_root, clean_root))
    if not all(p.is_absolute() for p in (root, train_root, clean_root)):
        raise ValueError("absolute experiment and paired data roots required")
    if architecture not in ARMS or budget not in BUDGETS:
        raise ValueError("registered architecture and budget required")
    if source_density_version not in ("v1", "v2", "v3", "v4"):
        raise ValueError("explicit registered source-density version required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("simple unique run name required")
    run = root / "density_variants" / run_name
    stages = []
    for cap in (512, 1024):
        cap_root = run / ("n%d" % cap)
        train = [python, "-m", "experiments.rachel_n512_formal_30k.train_score_density",
            "--checkpoint", METADATA_CHECKPOINT, "--dataset", DATASET,
            "--train-density-manifest", str(train_root / ("train_n%d.json" % cap)),
            "--clean-density-cache-root", str(clean_root), "--contour-cap", str(cap),
            "--architecture", architecture, "--stop-after-epoch", str(budget), "--workers", "4"]
        if source_density_version != "v1":
            train += ["--source-density-version", source_density_version]
        smoke, training = cap_root / "smoke32", cap_root / "training"
        stages.append(dict(name="density%d_smoke32" % cap,
            command=train + ["--output", str(smoke), "--smoke", "32"], marker=str(smoke),
            completion_path=str(smoke / "smoke.json"), completion_statuses=["smoke_complete"]))
        stages.append(dict(name="density%d_random_%03d" % (cap, budget),
            command=train + ["--output", str(training)], marker=str(training),
            completion_path=str(training / "status.json"),
            completion_statuses=["complete" if budget == 50 else "budget_complete"],
            resume_arguments=["--resume"]))
        for selection in SELECTIONS:
            for split in SPLITS:
                output = cap_root / "evaluation" / ("budget_%03d" % budget) / selection / split
                command = [python, "-m", "experiments.rachel_n512_formal_30k.evaluate_score_density",
                    "--training-run", str(training), "--budget", str(budget), "--selection", selection,
                    "--split", split, "--output", str(output), "--dataset", DATASET,
                    "--clean-density-cache-root", str(clean_root), "--batch-size", "4", "--workers", "4"]
                if split == "real":
                    command += ["--keep-ids", str(root / "keep_ids.json")]
                if cap == 1024:
                    baseline = run / "n512/evaluation" / ("budget_%03d" % budget) / selection / split
                    command += ["--baseline-evaluation", str(baseline)]
                stages.append(dict(name="density%d_%03d_%s_%s" % (cap, budget, selection, split),
                    command=command, marker=str(output), completion_path=str(output / "protocol.json"),
                    completion_statuses=["complete"]))
    return dict(root=str(root / "queues" / ("density_" + run_name)),
        source=str(root / "density_runtime_source"), dependencies=[], stages=stages)


def clean_split_receipt(cache_root, split, dataset_root=DATASET):
    cache_root, dataset_root = Path(cache_root), Path(dataset_root)
    receipt = json.loads((cache_root / ("clean_%s_preparation.json" % split)).read_text())
    original = [json.loads(line) for line in (dataset_root / "pairs" / (split + ".jsonl")).read_text().splitlines() if line]
    ids = [row["pair_id"] for row in original]
    records = receipt.get("records", [])
    if (len(ids) != 3000 or len(set(ids)) != 3000 or sum(r["label"] for r in original) != 1500
            or receipt.get("status") != "complete" or receipt.get("full_split") is not True
            or receipt.get("split") != split or receipt.get("failures") != []
            or any(receipt.get(k) != 3000 for k in ("original_split_count", "selected_count", "cached_pair_count"))
            or receipt.get("pair_ids") != ids or [r["pair_id"] for r in records] != ids
            or any(bool(a["label"]) != bool(b["label"]) for a, b in zip(records, original))
            or any(set(r.get("caps", {})) != {"512", "1024"} for r in records)
            or receipt.get("model_inference") is not False or receipt.get("model_or_threshold_selected") is not False):
        raise ValueError("requires completely prepared clean " + split + "3000 at both caps, not a subset smoke")
    return receipt


def prepare(root, architecture, budget, run_name, *, train_root, clean_root, s3_comparison,
            source_density_version="v1"):
    from experiments.rachel_n512_formal_30k.prepare_score_input_queue import verify_s3
    from experiments.rachel_n512_formal_30k.train_score_density import (
        validate_populations, make_clean_density_dataset, validate_source_density_protocols)
    from experiments.rachel_n512_formal_30k.train_edge_weathering import _sha256
    from staging.pairwise_v0_2.pairwise_data.rachel_paired_density_dataset import PairedSourceDensityWeatheredDataset
    root, train_root, clean_root = (Path(p).resolve(strict=True) for p in (root, train_root, clean_root))
    config = build_config(root, architecture, budget, run_name, train_root=train_root,
        clean_root=clean_root, source_density_version=source_density_version)
    queue, run = Path(config["root"]), root / "density_variants" / run_name
    if queue.exists() or run.exists():
        raise FileExistsError("density run/queue already exists; do not restart or overwrite")
    for module in ("train_score_density.py", "evaluate_score_density.py"):
        if not (Path(config["source"]) / "experiments/rachel_n512_formal_30k" / module).is_file():
            raise ValueError("isolated density runtime entry missing: " + module)
    kept_path = root / "keep_ids.json"
    kept = json.loads(kept_path.read_text())["kept_positive_pair_ids"]
    if len(kept) != 295 or len(set(kept)) != 295:
        raise ValueError("the existing reviewed295 cohort is required")
    s3 = verify_s3(root, s3_comparison, architecture, _sha256(kept_path))
    receipts = {split: clean_split_receipt(clean_root, split) for split in ("val", "test")}
    for receipt in receipts.values():
        validate_source_density_protocols(source_density_version, preparation=receipt)
    controls = []
    for cap in (512, 1024):
        training = PairedSourceDensityWeatheredDataset(train_root / ("train_n%d.json" % cap))
        validation = make_clean_density_dataset(DATASET, "val", cap, clean_root / ("val_n%d" % cap),
            source_density_version=source_density_version)
        validate_populations(training, validation, cap, clean_root, source_density_version)
        controls.append(dict(cap=cap, train_identity=training.identity,
            train_manifest_sha256=_sha256(training.manifest_path),
            original_train_manifest_sha256=training.protocol["source_manifest_sha256"],
            validation_identity=validation.identity))
    if controls[0]["train_identity"] != controls[1]["train_identity"]:
        raise ValueError("both caps must come from the identical fixed source selection/pipeline")
    record = dict(status="prepared_not_launched", architecture=architecture, budget=budget,
        source_density_version=source_density_version,
        s3_prerequisite=s3, controls=controls, full_clean_counts={s: r["cached_pair_count"] for s, r in receipts.items()},
        paired512_control_retrained=True, historical512_used_as_density_control=False,
        same_original_masks_labels_and_gt=True, only_changed_axis="contour_cap",
        stage_count=16, model_training_count=2, evaluation_count=12,
        held_out_metrics_used_to_choose_next_action=False, thresholds_fit_on="completeSIM VAL only")
    run.mkdir(parents=True, exist_ok=False)
    queue.mkdir(parents=True, exist_ok=False)
    for path, value in ((run / "preparation.json", record), (queue / "preparation.json", record), (queue / "config.json", config)):
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    result = dict(status="prepared_not_launched", config=str(queue / "config.json"))
    print(json.dumps(result))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--architecture", required=True, choices=ARMS)
    p.add_argument("--budget", required=True, type=int, choices=BUDGETS)
    p.add_argument("--run-name", required=True)
    p.add_argument("--train-root", required=True)
    p.add_argument("--clean-root", required=True)
    p.add_argument("--s3-comparison", required=True)
    p.add_argument("--source-density-version", choices=("v1", "v2", "v3", "v4"), default="v1")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    prepare(args.root, args.architecture, args.budget, args.run_name,
            train_root=args.train_root, clean_root=args.clean_root, s3_comparison=args.s3_comparison,
            source_density_version=args.source_density_version)
