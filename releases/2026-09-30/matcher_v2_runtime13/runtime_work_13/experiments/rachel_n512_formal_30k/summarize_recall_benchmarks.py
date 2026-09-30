"""Read-only experiment collection; no model loading, fitting, or job execution.

Discover runs from a frozen queue config. Missing or unfinished results remain
pending, never zero. --run-root remaps the configured root to a local mirror.
Only the two named report files in --output are created or refreshed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "rachel-recall-benchmark-summary/1"
POINTS = ("max_f1", "recall_first")
METRICS = ("threshold", "recall", "precision", "fp", "f1", "tp", "fn", "tn", "auprc")


class Reader:
    def __init__(self):
        self.sources, self.warnings = {}, []

    def json(self, path):
        path = Path(path)
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            return None
        self.sources[str(path)] = hashlib.sha256(payload).hexdigest()
        try:
            return json.loads(payload)
        except (ValueError, UnicodeDecodeError) as error:
            self.warnings.append(str(path) + ": unreadable/in-progress JSON: " + str(error))
            return None


def options(stage):
    command, result = stage["command"], {}
    for index, item in enumerate(command):
        if item.startswith("--"):
            result[item] = (command[index + 1] if index + 1 < len(command)
                            and not command[index + 1].startswith("--") else True)
    return result


def observed(reader, stage, remap):
    path = remap(stage["completion_path"])
    value = reader.json(path)
    complete = isinstance(value, dict) and value.get("status") in stage.get("completion_statuses", ["complete"])
    return dict(status="complete" if complete else "pending", source=str(path),
                observed_status=value.get("status") if isinstance(value, dict) else None), value


def compact_operating(value):
    if not isinstance(value, dict):
        return None
    validation = value.get("validation", {})
    return dict(selection_key=value.get("selection_key"),
        thresholds=value.get("thresholds"),
        validation={name: validation[name] for name in (*POINTS, "recall_95") if name in validation})


def normalize_metrics(value, count):
    """Do not convert absent metrics into successes or fabricate missing counts."""
    for name in ("threshold", "recall", "precision", "f1"):
        number = value.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("missing/invalid " + name)
    for name in ("tp", "fp", "fn", "tn"):
        if type(value.get(name)) is not int or value[name] < 0:
            raise ValueError("missing/invalid " + name)
    tp, fp, fn, tn = (value[name] for name in ("tp", "fp", "fn", "tn"))
    if tp + fp + fn + tn != count or tp + fn != count // 2:
        raise ValueError("confusion counts differ from complete balanced population")
    expected = dict(recall=tp / max(1, tp + fn), precision=tp / max(1, tp + fp), f1=2 * tp / max(1, 2 * tp + fp + fn))
    if any(not math.isclose(value[key], number, abs_tol=1e-7) for key, number in expected.items()):
        raise ValueError("classification values disagree with their confusion counts")
    if "auprc" in value and (not isinstance(value["auprc"], (int, float))
            or not math.isfinite(value["auprc"]) or not 0 <= value["auprc"] <= 1):
        raise ValueError("invalid auprc")
    return {key: value[key] for key in METRICS if key in value}


def evaluation(reader, stage, remap, method):
    result, artifact = observed(reader, stage, remap)
    result.update(split=options(stage)["--split"], operating_points=None, delta_recall_first_minus_max_f1=None)
    if result["status"] != "complete":
        return result
    parent = Path(result["source"]).parent
    protocol, receipt = reader.json(parent / "protocol.json"), reader.json(parent / "receipt.json")
    if any(not isinstance(item, dict) or item.get("status") != "complete" for item in (protocol, receipt)):
        result.update(status="pending", reason="complete protocol and receipt are not both available")
        return result
    try:
        split = result["split"]
        count = {"test": 3000, "real": 1016}[split]
        if artifact.get("split") != split or artifact.get("test_or_real_used_for_fit") is not False:
            raise ValueError("evaluation split or no-held-out-fit identity differs")
        summary = artifact if method != "full" else reader.json(Path(result["source"]).with_name("summary.json"))
        if (not isinstance(summary, dict) or summary.get("status") != "complete"
                or summary.get("sample_count") != count or summary.get("positive_count") != count // 2):
            raise ValueError("complete full-population summary unavailable")
        summary_path = str(parent / "summary.json")
        if "summary_sha256" in receipt and receipt["summary_sha256"] != reader.sources.get(summary_path):
            raise ValueError("summary differs from its completion receipt")
        checkpoint = summary.get("checkpoint_sha256")
        if checkpoint is not None and any(item.get("checkpoint_sha256") != checkpoint for item in (protocol, receipt)):
            raise ValueError("summary/protocol/receipt checkpoint identities differ")
        values = artifact["methods"] if method == "full" else artifact["operating_points"]
        points = {}
        for name in POINTS:
            point = values[name]
            metrics = point if method == "full" else point["classification"]
            points[name] = dict(classification=normalize_metrics(metrics, count),
                accepted_good_le10_count=point.get("accepted_good_layout_count" if method == "full" else "accepted_good_le10_count"))
            good = points[name]["accepted_good_le10_count"]
            if good is not None and (type(good) is not int or not 0 <= good <= metrics["tp"]):
                raise ValueError("accepted-good layout count differs from accepted true-pair count")
        first, base = points["recall_first"]["classification"], points["max_f1"]["classification"]
        if first["threshold"] > base["threshold"] or first["tp"] < base["tp"] or first["fp"] < base["fp"]:
            raise ValueError("recall_first does not preserve the max-F1 acceptance set")
        result.update(sample_count=count, positive_count=count // 2, operating_points=points,
            delta_recall_first_minus_max_f1={key: first[key] - base[key] for key in ("recall", "precision", "fp", "f1")},
            raw_layout=summary.get("raw_layout") if method != "full" else summary.get("layout"),
            strict547=summary.get("strict_summary"),
            checkpoint_identity=summary.get("model_identity", {"checkpoint_sha256": summary.get("checkpoint_sha256")}),
            same_checkpoint_threshold_comparison=True,
            validation_operating_points=compact_operating(reader.json(parent / "validation_freeze.json")))
    except (KeyError, TypeError, ValueError) as error:
        result.update(status="invalid", reason=str(error), operating_points=None)
    return result


def full_training(reader, root, state, receipt):
    protocol = reader.json(root / "protocol.json") or {}
    curves = []
    for path in sorted(root.glob("validation_[0-9]*.json")):
        row = reader.json(path)
        if isinstance(row, dict):
            curves.append(dict(global_exposure=row.get("global_exposure"),
                selection_eligible=row.get("selection_eligible"), operating_points=compact_operating(row.get("operating_points"))))
    segments = []
    for path in sorted(root.glob("segment_[0-9]*.json")):
        row = reader.json(path)
        if isinstance(row, dict):
            segments.append(dict(segment=row.get("segment"), training=row.get("training")))
    roles = {}
    for role, filename in (("winner", "train_val_freeze.json"), ("six_epoch_snapshot", "six_epoch_freeze.json")):
        freeze = reader.json(root / filename)
        roles[role] = dict(status="pending") if not isinstance(freeze, dict) else dict(
            status="complete" if freeze.get("status") == "complete" else "provisional",
            selected_epoch=freeze.get("selected_epoch"), selected_global_exposure=freeze.get("selected_global_exposure"),
            checkpoint_sha256=freeze.get("checkpoint_sha256"), selection_rule=freeze.get("selection_rule"),
            operating_points=compact_operating(freeze.get("operating_points")))
    return dict(**state, unique_count=protocol.get("unique_count"), seed=protocol.get("seed"),
        total_exposures=protocol.get("total_exposures"), planned_updates=protocol.get("planned_updates"),
        completed_exposures=protocol.get("completed_exposures", (receipt or {}).get("global_exposure")),
        completed_updates=protocol.get("completed_updates", (receipt or {}).get("optimizer_updates")),
        roles=roles, validation_curve=curves, training_segments=segments,
        convergence_demonstrated=None,
        interpretation="Equal-update validation-selected winner; six-dataset-epoch snapshot is a separate fixed diagnostic, not an extra selection opportunity.")


def shred_log_curves(reader, path):
    """Read emitted JSON epoch events only; never deserialize GPU checkpoints."""
    curves = {stage: [] for stage in ("coarse", "matching", "classify")}
    if not path.exists():
        return curves
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(row, dict) and row.get("event") == "rachel_shreddingnet_epoch" and row.get("stage") in curves:
                curves[row["stage"]].append(row)
    reader.sources[str(path)] = digest.hexdigest()
    # Resumes may repeat events. The latest emitted record for an epoch wins;
    # complete stage receipts take precedence over all log observations.
    return {stage: list({row["epoch"]: row for row in rows}.values()) for stage, rows in curves.items()}


def benchmark_training(reader, root, state, receipt, stage, method, run_root):
    args = options(stage)
    result = dict(**state, seed=int(args["--seed"]) if "--seed" in args else None,
        convergence_demonstrated=False,
        interpretation="First experimental budget only; completed 24 epochs is not evidence of convergence. Budgets are not equal compute across methods.")
    if method == "pairingnet":
        active = root
        if not active.exists():
            active = next((root.with_name("." + root.name + suffix) for suffix in (".partial", ".failed")
                           if root.with_name("." + root.name + suffix).exists()), root)
        rows = reader.json(active / "epochs.json") or []
        winner_validation = reader.json(active / "winner_validation_report.json") or {}
        result.update(planned_epochs={key: int(args[key]) for key in ("--min-epochs", "--max-epochs") if key in args},
            epochs_completed=(receipt or {}).get("epochs_completed"), winner_epoch=(receipt or {}).get("winner_epoch"),
            stop_reason=(receipt or {}).get("stop_reason"),
            training_reported_convergence=(receipt or {}).get("convergence_demonstrated"),
            winner_validation_operating_points=compact_operating(winner_validation.get("recall_operating_points")),
            validation_curve=[dict(epoch=row.get("epoch"), winner_updated=row.get("winner_updated"),
                selection_key=row.get("selection_key"), train=row.get("train"),
                operating_points=compact_operating(row.get("validation", {}).get("recall_operating_points"))) for row in rows],
            unique_count=(receipt or {}).get("population_audit", {}).get("population", {}).get("train", {}).get("rows"))
    else:
        emitted = shred_log_curves(reader, run_root / (stage["name"] + ".log"))
        stages = {}
        for name in emitted:
            complete = reader.json(root / "stages" / name / "completion.json")
            accepted = isinstance(complete, dict) and complete.get("status") == "complete_train_val_stage"
            rows = complete["history"] if accepted else emitted[name]
            stages[name] = dict(status="complete" if accepted else "pending",
                planned_epochs=int(args["--" + name + "-epochs"]) if "--" + name + "-epochs" in args else None,
                completed_epochs=complete.get("completed_epochs") if accepted else None,
                winner_epoch_one_based=complete["winner_epoch"] + 1 if accepted else None,
                curve_source="stage_completion" if accepted else "provisional_stdout",
                curve=[dict(epoch_one_based=row.get("epoch_one_based"), train=row.get("train"),
                    validation=row.get("val"), best_epoch_one_based=row["best_epoch"] + 1 if "best_epoch" in row else None) for row in rows])
        result.update(stages=stages,
            unique_count=(receipt or {}).get("dataset_audit", {}).get("counts", {}).get("train", {}).get("rows"))
    return result


def collect(config_path, run_root=None):
    reader = Reader()
    config_path = Path(config_path).resolve(strict=True)
    config = reader.json(config_path)
    if not isinstance(config, dict) or config.get("schema_version") != "rachel-recall-benchmark-queue/1":
        raise ValueError("requires frozen recall queue config")
    original_root = Path(config["root"])
    root = Path(run_root).resolve() if run_root else original_root

    def remap(value):
        path = Path(value)
        if not path.is_absolute():
            return root / path
        try:
            return root / path.relative_to(original_root)
        except ValueError:
            raise ValueError("result path lies outside the frozen queue root: " + str(path))

    models, by_training_path, stage_states = {}, {}, []
    for stage in config["stages"]:
        state, artifact = observed(reader, stage, remap)
        stage_states.append(dict(name=stage["name"], **state))
        args, marker = options(stage), stage.get("marker", "")
        if "--training-run" in args or "--smoke" in args:
            continue
        if marker.endswith("train_recall_data_volume"):
            method, key = "full", args["--arm"]
        elif marker.endswith("rachel_pairingnet_benchmark"):
            method, key = "pairingnet", "pairingnet"
        elif marker.endswith("rachel_shreddingnet_benchmark"):
            method, key = "shreddingnet", "shreddingnet"
        else:
            continue
        training_root = remap(args.get("--output-root", args.get("--output")))
        if key in models or str(training_root) in by_training_path:
            raise ValueError("duplicate configured training identity")
        training = (full_training(reader, training_root, state, artifact) if method == "full" else
                    benchmark_training(reader, training_root, state, artifact, stage, method, root))
        models[key] = dict(model=key, method=method, training_root=str(training_root), training=training, evaluations={})
        by_training_path[str(training_root)] = key
    for stage in config["stages"]:
        args = options(stage)
        if "--training-run" not in args or "--split" not in args:
            continue
        key = by_training_path.get(str(remap(args["--training-run"])))
        if key is None:
            raise ValueError("evaluation has no configured training identity: " + stage["name"])
        role, split = args.get("--checkpoint-role", "winner"), args["--split"]
        roles = models[key]["evaluations"].setdefault(role, {})
        if split in roles:
            raise ValueError("duplicate model/role/split evaluation")
        roles[split] = evaluation(reader, stage, remap, models[key]["method"])
    complete = bool(stage_states) and all(row["status"] == "complete" for row in stage_states)
    evaluations = [value for model in models.values() for role in model["evaluations"].values() for value in role.values()]
    valid = bool(evaluations) and all(value["status"] == "complete" for value in evaluations)
    coverage = reader.json(root / "DATA_READY.json")
    if coverage is None and root != config_path.parent:
        coverage = reader.json(config_path.parent / "DATA_READY.json")
    if not isinstance(coverage, dict) or coverage.get("status") != "complete":
        coverage = dict(status="pending", nested_subsets=None)
    return dict(schema_version=SCHEMA, status="complete" if complete and valid else "pending",
        config_path=str(config_path), configured_root=str(original_root), observed_root=str(root),
        all_configured_stages_complete=complete, all_evaluations_valid_and_complete=valid,
        models=list(models.values()), stages=stage_states, data_coverage=coverage,
        source_sha256=reader.sources, warnings=reader.warnings,
        notes=["No thresholds fitted and no held-out winner selection performed by this report.",
               "REAL threshold changes compare the same checkpoint; requested VAL recall is not guaranteed on REAL.",
               "Pending/invalid evaluations contribute no numeric results; absent configured roles are not invented.",
               "Full equal-update winners and fixed six-epoch snapshots have different exposure budgets.",
               "Benchmark initial 24-epoch budgets are not convergence claims or equal-compute comparisons.",
               "Smaller TRAIN-pair subsets mainly reduce pairing density while nearly retaining source-unit coverage; they do not represent fourfold fewer independent physical source images. Split-unit IDs are not verified physical-image counts, and pair counts are not paper image/fragment counts."])


def markdown(report):
    lines = ["# Recall benchmark summary", "", "Overall: **" + report["status"] + "**. Missing results are pending, not zero.", "",
             "## REAL: same-checkpoint threshold comparison", "",
             "| Model / checkpoint | Status | max-F1 R / P / FP / F1 | recall-first R / P / FP / F1 | ΔR / ΔFP |",
             "|---|---|---|---|---|"]
    for model in report["models"]:
        for role, splits in model["evaluations"].items():
            value = splits.get("real")
            if value is None:
                continue
            cells = [model["model"] + " / " + role, value["status"], "—", "—", "—"]
            if value["status"] == "complete":
                for index, name in enumerate(POINTS, 2):
                    point = value["operating_points"][name]["classification"]
                    cells[index] = "{recall:.3f} / {precision:.3f} / {fp} / {f1:.3f}".format(**point)
                delta = value["delta_recall_first_minus_max_f1"]
                cells[4] = "{recall:+.3f} / {fp:+d}".format(**delta)
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Training and TEST", ""]
    for model in report["models"]:
        training = model["training"]
        lines.append("- " + model["model"] + ": training " + training["status"] + ".")
        if model["method"] == "full":
            for role, snapshot in training["roles"].items():
                description = role + " " + snapshot["status"]
                if snapshot.get("selected_epoch") is not None:
                    description += "; epoch %s, exposures %s" % (snapshot["selected_epoch"], snapshot.get("selected_global_exposure"))
                operating = snapshot.get("operating_points") or {}
                key = operating.get("selection_key")
                if key and len(key) >= 2:
                    description += "; VAL P@R95/AP %.3f/%.3f" % (key[0], key[1])
                lines.append("  - " + description + ".")
        else:
            lines.append("  - First benchmark budget only; convergence is not established.")
            if model["method"] == "pairingnet" and training.get("epochs_completed") is not None:
                lines.append("  - Completed epochs %s; VAL winner epoch %s." % (training["epochs_completed"], training.get("winner_epoch")))
            if model["method"] == "shreddingnet":
                for name, value in training["stages"].items():
                    text = name + " " + value["status"]
                    if value.get("completed_epochs") is not None:
                        text += "; completed epochs %s; VAL winner epoch %s" % (value["completed_epochs"], value.get("winner_epoch_one_based"))
                    lines.append("  - " + text + ".")
        for role, splits in model["evaluations"].items():
            test = splits.get("test")
            if test:
                suffix = test["status"]
                if test["status"] == "complete":
                    point = test["operating_points"]["recall_first"]["classification"]
                    suffix += "; recall-first R={recall:.3f}, P={precision:.3f}, FP={fp}, F1={f1:.3f}".format(**point)
                lines.append("  - TEST " + role + ": " + suffix + ".")
    lines += ["", "VAL curves, training losses, checkpoint exposure identities, provenance hashes, and any errors are retained in the JSON.", "",
              "Full equal-update winners and fixed six-epoch snapshots are separate comparisons. VAL recall targets do not guarantee REAL recall; methods do not have equal compute budgets.", "",
              "Smaller TRAIN-pair subsets mainly reduce pairing density while nearly retaining source-unit coverage, not fourfold fewer independent original images. Split-unit IDs are not verified physical-image counts; training-pair counts must not be compared directly with paper image/fragment counts.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = collect(args.config, args.run_root)
    args.output.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = (args.output / name for name in ("recall_benchmark_summary.json", "recall_benchmark_summary.md"))
    if json_path.is_symlink() or markdown_path.is_symlink():
        raise ValueError("refusing to overwrite a symlinked output")
    if json_path.exists() and json.loads(json_path.read_text()).get("schema_version") != SCHEMA:
        raise ValueError("refusing to overwrite an unrelated output")
    if markdown_path.exists() and not json_path.exists():
        raise ValueError("refusing to overwrite an unrelated Markdown file")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    markdown_path.write_text(markdown(report))
    print(json.dumps(dict(status=report["status"], json=str(json_path), markdown=str(markdown_path))))
    return report


if __name__ == "__main__":
    main()
