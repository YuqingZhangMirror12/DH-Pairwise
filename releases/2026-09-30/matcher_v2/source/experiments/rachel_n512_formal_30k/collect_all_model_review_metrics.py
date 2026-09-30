"""Collect already-frozen nine-model results; no inference or threshold fitting.

Primary checkpoints are fixed before reading held-out rows. Auxiliary S3/S4
selections are separate records; never attach their thresholds to primary rows.
All classification counts are recomputed from saved pair scores. Stdlib only.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
REPORT = WORKSPACE / "reports"
CURRENT = REPORT / "rachel_score_design_20260913_001"
BENCH = REPORT / "rachel_recall_benchmarks_20260911_001"
OOD = REPORT / "turufan_ood_pairwise_20260912_001/evaluation_v1"
OP_NAMES = {"max_f1": "max_f1", "recall_90": "recall90", "recall_95": "recall95",
            "recall_98": "recall98", "recall_99": "recall99", "recall_first": "recall_first"}
SCHEMA = "rachel-all-model-review-metrics/1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path, score_field, decoder=None, authority=None, **extra):
    path = Path(path).resolve()
    result = dict(pair_scores_file=str(path), pair_scores_sha256=sha(path), score_field=score_field,
                  decoder=decoder, **extra)
    if authority:
        result.update(authority_file=str(Path(authority).resolve()), authority_sha256=sha(authority))
    return result


def ranking(labels_scores):
    positive = sum(label for label, _ in labels_scores)
    negative = len(labels_scores) - positive
    if not positive or not negative:
        return None, None
    ordered = sorted(labels_scores, key=lambda row: row[1], reverse=True)
    tp = seen = 0
    ap = wins = 0.0
    for _, grouped in itertools.groupby(ordered, key=lambda row: row[1]):
        group = list(grouped)
        pos = sum(row[0] for row in group)
        neg = len(group) - pos
        # Every earlier positive outranks each current negative; ties count .5.
        wins += neg * tp + .5 * pos * neg
        tp += pos
        seen += len(group)
        ap += (pos / positive) * (tp / seen)
    return wins / (positive * negative), ap


def rows_from_source(info, split):
    path = Path(info["pair_scores_file"])
    rows = read(path)["predictions"] if path.suffix == ".json" else [json.loads(line) for line in path.open()]
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate pair IDs: " + str(path))
    result = []
    for row in rows:
        value = row
        for key in info["score_field"].split("."):
            value = value[key]
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("nonfinite/out-of-range saved score")
        result.append(dict(pair_id=row["pair_id"], label=True if split == "ood" else bool(row["label"]),
            score=value, decision_valid=row.get("decision_valid", True),
            layout=row.get("layouts", {}).get(info["decoder"], {}) if info["decoder"] else {},
            strict_member=row.get("strict_member", False)))
    return result


def measurement(rows, threshold, *, layout_gt):
    positive = sum(row["label"] for row in rows)
    negative = len(rows) - positive
    accepted = [bool(row["decision_valid"]) and row["score"] >= threshold for row in rows]
    tp = sum(row["label"] and yes for row, yes in zip(rows, accepted))
    fp = sum(not row["label"] and yes for row, yes in zip(rows, accepted))
    fn, tn = positive - tp, negative - fp
    auc, ap = ranking([(int(row["label"]), row["score"]) for row in rows])
    binary = bool(positive and negative)
    result = dict(threshold=threshold, sample_count=len(rows), positive_count=positive, negative_count=negative,
        tp=tp, fn=fn, fp=fp if negative else None, tn=tn if negative else None,
        accuracy=(tp + tn) / len(rows) if binary else None,
        precision=tp / (tp + fp) if binary and tp + fp else (0.0 if binary else None),
        recall=tp / positive if positive else None,
        f1=2 * tp / (2 * tp + fp + fn) if binary else None,
        fpr=fp / negative if negative else None, auroc=auc, ap=ap,
        rejected_positive_count=fn, accepted_positive_count=tp,
        invalid_decision_count=sum(not row["decision_valid"] for row in rows),
        metric_scope="full_binary" if binary else "positive_only_acceptance",
        layout20=None)
    if layout_gt:
        good = [row["label"] and row["layout"].get("valid", False) and
            row["layout"].get("translation_l2_px") is not None and
            row["layout"]["translation_l2_px"] <= 20 for row in rows]
        raw = sum(good)
        gated = sum(g and yes for g, yes in zip(good, accepted))
        result["layout20"] = dict(positive_count=positive, raw_correct=raw, accepted_correct=gated,
            fn_good=raw - gated, raw_recall=raw / positive, gated_recall=gated / positive,
            denominator="all positive pairs, including rejected pairs")
    else:
        result["limitation"] = "301 known-positive pairs; no negatives or layout GT. Accuracy/Precision/F1/AUROC/AP are not estimable."
    return result


def first5_models(comparison):
    data = read(CURRENT / "s2_first5_readout_20260914/comparison.json")
    result = []
    for label in ("S0", "S1", "S2"):
        summary_paths = {}
        for split, population in (("test", "SIM TEST"), ("real", "敦煌保留"), ("ood", "Turufan")):
            matches = [row for row in data["sources"] if row["model"] == label and row["population"] == population]
            if len(matches) != 1:
                raise ValueError("first5 source ambiguity")
            summary_paths[split] = Path(matches[0]["summary"])
        meta = read(summary_paths["test"])["model"]
        sources = {}
        for split, path in summary_paths.items():
            report = read(path)
            if report["status"] != "complete" or report["model"]["checkpoint_sha256"] != meta["checkpoint_sha256"]:
                raise ValueError("first5 checkpoint differs between domains")
            if report["model"]["operating_points"] != meta["operating_points"]:
                raise ValueError("first5 domain threshold mismatch")
            sources[split] = source(path.with_name("pair_results.jsonl"), "classification.fused", "full_top2_mode", path)
        training = copy.deepcopy(next(row["training"] for row in comparison["models"] if row["model"] == label))
        selection = dict(id="max_f1", primary=True, epoch=meta["epoch"], checkpoint_path=meta["checkpoint_path"],
            checkpoint_sha256=meta["checkpoint_sha256"], thresholds=meta["operating_points"]["thresholds"],
            threshold_source=dict(file=str(summary_paths["test"]), field="model.operating_points.thresholds",
                original_freeze_path=meta["freeze_path"], original_freeze_sha256=meta["freeze_sha256"]), sources=sources)
        result.append(dict(id=label.lower(), label=label + " first5", primary_selection="max_f1", training=training,
            budget_note="随机初始化；首档5轮/120K曝光；后续预算未完成", selections=[selection]))
    return result


def historical_models(comparison, frozen):
    result = []
    for key, label in (("historical_e1", "E1"), ("full_e1_24k", "Full-E1-24K"),
                       ("pairingnet", "PairingNet"), ("shreddingnet", "ShreddingNet")):
        binding = frozen["methods"][key]
        threshold_source = binding["threshold_source"]
        thresholds = threshold_source.get("operating_points", threshold_source)["thresholds"]
        sources = {}
        if key in ("pairingnet", "shreddingnet"):
            metadata = threshold_source["model_identity"]
            hashes = metadata["checkpoint_sha256_by_stage"]
            for split in ("test", "real"):
                root = BENCH / "completed" / key / split
                report = read(root / "protocol.json")
                if report["status"] != "complete" or report["model_identity"]["checkpoint_sha256_by_stage"] != hashes:
                    raise ValueError("benchmark checkpoint hash mismatch across domains")
                if read(root / "validation_freeze.json")["thresholds"] != thresholds:
                    raise ValueError("benchmark threshold freeze differs from OOD frozen model")
                sources[split] = source(root / "pair_results.jsonl", "pair_probability", metadata["original_decoder"], root / "protocol.json")
            training = {k:metadata.get(k) for k in ("seed", "precision", "train_unique_count", "epochs_completed", "planned_stage_epochs", "winner_epoch", "selection_metric", "training_manifest_sha256", "training_run")}
            training.update(checkpoint_selection_rule="SIM VAL P@95R then AP; not held-out fitting",
                implementation="mask-only adapted benchmark; not an exact native-paper reproduction")
            if key == "shreddingnet":
                authority = BENCH / "completed/shreddingnet/training/train_val_freeze.json"
                stages = read(authority)["checkpoints"]
                weights = {stage:dict(path=str(Path(metadata["training_run"]) / value["path"]), sha256=value["sha256"],
                    epoch=value["winner_epoch_one_based"], selection_metric=value["selection_metric"]) for stage, value in stages.items()}
                training["stage_winners"] = weights
                training["checkpoint_selection_rule"] = "coarse: VAL infor_loss minimum; matching: VAL positive_loss minimum; classify: VAL P@95R then AP"
                epoch = stages["classify"]["winner_epoch_one_based"]
                note = "同24K数据；coarse/matching/classify各24轮预算，选中20/24/17轮；mask-only适配"
            else:
                authority = BENCH / "completed/pairingnet/training/completion_receipt.json"
                completed = read(authority)
                if completed["winner_checkpoint_sha256"] != hashes["winner"]:
                    raise ValueError("PairingNet training winner mismatch")
                training["adaptation"] = completed["adaptation"]
                weights = dict(winner=dict(path=str(Path(metadata["training_run"]) / "winner.pt"), sha256=hashes["winner"]))
                epoch = metadata["winner_epoch"]
                note = "同24K数据；24轮/576K曝光；新增二分类头的mask-only适配，非论文原版二分类指标"
        else:
            old = next(row for row in comparison["models"] if row["model"] == label)
            training = copy.deepcopy(old["training"])
            for split in ("test", "real"):
                root = (REPORT / "rachel_weathering_all_20260909_001/e0_e1_completed_snapshot/e1/evaluation" / split
                        if key == "historical_e1" else BENCH / "completed/full_e1_24k/evaluation/winner" / split)
                receipt = read(root / "receipt.json")
                if receipt["status"] != "complete" or receipt["checkpoint_sha256"] != training["checkpoint_sha256"]:
                    raise ValueError("historical checkpoint mismatch")
                sources[split] = source(root / "pair_results.jsonl", "classification.fused", "full_top2_mode", root / "receipt.json")
            expected = binding.get("expected_checkpoint_sha256", threshold_source.get("checkpoint_sha256"))
            if expected != training["checkpoint_sha256"]:
                raise ValueError("historical OOD checkpoint mismatch")
            weights = dict(winner=dict(path=receipt["checkpoint_path"], sha256=expected))
            epoch = training.get("selected_epoch", training.get("selected_final_stage_epoch"))
            authority = Path(sources["test"]["authority_file"])
            note = ("继承预训练权重，再腐蚀微调；最终选第5轮/120K额外曝光，预训练曝光未量化" if key == "historical_e1" else
                    "随机初始化，同24K混合数据6轮/144K曝光；checkpoint按VAL P@95R选，非max-F1选模")
        ood_path = OOD / (key + "_predictions.json")
        if read(ood_path)["status"] != "complete":
            raise ValueError("OOD incomplete")
        sources["ood"] = source(ood_path, "score", authority=OOD / "frozen_models.json",
            authority_field="methods." + key, label_source="user-provided 301 positive pairs")
        selection = dict(id="historical_winner", primary=True, epoch=epoch,
            weights=weights, checkpoint_sha256_by_stage={name:weight["sha256"] for name, weight in weights.items()},
            checkpoint_sha256=next(iter(weights.values()))["sha256"] if len(weights) == 1 else weights["classify"]["sha256"],
            threshold_source=dict(file=str((OOD / "frozen_models.json").resolve()), field="methods." + key + ".threshold_source",
                original_freeze_path=binding["threshold_file"], original_freeze_sha256=binding["threshold_file_sha256"]),
            thresholds=thresholds, sources=sources, training_authority=str(authority.resolve()))
        result.append(dict(id=key, label=label + ("（适配版）" if key in ("pairingnet", "shreddingnet") else "（历史）"),
            primary_selection="historical_winner", training=training, budget_note=note, selections=[selection]))
    return result


def decoupled_models(comparison):
    dataset = read(CURRENT / "s3_v3_complete_20260914/frozen_readout/results.json")
    result = []
    for arm, key, label in (("s3_matrix_per_pair_norm_v3", "s3v3", "S3 v3"),
                           ("s4_cross_attention", "s4", "S4"), ("s3_matrix", "s3v2", "S3 v2（实现诊断）")):
        evaluations = [row for row in dataset["evaluations"] if row["arm"] == arm and row["status"] == "complete"]
        selections = []
        for selection in ("fixed_epoch", "max_f1", "recall95"):
            matches = {row["split"]:row for row in evaluations if row["selection"] == selection}
            if set(matches) != {"test", "real", "ood"}:
                raise ValueError("incomplete decoupled selection")
            first = matches["test"]
            frozen_selection = read(first["provenance"]["local_freeze_path"])["selections"][selection]
            if (frozen_selection["checkpoint_sha256"] != first["checkpoint_sha256"] or
                    frozen_selection["operating_points"] != first["operating_points"]):
                raise ValueError("local typed freeze differs from evaluated checkpoint/thresholds")
            sources = {}
            for split, row in matches.items():
                if row["checkpoint_sha256"] != first["checkpoint_sha256"] or row["operating_points"] != first["operating_points"]:
                    raise ValueError("decoupled cross-domain checkpoint/threshold mismatch")
                directory = Path(row["evaluation"])
                sources[split] = source(directory / "pair_results.jsonl", "classification.fused", row["decoder"], directory / "protocol.json")
            selections.append(dict(id=selection, primary=selection == "fixed_epoch", epoch=first["selected_epoch"],
                checkpoint_path=first["checkpoint_path"], checkpoint_sha256=first["checkpoint_sha256"],
                thresholds=first["operating_points"]["thresholds"], sources=sources,
                threshold_source=dict(file=first["provenance"]["local_freeze_path"], field="selections." + selection + ".operating_points.thresholds",
                    original_freeze_path=first["provenance"]["freeze_path"], original_freeze_sha256=first["provenance"]["freeze_sha256"]),
                winner_record=first["winner_record"]))
        training = copy.deepcopy(next(row["training"] for row in comparison["models"] if row["model"] == arm))
        note = "M12+C8 / 480K逻辑曝光；固定epoch20为主，另列SIMVAL辅助选择；logical micro1不同于历史micro4"
        if key == "s3v3":
            note += "；v3继承完整权重，仅归一化推理转换及重新VAL冻结，新增训练0"
        result.append(dict(id=key, label=label, primary_selection="fixed_epoch", training=training,
            budget_note=note, interpretation_status="implementation_diagnostic_only" if key == "s3v2" else "formal_frozen_result",
            selections=selections))
    return result


def collect():
    comparison_path = CURRENT / "s3_v3_complete_20260914/comparison/comparison.json"
    comparison = read(comparison_path)
    frozen = read(OOD / "frozen_models.json")
    if read(OOD / "protocol.json")["thresholds_fitted"]:
        raise ValueError("OOD threshold fitting is not allowed")
    keep_path = REPORT / "rachel_curated_review_20260912_001/keep_ids.json"
    keep = set(read(keep_path)["kept_positive_pair_ids"])
    if len(keep) != 295:
        raise ValueError("curated population differs from requested295")
    models = first5_models(comparison) + decoupled_models(comparison) + historical_models(comparison, frozen)
    output, diagnostics = [], []
    population_ids = {}
    for model in models:
        for selection in model["selections"]:
            for split, info in selection["sources"].items():
                rows = rows_from_source(info, split)
                if split == "real":
                    if len(rows) != 1016 or {row["pair_id"] for row in rows if row["label"] and row["pair_id"] in keep} != keep:
                        raise ValueError("REAL source or keep join differs")
                    rows = [row for row in rows if not row["label"] or row["pair_id"] in keep]
                expected = {"test": (1500,1500), "real": (295,508), "ood": (301,0)}[split]
                if (sum(row["label"] for row in rows), sum(not row["label"] for row in rows)) != expected:
                    raise ValueError("requested population count differs")
                ids = {row["pair_id"] for row in rows}
                if split in population_ids and population_ids[split] != ids:
                    raise ValueError("methods use different pair populations")
                population_ids[split] = ids
                for op, threshold in selection["thresholds"].items():
                    if op not in OP_NAMES:
                        continue
                    item = dict(model_id=model["id"], model_label=model["label"],
                        dataset={"test":"test", "real":"dunhuang", "ood":"turufan"}[split],
                        selection=selection["id"], is_primary_selection=selection["primary"],
                        operating_point=OP_NAMES[op], source_operating_point=op, epoch=selection["epoch"],
                        budget_note=model["budget_note"], checkpoint_sha256=selection["checkpoint_sha256"],
                        sources=info, **measurement(rows, float(threshold), layout_gt=split != "ood"))
                    (diagnostics if model["id"] == "s3v2" else output).append(item)
    formal = [model for model in models if model["id"] != "s3v2"]
    order = ["s0","s1","s2","s3v3","s4","historical_e1","full_e1_24k","pairingnet","shreddingnet"]
    formal.sort(key=lambda row: order.index(row["id"]))
    # Existing published primary rows are an independent aggregation check.
    mapping = dict(s0="S0",s1="S1",s2="S2",s3v3="s3_matrix_per_pair_norm_v3",s4="s4_cross_attention",
        historical_e1="E1",full_e1_24k="Full-E1-24K")
    for key, old_name in mapping.items():
        old = next(model for model in comparison["models"] if model["model"] == old_name)
        for dataset, split in (("test","test"),("dunhuang","real"),("turufan","ood")):
            row = next(row for row in output if row["model_id"] == key and row["dataset"] == dataset and
                row["is_primary_selection"] and row["operating_point"] == "max_f1")
            for field in ("tp","fp","fn","tn","threshold","accuracy","recall","f1"):
                if row[field] != old[split][field] and not (isinstance(row[field],float) and isinstance(old[split][field],float) and
                        math.isclose(row[field],old[split][field],rel_tol=1e-12,abs_tol=1e-12)):
                    raise ValueError("existing primary readout mismatch: %s/%s/%s" % (key,split,field))
    return dict(schema_version=SCHEMA, status="complete", models=formal, rows=output,
        diagnostic_models=[model for model in models if model["id"] == "s3v2"], diagnostic_rows=diagnostics,
        default_operating_point="max_f1", primary_working_points=["max_f1","recall95"],
        population=dict(test=dict(positive=1500,negative=1500),dunhuang=dict(positive=295,negative=508),
            turufan=dict(positive=301,negative=0)),
        keep_ids_source=dict(file=str(keep_path),sha256=sha(keep_path)),
        evidence=dict(comparison_file=str(comparison_path),comparison_sha256=sha(comparison_path),
            collector_file=str(Path(__file__).resolve()),source_weights_reloaded=False),
        verification=dict(primary_existing_counts_reconciled=True,all_methods_same_pair_ids=True,
            new_inference=False,thresholds_fitted=False,heldout_model_selection=False),
        caveats=["所有阈值来自对应checkpoint的已有SIM VAL冻结；R95不保证真实域召回95%。",
            "S3v3正式主表固定epoch20；辅助max-F1/P@95R选中epoch19，独立列出，不套用到epoch20案例。",
            "E1/Full/S0–S4与两benchmark训练预算、warmstart与归一化不完全一致，不是同预算架构优劣试验。",
            "PairingNet/ShreddingNet为本数据mask-only适配，不是原论文RGB/native检索结果；PairingNet含额外pair BCE头。",
            "人工保留295正例是回顾性筛选，所有508负例保留不变；不可把筛选增益视为模型新学习。",
            "Turufan只有301正例且无layout GT，只能给正例接受/召回；Accuracy/Precision/F1/AUROC/AP为null。",
            "S3 v2保留于诊断附表，因TRAIN/eval归一化错位不能据其单独否定架构。"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT / "rachel_all_models_review_20260914/metrics.json")
    args = parser.parse_args()
    result = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    primary = dict(schema_version="rachel-all-model-review-primary-sources/1", status="complete",
        metrics_file=str(args.output.resolve()), models=[])
    for model in result["models"]:
        entry = copy.deepcopy(next(selection for selection in model["selections"] if selection["primary"]))
        entry["selection"] = entry.pop("id")
        entry.update(id=model["id"], label=model["label"], budget_note=model["budget_note"])
        primary["models"].append(entry)
    primary_path = args.output.with_name("primary_sources.json")
    primary_path.write_text(json.dumps(primary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"],models=len(result["models"]),rows=len(result["rows"]),
        output=str(args.output),primary_sources=str(primary_path)),ensure_ascii=False))


if __name__ == "__main__":
    main()
