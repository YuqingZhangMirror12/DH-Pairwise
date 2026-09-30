"""Read completed E0/E1/E2/E3 outputs; write one NEW JSON/Markdown snapshot.

No training, inference, threshold fitting, or REAL-based model selection occurs.
Missing/provisional results remain explicit, never zero-valued measurements.
Recall@FPR is a held-out ROC diagnostic, not a newly selected deployment point.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "rachel-weathering-ablation-summary/1"
DEFAULT_ALL = "/root/autodl-tmp/rachel_weathering_all_20260909_001"
DEFAULT_E0 = "/root/autodl-tmp/rachel_pairability_overnight_20260909_001"
DEFAULT_E1 = "/root/autodl-tmp/rachel_edge_weathering_train_20260909_001"
DECODER = "full_top2_mode"
POPULATIONS = {"test": (3000, 1500), "real": (1016, 508)}
E3_BUDGET = (10, 10, 24000, 240000, 940)  # planned/completed epochs, unique rows, exposures/updates per head
CLASS_FIELDS = ("precision", "recall", "f1", "auprc", "auroc", "tp", "fp", "fn", "tn", "threshold")
JOINT_FIELDS = ("tp", "fp", "fn", "precision", "recall", "f1")


def _read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("nonfinite/missing " + label)
    return value


def recall_at_fpr(labels, scores):
    """Max attainable empirical TPR at FPR<=cap; equal scores stay together.

No interpolation/randomization of ties and no deployment threshold is returned.
The always-reject point is included. Both classes must be present.
"""
    if len(labels) != len(scores) or not labels:
        raise ValueError("ROC diagnostic requires aligned nonempty labels and scores")
    if any(type(label) is not bool for label in labels):
        raise ValueError("ROC labels must be boolean")
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        raise ValueError("ROC diagnostic requires both classes")
    groups = {}
    for label, score in zip(labels, scores):
        score = float(_finite(score, "score"))
        counts = groups.setdefault(score, [0, 0])
        counts[0 if label else 1] += 1
    result = {"0.01": 0.0, "0.05": 0.0, "0.10": 0.0}
    tp = fp = 0
    for score in sorted(groups, reverse=True):
        positive, negative = groups[score]
        tp += positive
        fp += negative
        for cap in result:
            if fp / negatives <= float(cap) + 1e-12:
                result[cap] = max(result[cap], tp / positives)
    return result


def _pending(status, reason, sources):
    return dict(status=status, reason=reason, sources=sources, metrics=None)


def training_metadata(root):
    """Budget and winner facts only from the arm's own completed freeze."""
    root = Path(root)
    sources = dict(training_freeze=str(root / "train_val_freeze.json"),
                   training_protocol=str(root / "protocol.json"), winner_checkpoint=str(root / "winner.pt"))
    freeze, protocol, progress = _read(sources["training_freeze"]), _read(sources["training_protocol"]), _read(root / "status.json")
    result = dict(sources=sources, status="not_run", metadata=None)
    if freeze is None:
        if protocol is not None or progress is not None:
            result.update(status="failed" if (progress or protocol).get("status") == "failed" else "pending",
                          reason="no completed training freeze")
        return result
    if freeze.get("status") != "complete":
        result.update(status="failed" if (progress or protocol or {}).get("status") == "failed" else "pending",
                      reason="training freeze is " + str(freeze.get("status", "unmarked")))
        return result
    if protocol is None or protocol.get("status") != "complete":
        result.update(status="pending", reason="training protocol is missing/not complete")
        return result
    if (freeze.get("completed_global_exposures"), freeze.get("completed_optimizer_updates"),
            freeze.get("completed_validation_events")) != (120000, 7500, 5):
        raise ValueError("completed arm does not satisfy the fixed 120k/7500/5-VAL budget")
    if freeze.get("test_or_real_used_for_fit") is not False or freeze.get("original_validation_unchanged") is not True:
        raise ValueError("training freeze does not confirm unchanged VAL-only selection")
    val = freeze["validation"]
    if (val.get("sample_count"), val.get("positive_count"), val.get("negative_count")) != (3000, 1500, 1500):
        raise ValueError("training winner did not use clean balanced VAL3000")
    metadata = {key: freeze.get(key) for key in (
        "unique_count", "completed_global_exposures", "completed_optimizer_updates", "completed_dataset_epochs",
        "completed_validation_events", "selected_global_exposure", "selected_optimizer_updates", "selected_epoch",
        "selected_validation_event", "initial_checkpoint", "selection_rule", "seed", "precision")}
    for key in ("unique_count", "selected_global_exposure", "selected_optimizer_updates", "selected_epoch", "selected_validation_event"):
        _finite(metadata[key], key)
    metadata.update(initial_checkpoint_sha256=protocol.get("initial_checkpoint_sha256"),
                    selected_checkpoint=freeze.get("checkpoint", sources["winner_checkpoint"]),
                    classifier_thresholds=freeze["classifier_thresholds"],
                    selected_clean_val_fused=val.get("methods", {}).get("fused"),
                    selected_checkpoint_is_final_update=metadata["selected_global_exposure"] == metadata["completed_global_exposures"],
                    selected_incremental_exposure_covers_pool=metadata["selected_global_exposure"] >= metadata["unique_count"],
                    historical_warm_start_exposures_included=False,
                    weathering_policy=protocol.get("weathering_policy"),
                    target_policy=protocol.get("training_data", {}).get("target_policy"))
    result.update(status="complete", metadata=metadata)
    return result


def _rows(path, split):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    expected_count, expected_positive = POPULATIONS[split]
    if len(rows) != expected_count or len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("evaluation pair rows differ from the fixed population/unique IDs")
    if any(type(row.get("label")) is not bool for row in rows) or sum(row["label"] for row in rows) != expected_positive:
        raise ValueError("evaluation pair labels differ from the fixed population counts")
    identities = [(row["pair_id"], row["fragment_a"], row["fragment_b"], row["label"]) for row in rows]
    if any(not isinstance(value, str) or not value for row in identities for value in row[:3]):
        raise ValueError("ordered endpoint identity is missing")
    # Ordering of the rows is not a scientific difference; endpoint direction is.
    digest = hashlib.sha256(json.dumps(sorted(identities), separators=(",", ":")).encode()).hexdigest()
    return rows, digest


def _classification(values, positive, negative):
    result = {key: _finite(values.get(key), "classification." + key) for key in CLASS_FIELDS}
    if result["tp"] + result["fn"] != positive or result["fp"] + result["tn"] != negative:
        raise ValueError("classification counts do not cover the evaluation population")
    return result


def _joint(values, classification, positive):
    result = {key: _finite(values.get(key), "joint10." + key) for key in JOINT_FIELDS}
    if (result["tp"] + result["fn"] != positive or result["tp"] > classification["tp"]
            or result["tp"] + result["fp"] != classification["tp"] + classification["fp"]):
        raise ValueError("joint10 counts do not correspond to the same frozen classifier")
    return result


def native_result(evaluation_root, split, training):
    root = Path(evaluation_root) / split
    sources = {name: str(root / filename) for name, filename in (
        ("summary", "summary.json"), ("receipt", "receipt.json"), ("pair_results", "pair_results.jsonl"))}
    summary, receipt = _read(sources["summary"]), _read(sources["receipt"])
    if training["status"] != "complete":
        return _pending(training["status"], "training is not complete", sources)
    if summary is None:
        progress = _read(root / "protocol.json")
        return _pending("not_run" if progress is None else ("failed" if progress.get("status") == "failed" else "pending"),
                        "no completed evaluation summary", sources)
    if summary.get("status") != "complete" or receipt is None or receipt.get("status") != "complete" or not Path(sources["pair_results"]).is_file():
        return _pending("pending", "summary, receipt, and pair results must all be complete", sources)
    expected, positives = POPULATIONS[split]
    if (summary.get("sample_count"), summary.get("positive_count"), summary.get("split")) != (expected, positives, split):
        raise ValueError("summary population/split differs")
    if (summary.get("selected_full_decoder") != DECODER or summary.get("test_or_real_used_for_fit") is not False
            or receipt.get("test_or_real_used_for_fit") is not False):
        raise ValueError("native result must use frozen VAL decision and fixed Top2 decoder")
    if (not summary.get("checkpoint_sha256") or summary["checkpoint_sha256"] != receipt.get("checkpoint_sha256")
            or not receipt.get("pair_results_sha256")):
        raise ValueError("summary and receipt source identities differ")
    metadata = training["metadata"]
    for saved, selected in (("checkpoint_epoch", "selected_epoch"), ("global_exposure", "selected_global_exposure"),
                            ("optimizer_updates", "selected_optimizer_updates"), ("validation_event", "selected_validation_event")):
        if receipt.get(saved) != metadata[selected]:
            raise ValueError("evaluation receipt does not refer to this arm's selected winner: " + saved)
    thresholds = metadata["classifier_thresholds"]
    if summary.get("branch_validation_thresholds") != thresholds:
        raise ValueError("evaluation thresholds differ from the arm's frozen VAL thresholds")
    classification = _classification(summary["classification"]["fused"]["at_original_frozen_threshold"], positives, expected - positives)
    if classification["threshold"] != thresholds["fused"]:
        raise ValueError("native fused threshold differs")
    layout = summary["layout"][DECODER]
    recall = {radius: _finite(layout["recall"].get(radius), "layout R" + radius) for radius in ("2", "5", "10")}
    joint = _joint(layout["assembly"]["10"], classification, positives)
    rows, identity = _rows(sources["pair_results"], split)
    curves = recall_at_fpr([row["label"] for row in rows], [row["classification"]["fused"] for row in rows])
    return dict(status="complete", sources=sources, sample_count=expected, positive_count=positives,
                checkpoint_sha256=summary["checkpoint_sha256"], ordered_population_identity_sha256=identity,
                selected_head="native_fused", metrics=dict(classification=classification,
                recall_at_fpr=curves, unconditional_layout_recall=recall, joint10=joint))


def _safe(call, sources):
    try:
        return call()
    except (ValueError, KeyError, TypeError, OSError) as error:
        return _pending("invalid", str(error), sources)


def _arm_status(training, results):
    states = [training["status"]] + [result["status"] for result in results.values()]
    if "invalid" in states:
        return "invalid"
    if "failed" in states:
        return "failed"
    if all(status == "complete" for status in states):
        return "complete"
    return "not_run" if all(status == "not_run" for status in states) else "pending"


def native_arm(name, training_root, evaluation_root):
    metadata = _safe(lambda: training_metadata(training_root), {"training_root": str(training_root)})
    splits = {split: _safe(lambda split=split: native_result(evaluation_root, split, metadata),
                          {"evaluation_root": str(Path(evaluation_root) / split)}) for split in POPULATIONS}
    return dict(arm=name, status=_arm_status(metadata, splits), selected_head="native_fused",
                training=metadata, splits=splits)


def e3_training_metadata(root, matcher_training):
    root = Path(root)
    path = root / "validation_freeze.json"
    sources = dict(validation_freeze=str(path), head_protocol=str(root / "protocol.json"))
    freeze = _read(path)
    result = dict(status="not_run", metadata=None, sources=sources,
                  frozen_matcher_training=matcher_training, head_training_metadata=None)
    if freeze is None:
        protocol = _read(root / "protocol.json")
        if protocol is not None:
            result.update(status="failed" if protocol.get("status") == "failed" else "pending",
                          reason="E3 has no completed VAL head freeze")
        return result
    if freeze.get("status") != "complete":
        result.update(status="failed" if freeze.get("status") == "failed" else "pending",
                      reason="E3 VAL head freeze is not complete")
        return result
    if freeze.get("test_or_real_used_for_fit") is not False or freeze.get("primary_method") != "score_geometry":
        raise ValueError("E3 must retain the prespecified geometry head and VAL-only fit")
    if (freeze.get("planned_epochs"), freeze.get("completed_epochs"), freeze.get("train_unique_count"),
            freeze.get("completed_pair_exposures_per_head"), freeze.get("completed_optimizer_updates_per_head")) != E3_BUDGET:
        raise ValueError("E3 complete receipt does not match its fixed small-head training budget")
    if matcher_training["status"] != "complete":
        result.update(status="pending", reason="the fixed E2 matcher training receipt is not complete")
        return result
    selected = freeze["selected"]
    for head in ("score_only", "score_geometry"):
        choice = selected[head]
        for field in ("epoch", "threshold", "val_f1", "val_auprc"):
            _finite(choice.get(field), "E3 selected." + head + "." + field)
        if not 1 <= choice["epoch"] <= 10:
            raise ValueError("E3 head winner is outside its ten VAL opportunities")
    if not freeze.get("matcher_checkpoint_id"):
        raise ValueError("E3 freeze lacks frozen matcher identity")
    metadata = {key: freeze[key] for key in ("matcher_checkpoint_id", "primary_method", "selected",
        "planned_epochs", "completed_epochs", "train_unique_count", "completed_pair_exposures_per_head",
        "completed_optimizer_updates_per_head")}
    metadata.update(training_scope="two small CPU decision heads; frozen E2 matcher, no additional matcher updates",
                    validation_freeze_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    selection_rule=freeze.get("selection_rule"), seed=freeze.get("seed"),
                    selected_head_checkpoint_paths=freeze.get("head_checkpoint_paths"),
                    selected_head_checkpoint_sha256=freeze.get("head_checkpoint_sha256"),
                    raw_freeze=freeze)
    if freeze.get("head_checkpoint_path"):
        sources["selected_head_checkpoint"] = freeze["head_checkpoint_path"]
    result.update(status="complete", head_training_metadata=metadata)
    return result


def e3_result(root, split, training, matcher_result):
    root = Path(root) / split
    sources = {name: str(root / filename) for name, filename in (
        ("summary", "summary.json"), ("receipt", "receipt.json"), ("pair_results", "pair_results.jsonl"))}
    if training["status"] != "complete":
        return _pending(training["status"], "E3 head training is not complete", sources)
    saved, receipt = _read(sources["summary"]), _read(sources["receipt"])
    if saved is None:
        return _pending("not_run", "E3 split has no complete summary", sources)
    if saved.get("status") != "complete" or receipt is None or receipt.get("status") != "complete" or not Path(sources["pair_results"]).is_file():
        return _pending("pending", "E3 summary, receipt, and pair rows must all be complete", sources)
    if matcher_result["status"] != "complete":
        return _pending("pending", "fixed E2 matcher evaluation not complete", sources)
    expected, positives = POPULATIONS[split]
    if (saved.get("sample_count"), saved.get("positive_count"), saved.get("negative_count")) != (expected, positives, expected - positives):
        raise ValueError("E3 summary population differs")
    head_training = training["head_training_metadata"]
    if (saved.get("test_or_real_used_for_fit") is not False
            or saved.get("matcher_checkpoint_id") != head_training["matcher_checkpoint_id"]
            or saved.get("checkpoint_sha256") != matcher_result["checkpoint_sha256"]
            or saved.get("validation_freeze_sha256") != head_training["validation_freeze_sha256"]):
        raise ValueError("E3 summary is not bound to the frozen E2 matcher and selected head freeze")
    if (receipt.get("validation_freeze_sha256") != saved["validation_freeze_sha256"]
            or receipt.get("checkpoint_sha256") != saved["checkpoint_sha256"]):
        raise ValueError("E3 receipt and summary model/freeze identities differ")
    head_sha = head_training.get("selected_head_checkpoint_sha256")
    if head_sha and (saved.get("head_checkpoint_sha256") != head_sha or receipt.get("head_checkpoint_sha256") != head_sha):
        raise ValueError("E3 receipt and summary head checkpoint identities differ")
    rows, identity = _rows(sources["pair_results"], split)
    if identity != matcher_result["ordered_population_identity_sha256"]:
        raise ValueError("E3 rows differ from the ordered E2 matcher population")
    pose = saved["layout"][DECODER]
    layout = {radius: _finite(pose["recall"].get(radius), "E3 layout R" + radius) for radius in ("2", "5", "10")}
    if layout != matcher_result["metrics"]["unconditional_layout_recall"]:
        raise ValueError("E3 changed supposedly fixed E2 layout recalls")
    methods = {}
    for head in ("score_geometry", "score_only"):
        data, choice = saved["methods"][head], head_training["selected"][head]
        if data["threshold"] != choice["threshold"]:
            raise ValueError("E3 " + head + " threshold differs from its selected VAL event")
        classification = _classification(dict(data["classification"], threshold=data["threshold"]), positives, expected - positives)
        joint = _joint(data["joint"]["10"], classification, positives)
        curves = {key: _finite(data["recall_at_fpr"].get(key, data["recall_at_fpr"].get("0.1" if key == "0.10" else key)), "E3 ROC " + key)
                  for key in ("0.01", "0.05", "0.10")}
        for row in rows:
            _finite(row["head_scores"][head], "E3 frozen head score")
        methods[head] = dict(status="complete", sources=sources, selected_head=head,
                            metrics=dict(classification=classification, recall_at_fpr=curves,
                                         unconditional_layout_recall=layout, joint10=joint))
    primary = methods["score_geometry"]
    primary.update(sample_count=expected, positive_count=positives, checkpoint_sha256=saved["checkpoint_sha256"],
                   ordered_population_identity_sha256=identity, controls={"score_only": methods["score_only"]})
    return primary


def e3_arm(root, matcher):
    training = _safe(lambda: e3_training_metadata(root, matcher["training"]), {"validation_freeze": str(Path(root) / "validation_freeze.json")})
    splits = {split: _safe(lambda split=split: e3_result(root, split, training, matcher["splits"][split]),
                          {"evaluation_root": str(Path(root) / split)}) for split in POPULATIONS}
    for result in splits.values():
        if "controls" not in result:
            result["controls"] = {"score_only": _pending(result["status"], result.get("reason"), result["sources"])}
    return dict(arm="E3", status=_arm_status(training, splits), selected_head="score_geometry",
                training=training, splits=splits)


def build_summary(args):
    root = Path(args.root).resolve()
    e0, e1 = Path(args.e0_root).resolve(), Path(args.e1_root).resolve()
    e2, e3 = Path(args.e2_root or root / "e2").resolve(), Path(args.e3_root or root / "e3").resolve()
    arms = dict(E0=native_arm("E0", e0 / "realism_training" / "matched24k", e0 / "realism_evaluation" / "matched24k"),
                E1=native_arm("E1", e1 / "training" / "e1", e1 / "evaluation"),
                E2=native_arm("E2", e2 / "training" / "e2", e2 / "evaluation"))
    arms["E3"] = e3_arm(e3, arms["E2"])
    for split in POPULATIONS:
        reference = arms["E0"]["splits"][split]
        for name, arm in arms.items():
            result = arm["splits"][split]
            if reference["status"] == result["status"] == "complete":
                compatible = result["ordered_population_identity_sha256"] == reference["ordered_population_identity_sha256"]
                result["same_ordered_population_as_e0"] = compatible
                if not compatible:
                    result.update(status="invalid", reason="ordered endpoints/labels differ from E0", metrics=None)
                    for control in result.get("controls", {}).values():
                        control.update(status="invalid", reason=result["reason"], metrics=None)
                    arm["status"] = "invalid"
    return dict(schema_version=SCHEMA, created_at=datetime.now(timezone.utc).isoformat(), arms=arms,
                complete=all(arm["status"] == "complete" for arm in arms.values()),
                interpretation=dict(main_heads="E0/E1/E2 native fused; E3 fixed score+geometry head; score-only is an attribution control",
                    threshold_policy="Only each preselected head's clean VAL-frozen threshold; no REAL refitting or best-head selection",
                    recall_at_fpr_policy="ROC diagnostic: maximum empirical recall with FPR <= 1/5/10%, no tie interpolation; not a deployment operating point",
                    layout_policy="unconditional among all positives; missing/invalid pose is a failure; same fixed Top2 decoder",
                    joint_policy="accepted AND positive AND <=10px correct placement; rejected positives and invalid poses are failures",
                    budget_policy="record both completed budget and VAL-selected checkpoint exposure; historical warm-start training excluded",
                    evidence_policy="completed freeze + completed evaluation summary/receipt + fixed row-count/ordered-endpoint compatibility; no model rerun",
                    generalization_caveat="REAL has informed experiment design; this is exploratory comparison, not a fresh untouched confirmatory holdout"))


def _pct(value):
    return "—" if value is None else "%.2f" % (100 * value)


def _link(path):
    return "[" + Path(path).name + "](<" + str(path) + ">)"


def render_markdown(summary):
    lines = ["# E0–E3 weathering ablations", "", "Status: " + ("complete" if summary["complete"] else "incomplete; pending results are not measurements"), "",
             "E0/E1/E2 use native fused. E3's primary head is fixed score+geometry; score-only is an attribution control, not a REAL-selected alternative.", "",
             "## Training budget and selected checkpoint", "",
             "| Arm | Status | Unique pool | Completed exposures / updates / epochs / VAL | Selected exposure / updates / epoch / VAL | Sources |",
             "|---|---|---:|---|---|---|"]
    initialization_notes = []
    for name, arm in summary["arms"].items():
        training = arm["training"]
        metadata = training.get("metadata")
        source = " · ".join(_link(path) for path in training.get("sources", {}).values())
        if metadata:
            completed = " / ".join(str(metadata.get(key, "—")) for key in ("completed_global_exposures", "completed_optimizer_updates", "completed_dataset_epochs", "completed_validation_events"))
            selected = " / ".join(str(metadata.get(key, "—")) for key in ("selected_global_exposure", "selected_optimizer_updates", "selected_epoch", "selected_validation_event"))
            lines.append(f"| {name} | {arm['status']} | {metadata.get('unique_count', '—')} | {completed} | {selected} | {source} |")
            if metadata.get("initial_checkpoint"):
                initialization_notes += [name + " initialization: " + _link(metadata["initial_checkpoint"]) + "."]
        else:
            lines.append(f"| {name} | {arm['status']} | — | — | — | {source} |")
    for note in initialization_notes:
        lines += ["", note]
    e3 = summary["arms"]["E3"]["training"]
    head = e3.get("head_training_metadata")
    if head:
        lines += ["", "E3 is head-only training on the fixed E2 VAL-selected matcher, not another 120k-exposure matcher arm.", "",
                  "Each small head: %s unique TRAIN pairs; %s epochs; %s pair exposures; %s optimizer updates." %
                  (head["train_unique_count"], head["completed_epochs"], head["completed_pair_exposures_per_head"], head["completed_optimizer_updates_per_head"])]
        for method in ("score_geometry", "score_only"):
            choice = head["selected"][method]
            lines += ["", "E3 %s: selected epoch %s; clean VAL F1 %s%% / AP %s%%; frozen threshold %.8g." %
                      (method, choice["epoch"], _pct(choice["val_f1"]), _pct(choice["val_auprc"]), choice["threshold"])]
    for split in POPULATIONS:
        lines += ["", "## " + split.upper(), "", "All rates below are percentages; TP is a count.", "",
                  "| Arm / head | Status | P | R | F1 | AP | ROC-AUC | R@FPR1% | R@FPR5% | R@FPR10% | Layout R2 | R5 | R10 | Joint10 TP | P | R | F1 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, arm in summary["arms"].items():
            result = arm["splits"][split]
            methods = [(arm["selected_head"], result)] + list(result.get("controls", {}).items())
            for head, row in methods:
                metrics = row.get("metrics")
                if metrics is None:
                    lines.append("| " + name + " / " + head + " | " + row["status"] + " | " + " | ".join(["—"] * 15) + " |")
                    continue
                c, curves, layout, joint = (metrics[key] for key in ("classification", "recall_at_fpr", "unconditional_layout_recall", "joint10"))
                values = [_pct(c[key]) for key in ("precision", "recall", "f1", "auprc", "auroc")]
                values += [_pct(curves[key]) for key in ("0.01", "0.05", "0.10")]
                values += [_pct(layout[key]) for key in ("2", "5", "10")]
                values += [str(joint["tp"])] + [_pct(joint[key]) for key in ("precision", "recall", "f1")]
                lines.append("| " + name + " / " + head + " | " + row["status"] + " | " + " | ".join(values) + " |")
        for name, arm in summary["arms"].items():
            result = arm["splits"][split]
            lines += ["", name + ": " + " · ".join(_link(path) for path in result.get("sources", {}).values()) +
                      (" — " + result["reason"] if result.get("reason") else "")]
    lines += ["", "## Interpretation limits", ""]
    lines += ["- " + value for value in summary["interpretation"].values()]
    return "\n".join(lines) + "\n"


def run(args):
    summary = build_summary(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(dict(output=str(output), complete=summary["complete"], statuses={key: value["status"] for key, value in summary["arms"].items()})))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=DEFAULT_ALL)
    p.add_argument("--e0-root", default=DEFAULT_E0)
    p.add_argument("--e1-root", default=DEFAULT_E1)
    p.add_argument("--e2-root")
    p.add_argument("--e3-root")
    p.add_argument("--output", required=True, help="new report directory; existing snapshots never overwritten")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
