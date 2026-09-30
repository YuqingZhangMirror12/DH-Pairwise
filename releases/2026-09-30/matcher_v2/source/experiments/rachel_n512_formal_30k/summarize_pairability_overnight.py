"""Read-only extraction of saved overnight metrics into JSON and Chinese Markdown.

No Torch/NumPy imports, checkpoint/tensor reads, refitting, decoder
selection, REAL winner selection, or remote connections. Small completed pair
JSONLs may be joined1:1 to derive classifier+predeclared Top2 joint metrics.
Run this same script
on the server with its local --root to summarize remote experiment outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

SCHEMA = "rachel-pairability-overnight-summary/1"
HEADS = ("matrix_soft", "matrix_transport_only")
INFONCE_SCHEMA = "rachel-coarse-infonce-projection/v1"
INFONCE_POLICIES = ("infonce", "frozen_feature_cosine")
ARMS = ("original24k", "matched24k", "realism60k")
POPULATIONS = ("val", "test", "real_balanced1016", "real_strict547")
COUNTS = {"val": (3000, 1500), "test": (3000, 1500),
          "real_balanced1016": (1016, 508), "real_strict547": (547, 508)}
DECODERS = ("top1_mode", "top2_mode", "top5_mode", "hungarian_full_mode",
            "hungarian_partial_mode", "hungarian_partial_log_mode")
REGIMES = {"original_frozen": "旧模型原冻结阈值", "val_recalibrated": "本次 VAL 重校准",
           "own_model_val_frozen": "该新模型自己的 VAL 冻结阈值", "fixed_0_5": "固定 0.5（非旧冻结阈值）"}
POP_LABELS = {"val": "仿真 VAL（3000/1500 正例）", "test": "仿真 TEST（3000/1500 正例）",
              "real_balanced1016": "REAL balanced（1016/508 正例）", "real_strict547": "REAL strict（547/508 正例）"}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _classification(metrics):
    result = {key: _number(metrics.get(key)) for key in ("threshold", "accuracy", "precision", "recall", "f1", "auroc", "tp", "fp", "fn", "tn")}
    result["ap"] = _number(metrics.get("auprc"))  # these writers store average precision under auprc
    if result["accuracy"] is None and all(result[key] is not None for key in ("tp", "fp", "fn", "tn")):
        total = sum(result[key] for key in ("tp", "fp", "fn", "tn"))
        if total > 0:
            result["accuracy"] = (result["tp"] + result["tn"]) / total
    return result


def _base(stage, population, source):
    n, positives = COUNTS[population]
    return dict(stage=stage, population=population, status="pending", source=str(source),
                expected_sample_count=n, expected_positive_count=positives,
                sample_count=None, positive_count=None, classification=[], layout=[], joint=[],
                provenance={}, issues=[])


def _completed(path, node):
    if not path.is_file():
        node["issues"].append("completed source not present")
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError) as error:
        node.update(status="needs_attention", issues=["cannot parse saved JSON: " + str(error)])
        return None
    if not isinstance(data, dict):
        node.update(status="needs_attention", issues=["saved source must be a JSON object"])
        return None
    if data.get("status") != "complete":
        node["issues"].append("source status is " + str(data.get("status", "not declared complete")))
        if data.get("status") == "failed":
            node["status"] = "needs_attention"
        return None
    return data


def _population(node, payload):
    if not isinstance(payload, dict):
        node["issues"].append("requested population summary not present")
        return False
    node["sample_count"], node["positive_count"] = payload.get("sample_count"), payload.get("positive_count")
    actual = node["sample_count"], node["positive_count"]
    expected = node["expected_sample_count"], node["expected_positive_count"]
    if actual != expected:
        node.update(status="needs_attention", issues=["population counts differ: %s != %s" % (actual, expected)])
        return False
    node["status"] = "complete"
    return True


def _head(root, name, population):
    split = "real" if population.startswith("real_") else population
    # TEST/REAL require the declared native-probability correction; never fall
    # back to an old FP64-reconstructed AP while that correction is pending.
    original = root / "heads" / (name if split == "val" else name + "_" + split) / split / "metrics.json"
    source = original if split == "val" else root / "heads" / (name + "_" + split + "_native") / split / "metrics.json"
    node = _base(name, population, source)
    node["uncorrected_source"] = str(original) if split != "val" else None
    data = _completed(source, node)
    if data is None:
        if split != "val":
            node["issues"].append("native-probability correction required; old AP not substituted")
        return node
    native = data.get("native_probability_reporting") is True
    if split != "val" and not native:
        node.update(status="needs_attention", issues=["preferred native source lacks native_probability_reporting=true"])
        return node
    payload = data.get("strict_classification" if population == "real_strict547" else "classification")
    if not _population(node, payload):
        return node
    freeze_path = root / "heads" / name / "validation_freeze.json"
    freeze = _completed(freeze_path, dict(status="pending", issues=[]))
    if freeze is None:
        node.update(status="pending", issues=["completed head validation freeze not present"])
        return node
    if data.get("split") != split or data.get("test_or_real_used_for_fit") is not False:
        node.update(status="needs_attention", issues=["split or no-heldout-fit declaration differs from writer contract"])
        return node
    policies = freeze.get("policies", {})
    if data.get("cache", {}).get("matcher_checkpoint_id") != freeze.get("matcher_checkpoint_id"):
        node.update(status="needs_attention", issues=["head metrics and own freeze identify different matchers"])
        return node
    node["provenance"] = dict(validation_freeze=str(freeze_path), seed=freeze.get("seed"),
        matcher_checkpoint_id=data.get("cache", {}).get("matcher_checkpoint_id"),
        precision=data.get("cache", {}).get("precision"), selected_epochs=data.get("selected_epochs"),
        pose_evaluated=False, sampled_seam_gt_evaluated=False,
        native_probability_reporting=native,
        probability_reporting_status=data.get("probability_reporting_status", "uncorrected_logits_reconstruction_for_existing_branches"),
        probability_precision_caveat=("original-network branches use exact cached FP32 probabilities; frozen weights/thresholds unchanged" if native else
            "VAL existing_* use float64 sigmoid of FP32 logits, while matrix_* use FP32 sigmoid; saturation ties/ranking AP can differ from old native FP32 probabilities"),
        thresholds_modified=data.get("thresholds_modified"), weights_modified=data.get("weights_modified"),
        cascade_caveat=data.get("cascade_caveat"), original_threshold_comparison_available=data.get("original_threshold_comparison_available"))
    node["layout_status"] = "not_evaluated_by_pair_head"
    node["joint_status"] = "not_evaluated_by_pair_head"
    for policy_name, metrics in payload.get("policies", {}).items():
        policy = policies.get(policy_name, {})
        regime = "original_frozen" if policy_name.endswith("_original_frozen") else "val_recalibrated"
        branch = policy.get("branch", policy_name)
        row = dict(policy=policy_name, branch=branch, threshold_regime=regime, **_classification(metrics),
                   gate_threshold=_number(policy.get("gate_threshold")), threshold_fit=policy.get("threshold_fit"))
        if not native and (branch.startswith("existing_") or policy.get("gate_threshold") is not None):
            row["ap_comparison_caveat"] = "not directly comparable to old native-FP32-probability AP until corrected"
        for key in ("gate_rejected_positives", "gate_rejected_negatives",
                    "previously_accepted_positives_lost_to_gate", "previously_accepted_negatives_removed_by_gate"):
            if key in metrics:
                row[key] = _number(metrics[key])
        node["classification"].append(row)
    node["unavailable_policies"] = payload.get("unavailable_policies", {})
    if not node["classification"]:
        node.update(status="needs_attention", issues=["complete head population has no policy metrics"])
    return node


def _score_fusion(root, population):
    split = "real" if population.startswith("real_") else population
    original = root / "score_fusion" / split / "metrics.json"
    source = original if split == "val" else root / "score_fusion" / (split + "_native") / "metrics.json"
    node = _base("score_fusion", population, source)
    node["uncorrected_source"] = str(original) if split != "val" else None
    data = _completed(source, node)
    if data is None or not _population(node, data.get("strict_classification" if population == "real_strict547" else "classification")):
        if split != "val":
            node["issues"].append("native-probability correction required; old AP not substituted")
        return node
    native = data.get("native_probability_reporting") is True
    if split != "val" and not native:
        node.update(status="needs_attention", issues=["preferred native source lacks native_probability_reporting=true"])
        return node
    freeze_path = root / "score_fusion/val/validation_freeze.json"
    freeze = _completed(freeze_path, dict(status="pending", issues=[]))
    if freeze is None:
        node.update(status="pending", issues=["completed scalar fusion validation freeze not present"])
        return node
    if data.get("split") != split or data.get("test_or_real_used_for_fit") is not False or data.get("matcher_checkpoint_id") != freeze.get("matcher_checkpoint_id"):
        node.update(status="needs_attention", issues=["scalar fusion split/freeze declaration disagrees"])
        return node
    node["provenance"] = dict(validation_freeze=str(freeze_path), matcher_checkpoint_id=freeze.get("matcher_checkpoint_id"),
        precision=freeze.get("precision"), coefficients_fit="TRAIN", thresholds_fit="VAL",
        native_probability_reporting=native,
        probability_reporting_status=data.get("probability_reporting_status", "uncorrected_logits_reconstruction"),
        thresholds_modified=data.get("thresholds_modified"), weights_modified=data.get("weights_modified"),
        new_affine_or_average_policy_definitions_unchanged=data.get("new_affine_or_average_policy_definitions_unchanged"),
        score_source=data.get("score_source"), no_model_forward=True,
        not_pairingnet_infonce_training=True, policies=freeze.get("policies"))
    node["layout_status"] = node["joint_status"] = "not_evaluated_by_scalar_fusion_control"
    payload = data["strict_classification" if population == "real_strict547" else "classification"]
    for name, metrics in payload.get("policies", {}).items():
        policy = freeze.get("policies", {}).get(name, {})
        node["classification"].append(dict(policy=name, branch=policy.get("branch", name),
            threshold_regime="original_frozen" if name.endswith("_original_frozen") else "val_recalibrated",
            threshold_fit=policy.get("threshold_fit", "VAL equal-row F1"), **_classification(metrics)))
    return node


def _coarse_infonce(root, population):
    """Optional frozen-feature projection control; cosine is not probability."""
    split = "real" if population.startswith("real_") else population
    freeze_path = root / "coarse_infonce/training/validation_freeze.json"
    source = freeze_path if split == "val" else root / "coarse_infonce/evaluation" / split / "metrics.json"
    node = _base("coarse_infonce", population, source)
    freeze = _completed(freeze_path, node)
    if freeze is None:
        return node
    if (freeze.get("schema_version") != INFONCE_SCHEMA or freeze.get("selected_on") != "val"
            or freeze.get("test_or_real_used_for_fit") is not False or freeze.get("encoder_frozen") is not True
            or freeze.get("train_count") != 24000 or freeze.get("val_count") != 3000
            or freeze.get("completed_epochs") != 10):
        node.update(status="needs_attention", issues=["InfoNCE freeze differs from completed VAL-only frozen-encoder contract"])
        return node
    node["provenance"] = dict(validation_freeze=str(freeze_path),
        matcher_checkpoint_id=freeze.get("matcher_checkpoint_id"), selected_epoch=freeze.get("selected_epoch"),
        seed=freeze.get("config", {}).get("seed"), completed_epochs=freeze.get("completed_epochs"),
        completed_positive_pair_exposures=freeze.get("completed_positive_pair_exposures"),
        encoder_frozen=True, encoder_pretraining="original BCE pair classifier",
        projection_trained_with_infonce=True, device="cpu", feature_layer=freeze.get("feature_layer"),
        selection_rule=freeze.get("selection_rule"), thresholds_fit="VAL equal-pair F1",
        score_type="cosine [-1,1], not calibrated pairability probability",
        paper_reproduction=False, gallery_retrieval_evaluated=False,
        limitation="frozen BCE coarse CNN + trainable InfoNCE projection, not full PairingNet")
    node["layout_status"] = node["joint_status"] = "not_evaluated_by_infonce_projection"
    if split == "val":
        policies = dict(infonce=freeze.get("validation"), frozen_feature_cosine=freeze.get("baseline_validation"))
        first = policies["infonce"] if isinstance(policies["infonce"], dict) else {}
        positives = (first["tp"] + first["fn"] if _number(first.get("tp")) is not None
                     and _number(first.get("fn")) is not None else None)
        payload = dict(sample_count=freeze.get("val_count"), positive_count=positives, policies=policies)
    else:
        data = _completed(source, node)
        if data is None:
            return node
        if (data.get("split") != split or data.get("test_or_real_used_for_fit") is not False
                or data.get("matcher_checkpoint_id") != freeze.get("matcher_checkpoint_id")
                or data.get("selected_epoch") != freeze.get("selected_epoch")
                or data.get("encoder_frozen") is not True or data.get("projection_trained_with_infonce") is not True):
            node.update(status="needs_attention", issues=["InfoNCE evaluation and own freeze disagree"])
            return node
        payload = data.get("strict_classification" if population == "real_strict547" else "classification")
    if not _population(node, payload):
        return node
    for name in INFONCE_POLICIES:
        metrics = payload.get("policies", {}).get(name)
        threshold = _number(freeze.get("thresholds", {}).get(name))
        if (not isinstance(metrics, dict) or threshold is None or _number(metrics.get("threshold")) != threshold):
            node.update(status="needs_attention", classification=[], issues=["InfoNCE policy metrics/VAL-frozen threshold missing or inconsistent"])
            return node
        counts = [_number(metrics.get(key)) for key in ("tp", "fp", "fn", "tn")]
        if (any(value is None for value in counts) or sum(counts) != node["sample_count"]
                or counts[0] + counts[2] != node["positive_count"]):
            node.update(status="needs_attention", classification=[], issues=["InfoNCE policy confusion counts differ from population"])
            return node
        node["classification"].append(dict(policy=name, branch=name, threshold_regime="own_model_val_frozen",
            threshold_fit="VAL equal-pair F1", score_type="cosine [-1,1], not probability", **_classification(metrics)))
    return node


def _common(root, stage, population):
    split = "real" if population.startswith("real_") else population
    source = (root / "assignment" / split / "summary.json" if stage == "assignment" else
              root / "realism_evaluation" / stage / split / "summary.json")
    node = _base(stage, population, source)
    data = _completed(source, node)
    if data is None:
        return node
    payload = data.get("strict_summary") if population == "real_strict547" else data
    if not _population(node, payload):
        return node
    if data.get("split") != split:
        node.update(status="needs_attention", issues=["summary split differs from requested population"])
        return node
    if data.get("probe_only") is True:
        node.update(status="needs_attention", issues=["probe-only results are not full-population experiment results"])
        return node
    regime = "original_frozen" if stage == "assignment" else "own_model_val_frozen"
    node["provenance"] = {key: data.get(key) for key in ("selected_full_decoder", "selection_source", "precision",
        "checkpoint_sha256", "original_thresholds", "branch_validation_thresholds", "original_fused_threshold",
        "classifier_threshold_fitted", "original_probabilities_preserved", "test_or_real_used_for_fit")}
    if stage == "assignment":
        freeze_path = root / "assignment/val/validation_freeze.json"
        if freeze_path.is_file():
            try:
                freeze = read_json(freeze_path)
                node["provenance"].update(validation_freeze=str(freeze_path),
                    matcher_checkpoint_id=freeze.get("matcher_checkpoint_id"), precision=freeze.get("precision"),
                    selection_rule=freeze.get("selection_rule"))
            except (OSError, ValueError):
                node["issues"].append("assignment freeze unreadable; source metrics retained without new selection")
    else:
        node["provenance"]["fused_original_frozen_key_meaning"] = "alias of this NEW model's own VAL-fitted fused threshold; not the old candidate threshold"
    for branch, variants in payload.get("classification", {}).items():
        for key, metrics in variants.items():
            if key not in ("at_0_5", "at_original_frozen_threshold", "at_validation_row_f1_threshold"):
                continue
            alias = stage != "assignment" and key == "at_original_frozen_threshold"
            threshold_regime = "fixed_0_5" if key == "at_0_5" else regime
            node["classification"].append(dict(branch=branch, policy=key, threshold_regime=threshold_regime,
                duplicate_of_own_val_fused=alias, **_classification(metrics)))
    selected = data.get("selected_full_decoder")
    layouts = payload.get("layout", {})
    order = sorted(layouts, key=lambda key: (DECODERS.index(key) if key in DECODERS else len(DECODERS), key))
    for decoder in order:
        metrics = layouts[decoder]
        node["layout"].append(dict(decoder=decoder, predeclared_selected_decoder=decoder == selected,
            coverage=_number(metrics.get("positive_pose_coverage")),
            median_px_conditional=_number(metrics.get("median_px_conditional")),
            p90_px_conditional=_number(metrics.get("p90_px_conditional")),
            **{"r" + str(t): _number(metrics.get("recall", {}).get(str(t))) for t in (2, 5, 8, 10)}))
        for tolerance in (2, 5, 8, 10):
            joint = metrics.get("assembly", {}).get(str(tolerance))
            if joint is not None:
                node["joint"].append(dict(decoder=decoder, tolerance_px=tolerance,
                    classifier_branch="fused", threshold_regime=regime,
                    **{key: _number(joint.get(key)) for key in ("precision", "recall", "f1", "tp", "fp", "fn")}))
    if not node["classification"] or not node["layout"]:
        node.update(status="needs_attention", issues=["completed geometry summary lacks classification or layout metrics"])
    return node


def _data_validation(root, arm):
    source = root / "realism_training" / arm / "train_val_freeze.json"
    node = _base(arm, "val", source)
    data = _completed(source, node)
    if data is None:
        return node
    if not _population(node, data.get("validation")):
        return node
    node["provenance"] = {key: data.get(key) for key in ("seed", "precision", "selected_epoch",
        "selected_global_exposure", "selected_optimizer_updates", "selected_validation_event",
        "unique_count", "selection_rule", "checkpoint", "initial_checkpoint",
        "completed_global_exposures", "completed_optimizer_updates", "completed_dataset_epochs",
        "completed_validation_events", "test_or_real_used_for_fit")}
    node["layout_status"] = node["joint_status"] = "not_evaluated_by_training_validation"
    node["classification"] = [dict(branch=branch, policy="selected_own_validation_f1_threshold",
        threshold_regime="own_model_val_frozen", **_classification(metrics))
        for branch, metrics in data["validation"].get("methods", {}).items()]
    return node


def _data_training_metadata(node):
    """Separate a completed run's budget from its VAL-selected weight history.

    Only _data_validation's completed-freeze provenance is used. In particular,
    neither an unfinished winner nor a planned budget is presented as complete.
    No checkpoint file or other training-history source is opened here.
    """
    fields = ("unique_count", "completed_global_exposures", "completed_optimizer_updates",
              "completed_dataset_epochs", "completed_validation_events", "selected_global_exposure",
              "selected_optimizer_updates", "selected_epoch", "selected_validation_event")
    result = dict(arm=node["stage"], status=node["status"], source=node["source"],
        **{key: None for key in fields}, checkpoint=None, initial_checkpoint=None,
        checkpoint_artifact_kind="validation_selected_winner", last_checkpoint_saved_by_trainer=False,
        warm_start_history_included_in_exposures=False,
        selected_checkpoint_is_end_of_run_state=None,
        selected_checkpoint_completed_one_pool_pass_in_this_run=None,
        issues=list(node["issues"]))
    if node["status"] != "complete":
        return result
    saved = node["provenance"]
    result.update({key: _number(saved.get(key)) for key in fields})
    result.update(checkpoint=saved.get("checkpoint"), initial_checkpoint=saved.get("initial_checkpoint"),
                  selection_rule=saved.get("selection_rule"))
    if any(type(result[key]) is not int or result[key] <= 0 for key in fields):
        result.update(status="needs_attention", issues=["complete training freeze lacks positive integer budget/selection counters"])
        return result
    if (result["selected_global_exposure"] > result["completed_global_exposures"]
            or result["selected_optimizer_updates"] > result["completed_optimizer_updates"]
            or result["selected_epoch"] > result["completed_dataset_epochs"]
            or result["selected_validation_event"] > result["completed_validation_events"]):
        result.update(status="needs_attention", issues=["selected weight counters exceed the completed training run"])
        return result
    result["selected_checkpoint_is_end_of_run_state"] = (
        result["selected_global_exposure"] == result["completed_global_exposures"]
        and result["selected_optimizer_updates"] == result["completed_optimizer_updates"])
    result["selected_checkpoint_completed_one_pool_pass_in_this_run"] = (
        result["selected_global_exposure"] >= result["unique_count"])
    return result


def _pair_rows(path, *, kind, cache):
    key = (str(path), kind)
    if key in cache:
        return cache[key]
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            source = json.loads(line)
            identifier = source.get("pair_id")
            if not isinstance(identifier, str) or not identifier or identifier in result:
                raise ValueError("empty or duplicate pair_id: " + str(path))
            if type(source.get("label")) is not bool:
                raise ValueError("pair label is not an explicit bool")
            row = dict(label=source["label"], strict_member=source.get("strict_member"))
            if kind == "score":
                if not isinstance(source.get("scores"), dict):
                    raise ValueError("pair scores are missing")
                row["scores"] = source["scores"]
            else:
                layout = source.get("layouts", {}).get("top2_mode")
                if not isinstance(layout, dict) or type(layout.get("valid")) is not bool:
                    raise ValueError("fixed top2_mode pose is missing or malformed")
                row["pose_valid"] = layout["valid"]
                row["pose_error_px"] = _number(layout.get("translation_l2_px"))
                if row["pose_error_px"] is not None and row["pose_error_px"] < 0:
                    raise ValueError("negative pose error is not a valid saved distance")
            result[identifier] = row
    cache[key] = result
    return result


def _derive_fixed_top2(root, node, assignment_node, cache):
    """Exact full-population join; deliberately no best-decoder choice."""
    node["derived_joint_status"], node["derived_joint"] = "pending", []
    node["derived_joint_issues"] = []
    if node["status"] != "complete" or assignment_node["status"] != "complete":
        node["derived_joint_issues"].append("classification and assignment summaries must both be complete")
        return
    split = "real" if node["population"].startswith("real_") else "test"
    assignment_root = root / "assignment" / split
    score_path = Path(node["source"]).parent / "pair_scores.jsonl"
    pose_path = assignment_root / "pair_results.jsonl"
    protocol_path = assignment_root / "protocol.json"
    if any(not path.is_file() for path in (score_path, pose_path, protocol_path)):
        node["derived_joint_issues"].append("completed pair-score/pose files or assignment protocol not present")
        return
    try:
        protocol = read_json(protocol_path)
        freeze = read_json(node["provenance"]["validation_freeze"])
        if protocol.get("status") != "complete" or freeze.get("status") != "complete":
            node["derived_joint_issues"].append("assignment protocol or classifier freeze not complete")
            return
        checkpoint_id = freeze.get("matcher_checkpoint_id")
        if not checkpoint_id or protocol.get("matcher_checkpoint_id") != checkpoint_id or node["provenance"].get("matcher_checkpoint_id") != checkpoint_id:
            raise ValueError("classifier and assignment do not identify the same frozen matcher checkpoint")
        precision = freeze.get("matcher_precision", freeze.get("precision"))
        if precision is not None and protocol.get("precision") != precision:
            raise ValueError("classifier and assignment inference precision differs")
        scores = _pair_rows(score_path, kind="score", cache=cache)
        poses = _pair_rows(pose_path, kind="pose", cache=cache)
        n, positive_count = COUNTS["real_balanced1016" if split == "real" else "test"]
        if len(scores) != n or len(poses) != n or set(scores) != set(poses):
            raise ValueError("full pair_id bijection required: scores=%d poses=%d expected=%d; missing_pose=%d missing_score=%d" %
                (len(scores), len(poses), n, len(set(scores)-set(poses)), len(set(poses)-set(scores))))
        if sum(row["label"] for row in poses.values()) != positive_count:
            raise ValueError("full-population positive denominator differs")
        for identifier, row in scores.items():
            pose = poses[identifier]
            if row["label"] != pose["label"]:
                raise ValueError("label disagreement at pair_id " + identifier)
            if split == "real" and (type(row["strict_member"]) is not bool or type(pose["strict_member"]) is not bool
                                    or row["strict_member"] != pose["strict_member"]):
                raise ValueError("strict membership disagreement at pair_id " + identifier)
        identifiers = [identifier for identifier, row in poses.items()
                       if node["population"] != "real_strict547" or row["strict_member"]]
        positives = sum(poses[key]["label"] for key in identifiers)
        if (len(identifiers), positives) != COUNTS[node["population"]]:
            raise ValueError("selected balanced/strict denominator differs")
        policy_named_scores = node["stage"] in ("score_fusion", "coarse_infonce")
        for classification in node["classification"]:
            name = classification["policy"]
            policy = (dict(threshold=freeze["thresholds"][name]) if node["stage"] == "coarse_infonce"
                      else freeze["policies"][name])
            threshold = _number(policy.get("threshold"))
            gate_threshold = _number(policy.get("gate_threshold"))
            if threshold is None or (policy.get("gate_threshold") is not None and gate_threshold is None):
                raise ValueError("nonfinite frozen policy threshold")
            branch = name if policy_named_scores else policy["branch"]
            accepted = {}
            for identifier in identifiers:
                values = scores[identifier]["scores"]
                value = _number(values.get(branch))
                if value is None:
                    raise ValueError("frozen policy score absent/nonfinite: " + branch)
                keep = value >= threshold
                if gate_threshold is not None:
                    coarse = _number(values.get("existing_coarse"))
                    if coarse is None:
                        raise ValueError("coarse gate score absent/nonfinite")
                    keep = keep and coarse >= gate_threshold
                accepted[identifier] = keep
            accepted_count = sum(accepted.values())
            for tolerance in (2, 5, 8, 10):
                true_positive = sum(accepted[key] and poses[key]["label"] and poses[key]["pose_valid"]
                    and poses[key]["pose_error_px"] is not None and poses[key]["pose_error_px"] <= tolerance
                    for key in identifiers)
                fp, fn = accepted_count - true_positive, positives - true_positive
                node["derived_joint"].append(dict(policy=name, classifier_branch=branch,
                    threshold_regime=classification["threshold_regime"], threshold=threshold, gate_threshold=gate_threshold,
                    decoder="top2_mode", decoder_choice="predeclared_fixed_not_REAL_selected", tolerance_px=tolerance,
                    sample_count=len(identifiers), positive_count=positives, accepted_count=accepted_count,
                    precision=true_positive/max(1, true_positive+fp), recall=true_positive/max(1, true_positive+fn),
                    f1=2*true_positive/max(1, 2*true_positive+fp+fn), tp=true_positive, fp=fp, fn=fn))
        node["derived_joint_status"] = "complete"
        node["derived_joint_provenance"] = dict(score_source=str(score_path), pose_source=str(pose_path),
            assignment_protocol=str(protocol_path), classifier_freeze=node["provenance"]["validation_freeze"],
            matcher_checkpoint_id=checkpoint_id, full_pair_id_alignment="exact bijection; no inner join or dropped pairs",
            full_population_count=n, selected_population_count=len(identifiers), selected_positive_count=positives,
            label_agreement=True, strict_membership_agreement=split == "real", decoder="top2_mode")
    except (OSError, ValueError, KeyError, TypeError) as error:
        node["derived_joint_status"] = "needs_attention"
        node["derived_joint"] = []
        node["derived_joint_issues"].append(str(error))


def summarize(root):
    root = Path(root).resolve()
    results = [_head(root, name, population) for name in HEADS for population in POPULATIONS]
    results += [_score_fusion(root, population) for population in POPULATIONS]
    results += [_coarse_infonce(root, population) for population in POPULATIONS]
    results += [_common(root, "assignment", population) for population in POPULATIONS]
    for arm in ARMS:
        results.append(_data_validation(root, arm))
        results += [_common(root, arm, population) for population in POPULATIONS if population != "val"]
    assignment = {row["population"]: row for row in results if row["stage"] == "assignment"}
    pair_cache = {}
    for node in results:
        if node["stage"] in HEADS + ("score_fusion", "coarse_infonce") and node["population"] != "val":
            _derive_fixed_top2(root, node, assignment[node["population"]], pair_cache)
    states = Counter(row["status"] for row in results)
    joint_states = Counter(row["derived_joint_status"] for row in results if "derived_joint_status" in row)
    training_metadata = [_data_training_metadata(row) for row in results
                         if row["stage"] in ARMS and row["population"] == "val"]
    status = "needs_attention" if states["needs_attention"] or joint_states["needs_attention"] else "complete" if states["complete"] == len(results) and not joint_states["pending"] else "partial"
    return dict(schema_version=SCHEMA, status=status, root=str(root), created_at=datetime.now(timezone.utc).isoformat(),
        population_status_counts=dict(states), derived_joint_status_counts=dict(joint_states), results=results, winner_selection_performed=False,
        data_training_checkpoint_metadata=dict(arms=training_metadata,
            source_policy="each arm's completed train_val_freeze.json only; missing/provisional stays pending",
            unique_count_definition="number of unique pair rows in this arm's training pool, not manuscript/source count",
            completed_counters_definition="completed incremental training run since the shared warm start; not necessarily exposure of selected weights",
            selected_counters_definition="incremental exposure/update/epoch/VAL event of the saved validation-selected winner",
            warm_start_history_included_in_exposures=False,
            checkpoint_storage="trainer saves VAL winner.pt only, not a separate last.pt; no end-of-budget weight comparison is inferred"),
        source_reads="completed summaries/freezes and small pair-score/pose JSONLs; no checkpoints, matrices, raw masks or GPU",
        metric_definitions=dict(ap="writer's auprc is average precision, not trapezoidal PR area",
            layout_recall="correct valid pose / all GT-positive pairs; invalid pose counts as failure; independent of classification acceptance",
            layout_quantiles="median/P90 pixel errors conditional on valid finite GT-positive poses",
            joint="fused-threshold acceptance AND correct pose; FP includes accepted negatives and accepted wrong poses",
            head_joint="derived classifier+fixed Top2 only after same-checkpoint full pair_id bijection; no REAL decoder selection",
            real_populations="balanced1016:508 positives; strict547:508 positives; different prevalence, never pooled"),
        presentation_notes="Chinese Markdown exact-lookup tables requested; no charts or rank-based winner claim; JSON null is unavailable, never zero")


def _fmt(value, percent=True):
    value = _number(value)
    return "—" if value is None else ("%.2f%%" % (100 * value) if percent else "%.3f" % value)


def _table(headers, rows):
    def text(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(text(x) for x in row) + " |" for row in rows])


def _training_metadata_markdown(report):
    metadata = report.get("data_training_checkpoint_metadata")
    if not metadata:
        return []
    def count(value):
        return "—" if value is None else str(value)
    def paired(row, first, second):
        return count(row[first]) + " / " + count(row[second])
    rows = []
    for row in metadata["arms"]:
        rows.append([row["arm"], row["status"], count(row["unique_count"]),
            paired(row, "completed_global_exposures", "completed_optimizer_updates"),
            paired(row, "completed_dataset_epochs", "completed_validation_events"),
            paired(row, "selected_global_exposure", "selected_optimizer_updates"),
            paired(row, "selected_epoch", "selected_validation_event")])
    lines = ["## 数据三臂：训练预算与获选权重", "",
        "训练池对数是 unique_count，不是独立写卷数。exposure 是一对样本的一次训练呈现，update 是优化器更新；epoch 与 VAL 事件分别列出，60k 池可在一轮内部验证。全部计数仅指本轮从共享 warm-start 起的增量训练，不包含其历史训练。", "",
        _table(["实验", "状态", "训练池对数", "本轮完成 exposure / update", "完成 epoch / VAL事件",
                "获选 exposure / update", "获选 epoch / VAL事件"], rows), "",
        "来源为各臂 realism_training/<arm>/train_val_freeze.json，完整路径见 JSON 的 data_training_checkpoint_metadata。缺失或 provisional 的 freeze 保留 pending，不把临时 winner 当作最终选择。", "",
        "训练器只保存 VAL winner.pt，没有另存 last.pt。完成 120000 次 exposure 不代表获选权重经历了 120000 次；这里是 VAL winner 对照，不是固定 120k 末权重对照。", ""]
    for row in metadata["arms"]:
        if row["status"] != "complete":
            continue
        if row["selected_checkpoint_completed_one_pool_pass_in_this_run"] is False:
            lines += ["注意：%s 的训练池有 %d 对，但获选权重仅到本轮 %d exposure；该权重在本轮尚未遍历完整训练池。整个训练运行后来完成 %d exposure，不能追溯计入更早获选权重。" %
                (row["arm"], row["unique_count"], row["selected_global_exposure"], row["completed_global_exposures"]), ""]
        elif row["selected_checkpoint_is_end_of_run_state"] is False:
            lines += ["%s：本轮运行完成 %d exposure，获选权重对应 %d exposure，属于较早的 VAL winner，而非训练末权重。" %
                (row["arm"], row["completed_global_exposures"], row["selected_global_exposure"]), ""]
    return lines


def markdown(report):
    lines = ["# 夜间配对与摆放实验摘要", "", "## 已保存结果概况", "",
        "状态：`%s`；评估子集摘要状态：%s。仅汇总已完成文件，不按 TEST/REAL 选 winner。" %
        (report["status"], "，".join("%s=%d" % item for item in report["population_status_counts"].items())), "",
        "未完成为 pending；`—` 表示未保存/不适用，不是 0。比率使用百分数，误差使用原 800 画布像素。AP 为保存字段 auprc 的 average precision。", "",
        "分类策略 + 固定 Top2 的逐对联合汇总状态：%s。" %
        "，".join("%s=%d" % item for item in report["derived_joint_status_counts"].items()), "",
        "REAL balanced 和 strict 的正例比例不同，必须分别解释；原冻结阈值、VAL 重校准和固定 0.5 也不混为同一方案。", ""]
    lines += ["TEST/REAL 的矩阵头与标量融合只读取已完成的 native-probability 更正目录；未更正时保留 pending，不以旧重建 AP 代替。VAL 保留原先冻结依据并注明精度口径。", ""]
    lines += ["可选 coarse_infonce 对照：冻结原 BCE coarse CNN，只训练新的 InfoNCE projection；frozen_feature_cosine 是未投影特征的余弦基线。两者分数均为 [-1,1] cosine，不是概率；这不是完整 PairingNet，也未评估全图库检索。epoch 按 VAL F1/AP 选择，TEST/REAL 使用已冻结阈值。缺少或仍在训练的结果保留 pending。", ""]
    lines += _training_metadata_markdown(report)
    for population in POPULATIONS:
        nodes = [row for row in report["results"] if row["population"] == population]
        lines += ["## " + POP_LABELS[population], "", "### 两两分类（不含摆放成功条件）", ""]
        for regime in ("original_frozen", "val_recalibrated", "own_model_val_frozen", "fixed_0_5"):
            rows = []
            for node in nodes:
                if node["status"] != "complete":
                    continue
                for metric in node["classification"]:
                    if metric["threshold_regime"] != regime or metric.get("duplicate_of_own_val_fused"):
                        continue
                    rows.append([node["stage"], metric["branch"], metric["policy"], _fmt(metric["threshold"], False),
                        *[_fmt(metric[key]) for key in ("accuracy", "precision", "recall", "f1", "ap")]])
            if rows:
                lines += ["#### " + REGIMES[regime], "",
                          _table(["实验", "分支", "策略", "阈值", "Acc", "P", "R", "F1", "AP"], rows), ""]
        pending = ["%s：%s（%s）" % (node["stage"], node["status"], "; ".join(node["issues"]))
                   for node in nodes if node["status"] != "complete"]
        if pending:
            lines += ["未齐结果：" + "；".join(pending) + "。", ""]
        layout_rows, joint_rows = [], []
        for node in nodes:
            if node["status"] != "complete":
                continue
            for row in node["layout"]:
                layout_rows.append([node["stage"], row["decoder"], "已冻结" if row["predeclared_selected_decoder"] else "对照",
                    *[_fmt(row[key]) for key in ("coverage", "r2", "r5", "r8", "r10")],
                    _fmt(row["median_px_conditional"], False), _fmt(row["p90_px_conditional"], False)])
            for row in node["joint"]:
                if row["tolerance_px"] == 10:
                    joint_rows.append([node["stage"], row["decoder"], REGIMES[row["threshold_regime"]],
                        *[_fmt(row[key]) for key in ("precision", "recall", "f1")], row["tp"], row["fp"], row["fn"]])
        lines += ["### 仅摆放：无条件正例 R 与条件误差", "",
            "R@2/5/8/10 以全部 GT 正例为分母，无效位姿计失败；median/P90 仅统计有效且有限误差的正例，不代表全体表现。", ""]
        lines += [_table(["实验", "解算器", "既有选择", "覆盖", "R2", "R5", "R8", "R10", "median px", "P90 px"], layout_rows) if layout_rows else "暂无已完成的该人群摆放摘要；分类头与数据臂训练 VAL 本身不评估摆放。", ""]
        lines += ["### 联合成功：分类通过且位姿误差 ≤10 px", "",
            "联合 FP 包含所有被接受的负例及错误摆放；本表沿用各原生摘要的 fused 阈值，下一表另列分类策略 + 固定 Top2 的精确逐对结果。JSON 保留 2/5/8/10 px 全部容差。", "",
            _table(["实验", "解算器", "分类阈值来源", "联合P", "联合R", "联合F1", "TP", "FP", "FN"], joint_rows) if joint_rows else "暂无已完成的联合评估。", ""]
        derived_rows = []
        for node in nodes:
            if node.get("derived_joint_status") != "complete":
                continue
            for row in node["derived_joint"]:
                if row["tolerance_px"] == 10:
                    derived_rows.append([node["stage"], row["policy"], REGIMES[row["threshold_regime"]],
                        *[_fmt(row[key]) for key in ("precision", "recall", "f1")], row["tp"], row["fp"], row["fn"]])
        if population != "val":
            lines += ["### 分类策略 + 预先固定 Top2：逐对联合 ≤10 px", "",
                "只对同 checkpoint、完整 pair_id 双射且 label/strict 成员一致的文件计算。未用 REAL 挑选解算器；无效摆放仍留在正例分母，已接受但摆错的正例同时计 FP 和 FN。", "",
                _table(["实验", "分类策略", "阈值来源", "联合P", "联合R", "联合F1", "TP", "FP", "FN"], derived_rows) if derived_rows else "逐对联合结果尚未就绪。", ""]
            incomplete = ["%s：%s（%s）" % (node["stage"], node["derived_joint_status"], "; ".join(node["derived_joint_issues"]))
                          for node in nodes if node.get("derived_joint_status") in ("pending", "needs_attention")]
            if incomplete:
                lines += ["逐对联合未齐：" + "；".join(incomplete) + "。", ""]
    lines += ["## 解释边界与下一步", "",
        "本表为保存结果的描述性汇总，不计算显著性、不推断因果，也不选择 REAL 最佳模型。矩阵分类头不更新位姿；assignment 的分类分数在各解算器间共享。", "",
        "score_fusion 仅比较原 coarse/local 标量分数的均值或 TRAIN 拟合 affine 融合，阈值由 VAL 冻结；它不是 PairingNet InfoNCE 训练，也不改变匹配器。native 更正仅恢复原网络分支的缓存 FP32 概率，保留新均值/affine 策略定义、权重和阈值。", "",
        "尚未更正的 existing_* baseline 由 FP32 logits 经 float64 sigmoid 重建，matrix_* 则用 FP32 sigmoid；前者可能拆开原 FP32 饱和分数的 ties，因此不能把与历史 native 概率 AP 的差异当作模型提升。更正状态逐来源保存在 JSON。", "",
        "数据三臂的新模型使用各自 VAL 冻结阈值。其 fused 的 at_original_frozen_threshold 在评估代码里只是该新模型 VAL 阈值的别名，不是旧 candidate 的阈值对照。", "",
        "待 pending 的既定训练/评估完成后，以同一脚本输出新目录复跑；来源路径、checkpoint/precision、条件分母和全部指标见配套 JSON。", ""]
    return "\n".join(lines)


def write_report(root, output):
    report = summarize(root)
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "summary.json").write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    (destination / "summary.md").write_text(markdown(report), encoding="utf-8")
    return dict(status=report["status"], output=str(destination), population_status_counts=report["population_status_counts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="brand-new derived report directory")
    arguments = parser.parse_args()
    print(json.dumps(write_report(arguments.root, arguments.output), ensure_ascii=False))
