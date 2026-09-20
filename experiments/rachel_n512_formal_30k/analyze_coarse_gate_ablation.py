"""CPU-only frozen-policy ablation: Full fusion versus coarse gate -> local.

This reuses one unchanged Full checkpoint's cached coarse/local/fused scores.
It does NOT retrain a cascade, run a model, or fit anything on test/real data.
The gate recall targets are predeclared sensitivities, not test-selected policies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analyze_layout_v2_results import classification


GATE_RECALL_TARGETS = (0.99, 0.995, 1.0)
REJECTED_SCORE = -1.0  # Below all local probabilities; rejected pairs tie last.


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_scores(directory, *, expected_count, expected_positive):
    directory = Path(directory)
    with (directory / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if summary.get("status") != "complete":
        raise ValueError("Incomplete cached prediction population: " + str(directory))
    path = directory / "pair_results.jsonl"
    digest = hashlib.sha256()
    rows = []
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if line.strip():
                raw = json.loads(line)
                if type(raw["label"]) is not bool:
                    raise ValueError("Expected explicit boolean label")
                rows.append({key: raw[key] for key in
                             ("pair_id", "label", "classification", "strict_member")
                             if key in raw})
    labels = np.asarray([row["label"] for row in rows], dtype=bool)
    if (len(rows), int(labels.sum())) != (expected_count, expected_positive):
        raise ValueError("Unexpected cached population counts: " + str(directory))
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate pair IDs: " + str(directory))
    scores = {branch: np.asarray([row["classification"][branch] for row in rows], float)
              for branch in ("coarse", "local", "fused")}
    if any(not np.isfinite(value).all() or np.any((value < 0) | (value > 1))
           for value in scores.values()):
        raise ValueError("Expected complete finite probability scores")
    return dict(rows=rows, labels=labels, scores=scores,
                source=str(path.resolve()), sha256=digest.hexdigest())


def fit_f1_threshold(labels, scores, gate=None):
    """Maximize whole-validation F1; ties prefer the larger threshold.

    For a cascade, the denominator still counts ALL validation positives,
    including positives rejected by the gate. Rejected pairs cannot be accepted.
    """
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    gate = np.ones(len(labels), bool) if gate is None else np.asarray(gate, bool)
    if not gate.any():
        return 1.0
    order = np.argsort(-scores[gate], kind="stable")
    ranked = scores[gate][order]
    ranked_labels = labels[gate][order]
    last = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(ranked) - 1]
    tp = np.cumsum(ranked_labels)[last]
    f1 = 2 * tp / np.maximum(last + 1 + labels.sum(), 1)
    return float(ranked[last[np.argmax(f1)]])


def fit_high_recall_gate(labels, coarse_scores, target):
    """Highest observed positive score threshold satisfying recall >= target."""
    if not 0 < target <= 1:
        raise ValueError("Gate target must be in (0, 1]")
    positive_scores = np.sort(np.asarray(coarse_scores)[np.asarray(labels, bool)])[::-1]
    if len(positive_scores) == 0:
        raise ValueError("Gate calibration needs positive validation pairs")
    required = int(math.ceil(target * len(positive_scores)))
    return float(positive_scores[required - 1])


def freeze_policies(validation, prior_freeze):
    labels, scores = validation["labels"], validation["scores"]
    thresholds = {branch: fit_f1_threshold(labels, value)
                  for branch, value in scores.items()}
    policies = {}
    for name, branch, threshold in (
        ("fused_original_frozen", "fused", prior_freeze["original_fused_threshold"]),
        ("fused_val_f1", "fused", thresholds["fused"]),
        ("local_val_f1", "local", thresholds["local"]),
        ("coarse_val_f1", "coarse", thresholds["coarse"]),
    ):
        policies[name] = dict(branch=branch, decision_threshold=float(threshold),
                              coarse_gate_threshold=None, local_threshold_rule=None)
    for target in GATE_RECALL_TARGETS:
        gate_threshold = fit_high_recall_gate(labels, scores["coarse"], target)
        gate = scores["coarse"] >= gate_threshold
        prefix = "coarse_gate_r{:g}".format(target * 100).replace(".", "p")
        for suffix, local_threshold in (
            ("fixed_local", thresholds["local"]),
            ("refit_local", fit_f1_threshold(labels, scores["local"], gate)),
        ):
            policies[prefix + "_" + suffix] = dict(
                branch="local", decision_threshold=local_threshold,
                coarse_gate_threshold=gate_threshold, gate_target_validation_recall=target,
                achieved_validation_gate_recall=float(gate[labels].mean()),
                local_threshold_rule=("reuse ungated local validation F1 threshold" if
                    suffix == "fixed_local" else "maximize whole-validation cascade F1"))
    return dict(
        schema_version="rachel-coarse-gate-frozen-policy/1.0",
        experiment_type="fixed_model_policy_ablation_not_cascade_retraining",
        checkpoint_sha256=prior_freeze["checkpoint_sha256"],
        fit_population="validation_3000", fit_source=validation["source"],
        fit_source_sha256=validation["sha256"],
        sample_count=len(labels), positive_count=int(labels.sum()),
        gate_recall_targets=list(GATE_RECALL_TARGETS),
        target_selection="predeclared 99%, 99.5%, 100% sensitivity grid; no winner selected",
        threshold_tie_break="largest threshold among equal validation F1",
        gate_comparison=">=", final_decision="coarse >= gate AND local >= local_threshold",
        test_or_real_used_for_fit=False, policies=policies,
        ranking_definition="gate pass: original local probability; gate reject: -1 (all tied last)",
        gated_ranking_is_calibrated_probability=False,
        note="Coarse and local scores are subbranches of the same frozen Full model, not separately retrained models.",
    )


def evaluate_population(population, policies, include=None):
    include = np.ones(len(population["labels"]), bool) if include is None else include
    labels = population["labels"][include]
    scores = {key: value[include] for key, value in population["scores"].items()}
    local_threshold = policies["local_val_f1"]["decision_threshold"]
    ungated_local_accepted = scores["local"] >= local_threshold
    result = dict(sample_count=len(labels), positive_count=int(labels.sum()), policies={})
    for name, policy in policies.items():
        gate_threshold = policy["coarse_gate_threshold"]
        gate = (np.ones(len(labels), bool) if gate_threshold is None else
                scores["coarse"] >= gate_threshold)
        rank_scores = np.where(gate, scores[policy["branch"]], REJECTED_SCORE)
        metrics = classification(labels, rank_scores, policy["decision_threshold"])
        expected = gate & (scores[policy["branch"]] >= policy["decision_threshold"])
        if int((expected & labels).sum()) != metrics["tp"]:
            raise AssertionError("Gated ranking and conjunction decisions differ")
        result["policies"][name] = dict(
            **metrics, gate_threshold=gate_threshold,
            gate_pass_count=int(gate.sum()), gate_pass_fraction=float(gate.mean()),
            gate_positive_recall=float(gate[labels].mean()),
            gate_positive_rejected=int((~gate & labels).sum()),
            gate_negative_rejected=int((~gate & ~labels).sum()),
            local_true_positives_lost_to_gate=int((~gate & labels & ungated_local_accepted).sum()),
            local_false_positives_removed_by_gate=int((~gate & ~labels & ungated_local_accepted).sum()),
        )
    return result


def render_markdown(summary):
    freeze = summary["freeze"]
    lines = ["# Coarse gate → local: frozen-model policy ablation", "",
        "This is a cached-score decision-policy comparison, **not cascade retraining**. "
        "All branches come from the same unchanged Full checkpoint. No model forward, GPU work, "
        "new weights, or layout change is involved.", "",
        "Only validation 3,000 pairs select thresholds. The 99%, 99.5%, and 100% gate-recall "
        "targets are a fixed sensitivity grid; test/real results do not select a policy. "
        "The freeze is written before either evaluation population is opened.", "",
        "Two local operating points are retained: `fixed_local` isolates rejection by using "
        "the ungated local threshold; `refit_local` maximizes final cascade F1 on all validation "
        "pairs, including gated-out positives as false negatives. Original fused and "
        "validation-recalibrated fused are both reported.", "",
        "## Validation-frozen thresholds", "",
        "| Policy | Coarse gate | Decision threshold | Target gate recall |",
        "| --- | ---: | ---: | ---: |"]
    for name, policy in freeze["policies"].items():
        gate = policy["coarse_gate_threshold"]
        target = policy.get("gate_target_validation_recall")
        lines.append("| {} | {} | {:.9g} | {} |".format(
            name, "—" if gate is None else "{:.9g}".format(gate),
            policy["decision_threshold"], "—" if target is None else "{:.2%}".format(target)))
    for population_name, population in summary["populations"].items():
        lines.extend(["", "## {} (n={}, positives={})".format(
            population_name, population["sample_count"], population["positive_count"]), "",
            "| Policy | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | Gate positive recall | TP lost to gate* | FP removed by gate* |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for name, value in population["policies"].items():
            metric_values = [value[key] for key in
                             ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "gate_positive_recall")]
            lines.append("| {} | {} | {} | {} |".format(name,
                " | ".join("{:.4f}".format(value) for value in metric_values),
                value["local_true_positives_lost_to_gate"],
                value["local_false_positives_removed_by_gate"]))
    lines.extend(["", "## Interpretation limits", "",
        "- *TP lost / FP removed compare gate rejection with the ungated `local_val_f1` decisions; "
        "they do not include changes from a refitted local threshold.",
        "- Cascade AUROC/AP rank passing pairs by unchanged local probability and tie all "
        "rejected pairs below them (-1). This is a hard-rejection ranking, not a calibrated probability.",
        "- A 99–100% validation gate recall is not guaranteed on shifted real data. Gate-positive "
        "recall is an upper bound on any subsequent local classifier's recall.",
        "- Row-level point estimates only: this small policy ablation does not estimate uncertainty "
        "or claim independent-pair statistical significance. The strict real subset has a different class balance.",
        "- This cannot establish the performance of a newly trained coarse→fine architecture, "
        "a new coarse encoder, 7px windows, 1024-point sampling, or a full fragment candidate graph.",
        "- Gate coverage is reported to expose rejected positives, not to optimize throughput.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path,
                        default=Path("reports/pairwise_layout_v2_20260906"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/pairwise_ablation_v3_20260907/coarse_gate"))
    args = parser.parse_args()
    validation = load_scores(args.prediction_root / "validation_3000",
                             expected_count=3000, expected_positive=1500)
    with (args.prediction_root / "validation_3000/validation_freeze.json").open() as stream:
        prior_freeze = json.load(stream)
    if prior_freeze.get("source_split") != "validation" or prior_freeze.get("probe_only") is not False:
        raise ValueError("Expected formal validation freeze")
    freeze = freeze_policies(validation, prior_freeze)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = args.output_dir / "policy_freeze.json"
    write_json(freeze_path, freeze)
    # Deliberately do not read test/real scores until the policies are serialized.
    populations = {"validation_3000": evaluate_population(validation, freeze["policies"])}
    sources = {"validation_3000": dict(path=validation["source"], sha256=validation["sha256"])}
    for name, total, positives in (("test_3000", 3000, 1500), ("real_1016", 1016, 508)):
        population = load_scores(args.prediction_root / name,
                                 expected_count=total, expected_positive=positives)
        populations[name] = evaluate_population(population, freeze["policies"])
        sources[name] = dict(path=population["source"], sha256=population["sha256"])
        if name == "real_1016":
            strict = np.asarray([row.get("strict_member", False) for row in population["rows"]], bool)
            if int(strict.sum()) != 547 or int(population["labels"][strict].sum()) != 508:
                raise ValueError("Expected explicit strict real subset of 547 / 508 positives")
            populations["real_strict_547"] = evaluate_population(population, freeze["policies"], strict)
    summary = dict(status="complete", freeze=freeze, sources=sources, populations=populations,
                   test_or_real_used_for_fit=False, model_retrained=False,
                   policy_freeze_written_before_test_real_read=True)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(dict(status="complete", output_dir=str(args.output_dir.resolve()),
                          populations={key: value["sample_count"] for key, value in populations.items()})))


if __name__ == "__main__":
    main()
