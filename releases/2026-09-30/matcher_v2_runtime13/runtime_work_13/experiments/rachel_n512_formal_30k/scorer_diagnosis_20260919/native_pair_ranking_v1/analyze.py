"""Compare frozen native cosine and pair-head rankings on identical REAL pairs.

Read-only inputs. No threshold fitting, model selection, training or inference.
This is a system comparison, not a controlled dot-product/attention ablation.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RETRIEVAL = REPO / "reports/rachel_recall_benchmarks_20260911_001/completed/retrieval"
ENDPOINT = HERE.parent / "continuation_endpoint_v1/raw"
SOURCES = {}


def record(path):
    SOURCES[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path, lines=False):
    raw = record(path).read_text()
    return [json.loads(line) for line in raw.splitlines() if line] if lines else json.loads(raw)


def rank_metrics(labels, scores):
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    assert len(labels) == len(scores) and np.isfinite(scores).all()
    positive, negative = int(labels.sum()), int((~labels).sum())
    assert positive and negative
    groups = defaultdict(lambda: [0, 0])
    for label, score in zip(labels, scores):
        groups[float(score)][0 if label else 1] += 1
    tp = fp = 0
    previous_tpr = previous_fpr = auc = ap = 0.
    for score in sorted(groups, reverse=True):
        p, n = groups[score]
        tp += p
        fp += n
        tpr, fpr = tp / positive, fp / negative
        auc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2
        ap += (tpr - previous_tpr) * tp / (tp + fp)
        previous_tpr, previous_fpr = tpr, fpr
    # Independent ROC identity, including half-credit ties.
    differences = scores[labels, None] - scores[~labels][None, :]
    direct_auc = float(((differences > 0).sum() + .5 * (differences == 0).sum()) / differences.size)
    assert abs(auc - direct_auc) < 1e-12
    return dict(count=len(labels), positive=positive, negative=negative,
                auroc=auc, average_precision=ap,
                prevalence=positive / len(labels),
                positive_score_quantiles=np.quantile(scores[labels], [0, .25, .5, .75, 1]).tolist(),
                negative_score_quantiles=np.quantile(scores[~labels], [0, .25, .5, .75, 1]).tolist())


def main():
    manifest = read(RETRIEVAL / "prepared_manifest.json")
    ids, pairs = manifest["fragment_ids"], manifest["pairs"]
    assert len(ids) == len(set(ids)) == 938 and len(pairs) == 1016
    positions = {key: index for index, key in enumerate(ids)}
    ia = np.array([positions[p["fragment_a_id"]] for p in pairs])
    ib = np.array([positions[p["fragment_b_id"]] for p in pairs])
    assert np.all(ia != ib)
    labels = np.array([p["label"] for p in pairs], dtype=bool)
    strict = np.array([p["strict"] for p in pairs], dtype=bool)
    assert labels.sum() == 508 and strict.sum() == 547
    models, checks = {}, {}
    for family in ("pairing_stage2", "shredding_coarse"):
        root = RETRIEVAL / family
        receipt = read(root / "receipt.json")
        assert receipt["status"] == "complete" and not receipt["trained_or_calibrated_on_real"]
        matrix_path = record(root / "scores.npy")
        assert SOURCES[str(matrix_path.relative_to(REPO))] == receipt["score_matrix_sha256"]
        assert SOURCES[str((RETRIEVAL / "prepared_manifest.json").relative_to(REPO))] == receipt["identity"]["input_evidence"]["prepared_manifest_sha256"]
        matrix = np.load(matrix_path, allow_pickle=False)
        assert matrix.shape == (938, 938)
        cosine = matrix[ia, ib].astype(float)
        assert np.isfinite(cosine).all() and np.max(np.abs(cosine)) <= 1.00001
        assert np.max(np.abs(cosine - matrix[ib, ia])) < 1e-6
        binary = np.load(record(root / "selected_pair_binary_scores.npy"), allow_pickle=False)
        assert binary.shape == (1016,) and np.isfinite(binary).all()
        models[family + "_cosine"] = cosine
        models[family + "_separate_pair_head"] = binary
        checks[family] = dict(
            native_score_family=receipt["identity"]["model"]["score_family"],
            separately_stored_binary_scores_are_not_cosine=True,
            manifest_and_matrix_match_completed_receipt=True,
            cosine_symmetric_on_all_pairs=True)
    rows_by_arm = {}
    for arm, relative in (
        ("s6_d2_C8", "attention_depth_20260915/s4_cross_attention_depth2/evaluation/fixed_epoch/real"),
        ("s6_d2_C16", "scorer_diagnosis_20260919/continuation_v1/s6_d2_c16/evaluation/fixed_epoch/real"),
    ):
        rows = read(ENDPOINT / relative / "pair_results.jsonl", lines=True)
        assert len(rows) == len(pairs)
        for pair, row in zip(pairs, rows):
            assert (pair["pair_id"], pair["label"], pair["fragment_a_id"], pair["fragment_b_id"], pair["strict"]) == (
                row["pair_id"], row["label"], row["fragment_a"], row["fragment_b"], row["strict_member"])
        assert all(r["decision_valid"] for r in rows)
        models[arm] = np.array([r["classification"]["fused"] for r in rows])
        rows_by_arm[arm] = rows
    old, new = rows_by_arm.values()
    assert all(a["review_status"] == b["review_status"] for a, b in zip(old, new))
    keep = np.array([not r["label"] or r["review_status"] == "keep" for r in old])
    assert int(keep.sum()) == 803 and int(labels[keep].sum()) == 295
    populations = dict(full1016=np.ones(1016, dtype=bool), retained803=keep,
                       strict547=strict, retained_and_strict=keep & strict)
    result = dict(
        schema="rachel-frozen-native-pair-ranking/1", status="complete",
        executed_at=datetime.now(timezone.utc).isoformat(),
        fitting_or_inference=False, threshold_fitting=False,
        population="fixed annotated REAL pair list; never labels unknown gallery edges negative",
        negative_policy="39 strict negatives plus469 constructed distractors; report strict sensitivity separately",
        human_filtering="295 retained positives; exclusions based on previous review, exploratory not untouched confirmatory test",
        causal_architecture_ablation=False,
        limitations=["Different encoders, targets, training budgets and checkpoint selection; cannot isolate dot product versus attention.",
                     "Cosine is a retrieval score, not a calibrated probability; no accuracy/F1 at an invented threshold.",
                     "Mask-only N512 adaptations, not original full-RGB paper systems.",
                     "No Turufan native cosine outputs collected by this script."],
        metrics={name: {model: rank_metrics(labels[mask], score[mask]) for model, score in models.items()}
                 for name, mask in populations.items()}, checks=checks,
        sources=SOURCES)
    with (HERE / "results.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    for group in result["metrics"]:
        print(group, {model: {key: values[key] for key in ("count", "positive", "negative", "auroc", "average_precision")}
                      for model, values in result["metrics"][group].items()})


if __name__ == "__main__":
    main()
