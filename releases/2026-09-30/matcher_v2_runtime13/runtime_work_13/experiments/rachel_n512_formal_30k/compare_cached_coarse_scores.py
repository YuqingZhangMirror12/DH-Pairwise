"""Compare existing whole-pair coarse rankings without running or fitting models.

ShreddingNet-adapted validation stores raw coarse cosine in [-1, 1]; Full
stores a coarse probability. AUROC and grouped average precision do not require
the same calibration. Final pair probabilities are deliberately never read.
This compares trained checkpoints, not matched-budget encoder architectures.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.rachel_n512_formal_30k.calibrate_completed_benchmark_row_f1 import classification


ROOT = Path(__file__).resolve().parents[2]


def load_coarse(path, model, expected_count, expected_positive, split="validation"):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        if split == "real":
            inputs = json.load(stream)["balanced_1016"]["pairs"]
        else:
            inputs = (json.loads(line) for line in stream if line.strip())
        for raw in inputs:
            if split == "real":
                method = "shreddingnet_adapted" if model == "shreddingnet_adapted" else "full_n512"
                score = raw["methods"][method]["coarse_probability"]
            else:
                score = raw["coarse_score"] if model == "shreddingnet_adapted" else raw["classification"]["coarse"]
            lower = -1 if model == "shreddingnet_adapted" else 0
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not lower <= score <= 1):
                raise ValueError("Invalid coarse score in " + str(path))
            if type(raw["label"]) is not bool:
                raise ValueError("Pair labels must be explicit booleans")
            if not isinstance(raw["pair_id"], str) or not raw["pair_id"]:
                raise ValueError("Pair IDs must be nonempty strings")
            rows.append(dict(pair_id=raw["pair_id"], label=raw["label"], score=score))
    if len(rows) != expected_count or len({r["pair_id"] for r in rows}) != expected_count:
        raise ValueError("Wrong population size or duplicate pair IDs")
    if sum(r["label"] for r in rows) != expected_positive:
        raise ValueError("Wrong positive count")
    return rows


def compare(full_path, shred_path, expected_count=3000, expected_positive=1500, split="validation"):
    if split not in {"validation", "real"}:
        raise ValueError("Only cached validation and real coarse populations are available")
    sources = {"full_original_coarse": Path(full_path), "shreddingnet_adapted": Path(shred_path)}
    rows = {model: load_coarse(path, model, expected_count, expected_positive, split)
            for model, path in sources.items()}
    identities = [{r["pair_id"]: r["label"] for r in items} for items in rows.values()]
    if identities[0] != identities[1]:
        raise ValueError("Coarse models must cover the same identically labelled pair IDs")
    methods = {}
    for model, items in rows.items():
        # Threshold zero is immaterial to these two ranking metrics; discard
        # every threshold-dependent output instead of presenting an unfitted F1.
        metrics = classification(items, 0)
        methods[model] = {key: metrics[key] for key in ("auroc", "auprc")}
        methods[model].update(score_min=min(r["score"] for r in items),
                              score_max=max(r["score"] for r in items))
    fields = ({"full_original_coarse": "balanced_1016.pairs[].methods.full_n512.coarse_probability",
               "shreddingnet_adapted": "balanced_1016.pairs[].methods.shreddingnet_adapted.coarse_probability"}
              if split == "real" else
              {"full_original_coarse": "classification.coarse", "shreddingnet_adapted": "coarse_score"})
    return dict(status="complete", split=split, sample_count=expected_count,
                positive_count=expected_positive, same_pair_ids_and_labels=True,
                methods=methods, sources={k: str(v.resolve()) for k, v in sources.items()},
                score_fields=fields,
                score_semantics={"full_original_coarse": "coarse branch probability",
                                 "shreddingnet_adapted": "raw coarse embedding cosine"},
                score_direction="higher is more likely to match; no clipping or recalibration",
                auprc_definition="grouped average precision, not trapezoidal PR area",
                model_executed=False, thresholds_fitted=False, final_pair_score_used=False,
                training_budget_matched=False,
                interpretation_limit="mask-only upright adaptations; selected-pair population, not native full-graph top-k retrieval or isolated encoder causality; previously viewed real set")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "real"), default="validation")
    parser.add_argument("--full", type=Path)
    parser.add_argument("--shred", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    real_path = ROOT / "reports/pairwise_exact6_20260906/real_pair_only.json"
    full_path = args.full or (real_path if args.split == "real" else
                            ROOT / "reports/pairwise_layout_v2_20260906/validation_3000/pair_results.jsonl")
    shred_path = args.shred or (real_path if args.split == "real" else
                              ROOT / "reports/pairwise_ablation_v3_20260907/benchmark_row_f1/inputs/validation_threshold_scores.jsonl")
    count, positives = (1016, 508) if args.split == "real" else (3000, 1500)
    result = compare(full_path, shred_path, count, positives, args.split)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
