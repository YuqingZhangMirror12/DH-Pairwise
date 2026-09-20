"""Train/evaluate matrix-pattern pair heads on frozen matcher caches, no GPU default.

Train reads ONLY TRAIN and VAL, selects each head by VAL F1 (AP tie-break),
and writes head.pt plus validation_freeze.json. TEST/REAL are a separate CLI:

  python train_matrix_pair_head.py --train-manifest cache/train/manifest.json \
      --val-manifest cache/val/manifest.json --output-root heads --device cuda
  python train_matrix_pair_head.py --head heads/head.pt \
      --evaluate-manifest cache/test/manifest.json --output-root heads_test

Cache schema rachel-matrix-pair-cache/v1:
  {status:complete, split:train|val|test|real, matcher_checkpoint_id, precision,
   sample_count, chunks:[{path:chunk_000.npz, sample_count:64}],
   original_thresholds:{coarse:...,local:...,fused:...}}  # thresholds optional
NPZ (no pickle/object arrays): real_transport[B,N,M] float32; affinity[B,N,M]
float16/32; valid_a[B,N],valid_b[B,M] bool; coarse_logit/local_logit/fused_logit
[B] float32; label[B] bool; pair_id[B] Unicode. points_a_rc/points_b_rc may be
present and are accepted by the model but unused. REAL requires strict_member
[B] bool. Optional coarse_cosine[B] is a pooled-descriptor cosine diagnostic,
NOT native InfoNCE-trained retrieval. Other arrays, including target pose, are
never read. Chunks are shuffled only in TRAIN; NPZ arrays are loaded one chunk
at a time, not once per row or into a full-population matrix in RAM.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from analyze_coarse_gate_ablation import fit_f1_threshold, fit_high_recall_gate
from analyze_layout_v2_results import classification
from staging.pairwise_v0_2.models.matrix_pair_head import MatrixPairHead, MatrixPairHeadConfig


SCHEMA = "rachel-matrix-pair-cache/v1"
HEAD_SCHEMA = "rachel-matrix-pair-head/v1"
BRANCHES = ("matrix_only", "matrix_coarse")
MATRIX_KEYS = ("real_transport", "affinity", "valid_a", "valid_b")
LOGIT_KEYS = ("coarse_logit", "local_logit", "fused_logit")
REQUIRED_KEYS = MATRIX_KEYS + LOGIT_KEYS + ("label", "pair_id")
OPTIONAL_KEYS = ("strict_member", "case_id", "fragment_a", "fragment_b", "coarse_cosine")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


class MatrixCache:
    """Small manifest and bounded chunk reader; does not open a matcher or GT pose."""

    def __init__(self, path, *, expected_split=None, checkpoint_id=None, expected_count=None):
        self.path = Path(path).resolve()
        self.manifest = read_json(self.path)
        m = self.manifest
        if m.get("schema_version") != SCHEMA or m.get("status") != "complete":
            raise ValueError("cache must have complete v1 manifest: " + str(self.path))
        self.split = m.get("split")
        if self.split not in ("train", "val", "test", "real"):
            raise ValueError("cache split must be train, val, test, or real")
        if expected_split is not None and self.split != expected_split:
            raise ValueError("split mismatch: expected " + expected_split)
        self.checkpoint_id = m.get("matcher_checkpoint_id")
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("cache must identify frozen matcher checkpoint")
        if checkpoint_id is not None and self.checkpoint_id != checkpoint_id:
            raise ValueError("cache uses a different frozen matcher checkpoint")
        self.precision = m.get("precision")
        if not isinstance(self.precision, str) or not self.precision:
            raise ValueError("cache must record matcher inference precision")
        self.sample_count = int(m["sample_count"])
        self.chunks = m["chunks"]
        if not self.chunks or self.sample_count <= 0 or any(int(c["sample_count"]) <= 0 for c in self.chunks):
            raise ValueError("cache must contain nonempty chunks")
        if sum(int(c["sample_count"]) for c in self.chunks) != self.sample_count:
            raise ValueError("chunk counts do not sum to manifest sample_count")
        if expected_count is not None and self.sample_count != expected_count:
            raise ValueError("cache has unexpected sample_count")

    def batches(self, batch_size, *, shuffle=False, seed=0):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if shuffle and self.split != "train":
            raise ValueError("only TRAIN cache can be shuffled for optimization")
        rng = np.random.default_rng(seed)
        chunk_indices = rng.permutation(len(self.chunks)) if shuffle else range(len(self.chunks))
        for index in chunk_indices:
            chunk = self.chunks[index]
            path = self.path.parent / chunk["path"]
            with np.load(path, allow_pickle=False) as archive:
                missing = set(REQUIRED_KEYS) - set(archive.files)
                if missing:
                    raise ValueError("missing cache fields: " + str(sorted(missing)))
                keys = REQUIRED_KEYS + tuple(k for k in OPTIONAL_KEYS if k in archive.files)
                arrays = {k: archive[k] for k in keys}
            count = int(chunk["sample_count"])
            if any(a.shape[0] != count for a in arrays.values()):
                raise ValueError("cache arrays disagree with chunk row count")
            if arrays["label"].shape != (count,) or arrays["label"].dtype != np.bool_:
                raise ValueError("cache labels must be bool [B]")
            if arrays["pair_id"].shape != (count,) or arrays["pair_id"].dtype.kind not in "US":
                raise ValueError("pair_id must be a string [B] array, no pickle")
            if self.split == "real" and "strict_member" not in arrays:
                raise ValueError("REAL requires explicit strict_member")
            if "strict_member" in arrays and (arrays["strict_member"].shape != (count,) or arrays["strict_member"].dtype != np.bool_):
                raise ValueError("strict_member must be bool [B]")
            for key in LOGIT_KEYS + (("coarse_cosine",) if "coarse_cosine" in arrays else ()):
                if arrays[key].shape != (count,) or not np.isfinite(arrays[key]).all():
                    raise ValueError(key + " must be finite [B]")
            order = rng.permutation(count) if shuffle else np.arange(count)
            for start in range(0, count, batch_size):
                subset = order[start:start + batch_size]
                yield {key: value[subset] for key, value in arrays.items()}

    def provenance(self):
        return dict(manifest=str(self.path), split=self.split, sample_count=self.sample_count,
                    matcher_checkpoint_id=self.checkpoint_id, precision=self.precision)


def model_inputs(batch, device):
    return {name: torch.from_numpy(np.asarray(batch[name])).to(device)
            for name in MATRIX_KEYS + ("coarse_logit", "local_logit")}


def sigmoid(values):
    return torch.sigmoid(torch.as_tensor(np.asarray(values), dtype=torch.float64)).numpy()


def predict(cache, models, batch_size=8, device="cpu"):
    """Retain only scalar predictions/row metadata across chunks, never matrices."""
    result = dict(labels=[], pair_ids=[], strict=[], scores={}, metadata=[], invalid_count=0)
    optional_cosine = None
    for model in models.values():
        model.eval()
    with torch.no_grad():
        for batch in cache.batches(batch_size):
            tensors = model_inputs(batch, device)
            outputs = {}
            batch_scores = {}
            for name, model in models.items():
                if id(model) not in outputs:
                    outputs[id(model)] = model(**tensors)
                output = outputs[id(model)]
                value = output.matrix_logit if name == "matrix_only" else output.matrix_coarse_logit
                batch_scores[name] = torch.sigmoid(value).cpu().double().numpy()
            output = next(iter(outputs.values()))
            result["invalid_count"] += int((~output.valid_problem).sum().item())
            for name in ("coarse", "local", "fused"):
                batch_scores["existing_" + name] = sigmoid(batch[name + "_logit"])
            has_cosine = "coarse_cosine" in batch
            if optional_cosine is not None and optional_cosine != has_cosine:
                raise ValueError("coarse_cosine must be present in every chunk or absent from all")
            optional_cosine = has_cosine
            if has_cosine:
                batch_scores["pooled_coarse_cosine"] = batch["coarse_cosine"].astype(float)
            for key, values in batch_scores.items():
                if not np.isfinite(values).all():
                    raise ValueError("non-finite pair head prediction")
                result["scores"].setdefault(key, []).extend(values.tolist())
            result["labels"].extend(batch["label"].tolist())
            result["pair_ids"].extend(batch["pair_id"].astype(str).tolist())
            result["strict"].extend(batch.get("strict_member", np.ones(len(batch["label"]), bool)).tolist())
            for i in range(len(batch["label"])):
                result["metadata"].append({key: str(batch[key][i]) for key in
                                           ("case_id", "fragment_a", "fragment_b") if key in batch})
    if len(result["labels"]) != cache.sample_count or len(set(result["pair_ids"])) != cache.sample_count:
        raise ValueError("prediction rows must have unique pair IDs and exact manifest count")
    result["labels"] = np.asarray(result["labels"], bool)
    result["strict"] = np.asarray(result["strict"], bool)
    result["scores"] = {key: np.asarray(value) for key, value in result["scores"].items()}
    return result


def selection_key(metrics, primary="f1"):
    if primary == "f1":
        return metrics["f1"], metrics["auprc"]
    if primary == "auprc":
        return metrics["auprc"], metrics["f1"]
    raise ValueError("selection metric must be f1 or auprc")


def freeze_policies(validation, original_thresholds=None, gate_recalls=(0.99, 0.995, 1.0)):
    """Fit from validation scalar predictions ONLY; cascades count all positives."""
    labels, scores = validation["labels"], validation["scores"]
    if not labels.any() or labels.all():
        raise ValueError("validation requires both positive and negative pairs")
    policies = {}
    for branch, values in scores.items():
        policies[branch] = dict(branch=branch, threshold=fit_f1_threshold(labels, values),
                                gate_threshold=None, threshold_fit="validation_equal_row_F1")
    for branch, threshold in (original_thresholds or {}).items():
        name = "existing_" + branch
        if name in scores:
            if not np.isfinite(threshold):
                raise ValueError("original frozen threshold must be finite")
            policies[name + "_original_frozen"] = dict(branch=name, threshold=float(threshold),
                                                        gate_threshold=None, threshold_fit="original_matcher_validation_freeze")
    for recall in gate_recalls:
        gate_threshold = fit_high_recall_gate(labels, scores["existing_coarse"], recall)
        gate = scores["existing_coarse"] >= gate_threshold
        name = "coarse_gate_r" + str(recall).replace(".", "p") + "_then_local"
        policies[name] = dict(branch="existing_local", threshold=fit_f1_threshold(labels, scores["existing_local"], gate),
                             gate_threshold=gate_threshold, target_gate_recall=recall,
                             achieved_validation_gate_recall=float(gate[labels].mean()),
                             threshold_fit="validation_whole_population_cascade_F1")
    if "matrix_only" in scores:
        # Separate from TRAIN-learned affine fusion: freeze the independently
        # fitted matrix-only threshold, then add a VAL 99%-recall coarse gate.
        # No refit of the matrix threshold, no extra network, no pose changes.
        gate_threshold = fit_high_recall_gate(labels, scores["existing_coarse"], 0.99)
        gate = scores["existing_coarse"] >= gate_threshold
        policies["coarse_gate_r0p99_then_matrix_fixed"] = dict(
            branch="matrix_only", threshold=policies["matrix_only"]["threshold"],
            gate_threshold=gate_threshold, target_gate_recall=0.99,
            achieved_validation_gate_recall=float(gate[labels].mean()),
            threshold_fit="reuse ungated matrix-only validation equal-row F1 threshold",
            matrix_threshold_refitted_after_gate=False)
    return policies


def policy_values(predictions, policy):
    scores = predictions["scores"][policy["branch"]]
    if policy["gate_threshold"] is not None:
        gate = predictions["scores"]["existing_coarse"] >= policy["gate_threshold"]
        scores = np.where(gate, scores, -2.0)
    return scores


def population_metrics(predictions, policies, include=None):
    include = np.ones(len(predictions["labels"]), bool) if include is None else np.asarray(include, bool)
    labels = predictions["labels"][include]
    if not len(labels):
        return dict(sample_count=0, positive_count=0, policies={})
    results = {}
    unavailable = {}
    for name, policy in policies.items():
        if policy["branch"] not in predictions["scores"]:
            unavailable[name] = "optional cached score absent; no substitute fitted"
            continue
        scores = policy_values(predictions, policy)[include]
        results[name] = classification(labels, scores, policy["threshold"])
        if policy["gate_threshold"] is not None:
            gate = predictions["scores"]["existing_coarse"][include] >= policy["gate_threshold"]
            ungated_accept = predictions["scores"][policy["branch"]][include] >= policy["threshold"]
            results[name].update(gate_rejected_positives=int((~gate & labels).sum()),
                                 gate_rejected_negatives=int((~gate & ~labels).sum()),
                                 previously_accepted_positives_lost_to_gate=int((~gate & labels & ungated_accept).sum()),
                                 previously_accepted_negatives_removed_by_gate=int((~gate & ~labels & ungated_accept).sum()))
    return dict(sample_count=len(labels), positive_count=int(labels.sum()), negative_count=int((~labels).sum()),
                policies=results, unavailable_policies=unavailable)


def save_population(directory, cache, predictions, policies, receipt):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rows_path = directory / "pair_scores.jsonl"
    with rows_path.open("w", encoding="utf-8") as stream:
        for i, pair_id in enumerate(predictions["pair_ids"]):
            scores = {k: float(v[i]) for k, v in predictions["scores"].items()}
            row = dict(pair_id=pair_id, label=bool(predictions["labels"][i]),
                       strict_member=bool(predictions["strict"][i]), scores=scores,
                       **predictions["metadata"][i])
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    metrics = dict(status="complete", split=cache.split, cache=cache.provenance(),
                   matcher_frozen=True, pose_evaluated=False, sampled_seam_gt_evaluated=False,
                   selected_epochs={key: value["epoch"] for key, value in receipt["selected_heads"].items()},
                   prediction_count=len(predictions["labels"]), invalid_cached_problem_count=predictions["invalid_count"],
                   threshold_source="head.pt validation-frozen policies", test_or_real_used_for_fit=False,
                   classification=population_metrics(predictions, policies),
                   original_threshold_comparison_available=any(n.endswith("_original_frozen") for n in policies),
                   cosine_caveat="pooled coarse descriptor cosine on supplied pair list, not InfoNCE-trained retrieval or gallery Top-K",
                   cascade_caveat="cached fixed-matcher two-threshold policy; not cascade retraining; rejected rows rank last",
                   points_and_target_pose_used=False)
    if cache.split == "real":
        metrics["strict_classification"] = population_metrics(predictions, policies, predictions["strict"])
    write_json(directory / "metrics.json", metrics)
    return metrics


def selected_models(receipt, device):
    config = dict(receipt["model_config"])
    config["widths"] = tuple(config["widths"])
    models, by_epoch = {}, {}
    for branch in BRANCHES:
        winner = receipt["selected_heads"][branch]
        if winner["epoch"] not in by_epoch:
            model = MatrixPairHead(MatrixPairHeadConfig(**config))
            model.load_state_dict(winner["model_state_dict"], strict=True)
            by_epoch[winner["epoch"]] = model.to(device).eval().requires_grad_(False)
        models[branch] = by_epoch[winner["epoch"]]
    return models


def train(args):
    if args.head or args.evaluate_manifest or not args.val_manifest:
        raise ValueError("training accepts only TRAIN and VAL; evaluate TEST/REAL in a separate invocation")
    train_cache = MatrixCache(args.train_manifest, expected_split="train", expected_count=args.expected_train_count)
    val_cache = MatrixCache(args.val_manifest, expected_split="val", checkpoint_id=train_cache.checkpoint_id,
                            expected_count=args.expected_val_count)
    if val_cache.precision != train_cache.precision:
        raise ValueError("TRAIN and VAL matcher inference precision differs")
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    model = MatrixPairHead(MatrixPairHeadConfig(widths=tuple(args.widths), include_affinity=not args.transport_only)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    receipt = dict(schema_version=HEAD_SCHEMA, status="provisional", model_config=asdict(model.config),
                   architecture=model.architecture_metadata(), matcher_checkpoint_id=train_cache.checkpoint_id,
                   matcher_precision=train_cache.precision, train_cache=train_cache.provenance(), val_cache=val_cache.provenance(),
                   seed=args.seed, selection_primary=args.selection_metric,
                   selection_rule="per-head validation " + args.selection_metric + "; other of F1/AP breaks ties; earlier epoch wins exact ties",
                   selection_pose_used=False, test_or_real_used_for_fit=False,
                   head_threshold_rule="whole-validation equal-row F1; largest threshold on ties",
                   selected_heads={}, training_arguments=vars(args))
    write_json(output / "protocol.json", {key: value for key, value in receipt.items() if key != "selected_heads"})
    try:
        for epoch in range(1, args.epochs + 1):
            start = time.monotonic()
            model.train()
            counts, matrix_loss_sum, fusion_loss_sum = 0, 0.0, 0.0
            write_json(output / "status.json", dict(status="running", epoch=epoch, phase="train",
                                                     completed_epochs=epoch - 1))
            for batch_index, batch in enumerate(train_cache.batches(args.batch_size, shuffle=True, seed=args.seed + epoch), 1):
                tensors = model_inputs(batch, args.device)
                labels = torch.from_numpy(batch["label"]).to(args.device).float()
                optimizer.zero_grad(set_to_none=True)
                result = model(**tensors)
                if not result.valid_problem.all():
                    raise ValueError("TRAIN contains an empty contour problem; no silent sample exclusion")
                matrix_loss = F.binary_cross_entropy_with_logits(result.matrix_logit, labels)
                fusion_loss = F.binary_cross_entropy_with_logits(result.matrix_coarse_logit, labels)
                (matrix_loss + fusion_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                count = len(labels)
                counts += count
                matrix_loss_sum += float(matrix_loss.detach()) * count
                fusion_loss_sum += float(fusion_loss.detach()) * count
                if args.log_every_batches and batch_index % args.log_every_batches == 0:
                    print(json.dumps(dict(event="train_progress", epoch=epoch, processed=counts,
                                          sample_count=train_cache.sample_count,
                                          matrix_bce=matrix_loss_sum / counts,
                                          matrix_coarse_bce=fusion_loss_sum / counts)), flush=True)
            if counts != train_cache.sample_count:
                raise ValueError("training did not visit exact TRAIN row count")
            write_json(output / "status.json", dict(status="running", epoch=epoch, phase="validation",
                                                     completed_epochs=epoch - 1))
            validation = predict(val_cache, {name: model for name in BRANCHES}, args.batch_size, args.device)
            metrics = {}
            for name in BRANCHES:
                threshold = fit_f1_threshold(validation["labels"], validation["scores"][name])
                metrics[name] = classification(validation["labels"], validation["scores"][name], threshold)
                key = selection_key(metrics[name], args.selection_metric)
                previous = receipt["selected_heads"].get(name)
                if previous is None or key > tuple(previous["selection_key"]):
                    receipt["selected_heads"][name] = dict(epoch=epoch, threshold=threshold, metrics=metrics[name],
                        selection_key=list(key), model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            row = dict(epoch=epoch, train_sample_count=counts, matrix_bce=matrix_loss_sum / counts,
                       matrix_coarse_bce=fusion_loss_sum / counts, val_sample_count=val_cache.sample_count,
                       val_positive_count=int(validation["labels"].sum()), metrics=metrics,
                       elapsed_seconds=time.monotonic() - start,
                       selected_epochs={k: v["epoch"] for k, v in receipt["selected_heads"].items()})
            write_json(output / ("epoch_%02d.json" % epoch), row)
            torch.save(receipt, output / "head.provisional.pt")
            print(json.dumps(row, allow_nan=False), flush=True)
        # Re-evaluate ONLY the independently VAL-selected branch winners, then
        # freeze all learned-head and baseline decision thresholds before any
        # held-out manifest can be opened by the separate evaluation mode.
        validation = predict(val_cache, selected_models(receipt, args.device), args.batch_size, args.device)
        policies = freeze_policies(validation, val_cache.manifest.get("original_thresholds"), args.gate_recalls)
        receipt.update(status="complete", policies=policies, completed_epochs=args.epochs)
        torch.save(receipt, output / "head.pt")
        public_receipt = {key: value for key, value in receipt.items() if key != "selected_heads"}
        public_receipt["selected_heads"] = {name: {key: value for key, value in winner.items() if key != "model_state_dict"}
                                            for name, winner in receipt["selected_heads"].items()}
        write_json(output / "validation_freeze.json", public_receipt)
        save_population(output / "val", val_cache, validation, policies, receipt)
        write_json(output / "status.json", dict(status="complete", completed_epochs=args.epochs,
                    selected_epochs={k: v["epoch"] for k, v in receipt["selected_heads"].items()}))
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise


def evaluate(args):
    if not args.head or not args.evaluate_manifest or args.train_manifest or args.val_manifest:
        raise ValueError("evaluation needs frozen --head and --evaluate-manifest, no TRAIN/VAL inputs")
    receipt = torch.load(args.head, map_location="cpu", weights_only=False)
    if receipt.get("schema_version") != HEAD_SCHEMA or receipt.get("status") != "complete":
        raise ValueError("evaluation requires completed, validation-frozen head.pt")
    policies = receipt["policies"]
    models = selected_models(receipt, args.device)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    for index, path in enumerate(args.evaluate_manifest):
        cache = MatrixCache(path, checkpoint_id=receipt["matcher_checkpoint_id"])
        if cache.split not in ("test", "real"):
            raise ValueError("held-out evaluation accepts TEST/REAL only, not threshold refitting")
        if cache.precision != receipt["matcher_precision"]:
            raise ValueError("held-out cache matcher precision differs from TRAIN/VAL")
        predictions = predict(cache, models, args.batch_size, args.device)
        directory = output / cache.split if len(args.evaluate_manifest) == 1 else output / (str(index) + "_" + cache.split)
        metrics = save_population(directory, cache, predictions, policies, receipt)
        completed.append(dict(split=cache.split, path=str(directory), sample_count=metrics["prediction_count"]))
    write_json(output / "status.json", dict(status="complete", evaluations=completed, thresholds_refitted=False))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-manifest")
    p.add_argument("--val-manifest")
    p.add_argument("--head")
    p.add_argument("--evaluate-manifest", action="append", default=[])
    p.add_argument("--output-root", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--log-every-batches", type=int, default=100)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=260909)
    p.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--transport-only", action="store_true", help="omit affinity channel, keep soft transport plus valid mask")
    p.add_argument("--selection-metric", choices=["f1", "auprc"], default="f1")
    p.add_argument("--gate-recalls", type=float, nargs="+", default=[0.99, 0.995, 1.0])
    p.add_argument("--expected-train-count", type=int, default=24000)
    p.add_argument("--expected-val-count", type=int, default=3000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.cpu_threads < 1:
        raise ValueError("cpu threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0 or args.weight_decay < 0 or args.log_every_batches < 0:
        raise ValueError("invalid training hyperparameters")
    if args.train_manifest:
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
