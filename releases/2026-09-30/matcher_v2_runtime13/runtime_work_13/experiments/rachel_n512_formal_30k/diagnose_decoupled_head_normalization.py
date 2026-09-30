"""Eight TRAIN pairs, CPU-only: same frozen Q, two cloned BN forward modes.

This is not model selection, calibration or a replacement evaluation. It never
loads TEST/REAL/OOD, optimizes weights, changes tau, or writes a checkpoint.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import time

# This standalone diagnostic must not acquire the active training GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.nn import functional as F

from experiments.rachel_n512_formal_30k.train_joint_damage import state_digest
from experiments.rachel_n512_formal_30k.train_edge_weathering import _sha256
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import save_json
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import MaterializedRachelDataset

SCHEMA = "rachel-decoupled-train-head-normalization-diagnostic/1"
FIELDS = ("mask_a", "mask_b", "points_rc_a", "points_rc_b", "contour_valid_a", "contour_valid_b")


def select_train_indices(training):
    if training.split != "train":
        raise ValueError("normalization diagnostic is TRAIN-only")
    by_label = {0: [], 1: []}
    for index, row in enumerate(training.rows):
        label = row["label"]
        if label not in (0, 1, False, True):
            raise ValueError("requires binary TRAIN labels")
        if len(by_label[int(label)]) < 4:
            by_label[int(label)].append(index)
        if all(len(indices) == 4 for indices in by_label.values()):
            return by_label[1] + by_label[0]
    raise ValueError("requires four TRAIN positives and four TRAIN negatives")


def collect_frozen_q(model, loader):
    """Exactly one base forward per pair; classifier/GT layout are not used."""
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("diagnostic accepts only a CPU model")
    model.eval().requires_grad_(False)
    before = state_digest(model.base_model)
    samples = []
    with torch.no_grad():
        for batch in loader:
            if len(batch.pair_ids) != 1 or len(samples) >= 8:
                raise ValueError("diagnostic is bounded to exactly eight single-pair forwards")
            inputs = [torch.as_tensor(getattr(batch, field), device="cpu",
                dtype=torch.bool if field.startswith("contour_valid") else torch.float32) for field in FIELDS]
            output = model.base_model(*inputs)
            samples.append(dict(pair_id=batch.pair_ids[0], label=int(batch.labels[0]),
                assignment=output.assignment.detach().clone(), valid_a=inputs[4].clone(), valid_b=inputs[5].clone(),
                base_training_valid=bool(output.training_valid[0])))
    if len(samples) != 8 or state_digest(model.base_model) != before:
        raise ValueError("incomplete eight-pair probe or modified frozen base state")
    return samples


def compare_head_modes(head, samples):
    """Two independent clones; TRAIN clone changes only its disposable BN buffers."""
    if (len(samples) != 8 or [row["label"] for row in samples] != [1] * 4 + [0] * 4 or
            any(p.device.type != "cpu" for p in head.parameters())):
        raise ValueError("requires ordered 4-positive/4-negative CPU TRAIN evidence")
    source_before = state_digest(head)
    heads = {"eval_running_stats": deepcopy(head).eval().requires_grad_(False),
             "train_per_pair_stats": deepcopy(head).train().requires_grad_(False)}
    params_before = {name: {key: value.detach().clone() for key, value in clone.named_parameters()}
                     for name, clone in heads.items()}
    buffers_before = {name: {key: value.detach().clone() for key, value in clone.named_buffers()}
                      for name, clone in heads.items()}
    rows = []
    with torch.no_grad():
        for sample in samples:
            q, va, vb = sample["assignment"], sample["valid_a"], sample["valid_b"]
            if any(x.device.type != "cpu" for x in (q, va, vb)):
                raise ValueError("cached Q and valid masks must remain on CPU")
            original = (q.clone(), va.clone(), vb.clone())
            row = {key: sample[key] for key in ("pair_id", "label", "base_training_valid")}
            for name, clone in heads.items():
                logit = clone(q, va, vb).reshape(())
                if not torch.isfinite(logit):
                    raise ValueError("nonfinite normalization probe output")
                row[name] = dict(logit=float(logit), score=float(logit.sigmoid()),
                    bce=float(F.binary_cross_entropy_with_logits(logit, logit.new_tensor(sample["label"]))))
            if not all(torch.equal(before, after) for before, after in zip(original, (q, va, vb))):
                raise ValueError("head modified shared Q/valid input")
            row["eval_minus_train_score"] = row["eval_running_stats"]["score"] - row["train_per_pair_stats"]["score"]
            rows.append(row)
    modes = {}
    for name, clone in heads.items():
        selected = [row[name] for row in rows]
        modes[name] = dict(mean_bce=sum(x["bce"] for x in selected) / 8,
            logical_micro1_valid_weighted_bce=sum(row[name]["bce"] * row["base_training_valid"] for row in rows) / 8,
            positive_scores=[row[name]["score"] for row in rows[:4]],
            negative_scores=[row[name]["score"] for row in rows[4:]],
            positive_mean_score=sum(row[name]["score"] for row in rows[:4]) / 4,
            negative_mean_score=sum(row[name]["score"] for row in rows[4:]) / 4,
            parameters_unchanged=all(torch.equal(params_before[name][key], value) for key, value in clone.named_parameters()),
            changed_buffer_names=[key for key, value in clone.named_buffers() if not torch.equal(buffers_before[name][key], value)])
        modes[name]["positive_minus_negative_mean_score"] = modes[name]["positive_mean_score"] - modes[name]["negative_mean_score"]
    if state_digest(head) != source_before or not all(mode["parameters_unchanged"] for mode in modes.values()):
        raise ValueError("normalization diagnostic modified original or clone parameters")
    if modes["eval_running_stats"]["changed_buffer_names"]:
        raise ValueError("evaluation clone unexpectedly changed its buffers")
    return dict(rows=rows, modes=modes, sample_count=8,
        valid_training_pair_count=sum(row["base_training_valid"] for row in rows),
        mean_absolute_score_gap=sum(abs(row["eval_minus_train_score"]) for row in rows) / 8,
        eval_minus_train_mean_bce=modes["eval_running_stats"]["mean_bce"] - modes["train_per_pair_stats"]["mean_bce"],
        original_head_state_sha256=source_before, original_head_unchanged=True,
        identical_cached_q_for_both_modes=True, weights_updated=False, thresholds_fitted=False,
        caveat="Eight preselected TRAIN pairs diagnose forward-mode mismatch only; not generalization, model selection, or calibration.")


def verify_converted_head(head, samples):
    """Same eight cached TRAIN inputs; explicit V3 must match V2 training."""
    from staging.pairwise_v0_2.models.rachel_decoupled_score import (
        ThresholdedMatrixCNN, convert_matrix_head_revision_state)
    if head.revision != "bn_relu_pool_v2" or len(samples) != 8:
        raise ValueError("requires v2 and exactly eight cached TRAIN pairs")
    old = deepcopy(head).train().requires_grad_(False)
    new = ThresholdedMatrixCNN(head.threshold, revision="per_pair_norm_v3")
    new.load_state_dict(convert_matrix_head_revision_state(head.state_dict(),
        source_revision="bn_relu_pool_v2", target_revision="per_pair_norm_v3"), strict=True)
    new.requires_grad_(False)
    rows = []
    with torch.no_grad():
        for sample in samples:
            inputs = (sample["assignment"], sample["valid_a"], sample["valid_b"])
            before = old(*inputs)
            new.train()
            training = new(*inputs)
            new.eval()
            evaluation = new(*inputs)
            if not torch.allclose(before, training, atol=1e-6, rtol=1e-6):
                raise ValueError("converted TRAIN forward differs from v2 training")
            if not torch.equal(training, evaluation):
                raise ValueError("converted train/eval forward differs")
            rows.append(dict(pair_id=sample["pair_id"], label=sample["label"],
                v2_train_score=float(before.sigmoid()), v3_train_score=float(training.sigmoid()),
                v3_eval_score=float(evaluation.sigmoid()),
                v2_train_v3_eval_absolute_logit_error=float((before - evaluation).abs().max())))
    same_parameters = all(torch.equal(p, dict(new.named_parameters())[name]) for name, p in head.named_parameters())
    if not same_parameters:
        raise ValueError("conversion changed trainable parameters")
    return dict(revision="per_pair_norm_v3", rows=rows, sample_count=8,
        maximum_absolute_logit_error=max(row["v2_train_v3_eval_absolute_logit_error"] for row in rows),
        train_eval_identical=True, trainable_parameters_identical=True,
        additional_base_forwards=0, held_out_read=False, new_training_exposures=0)


def run(args):
    torch.set_num_threads(1)
    from experiments.rachel_n512_formal_30k.train_score_decoupled import load_decoupled_checkpoint
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    manifest = Path(args.train_materialized_manifest).resolve(strict=True)
    checkpoint_sha = _sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = payload["resume_identity"]
    if (payload.get("epoch") != 20 or payload.get("phase") != "classifier" or
            identity.get("sampling") != "original512" or identity.get("head_kind") != "matrix_cnn" or
            identity.get("matrix_head_revision") != "bn_relu_pool_v2"):
        raise ValueError("requires frozen epoch20 original512 matrix v2 checkpoint")
    model = load_decoupled_checkpoint(payload).cpu().eval().requires_grad_(False)
    del payload
    train_record = identity["populations"]["train"]
    if train_record["split"] != "train" or train_record["manifest_sha256"] != _sha256(manifest):
        raise ValueError("TRAIN manifest differs from frozen training identity")
    training = MaterializedRachelDataset(manifest)
    if len(training) != 24000 or training.contour_cap != model.config.contour_cap:
        raise ValueError("requires original fixed Full24 TRAIN population and contour cap")
    indices = select_train_indices(training)
    loader = make_ablation_loader(training, indices, batch_size=1, num_workers=0,
        seed=identity["seed"], contour_cap=model.config.contour_cap)
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_before = state_digest(model)
    samples = collect_frozen_q(model, loader)
    report = compare_head_modes(model.score_head, samples)
    report["converted_v3_mode_consistency"] = verify_converted_head(model.score_head, samples)
    if source_before != state_digest(model) or checkpoint_sha != _sha256(checkpoint):
        raise ValueError("original model or checkpoint changed during diagnostic")
    report.update(schema_version=SCHEMA, status="complete", checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        source_model_state_sha256=source_before, original_model_unchanged=True, original_checkpoint_unchanged=True,
        matrix_head_revision=model.matrix_head_revision, matrix_threshold=model.matrix_threshold,
        train_manifest=str(manifest), train_manifest_sha256=train_record["manifest_sha256"], source_indices=indices,
        population="TRAIN first4 positives then first4 negatives", base_forward_count=8, base_always_eval=True,
        device="cpu", torch_threads=1, torch_version=torch.__version__, held_out_read=False,
        elapsed_seconds=time.monotonic() - started)
    save_json(destination / "report.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--train-materialized-manifest", required=True)
    result.add_argument("--output", required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
