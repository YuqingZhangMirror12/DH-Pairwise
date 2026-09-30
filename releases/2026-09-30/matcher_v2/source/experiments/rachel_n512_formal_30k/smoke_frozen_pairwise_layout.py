"""One real-checkpoint validation batch tests the delivered layout factory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import Subset

from staging.pairwise_v0_2.models.rachel_pairwise_layout import load_frozen_pairwise_layout
from staging.pairwise_v0_2.models.translation_layout import estimate_translation_layout
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(1)
    with (Path(args.run_directory) / "run_receipt.json").open() as stream:
        run_receipt = json.load(stream)
    seed = int(run_receipt["config"]["seed"])
    sealed._set_determinism(seed)
    if args.shredding_freeze:
        from staging.pairwise_v0_2.models.rachel_full_shredding_layout import load_frozen_full_shredding_layout
        model = load_frozen_full_shredding_layout(
            args.run_directory, args.validation_freeze, args.shredding_freeze, device=args.device
        )
    else:
        model = load_frozen_pairwise_layout(args.run_directory, args.validation_freeze, device=args.device)
    selected = model.deployment_metadata["selected_full_decoder"]
    config = model.layout_configs[selected]
    assert list(model.layout_configs) == [selected]
    assert not model.training and not model.pair_model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    dataset = RachelPairDataset(Path(args.dataset), "val")
    sample_count = min(args.sample_count, len(dataset))
    first_batch = Subset(dataset, list(range(sample_count)))
    batch = next(iter(sealed._test_loader(first_batch, batch_size=16, num_workers=0, seed=seed)))
    device = torch.device(args.device)
    tensors = [sealed._tensor(getattr(batch, name), device, dtype) for name, dtype in (
        ("mask_a", torch.float32), ("mask_b", torch.float32),
        ("points_rc_a", torch.float32), ("points_rc_b", torch.float32),
        ("contour_valid_a", torch.bool), ("contour_valid_b", torch.bool),
    )]
    score_names = ("coarse_logit", "coarse_probability", "local_logit", "local_probability",
                   "fused_logit", "fused_probability")
    captured = []
    captured_shred = []

    def capture_original(_, __, output):
        captured.append((output, {name: getattr(output, name).clone() for name in score_names},
                         output.translation_hat_rc.clone(), output.translation_dispersion_px.clone()))

    hook = model.pair_model.register_forward_hook(capture_original)
    shred_hook = model.shredding_head.register_forward_hook(
        lambda module, inputs, output: captured_shred.append(output)
    ) if args.shredding_freeze else None
    try:
        with torch.inference_mode():
            output = model(*tensors)
    finally:
        hook.remove()
        if shred_hook is not None:
            shred_hook.remove()
    assert len(captured) == 1, "the smoke must execute exactly one Full forward"
    original, score_before, old_translation, old_dispersion = captured[0]
    assert output.pair_output is original
    score_checks = {}
    for name in score_names:
        identity = getattr(output, name) is getattr(original, name)
        unchanged = torch.equal(getattr(output, name), score_before[name])
        assert identity and unchanged, name + " changed across independent layout decoding"
        score_checks[name] = {"same_tensor_object": identity, "values_unchanged": unchanged,
                              "max_absolute_difference": 0.0}
    assert torch.equal(original.translation_hat_rc, old_translation)
    assert torch.equal(original.translation_dispersion_px, old_dispersion)
    layout = output.layouts[selected]
    assert layout.computed.all(), "a score must never route a pair away from layout"
    source = original.affinity if config.score_mode == "dual_softmax" else original.assignment
    confidence = source.detach().cpu().double().numpy()
    pa, pb = (value.detach().cpu().double().numpy() for value in tensors[2:4])
    va, vb = (value.detach().cpu().numpy() for value in tensors[4:6])
    max_translation_difference = 0.0
    max_batched_roundoff = 0.0
    max_inlier_residual = 0.0
    pair_checks = []
    scalar_names = ("valid", "reason", "candidate_count", "inlier_count", "inlier_fraction",
                    "weighted_inlier_fraction", "residual_px", "runner_up_support_ratio",
                    "support_weight", "runner_up_support_weight")
    for index, pair_id in enumerate(batch.pair_ids):
        independent = estimate_translation_layout(pa[index], pb[index], confidence[index],
                                                 va[index], vb[index], config=config)
        wrapped = layout.results[index]
        assert wrapped is not None
        assert np.array_equal(wrapped.t_a_to_b_rc, independent.t_a_to_b_rc, equal_nan=True)
        assert np.array_equal(wrapped.candidate_indices, independent.candidate_indices)
        assert np.array_equal(wrapped.inlier_mask, independent.inlier_mask)
        assert all(getattr(wrapped, name) == getattr(independent, name) for name in scalar_names)
        row = {"pair_id": pair_id, "valid": wrapped.valid,
               "candidate_count": wrapped.candidate_count, "inlier_count": wrapped.inlier_count,
               "independent_decoder_exact_match": True, "reason": wrapped.reason}
        if wrapped.valid:
            tensor_t = layout.t_a_to_b_rc[index].detach().cpu().double().numpy()
            tensor_offset = layout.offset_b_in_a_rc[index].detach().cpu().double().numpy()
            assert np.array_equal(tensor_offset, -tensor_t)
            difference = float(np.max(np.abs(wrapped.t_a_to_b_rc - independent.t_a_to_b_rc)))
            roundoff = float(np.max(np.abs(tensor_t - wrapped.t_a_to_b_rc)))
            max_translation_difference = max(max_translation_difference, difference)
            max_batched_roundoff = max(max_batched_roundoff, roundoff)
            edges = wrapped.candidate_indices[wrapped.inlier_mask]
            placed_b = pb[index, edges[:, 1]] - wrapped.t_a_to_b_rc
            residual = np.linalg.norm(placed_b - pa[index, edges[:, 0]], axis=1)
            assert np.all(residual <= config.inlier_radius_px + 1e-9)
            max_inlier_residual = max(max_inlier_residual, float(residual.max()))
            row.update(t_a_to_b_rc=wrapped.t_a_to_b_rc.tolist(),
                       offset_b_in_a_rc=(-wrapped.t_a_to_b_rc).tolist(),
                       tensor_offset_is_negative_translation=True,
                       max_placed_inlier_residual_px=float(residual.max()),
                       max_batched_float32_roundoff_px=roundoff)
        pair_checks.append(row)
    result = {
        "status": "complete_real_checkpoint_factory_smoke",
        "source_split": "validation", "selection": "first validation batch, manifest order",
        "sample_count": sample_count, "pair_ids": list(batch.pair_ids),
        "model_forward_invocations": len(captured), "training_performed": False,
        "model_parameters_frozen": True, "model_evaluation_mode": True,
        "metadata": model.deployment_metadata, "selected_decoder_config": asdict(config),
        "score_preservation": score_checks,
        "original_pair_output_object_preserved": True,
        "original_translation_and_dispersion_preserved": True,
        "all_pairs_decoded_without_score_routing": True,
        "valid_layout_count": int(layout.valid.sum().item()),
        "independent_decoder_exact_match_count": sample_count,
        "max_independent_translation_difference_px": max_translation_difference,
        "placement_convention": "B offset in A frame = -t_a_to_b_rc; rotation is zero",
        "max_placed_inlier_residual_px": max_inlier_residual,
        "max_batched_float32_roundoff_px": max_batched_roundoff,
        "pair_checks": pair_checks,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.shredding_freeze:
        assert len(captured_shred) == 1
        fine = captured_shred[0]
        primary = output.primary_layout
        assert primary.head_output is fine
        assert primary.computed.all()
        assert np.array_equal(primary.t_a_to_b_rc.cpu().numpy(), fine.translation_hat_rc,
                              equal_nan=True)
        assert np.array_equal(primary.valid.cpu().numpy(), fine.translation_valid)
        assert np.array_equal(primary.offset_b_in_a_rc.cpu().numpy(), -fine.translation_hat_rc,
                              equal_nan=True)
        result["primary_shredding_layout"] = {
            "primary_layout_name": output.primary_layout_name,
            "head_forward_invocations": len(captured_shred),
            "head_output_object_preserved": True,
            "head_translation_values_unchanged": True,
            "head_validity_values_unchanged": True,
            "b_offset_is_negative_translation": True,
            "all_pairs_use_fixed_fine_matcher": True,
            "shredding_coarse_scorer_executed": False,
            "shredding_pair_classifier_executed": False,
            "valid_layout_count": int(fine.translation_valid.sum()),
            "correspondence_counts": fine.correspondence_count.tolist(),
            "inlier_counts": fine.inlier_count.tolist(),
            "translations_rc": [value.tolist() if valid else None for value, valid in
                                zip(fine.translation_hat_rc, fine.translation_valid)],
        }
        result["status"] = "complete_real_checkpoint_full_shredding_factory_smoke"
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "sample_count": sample_count,
                      "valid_layout_count": result["valid_layout_count"],
                      "independent_decoder_exact_match_count": sample_count,
                      "max_independent_translation_difference_px": max_translation_difference,
                      "max_batched_float32_roundoff_px": max_batched_roundoff,
                      "elapsed_seconds": result["elapsed_seconds"], "receipt": str(destination)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--validation-freeze", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shredding-freeze")
    parser.add_argument("--sample-count", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.sample_count <= 16:
        parser.error("--sample-count must be between 1 and 16")
    run(args)


if __name__ == "__main__":
    main()
