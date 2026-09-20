"""Matched architecture runs; train/validation only.

By default every arm starts from the same completed Full checkpoint. Explicit
--initialization scratch instead shares a new, seed-specific random Full base
state across arms, borrowing only its config/loss config from the completed run.
Examples, optimizer settings, updates and selection are unchanged. Neither mode
alone establishes that a scale is optimal. Test/real sets are never opened here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Pairwise
from staging.pairwise_v0_2.models.rachel_model_factory import model_metadata
from staging.pairwise_v0_2.models.translation_layout import (
    TranslationLayoutConfig, estimate_translation_layout,
)
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed
from staging.pairwise_v0_2.training.rachel_n512_loss import compute_rachel_n512_loss
from experiments.rachel_n512_formal_30k.run_layout_decoder_experiment import (
    DEFAULT_DATA, DEFAULT_RUN, clean, classification, fit_threshold,
)
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader


VARIANTS = {
    "baseline_control": {},
    "coarse256": {"coarse_size": 256},
    "coarse512": {"coarse_size": 512},
    "window7_16": {"window_sizes_px": (7.0, 16.0)},
    "window7_only": {"window_sizes_px": (7.0,)},
    "window16_only": {"window_sizes_px": (16.0,)},
    "window32_only": {"window_sizes_px": (32.0,)},
    "window64_only": {"window_sizes_px": (64.0,)},
    "multiscale7_16_32_64": {"window_sizes_px": (7.0, 16.0, 32.0, 64.0)},
    "seam_loss": {},
}
ARCHITECTURE_VARIANTS = ("hierarchical_image", "hierarchical_contour", "multiscale_transport7_16_32_64")
ALL_VARIANTS = tuple(VARIANTS) + ARCHITECTURE_VARIANTS


def initialize_variant(base_config, base_state, variant):
    """Keep supplied base parameters, but regenerate the sampling-grid buffer.

    Loading a same-shaped 32/64 grid into a 7/16-configured model would silently
    invalidate the window ablation. The supplied base may be trained or freshly
    randomized; new architecture-specific parameters use their constructors.
    """
    if variant in ("hierarchical_image", "hierarchical_contour"):
        from staging.pairwise_v0_2.models.rachel_hierarchical import (
            RachelHierarchicalConfig, RachelHierarchicalPairwise,
        )
        model = RachelHierarchicalPairwise(base_config, RachelHierarchicalConfig(
            coarse_source=variant[len("hierarchical_"):]))
        model.load_full_state_dict(base_state)
        return model
    if variant == "multiscale_transport7_16_32_64":
        from staging.pairwise_v0_2.models.rachel_multiscale_transport import initialize_multiscale_transport_from_base
        return initialize_multiscale_transport_from_base(base_config, base_state)
    config = replace(base_config, **VARIANTS[variant])
    model = RachelN512Pairwise(config)
    state = dict(base_state)
    state["patch_sampler.offsets_rc"] = model.patch_sampler.offsets_rc
    model.load_state_dict(state, strict=True)
    return model


def make_initial_base_state(base_config, trained_state=None, *, initialization="warm-start", seed):
    """Shared base for all arms; scratch never reads ``trained_state``.

    Seed the CPU generator in an isolated scope because construction is on CPU.
    Do not perturb the caller's RNG or CUDA state. Per-arm construction and
    training retain their existing explicit determinism reset in ``run``.
    """
    if initialization == "warm-start":
        if trained_state is None:
            raise ValueError("warm-start requires the completed Full state")
        return {k: v.detach().cpu().clone() for k, v in trained_state.items()}
    if initialization != "scratch":
        raise ValueError("initialization must be 'warm-start' or 'scratch'")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        fresh_base = RachelN512Pairwise(base_config)
        return {k: v.detach().cpu().clone() for k, v in fresh_base.state_dict().items()}


def initialization_metadata(initialization, seed, checkpoint_path, checkpoint_epoch):
    inherited = initialization == "warm-start"
    return dict(initialization=initialization, seed=seed,
                inherits_trained_weights=inherited,
                initial_checkpoint=str(checkpoint_path) if inherited else None,
                initial_epoch=checkpoint_epoch if inherited else None,
                config_loss_source_checkpoint=str(checkpoint_path),
                config_loss_source_epoch=checkpoint_epoch,
                initial_state_source="completed_full_checkpoint" if inherited else "seeded_random_full_base",
                shared_base_state_across_arms=True)


def emit(value):
    print(json.dumps(clean(value), ensure_ascii=False, allow_nan=False), flush=True)


def save_json(path, value):
    runner._atomic_json(Path(path), clean(value))


def train_epoch(model, loader, optimizer, loss_config, device, args, epoch, variant):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_samples, total_loss, total_seam = 0, 0.0, 0.0
    started = time.perf_counter()
    updates = 0
    accumulation = args.effective_batch_size // args.batch_size
    for step, batch in enumerate(loader):
        group_start = (step // accumulation) * accumulation
        # Normalize by actual examples, including an incomplete final group.
        group_samples = min(args.effective_batch_size,
                            len(loader.dataset) - group_start * args.batch_size)
        inputs, targets = runner._full_batch(batch, device)
        output = model(*inputs)
        losses = compute_rachel_n512_loss(output, *targets, config=loss_config)
        loss = losses.total
        seam_value = loss.detach() * 0.0
        if variant == "seam_loss" or args.enable_seam_loss:
            from staging.pairwise_v0_2.training.seam_consistency_loss import (
                build_seam_consistency_targets, compute_seam_consistency_loss,
            )
            seam_targets = build_seam_consistency_targets(
                batch.points_rc_a, batch.points_rc_b,
                batch.contour_valid_a, batch.contour_valid_b,
                batch.target_a, batch.target_b,
            ).to(device)
            seam_result = compute_seam_consistency_loss(
                output.assignment, seam_targets, training_valid=output.training_valid,
            )
            seam_value = seam_result.total
            loss = loss + args.seam_weight * seam_value
        count = len(batch.pair_ids)
        (loss * (count / group_samples)).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        total_samples += count
        total_loss += float(loss.detach().cpu()) * count
        total_seam += float(seam_value.detach().cpu()) * count
        if (step + 1) % args.log_every == 0:
            emit(dict(event="train_progress", variant=variant, epoch=epoch,
                      samples=total_samples, updates=updates,
                      mean_loss=total_loss / total_samples,
                      seconds=time.perf_counter() - started))
    return dict(samples=total_samples, optimizer_updates=updates,
                mean_loss=total_loss / max(1, total_samples),
                mean_seam_loss=total_seam / max(1, total_samples),
                seconds=time.perf_counter() - started,
                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device))


def evaluate_validation(model, loader, device):
    model.eval()
    config = TranslationLayoutConfig(correspondence_mode="topk_union", top_k=2,
                                    max_candidates=512, min_inliers=3,
                                    inlier_radius_px=10.0)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            inputs, _ = runner._full_batch(batch, device)
            output = model(*inputs)
            matrices = output.assignment.detach().cpu().numpy()
            scores = {b: getattr(output, b + "_probability").detach().cpu().tolist()
                      for b in ("coarse", "local", "fused")}
            for i, pair_id in enumerate(batch.pair_ids):
                estimate = estimate_translation_layout(
                    batch.points_rc_a[i], batch.points_rc_b[i], matrices[i],
                    batch.contour_valid_a[i], batch.contour_valid_b[i], config=config,
                )
                error = None
                if batch.translation_valid[i] and estimate.valid:
                    error = float(np.linalg.norm(estimate.t_a_to_b_rc -
                                                 batch.translation_a_to_b_rc[i]))
                area_a, area_b = int(batch.mask_a[i].sum()), int(batch.mask_b[i].sum())
                rows.append(dict(pair_id=pair_id, label=bool(batch.labels[i]),
                                 classification={b: s[i] for b, s in scores.items()},
                                 decision_valid=bool(output.decision_valid[i].item()),
                                 translation_rc=clean(estimate.t_a_to_b_rc),
                                 translation_l2_px=error, layout_valid=estimate.valid,
                                 min_area_px=min(area_a, area_b),
                                 area_ratio=min(area_a, area_b) / max(1, area_a, area_b)))
    labels = np.array([r["label"] for r in rows], bool)
    report = dict(sample_count=len(rows), methods={}, pose_recall={})
    thresholds = {}
    for branch in ("coarse", "local", "fused"):
        values = np.array([r["classification"][branch] for r in rows])
        thresholds[branch] = fit_threshold(labels, values)
        report["methods"][branch] = classification(labels, values, thresholds[branch])
    errors = np.array([r["translation_l2_px"] if r["translation_l2_px"] is not None
                       else np.inf for r in rows])[labels]
    for tolerance in (2, 5, 8, 10):
        report["pose_recall"][str(tolerance)] = float(np.mean(errors <= tolerance))
    full = report["methods"]["fused"]
    pair_score = math.sqrt(full["auroc"] * full["auprc"])
    pose_score = float(np.mean(list(report["pose_recall"].values())))
    report["selection_score"] = math.sqrt(pair_score * pose_score)
    report["thresholds"] = thresholds
    report["decision_coverage"] = float(np.mean([r["decision_valid"] for r in rows]))
    return report, rows


def run(args):
    initialization = getattr(args, "initialization", "warm-start")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.effective_batch_size < args.batch_size or args.effective_batch_size % args.batch_size:
        raise ValueError("effective batch size must be a positive multiple of microbatch size")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("variants must not contain duplicates")
    if not torch.cuda.is_available():
        raise RuntimeError("This training entry point requires the remote CUDA server")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    receipt, _, winners = sealed._freeze_completed_winners(Path(args.run))
    full = next(w for w in winners if w.arm == "full_n512")
    del winners
    train = RachelPairDataset(Path(args.dataset), "train")
    validation = RachelPairDataset(Path(args.dataset), "val")
    base_config = full.model_config
    if args.resample_contour_cap is not None:
        from staging.pairwise_v0_2.pairwise_data.rachel_resampled_dataset import RachelResampledDataset
        train = RachelResampledDataset(train, contour_cap=args.resample_contour_cap)
        validation = RachelResampledDataset(validation, contour_cap=args.resample_contour_cap)
        base_config = replace(base_config, contour_cap=args.resample_contour_cap)
    base_state = make_initial_base_state(
        base_config, full.model.state_dict() if initialization == "warm-start" else None,
        initialization=initialization, seed=args.seed)
    initialization_record = initialization_metadata(
        initialization, args.seed, full.checkpoint_path, full.epoch)
    device = torch.device(args.device)
    torch.set_num_threads(1)
    save_json(destination / "protocol.json", dict(
        status="running", experiment=("matched-warm-start-screen/1" if initialization == "warm-start"
                                      else "matched-scratch-run/1"),
        **initialization_record,
        precision="fp32", variants={v: VARIANTS.get(v, {"architecture": v}) for v in args.variants},
        config=vars(args), training_pairs=len(train), validation_pairs=len(validation),
        selection="sqrt(sqrt(fused_AUROC*fused_AUPRC)*mean_pose_R2_R5_R8_R10)",
        thresholds="validation F1, fitted separately for each checkpoint and branch",
        no_from_scratch_claim=initialization != "scratch", test_used_for_selection=False,
        real_used_for_selection=False, token1024_requires_new_contours_and_targets=True,
        contour_source="mask_reextraction" if args.resample_contour_cap else "released_contours",
        target_policy="mask_dense_seam_rebuild/1" if args.resample_contour_cap else "released_targets",
        accumulation_note="Matched microbatches and order across arms; valid-target-normalized auxiliary losses are not mathematically identical to one large batch. Do not change microbatch for one arm only.",
    ))
    no_eligible_winner = []
    for variant in args.variants:
        runner._set_determinism(args.seed)
        arm_dir = destination / variant
        arm_dir.mkdir()
        model = initialize_variant(base_config, base_state, variant).to(device)
        save_json(arm_dir / "architecture.json", model_metadata(model))
        loss_config = replace(full.loss_config, collect_cpu_diagnostics=False,
                              validate_runtime_targets=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=1e-4)
        best_score = -math.inf
        for epoch in range(1, args.epochs + 1):
            torch.cuda.reset_peak_memory_stats(device)
            train_loader = make_ablation_loader(
                train, runner.epoch_indices(len(train), seed=args.seed, epoch=epoch, limit=None),
                batch_size=args.batch_size, num_workers=args.workers, seed=args.seed + epoch,
                contour_cap=base_config.contour_cap)
            train_report = train_epoch(model, train_loader, optimizer, loss_config,
                                       device, args, epoch, variant)
            val_loader = make_ablation_loader(validation, tuple(range(len(validation))),
                                        batch_size=args.batch_size, num_workers=args.workers,
                                        seed=args.seed, contour_cap=base_config.contour_cap)
            val_report, rows = evaluate_validation(model, val_loader, device)
            result = dict(epoch=epoch, variant=variant, training=train_report,
                          validation=val_report)
            save_json(arm_dir / ("epoch_%02d.json" % epoch), result)
            emit(dict(event="epoch_complete", **result))
            if val_report["decision_coverage"] == 1.0 and val_report["selection_score"] > best_score:
                best_score = val_report["selection_score"]
                checkpoint = dict(model_state_dict=model.state_dict(),
                                  **model_metadata(model),
                                  **initialization_record,
                                  loss_config=asdict(loss_config), epoch=epoch, variant=variant,
                                  resample_contour_cap=args.resample_contour_cap,
                                  seam_loss_enabled=variant == "seam_loss" or args.enable_seam_loss,
                                  seam_loss_weight=args.seam_weight)
                runner._atomic_torch_save(arm_dir / "winner.pt", checkpoint)
                save_json(arm_dir / "winner_validation.json", rows)
                save_json(arm_dir / "train_val_freeze.json", dict(
                    status="provisional", selected_epoch=epoch,
                    **initialization_record,
                    checkpoint=str(arm_dir / "winner.pt"), validation=val_report,
                    classifier_thresholds=val_report["thresholds"],
                    test_or_real_used_for_fit=False,
                ))
        freeze_path = arm_dir / "train_val_freeze.json"
        if freeze_path.exists():
            freeze = json.loads(freeze_path.read_text())
            freeze["status"] = "complete"
            save_json(freeze_path, freeze)
        else:
            no_eligible_winner.append(variant)
            save_json(arm_dir / "status.json", dict(
                status="no_eligible_winner", reason="No epoch had full validation decision coverage",
                training_completed=True, test_evaluation_authorized_by_freeze=False,
            ))
            emit(dict(event="no_eligible_winner", variant=variant))
        del model, optimizer
        torch.cuda.empty_cache()
    protocol_path = destination / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["status"] = "complete" if not no_eligible_winner else "completed_with_ineligible_arms"
    protocol["no_eligible_winner"] = no_eligible_winner
    save_json(protocol_path, protocol)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--dataset", default=DEFAULT_DATA)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=list(ALL_VARIANTS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260907)
    parser.add_argument("--initialization", choices=("warm-start", "scratch"), default="warm-start",
                        help="scratch borrows Full config/loss only and shares a fresh seeded base across arms")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seam-weight", type=float, default=0.1)
    parser.add_argument("--enable-seam-loss", action="store_true", help="Add the same seam auxiliary to every requested arm")
    parser.add_argument("--resample-contour-cap", type=int, choices=(512, 1024),
                        help="Rebuild both contours and GT; use a separate rebuilt512 control for a density comparison")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
