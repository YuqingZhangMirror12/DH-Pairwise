"""One real-data CUDA backward per configuration; never saves learned weights."""
import argparse
from dataclasses import replace
import json
import time
from pathlib import Path

import torch

from experiments.rachel_n512_formal_30k.run_architecture_ablation import initialize_variant
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from experiments.rachel_n512_formal_30k.run_layout_decoder_experiment import DEFAULT_RUN, DEFAULT_DATA
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed
from staging.pairwise_v0_2.training.rachel_n512_loss import compute_rachel_n512_loss
from staging.pairwise_v0_2.training.seam_consistency_loss import (
    build_seam_consistency_targets, compute_seam_consistency_loss,
)
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=DEFAULT_RUN)
    p.add_argument("--dataset", default=DEFAULT_DATA)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--resample-contour-cap", type=int, choices=(512, 1024))
    p.add_argument("--variants", nargs="+", default=["baseline_control", "coarse512", "multiscale7_16_32_64", "seam_loss"])
    args = p.parse_args()
    torch.set_num_threads(1)
    runner._set_determinism(260907)
    _, _, winners = sealed._freeze_completed_winners(Path(args.run))
    base = next(w for w in winners if w.arm == "full_n512")
    dataset = RachelPairDataset(Path(args.dataset), "train")
    model_config = base.model_config
    if args.resample_contour_cap:
        from staging.pairwise_v0_2.pairwise_data.rachel_resampled_dataset import RachelResampledDataset
        dataset = RachelResampledDataset(dataset, contour_cap=args.resample_contour_cap)
        model_config = replace(model_config, contour_cap=args.resample_contour_cap)
    manifest = runner._read_manifest_rows(Path(args.dataset), "train")
    indices = runner.stratified_subset_indices([r.label for r in manifest], limit=args.batch_size, seed=260907)
    batch = next(iter(make_ablation_loader(dataset, indices, batch_size=args.batch_size,
                 num_workers=0, seed=260907, contour_cap=model_config.contour_cap)))
    device = torch.device("cuda:0")
    inputs, targets = runner._full_batch(batch, device)
    cfg = replace(base.loss_config, collect_cpu_diagnostics=False)
    for variant in args.variants:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = initialize_variant(model_config, base.model.state_dict(), variant).to(device).train()
        started = time.perf_counter()
        output = model(*inputs)
        loss = compute_rachel_n512_loss(output, *targets, config=cfg).total
        seam_count = 0
        if variant == "seam_loss":
            target = build_seam_consistency_targets(batch.points_rc_a, batch.points_rc_b,
                batch.contour_valid_a, batch.contour_valid_b, batch.target_a, batch.target_b).to(device)
            seam = compute_seam_consistency_loss(output.assignment, target, output.training_valid)
            seam_count = target.adjacency_count
            loss = loss + .1 * seam.total
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        torch.cuda.synchronize()
        print(json.dumps(dict(variant=variant, batch_size=args.batch_size, loss=float(loss.detach()),
            resample_contour_cap=args.resample_contour_cap,
            gradient_norm=float(norm), seam_adjacencies=seam_count, seconds=time.perf_counter()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            decision_valid=output.decision_valid.detach().cpu().tolist(),
            strict_deterministic=torch.are_deterministic_algorithms_enabled())), flush=True)
        del output, loss, model


if __name__ == "__main__":
    main()
