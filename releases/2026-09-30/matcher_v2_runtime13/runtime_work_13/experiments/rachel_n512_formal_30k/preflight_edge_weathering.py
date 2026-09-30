"""E1 GPU preflight: deterministic TRAIN128, eight updates, no saved weights.

Only the first128 rows of the formal seed260909/epoch1 permutation are used.
The ordinary E1 loader/loss/training loop is reused. Runtime target validation
is enabled for this small preflight only. Detailed source-mask checks cover
the first16 pairs plus, if needed, the first later actually changed positive.
Formal training must initialize again from the original pinned checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from experiments.rachel_n512_formal_30k import train_realism_data_ablation as base
from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour
from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import RachelWeatheredDataset
from staging.pairwise_v0_2.training.rachel_weathering_training import make_weathering_loader, train_weathering_epoch
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed


SOURCE_SHA256 = "350bf95413f828698f3396294a317569b013891d0906e2d4212d046fba059233"
PAIR_COUNT = 128
UPDATE_COUNT = 8
AUDIT_PREFIX = 16


def preflight_indices(length):
    if length != 24000:
        raise ValueError("E1 preflight requires the supplied matched24k training pool")
    return base.runner.epoch_indices(length, seed=base.SEED, epoch=1, limit=PAIR_COUNT)


def _same(left, right):
    return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)


def check_sample_contract(clean, batch, index, report):
    """Compare one observed microbatch item with its unchanged TRAIN source."""
    if (clean.pair_id != batch.pair_ids[index] or bool(clean.label) != bool(batch.labels[index])
            or clean.fragment_a_token != batch.fragment_a_tokens[index]
            or clean.fragment_b_token != batch.fragment_b_tokens[index]):
        raise ValueError("weathering changed pair identity, label, or endpoints")
    for field in ("translation_a_to_b_rc", "translation_a_to_b_xy_cartesian", "translation_valid"):
        if not _same(getattr(clean, field), getattr(batch, field)[index]):
            raise ValueError("weathering changed original frame/GT field: " + field)
    changed_sides = 0
    for side in "ab":
        original = np.asarray(getattr(clean, "mask_" + side))
        observed = np.asarray(getattr(batch, "mask_" + side)[index])
        if original.shape != (1, 800, 800) or observed.shape != original.shape:
            raise ValueError("weathering resized or reframed a model mask")
        if not np.isfinite(observed).all() or not np.all((observed == 0) | (observed == 1)):
            raise ValueError("weathered mask is not finite binary input")
        if np.any(observed.astype(bool) & ~original.astype(bool)):
            raise ValueError("weathering added foreground outside the original mask")
        changed = not np.array_equal(observed, original)
        if changed != bool(report["changed_" + side]):
            raise ValueError("actual mask change differs from effective changed-side report")
        if int(report["side_" + side]["effective_removed_area_px"]) != int(original.sum() - observed.sum()):
            raise ValueError("reported effective removed area differs from actual native-mask pixel count")
        points = getattr(batch, "points_rc_" + side)[index]
        valid = getattr(batch, "contour_valid_" + side)[index]
        if changed:
            expected_points, expected_valid = extract_ordered_outer_contour(observed[0].astype(bool),
                cap=512, smoothing_sigma=3.)
            changed_sides += 1
        else:
            expected_points = getattr(clean, "points_rc_" + side)
            expected_valid = getattr(clean, "contour_valid_" + side)
        n = len(expected_points)
        if (not _same(points[:n], expected_points) or not _same(valid[:n], expected_valid)
                or np.any(valid[n:])):
            raise ValueError("changed points were not re-extracted, or unchanged points/padding drifted")
    if bool(changed_sides) != bool(report["changed_pair"]):
        raise ValueError("effective changed-pair report differs from observed masks")
    if bool(report["pose_supervision_enabled"]) != (bool(clean.label) and not changed_sides):
        raise ValueError("pose sidecar must select unchanged positives only")
    ta, tb = batch.target_a[index], batch.target_b[index]
    va, vb = batch.contour_valid_a[index], batch.contour_valid_b[index]
    for first, second, first_valid, second_valid in ((ta, tb, va, vb), (tb, ta, vb, va)):
        for i in np.flatnonzero(first >= 0):
            j = int(first[i])
            if not (first_valid[i] and 0 <= j < len(second) and second_valid[j] and second[j] == i):
                raise ValueError("observed inherited targets are not reciprocal valid-token matches")
    if not bool(clean.label) and (np.any(ta[va] != -1) or np.any(tb[vb] != -1)):
        raise ValueError("weathered negative gained non-dustbin valid-token supervision")
    if not changed_sides:
        for target in ("target_a", "target_b"):
            old = getattr(clean, target)
            actual = getattr(batch, target)[index]
            if not _same(actual[:len(old)], old):
                raise ValueError("unchanged/fallback sample did not retain original targets")
    return dict(pair_id=clean.pair_id, positive=bool(clean.label), changed_pair=bool(changed_sides),
                changed_sides_reextracted=changed_sides, effective_supervised_match_count=int((ta >= 0).sum()),
                subset_frame_gt_reciprocity_checks_passed=True)


class CheckedLoader:
    """Pass through the real E1 batches; add only a bounded main-process check."""
    def __init__(self, loader, source, indices):
        self.loader, self.source, self.indices = loader, source, tuple(indices)
        self.dataset = loader.dataset
        self.observed = Counter()
        self.audit_rows = []
        self.audited_changed_positive = False

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        position = 0
        for wrapped in self.loader:
            batch = wrapped.batch
            for i, report in enumerate(wrapped.reports):
                if position >= len(self.indices) or report["pair_id"] != batch.pair_ids[i]:
                    raise ValueError("weathering report or preflight exposure order is misaligned")
                positive, changed = bool(batch.labels[i]), bool(report["changed_pair"])
                actual_matches = int(np.count_nonzero(batch.target_a[i] >= 0))
                if report.get("effective_supervised_match_count") != actual_matches:
                    raise ValueError("effective supervised match report differs from observed targets")
                if positive and changed and (actual_matches <= 0 or report.get("inherited_match_count", 0) <= 0):
                    raise ValueError("actually changed positive has no effective inherited match")
                name = "positive" if positive else "negative"
                self.observed[name + "_pair_count"] += 1
                self.observed["changed_" + name + "_pair_count"] += int(changed)
                self.observed["changed_positive_effective_match_tokens"] += actual_matches if positive and changed else 0
                if position < AUDIT_PREFIX or (positive and changed and not self.audited_changed_positive):
                    clean = self.source[self.indices[position]]
                    result = check_sample_contract(clean, batch, i, report)
                    self.audit_rows.append(result)
                    self.audited_changed_positive |= positive and changed
                position += 1
            yield wrapped
        if position != len(self.indices):
            raise ValueError("preflight did not iterate exactly the selected TRAIN128 rows")


class CheckedAdamW(torch.optim.AdamW):
    """In addition to the original clip check, explicitly record eight checks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finite_gradient_checks = 0

    def step(self, closure=None):
        gradients = [p.grad for group in self.param_groups for p in group["params"] if p.grad is not None]
        if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item():
            raise FloatingPointError("missing or nonfinite preflight gradients")
        result = super().step(closure)
        self.finite_gradient_checks += 1
        return result


def run(args):
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version="rachel-edge-weathering-preflight/1", status="running", seed=base.SEED,
        epoch=1, requested_pair_exposures=PAIR_COUNT, requested_optimizer_updates=UPDATE_COUNT,
        precision="fp32", microbatch_size=4, effective_batch_size=16, checkpoint=str(Path(args.checkpoint).resolve()),
        train_manifest=str(Path(args.train_manifest).resolve()), cache_dir=str(Path(args.cache_dir).resolve()),
        subset_rule="first128 of formal epoch_indices(matched24k, seed260909, epoch1)",
        geometry_audit_rule="first16 pairs plus first later actually changed positive if needed; at most17 pairs",
        weights_saved=False, formal_training_must_restart_from_original_source=True,
        validation_test_real_opened=False)
    base.save_json(output / "status.json", dict(status="running", phase="initializing", global_exposure=0, optimizer_updates=0))
    started = time.perf_counter()
    checked = optimizer = None
    try:
        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("this preflight must be explicitly launched on the remote CUDA server")
        torch.set_num_threads(1)
        checkpoint_path = Path(args.checkpoint).resolve(strict=True)
        checkpoint_sha = sealed._sha256_file(checkpoint_path)
        if checkpoint_sha != SOURCE_SHA256:
            raise ValueError("preflight requires the original pinned 350bf9 candidate")
        checkpoint = sealed._torch_load_checkpoint(checkpoint_path)
        model = load_rachel_checkpoint(checkpoint)
        metadata = base.check_fixed_architecture(model, checkpoint)
        source_loss = base.RachelN512LossConfig(**checkpoint["loss_config"])
        loss_config = replace(source_loss, validate_runtime_targets=True, collect_cpu_diagnostics=True)
        source, data_record = base.make_training_dataset(Path(args.dataset), Path(args.train_manifest))
        indices = preflight_indices(len(source))
        weathered = RachelWeatheredDataset(source, seed=base.SEED, epoch=1, cache_dir=args.cache_dir)
        raw_loader = make_weathering_loader(weathered, indices, batch_size=4, num_workers=args.workers,
            seed=base.SEED + 1, contour_cap=512)
        checked = CheckedLoader(raw_loader, source, indices)
        base.runner._set_determinism(base.SEED)
        device = torch.device(args.device)
        model = model.to(device).float().train().requires_grad_(True)
        optimizer = CheckedAdamW(model.parameters(), lr=base.LEARNING_RATE, weight_decay=1e-4)
        args.batch_size, args.effective_batch_size, args.log_every = 4, 16, 4
        report.update(checkpoint_sha256=checkpoint_sha, model_metadata=metadata,
            source_loss_config=asdict(source_loss), effective_preflight_loss_config=asdict(loss_config),
            loss_weights_changed=False, target_validation_enabled_for_preflight_only=True,
            training_data=data_record, selected_indices=list(indices), weathering_parameters=weathered.parameters,
            optimizer="AdamW", learning_rate=base.LEARNING_RATE, weight_decay=1e-4,
            loss_finite_enforcement="original/E1 runtime target and total-loss validation on every microbatch",
            gradient_finite_enforcement="original clip_grad_norm(error_if_nonfinite=True) and explicit AdamW step finite check")
        base.save_json(output / "preflight.json", report)
        torch.cuda.reset_peak_memory_stats(device)
        trained = train_weathering_epoch(model, checked, optimizer, loss_config, device, args, 1, "edge_weathering_preflight")
        torch.cuda.synchronize(device)
        report.update(training=trained, observations=dict(checked.observed), limited_geometry_checks=checked.audit_rows,
            finite_gradient_checks_completed=optimizer.finite_gradient_checks,
            gpu_peak_allocated_bytes=trained["peak_allocated_gpu_bytes"])
        if (trained["samples"], trained["optimizer_updates"], optimizer.finite_gradient_checks) != (128, 8, 8):
            raise ValueError("preflight did not complete exactly128 exposures/eight finite-gradient updates")
        if not math.isfinite(trained["mean_loss"]):
            raise FloatingPointError("preflight mean loss is nonfinite")
        if (checked.observed["changed_positive_pair_count"] < 1
                or checked.observed["changed_positive_effective_match_tokens"] < 1
                or not checked.audited_changed_positive):
            raise ValueError("preflight requires an actually changed positive with effective inherited matches")
        report.update(status="complete", basic_contract_passed=True, loss_finite=True, gradients_finite=True,
            finite_loss_microbatches_completed=32, old_runtime_loss_validator_passed=True,
            elapsed_seconds=time.perf_counter() - started)
        base.save_json(output / "preflight.json", report)
        base.save_json(output / "status.json", dict(status="complete", global_exposure=128, optimizer_updates=8,
                       basic_contract_passed=True, weights_saved=False))
        base.emit(report)
    except Exception as error:
        report.update(status="failed", basic_contract_passed=False, error=repr(error),
            elapsed_seconds=time.perf_counter() - started,
            observations=dict(checked.observed) if checked is not None else {},
            limited_geometry_checks=checked.audit_rows if checked is not None else [],
            finite_gradient_checks_completed=optimizer.finite_gradient_checks if optimizer is not None else 0)
        base.save_json(output / "preflight.json", report)
        base.save_json(output / "status.json", dict(status="failed", phase="preflight", error=repr(error), weights_saved=False))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="original release root; only TRAIN sources are opened")
    p.add_argument("--train-manifest", required=True, help="matched24k composite TRAIN manifest")
    p.add_argument("--cache-dir", required=True, help="fragment-only weathering cache; contains no preflight model weights")
    p.add_argument("--output", required=True, help="new preflight output directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
