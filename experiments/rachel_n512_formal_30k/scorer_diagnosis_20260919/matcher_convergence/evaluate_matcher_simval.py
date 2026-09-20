"""Prepared, NOT launched: fixed M8/M10/M12 SIMVAL matcher diagnostics.

Default is stat/manifest-only preflight. --execute is an explicit future run.
No optimizer, training, checkpoint selection, threshold fitting, REAL or OOD.
The same heatmap-gpu.lock as C16 is acquired nonblocking before model imports.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/root/autodl-tmp/rachel_score_design_20260913_001")
DIAGNOSIS = ROOT / "scorer_diagnosis_20260919"
ARMS = {
    "shared_s3_s4_s6": ROOT / "new_s345_20260914/matrix_bnfix_20260914/archive/s3_matrix/training",
    "s7": ROOT / "s6_s7_20260915/priority_after_s5/s7_augmented_full24/training",
}
EPOCHS = (8, 10, 12)
VAL_HASH = "daa6ccdd7686e93ba91ddfb1452c987145c26898a1917d2ac7d3180e199a8af8"
SCHEMA = "fixed-matcher-simval-convergence/1"
TERM_NAMES = ("total", "assignment_nll", "match_nll", "dustbin_nll",
              "translation_smooth_l1", "sinkhorn_residual")
REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_manifest(rows):
    if (len(rows) != 3000 or sum(row["label"] == 1 for row in rows) != 1500
            or any(row["label"] not in (0, 1) for row in rows)):
        raise ValueError("requires fixed balanced clean SIMVAL3000")
    ids = [row["pair_id"] for row in rows]
    if len(set(ids)) != 3000:
        raise ValueError("duplicate SIMVAL pair IDs")
    return ids


def validate_payload_metadata(payload, epoch):
    """Check lightweight metadata before invoking the existing strict loader."""
    if epoch not in EPOCHS:
        raise ValueError("only prespecified M8/M10/M12 are permitted")
    expected = {"epoch": epoch, "phase": "matcher", "completed_segments": epoch * 4,
                "global_exposure": epoch * 24000, "checkpoint_role": "epoch_anchor"}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("requires an exact committed matcher epoch anchor")
    identity = payload.get("resume_identity", {})
    if (identity.get("sampling") != "original512"
            or identity.get("base_model_config", {}).get("contour_cap") != 512
            or identity.get("populations", {}).get("val", {}).get("manifest_sha256") != VAL_HASH
            or identity.get("populations", {}).get("val", {}).get("count") != 3000):
        raise ValueError("checkpoint sampling/cap/fixed SIMVAL identity differs")
    if payload.get("decoupled_score", {}).get("phase") != "matcher":
        raise ValueError("classifier checkpoint cannot substitute for M checkpoint")
    return identity


def preflight(args):
    if args.batch_size != 1 or args.workers < 0:
        raise ValueError("fixed batch1 forward contract; workers must be nonnegative")
    manifest = args.dataset.resolve() / "pairs/val.jsonl"
    if sha256(manifest) != VAL_HASH:
        raise ValueError("fixed SIMVAL manifest hash mismatch")
    ids = validate_manifest([json.loads(line) for line in manifest.read_text().splitlines() if line])
    names = tuple(ARMS) if args.arm == "all" else (args.arm,)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("new output directory required; no overwrite or implicit resume")
    for source in (args.dataset.resolve(),) + tuple(path.resolve() for path in ARMS.values()):
        if output == source or source in output.parents:
            raise ValueError("output must not be inside source data or training runs")
    entries = []
    for arm in names:
        for epoch in EPOCHS:
            checkpoint = ARMS[arm] / ("epoch_%03d.pt" % epoch)
            info = checkpoint.stat()  # NO torch.load or checkpoint-byte read in preflight.
            entries.append(dict(arm=arm, epoch=epoch, checkpoint=str(checkpoint), bytes=info.st_size))
    return dict(schema_version=SCHEMA, status="preflight_only", checkpoint_entries=entries,
                simval_manifest=str(manifest), simval_manifest_sha256=VAL_HASH,
                sample_count=3000, positive_count=1500, negative_count=1500,
                pair_ids=ids, pair_order_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                output=str(output), gpu_lock=str(DIAGNOSIS / "heatmap-gpu.lock"),
                execute_requested=args.execute)


@contextmanager
def exclusive_gpu(lock_path):
    """Never wait for/interrupt C16, and never replace/unlink its lock inode."""
    import fcntl
    with Path(lock_path).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("C16/heatmap GPU lock is held; diagnostic not started") from error
        try:
            occupied = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            if occupied:
                raise RuntimeError("GPU has compute processes; diagnostic not started: " + occupied)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def measure_losses(output, targets, config):
    """Reuse the registered logical-micro1 terms; no new loss definition."""
    from experiments.rachel_n512_formal_30k.decoupled_samplewise_loss import samplewise_phase_terms
    # Clean SIMVAL: pose supervision exists for every positive, no damage mask.
    values = samplewise_phase_terms(output, targets, targets[4], config, "matcher")
    valid = output.training_valid
    matched = (targets[1] >= 0) & valid[:, None]
    return ({key: values[key].detach().cpu().tolist() for key in TERM_NAMES},
            matched.sum(1).detach().cpu().tolist(),
            (targets[4] & valid).detach().cpu().tolist())


def evaluate_batch(model, batch, config, device):
    import numpy as np
    import torch
    from staging.pairwise_v0_2.training import rachel_n512_runner as runner
    from staging.pairwise_v0_2.models.translation_layout import estimate_translation_layout
    from experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint import TOP2_CONFIG
    inputs, targets = runner._full_batch(batch, device)
    with torch.inference_mode():
        output = model(*inputs)  # Forward receives only masks/points/validity, never targets.
        # Freeze target-blind raw predictions before evaluating supervised losses.
        assignments = output.assignment.detach().cpu().numpy()
        layouts = [estimate_translation_layout(batch.points_rc_a[i], batch.points_rc_b[i], assignments[i],
                   batch.contour_valid_a[i], batch.contour_valid_b[i], config=TOP2_CONFIG)
                   for i in range(len(batch.pair_ids))]
        terms, match_counts, pose_flags = measure_losses(output, targets, config)
        hats = output.translation_hat_rc.detach().cpu().numpy()
    rows = []
    training_valid = output.training_valid.detach().cpu().tolist()
    decision_valid = output.decision_valid.detach().cpu().tolist()
    for i, pair_id in enumerate(batch.pair_ids):
        positive = bool(batch.labels[i])
        layout = layouts[i]
        error = float(np.linalg.norm(layout.t_a_to_b_rc - batch.translation_a_to_b_rc[i])) if positive and layout.valid else None
        hat_error = float(np.linalg.norm(hats[i] - batch.translation_a_to_b_rc[i])) if pose_flags[i] else None
        row = dict(pair_id=pair_id, label=positive, training_valid=bool(training_valid[i]),
                   decision_valid=bool(decision_valid[i]),
                   losses={key: terms[key][i] for key in TERM_NAMES},
                   supervised_correspondence_count=int(match_counts[i]), pose_supervised=bool(pose_flags[i]),
                   differentiable_translation_l2_px=hat_error,
                   raw_layout_valid=bool(layout.valid), raw_layout_reason=layout.reason,
                   raw_translation_l2_px=error,
                   raw_layout20_correct=bool(positive and layout.valid and error <= 20.0))
        if any(not math.isfinite(v) for v in row["losses"].values()):
            raise FloatingPointError("nonfinite SIMVAL loss")
        if any(v is not None and not math.isfinite(v) for v in (error, hat_error)):
            raise FloatingPointError("nonfinite SIMVAL translation error")
        rows.append(row)
    return rows


def mean_or_none(values):
    return sum(values) / len(values) if values else None


def summarize(rows):
    if not rows or len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("nonempty unique pairs required")
    groups = {"all_pairs": rows, "positive_pairs": [r for r in rows if r["label"]],
              "negative_pairs": [r for r in rows if not r["label"]]}
    result = {}
    for name, values in groups.items():
        result[name] = dict(count=len(values),
            training_valid_count=sum(r["training_valid"] for r in values),
            decision_valid_count=sum(r["decision_valid"] for r in values),
            mean_losses={key: mean_or_none([r["losses"][key] for r in values]) for key in TERM_NAMES})
    matched = [r for r in rows if r["supervised_correspondence_count"] > 0]
    pose = [r for r in rows if r["pose_supervised"]]
    positives = groups["positive_pairs"]
    correct = sum(r["raw_layout20_correct"] for r in positives)
    result["conditional_supervision"] = dict(
        correspondence_pair_count=len(matched), correspondence_token_count=sum(r["supervised_correspondence_count"] for r in matched),
        correspondence_nll_per_supervised_pair=mean_or_none([r["losses"]["match_nll"] for r in matched]),
        pose_pair_count=len(pose), translation_smooth_l1_per_supervised_pair=mean_or_none([r["losses"]["translation_smooth_l1"] for r in pose]),
        differentiable_translation_l2_px_mean=mean_or_none([r["differentiable_translation_l2_px"] for r in pose]))
    result["raw_layout20"] = dict(positive_count=len(positives), correct_count=correct,
        success_rate=correct / len(positives) if positives else None,
        decoder_valid_positive_count=sum(r["raw_layout_valid"] for r in positives),
        denominator="all SIMVAL positives, including invalid decoder or transport decisions; no classifier gate",
        tolerance_px=20.0)
    return result


def execute(args, plan):
    # Lock before importing torch or reading checkpoint bytes. No queue edits.
    with exclusive_gpu(plan["gpu_lock"]):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = "1"
        import torch
        from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
        from staging.pairwise_v0_2.training.rachel_n512_sealed_test import _set_determinism
        from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
        from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
        from experiments.rachel_n512_formal_30k.train_score_decoupled import load_decoupled_checkpoint, matcher_contract
        from experiments.rachel_n512_formal_30k.train_joint_damage import state_digest
        from experiments.rachel_n512_formal_30k.evaluate_realism_checkpoint import TOP2_CONFIG
        from experiments.rachel_n512_formal_30k import decoupled_samplewise_loss
        torch.set_num_threads(1)
        _set_determinism(260913)
        device = torch.device("cuda:0")
        dataset = RachelPairDataset(args.dataset, "val")
        if len(dataset) != 3000 or dataset.split != "val":
            raise ValueError("reader must be fixed SIMVAL3000")
        destination = Path(plan["output"])
        destination.mkdir(parents=True, exist_ok=False)
        protocol = dict(plan, status="running", source_sha256=sha256(__file__),
            loss_source_sha256=sha256(decoupled_samplewise_loss.__file__),
            decoder="full_top2_mode", decoder_config=asdict(TOP2_CONFIG),
            batch_size=1, precision="fp32", model_mode="eval+inference_mode; frozen weights",
            checkpoint_selection_performed=False, thresholds_fitted=False,
            training_performed=False, real_or_ood_read=False, classifier_metrics_reported=False,
            loss_semantics="registered logical-micro1 M loss; all-pair means retain zero terms for invalid/unsupervised samples; conditional denominators reported separately",
            clean_pose_supervision="every SIMVAL positive; no online augmentation/weathering",
            interpretation="fixed three-point held-out curve, not proof of full convergence or a checkpoint selection experiment",
            runtime=dict(torch=str(torch.__version__), cuda=str(torch.version.cuda),
                         device=torch.cuda.get_device_name(device), num_threads=torch.get_num_threads()),
            started_at_unix=time.time(), evaluations=[])
        save(destination / "protocol.json", protocol)
        arm_contracts = {}
        try:
            for entry in plan["checkpoint_entries"]:
                checkpoint = Path(entry["checkpoint"])
                digest = sha256(checkpoint)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                identity = validate_payload_metadata(payload, entry["epoch"])
                contract = matcher_contract(identity)
                if entry["arm"] in arm_contracts and arm_contracts[entry["arm"]] != contract:
                    raise ValueError("M8/M10/M12 within an arm must have identical matcher contract")
                arm_contracts[entry["arm"]] = contract
                wrapper = load_decoupled_checkpoint(payload)
                config = replace(RachelN512LossConfig(**payload["loss_config"]), validate_runtime_targets=True)
                # Evaluate only base matcher. The untrained/irrelevant classifier is never called.
                model = wrapper.base_model.eval().requires_grad_(False).to(device)
                base_before = state_digest(model)
                del payload, wrapper
                output = destination / entry["arm"] / ("m%02d" % entry["epoch"])
                output.mkdir(parents=True, exist_ok=False)
                loader = make_ablation_loader(dataset, tuple(range(3000)), batch_size=1,
                    num_workers=args.workers, seed=260913, contour_cap=512)
                rows, started = [], time.perf_counter()
                with (output / "pair_metrics.jsonl").open("x") as stream:
                    for batch in loader:
                        current = evaluate_batch(model, batch, config, device)
                        for row in current:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                        rows.extend(current)
                        if len(rows) % 250 == 0:
                            print(json.dumps(dict(arm=entry["arm"], epoch=entry["epoch"], completed=len(rows), total=3000)), flush=True)
                if [r["pair_id"] for r in rows] != plan["pair_ids"]:
                    raise ValueError("reader changed fixed SIMVAL ordered population")
                if state_digest(model) != base_before or sha256(checkpoint) != digest:
                    raise ValueError("matcher/checkpoint changed during evaluation")
                summary = dict(schema_version=SCHEMA, status="complete", **entry,
                    checkpoint_sha256=digest, matcher_state_sha256=base_before, frozen_matcher_unchanged=True,
                    fixed_simval_sha256=VAL_HASH, checkpoint_selection_performed=False,
                    source_base_loss_config=asdict(config),
                    effective_matcher_objective_weights=dict(assignment_nll=.5, translation_smooth_l1=.5,
                                                            sinkhorn_residual=.05, all_pair_bce=0.),
                    metrics=summarize(rows),
                    elapsed_seconds=time.perf_counter() - started,
                    pair_metrics_sha256=sha256(output / "pair_metrics.jsonl"))
                save(output / "summary.json", summary)
                protocol["evaluations"].append(dict(arm=entry["arm"], epoch=entry["epoch"], summary=str(output / "summary.json"),
                                                     summary_sha256=sha256(output / "summary.json")))
                save(destination / "protocol.json", protocol)
                del model
                torch.cuda.empty_cache()
            if sha256(plan["simval_manifest"]) != VAL_HASH:
                raise ValueError("SIMVAL manifest changed during evaluation")
            # All prespecified points, in epoch order; no winner or best-checkpoint field.
            curve = []
            for item in protocol["evaluations"]:
                result = json.loads(Path(item["summary"]).read_text())
                curve.append(dict(arm=item["arm"], epoch=item["epoch"], metrics=result["metrics"],
                                  source=item["summary"], source_sha256=item["summary_sha256"]))
            save(destination / "curve.json", dict(schema_version=SCHEMA, status="complete",
                fixed_simval_sha256=VAL_HASH, checkpoint_selection_performed=False, points=curve))
            protocol.update(status="complete", completed_evaluations=len(protocol["evaluations"]))
        except BaseException as error:
            protocol.update(status="failed", error=repr(error))
            raise
        finally:
            protocol.update(finished_at_unix=time.time(), elapsed_seconds=time.time() - protocol["started_at_unix"])
            save(destination / "protocol.json", protocol)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/dataset_rachel_pairwise_n512_v1"))
    value.add_argument("--arm", choices=("all",) + tuple(ARMS), default="all")
    value.add_argument("--output", type=Path, default=DIAGNOSIS / "matcher_convergence/evaluation_v1")
    value.add_argument("--batch-size", type=int, choices=(1,), default=1)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--execute", action="store_true", help="future explicit GPU evaluation; omit for read-only preflight")
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    plan = preflight(args)
    if args.execute:
        execute(args, plan)
    else:
        print(json.dumps({k: v for k, v in plan.items() if k != "pair_ids"}, indent=2))
