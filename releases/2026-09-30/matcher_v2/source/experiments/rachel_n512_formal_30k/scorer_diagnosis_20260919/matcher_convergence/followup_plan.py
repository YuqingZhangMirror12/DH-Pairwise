"""PURE future S7 M20/equal-budget Scorer plan: no I/O, execution or waiting.

stage_plan(output_root, python, source_root) returns runnable argv and receipt
contracts. The root supervisor must separately seal source, verify prerequisite
receipts/process exit, reserve a new output root, and dispatch serially. This
module neither certifies those prerequisites nor changes the priority queue.

CLI prints a proposed JSON plan to stdout only; there is no execute flag.
"""
from __future__ import annotations

import argparse
import json
from pathlib import PurePosixPath as Path

SCHEMA = "s7-matcher-followup-pure-plan/1"
R = Path("/root/autodl-tmp/rachel_score_design_20260913_001")
D = R / "scorer_diagnosis_20260919"
DEFAULT_SOURCE = D / "matcher_followup_source_v1"
DEFAULT_OUTPUT = D / "matcher_followup_v1"
PYTHON = "/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python"
PACKAGE = "experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matcher_convergence."
M12 = R / "s6_s7_20260915/priority_after_s5/s7_augmented_full24/training/epoch_012.pt"
M12_SHA256 = "d8a93af1eb5f3b02baaf7d42b9d8675242a446a1b11e43cde0561ba89e670e07"
VAL_SHA256 = "daa6ccdd7686e93ba91ddfb1452c987145c26898a1917d2ac7d3180e199a8af8"
PRIORITY = D / "priority_s7_matched_g_v1"
BASELINE = PRIORITY / "training/all_tokens"
HARD_ROOT = D / "hard_validation_m12_v1"
HARD_MANIFEST = HARD_ROOT / "data/manifest.json"
DATASET = Path("/root/autodl-tmp/dataset_rachel_pairwise_n512_v1")
REAL = Path("/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared")
OOD = Path("/root/autodl-tmp/turufan_ood_pairwise_20260912_001/prepared")
REAL_GT = Path("/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json")
COUNTS = dict(test=3000, real=1016, ood=301)


def _absolute(value):
    path = Path(str(value))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute normalized remote paths required")
    return path


def _receipt(path, **expect):
    return dict(path=str(path), expect=expect)


def prerequisite_contracts():
    """Expected future receipts, NOT observations of their present status."""
    return dict(
        priority=_receipt(PRIORITY / "status.json", status="complete", completed_stages=60),
        predecessor_process_rule="Supervisor must bind PID/startticks and require priority dispatcher and its children exited; no signaling or reprioritization.",
        immutable_matcher=dict(path=str(M12), sha256=M12_SHA256, epoch=12, phase="matcher", completed_segments=48),
        hard_data=_receipt(HARD_MANIFEST, schema="s7-hard-simval-diagnostic/1", status="complete", split="val",
            protocol=dict(source_val_manifest_sha256=VAL_SHA256, diagnostic_only=True),
            summary=dict(count=6000, source_count=3000)),
        hard_m12=_receipt(HARD_ROOT / "m12/summary.json", status="complete", count=6000,
            model=dict(epoch=12, checkpoint_sha256=M12_SHA256)),
        baseline_status=_receipt(BASELINE / "status.json", status="complete", completed_segments=64,
            completed_head_epochs=16, classifier_pair_exposures=384000, optimizer_updates=24000,
            matcher_updated=False, real_ood_used=False),
        baseline_freeze=_receipt(BASELINE / "freezes/c16.json", status="complete_endpoint", budget_head_epochs=16,
            primary_selection="fixed_endpoint", selection_population="clean SIMVAL3000 only", real_ood_used=False,
            identity=dict(arm="all_tokens", source_checkpoint_sha256=M12_SHA256,
                          head_seed=260914, data_seed=260913)),
        baseline_evaluations=[_receipt(BASELINE / "evaluation/c16" / split / "protocol.json",
            status="complete", split=split, sample_count=count,
            model=dict(head_epoch=16, head_budget=16, selection="fixed_epoch"))
            for split, count in COUNTS.items()],
        source_compatibility="Seal a new full source snapshot. Its matched_only implementation must match the completed M12 baseline's identity/implementation_sha256; do not edit or repoint live snapshots.",
        status="requirements_not_checked_by_pure_builder")


def required_receipts():
    """Passive prerequisites for the root supervisor, no filesystem reads."""
    contracts = prerequisite_contracts()
    return [contracts[key] for key in ("hard_data", "hard_m12", "baseline_status", "baseline_freeze")] + contracts["baseline_evaluations"]


def stage_plan(output_root=DEFAULT_OUTPUT, python=PYTHON, source_root=DEFAULT_SOURCE):
    """Ordered 18-stage plan with explicit full child environments.

    completion_expect is a recursive JSON-subset contract. Additional reports
    and checkpoint_contracts describe durable evidence for the supervisor; they
    are not read or asserted by this pure builder. Every output is a fresh run.
    """
    root, source, python = _absolute(output_root), _absolute(source_root), str(_absolute(python))
    for protected in (source, PRIORITY, HARD_ROOT, M12.parent, DATASET, REAL, OOD):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError("new output must be separate from source, inputs and earlier runs")
    rows = []
    def add(name, kind, module, arguments, output, completion, expect, **extra):
        env = dict(PATH="/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                   HOME="/root", LANG="C.UTF-8", PYTHONPATH=str(source), PYTHONUNBUFFERED="1",
                   CUDA_VISIBLE_DEVICES="" if kind == "cpu" else "0",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        row = dict(name=name, kind=kind, command=[python, "-m", PACKAGE+module, *map(str, arguments)],
            cwd=str(source), env=env, env_mode="complete_explicit_environment",
            output=str(output), output_policy="new_directory_no_implicit_resume",
            completion=str(output/completion), completion_expect=expect,
            depends_on=[rows[-1]["name"]] if rows else ["external_prerequisites_verified"], **extra)
        rows.append(row)
    smoke = root / "smokes/matcher_discard32"
    add("S7_M12_matcher_discard32", "gpu", "train_continuation",
        ["--source", M12, "--output", smoke, "--device", "cuda:0", "--stop-after-epoch", 20, "--smoke", 32],
        smoke, "smoke.json", dict(status="smoke_complete", discarded_pair_exposures=32,
            discarded_optimizer_updates=2, exact_gpu_rng_restored=True, formal_training_counted=False,
            weights_and_optimizer_discarded=True, no_checkpoint_written=True, frozen_branches_unchanged=True))
    matcher = root / "matcher_training"
    add("S7_M13_M20_train", "gpu", "train_continuation",
        ["--source", M12, "--output", matcher, "--device", "cuda:0", "--stop-after-epoch", 20],
        matcher, "status.json", dict(status="training_complete", phase="matcher", epoch=20,
            completed_segments=80, last_committed_segment=80, global_exposure=480000,
            optimizer_updates=30000, continuation_pair_exposures=192000, continuation_optimizer_updates=12000,
            formal_training_counted=True, fixed_epoch_anchors_ready=[16,20],
            ready_for_fixed_matcher_evaluation=True, full_experiment_complete=False),
        checkpoint_contracts=[dict(path=str(matcher/('epoch_%03d.pt' % epoch)),
            loader=PACKAGE+"evaluate_continuation.load_endpoint", epoch=epoch, phase="matcher",
            completed_segments=epoch*4, checkpoint_role="epoch_anchor", rng_mode="exact_gpu",
            formal_training_counted=True) for epoch in (16,20)])
    for epoch in (16,20):
        arm = root / ("m%d" % epoch)
        checkpoint = matcher / ("epoch_%03d.pt" % epoch)
        binding = ["--matcher-checkpoint", checkpoint, "--matcher-epoch", epoch]
        out = arm / "hard_validation"
        add("M%d_hard_SIMVAL6000" % epoch, "cpu", "evaluate_hard_validation",
            ["--manifest", HARD_MANIFEST, "--source-val-manifest", DATASET/"pairs/val.jsonl",
             "--checkpoint", checkpoint, "--epoch", epoch, "--workers", 4, "--output", out, "--execute"],
            out, "summary.json", dict(status="complete", count=6000, diagnostic_only=True,
                classifier_metrics_reported=False, model=dict(epoch=epoch, current_matcher_epochs=epoch),
                metrics=dict(by_recipe=dict(clean=dict(all_pairs=dict(count=3000))))),
            reports=[_receipt(out/"protocol.json", status="complete", completed_count=6000)],
            pair_metrics=dict(path=str(out/"pair_metrics.jsonl"), expected_rows=6000),
            includes_clean_simval=True, duplicate_clean_gpu_evaluation=False)
        for split, count in (("train",24000),("val",3000)):
            out = arm / "cache" / split
            add("M%d_%s_cache" % (epoch,split), "cpu", "scorer_bridge",
                ["cache", *binding, "--split", split, "--workers", 4, "--output", out],
                out, "protocol.json", dict(status="complete", split=split, pair_count=count,
                    expected_full_count=count, completed_pairs=count, formal_training_eligible=True,
                    precompute_device="cpu", matcher_frozen=True,
                    matcher_endpoint=dict(matcher_epoch=epoch, matcher_checkpoint=str(checkpoint))),
                records=dict(path=str(out/"pairs.json"), expected_rows=count))
        head = arm / "training/all_tokens"
        train_args = ["train", *binding, "--arm", "all_tokens", "--train-cache", arm/"cache/train",
            "--val-cache", arm/"cache/val", "--device", "cuda:0", "--stop-after-head-epoch", 16]
        out = root / "smokes" / ("m%d_all_tokens" % epoch)
        add("M%d_all_tokens_discard32" % epoch, "gpu", "scorer_bridge",
            [*train_args, "--output", out, "--smoke", 32], out, "smoke.json",
            dict(status="smoke_complete", discarded_pairs=32, weights_discarded=True,
                 checkpoint_written=False, formal_training_counted=False,
                 training=dict(samples=32, optimizer_updates=2)))
        add("M%d_all_tokens_C16_train" % epoch, "gpu", "scorer_bridge",
            [*train_args, "--output", head], head, "status.json",
            dict(status="complete", completed_segments=64, completed_head_epochs=16,
                 absolute_epoch=epoch+16, matcher_epoch=epoch, head_data_shuffle_epoch=28,
                 classifier_pair_exposures=384000, optimizer_updates=24000, matcher_updated=False, real_ood_used=False),
            reports=[_receipt(head/"freezes/c16.json", status="complete_endpoint", budget_head_epochs=16,
                primary_selection="fixed_endpoint", real_ood_used=False,
                selections=dict(fixed_endpoint=dict(head_epoch=16, absolute_epoch=epoch+16)))],
            checkpoint_contracts=[dict(path=str(head/"head_epoch_016.pt"), completed_segments=64,
                head_epoch=16, matcher_epoch=epoch, absolute_epoch=epoch+16, head_data_shuffle_epoch=28)])
        for split, count in COUNTS.items():
            out = head / "evaluation/c16" / split
            arguments = ["evaluate", *binding, "--training-run", head, "--head-budget", 16,
                "--selection", "fixed_epoch", "--split", split, "--output", out,
                "--device", "cuda:0", "--batch-size", 1, "--workers", 4,
                "--dataset", DATASET, "--prepared-cache", REAL, "--ood-prepared", OOD,
                "--translation-gt-json", REAL_GT]
            if split == "real":
                arguments += ["--keep-ids", R/"keep_ids.json"]
            add("M%d_all_tokens_C16_%s" % (epoch,split), "gpu", "scorer_bridge", arguments,
                out, "protocol.json", dict(status="complete", split=split, sample_count=count,
                    thresholds_fitted=False, model=dict(matcher_epoch=epoch, head_epoch=16,
                        head_budget=16, epoch=epoch+16, selection="fixed_epoch")),
                reports=[_receipt(out/"summary.json", status="complete", split=split,
                    selection_on_this_population=False, threshold_fitting_performed=False)],
                pair_results=dict(path=str(out/"pair_results.jsonl"), expected_rows=count),
                reused_m12_baseline=str(BASELINE/"evaluation/c16"/split),
                operating_points="both frozen clean SIMVAL maxF1 and recall95; no extra held-out sweep",
                interpretation="positive-only Recall; no accuracy/F1/layout GT" if split == "ood" else
                    "Dunhuang1016 raw, kept295positive+508negative primary" if split == "real" else "balanced SIMTEST3000")
    return rows


def build_plan(output_root=DEFAULT_OUTPUT, python=PYTHON, source_root=DEFAULT_SOURCE):
    stages = stage_plan(output_root, python, source_root)
    return dict(schema=SCHEMA, status="proposed_not_executed", output_root=str(output_root),
        source_root=str(source_root), prerequisites=prerequisite_contracts(), stages=stages,
        stage_count=len(stages), fixed_matcher_endpoints=[16,20], primary_head_budget=16,
        reused_m12_c16_baseline=str(BASELINE), m12_retrained=False, c8_evaluations_added=False,
        supervisor_policy="Run only after all60 priority stages and true process exit; seal/verify code at start and dispatch, not in child loops; no automatic retry/resume or outer GPU lock.",
        scientific_contract=dict(head="same fresh all_tokens D2/4heads", head_seed=260914, data_seed=260913,
            head_data_shuffle_epochs=[13,28], train_pairs=24000, simval_pairs=3000, head_epochs=16,
            ordinary_physical_batch=16, effective_batch=16, head_exposures=384000, head_updates=24000,
            head_lr_first3=1e-4, head_lr_after3=2e-5, weight_decay=1e-4, grad_clip=5., precision="fp32",
            matcher_fixed_lr=2e-5, matcher_added_epochs=8, matcher_added_exposures=192000,
            matcher_added_updates=12000),
        limitations=["A pure plan is not evidence of readiness, launch, source sealing or completion.",
            "Hard-VAL includes exactly the clean3000 references; no duplicate GPU clean evaluation is planned.",
            "M16/M20 need their own features and validity caches; M12 caches cannot substitute.",
            "Source compatibility with the already-trained M12 baseline must be checked before dispatch.",
            "Fixed endpoints and equal head budgets do not establish convergence or REAL/OOD improvement."])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    p.add_argument("--python", default=PYTHON)
    p.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(build_plan(args.output_root, args.python, args.source_root), indent=2))
