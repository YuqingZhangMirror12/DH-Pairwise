"""Prepare one explicitly selected candidate under the existing controls' budget.

LOCAL PREPARATION ONLY until the registered 15-arm screen is complete. This
launcher never selects an architecture or seed. A validation-only choice record
is required for execution; dry-run only prints commands and creates no files.
Released-N512 candidates use the original Full47 warm start, NOT a new control
winner. Density-rebuilt and augmentation treatments need different controls and
are intentionally outside this launcher. No scratch/convergence claim is made.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

# Both launchers are deployed as siblings OUTSIDE the immutable source tree.
# Do not assume that source008 contains the later controls-launcher module.
_control_spec = importlib.util.spec_from_file_location(
    "rachel_confirmation_controls_launcher",
    Path(__file__).with_name("launch_confirmation_controls_20260908.py"))
controls = importlib.util.module_from_spec(_control_spec)
_control_spec.loader.exec_module(controls)


SCREEN_ROOT = Path("/root/autodl-tmp/rachel_ablation_v3_20260907_001")
SCREEN_GROUPS = {
    "training_screen_attempt002": (
        "baseline_control", "seam_loss", "coarse256", "coarse512",
        "window7_only", "window7_16", "multiscale7_16_32_64"),
    "structural_training_screen": (
        "hierarchical_contour", "hierarchical_image", "multiscale_transport7_16_32_64",
        "window16_only", "window32_only", "window64_only"),
    "density_rebuilt512": ("baseline_control",),
    "density_rebuilt1024": ("baseline_control",),
}
CANDIDATES = tuple(arm for group, arms in SCREEN_GROUPS.items()
                   if not group.startswith("density_") for arm in arms if arm != "baseline_control")


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_choice(aggregate, choice, variant):
    """Check completion/declared provenance, not infer honesty from JSON flags.

    The author still must justify the choice using validation evidence. These
    gates cannot prove that previously viewed real results had no influence.
    """
    if variant not in CANDIDATES:
        raise ValueError("candidate must be a registered released-N512 architecture")
    if (aggregate.get("schema_version") != "completed-ablation-arm-aggregate/1"
            or aggregate.get("status") != "complete" or aggregate.get("issues")):
        raise ValueError("screen aggregate is not complete and consistent")
    expected = {(str(SCREEN_ROOT / group), arm) for group, arms in SCREEN_GROUPS.items() for arm in arms}
    arms = aggregate.get("arms", [])
    keys = [arm.get("key") for arm in arms]
    if (len(arms) != 15 or {(arm.get("training_root"), arm.get("arm")) for arm in arms} != expected
            or any(arm.get("status") != "complete" or arm.get("issues") for arm in arms)
            or any(not isinstance(key, str) or not key.strip() for key in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError("all 15 specific registered arms must have complete evaluations")
    if (choice.get("schema_version") != "validation-only-confirmation-choice/1"
            or choice.get("variant") != variant or choice.get("source_split") != "validation"
            or choice.get("test_or_real_used_for_selection") is not False
            or not isinstance(choice.get("rationale"), str) or not choice["rationale"].strip()):
        raise ValueError("explicit validation-only candidate choice and rationale required")
    selected = [arm for arm in arms if arm.get("arm") == variant]
    if (len(selected) != 1 or not isinstance(choice.get("discovery_arm_key"), str)
            or not choice["discovery_arm_key"].strip() or choice["discovery_arm_key"] != selected[0]["key"]):
        raise ValueError("choice must identify the exact completed discovery arm")
    return selected[0]


def train_command(seed, variant, output):
    if variant not in CANDIDATES or seed not in controls.SEEDS:
        raise ValueError("only a registered candidate and both predeclared fresh seeds are supported")
    # Inherit every baseline-control flag, changing only destination and variant.
    command = controls.train_command(seed)
    command[command.index("--variants") + 1] = variant
    command[command.index("--output") + 1] = str(Path(output) / ("seed%d" % seed) / "training")
    return command


def evaluation_commands(seed, variant, output, freeze):
    if (freeze.get("status") != "complete" or freeze.get("test_or_real_used_for_fit") is not False):
        raise ValueError("training must produce a complete validation-only freeze")
    threshold = freeze.get("classifier_thresholds", {}).get("fused")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("invalid candidate frozen pairing threshold")
    seed_root = Path(output) / ("seed%d" % seed)
    arm = seed_root / "training" / variant
    common = ["--checkpoint", str(arm / "winner.pt"), "--pair-threshold", str(threshold),
              "--precision", "fp32", "--seed", str(seed), "--batch-size", "4"]
    evaluation = seed_root / "evaluation"
    commands = []
    for split in ("val", "test", "real"):
        module = "run_real_contiguous_seam_ablation" if split == "real" else "run_contiguous_seam_ablation"
        command = [sys.executable, "-u", "-m", controls.PACKAGE + module] + common
        command += ["--output", str(evaluation / split), "--save-seam-membership"]
        if split == "real":
            command += ["--prepared-cache", controls.PREPARED, "--translation-gt-json", controls.GT]
        else:
            command += ["--split", split, "--dataset", controls.DATA, "--workers", "4", "--seam-quality"]
        if split == "val":
            command += ["--refinement-ablation"]
        else:
            command += ["--freeze", str(evaluation / "val" / "validation_freeze.json")]
        commands.append((split, command))
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=CANDIDATES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen-summary", type=Path)
    parser.add_argument("--choice", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    commands = [train_command(seed, args.variant, args.output) for seed in controls.SEEDS]
    if args.dry_run:
        print(json.dumps({"dry_run_only": True, "candidate_selected": False,
                          "execution_gate_not_evaluated": True, "training_commands": commands,
                          "after_each_training": "own VAL then frozen TEST/REAL; same controls' settings"}, indent=2))
        return
    if args.screen_summary is None or args.choice is None:
        parser.error("execution requires --screen-summary and --choice after all15 arms complete")
    aggregate, choice = _read_json(args.screen_summary), _read_json(args.choice)
    discovery = validate_choice(aggregate, choice, args.variant)
    args.output.mkdir(parents=True, exist_ok=False)
    controls.write_json(args.output / "protocol.json", {
        "schema_version": "matched-confirmation-candidate/1", "source_root": str(controls.SOURCE),
        "variant": args.variant, "seeds": controls.SEEDS, "choice": choice,
        "screen_summary": str(args.screen_summary.resolve()),
        "discovery_checkpoint_sha256": discovery.get("checkpoint_sha256"),
        "initialization": "original-Full47-warm-start", "run": controls.RUN, "dataset": controls.DATA,
        "epochs": 5, "precision": "fp32", "batch_size": 4, "effective_batch_size": 16,
        "learning_rate": 2e-5, "weight_decay": 1e-4, "clip_norm": 5,
        "control_root": str(controls.OUTPUT), "all_seeds_must_be_reported": True,
        "new_control_winner_used_as_initialization": False, "from_scratch_claim": False,
        "test_and_real_previously_viewed": True, "test_or_real_used_for_fit": False,
        "registered_15_screen_unchanged": True, "training_commands": commands,
    })
    for seed, command in zip(controls.SEEDS, commands):
        seed_root = args.output / ("seed%d" % seed)
        seed_root.mkdir()
        controls.run_stage(seed_root, "training", command)
        freeze = _read_json(seed_root / "training" / args.variant / "train_val_freeze.json")
        for split, command in evaluation_commands(seed, args.variant, args.output, freeze):
            controls.run_stage(seed_root, split, command)
    controls.write_json(args.output / "complete.json", {
        "status": "complete", "variant": args.variant, "seeds": controls.SEEDS, "completed_stages": 8})


if __name__ == "__main__":
    main()
