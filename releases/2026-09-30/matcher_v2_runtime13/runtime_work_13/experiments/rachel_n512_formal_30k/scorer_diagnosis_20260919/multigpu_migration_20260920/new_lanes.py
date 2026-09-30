"""Derive independent leaf lanes from the two registered plans, never launch.

Only command/cwd/env/output/receipt contracts are inherited. Old supervisors,
dead PID dependencies, aggregate summaries and queue-wide60-stage barriers are
not executable migration stages. The root runner owns device binding and waits.
CLI reads two local plan JSONs and prints the derived lanes; no output writes.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path, PurePosixPath

SCHEMA = "s7-independent-leaf-lanes/1"
DIRECT_ARMS = ("all_tokens", "matched_tokens", "edge_seed", "edge_multi", "matched_edges")
SPLITS = ("test", "real", "ood")
M12_SHA256 = "d8a93af1eb5f3b02baaf7d42b9d8675242a446a1b11e43cde0561ba89e670e07"
TRAIN_SHA256 = "79a9e959f32ef9899116e299425d6350b17f6a04bd5070b5aa9a730319447c36"
VAL_SHA256 = "daa6ccdd7686e93ba91ddfb1452c987145c26898a1917d2ac7d3180e199a8af8"
LEAF_FIELDS = ("name", "command", "cwd", "env", "completion", "completion_expect", "output",
               "output_policy", "checkpoint_contracts", "pair_metrics", "pair_results", "records",
               "operating_points", "interpretation", "reused_m12_baseline")


def _flag(command, name, default=None):
    return command[command.index(name)+1] if name in command else default


def _leaf(stage, *, gpu=True):
    missing = [k for k in ("name", "command", "cwd", "env", "completion", "completion_expect") if k not in stage]
    if missing:
        raise ValueError("leaf contract missing " + ",".join(missing))
    command = stage["command"]
    if (not isinstance(command, list) or len(command) < 3 or command[1] != "-m"
            or any(s in command[2] for s in ("queue", "supervisor", "after_priority", "after_dependencies",
                "legacy_tail", "prepare_priority_plan", "summarize_priority"))):
        raise ValueError("migration must invoke only registered leaf modules")
    if any(not PurePosixPath(stage[k]).is_absolute() for k in ("cwd", "completion")):
        raise ValueError("leaf paths must be absolute")
    row = {k: deepcopy(stage[k]) for k in LEAF_FIELDS if k in stage}
    row.update(gpu=bool(gpu), kind="gpu" if gpu else "cpu", prerequisites=[],
               origin_stage_name=stage["name"], mutation_policy="no implicit resume or source/output rewriting")
    if stage.get("reports"):
        row["additional_completions"] = deepcopy(stage["reports"])
    return row


def _receipt(path, expect, producer_lane=None):
    result = dict(path=str(path), expect=deepcopy(expect))
    if producer_lane:
        result["producer_lane"] = producer_lane
    return result


def _feature_requirements(stage):
    result = []
    for flag, split in (("--train-cache", "train"), ("--val-cache", "val"),
                        ("--stage-cache", "train"), ("--train-stage-cache", "train"),
                        ("--val-stage-cache", "val")):
        root = _flag(stage["command"], flag)
        if root is not None:
            result.append(_receipt(PurePosixPath(root)/"protocol.json",
                dict(status="complete", split=split, completed_pairs=24000 if split == "train" else 3000)))
    return result


def _expected_priority_names(arms):
    names = [arm+"_discard32" for arm in arms]
    for arm in arms:
        names.append(arm+"_C16_train")
        names.extend("%s_C%d_%s" % (arm, budget, split) for budget in (16,8) for split in SPLITS)
    return names


def _priority_lane(priority, arms, name):
    expected = _expected_priority_names(arms)
    selected = [s for s in priority["stages"] if s["name"] in set(expected)]
    if [s["name"] for s in selected] != expected:
        raise ValueError("registered arm/order/budget stages differ for " + name)
    result = []
    for source in selected:
        stage = _leaf(source)
        command = stage["command"]
        arm = _flag(command, "--arm")
        if arm is not None and arm not in arms:
            raise ValueError("arm metadata differs from registered leaf")
        if "_C16_train" in stage["name"] and _flag(command, "--stop-after-head-epoch", "16") != "16":
            raise ValueError("migration cannot shorten the registered C16 budget")
        if "--head-budget" in command and (_flag(command, "--head-budget") not in ("8", "16")
                or _flag(command, "--selection", "fixed_epoch") != "fixed_epoch"):
            raise ValueError("only existing fixed C8/C16 endpoint selections")
        stage["prerequisites"] = _feature_requirements(stage)
        result.append(stage)
    return dict(name=name, schema=SCHEMA, stages=result, stage_count=len(result),
        arms=list(arms), budget="fresh C16/64segments; fixed C16 and C8 endpoints, no new selection sweep",
        source_contract=dict(frozen_matcher_epoch=12, frozen_matcher_sha256=M12_SHA256,
            train_manifest_sha256=TRAIN_SHA256, val_manifest_sha256=VAL_SHA256,
            authority="original leaf source/cache validators; this builder does not read weights"))


def _matcher_requirements(matcher):
    receipts = matcher.get("required_receipts")
    if receipts is None:
        p = matcher.get("prerequisites", {})
        receipts = [p[k] for k in ("hard_data", "hard_m12", "baseline_status", "baseline_freeze") if k in p]
        receipts += p.get("baseline_evaluations", [])
    if not isinstance(receipts, list):
        raise ValueError("registered Matcher prerequisite receipts required")
    return receipts


def _matcher_lane(matcher, lane1):
    registered = matcher["stages"]
    expected = ["S7_M12_matcher_discard32", "S7_M13_M20_train"]
    for epoch in (16,20):
        expected.extend(["M%d_hard_SIMVAL6000" % epoch, "M%d_train_cache" % epoch,
            "M%d_val_cache" % epoch, "M%d_all_tokens_discard32" % epoch,
            "M%d_all_tokens_C16_train" % epoch])
        expected.extend("M%d_all_tokens_C16_%s" % (epoch,split) for split in SPLITS)
    if [s["name"] for s in registered] != expected:
        raise ValueError("requires the registered18-stage finite Matcher+Scorer plan")
    receipts = _matcher_requirements(matcher)
    baseline_stage = next(s for s in lane1["stages"] if s["name"] == "all_tokens_C16_train")
    baseline = PurePosixPath(baseline_stage["completion"]).parent
    baseline_status = _receipt(baseline_stage["completion"], baseline_stage["completion_expect"], "lane1")
    freeze_path = str(baseline/"freezes/c16.json")
    freeze = next((r for r in receipts if r["path"] == freeze_path), None)
    if freeze is None or freeze["expect"].get("identity", {}).get("source_checkpoint_sha256") != M12_SHA256:
        raise ValueError("M12 baseline freeze identity/path differs from lane1")
    baseline_freeze = _receipt(freeze["path"], freeze["expect"], "lane1")
    result = []
    for original in registered:
        stage = _leaf(original, gpu=original.get("kind") != "cpu")
        command = stage["command"]
        if "--resume" in command or "--limit" in command or "--allow-pilot" in command:
            raise ValueError("registered formal lanes cannot silently resume or use limited/pilot inputs")
        if "hard_SIMVAL6000" in stage["name"]:
            manifest = _flag(command, "--manifest")
            data = next((r for r in receipts if r["path"] == manifest), None)
            m12 = [r for r in receipts if r["path"].endswith("/m12/summary.json")]
            if data is None or len(m12) != 1:
                raise ValueError("hard-VAL source/comparator receipts missing")
            stage["prerequisites"] = [deepcopy(data), deepcopy(m12[0])]
        if "--head-budget" in command:
            if _flag(command, "--head-budget") != "16" or _flag(command, "--selection") != "fixed_epoch":
                raise ValueError("Matcher comparison only adds fixed C16 endpoints")
            split = _flag(command, "--split")
            reference = next(s for s in lane1["stages"] if s["name"] == "all_tokens_C16_"+split)
            if stage.get("reused_m12_baseline") != str(PurePosixPath(reference["completion"]).parent):
                raise ValueError("Matcher endpoint refers to a different baseline output")
            stage["prerequisites"] = [deepcopy(baseline_status), deepcopy(baseline_freeze),
                _receipt(reference["completion"], reference["completion_expect"], "lane1")]
        result.append(stage)
    return dict(name="lane6", schema=SCHEMA, stages=result, stage_count=len(result),
        arms=["S7_M13_M20", "M16_all_tokens_C16", "M20_all_tokens_C16"],
        immediate_matcher_training=True, m12_retrained=False,
        baseline_wait="only lane1 all_tokens C16 status/freeze/corresponding endpoint; never lane1 completion or old60-stage PID",
        source_contract=dict(source_matcher_sha256=M12_SHA256, train_manifest_sha256=TRAIN_SHA256,
            val_manifest_sha256=VAL_SHA256, native_matcher_anchors=[16,20],
            own_cache_required=True, source_and_output_paths_unchanged=True))


def derive_lanes(priority_plan, matcher_plan):
    """Return lane1/2/3/6 dictionaries compatible with lane_runner.py.

    GPU UUID assignment belongs to the root; these definitions contain no PID,
    host connection, GPU selection, file writes or subprocess execution.
    """
    if len({s["name"] for s in priority_plan["stages"]}) != len(priority_plan["stages"]):
        raise ValueError("duplicate registered priority stages")
    lanes = dict(lane1=_priority_lane(priority_plan, DIRECT_ARMS, "lane1"),
                 lane2=_priority_lane(priority_plan, ("G0",), "lane2"),
                 lane3=_priority_lane(priority_plan, ("G1",), "lane3"))
    lanes["lane6"] = _matcher_lane(matcher_plan, lanes["lane1"])
    completions = [s["completion"] for lane in lanes.values() for s in lane["stages"]]
    if len(completions) != len(set(completions)):
        raise ValueError("independent lanes collide on output/completion paths")
    return lanes


def architecture_summary():
    """Code-backed interpretation notes, not an assertion of finished results."""
    return dict(
        direct5=dict(common="Frozen S7 M12 post-context96D tokens; fresh D2/4head CA; original PairBCE*training_valid averaged over every ordinary16 row. No global-score residual/rescue.",
            all_tokens="all valid contour tokens from both fragments",
            matched_tokens="unique endpoints of production final best-mode inlier edges; explicit edge pairing is not retained",
            matched_edges="final best-mode edges: Wself*fAi+Wmate*fBj+Wmeta[Qij,residual/10px], paired features retained before shared CA",
            edge_seed="same edge head on best-support raw seed before refinement",
            edge_multi="same shared edge head on up to5 separate refined modes; max group logit gets PairBCE, not union of modes",
            fallback="learnable no-evidence logit; no forced-negative and no global rescue",
            limits="selected tokens already contain frozen global context; edges add parameters/Q/residual versus token-only arms"),
        G0_G1=dict(input="six raw mask/contour/valid tensors; recompute independent copied S7 patch sampler/encoder/scale_gate/context stem, no cached features",
            head="fresh D2/4head CA -> full L2-cosine grid -> tau15 log-mean-exp -> positive affine logit",
            G0="copied stem frozen", G1="copied stem trainable at0.1*headLR; original Matcher remains frozen in both",
            loss="ordinary16 mean masked PairBCE, plus0.3*relu(0.15-raw_positive+max confirmed same-anchor raw_negative); no auxiliary BCE",
            budget="1500 updates/epoch,1470 known-anchor groups on1470 slots,30 no-aux;27028 pair-forwards/epoch,not an effective16 total-forward claim",
            limits="full cosine grid is not Matcher-selected correspondences; colleague-inspired, not a faithful reproduction"),
        Matcher=dict(input="same original S7 materialized TRAIN24K masks, contours, inherited correspondence/dustbin targets, explicit damaged-pose eligibility",
            loss="0.5 assignmentNLL +0.5 eligible translationSmoothL1 +0.05 Sinkhorn residual; no PairBCE or Scorer updates",
            budget="exact M12 optimizer/RNG continuation M13..M20 atLR2e-5; preserveM16/M20; +192K pairs/+12K updates physical/effective16",
            scorer_control="eachM16/M20 own CPU TRAIN24K/VAL3K features, fresh all_tokens D2 C16 same seed/budget; reuse lane1 M12C16 only",
            limits="C16 heads are required to compare classification; Matcher diagnostics alone do not establish classifier improvement"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--priority-plan", type=Path, required=True)
    p.add_argument("--matcher-plan", type=Path, required=True)
    args = p.parse_args()
    priority = json.loads(args.priority_plan.read_text())
    matcher = json.loads(args.matcher_plan.read_text())
    print(json.dumps(derive_lanes(priority, matcher), indent=2))


if __name__ == "__main__":
    main()
