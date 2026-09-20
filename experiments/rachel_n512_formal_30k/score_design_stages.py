"""Isolated S3 phase/freezing/loss helpers; NOT wired into the live trainer.

Registered comparison: staged M12+C8 versus joint20, each480000 TRAIN pairs.
M trains matcher/coarse, with fused/local pair and R losses disabled.
C freezes matcher/coarse including running state; trains new P/R and fusion,
with correspondence/shift/OT losses disabled. Coarse BCE is retained at its
original weight in C but is constant with respect to trainable parameters.
This changes per-phase objectives AND update allocation, not merely data order.

Integration contract:
1. Create one AdamW over optimizer_parameter_groups(model) BEFORE phase M.
   Both lifetime parameter groups are registered; never reset AdamW at M->C.
2. Apply configure_phase only at an optimizer boundary; use forward_for_phase
   rather than a raw model forward (M deliberately bypasses the candidate head).
3. After exactly12 M epochs, create/save a matcher_pretraining_receipt together
   with model/optimizer/RNG. C requires that receipt and its matching state hash.
4. Run compute_phase_loss only on TRAIN targets. Validation uses target-blind
   pair inference/metrics, never candidate correctness labels to select layout.
5. Persist phase protocol + receipt in the new future trainer's resume identity.
   The existing live train_score_design.py and queue are intentionally untouched.
6. Primary S3 comparison is the paired fixed epoch20 checkpoint. Auxiliary
   max-F1 and precision@95%-recall winners are selected only from epochs13..20
   in BOTH schedules. Never substitute a live S0-S2 winner from epochs5..20.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Dict

import torch
from torch import Tensor

from staging.pairwise_v0_2.models.rachel_candidate_score import (
    RachelCandidateScore, candidate_correctness_loss)
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig, RachelN512Loss
from staging.pairwise_v0_2.training.rachel_weathering_training import compute_weathering_loss

SCHEMA = "rachel-score-design-stages/1"
EVALUATION_SCHEMA = "rachel-score-design-s3-comparison/1"
TOTAL_EPOCHS, MATCHER_EPOCHS, CLASSIFIER_EPOCHS = 20, 12, 8
TRAIN_PAIRS, EFFECTIVE_BATCH = 24000, 16
R_WEIGHT, R_TOLERANCE_PX = 0.5, 20.0
WEIGHT_FIELDS = ("fused_pair_weight", "coarse_pair_weight", "local_pair_weight",
                 "assignment_weight", "translation_weight", "sinkhorn_residual_weight")


@dataclass(frozen=True)
class ScorePhase:
    epoch: int
    budget: int
    schedule: str
    architecture: str
    name: str
    start_epoch: int
    end_epoch: int


@dataclass(frozen=True)
class PhaseLoss:
    total: Tensor
    base_loss: RachelN512Loss
    candidate_correctness_bce: Tensor
    config: RachelN512LossConfig
    profile: Dict
    counts: Dict[str, Tensor]


def phase(epoch: int, budget: int = TOTAL_EPOCHS, *, schedule="staged",
          architecture="candidate_pair") -> ScorePhase:
    if type(budget) is not int or budget != TOTAL_EPOCHS:
        raise ValueError("S3 is registered at exactly20 total epochs")
    if type(epoch) is not int or not 1 <= epoch <= budget:
        raise ValueError("epoch must be in1..20")
    if schedule not in ("staged", "joint") or architecture not in ("candidate_pair", "candidate_dual"):
        raise ValueError("requires staged/joint and a registered candidate architecture")
    if schedule == "joint":
        name, start, end = "J", 1, TOTAL_EPOCHS
    elif epoch <= MATCHER_EPOCHS:
        name, start, end = "M", 1, MATCHER_EPOCHS
    else:
        name, start, end = "C", MATCHER_EPOCHS + 1, TOTAL_EPOCHS
    return ScorePhase(epoch, budget, schedule, architecture, name, start, end)


def phase_loss_profile(spec: ScorePhase, base=RachelN512LossConfig()):
    """Exact registered changes, including constant-but-retained coarse BCE."""
    overrides = {}
    if spec.name == "M":
        overrides = dict(fused_pair_weight=0.0, local_pair_weight=0.0)
    elif spec.name == "C":
        overrides = dict(assignment_weight=0.0, translation_weight=0.0, sinkhorn_residual_weight=0.0)
    elif spec.name != "J":
        raise ValueError("unknown phase")
    config = replace(base, **overrides)
    r_weight = R_WEIGHT if spec.architecture == "candidate_dual" and spec.name != "M" else 0.0
    profile = dict(schema_version=SCHEMA, phase=asdict(spec),
        original_weights={name: getattr(base, name) for name in WEIGHT_FIELDS},
        effective_weights={name: getattr(config, name) for name in WEIGHT_FIELDS},
        disabled_by_phase=list(overrides), candidate_correctness_weight=r_weight,
        candidate_correctness_tolerance_px=R_TOLERANCE_PX,
        candidate_correctness_reduction="mean valid candidates per pair; then mean supervised pairs",
        coarse_pair_bce_is_constant=spec.name == "C",
        old_local_head_always_frozen=True,
        shift_gate="original clean-positive E1 gate; no changed-positive shift auxiliary",
        r_gate="positive GT availability, INCLUDING changed/corroded positives; valid negative candidates are R0",
        pair_gt_never_replaced_by_candidate_gt=True)
    return config, profile


def stage_protocol(architecture="candidate_pair", *, schedule="staged", base=RachelN512LossConfig()):
    specs = [phase(e, schedule=schedule, architecture=architecture) for e in range(1, TOTAL_EPOCHS + 1)]
    epochs = {name: sum(s.name == name for s in specs) for name in ("M", "C", "J")}
    return dict(schema_version=SCHEMA, architecture=architecture, schedule=schedule,
        total_epochs=TOTAL_EPOCHS, unique_train_pairs=TRAIN_PAIRS,
        total_pair_exposures=TOTAL_EPOCHS * TRAIN_PAIRS,
        total_optimizer_updates=TOTAL_EPOCHS * TRAIN_PAIRS // EFFECTIVE_BATCH,
        phase_epoch_counts=epochs,
        matcher_update_exposures=TRAIN_PAIRS * (epochs["M"] + epochs["J"]),
        classifier_update_exposures=TRAIN_PAIRS * (epochs["C"] + epochs["J"]),
        phase_profiles=[phase_loss_profile(s, base)[1] for s in specs],
        evaluation=dict(schema_version=EVALUATION_SCHEMA,
            primary=dict(selection="fixed_epoch", epoch=TOTAL_EPOCHS,
                         checkpoint_role="paired_epoch20_anchor_not_a_VAL_winner"),
            auxiliary=dict(eligible_epochs=list(range(MATCHER_EPOCHS + 1, TOTAL_EPOCHS + 1)),
                selection_rules={"max_f1": "VAL fused F1, then AP, then earliest eligible epoch",
                                 "recall95": "VAL precision at empirical95% recall, then AP, then earliest eligible epoch"},
                identical_window_for_staged_and_joint=True),
            calibration_population="cleanVAL3000 only, on each selected checkpoint",
            held_out_used_for_model_or_threshold_selection=False,
            live_s0_s2_budget_freeze_compatible=False,
            live_winner_from_epochs5_through20_allowed=False,
            dedicated_freeze_namespace="s3_freezes",
            requires_dedicated_evaluator_schema=True),
        optimizer_reinitialized_at_transition=False, learning_rate_restarted_at_transition=False,
        initial_base_and_head_must_match_joint=True, held_out_targets_used_for_training=False,
        interpretation="equal total exposure; different phase objectives, frozen modules and per-module update budgets")


def _check_model(model, spec=None):
    if not isinstance(model, RachelCandidateScore):
        raise TypeError("S3 requires the candidate wrapper, not a random/frozen original-only model")
    if spec is not None and model.architecture != spec.architecture:
        raise ValueError("phase architecture differs from model")


def _family(name):
    if name.startswith("score_head.") or name.startswith("base_model.fusion."):
        return "classifier"
    if name.startswith("base_model.local_head."):
        return "unused_old_local"
    if name.startswith("base_model."):
        return "matcher"
    raise ValueError("unregistered parameter family: " + name)


def optimizer_parameter_groups(model):
    """Register both lifetime groups even while classifier requires_grad=False.

    AdamW skips parameters with grad=None. This preserves matcher moments and
    lets initially frozen classifier parameters acquire state only in C.
    """
    _check_model(model)
    groups = {name: [] for name in ("matcher", "classifier")}
    for name, parameter in model.named_parameters():
        family = _family(name)
        if family in groups:
            groups[family].append(parameter)
    return [dict(params=parameters, phase_family=name) for name, parameters in groups.items()]


def matcher_state_digest(model):
    """Parameters AND buffers, excluding both old and new classification heads."""
    _check_model(model)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if _family(name) != "matcher":
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _require_hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(name + " must be an explicit SHA256")


def matcher_pretraining_receipt(model, *, initial_matcher_sha256, completed_epochs,
                                pair_exposures, optimizer_updates, train_manifest_sha256,
                                run_identity_sha256, split="train"):
    """Create only at completed M12; this is consistency evidence, not attestation.

    Counters must come from the trainer's committed progress, not be invented by
    the caller. Hash/counter checks prevent accidental random-matcher head-only
    training; they are not cryptographic proof that an untrusted run trained.
    """
    _check_model(model)
    if split != "train" or (completed_epochs, pair_exposures, optimizer_updates) != (
            MATCHER_EPOCHS, MATCHER_EPOCHS * TRAIN_PAIRS, MATCHER_EPOCHS * TRAIN_PAIRS // EFFECTIVE_BATCH):
        raise ValueError("C requires exactly12 completed TRAIN matcher epochs within the20-epoch budget")
    for name, value in (("initial_matcher_sha256", initial_matcher_sha256),
                        ("train_manifest_sha256", train_manifest_sha256), ("run_identity_sha256", run_identity_sha256)):
        _require_hash(value, name)
    trained = matcher_state_digest(model)
    if trained == initial_matcher_sha256:
        raise ValueError("matcher is unchanged from initialization; cannot freeze random matcher for C")
    return dict(schema_version=SCHEMA, architecture=model.architecture, split="train",
        completed_phase="M", completed_epochs=MATCHER_EPOCHS, pair_exposures=pair_exposures,
        optimizer_updates=optimizer_updates, initial_matcher_sha256=initial_matcher_sha256,
        trained_matcher_sha256=trained, train_manifest_sha256=train_manifest_sha256,
        run_identity_sha256=run_identity_sha256)


def verify_pretraining_receipt(model, receipt, *, expected_train_manifest_sha256,
                               expected_run_identity_sha256):
    _check_model(model)
    _require_hash(expected_train_manifest_sha256, "expected_train_manifest_sha256")
    _require_hash(expected_run_identity_sha256, "expected_run_identity_sha256")
    if not isinstance(receipt, dict):
        raise ValueError("C requires the committed M12 pretraining receipt")
    expected = dict(schema_version=SCHEMA, architecture=model.architecture, split="train",
        completed_phase="M", completed_epochs=MATCHER_EPOCHS,
        pair_exposures=MATCHER_EPOCHS * TRAIN_PAIRS,
        optimizer_updates=MATCHER_EPOCHS * TRAIN_PAIRS // EFFECTIVE_BATCH,
        train_manifest_sha256=expected_train_manifest_sha256, run_identity_sha256=expected_run_identity_sha256)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("M12 receipt architecture/data/run identity or budget differs")
    for name in ("initial_matcher_sha256", "trained_matcher_sha256"):
        _require_hash(receipt.get(name), name)
    if (receipt["initial_matcher_sha256"] == receipt["trained_matcher_sha256"] or
            receipt["trained_matcher_sha256"] != matcher_state_digest(model)):
        raise ValueError("frozen matcher does not match completed M12 state")


def enforce_phase_mode(model, spec, *, training=True):
    """Call before forward: a surrounding model.train() cannot unfreeze buffers."""
    _check_model(model, spec)
    model.train(training)
    model.base_model.local_head.eval()
    if spec.name == "M":
        model.score_head.eval()
        model.base_model.fusion.eval()
    elif spec.name == "C":
        model.base_model.eval()
        model.base_model.fusion.train(training)
        model.score_head.train(training)


def configure_phase(model, spec, *, training=True, receipt=None,
                    expected_train_manifest_sha256=None, expected_run_identity_sha256=None,
                    at_optimizer_boundary=True):
    """Set trainable parameters once per transition/resume, not inside accumulation."""
    _check_model(model, spec)
    if not at_optimizer_boundary:
        raise ValueError("phase changes must occur after an optimizer boundary")
    if spec.name == "C":
        verify_pretraining_receipt(model, receipt,
            expected_train_manifest_sha256=expected_train_manifest_sha256,
            expected_run_identity_sha256=expected_run_identity_sha256)
    model.set_training_mode("head_only" if spec.name == "C" else "joint")
    for name, parameter in model.named_parameters():
        family = _family(name)
        enabled = family == "matcher" and spec.name in ("M", "J") or family == "classifier" and spec.name in ("C", "J")
        parameter.requires_grad_(enabled)
        parameter.grad = None  # Prevent stale-gradient AdamW updates of newly frozen parameters.
    model._score_design_stage = asdict(spec)
    enforce_phase_mode(model, spec, training=training)
    return dict(phase=asdict(spec), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_names=[n for n, p in model.named_parameters() if p.requires_grad],
        frozen_names=[n for n, p in model.named_parameters() if not p.requires_grad])


def forward_for_phase(model, spec, *inputs, training=True):
    """Targets are deliberately absent from this interface."""
    if getattr(model, "_score_design_stage", None) != asdict(spec):
        raise ValueError("configure_phase must run before this epoch's forward")
    enforce_phase_mode(model, spec, training=training)
    # M never runs untrained P/R or uses them in loss masks/decision validity.
    return model.base_model(*inputs) if spec.name == "M" else model(*inputs)


def compute_phase_loss(spec, output, labels, target_a, target_b, translation_target_rc,
                       translation_valid, *, pose_supervision_enabled,
                       base=RachelN512LossConfig(), split="train") -> PhaseLoss:
    if split != "train":
        raise ValueError("phase/candidate supervision is TRAIN-only; held-out evaluation is target-blind inference")
    config, profile = phase_loss_profile(spec, base)
    original = compute_weathering_loss(output, labels, target_a, target_b,
        translation_target_rc, translation_valid, pose_supervision_enabled=pose_supervision_enabled,
        config=config)
    correctness = original.total.new_zeros(())
    zero = torch.zeros((), dtype=torch.long, device=labels.device)
    counts = dict(pair_positive=(labels == 1).sum().detach(), pair_negative=(labels == 0).sum().detach(),
        r_valid=zero, r_positive=zero, r_negative=zero, r_supervised_pairs=zero,
        shift_supervised=((pose_supervision_enabled & translation_valid & output.training_valid).sum().detach()
                          if config.translation_weight else zero),
        assignment_supervised_matches=(((target_a >= 0) & output.training_valid[:, None]).sum().detach()
                                       if config.assignment_weight else zero))
    if spec.name != "M":
        correctness, stats = candidate_correctness_loss(output, labels, translation_target_rc,
            translation_valid, tolerance_px=R_TOLERANCE_PX)
        counts.update(r_valid=stats["candidate_supervised"].sum().detach(),
            r_positive=stats["candidate_positive_count"],
            r_negative=(stats["candidate_supervised"].sum() - stats["candidate_positive_count"]).detach(),
            r_supervised_pairs=stats["candidate_pair_count"])
    total = original.total
    if profile["candidate_correctness_weight"]:
        total = total + profile["candidate_correctness_weight"] * correctness
    return PhaseLoss(total, original, correctness, config, profile, counts)


__all__ = ["SCHEMA", "EVALUATION_SCHEMA", "ScorePhase", "PhaseLoss", "phase", "phase_loss_profile", "stage_protocol",
           "optimizer_parameter_groups", "matcher_state_digest", "matcher_pretraining_receipt",
           "verify_pretraining_receipt", "configure_phase", "enforce_phase_mode", "forward_for_phase",
           "compute_phase_loss"]
