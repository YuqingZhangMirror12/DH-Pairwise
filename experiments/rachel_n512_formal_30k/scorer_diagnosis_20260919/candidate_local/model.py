"""Isolated candidate-local CA residual prototype; no trainer/launch side effects.

C1 and C2 have identical parameters, initialization, eligibility and loss. Their
only forward difference is the residual branch's token mask: predicted inlier
endpoints (C1) or all valid endpoints (C2). No GT or pair label enters selection.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch import Tensor, nn

from staging.pairwise_v0_2.models.rachel_decoupled_score import CrossAttentionPairHead, DecoupledScoreModel
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Output
from staging.pairwise_v0_2.models.translation_layout import TranslationLayoutConfig, estimate_translation_layout

SCHEMA = "rachel-predicted-candidate-local-ca/1"
MODES = ("predicted_inliers", "all_valid_control")
# Exactly the production final layout decoder, not the network's soft mean.
DECODER_CONFIG = TranslationLayoutConfig(correspondence_mode="topk_union", top_k=2,
    max_candidates=512, min_inliers=3, inlier_radius_px=10.0)


@dataclass(frozen=True)
class CandidateSelection:
    mask_a: Tensor                       # bool[B,Na], UNIQUE predicted endpoints
    mask_b: Tensor                       # bool[B,Nb]
    layout_valid: Tensor                 # bool[B]; solver validity, not adjacency
    translation_a_to_b_rc: Tensor        # [B,2]; NaN for invalid solver output
    candidate_indices: Tensor            # int64[B,512,2], -1 only in padding
    candidate_valid: Tensor              # bool[B,512]
    candidate_inliers: Tensor            # bool[B,512], final selected peak
    reasons: tuple


@torch.no_grad()
def select_predicted_inliers(assignment, points_a_rc, points_b_rc, valid_a, valid_b):
    """Target-blind, detached adapter to the EXACT existing CPU decoder.

    No arbitrary translation or candidate override is accepted. Preserve all
    valid original points, deduplicate by token identity, do not grow neighbors
    or keep only one connected run. Corrosion-separated support may coexist.
    """
    if (assignment.ndim != 3 or points_a_rc.ndim != 3 or points_b_rc.ndim != 3
            or not assignment.is_floating_point() or not points_a_rc.is_floating_point()
            or not points_b_rc.is_floating_point()
            or points_a_rc.shape[-1] != 2 or points_b_rc.shape[-1] != 2
            or valid_a.dtype != torch.bool or valid_b.dtype != torch.bool
            or valid_a.shape != points_a_rc.shape[:2] or valid_b.shape != points_b_rc.shape[:2]
            or assignment.shape != (len(valid_a), valid_a.shape[1], valid_b.shape[1])
            or len(valid_a) != len(valid_b) or len(valid_a) == 0):
        raise ValueError("expected Q[B,Na,Nb], P[B,N,2], valid bool[B,N]")
    tensors = (assignment, points_a_rc, points_b_rc, valid_a, valid_b)
    if len({value.device for value in tensors}) != 1:
        raise ValueError("selection tensors must share a device")
    device, batch = assignment.device, len(assignment)
    arrays = [value.detach().cpu().numpy() for value in tensors]
    q, a, b, va, vb = arrays
    selected_a, selected_b = np.zeros_like(va), np.zeros_like(vb)
    indices = np.full((batch, 512, 2), -1, np.int64)
    present, inliers = np.zeros((batch, 512), bool), np.zeros((batch, 512), bool)
    translations, valid, reasons = [], [], []
    for i in range(batch):
        result = estimate_translation_layout(a[i], b[i], q[i], va[i], vb[i], config=DECODER_CONFIG)
        count = result.candidate_count
        indices[i, :count] = result.candidate_indices
        present[i, :count] = True
        inliers[i, :count] = result.inlier_mask
        if result.valid:
            pairs = result.candidate_indices[result.inlier_mask]
            selected_a[i, np.unique(pairs[:, 0])] = True
            selected_b[i, np.unique(pairs[:, 1])] = True
        translations.append(result.t_a_to_b_rc)
        valid.append(result.valid)
        reasons.append(result.reason)
    return CandidateSelection(
        torch.as_tensor(selected_a, device=device), torch.as_tensor(selected_b, device=device),
        torch.tensor(valid, dtype=torch.bool, device=device),
        torch.as_tensor(np.asarray(translations), dtype=points_a_rc.dtype, device=device),
        torch.as_tensor(indices, device=device), torch.as_tensor(present, device=device),
        torch.as_tensor(inliers, device=device), tuple(reasons))


@dataclass(frozen=True)
class CandidateScores:
    global_logit: Tensor
    local_delta: Tensor
    fused_logit: Tensor
    local_eligible: Tensor


class CandidateResidualHead(nn.Module):
    """Original global CA + signed, zero-initialized residual from the candidate.

    Clone both branches from the same C8 head without consuming RNG. Zero ONLY
    the residual's final Linear(64,1), not its features. This is identical for
    C1/C2 and preserves the initial score exactly. The residual may reduce a
    false-positive score; it is not a positive-only rescue/gate.
    """
    def __init__(self, source_head, mode="predicted_inliers"):
        super().__init__()
        if not isinstance(source_head, CrossAttentionPairHead) or source_head.depth != 2:
            raise ValueError("prototype is bound to S6 depth2 cross-attention")
        if mode not in MODES:
            raise ValueError("unregistered candidate residual mode")
        self.mode = mode
        self.global_head = deepcopy(source_head)
        self.local_head = deepcopy(source_head)
        nn.init.zeros_(self.local_head.classifier[-1].weight)
        nn.init.zeros_(self.local_head.classifier[-1].bias)
        nn.init.zeros_(self.local_head.no_evidence_logit)

    def forward(self, features_a, features_b, valid_a, valid_b, selection):
        if (selection.mask_a.shape != valid_a.shape or selection.mask_b.shape != valid_b.shape
                or selection.mask_a.dtype != torch.bool or selection.mask_b.dtype != torch.bool
                or selection.layout_valid.shape != (len(valid_a),)
                or selection.layout_valid.dtype != torch.bool
                or selection.mask_a.device != features_a.device or selection.mask_b.device != features_b.device):
            raise ValueError("candidate membership must align with original token tensors")
        if torch.any(selection.mask_a & ~valid_a) or torch.any(selection.mask_b & ~valid_b):
            raise ValueError("candidate mask includes padding/invalid tokens")
        eligible = selection.layout_valid & selection.mask_a.any(1) & selection.mask_b.any(1)
        # Same eligibility in C1 and capacity-control C2: ONLY token mask differs.
        ma, mb = ((selection.mask_a, selection.mask_b) if self.mode == "predicted_inliers"
                  else (valid_a, valid_b))
        ma, mb = ma & eligible[:, None], mb & eligible[:, None]
        global_logit = self.global_head(features_a, features_b, valid_a, valid_b)
        raw_delta = self.local_head(features_a, features_b, ma, mb)
        delta = torch.where(eligible, raw_delta, torch.zeros_like(raw_delta))
        return CandidateScores(global_logit, delta, global_logit + delta, eligible)


@dataclass(frozen=True)
class CandidateLocalOutput(RachelN512Output):
    score_details: dict


class FrozenCandidateLocalModel(nn.Module):
    """Six-input Rachel adapter; frozen original Matcher and separate CA heads."""
    def __init__(self, source_model, mode="predicted_inliers"):
        super().__init__()
        if (not isinstance(source_model, DecoupledScoreModel)
                or source_model.head_kind != "cross_attention" or source_model.phase != "classifier"):
            raise ValueError("requires the existing classifier-phase S6 model")
        self.base_model = deepcopy(source_model.base_model)
        self.config = self.base_model.config
        self.score_head = CandidateResidualHead(source_model.score_head, mode)
        self.source_metadata = deepcopy(source_model.metadata())
        self.phase = "classifier"
        self.reset_decoder_cost()
        self.set_phase("classifier")

    def reset_decoder_cost(self):
        self.decoder_cost = dict(calls=0, pairs=0, elapsed_s=0.,
            includes_device_to_host_sync=True, cached=False)

    def set_phase(self, phase):
        if phase != "classifier":
            raise ValueError("candidate-local prototype never trains the Matcher")
        self.phase = phase
        self.base_model.requires_grad_(False)
        self.score_head.requires_grad_(True)
        for parameter in self.parameters():
            parameter.grad = None
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b):
        with torch.no_grad():
            original = self.base_model(mask_a, mask_b, points_rc_a, points_rc_b,
                                       contour_valid_a, contour_valid_b)
        decoder_started = time.perf_counter()
        selected = select_predicted_inliers(original.assignment, points_rc_a, points_rc_b,
                                            contour_valid_a, contour_valid_b)
        self.decoder_cost["calls"] += 1
        self.decoder_cost["pairs"] += len(mask_a)
        self.decoder_cost["elapsed_s"] += time.perf_counter() - decoder_started
        scores = self.score_head(original.token_features_a, original.token_features_b,
                                 contour_valid_a, contour_valid_b, selected)
        finite = original.training_valid & torch.isfinite(scores.fused_logit)
        score = torch.where(finite, scores.fused_logit, torch.zeros_like(scores.fused_logit))
        values = {field.name: getattr(original, field.name) for field in fields(original)}
        values.update(fused_logit=score, local_logit=score, fused_probability=score.sigmoid(),
            local_probability=score.sigmoid(), training_valid=finite,
            decision_valid=finite & original.transport.diagnostics.converged,
            score_details=dict(global_logit=scores.global_logit, local_residual_logit=scores.local_delta,
                local_eligible=scores.local_eligible, predicted_layout_valid=selected.layout_valid,
                predicted_translation_a_to_b_rc=selected.translation_a_to_b_rc,
                inlier_token_mask_a=selected.mask_a, inlier_token_mask_b=selected.mask_b,
                unique_inlier_token_count_a=selected.mask_a.sum(1),
                unique_inlier_token_count_b=selected.mask_b.sum(1)))
        return CandidateLocalOutput(**values)

    def metadata(self):
        return dict(schema_version=SCHEMA, mode=self.score_head.mode,
            source_model=self.source_metadata, decoder_config=asdict(DECODER_CONFIG),
            selector="detached predicted best-mode inlier unique endpoints; no GT or labels",
            residual_initialization="copy C8 S6-D2 CA/pooling/MLP; zero final Linear and unused fallback",
            score="global_logit + signed local_residual; no coarse/matrix-CNN score input",
            fallback="invalid/no candidate: exactly global score, residual zero",
            classification_loss="PairBCE only; not candidate-layout correctness supervision",
            features="post-context frozen Matcher tokens; selected tokens still contain global context",
            decoder_and_matcher_unchanged=True, minimum_seam_length_gate_added=False,
            continuous_seam_or_overlap_constraint_added=False)


def build_from_s6_epoch20(checkpoint, mode="predicted_inliers"):
    """CPU-safe construction, no RNG restoration or training as a side effect."""
    from experiments.rachel_n512_formal_30k import train_score_decoupled as trainer
    if (checkpoint.get("epoch") != 20 or checkpoint.get("completed_segments") != 80
            or checkpoint.get("phase") != "classifier" or checkpoint.get("inference_only")):
        raise ValueError("requires original resumable S6 epoch20, not C16 or another winner")
    # The low-level model constructor consumes RNG while loading tensors. Restore
    # its state so C1/C2 branch construction itself cannot change shuffle/dropout.
    rng = trainer.capture_rng_state()
    try:
        source = trainer.load_decoupled_checkpoint(checkpoint)
        return FrozenCandidateLocalModel(source, mode)
    finally:
        trainer.restore_rng_state(rng)


def restore_source_optimizer(checkpoint, model):
    """Retain global Adam exactly; add ONE cold local group, never reset global.

    Caller must move model to the intended device BEFORE this and restore the
    source RNG AFTER all data/model construction. This helper intentionally does
    not change RNG or impersonate the old two-group checkpoint schema.
    """
    from experiments.rachel_n512_formal_30k import train_score_decoupled as trainer
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation import continue_classifier as continuation
    if (not isinstance(model, FrozenCandidateLocalModel) or checkpoint.get("epoch") != 20
            or checkpoint.get("completed_segments") != 80 or checkpoint.get("inference_only")):
        raise ValueError("new candidate head optimizer must originate at epoch20")
    rng = trainer.capture_rng_state()
    try:
        source = trainer.load_decoupled_checkpoint(checkpoint)
    finally:
        trainer.restore_rng_state(rng)
    facade = SimpleNamespace(base_model=model.base_model, score_head=model.score_head.global_head)
    continuation.check_optimizer(checkpoint, facade, expected_head_step=12000)
    for target, reference in ((model.base_model, source.base_model),
                              (model.score_head.global_head, source.score_head)):
        if trainer.state_digest(target) != trainer.state_digest(reference):
            raise ValueError("global/Matcher weights differ from declared source")
    if torch.count_nonzero(model.score_head.local_head.classifier[-1].weight).item() or torch.count_nonzero(model.score_head.local_head.classifier[-1].bias).item():
        raise ValueError("new local residual must start with exactly zero final projection")
    optimizer = trainer.create_optimizer(facade)
    optimizer.load_state_dict(deepcopy(checkpoint["optimizer_state_dict"]))
    group = {key: deepcopy(value) for key, value in optimizer.param_groups[1].items() if key not in ("params", "phase_family")}
    group.update(params=list(model.score_head.local_head.parameters()), phase_family="candidate_local")
    optimizer.add_param_group(group)
    receipt = dict(schema_version=SCHEMA, source_epoch=20, source_head_adam_step=12000,
        preserved_optimizer_groups=["base", "new_head"], added_group="candidate_local",
        added_group_initial_states=0, optimizer_reset=False, global_parameters_continued=True,
        source_rng_restoration_required=True, matcher_frozen=True, lr=2e-5,
        new_checkpoint_schema_required=True, source_mode=model.score_head.mode)
    return optimizer, receipt
