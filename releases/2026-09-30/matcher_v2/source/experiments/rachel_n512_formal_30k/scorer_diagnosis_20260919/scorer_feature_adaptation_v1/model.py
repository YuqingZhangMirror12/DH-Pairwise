"""Isolated G0/G1 Scorer feature adaptation prototype, not a paper reproduction.

No I/O, checkpoint loading, training loop or data/negative mining occurs here.
G0 and G1 differ only in whether their independent copied feature stem learns.
Use build_g0_g1 to obtain exactly shared initial weights, then separate optimizers.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from staging.pairwise_v0_2.models.rachel_decoupled_score import (
    CrossAttentionPairHead, DecoupledScoreModel, _indices,
)
from staging.pairwise_v0_2.models.rachel_n512 import (
    RachelN512Pairwise, RachelN512Output, _validate_binary_masks, _validate_contours,
)
from ..pair_grid_readout.model import PairGridHead, cosine_grid_lme


@dataclass(frozen=True)
class ScorerOutput:
    raw_similarity: Tensor
    calibrated_logit: Tensor

    @property
    def probability(self):
        return self.calibrated_logit.sigmoid()


@dataclass(frozen=True)
class FeatureAdaptationOutput(RachelN512Output):
    # token_features_* above remain the original MATCHER features, not the stem.
    raw_similarity: Tensor
    calibrated_logit: Tensor


class IndependentScorerStem(nn.Module):
    """Exact pre-primal/dual Rachel patch -> scale gate -> context computation."""
    def __init__(self, base_model):
        super().__init__()
        self.config = deepcopy(base_model.config)
        for name in ("patch_sampler", "patch_encoder", "scale_gate", "context"):
            setattr(self, name, deepcopy(getattr(base_model, name)))

    # Reuse the actual production implementation, including activation checkpointing.
    _encode_patches = RachelN512Pairwise._encode_patches

    def forward(self, mask_a, mask_b, points_rc_a, points_rc_b, valid_a, valid_b):
        mask_a, mask_b = _validate_binary_masks(mask_a, mask_b,
            self.config.canvas_size, validate_values=self.config.validate_runtime_inputs)
        points = []
        for name, p, v in (("A", points_rc_a, valid_a), ("B", points_rc_b, valid_b)):
            p, v = _validate_contours(p, v, batch_size=len(mask_a),
                canvas_size=self.config.canvas_size, contour_cap=self.config.contour_cap,
                name=name, validate_values=self.config.validate_runtime_inputs)
            if p.device != mask_a.device:
                raise ValueError("masks and contour points must share device")
            points.append((p, v))
        (points_rc_a, valid_a), (points_rc_b, valid_b) = points
        a = self._encode_patches(self.patch_sampler(mask_a, points_rc_a, valid_a), valid_a)
        b = self._encode_patches(self.patch_sampler(mask_b, points_rc_b, valid_b), valid_b)
        return self.context(a, b, valid_a, valid_b, points_rc_a, points_rc_b,
                            self.config.canvas_size)


class ExposedPairGridHead(PairGridHead):
    """Same PairGridHead formula, also returning pre-affine raw LME similarity.

    Unlike the generic head, this training prototype fails on empty contours;
    it must not turn absent evidence into a raw similarity/ranking example.
    """
    def forward_scores(self, features_a, features_b, valid_a, valid_b):
        if (features_a.ndim != 3 or features_b.ndim != 3 or not len(features_a)
                or len(features_a) != len(features_b)
                or features_a.shape[-1] != self.feature_dim
                or features_b.shape[-1] != self.feature_dim
                or valid_a.shape != features_a.shape[:2]
                or valid_b.shape != features_b.shape[:2]
                or valid_a.dtype != torch.bool or valid_b.dtype != torch.bool):
            raise ValueError("features/validity dimensions or dtype differ")
        similarities = []
        for a, b, va, vb in zip(features_a, features_b, valid_a, valid_b):
            a, b = a.index_select(0, _indices(va)), b.index_select(0, _indices(vb))
            if not len(a) or not len(b):
                raise ValueError("raw similarity requires nonempty valid contours")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("nonfinite valid descriptor")
            a, b = self.decoder(a, b)
            similarities.append(cosine_grid_lme(a, b, self.tau))
        raw = torch.stack(similarities)
        return ScorerOutput(raw, self.scale * raw + self.bias)

    def forward(self, features_a, features_b, valid_a, valid_b):
        return self.forward_scores(features_a, features_b, valid_a, valid_b).calibrated_logit


class ScorerFeatureAdaptationModel(nn.Module):
    def __init__(self, source_model, *, feature_trainable=False):
        super().__init__()
        if (not isinstance(source_model, DecoupledScoreModel)
                or source_model.phase not in ("matcher", "classifier")):
            raise ValueError("requires a matcher/classifier-phase S7 DecoupledScoreModel source")
        if not isinstance(feature_trainable, bool):
            raise TypeError("feature_trainable must be bool")
        self.base_model = deepcopy(source_model.base_model)
        self.config = self.base_model.config
        self.scorer_stem = IndependentScorerStem(source_model.base_model)
        # Deliberately fresh decoder; do not copy S7's pretrained Scorer head.
        fresh = CrossAttentionPairHead(self.config.feature_dim, 4, depth=2)
        reference = next(self.scorer_stem.parameters())
        self.score_head = ExposedPairGridHead(fresh, tau=15.0).to(
            device=reference.device, dtype=reference.dtype)
        self.feature_trainable = feature_trainable
        self.source_metadata = deepcopy(source_model.metadata())
        self.phase = "classifier"
        self.set_phase("classifier")

    @property
    def arm(self):
        return "G1" if self.feature_trainable else "G0"

    def set_phase(self, phase):
        if phase != "classifier":
            raise ValueError("this prototype never trains Matcher")
        self.base_model.requires_grad_(False)
        self.scorer_stem.requires_grad_(self.feature_trainable)
        self.score_head.requires_grad_(True)
        # No-evidence fallback is inherited for state compatibility but disallowed.
        self.score_head.no_evidence_logit.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        self.scorer_stem.train(bool(mode and self.feature_trainable))
        return self

    def scorer_forward(self, mask_a, mask_b, points_rc_a, points_rc_b, valid_a, valid_b):
        """Training-only path: six inputs, no Matcher/Sinkhorn/layout execution."""
        inputs = (mask_a, mask_b, points_rc_a, points_rc_b, valid_a, valid_b)
        if self.feature_trainable:
            a, b = self.scorer_stem(*inputs)
        else:
            with torch.no_grad():
                a, b = self.scorer_stem(*inputs)
        return self.score_head.forward_scores(a, b, valid_a, valid_b)

    def forward(self, mask_a, mask_b, points_rc_a, points_rc_b, valid_a, valid_b):
        """Evaluation adapter preserving every original Matcher/layout input field."""
        inputs = (mask_a, mask_b, points_rc_a, points_rc_b, valid_a, valid_b)
        with torch.no_grad():
            original = self.base_model(*inputs)
        score = self.scorer_forward(*inputs)
        finite = original.training_valid & torch.isfinite(score.calibrated_logit)
        logit = torch.where(finite, score.calibrated_logit, torch.zeros_like(score.calibrated_logit))
        updated = replace(original, fused_logit=logit, local_logit=logit,
            fused_probability=logit.sigmoid(), local_probability=logit.sigmoid(),
            training_valid=finite, decision_valid=finite & original.transport.diagnostics.converged)
        return FeatureAdaptationOutput(**vars(updated),
            raw_similarity=score.raw_similarity, calibrated_logit=score.calibrated_logit)

    def metadata(self):
        return dict(schema_version="scorer-feature-adaptation-prototype/1", arm=self.arm,
            source_model=self.source_metadata, matcher_frozen=True,
            scorer_feature_trainable=self.feature_trainable,
            scorer_feature_initialization="independent copy of supplied S7 base patch_sampler/patch_encoder/scale_gate/context",
            scorer_head_initialization="fresh; build_g0_g1 clones a single initialization",
            decoder_depth=2, decoder_heads=4, tau=15.0,
            readout="CA tokens -> L2 cosine grid -> LME raw_similarity -> positive affine calibrated_logit",
            label_input_to_forward=False, sinkhorn_or_layout_input_to_scorer=False,
            scorer_forward_runs_matcher=False, patch_coordinates_learned=False,
            loss="PairBCE + 0.3 * mean eligible positive relu(0.15 - raw_pos + max confirmed same-anchor raw_neg)",
            negative_policy="caller supplies confirmed same-anchor edges and canonical anchor IDs; no batch-negative inference",
            adaptation_not_reproduction=True, colleague_window_auxiliary_used=False,
            limitations=["Source checkpoint identity, TRAIN-only sampling and paired optimizer budget belong to trainer.",
                "No colleague window auxiliary: its frag_batch_loss does not exclude anchor/true correspondences.",
                "Full cosine grid is not Matcher-selected correspondences; LME retains match-fraction dilution."])


def build_g0_g1(source_model):
    """Return (frozen-stem G0, trainable-stem G1), bitwise-identical initialization."""
    g0 = ScorerFeatureAdaptationModel(source_model, feature_trainable=False)
    g1 = deepcopy(g0)
    g1.feature_trainable = True
    g1.set_phase("classifier")
    return g0, g1


@dataclass(frozen=True)
class PairRankingLoss:
    total: Tensor
    pair_bce: Tensor
    ranking: Tensor
    ranked_positive_count: int
    skipped_positive_count: int


def pair_bce_with_same_anchor_ranking(raw_similarity: Tensor, calibrated_logit: Tensor,
        labels: Tensor, *, known_negative_pairs: Optional[Tensor] = None,
        anchor_ids: Optional[Sequence[str]] = None, same_anchor_confirmed: bool = False):
    """Mean PairBCE + 0.3 margin-ranking loss on explicit confirmed edges only.

    Each [positive_batch_row, negative_batch_row] edge must be independently known
    to be a true negative with the same canonical fragment anchor (in either pair
    endpoint position). Batch membership/page membership is NOT such evidence.
    The caller supplies IDs and confirms this provenance; labels alone cannot
    establish it. Repeated edges are rejected. Positives with no edge contribute
    BCE but no ranking term. Each eligible positive gets equal ranking weight.
    """
    if (raw_similarity.ndim != 1 or not raw_similarity.numel()
            or calibrated_logit.shape != raw_similarity.shape
            or labels.shape != raw_similarity.shape
            or not raw_similarity.is_floating_point() or not calibrated_logit.is_floating_point()
            or raw_similarity.device != calibrated_logit.device or labels.device != calibrated_logit.device):
        raise ValueError("scores/logits/labels require matching nonempty vectors and device")
    if (not torch.isfinite(raw_similarity).all() or not torch.isfinite(calibrated_logit).all()
            or not ((labels == 0) | (labels == 1)).all()):
        raise ValueError("requires finite scores/logits and binary labels")
    y = labels.detach().cpu().tolist()
    groups = {}
    if known_negative_pairs is not None:
        if (not isinstance(known_negative_pairs, Tensor) or known_negative_pairs.dtype != torch.long
                or known_negative_pairs.ndim != 2 or known_negative_pairs.shape[1] != 2):
            raise ValueError("known_negative_pairs must be an int64 [E,2] tensor")
        edges = known_negative_pairs.detach().cpu().tolist()
        if edges and (same_anchor_confirmed is not True or anchor_ids is None or len(anchor_ids) != len(y)):
            raise ValueError("nonempty edges require explicit confirmation and one canonical anchor ID per row")
        seen = set()
        for positive, negative in edges:
            if (not 0 <= positive < len(y) or not 0 <= negative < len(y)
                    or y[positive] != 1 or y[negative] != 0):
                raise ValueError("edge must index a positive row and a known negative row")
            if (not isinstance(anchor_ids[positive], str) or not anchor_ids[positive]
                    or anchor_ids[positive] != anchor_ids[negative]):
                raise ValueError("edge canonical anchor identities must be known and equal")
            if (positive, negative) in seen:
                raise ValueError("duplicate known-negative edge")
            seen.add((positive, negative))
            groups.setdefault(positive, []).append(negative)
    terms = [F.relu(0.15 - raw_similarity[p] + raw_similarity[negatives].max())
             for p, negatives in sorted(groups.items())]
    ranking = torch.stack(terms).mean() if terms else raw_similarity.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(calibrated_logit, labels.to(calibrated_logit.dtype))
    return PairRankingLoss(bce + 0.3 * ranking, bce, ranking, len(groups),
                           sum(value == 1 for value in y) - len(groups))
