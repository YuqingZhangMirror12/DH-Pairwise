"""Scorer-only cosine-grid readout; not a reproduction of geo-attn training.

Keep our image-patch Matcher and Cross-Attention decoder. Replace only the
post-decoder readout. No training, checkpoint loading or I/O occurs on import.
LME is intentionally retained with its match-fraction dilution limitation.
"""
from copy import deepcopy
from dataclasses import replace
import math

import torch
from torch import nn
import torch.nn.functional as F

from staging.pairwise_v0_2.models.rachel_decoupled_score import (
    CrossAttentionPairHead, DecoupledScoreModel, _indices,
)


def cosine_grid_lme(a, b, tau=15.0):
    """Compact VALID post-decoder tokens -> scalar, not a probability.

    Matrix entries are cosine similarities, NOT Sinkhorn transport and not
    translation-verified correspondences. Every valid cell enters the LME.
    """
    if (isinstance(tau, bool) or not math.isfinite(tau) or tau <= 0
            or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]
            or min(a.shape) == 0 or min(b.shape) == 0):
        raise ValueError("requires nonempty compatible token matrices and finite tau>0")
    affinity = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T
    return (torch.logsumexp(tau * affinity.reshape(-1), dim=0)
            - math.log(affinity.numel())) / tau


class CopiedDecoder(nn.Module):
    """Exact existing synchronous CA path, with no unused old pooling weights."""
    def __init__(self, source):
        super().__init__()
        if not isinstance(source, CrossAttentionPairHead):
            raise TypeError("requires our CrossAttentionPairHead")
        self.depth = source.depth
        for name in ("norm", "cross_attention", "output_norm", "ffn", "extra_layers"):
            setattr(self, name, deepcopy(getattr(source, name)))

    def _interact(self, query, source):
        return CrossAttentionPairHead._interact(self, query, source)

    def forward(self, a, b):
        return CrossAttentionPairHead._decode_pair(self, a, b)


class PairGridHead(nn.Module):
    """CA -> L2 cosine grid -> LME -> positive-slope affine binary logit.

    A fresh readout requires training/calibration using TRAIN only. Initial
    logits do not reproduce the source MLP and are not deployable scores.
    """
    def __init__(self, source_head, tau=15.0):
        super().__init__()
        if isinstance(tau, bool) or not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        self.decoder = CopiedDecoder(source_head)
        self.tau = float(tau)
        self.feature_dim = source_head.norm.normalized_shape[0]
        # softplus keeps a positive monotone readout without exp overflow.
        self.raw_scale = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.bias = nn.Parameter(torch.zeros(()))
        self.no_evidence_logit = nn.Parameter(torch.zeros(()))

    @property
    def scale(self):
        return F.softplus(self.raw_scale)

    def forward(self, features_a, features_b, valid_a, valid_b):
        if (features_a.ndim != 3 or features_b.ndim != 3 or not len(features_a)
                or len(features_a) != len(features_b)
                or features_a.shape[-1] != self.feature_dim
                or features_b.shape[-1] != self.feature_dim
                or valid_a.shape != features_a.shape[:2]
                or valid_b.shape != features_b.shape[:2]
                or valid_a.dtype != torch.bool or valid_b.dtype != torch.bool):
            raise ValueError("features/validity dimensions or dtype differ")
        scores = []
        for a, b, va, vb in zip(features_a, features_b, valid_a, valid_b):
            a, b = a.index_select(0, _indices(va)), b.index_select(0, _indices(vb))
            if not len(a) or not len(b):
                scores.append(self.no_evidence_logit)
                continue
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("nonfinite valid descriptor")
            a, b = self.decoder(a, b)
            scores.append(self.scale * cosine_grid_lme(a, b, self.tau) + self.bias)
        return torch.stack(scores)


class FrozenPairGridModel(nn.Module):
    """Six-input adapter: original Matcher/Sinkhorn untouched and always frozen."""
    def __init__(self, source_model, tau=15.0):
        super().__init__()
        if (not isinstance(source_model, DecoupledScoreModel)
                or source_model.phase != "classifier"
                or source_model.head_kind != "cross_attention"):
            raise ValueError("requires classifier-phase source with Cross-Attention head")
        self.base_model = deepcopy(source_model.base_model)
        self.config = self.base_model.config
        self.score_head = PairGridHead(source_model.score_head, tau=tau)
        self.source_metadata = deepcopy(source_model.metadata())
        self.phase = "classifier"
        self.set_phase("classifier")

    def set_phase(self, phase):
        if phase != "classifier":
            raise ValueError("readout experiment never trains Matcher")
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

    def forward(self, mask_a, mask_b, points_rc_a, points_rc_b, valid_a, valid_b):
        with torch.no_grad():
            original = self.base_model(mask_a, mask_b, points_rc_a, points_rc_b, valid_a, valid_b)
        score = self.score_head(original.token_features_a, original.token_features_b, valid_a, valid_b)
        finite = original.training_valid & torch.isfinite(score)
        score = torch.where(finite, score, torch.zeros_like(score))
        return replace(original, fused_logit=score, local_logit=score,
            fused_probability=score.sigmoid(), local_probability=score.sigmoid(),
            training_valid=finite, decision_valid=finite & original.transport.diagnostics.converged)

    def metadata(self):
        return dict(schema_version="rachel-ca-pair-grid-readout/1",
            source_model=self.source_metadata, phase=self.phase,
            decoder_depth=self.score_head.decoder.depth,
            decoder_heads=self.score_head.decoder.cross_attention.num_heads,
            tau=self.score_head.tau, matcher_frozen=True,
            readout="CA tokens L2 -> full valid cosine grid -> LME -> positive affine logit",
            readout_initialization="fresh; does not replay pretrained MLP scores or Adam state",
            coarse_used=False, sinkhorn_or_layout_used_by_scorer=False,
            continuous_seam_constraint=False,
            limitation="LME still dilutes sparse strong matches among unrelated valid cells")
