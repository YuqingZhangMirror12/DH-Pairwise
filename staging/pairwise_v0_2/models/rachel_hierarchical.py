"""Global-to-local Rachel matcher, with image/contour coarse-source controls.

Both controls expose K pooled coarse tokens and one global embedding per
fragment to the SAME bidirectional, shared-weight local-conditioning module.
The update precedes primal/dual affinity and partial Sinkhorn; coarse scores
never reject pairs or tokens. Dustbins, complementary affinity, score heads,
and translation-only geometry retain the Full model's definitions.

Image coarse preserves the original image CNN and its score. Contour coarse
aggregates ordered local contour features BEFORE cross-fragment interaction,
inspired by ShreddingNet's local-to-global aggregation, not a reproduction of
its RGB/ResGCN network. The old image encoder/projection remain in the state
dict for strict Full warm starts, but are unused by the contour control.

Warm start: load_full_state_dict loads every original learned parameter and
regenerates the patch-grid buffer from the target config. The global-to-local
residual projection starts at zero. With an unchanged base config, image
outputs initially equal Full; contour affinity/OT/local geometry initially
equal Full on valid pairs, but its newly initialized global embedding changes
coarse/fused scores. Do not call those two initial coarse branches equivalent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .coarse import CoarseOutput
from .optimal_transport import dustbin_sinkhorn
from .rachel_n512 import (
    RachelN512Config, RachelN512Output, RachelN512Pairwise,
    _validate_binary_masks, _validate_contours,
)


@dataclass(frozen=True)
class RachelHierarchicalConfig:
    """Serializable model_options; the unchanged base config is separate."""

    coarse_source: str = "image"
    global_token_count: int = 8
    attention_heads: int = 4

    def __post_init__(self) -> None:
        if self.coarse_source not in {"image", "contour"}:
            raise ValueError("coarse_source must be 'image' or 'contour'")
        for name in ("global_token_count", "attention_heads"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(name + " must be a positive integer")


def _pool_ordered_tokens(tokens: Tensor, valid: Tensor, count: int) -> Tuple[Tensor, Tensor]:
    """Masked contiguous bins in VALID-token order, without adaptive-pool CUDA backward.

    Image feature pixels use row-major spatial order; contour tokens use their
    boundary order. Invalid slots neither shift bin assignments nor add mass.
    Empty bins stay invalid, so fewer than K valid tokens are also supported.
    """
    rank = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
    valid_count = valid.sum(dim=1, keepdim=True).clamp_min(1)
    bins = torch.div(rank * count, valid_count, rounding_mode="floor").clamp_max(count - 1)
    membership = F.one_hot(bins, num_classes=count).transpose(1, 2).to(tokens.dtype)
    membership = membership * valid[:, None].to(tokens.dtype)
    mass = membership.sum(dim=2, keepdim=True)
    safe_tokens = torch.where(valid[:, :, None], tokens, torch.zeros_like(tokens))
    pooled = torch.bmm(membership, safe_tokens) / mass.clamp_min(1.0)
    return pooled, mass.squeeze(2) > 0


def _mean_and_max(tokens: Tensor, valid: Tensor) -> Tensor:
    safe = torch.where(valid[:, :, None], tokens, torch.zeros_like(tokens))
    mean = safe.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1).to(tokens.dtype)
    maximum = torch.where(valid[:, :, None], tokens, torch.full_like(tokens, -torch.inf)).amax(dim=1)
    maximum = torch.where(valid.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=1)


class GlobalToLocalConditioner(nn.Module):
    """Each local query reads own/other global evidence, with no score gate.

    Shared role embeddings preserve A/B exchange equivariance: each direction
    sees role 0 for its own fragment and role 1 for the other fragment. The
    zero residual starts as identity but has a nonzero first-step gradient;
    upstream attention/source gradients open as the residual learns.
    """

    def __init__(self, feature_dim: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(feature_dim)
        self.memory_norm = nn.LayerNorm(feature_dim)
        self.role_embedding = nn.Parameter(torch.empty(2, feature_dim))
        nn.init.normal_(self.role_embedding, std=0.02)
        self.attention = nn.MultiheadAttention(feature_dim, heads, dropout=0.0, batch_first=True)
        self.residual = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.zeros_(self.residual.weight)

    def forward(
        self, local: Tensor, local_valid: Tensor,
        own: Tensor, own_valid: Tensor, other: Tensor, other_valid: Tensor,
    ) -> Tensor:
        memory = torch.cat((own + self.role_embedding[0], other + self.role_embedding[1]), dim=1)
        memory_valid = torch.cat((own_valid, other_valid), dim=1)
        has_memory = memory_valid.any(dim=1)
        memory = torch.where(memory_valid[:, :, None], memory, torch.zeros_like(memory))
        # MultiheadAttention needs one finite key even for an invalid pair.
        # This fallback key is never used as a real update below.
        safe_valid = memory_valid.clone()
        safe_valid[:, 0] = safe_valid[:, 0] | ~has_memory
        normalized_memory = self.memory_norm(memory)
        message, _ = self.attention(
            self.query_norm(local), normalized_memory, normalized_memory,
            key_padding_mask=~safe_valid, need_weights=False,
        )
        update = self.residual(message)
        update = torch.where(has_memory[:, None, None], update, torch.zeros_like(update))
        result = local + update
        return torch.where(local_valid[:, :, None], result, torch.zeros_like(result))


class RachelHierarchicalPairwise(RachelN512Pairwise):
    """Full-compatible forward interface with pre-affinity global guidance.

    Factory construction:
        model = RachelHierarchicalPairwise(
            RachelN512Config(**checkpoint['model_config']),
            RachelHierarchicalConfig(**checkpoint['model_options']))
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    For a NEW warm-start run, call load_full_state_dict instead. Do not use
    strict=False to restore hierarchical checkpoints: their source and new
    learned parameters are part of the architecture, not optional extras.
    """

    def __init__(
        self, config: Optional[RachelN512Config] = None,
        hierarchical_config: Optional[RachelHierarchicalConfig] = None,
    ) -> None:
        super().__init__(config)
        self.hierarchical_config = hierarchical_config or RachelHierarchicalConfig()
        if not isinstance(self.hierarchical_config, RachelHierarchicalConfig):
            raise TypeError("hierarchical_config must be RachelHierarchicalConfig")
        options, dim = self.hierarchical_config, self.config.feature_dim
        if dim % options.attention_heads:
            raise ValueError("feature_dim must be divisible by hierarchical attention_heads")
        embedding_dim = self.coarse.projection[2].out_features
        if options.coarse_source == "image":
            source_dim = self.coarse.projection[2].in_features
        else:
            source_dim = dim
            self.hierarchical_contour_embedding = nn.Sequential(
                nn.Linear(2 * dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.SiLU(),
            )
        self.hierarchical_arc_projection = nn.Linear(source_dim, dim, bias=False)
        self.hierarchical_embedding_projection = nn.Linear(embedding_dim, dim, bias=False)
        self.hierarchical_token_norm = nn.LayerNorm(dim)
        self.hierarchical_conditioner = GlobalToLocalConditioner(dim, options.attention_heads)

    def load_full_state_dict(self, full_state: Mapping[str, Tensor]) -> dict:
        """Strict Full warm start; new parameters retain their constructor initialization.

        Original state keys keep their names. The source must contain exactly
        the original Full keys, not a partial or hierarchical checkpoint.
        Sampling grids are configuration, not learned weights, and are rebuilt.
        """
        current = self.state_dict()
        original_keys = {key for key in current if not key.startswith("hierarchical_")}
        if set(full_state) != original_keys:
            raise ValueError("Warm start requires exactly the complete original Full state keys")
        for key in original_keys:
            if key != "patch_sampler.offsets_rc":
                current[key] = full_state[key]
        self.load_state_dict(current, strict=True)
        return self.architecture_metadata()

    def architecture_metadata(self) -> dict:
        additional = sum(parameter.numel() for name, parameter in self.named_parameters()
                         if name.startswith("hierarchical_"))
        total = sum(parameter.numel() for parameter in self.parameters())
        unused = 0
        unused_modules = []
        if self.hierarchical_config.coarse_source == "contour":
            unused_modules = ["coarse.encoder", "coarse.projection"]
            unused = sum(parameter.numel() for module in (self.coarse.encoder, self.coarse.projection)
                         for parameter in module.parameters())
        return dict(
            model_kind="hierarchical", model_config=asdict(self.config),
            model_options=asdict(self.hierarchical_config),
            global_tokens_per_fragment=self.hierarchical_config.global_token_count + 1,
            total_parameter_count=total, additional_parameter_count=additional,
            unused_original_parameter_count=unused, unused_original_modules=unused_modules,
            active_parameter_count=total - unused,
            injection_location="after local context, before primal/dual affinity and partial Sinkhorn",
            hard_coarse_gate=False, rotation_head=False,
            warmstart="all Full learned parameters; target-config patch grid; new parameters initialized separately",
            zero_residual_initial_equivalence=(
                "all Full outputs with identical base config" if self.hierarchical_config.coarse_source == "image"
                else "valid-pair affinity/OT/local geometry only; new contour coarse embedding changes coarse/fused scores"),
            comparison_scope="matched warm-start adaptation, not equal pretrained representations or official ShreddingNet reproduction",
        )

    def _coarse_output(self, embedding_a: Tensor, embedding_b: Tensor, valid: Tensor) -> CoarseOutput:
        symmetric = torch.cat((torch.abs(embedding_a - embedding_b), embedding_a * embedding_b,
                               0.5 * (embedding_a + embedding_b), torch.maximum(embedding_a, embedding_b)), dim=1)
        raw = self.coarse.head(symmetric).squeeze(1)
        logit = torch.where(valid, raw, torch.zeros_like(raw))
        return CoarseOutput(logit, torch.sigmoid(logit), embedding_a, embedding_b, valid)

    def _global_tokens(
        self, source: Tensor, valid: Tensor, embedding: Tensor, embedding_valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        pooled, pooled_valid = _pool_ordered_tokens(source, valid, self.hierarchical_config.global_token_count)
        tokens = torch.cat((self.hierarchical_arc_projection(pooled),
                            self.hierarchical_embedding_projection(embedding)[:, None]), dim=1)
        token_valid = torch.cat((pooled_valid, embedding_valid[:, None]), dim=1)
        tokens = self.hierarchical_token_norm(tokens)
        return torch.where(token_valid[:, :, None], tokens, torch.zeros_like(tokens)), token_valid

    def _image_global(self, mask_a: Tensor, mask_b: Tensor):
        masks = tuple(F.interpolate(mask, size=(self.config.coarse_size, self.config.coarse_size),
                                    mode="nearest") for mask in (mask_a, mask_b))
        valid = self.coarse._validate(*masks)
        embeddings, memories = [], []
        for mask in masks:
            safe = torch.where(valid[:, None, None, None], mask, torch.zeros_like(mask))
            spatial = self.coarse.encoder(safe)
            embedding = self.coarse.projection(spatial)
            embedding = torch.where(valid[:, None], embedding, torch.zeros_like(embedding))
            tokens = spatial.flatten(2).transpose(1, 2)
            embeddings.append(embedding)
            memories.append(self._global_tokens(tokens, valid[:, None].expand(-1, tokens.shape[1]), embedding, valid))
        return self._coarse_output(*embeddings, valid), memories

    def _contour_global(self, within_a, within_b, valid_a, valid_b, mask_a, mask_b):
        # Validate full-resolution masks, but never execute the image CNN.
        valid = self.coarse._validate(mask_a, mask_b) & (valid_a.sum(1) >= 4) & (valid_b.sum(1) >= 4)
        embeddings, memories = [], []
        for tokens, token_valid in ((within_a, valid_a), (within_b, valid_b)):
            embedding = self.hierarchical_contour_embedding(_mean_and_max(tokens, token_valid))
            embedding = torch.where(valid[:, None], embedding, torch.zeros_like(embedding))
            embeddings.append(embedding)
            memories.append(self._global_tokens(tokens, token_valid & valid[:, None], embedding, valid))
        return self._coarse_output(*embeddings, valid), memories

    def forward(
        self, mask_a: Tensor, mask_b: Tensor, points_rc_a: Tensor, points_rc_b: Tensor,
        contour_valid_a: Tensor, contour_valid_b: Tensor,
    ) -> RachelN512Output:
        mask_a, mask_b = _validate_binary_masks(
            mask_a, mask_b, self.config.canvas_size, validate_values=self.config.validate_runtime_inputs)
        for name, points, valid in (("A", points_rc_a, contour_valid_a), ("B", points_rc_b, contour_valid_b)):
            _validate_contours(points, valid, batch_size=mask_a.shape[0], canvas_size=self.config.canvas_size,
                               contour_cap=self.config.contour_cap, name=name,
                               validate_values=self.config.validate_runtime_inputs)
        if self.hierarchical_config.coarse_source == "image":
            coarse, memories = self._image_global(mask_a, mask_b)
        encoded_a = self._encode_patches(self.patch_sampler(mask_a, points_rc_a, contour_valid_a), contour_valid_a)
        encoded_b = self._encode_patches(self.patch_sampler(mask_b, points_rc_b, contour_valid_b), contour_valid_b)
        scale = 2.0 / float(self.config.canvas_size - 1)
        within_a = self.context._within(encoded_a, contour_valid_a, points_rc_a * scale - 1.0)
        within_b = self.context._within(encoded_b, contour_valid_b, points_rc_b * scale - 1.0)
        if self.hierarchical_config.coarse_source == "contour":
            coarse, memories = self._contour_global(within_a, within_b, contour_valid_a, contour_valid_b, mask_a, mask_b)
        context_a = self.context._cross(within_a, contour_valid_a, within_b, contour_valid_b)
        context_b = self.context._cross(within_b, contour_valid_b, within_a, contour_valid_a)
        (memory_a, memory_valid_a), (memory_b, memory_valid_b) = memories
        context_a = self.hierarchical_conditioner(context_a, contour_valid_a, memory_a, memory_valid_a, memory_b, memory_valid_b)
        context_b = self.hierarchical_conditioner(context_b, contour_valid_b, memory_b, memory_valid_b, memory_a, memory_valid_a)
        primal_a = F.normalize(self.primal(context_a), dim=2, eps=1e-6)
        primal_b = F.normalize(self.primal(context_b), dim=2, eps=1e-6)
        dual_a = F.normalize(self.dual(context_a), dim=2, eps=1e-6)
        dual_b = F.normalize(self.dual(context_b), dim=2, eps=1e-6)
        affinity = 0.5 * (torch.matmul(primal_a, dual_b.transpose(1, 2))
                          + torch.matmul(dual_a, primal_b.transpose(1, 2)))
        transport = dustbin_sinkhorn(
            affinity, contour_valid_a, contour_valid_b, dustbin_score=self.dustbin_score,
            temperature=self.config.matcher_temperature, num_iterations=self.config.sinkhorn_iterations,
            tolerance=self.config.sinkhorn_tolerance,
            checkpoint_iterations=(self.config.activation_checkpointing and self.training and torch.is_grad_enabled()))
        translation, dispersion, mass = self._translation(transport.real_transport, points_rc_a, points_rc_b)
        local_logit, _ = self.local_head(
            affinity, transport.real_transport, transport.dustbin_col, transport.dustbin_row,
            contour_valid_a, contour_valid_b, dispersion, self.config.canvas_size)
        fusion_input = torch.stack((coarse.logit, local_logit, coarse.logit * local_logit,
                                    torch.abs(coarse.logit - local_logit)), dim=1)
        fused_logit = self.fusion(fusion_input).squeeze(1)
        finite = (coarse.valid_problem & transport.diagnostics.valid_problem
                  & transport.diagnostics.finite_output & torch.isfinite(local_logit) & torch.isfinite(fused_logit))
        decision_valid = finite & transport.diagnostics.converged
        fused_logit = torch.where(finite, fused_logit, torch.zeros_like(fused_logit))
        local_logit = torch.where(finite, local_logit, torch.zeros_like(local_logit))
        return RachelN512Output(
            fused_logit=fused_logit, fused_probability=torch.sigmoid(fused_logit),
            coarse_logit=coarse.logit, coarse_probability=coarse.probability,
            local_logit=local_logit, local_probability=torch.sigmoid(local_logit),
            affinity=affinity, assignment=transport.real_transport,
            unmatched_a=transport.dustbin_col, unmatched_b=transport.dustbin_row,
            translation_hat_rc=translation,
            translation_hat_xy_cartesian=torch.stack((translation[:, 1], -translation[:, 0]), dim=1),
            translation_dispersion_px=dispersion, matched_mass=mass,
            token_features_a=context_a, token_features_b=context_b,
            coarse=coarse, transport=transport, training_valid=finite, decision_valid=decision_valid,
        )


__all__ = ["RachelHierarchicalConfig", "RachelHierarchicalPairwise", "GlobalToLocalConditioner"]
