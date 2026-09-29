"""Match each physical window scale independently, then mix partial transports.

Unlike RachelN512Pairwise, descriptors are not fused before matching. The
encoder, cyclic/landmark context, complementary primal/dual projections and
dustbin parameter are shared, but applied independently to each scale. The
scale axis is folded into the batch axis, never into the ordered token axis.

A single global non-negative scale weight vector sums to one. Mixing *all*
augmented transport entries with these same weights preserves the partial-OT
marginals up to the original numerical Sinkhorn residuals. Cell-/row-dependent
gating would not have this guarantee and is deliberately not used here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .optimal_transport import PartialTransportOutput, SinkhornDiagnostics, dustbin_sinkhorn
from .rachel_n512 import (
    RachelN512Config, RachelN512Output, RachelN512Pairwise,
    _validate_binary_masks, _validate_contours,
)


MODEL_KIND = "multiscale_transport"


@dataclass(frozen=True)
class RachelMultiscaleTransportConfig(RachelN512Config):
    window_sizes_px: Tuple[float, ...] = (7.0, 16.0, 32.0, 64.0)
    initial_scale_weights: Tuple[float, ...] = ()
    learn_scale_weights: bool = True
    affinity_readout: str = "mixed_product"

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.learn_scale_weights) is not bool:
            raise TypeError("learn_scale_weights must be bool")
        if not isinstance(self.affinity_readout, str) or self.affinity_readout not in (
            "mixed_product", "within_scale_product",
        ):
            raise ValueError("affinity_readout must be 'mixed_product' or 'within_scale_product'")
        if self.initial_scale_weights:
            if len(self.initial_scale_weights) != len(self.window_sizes_px):
                raise ValueError("initial_scale_weights must have one value per scale")
            weights = self.initial_scale_weights
            if any(isinstance(w, bool) or not math.isfinite(float(w)) or w < 0.0 for w in weights):
                raise ValueError("scale weights must be finite and non-negative")
            if sum(weights) <= 0.0:
                raise ValueError("at least one scale weight must be positive")
            if self.learn_scale_weights and min(weights) <= 0.0:
                raise ValueError("learned initial scale weights must be strictly positive")


@dataclass(frozen=True)
class RachelMultiscaleTransportOutput(RachelN512Output):
    scale_weights: Tensor                         # [S], shared across samples/cells
    scale_affinities: Tensor                      # [B,S,Na,Nb]
    scale_transports: Tuple[PartialTransportOutput, ...]  # each has batch B


def fuse_partial_transports(
    transports: Sequence[PartialTransportOutput],
    weights: Tensor,
    valid_a: Tensor,
    valid_b: Tensor,
    *,
    tolerance: float = 1e-3,
) -> PartialTransportOutput:
    """Convex-combine entire augmented plans, and recompute fused diagnostics.

    Weights are normalized internally (without detaching); the model supplies
    softmax or configured non-negative weights. A negative/non-finite vector
    fails the result closed. All scale problems must be valid, including a
    configured zero-weight scale: invalid branches are never silently hidden.
    ``iteration_count`` is the maximum depth of the independently run plans,
    not a claim that the mixture itself was passed through another Sinkhorn.
    """

    plans = tuple(transports)
    if not plans or weights.ndim != 1 or weights.numel() != len(plans):
        raise ValueError("one scalar weight is required for each nonempty scale plan")
    shape = plans[0].real_transport.shape
    if len(shape) != 3 or valid_a.shape != shape[:2] or valid_b.shape != (shape[0], shape[2]):
        raise ValueError("transport and token-mask shapes disagree")
    if valid_a.dtype != torch.bool or valid_b.dtype != torch.bool:
        raise TypeError("token masks must be boolean")
    if any(plan.real_transport.shape != shape for plan in plans):
        raise ValueError("every scale must use the same token coordinates and masks")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    dtype = plans[0].real_transport.dtype
    w = weights.to(device=plans[0].real_transport.device, dtype=dtype)
    usable_weights = torch.isfinite(w).all() & (w >= 0.0).all() & (w.sum() > 0.0)
    safe_w = torch.where(usable_weights, w, torch.ones_like(w))
    w = safe_w / safe_w.sum()

    def mixture(name: str) -> Tensor:
        values = torch.stack([getattr(plan, name) for plan in plans], dim=1)
        return (values * w.reshape(1, len(plans), *([1] * (values.ndim - 2)))).sum(dim=1)

    usable = torch.stack([p.diagnostics.input_is_usable for p in plans]).all(dim=0) & usable_weights
    valid_problem = torch.stack([p.diagnostics.valid_problem for p in plans]).all(dim=0) & usable
    raw = [mixture(name) for name in ("real_transport", "dustbin_col", "dustbin_row", "dustbin_corner")]
    raw_finite = torch.stack([
        value.reshape(shape[0], -1).isfinite().all(dim=1) for value in raw
    ]).all(dim=0)
    valid_problem = valid_problem & raw_finite
    real, col, row, corner = [
        torch.where(valid_problem.reshape(shape[0], *([1] * (value.ndim - 1))), value, torch.zeros_like(value))
        for value in raw
    ]
    count_a, count_b = valid_a.sum(dim=1), valid_b.sum(dim=1)
    active_a = valid_a & valid_problem[:, None]
    active_b = valid_b & valid_problem[:, None]
    expected_bin_row = torch.where(valid_problem, count_b.to(dtype), torch.zeros_like(corner))
    expected_bin_col = torch.where(valid_problem, count_a.to(dtype), torch.zeros_like(corner))
    row_residual = torch.maximum(
        (real.sum(dim=2) + col - active_a.to(dtype)).abs().amax(dim=1),
        (row.sum(dim=1) + corner - expected_bin_row).abs(),
    )
    col_residual = torch.maximum(
        (real.sum(dim=1) + row - active_b.to(dtype)).abs().amax(dim=1),
        (col.sum(dim=1) + corner - expected_bin_col).abs(),
    )
    # Preserve evidence of a non-finite branch even though its fused output is zeroed.
    finite = raw_finite & torch.stack([p.diagnostics.finite_output for p in plans]).all(dim=0)
    iterations = torch.stack([p.diagnostics.iteration_count for p in plans]).amax(dim=0)
    diagnostics = SinkhornDiagnostics(
        valid_problem=valid_problem, input_is_usable=usable, finite_output=finite,
        converged=valid_problem & finite & (row_residual <= tolerance) & (col_residual <= tolerance),
        valid_a_count=count_a, valid_b_count=count_b,
        iteration_count=torch.where(valid_problem, iterations, torch.zeros_like(iterations)),
        row_residual_max=row_residual, col_residual_max=col_residual,
        matched_mass=real.sum(dim=(1, 2)), unmatched_a_mass=col.sum(dim=1),
        unmatched_b_mass=row.sum(dim=1),
    )
    return PartialTransportOutput(real, row, col, corner, diagnostics)


def _split_scale_transport(
    flat: PartialTransportOutput, batch: int, scales: int,
) -> Tuple[PartialTransportOutput, ...]:
    def at(value: Tensor, scale: int) -> Tensor:
        return value.reshape(batch, scales, *value.shape[1:])[:, scale]
    result = []
    for scale in range(scales):
        diagnostics = SinkhornDiagnostics(**{
            name: at(value, scale) for name, value in vars(flat.diagnostics).items()
        })
        result.append(PartialTransportOutput(
            **{name: at(getattr(flat, name), scale) for name in (
                "real_transport", "dustbin_row", "dustbin_col", "dustbin_corner",
            )}, diagnostics=diagnostics,
        ))
    return tuple(result)


class RachelMultiscaleTransportPairwise(RachelN512Pairwise):
    """Parameter-shared per-scale complementary partial-OT model."""

    def __init__(self, config: Optional[RachelMultiscaleTransportConfig] = None) -> None:
        config = config or RachelMultiscaleTransportConfig()
        if not isinstance(config, RachelMultiscaleTransportConfig):
            raise TypeError("use RachelMultiscaleTransportConfig for this model")
        super().__init__(config)
        del self.scale_gate  # Descriptor pre-fusion is absent, not an unused parameter.
        weights = torch.tensor(config.initial_scale_weights or (1.0,) * len(config.window_sizes_px))
        weights = weights / weights.sum()
        if config.learn_scale_weights:
            self.scale_logits = nn.Parameter(weights.log())
        else:
            self.register_buffer("fixed_scale_weights", weights, persistent=True)

    def normalized_scale_weights(self) -> Tensor:
        if self.config.learn_scale_weights:
            return torch.softmax(self.scale_logits, dim=0)
        return self.fixed_scale_weights / self.fixed_scale_weights.sum()

    def checkpoint_metadata(self) -> dict:
        return {"model_kind": MODEL_KIND, "model_config": asdict(self.config), "model_options": {}}

    def _encode_scale_patches(self, patches: Tensor, valid: Tensor) -> Tensor:
        batch, tokens, scales = patches.shape[:3]
        flat = patches.reshape(batch, tokens * scales, *patches.shape[3:])
        flat_valid = valid[:, :, None].expand(-1, -1, scales).reshape(batch, -1)
        if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
            encoded = activation_checkpoint(
                self.patch_encoder, flat, flat_valid,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            encoded = self.patch_encoder(flat, flat_valid)
        return encoded.reshape(batch, tokens, scales, -1)

    def forward(
        self, mask_a: Tensor, mask_b: Tensor, points_rc_a: Tensor, points_rc_b: Tensor,
        contour_valid_a: Tensor, contour_valid_b: Tensor,
    ) -> RachelMultiscaleTransportOutput:
        cfg = self.config
        mask_a, mask_b = _validate_binary_masks(mask_a, mask_b, cfg.canvas_size, validate_values=cfg.validate_runtime_inputs)
        points_rc_a, contour_valid_a = _validate_contours(
            points_rc_a, contour_valid_a, batch_size=mask_a.shape[0], canvas_size=cfg.canvas_size,
            contour_cap=cfg.contour_cap, name="A", validate_values=cfg.validate_runtime_inputs,
        )
        points_rc_b, contour_valid_b = _validate_contours(
            points_rc_b, contour_valid_b, batch_size=mask_a.shape[0], canvas_size=cfg.canvas_size,
            contour_cap=cfg.contour_cap, name="B", validate_values=cfg.validate_runtime_inputs,
        )
        coarse = self.coarse(
            F.interpolate(mask_a, size=(cfg.coarse_size, cfg.coarse_size), mode="nearest"),
            F.interpolate(mask_b, size=(cfg.coarse_size, cfg.coarse_size), mode="nearest"),
        )
        encoded_a = self._encode_scale_patches(self.patch_sampler(mask_a, points_rc_a, contour_valid_a), contour_valid_a)
        encoded_b = self._encode_scale_patches(self.patch_sampler(mask_b, points_rc_b, contour_valid_b), contour_valid_b)
        batch, na, scales, dim = encoded_a.shape
        nb = encoded_b.shape[1]

        def expand(value: Tensor) -> Tensor:
            return value[:, None].expand(batch, scales, *value.shape[1:]).reshape(batch * scales, *value.shape[1:])

        flat_valid_a, flat_valid_b = expand(contour_valid_a), expand(contour_valid_b)
        context_a, context_b = self.context(
            encoded_a.permute(0, 2, 1, 3).reshape(batch * scales, na, dim),
            encoded_b.permute(0, 2, 1, 3).reshape(batch * scales, nb, dim),
            flat_valid_a, flat_valid_b, expand(points_rc_a), expand(points_rc_b), cfg.canvas_size,
        )
        primal_a = F.normalize(self.primal(context_a), dim=2, eps=1e-6)
        primal_b = F.normalize(self.primal(context_b), dim=2, eps=1e-6)
        dual_a = F.normalize(self.dual(context_a), dim=2, eps=1e-6)
        dual_b = F.normalize(self.dual(context_b), dim=2, eps=1e-6)
        flat_affinity = 0.5 * (
            torch.matmul(primal_a, dual_b.transpose(1, 2))
            + torch.matmul(dual_a, primal_b.transpose(1, 2))
        )
        flat_transport = dustbin_sinkhorn(
            flat_affinity, flat_valid_a, flat_valid_b, dustbin_score=self.dustbin_score,
            temperature=cfg.matcher_temperature, num_iterations=cfg.sinkhorn_iterations,
            tolerance=cfg.sinkhorn_tolerance,
            checkpoint_iterations=cfg.activation_checkpointing and self.training and torch.is_grad_enabled(),
        )
        scale_transports = _split_scale_transport(flat_transport, batch, scales)
        weights = self.normalized_scale_weights()
        transport = fuse_partial_transports(
            scale_transports, weights, contour_valid_a, contour_valid_b, tolerance=cfg.sinkhorn_tolerance,
        )
        scale_affinities = flat_affinity.reshape(batch, scales, na, nb)
        affinity = (scale_affinities * weights.to(flat_affinity.dtype)[None, :, None, None]).sum(dim=1)
        translation, dispersion, mass = self._translation(transport.real_transport, points_rc_a, points_rc_b)
        weighted_affinity_numerator = None
        if cfg.affinity_readout == "within_scale_product":
            # Readout-only ablation: retain each scale's Q_s * A_s association.
            # The mixed assignment and every layout/mass diagnostic are untouched.
            scale_assignments = flat_transport.real_transport.reshape(batch, scales, na, nb)
            weighted_affinity_numerator = (
                scale_assignments * scale_affinities
                * weights.to(scale_assignments.dtype)[None, :, None, None]
            ).sum(dim=1)
        local_logit, _ = self.local_head(
            affinity, transport.real_transport, transport.dustbin_col, transport.dustbin_row,
            contour_valid_a, contour_valid_b, dispersion, cfg.canvas_size,
            weighted_affinity_numerator=weighted_affinity_numerator,
        )
        fused_logit = self.fusion(torch.stack((
            coarse.logit, local_logit, coarse.logit * local_logit, torch.abs(coarse.logit - local_logit),
        ), dim=1)).squeeze(1)
        finite = (
            coarse.valid_problem & transport.diagnostics.valid_problem & transport.diagnostics.finite_output
            & torch.isfinite(local_logit) & torch.isfinite(fused_logit)
        )
        decision_valid = finite & transport.diagnostics.converged
        fused_logit = torch.where(finite, fused_logit, torch.zeros_like(fused_logit))
        local_logit = torch.where(finite, local_logit, torch.zeros_like(local_logit))
        return RachelMultiscaleTransportOutput(
            fused_logit=fused_logit, fused_probability=torch.sigmoid(fused_logit),
            coarse_logit=coarse.logit, coarse_probability=coarse.probability,
            local_logit=local_logit, local_probability=torch.sigmoid(local_logit),
            affinity=affinity, assignment=transport.real_transport,
            unmatched_a=transport.dustbin_col, unmatched_b=transport.dustbin_row,
            translation_hat_rc=translation,
            translation_hat_xy_cartesian=torch.stack((translation[:, 1], -translation[:, 0]), dim=1),
            translation_dispersion_px=dispersion, matched_mass=mass,
            # Diagnostic pooled token features only: matching used each scale separately.
            token_features_a=(context_a.reshape(batch, scales, na, dim) * weights[None, :, None, None]).sum(dim=1),
            token_features_b=(context_b.reshape(batch, scales, nb, dim) * weights[None, :, None, None]).sum(dim=1),
            coarse=coarse, transport=transport, training_valid=finite, decision_valid=decision_valid,
            scale_weights=weights, scale_affinities=scale_affinities, scale_transports=scale_transports,
        )


def create_multiscale_transport_model(
    model_config: Optional[Union[RachelMultiscaleTransportConfig, Mapping]] = None,
    *, model_state_dict: Optional[Mapping[str, Tensor]] = None,
) -> RachelMultiscaleTransportPairwise:
    """Restore from model_config metadata and an optional strict state dict."""

    if model_config is None:
        config = RachelMultiscaleTransportConfig()
    elif isinstance(model_config, RachelMultiscaleTransportConfig):
        config = model_config
    else:
        values = dict(model_config)
        for name in ("window_sizes_px", "initial_scale_weights"):
            if name in values:
                values[name] = tuple(values[name])
        config = RachelMultiscaleTransportConfig(**values)
    model = RachelMultiscaleTransportPairwise(config)
    if model_state_dict is not None:
        state = dict(model_state_dict)
        # The physical grid is defined by metadata, not by stale buffer contents.
        state["patch_sampler.offsets_rc"] = model.patch_sampler.offsets_rc
        model.load_state_dict(state, strict=True)
    return model


def initialize_multiscale_transport_from_base(
    base_config: RachelN512Config, base_state: Mapping[str, Tensor], *,
    config: Optional[RachelMultiscaleTransportConfig] = None,
) -> RachelMultiscaleTransportPairwise:
    """Keep all base learned weights except its removed descriptor scale gate.

    Newly initialized state is scale_logits (learned) or fixed_scale_weights
    (buffer). The sampling grid is always regenerated, even if its shape has
    not changed. Any other missing, unexpected or shape-mismatched key fails.
    """

    if config is None:
        values = asdict(base_config)
        values["window_sizes_px"] = (7.0, 16.0, 32.0, 64.0)
        config = RachelMultiscaleTransportConfig(**values)
    model = RachelMultiscaleTransportPairwise(config)
    state = dict(base_state)
    for name in ("scale_gate.weight", "scale_gate.bias"):
        if name not in state:
            raise ValueError("base state is missing " + name)
        del state[name]
    state["patch_sampler.offsets_rc"] = model.patch_sampler.offsets_rc
    weight_key = "scale_logits" if config.learn_scale_weights else "fixed_scale_weights"
    state[weight_key] = model.state_dict()[weight_key]
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "MODEL_KIND", "RachelMultiscaleTransportConfig", "RachelMultiscaleTransportOutput",
    "RachelMultiscaleTransportPairwise", "fuse_partial_transports",
    "create_multiscale_transport_model", "initialize_multiscale_transport_from_base",
]
