"""Learnable ordered contour-patch matching for Pairwise v0.2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .optimal_transport import PartialTransportOutput, dustbin_sinkhorn


class MatcherMode(str, Enum):
    """Local assignment ablations exposed by the same model contract."""

    DUAL_SOFTMAX = "dual_softmax"
    DUSTBIN_SINKHORN = "dustbin_sinkhorn"


class SinkhornTrainingPolicy(str, Enum):
    """How a finite but not-yet-converged transport participates in training.

    Production decisions always require convergence.  This policy affects only
    gradient-bearing training evidence: the default keeps a finite transport in
    the graph and lets the loss penalize its marginal residual, while the strict
    ablation reproduces converged-only training.
    """

    FINITE_WITH_RESIDUAL = "finite_with_residual"
    CONVERGED_ONLY = "converged_only"


DEFAULT_MATCHER_TEMPERATURE = 0.25
DEFAULT_SINKHORN_ITERATIONS = 100


@dataclass(frozen=True)
class LocalMatcherOutput:
    """Local evidence; assignment is not itself an adjacency prediction."""

    logit: Tensor
    probability: Tensor
    affinity: Tensor
    assignment: Tensor
    unmatched_a: Tensor
    unmatched_b: Tensor
    evidence: Tensor
    token_features_a: Tensor
    token_features_b: Tensor
    finite_problem: Tensor
    training_valid: Tensor
    decision_valid: Tensor
    valid_problem: Tensor
    transport_converged: Tensor
    row_residual_max: Tensor
    col_residual_max: Tensor
    transport: Optional[PartialTransportOutput]
    matcher_mode: str
    sinkhorn_training_policy: str


def _sinusoidal_positions(length: int, dimension: int, value: Tensor) -> Tensor:
    position = torch.arange(length, dtype=value.dtype, device=value.device)[:, None]
    even = torch.arange(0, dimension, 2, dtype=value.dtype, device=value.device)
    scale = torch.exp(-math.log(10000.0) * even / max(dimension, 1))
    result = torch.zeros((length, dimension), dtype=value.dtype, device=value.device)
    result[:, 0::2] = torch.sin(position * scale)
    if dimension > 1:
        result[:, 1::2] = torch.cos(position * scale[: result[:, 1::2].shape[1]])
    return result


class SharedPatchEncoder(nn.Module):
    """Encode mask/SDF/boundary-gradient patches with one shared CNN."""

    def __init__(
        self,
        input_channels: int,
        feature_dim: int = 96,
        widths: Tuple[int, ...] = (24, 48, 64),
    ) -> None:
        super().__init__()
        if input_channels <= 0 or feature_dim <= 0:
            raise ValueError("patch input_channels and feature_dim must be positive")
        if not widths or min(widths) <= 0:
            raise ValueError("patch widths must contain positive values")
        layers = []
        previous = input_channels
        for width in widths:
            layers.extend(
                [
                    nn.Conv2d(previous, width, 3, padding=1, bias=False),
                    nn.GroupNorm(1, width),
                    nn.SiLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            previous = width
        self.cnn = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(previous, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.input_channels = input_channels
        self.downsample_factor = 2 ** len(widths)

    def forward(self, patches: Tensor, token_mask: Tensor) -> Tensor:
        if patches.ndim != 5:
            raise ValueError("patches must have shape [B, L, C, H, W]")
        if patches.shape[2] != self.input_channels:
            raise ValueError("patch channel count does not match encoder config")
        if min(patches.shape[-2:]) < self.downsample_factor:
            raise ValueError("patches are too small for the encoder")
        batch, length = patches.shape[:2]
        safe = torch.where(
            token_mask[:, :, None, None, None], patches, torch.zeros_like(patches)
        )
        flat = safe.reshape(batch * length, *safe.shape[2:])
        encoded = self.projection(self.cnn(flat)).reshape(batch, length, -1)
        return torch.where(token_mask[:, :, None], encoded, torch.zeros_like(encoded))


class OrderedSelfCrossContext(nn.Module):
    """Shared ordered self-attention followed by shared cross-attention."""

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        ff_dim: int = 192,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.self_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(feature_dim)
        self.cross_norm = nn.LayerNorm(feature_dim)
        self.ff_norm = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, feature_dim),
        )

    @staticmethod
    def _safe_mask(mask: Tensor) -> Tensor:
        safe = mask.clone()
        empty = ~safe.any(dim=1)
        if empty.any().item():
            safe[empty, 0] = True
        return safe

    def _self(self, value: Tensor, mask: Tensor) -> Tensor:
        safe_mask = self._safe_mask(mask)
        normalized = self.self_norm(value)
        update, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        result = value + update
        return torch.where(mask[:, :, None], result, torch.zeros_like(result))

    def _cross(
        self,
        query: Tensor,
        query_mask: Tensor,
        source: Tensor,
        source_mask: Tensor,
    ) -> Tensor:
        safe_source = self._safe_mask(source_mask)
        normalized_query = self.cross_norm(query)
        normalized_source = self.cross_norm(source)
        update, _ = self.cross_attention(
            normalized_query,
            normalized_source,
            normalized_source,
            key_padding_mask=~safe_source,
            need_weights=False,
        )
        result = query + update
        result = result + self.feed_forward(self.ff_norm(result))
        return torch.where(query_mask[:, :, None], result, torch.zeros_like(result))

    def forward(
        self, a: Tensor, b: Tensor, mask_a: Tensor, mask_b: Tensor
    ) -> Tuple[Tensor, Tensor]:
        positions_a = _sinusoidal_positions(a.shape[1], a.shape[2], a)
        positions_b = _sinusoidal_positions(b.shape[1], b.shape[2], b)
        a = torch.where(mask_a[:, :, None], a + positions_a[None], a)
        b = torch.where(mask_b[:, :, None], b + positions_b[None], b)
        self_a = self._self(a, mask_a)
        self_b = self._self(b, mask_b)
        # Both calls share parameters and use the pre-cross states.  No side is
        # privileged by an in-place sequential update.
        return (
            self._cross(self_a, mask_a, self_b, mask_b),
            self._cross(self_b, mask_b, self_a, mask_a),
        )


def _validated_token_mask(
    value: Tensor, batch: int, length: int, name: str, device: torch.device
) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise TypeError("{} must be a bool torch.Tensor".format(name))
    if tuple(value.shape) != (batch, length):
        raise ValueError("{} must have shape ({}, {})".format(name, batch, length))
    return value.to(device=device)


class OrderedLocalMatcher(nn.Module):
    """Patch encoder, primal/dual affinity, assignment, and local score head."""

    def __init__(
        self,
        input_channels: int,
        feature_dim: int = 96,
        num_heads: int = 4,
        ff_dim: int = 192,
        matcher_mode: MatcherMode = MatcherMode.DUSTBIN_SINKHORN,
        matcher_temperature: float = DEFAULT_MATCHER_TEMPERATURE,
        sinkhorn_iterations: int = DEFAULT_SINKHORN_ITERATIONS,
        sinkhorn_tolerance: float = 1e-3,
        require_sinkhorn_convergence: bool = True,
        sinkhorn_training_policy: SinkhornTrainingPolicy = (
            SinkhornTrainingPolicy.FINITE_WITH_RESIDUAL
        ),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            matcher_mode = MatcherMode(getattr(matcher_mode, "value", matcher_mode))
        except ValueError as exc:
            raise ValueError("unsupported matcher_mode") from exc
        if not math.isfinite(matcher_temperature) or matcher_temperature <= 0.0:
            raise ValueError("matcher_temperature must be finite and positive")
        if (
            isinstance(sinkhorn_iterations, bool)
            or not isinstance(sinkhorn_iterations, int)
            or sinkhorn_iterations <= 0
        ):
            raise ValueError("sinkhorn_iterations must be a positive integer")
        if (
            isinstance(sinkhorn_tolerance, bool)
            or not math.isfinite(float(sinkhorn_tolerance))
            or float(sinkhorn_tolerance) < 0.0
        ):
            raise ValueError("sinkhorn_tolerance must be finite and non-negative")
        if type(require_sinkhorn_convergence) is not bool:
            raise TypeError("require_sinkhorn_convergence must be bool")
        if not require_sinkhorn_convergence:
            raise ValueError(
                "production decisions must require Sinkhorn convergence; use "
                "sinkhorn_training_policy for a finite-plan training ablation"
            )
        try:
            sinkhorn_training_policy = SinkhornTrainingPolicy(
                getattr(sinkhorn_training_policy, "value", sinkhorn_training_policy)
            )
        except ValueError as exc:
            raise ValueError("unsupported sinkhorn_training_policy") from exc
        self.patch_encoder = SharedPatchEncoder(input_channels, feature_dim)
        self.context = OrderedSelfCrossContext(feature_dim, num_heads, ff_dim, dropout)
        self.primal = nn.Linear(feature_dim, feature_dim, bias=False)
        self.dual = nn.Linear(feature_dim, feature_dim, bias=False)
        # Evidence: matched/min-count, mean coverage, coverage imbalance,
        # assignment-weighted affinity, and best affinity.
        self.local_head = nn.Sequential(
            nn.Linear(5, 32), nn.SiLU(inplace=True), nn.Linear(32, 1)
        )
        self.dustbin_score = nn.Parameter(torch.tensor(0.0))
        self.matcher_mode = matcher_mode
        self.matcher_temperature = float(matcher_temperature)
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_tolerance = sinkhorn_tolerance
        self.require_sinkhorn_convergence = True
        self.sinkhorn_training_policy = sinkhorn_training_policy

    def _validate_inputs(
        self,
        patches_a: Tensor,
        patches_b: Tensor,
        mask_a: Tensor,
        mask_b: Tensor,
        correspondence_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(patches_a, Tensor) or not isinstance(patches_b, Tensor):
            raise TypeError("local patches must be torch.Tensor values")
        if patches_a.ndim != 5 or patches_b.ndim != 5:
            raise ValueError("local patches must have shape [B, L, C, H, W]")
        if patches_a.shape[0] != patches_b.shape[0]:
            raise ValueError("local A/B batch sizes differ")
        if patches_a.shape[0] < 1 or patches_a.shape[1] < 1 or patches_b.shape[1] < 1:
            raise ValueError("local batch and padded sequence lengths must be positive")
        if patches_a.shape[2:] != patches_b.shape[2:]:
            raise ValueError("local A/B channel and patch shapes differ")
        if patches_a.device != patches_b.device or patches_a.dtype != patches_b.dtype:
            raise ValueError("local A/B patches must share device and dtype")
        if not patches_a.is_floating_point() or not patches_b.is_floating_point():
            raise TypeError("local patches must use a floating-point dtype")
        batch = patches_a.shape[0]
        mask_a = _validated_token_mask(
            mask_a, batch, patches_a.shape[1], "token_mask_a", patches_a.device
        )
        mask_b = _validated_token_mask(
            mask_b, batch, patches_b.shape[1], "token_mask_b", patches_b.device
        )
        if correspondence_mask is None:
            correspondence = torch.ones(
                (batch, patches_a.shape[1], patches_b.shape[1]),
                dtype=torch.bool,
                device=patches_a.device,
            )
        else:
            if (
                not isinstance(correspondence_mask, Tensor)
                or correspondence_mask.dtype != torch.bool
            ):
                raise TypeError("correspondence_mask must be a bool torch.Tensor")
            expected = (batch, patches_a.shape[1], patches_b.shape[1])
            if tuple(correspondence_mask.shape) != expected:
                raise ValueError(
                    "correspondence_mask must have shape {}".format(expected)
                )
            correspondence = correspondence_mask.to(device=patches_a.device)
        finite_a = torch.isfinite(patches_a).flatten(2).all(dim=2)
        finite_b = torch.isfinite(patches_b).flatten(2).all(dim=2)
        usable = (~mask_a | finite_a).all(dim=1) & (~mask_b | finite_b).all(dim=1)
        usable = usable & mask_a.any(dim=1) & mask_b.any(dim=1)
        correspondence = correspondence & mask_a[:, :, None] & mask_b[:, None, :]
        usable = usable & correspondence.flatten(1).any(dim=1)
        effective_a = mask_a & usable[:, None]
        effective_b = mask_b & usable[:, None]
        effective_correspondence = correspondence & usable[:, None, None]
        return effective_a, effective_b, effective_correspondence, usable

    @staticmethod
    def _dual_softmax(
        affinity: Tensor,
        mask_a: Tensor,
        mask_b: Tensor,
        correspondence_mask: Tensor,
        temperature: float,
    ) -> Tensor:
        candidate = mask_a[:, :, None] & mask_b[:, None, :] & correspondence_mask
        logits = affinity / temperature
        # Finite sentinel avoids all--inf softmax rows for padded/invalid
        # tokens; candidate multiplication removes their exact output mass.
        masked = torch.where(candidate, logits, torch.full_like(logits, -1e4))
        return torch.softmax(masked, dim=2) * torch.softmax(masked, dim=1) * candidate

    @staticmethod
    def _evidence(
        affinity: Tensor,
        assignment: Tensor,
        mask_a: Tensor,
        mask_b: Tensor,
        correspondence_mask: Tensor,
        valid: Tensor,
    ) -> Tensor:
        dtype = affinity.dtype
        count_a = mask_a.sum(dim=1).to(dtype).clamp_min(1.0)
        count_b = mask_b.sum(dim=1).to(dtype).clamp_min(1.0)
        mass = assignment.sum(dim=(1, 2))
        row_coverage = assignment.sum(dim=2).sum(dim=1) / count_a
        col_coverage = assignment.sum(dim=1).sum(dim=1) / count_b
        mean_coverage = (row_coverage + col_coverage) * 0.5
        coverage_difference = torch.abs(row_coverage - col_coverage)
        candidate = mask_a[:, :, None] & mask_b[:, None, :] & correspondence_mask
        safe_affinity = torch.where(candidate, affinity, torch.zeros_like(affinity))
        weighted_affinity = (assignment * safe_affinity).sum(
            dim=(1, 2)
        ) / mass.clamp_min(1e-6)
        best_affinity = (
            torch.where(candidate, affinity, torch.full_like(affinity, -1e4))
            .flatten(1)
            .amax(dim=1)
        )
        evidence = torch.stack(
            [
                mass / torch.minimum(count_a, count_b),
                mean_coverage,
                coverage_difference,
                weighted_affinity,
                best_affinity,
            ],
            dim=1,
        )
        return torch.where(valid[:, None], evidence, torch.zeros_like(evidence))

    def forward(
        self,
        patches_a: Tensor,
        patches_b: Tensor,
        token_mask_a: Tensor,
        token_mask_b: Tensor,
        correspondence_mask: Optional[Tensor] = None,
    ) -> LocalMatcherOutput:
        mask_a, mask_b, correspondence, input_valid = self._validate_inputs(
            patches_a,
            patches_b,
            token_mask_a,
            token_mask_b,
            correspondence_mask,
        )
        safe_a = torch.nan_to_num(patches_a, nan=0.0, posinf=0.0, neginf=0.0)
        safe_b = torch.nan_to_num(patches_b, nan=0.0, posinf=0.0, neginf=0.0)
        checkpoint_activations = (
            self.matcher_mode is MatcherMode.DUSTBIN_SINKHORN
            and self.training
            and torch.is_grad_enabled()
        )
        if checkpoint_activations:
            # The patch tensors do not require gradients; non-reentrant
            # checkpointing is therefore required so parameter gradients from
            # the shared encoder are retained.  The encoder has no stochastic
            # layers, so preserving RNG state would add overhead without
            # changing the formal computation.
            encoded_a = activation_checkpoint(
                self.patch_encoder,
                safe_a,
                mask_a,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            encoded_b = activation_checkpoint(
                self.patch_encoder,
                safe_b,
                mask_b,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            encoded_a = self.patch_encoder(safe_a, mask_a)
            encoded_b = self.patch_encoder(safe_b, mask_b)
        context_a, context_b = self.context(encoded_a, encoded_b, mask_a, mask_b)
        primal_a = F.normalize(self.primal(context_a), dim=2, eps=1e-6)
        primal_b = F.normalize(self.primal(context_b), dim=2, eps=1e-6)
        dual_a = F.normalize(self.dual(context_a), dim=2, eps=1e-6)
        dual_b = F.normalize(self.dual(context_b), dim=2, eps=1e-6)
        # Complementary primal-dual affinity; transposes exactly under A/B swap.
        affinity = 0.5 * (
            torch.matmul(primal_a, dual_b.transpose(1, 2))
            + torch.matmul(dual_a, primal_b.transpose(1, 2))
        )
        transport = None
        if self.matcher_mode is MatcherMode.DUAL_SOFTMAX:
            assignment = self._dual_softmax(
                affinity,
                mask_a,
                mask_b,
                correspondence,
                self.matcher_temperature,
            )
            unmatched_a = torch.clamp(1.0 - assignment.sum(dim=2), min=0.0) * mask_a
            unmatched_b = torch.clamp(1.0 - assignment.sum(dim=1), min=0.0) * mask_b
            converged = input_valid.clone()
            finite_problem = input_valid & torch.isfinite(assignment).flatten(1).all(
                dim=1
            )
            training_valid = finite_problem
            decision_valid = finite_problem
            row_residual_max = affinity.new_zeros((affinity.shape[0],))
            col_residual_max = affinity.new_zeros((affinity.shape[0],))
        else:
            transport_affinity = torch.where(
                correspondence,
                affinity,
                torch.full_like(affinity, -torch.inf),
            )
            transport = dustbin_sinkhorn(
                transport_affinity,
                mask_a,
                mask_b,
                dustbin_score=self.dustbin_score,
                temperature=self.matcher_temperature,
                num_iterations=self.sinkhorn_iterations,
                tolerance=self.sinkhorn_tolerance,
                checkpoint_iterations=checkpoint_activations,
            )
            assignment = transport.real_transport
            unmatched_a = transport.dustbin_col
            unmatched_b = transport.dustbin_row
            converged = transport.diagnostics.converged
            finite_problem = (
                input_valid
                & transport.diagnostics.valid_problem
                & transport.diagnostics.finite_output
            )
            decision_valid = finite_problem & converged
            if self.sinkhorn_training_policy is SinkhornTrainingPolicy.CONVERGED_ONLY:
                training_valid = decision_valid
            else:
                training_valid = finite_problem
            row_residual_max = transport.diagnostics.row_residual_max
            col_residual_max = transport.diagnostics.col_residual_max
        # A finite non-converged plan stays score-bearing under the default
        # training policy.  ``decision_valid`` remains false, so this diagnostic
        # score can never silently become a production adjacency decision.
        evidence = self._evidence(
            affinity,
            assignment,
            mask_a,
            mask_b,
            correspondence,
            training_valid,
        )
        raw_logit = self.local_head(evidence).squeeze(1)
        logit = torch.where(training_valid, raw_logit, torch.zeros_like(raw_logit))
        return LocalMatcherOutput(
            logit=logit,
            probability=torch.sigmoid(logit),
            affinity=affinity,
            assignment=assignment,
            unmatched_a=unmatched_a,
            unmatched_b=unmatched_b,
            evidence=evidence,
            token_features_a=context_a,
            token_features_b=context_b,
            finite_problem=finite_problem,
            training_valid=training_valid,
            decision_valid=decision_valid,
            # Backwards-compatible safety alias: callers that predate the split
            # continue to receive the fail-closed production validity.
            valid_problem=decision_valid,
            transport_converged=converged,
            row_residual_max=row_residual_max,
            col_residual_max=col_residual_max,
            transport=transport,
            matcher_mode=self.matcher_mode.value,
            sinkhorn_training_policy=self.sinkhorn_training_policy.value,
        )


__all__ = [
    "LocalMatcherOutput",
    "MatcherMode",
    "OrderedLocalMatcher",
    "OrderedSelfCrossContext",
    "SharedPatchEncoder",
    "SinkhornTrainingPolicy",
]
