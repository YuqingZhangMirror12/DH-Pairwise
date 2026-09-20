"""Frozen-backbone scalar readouts for dustbin transport quality.

This module is an additive research ablation.  It does not change
``OrderedLocalMatcher`` or any existing checkpoint score.  A frozen arc logit
``l`` is optionally adjusted as ``l + beta * q + intercept`` before the same
arc-within-direction and direction-within-pair log-mean-exp aggregation used by
Pairwise v0.2.

Two predeclared transport statistics are exposed:

``mass_only``
    Symmetric top-k real (non-dustbin) mass.  This controls for merely exposing
    a local high-mass match instead of the current whole-sequence mass mean.

``entropy_aware``
    The same real mass multiplied by one minus normalized assignment entropy.
    A confident dustbin route scores zero, a diffuse real assignment is
    discounted, and a concentrated real assignment scores highly.

Only ``beta`` and ``intercept`` are fit.  Assignment, patch encoder, affinity,
dustbin score, and the existing five-feature local head remain frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .pairwise import ArcPoolingConfig, ArcPoolingMode


READOUT_BASELINE = "frozen_exact_baseline"
READOUT_MASS_ONLY = "frozen_exact_plus_topk_mass"
READOUT_ENTROPY_AWARE = "frozen_exact_plus_entropy_quality"
READOUT_NAMES = (READOUT_BASELINE, READOUT_MASS_ONLY, READOUT_ENTROPY_AWARE)


@dataclass(frozen=True)
class ArcTransportQuality:
    """Per-arc transport scores, each with shape ``[N]``."""

    mass_only: Tensor
    entropy_aware: Tensor

    def __post_init__(self) -> None:
        if self.mass_only.ndim != 1 or not self.mass_only.is_floating_point():
            raise TypeError("mass_only must be floating-point [N]")
        if tuple(self.entropy_aware.shape) != tuple(self.mass_only.shape):
            raise ValueError("transport-quality vectors differ in shape")
        if not self.entropy_aware.is_floating_point():
            raise TypeError("entropy_aware must be floating point")


@dataclass(frozen=True)
class FrozenArcDataset:
    """Detached arc evidence and pair ownership for a frozen population."""

    arc_logit: Tensor
    mass_only: Tensor
    entropy_aware: Tensor
    arc_valid: Tensor
    sample_index: Tensor
    direction_index: Tensor
    geometry_valid: Tensor
    label: Optional[Tensor] = None

    def __post_init__(self) -> None:
        arc_shape = tuple(self.arc_logit.shape)
        if self.arc_logit.ndim != 1 or not self.arc_logit.is_floating_point():
            raise TypeError("arc_logit must be floating-point [N]")
        for name in ("mass_only", "entropy_aware"):
            value = getattr(self, name)
            if tuple(value.shape) != arc_shape or not value.is_floating_point():
                raise TypeError(name + " must be floating-point [N]")
        if self.arc_valid.dtype != torch.bool or tuple(self.arc_valid.shape) != arc_shape:
            raise TypeError("arc_valid must be bool [N]")
        for name in ("sample_index", "direction_index"):
            value = getattr(self, name)
            if value.dtype != torch.long or tuple(value.shape) != arc_shape:
                raise TypeError(name + " must be int64 [N]")
        if self.geometry_valid.ndim != 1 or self.geometry_valid.dtype != torch.bool:
            raise TypeError("geometry_valid must be bool [B]")
        if self.geometry_valid.numel() < 1:
            raise ValueError("a frozen population cannot be empty")
        if self.sample_index.numel() and (
            (self.sample_index < 0)
            | (self.sample_index >= self.geometry_valid.numel())
        ).any().item():
            raise ValueError("sample_index is out of range")
        if self.direction_index.numel() and (
            (self.direction_index < 0) | (self.direction_index >= 4)
        ).any().item():
            raise ValueError("direction_index is out of range")
        if self.label is not None and (
            self.label.dtype != torch.bool
            or tuple(self.label.shape) != tuple(self.geometry_valid.shape)
        ):
            raise TypeError("label must be bool [B] when present")

    @property
    def pair_count(self) -> int:
        return int(self.geometry_valid.numel())

    def feature(self, readout_name: str) -> Tensor:
        if readout_name == READOUT_BASELINE:
            return torch.zeros_like(self.arc_logit)
        if readout_name == READOUT_MASS_ONLY:
            return self.mass_only
        if readout_name == READOUT_ENTROPY_AWARE:
            return self.entropy_aware
        raise KeyError(readout_name)

    def to(self, device: torch.device, *, dtype: Optional[torch.dtype] = None) -> "FrozenArcDataset":
        floating_dtype = self.arc_logit.dtype if dtype is None else dtype
        return FrozenArcDataset(
            arc_logit=self.arc_logit.to(device=device, dtype=floating_dtype),
            mass_only=self.mass_only.to(device=device, dtype=floating_dtype),
            entropy_aware=self.entropy_aware.to(device=device, dtype=floating_dtype),
            arc_valid=self.arc_valid.to(device=device),
            sample_index=self.sample_index.to(device=device),
            direction_index=self.direction_index.to(device=device),
            geometry_valid=self.geometry_valid.to(device=device),
            label=None if self.label is None else self.label.to(device=device),
        )


@dataclass(frozen=True)
class FrozenReadoutParameters:
    name: str
    beta: float
    intercept: float
    train_loss: float
    optimizer_steps: int

    def __post_init__(self) -> None:
        if self.name not in READOUT_NAMES:
            raise ValueError("unknown frozen readout name")
        for name in ("beta", "intercept", "train_loss"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(name + " must be finite")
        if self.optimizer_steps < 0:
            raise ValueError("optimizer_steps must be non-negative")

    def to_dict(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "beta": float(self.beta),
            "intercept": float(self.intercept),
            "train_loss": float(self.train_loss),
            "optimizer_steps": int(self.optimizer_steps),
        }


@dataclass(frozen=True)
class FrozenReadoutOutput:
    pair_logit: Tensor
    pair_probability: Tensor
    pair_valid: Tensor
    direction_logits: Tensor
    direction_valid: Tensor
    best_direction_index: Tensor


def _validated_bool_mask(value: Tensor, shape: tuple, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise TypeError(name + " must be a bool torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError("{} must have shape {}".format(name, shape))
    return value


def _topk_token_mean(value: Tensor, valid: Tensor, top_k: int) -> Tensor:
    if value.ndim != 2 or tuple(valid.shape) != tuple(value.shape):
        raise ValueError("top-k token inputs must share shape [N,L]")
    width = min(top_k, int(value.shape[1]))
    masked = torch.where(valid, value, torch.full_like(value, -torch.inf))
    selected = torch.topk(masked, width, dim=1, sorted=False).values
    finite = torch.isfinite(selected)
    total = torch.where(finite, selected, torch.zeros_like(selected)).sum(dim=1)
    count = finite.sum(dim=1).clamp_min(1).to(value.dtype)
    return total / count


def _side_transport_quality(
    real_transport: Tensor,
    dustbin_mass: Tensor,
    token_mask: Tensor,
    edge_mask: Tensor,
    *,
    top_k: int,
    epsilon: float,
) -> tuple:
    """Return top-k real mass and entropy-discounted mass for one side."""

    allowed = token_mask[:, :, None] & edge_mask
    real = torch.where(allowed, real_transport, torch.zeros_like(real_transport))
    dustbin = torch.where(token_mask, dustbin_mass, torch.zeros_like(dustbin_mass))
    real_mass = real.sum(dim=2)
    total_mass = real_mass + dustbin
    safe_total = total_mass.clamp_min(epsilon)
    real_probability = real / safe_total[:, :, None]
    dustbin_probability = dustbin / safe_total

    real_entropy = -torch.where(
        real_probability > 0.0,
        real_probability * torch.log(real_probability.clamp_min(epsilon)),
        torch.zeros_like(real_probability),
    ).sum(dim=2)
    dustbin_entropy = -torch.where(
        dustbin_probability > 0.0,
        dustbin_probability * torch.log(dustbin_probability.clamp_min(epsilon)),
        torch.zeros_like(dustbin_probability),
    )
    support = edge_mask.sum(dim=2) + 1
    normalizer = torch.log(support.to(real.dtype).clamp_min(1.0))
    entropy = real_entropy + dustbin_entropy
    normalized_entropy = torch.where(
        support > 1,
        entropy / normalizer.clamp_min(epsilon),
        torch.zeros_like(entropy),
    ).clamp(min=0.0, max=1.0)
    real_fraction = (real_mass / safe_total).clamp(min=0.0, max=1.0)
    usable_token = token_mask & edge_mask.any(dim=2) & (total_mass > epsilon)
    mass_only = _topk_token_mean(real_fraction, usable_token, top_k)
    entropy_aware = _topk_token_mean(
        real_fraction * (1.0 - normalized_entropy), usable_token, top_k
    )
    return mass_only, entropy_aware


def transport_quality_features(
    assignment: Tensor,
    unmatched_a: Tensor,
    unmatched_b: Tensor,
    token_mask_a: Tensor,
    token_mask_b: Tensor,
    correspondence_mask: Optional[Tensor] = None,
    *,
    top_k: int = 3,
    epsilon: float = 1e-8,
) -> ArcTransportQuality:
    """Compute differentiable symmetric transport-quality features per arc."""

    if not isinstance(assignment, Tensor) or assignment.ndim != 3:
        raise ValueError("assignment must have shape [N,La,Lb]")
    if not assignment.is_floating_point():
        raise TypeError("assignment must be floating point")
    count, length_a, length_b = assignment.shape
    for name, value, shape in (
        ("unmatched_a", unmatched_a, (count, length_a)),
        ("unmatched_b", unmatched_b, (count, length_b)),
    ):
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError(name + " must be floating point")
        if tuple(value.shape) != shape:
            raise ValueError("{} must have shape {}".format(name, shape))
    mask_a = _validated_bool_mask(token_mask_a, (count, length_a), "token_mask_a")
    mask_b = _validated_bool_mask(token_mask_b, (count, length_b), "token_mask_b")
    if correspondence_mask is None:
        edges = mask_a[:, :, None] & mask_b[:, None, :]
    else:
        edges = _validated_bool_mask(
            correspondence_mask,
            (count, length_a, length_b),
            "correspondence_mask",
        )
        edges = edges & mask_a[:, :, None] & mask_b[:, None, :]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if (
        isinstance(epsilon, bool)
        or not math.isfinite(float(epsilon))
        or not 0.0 < epsilon < 1.0
    ):
        raise ValueError("epsilon must be finite in (0,1)")
    mass_a, entropy_a = _side_transport_quality(
        assignment,
        unmatched_a,
        mask_a,
        edges,
        top_k=top_k,
        epsilon=float(epsilon),
    )
    mass_b, entropy_b = _side_transport_quality(
        assignment.transpose(1, 2),
        unmatched_b,
        mask_b,
        edges.transpose(1, 2),
        top_k=top_k,
        epsilon=float(epsilon),
    )
    mass = 0.5 * (mass_a + mass_b)
    entropy = 0.5 * (entropy_a + entropy_b)
    finite = torch.isfinite(assignment).flatten(1).all(dim=1)
    finite = finite & torch.isfinite(unmatched_a).all(dim=1)
    finite = finite & torch.isfinite(unmatched_b).all(dim=1)
    mass = torch.where(finite, mass, torch.zeros_like(mass))
    entropy = torch.where(finite, entropy, torch.zeros_like(entropy))
    return ArcTransportQuality(mass_only=mass, entropy_aware=entropy)


def _group_log_mean_exp(
    value: Tensor,
    valid: Tensor,
    group_index: Tensor,
    *,
    group_count: int,
    temperature: float,
) -> tuple:
    selected = valid & torch.isfinite(value)
    selected_index = group_index[selected]
    selected_value = value[selected]
    counts = torch.zeros(group_count, dtype=torch.long, device=value.device)
    counts.scatter_add_(0, selected_index, torch.ones_like(selected_index))
    group_valid = counts > 0
    maximum = torch.full(
        (group_count,), -torch.inf, dtype=value.dtype, device=value.device
    )
    if selected_value.numel():
        maximum.scatter_reduce_(
            0, selected_index, selected_value, reduce="amax", include_self=True
        )
    exponent = torch.exp(
        (selected_value - maximum.index_select(0, selected_index)) / temperature
    )
    sum_exponent = torch.zeros(group_count, dtype=value.dtype, device=value.device)
    sum_exponent.scatter_add_(0, selected_index, exponent)
    output = maximum + float(temperature) * (
        torch.log(sum_exponent.clamp_min(torch.finfo(value.dtype).tiny))
        - torch.log(counts.clamp_min(1).to(value.dtype))
    )
    output = torch.where(group_valid, output, torch.zeros_like(output))
    return output, group_valid


def aggregate_frozen_readout(
    dataset: FrozenArcDataset,
    *,
    readout_name: str,
    beta: float | Tensor = 0.0,
    intercept: float | Tensor = 0.0,
    arc_pooling: Optional[ArcPoolingConfig] = None,
    direction_temperature: float = 0.25,
) -> FrozenReadoutOutput:
    """Apply one scalar readout under the unchanged hierarchical aggregation."""

    if not isinstance(dataset, FrozenArcDataset):
        raise TypeError("dataset must be FrozenArcDataset")
    pooling = arc_pooling or ArcPoolingConfig()
    if pooling.mode is not ArcPoolingMode.LOG_MEAN_EXP:
        raise ValueError("frozen scalar readout currently requires log-mean-exp arcs")
    if not math.isfinite(float(direction_temperature)) or direction_temperature <= 0.0:
        raise ValueError("direction_temperature must be finite and positive")
    feature = dataset.feature(readout_name)
    beta_tensor = torch.as_tensor(
        beta, dtype=dataset.arc_logit.dtype, device=dataset.arc_logit.device
    )
    intercept_tensor = torch.as_tensor(
        intercept, dtype=dataset.arc_logit.dtype, device=dataset.arc_logit.device
    )
    if beta_tensor.numel() != 1 or intercept_tensor.numel() != 1:
        raise ValueError("beta and intercept must be scalar")
    adjusted = dataset.arc_logit + beta_tensor.reshape(()) * feature
    adjusted = adjusted + intercept_tensor.reshape(())
    group_index = dataset.sample_index * 4 + dataset.direction_index
    direction_flat, direction_valid_flat = _group_log_mean_exp(
        adjusted,
        dataset.arc_valid,
        group_index,
        group_count=dataset.pair_count * 4,
        temperature=pooling.temperature,
    )
    direction_logits = direction_flat.reshape(dataset.pair_count, 4)
    direction_valid = direction_valid_flat.reshape(dataset.pair_count, 4)
    pair_valid = direction_valid.any(dim=1) & dataset.geometry_valid
    safe = torch.where(
        direction_valid,
        direction_logits,
        torch.full_like(direction_logits, -torch.inf),
    )
    count = direction_valid.sum(dim=1).clamp_min(1).to(direction_logits.dtype)
    pair_logit = float(direction_temperature) * (
        torch.logsumexp(safe / float(direction_temperature), dim=1)
        - torch.log(count)
    )
    pair_logit = torch.where(pair_valid, pair_logit, torch.zeros_like(pair_logit))
    best = safe.argmax(dim=1)
    best = torch.where(pair_valid, best, torch.full_like(best, -1))
    return FrozenReadoutOutput(
        pair_logit=pair_logit,
        pair_probability=torch.sigmoid(pair_logit),
        pair_valid=pair_valid,
        direction_logits=direction_logits,
        direction_valid=direction_valid,
        best_direction_index=best,
    )


def fit_frozen_readout(
    dataset: FrozenArcDataset,
    *,
    readout_name: str,
    arc_pooling: Optional[ArcPoolingConfig] = None,
    direction_temperature: float = 0.25,
    max_iterations: int = 64,
) -> FrozenReadoutParameters:
    """Fit only beta/intercept by deterministic full-batch synthetic BCE."""

    if dataset.label is None:
        raise ValueError("fitting requires synthetic labels")
    if readout_name == READOUT_BASELINE:
        output = aggregate_frozen_readout(
            dataset,
            readout_name=readout_name,
            arc_pooling=arc_pooling,
            direction_temperature=direction_temperature,
        )
        target = dataset.label.to(output.pair_logit.dtype)
        loss = F.binary_cross_entropy_with_logits(
            output.pair_logit[output.pair_valid], target[output.pair_valid]
        )
        return FrozenReadoutParameters(
            name=readout_name,
            beta=0.0,
            intercept=0.0,
            train_loss=float(loss.detach().cpu()),
            optimizer_steps=0,
        )
    if readout_name not in {READOUT_MASS_ONLY, READOUT_ENTROPY_AWARE}:
        raise ValueError("unsupported fitted readout")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    work = dataset.to(torch.device("cpu"), dtype=torch.float64)
    beta = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    intercept = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        (beta, intercept),
        lr=1.0,
        max_iter=max_iterations,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    calls = 0

    def closure() -> Tensor:
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        output = aggregate_frozen_readout(
            work,
            readout_name=readout_name,
            beta=beta,
            intercept=intercept,
            arc_pooling=arc_pooling,
            direction_temperature=direction_temperature,
        )
        valid = output.pair_valid
        if not valid.any().item():
            raise ValueError("frozen population has no valid pair")
        loss = F.binary_cross_entropy_with_logits(
            output.pair_logit[valid], work.label[valid].to(torch.float64)
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        output = aggregate_frozen_readout(
            work,
            readout_name=readout_name,
            beta=beta,
            intercept=intercept,
            arc_pooling=arc_pooling,
            direction_temperature=direction_temperature,
        )
        valid = output.pair_valid
        final_loss = F.binary_cross_entropy_with_logits(
            output.pair_logit[valid], work.label[valid].to(torch.float64)
        )
    return FrozenReadoutParameters(
        name=readout_name,
        beta=float(beta.detach()),
        intercept=float(intercept.detach()),
        train_loss=float(final_loss),
        optimizer_steps=calls,
    )


__all__ = [
    "ArcTransportQuality",
    "FrozenArcDataset",
    "FrozenReadoutOutput",
    "FrozenReadoutParameters",
    "READOUT_BASELINE",
    "READOUT_ENTROPY_AWARE",
    "READOUT_MASS_ONLY",
    "READOUT_NAMES",
    "aggregate_frozen_readout",
    "fit_frozen_readout",
    "transport_quality_features",
]
