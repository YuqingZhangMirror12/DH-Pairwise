"""Residual feature/logit upgrades before ONE unchanged partial Sinkhorn.

The user-supplied 2026-09-30 design is the specification, not a measured
performance claim. Explicit matmul/softmax avoids an implicit SDPA backend
change. Only the new modules are valid-token cyclic/padding invariant: the
retained historical convolution/landmark context is not silently repaired.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class MatcherV2Config:
    enabled: bool = False
    use_long_context: bool = True
    use_cross: bool = True
    use_multiscale: bool = True
    use_sharpness: bool = True
    use_matchability: bool = False
    layers: int = 2
    heads: int = 4
    ff_mult: int = 2
    max_harmonic: int = 256
    scale_head_dim: int = 48
    max_sharpness: float = 5.0

    def __post_init__(self):
        for name in ('enabled', 'use_long_context', 'use_cross',
                     'use_multiscale', 'use_sharpness', 'use_matchability'):
            if type(getattr(self, name)) is not bool:
                raise TypeError(name + ' must be bool')
        for name in ('layers', 'heads', 'ff_mult', 'max_harmonic', 'scale_head_dim'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if not math.isfinite(self.max_sharpness) or self.max_sharpness < 1:
            raise ValueError('max_sharpness must be finite and >= 1')
        if (self.enabled and self.use_sharpness
                and not (self.use_long_context or self.use_cross)):
            raise ValueError('standalone sharpness is not an authorized v2 variant')


def masked_tokens(x, valid):
    return torch.where(valid[..., None], x, torch.zeros_like(x))


def deterministic_prefix_sum(values):
    """Inclusive parallel scan without CUDA's nondeterministic cumsum kernel.

    Fixed pairwise additions in float64 keep512 arc increments accurate. Runs
    on the input device, uses O(B*N) storage and leaves deterministic mode on.
    Casting back once preserves the feature/coordinate dtype.
    """
    result = values.to(torch.float64)
    shift = 1
    while shift < values.shape[1]:
        result = torch.cat((result[:, :shift], result[:, shift:] + result[:, :-shift]), dim=1)
        shift *= 2
    return result.to(values.dtype)


def arc_positions(points, valid):
    """Arc coordinates in valid storage order, closing at n_valid, not N.

    Supports holes as well as prefix padding. Invalid/NaN storage is excluded
    before any arithmetic. Unique integer sorting keys make compaction
    deterministic. A wholly empty contour has arc=0, perimeter=1.
    """
    if points.ndim != 3 or points.shape[-1] != 2 or valid.shape != points.shape[:2]:
        raise ValueError('expected points [B,N,2] and valid [B,N]')
    if valid.dtype != torch.bool or valid.device != points.device:
        raise ValueError('valid must be a same-device boolean mask')
    if points.shape[1] < 1:
        raise ValueError('empty storage axis is not supported')
    n = points.shape[1]
    index = torch.arange(n, device=points.device)[None].expand_as(valid)
    order = torch.argsort(torch.where(valid, index, index + n), dim=1)
    safe = masked_tokens(points, valid)
    compact = safe.gather(1, order[..., None].expand(-1, -1, 2))
    counts = valid.sum(1)
    active = index < counts[:, None]
    nxt = torch.where(index + 1 < counts[:, None], index + 1, 0)
    following = compact.gather(1, nxt[..., None].expand(-1, -1, 2))
    steps = (following - compact).norm(dim=-1) * active
    perimeter = steps.sum(1)
    perimeter = torch.where(perimeter > 0, perimeter, torch.ones_like(perimeter))
    compact_arc = (deterministic_prefix_sum(steps) - steps) * active
    arc = torch.zeros_like(compact_arc).scatter(1, order, compact_arc)
    return arc, perimeter


def integer_harmonics(count, maximum):
    if maximum < count:
        raise ValueError('max_harmonic must allow one distinct integer per rotary pair')
    proposed = [int(round(v)) for v in
                torch.logspace(0, math.log10(maximum), count, dtype=torch.float64).tolist()]
    selected = sorted(set(proposed))
    for value in range(1, maximum + 1):
        if len(selected) == count:
            break
        if value not in selected:
            selected.append(value)
    return torch.tensor(sorted(selected), dtype=torch.float32)


def rotary(x, angles):
    even, odd = x[..., 0::2], x[..., 1::2]
    c, s = angles.cos(), angles.sin()
    return torch.stack((even * c - odd * s, even * s + odd * c), -1).flatten(-2)


def attention(q, k, v, key_valid):
    """Masked deterministic math attention, finite even with no valid keys."""
    logits = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    mask = key_valid[:, None, None, :]
    weights = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), -1)
    weights = weights * mask
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    return weights @ v


def zero_output(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class ContourSelfBlock(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, dim // cfg.heads
        if dim % cfg.heads or self.head_dim % 2:
            raise ValueError('feature_dim / heads must be an even integer')
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, cfg.ff_mult * dim), nn.GELU(),
                                nn.Linear(cfg.ff_mult * dim, dim))
        self.register_buffer('harmonics', integer_harmonics(self.head_dim // 2, cfg.max_harmonic))
        zero_output(self.out)
        zero_output(self.ff[-1])

    def forward(self, x, valid, arc, perimeter):
        x = masked_tokens(x, valid)
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(
            batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        angles = (2 * math.pi * arc / perimeter[:, None])[:, None, :, None]
        angles = angles * self.harmonics.to(angles.dtype)
        update = attention(rotary(q, angles), rotary(k, angles), v, valid)
        x = x + self.out(update.transpose(1, 2).reshape(batch, tokens, dim))
        x = x + self.ff(self.norm2(x))
        return masked_tokens(x, valid)


class CrossBlock(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        if dim % cfg.heads:
            raise ValueError('feature_dim must be divisible by heads')
        self.heads, self.head_dim = cfg.heads, dim // cfg.heads
        self.norm_q, self.norm_kv, self.norm2 = (nn.LayerNorm(dim) for _ in range(3))
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, cfg.ff_mult * dim), nn.GELU(),
                                nn.Linear(cfg.ff_mult * dim, dim))
        zero_output(self.out)
        zero_output(self.ff[-1])

    def forward(self, x, x_valid, y, y_valid):
        x, y = masked_tokens(x, x_valid), masked_tokens(y, y_valid)
        batch, tokens, dim = x.shape
        q = self.q(self.norm_q(x)).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(self.norm_kv(y)).reshape(
            batch, y.shape[1], 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        update = attention(q, k, v, y_valid).transpose(1, 2).reshape(batch, tokens, dim)
        # Suppress a learned output bias when the other contour has no keys.
        x = x + self.out(update) * y_valid.any(1)[:, None, None]
        x = x + self.ff(self.norm2(x))
        return masked_tokens(x, x_valid)


class MatcherV2Features(nn.Module):
    def __init__(self, dim, scales, cfg):
        super().__init__()
        if not cfg.enabled:
            raise ValueError('disabled v2 must not allocate any parameters')
        self.config = cfg
        self.self_blocks = nn.ModuleList([
            ContourSelfBlock(dim, cfg) for _ in range(cfg.layers)] if cfg.use_long_context else [])
        self.cross_blocks = nn.ModuleList([
            CrossBlock(dim, cfg) for _ in range(cfg.layers)] if cfg.use_cross else [])
        if cfg.use_multiscale:
            self.concat = nn.Linear(scales * dim, dim)
            zero_output(self.concat)
            self.scale_primal = nn.ModuleList([
                nn.Linear(2 * dim, cfg.scale_head_dim, bias=False) for _ in range(scales)])
            self.scale_dual = nn.ModuleList([
                nn.Linear(2 * dim, cfg.scale_head_dim, bias=False) for _ in range(scales)])
            self.scale_weight = nn.Parameter(torch.zeros(scales))
        if cfg.use_sharpness:
            self.log_sharpness = nn.Parameter(torch.zeros(()))
        if cfg.use_matchability:
            self.matchability = nn.Linear(dim, 1)
            zero_output(self.matchability)

    def token_correction(self, fused, encoded, valid):
        if self.config.use_multiscale:
            fused = fused + self.concat(encoded.flatten(2))
        return masked_tokens(fused, valid)

    def contexts(self, ha, hb, va, vb, pa, pb, checkpointing=False):
        aa, per_a = arc_positions(pa, va)
        ab, per_b = arc_positions(pb, vb)
        use_checkpoint = checkpointing and self.training and torch.is_grad_enabled()

        def run(module, *args):
            if use_checkpoint:
                return checkpoint(module, *args, use_reentrant=False, preserve_rng_state=False)
            return module(*args)

        for index in range(self.config.layers):
            if self.self_blocks:
                ha = run(self.self_blocks[index], ha, va, aa, per_a)
                hb = run(self.self_blocks[index], hb, vb, ab, per_b)
            if self.cross_blocks:
                # Both directions use the same PRE-update pair, not A_new -> B.
                old_a, old_b = ha, hb
                ha = run(self.cross_blocks[index], old_a, va, old_b, vb)
                hb = run(self.cross_blocks[index], old_b, vb, old_a, va)
        return ha, hb

    def affinity(self, cosine, ha, hb, ea, eb, temperature):
        affinity = cosine
        if self.config.use_sharpness:
            gain = self.log_sharpness.clamp(max=math.log(self.config.max_sharpness)).exp()
            affinity = affinity * gain
        if self.config.use_multiscale:
            for index, (primal, dual) in enumerate(zip(self.scale_primal, self.scale_dual)):
                xa = torch.cat((ha, ea[:, :, index]), -1)
                xb = torch.cat((hb, eb[:, :, index]), -1)
                ap, bp = F.normalize(primal(xa), dim=-1, eps=1e-6), F.normalize(primal(xb), dim=-1, eps=1e-6)
                ad, bd = F.normalize(dual(xa), dim=-1, eps=1e-6), F.normalize(dual(xb), dim=-1, eps=1e-6)
                cosine_scale = .5 * (ap @ bd.transpose(1, 2) + ad @ bp.transpose(1, 2))
                affinity = affinity + self.scale_weight[index] * cosine_scale
        if self.config.use_matchability:
            # Subtract the initialization constant: unlike the reference
            # wrapper this branch is EXACTLY neutral at initialization.
            neutral = F.logsigmoid(ha.new_zeros(()))
            ma = F.logsigmoid(self.matchability(ha)).squeeze(-1) - neutral
            mb = F.logsigmoid(self.matchability(hb)).squeeze(-1) - neutral
            affinity = affinity + temperature * (ma[:, :, None] + mb[:, None, :])
        return affinity
