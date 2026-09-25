"""One-Sinkhorn adapter exposing unchanged S7 Matcher evidence.

No legacy classifier or decoder is executed. The full base is retained for
strict checkpoint loading. Legacy Context storage-padding sensitivity is NOT
silently repaired: parity and new-module padding tests are separate gates.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from staging.pairwise_v0_2.models.rachel_n512 import (
    RachelN512Pairwise, _validate_binary_masks, _validate_contours,
)
from staging.pairwise_v0_2.models.optimal_transport import (
    PartialTransportOutput, dustbin_sinkhorn,
)
from .geometry import ContourGeometry, compact_contour


SOURCE_SHA = 'd8a93af1eb5f3b02baaf7d42b9d8675242a446a1b11e43cde0561ba89e670e07'
SOURCE_TRAIN_SHA = '79a9e959f32ef9899116e299425d6350b17f6a04bd5070b5aa9a730319447c36'
INPUTS = ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b', 'contour_valid_a', 'contour_valid_b')


@dataclass(frozen=True)
class MatcherEvidence:
    local_a: Tensor
    local_b: Tensor
    context_a: Tensor
    context_b: Tensor
    affinity: Tensor
    assignment: Tensor
    unmatched_a: Tensor
    unmatched_b: Tensor
    points_rc_a: Tensor
    points_rc_b: Tensor
    valid_a: Tensor
    valid_b: Tensor
    geometry_a: ContourGeometry
    geometry_b: ContourGeometry
    transport: PartialTransportOutput
    numeric_valid: Tensor


class S7MatcherAdapter(nn.Module):
    def __init__(self, base: RachelN512Pairwise, *, frozen: bool = True):
        super().__init__()
        self.base = base
        self.frozen = frozen
        self.set_frozen(frozen)

    def set_frozen(self, frozen: bool):
        self.frozen = bool(frozen)
        self.base.requires_grad_(not self.frozen)
        # These retained checkpoint tensors are not part of the new pipeline.
        for module in (self.base.coarse, self.base.local_head, self.base.fusion):
            module.requires_grad_(False)
        self.train(self.training)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(mode and not self.frozen)
        for module in (self.base.coarse, self.base.local_head, self.base.fusion):
            module.eval()
        return self

    @classmethod
    def from_s7_m12(cls, path):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
            raise ValueError('not the bound historical S7 M12 checkpoint')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if (checkpoint.get('epoch') != 12 or checkpoint.get('completed_segments') != 48
                or checkpoint.get('phase') != 'matcher'
                or checkpoint['resume_identity']['populations']['train']['manifest_sha256'] != SOURCE_TRAIN_SHA):
            raise ValueError('S7 source training identity differs')
        # This loader validates the original complete M12 receipt and loads
        # state_dict strictly. It does not initialize the new consensus head.
        from experiments.rachel_n512_formal_30k.train_score_decoupled import load_decoupled_checkpoint
        base = load_decoupled_checkpoint(checkpoint).base_model
        if (base.config.contour_cap != 512 or base.config.feature_dim != 96
                or tuple(base.config.window_sizes_px) != (7., 16., 32., 64.)):
            raise ValueError('unexpected S7 Matcher architecture')
        return cls(base, frozen=True)

    def forward(self, mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            return self._forward(mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b)

    def _forward(self, mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b):
        base = self.base
        cfg = base.config
        mask_a, mask_b = _validate_binary_masks(mask_a, mask_b, cfg.canvas_size,
                                               validate_values=cfg.validate_runtime_inputs)
        points = []
        for name, p, valid in (('A', points_rc_a, contour_valid_a), ('B', points_rc_b, contour_valid_b)):
            _validate_contours(p, valid, batch_size=len(mask_a), canvas_size=cfg.canvas_size,
                               contour_cap=cfg.contour_cap, name=name, validate_values=cfg.validate_runtime_inputs)
            if p.device != mask_a.device or p.dtype != mask_a.dtype:
                raise ValueError('masks and points must share device/dtype')
            # Padding is not content. Loader's standard zero padding remains
            # unchanged; arbitrary invalid values cannot enter the old Context.
            points.append(torch.where(valid[:, :, None], p, torch.zeros_like(p)))
        pa, pb = points
        fa = base._encode_patches(base.patch_sampler(mask_a, pa, contour_valid_a), contour_valid_a)
        fb = base._encode_patches(base.patch_sampler(mask_b, pb, contour_valid_b), contour_valid_b)
        ha, hb = base.context(fa, fb, contour_valid_a, contour_valid_b, pa, pb, cfg.canvas_size)
        primal_a = F.normalize(base.primal(ha), dim=2, eps=1e-6)
        primal_b = F.normalize(base.primal(hb), dim=2, eps=1e-6)
        dual_a = F.normalize(base.dual(ha), dim=2, eps=1e-6)
        dual_b = F.normalize(base.dual(hb), dim=2, eps=1e-6)
        affinity = .5 * (primal_a @ dual_b.transpose(1, 2) + dual_a @ primal_b.transpose(1, 2))
        # Exactly one partial Sinkhorn. Keep tiny probability mass in FP32;
        # formal training/inference disables autocast around the whole adapter.
        with torch.autocast(device_type=affinity.device.type, enabled=False):
            transport = dustbin_sinkhorn(affinity.float(), contour_valid_a, contour_valid_b,
                dustbin_score=base.dustbin_score.float(), temperature=cfg.matcher_temperature,
                num_iterations=cfg.sinkhorn_iterations, tolerance=cfg.sinkhorn_tolerance,
                checkpoint_iterations=(cfg.activation_checkpointing and base.training and torch.is_grad_enabled()))
        diag = transport.diagnostics
        return MatcherEvidence(fa, fb, ha, hb, affinity, transport.real_transport,
            transport.dustbin_col, transport.dustbin_row, pa, pb, contour_valid_a, contour_valid_b,
            compact_contour(pa, contour_valid_a), compact_contour(pb, contour_valid_b), transport,
            diag.valid_problem & diag.finite_output)
