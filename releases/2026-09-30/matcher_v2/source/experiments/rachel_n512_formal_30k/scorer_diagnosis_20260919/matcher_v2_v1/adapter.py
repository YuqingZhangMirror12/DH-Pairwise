"""S7-compatible v2 adapter; legacy base, evidence and Sinkhorn unchanged."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Pairwise
from ..s7_consensus_v1 import matcher as legacy
from .network import MatcherV2Config, MatcherV2Features


class MatcherV2Adapter(legacy.S7MatcherAdapter):
    def __init__(self, base, *, config=None, frozen=True):
        super().__init__(base, frozen=frozen)
        self.v2_config = config or MatcherV2Config()
        self.upgrades = None
        if self.v2_config.enabled:
            # Preserve downstream head initialization / data RNG sequences.
            with torch.random.fork_rng(devices=[]):
                self.upgrades = MatcherV2Features(
                    base.config.feature_dim, len(base.config.window_sizes_px), self.v2_config)
        self.set_frozen(frozen)

    def set_frozen(self, frozen):
        super().set_frozen(frozen)
        if getattr(self, 'upgrades', None) is not None:
            self.upgrades.requires_grad_(not self.frozen)
            self.upgrades.train(self.training and not self.frozen)
        return self

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'upgrades', None) is not None:
            self.upgrades.train(mode and not self.frozen)
        return self

    def _tokens(self, mask, points, valid):
        base = self.base
        patches = base.patch_sampler(mask, points, valid)
        batch, tokens, scales = patches.shape[:3]
        flat = patches.reshape(batch, tokens * scales, *patches.shape[3:])
        flat_valid = valid[:, :, None].expand(-1, -1, scales).reshape(batch, -1)
        if base.config.activation_checkpointing and base.training and torch.is_grad_enabled():
            encoded = checkpoint(base.patch_encoder, flat, flat_valid,
                                 use_reentrant=False, preserve_rng_state=False)
        else:
            encoded = base.patch_encoder(flat, flat_valid)
        encoded = encoded.reshape(batch, tokens, scales, -1)
        gates = torch.softmax(base.scale_gate(encoded).squeeze(3), dim=2)
        fused = (encoded * gates[:, :, :, None]).sum(dim=2)
        return self.upgrades.token_correction(fused, encoded, valid), encoded

    def _forward(self, mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b):
        if self.upgrades is None:
            return super()._forward(mask_a, mask_b, points_rc_a, points_rc_b,
                                    contour_valid_a, contour_valid_b)
        base, cfg = self.base, self.base.config
        mask_a, mask_b = legacy._validate_binary_masks(
            mask_a, mask_b, cfg.canvas_size, validate_values=cfg.validate_runtime_inputs)
        points = []
        for name, p, valid in (('A', points_rc_a, contour_valid_a), ('B', points_rc_b, contour_valid_b)):
            legacy._validate_contours(p, valid, batch_size=len(mask_a), canvas_size=cfg.canvas_size,
                                      contour_cap=cfg.contour_cap, name=name,
                                      validate_values=cfg.validate_runtime_inputs)
            if p.device != mask_a.device or p.dtype != mask_a.dtype:
                raise ValueError('masks and points must share device/dtype')
            points.append(torch.where(valid[:, :, None], p, torch.zeros_like(p)))
        pa, pb = points
        va, vb = contour_valid_a, contour_valid_b
        fa, ea = self._tokens(mask_a, pa, va)
        fb, eb = self._tokens(mask_b, pb, vb)
        ha, hb = base.context(fa, fb, va, vb, pa, pb, cfg.canvas_size)
        ha, hb = self.upgrades.contexts(ha, hb, va, vb, pa, pb, cfg.activation_checkpointing)
        ap, bp = F.normalize(base.primal(ha), dim=2, eps=1e-6), F.normalize(base.primal(hb), dim=2, eps=1e-6)
        ad, bd = F.normalize(base.dual(ha), dim=2, eps=1e-6), F.normalize(base.dual(hb), dim=2, eps=1e-6)
        cosine = .5 * (ap @ bd.transpose(1, 2) + ad @ bp.transpose(1, 2))
        affinity = self.upgrades.affinity(cosine, ha, hb, ea, eb, cfg.matcher_temperature)
        with torch.autocast(device_type=affinity.device.type, enabled=False):
            transport = legacy.dustbin_sinkhorn(
                affinity.float(), va, vb, dustbin_score=base.dustbin_score.float(),
                temperature=cfg.matcher_temperature, num_iterations=cfg.sinkhorn_iterations,
                tolerance=cfg.sinkhorn_tolerance,
                checkpoint_iterations=(cfg.activation_checkpointing and base.training and torch.is_grad_enabled()))
        diag = transport.diagnostics
        return legacy.MatcherEvidence(fa, fb, ha, hb, affinity, transport.real_transport,
            transport.dustbin_col, transport.dustbin_row, pa, pb, va, vb,
            legacy.compact_contour(pa, va), legacy.compact_contour(pb, vb), transport,
            diag.valid_problem & diag.finite_output)


def fresh_matcher_v2(reference_config, *, config=None, seed=26092407):
    """Same seeded legacy initialization; no checkpoint/old-head import."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = RachelN512Pairwise(reference_config)
        return MatcherV2Adapter(base, config=config, frozen=False)
