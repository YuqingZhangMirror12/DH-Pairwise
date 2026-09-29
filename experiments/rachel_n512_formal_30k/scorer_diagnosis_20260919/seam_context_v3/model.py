from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config, ContourPatchSampler
from staging.pairwise_v0_2.models.local_matcher import SharedPatchEncoder
from staging.pairwise_v0_2.models.optimal_transport import dustbin_sinkhorn
from .config import Config
from .valid_contour import compact
from .arc_context import ArcContext
from .primal_dual import PrimalDual
from .correspondence_context import CorrespondenceContext
from .seam_proposals import propose
from .seam_verifier import SeamVerifier


class SeamContextModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        c = self.cfg
        self.sampler = ContourPatchSampler(RachelN512Config(canvas_size=c.canvas_size,
            contour_cap=c.contour_cap, window_sizes_px=c.windows, patch_size=c.patch_size))
        self.patch_encoder = SharedPatchEncoder(1, c.dim)
        self.scale_gate = nn.Linear(c.dim, 1)
        self.arc_context = ArcContext(c)
        self.primal_dual = PrimalDual(c.dim)
        self.dustbin_logit = nn.Parameter(torch.tensor(0.))
        self.correspondence_context = CorrespondenceContext(c)
        self.verifier = SeamVerifier(c)

    def encode(self, mask, g):
        patches = self.sampler(mask, g.points, g.valid)
        b, n, s = patches.shape[:3]
        # Only real patches enter the encoder; patch-internal GroupNorm is safe.
        valid = g.valid[:, :, None].expand(b, n, s).reshape(-1)
        flat = patches.reshape(-1, 1, self.cfg.patch_size, self.cfg.patch_size)
        real = flat[valid][None]
        if real.shape[1]:
            def encode_real(x):
                return self.patch_encoder(x, torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            features = checkpoint(encode_real, real, use_reentrant=False) if (
                self.training and self.cfg.activation_checkpointing) else encode_real(real)
            encoded = features.new_zeros(b*n*s, self.cfg.dim).index_copy(0,
                torch.nonzero(valid).flatten(), features.squeeze(0))
        else:
            encoded = mask.new_zeros(b*n*s, self.cfg.dim)
        encoded = encoded.reshape(b, n, s, -1)
        gate = self.scale_gate(encoded).softmax(2)
        return (encoded*gate).sum(2)

    def transport(self, affinity, ga, gb):
        # Disable autocast explicitly: both iterations/marginals stay FP32.
        with torch.autocast(device_type=affinity.device.type, enabled=False):
            return dustbin_sinkhorn(affinity.float(), ga.valid, gb.valid,
                dustbin_score=self.dustbin_logit.float(), temperature=self.cfg.temperature,
                num_iterations=self.cfg.sinkhorn_iterations, tolerance=self.cfg.tolerance,
                checkpoint_iterations=self.training and self.cfg.activation_checkpointing)

    def forward(self, mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b,
                *, decode=False, verify=False):
        ga, gb = compact(points_rc_a, contour_valid_a), compact(points_rc_b, contour_valid_b)
        fa, fb = self.encode(mask_a, ga), self.encode(mask_b, gb)
        ha, hb = self.arc_context(fa, fb, ga, gb)
        s0, roles = self.primal_dual(ha, hb)
        ot0 = self.transport(s0, ga, gb)
        s1, records = self.correspondence_context(fa, fb, ha, hb, roles, s0, ot0, ga, gb)
        ot1 = self.transport(s1, ga, gb)
        result = SimpleNamespace(ga=ga, gb=gb, fa=fa, fb=fb, ha=ha, hb=hb, roles=roles,
            s0=s0, s1=s1, ot0=ot0, ot1=ot1, records=records,
            masks=(mask_a, mask_b), candidates=[], verified=[])
        if decode or verify:
            for b, record in enumerate(records):
                candidates = propose(record, ot1.real_transport[b], ga, gb, b, self.cfg)
                result.candidates.append(candidates)
                if verify:
                    result.verified.append(self.verify_candidates(result, b, candidates))
        return result

    def verify_candidates(self, output, b, candidates, record=None):
        return self.verifier(candidates, output.records[b] if record is None else record, b,
            output.fa, output.fb, output.ha, output.hb, output.ot1, output.ga, output.gb, *output.masks)
